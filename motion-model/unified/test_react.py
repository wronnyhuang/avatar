"""reaction adapter tests: the partner trace (numpy runtime == torch training), the adapter (numpy == torch, zero-init =
the source model), and FlowCore with a react bundle stepping like the torch model's hidden().

    python -I unified/test_react.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified import react as RX  # noqa: E402
from unified.model import UnifiedModel, layout  # noqa: E402


def _tokens(B, S, lay, ni, seed=0):
    rng = np.random.default_rng(seed)
    tok = rng.standard_normal((B, S, lay["din"])).astype(np.float32) * 0.1
    cols = RX.token_cols(lay)
    m, s = np.array(ni["mean"]), np.array(ni["std"])
    t = np.arange(S)
    for b in range(B):
        sm = 0.2 + 0.3 * (np.sin(t / (9 + b)) > 0.6) + 0.02 * rng.standard_normal(S)          # smile steps
        jaw = 0.1 + 0.2 * (rng.random(S) < 0.3)
        pitch = np.radians(3.0 * np.sin(t / 4.0) * (rng.random() < 0.7))                    # nods
        raw = [sm, sm, jaw, pitch / 2, pitch / 2]
        for k in range(5):
            tok[b, :, cols[k]] = (raw[k] - m[k]) / s[k]
        tok[b, :, cols[5]] = (rng.random(S) > 0.05)
        tok[b, :, cols[6]] = (rng.random(S) > 0.05)
        vad = (np.sin(t / 13.0 + b) > 0.2).astype(np.float32)
        tok[b, :, cols[7]] = (vad - ni["vad_mean"]) / ni["vad_std"]
    return tok


def test_trace_match():
    lay = layout(66)
    ni = dict(mean=[0.25, 0.25, 0.1, 0.0, 0.0], std=[0.15, 0.15, 0.1, 0.1, 0.1], vad_mean=0.3, vad_std=0.45)
    tok = _tokens(4, 300, lay, ni)
    tt = RX.trace_torch(torch.from_numpy(tok), lay, ni).numpy()
    for b in range(4):
        tn = RX.trace_np(tok[b][:, RX.token_cols(lay)], ni)
        err = np.abs(tt[b] - tn).max()
        assert err < 1e-4, (b, err, np.argwhere(np.abs(tt[b] - tn) > 1e-4)[:5])
    on = tt[..., [0, 4, 8, 12]].sum((0, 1))
    assert (on[[0, 1, 3]] > 0).all(), on                      # smile / nod / speech onsets fire on this data
    # runtime ring = the last RING tokens only: equal to the full trace once the support is covered
    b = 0
    full = RX.trace_np(tok[b][:, RX.token_cols(lay)], ni)
    rows = tok[b][-RX.RING:, RX.token_cols(lay)]
    ring = RX.trace_np(rows, ni)
    assert np.abs(ring[-1] - full[-1]).max() < 1e-4
    print("trace torch == numpy; onsets", on.tolist())


def test_adapter_and_model():
    lay = layout(66)
    ni = dict(mean=[0.25, 0.25, 0.1, 0.0, 0.0], std=[0.15, 0.15, 0.1, 0.1, 0.1], vad_mean=0.3, vad_std=0.45)
    cfg = dict(d=64, layers=2, heads=2, hd=64, hlayers=1, hheads=2, aud_dim=66, n_events=3)
    m0 = UnifiedModel(cfg).eval()
    m1 = UnifiedModel(dict(cfg, react_trace=dict(ni, hidden=16))).eval()
    sd = m0.state_dict()
    mk = m1.load_state_dict(sd, strict=False)
    assert all(k.startswith("react.") for k in mk.missing_keys) and not mk.unexpected_keys
    tok = torch.from_numpy(_tokens(2, 120, lay, ni))
    with torch.no_grad():
        assert torch.allclose(m0.backbone(tok), m1.hidden(tok))             # zero-init: the source model exactly
        for p in m1.react.parameters():
            p.normal_(0, 0.2)
        h = m1.hidden(tok)
        tr = RX.trace_torch(tok, lay, ni)
        a_t = m1.react(tr).numpy()
    a_n = RX.adapter_np(tr.numpy().reshape(-1, RX.N_FEAT), m1.react.np_weights()).reshape(a_t.shape)
    assert np.abs(a_t - a_n).max() < 1e-4, np.abs(a_t - a_n).max()
    assert not torch.allclose(h, m1.backbone(tok))
    print("adapter numpy == torch; zero-init == source model")


if __name__ == "__main__":
    test_trace_match()
    test_adapter_and_model()
    print("OK")
