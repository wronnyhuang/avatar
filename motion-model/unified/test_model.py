"""Model invariants: the fixed-cache incremental backbone equals the full sliding-window forward."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified.model import DIN, UnifiedModel, layout, null_tokens, rope_cos_sin  # noqa: E402


import pytest


@pytest.mark.parametrize("QK,A", [(False, 42), (True, 42), (True, 66)])
def test_incremental_equals_full(QK, A):
    torch.manual_seed(0)
    W = 20
    m = UnifiedModel(dict(d=64, layers=3, heads=4, window=W, hd=32, hlayers=1, mem=8, qknorm=QK, aud_dim=A,
                          n_events=3)).eval()
    T = 57
    tok = torch.randn(2, T, layout(A)["din"])
    dh = 16
    with torch.no_grad():
        full = m.backbone(tok, pos0=-1)
        kc = torch.zeros(3, 2, 4, W - 1, dh); vc = torch.zeros_like(kc); kvalid = torch.zeros(W - 1)
        outs, i = [], 0
        for n in [1, 3, 3, 2, 5, 3, 3, 3, 1, 3, 30]:
            n = min(n, T - i)
            if n <= 0:
                break
            cos, sin = rope_cos_sin(torch.arange(i - 1, i - 1 + n), dh)
            o, kc, vc = m.backbone.step_fixed(tok[:, i:i + n], cos, sin, kc, vc, kvalid)
            kvalid = torch.cat([kvalid, torch.ones(n)])[-(W - 1):]
            outs.append(o); i += n
        inc = torch.cat(outs, 1)
    assert inc.shape == full.shape
    assert torch.allclose(inc, full, atol=1e-4), (inc - full).abs().max()


def test_layout_and_null():
    assert layout(42)["din"] == DIN == 487
    lo = layout(66)
    tok = torch.randn(1, 4, lo["din"])
    n = null_tokens(tok)
    assert (n[..., lo["user_all"]] == 0).all() and (n[..., lo["unull"]] == 1).all()
    assert torch.equal(n[..., lo["self"]], tok[..., lo["self"]]) and torch.equal(n[..., lo["style"]], tok[..., lo["style"]])
    m = UnifiedModel(dict(d=32, layers=1, heads=2, hd=16, hlayers=1, mem=4, aud_dim=66, n_events=3))
    p = torch.sigmoid(m.events(m.backbone(tok)))
    assert p.shape == (1, 4, 3) and float(p.mean()) == pytest.approx(0.01, rel=1e-3)   # prior-initialised bias
