"""Export a bundle's torch weights to ONNX (backbone_step / head_prep / head_step [/ event_step]) for the ORT runtime
backend.

    python -I -m unified.export <bundle_dir>      (from the repo root)   or   python -I unified/export.py <bundle_dir>

event_step.onnx is written for bundles with an event head (model n_events > 0): h (N,d) backbone outputs -> p (N,n)
onset probabilities. check() compares torch vs ORT over a short planner rollout: the motion and (event bundles) the
per-frame event_p; enc-audio bundles get random raw encoder features.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified.model import D_MASK, D_MOT, BBStep, EvStep, HeadPrep, HeadStep, UnifiedModel  # noqa: E402


def export(bundle_dir, opset=17, detail=False):
    """writes the graphs; returns check() (max |ORT - torch|), or check_detail()'s dict when detail=True."""
    with open(os.path.join(bundle_dir, "config.json")) as f:
        cfg = json.load(f)
    mc = cfg["model"]
    m = UnifiedModel(mc)
    m.load_state_dict(torch.load(os.path.join(bundle_dir, "weights.pt"), map_location="cpu", weights_only=True))
    m.eval()
    L, h, d, W, M, H = mc["layers"], mc["heads"], mc["d"], mc["window"], mc["mem"], mc["horizon"]
    dh = d // h
    hl, hh, hd = mc["hlayers"], mc["hheads"], mc["hd"]
    ex = dict(dynamo=False, opset_version=opset, do_constant_folding=True)
    S = int(cfg.get("steps", 6))
    T = H // mc.get("patch", 1)
    with torch.no_grad():
        args = (torch.randn(2, 3, m.layout["din"]), torch.randn(3, dh // 2), torch.randn(3, dh // 2),
                torch.randn(L, 2, h, W - 1, dh), torch.randn(L, 2, h, W - 1, dh), torch.ones(W - 1))
        torch.onnx.export(BBStep(m.backbone), args, os.path.join(bundle_dir, "backbone_step.onnx"),
                          input_names=["tok", "cos", "sin", "kc", "vc", "kvalid"], output_names=["out", "kc2", "vc2"],
                          dynamic_axes={"tok": {1: "n"}, "cos": {0: "n"}, "sin": {0: "n"}, "out": {1: "n"}}, **ex)
        args = (torch.randn(2, M, d), torch.rand(S), torch.randn(1, H, D_MOT), torch.zeros(1, H, D_MASK),
                torch.tensor([[0.5, 0.0, 0.2, 0.0, 1.0, 0.0]]))
        torch.onnx.export(HeadPrep(m.head), args, os.path.join(bundle_dir, "head_prep.onnx"),
                          input_names=["mem", "s_all", "kvals", "kmask", "style"],
                          output_names=["memk", "memv", "bmods", "fmods", "kemb"],
                          dynamic_axes={"s_all": {0: "S"}, "bmods": {0: "S"}, "fmods": {0: "S"}}, **ex)
        args = (torch.randn(1, H, D_MOT), torch.randn(1, T, hd), torch.randn(hl, 2, hh, M, hd // hh),
                torch.randn(hl, 2, hh, M, hd // hh), torch.randn(hl, 2, 9, hd), torch.randn(2, 2, hd),
                torch.zeros(2, 1, 1, M), torch.ones(1))
        torch.onnx.export(HeadStep(m.head), args, os.path.join(bundle_dir, "head_step.onnx"),
                          input_names=["x", "kemb", "memk", "memv", "bmods", "fmods", "mbias", "g"],
                          output_names=["v"], **ex)
        if m.events is not None:
            torch.onnx.export(EvStep(m.events), (torch.randn(2, d),), os.path.join(bundle_dir, "event_step.onnx"),
                              input_names=["h"], output_names=["p"], dynamic_axes={"h": {0: "N"}, "p": {0: "N"}},
                              **ex)
        if m.react is not None:                           # reaction adapter: the partner-trace adapter, numpy at runtime
            np.savez(os.path.join(bundle_dir, "react.npz"), **m.react.np_weights())
    return check_detail(bundle_dir) if detail else check(bundle_dir, m)


def check_detail(bundle_dir, frames=40):
    """{"motion": max |ORT - torch| of the motion, "event_p": same for event_p (None without an event head)} over a
    short random rollout of the planner (both backends, same seed; enc bundles get random raw features)."""
    from unified.runtime import UnifiedPlanner
    outs, evp = {}, {}
    for be in ("torch", "ort"):
        p = UnifiedPlanner(bundle_dir, threads=2, backend=be)
        r = np.random.default_rng(1)
        De = p.feat_dim
        o, e = [], []
        p.set_style({"energy": 0.7, "expressiveness": 0.2})
        for k in range(frames):
            kw = {}
            if De > 0:
                kw = dict(user_feat=r.standard_normal(De).astype(np.float32),
                          self_feat_ahead=r.standard_normal(De).astype(np.float32) if k % 7 < 4 else None)
            res = p.step(r.standard_normal(189).astype(np.float32) * 0.1, np.ones(9),
                         r.standard_normal(42).astype(np.float32) - 5, None, **kw)
            o.append(res["motion"])
            if res.get("event_p") is not None:
                e.append(res["event_p"])
        outs[be] = np.stack(o)
        evp[be] = np.stack(e) if e else None
    err = float(np.abs(outs["torch"] - outs["ort"]).max())
    eerr = None if evp["torch"] is None else float(np.abs(evp["torch"] - evp["ort"]).max())
    return dict(motion=err, event_p=eerr)


def check(bundle_dir, m=None):
    """max |ORT - torch| over the motion and (event bundles) event_p of a short planner rollout."""
    r = check_detail(bundle_dir)
    return max(r["motion"], r["event_p"] or 0.0)


if __name__ == "__main__":
    r = export(sys.argv[1], detail=True)
    print("max |ort - torch| =", max(r["motion"], r["event_p"] or 0.0), r)
