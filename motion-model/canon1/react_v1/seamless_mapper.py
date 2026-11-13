"""Copied from the first learned-reaction model's data tools (react_v1); imports patched for the canon1 package.
"""
"""Seamless -> canonical mapper: inference (numpy only) for the learned MLP, falling back to the analytic map.

Mapper file: data/mapper/mlp_v1.npz (written by seamless_mapper_fit.py) with, per group g in
seamless_features.GROUP_INPUTS: g/x_mu, g/x_sd, g/W0 (k*Fin, H) conv-as-dense over the +-K window, g/b0, g/W1, g/b1,
g/W2, g/b2, g/y_mu, g/y_sd; plus json 'cfg' (K, version, offsets, mask thresholds).
"""

import json
from pathlib import Path

import numpy as np

from . import seamless_features as F
from . import seamless_lib as L

MLP_PATH = L.MAPPER / "mlp_v1.npz"
POST_VERSION = "+c2"   # bump when predict() post-processing changes (forces the converter to re-run every pair)
SMOOTH_SIGMA = 1.0     # frames; Gaussian on head / body / hands (keypoint jitter: frame-diff std was 1.1-2.2x
#                        MediaPipe's on held-out files; sigma 1 brings it to ~0.8-1.1x at equal or better R2)
spec = L.spec
CALIB_PATH = L.MAPPER / "analytic_calib.json"


def window(X, K):
    """(T,F) -> (T,(2K+1)*F) stacking frames t-K..t+K (edge-padded)."""
    if K == 0:
        return X
    T = len(X)
    P = np.concatenate([np.repeat(X[:1], K, 0), X, np.repeat(X[-1:], K, 0)], 0)
    return np.concatenate([P[i:i + T] for i in range(2 * K + 1)], 1)


class Mapper:
    def __init__(self, path=None, allow_mlp=True):
        path = Path(path) if path else MLP_PATH
        self.mlp = None
        if allow_mlp and path.exists():
            d = np.load(path, allow_pickle=False)
            self.mlp = {k: d[k] for k in d.files}
            self.cfg = json.loads(str(self.mlp["cfg"]))
            self.source = self.cfg.get("source", "mapped-seamless-mlp-v1")
            self.version = self.cfg.get("version", "mlp") + POST_VERSION
        else:
            self.cfg = {}
            self.calib = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else None
            self.source = "mapped-seamless-analytic"
            import hashlib
            self.version = "analytic-" + (hashlib.sha1(CALIB_PATH.read_bytes()).hexdigest()[:8] if self.calib
                                          else "uncalibrated")

    def _forward(self, g, X):
        """mean over the seed ensemble (keys g/..., g#1/..., g#2/...)."""
        K = int(self.cfg["K"])
        nt = len(F.GROUP_TARGETS[g])
        ys, ls = [], []
        for i in range(int(self.cfg.get("n_seeds", 1))):
            pre = g if i == 0 else f"{g}#{i}"
            p = {k[len(pre) + 1:]: v for k, v in self.mlp.items() if k.startswith(pre + "/")}
            Xn = np.clip((X - p["x_mu"]) / p["x_sd"], -8, 8)
            Xw = window(Xn.astype(np.float32), K)
            h = np.maximum(Xw @ p["W0"] + p["b0"], 0)
            h = np.maximum(h @ p["W1"] + p["b1"], 0)
            out = h @ p["W2"] + p["b2"]
            ys.append(out[:, :nt] * p["y_sd"] + p["y_mu"]); ls.append(out[:, nt:])
        y = np.mean(ys, 0); logits = np.mean(ls, 0)
        if self.cfg.get("residual"):
            y = y + X[:, -nt:]
        return y, logits

    def predict(self, B, aux, T):
        if self.mlp is None:
            return F.analytic(B, aux, T, self.calib)
        mot = np.zeros((T, 86), np.float32)
        msk = np.zeros((T, 4), np.uint8)
        if self.cfg.get("residual"):
            B = dict(B, A=F.analytic(B, aux, T, self.cfg["calib"])[0])
        thr = self.cfg.get("mask_thr", {})
        for g in F.GROUP_INPUTS:
            y, lg = self._forward(g, F.group_X(B, g, self.cfg.get("group_inputs")))
            mot[:, F.GROUP_TARGETS[g]] = y
            for j, mi in enumerate(F.GROUP_MASKS[g]):
                msk[:, mi] = lg[:, j] > float(thr.get(str(mi), 0.0))
        from scipy.ndimage import gaussian_filter1d
        mot[:, 52:58] = gaussian_filter1d(mot[:, 52:58], SMOOTH_SIGMA, axis=0, mode="nearest")
        mot[:, 60:86] = gaussian_filter1d(mot[:, 60:86], SMOOTH_SIGMA, axis=0, mode="nearest")
        mot[:, :52] = np.clip(mot[:, :52], 0, 1)
        for sl in (spec.SL_HAND_L, spec.SL_HAND_R):     # hand descriptor ranges: open/thumb/point 0..1, palm -1..1
            mot[:, sl.start:sl.start + 3] = np.clip(mot[:, sl.start:sl.start + 3], 0, 1)
            mot[:, sl.start + 3] = np.clip(mot[:, sl.start + 3], -1, 1)
        # face mask also requires valid Imitator features (garbage in -> masked out)
        msk[:, 0] &= (aux["mv_valid"] & aux["face_ok"]).astype(np.uint8)
        return mot, msk
