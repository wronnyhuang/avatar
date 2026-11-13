"""reaction adapter: causal PARTNER-EVENT TRACE + its adapter on the backbone output (cfg["model"]["react_trace"]).

The 40M HRC model's backbone tells the real partner from a shuffled one (contrast head 83-85%) but its outputs ignore
the partner's smiles / nods. This gives the heads a direct, explicit path: per token tau (which holds the user's
frame tau+1), causal detectors on the token's own USER block (the same values training and the runtime see: the
user's canon motion normalized with norm.npz, the user mask, the user's vad) produce onsets of
    smile   (mean mouthSmile rises >= 0.15 within 1 s; first crossing after 1 s without one)
    nod     (|pitch - its 1-s causal mean| >= 2 deg, pitch = neck_x + head_x; first after 1 s quiet)
    laugh   (smile >= 0.45 and jawOpen >= 0.2 while the user speaks; first after 1 s)
    speech  (user vad on after >= 0.5 s of silence)
and per class the features [onset now, exp(-d/10), exp(-d/30), d in 9..60 frames (the 0.3-2 s reaction window)],
d = frames since the class's last onset (none within 120 frames = all 0), plus the continuous [smile, smile rise,
jaw, pitch high-pass / 5, vad] -> F = 21 values, all 0 where the user group is masked (camera off, CFG-null stream).
ReactAdapter: h' = h + MLP(trace) (last layer zero-init, so a fine-tune starts as the source model); h' feeds the
event head AND the flow head's memory. The runtime (unified/flow_core.py) computes the same trace in numpy over the
last RING tokens (trace_np; test: unified/test_react.py) and adds the adapter (react.npz in the bundle).
"""
from __future__ import annotations

import math

import numpy as np

from canon1 import spec

FPS = 30
CLASSES = ("smile", "nod", "laugh", "speech")
N_FEAT = 4 * len(CLASSES) + 5
DMAX = 120                                # frames: an onset older than this is forgotten (finite support)
RING = DMAX + 45                          # runtime window (>= DMAX + the 1-s detector windows)
SMILE_L, SMILE_R = 135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]
JAW = 135 + spec.BS["jawOpen"]
NECK_X, HEAD_X = spec.COL["neck_x"], spec.COL["head_x"]
RAW_COLS = (SMILE_L, SMILE_R, JAW, NECK_X, HEAD_X)
G_HEAD, G_MOUTH = spec.GROUP_NAMES.index("head"), spec.GROUP_NAMES.index("mouth")
AMP_SMILE, AMP_NOD, QUIET = 0.15, 2.0, 30


def norm_info(mean, std, amean, astd):
    """the scalars the trace needs (stored in cfg["model"]["react_trace"]): raw = x * std + mean for RAW_COLS, vad."""
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    return dict(mean=[float(mean[c]) for c in RAW_COLS], std=[float(std[c]) for c in RAW_COLS],
                vad_mean=float(np.asarray(amean)[41]), vad_std=float(np.asarray(astd)[41]))


def token_cols(lay):
    """token columns the trace reads: the 5 user motion cols, the user head / mouth mask, the user vad."""
    u0, m0 = lay["user"].start, lay["user_mask"].start
    return [u0 + c for c in RAW_COLS] + [m0 + G_HEAD, m0 + G_MOUTH, lay["user_audio"].stop - 1]


# ------------------------------------------------------------------------------------------------ numpy (runtime)
def _win_any_np(x, lo, hi):
    """r[t] = any x[t-hi .. t-lo] (frames before t; lo >= 0), x bool (T,)."""
    c = np.concatenate([[0], np.cumsum(x.astype(np.int64))])
    t = np.arange(len(x))
    a = np.clip(t - hi, 0, None); e = np.clip(t - lo + 1, 0, None)
    return (c[e] - c[a]) > 0


def _win_min_np(x, w):
    """out[t] = min x[t-w+1 .. t] (window clipped at the start)."""
    xp = np.concatenate([np.full(w - 1, np.inf, x.dtype), x])
    return np.lib.stride_tricks.sliding_window_view(xp, w).min(1)


def trace_np(rows, ni):
    """rows (T, 8) = the token values at token_cols(lay) for consecutive tokens -> (T, N_FEAT) float32."""
    r = np.asarray(rows, np.float32)
    m, s = np.asarray(ni["mean"], np.float32), np.asarray(ni["std"], np.float32)
    raw = r[:, :5] * s + m
    vh, vm = r[:, 5] > 0, r[:, 6] > 0
    vad = (r[:, 7] * np.float32(ni["vad_std"]) + np.float32(ni["vad_mean"])) > 0.5
    T = len(r)
    sm = np.where(vm, 0.5 * (raw[:, 0] + raw[:, 1]), 0.0).astype(np.float32)
    jaw = np.where(vm, raw[:, 2], 0.0).astype(np.float32)
    rise = np.where(vm, sm - _win_min_np(np.where(vm, sm, np.float32(np.inf)), 30), 0.0).astype(np.float32)
    rise = np.where(np.isfinite(rise), rise, 0.0)
    p = np.where(vh, (raw[:, 3] + raw[:, 4]) * np.float32(180.0 / math.pi), 0.0).astype(np.float32)
    c = np.concatenate([[0.0], np.cumsum(p.astype(np.float64))]); cv = np.concatenate([[0], np.cumsum(vh)])
    t = np.arange(T); a = np.clip(t - 29, 0, None)
    mu = (c[t + 1] - c[a]) / np.maximum(cv[t + 1] - cv[a], 1)
    hp = np.where(vh, p - mu, 0.0).astype(np.float32)
    trig = [rise >= AMP_SMILE, np.abs(hp) >= AMP_NOD, vm & (sm >= 0.45) & (jaw >= 0.2) & vad, vad]
    quiet = [QUIET, QUIET, QUIET, 15]
    out = np.zeros((T, N_FEAT), np.float32)
    for k, (b, q) in enumerate(zip(trig, quiet)):
        on = b & ~_win_any_np(b, 1, q)
        last = np.maximum.accumulate(np.where(on, t, -10 ** 6))
        d = t - last
        has = d <= DMAX
        out[:, 4 * k] = on
        out[:, 4 * k + 1] = np.where(has, np.exp(-d / 10.0), 0.0)
        out[:, 4 * k + 2] = np.where(has, np.exp(-d / 30.0), 0.0)
        out[:, 4 * k + 3] = has & (d >= 9) & (d <= 60)
    out[:, 16], out[:, 17], out[:, 18] = sm, rise, jaw
    out[:, 19], out[:, 20] = np.clip(hp / 5.0, -3, 3), vad
    out[~vm, 0:4] = 0; out[~vm, 8:12] = 0; out[~vh, 4:8] = 0
    return out


def adapter_np(tr, w):
    """the ReactAdapter in numpy: w = dict(W1 (F,k), b1, W2 (k,d), b2)."""
    x = tr @ w["W1"] + w["b1"]
    x = 0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + 0.044715 * x ** 3)))
    return (x @ w["W2"] + w["b2"]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ torch (training)
def _torch():
    import torch
    return torch


def _win_any_t(x, lo, hi):
    T_ = _torch()
    B, T = x.shape
    c = T_.cat([T_.zeros(B, 1, device=x.device), T_.cumsum(x.float(), 1)], 1)
    t = T_.arange(T, device=x.device)
    a = (t - hi).clamp(min=0); e = (t - lo + 1).clamp(min=0)
    return (c[:, e] - c[:, a]) > 0


def _win_min_t(x, w):
    T_ = _torch()
    xp = T_.cat([T_.full((x.shape[0], w - 1), float("inf"), device=x.device), x], 1)
    return -T_.nn.functional.max_pool1d(-xp[:, None], w, stride=1)[:, 0]


def trace_torch(tok, lay, ni):
    """tok (B,S,DIN) tokens -> (B,S,N_FEAT) trace (no grad; the same rules as trace_np)."""
    T_ = _torch()
    with T_.no_grad():
        r = tok[..., token_cols(lay)].float()
        dev = r.device
        m = T_.tensor(ni["mean"], device=dev); s = T_.tensor(ni["std"], device=dev)
        raw = r[..., :5] * s + m
        vh, vm = r[..., 5] > 0, r[..., 6] > 0
        vad = (r[..., 7] * float(ni["vad_std"]) + float(ni["vad_mean"])) > 0.5
        B, T = r.shape[:2]
        z = T_.zeros_like(raw[..., 0])
        sm = T_.where(vm, 0.5 * (raw[..., 0] + raw[..., 1]), z)
        jaw = T_.where(vm, raw[..., 2], z)
        mn = _win_min_t(T_.where(vm, sm, T_.full_like(sm, float("inf"))), 30)
        rise = T_.where(vm & T_.isfinite(mn), sm - mn, z)
        p = T_.where(vh, (raw[..., 3] + raw[..., 4]) * (180.0 / math.pi), z)
        c = T_.cat([T_.zeros(B, 1, device=dev, dtype=T_.float64), T_.cumsum(p.double(), 1)], 1)
        cv = T_.cat([T_.zeros(B, 1, device=dev), T_.cumsum(vh.float(), 1)], 1)
        t = T_.arange(T, device=dev); a = (t - 29).clamp(min=0)
        mu = ((c[:, t + 1] - c[:, a]) / (cv[:, t + 1] - cv[:, a]).clamp(min=1).double()).float()
        hp = T_.where(vh, p - mu, z)
        trig = [rise >= AMP_SMILE, hp.abs() >= AMP_NOD, vm & (sm >= 0.45) & (jaw >= 0.2) & vad, vad]
        quiet = [QUIET, QUIET, QUIET, 15]
        feats = []
        tt = t[None].expand(B, T)
        for b, q in zip(trig, quiet):
            on = b & ~_win_any_t(b, 1, q)
            last = T_.cummax(T_.where(on, tt, T_.full_like(tt, -10 ** 6)), 1).values
            d = (tt - last).float()
            has = d <= DMAX
            f0 = on.float()
            f1 = T_.where(has, T_.exp(-d / 10.0), z)
            f2 = T_.where(has, T_.exp(-d / 30.0), z)
            f3 = (has & (d >= 9) & (d <= 60)).float()
            feats += [f0, f1, f2, f3]
        feats += [sm, rise, jaw, (hp / 5.0).clamp(-3, 3), vad.float()]
        out = T_.stack(feats, -1)
        gm = T_.ones_like(out)
        gm[..., 0:4] = vm[..., None].float(); gm[..., 8:12] = vm[..., None].float(); gm[..., 4:8] = vh[..., None].float()
        return out * gm


def make_adapter(d, hidden=64):
    T_ = _torch()
    nn = T_.nn

    class ReactAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, d))
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

        def forward(self, tr):
            return self.net(tr)

        def np_weights(self):
            n = self.net
            return dict(W1=n[0].weight.detach().cpu().numpy().T.copy(), b1=n[0].bias.detach().cpu().numpy(),
                        W2=n[2].weight.detach().cpu().numpy().T.copy(), b2=n[2].bias.detach().cpu().numpy())

    return ReactAdapter()
