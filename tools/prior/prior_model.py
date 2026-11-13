"""The causal motion prior: GRU + mixture-density head over next-frame motion deltas + blink hazard.

Per 30 Hz frame t the network sees ONLY live-available inputs at <= t:
  sv (4)  SELF voice   [level 0..1.2, f0 semitones/12 rel. own median, voiced, speaking]   (avatar TTS)
  pv (4)  PARTNER voice [same]                                                              (user mic)
  pf (9)  PARTNER face [present, smile, jaw, brow, pitch_vel/30, yaw_vel/30, pitch/10, yaw/20, arm]
  y_{t-1} (D) its own previous motion, normalized;  armvis (1);  blink clock (1) = min(time since blink, 4 s)/4
and outputs p(delta_t) as a K-component diagonal Gaussian mixture in normalized-delta units + blink logit.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = ["yaw", "pitch", "roll", "tx", "ty", "tz", "gx", "gy", "brow_l", "brow_r", "smile", "squint", "wide", "arm"]
NIN_EXT = 4 + 4 + 9          # sv, pv, pf


class Prior(nn.Module):
    def __init__(self, D=len(OUT), H=128, K=4, E=64, C=1, nod=False):
        super().__init__()
        self.D, self.H, self.K, self.C, self.nod = D, H, K, C, nod
        self.DO = D * C                       # one step predicts C frames of deltas jointly (chunk)
        self.nin = NIN_EXT + D + 2
        self.inp = nn.Linear(self.nin, E)
        self.gru = nn.GRU(E, H, batch_first=True)
        self.out = nn.Linear(H, K + 2 * K * self.DO + 1 + (1 if nod else 0))

    def forward(self, x, h=None):
        z = torch.tanh(self.inp(x))
        o, h = self.gru(z, h)
        return self.head(self.out(o)), h

    def head(self, y):
        K, D = self.K, self.DO
        logit = y[..., :K]
        mu = y[..., K:K + K * D].reshape(*y.shape[:-1], K, D)
        ls = y[..., K + K * D:K + 2 * K * D].reshape(*y.shape[:-1], K, D).clamp(-5, 2)
        blink = y[..., K + 2 * K * D]
        if self.nod:
            return logit, mu, ls, blink, y[..., K + 2 * K * D + 1]
        return logit, mu, ls, blink


def mdn_nll(logit, mu, ls, target, mask):
    """target (B,T,D) normalized delta, mask (B,T,D) 0/1 per channel -> per-frame NLL (B,T)"""
    t = target.unsqueeze(-2)
    m = mask.unsqueeze(-2)
    lp = (-0.5 * ((t - mu) / ls.exp()) ** 2 - ls - 0.9189385) * m
    lp = lp.sum(-1) + F.log_softmax(logit, -1)
    return -torch.logsumexp(lp, -1)


def b64(a):
    import base64
    return "b64f32:" + base64.b64encode(np.ascontiguousarray(np.asarray(a, np.float32).ravel()).tobytes()).decode()


def model_json(model, norm, meta):
    """one model as a JSON-able dict; tensors as base64 little-endian float32 (exact parity with torch)"""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return dict(arch=dict(type="gru-mdn", D=model.D, H=model.H, K=model.K, E=model.inp.out_features, nin=model.nin,
                          C=model.C, nod=bool(model.nod), out=OUT),
                inp_w=b64(sd["inp.weight"]), inp_b=b64(sd["inp.bias"]),
                w_ih=b64(sd["gru.weight_ih_l0"]), w_hh=b64(sd["gru.weight_hh_l0"]),
                b_ih=b64(sd["gru.bias_ih_l0"]), b_hh=b64(sd["gru.bias_hh_l0"]),
                out_w=b64(sd["out.weight"]), out_b=b64(sd["out.bias"]),
                norm={k: [float(x) for x in np.ravel(v)] if not isinstance(v, (int, float)) else v for k, v in norm.items()},
                meta=meta)


def export(model, norm, path, meta):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    def q(a):
        return [float(f"{x:.5g}") for x in np.asarray(a, np.float64).ravel()]
    W = dict(arch=dict(type="gru-mdn", D=model.D, H=model.H, K=model.K, E=model.inp.out_features, nin=model.nin,
                       C=model.C, out=OUT),
             inp_w=q(sd["inp.weight"]), inp_b=q(sd["inp.bias"]),
             w_ih=q(sd["gru.weight_ih_l0"]), w_hh=q(sd["gru.weight_hh_l0"]),
             b_ih=q(sd["gru.bias_ih_l0"]), b_hh=q(sd["gru.bias_hh_l0"]),
             out_w=q(sd["out.weight"]), out_b=q(sd["out.bias"]),
             norm={k: q(v) if not isinstance(v, (int, float)) else v for k, v in norm.items()}, meta=meta)
    with open(path, "w") as f:
        json.dump(W, f, separators=(",", ":"))
    return W


class RX(nn.Module):
    """dyadic reaction model (tools/prior/train_reaction.py): partner-only inputs (14) -> 8 outputs
    [smile_rel mu, head_speed mu, their log-sigmas, logits: bignod, avert, brow, smile_rise]"""
    def __init__(s, H=64, NIN=14):
        super().__init__(); s.inp = nn.Linear(NIN, 32); s.gru = nn.GRU(32, H, batch_first=True); s.out = nn.Linear(H, 8)

    def forward(s, x, h=None):
        o, h = s.gru(torch.tanh(s.inp(x)), h); return s.out(o), h
