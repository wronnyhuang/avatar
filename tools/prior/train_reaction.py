"""The DYADIC reaction model: from the PARTNER's live-visible behaviour only (face, head, arm energy, voice/turn
state) + the self speaking flag, predict what THIS person's face/head does in the next ~0.5 s. No own-motion history
(non-autoregressive), so everything it predicts must come from the partner -> it is the learned response function.

Data: every dyad clip gives two training tracks (A reacting to B, B reacting to A). Strictly causal GRU (inputs <= t).
Targets at t: smile_rel(t+8) [own smile - own median], head speed mean over [t,t+15) (log1p deg/s),
P(big nod onset in [t,t+15)), P(gaze averted at t+8), P(brow raise onset in [t,t+15)), P(smile rise onset in [t,t+30)).
Usage: python -I train_reaction.py <scratch> <out_dir> [--epochs 30] [--tag rx]"""
import json, sys, os, time, argparse, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import mstats
from motionlib import COLS
from prior_model import b64

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("outdir")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--tag", default="rx"); ap.add_argument("--H", type=int, default=64)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--ablate", type=int, default=0, help="1 = hide the partner (baseline)")
ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--noise", type=float, default=0.0); ap.add_argument("--drop", type=float, default=0.0)
a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0); random.seed(0)
root = Path(a.root); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
D = dict(np.load(root / "dataset.npz")); meta = json.loads((root / "dataset_meta.json").read_text())
val_ids = set(json.loads((root / "split.json").read_text())["val"])
segs = D["seg"]; Mh = D["M"]
C = {n: i for i, n in enumerate(COLS)}
NIN = 14
TGT = ["smile_rel", "head_speed", "bignod", "avert", "brow", "smile_rise"]

X_all = np.concatenate([D["pf"], D["pv"], D["sv"][:, 3:4]], 1).astype(np.float32)
if a.ablate:
    X_all[:, :13] = 0.0
Y_all = np.zeros((len(Mh), 6), np.float32); V_all = np.zeros((len(Mh), 6), np.float32)
sel = []
for i, (s0, s1) in enumerate(segs):
    if D["pf"][s0:s1, 0].mean() < 0.5 or s1 - s0 < 120:
        continue
    seg = dict(M=Mh[s0:s1], blink=D["blink"][s0:s1], selfsp=D["sv"][s0:s1, 3] > 0, partsp=D["pv"][s0:s1, 3] > 0,
               f0=D["sraw"][s0:s1, 1])
    ev = mstats.events(seg); N = s1 - s0
    sm = np.nan_to_num(Mh[s0:s1, C["smile"]]); smr = sm - np.median(sm)
    ang = np.nan_to_num(Mh[s0:s1, :3]); sp = np.linalg.norm(np.gradient(ang, axis=0) * 30, axis=1)
    csp = np.concatenate([[0], np.cumsum(sp)])
    def any_in(idx, lo, hi):
        e = np.zeros(N + 1); e[np.clip(idx, 0, N - 1)] = 1; ce = np.concatenate([[0], np.cumsum(e)])
        t = np.arange(N); a_ = np.clip(t + lo, 0, N); b_ = np.clip(t + hi, 0, N); return (ce[b_] - ce[a_] > 0).astype(np.float32)
    t = np.arange(N); f8 = np.clip(t + 8, 0, N - 1); w15 = np.clip(t + 15, 0, N)
    y = np.zeros((N, 6), np.float32); v = np.ones((N, 6), np.float32)
    y[:, 0] = smr[f8]
    y[:, 1] = np.log1p((csp[w15] - csp[t]) / np.maximum(w15 - t, 1))
    y[:, 2] = any_in(np.maximum(ev["bignod"] - 6, 0), 0, 15)
    y[:, 3] = ev["avert_mask"][f8]
    y[:, 4] = any_in(ev["brow"], 0, 15)
    y[:, 5] = any_in(ev["smile"], 0, 30)
    v[N - 30:] = 0.0                                      # targets need the future window inside the segment
    Y_all[s0:s1] = y; V_all[s0:s1] = v
    sel.append(i)
isval = np.array([meta[int(D["segvid"][i])]["id"] in val_ids for i in range(len(segs))])
tr = [i for i in sel if not isval[i]]; va = [i for i in sel if isval[i]]
print(f"dyad tracks: train {sum(segs[i][1]-segs[i][0] for i in tr)/30/3600:.2f} h, val {sum(segs[i][1]-segs[i][0] for i in va)/30/3600:.2f} h", flush=True)
ymu = Y_all[np.concatenate([np.arange(*segs[i]) for i in tr])].mean(0); ysd = Y_all[np.concatenate([np.arange(*segs[i]) for i in tr])].std(0) + 1e-6


class RX(nn.Module):
    def __init__(s, H=64):
        super().__init__(); s.inp = nn.Linear(NIN, 32); s.gru = nn.GRU(32, H, batch_first=True); s.out = nn.Linear(H, 2 + 2 + 4)
    def forward(s, x, h=None):
        o, h = s.gru(torch.tanh(s.inp(x)), h); return s.out(o), h


def crops(ids, L=256):
    c = []
    for i in ids:
        s0, s1 = segs[i]
        for st in range(s0, max(s0 + 1, s1 - L + 1), L // 2):
            c.append((st, min(s1, st + L)))
    return c


def batch(cl):
    T = max(b - a_ for a_, b in cl)
    X = np.zeros((len(cl), T, NIN), np.float32); Y = np.zeros((len(cl), T, 6), np.float32); V = np.zeros((len(cl), T, 6), np.float32)
    for k, (s0, s1) in enumerate(cl):
        X[k, :s1 - s0] = X_all[s0:s1]; Y[k, :s1 - s0] = Y_all[s0:s1]; V[k, :s1 - s0] = V_all[s0:s1]
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(V)


def loss_fn(o, Y, V):
    mu = o[..., 0:2]; ls = o[..., 2:4].clamp(-4, 3); yn = (Y[..., :2] - torch.from_numpy(ymu[:2])) / torch.from_numpy(ysd[:2])
    g = (0.5 * ((yn - mu) / ls.exp()) ** 2 + ls) * V[..., :2]
    b = F.binary_cross_entropy_with_logits(o[..., 4:8], Y[..., 2:6], reduction="none") * V[..., 2:6]
    return g.sum() / V[..., :2].sum() + 2 * b.sum() / V[..., 2:6].sum()


model = RX(a.H); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=a.wd)
trc, vac = crops(tr), crops(va)
best = 1e9; log = []
for ep in range(a.epochs):
    random.shuffle(trc); model.train()
    for k in range(0, len(trc) - 31, 32):
        X, Y, V = batch(trc[k:k + 32])
        if a.noise > 0:
            X = X + a.noise * torch.randn_like(X)
        if a.drop > 0:      # drop whole partner feature channels per sequence (robustness)
            X = X * (torch.rand(X.shape[0], 1, X.shape[2]) > a.drop).float()
        o, _ = model(X); l = loss_fn(o, Y, V)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        vl = np.mean([loss_fn(model(batch(vac[k:k + 64])[0])[0], *batch(vac[k:k + 64])[1:]).item() for k in range(0, len(vac), 64)])
    log.append(float(vl)); print(f"ep {ep} val {vl:.4f}", flush=True)
    if vl < best:
        best = vl; torch.save(model.state_dict(), out / f"{a.tag}.pt")
model.load_state_dict(torch.load(out / f"{a.tag}.pt")); model.eval()

# ---- held-out evaluation: how much of THIS person's reaction is predictable from the partner alone
def auc(y, p):
    y = np.asarray(y) > 0.5; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = np.argsort(np.argsort(p)) + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


P, Yv, Vv = [], [], []
with torch.no_grad():
    for i in va:
        s0, s1 = segs[i]
        o, _ = model(torch.from_numpy(X_all[s0:s1])[None]); o = o[0].numpy()
        P.append(o); Yv.append(Y_all[s0:s1]); Vv.append(V_all[s0:s1])
P = np.concatenate(P); Yv = np.concatenate(Yv); Vv = np.concatenate(Vv)
ok = Vv[:, 0] > 0
pred_smile = P[ok, 0] * ysd[0] + ymu[0]; pred_speed = P[ok, 1] * ysd[1] + ymu[1]
ev = dict(val_loss=best, smile_rel_r=float(np.corrcoef(pred_smile, Yv[ok, 0])[0, 1]),
          head_speed_r=float(np.corrcoef(pred_speed, Yv[ok, 1])[0, 1]),
          auc={n: auc(Yv[ok, 2 + j], P[ok, 4 + j]) for j, n in enumerate(TGT[2:])},
          base_rates={n: float(Yv[ok, 2 + j].mean()) for j, n in enumerate(TGT[2:])}, ablate=a.ablate, log=log)
print(json.dumps(ev, indent=1))
(out / f"{a.tag}_eval.json").write_text(json.dumps(ev, indent=1))
sd = {k: v.numpy() for k, v in model.state_dict().items()}
W = dict(arch=dict(type="gru-reaction", nin=NIN, E=32, H=a.H, targets=TGT), inp_w=b64(sd["inp.weight"]),
         inp_b=b64(sd["inp.bias"]), w_ih=b64(sd["gru.weight_ih_l0"]), w_hh=b64(sd["gru.weight_hh_l0"]),
         b_ih=b64(sd["gru.bias_ih_l0"]), b_hh=b64(sd["gru.bias_hh_l0"]), out_w=b64(sd["out.weight"]), out_b=b64(sd["out.bias"]),
         norm=dict(ymu=[float(x) for x in ymu], ysd=[float(x) for x in ysd]), meta=dict(tag=a.tag, eval={k: v for k, v in ev.items() if k != "log"}))
(out / f"{a.tag}_weights.json").write_text(json.dumps(W, separators=(",", ":")))
print("saved", out / f"{a.tag}_weights.json")
