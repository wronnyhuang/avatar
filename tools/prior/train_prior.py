"""Train the causal motion prior on dataset.npz (video-level split).
Usage: python -I train_prior.py <scratch> <out_dir> [--epochs N] [--tag name] [--subset all|dyad] [--init ckpt]"""
import json, sys, os, time, argparse, random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import Prior, mdn_nll, export, OUT
from motionlib import COLS

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("outdir")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--tag", default="general")
ap.add_argument("--subset", default="all"); ap.add_argument("--init", default=None)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--H", type=int, default=128)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--arnoise", type=float, default=0.05); ap.add_argument("--T", type=int, default=256)
ap.add_argument("--chunk", type=int, default=1); ap.add_argument("--nod", type=int, default=0)
ap.add_argument("--freeze", default="", help="unused"); ap.add_argument("--epochs-warm", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0); random.seed(0)
root = Path(a.root); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
D = dict(np.load(root / "dataset.npz")); meta = json.loads((root / "dataset_meta.json").read_text())
cols = [COLS.index(c) for c in OUT]
M = D["M"][:, cols].astype(np.float32)
armvis = np.isfinite(M[:, OUT.index("arm")]).astype(np.float32)

# ---- video-level split (stable: by hash of id), stratified by category
sp = root / "split.json"
if sp.exists():
    val_ids = set(json.loads(sp.read_text())["val"])
else:
    rng = random.Random(42); val_ids = set()
    bycat = {}
    for v in meta:
        bycat.setdefault(v["category"], []).append(v["id"])
    for c, ids in bycat.items():
        ids = sorted(set(ids)); rng.shuffle(ids); val_ids |= set(ids[: max(1, int(round(0.15 * len(ids))))])
    sp.write_text(json.dumps({"val": sorted(val_ids)}))
segs = D["seg"]; segvid = D["segvid"]
isval = np.array([meta[v]["id"] in val_ids for v in segvid])
dyad = np.array([D["pf"][s0:s1, 0].mean() > 0.5 for s0, s1 in segs])

# ---- normalization (train frames)
trmask = np.zeros(len(M), bool)
for (s0, s1), v in zip(segs, isval):
    if not v: trmask[s0:s1] = True
med = np.nanmedian(M[trmask], 0)
arm_fill = med[OUT.index("arm")]
M[:, OUT.index("arm")] = np.where(armvis > 0, M[:, OUT.index("arm")], arm_fill)
M = np.nan_to_num(M)
pos_c = med.copy(); pos_c[:8] = 0.0        # head/gaze already neutral-relative
pos_s = np.nanpercentile(M[trmask], 84, 0) - np.nanpercentile(M[trmask], 16, 0)
pos_s = np.maximum(pos_s / 2, 1e-3)
Y = (M - pos_c) / pos_s
Y = np.clip(Y, -5, 5)
dY = np.zeros_like(Y)
for s0, s1 in segs:
    dY[s0 + 1:s1] = Y[s0 + 1:s1] - Y[s0:s1 - 1]
d_s = np.maximum(dY[trmask].std(0), 1e-4)
dN = np.clip(dY / d_s, -8, 8)
# blink clock
bl = D["blink"]; clock = np.ones(len(bl), np.float32)
for s0, s1 in segs:
    c = 4.0
    for t in range(s0, s1):
        clock[t] = min(c, 4.0) / 4.0
        c = 0.0 if bl[t] > 0 else c + 1 / 30
ext = np.concatenate([D["sv"], D["pv"], D["pf"]], 1).astype(np.float32)
# big-nod onsets (the explicit nod channel): mstats nod detector, prominence >= 5 deg; onset = peak - 6 frames
import mstats
nodon = np.zeros(len(M), np.float32)
for s0, s1 in segs:
    ev = mstats.events(dict(M=D["M"][s0:s1], blink=D["blink"][s0:s1], selfsp=D["sv"][s0:s1, 3] > 0,
                            partsp=D["pv"][s0:s1, 3] > 0, f0=D["sraw"][s0:s1, 1]))
    for pk in ev["bignod"]:
        nodon[s0 + max(0, pk - 6)] = 1
print(f"big-nod onsets: {int(nodon.sum())} ({nodon.sum() / (len(M) / 30 / 60):.1f}/min)", flush=True)
ext = np.nan_to_num(ext)

def make_x(idx0, idx1):
    """inputs for frames idx0..idx1-1 (y_{t-1} from the previous frame, first frame uses itself)"""
    yprev = np.concatenate([Y[idx0:idx0 + 1], Y[idx0:idx1 - 1]])
    cprev = np.concatenate([[1.0], clock[idx0 + 1:idx1]]) if idx1 - idx0 > 1 else np.array([1.0])
    cprev = clock[idx0:idx1]
    return np.concatenate([ext[idx0:idx1], yprev, armvis[idx0:idx1, None], cprev[:, None]], 1)

def crops(select):
    out = []
    for i, (s0, s1) in enumerate(segs):
        if not select[i]: continue
        L = s1 - s0
        if L <= a.T:
            out.append((s0, s1))
        else:
            for st in range(s0, s1 - a.T + 1, a.T // 2):
                out.append((st, st + a.T))
    return out

sel_tr = ~isval & (dyad if a.subset == "dyad" else True)
sel_va = isval & (dyad if a.subset == "dyad" else True)
tr, va = crops(sel_tr), crops(sel_va)
print(f"train crops {len(tr)} ({sum(b-a_ for a_,b in tr)/30/3600:.2f} h)  val crops {len(va)} ({sum(b-a_ for a_,b in va)/30/3600:.2f} h)", flush=True)

CH = a.chunk
def batch(cl, noise):
    """chunked steps: anchor a = s0 + k*CH; inputs = mean ext over the last CH frames (causal) + y[a];
    targets = the CH deltas of frames a+1..a+CH (flattened), blink = any onset in a+1..a+CH"""
    nst = [max(1, (b - a_ - 1) // CH) for a_, b in cl]
    T = max(nst); Dd = len(OUT)
    X = np.zeros((len(cl), T, 17 + Dd + 2), np.float32); Tg = np.zeros((len(cl), T, CH * Dd), np.float32)
    Mk = np.zeros((len(cl), T, CH * Dd), np.float32); Bk = np.zeros((len(cl), T), np.float32); Fm = np.zeros((len(cl), T), np.float32)
    Nk = np.zeros((len(cl), T), np.float32)
    ai = OUT.index("arm")
    for i, (s0, s1) in enumerate(cl):
        n = nst[i]
        if (s1 - s0 - 1) // CH < 1:
            continue
        anc = s0 + np.arange(n) * CH
        if CH == 1:
            e = ext[anc]
        else:
            e = np.mean([ext[np.maximum(anc - j, s0)] for j in range(CH)], 0)
        X[i, :n] = np.concatenate([e, Y[anc], armvis[anc, None], clock[anc, None]], 1)
        fr = anc[:, None] + 1 + np.arange(CH)[None, :]
        Tg[i, :n] = dN[fr].reshape(n, -1)
        mk = np.ones((n, CH, Dd), np.float32); mk[:, :, ai] = armvis[fr]
        Mk[i, :n] = mk.reshape(n, -1)
        Bk[i, :n] = bl[fr].max(1); Fm[i, :n] = 1.0; Nk[i, :n] = nodon[fr].max(1)
    if noise > 0:
        X[:, :, 17:17 + Dd] += np.random.randn(*X[:, :, 17:17 + Dd].shape).astype(np.float32) * noise
    return [torch.from_numpy(z) for z in (X, Tg, Mk, Bk, Fm, Nk)]

model = Prior(H=a.H, K=a.K, C=a.chunk, nod=bool(a.nod))
if a.init:
    sd0 = torch.load(a.init); sd = model.state_dict()
    for k, v in sd0.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
        elif k.startswith("out.") and sd[k].shape[0] == v.shape[0] + 1:     # adding the nod logit row
            sd[k][:v.shape[0]] = v
    model.load_state_dict(sd)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
steps = a.epochs * max(1, len(tr) // 64)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps + 5, pct_start=0.1)

def evaluate(cl):
    model.eval(); tot = 0.0; n = 0; btot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, len(cl), 64):
            X, Tg, Mk, Bk, Fm, Nk = batch(cl[i:i + 64], 0.0)
            hd, _ = model(X); lg, mu, ls, b = hd[:4]
            nll = mdn_nll(lg, mu, ls, Tg, Mk)
            m = Fm
            tot += (nll * m).sum().item() / CH; n += m.sum().item()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(b, Bk, reduction="none")
            btot += (bce * Fm).sum().item(); nb += Fm.sum().item()
    model.train()
    return tot / max(n, 1), btot / max(nb, 1)

log = []
t0 = time.time(); best = 1e9
for ep in range(a.epochs):
    random.shuffle(tr)
    for i in range(0, len(tr) - 63, 64):
        X, Tg, Mk, Bk, Fm, Nk = batch(tr[i:i + 64], a.arnoise)
        hd, _ = model(X); lg, mu, ls, b = hd[:4]
        nll = mdn_nll(lg, mu, ls, Tg, Mk)
        m = Fm
        loss = (nll * m).sum() / m.sum() / CH
        bce = torch.nn.functional.binary_cross_entropy_with_logits(b, Bk, reduction="none")
        loss = loss + 4.0 * (bce * Fm).sum() / Fm.sum()
        if a.nod:
            nb = torch.nn.functional.binary_cross_entropy_with_logits(hd[4], Nk, reduction="none")
            loss = loss + 4.0 * (nb * Fm).sum() / Fm.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    vn, vb = evaluate(va)
    log.append(dict(epoch=ep, val_nll=vn, val_blink_bce=vb, train_loss=float(loss.item()), t=time.time() - t0))
    print(f"ep {ep} C={CH} train {loss.item():.3f} val_nll/frame {vn:.3f} (D={len(OUT)}) val_blink_bce {vb:.4f} t={time.time()-t0:.0f}s", flush=True)
    if vn + 4 * vb < best:
        best = vn + 4 * vb; torch.save(model.state_dict(), out / f"{a.tag}.pt")
model.load_state_dict(torch.load(out / f"{a.tag}.pt"))
norm = dict(pos_c=pos_c, pos_s=pos_s, d_s=d_s, arm_fill=float(arm_fill))
info = dict(tag=a.tag, subset=a.subset, val=log[-1], best=best, log=log, train_hours=sum(b-a_ for a_,b in tr)/30/3600/2,
            val_ids=sorted(val_ids))
export(model, norm, out / f"{a.tag}_weights.json", dict(tag=a.tag, subset=a.subset, val_nll=min(l["val_nll"] for l in log)))
(out / f"{a.tag}_train.json").write_text(json.dumps(info, indent=1))
np.savez(out / f"{a.tag}_norm.npz", **norm)
print("saved", out / f"{a.tag}_weights.json", os.path.getsize(out / f"{a.tag}_weights.json"))
