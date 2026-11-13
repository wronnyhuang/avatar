"""reaction adapter: measure the TRAIN data's reaction excess (labels) before / after the R3 cartoon injection.
    python -I tools/react/tune_inject.py --config <lane json> --pack-dir <dir> [--batches 30] [--f-smile 0.12 --f-nod 0.15]
excess = P(self onset 0.2-2 s after a partner onset) - P(self onset in a random 1.8-s window) (the scorecard's definition,
on the pack's onset labels; partner laugh = the causal trace detector)."""
import argparse, json, os, sys
import numpy as np
import torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "tools", "train"))
from train import resolve_pack, sampler_kwargs, DEFAULTS  # noqa: E402
from multi import merge  # noqa: E402
from unified.data import Sampler  # noqa: E402
import next_arms as NA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--pack-dir", required=True)
ap.add_argument("--batches", type=int, default=30); ap.add_argument("--fs", default="0.06,0.12,0.2,0.3")
ap.add_argument("--fn", default="0.1,0.2,0.3")
a = ap.parse_args()
top = json.load(open(a.config)); top.pop("models", None)
c0 = merge(DEFAULTS, top)
torch.set_num_threads(12)
smp = Sampler(resolve_pack(c0["pack"], a.pack_dir), "cpu", seed=5, **sampler_kwargs(c0, a.pack_dir))
lo, hi = 6, 60


def excess(trig, self_on):
    B, nf = trig.shape
    cs = torch.cat([torch.zeros(B, 1), torch.cumsum(self_on, 1)], 1)
    t = torch.arange(nf)
    a_ = (t + lo).clamp(max=nf); e_ = (t + hi + 1).clamp(max=nf)
    hitw = (cs[:, e_] - cs[:, a_]) > 0
    ok = t + hi < nf
    m = (trig > 0) & ok
    rnd = torch.zeros(nf, dtype=torch.bool); rnd[::5] = True
    return float(hitw[m].float().sum()), int(m.sum()), float(hitw[:, rnd & ok].float().mean())


res = {}
for i in range(a.batches):
    b = smp.batch(96, c0["seq"], c0["chunks"], split="train", aug=c0["aug"])
    ev_s, ev_p, inb, nf = NA._onsets_nf(b, smp)
    lau = NA._laugh_nf(b, smp, nf)
    trig = {"smile->smile": (ev_p[..., 2], 2), "laugh->smile": (lau, 2), "nod->nod": (ev_p[..., 0], 0)}
    for fs in [0.0] + [float(x) for x in a.fs.split(",")]:
        for fn in ([0.0] + [float(x) for x in a.fn.split(",")]) if fs == 0.0 else [0.0]:
            bb = dict(b)
            torch.manual_seed(100 + i)
            NA.inject_reactions(bb, smp, dict(NA.INJECT_DEFAULT, f_smile=fs, f_nod=fn))
            lab = bb.get("_inject_lab")
            so = ev_s if lab is None else torch.maximum(ev_s, lab)
            for k, (tr, c) in trig.items():
                h, n, ch = excess(tr, so[..., c])
                r = res.setdefault(f"fs{fs}_fn{fn}", {}).setdefault(k, [0.0, 0, 0.0, 0])
                r[0] += h; r[1] += n; r[2] += ch; r[3] += 1
out = {}
for key, d in res.items():
    out[key] = {k: dict(rate=round(v[0] / max(v[1], 1), 4), chance=round(v[2] / v[3], 4),
                        excess=round(v[0] / max(v[1], 1) - v[2] / v[3], 4), n=v[1]) for k, v in d.items()}
    print(key, json.dumps(out[key]), flush=True)
_od = os.path.join(os.environ.get("MM_WORK_DIR", os.path.join(R, "work")), "runs", "p2react")
os.makedirs(_od, exist_ok=True)
json.dump(out, open(os.path.join(_od, "tune_inject.json"), "w"), indent=1)
