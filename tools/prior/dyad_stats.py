"""Summarize dyad extraction: accepted / rejected counts by reason, hours of dyad footage, layouts, per-clip checks
(present inside dyad_ok, A/B median face position in full-frame coords, A/B jawOpen correlation).
Usage: python -I dyad_stats.py <scratch> [--per-clip] [--recheck]
"""
import json, sys, collections
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]); per = "--per-clip" in sys.argv
if "--recheck" in sys.argv:
    # post-hoc duplicate-feed check (for clips extracted before dyad_extract had it): move to feats_dyad_dup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dyad_extract import dup_check
    (root / "feats_dyad_dup").mkdir(exist_ok=True)
    for f in sorted((root / "feats_dyad").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        d, why = dup_check(z["present"], z["blend"], z["mat"])
        ok = z["dyad_ok"].astype(bool); pr = z["present"].astype(bool)
        pin = [float(pr[s][ok].mean()) if ok.any() else 0.0 for s in range(2)]
        if d:
            why = f"same person twice ({why})"
        elif min(pin) < 0.6:
            d, why = True, f"face tracked < 60% inside dyad_ok ({[round(v, 3) for v in pin]})"
        if d:
            (root / "feats_dyad_reject" / f"{f.stem}.json").write_text(json.dumps(dict(
                reason=f"{why} [post-hoc recheck; npz moved to feats_dyad_dup/]", info={})))
            f.rename(root / "feats_dyad_dup" / f.name)
            print(f"[dup] {f.stem} {why}")
feats = sorted((root / "feats_dyad").glob("*.npz"))
feats = [f for f in feats if not f.name.endswith(".tmp.npz")]
rejs = sorted((root / "feats_dyad_reject").glob("*.json"))
man = {}
for m in ("manifest_dyad.jsonl", "manifest.jsonl"):
    p = root / m
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                r = json.loads(l); man[Path(r["file"]).stem] = r
            except Exception:
                pass
reasons = collections.Counter()
for f in rejs:
    try:
        r = json.loads(f.read_text())["reason"]
    except Exception:
        r = "?"
    key = r.split(" (")[0].split(" <")[0].split(" 0.")[0] if not r.startswith("<2") else "<2 stable faces"
    if r.startswith("co-presence"): key = "co-presence < 0.5"
    if r.startswith("dyad layout covers"): key = "dyad layout < 50%"
    if r.startswith("face tracked"): key = "face tracked < 60% in dyad_ok"
    if r.startswith("same person"): key = "same person twice"
    reasons[key] += 1
lay = collections.Counter(); ok_h = both_h = raw_h = 0.0; vids = set(); good = 0
src = collections.Counter()
for f in feats:
    z = np.load(f)
    L = str(z["layout"]); lay[L] += 1
    ok = z["dyad_ok"].astype(bool); pr = z["present"].astype(bool)
    n = len(ok); raw_h += n / 30 / 3600; ok_h += ok.sum() / 30 / 3600
    both = ok & pr[0] & pr[1]; both_h += both.sum() / 30 / 3600
    vids.add(f.stem.rsplit("_", 1)[0])
    r = man.get(f.stem, {})
    src["harvest" if str(r.get("query", "")).startswith("harvest:") else r.get("category", "?")] += 1
    pin = [float(pr[s][ok].mean()) if ok.any() else 0.0 for s in range(2)]
    if min(pin) > 0.6: good += 1
    if per:
        cr = z["crops"]; lm = z["lmk"]
        pos = []
        for s in range(2):
            m = pr[s]
            if m.any():
                nx = np.median(lm[s, m, 0, 0]); ny = np.median(lm[s, m, 0, 1])
                pos.append((cr[s, 0] + nx * (cr[s, 2] - cr[s, 0]), cr[s, 1] + ny * (cr[s, 3] - cr[s, 1])))
            else:
                pos.append((np.nan, np.nan))
        jb = pr[0] & pr[1]
        jc = float(np.corrcoef(z["blend"][0, jb, 25], z["blend"][1, jb, 25])[0, 1]) if jb.sum() > 50 else np.nan
        print(f"{f.stem:22s} {L:9s} n={n} ok={ok.mean():.2f} in_ok=[{pin[0]:.2f},{pin[1]:.2f}] both={both.mean():.2f} "
              f"faceA=({pos[0][0]:.2f},{pos[0][1]:.2f}) faceB=({pos[1][0]:.2f},{pos[1][1]:.2f}) jawcorr={jc:.2f} "
              f"crops={np.round(cr, 2).tolist()}")
print(f"accepted clips: {len(feats)} from {len(vids)} videos (sources {dict(src)}); both-present>0.6 in dyad_ok: {good}")
print(f"rejected clips: {len(rejs)} {dict(reasons)}")
print(f"layouts: {dict(lay)}")
print(f"hours: raw {raw_h:.2f}  dyad_ok {ok_h:.2f}  dyad_ok & both faces present {both_h:.2f}")
