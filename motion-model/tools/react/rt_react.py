"""Reaction excess (smile->smile, nod->nod, laugh->smile ...) from rt_probe --save npz + the jobs pickle.

    python -I tools/react/rt_react.py <probe>.npz [--jobs runs/p2rt/jobs_val.pkl] [--json out.json]
Prints excess (hit rate after user events in 0.2-2 s minus chance rate) for the generated motion and, for reference,
for the real partner in the same dyads (the 'people' row). Same functions as evalkit/headline.py reactions().
"""
import argparse, json, os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
T = os.environ.get("MM_WORK_DIR", os.path.join(ROOT, "work"))
PAIRS = ("smile->smile", "nod->nod", "laugh->smile")


def compute(npz, jobs_pkl, which=("gen", "real")):
    sys.path.insert(0, os.path.join(ROOT, "evalkit"))
    import scorecard as SC
    SC._kit()
    M = SC.M
    z = np.load(npz)
    jobs = pickle.load(open(jobs_pkl, "rb"))["jobs"]
    per = {w: {} for w in which}
    sm = {w: [0, 0.0] for w in which}
    for i in range(len(z["gen"])):
        j = jobs[i]
        listen = np.asarray(j["listen"], bool); vad = np.asarray(j["ua"])[:, 41] > 0.5
        N = len(listen)
        own = M._bridge(~listen, int(1.2 * 30)); st = np.where(own, 1, np.where(vad, 2, 0))
        Ut = M.table_from_canon(np.asarray(j["um"], np.float32))
        uev = M.user_events(Ut, vad, N)
        for w in which:
            m = z["gen"][i].astype(np.float32) if w == "gen" else np.asarray(j["real"], np.float32)
            Tb = M.table_from_canon(m)
            rc = M.reaction_counts(uev, M.avatar_events(Tb, st), N)
            for k, v in rc.items():
                per[w].setdefault(k, []).append(v)
            sm[w][0] += len(M.avatar_events(Tb, st)["smile"]); sm[w][1] += N / 30 / 60
    out = {}
    for w in which:
        o = {}
        for k, rows in per[w].items():
            n = sum(r["n"] for r in rows); h = sum(r["hits"] for r in rows)
            ch = sum(r["chance_hits"] for r in rows) / max(1, sum(r["chance_n"] for r in rows))
            o[k] = dict(n=n, rate=h / n if n else float("nan"), chance=ch, excess=h / n - ch if n else float("nan"))
        o["smiles_per_min"] = sm[w][0] / sm[w][1]
        out[w] = o
    return out


def brief(o):
    s = ""
    for w in o:
        s += f"{w}: " + " ".join(f"{p.split('->')[0][:3]}>{p.split('->')[1][:3]} {o[w][p]['excess']:+.3f}(r{o[w][p]['rate']:.2f}/c{o[w][p]['chance']:.2f},n{o[w][p]['n']})" for p in PAIRS) + f" sm/min {o[w]['smiles_per_min']:.1f} | "
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--jobs", default=f"{T}/runs/p2rt/jobs_val.pkl"); ap.add_argument("--json")
    a = ap.parse_args()
    o = compute(a.npz, a.jobs)
    print(brief(o))
    if a.json:
        json.dump(o, open(a.json, "w"), indent=1)
