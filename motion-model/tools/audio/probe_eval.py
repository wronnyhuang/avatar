"""Linear-probe layer pick: per layer of a probe-extraction dir, ridge from causal pooled features (train streams)
to face/head targets on held-out val streams. SELF = the person's own audio -> own face; CROSS = the partner's
audio -> this person's face/head (the listening-reaction use). Baseline = audio42 (the current hand-made features).
Targets are centred per stream (within-person dynamics; a person's resting brow height is not an audio thing).
Writes a JSON table + prints a summary."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

S = C.spec
G = {g: i for i, g in enumerate(S.GROUP_NAMES)}


def targets(m, k, a):
    B = lambda n: m[:, 135 + S.BS[n]]
    hc = S.GROUPS["head"]
    d = np.diff(m[:, hc], axis=0, prepend=m[:1, hc])
    hs = uniform_filter1d(np.linalg.norm(d, axis=1) * 30, 5)
    pc = [S.jcols("neck")[0], S.jcols("head")[0]]
    pv = uniform_filter1d(np.abs(np.diff(m[:, pc].sum(1), prepend=m[0, pc].sum())) * 30, 5)
    fu, mo, he = k[:, G["face_upper"]] > 0, k[:, G["mouth"]] > 0, k[:, G["head"]] > 0
    T = {
        "smile": (0.5 * (B("mouthSmileLeft") + B("mouthSmileRight")), mo),
        "browInnerUp": (B("browInnerUp"), fu),
        "browOuterUp": (0.5 * (B("browOuterUpLeft") + B("browOuterUpRight")), fu),
        "browDown": (0.5 * (B("browDownLeft") + B("browDownRight")), fu),
        "cheekSquint": (0.5 * (B("cheekSquintLeft") + B("cheekSquintRight")), fu),
        "eyeBlink": (0.5 * (B("eyeBlinkLeft") + B("eyeBlinkRight")), fu),
        "jawOpen": (B("jawOpen"), mo),
        "head_speed": (hs, he),
        "nod_speed": (pv, he),
        "vad": (a[:, 41], np.ones(len(a), bool)),
    }
    return T


SELF_T = ["smile", "browInnerUp", "browOuterUp", "browDown", "cheekSquint", "eyeBlink", "jawOpen", "head_speed",
          "nod_speed", "vad"]
CROSS_T = ["smile", "browInnerUp", "head_speed", "nod_speed", "vad"]


def pooled(X):
    return np.concatenate([uniform_filter1d(X, 6, axis=0, origin=2), uniform_filter1d(X, 30, axis=0, origin=14)], 1)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pca", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {r["stream_id"]: r for r in C.streams()}
    res = {}
    cache = {}
    for d in a.dirs + ["audio42"]:
        if d != "audio42":
            ids = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
        else:
            ids = sorted(set.intersection(*[set(f[:-4] for f in os.listdir(x) if f.endswith(".npz")) for x in a.dirs]))
        ids = [i for i in ids if rows[i]["partner_stream_id"] in set(ids)]
        if not ids:
            continue
        feats = {}
        for i in ids:
            if d == "audio42":
                feats[i] = {"a42": None}
            else:
                z = np.load(os.path.join(d, i + ".npz"))
                feats[i] = {k: z[k].astype(np.float32) for k in z.files}
            if i not in cache:
                cache[i] = targets(*C.canon_person(rows[i]))
        if d == "audio42":
            for i in ids:
                feats[i] = {"a42": C.canon_person(rows[i])[2]}
        layers = sorted(next(iter(feats.values())).keys())
        tr = [i for i in ids if rows[i]["split"] == "train"]
        va = [i for i in ids if rows[i]["split"] == "val"]
        name = os.path.basename(d.rstrip("/"))
        res[name] = {"n_train": len(tr), "n_val": len(va), "layers": {}}
        for l in layers:
            Xtr = np.concatenate([feats[i][l] for i in tr])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Z = (Xtr[::3] - mu) / sd
            k = min(a.pca, Z.shape[1])
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            V = Vt[:k].T
            P = {i: pooled(((feats[i][l] - mu) / sd) @ V) for i in ids}
            out = {}
            for kind, tl in (("self", SELF_T), ("cross", CROSS_T)):
                src = (lambda i: i) if kind == "self" else (lambda i: rows[i]["partner_stream_id"])
                for t in tl:
                    def XY(lst):
                        xs, ys = [], []
                        for i in lst:
                            y, m = cache[i][t]
                            if m.sum() < 30:
                                continue
                            xs.append(P[src(i)][m]); ys.append(y[m] - y[m].mean())    # within-person dynamics
                        return np.concatenate(xs), np.concatenate(ys)
                    X1, y1 = XY(tr); X2, y2 = XY(va)
                    xm, ym = X1.mean(0), y1.mean()
                    A = X1 - xm
                    w = np.linalg.solve(A.T @ A + a.alpha * len(A) / 1000 * np.eye(A.shape[1]), A.T @ (y1 - ym))
                    out[f"{kind}:{t}"] = round(float(r2(y2, (X2 - xm) @ w + ym)), 4)
            sm = np.mean([v for kk, v in out.items() if kk.startswith("self:") and kk != "self:vad"])
            cm = np.mean([v for kk, v in out.items() if kk.startswith("cross:") and kk != "cross:vad"])
            out["self_mean"] = round(float(sm), 4)
            out["cross_mean"] = round(float(cm), 4)
            res[name]["layers"][l] = out
            print(name, l, "self_mean", out["self_mean"], "cross_mean", out["cross_mean"],
                  {kk: v for kk, v in out.items() if "mean" not in kk}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
