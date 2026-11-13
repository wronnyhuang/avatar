"""Session-level random subsample of a compact pack (qpack) without re-encoding: every stream starts with an absolute
frame, so whole streams can be sliced out of the column planes and concatenated. Keeps ALL val/test streams, the train
sessions in random order until --train-frac of the train frames, dyad partners together (selection by session).

    python -I tools/train/subsample_pack.py <in.qpack.npz> <out.qpack.npz> --train-frac 0.42 [--seed 0] [--zstd]
"""
import argparse
import json
import os
import subprocess

import numpy as np

FRAME_KEYS = ("m_lo", "m_hi", "a_q", "mask")          # (C, N) column planes
VEC_KEYS = ("wexp",)                                   # (N,)
ROW_KEYS = ("style",)                                  # (N, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--train-frac", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zstd", action="store_true")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    st = z["streams"]
    sess = st[:, 6]
    rng = np.random.default_rng(a.seed)
    tr_sess = np.unique(sess[st[:, 3] == 0])
    rng.shuffle(tr_sess)
    tr_total = st[st[:, 3] == 0, 1].sum()
    per = {s: 0 for s in tr_sess}
    for i in np.nonzero(st[:, 3] == 0)[0]:
        per[sess[i]] += st[i, 1]
    keep_s, acc = set(), 0
    for s in tr_sess:
        if acc >= a.train_frac * tr_total:
            break
        keep_s.add(int(s)); acc += per[s]
    keep = [i for i in range(len(st)) if st[i, 3] != 0 or int(sess[i]) in keep_s]
    new_id = {old: k for k, old in enumerate(keep)}
    idx = np.concatenate([np.arange(st[i, 0], st[i, 0] + st[i, 1]) for i in keep])
    ns = st[keep].copy()
    ns[:, 0] = np.concatenate([[0], np.cumsum(ns[:-1, 1])])
    ns[:, 2] = [new_id.get(int(p), -1) if p >= 0 else -1 for p in st[keep, 2]]
    out = {}
    for k in z.files:
        if k in FRAME_KEYS:
            out[k] = np.ascontiguousarray(z[k][:, idx])
        elif k in VEC_KEYS:
            out[k] = z[k][idx]
        elif k in ROW_KEYS:
            out[k] = z[k][idx]
        elif k == "streams":
            out[k] = ns
        elif k == "quality":
            out[k] = z[k][keep]
        else:
            out[k] = z[k]
    names = json.loads(str(z["names"]))
    dsets = names.get("datasets", [])
    hours = {}
    for r in ns:
        if r[5]:
            continue
        key = f"{dsets[r[4]] if r[4] < len(dsets) else r[4]}/{['train', 'val', 'test'][r[3]]}"
        hours[key] = hours.get(key, 0) + r[1] / 30 / 3600
    names["hours"] = hours
    names["subsample"] = dict(source=os.path.basename(a.inp), train_frac=a.train_frac, seed=a.seed)
    out["names"] = np.array(json.dumps(names))
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.out)
    print(json.dumps(dict(out=a.out, frames=int(len(idx)), streams=len(keep), hours=hours,
                          bytes=os.path.getsize(a.out))), flush=True)
    if a.zstd:
        subprocess.run(["zstd", "-q", "-f", "-T32", "-19", "--long=27", a.out, "-o", a.out + ".zst"], check=True)
        print("zst", os.path.getsize(a.out + ".zst"), flush=True)


if __name__ == "__main__":
    main()
