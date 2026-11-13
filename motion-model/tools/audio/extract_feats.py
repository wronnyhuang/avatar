"""phase-2 plan item 2: frozen speech-encoder features for every stream of a stream list ->
$P/feats/<enc>/<stream_id>.npy float16 (n_frames, D) after the train-set PCA (whitened), + pca.npz + META.json.

  fit-pca:  python -I extract_feats.py fit-pca --enc whisper_tiny --layer 2 --probe-dir <probe dir with train npz>
  run:      python -I extract_feats.py run --enc whisper_tiny --mode sliding --layer 2 --shard 0/5 --threads 4
  meta:     python -I extract_feats.py meta --enc whisper_tiny ...   (writes META.json; run after the bench)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

FEATS = os.path.join(C.P, "feats")


def fit_pca(a):
    rows = {r["stream_id"]: r for r in C.streams(a.streams)}
    ids = sorted(f[:-4] for f in os.listdir(a.probe_dir) if f.endswith(".npz") and not f.endswith("tmp.npz"))
    ids = [i for i in ids if i in rows and rows[i]["split"] == "train"]
    X = np.concatenate([np.load(os.path.join(a.probe_dir, i + ".npz"))[f"l{a.layer:02d}"].astype(np.float64)
                        for i in ids])
    mean = X.mean(0)
    Z = X - mean
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    comps = Vt[: a.dims]
    var = s ** 2 / len(Z)
    scale = np.sqrt(var[: a.dims])
    od = os.path.join(FEATS, a.enc)
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "pca.npz"), mean=mean.astype(np.float32), comps=comps.astype(np.float32),
             scale=scale.astype(np.float32), explained=np.float32(var[: a.dims].sum() / var.sum()),
             layer=np.int32(a.layer), n_frames=np.int64(len(X)), n_streams=np.int64(len(ids)))
    print("pca", a.enc, "layer", a.layer, "dims", a.dims, "explained", round(float(var[: a.dims].sum() / var.sum()), 4),
          "from", len(ids), "train streams", len(X), "frames")


def run(a):
    import torch
    from unified import speech_enc as E
    torch.set_num_threads(a.threads)
    od = os.path.join(FEATS, a.enc)
    proj = E.Projection(os.path.join(od, "pca.npz"))
    rows = C.streams(a.streams)
    if a.order == "random":                     # val first, then train in a fixed random order: any prefix is a
        val = [r for r in rows if r["split"] == "val"]   # uniform subset (same order for every encoder)
        tr = [r for r in rows if r["split"] != "val"]
        rows = val + [tr[i] for i in np.random.default_rng(4330).permutation(len(tr))]
    k, n = [int(x) for x in a.shard.split("/")]
    rows = [r for i, r in enumerate(rows) if i % n == k]
    if a.order == "split":                      # val first so the held-out set is complete early
        rows.sort(key=lambda r: r["split"] != "val")
    m = E.WindowEncoder(a.enc, max_layer=a.layer)
    t0 = time.time()
    done = 0
    for r in rows:
        fn = os.path.join(od, r["stream_id"] + ".npy")
        if os.path.exists(fn):
            continue
        if a.mode == "kv":
            x16, base = C.stream_audio(r, pre_s=3.0, post_s=0.5)
            o = E.kv_extract_stream(m, x16, r["n_frames"], r["frame0"], [a.layer], base)
        else:
            x16, base = C.stream_audio(r, pre_s=a.window + 0.5, post_s=a.lookahead / 30 + 0.5)
            o = E.extract_stream(m, x16, r["n_frames"], r["frame0"], a.window, a.lookahead, [a.layer],
                                 batch=a.batch, base=base)
        z = proj(o[a.layer]).astype(np.float16)
        assert z.shape == (r["n_frames"], proj.comps.shape[0]) and np.isfinite(z).all()
        tmp = fn[:-4] + f".tmp{os.getpid()}.npy"
        np.save(tmp, z)
        os.replace(tmp, fn)
        done += 1
        if done % 10 == 0 or done == 1:
            el = time.time() - t0
            print(json.dumps(dict(enc=a.enc, shard=a.shard, done=done, of=len(rows), s=round(el, 1),
                                  s_per_stream=round(el / done, 2), load=round(os.getloadavg()[0], 1))), flush=True)
    print("DONE", a.enc, a.shard, done, round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit-pca", "run"])
    ap.add_argument("--enc", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--probe-dir")
    ap.add_argument("--streams", default=None)
    ap.add_argument("--mode", default="sliding", choices=["sliding", "kv"])
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--order", default="split")
    a = ap.parse_args()
    fit_pca(a) if a.cmd == "fit-pca" else run(a)


if __name__ == "__main__":
    main()
