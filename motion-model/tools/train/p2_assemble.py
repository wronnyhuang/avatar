"""Assemble the FULL v3 pack: a v3 qpack (corrected faces + events) + per-stream encoder features
($P/feats/<enc>/<stream_id>.npy) + Audio2Face teacher ($P/a2f/<stream_id>.npy), aligned by the stream list.

    python -I tools/train/p2_assemble.py --pack $P/packs/abl_v3.qpack.npz --streams $P/streams_abl.jsonl \
        --out <abl_v3plus.npz> [--encoders whisper_tiny,hubert_base] [--a2f] [--only-ext]

Members added: enc_<name> f16 (N,D) (rows of stream s = feats file of s; 0 rows for streams without a file = NaN ->
the Sampler treats them as 0), a2f f16 (N,52) + a2f_mask u8 (N,) (the a2f_mask member of the a2f META if given as
<stream_id>_mask.npy, else frames where the person's own vad > 0.5). Checks: every feature file has exactly the
stream's n_frames rows; the stream order of the pack = the stream list's pack_stream.
--only-ext writes ONLY the added members (an extension file for Sampler pack_extra / evaluate).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from canon1 import paths as _paths  # noqa: E402

P = str(_paths.P2)          # $MM_WORK_DIR/p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", default="")
    ap.add_argument("--a2f", action="store_true")
    ap.add_argument("--only-ext", action="store_true")
    ap.add_argument("--feats-dir", default=os.path.join(P, "feats"))
    ap.add_argument("--a2f-dir", default=os.path.join(P, "a2f"))
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.pack, allow_pickle=False)
    st = z["streams"]
    N = int(st[:, 0].max() + st[np.argmax(st[:, 0]), 1])
    rows = [json.loads(x) for x in open(a.streams) if x.strip()]
    assert len(rows) == len(st), (len(rows), len(st))
    for r in rows:
        assert st[r["pack_stream"], 1] == r["n_frames"], r["stream_id"]
    extra = {}
    for enc in [e for e in a.encoders.split(",") if e]:
        d = os.path.join(a.feats_dir, enc)
        X = None
        miss = 0
        for r in rows:
            f = os.path.join(d, r["stream_id"] + ".npy")
            s0, n = st[r["pack_stream"], 0], st[r["pack_stream"], 1]
            if not os.path.exists(f):
                miss += 1
                continue
            v = np.load(f)
            assert v.shape[0] == n, (f, v.shape, n)
            if X is None:
                X = np.full((N, v.shape[1]), np.nan, np.float16)
            X[s0:s0 + n] = v.astype(np.float16)
        assert X is not None, f"no feature files in {d}"
        print(f"enc_{enc}: {X.shape}, {miss} streams missing ({miss / len(rows):.1%})", flush=True)
        extra[f"enc_{enc}"] = X
    if a.a2f:
        A = np.full((N, 52), np.nan, np.float16)
        Mk = np.zeros(N, np.uint8)
        miss = 0
        from unified.data import read_qpack
        vad = None
        for r in rows:
            f = os.path.join(a.a2f_dir, r["stream_id"] + ".npy")
            s0, n = st[r["pack_stream"], 0], st[r["pack_stream"], 1]
            if not os.path.exists(f):
                miss += 1
                continue
            v = np.load(f)
            assert v.shape == (n, 52), (f, v.shape, n)
            A[s0:s0 + n] = v.astype(np.float16)
            fm = os.path.join(a.a2f_dir, r["stream_id"] + "_mask.npy")
            if os.path.exists(fm):
                Mk[s0:s0 + n] = np.load(fm).reshape(-1)[:n] > 0
            else:
                if vad is None:
                    vad = read_qpack(z)[2][:, 41].astype(np.float32)
                Mk[s0:s0 + n] = vad[s0:s0 + n] > 0.5
        A = np.nan_to_num(A, nan=0.0)
        Mk[np.isnan(A).any(1)] = 0
        print(f"a2f: {miss} streams missing ({miss / len(rows):.1%}); speaking frames {Mk.mean():.3f}", flush=True)
        extra["a2f"] = A
        extra["a2f_mask"] = Mk
    for k, v in list(extra.items()):
        if k.startswith("enc_"):
            bad = np.isnan(v).any(1)
            v[bad] = 0
    out = dict(extra) if a.only_ext else dict({k: z[k] for k in z.files}, **extra)
    np.savez(a.out, **out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 2**20:.0f} MB, {time.time() - t0:.0f}s): "
          f"{sorted(extra)}", flush=True)


if __name__ == "__main__":
    main()
