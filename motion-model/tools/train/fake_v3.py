"""Test fixture: a qpack -> a FAKE pack v3 (the base members unchanged + events / a2f / a2f_mask / enc_<encoder>).

    python -I tools/train/fake_v3.py <in.qpack.npz> <out.npz> [--encoders whisper_tiny:384,hubert_base:768]
    python -I tools/train/fake_v3.py --synth <dir> <out.npz> [--sessions 6 --minutes 1]   (synth.py -> qpack -> v3)

events (N,3) u8 = single-frame {nod, blink, smile} onsets detected on the decoded motion (nod: band-passed head pitch
peaks > 0.035 rad >= 0.4 s apart; blink: eyeBlink crossing up through 0.5; smile: 5-frame smoothed smile crossing up
through 0.4, >= 1 s apart), a2f (N,52) f16 = motion cols 135..186, a2f_mask = VAD (audio col 41 > 0.5), enc_<name>
(N,D) f16 = a fixed random low-rank map of the stream's own normalized audio + noise. Only for tests: the real v3
members come from the data pipeline.
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from canon1 import spec  # noqa: E402
import slim_pack  # noqa: E402

FPS = spec.FPS
C = spec.COL


def _smooth(x, n):
    return np.convolve(x, np.ones(n) / n, "same") if n > 1 else x


def onsets_of(m):
    """m (T,189) -> (T,3) u8 onset frames {nod, blink, smile}."""
    T = len(m)
    out = np.zeros((T, 3), np.uint8)
    pitch = m[:, C["neck_x"]] + m[:, C["head_x"]]
    bp = _smooth(pitch, 5) - _smooth(pitch, 45)
    pk = np.nonzero((bp[1:-1] > bp[:-2]) & (bp[1:-1] >= bp[2:]) & (bp[1:-1] > 0.035))[0] + 1
    last = -99
    for p in pk:
        if p - last >= 12:
            out[p, 0] = 1; last = p
    bl = m[:, [135 + spec.BS["eyeBlinkLeft"], 135 + spec.BS["eyeBlinkRight"]]].mean(1)
    out[1:, 1] = (bl[1:] > 0.5) & (bl[:-1] <= 0.5)
    sm = _smooth(m[:, [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]].mean(1), 5)
    last = -99
    for p in np.nonzero((sm[1:] > 0.4) & (sm[:-1] <= 0.4))[0] + 1:
        if p - last >= FPS:
            out[p, 2] = 1; last = p
    return out


def add_v3_members(qpack, out, encoders=None, seed=0):
    from unified.data import read_qpack
    encoders = encoders if encoders is not None else {"whisper_tiny": 384}
    z = np.load(qpack, allow_pickle=False)
    M, K, A, _ = read_qpack(z)
    st = z["streams"]
    N = len(M)
    ev = np.zeros((N, 3), np.uint8)
    for s0, n, *_ in st:
        ev[s0:s0 + n] = onsets_of(M[s0:s0 + n].astype(np.float32))
    a2f = M[:, 135:187].astype(np.float16)
    a2f_mask = (A[:, 41].astype(np.float32) > 0.5).astype(np.uint8)
    an = (A.astype(np.float32) - z["amean"]) / z["astd"]
    an[:, 40] = np.clip(an[:, 40], -6, 6)
    rng = np.random.default_rng(seed)
    P = slim_pack.Pack(qpack)
    with zipfile.ZipFile(out + ".tmp.npz", "w", zipfile.ZIP_STORED, allowZip64=True) as zo:
        for k in P.files:
            P.copy_raw(k, zo)
        slim_pack.write_member(zo, "events", ev)
        slim_pack.write_member(zo, "a2f", a2f)
        slim_pack.write_member(zo, "a2f_mask", a2f_mask)
        for name, D in encoders.items():
            r = 12
            W1 = rng.standard_normal((42, r)).astype(np.float32) / np.sqrt(42)
            W2 = rng.standard_normal((r, D)).astype(np.float32)
            off = rng.standard_normal(D).astype(np.float32) * 3
            E = np.empty((N, D), np.float16)
            for r0 in range(0, N, 65536):
                h = np.tanh(an[r0:r0 + 65536] @ W1) @ W2
                E[r0:r0 + 65536] = (h + off + 0.1 * rng.standard_normal(h.shape).astype(np.float32)).astype(np.float16)
            slim_pack.write_member(zo, f"enc_{name}", E)
            del E
    os.replace(out + ".tmp.npz", out)
    return dict(out=out, frames=N, events_per_min=(ev.sum(0) / (N / FPS / 60)).round(2).tolist(),
                encoders=encoders)


def synth_qpack(d, sessions=6, minutes=1.0, seed=0, n_val=None):
    """synth.py sessions -> raw pack -> qpack in dir d; returns the qpack path. n_val: the last n_val sessions are
    val (default: synth's own split, or the last one when that has none)."""
    from unified.data import build_pack, load_index, write_qpack
    os.makedirs(d, exist_ok=True)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "synth.py"), "--out", os.path.join(d, "sess"), "--n",
                    str(sessions), "--minutes", str(minutes), "--seed", str(seed)], check=True, capture_output=True)
    recs = load_index([os.path.join(d, "sess", "index.jsonl")])
    if n_val:
        for r in recs[-n_val:]:
            r["split"] = "val"
    elif not any(r["split"] == "val" for r in recs):         # synth puts every 8th session in val
        recs[-1]["split"] = "val"
    raw, q = os.path.join(d, "raw.npz"), os.path.join(d, "qpack.npz")
    build_pack(recs, raw, log=lambda m: None, workers=4)
    write_qpack(raw, q, log=lambda m: None)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="a qpack, or (with --synth) a work dir")
    ap.add_argument("out")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--encoders", default="whisper_tiny:384")
    a = ap.parse_args()
    enc = {e.split(":")[0]: int(e.split(":")[1]) for e in a.encoders.split(",") if e}
    q = synth_qpack(a.inp, a.sessions, a.minutes) if a.synth else a.inp
    print(json.dumps(add_v3_members(q, a.out, enc)))


if __name__ == "__main__":
    main()
