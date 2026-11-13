"""Add fake phase-2 members to a qpack for CPU smoke tests (events, a2f, a2f_mask, enc_<name>).

    python -I tools/train/synth_v3.py --pack <qpack.npz> --out <v3.npz> [--enc fake_a:48,fake_b:32] [--int8]

events  = onsets of nods (band-passed head pitch peaks), blinks (blink crossing 0.5), smiles (smile crossing 0.4)
a2f     = the person's own face blendshapes + noise, a2f_mask = own vad
enc_<n> = a fixed random projection of the person's own 42-d audio + noise (D dims); --int8 writes the slim format
          (int8 + _scale/_qmean + a PCA basis) like tools/train/slim_pack.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from canon1 import spec  # noqa: E402
from unified.data import read_qpack  # noqa: E402


def _sm(x, n):
    return np.convolve(x, np.ones(n) / n, "same")


def fake_members(M, K, Au, streams, encs, int8=False, seed=0):
    rng = np.random.default_rng(seed)
    N = len(M)
    ev = np.zeros((N, 3), np.uint8)
    C = spec.COL
    bl = [135 + spec.BS["eyeBlinkLeft"], 135 + spec.BS["eyeBlinkRight"]]
    sm = [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]
    for s0, n, *_ in streams:
        m = M[s0:s0 + n].astype(np.float32)
        pitch = m[:, C["neck_x"]] + m[:, C["head_x"]]
        bp = _sm(pitch, 5) - _sm(pitch, 45)
        pk = np.nonzero((bp[1:-1] > bp[:-2]) & (bp[1:-1] >= bp[2:]) & (bp[1:-1] > 0.02))[0] + 1
        ev[s0 + np.maximum(pk - 4, 0), 0] = 1
        b = m[:, bl].mean(1)
        ev[s0 + 1 + np.nonzero((b[1:] > 0.5) & (b[:-1] <= 0.5))[0], 1] = 1
        s = _sm(m[:, sm].mean(1), 5)
        ev[s0 + 1 + np.nonzero((s[1:] > 0.4) & (s[:-1] <= 0.4))[0], 2] = 1
    out = dict(events=ev)
    a2f = np.clip(M[:, 135:187].astype(np.float32) + 0.05 * rng.standard_normal((N, 52)), 0, 1)
    out["a2f"] = a2f.astype(np.float16)
    out["a2f_mask"] = (Au[:, 41] > 0.5).astype(np.uint8)
    for name, D in encs:
        P = rng.standard_normal((42, D)).astype(np.float32) / np.sqrt(42)
        a = np.nan_to_num(Au.astype(np.float32))
        a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-3)
        f = a @ P + 0.1 * rng.standard_normal((N, D)).astype(np.float32)
        if not int8:
            out[f"enc_{name}"] = f.astype(np.float16)
            continue
        k = max(4, D // 4)
        mu = f.mean(0)
        _, _, Vt = np.linalg.svd(f[:: max(1, N // 20000)] - mu, full_matrices=False)
        V = Vt[:k]
        c = (f - mu) @ V.T
        sc = np.maximum(4 * c.std(0) / 127, 1e-6)
        out[f"enc_{name}"] = np.clip(np.round(c / sc), -127, 127).astype(np.int8)
        out[f"enc_{name}_scale"] = sc.astype(np.float32)
        out[f"enc_{name}_qmean"] = np.zeros(k, np.float32)
        out[f"enc_{name}_pca_mean"] = mu.astype(np.float32)
        out[f"enc_{name}_pca_V"] = V.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enc", default="fake_a:48")
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    z = np.load(a.pack, allow_pickle=False)
    M, K, Au, _ = read_qpack(z) if "qformat" in z.files else (z["motion"], z["mask"], z["audio"], None)
    encs = [(e.split(":")[0], int(e.split(":")[1])) for e in a.enc.split(",") if e]
    extra = fake_members(M, K, Au, z["streams"], encs, int8=a.int8)
    base = {k: z[k] for k in z.files}
    np.savez(a.out, **base, **extra)
    print(a.out, {k: (v.shape, str(v.dtype)) for k, v in extra.items()}, "onsets", extra["events"].sum(0))


if __name__ == "__main__":
    main()
