"""Extension member file for a training job (Sampler pack_extra): the enc_<name>* / a2f* members of a slim pack + the
encoder look-ahead, as a separate file placed in the pack dir (multi.py waits for <file>.ready).

    python -I tools/train/p2_ext.py --slim <slim.npz with the members> --out <enc_<name>.npz> --prefix enc_whisper_base \
        [--lookahead 1]
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True, help="enc_<name> or a2f")
    ap.add_argument("--lookahead", type=int, default=None)
    a = ap.parse_args()
    z = np.load(a.slim, allow_pickle=False)
    keep = {k: z[k] for k in z.files if k == a.prefix or k.startswith(a.prefix + "_")}
    assert keep, f"no {a.prefix}* members in {a.slim}"
    if a.lookahead is not None:
        keep[a.prefix + "_lookahead"] = np.asarray(a.lookahead, np.int64)
    np.savez(a.out, **keep)
    print(a.out, {k: (v.shape, str(v.dtype)) for k, v in keep.items()})


if __name__ == "__main__":
    main()
