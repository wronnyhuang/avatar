"""Per-frame STYLE labels for an existing pack -> <pack minus .npz>.style.npz (u8 (N,3), 255 = unknown + tables).

    python -I tools/train/style_sidecar.py <pack.npz> [--tables $MM_DATA_DIR/style/tables.json] [--procs 16]
Definitions = canon1/style.py. The Sampler picks the sidecar up by name; when it is absent it
computes the same thing on the fly. Labels are piecewise constant (20-s segments), so the file zstd's to almost nothing.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from unified.data import STYLE_NAMES, pack_style, read_qpack, style_sidecar_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.pack, allow_pickle=False)
    if "qformat" in z.files:
        M, K, _, _ = read_qpack(z)
    else:
        M, K = z["motion"], z["mask"]
    st = z["streams"]
    print(f"decoded {len(M)} frames in {time.time() - t0:.0f}s", flush=True)
    tab = None
    if a.tables:
        with open(a.tables) as f:
            tab = json.load(f)
    S, tab = pack_style(M, K, st, tables=tab, procs=a.procs)
    out = style_sidecar_path(a.pack)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, style=S, tables=np.array(json.dumps(tab)))
    os.replace(tmp, out)
    tr = np.concatenate([np.arange(s0, s0 + n) for s0, n, _, sp, _, gh, _ in st if sp == 0 and gh == 0])
    q = {n: np.percentile(S[tr, j][S[tr, j] < 255] / 254, [10, 50, 90]).round(3).tolist()
         for j, n in enumerate(STYLE_NAMES)}
    print(json.dumps(dict(out=out, bytes=os.path.getsize(out), train_p10_50_90=q, secs=round(time.time() - t0))))


if __name__ == "__main__":
    main()
