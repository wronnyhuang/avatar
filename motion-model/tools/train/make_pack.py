"""Build a training pack from CANON v1 sessions.

    python -I tools/train/make_pack.py --index <dir-or-index.jsonl> [...] --out pack.npz [--splits splits.json]
           [--max-sessions N] [--zstd]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from unified.data import build_pack, load_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None, help="splits.json: overrides each session's split by person id")
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--zstd", action="store_true")
    ap.add_argument("--datasets", default=None, help="comma list to keep")
    ap.add_argument("--frac", default=None, help='json {"train":0.3,"val":1.0,"test":0}: per-session fraction kept')
    ap.add_argument("--qpack", default=None, help="also write the compact quantized pack here (+ .zst)")
    ap.add_argument("--finger-pca", type=int, default=12)
    a = ap.parse_args()
    recs = load_index(a.index)
    if a.datasets:
        keep = set(a.datasets.split(","))
        recs = [r for r in recs if r.get("dataset") in keep]
    t0 = time.time()
    frac = json.loads(a.frac) if a.frac else None
    names = build_pack(recs, a.out, max_sessions=a.max_sessions, frac=frac)
    print(json.dumps(names["hours"]), f"{time.time() - t0:.1f}s", os.path.getsize(a.out) / 1e6, "MB")
    if a.zstd:
        subprocess.run(["zstd", "-q", "-f", "-T16", "-12", a.out, "-o", a.out + ".zst"], check=True)
        print("zst", os.path.getsize(a.out + ".zst") / 1e6, "MB")
    if a.qpack:
        from unified.data import write_qpack
        write_qpack(a.out, a.qpack, finger_pca=a.finger_pca)
        subprocess.run(["zstd", "-q", "-f", "-T16", "-19", "--long=27", a.qpack, "-o", a.qpack + ".zst"], check=True)
        print("qpack", os.path.getsize(a.qpack) / 1e6, "MB; zst", os.path.getsize(a.qpack + ".zst") / 1e6, "MB")


if __name__ == "__main__":
    main()
