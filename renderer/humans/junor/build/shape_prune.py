"""head-lod — drop blendshape entries that do nothing (|dpos| < 1 um AND |dnrm| < 5e-4) from a written lod.

parts.export stores every shape on its moved vertices plus their 1-ring (for the normal deltas); ~40 % of junor's
entries are numerically zero. Rewrites <lod>/shapes.bin and the shape table in <lod>/mesh.json (same format).
Env JUNOR_SHAPE_PRUNE=0 skips it.
"""
from __future__ import annotations

import json
import os

import numpy as np

DPOS_EPS = 1e-6
DNRM_EPS = 5e-4


def prune(lod_dir, log=print):
    mj = os.path.join(lod_dir, "mesh.json")
    with open(mj) as f:
        meta = json.load(f)
    with open(os.path.join(lod_dir, "shapes.bin"), "rb") as f:
        sb = f.read()
    out, so, n0, n1 = [], 0, 0, 0
    for i, e in enumerate(meta["shapes"]):
        K = e["count"]
        idx = np.frombuffer(sb, np.uint32, K, e["index_offset"])
        dp = np.frombuffer(sb, np.float32, 3 * K, e["dpos_offset"]).reshape(K, 3)
        dn = np.frombuffer(sb, np.float32, 3 * K, e["dnrm_offset"]).reshape(K, 3)
        keep = (np.linalg.norm(dp, axis=1) >= DPOS_EPS) | (np.linalg.norm(dn, axis=1) >= DNRM_EPS)
        n0 += K
        n1 += int(keep.sum())
        e = dict(e, count=int(keep.sum()))
        for key, arr in (("index_offset", idx[keep]), ("dpos_offset", dp[keep]), ("dnrm_offset", dn[keep])):
            e[key] = so
            b = np.ascontiguousarray(arr).tobytes()
            out.append(b)
            so += len(b)
        meta["shapes"][i] = e
    with open(os.path.join(lod_dir, "shapes.bin"), "wb") as f:
        f.write(b"".join(out))
    with open(mj, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"shape_prune {os.path.basename(lod_dir)}: entries {n0} -> {n1}, shapes.bin {len(sb) / 1e6:.1f} -> {so / 1e6:.1f} MB")
