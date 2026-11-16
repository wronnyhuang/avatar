"""asset-hair — fast hair iteration: re-groom an already built asset without rebuilding skin/textures.

python hair_quick.py <mpfb_data_dir> <built_asset_dir> <out_asset_dir> [scale]
Copies the built asset (hard links), runs hair.build(base, S) and rebinds the roots to the asset's own head submesh
(lod0 + lod1), rewriting hair/*.bin and the hair entries of juno.asset.json.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from humans import asset_io  # noqa: E402
import juno_base, face_shapes, hair, build_juno  # noqa: E402,E401


class Shim:
    def __init__(self, A, name):
        s = [s for s in A.submeshes if s["name"] == name][0]
        self.name = name
        lo, n = s["first_index"], s["index_count"]
        I = A.mesh["index"][lo:lo + n].reshape(-1, 3).astype(np.int64)
        v0 = s["first_vertex"]
        if I.max() < s["vertex_count"] and v0 > 0:
            I = I + v0                       # local indices -> global
        self.P = A.mesh["position"].astype(np.float64)
        self.UV = A.mesh["uv0"].astype(np.float64)
        self.tris = I            # global vertex ids; tri index is local to the submesh
        self.v0 = v0


def main():
    data, src, out = sys.argv[1:4]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    t0 = time.time()
    if os.path.exists(out):
        shutil.rmtree(out)
    subprocess.check_call(["cp", "-al", src, out])
    shutil.rmtree(os.path.join(out, "hair"))
    os.makedirs(os.path.join(out, "hair"))
    base = juno_base.Base(data)
    S = face_shapes.build(base) if os.environ.get("HAIR_QUICK_SHAPES") else None   # ASSET3: hair.build ignores S
    t1 = time.time()
    raw = hair.build(base, S, scale=scale)
    t2 = time.time()
    parts = {}
    for lod in ("lod0", "lod1"):
        A = asset_io.load_asset(src, lod, hair=False, validate=False)
        parts[lod] = [Shim(A, "head")]
    groups = build_juno.write_hair(out, parts, raw)
    jp = os.path.join(out, "juno.asset.json")
    os.unlink(jp)                       # hard link -> write a fresh file
    meta = json.load(open(os.path.join(src, "juno.asset.json")))
    meta["hair_groups"] = groups
    for l in meta["lods"]:
        l["hair"] = {g: gi["lod"][l["name"]] for g, gi in groups.items()}
    meta["materials"] = {k: v for k, v in meta["materials"].items() if not k.startswith("hair_")}
    for g, gi in groups.items():
        meta["materials"]["hair_" + g] = {"type": "hair", "params": {k: gi[k] for k in gi if k in ("color_root", "color_tip", "melanin")}}
    json.dump(meta, open(jp, "w"), indent=1)
    for lod in ("lod0", "lod1"):
        asset_io.load_asset(out, lod)
    st = {g: (gi["lod_strands"]["lod0"], gi["lod_strands"]["lod1"], gi["points"], gi["root_dist_max_mm"]) for g, gi in groups.items()}
    mb = {lod: round(sum(os.path.getsize(os.path.join(out, "hair", f)) for f in os.listdir(os.path.join(out, "hair")) if f.startswith(lod)) / 1e6, 2)
          for lod in ("lod0", "lod1")}
    print(json.dumps({"groups": st, "MB": mb, "t_base": round(t1 - t0, 1), "t_hair": round(t2 - t1, 1), "t": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
