"""asset — reference loader / validator / writer for the juno.asset/1 format (docs/CONTRACT--asset.md).

Pure numpy. RENDER and RIG import `load_asset`; the build scripts use `write_*`.

    a = load_asset("renderer/humans/juno/juno.asset", lod="lod0")
    a.mesh["position"]  # (N,3) f4        a.submeshes -> list of dicts
    a.shapes            # list of (name, idx u4[K], dpos f4[K,3], dnrm f4[K,3])  in juno.asset.json order
    a.bones             # list of dicts; a.rest (nb,4,4), a.inv_bind (nb,4,4), a.parents (nb,) i4
    a.hair[group]       # dict pos (S,P,3), root_uv, width, seed, clump, [tri, bary]
    a.dense_shapes()    # (nshapes, N, 3) float32 position deltas (for GPU textures / tests)
    a.skin(bones_world, weights) -> (N,3) skinned positions (CPU reference skinning, used by tests)

CLI:  python -m renderer.humans.asset_io <asset_dir> [lod]   -> validates and prints stats.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

FORMAT = "juno.asset/1"

REQUIRED_BONES = (
    ["root", "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head", "jaw", "eye_l", "eye_r"]
    + [f"{b}_{s}" for s in ("l", "r") for b in ("clavicle", "upperarm", "lowerarm", "hand")]
    + [f"{f}_0{i}_{s}" for s in ("l", "r") for f in ("thumb", "index", "middle", "ring", "pinky") for i in (1, 2, 3)]
    + [f"{b}_{s}" for s in ("l", "r") for b in ("thigh", "calf", "foot", "ball")]
)

ARKIT52 = (
    "eyeBlinkLeft eyeLookDownLeft eyeLookInLeft eyeLookOutLeft eyeLookUpLeft eyeSquintLeft eyeWideLeft "
    "eyeBlinkRight eyeLookDownRight eyeLookInRight eyeLookOutRight eyeLookUpRight eyeSquintRight eyeWideRight "
    "jawForward jawLeft jawRight jawOpen mouthClose mouthFunnel mouthPucker mouthLeft mouthRight "
    "mouthSmileLeft mouthSmileRight mouthFrownLeft mouthFrownRight mouthDimpleLeft mouthDimpleRight "
    "mouthStretchLeft mouthStretchRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper "
    "mouthPressLeft mouthPressRight mouthLowerDownLeft mouthLowerDownRight mouthUpperUpLeft mouthUpperUpRight "
    "browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight "
    "noseSneerLeft noseSneerRight tongueOut"
).split()
EXTRAS = ("labioDental lipTuckLower lipTuckUpper tongueTipUp tongueDental tongueBack lipsTogether "
          "eyeHappyLeft eyeHappyRight browTiltLeft browTiltRight").split()
REQUIRED_SHAPES = ARKIT52 + EXTRAS
assert len(ARKIT52) == 52

_DT = {"f4": np.float32, "u2": np.uint16, "u4": np.uint32, "i4": np.int32}


def m16(m) -> np.ndarray:
    return np.asarray(m, dtype=np.float32).reshape(4, 4)


class Asset:
    pass


def _read(buf: bytes, off: int, dtype, count: int, comps: int):
    a = np.frombuffer(buf, dtype=np.dtype(dtype).newbyteorder("<"), count=count * comps, offset=off)
    return a.reshape(count, comps) if comps > 1 else a


def load_hair(path: str) -> dict:
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"JHR1":
        raise ValueError(f"{path}: bad magic")
    S, P, flags = struct.unpack_from("<III", b, 4)
    o = 16
    out = {}

    def take(name, dt, shape):
        nonlocal o
        n = int(np.prod(shape))
        out[name] = np.frombuffer(b, dtype=np.dtype(dt).newbyteorder("<"), count=n, offset=o).reshape(shape)
        o += n * np.dtype(dt).itemsize

    take("pos", np.float32, (S, P, 3))
    take("root_uv", np.float32, (S, 2))
    take("width", np.float32, (S, 2))
    take("seed", np.float32, (S,))
    take("clump", np.uint32, (S,))
    if flags & 1:
        take("tri", np.uint32, (S,))
        take("bary", np.float32, (S, 3))
    if o != len(b):
        raise ValueError(f"{path}: size {len(b)} != expected {o}")
    out["S"], out["P"], out["flags"] = S, P, flags
    return out


def write_hair(path: str, pos, root_uv, width, seed, clump, tri=None, bary=None) -> None:
    pos = np.ascontiguousarray(pos, np.float32)
    S, P, _ = pos.shape
    flags = 1 if tri is not None else 0
    with open(path, "wb") as f:
        f.write(b"JHR1" + struct.pack("<III", S, P, flags))
        for arr, dt in ((pos, np.float32), (root_uv, np.float32), (width, np.float32), (seed, np.float32),
                        (clump, np.uint32)):
            f.write(np.ascontiguousarray(arr, dt).astype(np.dtype(dt).newbyteorder("<")).tobytes())
        if flags:
            f.write(np.ascontiguousarray(tri, np.uint32).tobytes())
            f.write(np.ascontiguousarray(bary, np.float32).tobytes())


def write_mesh(lod_dir: str, arrays: dict, submeshes: list, shapes: list) -> None:
    """arrays: position (N,3) normal (N,3) tangent (N,4) uv0 (N,2) joints (N,4) u2 weights (N,4) index (M,) u4.
    shapes: list of (name, idx, dpos, dnrm[, extra dict])."""
    os.makedirs(lod_dir, exist_ok=True)
    order = [("position", "f4", 3), ("normal", "f4", 3), ("tangent", "f4", 4), ("uv0", "f4", 2),
             ("joints", "u2", 4), ("weights", "f4", 4), ("index", "u4", 1)]
    buffers, off, chunks = {}, 0, []
    N = len(arrays["position"])
    for name, dt, comps in order:
        a = np.ascontiguousarray(arrays[name], _DT[dt]).reshape(-1)
        bts = a.tobytes()
        buffers[name] = {"offset": off, "dtype": dt, "comps": comps}
        chunks.append(bts)
        off += len(bts)
        pad = (-off) % 16
        chunks.append(b"\0" * pad)
        off += pad
    with open(os.path.join(lod_dir, "mesh.bin"), "wb") as f:
        f.write(b"".join(chunks))
    sh_meta, sh_chunks, so = [], [], 0
    for s in shapes:
        name, idx, dpos, dnrm = s[:4]
        extra = s[4] if len(s) > 4 else {}
        idx = np.ascontiguousarray(idx, np.uint32)
        order_ = np.argsort(idx, kind="stable")
        idx = idx[order_]
        dpos = np.ascontiguousarray(np.asarray(dpos, np.float32)[order_])
        dnrm = np.ascontiguousarray(np.asarray(dnrm, np.float32)[order_])
        e = {"name": name, "count": int(len(idx))}
        for key, arr in (("index_offset", idx), ("dpos_offset", dpos), ("dnrm_offset", dnrm)):
            e[key] = so
            bts = arr.tobytes()
            sh_chunks.append(bts)
            so += len(bts)
        e.update(extra)
        sh_meta.append(e)
    with open(os.path.join(lod_dir, "shapes.bin"), "wb") as f:
        f.write(b"".join(sh_chunks))
    meta = {"vertex_count": int(N), "index_count": int(len(arrays["index"])), "buffers": buffers,
            "submeshes": submeshes, "shapes": sh_meta}
    with open(os.path.join(lod_dir, "mesh.json"), "w") as f:
        json.dump(meta, f, indent=1)


def load_asset(asset_dir: str, lod: str = "lod0", hair: bool = True, validate: bool = True) -> Asset:
    A = Asset()
    A.dir = asset_dir
    with open(os.path.join(asset_dir, "juno.asset.json")) as f:
        A.meta = json.load(f)
    if A.meta.get("format") != FORMAT:
        raise ValueError(f"format {A.meta.get('format')!r} != {FORMAT}")
    lods = {l["name"]: l for l in A.meta["lods"]}
    if lod not in lods:
        raise KeyError(f"lod {lod} not in {list(lods)}")
    A.lod = lods[lod]
    with open(os.path.join(asset_dir, A.meta["skeleton"])) as f:
        sk = json.load(f)
    A.skeleton = sk
    A.bones = sk["bones"]
    A.bone_index = {b["name"]: i for i, b in enumerate(A.bones)}
    A.parents = np.array([b["parent"] for b in A.bones], np.int32)
    A.rest = np.stack([m16(b["rest"]) for b in A.bones])
    A.inv_bind = np.stack([m16(b["inverse_bind"]) for b in A.bones])
    mpath = os.path.join(asset_dir, A.lod["mesh"])
    ldir = os.path.dirname(mpath)
    with open(mpath) as f:
        A.mesh_meta = json.load(f)
    with open(os.path.join(ldir, "mesh.bin"), "rb") as f:
        mb = f.read()
    N, M = A.mesh_meta["vertex_count"], A.mesh_meta["index_count"]
    A.mesh = {}
    for name, d in A.mesh_meta["buffers"].items():
        cnt = M if name == "index" else N
        A.mesh[name] = _read(mb, d["offset"], _DT[d["dtype"]], cnt, d["comps"])
    A.submeshes = A.mesh_meta["submeshes"]
    with open(os.path.join(ldir, "shapes.bin"), "rb") as f:
        sb = f.read()
    A.shapes = []
    for e in A.mesh_meta["shapes"]:
        K = e["count"]
        A.shapes.append((e["name"], _read(sb, e["index_offset"], np.uint32, K, 1),
                         _read(sb, e["dpos_offset"], np.float32, K, 3), _read(sb, e["dnrm_offset"], np.float32, K, 3)))
    A.shape_names = [s[0] for s in A.shapes]
    A.hair = {}
    if hair:
        for g, p in (A.lod.get("hair") or {}).items():
            A.hair[g] = load_hair(os.path.join(asset_dir, p))
    if validate:
        problems = validate_asset(A)
        if problems:
            raise ValueError("asset invalid:\n  " + "\n  ".join(problems))
    A.dense_shapes = lambda: _dense(A)
    A.skin = lambda bw, w=None: _skin(A, bw, w)
    return A


def _dense(A):
    N = A.mesh_meta["vertex_count"]
    D = np.zeros((len(A.shapes), N, 3), np.float32)
    for i, (_, idx, dp, _) in enumerate(A.shapes):
        D[i, idx] = dp
    return D


def _skin(A, bones_world, weights=None):
    p = A.mesh["position"].astype(np.float32).copy()
    if weights is not None:
        for wv, (_, idx, dp, _) in zip(weights, A.shapes):
            if wv:
                p[idx] += wv * dp
    K = np.einsum("bij,bjk->bik", np.asarray(bones_world, np.float32), A.inv_bind)
    ph = np.concatenate([p, np.ones((len(p), 1), np.float32)], 1)
    out = np.zeros_like(p)
    J, W = A.mesh["joints"].astype(np.int64), A.mesh["weights"]
    for k in range(4):
        out += W[:, k:k + 1] * np.einsum("nij,nj->ni", K[J[:, k]], ph)[:, :3]
    return out


def validate_asset(A) -> list:
    pr = []
    names = [b["name"] for b in A.bones]
    for b in REQUIRED_BONES:
        if b not in names:
            pr.append(f"missing bone {b}")
    for i, b in enumerate(A.bones):
        if b["parent"] >= i:
            pr.append(f"bone {b['name']} parent {b['parent']} not before it")
    err = np.abs(np.einsum("bij,bjk->bik", A.rest, A.inv_bind) - np.eye(4)).max() if len(A.bones) else 0
    if err > 1e-3:
        pr.append(f"rest @ inverse_bind != I (max err {err:.2e})")
    want = A.meta.get("shapes", [])
    if A.shape_names != want:
        pr.append("mesh shapes order != juno.asset.json shapes")
    for s in REQUIRED_SHAPES:
        if s not in A.shape_names:
            pr.append(f"missing shape {s}")
    N = A.mesh_meta["vertex_count"]
    idx = A.mesh["index"]
    if len(idx) % 3:
        pr.append("index count not a multiple of 3")
    if len(idx) and idx.max() >= N:
        pr.append("index out of range")
    W = A.mesh["weights"]
    if np.abs(W.sum(1) - 1).max() > 1e-3:
        pr.append(f"weights do not sum to 1 (max err {np.abs(W.sum(1) - 1).max():.2e})")
    if A.mesh["joints"].max() >= len(A.bones):
        pr.append("joint index out of range")
    for k in ("position", "normal", "uv0"):
        if not np.isfinite(A.mesh[k]).all():
            pr.append(f"non-finite {k}")
    mats = A.meta.get("materials", {})
    for s in A.submeshes:
        if s["material"] not in mats:
            pr.append(f"submesh {s['name']} material {s['material']} not in materials")
        sl = idx[s["first_index"]:s["first_index"] + s["index_count"]]
        if len(sl) and (sl.min() < s["first_vertex"] or sl.max() >= s["first_vertex"] + s["vertex_count"]):
            pr.append(f"submesh {s['name']} indices outside its vertex range")
    for _, ix, dp, dn in A.shapes:
        if len(ix) and ix.max() >= N:
            pr.append("shape index out of range")
    for m, d in mats.items():
        for key in ("albedo", "normal", "packed"):
            if key in d:
                for suf in {"", A.lod.get("tex_suffix", "")}:
                    p = os.path.join(A.dir, d[key].replace(".png", f"{suf}.png") if ".png" in d[key]   # head-lod: .jpg/.webp
                                     else "{}{}{}".format(os.path.splitext(d[key])[0], suf, os.path.splitext(d[key])[1]))
                    if not os.path.exists(p):
                        pr.append(f"material {m} {key} missing file {p}")
    for g, h in A.hair.items():
        if not np.isfinite(h["pos"]).all():
            pr.append(f"hair {g} non-finite")
    return pr


def stats(A) -> dict:
    tri = {s["name"]: s["index_count"] // 3 for s in A.submeshes}
    return {"vertices": A.mesh_meta["vertex_count"], "tris": sum(tri.values()), "tris_by_submesh": tri,
            "bones": len(A.bones), "shapes": len(A.shapes),
            "strands": {g: int(h["S"]) for g, h in A.hair.items()},
            "bounds": [A.mesh["position"].min(0).round(3).tolist(), A.mesh["position"].max(0).round(3).tolist()]}


if __name__ == "__main__":
    d = sys.argv[1]
    lod = sys.argv[2] if len(sys.argv) > 2 else "lod0"
    a = load_asset(d, lod)
    print(json.dumps(stats(a), indent=1))
    print("OK")
