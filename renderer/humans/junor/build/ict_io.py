"""junor — ICT-FaceKit (MIT) reader: OBJ -> one cached npz (neutral, faces, uvs, 100 identity + 53 expression deltas).

usage: python ict_io.py <ICT-FaceKit dir> <out.npz>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def read_v(path):
    vs = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                vs.append(ln.split()[1:4])
    return np.asarray(vs, np.float64)


def read_full(path):
    V, VT, F, FT, M = [], [], [], [], []
    mat = ""
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                V.append(ln.split()[1:4])
            elif ln.startswith("vt "):
                VT.append(ln.split()[1:3])
            elif ln.startswith("usemtl"):
                mat = ln.split()[1]
            elif ln.startswith("f "):
                tok = ln.split()[1:]
                F.append([int(t.split("/")[0]) - 1 for t in tok])
                FT.append([int(t.split("/")[1]) - 1 for t in tok])
                M.append(mat)
    return np.asarray(V, np.float64), np.asarray(VT, np.float64), F, FT, M


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = os.path.join(src, "FaceXModel")
    V, VT, F, FT, M = read_full(os.path.join(d, "generic_neutral_mesh.obj"))
    names = json.load(open(os.path.join(d, "vertex_indices.json")))
    ex = list(names["expressions"])
    ex += sorted(n[:-4] for n in os.listdir(d) if n.endswith(".obj") and n[:-4] not in ex
                 and not n.startswith(("identity", "generic_neutral")))
    ids = sorted(n[:-4] for n in os.listdir(d) if n.startswith("identity"))
    E = np.stack([read_v(os.path.join(d, n + ".obj")) - V for n in ex]).astype(np.float32)
    I = np.stack([read_v(os.path.join(d, n + ".obj")) - V for n in ids]).astype(np.float32)
    # faces: pad to quads (-1 for tris)
    Fq = np.full((len(F), 4), -1, np.int64)
    FTq = np.full((len(F), 4), -1, np.int64)
    for i, (f, ft) in enumerate(zip(F, FT)):
        Fq[i, :len(f)] = f
        FTq[i, :len(ft)] = ft
    mats = sorted(set(M))
    Mi = np.array([mats.index(m) for m in M])
    np.savez_compressed(out, V=V, VT=VT, F=Fq, FT=FTq, M=Mi, mats=np.array(mats), expr=E, expr_names=np.array(ex),
                        ident=I, ident_names=np.array(ids))
    print(V.shape, VT.shape, Fq.shape, E.shape, I.shape, mats, V.min(0), V.max(0))


if __name__ == "__main__":
    main()
