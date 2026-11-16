"""asset — MakeHuman (MPFB2 data, CC0) base mesh tools: OBJ parse, targets, joint cubes, rig json, weights.

Pure numpy. All coordinates returned in ASSET space (metres, +y up, +z forward, +x = character's left) unless named *_b
(Blender space: +z up, -y forward). MH obj units are decimetres with +y up / +z forward already.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np

# asset <- blender:  (x, y, z)_a = (x_b, z_b, -y_b)
C_AB = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
C_BA = C_AB.T


class MHObj:
    def __init__(self, path):
        vs, vts, faces, fgroup = [], [], [], []
        group = None
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    vs.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("vt "):
                    vts.append([float(x) for x in line.split()[1:3]])
                elif line.startswith("g "):
                    group = line.split(None, 1)[1].strip()
                elif line.startswith("f "):
                    fv = []
                    for tok in line.split()[1:]:
                        p = tok.split("/")
                        fv.append((int(p[0]) - 1, int(p[1]) - 1 if len(p) > 1 and p[1] else -1))
                    faces.append(fv)
                    fgroup.append(group)
        self.v = np.array(vs, np.float64)          # decimetres, MH space
        self.vt = np.array(vts, np.float64)
        self.faces = faces
        self.fgroup = fgroup
        self.groups = {}
        for i, g in enumerate(fgroup):
            self.groups.setdefault(g, []).append(i)

    def group_verts(self, g):
        s = set()
        for fi in self.groups.get(g, []):
            for vi, _ in self.faces[fi]:
                s.add(vi)
        return np.array(sorted(s), np.int64)


def read_target(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        rows = [l.split() for l in f if l.strip() and not l.startswith("#")]
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    a = np.array(rows, np.float64)
    return a[:, 0].astype(np.int64), a[:, 1:4]


def target_dense(path, n):
    idx, d = read_target(path)
    out = np.zeros((n, 3))
    out[idx] = d
    return out


def vec_roll_to_mat3(vec, roll):
    """Blender's bone matrix from a direction + roll (Blender space), returned as a standard 3x3 (columns = axes)."""
    nor = vec / np.linalg.norm(vec)
    x, y, z = nor
    theta = 1.0 + y
    if theta > 1e-5:
        M = np.array([[1 - x * x / theta, x, -x * z / theta],
                      [-x, y, -z],
                      [-x * z / theta, z, 1 - z * z / theta]])
    else:
        M = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    c, s = np.cos(roll), np.sin(roll)
    ax = nor
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) * c + s * K + (1 - c) * np.outer(ax, ax)
    return R @ M


def bone_matrix_asset(head_a, tail_a, roll):
    """Rest world matrix in asset space for a bone given head/tail in asset space and a Blender roll."""
    hb, tb = C_BA @ head_a, C_BA @ tail_a
    Mb = vec_roll_to_mat3(tb - hb, roll)
    M = np.eye(4)
    M[:3, :3] = C_AB @ Mb
    M[:3, 3] = head_a
    return M
