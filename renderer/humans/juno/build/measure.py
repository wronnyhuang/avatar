"""asset — measure Juno's craft ratios (T/ref/SPEC.md s1) on the Base, from a front orthographic raster.

python measure.py <mpfb_data> [--png out.png]   -> prints a json of ratios
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import juno_base  # noqa: E402

RES = 0.00025  # m per pixel


def raster_front(V, tris, ids, box):
    """front ortho (+z toward the viewer). returns id image (-1 empty) and depth image."""
    x0, x1, y0, y1 = box
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
    Z = np.full((H, W), -np.inf)
    I = np.full((H, W), -1, np.int32)
    P = np.stack([(V[:, 0] - x0) / RES, (y1 - V[:, 1]) / RES, V[:, 2]], 1)
    for t, idv in zip(tris, ids):
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        mnx, mxx = int(max(0, np.floor(min(a[0], b[0], c[0])))), int(min(W - 1, np.ceil(max(a[0], b[0], c[0]))))
        mny, mxy = int(max(0, np.floor(min(a[1], b[1], c[1])))), int(min(H - 1, np.ceil(max(a[1], b[1], c[1]))))
        if mnx > mxx or mny > mxy:
            continue
        xs, ys = np.meshgrid(np.arange(mnx, mxx + 1) + 0.5, np.arange(mny, mxy + 1) + 0.5)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        l3 = 1 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any():
            continue
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        sub = Z[mny:mxy + 1, mnx:mxx + 1]
        upd = m & (z > sub)
        sub[upd] = z[upd]
        I[mny:mxy + 1, mnx:mxx + 1][upd] = idv
    return I, Z


def measure(base, png=None):
    V = base.VA
    lm = base.lm
    hv = np.where(base.skin_mask & (V[:, 1] > lm["neck"][1] - 0.02))[0]
    hset = np.zeros(base.nV, bool)
    hset[hv] = True
    faces = []
    for fi in base.faces_of("body"):
        f = [v for v, _ in base.obj.faces[fi]]
        if hset[f].all():
            faces += juno_base.tri_of(f)
    ids = [0] * len(faces)
    for k, side in enumerate(("l", "r")):
        for fi in base.faces_of(f"helper-{side}-eye"):
            f = [v for v, _ in base.obj.faces[fi]]
            for t in juno_base.tri_of(f):
                faces.append(t)
                ids.append(1 + k)
    faces = np.array(faces)
    box = (-0.14, 0.14, lm["neck"][1] - 0.03, lm["head_top"][1] + 0.01)
    I, Z = raster_front(V, faces, ids, box)
    out = {}
    eye_w = []
    for k, side in enumerate(("l", "r")):
        ys, xs = np.where(I == 1 + k)
        eye_w.append((xs.max() - xs.min()) * RES)
        out[f"eye_h_{side}"] = (ys.max() - ys.min()) * RES
    out["eye_w"] = float(np.mean(eye_w))
    out["eye_aspect"] = out["eye_w"] / out["eye_h_l"]
    crown, eye_y = lm["head_top"][1], lm["eye_c_l"][1]
    # chin bottom: midline verts below the mouth, first whose normal points mostly down
    mid = hv[(np.abs(V[hv, 0]) < 0.004) & (V[hv, 1] < lm["mouth"][1] - 0.018) & (V[hv, 2] > lm["jaw_hinge"][2])]
    mid = mid[np.argsort(-V[mid, 1])]
    chin = next((v for v in mid if base.N[v, 1] < -0.6), mid[-1])
    chin_y = V[chin, 1]
    out["crown_chin"] = crown - chin_y
    out["eye_line_from_crown"] = (crown - eye_y) / out["crown_chin"]
    # face width at cheekbones (exclude ears: only skin in front of the jaw hinge + 1 cm)
    front = Z > lm["jaw_hinge"][2] + 0.012
    rows = lambda y: int((box[3] - y) / RES)
    best = 0
    for y in np.linspace(lm["mouth"][1], eye_y, 20):
        r = rows(y)
        xs = np.where((I[r] == 0) & front[r])[0]
        if len(xs):
            best = max(best, (xs.max() - xs.min()) * RES)
    out["face_w"] = best
    full = 0
    for y in np.linspace(chin_y, crown, 60):
        xs = np.where(I[rows(y)] >= 0)[0]
        if len(xs):
            full = max(full, (xs.max() - xs.min()) * RES)
    out["head_w_with_ears"] = full
    out["eye_w/face_w"] = out["eye_w"] / best
    out["eye_gap/eye_w"] = (abs(lm["eye_c_l"][0] - lm["eye_c_r"][0]) - out["eye_w"]) / out["eye_w"]
    sm = __import__("juno_design").scalp_mask(base)
    fm = hv[(np.abs(V[hv, 0]) < 0.004) & (V[hv, 2] > lm["eye_c_l"][2] - 0.01) & (V[hv, 1] > eye_y)]
    hl = fm[sm[fm] > 0.5]
    hair_y = V[hl, 1].min() if len(hl) else crown
    out["face_h(hairline-chin)"] = hair_y - chin_y
    out["face_w/face_h"] = best / out["face_h(hairline-chin)"]
    out["head_w/crown_chin"] = best / out["crown_chin"]
    # nose: subnasale = min z on midline between nose tip and mouth
    nt = lm["nose_tip"]
    seg = hv[(np.abs(V[hv, 0]) < 0.004) & (V[hv, 1] < nt[1]) & (V[hv, 1] > lm["mouth"][1]) & (V[hv, 2] > lm["mouth"][2] - 0.02)]
    sub = seg[np.argmin(V[seg, 2])] if len(seg) else None
    sub_y = V[sub, 1] if sub is not None else nt[1]
    out["nose_len/face_h"] = (eye_y - sub_y) / out["face_h(hairline-chin)"]
    out["philtrum/face_h"] = (sub_y - lm["mouth"][1]) / out["face_h(hairline-chin)"]
    cp = np.linalg.norm(base.target("expression/units/caucasian/mouth-corner-puller.target.gz"), axis=1)
    cpv = np.argsort(-cp)[:6]
    out["mouth_w/face_w"] = (V[cpv, 0].max() - V[cpv, 0].min()) / best
    # neck: width at neck joint + 3 cm
    ny = lm["neck"][1] + 0.03
    xs = np.where(I[rows(ny)] >= 0)[0]
    out["neck_w/head_w"] = (xs.max() - xs.min()) * RES / best if len(xs) else None
    out["heads_tall"] = base.height() / out["crown_chin"]
    sh = base.cube("joint-l-shoulder")[0] - base.cube("joint-r-shoulder")[0]
    out["head_w_ears/shoulder_w"] = full / (sh + 0.09)
    out = {k: (round(float(v), 4) if v is not None else None) for k, v in out.items()}
    if png:
        from PIL import Image
        col = np.zeros(I.shape + (3,), np.uint8)
        col[I == 0] = (200, 160, 130)
        col[I == 1] = (60, 90, 200)
        col[I == 2] = (60, 90, 200)
        Image.fromarray(col).save(png)
    return out


if __name__ == "__main__":
    b = juno_base.Base(sys.argv[1])
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
    print(json.dumps(measure(b, png), indent=1))
