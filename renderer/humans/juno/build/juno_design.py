"""asset — Juno's DESIGN (owned by the ASSET lead): MH targets + custom sculpt deformers on the hm08 base.

stylize(base, V, design) -> V   (V in metres, MH vertex order; every vertex incl. helpers + joint cubes moves together so
the skeleton, eyes, teeth follow the sculpt).
"""
from __future__ import annotations

import json
import os

import numpy as np

DESIGN = {
    "macro": [("macrodetails/african-female-young.target.gz", 1 / 3), ("macrodetails/asian-female-young.target.gz", 1 / 3),
              ("macrodetails/caucasian-female-young.target.gz", 1 / 3),
              ("macrodetails/proportions/female-young-averagemuscle-averageweight-idealproportions.target.gz", 1.0),
              ("macrodetails/universal-female-young-averagemuscle-minweight.target.gz", 0.35)],
    "targets": [
        # heart-shaped face: broad cheekbones/forehead, small pointed chin
        ("head/head-invertedtriangular.target.gz", 0.55), ("head/head-oval.target.gz", 0.25),
        ("chin/chin-width-decr.target.gz", 0.45), ("chin/chin-height-decr.target.gz", 0.25),
        ("chin/chin-bones-decr.target.gz", 0.3),
        ("cheek/l-cheek-volume-incr.target.gz", 0.55), ("cheek/r-cheek-volume-incr.target.gz", 0.55),
        ("cheek/l-cheek-bones-incr.target.gz", 0.25), ("cheek/r-cheek-bones-incr.target.gz", 0.25),
        ("head/head-scale-vert-decr.target.gz", 0.25),
        # small button nose
        ("nose/nose-scale-vert-decr.target.gz", 0.55), ("nose/nose-scale-horiz-decr.target.gz", 0.35),
        ("nose/nose-scale-depth-decr.target.gz", 0.3), ("nose/nose-point-up.target.gz", 0.45),
        ("nose/nose-volume-decr.target.gz", 0.4), ("nose/nose-hump-decr.target.gz", 0.6),
        ("nose/nose-nostrils-width-decr.target.gz", 0.3),
        # mouth: a bit smaller, fuller lips
        ("mouth/mouth-scale-horiz-decr.target.gz", 0.15), ("mouth/mouth-upperlip-volume-incr.target.gz", 0.35),
        ("mouth/mouth-lowerlip-volume-incr.target.gz", 0.35), ("mouth/mouth-angles-up.target.gz", 0.25),
        # eyes (targets), then the custom eye enlarger below
        ("eyes/l-eye-scale-incr.target.gz", 0.7), ("eyes/r-eye-scale-incr.target.gz", 0.7),
        # long slender neck, small ears
        ("neck/neck-scale-vert-incr.target.gz", 0.45), ("neck/neck-scale-horiz-decr.target.gz", 0.45),
        ("ears/l-ear-scale-decr.target.gz", 0.3), ("ears/r-ear-scale-decr.target.gz", 0.3),
    ],
    "head_scale": 1.27,      # whole head about the head joint (Pixar head/body ratio)
    "eye_scale": 1.0,       # eyeball + lids about each eye centre
    "eye_sink": 0.55,        # push the grown eye back by this fraction of the radius growth (no bug eyes)
    "eye_spread_m": 0.0,  # move each eye region outward (Pixar wide-set big eyes)
    "philtrum_lift_m": 0.0, "lower_face_k": 1.0, "chin_narrow": 1.0,
    "cheek_pad_m": 0.0, "rbf_sculpt": True, "face_smooth_iters": 80,
    "cranium_wide": 0.06, "cranium_low": 0.05,
    "finger_len": 1.08, "finger_slim": 0.90, "wrist_slim": 0.90,
    "shoulder_in_m": 0.0065,   # ASSET3 v7: was 0.016 -> ~5 % broader shoulders (judge r4: thin build)
    "leg_scale": 1.05,
    "eye_open_up_deg": 0.0,  # upper lid rotated up about the eye centre (rounder, bigger Pixar aperture)
    "eye_open_dn_deg": 0.0,  # lower lid rotated down
    # ASSET2 v5 proportions (applied after the fit, juno_design._proportions): ~6.3 heads tall, wider neck, bigger hands
    "head_post_scale": 1.14, "neck_shorten_m": 0.008, "neck_widen": 0.75, "hand_scale": 1.06,
    "ear_scale": 1.15, "ear_flare_deg": 10.0, "ear_thick": 1.15, "nipple_smooth_iters": 40,
    # asset-mouth3: rest mouth = soft slight smile (before head_post_scale x1.14)
    "mouth_rest_corner_up_m": 0.0022, "mouth_rest_lower_back_m": 0.0018, "mouth_rest_lower_thin_m": 0.0032,
    "mouth_rest_upper_full_m": 0.0007,
    "mouth_rest_widen": 0.15,   # asset-mouth4: mouth ~15 % wider (x stretch about the midline, corners out by .15 mw)
}

HEAD_HELPERS = ("helper-l-eye", "helper-r-eye", "helper-upper-teeth", "helper-lower-teeth", "helper-tongue",
                "joint-head", "joint-head-2", "joint-jaw", "joint-mouth", "joint-l-eye", "joint-r-eye",
                "joint-l-eye-target", "joint-r-eye-target", "joint-l-upperlid", "joint-r-upperlid",
                "joint-l-lowerlid", "joint-r-lowerlid", "joint-tongue-1", "joint-tongue-2", "joint-tongue-3",
                "joint-tongue-4", "helper-l-eyelashes-1", "helper-l-eyelashes-2", "helper-r-eyelashes-1",
                "helper-r-eyelashes-2", "helper-hair")


def _bone_weights(base, names):
    wj = json.load(open(os.path.join(base.data, "rigs", "standard", "weights.game_engine.json")))["weights"]
    w = np.zeros(base.nV)
    for n in names:
        for vi, x in wj.get(n, []):
            w[vi] += x
    return np.clip(w, 0, 1)


def _eye_centre(base, V, side):
    ev = base.obj.group_verts(f"helper-{side}-eye")
    c = V[ev].mean(0)
    return c, float(np.linalg.norm(V[ev] - c, axis=1).mean())


def _mask(base, groups):
    m = np.zeros(base.nV, bool)
    for g in groups:
        m[base.obj.group_verts(g)] = True
    return m


def _mouth(base, V):
    c = np.linalg.norm(base.target("expression/units/caucasian/mouth-compression.target.gz"), axis=1) ** 2
    m = (V * c[:, None]).sum(0) / c.sum()
    m[0] = 0.0
    return m


def _taubin(V, tris, mask, iters, lam=0.5, mu=-0.53):
    n = len(V)
    import scipy.sparse as sp
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        for f in (lam, mu):
            L = A @ V / deg[:, None] - V
            V = V + f * mask[:, None] * L
    return V


def _post_fit_cleanup(base, V0, V, relax_iters=12, margin_iters=6):
    """(1) relax triangles the fit flipped (normal vs the pre-fit normal) with their 1-ring; (2) smooth each lid-margin
    loop along its order (angle about the eye centre), keeping every margin vertex's distance to the eyeball centre."""
    from juno_base import tri_of, vertex_normals
    import scipy.sparse as sp
    tris = np.array([t for fi in base.obj.groups["body"] for t in tri_of([vi for vi, _ in base.obj.faces[fi]])])
    def fn(P):
        n = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    n0, n1 = fn(V0), fn(V)
    bad = (n0 * n1).sum(1) < 0.2
    rep = {"flipped_before": int(bad.sum())}
    if bad.any():
        mv = np.zeros(len(V), bool)
        mv[tris[bad].reshape(-1)] = True
        ring = np.isin(tris, np.where(mv)[0]).any(1)
        mv[tris[ring].reshape(-1)] = True
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
        A.data[:] = 1.0
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        for _ in range(relax_iters):
            L = A @ V / deg[:, None] - V
            V = V + 0.6 * mv[:, None] * L
        rep["flipped_after"] = int(((n0 * fn(V)).sum(1) < 0.2).sum())
    from juno_sculpt import lid_margins
    for side in ("l", "r"):
        c, r = _eye_centre(base, V, side)
        for ids in lid_margins(base, side):
            if len(ids) < 4:
                continue
            q = V[ids] - c
            order = ids[np.argsort(np.arctan2(q[:, 1], q[:, 0]))]
            rad = np.linalg.norm(V[order] - c, axis=1)
            P = V[order].copy()
            # open polyline (upper or lower lid): keep the two ends (corners) fixed
            for _ in range(margin_iters):
                P[1:-1] = P[1:-1] + 0.5 * (0.5 * (P[:-2] + P[2:]) - P[1:-1])
            dq = P - c
            P = c + dq / np.linalg.norm(dq, axis=1, keepdims=True) * rad[:, None]
            V[order] = P
    base.fit_cleanup = rep
    return V


def _design_fit_hook(d):
    """ONE switch: use renderer/humans/juno/build/design_fit.npz via apply_design_fit.apply(base, V) -> V when present.
    Disable with JUNO_DESIGN_FIT=0 or design={"design_fit": False}. The hook must return ALL nV rest positions, moving
    the MH eye helpers / joint cubes with the head so base.eye, the skeleton and lid margins follow."""
    if not d.get("design_fit", True) or os.environ.get("JUNO_DESIGN_FIT", "1") == "0":
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    if not (os.path.exists(os.path.join(here, "design_fit.npz")) and os.path.exists(os.path.join(here, "apply_design_fit.py"))):
        return None
    import importlib
    return importlib.import_module("apply_design_fit").apply


def stylize(base, V, design=None):
    d = dict(DESIGN)
    if design:
        d.update(design)
    for rel, w in d["macro"] + d["targets"]:
        V = V + w * base.target(rel)
    hs = d.get("head_scale", 1.0)
    if hs != 1.0:   # whole head about the head joint, weighted by the MH head-bone weight (helpers rigid)
        h = _bone_weights(base, ["head"])
        for g in HEAD_HELPERS:
            h[base.obj.group_verts(g)] = 1.0
        piv = V[base.obj.group_verts("joint-head")].mean(0) + np.array([0, -0.01, 0])
        V = piv + (V - piv) * (1 + (hs - 1) * h)[:, None]
    es = d.get("eye_scale", 1.0)
    if es != 1.0:   # eyeball uniform, lids follow with a falloff
        for side in ("l", "r"):
            c, r = _eye_centre(base, V, side)
            dist = np.linalg.norm(V - c, axis=1)
            w = 1 - np.clip((dist - 1.05 * r) / (1.5 * r), 0, 1)
            w = w * w * (3 - 2 * w)
            for g in (f"helper-{side}-eye", f"joint-{side}-eye", f"helper-{side}-eyelashes-1",
                      f"helper-{side}-eyelashes-2", f"joint-{side}-upperlid", f"joint-{side}-lowerlid"):
                w[base.obj.group_verts(g)] = 1.0
            w[base.obj.group_verts(f"joint-{side}-eye-target")] = 0.0
            V = V + (V - c) * ((es - 1) * w)[:, None]
            # sink + spread (same falloff, wider)
            w2 = 1 - np.clip((dist - 1.2 * r) / (2.2 * r), 0, 1)
            w2 = w2 * w2 * (3 - 2 * w2)
            w2[_mask(base, (f"helper-{side}-eye", f"joint-{side}-eye", f"helper-{side}-eyelashes-1",
                            f"helper-{side}-eyelashes-2", f"joint-{side}-upperlid", f"joint-{side}-lowerlid",
                            f"joint-{side}-eye-target"))] = 1.0
            sx = 1.0 if side == "l" else -1.0
            mv = np.array([sx * d.get("eye_spread_m", 0.0), 0.0, -d.get("eye_sink", 0.0) * (es - 1) * r])
            V = V + w2[:, None] * mv
    up, dn = d.get("eye_open_up_deg", 0.0), d.get("eye_open_dn_deg", 0.0)
    if up or dn:    # open the aperture: rotate lid skin about the eye centre's x axis, falloff away from the lids
        skin = np.zeros(base.nV, bool)
        skin[base.obj.group_verts("body")] = True
        for side in ("l", "r"):
            c, r = _eye_centre(base, V, side)
            q = V - c
            dist = np.linalg.norm(q, axis=1)
            w = 1 - np.clip((dist - 1.0 * r) / (1.3 * r), 0, 1)
            w = w * w * (3 - 2 * w) * skin * (q[:, 2] > -0.3 * r)
            ang = np.where(q[:, 1] >= 0, up, -dn) * w
            # fade near the horizontal (lid corners stay put)
            elev = np.arctan2(q[:, 1], np.maximum(np.abs(q[:, 0]), 1e-6))
            ang = ang * np.clip(np.abs(np.sin(elev)) * 1.6, 0, 1)
            a = np.radians(ang)
            ca, sa = np.cos(a), np.sin(a)
            y = q[:, 1] * ca + q[:, 2] * sa
            z = -q[:, 1] * sa + q[:, 2] * ca
            V = V + np.stack([np.zeros_like(y), y - q[:, 1], z - q[:, 2]], 1)
    fit = _design_fit_hook(d)
    if fit is not None:
        # SCULPT's non-rigid fit of the concept head replaces the scripted RBF sculpt + head polish
        V_pre = V.copy()
        V = fit(base, V)
        if d.get("post_fit_cleanup", False):   # OFF: its flip test vs pre-fit normals misfires on the fit's lid tuck
            V = _post_fit_cleanup(base, V_pre, V)
        d = dict(d, rbf_sculpt=False, face_smooth_iters=0, cranium_wide=0.0, cranium_low=0.0)
        base.design_source = "design_fit"
        # asset-eyes2: almond / hooded lid redesign on the fitted head (lid_design.LID keys, lid_* in DESIGN)
        from lid_design import redesign as _lid_redesign
        V = _lid_redesign(base, V, d)
    else:
        base.design_source = "rbf_sculpt"
    if d.get("rbf_sculpt"):
        from juno_sculpt import sculpt
        from juno_base import vertex_normals, tri_of
        hw0 = _bone_weights(base, ["head"]) + 0.6 * _bone_weights(base, ["neck_01"])
        hw0 = np.clip(hw0, 0, 1)
        hw0[_mask(base, HEAD_HELPERS)] = 1.0
        tris = np.array([t for f in base.obj.faces for t in tri_of([vi for vi, _ in f])])
        rep = {}
        V = sculpt(base, V, hw0, _mask(base, ("body",)), vertex_normals(V, tris), d.get("layout"), rep)
        base.sculpt_report = rep
    # ---- lower face: shorter philtrum + shorter/narrower chin, fuller cheek pads
    hw = _bone_weights(base, ["head"])
    hw[_mask(base, HEAD_HELPERS)] = 1.0
    ec = 0.5 * (_eye_centre(base, V, "l")[0] + _eye_centre(base, V, "r")[0])
    er = _eye_centre(base, V, "l")[1]
    mo = _mouth(base, V)
    hinge = V[base.obj.group_verts("joint-mouth")].mean(0)
    front = np.clip((V[:, 2] - (hinge[2] - 0.005)) / 0.03, 0, 1)
    nose_base_y = mo[1] + 0.6 * (ec[1] - mo[1]) * 0.5
    below = np.clip((nose_base_y - V[:, 1]) / 0.012, 0, 1)
    wlf = hw * front * below
    lift = d.get("philtrum_lift_m", 0.0)
    k = d.get("lower_face_k", 1.0)
    newy = nose_base_y - lift + (V[:, 1] - nose_base_y) * k
    V[:, 1] = V[:, 1] + wlf * (newy - V[:, 1])
    mo = _mouth(base, V)
    cn = d.get("chin_narrow", 1.0)
    wch = hw * front * np.clip((mo[1] - V[:, 1]) / 0.02, 0, 1)
    V[:, 0] = V[:, 0] * (1 + (cn - 1) * wch)
    cp = d.get("cheek_pad_m", 0.0)
    if cp:
        skin = _mask(base, ("body",))
        from juno_base import vertex_normals, tri_of
        tris = np.array([t for f in base.obj.faces for t in tri_of([vi for vi, _ in f])])
        N = vertex_normals(V, tris)
        for sx in (1, -1):
            cc = np.array([sx * (abs(_eye_centre(base, V, "l")[0][0]) + 0.1 * er), 0.5 * (ec[1] + mo[1]) - 0.004,
                           ec[2] - 0.2 * er])
            g = np.exp(-((V - cc) ** 2).sum(1) / (2 * (1.25 * er) ** 2)) * skin
            V = V + (cp * g)[:, None] * N
    it = d.get("face_smooth_iters", 0)
    if it:
        from juno_base import tri_of
        skin = _mask(base, ("body",))
        tris = np.array([t for fi in base.obj.groups["body"] for t in tri_of([vi for vi, _ in base.obj.faces[fi]])])
        m = hw * skin
        for side in ("l", "r"):
            c, r = _eye_centre(base, V, side)
            m *= np.clip((np.linalg.norm(V - c, axis=1) - 1.04 * r) / (0.12 * r), 0, 1)
        nt = V[np.where(skin & (np.abs(V[:, 0]) < 0.004))[0]]
        ntip = nt[np.argmax(nt[:, 2] - 10 * np.clip(ec[1] - 0.01 - nt[:, 1], -1, 0) - 10 * np.clip(nt[:, 1] - mo[1], -1, 0) * 0)]
        m *= np.clip((np.linalg.norm(V - mo, axis=1) - 0.022) / 0.01, 0, 1)
        m *= np.clip((np.linalg.norm(V - ntip, axis=1) - 0.018) / 0.01, 0, 1)
        hz = V[base.obj.group_verts("joint-mouth")].mean(0)[2]
        m *= np.clip((V[:, 2] - (hz - 0.012)) / 0.012, 0, 1)   # keep ears (behind the jaw hinge)
        V = _taubin(V, tris, m * 0.8, it)
    cw, cl = d.get("cranium_wide", 0.0), d.get("cranium_low", 0.0)
    if cw or cl:   # rounder, wider, lower cranium (above the eye line; eyes/lids untouched)
        hw2 = _bone_weights(base, ["head"])
        hw2[_mask(base, HEAD_HELPERS)] = 1.0
        ey = _eye_centre(base, V, "l")[0][1]
        w = np.clip((V[:, 1] - (ey + 0.012)) / 0.04, 0, 1)
        w = w * w * (3 - 2 * w) * hw2
        V[:, 0] *= 1 + cw * w
        V[:, 1] = V[:, 1] - cl * w * (V[:, 1] - ey)
    # ---- hands: longer, slimmer fingers, slim wrists (elegant hands)
    fl, fs, ws = d.get("finger_len", 1.0), d.get("finger_slim", 1.0), d.get("wrist_slim", 1.0)
    if fl != 1.0 or fs != 1.0 or ws != 1.0:
        for side in ("l", "r"):
            for fi, f in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                # finger axis from the joint cubes finger-<n>-1 .. -4 (MH numbering: 1 = thumb)
                pts = [V[base.obj.group_verts(f"joint-{side}-finger-{fi + 1}-{k}")].mean(0) for k in (1, 2, 3, 4)]
                w = np.clip(sum(_bone_weights(base, [f"{f}_0{k}_{side}"]) for k in (1, 2, 3)), 0, 1)
                for k in (1, 2, 3, 4):
                    w[base.obj.group_verts(f"joint-{side}-finger-{fi + 1}-{k}")] = 1.0
                a, b = pts[0], pts[3]
                ax = (b - a) / np.linalg.norm(b - a)
                t = (V - a) @ ax
                foot = a + t[:, None] * ax
                rad = V - foot
                tt = np.clip(t, 0, None)
                newp = a + (tt * fl + np.minimum(t, 0))[:, None] * ax + rad * fs
                V = V + w[:, None] * (newp - V)
            # wrist: radial slim around the forearm axis near the wrist
            el = V[base.obj.group_verts(f"joint-{side}-elbow")].mean(0)
            wr = V[base.obj.group_verts(f"joint-{side}-hand")].mean(0)
            ax = (wr - el) / np.linalg.norm(wr - el)
            t = (V - el) @ ax
            L = np.linalg.norm(wr - el)
            wa = np.clip(_bone_weights(base, [f"lowerarm_{side}", f"hand_{side}"]), 0, 1)
            prof = np.clip((t / L - 0.35) / 0.5, 0, 1) * np.clip((L * 1.12 - t) / (0.1 * L), 0, 1)
            foot = el + t[:, None] * ax
            V = V + ((ws - 1) * prof * wa)[:, None] * (V - foot)
    # ---- body: narrower shoulders, longer legs
    si = d.get("shoulder_in_m", 0.0)
    if si:
        for side, sx in (("l", 1), ("r", -1)):
            names = [f"{b}_{side}" for b in ("upperarm", "lowerarm", "hand")] + \
                    [f"{f}_0{i}_{side}" for f in ("thumb", "index", "middle", "ring", "pinky") for i in (1, 2, 3)]
            w = np.clip(_bone_weights(base, names) + 0.6 * _bone_weights(base, [f"clavicle_{side}"]), 0, 1)
            V[:, 0] -= sx * si * w
    ls = d.get("leg_scale", 1.0)
    if ls != 1.0:
        py = V[base.obj.group_verts("joint-pelvis")].mean(0)[1]
        w = np.clip((py - V[:, 1]) / 0.08, 0, 1)
        V[:, 1] = np.where(V[:, 1] < py, py + (V[:, 1] - py) * (1 + (ls - 1) * w), V[:, 1])
    V = _mouth_rest(base, V, d)
    V = _proportions(base, V, d)
    return V


def _mouth_rest(base, V, d):
    """asset-mouth3: the REST mouth = a closed soft slight smile (concept sheet_v4_0). The fitted head's rest read
    sad/pouty (corners down, lower lip pushed out). Lip line bends up toward the corners (both lips move together near
    the seam -> no shear, seam stays closed), the lower lip's vermilion goes back and its lower border rises (thinner,
    no pout). Distances in metres BEFORE _proportions (head_post_scale scales them). Every face shape is built on this."""
    up = d.get("mouth_rest_corner_up_m", 0.0)
    back = d.get("mouth_rest_lower_back_m", 0.0)
    thin = d.get("mouth_rest_lower_thin_m", 0.0)
    ov = os.environ.get("JUNO_MOUTH_REST", "")
    if "," in ov:                                                  # tuning: "up,back,thin" in mm
        up, back, thin = (float(t) * 1e-3 for t in ov.split(","))
    if not (up or back or thin) or os.environ.get("JUNO_MOUTH_REST", "1") == "0":
        return V
    from scipy.spatial import cKDTree
    skin = _mask(base, ("body",))
    mo = _mouth(base, V)
    op = np.linalg.norm(base.target("expression/units/caucasian/mouth-open.target.gz"), axis=1)
    near = skin & (np.linalg.norm(V - mo, axis=1) < 0.04) & (V[:, 2] > mo[2] - 0.008)
    om = op[near].max()
    U = np.where(near & (op < 0.35 * om))[0]
    L = np.where(near & (op > 0.6 * om))[0]
    dd, jj = cKDTree(V[U]).query(V[L])
    ok = dd < 0.0018
    sP = 0.5 * (V[L[ok]] + V[U[jj[ok]]])
    # front-most seam point per 1.5 mm x bin (the mouth bag behind touches too)
    xb_ = np.round(sP[:, 0] / 0.0015).astype(int)
    sP = np.array([sP[xb_ == b][np.argmax(sP[xb_ == b][:, 2])] for b in np.unique(xb_)])
    sP = sP[np.argsort(sP[:, 0])]
    mw = 0.5 * (sP[:, 0].max() - sP[:, 0].min())
    xs = np.linspace(-mw, mw, 25)
    sy = np.interp(xs, sP[:, 0], sP[:, 1])
    sz = np.interp(xs, sP[:, 0], sP[:, 2])
    sy, sz = 0.5 * (sy + sy[::-1]), 0.5 * (sz + sz[::-1])          # symmetric frame
    if os.environ.get("JUNO_MOUTH_REST_DBG"):
        print("mouth_rest mw", mw, "seam y", sy[12], file=__import__("sys").stderr)
    x = V[:, 0]
    u = x / mw
    h = (V[:, 1] - np.interp(np.clip(x, -mw, mw), xs, sy)) / mw
    dz = (V[:, 2] - np.interp(np.clip(x, -mw, mw), xs, sz)) / mw
    hw = _bone_weights(base, ["head"])
    hw[_mask(base, HEAD_HELPERS)] = 0.0                          # teeth/tongue/eyes stay
    m = skin * hw * np.clip((dz + 1.3) / 0.6, 0, 1)                # in front of the mouth bag
    au = np.abs(u)
    # 1) lip line bends up toward the corners (u^2 profile, peak at the corner, fades beyond it), both lips + the skin
    #    around them alike (no shear at the seam)
    cprof = np.clip(au, 0, 1) ** 2 * np.exp(-(np.maximum(au - 1.0, 0) / 0.55) ** 2)
    vfall = np.exp(-(h / 0.75) ** 2)
    D = np.zeros_like(V)
    D[:, 1] += up * cprof * vfall
    D[:, 0] += 0.25 * up * np.sign(x) * cprof * vfall                 # corners a touch out (a smile widens)
    # 2) lower lip: vermilion back (no pout); 0 at the seam (stays sealed), peak mid-lip, chin stays
    lo = (h < 0).astype(float)
    a = 0.30                                                       # ~ lower-vermilion height (mw units)
    hp = -h / a
    lowv = lo * np.clip(hp / 0.35, 0, 1) * np.exp(-(np.maximum(hp - 0.9, 0) / 0.8) ** 2) * np.exp(-(u / 0.85) ** 2)
    D[:, 2] -= back * lowv
    # 3) lower border rises toward the seam (thinner lower lip): 0 at the seam, peak at the border, fades to the chin
    g = lo * hp * np.exp(0.5 * (1 - hp ** 2)) * np.exp(-(u / 0.9) ** 4)
    D[:, 1] += thin * g
    # 4) upper lip volume: vermilion forward (0 at the seam, peak mid-lip, centre most) -> a fuller, rounder upper lip
    full = d.get("mouth_rest_upper_full_m", 0.0)
    if full:
        hu_ = np.clip(h / 0.08, 0, 1) * np.exp(-(np.maximum(h / 0.30 - 0.8, 0) / 0.5) ** 2) * (h > 0)
        D[:, 2] += full * hu_ * np.exp(-(u / 0.8) ** 2)
    V = V + D * m[:, None]
    # 5) asset-mouth4: a WIDER mouth (stylized, the reference's broad smile): x stretch about the midline, linear
    #    in u inside the mouth (corners move out by widen*mw), fading past the corners into the cheeks; vertical falloff
    #    keeps the nose and chin; the mouth bag behind widens alike (lining + teeth refit from it)
    wd = d.get("mouth_rest_widen", 0.0)
    ovw = os.environ.get("JUNO_MOUTH_WIDEN", "")
    if ovw:
        wd = float(ovw)
    if wd:
        sprof = u * np.exp(-(np.maximum(au - 0.95, 0) / 0.85) ** 2)
        JLO0, JLO1 = (float(t) for t in os.environ.get("JUNO_WLO", "0.45,0.9").split(","))
        vf = np.where(h > 0, np.exp(-(np.maximum(h - 0.15, 0) / 0.40) ** 2), np.exp(-(np.maximum(-h - JLO0, 0) / JLO1) ** 2))
        mz = skin * hw * np.clip((dz + 2.2) / 0.6, 0, 1)
        V = V.copy()
        V[:, 0] += wd * mw * sprof * vf * mz
    return V


def _proportions(base, V, d):
    """ASSET2 v5: Pixar proportions AFTER the fit (the fitted head shape is kept, only its size changes).
    head_post_scale: whole head (+ eyes, teeth, lash/eye helpers, joint cubes) uniformly about the skull base (joint-head),
    the neck blends; neck_shorten_m moves the head down (neck compresses); neck_widen: radial xz widening of the neck about
    its axis; hand_scale: hand + fingers about the wrist. base.design_fit (eye centres / R / iris_r) is transformed too."""
    hs = d.get("head_post_scale", 1.0)
    sh = d.get("neck_shorten_m", 0.0)
    nw = d.get("neck_widen", 0.0)
    if hs != 1.0 or sh or nw:
        wh = _bone_weights(base, ["head"])
        wh[_mask(base, HEAD_HELPERS)] = 1.0
        wn = _bone_weights(base, ["neck_01"])
        piv = V[base.obj.group_verts("joint-head")].mean(0)
        nk = V[base.obj.group_verts("joint-neck")].mean(0)
        # neck: radial widening about the neck axis (nk -> piv), strongest mid-neck, none on the shoulders/head
        if nw:
            ax = (piv - nk) / np.linalg.norm(piv - nk)
            t = (V - nk) @ ax
            foot = nk + t[:, None] * ax
            L = np.linalg.norm(piv - nk)
            prof = np.clip(t / L + 0.35, 0, 1) * np.clip((1.5 - t / L) / 0.5, 0, 1)   # v6: wider up to the jaw
            k = nw * np.clip(wn + 0.6 * wh, 0, 1) * (1 - wh) * prof
            V = V + k[:, None] * (V - foot)
        sc = 1 + (hs - 1) * wh
        V = piv + (V - piv) * sc[:, None]
        V[:, 1] -= sh * wh
        df = getattr(base, "design_fit", None)
        if df:
            for side in ("l", "r"):
                c = np.asarray(df[f"eye_center_{side}"], float)
                df[f"eye_center_{side}"] = (piv + (c - piv) * hs - np.array([0, sh, 0])).tolist()
            df["eye_radius"] = float(df["eye_radius"]) * hs
            if df.get("iris_radius"):
                df["iris_radius"] = float(df["iris_radius"]) * hs
            df["post_scale"] = hs
    es, ef = d.get("ear_scale", 1.0), d.get("ear_flare_deg", 0.0)
    if es != 1.0 or ef:   # concept ears: a bit bigger, flaring out at the back (weight = MH ear-scale target region)
        for side, sg in (("l", 1.0), ("r", -1.0)):
            w = np.clip(_ear_tn(base, side, V) * 2.0, 0, 1)   # v6: distance-gated (raw target reached 10 cm of scalp)
            w = w * w * (3 - 2 * w)
            w[~_mask(base, ("body",))] = 0.0
            ring = (w > 0.15) & (w < 0.5)
            piv = (V[ring] * w[ring, None]).sum(0) / w[ring].sum()
            q = V - piv
            V = V + (es - 1) * w[:, None] * q
            V[:, 0] = V[:, 0] + (d.get("ear_thick", 1.0) - 1) * w * (V[:, 0] - piv[0])   # v6: fuller ear rim in front view
            q = V - piv
            a = -sg * np.radians(ef) * w * np.clip((0.004 - q[:, 2]) / 0.012, 0, 1)   # flare the back rim only (cheek untouched)
            ca, sa = np.cos(a), np.sin(a)
            V = V + np.stack([q[:, 0] * ca + q[:, 2] * sa - q[:, 0], np.zeros(len(V)),
                              -q[:, 0] * sa + q[:, 2] * ca - q[:, 2]], 1)
    ns = d.get("nipple_smooth_iters", 0)
    if ns:   # hidden under the cardigan, but the knit shell follows the body: no nipple pokes through the knit
        from juno_base import tri_of
        skin = _mask(base, ("body",))
        tris = np.array([t for fi in base.obj.groups["body"] for t in tri_of([vi for vi, _ in base.obj.faces[fi]])])
        sp = V[base.obj.group_verts("joint-spine-1")].mean(0)
        nk = V[base.obj.group_verts("joint-neck")].mean(0)
        m = np.zeros(len(V))
        for sg in (1.0, -1.0):
            sel = np.where(skin & (sg * V[:, 0] > 0.03) & (V[:, 1] > sp[1]) & (V[:, 1] < nk[1] - 0.05))[0]
            tip = V[sel[np.argmax(V[sel, 2])]]
            dd = np.linalg.norm(V - tip, axis=1)
            m = np.maximum(m, np.clip((0.05 - dd) / 0.02, 0, 1))
        V = _taubin(V, tris, m * skin, ns)
    hsc = d.get("hand_scale", 1.0)
    if hsc != 1.0:
        for side in ("l", "r"):
            names = [f"hand_{side}"] + [f"{f}_0{i}_{side}" for f in ("thumb", "index", "middle", "ring", "pinky") for i in (1, 2, 3)]
            w = _bone_weights(base, names)
            for g in base.obj.groups:
                if g.startswith(f"joint-{side}-finger-") or g in (f"joint-{side}-hand-2", f"joint-{side}-hand-3"):
                    w[base.obj.group_verts(g)] = 1.0
            piv = V[base.obj.group_verts(f"joint-{side}-hand")].mean(0)
            V = piv + (V - piv) * (1 + (hsc - 1) * w)[:, None]
    return V


def skull_center(base):
    e = 0.5 * (base.lm["eye_c_l"] + base.lm["eye_c_r"])
    hj = base.lm["head_joint"]
    return np.array([0.0, e[1] + 0.012, 0.5 * (hj[2] + e[2]) - 0.035])


def _ear_tn(base, side, V=None):
    """MH ear-scale target magnitude (0..1), gated by distance to the ear core (the raw target also nudges far scalp)."""
    V = base.VA if V is None else V
    t = np.linalg.norm(base.target(f"ears/{side}-ear-scale-incr.target.gz"), axis=1)
    tn = t / max(t.max(), 1e-12)
    core = V[tn > 0.6].mean(0)
    d = np.linalg.norm(V - core, axis=1)
    return tn * np.clip((0.045 - d) / 0.015, 0, 1)


def ear_weight(base):
    """0..1 per MH vertex: 1 on the ear itself, 0 on the scalp around the ear root."""
    w = np.zeros(base.nV)
    for side in ("l", "r"):
        x = np.clip((_ear_tn(base, side) - 0.3) / 0.3, 0, 1)
        w = np.maximum(w, x * x * (3 - 2 * x))
    return w


def scalp_mask(base, soft_deg=5.0):
    """Agreed hairline (0..1 per MH vertex, 1 = under hair). Front hairline high on the forehead, temples, above the
    ears, down to the nape. Spherical around skull_center: elevation threshold as a function of azimuth."""
    c = skull_center(base)
    d = base.VA - c
    r = np.linalg.norm(d, axis=1) + 1e-9
    elev = np.degrees(np.arcsin(np.clip(d[:, 1] / r, -1, 1)))
    az = np.degrees(np.arctan2(np.abs(d[:, 0]), d[:, 2]))      # 0 = front, 90 = side, 180 = back
    # hairline elevation (deg) by azimuth: front 33, temple 22 @ 45, above-ear 12 @ 95, behind ear -10 @ 125, nape -38 @ 180
    # v3 (lead review): front hairline ~1/3 higher with a soft M (temples recede a little more), tapered edge
    # v6 (lead: hard cut edge above/in front of the ear): a tapered sideburn in front of the ear (az ~80), hair down to the
    # ear root above the ear (the ear itself is excluded below), nape lower
    kx = np.array([0, 15, 32, 50, 65, 80, 92, 105, 125, 150, 180.0])
    ky = np.array([37, 37.5, 38, 24, 8, -10, -6, -9, -18, -30, -36.0])
    thr = np.interp(az, kx, ky)
    m = np.clip((elev - thr) / soft_deg + 0.5, 0, 1)
    m[~base.skin_mask] = 0
    if hasattr(base, "obj"):
        m *= 1 - ear_weight(base)
    # only the head (not shoulders): within ~0.16 m of the skull centre
    m *= (r < 0.19)
    return m
