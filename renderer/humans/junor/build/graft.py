"""junor — graft the ICT head onto the MPFB2 adult body.

make_base(mpfb_data, ict_npz, identity) -> base (juno_base.Base, adult design) with:
    base.ict         ICTHead placed in asset metres (ICT scale 1:1, real size)
    base.VA          MH rest positions with the NECK warped so the MH surface passes through the ICT neck loop
    base.cut_faces   set of MH body face ids above the ICT neck loop (replaced by the ICT head; never exported)
    base.bridge      dict(P, UV, tris, J, W) the strip that welds the MH cut boundary to the ICT neck loop
    base.ict_JW      (J, W) skin weights for every ICT vertex (MH weights interpolated; teeth/tongue/eyes rigid)
    skeleton: eye_l/eye_r at the ICT eyeball centres, jaw at the hinge fitted to ICT jawOpen (lower teeth motion)
"""
from __future__ import annotations

import json
from collections import defaultdict
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
JUNO_BUILD = os.path.abspath(os.path.join(HERE, "..", "..", "juno", "build"))
for p in (HERE, JUNO_BUILD):
    if p not in sys.path:
        sys.path.insert(0, p)
import juno_base  # noqa: E402
from juno_base import tri_of, vertex_normals  # noqa: E402
from mh_base import bone_matrix_asset  # noqa: E402
import junor_design  # noqa: E402
from ict_head import ICTHead  # noqa: E402

FADE_LOOP_M = 0.035
TONGUE_DROP_M = 0.003  # tongue front lowered at rest (ICT tongue rests high against the upper teeth)      # expression deltas fade to 0 this close to the neck loop (the weld stays exact)


def _cyl(P, c):
    q = P - c
    return np.arctan2(q[:, 0], q[:, 2]), np.hypot(q[:, 0], q[:, 2])


def _interp_periodic(th_q, th, val):
    o = np.argsort(th)
    th, val = th[o], val[o]
    th2 = np.r_[th - 2 * np.pi, th, th + 2 * np.pi]
    v2 = np.r_[val, val, val]
    return np.interp(th_q, th2, v2)


def closest_on_tris(P, tris, Q, k=16):
    """closest point on a triangle soup for each query -> (tri idx, bary (n,3), dist)."""
    cen = P[tris].mean(1)
    _, cand = cKDTree(cen).query(Q, k=min(k, len(tris)))
    best_t = np.zeros(len(Q), np.int64)
    best_b = np.zeros((len(Q), 3))
    best_d = np.full(len(Q), np.inf)
    for j in range(cand.shape[1]):
        t = cand[:, j]
        a, b, c = P[tris[t, 0]], P[tris[t, 1]], P[tris[t, 2]]
        v0, v1, v2 = b - a, c - a, Q - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bc = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
        bc /= bc.sum(1, keepdims=True)
        q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
        d = np.linalg.norm(q - Q, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], t[m], bc[m]
    return best_t, best_b, best_d


NECK_SHORTEN_M = 0.012
IDENTITY_SYM = 0.6
REST_LID = 0.12
# v7 (HEAD2): a concept-fitted ICT neutral (cm, full ICT vertex array; scratch/head2/fit: multi-view landmark +
# silhouette fit of the identity PCA + Laplacian residual). Set JUNOR_NEUTRAL='' to fall back to identity.npy.
NEUTRAL = os.environ.get("JUNOR_NEUTRAL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "neutral_concept.npy"))
REST_LID_NEUTRAL = float(os.environ.get("JUNOR_REST_LID", "0.0"))
REST_WIDE = float(os.environ.get("JUNOR_REST_WIDE", "0.35"))
LOWER_GUM_DROP_M = 0.0022
NECK_RAMP_M = 0.07
NECK_SMOOTH_ITERS = 6   # v2: the scan neck's muscle relief (SCM cords) read heavy/masculine under the key light


def _smooth_neck(H, L, iters=NECK_SMOOTH_ITERS):
    """Taubin smoothing of the ICT neck skin between the jaw line and the weld (loop fixed, fades in over 2.5 cm)."""
    if not iters:
        return
    F, _ = H.skin_faces()
    rows, cols = [], []
    for f in F:
        f = [x for x in f if x >= 0]
        for i in range(len(f)):
            rows += [f[i], f[(i + 1) % len(f)]]
            cols += [f[(i + 1) % len(f)], f[i]]
    from scipy.sparse import coo_matrix
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(H.N, H.N)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    P = H.P
    chin_y = P[H.region("face"), 1].min() if False else P[1447, 1]   # jaw-line landmark (Multi-PIE 7)
    dl, _ = cKDTree(L).query(P)
    m = np.clip(dl / 0.025, 0, 1) * np.clip((chin_y - 0.012 - P[:, 1]) / 0.02, 0, 1)
    m[~H.region_mask("headneck", "face")] = 0
    m = m[:, None]
    for _ in range(iters):
        for lam in (0.5, -0.53):
            avg = (A @ P) / np.maximum(deg, 1)[:, None]
            P = P + lam * m * (avg - P)
    H.P = P


def _rebuild_skeleton(base):
    order = base.bone_names
    bones = base.bones
    base.rest = np.stack([bone_matrix_asset(bones[n]["head"], bones[n]["tail"], bones[n]["roll"]) for n in order])
    base.skeleton = {"bones": [{"name": n, "parent": base.bidx[bones[n]["parent"]] if bones[n]["parent"] else -1,
                                "rest": base.rest[i].reshape(-1).tolist(),
                                "inverse_bind": np.linalg.inv(base.rest[i]).reshape(-1).tolist(), "helper": False,
                                "length": float(np.linalg.norm(bones[n]["tail"] - bones[n]["head"]))}
                               for i, n in enumerate(order)]}


def make_base(mpfb_data, ict_npz, identity, design=None, log=print):
    d = dict(junor_design.NEUTRAL)
    if design:
        d.update(design)
    base = juno_base.Base(mpfb_data, design=d)
    H = ICTHead(ict_npz, identity, sym=IDENTITY_SYM)
    rest_lid = REST_LID
    if NEUTRAL and os.path.exists(NEUTRAL):
        H.Vcm = np.load(NEUTRAL).astype(np.float64)
        H.place(H.scale, H.offset)
        rest_lid = REST_LID_NEUTRAL
        log(f"graft: concept neutral {NEUTRAL}")
    VA = base.VA.copy()
    body_v = base.verts("body")
    body_f = base.faces_of("body")
    btris = np.array([t for fi in body_f for t in tri_of([v for v, _ in base.obj.faces[fi]])], np.int64)
    btri_face = np.array([fi for fi in body_f for _ in tri_of(base.obj.faces[fi])], np.int64)

    # ---- 1. place the ICT head (real scale): the neck loop should lie ON the MH surface
    loop_cm = H.Vcm[H.neck_loop]
    kd = cKDTree(VA[body_v])
    neck = base.cube("joint-neck")
    best = None
    for dy in np.arange(-0.06, 0.06, 0.004):
        for dz in np.arange(-0.04, 0.04, 0.004):
            off = np.array([0.0, neck[1] + dy, neck[2] + dz]) - np.array([0.0, loop_cm[:, 1].mean() * 0.01, loop_cm[:, 2].mean() * 0.01])
            dd, _ = kd.query(loop_cm * 0.01 + off)
            s = float(np.mean(dd ** 2))
            if best is None or s < best[0]:
                best = (s, off)
    off = best[1]
    for _ in range(2):   # refine
        for dy in np.arange(-0.004, 0.0041, 0.001):
            for dz in np.arange(-0.004, 0.0041, 0.001):
                o2 = off + np.array([0, dy, dz])
                dd, _ = kd.query(loop_cm * 0.01 + o2)
                s = float(np.mean(dd ** 2))
                if s < best[0]:
                    best = (s, o2)
        off = best[1]
    H.place(0.01, off)
    log(f"graft: ICT offset {np.round(off, 4).tolist()} loop rms {math.sqrt(best[0]) * 1000:.1f} mm")
    L = H.P[H.neck_loop]
    # v2: shorter neck (lead: long, cylindrical) - everything above the weld moves down, the loop itself stays
    dl0, _ = cKDTree(L).query(H.P)
    ramp = np.clip(dl0 / NECK_RAMP_M, 0, 1)
    H.P = H.P - np.outer(NECK_SHORTEN_M * ramp * ramp * (3 - 2 * ramp), [0.0, 1.0, 0.0])
    _smooth_neck(H, L)
    # v3: relaxed upper lids - bake REST_LID of the scan blink into the rest pose (upper lid covers ~1.5 mm of the iris
    # instead of a stare); the blink shapes keep closing exactly (scaled by 1 - REST_LID)
    for sd in ("L", "R"):
        D = H.expr[f"eyeBlink_{sd}"]
        H.P = H.P + rest_lid * D
        H.expr[f"eyeBlink_{sd}"] = D * (1 - rest_lid)
        if rest_lid is not REST_LID and REST_WIDE:
            # v11 (HEAD2): open the palpebral fissure with the scan eyeWide (upper lid up + crease), the blink still
            # closes onto the same closed lid (delta re-based on the new rest)
            Wd = H.expr[f"eyeWide_{sd}"]
            H.P = H.P + REST_WIDE * Wd
            H.expr[f"eyeBlink_{sd}"] = H.expr[f"eyeBlink_{sd}"] - REST_WIDE * Wd

    # ---- 2. warp the MH neck/shoulders so the surface passes through the loop
    tri_i, bary, dist = closest_on_tris(VA, btris, L)
    foot = (VA[btris[tri_i]] * bary[:, :, None]).sum(1)
    res = L - foot                                         # where the MH surface must move at each loop point
    kdl = cKDTree(L)
    dl, il = kdl.query(VA, k=6)
    w = np.exp(-(dl / 0.02) ** 2)
    disp = (w[:, :, None] * res[il]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    fall = np.clip(1 - dl[:, 0] / 0.07, 0, 1)
    fall = fall * fall * (3 - 2 * fall)
    VA = VA + disp * fall[:, None]
    # ---- 3. cut: MH vertices above the loop (cylindrical, about the neck axis)
    c = np.array([0.0, 0.0, L[:, 2].mean()])
    thL, rL = _cyl(L, c)
    th, r = _cyl(VA, c)
    yL = _interp_periodic(th, thL, L[:, 1])
    rLq = _interp_periodic(th, thL, rL)
    above = (VA[:, 1] > yL - 0.004) & (r < rLq + 0.03) & (VA[:, 1] > L[:, 1].min() - 0.02)
    above |= VA[:, 1] > L[:, 1].max() + 0.01
    cut = set()
    for fi in body_f:
        if any(above[v] for v, _ in base.obj.faces[fi]):
            cut.add(int(fi))
    base.cut_faces = cut
    # boundary of kept body faces around the neck
    from collections import defaultdict
    E = defaultdict(list)
    for fi in body_f:
        if fi in cut:
            continue
        f = base.obj.faces[fi]
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E[(min(a[0], b[0]), max(a[0], b[0]))].append((fi, a, b))
    cut_adj = defaultdict(list)
    for fi in cut:
        f = base.obj.faces[fi]
        for i in range(len(f)):
            a, b = f[i][0], f[(i + 1) % len(f)][0]
            cut_adj[(min(a, b), max(a, b))].append(fi)
    bedges = [(k, v[0]) for k, v in E.items() if len(v) == 1 and k in cut_adj]
    adj = defaultdict(list)
    uvof = {}
    for (a, b), (fi, ca, cb) in bedges:
        adj[a].append(b)
        adj[b].append(a)
        uvof[a] = ca[1]
        uvof[b] = cb[1]
    start = max(adj, key=lambda v: VA[v, 2])
    mloop, prev, cur = [start], None, start
    while True:
        nx = [n for n in adj[cur] if n != prev]
        if not nx or nx[0] == start:
            break
        prev, cur = cur, nx[0]
        mloop.append(cur)
    if len(mloop) != len(adj):
        log(f"graft: WARNING MH cut boundary has {len(adj)} verts, loop walk {len(mloop)}")
    mloop = np.array(mloop)
    # same orientation + start angle as the ICT loop
    thm, _ = _cyl(VA[mloop], c)
    il_ = np.array(H.neck_loop)
    thi = thL.copy()
    if np.sum(np.sin(np.diff(np.unwrap(thm)))) * np.sum(np.sin(np.diff(np.unwrap(thi)))) < 0:
        il_, thi = il_[::-1], thi[::-1]
    mloop = np.roll(mloop, -int(np.argmin(np.abs(np.angle(np.exp(1j * thm))))))
    k0 = int(np.argmin(np.abs(np.angle(np.exp(1j * thi)))))
    il_, thi = np.roll(il_, -k0), np.roll(thi, -k0)
    thm, _ = _cyl(VA[mloop], c)
    # snap the MH boundary onto the ICT loop (3 mm below it) and drag the next rings along -> a narrow, smooth weld
    tgt = np.stack([_interp_periodic(thm, thL, L[:, k]) for k in range(3)], 1) - np.array([0.0, 0.003, 0.0])
    dsn = tgt - VA[mloop]
    kdm = cKDTree(VA[mloop])
    keep_v = np.unique([v for fi in body_f if fi not in cut for v, _ in base.obj.faces[fi]])
    dd, jj = kdm.query(VA[keep_v])
    fall2 = np.clip(1 - dd / 0.05, 0, 1) ** 2
    fall2[np.isin(keep_v, mloop)] = 1.0
    VA[keep_v] += dsn[jj] * fall2[:, None]
    VA[mloop] = tgt
    # ---- 4. bridge (zip by unwrapped angle)
    um = np.unwrap(thm) - thm[0]
    ui = np.unwrap(thi) - thi[0]
    sgn = np.sign(um[-1]) or 1.0
    um, ui = um * sgn, ui * sgn
    nm, ni = len(mloop), len(il_)
    PB = np.concatenate([VA[mloop], H.P[il_]])
    tris = []
    a = b = 0
    while a < nm or b < ni:
        ua = um[a + 1] if a + 1 < nm else 2 * np.pi
        ub = ui[b + 1] if b + 1 < ni else 2 * np.pi
        if (ua <= ub and a < nm) or b >= ni:
            tris.append([a % nm, (a + 1) % nm, nm + b % ni])
            a += 1
        else:
            tris.append([a % nm, nm + (b + 1) % ni, nm + b % ni])
            b += 1
    tris = np.array(tris)
    # orient outward: the normal should point away from the neck axis
    fn = np.cross(PB[tris[:, 1]] - PB[tris[:, 0]], PB[tris[:, 2]] - PB[tris[:, 0]])
    cen = PB[tris].mean(1)
    out = cen - np.c_[np.zeros(len(cen)), cen[:, 1], np.full(len(cen), c[2])]
    out[:, 1] = 0
    if np.mean((fn * out).sum(1) > 0) < 0.5:
        tris = tris[:, [0, 2, 1]]
    # bridge UVs: MH loop -> its own vt; ICT side -> closest point on the (uncut) MH surface in the body UV island
    UVm = np.array([base.obj.vt[uvof[v]] for v in mloop])
    ti2, b2, _ = closest_on_tris(VA, btris, H.P[il_])
    UVi = []
    for t, bb in zip(ti2, b2):
        fi = btri_face[t]
        f = base.obj.faces[fi]
        tv = btris[t]
        uvs = []
        for vv in tv:
            ti_ = [x[1] for x in f if x[0] == vv][0]
            uvs.append(base.obj.vt[ti_])
        UVi.append((np.array(uvs) * bb[:, None]).sum(0))
    UVB = np.concatenate([UVm, np.array(UVi)])
    base.VA = VA
    base.N = vertex_normals(VA, base.tris)
    # ---- 5. skin weights for the ICT head (MH weights at the closest MH skin point; rigid parts)
    Wd = base.weights_dense()
    tri3, b3, _ = closest_on_tris(VA, btris, H.P)
    Wict = (Wd[btris[tri3]] * b3[:, :, None]).sum(1)
    hb, jb = base.bidx["head"], base.bidx["jaw"]
    # jaw rigid motion fitted to the ICT lower teeth (jawOpen)
    lo = H.region("teeth_lower")
    Rj, tj = H.rigid_fit(lo, H.expr["jawOpen"])
    gt = H.region("gums_tongue")
    m = (H.P[gt] @ Rj.T + tj) - H.P[gt]
    mn = np.linalg.norm(m, axis=1)
    wj = np.clip((H.expr["jawOpen"][gt] * m).sum(1) / np.maximum(mn, 1e-9) ** 2, 0, 1)
    H.jaw_w = np.zeros(H.N)
    H.jaw_w[gt] = wj
    H.jaw_w[lo] = 1.0
    H.lower_gt = gt[wj > 0.5]
    H.upper_gt = gt[wj <= 0.5]
    H.jaw_fit = (Rj, tj)
    # tongue = the gums_tongue UV island(s) that only partly follow the jaw (its top surface); lower its front so an
    # open mouth shows the lower incisors in front of it instead of a tongue tip pressed against the upper teeth
    Fg, FTg = H.faces("gums_tongue")
    par = np.arange(len(H.VT))

    def fnd(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for ft in FTg:
        tt = [x for x in ft if x >= 0]
        for q in tt[1:]:
            par[fnd(q)] = fnd(tt[0])
    isl = defaultdict(list)
    for i, ft in enumerate(FTg):
        isl[fnd(ft[0])].append(i)
    tongue = []
    for fl in isl.values():
        vs = np.unique(Fg[fl][Fg[fl] >= 0])
        mvm = np.linalg.norm(H.expr["jawOpen"][vs], axis=1).mean()
        if 0.010 < mvm < 0.030 and H.P[vs, 2].max() > H.P[lo, 2].max() - 0.012:
            tongue.append(vs)
    H.tongue_ids = np.unique(np.concatenate(tongue)) if tongue else np.zeros(0, np.int64)
    if len(H.tongue_ids):
        tz = H.P[H.tongue_ids, 2]
        fr = np.clip((tz - tz.min()) / max(np.ptp(tz), 1e-9), 0, 1) ** 1.5
        H.P[H.tongue_ids] += fr[:, None] * np.array([0.0, -TONGUE_DROP_M, -0.4 * TONGUE_DROP_M])
    # lower gum line 2 mm lower at the front so the lower incisor crowns show when the jaw opens (the ICT gums sit high)
    lg = np.setdiff1d(gt[wj > 0.5], H.tongue_ids)
    zf = H.P[lo, 2].max()
    fw = np.clip((H.P[lg, 2] - (zf - 0.016)) / 0.008, 0, 1)
    H.P[lg] += fw[:, None] * np.array([0.0, -LOWER_GUM_DROP_M, -0.0006])
    Wict[gt] = 0
    Wict[gt, jb] = wj
    Wict[gt, hb] = 1 - wj
    Wict[H.region("teeth_upper")] = 0
    Wict[H.region("teeth_upper"), hb] = 1
    Wict[lo] = 0
    Wict[lo, jb] = 1
    for side in ("l", "r"):
        for rg in (f"eyeball_{side}",):
            ids = H.region(rg)
            Wict[ids] = 0
            Wict[ids, base.bidx[f"eye_{side}"]] = 1
        for rg in (f"lacrimal_{side}", f"eyeblend_{side}", f"occlusion_{side}", f"lashes_{side}", f"eye_socket_{side}"):
            ids = H.region(rg)
            Wict[ids] = 0
            Wict[ids, hb] = 1
    Wict[H.region("mouth_socket")] = 0
    Wict[H.region("mouth_socket"), hb] = 1
    # the face itself: head bone above the jaw line (MH weights can leak neck onto a bigger chin)
    hw = Wict[:, hb]
    J4 = np.argsort(-Wict, 1)[:, :4]
    W4 = np.take_along_axis(Wict, J4, 1)
    W4 /= np.maximum(W4.sum(1, keepdims=True), 1e-9)
    base.ict_JW = (J4.astype(np.uint16), W4.astype(np.float32))
    # bridge weights (exactly the weights of the vertices it welds)
    Jm, Wm = base.J[mloop], base.W[mloop]
    base.bridge = dict(P=PB, UV=UVB, tris=tris, J=np.concatenate([Jm, base.ict_JW[0][il_]]),
                       W=np.concatenate([Wm, base.ict_JW[1][il_]]), mloop=mloop, iloop=il_)
    # ---- 6. skeleton: eyes, jaw hinge, head (keep the MH spine/neck)
    for side in ("l", "r"):
        ce, R = H.eye_sphere(side)
        base.bones[f"eye_{side}"]["head"] = ce
        base.bones[f"eye_{side}"]["tail"] = ce + np.array([0, 0, 0.02])
        base.eye[side] = (ce, R)
    ang = math.degrees(math.atan2(Rj[2, 1], Rj[1, 1]))
    A = (np.eye(3) - Rj)[1:, 1:]
    hyz = np.linalg.solve(A, tj[1:])
    hinge = np.array([0.0, hyz[0], hyz[1]])
    chin = H.P[H.region("teeth_lower")]
    chin = np.array([0.0, chin[:, 1].min(), chin[:, 2].max()])
    base.bones["jaw"]["head"] = hinge
    base.bones["jaw"]["tail"] = chin
    _rebuild_skeleton(base)
    base.jaw = dict(open_deg=ang, hinge=hinge)
    RL, tL = H.rigid_fit(lo, H.expr["jawLeft"])
    base.jaw["side_deg"] = abs(math.degrees(math.atan2(RL[0, 2], RL[0, 0])))
    base.jaw["forward_m"] = float(H.expr["jawForward"][lo, 2].mean())
    # landmarks from the ICT head (Multi-PIE 68 indices in the README)
    P = H.P
    lm68 = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493,
            1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507,
            1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213,
            6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]
    base.lm68 = P[lm68]
    face_ids = H.region("face")
    skin = np.concatenate([face_ids, H.region("headneck")])
    mid = skin[np.abs(P[skin, 0]) < 0.003]
    nose_tip = P[mid[np.argmax(P[mid, 2])]]
    mouth = 0.5 * (P[lm68[62]] + P[lm68[66]])
    mouth[0] = 0
    chin_s = P[lm68[8]].copy()
    chin_s[0] = 0
    top = P[skin[np.argmax(P[skin, 1])]]
    base.lm.update({"eye_c_l": base.eye["l"][0], "eye_c_r": base.eye["r"][0], "eye_r_l": base.eye["l"][1],
                    "eye_r_r": base.eye["r"][1], "mouth": mouth, "jaw_hinge": hinge, "chin": chin_s, "head_top": top,
                    "nose_tip": nose_tip})
    # expression fade near the weld
    dl2, _ = cKDTree(L).query(H.P)
    H.fade = np.clip((dl2 - 0.012) / (FADE_LOOP_M - 0.012), 0, 1)
    base.ict = H
    log(f"graft: MH loop {nm} / ICT loop {ni} verts, {len(tris)} bridge tris, {len(cut)} MH faces cut, jaw {ang:.1f} deg "
        f"hinge {np.round(hinge, 4).tolist()}, eyes R {base.eye['l'][1] * 1000:.2f} mm, height {base.height():.3f}")
    return base
