"""asset — landmark RBF sculpt of the head (owned by the ASSET lead).

Picks ~40 control vertices on the hm08 head by geometric queries (eye lid rings, nose, lips, chin, jaw, cheeks, brow),
places their TARGETS from a stylised layout expressed in face units (F = cheekbone face width, measured), adds zero-
displacement anchors on the neck / back of the skull, and solves a biharmonic RBF (phi = r, + linear) displacement
field applied to every head vertex (helpers and joint cubes included). The eyeballs are scaled rigidly about their new
centres and the lid skin is pushed off the new eyeball sphere, so lids always seal on the ball.
"""
from __future__ import annotations

import numpy as np

LAYOUT = {
    "eye_cc": 0.465,          # eye centre-to-centre / F
    "eye_w": 0.285,          # lid aperture width / F
    "eye_aspect": 1.85,      # aperture width / height
    "eye_ball_k": 0.84,      # eyeball radius = eye_ball_k * aperture width / 2 ... (>= aperture half-height)
    "eye_front_m": -0.0035,        # fraction of the radius growth the centre moves back
    "eye_dy": 0.0,           # eye line shift (m)
    "chin_up": 0.0135,       # m, chin bottom raised (shorter lower face)
    "mouth_up": 0.0060,      # m, lip centre raised (short philtrum)
    "subnasal_up": 0.0030,
    "nose_tip_back": 0.0035, "nose_tip_up": 0.0025, "nose_wing_in": 0.0025,
    "jaw_in": 0.0035,        # jaw angles move toward the midline (heart shape)
    "chin_narrow": 0.0012,
    "cheek_out": 0.0075,     # cheek fat pads forward/out
    "brow_up": 0.0020,
    "lip_lower_out": 0.0012,
}


_MARGIN = {}


def lid_margins(base, side):
    """skin vertices of the upper / lower lid margin: nearest skin vertices to the MH lash-card roots (topology-fixed)."""
    if side not in _MARGIN:
        from scipy.spatial import cKDTree
        V0 = base.obj.v * 0.1
        sk = base.obj.group_verts("body")
        kd = cKDTree(V0[sk])
        out = []
        for g in (f"helper-{side}-eyelashes-2", f"helper-{side}-eyelashes-1"):   # 2 = upper, 1 = lower
            lv = base.obj.group_verts(g)
            d, i = kd.query(V0[lv])
            out.append(np.unique(sk[i[d < 0.0012]]))
        up, lo = out
        lo = np.setdiff1d(lo, up)
        _MARGIN[side] = (up, lo)
    return _MARGIN[side]


def _rbf_solve(C, D, X, chunk=20000):
    """biharmonic RBF interpolation of displacements D (k,3) at centres C (k,3), evaluated at X (n,3)."""
    k = len(C)
    K = np.linalg.norm(C[:, None] - C[None], axis=2)
    Pm = np.concatenate([np.ones((k, 1)), C], 1)
    A = np.zeros((k + 4, k + 4))
    A[:k, :k] = K
    A[:k, k:] = Pm
    A[k:, :k] = Pm.T
    rhs = np.zeros((k + 4, 3))
    rhs[:k] = D
    sol = np.linalg.lstsq(A + 1e-9 * np.eye(k + 4), rhs, rcond=None)[0]
    out = np.zeros((len(X), 3))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        r = np.linalg.norm(x[:, None] - C[None], axis=2)
        out[i:i + chunk] = r @ sol[:k] + np.concatenate([np.ones((len(x), 1)), x], 1) @ sol[k:]
    return out


def sculpt(base, V, hw, skin, N, L=None, report=None):
    """V: (nV,3) after targets + head scale. hw: head-ness weight (0..1). skin: bool mask of the body group."""
    L = dict(LAYOUT, **(L or {}))
    g = base.obj.group_verts
    eye = {}
    for side in ("l", "r"):
        ev = g(f"helper-{side}-eye")
        c = V[ev].mean(0)
        eye[side] = (c, float(np.linalg.norm(V[ev] - c, axis=1).mean()))
    ec = 0.5 * (eye["l"][0] + eye["r"][0])
    sk = np.where(skin & (hw > 0.5))[0]
    P = V[sk]
    # --- measure F (cheekbone width, front of the hinge) and the lid rings
    hinge = V[g("joint-mouth")].mean(0)
    band = sk[(P[:, 1] < ec[1] - 0.01) & (P[:, 1] > ec[1] - 0.045) & (P[:, 2] > hinge[2] + 0.012)]
    F = float(V[band, 0].max() - V[band, 0].min())
    ctrl_i, ctrl_t = [], []

    def add(vi, target):
        ctrl_i.append(int(vi))
        ctrl_t.append(np.asarray(target, float))

    new_eye = {}
    for side, sx in (("l", 1.0), ("r", -1.0)):
        c, r = eye[side]
        q = V[sk] - c
        dist = np.linalg.norm(q, axis=1)
        up_m, lo_m = lid_margins(base, side)
        margin = np.concatenate([up_m, lo_m])
        is_up = np.concatenate([np.ones(len(up_m), bool), np.zeros(len(lo_m), bool)])
        mq = V[margin] - c
        aw_now = float(mq[:, 0].max() - mq[:, 0].min())
        ah_now = float(mq[:, 1].max() - mq[:, 1].min())
        my = 0.5 * (mq[:, 1].max() + mq[:, 1].min())
        aw = L["eye_w"] * F
        ah = aw / L["eye_aspect"]
        R = max(L["eye_ball_k"] * aw / 2, 0.62 * ah / 0.95)
        s_ball = R / r
        # the new ball's front sits eye_front_m ahead of the old ball's front (big eyes set INTO the head)
        c_new = np.array([sx * L["eye_cc"] * F / 2, c[1] + L["eye_dy"], c[2] + r + L["eye_front_m"] - R])
        new_eye[side] = (c, r, c_new, R)
        if report is not None:
            report[f"aperture_now_{side}"] = (aw_now, ah_now)
        for j, upper in zip(margin, is_up):
            v = V[j] - c
            u = np.clip((v[0] - 0.5 * (mq[:, 0].max() + mq[:, 0].min())) / (aw_now / 2), -1, 1)
            ui = u * sx                     # +1 = outer corner
            if upper:        # upper lid: high round arc, peak a little toward the inner side
                yy = 0.5 * ah * (1 - u * u) ** 0.5 * (1 - 0.12 * ui)
            else:            # lower lid: flatter arc
                yy = -0.5 * ah * 0.8 * (1 - u * u) ** 0.75
            t = np.array([u * aw / 2, yy - 0.04 * ah * ui, 0.0])
            zz = np.sqrt(max((1.03 * R) ** 2 - t[0] ** 2 - t[1] ** 2, (0.3 * R) ** 2))
            t[2] = zz
            add(j, c_new + t)
        # orbit rim (4 points at ~1.9 r): move with the eye centre, scaled a bit
        far = sk[(dist > 1.8 * r) & (dist < 2.1 * r) & (q[:, 2] > 0.0)]
        fq = V[far] - c
        fang = np.arctan2(fq[:, 1], fq[:, 0] * sx)
        for a0 in (np.pi / 2, 0.0, -np.pi / 2, np.pi):
            j = far[np.argmin(np.abs(np.angle(np.exp(1j * (fang - a0)))))]
            v = V[j] - c
            t = v * (1 + 0.25 * (s_ball - 1))
            t[2] = v[2] - 0.3 * (R - r)
            add(j, c_new + t + (np.array([0, L["brow_up"], 0]) if a0 > 1 else 0))
    # --- midline profile controls
    mid = sk[np.abs(V[sk, 0]) < 0.0035]
    mo_c = np.linalg.norm(base.target("expression/units/caucasian/mouth-compression.target.gz"), axis=1) ** 2
    mo = (V * mo_c[:, None]).sum(0) / mo_c.sum()
    nose_c = mid[(V[mid, 1] < ec[1]) & (V[mid, 1] > mo[1])]
    nt = nose_c[np.argmax(V[nose_c, 2])]
    add(nt, V[nt] + [0, L["nose_tip_up"], -L["nose_tip_back"]])
    sub_c = mid[(V[mid, 1] < V[nt, 1] - 0.004) & (V[mid, 1] > mo[1] + 0.004) & (V[mid, 2] > mo[2] - 0.03)]
    sn = sub_c[np.argmin(V[sub_c, 2])]
    add(sn, V[sn] + [0, L["subnasal_up"], -0.5 * L["nose_tip_back"]])
    lipu = mid[(V[mid, 1] > mo[1]) & (V[mid, 1] < V[sn, 1])]
    lu = lipu[np.argmax(V[lipu, 2])]
    add(lu, V[lu] + [0, L["mouth_up"], 0])
    lipl = mid[(V[mid, 1] < mo[1]) & (V[mid, 1] > mo[1] - 0.015)]
    ll = lipl[np.argmax(V[lipl, 2])]
    add(ll, V[ll] + [0, L["mouth_up"] * 0.95, L["lip_lower_out"]])
    chin_c = mid[(V[mid, 1] < mo[1] - 0.018) & (V[mid, 2] > hinge[2])]
    chin_c = chin_c[np.argsort(-V[chin_c, 1])]
    cb = next((v for v in chin_c if N[v, 1] < -0.6), chin_c[-1])
    cf = chin_c[np.argmax(V[chin_c, 2])]
    add(cb, V[cb] + [0, L["chin_up"], 0])
    add(cf, V[cf] + [0, L["chin_up"] * 0.9, 0])
    # mouth corners
    cp = np.linalg.norm(base.target("expression/units/caucasian/mouth-corner-puller.target.gz"), axis=1)
    for sx in (1, -1):
        cand = sk[(V[sk, 0] * sx > 0.01)]
        j = cand[np.argmax(cp[cand])]
        add(j, V[j] + [0, L["mouth_up"], 0])
        # nose wing
        wc = sk[(np.abs(V[sk, 1] - (V[sn, 1] + 0.004)) < 0.004) & (V[sk, 0] * sx > 0) & (V[sk, 2] > V[sn, 2] - 0.01)]
        if len(wc):
            j = wc[np.argmax(np.abs(V[wc, 0]))]
            add(j, V[j] + [-sx * L["nose_wing_in"], L["subnasal_up"], -0.5 * L["nose_tip_back"]])
        # jaw angle: lowest-outermost point in front of the ear at mouth height - 2.5 cm
        jc = sk[(np.abs(V[sk, 1] - (mo[1] - 0.022)) < 0.006) & (V[sk, 0] * sx > 0) & (V[sk, 2] > hinge[2] - 0.01)]
        j = jc[np.argmax(np.abs(V[jc, 0]))]
        add(j, V[j] + [-sx * L["jaw_in"], L["chin_up"] * 0.5, 0])
        # chin sides
        cs = sk[(np.abs(V[sk, 1] - (V[cf, 1])) < 0.004) & (np.abs(V[sk, 0] - sx * 0.018) < 0.004) & (V[sk, 2] > hinge[2])]
        if len(cs):
            j = cs[np.argmax(V[cs, 2])]
            add(j, V[j] + [-sx * L["chin_narrow"], L["chin_up"] * 0.9, 0])
        # cheek apple: below the eye, beside the nose
        ca = sk[(np.abs(V[sk, 1] - (0.5 * (ec[1] + mo[1]) - 0.002)) < 0.004) &
                (np.abs(V[sk, 0] - sx * 0.36 * F) < 0.006)]
        if len(ca):
            j = ca[np.argmax(V[ca, 2])]
            add(j, V[j] + [sx * 0.3 * L["cheek_out"], 0.4 * L["mouth_up"], L["cheek_out"]])
    # --- anchors: neck ring, back/top of the skull, ears (zero displacement)
    anc = []
    neck_y = V[g("joint-neck")].mean(0)[1]
    nk = np.where(skin & (np.abs(V[:, 1] - (neck_y + 0.01)) < 0.006))[0]
    anc += list(nk[:: max(1, len(nk) // 12)])
    back = sk[V[sk, 2] < hinge[2] - 0.04]
    anc += list(back[:: max(1, len(back) // 20)])
    top = sk[V[sk, 1] > ec[1] + 0.08]
    anc += list(top[:: max(1, len(top) // 10)])
    ears = sk[(np.abs(V[sk, 0]) > 0.5 * F + 0.004) & (V[sk, 2] < hinge[2] + 0.01)]
    anc += list(ears[:: max(1, len(ears) // 10)])
    ci = np.array(ctrl_i + anc)
    D = np.concatenate([np.array(ctrl_t) - V[ctrl_i], np.zeros((len(anc), 3))])
    ci, uq = np.unique(ci, return_index=True)
    D = D[uq]
    move = np.where(hw > 0.0)[0]
    disp = _rbf_solve(V[ci], D, V[move])
    V = V.copy()
    V[move] += disp * hw[move, None]
    # eyeballs (+ their joint cubes / lash helpers) rigidly scaled about the new centres
    for side in ("l", "r"):
        c, r, c_new, R = new_eye[side]
        c_cur = V[g(f"helper-{side}-eye")].mean(0)
        for grp in (f"helper-{side}-eye", f"joint-{side}-eye"):
            vv = g(grp)
            V[vv] = c_new + (V[vv] - c_cur) * (R / r)
        tg = g(f"joint-{side}-eye-target")
        V[tg] = V[tg] - V[tg].mean(0) + c_new + np.array([0, 0, 0.1])
        # skin around the ball wraps it: smooth radial push d -> Rt + b*softplus((d-Rt)/b) in the front cone
        q = V - c_new
        dist = np.linalg.norm(q, axis=1) + 1e-12
        Rt, b = 1.035 * R, 0.22 * R
        cone = np.clip((q[:, 2] / dist + 0.15) / 0.35, 0, 1)
        fd = Rt + b * np.logaddexp(0, (dist - Rt) / b)
        fade = np.clip((2.4 * R - dist) / (0.9 * R), 0, 1)
        newd = dist + cone * fade * fade * (3 - 2 * fade) * (fd - dist)
        m = skin & (newd > dist)
        V[m] = c_new + q[m] / dist[m, None] * newd[m, None]
    if report is not None:
        report.update({"F": F, "ctrl": len(ctrl_i), "anchors": len(anc),
                       "eye_R": {s: new_eye[s][3] for s in new_eye}, "eye_c": {s: new_eye[s][2].tolist() for s in new_eye}})
    return V
