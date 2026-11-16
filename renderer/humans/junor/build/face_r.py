"""junor — face shapes on the ICT head: the scan-derived ICT expressions renamed to ARKit, plus the juno extras
and correctives built from them.  build(base) -> (S {name: (N,3) metres on ICT vertices}, CORRECTIVES {name: (a, b)}).

  ARKit 52     ICT X_L/X_R -> XLeft/XRight; browInnerUp, cheekPuff = L + R; tongueOut procedural (ICT has none)
  jaw*         the jaw bone carries the lower teeth + lower gums/tongue rigidly (graft: hinge fitted to ICT jawOpen);
               on those vertices the shape is the RESIDUAL after the bone's motion; upper teeth/gums never move
  extras       lipTuck* = mouthRoll*, labioDental = .7 rollLower + .35 shrugLower + .15 upperUp, lipsTogether =
               .25 press, eyeHappy = cheekRaiser + .3 eyeSquint, browTilt = .8 innerUp(side) - .35 outerUp,
               tongue* procedural on the tongue (distance-from-teeth mask)
  correctives  corr_eyeBlinkX_eyeBlinkX: the lid travels on an ARC over the eyeball (linear scan blink cuts the cornea
               half-way); corr_jawOpen_lipsTogether = ICT mouthClose (lips sealed at any jaw: product rule j*1),
               corr_jawOpen_mouthPressL/R = side halves of mouthClose
Every skin delta fades to 0 near the neck weld (graft.FADE_LOOP_M) so the head/body seam never opens.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ict_head import arkit_name

CORRECTIVES = {}
CORNEA_CLEAR = 1.085     # closed / half-closed lid stays this x eyeball R from the centre over the cornea cap


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


SMILE_UPPER_UP = 0.0   # v11: closed smiles keep the lips together; the open/strong smile correctives bare the teeth
SMILE_X = 0.7
SMILE_Y = 1.15
SMILE_RAISER = 0.25
OPEN_LOWER_DOWN = 0.5
OPEN_UPPER_UP = 0.45


def build(base):
    H = base.ict
    P = H.P
    E = H.expr
    fade = H.fade[:, None]
    S = {}
    for n, d in E.items():
        if n.startswith(("browInnerUp", "cheekPuff", "PupilDilate", "cheekRaiser")):
            continue
        S[arkit_name(n)] = d * fade
    S["browInnerUp"] = (E["browInnerUp_L"] + E["browInnerUp_R"]) * fade
    S["cheekPuff"] = (E["cheekPuff_L"] + E["cheekPuff_R"]) * fade
    raiser = {"Left": E["cheekRaiser_L"] * fade, "Right": E["cheekRaiser_R"] * fade}
    # ---- rigid parts
    upper = H.region("teeth_upper")
    lower_teeth = H.region("teeth_lower")
    eyes = np.concatenate([H.region("eyeball_l"), H.region("eyeball_r")])
    for k in S:
        S[k][upper] = 0
        S[k][lower_teeth] = 0
        S[k][eyes] = 0
    # jaw residuals on the gums/tongue (skinned jaw_w to the jaw bone): shape = ICT delta - jaw_w * bone motion
    J = base.jaw
    hinge = J["hinge"]
    gt = H.region("gums_tongue")
    wj = H.jaw_w[gt][:, None]
    q = P[gt] - hinge
    rig = {"jawOpen": (_rotx(math.radians(J["open_deg"])) @ q.T).T - q,
           "jawLeft": (_roty(math.radians(J["side_deg"])) @ q.T).T - q,
           "jawRight": (_roty(-math.radians(J["side_deg"])) @ q.T).T - q,
           "jawForward": np.tile([0.0, 0.0, J["forward_m"]], (len(q), 1))}
    for k, m in rig.items():
        S[k][gt] = E[k][gt] - wj * m
    for k in S:
        if k not in rig:
            S[k][gt] = 0
    lower_gt = H.tongue_ids
    tz = P[lower_gt, 2]
    tip = _smooth(np.percentile(tz, 30), tz.max(), tz)
    back = 1 - tip
    H.lower_gum_ids = np.setdiff1d(H.lower_gt, H.tongue_ids)

    def T(vec, w):
        d = np.zeros_like(P)
        d[lower_gt] = w[:, None] * np.asarray(vec)
        return d
    S["tongueOut"] = T([0, -0.004, 0.026], tip ** 1.5)
    S["tongueTipUp"] = T([0, 0.009, 0.002], tip ** 2)
    S["tongueDental"] = T([0, 0.004, 0.009], tip ** 2)
    S["tongueBack"] = T([0, 0.004, -0.006], back ** 1.5 + 0.3 * tip)
    # v10 (HEAD2): smiles lift. The scan mouthSmile pulls the corners mostly sideways; add the upper-lip raiser and the
    # cheek raiser so a smile shows the upper incisors with a curved smile line, the cheeks round and the fold forms.
    for sd in ("Left", "Right"):
        # v11 (lead 12:40, rig closed smile read as a grimace): the scan smile spreads the corners ~sideways and pushes
        # the lower cheeks out (jowl); keep less of the lateral spread and more of the lift
        D = S[f"mouthSmile{sd}"].copy()
        D[:, 0] *= SMILE_X
        D[:, 1] *= SMILE_Y
        S[f"mouthSmile{sd}"] = D + SMILE_UPPER_UP * S[f"mouthUpperUp{sd}"] + SMILE_RAISER * raiser[sd]
    # ---- extras
    S["lipTuckLower"] = S["mouthRollLower"].copy()
    S["lipTuckUpper"] = S["mouthRollUpper"].copy()
    S["labioDental"] = 0.7 * S["mouthRollLower"] + 0.35 * S["mouthShrugLower"] + 0.22 * (S["mouthUpperUpLeft"] + S["mouthUpperUpRight"])   # v11: upper incisors show on f/v
    S["lipsTogether"] = 0.25 * (S["mouthPressLeft"] + S["mouthPressRight"])
    for sd in ("Left", "Right"):
        S[f"eyeHappy{sd}"] = raiser[sd] + 0.3 * S[f"eyeSquint{sd}"]
        S[f"browTilt{sd}"] = 0.8 * (E["browInnerUp_L" if sd == "Left" else "browInnerUp_R"] * fade) - 0.35 * S[f"browOuterUp{sd}"]
    # ---- correctives
    CORRECTIVES.clear()
    mc = S["mouthClose"]
    S["corr_jawOpen_lipsTogether"] = mc.copy()
    CORRECTIVES["corr_jawOpen_lipsTogether"] = ("jawOpen", "lipsTogether")
    xs = _smooth(-0.012, 0.012, P[:, 0])[:, None]
    S["corr_jawOpen_mouthPressLeft"] = mc * xs
    S["corr_jawOpen_mouthPressRight"] = mc * (1 - xs)
    CORRECTIVES["corr_jawOpen_mouthPressLeft"] = ("jawOpen", "mouthPressLeft")
    CORRECTIVES["corr_jawOpen_mouthPressRight"] = ("jawOpen", "mouthPressRight")
    # Duchenne smile: a strong smile (w^2) adds the scan cheek raiser + cheek squint (round cheek, deeper fold)
    for sd in ("Left", "Right"):
        S[f"corr_mouthSmile{sd}_mouthSmile{sd}"] = 0.4 * raiser[sd] + 0.15 * S[f"cheekSquint{sd}"] + 0.25 * S[f"mouthUpperUp{sd}"]
        CORRECTIVES[f"corr_mouthSmile{sd}_mouthSmile{sd}"] = (f"mouthSmile{sd}", f"mouthSmile{sd}")
        # open-mouth smile (teeth smile / laugh): the upper lip clears the incisors
        S[f"corr_jawOpen_mouthSmile{sd}"] = 0.8 * S[f"mouthUpperUp{sd}"]
        CORRECTIVES[f"corr_jawOpen_mouthSmile{sd}"] = ("jawOpen", f"mouthSmile{sd}")
    # v11: open vowels drop the lower lip off the lower incisors and lift the upper lip a little (j^2: none at small jaw)
    S["corr_jawOpen_jawOpen"] = OPEN_LOWER_DOWN * (S["mouthLowerDownLeft"] + S["mouthLowerDownRight"]) + OPEN_UPPER_UP * (S["mouthUpperUpLeft"] + S["mouthUpperUpRight"])
    CORRECTIVES["corr_jawOpen_jawOpen"] = ("jawOpen", "jawOpen")
    # blink arc: P(w) = P0 + w (D - C) + w^2 C with P(.5) on the radius interpolated between rest and closed
    for sd, s in (("Left", "l"), ("Right", "r")):
        c, R = base.eye[s]
        D = S[f"eyeBlink{sd}"]
        # closed lid must clear the cornea bulge (white specular dots poked through the closed lids in v1/b6)
        gaze = np.array([0.0, 0.0, 1.0])
        q1 = P + D - c
        r1_ = np.linalg.norm(q1, axis=1)
        cosg = (q1 @ gaze) / np.maximum(r1_, 1e-9)
        Rc = CORNEA_CLEAR * R
        need = (np.linalg.norm(D, axis=1) > 2e-4) & (cosg > 0.55) & (r1_ < Rc) & (np.linalg.norm(P - c, axis=1) < 1.6 * R)
        D = D.copy()
        D[need] = (c + q1[need] / r1_[need, None] * Rc) - P[need]
        P0, P1 = P, P + D
        r0 = np.linalg.norm(P0 - c, axis=1)
        r1 = np.linalg.norm(P1 - c, axis=1)
        M = P0 + 0.5 * D
        rm = np.linalg.norm(M - c, axis=1)
        want = np.maximum(0.5 * (r0 + r1), np.where(((M - c) @ np.array([0.0, 0, 1])) / np.maximum(rm, 1e-9) > 0.55, CORNEA_CLEAR * R, 0))
        near = (r0 < 1.6 * R) & (np.linalg.norm(D, axis=1) > 2e-4) & (rm < want)
        C = np.zeros_like(P)
        Mp = c + (M - c) / np.maximum(rm, 1e-9)[:, None] * want[:, None]
        C[near] = 4 * (M[near] - Mp[near])      # P(.5) = M - C/4 = Mp
        S[f"eyeBlink{sd}"] = D - C
        S[f"corr_eyeBlink{sd}_eyeBlink{sd}"] = C
        CORRECTIVES[f"corr_eyeBlink{sd}_eyeBlink{sd}"] = (f"eyeBlink{sd}", f"eyeBlink{sd}")
    return S, dict(CORRECTIVES)
