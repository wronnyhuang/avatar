"""rig: build Juno's page/renderer JSON from the asset + rig.py (numbers are never hand-edited).

  python renderer/humans/tools/juno_body_build.py [--asset DIR] [--reach]
writes renderer/humans/juno/:
  juno.body.json config.body (same keys as animals/muse.body.json, docs/CONTRACT--rig.md s3)
  juno.json                 the character config (camera / lights = muse.json's set, framed for her)
  juno.face.json            the param -> blendshape table rig.py reads
  juno.poses.json           lip viseme poses (lip_* columns; start = Muse's, tuned for a human mouth)
  juno.props.override.json  the override of the shared muse.props.json (renderer block, worn-prop fit, reach map with --reach)
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUM = os.path.dirname(HERE)
RR = os.path.dirname(HUM)
sys.path.insert(0, RR)
from anim.schema import IDX, NPARAMS  # noqa: E402
from humans import rig as RIG  # noqa: E402

NAME = "juno"         # --name junor writes renderer/humans/junor/junor.* (juno stays the default, byte-identical)
OUT = os.path.join(HUM, "juno")
H = 2.30
CAM_IN = 4.0          # rig2: Juno's cam_zoom -1 = 4x tighter (Muse/rt_engine CAM_IN 2): a human is ~1/7 head
HOME_HEAD = 0.40      # home close-up (RIG3: a head-and-shoulders bust like the reference card; RIG2 0.28): crown-to-chin = this share of the frame height ...
CHALK_K = 0.8         # rig3: Juno's chalk = Muse's x this (a real stick in her hand)
HOME_MARGIN = 0.12    # the page tests the WRIST point against the frame shrunk by this share ( body: junor 0.36, her open hand is ~0.1 H)
HOME_EYE = 1.0/3.0    # ... with the eye line this far down from the top (the upper third)


def live_cam():
    """rig3: Juno's LIVE camera (renderer/animals/juno.json camera: her own eye-level position); Muse's if absent"""
    for n in (NAME, "muse"):
        p = os.path.join(RR, "animals", n + ".json")
        if os.path.isfile(p):
            c = json.load(open(p)).get("camera")
            if c and "pos" in c:
                return c
    raise SystemExit("no camera")


def restation(c, place, cam0, cam1):
    """rig3: a station camera {cam_zoom, cam_x, cam_y} set for Muse's camera cam0 -> the same aim for Juno's cam1:
    the aim ray through the place (its surface centre's depth) hits the same point.  camera_for aims at
    target + H*(cam_x, cam_y, 0) (no dolly; zoom <= 0), so only cam_x / cam_y change."""
    g = place.get("plane") or place.get("screen")
    if not g or not cam0 or not cam1:
        return c
    o, u, v, sz = (np.array(g[k], float) if k != "size" else g[k] for k in ("origin", "u", "v", "size"))
    zp = float((o + 0.5*u*sz[0] + 0.5*v*sz[1])[2])*H
    p0, t0 = np.array(cam0["pos"], float), np.array(cam0["target"], float)
    p1, t1 = np.array(cam1["pos"], float), np.array(cam1["target"], float)
    a0 = t0 + H*np.array([float(c.get("cam_x", 0)), float(c.get("cam_y", 0)), 0.0])
    q = p0 + (a0 - p0)*((zp - p0[2])/(a0[2] - p0[2]))                 # the point the old aim looks at, on the place
    a1 = p1 + (q - p1)*((t1[2] - p1[2])/(q[2] - p1[2]))                # Juno's aim through it, on her target plane
    return dict(c, cam_x=r4((a1[0] - t1[0])/H), cam_y=r4((a1[1] - t1[1])/H))


def r4(v, n=4):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r4(x, n) for x in v]
    return round(float(v), n)


def default_asset():
    """the committed asset (renderer/humans/juno/juno.asset), else the newest T/asset-src/v<N>/juno.asset"""
    p = os.path.join(OUT, "juno.asset")
    if os.path.isfile(os.path.join(p, "skeleton.json")):
        return p
    src = os.environ.get("JUNO_ASSET_SRC", "asset-src")
    vs = sorted((int(d[1:]), d) for d in os.listdir(src) if d[:1] == "v" and d[1:].isdigit()
                and os.path.isfile(os.path.join(src, d, "juno.asset", "skeleton.json"))) if os.path.isdir(src) else []
    if vs:
        return os.path.join(src, vs[-1][1], "juno.asset")
    raise SystemExit("no asset: pass --asset")


def body_json(rig):
    S = rig.stature
    f = rig.solve(np.zeros(NPARAMS))
    J = f.joints
    bi = rig.bi
    Ploc = f.bones_local[:, :3, 3].astype(float)
    pel = Ploc[bi["pelvis"]]; waist = Ploc[bi["spine_01"]]; neck = Ploc[bi["neck_01"]]
    sh = [J["shoulder_l"], J["shoulder_r"]]; wr = [rig.wr_hang[0], rig.wr_hang[1]]
    hip = [J["hip_l"], J["hip_r"]]; ank = [J["ankle_l"], J["ankle_r"]]
    allp = np.concatenate([f.proxies_local[:, 0:3], f.proxies_local[:, 3:6]])
    rr = np.concatenate([f.proxies_local[:, 6], f.proxies_local[:, 6]])[:, None]
    lo = (allp - rr).min(0); hi = (allp + rr).max(0)
    lo[1] = 0.0; hi[1] = max(hi[1], S)
    com = np.array([0.0, 0.55*S, 0.0])
    bc = 0.5*(lo + hi); brad = float(np.linalg.norm(hi - lo)*0.5 + 0.05)
    hc = rig.head_c; hrad = rig.head_radii
    L = rig.L
    th, shn = rig.l_th, rig.l_sh
    sole = float(ank[0][1])
    foot_i = bi["foot_r"]; ball_i = bi["ball_r"]
    foot_len = float(np.linalg.norm(rig.r[ball_i] - rig.r[foot_i])*1.55)
    # sitting: the seat touches ~0.10 su below the hip joints (buttocks), a little behind them
    contact = np.array([0.0, float(hip[0][1]) - 0.048*S, -0.02*S])
    seat = 0.44                      # the shared chair's seat (muse.props.json chair_seat top_y 0.1913 H)
    ry = (seat - contact[1])/H
    hipY = float(hip[0][1]) + ry*H
    # feet on the floor in front of the seat, shins near vertical: knee = hip + thigh forward (rising if the seat is low)
    dy = hipY - sole
    if dy < shn:
        kz = np.sqrt(max(th*th - (shn - dy)**2, 0.0))
    else:
        kz = th
    feet = {}
    for i, sd in enumerate(("l", "r")):
        tgt = np.array([ank[i][0], sole, hip[i][2] + kz])      # local, after the root lowered by ry
        off = (tgt - ank[i])/H
        off[1] = 0.0
        feet["foot_" + sd] = r4(off)
    # face points (head space, x outward on the hand's side, scene units): where a palm goes for a face touch
    lm = dict(rig.meta.get("landmarks") or {})
    k = rig.k
    eye = rig.r[bi["eye_l"]] - hc
    mouth = (np.array(lm["mouth"])*k - hc) if "mouth" in lm else np.array([0.0, eye[1] - 0.45*hrad[1], eye[2]])
    chin = rig.chin - hc
    mdrop = mfwd = 0.0
    face = {"units": "head space: origin = head_scene (the head centre at rest), x OUTWARD on the hand's own side, y up, "
                     "z out of the face; scene units. Where a palm goes for a face touch; the rig rests it on the face.",
            "mouth": r4([0.035, mouth[1] - mdrop, mouth[2] + 0.02 + mfwd]), "mouth_mid": r4([0.0, mouth[1], mouth[2] + 0.02]),
            "chin": r4([0.04, chin[1] - 0.01, chin[2] - 0.005]), "eye": r4([abs(eye[0]), eye[1], eye[2] + 0.03]),
            "cheek": r4([0.72*hrad[0], 0.5*(eye[1] + mouth[1]), 0.62*hrad[2]]),
            "temple": r4([0.95*hrad[0], eye[1] + 0.30*hrad[1], 0.25*hrad[2]]),
            "brow": r4([abs(eye[0]), eye[1] + 0.22*hrad[1], eye[2] + 0.015]),
            "neck_pivot_scene": r4(neck),
            "note": " rig: generated by renderer/humans/tools/juno_body_build.py from the asset landmarks + rig.py"}
    hm = rig.hand_map
    b = {
        "character": NAME,
        "units": "H = 2.30 scene units = the SHARED motion unit (Muse's; root_*, foot_*, props and places are in it). "
                 "Juno's real stature is stature_scene. World frame = (p_scene - origin_scene)/H, x screen-right, y up, z toward the camera",
        "H": H, "stature_scene": r4(S), "origin_scene": [0.0, 0.0, 0.0], "base_y": 0.0, "floor_y": 0.0,
        "com": r4(com/H), "com_scene": r4(com),
        "extents": {"x": r4(max(-lo[0], hi[0])/H), "y_down": r4(com[1]/H), "y_up": r4((hi[1] - com[1])/H),
                    "z_back": r4(-lo[2]/H), "z_front": r4(hi[2]/H), "radius": r4(brad/H)},
        "bound": {"center_scene": r4(bc), "radius_scene": r4(brad)},
        "head_scene": r4(hc), "head_radii_scene": r4(hrad),
        "squash_k": RIG.SQUASH_K, "squash_human": rig.tun["squash_human"],
        "joints_scene": {"pelvis": r4(pel), "waist": r4(waist), "neck": r4(neck),
                         "shoulder_l": r4(sh[0]), "shoulder_r": r4(sh[1]), "wrist_rest_l": r4(wr[0]), "wrist_rest_r": r4(wr[1]),
                         "hip_l": r4(hip[0]), "hip_r": r4(hip[1]), "ankle_rest_l": r4(ank[0]), "ankle_rest_r": r4(ank[1])},
        "arm": {"upper": r4(rig.l_up), "fore": r4(rig.l_fo), "length_scene": r4(L), "length_H": r4(L/H),
                "grasp_n_H": r4(RIG.GRASP_OFS/H, 5), "paw_r_H": r4(rig.tun["palm_half_thick"]*S/H, 5),   # rig2: moves.js GRASP_N / PAW_R
                "hand_target": "upper-body space (spine_03's): wrist = wrist_rest + (s*(hx>0 ? gain_x_out : gain_x_in)*L*hx, "
                               "gain_y*L*hy, gain_z*L*hz), s = -1 screen-left; |x| >= min_x; pushed out of the chest ellipsoid; an "
                               "EMPTY hand coming to the face rests on its surface in front (head inflated by the hand); reach "
                               "clamped to 0.995 L; clavicle shares far reaches; rig.py JunoRig._clavicle",
                "grasp": "grasp = palm contact + 0.022 * palm normal; frame columns = across the palm, toward the fingertips, "
                         "out of the palm (museGrasp's convention)", "grasp_offset_scene": RIG.GRASP_OFS,
                "palm_centre_to_grasp_scene": r4(rig.palm_local[0][2] + RIG.GRASP_OFS)},
        "leg": {"thigh": r4(th), "shin": r4(shn), "length_scene": r4(th + shn), "length_H": r4((th + shn)/H),
                "hip_width_scene": r4(abs(hip[1][0] - hip[0][0])), "hip_width_H": r4(abs(hip[1][0] - hip[0][0])/H),
                "foot_len_scene": r4(foot_len), "foot_width_scene": r4(0.045*S), "sole_below_ankle_scene": r4(sole),
                "foot_target": "ankle = ankle_rest + root_linear^-1 * (H * foot_{l,r}_{x,y,z}) (WORLD-axis offsets in H, as Muse); "
                               "knees bend forward (pole slightly out); reach clamped to 0.995*(thigh+shin)"},
        "stance": {"feet_x_scene": r4(abs(ank[1][0] - ank[0][0])), "feet_x_H": r4(abs(ank[1][0] - ank[0][0])/H),
                   "standing_hip_y_scene": r4(hip[0][1])},
        "step": {"max_comfortable_H": 0.17, "max_H": 0.22, "foot_lift_H": 0.04, "cadence_hz": 1.8,
                 "notes": "a grown woman's walk in the shared H unit: steps ~0.15-0.17 H (0.35-0.39 su), lift ~0.04 H"},
        "sitting": {"seat_contact_local_scene": r4(contact), "seat_contact_H": r4(contact[1]/H),
                    "root_y_for_seat": f"root_y = (seat_height_scene - {contact[1]:.3f})/2.30 (seat top above the floor, scene units)",
                    "feet_for_thighs_horizontal_H": feet,
                    "hip_height_above_seat_scene": r4(hip[0][1] - contact[1]), "knee_forward_of_hip_scene": r4(th),
                    "notes": "the shared chair (seat 0.44 su) is low for her: knees sit a little above the hips, feet flat in front"},
        "hand_map": {"rest_wrist_scene": r4(wr[1]), "L_scene": r4(L), "gain_x_out": hm["gain_x_out"], "gain_x_in": hm["gain_x_in"],
                     "gain_y": hm["gain_y"], "gain_z": hm["gain_z"], "min_x_scene": r4(hm["min_x_scene"]),
                     "palm_above_wrist_scene": r4(rig.palm_above_wrist),
                     "note": "rig.py, numbers for code: upper-body wrist = rest_wrist + (s*gain_x*L*hx, gain_y*L*hy, gain_z*L*hz), "
                             "s = -1 screen-left; gain_x = gain_x_out for hx > 0 else gain_x_in; |x| >= min_x"},
        "face": face,
        "rig_ranges": {"hand_l_y": [-1, 1.2], "hand_r_y": [-1, 1.2]},
        "gesture_tuning": {"nod": {"amp": 1.6, "decay": 0.9, "cycles": 3,
                                   "note": ": a human head framed full-body; between the seated characters' 1.0 and Muse's 2.4"},
                           "think": {"amp": 1.6}},
        # speech gestures (docs/GESTURE-RUNTIME.md, anim/gesture_mix.py): gain / cap / clamp of the speech -> arm net's
        # offsets for Juno; hand_*_y never below -0.22 (below about -0.3 the IK bends her knees)
        "gesture": JUNO_GESTURE,
        "stand": False,
        "views": home_view(rig),
        "stand_note": "Juno STANDS by default (root_y 0 = standing on her feet); the 'stand' param is ignored",
        "notes": " rig: generated by renderer/humans/tools/juno_body_build.py from the asset skeleton + renderer/humans/rig.py",
    }
    return b


def home_view(rig, cam=None):
    """rig2: the HOME VIEW (app/moves.js reads body.views.home): a medium close-up while she stands still, eased
    out whenever a raised hand / held thing would leave it or a move / walk / sit / object job runs (docs/CONTRACT--rig.md
    s7).  Numbers from the asset (crown, chin, eyes) and the rest camera; all in H (the shared motion unit), world at
    the character's depth.  The page aims the frame at the live head; cam_zoom is the renderer's (camera.cam_in)."""
    import math
    cam = cam or live_cam()
    f = rig.solve(np.zeros(NPARAMS))
    eye_y = float(f.eyes[:, 1].mean())
    lm = rig.meta.get("landmarks", {}) if hasattr(rig, "meta") else {}
    crown = float(lm["head_top"][1])*rig.k if "head_top" in lm else float(rig.head_c[1] + rig.head_radii[1])
    chin = float(lm["chin"][1])*rig.k if "chin" in lm else float(rig.head_c[1] - rig.head_radii[1])
    vh = (crown - chin)/HOME_HEAD                                        # frame height at her depth, scene units
    d = float(np.linalg.norm(np.subtract(cam["target"], cam["pos"])))
    vh0 = 2.0*d*math.tan(math.radians(cam["fov"])*0.5)                    # the rest view's height there
    kz = vh0/vh
    zoom = -math.log(kz)/math.log(CAM_IN)
    cy = eye_y - (0.5 - HOME_EYE)*vh                                      # frame centre (scene y): the eyes above it
    return {"home": {"cam_zoom": r4(zoom), "center_y": r4(cy/H), "head_y": r4(float(rig.head_c[1])/H),
                     "half_h": r4(0.5*vh/H), "half_w": r4(0.5*vh/H*16.0/9.0), "target_y": r4(float(cam["target"][1])/H),
                     "target_x": r4(float(cam["target"][0])/H), "margin": HOME_MARGIN, "settle_s": 0.5, "in_hz": 0.45, "out_hz": 0.9,
                     "head_frac": HOME_HEAD, "eye_from_top": round(HOME_EYE, 4),
                     "note": " rig2: medium close-up (crown-chin = head_frac of the frame, eyes eye_from_top down). "
                             "center_y / head_y: the frame centre and the head centre at rest (H); the page keeps their "
                             "difference while aiming at the live head. Raised hands (above the frame bottom) and held things "
                             "must stay inside half_w / half_h shrunk by margin, else it eases out (out_hz) to the follow "
                             "framing; back in (in_hz) after settle_s still"}}


def char_json(rig):
    m = json.load(open(os.path.join(RR, "animals", "muse.json")))
    c = copy.deepcopy(m)
    c["char_defines"] = {"CHAR_ROOT": 1, "CHAR_MARKS": 1, "CHAR_MESH": 1}
    c["mesh"] = {"asset": f"humans/{NAME}/{NAME}.asset", "rig": "humans/rig.py", "lod_cloud": "lod0", "lod_hub": "lod1"}
    lc = live_cam()                      # rig3: her own eye-level camera (animals/juno.json)
    c["camera"].update({k: lc[k] for k in ("pos", "target", "fov") if k in lc})
    c["camera"]["cam_in"] = CAM_IN       # rig2: rt_engine.camera_for push-in base (cam_zoom -1 = 4x; Muse 2x)
    c["note"] = (" rig: Muse's set and lights (the SAME teacher scene); camera = animals/juno.json's (rig3: at her eye level); Juno is ~0.9 of Muse's "
                 "height, so the rest framing keeps Muse's camera (room above her head for jumps)")
    return c


def poses_json():
    m = json.load(open(os.path.join(RR, "animals", "muse.poses.json")))
    out = {"poses": copy.deepcopy(m["poses"]), "gains": {}, "speed": 1.0,
           "note": " rig: Muse's viseme poses in lip_* columns; rig.py maps them to ARKit shapes (bilabial priority)"}
    return out


def props_override(rig, reach=None):
    hr = rig.head_radii
    f = float(hr[0]*1.15/0.70)              # Juno's head half-width vs Muse's (0.70): worn props scale with it
    pr = json.load(open(os.path.join(RR, "animals", "muse.props.json")))
    byid = {p["id"]: p for p in pr["props"]}
    hc = rig.head_c
    eye = rig.r[rig.bi["eye_l"]] - hc
    ov = {"base": "muse", "_doc": "docs/CONTRACT--rig.md s4: deep-merged over muse.props.json at load time "
                                  "(renderer/humans/charfiles.py); lists of objects merge by id",
          "character": NAME,
          "renderer": {"grasp_fn": "junoGrasp", "head_fn": "junoHeadFrame", "grasp_offset": RIG.GRASP_OFS, "char_bound": None,
                       "notes": ": the grasp / head frames come from rig.py by uniform (uJGrasp / uJHead)"},
          "props": []}
    hp = byid.get("headphones")
    if hp:
        fit = hp.get("fit", {})
        ov["props"].append({"id": "headphones", "size": r4(hp["size"]*f), "scale": r4(hp.get("scale", 1)*f),
                            "fit": {"w": r4(fit.get("w", 0.75)*f), "arc": fit.get("arc", 0.72), "bz": r4(fit.get("bz", 0.19)*f),
                                    "w_rest": r4(fit.get("w_rest", 0.42)*f)},
                            "worn": {"neck": {"y": r4(hp["worn"]["neck"]["y"]*f), "rx": hp["worn"]["neck"]["rx"]}}})
    hat = byid.get("hat")
    if hat:
        ov["props"].append({"id": "hat", "size": r4(hat["size"]*f*1.05), "scale": r4(hat.get("scale", 1)*f*1.05),
                            "afford": {"wear": {"offset": [0.0, r4((hr[1]*0.95)/H), r4(-0.05*hr[2]/H)]}}})
    ch = byid.get("chalk")
    if ch:          # rig3: Muse's chalk is ~11 cm x 2.6 cm at Juno's scale (RENDER3: "oversized, lies across the palm
        k = CHALK_K  # into the wrist"); a real stick is ~8 x 1 cm: scaled with its tip (teach.js reads tip.grasp_to_tip_H)
        tip = dict(ch.get("tip") or {})
        e = {"id": "chalk", "size": r4(ch["size"]*k), "scale": r4(ch.get("scale", ch["size"])*k)}
        if tip:
            e["tip"] = {"grasp_to_tip_scene": r4(np.multiply(tip.get("grasp_to_tip_scene", [0, 0, 0]), k)),
                        "grasp_to_tip_H": r4(np.multiply(tip.get("grasp_to_tip_H", [0, 0, 0]), k))}
        ov["props"].append(e)
    gl = byid.get("glasses")
    if gl:
        ov["props"].append({"id": "glasses", "size": r4(gl["size"]*f), "scale": r4(gl.get("scale", 1)*f),
                            "afford": {"wear": {"offset": [0.0, r4(eye[1]/H), r4((eye[2] + 0.025)/H)]}}})
    # the board band she writes in: taller than Muse, she reaches higher and stoops for the lowest rows -> the brain
    # lays out from near the top down to hip height (uv-y; board top 0.8565 H, height 0.6087 H)
    bd = [p for p in pr["places"] if p["id"] == "board"]
    if bd and "station" in bd[0]:
        top, hgt = 0.8565, 0.6087
        pl = bd[0].get("plane") or {}
        if pl.get("origin") and pl.get("size"):
            top, hgt = float(pl["origin"][1]), float(pl["size"][1])
        v0, v1 = 0.12, 0.72
        ov["places"] = [{"id": "board", "station": {"reach_uv_y": [v0, v1], "y_range_H": [r4(top - v1*hgt), r4(top - v0*hgt)],
                                                    "note_t4430": "Juno: writes uv-y 0.12-0.72 (Muse 0.32-0.95)"}}]
    # rig2: Juno's push-in base is CAM_IN (4x at -1) where the station cameras were set for 2x: the same framing
    # = cam_zoom * ln 2 / ln CAM_IN (pull-outs unchanged)
    import math
    k = math.log(2.0)/math.log(CAM_IN)
    pls = {p["id"]: p for p in ov.get("places", [])}
    for p in pr["places"]:
        c = p.get("camera")
        if isinstance(c, dict) and float(c.get("cam_zoom", 0)) < 0:
            e = pls.get(p["id"])
            if e is None:
                e = pls[p["id"]] = {"id": p["id"]}; ov.setdefault("places", []).append(e)
            c = restation(c, p, json.load(open(os.path.join(RR, "animals", "muse.json")))["camera"], live_cam())
            e["camera"] = dict(c, cam_zoom=r4(float(c["cam_zoom"])*k),
                               note_t4430="rescaled for camera.cam_in %g; aim re-derived for Juno's camera (rig3)" % CAM_IN)
    if reach:
        ov["reach"] = reach
    return ov


def asset_sig(asset_dir):
    """the skeleton.json fingerprint the reach map was solved on (tests skip a map built from another asset)"""
    import hashlib
    return hashlib.sha1(open(os.path.join(asset_dir, "skeleton.json"), "rb").read()).hexdigest()[:12]


def measure_reach(rig):
    gx = np.linspace(-1.0, 1.2, 12)
    out = {}
    for side, pre in ((0, "hand_l_"), (1, "hand_r_")):
        Ps, Ns = [], []
        for x in gx:
            for y in gx:
                for z in gx:
                    v = np.zeros(NPARAMS); v[IDX[pre + "x"]] = x; v[IDX[pre + "y"]] = y; v[IDX[pre + "z"]] = z
                    f = rig.solve(v)
                    G = f.grasp_local[side].astype(float)
                    Ps.append((G[:3, 3] - G[:3, 2]*RIG.GRASP_OFS)/H); Ns.append(G[:3, 2])
        P = np.array(Ps); N = np.array(Ns)
        out["lr"[side]] = {"param": [pre + "x", pre + "y", pre + "z"], "axes": [gx.round(4).tolist()]*3,
                           "shape": [12, 12, 12, 3], "layout": "flat, C order: index ((i*ny + j)*nz + k)*3 + c",
                           "pos": [round(float(v), 4) for v in P.reshape(-1)],
                           "palm_normal": [round(float(v), 2) for v in N.reshape(-1)]}
    out["asset_sig"] = asset_sig(rig.asset_dir)
    out["notes"] = (" rig: Juno's palm contact (grasp - 0.022 palm normal), world H at the rest body pose, from rig.py's "
                    "own forward solve (exact, not measured); same layout as Muse's")
    return out


# speech gestures: Juno's block (tuned by verify on GPU renders; code default anim/gesture_mix.py DEFAULTS["juno"])
JUNO_GESTURE = {"gain_xyz": 2.8, "gain_other": 1.0, "max_xyz": 0.50, "max_other": 0.6,   # verify: knob 0.5 ~ raw human scale (lead 11-12)
                "clamp": {"hand_l_x": [-0.25, 0.45], "hand_r_x": [-0.25, 0.45], "hand_l_y": [-0.22, 0.40],
                          "hand_r_y": [-0.22, 0.40], "hand_l_z": [-0.15, 0.45], "hand_r_z": [-0.15, 0.45]}}


def main():
    global NAME, OUT, CAM_IN, HOME_HEAD, HOME_MARGIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--reach", action="store_true")
    ap.add_argument("--name", default="juno", help=": character id (junor -> renderer/humans/junor/junor.*)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    NAME = a.name
    OUT = os.path.join(HUM, NAME)
    if NAME != "juno":   # body: the realistic Juno's home close-up: head 41 % of the frame, eyes on the upper third.
        # cam_in 4 needed cam_zoom -1.03 for her .40 (the renderer clamps at -1: the frame sat low, eyes at .375); 4.6 =
        # the same push-in range with headroom (renderer/animals/<name>.json camera.cam_in must match: the build writes it)
        CAM_IN, HOME_HEAD = 4.6, 0.41
        HOME_MARGIN = 0.36   # the wrist test must leave room for the adult hand (a wave's fingers left the frame at .12)
    a.out = a.out or OUT
    asset = a.asset or (default_asset() if NAME == "juno" else os.path.join(OUT, f"{NAME}.asset"))
    os.makedirs(a.out, exist_ok=True)
    cj = char_json(None)
    rig = RIG.JunoRig(asset, char_cfg=cj)
    b = body_json(rig)
    json.dump(b, open(os.path.join(a.out, f"{NAME}.body.json"), "w"), indent=1)
    json.dump(cj, open(os.path.join(a.out, f"{NAME}.json"), "w"), indent=1)
    fj = os.path.join(a.out, f"{NAME}.face.json")
    if NAME == "juno" or not os.path.isfile(fj):     # never clobber a tuned junor.face.json with stale defaults
        json.dump(RIG.DEFAULT_FACE, open(fj, "w"), indent=0)
    json.dump(poses_json(), open(os.path.join(a.out, f"{NAME}.poses.json"), "w"), indent=1)
    reach = measure_reach(rig) if a.reach else None
    if reach is None:
        old = os.path.join(a.out, f"{NAME}.props.override.json")
        if os.path.isfile(old):
            reach = json.load(open(old)).get("reach")
    ov = props_override(rig, reach)
    txt = json.dumps({k: v for k, v in ov.items() if k != "reach"}, indent=1)
    if reach:
        txt = txt[:-2] + ',\n "reach": ' + json.dumps(reach, separators=(",", ":")) + "\n}"
    open(os.path.join(a.out, f"{NAME}.props.override.json"), "w").write(txt)
    print("wrote", a.out, "asset", asset, "stature", rig.stature, "L", round(rig.L, 3), "reach", bool(reach))


if __name__ == "__main__":
    main()
