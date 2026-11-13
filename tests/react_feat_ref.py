"""Python reference for app/react_feat.js (learned-reaction mode): the canonical frame from the SAME raw MediaPipe vectors, computed
with react/spec.py exactly as the dataset path does (seamless_lib.mp_raw_to_canon, per frame; head_t = RAW cm).

  python -I tests/react_feat_ref.py canon  <records.json> [...]   JS canonFrame output vs this reference (+ 'H' decode)
  python -I tests/react_feat_ref.py layout <layout/clip.js.json> [...]
        Python MediaPipe (same .task files, VIDEO mode) on the same PNG frames the JS workers saw: matrix layout
        (JS .data column-major?), Euler angles, blendshape names/order/values, pose + hand landmarks, canon per group.
Prints a human summary and, last line, `RESULT <json>`. Env: SPEC_DIR (default $REACT_ROOT/react: a learned-reaction checkout's react/;
not shipped with this repo), SEAMLESS_DIR (that checkout's seamless/data).
"""
import base64
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

REACT_ROOT = os.environ.get("REACT_ROOT", "")
SPEC_DIR = os.environ.get("SPEC_DIR", os.path.join(REACT_ROOT, "react") if REACT_ROOT else "")
SEAMLESS_DIR = os.environ.get("SEAMLESS_DIR", os.path.join(REACT_ROOT, "seamless", "data") if REACT_ROOT else "")
if not SPEC_DIR or not os.path.isfile(os.path.join(SPEC_DIR, "spec.py")):
    sys.exit("react_feat_ref.py needs the learned-reaction spec: set REACT_ROOT (or SPEC_DIR) to a checkout with react/spec.py")
sys.path.insert(0, SPEC_DIR)
import spec  # noqa: E402

GROUPS = {"bs": spec.SL_BS, "head_rot": spec.SL_HEAD_ROT, "head_t_cm": spec.SL_HEAD_T, "gaze": spec.SL_GAZE,
          "body": spec.SL_BODY, "hand_l": spec.SL_HAND_L, "hand_r": spec.SL_HAND_R}
H_HDR = struct.Struct("<BBHd")


def decode_h(b):
    """'H' uplink message -> (t, flags, mask(4), motion(86) with cols 55-57 in cm)."""
    typ, flags, n, t = H_HDR.unpack_from(b, 0)
    assert typ == 0x48 and n == 86 and len(b) >= 12 + 2 * n, (typ, n, len(b))
    q = np.frombuffer(b, "<i2", count=n, offset=12).astype(np.float64)
    v = q / 1e4
    v[55:58] = q[55:58] / 1e4 * 100.0
    v[60:78] = q[60:78] / 1e3
    return t, flags, np.array([(flags >> i) & 1 for i in range(4)], np.uint8), v


def col_to_row(m):
    return np.asarray(m, np.float32).reshape(4, 4).T.reshape(16)


def bs_spec(bs, names):
    bs = np.asarray(bs, np.float32)
    if not names:
        return bs[:52]
    at = {n: i for i, n in enumerate(names)}
    return np.array([bs[at[n]] if n in at else 0.0 for n in spec.BS_NAMES], np.float32)


def canon_ref(rec, names):
    """One record {w,h,face:{bs (worker order), mat (col-major)}|null, pose:{lm 132}|null, hands:[{lm 63}]} ->
    (motion86 float32, mask4) — mp_raw_to_canon for one frame, head_t = raw cm."""
    w, h = rec["w"], rec["h"]
    mot = np.zeros(86, np.float32)
    msk = np.zeros(4, np.uint8)
    f = rec.get("face")
    if f and f.get("bs") is not None and f.get("mat") is not None:
        bs = bs_spec(f["bs"], names).astype(np.float64)
        rot, tcm = spec.euler_from_mat(col_to_row(f["mat"])[None])
        mot[spec.SL_BS] = bs
        mot[spec.SL_HEAD_ROT] = rot[0]
        mot[spec.SL_HEAD_T] = tcm[0]
        mot[spec.SL_GAZE] = spec.gaze_from_bs(bs[None])[0]
        msk[0] = 1
    p = rec.get("pose")
    P = np.asarray(p["lm"], np.float32).reshape(33, 4) if p else np.zeros((33, 4), np.float32)
    body, ok = spec.body_from_pose(P[None], w, h)
    mot[spec.SL_BODY] = body[0]
    msk[1] = bool(ok[0]) and p is not None
    hands = rec.get("hands") or []
    n = min(2, len(hands))
    if p and n:
        H = [np.asarray(x["lm"], np.float32).reshape(21, 3) for x in hands[:n]]
        wl = (P[15, 0] * w, P[15, 1] * h); wr = (P[16, 0] * w, P[16, 1] * h)
        a = spec.assign_hands_to_wrists([(H[k][0, 0] * w, H[k][0, 1] * h) for k in range(n)], wl, wr)
        for side, sl, mi in (("l", spec.SL_HAND_L, 2), ("r", spec.SL_HAND_R, 3)):
            k = a[side]
            if k is None:
                continue
            mot[sl] = spec.hand_descriptor_side(H[k].astype(np.float64) * np.array([w, h, w]), side == "l")
            msk[mi] = 1
    return mot, msk


def records_from_dump(d):
    """live dump (tests/react_feat_dump.cjs live) or synthetic cases file -> list of records + names."""
    if "cases" in d:
        return d["cases"], d.get("names")
    out = []
    for c in d["canon"]:
        s = c["src"]
        pose, hand = s.get("pose"), s.get("hand")
        src = pose or hand or s.get("face") or {}
        out.append({"w": src.get("w"), "h": src.get("h"), "face": s.get("face"), "pose": pose,
                    "hands": (hand or {}).get("hands", []), "js": {"motion": c["motion"], "mask": c["mask"], "tcm": c["tcm"]},
                    "packed": c.get("packed"), "t": c["t"] / 1000.0})
    return out, d.get("names")


def via_mp_raw_to_canon(recs, names):
    """The dataset function itself on the stacked records (head_t compared after its own median / 10)."""
    try:
        sys.path.insert(0, SEAMLESS_DIR)
        import seamless_lib  # noqa: F401  (read-only import of the seamless dataset helpers)
    except Exception as e:  # pragma: no cover
        return None, f"seamless_lib unavailable: {e}"
    if not hasattr(seamless_lib, "mp_raw_to_canon"):     # renamed to the resampling mp_raw_to_30 on the seamless branch
        return None, "seamless_lib has no per-frame mp_raw_to_canon any more (spec mirror canon_ref used)"
    wh = {(r["w"], r["h"]) for r in recs if r.get("pose") or r.get("hands")}
    if len(wh) > 1:
        return None, f"several frame sizes {wh}"
    w, h = (list(wh)[0] if wh else (recs[0]["w"], recs[0]["h"]))
    N = len(recs)
    R = {"wh": np.array([w, h]), "face": np.zeros(N, np.uint8), "bs": np.zeros((N, 52), np.float32),
         "mat": np.zeros((N, 16), np.float32), "pose": np.zeros((N, 33, 4), np.float32), "pose_ok": np.zeros(N, np.uint8),
         "hands": np.zeros((N, 2, 21, 3), np.float32), "hand_n": np.zeros(N, np.uint8)}
    for i, r in enumerate(recs):
        f = r.get("face")
        if f and f.get("bs") is not None and f.get("mat") is not None:
            R["face"][i] = 1; R["bs"][i] = bs_spec(f["bs"], names); R["mat"][i] = col_to_row(f["mat"])
        if r.get("pose"):
            R["pose"][i] = np.asarray(r["pose"]["lm"], np.float32).reshape(33, 4); R["pose_ok"][i] = 1
        hs = (r.get("hands") or [])[:2]
        for k, x in enumerate(hs):
            R["hands"][i, k] = np.asarray(x["lm"], np.float32).reshape(21, 3)
        R["hand_n"][i] = len(hs)
    mot, msk = seamless_lib.mp_raw_to_canon(R)
    return (mot, msk), None


def cmd_canon(paths):
    allres = {}
    tot = {"frames": 0, "max": {g: 0.0 for g in GROUPS}, "mask_mismatch": 0, "mask_counts": [0, 0, 0, 0],
           "h_max": 0.0, "h_cm_max": 0.0, "h_clamped": 0, "h_t_err": 0.0, "h_flags_bad": 0}
    for p in paths:
        d = json.loads(Path(p).read_text())
        recs, names = records_from_dump(d)
        if not recs:
            print(f"{p}: no records"); continue
        js = np.array([r["js"]["motion"] for r in recs], np.float64)
        jm = np.array([r["js"]["mask"] for r in recs], np.uint8)
        ref = np.zeros_like(js); rm = np.zeros_like(jm)
        for i, r in enumerate(recs):
            m, k = canon_ref(r, names); ref[i] = m; rm[i] = k
            tc = r["js"].get("tcm")
            if (tc is None) != (k[0] == 0) or (tc is not None and np.abs(np.array(tc) - m[55:58]).max() > 1e-4):
                tot["mask_mismatch"] += 1
        diff = np.abs(js - ref)
        per = {g: float(diff[:, sl].max()) for g, sl in GROUPS.items()}
        mm = int((jm != rm).any(1).sum())
        res = {"frames": len(recs), "max_abs_diff": per, "mask_mismatch_frames": mm,
               "mask_counts": jm.sum(0).tolist(), "finite": bool(np.isfinite(js).all())}
        # the dataset function itself (same raw vectors, stacked)
        mr, why = via_mp_raw_to_canon(recs, names)
        if mr is not None:
            mot, msk = mr
            face = rm[:, 0].astype(bool)
            ht = spec.head_translation_feature(js[:, 55:58], face)
            ht = np.where(face[:, None], ht, 0)
            cmp = js.copy(); cmp[:, 55:58] = ht
            d2 = np.abs(cmp - mot)
            res["vs_mp_raw_to_canon"] = {g: float(d2[:, sl].max()) for g, sl in GROUPS.items()}
            res["vs_mp_raw_to_canon_mask_mismatch"] = int((msk != jm).any(1).sum())
        else:
            res["vs_mp_raw_to_canon"] = why
        # the 'H' bytes the page produced (JS packHuman) decoded here
        hs = [r for r in recs if r.get("packed")]
        if hs:
            hm, hcm, hcl, hte, hfb = 0.0, 0.0, 0, 0.0, 0
            for r in hs:
                t, fl, mk, v = decode_h(base64.b64decode(r["packed"]))
                x = np.array(r["js"]["motion"], np.float64)
                lim = np.full(86, 3.2767); lim[55:58] = 327.67; lim[60:78] = 32.767
                cl = np.abs(x) > lim
                hcl += int(cl.sum())
                e = np.abs(v - np.clip(x, -lim, lim))
                e[60:78] /= 10.0                  # body travels x1e3: its quantization is 10x coarser (<= 5e-4)
                hm = max(hm, float(np.delete(e, [55, 56, 57]).max())); hcm = max(hcm, float(e[55:58].max()))
                hte = max(hte, abs(t - r["t"]) if r.get("t") is not None else 0.0)
                hfb += int((mk != np.array(r["js"]["mask"], np.uint8)).any())
            res["H"] = {"n": len(hs), "max_err": hm, "max_err_cm": hcm, "clamped_values": hcl, "t_err": hte, "flags_bad": hfb}
            tot["h_max"] = max(tot["h_max"], hm); tot["h_cm_max"] = max(tot["h_cm_max"], hcm); tot["h_clamped"] += hcl
            tot["h_t_err"] = max(tot["h_t_err"], hte); tot["h_flags_bad"] += hfb
        allres[Path(p).name] = res
        tot["frames"] += len(recs); tot["mask_mismatch"] += mm
        for g in GROUPS:
            tot["max"][g] = max(tot["max"][g], per[g])
        tot["mask_counts"] = [a + b for a, b in zip(tot["mask_counts"], res["mask_counts"])]
        print(f"{Path(p).name}: {json.dumps(res)}")
    print("RESULT " + json.dumps({"files": allres, "total": tot}))


# ------------------------------------------------------------------------------------------------ layout proof
def run_python_mp(frames_dir, n, models_dir):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    m = Path(models_dir)
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(m / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))
    pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(m / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    gr = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(m / "gesture_recognizer.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    out, names = [], None
    for i in range(n):
        img = cv2.cvtColor(cv2.imread(str(Path(frames_dir) / f"{i + 1:04d}.png")), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
        ts = int(round(i * 1000 / 30))
        rec = {"w": W, "h": H}
        r = fl.detect_for_video(im, ts)
        if r.face_landmarks:
            if names is None:
                names = [c.category_name for c in r.face_blendshapes[0]]
            rec["face"] = {"bs": [c.score for c in r.face_blendshapes[0]],
                           "mat_row": np.asarray(r.facial_transformation_matrixes[0], np.float32).reshape(16).tolist()}
        else:
            rec["face"] = None
        rp = pl.detect_for_video(im, ts)
        rec["pose"] = {"lm": [v for q in rp.pose_landmarks[0] for v in (q.x, q.y, q.z, q.visibility)]} if rp.pose_landmarks else None
        rh = gr.recognize_for_video(im, ts)
        rec["hands"] = [{"lm": [v for q in lm for v in (q.x, q.y, q.z)]} for lm in rh.hand_landmarks]
        out.append(rec)
    for o in (fl, pl, gr):
        o.close()
    return out, names


def stats(a):
    a = np.abs(np.asarray(a, np.float64)).ravel()
    return {"max": float(a.max()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))} if a.size else None


def cmd_layout(paths):
    models = os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "app/vision"))
    allres = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        clip, n = d["clip"], d["n"]
        frames_dir = Path(p).resolve().parent.parent / "frames" / clip
        py, pnames = run_python_mp(frames_dir, n, models)
        jn = d["names"]
        res = {"clip": clip, "frames": n,
               "names_js_eq_python": jn == pnames, "names_js_eq_spec": jn == spec.BS_NAMES, "names_python_eq_spec": pnames == spec.BS_NAMES}
        both = [i for i in range(n) if d["frames"][i].get("face") and py[i]["face"]]
        res["face_present"] = {"js": sum(1 for f in d["frames"] if f.get("face")), "python": sum(1 for r in py if r["face"]), "both": len(both)}
        jm = np.array([d["frames"][i]["face"]["mat"] for i in both], np.float64)
        pm = np.array([py[i]["face"]["mat_row"] for i in both], np.float64)
        jrow = jm.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)       # JS data read as COLUMN-major
        res["mat_diff_if_js_col_major"] = stats(jrow - pm)
        res["mat_diff_if_js_row_major"] = stats(jm - pm)
        ej, tj = spec.euler_from_mat(jrow); ep, tp = spec.euler_from_mat(pm)
        er, _ = spec.euler_from_mat(jm)                                       # the wrong (row-major) reading
        res["euler_python_range_deg"] = {k: [round(float(np.degrees(ep[:, j].min())), 1), round(float(np.degrees(ep[:, j].max())), 1)]
                                         for j, k in enumerate(("yaw", "pitch", "roll"))}
        res["euler_diff_rad_col_major"] = {k: stats(ej[:, j] - ep[:, j]) for j, k in enumerate(("yaw", "pitch", "roll"))}
        res["euler_diff_rad_row_major"] = {k: stats(er[:, j] - ep[:, j]) for j, k in enumerate(("yaw", "pitch", "roll"))}
        res["euler_corr_col_major"] = {k: float(np.corrcoef(ej[:, j], ep[:, j])[0, 1]) for j, k in enumerate(("yaw", "pitch", "roll"))}
        res["euler_corr_row_major"] = {k: float(np.corrcoef(er[:, j], ep[:, j])[0, 1]) for j, k in enumerate(("yaw", "pitch", "roll"))}
        res["head_t_cm_diff"] = stats(tj - tp)
        bj = np.array([bs_spec(d["frames"][i]["face"]["bs"], jn) for i in both], np.float64)
        bp = np.array([bs_spec(py[i]["face"]["bs"], pnames) for i in both], np.float64)
        res["bs_diff_by_name"] = stats(bj - bp)
        res["bs_diff_by_position_py"] = stats(np.array([d["frames"][i]["face"]["bs"][:52] for i in both]) - np.array([py[i]["face"]["bs"][:52] for i in both]))
        # pose / hand landmarks + canonical frames from both runtimes (each through the same reference formula)
        pb = [i for i in range(n) if d["frames"][i].get("pose") and py[i]["pose"]]
        if pb:
            res["pose_lm_xy_diff"] = stats(np.array([np.array(d["frames"][i]["pose"]["lm"]).reshape(33, 4)[11:17, :2] for i in pb]) -
                                           np.array([np.array(py[i]["pose"]["lm"]).reshape(33, 4)[11:17, :2] for i in pb]))
        hj = sum(len(f["hand"]["hands"]) for f in d["frames"] if f.get("hand")); hp = sum(len(r["hands"]) for r in py)
        res["hands_detected"] = {"js": hj, "python": hp}
        recJ, recP = [], []
        for i in range(n):
            f = d["frames"][i]
            fj = f.get("face"); fp_ = py[i]["face"]
            recJ.append({"w": py[i]["w"], "h": py[i]["h"], "face": {"bs": fj["bs"], "mat": fj["mat"]} if fj else None,
                         "pose": f.get("pose"), "hands": (f.get("hand") or {}).get("hands", [])})
            recP.append({"w": py[i]["w"], "h": py[i]["h"],
                         "face": {"bs": bs_spec(fp_["bs"], pnames).tolist(), "mat": np.asarray(fp_["mat_row"]).reshape(4, 4).T.reshape(16).tolist()} if fp_ else None,
                         "pose": py[i]["pose"], "hands": py[i]["hands"]})
        CJ = [canon_ref(r, jn) for r in recJ]; CP = [canon_ref(r, None) for r in recP]
        mJ = np.array([c[0] for c in CJ], np.float64); mP = np.array([c[0] for c in CP], np.float64)
        kJ = np.array([c[1] for c in CJ]); kP = np.array([c[1] for c in CP])
        res["canon_js_vs_python_runtime"] = {}
        for g, sl in GROUPS.items():
            mi = {"bs": 0, "head_rot": 0, "head_t_cm": 0, "gaze": 0, "body": 1, "hand_l": 2, "hand_r": 3}[g]
            sel = (kJ[:, mi] > 0) & (kP[:, mi] > 0)
            res["canon_js_vs_python_runtime"][g] = dict(stats(mJ[sel][:, sl] - mP[sel][:, sl]) or {}, n=int(sel.sum()))
        res["mask_agree"] = [float((kJ[:, j] == kP[:, j]).mean()) for j in range(4)]
        allres[clip] = res
        print(json.dumps(res, indent=1))
    print("RESULT " + json.dumps(allres))


if __name__ == "__main__":
    cmd, files = sys.argv[1], sys.argv[2:]
    {"canon": cmd_canon, "layout": cmd_layout}[cmd](files)
