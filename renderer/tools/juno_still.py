#!/usr/bin/env python3
"""render: one still of Juno in the shared scene (docs/CONTRACT--render.md section 6).  STABLE CLI.

  python renderer/tools/juno_still.py --out x.png [--w 1920 --h 1080] [--cam face|card|bust|mid|hands|full|scene]
      [--pose NAME] [--expr NAME] [--params shot.json | P.npy | P.json] [--frame N] [--asset DIR] [--lod lod0|lod1]
      [--device cpu|gpu] [--threads 16] [--no-scene] [--alpha-out a.png] [--json meta.json] [--preset full1]

--params: (a) JUDGE's shot params {"blendshapes":{ARKit/extras: w}, "head_deg":{"yaw","pitch","roll"},
  "gaze_deg":{"yaw","pitch"}} (a whole shots.json entry with a "params" key also works) -> applied straight to the
  asset's shapes + head/eye bones; (b) a control vector (.npy of NPARAMS, or a json {param: value}) -> the engine's rig.
--cam card = the reference card framing (use --w 480 --h 836): head+ears 58 % of the width, crown 10 %, eye line 35 %,
  chin 57.5 % of the height, eye-level, long lens.  scene/full = the live camera of juno.json.
--no-scene: RGBA, the character alone (alpha = coverage) over transparent; --alpha-out writes the character's alpha
  next to a normal render.  CPU = Mesa llvmpipe (nice it yourself), GPU = flock the K620 lock yourself.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), os.path.join(ROOT, "mesh")]

CAMS = ("face", "card", "bust", "bust3", "mid", "hands", "full", "scene")


def _head_points(asset, rf):
    """posed (scene) crown, chin, eye centre and ear-to-ear width from the asset landmarks + the head bone."""
    bi = {b["name"]: i for i, b in enumerate(asset.bones)}
    K = rf.bones[bi["head"]].astype(np.float64) @ asset.inv_bind[bi["head"]].astype(np.float64)
    lm = asset.meta.get("landmarks", {})
    P = lambda n, d: (K @ np.array([*lm[n][:3], 1.0]))[:3] if n in lm and len(lm[n]) >= 3 else d
    pos = asset.mesh["position"].astype(np.float64)
    hs = [s for s in asset.submeshes if s["material"] == "skin_head" or s["name"] == "head"]
    if hs:
        s = hs[0]; v = pos[s["first_vertex"]:s["first_vertex"] + s["vertex_count"]]
    else:
        v = pos
    vw = (K @ np.c_[v, np.ones(len(v))].T).T[:, :3]
    top = P("head_top", vw[np.argmax(vw[:, 1])])
    hg = getattr(asset, "hair", {}) or {}
    if "scalp" in hg:                 # the crown of the HAIR (the card's 10 % line is the top of the hair, shots.json)
        hp = np.asarray(hg["scalp"]["pos"], np.float64).reshape(-1, 3)
        hpw = (K @ np.c_[hp, np.ones(len(hp))].T).T[:, :3]
        hpw = hpw[np.abs(hpw[:, 0] - top[0]) < 0.6*float(np.ptp(vw[:, 0]))]      # the dome, not a bun off to the back
        if len(hpw) and hpw[:, 1].max() > top[1]:
            top = np.array([top[0], hpw[:, 1].max(), top[2]])
    chin = P("chin", None)
    eye_l, eye_r = P("eye_l", None), P("eye_r", None)
    if chin is None:
        chin = top - np.array([0, 0.24, 0])*np.linalg.norm(K[:3, 0])
    eye = 0.5*(eye_l + eye_r) if (eye_l is not None and eye_r is not None) else top + 0.55*(chin - top)
    near = vw[(vw[:, 1] > chin[1]) & (vw[:, 1] < top[1])]
    width = float(near[:, 0].max() - near[:, 0].min()) if len(near) else 0.7*float(top[1] - chin[1])
    return top, chin, eye, width


def camera_for(name, asset, rf, w, h, base_cam):
    """a camera override dict for MeshEngine (pos, target, fov in degrees, focus, dofk) or None (the live camera)."""
    if name in ("scene", "full"):
        return None
    top, chin, eye, width = _head_points(asset, rf)
    hh = float(top[1] - chin[1])
    aspect = w/float(h)
    mm = 90.0                             # long lens, full-frame equivalent on the long side
    long_side = 36.0
    vfov = 2*np.degrees(np.arctan((long_side if h >= w else long_side/aspect)/2.0/mm))
    if name == "card":
        # render2: crown 10 % .. chin 57.5 % of the height AND head+ears <= 58 % of the width (a fuller groom /
        # bigger head made the width bind: v4 measured 65 %): the larger frame wins, eye line stays at 35 %
        frame_h = max(hh/(0.575 - 0.10), width/(0.615*aspect))   # (landmark width runs ~6 % under the ears)
        tgt_y = eye[1] - (0.50 - 0.35)*frame_h
        x = float(eye[0])
    elif name == "face":
        frame_h = hh/0.70
        tgt_y = eye[1] - 0.12*frame_h
        x = float(eye[0])
    elif name == "bust":
        frame_h = hh*2.3
        tgt_y = eye[1] - 0.15*frame_h
        x = float(eye[0])
    elif name == "bust3":                 # judge: head ~33 % of the frame height (the AAA head-and-shoulders framing)
        frame_h = hh/0.33
        tgt_y = eye[1] - (0.5 - 0.26)*frame_h
        x = float(eye[0])
    elif name == "mid":
        frame_h = hh*4.2
        tgt_y = eye[1] - 0.30*frame_h
        x = float(eye[0])
    elif name == "hands":
        bi = {b["name"]: i for i, b in enumerate(asset.bones)}
        hl, hr = rf.bones[bi["hand_l"]][:3, 3], rf.bones[bi["hand_r"]][:3, 3]
        hc = 0.5*(hl + hr)
        frame_h = hh*2.4
        if abs(float(hl[1] - hr[1])) > 0.5*hh:         # render2: one hand up (chalk, a held prop): frame THAT hand
            hc = hl if hl[1] > hr[1] else hr
            frame_h = hh*1.6
        tgt_y = float(hc[1]); x = float(hc[0])
    else:
        raise SystemExit(f"--cam {name}: one of {CAMS}")
    d = (frame_h/2.0)/np.tan(np.radians(vfov)/2.0)
    zc = float(eye[2])
    pos = [x, float(eye[1]) - 0.02*hh, zc + d]
    tgt = [x, tgt_y, zc]
    dofk = float(base_cam.get("dofk", 40.0))
    # render2: the card is a portrait lens: the room goes to creamy bokeh behind her (the reference card)
    k = float(os.environ.get("JUNO_CARD_DOF", 5.0))   # render3: 2.5 -> 5 (the reference card: the room is creamy bokeh, ld3/d_strip.png)
    # render3: the SDF character layer (held / worn props) is marched only inside [CHAR_T_NEAR, CHAR_T_FAR]
    # (tuned for the live camera ~7.7 away): a close camera starts its march PAST the hands -> the chalk in her hand
    # never drew in the hands close-up.  Bracket the march around this camera's subject.
    q = {"CHAR_T_NEAR": max(0.05, float(d) - 1.2), "CHAR_T_FAR": float(d) + 2.5}
    return {"pos": pos, "target": tgt, "fov": float(vfov), "focus": float(d), "dofk": dofk*k, "quality": q}


def _apply_shot(rig, asset, prm):
    """JUDGE shot params -> RigFrame through the stand-in FK rig (direct shapes + head/eye bones)."""
    from testrig import euler
    bs = dict(prm.get("blendshapes", {}))
    hd = prm.get("head_deg", {}) or {}
    gd = prm.get("gaze_deg", {}) or {}
    yaw, pitch, roll = float(hd.get("yaw", 0)), float(hd.get("pitch", 0)), float(hd.get("roll", 0))
    rots = {}
    if yaw or pitch or roll:          # 40 % neck, 60 % head; + yaw = the face turns to screen-right (asset +x)
        rots["neck_01"] = euler(-0.4*pitch, 0.4*yaw, -0.4*roll)
        rots["head"] = euler(-0.6*pitch, 0.6*yaw, -0.6*roll)
    return rig.frame("relaxed", "neutral", weights=bs, extra_rots=rots,
                     gaze=(float(gd.get("yaw", 0)), float(gd.get("pitch", 0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--cam", default="scene", choices=CAMS)
    ap.add_argument("--pose", default="relaxed")
    ap.add_argument("--expr", default="neutral")
    ap.add_argument("--params")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--asset")
    ap.add_argument("--char", default="juno", help=": character id (config json + engine); default juno = today's path")
    ap.add_argument("--lod")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--preset", default="full1")
    ap.add_argument("--msaa", type=int)
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--alpha-out")
    ap.add_argument("--json")
    ap.add_argument("--frames", type=int, default=1, help="render N frames (temporal settles), keep the last")
    ap.add_argument("--orbit", type=float, default=0.0, help=": orbit the (non-scene) camera about the head's "
                    "vertical axis, deg (+ = toward her left / screen right); 180 = from behind")
    ap.add_argument("--elev", type=float, default=0.0, help=": raise the camera about the head, deg")
    a = ap.parse_args()
    t0 = time.time()
    from anim.schema import NPARAMS, IDX
    import assetload
    from testrig import TestRig
    import render as _render
    lod = a.lod or os.environ.get("JUNO_LOD") or "lod0"
    asset = assetload.load(a.asset or assetload.char_asset_dir(a.char), lod=lod)   # --char junor = its own asset
    cfg = _render.load_cfg(a.char)
    H_scene = float(cfg.get("mesh", {}).get("H_scene", 2.10))
    trig = TestRig(asset, H_scene=H_scene)
    P = np.zeros(NPARAMS, np.float32)
    rf = None
    if a.params:
        if a.params.endswith(".npy"):
            arr = np.load(a.params); P = (arr[a.frame] if arr.ndim == 2 else arr).astype(np.float32)
        else:
            j = json.load(open(a.params))
            if "params" in j and isinstance(j["params"], dict):
                j = j["params"]
            if any(k in j for k in ("blendshapes", "head_deg", "gaze_deg")):
                rf = _apply_shot(trig, asset, j)
            else:
                for k, v in j.items():
                    if k in IDX:
                        P[IDX[k]] = float(v)
    use_testrig = rf is not None or a.pose != "relaxed" or a.expr != "neutral" or not a.params
    if rf is None and use_testrig:
        rf = trig.frame(a.pose, a.expr)
    rf_cam = rf
    if rf_cam is None and a.cam not in ("scene", "full"):
        # render2: a control vector moves the body (a raised hand, a lean): frame the REAL rig's pose, not the
        # stand-in's rest pose -- a throwaway tiny engine solves it (the camera is baked into the engine at init)
        from engine import MeshEngine as _ME
        e0 = _ME(a.char, 64, 36, a.preset, backend=a.device, threads=a.threads, log=False, asset_dir=asset.dir, lod=lod)
        e0._t_now = 0.0
        rf_cam = e0.rig_frame(P)
        try:
            e0.ctx.release()
        except Exception:
            pass
        del e0
    cam = camera_for(a.cam, asset, rf_cam if rf_cam is not None else trig.frame(), a.w, a.h, cfg["camera"])
    if cam is not None and (a.orbit or a.elev):        # orbit about the eye-centre vertical axis
        _, _, eyec, _ = _head_points(asset, rf_cam if rf_cam is not None else trig.frame())
        pc, tc = np.asarray(cam["pos"], float), np.asarray(cam["target"], float)
        piv = np.array([float(eyec[0]), float(tc[1]), float(eyec[2]) - 0.03])
        def _rot(v):
            ya, el = np.radians(a.orbit), np.radians(a.elev)
            v = v - piv
            c, s_ = np.cos(ya), np.sin(ya)
            v = np.array([c*v[0] + s_*v[2], v[1], -s_*v[0] + c*v[2]])
            hz = np.hypot(v[0], v[2]); ang = np.arctan2(v[1], hz) + el
            r = np.hypot(hz, v[1]); hd = np.array([v[0], 0, v[2]])/max(hz, 1e-9)
            return piv + hd*r*np.cos(ang) + np.array([0, r*np.sin(ang), 0])
        cam["pos"] = [float(x) for x in _rot(pc)]
        cam["target"] = [float(x) for x in _rot(tc)]
    if cam is not None:
        cam["motion_layers"] = os.environ.get("JUNO_STILL_ML", "0") == "1"
    from engine import MeshEngine
    os.environ.setdefault("JUNO_TESTRIG", "1" if use_testrig else os.environ.get("JUNO_TESTRIG", "0"))
    eng = MeshEngine(a.char, a.w, a.h, a.preset, backend=a.device, threads=a.threads, log=False,
                     asset_dir=asset.dir, lod=lod, cam=cam, msaa=a.msaa)
    if rf is not None:
        eng.rig_override = rf
    t1 = time.time()
    for i in range(max(1, a.frames)):
        img = eng.render(a.frame + i, P)
    t2 = time.time()
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    alpha = None
    if a.no_scene or a.alpha_out:
        alpha = np.frombuffer(eng.mc.t_md.read(), dtype=np.float16).reshape(a.h, a.w, 4)[::-1, :, 3]
        alpha = (np.clip(alpha.astype(np.float32), 0, 1)*255 + 0.5).astype(np.uint8)
    if a.no_scene:
        c = np.frombuffer(eng.mc.t_mc.read(), dtype=np.float16).reshape(a.h, a.w, 4)[::-1].astype(np.float32)
        # the mesh layer through the same post would need the scene; here: un-premultiplied, a simple filmic map
        rgb = c[:, :, :3]/np.maximum(c[:, :, 3:4], 1e-4)
        rgb = rgb*float(eng.grade.get("exposure", 1.0))
        rgb = np.clip((rgb*(2.51*rgb + 0.03))/(rgb*(2.43*rgb + 0.59) + 0.14), 0, 1)**(1/2.2)
        out = np.dstack([(rgb*255 + 0.5).astype(np.uint8), alpha])
        Image.fromarray(out, "RGBA").save(a.out)
    else:
        Image.fromarray(img).save(a.out)
    if a.alpha_out:
        Image.fromarray(alpha, "L").save(a.alpha_out)
    meta = {"out": a.out, "w": a.w, "h": a.h, "cam": a.cam, "camera": cam, "pose": a.pose, "expr": a.expr,
            "params": a.params, "device": a.device, "asset": asset.dir, "lod": lod, "msaa": eng.mc.msaa,
            "init_s": round(t1 - t0, 2), "render_s": round(t2 - t1, 2), "init_times": eng.init_times}
    if a.json:
        json.dump(meta, open(a.json, "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
