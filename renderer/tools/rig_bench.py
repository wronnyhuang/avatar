#!/usr/bin/env python3
"""Frame-time benchmark of the real-time engine with a MOVING pose (head, mouth and, with --motion, the hands).

  flock gpu.lock python tools/rig_bench.py --animal fox --w 480 --h 270 --frames 600 --motion rest|move|raised [--encode]

Times engine.render() (draw + PBO readback, synchronous, the worker's render_ms) and, with --encode, libx264 encode of
the same frame (the worker runs it on its own thread; both are reported and their sum).  Prints one JSON line.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if "--root" in sys.argv:          # bench another renderer tree (e.g. the head-only baseline) with this tool
    ROOT = os.path.abspath(sys.argv[sys.argv.index("--root") + 1])
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402


def pose_track(n, motion, fps=30):
    P = np.zeros((n, NPARAMS), "f4")
    t = np.arange(n)/fps
    def s(name, v):
        if name in IDX:
            P[:, IDX[name]] = v
    s("head_yaw", 0.25*np.sin(t*0.9)); s("head_pitch", 0.08*np.sin(t*1.3)); s("head_roll", 0.06*np.sin(t*0.7))
    s("teeth_gap", 0.5 + 0.5*np.sin(t*9.0)); s("lip_open", 0.5 + 0.5*np.sin(t*9.0 + 0.4)); s("jaw", 0.4 + 0.4*np.sin(t*9.0))
    s("blink_l", (np.sin(t*2.1) > 0.97).astype(float)); s("blink_r", (np.sin(t*2.1) > 0.97).astype(float))
    s("breath", np.sin(t*1.6))
    if motion == "rest":
        s("hand_l_open", 0.2); s("hand_r_open", 0.2)
    elif motion == "raised":      # WAVE_UP on the right, CHIN on the left, both waving a little
        s("hand_r_x", 0.20 + 0.12*np.sin(t*2*np.pi*2.5)); s("hand_r_y", 1.0); s("hand_r_z", 0.2); s("hand_r_open", 1.0)
        s("hand_r_turn", 1.0 - 0.3*(0.5 + 0.5*np.sin(t*2*np.pi*2.5)))
        s("hand_l_x", -0.35); s("hand_l_y", 0.75 + 0.05*np.sin(t*3)); s("hand_l_z", 0.35); s("hand_l_open", 0.3)
    elif motion == "move":        # both hands sweeping through the whole reach
        for side, ph in (("l", 0.0), ("r", 1.7)):
            s(f"hand_{side}_x", 0.25*np.sin(t*0.8 + ph)); s(f"hand_{side}_y", 0.5 + 0.5*np.sin(t*1.1 + ph))
            s(f"hand_{side}_z", 0.3 + 0.3*np.sin(t*0.6 + ph)); s(f"hand_{side}_open", 0.5 + 0.5*np.sin(t*1.7 + ph))
            s(f"hand_{side}_turn", np.sin(t*0.9 + ph))
        s("tail_wag", np.sin(t*5.0)); s("ear_l_turn", 0.5*np.sin(t*1.2)); s("nose_wiggle", np.sin(t*8.0))
    # body: whole-body motion (root transform, legs, camera)
    elif motion == "spin":        # 1.5 turns per second, arms out a little
        s("root_yaw", 1.5*t); s("hand_l_y", 0.4); s("hand_r_y", 0.4); s("hand_l_x", 0.4); s("hand_r_x", 0.4)
    elif motion == "flip":        # a backflip every 1.2 s
        u = (t % 1.2)/1.2
        s("root_pitch", -u); s("root_y", 0.9*np.sin(np.pi*u)); s("squash", -0.2*np.sin(np.pi*u))
    elif motion == "walk":        # walking across, 1.8 steps/s, the camera following
        ph = 2*np.pi*0.9*t
        s("root_x", 1.2*np.sin(t*0.35)); s("cam_x", 1.0*np.sin(t*0.35))
        s("foot_l_y", 0.05*np.maximum(np.sin(ph), 0)); s("foot_r_y", 0.05*np.maximum(-np.sin(ph), 0))
        s("foot_l_x", 0.06*np.cos(ph)); s("foot_r_x", -0.06*np.cos(ph)); s("body_bounce", 0.4*np.cos(2*ph))
        s("hand_l_z", 0.15*np.cos(ph)); s("hand_r_z", -0.15*np.cos(ph))
    elif motion == "wide":        # the wide shot, walking about in it
        ph = 2*np.pi*0.9*t
        s("cam_zoom", 1.0); s("root_x", 1.5*np.sin(t*0.4)); s("root_z", 0.4*np.sin(t*0.3))
        s("foot_l_y", 0.05*np.maximum(np.sin(ph), 0)); s("foot_r_y", 0.05*np.maximum(-np.sin(ph), 0))
    elif motion == "worn":        # the muse desk scene (props branch): headphones worn, laptop open, walking in the wide shot
        ph = 2*np.pi*0.9*t
        s("prop0_hh", 1.0); s("prop3_s", 1.0); s("cam_zoom", 0.6); s("root_x", 0.6*np.sin(t*0.4)); s("cam_x", 0.5*np.sin(t*0.4))
        s("foot_l_y", 0.05*np.maximum(np.sin(ph), 0)); s("foot_r_y", 0.05*np.maximum(-np.sin(ph), 0))
        s("hand_r_y", 0.5 + 0.2*np.sin(t*2.0)); s("hand_r_z", 0.4)
    # scene: the muse's live framings (home = standing, cam_zoom 0.3) and the station cameras, writing
    elif motion == "home":
        s("cam_zoom", 0.3)
    elif motion in ("board", "tvview"):
        cam = (-0.50, -0.33, 0.10) if motion == "board" else (float(os.environ.get("TVZ", -0.77)), 0.565, 0.21)
        st = (-0.2379, -0.8537, -0.0833) if motion == "board" else (0.3478, -0.8478, 0.07)
        s("cam_zoom", cam[0]); s("cam_x", cam[1]); s("cam_y", cam[2])
        s("root_x", st[0] + 0.02*np.sin(t*0.5)); s("root_z", st[1]); s("root_yaw", st[2]); s("gaze_free", 0.6)
        s("hand_l_x", 0.55 + 0.1*np.sin(t*3.1)); s("hand_l_y", 0.8 + 0.2*np.sin(t*2.3)); s("hand_l_z", -0.5)
    # legacy columns for the old tree (ignored where absent)
    if motion in ("raised", "move"):
        s("arm_r", 1.0); s("wave_r", 1.0); s("paw_open_r", 1.0)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="fox")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--motion", default="rest")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--save")
    ap.add_argument("--root")
    ap.add_argument("--surf", action="store_true", help="upload a chalk rect every frame and a full TV frame every 2nd")
    a = ap.parse_args()
    import rt_engine
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.animal, a.w, a.h, 24)
    t0 = time.time()
    eng = rt_engine.RTEngine(a.animal, a.w, a.h, preset, backend=a.backend)
    init = time.time() - t0
    P = pose_track(a.frames + 10, a.motion)
    enc = None
    if a.encode:
        from avatar.encoder import open_encoder
        enc = open_encoder("h264", a.w, a.h, fps=24, bitrate=250000, gop_s=2.0, nvenc=False, threads=2, quality=80)
    tr, te = [], []
    for i in range(a.frames + 10):
        t = time.perf_counter()
        if a.surf and hasattr(eng, "surface_write") and eng.surface_names():     # a chalk segment + a TV frame
            x, y = (37*i) % 980, (23*i) % 470
            eng.surface_write("board", x, y, 40, 40, np.full((40, 40, 4), 200, np.uint8))
            if i % 2 == 0 and "tv" in eng.surface_names():
                eng.surface_write("tv", 0, 0, 512, 288, np.full((288, 512, 4), (i*7) % 255, np.uint8))
        img = eng.render(i, P[i])
        t1 = time.perf_counter()
        if enc is not None:
            enc.encode(img, i, i == 0)
        t2 = time.perf_counter()
        if i >= 10:
            tr.append(t1 - t); te.append(t2 - t1)
        if a.save and i == a.frames//2:
            from PIL import Image
            Image.fromarray(img).save(a.save)
    tr, te = np.array(tr)*1000, np.array(te)*1000
    tot = tr + te
    out = dict(tree=ROOT, animal=a.animal, w=a.w, h=a.h, preset=preset if isinstance(preset, str) else "dict", motion=a.motion,
               surf=bool(a.surf), frames=len(tr), init_s=round(init, 1), render_mean=round(tr.mean(), 2), render_p95=round(np.percentile(tr, 95), 2),
               enc_mean=round(te.mean(), 2), total_mean=round(tot.mean(), 2), total_p95=round(np.percentile(tot, 95), 2))
    print(json.dumps(out), flush=True)
    eng.close()


if __name__ == "__main__":
    main()
