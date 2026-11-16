#!/usr/bin/env python3
"""hair-shade: hair GPU cost + temporal stability over a slow head turn (1080p).

  flock gpu.lock python tools/hair_bench.py [--asset DIR] [--cam face|bust] [--frames 24] [--yaw-step 0.5]
        [--w 1920 --h 1080] [--save-dir DIR] [--no-shimmer]

Per frame: the head turns yaw0 + i*yaw_step degrees (juno_still's shot params: 40 % neck / 60 % head).
cost    GL timestamps around HairRenderer.shadow (key-map strands + the v2 deep opacity pass) and HairRenderer.draw
        (cap + strands), glFinish-free (GpuStamps), mean ms over frames 2.. ; also the whole mesh frame wall time.
shimmer the hair region (a JUNO_HAIR_DEBUG=1 flat-green render of the same frame, eroded 2 px, in 3 consecutive
        frames) of the FINAL image: mean and p99 of |L(t) - (L(t-1) + L(t+1))/2| (luma 0..255), the 2nd temporal
        difference (a smooth turn cancels, frame-to-frame crawl / sparkle does not).  One JSON line.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), os.path.join(ROOT, "mesh"), HERE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset")
    ap.add_argument("--cam", default="face", help="face|bust|mid|scene (scene = the live camera)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--yaw0", type=float, default=-4.0)
    ap.add_argument("--yaw-step", type=float, default=0.5)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--preset", default="full1")
    ap.add_argument("--save-dir")
    ap.add_argument("--no-shimmer", action="store_true")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--char", default="juno", help=": character config (junor = its own look)")
    ap.add_argument("--save-all", action="store_true", help=" HAIR2: save every frame to --save-dir (stream tests)")
    ap.add_argument("--frac", type=float, help="force HairRenderer.frac (the quality ladder sets it otherwise)")
    ap.add_argument("--seg", type=int, help="force HairRenderer.seg")
    a = ap.parse_args()
    from anim.schema import NPARAMS
    import assetload
    from testrig import TestRig
    import render as _render
    from juno_still import camera_for, _apply_shot
    from gpuprof import GpuStamps
    asset = assetload.load(a.asset, lod="lod0")
    cfg = _render.load_cfg(a.char)
    trig = TestRig(asset, H_scene=float(cfg.get("mesh", {}).get("H_scene", 2.10)))
    cam = camera_for(a.cam, asset, trig.frame(), a.w, a.h, cfg["camera"])
    if cam is not None:
        cam["motion_layers"] = False
    os.environ.setdefault("JUNO_TESTRIG", "1")
    from engine import MeshEngine
    eng = MeshEngine(a.char, a.w, a.h, a.preset, backend=a.device, log=False, asset_dir=asset.dir, lod="lod0", cam=cam)
    hr = eng.mc.hair
    st = GpuStamps()
    seg = {}
    if hr is not None:
        sh0, dr0 = hr.shadow, hr.draw
        def shadow(*x, **k):
            st.mark("s0"); r = sh0(*x, **k); st.mark("hair_shadow"); return r
        def draw(*x, **k):
            st.mark("d0"); r = dr0(*x, **k); st.mark("hair_draw"); return r
        hr.shadow, hr.draw = shadow, draw
        if hasattr(hr, "composite"):                                 # HAIR2 layer composite
            cp0 = hr.composite
            def comp(*x, **k):
                st.mark("c0"); r = cp0(*x, **k); st.mark("hair_comp"); return r
            hr.composite = comp
        hr._stamp = st.mark
    P = np.zeros(NPARAMS, np.float32)
    frames, masks, walls = [], [], []
    for i in range(a.frames):
        eng.rig_override = _apply_shot(trig, asset, {"head_deg": {"yaw": a.yaw0 + i*a.yaw_step}})
        if hr is not None and a.frac is not None:
            hr.frac = a.frac
        if hr is not None and a.seg is not None:
            hr.seg = a.seg
        t0 = time.time()
        img = eng.render(i, P)
        walls.append(time.time() - t0)
        if a.save_all and a.save_dir:
            from PIL import Image
            os.makedirs(a.save_dir, exist_ok=True)
            Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(a.save_dir, f"a{i:03d}.png"))
        out = st.collect()
        # pair the marks: hair_shadow = s0 -> hair_shadow, hair_draw = d0 -> hair_draw (other segments are gaps)
        if i >= 2:
            for k in ("hair_shadow", "hair_cap", "hair_cov", "hair_prepass", "hair_draw", "hair_comp"):
                seg.setdefault(k, []).append(out.get(k, 0.0))
        if not a.no_shimmer:
            frames.append(np.asarray(img, np.float32)[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32))
            os.environ["JUNO_HAIR_DEBUG"] = "1"
            m = np.asarray(eng.render(i, P), np.float32)
            os.environ["JUNO_HAIR_DEBUG"] = "0"
            st.collect()
            g = (m[..., 1] > 200) & (m[..., 0] < 90) & (m[..., 2] < 90)
            masks.append(g)
            if a.save_dir and i in (a.frames//2 - 1, a.frames//2, a.frames//2 + 1):
                from PIL import Image
                os.makedirs(a.save_dir, exist_ok=True)
                Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(a.save_dir, f"f{i:03d}.png"))
    res = {"asset": asset.dir, "cam": a.cam, "w": a.w, "h": a.h, "frames": a.frames, "yaw_step": a.yaw_step,
           "v2": bool(getattr(hr, "v2", False)), "strands": (hr.stats if hr is not None else {}),
           "ms": {k: round(float(np.mean(v)), 3) for k, v in seg.items()},
           "ms_total_hair": round(float(sum(np.mean(v) for v in seg.values())), 3),
           "_note": "hair_draw = the shading pass when the pre-pass runs (else cap excluded, all strands)",
           "wall_ms": round(float(np.median(walls[2:]))*1000, 1)}
    if not a.no_shimmer and len(frames) >= 3:
        import cv2
        F = np.stack(frames); M = np.stack(masks)
        ker = np.ones((5, 5), np.uint8)
        d2s = []
        for t in range(1, len(F) - 1):
            m = M[t - 1] & M[t] & M[t + 1]
            m = cv2.erode(m.astype(np.uint8), ker) > 0
            if m.sum() < 100:
                continue
            d2 = np.abs(F[t] - 0.5*(F[t - 1] + F[t + 1]))[m]
            d2s.append(d2)
        allv = np.concatenate(d2s)
        res["shimmer"] = {"mean": round(float(allv.mean()), 3), "p99": round(float(np.percentile(allv, 99)), 2),
                          "px": int(allv.size//max(len(d2s), 1))}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
