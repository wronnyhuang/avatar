#!/usr/bin/env python3
"""render-perf: Juno's frame-time bench (MeshEngine; also any RTEngine character for comparison).  One JSON out.

  python renderer/tools/juno_bench.py --w 1920 --h 1080 [--char juno] [--preset taa] [--tier N | --ladder]
      [--frames 120] [--motion talk] [--encode x264,nvenc] [--bitrate 4000000] [--device gpu|cpu] [--lod lod0|lod1]
      [--asset DIR] [--json out.json] [--save mid.png] [--no-prof]

What it measures (all on a MOVING control vector: talk + head motion + a hand gesture, anim/schema PARAMS):
  init_s        cold engine construction (shader compiles, asset upload, bakes; RTEngine.init_times + mesh_s)
  prewarm_s     eng.prewarm() (every SDF path + every mesh tier; the room probe bake + face cache happen before it)
  first_ms      the first live frame AFTER prewarm on a fresh pose (a stall shows here)
  render        sync eng.render() wall ms (draw + readback: the worker's render_ms) mean / p95
  submit        pipelined eng.submit() wall ms per frame (the live worker's path) mean / p95
  rig_ms        RIG's CPU solve per frame (MeshEngine.mesh_ms["rig"])
  gpu_ms        GPU time per pass: RTEngine._draw's GL_TIME_ELAPSED marks (comp_dof INCLUDES the mesh) + the mesh
                passes by GL timestamps (skin, shadow, prefill, forward, hair, resolve, sss, composite)
  enc           encode ms per codec (x264 = libx264, nvenc = h264_nvenc; the encoder actually opened + errors)
  vram_mb       GL_NVX_gpu_memory_info delta (used by this process's engine) + nvidia-smi used before/after
  --ladder      every mesh quality tier (renderer/mesh/quality.py) fixed for --frames: render ms + its gpu passes
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), os.path.join(ROOT, "mesh"), HERE]
from anim.schema import IDX, NPARAMS  # noqa: E402

MESH_PASSES = ("skin", "shadow", "prefill", "forward", "hair", "resolve", "sss", "ao", "taa", "composite")   # ao = mesh.pbr.ao


def motion_track(n, kind="talk", fps=30, cam_zoom=None, cam_y=None):
    """talk + head motion + a hand gesture (the live mix), or rig_bench's tracks by name."""
    from rig_bench import pose_track
    if kind != "talk":
        P = pose_track(n, kind, fps)
    else:
        P = pose_track(n, "raised", fps)          # head yaw/pitch/roll, jaw/lip/teeth talk, blink, breath, R hand waving
        t = np.arange(n)/fps
        def s(name, v):
            if name in IDX:
                P[:, IDX[name]] = v
        s("smile", 0.3 + 0.3*np.sin(t*0.7)); s("brow_l", 0.3*np.sin(t*1.1)); s("brow_r", 0.3*np.sin(t*1.1 + 0.3))
        s("gaze_x", 0.3*np.sin(t*0.5)); s("gaze_y", 0.1*np.sin(t*0.8))
        s("mouth_open", 0.5 + 0.5*np.sin(t*7.0)); s("lip_round", 0.5 + 0.5*np.sin(t*3.3))
    if cam_zoom is not None and "cam_zoom" in IDX:
        P[:, IDX["cam_zoom"]] = cam_zoom
    if cam_y is not None and "cam_y" in IDX:        # PERF2: Juno's home close-up aims at the head (moves.js views.home)
        P[:, IDX["cam_y"]] = cam_y
    return P


def nvx_free_kb():
    """GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX of the CURRENT context (None off NVIDIA)."""
    import ctypes
    for name in ("libOpenGL.so.0", "libGL.so.1"):
        try:
            gl = ctypes.CDLL(name); break
        except OSError:
            gl = None
    if gl is None:
        return None
    v = ctypes.c_int(-1)
    try:
        gl.glGetError()
        gl.glGetIntegerv(0x9049, ctypes.byref(v))
        return None if gl.glGetError() != 0 or v.value < 0 else int(v.value)
    except Exception:
        return None


def smi_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        u, name = out.split(",", 1)
        return int(u), name.strip()
    except Exception:
        return None, None


def stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    return dict(mean=round(float(a.mean()), 2), p50=round(float(np.percentile(a, 50)), 2),
                p95=round(float(np.percentile(a, 95)), 2), max=round(float(a.max()), 2))


def make_engine(a):
    import rt_engine
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.char, a.w, a.h, 24)
    P = rt_engine.resolve_preset(preset)
    if a.tiers:
        P["tiers"] = True                           # as the live backend does for the adaptive controller
    try:
        from live.backends.rt import engine_class   # PIPE's hook (mesh for backend:"mesh" characters)
        cls = engine_class(a.char)
    except ImportError:
        cls = rt_engine.RTEngine
        if a.char == "juno":
            from mesh.engine import MeshEngine as cls
    kw = dict(backend=a.device, threads=a.threads)
    if cls.__name__ == "MeshEngine":
        kw.update(asset_dir=a.asset, lod=a.lod, log=a.log)
    return cls(a.char, a.w, a.h, P, **kw), preset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="juno")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--preset", default="taa")
    ap.add_argument("--no-tiers", dest="tiers", action="store_false", help="the preset alone (no RTEngine levels)")
    ap.add_argument("--tier", type=int, default=None, help="mesh quality tier (default: the ladder's start = best)")
    ap.add_argument("--ladder", action="store_true", help="also time every mesh tier")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ladder-frames", type=int, default=40)
    ap.add_argument("--motion", default="talk")
    ap.add_argument("--cam-zoom", type=float, default=None)
    ap.add_argument("--cam-y", type=float, default=None, help="Juno home close-up: --cam-zoom -0.6212 --cam-y 0.2934")
    ap.add_argument("--encode", default="x264,nvenc", help="comma list of x264,nvenc ('' = none)")
    ap.add_argument("--bitrate", type=int, default=None)
    ap.add_argument("--enc-threads", type=int, default=4)
    ap.add_argument("--device", default=os.environ.get("AVATAR_RT_DEVICE", "gpu"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LP_NUM_THREADS", "8")))
    ap.add_argument("--lod", default=None)
    ap.add_argument("--asset", default=os.environ.get("JUNO_ASSET"))
    ap.add_argument("--json")
    ap.add_argument("--save")
    ap.add_argument("--no-prof", dest="prof", action="store_false")
    ap.add_argument("--no-prewarm", dest="prewarm", action="store_false")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    out = dict(char=a.char, w=a.w, h=a.h, device=a.device, motion=a.motion, frames=a.frames)
    smi0, gpu_name = smi_used_mb()
    out["gpu"] = gpu_name
    # a bare context's free memory first: the engine's own allocation = the drop after construction
    free0 = None
    if a.device == "gpu":
        try:
            import moderngl
            c0 = moderngl.create_standalone_context(backend="egl")
            free0 = nvx_free_kb()
            out["gl_renderer"] = c0.info.get("GL_RENDERER")
            c0.release()
        except Exception as e:
            out["bare_ctx_err"] = repr(e)[:200]
    t0 = time.time()
    eng, preset = make_engine(a)
    out["engine"] = type(eng).__name__
    out["preset"] = preset if isinstance(preset, str) else "dict"
    out["init_s"] = round(time.time() - t0, 2)
    out["init_times"] = dict(getattr(eng, "init_times", {}))
    mesh = hasattr(eng, "mc")
    if mesh:
        out["asset"] = str(eng.asset.dir); out["lod"] = eng.asset.lod.get("name") if isinstance(eng.asset.lod, dict) else str(eng.asset.lod)
        out["verts"] = int(eng.asset.mesh_meta.get("vertex_count", 0))
        if a.tier is not None:
            eng.set_tier(a.tier)
        out["tiers"] = getattr(eng, "tier_names", None)
    if a.prewarm:
        t1 = time.time()
        eng.prewarm()
        out["prewarm_s"] = round(time.time() - t1, 2)
        if mesh and a.tier is not None:
            eng.set_tier(a.tier)
    if mesh:
        out["probe"] = getattr(eng.mc.light, "probe_info", None)
    free1 = nvx_free_kb() if a.device == "gpu" else None
    if free0 is not None and free1 is not None:
        out["vram_mb"] = round((free0 - free1)/1024.0, 1)
    smi1, _ = smi_used_mb()
    if smi0 is not None and smi1 is not None:
        out["smi_used_mb"] = [smi0, smi1]
    n = a.frames
    P = motion_track(n + 40, a.motion, cam_zoom=a.cam_zoom, cam_y=a.cam_y)
    # the first live frame after prewarm (a fresh pose, mid-gesture)
    t = time.perf_counter(); eng.render(1000, P[n//2]); out["first_ms"] = round((time.perf_counter() - t)*1e3, 2)
    for i in range(5):
        eng.render(i, P[i])
    # 1. sync render (the worker's render_ms) + encoders on the same frames
    encs = {}
    bitrate = a.bitrate or int(4_000_000*(a.w*a.h)/(1920*1080.0)) or 250_000
    from avatar.encoder import H264Encoder
    for k in [x for x in a.encode.split(",") if x]:
        name = {"x264": "libx264", "nvenc": "h264_nvenc"}[k]
        try:
            e = H264Encoder(a.w, a.h, 24, bitrate, 2.0, prefer=(name,), threads=a.enc_threads)
            encs[k] = (e, [])
        except Exception as ex:
            out.setdefault("enc", {})[k] = dict(ok=False, err=str(ex)[:300])
    tr, trig = [], []
    for i in range(n):
        f = 5 + i
        t = time.perf_counter()
        img = eng.render(f, P[f])
        tr.append((time.perf_counter() - t)*1e3)
        if mesh:
            trig.append(eng.mesh_ms.get("rig", 0.0))
        for k, (e, ts) in encs.items():
            t = time.perf_counter(); e.encode(img, i, i == 0); ts.append((time.perf_counter() - t)*1e3)
        if a.save and i == n//2:
            from PIL import Image
            Image.fromarray(img).save(a.save)
    out["render"] = stats(tr)
    if trig:
        out["rig_ms"] = stats(trig)
    for k, (e, ts) in encs.items():
        out.setdefault("enc", {})[k] = dict(ok=True, encoder=e.name, ms=stats(ts[2:]), bitrate=bitrate)
        e.close() if hasattr(e, "close") else None
    # 2. pipelined (submit/fetch, the live path): wall per frame
    ts = []
    for i in range(n):
        f = 5 + i
        t = time.perf_counter(); eng.submit(f, P[f]); ts.append((time.perf_counter() - t)*1e3)
    eng.flush()
    out["submit"] = stats(ts[2:])
    # 3. GPU per pass (profiled frames: the queries wait, so these frames are not the timed ones)
    if a.prof and a.device == "gpu":
        out["gpu_ms"] = profile(eng, P, min(n, 60), mesh)
    # 4. the ladder
    if a.ladder and mesh:
        lad = []
        start = eng.mtier
        for tier in range(eng.mladder.n):
            eng.set_tier(tier)
            for i in range(3):
                eng.render(i, P[i])
            tt = []
            for i in range(a.ladder_frames):
                f = 5 + i
                t = time.perf_counter(); eng.render(f, P[f]); tt.append((time.perf_counter() - t)*1e3)
            row = dict(tier=tier, knobs={k: v for k, v in eng.mladder.values(tier).items()}, render=stats(tt))
            if a.prof and a.device == "gpu":
                row["gpu_ms"] = profile(eng, P, 20, mesh)
            lad.append(row)
        eng.set_tier(start)
        out["ladder"] = lad
    js = json.dumps(out)
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    print(js, flush=True)
    eng.close()


def profile(eng, P, n, mesh):
    from gpuprof import GpuStamps
    st = GpuStamps() if mesh else None
    if mesh:
        eng.stamps = st if st.ok else None
    acc, macc = {}, {}
    for i in range(n):
        f = 5 + i
        prof = {}
        eng.render(f, P[f], prof)
        for k, v in prof.items():
            acc[k] = acc.get(k, 0.0) + v*1e3
        if mesh and eng.stamps is not None:
            for k, v in st.collect().items():
                macc[k] = macc.get(k, 0.0) + v
    if mesh:
        eng.stamps = None
    res = {k: round(v/n, 3) for k, v in acc.items()}
    res["total"] = round(sum(acc.values())/n, 3)
    if macc:
        res["mesh"] = {k: round(macc[k]/n, 3) for k in MESH_PASSES if k in macc}
        res["mesh_total"] = round(sum(macc.values())/n, 3)
        if "comp_dof" in res:
            res["scene_comp"] = round(res["comp_dof"] - res["mesh_total"], 3)
    return res


if __name__ == "__main__":
    main()
