"""The render server's GPU child process: one backend instance (one engine, one GL context) + the session's encoder.

A child process, not a thread: the GL context lives and dies with it (a character switch on the K620 frees the old
engine's textures for certain by ending the process), a driver crash cannot take the server down, and the GIL of the
WebSocket process is never held by a GL readback.

parent -> child (multiprocessing Connection):
  ("encoder", {codec, width, height, fps, bitrate, gop_s, nvenc, threads, quality})   (re)open: next frame is a key
  ("job", j, index, params, force_key, t_sent)        render 30 fps row `index` with `params` as output frame j
  ("job", j, index, params, force_key, t_sent, m2)    motion2: + (canon (189), canon_w, modes) for Juno's rig
                                                      (JunoRig.solve(..., canon=, canon_w=, modes=); see _motion2_hook)
  ("keyframe",)                                       next encoded frame is an IDR
  ("quit",)
child -> parent:
  ("status", stage, detail, eta_s)                    stage: init | compile | bake | warmup
  ("ready", info)                                     backend, opts, init_ms, warm_ms, gpu
  ("encoder", info)                                   encoder name, codec_string, errors
  ("rendered", j, render_ms)                          the GPU is free again: the parent picks the next job NOW
  ("frame", j, flags, payload, render_ms, encode_ms, t_render0, t_done)
  ("error", message)

Render and encode overlap: the main thread renders frame k+1 while a second thread encodes frame k.  The parent sends
the next job only when "rendered" says the GPU is free (and at most two frames wait for the encoder), so a job is
chosen at the last moment -- the freshest due frame -- and the frame period is max(render, encode), not their sum.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_RT_TREE = os.environ.get("AVATAR_RT_TREE", "")
mono = time.monotonic


def gpu_name():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _load_spec(spec):
    import importlib
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def _rt_from_tree(tree):
    """The rt backend from another checkout (branch rt-gpu, before it is merged): its engine/ goes first on sys.path so
    `import rt_engine` (and the render.py it imports) come from that tree; live/ and anim/ stay this tree's (the same
    files: the contract and the schema are shared)."""
    import importlib.util
    eng = os.path.join(tree, "engine")
    path = os.path.join(tree, "live", "backends", "rt.py")
    if not os.path.exists(path):
        raise ImportError(f"no rt backend at {path}")
    sys.path.insert(0, eng)
    spec = importlib.util.spec_from_file_location("avatar_rt_backend_ext", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RTBackend


def resolve_backend(name, rt_tree=None):
    """-> (class, resolved name, source).  name: rt | v2 | sprite | raster | auto (rt if it imports, else v2) |
    package.module:Class."""
    from live.backend_api import REGISTRY, get_backend
    tree = rt_tree if rt_tree is not None else DEFAULT_RT_TREE
    if name == "auto":
        try:
            return resolve_backend("rt", rt_tree)
        except Exception:
            return get_backend("v2"), "v2", "tree"
    if name in REGISTRY:
        try:
            return get_backend(name), name, "tree"
        except ImportError:
            if name == "rt" and tree:
                return _rt_from_tree(tree), "rt", tree
            raise
    return _load_spec(name), name, "spec"


def default_opts(backend, opts):
    o = dict(opts or {})
    if backend == "v2":
        o.setdefault("spp", 1)
        o.setdefault("backend", "gpu")
    return o


def bg_cached(character, w, h):
    """Is this character's background already baked at this size (render.py's cache under $TA_OUT)?"""
    try:
        sys.path.insert(0, os.path.join(ROOT, "engine"))
        import render
        sh = render.Shaders(render.load_cfg(character))
        return os.path.exists(os.path.join(render.OUT_ROOT, character, f"bg_{w}x{h}_{sh.key()}.npy"))
    except Exception:
        return True


def _motion2_hook(be):
    """motion2: -> box (a 1-item list) whose content (canon, canon_w, modes) the NEXT render passes to the mesh
    engine's JunoRig.solve as keyword arguments, or None when this backend has no such rig (the job's m2 is then ignored).
    The rig's solve is wrapped on the instance: rig_override, the light probe and every other caller are unchanged and a
    job without m2 calls the original solve with the original arguments (bit-identical)."""
    import inspect
    eng = getattr(be, "eng", be)
    rig = getattr(eng, "rig", None)
    orig = getattr(rig, "solve", None)
    if orig is None:
        return None
    try:
        ps = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        return None
    if "canon" not in ps:
        print("[motion2] this rig's solve has no canon input: motion2 frames are ignored", file=sys.stderr, flush=True)
        return None
    box = [None]
    broken = []

    def solve(P, *a, **k):
        m = box[0]
        if m is None or broken or "canon" in k:
            return orig(P, *a, **k)
        try:
            return orig(P, *a, canon=m[0], canon_w=m[1], modes=m[2], **k)
        except Exception as ex:          # a canon bug never costs a frame: this session's rig goes back to the dials
            broken.append(1)
            print(f"[motion2] canon solve failed, motion2 off for this engine: {type(ex).__name__}: {ex}",
                  file=sys.stderr, flush=True)
            return orig(P, *a, **k)

    rig.solve = solve
    return box


def main(conn, character, width, height, backend, opts, rt_tree=None, warmup=3):
    send_lock = threading.Lock()

    def send(*msg):
        with send_lock:
            conn.send(msg)

    try:
        t0 = mono()
        send("status", "init", f"{character} {width}x{height} {backend}", None)
        cls, bname, src = resolve_backend(backend, rt_tree)
        o = default_opts(bname, opts)
        cached = bg_cached(character, width, height)
        if not cached:
            px = width * height / (1280 * 720)
            send("status", "bake", f"baking the {character} background at {width}x{height} (once, cached)",
                 round(45 * px + 5, 1))
        else:
            send("status", "compile", f"compiling {character} shaders ({bname})", 5)
        be = cls(character, width, height, **o)
        init_ms = 1000 * (mono() - t0)
        send("status", "warmup", f"{warmup} warm-up frames", 2)
        from anim.schema import NPARAMS
        warm = []
        for k in range(warmup):
            t = mono()
            be.render(k, np.zeros(NPARAMS, np.float32))
            warm.append(1000 * (mono() - t))
        info = {"backend": bname, "backend_src": src, "opts": o, "init_ms": round(init_ms, 1),
                "warm_ms": [round(x, 1) for x in warm], "gpu": gpu_name(), "pid": os.getpid(), "bg_cached": cached,
                "preset": getattr(be, "preset", None), "tiers": getattr(be, "tier_names", None),
                "adaptive": getattr(be, "ctl", None) is not None,
                "surfaces": list(getattr(be, "surface_names", lambda: [])())}      # live surfaces it draws
        send("ready", info)
    except Exception:
        send("error", "backend init failed: " + traceback.format_exc()[-1500:])
        return

    note_frame = getattr(be, "note_frame", None)
    m2box = None                                            # motion2: built on the first job that carries m2
    m2tried = False
    enc_box = {"enc": None, "cfg": None, "times": []}
    encq: queue.Queue = queue.Queue(maxsize=2)
    # with a cheap CPU encoder (libx264 / jpeg at <= 640x360, ~1-2 ms a frame) encode each frame BEFORE
    # rendering the next. In parallel the encoder thread fights the next render for the GIL and a 1.5 ms encode
    # showed up as 18 ms mean / 36 ms p95 (K620, 480x270), all of it added to every frame's round trip.
    # AVATAR_ENC_SYNC=0 restores the overlapped encoder; =1 forces sync for any encoder.
    enc_sync_env = os.environ.get("AVATAR_ENC_SYNC", "auto")
    enc_done = threading.Event()

    def enc_sync():
        if enc_sync_env == "0":
            return False
        if enc_sync_env == "1":
            return True
        e = enc_box["enc"]
        return e is not None and e.name in ("libx264", "jpeg", "mjpeg") and width * height <= 640 * 360

    def open_enc(cfg, nvenc, reason=None):
        from avatar.encoder import open_encoder, X264_FAST
        t = mono()
        # x264 preset guard: once this session's encode time tripped the guard, every later open (NVENC fallback) stays
        # on the fast preset; otherwise the encoder's size-aware default (AVATAR_X264_PRESET wins).  The new encoder is
        # opened before the old one is closed, so a failed open leaves the running encoder in place.
        e = open_encoder(cfg.get("codec", "h264"), width, height, fps=cfg.get("fps", 24),
                         bitrate=cfg.get("bitrate", 2_000_000), gop_s=cfg.get("gop_s", 2.0), nvenc=nvenc,
                         threads=cfg.get("threads", 4), quality=cfg.get("quality", 80),
                         preset=X264_FAST if enc_box.get("x264_fast") else None)
        if enc_box["enc"] is not None:
            enc_box["enc"].close()
        enc_box.update(enc=e, cfg=cfg, times=[], guard=None, guard_wait=0)
        send("encoder", {"encoder": e.name, "codec": e.kind, "codec_string": e.codec_string, "errors": e.errors,
                         "open_ms": round(1000 * (mono() - t), 1), "fallback": reason,
                         "preset": getattr(e, "preset", None), "preset_guard": enc_box.get("x264_fast")})
        return e

    # x264 preset guard: the size-aware default (superfast at <= 720p) must never break the frame budget on a slow host
    # CPU.  p90 of the last ~2 s of encode times over 45 % of the frame period -> the session reopens at ultrafast at the
    # next keyframe boundary (the GOP's own IDR, or a requested one; at most one GOP of waiting) and stays there.
    # A preset pinned by AVATAR_X264_PRESET is left alone (an explicit value wins); AVATAR_X264_GUARD=0 turns it off.
    def guard_note(e, ms):
        from avatar.encoder import PresetGuard, X264_FAST
        if getattr(e, "preset", None) in (None, X264_FAST) or getattr(e, "x264_preset_src", "") == "env":
            return
        g = enc_box.get("guard")
        if g is None:
            g = enc_box["guard"] = PresetGuard(fps=(enc_box["cfg"] or {}).get("fps", 24))
        if g.note(ms):
            enc_box["x264_fast"] = g.reason
            enc_box["guard_wait"] = 0
            # one bounded second chance later in the session (a cold host's first minute); a second trip is final
            from avatar.encoder import RetryGate
            enc_box["retry"] = None if enc_box.get("retry_used") else RetryGate(g.limit_ms, mono(), fps=g.fps)
            print(f"[encoder] x264 preset guard: {g.reason}; {e.preset} -> {X264_FAST} at the next keyframe, "
                  f"for the rest of the session ({e.w}x{e.h})", file=sys.stderr, flush=True)

    def guard_switch_due(e, force_key):
        from avatar.encoder import X264_FAST
        if e.name != "libx264" or e.preset == X264_FAST or getattr(e, "x264_preset_src", "") == "env":
            return False
        enc_box["guard_wait"] = enc_box.get("guard_wait", 0) + 1
        return bool(force_key) or e.next_is_key() or enc_box["guard_wait"] > e.gop

    def guard_retry_note(e, ms):
        """On the fast preset after a trip: when the host has had room for a while, go back to the size default once."""
        r = enc_box.get("retry")
        if r is None or not r.note(ms, mono()):
            return
        enc_box.update(retry=None, retry_used=True, x264_fast=None, hq_due=True)
        print(f"[encoder] x264 preset guard: fast-preset encode p{r.q:.0f} {r.stat_ms:.1f} ms < {r.ok_ms:.1f} ms after "
              f"{r.retry_s:.0f} s; one retry of the size default at the next keyframe ({e.w}x{e.h})", file=sys.stderr, flush=True)

    def guard_reopen(e, why=None):
        """Encoder thread, just before a frame that is an IDR anyway: swap in the fast-preset encoder (its first frame
        is that IDR, announced with its codec_string like an encoder fallback)."""
        try:
            return open_enc(enc_box["cfg"], False, why or ("x264 preset guard: " + str(enc_box["x264_fast"])))
        except Exception as ex:
            send("error", f"x264 preset guard: reopen failed: {ex}")
            enc_box["guard_wait"] = -10 ** 9          # keep the running encoder; do not retry every frame
            return e

    def encoder_loop():
        while True:
            item = encq.get()
            if item is None:
                return
            kind = item[0]
            if kind == "open":
                try:
                    open_enc(item[1], item[1].get("nvenc", True))
                except Exception as ex:
                    enc_box["enc"] = None
                    send("error", f"encoder open failed: {ex}")
                continue
            if kind == "key":
                if enc_box["enc"] is not None:
                    enc_box["enc"].request_key()
                continue
            _, j, img, force_key, render_ms, t_r0 = item
            e = enc_box["enc"]
            if e is not None and enc_box.get("x264_fast") and guard_switch_due(e, force_key):
                e = guard_reopen(e)
            elif e is not None and enc_box.get("hq_due") and e.name == "libx264" and (force_key or e.next_is_key()):
                enc_box["hq_due"] = False
                e = guard_reopen(e, "x264 preset guard: retry of the size default")
            if e is None:
                send("frame", j, 0, b"", render_ms, 0.0, t_r0, mono())
                enc_done.set()
                continue
            t = mono()
            try:
                outs = e.encode(img, j, force_key)
            except Exception as ex:
                send("error", f"encode failed: {ex}")
                outs = []
            enc_ms = 1000 * (mono() - t)
            if note_frame is not None:          # the adaptive quality controller reads render + encode
                try:
                    note_frame(render_ms, enc_ms)
                except Exception:
                    pass
            if e.name == "libx264" and not enc_box.get("x264_fast"):
                guard_note(e, enc_ms)
            elif e.name == "libx264" and enc_box.get("retry") is not None:
                guard_retry_note(e, enc_ms)
            # NVENC on a GPU someone else saturates (the K620 under Blender) can take 100-600 ms a frame: when its
            # recent mean passes half a frame period, fall back to libx264 for the rest of the session (IDR first)
            if e.name == "h264_nvenc":
                tt = enc_box["times"]
                tt.append(enc_ms)
                budget = 500.0 / max(1, enc_box["cfg"].get("fps", 24))
                if len(tt) >= 12 and float(np.mean(tt[-12:])) > budget:
                    try:
                        open_enc(enc_box["cfg"], False, f"h264_nvenc mean {np.mean(tt[-12:]):.0f} ms > {budget:.0f} ms")
                    except Exception as ex:
                        send("error", f"encoder fallback failed: {ex}")            # never with zerolatency settings; keep the parent's in-flight count right anyway
                send("frame", j, 0, b"", render_ms, enc_ms, t_r0, mono())
            for fj, data, key in outs:
                flags = (1 if key else 0) | (2 if e.kind == "jpeg" else 0)
                send("frame", fj, flags, data, render_ms, enc_ms, t_r0, mono())
            enc_done.set()

    th = threading.Thread(target=encoder_loop, daemon=True)
    th.start()
    try:
        while True:
            msg = conn.recv()
            k = msg[0]
            if k == "quit":
                break
            if k == "encoder":
                encq.put(("open", msg[1]))
            elif k == "keyframe":
                encq.put(("key",))
            elif k == "surf":                  # ("surf", name, x, y, w, h, premultiplied RGBA bytes)
                fn = getattr(be, "set_surface", None)
                if fn is not None:
                    fn(*msg[1:])
            elif k == "job":
                _, j, index, params, force_key, _t_sent = msg[:6]
                m2 = msg[6] if len(msg) > 6 else None           # motion2
                if m2 is not None and not m2tried:
                    m2tried = True
                    try:
                        m2box = _motion2_hook(be)
                    except Exception as ex:                      # never stop rendering for it
                        send("error", f"motion2 hook: {ex}")
                        m2box = None
                if m2box is not None:
                    m2box[0] = m2
                t = mono()
                try:
                    img = be.render(int(index), params)
                finally:
                    if m2box is not None:
                        m2box[0] = None
                    rms = 1000 * (mono() - t)
                    send("rendered", j, rms)
                sync = enc_sync()
                if sync:
                    enc_done.clear()
                encq.put(("frame", j, img, bool(force_key), rms, t))
                if sync:
                    enc_done.wait(0.5)
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        try:
            send("error", "render failed: " + traceback.format_exc()[-1500:])
        except Exception:
            pass
    finally:
        encq.put(None)
        th.join(timeout=2)
        try:
            be.close()
        except Exception:
            pass
        if enc_box["enc"] is not None:
            enc_box["enc"].close()


if __name__ == "__main__":
    print(json.dumps({"gpu": gpu_name()}))
