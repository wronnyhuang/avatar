#!/usr/bin/env python3
"""Talking-animal renderer: headless OpenGL (EGL + moderngl), SDF character over a cached background.

  still:  render.py still --animal fox --w 1280 --h 720 --spp 2 --set jaw=0.8 round=0.5 --out a.png
  video:  render.py video --animal fox --params params.npy --start 0 --end 90 --out-dir frames/
  bake:   render.py bake  --animal fox --w 1920 --h 1080

A frame's pixels depend only on (animal, frame index, params vector, resolution, spp); the thread
count does not change them.
See docs/ANIMAL_GUIDE.md.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from anim.schema import PARAMS, NPARAMS, IDX, FPS  # noqa: E402

OUT_ROOT = os.environ.get("TA_OUT", os.path.join(os.path.dirname(ROOT), "out"))
TILE = 8           # px, screen tile of the character prepass

VS_FULL = """#version 330
in vec2 in_pos; void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }"""
VS_TILE = """#version 330
in vec2 in_pos; in float in_t; flat out float vTStart;
void main(){ vTStart = in_t; gl_PointSize = 8.0; gl_Position = vec4(in_pos, 0.0, 1.0); }"""


def backend_env(backend, threads):
    if backend == "cpu":
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"
        os.environ["LP_NUM_THREADS"] = str(threads)
        return 1
    # keep the NVIDIA on-disk shader cache on and big enough that 11 characters x the real-time programs stay in
    # it (a cold compile of a round-3 animal is ~20-35 s; a warm start reads the cache in well under a second)
    os.environ.setdefault("__GL_SHADER_DISK_CACHE", "1")
    # (kept < 2^31 in case the driver reads it as 32 bits.  Tonight's warm starts missed now and then while several
    # processes compiled different programs at once — the per-app cache file looks last-writer-wins; one render worker
    # at a time, as in production, does not race)
    os.environ.setdefault("__GL_SHADER_DISK_CACHE_SIZE", "1610612736")
    os.environ.setdefault("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1")
    return 0


def param_defines():
    """#define NPARAMS + P_<NAME>, generated from anim/schema.py — never hand-copied."""
    lines = [f"#define NPARAMS {NPARAMS}"]
    lines += [f"#define P_{name.upper()} {i}" for i, name in enumerate(PARAMS)]
    return "\n".join(lines) + "\n"


def glsl_literal(v):
    if isinstance(v, (list, tuple)):
        return f"vec{len(v)}(" + ",".join(f"{float(x):.6f}" for x in v) + ")"
    if isinstance(v, str):
        return v
    return f"{float(v):.6f}"


class Shaders:
    """String-concatenates the character + set into each pass.  No file is ever generated on disk."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.files = {
            "common": os.path.join(HERE, "common.glsl"),
            "set": os.path.join(HERE, "sets", cfg["set"] + ".glsl"),
            "animal": os.path.join(ROOT, "animals", cfg["name"] + ".glsl"),
            "lips": os.path.join(HERE, "lips.glsl"),     # a character pulls these in with its own
            "face": os.path.join(HERE, "face.glsl"),     # `#pragma include lips|face` line
            "rig": os.path.join(HERE, "body_rig.glsl"),
            "shape_arms": os.path.join(HERE, "shape_arms.glsl"),
            "props": os.path.join(HERE, "props.glsl"),            # props (models, attach, painter)
            "props_hold": os.path.join(HERE, "props_hold.glsl"),  # props (what each paw holds; before the animal)
            "places": os.path.join(HERE, "places.glsl"),          # props (the stump / rock / table props rest on)
        }
        q = dict(DEFAULT_QUALITY); q.update(cfg.get("quality", {}))
        self.head = "#version 330\n" + param_defines()
        rig_colors = {
            "fox": ((.72,.185,.022),(.115,.055,.035),(.84,.72,.55),1),
            "bear": ((.37,.21,.14),(.27,.14,.10),(.48,.31,.22),1),
            "rabbit": ((.68,.60,.55),(.83,.76,.70),(.89,.84,.79),1),
            "hippo": ((.45,.31,.43),(.66,.49,.61),(.60,.44,.54),0),
            "monkey": ((.42,.24,.14),(.51,.29,.19),(.61,.39,.24),1),
            "orb": ((.55,.72,.84),(.78,.88,.91),(.59,.75,.85),0),
            "cube": ((.75,.46,.10),(.91,.69,.31),(.77,.50,.14),0),
            "drop": ((.20,.45,.73),(.50,.73,.89),(.30,.54,.77),0),
            "star": ((.77,.47,.07),(.96,.72,.30),(.83,.55,.13),0),
            "pyramid": ((.66,.30,.20),(.86,.51,.36),(.70,.37,.27),0),
            "blob": ((.43,.30,.67),(.67,.51,.83),(.52,.36,.71),0),
            "muse": ((.86,.82,.74),(.88,.84,.77),(.90,.86,.80),1),     # body: cream plush (fuzzy)
        }
        arm, paw, chest, fur = rig_colors[cfg["name"]]
        self.head += f"#define RIG_ARM {glsl_literal(arm)}\n#define RIG_PAW {glsl_literal(paw)}\n#define RIG_CHEST {glsl_literal(chest)}\n#define RIG_FUR {fur}\n"
        # the five animals carry their own arms, paws, body and tail inside the character SDF (round 3 + the full-body rig);
        # the generic composite-pass rig (body_rig.glsl) is only for characters that do not
        self.head += "#define RIG_PASS 0\n"     # every character carries its arms in its own SDF now (shapes: engine/shape_arms.glsl)
        for k, v in q.items():
            self.head += f"#define {k} {glsl_literal(v)}\n"
        for k, v in cfg.get("set_params", {}).items():
            self.head += f"#define SP_{k.upper()} {glsl_literal(v)}\n"
        for k, v in cfg.get("char_defines", {}).items():
            self.head += f"#define {k} {glsl_literal(v)}\n"
        # props: places are part of the baked set (in the key); the props are not (frame passes only)
        import props_defs
        place_head, self.props_head = props_defs.defines(cfg["name"])
        self.head += place_head

    def resolve(self, text):
        def rep(m):
            return open(self.files[m.group(1)]).read()
        pat = r"^#pragma include (\w+)"
        while re.search(pat, text, flags=re.M):      # includes nest: the character file includes the rig
            text = re.sub(pat, rep, text, flags=re.M)
        return text

    def frag(self, fname, *defs):
        hdr = self.head + "".join(f"#define {d}\n" for d in defs)
        src = self.resolve(open(os.path.join(HERE, fname)).read())
        for name in ("SMIRK",):          # round-3 columns the live schema does not carry: read as 0
            if name.lower() not in IDX:
                src = src.replace(f"uP[P_{name}]", "0.0")
        return hdr + getattr(self, "props_head", "") + src

    @staticmethod
    def _key_text(path):
        # body: the root-motion blocks (//body-root{ .. //body-root}) are the identity in a bake (uRootOn = 0,
        # BG_NO_CHAR undefined), so they are left out of the key: the existing cached backgrounds stay valid
        b = open(path, "rb").read()
        # props: the //props-blk{ .. //props-blk} blocks are the identity in a bake without a props file
        # (places.glsl + the PLACE defines are keyed in key() / in the header), so the cached backgrounds stay valid
        b = re.sub(rb"^//props-blk\{[^\n]*\n.*?^//props-blk\}[^\n]*\n", b"", b, flags=re.S | re.M)
        # look: the //look-v2{ .. //look-v2} blocks only exist under LOOK_V2, which no bake defines
        b = re.sub(rb"^//look-v2\{[^\n]*\n.*?^//look-v2\}[^\n]*\n", b"", b, flags=re.S | re.M)
        # scene: the //scene-surf{ .. //scene-surf} blocks are the live pass's surfaces (never in a bake)
        b = re.sub(rb"^//scene-surf\{[^\n]*\n.*?^//scene-surf\}[^\n]*\n", b"", b, flags=re.S | re.M)
        if b"body-root" in b or b"CHAR_T_NEAR_RT" in b:
            b = re.sub(rb"^//body-root\{[^\n]*\n.*?^//body-root\}[^\n]*\n", b"", b, flags=re.S | re.M)
            b = b.replace(b"CHAR_T_NEAR_RT", b"CHAR_T_NEAR").replace(b"CHAR_T_FAR_RT", b"CHAR_T_FAR")
        return b

    def key(self):
        """Hash of everything the cached background depends on."""
        h = hashlib.sha1()
        for f in ("common", "set"):
            h.update(self._key_text(self.files[f]))
        if "PLACES_N" in self.head:     # props: a props file with places
            h.update(open(self.files["places"], "rb").read())
        for f in ("scene_main.glsl", "post.glsl"):
            h.update(self._key_text(os.path.join(HERE, f)))
        # the character only enters through its rest-pose shadow proxy
        src = open(self.files["animal"]).read()
        m = re.search(r"//@shadow-begin(.*?)//@shadow-end", src, re.S)
        h.update((m.group(1) if m else src).encode())
        h.update(json.dumps({k: self.cfg.get(k) for k in ("camera", "sun", "rim", "set", "set_params")}, sort_keys=True).encode())
        h.update(self.head.encode())
        return h.hexdigest()[:12]


DEFAULT_QUALITY = {
    "CHAR_T_NEAR": 1.0, "CHAR_T_FAR": 14.0,
    "CHAR_COARSE_STEPS": 26, "CHAR_FINE_STEPS": 30,
    "CHAR_SHADOW_STEPS": 22, "CHAR_AO_STEPS": 4, "CHAR_FUR_RELIEF": 0.30, "CHAR_ESCAPE_STEPS": 16,
}


def load_cfg(animal):
    with open(os.path.join(ROOT, "animals", animal + ".json")) as f:
        cfg = json.load(f)
    cfg["name"] = animal
    return cfg


def camera_basis(pos, target, roll=0.0):
    pos, target = np.array(pos, float), np.array(target, float)
    w = target - pos; w /= np.linalg.norm(w)
    u = np.cross(w, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(u, w)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        u, v = c*u + s*v, -s*u + c*v
    return u, v, w


def r2(i):
    return ((0.5 + i*0.7548776662) % 1.0, (0.5 + i*0.5698402910) % 1.0)


class Engine:
    def __init__(self, animal, w, h, spp, backend="cpu", threads=8, bg_spp=None, log=False):
        idx = backend_env(backend, threads)
        import moderngl
        self.mgl = moderngl
        self.w, self.h, self.spp, self.log = w, h, spp, log
        self.no_char = False
        self.cfg = cfg = load_cfg(animal)
        self.animal = animal
        self.ctx = ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
        self.sh = Shaders(cfg)
        cam = cfg["camera"]
        self.cam_pos = np.array(cam["pos"], float)
        self.u, self.v, self.ww = camera_basis(cam["pos"], cam["target"], cam.get("roll", 0.0))
        self.fl = 1.0/np.tan(np.radians(cam["fov"])*0.5)
        self.focus = float(cam["focus"])
        self.dofk = float(cam.get("dofk", 16.0))
        sun = np.array(cfg["sun"]["dir"], float); self.sun = sun/np.linalg.norm(sun)
        self.sun_col = np.array(cfg["sun"]["color"], float)*float(cfg["sun"].get("power", 1.0))
        rim = np.array(cfg["rim"]["dir"], float); self.rim = rim/np.linalg.norm(rim)
        self.rim_col = np.array(cfg["rim"]["color"], float)
        g = cfg.get("grade", {})
        self.grade = dict(exposure=1.0, bloom=1.0, saturation=1.08, vignette=0.28, grain=2.5,
                          gain=[1.02, 1.0, 0.96], lift=[0.004, 0.010, 0.012])
        self.grade.update(g)
        self.dapple_floor = float(cfg.get("dapple_floor", 0.45))
        self.char_key = float(cfg.get("char_key", 0.7))
        self.vol_gain = float(cfg.get("vol_gain", 1.6))
        self.bg_sun = cfg.get("bg_sun", [0.2, 0.9])    # floor + gain of the drifting light on the set
        mt = cfg.get("motes", {})
        self.motes_min = np.array(mt.get("min", [-5, -2.5, -7]), float)
        self.motes_size = np.array(mt.get("size", [10, 7, 11]), float)
        self.motes_n = int(mt.get("count", 70))
        self.mote_gain = float(mt.get("gain", 0.9))
        self.leaves_n = int(mt.get("leaves", 5))
        self.leaf_a = mt.get("leaf_a", [0.85, 0.33, 0.05])
        self.leaf_b = mt.get("leaf_b", [0.92, 0.60, 0.12])

        t0 = time.time()
        sf = self.sh.frag
        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_bg_src = sf("scene_main.glsl", "PASS_BG")
        self.p_pre = P(VS_FULL, sf("scene_main.glsl", "PASS_PRE"))
        self.p_char = P(VS_TILE, sf("scene_main.glsl", "PASS_CHAR"))
        self.p_live = P(VS_FULL, sf("scene_main.glsl", "PASS_LIVE"))
        self.p_vol = P(VS_FULL, sf("scene_main.glsl", "PASS_VOL"))
        self.p_cdof = P(VS_FULL, sf("post.glsl", "PASS_CHARDOF"))
        self.p_comp = P(VS_FULL, sf("post.glsl", "PASS_COMP"))
        self.p_post = P(VS_FULL, sf("post.glsl", "PASS_POST"))
        self.p_motes = P(sf("sprites.glsl", "VS", "MOTES"), sf("sprites.glsl", "FS", "MOTES"))
        self.p_leaves = P(sf("sprites.glsl", "VS", "LEAVES"), sf("sprites.glsl", "FS", "LEAVES"))
        if log:
            print(f"shaders compiled in {time.time()-t0:.1f}s", flush=True)
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vaos = {}
        for name in ("pre", "live", "vol", "cdof", "comp", "post"):
            self.vaos[name] = ctx.vertex_array(getattr(self, "p_"+name), [(self.quad, "2f", "in_pos")])
        self.vao_motes = ctx.vertex_array(self.p_motes, [])
        self.vao_leaves = ctx.vertex_array(self.p_leaves, [])
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        tex = lambda size, comps=4, dtype="f4", filt=True: self._tex(size, comps, dtype, filt)
        self.tw, self.th = (w+TILE-1)//TILE, (h+TILE-1)//TILE
        self.t_pre = tex((self.tw, self.th), filt=False); self.f_pre = ctx.framebuffer([self.t_pre])
        self.t_c = tex((w, h)); self.t_d = tex((w, h))
        self.f_char = ctx.framebuffer([self.t_c, self.t_d])
        self.t_c2 = tex((w, h)); self.t_d2 = tex((w, h))
        self.f_cdof = ctx.framebuffer([self.t_c2, self.t_d2])
        self.t_live = tex((w, h), dtype="f2"); self.f_live = ctx.framebuffer([self.t_live])
        self.vw, self.vh = max(w//3, 1), max(h//3, 1)
        self.t_vol = tex((self.vw, self.vh), dtype="f2"); self.f_vol = ctx.framebuffer([self.t_vol])
        self.t_hdr = tex((w, h), dtype="f2")
        self.t_hdr.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.f_hdr = ctx.framebuffer([self.t_hdr])
        self.rb_out = ctx.renderbuffer((w, h), 4)
        self.f_out = ctx.framebuffer([self.rb_out])
        self._common_uniforms()
        self.load_or_bake_bg(bg_spp if bg_spp else int(cfg.get("bg_spp", 12)))

    def _tex(self, size, comps, dtype, filt):
        t = self.ctx.texture(size, comps, dtype=dtype)
        f = self.mgl.LINEAR if filt else self.mgl.NEAREST
        t.filter = (f, f); t.repeat_x = False; t.repeat_y = False
        return t

    def set(self, prog, name, val):
        if name not in prog:
            return
        u = prog[name]
        if name == "uP":     # the GLSL linker may trim the array to the highest index a pass reads
            u.write(np.array(val[:u.array_length], dtype="f4").tobytes())
        else:
            u.value = tuple(val.tolist()) if isinstance(val, np.ndarray) else val

    def _cam_uniforms(self, p):
        self.set(p, "uRes", (self.w, self.h))
        self.set(p, "uCamPos", self.cam_pos); self.set(p, "uCamU", self.u)
        self.set(p, "uCamV", self.v); self.set(p, "uCamW", self.ww); self.set(p, "uFl", float(self.fl))
        self.set(p, "uSunDir", self.sun); self.set(p, "uSunCol", self.sun_col)
        self.set(p, "uRimDir", self.rim); self.set(p, "uRimCol", self.rim_col)
        self.set(p, "uDappleFloor", self.dapple_floor); self.set(p, "uCharKey", self.char_key)
        self.set(p, "uFocus", self.focus); self.set(p, "uDofK", self.dofk*self.h/1080.0)
        self.set(p, "uPxScale", self.h/1080.0)

    def _common_uniforms(self):
        for p in (self.p_pre, self.p_char, self.p_live, self.p_vol, self.p_cdof, self.p_comp,
                  self.p_post, self.p_motes, self.p_leaves):
            self._cam_uniforms(p)
        self.set(self.p_vol, "uRes", (self.vw, self.vh))
        self.set(self.p_pre, "uTile", float(TILE))
        self.set(self.p_vol, "uVolGain", self.vol_gain)
        self.set(self.p_live, "uBgSunFloor", float(self.bg_sun[0])); self.set(self.p_live, "uBgSunGain", float(self.bg_sun[1]))
        self.set(self.p_pre, "uTileEps", 0.022*1080.0/self.h)
        for k, u in (("exposure", "uExposure"), ("bloom", "uBloom"), ("saturation", "uSaturation"),
                     ("vignette", "uVignette"), ("grain", "uGrain")):
            self.set(self.p_post, u, float(self.grade[k]))
        self.set(self.p_post, "uGain", tuple(self.grade["gain"]))
        self.set(self.p_post, "uLift", tuple(self.grade["lift"]))
        self.set(self.p_post, "uFade", 1.0)
        for p in (self.p_motes, self.p_leaves):
            self.set(p, "uMotesMin", self.motes_min); self.set(p, "uMotesSize", self.motes_size)
            self.set(p, "uMoteGain", self.mote_gain)
            self.set(p, "uLeafColA", tuple(self.leaf_a)); self.set(p, "uLeafColB", tuple(self.leaf_b))

    # ------------------------------------------------------------------ cached background
    def bg_path(self):
        d = os.path.join(OUT_ROOT, self.animal)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"bg_{self.w}x{self.h}_{self.sh.key()}.npy")

    def load_or_bake_bg(self, spp):
        path = self.bg_path()
        if os.path.exists(path):
            arr = np.load(path, mmap_mode="r")
        else:
            arr = self.bake_bg(spp)
            tmp = path + f".tmp{os.getpid()}.npy"
            np.save(tmp, arr); os.replace(tmp, path)
        w, h = self.w, self.h
        self.t_bgA = self._tex((w, h), 4, "f4", True); self.t_bgS = self._tex((w, h), 4, "f4", True)
        self.t_bgA.write(np.array(arr[:, :, :4], dtype="f4").tobytes())
        self.t_bgS.write(np.array(arr[:, :, 4:], dtype="f4").tobytes())

    def bake_bg(self, spp):
        ctx, mgl = self.ctx, self.mgl
        t0 = time.time()
        p_bg = ctx.program(vertex_shader=VS_FULL, fragment_shader=self.p_bg_src)
        self._cam_uniforms(p_bg)
        self.set(p_bg, "uTime", 0.0); self.set(p_bg, "uFrame", 0)
        self.set(p_bg, "uP", tuple([0.0]*NPARAMS))
        vb = ctx.vertex_array(p_bg, [(self.quad, "2f", "in_pos")])
        w, h = self.w, self.h
        ta, ts = self._tex((w, h), 4, "f4", True), self._tex((w, h), 4, "f4", True)
        fb = ctx.framebuffer([ta, ts])
        fb.use(); ctx.viewport = (0, 0, w, h); fb.clear(0, 0, 0, 0)
        ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE); ctx.blend_equation = mgl.FUNC_ADD
        strips = 8; sh = (h + strips - 1)//strips
        for s in range(spp):
            self.set(p_bg, "uJitter", r2(s))
            for i in range(strips):
                ctx.scissor = (0, i*sh, w, min(sh, h - i*sh))
                vb.render(mgl.TRIANGLE_STRIP)
                ctx.finish()
            if self.log:
                print(f"bg sample {s+1}/{spp} {time.time()-t0:.1f}s", flush=True)
        ctx.scissor = None
        ctx.disable(mgl.BLEND)
        # average
        a = np.frombuffer(ta.read(), dtype="f4").reshape(h, w, 4).copy()/spp
        s_ = np.frombuffer(ts.read(), dtype="f4").reshape(h, w, 4).copy()/spp
        if self.log:
            print(f"bg bake {w}x{h} {spp}spp in {time.time()-t0:.1f}s, lens...", flush=True)
        arr = self.bg_lens(np.concatenate([a, s_], axis=2))
        if self.log:
            print(f"bg done in {time.time()-t0:.1f}s", flush=True)
        return arr

    def bg_lens(self, img):
        """Depth of field is baked: the camera and focus never change.  Layered FFT disc blur, far to
        near; no gather ringing, no contour lines, bright sky bleeds like real bokeh."""
        from scipy.signal import fftconvolve
        h, w = img.shape[:2]
        px = h/1080.0
        depth = img[:, :, 3].copy()
        coc = np.clip(self.dofk*np.abs(1.0/self.focus - 1.0/np.maximum(depth, 0.05)), 0.0, 44.0)*px
        radii = [0.0] + [r*px for r in (2, 4, 6.5, 9.5, 13, 17, 22, 28, 35, 46)]
        pad = int(radii[-1]*1.2) + 2
        chans = [0, 1, 2, 3, 4, 5, 6]
        P = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        C = np.pad(coc, pad, mode="edge")
        out = np.zeros(P.shape[:2] + (8,), dtype="f4")
        for i in range(len(radii)-1, -1, -1):
            r = radii[i]
            lo = radii[i-1] if i > 0 else -1.0
            hi = radii[i+1] if i+1 < len(radii) else 1e9
            wgt = np.where(C < r, (C - lo)/max(r - lo, 1e-6), (hi - C)/max(hi - r, 1e-6))
            wgt = np.clip(wgt, 0, 1).astype("f4")
            if wgt.max() <= 0:
                continue
            if r < 0.5:
                aL = wgt
                layer = [P[:, :, c]*wgt for c in chans]
            else:
                R = int(np.ceil(r*1.15)) + 1
                yy, xx = np.mgrid[-R:R+1, -R:R+1]
                k = np.clip(1.0 - (xx*xx + yy*yy)/(r*1.15)**2, 0, 1).astype("f4")**1.5   # soft disc: no edge echoes
                k /= k.sum()
                aL = np.clip(fftconvolve(wgt, k, mode="same"), 0, 1).astype("f4")
                layer = [fftconvolve(P[:, :, c]*wgt, k, mode="same").astype("f4") for c in chans]
            for j, c in enumerate(chans):
                out[:, :, c] = out[:, :, c]*(1.0 - aL) + layer[j]
            out[:, :, 7] = out[:, :, 7]*(1.0 - aL) + aL
        wsum = np.maximum(out[:, :, 7:8], 1e-4)
        out[:, :, :7] /= wsum
        res = out[pad:pad+h, pad:pad+w].copy()
        res[:, :, 7] = 1.0
        return res

    # ------------------------------------------------------------------ one frame
    def render(self, frame, params, profile=None):
        ctx, mgl = self.ctx, self.mgl
        t = frame/float(FPS)
        P = tuple(float(x) for x in params)
        P += (0.0,)*(NPARAMS - len(P))     # older, shorter vectors: the columns added since are 0
        tick = time.perf_counter()
        def lap(name):
            nonlocal tick
            if profile is not None:
                ctx.finish()
                now = time.perf_counter(); profile[name] = profile.get(name, 0.0) + now - tick; tick = now
        w, h = self.w, self.h

        # 1. prepass: which tiles does the character touch, and from what distance
        self.set(self.p_pre, "uP", P); self.set(self.p_pre, "uTime", t)
        self.f_pre.use(); ctx.viewport = (0, 0, self.tw, self.th)
        self.vaos["pre"].render(mgl.TRIANGLE_STRIP)
        pre = np.frombuffer(self.t_pre.read(), dtype="f4").reshape(self.th, self.tw, 4)[:, :, 0]
        k = np.array(pre)
        # 3x3 min filter
        pad = np.pad(k, 1, mode="constant", constant_values=1e6)
        k = np.min(np.stack([pad[1+dy:1+dy+self.th, 1+dx:1+dx+self.tw] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
        ys, xs = np.nonzero(k < 1e5)
        if self.no_char:
            ys, xs = ys[:0], xs[:0]
        tiles = len(xs)
        vbuf = np.empty((tiles*6, 3), dtype="f4")
        if tiles:
            x0 = (xs*TILE)/w*2 - 1; x1 = ((xs+1)*TILE)/w*2 - 1
            y0 = (ys*TILE)/h*2 - 1; y1 = ((ys+1)*TILE)/h*2 - 1
            ts = np.maximum(k[ys, xs] - 0.12, 0.2)
            # points, not two triangles: quads split by a tile diagonal get wrong normals on llvmpipe
            vbuf = np.stack([(x0 + x1)*0.5, (y0 + y1)*0.5, ts], axis=1).astype("f4")
        lap("prepass")

        # 2. character samples, additive; order is fixed so the result is deterministic
        self.f_char.use(); ctx.viewport = (0, 0, w, h); self.f_char.clear(0, 0, 0, 0)
        if tiles:
            vb = ctx.buffer(vbuf.tobytes())
            vao = ctx.vertex_array(self.p_char, [(vb, "2f 1f", "in_pos", "in_t")])
            self.set(self.p_char, "uP", P); self.set(self.p_char, "uTime", t); self.set(self.p_char, "uFrame", frame)
            ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE)
            for s in range(self.spp):
                self.set(self.p_char, "uJitter", r2(s + 1 + (frame % 4)*self.spp))
                vao.render(mgl.POINTS)
            ctx.disable(mgl.BLEND)
            vao.release(); vb.release()
        lap("char")

        # 3. small DoF on the character, premultiplied (also the edge clean-up)
        self.f_cdof.use(); ctx.viewport = (0, 0, w, h)
        self.t_c.use(0); self.t_d.use(1)
        # the sample sum -> average lives in the shader's input scale
        self.set(self.p_cdof, "uChar", 0); self.set(self.p_cdof, "uCharD", 1)
        self.set(self.p_cdof, "uInvSpp", 1.0/self.spp); self.set(self.p_cdof, "uFrame", frame)
        if tiles:
            xa, xb = max(xs.min()-2, 0)*TILE, min(xs.max()+3, self.tw)*TILE
            ya, yb = max(ys.min()-2, 0)*TILE, min(ys.max()+3, self.th)*TILE
            self.f_cdof.clear(0, 0, 0, 0)
            ctx.scissor = (int(xa), int(ya), int(min(xb, w)-xa), int(min(yb, h)-ya))
            self.vaos["cdof"].render(mgl.TRIANGLE_STRIP)
            ctx.scissor = None
        lap("char_dof")

        # 4. living background
        self.f_live.use(); ctx.viewport = (0, 0, w, h)
        self.t_bgA.use(0); self.t_bgS.use(1)
        self.set(self.p_live, "uBgA", 0); self.set(self.p_live, "uBgS", 1); self.set(self.p_live, "uTime", t)
        self.vaos["live"].render(mgl.TRIANGLE_STRIP)
        # sprites behind the character
        self.sprites(t, frame, 0)
        lap("live_bg")

        # 5. god rays, 1/3 res
        self.f_vol.use(); ctx.viewport = (0, 0, self.vw, self.vh)
        self.t_bgA.use(0)
        self.set(self.p_vol, "uDepth", 0); self.set(self.p_vol, "uTime", t); self.set(self.p_vol, "uFrame", frame)
        self.vaos["vol"].render(mgl.TRIANGLE_STRIP)
        lap("godrays")

        # 6. composite + front sprites + bloom chain + grade
        self.f_hdr.use(); ctx.viewport = (0, 0, w, h)
        self.t_live.use(0); self.t_vol.use(1); self.t_c2.use(2); self.t_d2.use(3)
        self.set(self.p_comp, "uLive", 0); self.set(self.p_comp, "uVol", 1)
        self.set(self.p_comp, "uChar", 2); self.set(self.p_comp, "uCharD", 3)
        self.set(self.p_comp, "uHasChar", 1.0 if tiles else 0.0)
        self.vaos["comp"].render(mgl.TRIANGLE_STRIP)
        self.sprites(t, frame, 1)
        self.t_hdr.build_mipmaps()
        self.f_out.use(); ctx.viewport = (0, 0, w, h)
        self.t_hdr.use(0)
        self.set(self.p_post, "uHdr", 0); self.set(self.p_post, "uFrame", frame)
        self.vaos["post"].render(mgl.TRIANGLE_STRIP)
        data = self.f_out.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[::-1]
        lap("post")
        return img

    def sprites(self, t, frame, layer):
        ctx, mgl = self.ctx, self.mgl
        ctx.enable(mgl.BLEND)
        for p in (self.p_motes, self.p_leaves):
            self.set(p, "uTime", t); self.set(p, "uFrame", frame); self.set(p, "uLayer", layer)
        ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA, mgl.ZERO, mgl.ONE)
        self.vao_leaves.render(mgl.TRIANGLE_STRIP, vertices=4, instances=self.leaves_n)
        ctx.blend_func = (mgl.ONE, mgl.ONE, mgl.ZERO, mgl.ONE)
        self.vao_motes.render(mgl.POINTS, vertices=self.motes_n)
        ctx.disable(mgl.BLEND)


def parse_sets(items):
    v = np.zeros(NPARAMS, dtype="f4")
    for it in items or []:
        k, x = it.split("=")
        if k not in IDX:
            raise SystemExit(f"unknown param {k}; known: {' '.join(PARAMS)}")
        v[IDX[k]] = float(x)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "video", "bake"])
    ap.add_argument("--animal", default="fox")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--bg-spp", type=int, default=0)
    ap.add_argument("--backend", default=os.environ.get("TA_BACKEND", "cpu"), choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TA_THREADS", "8")))
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="still mode: param=value overrides")
    ap.add_argument("--params", help="frames x NPARAMS .npy")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--out", default="still.png")
    ap.add_argument("--out-dir")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-char", action="store_true", help="background only (set preview)")
    a = ap.parse_args()
    from PIL import Image
    t0 = time.time()
    eng = Engine(a.animal, a.w, a.h, a.spp, a.backend, a.threads, a.bg_spp or None, log=True)
    print(f"engine ready in {time.time()-t0:.1f}s", flush=True)
    eng.no_char = a.no_char
    if a.mode == "bake":
        return
    if a.mode == "still":
        prof = {} if a.profile else None
        t1 = time.time()
        img = eng.render(a.frame, parse_sets(a.set), prof)
        Image.fromarray(img).save(a.out)
        print(f"still frame={a.frame} in {time.time()-t1:.2f}s -> {a.out}", flush=True)
        if prof:
            print(" ".join(f"{k}={v:.2f}" for k, v in prof.items()))
        return
    arr = np.load(a.params)
    assert arr.shape[1] <= NPARAMS, f"params.npy has {arr.shape[1]} columns, schema has {NPARAMS}"
    end = arr.shape[0] if a.end < 0 else min(a.end, arr.shape[0])
    os.makedirs(a.out_dir, exist_ok=True)
    for i in range(a.start, end):
        t1 = time.time()
        img = eng.render(i, arr[i])
        Image.fromarray(img).save(os.path.join(a.out_dir, f"{i:06d}.png"), compress_level=1)
        print(f"frame {i} {time.time()-t1:.2f}s", flush=True)


if __name__ == "__main__":
    main()
