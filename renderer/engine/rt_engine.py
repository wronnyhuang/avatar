#!/usr/bin/env python3
"""Real-time variant of the v2 engine (engine/render.py) for the live loop: the SAME scene, set and character
shaders, re-plumbed so a frame never waits on the CPU and the expensive character march can run at lower cost.

What changes against render.Engine (the look-defining code is untouched; render.py is not modified):
  * no CPU round trip: the 1/8-res prepass is min-filtered on the GPU (PASS_MASK) and the character pass draws one
    point per tile with GPU culling in its vertex shader (VS_CHARPTS) - no readback, no per-frame buffers or VAOs;
  * the character's DoF is fused into the composite (PASS_COMPDOF), only inside the character's tiles;
  * the final frame is read back through pixel-buffer objects (render() sync, or submit()/fetch() pipelined);
  * presets: march step counts, DoF taps, god-ray samples, an internal resolution for the character pass, and a
    temporal resolve (TAA / temporal upsampling, causal: history of past frames only).
Quality preset "v2" reproduces the shipped settings (spp 3, full res, no TAA) for checking the plumbing.

  python engine/rt_engine.py --animal fox --w 960 --h 540 --preset fast --frames 60
"""
import argparse, copy, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import render  # noqa: E402
from render import (Shaders, DEFAULT_QUALITY, load_cfg, camera_basis, r2, backend_env, VS_FULL, TILE)  # noqa: E402
from anim.schema import NPARAMS, FPS, IDX  # noqa: E402

# ---------------------------------------------------------------------------------------- body: root + camera
# docs/INTENT-RULES.md section 1.  Body facts per character: renderer/body/<character>.json (H, origin, com, bound).
BODY_DIR = os.path.join(ROOT, "body")
ROOT_COLS = [IDX[n] for n in ("root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash")]
CAM_COLS = [IDX[n] for n in ("cam_zoom", "cam_x", "cam_y")]
SQUASH_K = 0.55        # squash s: height x exp(-0.55 s), width and depth x exp(+0.275 s) (volume kept)
CAM_WIDE_H = 2.2       # cam_zoom 1: the frame's height at the character = 2.2 H (the character ~45% of it, ~±2 H across)
CAM_WIDE_AIM = 0.62    # cam_zoom 1: aimed 0.62 H above the floor under the character (room above the head for jumps)
CAM_IN = 2.0           # scene: cam_zoom -1 = the rest view 2x tighter (the station cameras, docs/SCENE.md 6)
PAN_K = 1.6            # the wide panorama bake: tan extent PAN_K x the cam_zoom 1 view (room for cam_x/cam_y pans)
PAN_RS = 2.0           # ... at PAN_RS x the output resolution
PAN_SPP = 3
MOTION_W = 0.5      # max extra TAA weight of the current frame under fast root motion


def load_body(animal, cfg):
    """renderer/body/<animal>.json, or a rough fallback from the camera framing."""
    try:
        b = json.load(open(os.path.join(BODY_DIR, animal + ".json")))
        H = float(b["H"]); org = np.array(b["origin_scene"], float)
        com = org + np.array(b["com"], float)*H
        bd = b.get("bound") or {}
        bc = np.array(bd.get("center_scene", com), float); br = float(bd.get("radius_scene", 1.2*H))
        return dict(H=H, origin=org, C=com, B=np.array([com[0], org[1], com[2]]), bc=bc, br=br, src="body json")
    except (OSError, ValueError, KeyError):
        cam = cfg["camera"]
        vis = 2.0*np.linalg.norm(np.subtract(cam["target"], cam["pos"]))*np.tan(np.radians(cam["fov"])*0.5)
        H = 0.8*vis; gy = float(cfg.get("set_params", {}).get("ground_y", cam["target"][1] - 0.45*vis))
        org = np.array([0.0, gy, 0.0]); com = org + np.array([0.0, 0.45*H, 0.0])
        return dict(H=H, origin=org, C=com, B=org.copy(), bc=com, br=1.0*H, src="fallback")


def _rot(yaw, pitch, roll):
    """R = Ry(yaw) Rx(pitch) Rz(roll), angles in turns; + yaw = the front swings to screen-right, + pitch = the top
    tips toward the camera, + roll = the top to screen-right."""
    a, b, c = 2*np.pi*yaw, 2*np.pi*pitch, -2*np.pi*roll
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def _m3(m):
    return tuple(float(v) for v in np.asarray(m, float).flatten(order="F"))     # GLSL mat3 = column-major

STEP_KEYS = {"coarse": "CHAR_COARSE_STEPS", "fine": "CHAR_FINE_STEPS", "shadow": "CHAR_SHADOW_STEPS",
             "ao": "CHAR_AO_STEPS", "escape": "CHAR_ESCAPE_STEPS"}

# A preset is a dict; every key optional (defaults below).  steps = multipliers on the character's own step counts.
PRESET_DEFAULTS = dict(scale=1.0, spp=1, taa=False, alpha=0.50, alpha_miss=0.30, gamma=1.0, reproj=True,
                       steps=dict(coarse=1.0, fine=1.0, shadow=1.0, ao=1.0, escape=1.0), relief=None,
                       dof_n=18, vol_nv=24, jitter_n=8, pstart=0.0, roi=0.0, roi_feather=6.0,
                       look=True,      # the look upgrade (LOOK_V2 shader blocks); False = the picture before it
                       tiers=False)    # True = build the ladder of internal resolutions (set_tier / adaptive)
# look: gains (a character's json may override any under "look"): key gain, fill gain, back-rim gain (+ its
# colour: null = the set's rim colour; + its tightness), wide-occlusion strength, velvet (fuzz rim on pile), gloss on
# hard surfaces, the floor pool under a standing character, and the grade (clarity, contrast, grain scale)
LOOK_DEFAULTS = dict(key=1.0, fill=1.0, rim=1.2, rim_col=None, rim_pow=2.2, wide=1.6, velvet=0.5, gloss=0.5, floor=0.0,
                     clarity=0.25, contrast=0.0, grain=0.35)
# adaptive tiers: the character's internal resolution steps through this ladder (absolute scales), between
# half the preset's scale and full resolution; the lowest tier also drops the wide occlusion taps
TIER_SCALES = [0.25, 0.30, 0.36, 0.42, 0.50, 0.60, 0.68, 0.75, 0.87, 1.0]
S_STEPS = dict(coarse=0.8, fine=0.8, shadow=0.6, ao=0.75, escape=0.6)       # "_s": ~20% fewer march steps
SS_STEPS = dict(coarse=0.7, fine=0.6, shadow=0.45, ao=0.5, escape=0.5)     # "_ss": ~35% fewer
PRESETS = {
    # the shipped settings through the real-time plumbing (spp 3, no temporal): checks the plumbing against v2
    "v2":      dict(spp=3),
    "full1":   dict(spp=1),                                   # 1 centred sample, no temporal
    "s75":     dict(scale=0.75),                              # spatial only: character at 0.75x, bilinear up
    "s50":     dict(scale=0.5),
    # temporal resolve (jittered 1 spp + history clipped to this frame's neighbourhood, head-pose reprojection)
    "taa":     dict(taa=True),
    # for a big GPU at 1280x720: the character at full resolution, temporal (1 spp/frame); "v2" is the exact look
    "quality": dict(taa=True),
    "q75":     dict(taa=True, scale=0.75),
    "q60":     dict(taa=True, scale=0.6),
    "q50":     dict(taa=True, scale=0.5),
    "q45":     dict(taa=True, scale=0.45),                  # the fox with arms raised by its face at 480x270
    "q36":     dict(taa=True, scale=0.36),                  # c3: the monkey framed closer
    "q40":     dict(taa=True, scale=0.4),
    "q33":     dict(taa=True, scale=0.333),
    "q25":     dict(taa=True, scale=0.25),
    # + the mouth box (engine/rt_mouth_boxes.json) marched at full output resolution
    "q60r":    dict(taa=True, scale=0.6, roi=1.0),
    "q50r":    dict(taa=True, scale=0.5, roi=1.0),
    "q40r":    dict(taa=True, scale=0.4, roi=1.0),
    "q33r":    dict(taa=True, scale=0.333, roi=1.0),
    # fewer march steps: measured on fox 640x360 they save <10% of the frame and cost 3-9 dB (tools/rt_bench.py runs
    # in REPORT-rt-gpu.md) - kept for the record, not in the ladder
    "taa_s":   dict(taa=True, steps=S_STEPS),
    "q50_s":   dict(taa=True, scale=0.5, steps=S_STEPS),
    "q50_ss":  dict(taa=True, scale=0.5, steps=SS_STEPS, dof_n=12),
}
LADDER = ["q25", "q33", "q40", "q50", "q60", "q75", "taa"]      # cheapest -> best: the character's internal resolution
ROI_LADDER = ["q33r", "q40r", "q50r", "q60r"]           # + the mouth at full resolution


# Optional (preset key pstart = margin, world units): start the march just in front of where last frame's ray hit
# solid character, if this pixel and its 4 neighbours (2 px away) were solid last frame.  Saves most of the coarse
# envelope march on the still body; an edge, an empty pixel or a tile not marched last frame falls back to the tile.
PSTART_GLSL = """
uniform sampler2D uPrevD, uPrevMask;
uniform float uPStart;
float rtPrevStart(float t){
  if(uPStart <= 0.0) return t;
  ivec2 ip = ivec2(gl_FragCoord.xy), lim = textureSize(uPrevD, 0) - 1;
  if(texelFetch(uPrevMask, ip/8, 0).g < 0.5) return t;
  float dmin = 1e9;
  for(int k=0; k<5; k++){
    ivec2 o = k == 0 ? ivec2(0) : (k == 1 ? ivec2(2,0) : (k == 2 ? ivec2(-2,0) : (k == 3 ? ivec2(0,2) : ivec2(0,-2))));
    vec4 d = texelFetch(uPrevD, clamp(ip + o, ivec2(0), lim), 0);
    if(d.a < 0.98) return t;
    dmin = min(dmin, d.r/d.a);
  }
  return max(t, dmin - uPStart);
}
"""

POSE_MAIN = """
out vec4 fragPose;
void main(){
  setInit(); charInit();
  int i = int(gl_FragCoord.x);
  mat3 R = uRootOn > 0.5 ? gHR*transpose(uRootRm) : gHR;      // body: world -> head with the root composed
  fragPose = i < 3 ? vec4(R[i], 0.0) : vec4(rootFwd(gHO), 1.0);
}
"""


# props: the props' poses once per frame (16 x 4 texels (16 slots): centre + articulation, then the world -> unit rotation),
# read by the live background pass for their shadows / contact shade on the set
PROPS_SLOTS = 16       # the most prop slots a character may have (props_defs: props[:16]; props.glsl arrays, masks)
PROPS_POSE_MAIN = """
out vec4 fragPP;
void main(){
  setInit(); charInit();
  int k = int(gl_FragCoord.x), r = int(gl_FragCoord.y);
#if PROPS_N > 0
  if(k < PROPS_N){ fragPP = r == 0 ? vec4(gPrC[k], gPrArt[k]) : vec4(gPrM[k][r - 1], r == 1 ? gPrRest[k] : 0.0); return; }
#endif
  fragPP = vec4(0.0, -1e4, 0.0, 0.0);
}
"""


def resolve_preset(preset):
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset}; known: {' '.join(PRESETS)}")
        preset = PRESETS[preset]
    p = copy.deepcopy(PRESET_DEFAULTS)
    for k, v in (preset or {}).items():
        if k == "steps":
            p["steps"].update(v)
        else:
            p[k] = v
    return p


def _patch(src, old, new, count=1):
    n = src.count(old)
    if n != count:
        raise RuntimeError(f"rt_engine patch target found {n}x (want {count}): {old[:60]!r}")
    return src.replace(old, new)


class RTEngine(render.Engine):
    """render.Engine's scene, faster plumbing.  render(frame, params) -> (h, w, 3) uint8, top row first."""

    def __init__(self, animal, w, h, preset="taa", backend="gpu", threads=8, log=False, bg_spp=None):
        idx = backend_env(backend, threads)
        import moderngl
        self.mgl = moderngl
        self.P = P_ = resolve_preset(preset)
        self.w, self.h, self.log = w, h, log
        self.spp = int(P_["spp"])
        self.taa = bool(P_["taa"])
        self.scale = float(P_["scale"])
        self.no_char = False
        self.cfg = cfg = load_cfg(animal)
        self.animal = animal
        self.body = load_body(animal, cfg)          # body
        self.root_ok = bool(cfg.get("char_defines", {}).get("CHAR_ROOT"))      # body: compiled with the root wrappers
        self.ctx = ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
        # --- same camera / light / grade as render.Engine
        cam = cfg["camera"]
        self.cam_pos = np.array(cam["pos"], float)
        self.u, self.v, self.ww = camera_basis(cam["pos"], cam["target"], cam.get("roll", 0.0))
        self.fl = 1.0/np.tan(np.radians(cam["fov"])*0.5)
        self.focus = float(cam["focus"]); self.dofk = float(cam.get("dofk", 16.0))
        sun = np.array(cfg["sun"]["dir"], float); self.sun = sun/np.linalg.norm(sun)
        self.sun_col = np.array(cfg["sun"]["color"], float)*float(cfg["sun"].get("power", 1.0))
        rim = np.array(cfg["rim"]["dir"], float); self.rim = rim/np.linalg.norm(rim)
        self.rim_col = np.array(cfg["rim"]["color"], float)
        self.grade = dict(exposure=1.0, bloom=1.0, saturation=1.08, vignette=0.28, grain=2.5,
                          gain=[1.02, 1.0, 0.96], lift=[0.004, 0.010, 0.012])
        self.grade.update(cfg.get("grade", {}))
        self.dapple_floor = float(cfg.get("dapple_floor", 0.45)); self.char_key = float(cfg.get("char_key", 0.7))
        self.vol_gain = float(cfg.get("vol_gain", 1.6)); self.bg_sun = cfg.get("bg_sun", [0.2, 0.9])
        mt = cfg.get("motes", {})
        self.motes_min = np.array(mt.get("min", [-5, -2.5, -7]), float)
        self.motes_size = np.array(mt.get("size", [10, 7, 11]), float)
        self.motes_n = int(mt.get("count", 70)); self.mote_gain = float(mt.get("gain", 0.9))
        self.leaves_n = int(mt.get("leaves", 5))
        self.leaf_a = mt.get("leaf_a", [0.85, 0.33, 0.05]); self.leaf_b = mt.get("leaf_b", [0.92, 0.60, 0.12])

        # --- shaders.  The background cache key comes from the UNMODIFIED config, so every engine shares bakes.
        self.sh_bg = Shaders(cfg)
        q = dict(DEFAULT_QUALITY); q.update(cfg.get("quality", {}))
        for k, key in STEP_KEYS.items():
            q[key] = max(1, int(round(q[key]*float(P_["steps"].get(k, 1.0)))))
        if P_["relief"] is not None:
            q["CHAR_FUR_RELIEF"] = float(P_["relief"])
        self.quality = q
        cfg2 = dict(cfg); cfg2["quality"] = q
        self.sh = Shaders(cfg2)
        self.sh.head += f"#define DOF_N {int(P_['dof_n'])}\n#define VOL_NV {int(P_['vol_nv'])}\n"
        self.look = bool(P_["look"]) and os.environ.get("AVATAR_LOOK", "1") != "0"      # the look upgrade
        self.look_k = dict(LOOK_DEFAULTS); self.look_k.update(cfg.get("look", {}))
        self.look_k.update(json.loads(os.environ.get("AVATAR_LOOK_K", "{}")))      # tuning only
        if self.look:
            self.sh.head += "#define LOOK_V2 1\n"
        t0 = time.time()
        # props: the frame passes read the props' tables and per-frame poses as uniforms (PROPS_UNI; the poses
        # come from the tiny p_ppose pass, read back once per frame) — no per-fragment prop state on the GPU
        self.has_props = "PROPS_N 0\n" not in getattr(self.sh, "props_head", "PROPS_N 0\n")
        scene0 = lambda *d: self.sh.frag("scene_main.glsl", *d)
        scene = (lambda *d: scene0(*d, "PROPS_UNI") if "PASS_NONE" not in d else scene0(*d)) if self.has_props else scene0
        rt = lambda *d: self.sh.frag("rt_passes.glsl", *d)
        char_src = scene("PASS_CHAR")
        char_src = _patch(char_src, "  vec3 ro = uCamPos, rd = camRay(pix);\n  float t = vTStart;",
                          "  vec3 ro = uCamPos, rd = camRay(pix);\n"
                          "  if(vTStart < 0.0){ fragC = vec4(0.0); fragD = vec4(0.0); return; }\n  float t = vTStart;\n"
                          "  t = rtPrevStart(t);")
        char_src = _patch(char_src, "flat in float vTStart;\n", "flat in float vTStart;\n" + PSTART_GLSL +
                          "uniform float uPixScale;\nuniform vec2 uPixOrg;\n")
        char_src = _patch(char_src, "  vec2 pix = gl_FragCoord.xy - 0.5 + uJitter;\n  vec3 ro = uCamPos, rd = camRay(pix);\n  if(vTStart",
                          "  vec2 pix = (gl_FragCoord.xy - 0.5 + uJitter)*uPixScale + uPixOrg;\n"
                          "  vec3 ro = uCamPos, rd = camRay(pix);\n  if(vTStart")
        vol_src = _patch(scene("PASS_VOL"), "const int NV = 24;\n  const float DT", "const int NV = VOL_NV;\n  const float DT")
        vs_pts = rt("VS_CHARPTS")
        Pg = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_pre = Pg(VS_FULL, scene("PASS_PRE"))
        self.p_mask = Pg(VS_FULL, rt("PASS_MASK"))
        self.p_char = Pg(vs_pts, char_src)
        self.roi = float(P_["roi"]) if self.taa else 0.0
        self.p_roi = Pg(rt("VS_ROIPTS"), char_src) if self.roi > 0 else None
        self.p_live = Pg(VS_FULL, scene("PASS_LIVE", "PROPS_LIVE") if self.has_props else scene("PASS_LIVE"))
        self.p_ppose = Pg(VS_FULL, scene0("PASS_NONE") + PROPS_POSE_MAIN) if self.has_props else None   # props
        self.p_vol = Pg(VS_FULL, vol_src)
        self.p_res = Pg(rt("VS_TILEQUAD"), rt("PASS_RESOLVE"))
        self.p_pose = Pg(VS_FULL, scene("PASS_NONE") + POSE_MAIN)
        self.p_comp = Pg(VS_FULL, rt("PASS_COMPDOF"))
        self.p_post = Pg(VS_FULL, self.sh.frag("post.glsl", "PASS_POST"))
        self.p_bgv = Pg(VS_FULL, rt("PASS_BGVIEW"))          # body: the background for a moved camera / root
        self.p_motes = Pg(self.sh.frag("sprites.glsl", "VS", "MOTES"), self.sh.frag("sprites.glsl", "FS", "MOTES"))
        self.p_leaves = Pg(self.sh.frag("sprites.glsl", "VS", "LEAVES"), self.sh.frag("sprites.glsl", "FS", "LEAVES"))
        self.init_times = {"compile_s": round(time.time() - t0, 2)}
        if log:
            print(f"rt shaders compiled in {time.time()-t0:.1f}s  quality={q}", flush=True)
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vaos = {n: ctx.vertex_array(getattr(self, "p_"+n), [(self.quad, "2f", "in_pos")])
                     for n in ("pre", "mask", "live", "vol", "comp", "post", "pose", "bgv")}
        if self.has_props:      # props
            self.vaos["ppose"] = ctx.vertex_array(self.p_ppose, [(self.quad, "2f", "in_pos")])
            self.t_ppose = ctx.texture((PROPS_SLOTS, 4), 4, dtype="f4"); self.t_ppose.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.f_ppose = ctx.framebuffer([self.t_ppose])
        self._surf_init()          # scene: the board / TV live surfaces (muse only)
        self.vao_char = ctx.vertex_array(self.p_char, [])
        self.vao_roi = ctx.vertex_array(self.p_roi, []) if self.p_roi else None
        self.vao_res = ctx.vertex_array(self.p_res, [])
        self.vao_motes = ctx.vertex_array(self.p_motes, [])
        self.vao_leaves = ctx.vertex_array(self.p_leaves, [])
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # --- targets
        tex = lambda size, comps=4, dtype="f4", filt=True: self._tex(size, comps, dtype, filt)
        # the character-side targets exist once per tier (internal resolution); without tiers, one level
        scales = [self.scale]
        if P_["tiers"] and self.taa:
            lo = max(0.25, 0.5*self.scale)
            scales = sorted([sc for sc in TIER_SCALES if lo - 1e-6 <= sc <= 1.0 and abs(sc/self.scale - 1.0) > 0.07] + [self.scale])
        self.levels = [self._alloc_level(sc) for sc in scales]
        self.tier = scales.index(self.scale)
        self.tier_pending = None
        self._tier_switched = False
        self._use_level(self.tier, uniforms=False)
        self.t_pose = [tex((4, 1), filt=False) for _ in range(2)]
        self.f_pose = [ctx.framebuffer([t]) for t in self.t_pose]
        if self.taa:
            self.t_hc = [tex((w, h), 4, "f2") for _ in range(2)]
            self.t_hd = [tex((w, h), 4, "f2") for _ in range(2)]
            self.f_hist = [ctx.framebuffer([self.t_hc[i], self.t_hd[i]]) for i in range(2)]
            for f in self.f_hist:
                f.clear(0, 0, 0, 0)
        self.roi_box = (0, 0, 0, 0)
        if self.roi > 0:
            box = json.load(open(os.path.join(HERE, "rt_mouth_boxes.json")))[animal]     # top-down, normalized
            X0, X1 = int(np.floor(box[0]*w)), int(np.ceil(box[2]*w))
            Y0, Y1 = int(np.floor((1.0 - box[3])*h)), int(np.ceil((1.0 - box[1])*h))
            self.roi_box = (X0, Y0, X1, Y1)
            self.rw, self.rh = int(np.ceil((X1 - X0)*self.roi)), int(np.ceil((Y1 - Y0)*self.roi))
            self.rtw, self.rth = (self.rw + TILE - 1)//TILE, (self.rh + TILE - 1)//TILE
            self.t_rc = tex((self.rw, self.rh)); self.t_rd = tex((self.rw, self.rh))
            self.f_roi = ctx.framebuffer([self.t_rc, self.t_rd])
        self.t_live = tex((w, h), dtype="f2"); self.f_live = ctx.framebuffer([self.t_live])
        self.t_bva, self.t_bvs = tex((w, h)), tex((w, h))           # body: PASS_BGVIEW's layers
        self.f_bv = ctx.framebuffer([self.t_bva, self.t_bvs])
        self.vw, self.vh = max(w//3, 1), max(h//3, 1)
        self.t_vol = tex((self.vw, self.vh), dtype="f2"); self.f_vol = ctx.framebuffer([self.t_vol])
        self.t_hdr = tex((w, h), dtype="f2")
        self.t_hdr.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.f_hdr = ctx.framebuffer([self.t_hdr])
        self.rb_out = ctx.renderbuffer((w, h), 4)
        self.f_out = ctx.framebuffer([self.rb_out])
        self.pbo = [ctx.buffer(reserve=w*h*4) for _ in range(2)]
        self.pending = []           # (pbo index, frame) submitted, not yet fetched
        self.nsub = 0
        self.frames_in_hist = 0

        self._common_uniforms()
        # the background: identical to render.Engine (shared cache under $TA_OUT)
        sh_frame = self.sh
        self.sh = self.sh_bg
        self.p_bg_src = self.sh_bg.frag("scene_main.glsl", "PASS_BG")
        t1 = time.time()
        self.load_or_bake_bg(bg_spp if bg_spp else int(cfg.get("bg_spp", 12)))
        self.init_times["bg_s"] = round(time.time() - t1, 2)
        self.sh = sh_frame
        self.jit = self._jitter_seq(int(P_["jitter_n"]))
        t1 = time.time()
        self._body_init(bg_spp if bg_spp else int(cfg.get("bg_spp", 12)))      # body
        self.init_times["body_s"] = round(time.time() - t1, 2)

    # ------------------------------------------------------------------ scene: live surfaces
    SURF_SIZES = {"board": (1024, 512), "tv": (512, 288)}

    def _surf_init(self):
        """The app server's live surfaces (docs/SCENE.md section 7): one RGBA8 texture each (premultiplied,
        row 0 = the canvas's top row), sampled by the live background pass on the board's slate / the TV screen.
        Only a character whose set has them (PLACE_BOARD / PLACE_TV in the header) gets them."""
        head = self.sh.head
        self.surf = {}
        self.surf_dirty = set()
        self.surf_bytes = 0
        for name, tag, unit in (("board", "PLACE_BOARD", 2), ("tv", "PLACE_TV", 3)):
            if f"#define {tag} " not in head:
                continue
            w, h = self.SURF_SIZES[name]
            t = self.ctx.texture((w, h), 4, dtype="f1")
            t.write(np.zeros((h, w, 4), np.uint8).tobytes())
            t.build_mipmaps()
            t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            t.repeat_x = t.repeat_y = False
            try:
                t.anisotropy = 4.0
            except Exception:
                pass
            self.surf[name] = (t, unit)
        if self.surf:
            self.set(self.p_live, "uSurfB", 2); self.set(self.p_live, "uSurfT", 3)
            self.set(self.p_live, "uSurfOn", (1.0 if "board" in self.surf else 0.0, 1.0 if "tv" in self.surf else 0.0))
            self.set(self.p_live, "uTvGain", float(self.cfg.get("tv_gain", 0.85)))

    def surface_write(self, name, x, y, w, h, rgba):
        """Upload a rect of a surface: rgba = (h, w, 4) uint8 PREMULTIPLIED (or bytes), x, y = its top-left texel in
        the canvas (row 0 = top).  Mipmaps are rebuilt once before the next frame."""
        if name not in self.surf or w <= 0 or h <= 0:
            return False
        t, _ = self.surf[name]
        data = rgba if isinstance(rgba, (bytes, bytearray, memoryview)) else np.ascontiguousarray(rgba, np.uint8).tobytes()
        t.write(data, viewport=(int(x), int(y), int(w), int(h)), alignment=1)
        self.surf_dirty.add(name)
        self.surf_bytes += len(data)
        return True

    def surface_names(self):
        return list(self.surf)

    def _surf_bind(self):
        for name in list(self.surf_dirty):
            self.surf[name][0].build_mipmaps()
        self.surf_dirty.clear()
        for t, unit in self.surf.values():
            t.use(unit)

    # ------------------------------------------------------------------ quality tiers
    def _alloc_level(self, scale):
        ctx, w, h = self.ctx, self.w, self.h
        L = dict(scale=float(scale))
        L["iw"], L["ih"] = max(int(round(w*scale)), 8), max(int(round(h*scale)), 8)
        L["tw"], L["th"] = (L["iw"]+TILE-1)//TILE, (L["ih"]+TILE-1)//TILE
        L["t_pre"] = self._tex((L["tw"], L["th"]), 4, "f4", False); L["f_pre"] = ctx.framebuffer([L["t_pre"]])
        L["t_mask"] = [self._tex((L["tw"], L["th"]), 4, "f4", False) for _ in range(2)]
        L["f_mask"] = [ctx.framebuffer([t]) for t in L["t_mask"]]
        for f in L["f_mask"]:
            f.clear(0, 0, 0, 0)
        L["t_c"] = self._tex((L["iw"], L["ih"]), 4, "f4", True)
        L["t_dd"] = [self._tex((L["iw"], L["ih"]), 4, "f4", True) for _ in range(2)]   # depth ping-pong: last frame's feeds pstart
        L["f_chars"] = [ctx.framebuffer([L["t_c"], t]) for t in L["t_dd"]]
        for f in L["f_chars"]:
            f.clear(0, 0, 0, 0)
        return L

    def _use_level(self, i, uniforms=True):
        L = self.levels[i]
        self.tier = i
        self.scale = L["scale"]
        for k in ("iw", "ih", "tw", "th", "t_pre", "f_pre", "t_mask", "f_mask", "t_c", "t_dd", "f_chars"):
            setattr(self, k, L[k])
        if uniforms:
            self._level_uniforms()

    def _level_uniforms(self):
        """Every uniform that depends on the character's internal resolution."""
        for p in (self.p_pre, self.p_char):
            self.set(p, "uRes", (self.iw, self.ih))
        self.set(self.p_res, "uResI", (self.iw, self.ih))
        for p in (self.p_mask, self.p_char, self.p_res):
            self.set(p, "uTiles", (self.tw, self.th))
        self.set(self.p_res, "uTileOut", (TILE*self.w/self.iw, TILE*self.h/self.ih))
        for p in (self.p_res, self.p_comp):
            self.set(p, "uScale", self.iw/self.w)
        if self.p_roi is not None:
            self.set(self.p_roi, "uScale", self.iw/self.w)
        self.set(self.p_pre, "uTileEps", 0.022*1080.0/self.ih)
        if self.look:       # the lowest of several tiers gives up the wide occlusion taps
            wide = 0.0 if (len(self.levels) > 1 and self.tier == 0) else 1.0
            for p in (self.p_char, self.p_roi):
                if p is not None:
                    self.set(p, "uLookWide", wide)

    def prewarm(self):
        """Draw every code path once (raised hands, root motion, the moved camera, the wide shot, worn props, each
        tier's targets): the GL driver's first-use work (it finishes compiling a program the first time a branch
        runs) then never lands in a live frame.  Measured on the K620: the first walk / spin of a session used to
        cost one 60-85 ms frame."""
        Z = np.zeros(NPARAMS, "f4")
        def pose(**kw):
            v = Z.copy()
            for k, x in kw.items():
                if k in IDX:
                    v[IDX[k]] = x
            return v
        poses = [Z, pose(hand_r_y=1.0, hand_r_x=0.2, hand_r_open=1.0, hand_l_y=0.8, hand_l_x=-0.35, mouth_open=0.8,
                         lip_open=0.8, jaw=0.7, teeth_gap=0.6, blink_l=1.0, blink_r=1.0, tongue_out=0.5),
                 pose(root_x=0.5, root_yaw=0.3, cam_x=0.4), pose(root_x=0.6, root_yaw=0.4, cam_x=0.5),
                 pose(root_pitch=-0.4, root_y=0.6, squash=-0.2), pose(cam_zoom=1.0, root_x=1.0),
                 pose(prop0_hh=1.0, prop3_s=1.0, cam_zoom=0.6, root_x=0.3), pose(prop0_hh=1.0, prop3_s=1.0, cam_zoom=0.6, root_x=0.35)]
        t0, tier0, n = time.time(), self.tier, 0
        for p in poses:
            self.render(n, p); n += 1
        for t in list(range(len(self.levels))) + [tier0]:
            if len(self.levels) > 1:
                self.set_tier(t)
                self.render(n, Z); n += 1
        self.render(n, Z)
        self.reset_history()
        self.body_prev = None
        self.init_times["prewarm_s"] = round(time.time() - t0, 2)

    @property
    def tiers(self):
        return [L["scale"] for L in self.levels]

    def set_tier(self, i):
        """Quality tier for the following frames (0 = cheapest).  Takes effect at the next frame; the temporal history
        is at output resolution and carries over, so the switch shows no pop."""
        i = max(0, min(len(self.levels) - 1, int(i)))
        self.tier_pending = i if i != self.tier else None

    def _jitter_seq(self, n):
        # Halton(2,3): even coverage of the internal pixel in a short cycle (temporal upsampling needs it)
        def hal(i, b):
            f, r = 1.0, 0.0
            while i > 0:
                f /= b; r += f*(i % b); i //= b
            return r
        return [(hal(i+1, 2), hal(i+1, 3)) for i in range(n)]

    def _set_arr(self, prog, name, typ, vals):
        """props: a uniform array (the linker may trim it to the highest index a pass reads)."""
        if name not in prog:
            return
        u = prog[name]
        a = np.asarray(vals, dtype="i4" if typ == "int" else "f4").reshape(len(vals), -1)
        u.write(np.ascontiguousarray(a[:max(1, u.array_length)]).tobytes())

    def _common_uniforms(self):
        progs = (self.p_pre, self.p_char, self.p_live, self.p_vol, self.p_comp, self.p_post, self.p_motes,
                 self.p_leaves, self.p_res, self.p_mask, self.p_pose, self.p_bgv)
        for p in progs:
            self._cam_uniforms(p)
        if self.has_props:      # props: the static per-slot tables as uniforms
            self._cam_uniforms(self.p_ppose)
            import props_defs
            self.props_progs = [p for p in (self.p_pre, self.p_char, self.p_roi, self.p_live, self.p_vol) if p is not None]
            for name, (typ, vals) in props_defs.tables(self.animal).items():
                for p in self.props_progs:
                    self._set_arr(p, name, typ, vals)
            n_props = len(props_defs.tables(self.animal).get("PR_KIND", ("int", []))[1])
            for p in self.props_progs:          # the slot loops' runtime bound (props.glsl PR_LOOP_N)
                self.set(p, "uPrN", n_props)
        # the character side runs at internal resolution (_level_uniforms, again at every tier switch)
        self.set(self.p_char, "uTilePx", float(TILE))
        self.set(self.p_char, "uPixScale", 1.0); self.set(self.p_char, "uPixOrg", (0.0, 0.0))
        self.set(self.p_res, "uRoiBox", tuple(float(v) for v in self.roi_box))
        self.set(self.p_res, "uRoiFeather", float(self.P["roi_feather"]))
        self.set(self.p_res, "uRoiC", 8); self.set(self.p_res, "uRoiD", 9)
        if self.p_roi is not None:
            pr = self.p_roi
            self._cam_uniforms(pr)          # uRes = output: the ROI maps its pixels into output px itself
            self.set(pr, "uPixScale", 1.0/self.roi); self.set(pr, "uPixOrg", (float(self.roi_box[0]), float(self.roi_box[1])))
            self.set(pr, "uRoiTiles", (self.rtw, self.rth)); self.set(pr, "uRoiOrg", (float(self.roi_box[0]), float(self.roi_box[1])))
            self.set(pr, "uRoiScale", self.roi); self.set(pr, "uScale", self.iw/self.w); self.set(pr, "uTilePx", float(TILE))
            self.set(pr, "uRoiRes", (self.rw, self.rh)); self.set(pr, "uMask", 0)
            self.set(self.p_res, "uRoiScale", self.roi); self.set(self.p_res, "uRoiRes", (self.rw, self.rh))
        self.set(self.p_vol, "uRes", (self.vw, self.vh))
        self.set(self.p_pre, "uTile", float(TILE))
        self.set(self.p_vol, "uVolGain", self.vol_gain)
        self.set(self.p_live, "uBgSunFloor", float(self.bg_sun[0])); self.set(self.p_live, "uBgSunGain", float(self.bg_sun[1]))
        for k, u in (("exposure", "uExposure"), ("bloom", "uBloom"), ("saturation", "uSaturation"),
                     ("vignette", "uVignette"), ("grain", "uGrain")):
            self.set(self.p_post, u, float(self.grade[k]))
        self.set(self.p_post, "uGain", tuple(self.grade["gain"]))
        self.set(self.p_post, "uLift", tuple(self.grade["lift"]))
        self.set(self.p_post, "uFade", 1.0)
        if self.look:       # the look upgrade
            lk = self.look_k
            self.set(self.p_post, "uClarity", float(lk["clarity"])); self.set(self.p_post, "uContrast", float(lk["contrast"]))
            self.set(self.p_post, "uGrain", float(self.grade["grain"])*float(lk["grain"]))
            for p in (self.p_char, self.p_roi):
                if p is not None:
                    self.set(p, "uLookK", (float(lk["key"]), float(lk["rim"]), float(lk["wide"]), float(lk["velvet"])))
                    self.set(p, "uLookF", (float(lk["fill"]), float(lk["gloss"]), float(lk["rim_pow"]), 0.0))
                    self.set(p, "uLookRimCol", tuple(float(v) for v in (lk["rim_col"] or self.cfg["rim"]["color"])))
            self.set(self.p_live, "uLookFloor", float(lk["floor"]) if "CHAR_ROOT" in self.cfg.get("char_defines", {}) else 0.0)
        for p in (self.p_motes, self.p_leaves):
            self.set(p, "uMotesMin", self.motes_min); self.set(p, "uMotesSize", self.motes_size)
            self.set(p, "uMoteGain", self.mote_gain)
            self.set(p, "uLeafColA", tuple(self.leaf_a)); self.set(p, "uLeafColB", tuple(self.leaf_b))
        self.set(self.p_res, "uAlpha", float(self.P["alpha"])); self.set(self.p_res, "uAlphaMiss", float(self.P["alpha_miss"]))
        self.set(self.p_res, "uGamma", float(self.P["gamma"]))
        # samplers: fixed units
        self.set(self.p_mask, "uPre", 0)
        self.set(self.p_char, "uMask", 0); self.set(self.p_char, "uPrevD", 1); self.set(self.p_char, "uPrevMask", 2)
        self.set(self.p_res, "uReproj", 1.0 if self.P["reproj"] else 0.0)
        for n, u in (("uCur", 0), ("uCurD", 1), ("uHist", 2), ("uHistD", 3), ("uMask", 4), ("uMaskPrev", 5),
                     ("uPoseC", 6), ("uPoseP", 7)):
            self.set(self.p_res, n, u)
        for n, u in (("uLive", 0), ("uVol", 1), ("uChar", 2), ("uCharD", 3), ("uMask", 4)):
            self.set(self.p_comp, n, u)
        self.set(self.p_live, "uBgA", 0); self.set(self.p_live, "uBgS", 1)
        self.set(self.p_vol, "uDepth", 0)
        self.set(self.p_post, "uHdr", 0)
        for n, u in (("uBgA", 0), ("uBgS", 1), ("uBgA0", 2), ("uBgS0", 3), ("uPanA", 4), ("uPanS", 5)):
            self.set(self.p_bgv, n, u)
        self._level_uniforms()

    # ------------------------------------------------------------------ body: root motion + camera
    def _body_init(self, spp):
        """The rest camera, the zoom range, the extra background layers (baked once, cached like the background) and
        the head's rest pose (gaze_free's reference)."""
        b = self.body
        self.cam0 = (self.cam_pos.copy(), self.u.copy(), self.v.copy(), self.ww.copy(), float(self.fl))
        self.target0 = np.array(self.cfg["camera"]["target"], float)
        d0 = float(np.dot(b["C"] - self.cam_pos, self.ww))
        self.cam_kw = max(1.0, CAM_WIDE_H*b["H"]/(2.0*d0/self.fl))       # tan factor of the cam_zoom 1 view
        self.pan_fl = self.fl/(self.cam_kw*PAN_K)
        self.pan_w, self.pan_h = int(round(self.w*PAN_RS)), int(round(self.h*PAN_RS))
        key = self.sh_bg.key()
        px = self.h/1080.0
        # only characters that opt in (cfg "motion_layers": the muse) bake the extra layers and get the live shadow +
        # the camera; the 11 others keep today's init exactly (root motion still transforms them, shadow baked)
        self.motion_layers = bool(self.cfg.get("motion_layers", False))
        if self.motion_layers:
            self.t_bgA0, self.t_bgS0 = self._bake_layer(f"bgnc_{self.w}x{self.h}_{key}", self.w, self.h, self.fl, spp, px)
            pkey = f"bgpan_{self.pan_w}x{self.pan_h}_{key}_{self.pan_fl:.4f}"
            self.t_panA, self.t_panS = self._bake_layer(pkey, self.pan_w, self.pan_h, self.pan_fl, PAN_SPP,
                                                        px*PAN_RS*self.pan_fl/self.fl)
        else:
            self.t_bgA0, self.t_bgS0, self.t_panA, self.t_panS = self.t_bgA, self.t_bgS, self.t_bgA, self.t_bgS
        # the head's rest pose -> the camera in head space at rest
        ctx, mgl = self.ctx, self.mgl
        # (into a scratch target: the TAA's pose history t_pose[*] must start exactly as before, or the first frames'
        # head reprojection differs and the 11 characters are not bit-identical to main)
        self.set(self.p_pose, "uP", (0.0,)*NPARAMS); self.set(self.p_pose, "uTime", 0.0)
        t_tmp = self._tex((4, 1), 4, "f4", False); f_tmp = ctx.framebuffer([t_tmp])
        f_tmp.use(); ctx.viewport = (0, 0, 4, 1)
        self.vaos["pose"].render(mgl.TRIANGLE_STRIP)
        pz = np.frombuffer(t_tmp.read(), dtype="f4").reshape(4, 4)
        f_tmp.release(); t_tmp.release()
        hr = np.stack([pz[0, :3], pz[1, :3], pz[2, :3]], axis=1)       # columns
        self.cam_h0 = hr @ (self.cam_pos - pz[3, :3])
        self.scene_progs = [p for p in (self.p_pre, self.p_char, self.p_roi, self.p_live, self.p_vol, self.p_pose,
                                           self.p_ppose) if p]      # props: p_ppose runs charInit too
        for p in self.scene_progs:
            self.set(p, "uCamH0", tuple(float(x) for x in self.cam_h0))
            self.set(p, "uRootOn", 0.0)
        self.set(self.p_bgv, "uCam0U", self.u); self.set(self.p_bgv, "uCam0V", self.v); self.set(self.p_bgv, "uCam0W", self.ww)
        self.set(self.p_bgv, "uCam0Fl", float(self.fl)); self.set(self.p_bgv, "uPanFl", float(self.pan_fl))
        self.set(self.p_bgv, "uPanRes", (float(self.pan_w), float(self.pan_h)))
        self.body_prev = None
        self.body_state = {}

    def _bake_layer(self, name, w, h, fl, spp, px):
        """PASS_BG without the character (BG_NO_CHAR) from the rest camera's position and orientation, lens fl, at w x h;
        the depth of field baked with `px` pixels per 1080p pixel (the same angular blur as the main bake)."""
        d = os.path.join(render.OUT_ROOT, self.animal)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".npy")
        if os.path.exists(path):
            arr = np.load(path, mmap_mode="r")
        else:
            ctx, mgl = self.ctx, self.mgl
            t0 = time.time()
            p_bg = ctx.program(vertex_shader=VS_FULL, fragment_shader=self.sh_bg.frag("scene_main.glsl", "PASS_BG", "BG_NO_CHAR"))
            self._cam_uniforms(p_bg)
            self.set(p_bg, "uRes", (w, h)); self.set(p_bg, "uFl", float(fl))
            self.set(p_bg, "uTime", 0.0); self.set(p_bg, "uFrame", 0); self.set(p_bg, "uP", tuple([0.0]*NPARAMS))
            vb = ctx.vertex_array(p_bg, [(self.quad, "2f", "in_pos")])
            ta, ts = self._tex((w, h), 4, "f4", True), self._tex((w, h), 4, "f4", True)
            fb = ctx.framebuffer([ta, ts])
            fb.use(); ctx.viewport = (0, 0, w, h); fb.clear(0, 0, 0, 0)
            ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE); ctx.blend_equation = mgl.FUNC_ADD
            strips = max(8, h//34); sh = (h + strips - 1)//strips
            for s in range(spp):
                self.set(p_bg, "uJitter", r2(s))
                for i in range(strips):
                    ctx.scissor = (0, i*sh, w, min(sh, h - i*sh))
                    vb.render(mgl.TRIANGLE_STRIP)
                    ctx.finish()
            ctx.scissor = None
            ctx.disable(mgl.BLEND)
            a = np.frombuffer(ta.read(), dtype="f4").reshape(h, w, 4).copy()/spp
            s_ = np.frombuffer(ts.read(), dtype="f4").reshape(h, w, 4).copy()/spp
            for o in (vb, fb, ta, ts, p_bg):
                o.release()
            arr = self._lens(np.concatenate([a, s_], axis=2), px)
            tmp = path + f".tmp{os.getpid()}.npy"
            np.save(tmp, arr); os.replace(tmp, path)
            if self.log:
                print(f"body layer {name} baked in {time.time()-t0:.1f}s", flush=True)
        tA, tS = self._tex((w, h), 4, "f4", True), self._tex((w, h), 4, "f4", True)
        tA.write(np.array(arr[:, :, :4], dtype="f4").tobytes()); tS.write(np.array(arr[:, :, 4:], dtype="f4").tobytes())
        return tA, tS

    def _lens(self, img, px):
        """render.Engine.bg_lens with an explicit pixel scale (the panorama's pixels are not 1080p-relative)."""
        from scipy.signal import fftconvolve
        h, w = img.shape[:2]
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
                k = np.clip(1.0 - (xx*xx + yy*yy)/(r*1.15)**2, 0, 1).astype("f4")**1.5
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

    def camera_for(self, zoom, cx, cy):
        """(pos, u, v, w, fl) for cam_zoom / cam_x / cam_y: the rest camera turned and zoomed about its own position
        (no dolly, so the baked set stays exact); the view is kept inside the baked panorama."""
        b = self.body
        pos, u0, v0, w0, fl0 = self.cam0
        zoom = float(np.clip(zoom, -1.0, 1.0))
        kz = self.cam_kw**zoom if zoom >= 0.0 else CAM_IN**zoom       # scene: cam_zoom < 0 pushes in (2x at -1)
        fl = fl0/kz
        f = (1.0 - 1.0/kz)/max(1.0 - 1.0/self.cam_kw, 1e-6) if zoom > 0.0 else 0.0   # 0..1: how far out we are
        aim1 = np.array([b["C"][0], b["origin"][1] + CAM_WIDE_AIM*b["H"], b["C"][2]])
        tgt = self.target0 + (aim1 - self.target0)*f + b["H"]*np.array([cx, cy, 0.0])
        # keep the view inside the panorama: the aim's tangent offsets + the half view <= the panorama's half extent
        d = tgt - pos
        ax, ay = np.dot(d, u0)/np.dot(d, w0), np.dot(d, v0)/np.dot(d, w0)
        asp = self.w/self.h
        hy, hx = 1.0/fl, asp/fl
        py, pxm = 0.97/self.pan_fl, 0.97*asp/self.pan_fl
        ax = float(np.clip(ax, -(pxm - hx*1.08), pxm - hx*1.08)); ay = float(np.clip(ay, -(py - hy*1.08), py - hy*1.08))
        tgt = pos + np.linalg.norm(d)*(w0 + ax*u0 + ay*v0)/np.sqrt(1.0 + ax*ax + ay*ay)
        u, v, w = render.camera_basis(pos, tgt, self.cfg["camera"].get("roll", 0.0))
        return pos, u, v, w, fl

    def _set_cam(self, p, cam):
        pos, u, v, w, fl = cam
        self.set(p, "uCamPos", tuple(float(x) for x in pos)); self.set(p, "uCamU", tuple(float(x) for x in u))
        self.set(p, "uCamV", tuple(float(x) for x in v)); self.set(p, "uCamW", tuple(float(x) for x in w))
        self.set(p, "uFl", float(fl))

    def _body_frame(self, P):
        """Per frame: the root transform and the camera from the params -> uniforms.  All root/camera params 0 sets
        nothing but uRootOn = 0 (and restores the rest camera once after a move): today's frame exactly."""
        b = self.body
        r = [P[i] for i in ROOT_COLS]; c = [P[i] for i in CAM_COLS]
        root_on = self.root_ok and any(abs(x) > 1e-7 for x in r)     # only a CHAR_ROOT character (the muse) has the root
        cam_on = self.motion_layers and any(abs(x) > 1e-7 for x in c)
        cam = self.camera_for(*c) if cam_on else self.cam0
        I3 = np.eye(3)
        if root_on:
            x, y, z, yaw, pitch, roll, sq = r
            R = _rot(yaw, pitch, roll)
            sy = float(np.exp(-SQUASH_K*np.clip(sq, -1.0, 1.0))); sx = 1.0/np.sqrt(sy)
            A = R @ np.diag([sx, sy, sx]); Ai = np.diag([1/sx, 1/sy, 1/sx]) @ R.T
            o = b["H"]*np.array([x, y, z]) + b["C"] + R @ (b["B"] - b["C"]) - A @ b["B"]
            lip, smax = min(sx, sy), max(sx, sy)
            th = float(np.arccos(np.clip((np.trace(R) - 1.0)*0.5, -1.0, 1.0)))/(2*np.pi)
            act = float(np.clip(max(abs(x), abs(y), abs(z))/0.04 + th/0.015 + abs(sq)/0.04, 0.0, 1.0))
        else:
            R, A, Ai, o, lip, smax, act = I3, I3, I3, np.zeros(3), 1.0, 1.0, 0.0
        live = max(act, float(np.clip(max(abs(x) for x in c)/0.02, 0.0, 1.0))) if cam_on else act
        if not self.motion_layers:
            live = 0.0
        bc = A @ b["bc"] + o; br = b["br"]*smax
        dist = float(np.linalg.norm(bc - cam[0]))
        tn = min(float(self.quality["CHAR_T_NEAR"]), max(0.2, dist - br))
        tf = max(float(self.quality["CHAR_T_FAR"]), dist + br)
        prev = self.body_prev
        cam_changed = prev is not None and (cam_on or prev["cam_on"]) and any(
            not np.array_equal(a_, b_) for a_, b_ in zip(cam, prev["cam"]))
        for p in self.scene_progs:
            self.set(p, "uRootOn", 1.0 if root_on else 0.0)
            if root_on:
                self.set(p, "uRootA", _m3(A)); self.set(p, "uRootAi", _m3(Ai)); self.set(p, "uRootRm", _m3(R))
                self.set(p, "uRootO", tuple(float(v) for v in o)); self.set(p, "uRootLip", float(lip))
                self.set(p, "uRootTNear", tn); self.set(p, "uRootTFar", tf)
        if cam_on or (prev is not None and prev["cam_on"]):
            for p in (self.p_pre, self.p_char, self.p_roi, self.p_live, self.p_vol, self.p_pose, self.p_res, self.p_comp,
                      self.p_motes, self.p_leaves, self.p_bgv, self.p_ppose):
                if p is not None:
                    self._set_cam(p, cam)
            # autofocus on the character, the same angular blur at every zoom
            fc = self.focus + (dist - float(np.linalg.norm(b["bc"] - self.cam0[0]))) if (cam_on or root_on) else self.focus
            self.set(self.p_comp, "uFocus", float(max(fc, 0.3)))
            self.set(self.p_comp, "uDofK", self.dofk*self.h/1080.0*cam[4]/self.cam0[4])
        elif root_on or (prev is not None and prev["root_on"]):
            fc = self.focus + (dist - float(np.linalg.norm(b["bc"] - self.cam0[0]))) if root_on else self.focus
            self.set(self.p_comp, "uFocus", float(max(fc, 0.3)))
        self.set(self.p_live, "uRootLive", live)
        self.set(self.p_live, "uRootBound", (float(bc[0]), float(bc[1]), float(bc[2]), float(br)))
        self.set(self.p_bgv, "uRootLive", live); self.set(self.p_bgv, "uCamOn", 1.0 if cam_on else 0.0)
        # TAA: last frame's camera and root
        root_changed = prev is not None and (root_on or prev["root_on"]) and not (
            np.array_equal(A, prev["A"]) and np.array_equal(o, prev["o"]))
        moved = bool(cam_changed or root_changed)
        pr = self.p_res
        self.set(pr, "uMoveOn", 1.0 if moved else 0.0)
        if moved:
            pc = prev["cam"]
            self._set_cam(pr, cam)
            self.set(pr, "uPCamPos", tuple(float(x) for x in pc[0])); self.set(pr, "uPCamU", tuple(float(x) for x in pc[1]))
            self.set(pr, "uPCamV", tuple(float(x) for x in pc[2])); self.set(pr, "uPCamW", tuple(float(x) for x in pc[3]))
            self.set(pr, "uPFl", float(pc[4]))
            self.set(pr, "uRootMv", 1.0 if root_changed else 0.0)
            self.set(pr, "uRootAiC", _m3(Ai)); self.set(pr, "uRootOC", tuple(float(x) for x in o))
            self.set(pr, "uRootAP", _m3(prev["A"])); self.set(pr, "uRootOP", tuple(float(x) for x in prev["o"]))
            # fast motion: lean on this frame (the variance clip alone lets a 30-degree-per-frame spin smear)
            dth = float(np.arccos(np.clip((np.trace(prev["R"].T @ R) - 1.0)*0.5, -1.0, 1.0)))/(2*np.pi)
            dT = float(np.linalg.norm(o - prev["o"]))/b["H"] if root_changed else 0.0
            self.set(pr, "uMotionW", float(np.clip(dth/0.06 + dT/0.08, 0.0, 1.0))*MOTION_W)
        else:
            self.set(pr, "uMotionW", 0.0)
        self.body_prev = dict(cam=cam, cam_on=cam_on, root_on=root_on, A=A, o=o, R=R)
        self.body_state = dict(root_on=root_on, cam_on=cam_on, live=live, moved=moved)
        return self.body_state

    def reset_history(self):
        self.frames_in_hist = 0

    # ------------------------------------------------------------------ one frame
    def _draw(self, frame, params, prof=None):
        ctx, mgl = self.ctx, self.mgl
        t = frame/float(FPS)
        P = tuple(float(x) for x in params)[:NPARAMS]
        P += (0.0,)*(NPARAMS - len(P))
        w, h = self.w, self.h
        k = self.nsub & 1
        self._tier_switched = False
        if self.tier_pending is not None:          # the tier changes between frames
            self._use_level(self.tier_pending)
            self.tier_pending = None
            self._tier_switched = True
        bs = self._body_frame(P)          # body: root transform + camera uniforms
        use_bv = bs["cam_on"] or bs["live"] > 0.0
        qs = []
        def mark(name):
            if prof is not None:
                q = ctx.query(time=True); q.__enter__(); qs.append((name, q))
        def done():
            if prof is not None and qs:
                qs[-1][1].__exit__(None, None, None)
        # 1. prepass + GPU min filter -> tile mask (R1 march, R2 zero ring)
        if self.has_props:      # props: the props' poses for this frame -> uniforms of every frame pass
            self.set(self.p_ppose, "uP", P); self.set(self.p_ppose, "uTime", t)
            self.f_ppose.use(); ctx.viewport = (0, 0, PROPS_SLOTS, 4)
            self.vaos["ppose"].render(mgl.TRIANGLE_STRIP)
            pp = np.frombuffer(self.t_ppose.read(), dtype="f4").reshape(4, PROPS_SLOTS, 4)
            pc = pp[0, :, :3]; par = np.stack([pp[0, :, 3], pp[1, :, 3]], 1)
            pm = np.stack([pp[1, :, :3], pp[2, :, :3], pp[3, :, :3]], 1)        # (slot, column, row)
            for p in self.props_progs:
                self._set_arr(p, "uPrC", "vec3", pc); self._set_arr(p, "uPrM", "mat3", pm)
                self._set_arr(p, "uPrAR", "vec2", par)
        mark("prepass")
        self.set(self.p_pre, "uP", P); self.set(self.p_pre, "uTime", t)
        self.f_pre.use(); ctx.viewport = (0, 0, self.tw, self.th)
        self.vaos["pre"].render(mgl.TRIANGLE_STRIP)
        self.f_mask[k].use(); self.t_pre.use(0)
        if self.no_char:
            self.f_mask[k].clear(0, 0, 0, 0)
        else:
            self.vaos["mask"].render(mgl.TRIANGLE_STRIP)
            if self._tier_switched:                # this tier's "last frame" mask = this frame's: the history stays valid
                self.f_mask[1-k].use()
                self.vaos["mask"].render(mgl.TRIANGLE_STRIP)
        done()
        # 2. character: one point per R2 tile, culled in the vertex shader
        mark("char")
        self.t_d = self.t_dd[k]
        self.f_chars[k].use(); ctx.viewport = (0, 0, self.iw, self.ih)
        self.t_mask[k].use(0); self.t_dd[1-k].use(1); self.t_mask[1-k].use(2)
        ps = float(self.P["pstart"]) if (self.frames_in_hist > 0 and self.spp == 1 and not self._tier_switched) else 0.0
        self.set(self.p_char, "uPStart", ps)
        self.set(self.p_char, "uP", P); self.set(self.p_char, "uTime", t); self.set(self.p_char, "uFrame", frame)
        ntiles = self.tw*self.th
        if self.taa:
            jit = self.jit[frame % len(self.jit)]
            self.set(self.p_char, "uJitter", jit)
            self.vao_char.render(mgl.POINTS, vertices=ntiles)
            if self.p_roi is not None and not (bs["root_on"] or bs["cam_on"]):      # the mouth, marched finer (its box is fixed)
                self.f_roi.use(); ctx.viewport = (0, 0, self.rw, self.rh); self.f_roi.clear(0, 0, 0, 0)
                self.set(self.p_roi, "uP", P); self.set(self.p_roi, "uTime", t); self.set(self.p_roi, "uFrame", frame)
                self.set(self.p_roi, "uJitter", jit)
                self.vao_roi.render(mgl.POINTS, vertices=self.rtw*self.rth)
        else:
            for s in range(self.spp):
                jit = (0.5, 0.5) if self.spp == 1 else r2(s + 1 + (frame % 4)*self.spp)
                self.set(self.p_char, "uJitter", jit)
                if s == 1:
                    ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE)
                self.vao_char.render(mgl.POINTS, vertices=ntiles)
            ctx.disable(mgl.BLEND)
            jit = (0.5, 0.5)
        done()
        # 3. temporal resolve into the output-res history
        if self.taa:
            mark("resolve")
            if self.P["reproj"]:
                self.set(self.p_pose, "uP", P); self.set(self.p_pose, "uTime", t)
                self.f_pose[k].use(); ctx.viewport = (0, 0, 4, 1)
                self.vaos["pose"].render(mgl.TRIANGLE_STRIP)
                self.t_pose[k].use(6); self.t_pose[1-k].use(7)
            f = self.f_hist[k]; f.use(); ctx.viewport = (0, 0, w, h)
            self.t_c.use(0); self.t_d.use(1); self.t_hc[1-k].use(2); self.t_hd[1-k].use(3)
            self.t_mask[k].use(4); self.t_mask[1-k].use(5)
            if self.p_roi is not None:
                self.t_rc.use(8); self.t_rd.use(9)
                self.set(self.p_res, "uRoiBox", (0.0, 0.0, 0.0, 0.0) if (bs["root_on"] or bs["cam_on"])
                         else tuple(float(v) for v in self.roi_box))
            self.set(self.p_res, "uJit", jit); self.set(self.p_res, "uInvSpp", 1.0)
            self.set(self.p_res, "uReset", 1.0 if self.frames_in_hist == 0 else 0.0)
            self.vao_res.render(mgl.TRIANGLE_STRIP, vertices=4, instances=ntiles)
            self.frames_in_hist += 1
            ch_c, ch_d, inv = self.t_hc[k], self.t_hd[k], 1.0
            done()
        else:
            ch_c, ch_d, inv = self.t_c, self.t_d, 1.0/self.spp
        # 4. living background + sprites behind
        mark("live_bg")
        if use_bv:          # body: the cached layers for this camera / without the character's shadow
            self.f_bv.use(); ctx.viewport = (0, 0, w, h)
            for i, t_ in enumerate((self.t_bgA, self.t_bgS, self.t_bgA0, self.t_bgS0, self.t_panA, self.t_panS)):
                t_.use(i)
            self.vaos["bgv"].render(mgl.TRIANGLE_STRIP)
        bgA, bgS = (self.t_bva, self.t_bvs) if use_bv else (self.t_bgA, self.t_bgS)
        self.f_live.use(); ctx.viewport = (0, 0, w, h)
        bgA.use(0); bgS.use(1)
        if self.surf:              # scene
            self._surf_bind()
        self.set(self.p_live, "uTime", t); self.set(self.p_live, "uP", P)   # a set may draw what the columns move (river)
        self.vaos["live"].render(mgl.TRIANGLE_STRIP)
        self.sprites(t, frame, 0)
        done()
        # 5. god rays
        mark("godrays")
        self.f_vol.use(); ctx.viewport = (0, 0, self.vw, self.vh)
        bgA.use(0)
        self.set(self.p_vol, "uTime", t); self.set(self.p_vol, "uFrame", frame)
        self.vaos["vol"].render(mgl.TRIANGLE_STRIP)
        done()
        # 6. composite with the character DoF, front sprites
        mark("comp_dof")
        self.f_hdr.use(); ctx.viewport = (0, 0, w, h)
        self.t_live.use(0); self.t_vol.use(1); ch_c.use(2); ch_d.use(3); self.t_mask[k].use(4)
        self.set(self.p_comp, "uInvSpp", inv); self.set(self.p_comp, "uFrame", frame)
        self.set(self.p_comp, "uP", P); self.set(self.p_comp, "uTime", t)
        self.vaos["comp"].render(mgl.TRIANGLE_STRIP)
        self.sprites(t, frame, 1)
        done()
        # 7. bloom chain + grade, readback into a PBO (asynchronous until fetched)
        mark("post")
        self.t_hdr.build_mipmaps()
        self.f_out.use(); ctx.viewport = (0, 0, w, h)
        self.t_hdr.use(0)
        self.set(self.p_post, "uFrame", frame)
        self.vaos["post"].render(mgl.TRIANGLE_STRIP)
        done()
        mark("readback")
        self.f_out.read_into(self.pbo[k], components=4, alignment=1)
        done()
        self.pending.append((k, frame))
        self.nsub += 1
        if prof is not None:
            for name, q in qs:
                prof[name] = prof.get(name, 0.0) + q.elapsed*1e-9

    def _fetch(self):
        k, frame = self.pending.pop(0)
        data = self.pbo[k].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.h, self.w, 4)[::-1, :, :3]
        return frame, img

    def render(self, frame, params, profile=None):
        """Synchronous: draws `frame` and returns its image."""
        while self.pending:
            self._fetch()
        self._draw(frame, params, profile)
        return self._fetch()[1]

    def submit(self, frame, params, profile=None):
        """Pipelined: queue `frame`; returns the PREVIOUS submitted frame's (index, image) or None.  The GPU draws
        frame i while the CPU copies frame i-1 out — one frame of extra latency, higher throughput."""
        self._draw(frame, params, profile)
        if len(self.pending) > 1:
            return self._fetch()
        return None

    def flush(self):
        out = []
        while self.pending:
            out.append(self._fetch())
        return out

    def close(self):
        try:
            self.ctx.release()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="fox")
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--preset", default="taa")
    ap.add_argument("--params")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    eng = RTEngine(a.animal, a.w, a.h, a.preset, log=True)
    arr = np.load(a.params) if a.params else np.zeros((a.frames, NPARAMS), "f4")
    ts = []
    for i in range(min(a.frames, len(arr))):
        t0 = time.perf_counter(); img = eng.render(i, arr[i]); ts.append(time.perf_counter() - t0)
        if a.out_dir:
            from PIL import Image
            os.makedirs(a.out_dir, exist_ok=True)
            Image.fromarray(img).save(os.path.join(a.out_dir, f"{i:06d}.png"), compress_level=1)
    ts = np.array(ts[3:])*1000
    print(f"{a.animal} {a.w}x{a.h} {a.preset}: mean {ts.mean():.1f} ms  p95 {np.percentile(ts, 95):.1f} ms")


if __name__ == "__main__":
    main()
