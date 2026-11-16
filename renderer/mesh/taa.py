"""shade-taa: temporal antialiasing of the MESH layer (GATED: the character's mesh config "pbr": {"taa": {...}};
absent / false = no jitter, no velocity attachment, no resolve: MeshCharacter takes exactly the earlier code path).

What it does (character.py calls it; taa.glsl is the resolve):
  jitter    every raster pass that projects with mesh.glsl camClip (forward, hair strands, hair cap) gets this frame's
            Halton(2,3) subpixel offset (uJit, MESH_TAA_JIT in the shader head)
  velocity  the forward pass writes one more MSAA attachment (velocity px, ray distance, 1) from the previous frame's
            posed vertex buffer (a copy of vb_posed taken before the skinning) + the previous camera.  Hair strands and
            blended overlays never write it (colour-masked); the resolve reprojects strands by the head bone's rigid motion
  resolve   after the SSS (+AO): history (Catmull-Rom) reprojected by the 3x3 closest-depth velocity, YCoCg+alpha
            variance clipping, velocity / depth disocclusion, luma-weighted blend -> the composite reads the result
  reset     first frame, camera cuts (pos / direction / lens jumps), size changes, MeshEngine.reset_history / tier changes

Config (all optional):  "taa": {"blend": 0.1, "gamma": 1.0, "jitter": 0.35, "jitter_n": 8, "vel_reject": 2.0,
    "hair_eps": 0.02, "depth_reject": 0.0, "luma_k": 0.0, "catmull": true, "unjitter": true, "motion_blend": 0.3,
    "max_blend": 0.5, "cut_pos": 0.25, "cut_deg": 8.0, "cut_fl": 0.15}

 shade2-taa (both optional, absent = the shade-taa resolve above, unchanged):
  "classes": {"eye": {"react": 0.6, "gamma": 0.75, "sharpen": 1.0}, "hair": {...}, ...}
            per material class: the forward pass (and the hair strands / cap) write (react, sharpen, gamma/4) into one
            more MSAA attachment (location 5); the resolve reads it per pixel: react = the floor of the new frame's
            weight (1 = no history), gamma = a multiplier on the clip box sigma, sharpen = a multiplier on "sharpen".
            classes: skin, eye (eye, caruncle; the cornea / tearline / eye_occlusion overlays keep the eye's or skin's
            class), teeth (teeth, gums, tongue, mouth_interior), lashes (mesh lashes + lash strands), brows, hair (scalp,
            fuzz, hair cap), knit (cardigan + rib), cloth (blouse, trousers, buttons, shoes), default
  "sharpen": 0.3   after the resolve, a CAS-like luma-adaptive sharpen of the resolved layer (never fed back into the
            history), clamped to the 3x3 min/max (no halos); strength x the pixel's class "sharpen"
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(blend=0.1, gamma=1.0, blend_static=0.1, gamma_static=1.0, still_px=0.5, react_blend=0.5, react_gamma=0.75, jitter=0.35, jitter_n=8, vel_reject=2.0, hair_eps=0.02, depth_reject=0.0,
                luma_k=0.0, catmull=True, unjitter=True, motion_blend=0.3, max_blend=0.5,
                cut_pos=0.25, cut_deg=8.0, cut_fl=0.15)
TGT_NAMES = ("rb_v", "f_ms_v", "t_v", "f_v", "rb_pad", "rb_r", "f_ms_r", "t_r", "f_r")      # quality.py caches these with the forward targets


CLASS_DEFAULT = dict(react=0.0, gamma=1.0, sharpen=1.0)
CLASS_OF = {"eye": "eye", "cornea": "eye", "tearline": "eye", "eye_occlusion": "eye", "teeth": "teeth", "gums": "teeth",
            "tongue": "teeth", "mouth_interior": "teeth", "lashes": "lashes", "skin": "skin", "cloth": "cloth",
            "buttons": "cloth", "shoes": "cloth", "hair": "hair"}


def taa_cfg(pbr):
    """the taa sub-block of the pbr block, or None (= off)."""
    if not pbr or os.environ.get("JUNO_TAA") == "0":
        return None
    t = pbr.get("taa")
    if t is None or t is False:
        return None
    d = dict(DEFAULTS)
    if isinstance(t, dict):
        d.update(t)
    return d


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f*(i % b)
        i //= b
    return r


class MeshTAA:
    def __init__(self, mc, cfg):
        self.mc, self.ctx, self.mgl = mc, mc.ctx, mc.mgl
        self.k = dict(cfg)
        n = max(1, int(self.k["jitter_n"]))
        s = float(self.k["jitter"])
        self.seq = [((halton(i + 1, 2) - 0.5)*s, (halton(i + 1, 3) - 0.5)*s) for i in range(n)]
        src = open(os.path.join(HERE, "taa.glsl")).read()
        self.prog = self.ctx.program(vertex_shader=mc.head + "#define VS_TAA\n" + src,
                                     fragment_shader=mc.head + "#define FS_TAA\n" + src)
        self.vao = self.ctx.vertex_array(self.prog, [(mc.quad, "2f", "in_pos")])
        for nm, u in (("uCur", 0), ("uCurD", 1), ("uVel", 2), ("uHist", 3), ("uHistA", 4)):
            if nm in self.prog:
                self.prog[nm].value = u
        bi = getattr(mc.A, "bone_index", None) or {b["name"]: i for i, b in enumerate(mc.A.bones)}
        self.head_bone = bi.get("head")
        self.size = None
        self.n = 0
        self.jit = (0.0, 0.0)
        self.prev_cam = None
        self.cur_cam = None
        self.prev_head = None
        self.need_reset = True
        self.copy_prev = True          # pose(): vb_prev := vb_posed after the skinning (no motion on a reset frame)
        self.out = None
        self.stats = {}
        # shade2-taa (gated): per material class weights + the sharpen pass
        self.classes = dict(self.k.get("classes") or {})
        self.cls_on = bool(self.classes)
        self.sharp = float(self.k.get("sharpen") or 0.0)
        self.p_sh = None
        if self.sharp > 0.0:
            self.p_sh = self.ctx.program(vertex_shader=mc.head + "#define VS_TAA\n" + src,
                                         fragment_shader=mc.head + "#define FS_SHARP\n" + src)
            self.vao_sh = self.ctx.vertex_array(self.p_sh, [(mc.quad, "2f", "in_pos")])
            for nm, u in (("uIn", 0), ("uCls", 5)):
                if nm in self.p_sh:
                    self.p_sh[nm].value = u
        if "uCls" in self.prog:
            self.prog["uCls"].value = 5
        self.t_sh = None
        self._cv = {}

    def mat_class(self, mtype, sub_name=""):
        """material type (+ submesh name) -> the class key of "classes"."""
        if mtype == "cloth" and ("cardigan" in sub_name or "knit" in sub_name):
            return "knit"
        if mtype == "gums" and "caruncle" in sub_name:
            return "eye"
        return CLASS_OF.get(mtype, "default")

    def cls_vec(self, name):
        """(react, sharpen, gamma/4, 1) of a class (fallbacks: knit -> cloth, brows -> hair, -> default)."""
        if name in self._cv:
            return self._cv[name]
        c = dict(CLASS_DEFAULT)
        c.update(self.classes.get("default") or {})
        fb = {"knit": "cloth", "brows": "hair"}.get(name)
        if fb and fb in self.classes:
            c.update(self.classes[fb] or {})
        c.update(self.classes.get(name) or {})
        v = (min(max(float(c["react"]), 0.0), 1.0), min(max(float(c["sharpen"]), 0.0), 1.0),
             min(max(float(c["gamma"])/4.0, 0.0), 1.0), 1.0)
        self._cv[name] = v
        return v

    def hair_masks(self):
        """the strands write colour / depth / SSS (+ the class): never the velocity nor the skin-albedo attachment."""
        from character import _gl_color_mask
        if self.mc.pbr_split:
            _gl_color_mask(3, False)
        _gl_color_mask(4, False)
        if self.cls_on:
            _gl_color_mask(5, True)

    # ------------------------------------------------------------------ targets
    def ms_attachment(self, w, h, samples, n_extra):
        """the velocity MSAA attachment (+ a pad at location 2 when the SSS stage adds none) for _alloc_targets."""
        ctx, mgl, mc = self.ctx, self.mgl, self.mc
        mc.rb_v = ctx.renderbuffer((w, h), 4, samples=samples, dtype="f2")
        mc.f_ms_v = ctx.framebuffer([mc.rb_v])
        mc.t_v = ctx.texture((w, h), 4, dtype="f2")
        mc.t_v.filter = (mgl.NEAREST, mgl.NEAREST)
        mc.f_v = ctx.framebuffer([mc.t_v])
        # FS_FWD writes fragS at 2 (and fragA at 3 with the albedo split): pads so the velocity is always location 4
        mc.rb_pad = [ctx.renderbuffer((w, h), 4, samples=samples, dtype="f2") for _ in range(max(0, 2 - n_extra))]
        mc.rb_r = mc.f_ms_r = mc.t_r = mc.f_r = None
        if self.cls_on:                                  # shade2-taa: the class attachment at location 5
            mc.rb_r = ctx.renderbuffer((w, h), 4, samples=samples, dtype="f1")
            mc.f_ms_r = ctx.framebuffer([mc.rb_r])
            mc.t_r = ctx.texture((w, h), 4, dtype="f1")
            mc.t_r.filter = (mgl.NEAREST, mgl.NEAREST)
            mc.f_r = ctx.framebuffer([mc.t_r])
            return list(mc.rb_pad) + [mc.rb_v, mc.rb_r]
        return list(mc.rb_pad) + [mc.rb_v]

    def _hist(self, w, h):
        if self.size == (w, h):
            return
        ctx, mgl = self.ctx, self.mgl
        if self.size is not None:
            for f in self.f_h:
                f.release()
            for t in self.t_hc + self.t_ha:
                t.release()
        self.t_hc = [ctx.texture((w, h), 4, dtype="f2") for _ in range(2)]
        self.t_ha = [ctx.texture((w, h), 4, dtype="f2") for _ in range(2)]
        if self.p_sh is not None:                        # shade2-taa: the sharpened output (not the history)
            if self.t_sh is not None:
                self.f_sh.release(); self.t_sh.release()
            self.t_sh = ctx.texture((w, h), 4, dtype="f2")
            self.t_sh.filter = (mgl.LINEAR, mgl.LINEAR); self.t_sh.repeat_x = self.t_sh.repeat_y = False
            self.f_sh = ctx.framebuffer([self.t_sh])
        for t in self.t_hc + self.t_ha:
            t.filter = (mgl.LINEAR, mgl.LINEAR); t.repeat_x = t.repeat_y = False
        self.f_h = [ctx.framebuffer([self.t_hc[i], self.t_ha[i]]) for i in range(2)]
        self.size = (w, h)
        self.need_reset = True

    def reset(self):
        self.need_reset = True
        self.copy_prev = True

    # ------------------------------------------------------------------ per frame
    def _cut(self, a, b):
        if a is None or b is None:
            return True
        k = self.k
        pa, pb = np.asarray(a[0], float), np.asarray(b[0], float)
        if np.linalg.norm(pa - pb) > float(k["cut_pos"]):
            return True
        wa, wb = np.asarray(a[3], float), np.asarray(b[3], float)
        c = float(np.dot(wa, wb)/max(np.linalg.norm(wa)*np.linalg.norm(wb), 1e-9))
        if np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) > float(k["cut_deg"]):
            return True
        fa, fb = float(a[4]), float(b[4])
        return abs(fa - fb) > float(k["cut_fl"])*max(abs(fb), 1e-6)

    def begin(self, rig, cam):
        """before the forward pass: this frame's jitter, the previous camera / head (reset on a cut)."""
        mc = self.mc
        self._hist(mc.iw, mc.ih)
        cam = tuple(np.asarray(x, float).copy() if np.ndim(x) else float(x) for x in cam)
        if self._cut(self.prev_cam, cam):
            self.need_reset = True
        if self.need_reset:
            self.prev_cam = cam
            self.prev_head = None
        self.cur_cam = cam
        jx, jy = self.seq[self.n % len(self.seq)]
        self.n += 1
        self.jit_px = (jx, jy)
        self.jit = (2.0*jx/mc.iw, 2.0*jy/mc.ih)
        hb = None
        if self.head_bone is not None:
            hb = np.asarray(rig.bones[self.head_bone], np.float64)
        self.head_rel = np.eye(4)
        if hb is not None and self.prev_head is not None and not self.need_reset:
            try:
                self.head_rel = self.prev_head @ np.linalg.inv(hb)
            except np.linalg.LinAlgError:
                pass
        self.cur_head = hb

    def cam_uniforms(self, prog):
        """the jitter (every camClip program) + the previous camera (the forward programs' velocity)."""
        mc = self.mc
        mc._set(prog, "uJit", (float(self.jit[0]), float(self.jit[1])))
        if "uPCamPos" in prog and self.prev_cam is not None:
            pos, u, v, w, fl = self.prev_cam
            mc._set(prog, "uPCamPos", tuple(float(x) for x in pos)); mc._set(prog, "uPCamU", tuple(float(x) for x in u))
            mc._set(prog, "uPCamV", tuple(float(x) for x in v)); mc._set(prog, "uPCamW", tuple(float(x) for x in w))
            mc._set(prog, "uPFl", float(fl))

    def resolve(self):
        """after the SSS: t_mc / t_md / t_v + history -> self.out (the composite's mesh colour)."""
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        ctx.copy_framebuffer(mc.f_v, mc.f_ms_v)
        if self.cls_on:
            ctx.copy_framebuffer(mc.f_r, mc.f_ms_r)
        k = self.k
        i = self.n & 1
        dst, src = i, 1 - i
        p = self.prog
        mc._set(p, "uRes", (float(mc.iw), float(mc.ih)))
        mc._set(p, "uK", (float(k["blend"]), float(k["gamma"]), float(k["vel_reject"]), 1.0 if self.need_reset else 0.0))
        mc._set(p, "uK2", (float(k["hair_eps"]), float(k["depth_reject"]), float(k["luma_k"]), 1.0 if k["catmull"] else 0.0))
        mc._set(p, "uK3", (float(k["motion_blend"]), float(k["max_blend"]), 1.0 if k["unjitter"] else 0.0,
                                float(k["blend_static"])))
        mc._set(p, "uK4", (float(k["gamma_static"]), max(float(k["still_px"]), 1e-3), float(k["react_blend"]),
                           float(k["react_gamma"])))
        mc._set(p, "uJitPx", (float(self.jit_px[0]), float(self.jit_px[1])))
        mc._set(p, "uClsOn", 1.0 if self.cls_on else 0.0)
        if "uHeadRel" in p:
            p["uHeadRel"].write(np.ascontiguousarray(self.head_rel.T, np.float32).tobytes())
        pos, u, v, w, fl = self.cur_cam
        mc._set(p, "uCamPos", tuple(float(x) for x in pos)); mc._set(p, "uCamU", tuple(float(x) for x in u))
        mc._set(p, "uCamV", tuple(float(x) for x in v)); mc._set(p, "uCamW", tuple(float(x) for x in w))
        mc._set(p, "uFl", float(fl))
        pos, u, v, w, fl = self.prev_cam
        mc._set(p, "uPCamPos", tuple(float(x) for x in pos)); mc._set(p, "uPCamU", tuple(float(x) for x in u))
        mc._set(p, "uPCamV", tuple(float(x) for x in v)); mc._set(p, "uPCamW", tuple(float(x) for x in w))
        mc._set(p, "uPFl", float(fl))
        mc.t_mc.use(0); mc.t_md.use(1); mc.t_v.use(2); self.t_hc[src].use(3); self.t_ha[src].use(4)
        if self.cls_on:
            mc.t_r.use(5)
        self.f_h[dst].use(); ctx.viewport = (0, 0, mc.iw, mc.ih)
        ctx.disable(mgl.BLEND); ctx.disable(mgl.DEPTH_TEST)
        self.vao.render(mgl.TRIANGLE_STRIP)
        self.out = self.t_hc[dst]
        if self.p_sh is not None:                        # shade2-taa: sharpen the output (the history stays as is)
            q = self.p_sh
            mc._set(q, "uRes", (float(mc.iw), float(mc.ih)))
            mc._set(q, "uSharp", (float(self.sharp), 1.0 if self.cls_on else 0.0))
            self.t_hc[dst].use(0)
            if self.cls_on:
                mc.t_r.use(5)
            self.f_sh.use(); ctx.viewport = (0, 0, mc.iw, mc.ih)
            self.vao_sh.render(mgl.TRIANGLE_STRIP)
            self.out = self.t_sh
        # this frame becomes the previous one
        self.prev_cam = self.cur_cam
        self.prev_head = self.cur_head
        self.need_reset = False
