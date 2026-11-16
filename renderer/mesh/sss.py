"""render-skin: screen-space separable subsurface scattering for the mesh character (M5 of
docs/CONTRACT--render.md).

The forward pass writes a THIRD colour attachment (MSAA, RGBA16F) "S": rgb = the skin's diffuse radiance (key + fill +
ambient + transmission, already * albedo and * coverage), a = 1 - skin coverage.  Non-skin materials write
(0,0,0,alpha), so an opaque occluder (eye, cloth, hair) zeroes S and a blended one (cornea, hair coverage) attenuates
it exactly like the colour layer.  The skin's colour attachment then carries only specular + rim.
Per frame, after the MSAA resolve:  S -> t_s (blit) -> H blur -> t_h -> V blur, ADDED (ONE, ONE) onto the resolved
char layer t_mc.  So SSS works on the resolved buffers; MSAA edges stay those of the colour layer.

Quality knob  mesh.sss.taps : 13 / 9 / 5 / 0.  0 = no extra attachment work: the skin shader falls back to its
pre-integrated wrap (all diffuse in the colour attachment) and no blur pass runs.

Config (juno.json "mesh": {"sss": {...}}), all optional:
  taps 13, scale 1.0 (x the asset's sss_radius_mm), range 3.0 (half range in red radii), strength [r,g,b] (share of the
  diffuse that scatters; the rest stays sharp), depth_sigma_mm 6.0, max_px_frac 0.06 (of the frame height),
  trans_key 1.0, trans_rim 1.0 (transmission gains), wrap 1.0 (pre-integrated wrap amount, fallback + residual).
Units: the asset is in metres; the rig scales it by k = H_scene/height_m scene units per metre."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAXK = 17


def burley_1d(x, d):
    """Burley normalised-diffusion shape (two exponentials), unnormalised, at |x| for scatter distance d."""
    x = np.abs(x)
    return np.exp(-x/d) + np.exp(-x/(3.0*d))


def make_kernel(n, radii, half_range, strength):
    """n taps, offsets in [-1, 1] of half_range (denser near the centre: x^2 spacing), per-channel weights.  Each
    channel's weights sum to 1; the centre tap also carries the (1 - strength) unscattered share."""
    n = int(n) | 1
    u = np.linspace(-1.0, 1.0, n)
    off = np.sign(u)*u*u
    # tap areas (midpoints between neighbours)
    edges = np.concatenate([[-1.0], 0.5*(off[1:] + off[:-1]), [1.0]])
    area = np.diff(edges)
    K = np.zeros((n, 4), np.float32)
    K[:, 3] = off
    c = n//2
    for ch in range(3):
        d = max(float(radii[ch]), 1e-6)
        # the area-weighted profile, d from the asset's radius (Burley's d ~ radius/2.5 makes the tail end near 3 r)
        w = burley_1d(off*half_range, d/2.5)*area
        w = w/w.sum()
        s = float(np.clip(strength[ch], 0.0, 1.0))
        w = s*w
        w[c] += 1.0 - s
        K[:, ch] = w
    return K


class SkinSSS:
    def __init__(self, mc, cfg=None, radii_m=None, k_su_per_m=1.0):
        """mc = the MeshCharacter (ctx, w, h, msaa, head, src helpers).  radii_m = the skin's scatter radii in METRES
        per channel (asset sss_radius_mm / 1000); k_su_per_m = scene units per asset metre."""
        import moderngl
        self.mgl = moderngl
        self.mc = mc
        self.ctx = mc.ctx
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.taps = int(os.environ.get("JUNO_SSS_TAPS", cfg.get("taps", 13)))
        self.scale = float(cfg.get("scale", 1.0))
        self.rgb_scale = np.asarray(cfg.get("rgb_scale", [1.0, 1.0, 1.0]), float)[:3]   # render8: per-channel radius (wider red)
        self.range = float(cfg.get("range", 3.0))
        self.strength = list(cfg.get("strength", [0.85, 0.70, 0.55]))
        self.depth_sigma_mm = float(cfg.get("depth_sigma_mm", 6.0))
        self.max_px_frac = float(cfg.get("max_px_frac", 0.06))
        self.trans_key = float(cfg.get("trans_key", 1.0))
        self.trans_rim = float(cfg.get("trans_rim", 1.0))
        self.wrap = float(cfg.get("wrap", 1.0))
        self.k = float(k_su_per_m)
        self.radii_m = np.asarray(radii_m if radii_m is not None else [3.6e-3, 1.4e-3, 0.8e-3], float)
        self.on = self.taps > 0
        # shade (gated, mesh.pbr.skin.albedo_split): S carries skin IRRADIANCE, a 4th attachment the skin albedo
        # (premultiplied by skin coverage); the V pass multiplies the scattered light by the albedo at the pixel
        # (post-scatter texturing: freckles / pores / lip line stay sharp instead of being smeared by the profile)
        self.alb_split = bool(getattr(mc, "pbr_split", False)) and self.on
        src = open(os.path.join(HERE, "sss.glsl")).read()
        if self.on:
            fdef = "#define FS_SSS\n" + ("#define SSS_ALB\n" if self.alb_split else "")
            self.prog = self.ctx.program(vertex_shader=mc.head + "#define VS_SSS\n" + src,
                                         fragment_shader=mc.head + fdef + src)
            self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
            self.vao = self.ctx.vertex_array(self.prog, [(self.quad, "2f", "in_pos")])
            self._set("uS", 0); self._set("uS0", 1); self._set("uMD", 2)
            self._set("uAlb", 3)                                        # shade (SSS_ALB only)
            self.set_taps(self.taps)

    # ------------------------------------------------------------------ knobs
    def set_taps(self, n):
        n = int(n)
        if n <= 0:
            self.taps = 0
            return
        if not self.on:
            raise RuntimeError("SSS was built off (taps 0); rebuild MeshCharacter to turn it on")
        self.taps = min(n | 1, MAXK)
        su = self.radii_m*self.k*self.scale*self.rgb_scale   # per-channel scatter radius, scene units
        self.half_range = self.range*float(su.max())
        K = make_kernel(self.taps, su, self.half_range, self.strength)
        Kp = np.zeros((MAXK, 4), np.float32); Kp[:len(K)] = K
        if "uK" in self.prog:
            self.prog["uK"].write(Kp.tobytes())
        self._set("uN", int(self.taps))
        self._set("uWorldR", float(self.half_range))
        sig = self.depth_sigma_mm*1e-3*self.k*self.scale
        self._set("uDepthK", float(1.0/(2.0*sig*sig)))

    def _set(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    # ------------------------------------------------------------------ targets (called from MeshCharacter._alloc_targets)
    def ms_attachments(self, w, h, samples):
        """the extra MSAA colour attachment(s) for the forward framebuffer ([] when off)."""
        if not self.on:
            return []
        ctx, mgl = self.ctx, self.mgl
        self.w, self.h = w, h
        self.rb_s = ctx.renderbuffer((w, h), 4, samples=samples, dtype="f2")
        self.f_ms_s = ctx.framebuffer([self.rb_s])
        self.t_s = ctx.texture((w, h), 4, dtype="f2"); self.t_h = ctx.texture((w, h), 4, dtype="f2")
        for t in (self.t_s, self.t_h):
            t.filter = (mgl.LINEAR, mgl.LINEAR); t.repeat_x = t.repeat_y = False
        self.f_s = ctx.framebuffer([self.t_s]); self.f_h = ctx.framebuffer([self.t_h])
        if self.alb_split:                                              # shade
            self.rb_a = ctx.renderbuffer((w, h), 4, samples=samples)
            self.f_ms_a = ctx.framebuffer([self.rb_a])
            self.t_a = ctx.texture((w, h), 4)
            self.t_a.repeat_x = self.t_a.repeat_y = False
            self.f_a = ctx.framebuffer([self.t_a])
            return [self.rb_s, self.rb_a]
        return [self.rb_s]

    # ------------------------------------------------------------------ per frame
    def clear(self):
        """after the depth prefill: S = (0,0,0,1) (nothing is skin)."""
        if self.on and self.taps > 0:
            self.f_ms_s.clear(0.0, 0.0, 0.0, 1.0)
            if self.alb_split:
                self.f_ms_a.clear(0.0, 0.0, 0.0, 0.0)

    def uniforms(self, prog):
        """the skin forward program's SSS / transmission uniforms."""
        split = 1.0 if (self.on and self.taps > 0) else 0.0
        vals = dict(uSssSplit=split, uSssWrap=self.wrap, uTransKey=self.trans_key, uTransRim=self.trans_rim,
                    uSssRadSu=tuple(float(x) for x in self.radii_m*self.k*self.scale*self.rgb_scale))
        for k, v in vals.items():
            if k in prog:
                prog[k].value = v

    def apply(self, cam, t_md, f_dst):
        """resolve S, blur H then V, add onto f_dst (the resolved char layer framebuffer)."""
        if not (self.on and self.taps > 0):
            return
        ctx, mgl = self.ctx, self.mgl
        ctx.copy_framebuffer(self.f_s, self.f_ms_s)
        if self.alb_split:                                              # shade
            ctx.copy_framebuffer(self.f_a, self.f_ms_a)
            self.t_a.use(3)
        fl = float(cam[4])
        self._set("uRes", (float(self.w), float(self.h)))
        self._set("uFlPx", fl*self.h*0.5)
        self._set("uMaxPx", self.max_px_frac*self.h)
        ctx.disable(mgl.DEPTH_TEST); ctx.disable(mgl.CULL_FACE)
        self.t_s.use(1); t_md.use(2)
        # H
        self.f_h.use(); ctx.viewport = (0, 0, self.w, self.h)
        self.f_h.clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(mgl.BLEND)
        self.t_s.use(0)
        self._set("uPass", 0); self._set("uDir", (1.0, 0.0))
        self.vao.render(mgl.TRIANGLE_STRIP)
        # V + recombine
        f_dst.use(); ctx.viewport = (0, 0, self.w, self.h)
        self.t_h.use(0)
        self._set("uPass", 1); self._set("uDir", (0.0, 1.0))
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
        self.vao.render(mgl.TRIANGLE_STRIP)
        ctx.disable(mgl.BLEND)


def from_asset(mc, asset, mcfg):
    """build the SSS stage from the asset's skin material params + the "mesh" config block."""
    mats = asset.meta.get("materials", {}) or {}
    radii = None
    for mid, md in mats.items():
        md = md or {}
        if md.get("type") == "skin" or "skin" in mid:
            r = (md.get("params") or {}).get("sss_radius_mm")
            if r:
                radii = np.asarray(r[:3], float)*1e-3
                if "head" in mid:
                    break
    scfg = dict((mcfg or {}).get("sss", {}) or {})
    if radii is None and scfg.get("radius_mm"):
        radii = np.asarray(scfg["radius_mm"][:3], float)*1e-3
    hm = float(asset.meta.get("height_m") or 1.7)
    k = float((mcfg or {}).get("H_scene", 2.10))/hm
    return SkinSSS(mc, scfg, radii, k)
