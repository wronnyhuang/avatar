"""shade-post: the final image of a physically based character (post_pbr.glsl).  GATED: MeshEngine builds it only
when the mesh config has mesh.pbr.post (a dict); absent = juno's post_juno.glsl / the shared post, untouched.

The engine's own post step (RTEngine._draw step 7) builds the HDR mip chain and draws vaos["post"] into f_out.  The
pyramid bloom needs its passes first, so vaos["post"] is a small proxy: render() runs the 13-tap down chain + the
tent up chain (half res .. 1/64), re-binds f_out / the viewport / t_hdr and draws the grade.  No rt_engine change.

mesh.pbr.post keys (all optional; defaults = POST_DEFAULTS):
  exposure   stops added to grade.exposure (scene linear)       bloom      energy-preserving lens spread (0..1 mix)
  bloom_wide falloff of the pyramid (0 = tight .. 1 = wide)      levels     pyramid depth (2..7)
  vignette   x grade.vignette, optical (scene light)             grain      x grade.grain
  slope / power / sat   AgX ASC-CDL look (Blender "Punchy" = 1, 1.35, 1.4)
  hue_restore  0..1 Oklab hue pull toward the scene's own hue    desat_start  display L above which the restore fades
  black      display-linear black floor (toe), 0 = AgX's own     ca         lateral CA px (0 = off)
  white_ev / black_ev  AgX log range about mid grey (6.5 / 10 = Blender); a lower white_ev = brighter, punchier highlights
  saturation (optional) replaces grade.saturation                debug      1 = show the bloom buffer
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# r2 (junor v1, lead: "contrasty cinematic portrait"): a shorter log range than Blender AgX (white 4.5 / shadow 7 stops
# about mid grey) = a deeper toe + brighter highlights; look power 1.4, sat 1.1; hue restore 0.8 keeps shadows from going red
POST_DEFAULTS = dict(exposure=0.5, bloom=0.04, bloom_wide=0.62, levels=6, vignette=1.0, grain=0.8, slope=1.0,
                     power=1.4, sat=1.1, hue_restore=0.8, desat_start=0.72, black=0.0, ca=0.0, white_ev=4.5, black_ev=7.0, saturation=None,
                     debug=0)


class PostVao:
    """stands in for engine.vaos["post"]: the bloom pyramid, then the grade."""

    def __init__(self, pp, vao):
        self.pp, self.vao = pp, vao

    def render(self, mode=None, *a, **k):
        self.pp.bloom()
        e = self.pp.eng
        e.f_out.use(); e.ctx.viewport = (0, 0, e.w, e.h)
        e.t_hdr.use(0)
        self.pp.t_up[0].use(1)
        self.vao.render(mode, *a, **k)

    def release(self):
        self.vao.release()


class PbrPost:
    def __init__(self, eng, cfg):
        import moderngl
        import rt_engine
        self.eng, self.mgl = eng, moderngl
        self.k = dict(POST_DEFAULTS); self.k.update({k: v for k, v in (cfg or {}).items() if not k.startswith("_")})
        ctx = eng.ctx
        src = open(os.path.join(HERE, "post_pbr.glsl")).read()
        P = lambda d: ctx.program(vertex_shader=rt_engine.VS_FULL, fragment_shader=eng.sh.head + f"#define {d}\n" + src)
        self.p_post = P("PBR_POST")
        self.p_down = P("PBR_BLOOM_DOWN")
        self.p_up = P("PBR_BLOOM_UP")
        self.v_down = ctx.vertex_array(self.p_down, [(eng.quad, "2f", "in_pos")])
        self.v_up = ctx.vertex_array(self.p_up, [(eng.quad, "2f", "in_pos")])
        self.vao = PostVao(self, ctx.vertex_array(self.p_post, [(eng.quad, "2f", "in_pos")]))
        self._alloc(eng.w, eng.h)

    def _alloc(self, w, h):
        ctx, mgl = self.eng.ctx, self.mgl
        n = max(2, min(7, int(self.k["levels"])))
        self.t_dn, self.f_dn, self.t_up, self.f_up = [], [], [], []
        for i in range(n):
            s = (max(1, w >> (i + 1)), max(1, h >> (i + 1)))
            for L, F in ((self.t_dn, self.f_dn), (self.t_up, self.f_up)):
                if L is self.t_up and i == n - 1:
                    break
                t = ctx.texture(s, 4, dtype="f2")
                t.filter = (mgl.LINEAR, mgl.LINEAR); t.repeat_x = t.repeat_y = False
                L.append(t); F.append(ctx.framebuffer([t]))
        self.n = n

    def bloom(self):
        e, ctx, mgl = self.eng, self.eng.ctx, self.mgl
        ctx.disable(mgl.BLEND)
        src, ssz = e.t_hdr, (e.w, e.h)
        p = self.p_down
        p["uSrc"].value = 0
        for i in range(self.n):
            t = self.t_dn[i]
            self.f_dn[i].use(); ctx.viewport = (0, 0) + t.size
            src.use(0)
            p["uRes"].value = (float(t.size[0]), float(t.size[1]))
            p["uSrcTexel"].value = (1.0/ssz[0], 1.0/ssz[1])
            p["uKaris"].value = 1.0 if i == 0 else 0.0
            self.v_down.render(mgl.TRIANGLE_STRIP)
            src, ssz = t, t.size
        p = self.p_up
        p["uLow"].value = 0; p["uCur"].value = 1
        p["uMixLow"].value = float(self.k["bloom_wide"])
        low = self.t_dn[self.n - 1]
        for i in range(self.n - 2, -1, -1):
            t = self.t_up[i]
            self.f_up[i].use(); ctx.viewport = (0, 0) + t.size
            low.use(0); self.t_dn[i].use(1)
            p["uRes"].value = (float(t.size[0]), float(t.size[1]))
            p["uLowTexel"].value = (1.0/low.size[0], 1.0/low.size[1])
            self.v_up.render(mgl.TRIANGLE_STRIP)
            low = t

    def uniforms(self):
        e, p, k = self.eng, self.p_post, self.k
        e._cam_uniforms(p)
        g = e.grade
        e.set(p, "uExposure", float(g["exposure"])); e.set(p, "uBloom", float(g.get("bloom", 1.0)))
        sat = k["saturation"] if k.get("saturation") is not None else g["saturation"]
        e.set(p, "uSaturation", float(sat)); e.set(p, "uVignette", float(g["vignette"]))
        e.set(p, "uGrain", float(g["grain"]))
        e.set(p, "uGain", tuple(g["gain"])); e.set(p, "uLift", tuple(g["lift"])); e.set(p, "uFade", 1.0)
        e.set(p, "uPP", (float(k["exposure"]), float(k["bloom"]), float(k["vignette"]), float(k["grain"])))
        e.set(p, "uPP2", (float(k["slope"]), float(k["power"]), float(k["sat"]), float(k["hue_restore"])))
        e.set(p, "uPP3", (float(k["black"]), float(k["desat_start"]), float(k["ca"]), float(k["debug"])))
        e.set(p, "uPP4", (float(k["white_ev"]), float(k["black_ev"]), 0.0, 0.0))
        e.set(p, "uHdr", 0); e.set(p, "uBloomT", 1)
