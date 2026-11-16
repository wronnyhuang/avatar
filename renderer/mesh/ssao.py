"""shade-post: contact AO on the mesh layer (ssao.glsl).  GATED: MeshCharacter builds it only when the mesh config
has mesh.pbr.ao (a dict, or true); absent = no pass, FS_COMP compiled exactly as before (no PBR_AO define).

mesh.pbr.ao keys (defaults AO_DEFAULTS):
  radius    world radius of the obscurance sphere, scene units (her height ~2.1 units: 0.045 ~ 3.6 cm)
  strength  how dark a crease gets (1 = a 90-degree inner corner at 50 %)
  far       an occluder more than far x radius in front fades to neutral (no halos from a lock of hair far in front)
  min_px    below this projected radius (px) the pass reads open (the wide shot costs ~nothing)
  amount    the composite's mix: colour *= mix(1, ao, amount)
  scale     resolution of the pass (0.5 = half)
  floor     the darkest the AO gets (0.5)
  hole      1 = a pixel with nearly every sample in front of it (the open-mouth cavity, nostrils) is left open: those
            materials model their own occlusion (mat_mouth_pbr); 0 = off
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AO_DEFAULTS = dict(radius=0.05, strength=1.4, far=2.5, min_px=1.5, amount=0.85, scale=0.5, floor=0.5, hole=1.0)


class MeshAO:
    def __init__(self, mc, cfg):
        import moderngl
        self.mc, self.mgl = mc, moderngl
        self.k = dict(AO_DEFAULTS)
        if isinstance(cfg, dict):
            self.k.update({k: v for k, v in cfg.items() if not k.startswith("_")})
        ctx = mc.ctx
        src = open(os.path.join(HERE, "ssao.glsl")).read()
        vs = mc._src("VS_FULLM")
        P = lambda d: ctx.program(vertex_shader=vs, fragment_shader=mc.head + f"#define {d}\n" + src)
        self.p_main, self.p_blur = P("SSAO_MAIN"), P("SSAO_BLUR")
        self.v_main = ctx.vertex_array(self.p_main, [(mc.quad, "2f", "in_pos")])
        self.v_blur = ctx.vertex_array(self.p_blur, [(mc.quad, "2f", "in_pos")])
        self.t = []
        self.size = None

    def _alloc(self):
        mc, mgl = self.mc, self.mgl
        s = (max(1, int(mc.w*float(self.k["scale"]))), max(1, int(mc.h*float(self.k["scale"]))))
        if s == self.size:
            return
        for t, f in self.t:
            f.release(); t.release()
        self.t = []
        for _ in range(2):
            t = mc.ctx.texture(s, 2, dtype="f2")
            t.filter = (mgl.LINEAR, mgl.LINEAR); t.repeat_x = t.repeat_y = False
            self.t.append((t, mc.ctx.framebuffer([t])))
        self.size = s

    def run(self, cam):
        """after the resolve (+ SSS): t_md -> the blurred AO in self.t[0][0]."""
        mc, ctx, mgl = self.mc, self.mc.ctx, self.mgl
        self._alloc()
        k = self.k
        sz = (float(self.size[0]), float(self.size[1]))
        ctx.disable(mgl.BLEND)
        p = self.p_main
        mc._cam(p, cam, getattr(mc, "nf", (0.1, 100.0)), (mc.w, mc.h))
        mc._set(p, "uFull", (float(mc.w), float(mc.h))); mc._set(p, "uRes", sz)
        mc._set(p, "uAo", (float(k["radius"]), float(k["strength"]), float(k["far"]), float(k["min_px"])))
        mc._set(p, "uAo2", (float(k["floor"]), float(k["hole"])))
        mc._set(p, "uMD", 0)
        mc.t_md.use(0)
        self.t[0][1].use(); ctx.viewport = (0, 0) + self.size
        self.v_main.render(mgl.TRIANGLE_STRIP)
        p = self.p_blur
        mc._set(p, "uRes", sz); mc._set(p, "uSrc", 0)
        for a, b, d in ((0, 1, (1.0/sz[0], 0.0)), (1, 0, (0.0, 1.0/sz[1]))):
            self.t[a][0].use(0)
            self.t[b][1].use(); ctx.viewport = (0, 0) + self.size
            mc._set(p, "uDir", d)
            self.v_blur.render(mgl.TRIANGLE_STRIP)

    def bind(self, prog, unit=10):
        self.t[0][0].use(unit)
        self.mc._set(prog, "uAO", unit)
        self.mc._set(prog, "uAoAmt", float(self.k["amount"]))
