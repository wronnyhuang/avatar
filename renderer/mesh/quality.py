"""render-perf: Juno's mesh quality ladder (docs/CONTRACT--render.md section 5).

MeshEngine exposes it through RTEngine's interface: `eng.tiers` (cheapest first; floats whose SQUARE is the tier's
relative cost, which is what live/backends/rt.py feeds rt_adaptive.AdaptiveTier), `eng.tier`, `eng.tier_pending`,
`eng.set_tier(i)` (applied between frames), `eng.tier_names`.

KNOB REGISTRY — how a helper plugs a feature in
-----------------------------------------------
A knob = a name -> one value per tier (cheapest .. best) -> an attribute on MeshCharacter (`mc.<attr>`), which the
feature's code reads every frame (e.g. a uniform it sets from `self.sss_taps`).  To add one:

    from quality import register
    register("sss_taps", attr="sss_taps", values=[0, 5, 5, 9, 9, 13, 13, 13], doc="skin SSS blur taps (0 = pre-integrated only)")

`values` may be shorter than the ladder (the last value repeats) or a dict {tier index: value} (unlisted tiers take
the nearest LOWER listed tier).  A knob whose change needs GPU work (re-allocating targets) gives `apply=fn(mc, v)`.
The ladder's tier count is LADDER_N; the order in which tiers give things up is the GIVE-UP ORDER (the brief: scene scale -> hair fraction -> SSS taps -> shadow res/taps -> MSAA 4->2 -> mesh scale; the
FACE last — eye/cornea/skin materials keep full shading at every tier).  Tier 7 = everything on:
  7  best: scene at its preset scale, hair 100 %, SSS 13 taps, light "high" (shadow 2048^2, PCSS 16), MSAA 4x, mesh 1.0x
  6  scene (SDF props / set) internal scale 0.75
  5  hair strands 70 %
  4  hair 50 %, scene 0.6
  3  SSS 9 taps, hair 35 %
  2  SSS 5, light "mid" (2048^2, PCSS 8)
  1  light "low" (1024^2, PCSS 8), MSAA 2x, scene 0.5
  0  mesh at 0.75x (bilinear up in the composite), light "min" (1024^2, fixed PCF, analytic env), DoF 12 taps
Knob names in the integrated code: mesh.sss.taps (sss.SkinSSS.set_taps), mesh.hair.frac (hair.HairRenderer.frac),
mesh.light quality (light.MeshLight.set_quality -> shadow res + pcss_taps + probe), mc.msaa / mc.mesh_scale (targets).
Measured costs (renderer/tools/juno_bench.py --ladder) replace the estimates in COST via juno.json "mesh.quality.cost".
"""
import numpy as np

LADDER_N = 8


class Knob:
    __slots__ = ("name", "attr", "values", "doc", "apply")

    def __init__(self, name, attr, values, doc="", apply=None):
        self.name, self.attr, self.doc, self.apply = name, attr or name, doc, apply
        self.values = values

    def at(self, tier, n=LADDER_N):
        v = self.values
        if isinstance(v, dict):
            d = {int(k): x for k, x in v.items()}
            lo = [k for k in sorted(d) if k <= tier]
            return d[lo[-1] if lo else min(d)]
        v = list(v)
        return v[min(tier, len(v) - 1)]


KNOBS = {}


def register(name, attr=None, values=None, doc="", apply=None):
    """Add or replace a knob (see the module doc)."""
    KNOBS[name] = Knob(name, attr, values, doc, apply)
    return KNOBS[name]


# ---------------------------------------------------------------- the GPU-side appliers (re-used targets, cached)
def _apply_msaa_scale(mc, _v=None):
    """MSAA sample count + internal mesh scale -> the forward targets (cached per (samples, iw, ih))."""
    s = max(1, min(int(getattr(mc, "msaa", 4)), mc.ctx.max_samples))
    sc = float(getattr(mc, "mesh_scale", 1.0))
    iw, ih = max(8, int(round(mc.w*sc))), max(8, int(round(mc.h*sc)))
    key = (s, iw, ih)
    cache = mc.__dict__.setdefault("_tgt_cache", {})
    names = ("rb_c", "rb_d", "rb_z", "f_ms", "t_mc", "t_md", "f_res", "f_ms_c", "f_ms_d")
    snames = ("w", "h", "rb_s", "f_ms_s", "t_s", "t_h", "f_s", "f_h", "rb_a", "f_ms_a", "t_a", "f_a")   # + albedo split     # the SSS stage's targets (sss.ms_attachments)
    if getattr(mc, "mtaa", None) is not None:              # shade-taa (gated): the velocity attachment
        from taa import TGT_NAMES
        names = names + TGT_NAMES
    sss = getattr(mc, "sss", None)
    grab = lambda: ({n: getattr(mc, n) for n in names}, {n: getattr(sss, n) for n in snames if sss is not None and hasattr(sss, n)})
    cur = mc._tgt_key
    if cur not in cache:
        cache[cur] = grab()
    if key == cur:
        return
    if key not in cache:
        msaa0 = mc.msaa
        mc.msaa = s
        mc._alloc_targets(iw, ih)
        mc.msaa = msaa0
        cache[key] = grab()
    for n, v in cache[key][0].items():
        setattr(mc, n, v)
    for n, v in cache[key][1].items():
        setattr(sss, n, v)
    mc.msaa, mc.iw, mc.ih = s, iw, ih
    mc._tgt_key = key


def _apply_shadow_res(mc, v):
    v = int(v)
    if v == mc.shadow_res and getattr(mc, "f_shadow", None) is not None:
        return
    cache = mc.__dict__.setdefault("_shadow_cache", {})
    if mc.shadow_res not in cache:
        cache[mc.shadow_res] = (mc.t_shadow, mc.f_shadow)
    if v not in cache:
        mgl = mc.mgl
        t = mc.ctx.depth_texture((v, v))
        t.compare_func = ""
        t.filter = (mgl.LINEAR, mgl.LINEAR)
        t.repeat_x = t.repeat_y = False
        cache[v] = (t, mc.ctx.framebuffer(depth_attachment=t))
    mc.t_shadow, mc.f_shadow = cache[v]
    mc.shadow_res = v


def _apply_light(mc, v):
    """mesh.light quality name -> pcss taps + probe on MeshLight, and the shadow map size it wants."""
    lt = getattr(mc, "light", None)
    if lt is not None and hasattr(lt, "set_quality"):
        _apply_shadow_res(mc, lt.set_quality(str(v)))


def _apply_sss(mc, v):
    s = getattr(mc, "sss", None)
    if s is not None and getattr(s, "on", False):
        s.set_taps(int(v))


def _apply_hair(mc, v):
    if getattr(mc, "hair", None) is not None:
        mc.hair.frac = float(v)


# ---------------------------------------------------------------- the knobs (cheapest tier first)
register("mesh_scale", "mesh_scale", [0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         "internal resolution of the mesh forward pass (bilinear up in the composite)", apply=_apply_msaa_scale)
register("msaa", "msaa", [2, 2, 4, 4, 4, 4, 4, 4], "MSAA samples of the forward pass", apply=_apply_msaa_scale)
register("light", "light_q", ["min", "low", "mid", "high", "high", "high", "high", "high"],
         "mesh.light quality (light.QUALITY: shadow res, PCSS taps, probe)", apply=_apply_light)
register("sss_taps", "sss_taps", [5, 5, 5, 9, 13, 13, 13, 13], "mesh.sss.taps (screen-space SSS taps per direction)",
         apply=_apply_sss)
register("hair_frac", "hair_frac", [0.35, 0.35, 0.35, 0.35, 0.5, 0.7, 1.0, 1.0], "mesh.hair.frac (strand fraction)",
         apply=_apply_hair)
def _apply_detail(mc, v):         # render4: procedural micro-detail (skin pores, knit warp) -- off on the 2 cheapest tiers
    if getattr(mc, "face", None) is not None:
        mc.face.detail = float(v)


def _apply_hair_shadow(mc, v):     # render-perf (PERF2): the key-shadow strand subset (hair.py shadow_frac/seg)
    import os
    h = getattr(mc, "hair", None)
    if h is None or os.environ.get("JUNO_HAIR_SHADOW_FRAC"):
        return
    h.shadow_frac, h.shadow_seg = float(v[0]), int(v[1])


register("hair_shadow", "hair_shadow", [(0.35, 2)],
         "PERF2: (strand fraction, point stride) drawn into the key shadow map; .35/2 = 13.6 -> 11.7 ms back-to-back at "
         "1080p on an A40, face unchanged (cloud/results-p2r2/hs_sheet.png: mean |d| .06/255, diffs only inside the hair)",
         apply=_apply_hair_shadow)
register("detail", "detail", [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         "render4 procedural micro-detail (skin pore spec grain, knit macro warp): 0 off", apply=_apply_detail)
register("dof_n", "dof_n", [12, 24, 24, 24, 24, 24, 24, 24], "mesh DoF gather taps (in-focus pixels pay nothing)")
register("scene_scale", "scene_scale", [0.5, 0.5, 0.6, 0.6, 0.6, 0.75, 0.75, 1.0],
         "the SDF scene pass's internal scale (props; RTEngine levels, nearest level at or below; 1.0 = the preset)")

# estimated relative cost per tier (cheapest first, best = 1.0); juno_bench measures the real ones
COST = [0.40, 0.50, 0.62, 0.70, 0.78, 0.86, 0.93, 1.0]


class Ladder:
    def __init__(self, mcfg=None):
        q = dict((mcfg or {}).get("quality", {}) or {})
        self.over = {k: v for k, v in q.items() if k in KNOBS or k not in ("cost", "start")}
        self.cost = [float(c) for c in q.get("cost", COST)]
        self.n = LADDER_N
        self.start = int(q.get("start", self.n - 1))

    def values(self, tier):
        tier = max(0, min(self.n - 1, int(tier)))
        out = {}
        for name, k in KNOBS.items():
            out[name] = Knob(name, k.attr, self.over[name]).at(tier) if name in self.over else k.at(tier)
        return out

    def apply(self, mc, tier):
        vals = self.values(tier)
        if getattr(mc, "_tgt_key", None) is None:     # the targets MeshCharacter.__init__ made: full size, mc.msaa
            mc._tgt_key = (mc.msaa, mc.w, mc.h)
        for name, v in vals.items():
            k = KNOBS[name]
            if k.apply is None:
                setattr(mc, k.attr, v)
        # the target-changing knobs: attributes first, then one re-target
        if "msaa" in vals:
            mc.msaa = int(vals["msaa"])
        if "mesh_scale" in vals:
            mc.mesh_scale = float(vals["mesh_scale"])
        for name, v in vals.items():
            k = KNOBS[name]
            if k.apply is not None and k.apply is not _apply_msaa_scale:
                k.apply(mc, v)
        _apply_msaa_scale(mc)
        mc.quality_tier = int(tier)
        return vals

    def names(self):
        return [f"m{i}" for i in range(self.n)]

    def tiers(self):
        return [float(np.sqrt(max(c, 1e-6))) for c in self.cost]
