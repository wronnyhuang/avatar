"""render-hair: strand hair, brows and lashes for MeshCharacter (asset format: CONTRACT--asset.md section 6).

    hr = HairRenderer(mc)            # mc = MeshCharacter (its ctx, asset, posed vertex buffer, shadow map, lights)
    hr.shadow(rig, light_vp)         # into the bound key shadow framebuffer (ribbons face the light)
    hr.draw(rig, cam, nf, lights)    # into the bound MSAA forward target, after the opaque submeshes

Every strand is an instanced camera-facing ribbon expanded in the vertex shader (GL 3.3 core, gl_VertexID/InstanceID;
no vertex buffers): the points live in a float texture (bone groups: rest positions; surface groups: positions in the
REST frame of the root triangle), so deformation is one skin matrix (follow "bone") or the posed root triangle read
from a copy of the skin pass's vb_posed (follow "surface": brows / lashes ride the lid and brow shapes).
Coverage: physical width x width_k, drawn at >= min_px pixels, coverage = width/drawn -> k = floor(cov*N + hash) of the
N MSAA samples via gl_SampleMask (GL_ARB_sample_shading; a per-strand rotated mask so stacked strands land on different
samples — plain alpha-to-coverage uses one pattern per alpha value and caps a stack at k/N), alpha-to-coverage fallback;
depth-tested + depth-written, the MSAA resolve is the order-independent blend; no sorting.
Hair cap: the groom's root density (root_uv) baked once into a head-uv texture (dark scalp under the strands with a soft
hairline, a soft base under the brows, a lash line), drawn premultiplied over the head skin before the strands.
Knobs also: cap (false | {res, strength, <kind>_alpha, <kind>_sigma_mm}), width_k / len_k per kind, fuzz_frac, diffuse_k,
deep_mm; debug envs JUNO_HAIR_CAP=0, JUNO_HAIR_STRANDS=0, JUNO_HAIR_DEBUG=1 (flat green), JUNO_HAIR_MINPX.

 hair-shade -- CONFIG SOURCE: the asset json's "hair_render" dict when present (e.g. junor.asset), else the
character json's mesh.hair; env JUNO_HAIR_CFG='{"k": v}' shallow-updates it (dev).  "model": "v2" switches on the
physically based path (#define HAIR_V2 in every hair program; absent = the code path, pixel-identical):
  shading   Marschner R / TT / TRT in Karis 2016 form (longitudinal Gaussians on sin(theta), cuticle tilt R -alpha,
            TT +alpha/2, TRT +1.5 alpha; cheap azimuthal terms; Fresnel eta 1.55), absorption from melanin (Chiang 2016:
            sigma_a = eu (.419,.697,1.37) + pheo (.187,.4,1.05)) so TT / TRT are warm, Karis's fake multiple
            scattering (wrap diffuse on a view-facing normal blended with the volume normal, saturating in shadow) on
            the groom's colour; key (deep-opacity shadowed) + fill (R+TRT, widened) + rim (R+TT) + setAmbient.
  shadow    a deep opacity map: one extra light-space pass of a strand subset accumulates coverage into 4 cumulative
            slabs behind the key map's first occluder (RGBA16F, shadow_res x dom_scale); 4 rotated bilinear taps.
  cost      a depth-only pre-pass of the strands (coverage via the sample mask) + a shading pass with forced early
            fragment tests and depth EQUAL: every sample is shaded once (the front strand), not once per layer.
  v2 knobs (defaults in V2_DEFAULTS):
    eumelanin 3.0, pheomelanin .25   absorption (dark brown; 1.5 = light brown / orange TT, 6+ = near black)
    tip_melanin .9                   tip melanin x;  melanin_jitter (.12, .3) per strand / per clump
    root_mul / tip_mul (1,1,1) / (.8,.8,.8)   colour multipliers over the groom's color_root / color_tip (MS colour)
    alpha_deg 3.0, beta .30          cuticle tilt (deg), R longitudinal roughness; rough_jitter .35 (per strand +-)
    shift_jitter_deg (1.5, 4, 1.2)   highlight shift noise per strand, per clump, along the strand (breaks the band)
    gain (5, .5, 1.5, 3.0)           R (key), TT, TRT, multiple scattering;  r_tint .45 (R takes the hair's hue)
    tt_sat .55, ms_sat .4            TT/TRT colour saturation, MS saturation in key shadow (1 = Karis: copper backs)
    ms_pow 1.0, ms_vol .6, ms_trans .5   MS colour = colour^ms_pow, volume-normal share, key-shadow exponent on MS
    root_occ (.45, .25), mass_occ .55    root occlusion floor / length (u), lock-under-lock occlusion
    amb 1.0, amb_spec .5, fill_area .25  ambient diffuse, ambient R sheen, fill lobe widening
    r_var .5                         per-strand highlight brightness +- (streaks)
    aa 1.0                           specular AA: sub-pixel strands widen their lobes / damp their shift
    tip_fade (.92, .85)              u where the tip alpha starts to fall, alpha left at the tip
    dom true, dom_scale .5, dom_mm (1.5, 4, 10, 30), dom_k .55, dom_frac .5   deep opacity map: on, resolution x the
                                     key map, slab boundaries (mm behind the first occluder), optical depth per unit
                                     strand coverage, strand fraction drawn into it
    follow {"bone": "spine_03", "hang_k": .9, "u0": .3, "lag_s": .12, "lag_k": .5, "lag_max_deg": 4,
            "y_hi": 0, "y_lo": -.10}  (absent = rigid; y_* = rest-space band rel. to the head joint: rigid above y_hi)
                                     past u0 (and below the y band) the strand blends toward a target that keeps
                                     only the head's twist about the vertical relative to the follow bone (hangs on
                                     a nod / tilt, still turns with a yaw) and trails a critically damped copy of the head
                                     rotation (about the head pivot, clamped); dt = 1/FPS per frame
    lod_frac [[px_per_mm, frac], ...]  camera-distance strand fraction (head scale on screen; min with the quality
                                     ladder's frac; absent = off).  1080p: face cam ~2.0 px/mm, bust ~1.3, scene ~.6
    far {"ppm": [1.0, .45], "min_px": 2.5, "width_k": 1.8, "frac": .5, "smooth": 1.0, "edge_w": 4.0}  (on in v2;
                                     {"on": false} = off) far field: as the head's px/mm drops from ppm[0] to ppm[1]
                                     strands get fewer (frac), wider (width_k) and >= min_px, and per-strand / per-lock
                                     shift, R and colour variance fade by "smooth", lock-rim strands widen by edge_w -> a soft solid mass, one band
    cap: v2 default {"scalp_alpha": .85, "scalp_sigma_mm": 6} (soft wide under-paint), shaded with the DOM
    shadow_keep 4.0                  key-map strands survive with p = true texel coverage x this (stable hash) -> a
                                     sparse hairline casts a soft partial shadow on the skin, the mass stays opaque;
                                     fuzz casts none (0 = the binary map)
    prepass true                     the pre-pass + early-Z shading pass (env JUNO_HAIR_PREPASS=0 / JUNO_HAIR_EARLYZ=0)
  debug: JUNO_HAIR_DEBUG=2 shows the key transmittance; JUNO_HAIR_DOM=0 (single-map transmittance);
  JUNO_HAIR_DOM_DUMP=x.npy saves the DOM.  Bench: tools/hair_bench.py.

 HAIR2 -- hair_render.layer (v2 only; absent / {"on": false} / env JUNO_HAIR_LAYER=0 = the MSAA path above):
  the analytic-coverage hair layer.  (1) the opaque depth of the forward target is blitted into the hair's own targets;
  (2) COVERAGE pass: every strand at its TRUE width (layer.width_k per kind, default 1), drawn as a ribbon of
  width + 2 x tent_px and accumulating width x tent(offset) additively into a single-sample R16F target (sums to the
  true width in px per pixel row whatever the sub-pixel position: thin strands are faint and smooth, never dotted),
  depth-tested (no writes) against the opaque depth; it follows lod_frac with coverage x 1/f; (3) COLOUR pass: the v2
  pre-pass + early-Z shading into the hair's own MSAA colour/distance target with full sample masks (col_minpx ribbons,
  col_frac strands); (4) composite() after the char layer's resolve + SSS: alpha = 1 - exp(-kappa x coverage), colour
  = resolved hair colour / its sample coverage (hole fill from 16 taps where the colour pass drew nothing), blended
  premultiplied over t_mc / t_md.  Strand ends: width constant to taper_u then tapering, per-strand / per-lock length
  cut len_var -> ends thin by count.  followers (extra GPU strands around each strand in the coverage pass) exist but
  are off (the groom's density at true width is already opaque).  Debug: JUNO_HAIR_LDBG=1 alpha / 2 colour coverage /
  3 colour at alpha 1, JUNO_HAIR_COV_DUMP=x.npy.  NOTE moderngl 5.13 has no ctx.depth_mask: depth writes are per
  framebuffer (fbo.depth_mask).

Quality knobs (juno.json mesh.hair or the constructor): frac (strand fraction 1.0/0.7/0.5/0.35 — strands are
pre-shuffled so any prefix is a uniform subsample; widths grow 1/sqrt(frac) to keep the mass), seg (point stride 1/2/3),
min_px (narrowest drawn ribbon), shadow (hair into the key map on/off), env JUNO_HAIR_FRAC / JUNO_HAIR_SEG override.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEXW = 2048            # texels per row of the data textures

# group defaults when juno.asset.json misses a key (shading only; geometry always comes from the .bin)
DEFAULTS = dict(width_root_m=8e-5, width_tip_m=2e-5, color_root=[0.06, 0.035, 0.022], color_tip=[0.15, 0.09, 0.05],
                roughness=0.35, melanin=0.8)


def _quat(R):
    """3x3 rotation -> unit quaternion (w, x, y, z)"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0)*2
        q = np.array([0.25*s, (R[2, 1] - R[1, 2])/s, (R[0, 2] - R[2, 0])/s, (R[1, 0] - R[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))*2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k])/s
        q[1 + i] = 0.25*s
        q[1 + j] = (R[j, i] + R[i, j])/s
        q[1 + k] = (R[k, i] + R[i, k])/s
    return q/np.linalg.norm(q)


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _slerp(a, b, t):
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d < 0:
        b = -b; d = -d
    if d > 0.9995:
        r = a + t*(b - a)
        return r/np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - t)*th)*a + np.sin(t*th)*b)/np.sin(th)


def _rows(n):
    return max(1, (n + TEXW - 1)//TEXW)


def _pack(arr4):
    """(n,4) float32 -> padded (rows*TEXW, 4)"""
    n = arr4.shape[0]
    out = np.zeros((_rows(n)*TEXW, 4), np.float32)
    out[:n] = arr4
    return out



# HAIR2: the hair layer's composite (fullscreen, premultiplied over the char layer)
_LCOMP_VS = """
void main(){ vec2 q = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)); gl_Position = vec4(q*2.0 - 1.0, 0.0, 1.0); }
"""
_LCOMP_FS = """
uniform sampler2D uHC, uHD, uCov;
uniform float uKappa;
uniform int uDbg;
uniform int uFill;
layout(location=0) out vec4 oC;
layout(location=1) out vec4 oD;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(uHC, p, 0);
  float cv = texelFetch(uCov, p, 0).r;
  if(cv <= 0.0) discard;
  vec4 dd = texelFetch(uHD, p, 0);
  if(c.a < 0.5 && uFill == 1){
    // coverage without (enough) colour samples: a strand the colour pass did not draw (subset / thinner ribbon) ->
    // the colour of the hair around it (two rings of taps, weighted by their sample coverage)
    vec4 acc = c*4.0, accd = dd*4.0;
    for(int i = 0; i < 8; i++){
      float an = 0.785398*float(i);
      for(int r = 1; r <= 2; r++){
        ivec2 q = p + ivec2(round(vec2(cos(an), sin(an))*2.0*float(r)));
        acc += texelFetch(uHC, q, 0); accd += texelFetch(uHD, q, 0);
      }
    }
    c = acc; dd = accd;
  }
  if(c.a < 1e-4) discard;
  float al = 1.0 - exp(-uKappa*cv);
  vec3 col = c.rgb/c.a;
  float t = dd.r/c.a;
  if(uDbg == 1){ col = vec3(al); }
  if(uDbg == 2){ col = vec3(c.a); al = 1.0; }
  if(uDbg == 3){ al = 1.0; }
  oC = vec4(col*al, al);
  oD = vec4(t*al, 0.0, 0.0, al);
}
"""


class HairGroup:
    pass


class HairRenderer:
    def __init__(self, mc, cfg=None):
        import moderngl
        self.mgl = moderngl
        self.mc = mc
        ctx = self.ctx = mc.ctx
        A = mc.A
        if cfg is None:
            # hair-shade: the ASSET may carry its own hair config (meta "hair_render", e.g. junor), else the
            # character json's mesh.hair; env JUNO_HAIR_CFG (json) shallow-updates either (dev knob)
            hr = A.meta.get("hair_render") if isinstance(getattr(A, "meta", None), dict) else None
            cfg = hr if isinstance(hr, dict) else (mc.cfg.get("hair") or {})
            if os.environ.get("JUNO_HAIR_CFG"):
                import json
                cfg = dict(cfg); cfg.update(json.loads(os.environ["JUNO_HAIR_CFG"]))
        cfg = dict(cfg)
        self.v2 = str(cfg.get("model", "")) == "v2"                  # hair-shade: the physically based path
        self.layer = None                                              # HAIR2 (set by _v2_cfg)
        if self.v2:
            self._v2_cfg(cfg)
        self.frac = float(os.environ.get("JUNO_HAIR_FRAC", cfg.get("frac", 1.0)))
        self.seg = int(os.environ.get("JUNO_HAIR_SEG", cfg.get("seg", 1)))
        self.min_px = float(os.environ.get("JUNO_HAIR_MINPX", cfg.get("min_px", 1.5)))
        # render-perf (PERF2): the key shadow map takes a strand SUBSET (shadow_frac of the drawn fraction, widths
        # grow 1/sqrt so the mass/opacity stays) and a coarser point stride (shadow_seg): the map is PCSS-softened, so a
        # uniform subset casts the same shadow; at 1080p the full groom in the 2048^2 map was the largest GPU pass
        self.shadow_frac = float(os.environ.get("JUNO_HAIR_SHADOW_FRAC", cfg.get("shadow_frac", 1.0)))
        self.shadow_seg = int(os.environ.get("JUNO_HAIR_SHADOW_SEG", cfg.get("shadow_seg", 1)))
        # stylised thickness per group kind (a Pixar groom reads as a solid sculpted mass with fine lines, brows as a
        # dense dark shape): multiplies the asset's physical widths
        wk = cfg.get("width_k", {})
        self.width_k = dict(scalp=3.4, fuzz=0.9, brows=2.2, lashes=2.6)   # render look-dev: soft brows, a lash LINE (not wires)
        self.len_k = dict(scalp=1.0, fuzz=1.0, brows=1.0, lashes=0.72)       # strand length about its root
        self.len_k.update({k: float(v) for k, v in (cfg.get("len_k") or {}).items()})
        # render3: shading knobs (x primary R gain, y R lobe width (x roughness), z broad sheen gain, w clump
        # shadow depth): judge r1 "flat helmet, no sheen / clumps"
        self.shade_k = tuple(float(x) for x in cfg.get("shade_k", (1.0, 1.0, 0.0, 0.0)))
        # x per-strand band jitter (1 = r2's streaky bands), y how much R takes the hair's hue (0.65 = r2), z TRT gain, w spare
        self.shade_k2 = tuple(float(x) for x in cfg.get("shade_k2", (1.0, 0.65, 1.0, 0.0)))
        # render7 (mesh.hair.shade_k3): x tip colour pulled toward the root colour, y tip coverage kept (scalp:
        # thin tips go stochastic -> the room shows through = pale washed tips), z root shade floor (the part / exposed
        # roots: 0.30 = old), w flyaway (fuzz) alpha x.  (0, 0, .30, 1) = the old look exactly
        self.shade_k3 = tuple(float(x) for x in cfg.get("shade_k3", (0.0, 0.0, 0.30, 1.0)))
        # render8: per-group colour multiplier over the groom's root/tip colours (hair.color_mul {"scalp": [r,g,b]})
        self.color_mul = {k: np.asarray(v, float)[:3] for k, v in (cfg.get("color_mul") or {}).items()}
        self.dif_k = float(cfg.get("diffuse_k", 0.6))     # hair diffuse vs a skin-like Lambert (light goes to R/TT/TRT)
        if isinstance(wk, dict):
            self.width_k.update({k: float(v) for k, v in wk.items()})
        else:
            self.width_k = {k: v*float(wk) for k, v in self.width_k.items()}
        self.arc_n = float(cfg.get("arc_cells", 1.0))
        self.fuzz_frac = float(cfg.get("fuzz_frac", 0.25))     # flyaways: a few, not a halo (SPEC 3: 1-3 places)
        self.do_shadow = bool(cfg.get("shadow", True))
        # render5 (mesh.hair.mass_ao = [strength, depth_m]): lock-on-lock occlusion -- a strand whose points sit
        # BELOW the groom's outer shell (other locks over it) is darker; the shell = the outermost points per direction
        # from the volume centre.  Per strand (load time, no frame cost); 0 = off.
        self.mass_ao = tuple(float(x) for x in (cfg.get("mass_ao") or (0.0, 0.015)))
        self.deep_mm = float(cfg.get("deep_mm", 6.0))          # self-shadow: depth into the hair where key light is 1/e
        groups_meta = A.meta.get("hair_groups", {}) or {}
        mats = A.meta.get("materials", {}) or {}
        rest_pos = A.mesh["position"].astype(np.float64)
        index = A.mesh["index"].astype(np.int64)
        subs = {s.get("name", s.get("material")): s for s in A.submeshes}
        self.groups = []
        rng = np.random.default_rng(4430)
        for gname, h in A.hair.items():
            gm = dict(DEFAULTS); gm.update(groups_meta.get(gname, {}) or {})
            mp = (mats.get(f"hair_{gname}", {}) or {}).get("params", {}) or {}
            for k in ("color_root", "color_tip", "melanin", "roughness"):
                if k in mp:
                    gm[k] = mp[k]
            pos = np.asarray(h["pos"], np.float64)
            S, P = pos.shape[:2]
            if S == 0:
                continue
            perm = rng.permutation(S)                   # any prefix = a uniform subsample (strand-fraction knob)
            gsurf = gm.get("follow", "bone") == "surface" and "tri" in h
            nch = int(self.v2k.get("sort_chunks", 0)) if self.v2 else 0
            if nch > 0 and not gsurf and S > 4*nch:
                # outer-shell strands first INSIDE each of nch uniform chunks (a chunk prefix stays a uniform
                # subsample): the depth pre-pass then sees mostly front-to-back order -> early depth rejection
                md = self._mass_depth(pos[perm], 0.015)
                bnd = np.linspace(0, S, nch + 1).astype(int)
                perm = np.concatenate([perm[a:b][np.argsort(md[a:b], kind="stable")] for a, b in zip(bnd[:-1], bnd[1:])])
            pos = pos[perm]
            width = np.asarray(h["width"], np.float64)[perm]
            seed = np.asarray(h["seed"], np.float64)[perm]
            clump = np.asarray(h["clump"], np.float64)[perm]
            surface = gm.get("follow", "bone") == "surface" and "tri" in h
            tris = np.zeros((S, 3), np.int64); bary = np.zeros((S, 3))
            if surface:
                sub = subs.get(gm.get("surface_submesh", "head")) or A.submeshes[0]
                tri = np.asarray(h["tri"], np.int64)[perm]
                base = int(sub["first_index"])
                tris = index[base + 3*tri[:, None] + np.arange(3)[None, :]].reshape(S, 3)
                bary = np.asarray(h["bary"], np.float64)[perm]
                v0, v1, v2 = (rest_pos[tris[:, k]] for k in range(3))
                root = bary[:, 0:1]*v0 + bary[:, 1:2]*v1 + bary[:, 2:3]*v2
                F = self._frame(v0, v1, v2)              # (S,3,3) rows e0, n, b
                pos = np.einsum("sij,spj->spi", F, pos - root[:, None, :])
            g = HairGroup()
            g.name, g.S, g.P, g.surface, g.meta = gname, S, P, surface, gm
            bone = gm.get("bone", "head")
            g.bone = A.bone_index.get(bone, A.bone_index.get("head", 0))
            # points texture: (S*P) xyz + u along the strand
            u = np.linspace(0.0, 1.0, P)
            pts = np.concatenate([pos.reshape(S*P, 3), np.tile(u, S)[:, None]], 1).astype(np.float32)
            g.t_pts = self._ftex(_pack(pts))
            # per strand: (w_root, w_tip, seed, clump), (v0, v1, v2, surface), (b0, b1, b2, cluster-noise)
            wr = width[:, 0] if width.ndim == 2 else width
            wt = width[:, 1] if width.ndim == 2 else width*0.3
            st = np.zeros((S, 3, 4), np.float32)
            st[:, 0] = np.stack([wr, wt, seed, clump % 997], 1)
            if self.v2 and not surface:                  # v2 uses .w = the strand's place in its lock (0 core, 1 rim)
                st[:, 0, 3] = self._lock_edge(pos, clump)
            st[:, 1, :3] = tris; st[:, 1, 3] = 0.0
            if not surface and gname in ("scalp", "fuzz") and (self.mass_ao[0] > 0.0 or self.v2):
                st[:, 1, 3] = self._mass_depth(pos, self.mass_ao[1])    # render5
            st[:, 2, :3] = bary; st[:, 2, 3] = (np.sin(clump*12.9898)*43758.5453) % 1.0
            g.t_str = self._ftex(_pack(st.reshape(S*3, 4)))
            # a rough "volume centre" for the mass normal of bone groups (round shading of the hair volume)
            ctr = pos.reshape(-1, 3).mean(0) if not surface else np.zeros(3)
            if not surface and gname in ("scalp", "fuzz"):
                hb = A.bone_index.get("head")
                if hb is not None:                       # the skull centre ~ the head bone + a bit up/back of the roots
                    roots = pos[:, 0, :]
                    ctr = roots.mean(0) - np.array([0.0, 0.02, 0.0])
            g.ctr = ctr
            self.groups.append(g)
        # sort: scalp + fuzz first (big), then brows, lashes last
        self.groups.sort(key=lambda g: (g.surface, g.name))
        self.n_strands = sum(g.S for g in self.groups)
        # posed-vertex copy (vb_posed -> R32F texture) for surface groups
        self.has_surface = any(g.surface for g in self.groups)
        if self.has_surface:
            nfl = mc.N*10
            self.vrows = _rows(nfl)
            self.b_posed_pad = ctx.buffer(reserve=self.vrows*TEXW*4)
            self.t_posed = ctx.texture((TEXW, self.vrows), 1, dtype="f4")
            self.t_posed.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # programs
        src = open(os.path.join(HERE, "mesh.glsl")).read()
        hsrc = open(os.path.join(HERE, "hair.glsl")).read()
        # GL_ARB_sample_shading (core in 4.0; llvmpipe, K620 and every cloud NVIDIA expose it): gl_SampleMask lets each
        # strand pick ITS OWN k of N samples -> layered thin strands accumulate; else alpha-to-coverage
        self.smask = (mc.msaa > 1 and "GL_ARB_sample_shading" in ctx.extensions
                      and str(cfg.get("coverage", "mask")) == "mask")
        def build(sec, lib, earlyz=False, layer=None):
            # mesh.glsl with VS_HAIR defined contributes only its camera helpers (camClip / camRayM)
            ext = "#extension GL_ARB_sample_shading : enable\n#define HAIR_SMASK\n" if (self.smask and sec == "HAIR_FRAG") else ""
            if self.v2:
                ext += "#define HAIR_V2\n"                      # hair-shade
                ext += "#define HAIR_DIAG %d\n" % int(os.environ.get("JUNO_HAIR_DIAG", "0"))
                if os.environ.get("JUNO_HAIR_NODISCARD") == "1":
                    ext += "#define HAIR_NODISCARD\n"
                if earlyz and sec == "HAIR_FRAG":
                    ext += "#extension GL_ARB_shader_image_load_store : enable\n#define HAIR_EARLYZ\n"
                if layer:
                    ext += "#define HAIR_LAYER\n" + ("#define HAIR_COV\n" if layer == "cov" else "")
                    if layer == "cov" and os.environ.get("JUNO_HAIR_COVW0") == "1":
                        ext += "#define HAIR_COVW0\n"
            if sec in ("HAIR_FRAG", "HAIR_CAPF") and getattr(getattr(mc, "mtaa", None), "cls_on", False):
                ext += "#define MESH_TAA_R\n"                  # shade2-taa (gated): the TAA class attachment
            hdr = mc.head + ext + f"#define {sec}\n#define VS_HAIR\n"
            pre = (mc.scene_lib + "\n") if lib else "#define PI 3.14159265359\n"
            return hdr + pre + src + "\n" + hsrc
        ly = "col" if (self.v2 and getattr(self, "layer", None)) else None      # HAIR2 layer: colour programs
        self.p_draw = ctx.program(vertex_shader=build("HAIR_VERT", False, layer=ly),
                                  fragment_shader=build("HAIR_FRAG", True, layer=ly))
        self.p_shadow = ctx.program(vertex_shader=build("HAIR_SVERT", False),
                                    fragment_shader=build("HAIR_SFRAG", False))
        # hair-shade: v2 = a depth-only pre-pass (this program, no early-Z: discard / gl_SampleMask decide which
        # samples get depth) + the shading pass with forced early fragment tests (depth EQUAL, no writes): only the
        # front strand of each sample is shaded (~10x fewer shaded fragments at face framing)
        self.p_pre = None
        self.earlyz = False
        if self.v2 and bool(self.v2k.get("prepass", True)) and os.environ.get("JUNO_HAIR_PREPASS", "1") != "0":
            self.earlyz = ("GL_ARB_shader_image_load_store" in ctx.extensions or ctx.version_code >= 420) \
                and os.environ.get("JUNO_HAIR_EARLYZ", "1") != "0"
            self.p_pre = self.p_draw
            if self.earlyz:
                self.p_draw = ctx.program(vertex_shader=build("HAIR_VERT", False, layer=ly),
                                          fragment_shader=build("HAIR_FRAG", True, earlyz=True, layer=ly))
        self.p_cov = None
        if ly:                                                       # HAIR2: the analytic coverage pass
            self.p_cov = ctx.program(vertex_shader=build("HAIR_VERT", False, layer="cov"),
                                     fragment_shader=build("HAIR_FRAG", True, layer="cov"))
            self.vao_cov = ctx.vertex_array(self.p_cov, [])
            self.p_lcomp = ctx.program(vertex_shader=mc.head + _LCOMP_VS, fragment_shader=mc.head + _LCOMP_FS)
            self.vao_lcomp = ctx.vertex_array(self.p_lcomp, [])
            self._lt = None
        self.p_dom = None
        if self.v2 and self.dom_on and self.do_shadow:              # hair-shade: deep opacity map
            self.p_dom = ctx.program(vertex_shader=build("HAIR_DVERT", False), fragment_shader=build("HAIR_DFRAG", False))
            dres = max(64, int(mc.shadow_res*self.dom_scale))
            self.t_dom = ctx.texture((dres, dres), 4, dtype="f2")
            self.t_dom.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.t_dom.repeat_x = self.t_dom.repeat_y = False
            self.f_dom = ctx.framebuffer(color_attachments=[self.t_dom])
            self.vao_dom = ctx.vertex_array(self.p_dom, [])
            self.dom_res = dres
        for p in [q for q in (self.p_draw, self.p_shadow, self.p_dom, self.p_pre, self.p_cov) if q is not None]:
            for nm, unit in (("tPts", 5), ("tStr", 6), ("tPosed", 7), ("uShadow", 4), ("uDom", 8)):
                if nm in p:
                    p[nm].value = unit
        self.cap = None
        cap_cfg = cfg.get("cap", {})
        if self.v2 and isinstance(cap_cfg, dict):          # hair-shade: v2 cap = a soft, wide under-paint
            cap_cfg = dict({"scalp_alpha": 0.85, "scalp_sigma_mm": 6.0}, **cap_cfg)
        if os.environ.get("JUNO_HAIR_CAP") == "0":
            cap_cfg = False
        if cap_cfg is not False and self.groups:
            try:
                self._bake_cap(A, dict(cap_cfg or {}), build)
            except Exception as e:                       # never lose the strands over the cap
                print(f"[hair] cap bake failed: {e}", flush=True)
                self.cap = None
        self.vao_draw = ctx.vertex_array(self.p_draw, [])
        self.vao_pre = ctx.vertex_array(self.p_pre, []) if (self.p_pre is not None and self.p_pre is not self.p_draw) else None
        self.vao_shadow = ctx.vertex_array(self.p_shadow, [])
        self.stats = {}

    # ------------------------------------------------------------------ hair-shade: v2 knobs
    V2_DEFAULTS = dict(
        eumelanin=3.0, pheomelanin=0.25,      # Chiang 2016 sigma_a = eu*(.419,.697,1.37) + pheo*(.187,.4,1.05) (per unit radius)
        tip_melanin=0.9,                      # tip melanin x (sun-bleached, lighter warm tips)
        melanin_jitter=(0.12, 0.3),           # per strand, per clump (+- fraction)
        root_mul=(1.0, 1.0, 1.0), tip_mul=(0.8, 0.8, 0.8),   # colour multipliers over the groom's root / tip colour
        alpha_deg=3.0,                        # cuticle tilt: R at -alpha, TT +alpha/2, TRT +1.5 alpha
        beta=0.30,                            # longitudinal roughness of R (TT x .5, TRT x 2 in variance terms, Karis)
        rough_jitter=0.35,                    # per-strand roughness +- fraction
        shift_jitter_deg=(1.5, 4.0, 1.2),     # per strand, per clump, along-strand wave (degrees)
        gain=(5.0, 0.5, 1.5, 3.0),            # R, TT, TRT, multiple scattering
        r_tint=0.45,                          # share of the R lobe that takes the hair's hue (0 = white light colour)
        ms_pow=1.0, ms_vol=0.6,  ms_trans=0.5, # MS albedo = colour^ms_pow; mass-normal share; key transmittance exponent
        root_occ=(0.45, 0.25),                # occlusion floor at the root, u where it reaches 1
        mass_occ=0.55,                        # lock-under-lock (load-time mass depth) occlusion strength
        amb=1.0, amb_spec=0.5, fill_area=0.25, # ambient diffuse gain, ambient R sheen, fill light extra lobe width
        tt_sat=0.55, ms_sat=0.4,              # TT/TRT colour saturation; MS saturation in key shadow (Karis = 1: copper)
        fill_r=0.15,                          # R gain of fill / rim relative to the key (broad lights wash grey otherwise)
        lock_var=(0.3, 0.6, 0.45),            # per LOCK (clump id): MS brightness +-, R gain +-, edge (rim strand) occlusion
        r_var=0.6,                            # per-strand highlight brightness +- (fine streaks, not a smooth band)
        aa=1.0,                               # specular AA: sub-pixel strands widen their lobes (0 = off)
        tip_fade=(0.92, 0.85),                # u where tip alpha starts falling, alpha left at the very tip
        dom=True, dom_scale=0.5,              # deep opacity map on, resolution x the key shadow map
        dom_mm=(1.5, 4.0, 10.0, 30.0),        # slab boundaries behind the first occluder (mm)
        dom_k=0.55, dom_frac=0.5,             # optical depth per unit strand coverage; strand subset in the DOM pass
        dom_skin=0.0,                         # (reserved) hair -> skin DOM transmittance
        sort_chunks=int(os.environ.get("JUNO_HAIR_SORT", "0")),   # outer strands first within N uniform chunks (0 = off)
        shadow_keep=4.0,                      # key map: strand kept with p = coverage x this (soft hairline shadow on skin; 0 = binary)
        prepass=True,                         # depth pre-pass of the strands, then shade only the front samples
    )

    # HAIR2 layer knobs: kappa (opacity = 1 - exp(-kappa x summed true coverage)), followers (extra strands per
    # strand in the coverage pass, spread_mm [root, tip], spread_n share across the lock layer, curl rad along u),
    # len_var [per strand, per lock, taper] (fractions of u), cov_k per kind, col_frac (strand fraction in the colour pass,
    # x the lod fraction), col_minpx (colour ribbon width, px), width_k per kind (overrides hair_render.width_k in the layer)
    LAYER_DEFAULTS = dict(kappa=1.0, followers=1, spread_mm=(0.8, 5.0), spread_n=0.4, curl=0.0,
                          len_var=(0.10, 0.08, 0.10), cov_k={}, col_frac=1.0, col_minpx=1.5, width_k={},
                          tent_px=1.0, cov_mass=2.0, fill=True, taper_u=1.0)

    def _v2_cfg(self, cfg):
        k = dict(self.V2_DEFAULTS)
        k.update({a: b for a, b in cfg.items() if a in k})
        self.v2k = k
        # follow + lag: {"bone": "spine_03", "hang_k": .9, "u0": .3, "lag_s": .12, "lag_k": .5, "lag_max_deg": 4}
        fo = cfg.get("follow")
        self.follow = None
        if isinstance(fo, dict) and fo.get("on", True):
            self.follow = dict(dict(bone="spine_03", hang_k=0.9, u0=0.3, lag_s=0.12, lag_k=0.5, lag_max_deg=4.0), **fo)
        self._lag = None
        fz = cfg.get("far", {})
        self.far = None
        if isinstance(fz, dict) and fz.get("on", True):
            self.far = dict(dict(ppm=(1.0, 0.45), min_px=2.5, width_k=1.8, frac=0.5, smooth=1.0, edge_w=4.0), **fz)
        self._far = 0.0
        lf = cfg.get("lod_frac")
        self.lod_frac = [tuple(map(float, x)) for x in lf] if isinstance(lf, (list, tuple)) and lf else None
        # HAIR2: the analytic-coverage hair layer (hair_render.layer; absent / {"on": false} = the MSAA path)
        ly = cfg.get("layer")
        self.layer = None
        if isinstance(ly, dict) and ly.get("on", True) and os.environ.get("JUNO_HAIR_LAYER", "1") != "0":
            self.layer = dict(self.LAYER_DEFAULTS, **ly)
        self.dom_on = bool(k["dom"]) and os.environ.get("JUNO_HAIR_DOM", "1") != "0"
        self.dom_scale = float(k["dom_scale"])
        ma = cfg.get("mass_ao")                          # hair_render mass_ao [strength, depth_m]: strength > 0 wins
        if isinstance(ma, (list, tuple)) and len(ma) >= 1 and float(ma[0]) > 0.0 and "mass_occ" not in cfg:
            k["mass_occ"] = float(ma[0])

    def _follow_update(self, rig):
        """per frame: the follow bone's skin matrix + a critically damped lagged copy of the head's rotation (about the
        head pivot), clamped to lag_max_deg; dt = 1/FPS per rendered frame (deterministic for a frame sequence)"""
        f = self.follow
        if f is None or not self.groups:
            self._fol = None
            return
        A = self.mc.A
        hb = A.bone_index.get("head", 0)
        fb = A.bone_index.get(f["bone"], hb)
        Mh = self._skin(rig, hb)
        Mf = self._skin(rig, fb)
        if not hasattr(self, "_fol_y"):                  # rest head joint height -> the band where strands hang
            yh = float(np.linalg.inv(np.asarray(self.mc.inv_bind[hb], np.float64))[1, 3])
            self._fol_y = (yh + float(f.get("y_hi", 0.0)), yh + float(f.get("y_lo", -0.10)))
        R = Mh[:3, :3]/max(abs(np.linalg.det(Mh[:3, :3]))**(1.0/3.0), 1e-9)
        q = _quat(R)
        try:
            from anim.schema import FPS
        except Exception:
            FPS = 24.0
        dt = 1.0/float(FPS)
        if self._lag is None:
            self._lag = [q.copy(), np.zeros(4)]
        x, v = self._lag
        if np.dot(x, q) < 0.0:
            x = -x; v = -v
        w = 2.0/max(float(f["lag_s"]), 1e-3)
        d = x - q
        e = np.exp(-w*dt)
        tmp = v + w*d
        d = (d + tmp*dt)*e
        v = (v - w*tmp*dt)*e
        x = q + d
        x /= max(np.linalg.norm(x), 1e-12)
        ang = 2.0*np.degrees(np.arccos(min(1.0, abs(float(np.dot(x, q))))))
        amax = float(f["lag_max_deg"])
        if ang > amax:                                   # clamp: slerp toward the head
            x = _slerp(q, x, amax/ang); v *= amax/ang
        self._lag = [x, v]
        D = _qmat(x) @ R.T                               # lag delta rotation (world)
        c = np.asarray(rig.bones[hb], np.float64)[:3, 3]
        T = np.eye(4); T[:3, :3] = D; T[:3, 3] = c - D @ c
        # hang: the head's rotation relative to the follow bone split swing / twist about the vertical; the hanging
        # target keeps the twist (a head turn still carries the hair round) and drops the swing (nod / tilt)
        Rf = Mf[:3, :3]/max(abs(np.linalg.det(Mf[:3, :3]))**(1.0/3.0), 1e-9)
        qr = _quat(Rf.T @ R)
        up = Rf.T @ np.array([0.0, 1.0, 0.0])
        vp = np.dot(qr[1:], up)*up
        tw = np.array([qr[0], *vp])
        tw = tw/np.linalg.norm(tw) if np.linalg.norm(tw) > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        Dh = (Rf @ _qmat(tw)) @ R.T                      # head -> hanging target (world, about the head pivot)
        H = np.eye(4); H[:3, :3] = Dh; H[:3, 3] = c - Dh @ c
        self._fol = (H @ Mh, T @ Mh)

    def _v2_uniforms(self, p):
        k, mc = self.v2k, self.mc
        f = lambda v: tuple(float(x) for x in v)
        mc._set(p, "uV2Mel", (float(k["eumelanin"]), float(k["pheomelanin"]), float(k["tip_melanin"]), 0.0))
        mc._set(p, "uV2MelJ", (float(k["melanin_jitter"][0]), float(k["melanin_jitter"][1])))
        d2r = np.pi/180.0
        sj = k["shift_jitter_deg"]
        mc._set(p, "uV2Lobe", (float(k["alpha_deg"])*d2r, float(k["beta"]), float(k["rough_jitter"]), float(k["fill_area"])))
        mc._set(p, "uV2Jit", (float(sj[0])*d2r, float(sj[1])*d2r, float(sj[2])*d2r, 0.0))
        mc._set(p, "uV2Gain", f(k["gain"]))
        mc._set(p, "uV2RVar", float(k["r_var"]))
        mc._set(p, "uV2Lock", tuple(float(x) for x in k["lock_var"]))
        mc._set(p, "uV2FillR", float(k["fill_r"]))
        mc._set(p, "uV2Far", float(getattr(self, "_far", 0.0))*float((self.far or {}).get("smooth", 0.0)))
        mc._set(p, "uV2Sat", (float(k["tt_sat"]), float(k["ms_sat"])))
        mc._set(p, "uV2Misc", (float(k["r_tint"]), float(k["ms_pow"]), float(k["ms_vol"]), float(k["ms_trans"])))
        mc._set(p, "uV2Occ", (float(k["root_occ"][0]), float(k["root_occ"][1]), float(k["mass_occ"]), float(k["amb"])))
        mc._set(p, "uV2Tip", (float(k["tip_fade"][0]), float(k["tip_fade"][1]), float(k["amb_spec"]), float(k["aa"])))
        dm = [float(x)*1e-3*self._range_k for x in k["dom_mm"]]
        mc._set(p, "uDomZ", tuple(dm))
        mc._set(p, "uDomK", float(k["dom_k"]) if self.p_dom is not None else -1.0)
        mc._set(p, "uDomRes", float(getattr(self, "dom_res", 1)))

    # ------------------------------------------------------------------ the hair cap (root density -> skin uv)
    CAP_KIND = {"scalp": (1.0, 4.0), "brows": (0.55, 1.0), "lashes": (0.9, 0.45)}   # alpha max, sigma (mm)

    def _bake_cap(self, A, ccfg, build):
        import cv2
        mc, ctx = self.mc, self.ctx
        res = int(ccfg.get("res", 2048 if A.lod.get("name", "lod0") == "lod0" else 1024))
        subs = {s.get("name", s.get("material")): s for s in A.submeshes}
        sub = subs.get("head")
        if sub is None:
            return
        idx = A.mesh["index"][sub["first_index"]:sub["first_index"] + sub["index_count"]].astype(np.int64).reshape(-1, 3)
        P = A.mesh["position"].astype(np.float64); UV = A.mesh["uv0"].astype(np.float64)
        e1, e2 = P[idx[:, 1]] - P[idx[:, 0]], P[idx[:, 2]] - P[idx[:, 0]]
        u1, u2 = UV[idx[:, 1]] - UV[idx[:, 0]], UV[idx[:, 2]] - UV[idx[:, 0]]
        ap = 0.5*np.linalg.norm(np.cross(e1, e2), axis=1)
        au = 0.5*np.abs(u1[:, 0]*u2[:, 1] - u1[:, 1]*u2[:, 0])
        ok = (ap > 1e-12) & (au > 1e-12)
        k_uv = float(np.median(np.sqrt(au[ok]/ap[ok])))           # uv units per metre (median over the head)
        px_mm = k_uv*res*1e-3
        out_c = np.zeros((res, res, 3), np.float32); out_a = np.zeros((res, res), np.float32)
        kinds = {}
        for g in self.groups:
            kind = "brows" if g.name.startswith("brow") else ("lashes" if g.name.startswith("lash") else g.name)
            if kind in self.CAP_KIND:
                kinds.setdefault(kind, []).append(g)
        for kind in ("scalp", "brows", "lashes"):
            if kind not in kinds:
                continue
            amax, sig = self.CAP_KIND[kind]
            amax = float(ccfg.get(f"{kind}_alpha", amax)); sig = float(ccfg.get(f"{kind}_sigma_mm", sig))
            D = np.zeros((res, res), np.float32)
            for g in kinds[kind]:
                ruv = np.asarray(A.hair[g.name]["root_uv"], np.float64)
                x = np.clip((ruv[:, 0]*res).astype(np.int64), 0, res - 1)
                y = np.clip(((1.0 - ruv[:, 1])*res).astype(np.int64), 0, res - 1)   # image rows top-down
                np.add.at(D, (y, x), 1.0)
            sp = max(sig*px_mm, 0.7)
            D = cv2.GaussianBlur(D, (0, 0), sp)
            nz = D[D > 1e-6]
            if nz.size == 0:
                continue
            ref = float(np.percentile(nz, 60))
            t = np.clip((D/max(ref, 1e-9) - 0.08)/0.85, 0.0, 1.0)
            a = t*t*(3.0 - 2.0*t)*amax                  # eroded + soft: the strands overhang the cap's edge
            a = cv2.GaussianBlur(a, (0, 0), max(sp*0.5, 0.5))
            col = np.asarray(kinds[kind][0].meta["color_root"][:3], np.float32)
            if kind == "brows":
                col = 0.7*col + 0.3*np.asarray(kinds[kind][0].meta["color_tip"][:3], np.float32)
            out_c = out_c*(1.0 - a[..., None]) + col[None, None, :]*a[..., None]
            out_a = out_a*(1.0 - a) + a
        img = np.concatenate([out_c, out_a[..., None]], 2)[::-1]          # GL row 0 = v 0
        self.t_cap = ctx.texture((res, res), 4, np.ascontiguousarray(img, np.float32).astype(np.float16).tobytes(),
                                 dtype="f2")
        self.t_cap.build_mipmaps()
        self.t_cap.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        self.p_cap = ctx.program(vertex_shader=build("HAIR_CAPV", False), fragment_shader=build("HAIR_CAPF", True))
        for nm, unit in (("tCap", 5), ("uShadow", 4), ("uDom", 8)):
            if nm in self.p_cap:
                self.p_cap[nm].value = unit
        names = [n for n in ("aP", "aN") if n in self.p_cap]
        fmt = " ".join(["3f" if "aP" in self.p_cap else "12x", "3f" if "aN" in self.p_cap else "12x", "16x"])
        cont = [(mc.vb_posed, fmt, *names)]
        if "aUV" in self.p_cap:
            cont.append((mc.vb_uv, "2f", "aUV"))
        self.vao_cap = ctx.vertex_array(self.p_cap, cont, index_buffer=mc.ib, index_element_size=4)
        self.cap = sub
        self.cap_k = float(ccfg.get("strength", 1.0))
        self.cap_stats = dict(res=res, uv_per_m=round(k_uv, 2), px_per_mm=round(px_mm, 2))

    def _draw_cap(self, cam, nf, lights):
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        p = self.p_cap
        mc._cam(p, cam, nf)
        L = lights
        for k, val in (("uKeyDir", L["key_dir"]), ("uKeyCol", L["key_col"]), ("uFillCol", L["fill_col"])):
            mc._set(p, k, tuple(float(x) for x in val))
        for k in ("uKeyGain", "uAmbGain", "uCharK"):
            mc._set(p, k, float(L[k]))
        p["uLightVP"].write(np.ascontiguousarray(mc.light_vp.T, np.float32).tobytes())
        mc._set(p, "uShadowZ", self._shadow_z)
        mc._set(p, "uDeep", float(self.deep_mm*1e-3) if self.do_shadow else -1.0)
        mc._set(p, "uCapK", float(self.cap_k))
        if getattr(mc, "mtaa", None) is not None and mc.mtaa.cls_on:      # shade2-taa (gated)
            mc._set(p, "uTaaR", mc.mtaa.cls_vec("hair"))
        if self.v2:                                                     # hair-shade: the cap reads the DOM
            mc._set(p, "uShadowZ", getattr(self, "_dom_z", self._shadow_z))
            self._v2_uniforms(p)
            if self.p_dom is not None:
                self.t_dom.use(8)
        self.t_cap.use(5); mc.t_shadow.use(4)
        ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
        ctx.depth_mask = False; ctx.disable(mgl.CULL_FACE)
        self.vao_cap.render(mgl.TRIANGLES, vertices=self.cap["index_count"], first=self.cap["first_index"])
        ctx.disable(mgl.BLEND); ctx.depth_mask = True

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mass_depth(pos, depth_m, nb=(18, 36)):
        """render5: per strand 0..1 = how far below the groom's outer shell its outer 70 % lies (lock-on-lock)."""
        S, P = pos.shape[:2]
        pts = pos.reshape(-1, 3)
        c = pos[:, 0, :].mean(0) - np.array([0.0, 0.02, 0.0])
        d = pts - c
        r = np.linalg.norm(d, axis=1) + 1e-9
        th = np.clip((np.arccos(np.clip(d[:, 1]/r, -1, 1))/np.pi*nb[0]).astype(int), 0, nb[0] - 1)
        ph = np.clip(((np.arctan2(d[:, 2], d[:, 0]) + np.pi)/(2*np.pi)*nb[1]).astype(int), 0, nb[1] - 1)
        b = th*nb[1] + ph
        # the shell = the 85th percentile radius per direction bin (flyaways / the bun's tip are not the surface)
        nbin = nb[0]*nb[1]
        o = np.lexsort((r, b)); bs = b[o]; rs = r[o]
        cnt = np.bincount(bs, minlength=nbin); st0 = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        shell = np.zeros(nbin)
        ok = cnt > 0
        shell[ok] = rs[st0[ok] + (0.85*(cnt[ok] - 1)).astype(int)]
        dep = np.clip((shell[b] - r)/max(depth_m, 1e-4), 0.0, 1.0).reshape(S, P)
        k0 = int(P*0.3)
        return dep[:, k0:].mean(1).astype(np.float32)

    @staticmethod
    def _lock_edge(pos, clump):
        """per strand 0..1 = distance of its outer half from its lock's (clump id) centroid line, over the lock's
        80th-percentile radius -> rim strands of a lock (the dark gaps between locks) vs its core"""
        S, P = pos.shape[:2]
        ks = [int(P*f) for f in (0.5, 0.7, 0.9)]
        cid = np.unique(clump.astype(np.int64), return_inverse=True)[1]
        nc = int(cid.max()) + 1
        cnt = np.maximum(np.bincount(cid, minlength=nc), 1)
        e = np.zeros(S)
        for k in ks:
            q = pos[:, min(k, P - 1), :]
            c = np.stack([np.bincount(cid, q[:, j], nc)/cnt for j in range(3)], 1)
            e += np.linalg.norm(q - c[cid], axis=1)
        e /= len(ks)
        o = np.lexsort((e, cid)); r80 = np.zeros(nc)
        st0 = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        cn = np.bincount(cid, minlength=nc)
        ok = cn > 0
        r80[ok] = e[o][st0[ok] + (0.8*(cn[ok] - 1)).astype(int)]
        return np.clip(e/np.maximum(r80[cid], 1e-5), 0.0, 1.5).astype(np.float32)/1.5

    @staticmethod
    def _frame(v0, v1, v2):
        e0 = v1 - v0
        e0 /= np.maximum(np.linalg.norm(e0, axis=-1, keepdims=True), 1e-12)
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        b = np.cross(n, e0)
        return np.stack([e0, n, b], 1)

    def _ftex(self, a4):
        t = self.ctx.texture((TEXW, a4.shape[0]//TEXW), 4, np.ascontiguousarray(a4, np.float32).tobytes(), dtype="f4")
        t.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        return t

    def _skin(self, rig, b):
        return np.asarray(rig.bones[b], np.float64) @ self.mc.inv_bind[b]

    def _common(self, p, rig, g, scale):
        M = self._skin(rig, g.bone)
        sc = abs(np.linalg.det(M[:3, :3]))**(1.0/3.0)
        p["uBoneM"].write(np.ascontiguousarray(M.T, np.float32).tobytes())
        self.mc._set(p, "uScale", float(sc))
        if self.v2:                                                 # follow + lag (bone groups)
            fo = self._fol if (self.follow is not None and not g.surface and getattr(self, "_fol", None)) else None
            if fo is not None and "uFollowM" in p:
                p["uFollowM"].write(np.ascontiguousarray(fo[0].T, np.float32).tobytes())
                p["uLagM"].write(np.ascontiguousarray(fo[1].T, np.float32).tobytes())
                f = self.follow
                self.mc._set(p, "uFollow", (float(f["hang_k"]), float(f["u0"]), float(f["lag_k"]), 1.0))
                self.mc._set(p, "uFollowY", self._fol_y)
            else:
                self.mc._set(p, "uFollow", (0.0, 0.0, 0.0, 0.0))
        self.mc._set(p, "uP", int(g.P))
        self.mc._set(p, "uSeg", int(max(1, self.seg)))
        self.mc._set(p, "uSurface", 1 if g.surface else 0)
        kind = "brows" if g.name.startswith("brow") else ("lashes" if g.name.startswith("lash") else
                                                           ("fuzz" if g.name == "fuzz" else "scalp"))
        fk = 1.0 if g.surface else 1.0/np.sqrt(max(self.frac, 0.05))
        wk0 = self.width_k.get(kind, 1.0)
        if getattr(self, "_in_layer", False):                        # HAIR2: true widths, no subset widening
            fk = 1.0
            wk0 = float((self.layer.get("width_k") or {}).get(kind, 1.0))
        wkf = 1.0
        if self.v2 and self.far is not None and not g.surface:      # far field: wider strands -> a solid mass
            wkf = 1.0 + self._far*(float(self.far["width_k"]) - 1.0)
        self.mc._set(p, "uWidthK", float(wk0*fk*scale*wkf))
        if self.v2:
            self.mc._set(p, "uFarEdge", float(self._far*float(self.far.get("edge_w", 0.0))) if self.far is not None else 0.0)
        self.mc._set(p, "uLenK", float(self.len_k.get(kind, 1.0)))
        ctr = (M @ np.append(g.ctr, 1.0))[:3]
        self.mc._set(p, "uCtr", tuple(float(x) for x in ctr))
        g.t_pts.use(5); g.t_str.use(6)
        if self.has_surface:
            self.t_posed.use(7)
        return self._common_count(g)

    def _npts(self, g):
        return (g.P - 1)//max(1, self.seg) + 1

    def _sync_posed(self):
        if self.has_surface:
            self.ctx.copy_buffer(self.b_posed_pad, self.mc.vb_posed)
            self.t_posed.write(self.b_posed_pad)

    # ------------------------------------------------------------------ passes
    def shadow(self, rig, light_vp, key_dir, texel_world):
        """into the CURRENT framebuffer (the key shadow map): ribbons facing the light, >= 1 shadow texel wide"""
        if self.v2 and getattr(self, "_lodf", None) is not None:
            f0 = self.frac
            self.frac = min(self.frac, self._lodf)                  # last frame's camera-distance fraction
            try:
                return self._shadow(rig, light_vp, key_dir, texel_world)
            finally:
                self.frac = f0
        return self._shadow(rig, light_vp, key_dir, texel_world)

    def _shadow(self, rig, light_vp, key_dir, texel_world):
        if self.v2:
            self._follow_update(rig)                                # once per frame (the shadow runs first)
        if not self.do_shadow or not self.groups:
            return
        ctx, mgl = self.ctx, self.mgl
        self._sync_posed()
        p = self.p_shadow
        p["uLightVP"].write(np.ascontiguousarray(light_vp.T, np.float32).tobytes())
        self.mc._set(p, "uLightDir", tuple(float(x) for x in key_dir))
        self.mc._set(p, "uTexelW", float(texel_world))
        if self.v2:
            self.mc._set(p, "uShadowKeep", float(self.v2k["shadow_keep"]))
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "<"
        frac0, seg0 = self.frac, self.seg                 # render-perf: the shadow subset (restored below)
        self.frac = max(0.05, min(1.0, self.frac*max(0.05, min(1.0, self.shadow_frac))))
        self.seg = max(seg0, int(self.shadow_seg))
        try:
            for g in self.groups:
                if g.name.startswith("lashes"):
                    continue                              # lashes would only darken the eye; the lid line does that
                if self.v2 and g.name == "fuzz" and float(self.v2k["shadow_keep"]) > 0.0:
                    continue                              # flyaways cast no key-map shadow (blotches on skin)
                n = self._common(p, rig, g, 1.0)
                self.vao_shadow.render(mgl.TRIANGLE_STRIP, vertices=2*self._npts(g), instances=n)
            if self.p_dom is not None:
                self._dom_pass(rig, light_vp, key_dir, texel_world)
        finally:
            self.frac, self.seg = frac0, seg0

    def _dom_pass(self, rig, light_vp, key_dir, texel_world):
        """hair-shade: deep opacity map -- the strand subset again, additively accumulating coverage into 4
        cumulative depth slabs behind the first occluder (the key map just rendered); read by keyTransV2()"""
        ctx, mgl, mc = self.ctx, self.mgl, self.mc
        p = self.p_dom
        rng = float(getattr(mc.light, "range", 4.0*float(rig.bound[3]))) if getattr(mc, "light", None) is not None \
            else 4.0*float(rig.bound[3])
        self._range_k = 1.0
        self._dom_z = (rng, 1.0/mc.shadow_res)
        frac_draw = self.frac                              # already the shadow subset here
        self.frac = max(0.05, min(1.0, frac_draw*float(self.v2k["dom_frac"])/max(float(self.shadow_frac), 1e-3)))
        try:
            self.f_dom.use(); ctx.viewport = (0, 0, self.dom_res, self.dom_res)
            self.f_dom.clear(0.0, 0.0, 0.0, 0.0)
            ctx.disable(mgl.DEPTH_TEST); ctx.depth_mask = False
            ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE)
            p["uLightVP"].write(np.ascontiguousarray(light_vp.T, np.float32).tobytes())
            self.mc._set(p, "uLightDir", tuple(float(x) for x in key_dir))
            dtex = float(texel_world)*mc.shadow_res/self.dom_res
            self.mc._set(p, "uTexelW", dtex)
            self.mc._set(p, "uShadowZ", self._dom_z)
            self.mc._set(p, "uDomZ", tuple(float(x)*1e-3 for x in self.v2k["dom_mm"]))
            self.mc._set(p, "uDomRes", float(self.dom_res))
            # strands drawn: a uniform fraction of the full groom; their summed coverage x 1/(fraction x width growth)
            fk = 1.0/np.sqrt(max(self.frac, 0.05))
            self.mc._set(p, "uDomA", float(1.0/(max(self.frac, 0.05)*fk)))
            mc.t_shadow.use(4)
            for g in self.groups:
                if g.surface:
                    continue
                n = self._common(p, rig, g, 1.0)
                self.vao_dom.render(mgl.TRIANGLE_STRIP, vertices=2*self._npts(g), instances=n)
            if os.environ.get("JUNO_HAIR_DOM_DUMP"):            # debug: the DOM as .npy (res, res, 4)
                a = np.frombuffer(self.t_dom.read(), np.float16).reshape(self.dom_res, self.dom_res, 4)
                np.save(os.environ["JUNO_HAIR_DOM_DUMP"], a.astype(np.float32))
        finally:
            ctx.disable(mgl.BLEND); ctx.depth_mask = True; ctx.enable(mgl.DEPTH_TEST)
            mc.f_shadow.use(); ctx.viewport = (0, 0, mc.shadow_res, mc.shadow_res)

    def draw(self, rig, cam, nf, lights, uniforms_from=None):
        """into the CURRENT framebuffer (the MSAA forward target, depth prefilled + opaque submeshes drawn)"""
        if not self.groups:
            return
        ctx, mgl = self.ctx, self.mgl
        from character import GL_SAMPLE_ALPHA_TO_COVERAGE, GL_SAMPLE_ALPHA_TO_ONE, _gl_enable
        mc = self.mc
        if not self.do_shadow:
            self._sync_posed()
            if self.v2:
                self._follow_update(rig)
        r = float(rig.bound[3])
        self._shadow_z = (4.0*r, 1.0/mc.shadow_res)
        frac_saved = self.frac
        if self.v2:                                                     # camera-distance strand fraction / far field
            self._lod_frac(rig, cam)
            self.frac = min(self.frac, self._lodf)
        # hair-shade: v2 with the pre-pass = two programs (depth-only pre-pass, early-Z shading); old = one
        progs = [self.p_draw]
        if self.v2 and self.p_pre is not None and self.p_pre is not self.p_draw:
            progs = [self.p_pre, self.p_draw]
        for p in progs:
            self._draw_uniforms(p, cam, nf, lights)
        mc.t_shadow.use(4)
        if self.v2 and self.p_dom is not None:                        # hair-shade
            self.t_dom.use(8)
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "<="
        if self.cap is not None:
            self._draw_cap(cam, nf, lights)
            mc.t_shadow.use(4)
        st = getattr(self, "_stamp", None)                             # bench: GPU timestamps per hair pass
        if st:
            st("hair_cap")
        ctx.depth_mask = True
        ctx.disable(mgl.BLEND); ctx.disable(mgl.CULL_FACE)
        a2c = mc.msaa > 1 and (self.smask or _gl_enable(GL_SAMPLE_ALPHA_TO_COVERAGE))
        if a2c and not self.smask:
            _gl_enable(GL_SAMPLE_ALPHA_TO_ONE)
        for p in progs:
            mc._set(p, "uA2C", 1 if a2c else 0)
            mc._set(p, "uDebug", int(os.environ.get("JUNO_HAIR_DEBUG", "0")))
            mc._set(p, "uSamples", int(mc.msaa if a2c else 1))
            mc._set(p, "uArcN", float(self.arc_n))
        pre = self.v2 and self.p_pre is not None
        passes = [(1, self.p_pre, self.vao_pre or self.vao_draw), (0, self.p_draw, self.vao_draw)] if pre \
            else [(0, self.p_draw, self.vao_draw)]
        self._in_layer = self.layer is not None
        if self.layer is not None:                                      # HAIR2: coverage layer, then colour
            self._layer_cov(rig, cam, nf, lights)                       # into the hair's own MSAA target
            if st:
                st("hair_cov")
            self._lt["f_hc"].use(); ctx.viewport = (0, 0, mc.iw, mc.ih)
            ctx.enable(mgl.DEPTH_TEST); ctx.disable(mgl.BLEND); ctx.disable(mgl.CULL_FACE)
        for ipass, p, vao in passes:                                    # hair-shade: v2 depth pre-pass
            if self.v2:
                mc._set(p, "uPrepass", ipass)
                if ipass == 1:
                    ctx.depth_func = "<="; ctx.depth_mask = True
                    self._color_mask(False)
                elif pre:
                    self._color_mask(True)
                    if getattr(mc, "mtaa", None) is not None:   # shade2-taa (gated): glColorMask(all) re-enabled the
                        mc.mtaa.hair_masks()                     # velocity (+ albedo) buffers the strands never write
                    ctx.depth_func = "==" if self.earlyz else "<="; ctx.depth_mask = False
            for g in (self.groups if os.environ.get("JUNO_HAIR_STRANDS") != "0" else []):
                self._group_draw(p, vao, rig, g)
            if st and ipass == 1:
                st("hair_prepass")
        if pre:
            self._color_mask(True); ctx.depth_mask = True
        if self.layer is not None:                                      # HAIR2: back to the forward target
            mc.f_ms.use(); ctx.viewport = (0, 0, mc.iw, mc.ih)
        self._in_layer = False
        if a2c and not self.smask:
            _gl_enable(GL_SAMPLE_ALPHA_TO_COVERAGE, False); _gl_enable(GL_SAMPLE_ALPHA_TO_ONE, False)
        self.stats = dict(strands=sum(self._common_count(g) for g in self.groups))
        self.frac = frac_saved

    # ------------------------------------------------------------------ HAIR2: analytic coverage layer
    def _layer_targets(self):
        """hair-own targets at the forward size: MSAA colour + distance + depth (colour pass), single-sample opaque depth
        + R16F coverage (coverage pass), resolve textures; re-made when the forward target changes (quality tiers)"""
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        key = (mc.iw, mc.ih, int(mc.rb_z.samples), id(mc.t_mc))
        if self._lt is not None and self._lt["key"] == key:
            return self._lt
        if self._lt is not None:
            for k, v in self._lt.items():
                if k != "key" and hasattr(v, "release"):
                    v.release()
        w, h, ns = mc.iw, mc.ih, int(mc.rb_z.samples)
        T = dict(key=key)
        T["rb_hc"] = ctx.renderbuffer((w, h), 4, samples=ns, dtype="f2")
        T["rb_hd"] = ctx.renderbuffer((w, h), 4, samples=ns, dtype="f2")
        T["rb_hz"] = ctx.depth_renderbuffer((w, h), samples=ns)
        T["f_hc"] = ctx.framebuffer([T["rb_hc"], T["rb_hd"]], T["rb_hz"])
        T["t_dop"] = ctx.depth_texture((w, h))
        T["t_cov"] = ctx.texture((w, h), 1, dtype="f2")
        T["f_cov"] = ctx.framebuffer([T["t_cov"]], T["t_dop"])
        T["f_cov"].depth_mask = False                    # moderngl: the depth mask is per framebuffer (ctx.depth_mask is a no-op)
        T["t_hc"] = ctx.texture((w, h), 4, dtype="f2"); T["t_hd"] = ctx.texture((w, h), 4, dtype="f2")
        T["f_rhc"] = ctx.framebuffer([T["t_hc"]]); T["f_rhd"] = ctx.framebuffer([T["t_hd"]])
        T["f_rhc_ms"] = ctx.framebuffer([T["rb_hc"]]); T["f_rhd_ms"] = ctx.framebuffer([T["rb_hd"]])
        T["f_out"] = ctx.framebuffer([mc.t_mc, mc.t_md])
        for t in ("t_cov", "t_hc", "t_hd"):
            T[t].filter = (mgl.NEAREST, mgl.NEAREST)
        self._lt = T
        return T

    def _gl(self):
        if not hasattr(self, "_glx"):
            import ctypes
            gl = None
            for name in ("libOpenGL.so.0", "libGL.so.1"):
                try:
                    gl = ctypes.CDLL(name); break
                except OSError:
                    continue
            gl.glBindFramebuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
            gl.glBlitFramebuffer.argtypes = [ctypes.c_int]*8 + [ctypes.c_uint, ctypes.c_uint]
            self._glx = gl
        return self._glx

    def _blit_depth(self, src, dst, w, h):
        gl = self._gl()
        gl.glBindFramebuffer(0x8CA8, src.glo)              # GL_READ_FRAMEBUFFER
        gl.glBindFramebuffer(0x8CA9, dst.glo)              # GL_DRAW_FRAMEBUFFER
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, 0x00000100, 0x2600)   # GL_DEPTH_BUFFER_BIT, GL_NEAREST
        if os.environ.get("JUNO_HAIR_GLERR"):
            print("[hair] blit depth glGetError", gl.glGetError(), flush=True)
        dst.use()

    def _layer_cov(self, rig, cam, nf, lights):
        """the opaque depth -> the hair's targets, then every strand (+ followers) adds its true-width coverage"""
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        T = self._layer_targets()
        w, h = mc.iw, mc.ih
        self._blit_depth(mc.f_ms, T["f_hc"], w, h)
        self._blit_depth(mc.f_ms, T["f_cov"], w, h)
        T["f_cov"].use(); ctx.viewport = (0, 0, w, h)
        self._clear_color(T["f_cov"])
        p = self.p_cov
        self._draw_uniforms(p, cam, nf, lights)
        mc._set(p, "uA2C", 0); mc._set(p, "uSamples", 1); mc._set(p, "uDebug", 0); mc._set(p, "uArcN", float(self.arc_n))
        if self.v2:
            mc._set(p, "uPrepass", 0)
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "<=" if os.environ.get("JUNO_HAIR_COVZ", "1") == "1" else "1"; ctx.depth_mask = False
        ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE); ctx.disable(mgl.CULL_FACE)
        mc.t_shadow.use(4)
        for g in (self.groups if os.environ.get("JUNO_HAIR_STRANDS") != "0" else []):
            self._group_draw(p, self.vao_cov, rig, g, cov=True)
        ctx.disable(mgl.BLEND); ctx.depth_mask = True
        # the colour target: cleared colour, the opaque depth
        T["f_hc"].use()
        self._clear_color(T["f_hc"])

    def _clear_color(self, fb):
        """clear the colour attachments only (the blitted depth stays)"""
        fb.use()
        gl = self._gl()
        import ctypes
        if not hasattr(self, "_glclr"):
            gl.glClearBufferfv.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            self._glclr = (ctypes.c_float*4)(0.0, 0.0, 0.0, 0.0)
        for i in range(len(fb.color_attachments)):
            gl.glClearBufferfv(0x1800, i, self._glclr)     # GL_COLOR, draw buffer i

    def composite(self):
        """after the char layer's resolve + SSS: resolve the hair colour, premultiply by the analytic opacity
        1 - exp(-kappa x coverage) and blend it over t_mc / t_md (premultiplied over).  No-op without the layer."""
        if self.layer is None or self._lt is None or not self.groups:
            return
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        T = self._lt
        if T["key"][3] != id(mc.t_mc):
            return
        ctx.copy_framebuffer(T["f_rhc"], T["f_rhc_ms"])
        ctx.copy_framebuffer(T["f_rhd"], T["f_rhd_ms"])
        p = self.p_lcomp
        T["t_hc"].use(0); T["t_hd"].use(1); T["t_cov"].use(2)
        mc._set(p, "uHC", 0); mc._set(p, "uHD", 1); mc._set(p, "uCov", 2)
        mc._set(p, "uKappa", float(self.layer["kappa"]))
        mc._set(p, "uFill", 1 if self.layer.get("fill", True) else 0)
        mc._set(p, "uDbg", int(os.environ.get("JUNO_HAIR_LDBG", "0")))
        T["f_out"].use(); ctx.viewport = (0, 0, mc.iw, mc.ih)
        ctx.disable(mgl.DEPTH_TEST); ctx.disable(mgl.CULL_FACE)
        ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
        self.vao_lcomp.render(mgl.TRIANGLE_STRIP, vertices=4)
        ctx.disable(mgl.BLEND)
        if os.environ.get("JUNO_HAIR_DOP_DUMP"):
            a = np.frombuffer(T["t_dop"].read(alignment=1), np.float32 if T["t_dop"].dtype == "f4" else np.uint8)
            print("[hair] dop", T["t_dop"].dtype, a.size, flush=True)
            np.save(os.environ["JUNO_HAIR_DOP_DUMP"], a)
        if os.environ.get("JUNO_HAIR_COV_DUMP"):
            a = np.frombuffer(T["t_cov"].read(), np.float16).reshape(mc.ih, mc.iw)
            np.save(os.environ["JUNO_HAIR_COV_DUMP"], a.astype(np.float32))

    def _group_draw(self, p, vao, rig, g, cov=False):
        """uniforms of one strand group + its instanced ribbon draw (both draw passes and the HAIR2 coverage pass)"""
        mc, mgl = self.mc, self.mgl
        n = self._common(p, rig, g, 1.0)
        gm = g.meta
        cm = self.color_mul.get(g.name, self.color_mul.get("lashes" if g.name.startswith("lash") else
                                                           ("brows" if g.name.startswith("brow") else g.name), (1.0, 1.0, 1.0)))
        mc._set(p, "uColRoot", tuple(float(x)*float(m) for x, m in zip(gm["color_root"][:3], cm)))   # render8 color_mul
        mc._set(p, "uColTip", tuple(float(x)*float(m) for x, m in zip(gm["color_tip"][:3], cm)))
        kind = 0 if g.name in ("scalp",) else (1 if g.name == "fuzz" else (2 if g.name.startswith("brow") else 3))
        mc._set(p, "uKind", kind)
        if getattr(mc, "mtaa", None) is not None and mc.mtaa.cls_on:  # shade2-taa (gated): per group class
            mc._set(p, "uTaaR", mc.mtaa.cls_vec(("hair", "hair", "brows", "lashes")[kind]))
        mc._set(p, "uMinPx", float(self.min_px)*(0.65 if kind >= 2 else 1.0))     # render look-dev: fine brows/lashes
        if self.v2 and self.far is not None and kind <= 1:                         # far field
            mc._set(p, "uMinPx", float(self.min_px) + self._far*max(float(self.far["min_px"]) - float(self.min_px), 0.0))
        mc._set(p, "uRough", float(gm.get("roughness", 0.35)))
        mc._set(p, "uHairK", self.shade_k)                                        # render3
        mc._set(p, "uHairK2", self.shade_k2)
        mc._set(p, "uHairK3", self.shade_k3)                                              # render7
        mc._set(p, "uHairMass", float(self.mass_ao[0]))                                   # render5
        if self.v2:                                                                       # hair-shade
            rm, tm = self.v2k["root_mul"], self.v2k["tip_mul"]
            mc._set(p, "uColRoot", tuple(float(x)*float(m)*float(r) for x, m, r in zip(gm["color_root"][:3], cm, rm)))
            mc._set(p, "uColTip", tuple(float(x)*float(m)*float(r) for x, m, r in zip(gm["color_tip"][:3], cm, tm)))
            mk = {0: 1.0, 1: 1.0, 2: 1.15, 3: 1.6}[kind]
            mc._set(p, "uV2MelK", float(mk))
        mc._set(p, "uPull", 0.0004 if g.surface else 0.0)
        if self.layer is not None:                                                        # HAIR2 layer
            L = self.layer
            kn = ("scalp", "fuzz", "brows", "lashes")[kind]
            lv = L["len_var"] if kind == 0 else (0.0, 0.0, 0.1)
            mc._set(p, "uLenVar", tuple(float(x) for x in lv))
            fnum = int(L["followers"]) if (cov and kind == 0) else 1
            mc._set(p, "uFollowN", fnum)
            sp = L["spread_mm"]
            mc._set(p, "uFollowR", (float(sp[0])*1e-3, float(sp[1])*1e-3, float(L["spread_n"]), float(L["curl"])))
            mc._set(p, "uCovK", float((L.get("cov_k") or {}).get(g.name, (L.get("cov_k") or {}).get(kn, 1.0))))
            mc._set(p, "uCovTent", (float(L["tent_px"]), float(L["cov_mass"])))
            mc._set(p, "uTaperU", float(L["taper_u"]) if kind <= 1 else 1.0)
            nfull = g.S if g.surface else int(max(1, round(g.S*(self.fuzz_frac if g.name == "fuzz" else 1.0))))
            if cov:
                n = nfull*fnum
                if L.get("cov_lod", True) and not g.surface and self.frac < 0.999:
                    # the camera-distance strand fraction applies to the coverage too: a uniform subset (strands are
                    # pre-shuffled), each strand's coverage x 1/fraction keeps the summed coverage
                    f = max(min(self.frac, 1.0), 0.05)
                    n = max(1, int(round(nfull*f)))*fnum
                    mc._set(p, "uCovK", float((L.get("cov_k") or {}).get(g.name, (L.get("cov_k") or {}).get(kn, 1.0)))/f)
            else:
                mc._set(p, "uMinPx", float(L["col_minpx"])*(0.65 if kind >= 2 else 1.0))
                n = nfull if g.surface else max(1, int(round(nfull*float(L["col_frac"])*min(self.frac, 1.0))))
        vao.render(mgl.TRIANGLE_STRIP, vertices=2*self._npts(g), instances=n)

    def _lod_frac(self, rig, cam):
        """hair_render.lod_frac = [[px_per_mm, frac], ...] (ascending px_per_mm): the strand fraction for the head's
        on-screen scale, linear in between, clamped at the ends; min()'d with the quality ladder's frac"""
        pos, u, v, w, fl = cam
        hb = self.mc.A.bone_index.get("head", 0)
        c = np.asarray(rig.bones[hb], np.float64)[:3, 3]
        z = max(float(np.dot(c - np.asarray(pos, float), np.asarray(w, float))), 1e-3)
        ppm = float(self.mc.ih)*float(fl)/(2.0*z)*1e-3
        lf = 1.0
        if self.lod_frac:
            t = np.asarray(self.lod_frac, float)
            lf = float(np.interp(ppm, t[:, 0], t[:, 1]))
        fz = self.far
        far = 0.0
        if fz is not None:
            a, b = float(fz["ppm"][0]), float(fz["ppm"][1])          # far = 0 above a px/mm, 1 below b
            x = min(max((a - ppm)/max(a - b, 1e-6), 0.0), 1.0)
            far = x*x*(3.0 - 2.0*x)
            lf = min(lf, 1.0 + far*(float(fz["frac"]) - 1.0))
        self._far = far
        self._lodf = lf
        self._ppm = ppm

    def _draw_uniforms(self, p, cam, nf, lights):
        mc = self.mc
        mc._cam(p, cam, nf)
        L = lights
        for k, val in (("uKeyDir", L["key_dir"]), ("uKeyCol", L["key_col"]), ("uRimDir2", L["rim_dir"]),
                       ("uRimCol2", L["rim_col"]), ("uFillDir", L["fill_dir"]), ("uFillCol", L["fill_col"])):
            mc._set(p, k, tuple(float(x) for x in val))
        for k in ("uKeyGain", "uRimGain", "uAmbGain", "uCharK"):
            mc._set(p, k, float(L[k]))
        if "uLightVP" in p:
            p["uLightVP"].write(np.ascontiguousarray(mc.light_vp.T, np.float32).tobytes())
        mc._set(p, "uShadowZ", self._shadow_z)                          # ortho depth range (world), texel
        mc._set(p, "uDeep", float(self.deep_mm*1e-3) if self.do_shadow else -1.0)
        mc._set(p, "uMinPx", float(self.min_px))
        mc._set(p, "uDifK", float(self.dif_k))
        if self.v2:                                                     # hair-shade
            self._range_k = 1.0
            mc._set(p, "uShadowZ", getattr(self, "_dom_z", self._shadow_z))
            self._v2_uniforms(p)

    def _color_mask(self, on):
        """glColorMask on every draw buffer of the bound target (raw GL: moderngl's color_mask covers buffer 0 only)"""
        if not hasattr(self, "_gl_cm"):
            import ctypes
            gl = None
            for name in ("libOpenGL.so.0", "libGL.so.1"):
                try:
                    gl = ctypes.CDLL(name); break
                except OSError:
                    continue
            self._gl_cm = gl.glColorMask if gl is not None else None
            if self._gl_cm is not None:
                self._gl_cm.argtypes = [ctypes.c_ubyte]*4
        if self._gl_cm is not None:
            v = 1 if on else 0
            self._gl_cm(v, v, v, v)

    def _common_count(self, g):
        f = min(max(self.frac, 0.0), 1.0)*(self.fuzz_frac if g.name == "fuzz" else 1.0)
        return g.S if g.surface else int(max(1, round(g.S*f)))
