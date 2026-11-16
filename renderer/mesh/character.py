"""render: MeshCharacter — Juno's skinned mesh in the engine's GL context (docs/CONTRACT--render.md).

    mc = MeshCharacter(ctx, asset, w, h, shader_head, scene_lib, mesh_cfg)
    mc.draw(rig, cam, lights, scene)     # -> fills mc.t_mc (premultiplied colour) and mc.t_md ((t*a, ., ., a))
    mc.composite(fbo, cam, dof)          # mesh layer over the scene target with its own depth of field

GL 3.3 core only (the engine's context): transform feedback for the skinning, MSAA renderbuffers, texelFetch."""
import os
import re
import ctypes
import numpy as np
import poison                      # render6: JUNO_POISON debug switch (no-op unless set)
from light import MeshLight          # render-light

HERE = os.path.dirname(os.path.abspath(__file__))

# material type -> (shader define, defaults: rough, spec level, sss/sheen, detail tile), colour param
MAT_TYPES = {
    "skin":           ("MAT_SKIN", (0.42, 1.0, 1.0, 40.0), (1.0, 0.36, 0.22)),
    "eye":            ("MAT_EYE", (0.30, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
    "cornea":         ("MAT_CORNEA", (0.04, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
    "eye_occlusion":  ("MAT_EYE_OCCLUSION", (0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    "tearline":       ("MAT_TEARLINE", (0.05, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
    "teeth":          ("MAT_GENERIC", (0.28, 0.9, 0.6, 1.0), (1.0, 1.0, 1.0)),
    "gums":           ("MAT_GENERIC", (0.30, 0.9, 0.8, 1.0), (1.0, 1.0, 1.0)),
    "tongue":         ("MAT_GENERIC", (0.34, 0.9, 0.8, 1.0), (1.0, 1.0, 1.0)),
    "mouth_interior": ("MAT_MOUTH", (0.5, 0.3, 0.5, 1.0), (1.0, 1.0, 1.0)),
    "lashes":         ("MAT_LASHES", (0.5, 0.2, 0.0, 1.0), (1.0, 1.0, 1.0)),
    "cloth":          ("MAT_CLOTH", (0.70, 0.3, 0.6, 12.0), (1.0, 0.95, 0.85)),
    "buttons":        ("MAT_GENERIC", (0.22, 0.9, 0.8, 1.0), (1.0, 1.0, 1.0)),   # render-cloth: z clear coat
    "shoes":          ("MAT_GENERIC", (0.40, 0.7, 1.0, 1.0), (1.0, 1.0, 1.0)),   # render-cloth: z grain
    "hair":           ("MAT_GENERIC", (0.5, 0.5, 0.0, 1.0), (1.0, 1.0, 1.0)),
}
# material type -> its shader file renderer/mesh/mat_<file>.glsl (one owner per file)
MAT_FILES = {"skin": "skin", "eye": "eye", "cornea": "cornea", "eye_occlusion": "eye_occlusion", "tearline": "tearline",
             "teeth": "generic", "gums": "generic", "tongue": "generic", "mouth_interior": "generic", "lashes": "lashes",
             "cloth": "cloth", "buttons": "generic", "shoes": "generic", "hair": "generic"}
MAT_FILES.update(buttons="trim", shoes="trim")            # render-cloth: garment trims -> mat_trim.glsl
TRIM_DEFS = {"buttons": ["TRIM_BUTTONS"], "shoes": ["TRIM_SHOES"]}   # render-cloth
BLEND_TYPES = {"cornea", "eye_occlusion", "tearline"}
MAX_MORPH = 32
_HIDE = set(filter(None, os.environ.get("JUNO_HIDE", "").split(",")))   # render: look-dev toggles
_NANCHECK = os.environ.get("JUNO_NANCHECK") == "1"                     # render6: per-frame NaN readback (debug)


def _gl_enable(cap, on=True):
    """Raw glEnable/glDisable for caps moderngl does not wrap (alpha-to-coverage).  GLVND dispatches to the current
    context; returns False when the entry point cannot be found."""
    global _GL
    try:
        _GL
    except NameError:
        _GL = None
        for name in ("libOpenGL.so.0", "libGL.so.1"):
            try:
                _GL = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if _GL is None:
        return False
    (_GL.glEnable if on else _GL.glDisable)(ctypes.c_uint(cap))
    return True


def _gl_color_mask(buf, on):
    """shade: glColorMaski(buf, on x4) -- GL 3.0 core, not wrapped by moderngl."""
    global _GLM
    try:
        _GLM
    except NameError:
        _GLM = None
        for name in ("libOpenGL.so.0", "libGL.so.1"):
            try:
                _GLM = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if _GLM is None:
        return False
    b = ctypes.c_ubyte(1 if on else 0)
    _GLM.glColorMaski(ctypes.c_uint(buf), b, b, b, b)
    return True


GL_SAMPLE_ALPHA_TO_COVERAGE = 0x809E
GL_SAMPLE_ALPHA_TO_ONE = 0x809F


def material_type(mat_id, mat):
    t = (mat or {}).get("type")
    if t in MAT_TYPES:
        return t
    for k in ("cornea", "eye_occlusion", "tearline", "eye", "skin", "teeth", "gums", "tongue", "mouth", "lashes",
              "cloth", "button", "shoe"):
        if k in mat_id:
            return {"mouth": "mouth_interior", "button": "buttons", "shoe": "shoes"}.get(k, k)
    return "shoes"


class MeshCharacter:
    def __init__(self, ctx, asset, w, h, shader_head, scene_lib, mcfg=None, msaa=None, shadow_res=None, log=False):
        import moderngl
        self.mgl = moderngl
        self.ctx, self.A, self.w, self.h, self.log = ctx, asset, w, h, log
        self.cfg = dict(mcfg or {})
        self.msaa = int(msaa or self.cfg.get("msaa", 4))
        self.msaa = max(1, min(self.msaa, ctx.max_samples))
        self.shadow_res = int(shadow_res or self.cfg.get("shadow_res", 2048))
        self.shadow_taps = 16
        self.light = MeshLight(ctx, self.cfg, asset=asset)           # render-light: PCSS / capsule AO / probe knobs
        self.dof_n = 24
        self.iw, self.ih, self.mesh_scale = w, h, 1.0     # render-perf: forward-pass size (quality.py mesh_scale)
        self.stamp = None                                  # render-perf: GPU timestamp callback (juno_bench)
        src = open(os.path.join(HERE, "mesh.glsl")).read()
        if os.environ.get("JUNO_NOGUARD") == "1":            # render6: debug -- the finite guards off (find NaN sources)
            shader_head = shader_head + "#define JUNO_NOGUARD\n"
        import taa as _taa                                           # shade-taa (gated): mesh.pbr.taa
        self.taa_k = _taa.taa_cfg((self.cfg or {}).get("pbr") if isinstance((self.cfg or {}).get("pbr"), dict) else None)
        if self.taa_k is not None:
            shader_head = shader_head + "#define MESH_TAA_JIT\n"       # camClip jitter in every raster pass (hair too)
        self.mtaa = None
        self.src, self.head, self.scene_lib = src, shader_head, scene_lib
        A = asset
        N = int(A.mesh_meta["vertex_count"])
        self.N = N
        mats = A.meta.get("materials", {})
        self.has_cornea = any(material_type(s["material"], mats.get(s["material"])) == "cornea" for s in A.submeshes)
        import pbr as _pbr                                           # shade: the gated PBR face path
        self.pbr = _pbr.pbr_cfg(self.cfg)                            # None = the path (bit-identical)
        self.pbr_split = bool(_pbr.skin_on(self.pbr) and _pbr.sub(self.pbr, "skin", _pbr.SKIN_DEFAULTS).get("albedo_split"))
        # ---------------- programs
        P = lambda vs, fs=None, **kw: ctx.program(vertex_shader=vs, fragment_shader=fs, **kw)
        self.p_skin = ctx.program(vertex_shader=self._src("VS_SKIN"), varyings=["oPos", "oNrm", "oTan"],
                                  varyings_capture_mode="interleaved")
        self.p_shadow = P(self._src("VS_SHADOW"), self._src("FS_SHADOW"))
        self.p_prefill = P(self._src("VS_FULLM"), self._src("FS_PREFILL"))
        self.ao_on = bool(self.pbr) and self.pbr.get("ao", False) not in (False, None)          # shade-post (gated): mesh.pbr.ao contact AO
        self.p_comp = P(self._src("VS_FULLM"), self._src("FS_COMP", *(["PBR_AO"] if self.ao_on else [])))
        self.p_bbdown = P(self._src("VS_FULLM"), self._src("FS_BBDOWN"))   # render7: zoom-driven room blur
        self.t_bb = None
        self.p_fwd = {}
        for s in A.submeshes:
            t = material_type(s["material"], mats.get(s["material"]))
            if t not in self.p_fwd:
                d = [MAT_TYPES[t][0]] + (["EYE_HAS_CORNEA"] if (t == "eye" and self.has_cornea) else [])
                d += TRIM_DEFS.get(t, [])                                  # render-cloth
                mf = MAT_FILES[t]
                if self.pbr:                                               # shade (gated)
                    mf, on = _pbr.material_file(self.pbr, t, mf)
                    d += (["PBR_MAT", "PBR_T_" + t.upper()] if on else []) + (["PBR_ALB_SPLIT"] if self.pbr_split else [])
                    if isinstance((self.pbr.get("light") or {}).get("spot"), dict):   # shade2: portrait key falloff
                        d += ["PBR_SPOT"]
                    if self.pbr.get("debug"):                              # look-dev: PBR_DBG_<NAME> views
                        d += ["PBR_DBG_" + str(self.pbr["debug"]).upper()]
                if self.taa_k is not None:                                 # shade-taa: velocity attachment
                    d += ["MESH_TAA", "MESH_TAA_LOC 4"]
                    if self.taa_k.get("classes"):                          # shade2-taa: class attachment (loc 5)
                        d += ["MESH_TAA_R"]
                vd = ["MESH_TAA"] if self.taa_k is not None else []          # shade-taa: the VS's motion varyings
                self.p_fwd[t] = P(self._src("VS_FWD", *vd), self._src("FS_FWD", *d, lib=True, mat=mf))
        # ---------------- buffers
        m = A.mesh
        jn = m["joints"].astype(np.float32)
        wt = m["weights"].astype(np.float32)
        # morph region: every vertex any shape touches
        touched = np.zeros(N, bool)
        for _, idx, _, _ in A.shapes:
            touched[idx] = True
        reg_ids = np.nonzero(touched)[0]
        reg = np.full(N, -1.0, np.float32)
        reg[reg_ids] = np.arange(len(reg_ids), dtype=np.float32)
        # render8: the face as one smooth stylized form (smoothform.py; mesh.face.smooth_form, absent = off)
        from smoothform import smooth_form
        sf = smooth_form(A, (self.cfg.get("face") or {}).get("smooth_form"))
        nrm0 = m["normal"].astype(np.float32) if sf is None else sf[0]
        rest = np.concatenate([m["position"].astype(np.float32), nrm0,
                               m["tangent"].astype(np.float32), jn, wt, reg[:, None]], 1)
        self.vb_rest = ctx.buffer(np.ascontiguousarray(rest, np.float32).tobytes())
        self.vb_posed = ctx.buffer(reserve=N*10*4)
        self.vb_prev = ctx.buffer(reserve=N*10*4) if self.taa_k is not None else None   # shade-taa
        self.vb_uv = ctx.buffer(np.ascontiguousarray(m["uv0"], np.float32).tobytes())
        self.ib = ctx.buffer(np.ascontiguousarray(m["index"], np.uint32).tobytes())
        self.vao_skin = ctx.vertex_array(self.p_skin, [(self.vb_rest, "3f 3f 4f 4f 4f 1f",
                                                        "aPos", "aNrm", "aTan", "aJnt", "aWt", "aReg")])
        # morph texture array
        R = max(len(reg_ids), 1)
        MW = min(2048, R); MH = (R + MW - 1)//MW
        ns = max(len(A.shapes), 1)
        D = np.zeros((2*ns, MH*MW, 4), np.float16)
        for s, (_, idx, dp, dn) in enumerate(A.shapes):
            r = reg[idx].astype(np.int64)
            D[2*s, r, :3] = dp
            D[2*s + 1, r, :3] = dn
            if sf is not None and s in sf[2]:                            # render8: smoothed normal deltas
                rv = reg[sf[1]].astype(np.int64); okv = rv >= 0
                D[2*s + 1, rv[okv], :3] = sf[2][s][okv]
        self.morph_mb = D.nbytes/1e6
        self.t_morph = ctx.texture_array((MW, MH, 2*ns), 4, D.reshape(2*ns, MH, MW, 4).tobytes(), dtype="f2")
        self.t_morph.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.p_skin["uMorphW"].value = MW
        self._set(self.p_skin, "uMorph", 1)
        self._set(self.p_skin, "uBones", 0)
        nb = len(A.bones)
        self.t_bones = ctx.texture((3, nb), 4, dtype="f4")
        self.t_bones.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.inv_bind = A.inv_bind.astype(np.float64)
        # per-program draw vaos over the posed buffer
        def vao_for(prog):
            fmt, names = [], []
            for nm, f, nbytes in (("aP", "3f", 12), ("aN", "3f", 12), ("aT", "4f", 16)):
                if nm in prog:
                    fmt.append(f); names.append(nm)
                else:
                    fmt.append(f"{nbytes}x")
            cont = [(self.vb_posed, " ".join(fmt), *names)]
            if "aUV" in prog:
                cont.append((self.vb_uv, "2f", "aUV"))
            if self.vb_prev is not None and "aPp" in prog:           # shade-taa: last frame's posed position
                cont.append((self.vb_prev, "3f 28x", "aPp"))
            return ctx.vertex_array(prog, cont, index_buffer=self.ib, index_element_size=4)
        self.vao_shadow = vao_for(self.p_shadow)
        self.vao_fwd = {t: vao_for(p) for t, p in self.p_fwd.items()}
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao_prefill = ctx.vertex_array(self.p_prefill, [(self.quad, "2f", "in_pos")])
        self.vao_comp = ctx.vertex_array(self.p_comp, [(self.quad, "2f", "in_pos")])
        self.vao_bbdown = ctx.vertex_array(self.p_bbdown, [(self.quad, "2f", "in_pos")])   # render7
        self.ao = None
        if self.ao_on:                                                   # shade-post (gated)
            from ssao import MeshAO
            self.ao = MeshAO(self, self.pbr.get("ao"))
        # ---------------- materials
        self.tex_cache = {}
        self.draws = []          # (order, type, submesh, material uniforms)
        suf = A.lod.get("tex_suffix", "")
        _hide = set((A.meta.get("hair_render") or {}).get("hide_submeshes") or [])   # hair: strands replace cards
        for s in A.submeshes:
            if _hide and s.get("name") in _hide:                                     # hair (absent = all drawn)
                continue
            mid = s["material"]; md = mats.get(mid, {}) or {}
            t = material_type(mid, md)
            dfl, col = MAT_TYPES[t][1], MAT_TYPES[t][2]
            pr = md.get("params", {}) or {}
            texs = {}
            for key, unit in (("albedo", 0), ("normal", 1), ("packed", 2)):
                if md.get(key):
                    texs[unit] = self._tex(os.path.join(A.dir, md[key].replace(".png", f"{suf}.png") if ".png" in md[key]   # head-lod: .jpg/.webp maps
                                                        else "{}{}{}".format(os.path.splitext(md[key])[0], suf, os.path.splitext(md[key])[1])),
                                           srgb=False, fallback=os.path.join(A.dir, md[key]))
            det = pr.get("detail_normal")
            if det:
                texs[3] = self._tex(os.path.join(A.dir, det), srgb=False)
            alb_c = pr.get("albedo_color") or md.get("color") or [0.6, 0.6, 0.6]
            u = dict(
                uTexOn=tuple(1.0 if (k in texs and texs[k] is not None) else 0.0 for k in range(4)),
                uAlbC=tuple(float(x) for x in alb_c[:3]),
                uMatP=(float(pr.get("roughness", dfl[0])), float(pr.get("spec_level", dfl[1])),
                       float(pr.get("sss_strength", pr.get("sheen", dfl[2]))), float(pr.get("detail_tile", dfl[3]))),
                uMatC=tuple(float(x) for x in (pr.get("sss_color") or pr.get("sheen_color") or col)[:3]),
                uEyeP=(float(pr.get("iris_uv_radius", 0.16)), float(pr.get("pupil_uv_radius", 0.06)),
                       float(pr.get("iris_depth_uv", 0.035)), float(pr.get("limbus_width", 0.02))),
                uAlphaMul=1.0)
            order = 0 if t not in BLEND_TYPES and t != "lashes" else (1 if t == "lashes" else
                                                                      {"eye_occlusion": 2, "tearline": 3, "cornea": 4}[t])
            if t == "cloth":                                   # render-cloth
                import cloth as _cloth                         # render-cloth
                _cloth.setup(self, u, texs, mid, md, s)        # render-cloth
                if _pbr.cloth_on(self.pbr):                    # shade-cloth (gated: mesh.pbr.cloth)
                    _pbr.cloth_setup(self, u, texs, mid, md, s, self.pbr)
            self.draws.append(dict(order=order, type=t, sub=s, u=u, tex=texs,
                                   cull=not s.get("double_sided", False) and t not in ("lashes", "eye_occlusion", "tearline")))
        self.draws.sort(key=lambda d: d["order"])
        for t, p in self.p_fwd.items():
            for nm, unit in (("tAlb", 0), ("tNrm", 1), ("tPack", 2), ("tDet", 3), ("uShadow", 4)):
                self._set(p, nm, unit)
        # ---------------- targets
        from sss import from_asset as _sss_from_asset          # render-skin
        self.sss = _sss_from_asset(self, A, self.cfg)          # render-skin
        from eyesmouth import face_setup                       # render-face
        face_setup(self)                                       # render-face
        self.pbr_face = None
        self.face_shadow = None
        if self.taa_k is not None:                             # shade-taa (gated)
            from taa import MeshTAA
            self.mtaa = MeshTAA(self, self.taa_k)
        if self.pbr:                                           # shade (gated)
            self.pbr_face = _pbr.PbrFace(self, self.pbr)
            fsc = (self.pbr.get("light") or {}).get("face_shadow", {})
            if fsc is not False:
                self.face_shadow = _pbr.FaceShadow(self, fsc if isinstance(fsc, dict) else {})
        self._alloc_targets(w, h)
        self.t_shadow = ctx.depth_texture((self.shadow_res, self.shadow_res))
        self.t_shadow.compare_func = ""
        self.t_shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_shadow.repeat_x = self.t_shadow.repeat_y = False
        self.f_shadow = ctx.framebuffer(depth_attachment=self.t_shadow)
        self._set(self.p_prefill, "uBgD", 0); self._set(self.p_prefill, "uChC", 1); self._set(self.p_prefill, "uChD", 2)
        self._set(self.p_prefill, "uChM", 3); self._set(self.p_comp, "uChM", 9)        # render2: tile mask
        self._set(self.p_comp, "uMC", 0); self._set(self.p_comp, "uMD", 1)
        for nm, unit in (("uBgD", 2), ("uChC", 3), ("uChD", 4)):          # render-light
            self._set(self.p_comp, nm, unit)
        self.light_vp = np.eye(4)
        self.stats = {}
        self.hair = None                                                     # render-hair
        if getattr(A, "hair", None) and self.cfg.get("hair", {}).get("on", True):  # render-hair
            from hair import HairRenderer                                    # render-hair
            self.hair = HairRenderer(self)                                   # render-hair

    # ------------------------------------------------------------------ helpers
    def _src(self, section, *defs, lib=False, mat=None):
        hdr = self.head + "".join(f"#define {d}\n" for d in (section,) + defs)
        body = self.src
        body = body.replace("#pragma include light\n", open(os.path.join(HERE, "light.glsl")).read() + "\n")  # render-light
        if mat is not None:          # the material's own file (mat_<kind>.glsl) at the "#pragma include mat" line
            mtxt = open(os.path.join(HERE, f"mat_{mat}.glsl")).read()
            if mat.endswith("_pbr"):                                         # shade: the PBR materials' shared terms
                mtxt = open(os.path.join(HERE, "pbr_common.glsl")).read() + "\n" + mtxt
            body = body.replace("#pragma include mat\n", mtxt + "\n")
        if lib:
            return hdr + self.scene_lib + "\n" + body
        # the passes without the scene library still need PI / hash12
        return hdr + "#define PI 3.14159265359\nfloat hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); " \
                     "p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }\n" + body

    @staticmethod
    def _set(prog, name, value):
        if name in prog:
            prog[name].value = value

    def _tex(self, path, srgb=False, fallback=None):
        if not os.path.exists(path):
            path = fallback if (fallback and os.path.exists(fallback)) else None
        if path is None:
            return None
        if path in self.tex_cache:
            return self.tex_cache[path]
        from PIL import Image
        im = Image.open(path)
        im = im.convert("RGBA")
        a = np.asarray(im, np.uint8)[::-1]          # OpenGL: row 0 = bottom (uv v = 0 at the bottom)
        t = self.ctx.texture((a.shape[1], a.shape[0]), 4, np.ascontiguousarray(a).tobytes())
        t.build_mipmaps()
        t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        try:
            t.anisotropy = 8.0
        except Exception:
            pass
        self.tex_cache[path] = t
        return t

    def _alloc_targets(self, w, h):
        ctx, mgl = self.ctx, self.mgl
        s = self.msaa
        self.rb_c = ctx.renderbuffer((w, h), 4, samples=s, dtype="f2")
        self.rb_d = ctx.renderbuffer((w, h), 4, samples=s, dtype="f2")
        self.rb_z = ctx.depth_renderbuffer((w, h), samples=s)
        ext = self.sss.ms_attachments(w, h, s)                                                          # render-skin
        if self.mtaa is not None:                              # shade-taa: velocity at location 4 (pads 2..3)
            ext = ext + self.mtaa.ms_attachment(w, h, s, len(ext))
        self.f_ms = ctx.framebuffer([self.rb_c, self.rb_d] + ext, self.rb_z)
        self.t_mc = ctx.texture((w, h), 4, dtype="f2"); self.t_md = ctx.texture((w, h), 4, dtype="f2")
        for t in (self.t_mc, self.t_md):
            t.filter = (mgl.LINEAR, mgl.LINEAR); t.repeat_x = t.repeat_y = False
        self.f_res = [ctx.framebuffer([self.t_mc]), ctx.framebuffer([self.t_md])]
        self.f_ms_c = ctx.framebuffer([self.rb_c]); self.f_ms_d = ctx.framebuffer([self.rb_d])

    # ------------------------------------------------------------------ per frame
    def pose(self, rig):
        """M1: bones + shape weights -> the posed vertex buffer (transform feedback)."""
        ctx, mgl = self.ctx, self.mgl
        skin = np.einsum("bij,bjk->bik", np.asarray(rig.bones, np.float64), self.inv_bind)
        self.t_bones.write(np.ascontiguousarray(skin[:, :3, :], np.float32).tobytes())
        w = np.asarray(rig.weights, np.float32)
        nz = np.nonzero(np.abs(w) > 1e-4)[0]
        if len(nz) > MAX_MORPH:
            nz = nz[np.argsort(-np.abs(w[nz]))[:MAX_MORPH]]
        ids = np.zeros(MAX_MORPH, np.int32); wts = np.zeros(MAX_MORPH, np.float32)
        ids[:len(nz)] = nz; wts[:len(nz)] = w[nz]
        self.p_skin["uMorphN"].value = int(len(nz))
        if "uMorphId" in self.p_skin:
            self.p_skin["uMorphId"].write(ids.tobytes())
            self.p_skin["uMorphWt"].write(wts.tobytes())
        self.t_bones.use(0); self.t_morph.use(1)
        if self.mtaa is not None and not self.mtaa.copy_prev:   # shade-taa: keep last frame's posed vertices
            ctx.copy_buffer(self.vb_prev, self.vb_posed)
        self.vao_skin.transform(self.vb_posed, mgl.POINTS, vertices=self.N)
        if self.mtaa is not None and self.mtaa.copy_prev:       # a reset frame: no motion
            ctx.copy_buffer(self.vb_prev, self.vb_posed)
            self.mtaa.copy_prev = False

    def shadow(self, rig, key_dir):
        """M2: orthographic key-light shadow map fitted to the character's bound."""
        ctx, mgl = self.ctx, self.mgl
        c = np.asarray(rig.bound[:3], float); r = float(rig.bound[3])
        L = np.asarray(key_dir, float); L = L/np.linalg.norm(L)
        f = -L
        up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
        rt = np.cross(f, up); rt /= np.linalg.norm(rt)
        u2 = np.cross(rt, f)
        M, self.shadow_world_texel = self.light.fit(rig.bound, key_dir, self.shadow_res)   # render-light: snapped
        self.light_vp = M
        self.p_shadow["uLightVP"].write(np.ascontiguousarray(M.T, np.float32).tobytes())
        self.f_shadow.use(); ctx.viewport = (0, 0, self.shadow_res, self.shadow_res)
        self.f_shadow.clear(depth=1.0)
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "<"
        ctx.disable(mgl.CULL_FACE)
        for d in self.draws:
            if d["order"] == 0 and d["type"] not in ("mouth_interior",):
                s = d["sub"]
                self.vao_shadow.render(mgl.TRIANGLES, vertices=s["index_count"], first=s["first_index"])
        if self.hair is not None:                                            # render-hair
            self.hair.shadow(rig, M, L, self.shadow_world_texel)             # render-hair
        ctx.disable(mgl.DEPTH_TEST)
        if self.face_shadow is not None:                                     # shade (gated): the head cascade
            self.face_shadow.render(rig, L)

    def _cam(self, prog, cam, nf, res=None):
        pos, u, v, w, fl = cam
        self._set(prog, "uCamPos", tuple(float(x) for x in pos)); self._set(prog, "uCamU", tuple(float(x) for x in u))
        self._set(prog, "uCamV", tuple(float(x) for x in v)); self._set(prog, "uCamW", tuple(float(x) for x in w))
        res = res or (self.iw, self.ih)        # render-perf: the forward-pass size (hair/prefill/forward); composite passes (w, h)
        self._set(prog, "uFl", float(fl)); self._set(prog, "uRes", (float(res[0]), float(res[1])))
        self._set(prog, "uNF", (float(nf[0]), float(nf[1])))
        if self.mtaa is not None:                              # shade-taa: jitter + the previous camera
            self.mtaa.cam_uniforms(prog)

    def draw(self, rig, cam, lights, scene, time_s=0.0):
        """M3-M6: scene depth prefill, forward (MSAA), resolve -> t_mc / t_md.  scene = (bg texture with a = ray
        distance, char colour, char depth) of the SDF frame."""
        ctx, mgl = self.ctx, self.mgl
        pos = np.asarray(cam[0], float); W = np.asarray(cam[3], float)
        c = np.asarray(rig.bound[:3], float); r = float(rig.bound[3])
        zc = float(np.dot(c - pos, W))
        nf = (max(0.05, zc - 1.6*r), zc + 1.6*r)
        self.nf = nf
        ires = (self.iw, self.ih)          # render-perf: internal size (quality.py mesh_scale)
        if self.mtaa is not None:                               # shade-taa: this frame's jitter / previous camera
            self.mtaa.begin(rig, cam)
        self.f_ms.use(); ctx.viewport = (0, 0, self.iw, self.ih)
        self.f_ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        # M3: scene depth
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "1"
        self._cam(self.p_prefill, cam, nf, ires)
        scene[0].use(0); scene[1].use(1); scene[2].use(2)
        self._ch_gate(self.p_prefill, scene, 3)                 # render2
        self.vao_prefill.render(mgl.TRIANGLE_STRIP)
        ctx.depth_func = "<="
        self.sss.clear(); self.f_ms.use()                      # render-skin
        if self.mtaa is not None:                               # shade-taa: no velocity = (0, 0, 0, 0)
            self.f_ms_v.clear(0.0, 0.0, 0.0, 0.0); self.f_ms.use()
        if self.stamp:
            self.stamp("prefill")
        # M4: forward
        self.t_shadow.use(4)
        self._scene = scene                                     # render-light: the composite reads it
        self.light.capsules(rig); self.light.bind(self.t_shadow, 5, 6)
        if self.face_shadow is not None:                        # shade
            self.face_shadow.bind()
        L = lights
        lvp = np.ascontiguousarray(self.light_vp.T, np.float32).tobytes()
        for t, p in self.p_fwd.items():
            self._cam(p, cam, nf, ires)
            if "uLightVP" in p:
                p["uLightVP"].write(lvp)
            for k, val in (("uKeyDir", L["key_dir"]), ("uKeyCol", L["key_col"]), ("uRimDir2", L["rim_dir"]),
                           ("uRimCol2", L["rim_col"]), ("uFillDir", L["fill_dir"]), ("uFillCol", L["fill_col"]),
                           ("uSunDir", L["key_dir"]), ("uSunCol", L["key_col"]), ("uRimDir", L["rim_dir"]),
                           ("uRimCol", L["rim_col"])):
                self._set(p, k, tuple(float(x) for x in val))
            for k in ("uKeyGain", "uRimGain", "uAmbGain", "uCharK"):
                self._set(p, k, float(L[k]))
            self._set(p, "uAmbTint", tuple(float(x) for x in L.get("amb_tint", (1.0, 1.0, 1.0))))   # render look-dev
            self._set(p, "uShadowTexel", (1.0/self.shadow_res, float(self.shadow_world_texel)))
            self._set(p, "uShadowTaps", float(self.shadow_taps))
            self._set(p, "uTime", float(time_s))
            self.sss.uniforms(p)                               # render-skin
            self.face.apply(p, rig)                            # render-face
            self.light.uniforms(p, rig)                         # render-light
            if self.pbr_face is not None:                       # shade (gated)
                self.pbr_face.apply(p, rig)
            if self.face_shadow is not None:
                self.face_shadow.uniforms(p)
        cur_cull = None
        for d in self.draws:
            if d["sub"]["name"] in _HIDE: continue                 # render: debug JUNO_HIDE=name,name
            t = d["type"]; p = self.p_fwd[t]; s = d["sub"]
            for k, val in d["u"].items():
                self._set(p, k, val)
            for unit, tx in d["tex"].items():
                if tx is not None:
                    tx.use(unit)
            if d["order"] >= 2:
                ctx.enable(mgl.BLEND)
                ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
                ctx.depth_mask = False
            else:
                ctx.disable(mgl.BLEND)
                ctx.depth_mask = True
            if self.mtaa is not None:                           # shade-taa: overlays keep the surface's velocity
                _gl_color_mask(4, d["order"] < 2)
                if self.mtaa.cls_on:                            # shade2-taa: ... and its TAA class
                    _gl_color_mask(5, d["order"] < 2)
                    self._set(p, "uTaaR", self.mtaa.cls_vec(self.mtaa.mat_class(t, s.get("name", ""))))
            if d["cull"] != cur_cull:
                (ctx.enable if d["cull"] else ctx.disable)(mgl.CULL_FACE)
                cur_cull = d["cull"]
            self.vao_fwd[t].render(mgl.TRIANGLES, vertices=s["index_count"], first=s["first_index"])
        ctx.disable(mgl.BLEND); ctx.depth_mask = True
        self.light.unbind()                                     # render-light (before hair: it uses units 5-7)
        if self.face_shadow is not None:
            self.face_shadow.unbind()
        if self.stamp:
            self.stamp("forward")                               # render-perf
        if self.hair is not None:                                            # render-hair
            if self.pbr_split:                                               # shade: hair never writes the
                _gl_color_mask(3, False)                                     # skin-albedo attachment (location 3)
            if self.mtaa is not None:                                        # shade-taa: nor the velocity
                _gl_color_mask(4, False)
                if self.mtaa.cls_on:                                         # shade2-taa: strands write their class
                    _gl_color_mask(5, True)
            self.hair.draw(rig, cam, nf, lights)                             # render-hair
            if self.pbr_split:
                _gl_color_mask(3, True)
            if self.mtaa is not None:
                _gl_color_mask(4, True)
                if self.mtaa.cls_on:
                    _gl_color_mask(5, True)
        ctx.disable(mgl.CULL_FACE); ctx.disable(mgl.DEPTH_TEST)
        if self.stamp:
            self.stamp("hair")                                  # render-perf
        # M6: resolve both layers
        ctx.copy_framebuffer(self.f_res[0], self.f_ms_c)
        ctx.copy_framebuffer(self.f_res[1], self.f_ms_d)
        if self.stamp:
            self.stamp("resolve")                               # render-perf
        self.sss.apply(cam, self.t_md, self.f_res[0])          # render-skin: M5 SSS onto the resolved layer
        if self.hair is not None and getattr(self.hair, "layer", None) is not None:   # HAIR2 (hair_render.layer;
            self.hair.composite()                                                     # absent = no call)
        if self.ao is not None:                                 # shade-post (gated): contact AO from the layer depth
            if self.stamp:
                self.stamp("sss")
            self.ao.run(cam)
            if self.stamp:
                self.stamp("ao")
        if self.mtaa is not None:                               # shade-taa: the temporal resolve
            self.mtaa.resolve()
            if self.stamp:
                self.stamp("taa")
        if _NANCHECK:                                           # render6: JUNO_NANCHECK=1 debug readback
            self._nancheck()

    def _nancheck(self):
        """render6: count non-finite / negative texels in the mesh layers after the SSS (one line per frame)."""
        out = {}
        for nm, t in (("mc", self.t_mc), ("md", self.t_md), ("s", getattr(self.sss, "t_s", None)), ("h", getattr(self.sss, "t_h", None))):
            if t is None:
                continue
            a = np.frombuffer(t.read(), np.float16).astype(np.float32)
            out[nm] = (int((~np.isfinite(a)).sum()), int((a < -1e-3).sum()))
        self.nan_stats = out
        print(f"[nancheck] {out}", flush=True)

    def _ch_gate(self, prog, scene, unit):
        """render2: scene = (bg, ch_c, ch_d[, tile_mask, inv_spp, (iw, ih)]): the SDF character layer is valid only in
        the tiles this frame's mask marks (both the raw layer and the TAA history), the raw layer is summed over spp."""
        if len(scene) > 3 and scene[3] is not None and os.environ.get("JUNO_DBG_NOCHMASK") != "1":
            scene[3].use(unit)
            self._set(prog, "uChMOn", 1.0); self._set(prog, "uChInv", float(scene[4]))
            self._set(prog, "uChTileRes", (float(scene[5][0]), float(scene[5][1])))
        else:
            self._set(prog, "uChMOn", 0.0); self._set(prog, "uChInv", 1.0)

    def zoom_weight(self, zoom):
        """render7: 0 (rest / wide / board view) .. 1 (the home close-up) from the camera's zoom factor kz = fl/fl0
        (mesh.bg_zoom z0..z1, smoothstep): the live close-up is a ZOOM of the rest camera (focus stays ~7.7), so the
        close-view look (room haze + a longer lens's background blur) is driven by it -- it eases with the view weight."""
        bz = (self.cfg or {}).get("bg_zoom") or {}
        if not bz or zoom is None:
            return 0.0
        z0, z1 = float(bz.get("z0", 1.6)), float(bz.get("z1", 3.0))
        g = min(1.0, max(0.0, (float(zoom) - z0)/max(z1 - z0, 1e-3)))
        return g*g*(3.0 - 2.0*g)

    def _bb_pass(self, src, focus, zw):
        """render7: the living background -> half-res, premultiplied by its far weight, mipmapped (uBB)."""
        ctx, mgl = self.ctx, self.mgl
        bw, bh = max(1, self.w//2), max(1, self.h//2)
        if self.t_bb is None or self.t_bb.size != (bw, bh):
            if self.t_bb is not None:
                self.f_bb.release(); self.t_bb.release()
            self.t_bb = ctx.texture((bw, bh), 4, dtype="f2")
            self.t_bb.repeat_x = self.t_bb.repeat_y = False
            self.f_bb = ctx.framebuffer([self.t_bb])
        src.use(2)
        self._set(self.p_bbdown, "uBgD", 2); self._set(self.p_bbdown, "uBBRes", (float(bw), float(bh)))
        self._set(self.p_bbdown, "uBBFocus", float(focus))
        bz = (self.cfg or {}).get("bg_zoom") or {}                 # render8: highlight-weighted gather
        self._set(self.p_bbdown, "uBBH", (float(bz.get("hl", 0.0)), float(bz.get("hl_th", 0.6))))
        self.f_bb.use(); ctx.viewport = (0, 0, bw, bh)
        ctx.disable(mgl.BLEND)
        self.vao_bbdown.render(mgl.TRIANGLE_STRIP)
        self.t_bb.build_mipmaps()
        self.t_bb.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)

    def composite(self, fbo, cam, focus, dofk, frame=0, dof_max=12.0, zoom=None):
        """M7: the mesh layer over the scene target (premultiplied; the target's alpha = scene depth is kept).
        zoom: the live camera's zoom factor fl/fl0 ( render7: drives the close-view haze + background blur)."""
        ctx, mgl = self.ctx, self.mgl
        p = self.p_comp
        zw = self.zoom_weight(zoom)
        bz = (self.cfg or {}).get("bg_zoom") or {}
        sc = getattr(self, "_scene", None)
        bbk = (0.0, 0.0, 0.0, 0.0)
        # render8: bg_zoom.card -- the close-focus views (card stills) get the lens gather too, weighted like the haze
        zb = zw
        if float(bz.get("card", 0.0)) > 0.0:
            bgc = (self.cfg or {}).get("bg") or {}
            nr, fr = float(bgc.get("near", 3.5)), float(bgc.get("far", 5.5))
            g3 = min(1.0, max(0.0, (fr - float(focus))/max(fr - nr, 1e-3)))
            zb = max(zb, g3*g3*(3.0 - 2.0*g3)*float(bz["card"]))
        if zb > 1e-3 and sc is not None and "uBBK" in p and float(bz.get("r", 0.0)) > 0.0:
            self._bb_pass(sc[0], focus, zb)
            px = self.h/1080.0
            bbk = (float(bz["r"])*zb*px, float(bz.get("max", 36.0))*px, float(bz.get("lod_bias", 0.0)), float(bz.get("taps", 0)))
        if "uBBK" in p:
            self._set(p, "uBBK", bbk); self._set(p, "uBB", 11)
        self._cam(p, cam, getattr(self, "nf", (0.1, 100.0)), (self.w, self.h))   # render-perf
        self._set(p, "uFocus", float(focus)); self._set(p, "uDofK", float(dofk)); self._set(p, "uDofMax", float(dof_max))
        self._set(p, "uFrame", int(frame)); self._set(p, "uDofN", int(self.dof_n))
        # render5: close-view room haze (mesh.bg): full at focus <= near, off at focus >= far (the wide set shot)
        bg = (self.cfg or {}).get("bg") or {}
        if bg and "uBgK" in p:
            nr, fr = float(bg.get("near", 3.5)), float(bg.get("far", 5.5))
            g = min(1.0, max(0.0, (fr - float(focus))/max(fr - nr, 1e-3)))
            g = g*g*(3.0 - 2.0*g)
            g = max(g, zw*float(bz.get("haze", 1.0)))          # render7: the live close-up (zoom) gets the haze too
            self._set(p, "uBgK", (float(bg.get("k", 0.0))*g, float(bg.get("start", 0.6)), float(bg.get("ramp", 2.5)), float(bg.get("replace", 1.0))))
            self._set(p, "uBgCol", tuple(float(x) for x in bg.get("col", (0.6, 0.6, 0.56))))
        elif "uBgK" in p:
            self._set(p, "uBgK", (0.0, 0.0, 1.0, 0.0))
        # render8: close-view background grade (mesh.bg_grade), the same close weight as the haze
        bgg = (self.cfg or {}).get("bg_grade") or {}
        if "uBgG" in p:
            gw = 0.0
            if bgg:
                nr, fr = float(bg.get("near", 3.5)), float(bg.get("far", 5.5))
                g2 = min(1.0, max(0.0, (fr - float(focus))/max(fr - nr, 1e-3)))
                gw = max(g2*g2*(3.0 - 2.0*g2), zw)*float(bgg.get("w", 1.0))
            kd = np.asarray(self.light.key_dir, float)
            ux = np.asarray(cam[1], float) if cam is not None else np.array([1.0, 0.0, 0.0])
            ks = 1.0 if float(np.dot(kd, ux)) >= 0.0 else -1.0      # the key's side of the frame
            self._set(p, "uBgG", (float(bgg.get("level", 1.0)), float(bgg.get("grad", 0.0)), ks, float(gw)))
            self._set(p, "uBgCool", tuple(float(x) for x in bgg.get("cool", (0.0, 0.0, 0.0))))
        if os.environ.get("JUNO_DBG_FOCUS"):
            print(f"[mesh] composite focus {float(focus):.3f} dofk {float(dofk):.2f} zoom {zoom} zw {zw:.3f} bbk {bbk}", flush=True)
        fbo.use(); ctx.viewport = (0, 0, self.w, self.h)
        (self.mtaa.out if (self.mtaa is not None and self.mtaa.out is not None) else self.t_mc).use(0)   # shade-taa
        self.t_md.use(1)
        if sc is not None:
            sc[0].use(2); sc[1].use(3); sc[2].use(4)
            self._ch_gate(p, sc, 9)                             # render2
        if bbk[0] > 0.0:
            self.t_bb.use(11)                                   # render7
        self.light.uniforms(p)
        if "uLightVP" in p:
            p["uLightVP"].write(np.ascontiguousarray(self.light_vp.T, np.float32).tobytes())
        self._set(p, "uShadowTexel", (1.0/self.shadow_res, float(self.shadow_world_texel)))
        self._set(p, "uShadowC", 5); self._set(p, "uKeySet", float(self.light.c["key_on_set"]))
        self._set(p, "uSunDirC", tuple(float(x) for x in self.light.key_dir))
        self.light.bind(self.t_shadow, 5, 6)
        if self.ao is not None:                                 # shade-post (gated)
            self.ao.bind(p, 10)   # (12 = pbr.UNIT_FACE_CMP)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA, mgl.ZERO, mgl.ONE)
        self.vao_comp.render(mgl.TRIANGLE_STRIP)
        self.light.unbind()
        ctx.disable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
