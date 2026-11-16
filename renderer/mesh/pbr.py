"""shade: the physically based face path for a realistic character (`junor`).  GATED: everything here runs only
when the character's mesh config has a "pbr" block (renderer/animals/<char>.json "mesh": {"pbr": {...}}, or the
JUNO_MESH_CFG look-dev override).  Absent = MeshCharacter / MeshEngine take exactly the code path (juno, Muse and
every other character stay bit-identical).

What the block switches (each sub-key optional; a sub-block set to false keeps that part on the path):
  skin   mat_skin_pbr.glsl     dual-lobe GGX (area-key widened, Toksvig-filtered) + pore micro-normal (tiling, blended
                               with the asset's normal map) + cavity / specular occlusion + Burley-profile screen-space
                               SSS on IRRADIANCE with post-scatter albedo (freckles stay sharp) + shadow-map thickness
                               transmission + peach-fuzz sheen + region roughness (T-zone / lips) when the asset has
                               no roughness map + wrinkle-normal blending (asset wrinkle maps + per-shape masks)
  eye    mat_eye_pbr.glsl      iris seen THROUGH the cornea: the view ray refracted at the cornea sphere, marched to
                               the iris plane at its depth (parallax + refraction), limbal ring, caustic on the far side
                               of the iris, sclera with soft scatter + veins, lid contact AO / shadow over the ball
  cornea mat_cornea_pbr.glsl   GGX on the real cornea dome: key (area) + fill catchlights + the room probe, Fresnel
  mouth  mat_mouth_pbr.glsl    teeth (translucent enamel, per-tooth AO from the cavity depth, rough spec), gums and
                               tongue (wet spec), cavity darkened by depth behind the lip seam (never black)
  cloth  mat_cloth_pbr.glsl    (OPT-IN: only when "cloth" is present) knit with lattice-aligned tile shuffling (no repeat),
                               fuzz-softened diffuse relief + Charlie fibre layer (energy-conserving), yarn multiple
                               scattering (saturated cavities) + coloured multi-bounce AO, per-stitch / heather variation;
                               cotton blouse at a real thread pitch with slub, drape rumple, translucency, rough fibre spec;
                               neck contact AO, fold AO, PBR key shadow (head cascade).  knit_tint = documented hue knob.
  post   post_pbr.glsl         (engine.py) AgX-style filmic tone map, energy-preserving bloom, sRGB OETF
Config (all optional):
  "pbr": {"skin": {"micro": 0.6, "micro_tile": 28.0, "spec": 1.0, "rough": 1.0, "lobe2": 0.15, "fuzz": 0.35,
                   "cavity": 1.0, "trans": 1.0, "nrm_diffuse": 0.45, "region_rough": 1.0, "albedo_split": true,
                   "wrinkle": 1.0},
          "eye": {...}, "cornea": {...}, "mouth": {...}, "post": {...}}
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# material type -> the PBR shader file (renderer/mesh/mat_<file>.glsl) and the pbr sub-block that switches it
PBR_FILES = {
    "skin": ("skin_pbr", "skin"),
    "eye": ("eye_pbr", "eye"),
    "cornea": ("cornea_pbr", "cornea"),
    "eye_occlusion": ("eye_occlusion_pbr", "eye"),
    "tearline": ("tearline_pbr", "eye"),
    "teeth": ("mouth_pbr", "mouth"),
    "gums": ("mouth_pbr", "mouth"),
    "tongue": ("mouth_pbr", "mouth"),
    "mouth_interior": ("mouth_pbr", "mouth"),
    "cloth": ("cloth_pbr", "cloth"),
}
OPT_IN = {"cloth"}             # sub-blocks that must be present (not just "not false") to switch their materials

SKIN_DEFAULTS = dict(micro=0.6, micro_tile=28.0, spec=0.85, rough=1.0, lobe2=0.15, fuzz=0.35, cavity=1.0, trans=0.6,
                     nrm_diffuse=0.45, region_rough=1.0, albedo_split=True, wrinkle=1.0, key_size=1.0, amb_spec=1.0,
                     ao=1.0, rim=0.5, fill_spec=1.0, toksvig=0.5)
EYE_DEFAULTS = dict(ior=1.376, iris_depth_mm=2.6, limbus=0.7, caustic=1.0, sclera_veins=0.5, lid_ao=1.0, iris_gain=1.3,
                    sclera=0.42, lid_shadow=1.0, refract=1.0, iris_conc=1.0, scatter=1.0, occl=1.0, tear=3.0, iris_grade=1.0, pupil=0.75, socket=1.0, corner=1.0,
                    iris_coll=[0.175, 0.100, 0.042], iris_mid=[0.100, 0.060, 0.032], iris_rim=[0.064, 0.041, 0.026], iris_fibre=1.0)
CORNEA_DEFAULTS = dict(rough=0.035, key=1.0, env=1.8, fill=1.0, key_size=1.0, key_pull=0.5, fill_size=0.35, lid_occ=1.0, box=0.0, box_aspect=1.3, box_round=0.35, place=None)
# shade-cloth (mat_cloth_pbr.glsl; opt-in: "pbr": {"cloth": {...}}).  knit_tint = linear albedo multiplier of the
# knit / rib (the hue knob; [1,1,1] = the asset's albedo), bright_expo = exposure of bright (white) cloth.
CLOTH_DEFAULTS = dict(relief=1.1, shuffle=1.0, ms=1.6, stitch_var=1.0, fuzz=1.2, fuzz_rough=0.7, soft=0.65,
                      knit_scale=0.75, weave_scale=0.5, trans=0.5, neck=1.0, fold=1.0, knit_tint=[1.0, 1.0, 1.0],
                      bright_expo=1.0, neck_r=1.0, cotton_spec=1.0, wash=1.0, rim=0.5, halo=1.0, cotton_rough=0.4,
                      heather=1.0, self_shadow=1.5, albedo_flat=1.0, drape=2.5, slub=0.6, twill_sheen=0.4, stretch_fade=0.0,
                      stretch_at=1.7, cavity=1.15)
# shade2: mesh.pbr.light.spot = the key feathered off below the face (m from between the eyes): full to r0, floor at r1
SPOT_DEFAULTS = dict(r0=0.16, r1=0.55, floor=0.72, side=0.8, amb=0.9)
MOUTH_DEFAULTS = dict(teeth_sss=1.0, cavity=1.0, wet=1.0, teeth_rough=0.35, floor=0.10, tissue_min=[0.30, 0.07, 0.06], inner=1.0,
                      aperture=0.0, tongue_sat=1.0, lower=1.0, interprox=0.0, incisal=0.0, gumline=0.0, throat=0.02,
                      bounce=0.0, caruncle=0.0, caruncle_wet=0.0, buccal=0.0, tongue_wet=1.0, room=1.0, key_bounce=0.0, ap_soft=0.0, teeth_level=1.0,
                      gum_level=1.0, gum_sat=1.0)
# shade2-mouth: aperture.. tongue_wet = the aperture model (mat_mouth_pbr.glsl; aperture 0 = the r4 model, exactly)


def skin_on(pbr):
    return bool(pbr) and pbr.get("skin", True) is not False and os.path.exists(os.path.join(HERE, "mat_skin_pbr.glsl"))


def pbr_cfg(mcfg):
    """the "pbr" block of a mesh config, or None (= the path)."""
    p = (mcfg or {}).get("pbr")
    return dict(p) if isinstance(p, dict) else None


def material_file(pbr, mat_type, default_file):
    """the shader file a material type compiles with: its PBR file when the block switches it on."""
    if not pbr or mat_type not in PBR_FILES:
        return default_file, False
    f, key = PBR_FILES[mat_type]
    sub = pbr.get(key, key not in OPT_IN)
    if sub is False or sub is None or not os.path.exists(os.path.join(HERE, f"mat_{f}.glsl")):
        return default_file, False
    return f, True


def sub(pbr, key, defaults):
    d = dict(defaults)
    s = pbr.get(key) if pbr else None
    if isinstance(s, dict):
        d.update(s)
    return d


# ------------------------------------------------------------------------------------------------ the micro-normal map
def _tile_noise(n, freq, rng):
    """tileable value-ish noise: random lattice at `freq` cells across, bicubic-smoothed via FFT low-pass of white noise"""
    w = rng.standard_normal((n, n))
    fx = np.fft.fftfreq(n)[:, None]*n; fy = np.fft.fftfreq(n)[None, :]*n
    r = np.sqrt(fx*fx + fy*fy)
    F = np.fft.fft2(w)*np.exp(-(r/freq)**2)
    out = np.real(np.fft.ifft2(F))
    return (out - out.mean())/(out.std() + 1e-9)


def micro_height(n=512, seed=4557):
    """a tileable skin micro-relief height field in [0,1]-ish: pores (sharp round dimples on a jittered lattice, a few
    sizes), the fine criss-cross of skin lines (two oriented ridge families), and a little fine-grain noise.
    Physical scale: the map tiles every 1/micro_tile uv units (~1.5-2 cm on a head); pores ~0.15-0.35 mm."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = np.zeros((n, n))
    # pores: Poisson-ish points on a jittered grid, dimples with a soft rim
    for cells, depth, rad in ((44, 1.0, 2.1), (70, 0.6, 1.5), (110, 0.35, 1.1)):
        cs = n/cells
        gx, gy = np.meshgrid(np.arange(cells), np.arange(cells))
        px = ((gx + rng.uniform(0.15, 0.85, gx.shape))*cs).ravel()
        py = ((gy + rng.uniform(0.15, 0.85, gy.shape))*cs).ravel()
        keep = rng.uniform(size=px.shape) < 0.75
        px, py = px[keep], py[keep]
        rr = rad*rng.uniform(0.7, 1.3, px.shape)
        dd = depth*rng.uniform(0.6, 1.0, px.shape)
        R = int(np.ceil(rad*1.3*2.5)) + 1
        for cx, cy, r0, d0 in zip(px, py, rr, dd):
            ix, iy = int(cx), int(cy)
            xs = (np.arange(ix - R, ix + R + 1)) % n; ys = (np.arange(iy - R, iy + R + 1)) % n
            ox = np.arange(ix - R, ix + R + 1) - cx; oy = np.arange(iy - R, iy + R + 1) - cy
            d2 = (ox[None, :]**2 + oy[:, None]**2)/(r0*r0)
            dimple = -d0*np.exp(-d2*1.6) + 0.18*d0*np.exp(-(np.sqrt(d2) - 1.35)**2*3.0)
            h[np.ix_(ys, xs)] += dimple
    # skin lines: two families of fine ridges (+-35 deg) modulated by low-frequency noise (cells, not stripes)
    m1 = _tile_noise(n, 6.0, rng); m2 = _tile_noise(n, 6.0, rng)
    for ang, mod in ((np.radians(35.0), m1), (np.radians(-40.0), m2)):
        k = 2*np.pi*24/n
        # keep it tileable: integer wave numbers along x and y
        kx = np.round(np.cos(ang)*24); ky = np.round(np.sin(ang)*24)
        ph = 2*np.pi*(kx*x + ky*y)/n + 0.8*_tile_noise(n, 10.0, rng)
        ridge = 1.0 - np.abs(np.sin(ph))
        h += 0.22*(ridge**3 - 0.25)*np.clip(0.5 + 0.5*mod, 0.0, 1.0)
    h += 0.12*_tile_noise(n, 90.0, rng)
    # high-pass: no structure larger than ~1/16 tile (a mip-averaged tile must read flat, or the tile repeats as
    # centimetre blotches at bust distance)
    fx = np.fft.fftfreq(n)[:, None]*n; fy = np.fft.fftfreq(n)[None, :]*n
    H = np.fft.fft2(h)*(1.0 - np.exp(-(fx*fx + fy*fy)/(16.0**2)))
    h = np.real(np.fft.ifft2(H))
    h = (h - h.min())/(np.ptp(h) + 1e-9)
    return h


def micro_texture(n=512, strength=1.0, seed=4557):
    """RGBA8: rg = tangent-space normal xy (0.5 = flat), b = normal z, a = cavity (1 open, 0 in a pore) -- tileable."""
    h = micro_height(n, seed)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1))*0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0))*0.5
    k = 9.0*strength
    nx, ny, nz = -gx*k, -gy*k, np.ones_like(h)
    l = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx, ny, nz = nx/l, ny/l, nz/l
    cav = np.clip((h - np.percentile(h, 3))/(np.percentile(h, 60) - np.percentile(h, 3) + 1e-9), 0, 1)
    img = np.stack([nx*0.5 + 0.5, ny*0.5 + 0.5, nz*0.5 + 0.5, cav], -1)
    return (np.clip(img, 0, 1)*255 + 0.5).astype(np.uint8)


def micro_tex(mc, seed=4557):
    """the cached GL texture (generated once per process, ~1 s; also cached on disk next to this file)."""
    import moderngl
    path = os.path.join(os.path.expanduser("~"), ".cache", "juno_face", f"pbr_micro_{seed}_v2.npy")
    img = None
    try:
        img = np.load(path)
    except Exception:
        pass
    if img is None:
        img = micro_texture(512, 1.0, seed)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path + ".tmp.npy", img); os.replace(path + ".tmp.npy", path)
        except Exception:
            pass
    t = mc.ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(img[::-1]).tobytes())
    t.build_mipmaps()
    t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    t.repeat_x = t.repeat_y = True
    try:
        t.anisotropy = 8.0
    except Exception:
        pass
    return t


# ------------------------------------------------------------------------------------------------ per-frame uniforms
UNIT_MICRO = 13
UNIT_WRINKLE = 14
UNIT_WRINKLE_MASK = 16


class PbrFace:
    """MeshCharacter.pbr_face: the PBR materials' uniforms + textures (bound per forward program each frame)."""

    def __init__(self, mc, pbr):
        self.mc = mc
        self.pbr = pbr
        self.skin = sub(pbr, "skin", SKIN_DEFAULTS)
        self.eye = sub(pbr, "eye", EYE_DEFAULTS)
        self.cornea = sub(pbr, "cornea", CORNEA_DEFAULTS)
        self.mouth = sub(pbr, "mouth", MOUTH_DEFAULTS)
        self.cloth = sub(pbr, "cloth", CLOTH_DEFAULTS)
        self.t_micro = micro_tex(mc)
        self.detail = 1.0
        # the asset's skin roughness / spec: is the packed map's G a real roughness map (not a constant)?
        from eyesmouth import map_is_flat
        A = mc.A
        self.rough_map = 0.0
        for mid, md in (A.meta.get("materials", {}) or {}).items():
            md = md or {}
            if md.get("type") == "skin" and "head" in mid and md.get("packed"):
                self.rough_map = 0.0 if map_is_flat(os.path.join(A.dir, md["packed"]), 1) else 1.0
        # the asset's wrinkle maps (HEAD: materials.skin_head.params.wrinkle = {"normal": png, "mask": png RGBA,
        # "drive": {"r": {shape: gain}, "g": ..., "b": ..., "a": ...}}) -> uWrinkleW per frame from the shape weights
        self.wrinkle = None
        self.eyef = eye_frame(mc)                      # the eyeballs' rest gaze frame + iris geometry (or None)
        for mid, md in (A.meta.get("materials", {}) or {}).items():
            wr = ((md or {}).get("params") or {}).get("wrinkle")
            if (md or {}).get("type") == "skin" and isinstance(wr, dict) and wr.get("normal") and wr.get("mask"):
                tn = mc._tex(os.path.join(A.dir, wr["normal"])); tm = mc._tex(os.path.join(A.dir, wr["mask"]))
                if tn is not None and tm is not None:
                    names = {n: i for i, n in enumerate(A.shape_names)}
                    drive = []
                    for ch in "rgba":
                        drive.append([(names[k], float(g)) for k, g in (wr.get("drive", {}).get(ch) or {}).items() if k in names])
                    self.wrinkle = (tn, tm, drive)
                    break
        self.rig = None

    def uniforms(self):
        s, e, c, m = self.skin, self.eye, self.cornea, self.mouth
        det = float(self.detail)
        return {
            "uPbrSkin": (float(s["micro"])*det, float(s["micro_tile"]), float(s["spec"]), float(s["rough"])),
            "uPbrSkin2": (float(s["lobe2"]), float(s["fuzz"]), float(s["cavity"]), float(s["trans"])),
            "uPbrSkin3": (float(s["nrm_diffuse"]), float(s["region_rough"])*(1.0 - self.rough_map),
                          float(s["key_size"]), float(s["amb_spec"])),
            "uPbrSkin4": (float(s["ao"]), float(s["rim"]), float(s["fill_spec"]), float(s["wrinkle"])),
            "uPbrEye": (float(e["ior"]), float(e["iris_depth_mm"])*1e-3, float(e["limbus"]), float(e["caustic"])),
            "uPbrEye2": (float(e["sclera_veins"]), float(e["lid_ao"]), float(e["iris_gain"]), float(e["sclera"])),
            "uPbrEye3": (float(e["lid_shadow"]), float(e["refract"]), float(e["iris_conc"]), float(e["scatter"])),
            "uPbrEye4": (float(e["occl"]), float(e["tear"]), float(e["iris_grade"]), float(e["pupil"])),
            "uPbrEye5": (float(e["socket"]), float(e["corner"]), 0.0, 0.0),
            "uPbrIrisC": tuple(float(x) for x in e["iris_coll"][:3]), "uPbrIrisM": tuple(float(x) for x in e["iris_mid"][:3]),
            "uPbrIrisR": tuple(float(x) for x in e["iris_rim"][:3]), "uPbrIrisF": float(e["iris_fibre"]),
            "uPbrCornea": (float(c["rough"]), float(c["key"]), float(c["env"]), float(c["fill"])),
            "uPbrCornea4": ((float(c["place"][0]), float(c["place"][1]), 1.0, 0.0) if c.get("place") else (0.0, 0.0, 0.0, 0.0)),
            "uPbrCornea3": (float(c["box"]), float(c["box_aspect"]), float(c["box_round"]), 0.0),
            "uPbrCornea2": (float(c["key_size"]), float(c["key_pull"]), float(c["fill_size"]), float(c["lid_occ"])),
            "uPbrMouth": (float(m["teeth_sss"]), float(m["cavity"]), float(m["wet"]), float(m["teeth_rough"])),
            "uPbrMouthIn": float(m["inner"]),
            "uPbrMouth3": (float(m["aperture"]), float(m["tongue_sat"]), float(m["lower"]), float(m["interprox"])),
            "uPbrMouth4": (float(m["incisal"]), float(m["gumline"]), float(m["throat"]), float(m["bounce"])),
            "uPbrMouth5": (float(m["caruncle"]), float(m["caruncle_wet"]), float(m["buccal"]), float(m["tongue_wet"])),
            "uPbrMouth7": (float(m["gum_level"]), float(m["gum_sat"]), 0.0, 0.0),
            "uPbrMouth6": (float(m["room"]), float(m["key_bounce"]), float(m["ap_soft"]), float(m["teeth_level"])),
            "uPbrToks": float(s["toksvig"]),
            "uPbrMouth2": (float(m["floor"]),) + tuple(float(x) for x in m["tissue_min"][:3]),
            **self.cloth_uniforms(),
        }

    def cloth_uniforms(self):
        k = self.cloth
        t = [float(x) for x in (list(k["knit_tint"]) + [1.0, 1.0, 1.0])[:3]]
        return {
            "uPbrCloth": (float(k["relief"]), float(k["shuffle"]), float(k["ms"]), float(k["stitch_var"])),
            "uPbrCloth2": (float(k["fuzz"]), float(k["fuzz_rough"]), float(k["soft"]), float(k["knit_scale"])),
            "uPbrCloth3": (float(k["weave_scale"]), float(k["trans"]), float(k["neck"]), float(k["fold"])),
            "uPbrCloth4": (t[0], t[1], t[2], float(k["bright_expo"])),
            "uPbrCloth5": (float(k["neck_r"]), float(k["cotton_spec"]), float(k["wash"]), float(k["rim"])),
            "uPbrCloth6": (float(k["halo"]), float(k["cotton_rough"]), float(k["heather"]), float(k["self_shadow"])),
            "uPbrCloth7": (float(k["albedo_flat"]), float(k["drape"]), float(k["slub"]), float(k["twill_sheen"])),
            "uPbrCloth8": (float(k["stretch_fade"]), float(k["stretch_at"]), float(k["cavity"]), 0.0),
        }

    def wrinkle_w(self):
        if self.wrinkle is None or self.rig is None:
            return (0.0, 0.0, 0.0, 0.0)
        w = np.asarray(self.rig.weights, float)
        return tuple(float(np.clip(sum(g*w[i] for i, g in ch if i < len(w)), 0.0, 1.0)) for ch in self.wrinkle[2])

    def apply(self, prog, rig=None):
        if rig is not None:
            self.rig = rig
        if "uWrinkleW" in prog:
            on = self.wrinkle is not None
            prog["uWrinkleW"].value = self.wrinkle_w()
            if "uWrinkleOn" in prog:
                prog["uWrinkleOn"].value = 1.0 if on else 0.0
            if on:
                self.wrinkle[0].use(UNIT_WRINKLE); self.wrinkle[1].use(UNIT_WRINKLE_MASK)
                for k, u in (("tWrinkleN", UNIT_WRINKLE), ("tWrinkleM", UNIT_WRINKLE_MASK)):
                    if k in prog:
                        prog[k].value = u
        for k, v in self.uniforms().items():
            if k in prog:
                prog[k].value = v
        if rig is not None and self.eyef is not None and "uEyeGZ" in prog:
            for k, v in self.eyef.frame(rig).items():
                if k in prog:
                    prog[k].write(np.ascontiguousarray(v, np.float32).tobytes())
        if "tMicro" in prog:
            prog["tMicro"].value = UNIT_MICRO
            self.t_micro.use(UNIT_MICRO)
        if "uPbrSpot" in prog and rig is not None:          # shade2: mesh.pbr.light.spot (portrait key falloff)
            sp = dict(SPOT_DEFAULTS); sp.update((self.pbr.get("light") or {}).get("spot") or {})
            try:
                u = _face_frame(self.mc, rig)
                ec = np.asarray(u["uEyeC"], float).reshape(2, 3).mean(0)
            except Exception:
                ec = np.zeros(3)
            prog["uPbrSpot"].value = (float(ec[0]), float(ec[1]), float(ec[2]), float(sp["r0"]))
            prog["uPbrSpot2"].value = (float(sp["r1"]), float(sp["floor"]), float(sp["side"]), float(sp["amb"]))


# ------------------------------------------------------------------------------------------------ the eyes' frame
class EyeFrame:
    """Each eyeball's gaze frame + iris geometry, measured once from the asset (rest pose), posed per frame by the eye
    bones.  The PBR eye shaders model the cornea as a sphere cap over the iris (radius from the iris radius and the
    anterior-chamber depth) and need, per eye in WORLD units:
      uEyeGX[2], uEyeGY[2], uEyeGZ[2]  the eye's frame (GZ = the gaze axis = eye centre -> the pupil centre uv .5,.5;
                                        GY ~ the head's up, GX = GY x GZ), unit vectors
      uEyeUV[2]  vec4 (a, b, c, d): uv - 0.5 = [[a, b], [c, d]] . (x, y)  where (x, y) = the frame's (GX, GY)
                                        components of a point's direction from the centre, scaled to its angle from the
                                        gaze axis (radians) -- the asset's iris uv layout (mirrored eyes included)
      uEyeIris   vec4 (iris radius, limbus plane: axial distance from the centre, iris dish depth, iris half-angle rad)
    """

    def __init__(self, A, fm, iris_uv):
        P = A.mesh["position"].astype(np.float64)
        UV = A.mesh["uv0"].astype(np.float64)
        I = A.mesh["index"]
        mats = A.meta.get("materials", {}) or {}
        self.b_eye = list(fm.b_eye)
        self.c0, self.g0, self.x0, self.y0, self.M = [], [], [], [], []
        rI, zL, zC, th = [], [], [], []
        up = np.array([0.0, 1.0, 0.0])
        for k, side in enumerate(("l", "r")):
            subs = [s for s in A.submeshes if (mats.get(s["material"], {}) or {}).get("type", s["material"]) == "eye"
                    and s["name"].endswith("_" + side)]
            if not subs:
                raise ValueError("no eye submesh")
            s = subs[0]
            idx = np.unique(I[s["first_index"]:s["first_index"] + s["index_count"]])
            p, uv = P[idx], UV[idx]
            c = np.asarray(fm.eye_c[k], np.float64)
            du = np.linalg.norm(uv - 0.5, axis=1)
            j = np.argsort(du)[:6]
            g = (p[j] - c).mean(0); g /= np.linalg.norm(g)
            y = up - g*np.dot(up, g); y /= np.linalg.norm(y)
            x = np.cross(y, g)
            d = p - c
            ax = d @ g
            perp = np.stack([d @ x, d @ y], 1)
            pl = np.linalg.norm(perp, axis=1)
            ang = np.arctan2(pl, ax)
            X = perp/np.maximum(pl, 1e-12)[:, None]*ang[:, None]
            m = du < 1.4*iris_uv
            M, *_ = np.linalg.lstsq(X[m], uv[m] - 0.5, rcond=None)      # (2 x 2): uv-.5 = X @ M
            rim = np.abs(du - iris_uv) < 0.12*iris_uv
            rI.append(float(np.median(pl[rim]))); zL.append(float(np.median(ax[rim])))
            zC.append(float(np.median(ax[j]))); th.append(float(np.median(ang[rim])))
            self.c0.append(c); self.g0.append(g); self.x0.append(x); self.y0.append(y); self.M.append(M.T)
        self.r_iris = float(np.mean(rI)); self.z_limbus = float(np.mean(zL))
        self.dish = float(max(np.mean(zL) - np.mean(zC), 0.0)); self.theta = float(np.mean(th))

    def frame(self, rig):
        B = np.asarray(rig.bones, np.float64)
        out = {"uEyeGX": [], "uEyeGY": [], "uEyeGZ": [], "uEyeUV": []}
        s = 1.0
        for k in range(2):
            b = self.b_eye[k]
            Sk = B[b] @ self.A_inv[b]
            R = Sk[:3, :3]
            s = float(np.linalg.norm(R[:, 0]))
            for nm, v in (("uEyeGX", self.x0[k]), ("uEyeGY", self.y0[k]), ("uEyeGZ", self.g0[k])):
                w = R @ v
                out[nm].extend(w/max(np.linalg.norm(w), 1e-12))
            out["uEyeUV"].extend(self.M[k].reshape(-1))
        out = {k: np.asarray(v, np.float32) for k, v in out.items()}
        out["uEyeIris"] = np.asarray([self.r_iris*s, self.z_limbus*s, self.dish*s, self.theta], np.float32)
        return out


def eye_frame(mc):
    try:
        fm = mc.face.fm
        iris_uv = next((d["u"]["uEyeP"][0] for d in mc.draws if d["type"] == "eye"), None)
        if iris_uv is None:
            return None
        ef = EyeFrame(mc.A, fm, float(iris_uv))
        ef.A_inv = mc.inv_bind
        return ef
    except Exception as ex:                    # no eyes / odd asset: the shaders fall back to the mesh normal frame
        print(f"[pbr] eye frame unavailable: {ex}", flush=True)
        return None


# ------------------------------------------------------------------------------------------------ the face shadow cascade
FACE_SHADOW_DEFAULTS = dict(res=1024, radius_m=0.15, taps=16, on=True, lift_m=0.0)
UNIT_FACE_RAW = 15
UNIT_FACE_CMP = 12


def _face_frame(mc, rig):
    fm = mc.face.fm
    if fm._last is None or fm._last[0] is not rig:
        fm._last = (rig, fm.frame(rig))
    return fm._last[1]


class FaceShadow:
    """shade (mesh.pbr.light.face_shadow): a second orthographic key shadow map around the HEAD only (res^2 over
    ~30 cm: ~0.3 mm texels vs ~1 mm for the body map) -> the nose / lids / lips / ear contact shadows are crisp near
    the caster and soften with distance (PCSS with the same key angular size), the lid's shadow on the eyeball reads.
    The PBR materials take min(body map, face map) inside it (the body map still carries the hair's shadow)."""

    def __init__(self, mc, cfg):
        import moderngl
        self.mgl, self.mc, self.ctx = moderngl, mc, mc.ctx
        c = dict(FACE_SHADOW_DEFAULTS); c.update(cfg or {})
        self.c = c
        self.res = int(c["res"])
        self.t = self.ctx.depth_texture((self.res, self.res))
        self.t.compare_func = ""
        self.t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t.repeat_x = self.t.repeat_y = False
        self.fb = self.ctx.framebuffer(depth_attachment=self.t)
        self.sampler = self.ctx.sampler(filter=(moderngl.LINEAR, moderngl.LINEAR), compare_func="<=",
                                        repeat_x=False, repeat_y=False)
        self.vp = np.eye(4); self.texel = 1e-3; self.range = 1.0; self.on = False

    def fit(self, centre, r, key_dir):
        L = np.asarray(key_dir, float); L = L/np.linalg.norm(L)
        f = -L
        up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
        rt = np.cross(f, up); rt /= np.linalg.norm(rt)
        u2 = np.cross(rt, f)
        texel = 2.0*r/self.res
        cx = np.round(np.dot(rt, centre)/texel)*texel        # texel-snapped: no crawl as she moves
        cy = np.round(np.dot(u2, centre)/texel)*texel
        R = 4.0*r                                            # depth: the whole head + what is in front of it
        cz = np.dot(f, centre) - 0.5*r
        M = np.eye(4)
        M[0, :3] = rt/r; M[0, 3] = -cx/r
        M[1, :3] = u2/r; M[1, 3] = -cy/r
        M[2, :3] = f/R; M[2, 3] = -cz/R
        self.vp, self.texel, self.range = M, texel, 2.0*R
        return M

    def render(self, rig, key_dir):
        mc, ctx, mgl = self.mc, self.ctx, self.mgl
        if not self.c.get("on", True):
            self.on = False
            return
        u = _face_frame(mc, rig)
        ec = np.asarray(u["uEyeC"], float).reshape(2, 3).mean(0)
        R = np.asarray(u["uHeadR"], float).reshape(3, 3).T
        s = float(u["uHeadS"])
        centre = ec - R[:, 2]*0.05*s - R[:, 1]*(0.02 - float(self.c["lift_m"]))*s
        M = self.fit(centre, float(self.c["radius_m"])*s, key_dir)
        mc.p_shadow["uLightVP"].write(np.ascontiguousarray(M.T, np.float32).tobytes())
        self.fb.use(); ctx.viewport = (0, 0, self.res, self.res)
        self.fb.clear(depth=1.0)
        ctx.enable(mgl.DEPTH_TEST); ctx.depth_func = "<"
        ctx.disable(mgl.CULL_FACE)
        for d in mc.draws:
            if d["order"] == 0 and d["type"] not in ("mouth_interior",):
                sb = d["sub"]
                mc.vao_shadow.render(mgl.TRIANGLES, vertices=sb["index_count"], first=sb["first_index"])
        ctx.disable(mgl.DEPTH_TEST)
        mc.p_shadow["uLightVP"].write(np.ascontiguousarray(mc.light_vp.T, np.float32).tobytes())
        self.on = True

    def bind(self):
        self.t.use(UNIT_FACE_RAW)
        self.sampler.texture = self.t
        self.sampler.use(UNIT_FACE_CMP)

    def unbind(self):
        try:
            self.sampler.clear(UNIT_FACE_CMP)
        except Exception:
            pass

    def uniforms(self, prog):
        if "uFaceVP" in prog:
            prog["uFaceVP"].write(np.ascontiguousarray(self.vp.T, np.float32).tobytes())
        for k, v in (("uFaceSh", UNIT_FACE_RAW), ("uFaceShC", UNIT_FACE_CMP),
                     ("uFaceShP", (1.0 if self.on else 0.0, float(self.texel), float(self.range), float(self.c["taps"])))):
            if k in prog:
                prog[k].value = v


# ------------------------------------------------------------------------------------------------ shade-cloth
def cloth_on(pbr):
    """mesh.pbr.cloth present (opt-in) and the shader exists."""
    return bool(pbr) and isinstance(pbr.get("cloth"), (dict, bool)) and pbr.get("cloth") is not False \
        and os.path.exists(os.path.join(HERE, "mat_cloth_pbr.glsl"))


def knit_relief_v2(N=12, M=16, P=32, rib=False, seed=7, tilt=20.0, groove=0.35, halo=1.2):
    """Stockinette / k2p2 rib relief for the PBR cloth: the same lattice / tile as cloth._knit (12 x 16 stitches, row
    pitch .75 / .70 -> drop-in, the shader's shuffle parity holds) but the legs stand more upright (tilt ~20 deg, real
    stockinette: columns of V's, not a herringbone of slanted strokes), rounder (elliptic yarn section), a groove
    between the columns, the lower leg end tucked under the row below, a plied twist, and a fibre halo (the height
    edge blurred + fine fibre noise) so stitches have soft fuzzy borders instead of a cut-out outline."""
    from scipy.ndimage import gaussian_filter
    rh = 0.75 if not rib else 0.70
    W, H = N*P, int(round(M*rh*P))
    ys, xs = (np.mgrid[0:H, 0:W] + 0.5)/P
    r = np.random.default_rng(seed)
    jw = r.normal(0, 1, (M, N, 2, 4))
    cvar = r.uniform(0, 1, (M, N))
    rowv = r.normal(0, 1, M)
    colv = r.normal(0, 1, N)
    h = np.zeros((H, W)); var = np.full((H, W), 0.5)
    ci0 = np.floor(xs).astype(int); cj0 = np.floor(ys/rh).astype(int)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            ci = ci0 + di; cj = cj0 + dj
            i = ci % N; j = cj % M
            cx = ci + 0.5; y0 = cj*rh
            purl = (i % 4 >= 2) if rib else np.zeros_like(i, bool)
            for leg, sgn in ((0, -1.0), (1, 1.0)):
                J = jw[j, i, leg]
                Jw, Jt, Jh, Jo = J[..., 0], J[..., 1], J[..., 2], J[..., 3]
                th = np.radians((tilt if not rib else 14.0) + 3.5*Jt)*sgn
                axx, axy = np.sin(th), np.cos(th)
                kcx = cx + sgn*(0.215 if not rib else 0.19) + 0.03*Jo; kcy = y0 + (0.55 + 0.03*Jh)*rh
                ka = 0.70*rh/0.75; kb = (0.225 if not rib else 0.20)*(1.0 + 0.10*Jw + 0.04*colv[i]); khs = 1.0 + 0.08*Jh
                pcx = cx + 0.18*np.where(j % 2 == 1, 1.0, -1.0)*sgn; pcy = y0 + 0.5*rh
                pb = 0.26*rh/0.75*(1.0 + 0.07*Jw)
                axx = np.where(purl, 1.0, axx); axy = np.where(purl, 0.0, axy)
                ccx = np.where(purl, pcx, kcx); ccy = np.where(purl, pcy, kcy)
                a = np.where(purl, 0.36, ka); b = np.where(purl, pb, kb); hs = np.where(purl, 0.5, khs)
                dx = xs - ccx; dy = ys - ccy
                u = dx*axx + dy*axy
                w = dx*axy - dy*axx
                un = u/a
                bb = b*(1.0 - 0.22*un**2)
                prof = np.clip(1.0 - (w/np.maximum(bb, 1e-3))**2, 0, 1)**0.42          # round yarn section
                along = np.sqrt(np.clip(1.0 - np.abs(un)**3, 0, 1))
                tuck = 1.0 - np.where(purl, 0.0, 0.30)*np.clip(-un, 0, 1)            # lower end dives under
                hl = prof*along*hs*tuck
                ply = np.cos(2*np.pi*(u*4.6 + w/np.maximum(b, 1e-3)*1.4))
                hl = hl*(1.0 + 0.06*ply*prof)
                m = hl > h
                h = np.where(m, hl, h)
                var = np.where(m, 0.5 + 0.42*(cvar[j, i] - 0.5) + 0.05*ply + 0.08*np.tanh(rowv[j]*0.5)
                               + 0.05*np.tanh(colv[i]*0.5), var)
    # the column groove: yarn running behind between the V columns sits lower
    gx = np.abs((xs % 1.0) - 0.5)*2.0                     # 0 mid-column .. 1 at the column border
    h = h*(1.0 - groove*np.clip((gx - 0.75)/0.25, 0, 1)**2)
    if rib:
        h = np.maximum(h, 0.18 + 0.05*np.cos(2*np.pi*ys/rh*2))
    # fibre halo: soften the yarn borders, then fine fibre noise riding on top (and a little standing proud of the gaps)
    hb = gaussian_filter(h, halo, mode="wrap")
    h = np.maximum(h, hb*0.92)
    rng2 = np.random.default_rng(seed + 1)
    fz = gaussian_filter(rng2.normal(0, 1, (H, W)), 0.7, mode="wrap")
    fz2 = gaussian_filter(rng2.normal(0, 1, (H, W)), (0.5, 2.5), mode="wrap")          # short fibres, oriented
    fz = fz/fz.std()*0.6 + fz2/fz2.std()*0.4
    h = h + 0.06*fz*np.clip(h*3.0, 0.3, 1.0)
    var = var + 0.04*fz
    return h, var, (N, M*rh), P


_V2 = {}


def knit_v2_data(kind, cfg=None):
    """(rgba float32 [H,W,4] = gx, gy, h, var ; tile) like cloth.relief, for knit / rib with the v2 generator."""
    c = dict(tilt=20.0, groove=0.35, halo=1.2)
    c.update({k: float(v) for k, v in (cfg or {}).items() if k in c})
    key = (kind, tuple(sorted(c.items())))
    if key in _V2:
        return _V2[key]
    from scipy.ndimage import gaussian_filter
    h, var, tile, P = knit_relief_v2(rib=(kind == "rib"), seed=9 if kind == "rib" else 7, **c)
    h = gaussian_filter(h, 0.6, mode="wrap")
    h = (h - h.min())/max(h.max() - h.min(), 1e-6)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1))*0.5*P
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0))*0.5*P
    out = np.stack([gx, gy, h, np.clip(var, 0, 1)], -1).astype(np.float32)
    _V2[key] = (out, tile)
    return _V2[key]


def cloth_setup(char, u, texs, mid, md, sub, pbr):
    """shade-cloth: after cloth.setup, on the PBR cloth path only.  knit / rib -> the v2 relief (mesh.pbr.cloth.
    knit_v2: true (default) | false | {tilt, groove, halo}); the mean cavity follows the new relief."""
    import cloth as _cloth
    cc = pbr.get("cloth") if isinstance(pbr.get("cloth"), dict) else {}
    kv = cc.get("knit_v2", True)
    info = (getattr(char, "cloth_info", {}) or {}).get(mid, {})
    kind = info.get("kind")
    if kv is False or kind not in ("knit", "rib"):
        return
    data, tile = knit_v2_data(kind, kv if isinstance(kv, dict) else None)
    key = ("cloth_relief_v2", kind, id(data))
    if key not in char.tex_cache:
        t = char.ctx.texture((data.shape[1], data.shape[0]), 4, np.ascontiguousarray(data.astype(np.float16)).tobytes(),
                             dtype="f2")
        t.build_mipmaps()
        t.filter = (char.mgl.LINEAR_MIPMAP_LINEAR, char.mgl.LINEAR)
        try:
            t.anisotropy = 8.0
        except Exception:
            pass
        char.tex_cache[key] = t
    texs[7] = char.tex_cache[key]
    tt = data[..., 2]
    ss = np.clip(tt/0.8, 0, 1); ss = ss*ss*(3 - 2*ss)
    cav = _cloth.KIND_DFL[kind][6]
    w = list(u["uClothW"]); w[2] = float(np.mean(1.0 + cav*((0.35 + 0.65*ss) - 1.0))); u["uClothW"] = tuple(w)
    t0 = list(u["uClothT"]); t0[0] = 1.0/float(tile[0]); t0[1] = 1.0/float(tile[1]); u["uClothT"] = tuple(t0)
