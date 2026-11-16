"""render-light: Juno's light terms on the CPU side (renderer/mesh/light.glsl owns the shader half).

    lt = MeshLight(ctx, mcfg)           # MeshCharacter creates it; knobs from juno.json "mesh" -> "light"
    M, info = lt.fit(bound, key_dir, res)   # texel-snapped, radius-quantised orthographic key projection
    lt.uniforms(prog, rf)               # PCSS / capsule AO / probe uniforms for a forward or composite program
    lt.bind(4, 5, 6) / lt.unbind()      # compare sampler on the shadow map + the probe texture
    lt.bake_probe(eng, rf)              # once: the room's radiance at Juno's chest from the SDF set (cached)

Knobs (juno.json mesh.light, every one optional; set_quality() is the ladder's entry):
    pcss_taps   16 (8 low, 0 = the legacy fixed-radius PCF)       ~ 12 blocker + 2xN hardware-2x2 taps per pixel
    key_tan     0.06  angular radius (tan) of the key: the penumbra per metre of blocker distance
    pen_max     22    max penumbra (shadow texels)
    cap_ao      0.8   capsule AO strength on the mesh (0 = off)     18 capsules, ~20 ALU each per pixel
    ground_ao   1.3   her capsule contact occlusion on the scene (floor / desk / chair) in the composite
    key_on_set  0.6   her key shadow (shadow map, 10 soft taps) darkening the scene where it faces the key (the set's
                      own live shadow only shows inside the window-beam pools)
    probe       1     the baked radiance probe for specular (0 = the analytic envSpec)
    probe_res   32    cube face size of the bake (6 x res^2 rays of the set's own PASS_BG, once)
    probe_diffuse 0.35 share of the probe's cosine irradiance in the mesh's ambient (the room's warm bounce colour and
                  direction at setAmbient's level; 0 = setAmbient only, as the SDF characters)
    probe_gain  1.0   x the calibration that matches the probe's irradiance to setAmbient (the room level of Muse)
"""
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NCAP = 20                        # light.glsl uCap: the rig's 18 proxies + the jaw (from the asset's chin/mouth landmarks)

DEFAULTS = dict(term_bias=0.8, pen_min=1.0, pcss_taps=16, key_tan=0.06, pen_max=22.0, cap_ao=0.8, cap_floor=0.35, ground_ao=1.3, key_on_set=0.6, probe=1, probe_diffuse=0.35,
                probe_res=32, probe_spp=4, probe_gain=1.0, probe_w=64)
QUALITY = {                      # ladder: name -> (shadow_res, pcss_taps, probe)
    "high": (2048, 16, 1),
    "mid": (2048, 8, 1),
    "low": (1024, 8, 1),
    "min": (1024, 0, 0),
}
ROUGH_LAYERS = (0.0, 0.25, 0.5, 1.0)
PROBE_VERSION = 1


class MeshLight:
    def __init__(self, ctx, mcfg=None, asset=None):
        import moderngl
        self.mgl, self.ctx = moderngl, ctx
        c = dict(DEFAULTS)
        c.update((mcfg or {}).get("light", {}) or {})
        env = os.environ.get("JUNO_LIGHT")          # dev override: JUNO_LIGHT='{"pcss_taps":8}'
        if env:
            c.update(json.loads(env))
        self.c = c
        self.asset = asset
        self._jaw = None                 # (head bone index, mouth, chin) in bind space: the jaw capsule (landmarks)
        if asset is not None:
            lm = asset.meta.get("landmarks", {}) or {}
            bi = {b["name"]: i for i, b in enumerate(asset.bones)}
            if "head" in bi and len(lm.get("chin", [])) >= 3 and len(lm.get("mouth", [])) >= 3:
                self._jaw = (bi["head"], np.array(lm["mouth"][:3], float), np.array(lm["chin"][:3], float))
        self.snap = None                 # (radius quantum) -> stable between frames
        self.range = 1.0
        self.sampler = None
        self.t_probe = None
        self.probe_info = {}
        self.probe_calib = 1.0
        self.caps = np.zeros((2*NCAP, 4), np.float32)
        self.ncap = 0

    # ------------------------------------------------------------------ quality ladder
    def set_quality(self, name):
        """returns the shadow resolution the ladder wants (MeshCharacter reallocates the map when it changes)."""
        res, taps, probe = QUALITY.get(name, QUALITY["high"])
        self.c["pcss_taps"] = taps
        self.c["probe"] = probe
        return res

    # ------------------------------------------------------------------ the key's projection
    def fit(self, bound, key_dir, res):
        """orthographic light view-projection round the rig's bounding sphere, STABLE: the radius is quantised to
        6 % steps (it changes only when the bound really grows) and the centre is snapped to whole shadow texels in
        the light's plane, so a still or a slowly moving body does not make the shadow edges crawl."""
        c = np.asarray(bound[:3], float); r0 = float(bound[3])*1.04
        q = np.exp(np.ceil(np.log(max(r0, 1e-3))/np.log(1.06))*np.log(1.06))
        r = float(q)
        L = np.asarray(key_dir, float); L = L/np.linalg.norm(L)
        self.key_dir = L
        f = -L
        up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
        rt = np.cross(f, up); rt /= np.linalg.norm(rt)
        u2 = np.cross(rt, f)
        texel = 2.0*r/res
        cx = np.round(np.dot(rt, c)/texel)*texel
        cy = np.round(np.dot(u2, c)/texel)*texel
        cz = np.dot(f, c)
        R = 2.0*r
        M = np.eye(4)
        M[0, :3] = rt/r; M[0, 3] = -cx/r
        M[1, :3] = u2/r; M[1, 3] = -cy/r
        M[2, :3] = f/R; M[2, 3] = -cz/R
        self.range = 2.0*R                    # world depth spanned by window depth 0..1
        self.texel = texel
        return M, texel

    # ------------------------------------------------------------------ samplers
    def bind(self, t_shadow, unit_cmp=5, unit_probe=6):
        mgl = self.mgl
        if self.sampler is None:
            self.sampler = self.ctx.sampler(filter=(mgl.LINEAR, mgl.LINEAR), compare_func="<=",
                                            repeat_x=False, repeat_y=False)
        self.sampler.texture = t_shadow
        self.sampler.use(unit_cmp)
        if self.t_probe is not None:
            self.t_probe.use(unit_probe)
        self._bound_unit = unit_cmp

    def unbind(self):
        """the compare sampler must not leak into the engine's later passes on that unit."""
        if self.sampler is not None:
            try:
                self.sampler.clear(getattr(self, "_bound_unit", 5))
            except Exception:
                pass

    # ------------------------------------------------------------------ per-frame uniforms
    def capsules(self, rf):
        prox = np.asarray(getattr(rf, "proxies", None) if getattr(rf, "proxies", None) is not None else [], np.float32)
        cap = np.zeros((2*NCAP, 4), np.float32)
        n = min(len(prox), NCAP - 1)
        if n:
            cap[0:2*n:2, :3] = prox[:n, 0:3]; cap[0:2*n:2, 3] = prox[:n, 6]
            cap[1:2*n:2, :3] = prox[:n, 3:6]; cap[1:2*n:2, 3] = prox[:n, 7]
        # the jaw: the rig's head capsule sits in the skull; the chin over the neck/collar needs its own (kind 9)
        if self._jaw is not None and getattr(rf, "bones", None) is not None:
            hb, mo, ch = self._jaw
            K = np.asarray(rf.bones[hb], float) @ self.asset.inv_bind[hb].astype(float)
            a = (K @ np.r_[mo, 1.0])[:3]; b = (K @ np.r_[ch, 1.0])[:3]
            r = 0.42*float(np.linalg.norm(b - a)) + 0.012
            a = a + (a - b)*0.15
            cap[2*n, :3] = a; cap[2*n, 3] = r; cap[2*n + 1, :3] = b - (b - a)/max(np.linalg.norm(b - a), 1e-6)*r*0.5; cap[2*n + 1, 3] = 9
            n += 1
        self.caps, self.ncap = cap, n

    def uniforms(self, prog, rf=None, units=(5, 6)):
        c = self.c
        S = lambda k, v: (k in prog) and setattr(prog[k], "value", v)
        S("uShadowC", units[0]); S("uProbe", units[1])
        S("uLightB", float(c["term_bias"])); S("uLightB2", float(c["pen_min"]))   # render2
        S("uLightP", (float(c["key_tan"]), float(self.range), float(c["pen_max"]), float(c["pcss_taps"])))
        if "uCap" in prog:
            prog["uCap"].write(self.caps.tobytes())
        S("uCapP", (float(c["cap_ao"]), float(c["cap_floor"]), float(self.ncap), float(c["ground_ao"])))
        on = 1.0 if (self.t_probe is not None and int(c["probe"])) else 0.0
        S("uProbeP", (on, float(c["probe_gain"])*self.probe_calib, float(c["probe_diffuse"]), float(c.get("debug", 0))))

    # ------------------------------------------------------------------ the radiance probe
    def chest(self, rf):
        prox = np.asarray(rf.proxies, float)
        tor = [p for p in prox if int(round(p[7])) == 2]
        if tor:
            t = tor[0]
            top = t[3:6] if t[4] > t[1] else t[0:3]
            return top + np.array([0.0, -0.05, 0.12])
        b = np.asarray(rf.bound, float)
        return b[:3] + np.array([0.0, 0.25*b[3], 0.1])

    def bake_probe(self, eng, rf, log=False):
        """6 cube faces of the room from Juno's chest with the engine's own background program (PASS_BG BG_NO_CHAR: the
        set + places, the bake's light and haze), radiance = ambient layer + sun layer x the live pass's mean cookie;
        prefiltered on the CPU into a lat-long array (4 roughness layers).  Cached under $TA_OUT/<animal>/."""
        if not int(self.c["probe"]):
            return
        import rt_engine
        import render
        ctx, mgl = self.ctx, self.mgl
        res, spp = int(self.c["probe_res"]), int(self.c["probe_spp"])
        pos = np.round(self.chest(rf), 2)
        src = eng.sh_bg.frag("scene_main.glsl", "PASS_BG", "BG_NO_CHAR")
        key = hashlib.sha1((src + repr((PROBE_VERSION, pos.tolist(), res, spp, eng.cfg.get("set"),
                                         tuple(np.round(eng.sun, 4)), tuple(eng.bg_sun)))).encode()).hexdigest()[:16]
        d = os.path.join(render.OUT_ROOT, eng.animal)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"probe_{key}.npz")
        t0 = time.time()
        cube = None
        if os.path.exists(path):
            z = np.load(path)
            cube, amb = z["cube"], z["amb"]
            if not (np.isfinite(cube).all() and np.isfinite(amb).all()):   # render6: a bad cached bake is re-baked
                print(f"[light] probe cache {path} has non-finite values: re-baking", flush=True)
                cube = None
        if cube is None:
            p = ctx.program(vertex_shader=rt_engine.VS_FULL, fragment_shader=src)
            eng._cam_uniforms(p)
            for k, v in (("uTime", 0.0), ("uFrame", 0)):
                if k in p:
                    p[k].value = v
            if "uP" in p:
                p["uP"].value = tuple([0.0]*len(p["uP"].value))
            vb = ctx.vertex_array(p, [(eng.quad, "2f", "in_pos")])
            ta = ctx.texture((res, res), 4, dtype="f4"); ts = ctx.texture((res, res), 4, dtype="f4")
            fb = ctx.framebuffer([ta, ts])
            cube = np.zeros((6, res, res, 3), np.float32)
            ck = float(eng.bg_sun[0]) + 0.5*float(eng.bg_sun[1])
            for fi, (W, U, V) in enumerate(_faces()):
                p["uCamPos"].value = tuple(float(x) for x in pos)
                for k, v in (("uCamU", U), ("uCamV", V), ("uCamW", W)):
                    p[k].value = tuple(float(x) for x in v)
                p["uFl"].value = 1.0
                p["uRes"].value = (float(res), float(res))
                fb.use(); ctx.viewport = (0, 0, res, res); fb.clear(0, 0, 0, 0)
                ctx.enable(mgl.BLEND); ctx.blend_func = (mgl.ONE, mgl.ONE)
                for s in range(spp):
                    if "uJitter" in p:
                        p["uJitter"].value = tuple(float(x) for x in rt_engine.r2(s))
                    vb.render(mgl.TRIANGLE_STRIP)
                ctx.disable(mgl.BLEND)
                a = np.frombuffer(ta.read(), dtype="f4").reshape(res, res, 4)/spp
                s_ = np.frombuffer(ts.read(), dtype="f4").reshape(res, res, 4)/spp
                cube[fi] = a[:, :, :3] + s_[:, :, :3]*ck
            amb = self._amb_dirs(eng, pos, res)
            for o in (vb, fb, ta, ts, p):
                o.release()
            bad = int((~np.isfinite(cube)).sum() + (~np.isfinite(amb)).sum())
            cube = np.clip(np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 6.0e4).astype(np.float32)   # render6
            amb = np.clip(np.nan_to_num(amb, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 6.0e4).astype(np.float32)
            if bad:
                print(f"[light] probe bake had {bad} non-finite texels (zeroed; not cached)", flush=True)
            else:
                tmp = path + f".tmp{os.getpid()}.npz"
                np.savez(tmp, cube=cube, amb=amb); os.replace(tmp, path)
        dirs, wsa = _cube_dirs(res)
        rad = cube.reshape(-1, 3)
        lat, ldirs = self._prefilter(rad, dirs, wsa)
        # calibration: the probe's cosine irradiance (what a rough surface would see) vs setAmbient (what the mesh and
        # Muse are lit with) over the same directions; the probe keeps its direction structure, the level is the room's
        irr = lat[-1].reshape(-1, 3)
        la = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
        ambL = la(amb.reshape(-1, 3))
        irrC = la(self._cos_at(rad, dirs, wsa, dirs[::7]))
        self.probe_calib = float(np.mean(ambL[::7])/max(np.mean(irrC), 1e-6))
        lat = np.clip(np.nan_to_num(lat, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 6.0e4)   # render6: fp16-safe, finite
        if not np.isfinite(self.probe_calib) or self.probe_calib <= 0.0:
            self.probe_calib = 1.0
        w, h = lat.shape[2], lat.shape[1]
        self.t_probe = ctx.texture_array((w, h, len(ROUGH_LAYERS)), 4,
                                         np.concatenate([lat, np.ones(lat.shape[:3] + (1,), np.float32)], -1)
                                         .astype(np.float16).tobytes(), dtype="f2")
        self.t_probe.filter = (mgl.LINEAR, mgl.LINEAR)
        self.t_probe.repeat_x = True; self.t_probe.repeat_y = False
        self.probe_info = dict(path=path, pos=pos.tolist(), res=res, calib=round(self.probe_calib, 3),
                               mean_rad=float(np.mean(la(rad))), mean_amb=float(np.mean(ambL)),
                               s=round(time.time() - t0, 2))
        if log:
            print(f"[light] probe {self.probe_info}", flush=True)

    def _amb_dirs(self, eng, pos, res):
        """setAmbient(pos, dir) for every cube texel direction (the character branch of the set's ambient)."""
        ctx, mgl = self.ctx, self.mgl
        dirs, _ = _cube_dirs(res)
        n = len(dirs)
        fs = (eng.sh.head + "uniform float uTime;\nuniform vec3 uSunDir, uSunCol, uRimDir, uRimCol;\n"
              + eng.sh.resolve("#pragma include common\n#pragma include set\n") + """
uniform sampler2D uDirs; uniform vec3 uPos; uniform int uW;
out vec4 fragColor;
void main(){ setInit(); ivec2 ip = ivec2(gl_FragCoord.xy); vec3 d = texelFetch(uDirs, ip, 0).xyz;
  fragColor = vec4(setAmbient(uPos, normalize(d)), 1.0); }
""")
        p = ctx.program(vertex_shader=_VS, fragment_shader=fs)
        W = 256; H = (n + W - 1)//W
        D = np.zeros((H*W, 4), np.float32); D[:n, :3] = dirs
        td = ctx.texture((W, H), 4, D.tobytes(), dtype="f4"); td.filter = (mgl.NEAREST, mgl.NEAREST)
        to = ctx.texture((W, H), 4, dtype="f4"); fb = ctx.framebuffer([to])
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        vb = ctx.vertex_array(p, [(quad, "2f", "in_pos")])
        td.use(0)
        if "uDirs" in p:
            p["uDirs"].value = 0
        p["uPos"].value = tuple(float(x) for x in pos)
        if "uTime" in p:
            p["uTime"].value = 0.0
        fb.use(); ctx.viewport = (0, 0, W, H)
        vb.render(mgl.TRIANGLE_STRIP)
        out = np.frombuffer(to.read(), dtype="f4").reshape(-1, 4)[:n, :3].copy()
        for o in (vb, quad, fb, to, td, p):
            o.release()
        return out

    def _prefilter(self, rad, dirs, wsa):
        W = int(self.c["probe_w"]); H = W//2
        u = (np.arange(W) + 0.5)/W; v = (np.arange(H) + 0.5)/H
        ph = (u - 0.5)*2*np.pi; th = v*np.pi
        PH, TH = np.meshgrid(ph, th)
        R = np.stack([np.sin(TH)*np.cos(PH), np.cos(TH), np.sin(TH)*np.sin(PH)], -1).reshape(-1, 3)
        dot = R @ dirs.T                                   # (W*H, 6 res^2)
        out = []
        for rough in ROUGH_LAYERS:
            a = max(rough*rough, 0.012)
            k = 2.0/(a*a) - 2.0                            # phong exponent of the GGX lobe at this roughness
            k = min(max(k, 1.0), 4000.0)
            wgt = np.where(dot > 0, np.power(np.clip(dot, 0, 1), k), 0.0)*wsa[None, :]
            col = (wgt @ rad)/np.maximum(wgt.sum(1, keepdims=True), 1e-12)
            out.append(col.reshape(H, W, 3).astype(np.float32))
        return np.stack(out), R

    @staticmethod
    def _cos_at(rad, dirs, wsa, at):
        w = np.clip(at @ dirs.T, 0, None)*wsa[None, :]
        return (w @ rad)/np.pi


_VS = """#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


def _faces():
    """(forward W, right U, up V) of the 6 cube faces, the engine camera's convention (ray = x U + y V + fl W)."""
    out = []
    for W in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        W = np.array(W, float)
        up = np.array([0.0, 1.0, 0.0]) if abs(W[1]) < 0.5 else np.array([0.0, 0.0, -W[1]])
        U = np.cross(W, up); U /= np.linalg.norm(U)
        V = np.cross(U, W)
        out.append((W, U, V))
    return out


def _cube_dirs(res):
    """world direction + solid angle of every cube texel, in the order the bake reads them back (face, row from the
    bottom (GL), column)."""
    dirs, w = [], []
    c = (np.arange(res) + 0.5)/res*2 - 1
    X, Y = np.meshgrid(c, c)                # Y row 0 = gl_FragCoord.y 0.5 = the bottom
    for W, U, V in _faces():
        d = X[..., None]*U + Y[..., None]*V + W
        l = np.linalg.norm(d, axis=-1, keepdims=True)
        dirs.append((d/l).reshape(-1, 3))
        w.append((1.0/l[..., 0]**3).reshape(-1))
    w = np.concatenate(w)
    return np.concatenate(dirs).astype(np.float64), w*(4*np.pi/w.sum())
