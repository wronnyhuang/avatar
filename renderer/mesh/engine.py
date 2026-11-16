"""render: MeshEngine — the shared SDF scene engine (engine/rt_engine.RTEngine) with Juno as a rasterized
skinned mesh (renderer/mesh/character.py) depth-composited into it.  Same interface as RTEngine (render / submit /
flush / prewarm / set_tier / surface_write ...), so the live backend only picks the class (juno.json "backend": "mesh").

The SDF side runs unchanged with animal "juno" (animals/juno.glsl: an invisible capsule proxy that casts her live
shadow on the set + held props, and supplies the props' grasp / head frames).  The mesh passes run inside the
composite step (RTEngine._draw calls sprites(.., layer 1) with the HDR target bound, right after the scene
composite): draw -> composite over t_hdr -> the front sprites -> the shared post.

    python renderer/mesh/engine.py --w 960 --h 540 --frames 30 --device cpu
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
import rt_engine  # noqa: E402
from rt_engine import RTEngine, _m3  # noqa: E402
from anim.schema import NPARAMS, IDX, FPS  # noqa: E402
import assetload  # noqa: E402
from character import MeshCharacter  # noqa: E402
from testrig import TestRig, RigFrame  # noqa: E402
import quality  # noqa: E402 render-perf

# render-perf (PERF2): `"uX" in prog` (every _set / RTEngine.set guards with it) had no __contains__ on
# moderngl.Program, so Python fell back to __iter__ = a generator over ALL members: ~44k iterations per Juno frame
# (cProfile).  Same answer as a dict lookup on the members; patched once, process-wide, only where Juno's engine loads.
import moderngl as _mgl  # noqa: E402
if "__contains__" not in vars(_mgl.Program):
    _mgl.Program.__contains__ = lambda self, k: k in self._members

JCAP_N = 18          # animals/juno.glsl

SCENE_LIB_PRE = """
uniform float uTime;
uniform vec3  uSunDir, uSunCol, uRimDir, uRimCol;
"""


class MeshEngine(RTEngine):
    """RTEngine(animal="juno") + Juno's mesh.  `rig`: an object with frame_from_params(P) -> RigFrame (RIG's
    renderer/humans/rig.py), else the stand-in TestRig.  `cam`: optional {"pos","target","fov","focus","dofk"}
    override (stills/judge framings; the background bakes for that camera, cached)."""

    def __init__(self, animal="juno", w=960, h=540, preset="full1", backend="gpu", threads=8, log=False, bg_spp=None,
                 asset_dir=None, lod=None, cam=None, msaa=None, rig=None, mesh_quality=None):
        self._cam_override = cam
        orig = rt_engine.load_cfg
        def load_cfg2(a):
            c = orig(a)
            if cam:
                cc = dict(cam)
                if "motion_layers" in cc:          # stills: no panorama layers (one camera, no pans)
                    c["motion_layers"] = bool(cc.pop("motion_layers"))
                if "quality" in cc:                # render3: stills: the char layer's march range for a close camera
                    c["quality"] = dict(c.get("quality") or {}); c["quality"].update(cc.pop("quality"))
                c["camera"] = dict(c["camera"]); c["camera"].update({k: v for k, v in cc.items() if v is not None})
            return c
        rt_engine.load_cfg = load_cfg2
        try:
            super().__init__(animal, w, h, preset, backend=backend, threads=threads, log=log, bg_spp=bg_spp)
        finally:
            rt_engine.load_cfg = orig
        t0 = time.time()
        self.mcfg = dict(self.cfg.get("mesh", {}))
        if os.environ.get("JUNO_MESH_CFG"):    # render2: look-dev knob — a json dict merged over juno.json "mesh"
            self.mcfg.update(json.loads(os.environ["JUNO_MESH_CFG"]))
        lod = lod or os.environ.get("JUNO_LOD") or self.mcfg.get("lod", "lod0")
        self.asset = assetload.load(asset_dir or assetload.char_asset_dir(animal), lod=lod)   # pipe: junor's own asset
        lib = SCENE_LIB_PRE + self.sh.resolve("#pragma include common\n#pragma include set\n")
        self.mc = MeshCharacter(self.ctx, self.asset, w, h, self.sh.head, lib, self.mcfg, msaa=msaa, log=log)
        self.rig = rig or self._make_rig()
        self.rig_override = None          # a RigFrame for the next frames (stills / tests)
        self.rf = None
        fill = self.mcfg.get("fill", {})
        fd = np.array(fill.get("dir", [0.7, 0.15, 0.7]), float)
        kt = np.array(self.mcfg.get("key_tint", [1.0, 1.0, 1.0]), float)       # render look-dev: her white balance
        # render2: her own key direction (a film cheat: the set keeps the scene sun; mesh.key_dir, world, optional)
        kd = np.array(self.mcfg.get("key_dir", self.sun), float); kd = kd/np.linalg.norm(kd)
        self.lights = dict(key_dir=kd, key_col=self.sun_col*kt, amb_tint=tuple(self.mcfg.get("amb_tint", [1.0, 1.0, 1.0])), rim_dir=self.rim, rim_col=self.rim_col,
                           fill_dir=fd/np.linalg.norm(fd),
                           fill_col=np.array(fill.get("color", [0.55, 0.62, 0.75]), float)*float(fill.get("gain", 0.35)),
                           uKeyGain=float(self.mcfg.get("key_gain", 1.0)), uRimGain=float(self.mcfg.get("rim_gain", 1.0)),
                           uAmbGain=float(self.mcfg.get("amb_gain", 1.0)), uCharK=float(self.char_key))
        self.mesh_ms = {}
        self._post_init()                      # render5: Juno's own final grade (mesh.post), else the shared post
        self._perf_init(mesh_quality)          # render-perf: the mesh quality ladder (quality.py)
        self.init_times["mesh_s"] = round(time.time() - t0, 2)
        if log:
            print(f"[mesh] asset {self.asset.dir} lod {lod}: {self.asset.mesh_meta['vertex_count']} verts, "
                  f"{len(self.asset.shapes)} shapes, msaa {self.mc.msaa}, init {self.init_times['mesh_s']} s", flush=True)

    # ------------------------------------------------------------------ render5: Juno's final grade
    POST_DEFAULTS = dict(bloom=0.5, ca=0.0, contrast=0.0, shoulder=0.0, hue_keep=0.0, grain_mids=0.0, threshold=0.95, veil=0.0)

    def _post_init(self):
        self.pbr_post = None
        pb = self.mcfg.get("pbr")                # shade-post (gated): mesh.pbr.post = the PBR final image
        if isinstance(pb, dict) and isinstance(pb.get("post"), dict) and os.environ.get("JUNO_POST") != "0":
            from post_pbr import PbrPost
            self.post_k = None
            self.pbr_post = PbrPost(self, pb["post"])
            self.p_post = self.pbr_post.p_post
            self.vaos["post"] = self.pbr_post.vao
            self.pbr_post.uniforms()
            return
        pc = self.mcfg.get("post")
        if not pc or os.environ.get("JUNO_POST") == "0":
            self.post_k = None
            return
        self.post_k = dict(self.POST_DEFAULTS); self.post_k.update(pc)
        src = self.sh.head + open(os.path.join(HERE, "post_juno.glsl")).read()
        self.p_post = self.ctx.program(vertex_shader=rt_engine.VS_FULL, fragment_shader=src)
        self.vaos["post"] = self.ctx.vertex_array(self.p_post, [(self.quad, "2f", "in_pos")])
        self._post_uniforms()

    def _post_uniforms(self):
        if getattr(self, "pbr_post", None) is not None:      # shade-post (gated)
            self.pbr_post.uniforms()
            return
        k = getattr(self, "post_k", None)
        if not k:
            return
        p = self.p_post
        self._cam_uniforms(p)
        for g, u in (("exposure", "uExposure"), ("bloom", "uBloom"), ("saturation", "uSaturation"),
                     ("vignette", "uVignette"), ("grain", "uGrain")):
            self.set(p, u, float(self.grade[g]))
        self.set(p, "uGain", tuple(self.grade["gain"])); self.set(p, "uLift", tuple(self.grade["lift"]))
        self.set(p, "uFade", 1.0)
        lk = self.look_k if self.look else {}
        self.set(p, "uClarity", float(lk.get("clarity", 0.0))); self.set(p, "uContrast", float(lk.get("contrast", 0.0)))
        if self.look:
            self.set(p, "uGrain", float(self.grade["grain"])*float(lk.get("grain", 1.0)))
        self.set(p, "uPostK", (float(k["bloom"]), float(k["ca"]), float(k["contrast"]), float(k["shoulder"])))
        self.set(p, "uPostK2", (float(k["hue_keep"]), float(k["grain_mids"]), float(k["threshold"]), float(k["veil"])))
        self.set(p, "uHdr", 0)

    def _common_uniforms(self):
        super()._common_uniforms()
        self._post_uniforms()

    # ------------------------------------------------------------------ the rig
    def _make_rig(self):
        if os.environ.get("JUNO_TESTRIG") != "1":
            try:
                try:
                    from humans.rig import JunoRig
                except ImportError:             # dev: RIG's branch not merged yet -> its worktree (JUNO_RIG_DIR)
                    rd = os.environ.get("JUNO_RIG_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    if rd not in sys.path:
                        sys.path.append(rd)
                    from humans.rig import JunoRig
                return JunoRig(self.asset.dir)
            except Exception as e:          # not merged yet / failed: the stand-in rig (say so once)
                if self.log:
                    print(f"[mesh] RIG's JunoRig unavailable ({type(e).__name__}: {e}); using the stand-in TestRig", flush=True)
        return TestRig(self.asset, H_scene=float(self.mcfg.get("H_scene", 2.10)))

    def rig_frame(self, P):
        if self.rig_override is not None:
            return self.rig_override
        if hasattr(self.rig, "solve"):        # RIG's renderer/humans/rig.py JunoRig: the engine's root is passed in
            bp = self.body_prev
            root = (bp["A"], bp["o"], bp["R"]) if bp is not None else None
            cam = bp["cam"][0] if bp is not None else self.cam_pos
            return self.rig.solve(P, t=self._t_now, cam_pos=cam, root=root)
        # stand-in mapping of a few columns (RIG replaces this with the full control vector)
        g = lambda n: float(P[IDX[n]]) if n in IDX else 0.0
        wts = {"jawOpen": max(g("jaw"), g("mouth_open")*0.8),
               "eyeBlinkLeft": g("blink_r"), "eyeBlinkRight": g("blink_l"),
               "mouthSmileLeft": max(g("smile"), 0.0), "mouthSmileRight": max(g("smile"), 0.0)}
        return self.rig.frame("relaxed", "neutral", weights=wts)

    def _juno_uniforms(self, rf):
        """animals/juno.glsl uniforms (names: docs/CONTRACT--rig.md section 2), all CHAR-LOCAL."""
        loc = lambda n, d: getattr(rf, n, None) if getattr(rf, n, None) is not None else d
        prox = np.asarray(loc("proxies_local", rf.proxies), np.float32)
        cap = np.zeros((JCAP_N*2, 4), np.float32)
        n = min(len(prox), JCAP_N)
        cap[0:2*n:2, :3] = prox[:n, 0:3]; cap[0:2*n:2, 3] = prox[:n, 6]
        cap[1:2*n:2, :3] = prox[:n, 3:6]; cap[1:2*n:2, 3] = prox[:n, 7]
        gl = np.asarray(loc("grasp_local", rf.grasp), np.float32)
        hl = np.asarray(loc("head_local", rf.head), np.float32)
        hc = np.eye(4, dtype=np.float32)
        hrot = getattr(rf, "head_rot_local", None)
        hc[:3, :3] = hrot if hrot is not None else hl[:3, :3]
        hcen = getattr(rf, "head_center_local", None)
        hc[:3, 3] = hcen if hcen is not None else hl[:3, 3]
        cm = lambda M: np.ascontiguousarray(np.asarray(M, np.float32).T).reshape(-1)     # GLSL column-major
        for p in self.scene_progs:
            self._set_arr(p, "uJCap", "vec4", cap)
            self._set_arr(p, "uJGrasp", "mat4", np.stack([cm(gl[0]), cm(gl[1])]))
            if "uJHead" in p:
                p["uJHead"].write(cm(hl).tobytes())
            if "uJHeadC" in p:
                p["uJHeadC"].write(cm(hc).tobytes())

    def _body_frame(self, P):
        bs = super()._body_frame(P)
        self._t_now = getattr(self, "_t_now", 0.0)
        t0 = time.perf_counter()
        rf = self.rig_frame(P)
        self.rf = rf
        self.mesh_ms["rig"] = (time.perf_counter() - t0)*1e3
        self._juno_uniforms(rf)
        # Juno's shadow on the set is always live (the bake never has her): the no-character layers + the proxy
        b = rf.bound
        self.set(self.p_live, "uRootLive", 1.0)
        self.set(self.p_live, "uRootBound", (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        self.set(self.p_bgv, "uRootLive", 1.0)
        bs = dict(bs); bs["live"] = 1.0
        self.body_state = bs
        return bs

    def _draw(self, frame, params, prof=None):
        hp = getattr(self.rig, "human_params", None)       # rig3: the human reading of the plush-built columns
        if hp is not None and self.rig_override is None:     # (face gains, no cartoon marks) — once per frame, for all
            params = hp(params)
        self._t_now = frame/float(FPS)
        self._prof = prof
        if not getattr(self.mc.light, "_probe_tried", False):        # render-light: the room probe, once, before
            self.mc.light._probe_tried = True                        # the first frame (never inside a frame's passes)
            self.mc.light.bake_probe(self, self.rig_frame(params), log=self.log)
        # render-perf: `tier` / `tier_pending` are the MESH ladder's; the SDF level follows its scene_scale
        if self.tier_pending is not None:
            self.mtier = self.tier_pending
            self.mladder.apply(self.mc, self.mtier)
            if getattr(self.mc, "mtaa", None) is not None:   # shade-taa (gated): a tier change resets the history
                self.mc.mtaa.reset()
            lvl = self._sdf_level_for(self.mc.scene_scale)
            self.tier = self.sdf_level
            self.tier_pending = lvl if lvl != self.sdf_level else None
        else:
            self.tier = self.sdf_level
        try:
            return super()._draw(frame, params, prof)
        finally:
            self.sdf_level = self.tier
            self.tier = self.mtier

    # ------------------------------------------------------------------ render-perf (PERF2): GPU I420 output
    def set_output_format(self, fmt):
        """'i420' -> render() returns uint8 (1.5h, w) planar YUV 4:2:0 converted on the GPU (engine/yuv_out.py: the
        H.264 encoder takes it as is: no CPU colour conversion, 3.1 MB read back at 1080p instead of 8.3); 'rgb' -> the
        (h, w, 3) RGB view as before.  AVATAR_GPU_YUV=0 keeps RGB.  -> the format now in effect."""
        want = fmt == "i420" and os.environ.get("AVATAR_GPU_YUV", "1") != "0" and self.w % 2 == 0 and self.h % 2 == 0
        if want and getattr(self, "yuv_out", None) is None:
            from yuv_out import YuvOut
            self.yuv_out = YuvOut(self)
        if getattr(self, "yuv_out", None) is not None:
            self.yuv_out.on = bool(want)
        return "i420" if want else "rgb"

    # ------------------------------------------------------------------ render-perf: quality ladder
    def _perf_init(self, mesh_quality):
        # our ladder is `mladder`; RTEngine's `ladder` ( rq-perf: (level, uRQ) pairs for LOOK_V3 chars) stays
        # None for a mesh character so live/backends/rt.py drives the mesh tiers (tiers/set_tier), not the SDF uRQ ladder
        self.ladder = None
        self.mladder = quality.Ladder(self.mcfg)
        self.sdf_level = self.tier                  # RTEngine's level index (internal scale of the SDF char pass)
        self.sdf_top = self.tier                    # the preset's own level
        q = mesh_quality if mesh_quality is not None else os.environ.get("JUNO_TIER")
        self.mtier = self.mladder.start if q in (None, "") else max(0, min(self.mladder.n - 1, int(q)))
        self.mladder.apply(self.mc, self.mtier)
        lvl = self._sdf_level_for(self.mc.scene_scale)
        if lvl != self.sdf_level:
            self._use_level(lvl); self.sdf_level = lvl
        self.tier = self.mtier
        self.tier_pending = None
        self.stamps = None                          # quality.GpuStamps when profiling (juno_bench)

    def _sdf_level_for(self, frac):
        """scene_scale -> the RTEngine level: the largest level scale <= frac * the preset's scale."""
        if len(self.levels) <= 1:
            return self.sdf_level
        top = self.levels[self.sdf_top]["scale"]
        want = float(frac)*top + 1e-6
        ok = [i for i, L in enumerate(self.levels) if L["scale"] <= want]
        return ok[-1] if ok else 0

    def reset_history(self):
        super().reset_history()
        mc = getattr(self, "mc", None)                       # shade-taa (gated): the mesh layer's history too
        if mc is not None and getattr(mc, "mtaa", None) is not None:
            mc.mtaa.reset()

    @property
    def tiers(self):
        return self.mladder.tiers()

    @property
    def tier_names(self):
        return self.mladder.names()

    def set_tier(self, i):
        """Mesh quality tier for the following frames (0 = cheapest, quality.py)."""
        i = max(0, min(self.mladder.n - 1, int(i)))
        self.tier_pending = i if i != self.mtier else None

    def prewarm(self):
        """RTEngine's prewarm (every SDF path) + one frame on every mesh tier (MSAA targets, shadow sizes, scales:
        their first-use driver work never lands in a live frame), back to the tier it started on."""
        t0 = time.time()
        start = self.mtier
        super().prewarm()
        Z = np.zeros(NPARAMS, "f4")
        for t in list(range(self.mladder.n)) + [start]:
            self.set_tier(t)
            self.render(0, Z)
        self.reset_history()
        self.body_prev = None
        self.init_times["prewarm_s"] = round(time.time() - t0, 2)

    # ------------------------------------------------------------------ the mesh passes, inside the composite step
    def sprites(self, t, frame, layer):
        if layer == 1 and self.rf is not None and not getattr(self, "_no_mesh", False):
            self._mesh_pass(t, frame)
        super().sprites(t, frame, layer)

    def _mesh_pass(self, t, frame):
        ctx = self.ctx
        k = self.nsub & 1
        cam = self.body_prev["cam"] if self.body_prev else self.cam0
        # render2: the SDF char layer + this frame's tile mask (the engine's composite reads it only inside the mask)
        if self.taa:
            scene = (self.t_live, self.t_hc[k], self.t_hd[k], self.t_mask[k], 1.0, (self.iw, self.ih))
        else:
            scene = (self.t_live, self.t_c, self.t_d, self.t_mask[k], 1.0/max(1, int(self.spp)), (self.iw, self.ih))
        mc = self.mc
        st = self.stamps                 # render-perf: GPU timestamps per mesh pass (juno_bench)
        mc.stamp = st.mark if st is not None else None
        if st is not None:
            st.mark("start")
        mc.pose(self.rf)
        if st is not None:
            st.mark("skin")
        mc.shadow(self.rf, self.lights["key_dir"])
        if st is not None:
            st.mark("shadow")
        mc.draw(self.rf, cam, self.lights, scene, time_s=t)
        if st is not None:
            st.mark("sss")
        focus = self.p_comp["uFocus"].value if "uFocus" in self.p_comp else self.focus
        dofk = self.p_comp["uDofK"].value if "uDofK" in self.p_comp else self.dofk*self.h/1080.0
        px = self.h/1080.0
        kz = float(cam[4])/float(self.cam0[4]) if cam is not None and float(self.cam0[4]) > 0 else 1.0   # render7: zoom
        dofk = dofk*float(self.mcfg.get("dof_k", 1.0))   # render7: the mesh layer's own lens blur (her face stays crisp under the card's shallow DoF)
        mc.composite(self.f_hdr, cam, focus, dofk*px, frame, dof_max=7.0*px*float(self.mcfg.get("dof_max_k", 1.0)), zoom=kz)
        if st is not None:
            st.mark("composite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--preset", default="full1")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    eng = MeshEngine("juno", a.w, a.h, a.preset, backend=a.device, threads=a.threads, log=True)
    Z = np.zeros(NPARAMS, "f4")
    ts = []
    for i in range(a.frames):
        t0 = time.perf_counter(); img = eng.render(i, Z); ts.append(time.perf_counter() - t0)
    if a.out:
        from PIL import Image
        Image.fromarray(img).save(a.out)
    ts = np.array(ts[2:] or ts)*1000
    print(f"juno {a.w}x{a.h}: mean {ts.mean():.1f} ms  init {eng.init_times}")


if __name__ == "__main__":
    main()
