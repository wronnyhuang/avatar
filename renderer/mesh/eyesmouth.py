"""render-eyes: per-frame uniforms for the eye and mouth materials (mat_eye / mat_cornea / mat_generic).

    fm = FaceMetrics(asset)          # once: eye centres + radius, the lid APERTURE per eye (measured from the head
                                     # mesh, rest + a linear delta per lid-moving shape), the lip seam points
    fm.apply(prog, rig)              # per frame, per program: sets the uniforms the program declares

Everything is measured from the asset generically (skeleton_hints eye_center_*/eye_radius_m or the eye submesh bound,
landmark "mouth" or the teeth bound) so the shaders get:
  uHeadR   mat3  head frame (columns x right, y up, z out of the face; unit length) in world
  uHeadS   float scene units per asset metre
  uEyeC[2] vec3  eyeball centres (world), uEyeR float eyeball radius (world)
  uLidTop[2*NAZ], uLidBot[2*NAZ]   the visible aperture of each eyeball: elevation (deg, head frame) of the upper /
           lower lid margin at NAZ azimuths in [-AZMAX, AZMAX] (az = atan2(x, z) about the eye centre); top < bot = shut
  uMouthC  vec3  lip-seam centre (world), uMouthU / uMouthL vec3 upper / lower inner lip points (world), uMouthW (world)
The aperture is what makes the shader-side lid AO / upper-lid shadow band / meniscus / caruncle fallbacks work when
the asset has no eye_occl_* / tearline_* geometry (and keeps the iris shading honest when it has)."""
import time

import numpy as np

NAZ = 13
AZMAX = 66.0          # degrees either side of the eye's forward axis
EL_STEP = 0.5


def _skin_mats(A, rig):
    return np.einsum("bij,bjk->bik", np.asarray(rig.bones, np.float64), A.inv_bind.astype(np.float64))


class FaceMetrics:
    def __init__(self, A, log=False):
        self.A = A
        bi = A.bone_index
        self.b_head = bi.get("head", 0)
        self.b_eye = [bi.get("eye_l", self.b_head), bi.get("eye_r", self.b_head)]
        P = A.mesh["position"].astype(np.float64)
        I = A.mesh["index"].reshape(-1, 3)
        hints = (A.skeleton or {}).get("skeleton_hints", {}) or {}
        mats = A.meta.get("materials", {})
        sub_t = []
        for s in A.submeshes:
            t = (mats.get(s["material"], {}) or {}).get("type", s["material"])
            sub_t.append((s, t))
        def sub_verts(pred):
            out = []
            for s, t in sub_t:
                if pred(s, t):
                    tri = I[s["first_index"]//3:(s["first_index"] + s["index_count"])//3]
                    out.append(tri)
            return np.concatenate(out) if out else np.zeros((0, 3), np.int64)
        # ---- eyes: centre + radius (hints, else the eye submesh bound)
        self.eye_c = []
        R = hints.get("eye_radius_m")
        for k, side in enumerate(("l", "r")):
            c = hints.get(f"eye_center_{side}")
            if c is None:
                tri = sub_verts(lambda s, t, side=side: t == "eye" and s["name"].endswith("_" + side))
                v = P[np.unique(tri)] if len(tri) else np.zeros((1, 3))
                c = 0.5*(v.min(0) + v.max(0))
                if R is None and len(tri):
                    R = float(0.5*(v.max(0) - v.min(0))[1])
            self.eye_c.append(np.asarray(c, np.float64))
        self.eye_r = float(R or 0.0125)
        # ---- the lid aperture, measured on the skin around each eye (projected along the head's +z)
        skin_tri = sub_verts(lambda s, t: t == "skin")
        self.az = np.linspace(-AZMAX, AZMAX, NAZ)
        self.top0 = np.zeros((2, NAZ)); self.bot0 = np.zeros((2, NAZ))
        self.dtop = []; self.dbot = []          # per eye: [(shape index, dtop[NAZ], dbot[NAZ])]
        self.shape_of = {}
        for k in range(2):
            c = self.eye_c[k]
            cen = P[skin_tri].mean(1)
            near = (np.linalg.norm(cen - c, axis=1) < 3.0*self.eye_r) & \
                   (np.linalg.norm((cen - c)[:, :2], axis=1) < 1.6*self.eye_r) & (cen[:, 2] > c[2])
            tri = skin_tri[near]
            vids = np.unique(tri)
            t0, b0 = self._aperture(P, tri, c)
            self.top0[k], self.bot0[k] = t0, b0
            lst = []
            for si, (nm, idx, dp, dn) in enumerate(A.shapes):
                m = np.isin(idx, vids)
                if not m.any() or np.abs(dp[m]).max() < 2e-4:
                    continue
                Q = P.copy()
                wt = 0.3
                Q[idx[m]] += wt*dp[m]
                t1, b1 = self._aperture(Q, tri, c)
                ok = (t0 > b0) & (t1 > b1)
                dt = np.where(ok, (t1 - t0)/wt, 0.0); db = np.where(ok, (b1 - b0)/wt, 0.0)
                if np.abs(dt).max() + np.abs(db).max() > 0.5:
                    lst.append((si, dt, db))
            self.dtop.append([(s, d) for s, d, _ in lst]); self.dbot.append([(s, d) for s, _, d in lst])
            if log:
                print(f"[eyesmouth] eye {k}: top {np.round(t0, 1)} bot {np.round(b0, 1)}; {len(lst)} lid shapes "
                      f"({', '.join(A.shape_names[s] for s, _, _ in lst)})")
        # ---- mouth: the two midline lip points around the seam (rest), skinned per frame on the CPU
        lm = A.meta.get("landmarks", {}) or {}
        teeth = sub_verts(lambda s, t: t == "teeth")
        tv = P[np.unique(teeth)] if len(teeth) else None
        mc = lm.get("mouth")
        if mc is None:
            mc = (0.5*(tv.min(0) + tv.max(0)) + [0, 0, 0.01]) if tv is not None else self.eye_c[0]*[0, 1, 1] - [0, 0.06, 0]
        mc = np.asarray(mc, np.float64)
        self.mouth_w = float((tv.max(0) - tv.min(0))[0]*0.85) if tv is not None else 0.05
        sv = np.unique(skin_tri)
        q = P[sv] - mc
        mid = (np.abs(q[:, 0]) < 0.0025) & (np.abs(q[:, 1]) < 0.015) & (q[:, 2] > -0.012) & (q[:, 2] < 0.01)
        cand = sv[mid]
        def pick(sign):
            cc = cand[(P[cand, 1] - mc[1])*sign > 0]
            if not len(cc):
                return None
            return int(cc[np.argmin(np.linalg.norm(P[cc] - mc, axis=1))])
        self.lip = [pick(+1), pick(-1)]
        self.mouth_c = mc
        self.lip_sh = []
        for v in self.lip:
            lst = []
            if v is not None:
                for si, (nm, idx, dp, dn) in enumerate(A.shapes):
                    j = np.nonzero(idx == v)[0]
                    if len(j):
                        lst.append((si, dp[j[0]].astype(np.float64)))
            self.lip_sh.append(lst)
        if log:
            print(f"[eyesmouth] eye R {self.eye_r:.4f}, mouth {np.round(mc, 4)} w {self.mouth_w:.4f}, lips {self.lip}")
        self._last = None

    def _aperture(self, P, tri, c):
        """Visible elevation range [bot, top] (deg) per azimuth of the eyeball sphere seen along +z."""
        R = self.eye_r
        el = np.arange(-70.0, 70.0 + 1e-6, EL_STEP)
        az = np.radians(self.az)[:, None]; e = np.radians(el)[None, :]
        q = np.stack([np.sin(az)*np.cos(e), np.sin(e)*np.ones_like(az), np.cos(az)*np.cos(e)], -1)*R*1.002 + c
        pts = q.reshape(-1, 3)
        a, b, d = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        cov = np.zeros(len(pts), bool)
        for s in range(0, len(pts), 512):
            p = pts[s:s + 512][:, None, :]
            v0 = (b - a)[None, :, :2]; v1 = (d - a)[None, :, :2]; v2 = p[..., :2] - a[None, :, :2]
            den = v0[..., 0]*v1[..., 1] - v1[..., 0]*v0[..., 1]
            den = np.where(np.abs(den) < 1e-14, 1e-14, den)
            u = (v2[..., 0]*v1[..., 1] - v1[..., 0]*v2[..., 1])/den
            w = (v0[..., 0]*v2[..., 1] - v2[..., 0]*v0[..., 1])/den
            inside = (u >= 0) & (w >= 0) & (u + w <= 1)
            z = a[None, :, 2] + u*(b - a)[None, :, 2] + w*(d - a)[None, :, 2]
            cov[s:s + 512] = (inside & (z > p[..., 2])).any(1)
        vis = ~cov.reshape(len(self.az), len(el))
        top = np.full(len(self.az), -5.0); bot = np.full(len(self.az), 5.0)     # shut by default
        for i in range(len(self.az)):
            idx = np.nonzero(vis[i])[0]
            if not len(idx):
                continue
            # the contiguous visible run closest to the horizon
            runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
            r = min(runs, key=lambda r: np.min(np.abs(el[r])))
            top[i], bot[i] = el[r[-1]] + 0.5*EL_STEP, el[r[0]] - 0.5*EL_STEP
        return top, bot

    def frame(self, rig):
        A = self.A
        Sk = _skin_mats(A, rig)
        Rh = Sk[self.b_head][:3, :3]
        s = float(np.linalg.norm(Rh[:, 0]))
        Rn = Rh/s
        w = np.asarray(rig.weights, np.float64)
        u = {"uHeadR": tuple(Rn.T.reshape(-1)), "uHeadS": s, "uEyeR": self.eye_r*s}
        ec = []
        for k in range(2):
            M = Sk[self.b_eye[k]]
            ec.extend(M[:3, :3] @ self.eye_c[k] + M[:3, 3])
        u["uEyeC"] = np.array(ec, np.float32)
        top = self.top0.copy(); bot = self.bot0.copy()
        for k in range(2):
            for si, d in self.dtop[k]:
                top[k] += w[si]*d
            for si, d in self.dbot[k]:
                bot[k] += w[si]*d
        u["uLidTop"] = top.astype(np.float32).reshape(-1); u["uLidBot"] = bot.astype(np.float32).reshape(-1)
        # lips: shapes + linear blend skinning on the CPU (two vertices)
        J = A.mesh["joints"]; W = A.mesh["weights"]
        lp = []
        for v, sh in zip(self.lip, self.lip_sh):
            if v is None:
                lp.append(None); continue
            p = A.mesh["position"][v].astype(np.float64).copy()
            for si, dp in sh:
                p += w[si]*dp
            M = sum(float(W[v][j])*Sk[int(J[v][j])] for j in range(4))
            lp.append(M[:3, :3] @ p + M[:3, 3])
        Mh = Sk[self.b_head]
        mc = Mh[:3, :3] @ self.mouth_c + Mh[:3, 3]
        up = lp[0] if lp[0] is not None else mc
        lo = lp[1] if lp[1] is not None else mc
        u["uMouthC"] = tuple(0.5*(up + lo)); u["uMouthU"] = tuple(up); u["uMouthL"] = tuple(lo)
        u["uMouthW"] = self.mouth_w*s
        return u

    detail = 1.0      # render4: quality.py "detail" knob (0 on the cheapest tiers: no pore / knit-warp ALU)

    def _k4_tier(self):
        if self.detail >= 1.0:
            return self.k4
        k = dict(self.k4)
        sk = list(k["uSkinK2"]); sk[0] *= self.detail; k["uSkinK2"] = tuple(sk)
        ck = list(k["uClothK"]); ck[1] *= self.detail; k["uClothK"] = tuple(ck)
        return k

    def apply(self, prog, rig):
        """Set every face uniform `prog` declares (one CPU solve per frame, cached on the rig object)."""
        if self._last is None or self._last[0] is not rig:
            self._last = (rig, self.frame(rig))
        for k, v in self._last[1].items():
            if k not in prog:
                continue
            if isinstance(v, np.ndarray):
                prog[k].write(np.ascontiguousarray(v, np.float32).tobytes())
            else:
                prog[k].value = v


# ---------------------------------------------------------------------------------------------------------- render-face
_KEEP = ("b_head", "b_eye", "eye_c", "eye_r", "az", "top0", "bot0", "dtop", "dbot", "lip", "lip_sh", "mouth_c", "mouth_w")


def _finite_fm(fm):
    try:
        ok = all(np.isfinite(np.asarray(getattr(fm, k), float)).all() for k in ("eye_c", "eye_r", "top0", "bot0", "mouth_c", "mouth_w"))
        ok = ok and all(np.isfinite(np.asarray(d, float)).all() for lst in (fm.dtop + fm.dbot) for _, d in lst)
        ok = ok and all(np.isfinite(np.asarray(d, float)).all() for lst in fm.lip_sh for _, d in lst)
        return bool(ok)
    except Exception:
        return False


def face_cache_name(A):
    """render6: the cache file name of this asset's face solve (geometry + shapes + hints + solver params)."""
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(A.mesh["position"]).tobytes()); h.update(np.ascontiguousarray(A.mesh["index"]).tobytes())
    for nm, idx, dp, dn in A.shapes:
        h.update(nm.encode()); h.update(np.ascontiguousarray(idx).tobytes()); h.update(np.ascontiguousarray(dp).tobytes())
    h.update(repr(((A.skeleton or {}).get("skeleton_hints"), (A.meta or {}).get("landmarks"), NAZ, AZMAX, EL_STEP, 2)).encode())
    return h.hexdigest()[:20] + ".pkl"


SHIP_LOOKUP = True        # read renderer/mesh/face_cache/ first (renderer/tools/juno_face_cache.py turns it off to re-solve)


def face_metrics(A, log=False):
    """FaceMetrics with an on-disk cache (the lid-aperture solve is ~seconds; keyed by the asset geometry + shapes)."""
    import os, pickle
    d = os.environ.get("JUNO_FACE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "juno_face"))
    name = face_cache_name(A)
    path = os.path.join(d, name)
    fm = FaceMetrics.__new__(FaceMetrics)
    fm.A = A; fm._last = None
    # render: the shipped cache (renderer/mesh/face_cache/, committed for the shipped asset) is read first, so a
    # fresh pod / box never pays the cold solve (~40 s for v1) on its first start
    ship = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_cache", name)
    for pth in ((ship, path) if SHIP_LOOKUP else (path,)):
        try:
            with open(pth, "rb") as f:
                st = pickle.load(f)
            for k in _KEEP:
                setattr(fm, k, st[k])
            if _finite_fm(fm):                    # render6: a corrupt / non-finite cache is solved again
                return fm
        except Exception:
            pass
    t0 = time.time()
    fm = FaceMetrics(A, log=log)
    print(f"[eyesmouth] face solve (no shipped cache {name}): {time.time() - t0:.1f} s -- run renderer/tools/juno_face_cache.py",
          flush=True)
    try:
        os.makedirs(d, exist_ok=True)
        tmp = path + f".{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump({k: getattr(fm, k) for k in _KEEP}, f)
        os.replace(tmp, path)
    except Exception:
        pass
    return fm


def map_is_flat(tex_path, ch):
    """True when a texture channel carries no information (a placeholder constant): the shaders then use their
    procedural fallbacks (face-region AO / thickness / tint from the landmarks)."""
    try:
        from PIL import Image
        a = np.asarray(Image.open(tex_path).convert("RGBA"))[..., ch].astype(np.float32)
        return float(a.std()) < 1.5
    except Exception:
        return True


def fix_lash_cards(mc, n_strands=36, res=512):
    """Lash CARDS whose albedo has no alpha (an opaque texture) render as solid bands.  Replace that texture (for the
    lash draws only) with a procedural strand card per draw, covering just the card's UV box (uLashBox remaps uv in
    mat_lashes, so the strands get full texture resolution): per texel the root->tip fraction t (distance from the
    eyeball) and the along-lid coordinate s (azimuth about the eye) are rasterised from the card's own triangles, then
    alpha = clumped tapered strands along s over a dense root line; lower-lid lashes (below the eye centre) are shorter
    and lighter.  Generic: reads only the lash submeshes + eye centres; a real lash texture (with alpha) is untouched."""
    import os
    A = mc.A
    draws = [d for d in mc.draws if d["type"] == "lashes"]
    if not draws:
        return False
    mats = A.meta.get("materials", {})
    md = mats.get(draws[0]["sub"]["material"], {}) or {}
    p = os.path.join(A.dir, md.get("albedo", "")) if md.get("albedo") else None
    if p and os.path.exists(p) and not map_is_flat(p, 3):
        return False
    P = A.mesh["position"].astype(np.float64); UV = A.mesh["uv0"].astype(np.float64)
    I = A.mesh["index"].reshape(-1, 3)
    hints = (A.skeleton or {}).get("skeleton_hints", {}) or {}
    lm = A.meta.get("landmarks", {}) or {}
    col = np.asarray(Image_mean(p) if p and os.path.exists(p) else (30, 20, 16), np.float64)*0.8
    rng = np.random.default_rng(7)
    for d in draws:
        sub = d["sub"]
        tri = I[sub["first_index"]//3:(sub["first_index"] + sub["index_count"])//3]
        v = np.unique(tri)
        side = "l" if sub["name"].endswith("_l") else "r"
        c = np.asarray(hints.get(f"eye_center_{side}") or lm.get(f"eye_{side}") or P[v].mean(0), np.float64)
        q = P - c
        dist = np.linalg.norm(q, axis=1)
        tv = np.zeros(len(P)); sv = np.zeros(len(P)); lo = np.zeros(len(P))
        # each connected card (upper / lower row): its UV long axis runs along the lid (s), the short axis root->tip
        # (t, signed so that t grows away from the eye: distance + height off the eye's horizontal plane)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        rr = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]]); cc = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
        _, lab = connected_components(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(P), len(P))), directed=False)
        for L in np.unique(lab[v]):
            g = v[lab[v] == L]
            if len(g) < 4:
                continue
            uvg = UV[g] - UV[g].mean(0)
            w_, V_ = np.linalg.eigh(np.cov(uvg.T))
            tt = uvg @ V_[:, 0]; ss = uvg @ V_[:, 1]
            away = dist[g] + np.abs(q[g, 1])
            if np.corrcoef(tt, away)[0, 1] < 0:
                tt = -tt
            tv[g] = (tt - tt.min())/max(np.ptp(tt), 1e-9)
            sv[g] = (ss - ss.min())/max(np.ptp(ss), 1e-9)
            lo[g] = float(q[g, 1].mean() < 0)
        u0 = UV[v].min(0); u1 = UV[v].max(0); span = np.maximum(u1 - u0, 1e-6)
        T = np.zeros((res, res, 4), np.float32)
        for a, b, e in tri:
            uv = (UV[[a, b, e]] - u0)/span*res
            x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, res - 1), min(y1, res - 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            v0 = uv[1] - uv[0]; v1 = uv[2] - uv[0]
            den = v0[0]*v1[1] - v1[0]*v0[1]
            if abs(den) < 1e-12:
                continue
            px = gx - uv[0, 0]; py = gy - uv[0, 1]
            l1 = (px*v1[1] - v1[0]*py)/den; l2 = (v0[0]*py - px*v0[1])/den; l0 = 1.0 - l1 - l2
            m = (l0 >= -0.03) & (l1 >= -0.03) & (l2 >= -0.03)
            if not m.any():
                continue
            sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            for ch, arr in ((0, tv), (1, sv), (3, lo)):
                T[sl][..., ch] = np.where(m, l0*arr[a] + l1*arr[b] + l2*arr[e], T[sl][..., ch])
            T[sl][..., 2] = np.where(m, 1.0, T[sl][..., 2])
        t, s, cov, low = T[..., 0], T[..., 1], T[..., 2], T[..., 3] > 0.5
        n = n_strands
        jit = rng.uniform(-0.3, 0.3, n); wid = rng.uniform(0.40, 0.60, n); lenf = rng.uniform(0.75, 1.0, n)
        x = s*n
        i = np.clip(np.floor(x).astype(int), 0, n - 1)
        al = np.zeros_like(s)
        for o in (-1, 0, 1):
            j = np.clip(i + o, 0, n - 1)
            ctr = j + 0.5 + jit[j] + (t**2)*0.8*(s - 0.5)*2.0
            w = wid[j]*(1.0 - 0.8*t)
            prof = np.clip(1.0 - np.abs(x - ctr)/np.maximum(w, 1e-3), 0.0, 1.0)
            al = np.maximum(al, prof*(t < lenf[j]))
        al = np.maximum(al, np.clip((0.42 - t)/0.15, 0.0, 1.0))          # the dense root line along the lid
        al = np.where(low, al*np.clip((0.6 - t)/0.3, 0.0, 1.0)*0.55, al)    # lower lashes: short, light
        al *= cov
        img = np.zeros((res, res, 4), np.uint8)
        img[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
        img[..., 3] = np.clip(al*255.0, 0, 255).astype(np.uint8)
        tx = mc.ctx.texture((res, res), 4, np.ascontiguousarray(img).tobytes())
        tx.build_mipmaps()
        tx.filter = (mc.mgl.LINEAR_MIPMAP_LINEAR, mc.mgl.LINEAR)
        tx.repeat_x = tx.repeat_y = False
        d["tex"][0] = tx
        d["u"]["uTexOn"] = (1.0,) + tuple(d["u"]["uTexOn"][1:])
        d["u"]["uLashBox"] = (float(u0[0]), float(u0[1]), float(span[0]), float(span[1]))
    return True


def Image_mean(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), np.float64).reshape(-1, 3).mean(0)


class Face:
    """render-face: everything the face materials need beyond mesh.glsl's SIn, set up once from the asset.
       face_setup(mc) -> mc.face;  mc.face.apply(prog, rig) per program per frame."""
    def __init__(self, mc, cfg=None):
        import os
        self.mc = mc
        A = mc.A
        cfg = dict(cfg or {})
        self.fm = face_metrics(A)
        self.lash_fixed = fix_lash_cards(mc) if cfg.get("lash_fix", True) else False
        # procedural face maps (crease AO, thin parts, blush/violet/lip tint) when the skin's packed map is a constant
        mats = A.meta.get("materials", {})
        flat = True
        for mid, md in mats.items():
            md = md or {}
            if md.get("type") == "skin" and "head" in mid and md.get("packed"):
                flat = map_is_flat(os.path.join(A.dir, md["packed"]), 0)
        self.proc = float(cfg.get("proc", 1.0 if flat else 0.0))   # render: the asset's real maps win (lead 00:40)
        self.tint = float(cfg.get("tint", 1.0))
        self.skin_spec = float(cfg.get("skin_spec", 0.5))      # render look-dev
        self.eye_q = float(cfg.get("eye_quality", 1.0))
        self.face_ao = float(cfg.get("ao", 0.0))           # render2: landmark crease AO even over real maps (0..1)
        self.liner = float(cfg.get("liner", 0.0))            # render2: upper lash-line darkening on the lid margin
        self.iris_grade = float(cfg.get("iris_grade", 0.0))  # render2: iris toward a deep layered brown (0..1)
        # render3: skin look knobs (terminator glow, coloured-penumbra spread, grazing sheen, shadow chroma clamp)
        self.skin_k = tuple(float(x) for x in cfg.get("skin_k", (1.0, 1.0, 0.0, 0.0)))
        # render4: more vec4 look knobs (mesh.face.<key> -> uniform; defaults = the render3 look, identity)
        self.k4 = {u: tuple(float(x) for x in cfg.get(key, dflt)) for key, u, dflt in R4_KNOBS}
        # the cornea shares the eyeball's iris layout (uv): copy the eye's iris params onto the cornea draws
        eyeP = next((d["u"]["uEyeP"] for d in mc.draws if d["type"] == "eye"), None)
        bulge = float(cfg.get("cornea_bulge", 0.9))
        for d in mc.draws:
            if d["type"] == "cornea":
                if eyeP is not None:
                    d["u"]["uEyeP"] = eyeP
                d["u"]["uCorneaBulge"] = bulge
            if d["sub"].get("name", "").startswith("lashline"):      # render4: the lash-line strip's opacity (eye_k.x)
                d["u"]["uAlphaMul"] = self.k4["uEyeK"][0]
            if d["type"] == "lashes" and self.lash_fixed:
                d["order"] = 2          # blended (premultiplied, no depth write) like the other eye layers

    detail = 1.0      # render4: quality.py "detail" knob (0 on the cheapest tiers: no pore / knit-warp ALU)

    def _k4_tier(self):
        if self.detail >= 1.0:
            return self.k4
        k = dict(self.k4)
        sk = list(k["uSkinK2"]); sk[0] *= self.detail; k["uSkinK2"] = tuple(sk)
        ck = list(k["uClothK"]); ck[1] *= self.detail; k["uClothK"] = tuple(ck)
        return k

    def apply(self, prog, rig):
        self.fm.apply(prog, rig)
        for k, v in (("uFaceProc", self.proc), ("uFaceTint", self.tint), ("uEyeQ", self.eye_q), ("uSkinSpec", self.skin_spec),
                     ("uFaceAO", self.face_ao), ("uIrisGrade", self.iris_grade), ("uFaceLiner", self.liner),
                     ("uSkinK", self.skin_k)) + tuple(self._k4_tier().items()):
            if k in prog:
                prog[k].value = v


# render4 look knobs: (juno.json mesh.face key, uniform, default = no change vs render3)
R4_KNOBS = (
    ("mouth_k", "uMouthK", (0.0, 1.0, 0.0, 0.0)),      # x key through the lip opening, y cavity floor gain
    ("rim_k", "uRimK", (0.0, 0.75, 0.45, 0.0)),        # render7: skin rim only on the head's outer silhouette: x strength, y/z |radial.view| kill/keep
    ("mouth_k2", "uMouthK2", (1.0, 0.0, 0.0, 0.0)),    # render7: teeth: x exposure, y shade toward the corners / back, z upper-lip shade on the upper row, w -
    ("skin_k2", "uSkinK2", (0.0, 300.0, 0.0, 0.0)),    # x micro amplitude, y pore freq / uv, z region wet spec, w broad sheen
    ("eye_k", "uEyeK", (1.0, 0.0, 0.0, 0.0)),
    ("sss_k", "uSssK", (0.0, 0.0, 0.0, 0.0)),          # x painted-hair luminance threshold (out of the SSS blur)
    ("skin_k3", "uSkinK3", (0.0, 0.0, 0.0, 0.0)),      # render5: x de-blotch, y warm bounce below-front, z tan depth, w broad sheen
    ("skin_k4", "uSkinK4", (0.0, 0.0, 0.0, 0.0)),      # render5: x under-jaw neck shade, y form falloff to the silhouette
    ("eye_k2", "uEyeK2", (0.0, 0.0, 0.0, 0.0)),        # render5: x window catchlight, y 2nd small catchlight, z sclera lift by the iris, w lash line warmth
    ("detail_k", "uDetailK", (1.0, 1.0, 0.0, 1.0)),    # render8: x normal-map strength, y baked AO, z albedo mottling flatten, w crease AO scale
    ("warm_k", "uWarmK", (0.0, 0.0, 0.0, 0.0)),        # render8: x warm cheeks/nose/ears, y thin-part boost, z cooler brow/jaw, w lip tint
    ("eye_k3", "uEyeK3", (0.0, 0.0, 0.0, 0.0)),        # render8: key catchlight x roundness, y size (0 = off), z/w offset in the window plane
    ("cloth_k", "uClothK", (1.0, 0.0, 0.0, 0.0)),      # x knit relief contrast, y knit macro warp, z bright-cloth exposure, w its form          # x lashline alpha mul, y upper-lid shade on the ball, z sclera tone, w wet corner
)


def face_setup(mc):
    mc.face = Face(mc, (mc.cfg or {}).get("face", {}))
    return mc.face
