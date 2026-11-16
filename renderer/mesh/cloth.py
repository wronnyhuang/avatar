"""render-cloth: procedural yarn / weave relief for mat_cloth.glsl.

Per cloth material the shader samples ONE small tileable, mip-mapped relief texture (RGBA16F: dh/dx, dh/dy in units
of the stitch pitch, height 0..1, per-stitch colour variation 0..1) in the pattern space q = rot(uv) * m_per_uv /
stitch_m (q in stitch pitches).  Mip-mapping + anisotropic filtering band-limits the stitch / ply detail, so it does
not sparkle or moire under motion; the shader adds cavity AO, a few parallax steps and a self-shadow tap from it.

Material params (juno.asset.json materials.<id>.params, all optional; see docs in mat_cloth.glsl):
    weave          "knit" | "rib" | "plain" | "twill" | "none"    (default: guessed from the material id)
    stitch_mm      stitch / thread pitch across, mm              (chunky knit 8.5, rib 6, plain 1.3, twill 1.7: stylised, readable at bust framing)
    weave_depth    relief scale 0..2                              (1)
    uv_m_per_unit  metres per UV unit                            (estimated from the mesh when absent)
    weave_rot_deg  rotation of the pattern in UV (0 = stitch columns along +v)
    sheen_color, sheen_roughness, fuzz (0..1 grazing fibre halo), color_var (0..1 per-stitch colour variation),
    wash (0..1 large-scale heather / wash mottling, ~25 stitches across; default 0.5), wrap (diffuse wrap)

Quality (MeshCharacter cfg "cloth_quality" or env JUNO_CLOTH_Q, 0..3): 0 = maps + sheen only, 1 = relief normal +
cavity, 2 = + 2 parallax steps + 1 self-shadow tap, 3 = + 4 parallax steps + 2 self-shadow taps.
"""
import os
import numpy as np

KINDS = {"none": 0, "knit": 1, "rib": 2, "plain": 3, "twill": 4}
# kind -> (stitch_mm, relief depth / pitch, wrap, sheen roughness, fuzz, colour var, cavity)
KIND_DFL = {
    "knit":  (8.5, 0.42, 0.55, 0.55, 0.70, 0.55, 0.85),
    "rib":   (6.0, 0.42, 0.55, 0.55, 0.70, 0.45, 0.85),
    "plain": (1.3, 0.32, 0.30, 0.45, 0.25, 0.25, 0.45),
    "twill": (1.7, 0.42, 0.30, 0.50, 0.30, 0.25, 0.55),
    "none":  (6.0, 0.00, 0.35, 0.50, 0.40, 0.00, 0.00),
}
QUALITY = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 2, 1), 3: (3, 4, 2)}   # level -> (level, parallax steps, shadow taps)
_TEX = {}


# ---------------------------------------------------------------------------------------------- relief generators
def _rng(seed):
    return np.random.default_rng(seed)


def _knit(N=12, M=16, P=32, rib=False, seed=7):
    """Stockinette (or k2p2 rib) relief.  Units: x in stitch pitches, row pitch rh.  Returns h[H,W], var[H,W], tile."""
    rh = 0.75 if not rib else 0.70
    W, H = N*P, int(round(M*rh*P))
    ys, xs = (np.mgrid[0:H, 0:W] + 0.5)/P
    r = _rng(seed)
    jw = r.normal(0, 1, (M, N, 2, 4))                    # per stitch, per leg: width, tilt, height, offset jitter
    cvar = r.uniform(0, 1, (M, N))
    rowv = r.normal(0, 1, M)
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
                # knit leg: a tilted, tapering yarn tube (top leans outward)
                th = np.radians((28.0 if not rib else 17.0) + 5.0*Jt)*sgn
                axx, axy = np.sin(th), np.cos(th)
                kcx = cx + sgn*(0.235 if not rib else 0.20) + 0.035*Jo; kcy = y0 + (0.52 + 0.03*Jh)*rh
                ka = 0.66*rh/0.75; kb = (0.255 if not rib else 0.215)*(1.0 + 0.12*Jw); khs = 1.0 + 0.10*Jh
                # purl bump: a horizontal half loop, alternate rows shifted, recessed
                pcx = cx + 0.18*np.where(j % 2 == 1, 1.0, -1.0)*sgn; pcy = y0 + 0.5*rh
                pb = 0.26*rh/0.75*(1.0 + 0.07*Jw)
                axx = np.where(purl, 1.0, axx); axy = np.where(purl, 0.0, axy)
                ccx = np.where(purl, pcx, kcx); ccy = np.where(purl, pcy, kcy)
                a = np.where(purl, 0.36, ka); b = np.where(purl, pb, kb); hs = np.where(purl, 0.5, khs)
                dx = xs - ccx; dy = ys - ccy
                u = dx*axx + dy*axy                                           # along the leg (+ = up)
                w = dx*axy - dy*axx                                           # across
                un = u/a
                bb = b*(1.0 - 0.28*un**2)                                     # tapering ends
                prof = np.sqrt(np.clip(1.0 - (w/np.maximum(bb, 1e-3))**2, 0, 1))
                along = np.sqrt(np.clip(1.0 - un**4, 0, 1))
                hl = prof*along*hs*(1.0 - np.where(purl, 0.0, 0.22)*un)       # the lower end lies on top
                # ply twist: 3-4 plies spiralling across the leg
                ply = np.cos(2*np.pi*(u*4.2 + w/np.maximum(b, 1e-3)*1.3))
                hl = hl*(1.0 + 0.07*ply*prof)
                m = hl > h
                h = np.where(m, hl, h)
                var = np.where(m, 0.5 + 0.45*(cvar[j, i] - 0.5) + 0.06*ply + 0.10*np.tanh(rowv[j]*0.5), var)
    if rib:                                   # the purl valley floor: yarn running behind, never a black hole
        h = np.maximum(h, 0.18 + 0.05*np.cos(2*np.pi*ys/rh*2))
    # fibre fuzz: fine noise riding the yarn
    fz = _rng(seed + 1).normal(0, 1, (H, W))
    from scipy.ndimage import gaussian_filter
    fz = gaussian_filter(fz, 0.8, mode="wrap")
    h = h + 0.055*fz/fz.std()*(h > 0.05)
    return h, var, (N, M*rh), P


def _weave(N=16, P=16, twill=False, seed=11):
    """Plain weave (1/1) or 2/1 twill relief; units: thread pitch."""
    if twill:
        N = 15
    W = H = N*P
    ys, xs = (np.mgrid[0:H, 0:W] + 0.5)/P
    r = _rng(seed)
    tw = r.normal(0, 1, N)          # thread thickness jitter (warp)
    tf = r.normal(0, 1, N)          # weft
    i = np.floor(xs).astype(int) % N; j = np.floor(ys).astype(int) % N
    fx = xs - np.floor(xs) - 0.5; fy = ys - np.floor(ys) - 0.5

    def over(ii, jj):               # warp over weft at crossing (ii, jj)
        if twill:
            return ((jj - ii) % 3 != 2).astype(float)
        return ((ii + jj) % 2 == 0).astype(float)

    # warp (vertical threads): height along y blends the crossing states of rows j and j+-1
    t = fy
    jn = np.where(t >= 0, j + 1, j - 1) % N
    s = 0.5 - 0.5*np.cos(np.pi*np.abs(t)*1.0)                 # 0 at the crossing centre, 0.5 at the cell edge
    ow = over(i, j)*(1 - s) + over(i, jn)*s
    pw = np.sqrt(np.clip(1 - (fx/(0.46*(1 + 0.05*tw[i])))**2, 0, 1))
    hw = pw*(0.45 + 0.55*ow)
    t2 = fx
    inb = np.where(t2 >= 0, i + 1, i - 1) % N
    s2 = 0.5 - 0.5*np.cos(np.pi*np.abs(t2))
    of = (1 - over(i, j))*(1 - s2) + (1 - over(inb, j))*s2
    pf = np.sqrt(np.clip(1 - (fy/(0.46*(1 + 0.05*tf[j])))**2, 0, 1))
    hf = pf*(0.45 + 0.55*of)
    mw = hw >= hf
    h = np.maximum(hw, hf)
    # fibre striation along each thread
    st = np.where(mw, np.cos(2*np.pi*(ys*3.0 + fx*1.5)), np.cos(2*np.pi*(xs*3.0 + fy*1.5)))
    h = h*(1 + 0.05*st)
    var = 0.5 + np.where(mw, 0.18*np.tanh(tw[i]), 0.18*np.tanh(tf[j])) + 0.05*st
    return h, var, (N, N), P


def relief(kind):
    """(rgba float32 [H,W,4] = gx, gy, h, var ; tile (x, y) in pitches) for a weave kind, cached."""
    if kind in _TEX:
        return _TEX[kind]
    from scipy.ndimage import gaussian_filter
    if kind == "knit":
        h, var, tile, P = _knit()
    elif kind == "rib":
        h, var, tile, P = _knit(rib=True, seed=9)
    elif kind == "plain":
        h, var, tile, P = _weave()
    elif kind == "twill":
        h, var, tile, P = _weave(twill=True)
    else:
        h = np.zeros((4, 4)); var = np.full((4, 4), 0.5); tile, P = (1.0, 1.0), 4
    h = gaussian_filter(h, 0.6, mode="wrap")
    h = (h - h.min())/max(h.max() - h.min(), 1e-6)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1))*0.5*P
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0))*0.5*P
    out = np.stack([gx, gy, h, np.clip(var, 0, 1)], -1).astype(np.float32)
    _TEX[kind] = (out, tile)
    return _TEX[kind]


# ---------------------------------------------------------------------------------------------- material setup
def guess_kind(mid, md):
    s = (mid + " " + str((md or {}).get("name", ""))).lower()
    for k, v in (("rib", "rib"), ("cuff", "rib"), ("hem", "rib"), ("band", "rib"), ("knit", "knit"),
                 ("cardigan", "knit"), ("sweater", "knit"), ("blouse", "plain"), ("shirt", "plain"),
                 ("collar", "plain"), ("trouser", "twill"), ("pant", "twill"), ("denim", "twill"),
                 ("twill", "twill")):
        if k in s:
            return v
    return "plain"


def m_per_uv(A, sub):
    """Median metres per UV unit over a submesh's triangle edges (scene metres before scale_to_H)."""
    m = A.mesh
    idx = np.asarray(m["index"][sub["first_index"]:sub["first_index"] + sub["index_count"]]).reshape(-1, 3)
    if len(idx) == 0:
        return 1.0
    P = np.asarray(m["position"], np.float64)[idx]; U = np.asarray(m["uv0"], np.float64)[idx]
    e3 = np.linalg.norm(P[:, [1, 2, 0]] - P, axis=2).ravel(); e2 = np.linalg.norm(U[:, [1, 2, 0]] - U, axis=2).ravel()
    ok = e2 > 1e-7
    if not ok.any():
        return 1.0
    return float(np.median(e3[ok]/e2[ok]))


def ccfg(char):
    """juno.json mesh.cloth block: {"quality": 0..3, "fold_ao": 0..2, "fuzz_mul": x}."""
    return ((getattr(char, "cfg", {}) or {}).get("cloth") or {})


def quality_level(char):
    q = os.environ.get("JUNO_CLOTH_Q")
    if q is None:
        q = ccfg(char).get("quality", (getattr(char, "cfg", {}) or {}).get("cloth_quality", 3))
    return max(0, min(3, int(q)))


def mean_cavity(kind, cavity):
    """Tile mean of the shader's cavity AO (what a pixel covering many stitches must converge to)."""
    if cavity <= 0:
        return 1.0
    h = relief(kind)[0][..., 2]
    t = np.clip(h/0.8, 0, 1); ss = t*t*(3 - 2*t)
    return float(np.mean(1.0 + cavity*((0.35 + 0.65*ss) - 1.0)))


def _texture(char, kind):
    key = ("cloth_relief", kind)
    if key in char.tex_cache:
        return char.tex_cache[key]
    data, tile = relief(kind)
    t = char.ctx.texture((data.shape[1], data.shape[0]), 4, np.ascontiguousarray(data.astype(np.float16)).tobytes(), dtype="f2")
    t.build_mipmaps()
    t.filter = (char.mgl.LINEAR_MIPMAP_LINEAR, char.mgl.LINEAR)
    try:
        t.anisotropy = 8.0
    except Exception:
        pass
    char.tex_cache[key] = (t, tile)
    return char.tex_cache[key]


def setup(char, u, texs, mid, md, sub):
    """Called by MeshCharacter for every cloth draw: adds the relief texture (unit 7; 5/6 = render-light shadow compare + probe) and the cloth uniforms."""
    pr = (md or {}).get("params", {}) or {}
    kind = str(pr.get("weave") or guess_kind(mid, md)).lower()
    if kind not in KINDS:
        kind = "plain"
    d = KIND_DFL[kind]
    stitch = float(pr.get("stitch_mm", d[0]))*1e-3
    mpu = float(pr.get("uv_m_per_unit") or 0.0) or m_per_uv(char.A, sub)
    depth = float(pr.get("weave_depth", 1.0))*d[1]
    rot = np.radians(float(pr.get("weave_rot_deg", 0.0)))
    tex, tile = _texture(char, kind)
    texs[7] = tex
    lv, steps, taps = QUALITY[quality_level(char)]
    if kind == "none":
        lv = 0
    sc = mpu/stitch
    u.update(
        tCloth=7,
        uClothQ=(float(lv), float(steps), float(taps), float(ccfg(char).get("fold_ao", 1.0)) if lv > 0 else 0.0),
        uClothX=(float(np.cos(rot)*sc), float(-np.sin(rot)*sc), float(np.sin(rot)*sc), float(np.cos(rot)*sc)),
        uClothT=(1.0/float(tile[0]), 1.0/float(tile[1]), float(depth), float(stitch)),
        uClothS=(float(pr.get("sheen_roughness", d[3])), float(pr.get("fuzz", d[4]))*float(ccfg(char).get("fuzz_mul", 1.0)),
                 float(pr.get("color_var", d[5])), float(d[6])),
        uClothW=(float(pr.get("wrap", d[2])), float(KINDS[kind]), mean_cavity(kind, d[6]) if kind != "none" else 1.0,
                 float(pr.get("wash", 0.5))),
    )
    char.cloth_info = getattr(char, "cloth_info", {})
    char.cloth_info[mid] = dict(kind=kind, stitch_mm=stitch*1e3, m_per_uv=mpu, quality=lv)


def set_quality(char, level):
    """Ladder hook: re-level every cloth draw (0..3) without recompiling."""
    lv, steps, taps = QUALITY[max(0, min(3, int(level)))]
    for d in getattr(char, "draws", []):
        if d.get("type") == "cloth" and "uClothQ" in d["u"]:
            kind_none = d["u"].get("uClothW", (0, 0))[1] == 0
            fo = d["u"]["uClothQ"][3]
            d["u"]["uClothQ"] = (0.0 if kind_none else float(lv), float(steps), float(taps), fo)


def save_relief_pngs(out_dir):
    """Proposal for ASSET: the generated reliefs as 8-bit tangent normal maps + height + colour variation PNGs."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for k in ("knit", "rib", "plain", "twill"):
        data, tile = relief(k)
        g = data[..., :2]*KIND_DFL[k][1]
        n = np.concatenate([-g, np.ones_like(g[..., :1])], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        Image.fromarray(np.clip((n*0.5 + 0.5)*255, 0, 255).astype(np.uint8)[::-1]).save(f"{out_dir}/{k}_normal.png")
        Image.fromarray(np.clip(data[..., 2]*255, 0, 255).astype(np.uint8)[::-1]).save(f"{out_dir}/{k}_height.png")
        Image.fromarray(np.clip(data[..., 3]*255, 0, 255).astype(np.uint8)[::-1]).save(f"{out_dir}/{k}_var.png")
