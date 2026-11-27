#!/usr/bin/env python3
"""where a worn prop sits on a character's head, DERIVED from the head's own geometry.

The scene / props file is shared by every character on the Muse skeleton (animals/<mascot>.props.json -> muse.props.json),
and its afford.wear offsets were tuned on the Muse's head.  A head of another size or shape (a lizard, a candy shell, a
snowman with twig hair, a top hat) needs its own wear anchor.  This tool reads the character's SDF back through the
renderer's own shader (tools/props_probe.py), in the HEAD FRAME the worn prop rides (PROPS_HEAD_FN), and solves:

  The Muse's hand-tuned fit is read as RULES against her own head (reference()), then applied to each head:
  hat      scale: the head's width (its widest slice) relative to the Muse's (never bigger than the scene's hat);
           offset: centred over the head where the brim sits, at the Muse's rise above the TOP of the head, then lifted off
           anything it would cut into; if that leaves the default view (a head already under a tall hat of its own) it
           goes down over that instead, opened to what it goes round.  Thin things (hair twigs, lashes, feather tips) do not hold a hat up: features
           thinner than 2*THIN are ignored.
  glasses  scale: the lens centres on the eyes (eye spacing relative to the Muse's); offset: on the eye line, pushed
           back until the frame / lenses meet the face (eyeballs included), plus the Muse's stand-off.

  python tools/wear_fit.py                     # every character with a "props_wear" block or in --chars: print the table
  python tools/wear_fit.py --write             # ... and write renderer/animals/<name>.props.override.json
  python tools/wear_fit.py --chars muse        # the rules back on the Muse = her own numbers (never written)

Output units: offsets in body heights H in the head frame (as afford.wear.offset), scale = a factor on the prop's size
while it is worn (engine/props.glsl PROPS_WEARFIT; 1 = as in the scene).  Poses do not enter: the prop rides the head
frame, so a fit at rest holds for every head turn / tilt.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]

# the characters on the Muse skeleton whose head is not the Muse's (anim/mascots.py MASCOTS): they get a block
FIT_CHARS = ("gekkie", "hooty", "doughlad", "bonbon", "pneu", "goober", "jade", "ruby", "flurry")
STEP = 0.02                 # voxel (scene units)
BOX = ((-1.5, 1.5), (-1.1, 2.3), (-1.5, 1.5))     # head-frame box sampled (scene units)
THIN = 0.10                 # features thinner than 2*THIN (tufts, twigs, a bow, lashes) do not carry a hat
HAT_MIN, GL_MIN, GL_MAX = 0.45, 0.40, 1.30
SINK = 0.012                # how far a worn thing may press into the head (scene units)

VOL_MAIN = """
out vec4 fragOut;
uniform vec3 uLo; uniform float uStep; uniform float uZi;
void main(){
  setInit(); charInit();
  vec3 hp; mat3 hf;
  PROPS_HEAD_FN(hp, hf);
  vec3 g = uLo + uStep*vec3(floor(gl_FragCoord.xy), uZi);
  vec3 pw = rootFwd(hp + hf*g);
  charEvalA(pw);
  fragOut = vec4(min(gcBare, gcBase - gcLen), 0.0, 0.0, 1.0);
}
"""
EYE_MAIN = """
out vec4 fragOut;
void main(){
  setInit(); charInit();
  vec3 hp; mat3 hf;
  PROPS_HEAD_FN(hp, hf);
  int c = int(gl_FragCoord.x);
  vec3 e = vec3(0.0); float r = 0.0;
#ifdef EYE_C
  e = EYE_C; r = EYE_R;
#endif
#ifdef MR_HEAD_SCALE
  e *= MR_HEAD_SCALE; r *= MR_HEAD_SCALE;
#endif
  // the eye in the head frame: head space -> local (gHO + gHR^T q) -> head frame
  vec3 el = transpose(hf)*(gHO + transpose(gHR)*e - hp);
  fragOut = c == 0 ? vec4(el, r) : vec4(rootFwd(hp), 0.0);
}
"""


def volume(pr):
    """The character's SDF on the head-frame grid at rest: (nx, ny, nz) array + the axes."""
    ax = [np.arange(lo, hi + 1e-6, STEP) for lo, hi in BOX]
    nx, ny, nz = (len(a) for a in ax)
    p, vao = pr.prog("wear_vol", VOL_MAIN)
    from anim.schema import NPARAMS
    pr._setP(p, np.zeros(NPARAMS, "f4"))
    p["uLo"].value = (BOX[0][0], BOX[1][0], BOX[2][0]); p["uStep"].value = STEP
    tex = pr.ctx.texture((nx, ny), 4, dtype="f4"); fb = pr.ctx.framebuffer([tex])
    fb.use(); pr.ctx.viewport = (0, 0, nx, ny)
    V = np.empty((nx, ny, nz), "f4")
    for k in range(nz):
        p["uZi"].value = float(k)
        vao.render(pr.mgl.TRIANGLE_STRIP)
        V[:, :, k] = np.frombuffer(fb.read(components=4, dtype="f4"), "f4").reshape(ny, nx, 4)[..., 0].T
    return V, ax


def eye(pr):
    p, vao = pr.prog("wear_eye", EYE_MAIN)
    from anim.schema import NPARAMS
    pr._setP(p, np.zeros(NPARAMS, "f4"))
    tex = pr.ctx.texture((2, 1), 4, dtype="f4"); fb = pr.ctx.framebuffer([tex])
    fb.use(); pr.ctx.viewport = (0, 0, 2, 1)
    vao.render(pr.mgl.TRIANGLE_STRIP)
    a = np.frombuffer(fb.read(components=4, dtype="f4"), "f4").reshape(2, 4)
    return a[0, :3].astype(float), float(a[0, 3]), a[1, :3].astype(float)


# ---- the models' undersides, as in engine/props.glsl (unit frame), vectorised
def _rbox2(px, py, bx, by, r):
    ox, oy = np.abs(px) - bx + r, np.abs(py) - by + r
    return np.hypot(np.maximum(ox, 0), np.maximum(oy, 0)) + np.minimum(np.maximum(ox, oy), 0) - r


def hat_sdf(q):                 # pk_hat without the ribbon and bow
    r = np.hypot(q[..., 0], q[..., 2]); y = q[..., 1]
    tp = 1.0 + 0.10*np.clip(y - 0.03, -0.3, 0.3)
    outer = _rbox2(r*tp, y - 0.03, 0.60, 0.25, 0.14)
    inner = _rbox2(r*tp, y + 0.04, 0.55, 0.25, 0.10)
    crown = np.maximum(outer, -inner)
    e = np.maximum(r - 0.55, 0.0)
    yb = -0.205 - 0.17*e*e
    brim = np.maximum(np.maximum(np.abs(y - yb) - 0.024, r - 1.0), 0.53 - r)*0.92
    return np.minimum(crown, brim)


def glasses_sdf(q):             # pk_glasses' front: rings, lenses, bridge (the arms pass through / beside a head)
    sx, y, z = np.abs(q[..., 0]), q[..., 1], q[..., 2]
    lr = np.hypot(sx - 0.46, y)
    ring = np.hypot(lr - 0.255, z) - 0.036
    lens = np.maximum(lr - 0.26, np.abs(z) - 0.016)
    return np.minimum(ring, lens)


HAT_TOP = 0.28                  # the crown's top (unit frame)
VIEW_MARGIN = 0.03              # scene units kept between the hat's top and the top of the default view
HAT_OPEN = 0.55                 # the crown opening's radius (unit frame)
LENS_X = 0.46                   # lens centre (unit frame)


def _head(V, ax):
    """The head's firm solid (thin features opened away, the piece round the head frame's origin above the chin)."""
    from scipy import ndimage
    X, Y, Z = ax
    solid = V < 0.0
    core = V < -THIN
    dist = ndimage.distance_transform_edt(~core, sampling=STEP)
    firm = (dist <= THIN) & solid
    iy0 = int(np.searchsorted(Y, -0.20))
    top_reg = firm.copy(); top_reg[:, :iy0, :] = False
    lab, n = ndimage.label(top_reg)
    io = (int(np.searchsorted(X, 0.0)), int(np.searchsorted(Y, 0.0)), int(np.searchsorted(Z, 0.0)))
    l0 = lab[io]
    if l0 == 0:      # the origin is not inside the head: the biggest piece
        l0 = 1 + int(np.argmax(ndimage.sum(top_reg, lab, range(1, n + 1))))
    head = lab == l0
    ys = np.where(head.any(axis=(0, 2)))[0]
    return solid, firm, head, float(Y[ys.max()]) + 0.5*STEP


def _slice(head, ax, y):       # the head's slice at height y: min half extent, centre z
    X, Y, Z = ax
    j = int(np.clip(np.searchsorted(Y, y), 0, len(Y) - 1))
    sl = head[:, j, :]
    if not sl.any():
        return 0.0, 0.0
    xi = np.where(sl.any(axis=1))[0]; zi = np.where(sl.any(axis=0))[0]
    hx = 0.5*(X[xi.max()] - X[xi.min()]) + 0.5*STEP; hz = 0.5*(Z[zi.max()] - Z[zi.min()]) + 0.5*STEP
    return min(hx, hz), 0.5*(Z[zi.max()] + Z[zi.min()])


def _width(head, ax, y_top):      # the head's size: its widest slice (min of the half extents), chin line up
    return max(_slice(head, ax, y)[0] for y in np.arange(-0.20, y_top, 2*STEP))


def _hat_pen(firm, ax, c, s, rx):
    X, Y, Z = ax
    pts = np.argwhere(firm & (Y[None, :, None] > c[1] - 0.5*s) & (Y[None, :, None] < c[1] + 0.4*s))
    if not len(pts):
        return 1.0
    d = np.stack([X[pts[:, 0]], Y[pts[:, 1]], Z[pts[:, 2]]], 1) - np.array(c)
    cr, sr = np.cos(rx), np.sin(rx)
    q = np.stack([d[:, 0], cr*d[:, 1] + sr*d[:, 2], -sr*d[:, 1] + cr*d[:, 2]], 1)/s      # R^T d (prEuler: rx about x)
    return float(hat_sdf(q).min()*s)


def _gl_pen(solid, ax, c, s):
    X, Y, Z = ax
    pts = np.argwhere(solid & (np.abs(Y[None, :, None] - c[1]) < 0.40*s) & (np.abs(X[:, None, None]) < 0.85*s))
    if not len(pts):
        return 1.0
    q = (np.stack([X[pts[:, 0]], Y[pts[:, 1]], Z[pts[:, 2]]], 1) - np.array(c))/s
    return float(glasses_sdf(q).min()*s)


def _gl_contact(solid, ax, ey, s, zmin):
    z = float(ax[2][-1])
    while z > zmin and _gl_pen(solid, ax, (0.0, ey, z - 0.004), s) > 0.0:
        z -= 0.004
    return z


def reference(V, ax, eye_c, H, hat, glasses):
    """The Muse's hand-tuned fit, read as RULES against her own head (what every other head then gets)."""
    solid, firm, head, y_top = _head(V, ax)
    s0 = float(hat["scale"])*H; w = hat["afford"]["wear"]
    hy, hz = float(w["offset"][1])*H, float(w["offset"][2])*H
    rx = 6.2831853*float(w.get("rot", [0, 0, 0])[0])
    depth = y_top - (hy - 0.205*s0)                    # the brim's inner edge sits this far below the top of the head
    hw, cz = _slice(head, ax, y_top - depth)
    R = {"hat_rise": hy - y_top, "hat_depth": depth, "hat_hw": _width(head, ax, y_top), "hat_dz": hz - cz, "hat_rx": rx,
         "hat_pen": min(_hat_pen(firm, ax, (0.0, hy, hz), s0, rx), 0.0)}
    g0 = float(glasses["scale"])*H; gw = glasses["afford"]["wear"]["offset"]
    zc = _gl_contact(solid, ax, float(eye_c[1]), g0, float(eye_c[2]) - 0.2)
    R.update(eye_x=abs(float(eye_c[0])), gl_dy=float(gw[1])*H - float(eye_c[1]), gl_dz=float(gw[2])*H - zc)
    return R


def frame_top(character, hp):
    """How far above the head frame's origin the DEFAULT view ends (scene units, at the character's plane): a worn hat
    must stay in the picture.  From the character's own camera (animals/<c>.json: pos, target, vertical fov)."""
    import math
    cam = json.load(open(os.path.join(ROOT, "animals", f"{character}.json")))["camera"]
    p, t = cam["pos"], cam["target"]
    dz = p[2] - t[2]
    ang = math.atan2(t[1] - p[1], dz) + math.radians(float(cam["fov"]))/2
    return p[1] + (p[2] - float(hp[2]))*math.tan(ang) - float(hp[1])


def fit(V, ax, eye_c, eye_r, H, hat, glasses, R, ytop_view=None):
    """hat / glasses: the scene's props entries; R = reference().  -> {"hat": {...}, "glasses": {...}, "meas": {...}}"""
    solid, firm, head, y_top = _head(V, ax)
    out = {"meas": {"head_top": round(y_top, 4)}}
    # ---- the hat: sized to the head where its brim sits (as on the Muse, never bigger than the scene's hat), centred
    # over the head there, at the Muse's rise above the top of the head; then lifted off anything it would cut into
    s0 = float(hat["scale"])*H
    hw = _width(head, ax, y_top)
    k = float(np.clip(hw/R["hat_hw"], HAT_MIN, 1.0))
    s = s0*k
    _, cz = _slice(head, ax, y_top - R["hat_depth"]*k)
    y, z = y_top + R["hat_rise"]*k, cz + R["hat_dz"]*k
    lift = 0.0
    while _hat_pen(firm, ax, (0.0, y, z), s, R["hat_rx"]) < R["hat_pen"]*k - SINK and lift < 0.6:
        y += 0.004; lift += 0.004
    # ... and it stays in the default view: a head that already wears something tall (a chef's hat, a top hat) would
    # carry it out of the picture — there it goes down OVER that, opened to what it goes round (never past the scene's size)
    capped = False
    for _ in range(3):
        if ytop_view is None or y + HAT_TOP*s <= ytop_view - VIEW_MARGIN:
            break
        capped = True
        y = ytop_view - VIEW_MARGIN - HAT_TOP*s
        hw2, cz = _slice(head, ax, y - 0.205*s)
        k = float(np.clip(hw2/(HAT_OPEN*s0), k, 1.0)); s = s0*k
        z = cz + R["hat_dz"]*k
    out["hat"] = {"offset": [0.0, round(y/H, 4), round(z/H, 4)], "scale": round(k, 4)}
    out["meas"].update(hat_head_hw=round(hw, 4), hat_lift=round(lift, 3), brim_to_crown=round((y - 0.205*s) - y_top, 4),
                       hat_in_view_capped=capped)
    # ---- the glasses: the lenses on the eyes (sized by the eye spacing), on the eye line, resting on the face
    if eye_r > 0:
        g0 = float(glasses["scale"])*H
        ex, ey, ez = (float(v) for v in eye_c)
        k = float(np.clip(abs(ex)/R["eye_x"], GL_MIN, GL_MAX))
        s = g0*k
        zc = _gl_contact(solid, ax, ey + R["gl_dy"]*k, s, ez - 0.2)
        z = zc + R["gl_dz"]*k
        out["glasses"] = {"offset": [0.0, round((ey + R["gl_dy"]*k)/H, 4), round(z/H, 4)], "scale": round(k, 4)}
        out["meas"].update(eye=[round(ex, 4), round(ey, 4), round(ez, 4)], eye_r=round(eye_r, 4), glasses_contact_z=round(zc, 4))
    return out


def scene_props():
    cfg = json.load(open(os.path.join(ROOT, "animals", "muse.props.json")))
    by = {p.get("id"): p for p in cfg["props"]}
    return by.get("hat"), by.get("glasses")


def sample(character, backend="cpu", threads=6, cache=None):
    cp = cache and os.path.join(cache, f"wearvol_{character}.npz")
    if cp and os.path.isfile(cp):
        z = np.load(cp)
        return z["V"], [z["X"], z["Y"], z["Z"]], z["eye"], float(z["eye_r"]), z["hp"]
    from props_probe import Probe
    pr = Probe(character, backend, threads)
    V, ax = volume(pr)
    ec, er, hp = eye(pr)
    if cp:
        os.makedirs(cache, exist_ok=True)
        np.savez_compressed(cp, V=V, X=ax[0], Y=ax[1], Z=ax[2], eye=ec, eye_r=er, hp=hp)
    return V, ax, ec, er, hp


def main():
    import props_defs
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=",".join(FIT_CHARS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--backend", default="cpu"); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cache", default=None, help="keep the sampled volumes here (re-fit without the shader)")
    a = ap.parse_args()
    hat, gl = scene_props()
    Vm, axm, ecm, erm, hpm = sample("muse", a.backend, a.threads, a.cache)
    Hm = props_defs.units("muse")[0]
    R = reference(Vm, axm, ecm, Hm, hat, gl)
    print("reference (muse)", json.dumps({k: round(float(v), 4) for k, v in R.items()}), flush=True)
    res = {}
    for c in [x for x in a.chars.split(",") if x]:
        V, ax, ec, er, hp = sample(c, a.backend, a.threads, a.cache)
        H = props_defs.units(c)[0]
        r = fit(V, ax, ec, er, H, hat, gl, R, frame_top(c, hp))
        # where the page thinks the head is (body.json head_scene) vs the head frame the worn prop rides: the hand
        # carries the prop to the page's head point + offset + this
        try:
            hs = json.load(open(os.path.join(ROOT, "animals", f"{c}.body.json")))["head_scene"]
            carry = [round((float(hp[i]) - float(hs[i]))/H, 4) for i in range(3)]
        except (OSError, KeyError, ValueError):
            carry = None
        res[c] = r
        print(c, json.dumps(r), "carry", carry, flush=True)
        if a.write and c in FIT_CHARS:
            ov = {"base": "muse", "_doc": "worn props on THIS head, derived from its geometry by "
                  "renderer/tools/wear_fit.py (re-run it after the head changes; never hand-edit). Deep-merged over the "
                  "shared scene file at load time (renderer/humans/charfiles.py): afford.wear.offset = H, head frame; "
                  "scale = size factor while worn; carry = head frame minus the page's head point (H)",
                  "character": c, "props": []}
            for pid in ("hat", "glasses"):
                if pid in r:
                    w = dict(r[pid])
                    if carry is not None:
                        w["carry"] = carry
                    ov["props"].append({"id": pid, "afford": {"wear": w}})
            with open(os.path.join(ROOT, "animals", f"{c}.props.override.json"), "w") as f:
                json.dump(ov, f, indent=1); f.write("\n")
    return res


if __name__ == "__main__":
    main()
