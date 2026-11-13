"""tvdsl.verify -- did a change to the picture actually change the DRAWING?

The agent snapshots the live Doc before a follow-up patch, applies the patch, and compares the two pictures rendered
at the same (settled) moment: once without any text (geometry) and once with it. A patch that only rewrote captions
(field case r6: "make the lens more curved" changed the formula text, the lens stayed the same) has geometry ~0 and
is escalated by the agent. Also renders still frames of a Doc for the careful model to LOOK at.
"""
import copy
import os

import numpy as np

from .render import Renderer

_R = {}


def _renderer(w, h):
    r = _R.get((w, h))
    if r is None:
        r = _R[(w, h)] = Renderer(w, h, glow=0.0, autofit=False)
    return r


def snapshot(doc):
    """an independent copy of a Doc (expressions are closures: shared, which is fine -- they are immutable)"""
    return copy.deepcopy(doc)


def _gray(img):
    # per-channel (a colour change -- green light to red -- is a change of the drawing; grey levels barely move)
    return img.astype(np.int16)


def picture_change(before, after, now, w=256, h=144, thr=40):
    """{'geom': fraction of pixels whose geometry changed, 'text': fraction changed by texts only, 'any': both}
    before/after: Docs; both rendered at `now` (wall seconds on the Doc clock), so the animation phase is the same"""
    R = _renderer(w, h)
    out = {}
    if getattr(before, "scene_at", None) == getattr(after, "scene_at", None):
        # the same scene CLOCK for both (the change's `hold` re-phases the loop, so a moving part --
        # a test charge stepping outwards -- read as "the drawing changed" when the patch had drawn nothing)
        before = copy.copy(before)
        before.t0 = after.t0
        before._phase = getattr(after, "_phase", None)
        before.loop = after.loop
    try:
        ga = _gray(R.render(before, now, texts=False))
        gb = _gray(R.render(after, now, texts=False))
        ta = _gray(R.render(before, now))
        tb = _gray(R.render(after, now))
    except Exception as e:  # noqa: BLE001
        return {"geom": None, "text": None, "any": None, "error": f"{type(e).__name__}: {e}"[:120]}
    dg = (np.abs(ga - gb) > thr).any(axis=2)
    dt = (np.abs(ta - tb) > thr).any(axis=2)
    n = float(w * h)
    out["geom"] = round(float(np.count_nonzero(dg)) / n, 5)
    out["any"] = round(float(np.count_nonzero(dt)) / n, 5)
    out["text"] = round(max(0.0, out["any"] - out["geom"]), 5)
    if np.count_nonzero(dg):
        ys, xs = np.nonzero(dg)
        out["box"] = [int(xs.min()) / w, int(ys.min()) / h, int(xs.max()) / w, int(ys.max()) / h]
    return out


def stills(doc, times, path_prefix, w=512, h=288):
    """render the Doc at the given wall times -> list of PNG paths (one image per time, side by side if several)"""
    from PIL import Image
    R = Renderer(w, h)
    imgs = []
    for T in times:
        try:
            imgs.append(R.render(doc, T))
        except Exception:  # noqa: BLE001
            continue
    if not imgs:
        return []
    if len(imgs) == 1:
        M = Image.fromarray(imgs[0])
    else:
        M = Image.new("RGB", (w * len(imgs) + 8 * (len(imgs) - 1), h), (90, 90, 90))
        for k, im in enumerate(imgs):
            M.paste(Image.fromarray(im), (k * (w + 8), 0))
    p = path_prefix + ".png"
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    M.save(p)
    return [p]


def lint(doc, now, before=None):
    """objects that draw NOTHING at two settled moments, or texts that print '?' (an undefined name, a NaN formula):
    [(id, source line, why)] -- the agent sends these back to the model for a one-line fix"""
    from .scene import Frame
    out = []
    if doc is None or doc.empty:
        return out
    seen = set()
    try:
        frames = [Frame(doc, now + dt, 32.0) for dt in (2.0, 6.0)]
    except Exception as e:  # noqa: BLE001
        return [(None, "", f"the program fails to evaluate: {type(e).__name__}")]
    for o in list(doc.objs.values()):
        if o.kind == "group" or o.src in seen:
            continue
        drew, q = False, False
        for fr in frames:
            try:
                g = fr._geom(o.id)
            except Exception:  # noqa: BLE001
                g = None
            if g and g.get("marks"):
                drew = True
                src_q = (o.text or "").count("?") + (str(o.props.get("label")) if o.props.get("label") is not None else "").count("?")
                for m in g["marks"]:
                    if m.kind == "text" and m.text and m.text.count("?") > src_q and "{" in (o.src or ""):
                        q = True
                    if m.kind == "text" and m.text and ("{" in m.text or "}" in m.text):
                        q = "raw"           # a broken {...} template printed as source ('{{m>0.5 ? ')
        off = None
        if drew and before is not None and o.kind == "plot" and isinstance(o.props.get("on"), str):
            # only a curve THIS change pushed off its axes (a parabola leaving its box at both ends is normal)
            off = _off_axes(frames, o)
            if off and o.id in before.objs:
                try:
                    fb = [Frame(before, now + dt, 32.0) for dt in (2.0, 6.0)]
                    if _off_axes(fb, before.objs[o.id]):
                        off = None
                except Exception:  # noqa: BLE001
                    off = None
        if drew and not off:
            off = _off_screen(frames, o, doc.view)
        if not drew:
            out.append((o.id, o.src, "draws nothing (an undefined name, or a value that is not a number)"))
            seen.add(o.src)
        elif off:
            out.append((o.id, o.src, off))
            seen.add(o.src)
        elif q == "raw":
            out.append((o.id, o.src, "prints its {...} template as raw text (broken braces or quotes inside them: use "
                                     "single {expr} and single quotes for words)"))
            seen.add(o.src)
        elif q:
            out.append((o.id, o.src, "prints '?' (a {...} in its text uses an undefined name or is not a number)"))
            seen.add(o.src)
    return out[:8]


def _off_axes(frames, o):
    """a curve drawn on an axes box whose values leave the box (fu3 f_population: 'double twice as fast'
    sent the curve off the top of a 0..160 axis, the 1024 at 10 h never visible) -> the axes line must change"""
    if o.kind != "plot":
        return None
    on = o.props.get("on")
    if not isinstance(on, str):
        return None
    worst, ymax, ymin, yr = 0.0, None, None, None
    for fr in frames:
        try:
            mp = fr._axes_map(on)
            g = fr._geom(o.id)
        except Exception:  # noqa: BLE001
            return None
        if mp is None or not g or g.get("overshoot") is None:
            continue
        if g["overshoot"] > worst:
            worst = g["overshoot"]
            ymin, ymax = g["data_y"]
            yr = mp[1]
    if worst < 0.15:
        return None
    return (f"leaves its axes box {on} (the curve reaches {ymin:.3g}..{ymax:.3g}, beyond the axes y range "
            f"{min(yr):g}..{max(yr):g}) -- re-declare axes {on} with a y range that holds the whole curve (compute "
            f"it from the params), or rescale the curve; keep the plot")


_FREE_KINDS = {"beam", "ray", "rays3", "image", "group", "axes", "grid", "wave", "ripple", "field"}


def _off_screen(frames, o, view):
    """an object that is ENTIRELY off the screen at both moments (a text placed at y=6 on a -4.5..4.5
    screen, a part drawn at x=-14) -> one repair line"""
    if o.kind in _FREE_KINDS or o.props.get("on") is not None or o.props.get("in") is not None:
        return None
    x0, x1, y0, y1 = min(view[0], view[1]), max(view[0], view[1]), min(view[2], view[3]), max(view[2], view[3])
    mx, my = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
    boxes = []
    for fr in frames:
        try:
            g = fr._geom(o.id)
        except Exception:  # noqa: BLE001
            return None
        pts = [p for m in (g or {}).get("marks") or [] for p in (m.pts or []) if p and len(p) == 2]
        if not pts:
            return None
        bx = (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))
        if bx[1] >= x0 - mx and bx[0] <= x1 + mx and bx[3] >= y0 - my and bx[2] <= y1 + my:
            return None                     # some of it is on the screen
        boxes.append(bx)
    b = boxes[-1]
    return (f"is entirely off the screen (it sits at x {b[0]:.3g}..{b[1]:.3g}, y {b[2]:.3g}..{b[3]:.3g}; the screen is "
            f"x {x0:g}..{x1:g}, y {y0:g}..{y1:g}) -- move it onto the screen where it belongs")
