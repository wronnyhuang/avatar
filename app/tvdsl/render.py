"""tvdsl.render -- deterministic anti-aliased rasterizer: Doc @ wall time -> RGB uint8 frame.

    R = Renderer(512, 288)
    rgb = R.render(doc, now)          # HxWx3 uint8;  R.idle(now) = the idle look
Uses OpenCV (LINE_AA, sub-pixel shift) for geometry and Pillow for text (cached glyph runs). No other deps.
"""
import math
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene import Frame, point_along, path_length

SHIFT = 4
SUB = 1 << SHIFT
_K9 = np.ones((9, 9), np.uint8)
TEXT_FLOOR_PX = 12          # no text smaller than this at 512 px wide (scales with the width)
BG = (0.027, 0.035, 0.06)
_FONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
_BOLD_PATHS = [
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_fonts = {}


def _font(px, bold=False):
    key = (int(px), bold)
    f = _fonts.get(key)
    if f is None:
        f = None
        for p in (_BOLD_PATHS if bold else []) + _FONT_PATHS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, max(6, int(px)))
                    break
                except OSError:
                    continue
        if f is None:
            f = ImageFont.load_default()
        _fonts[key] = f
    return f


_RICH = re.compile(r"([_^])(\[[^\]]*\]|-?[A-Za-z0-9+\-−′'*]?[A-Za-z0-9′'*]*)")


def rich_runs(s):
    """'H_2O', 'x^2', 'e^[-x]', 'v_[max]' -> [(text, level)] level 0 normal, -1 sub, +1 sup"""
    out = []
    i = 0
    for m in _RICH.finditer(s):
        if m.start() > i:
            out.append((s[i:m.start()], 0))
        tok = m.group(2)
        if not tok:
            out.append((m.group(0), 0))
        else:
            if tok.startswith("["):
                tok = tok[1:-1]
            else:
                # a single sub/superscript token: digits, or one letter, or a signed number
                # (a lowercase word of 2+ letters is a whole subscript/superscript: v_max, F_net, n_air)
                mm = re.match(r"[a-z]{2,}(?![A-Za-z0-9])|-?[0-9]+|[A-Za-z0-9+\-−′'*]", tok)
                rest = tok[mm.end():] if mm else ""
                tok = mm.group(0) if mm else tok
                out.append((tok.replace("-", "−"), -1 if m.group(1) == "_" else 1))
                if rest:
                    out.append((rest, 0))
                i = m.end()
                continue
            out.append((tok.replace("-", "−"), -1 if m.group(1) == "_" else 1))
        i = m.end()
    if i < len(s):
        out.append((s[i:], 0))
    return out


def _norm_text(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


class _TextCache:
    def __init__(self, cap=600):
        self.d = {}
        self.cap = cap

    def get(self, text, px, bold):
        key = (text, int(px), bold)
        v = self.d.get(key)
        if v is None:
            v = self._make(text, int(px), bold)
            if len(self.d) > self.cap:
                self.d.clear()
            self.d[key] = v
        return v

    def _make(self, text, px, bold):
        lines = text.split("\\n") if "\\n" in text else text.split("\n")
        f0 = _font(px, bold)
        fs = _font(max(6, int(px * 0.68)), bold)
        asc, desc = f0.getmetrics()
        lh = int((asc + desc) * 1.08)
        widths = []
        runs_l = []
        for ln in lines:
            runs = rich_runs(ln)
            runs_l.append(runs)
            w = 0
            for t, lv in runs:
                w += (fs if lv else f0).getlength(t)
            widths.append(int(math.ceil(w)))
        W = max(widths + [1]) + 4
        H = lh * len(lines) + int(px * 0.35) + 2
        img = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(img)
        y = int(px * 0.12)
        for runs, w in zip(runs_l, widths):
            x = (W - w) / 2
            for t, lv in runs:
                f = fs if lv else f0
                dy = -int(px * 0.32) if lv > 0 else int(px * 0.38) if lv < 0 else 0
                dr.text((x, y + dy + (int(px * 0.22) if lv else 0)), t, fill=255, font=f)
                x += f.getlength(t)
            y += lh
        a = np.asarray(img, dtype=np.float32) / 255.0
        return a


class Renderer:
    def __init__(self, w=512, h=288, glow=0.42, autofit=True):
        self.w, self.h = int(w), int(h)
        self.autofit = autofit
        self.glow_k = glow
        self.texts = _TextCache()
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cx, cy = (self.w - 1) / 2, (self.h - 1) / 2
        r = np.sqrt(((xx - cx) / (self.w * 0.62)) ** 2 + ((yy - cy) / (self.h * 0.62)) ** 2)
        self.vig = np.clip(1.08 - 0.42 * r * r, 0.55, 1.0)[..., None]
        self._vig3 = np.ascontiguousarray(np.repeat(self.vig, 3, axis=2).astype(np.float32))
        self.last_marks = 0
        self.last_texts = []
        self.last_leaders = []

    # ---- coordinate mapping
    def _map(self, view):
        x0, x1, y0, y1 = view
        vw, vh = max(1e-6, abs(x1 - x0)), max(1e-6, abs(y1 - y0))
        k = min(self.w / vw, self.h / vh)
        ox = (self.w - vw * k) / 2 - min(x0, x1) * k
        oy = (self.h - vh * k) / 2 + max(y0, y1) * k
        return k, ox, oy

    # ---- automatic view fit: solid shapes and texts that would leave the screen pull the view out (eased)
    def _content_box(self, doc, now):
        """union box of the solid shapes / dots / texts over the next few seconds of the picture (None if empty)"""
        bx = [math.inf, -math.inf, math.inf, -math.inf]
        x0, x1, y0, y1 = doc.view
        vw, vh = abs(x1 - x0), abs(y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for dt in (0.0, 2.0, 4.0, 7.0):
            try:
                marks = Frame(doc, now + dt, self.w / max(1e-6, vw), self.w).marks()
            except Exception:  # noqa: BLE001
                continue
            for m in marks:
                if m.alpha < 0.05 or not m.pts:
                    continue
                if m.kind == "text":
                    if not m.text or m.soft in ("callout", True, "tick"):
                        # callouts / automatic labels / axis numbers are kept on screen by the renderer's placement
                        continue
                    p = m.pts[0]
                    lines = str(m.text).split("\n")
                    tw = m.size * 0.56 * max(len(re.sub(r"[_^\[\]{}]", "", ln)) for ln in lines)
                    th = m.size * 1.15 * len(lines)
                    if tw > vw * 0.95:          # a text wider than the screen is wrapped by the renderer, not fitted
                        th *= math.ceil(tw / (vw * 0.95))
                        tw = vw * 0.95
                    an = m.anchor or "c"
                    lx = p[0] if "l" in an else (p[0] - tw if "r" in an else p[0] - tw / 2)
                    ly = p[1] - th if "t" in an else (p[1] if "b" in an else p[1] - th / 2)
                    pts = ((lx, ly), (lx + tw, ly + th))
                elif m.fill is not None or m.closed:
                    pts = m.pts
                else:
                    continue
                for q in pts:
                    if not (math.isfinite(q[0]) and math.isfinite(q[1])):
                        continue
                    if abs(q[0] - cx) > 2.0 * vw or abs(q[1] - cy) > 2.0 * vh:
                        continue                      # far off on purpose (or broken): never chase it
                    bx[0], bx[1] = min(bx[0], q[0]), max(bx[1], q[0])
                    bx[2], bx[3] = min(bx[2], q[1]), max(bx[3], q[1])
        return bx if bx[0] <= bx[1] and bx[2] <= bx[3] else None

    def fit_view(self, doc, now):
        x0, x1, y0, y1 = doc.view
        if not (x1 > x0 and y1 > y0):
            return doc.view
        key = (id(doc), doc.version, doc.scene_at, x0, x1, y0, y1)
        st = getattr(self, "_fit", None)
        if st is None or st["scene"] != (id(doc), doc.scene_at):
            st = self._fit = {"scene": (id(doc), doc.scene_at), "key": None, "from": doc.view, "to": doc.view,
                              "at": now, "calc": -1e9}
        if st["key"] != key and (now - st["calc"] >= 0.25 or st["key"] is None):
            st["calc"] = now
            st["key"] = key
            b = self._content_box(doc, now)
            m = 0.25 * (y1 - y0) / 9.0 + 0.15
            if b is None:
                to = doc.view
            else:
                nx0, nx1 = min(x0, b[0] - m), max(x1, b[1] + m)
                ny0, ny1 = min(y0, b[2] - m), max(y1, b[3] + m)
                # at most 1.6x the declared span (beyond that the program is broken, not the view)
                sx = min(1.6, (nx1 - nx0) / (x1 - x0))
                sy = min(1.6, (ny1 - ny0) / (y1 - y0))
                if (nx1 - nx0) > (x1 - x0) * sx:
                    c = (nx0 + nx1) / 2
                    nx0, nx1 = c - (x1 - x0) * sx / 2, c + (x1 - x0) * sx / 2
                if (ny1 - ny0) > (y1 - y0) * sy:
                    c = (ny0 + ny1) / 2
                    ny0, ny1 = c - (y1 - y0) * sy / 2, c + (y1 - y0) * sy / 2
                to = (nx0, nx1, ny0, ny1)
            if to != st["to"]:
                st["from"] = self._fit_now(st, now)
                st["to"] = to
                st["at"] = now
        return self._fit_now(st, now)

    @staticmethod
    def _fit_now(st, now):
        u = min(1.0, max(0.0, (now - st["at"]) / 0.6))
        u = u * u * (3 - 2 * u)
        return tuple(a + (b - a) * u for a, b in zip(st["from"], st["to"]))

    def render(self, doc, now, bg=None, texts=True):
        """texts=False: geometry only (the agent's change check compares drawings, not captions)"""
        k, ox, oy = self._map(self.fit_view(doc, now) if self.autofit else doc.view)
        ppu = k
        fr = Frame(doc, now, ppu, self.w)
        marks = fr.marks()
        self.last_marks = len(marks)
        bgc = BG
        if doc.bg is not None:
            from .expr import Env
            b = doc.bg.eval(Env(fr.base, fr))
            if isinstance(b, tuple):
                bgc = b
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = np.array([int(c * 255) for c in bgc[::-1]], np.uint8)   # BGR internally (cv2)
        glow = np.zeros((self.h, self.w, 3), np.uint8)
        scale = self.w / 512.0
        self._placed = []
        self._leaders = []
        self._lum = None
        self._floor = max(7, int(round(TEXT_FLOOR_PX * self.w / 512.0)))
        # geometry first, then every text on top (labels stay legible; the free-spot search sees all the lines).
        # A free text that repeats an object's own label ("object" twice) is dropped: the attached label stays.
        want_texts = texts
        texts = [m for m in marks if m.kind == "text" and m.soft != "inline"]
        auto = {_norm_text(m.text) for m in texts if m.soft is True and m.text}
        if auto:
            texts = [m for m in texts if m.soft or _norm_text(m.text) not in auto]
        if want_texts is False:
            texts = []
        for m in [m for m in marks if m.kind != "text" or m.soft == "inline"]:     # inline: part of a shape (an atom's symbol)
            try:
                self._draw(img, glow, m, k, ox, oy, scale)
            except (cv2.error, ValueError, OverflowError):
                continue
        # texts are LAID OUT first (all of them, in priority order: the program's own free texts,
        # callouts as one set, automatic labels, axis numbers last), then leaders are drawn, then every text is blitted.
        # Each later text avoids the earlier ones, the leaders, and the drawn lines / shape edges.
        prio = {False: 0, None: 0, "callout": 1, True: 2, "tick": 3, "axis": 4}
        texts = [m for m in texts if m.alpha > 0.004 and m.text]
        texts.sort(key=lambda m: prio.get(m.soft, 0))          # stable: draw order kept inside a class
        prev = getattr(self, "_prev_off", {})
        self._cur_off = {}
        laid = []
        calls = [m for m in texts if m.soft == "callout"]
        for m in texts:
            if m.soft == "callout":
                if calls:
                    try:
                        laid += self._place_callouts(img, calls, k, ox, oy)
                    except (cv2.error, ValueError, OverflowError):
                        pass
                    calls = []
                continue
            try:
                it = self._layout(m, k, ox, oy)
                if it is not None and self._place(img, m, it, prev):
                    laid.append(it)
            except (cv2.error, ValueError, OverflowError):
                continue
        self._prev_off = self._cur_off
        # where every text went (pixels) and the leaders: for tests / the agent's picture checks
        self.last_texts = [(it["text"], it["x"], it["y"], it["w"], it["h"], it["m"].soft) for it in laid]
        self.last_leaders = [(a, b) for (a, b, _c, _al) in self._leaders]
        for (a, b, col, al) in self._leaders:
            self._leader(img, a, b, col, al)
        for it in laid:
            self._blit(img, glow, it)
        return self._finish(img, glow)

    def idle(self, now):
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = np.array([int(c * 255) for c in BG[::-1]], np.uint8)
        glow = np.zeros_like(img)
        # a slow breathing dot + faint ring: the TV is on and waiting
        b = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 4.0)
        c = (self.w // 2 * SUB, self.h // 2 * SUB)
        col = tuple(int(v * (0.35 + 0.3 * b)) for v in (255, 200, 120))
        cv2.circle(img, c, int(5 * SUB * self.w / 512), col, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, c, int((18 + 4 * b) * SUB * self.w / 512), tuple(int(v * 0.25) for v in col), 1, cv2.LINE_AA, SHIFT)
        cv2.circle(glow, c, int(6 * SUB * self.w / 512), col, -1, cv2.LINE_AA, SHIFT)
        return self._finish(img, glow)

    def _finish(self, img, glow):
        if self.glow_k > 0:
            sm = cv2.resize(glow, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            sm = cv2.GaussianBlur(sm, (0, 0), 2.2)
            big = cv2.resize(sm, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            out = cv2.addWeighted(img, 1.0, big, self.glow_k, 0.0)
        else:
            out = img
        # vignette + BGR -> RGB with OpenCV (the numpy float multiply + reversed copy cost ~2.5 ms a frame)
        out = cv2.multiply(out, self._vig3, dtype=cv2.CV_8U)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # ---- one mark
    def _draw(self, img, glow, m, k, ox, oy, scale):
        a = m.alpha
        if a <= 0.004:
            return
        if m.kind == "text":
            self._text(img, glow, m, k, ox, oy)
            return
        if m.segs:
            self._draw_segs(img, glow, m, k, ox, oy, scale)
            return
        pts = m.pts
        if not pts or len(pts) < 1:
            return
        if m.draw < 0.999:
            if m.draw <= 0.002:
                return
            pts = _trim(pts + ([pts[0]] if m.closed and m.stroke is not None else []), m.draw)
            closed = False
        else:
            closed = m.closed
        P = np.array(pts, np.float64)
        if P.ndim != 2 or P.shape[1] != 2:
            return
        px = np.empty_like(P)
        with np.errstate(over="ignore", invalid="ignore"):
            px[:, 0] = P[:, 0] * k + ox
            px[:, 1] = oy - P[:, 1] * k
        lo = px.min(axis=0)
        hi = px.max(axis=0)
        if not (math.isfinite(lo[0]) and math.isfinite(lo[1]) and math.isfinite(hi[0]) and math.isfinite(hi[1])):
            px = px[np.all(np.isfinite(px), axis=1)]
            if len(px) < 2:
                return
            lo = px.min(axis=0)
            hi = px.max(axis=0)
        # cull far-off geometry (keeps cv2 integer math sane)
        lim = 4 * max(self.w, self.h)
        if hi[0] < -lim or lo[0] > self.w + lim or hi[1] < -lim or lo[1] > self.h + lim:
            return
        if lo[0] < -lim or lo[1] < -lim or hi[0] > self.w + lim or hi[1] > self.h + lim:
            px = np.clip(px, -lim, self.w + lim)
            lo = px.min(axis=0)
            hi = px.max(axis=0)
        ip = np.round(px * SUB).astype(np.int32)
        width = max(1, int(round(m.width * scale)))
        x0 = int(max(0, math.floor(lo[0]) - width - 12))
        x1 = int(min(self.w, math.ceil(hi[0]) + width + 12))
        y0 = int(max(0, math.floor(lo[1]) - width - 12))
        y1 = int(min(self.h, math.ceil(hi[1]) + width + 12))
        if x1 <= x0 or y1 <= y0:
            return
        fill = m.fill if (m.fill is not None and len(ip) >= 3 and (closed or m.closed)) else None
        if m.fill is not None and m.draw < 0.999 and m.closed:
            # a filling shape that is still being traced: its fill fades in with the trace
            fill = m.fill
        if fill is not None:
            fa = a * (1.0 if m.draw >= 0.999 else m.draw ** 2)
            fpts = np.round(np.column_stack([np.array(m.pts)[:, 0] * k + ox, oy - np.array(m.pts)[:, 1] * k]).clip(-lim, self.w + lim) * SUB).astype(np.int32) \
                if m.draw < 0.999 else ip
            self._blend_op(img, (x0, y0, x1, y1), fa, lambda im, off: cv2.fillPoly(
                im, [fpts - off], _bgr(fill), cv2.LINE_AA, SHIFT))
        if m.stroke is not None and len(ip) >= 2:
            col = _bgr(m.stroke)
            segs = _dash(ip, m.dash * scale * SUB) if m.dash and m.dash > 0 else [ip]
            def op(im, off, segs=segs, col=col, closed=closed):
                for s in segs:
                    if len(s) >= 2:
                        cv2.polylines(im, [s - off], closed and len(segs) == 1, col, width, cv2.LINE_AA, SHIFT)
            if m.add:
                roi = img[y0:y1, x0:x1]
                tmp = np.zeros_like(roi)
                op(tmp, np.array([x0 * SUB, y0 * SUB], np.int32))
                cv2.addWeighted(roi, 1.0, tmp, a, 0.0, dst=roi)
            else:
                self._blend_op(img, (x0, y0, x1, y1), a, op)
            if m.glow > 0:
                ga = min(1.0, a * m.glow)
                gcol = tuple(int(c * ga) for c in col)
                for s in segs:
                    if len(s) >= 2:
                        cv2.polylines(glow, [s], closed and len(segs) == 1, gcol, width + 2, cv2.LINE_AA, SHIFT)
            if m.head:
                hl = (5.0 + 2.6 * m.width) * scale
                self._head(img, glow, px, hl, col, a, m.glow)
                if m.head == 2:
                    self._head(img, glow, px[::-1], hl, col, a, m.glow)
        elif fill is not None and m.glow > 0 and m.glow != 1.0:
            pass

    def _draw_segs(self, img, glow, m, k, ox, oy, scale):
        """many polylines of one mark, each with its own strength, added as light in ONE pass (ripple wavefronts:
        two sources overlap additively, so crossing crests are brighter)"""
        lim = 4 * max(self.w, self.h)
        polys, alls = [], []
        for pts, a in m.segs:
            if a <= 0.01 or len(pts) < 2:
                continue
            P = np.asarray(pts, np.float64)
            px = np.empty_like(P)
            px[:, 0] = P[:, 0] * k + ox
            px[:, 1] = oy - P[:, 1] * k
            if not np.all(np.isfinite(px)):
                continue
            px = np.clip(px, -lim, self.w + lim)
            polys.append((np.round(px * SUB).astype(np.int32), a))
            alls.append(px)
        if not polys:
            return
        allp = np.vstack(alls)
        width = max(1, int(round(m.width * scale)))
        x0 = int(max(0, math.floor(allp[:, 0].min()) - width - 2))
        x1 = int(min(self.w, math.ceil(allp[:, 0].max()) + width + 2))
        y0 = int(max(0, math.floor(allp[:, 1].min()) - width - 2))
        y1 = int(min(self.h, math.ceil(allp[:, 1].max()) + width + 2))
        if x1 <= x0 or y1 <= y0:
            return
        col = np.array(_bgr(m.stroke or (1, 1, 1)), np.float64)
        roi = img[y0:y1, x0:x1]
        tmp = np.zeros_like(roi)
        off = np.array([x0 * SUB, y0 * SUB], np.int32)
        for ip, a in polys:
            c = tuple(int(v) for v in np.clip(col * min(1.0, a * m.alpha), 0, 255))
            cv2.polylines(tmp, [ip - off], m.closed, c, width, cv2.LINE_AA, SHIFT)
        cv2.add(roi, tmp, dst=roi)
        if m.glow > 0:
            g = glow[y0:y1, x0:x1]
            cv2.addWeighted(g, 1.0, tmp, min(1.0, m.glow), 0.0, dst=g)

    def _head(self, img, glow, px, hl, col, a, gk):
        tip = px[-1]
        j = len(px) - 2
        while j >= 0 and math.hypot(*(tip - px[j])) < 1e-3:
            j -= 1
        if j < 0:
            return
        d = tip - px[j]
        n = math.hypot(d[0], d[1])
        d = d / n
        pv = np.array([-d[1], d[0]])
        base = tip - d * hl
        tri = np.array([tip + d * hl * 0.15, base + pv * hl * 0.45, base - pv * hl * 0.45])
        ip = np.round(tri * SUB).astype(np.int32)
        bx = (int(max(0, tri[:, 0].min() - 3)), int(max(0, tri[:, 1].min() - 3)),
              int(min(self.w, tri[:, 0].max() + 3)), int(min(self.h, tri[:, 1].max() + 3)))
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            return
        self._blend_op(img, bx, a, lambda im, off: cv2.fillPoly(im, [ip - off], col, cv2.LINE_AA, SHIFT))
        if gk > 0:
            cv2.fillPoly(glow, [ip], tuple(int(c * min(1, a * gk)) for c in col), cv2.LINE_AA, SHIFT)

    def _lum_now(self, img):
        if self._lum is None:
            # once per frame (texts are laid out after all the geometry): brightness, and two summed-area tables so
            # a candidate text box is scored in O(1) --
            #   ink   = thin bright marks (lines, curves, outlines): brighter than the local background
            #   solid = filled areas (what survives a 9 px opening)
            ch = cv2.split(img)
            lum = cv2.max(cv2.max(ch[0], ch[1]), ch[2])      # (numpy's max over the channel axis was ~5 ms)
            opened = cv2.morphologyEx(lum, cv2.MORPH_OPEN, _K9)
            ink = (cv2.subtract(lum, opened) > 28).view(np.uint8)
            bgl = int(np.percentile(lum[::4, ::4], 30)) + 28
            solid = (opened > bgl).view(np.uint8)
            self._lum = lum
            self._ink = cv2.integral(ink)
            self._solid = cv2.integral(solid)
        return self._lum

    def _cover(self, lum, msk, cx, cy, w, h):
        """how much drawn ink a text box at (cx, cy) would sit on: thin marks inside it, plus the part of a filled
        shape it straddles (a text wholly inside a filled box is fine; half on, half off is not)"""
        x0, y0 = max(0, int(cx)), max(0, int(cy))
        x1, y1 = min(self.w, int(cx + w)), min(self.h, int(cy + h))
        if x1 <= x0 or y1 <= y0:
            return 10 ** 6
        I, S = self._ink, self._solid
        ink = int(I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0])
        sol = int(S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0])
        area = (x1 - x0) * (y1 - y0)
        return 2 * ink + min(sol, area - sol)

    def _hits(self, cx, cy, w, h, placed):
        return any(cx < bx1 - 2 and cx + w > bx0 + 2 and cy < by1 - 3 and cy + h > by0 + 3 for bx0, by0, bx1, by1 in placed)

    def _crosses_leader(self, cx, cy, w, h):
        for (a, b, _c, _al) in self._leaders:
            if _seg_rect(a, b, cx + 1, cy + 2, cx + w - 1, cy + h - 2):
                return True
        return False

    def _free_spot(self, img, mask, x, y, w, h, placed, free=False, hint=None):
        """a label slides (a little) to where it covers the fewest drawn lines, no other text and no leader.
        free=True: a text the program placed itself -- it stays unless its spot is really covered, and moves less far.
        hint: last frame's offset for this text (tried first, favoured: labels do not flicker between spots)"""
        lum = self._lum_now(img)
        msk = mask > 0.3
        best, bx, by = None, x, y
        sy = max(5, int(h * 0.6))
        sx = max(8, int(h * 1.1))
        cands = [(0, 0)]
        if hint is not None and hint != (0, 0):
            cands.append(hint)
        for r in ((1, 2, 3, 4) if free else (1, 2, 3)):
            cands += [(0, -sy * r), (0, sy * r), (sx * r, 0), (-sx * r, 0), (sx * r, -sy * r), (-sx * r, -sy * r),
                      (sx * r, sy * r), (-sx * r, sy * r)]
        tol = max(3, int(0.012 * w * h)) if free else 3
        dk = 1.0 if free else 0.6
        for k, (dx, dy) in enumerate(cands):
            cx, cy = int(round(x + dx)), int(round(y + dy))
            if cx < 1 or cy < 1 or cx + w > self.w - 1 or cy + h > self.h - 1:
                continue
            cover = self._cover(lum, msk, cx, cy, w, h)
            hit = self._hits(cx, cy, w, h, placed)
            lead = bool(self._leaders) and self._crosses_leader(cx, cy, w, h)
            score = cover + (10 ** 6 if hit else 0) + (4000 if lead else 0) + int(math.hypot(dx, dy) * dk)
            if hint is not None and (dx, dy) == hint:
                score = int(score * 0.7) - 6
            if best is None or score < best:
                best, bx, by = score, cx, cy
            if k == 0 and cover < tol and not hit and not lead:
                break
        return bx, by

    # ---- texts: layout (size floor, shrink/wrap, anchor, on screen) -> placement -> blit
    def _layout(self, m, k, ox, oy, maxw=None):
        p = m.pts[0]
        px = p[0] * k + ox
        py = oy - p[1] * k
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        floor = getattr(self, "_floor", 7)
        size_px = max(floor, min(140, int(round(m.size * k))))
        text = m.text
        mask = self.texts.get(text, size_px, m.bold)
        h, w = mask.shape
        avail = min(self.w - 8, int(maxw)) if maxw else self.w - 8
        if w > avail:
            s2 = int(size_px * avail / w)
            if s2 >= max(floor, int(size_px * 0.8)) or " " not in text:
                size_px = max(floor, s2)            # a little too wide (or one word): shrink, never below the floor
            else:
                size_px = max(floor, int(size_px * 0.85))     # much too wide: wrap at spaces instead
                text = _wrap(text, size_px, m.bold, avail)
            mask = self.texts.get(text, size_px, m.bold)
            h, w = mask.shape
        an = m.anchor or "c"
        x = px - w / 2
        y = py - h / 2
        if "l" in an:
            x = px
        elif "r" in an:
            x = px - w
        if "t" in an:
            y = py
        elif "b" in an:
            y = py - h
        # fully on screen (an inline text belongs to its shape: it goes where the shape goes)
        if m.soft != "inline":
            x = min(max(x, 3), self.w - w - 3) if w < self.w - 6 else (self.w - w) / 2
            y = min(max(y, 2), self.h - h - 2) if h < self.h - 4 else (self.h - h) / 2
        return {"m": m, "mask": mask, "x": x, "y": y, "w": w, "h": h, "px": px, "py": py, "text": text}

    def _place(self, img, m, it, prev):
        placed = self._placed
        x, y, w, h = it["x"], it["y"], it["w"], it["h"]
        if m.soft == "tick":
            # an axis number stays exactly at its tick (a shuffled scale is wrong); if it would overlap a text, skip it
            if self._hits(x, y, w, h, [(a + 1, b + 1, c - 1, d - 1) for a, b, c, d in placed]):
                return False
        elif m.soft != "inline":
            key = (m.text, int(it["px"]) // 12, int(it["py"]) // 12)
            hint = prev.get(key)
            nx, ny = self._free_spot(img, it["mask"], x, y, w, h, placed, free=m.soft in (False, None, "axis"), hint=hint)
            self._cur_off[key] = (nx - int(round(x)), ny - int(round(y)))
            x, y = nx, ny
        x, y = int(round(x)), int(round(y))
        it["x"], it["y"] = x, y
        if m.alpha > 0.15:
            placed.append((x + 2, y + int(h * 0.12), x + w - 2, y + h - int(h * 0.2)))
        return True

    def _place_callouts(self, img, calls, k, ox, oy):
        """labelled-parts callouts. Each label goes in free space beside the picture (margins first),
        labels on one side are stacked in the order of their anchors (so leaders do not cross), and a thin leader
        joins each label to its point (drawn later, with a dot at the point)."""
        lum = self._lum_now(img)
        W, H = self.w, self.h
        ink = lum > 60
        cols = np.flatnonzero(ink.any(axis=0))
        rows = np.flatnonzero(ink.any(axis=1))
        if len(cols) and len(rows):
            cb = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
        else:
            cb = (W // 2, H // 2, W // 2, H // 2)
        ccx = (cb[0] + cb[2]) / 2
        gap = max(3, int(3 * W / 512))
        items = []
        for m in calls:
            it = self._layout(m, k, ox, oy, maxw=W * 0.36)
            if it is None:
                continue
            side = (getattr(m, "side", None) or "a")[:1]
            if side not in "lrtb":
                side = "l" if it["px"] < ccx else "r"
                it["auto"] = True
            it["side"] = side
            items.append(it)
        # balance auto labels between left and right when one side would overflow
        for _ in range(len(items)):
            hl = sum(i["h"] + gap for i in items if i["side"] == "l")
            hr = sum(i["h"] + gap for i in items if i["side"] == "r")
            if hl > H - 8 and hr + 40 < hl:
                mv = [i for i in items if i["side"] == "l" and i.get("auto")]
                if not mv:
                    break
                max(mv, key=lambda i: i["px"])["side"] = "r"
            elif hr > H - 8 and hl + 40 < hr:
                mv = [i for i in items if i["side"] == "r" and i.get("auto")]
                if not mv:
                    break
                min(mv, key=lambda i: i["px"])["side"] = "l"
            else:
                break
        out = []
        pad = max(6, int(8 * W / 512))
        for side in "lrtb":
            grp = [i for i in items if i["side"] == side]
            if not grp:
                continue
            if side in "lr":
                grp.sort(key=lambda i: i["py"])
                # a label wider than the margin beside the picture wraps to fit it (when the margin is usable)
                room = (min(cb[0], min(i["px"] for i in grp)) - pad - 3) if side == "l" else \
                    (W - 3 - (max(cb[2], max(i["px"] for i in grp)) + pad))
                if room >= 56 * W / 512:
                    for n_, i in enumerate(grp):
                        if i["w"] > room:
                            j = self._layout(i["m"], k, ox, oy, maxw=room)
                            if j is not None and j["w"] <= room + 1:
                                j["side"] = side
                                grp[n_] = j
                                items[items.index(i)] = j
                # the column: in the margin beside the picture when it fits there, else just beside the points
                maxw = max(i["w"] for i in grp)
                if side == "l":
                    edge = min(cb[0], min(i["px"] for i in grp)) - pad
                    if edge - maxw < 3:
                        edge = None
                else:
                    edge = max(cb[2], max(i["px"] for i in grp)) + pad
                    if edge + maxw > W - 3:
                        edge = None
                # stack (top -> bottom in anchor order), then pull back up from the bottom edge
                y = 2
                for i in grp:
                    i["y"] = max(y, i["py"] - i["h"] / 2)
                    y = i["y"] + i["h"] + gap
                y = H - 2
                for i in reversed(grp):
                    i["y"] = min(i["y"], y - i["h"])
                    y = i["y"] - gap
                for i in grp:
                    i["y"] = max(2, i["y"])
                    if edge is not None:
                        i["x"] = edge - i["w"] if side == "l" else edge
                    else:
                        i["x"] = self._scan_x(lum, i, side, gap)
            else:
                grp.sort(key=lambda i: i["px"])
                maxh = max(i["h"] for i in grp)
                if side == "t":
                    edge = min(cb[1], min(i["py"] for i in grp)) - pad
                    edge = edge if edge - maxh >= 2 else None
                else:
                    edge = max(cb[3], max(i["py"] for i in grp)) + pad
                    edge = edge if edge + maxh <= H - 2 else None
                x = 3
                for i in grp:
                    i["x"] = max(x, i["px"] - i["w"] / 2)
                    x = i["x"] + i["w"] + 2 * gap
                x = W - 3
                for i in reversed(grp):
                    i["x"] = min(i["x"], x - i["w"])
                    x = i["x"] - 2 * gap
                for i in grp:
                    i["x"] = max(3, i["x"])
                    if edge is not None:
                        i["y"] = edge - i["h"] if side == "t" else edge
                    else:
                        i["y"] = self._scan_y(lum, i, side, gap)
            for i in grp:
                x, y, w, h = int(round(i["x"])), int(round(i["y"])), i["w"], i["h"]
                if self._hits(x, y, w, h, self._placed):        # a title / earlier text there: step along the column
                    st = max(4, h // 2)
                    for d in (st, -st, 2 * st, -2 * st, 3 * st, -3 * st, 4 * st, -4 * st):
                        nx, ny = (x, y + d) if side in "lr" else (x + 2 * d, y)
                        if 1 <= nx <= W - w - 1 and 1 <= ny <= H - h - 1 and not self._hits(nx, ny, w, h, self._placed):
                            x, y = nx, ny
                            break
                i["x"], i["y"] = x, y
                m = i["m"]
                if m.alpha > 0.15:
                    self._placed.append((x + 2, y + int(h * 0.12), x + w - 2, y + h - int(h * 0.2)))
                if side == "l":
                    b = (x + w - 1, y + h * 0.5)
                elif side == "r":
                    b = (x + 1, y + h * 0.5)
                elif side == "t":
                    b = (x + w * 0.5, y + h - 2)
                else:
                    b = (x + w * 0.5, y + 2)
                col = m.stroke or (1, 1, 1)
                self._leaders.append(((i["px"], i["py"]), b, col, m.alpha))
                out.append(i)
        return out

    def _scan_x(self, lum, i, side, gap):
        """no margin: the label goes beside its point, as close as the drawn ink allows"""
        w, h, y = i["w"], i["h"], int(round(i["y"]))
        best, bx = None, None
        step = max(4, w // 8)
        for n in range(0, 40):
            x = (i["px"] - 12 - w - n * step) if side == "l" else (i["px"] + 12 + n * step)
            if x < 2 or x + w > self.w - 2:
                break
            c = self._cover(lum, None, int(x), y, w, h) + (10 ** 6 if self._hits(int(x), y, w, h, self._placed) else 0)
            s = c + n * 3
            if best is None or s < best:
                best, bx = s, x
            if c < 4:
                break
        if bx is None:
            bx = 3 if side == "l" else self.w - w - 3
        return bx

    def _scan_y(self, lum, i, side, gap):
        w, h, x = i["w"], i["h"], int(round(i["x"]))
        best, by = None, None
        step = max(3, h // 3)
        for n in range(0, 40):
            y = (i["py"] - 10 - h - n * step) if side == "t" else (i["py"] + 10 + n * step)
            if y < 2 or y + h > self.h - 2:
                break
            c = self._cover(lum, None, x, int(y), w, h) + (10 ** 6 if self._hits(x, int(y), w, h, self._placed) else 0)
            s = c + n * 3
            if best is None or s < best:
                best, by = s, y
            if c < 4:
                break
        if by is None:
            by = 2 if side == "t" else self.h - h - 2
        return by

    def _leader(self, img, a, b, col, al):
        if al <= 0.01 or not all(math.isfinite(v) for v in (*a, *b)):
            return
        s = self.w / 512.0
        c = _bgr(tuple(0.8 * v for v in col))
        A = (int(round(a[0] * SUB)), int(round(a[1] * SUB)))
        B = (int(round(b[0] * SUB)), int(round(b[1] * SUB)))
        r = max(2, int(round(2.6 * s)))
        x0 = int(max(0, min(a[0], b[0]) - r - 3))
        y0 = int(max(0, min(a[1], b[1]) - r - 3))
        x1 = int(min(self.w, max(a[0], b[0]) + r + 3))
        y1 = int(min(self.h, max(a[1], b[1]) + r + 3))
        if x1 <= x0 or y1 <= y0:
            return

        def op(im, off):
            o = (int(off[0]), int(off[1]))
            cv2.line(im, (A[0] - o[0], A[1] - o[1]), (B[0] - o[0], B[1] - o[1]), c, max(1, int(round(1.2 * s))), cv2.LINE_AA, SHIFT)
            cv2.circle(im, (A[0] - o[0], A[1] - o[1]), r * SUB, _bgr(col), -1, cv2.LINE_AA, SHIFT)
        self._blend_op(img, (x0, y0, x1, y1), min(1.0, al * 0.9), op)

    def _blend_op(self, img, box, a, op):
        x0, y0, x1, y1 = box
        if a >= 0.995:
            op(img, np.array([0, 0], np.int32))
            return
        roi = img[y0:y1, x0:x1]
        tmp = roi.copy()
        sub = np.ascontiguousarray(tmp)
        op(sub, np.array([x0 * SUB, y0 * SUB], np.int32))
        cv2.addWeighted(sub, a, roi, 1.0 - a, 0.0, dst=roi)

    def _text(self, img, glow, m, k, ox, oy):
        """an inline text (an atom's symbol, a flow node's name): exactly on its shape, drawn with the geometry"""
        if not m.text:
            return
        if not hasattr(self, "_placed"):
            self._placed, self._leaders, self._lum, self._cur_off = [], [], None, {}
        it = self._layout(m, k, ox, oy)
        if it is None:
            return
        it["x"], it["y"] = int(round(it["x"])), int(round(it["y"]))
        if m.alpha > 0.15:
            x, y, w, h = it["x"], it["y"], it["w"], it["h"]
            self._placed.append((x + 2, y + int(h * 0.12), x + w - 2, y + h - int(h * 0.2)))
        self._blit(img, glow, it)

    def _blit(self, img, glow, it):
        m, mask, x, y = it["m"], it["mask"], it["x"], it["y"]
        h, w = mask.shape
        xa, ya = max(0, x), max(0, y)
        xb, yb = min(self.w, x + w), min(self.h, y + h)
        if xb <= xa or yb <= ya:
            return
        mm = mask[ya - y:yb - y, xa - x:xb - x][..., None] * m.alpha
        col = np.array(_bgr(m.stroke or (1, 1, 1)), np.float32)
        roi = img[ya:yb, xa:xb].astype(np.float32)
        # a soft dark halo keeps text legible over lines
        halo = cv2.dilate(mm, np.ones((3, 3), np.float32))
        if halo.ndim == 2:
            halo = halo[..., None]
        roi = roi * (1 - 0.55 * halo) + 0.0
        roi = roi * (1 - mm) + col * mm
        img[ya:yb, xa:xb] = np.clip(roi, 0, 255).astype(np.uint8)
        g = glow[ya:yb, xa:xb].astype(np.float32)
        glow[ya:yb, xa:xb] = np.clip(np.maximum(g, col * mm * 0.5), 0, 255).astype(np.uint8)


def _plain(s):
    return re.sub(r"[_^]\[([^\]]*)\]", r"\1", s).replace("_", "").replace("^", "")


_WRAPS = {}


def _wrap(text, px, bold, maxw):
    """greedy word wrap of each line so it fits maxw pixels at px (sub/superscript marks measured as plain text)"""
    key = (text, px, bold, int(maxw))
    v = _WRAPS.get(key)
    if v is None:
        if len(_WRAPS) > 500:
            _WRAPS.clear()
        v = _WRAPS[key] = _wrap1(text, px, bold, maxw)
    return v


def _wrap1(text, px, bold, maxw):
    f = _font(px, bold)
    out = []
    for ln in text.split("\n"):
        words = ln.split(" ")
        cur = ""
        for wd in words:
            cand = (cur + " " + wd) if cur else wd
            if cur and f.getlength(_plain(cand)) > maxw - 6:
                out.append(cur)
                cur = wd
            else:
                cur = cand
        out.append(cur)
    return "\n".join(out)


def _seg_rect(a, b, x0, y0, x1, y1):
    """does segment a-b cross the rectangle (pixel coords)?"""
    if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1 or max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
        return False
    if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
        return True
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def _bgr(c):
    return (int(max(0, min(1, c[2])) * 255), int(max(0, min(1, c[1])) * 255), int(max(0, min(1, c[0])) * 255))


def _trim(pts, frac):
    total = path_length(pts)
    if total <= 0:
        return pts
    target = total * frac
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= target:
            u = (target - acc) / L if L else 0
            out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
            return out
        out.append(b)
        acc += L
    return out


def _dash(ip, dl):
    """split an int (subpixel) polyline into dashes of length dl (subpixel units), gaps 0.7*dl"""
    if dl < SUB * 2:
        return [ip]
    gap = dl * 0.7
    out, cur = [], [ip[0]]
    on = True
    left = dl
    for a, b in zip(ip[:-1].astype(np.float64), ip[1:].astype(np.float64)):
        seg = b - a
        L = math.hypot(seg[0], seg[1])
        pos = 0.0
        while L - pos > left:
            pos += left
            q = a + seg * (pos / L)
            if on:
                cur.append(q)
                out.append(np.round(np.array(cur)).astype(np.int32))
                cur = []
            else:
                cur = [q]
            on = not on
            left = dl if on else gap
        left -= (L - pos)
        if on:
            cur.append(b)
    if on and len(cur) >= 2:
        out.append(np.round(np.array(cur)).astype(np.int32))
    return out or [ip]
