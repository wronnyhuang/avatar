"""deterministic geometric collision checker for a lesson's board items (seam-S3 dicts).

Works in an ISOTROPIC frame: X = u * aspect, Y = v (board heights), so distances mean the same in both directions.
Items are replayed in order (erase items remove their targets and the marks attached to them) and every new item is
checked against itself and against what is on the board at that moment:

  (a) label_ink   a label's glyph-run bbox (item["labels"], see below) within HALO of a stroke segment of any other
                  NON-label mark: the item's own lines/boxes/edges, or another item's ink (labels touching labels are
                  (b)). A label's own leader line is not counted against it.
  (b) label_label two label bboxes (same item or a label of another item) closer than HALO.
  (c) item_touch  two content items whose ink comes closer than CHALK (0.011, the chalk line width: they touch);
      item_close  ... closer than CLOSE (cramped), but not touching;
      annot_touch an annotation / divider touching an item it does not mark.
  (d) out_band    ink outside the usable band (v < reach0, v > reach1, u < margin, u > 1 - margin);
      edge_band   ink within EDGE of those limits (touching the band edge).
  (e) order_back  reading order: a written (text/math/column) item of the checked lesson that sits wholly LEFT of an
                  earlier written item of the same lesson that is still on the board, and starts above that item's bottom (columns read
                  top-to-bottom, then left-to-right; going back up-left reads as scrambled).

Labels: figure / diagram items carry item["labels"] = [{"text", "strokes": [i0, i1], "leader": i | None,
"anchor": [u, v] | None, "bbox": [u0, v0, u1, v1]}] — strokes[i0:i1] are the label's glyphs (plus the leader stroke
at index `leader` if any). Items without the key are treated as plain ink (text/math: the whole item is one run).
"""
from __future__ import annotations

import math

CHALK = 0.011          # chalk line width (v)
HALO = 0.006           # label clearance: a stroke centreline nearer than this to the glyph-run bbox reads as touching
CLOSE = 0.025          # two content items closer than this (ink to ink) look cramped
EDGE = 0.008           # ink this close to the band limit touches the edge
REACH = (0.321, 0.95)
MARGIN_U = 0.015
CONTENT = ("text", "math", "column", "diagram", "figure")
FLOWK = ("text", "math", "column")
MARKS = ("underline", "box", "circle", "strike", "check", "highlight", "bracket", "arrow", "divider")


# ------------------------------------------------------------------ primitives (isotropic frame)

def seg_rect_dist(a, b, r):
    """Distance between segment ab and axis-aligned rect r=(x0,y0,x1,y1) (0 when they intersect)."""
    if _seg_hits_rect(a, b, r):
        return 0.0
    d = min(_pt_rect_dist(a, r), _pt_rect_dist(b, r))
    x0, y0, x1, y1 = r
    for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        d = min(d, pt_seg_dist(c, a, b))
    return d


def _pt_rect_dist(p, r):
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def _seg_hits_rect(a, b, r):
    """Liang-Barsky clip test."""
    x0, y0, x1, y1 = r
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-15:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def pt_seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def seg_seg_dist(a, b, c, d):
    if _segs_cross(a, b, c, d):
        return 0.0
    return min(pt_seg_dist(a, c, d), pt_seg_dist(b, c, d), pt_seg_dist(c, a, b), pt_seg_dist(d, a, b))


def _segs_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def rect_gap(a, b):
    """Gap between two rects (0 if they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def bbox_pts(strokes):
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


class SegIndex:
    """Uniform grid over stroke segments; each segment tagged with an owner key."""

    def __init__(self, cell=0.03):
        self.cell = cell
        self.grid = {}

    def add_stroke(self, pts, tag):
        c = self.cell
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            for gx in range(int(math.floor(min(a[0], b[0]) / c)), int(math.floor(max(a[0], b[0]) / c)) + 1):
                for gy in range(int(math.floor(min(a[1], b[1]) / c)), int(math.floor(max(a[1], b[1]) / c)) + 1):
                    self.grid.setdefault((gx, gy), []).append((a, b, tag))
        if len(pts) == 1:
            self.grid.setdefault((int(math.floor(pts[0][0] / c)), int(math.floor(pts[0][1] / c))), []).append(
                (pts[0], pts[0], tag))

    def near(self, r, pad):
        c = self.cell
        seen = set()
        for gx in range(int(math.floor((r[0] - pad) / c)), int(math.floor((r[2] + pad) / c)) + 1):
            for gy in range(int(math.floor((r[1] - pad) / c)), int(math.floor((r[3] + pad) / c)) + 1):
                for s in self.grid.get((gx, gy), ()):
                    k = id(s)
                    if k in seen:
                        continue
                    seen.add(k)
                    yield s

    def count_near(self, r, pad, cap=6):
        """Fast count (up to cap) of segments within ~pad of rect r (rect inflated by pad, segment clip test).
        A segment spanning two cells may count twice; callers only compare counts."""
        c = self.cell
        x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
        R = (x0, y0, x1, y1)
        n = 0
        for gx in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for gy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                for a, b, _ in self.grid.get((gx, gy), ()):
                    if (a[0] < x0 and b[0] < x0) or (a[0] > x1 and b[0] > x1) or \
                            (a[1] < y0 and b[1] < y0) or (a[1] > y1 and b[1] > y1):
                        continue
                    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or (x0 <= b[0] <= x1 and y0 <= b[1] <= y1) or \
                            _seg_hits_rect(a, b, R):
                        n += 1
                        if n >= cap:
                            return n
        return n

    def rect_hits(self, r, pad, skip=lambda tag: False):
        """Tags of segments within `pad` of rect r."""
        out = set()
        for a, b, tag in self.near(r, pad):
            if tag in out or skip(tag):
                continue
            if seg_rect_dist(a, b, r) < pad:
                out.add(tag)
        return out


# ------------------------------------------------------------------ item geometry

def _iso(strokes, A):
    return [[(p[0] * A, p[1]) for p in s] for s in strokes]


def item_parts(it, A=2.0):
    """-> (ink strokes iso, [label dicts {text, rect, idx(set of stroke idx incl. leader), glyph_idx}])."""
    st = _iso(it.get("strokes") or [], A)
    labs = []
    for k, L in enumerate(it.get("labels") or []):
        i0, i1 = L.get("strokes", [0, 0])
        gl = [i for i in range(i0, i1) if i != L.get("leader")]
        rect = bbox_pts([st[i] for i in gl if i < len(st)])
        if rect is None:
            continue
        idx = set(range(i0, i1))
        if L.get("leader") is not None:
            idx.add(L["leader"])
        labs.append({"k": k, "text": L.get("text", ""), "rect": rect, "idx": idx})
    return st, labs


def _targets(it):
    t = it.get("target")
    if t is None:
        return []
    if isinstance(t, str):
        t = t.split(",")
    return [str(x).split(":")[0].strip() for x in t]


# ------------------------------------------------------------------ the checker

def check_items(items, aspect=2.0, reach=REACH, margin_u=MARGIN_U, start=0):
    """Replay items in order; collisions found while adding items[start:] (earlier ones = setup, only tracked).
    Returns {"counts": {...}, "events": [...]}"""
    A = aspect
    present = {}          # id -> rec
    counts = {k: 0 for k in ("label_ink", "label_label", "item_touch", "item_close", "annot_touch", "out_band",
                             "edge_band", "order_back")}
    events = []

    def ev(kind, it, other=None, text=None, d=None):
        counts[kind] += 1
        e = {"kind": kind, "item": it.get("id"), "ikind": it.get("kind")}
        if other is not None:
            e["other"] = other
        if text:
            e["label"] = text
        if d is not None:
            e["d"] = round(d, 4)
        events.append(e)

    for n, it in enumerate(items):
        kind = it.get("kind")
        if kind == "erase":
            gone = set(_targets(it))
            for rid, r in list(present.items()):
                if rid in gone or (set(r["targets"]) & gone):
                    present.pop(rid, None)
            continue
        if not it.get("strokes"):
            continue
        st, labs = item_parts(it, A)
        rec = {"id": it.get("id"), "kind": kind, "strokes": st, "labels": labs, "targets": _targets(it),
               "bbox": bbox_pts(st)}
        check = n >= start
        rec["mine"] = check
        if check:
            _check_new(rec, it, present, ev, A, reach, margin_u)
            if kind in FLOWK and rec["bbox"]:
                for r in present.values():
                    if r.get("mine") and r["kind"] in FLOWK and r["bbox"] and \
                            rec["bbox"][2] < r["bbox"][0] - 1e-6 and rec["bbox"][1] < r["bbox"][3] - 1e-6:
                        ev("order_back", it, other=r["id"])
                        break
        present[rec["id"]] = rec
    return {"counts": counts, "total_labels": sum(counts[k] for k in ("label_ink", "label_label")),
            "total": sum(counts.values()), "events": events}


def _check_new(rec, it, present, ev, A, reach, margin_u):
    st, labs = rec["strokes"], rec["labels"]
    # (d) band
    if rec["bbox"]:
        x0, y0, x1, y1 = rec["bbox"]
        lo_x, hi_x = margin_u * A, (1 - margin_u) * A
        if y0 < reach[0] - 1e-6 or y1 > reach[1] + 1e-6 or x0 < lo_x - 1e-6 or x1 > hi_x + 1e-6:
            ev("out_band", it, d=max(reach[0] - y0, y1 - reach[1], lo_x - x0, x1 - hi_x))
        elif y0 < reach[0] + EDGE or y1 > reach[1] - EDGE or x0 < lo_x + EDGE or x1 > hi_x - EDGE:
            ev("edge_band", it, d=min(y0 - reach[0], reach[1] - y1, x0 - lo_x, hi_x - x1))
    # own index (for label vs own ink)
    if labs:
        # label vs NON-label ink (a label on another label is the label_label category)
        lab_idx = set().union(*[L["idx"] for L in labs])
        own = SegIndex()
        for i, s in enumerate(st):
            if i not in lab_idx:
                own.add_stroke(s, i)
        others = SegIndex()
        for r in present.values():
            ridx = set().union(*[L["idx"] for L in r["labels"]]) if r["labels"] else ()
            for i, s in enumerate(r["strokes"]):
                if i not in ridx:
                    others.add_stroke(s, r["id"])
        for L in labs:
            hit = own.rect_hits(L["rect"], HALO, skip=lambda i, L=L: i in L["idx"])
            ho = others.rect_hits(L["rect"], HALO)
            if hit or ho:
                ev("label_ink", it, other=sorted(ho) if ho else "own", text=L["text"])
        # (b) labels vs labels (own, then present items' labels)
        for i, L in enumerate(labs):
            for M in labs[i + 1:]:
                if rect_gap(L["rect"], M["rect"]) < HALO:
                    ev("label_label", it, other="own", text=L["text"] + " / " + M["text"])
            for r in present.values():
                for M in r["labels"]:
                    if rect_gap(L["rect"], M["rect"]) < HALO:
                        ev("label_label", it, other=r["id"], text=L["text"] + " / " + M["text"])
    # (c) item vs item
    if not rec["bbox"]:
        return
    mine_mark = rec["kind"] in MARKS
    for r in present.values():
        if not r["bbox"]:
            continue
        # a mark may touch what it marks (a box round its item); a divider never touches anything
        if mine_mark and rec["kind"] != "divider" and r["id"] in rec["targets"]:
            continue
        if r["kind"] in MARKS and r["kind"] != "divider" and rec["id"] in r["targets"]:
            continue
        thr = CHALK if (mine_mark or r["kind"] in MARKS) else CLOSE
        if rect_gap(rec["bbox"], r["bbox"]) >= thr:
            continue
        d = min_ink_dist(st, r["strokes"], thr)
        if d < thr:
            if mine_mark or r["kind"] in MARKS:
                ev("annot_touch", it, other=r["id"], d=d)
            elif d < CHALK:
                ev("item_touch", it, other=r["id"], d=d)
            else:
                ev("item_close", it, other=r["id"], d=d)


def min_ink_dist(sa, sb, thr):
    """Min centreline distance between two stroke sets (capped at thr)."""
    idx = SegIndex(cell=max(thr, 0.02))
    for s in sb:
        idx.add_stroke(s, 0)
    best = thr
    for s in sa:
        for i in range(1, len(s)):
            a, b = s[i - 1], s[i]
            r = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            for c, d, _ in idx.near(r, best):
                dd = seg_seg_dist(a, b, c, d)
                if dd < best:
                    best = dd
                    if best == 0.0:
                        return 0.0
        if len(s) == 1:
            for c, d, _ in idx.near((s[0][0], s[0][1], s[0][0], s[0][1]), best):
                best = min(best, pt_seg_dist(s[0], c, d))
    return best
