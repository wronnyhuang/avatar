"""Two-person (dyad) extraction: find each person's PANEL in a split-screen / video-call / picture-in-picture /
in-room two-shot clip, crop each panel with a FIXED rectangle for the whole clip, and run MediaPipe face + pose per
panel on the SAME decoded frames (time-synchronized streams A and B).

Output npz = extract.py's contract (layout, streams, probe, fps, present, blend, mat, lmk, pose, cut, prosody, wh)
plus: crops (S,4) normalized [x0,y0,x1,y1], dyad_ok (N,) bool, masks (S,4) normalized region filled gray in that
stream (PiP inset inside the big panel; [0,0,0,0] = none).
layout in {dyad-lr, dyad-tb, dyad-pip, dyad-room}; streams = ["A","B"], A = left / top / big panel.
dyad_ok: per frame, True where the dyad layout holds (from 0.5 fps samples where BOTH panel faces were found; a
single-sample dropout is filled; at a layout switch the OK side extends to the strongest cut in the gap). MediaPipe
is NOT run on frames whose two neighbouring samples are both non-dyad (present=0, zeros there).
probe (json): clusters, co-presence, split line/kind, size ratio, layout cover, dup check, present inside dyad_ok.
Rejected clips get <out>_reject/<stem>.json with the reason (and no npz); accepted ones also get an early
<out>_accept/<stem>.json right after layout detection (dyad_fetch.py --harvest reads it).
Usage: python -I dyad_extract.py <models_dir> <out_dir> <debug_dir> <clip.mp4> [...]
"""
import sys, os, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import prosody, audio, frames, probe, face_opts, pose_opts, LMK, POSE, FPS  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

SAMPLE_FPS = 0.5


def mpimg(a):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a))


# ----------------------------------------------------------------------------------------------- panel detection
def sample_faces(path, models, w, h):
    """Per 0.5 fps sample: list of face boxes [x0,y0,x1,y1] in full-frame pixels; plus gray frames."""
    regions = {"full": (0, 0, w, h), "L": (0, 0, w // 2, h), "R": (w // 2, 0, w, h),
               "T": (0, 0, w, h // 2), "B": (0, h // 2, w, h),
               # quadrants: small corner faces (picture-in-picture insets, gallery tiles)
               "TL": (0, 0, w // 2, h // 2), "TR": (w // 2, 0, w, h // 2),
               "BL": (0, h // 2, w // 2, h), "BR": (w // 2, h // 2, w, h)}
    out, grays = [], []
    with vision.FaceLandmarker.create_from_options(face_opts(models, vision.RunningMode.IMAGE, 4)) as fl:
        for img in frames(path, w, h, SAMPLE_FPS):
            grays.append(img.mean(2).astype(np.float32))
            boxes = []
            for name, (x0, y0, x1, y1) in regions.items():
                r = fl.detect(mpimg(img[y0:y1, x0:x1]))
                for L in r.face_landmarks:
                    xs = np.array([p.x for p in L]); ys = np.array([p.y for p in L])
                    if name != "full" and (xs.min() < 0.01 or xs.max() > 0.99 or ys.min() < 0.01 or ys.max() > 0.99):
                        continue        # face cut by the half boundary
                    fw = max(xs.max() - xs.min(), 1e-6); fh = max(ys.max() - ys.min(), 1e-6)
                    b = np.array([x0 + xs.min() * (x1 - x0), y0 + ys.min() * (y1 - y0),
                                  x0 + xs.max() * (x1 - x0), y0 + ys.max() * (y1 - y0),
                                  (ys[14] - ys[13]) / fh,                      # mouth open
                                  (xs[1] - (xs.min() + xs.max()) / 2) / fw,    # yaw proxy
                                  (ys[1] - (ys.min() + ys.max()) / 2) / fh])   # pitch proxy
                    if (b[2] - b[0]) < 12 or (b[3] - b[1]) < 12:
                        continue
                    if any(iou(b, c) > 0.3 for c in boxes):
                        continue
                    boxes.append(b)
            out.append(boxes)
    return out, np.array(grays, np.float32)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-6)


def cluster(samples):
    """Greedy clustering of face centers over time -> list of clusters {members: [(k, box)]}."""
    cls = []
    for k, boxes in enumerate(samples):
        for b in boxes:
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); s = np.sqrt((b[2] - b[0]) * (b[3] - b[1]))
            best, bd = None, 1e9
            for j, cl in enumerate(cls):
                d = np.linalg.norm(c - cl["c"])
                if d < 0.6 * max(s, cl["s"]) and 0.5 < s / cl["s"] < 2.0 and d < bd:
                    best, bd = j, d
            if best is None:
                cls.append(dict(members=[(k, b)], c=c, s=s))
            else:
                cl = cls[best]; cl["members"].append((k, b))
                arr = np.array([m[1] for m in cl["members"]])
                cl["c"] = np.median(np.stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2], 1), 0)
                cl["s"] = float(np.median(np.sqrt((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1]))))
    # merge fragments of one moving person: near each other and (almost) never seen in the same sample
    def stats(cl):
        arr = np.array([m[1] for m in cl["members"]])
        cl["c"] = np.median(np.stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2], 1), 0)
        cl["s"] = float(np.median(np.sqrt((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1]))))
        cl["ks"] = set(m[0] for m in cl["members"])
    for cl in cls:
        stats(cl)
    merged = True
    while merged:
        merged = False
        order = sorted(range(len(cls)), key=lambda j: -len(cls[j]["ks"]))
        for ii, i in enumerate(order):
            for j in order[ii + 1:]:
                a, b = cls[i], cls[j]
                d = np.linalg.norm(a["c"] - b["c"]); smax = max(a["s"], b["s"])
                both = len(a["ks"] & b["ks"])
                if d < 1.5 * smax and 0.4 < a["s"] / b["s"] < 2.5 and both <= 0.1 * min(len(a["ks"]), len(b["ks"])):
                    a["members"] += b["members"]; stats(a); cls.pop(j); merged = True
                    break
            if merged:
                break
    for cl in cls:
        arr = np.array([m[1] for m in cl["members"]])
        cl["box"] = np.median(arr[:, :4], 0)
    return cls


def same_person(A, B):
    """Duplicate-feed guard: one person shown twice (main view + circle overlay, blurred padding copy) moves in
    lockstep. Correlate mouth-open / |yaw| / pitch over samples where both clusters are seen."""
    fa = {k: b[4:7] for k, b in A["members"]}; fb = {k: b[4:7] for k, b in B["members"]}
    ks = sorted(set(fa) & set(fb))
    if len(ks) < 8:
        return 0.0
    xa = np.array([fa[k] for k in ks]); xb = np.array([fb[k] for k in ks])
    xa[:, 1] = np.abs(xa[:, 1]); xb[:, 1] = np.abs(xb[:, 1])
    cs = []
    for j in range(3):
        if xa[:, j].std() > 1e-3 and xb[:, j].std() > 1e-3:
            cs.append(float(np.corrcoef(xa[:, j], xb[:, j])[0, 1]))
    return float(np.mean(cs)) if cs else 0.0


def presence(cls, K):
    P = np.zeros((K, len(cls)), bool)
    for j, cl in enumerate(cls):
        for k, _ in cl["members"]:
            P[k, j] = True
    return P


def live_mask(G, axis):
    """Rows (axis=1 -> per row) / cols (axis=0) that are not static letterbox: temporal std or spatial std > small."""
    tstd = G.std(0)                                     # (h, w)
    if axis == 1:
        return (tstd.mean(1) > 2.0) | (G.std(2).mean(0) > 3.0)
    return (tstd.mean(0) > 2.0) | (G.std(1).mean(0) > 3.0)


def tile_trim(G, box, cl, rows):
    """Inside a panel, cut away neighbouring grid tiles (e.g. a 2x2 call grid): look for a strong split line
    (static bar or edge across >= 80% of the panel) above/below (rows=True) or left/right of the face."""
    x0, y0, x1, y1 = [int(v) for v in box]
    fx0, fy0, fx1, fy1 = cl["box"]; s = cl["s"]
    sub = G[:, y0:y1, x0:x1]
    if rows:
        a_lo, a_hi = 2, int(fy0 - y0 - 0.4 * s)           # above the face
        b_lo, b_hi = int(fy1 - y0 + 0.8 * s), (y1 - y0) - 2  # below the face (leave room for shoulders)
    else:
        a_lo, a_hi = 2, int(fx0 - x0 - 0.5 * s)
        b_lo, b_hi = int(fx1 - x0 + 0.5 * s), (x1 - x0) - 2
    sb = find_split(sub, b_lo, b_hi, not rows, strict=True) if b_hi - b_lo > 2 else None
    sa = find_split(sub, a_lo, a_hi, not rows, strict=True) if a_hi - a_lo > 2 else None
    if rows:
        if sb: y1 = y0 + sb[0]
        if sa: y0 = y0 + sa[1]
    else:
        if sb: x1 = x0 + sb[0]
        if sa: x0 = x0 + sa[1]
    return (x0, y0, x1, y1)


def find_split(G, lo, hi, vertical, strict=False):
    """Split line between two panels in [lo, hi): vertical=True searches columns (lr), else rows (tb).
    Returns (a_end, b_start, score, kind) or None. Uses a static uniform bar if present, else the line with the
    most temporally-consistent full-length edge."""
    if hi - lo < 2:
        return None
    Gt = G if vertical else G.transpose(0, 2, 1)       # make the split a column search
    live = live_mask(Gt, 1)                              # live rows (along the split line)
    if live.sum() < 10:
        live[:] = True
    Gl = Gt[:, live, :]
    # static uniform bar columns
    sstd = np.median(Gl.std(1), 0)                       # (w,) spatial std along the column
    tstd = Gl.std(0).mean(0)                             # (w,) temporal std
    bar = (sstd < 4.0) & (tstd < 3.0)
    run, best = None, None
    for x in range(lo, hi):
        if bar[x]:
            run = [x, x] if run is None else [run[0], x]
        else:
            if run is not None and (best is None or run[1] - run[0] > best[1] - best[0]):
                best = run
            run = None
    if run is not None and (best is None or run[1] - run[0] > best[1] - best[0]):
        best = run
    if best is not None and best[1] - best[0] >= (2 if strict else 1):
        return best[0], best[1] + 1, 1.0, "bar"
    dx = np.abs(np.diff(Gl, axis=2)) > 12               # (K, rows, w-1)
    frac = np.median(dx.mean(1), 0)                     # (w-1,) per column: fraction of rows with an edge
    seg = frac[lo:min(hi, len(frac))]
    if len(seg) == 0:
        return None
    x = lo + int(np.argmax(seg)); sc = float(seg.max())
    if sc >= (0.8 if strict else 0.45):
        return x + 1, x + 1, sc, "edge"
    return None


def trim_dead(G, box):
    """Shrink box (x0,y0,x1,y1) px to exclude static letterbox rows/cols inside it."""
    x0, y0, x1, y1 = box
    sub = G[:, y0:y1, x0:x1]
    if sub.shape[1] < 20 or sub.shape[2] < 20:
        return box
    rows = np.where(live_mask(sub, 1))[0]; cols = np.where(live_mask(sub, 0))[0]
    if len(rows) > 10:
        y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
    if len(cols) > 10:
        x0, x1 = x0 + cols[0], x0 + cols[-1] + 1
    return (x0, y0, x1, y1)


def pip_rect(G, cl, w, h):
    """Inset tile around the small face: nearest consistent straight edges outward from the face, else the frame
    border if close, else a fixed 2.5x-face box."""
    cx, cy = cl["c"]; s = cl["s"]
    ex = (np.abs(np.diff(G, axis=2)) > 12).mean(0)       # (h, w-1) fraction of samples with a vertical edge
    ey = (np.abs(np.diff(G, axis=1)) > 12).mean(0)
    ylo, yhi = int(max(0, cy - s)), int(min(h, cy + s))
    xlo, xhi = int(max(0, cx - s)), int(min(w - 1, cx + s))

    def scan(prof, start, stop, step, border):
        for v in range(int(start), int(stop), step):
            if 0 <= v < len(prof) and prof[v] > 0.7:
                return v + 1, True
        if abs(border - start) < 3 * s:
            return border, True
        return None, False

    colp = ex[ylo:yhi].mean(0) if yhi > ylo else np.zeros(w - 1)
    left, _ = scan(colp, cx - 0.7 * s, max(-1, cx - 4 * s), -1, 0)
    right, _ = scan(colp, cx + 0.7 * s, min(w - 1, cx + 4 * s), 1, w)
    if right is not None and right != w:
        right = right - 1
    left = int(cx - 1.25 * s) if left is None else left
    right = int(cx + 1.25 * s) if right is None else right
    rowp = ey[:, max(0, left):min(w, right)].mean(1) if right > left else np.zeros(h - 1)
    top, _ = scan(rowp, cy - 0.8 * s, max(-1, cy - 4 * s), -1, 0)
    bot, _ = scan(rowp, cy + 0.8 * s, min(h - 1, cy + 4 * s), 1, h)
    if bot is not None and bot != h:
        bot = bot - 1
    top = int(cy - 1.25 * s) if top is None else top
    bot = int(cy + 1.25 * s) if bot is None else bot
    return (int(max(0, left)), int(max(0, top)), int(min(w, right)), int(min(h, bot)))


def room_box(cl, other, w, h):
    cx, cy = cl["c"]; s = cl["s"]
    x0, x1 = cx - 1.8 * s, cx + 1.8 * s
    y0, y1 = cy - 1.3 * s, cy + 3.0 * s
    ox, oy = other["c"]
    if abs(ox - cx) >= abs(oy - cy):
        mid = (ox + cx) / 2
        if ox > cx: x1 = min(x1, mid)
        else: x0 = max(x0, mid)
    else:
        mid = (oy + cy) / 2
        if oy > cy: y1 = min(y1, mid)
        else: y0 = max(y0, mid)
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))


def detect_dyad(path, models, w, h):
    samples, G = sample_faces(path, models, w, h)
    K = len(samples)
    info = dict(samples=K)
    if K < 5:
        return None, "too-short", info
    cls = cluster(samples)
    P = presence(cls, K)
    sup = P.mean(0)
    stable = [j for j in range(len(cls)) if sup[j] >= 0.25]
    info["clusters"] = [dict(c=[round(float(v), 1) for v in cls[j]["c"]], s=round(cls[j]["s"], 1),
                             sup=round(float(sup[j]), 2)) for j in stable]
    info["max_faces"] = int(max(len(b) for b in samples))
    if len(stable) < 2:
        return None, f"<2 stable faces (stable={len(stable)}, max_faces={info['max_faces']})", info
    best, bco = None, -1
    for a in stable:
        for b in stable:
            if a < b:
                d = np.linalg.norm(cls[a]["c"] - cls[b]["c"])
                if d < 1.0 * max(cls[a]["s"], cls[b]["s"]):
                    continue
                co = float((P[:, a] & P[:, b]).mean())
                if co > bco:
                    best, bco = (a, b), co
    if best is None:
        return None, "faces not distinct", info
    info["copresence"] = round(bco, 3)
    a, b = best
    for c in stable:
        if c not in best and float((P[:, a] & P[:, b] & P[:, c]).mean()) >= 0.5:
            return None, "3+ people", info
    if bco < 0.5:
        return None, f"co-presence {bco:.2f} < 0.5", info
    ca, cb = cls[a], cls[b]
    sp_corr = same_person(ca, cb)
    info["motion_corr"] = round(sp_corr, 3)
    if sp_corr > 0.8:
        return None, f"same person twice (motion corr {sp_corr:.2f})", info
    # orientation: lr if the faces are separated more horizontally than vertically
    ddx = abs(ca["c"][0] - cb["c"][0]); ddy = abs(ca["c"][1] - cb["c"][1])
    vertical = ddx >= ddy
    if vertical:
        first, second = (ca, cb) if ca["c"][0] < cb["c"][0] else (cb, ca)
        lo = int(first["box"][2] + 0.1 * first["s"]); hi = int(second["box"][0] - 0.1 * second["s"])
    else:
        first, second = (ca, cb) if ca["c"][1] < cb["c"][1] else (cb, ca)
        lo = int(first["box"][3] + 0.1 * first["s"]); hi = int(second["box"][1] - 0.1 * second["s"])
    sp = find_split(G, max(1, lo), min((w if vertical else h) - 1, hi), vertical)
    ratio = min(ca["s"], cb["s"]) / max(ca["s"], cb["s"])
    info["size_ratio"] = round(ratio, 2)
    mask = None
    if sp is not None:
        ae, bs, sc, kind = sp
        info["split"] = dict(at=[int(ae), int(bs)], score=round(sc, 2), kind=kind)
        if vertical:
            boxA, boxB, layout = (0, 0, ae, h), (bs, 0, w, h), "dyad-lr"
        else:
            boxA, boxB, layout = (0, 0, w, ae), (0, bs, w, h), "dyad-tb"
        A, B = first, second
        boxA = trim_dead(G, tile_trim(G, boxA, A, rows=vertical))
        boxB = trim_dead(G, tile_trim(G, boxB, B, rows=vertical))
    elif ratio < 0.6:
        big, small = (ca, cb) if ca["s"] >= cb["s"] else (cb, ca)
        boxB = pip_rect(G, small, w, h)
        boxA = trim_dead(G, (0, 0, w, h))
        bx0, by0, bx1, by1 = big["box"]
        if not (boxB[2] <= bx0 or boxB[0] >= bx1 or boxB[3] <= by0 or boxB[1] >= by1):
            return None, "pip inset overlaps the big face", info
        mask = boxB
        layout, A, B = "dyad-pip", big, small
    else:
        A, B = first, second
        boxA, boxB, layout = room_box(A, B, w, h), room_box(B, A, w, h), "dyad-room"
    # face-in-crop sanity
    for nm, cl, bx in (("A", A, boxA), ("B", B, boxB)):
        fx0, fy0, fx1, fy1 = cl["box"]
        if fx0 < bx[0] - 4 or fx1 > bx[2] + 4 or fy0 < bx[1] - 4 or fy1 > bx[3] + 4:
            return None, f"face {nm} not inside its crop", info
        if bx[2] - bx[0] < 32 or bx[3] - bx[1] < 32:
            return None, f"crop {nm} too small", info
    ia = a if A is cls[a] else b
    ib = b if B is cls[b] else a
    # a third distinct face inside either crop (even if it is not stable) means a 3+ person tile/grid
    for j, cl in enumerate(cls):
        if j in (ia, ib) or sup[j] < 0.08:
            continue
        for own, bx, nm in ((ia, boxA, "A"), (ib, boxB, "B")):
            cx, cy = cl["c"]
            if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] and \
                    np.linalg.norm(cl["c"] - cls[own]["c"]) > 1.0 * max(cl["s"], cls[own]["s"]) and \
                    int((P[:, j] & P[:, own]).sum()) >= 3:
                return None, f"3+ people (third face inside crop {nm})", info
    ok = P[:, ia] & P[:, ib]
    # fill single-sample dropouts (a look-down at the sampling instant is not a layout change)
    okf = ok.copy()
    for k in range(1, K - 1):
        if not ok[k] and ok[k - 1] and ok[k + 1]:
            okf[k] = True
    info["layout_cover"] = round(float(okf.mean()), 3)
    if okf.mean() < 0.5:
        return None, f"dyad layout covers {okf.mean():.2f} < 0.5", info
    return dict(layout=layout, boxA=boxA, boxB=boxB, mask=mask, ok_samples=okf, G0=G[0], A=A, B=B), "ok", info


def frame_ok(ok_s, cut, N):
    """Per-frame dyad_ok from 0.5 fps samples: True between two OK samples; on a switch, the OK side extends up to
    the strongest cut in between (if there is a clear cut), else False for the whole gap."""
    K = len(ok_s)
    t = np.arange(N) / FPS
    st = np.arange(K) / SAMPLE_FPS                       # ffmpeg fps filter: sample k ~ t = k / 0.5 s
    idx = np.clip(np.searchsorted(st, t, side="right") - 1, 0, K - 1)    # previous sample
    nxt = np.clip(idx + 1, 0, K - 1)
    out = ok_s[idx] & ok_s[nxt]
    out[t >= st[-1]] = ok_s[-1]
    for k in range(K - 1):
        if ok_s[k] != ok_s[k + 1]:
            f0, f1 = int(st[k] * FPS), min(N, int(st[k + 1] * FPS) + 1)
            if f1 - f0 < 3:
                continue
            seg = cut[f0:f1]
            j = int(np.argmax(seg))
            if seg[j] > 12.0:
                if ok_s[k]:
                    out[f0:max(f0, f0 + j - 3)] = True
                else:
                    out[min(f1, f0 + j + 3):f1] = True
    return out


def save_debug(dbg_dir, name, G0, boxes, mask, info):
    try:
        import cv2
        im = np.stack([G0] * 3, 2).astype(np.uint8).copy()
        cols = [(0, 255, 0), (0, 0, 255)]
        for (x0, y0, x1, y1), c, lab in zip(boxes, cols, "AB"):
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), c, 2)
            cv2.putText(im, lab, (int(x0) + 4, int(y0) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if mask is not None:
            cv2.rectangle(im, (int(mask[0]), int(mask[1])), (int(mask[2]) - 1, int(mask[3]) - 1), (255, 0, 255), 1)
        for cl in info.get("clusters", []):
            cv2.circle(im, (int(cl["c"][0]), int(cl["c"][1])), 3, (255, 255, 0), -1)
        cv2.imwrite(str(Path(dbg_dir) / f"{name}.jpg"), im)
    except Exception as e:
        print(f"[dbg-err] {name} {e!r}", flush=True)


def dup_check(present, blend, mat):
    """True if streams A and B look like the same feed: blink correlation (eyeBlinkLeft/Right, frame level) and
    head-yaw correlation (transformation matrix r02) BOTH elevated (> 0.3 and |.| > 0.5), or either one extreme
    (> 0.6 / |.| > 0.85). Independent people: both ~0 (Zoom recordings with shared frame drops reach blink ~0.35)."""
    both = (present[0] == 1) & (present[1] == 1)
    if both.sum() < 150:
        return False, "n/a"

    def c(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-4 and b.std() > 1e-4 else 0.0
    bl = np.mean([c(blend[0, both, k], blend[1, both, k]) for k in (9, 10)])
    yw = c(mat[0, both, 2], mat[1, both, 2])
    dup = (bl > 0.3 and abs(yw) > 0.5) or bl > 0.6 or abs(yw) > 0.85
    return bool(dup), f"blink_corr={bl:.2f} yaw_corr={yw:.2f}"


# ----------------------------------------------------------------------------------------------- main per clip
def run(models, out_dir, dbg_dir, path, hands=False):
    t0 = time.time()
    path = str(path); name = Path(path).stem
    out = Path(out_dir) / f"{name}.npz"
    rej = Path(str(out_dir) + "_reject") / f"{name}.json"
    if out.exists() or rej.exists():
        return
    W0, H0 = probe(path)
    w = min(640, W0 - W0 % 2); h = int(round(H0 * w / W0 / 2)) * 2
    det, why, info = detect_dyad(path, models, w, h)
    if det is None:
        rej.parent.mkdir(parents=True, exist_ok=True)
        rej.write_text(json.dumps(dict(reason=why, info=info, secs=round(time.time() - t0, 1))))
        print(f"[reject] {name} {why} {time.time()-t0:.0f}s", flush=True)
        return
    boxes = [det["boxA"], det["boxB"]]
    masks = [det["mask"], None]
    save_debug(dbg_dir, name, det["G0"], boxes, det["mask"], info)
    acc = Path(str(out_dir) + "_accept") / f"{name}.json"        # early accept marker (lets the fetcher harvest)
    acc.parent.mkdir(parents=True, exist_ok=True)
    acc.write_text(json.dumps(dict(layout=det["layout"], info=info)))
    fls = [vision.FaceLandmarker.create_from_options(face_opts(models, vision.RunningMode.VIDEO)) for _ in boxes]
    pls = [vision.PoseLandmarker.create_from_options(pose_opts(models)) for _ in boxes]
    S = 2
    imgs_cut = []; prev = None
    rec = {k: [[] for _ in range(S)] for k in ("present", "blend", "mat", "lmk", "pose")}
    Z_LMK = np.zeros((len(LMK), 3), np.float32); Z_POSE = np.zeros((len(POSE), 4), np.float32)
    ok_s = det["ok_samples"]
    # per-frame OK is needed before running MediaPipe: a cheap prepass over the cut signal needs all frames, so
    # we run MediaPipe on every frame whose nearest samples are not BOTH non-dyad (skip only clear non-dyad gaps)
    K = len(ok_s)
    for i, img in enumerate(frames(path, w, h, FPS)):
        ts = int(i * 1000 / FPS)
        g = img[::8, ::8].mean(2)
        imgs_cut.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
        k = min(int(i / FPS * SAMPLE_FPS), K - 1); k2 = min(k + 1, K - 1)
        skip = (not ok_s[k]) and (not ok_s[k2])
        for s in range(S):
            x0, y0, x1, y1 = boxes[s]
            if skip:
                rec["present"][s].append(0); rec["blend"][s].append(np.zeros(52, np.float32))
                rec["mat"][s].append(np.zeros(16, np.float32)); rec["lmk"][s].append(Z_LMK)
                rec["pose"][s].append(Z_POSE)
                continue
            im = img[y0:y1, x0:x1]
            if masks[s] is not None:
                im = im.copy()
                mx0, my0, mx1, my1 = masks[s]
                im[max(0, my0 - y0 - 2):max(0, my1 - y0 + 2), max(0, mx0 - x0 - 2):max(0, mx1 - x0 + 2)] = 128
            mi = mpimg(im)
            r = fls[s].detect_for_video(mi, ts)
            if r.face_landmarks:
                L = r.face_landmarks[0]
                rec["present"][s].append(1)
                rec["blend"][s].append(np.array([b.score for b in r.face_blendshapes[0]], np.float32))
                rec["mat"][s].append(np.asarray(r.facial_transformation_matrixes[0], np.float32).reshape(16))
                rec["lmk"][s].append(np.array([[L[j].x, L[j].y, L[j].z] for j in LMK], np.float32))
            else:
                rec["present"][s].append(0); rec["blend"][s].append(np.zeros(52, np.float32))
                rec["mat"][s].append(np.zeros(16, np.float32)); rec["lmk"][s].append(Z_LMK)
            if i % 2 == 0:
                rp = pls[s].detect_for_video(mi, ts)
                if rp.pose_landmarks:
                    Pp = rp.pose_landmarks[0]
                    rec["pose"][s].append(np.array([[Pp[j].x, Pp[j].y, Pp[j].z, Pp[j].visibility] for j in POSE],
                                                   np.float32))
                else:
                    rec["pose"][s].append(Z_POSE)
            else:
                rec["pose"][s].append(rec["pose"][s][-1])
    for o in fls + pls:
        o.close()
    cut = np.array(imgs_cut, np.float32); N = len(cut)
    dyad_ok = frame_ok(ok_s, cut, N)
    present = np.array(rec["present"], np.uint8)
    blend = np.array(rec["blend"], np.float32)
    # duplicate-person guard: one camera feed shown twice (app demo with a self-preview circle, screen recording
    # of an editor, blurred padding copy) -> the two streams blink / turn together
    dup, why_dup = dup_check(present, blend, np.array(rec["mat"], np.float32))
    info["dup"] = why_dup
    pin = [float(present[s][dyad_ok].mean()) if dyad_ok.any() else 0.0 for s in range(S)]
    info["present_in_ok"] = [round(v, 3) for v in pin]
    if not dup and min(pin) < 0.6:
        rej.parent.mkdir(parents=True, exist_ok=True)
        rej.write_text(json.dumps(dict(reason="face tracked < 60%% inside dyad_ok (%s)" % info["present_in_ok"],
                                       info=info, secs=round(time.time() - t0, 1))))
        print(f"[reject] {name} face tracked < 60% inside dyad_ok {info['present_in_ok']} {time.time()-t0:.0f}s",
              flush=True)
        return
    if dup:
        rej.parent.mkdir(parents=True, exist_ok=True)
        rej.write_text(json.dumps(dict(reason="same person twice (%s)" % why_dup, info=info,
                                       secs=round(time.time() - t0, 1))))
        print(f"[reject] {name} same person twice {why_dup} {time.time()-t0:.0f}s", flush=True)
        return
    x = audio(path)
    pr = prosody(x)
    crops = np.array([[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes], np.float32)
    mk = np.array([[m[0] / w, m[1] / h, m[2] / w, m[3] / h] if m is not None else [0, 0, 0, 0] for m in masks],
                  np.float32)
    info["layout"] = det["layout"]
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, layout=det["layout"], streams=np.array(["A", "B"]), probe=json.dumps(info), fps=FPS,
                        present=present, blend=blend, mat=np.array(rec["mat"], np.float32),
                        lmk=np.array(rec["lmk"], np.float32), pose=np.array(rec["pose"], np.float32), cut=cut,
                        prosody=pr, wh=np.array([w, h]), crops=crops, masks=mk, dyad_ok=dyad_ok)
    os.replace(tmp, out)
    print(f"[ok] {name} {det['layout']} {N} present={[round(float(p.mean()), 3) for p in present]} "
          f"in_ok={info['present_in_ok']} ok={dyad_ok.mean():.2f} {time.time()-t0:.0f}s", flush=True)


def detect_only(models, dbg_dir, path):
    t0 = time.time(); name = Path(path).stem
    W0, H0 = probe(path)
    w = min(640, W0 - W0 % 2); h = int(round(H0 * w / W0 / 2)) * 2
    det, why, info = detect_dyad(path, models, w, h)
    if det is None:
        print(f"[reject] {name} {why} {json.dumps(info)} {time.time()-t0:.0f}s", flush=True)
        return
    save_debug(dbg_dir, name, det["G0"], [det["boxA"], det["boxB"]], det["mask"], info)
    print(f"[detect] {name} {det['layout']} A={det['boxA']} B={det['boxB']} mask={det['mask']} {json.dumps(info)} "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    models, out_dir, dbg_dir = args[0], args[1], args[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True); Path(dbg_dir).mkdir(parents=True, exist_ok=True)
    if "--detect-only" in sys.argv:
        for p in args[3:]:
            try:
                detect_only(models, dbg_dir, p)
            except Exception as e:
                import traceback
                print(f"[err] {Path(p).stem}: {e!r} {traceback.format_exc()[-600:]!r}", flush=True)
        sys.exit(0)
    for p in args[3:]:
        try:
            run(models, out_dir, dbg_dir, p)
        except Exception as e:
            import traceback
            print(f"[err] {Path(p).stem}: {e!r} {traceback.format_exc()[-400:]!r}", flush=True)
