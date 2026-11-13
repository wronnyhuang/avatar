#!/usr/bin/env python3
"""Board visibility: how much of the written chalk the muse's body hides, over a whole lesson.

    python tools/board_vis_metric.py DIR [DIR ...] [--json OUT]

DIR = one lesson: rows.jsonl / ink.jsonl / phases.jsonl / meta.json from tools/board_vis_sim.cjs and masks.npz from
renderer/tools/board_vis_render.py (per rendered frame: which board texels the camera sees). The ink present at each
rendered frame (every chalk mark stamped at or before it, minus what was erased) is rasterised on the same uv grid;
hidden = ink texels the mask says are covered. Time-weighted over the frames of each phase:
  write : the item being written (its own ink so far) while the director is writing it
  write_all : all ink on the board while writing
  say   : all ink while it says a content line / waits for the next item (phases saying, waiting; not the
          'let me make some room' line of an erase, said just before it wipes)
  end   : all ink on the last frame (the lesson finished, the muse turned to him)
  vis80 : fraction of writing frames in which >= 80% of the current item's ink is visible
  line  : the LINE being written: the current item's ink within +-0.06 v of the pen's row (the newest mark), while
          writing (a multi-line item's earlier lines are left out: the head above the paw covers them as it writes
          the next line — they count in write / write_all)
  say_aud: all ink while its voice is actually audible in a say phase (the line playing, not the ~0.6 s before the
          voice starts while it steps clear)
  wipe  : felt stamps (eraser marks) the body hides from the camera in the frame they are stamped (>= 3 of 4 points
          just outside the felt pad hidden: the duster itself always covers the stamp's centre)
  line50: fraction of writing frames in which >= 50% of that line is visible
"""
import json, os, sys
import numpy as np


def raster(pts, gu, gv, grid):
    """mark the texels a polyline crosses (sampled at a quarter texel)"""
    P = np.asarray(pts, float)
    if len(P) == 1:
        P = np.vstack([P, P])
    out = []
    for a, b in zip(P[:-1], P[1:]):
        L = max(abs(b[0] - a[0])*gu, abs(b[1] - a[1])*gv)
        n = int(np.ceil(L*4)) + 1
        s = np.linspace(0, 1, n)[:, None]
        q = a[None] + (b - a)[None]*s
        out.append(q)
    q = np.vstack(out)
    iu = np.clip((q[:, 0]*gu).astype(int), 0, gu - 1)
    iv = np.clip((q[:, 1]*gv).astype(int), 0, gv - 1)
    grid[iv, iu] = True


def lesson(d):
    M = np.load(os.path.join(d, "masks.npz"))
    masks, ts = M["masks"].astype(bool), M["t"]
    gu, gv = (int(x) for x in M["grid"])
    ink = [json.loads(l) for l in open(os.path.join(d, "ink.jsonl")) if l.strip()]
    ph = [json.loads(l) for l in open(os.path.join(d, "phases.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(d, "meta.json")))
    pt = np.array([p["t"] for p in ph])
    try:
        kinds = {it.get("id"): it.get("kind") for it in json.load(open(meta["lesson"]))["items"]}
    except (OSError, ValueError, KeyError):
        kinds = {}
    # ink layers per item, updated as the events arrive (in time order)
    ev = sorted(ink, key=lambda e: (e["t"] if e.get("t") is not None else e["at"]))
    items = {}           # item id -> bool grid
    pen = {}             # item id -> its newest mark (u, v)
    vrow = (np.arange(gv) + 0.5)/gv
    ei = 0
    acc = {k: [0.0, 0.0] for k in ("write", "write_all", "say", "all", "line", "say_aud")}
    vis_ok = [0, 0]
    line_ok = [0, 0]
    wipe = [0, 0]
    tsa = np.asarray(ts, float)
    end = None
    per = []
    for fi, t in enumerate(ts):
        while ei < len(ev) and (ev[ei].get("t") if ev[ei].get("t") is not None else ev[ei]["at"]) <= t + 0.02:
            e = ev[ei]; ei += 1
            if "segs" in e:
                for s in e["segs"]:
                    if s.get("erase"):
                        fj = int(np.argmin(np.abs(tsa - (e["t"] if e.get("t") is not None else e["at"]))))
                        qc = s["pts"][-1]
                        hu_, hv_ = s.get("ru", 0.05)/2 + 0.04, s.get("rv", 0.05)/2 + 0.04
                        ring = [(qc[0] - hu_, qc[1]), (qc[0] + hu_, qc[1]), (qc[0], qc[1] - hv_), (qc[0], qc[1] + hv_)]
                        nh = sum(int(not masks[fj][min(gv - 1, max(0, int(b*gv))), min(gu - 1, max(0, int(a*gu)))]) for a, b in ring)
                        wipe[1] += 1; wipe[0] += int(nh >= 3)
                        Q = np.asarray(s["pts"], float); ru, rv = s.get("ru", 0.05)/2, s.get("rv", 0.05)/2   # (a sweep: the box of both stamps)
                        u0, u1 = int(max(0, (Q[:, 0].min() - ru)*gu)), int(min(gu, np.ceil((Q[:, 0].max() + ru)*gu)))
                        v0, v1 = int(max(0, (Q[:, 1].min() - rv)*gv)), int(min(gv, np.ceil((Q[:, 1].max() + rv)*gv)))
                        for g in items.values():
                            g[v0:v1, u0:u1] = False
                    else:
                        g = items.setdefault(s.get("item") or "?", np.zeros((gv, gu), bool))
                        raster(s["pts"], gu, gv, g)
                        pen[s.get("item") or "?"] = s["pts"][-1]
            elif "erased" in e:
                for r in e["erased"] or []:
                    u0, v0 = int(max(0, r[0]*gu)), int(max(0, r[1]*gv)); u1, v1 = int(np.ceil(r[2]*gu)), int(np.ceil(r[3]*gv))
                    for g in items.values():
                        g[v0:v1, u0:u1] = False
        j = int(np.clip(np.searchsorted(pt, t + 1e-6) - 1, 0, len(ph) - 1))
        p = ph[j]
        allg = np.zeros((gv, gu), bool)
        for g in items.values():
            allg |= g
        n_all = int(allg.sum()); h_all = int((allg & ~masks[fi]).sum())
        acc["all"][0] += h_all; acc["all"][1] += n_all
        rec = {"t": round(float(t), 2), "ph": p["ph"], "item": p["item"], "ink": n_all, "hid": h_all}
        if p["ph"] == "writing" and p["item"]:
            cur = items.get(p["item"])
            acc["write_all"][0] += h_all; acc["write_all"][1] += n_all
            if cur is not None and cur.any():
                n = int(cur.sum()); h = int((cur & ~masks[fi]).sum())
                acc["write"][0] += h; acc["write"][1] += n
                vis_ok[1] += 1; vis_ok[0] += int(h <= 0.2*n)
                rec["cur"] = n; rec["cur_hid"] = h
                pv = pen.get(p["item"])
                if pv is not None:
                    ln = cur & (np.abs(vrow - pv[1]) <= 0.06)[:, None]
                    nl = int(ln.sum()); hl = int((ln & ~masks[fi]).sum())
                    if nl:
                        acc["line"][0] += hl; acc["line"][1] += nl
                        line_ok[1] += 1; line_ok[0] += int(hl <= 0.5*nl)
                        rec["line"] = nl; rec["line_hid"] = hl
        elif p["ph"] in ("saying", "waiting") and not (p.get("say") and kinds.get(p["say"]) == "erase"):
            acc["say"][0] += h_all; acc["say"][1] += n_all
            if p.get("aud"):
                acc["say_aud"][0] += h_all; acc["say_aud"][1] += n_all
        per.append(rec)
        end = (h_all, n_all)
    res = {k: (round(v[0]/v[1], 4) if v[1] else None) for k, v in acc.items()}
    res["end"] = round(end[0]/end[1], 4) if end and end[1] else None
    res["vis80"] = round(vis_ok[0]/vis_ok[1], 3) if vis_ok[1] else None
    res["wipe"] = round(wipe[0]/wipe[1], 3) if wipe[1] else None
    res["wipe_n"] = wipe[1]
    res["line50"] = round(line_ok[0]/line_ok[1], 3) if line_ok[1] else None
    res["id"] = meta.get("id")
    res["dur"] = round(float(meta["done"] - meta["t0"]), 1) if meta.get("done") else None
    res["first_mark"] = meta.get("firstMark", {}) and meta["firstMark"].get("s_after_call")
    res["items"] = meta.get("stats", {}).get("items")
    res["misses"] = meta.get("stats", {}).get("misses")
    res["frames"] = len(ts)
    return res, per


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        args = [a for a in args if a != out]
    R = []
    for d in args:
        r, per = lesson(d)
        with open(os.path.join(d, "per_frame.jsonl"), "w") as f:
            for x in per:
                f.write(json.dumps(x) + "\n")
        R.append(r)
        print(json.dumps(r))
    keys = ["write", "write_all", "line", "line50", "say", "say_aud", "wipe", "end", "all", "vis80", "dur"]
    summ = {k: round(float(np.mean([r[k] for r in R if r.get(k) is not None])), 4) for k in keys if any(r.get(k) is not None for r in R)}
    summ["n"] = len(R)
    summ["worst_say"] = max(R, key=lambda r: r.get("say") or 0)["id"] if R else None
    print("SUMMARY " + json.dumps(summ))
    if out:
        json.dump({"lessons": R, "summary": summ}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
