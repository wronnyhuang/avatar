#!/usr/bin/env python3
"""r2-boardvis: render chosen moments of a simulated lesson WITH its chalk (the ink up to each moment) — the
look check for the visibility metric.  python tools/board_vis_frames.py --dir LESSON_DIR --t 15.6,24 --out DIR
[--backend cpu|gpu] [--w 960 --h 540] [--view board|follow] [--strip OUT.png]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
from anim.schema import IDX, NPARAMS  # noqa: E402
from scene_stills import raster_chalk, premul  # noqa: E402


def canvas_at(ink, t):
    c = np.zeros((512, 1024, 4), np.uint8)
    strokes = []
    for e in ink:
        if (e.get("t") if e.get("t") is not None else e["at"]) > t + 0.02:
            break
        for s in e.get("segs") or []:
            if s.get("erase"):
                if strokes:
                    raster_chalk(c, strokes, 0.011); strokes = []
                Q = s["pts"]; ru, rv = s.get("ru", .05)/2, s.get("rv", .05)/2
                u0, u1 = min(q[0] for q in Q), max(q[0] for q in Q); v0, v1 = min(q[1] for q in Q), max(q[1] for q in Q)
                c[int(max(0, (v0-rv)*512)):int((v1+rv)*512)+1, int(max(0, (u0-ru)*1024)):int((u1+ru)*1024)+1] = 0
            else:
                strokes.append([tuple(p) for p in s["pts"]] if len(s["pts"]) > 1 else [tuple(s["pts"][0]), tuple(s["pts"][0])])
        for r in e.get("erased") or []:
            if strokes:
                raster_chalk(c, strokes, 0.011); strokes = []
            c[int(r[1]*512):int(r[3]*512)+1, int(r[0]*1024):int(r[2]*1024)+1] = 0
    if strokes:
        raster_chalk(c, strokes, 0.011)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--t", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cpu"); ap.add_argument("--w", type=int, default=960); ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--view", default="follow"); ap.add_argument("--strip", default=""); ap.add_argument("--preset", default="v2")
    a = ap.parse_args()
    import rt_engine
    from PIL import Image, ImageDraw
    rows = [json.loads(l) for l in open(os.path.join(a.dir, "rows.jsonl")) if l.strip()]
    ink = sorted([json.loads(l) for l in open(os.path.join(a.dir, "ink.jsonl")) if l.strip()], key=lambda e: e.get("t") if e.get("t") is not None else e["at"])
    ph = [json.loads(l) for l in open(os.path.join(a.dir, "phases.jsonl")) if l.strip()]
    props = json.load(open(os.path.join(ROOT, "animals", "muse.props.json")))
    cam_b = {p["id"]: p for p in props["places"]}["board"]["camera"]
    eng = rt_engine.RTEngine("muse", a.w, a.h, a.preset, backend=a.backend, threads=12)
    os.makedirs(a.out, exist_ok=True)
    ims = []
    ts = [float(x) for x in a.t.split(",")] if a.t != "end" else [rows[-1]["t"]]
    for t in ts:
        r = min(rows, key=lambda q: abs(q["t"] - t))
        P = np.zeros(NPARAMS, np.float32)
        for k, v in r["p"].items():
            if k in IDX and v is not None:
                P[IDX[k]] = float(v)
        if a.view == "board":
            P[IDX["cam_zoom"]], P[IDX["cam_x"]], P[IDX["cam_y"]] = cam_b["cam_zoom"], cam_b["cam_x"], cam_b["cam_y"]
        eng.surface_write("board", 0, 0, 1024, 512, premul(canvas_at(ink, r["t"])))
        eng.reset_history()
        img = eng.render(0, P)
        im = Image.fromarray(img[..., :3].copy())
        p = min(ph, key=lambda q: abs(q["t"] - r["t"]))
        ImageDraw.Draw(im).text((8, 6), f"t={r['t']:.1f} {p['ph']} {p['item'] or ''}", fill=(255, 255, 0))
        fn = os.path.join(a.out, f"t{r['t']:06.1f}.png"); im.save(fn); ims.append(im)
        print(fn, flush=True)
    if a.strip and ims:
        W, H = ims[0].size; n = len(ims); cols = min(4, n); rws = (n + cols - 1)//cols
        S = Image.new("RGB", (W*cols//2, H*rws//2))
        for i, im in enumerate(ims):
            S.paste(im.resize((W//2, H//2)), ((i % cols)*W//2, (i//cols)*H//2))
        S.save(a.strip); print(a.strip)
    eng.close()


if __name__ == "__main__":
    main()
