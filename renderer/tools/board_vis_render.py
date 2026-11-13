#!/usr/bin/env python3
"""r2-boardvis: what of the chalkboard the muse's body hides, from the REAL renderer and camera.

    python tools/board_vis_render.py --rows ROWS.jsonl --out MASKS.npz [--backend cpu|gpu] [--w 480 --h 270]
                                     [--grid 128x64] [--every 1] [--png-dir DIR]

ROWS.jsonl: one rig frame per line {"t": s, "p": {param: value, ...}} (tests/board_vis_sim.cjs writes them).
Each frame is rendered twice with the same pose — the board surface all white, then empty — at the board station's
camera (places.board.camera). A board texel is VISIBLE in that frame where the two pictures differ (the slate shows
through); where they agree something is in front of it (the muse, a held prop). Occlusion only depends on the camera
POSITION (cam_zoom / cam_x / cam_y turn + zoom about a fixed point, docs/SCENE.md 6), so one framing
that holds the whole board measures every frame. Output: masks[n, gv, gu] (1 = visible) over the board's writable
uv grid (texel centres), t[n].
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402


def board_uv_to_world(props, H, origin, gu, gv):
    """(gv, gu, 3) render-world points (scene units) of the writable area's texel centres."""
    pl = {p["id"]: p for p in props["places"]}["board"]["plane"]
    O, U, V = (np.array(pl[k], float) for k in ("origin", "u", "v"))
    W, Hh = pl["size"]
    a = (np.arange(gu) + 0.5)/gu
    b = (np.arange(gv) + 0.5)/gv
    P = O[None, None] + U[None, None]*(a[None, :, None]*W) + V[None, None]*(b[:, None, None]*Hh)   # H units
    return origin[None, None] + H*P


def project(cam, X, w, h):
    pos, u, v, ww, fl = cam
    d = X - pos
    xc, yc, zc = (d @ u), (d @ v), (d @ ww)
    px = (xc/zc*fl*h + w)*0.5
    py = (yc/zc*fl*h + h)*0.5          # gl_FragCoord y (bottom-up)
    return px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--grid", default="128x64")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--preset", default="full1")
    ap.add_argument("--png-dir", default="")
    ap.add_argument("--png-every", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=40.0)
    a = ap.parse_args()
    gu, gv = (int(x) for x in a.grid.split("x"))
    import rt_engine
    from PIL import Image
    rows = [json.loads(l) for l in open(a.rows) if l.strip()][::a.every]
    props = json.load(open(os.environ.get("AVATAR_PROPS_JSON") or os.path.join(ROOT, "animals", "muse.props.json")))
    cam_b = {p["id"]: p for p in props["places"]}["board"]["camera"]
    t0 = time.time()
    eng = rt_engine.RTEngine("muse", a.w, a.h, a.preset, backend=a.backend, threads=a.threads)
    cam = eng.camera_for(cam_b["cam_zoom"], cam_b["cam_x"], cam_b["cam_y"])
    u = props.get("units") or {}
    Hs = float(u.get("H", 2.3))                                   # the props frame (H units) -> scene units
    Xw = board_uv_to_world(props, Hs, np.array([0.0, float(u.get("base_y", 0.0)), 0.0]), gu, gv)
    px, py = project(cam, Xw, a.w, a.h)
    ix = np.clip(np.round(px - 0.5).astype(int), 0, a.w - 1)
    iy = np.clip(np.round(py - 0.5).astype(int), 0, a.h - 1)
    onscreen = (px >= 0) & (px < a.w) & (py >= 0) & (py < a.h)
    white = np.full((512, 1024, 4), 255, np.uint8)
    empty = np.zeros((512, 1024, 4), np.uint8)
    masks = np.zeros((len(rows), gv, gu), np.uint8)
    ts = np.zeros(len(rows), np.float64)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    flip = None
    for i, r in enumerate(rows):
        P = np.zeros(NPARAMS, np.float32)
        for k, v in r["p"].items():
            j = IDX.get(k)
            if j is not None and v is not None:
                P[j] = float(v)
        P[IDX["cam_zoom"]], P[IDX["cam_x"]], P[IDX["cam_y"]] = cam_b["cam_zoom"], cam_b["cam_x"], cam_b["cam_y"]
        eng.surface_write("board", 0, 0, 1024, 512, white)
        eng.reset_history()
        A = eng.render(2*i, P).astype(np.int16)
        eng.surface_write("board", 0, 0, 1024, 512, empty)
        eng.reset_history()
        B0 = eng.render(2*i + 1, P).astype(np.int16)
        D = np.abs(A[..., :3] - B0[..., :3]).sum(-1) > a.thresh            # (h, w): the board shows here
        if flip is None:                                                     # image rows: top-down or bottom-up?
            hit_td = D[a.h - 1 - iy, ix][onscreen].mean()
            hit_bu = D[iy, ix][onscreen].mean()
            flip = hit_td >= hit_bu
            print(f"orientation: top-down {hit_td:.3f} bottom-up {hit_bu:.3f}", file=sys.stderr)
        yy = (a.h - 1 - iy) if flip else iy
        masks[i] = (D[yy, ix] & onscreen).astype(np.uint8)
        ts[i] = float(r["t"])
        if a.png_dir and a.png_every and i % a.png_every == 0:
            im = A[..., :3].astype(np.uint8).copy()
            Image.fromarray(im).save(os.path.join(a.png_dir, f"w{i:05d}.png"))
        if i % 50 == 0:
            print(f"{i}/{len(rows)} {time.time()-t0:.1f}s vis {masks[i].mean():.3f}", file=sys.stderr, flush=True)
    np.savez_compressed(a.out, masks=masks, t=ts, grid=np.array([gu, gv]))
    print(f"done {len(rows)} frames {time.time()-t0:.1f}s", file=sys.stderr)
    eng.close()


if __name__ == "__main__":
    main()
