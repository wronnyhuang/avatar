#!/usr/bin/env python3
"""scene: stills of the muse's chalkboard + wall TV with test content on the live surfaces.

    python tools/scene_stills.py --out DIR [--backend cpu|gpu] [--w 960 --h 540] [--views home,board,tv,write,desk]
                                 [--text "2+2=4"] [--digit 0.09]

Writes test chalk (a stroke font, digit height --digit of the board's writable height) and a TV test picture into
the engine's surfaces (the same textures the render server fills from the app server's 'S' messages), then renders each
view (TAA settled) to DIR/<view>.png.  Views: home (rest camera), board (the board station camera), tv (the TV
station camera), write (the muse at the board station, paw raised, board camera: occlusion), desk (the muse at the
desk area, rest camera: the old routine's look), wide (cam_zoom 1).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402

# a single-stroke font in a 0..1 box (x right, y DOWN), for test writing only (BOARDBRAIN has the real one)
FONT = {
    "0": [[(.5, 0), (.15, .15), (.05, .5), (.15, .85), (.5, 1), (.85, .85), (.95, .5), (.85, .15), (.5, 0)]],
    "1": [[(.3, .2), (.55, 0), (.55, 1)], [(.3, 1), (.8, 1)]],
    "2": [[(.1, .25), (.3, .03), (.65, .02), (.88, .22), (.85, .45), (.1, 1), (.92, 1)]],
    "3": [[(.12, .1), (.45, 0), (.82, .12), (.8, .38), (.45, .48), (.85, .6), (.88, .85), (.5, 1), (.1, .9)]],
    "4": [[(.7, 1), (.7, 0), (.05, .7), (.95, .7)]],
    "5": [[(.85, 0), (.2, 0), (.15, .45), (.55, .38), (.88, .6), (.82, .9), (.45, 1), (.1, .9)]],
    "6": [[(.8, .05), (.4, .05), (.12, .45), (.15, .85), (.5, 1), (.85, .82), (.8, .55), (.45, .45), (.13, .6)]],
    "7": [[(.08, 0), (.92, 0), (.4, 1)]],
    "8": [[(.5, .47), (.18, .3), (.25, .05), (.5, 0), (.75, .05), (.82, .3), (.5, .47), (.12, .7), (.2, .95), (.5, 1),
           (.8, .95), (.88, .7), (.5, .47)]],
    "9": [[(.85, .4), (.5, .55), (.15, .4), (.2, .1), (.5, 0), (.85, .15), (.85, .6), (.6, 1), (.2, .95)]],
    "+": [[(.5, .2), (.5, .8)], [(.15, .5), (.85, .5)]],
    "-": [[(.15, .5), (.85, .5)]],
    "=": [[(.12, .35), (.88, .35)], [(.12, .65), (.88, .65)]],
    "x": [[(.15, .3), (.85, 1)], [(.85, .3), (.15, 1)]],
    "y": [[(.15, .3), (.5, .7)], [(.85, .3), (.3, 1.25)]],
    "(": [[(.7, -.05), (.35, .3), (.35, .7), (.7, 1.05)]],
    ")": [[(.3, -.05), (.65, .3), (.65, .7), (.3, 1.05)]],
    "/": [[(.85, -.05), (.15, 1.05)]],
    " ": [],
}


def text_strokes(s, x0, y0, h):
    """uv polylines for string s, top-left (x0, y0) in board uv, digit height h (v units); board aspect 2."""
    out, x = [], x0
    w = 0.62*h/2.0           # glyph width in uv-x (aspect 2: 1 uv-x = 2 v)
    for ch in s:
        for st in FONT.get(ch, []):
            out.append([(x + px*w, y0 + py*h) for px, py in st])
        x += w*1.35
    return out


def raster_chalk(canvas, strokes, width_v, rgba=(242, 242, 232, 235)):
    """Soft anti-aliased thick polylines OVER the canvas (straight alpha), via a 4x supersampled PIL mask."""
    from PIL import Image, ImageDraw, ImageFilter
    H, W = canvas.shape[:2]
    S = 4
    m = Image.new("L", (W*S, H*S), 0)
    d = ImageDraw.Draw(m)
    wpx = max(1, int(round(width_v*H*S)))
    for st in strokes:
        pts = [(u*W*S, v*H*S) for u, v in st]
        d.line(pts, fill=255, width=wpx, joint="curve")
        for p in (pts[0], pts[-1]):
            d.ellipse([p[0]-wpx/2, p[1]-wpx/2, p[0]+wpx/2, p[1]+wpx/2], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(S*0.6)).resize((W, H), Image.BOX)
    a = np.asarray(m, np.float32)/255.0*(rgba[3]/255.0)
    old_a = canvas[..., 3]/255.0
    out_a = a + old_a*(1 - a)
    rgb = (np.array(rgba[:3], np.float32)[None, None]*a[..., None] + canvas[..., :3]*(old_a*(1 - a))[..., None])/np.maximum(out_a, 1e-6)[..., None]
    canvas[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[..., 3] = np.clip(out_a*255 + 0.5, 0, 255).astype(np.uint8)
    return canvas


def premul(c):
    a = c[..., 3:4].astype(np.float32)/255.0
    out = c.copy()
    out[..., :3] = np.clip(c[..., :3]*a + 0.5, 0, 255).astype(np.uint8)
    return out


def tv_test(w=512, h=288, phase=0.0):
    """A colourful test illustration: a sky gradient, a lens with rays, a label."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h))
    px = np.zeros((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    px[..., 0] = 20 + 30*(1 - yy); px[..., 1] = 30 + 40*(1 - yy); px[..., 2] = 70 + 60*(1 - yy)
    im = Image.fromarray(px.astype(np.uint8))
    d = ImageDraw.Draw(im)
    cx, cy = w*0.5, h*0.55
    d.ellipse([cx - 14, cy - 80, cx + 14, cy + 80], outline=(170, 220, 255), width=3, fill=(60, 110, 160))
    d.line([(20, cy), (w - 20, cy)], fill=(200, 200, 200), width=1)
    f = 110
    for k, yo in enumerate((-55, -25, 25, 55)):
        col = [(255, 210, 80), (255, 140, 80), (120, 230, 140), (120, 180, 255)][k]
        d.line([(30, cy + yo), (cx, cy + yo)], fill=col, width=3)
        d.line([(cx, cy + yo), (cx + f*1.6, cy - yo*0.6)], fill=col, width=3)
    d.ellipse([cx + f - 6, cy - 6, cx + f + 6, cy + 6], fill=(255, 255, 255))
    x = 40 + (w - 80)*(0.5 + 0.5*np.sin(phase))
    d.ellipse([x - 10, 20, x + 10, 40], fill=(255, 90, 120))
    d.rectangle([12, h - 40, 220, h - 12], fill=(250, 250, 250))
    d.text((20, h - 34), "converging lens  f", fill=(20, 20, 30))
    for i, c in enumerate([(230, 60, 60), (240, 180, 40), (60, 200, 90), (60, 140, 230), (160, 80, 220)]):
        d.rectangle([w - 160 + i*28, 14, w - 136 + i*28, 46], fill=c)
    return np.asarray(im, np.uint8)


def fill_surfaces(eng, text, digit, board=True, tv=True):
    if board and "board" in eng.surface_names():
        c = np.zeros((512, 1024, 4), np.uint8)
        # rows inside the chalk's reach band (board uv-y 0.32..0.95, places.board.station.reach_uv_y)
        y0, dy = 0.36, digit*1.9
        strokes = text_strokes(text, 0.05, y0, digit)
        strokes += text_strokes("3x+1=7", 0.05, y0 + dy, digit)
        strokes += text_strokes("3x=6", 0.05, y0 + 2*dy, digit)
        strokes += text_strokes("x=2", 0.05, y0 + 3*dy, digit)
        # a diagram: a right triangle with its right-angle box, legs 3 and 4, and a circle
        strokes += [[(0.56, 0.88), (0.80, 0.88), (0.80, 0.42), (0.56, 0.88)], [(0.775, 0.88), (0.775, 0.83), (0.80, 0.83)]]
        strokes += text_strokes("3", 0.665, 0.895, digit*0.6) + text_strokes("4", 0.815, 0.62, digit*0.6)
        th = np.linspace(0, 2*np.pi, 48)
        strokes += [[(0.91 + 0.04*np.cos(t), 0.55 + 0.08*np.sin(t)) for t in th]]
        raster_chalk(c, strokes, 0.012)
        raster_chalk(c, text_strokes("(2+2)", 0.30, y0, digit), 0.012, rgba=(250, 214, 90, 235))   # yellow chalk
        eng.surface_write("board", 0, 0, 1024, 512, premul(c))
    if tv and "tv" in eng.surface_names():
        img = tv_test()
        if TV_PNG[0]:                     # tv: a real TV-agent frame instead of the test picture
            from PIL import Image
            img = np.asarray(Image.open(TV_PNG[0]).convert("RGB").resize((512, 288)))
        rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], 2)
        eng.surface_write("tv", 0, 0, 512, 288, rgba)


CHALK_SLOT = [6]
TV_PNG = [None]                           # tv: --tv-png


def view_params(props, view):
    P = np.zeros(NPARAMS, np.float32)
    def setp(**kw):
        for k, v in kw.items():
            P[IDX[k]] = v
    pl = {p["id"]: p for p in props["places"]}
    cam = lambda c: setp(cam_zoom=c["cam_zoom"], cam_x=c["cam_x"], cam_y=c["cam_y"])
    if view.startswith("p:"):         # free params: "p:cam_zoom=0.3;prop6_hl=1;hand_l_y=0.5"
        for kv in view[2:].split(";"):
            if kv:
                k, v = kv.split("=")
                setp(**{k: float(v)})
    elif view == "home":
        setp(cam_zoom=0.3)            # the standing muse's own framing (app/moves.js: cam_zoom 0.3 while on its legs)
    elif view == "rest":
        pass                          # cam_zoom 0: the seated characters' framing
    elif view == "board":
        cam(pl["board"]["camera"])
    elif view == "tv":
        cam(pl["tv"]["camera"])
    elif view == "wide":
        setp(cam_zoom=1.0)
    elif view == "write":
        st = pl["board"]["station"]
        cam(pl["board"]["camera"])
        setp(root_x=st["pos"][0], root_z=st["pos"][2], root_yaw=st["yaw"], gaze_free=0.6,
             hand_l_x=0.55, hand_l_y=0.85, hand_l_z=-0.55, hand_l_open=0.3, head_yaw=-0.25)
    elif view.startswith("write@"):     # write@0.35: the muse at station(a) (the contract's S(a)), writing column a
        a = float(view.split("@")[1])
        b = pl["board"]["plane"]; o, u, n = np.array(b["origin"]), np.array(b["u"]), np.array(b["n"])
        S = o*np.array([1, 0, 1]) + u*(a*b["size"][0] + 0.2348) + n*0.3043
        st = pl["board"]["station"]
        cam(pl["board"]["camera"])
        setp(root_x=S[0], root_z=S[2], root_yaw=st["yaw"], gaze_free=0.6,
             hand_l_x=0.55, hand_l_y=0.85, hand_l_z=-0.55, hand_l_open=0.3, head_yaw=-0.25)
    elif view == "aside":              # presenting the finished board: the muse at station.aside, facing the user
        st = pl["board"]["station"]
        cam(pl["board"]["camera"])
        setp(root_x=st["aside"][0], root_z=st["aside"][2], root_yaw=-0.03, hand_l_x=0.5, hand_l_y=0.5, hand_l_z=-0.2,
             point_l=1.0)
    elif view in ("hold", "holdz"):       # (with a test props file that puts the chalk in slot 6, --chalk-slot 6)
        st = pl["board"]["station"]
        cam(pl["board"]["camera"])
        k = CHALK_SLOT[0]
        setp(root_x=st["pos"][0], root_z=st["pos"][2], root_yaw=st["yaw"], gaze_free=0.6,
             hand_l_x=0.55, hand_l_y=0.85, hand_l_z=-0.55, hand_l_open=0.3, head_yaw=-0.25)
        setp(**{f"prop{k}_hl": 1.0})
        if view == "holdz":
            setp(**{f"prop{k}_rz": 0.2, f"prop{k}_ry": -0.1})
    elif view == "present":
        st = pl["tv"]["station"]
        cam(pl["tv"]["camera"])
        setp(root_x=st["pos"][0], root_z=st["pos"][2], root_yaw=st["yaw"], hand_r_x=0.6, hand_r_y=0.9, hand_r_z=-0.2,
             point_r=1.0, head_yaw=0.15)
    elif view == "desk":
        setp(root_x=0.55, root_z=-0.25, root_yaw=0.12, cam_x=0.4, cam_zoom=0.3)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--views", default="home,board,tv,write,present,desk,wide")
    ap.add_argument("--text", default="2+2=4")
    ap.add_argument("--digit", type=float, default=0.09)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--empty", action="store_true", help="no surface content (the idle look)")
    ap.add_argument("--tv-png", default="", help="tv: a 512x288 TV frame to show instead of the test picture")
    ap.add_argument("--chalk-slot", type=int, default=6)
    a = ap.parse_args()
    TV_PNG[0] = a.tv_png or None          # tv
    CHALK_SLOT[0] = a.chalk_slot
    import rt_engine
    from PIL import Image
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset("muse", a.w, a.h, 24)
    t0 = time.time()
    eng = rt_engine.RTEngine("muse", a.w, a.h, preset, backend=a.backend, threads=a.threads)
    print(f"engine {time.time()-t0:.1f}s {eng.init_times} surfaces={eng.surface_names()}", file=sys.stderr, flush=True)
    if not a.empty:
        fill_surfaces(eng, a.text, a.digit)
    props = json.load(open(os.environ.get("AVATAR_PROPS_JSON") or os.path.join(ROOT, "animals", "muse.props.json")))
    os.makedirs(a.out, exist_ok=True)
    for v in a.views.split(","):
        P = view_params(props, v)
        eng.reset_history()
        for f in range(a.settle):
            img = eng.render(100 + f, P)
        fn = os.path.join(a.out, (v if not v.startswith("p:") else "p_" + v[2:].replace(";", "_").replace("=", ""))[:80] + ".png")
        Image.fromarray(img).save(fn)
        print(fn, f"{time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    eng.close()


if __name__ == "__main__":
    main()
