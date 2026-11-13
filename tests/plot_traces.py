"""Draw motion test traces to PNG with PIL only (no matplotlib on the hub). python3 plot_traces.py plots.json"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207)]


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(p):
    t = p["t"]
    W, ph, L, Rm, top = 1400, 170, 70, 170, 40
    H = top + ph * len(p["panels"]) + 30
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f, fs = font(16), font(12)
    d.text((L, 10), p["title"], fill="black", font=f)
    t0, t1 = t[0], t[-1]
    X = lambda x: L + (x - t0) / max(1e-9, t1 - t0) * (W - L - Rm)
    for i, pan in enumerate(p["panels"]):
        y0 = top + i * ph
        ys = [v for ln in pan["lines"] for v in ln["y"]]
        lo, hi = min(ys + [0]), max(ys + [0])
        if hi - lo < 1e-3:
            hi, lo = hi + 0.05, lo - 0.05
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        Y = lambda v: y0 + 8 + (hi - v) / (hi - lo) * (ph - 22)
        d.rectangle([L, y0 + 8, W - Rm, y0 + ph - 14], outline=(200, 200, 200))
        d.line([L, Y(0), W - Rm, Y(0)], fill=(225, 225, 225))
        for m in p.get("marks", []):
            d.line([X(m), y0 + 8, X(m), y0 + ph - 14], fill=(255, 190, 190))
        d.text((4, y0 + 8), f"{hi:.2f}", fill="gray", font=fs)
        d.text((4, y0 + ph - 28), f"{lo:.2f}", fill="gray", font=fs)
        for j, ln in enumerate(pan["lines"]):
            c = COLORS[j % len(COLORS)]
            pts = [(X(a), Y(b)) for a, b in zip(t, ln["y"])]
            d.line(pts, fill=c, width=2)
            d.text((W - Rm + 8, y0 + 10 + 16 * j), ln["name"], fill=c, font=fs)
        d.text((W - Rm + 8, y0 + ph - 30), pan["label"], fill="black", font=fs)
    for s in range(int(t0), int(t1) + 1):
        d.line([X(s), H - 30, X(s), H - 25], fill="black")
        d.text((X(s) - 4, H - 22), str(s), fill="black", font=fs)
    im.save(p["file"])


for p in json.load(open(sys.argv[1])):
    draw(p)
print("plots written")
