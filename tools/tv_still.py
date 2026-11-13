"""TV: render a TVL program (whole file applied at t=0, or an eval .tvl timeline) to a 2x2 montage PNG.
  python tools/tv_still.py prog.tvl out.png [t1,t2,t3,t4]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from PIL import Image, ImageDraw  # noqa: E402
from tvdsl.lang import Doc  # noqa: E402
from tvdsl.render import Renderer  # noqa: E402


def main(a):
    times = [float(x) for x in (a[2].split(",") if len(a) > 2 else "0.8,2,4,7".split(","))]
    d = Doc()
    for e in d.apply_text(open(a[0]).read(), 0.0):
        print("skipped line", e)
    R = Renderer(512, 288)
    W, H = 512, 288
    M = Image.new("RGB", (W * 2, (H + 16) * ((len(times) + 1) // 2)), (40, 40, 40))
    dr = ImageDraw.Draw(M)
    for k, t in enumerate(times):
        x, y = (k % 2) * W, (k // 2) * (H + 16)
        M.paste(Image.fromarray(R.render(d, t)), (x, y + 16))
        dr.text((x + 4, y + 2), f"t={t:g}s", fill=(230, 230, 230))
    M.save(a[1])
    print(a[1])


if __name__ == "__main__":
    main(sys.argv[1:])
