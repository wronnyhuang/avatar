"""TV: render a TVL program (or an eval timeline) to an MP4 / PNG strip.
  python tools/tv_clip.py prog.tvl out.mp4 [seconds=10] [fps=12]"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from tvdsl.lang import Doc  # noqa: E402
from tvdsl.render import Renderer  # noqa: E402


def main(a):
    src, out = a[0], a[1]
    secs = float(a[2]) if len(a) > 2 else 10.0
    fps = int(a[3]) if len(a) > 3 else 12
    d = Doc()
    errs = d.apply_text(open(src).read(), 0.0)
    for e in errs:
        print("skipped line", e)
    R = Renderer(512, 288)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x288",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                         stdin=subprocess.PIPE)
    for k in range(int(secs * fps)):
        p.stdin.write(R.render(d, k / fps).tobytes())
    p.stdin.close()
    p.wait()
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
