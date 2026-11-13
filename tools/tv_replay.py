"""TV: re-render the wall TV of a recorded session from its report (report["tv"]["program"]: [[t_session, line]]).
  python tools/tv_replay.py session-....json out.mp4 [t0 t1] [fps=12]      (or out.png = a contact sheet every 2 s)"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import tv_agent  # noqa: E402


def main(a):
    rep = json.load(open(a[0]))
    prog = [(float(t), ln) for t, ln in (rep.get("tv") or {}).get("program") or []]
    if not prog:
        print("no TV program in this report")
        return
    t0 = float(a[2]) if len(a) > 2 else prog[0][0] - 0.5
    t1 = float(a[3]) if len(a) > 3 else prog[-1][0] + 8.0
    fps = int(a[4]) if len(a) > 4 else 12
    out = a[1]
    if out.endswith(".png"):
        from PIL import Image
        times = [t0 + 2.0 * k for k in range(int((t1 - t0) / 2.0) + 1)][:40]
        ims = tv_agent.replay(prog, times, t_origin=0.0)
        cols = 4
        W = Image.new("RGB", (512 * cols, 288 * ((len(ims) + cols - 1) // cols)))
        for k, im in enumerate(ims):
            W.paste(Image.fromarray(im), ((k % cols) * 512, (k // cols) * 288))
        W.save(out)
    else:
        times = [t0 + k / fps for k in range(int((t1 - t0) * fps))]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x288",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                             stdin=subprocess.PIPE)
        for k in range(0, len(times), 24):
            for im in tv_agent.replay(prog, times[k:k + 24], t_origin=0.0):
                p.stdin.write(im.tobytes())
        p.stdin.close()
        p.wait()
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
