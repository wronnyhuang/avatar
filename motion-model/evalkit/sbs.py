#!/usr/bin/env python3
"""Side-by-side videos for review: user camera | rules | unified | human (where it exists), phone-friendly.

  python -I evalkit/sbs.py --ids a,b [--runs rules,unified,human] [--out-dir $T/eval/sbs]

2x2 grid of 480x270 panels (960x540), labelled; Seamless user camera (portrait) cropped to its upper 16:9 band.
Audio: the user's mic (left-centre) + the avatar's voice (right-centre). H.264 <= ~0.9 Mbps video + 64 kbps audio.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E  # noqa: E402

LABELS = {"rules": "TODAY (rules + GRU prior)", "unified": "NEW unified model", "human": "REAL human (reference)",
          "gru-prior": "GRU prior alone", "copy-delay": "copy user, 1 s delay"}


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def lab(t):
    t = t.replace(":", r"\:").replace("'", "")
    return f"drawtext=text='{t}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6"


def make(cid, runs, out_dir, label_unified=None):
    man = manifest()
    c = man[cid]
    cam = E / "clips" / cid / "camera.mp4"
    vids = [(r, E / "runs" / r / cid / "avatar.mp4") for r in runs]
    vids = [(r, p) for r, p in vids if p.exists()]
    if not vids:
        return None
    W, H = 480, 270
    portrait = c["group"] == "seamless"
    crop = "crop=iw:iw*9/16:0:ih*0.10," if portrait else ""
    fc = [f"[0:v]{crop}scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
          f"{lab('USER (camera)')}[p0]"]
    for k, (r, _) in enumerate(vids, start=1):
        name = (label_unified if (r.startswith("unified") and label_unified) else LABELS.get(r.split("~")[0], r))
        fc.append(f"[{k}:v]scale={W}:{H},setsar=1,fps=30,{lab(name)}[p{k}]")
    n = 1 + len(vids)
    while n < 4:
        fc.append(f"color=c=black:s={W}x{H}:r=30[p{n}]"); n += 1
    fc.append("[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=1[v]")
    fc.append("[0:a]pan=stereo|c0=0.9*c0|c1=0.35*c0[ua];[1:a]pan=stereo|c0=0.35*c0|c1=0.9*c0[aa];"
              "[ua][aa]amix=2:duration=shortest:normalize=0[a]")
    out = Path(out_dir) / f"{cid}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(cam)]
    for _, p in vids:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", "-t", f"{float(c['duration_s']):.2f}",
            "-c:v", "libx264", "-preset", "medium", "-b:v", "850k", "-maxrate", "900k", "-bufsize", "1800k",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True); ap.add_argument("--runs", default="rules,unified,human")
    ap.add_argument("--out-dir", default=str(E / "sbs")); ap.add_argument("--label-unified")
    a = ap.parse_args()
    for cid in a.ids.split(","):
        print(make(cid, a.runs.split(","), a.out_dir, a.label_unified))


if __name__ == "__main__":
    main()
