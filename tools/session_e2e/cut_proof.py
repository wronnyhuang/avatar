#!/usr/bin/env python3
"""cut_proof.py — assemble a proof video (H.264 MP4, <= ~2.5 min) from one or more session runs (run_session.py out dirs).

  python3 tools/session_e2e/cut_proof.py --out proof.mp4 RUN_DIR[:SPEC] [RUN_DIR2[:SPEC] ...] [--max-s 150] [--pad 1.5] [--title "..."]
      SPEC = comma list of turn numbers or ranges ("2-5,7"), or "t=30-90" (recording seconds). No SPEC = every turn.
      --only-pass   keep only turns the judge passed (when a report exists)
      --height 540  output height (default: the recording's)   --no-caption   no burned-in captions   --no-avatar-caption

Each kept turn = from `pad` s before the user starts to `pad` s after the avatar finished (cap --turn-max-s, default 40); if the
total exceeds --max-s every segment is trimmed proportionally from the quiet tail first. Captions (light, bottom): what the user
said, shown while they speak and for 1.2 s after; a small top-left line names the session / turn.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_lib as L  # noqa: E402


def parse_spec(spec: str, tl: dict) -> list[tuple[str, float, float, dict | None]]:
    """-> [(label, mp4_from, mp4_to, turn)] before padding"""
    turns = tl["turns"]
    if not spec:
        idx = [t["idx"] for t in turns]
    elif spec.startswith("t="):
        a, b = spec[2:].split("-")
        return [("t", float(a), float(b), None)]
    else:
        idx = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-"); idx += list(range(int(a), int(b) + 1))
            else:
                idx.append(int(part))
    out = []
    for t in turns:
        if t["idx"] in idx and t.get("mp4_user_start") is not None:
            out.append((f"turn {t['idx']}", t["mp4_user_start"], t["mp4_active_end"] or t["mp4_window_end"], t))
    return out


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%").replace(",", "\\,")


def wrap(s: str, n: int = 52) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-s", type=float, default=150.0)
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--turn-max-s", type=float, default=40.0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--only-pass", action="store_true")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--crf", type=int, default=21)
    a = ap.parse_args()

    segs = []   # (mp4, t0, t1, caption_text, cap_t0, cap_t1, label)
    for spec in a.runs:
        d, _, sp = spec.partition(":")
        d = os.path.abspath(d)
        tl = json.load(open(os.path.join(d, "report.json"))) if os.path.exists(os.path.join(d, "report.json")) else json.load(open(os.path.join(d, "timeline.json")))
        mp4 = os.path.join(d, "session.mp4")
        name = (tl.get("session") or {}).get("script", os.path.basename(d))
        for label, f0, f1, t in parse_spec(sp, tl):
            if a.only_pass and t and (t.get("judge") or {}).get("verdict") != "PASS":
                continue
            t0 = max(0.0, f0 - a.pad); t1 = min(f1 + a.pad, t0 + a.turn_max_s + a.pad)
            cap = t["say"] if t else ""
            segs.append({"mp4": mp4, "t0": t0, "t1": t1, "cap": cap, "cap_a": (f0 - t0) if t else 0, "cap_b": ((t["mp4_user_end"] - t0) + 1.2) if t else 0,
                         "label": f"{name} - {label}"})
    if not segs:
        print("nothing to cut"); return 1
    total = sum(s["t1"] - s["t0"] for s in segs)
    if total > a.max_s:   # trim every segment's tail proportionally (never below its user line + 4 s)
        k = a.max_s / total
        for s in segs:
            keep = max(s["cap_b"] + 3.0, (s["t1"] - s["t0"]) * k)
            s["t1"] = min(s["t1"], s["t0"] + keep)
        total = sum(s["t1"] - s["t0"] for s in segs)
    print(f"[cut] {len(segs)} segments, {total:.1f}s")
    tmp = tempfile.mkdtemp(prefix="cutproof_")
    parts = []
    for i, s in enumerate(segs):
        p = os.path.join(tmp, f"seg{i:03d}.mp4")
        info = L.probe(s["mp4"]); h = a.height or info.get("h") or 540
        h -= h % 2
        vf = [f"scale=-2:{h}"]
        if not a.no_caption:
            if s["cap"]:
                tf = os.path.join(tmp, f"cap{i:03d}.txt"); open(tf, "w").write(wrap('You: "' + s["cap"] + '"'))
                vf.append(f"drawtext=fontfile={L.FONT}:textfile={tf}:x=(w-text_w)/2:y=h-text_h-{int(h*0.05)}:fontsize={max(16, h//24)}:fontcolor=white:"
                          f"box=1:boxcolor=black@0.55:boxborderw=8:enable='between(t,{s['cap_a']:.2f},{s['cap_b']:.2f})'")
            vf.append(f"drawtext=fontfile={L.FONT}:text='{esc(s['label'])}':x=8:y=6:fontsize={max(12, h//38)}:fontcolor=white@0.8:box=1:boxcolor=black@0.35:boxborderw=4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{s['t0']:.2f}", "-t", f"{s['t1']-s['t0']:.2f}", "-i", s["mp4"], "-vf", ",".join(vf),
               "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "128k", p]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print("ffmpeg failed:", r.stderr[-400:]); return 2
        parts.append(p)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out], capture_output=True, text=True)
    if r.returncode:
        print("concat failed:", r.stderr[-400:]); return 2
    print("[cut]", a.out, json.dumps(L.probe(a.out)))
    json.dump({"segments": [{k: v for k, v in s.items()} for s in segs], "total_s": round(total, 1)}, open(os.path.splitext(a.out)[0] + ".cut.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
