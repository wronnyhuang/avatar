#!/usr/bin/env python3
"""Scores table for judged runs (markdown): judge axes + the objective measurements, optionally on a clip subset.

  python eval/summarize.py integ-mock-r1 live-real-r1 integ-realtime-r1 [--clips @eval/subsets/old12.txt|holdout]
         [--judge gflash] [--out eval/results/baselines.md]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import os

_EVAL = Path(os.environ.get("EVAL_OUT", str(Path(os.environ.get("EVAL_T", os.environ.get(
    "MM_WORK_DIR", str(Path(__file__).resolve().parents[2] / "work")))) / "eval")))
S = _EVAL      # runs/ live here
T = _EVAL      # clips/manifest.json
AX = ["naturalness", "reaction_timing", "emotional_fit", "gaze", "motion_variety", "commands", "talk_amount", "lip_sync"]
SHORT = {"naturalness": "nat", "reaction_timing": "react", "emotional_fit": "emo", "gaze": "gaze",
         "motion_variety": "var", "commands": "cmd", "talk_amount": "talk", "lip_sync": "lips"}


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(np.median(xs)) if xs else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(np.mean(xs)) if xs else None


def f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def clip_filter(spec: str | None, man: dict):
    if not spec:
        return None
    if spec.startswith("@"):
        return {x.strip() for x in Path(spec[1:]).read_text().split() if x.strip() and not x.startswith("#")}
    if spec in ("holdout", "train"):
        return {k for k, v in man.items() if v.get("split") == spec}
    return set(spec.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--clips")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    manp = next(p for p in [Path(__file__).resolve().parent / "clips" / "manifest.json",
                            T / "clips" / "manifest.json"] if p.exists())
    man = {c["id"]: c for c in json.loads(manp.read_text())["clips"]}
    keep = clip_filter(a.clips, man)
    rows = []
    for rn in a.runs:
        rd = Path(rn) if Path(rn).is_dir() else S / "runs" / rn
        sc = json.loads((rd / "scores.json").read_text())
        j = a.judge or sc.get("primary") or next(iter(sc["judges"]))
        clips = {k: v for k, v in sc["judges"].get(j, {}).items() if keep is None or re.sub(r"__r\d+$", "", k) in keep}
        ms = {}
        for k in clips:
            mp = rd / k / "metrics.json"
            ms[k] = json.loads(mp.read_text()) if mp.exists() else {}
        obj = {}
        for k in clips:
            jp = rd / k / f"judge.{j}.json"
            obj[k] = (json.loads(jp.read_text()).get("objective") or {}) if jp.exists() else {}
        cue_hits = [c.get("in_window") for o in obj.values() for c in (o.get("cues") or [])]
        cue_lat = [c.get("onset_latency_s") for o in obj.values() for c in (o.get("cues") or []) if c.get("in_window")]
        cmd_hits = [c.get("in_window") for o in obj.values() for c in (o.get("cues") or []) if c.get("kind") == "command"]
        rl = [x for o in obj.values() for x in ((o.get("talk") or {}).get("reply_latency_s") or [])[:1]]
        r = {"run": rd.name, "judge": j, "n": len(clips),
             "weighted": mean([v.get("weighted") for v in clips.values()]),
             **{ax: mean([(v.get("axes") or {}).get(ax) for v in clips.values()]) for ax in AX},
             "fps": med([m.get("fps_display_p50") for m in ms.values()]),
             "frozen": mean([(o.get("motion") or {}).get("frozen_frac") for o in obj.values()]),
             "jitter": med([(o.get("motion") or {}).get("jitter_ratio") for o in obj.values()]),
             "loop": mean([(o.get("motion") or {}).get("loop_score") for o in obj.values()]),
             "lip_ms": med([(o.get("lipsync") or {}).get("offset_ms") for o in obj.values()]),
             "lip_corr": med([(o.get("lipsync") or {}).get("peak_corr") for o in obj.values()]),
             "cue_hit": (sum(1 for x in cue_hits if x) / len(cue_hits)) if cue_hits else None,
             "cue_ms": med([x * 1000 for x in cue_lat]) if cue_lat else None,
             "cmd_hit": (sum(1 for x in cmd_hits if x) / len(cmd_hits)) if cmd_hits else None,
             "reply_s": med(rl), "overlap_s": mean([(o.get("talk") or {}).get("overlap_user_s") for o in obj.values()]),
             "usd_voice": sum(float(m.get("usd") or 0) for m in ms.values()),
             "attend": mean([(m.get("perception") or {}).get("attending_ratio") for m in ms.values()]),
             "face": mean([(m.get("perception") or {}).get("face_present_ratio") for m in ms.values()]),
             "blind": sorted(k for k, m in ms.items() if (m.get("perception") or {}).get("face_present_ratio", 0) > 0.5
                             and (m.get("perception") or {}).get("attending_ratio") == 0)}
        rows.append(r)
    hdr = ["run", "n", "weighted"] + [SHORT[x] for x in AX] + ["fps", "frozen", "jitter", "loop", "lip ms", "lip r",
                                                              "cue hit", "cue ms", "cmd hit", "reply s", "over s", "$voice",
                                                              "face", "attend"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join([r["run"], str(r["n"]), f(r["weighted"])] + [f(r[x], 1) for x in AX] +
                                     [f(r["fps"], 0), f(r["frozen"]), f(r["jitter"]), f(r["loop"]), f(r["lip_ms"], 0),
                                      f(r["lip_corr"]), f(r["cue_hit"]), f(r["cue_ms"], 0), f(r["cmd_hit"]),
                                      f(r["reply_s"]), f(r["overlap_s"]), f(r["usd_voice"]), f(r["face"]),
                                      f(r["attend"])]) + " |")
    txt = (f"clips: {a.clips or 'all judged'}\n\n" + "\n".join(out) + "\n")
    for r in rows:
        if r["blind"]:
            txt += (f"\n{r['run']}: perception saw a face but attending stayed 0 in {len(r['blind'])} clips "
                    f"(gaze / wave-back judgments there depend on attention): {', '.join(r['blind'])}\n")
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
