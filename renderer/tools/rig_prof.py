#!/usr/bin/env python3
"""Per-pass GPU time (GL timer queries) of the real-time engine on a moving pose.  flock gpu.lock python tools/rig_prof.py --animal fox"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
from rig_bench import pose_track  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--animal", default="fox"); ap.add_argument("--w", type=int, default=480); ap.add_argument("--h", type=int, default=270)
ap.add_argument("--preset", default="q50"); ap.add_argument("--frames", type=int, default=60); ap.add_argument("--motion", default="rest")
a = ap.parse_args()
import rt_engine
eng = rt_engine.RTEngine(a.animal, a.w, a.h, a.preset)
P = pose_track(a.frames + 5, a.motion)
for i in range(5):
    eng.render(i, P[i])
prof = {}
for i in range(5, a.frames + 5):
    eng.render(i, P[i], profile=prof)
print(json.dumps({k: round(v*1000/a.frames, 2) for k, v in prof.items()}), flush=True)
