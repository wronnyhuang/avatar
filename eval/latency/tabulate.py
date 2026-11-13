#!/usr/bin/env python3
"""One row per run dir (each holding a server session-*.json, optionally run_eval's report.json):
    python tabulate.py DIR [DIR ...]"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from session_report import report, pc  # noqa: E402

print("| run | fps disp p50/p05 | video RTT p50/p90 ms | delay ms | first audio ms (each) | renderer fps / render p95 / dropped | req/s | up kbps p50/p95 / down kbps |")
print("|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    f = sorted(glob.glob(os.path.join(d, "session-*.json")))
    if not f:
        print(f"| {os.path.basename(d)} | (no session) |")
        continue
    s = json.load(open(f[-1]))
    r = report(s)
    v = r.get("video_rtt_ms_last") or {}
    rl = r.get("renderer_last") or {}
    x = r.get("xchg") or {}
    fa = s.get("first_audio_each")
    rep = os.path.join(d, "report.json")
    if os.path.exists(rep):
        try:
            t = json.load(open(rep)).get("turns") or []
            fa = [x2.get("audible_lat_ms") for x2 in t] or fa
        except Exception:
            pass
    print(f"| {os.path.basename(d.rstrip('/'))} | {r['fps_displayed']['p50']}/{r['fps_displayed']['p05']} | {v.get('p50')}/{v.get('p90')} | "
          f"{r.get('delay_ms_last')} | {fa} | {rl.get('fps')} / {rl.get('render_p95')} / {s.get('dropped_frames')},{rl.get('dropped')} | "
          f"{x.get('req_s_p50')} | {x.get('up_kbps_p50')}/{x.get('up_kbps_p95')} / {x.get('down_kbps_p50')} |")
