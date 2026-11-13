#!/usr/bin/env python3
"""fps table markdown from tools/rig_fps_table.sh JSON lines: python tools/rig_fps_md.py base.jsonl rig.jsonl
(later lines win for the same character/size/motion/tree kind)."""
import json, sys
rows = {}
for f in sys.argv[1:]:
    for l in open(f):
        if not l.startswith("{"):
            continue
        o = json.loads(l)
        kind = "head-only baseline" if "base_tree" in o.get("tree", "") else "rig"
        rows[(o["animal"], f"{o['w']}x{o['h']}", kind, o["motion"])] = o
order = ["fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob", "muse"]   # body: + muse
print("| character | size | build | motion | preset | render mean / p95 ms | + encode: total mean / p95 ms | ≤30/38 |")
print("|---|---|---|---|---|---|---|---|")
for (a, sz, kind, m), o in sorted(rows.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1] != "480x270", kv[0][2] != "head-only baseline", kv[0][3])):
    ok = "yes" if o["total_mean"] <= 30 and o["total_p95"] <= 38 else "no"
    print(f"| {a} | {sz} | {kind} | {m} | {o['preset']} | {o['render_mean']} / {o['render_p95']} | {o['total_mean']} / {o['total_p95']} | {ok} |")
