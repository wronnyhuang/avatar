#!/usr/bin/env python3
"""Cold vs warm engine init on the K620 (under the GPU lock): cold = the NVIDIA shader disk cache disabled for this
process (__GL_SHADER_DISK_CACHE=0: every program compiles), warm = cache on (a second start).  Backgrounds must already be
baked (bg_s shows it).  One JSON line per character.
  flock gpu.lock python tools/rig_init_time.py fox bear ...  [--w 480 --h 270]
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
w, h = 480, 270
if "--w" in sys.argv: w = int(sys.argv[sys.argv.index("--w") + 1])
if "--h" in sys.argv: h = int(sys.argv[sys.argv.index("--h") + 1])
args = [a for a in args if not a.isdigit()]
for ch in args:
    code = ("import sys,time,json; sys.path[:0]=[%r,%r]; t=time.time(); import rt_engine; "
            "from live.backends.rt import default_preset; e=rt_engine.RTEngine(%r,%d,%d,default_preset(%r,%d,%d,24),'gpu'); "
            "print(json.dumps(dict(total_s=round(time.time()-t,2), **e.init_times)))" % (ROOT, os.path.join(ROOT, "engine"), ch, w, h, ch, w, h))
    rec = {"character": ch, "w": w, "h": h}
    for mode, env in (("warmup", {}), ("cold", {"__GL_SHADER_DISK_CACHE": "0"}), ("warm", {})):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, **env))
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        rec[mode] = json.loads(line[-1]) if line else {"error": r.stderr[-300:]}
    rec.pop("warmup", None)
    print(json.dumps(rec), flush=True)
