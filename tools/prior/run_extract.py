"""Keep K extraction workers busy on clips listed in the manifest until the fetch is done (pid file gone/dead)."""
import json, os, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1]); K = int(sys.argv[2]); here = Path(__file__).parent
done = set(); running = {}
def alive(pid):
    try: os.kill(pid, 0); return True
    except Exception: return False
fpid = int((root / "fetch.pid").read_text())
log = open(root / "extract.log", "a")
while True:
    clips = []
    if (root / "manifest.jsonl").exists():
        for l in (root / "manifest.jsonl").read_text().splitlines():
            try: clips.append(json.loads(l)["file"])
            except Exception: pass
    for p in clips:
        if p in done or p in running or len(running) >= K: continue
        if (root / "feats" / (Path(p).stem + ".npz")).exists(): done.add(p); continue
        running[p] = subprocess.Popen([sys.executable, "-I", str(here / "extract.py"), str(root / "models"), str(root / "feats"), p],
                                      stdout=log, stderr=subprocess.DEVNULL)
    for p, pr in list(running.items()):
        if pr.poll() is not None: done.add(p); running.pop(p)
    if not running and not alive(fpid) and all(p in done for p in clips):
        break
    time.sleep(3)
print("all done", file=log, flush=True)
