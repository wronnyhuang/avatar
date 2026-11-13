"""Keep K dyad_extract workers busy until every fetch (dyad_fetch*.pid) is finished and all clips are processed.
Sources: manifest_dyad.jsonl (first), then the 'listen' clips of manifest.jsonl already on disk.
Usage: taskset -c 72-83 nice -n 10 python -I dyad_run.py <scratch> <K>
"""
import json, os, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1]); K = int(sys.argv[2]); here = Path(__file__).resolve().parent
out = root / "feats_dyad"; rej = root / "feats_dyad_reject"; dbg = root / "dyad_debug"
for d in (out, rej, dbg):
    d.mkdir(parents=True, exist_ok=True)
done = set(); running = {}


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def fetch_alive():
    for pf in root.glob("dyad_fetch*.pid"):
        try:
            if alive(int(pf.read_text().strip())):
                return True
        except Exception:
            pass
    return False


def clips():
    out_l = []
    for man, cat in ((root / "manifest_dyad.jsonl", None), (root / "manifest.jsonl", "listen")):
        if not man.exists():
            continue
        for l in man.read_text().splitlines():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if cat is None or r.get("category") == cat:
                if r["file"] not in out_l:
                    out_l.append(r["file"])
    return out_l


log = open(root / "dyad_extract.log", "a")
while True:
    cl = clips()
    for p in cl:
        if p in done or p in running or len(running) >= K:
            continue
        st = Path(p).stem
        if (out / f"{st}.npz").exists() or (rej / f"{st}.json").exists():
            done.add(p); continue
        running[p] = subprocess.Popen([sys.executable, "-I", str(here / "dyad_extract.py"), str(root / "models"),
                                       str(out), str(dbg), p], stdout=log, stderr=subprocess.DEVNULL)
    for p, pr in list(running.items()):
        if pr.poll() is not None:
            done.add(p); running.pop(p)
    if not running and not fetch_alive() and all(p in done for p in clips()):
        break
    time.sleep(3)
print("all done", file=log, flush=True)
