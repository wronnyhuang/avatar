"""Live-path check: real held-out audio -> raw per-frame {rms, pitch} (what the browser would measure) -> JSON for
sim_live.mjs, which drives prior.step() through the JS feature trackers (not the training features).
Usage: python -I live_path_check.py <scratch> <out.json> [n_clips]"""
import json, sys, os, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from extract import audio, prosody

root = Path(sys.argv[1]); outp = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
val = set(json.loads((root / "split.json").read_text())["val"])
man = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
meta = json.loads((root / "dataset_meta.json").read_text())
talk = {v["id"]: max([p["speak_frac"] for p in v["persons"]] or [0]) for v in meta}
clips = [m for m in man if m["id"] in val and m["category"] == "speak" and talk.get(m["id"], 0) > 0.5][:n]
seqs = []
for m in clips:
    x = audio(m["file"])
    pr = prosody(x)                                   # 100 Hz: dB, f0, aperiodicity
    N = int(len(pr) / 100 * 30)
    idx = np.clip((np.arange(N) / 30 * 100).astype(int), 0, len(pr) - 1)
    rms = 10 ** (pr[idx, 0] / 20); f0 = pr[idx, 1]
    seqs.append(dict(id=m["id"], rms=[round(float(v), 5) for v in rms], pitch=[round(float(v), 1) for v in f0]))
json.dump(seqs, open(outp, "w"))
print("clips", [s["id"] for s in seqs], [len(s["rms"]) for s in seqs])
