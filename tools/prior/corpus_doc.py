"""manifest(s) + dataset_meta -> docs/CORPUS.md + docs/corpus_manifest.jsonl (source URL, license, duration, category,
layout, usable person-minutes). Usage: python -I corpus_doc.py <scratch> <docs_dir>"""
import json, sys
from pathlib import Path
root, docs = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for mf in ("manifest.jsonl", "manifest_dyad.jsonl"):
    p = root / mf
    if p.exists():
        for l in p.read_text().splitlines():
            try: rows.append(json.loads(l))
            except Exception: pass
meta = json.loads((root / "dataset_meta.json").read_text()) if (root / "dataset_meta.json").exists() else []
used = {}
for v in meta:
    key = (v["id"], v["start"])
    used[key] = dict(layout=v.get("layout"), person_min=sum(p["frames"] for p in v["persons"]) / 30 / 60,
                     speak_frac=[round(p["speak_frac"], 2) for p in v["persons"]])
seen = set(); out = []
for r in rows:
    key = (r["id"], r["start"])
    if key in seen: continue
    seen.add(key)
    u = used.get(key)
    out.append(dict(id=r["id"], url=r["url"], title=r.get("title"), channel=r.get("channel"), category=r["category"],
                    start_s=r["start"], dur_s=r["dur"], license=r.get("license"), used=bool(u),
                    layout=u["layout"] if u else None, usable_person_min=round(u["person_min"], 2) if u else 0.0))
(docs / "corpus_manifest.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
cats = {}
for o in out:
    c = cats.setdefault(o["category"], dict(clips=0, raw_h=0.0, used=0, person_h=0.0, videos=set()))
    c["clips"] += 1; c["raw_h"] += o["dur_s"] / 3600; c["videos"].add(o["id"])
    if o["used"]:
        c["used"] += 1; c["person_h"] += o["usable_person_min"] / 60
lay = {}
for o in out:
    if o["used"]: lay[o["layout"]] = lay.get(o["layout"], 0) + 1
tot_raw = sum(c["raw_h"] for c in cats.values()); tot_p = sum(c["person_h"] for c in cats.values())
md = ["# Real-people motion corpus (motion prior)", "",
      "Source: YouTube clip windows (150–180 s, <=480p) found by category search queries "
      "(`tools/prior/fetch_corpus.py`, `tools/prior/dyad_fetch.py`). License: YouTube Standard License — the media "
      "is NOT redistributed and stays in the private scratch dir; only derived per-frame features and aggregate "
      "statistics are used. No accounts/cookies were used. Full per-clip list: `docs/corpus_manifest.jsonl`.", "",
      "| category | clips | videos | raw hours | clips usable | usable person-hours (face tracked, no cuts) |",
      "|---|---|---|---|---|---|"]
for k, c in cats.items():
    md.append(f"| {k} | {c['clips']} | {len(c['videos'])} | {c['raw_h']:.2f} | {c['used']} | {c['person_h']:.2f} |")
md.append(f"| **total** | {len(out)} | {len({o['id'] for o in out})} | {tot_raw:.2f} | {sum(c['used'] for c in cats.values())} | {tot_p:.2f} |")
md += ["", "Layouts of the usable clips: " + ", ".join(f"{k}: {v}" for k, v in sorted(lay.items(), key=lambda x: -x[1])), "",
       "Category = the search intent (speak: talking to camera; listen: split-screen podcasts / calls / interviews / "
       "reactions; idle: study-with-me / silent webcam; dyad: two-person calls with BOTH faces visible, panel-cropped). "
       "Per-frame states (speaking / listening / idle) are assigned from audio VAD + the face's own jaw activity, "
       "not from the category — see docs/MOTION_STATS.md."]
(docs / "CORPUS.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
