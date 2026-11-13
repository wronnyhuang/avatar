#!/usr/bin/env python3
"""Board visibility: run the geometric collision checker (app/board/collide.py) over a replay/eval run directory
(<id>.json files with "items" and "n_setup", as written by tools/board_eval.py) and report per-lesson counts.

  board_collide.py DIR [--out counts.json] [--events] [--all]   (--all: check the setup items too)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

from board.collide import check_items  # noqa: E402

CATS = ("label_ink", "label_label", "item_touch", "item_close", "annot_touch", "out_band", "edge_band", "order_back")


def run(d, check_setup=False, events=False):
    per = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f))
        except ValueError:
            continue
        if not isinstance(j, dict) or "items" not in j:
            continue
        cid = (j.get("case") or {}).get("id") or os.path.basename(f)[:-5]
        rep = check_items(j["items"], start=0 if check_setup else j.get("n_setup", 0))
        row = dict(rep["counts"])
        row["n_labels"] = sum(len(it.get("labels") or []) for it in j["items"][j.get("n_setup", 0):])
        row["kinds"] = sorted({it.get("kind") for it in j["items"] if it.get("strokes")})
        if events:
            row["events"] = rep["events"]
        per[cid] = row
    tot = {c: sum(r[c] for r in per.values()) for c in CATS}
    tot["labels"] = tot["label_ink"] + tot["label_label"]
    tot["all"] = sum(tot[c] for c in CATS)
    tot["n_labels"] = sum(r["n_labels"] for r in per.values())
    dist = collections.Counter(sum(r[c] for c in CATS) for r in per.values())
    clean = sum(1 for r in per.values() if sum(r[c] for c in CATS) == 0)
    return {"dir": d, "n": len(per), "totals": tot, "lessons_clean": clean,
            "per_lesson_total_hist": dict(sorted(dist.items())), "per_case": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    rep = run(a.dir, a.all, a.events)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("n", "totals", "lessons_clean", "per_lesson_total_hist")}))
    for cid, r in rep["per_case"].items():
        s = sum(r[c] for c in CATS)
        if s:
            print("  %-20s %s" % (cid, " ".join("%s=%d" % (c, r[c]) for c in CATS if r[c])))


if __name__ == "__main__":
    main()
