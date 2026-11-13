#!/usr/bin/env python3
"""Board brain: serial latency bench (warm claude_door, one plan at a time, as in a live session).

  board_latency.py --model haiku --n 12 [--out results.json]
"""
import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

import board_brain  # noqa: E402

REQS = ["show how to add 2/5 and 1/4", "solve 5x - 3 = 17", "write the formula for the area of a trapezoid",
        "balance the combustion of methane", "graph y = 2x + 1", "make a flowchart of how a bill becomes a law",
        "explain 23 times 14 in columns", "what is the derivative of x^3 sin x", "write three German greetings",
        "sketch a simple fish", "show Ohm's law with an example: 12 V across 4 ohms", "find the mean of 3, 7, 8, 10"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args()
    board_brain.prewarm(a.model)
    await asyncio.sleep(4)
    rows = []
    for req in REQS[:a.n]:
        st = {}
        items = await board_brain.plan([], req, None, model=a.model, stats=st)
        rows.append({"req": req, "first_item": st["ms_first_item"], "first_token": st["ms_first_token"],
                     "total": st["ms_total"], "n": len(items)})
        print(json.dumps(rows[-1]), flush=True)
        await asyncio.sleep(1.0)       # let the door refill its warm pool, as between real requests
    def p(k, q):
        v = sorted(r[k] for r in rows if r[k] is not None)
        return v[min(len(v) - 1, int(q * len(v)))] if v else None
    summ = {"model": a.model, "n": len(rows)}
    for k in ("first_token", "first_item", "total"):
        summ[k + "_p50"], summ[k + "_p90"] = p(k, 0.5), p(k, 0.9)
    print("SUMMARY " + json.dumps(summ))
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    await board_brain.get_door().close()


if __name__ == "__main__":
    asyncio.run(main())
