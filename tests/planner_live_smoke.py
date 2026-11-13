"""planner LIVE smoke (real gpt-6.1-sol calls, ~$0.01 each; needs OPENAI_API_KEY, never printed).
    python3 tests/planner_live_smoke.py [--out FILE] [--spend FILE]
Runs 'do three jumping jacks' and one unusual request on muse, fox and orb through Planner.plan (library -> cache ->
planner, a fresh cache dir), one at a time so the latencies are honest, then repeats one request to show the cache hit.
Prints a table (first pose / done ms, poses, usd); appends one spend line per call to --spend."""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import aiohttp  # noqa: E402
import planner  # noqa: E402

CASES = [(c, r) for c in ("muse", "fox", "orb")
         for r in ("do three jumping jacks", "play an invisible drum solo that gets faster and faster")]


async def main(a) -> int:
    cache = tempfile.mkdtemp(prefix="avatar-live-cache-")
    pl = planner.Planner(None, print, cache_dir=cache)
    rows, total = [], 0.0
    async with aiohttp.ClientSession() as http:
        for ch, req in CASES + [("fox", "Can you do 3 jumping jacks please")]:
            spent = []
            batches, last = 0, None
            t0 = time.monotonic()
            async for ev in pl.plan(ch, req, None, http=http, on_usd=spent.append):
                if ev["done"]:
                    last = ev
                else:
                    batches += 1
            usd = sum(spent)
            total += usd
            row = {"character": ch, "request": req, "source": last["source"], "ok": last["ok"], "error": last.get("error"),
                   "ms_first_pose": last["ms_first_pose"], "ms_done": last["ms_done"], "poses": len(last["all_poses"]),
                   "batches": batches, "usd": round(usd, 6), "summary": last.get("summary"),
                   "plan_s": last["all_poses"][-1]["t"] if last["all_poses"] else 0, "unknown": last.get("unknown"),
                   "attempts": len(last.get("attempts") or []), "wall_ms": round((time.monotonic() - t0) * 1000),
                   "prompt_sha": last.get("prompt_sha"), "poses_list": last["all_poses"]}
            rows.append(row)
            print(f"{ch:5s} {req[:44]:44s} {row['source']:8s} ok={row['ok']} first {row['ms_first_pose']} done {row['ms_done']} "
                  f"poses {row['poses']} ({row['plan_s']} s) ${usd:.4f} {row['summary']!r}"
                  + (f" unknown={row['unknown']}" if row["unknown"] else "") + (f" ERR {row['error']}" if row["error"] else ""),
                  flush=True)
            if usd > 0 and a.spend:
                with open(a.spend, "a") as f:
                    f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "vendor": "openai",
                                        "what": f"planner live smoke: {ch} '{req}' (gpt-6.1-sol)", "usd": round(usd, 6)}) + "\n")
    print(f"total ${total:.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"model": pl.cfg["model"], "effort": pl.cfg["effort"], "rows": rows, "usd_total": round(total, 6),
                       "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--spend", default="")
    sys.exit(asyncio.run(main(ap.parse_args())))
