#!/usr/bin/env python3
"""semantic planner END TO END on a local instance, as a page would drive it (no browser, no voice vendor).

    (a running app server of this tree; the semantic planner needs the `claude` CLI or ANTHROPIC_API_KEY)
    python tests/sem_hub_smoke.py --url 'http://127.0.0.1:PORT/?k=KEY' [--character muse] [--out FILE]

Opens a direct-voice session ($0), sends {type:'semwarm'}, then plans over the control channel ('J' uplink):
  1. a fresh command with planner:'semantic'      -> kf space:'rig' batches + sem start/stmt/done, ok
  2. the same command again                        -> source 'cache', one kf message
  3. a library move ('wave') with planner:'semantic' -> today's library kf + planner:'semantic'
  4. a fresh command WITHOUT planner (today's path) -> no sem message, no space:'rig'
Prints one JSON line (timings by the client's clock: plan sent -> first kf batch); exit 1 on a failed check.
Note 4 runs today's keyframe planner (its configured model: one paid call of a few tenths of a cent at most) unless
--skip-today."""
import argparse
import asyncio
import json
import os
import sys
import time
from urllib.parse import urlsplit, parse_qs

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rigdbg_relay_test import Page  # noqa: E402

mono = time.monotonic


async def plan(page, pid, desc, sem=True, character="muse", timeout=25.0):
    n0 = len(page.got)
    t0 = mono()
    msg = {"type": "plan", "id": pid, "character": character, "description": desc,
           "state": {"posture": "standing", "sem": {"user": {"present": True, "ax": 0.1, "ay": 0.0}}} if sem else
           {"posture": "standing"}}
    if sem:
        msg["planner"] = "semantic"
    await page.ctl(msg)
    while mono() - t0 < timeout:
        mine = [(t, j) for t, j in page.got[n0:] if j.get("id") == pid]
        kf = [j for t, j in mine if j.get("type") == "kf"]
        if kf and kf[-1].get("done") and (not sem or any(j.get("type") == "sem" and j.get("phase") == "done"
                                                         for _, j in mine)):
            break
        await asyncio.sleep(0.02)
    mine = [(round((t - t0) * 1000), j) for t, j in page.got[n0:] if j.get("id") == pid]
    first_kf = next((ms for ms, j in mine if j.get("type") == "kf" and j.get("poses")), None)
    return mine, first_kf


async def main(a):
    u = urlsplit(a.url)
    base = f"{u.scheme}://{u.netloc}"
    key = parse_qs(u.query).get("k", [""])[0]
    res, fails = {}, []
    async with aiohttp.ClientSession() as http:
        page = Page(http, base, key)
        await page.open(a.character)
        pl = asyncio.create_task(page.poll_loop())
        try:
            await page.ctl({"type": "semwarm", "character": a.character})
            await asyncio.sleep(6.0)                  # the door warms its processes
            warm = [j for _, j in page.got if j.get("type") == "semwarm"]
            res["semwarm"] = warm[-1] if warm else None
            if not warm or not warm[-1].get("ok"):
                fails.append(f"semwarm: {warm}")
            cmd = a.command
            m1, f1 = await plan(page, "s1", cmd, character=a.character)
            kf = [j for _, j in m1 if j["type"] == "kf"]
            sm = [j for _, j in m1 if j["type"] == "sem"]
            res["fresh"] = {"first_kf_ms": f1, "batches": len(kf) - 1, "poses": sum(len(j["poses"]) for j in kf),
                            "done": kf[-1] if kf else None, "sem_done": sm[-1] if sm else None}
            if res["fresh"]["done"]:
                res["fresh"]["done"] = {k: v for k, v in res["fresh"]["done"].items() if k != "poses"}
            if not kf or not kf[-1].get("ok") or not all(j.get("space") == "rig" for j in kf):
                fails.append("fresh: not ok / not rig")
            if not sm or sm[0]["phase"] != "start" or sm[-1]["phase"] != "done":
                fails.append("fresh: sem phases")
            m2, f2 = await plan(page, "s2", cmd + " please", character=a.character)
            kf2 = [j for _, j in m2 if j["type"] == "kf"]
            res["cache"] = {"first_kf_ms": f2, "msgs": len(kf2), "source": kf2[-1].get("source") if kf2 else None}
            if not kf2 or kf2[-1].get("source") != "cache" or len(kf2) != 1:
                fails.append(f"cache: {res['cache']}")
            m3, f3 = await plan(page, "s3", "wave", character=a.character)
            kf3 = [j for _, j in m3 if j["type"] == "kf"]
            res["library"] = {"source": kf3[-1].get("source") if kf3 else None,
                              "planner": kf3[-1].get("planner") if kf3 else None}
            if res["library"] != {"source": "library", "planner": "semantic"}:
                fails.append(f"library: {res['library']}")
            if not a.skip_today:
                m4, f4 = await plan(page, "s4", "wave like a tired robot", sem=False, character=a.character)
                kf4 = [j for _, j in m4 if j["type"] == "kf"]
                res["today"] = {"first_kf_ms": f4, "source": kf4[-1].get("source") if kf4 else None,
                                "keys": list(kf4[-1]) if kf4 else None}
                if any(j.get("type") == "sem" for _, j in m4) or any("space" in j or "planner" in j for j in kf4):
                    fails.append("today's path got semantic fields")
        finally:
            await page.bye()
            pl.cancel()
    out = {"ok": not fails, "fails": fails, **res}
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--character", default="muse")
    ap.add_argument("--command", default="pretend to tie an invisible bow on a present")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-today", action="store_true")
    sys.exit(asyncio.run(main(ap.parse_args())))
