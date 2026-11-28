#!/usr/bin/env python3
"""where the planner's time-to-first-pose goes, per call, on the live planner path (no cache, no library).

    python3 tools/plan_latency_probe.py --set <items.jsonl> [--n 20] [--arms q1,frozen]
        [--fallback none|<after_s>] [--character muse] [--gap 1.5] [--out result.json]

Arms are interleaved request by request (ABBA) on the same warm door, so load and door state hit every arm alike.
Per call: door time-to-first-token (API TTFT incl. process hand-off), first pose, done, the gap first token -> first pose
(prompt ordering / parse), usage (input / cache_read / cache_creation / output tokens: is the system prompt cache-hit),
warm (a pre-spawned process or a cold spawn), the 1-min load average, and whether the fallback won.
--fallback none (default) = the primary alone (a slow call is seen, not hidden); a number = the live fallback (gem35lite)
hedged in after that many seconds (the race as served)."""
import argparse, asyncio, json, os, statistics as st, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import planner  # noqa: E402


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--arms", default="q1,frozen"); ap.add_argument("--fallback", default="none")
    ap.add_argument("--character", default="muse"); ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--model", default="sonnet55door_nt"); ap.add_argument("--out")
    ap.add_argument("--arm-cfg", default="{}", help="json: arm name -> extra planner cfg (e.g. {\"q1nt\": {\"prompt_variant\": \"q1\", \"no_think\": true}})")
    a = ap.parse_args()
    texts = [json.loads(l)["utterance"] for l in open(a.set) if l.strip()][:a.n]
    fb = None if a.fallback == "none" else {"model": "gem35lite", "after_s": float(a.fallback)}
    base = {"model": a.model, "fallback": fb, "cache": False, "library": False}
    arms = {}
    extra = json.loads(a.arm_cfg)
    for name in a.arms.split(","):
        c = dict(base, prompt_variant=name, check=name != "frozen", no_think=False)
        c.update(extra.get(name) or {})
        arms[name] = planner.Planner(c, lambda m: None)
    for pl in arms.values():
        pl.prewarm(a.character)
    await asyncio.sleep(8)
    rows = []
    names = list(arms)
    for i, text in enumerate(texts):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            pl = arms[name]
            la = os.getloadavg()[0]
            ev = None
            async for ev in pl.plan(a.character, text, None):
                pass
            atts = ev.get("attempts") or []
            pr = next((x for x in atts if x.get("role") == "primary"), {})
            d = pr.get("door") or {}
            u = pr.get("usage") or {}
            row = {"arm": name, "i": i, "text": text, "ok": ev.get("ok"), "first": ev.get("ms_first_pose"),
                   "done": ev.get("ms_done"), "fallback_used": ev.get("fallback_used"), "model": ev.get("model"),
                   "poses": len(ev.get("all_poses") or []), "ttft": d.get("ms_first"), "door_done": d.get("ms_done"),
                   "primary_first_pose": pr.get("ms_first_pose"), "warm": d.get("warm"), "err": pr.get("error"),
                   "in": u.get("input_tokens"), "cache_read": u.get("cache_read_input_tokens"),
                   "cache_create": u.get("cache_creation_input_tokens"), "out": u.get("output_tokens"),
                   "chars": pr.get("chars"), "load1": round(la, 1), "t": time.strftime("%H:%M:%S")}
            rows.append(row)
            print(json.dumps(row), flush=True)
            pl.prewarm(a.character)
            await asyncio.sleep(a.gap)
    res = {}
    for name in arms:
        r = [x for x in rows if x["arm"] == name]
        ok = [x for x in r if x["ok"] and x["first"] is not None]
        gaps = [x["primary_first_pose"] - x["ttft"] for x in r if x["primary_first_pose"] and x["ttft"] and x["ttft"] > 0]
        res[name] = {"n": len(r), "ok": len(ok),
                     "first_p50": q([x["first"] for x in ok], .5), "first_p90": q([x["first"] for x in ok], .9),
                     "first_max": max([x["first"] for x in ok], default=None),
                     "ttft_p50": q([x["ttft"] for x in r], .5), "ttft_p90": q([x["ttft"] for x in r], .9),
                     "tok_to_pose_p50": q(gaps, .5), "tok_to_pose_p90": q(gaps, .9),
                     "done_p50": q([x["done"] for x in ok], .5), "done_p90": q([x["done"] for x in ok], .9),
                     "over_2500": sum(1 for x in ok if x["first"] > 2500), "over_4500": sum(1 for x in ok if x["first"] > 4500),
                     "fallback_used": sum(1 for x in r if x["fallback_used"]),
                     "cold": sum(1 for x in r if x["warm"] is False),
                     "cache_read_p50": q([x["cache_read"] for x in r], .5),
                     "cache_create_p50": q([x["cache_create"] for x in r], .5),
                     "cache_create_max": max([x["cache_create"] or 0 for x in r], default=None),
                     "out_tokens_p50": q([x["out"] for x in r], .5),
                     "load1_mean": round(st.mean(x["load1"] for x in r), 1),
                     "prompt_bytes": len(arms[name].prompts[a.character]["text"])}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump({"args": vars(a), "summary": res, "rows": rows}, open(a.out, "w"), indent=1)
    await planner.close_doors()

asyncio.run(main())
