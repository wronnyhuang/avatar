#!/usr/bin/env python3
"""planner latency A/B, frozen prompt vs the q1 variant — plans only (no render, no judge), interleaved
request by request on the same warm claude door so machine load and door state hit both arms alike.

    ~/services/avatar-dev/venv/bin/python tools/plan_latency_ab.py --set <items.jsonl> [--n 16] [--out result.json]

Per arm: first-pose ms and plan-done ms (p50 / p90 / mean), poses per plan, output chars. Cache off, library off, no
fallback (the primary alone, so a slow call is seen, not hidden by the hedge)."""
import argparse, asyncio, json, os, statistics as st, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import planner  # noqa: E402


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out"); ap.add_argument("--character", default="muse")
    a = ap.parse_args()
    texts = [json.loads(l)["utterance"] for l in open(a.set)][:a.n]
    base = {"model": "sonnet55door_nt", "fallback": None, "cache": False, "library": False}
    arms = {"frozen": planner.Planner(dict(base, prompt_variant="frozen", check=False), lambda m: None),
            "q1": planner.Planner(dict(base, prompt_variant="q1"), lambda m: None)}
    for pl in arms.values():
        pl.prewarm(a.character)
    await asyncio.sleep(6)
    rows = []
    for i, text in enumerate(texts):
        for name in (("frozen", "q1") if i % 2 == 0 else ("q1", "frozen")):
            pl = arms[name]
            ev = None
            async for ev in pl.plan(a.character, text, None):
                pass
            att = (ev.get("attempts") or [{}])[0]
            rows.append({"arm": name, "text": text, "ok": ev.get("ok"), "first": ev.get("ms_first_pose"), "done": ev.get("ms_done"),
                         "poses": len(ev.get("all_poses") or []), "chars": att.get("chars"), "warm": (att.get("door") or {}).get("warm"),
                         "door_first": (att.get("door") or {}).get("ms_first"), "model": ev.get("model")})
            print(json.dumps(rows[-1]), flush=True)
            pl.prewarm(a.character)
            await asyncio.sleep(1.5)
    res = {}
    for name in arms:
        r = [x for x in rows if x["arm"] == name and x["ok"] and x["first"] is not None]
        res[name] = {"n": len(r), "first_p50": q([x["first"] for x in r], .5), "first_p90": q([x["first"] for x in r], .9),
                     "first_mean": round(st.mean(x["first"] for x in r)), "done_p50": q([x["done"] for x in r], .5),
                     "poses_p50": q([x["poses"] for x in r], .5), "chars_p50": q([x["chars"] or 0 for x in r], .5),
                     "door_first_token_p50": q([x["door_first"] or 0 for x in r], .5), "prompt_bytes": len(arms[name].prompts[a.character]["text"])}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1)
    await planner.close_doors()

asyncio.run(main())
