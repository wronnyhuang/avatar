"""planner model switch LIVE smoke (paid; keys by reference only, never printed).
    python3 tests/planner_routes_smoke.py --spend FILE [--out FILE] [--cap 0.30]
The muse prompt, library and cache off, through Planner.plan on one warm aiohttp session, one call at a time:
2 requests x presets gem35lite / dsv41flash / gpt54mini (each alone, the bench-evaluated settings), then two forced
fallbacks on the live default (gem35lite -> dsv41flash): the primary pointed at an invalid model id (fails at once) and
at Sol with after_s 0.5 (slow first pose: the hedge starts, the first to stream wins, the loser is cancelled + billed).
Every vendor call (attempt) is one JSON line in --spend; paid calls stop at --cap or on a quota / billing error."""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("AVATAR_HOME", tempfile.mkdtemp(prefix="avatar-smoke-home-"))
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S"):
    os.environ.pop(_k, None)
import aiohttp  # noqa: E402
import planner  # noqa: E402

REQUESTS = ["do three jumping jacks", "wave with your left paw then spin around"]
BASE = {"library": False, "cache": False}
VENDOR = {"openai": "openai", "openrouter": "openrouter", "openrouter_zdr": "openrouter", "gemini": "gemini",
          "meta": "meta"}
BILLING = ("402", "insufficient", "quota", "credit", "billing", "payment")


def runs():
    for preset in ("gem35lite", "dsv41flash", "gpt54mini"):
        for r in REQUESTS:
            yield f"{preset}", {**BASE, "model": preset, "fallback": None}, r
    yield ("forced fallback: invalid primary", {**BASE, "model": "google/gemini-no-such-model", "route": "openrouter",
                                                "fallback": {"model": "dsv41flash", "after_s": 2.0}}, REQUESTS[0])
    yield ("forced fallback: slow primary (sol, after_s 0.5)",
           {**BASE, "model": "sol", "fallback": {"model": "dsv41flash", "after_s": 0.5}}, REQUESTS[1])


async def main(a) -> int:
    need = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
    have = {k: bool(os.environ.get(v)) for k, v in need.items()}
    print("keys present:", have)
    rows, total, stop = [], 0.0, ""
    async with aiohttp.ClientSession() as http:
        for name, cfg, req in runs():
            pl = planner.Planner(cfg, lambda m: print("   log:", m))
            routes = {pl.primary()["route"]} | ({pl.fallback()["route"]} if pl.fallback() else set())
            missing = [r for r in routes if not os.environ.get(planner.KEY_ENV[r])]
            if stop or missing:
                rows.append({"run": name, "request": req, "skipped": stop or f"no key for {missing}"})
                print(f"SKIP {name}: {rows[-1]['skipped']}")
                continue
            if total >= a.cap - 0.02:
                stop = f"cap ${a.cap} nearly reached"
                rows.append({"run": name, "request": req, "skipped": stop})
                continue
            t0 = time.monotonic()
            batches, last = [], None
            async for ev in pl.plan("muse", req, None, http=http):
                if ev["done"]:
                    last = ev
                else:
                    batches.append(round((time.monotonic() - t0) * 1000))
            for at in last.get("attempts") or []:
                with open(a.spend, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 3), "who": "planner_routes_smoke",
                                        "vendor": VENDOR[at["route"]], "usd": round(at["usd"], 6),
                                        "what": f"voice-planner smoke: {name} / {at['role']} {at['model']} "
                                                f"'{req}'" + (" (estimated)" if at["estimated"] else "")}) + "\n")
                total += at["usd"]
            row = {"run": name, "request": req, "ok": last["ok"], "model": last.get("model"), "route": last.get("route"),
                   "fallback_used": last.get("fallback_used"), "ms_first_pose": last["ms_first_pose"],
                   "ms_done": last["ms_done"], "poses": len(last["all_poses"]),
                   "plan_s": last["all_poses"][-1]["t"] if last["all_poses"] else 0, "usd": last["usd"],
                   "summary": last.get("summary"), "unknown": last.get("unknown"), "stalls": last.get("stalls"),
                   "error": last.get("error"), "batches_ms": batches,
                   "attempts": [{k: at.get(k) for k in ("role", "model", "route", "effort", "ms_start", "ms_first_pose",
                                                        "ms_end", "usd", "estimated", "cancelled", "error")}
                                for at in last.get("attempts") or []], "poses_list": last["all_poses"]}
            rows.append(row)
            print(f"{name:48s} {req[:40]:40s} ok={row['ok']} {row['model']} first {row['ms_first_pose']} done "
                  f"{row['ms_done']} poses {row['poses']} ({row['plan_s']} s) ${row['usd']:.5f} fb={row['fallback_used']}"
                  + (f" ERR {str(row['error'])[:160]}" if row["error"] else ""), flush=True)
            for at in row["attempts"]:
                print(f"     {at['role']:8s} {at['model']:32s} start {at['ms_start']} first {at['ms_first_pose']} end "
                      f"{at['ms_end']} ${at['usd']:.5f}{' est' if at['estimated'] else ''}"
                      f"{' cancelled' if at['cancelled'] else ''}{' err ' + str(at['error'])[:120] if at['error'] else ''}")
            errs = " ".join(str(at.get("error") or "") for at in row["attempts"]).lower()
            if any(b in errs for b in BILLING) and "no-such-model" not in errs:
                stop = "quota / billing error seen: paid calls stopped"
    print(f"total ${total:.5f}" + (f" ({stop})" if stop else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "usd_total": round(total, 6), "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "defaults": planner.Planner(None, lambda m: None).describe()}, f, indent=1)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spend", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--cap", type=float, default=0.30)
    sys.exit(asyncio.run(main(ap.parse_args())))
