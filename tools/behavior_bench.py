#!/usr/bin/env python3
"""Program layer: coder-model benchmark on the DEV set (tests/behavior_dev.cjs, 25 instructions + 2 live edits).

  python3 tools/behavior_bench.py --models gpt-5.4-mini,deepseek-v4.1-flash [--items d01,d02] [--par 3] [--out FILE] [--smoke]

Per model and item: app/behavior_coder.BehaviorCoder.code (forced model, the server's full path: prompt, check, one repair) ->
the item's behaviour check on the returned program (tools/behavior_check.cjs op 'dev'). Scores: check-pass (the server's static
check + dry run), behaviour-pass, repair rate, time-to-running = model + check time over the rounds (+ the page's compile/start,
~2-5 ms), estimated $. Edits use that model's own base program as prev_code.
"""
import argparse, asyncio, json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import behavior_coder as BC  # noqa: E402

def pct(a, p):
    if not a:
        return None
    s = sorted(a)
    return s[min(len(s) - 1, int(p * len(s)))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4-mini,deepseek-v4.1-flash,gemini-3.5-flash-lite")
    ap.add_argument("--items", default="")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "behavior_bench.jsonl"), help="results jsonl (default <repo>/runs/behavior_bench.jsonl)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cap", type=float, default=3.5)
    a = ap.parse_args()
    import aiohttp
    ck = BC.Checker()
    dev = await ck.call({"op": "doc", "character": "fox"})
    # the item list from node
    proc = await asyncio.create_subprocess_exec("node", "-e", "const D=require('./tests/behavior_dev.cjs');console.log(JSON.stringify(D.ALL.map(x=>({id:x.id,character:x.character||null,base:x.base||null,instruction:x.instruction}))))",
                                                cwd=os.path.join(HERE, ".."), stdout=asyncio.subprocess.PIPE)
    items = json.loads((await proc.communicate())[0])
    if a.items:
        want = a.items.split(",")
        items = [x for x in items if any(x["id"].startswith(w) for w in want)]
    if a.smoke:
        items = items[:1]
    base_of = {x["id"]: x for x in items}
    spent = [0.0]
    coders = [BC.BehaviorCoder({"timeout_s": 30.0}, log=lambda *_: None) for _ in range(a.par)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "a")
    summary = {}
    async with aiohttp.ClientSession() as http:
        for model in a.models.split(","):
            progs, rows = {}, []
            q = asyncio.Queue()
            for x in [x for x in items if not x["base"]]:
                q.put_nowait(x)

            async def worker(coder):
                while not q.empty():
                    x = q.get_nowait()
                    if spent[0] > a.cap:
                        return
                    msg = {"id": x["id"], "instruction": x["instruction"], "name": x["id"], "character": x["character"] or "fox"}
                    if x["base"]:
                        b = base_of.get(x["base"]) or {}
                        prev = progs.get(x["base"])
                        if not prev:
                            rows.append({"model": model, "item": x["id"], "skipped": "no base program"})
                            continue
                        msg.update(prev_code=prev, prev_instruction=b.get("instruction"), character=b.get("character") or "fox")
                    t0 = time.monotonic()
                    res = await coder.code(msg, http, model=model)
                    wall = round((time.monotonic() - t0) * 1000)
                    spent[0] += float(res.get("usd") or 0)
                    att = res.get("attempts") or []
                    ttr = sum((r.get("ms") or 0) + (r.get("check_ms") or 0) for r in att)
                    dv = await coder.checker.call({"op": "dev", "item": x["id"], "code": res.get("code") or "({ per_frame(api) {} })"}) if res.get("code") else {"ok": False, "detail": "no code"}
                    if res.get("ok"):
                        progs[x["id"]] = res["code"]
                    row = {"model": model, "item": x["id"], "check_ok": bool(res.get("ok")), "behave_ok": bool(res.get("ok")) and bool(dv.get("ok")),
                           "behave_any": bool(dv.get("ok")), "rounds": len([r for r in att if "ms" in r]), "ttr_ms": ttr, "wall_ms": wall,
                           "model_ms": [r.get("ms") for r in att], "usd": res.get("usd"), "detail": dv.get("detail"), "error": None if res.get("ok") else res.get("error"),
                           "fix": [r.get("fix") for r in att], "code": res.get("code")}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n"); out.flush()
                    print(f"{model:22s} {x['id']:26s} check={row['check_ok']!s:5s} behave={row['behave_ok']!s:5s} rounds={row['rounds']} "
                          f"ttr={ttr} ms ${float(res.get('usd') or 0):.4f}  {str(dv.get('detail'))[:90]}" + (f"  ERR {str(res.get('error'))[:120]}" if not res.get('ok') else ""), flush=True)

            await asyncio.gather(*[worker(c) for c in coders])
            for x in [x for x in items if x["base"]]:     # edits after their bases
                q.put_nowait(x)
            await asyncio.gather(*[worker(c) for c in coders[:1]])
            done = [r for r in rows if "check_ok" in r]
            usd = sum(float(r["usd"] or 0) for r in done)
            ttr = [r["ttr_ms"] for r in done if r["check_ok"]]
            summary[model] = {"n": len(done), "check_pass": sum(r["check_ok"] for r in done), "behave_pass": sum(r["behave_ok"] for r in done),
                              "repaired": sum(r["rounds"] > 1 for r in done), "ttr_p50": pct(ttr, .5), "ttr_p90": pct(ttr, .9),
                              "model_ms_p50": pct([m for r in done for m in r["model_ms"] if m], .5), "usd": round(usd, 4)}
            print("SUMMARY", model, json.dumps(summary[model]), flush=True)
    for c in coders:
        await c.checker.aclose()
    await ck.aclose()
    print(json.dumps(summary, indent=1))
    print(f"estimated vendor cost ${spent[0]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
