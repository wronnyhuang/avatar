"""semantic-planner LATENCY bench through Planner end to end (run_plan mode='semantic', the real Claude door,
the real app/semantic package). Subscription calls (no API money); every call's outcome is saved and NOT re-run.

    .venv/bin/python tests/sem_hub_latency.py --out DIR [--sem APP_DIR_WITH_semantic]
        [--tiers haiku_nt,sonnet_nt,haiku_low,sonnet_low] [--n 24] [--character muse] [--suffix FILE] [--tag NAME]
        [--cache-hits 5] [--report-only]

Stages (ms since the hub received the plan): first_token, first_stmt (Stream.feed returned the first batch),
first_pose (the first kf batch was yielded = sent), done. Text-only (independent of the interpreter's speed):
stmt_text = the first top-level statement complete in the text by the column-0 rule; stmt_early = a simple first
statement complete at its newline (balanced, parses). Commands: COMMANDS below (written for this bench; fresh, varied;
not the bake-off's sealed set)."""
import argparse
import ast
import asyncio
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")

TIERS = {"haiku_nt": ("claude-haiku-5-5", "none"), "haiku_low": ("claude-haiku-5-5", "low"),
         "sonnet_nt": ("claude-sonnet-5-5", "none"), "sonnet_low": ("claude-sonnet-5-5", "low"),
         "opus_nt": ("claude-opus-5-5", "none")}

COMMANDS = [
    "scratch your chin like you're puzzled", "pretend to take a photo of me", "act like you're freezing cold",
    "show me how big the fish was", "do a slow-motion facepalm", "salute me",
    "look over your left shoulder suspiciously", "pretend to type on a keyboard",
    "give me a double thumbs up and a wink", "act like you just smelled something terrible",
    "cup your ear like you can't hear me", "stretch your arms up like you just woke up",
    "count to three on your fingers", "point at yourself and then at me",
    "rub your hands together like you have an evil plan", "pretend to juggle three balls",
    "shiver and hug yourself", "do a little drumroll in the air", "fan yourself like it's really hot",
    "shush me with a finger on your lips", "flex your right arm like a bodybuilder",
    "pretend to look through binoculars", "shake your fist at the sky", "tip an imaginary hat to me",
    "wipe sweat off your forehead with relief", "conduct an invisible orchestra", "yawn and stretch your neck",
    "beckon me to come closer", "pretend to knock on a door", "do a chef's kiss",
]


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(len(v) - 1, int(k) + 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo))


def _code_start(text):
    i = text.find("```")
    if i < 0:
        return None
    j = text.find("\n", i)
    return None if j < 0 else j + 1


def stmt_text_ms(deltas, early=False):
    """ms at which the first top-level statement is complete in the streamed text"""
    text = ""
    for ms, d in deltas:
        text += d
        s = _code_start(text)
        if s is None:
            continue
        body = text[s:]
        lines = body.split("\n")
        seen = None
        for k, ln in enumerate(lines[:-1]):
            if ln.startswith("```"):
                if seen is not None:
                    return ms
                break
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if ln[0] not in " \t":
                if seen is not None:
                    return ms
                seen = k
                if early and not ln.rstrip().endswith(":"):
                    try:
                        ast.parse(ln)
                        return ms
                    except SyntaxError:
                        pass
    return None


async def run_tier(planner, tier, cmds, outdir, character, suffix, log, hedge_after=None):
    model, effort = TIERS[tier]
    os.makedirs(outdir, exist_ok=True)
    todo = [(i, c) for i, c in enumerate(cmds) if not os.path.exists(os.path.join(outdir, f"{i:02d}.json"))]
    if not todo:
        return
    hedge = None if hedge_after is None else {"model": model, "effort": effort, "after_s": hedge_after}
    pl = planner.Planner({"library": False, "semantic": {"model": model, "effort": effort, "cache": False,
                                                         "suffix": suffix, "hedge": hedge}}, log, cache_dir=None)
    rec_deltas = []
    clock = {"t0": 0.0}

    async def fn(spec, system, user, deadline, info, tag):
        async for d in planner.door_stream(spec, system, user, deadline, info, tag,
                                           door=planner.sem_door(spec.get("effort"))):
            rec_deltas.append((round((time.monotonic() - clock["t0"]) * 1000, 1), d))
            yield d
    pl.sem_stream_fn = fn
    await pl.prewarm_semantic_async(character)
    await asyncio.sleep(6.0)
    for i, c in todo:
        rec_deltas.clear()
        sent = []
        clock["t0"] = t0 = time.monotonic()
        rec = await planner.run_plan(pl, f"b{i}", character, c, None, sent.append, mode="semantic", t_recv=t0)
        sems = [m for m in sent if m["type"] == "sem"]
        done = sems[-1] if sems else {}
        kf_sent = [(m.get("done"), len(m.get("poses") or [])) for m in sent if m["type"] == "kf"]
        att = (rec.get("attempts") or [{}])[-1]
        out = {"tier": tier, "model": model, "effort": effort, "i": i, "command": c, "character": character,
               "ok": rec.get("ok"), "error": rec.get("error"), "ms": done.get("ms"), "program": done.get("program"),
               "warnings": done.get("warnings"), "errors": done.get("errors"), "exec_ms": done.get("exec_ms"),
               "n_statements": done.get("n_statements"), "duration_s": done.get("duration_s"),
               "deltas": list(rec_deltas), "kf": kf_sent, "attempts": rec.get("attempts"), "retried": rec.get("retried"),
               "suffix": bool(suffix), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        # the door's done event (usage: cache reads) lives on the planner's attempt record
        out["usage"] = None
        with open(os.path.join(outdir, f"{i:02d}.json"), "w") as f:
            json.dump(out, f, indent=1)
        m = out["ms"] or {}
        log(f"[{tier}] {i:02d} ok={out['ok']} tok {m.get('first_token')} stmt {m.get('first_stmt')} pose "
            f"{m.get('first_pose')} done {m.get('done')} st={out['n_statements']} err={len(out['errors'] or [])} "
            f"{c!r}")
        await asyncio.sleep(2.0)                 # the door refills its warm process
    await planner.close_doors()


def summarize(outdir, tiers):
    rows = []
    for tier in tiers:
        d = os.path.join(outdir, tier)
        if not os.path.isdir(d):
            continue
        R = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]
        if not R:
            continue
        ms = lambda k: [((r.get("ms") or {}).get(k)) for r in R if r.get("ok")]   # noqa: E731
        st = [stmt_text_ms(r["deltas"]) for r in R if r.get("ok")]
        se = [stmt_text_ms(r["deltas"], early=True) for r in R if r.get("ok")]
        row = {"tier": tier, "n": len(R), "ok": sum(1 for r in R if r.get("ok")),
               "with_errors": sum(1 for r in R if r.get("errors")),
               "with_warnings": sum(1 for r in R if r.get("warnings")),
               "stmts": round(statistics.mean([r.get("n_statements") or 0 for r in R]), 1),
               "dur_s": round(statistics.mean([r.get("duration_s") or 0 for r in R]), 2)}
        for k in ("first_token", "first_stmt", "first_pose", "done"):
            row[k] = (pct(ms(k), .5), pct(ms(k), .9))
        row["stmt_text"] = (pct(st, .5), pct(st, .9))
        row["stmt_early"] = (pct(se, .5), pct(se, .9))
        ex1 = [r["exec_ms"][0] for r in R if r.get("exec_ms")]
        row["exec_first"] = (pct(ex1, .5), pct(ex1, .9))
        rows.append(row)
    hdr = ["tier", "n", "ok", "with_errors", "with_warnings", "stmts", "dur_s", "first_token", "stmt_text", "stmt_early",
           "first_stmt", "exec_first", "first_pose", "done"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[h][0]}/{r[h][1]}" if isinstance(r[h], tuple) else str(r[h]) for h in hdr)
                     + " |")
    return rows, "\n".join(lines)


async def replay_tier(planner, src, dst, character, log):
    """the recorded door streams of a tier (src/NN.json deltas, at their recorded times) through run_plan with THIS
    semantic package: hub + interpreter latency without new model calls"""
    os.makedirs(dst, exist_ok=True)
    pl = planner.Planner({"library": False, "semantic": {"model": "claude-haiku-5-5", "cache": False, "warm_n": 0}}, log,
                         cache_dir=None)
    cur = {"deltas": []}

    async def fn(spec, system, user, deadline, info, tag):
        t0 = time.monotonic()
        for ms, d in cur["deltas"]:
            w = ms / 1000.0 - (time.monotonic() - t0)
            if w > 0:
                await asyncio.sleep(w)
            yield d
    pl.sem_stream_fn = fn
    for f in sorted(x for x in os.listdir(src) if x.endswith(".json")):
        r = json.load(open(os.path.join(src, f)))
        if not r.get("deltas"):
            continue
        cur["deltas"] = r["deltas"]
        sent = []
        t0 = time.monotonic()
        rec = await planner.run_plan(pl, "r", character, r["command"], None, sent.append, mode="semantic", t_recv=t0)
        sems = [m for m in sent if m["type"] == "sem"]
        done = sems[-1] if sems else {}
        out = dict(r, ok=rec.get("ok"), error=rec.get("error"), ms=done.get("ms"), warnings=done.get("warnings"),
                   errors=done.get("errors"), exec_ms=done.get("exec_ms"), n_statements=done.get("n_statements"),
                   duration_s=done.get("duration_s"), replayed_from=src)
        json.dump(out, open(os.path.join(dst, f), "w"), indent=1)
        m = out["ms"] or {}
        log(f"[replay] {f} stmt {m.get('first_stmt')} pose {m.get('first_pose')} done {m.get('done')}")


def hedge_sim(outdir, tier, k="first_pose", n=2, trials=4000):
    """p50 / p90 of the min of n independent draws of a stage (a hedge of n identical calls started together)"""
    import random
    d = os.path.join(outdir, tier)
    v = [((json.load(open(os.path.join(d, f))).get("ms") or {}).get(k)) for f in os.listdir(d) if f.endswith(".json")]
    v = [x for x in v if x is not None]
    if not v:
        return None
    rng = random.Random(1)
    sims = [min(rng.choice(v) for _ in range(n)) for _ in range(trials)]
    return pct(sims, .5), pct(sims, .9)


async def cache_hits(planner, outdir, tier, character, n, log):
    """the same plans again with the cache on: write once (fresh dir), then time the hits"""
    import tempfile
    model, effort = TIERS[tier]
    d = tempfile.mkdtemp(prefix="sem-latency-cache-")
    pl = planner.Planner({"library": False, "semantic": {"model": model, "effort": effort, "cache": True}}, log,
                         cache_dir=os.path.join(d, "plans"))
    src = os.path.join(outdir, tier)
    files = sorted(os.listdir(src))[:n]
    times = []
    import semantic as sem
    for f in files:
        r = json.load(open(os.path.join(src, f)))
        if not r.get("ok") or not r.get("program"):
            continue
        # seed the cache with the bench's program (no door call), as a planner answer would
        targets = sem.resolve_targets(character, None)
        system = pl._sem_system(sem, character, targets)
        import hashlib
        key = planner.cache_key("semantic\n" + character, planner.normalize(r["command"]))
        poses, rep = sem.replay(character, r["program"], targets, 0, 20)
        pl.sem_cache_put(key, {"program": r["program"], "poses": poses, "report": rep, "model": model,
                               "prompt_sha": hashlib.sha256(system.encode()).hexdigest(),
                               "targets": planner.sem_targets_wire(targets), "ts": time.time()})
        sent = []
        t0 = time.monotonic()
        rec = await planner.run_plan(pl, "c", character, r["command"], None, sent.append, mode="semantic", t_recv=t0)
        times.append(round((time.monotonic() - t0) * 1000, 1))
        assert rec["source"] == "cache", rec
    log(f"cache hits ({len(times)}): median {statistics.median(times)} ms, max {max(times)} ms")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sem", default=None, help="an app/ dir holding the semantic package (default: this tree's app/)")
    ap.add_argument("--tiers", default="haiku_nt,sonnet_nt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--character", default="muse")
    ap.add_argument("--suffix", default=None, help="a file whose text is appended to the system prompt")
    ap.add_argument("--cache-hits", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--replay", default=None, help="re-run the recorded streams of --tiers[0] from this bench dir")
    ap.add_argument("--hedge-sim", action="store_true")
    ap.add_argument("--hedge-after", type=float, default=None, help="hedge with a 2nd call of the same tier after S s")
    a = ap.parse_args()
    sys.path.insert(0, APP)                          # this tree's planner first; the semantic package from --sem
    if a.sem:
        sys.path.insert(1, os.path.abspath(a.sem))
    import planner
    tiers = a.tiers.split(",")
    suffix = a.suffix if a.suffix in (None, "auto") else open(a.suffix).read()
    cmds = COMMANDS[:a.n]
    log = lambda m: print(m, flush=True)   # noqa: E731

    async def go():
        if a.replay:
            await replay_tier(planner, os.path.join(a.replay, tiers[0]), os.path.join(a.out, tiers[0]), a.character, log)
        elif not a.report_only:
            for t in tiers:
                await run_tier(planner, t, cmds, os.path.join(a.out, t), a.character, suffix, log, a.hedge_after)
        if a.cache_hits:
            await cache_hits(planner, a.out, tiers[0], a.character, a.cache_hits, log)
    asyncio.run(go())
    rows, table = summarize(a.out, tiers)
    print(table)
    if a.hedge_sim:
        for t in tiers:
            print(f"hedge x2 (simulated, {t}): first_stmt {hedge_sim(a.out, t, 'first_stmt')}, "
                  f"first_pose {hedge_sim(a.out, t, 'first_pose')}, first_token {hedge_sim(a.out, t, 'first_token')}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
