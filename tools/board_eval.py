#!/usr/bin/env python3
"""Board brain eval: run the board brain live on a dev set, render each board, judge it with a Claude vision
judge (claude -p --model sonnet --tools Read), report pass rates + latency.

  board_eval.py --cases eval/board_dev.json --model haiku --out DIR [--only id,id] [--jobs 4] [--no-judge]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.path.insert(0, HERE)

from board_brain import BoardSession  # noqa: E402
from board_preview import render, reveal_frames, save_gif, render_pages, storyboard  # noqa: E402

JUDGE = """You are grading a teaching chalkboard produced by an AI teacher. Read the image file {png} (the board, 2:1, chalk on slate; if it is a tall image it shows several snapshots of the same board over time: the teacher erased parts to make room, which is fine).
The darker strip across the TOP third is out of the teacher's reach (it cannot write there): that it stays empty is by design, not a layout fault; judge the layout of the lighter writing band.
The student asked: "{request}"
{setup}The teacher wrote these items in this order (kind | what it says before writing | source):
{items}

Grade strictly, each 1 (yes) or 0 (no):
- legible: every mark the new request added is readable handwriting/drawing at this size (nothing garbled, no stray boxes or symbols, no overlapping text).
- correct: the content is factually and mathematically correct (check every step and number; a diagram/figure shows the right relations).
- layout: sensible placement and size: no overlaps or clipping at the edges, reading order clear, not cramped, not tiny.
- say: the spoken lines make sense as what a teacher says right before writing each item (short, natural, in order).
- complete: the lesson actually answers the request (not missing the key step/answer; follow-ups did what was asked). Judge over the whole lesson, all snapshots: the student watched every item being written, so work that was later erased to make room still counts as shown.
Reply with ONLY a JSON object: {{"legible":0|1,"correct":0|1,"layout":0|1,"say":0|1,"complete":0|1,"notes":"<= 25 words on the main problem"}}"""


def _item_lines(items):
    out = []
    for it in items:
        src = (it.get("text") or "")
        if it["kind"] == "erase":
            src = "erase " + ",".join(it.get("target") or [])
        out.append("%s | %s | %s" % (it["kind"], it.get("say_before") or "", src[:200]))
    return "\n".join(out)


async def judge(png, case, new_items, setup_items):
    setup = ""
    if case.get("setup"):
        setup = "Before this request the board already had (from earlier requests %s):\n%s\n" % (
            json.dumps(case["setup"]), _item_lines(setup_items))
    prompt = JUDGE.format(png=png, request=case["request"], setup=setup, items=_item_lines(new_items))
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        os.environ.get("AVATAR_CLAUDE_BIN") or "claude", "-p", "--model", "sonnet", "--tools", "Read", "--permission-mode", "dontAsk",
        "--output-format", "text", "--no-session-persistence", "--add-dir", os.path.dirname(os.path.abspath(png)),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(prompt.encode()), 240)
    except asyncio.TimeoutError:
        proc.kill()
        return {"error": "judge timeout"}
    txt = out.decode(errors="replace")
    m = re.search(r"\{[^{}]*\"legible\"[^{}]*\}", txt, re.S)
    if not m:
        return {"error": "no json", "raw": txt[-300:]}
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return {"error": "bad json", "raw": m.group(0)}
    v["judge_s"] = round(time.time() - t0, 1)
    return v


async def run_case(case, model, outdir, do_judge, sem, gif=False):
    async with sem:
        sess = BoardSession(seed=hash(case["id"]) & 0xFFFF)
        all_items, setup_items = [], []
        ctx = [{"role": r, "text": t} for r, t in case.get("context", [])]
        setup_raw = []
        for req in case.get("setup", []):
            sst = {}
            got = await sess.plan(ctx, req, model=model, stats=sst)
            setup_raw.append(sst.get("raw", ""))
            setup_items += got
            all_items += got
            ctx = ctx + [{"role": "user", "text": req}, {"role": "assistant", "text": "(wrote it on the board)"}]
        stats = {}
        t0 = time.time()
        new = await sess.plan(ctx, case["request"], model=model, stats=stats)
        all_items += new
        png = os.path.join(outdir, case["id"] + ".png")
        render(all_items).save(png)
        pages = render_pages(all_items)
        judge_png = png
        if len(pages) > 1:
            judge_png = os.path.join(outdir, case["id"] + "_story.png")
            storyboard(pages).save(judge_png)
        if gif:
            save_gif(reveal_frames(all_items), os.path.join(outdir, case["id"] + ".gif"))
        with open(os.path.join(outdir, case["id"] + ".json"), "w") as f:
            json.dump({"case": case, "items": all_items, "n_setup": len(setup_items), "setup_raw": setup_raw,
                       "stats": {k: v for k, v in stats.items() if k != "board_state"}}, f)
        res = {"id": case["id"], "model": model, "ms_first_item": stats.get("ms_first_item"),
               "ms_total": stats.get("ms_total"), "ms_first_token": stats.get("ms_first_token"),
               "n_items": len(new), "errors": stats.get("errors"), "ok": stats.get("ok"),
               "est_s": round(sum(i.get("est_s", 0) for i in new), 1),
               "warn": sorted({w for i in new for w in (i.get("warnings") or [])})[:4]}
        if do_judge:
            res["judge"] = await judge(judge_png, case, new, setup_items)
        print(json.dumps(res), flush=True)
        return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "..", "eval", "board_dev.json"))
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    cases = json.load(open(a.cases))
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in want]
    os.makedirs(a.out, exist_ok=True)
    import board_brain
    board_brain.prewarm(a.model)
    await asyncio.sleep(2.0)
    sem = asyncio.Semaphore(a.jobs)
    res = await asyncio.gather(*[run_case(c, a.model, a.out, not a.no_judge, sem, a.gif) for c in cases])
    try:
        await board_brain.get_door().close()
    except Exception:  # noqa: BLE001
        pass
    with open(os.path.join(a.out, "results.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    summarize(res)


def replay(src_dir, outdir, do_judge=True):
    """Re-run placement/rendering with the CURRENT code on the model outputs stored in src_dir (no model calls),
    then judge. Lets layout/font changes be evaluated on identical model text."""
    from board_brain import BoardSession, _Stream, Feeder
    os.makedirs(outdir, exist_ok=True)
    res = []
    for l in open(os.path.join(src_dir, "results.jsonl")):
        r = json.loads(l)
        d = json.load(open(os.path.join(src_dir, r["id"] + ".json")))
        case = d["case"]
        sess = BoardSession(seed=hash(case["id"]) & 0xFFFF)
        b = sess.board
        all_items = []
        n_setup = 0
        lay_first = lay_tot = None
        for raw in (d.get("setup_raw") or []) + [d["stats"]["raw"]]:
            b.new_plan()
            st = _Stream()
            got = []
            fd = Feeder(b)
            main_raw = raw is d["stats"]["raw"]
            tl0 = time.perf_counter()
            for op in st.feed(raw + "\n") + st.end():
                try:
                    got += fd.push(op)
                except Exception as e:  # noqa: BLE001
                    print("replay error", case["id"], e)
                if main_raw and got and lay_first is None:
                    lay_first = (time.perf_counter() - tl0) * 1000
            try:
                got += fd.flush()
            except Exception as e:  # noqa: BLE001
                print("replay batch error", case["id"], e)
            if main_raw:
                lay_tot = (time.perf_counter() - tl0) * 1000
                if lay_first is None and got:
                    lay_first = lay_tot
            all_items += got
            if raw is not d["stats"]["raw"]:
                n_setup = len(all_items)
        if case.get("setup") and not d.get("setup_raw"):
            continue
        png = os.path.join(outdir, case["id"] + ".png")
        render(all_items).save(png)
        pages = render_pages(all_items)
        if len(pages) > 1:
            storyboard(pages).save(os.path.join(outdir, case["id"] + "_story.png"))
        with open(os.path.join(outdir, case["id"] + ".json"), "w") as f:
            json.dump({"case": case, "items": all_items, "n_setup": n_setup, "setup_raw": d.get("setup_raw"),
                       "stats": d["stats"]}, f)
        r2 = {k: r[k] for k in r if k != "judge"}
        r2["layout_first_ms"] = round(lay_first or 0, 1)
        r2["layout_ms"] = round(lay_tot or 0, 1)
        res.append(r2)
    with open(os.path.join(outdir, "results.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    if do_judge:
        rejudge(outdir)
    else:
        summarize(res)


def rejudge(outdir):
    """Re-run only the judge over an existing run directory (cases' json + png)."""
    async def one(r, sem):
        async with sem:
            d = json.load(open(os.path.join(outdir, r["id"] + ".json")))
            n_setup = d["n_setup"]
            png = os.path.join(outdir, r["id"] + "_story.png")
            if not os.path.exists(png):
                png = os.path.join(outdir, r["id"] + ".png")
            r["judge"] = await judge(png, d["case"], d["items"][n_setup:], d["items"][:n_setup])
            print(json.dumps({"id": r["id"], "judge": r["judge"]}), flush=True)
            return r
    res = [json.loads(l) for l in open(os.path.join(outdir, "results.jsonl"))]
    async def go():
        sem = asyncio.Semaphore(6)
        return await asyncio.gather(*[one(r, sem) for r in res])
    res = asyncio.run(go())
    with open(os.path.join(outdir, "results.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    summarize(res)


def summarize(res):
    n = len(res)
    keys = ["legible", "correct", "layout", "say", "complete"]
    judged = [r for r in res if isinstance(r.get("judge"), dict) and "legible" in r["judge"]]
    line = {"n": n, "judged": len(judged)}
    for k in keys:
        if judged:
            line[k] = round(sum(int(r["judge"].get(k, 0)) for r in judged) / len(judged), 3)
    if judged:
        line["all5"] = round(sum(all(int(r["judge"].get(k, 0)) for k in keys) for r in judged) / len(judged), 3)
    fi = sorted(r["ms_first_item"] for r in res if r.get("ms_first_item"))
    tt = sorted(r["ms_total"] for r in res if r.get("ms_total"))
    if fi:
        line["first_item_ms_p50"] = fi[len(fi) // 2]
        line["first_item_ms_p90"] = fi[int(len(fi) * 0.9) - 1 if len(fi) > 1 else 0]
    if tt:
        line["total_ms_p50"] = tt[len(tt) // 2]
        line["total_ms_p90"] = tt[int(len(tt) * 0.9) - 1 if len(tt) > 1 else 0]
    lf = sorted(r["layout_first_ms"] for r in res if "layout_first_ms" in r)
    lt = sorted(r["layout_ms"] for r in res if "layout_ms" in r)
    if lt:
        line["layout_first_ms_p50"], line["layout_first_ms_max"] = lf[len(lf) // 2], lf[-1]
        line["layout_ms_p50"], line["layout_ms_max"] = lt[len(lt) // 2], lt[-1]
    print("SUMMARY " + json.dumps(line))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--replay":
        replay(sys.argv[2], sys.argv[3], "--no-judge" not in sys.argv)
    elif len(sys.argv) == 3 and sys.argv[1] == "--rejudge":
        rejudge(sys.argv[2])
    else:
        asyncio.run(main())
