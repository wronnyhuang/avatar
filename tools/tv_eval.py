"""TV: live eval of the TV agent -- run concepts / live-update sequences through the real model (claude_door),
render frames at several moments from the recorded timeline, judge each with Claude vision (sonnet), report.

  python tools/tv_eval.py --model haiku --out runs/tv_eval_haiku [--only id,id] [--conc 3] [--nojudge] [--seq-only]
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
sys.path.insert(0, APP)
import tv_agent  # noqa: E402
from claude_door import door  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REVIEW = None
UPGRADE = None
EXTRA = {}


def montage(imgs, caps, path):
    W, H = 512, 288
    cols = 2
    rows = (len(imgs) + 1) // 2
    M = Image.new("RGB", (W * cols, (H + 18) * rows), (40, 40, 40))
    d = ImageDraw.Draw(M)
    for k, (im, cap) in enumerate(zip(imgs, caps)):
        x, y = (k % cols) * W, (k // cols) * (H + 18)
        M.paste(Image.fromarray(im), (x, y + 18))
        d.text((x + 4, y + 3), cap, fill=(230, 230, 230))
    M.save(path)


async def wait_done(tv, timeout=40):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if tv.cur is not None and not tv.cur["task"].done():
            await asyncio.sleep(0.05)
            continue
        if tv.tap_task is not None and not tv.tap_task.done():
            await asyncio.sleep(0.05)
            continue
        if tv.upgrading():
            await asyncio.sleep(0.05)
            continue
        await asyncio.sleep(0.1)
        if hasattr(tv, "settled") and not tv.settled():
            continue
        return
    return


async def run_concept(c, model, out, sem):
    async with sem:
        cfg = {"model": model, "render": False, "tap": True, "review": REVIEW, "upgrade": UPGRADE}
        cfg.update(EXTRA)
        tv = tv_agent.TVAgent(cfg=cfg, door=door)
        t0 = time.monotonic()
        tv.show(c["concept"])
        await wait_done(tv)
        await asyncio.sleep(0.2)
        await wait_done(tv)                       # the self-review (if any) runs right after the picture is up
        new = [h for h in tv.history if h.get("kind") == "new"]
        rev = [h for h in tv.history if h.get("kind") in ("review", "upgrade", "check")]
        rec = dict(new[-1]) if new else {"status": "none"}
        if rev:
            # the careful version may be a parallel rewrite, a look-and-fix check, or both: the frames
            # are taken after the LAST one that changed the picture
            done = [h for h in rev if h.get("status") in ("reviewed", "upgraded", "checked")]
            last = done[-1] if done else rev[-1]
            rec["review"] = {k: last.get(k) for k in ("status", "ms_done", "lines", "bad", "kind")}
            if done:
                rec["review"]["status"] = "upgraded" if last.get("status") == "checked" else last.get("status")
                rec["review"]["ms_done"] = int((last["t_end"] - t0) * 1000)
                rec["review"]["kinds"] = [h.get("kind") + ":" + str(h.get("status")) for h in rev]
        steps = [{"say": None, "rec": rec, "t_done": time.monotonic(), "t_start": t0}]
        for say in c.get("follow", []):
            await asyncio.sleep(1.0)
            steps[-1]["t_end"] = time.monotonic()
            t_s = time.monotonic()
            n0 = len(tv.history)
            tv.hear("you", say)
            await asyncio.sleep(0.1)
            await wait_done(tv)
            r2 = dict(tv.history[-1]) if len(tv.history) > n0 else {"status": "no_call"}
            steps.append({"say": say, "rec": r2, "t_done": time.monotonic(), "t_start": t_s})
        await asyncio.sleep(0)
        res = {"id": c["id"], "subject": c["subject"], "concept": c["concept"], "model": model, "steps": []}
        prog = "\n".join(l for _, l in tv.timeline)
        open(os.path.join(out, f"{c['id']}.tvl"), "w").write(prog)
        # frames
        for k, st in enumerate(steps):
            r = st["rec"]
            if k == 0:
                fd = (r.get("ms_first_draw") or r.get("ms_done") or 0) / 1000.0
                dn = (r.get("ms_done") or 0) / 1000.0
                times = [fd + 0.3, dn + 0.3, dn + 2.5, dn + 5.5]
                caps = [f"first draw +0.3s ({fd:.1f}s)", f"complete +0.3s ({dn:.1f}s)", "complete +2.5s", "complete +5.5s"]
                rv = r.get("review")
                if rv and rv.get("status") in ("reviewed", "upgraded"):
                    rd = (rv.get("ms_done") or 0) / 1000.0 if rv.get("kinds") else \
                        (dn if rv.get("status") == "reviewed" else 0.0) + (rv.get("ms_done") or 0) / 1000.0
                    times = [fd + 0.3, dn + 0.3, rd + 0.8, rd + 3.5]
                    nm = "self-review" if rv.get("status") == "reviewed" else "careful version"
                    caps = [f"first draw +0.3s ({fd:.1f}s)", f"complete +0.3s ({dn:.1f}s)", f"after {nm} +0.8s ({rd:.1f}s)",
                            f"after {nm} +3.5s"]
            else:
                dn = st["t_done"] - t0
                times = [dn + 0.2, dn + 1.2, dn + 3.0, dn + 5.0]
                r = dict(r)
                r["ms_done"] = int((st["t_done"] - st["t_start"]) * 1000)          # the user's words -> picture updated
                caps = [f"after '{st['say'][:40]}' +0.2s", "+1.2s", "+3s", "+5s"]
            tl = [e for e in tv.timeline if e[0] < st.get("t_end", 1e18)]     # this step's picture, not the next one's
            imgs = tv_agent.replay(tl, times, t_origin=t0)
            path = os.path.join(out, f"{c['id']}_{k}.png")
            montage(imgs, caps, path)
            res["steps"].append({"say": st["say"], "png": path, "status": r.get("status"), "lines": r.get("lines"), "review": r.get("review"),
                                 "bad": r.get("bad"), "bad_lines": r.get("bad_lines"),
                                 "ms_first_token": r.get("ms_first_token"), "ms_first_draw": r.get("ms_first_draw"),
                                 "ms_done": r.get("ms_done"), "error": r.get("error")})
        await tv.close()
        print(f"  {c['id']}: " + " | ".join(f"{s['status']} fd={s['ms_first_draw']} done={s['ms_done']} bad={s['bad']}" for s in res["steps"]), flush=True)
        return res


JUDGE = """You are grading an animated teaching illustration shown on a small wall TV (512x288) beside a teacher.
The image {png} is a 2x2 montage of frames at the moments in their captions.
{task}
Look carefully at the frames (open the image with the Read tool). Grade strictly:
- correct: is everything drawn scientifically/mathematically right (geometry, directions, labels, formulas, values, ray paths: e.g. rays through a converging lens really meet at the image point; parallel rays meet at F)? 0 wrong, 1 minor flaw, 2 right
- clear: would a learner get the idea at a glance? 0/1/2
- animated: does it move meaningfully across the frames (tracing, motion that shows the process), not just static? 0/1/2
- legible: text readable, nothing important overlapping or cut off, uses the screen well at this size? 0/1/2
PASS only if correct = 2 and clear >= 1 and legible >= 1 (and, for an update, the change he asked for is visibly made).
Reply with ONLY one JSON object: {{"correct":n,"clear":n,"animated":n,"legible":n,"pass":true|false,"issues":"<one short sentence>"}}"""


def judge(png, task):
    prompt = JUDGE.format(png=png, task=task)
    for attempt in range(2):
        try:
            p = subprocess.run([os.environ.get("AVATAR_CLAUDE_BIN") or "claude", "-p", "--model", "sonnet", "--tools", "Read", "--permission-mode", "dontAsk",
                                "--setting-sources", "", "--no-session-persistence", prompt],
                               capture_output=True, text=True, timeout=180, cwd=os.path.dirname(png))
            txt = p.stdout.strip()
            j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            return j
        except Exception as e:  # noqa: BLE001
            err = str(e)[:200]
    return {"pass": False, "issues": "judge failed: " + err, "error": True}


async def judge_all(results, conc=4):
    sem = asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()

    async def one(res, k, st):
        if k == 0:
            task = f"It should illustrate: \"{res['concept']}\"."
            rv = st.get("review") or {}
            if rv.get("status") == "upgraded":
                task += (" The first two frames are a quick sketch that is then REPLACED by a careful version (frames 3 "
                         "and 4, captioned 'after careful version'). Grade the careful version (frames 3-4); do not "
                         "count the change from the sketch as an inconsistency, but a wrong sketch is a minor flaw.")
        else:
            hist = " -> ".join(f"\"{x['say']}\"" for x in res["steps"][1:k])
            task = (f"The picture first illustrated \"{res['concept']}\"; " + (f"then, in order, the user said {hist}; " if hist else "")
                    + f"the user then said \"{st['say']}\". Earlier changes stay in force (they accumulate). The frames are "
                    f"right after this last update. Judge whether the picture now shows what the user asked (or correctly stays the "
                    f"same if the user's words needed no change) and is still correct.")
        async with sem:
            st["judge"] = await loop.run_in_executor(None, judge, st["png"], task)
    await asyncio.gather(*[one(r, k, s) for r in results for k, s in enumerate(r["steps"])])


def report(results, path):
    rows = []
    for r in results:
        for k, s in enumerate(r["steps"]):
            j = s.get("judge") or {}
            rows.append((r["id"], r["subject"], k, s["say"] or r["concept"], s["status"], s["ms_first_draw"], s["ms_done"],
                         j.get("pass"), j.get("correct"), j.get("clear"), j.get("animated"), j.get("legible"), j.get("issues")))
    n = [x for x in rows if x[7] is not None]
    passed = sum(1 for x in n if x[7])
    # lenient = no outright error and readable (correct >= 1, clear >= 1, legible >= 1)
    len_ok = lambda x: (x[8] or 0) >= 1 and (x[9] or 0) >= 1 and (x[11] or 0) >= 1
    lenient = sum(1 for x in n if len_ok(x))
    subj = {}
    for x in n:
        s_ = subj.setdefault(x[1], [0, 0, 0])
        s_[0] += 1
        s_[1] += bool(x[7])
        s_[2] += bool(len_ok(x))
    first = sorted(x[5] for x in rows if x[2] == 0 and x[5])
    done = sorted(x[6] for x in rows if x[2] == 0 and x[6])
    upd = sorted(x[6] for x in rows if x[2] > 0 and x[6])
    car = sorted((s["review"] or {}).get("ms_done") for r in results for k, s in enumerate(r["steps"])
                 if k == 0 and (s.get("review") or {}).get("status") in ("upgraded", "reviewed") and (s["review"] or {}).get("ms_done"))
    pct = lambda v, q: v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else None
    lines = [f"# TV eval {results[0]['model'] if results else ''}", "",
             f"pass {passed}/{len(n)} ({100 * passed / max(1, len(n)):.0f}%) -- new scenes {sum(1 for x in n if x[2] == 0 and x[7])}/{sum(1 for x in n if x[2] == 0)}, "
             f"updates {sum(1 for x in n if x[2] > 0 and x[7])}/{sum(1 for x in n if x[2] > 0)}",
             f"latency new scene: first drawing p50 {pct(first, .5)} ms p90 {pct(first, .9)} ms; complete p50 {pct(done, .5)} p90 {pct(done, .9)} ms",
             f"latency update: complete p50 {pct(upd, .5)} p90 {pct(upd, .9)} ms; careful version landed ({len(car)}): p50 {pct(car, .5)} p90 {pct(car, .9)} ms",
             f"lenient {lenient}/{len(n)} ({100 * lenient / max(1, len(n)):.0f}%)",
             "per subject (strict/lenient/n): " + ", ".join(f"{k} {v[1]}/{v[2]}/{v[0]}" for k, v in sorted(subj.items())), "",
             "| id | step | words | status | first ms | done ms | pass | c/cl/an/lg | issues |", "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x[0]} | {x[2]} | {str(x[3])[:50]} | {x[4]} | {x[5]} | {x[6]} | {'PASS' if x[7] else 'fail'} | "
                     f"{x[8]}/{x[9]}/{x[10]}/{x[11]} | {str(x[12])[:140]} |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--nojudge", action="store_true")
    ap.add_argument("--seq-only", action="store_true")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--set", default="tv_devset", help="eval/<set>.json: tv_devset (tuning) or tv_holdout (never tuned on)")
    ap.add_argument("--upgrade", default="", help="careful-version model drawn in parallel and swapped in (sonnet)")
    ap.add_argument("--review", default="", help="self-review model after each new picture (haiku|sonnet)")
    ap.add_argument("--rejudge", action="store_true", help="judge an existing OUT/results.json again (no model runs)")
    ap.add_argument("--cfg", default="", help="extra TVAgent cfg (JSON), e.g. {\"careful\":\"edit\"}")
    a = ap.parse_args()
    ap2 = a
    EXTRA.update(json.loads(a.cfg) if a.cfg else {})
    global REVIEW, UPGRADE
    REVIEW = a.review or None
    UPGRADE = ap2.upgrade or None
    os.makedirs(a.out, exist_ok=True)
    ds = json.load(open(os.path.join(os.path.dirname(HERE), "eval", a.set + ".json")))
    items = ([] if a.seq_only else ds["concepts"]) + ([] if a.no_seq else ds["sequences"])
    if a.only:
        keep = set(a.only.split(","))
        items = [c for c in ds["concepts"] + ds["sequences"] if c["id"] in keep]
    if a.rejudge:
        results = json.load(open(os.path.join(a.out, "results.json")))
        await judge_all(results)
        json.dump(results, open(os.path.join(a.out, "results_rejudged.json"), "w"), indent=1)
        report(results, os.path.join(a.out, "REPORT_rejudged.md"))
        return
    door.prewarm(a.model, tv_agent.SYSTEM, n=min(6, a.conc * 2))
    if UPGRADE and UPGRADE != a.model:
        door.prewarm(UPGRADE, tv_agent.SYSTEM, n=a.conc)
    if EXTRA.get("careful") in ("edit", "parallel+check") and UPGRADE != "sonnet":
        door.prewarm(EXTRA.get("check_model", "sonnet"), tv_agent.SYSTEM, n=a.conc)
    if REVIEW and REVIEW != a.model:
        door.prewarm(REVIEW, tv_agent.SYSTEM, n=a.conc)
    await asyncio.sleep(5)
    sem = asyncio.Semaphore(a.conc)
    results = await asyncio.gather(*[run_concept(c, a.model, a.out, sem) for c in items])
    await door.close()
    if not a.nojudge:
        await judge_all(results)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    report(results, os.path.join(a.out, "REPORT.md"))


if __name__ == "__main__":
    asyncio.run(main())
