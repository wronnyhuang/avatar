"""TV: live FOLLOW-UP eval of the TV agent. Each sequence: a picture is drawn for a concept, then the user says
two follow-ups through the live conversation tap (TVAgent.hear, exactly as in a session). For every follow-up we keep
the picture as it was when he spoke (BEFORE) and the picture after the agent finished (AFTER, three moments), measure
how fast the picture visibly changed, and a Claude-vision judge compares BEFORE with AFTER: did the DRAWING change the
way he asked, and is it still right?

  python tools/tv_follow_eval.py --out DIR [--split dev|hold|all] [--only id,id] [--conc 3] [--upgrade sonnet]
  python tools/tv_follow_eval.py --out DIR --rejudge

Metrics per follow-up: strict PASS = changed 2 and correct 2 and legible >= 1; lenient = changed >= 1 and correct >= 1;
ms_visible = the user's words (end of utterance) -> first frame whose drawing differs from BEFORE (geometry, not captions);
ms_done = -> the agent finished (incl. any escalation). The frame diff is measured on replays of the recorded timeline.
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
from tvdsl.lang import Doc, parse_line  # noqa: E402
from tvdsl.render import Renderer  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

CFG = {}


def doc_at(timeline, now):
    """the Doc as it stood at wall time `now` (same rules as tv_agent.replay)"""
    doc = Doc()
    doc.empty = True
    for mono, line in timeline:
        if mono > now:
            break
        if line == "clear":
            doc = Doc()
            doc.empty = True
            continue
        for st in parse_line(line):
            if st.op == "scene" and not doc.empty:
                doc = Doc()
            doc.apply(st, mono)
    return doc


def settled(tv):
    f = getattr(tv, "settled", None)
    if f is not None:
        return f()
    if tv.cur is not None and not tv.cur["task"].done():
        return False
    if tv.tap_task is not None and not tv.tap_task.done():
        return False
    if tv.upgrading():
        return False
    return True


async def wait_done(tv, timeout=45):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not settled(tv):
            await asyncio.sleep(0.05)
            continue
        await asyncio.sleep(0.15)
        if settled(tv):
            return
    return


def montage(imgs, caps, path):
    W, H = 512, 288
    M = Image.new("RGB", (W * 2, (H + 18) * 2), (40, 40, 40))
    d = ImageDraw.Draw(M)
    for k, (im, cap) in enumerate(zip(imgs, caps)):
        x, y = (k % 2) * W, (k // 2) * (H + 18)
        M.paste(Image.fromarray(im), (x, y + 18))
        d.text((x + 4, y + 3), cap, fill=(255, 220, 120) if k == 0 else (230, 230, 230))
    M.save(path)


def visible_ms(timeline, t_say, t_end, R):
    """first moment after the user's words whose DRAWING (no texts) differs from the drawing as it would have been without
    any change (the BEFORE doc rendered at the same moment, so ordinary animation does not count)"""
    before = doc_at(timeline, t_say)
    T = t_say + 0.1
    while T <= t_end:
        after = doc_at(timeline, T)
        a = R.render(before, T, texts=False).astype(np.int16).sum(axis=2)
        b = R.render(after, T, texts=False).astype(np.int16).sum(axis=2)
        if np.count_nonzero(np.abs(a - b) > 120) > 0.004 * a.size:
            return int((T - t_say) * 1000)
        T += 0.1
    return None


async def run_seq(c, out, sem):
    async with sem:
        cfg = {"model": "haiku", "render": False, "tap": True, "upgrade": CFG.get("upgrade")}
        cfg.update(CFG.get("extra") or {})
        tv = tv_agent.TVAgent(cfg=cfg, door=door)
        t0 = time.monotonic()
        tv.show(c["concept"])
        await wait_done(tv)
        await asyncio.sleep(1.5)
        steps = []
        for say in c["follow"]:
            t_say = time.monotonic()
            n0 = len(tv.history)
            tv.hear("you", say)
            await asyncio.sleep(0.1)
            await wait_done(tv)
            t_done = time.monotonic()
            recs = [dict(h) for h in tv.history[n0:]]
            steps.append({"say": say, "t_say": t_say, "t_done": t_done, "recs": recs})
            await asyncio.sleep(4.0)
        t_fin = time.monotonic()
        await tv.close()
        tl = list(tv.timeline)
        open(os.path.join(out, f"{c['id']}.tvl"), "w").write(
            "\n".join(f"{m - t0:7.2f} {l}" for m, l in tl))
        R = Renderer(512, 288)
        Rv = Renderer(256, 144, glow=0.0, autofit=False)
        res = {"id": c["id"], "subject": c["subject"], "split": c.get("split"), "concept": c["concept"], "steps": []}
        # the first picture (for reference)
        fp = os.path.join(out, f"{c['id']}_0.png")
        t_first_end = steps[0]["t_say"] if steps else t_fin
        imgs = tv_agent.replay([e for e in tl if e[0] < t_first_end], [t_first_end - t0 - 0.05], t_origin=t0)
        Image.fromarray(imgs[0]).save(fp)
        for k, st in enumerate(steps):
            nxt = steps[k + 1]["t_say"] if k + 1 < len(steps) else t_fin
            tl_k = [e for e in tl if e[0] < nxt]
            T_say = st["t_say"] - t0
            T_done = st["t_done"] - t0
            times = [T_say - 0.05, T_done + 0.3, T_done + 1.5, min(T_done + 3.5, nxt - t0 - 0.05)]
            imgs = tv_agent.replay(tl_k, times, t_origin=t0)
            path = os.path.join(out, f"{c['id']}_{k + 1}.png")
            montage(imgs, [f"BEFORE: as it was when he said it", f"AFTER: +{T_done - T_say:.1f}s (agent done) +0.3s",
                           "AFTER +1.5s", "AFTER +3.5s"], path)
            vis = visible_ms(tl_k, st["t_say"], st["t_done"] + 1.0, Rv)
            res["steps"].append({"say": st["say"], "png": path, "ms_done": int((st["t_done"] - st["t_say"]) * 1000),
                                 "ms_visible": vis,
                                 "recs": [{k2: r.get(k2) for k2 in ("id", "kind", "source", "model", "status", "lines",
                                                                     "ms_first_token", "ms_first_draw", "ms_done", "bad", "error", "check", "fix", "winner", "winner_model", "correct", "result", "lint",
                                                                     "change")} for r in st["recs"]]})
        print(f"  {c['id']}: " + " | ".join(f"vis={s['ms_visible']} done={s['ms_done']} "
                                            + ",".join(f"{r['kind']}:{r['status']}" for r in s["recs"]) for s in res["steps"]),
              flush=True)
        return res


JUDGE = """You are grading a LIVE UPDATE of an animated teaching illustration on a small wall TV (512x288).
The image {png} is a 2x2 montage. TOP-LEFT (yellow caption "BEFORE") = the TV at the moment the user spoke. The other
three = the TV AFTER the update, at the moments in their captions.
The picture illustrates: "{concept}".{hist}
The user now said: "{say}"
Open the image with the Read tool and compare BEFORE with AFTER. Grade strictly:
- changed: did the DRAWING itself (shapes, positions, curves, rays, particles -- not only captions or formula text)
  change the way he asked? 0 = no visible change of the drawing, or only text changed, or the wrong thing changed;
  1 = partly / too little / ambiguous; 2 = clearly the change he asked for
- correct: is the AFTER picture scientifically/mathematically right, with labels and numbers consistent with what is
  drawn? 0 wrong, 1 minor flaw, 2 right
- kept: does it keep what should stay (the rest of the picture, earlier changes) instead of an unrelated redraw? 0/1/2
- legible: text readable, nothing important overlapping or cut off? 0/1/2
Reply with ONLY one JSON object:
{{"changed":n,"correct":n,"kept":n,"legible":n,"issues":"<one short sentence>"}}"""


def judge(png, concept, hist, say):
    h = (" Earlier, in order, the user said " + " -> ".join(f'"{x}"' for x in hist)
         + " (those changes stay in force).") if hist else ""
    prompt = JUDGE.format(png=png, concept=concept, hist=h, say=say)
    err = ""
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
    return {"issues": "judge failed: " + err, "error": True}


def verdict(j):
    ch, co, lg = j.get("changed", 0) or 0, j.get("correct", 0) or 0, j.get("legible", 0) or 0
    return ch >= 2 and co >= 2 and lg >= 1, ch >= 1 and co >= 1


async def judge_all(results, conc=4):
    sem = asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()

    async def one(res, k, st):
        hist = [s["say"] for s in res["steps"][:k]]
        async with sem:
            st["judge"] = await loop.run_in_executor(None, judge, st["png"], res["concept"], hist, st["say"])
    await asyncio.gather(*[one(r, k, s) for r in results for k, s in enumerate(r["steps"])])


def report(results, path, title):
    rows = []
    for r in results:
        for k, s in enumerate(r["steps"]):
            j = s.get("judge") or {}
            strict, lenient = verdict(j)
            rows.append((r, k, s, j, strict, lenient))
    pct = lambda v, q: v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else None

    def summ(sel, name):
        if not sel:
            return f"{name}: -"
        st = sum(1 for x in sel if x[4])
        le = sum(1 for x in sel if x[5])
        return f"{name}: strict {st}/{len(sel)} ({100 * st / len(sel):.0f}%), lenient {le}/{len(sel)} ({100 * le / len(sel):.0f}%)"
    vis = sorted(x[2]["ms_visible"] for x in rows if x[2]["ms_visible"] is not None)
    done = sorted(x[2]["ms_done"] for x in rows)
    nochange = sum(1 for x in rows if x[2]["ms_visible"] is None)
    lines = [f"# TV follow-up eval {title}", "",
             summ(rows, "all"), summ([x for x in rows if x[0].get("split") == "dev"], "dev"),
             summ([x for x in rows if x[0].get("split") == "hold"], "hold"),
             f"drawing visibly changed (frame diff): {len(rows) - nochange}/{len(rows)}; words->visible change p50 {pct(vis, .5)} "
             f"p90 {pct(vis, .9)} ms; words->agent done p50 {pct(done, .5)} p90 {pct(done, .9)} ms", "",
             "| id | split | # | words | visible ms | done ms | requests | ch/co/kp/lg | strict | lenient | issues |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, k, s, j, strict, lenient in rows:
        reqs = ",".join(f"{q['kind']}:{q['status']}" for q in s["recs"])
        lines.append(f"| {r['id']} | {r.get('split')} | {k + 1} | {s['say'][:45]} | {s['ms_visible']} | {s['ms_done']} | {reqs} | "
                     f"{j.get('changed')}/{j.get('correct')}/{j.get('kept')}/{j.get('legible')} | {'PASS' if strict else 'fail'} | "
                     f"{'ok' if lenient else '-'} | {str(j.get('issues'))[:150]} |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--upgrade", default="sonnet")
    ap.add_argument("--cfg", default="", help="extra TVAgent cfg as JSON")
    ap.add_argument("--nojudge", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    a = ap.parse_args()
    CFG["upgrade"] = None if a.upgrade in ("", "none") else a.upgrade
    CFG["extra"] = json.loads(a.cfg) if a.cfg else {}
    os.makedirs(a.out, exist_ok=True)
    if a.rejudge:
        results = json.load(open(os.path.join(a.out, "results.json")))
        await judge_all(results)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
        report(results, os.path.join(a.out, "REPORT.md"), a.out)
        return
    ds = json.load(open(os.path.join(os.path.dirname(HERE), "eval", "tv_followups.json")))
    items = [c for c in ds["sequences"] if a.split == "all" or c.get("split") == a.split]
    if a.only:
        keep = set(a.only.split(","))
        items = [c for c in ds["sequences"] if c["id"] in keep]
    door.prewarm("haiku", tv_agent.SYSTEM, n=min(6, a.conc * 2))
    if CFG["upgrade"]:
        door.prewarm(CFG["upgrade"], tv_agent.SYSTEM, n=a.conc)
    await asyncio.sleep(5)
    sem = asyncio.Semaphore(a.conc)
    results = await asyncio.gather(*[run_seq(c, a.out, sem) for c in items])
    await door.close()
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    if not a.nojudge:
        await judge_all(results)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    report(results, os.path.join(a.out, "REPORT.md"), a.out)


if __name__ == "__main__":
    asyncio.run(main())
