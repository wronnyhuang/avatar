#!/usr/bin/env python3
"""attention_judge_bench.py — which vision model judges attention candidates (perception). REAL vendor calls.

    python3 -I tests/attention_judge_bench.py [--models a,b,...] [--reps 1] [--out FILE] [--spend FILE] [--cap 1.0]

Items: 1-2 frames ("before" ~1 s earlier, "now") cut from the eval clips at their annotated cue times, with the
candidate kinds / summary the page's detector would send, the standing rules in force and the expected worth_reply
(True / False; None = either is fine, scored only for latency). The prompt, payload and parsing are app/attention_judge.py's
own (AttentionJudge.build_messages / payload / parse_verdict) — the bench measures what the server would run.
Scores per model: worth_reply accuracy on the scored items, the rule items alone, invalid answers, latency p50/p90 (ms,
wall clock of the HTTP call), USD per call (OpenRouter usage.cost). Spend is appended to --spend (if given) as JSON lines.
"""
import argparse, base64, concurrent.futures as cf, json, os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import attention_judge as AJ  # noqa: E402

CLIPS = os.environ.get("AVATAR_EVAL_CLIPS", os.path.join(HERE, "..", "data", "eval_clips"))
T = os.path.join(tempfile.gettempdir(), "avatar-tests")
S = lambda **k: dict({"speaking": False, "avatar_speaking": False, "face": True, "attending": True}, **k)  # noqa: E731
COFFEE = "don't comment on my coffee"
CAT = "tell me if my cat walks in"
WAVE_NO = "don't react when I wave, I do it all the time"
# (id, clip, t_before, t_now, kinds, summary, rules, expect)
ITEMS = [
    ("enter_person", "appear_bedroom_woman_goes_to_bed", 1.8, 3.2, ["scene_change"], S(face=False, attending=False), [], True),
    ("yawn", "boredom_yawn_office_man", 1.2, 4.0, ["expression_spike"], S(expression="surprised"), [], True),
    ("turns_to_cam", "silence_waiting_woman_turns_smiles", 1.0, 3.2, ["gaze_to_camera", "new_face"], S(), [], True),
    ("hands_at_lens", "startle_vcd_hands_at_lens_woman", 4.6, 5.6, ["motion_burst", "scene_change"], S(), [], True),
    ("lens_covered", "startle_vcd_hands_at_lens_woman", 7.2, 8.2, ["scene_change"], S(face=False, attending=False), [], True),
    ("wave_hello", "wave_vcd_bald_man", 1.8, 3.2, ["gesture", "motion_burst"], S(), [], True),
    ("wave_far", "wave_silent", 1.5, 3.0, ["motion_burst"], S(face=False, attending=False), [], True),
    ("watch_shown", "watch_object_silent", 2.0, 3.3, ["new_object"], S(attending=False), [], None),
    ("mug_raise", "object_mixkit_mug_pink_tee_man", 2.4, 4.5, ["new_object"], S(attending=False), [], None),
    ("lean_in", "approach_leave_vcd_lean_in_man", 4.8, 6.2, ["distance_change"], S(), [], None),
    ("walks_room", "move_room_walk_phone_woman", 9.2, 11.0, ["scene_change"], S(face=False, attending=False), [], None),
    # normal conversation frames (the detector's typical false candidates): expect quiet
    ("talk_a", "sl_S0648_0372_P0299A", 10.0, 11.0, ["motion_burst"], S(speaking=True), [], False),
    ("talk_b", "sl_S2025_1080_P1281A", 20.0, 21.0, ["expression_spike"], S(speaking=True, expression="happy"), [], False),
    ("listen_c", "sl_S2090_1011_P1346A", 15.0, 16.0, ["motion_burst"], S(), [], False),
    ("listen_d", "sl_S0152_0371_P0216", 8.0, 9.0, ["gaze_to_camera"], S(), [], False),
    ("talk_e", "talk_john_intro", 5.0, 6.0, ["motion_burst"], S(speaking=True), [], False),
    # standing rules
    ("mug_coffee_rule", "object_mixkit_mug_pink_tee_man", 2.4, 4.5, ["new_object"], S(attending=False), [COFFEE], False),
    ("cat_rule_no_cat", "sl_S0648_0373_P0299A", 12.0, 13.0, ["scene_change"], S(), [CAT], False),
    ("wave_rule", "wave_vcd_bald_man", 1.8, 3.2, ["gesture", "motion_burst"], S(), [WAVE_NO], False),
    ("enter_cat_rule", "appear_bedroom_woman_goes_to_bed", 1.8, 3.2, ["scene_change"], S(face=False, attending=False), [CAT], None),
]
MODELS = ["google/gemini-3.1-flash-lite", "google/gemini-3.5-flash-lite", "google/gemini-3.8-flash", "openai/gpt-5.4-nano",
          "openai/gpt-5.4-mini", "qwen/qwen3.8-flash", "anthropic/claude-haiku-5.5"]


def frame_url(clip, t, cache):
    key = f"{clip}@{t:.2f}"
    if key in cache:
        return cache[key]
    out = os.path.join(cache["_dir"], key.replace("@", "_") + ".jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", f"{t:.3f}", "-i", os.path.join(CLIPS, clip, "camera.mp4"),
                    "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "7", out], check=True)
    u = "data:image/jpeg;base64," + base64.b64encode(open(out, "rb").read()).decode()
    cache[key] = u
    return u


def call(model, payload, timeout=20):
    key = os.environ.get("OPENROUTER_API_KEY", "")
    req = urllib.request.Request(AJ.OPENROUTER_URL, data=json.dumps(dict(payload, model=model)).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        err = None
    except Exception as e:  # noqa: BLE001
        resp, err = {}, f"{type(e).__name__}: {str(e)[:120]}"
    ms = (time.monotonic() - t0) * 1000
    txt = ""
    try:
        txt = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    usage = resp.get("usage") or {}
    usd = float(usage.get("cost") or 0.0)
    return {"ms": ms, "text": txt, "usd": usd, "err": err or (resp.get("error") or {}).get("message"), "verdict": AJ.parse_verdict(txt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(T, "report", "attention_judge_bench.json"))
    ap.add_argument("--spend", default="")
    ap.add_argument("--cap", type=float, default=1.0)
    a = ap.parse_args()
    cache = {"_dir": tempfile.mkdtemp(prefix="avatar-judge_")}
    judge = AJ.AttentionJudge(None, log=lambda *x: None, cfg={"provider": "openrouter"})
    jobs = []
    for it in ITEMS:
        iid, clip, tb, tn, kinds, summ, rules, exp = it
        judge.state["rules"] = [{"id": f"r{i}", "text": r, "t": 0, "source": "user"} for i, r in enumerate(rules)]
        imgs = [frame_url(clip, tb, cache), frame_url(clip, tn, cache)]
        msg = {"type": "attention_candidate", "id": iid, "kinds": kinds, "summary": summ, "details": {}}
        payload = judge.payload(judge.build_messages(msg, imgs))
        payload.pop("model", None)
        for m in a.models.split(","):
            for r in range(a.reps):
                jobs.append((m, iid, exp, bool(rules), payload))
    spent = 0.0
    res = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for j in jobs:
            futs[ex.submit(call, j[0], j[4])] = j
        for f in cf.as_completed(futs):
            m, iid, exp, rule, _ = futs[f]
            r = f.result()
            spent += r["usd"]
            v = r["verdict"]
            res.append({"model": m, "item": iid, "expect": exp, "rule": rule, "ms": round(r["ms"]), "usd": r["usd"], "err": r["err"],
                        "worth_reply": v["worth_reply"] if v else None, "what": v["what"] if v else None, "salience": v["salience"] if v else None})
            if spent > a.cap:
                print("cap reached", spent, file=sys.stderr)
                break
    if a.spend:
        with open(a.spend, "a") as fo:
            fo.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": "attention_judge_bench", "vendor": "openrouter",
                                 "usd": round(spent, 5), "what": f"attention judge bench {len(res)} calls"}) + "\n")
    table = []
    for m in a.models.split(","):
        rs = [r for r in res if r["model"] == m]
        sc = [r for r in rs if r["expect"] is not None]
        ok = [r for r in sc if r["worth_reply"] is not None and r["worth_reply"] == r["expect"]]
        rl = [r for r in sc if r["rule"]]
        rok = [r for r in rl if r["worth_reply"] is not None and r["worth_reply"] == r["expect"]]
        ms = sorted(r["ms"] for r in rs if not r["err"])
        pct = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))] if ms else None  # noqa: E731
        table.append({"model": m, "n": len(rs), "scored": len(sc), "acc": round(len(ok) / max(1, len(sc)), 3),
                      "rules": f"{len(rok)}/{len(rl)}", "invalid": sum(1 for r in rs if r["worth_reply"] is None),
                      "p50_ms": pct(0.5), "p90_ms": pct(0.9), "usd_per_call": round(sum(r["usd"] for r in rs) / max(1, len(rs)), 6),
                      "wrong": [r["item"] for r in sc if r not in ok]})
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "items": len(ITEMS), "spent_usd": round(spent, 5), "table": table, "results": res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for row in table:
        print(json.dumps(row))
    print("spent", round(spent, 4))


if __name__ == "__main__":
    main()
