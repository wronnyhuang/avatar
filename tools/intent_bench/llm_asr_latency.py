"""intent c2: speech end -> command latency of the server's LLM fallback, end to end: OpenAI tts-1 audio of c2 DEV
phrasings the grammar misses, through the real app/asr_worker.py (vosk, full mode), the muse matcher stream, and on the
FINAL with no command, LlmFallback.wants() + ask() (gpt-4.1-mini), as intent_hub._on_line does.

    OPENAI_API_KEY=... python3 tools/intent_bench/llm_asr_latency.py --n 30 --out x.json

latency = (final's audio time + the worker's compute for it - speech end) + the LLM call's wall time.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "app"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import intent  # noqa: E402
import intent_hub  # noqa: E402
import intent_llm  # noqa: E402
from bench import SR, Worker, load_pcm, speech_end  # noqa: E402
from act_asr_check import tts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    scene = intent_hub.scene_for("muse")
    fb = intent_llm.LlmFallback({"llm_model": "gpt-4.1-mini"})
    items = json.load(open(os.path.join(HERE, "..", "..", "tests", "intent_c2_dev.json")))["items"]
    pick = []
    for it in items:
        if it["label"] and not intent.Matcher(scene=scene).parse(it["text"]) and fb.wants(it["text"], scene):
            pick.append(it)
    pick = pick[:a.n]
    chars = 0
    clips = []
    for it in pick:
        fn, c = tts(it["text"], "alloy", 1.0)
        chars += c
        clips.append((it, fn))
    w = Worker(intent_hub.DEFAULTS["python"], ["--model", intent_hub.DEFAULTS["model"], "--mode", "full", "--rules",
                                                os.path.join(HERE, "..", "..", "app", "intent_rules.json")])
    rows, lat, asr_part, llm_part = [], [], [], []
    for it, fn in clips:
        x = load_pcm(fn)
        y = np.concatenate([np.zeros(int(0.4 * SR), np.float32), x, np.zeros(int(1.2 * SR), np.float32)])
        y = (y + np.random.default_rng(1).normal(0, 30, len(y))).clip(-32768, 32767).astype("<i2")
        end = 0.4 + speech_end(x)
        lines = w.run(y, 1)
        st = intent.Matcher(scene=scene).stream()
        fired, row = False, {"text": it["text"]}
        for j in lines:
            hits = st.update(j["text"], j["t"] == "f", j["a"] * 1000)
            if hits:
                fired = True
                row.setdefault("grammar", [h["cmd"] for h in hits])
            if j["t"] == "f" and j.get("text") and not fired:
                row["asr"] = j["text"]
                t_final = j["a"] + j.get("c", 0) / 1000 - end
                if fb.wants(j["text"], scene):
                    cmds, info = fb.ask(j["text"], scene)
                    row.update(cmds=cmds, final_ms=round(t_final * 1000), llm_ms=info["ms"])
                    if cmds:
                        lat.append(t_final * 1000 + info["ms"])
                        asr_part.append(t_final * 1000)
                        llm_part.append(info["ms"])
                else:
                    row["skipped"] = "wants() false on the ASR text"
                break
        rows.append(row)
    w.close()
    pct = lambda v, q: round(sorted(v)[int(len(v) * q)]) if v else None
    res = {"clips": len(rows), "with_command": len(lat), "grammar_fired_on_audio": sum(1 for r in rows if "grammar" in r),
           "speech_end_to_cmd_ms": {"p50": pct(lat, 0.5), "p90": pct(lat, 0.9)},
           "speech_end_to_final_ms": {"p50": pct(asr_part, 0.5), "p90": pct(asr_part, 0.9)},
           "llm_ms": {"p50": pct(llm_part, 0.5), "p90": pct(llm_part, 0.9)}, "llm_usd": round(fb.usd, 4),
           "tts_usd": round(chars * 15 / 1e6, 4), "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
