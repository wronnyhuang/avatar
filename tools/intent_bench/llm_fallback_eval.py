"""intent c2: score grammar + the server's LLM fallback (app/intent_llm.py) on a fresh c2 set, and time the calls.

    OPENAI_API_KEY=... python3 tools/intent_bench/llm_fallback_eval.py tests/intent_c2_holdout.json \
        --model gpt-4.1-mini [--json out.json]

Grammar first (the server's muse matcher); only when it fires nothing and LlmFallback.wants() says the final looks like a
request is the model asked (as on the server). Prints the estimated LLM cost of the run.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import intent  # noqa: E402
import intent_llm  # noqa: E402
import intent_c2_eval as ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--json")
    a = ap.parse_args()
    items = json.load(open(a.file))["items"]
    sc = ev.scene()
    base = intent.Matcher(scene=sc)
    fb = intent_llm.LlmFallback({"llm_model": a.model, "llm_timeout_s": 1.5})
    calls = []

    class Combo:
        def parse(self, text, final=True):
            m = intent.Matcher(scene=sc)
            m.A = base.A
            hits = m.parse(text, final=True)
            if hits or not fb.wants(text, sc):
                return hits
            cmds, info = fb.ask(text, sc)
            calls.append(dict(info, text=text, n=len(cmds)))
            return [{"cmd": c} for c in cmds]

    rows, bad, lat = ev.score(items, lambda: Combo())
    for k, r in rows.items():
        print(f"{k:12s} " + " ".join(f"{kk}={vv}" for kk, vv in r.items()))
    ms = sorted(c["ms"] for c in calls)
    errs = sum(1 for c in calls if c.get("err"))
    st = {"model": a.model, "calls": len(calls), "errors": errs, "usd": round(fb.usd, 4),
          "ms_p50": ms[len(ms) // 2] if ms else None, "ms_p90": ms[int(len(ms) * 0.9)] if ms else None, "ms_max": ms[-1] if ms else None}
    print("llm", st)
    if a.json:
        json.dump({"rows": rows, "llm": st, "calls": calls, "bad": [{"text": i["text"], "label": i["label"], "why": w,
                   "got": [ev.intent_summ(c) for c in c_]} for i, c_, w in bad]}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
