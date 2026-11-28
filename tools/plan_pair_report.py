#!/usr/bin/env python3
"""paired per-item judge scores of plan_eval runs on the same set, split by whether the Claude plan of the
FIRST run opened a hidden thinking block (output tokens > chars / 1.4: thinking tokens are counted but not written).

    python3 tools/plan_pair_report.py <runA> <runB> [<runC> ...]      (run dirs of tools/plan_eval.py; judge sonnet-zoom)"""
import json, os, sys


def load(run):
    out = {}
    for d in sorted(os.listdir(run)):
        j, p = os.path.join(run, d, "judge.sonnet-zoom.json"), os.path.join(run, d, "plan.json")
        if not os.path.exists(p):
            continue
        plan = json.load(open(p))
        score = json.load(open(j)).get("score") if os.path.exists(j) else 1
        att = (plan.get("attempts") or [{}])[0]
        u = att.get("usage") or {}
        chars = att.get("chars") or 0
        think = bool(u.get("output_tokens") and chars and u["output_tokens"] > chars / 1.4)
        out[d] = {"score": score, "first": plan.get("ms_first_pose"), "think": think, "model": plan.get("model"),
                  "fallback": plan.get("fallback_used"), "text": plan.get("text")}
    return out


def mean(v):
    return round(sum(v) / len(v), 3) if v else None


def main():
    runs = sys.argv[1:]
    R = [load(r) for r in runs]
    ids = sorted(set.intersection(*[set(r) for r in R]))
    names = [os.path.basename(r.rstrip("/")) for r in runs]
    think_ids = [i for i in ids if R[0][i]["think"]]
    rest = [i for i in ids if i not in think_ids]
    print(f"items {len(ids)}; the first run's plan thought on {len(think_ids)}")
    for name, r in zip(names, R):
        s = [r[i]["score"] for i in ids]
        print(f"{name:32s} mean {mean(s)}  pass(>=3) {sum(x >= 3 for x in s)}/{len(s)}  | thinking items {mean([r[i]['score'] for i in think_ids])}"
              f"  other {mean([r[i]['score'] for i in rest])}  | fallback {sum(bool(r[i]['fallback']) for i in ids)}"
              f"  thought {sum(r[i]['think'] for i in ids)}")
    print("per thinking item: " + " | ".join(names))
    for i in think_ids:
        print(f"  {i} {str(R[0][i]['text'])[:44]:44s} " + "  ".join(f"{r[i]['score']} ({r[i]['first']} ms)" for r in R))


if __name__ == "__main__":
    main()
