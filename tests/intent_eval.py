"""intent: score the fast matcher on a labelled set (tests/intent_*.txt format).

    python3 tests/intent_eval.py tests/intent_dev.txt [--show 50] [--json out.json] [--min-recall R]

Per line: `phrase | cmd ; cmd` (empty = no command). The matcher runs with the union of the mock scenes
(tests/props_mock/*.json, what the hub loads for every character). Reports
  head precision / recall  (move/<name>, object/<verb>, legacy heads) over all lines,
  exact-line accuracy      (every command and every argument as labelled, in order),
  exact-arg accuracy       (of the commands whose head matched, the share with all arguments right),
  negatives                (lines labelled empty: share that fired anything — the gate wants 0),
  latency                  (µs per utterance, Matcher.parse on a final transcript).
Exit 1 when --min-recall / --max-neg are given and missed (for CI).
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.path.insert(0, HERE)
import intent  # noqa: E402
from check_cases import summ  # noqa: E402


def load(fn):
    out = []
    for ln in open(fn, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#") or "|" not in ln:
            continue
        ph, _, exp = ln.partition("|")
        exp = exp.split("  #")[0]
        cmds = [norm(e) for e in exp.split(";") if e.strip()]
        out.append((ph.strip(), cmds))
    return out


def norm(s):
    toks = s.strip().split()
    if not toks:
        return ""
    head, kv = toks[0], sorted(t for t in toks[1:])
    # legacy heads: keep only action/name for gestures/poses (side suffix optional in labels)
    return " ".join([head] + kv)


def head(s):
    return s.split(" ")[0]


def scene():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "props_mock", "*.json")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--json", default="")
    ap.add_argument("--min-recall", type=float, default=0)
    ap.add_argument("--max-neg", type=float, default=1)
    a = ap.parse_args()
    cases = load(a.file)
    sc = scene()
    m0 = intent.Matcher(scene=sc)
    A = m0.A
    tp = n_got = n_exp = 0
    exact_lines = 0
    arg_ok = arg_n = 0
    neg_n = neg_fired = 0
    pos_n = pos_full = 0
    bad = []
    per_head = Counter()
    per_head_ok = Counter()
    t_us = []
    for ph, exp in cases:
        m = intent.Matcher(scene=sc)
        m.A = A                                  # share the compiled scene (as the hub does)
        t0 = time.perf_counter()
        hits = m.parse(ph, final=True)
        t_us.append((time.perf_counter() - t0) * 1e6)
        got = [norm(summ(h["cmd"])) for h in hits]
        eh, gh = Counter(head(e) for e in exp), Counter(head(g) for g in got)
        tp += sum((eh & gh).values())
        n_got += len(got)
        n_exp += len(exp)
        for e in exp:
            per_head[head(e)] += 1
        for k, v in (eh & gh).items():
            per_head_ok[k] += v
        if got == exp:
            exact_lines += 1
        # args: pair commands by order among same heads
        gl = list(got)
        for e in exp:
            for g in gl:
                if head(g) == head(e):
                    arg_n += 1
                    arg_ok += g == e
                    gl.remove(g)
                    break
        is_neg = not exp or all(head(e) in ("release/all",) for e in exp) and "|" not in ph
        if not exp:
            neg_n += 1
            neg_fired += bool(got)
        else:
            pos_n += 1
            pos_full += [head(g) for g in got] == [head(e) for e in exp]
        if got != exp:
            bad.append((ph, exp, got))
    P = tp / n_got if n_got else 1.0
    Rc = tp / n_exp if n_exp else 1.0
    t_us.sort()
    res = {"file": os.path.basename(a.file), "lines": len(cases), "positives": pos_n, "negatives": neg_n,
           "head_precision": round(P, 4), "head_recall": round(Rc, 4),
           "positive_lines_heads_right": round(pos_full / max(1, pos_n), 4),
           "exact_line_accuracy": round(exact_lines / max(1, len(cases)), 4),
           "exact_arg_accuracy": round(arg_ok / max(1, arg_n), 4),
           "negatives_fired": neg_fired, "negative_fire_rate": round(neg_fired / max(1, neg_n), 4),
           "latency_us": {"median": round(t_us[len(t_us) // 2], 1), "p95": round(t_us[int(len(t_us) * 0.95)], 1),
                          "max": round(t_us[-1], 1)},
           "per_head_recall": {k: round(per_head_ok[k] / v, 3) for k, v in sorted(per_head.items())}}
    print(json.dumps({k: v for k, v in res.items() if k != "per_head_recall"}, indent=1))
    weak = {k: v for k, v in res["per_head_recall"].items() if v < 0.9}
    if weak:
        print("heads under 90% recall:", weak)
    for ph, exp, got in bad[:a.show]:
        print(f"  {ph!r}\n      want {exp}\n      got  {got}")
    if a.json:
        res["mismatches"] = [{"text": p, "want": e, "got": g} for p, e, g in bad]
        json.dump(res, open(a.json, "w"), indent=1)
    if Rc < a.min_recall or res["negative_fire_rate"] > a.max_neg:
        sys.exit(1)


if __name__ == "__main__":
    main()
