#!/usr/bin/env python3
"""diagnosis digest: one block per (char, command) with both modes side by side — judge verdicts, natural, pairwise,
source, timings, errors, the semantic program + its report warnings/errors, kf stats (n poses, channels, planned length),
clip + sheet paths. For the human diagnosing failures.
  python3 tools/sem_eval/digest.py --runs EVAL/runs/<tag> --clips EVAL/clips/<tag> --judge EVAL/judge/<tag> --out EVAL/digest-<tag>.md
        [--only-bad]   (only blocks where semantic is partly/no, or errors)
"""
import argparse
import json
import os


def lj(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmds", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                    "tests", "sem_eval", "commands.jsonl"))
    ap.add_argument("--only-bad", action="store_true")
    a = ap.parse_args()
    cmds = [json.loads(l) for l in open(a.cmds) if l.strip()]
    J = {(r["char"], r["mode"], r["id"]): r for r in lj(os.path.join(a.judge, "action.jsonl")) if "verdict" in r}
    PW = {(r["char"], r["id"]): r for r in lj(os.path.join(a.judge, "pairwise.jsonl")) if "better" in r}
    out = []
    for ch in ("muse", "fox", "juno"):
        for c in cmds:
            rows = {}
            for m in ("current", "semantic"):
                p = os.path.join(a.runs, ch, m, "fresh", c["id"] + ".json")
                rows[m] = json.load(open(p)) if os.path.exists(p) else None
            if not any(rows.values()):
                continue
            js = J.get((ch, "semantic", c["id"]))
            S = rows["semantic"] or {}
            bad = (js and js["verdict"] != "yes") or S.get("errors")
            if a.only_bad and not bad:
                continue
            out.append(f"## {ch} / {c['id']} ({c['category']})  — \"{c['utterance']}\"")
            pw = PW.get((ch, c["id"]))
            if pw:
                out.append(f"pairwise: better={pw.get('better_mode') or 'tie'} natural={pw.get('more_natural_mode') or 'tie'} — {pw.get('reason', '')}")
            for m in ("current", "semantic"):
                R = rows[m]
                if not R:
                    out.append(f"- {m}: not run")
                    continue
                T = R.get("timings") or {}
                j = J.get((ch, m, c["id"])) or {}
                kf = R.get("kf") or []
                n = sum(len(k.get("poses") or []) for k in kf)
                chans = sorted({x for k in kf for p in (k.get("poses") or []) for x in p if x != "t"})
                tmax = max([p.get("t", 0) for k in kf for p in (k.get("poses") or [])] or [0])
                out.append(f"- **{m}**: {j.get('verdict', '-')} nat {j.get('natural', '-')} | src {T.get('source')} out {T.get('outcome')} | "
                           f"hub1 {T.get('hub_first_pose_ms')} move {T.get('first_motion_ms')} end {T.get('end_ms')} | poses {n} over {tmax:.1f}s | "
                           f"errors {R.get('errors')}")
                if j:
                    out.append(f"  - judge: {j.get('reason')} / robotic: {j.get('robotic')}")
                out.append(f"  - channels: {', '.join(chans)[:400]}")
                out.append(f"  - sheet: {os.path.join(a.clips, ch, m, c['id'] + '.sheet.jpg')}")
                if m == "semantic":
                    last = [s for s in R.get("sem") or [] if s.get("phase") == "done"]
                    sd = last[-1] if last else {}
                    out.append(f"  - sem: model {sd.get('model')} cached {sd.get('cached')} ms {sd.get('ms')} warnings {sd.get('warnings')} errors {sd.get('errors')} "
                               f"targets {list((sd.get('targets') or {}).keys())[:12]}")
                    out.append("```\n" + (R.get("program") or "(no program)") + "\n```")
                elif R.get("summary"):
                    out.append(f"  - summary: {R['summary']}")
            out.append("")
    open(a.out, "w").write("\n".join(out) + "\n")
    print(a.out, len(out), "lines")


if __name__ == "__main__":
    main()
