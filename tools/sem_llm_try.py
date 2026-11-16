#!/usr/bin/env python3
"""Smoke-run the live prompt through `claude -p` (sonnet / haiku) streaming into sem.Stream.

  python tools/sem_llm_try.py --character muse --model sonnet [--workers 4] "command one" "command two" ...

Per command: time to first token, to the first complete statement, to the first pose batch (incl. exec), total,
statements, errors / warnings, and the program.  The hub owns the real latency bench (tests/sem_hub_latency.py); this
only checks that the prompt produces programs the library runs.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import semantic as sem  # noqa: E402


def run(cmd, ch, model):
    targets = sem.resolve_targets(ch, None)
    system = sem.system_prompt(ch, sorted(targets))
    args = ["claude", "-p", "--model", model, "--output-format", "stream-json", "--verbose",
            "--include-partial-messages", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
            "--system-prompt", system]
    os.makedirs("/tmp/t4555core-cwd", exist_ok=True)
    t0 = time.time()
    p = subprocess.Popen(args, cwd="/tmp/t4555core-cwd", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    p.stdin.write(sem.user_prompt(cmd)); p.stdin.close()
    s = sem.Stream(ch, targets=targets)
    first_tok = first_pose = None
    n_poses = 0
    for line in p.stdout:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "stream_event":
            ev = d.get("event", {})
            if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                if first_tok is None:
                    first_tok = time.time() - t0
                for b in s.feed(ev["delta"]["text"]):
                    if b["poses"] and first_pose is None:
                        first_pose = time.time() - t0
                    n_poses += len(b["poses"])
    p.wait()
    for b in s.finish():
        n_poses += len(b["poses"])
        if b["poses"] and first_pose is None:
            first_pose = time.time() - t0
    r = s.report()
    return {"cmd": cmd, "first_token_s": first_tok and round(first_tok, 2), "first_pose_s": first_pose and
            round(first_pose, 2), "total_s": round(time.time() - t0, 2), "n_statements": r["n_statements"],
            "duration_s": r["duration_s"], "errors": r["errors"], "warnings": r["warnings"], "poses": n_poses,
            "program": r["program"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="muse")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("commands", nargs="+")
    a = ap.parse_args()
    with cf.ThreadPoolExecutor(max_workers=min(4, a.workers)) as ex:
        res = list(ex.map(lambda c: run(c, a.character, a.model), a.commands))
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "program"}))
        print(r["program"])
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
