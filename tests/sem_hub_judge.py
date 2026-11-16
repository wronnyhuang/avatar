"""blind judge of semantic programs per tier (claude -p --model sonnet, the subscription; <= 4 at once).
    python3 tests/sem_hub_judge.py --bench DIR --tiers haiku_nt,sonnet_nt --n 15 --api PATH/TO/API.md
Reads DIR/<tier>/NN.json (sem_hub_latency.py outputs), shuffles every (command, program) of all tiers together (the
judge never sees the tier or model), asks "does this program do the command?" -> DIR/judge/<sha>.json (cached: never
re-asked), prints per tier: mean score (1-5), yes / partly / no counts."""
import argparse
import tempfile
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import subprocess

SYS = """You judge short animation programs for a 3-D avatar. Each program is written in the motion API below and is
supposed to perform a spoken command. Decide whether RUNNING the program would make the avatar visibly do what the
command asks (right body parts, right sides — the avatar's own left/right —, counts, the gesture readable and
recognisable). Ignore style and code elegance. Answer ONLY a JSON object:
{"score": 1-5, "verdict": "yes"|"partly"|"no", "why": "<one short sentence>"}
5 = clearly does it, 4 = does it with a small flaw, 3 = partly / ambiguous, 2 = mostly wrong, 1 = does not do it.

API reference:
"""


def ask(system, command, program, cwd):
    user = f'Command: "{command}"\n\nProgram:\n```python\n{program}\n```\n'
    p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--strict-mcp-config",
                        "--no-session-persistence", "--setting-sources", "", "--system-prompt", system,
                        "--output-format", "json"], input=user, capture_output=True, text=True, cwd=cwd, timeout=180)
    try:
        res = json.loads(p.stdout)["result"]
        m = re.search(r"\{.*\}", res, re.S)
        return json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {p.stdout[-300:]} {p.stderr[-300:]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--api", required=True, help="the API text the judge reads (best: the planner's full system prompt)")
    ap.add_argument("--jdir", default="judge")
    a = ap.parse_args()
    system = SYS + open(a.api).read()
    jd = os.path.join(a.bench, a.jdir)
    os.makedirs(jd, exist_ok=True)
    cwd = os.path.join(tempfile.gettempdir(), "sem-judge-cwd")
    os.makedirs(cwd, exist_ok=True)
    items = []
    for tier in a.tiers.split(","):
        d = os.path.join(a.bench, tier)
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))[:a.n]
        for f in files:
            r = json.load(open(os.path.join(d, f)))
            prog = (r.get("program") or "").strip()
            h = hashlib.sha256((r["command"] + "\n" + prog).encode()).hexdigest()[:16]
            items.append((tier, r["command"], prog, h))
    random.Random(7).shuffle(items)
    todo = [it for it in items if not os.path.exists(os.path.join(jd, it[3] + ".json"))]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(ask, system, c, p or "# (no program)", cwd): (t, c, p, h) for t, c, p, h in todo}
        for f in cf.as_completed(futs):
            t, c, p, h = futs[f]
            v = f.result()
            v.update({"command": c, "program": p})
            json.dump(v, open(os.path.join(jd, h + ".json"), "w"), indent=1)
    out = {}
    for t, c, p, h in items:
        v = json.load(open(os.path.join(jd, h + ".json")))
        o = out.setdefault(t, {"n": 0, "score": 0.0, "yes": 0, "partly": 0, "no": 0, "err": 0, "low": []})
        if "score" not in v:
            o["err"] += 1
            continue
        o["n"] += 1
        o["score"] += float(v["score"])
        o[v.get("verdict", "no") if v.get("verdict") in ("yes", "partly", "no") else "no"] += 1
        if float(v["score"]) <= 2:
            o["low"].append(f"{c} -> {v.get('why')}")
    for t, o in out.items():
        o["mean"] = round(o["score"] / max(1, o["n"]), 2)
        print(f"{t:12s} n={o['n']} mean={o['mean']} yes={o['yes']} partly={o['partly']} no={o['no']} err={o['err']}")
        for x in o["low"]:
            print("    low:", x)
    json.dump(out, open(os.path.join(a.bench, a.jdir + "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
