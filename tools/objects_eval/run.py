#!/usr/bin/env python3
"""objects-eval — end-to-end manipulation eval against a candidate worktree.

  python3 tools/objects_eval/run.py --worktree <path> --run <name>                 # full: sync, simulate, render, judge, report
  python3 tools/objects_eval/run.py --worktree <path> --run <name> --only o01,o05  # subset (also --kinds physics,relation)
  Workspace (candidate/, runs/, cache/): $OBJECTS_EVAL_DIR, default <repo>/runs/objects_eval.
  PYTHON = interpreter for the renderer child (default: this one). AVATAR_CLAUDE_BIN = the `claude` CLI.
  options: --cand NAME (an already-synced candidate/<NAME>; --worktree syncs into candidate/<basename> first)
           --direct (cases' `direct` tool calls instead of the Claude stand-in; cases without one are skipped)
           --phases sim,render,judge,report (default all; a later phase reuses the run dir's earlier outputs)
           --jobs 4 (parallel sim / judge; claude -p is capped at 4 at once)  --render-jobs 2 --threads 6 (<= 12 total)
           --max-usd 1.5 (estimated planner cost cap for this run)
           --requests FILE (default tools/objects_eval/requests.json if present, else requests_direct.json)
Outputs: <OBJECTS_EVAL_DIR>/runs/<run>/{results.json, report.md, <case>/case.json rig.jsonl strip.png strip_a/b.png kNN.png}
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time
from pathlib import Path

EVAL = Path(os.environ.get("OBJECTS_EVAL_DIR") or Path(__file__).resolve().parents[2] / "runs" / "objects_eval")   # workspace: candidate/, runs/, cache/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REQUESTS = HERE / "requests.json" if (HERE / "requests.json").exists() else HERE / "requests_direct.json"
import sim, checks  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree"); ap.add_argument("--cand"); ap.add_argument("--run", required=True)
    ap.add_argument("--only", default=""); ap.add_argument("--kinds", default="")
    ap.add_argument("--direct", action="store_true"); ap.add_argument("--phases", default="sim,render,judge,report")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--render-jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--max-usd", type=float, default=1.5)
    ap.add_argument("--requests", default=str(REQUESTS)); ap.add_argument("--nkeys", type=int, default=12)
    a = ap.parse_args()
    if a.render_jobs * a.threads > 12:
        sys.exit("render-jobs x threads must be <= 12")
    if a.worktree:
        name = a.cand or Path(a.worktree).resolve().name
        subprocess.run([str(HERE / "sync.sh"), a.worktree, name], check=True, env=dict(os.environ, OBJECTS_EVAL_DIR=str(EVAL)))
        a.cand = name
    if not a.cand:
        sys.exit("--worktree or --cand")
    cand = EVAL / "candidate" / a.cand
    src = cand / "source"
    sim.load_candidate(cand)
    run_dir = EVAL / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.requests).read_text())["cases"]
    only, kinds = set(filter(None, a.only.split(","))), set(filter(None, a.kinds.split(",")))
    cases = [c for c in cases if (not only or c["id"] in only) and (not kinds or c.get("kind") in kinds)]
    if a.direct:
        cases = [c for c in cases if c.get("direct")]
    phases = set(a.phases.split(","))
    meta = {"run": a.run, "cand": a.cand, "commit": (cand / "COMMIT").read_text().strip(), "source": (cand / "SOURCE").read_text().strip(),
            "mode": "direct" if a.direct else "standin", "spec_hash": sim.session_spec()["hash"], "cases": [c["id"] for c in cases],
            "started": time.strftime("%F %T")}
    log(f"run {a.run}: {len(cases)} cases on {a.cand} {meta['commit']} ({meta['mode']}), planner est. cost so far ${sim.spent_usd():.4f}")
    usd0 = sim.spent_usd()

    # 1+2: simulate
    if "sim" in phases:
        def one(c):
            r = sim.run_case(c, run_dir, a.direct, a.max_usd)
            log("sim", c["id"], "ERROR " + r["error"] if r.get("error") else f"{r['wall_s']}s plans={len(r['plans'])} calls={sum(len(rr['calls']) for t in r['turns'] for rr in t['rounds'])}")
            return r
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, cases))
    recs = {}
    for c in cases:
        f = run_dir / c["id"] / "case.json"
        if f.exists():
            recs[c["id"]] = json.loads(f.read_text())
    # state checks (cheap, always)
    for cid, r in recs.items():
        if r.get("error"):
            continue
        r["state_check"] = checks.run_checks(src, run_dir / cid, checks.CHECKS.get(cid, []), r["main_frame0"])
        (run_dir / cid / "case.json").write_text(json.dumps(r, indent=1))

    # 3: render
    import strip as stripmod
    strips = {}
    if "render" in phases:
        def rend(cid):
            try:
                s = stripmod.make_strip(src, run_dir / cid, recs[cid], n=a.nkeys, threads=a.threads)
                (run_dir / cid / "strip.json").write_text(json.dumps(s))
                log("render", cid, "ok")
                return cid, s
            except Exception as e:  # noqa: BLE001
                log("render", cid, "FAILED", repr(e)[:300])
                return cid, {"error": repr(e)[:300]}
        with cf.ThreadPoolExecutor(a.render_jobs) as ex:
            strips = dict(ex.map(rend, [cid for cid, r in recs.items() if not r.get("error")]))
    for cid in recs:
        f = run_dir / cid / "strip.json"
        if cid not in strips and f.exists():
            strips[cid] = json.loads(f.read_text())

    # 4: judge
    import judge as judgemod
    verdicts = {}
    if "judge" in phases:
        def jd(cid):
            s = strips.get(cid)
            if not s or s.get("error"):
                return cid, {"error": "no strip", "score": 0}
            try:
                v = judgemod.judge_case(recs[cid], s, run_dir, EVAL / "cache/judge")
            except Exception as e:  # noqa: BLE001
                v = {"error": repr(e)[:300], "score": 0}
            (run_dir / cid / "verdict.json").write_text(json.dumps(v, indent=1))
            log("judge", cid, v.get("score"), v.get("did_what_asked"), (v.get("notes") or v.get("error") or "")[:100])
            return cid, v
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            verdicts = dict(ex.map(jd, [cid for cid in recs if not recs[cid].get("error")]))
    for cid in recs:
        f = run_dir / cid / "verdict.json"
        if cid not in verdicts and f.exists():
            verdicts[cid] = json.loads(f.read_text())

    # 5: report
    if "report" in phases:
        report(run_dir, meta, cases, recs, strips, verdicts, sim.spent_usd() - usd0)


def tool_line(r):
    out = []
    for t in r.get("turns") or []:
        if t.get("setup"):
            continue
        for rr in t["rounds"]:
            for c in rr["calls"]:
                a = c["args"]
                if c["name"] == "object":
                    o = a.get("object") or {}
                    bits = [a.get("verb"), o.get("name") or o.get("ref"), a.get("target"), a.get("place")]
                    s = "object " + " ".join(str(b) for b in bits if b)
                elif c["name"] in ("do_action", "perform_action"):
                    s = "do_action '" + str(a.get("description", ""))[:50] + "'"
                else:
                    s = c["name"] + " " + json.dumps(a)[:40]
                ok = c["result"].get("ok")
                out.append(s + ("" if ok is not False else " ✗"))
    return "; ".join(out) or "(no call)"


def report(run_dir, meta, cases, recs, strips, verdicts, usd):
    rows = []
    for c in cases:
        cid = c["id"]; r = recs.get(cid) or {"error": "no record"}; v = verdicts.get(cid) or {}
        st = (r.get("state_check") or {}).get("status", "n/a")
        sc = v.get("score")
        if r.get("error"):
            res = "error"
        elif sc == 2 and st in ("ok", "n/a"):
            res = "pass"
        elif sc is not None and sc >= 1:
            res = "partial"
        else:
            res = "fail"
        rows.append({"id": cid, "kind": c.get("kind"), "turns": c.get("turns"), "setup": c.get("setup") or [], "expect": c.get("expect"),
                     "tools": tool_line(r) if not r.get("error") else r.get("error"), "said": " | ".join(s for t in r.get("turns") or [] if not t.get("setup") for s in t.get("said") or []),
                     "plans": [{k: p.get(k) for k in ("description", "ok", "source", "error")} for p in r.get("plans") or []],
                     "state_check": r.get("state_check"), "verdict": v, "result": res, "strip": (strips.get(cid) or {}).get("strip"),
                     "teleports": (r.get("dump_main") or {}).get("teleports"), "errors": r.get("errors"), "case": str(run_dir / cid / "case.json")})
    n = len(rows); p = sum(x["result"] == "pass" for x in rows); pa = sum(x["result"] == "partial" for x in rows)
    st_ok = sum((x["state_check"] or {}).get("status") == "ok" for x in rows); st_n = sum((x["state_check"] or {}).get("status") in ("ok", "fail") for x in rows)
    score = sum((x["verdict"] or {}).get("score") or 0 for x in rows)
    by_kind = {}
    for x in rows:
        k = by_kind.setdefault(x["kind"], [0, 0, 0]); k[0] += 1; k[1] += x["result"] == "pass"; k[2] += x["result"] == "partial"
    summ = {"n": n, "pass": p, "partial": pa, "fail": n - p - pa, "pass_rate": round(p / n, 3) if n else 0,
            "judge_score": f"{score}/{2 * n}", "state_ok": f"{st_ok}/{st_n}", "by_kind": by_kind, "planner_usd_this_run": round(usd, 4)}
    out = {"meta": dict(meta, finished=time.strftime("%F %T")), "summary": summ, "rows": rows}
    (run_dir / "results.json").write_text(json.dumps(out, indent=1))
    L = [f"# objects-eval run `{meta['run']}` — {meta['cand']} {meta['commit']} ({meta['mode']})", "",
         f"source {meta['source']}; session spec {meta['spec_hash']}; {meta['started']} .. {out['meta']['finished']}", "",
         f"**pass {p}/{n} ({100 * p / max(1, n):.0f}%), partial {pa}, fail {n - p - pa}**; judge score {score}/{2 * n}; state checks ok {st_ok}/{st_n}; "
         f"planner est. cost this run ${usd:.4f}", "",
         "by kind: " + ", ".join(f"{k} {v[1]}/{v[0]}" + (f" (+{v[2]} partial)" if v[2] else "") for k, v in by_kind.items()), "",
         "pass = judge score 2 and state check ok (or n/a); partial = judge score >= 1", "",
         "| case | kind | request | tool calls | state | judge | result | strip |", "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        v = x["verdict"] or {}
        stc = x["state_check"] or {}
        st = stc.get("status", "n/a") + ("" if stc.get("status") != "fail" else ": " + "; ".join(" ".join(map(str, i["check"])) for i in stc["items"] if not i["ok"]))
        j = f"{v.get('score', '-')} {v.get('did_what_asked', '')}" + (" tele" if v.get("teleport_or_pop") == "yes" else "") + \
            (" clip" if v.get("interpenetration") == "yes" else "") + (" DISHONEST" if v.get("honest_report") == "no" else "")
        req = (("[" + " / ".join(x["setup"]) + "] ") if x["setup"] else "") + " / ".join(x["turns"] or [])
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
        L.append(f"| {x['id']} | {x['kind']} | {cell(req)[:110]} | {cell(x['tools'])[:160]} | {cell(st)[:80]} | {cell(j)} | {x['result']} | {x['strip'] or '-'} |")
    L += ["", "## judge notes", ""] + [f"- **{x['id']}** ({x['result']}): {(x['verdict'] or {}).get('notes') or (x['verdict'] or {}).get('error') or ''} — said: \"{x['said'][:160]}\"" for x in rows]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    log(f"REPORT {run_dir / 'report.md'}: pass {p}/{n}, partial {pa}, judge {score}/{2 * n}, state {st_ok}/{st_n}, planner ${usd:.4f}")


if __name__ == "__main__":
    main()
