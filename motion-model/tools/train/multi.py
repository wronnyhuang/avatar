"""Run several training configs CONCURRENTLY on one GPU (ablations), optionally in sequential BATCHES.

    python3 tools/train/multi.py --config multi.json --run-dir R --pack-dir P [--train-script tools/train/train.py]

multi.json = {"common": {...}, "runs": {"name": {...overrides..., "env": {"K": "V"}}, ...},
              "batches": [["a", "b"], ["c", "d"]],        (optional; absent = every run at once)
              "min_batch_minutes": 10}                    (optional; default 10)
A run's "wait_minutes" (optional) bounds how long that run waits for its data (default: its batch's wait).
Each run trains into R/<name>/ with train.py (or --train-script). "env" (in "common" and/or a run; the run's keys win)
is not part of the train config: it is added to that child's environment.
Batches run one after another, the runs inside a batch concurrently. Before each batch TRAIN_TIME_BUDGET_S is
recomputed for the children from TRAIN_DEADLINE_UNIX (or, when only TRAIN_TIME_BUDGET_S is set, from multi's own
start + that budget); a batch with less than min_batch_minutes left is skipped. Runs named in no batch form a final
batch. SIGTERM is forwarded to the live children and no new run / batch starts after it. R/DONE is written when every
child has exited: {"<run>": {"rc", "done"}, ..., "skipped": {"<run>": "<reason>"}}.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = {"flag": False}
LIVE = {}
PRESENT = set()                     # files in the pack dir when multi started


def on_term(signum, frame):
    STOP["flag"] = True
    for p in LIVE.values():
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)


def merge(common, over):
    cfg = dict(common)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = dict(cfg[k], **v)
        else:
            cfg[k] = v
    return cfg


def plan_batches(mc):
    runs = list(mc["runs"])
    if "skipped" in mc["runs"]:
        raise SystemExit('multi: "skipped" is reserved (DONE summary key), rename that run')
    if not mc.get("batches"):
        return [runs]
    seen, out = set(), []
    for b in mc["batches"]:
        for n in b:
            if n not in mc["runs"]:
                raise SystemExit(f"multi: batch names unknown run {n!r}")
            if n in seen:
                raise SystemExit(f"multi: run {n!r} is in more than one batch")
            seen.add(n)
        if b:
            out.append(list(b))
    rest = [n for n in runs if n not in seen]
    if rest:
        out.append(rest)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--train-script", default=os.path.join(HERE, "train.py"),
                    help="the per-run trainer (tests: a fake); gets --config --run-dir --pack-dir")
    ap.add_argument("--stagger", type=float, default=2.0, help="seconds between child starts inside a batch")
    a = ap.parse_args()
    with open(a.config) as f:
        mc = json.load(f)
    os.makedirs(a.run_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    batches = plan_batches(mc)
    min_left = 60 * float(mc.get("min_batch_minutes", 10))
    t_start = time.time()
    deadline = None
    if os.environ.get("TRAIN_DEADLINE_UNIX"):
        deadline = float(os.environ["TRAIN_DEADLINE_UNIX"])
    elif os.environ.get("TRAIN_TIME_BUDGET_S"):
        deadline = t_start + float(os.environ["TRAIN_TIME_BUDGET_S"])
    rc, skipped, order = {}, {}, []
    PRESENT.update(os.path.join(a.pack_dir, f) for f in (os.listdir(a.pack_dir) if os.path.isdir(a.pack_dir) else []))
    for bi, batch in enumerate(batches):
        if STOP["flag"]:
            for n in batch:
                skipped[n] = "sigterm"
            print(f"batch {bi} {batch}: skipped (SIGTERM)", flush=True)
            continue
        env0 = dict(os.environ)
        if deadline is not None:
            left = deadline - time.time()
            if left < min_left:
                for n in batch:
                    skipped[n] = f"{left / 60:.1f} min left < min_batch_minutes {min_left / 60:g}"
                print(f"batch {bi} {batch}: skipped ({skipped[batch[0]]})", flush=True)
                continue
            env0["TRAIN_TIME_BUDGET_S"] = str(int(left))
        # phase 2: runs may need data that is still being produced (cfg "pack_extra" / "needs", files in the pack dir):
        # <file>.ready (written by whatever produces/copies the file, once it is complete) or a file present at multi's
        # start. Each run of the batch STARTS AS SOON AS ITS OWN DATA IS THERE (the others keep training);
        # a run whose data has not come within the batch's wait ("batch_wait_minutes"[bi], else "max_wait_minutes")
        # is skipped. The batch ends when every started run has exited.
        need = {n: [x for x in (mc["runs"][n].get("pack_extra") or mc.get("common", {}).get("pack_extra") or [])]
                + list(mc["runs"][n].get("needs") or []) for n in batch}
        bw = mc.get("batch_wait_minutes") or []
        t_wait = time.time() + 60 * float(bw[bi] if bi < len(bw) and bw[bi] is not None else mc.get("max_wait_minutes", 20))
        # a run may wait less than its batch ("wait_minutes", e.g. an optional late lane): then it is skipped
        t_wait_run = {n: (time.time() + 60 * float(mc["runs"][n]["wait_minutes"])
                          if mc["runs"][n].get("wait_minutes") is not None else t_wait) for n in batch}

        def _have(f):
            pth = f if os.path.isabs(f) else os.path.join(a.pack_dir, f)
            return os.path.exists(pth + ".ready") or (os.path.exists(pth) and pth in PRESENT)
        print(f"batch {bi} {batch}: start", flush=True)
        procs = {}
        pending = list(batch)
        while pending:
            if STOP["flag"]:
                for n in pending:
                    skipped[n] = "sigterm"
                break
            ready = [n for n in pending if all(_have(f) for f in need[n])]
            for name in ready:
                pending.remove(name)
                env = dict(os.environ)
                if deadline is not None:
                    left = deadline - time.time()
                    if left < min_left:
                        skipped[name] = f"{left / 60:.1f} min left < min_batch_minutes {min_left / 60:g}"
                        print(f"run {name}: skipped ({skipped[name]})", flush=True)
                        continue
                    env["TRAIN_TIME_BUDGET_S"] = str(int(left))
                rd = os.path.join(a.run_dir, name)
                os.makedirs(rd, exist_ok=True)
                cfg = merge(mc.get("common", {}), mc["runs"][name])
                env.update({str(k): str(v) for k, v in (cfg.pop("env", None) or {}).items()})
                cp = os.path.join(rd, "cfg.json")
                with open(cp, "w") as f:
                    json.dump(cfg, f, indent=1)
                log = open(os.path.join(rd, "stdout.log"), "a")
                p = subprocess.Popen([sys.executable, a.train_script, "--config", cp, "--run-dir", rd,
                                      "--pack-dir", a.pack_dir], stdout=log, stderr=subprocess.STDOUT, env=env)
                log.close()
                procs[name] = LIVE[name] = p
                order.append(name)
                print(f"run {name}: started" + (f", budget {env['TRAIN_TIME_BUDGET_S']}s" if deadline else ""),
                      flush=True)
                if STOP["flag"] and p.poll() is None:       # SIGTERM landed between the check and the spawn
                    p.send_signal(signal.SIGTERM)
                if a.stagger > 0:
                    time.sleep(a.stagger)
            if not pending:
                break
            for n in [n for n in pending if time.time() > t_wait_run[n]]:
                pending.remove(n)
                skipped[n] = f"data not in the pack dir: {[f for f in need[n] if not _have(f)]}"
                print(f"run {n}: skipped ({skipped[n]})", flush=True)
            if not pending:
                break
            time.sleep(15)
        for name, p in procs.items():
            while True:
                try:
                    rc[name] = p.wait()
                    break
                except InterruptedError:
                    continue
            LIVE.pop(name, None)
        print(f"batch {bi}: done rc={ {n: rc[n] for n in procs} }", flush=True)
    summary = {}
    for name in order:
        dp = os.path.join(a.run_dir, name, "DONE")
        done = None
        if os.path.exists(dp):
            try:
                with open(dp) as f:
                    done = json.load(f)
            except ValueError:
                done = {"unreadable": True}
        summary[name] = dict(rc=rc[name], done=done)
    summary["skipped"] = skipped
    tmp = os.path.join(a.run_dir, "DONE.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, os.path.join(a.run_dir, "DONE"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
