"""runtime recipe for a NEW bundle (re-apply to the sweep bundles): flow settings -> closed-loop event calibration
-> a runtime variant bundle.

    python -I tools/train/rt_recipe.py --bundle <src bundle dir> --name <run name> --flow '<runtime json>'
        [--events nod,blink] [--gain '{"user": -23, "self": -19}'] [--jobs runs/p2rt/jobs_val.pkl] [--workers 8]

1. rt_probe run with the flow settings and events OFF (val pack, 24 x 60 s closed loop). The event head's p does not
   depend on firing when feed_history is false, so this run's p traces calibrate the thresholds EXACTLY.
2. thresholds = the highest threshold whose rising-crossing + refractory fire count over those p traces matches the
   target rate per class: --target label = the val label rate (a T0 flow makes no events of its own); --target deficit
   (default) = what the sampled flow LACKS: label rate x max(0, 1 - gen / real) with gen / real = evaluate.py's own
   detector (nods / blinks / smile onsets per min) on the events-off rollouts vs the real people - a sampled flow that
   already blinks 26/min of the people's 34 gets 23% of the label rate as events, one that smiles at the human rate
   gets none. Classes not in --events get threshold 1.1 (never fire).
3. rt_probe run with {"events": {"mode": "threshold", "feed_history": false, "thresholds": ...}} (the check).
4. runs/p2rt/<name>/bundle = a copy of the source bundle whose config["runtime"] = flow + events (+ input_gain_db).
Prints both probe lines and the runtime written. The source bundle is never modified.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
T = os.environ.get("MM_WORK_DIR", os.path.join(ROOT, "work"))
EV = ("nod", "blink", "smile")


def probe(a, tag, rt):
    out = os.path.join(a.outdir, f"{tag}.json")
    if not os.path.exists(out):
        env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", OMP_NUM_THREADS="1")
        subprocess.run([PY, "-I", os.path.join(ROOT, "tools/train/rt_probe.py"), "run", "--jobs", a.jobs, "--bundle",
                        a.bundle, "--tag", tag, "--rt", json.dumps(rt), "--workers", str(a.workers), "--outdir", a.outdir]
                       + (["--save"] if a.save else []), check=True, cwd=ROOT, env=env)
    else:
        subprocess.run([PY, "-I", os.path.join(ROOT, "tools/train/rt_probe.py"), "table", out], check=True, cwd=ROOT)
    with open(out) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--flow", required=True, help="runtime json without events (temperature, groups, rho, box...)")
    ap.add_argument("--events", default="nod,blink", help="classes that may fire (others: threshold 1.1 = never). "
                    "Smiles are off by default: the sampled flows hold a smile mean ABOVE the people's (.47-.55 vs .30) "
                    "and 2-s smile curves on top only raise it (base2: deficit smiles 9.6/min -> scorecard 12.78 vs "
                    "11.8 proxy off; tgA: smile events 18.9 -> 19.9)")
    ap.add_argument("--target", default="deficit", choices=("deficit", "label"))
    ap.add_argument("--gain", default=None, help='input_gain_db json for the written bundle (not used by the val probe: '
                    'the pack features are at training levels)')
    ap.add_argument("--jobs", default=f"{T}/runs/p2rt/jobs_val.pkl")
    ap.add_argument("--outdir", default=f"{T}/runs/p2rt/probe")
    ap.add_argument("--dest", default=f"{T}/runs/p2rt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    flow = json.loads(a.flow)
    flow.pop("events", None)
    on = [x for x in a.events.split(",") if x]
    a.save = True                                     # the p traces of the events-off run are the calibration data
    o = probe(a, f"{a.name}_off", dict(flow, events={"mode": "off"}))
    ev = o.get("events")
    if not ev:
        raise SystemExit("bundle has no event head: nothing to calibrate")
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "tools/train"))
    import rt_probe as RP
    z = np.load(os.path.join(a.outdir, f"{a.name}_off.npz"))
    evp = z["evp"]
    mins = evp.shape[0] * evp.shape[1] / 30 / 60
    names = list(ev)
    det = {"nod": "nods_per_min", "blink": "blinks_per_min", "smile": "smiles_per_min"}
    thr, targets = [], {}
    for c, n in enumerate(names):
        rate = float(ev[n]["label_all"])
        if a.target == "deficit":
            g, r = o["gen"].get(det[n]), o["real"].get(det[n])
            rate *= max(0.0, 1.0 - g / r) if g is not None and r else 1.0
        targets[n] = round(rate, 2)
        if n not in on or rate < 0.5:
            thr.append(1.1)
            continue
        refr = ev[n].get("refr")
        if refr is None:
            with open(os.path.join(a.bundle, "config.json")) as f:
                refr = (json.load(f).get("events") or {}).get("refractory", [0] * len(names))[c]
        th = RP.rate_match_threshold([evp[i, :, c] for i in range(len(evp))], rate * mins, int(refr))
        thr.append(round(th, 4) if th == th else 1.1)
    events = {"mode": "threshold", "feed_history": False, "thresholds": thr}
    o2 = probe(a, f"{a.name}_ev", dict(flow, events=events))
    rt = dict(flow, events=events)
    if a.gain:
        rt["input_gain_db"] = json.loads(a.gain)
    r = subprocess.run([PY, "-I", os.path.join(ROOT, "tools/train/rt_bundle.py"), a.bundle,
                        os.path.join(a.dest, a.name), json.dumps(rt)], check=True, cwd=ROOT, capture_output=True,
                       text=True)
    print(r.stdout.strip())
    print(json.dumps({"name": a.name, "targets_per_min": targets, "runtime": rt, "proxy_off": o.get("realism"), "proxy_ev": o2.get("realism"),
                      "fired": {n: round(o2["events"][n]["fired_all"], 1) for n in names},
                      "label": {n: round(o2["events"][n]["label_all"], 1) for n in names}}))


if __name__ == "__main__":
    main()
