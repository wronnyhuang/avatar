"""per-bundle runtime AUTO-TUNE: search the flow knobs of ONE bundle so its closed-loop rollouts match the held-out
people, then the deficit-event recipe (rt_recipe.py) on the winner -> a runtime variant bundle.

    python -I tools/train/rt_autotune.py --bundle <src bundle dir> --name <run name> [--budget 12]
        [--base '<flow json>'] [--gain '{"user": -23, "self": -19}'] [--workers 8] [--no-recipe]

Objective = rt_probe's realism proxy (window-feature Frechet, gen minus the real val people; events OFF). Diagnostics
per probe: stillness %, head speed, blinks / nods per min (listening), eyes-closed fraction, gaze-away - gen / real.
Search = coordinate descent from the base flow (default: the ev1-A3 flow), one knob at a time, keeping a move only
when the proxy improves by more than --eps:
  1. mult   : one multiplier on temperature + every temperature_groups entry (1, 1.3, 1.6, 2; the grid is extended
              once past an edge winner: 2.5 / 0.75);
  2. head   : the head group temperature x (0.75, 1.5);      3. gaze: the gaze group temperature x (0.75, 1.5);
  4. rho    : noise_rho (the other of .8 / .9);               5. guidance (the other of 2 / 3).
Every probe = one rt_probe run (24 x 60 s closed loop, ~100 s at 8 workers) cached as <outdir>/<name>_at_<hash>.json:
a re-run skips finished probes (resumable); --budget caps the probes of the SEARCH (cached ones count; when it is
spent the best so far wins). Then rt_recipe.py (2 more probes: events-off traces + the check) writes
runs/p2rt/<name>/bundle; runs/p2rt/<name>/autotune.json = every probe (knobs, proxy, rates), the winner, the recipe.
Nothing here is fitted to a particular bundle: the grid is fixed, the bundle's own probes pick the point.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
T = os.environ.get("MM_WORK_DIR", os.path.join(ROOT, "work"))
BASE = {"temperature": 0.3, "noise_rho": 0.8,
        "temperature_groups": {"torso": 0.1, "arm_l": 0.15, "arm_r": 0.15, "fingers_l": 0.15, "fingers_r": 0.15,
                               "head": 0.2, "face_upper": 0.3, "mouth": 0.2, "gaze": 0.15},
        "box_groups": {"torso": 9},
        "anchor": {"cols": ["bs_eyeBlinkLeft", "bs_eyeBlinkRight", "bs_eyeSquintLeft", "bs_eyeSquintRight",
                            "bs_mouthSmileLeft", "bs_mouthSmileRight"], "tau_s": 4}}
KNOB0 = {"mult": 1.0, "head": 1.0, "gaze": 1.0, "rho": None, "guidance": None}
DIAG = ("still_pct", "head_speed", "blinks_L", "nods_L", "smiles_L", "smile_mean", "eye_closed_frac",
        "look_away_listen_pct", "gaze_sd")


def flow_of(base, k):
    f = json.loads(json.dumps(base))
    f.pop("events", None)
    m = float(k["mult"])
    f["temperature"] = round(float(f.get("temperature", 1.0)) * m, 4)
    tg = {g: float(v) * m for g, v in (f.get("temperature_groups") or {}).items()}
    for g in ("head", "gaze"):
        tg[g] = tg.get(g, f["temperature"] / m * m) * float(k[g])
    f["temperature_groups"] = {g: round(v, 4) for g, v in tg.items()}
    if k.get("rho") is not None:
        f["noise_rho"] = float(k["rho"])
    if k.get("guidance") is not None:
        f["guidance"] = float(k["guidance"])
    return f


def tag_of(name, flow):
    return f"{name}_at_" + hashlib.sha1(json.dumps(flow, sort_keys=True).encode()).hexdigest()[:8]


class Tuner:
    def __init__(self, a):
        self.a = a
        self.base = json.loads(a.base) if a.base else BASE
        self.dir = os.path.join(a.dest, a.name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "autotune.json")
        self.probes = []                                # in order of first use this invocation
        self.used = 0

    def probe(self, k):
        """knobs -> probe record (cached on disk by flow hash). None when the budget is spent."""
        flow = flow_of(self.base, k)
        tag = tag_of(self.a.name, flow)
        for p in self.probes:
            if p["tag"] == tag:
                return p
        out = os.path.join(self.a.outdir, f"{tag}.json")
        if self.used >= self.a.budget:
            return None
        self.used += 1
        t0 = time.time()
        if not os.path.exists(out):
            env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", OMP_NUM_THREADS="1")
            subprocess.run([PY, "-I", os.path.join(ROOT, "tools/train/rt_probe.py"), "run", "--jobs", self.a.jobs,
                            "--bundle", self.a.bundle, "--tag", tag, "--rt", json.dumps(dict(flow, events={"mode": "off"})),
                            "--workers", str(self.a.workers), "--outdir", self.a.outdir], check=True, cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL)
        with open(out) as f:
            o = json.load(f)
        r = o.get("realism")
        rec = dict(tag=tag, knobs=dict(k), flow=flow, proxy=r if r is not None and r == r else float("inf"),
                   proxy_LS=o.get("realism_per_class"), gaps=(o.get("realism_gaps") or [])[:4],
                   gen={d: o["gen"].get(d) for d in DIAG}, real={d: o["real"].get(d) for d in DIAG},
                   step_ms=o.get("step_ms", [None] * 4)[3], wall_s=round(time.time() - t0, 1))
        self.probes.append(rec)
        self.write()
        print(row(rec), flush=True)
        return rec

    def write(self, **extra):
        d = dict(name=self.a.name, bundle=os.path.abspath(self.a.bundle), base=self.base, budget=self.a.budget,
                 probes=self.probes)
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    old = json.load(f)
                d.update({k: old[k] for k in ("winner", "recipe") if k in old})
            except (OSError, ValueError):
                pass
        d.update(extra)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, self.path)

    def line(self, best, key, values):
        """try `values` for knob `key` from the incumbent; returns the new incumbent."""
        for v in values:
            k = dict(best["knobs"], **{key: v})
            p = self.probe(k)
            if p is None:
                break
            if p["proxy"] < best["proxy"] - self.a.eps:
                best = p
        return best

    def search(self):
        best = self.probe(dict(KNOB0))
        if best is None:
            raise SystemExit("budget 0: nothing probed")
        grid = [1.3, 1.6, 2.0]
        best = self.line(best, "mult", grid)
        if best["knobs"]["mult"] == 2.0:
            best = self.line(best, "mult", [2.5])
        elif best["knobs"]["mult"] == 1.0:
            best = self.line(best, "mult", [0.75])
        best = self.line(best, "head", [0.75, 1.5])
        best = self.line(best, "gaze", [0.75, 1.5])
        rho0 = float(self.base.get("noise_rho", 0.8))
        best = self.line(best, "rho", [0.9 if rho0 < 0.85 else 0.8])
        best = self.line(best, "guidance", [3.0])      # bundles default to guidance 2
        return best


def row(p):
    g, r, k = p["gen"], p["real"], p["knobs"]
    f = lambda x, n=1: "nan" if x is None else f"{x:.{n}f}"      # noqa: E731
    return (f"| x{k['mult']} h{k['head']} g{k['gaze']} rho {k['rho'] or '-'} gd {k['guidance'] or '-'} | {f(p['proxy'], 2)} "
            f"[{f((p['proxy_LS'] or [None])[0])}/{f((p['proxy_LS'] or [None, None])[1])}] | {f(g['still_pct'], 0)}/{f(r['still_pct'], 0)} "
            f"| {f(g['head_speed'], 3)}/{f(r['head_speed'], 3)} | {f(g['blinks_L'])}/{f(r['blinks_L'])} | {f(g['nods_L'])}/{f(r['nods_L'])} "
            f"| {f(g['eye_closed_frac'], 2)}/{f(r['eye_closed_frac'], 2)} | {f(g['look_away_listen_pct'], 0)}/{f(r['look_away_listen_pct'], 0)} "
            f"| {f(g['smile_mean'], 2)}/{f(r['smile_mean'], 2)} | " + ",".join(f"{a}:{b:.1f}" for a, b in p["gaps"][:3]) + " |")


HEAD = ("| knobs | proxy [L/S] | still % | head speed | blinks L | nods L | eyes closed | away L % | smile mean | top gaps |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, default=12, help="max probes of the search (cached probes count)")
    ap.add_argument("--base", default=None, help="base flow json (default: the ev1-A3 flow)")
    ap.add_argument("--eps", type=float, default=0.15, help="min proxy gain to accept a move")
    ap.add_argument("--gain", default='{"user": -23, "self": -19}')
    ap.add_argument("--jobs", default=f"{T}/runs/p2rt/jobs_val.pkl")
    ap.add_argument("--outdir", default=f"{T}/runs/p2rt/probe")
    ap.add_argument("--dest", default=f"{T}/runs/p2rt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-recipe", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tu = Tuner(a)
    print(HEAD, flush=True)
    best = tu.search()
    tu.write(winner=best)
    print("WINNER " + json.dumps({"knobs": best["knobs"], "proxy": best["proxy"], "flow": best["flow"]}), flush=True)
    if not a.no_recipe:
        cmd = [PY, "-I", os.path.join(ROOT, "tools/train/rt_recipe.py"), "--bundle", a.bundle, "--name", a.name, "--flow",
               json.dumps(best["flow"]), "--jobs", a.jobs, "--outdir", a.outdir, "--dest", a.dest, "--workers",
               str(a.workers)] + (["--gain", a.gain] if a.gain else [])
        r = subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
        last = [x for x in r.stdout.strip().splitlines() if x.startswith("{")][-1]
        tu.write(winner=best, recipe=json.loads(last))
        print("RECIPE " + last, flush=True)
    print(f"DONE {a.name} probes {len(tu.probes)} wall {time.time() - t0:.0f}s -> {os.path.join(tu.dir, 'bundle')}", flush=True)


if __name__ == "__main__":
    main()
