"""p2_sweep_fit.py on a synthetic sweep drawn from a known law.

    python -I -m pytest -q tools/train/test_p2_sweep_fit.py
    python -I tools/train/test_p2_sweep_fit.py
"""
import atexit
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p2_sweep_fit as sf  # noqa: E402

TRUE = dict(E=0.62, A=0.35, alpha=0.32, B=0.9, beta=0.28)
SIZES = {"s5": 5.714362, "s20": 21.6, "s40": 39.9, "s80": 84.7}
LANES = {"d25": 69.1, "d50": 138.2, "d100": 276.5}
CANDOR_D, CANDOR_GAIN = 415.0, 0.98            # CANDOR runs land 2% below the law at their D
TOTAL, EVERY = 24000, 1500
RUN_NOISE, EVAL_NOISE = 3e-4, 1e-4             # relative noise per run (its converged L) and per eval


def true_L(N, D):
    return TRUE["E"] + TRUE["A"] * N ** -TRUE["alpha"] + TRUE["B"] * D ** -TRUE["beta"]


def _tmp():
    d = tempfile.mkdtemp(prefix="p2fit_")
    atexit.register(shutil.rmtree, d, True)
    return d


def val_curve(L, rng, tau=1500.0, overfit=0.0, start=EVERY, stop=TOTAL):
    """Decay toward L; overfit > 0 bends it up over the last 30% of training (small D, big N)."""
    out = []
    for s in range(start, stop + 1, EVERY):
        v = L * (1 + 0.8 * math.exp(-s / tau)) + L * overfit * max(0.0, (s - 0.7 * TOTAL) / TOTAL)
        out.append((s, v * (1 + EVAL_NOISE * rng.standard_normal())))
    return out


def start_row(N, D, step=0):
    return {"event": "start", "step": step, "params": int(round(N * 1e6)), "params_backbone": int(N * 0.8e6),
            "params_head": int(N * 0.2e6), "train_hours": D, "val_hours": 6.2, "datasets": ["seamless", "l2l"],
            "pack_hours": {"seamless": D * 0.7, "l2l": D * 0.3}, "gpu": "NVIDIA A40"}


def train_rows(a, b):
    return [{"step": s, "t": s * 0.15, "lr": 2e-4, "flow": 0.5, "vel": 0.1, "event": 0.02, "gn": 1.3}
            for s in range(a, b + 1, 500)]


def val_row(s, v):
    return {"step": s, "t": s * 0.15, "skipped": 0, "val_loss": v * 1.1, "val_core": v, "val_flow": v * 0.8,
            "val_mae": {"pose": v * 0.1}, "val_resp": 0.3}


def write_run(d, N, D, curve, *, final=True, done=True, cfg=None, rows=None, garble=False):
    os.makedirs(os.path.join(d, "bundle"), exist_ok=True)
    if rows is None:
        rows = [start_row(N, D)] + train_rows(500, curve[-1][0] if curve else 1000)
        rows += [val_row(s, v) for s, v in curve]
        rows.sort(key=lambda r: (r["step"], "val_core" in r))
    lines = [json.dumps(r) for r in rows]
    if garble:
        lines[3:3] = ["", '{"step": 3000, "val_co', "garbage!!"]
    if final and curve:
        vals = [v for _, v in curve]
        i = int(np.argmin(vals))
        lines.append(json.dumps({"step": curve[-1][0], "t": 1.0, "final": True, "val_core": vals[-1] * 0.997,
                                 "best_val_loss_cmp": vals[i], "last_val_loss_cmp": vals[-1],
                                 "from_best_step": curve[i][0] if i < len(vals) - 1 else None,
                                 "stop_reason": None}))
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if done and curve:
        json.dump({"step": curve[-1][0], "minutes": 60.0, "final_val": {"val_core": curve[-1][1]},
                   "params": int(N * 1e6)}, open(os.path.join(d, "DONE"), "w"))
    if cfg:
        json.dump(cfg, open(os.path.join(d, "cfg.json"), "w"))


def make_sweep(base, seed=1):
    """podA/<lane>/<size> (d25, d50) + runs/jobB/latest -> durable/<stamp> (d100, candor, odd cfg-named run)."""
    rng = np.random.default_rng(seed)
    podA = os.path.join(base, "podA")
    stamp = os.path.join(base, "runs", "jobB", "durable", "20261109-1200")
    truth = {}
    for lane, D in list(LANES.items()) + [("candor", CANDOR_D)]:
        for size, N in SIZES.items():
            L = true_L(N, D) * (CANDOR_GAIN if lane == "candor" else 1.0) * (1 + RUN_NOISE * rng.standard_normal())
            truth[(lane, size)] = L
            root = podA if lane in ("d25", "d50") else stamp
            d = os.path.join(root, lane, size)
            kw = {}
            if lane == "d100" and size == "s40":           # odd dir name: lane/size come from cfg.json
                d = os.path.join(stamp, "odd", "name_x")
                kw["cfg"] = {"sweep": {"lane": "d100", "size": "s40", "frac": 1.0, "candor": False}}
            if lane == "candor" and size == "s80":         # still improving at its last eval
                write_run(d, N, D, val_curve(L, rng, tau=7000.0))
                continue
            if lane == "d50" and size == "s20":            # resumed at 9000: two start rows, overlapping evals
                c1 = [(s, 0.3) if s > 9000 else (s, v) for s, v in val_curve(L, rng, stop=12000)]
                c2 = val_curve(L, rng, start=10500)
                rows = ([start_row(1.0, 1.0)] + train_rows(500, 12000) + [val_row(s, v) for s, v in c1]
                        + [start_row(N, D, 9000)] + train_rows(9500, TOTAL) + [val_row(s, v) for s, v in c2])
                write_run(d, N, D, [c for c in c1 if c[0] <= 9000] + c2, rows=rows)
                continue
            overfit = 0.08 if lane == "d25" and size in ("s40", "s80") else 0.0
            write_run(d, N, D, val_curve(L, rng, overfit=overfit), garble=(lane, size) == ("d25", "s5"),
                      final=(lane, size) != ("d50", "s80"), done=(lane, size) != ("d50", "s80"), **kw)
    os.makedirs(os.path.join(stamp, "d100", "s160"))                                # started, no evals yet
    with open(os.path.join(stamp, "d100", "s160", "metrics.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in [start_row(170.0, LANES["d100"])] + train_rows(500, 2000)) + "\n")
    L = truth[("d50", "s5")]                                # duplicate of podA's d50/s5 with fewer evals
    write_run(os.path.join(stamp, "d50", "s5"), SIZES["s5"], LANES["d50"], val_curve(L, rng, stop=6000))
    old = os.path.join(base, "runs", "jobB", "durable", "20261108-0900", "d100", "s5")   # an older snapshot
    write_run(old, SIZES["s5"], LANES["d100"], val_curve(truth[("d100", "s5")], rng, stop=4500))
    os.symlink("durable/20261109-1200", os.path.join(base, "runs", "jobB", "latest"))
    return podA, os.path.join(base, "runs", "jobB", "latest"), truth


_CACHE = {}


def main_result():
    if not _CACHE:
        base = _tmp()
        podA, latest, truth = make_sweep(base)
        out = os.path.join(base, "out")
        t0 = time.time()
        rc = sf.main(["--root", podA, "--root", latest, "--out", out, "--boot", "400", "--seed", "0"])
        _CACHE.update(rc=rc, secs=time.time() - t0, out=out, truth=truth, base=base,
                      res=json.load(open(os.path.join(out, "scaling.json")),
                                    parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))),
                      page=open(os.path.join(out, "report.html")).read())
    return _CACHE


def run_by(res, lane, size):
    return next(r for r in res["runs"] if r["lane"] == lane and r["size"] == size)


def test_inputs_parsed():
    c = main_result()
    res = c["res"]
    assert c["rc"] == 0
    assert len(res["runs"]) == 17, [(r["lane"], r["size"]) for r in res["runs"]]
    assert sum(r["fitted"] for r in res["runs"]) == 12
    r = run_by(res, "d50", "s20")                           # resumed: last start row, later duplicate evals win
    assert r["params"] == int(round(21.6e6)) and r["D"] == LANES["d50"] and r["n_starts"] == 2
    assert r["n_evals"] == TOTAL // EVERY and abs(r["L"] / c["truth"][("d50", "s20")] - 1) < 0.003
    assert run_by(res, "d25", "s5")["garbled_lines"] == 2
    assert any("d25/s5" in w and "unreadable" in w for w in res["warnings"])
    r = run_by(res, "d100", "s160")
    assert r["n_evals"] == 0 and not r["fitted"] and r["L"] is None
    r = run_by(res, "d50", "s80")
    assert not r["done"] and not r["has_final"] and r["fitted"]
    assert run_by(res, "d100", "s40")["path"].endswith("name_x")
    assert any("d50/s5 found twice" in w and "podA" in w.split("dropped")[0] for w in res["warnings"])
    assert run_by(res, "d50", "s5")["n_evals"] == TOTAL // EVERY
    assert run_by(res, "d100", "s5")["n_evals"] == TOTAL // EVERY      # the old durable stamp is not reached
    r = run_by(res, "candor", "s80")
    assert r["still_improving"] and r["candor"] and not r["fitted"]
    assert any("candor/s80" in w and "still improving" in w for w in res["warnings"])
    assert not any(r["still_improving"] for r in res["runs"] if r["lane"] != "candor")
    r = run_by(res, "d25", "s80")                           # overfits at the end: best is before the last eval
    assert r["step_best"] < r["steps"] and r["L_last"] > r["L"] * 1.01


def test_fit_recovers_law():
    res = main_result()["res"]
    f = res["fit"]
    p, ci = f["params"], f["ci"]
    assert abs(p["alpha"] - TRUE["alpha"]) <= 0.08, p
    assert abs(p["beta"] - TRUE["beta"]) <= 0.08, p
    assert abs(p["E"] / TRUE["E"] - 1) <= 0.05, p
    for k in sf.PNAMES:
        assert ci[k][0] <= TRUE[k] <= ci[k][2], (k, ci[k], TRUE[k])
    assert f["n_runs"] == 12 and f["dof"] == 7 and not f["underdetermined"]
    assert f["rms_log_resid"] < 0.005
    b = res["bootstrap"]
    assert b["ok"] + b["failed"] + b["degenerate"] == 400 and b["ok"] >= 350


def test_extrapolation_and_candor():
    res = main_result()["res"]
    ex = res["extrapolation"]
    assert abs(ex["N_next"] - 2 * SIZES["s80"]) < 1e-9 and abs(ex["D_max"] - LANES["d100"]) < 1e-9
    labels = [r["label"] for r in ex["rows"]]
    assert len(ex["rows"]) == 3 and "CANDOR" in labels[1] and "2x" in labels[2]
    for r in ex["rows"]:
        want = true_L(r["N"], r["D"])
        assert r["ci"][0] - 0.01 <= want <= r["ci"][2] + 0.01 and abs(r["L"] / want - 1) < 0.02
        assert r["gain_vs_best"] < 0                         # bigger + more data beats every observed run
    m = res["marginal"]
    d2D = true_L(SIZES["s80"], 2 * LANES["d100"]) - true_L(SIZES["s80"], LANES["d100"])
    d2N = true_L(2 * SIZES["s80"], LANES["d100"]) - true_L(SIZES["s80"], LANES["d100"])
    assert m["dL_2x_data"] < 0 and m["dL_2x_params"] < 0
    assert abs(m["dL_2x_data"] - d2D) < 0.01 and abs(m["dL_2x_params"] - d2N) < 0.01
    cand = {c["run"]: c for c in res["candor"]}
    assert set(cand) == {"candor/" + s for s in SIZES}
    for s in ("s5", "s20", "s40"):                          # converged CANDOR runs beat "more of the same data"
        assert cand["candor/" + s]["resid"] < 0 and -0.035 < cand["candor/" + s]["resid_rel"] < -0.005
    assert len(res["per_axis"]["by_lane"]) == 4 and len(res["per_axis"]["by_size"]) == 4
    for g in res["per_axis"]["by_lane"]:
        assert g["loglog_slope"] < 0 and g["k"] is not None
    assert all(g["k"] is not None and g["loglog_slope"] < 0 for g in res["per_axis"]["by_size"])


def test_report_selfcontained_and_fast():
    c = main_result()
    page = c["page"]
    assert page.count("<svg") >= 6
    for bad in ("http://", "https://", "<script src", "<link", "@import", "url(http"):
        assert bad not in page, bad
    assert "d100" in page and "candor" in page and "Warnings" in page
    assert c["res"]["tool"] == sf.VERSION and c["res"]["generated"]
    assert c["secs"] < 60, f"fit + 400 bootstraps took {c['secs']:.1f} s"


def test_job_dir_root_prefers_latest():
    base = main_result()["base"]
    warn = []
    runs = sf.discover([os.path.join(base, "runs", "jobB")], [], warn)
    r = next(r for r in runs if (r["lane"], r["size"]) == ("d100", "s5"))
    assert r["n_evals"] == TOTAL // EVERY and not any("d100/s5 found twice" in w for w in warn)


def test_degenerate_three_runs_cli():
    base = _tmp()
    rng = np.random.default_rng(3)
    for lane, size in (("d25", "s5"), ("d50", "s20"), ("d100", "s40")):
        N, D = SIZES[size], LANES[lane]
        write_run(os.path.join(base, "sw", lane, size), N, D, val_curve(true_L(N, D), rng))
    out = os.path.join(base, "out")
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "p2_sweep_fit.py"), "--root",
                        os.path.join(base, "sw"), "--out", out, "--boot", "50"], capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr
    res = json.load(open(os.path.join(out, "scaling.json")))
    assert res["fit"]["underdetermined"] and res["fit"]["dof"] < 0
    assert any("underdetermined" in w for w in res["warnings"])
    assert "Joint fit over 3 runs" in p.stdout
    page = open(os.path.join(out, "report.html")).read()
    assert "<svg" in page and "http" not in page


def test_single_run_and_empty_root():
    base = _tmp()
    write_run(os.path.join(base, "sw", "d25", "s5"), 5.7, 69.1, val_curve(0.9, np.random.default_rng(0)))
    out = os.path.join(base, "out")
    assert sf.main(["--root", os.path.join(base, "sw"), "--out", out, "--boot", "10"]) == 0
    res = json.load(open(os.path.join(out, "scaling.json")))
    assert res["fit"] is None and any("joint fit skipped" in w for w in res["warnings"])
    os.makedirs(os.path.join(base, "empty"))
    assert sf.main(["--root", os.path.join(base, "empty"), "--out", out]) == 2


if __name__ == "__main__":
    t0 = time.time()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        t = time.time()
        f()
        print(f"ok  {n}  ({time.time() - t:.1f} s)")
    print(f"{len(tests)} passed in {time.time() - t0:.1f} s")
