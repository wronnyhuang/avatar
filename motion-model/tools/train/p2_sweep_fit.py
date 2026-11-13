"""Scaling-law fit + one-file HTML report for the avatar motion-model scaling sweep.

    python -I tools/train/p2_sweep_fit.py --root <dir> [--root <dir2> ...] [--runs <run_dir> ...] --out <dir>
        [--boot 400] [--seed 0] [--fit-candor] [--next-mult 2] [--next-params 170] [--title "..."] [--min-evals 2]

Runs = every directory holding a metrics.jsonl under each --root (symlinks followed; next to a `latest` snapshot link
its `durable/` siblings are skipped), plus explicit --runs. Lane / size come from cfg.json "sweep" when present, else
the parent / run dir names; candor = lane name contains "candor". Per run: N = params of the LAST start row, D =
train_hours (person-hours of TRAIN data) of that row, L = min val_core over the regular evals (final rows excluded,
evals deduped by step keeping the later row).

Fits (N in M params, D in person-hours): joint L = E + A/N^alpha + B/D^beta, Chinchilla style (Huber 1e-3 on log L,
L-BFGS-B over a grid of inits) on the non-candor runs (--fit-candor: all); run-level bootstrap CIs; per-lane L(N) and
per-size L(D) power laws + log-log slopes; extrapolation to the next size; 2x data vs 2x params at the largest point;
CANDOR runs vs the fit's prediction. Writes <out>/scaling.json and <out>/report.html (inline SVG/CSS, a small inline
tooltip script, no external references).
"""
import argparse
import datetime as _dt
import html
import json
import math
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import numpy as np  # noqa: E402
from scipy.optimize import least_squares, minimize  # noqa: E402

VERSION = "p2_sweep_fit 1.0"
HUBER_DELTA = 1e-3
PNAMES = ("E", "A", "alpha", "B", "beta")
BOUND = (0.01, 3.0)                                   # exponent bounds (joint and per-axis)


# ----------------------------------------------------------------------------------------------------------- inputs
def read_jsonl(path):
    rows, bad = [], 0
    try:
        f = open(path, errors="replace")
    except OSError:
        return rows, 0
    with f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        return None, f"{path}: unreadable ({e.__class__.__name__})"


def load_run(rd):
    """One run dir -> dict (see module doc; its warnings in run["warnings"]). Never raises on bad content."""
    warn = []
    rows, bad = read_jsonl(os.path.join(rd, "metrics.jsonl"))
    cfg, err = _load_json(os.path.join(rd, "cfg.json"))
    if err:
        warn.append(err)
    done, err = _load_json(os.path.join(rd, "DONE"))
    if err:
        warn.append(err)
    sw = (cfg or {}).get("sweep") if isinstance(cfg, dict) else None
    sw = sw if isinstance(sw, dict) else {}
    lane = str(sw.get("lane") or os.path.basename(os.path.dirname(os.path.abspath(rd))))
    size = str(sw.get("size") or os.path.basename(os.path.abspath(rd)))
    candor = bool(sw["candor"]) if "candor" in sw else "candor" in lane.lower()
    starts = [r for r in rows if r.get("event") == "start"]
    st = starts[-1] if starts else {}
    vals, final, max_step, packs = {}, None, 0, []
    for r in rows:
        s = r.get("step")
        if isinstance(s, (int, float)) and math.isfinite(s):
            max_step = max(max_step, int(s))
        if r.get("final"):
            final = r                                  # a different-batch eval: never part of the curve
        elif r.get("event") == "pack_switch":
            packs.append(r)
        elif isinstance(r.get("event"), str):
            continue
        elif "val_core" in r and isinstance(s, (int, float)):
            vals[int(s)] = r                           # dedupe by step, the later row wins (resumes)
    curve = [(s, _num(vals[s].get("val_core"))) for s in sorted(vals)]
    n_nonfinite = sum(v is None for _, v in curve)
    curve = [(s, v) for s, v in curve if v is not None]
    done_d = done if isinstance(done, dict) else {}
    N = _num(st.get("params")) or _num(done_d.get("params"))
    D = _num(st.get("train_hours"))
    run = dict(lane=lane, size=size, frac=_num(sw.get("frac")), candor=candor, path=os.path.abspath(rd),
               real_path=os.path.realpath(rd), params=int(N) if N else None, N=N / 1e6 if N else None, D=D,
               val_hours=_num(st.get("val_hours")), gpu=st.get("gpu"), n_starts=len(starts), garbled_lines=bad,
               n_evals=len(curve), curve=curve, done=done is not None, interrupted=bool(done_d.get("interrupted")),
               minutes=_num(done_d.get("minutes")), steps=int(_num(done_d.get("step")) or max_step),
               stop_reason=(final or {}).get("stop_reason"), from_best_step=(final or {}).get("from_best_step"),
               final_best=_num((final or {}).get("best_val_loss_cmp")),
               final_last=_num((final or {}).get("last_val_loss_cmp")), has_final=final is not None,
               L=None, step_best=None, L_last=None, still_improving=False, fitted=False, excluded=None,
               warnings=warn)
    name = f"{lane}/{size}"
    if bad:
        warn.append(f"{name}: {bad} unreadable line(s) in metrics.jsonl skipped")
    if n_nonfinite:
        warn.append(f"{name}: {n_nonfinite} eval(s) with non-finite val_core ignored")
    if len(starts) > 1:
        warn.append(f"{name}: resumed run ({len(starts)} start rows) - last start row used, evals deduped by step")
    for p in packs:
        h = _num(p.get("train_hours"))
        if h is not None and D is not None and abs(h - D) > 1e-6 * max(1.0, D):
            warn.append(f"{name}: pack_switch at step {p.get('step')} changed train_hours to {h:g} "
                        f"(D = start row's {D:g} kept)")
    if curve:
        v = [c[1] for c in curve]
        i = int(np.argmin(v))
        run.update(L=v[i], step_best=curve[i][0], L_last=v[-1])
        ref = v[max(0, len(v) - 4)]
        run["still_improving"] = len(v) >= 2 and i == len(v) - 1 and (ref - v[-1]) / ref > 0.002
        if run["final_best"] is not None and abs(run["final_best"] - run["L"]) > 1e-4 * run["L"]:
            warn.append(f"{name}: final best_val_loss_cmp {run['final_best']:.5f} != min over evals "
                        f"{run['L']:.5f} (min over evals used)")
    if run["still_improving"]:
        warn.append(f"{name}: still improving at its last eval (step {run['step_best']}) - L is an upper bound "
                    f"of the converged loss")
    if run["interrupted"]:
        warn.append(f"{name}: DONE says interrupted")
    return run


def find_run_dirs(root):
    out, seen = [], set()
    for dp, dns, fns in os.walk(root, followlinks=True):
        real = os.path.realpath(dp)
        if real in seen:
            dns[:] = []
            continue
        seen.add(real)
        if "metrics.jsonl" in fns:
            out.append(dp)
            dns[:] = []                                 # a run dir: bundle/ etc. are not runs
            continue
        if "latest" in dns and "durable" in dns:
            dns.remove("durable")                       # snapshot layout: latest -> durable/<stamp>
        dns[:] = sorted(d for d in dns if d != "bundle" and not d.startswith("."))
    return out


def discover(roots, run_dirs, warn):
    cands = []
    for r in roots:
        if not os.path.isdir(r):
            warn.append(f"root {r}: not a directory")
            continue
        found = find_run_dirs(r)
        if not found:
            warn.append(f"root {r}: no metrics.jsonl found")
        cands += found
    for r in run_dirs:
        if os.path.isfile(os.path.join(r, "metrics.jsonl")):
            cands.append(r)
        else:
            warn.append(f"run {r}: no metrics.jsonl")
    by_real, by_key = set(), {}
    for rd in cands:
        real = os.path.realpath(rd)
        if real in by_real:
            continue                                     # the same run reached twice (a root and its latest link)
        by_real.add(real)
        run = load_run(rd)
        key = (run["lane"], run["size"])
        old = by_key.get(key)
        if old is None:
            by_key[key] = run
            continue
        keep, drop = (run, old) if run["n_evals"] > old["n_evals"] else (old, run)
        warn.append(f"{key[0]}/{key[1]} found twice: kept {keep['path']} ({keep['n_evals']} evals), "
                    f"dropped {drop['path']} ({drop['n_evals']} evals)")
        by_key[key] = keep
    runs = sorted(by_key.values(), key=lambda r: (r["candor"], r["D"] or 0, r["lane"], r["N"] or 0, r["size"]))
    for r in runs:
        warn += r["warnings"]
    return runs


# --------------------------------------------------------------------------------------------------------- joint fit
# theta = [log E, log A_c, alpha, log B_c, beta] on CENTRED logs (lN - cN, lD - cD): log A = log A_c + alpha * cN.
# Centring decorrelates intercept and slope (better-conditioned optimisation); the law itself is unchanged.
def _parts(th, lN, lD, lL):
    t0, t1, t2 = th[..., 0:1], th[..., 1:2] - th[..., 2:3] * lN, th[..., 3:4] - th[..., 4:5] * lD
    m = np.maximum(np.maximum(t0, t1), t2)
    e0, e1, e2 = np.exp(t0 - m), np.exp(t1 - m), np.exp(t2 - m)
    s = e0 + e1 + e2
    return m + np.log(s) - lL, e0 / s, e1 / s, e2 / s              # log residual, softmax weights


def _huber(r, delta):
    """Huber(delta) summed over the last axis, scaled by 1/delta^2 (same minimiser, sane optimiser tolerances)."""
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)).sum(-1) / delta ** 2


def _huber_obj(th, lN, lD, lL, delta):
    r, w0, w1, w2 = _parts(th, lN, lD, lL)
    g = np.where(np.abs(r) <= delta, r, delta * np.sign(r)) / delta ** 2
    g1, g2 = g * w1, g * w2
    return float(_huber(r, delta)), np.array([(g * w0).sum(), g1.sum(), -(g1 * lN).sum(), g2.sum(),
                                              -(g2 * lD).sum()])


def _bounds(lL):
    lmin = lL.min(-1)
    lo = np.stack(np.broadcast_arrays(lmin - 12.0, -30.0, BOUND[0], -30.0, BOUND[0]), -1)
    hi = np.stack(np.broadcast_arrays(lmin - 1e-6, 30.0, BOUND[1], 30.0, BOUND[1]), -1)
    return lo, hi                                                   # E < min(L), exponents in BOUND


def init_grid(lL):
    """The Chinchilla-style grid: alpha, beta x A, B scales (term at the centre N, D = c * min L) x E."""
    m = math.exp(lL.min())
    return [np.array([math.log(fE * m), math.log(cA * m), a, math.log(cB * m), b])
            for a in (0.1, 0.3, 0.6, 1.0) for b in (0.1, 0.3, 0.6, 1.0)
            for cA in (0.03, 0.15, 0.6) for cB in (0.03, 0.15, 0.6) for fE in (0.5, 0.9)]


def fit_joint(lN, lD, lL, inits, delta=HUBER_DELTA):
    """L-BFGS-B from every init -> [(loss, theta)] sorted by loss (lN, lD centred)."""
    lo, hi = _bounds(lL)
    out = []
    with np.errstate(all="ignore"):
        for th0 in inits:
            r = minimize(_huber_obj, np.clip(th0, lo, hi), args=(lN, lD, lL, delta), jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo, hi)), options=dict(maxiter=3000, ftol=1e-12, gtol=1e-8))
            if np.isfinite(r.fun) and np.all(np.isfinite(r.x)):
                out.append((float(r.fun), r.x))
    out.sort(key=lambda t: t[0])
    return out


def lm_batch(lN, lD, lL, TH, delta=HUBER_DELTA, iters=300):
    """Huber-IRLS Levenberg-Marquardt with box projection, vectorised over B problems (bootstrap refits).
    lN, lD, lL: (B, n); TH: (B, 5) inits -> (theta (B, 5), loss (B,)), same objective as fit_joint."""
    lo, hi = _bounds(lL)
    TH = np.clip(np.array(TH, float), lo, hi)
    with np.errstate(all="ignore"):
        r, w0, w1, w2 = _parts(TH, lN, lD, lL)
        f, lam = _huber(r, delta), np.full(len(TH), 1e-3)
        eye = np.eye(5)
        for _ in range(iters):
            W = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-300))
            J = np.stack([w0, w1, -w1 * lN, w2, -w2 * lD], -1)
            JW = J * W[..., None]
            H = np.einsum("bni,bnj->bij", JW, J)
            g = np.einsum("bni,bn->bi", JW, r)
            dg = np.einsum("bii->bi", H)
            A = H + (lam[:, None] * dg + 1e-12 * dg.max(-1, keepdims=True) + 1e-300)[..., None] * eye
            try:
                step = -np.linalg.solve(A, g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = -np.einsum("bij,bj->bi", np.linalg.pinv(A), g)
            T2 = np.clip(TH + step, lo, hi)
            r2, v0, v1, v2 = _parts(T2, lN, lD, lL)
            f2 = _huber(r2, delta)
            ok = np.isfinite(f2) & (f2 < f)
            done = ~ok & (lam > 1e8) | ok & (f - f2 < 1e-12 * np.maximum(f, 1.0))
            TH[ok], f[ok], r[ok], w0[ok], w1[ok], w2[ok] = T2[ok], f2[ok], r2[ok], v0[ok], v1[ok], v2[ok]
            lam = np.where(ok, np.maximum(lam / 3, 1e-9), lam * 4)
            if done.all():
                break
    return TH, f


def th_to_p(th, cN, cD):
    """centred theta -> [E, A, alpha, B, beta] (A, B in M-params / person-hour units)."""
    th = np.asarray(th, float)
    return np.stack([np.exp(th[..., 0]), np.exp(th[..., 1] + th[..., 2] * cN), th[..., 2],
                     np.exp(th[..., 3] + th[..., 4] * cD), th[..., 4]], -1)


def law(P, N, D):
    """P: (5,) or (k, 5) of [E, A, alpha, B, beta]; N in M params, D in person-h."""
    P = np.asarray(P, float)
    return P[..., 0] + P[..., 1] * N ** -P[..., 2] + P[..., 3] * D ** -P[..., 4]


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 50, 97.5])] if x.size else None


def bootstrap(lN, lD, lL, tops, n_boot, seed):
    """Resample runs with replacement; refit every resample from the main fit's distinct optima + 2 grid points
    (batched LM, best per resample). -> (centred thetas (k, 5), failed, degenerate)."""
    rng = np.random.default_rng(seed)
    n = len(lL)
    idx = rng.integers(0, n, (n_boot, n))
    ok = np.array([len(set(lN[i])) >= 2 and len(set(lD[i])) >= 2 for i in idx], bool)
    idx = idx[ok]
    if not len(idx):
        return np.zeros((0, 5)), 0, int((~ok).sum())
    m = math.exp(lL.min())
    inits = [th for _, th in tops] + [np.array([math.log(0.9 * m), math.log(0.15 * m), a, math.log(0.15 * m), a])
                                      for a in (0.3, 0.6)]
    best_t, best_f = None, np.full(len(idx), np.inf)
    for th0 in inits:
        T, f = lm_batch(lN[idx], lD[idx], lL[idx], np.tile(th0, (len(idx), 1)))
        f = np.where(np.isfinite(f) & np.all(np.isfinite(T), -1), f, np.inf)
        better = f < best_f
        best_t = T if best_t is None else np.where(better[:, None], T, best_t)
        best_f = np.where(better, f, best_f)
    good = np.isfinite(best_f)
    return best_t[good], int((~good).sum()), int((~ok).sum())


def distinct_tops(res, k=4):
    out, seen = [], set()
    for f, th in res:
        key = (round(float(th[2]), 2), round(float(th[4]), 2))
        if key not in seen:
            seen.add(key)
            out.append((f, th))
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------------------------------- per-axis fits
def fit_axis(x, L):
    """L(x) = E + C x^-k (bounded least squares on log L) when >= 3 distinct x, and the log-log slope."""
    x, L = np.asarray(x, float), np.asarray(L, float)
    o = np.argsort(x)
    x, L = x[o], L[o]
    out = dict(n=int(len(x)), x=[float(v) for v in x], L=[float(v) for v in L],
               loglog_slope=float(np.polyfit(np.log(x), np.log(L), 1)[0]), E=None, C=None, k=None, rms=None,
               dof=None)
    if len(set(x)) < 3:
        return out
    minL, best = float(L.min()), None
    lo, hi = [0.0, 1e-12, BOUND[0]], [minL * (1 - 1e-9), np.inf, BOUND[1]]
    with np.errstate(all="ignore"):
        for k0 in (0.1, 0.3, 0.6, 1.0):
            for fE in (0.1, 0.5, 0.9):
                E0 = fE * minL
                C0 = max((L.max() - E0) * x.min() ** k0, 1e-9)
                try:
                    r = least_squares(lambda p: np.log(p[0] + p[1] * x ** -p[2]) - np.log(L), [E0, C0, k0],
                                      bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=2000)
                except ValueError:
                    continue
                if np.all(np.isfinite(r.fun)) and (best is None or r.cost < best.cost):
                    best = r
    if best is not None:
        out.update(E=float(best.x[0]), C=float(best.x[1]), k=float(best.x[2]),
                   rms=float(np.sqrt(np.mean(best.fun ** 2))), dof=int(len(x) - 3))
    return out


# ---------------------------------------------------------------------------------------------------------- analysis
def analyse(runs, a, warn):
    for r in runs:
        if r["n_evals"] < a.min_evals:
            r["excluded"] = f"{r['n_evals']} eval(s) < --min-evals {a.min_evals}"
        elif not r["N"] or not r["D"]:
            r["excluded"] = "no params / train_hours in a start row"
        elif r["candor"] and not a.fit_candor:
            r["excluded"] = "CANDOR (held out of the fit)"
        r["fitted"] = r["excluded"] is None
    usable = [r for r in runs if r["L"] is not None and r["N"] and r["D"] and r["n_evals"] >= a.min_evals]
    fit_runs = [r for r in usable if r["fitted"]]
    n_nc = sum(not r["candor"] for r in fit_runs)
    out = dict(fit=None, bootstrap=None, per_axis=dict(by_lane=[], by_size=[]), extrapolation=None, marginal=None,
               candor=[])
    if len(fit_runs) < 3 or len({r["N"] for r in fit_runs}) < 2 or len({r["D"] for r in fit_runs}) < 2:
        warn.append(f"joint fit skipped: {len(fit_runs)} fitted run(s) - needs >= 3 with >= 2 distinct N and D")
    else:
        lN, lD = np.log([r["N"] for r in fit_runs]), np.log([r["D"] for r in fit_runs])
        lL = np.log([r["L"] for r in fit_runs])
        cN, cD = float(lN.mean()), float(lD.mean())
        lN, lD = lN - cN, lD - cD
        t0 = time.time()
        res = fit_joint(lN, lD, lL, init_grid(lL))
        tops = distinct_tops(res)
        P = th_to_p(tops[0][1], cN, cD)
        resid = lL - np.log(law(P, np.exp(lN + cN), np.exp(lD + cD)))
        n, dof = len(fit_runs), len(fit_runs) - 5
        fit = dict(law="L = E + A / N^alpha + B / D^beta", units=dict(N="M params", D="train person-hours"),
                   params={k: float(v) for k, v in zip(PNAMES, P)}, n_runs=n, dof=dof,
                   rms_log_resid=float(np.sqrt(np.mean(resid ** 2))), huber_delta=HUBER_DELTA,
                   huber_loss=float(tops[0][0] * HUBER_DELTA ** 2), n_inits=len(res),
                   underdetermined=n_nc < 6 or dof <= 0, fit_candor=bool(a.fit_candor),
                   other_optima=[dict(loss=float(f * HUBER_DELTA ** 2),
                                      **{k: float(v) for k, v in zip(PNAMES, th_to_p(th, cN, cD))})
                                 for f, th in tops[1:]])
        if fit["underdetermined"]:
            warn.append(f"joint fit underdetermined: {n_nc} non-candor run(s), {dof} dof for 5 parameters")
        for k, i in (("alpha", 2), ("beta", 4)):
            if min(abs(P[i] - BOUND[0]), abs(P[i] - BOUND[1])) < 1e-4:
                warn.append(f"joint fit: {k} at its bound ({P[i]:.3g})")
        G = np.array(init_grid(lL))                  # the bootstrap optimiser on the full data, from the whole grid
        Tl, fl = lm_batch(*(np.repeat(v[None], len(G), 0) for v in (lN, lD, lL)), G)
        j = int(np.nanargmin(np.where(np.isfinite(fl), fl, np.nan)))
        fit["lm_check"] = dict(loss=float(fl[j] * HUBER_DELTA ** 2),
                               **{k: float(v) for k, v in zip(PNAMES, th_to_p(Tl[j], cN, cD))})
        if not fit["underdetermined"] and (abs(Tl[j][2] - tops[0][1][2]) > 0.01 or
                                           abs(Tl[j][4] - tops[0][1][4]) > 0.01):
            warn.append(f"bootstrap optimiser (batched LM) lands elsewhere than L-BFGS-B on the full data: alpha "
                        f"{Tl[j][2]:.3f} vs {tops[0][1][2]:.3f}, beta {Tl[j][4]:.3f} vs {tops[0][1][4]:.3f}")
        T, failed, degen = bootstrap(lN, lD, lL, tops, a.boot, a.seed)
        S = th_to_p(T, cN, cD).reshape(-1, 5)
        fit["ci"] = {k: pct(S[:, i]) if len(S) else None for i, k in enumerate(PNAMES)}
        out["bootstrap"] = dict(n=a.boot, ok=int(len(S)), failed=failed, degenerate=degen, seed=a.seed,
                                seconds=round(time.time() - t0, 2))
        nD = len({round(float(v), 6) for v in lD})
        if nD < 5 and len(S):
            warn.append(f"bootstrap CIs resample {n} runs over {nD} distinct D values: the E / B / beta intervals "
                        f"understate the uncertainty (E, B, beta trade off along D); read them as a lower bound")
        if failed or degen:
            warn.append(f"bootstrap: {failed} failed + {degen} degenerate (< 2 distinct N or D) of {a.boot} resamples")
        out["fit"] = fit

        def q(N, D):
            v = float(law(P, N, D))
            return v, (pct(law(S, N, D)) if len(S) else None)

        for r in usable:
            r["L_pred"] = float(law(P, r["N"], r["D"]))
            r["resid_rel"] = r["L"] / r["L_pred"] - 1
        # extrapolation
        N_max = max(r["N"] for r in fit_runs)
        nc = [r for r in fit_runs if not r["candor"]] or fit_runs
        D_max = max(r["D"] for r in nc)
        N_next = a.next_params if a.next_params else a.next_mult * N_max
        best = min(fit_runs, key=lambda r: r["L"])
        cand_D = sorted({round(r["D"], 6) for r in usable if r["candor"]})
        scen = [("next size, largest data", N_next, D_max)]
        scen += [(f"next size, CANDOR data ({d:g} h)", N_next, d) for d in cand_D]
        scen += [("next size, 2x largest data", N_next, 2 * D_max)]
        rows = []
        for label, N, D in scen:
            v, ci = q(N, D)
            at_D = [r for r in usable if abs(r["D"] - D) < 1e-6 * D]
            bD = min(at_D, key=lambda r: r["L"]) if at_D else None
            rows.append(dict(label=label, N=N, D=D, L=v, ci=ci, gain_vs_best=v - best["L"],
                             gain_ci=[c - best["L"] for c in ci] if ci else None,
                             best_at_D=f"{bD['lane']}/{bD['size']}" if bD else None,
                             gain_vs_best_at_D=v - bD["L"] if bD else None))
        out["extrapolation"] = dict(N_next=N_next, N_max=N_max, D_max=D_max,
                                    best_run=f"{best['lane']}/{best['size']}", best_L=best["L"], rows=rows)
        base = law(S, N_max, D_max) if len(S) else None
        d2D, d2N = float(law(P, N_max, 2 * D_max) - law(P, N_max, D_max)), \
            float(law(P, 2 * N_max, D_max) - law(P, N_max, D_max))
        out["marginal"] = dict(N=N_max, D=D_max, L=float(law(P, N_max, D_max)), dL_2x_data=d2D, dL_2x_params=d2N,
                               dL_2x_data_ci=pct(law(S, N_max, 2 * D_max) - base) if len(S) else None,
                               dL_2x_params_ci=pct(law(S, 2 * N_max, D_max) - base) if len(S) else None,
                               p_data_beats_params=float(np.mean(law(S, N_max, 2 * D_max) < law(S, 2 * N_max, D_max)))
                               if len(S) else None)
        for r in usable:
            if r["candor"]:
                v, ci = q(r["N"], r["D"])
                out["candor"].append(dict(run=f"{r['lane']}/{r['size']}", N=r["N"], D=r["D"], L=r["L"], L_pred=v,
                                          ci=ci, resid=r["L"] - v, resid_rel=r["L"] / v - 1,
                                          resid_ci=[r["L"] - c for c in ci[::-1]] if ci else None,
                                          fitted=r["fitted"], still_improving=r["still_improving"]))
    # per-axis fits
    for lane in sorted({r["lane"] for r in usable}, key=lambda l: _lane_key(usable, l)):
        g = [r for r in usable if r["lane"] == lane]
        if len({r["N"] for r in g}) >= 2:
            f = fit_axis([r["N"] for r in g], [r["L"] for r in g])
            out["per_axis"]["by_lane"].append(dict(group=lane, D=float(np.median([r["D"] for r in g])),
                                                   candor=g[0]["candor"], **f))
    for size in sorted({r["size"] for r in usable}, key=lambda s: _size_key(usable, s)):
        g = [r for r in usable if r["size"] == size and (a.fit_candor or not r["candor"])]
        if len({r["D"] for r in g}) >= 2:
            f = fit_axis([r["D"] for r in g], [r["L"] for r in g])
            out["per_axis"]["by_size"].append(dict(group=size, N=float(np.median([r["N"] for r in g])), **f))
    return out


def _lane_key(rs, lane):
    g = [r for r in rs if r["lane"] == lane]
    return (g[0]["candor"], float(np.median([r["D"] or 0 for r in g])), lane)


def _size_key(rs, size):
    g = [r for r in rs if r["size"] == size]
    return (float(np.median([r["N"] or 0 for r in g])), size)


# ------------------------------------------------------------------------------------------------------- formatting
def f4(x):
    return "-" if x is None else f"{x:.4f}"


def fp(v):
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.5g}"


def fci(ci, d=4):
    return "" if not ci else f"[{ci[0]:.{d}f}, {ci[2]:.{d}f}]"


def headline(res):
    fit = res["fit"]
    if not fit:
        return "No joint fit (too few usable runs); see the run table and warnings."
    p, ci = fit["params"], fit["ci"]
    c = lambda k, d: f" {fci(ci[k], d)}" if ci.get(k) else ""  # noqa: E731
    s = (f"Joint fit over {fit['n_runs']} runs ({fit['dof']} dof, RMS log residual {100 * fit['rms_log_resid']:.2f}%"
         f"{', UNDERDETERMINED' if fit['underdetermined'] else ''}): "
         f"L = {p['E']:.3f}{c('E', 3)} + {p['A']:.3g}/N^{p['alpha']:.3f}{c('alpha', 3)} + "
         f"{p['B']:.3g}/D^{p['beta']:.3f}{c('beta', 3)} (N in M params, D in train person-hours; 95% bootstrap CIs).")
    ex = res["extrapolation"]
    r0 = ex["rows"][0]
    s += (f" Next size {r0['N']:.1f} M at {r0['D']:.1f} h: predicted L = {r0['L']:.4f} {fci(r0['ci'])}, "
          f"{r0['gain_vs_best']:+.4f} vs the best observed run ({ex['best_run']}, {ex['best_L']:.4f}).")
    m = res["marginal"]
    s += (f" At {m['N']:.1f} M / {m['D']:.1f} h, 2x data changes L by {m['dL_2x_data']:+.4f} "
          f"{fci(m['dL_2x_data_ci'])} vs {m['dL_2x_params']:+.4f} {fci(m['dL_2x_params_ci'])} for 2x params"
          f" ({'data' if abs(m['dL_2x_data']) > abs(m['dL_2x_params']) else 'params'} worth more"
          + (f", {100 * m['p_data_beats_params']:.0f}% of bootstrap fits favour data" if m['p_data_beats_params']
             is not None else "") + ").")
    if res["candor"]:
        rr = res["candor"]
        mean = float(np.mean([c_["resid_rel"] for c_ in rr]))
        s += (f" CANDOR ({len(rr)} run{'s' if len(rr) > 1 else ''}{', fitted' if rr[0]['fitted'] else ''}): "
              f"observed vs the fit at the same N, D: mean {100 * mean:+.2f}% "
              f"({'better' if mean < 0 else 'worse'} than more of the same data would give).")
    return s


def flags(r):
    f = ["DONE" if r["done"] else "no DONE"]
    if r["interrupted"]:
        f.append("interrupted")
    if r["n_starts"] > 1:
        f.append(f"resumed x{r['n_starts'] - 1}")
    if r["still_improving"]:
        f.append("still improving")
    if r["stop_reason"]:
        f.append(str(r["stop_reason"]))
    if r["excluded"] and not r["excluded"].startswith("CANDOR"):
        f.append("not fitted: " + r["excluded"])
    return ", ".join(f)


def text_summary(res, runs):
    lines = [headline(res), "",
             f"{'lane':<10}{'size':<8}{'N(M)':>8}{'D(h)':>8}{'L_best':>9}{'@step':>8}{'steps':>8}{'evals':>6}  flags"]
    for r in runs:
        lines.append(f"{r['lane']:<10}{r['size']:<8}{(r['N'] or 0):>8.2f}{(r['D'] or 0):>8.1f}{f4(r['L']):>9}"
                     f"{(r['step_best'] or 0):>8}{r['steps']:>8}{r['n_evals']:>6}  {flags(r)}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- SVG pieces
E_ = html.escape
SLOTS = 8


def svar(i):
    return f"var(--s{i + 1})" if i < SLOTS else "var(--other)"


def log_ticks(lo, hi):
    for ms in ((1, 2, 5), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
        t = [m * 10.0 ** k for k in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1)
             for m in ms if lo <= m * 10.0 ** k <= hi]
        if len(t) >= 3:
            return t
    return t


def lin_ticks(lo, hi, n=5):
    raw = (hi - lo) / n or 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * 0.75)
    k = math.ceil(lo / step - 1e-9)
    t = []
    while k * step <= hi + 1e-9 * step:
        t.append(round(k * step, 12))
        k += 1
    return t, step


def tick_fmt(v, step=None):
    if step is not None:
        d = max(0, -math.floor(math.log10(step) + 1e-9) + (1 if round(step / 10 ** math.floor(math.log10(step)), 3)
                                                        == 2.5 else 0))
        return f"{v:.{d}f}"
    return f"{v:g}"


def kfmt(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


class Plot:
    """One SVG chart: scales, axes, clipped data layer."""
    _n = 0

    def __init__(self, w, h, xlim, ylim, xlog=False, m=(14, 92, 46, 58)):   # top, right, bottom, left
        Plot._n += 1
        self.id = f"c{Plot._n}"
        self.w, self.h, self.xlog = w, h, xlog
        self.t, self.r, self.b, self.l = m
        self.x0, self.x1 = (math.log(xlim[0]), math.log(xlim[1])) if xlog else xlim
        self.y0, self.y1 = ylim
        self.parts = []

    def sx(self, x):
        x = math.log(x) if self.xlog else x
        return self.l + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.l - self.r)

    def sy(self, y):
        return self.h - self.b - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.t - self.b)

    def axes(self, xlabel, ylabel, xticks, yticks, xfmt=tick_fmt, ystep=None):
        p, L, R, T, B = self.parts, self.l, self.w - self.r, self.t, self.h - self.b
        for y in yticks:
            Y = self.sy(y)
            p.append(f'<line class="gl" x1="{L}" x2="{R}" y1="{Y:.1f}" y2="{Y:.1f}"/>'
                     f'<text class="tk" x="{L - 6}" y="{Y + 4:.1f}" text-anchor="end">{tick_fmt(y, ystep)}</text>')
        for x in xticks:
            X = self.sx(x)
            p.append(f'<line class="gl" x1="{X:.1f}" x2="{X:.1f}" y1="{T}" y2="{B}"/>'
                     f'<text class="tk" x="{X:.1f}" y="{B + 16}" text-anchor="middle">{xfmt(x)}</text>')
        p.append(f'<line class="ax" x1="{L}" x2="{R}" y1="{B}" y2="{B}"/>'
                 f'<text class="al" x="{(L + R) / 2:.1f}" y="{self.h - 8}" text-anchor="middle">{E_(xlabel)}</text>'
                 f'<text class="al" transform="translate(14 {(T + B) / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle">{E_(ylabel)}</text>'
                 f'<clipPath id="{self.id}"><rect x="{L}" y="{T}" width="{R - L}" height="{B - T}"/></clipPath>')

    def line(self, pts, color, cls="ln", op=1.0):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.sx(x):.1f},{self.sy(y):.1f}" for i, (x, y) in enumerate(pts))
        self.parts.append(f'<path class="{cls}" clip-path="url(#{self.id})" d="{d}" style="stroke:{color}"'
                          + (f' opacity="{op}"' if op < 1 else "") + "/>")

    def mark(self, x, y, color, shape="dot", tip=None, ci=None):
        X, Y = self.sx(x), self.sy(y)
        s = ""
        if ci:
            y0, y1 = self.sy(ci[0]), self.sy(ci[1])
            s += (f'<path class="ci" d="M{X:.1f},{y0:.1f}V{y1:.1f}M{X - 4:.1f},{y0:.1f}h8M{X - 4:.1f},{y1:.1f}h8" '
                  f'style="stroke:{color}"/>')
        if shape == "dot":
            s += f'<circle class="mk" cx="{X:.1f}" cy="{Y:.1f}" r="5" style="fill:{color}"/>'
        elif shape == "diamond":
            s += (f'<path class="mk" d="M{X:.1f},{Y - 7:.1f}L{X + 7:.1f},{Y:.1f}L{X:.1f},{Y + 7:.1f}'
                  f'L{X - 7:.1f},{Y:.1f}Z" style="fill:{color}"/>')
        else:                                                                        # prediction: hollow
            s += f'<circle class="pr" cx="{X:.1f}" cy="{Y:.1f}" r="4.5" style="stroke:{color}"/>'
        if tip:
            s = (f'<g class="hit" tabindex="0" data-tip="{E_(tip)}"><circle cx="{X:.1f}" cy="{Y:.1f}" r="12" '
                 f'fill="transparent"/>{s}</g>')
        self.parts.append(s)

    def end_labels(self, items):
        """items: [(x, y, text)] -> labels right of the plot; collisions spread with leader lines."""
        if not items:
            return
        R = self.w - self.r
        pos = sorted(([self.sy(y), self.sy(y), self.sx(x), t] for x, y, t in items), key=lambda v: v[0])
        gap, top, bot = 13, self.t + 4, self.h - self.b
        for i in range(1, len(pos)):
            pos[i][1] = max(pos[i][1], pos[i - 1][1] + gap)
        over = pos[-1][1] - bot
        if over > 0:
            for p in pos:
                p[1] -= over
            for i in range(len(pos) - 2, -1, -1):
                pos[i][1] = min(pos[i][1], pos[i + 1][1] - gap)
        for p in pos:
            p[1] = max(p[1], top)
        for y_true, y_lab, x_end, t in pos:
            if abs(y_lab - y_true) > 1.5 or x_end < R - 2:
                self.parts.append(f'<path class="ld" d="M{x_end + 6:.1f},{y_true:.1f}L{R + 4},{y_lab:.1f}"/>')
            self.parts.append(f'<text class="dl" x="{R + 7}" y="{y_lab + 4:.1f}">{E_(t)}</text>')

    def svg(self, label):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{E_(label)}">'
                + "".join(self.parts) + "</svg>")


def legend(items):
    """items: [(color, text, kind)] kind in line|dot|diamond|hollow|bar."""
    out = []
    for color, text, kind in items:
        if kind == "line":
            k = f'<svg viewBox="0 0 18 10" class="lk"><path d="M1,5H17" class="ln" style="stroke:{color}"/></svg>'
        elif kind == "diamond":
            k = f'<svg viewBox="0 0 14 14" class="lk"><path d="M7,1L13,7L7,13L1,7Z" style="fill:{color}"/></svg>'
        elif kind == "hollow":
            k = (f'<svg viewBox="0 0 14 14" class="lk"><circle cx="7" cy="7" r="4.5" class="pr" '
                 f'style="stroke:{color}"/></svg>')
        elif kind == "bar":
            k = (f'<svg viewBox="0 0 14 14" class="lk"><rect x="1" y="3" width="12" height="8" rx="2" '
                 f'style="fill:{color}"/></svg>')
        else:
            k = f'<svg viewBox="0 0 14 14" class="lk"><circle cx="7" cy="7" r="5" style="fill:{color}"/></svg>'
        out.append(f'<span class="li">{k}{E_(text)}</span>')
    return '<div class="lg">' + "".join(out) + "</div>"


def _yrange(vals, pad=0.06):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    lo, hi = min(vals), max(vals)
    sp = (hi - lo) or abs(hi) * 0.05 or 0.1
    return lo - pad * sp, hi + pad * sp


def run_tip(r):
    t = f"L {r['L']:.4f}|{r['lane']} / {r['size']}|N {r['N']:.1f} M · D {r['D']:.1f} h"
    if r.get("L_pred") is not None:
        t += f"|fit {r['L_pred']:.4f} ({100 * r['resid_rel']:+.2f}%)"
    return t + f"|best @ step {r['step_best']} of {r['steps']}" + ("|still improving" if r["still_improving"] else "")


# ----------------------------------------------------------------------------------------------------------- charts
# Each chart renders twice (wide for desktop, narrow for phones, swapped by a CSS media query) so the text stays
# readable instead of shrinking with the viewBox.
WIDE, NARROW = dict(w=640, h=380, m=(14, 116, 46, 58)), dict(w=360, h=330, m=(12, 66, 44, 48))


def dual(fn, *args):
    leg, wide = fn(*args, narrow=False)
    _, narrow = fn(*args, narrow=True)
    return leg + f'<div class="wide">{wide}</div><div class="narrow">{narrow}</div>'


def _pred_tip(er):
    return f"L {er['L']:.4f} {fci(er['ci'])}|predicted: {er['label']}|N {er['N']:.1f} M · D {er['D']:.1f} h"


def chart_LN(res, runs, lanes, lane_col, narrow=False):
    P = np.array([res["fit"]["params"][k] for k in PNAMES])
    ex = res["extrapolation"]
    us = [r for r in runs if r.get("L_pred") is not None]
    Ns = [r["N"] for r in us]
    xlim = (min(Ns) / 1.3, ex["N_next"] * 1.3)
    yv = [r["L"] for r in us] + [v for r in ex["rows"] for v in (r["ci"] or [r["L"]])]
    lay = NARROW if narrow else WIDE
    pl = Plot(lay["w"], lay["h"], xlim, _yrange(yv), xlog=True, m=lay["m"])
    yt, ys = lin_ticks(pl.y0, pl.y1)
    pl.axes("N (M params, log)" if narrow else "model size N (M params, log scale)",
            "best val_core" if narrow else "best val_core (lower is better)", log_ticks(*xlim), yt, ystep=ys)
    labels, xs = [], np.exp(np.linspace(math.log(min(Ns) / 1.15), math.log(ex["N_next"] * 1.1), 80))
    curves = [(lane, lane_col[lane], float(np.median([r["D"] for r in us if r["lane"] == lane]))) for lane in lanes]
    curves.append(("2x data", "var(--muted)", 2 * ex["D_max"]))
    for lane, col, D in curves:
        pts = list(zip(xs, law(P, xs, D)))
        k = int(np.sum(xs <= ex["N_max"] * 1.001))
        pl.line(pts[:k], col)
        pl.line(pts[max(k - 1, 0):], col, op=0.45)                          # beyond the largest run: extrapolated
        er = next((r for r in ex["rows"] if abs(r["D"] - D) <= 1e-6 * D), None)
        if er:
            pl.mark(er["N"], er["L"], col, "hollow", ci=er["ci"], tip=_pred_tip(er))
        labels.append((*pts[-1], lane if narrow else f"{lane} · {D:.3g} h"))
    for r in us:
        pl.mark(r["N"], r["L"], lane_col[r["lane"]], "diamond" if r["candor"] else "dot", tip=run_tip(r))
    pl.end_labels(labels)
    leg = [(c[1], f"{c[0]} ({c[2]:.3g} h)", "line") for c in curves[:-1]]
    leg += [("var(--muted)", f"fit at 2x data ({curves[-1][2]:.3g} h)", "line"), ("var(--ink2)", "run (fitted)", "dot"),
            ("var(--ink2)", "CANDOR run", "diamond"), ("var(--ink2)", "prediction, 95% CI", "hollow")]
    return legend(leg), pl.svg("best val_core vs model size, one series per data lane, joint-fit curves")


def chart_LD(res, runs, sizes, size_col, fit_candor, narrow=False):
    P = np.array([res["fit"]["params"][k] for k in PNAMES])
    ex = res["extrapolation"]
    us = [r for r in runs if r.get("L_pred") is not None]
    Ds = [r["D"] for r in us] + [2 * ex["D_max"]]
    xlim = (min(Ds) / 1.25, max(Ds) * 1.25)
    yv = [r["L"] for r in us] + [v for r in ex["rows"] for v in (r["ci"] or [r["L"]])]
    lay = NARROW if narrow else WIDE
    pl = Plot(lay["w"], lay["h"], xlim, _yrange(yv), xlog=True, m=lay["m"])
    yt, ys = lin_ticks(pl.y0, pl.y1)
    pl.axes("D (train person-hours, log)" if narrow else "train data D (person-hours, log scale)",
            "best val_core" if narrow else "best val_core (lower is better)", log_ticks(*xlim), yt, ystep=ys)
    xs = np.exp(np.linspace(math.log(min(Ds) / 1.12), math.log(max(Ds) * 1.1), 80))
    labels, cur = [], [(s, size_col[s], float(np.median([r["N"] for r in us if r["size"] == s]))) for s in sizes]
    cur.append(("next", "var(--muted)", ex["N_next"]))
    for s, col, N in cur:
        y = law(P, N, xs)
        pl.line(list(zip(xs, y)), col)
        labels.append((xs[-1], float(y[-1]), s if narrow else f"{s} · {N:.3g} M"))
    for er in ex["rows"]:
        pl.mark(er["D"], er["L"], "var(--muted)", "hollow", ci=er["ci"], tip=_pred_tip(er))
    for r in us:
        pl.mark(r["D"], r["L"], size_col[r["size"]], "diamond" if r["candor"] else "dot", tip=run_tip(r))
    pl.end_labels(labels)
    leg = [(c[1], f"{c[0]} ({c[2]:.3g} M)", "line") for c in cur[:-1]]
    leg += [("var(--muted)", f"next size ({ex['N_next']:.3g} M)", "line"), ("var(--ink2)", "run (fitted)", "dot"),
            ("var(--ink2)", "CANDOR run" + ("" if fit_candor else " (not fitted)"), "diamond"),
            ("var(--ink2)", "prediction, 95% CI", "hollow")]
    return legend(leg), pl.svg("best val_core vs train data, one series per model size, joint-fit curves")


def chart_curves(runs, lanes, sizes, size_col):
    rs = [r for r in runs if r["curve"]]
    if not rs:
        return "<p>No evals yet.</p>"
    smax = max(r["curve"][-1][0] for r in rs)
    bests = [r["L"] for r in rs]
    lo = min(bests)
    ylim = _yrange([lo, max(max(bests) + 0.6 * (max(bests) - lo), lo * 1.08)], 0.04)
    panels = []
    for lane in lanes:
        g = [r for r in rs if r["lane"] == lane]
        if not g:
            continue
        pl = Plot(300, 220, (0, smax * 1.02), ylim, m=(10, 46, 40, 46))
        yt, ys = lin_ticks(*ylim, n=4)
        pl.axes("step", "val_core", lin_ticks(0, smax, n=3)[0], yt, xfmt=kfmt, ystep=ys)
        labels = []
        for r in sorted(g, key=lambda r: _size_key(rs, r["size"])):
            col = size_col.get(r["size"], "var(--other)")
            pl.line(r["curve"], col)
            for s, v in r["curve"]:
                if ylim[0] <= v <= ylim[1]:
                    pl.parts.append(f'<g class="hit" data-tip="val_core {v:.4f}|{E_(r["size"])} · step {s}">'
                                    f'<circle cx="{pl.sx(s):.1f}" cy="{pl.sy(v):.1f}" r="7" fill="transparent"/></g>')
            pl.mark(r["step_best"], r["L"], col, tip=f"best {r['L']:.4f}|{r['lane']} / {r['size']}|step "
                                                     f"{r['step_best']} of {r['steps']}")
            s_end, v_end = r["curve"][-1]
            labels.append((s_end, min(max(v_end, ylim[0]), ylim[1]), r["size"]))
        pl.end_labels(labels)
        D = float(np.median([r["D"] or 0 for r in g]))
        panels.append(f'<div class="sm"><div class="smt">{E_(lane)} · {D:.3g} h</div>'
                      + pl.svg(f"val_core vs step, lane {lane}") + "</div>")
    leg = legend([(size_col[s], s, "line") for s in sizes] + [("var(--ink2)", "best eval", "dot")])
    return leg + '<div class="smg">' + "".join(panels) + "</div>"


def chart_resid(runs, narrow=False):
    rs = [r for r in runs if r.get("L_pred") is not None]
    vals = [100 * r["resid_rel"] for r in rs]
    rowh, top = 24, 10
    w, (r_, l_) = (360, (40, 84)) if narrow else (640, (56, 110))
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    pad = max(hi - lo, 0.05) * 54 / max(w - r_ - l_ - 108, 60)                 # room for the value labels
    pl = Plot(w, top + rowh * len(rs) + 46, (lo - pad, hi + pad), (0, 1), m=(top, r_, 46, l_))
    pl.sy = lambda i: top + rowh * (i + 0.5)                                         # row index -> y
    xt, xs = lin_ticks(lo - pad, hi + pad, n=4 if narrow else 6)
    for x in xt:
        X = pl.sx(x)
        pl.parts.append(f'<line class="gl" x1="{X:.1f}" x2="{X:.1f}" y1="{top}" y2="{pl.h - pl.b}"/>'
                        f'<text class="tk" x="{X:.1f}" y="{pl.h - pl.b + 16}" text-anchor="middle">'
                        f'{tick_fmt(x, xs)}</text>')
    X0 = pl.sx(0)
    pl.parts.append(f'<line class="ax" x1="{X0:.1f}" x2="{X0:.1f}" y1="{top}" y2="{pl.h - pl.b}"/>'
                    f'<text class="al" x="{(pl.l + pl.w - pl.r) / 2:.1f}" y="{pl.h - 8}" text-anchor="middle">'
                    + ("observed / fit - 1 (%)" if narrow else
                       "observed / fit - 1 (%) · negative = better than the fit")
                    + "</text>")
    for i, (r, v) in enumerate(zip(rs, vals)):
        Y, X1 = pl.sy(i), pl.sx(v)
        col = "var(--s1)" if r["candor"] else "var(--muted)"
        bw, h = X1 - X0, 12
        rad = min(4.0, abs(bw) / 2, h / 2)
        sgn = 1 if bw >= 0 else 0
        xe = X1 - (rad if sgn else -rad)
        d = (f"M{X0:.1f},{Y - h / 2:.1f}H{xe:.1f}A{rad:.1f},{rad:.1f} 0 0 {sgn} {X1:.1f},{Y - h / 2 + rad:.1f}"
             f"V{Y + h / 2 - rad:.1f}A{rad:.1f},{rad:.1f} 0 0 {sgn} {xe:.1f},{Y + h / 2:.1f}H{X0:.1f}Z")
        pl.parts.append(f'<g class="hit" tabindex="0" data-tip="{E_(run_tip(r))}"><rect x="{min(X0, X1) - 4:.1f}" '
                        f'y="{Y - rowh / 2:.1f}" width="{abs(bw) + 8:.1f}" height="{rowh}" fill="transparent"/>'
                        f'<path d="{d}" style="fill:{col}"/></g>'
                        f'<text class="tk" x="{pl.l - 8}" y="{Y + 4:.1f}" text-anchor="end">'
                        f'{E_(r["lane"])} / {E_(r["size"])}</text>'
                        f'<text class="dl" x="{X1 + (6 if v >= 0 else -6):.1f}" y="{Y + 4:.1f}" '
                        f'text-anchor="{"start" if v >= 0 else "end"}">{v:+.2f}%</text>')
    leg = [("var(--muted)", "fitted run", "bar")]
    if any(r["candor"] for r in rs):
        leg.append(("var(--s1)", "CANDOR run" + ("" if all(r["fitted"] for r in rs if r["candor"]) else
                                                  " (held out: observed vs prediction)"), "bar"))
    return legend(leg), pl.svg("residual of the joint fit per run")


# ------------------------------------------------------------------------------------------------------------ report
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--other:#898781;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:14px}
h1{font-size:21px;margin:6px 0 10px}h2{font-size:17px;margin:22px 0 6px}
p.hl{margin:0 0 8px;color:var(--ink)}p.sub{color:var(--ink2);font-size:13px;margin:4px 0}
figure{background:var(--surface);border:1px solid var(--border);border-radius:10px;margin:12px 0;padding:10px 12px}
figcaption{font-weight:600;margin-bottom:4px}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}
.tk{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.al{fill:var(--ink2);font-size:12px}.dl{fill:var(--ink2);font-size:11px}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ld{fill:none;stroke:var(--axis);stroke-width:1}
.mk{stroke:var(--surface);stroke-width:2}.pr{fill:var(--surface);stroke-width:2}
.ci{fill:none;stroke-width:1.5}
.hit{cursor:pointer;outline:none}.hit:hover .mk,.hit:focus .mk{stroke:var(--ink)}
.hit:hover path,.hit:focus path{opacity:.85}
.lg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.li{display:inline-flex;align-items:center;gap:5px}.lk{width:16px;height:12px;display:inline-block}
.smg{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px}
.wide svg{max-width:820px;margin:0 auto}.narrow{display:none}
@media (max-width:600px){.wide{display:none}.narrow{display:block}main{padding:10px}
figure{padding:8px}}
.smt{font-size:13px;font-weight:600;color:var(--ink2);margin:4px 0 0 4px}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;margin:8px 0}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:5px 8px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}td{font-variant-numeric:tabular-nums}
th:first-child,td:first-child,td.t,th.t{text-align:left}td.t{white-space:normal;min-width:180px}
ul.w{font-size:13px;color:var(--ink2);padding-left:18px}
#tip{position:fixed;z-index:9;pointer-events:none;background:var(--surface);color:var(--ink2);
border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.18);
max-width:280px}
#tip .tv{color:var(--ink);font-weight:600;font-size:13px}
"""

JS = """
(function(){var tip=document.getElementById('tip');
function show(e){var t=e.target.closest&&e.target.closest('[data-tip]');if(!t)return;tip.textContent='';
t.getAttribute('data-tip').split('|').forEach(function(s,i){var d=document.createElement('div');d.textContent=s;
if(!i)d.className='tv';tip.appendChild(d);});tip.hidden=false;var b=t.getBoundingClientRect();
var x=(e.clientX!==undefined&&e.type!=='focusin')?e.clientX:b.left+b.width/2;
var y=(e.clientY!==undefined&&e.type!=='focusin')?e.clientY:b.top;var w=tip.offsetWidth,h=tip.offsetHeight;
tip.style.left=Math.max(4,Math.min(x+12,innerWidth-w-4))+'px';
tip.style.top=Math.max(4,(y-h-10<4?y+16:y-h-10))+'px';}
function hide(e){if(e.target.closest&&e.target.closest('[data-tip]'))tip.hidden=true;}
document.addEventListener('pointerover',show);document.addEventListener('focusin',show);
document.addEventListener('pointerout',hide);document.addEventListener('focusout',hide);})();
"""


def table(head, rows, text_cols=()):
    h = "".join(f'<th class="{"t" if i in text_cols else ""}">{E_(c)}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"t" if i in text_cols else ""}">{E_(str(c))}</td>'
                                 for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def render_html(res, runs, title, fit_candor):
    lanes = sorted({r["lane"] for r in runs if r["L"] is not None}, key=lambda l: _lane_key(runs, l))
    sizes = sorted({r["size"] for r in runs if r["L"] is not None}, key=lambda s: _size_key(runs, s))
    lane_col = {l: svar(i) for i, l in enumerate(lanes)}
    size_col = {s: svar(i) for i, s in enumerate(sizes)}
    Plot._n = 0
    o = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{E_(title)}</title>'
         f'<style>{CSS}</style></head><body><main><h1>{E_(title)}</h1><p class="hl">{E_(headline(res))}</p>'
         f'<p class="sub">Generated {E_(res["generated"])} by {E_(res["tool"])}. L = best val_core over the regular '
         f'evals (final-row eval excluded). N = trainable params (M), D = train person-hours.</p>']
    if res["fit"]:
        o.append('<figure><figcaption>(a) Best val_core vs model size</figcaption>'
                 + dual(chart_LN, res, runs, lanes, lane_col) + "</figure>")
        o.append('<figure><figcaption>(b) Best val_core vs train data</figcaption>'
                 + dual(chart_LD, res, runs, sizes, size_col, fit_candor) + "</figure>")
    o.append('<figure><figcaption>(c) val_core vs step, per data lane</figcaption>'
             + chart_curves(runs, lanes, sizes, size_col) + "</figure>")
    if res["fit"]:
        o.append('<figure><figcaption>(d) Joint-fit residual per run</figcaption>' + dual(chart_resid, runs)
                 + "</figure>")
    o.append("<h2>Runs</h2>")
    o.append(table(["run", "N (M)", "D (h)", "L best", "final best", "L fit", "resid", "step of best",
                    "steps", "evals", "flags"],
                   [[f"{r['lane']} / {r['size']}", f"{r['N']:.2f}" if r["N"] else "-",
                     f"{r['D']:.1f}" if r["D"] else "-",
                     f4(r["L"]), f4(r["final_best"]), f4(r.get("L_pred")),
                     f"{100 * r['resid_rel']:+.2f}%" if r.get("resid_rel") is not None else "-",
                     r["step_best"] if r["step_best"] is not None else "-", r["steps"], r["n_evals"],
                     flags(r) + ("" if r["fitted"] or not r["candor"] else ", held out")] for r in runs],
                   text_cols=(0, 10)))
    if res["fit"]:
        f, b = res["fit"], res["bootstrap"]
        o.append("<h2>Joint fit</h2>")
        o.append(f'<p class="sub">{E_(f["law"])}; N in M params, D in train person-hours. Huber delta '
                 f'{f["huber_delta"]:g} on log L, {f["n_inits"]} inits. {f["n_runs"]} runs, {f["dof"]} dof, RMS log '
                 f'residual {100 * f["rms_log_resid"]:.2f}%{", underdetermined" if f["underdetermined"] else ""}. '
                 f'Bootstrap over runs: {b["ok"]} of {b["n"]} refits ok, {b["failed"]} failed, {b["degenerate"]} '
                 f'degenerate (seed {b["seed"]}; fit + bootstrap {b["seconds"]} s).</p>')
        o.append(table(["param", "fit", "2.5%", "median", "97.5%"],
                       [[k, fp(f["params"][k])] + ([fp(v) for v in f["ci"][k]] if f["ci"][k] else ["-"] * 3)
                        for k in PNAMES]))
        ex, m = res["extrapolation"], res["marginal"]
        o.append("<h2>Extrapolation</h2>")
        o.append(table(["scenario", "N (M)", "D (h)", "L pred", "95% CI", "vs best observed", "vs best at this D"],
                       [[r["label"], f"{r['N']:.1f}", f"{r['D']:.1f}", f4(r["L"]), fci(r["ci"]),
                         f"{r['gain_vs_best']:+.4f}",
                         f"{r['gain_vs_best_at_D']:+.4f} ({r['best_at_D']})" if r["best_at_D"] else "-"]
                        for r in ex["rows"]], text_cols=(0,)))
        o.append(f'<p class="sub">Best observed fitted run: {E_(ex["best_run"])} at {ex["best_L"]:.4f}. Marginal at '
                 f'{m["N"]:.1f} M / {m["D"]:.1f} h (L fit {m["L"]:.4f}): 2x data {m["dL_2x_data"]:+.4f} '
                 f'{fci(m["dL_2x_data_ci"])}, 2x params {m["dL_2x_params"]:+.4f} {fci(m["dL_2x_params_ci"])}'
                 + (f'; 2x data wins in {100 * m["p_data_beats_params"]:.0f}% of bootstrap fits'
                    if m["p_data_beats_params"] is not None else "") + ".</p>")
        if res["candor"]:
            o.append("<h2>CANDOR vs the fit</h2>")
            o.append(table(["run", "N (M)", "D (h)", "L observed", "L predicted", "95% CI", "residual", "rel"],
                           [[c["run"] + (" (fitted)" if c["fitted"] else "") + (" (still improving)" if
                                                                               c["still_improving"] else ""),
                             f"{c['N']:.1f}", f"{c['D']:.1f}", f4(c["L"]), f4(c["L_pred"]), fci(c["ci"]),
                             f"{c['resid']:+.4f}", f"{100 * c['resid_rel']:+.2f}%"] for c in res["candor"]],
                           text_cols=(0,)))
    pa = res["per_axis"]
    if pa["by_lane"] or pa["by_size"]:
        o.append("<h2>Per-axis power laws</h2>")
        rows = [[f"lane {g['group']} (D {g['D']:.0f} h)", "L(N)", g["n"], f"{g['loglog_slope']:+.3f}",
                 f4(g["E"]), f"{g['C']:.4g}" if g["C"] else "-", f"{g['k']:.3f}" if g["k"] else "-",
                 f"{100 * g['rms']:.2f}%" if g["rms"] is not None else "-"] for g in pa["by_lane"]]
        rows += [[f"size {g['group']} (N {g['N']:.1f} M)", "L(D)", g["n"], f"{g['loglog_slope']:+.3f}", f4(g["E"]),
                  f"{g['C']:.4g}" if g["C"] else "-", f"{g['k']:.3f}" if g["k"] else "-",
                  f"{100 * g['rms']:.2f}%" if g["rms"] is not None else "-"] for g in pa["by_size"]]
        o.append(table(["group", "law", "points", "log-log slope", "E", "C", "exponent", "RMS log resid"], rows,
                       text_cols=(0,)))
        o.append('<p class="sub">L(x) = E + C x^-exponent, bounded least squares on log L (3+ points; 3 points = '
                 'exact, 0 dof). Log-log slope = straight-line fit of log L vs log x.</p>')
    o.append("<h2>Warnings</h2>")
    o.append('<ul class="w">' + ("".join(f"<li>{E_(w)}</li>" for w in res["warnings"]) or "<li>none</li>") + "</ul>")
    o.append(f'</main><div id="tip" hidden></div><script>{JS}</script></body></html>')
    return "".join(o)


# --------------------------------------------------------------------------------------------------------------- main
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=[], help="sweep root (<root>/<lane>/<size>/), repeatable")
    ap.add_argument("--runs", nargs="+", default=[], help="explicit run dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-candor", action="store_true")
    ap.add_argument("--next-mult", type=float, default=2.0)
    ap.add_argument("--next-params", type=float, default=None, help="next size in M params (overrides --next-mult)")
    ap.add_argument("--title", default="Avatar motion model - scaling sweep")
    ap.add_argument("--min-evals", type=int, default=2)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    warn = []
    runs = discover(a.root, a.runs, warn)
    if not runs:
        print("p2_sweep_fit: no run dirs (metrics.jsonl) found", file=sys.stderr)
        for w in warn:
            print("  " + w, file=sys.stderr)
        return 2
    res = analyse(runs, a, warn)
    res.update(tool=VERSION, generated=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               inputs=dict(roots=[os.path.abspath(r) for r in a.root], runs=[os.path.abspath(r) for r in a.runs],
                           boot=a.boot, seed=a.seed, fit_candor=a.fit_candor, next_mult=a.next_mult,
                           next_params=a.next_params, min_evals=a.min_evals, title=a.title),
               warnings=warn)
    res["runs"] = [{k: v for k, v in r.items() if k != "curve"} | dict(curve=[[s, v] for s, v in r["curve"]])
                   for r in runs]
    os.makedirs(a.out, exist_ok=True)
    page = render_html(res, runs, a.title, a.fit_candor)
    for name, body in (("scaling.json", json.dumps(clean(res), indent=1, allow_nan=False)), ("report.html", page)):
        tmp = os.path.join(a.out, name + ".tmp")
        with open(tmp, "w") as f:
            f.write(body)
        os.replace(tmp, os.path.join(a.out, name))
    print(text_summary(res, runs))
    if warn:
        print(f"\n{len(warn)} warning(s):\n  " + "\n  ".join(warn))
    print(f"\nwrote {os.path.join(a.out, 'scaling.json')} and {os.path.join(a.out, 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
