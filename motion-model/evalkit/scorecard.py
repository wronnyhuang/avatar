#!/usr/bin/env python3
"""Phase-2 SCORECARD — the fast automatic check run after EVERY training run (phase-2 plan item 6).

  python -I evalkit/scorecard.py --bundle <bundle dir> --tag <name> [--out <dir>] [--no-judge] [--index] [--kw '{..}']
  python -I evalkit/scorecard.py --system rules|gru-prior|copy-delay|human|a2f|rules+a2f-mouth|rules+a2f-page [--tag ..]

Clip set: the 29 Seamless TEST clips of run_all (36 s each; the real avatar-side partner = `human`). Cached inputs
(clips/, ablate_inputs/) and cached baseline traces ($E/runs/rules|human|..) are reused; only the new system is stepped
(systems.py: normal + user-zeroed + user-shuffled), into $E/runs/unified-<tag>/ (a bundle-sha marker guards reuse).

Numbers (space = headline_rig.json's `auto`: rules/gru-prior rig -> human units, canon systems by the data pipeline's FK = the held-out
reference's own space):
  rates       nods / blinks / smiles per min (listening | speaking), gaze-away % (listening | speaking), head speed deg/s
              (listening | speaking), stillness (listening | speaking), head-pitch power log10 in 4-8 Hz and 8-15 Hz
              (the 10 Hz line = a re-plan-every-3 sawtooth), head jerk, headline's human-range |log err| + cells inside
              the held-out people's IQR
  reactions   event-locked excess over chance (0.2-2 s; measures.REACTIONS) + smile x-corr gain; paired clip bootstrap
              of (system - rules)
  respond     output change with the USER zeroed / shuffled (RMS / own SD over head, gaze, brows, smile)
  REALISM     Fréchet distance between per-window feature vectors (4 s windows, 1 s hop: head yaw/pitch/roll SD, head
              speed median/p90 on a 5-frame-smoothed angle, head jitter = log 4-15 Hz yaw+pitch power clamped to the
              people's [median, p99], gaze deviation + away fraction, eye-in-head SD, brow SD, smile mean/SD, counts of
              blinks / nods / smile rises / gaze shifts / brow raises; z-scored by the reference) of the
              system vs held-out Seamless TEST people (the test sessions NOT used by any clip, cut into 36 s chunks
              like the clips), per class (avatar speaking / not), averaged; ZERO = the real partner on the same clips:
              realism = FD(system) - FD(human on the clips). Check row: random held-out human chunks (29, leave-their-
              people-out) score ~0 on the same scale. Lower = more human; rules reported on the same scale.
  canon       (canon systems only) tools/train/evaluate.py's stats on the clips' canon (the model side's definitions)
  judge       blind pairwise vs rules on 8 Seamless clips (pairjudge.py, gemini flash via OpenRouter; renders of the
              system made on demand), <= $0.15 per scorecard; --no-judge skips it
VERDICT "beats rules" = realism better (lower) than rules AND judge naturalness pref > 0.5 AND no reaction lost
(LOST = a reaction rules HAS on these clips — excess > 0 — whose paired delta CI is entirely below 0 AND that the system
keeps at < half the real partner's strength; or user-shuffle sensitivity < half of rules'. "Below rules" alone is
listed for information: rules over-mirrors smiles, the real partner is below rules on smile->smile too).
Without the judge: PASS-PENDING-JUDGE / FAIL. Calibration on these clips: the real partner FAILS the judge leg (pref
0.25 vs rules on the 8 judge clips) — the bar is hard by construction.
Outputs: <out>/scorecard.json + scorecard.md (default $P/scorecards/<tag>/); --index appends the table to INDEX.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

T = Path(os.environ.get("EVAL_T", os.environ.get("MM_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))))
P = T / "p2"
SC_ROOT = Path(os.environ.get("SCORECARD_ROOT", str(P / "scorecards")))
CACHE = SC_ROOT / "_cache"
VERSION = "sc-v2"
WHO = os.environ.get("SCORECARD_WHO", "p2-score")
JUDGE_CAP_EACH = 0.15
JUDGE_CAP_TOTAL = float(os.environ.get("SCORECARD_JUDGE_CAP_TOTAL", "0.50"))   # all scorecard judging (who=WHO)
EST_CALL = 0.012                                                               # refuse a call past the caps
# judge clips: one Seamless clip per user person (= run_all's SBS/judge order; rules + human renders are cached)
JUDGE_IDS = ["sl_S0152_0371_P0216", "sl_S0274_0130_P0370", "sl_S0515_0503_P0371", "sl_S0644_0578_P0299A",
             "sl_S0948_0372_P1040", "sl_S2025_1079_P1281A", "sl_S2090_1010_P1346A", "sl_S2091_1095_P1282A"]
ALIAS = {"prior": "gru-prior", "gru_prior": "gru-prior", "copy_delay": "copy-delay", "copy": "copy-delay"}
A2F_RUNS = ("a2f", "rules+a2f-mouth", "rules+a2f-page")          # evalkit/a2f_face.py
BASELINES = ("rules", "gru-prior", "copy-delay", "human") + A2F_RUNS
RIG_RUNS = ("rules", "gru-prior") + A2F_RUNS     # rig.jsonl systems (no canon)
WIN, HOP, CHUNK = 120, 30, 1080          # realism windows: 4 s, hop 1 s; reference chunks 36 s (= a clip)
FEATS = ["yaw_sd", "pitch_sd", "roll_sd", "head_speed_med", "head_speed_p90", "head_jitter", "gaze_dev", "gaze_away",
         "eye_sd", "brow_sd", "smile_mean", "smile_sd", "n_blink", "n_nod", "n_smile", "n_gshift", "n_brow"]
CLASSES = ("not_speaking", "speaking")
# head_jitter clamped to the people's [median, p99]: smoother than their tracking noise is no fault (floor), and a few
# noisy-tracking sessions must not set the scale (cap); a re-plan sawtooth sits at the cap in every window
CLAMP = {"head_jitter": (50, 99)}


def _kit():
    """import the eval kit AFTER EVAL_MODEL_CODE is set (common reads it at import)."""
    global C, AN, M, HL, mstats, bundles
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import common as C  # noqa: N812
    import analyze as AN  # noqa: N812
    import measures as M  # noqa: N812
    import headline as HL  # noqa: N812
    import mstats  # noqa: F401
    import bundles  # noqa: F401
    globals().update(C=C, AN=AN, M=M, HL=HL, mstats=sys.modules["mstats"], bundles=sys.modules["bundles"])


def log(msg, out=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if out is not None:
        with open(Path(out) / "scorecard.log", "a") as f:
            f.write(line + "\n")


def sh(cmd, out, env=None):
    log("+ " + " ".join(map(str, cmd))[:300], out)
    r = subprocess.run(list(map(str, cmd)), env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"},
                       capture_output=True, text=True)
    with open(Path(out) / "scorecard.log", "a") as f:
        f.write(r.stdout[-4000:] + ("\nSTDERR:\n" + r.stderr[-3000:] if r.returncode else "") + "\n")
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(map(str, cmd))[:300]}\n{r.stderr[-1500:]}")
    return r


def _atomic_pickle(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, p)


# ============================================================================= clips / systems
def seamless_ids(man):
    return AN.clip_ids(man, "seamless")


def trace_file(run, cid):
    d = C.E / "runs" / run / cid
    for fn in ("canon.npz", "rig.jsonl"):
        if (d / fn).exists():
            return d / fn
    return None


def missing_traces(run, ids, ablated=False):
    out = []
    for cid in ids:
        d = C.E / "runs" / run / cid
        if run.split("~")[0] in RIG_RUNS:
            need = ("rig.jsonl",)
        else:
            need = ("canon.npz",) if ablated else ("canon.npz", "rig.jsonl")
        if not all((d / fn).exists() for fn in need):
            out.append(cid)
    return out


def fingerprint(runs, ids):
    """cheap identity of a run's traces (paths + sizes + mtimes) -> cache key for baseline metrics."""
    import hashlib
    h = hashlib.sha256(VERSION.encode())
    for run in runs:
        for cid in ids:
            f = trace_file(run, cid)
            if f is not None:
                st = f.stat()
                h.update(f"{run}/{cid}/{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def guard_bundle_cache(meta, force, out):
    """$E/runs/unified-<tag>/ must hold THIS bundle's traces: a SCORECARD_BUNDLE.json marker (sha + path); traces made by
    run_all before the marker existed are accepted when their canon.npz meta names the same bundle path."""
    rd = C.E / "runs" / meta["run"]
    mk = rd / "SCORECARD_BUNDLE.json"
    want = {"bundle": meta["bundle"], "bundle_sha": meta["bundle_sha"], "code": meta["code"]}
    stale = None
    if mk.exists():
        have = json.loads(mk.read_text())
        if have.get("bundle_sha") != meta["bundle_sha"] or have.get("bundle") != meta["bundle"]:
            stale = f"marker says {have.get('bundle')} sha {have.get('bundle_sha')}"
    elif rd.exists():
        for f in sorted(rd.glob("*/canon.npz"))[:3]:
            try:
                m = json.loads(str(np.load(f)["meta"]))
            except Exception:  # noqa: BLE001
                continue
            if m.get("bundle") != meta["bundle"]:
                stale = f"traces were made from {m.get('bundle')}"
            break
    if stale:
        if not force:
            raise SystemExit(f"{rd} holds another bundle's traces ({stale}); use a new --tag, or --force to re-step")
        ts = time.strftime("%Y%m%d-%H%M%S")
        for d in [rd, rd.with_name(rd.name + "~zero"), rd.with_name(rd.name + "~shuffle"),
                  C.E / "judge/pair" / f"rules__{meta['run']}"]:
            if d.exists():
                d.rename(d.with_name(d.name + f".stale-{ts}"))
                log(f"moved stale {d.name} -> {d.name}.stale-{ts}", out)
    rd.mkdir(parents=True, exist_ok=True)
    C.atomic_json(mk, {**want, "t": time.strftime("%Y-%m-%d %H:%M:%S")})


def step_system(run, meta, ids, jobs, kw, out):
    """step only what is missing (systems.py caches per clip): normal (+ canon->rig), user zeroed, user shuffled."""
    py = str(C.VENV_PY)
    kit = C.KIT
    t0 = time.time()
    if meta is not None:
        env = {"EVAL_MODEL_CODE": meta["code"]}
        base = [py, "-I", kit / "systems.py", "--systems", "unified", "--tag", meta["tag"], "--bundle", meta["bundle"],
                "--kw", json.dumps(kw), "--jobs", str(jobs), "--group", "seamless"]
        for abl in ("", "zero", "shuffle"):
            r = run + (("~" + abl) if abl else "")
            if missing_traces(r, ids, bool(abl)):
                sh(base + (["--ablate", abl] if abl else []), out, env)
    else:
        env = {}
        base = [py, "-I", kit / "systems.py", "--systems", run, "--jobs", str(jobs), "--group", "seamless"]
        for abl in (("",) if run == "human" else ("", "zero", "shuffle")):
            r = run + (("~" + abl) if abl else "")
            if missing_traces(r, ids, bool(abl)):
                sh(base + (["--ablate", abl] if abl else []), out, env)
    miss = {r: missing_traces(r, ids, "~" in r) for r in [run] + ([] if run == "human" else [run + "~zero", run + "~shuffle"])}
    miss = {k: v for k, v in miss.items() if v}
    if miss:
        raise RuntimeError(f"missing traces after stepping: { {k: v[:4] for k, v in miss.items()} }")
    return time.time() - t0


# ============================================================================= realism features
def seg_windows(seg):
    """one segment (clip or reference chunk) -> (n_windows, F) features, (n_windows,) class (1 = avatar speaking)."""
    Mt = seg["M"]
    N = len(Mt)
    if N < WIN:
        return np.zeros((0, len(FEATS)), np.float32), np.zeros(0, np.int8)
    if "_ev" not in seg:
        seg["_ev"] = mstats.events(seg)
    ev = seg["_ev"]
    Cc = mstats.C
    ang = np.nan_to_num(Mt[:, :3].astype(np.float64))
    # speed on a 5-frame box-smoothed angle (true motion; a re-plan sawtooth must not pass for human speed)
    k5 = np.ones(5) / 5
    angs = np.stack([np.convolve(np.pad(ang[:, j], 2, mode="edge"), k5, "valid") for j in range(3)], 1)
    sp = np.linalg.norm(np.gradient(angs, axis=0) * C.FPS, axis=1)
    gz = np.nan_to_num(Mt[:, Cc["gx"]] + Mt[:, Cc["yaw"]]); gzy = np.nan_to_num(Mt[:, Cc["gy"]] - Mt[:, Cc["pitch"]])
    tot = np.hypot(gz - np.median(gz), gzy - np.median(gzy))
    eye = np.nan_to_num(Mt[:, 6:8].astype(np.float64))
    br = np.nan_to_num((Mt[:, Cc["brow_l"]] + Mt[:, Cc["brow_r"]]) / 2).astype(np.float64)
    sm = np.nan_to_num(Mt[:, Cc["smile"]]).astype(np.float64)
    cs = {}
    for k in ("blink", "nod", "smile", "gshift", "brow"):
        e = np.zeros(N + 1)
        t = np.asarray(ev[k], int)
        t = t[(t >= 0) & (t < N)]
        np.add.at(e, t, 1)
        cs[k] = np.concatenate([[0], np.cumsum(e[:N])])
    f_ = np.fft.rfftfreq(WIN, 1 / C.FPS)
    band = (f_ >= 4) & (f_ <= 15)      # jitter band: 4-8 Hz (old runtime) + the 10 Hz re-plan sawtooth
    hann = np.hanning(WIN)
    tt = np.arange(WIN) - (WIN - 1) / 2
    st = seg["st"]
    rows, cls = [], []
    for w0 in range(0, N - WIN + 1, HOP):
        w1 = w0 + WIN
        a = ang[w0:w1]
        pw = 0.0
        for j in (0, 1):                # yaw + pitch, detrended
            p = a[:, j] - a[:, j].mean()
            p = p - tt * (tt @ p) / (tt @ tt)
            pw = pw + np.abs(np.fft.rfft(p * hann)) ** 2 / WIN
        rows.append([np.log1p(a[:, 0].std()), np.log1p(a[:, 1].std()), np.log1p(a[:, 2].std()),
                     np.log1p(np.median(sp[w0:w1])), np.log1p(np.percentile(sp[w0:w1], 90)),
                     np.log10(pw[band].mean() + 1e-7),
                     np.log1p(tot[w0:w1].mean()), float((tot[w0:w1] > 10.0).mean()),
                     np.log1p(eye[w0:w1].std(0).mean()), br[w0:w1].std(), sm[w0:w1].mean(), sm[w0:w1].std()]
                    + [cs[k][w1] - cs[k][w0] for k in ("blink", "nod", "smile", "gshift", "brow")])
        cls.append(1 if (st[w0:w1] == 1).mean() >= 0.5 else 0)
    return np.asarray(rows, np.float32), np.asarray(cls, np.int8)


def clip_sessions(man):
    return sorted({man[c]["session"] for c in seamless_ids(man)})


def reference_windows(man):
    """held-out Seamless TEST people (test sessions NOT used by any clip; both persons), valid head+face runs cut into
    36 s chunks (remainder >= 4 s kept) -> windows. Cached in $SC/_cache/ref_windows_<VERSION>.pkl."""
    p = CACHE / f"ref_windows_{VERSION}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    spec = C.canon_spec()
    S = json.loads((C.DATA / "splits.json").read_text())
    used = set(clip_sessions(man))
    sess = sorted(k.split(":", 1)[1] for k, v in (S.get("sessions") or {}).items()
                  if v == "test" and k.startswith("seamless:") and k.split(":", 1)[1] not in used)
    X, Y, person, chunk = [], [], [], []
    persons, nchunk = [], 0
    for s in sess:
        f = C.DATA / "canon1/seamless" / f"{s}.npz"
        if not f.exists():
            continue
        d = spec.load_session(str(f))
        Pn = d["motion"].shape[0]
        vad = d["audio"][:, :, 41] > 0.5
        for pi in range(Pn):
            pid = d["meta"]["persons"][pi]
            if pid not in persons:
                persons.append(pid)
            mk = d["mask"][pi]
            ok = (mk[:, spec.GROUP_NAMES.index("head")] > 0) & (mk[:, spec.GROUP_NAMES.index("face_upper")] > 0)
            own = M._bridge(vad[pi], int(1.2 * C.FPS))
            oth = vad[1 - pi] if Pn == 2 else np.zeros_like(own)
            st = np.where(own, 1, np.where(oth, 2, 0))
            for a, b in mstats._runs(ok, 300):
                Tb = M.table_from_canon(d["motion"][pi, a:b])
                for c0 in range(0, b - a, CHUNK):
                    c1 = min(b - a, c0 + CHUNK)
                    if c1 - c0 < WIN:
                        continue
                    seg = M.make_seg(Tb[c0:c1].copy(), st[a + c0:a + c1], oth[a + c0:a + c1], own[a + c0:a + c1])
                    w, cl = seg_windows(seg)
                    if len(w):
                        X.append(w); Y.append(cl)
                        person.append(np.full(len(w), persons.index(pid), np.int32))
                        chunk.append(np.full(len(w), nchunk, np.int32)); nchunk += 1
    ref = {"X": np.concatenate(X), "cls": np.concatenate(Y), "person": np.concatenate(person),
           "chunk": np.concatenate(chunk), "persons": persons, "sessions": sess, "feats": FEATS, "version": VERSION}
    ref["mu"] = ref["X"].mean(0)
    ref["sd"] = ref["X"].std(0) + 1e-6
    _atomic_pickle(p, ref)
    return ref


def _moments(Z):
    mu = Z.mean(0)
    S = np.cov(Z, rowvar=False) if len(Z) > 1 else np.zeros((Z.shape[1],) * 2)
    return mu, S + 1e-6 * np.eye(Z.shape[1])


def frechet(m1, m2):
    """|mu1-mu2|^2 + tr(S1 + S2 - 2 (S1 S2)^1/2); tr((S1 S2)^1/2) = sum sqrt eig(S1^1/2 S2 S1^1/2) (symmetric, stable)."""
    mu1, S1 = m1; mu2, S2 = m2
    w, V = np.linalg.eigh(S1)
    A = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr_cm = np.sqrt(np.clip(np.linalg.eigvalsh(A @ S2 @ A), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr_cm)


class Realism:
    """FD of z-scored window features vs the held-out reference, per class; zero = the human on the same clips."""

    def __init__(self, ref, drop=()):
        self.ref = ref
        self.cols = [i for i, k in enumerate(FEATS) if k not in drop]
        self.clamp = {FEATS.index(k): tuple(float(np.percentile(ref["X"][:, FEATS.index(k)], q)) for q in qs)
                      for k, qs in CLAMP.items()}
        X = self._floor(ref["X"])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.Z = ((X - self.mu) / self.sd)[:, self.cols]
        self.rm = {c: _moments(self.Z[ref["cls"] == c]) for c in (0, 1)}

    def _floor(self, X):
        X = np.array(X, np.float64)
        for i, (lo, hi) in self.clamp.items():
            X[:, i] = np.clip(X[:, i], lo, hi)
        return X

    def z(self, X):
        return ((self._floor(X) - self.mu) / self.sd)[:, self.cols]

    def fd(self, W, cls, rm=None):
        rm = rm or self.rm
        out = {}
        for c in (0, 1):
            Zc = self.z(W[cls == c])
            out[c] = frechet(_moments(Zc), rm[c]) if len(Zc) >= 20 else float("nan")
        return out

    def contrib(self, W, cls, top=4):
        """per-feature (Δmean² + ΔSD²) in reference SD units, both classes summed -> the biggest gaps."""
        tot = np.zeros(len(self.cols))
        for c in (0, 1):
            Zc = self.z(W[cls == c]); R = self.Z[self.ref["cls"] == c]
            if len(Zc) < 20:
                continue
            tot += (Zc.mean(0) - R.mean(0)) ** 2 + (Zc.std(0) - R.std(0)) ** 2
        idx = np.argsort(-tot)[:top]
        return [(FEATS[self.cols[i]], round(float(tot[i]), 2)) for i in idx]


def stack_windows(per_clip, pick):
    W = [per_clip[c][0] for c in pick if c in per_clip]
    K = [per_clip[c][1] for c in pick if c in per_clip]
    return (np.concatenate(W), np.concatenate(K)) if W else (np.zeros((0, len(FEATS))), np.zeros(0, np.int8))


# ============================================================================= per-run metrics
def responsiveness(run, ids):
    res = {}
    for kind in ("zero", "shuffle"):
        rows = []
        for cid in ids:
            A = M.system_table(run, cid); B = M.system_table(run + "~" + kind, cid)
            if A is None or B is None:
                continue
            n = min(len(A), len(B)); A, B = np.nan_to_num(A[:n]), np.nan_to_num(B[:n])
            r = []
            for g in ("head", "gaze", "face_upper", "smile"):
                cols = M.GROUPS_T[g]
                sd = A[:, cols].std(0); diff = np.sqrt(((A[:, cols] - B[:, cols]) ** 2).mean(0)); ok = sd > 1e-6
                r.append(float(np.mean(diff[ok] / sd[ok])) if ok.any() else np.nan)
            rows.append(np.nanmean(r))
        if rows:
            res[kind] = {"n": len(rows), "overall": float(np.nanmean(rows)), "ci95": list(M.boot_ci(np.array(rows)))}
    return res


def canon_block(run, ids, man):
    """tools/train/evaluate.py's stats on each clip's canon (mean over clips) — the model side's own definitions."""
    sys.path.insert(0, str(C.WT / "tools/train"))
    try:
        import evaluate as EV
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    rows = []
    for cid in ids:
        f = C.E / "runs" / run / cid / "canon.npz"
        uf = C.E / "clips" / cid / "user_fit.npz"
        if not f.exists():
            continue
        m = np.load(f)["motion"].astype(np.float64)
        N = len(m)
        st, _, _ = M.clip_states(man[cid], N)
        user = np.load(uf)["motion"][:N].astype(np.float64) if uf.exists() else None
        if user is not None and len(user) < N:
            user = None
        rows.append(EV.stats(m, user=user, listen=st != 1))
    if not rows:
        return {}
    keys = ["head_speed", "still_pct", "nods_per_min", "blinks_per_min", "smile_mean", "look_away_listen_pct",
            "look_away_speak_pct", "hp48_log10", "head_jerk", "smile_coupling", "brow_sd"]
    return {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys} | {"n_clips": len(rows)}


def hf_power(segs):
    """median over clips of the head-pitch Welch power (log10 deg^2/Hz) in 4-8 Hz and 8-15 Hz (10 Hz = re-plan sawtooth)."""
    from scipy.signal import welch
    a, b = [], []
    for s in segs:
        p = np.nan_to_num(s["M"][:, 1].astype(np.float64))
        f, Pw = welch(p - p.mean(), fs=C.FPS, nperseg=120)
        a.append(Pw[(f >= 4) & (f < 8)].mean()); b.append(Pw[(f >= 8) & (f <= 15)].mean())
    return [float(np.log10(np.median(a) + 1e-12)), float(np.log10(np.median(b) + 1e-12))] if a else [float("nan")] * 2


def run_metrics(run, ids, man, ref_stats, with_ablate=True, onscreen=False):
    segs = HL.segs_for(run, man, ids, "auto")
    have = [c for c in ids if c in segs]
    S = [segs[c] for c in have]
    st = mstats.person_stats(S)
    err, inside, _ = M.compare_ref(st, ref_stats)
    g = lambda state, k: float((st.get(state) or {}).get(k, float("nan")))  # noqa: E731
    out = {"run": run, "n_clips": len(have), "clips": have,
           "abs_log_err": err, "inside_iqr": inside,
           "nods_per_min": [g("listening", "nod_per_min"), g("speaking", "nod_per_min")],
           "blinks_per_min": [g("listening", "blink_per_min"), g("speaking", "blink_per_min")],
           "smiles_per_min": [g("listening", "smile_per_min"), g("speaking", "smile_per_min")],
           "gaze_away": [g("listening", "avert_frac"), g("speaking", "avert_frac")],
           "head_speed": [HL.head_speed(S, 2), HL.head_speed(S, 1)],
           "stillness": [g("listening", "stillness_frac"), g("speaking", "stillness_frac")],
           "head_pitch_psd_4_8": g("all", "head_pitch_psd_4.0_8.0Hz"),
           "head_pitch_hf": hf_power(S),
           "head_jerk_med": HL.head_jerk(S),
           "minutes": {k: (st.get(k) or {}).get("minutes") for k in M.STATE_NAMES}}
    react = HL.reactions(run, man, have, segs, "auto")
    out["react"] = {k: v for k, v in react.items() if k != "_per_clip"}
    out["_per_clip_react"] = react["_per_clip"]
    out["_windows"] = {c: seg_windows(segs[c]) for c in have}
    out["_avert"] = {c: HL._avert_counts(segs[c]) for c in have}
    if with_ablate and run != "human":
        out["respond"] = responsiveness(run, have)
    if run in ("human", "copy-delay") or run.startswith("unified"):
        out["canon"] = canon_block(run, have, man)
        if onscreen:      # what the renderer receives (canon -> RETARGET rig -> human units), the retarget check
            rs = HL.segs_for(run, man, have, "rig")
            st2 = mstats.person_stats(list(rs.values()))
            g2 = lambda state, k: float((st2.get(state) or {}).get(k, float("nan")))  # noqa: E731
            out["onscreen_rig"] = {"nods_per_min_listen": g2("listening", "nod_per_min"),
                                   "blinks_per_min": g2("all", "blink_per_min"), "smiles_per_min": g2("all", "smile_per_min"),
                                   "gaze_away_listen": g2("listening", "avert_frac"),
                                   "head_speed_listen": HL.head_speed(list(rs.values()), 2)}
    return out


def cached_metrics(run, ids, man, ref_stats, out):
    """baseline metrics are cached by trace fingerprint (rules/human/gru-prior/copy-delay never change)."""
    runs = [run] + ([] if run == "human" else [run + "~zero", run + "~shuffle"])
    key = fingerprint(runs, ids)
    p = CACHE / f"metrics_{run}_{key}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    t0 = time.time()
    m = run_metrics(run, ids, man, ref_stats, onscreen=run != "rules")
    _atomic_pickle(p, m)
    log(f"metrics {run}: {time.time() - t0:.0f}s (cached -> {p.name})", out)
    return m


# ============================================================================= comparisons
def paired_vs_rules(sysm, rules, human, realism, nboot=300, seed=3):
    """paired clip bootstrap (same resamples): reaction excess deltas, gaze-away delta, realism (system, rules, delta)."""
    have = [c for c in sysm["clips"] if c in rules["clips"] and c in human["clips"]]
    rng = np.random.default_rng(seed)
    PAIRS = HL.PAIRS

    def excess(m, pick):
        return {k: HL._excess([m["_per_clip_react"][k][c] for c in pick if c in m["_per_clip_react"][k]]) for k in PAIRS}

    def away(m, pick):
        a = np.array([m["_avert"][c] for c in pick])
        return a[:, 0].sum() / max(1, a[:, 1].sum())

    def real(m, pick):
        W, K = stack_windows(m["_windows"], pick)
        return realism.fd(W, K)

    def one(pick):
        fh = real(human, pick)
        fs, fr = real(sysm, pick), real(rules, pick)
        rs = np.nanmean([fs[c] - fh[c] for c in (0, 1)]); rr = np.nanmean([fr[c] - fh[c] for c in (0, 1)])
        es, er = excess(sysm, pick), excess(rules, pick)
        d = {"realism_sys": rs, "realism_rules": rr, "realism_delta": rs - rr,
             "gaze_away_listen": away(sysm, pick) - away(rules, pick)}
        for k in PAIRS:
            d["excess " + k] = es[k] - er[k]
        return d
    point = one(have)
    boots = [one([have[i] for i in rng.integers(0, len(have), len(have))]) for _ in range(nboot)]
    res = {"n_clips": len(have), "nboot": nboot}
    for k, v in point.items():
        b = np.array([x[k] for x in boots], float)
        b = b[np.isfinite(b)]
        res[k] = {"delta": float(v), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 10
                  else [float("nan")] * 2}
    return res


def realism_block(m, human, realism, clips):
    W, K = stack_windows(m["_windows"], clips)
    Wh, Kh = stack_windows(human["_windows"], clips)
    f, fh = realism.fd(W, K), realism.fd(Wh, Kh)
    per = {CLASSES[c]: f[c] - fh[c] for c in (0, 1)}
    return {"realism": float(np.nanmean(list(per.values()))), "per_class": per,
            "fd_raw": {CLASSES[c]: f[c] for c in (0, 1)}, "fd_human_raw": {CLASSES[c]: fh[c] for c in (0, 1)},
            "n_windows": {CLASSES[c]: int((K == c).sum()) for c in (0, 1)},
            "top_gaps": realism.contrib(W, K)}


def heldout_check(realism, human, clips, draws=20, seed=11):
    """random held-out human chunks (as many as clips, distinct people) scored like a system, each against the reference
    without those people: mean ± sd should be ~0 on the realism scale (the normalisation check)."""
    ref = realism.ref
    rng = np.random.default_rng(seed)
    Wh, Kh = stack_windows(human["_windows"], clips)
    chunks_by_person = {}
    for ch, pe in zip(ref["chunk"], ref["person"]):
        chunks_by_person.setdefault(int(pe), set()).add(int(ch))
    people = sorted(chunks_by_person)
    vals = []
    for _ in range(draws):
        pick_p = rng.choice(people, size=min(len(clips), len(people) // 2), replace=False)
        chs = [int(rng.choice(sorted(chunks_by_person[p]))) for p in pick_p]
        sel = np.isin(ref["chunk"], chs)
        keep = ~np.isin(ref["person"], pick_p)
        Zk = realism.Z[keep]; ck = ref["cls"][keep]
        rm = {c: _moments(Zk[ck == c]) for c in (0, 1)}
        f = realism.fd(ref["X"][sel], ref["cls"][sel], rm)
        fh = realism.fd(Wh, Kh, rm)
        vals.append(float(np.nanmean([f[c] - fh[c] for c in (0, 1)])))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "draws": draws, "n_chunks": len(clips)}


def reactions_lost(pv, rules, sysm, human):
    """-> (lost, below_rules). below_rules (strict, info) = a reaction rules HAS (excess > 0) where the system's paired
    delta CI is entirely below 0. LOST (the verdict) = below_rules AND the system keeps less than half of the real
    partner's reaction strength (excess < max(0.02, 0.5 x human's)): rules over-mirrors smiles (+0.20 vs the real
    partner's +0.07 on these clips), so a human-level reaction must not count as lost (the real partner itself would
    'lose' smile->smile under the strict rule). Plus: user-shuffle sensitivity < half of rules' (ignores the user)."""
    lost, below = [], []
    for k in HL.PAIRS:
        d = pv.get("excess " + k)
        if d and rules["react"][k]["excess"] > 0 and d["ci95"][1] < 0:
            ex, hx = sysm["react"][k]["excess"], human["react"][k]["excess"]
            txt = (f"{k} (excess {ex:+.2f} vs rules {rules['react'][k]['excess']:+.2f}, real partner {hx:+.2f}; "
                   f"Δ vs rules {d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}])")
            below.append(txt)
            if ex < max(0.02, 0.5 * hx):
                lost.append(txt)
    rs, rr = (sysm.get("respond") or {}).get("shuffle"), (rules.get("respond") or {}).get("shuffle")
    if rs and rr and rs["overall"] < 0.5 * rr["overall"]:
        lost.append(f"user-shuffle sensitivity {rs['overall']:.2f} vs rules {rr['overall']:.2f}")
    return lost, below


# ============================================================================= judge
_SPEND_LOCK = threading.Lock()


def who_spend():
    tot = 0.0
    if C.SPEND.exists():
        for line in C.SPEND.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("who") == WHO:
                tot += float(r.get("usd") or 0)
    return tot


def judge(run, ids, out, render_jobs=None):
    # renders: jobs x threads (default 4 x 8; SCORECARD_RENDER_JOBS / _THREADS cap it on a shared box)
    render_jobs = int(render_jobs or os.environ.get("SCORECARD_RENDER_JOBS", "4"))
    render_threads = os.environ.get("SCORECARD_RENDER_THREADS", "8")
    import pairjudge as PJ
    PJ.J.SPEND_AGENT = WHO                       # cost lines -> $E/spend.jsonl as who=p2-score
    os.environ["JUDGE_ALLOW_PAID"] = "1"
    t0 = time.time()
    for r in ("rules", run):
        need = [c for c in ids if not (C.E / "runs" / r / c / "avatar.mp4").exists()]
        if need:
            sh([C.VENV_PY, "-I", C.KIT / "render.py", "--runs", r, "--ids", ",".join(need), "--jobs", str(render_jobs),
                "--threads", render_threads], out)
    t_render = time.time() - t0
    w0 = who_spend()
    refused = []

    def go(cid):
        with _SPEND_LOCK:      # spend lines land as calls finish: <= 4 calls in flight past a cap check
            now = who_spend()
            if now - w0 + EST_CALL > JUDGE_CAP_EACH or now + EST_CALL > JUDGE_CAP_TOTAL:
                refused.append(cid)
                return None
        return PJ.judge_clip(cid, "rules", run)
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(4) as pool:
        rows = [r for r in pool.map(go, ids) if r]
    main = [r for r in rows if not r.get("error") and r.get("naturalness")]
    spent = {"usd": who_spend() - w0}      # new paid calls only (cached judgements cost nothing)
    res = {"a": "rules", "b": run, "ids": ids, "n": len(main), "usd": round(spent["usd"], 4), "render_s": round(t_render),
           "refused_cap": refused, "errors": [r.get("error") for r in rows if r.get("error")]}
    for ax in ("naturalness", "reacts_to_user"):
        v = [PJ.score_of(r, ax, run) for r in main]
        w = sum(1 for r in main if r[ax] == run); lo = sum(1 for r in main if r[ax] == "rules")
        res[ax] = {"pref": float(np.mean(v)) if v else float("nan"), "ci95": PJ.boot(v) if len(v) > 1 else [float("nan")] * 2,
                   "wins": w, "losses": lo, "ties": len(main) - w - lo, "sign_p": PJ.sign_p(w, lo)}
    res["per_clip"] = {r["clip"]: {"nat": r["naturalness"], "react": r["reacts_to_user"]} for r in main}
    # calibration (cached judgements only, no call): the real partner, retargeted the same way, vs rules on these clips
    hj = [json.loads(p.read_text()) for p in (C.E / "judge/pair/rules__human" / f"{c}.json" for c in ids) if p.exists()]
    hj = [r for r in hj if r.get("naturalness") and not r.get("error")]
    if hj and run != "human":
        v = [PJ.score_of(r, "naturalness", "human") for r in hj]
        res["calibration_human"] = {"n": len(hj), "naturalness_pref": float(np.mean(v)),
                                    "wins": sum(1 for r in hj if r["naturalness"] == "human"),
                                    "losses": sum(1 for r in hj if r["naturalness"] == "rules")}
    log(f"judge {run}: nat {res['naturalness']['pref']:.2f} react {res['reacts_to_user']['pref']:.2f} n={len(main)} "
        f"${spent['usd']:.4f} (render {t_render:.0f}s)", out)
    return res


# ============================================================================= report
def f_(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def pair(v, nd=1, pct=False):
    return f"{f_(v[0], nd, pct)} / {f_(v[1], nd, pct)}" if v else "–"


def ref_pair(ref, key, scale=1.0):
    a, b = (ref.get("listening") or {}).get(key), (ref.get("speaking") or {}).get(key)
    if not a or not b:
        return ""
    fm = (lambda r: f"{r['med'] * scale:.0f}") if scale > 1 else (lambda r: f"{r['med']:.1f}")
    return f"{fm(a)} / {fm(b)}"


def build_md(res):
    tag, S, R, H = res["tag"], res["system"], res["rules"], res["human"]
    ref = res["ref_people"]
    name = S["run"]
    is_base = res["kind"] == "baseline"
    pv = res.get("vs_rules") or {}
    L = [f"### scorecard `{tag}` — {res['t']}  ({res['n_clips']} Seamless TEST clips; {res['space_note']})", ""]
    cols = [f"{tag}" if not is_base else name] + ([] if name == "rules" else ["rules (today)"]) + \
        ([] if name == "human" else ["human (real partner)"]) + ["held-out people med (L / S)"]
    L.append("| measure | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    systems = [S] + ([] if name == "rules" else [R]) + ([] if name == "human" else [H])

    def row(label, fn, refcell=""):
        L.append(f"| {label} | " + " | ".join(fn(m) for m in systems) + f" | {refcell} |")

    def rl(m):
        b = res["realism"].get(m["run"])
        if not b:
            return "–"
        s = f"**{b['realism']:.2f}**"
        k = "realism_sys" if m["run"] == name else ("realism_rules" if m["run"] == "rules" else None)
        if k and pv.get(k):
            s += f" [{pv[k]['ci95'][0]:.2f}, {pv[k]['ci95'][1]:.2f}]"
        return s
    hc = res.get("heldout_check") or {}
    row("REALISM ↓ (Fréchet gap to held-out people; real partner = 0)", rl,
        f"random held-out chunks: {hc.get('mean', float('nan')):.2f} ± {hc.get('sd', float('nan')):.2f}")
    row("realism without the jitter feature (behaviour only)",
        lambda m: f_(res["realism"][m["run"]].get("realism_no_jitter"), 2) if m["run"] in res["realism"] else "–")
    row("realism not-speaking / speaking", lambda m: pair([res["realism"][m["run"]]["per_class"][c] for c in CLASSES], 2)
        if m["run"] in res["realism"] else "–")
    row("human-range \\|log err\\| ↓ / inside IQR %", lambda m: f"{f_(m['abs_log_err'], 2)} / {f_(m['inside_iqr'], 0, True)}")
    row("nods/min (L / S)", lambda m: pair(m["nods_per_min"]), ref_pair(ref, "nod_per_min"))
    row("blinks/min (L / S)", lambda m: pair(m["blinks_per_min"]), ref_pair(ref, "blink_per_min"))
    row("smiles/min (L / S)", lambda m: pair(m["smiles_per_min"]), ref_pair(ref, "smile_per_min"))
    row("gaze-away % (L / S)", lambda m: pair(m["gaze_away"], 0, True), ref_pair(ref, "avert_frac", 100))
    row("head speed deg/s (L / S)", lambda m: pair(m["head_speed"]))
    row("stillness % (L / S)", lambda m: pair(m["stillness"], 0, True), ref_pair(ref, "stillness_frac", 100))
    a48 = (ref.get("all") or {}).get("head_pitch_psd_4.0_8.0Hz")
    row("head-pitch power log10 4–8 Hz / 8–15 Hz (jitter; 10 Hz = re-plan sawtooth)", lambda m: pair(m["head_pitch_hf"], 2),
        f"{a48['med']:.2f} / –" if a48 else "")
    row("head jerk median deg/s³", lambda m: f_(m["head_jerk_med"], 0))
    for k in HL.PAIRS:
        def rc(m, k=k):
            r = m["react"][k]
            s = f"{r['excess']:+.2f}"
            if m["run"] == name and name != "rules" and pv.get("excess " + k):
                d = pv["excess " + k]
                s += f" (Δ {d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}])"
            return s
        row(f"react {k} excess", rc)
    row("smile x-corr gain (real − shuffled user)", lambda m: f_(m["react"].get("smile_xcorr_gain"), 2))
    row("responds: user zeroed / shuffled", lambda m: pair([((m.get("respond") or {}).get(k) or {}).get("overall")
                                                             for k in ("zero", "shuffle")], 2) if m.get("respond") else "–")
    J = res.get("judge")
    if J and J.get("n"):
        L.append(f"| blind judge vs rules, n={J['n']} (pref ↑, 0.5 = rules; W/L/T): naturalness / reacts | "
                 f"{J['naturalness']['pref']:.2f} ({J['naturalness']['wins']}/{J['naturalness']['losses']}/"
                 f"{J['naturalness']['ties']}) / {J['reacts_to_user']['pref']:.2f} |" + " – |" * (len(cols) - 2)
                 + (f" real partner vs rules, same clips: {J['calibration_human']['naturalness_pref']:.2f} "
                    f"({J['calibration_human']['wins']}/{J['calibration_human']['losses']}) |" if J.get("calibration_human") else " |"))
    elif J is not None:
        L.append(f"| blind judge vs rules | {J.get('note', 'not run')} |" + " – |" * (len(cols) - 2) + " |")
    L.append("")
    cb = [(m["run"], m["canon"]) for m in systems if m.get("canon") and "error" not in m["canon"]]
    if cb:
        L.append("canon space (tools/train/evaluate.py stats on the clips' canon): " + "; ".join(
            f"{('**' + r + '**') if r == name else r}: nods {c['nods_per_min']:.1f}/min, blinks {c['blinks_per_min']:.1f}/min, "
            f"still {c['still_pct']:.0f}%, look-away L {c.get('look_away_listen_pct', float('nan')):.0f}%, "
            f"smile {c['smile_mean']:.2f}, hp48 {c['hp48_log10']:.2f}, smile coupling {c.get('smile_coupling', float('nan')):.2f}"
            for r, c in cb))
    if S.get("onscreen_rig"):
        o = S["onscreen_rig"]
        L.append(f"on screen (canon → retarget rig): nods L {o['nods_per_min_listen']:.1f}/min, blinks {o['blinks_per_min']:.1f}/min, "
                 f"smiles {o['smiles_per_min']:.1f}/min, gaze-away L {100 * o['gaze_away_listen']:.0f}%")
    rb = res["realism"].get(name)
    if rb and name != "human":
        L.append("biggest realism gaps (Δmean²+ΔSD² in reference SDs): " + ", ".join(f"{k} {v}" for k, v in rb["top_gaps"]))
    L.append("")
    L.append(f"**{res['verdict']['line']}**")
    L.append("")
    sf = res.get("speaking_face")
    if sf and sf.get("systems"):
        import speakface as SF
        L.append(SF.md(sf, [k for k in sf["runs"]] + ["real"]))
    elif sf and sf.get("error"):
        L.append(f"speaking face: {sf['error']}")
        L.append("")
    return "\n".join(L)


def verdict(res):
    name = res["system"]["run"]
    if name == "rules":
        return {"line": "rules = the baseline (no verdict)", "pass": None}
    real_s = res["realism"][name]["realism"]; real_r = res["realism"]["rules"]["realism"]
    better_real = real_s < real_r
    lost = res.get("reactions_lost") or []
    J = res.get("judge") or {}
    jp = (J.get("naturalness") or {}).get("pref") if J.get("n") else None
    d = (res.get("vs_rules") or {}).get("realism_delta")
    dci = f", Δ {d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d else ""
    parts = [f"realism {real_s:.2f} vs rules {real_r:.2f} ({'better' if better_real else 'NOT better'}{dci})"]
    parts.append(f"judge naturalness pref {jp:.2f} (n={J['n']})" if jp is not None else "judge not run")
    parts.append("no reaction lost" if not lost else "LOST: " + "; ".join(lost))
    extra = [b for b in (res.get("reactions_below_rules") or []) if b not in lost]
    if extra:
        parts.append("below rules but at the real partner's level (not lost): " + "; ".join(x.split(" (")[0] for x in extra))
    if not better_real or lost or (jp is not None and jp <= 0.5):
        word, ok = "FAIL", False
    elif jp is None:
        word, ok = "PASS-PENDING-JUDGE", None
    else:
        word, ok = "PASS", True
    return {"line": f"beats rules: {word} — " + "; ".join(parts), "pass": ok, "word": word,
            "realism_better": better_real, "judge_pref": jp, "reactions_lost": lost}


def append_index(md, res):
    import fcntl
    p = SC_ROOT / "INDEX.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        if not f.read(1):
            f.write("# phase 2 scorecards (evalkit/scorecard.py; newest last)\n\n")
        f.seek(0, 2)
        f.write(md + f"\n(json: {res['out']}/scorecard.json)\n\n---\n\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def strip_private(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle", help="bundle dir (config.json + weights), or a run dir holding bundle/")
    g.add_argument("--system", help="baseline: rules | gru-prior | copy-delay | human")
    ap.add_argument("--tag", help="run name (unique per bundle+runtime knobs); default = the bundle's run / the system")
    ap.add_argument("--out", help="default $P/scorecards/<tag>")
    ap.add_argument("--code", help="the unified/ runtime code for the bundle (default: $EVAL_MODEL_CODE, else this checkout); "
                                   "snapshotted once per tag into $E/code/<tag>")
    ap.add_argument("--kw", default="{}", help="extra UnifiedPlanner kwargs (json); use a distinct --tag")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-ids", default=",".join(JUDGE_IDS))
    ap.add_argument("--index", action="store_true", help="append the table to $P/scorecards/INDEX.md")
    ap.add_argument("--force", action="store_true", help="re-step a tag whose cached traces came from another bundle")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args()
    t_start = time.time()
    if a.bundle:
        code = a.code or os.environ.get("EVAL_MODEL_CODE")
        if not code:
            code = str(Path(__file__).resolve().parents[1])
        os.environ["EVAL_MODEL_CODE"] = str(Path(code).resolve())
    _kit()
    man = AN.manifest()
    ids = seamless_ids(man)
    meta = None
    if a.bundle:
        bd = bundles.resolve_bundle(a.bundle)
        tag = a.tag or bundles.bundle_tag(bd)
        meta = bundles.resolve(f"unified:{bd}#{tag}")
        run = meta["run"]
    else:
        run = ALIAS.get(a.system, a.system)
        if run not in BASELINES:
            raise SystemExit(f"--system must be one of {BASELINES}")
        tag = a.tag or run
    out = Path(a.out or SC_ROOT / tag).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.log").write_text("")
    log(f"scorecard {tag}: run={run} bundle={meta and meta['bundle']} code={meta and meta['code']} clips={len(ids)}", out)
    t_step = 0.0
    if meta is not None:
        guard_bundle_cache(meta, a.force, out)
        t_step = step_system(run, meta, ids, a.jobs, json.loads(a.kw), out)
    elif run != "rules":
        t_step = step_system(run, None, ids, a.jobs, {}, out)
    step_system("rules", None, ids, a.jobs, {}, out) if missing_traces("rules", ids) else None
    step_system("human", None, ids, a.jobs, {}, out) if missing_traces("human", ids) else None
    log(f"stepping {t_step:.0f}s", out)
    t0 = time.time()
    ref_stats = AN.get_ref()["ref"]
    ref = reference_windows(man)
    realism = Realism(ref)
    rules = cached_metrics("rules", ids, man, ref_stats, out)
    human = cached_metrics("human", ids, man, ref_stats, out)
    if run in BASELINES:
        sysm = cached_metrics(run, ids, man, ref_stats, out)
    else:
        sysm = run_metrics(run, ids, man, ref_stats, onscreen=True)
    clips = [c for c in ids if c in sysm["clips"] and c in rules["clips"] and c in human["clips"]]
    uniq = {m["run"]: m for m in (sysm, rules, human)}
    rb = {r: realism_block(m, human, realism, clips) for r, m in uniq.items()}
    realism_nj = Realism(ref, drop=("head_jitter",))      # diagnostic: behaviour only (runtime jitter excluded)
    for r, m in uniq.items():
        rb[r]["realism_no_jitter"] = realism_block(m, human, realism_nj, clips)["realism"]
    hp = CACHE / f"heldout_check_{VERSION}_{fingerprint(['human'], clips)}.json"
    if hp.exists():
        hc = json.loads(hp.read_text())
    else:
        hc = heldout_check(realism, human, clips)
        C.atomic_json(hp, hc)
    pv = paired_vs_rules(sysm, rules, human, realism, a.nboot) if run != "rules" else {}
    lost, below = reactions_lost(pv, rules, sysm, human) if run != "rules" else ([], [])
    t_meas = time.time() - t0
    log(f"measures {t_meas:.0f}s", out)
    res = {"version": VERSION, "t": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "tag": tag,
           "kind": "bundle" if meta else "baseline", "meta": meta, "out": str(out), "n_clips": len(clips), "clips": clips,
           "space_note": "rules rig→human units, canon systems by FK (= headline_rig.json's auto)",
           "system": strip_private(sysm), "rules": strip_private(rules), "human": strip_private(human),
           "ref_people": {s: {k: v for k, v in (ref_stats.get(s) or {}).items()
                              if k in ("nod_per_min", "blink_per_min", "smile_per_min", "avert_frac", "stillness_frac",
                                       "head_pitch_psd_4.0_8.0Hz")} for s in ("listening", "speaking", "all")},
           "realism": rb, "realism_def": {"feats": FEATS, "win_s": WIN / 30, "hop_s": HOP / 30, "classes": CLASSES,
                                          "ref_windows": int(len(ref["X"])), "ref_people": len(ref["persons"]),
                                          "ref_sessions": len(ref["sessions"])},
           "heldout_check": hc, "vs_rules": pv, "reactions_lost": lost, "reactions_below_rules": below}
    if a.no_judge:
        res["judge"] = {"note": "skipped (--no-judge)"}
    elif run == "rules":
        res["judge"] = None
    else:
        try:
            res["judge"] = judge(run, [c for c in a.judge_ids.split(",") if c], out)
        except Exception as e:  # noqa: BLE001
            log(f"judge failed: {e!r}"[:600], out)
            res["judge"] = {"note": f"failed: {str(e)[:120]}"}
    # the speaking face ON SCREEN (lipstream + page rig, renderer overrides) vs the real face, this system + baselines
    try:
        import speakface as SF
        t_sf = time.time()
        sfr = {} if run == "human" else {tag: run}
        for lab, r in (("rules", "rules"), ("abl_t0", "unified-abl_t0")):
            if r != run and not missing_traces(r, clips):
                sfr[lab] = r
        res["speaking_face"] = SF.section(sfr, clips, man)
        log(f"speaking face {time.time() - t_sf:.0f}s", out)
    except Exception as e:  # noqa: BLE001
        log(f"speaking face failed: {e!r}"[:400], out)
        res["speaking_face"] = {"error": f"failed: {str(e)[:160]}"}
    res["verdict"] = verdict(res)
    res["timing_s"] = {"step": round(t_step), "measures": round(t_meas), "total": round(time.time() - t_start)}
    res["spend_who_total_usd"] = round(who_spend(), 4)
    C.atomic_json(out / "scorecard.json", res)
    md = build_md(res)
    (out / "scorecard.md").write_text(md + f"\ntiming: step {t_step:.0f}s, measures {t_meas:.0f}s, total "
                                           f"{time.time() - t_start:.0f}s\n")
    if a.index:
        append_index(md, res)
    print(md)
    log(f"done in {time.time() - t_start:.0f}s -> {out}/scorecard.md", out)


if __name__ == "__main__":
    main()
