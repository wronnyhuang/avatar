#!/usr/bin/env python3
"""Paired comparison of two judged runs (A = baseline, B = candidate) on the held-out clips + side-by-side videos.

  python eval/compare.py <runA> <runB> [--split holdout] [--judge <id>] [--out cmp.json]
  python eval/compare.py <runA> <runB> --sbs <dir> [--sbs-clips id,id|all] [--reel reel.mp4]

<run> = a run name under <S>/runs/ or a run dir; each needs scores.json from eval/judge.py.
Per axis: mean delta (B − A) over paired clips, bootstrap 95 % CI (clips resampled, 5000×), win rate (B > A; ties ½).
Verdict on `weighted`: BETTER if the CI's lower bound > 0, WORSE if the upper bound < 0, else NO_DIFFERENCE —
"keep only held-out-better changes" = keep B only on BETTER. Repeats (<clip>__rN) are averaged per clip first.
Side-by-side: user camera | A | B (vertical stack for a phone, labelled), user voice centre, A's voice left, B's right,
H.264 ≤ 1 Mbps.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

import os

_EVAL = Path(os.environ.get("EVAL_OUT", str(Path(os.environ.get("EVAL_T", os.environ.get(
    "MM_WORK_DIR", str(Path(__file__).resolve().parents[2] / "work")))) / "eval")))
S = _EVAL      # runs/ live here
T = _EVAL      # clips/manifest.json
AXES = ["weighted", "naturalness", "reaction_timing", "emotional_fit", "gaze", "motion_variety", "commands",
        "talk_amount", "lip_sync"]


def run_dir(x: str) -> Path:
    p = Path(x)
    return p if p.is_dir() else S / "runs" / x


def manifest_split(run: Path) -> dict:
    rj = run / "run.json"
    cands = []
    if rj.exists():
        cands.append(Path(json.loads(rj.read_text()).get("manifest", "")))
    cands += [T / "clips" / "manifest.json"]
    for m in cands:
        if m and m.is_file():
            return {c["id"]: c for c in json.loads(m.read_text())["clips"]}
    return {}


def load_scores(run: Path, judge: str | None) -> tuple[dict, str]:
    sc = json.loads((run / "scores.json").read_text())
    j = judge or sc.get("primary") or next(iter(sc["judges"]))
    per = {}
    for cid, v in sc["judges"].get(j, {}).items():
        base = re.sub(r"__r\d+$", "", cid)
        row = {"weighted": v.get("weighted")}
        row.update(v.get("axes") or {})
        per.setdefault(base, []).append(row)
    out = {}
    for cid, rows in per.items():
        out[cid] = {ax: (float(np.mean([r[ax] for r in rows if isinstance(r.get(ax), (int, float))]))
                         if any(isinstance(r.get(ax), (int, float)) for r in rows) else None) for ax in AXES}
    return out, j


def boot_ci(d: np.ndarray, n: int = 5000, seed: int = 7) -> tuple[float, float]:
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(a: Path, b: Path, split: str, judge: str | None, margin: float = 0.0) -> dict:
    A, ja = load_scores(a, judge)
    B, jb = load_scores(b, judge)
    man = manifest_split(b) or manifest_split(a)
    common = sorted(set(A) & set(B))
    if split != "all" and man:
        common = [c for c in common if man.get(c, {}).get("split") == split]
    shas = []
    for r in (a, b):
        rj = r / "run.json"
        shas.append(json.loads(rj.read_text()).get("manifest_sha") if rj.exists() else None)
    if shas[0] != shas[1]:
        print(f"WARNING: the runs used different clip manifests ({shas[0]} vs {shas[1]}): pairs may not be like for like")
    res = {"A": a.name, "B": b.name, "judge": [ja, jb], "manifest_sha": shas, "split": split, "n_clips": len(common), "clips": common, "axes": {}}
    for ax in AXES:
        pairs = [(A[c][ax], B[c][ax]) for c in common if A[c].get(ax) is not None and B[c].get(ax) is not None]
        if not pairs:
            res["axes"][ax] = None
            continue
        pa, pb = np.array(pairs).T
        d = pb - pa
        lo, hi = boot_ci(d)
        res["axes"][ax] = {"n": len(d), "A": round(float(pa.mean()), 3), "B": round(float(pb.mean()), 3),
                           "delta": round(float(d.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)],
                           "win_rate_B": round(float(((d > 0) + 0.5 * (d == 0)).mean()), 3)}
    w = res["axes"].get("weighted")
    if w:
        lo, hi = w["ci95"]
        # margin = the A/A noise floor for this judge (same config run twice, scaled by 1/sqrt(n)): Astra on 12 clips
        # showed a spurious −0.62 [−1.24, −0.13] (gflash: −0.08 [−0.69, +0.53]). BETTER/WORSE need the CI to exclude 0
        # AND |Δ| ≥ margin.
        if margin < 0:                                   # auto: per-judge floor at this n
            margin = (0.65 if str(res["judge"][1]).startswith("astra") else 0.15) * (12 / max(1, w["n"])) ** 0.5
        res["margin"] = round(margin, 3)
        d = w["delta"]
        res["verdict"] = ("BETTER" if (lo > 0 and d >= margin) else
                          "WORSE" if (hi < 0 and d <= -margin) else "NO_DIFFERENCE")
    res["per_clip"] = {c: {"A": A[c]["weighted"], "B": B[c]["weighted"]} for c in common}
    # run-level human-likeness of the rig motion (eval/motion_compare.py vs the prior's MOTION_STATS.json; lower err better)
    mr = {}
    for k, r in (("A", a), ("B", b)):
        mp = r / "motion_stats.json"
        if mp.exists():
            mj = json.loads(mp.read_text())
            mr[k] = {"mean_abs_log_err": mj.get("mean_abs_log_err"), "inside_real_iqr": mj.get("frac_inside_real_iqr"),
                     "seconds": mj.get("seconds")}
    if mr:
        res["motion_realism"] = mr
    if man:
        cats = {}
        for c in common:
            if A[c]["weighted"] is not None and B[c]["weighted"] is not None:
                cats.setdefault(man.get(c, {}).get("category", "?"), []).append(B[c]["weighted"] - A[c]["weighted"])
        res["by_category"] = {k: {"n": len(v), "delta": round(float(np.mean(v)), 3)} for k, v in sorted(cats.items())}
    return res


def table(res: dict) -> str:
    lines = [f"A = {res['A']}  B = {res['B']}  split = {res['split']}  n = {res['n_clips']}  verdict: {res.get('verdict')}"
             + (f" (margin ±{res['margin']})" if res.get("margin") else ""),
             f"{'axis':16s} {'n':>3s} {'A':>6s} {'B':>6s} {'Δ':>7s}  {'95% CI':>17s} {'win B':>6s}"]
    for ax in AXES:
        r = res["axes"].get(ax)
        if r:
            lines.append(f"{ax:16s} {r['n']:3d} {r['A']:6.2f} {r['B']:6.2f} {r['delta']:+7.2f}  "
                         f"[{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}] {r['win_rate_B']:6.2f}")
    mr = res.get("motion_realism") or {}
    if mr:
        f = lambda d, k: "–" if not d or d.get(k) is None else f"{d[k]:.3f}"
        lines.append(f"motion realism (rig log vs real people, run level): mean|log err| A {f(mr.get('A'), 'mean_abs_log_err')} "
                     f"B {f(mr.get('B'), 'mean_abs_log_err')} (lower = more human); inside real IQR A "
                     f"{f(mr.get('A'), 'inside_real_iqr')} B {f(mr.get('B'), 'inside_real_iqr')}")
    return "\n".join(lines)


def sbs(a: Path, b: Path, cid: str, out: Path, man: dict, la: str, lb: str) -> Path | None:
    va, vb = a / cid / "avatar.mp4", b / cid / "avatar.mp4"
    clip = man.get(cid)
    if not (va.exists() and vb.exists() and clip):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = 480, 270
    def lab(t):
        t = t.replace(":", r"\:").replace("'", "")
        return f"drawtext=text='{t}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55"
    fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=8,fps=30,{lab('you: ' + cid)}[u];"
          f"[1:v]scale={W}:{H},setsar=1,fps=30,{lab('A: ' + la)}[x];[2:v]scale={W}:{H},setsar=1,fps=30,{lab('B: ' + lb)}[y];"
          f"[u][x][y]vstack=3[v];"
          f"[0:a]apad,pan=stereo|c0=0.6*c0|c1=0.6*c0[ua];[1:a]pan=stereo|c0=c0|c1=0.15*c0[aa];"
          f"[2:a]pan=stereo|c0=0.15*c0|c1=c0[ba];[ua][aa][ba]amix=3:duration=shortest:normalize=0[a]")
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                         str(va)]).decode().strip())
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-threads", "4", "-i", clip["camera_mp4"], "-i", str(va), "-i", str(vb),
                    "-filter_complex", fc, "-filter_complex_threads", "4", "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
                    "-c:v", "libx264", "-threads", "6", "-preset", "veryfast", "-b:v", "750k", "-maxrate", "850k", "-bufsize", "1700k",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)], check=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--split", default="holdout", help="holdout | train | all")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--margin", type=float, default=None,
                    help="verdict margin on Δweighted; default auto = A/A noise floor × sqrt(12/n): astra 0.65, gflash 0.15")
    ap.add_argument("--out", help="write the comparison JSON here")
    ap.add_argument("--sbs", help="directory for side-by-side videos")
    ap.add_argument("--sbs-clips", default="all", help="clip ids (comma) or 'all' paired clips of the split")
    ap.add_argument("--reel", help="concatenate the side-by-sides into one phone video")
    ap.add_argument("--label-a"); ap.add_argument("--label-b")
    x = ap.parse_args()
    a, b = run_dir(x.a), run_dir(x.b)
    res = None
    if (a / "scores.json").exists() and (b / "scores.json").exists():
        margin = x.margin if x.margin is not None else -1.0
        res = compare(a, b, x.split, x.judge, margin)
        print(table(res))
        if res.get("by_category"):
            print("by category (Δ weighted):", ", ".join(f"{k} {v['delta']:+.2f} (n={v['n']})" for k, v in res["by_category"].items()))
        if x.out:
            Path(x.out).write_text(json.dumps(res, indent=1))
    else:
        print("no scores.json in one of the runs: side-by-side only")
    if x.sbs:
        man = manifest_split(b) or manifest_split(a)
        if x.sbs_clips == "all":
            ids = res["clips"] if res else sorted({p.name for p in a.iterdir() if (p / "avatar.mp4").exists()} &
                                                  {p.name for p in b.iterdir() if (p / "avatar.mp4").exists()})
        else:
            ids = x.sbs_clips.split(",")
        made = []
        for cid in ids:
            o = sbs(a, b, cid, Path(x.sbs) / f"{cid}.mp4", man, x.label_a or a.name, x.label_b or b.name)
            if o:
                made.append(o)
                print("sbs", o)
        if x.reel and made:
            lst = Path(x.sbs) / "_reel.txt"
            lst.write_text("".join(f"file '{p.resolve()}'\n" for p in made))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                            "-movflags", "+faststart", x.reel], check=True)
            print("reel", x.reel)


if __name__ == "__main__":
    main()
