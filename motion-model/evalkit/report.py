#!/usr/bin/env python3
"""EVAL: <out>/REPORT.md + <out>/index.html + <out>/results.json from run_all.py's stage outputs.

  python -I evalkit/report.py --out <dir>

Verdict (acceptance bar: more natural, nothing lost on reactions or speed):
  NATURAL  better  = judge naturalness preference for unified > 0.5 with the 95 % CI above 0.5, AND human-range
                     |log err| not worse than rules (paired CI upper bound <= +0.05)
           worse   = judge preference CI below 0.5, or |log err| paired CI lower bound > 0
           else    = not shown better
  REACTIONS lost   = any reaction pair where unified's excess-over-chance is below rules' with the paired CI upper
                     bound < 0, OR judge reacts-to-user preference CI below 0.5, OR shuffle-ablation sensitivity < half
                     of rules' (it ignores the user)
  SPEED lost       = e2e visible-change median slower than rules' by > 100 ms on a probe both systems react to, or
                     re-plan p99 > 33 ms (a tick) on the server
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, atomic_json, eval_spend  # noqa: E402


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:.{nd}f}"


def ci(c, nd=2, pct=False):
    if not c or any(v is None or not math.isfinite(v) for v in c):
        return ""
    return f" [{f(c[0], nd, pct)}, {f(c[1], nd, pct)}]"


NAMES = {"rules": "rules (today)", "gru-prior": "GRU prior", "copy-delay": "copy-delay", "human": "human (real partner)"}


def nm(run):
    if run.startswith("unified"):
        return run.replace("unified-", "unified ")
    return NAMES.get(run, run)


def build(out: Path, cmd: str = ""):
    cfg = J(out / "config.json") or {}
    prim = (cfg.get("unified") or [{}])[0]
    U = prim.get("run")
    H = J(out / "headline_rig.json")
    AB = J(out / "ablate.json") or {}
    DR = J(out / "drift.json")
    LA = J(out / "latency.json") or {}
    E2 = J(out / "e2e.json")
    JU = J(out / "judge_rules_vs_unified.json")
    JH = J(out / "judge_rules_vs_human.json")
    ST = J(out / "style.json")
    md = []
    res = {"config": cfg, "verdict": {}}
    runs = []
    if H:
        runs = [r for r in ["rules", "gru-prior", "copy-delay", U, "human"] if r in H["groups"].get("all", {}) or r in H["groups"].get("seamless", {})]
    # ------------------------------------------------------------------ verdict pieces
    nat, react_lost, speed_lost, why = None, None, None, []
    if JU and "all" in JU.get("groups", {}):
        g = JU["groups"]["all"]["naturalness"]
        lo, hi = g["ci95"]
        nat_j = "better" if lo > 0.5 else ("worse" if hi < 0.5 else "same")
        why.append(f"judge naturalness pref {f(g['pref_b'], 2)}{ci(g['ci95'])} (n={JU['groups']['all']['n']})")
    else:
        nat_j = None
    hl = None
    if H and U in H["groups"].get("all", {}):
        v = H["groups"]["all"][U].get("vs_rules") or {}
        hl = v.get("abs_log_err")
    if hl:
        why.append(f"human-range |log err| vs rules {hl['delta']:+.3f}{ci(hl['ci95'], 3)}")
    if nat_j == "better" and (hl is None or hl["ci95"][1] <= 0.05):
        nat = "better"
    elif nat_j == "worse" or (hl and hl["ci95"][0] > 0):
        nat = "worse"
    elif nat_j is None and hl is not None:
        nat = "better" if hl["ci95"][1] < 0 else ("worse" if hl["ci95"][0] > 0 else "same")
    else:
        nat = "same" if nat_j else None
    lost = []
    if H and U in H["groups"].get("all", {}):
        v = H["groups"]["all"][U].get("vs_rules") or {}
        for k in H["pairs"]:
            d = v.get("excess " + k)
            if d and d["ci95"][1] < 0:
                lost.append(f"{k} (excess {d['delta']:+.2f}{ci(d['ci95'])})")
    if JU and "all" in JU.get("groups", {}):
        g = JU["groups"]["all"]["reacts_to_user"]
        if g["ci95"][1] < 0.5:
            lost.append(f"judge reacts-to-user pref {f(g['pref_b'])}{ci(g['ci95'])}")
    if AB.get(U) and AB.get("rules") and "shuffle" in AB[U] and "shuffle" in AB["rules"]:
        if AB[U]["shuffle"]["overall"] < 0.5 * AB["rules"]["shuffle"]["overall"]:
            lost.append(f"user-shuffle sensitivity {f(AB[U]['shuffle']['overall'])} vs rules {f(AB['rules']['shuffle']['overall'])}")
    react_lost = bool(lost) if (H or JU) else None
    sl = []
    if E2 and E2.get("summary"):
        for probe in ("smile", "turn"):
            a, b = E2["summary"].get(f"unified/{probe}"), E2["summary"].get(f"rules/{probe}")
            if a and b and a.get("visible_med_ms") is not None and b.get("visible_med_ms") is not None:
                ua = a["visible_med_ms"] + (E2.get("worker_rtt_ms") or {}).get("p50", 0)
                if ua > b["visible_med_ms"] + 100:
                    sl.append(f"{probe}: unified {ua:.0f} ms vs rules {b['visible_med_ms']:.0f} ms")
            elif b and b.get("visible_n") and a and not a.get("visible_n"):
                sl.append(f"{probe}: rules reacts visibly, unified never did")
    lat = (LA.get(U) or {}).get("unified") or {}
    if lat.get("plan_ms_p99") and lat["plan_ms_p99"] > 33.3:
        sl.append(f"re-plan p99 {lat['plan_ms_p99']:.1f} ms > one 33 ms tick")
    speed_lost = bool(sl) if (E2 or lat) else None
    nat_s = {"better": "better", "worse": "WORSE", "same": "not better", None: "not measured"}[nat]
    r_s = {True: "LOSES reactions/speed", False: "does not lose any reaction or speed", None: "reactions not measured"}[
        bool(react_lost or speed_lost) if (react_lost is not None or speed_lost is not None) else None]
    verdict = f"unified ({prim.get('tag')}) is {nat_s} than rules on naturalness, and {r_s}."
    ship = nat == "better" and react_lost is False and speed_lost is False
    res["verdict"] = {"line": verdict, "naturalness": nat, "naturalness_why": why, "reactions_lost": lost,
                      "speed_lost": sl, "acceptance_bar_met": ship}
    md.append(f"# unified avatar motion — eval report ({cfg.get('t', '')})\n")
    md.append("Rerun on another checkpoint (from motion-model/; ~1.5 h on a busy CPU host; traces cached):\n")
    md.append("`nohup python -I "
              "evalkit/run_all.py --systems rules,prior,copy_delay,unified:<bundle dir | BEST> --out "
              "$MM_WORK_DIR/eval/<name> --style 'energy=0.1,expressiveness=0.1;energy=0.9,expressiveness=0.9' "
              "--publish > $MM_WORK_DIR/eval/<name>.log 2>&1 &`\n")
    if cmd:
        md.append(f"This report was made by: `{cmd}`\n")
    md.append(f"**VERDICT: {verdict}**  Acceptance bar (more natural, nothing lost on reactions or speed): "
              f"**{'MET' if ship else 'NOT MET'}**.\n")
    if why:
        md.append("- naturalness evidence: " + "; ".join(why))
    md.append("- reactions lost: " + ("; ".join(lost) if lost else "none detected"))
    md.append("- speed lost: " + ("; ".join(sl) if sl else "none detected"))
    md.append(f"\nCandidate: `{prim.get('run')}` — bundle `{prim.get('bundle')}` (sha {prim.get('bundle_sha')}, "
              f"kind {prim.get('bundle_kind')}), runtime code snapshot `{prim.get('code')}`. Baseline rules = node replay "
              "of the integrated build's controller (bit-identical to the live page on all 50 clips). "
              "Clip set: 29 Seamless TEST sessions (36 s, real partner = `human`) + 21 held-out suite clips (no commands).\n")
    # ------------------------------------------------------------------ main table
    if H:
        md.append("## Measures (Seamless clips where the real partner exists, n=29; rules in rig→human units, canon systems in the reference's FK space)\n")
        Sg = H["groups"].get("seamless", {}); Ag = H["groups"].get("all", {})
        ref = H.get("reference", {})
        cols = [r for r in runs if r in Sg]
        rows = [("human-range |log err| vs held-out people ↓", "abs_log_err", 3, False),
                ("cells inside the people's IQR ↑", "inside_iqr", 2, True),
                ("head speed listening (deg/s)", "head_speed_listen", 1, False),
                ("head speed speaking (deg/s)", "head_speed_speak", 1, False),
                ("gaze-away while listening", "gaze_away_listen", 2, True),
                ("nods/min listening", "nods_per_min_listen", 1, False),
                ("blinks/min", "blinks_per_min", 1, False),
                ("smiles/min", "smiles_per_min", 1, False),
                ("head jerk median (deg/s³)", "head_jerk_med", 0, False),
                ("4-8 Hz head pitch power (log10) ↓ = smoother", "head_pitch_psd_4_8", 2, False)]
        refcol = {"gaze_away_listen": (ref.get("listening") or {}).get("avert_frac"),
                  "nods_per_min_listen": (ref.get("listening") or {}).get("nod_per_min"),
                  "blinks_per_min": (ref.get("all") or {}).get("blink_per_min"),
                  "smiles_per_min": (ref.get("all") or {}).get("smile_per_min")}
        md.append("| measure | " + " | ".join(nm(c) for c in cols) + " | held-out people (median [IQR]) |")
        md.append("|---|" + "---|" * (len(cols) + 1))
        for lab, k, nd, pct in rows:
            rv = refcol.get(k)
            rtxt = f"{f(rv['med'], nd, pct)} [{f(rv['q25'], nd, pct)}–{f(rv['q75'], nd, pct)}]" if rv else ""
            md.append(f"| {lab} | " + " | ".join(f(Sg[c].get(k), nd, pct) for c in cols) + f" | {rtxt} |")
        md.append("\n## Reactions (all 50 clips; event-locked 0.2–2 s; hit rate vs chance, median lag)\n")
        colsA = [r for r in runs if r in Ag]
        md.append("| user event → avatar | " + " | ".join(nm(c) for c in colsA) + " |")
        md.append("|---|" + "---|" * len(colsA))
        for k in H["pairs"]:
            cells = []
            for c in colsA:
                r = Ag[c]["react"].get(k) or {}
                cells.append(f"{f(r.get('rate'), 2)}/{f(r.get('chance'), 2)} (+{f(r.get('excess'), 2)}) lag {f(r.get('lag_med_s'), 2)} s"
                             if r.get("n") else "–")
            md.append(f"| {k} (n={(Ag['rules']['react'].get(k) or {}).get('n', '?') if 'rules' in Ag else '?'}) | " + " | ".join(cells) + " |")
        md.append("| smile x-corr gain (real − shuffled user) | " + " | ".join(f(Ag[c]["react"].get("smile_xcorr_gain"), 3) for c in colsA) + " |")
        if U in Ag and Ag[U].get("vs_rules"):
            v = Ag[U]["vs_rules"]
            md.append(f"\nPaired vs rules (clip bootstrap, n={v['n_clips']}): " + "; ".join(
                f"{k} {v[k]['delta']:+.3f}{ci(v[k]['ci95'], 3)}" for k in v if k != "n_clips"))
    if AB:
        md.append("\n## Responsiveness: output change when the USER stream is removed (zero) or swapped (shuffle)\n")
        md.append("RMS change of head/gaze/brows/smile ÷ the output's own SD (0 = ignores the user).\n")
        md.append("| system | zero (no user) | shuffle (another user) |")
        md.append("|---|---|---|")
        for r in [x for x in ["rules", "gru-prior", "copy-delay", U] if x in AB]:
            md.append(f"| {nm(r)} | {f(AB[r].get('zero', {}).get('overall'))}{ci(AB[r].get('zero', {}).get('ci95'))} | "
                      f"{f(AB[r].get('shuffle', {}).get('overall'))}{ci(AB[r].get('shuffle', {}).get('ci95'))} |")
    if JU or JH:
        md.append("\n## Blind pairwise judge (gemini-3.7-flash via OpenRouter; A/B order a per-clip coin)\n")
        md.append("| comparison | clips | naturalness: pref for 2nd [95% CI] (wins/losses/ties, sign p) | reacts-to-user |")
        md.append("|---|---|---|---|")
        for JJ, lab in ((JU, f"rules vs {nm(U)}"), (JH, "rules vs human (judge sanity check)")):
            if not JJ:
                continue
            for grp in ("all", "seamless", "youtube"):
                g = JJ["groups"].get(grp)
                if not g:
                    continue
                cell = lambda a: (f"{f(g[a]['pref_b'])}{ci(g[a]['ci95'])} ({g[a]['wins_b']}/{g[a]['wins_a']}/{g[a]['ties']}, "  # noqa: E731
                                  f"p={g[a]['sign_p']:.3f})")
                md.append(f"| {lab} — {grp} | {g['n']} | {cell('naturalness')} | {cell('reacts_to_user')} |")
        if JU and JU.get("position_consistency"):
            pc = JU["position_consistency"]
            md.append("\nPosition check (same clip, A/B swapped): " + "; ".join(
                f"{ax} same winner {v['same_winner']}/{v['n']}" for ax, v in pc.items()))
    if E2 or LA:
        md.append("\n## Latency\n")
        if lat:
            md.append(f"- planner re-plan on the server (2 threads, busy shared host, load {', '.join((LA.get(U) or {}).get('host_load', []))}): "
                      f"p50 {f(lat.get('plan_ms_p50'), 1)} ms, p99 {f(lat.get('plan_ms_p99'), 1)} ms, max {f(lat.get('plan_ms_max'), 1)} ms "
                      f"({lat.get('n_plans')} plans); per frame amortised {f(lat.get('step_ms_mean'), 2)} ms")
            fz = (LA.get(U) or {}).get("fitter")
            if fz:
                md.append(f"- fitter (MediaPipe → canon) p50 {f(fz.get('ms_p50'), 2)} ms, p99 {f(fz.get('ms_p99'), 2)} ms per frame")
            rz = (LA.get(U) or {}).get("rules")
            if rz:
                md.append(f"- rules controller (node) step p50 {f(rz.get('step_ms_p50_med'), 2)} ms, p99 {f(rz.get('step_ms_p99_med'), 2)} ms")
        if E2:
            w = E2.get("worker_rtt_ms") or {}
            md.append(f"- unified worker round trip per tick (pack → pipe → fitter → planner → retarget → pipe), app/unified_worker.py "
                      f"of the integrated build: p50 {f(w.get('p50'), 1)} ms, p95 {f(w.get('p95'), 1)} ms, p99 {f(w.get('p99'), 1)} ms")
            md.append("- user move → visible avatar change (probe vs control run; ticks of 33 ms; rules' number skips the page's "
                      "perception smoothing, i.e. it is a lower bound; network + render not included):\n")
            md.append("| probe | unified: visible (median, max) | unified: any change | rules: visible (median, max) | trials hit (U / R) |")
            md.append("|---|---|---|---|---|")
            for probe in E2.get("probes", []):
                a = E2["summary"].get(f"unified/{probe}") or {}; b = E2["summary"].get(f"rules/{probe}") or {}
                md.append(f"| {probe} | {f(a.get('visible_med_ms'), 0)} / {f(a.get('visible_max_ms'), 0)} ms | {f(a.get('any_med_ms'), 0)} ms | "
                          f"{f(b.get('visible_med_ms'), 0)} / {f(b.get('visible_max_ms'), 0)} ms | {a.get('visible_n', 0)}/{a.get('n', 0)} · "
                          f"{b.get('visible_n', 0)}/{b.get('n', 0)} |")
    if DR:
        md.append("\n## Drift: 5-minute closed-loop rollout (one concatenated conversation)\n")
        md.append("| system | worst per-minute SD ratio vs minute 1 | mean-pose wander (deg) | longest freeze (s) | |head|>30° frames | drifts? |")
        md.append("|---|---|---|---|---|---|")
        for r, v in DR["systems"].items():
            md.append(f"| {nm(r)} | {f(v['worst_sd_ratio'])} | {f(max(v['mean_wander_deg'].values()), 1)} | {f(v['longest_frozen_s'], 1)} | "
                      f"{f(v['frac_extreme_head'], 3)} | {'YES' if v['drifts'] else 'no'} |")
    if ST:
        md.append("\n## Style vector (set_style) — direction check vs the neutral run (Seamless clips)\n")
        keys = ["head_speed_listen", "head_speed_speak", "head_motion_sd_deg", "face_sd", "smile_xcorr_gain", "head_follow_xcorr"]
        md.append("| setting | " + " | ".join(keys) + " |")
        md.append("|---|" + "---|" * len(keys))
        md.append("| neutral (null = infer) | " + " | ".join(f(ST["neutral"].get(k), 3) for k in keys) + " |")
        for s in ST["settings"]:
            md.append(f"| {s['style']} | " + " | ".join(f(s.get(k), 3) for k in keys) + " |")
    sbs = sorted((out / "sbs").glob("*.mp4")) if (out / "sbs").exists() else []
    if sbs:
        md.append("\n## Side-by-side videos (user | rules | unified | human)\n")
        for p in sbs:
            md.append(f"- {p}")
    md.append(f"\nPaid-judge cost so far (who=eval, $E/spend.jsonl): ${eval_spend():.4f}.")
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    res["spend_usd"] = eval_spend()
    atomic_json(out / "results.json", res)
    # ------------------------------------------------------------------ page (self-contained; videos by path)
    body = []
    import re
    for line in md:
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-") for c in cells):
                continue
            tag = "th" if body and not body[-1].startswith("<tr") and not body[-1].startswith("<table") else "td"
            if not body or not (body[-1].startswith("<tr") or body[-1] == "<table>"):
                body.append("<table>")
                tag = "th"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if body and body[-1].startswith("<tr"):
                body.append("</table>")
            t = html.escape(line)
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
            if t.startswith("- "):
                body.append(f"<p class=li>• {t[2:]}</p>")
            elif t.strip():
                body.append(f"<p>{t}</p>")
    if body and body[-1].startswith("<tr"):
        body.append("</table>")
    page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>unified eval</title><style>body{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:auto;"
            "padding:12px;background:#111;color:#eee}table{border-collapse:collapse;margin:8px 0;font-size:13px;display:block;"
            "overflow-x:auto}td,th{border:1px solid #444;padding:4px 7px;text-align:left}th{background:#222}code{color:#9cf}"
            "h1{font-size:20px}h2{font-size:17px;margin-top:22px;border-bottom:1px solid #333}p.li{margin:2px 0 2px 12px}"
            "</style>" + "\n".join(body))
    (out / "index.html").write_text(page)
    print(verdict)
    print(out / "REPORT.md")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmd", default="")
    a = ap.parse_args()
    build(Path(a.out), a.cmd)


if __name__ == "__main__":
    main()
