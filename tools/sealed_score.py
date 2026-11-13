#!/usr/bin/env python3
"""Planner: score ONE config on a SEALED set without the builder seeing any item.

  tools/sealed_score.py --set <sealed.jsonl> --config <name> --results <dir> [--root <tree>] [--max-usd 2] [--jobs 3 --threads 2]

Runs tools/plan_eval.py with every per-item line, the failure list and summary.md going to <results>/<config>/ (run.log,
summary.md, item folders — nobody opens them before the decision) and prints ONLY aggregates read from summary.json:
n, success % (judge score >= pass), strict %, mean, first motion p50/p90 (all / planner-sourced), plan ms, $/plan,
sources, auto-fails, hedged fallback use, and the per-category success %. Re-run = resume (plan_eval is resumable).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.environ.get('PYTHON') or sys.executable
KEYS = ('n', 'scored', 'success_pct', 'strict_pct', 'mean_score', 'first_motion_ms_p50', 'first_motion_ms_p90',
        'planner_first_motion_ms_p50', 'planner_first_motion_ms_p90', 'plan_ms_p50', 'plan_ms_p90', 'usd_per_plan',
        'plan_usd', 'judge_usd', 'auto_fail', 'fallback_used', 'sources')


def aggregates(out: Path) -> dict:
    S = json.loads((out / 'summary.json').read_text())
    o = S.get('overall') or {}
    cats = {}
    for name, g in (S.get('categories') or S.get('by_category') or {}).items():
        if isinstance(g, dict) and 'success_pct' in g and '/' not in name:
            cats[name] = g.get('success_pct')
    return {'config': S.get('config'), 'judge': S.get('judge'), 'root_sha': S.get('root_sha'),
            'overall': {k: o.get(k) for k in KEYS if k in o}, 'per_category_success_pct': cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True); ap.add_argument('--config', required=True); ap.add_argument('--results', required=True)
    ap.add_argument('--root'); ap.add_argument('--max-usd', type=float, default=2.0); ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--plan-jobs', type=int, default=2)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--paired-with', help='another config in the same results dir: print paired pass wins / losses (counts only)')
    a = ap.parse_args()
    out = Path(a.results) / a.config
    out.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        cmd = [PY, str(HERE / 'plan_eval.py'), '--set', a.set, '--config', a.config, '--out', str(out), '--jobs', str(a.jobs),
               '--threads', str(a.threads), '--plan-jobs', str(a.plan_jobs), '--max-usd', str(a.max_usd), '--who', 'sealed3']
        if a.root:
            cmd += ['--root', a.root]
        with open(out / 'run.log', 'a') as f:
            f.write('\n=== ' + ' '.join(cmd) + '\n'); f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f'plan_eval rc={rc} (log in the results dir, not printed)', flush=True)
    print(json.dumps(aggregates(out), indent=1))
    if a.paired_with:
        print(json.dumps(paired(out, Path(a.results) / a.paired_with), indent=1))


def paired(a_dir: Path, b_dir: Path) -> dict:
    """counts only: items both scored; a pass / b fail, a fail / b pass, both pass, both fail; mean score diff"""
    def m(d):
        S = json.loads((d / 'summary.json').read_text())
        thr = S.get('pass_score', 3)
        return {r['id']: r.get('score') for r in S.get('items') or [] if r.get('score') is not None}, thr
    A, thr = m(a_dir)
    B, _ = m(b_dir)
    ids = sorted(set(A) & set(B))
    w = sum(A[i] >= thr > B[i] for i in ids); l = sum(B[i] >= thr > A[i] for i in ids)
    return {'a': a_dir.name, 'b': b_dir.name, 'shared': len(ids), 'a_wins': w, 'b_wins': l,
            'both_pass': sum(A[i] >= thr and B[i] >= thr for i in ids), 'both_fail': sum(A[i] < thr and B[i] < thr for i in ids),
            'mean_score_diff_a_minus_b': round(sum(A[i] - B[i] for i in ids) / max(1, len(ids)), 3)}


if __name__ == '__main__':
    main()
