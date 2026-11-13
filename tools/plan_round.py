#!/usr/bin/env python3
"""Planner search: run one ROUND (a list of plan_eval configs on one row set) and table it.

  plan_round.py run   --round <dir> --set <jsonl> --configs a,b,c [--judge mini-zoom] [--parallel 3] [--jobs 8] [--plan-jobs 3]
                      [--max-usd-each 1.0] [--root <tree>] [--configs-file tools/plan_configs.json]
  plan_round.py table --round <dir> [--pass 3] [--paired]

run: each config -> <round>/<config>/ via tools/plan_eval.py (resumable: plan_eval skips every finished stage, so re-running
the same command after a kill continues; finished configs cost nothing). `table` reads every <round>/*/summary.json and writes
<round>/TABLE.md: per config success % (judge score >= pass), strict % (>= 4), mean, first motion p50/p90 (planner rows), plan
done p50/p90, $/plan; per category success; and (--paired) the paired win/loss of each config vs the best on shared items.
"""
import argparse, json, os, statistics, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.environ.get('PYTHON') or sys.executable


def run(a):
    rd = Path(a.round); rd.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in a.configs.split(',') if c]

    def one(c):
        out = rd / c; out.mkdir(exist_ok=True)
        cmd = [PY, str(HERE / 'plan_eval.py'), '--set', a.set, '--config', c, '--out', str(out), '--judge', a.judge,
               '--jobs', str(a.jobs), '--threads', str(a.threads), '--plan-jobs', str(a.plan_jobs), '--max-usd', str(a.max_usd_each), '--who', a.who]
        if a.root: cmd += ['--root', a.root]
        if a.ids: cmd += ['--ids', a.ids]
        if a.configs_file: cmd += ['--configs', a.configs_file]
        with open(out / 'run.log', 'a') as f:
            f.write(f"\n=== {' '.join(cmd)}\n"); f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'{c}: rc={r.returncode}', flush=True)
        return c, r.returncode

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(one, cfgs))
    table(argparse.Namespace(round=a.round, pass_score=3, paired=True))
    print('ROUND-DONE', res, flush=True)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    return int(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))])


def table(a):
    rd = Path(a.round); P = a.pass_score
    S = {}
    for f in sorted(rd.glob('*/summary.json')):
        try: S[f.parent.name] = json.load(open(f))
        except Exception: pass
    if not S:
        print('no summaries yet'); return
    rows, cats = [], sorted({it['category'] for s in S.values() for it in s['items']})
    for c, s in S.items():
        its = [it for it in s['items'] if it.get('score') is not None]
        n = len(its)
        ok = sum(it['score'] >= P for it in its); st = sum(it['score'] >= 4 for it in its)
        fm = [it.get('first_motion_ms') for it in its if it.get('source') in ('planner', 'library')]
        dn = [it.get('ms_done') for it in its if it.get('source') in ('planner', 'library')]
        usd = [float(it.get('usd') or 0) for it in its if it.get('source') == 'planner']
        bycat = {k: [it['score'] >= P for it in its if it['category'] == k] for k in cats}
        rows.append({'config': c, 'n': n, 'ok': ok, 'succ': 100 * ok / max(1, n), 'strict': 100 * st / max(1, n),
                     'mean': statistics.mean([it['score'] for it in its]) if its else 0, 'fm50': pct(fm, .5), 'fm90': pct(fm, .9),
                     'dn50': pct(dn, .5), 'dn90': pct(dn, .9), 'usd': statistics.mean(usd) if usd else 0,
                     'judge': s.get('judge'), 'auto': sum(1 for it in its if it.get('auto')),
                     'cat': {k: (100 * sum(v) / len(v) if v else None) for k, v in bycat.items()},
                     'scores': {it['id']: it['score'] for it in its}})
    rows.sort(key=lambda r: -r['succ'])
    L = [f'# round {rd.name} — judge {rows[0]["judge"]}, pass >= {P}', '', 'DEV numbers are for tuning only (the dev set was looked at while tuning); headline numbers come from a sealed held-out set (tools/sealed_score.py).', '',
         '| config | n | success % | strict % | mean | auto-fail | first motion p50/p90 ms | done p50/p90 ms | $/plan |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['config']} | {r['n']} | {r['succ']:.1f} | {r['strict']:.1f} | {r['mean']:.2f} | {r['auto']} | {r['fm50']}/{r['fm90']} | {r['dn50']}/{r['dn90']} | {r['usd']:.5f} |")
    L += ['', '| config | ' + ' | '.join(cats) + ' |', '|---|' + '---|' * len(cats)]
    for r in rows:
        L.append(f"| {r['config']} | " + ' | '.join('-' if r['cat'][k] is None else f"{r['cat'][k]:.0f}" for k in cats) + ' |')
    if a.paired and len(rows) > 1:
        best = rows[0]
        L += ['', f"paired vs the top config `{best['config']}` on shared items (wins / losses / ties at pass >= {P}; mean score delta):", '']
        for r in rows[1:]:
            sh = set(best['scores']) & set(r['scores'])
            w = sum((r['scores'][i] >= P) and not (best['scores'][i] >= P) for i in sh)
            l = sum((best['scores'][i] >= P) and not (r['scores'][i] >= P) for i in sh)
            dm = statistics.mean([r['scores'][i] - best['scores'][i] for i in sh]) if sh else 0
            L.append(f"- {r['config']}: n={len(sh)} wins {w} / losses {l} / ties {len(sh) - w - l}; mean delta {dm:+.2f}")
    (rd / 'TABLE.md').write_text('\n'.join(L) + '\n')
    json.dump([{k: v for k, v in r.items() if k != 'scores'} for r in rows], open(rd / 'table.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run'); r.add_argument('--round', required=True); r.add_argument('--set', required=True)
    r.add_argument('--configs', required=True); r.add_argument('--judge', default='mini-zoom'); r.add_argument('--parallel', type=int, default=3)
    r.add_argument('--jobs', type=int, default=8); r.add_argument('--threads', type=int, default=2); r.add_argument('--plan-jobs', type=int, default=3); r.add_argument('--max-usd-each', type=float, default=1.0)
    r.add_argument('--root'); r.add_argument('--ids', help='a,b,c or a file with one id per line'); r.add_argument('--configs-file'); r.add_argument('--who', default='planner')
    t = sp.add_parser('table'); t.add_argument('--round', required=True); t.add_argument('--pass', dest='pass_score', type=int, default=3)
    t.add_argument('--paired', action='store_true')
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else table(a)
