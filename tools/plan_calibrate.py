#!/usr/bin/env python3
"""Judge calibration: hand labels vs judge candidates on the calibration clips.

  tools/plan_calibrate.py build   --calib <plan_eval out dir of calibration_candidates.jsonl> --set <calibration_candidates.jsonl> --out <planeval>
        -> <out>/calibset.jsonl: every rendered calibration clip with its own item ('real'), deliberate negatives: a clip
           judged against ANOTHER item's request ('mismatch'), and an idle clip with no motion ('nomotion')
  (then hand-label: <out>/calib/hand_labels.jsonl, one {"cid", "label": "pass|fail", "score": 1-5, "borderline", "note"} per line — BEFORE any judge runs)
  tools/plan_calibrate.py run     --out <planeval> --judges gem38-video,gem35-video,sol-strips [--jobs 4]
  tools/plan_calibrate.py report  --out <planeval>           -> table on stdout + <out>/calibration_report.json
Verdicts: <out>/calib_verdicts/<judge>/<cid>.json. Refuses to run a judge while a calibset clip has no label.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = os.environ.get('PYTHON') or sys.executable
# (clip of item A, judged against item B): clearly different actions; the last two are face items (close-up included)
MISMATCH = [('dev-0001', 'dev-0029'), ('dev-0029', 'dev-0021'), ('dev-0009', 'dev-0001'), ('dev-0057', 'dev-0067'), ('dev-0077', 'dev-0061'),
            ('dev-0217', 'dev-0211'), ('dev-0115', 'dev-0147'), ('dev-0147', 'dev-0155')]
NOMOTION = ['dev-0039', 'dev-0115']     # an idle clip judged against a body request and a face request


def labels_path(a):
    return Path(a.labels) if a.labels else Path(a.out) / 'calib' / 'hand_labels.jsonl'


def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def build(a):
    out, calib = Path(a.out), Path(a.calib)
    items = {x['id']: x for x in load(a.set)}
    rows = []
    for i, it in items.items():
        rj = calib / i / 'render.json'
        if rj.exists() and json.loads(rj.read_text()).get('ok'):
            rows.append({'cid': i, 'kind': 'real', 'clip_dir': str(calib / i), 'item': it})
    have = {r['cid'] for r in rows}
    for ca, ib in MISMATCH:
        if ca in have and ib in items:
            it = dict(items[ib])
            if it.get('face_closeup') and not (calib / ca / 'face.mp4').exists():
                continue
            rows.append({'cid': f'neg-{ca[4:]}-as-{ib[4:]}', 'kind': 'mismatch', 'clip_dir': str(calib / ca), 'item': it, 'clip_of': ca})
    idle = out / 'calib_idle'
    idle.mkdir(parents=True, exist_ok=True)
    if not (idle / 'render.json').exists():
        (idle / 'plan.json').write_text(json.dumps({'id': 'idle', 'source': 'planner', 'ok': False, 'poses': [], 'ms_done': 2500}))
        subprocess.run(['node', str(HERE / 'plan_sim.cjs'), str(idle / 'plan.json'), str(idle)], check=True)
        subprocess.run([PY, str(HERE / 'plan_render.py'), str(idle), '--face', '--threads', '8'], check=True)
    for ib in NOMOTION:
        rows.append({'cid': f'neg-idle-as-{ib[4:]}', 'kind': 'nomotion', 'clip_dir': str(idle), 'item': items[ib]})
    (out / 'calibset.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} calibration clips -> {out}/calibset.jsonl', {k: sum(r["kind"] == k for r in rows) for k in ("real", "mismatch", "nomotion")})


def run(a):
    import plan_judge
    out = Path(a.out)
    rows = load(out / 'calibset.jsonl')
    labels = {x['cid'] for x in load(labels_path(a))} if labels_path(a).exists() else set()
    missing = [r['cid'] for r in rows if r['cid'] not in labels]
    if missing:
        sys.exit(f'labels first (blind to the judges): {len(missing)} clips unlabelled, e.g. {missing[:5]}')
    for jn in a.judges.split(','):
        vd = out / 'calib_verdicts' / jn
        vd.mkdir(parents=True, exist_ok=True)
        todo = [r for r in rows if not (vd / f"{r['cid']}.json").exists()]
        t0 = time.time()

        def one(r):
            v = plan_judge.judge_item(r['item'], r['clip_dir'], jn)
            if v.get('score') is not None:
                (vd / f"{r['cid']}.json").write_text(json.dumps(v, indent=1))
            return v
        with ThreadPoolExecutor(a.jobs) as ex:
            vs = list(ex.map(one, todo))
        usd = sum(v.get('usd') or 0 for v in vs)
        errs = [v.get('error') for v in vs if v.get('score') is None]
        print(f'{jn}: {len(todo)} new verdicts, {len(errs)} errors, ${usd:.3f}, {time.time() - t0:.0f}s' + (f' — first error: {errs[0][:200]}' if errs else ''), flush=True)


def stats(rows, labels, verdicts, thr):
    n = fp = fn = agree = 0
    neg = pos = 0
    diffs = []
    for r in rows:
        L, v = labels.get(r['cid']), verdicts.get(r['cid'])
        if not L or not v:
            continue
        n += 1
        lp, jp = L['label'] == 'pass', v['score'] >= thr
        pos += lp
        neg += not lp
        agree += lp == jp
        fp += (not lp) and jp
        fn += lp and not jp
        diffs.append(abs(v['score'] - L['score']))
    if not n:
        return None
    return {'n': n, 'agree_pct': round(100 * agree / n, 1), 'false_pass': f'{fp}/{neg}', 'false_fail': f'{fn}/{pos}', 'false_pass_pct': round(100 * fp / neg, 1) if neg else None,
            'false_fail_pct': round(100 * fn / pos, 1) if pos else None, 'mean_abs_score_diff': round(sum(diffs) / n, 2), 'within1_pct': round(100 * sum(d <= 1 for d in diffs) / n, 1)}


def report(a):
    out = Path(a.out)
    rows = load(out / 'calibset.jsonl')
    labels = {x['cid']: x for x in load(labels_path(a))}
    R = {'n_clips': len(rows), 'labels': {k: sum(labels[r['cid']]['label'] == k for r in rows if r['cid'] in labels) for k in ('pass', 'partial', 'fail')}, 'judges': {}}
    face = [r for r in rows if r['item'].get('face_closeup')]
    print(f"{len(rows)} clips; my labels: {R['labels']}; face close-up subset: {len(face)}")
    for vd in sorted((out / 'calib_verdicts').glob('*')):
        V = {p.stem: json.loads(p.read_text()) for p in vd.glob('*.json')}
        if not V:
            continue
        J = {'verdicts': len(V), 'usd_per_clip': round(sum(v.get('usd_first') or 0 for v in V.values()) / len(V), 4),
             'sec_per_clip': round(sum(v.get('seconds') or 0 for v in V.values()) / len(V), 1)}
        for thr in (3, 4, 5):
            J[f'pass>={thr}'] = {'all': stats(rows, labels, V, thr), 'clear_only': stats([r for r in rows if not labels.get(r['cid'], {}).get('borderline')], labels, V, thr), 'face': stats(face, labels, V, thr), 'real_only': stats([r for r in rows if r['kind'] == 'real'], labels, V, thr),
                                 'negatives': stats([r for r in rows if r['kind'] != 'real'], labels, V, thr)}
        J['disagreements@4'] = [{'cid': r['cid'], 'mine': labels[r['cid']]['score'], 'judge': V[r['cid']]['score'], 'why': V[r['cid']].get('reason', '')[:200]}
                                for r in rows if r['cid'] in V and r['cid'] in labels and (labels[r['cid']]['label'] == 'pass') != (V[r['cid']]['score'] >= 4)]
        R['judges'][vd.name] = J
        print(f"\n== {vd.name}: {J['verdicts']} verdicts, ${J['usd_per_clip']}/clip, {J['sec_per_clip']} s/clip")
        for thr in (3, 4, 5):
            for k in ('all', 'clear_only', 'real_only', 'face', 'negatives'):
                s = J[f'pass>={thr}'][k]
                if s:
                    print(f"   pass>={thr} {k:10s} n={s['n']:2d} agree {s['agree_pct']}%  false-pass {s['false_pass']}  false-fail {s['false_fail']}  |score diff| {s['mean_abs_score_diff']}  within1 {s['within1_pct']}%")
        for d in J['disagreements@4']:
            print(f"     x {d['cid']}: mine {d['mine']} judge {d['judge']} — {d['why'][:150]}")
    (out / 'calibration_report.json').write_text(json.dumps(R, indent=1))
    # the table for CALIBRATION.md (the pick and the notes are written by hand under it)
    md = ['| judge | verdicts | pass >= | agreement | false-pass (my fail, judge pass) | false-fail (my pass, judge fail) | agreement, clear clips only | false-pass, clear only | face subset agreement | within 1 point | $/clip | s/clip |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for jn, J in R['judges'].items():
        for thr in (3, 4, 5):
            A, Cl, F = J[f'pass>={thr}']['all'], J[f'pass>={thr}']['clear_only'], J[f'pass>={thr}']['face']
            if not A:
                continue
            md.append(f"| {jn} | {J['verdicts']} | {thr} | {A['agree_pct']}% | {A['false_pass']} ({A['false_pass_pct']}%) | {A['false_fail']} ({A['false_fail_pct']}%) | "
                      f"{Cl['agree_pct'] if Cl else '-'}% (n={Cl['n'] if Cl else 0}) | {Cl['false_pass'] if Cl else '-'} | {F['agree_pct'] if F else '-'}% (n={F['n'] if F else 0}) | {A['within1_pct']}% | {J['usd_per_clip']} | {J['sec_per_clip']} |")
    (out / 'calibration_table.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'run', 'report'])
    ap.add_argument('--out', default=str(HERE.parent / 'runs' / 'planeval'), help='planeval dir (default <repo>/runs/planeval)'); ap.add_argument('--calib'); ap.add_argument('--set')
    ap.add_argument('--labels'); ap.add_argument('--judges', default='gem38-video,gem35-video,sol-strips'); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    {'build': build, 'run': run, 'report': report}[a.cmd](a)
