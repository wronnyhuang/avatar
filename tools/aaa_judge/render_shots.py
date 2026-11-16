#!/usr/bin/env python3
"""render_shots.py -- the fixed AAA-judge shot list at 1920x1080 through renderer/tools/juno_still.py.

  render_shots.py --worktree WT --char juno|junor --asset DIR --out DIR [--cams closeup,bust] [--ids a,b] [--jobs 4] [--threads 16]

Writes DIR/<cam>/<shot_id>.png (1920x1080, in the classroom scene) for cam in closeup (--cam face: head ~70 % of the
frame height) and bust (--cam bust3: head ~33 %).  Shots = shots_aaa.json (neutral, soft smile, teeth smile, brows up, half blink,
look aside, head turn 25, visemes ah / oo / mm).  CPU llvmpipe by default (~25 s a shot, 3-4 in parallel on the hub's 70 cores);
--device gpu needs `flock` of the K620 lock (see juno_still.py's docstring) and is only worth it for a handful of shots.
--asset may be omitted for juno (the asset the char loads by default).
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = {'closeup': 'face', 'bust': 'bust3'}
ap = argparse.ArgumentParser()
ap.add_argument('--worktree', required=True); ap.add_argument('--char', default='juno'); ap.add_argument('--asset', default='')
ap.add_argument('--out', required=True); ap.add_argument('--cams', default='closeup,bust'); ap.add_argument('--ids', default='')
ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--device', default='cpu'); ap.add_argument('--extra', default='')
ap.add_argument('--python', default=sys.executable)
a = ap.parse_args()
shots = json.load(open(f'{HERE}/shots_aaa.json'))['shots']
want = set(filter(None, a.ids.split(',')))
jobs = []
for cam in a.cams.split(','):
    os.makedirs(f'{a.out}/{cam}', exist_ok=True); os.makedirs(f'{a.out}/params', exist_ok=True)
    for s in shots:
        if want and s['id'] not in want: continue
        pj = f'{a.out}/params/{s["id"]}.json'; json.dump(s['params'], open(pj, 'w'), indent=1)
        cmd = [a.python, f'{a.worktree}/renderer/tools/juno_still.py', '--out', f'{a.out}/{cam}/{s["id"]}.png', '--w', '1920', '--h', '1080',
               '--cam', CAMS[cam], '--params', pj, '--device', a.device, '--threads', str(a.threads), '--char', a.char]
        if a.asset: cmd += ['--asset', a.asset]
        jobs.append((cam, s['id'], cmd + a.extra.split()))
def run(j):
    cam, sid, cmd = j; t = time.time()
    r = subprocess.run(['nice', '-n', '10'] + cmd, capture_output=True, text=True, cwd=a.worktree)
    print(cam, sid, r.returncode, f'{time.time()-t:.0f}s', (r.stderr[-300:] if r.returncode else ''), flush=True)
    return r.returncode
with ThreadPoolExecutor(a.jobs) as ex:
    rc = list(ex.map(run, jobs))
sys.exit(1 if any(rc) else 0)
