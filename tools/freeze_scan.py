#!/usr/bin/env python3
"""freeze_scan.py — find frozen-avatar intervals in a recorded session video.

usage: freeze_scan.py FILE [--region left:0.735] [--thr 0.02] [--min-s 1.0] [--gap-s 0.5] [--json]

The recording composites the avatar stage and the user's camera inset (bottom-right corner); only the avatar
region is scanned (default: the left 73.5% of the frame, full height, which excludes the inset). Every frame is
decoded to 160 px wide grey; a frame is "unchanged" when the mean absolute difference to the previous frame is
below --thr (0..255 grey levels; 0.02 = no visible change). A freeze interval is a run of unchanged frames, plus any
gap between frame timestamps longer than --gap-s (a canvas-capture recorder emits no frame while nothing is drawn,
e.g. while the page is hidden), lasting at least --min-s; freezes separated by at most --merge-s of change (a ticking status line on the
placeholder, one stray frame) are joined and the joined frames counted as "blips". A 24 fps source recorded at 30 fps repeats about one frame
in five, which yields runs of 1-2 unchanged frames: --min-s 1.0 is well above that.

Prints one line per interval: start_s, end_s, duration_s, frames, kind (static | gap) and the mean grey level of the
region during the interval (near 0 = black), the hidden-page gap time inside it and the blip count. Needs numpy and ffmpeg/ffprobe on PATH.
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                          '-of', 'csv=p=0:s=x', path], capture_output=True, text=True).stdout.strip()
    w, h = [int(v) for v in out.split('\n')[0].split('x')[:2]]
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'frame=best_effort_timestamp_time,pts_time', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout
    ts = []
    for line in out.split('\n'):
        for x in line.split(','):
            try:
                ts.append(float(x))
                break
            except ValueError:
                pass
    return w, h, np.array(ts)


def region_crop(w, h, spec):
    kind, _, val = spec.partition(':')
    if kind == 'left':
        cw = int(w * float(val)) // 2 * 2
        return 0, 0, cw, h
    if kind == 'box':  # box:x:y:w:h in pixels
        x, y, cw, ch = [int(v) for v in val.split(':')]
        return x, y, cw, ch
    if kind == 'full':
        return 0, 0, w, h
    raise SystemExit('bad --region ' + spec)


def frames_gray(path, crop, sw=160):
    x, y, cw, ch = crop
    sh = max(2, int(round(ch * sw / cw / 2)) * 2)
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-fps_mode', 'passthrough', '-vf',
           f'crop={cw}:{ch}:{x}:{y},scale={sw}:{sh}', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    raw = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.uint8)
    n = len(raw) // (sw * sh)
    return raw[:n * sw * sh].reshape(n, sh, sw)


def scan(path, region='left:0.735', thr=0.02, min_s=1.0, gap_s=0.5, merge_s=0.25):
    w, h, ts = probe(path)
    fr = frames_gray(path, region_crop(w, h, region))
    n = min(len(fr), len(ts))
    fr, ts = fr[:n], ts[:n]
    ts = ts - ts[0] if n else ts
    d = np.abs(np.diff(fr.astype(np.int16), axis=0)).mean(axis=(1, 2)) if n > 1 else np.zeros(0)
    dt = np.diff(ts)
    still = (d < thr) | (dt > gap_s)          # step i -> i+1 shows no change on screen
    runs = []                                  # [i, j]: frames i..j identical (or separated by a gap)
    i = 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        j = i
        while j < len(still) and still[j]:
            j += 1
        runs.append([i, j])
        i = j
    # a placeholder status line that ticks (an elapsed-seconds counter) or one stray changed frame splits a freeze:
    # join runs separated by <= merge_s of change
    # only runs that are already a visible freeze (>= core_s) are joined: a 24 fps source recorded at 30 fps gives
    # 1-2 frame repeat runs every few frames, which must never chain together
    core_s = min(0.5, min_s / 2)
    runs = [r for r in runs if ts[r[1]] - ts[r[0]] >= core_s]
    merged = []
    for r in runs:
        if merged and ts[r[0]] - ts[merged[-1][1]] <= merge_s:
            merged[-1][1] = r[1]
            merged[-1][2] += 1
        else:
            merged.append([r[0], r[1], 0])
    out = []
    for i, j, joins in merged:
        t0, t1 = float(ts[i]), float(ts[j])
        if t1 - t0 >= min_s:
            gaps = dt[i:j] > gap_s
            kind = 'gap' if gaps.sum() and dt[i:j][gaps].sum() > 0.5 * (t1 - t0) else 'static'
            changed = int((~still[i:j]).sum())
            out.append({'start_s': round(t0, 2), 'end_s': round(t1, 2), 'dur_s': round(t1 - t0, 2),
                        'frames': int(j - i + 1), 'kind': kind, 'luma': round(float(fr[i:j + 1].mean()), 1),
                        'gap_s': round(float(dt[i:j][gaps].sum()), 2), 'blips': changed})
    summ = {'file': path.rsplit('/', 1)[-1], 'span_s': round(float(ts[-1]), 2) if n else 0, 'frames': int(n),
            'frozen_s': round(sum(x['dur_s'] for x in out), 2),
            'longest_s': max([x['dur_s'] for x in out] or [0]), 'intervals': len(out)}
    return summ, out


def label(out, session_path, video_path, offset_s=None):
    """Name the session stage(s) each freeze falls in: 'warming: box' / 'warming: engine' / 'switching' (an expected
    no-video wait), 'hidden' (a recorder gap: the page drew nothing), or 'live' (a stall candidate)."""
    import datetime as _dt
    import os as _os
    d = json.load(open(session_path))
    sl = d.get('status_log') or []
    if offset_s is None:
        offset_s = 0.0
        side = _os.path.splitext(video_path)[0] + '.json'
        try:
            rec_ts = float(json.load(open(side))['started_ts'])
            ses_ts = _dt.datetime.fromisoformat(d['started']).timestamp()
            offset_s = rec_ts - ses_ts
        except (OSError, KeyError, ValueError, TypeError):
            pass

    def stage_at(t):
        st = ('warming', '')
        for e in sl:
            if e[0] <= t:
                st = (e[1], e[2])
        if st[0] == 'warming':
            return 'warming: ' + ('box' if any(w in st[1] for w in ('box', 'container', 'bootstrap', 'pod')) else 'engine')
        return st[0]

    for x in out:
        if x['kind'] == 'gap':
            x['stage'] = 'hidden'
            continue
        names = []
        t = x['start_s'] + offset_s
        while t <= x['end_s'] + offset_s:
            n = stage_at(t)
            if not names or names[-1] != n:
                names.append(n)
            t += 0.5
        x['stage'] = ' -> '.join(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='+')
    ap.add_argument('--region', default='left:0.735', help='left:FRAC | box:x:y:w:h | full')
    ap.add_argument('--thr', type=float, default=0.02)
    ap.add_argument('--min-s', type=float, default=1.0)
    ap.add_argument('--gap-s', type=float, default=0.5)
    ap.add_argument('--merge-s', type=float, default=0.25, help='join freezes separated by <= this much change')
    ap.add_argument('--session', help="the hub's session record (session-*.json): label each freeze with the session "
                    "stage it falls in (status_log: warming = box / engine wait, switching, live)")
    ap.add_argument('--offset-s', type=float, default=None, help='recording start minus session start, seconds (default: '
                    'from the recording sidecar .json started_ts and the record\'s start time, else 0)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    for f in a.files:
        summ, out = scan(f, a.region, a.thr, a.min_s, a.gap_s, a.merge_s)
        if a.session:
            label(out, a.session, f, a.offset_s)
        if a.json:
            print(json.dumps({**summ, 'list': out}))
            continue
        print(f"{summ['file']}: span {summ['span_s']} s, {summ['frames']} frames, {summ['intervals']} freezes >= "
              f"{a.min_s} s, frozen {summ['frozen_s']} s, longest {summ['longest_s']} s")
        for x in out:
            print(f"  {x['start_s']:8.2f} - {x['end_s']:8.2f}  {x['dur_s']:7.2f} s  {x['frames']:5d} fr  "
                  f"{x['kind']:6s} luma {x['luma']:5.1f}  hidden/gap {x['gap_s']} s  blips {x['blips']}"
                  + (f"  [{x['stage']}]" if 'stage' in x else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
