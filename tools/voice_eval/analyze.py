#!/usr/bin/env python3
"""Summarize voice-eval runs.

  python3 -I analyze.py <run_dir> [<run_dir> ...] [--json combined.json] [--tol 0.6] [--include-extra]

Per run dir (written by run_eval.js): reads report.json (+ report_after_stop.json for final usage/cost),
run_meta.json and the mic manifest it names. Prints markdown: latency stats (n, p50, p95 nearest-rank, max) for
turn tap_lat_ms / audible_lat_ms and call start_lat_ms / done_lat_ms, delay_ms, video RTT, usd_est and $/min,
then a turn-vs-clip cross-check and a command-vs-call table. Writes <run_dir>/results.json; --json also
writes all runs combined.

Cross-check: each turn's user_end_ms is mapped to WAV time via the page's audio getUserMedia resolve time
(t_wav = (user_end_ms - gum_ms)/1000; validated to +-25 ms on the test page), then a single offset is fitted
from the turn/clip END spacing (the offset = the page VAD's detection lag). A turn with no clip end within
--tol s of (t_wav - offset) is flagged (split / spurious); a clip with 2+ turns is flagged split, one with none
missed/merged. With a looping mic (run_meta.mic_loop) turns that land in the second pass are labelled
'extra (loop N)' and left out of the stats unless --include-extra.
"""
import argparse, json, math, os, sys


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def pct(vals, p):
    v = sorted(vals)
    if not v:
        return None
    return v[max(1, math.ceil(p / 100 * len(v))) - 1]


def stats(vals):
    v = [x for x in (num(x) for x in vals) if x is not None]
    return {'n': len(v), 'p50': pct(v, 50), 'p95': pct(v, 95), 'max': max(v) if v else None}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(v, d=0):
    if v is None:
        return '-'
    return f'{v:.{d}f}' if isinstance(v, float) or d else str(v)


def fit_offset(u, ends, tol, bounded):
    """Offset maximizing #turns within tol of some clip end; ties -> smallest total residual."""
    best = (-1, 0.0, 0.0)
    for ui in u:
        for e in ends:
            off = ui - e
            if bounded and not (-3.0 <= off <= 6.0):
                continue
            res = [min(abs(x - off - e2) for e2 in ends) for x in u]
            hit = [r for r in res if r <= tol]
            key = (len(hit), -sum(hit))
            if key > (best[0], -best[1]):
                best = (len(hit), sum(hit), off)
    if best[0] <= 0:
        return None
    off = best[2]
    # refine: median signed residual of the matched set
    sr = []
    for x in u:
        e = min(ends, key=lambda e2: abs(x - off - e2))
        if abs(x - off - e) <= tol:
            sr.append(x - off - e)
    sr.sort()
    return off + sr[len(sr) // 2] if sr else off


def analyze(run_dir, tol, include_extra):
    rep = load(os.path.join(run_dir, 'report.json')) or {}
    after = load(os.path.join(run_dir, 'report_after_stop.json')) or {}
    meta = load(os.path.join(run_dir, 'run_meta.json')) or {}
    man = load(meta['manifest']) if meta.get('manifest') else None
    if not isinstance(rep, dict):
        rep = {}
    if not isinstance(after, dict):
        after = {}
    base = rep if len(rep.get('turns') or []) >= len(after.get('turns') or []) else after
    turns = [t for t in (base.get('turns') or []) if isinstance(t, dict)]
    calls = [c for c in (base.get('calls') or []) if isinstance(c, dict)]
    gum = num((meta.get('gum_audio') or {}).get('t_res_ms'))
    session_s = num(meta.get('session_s'))
    loop = bool(meta.get('mic_loop'))
    out = {'run': run_dir, 'mode': rep.get('mode') or after.get('mode') or meta.get('mode'),
           'vad': rep.get('vad') or meta.get('vad'), 'state': rep.get('state'), 'session_s': session_s,
           'report_ok': bool(rep), 'n_turns': len(turns), 'n_calls': len(calls), 'flags': []}
    if not rep:
        out['flags'].append('no report.json (page never exposed voiceReport?)')

    # ---- cross-check turns vs manifest clips ----
    xc, turn_extra = [], [False] * len(turns)
    if man and man.get('clips'):
        D = num(man.get('duration_s')) or num((meta.get('mic_info') or {}).get('duration_s')) or 0
        npass = 1
        if loop and D and session_s:
            npass = int(session_s // D) + 1
        clips = []
        for k in range(npass):
            for c in man['clips']:
                if num(c.get('end_s')) is None:
                    continue
                clips.append({**c, 'loop': k + 1, 'end_w': c['end_s'] + k * D, 'start_w': (c.get('start_s') or c['end_s']) + k * D})
        ends = [c['end_w'] for c in clips]
        u = [((num(t.get('user_end_ms')) - gum) / 1000 if gum is not None and num(t.get('user_end_ms')) is not None else
              (num(t.get('user_end_ms')) / 1000 if num(t.get('user_end_ms')) is not None else None)) for t in turns]
        uu = [x for x in u if x is not None]
        off = fit_offset(uu, ends, tol, bounded=gum is not None) if uu and ends else None
        out['gum_ms'] = gum
        out['fit_offset_ms'] = round(off * 1000) if off is not None else None
        hits = {}
        for i, (t, x) in enumerate(zip(turns, u)):
            row = {'turn': i + 1, 't_wav_s': round(x, 3) if x is not None else None, 'text': t.get('text')}
            if x is None or off is None:
                row['flag'] = 'no user_end_ms' if x is None else 'no offset fit'
            else:
                j = min(range(len(clips)), key=lambda j: abs(x - off - ends[j]))
                r = x - off - ends[j]
                if abs(r) <= tol:
                    c = clips[j]
                    row.update(clip=c['id'], kind=c.get('kind'), clip_text=c.get('text'), loop=c['loop'],
                               resid_ms=round(r * 1000), detect_lag_ms=round((x - ends[j]) * 1000) if gum is not None else None)
                    hits.setdefault(j, []).append(i)
                    if c['loop'] > 1:
                        row['flag'] = f'extra (loop {c["loop"]})'; turn_extra[i] = True
                else:
                    row['flag'] = f'no clip end within {tol}s (nearest {clips[j]["id"]} {r:+.2f}s): split/spurious?'
            xc.append(row)
        for j, idxs in hits.items():
            if len(idxs) > 1:
                for i in idxs:
                    xc[i]['flag'] = (xc[i].get('flag', '') + f' split: {len(idxs)} turns on {clips[j]["id"]}').strip()
        last_t = session_s if session_s else (max(uu) if uu else 0)
        missed = [c['id'] + (f'(loop {c["loop"]})' if c['loop'] > 1 else '') for j, c in enumerate(clips)
                  if j not in hits and c['end_w'] + 3 < last_t]
        if missed and turns:
            out['flags'].append('clips with no turn (missed/merged): ' + ', '.join(missed))
        bad = [r for r in xc if r.get('flag') and not r['flag'].startswith('extra')]
        if bad:
            out['flags'].append(f'{len(bad)} turn(s) do not line up with a clip')
        # spacing check on consecutive matched turns
        m = [(r['t_wav_s'], ends[next(j for j, c in enumerate(clips) if c['id'] == r['clip'] and c['loop'] == r['loop'])])
             for r in xc if r.get('clip')]
        sp = [abs((b[0] - a[0]) - (b[1] - a[1])) * 1000 for a, b in zip(m, m[1:])]
        out['spacing_err_ms'] = {'max': round(max(sp)) if sp else None, 'p50': round(pct(sp, 50)) if sp else None, 'n': len(sp)}
        lags = [r['detect_lag_ms'] for r in xc if r.get('detect_lag_ms') is not None and not r.get('flag')]
        out['detect_lag_ms'] = stats(lags)
    else:
        out['flags'].append('no manifest: cross-check skipped')
    out['crosscheck'] = xc

    keep = [t for i, t in enumerate(turns) if include_extra or not turn_extra[i]]
    out['n_turns_stats'] = len(keep)
    out['tap_lat_ms'] = stats(t.get('tap_lat_ms') for t in keep)
    out['audible_lat_ms'] = stats(t.get('audible_lat_ms') for t in keep)
    # voice-side breakdown (added by voice): per clip kind, and the vendor events relative to the local end of speech
    kinds = {i: (xc[i].get('kind') if i < len(xc) else None) for i in range(len(turns))}
    for kd in ('question', 'command'):
        sel = [t for i, t in enumerate(turns) if kinds.get(i) == kd and (include_extra or not turn_extra[i])]
        out[f'tap_lat_ms_{kd}'] = stats(t.get('tap_lat_ms') for t in sel)
    for ev in ('speech_stopped', 'response_created', 'first_response_done', 'audio_started'):
        out[f'ev_{ev}_ms'] = stats((t.get('events') or {}).get(ev) for t in keep)

    # ---- calls vs command clips: attach each call to the latest turn ending before it ----
    tends = [(num(t.get('user_end_ms')), i) for i, t in enumerate(turns) if num(t.get('user_end_ms')) is not None]
    crow, used_calls = [], []
    for c in calls:
        s = num(c.get('start_ms'))
        ti = max([(e, i) for e, i in tends if s is not None and e <= s], default=(None, None))[1]
        x = xc[ti] if ti is not None and ti < len(xc) else {}
        extra = ti is not None and turn_extra[ti]
        row = {'name': c.get('name'), 'args': c.get('args'), 'turn': ti + 1 if ti is not None else None,
               'clip': x.get('clip'), 'clip_kind': x.get('kind'), 'clip_text': x.get('clip_text'),
               'start_lat_ms': num(c.get('start_lat_ms')), 'done_lat_ms': num(c.get('done_lat_ms'))}
        if x.get('kind') == 'question':
            row['flag'] = 'call on a question turn'
        if extra:
            row['flag'] = 'extra (loop 2+)'
        crow.append(row)
        if include_extra or not extra:
            used_calls.append(c)
    if man and man.get('clips'):
        called = {r['clip'] for r in crow if r.get('clip')}
        nocall = [r['clip'] for r in xc if r.get('kind') == 'command' and r.get('clip') not in called]
        if nocall:
            out['flags'].append('commands with no function call: ' + ', '.join(nocall))
    out['calls_detail'] = crow
    out['call_start_lat_ms'] = stats(c.get('start_lat_ms') for c in used_calls)
    out['call_done_lat_ms'] = stats(c.get('done_lat_ms') for c in used_calls)
    cmd_calls = [c for c, r in zip(calls, crow) if r.get('clip_kind') == 'command' and not r.get('flag')]
    out['cmd_call_start_lat_ms'] = stats(c.get('start_lat_ms') for c in cmd_calls)
    out['cmd_call_early_lat_ms'] = stats(c.get('early_lat_ms') for c in cmd_calls)
    out['cmd_call_done_lat_ms'] = stats(c.get('done_lat_ms') for c in cmd_calls)
    out['spontaneous_calls'] = sum(1 for r in crow if r.get('clip_kind') == 'question')

    # ---- delay, video, cost ----
    d = base.get('delay_ms')
    if isinstance(d, list):
        out['delay_ms'] = stats(d)
    elif isinstance(d, dict):
        out['delay_ms'] = d
    else:
        out['delay_ms'] = num(d)
    vr = base.get('video_rtt_ms') if isinstance(base.get('video_rtt_ms'), dict) else {}
    out['video_rtt_ms'] = {'p50': num(vr.get('p50')), 'p90': num(vr.get('p90')), 'n': vr.get('n')}
    usd = [num(r.get('usd_est')) for r in (rep, after) if num(r.get('usd_est')) is not None]
    out['usd_est'] = max(usd) if usd else None
    out['usage'] = after.get('usage') or rep.get('usage')
    out['usd_per_min'] = (out['usd_est'] / (session_s / 60)) if out['usd_est'] is not None and session_s else None
    return out


def md_run(r):
    L = [f'### {r["run"]}', '',
         f'mode **{r["mode"]}** vad {r["vad"] or "-"} · session {fmt(r["session_s"], 1)} s · turns {r["n_turns"]} '
         f'({r["n_turns_stats"]} in stats) · calls {r["n_calls"]} · final state {r["state"]}', '',
         '| metric | n | p50 | p95 | max |', '|---|---:|---:|---:|---:|']
    for k, lab in (('tap_lat_ms', 'end-of-speech → first tap (ms)'), ('audible_lat_ms', 'end-of-speech → first audible (ms)'),
                   ('call_start_lat_ms', 'end-of-speech → call start (ms)'), ('call_done_lat_ms', 'end-of-speech → call done (ms)'),
                   ('tap_lat_ms_question', '  … first tap, question turns'), ('tap_lat_ms_command', '  … first tap, command turns'),
                   ('ev_speech_stopped_ms', 'vendor speech_stopped (ms)'), ('ev_response_created_ms', 'vendor response.created (ms)'),
                   ('ev_first_response_done_ms', 'tool-only response done (ms)'), ('ev_audio_started_ms', 'vendor output_audio_buffer.started (ms)'),
                   ('cmd_call_start_lat_ms', 'command turns: call start (ms)'), ('cmd_call_early_lat_ms', 'command turns: early act (ms)'),
                   ('cmd_call_done_lat_ms', 'command turns: call done (ms)'),
                   ('detect_lag_ms', 'page end-of-speech − WAV end (ms)')):
        s = r.get(k)
        if s:
            L.append(f'| {lab} | {s["n"]} | {fmt(s["p50"])} | {fmt(s["p95"])} | {fmt(s["max"])} |')
    d = r['delay_ms']
    dtxt = json.dumps(d) if isinstance(d, dict) else fmt(d)
    v = r['video_rtt_ms']
    L += ['', f'delay_ms {dtxt} · video RTT p50 {fmt(v["p50"])} / p90 {fmt(v["p90"])} ms (n {v["n"] if v["n"] is not None else "-"}) · '
          f'usd_est {fmt(r["usd_est"], 4) if r["usd_est"] is not None else "-"} · '
          f'$/min {fmt(r["usd_per_min"], 4) if r["usd_per_min"] is not None else "-"}']
    if r.get('fit_offset_ms') is not None:
        sp = r.get('spacing_err_ms') or {}
        L.append(f'cross-check: fitted detect offset {r["fit_offset_ms"]} ms; turn-spacing vs clip-spacing error p50 '
                 f'{fmt(sp.get("p50"))} / max {fmt(sp.get("max"))} ms over {sp.get("n")} gaps')
    if r['crosscheck']:
        L += ['', '| turn | t_wav s | clip | lag ms | resid ms | page text | flag |', '|---:|---:|---|---:|---:|---|---|']
        for x in r['crosscheck']:
            txt = (x.get('text') or '')[:48].replace('|', '/').replace('\n', ' ')
            L.append(f'| {x["turn"]} | {fmt(x.get("t_wav_s"), 2) if x.get("t_wav_s") is not None else "-"} | {x.get("clip") or "-"} | '
                     f'{fmt(x.get("detect_lag_ms"))} | {fmt(x.get("resid_ms"))} | {txt} | {x.get("flag", "")} |')
    if r['calls_detail']:
        L += ['', '| call | args | turn | clip | start ms | done ms | flag |', '|---|---|---:|---|---:|---:|---|']
        for c in r['calls_detail']:
            a = json.dumps(c.get('args'))[:40].replace('|', '/')
            L.append(f'| {c.get("name")} | {a} | {c.get("turn") or "-"} | {c.get("clip") or "-"} | {fmt(c.get("start_lat_ms"))} | '
                     f'{fmt(c.get("done_lat_ms"))} | {c.get("flag", "")} |')
    if r['flags']:
        L += [''] + [f'- ⚠ {f}' for f in r['flags']]
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--json', help='also write all runs combined to this path')
    ap.add_argument('--tol', type=float, default=0.6, help='turn/clip end match tolerance, s (after the offset fit)')
    ap.add_argument('--include-extra', action='store_true', help='keep loop-2+ turns in the stats')
    a = ap.parse_args()
    res = []
    for d in a.runs:
        if not os.path.isdir(d):
            print(f'skip {d}: not a dir', file=sys.stderr); continue
        r = analyze(d, a.tol, a.include_extra)
        with open(os.path.join(d, 'results.json'), 'w') as f:
            json.dump(r, f, indent=1)
        res.append(r)
    if len(res) > 1:
        print('| run | mode | vad | turns | tap p50/p95 | audible p50/p95 | call start p50/p95 | call done p50/p95 | usd | $/min |')
        print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in res:
            pp = lambda s: f'{fmt(s["p50"])}/{fmt(s["p95"])}'
            print(f'| {os.path.basename(r["run"].rstrip("/"))} | {r["mode"]} | {r["vad"] or "-"} | {r["n_turns_stats"]} | {pp(r["tap_lat_ms"])} | '
                  f'{pp(r["audible_lat_ms"])} | {pp(r["call_start_lat_ms"])} | {pp(r["call_done_lat_ms"])} | '
                  f'{fmt(r["usd_est"], 4) if r["usd_est"] is not None else "-"} | {fmt(r["usd_per_min"], 4) if r["usd_per_min"] is not None else "-"} |')
        print()
    for r in res:
        print(md_run(r))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
