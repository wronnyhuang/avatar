#!/usr/bin/env python3
"""npz_to_frames.py — real-conversation face frames + OFFLINE nod / head-shake ground truth.

  nice -n 10 python3 -I tests/npz_to_frames.py [--n 40 | --all] [--workers 8] [--raw]

Input : $AVATAR_DATA/prior/feats/<clip>.npz (MediaPipe Python: present (S,N), blend (S,N,52) incl. '_neutral',
        mat (S,N,16) ROW-major 4x4 facial transformation matrix, cut (N,) frame-difference, fps 30).
Clips : the first N clip names in Python sorted() order that have >= 1 stream with face presence >= 50 %
        (default 40; --all = every clip). Every stream with presence >= 50 % is written.
Output: <out>/frames/<clip>_s<k>.jsonl, one line per video frame (30 fps):
          {"t": s, "face": {"present": true, "blend": {51 names}, "yaw", "pitch", "roll", "matrix": [16 col-major],
                            "raw": [yaw, pitch, roll]}}   |   {"t": s, "face": {"present": false}[, "gap": "cut"|"jump"]}
        - yaw/pitch/roll: EXACTLY app/perception.js — the Python row-major matrix is transposed to the JS column-major
          array m, raw Euler pitch=atan2(m6/sy, m10/sz), yaw=asin(-m2/sx), roll=atan2(m1/sx, m0/sx), then SIGN (parsed
          from perception.js): yaw>0 = nose toward screen-right (mirrored view), pitch>0 = nose down, roll>0 = crown to
          screen-right. Then, like perception.js, One-Euro filtered (FILT.ang) with the filters reset while the face is
          absent; blend is One-Euro filtered (FILT.blend) and clamped 0..1 the same way. `raw` = the unfiltered angles.
          --raw writes unfiltered yaw/pitch/roll/blend instead.
        - shot cuts (motionlib.cut_mask rule) and tracking jumps (> 12 deg in one frame) become a 0.2 s face-absent
          gap ("gap" field) — a phone camera never cuts, the closest real equivalent is a re-acquired face.
        <out>/labels/<clip>_s<k>.json — offline, NON-causal labels from the RAW (unfiltered) angles, see below;
        <out>/index.json — the stream list (+ sign sanity checks) the evaluator reads.
LABELS (per continuous face run >= 1 s; parameters in LAB, written into every label file):
  bp = zero-phase Butterworth band-pass 0.6–4 Hz (turning points, timing, dominance); lp = zero-phase 4 Hz low-pass of
  the raw angle (true amplitudes). Turning points = zigzag on bp with hysteresis (pitch 0.02, yaw 0.025 rad).
  NOD swing: pitch turning points a→b→c (down-then-up or up-then-down), a..c within 0.25–1.2 s, lp half-swings
    h1=|b-a|, h2=|c-b|: max >= 0.07 rad (4°) and min >= 0.5·max (it comes back), bp yaw and roll ptp over a..c
    < 0.6 × bp pitch ptp, amplitude <= 0.35 rad (20°). Swings within 0.6 s merge into one nod {onset, peak, end,
    n_swings, amp, dir}. onset = first frame the lp pitch has moved 20 % of the way a→b.
  SHAKE: >= 2 yaw reversals = 3 half-swings E0..E3 within 0.4–1.6 s, each half <= 0.8 s, lp yaw ptp >= 0.1 rad (6°),
    every lp half-swing >= 0.4 × that ptp, bp pitch ptp < 0.7 × bp yaw ptp, ptp <= 0.6 rad; extended while further
    half-swings qualify; shakes within 0.6 s merge. {onset, end, reversals, amp}.
  IGNORE intervals (per type): swings that pass the timing/amplitude/return rules but are only weakly dominant (yaw/roll
    or pitch still < the main axis, i.e. the plain "smaller than" rule) or too big (look down/up, look around) —
    the evaluator counts a detection there as neither a hit nor a false alarm.
  `frontal` = |lp yaw| <= 0.6 rad through the label (dyad crops often show 45° profiles).
"""
import argparse, glob, json, os, re, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.join(HERE, '..', 'app', 'perception.js')
DATA = os.environ.get('AVATAR_DATA', os.path.join(HERE, '..', 'data'))
FEATS = os.path.join(DATA, 'prior', 'feats')
MANIFESTS = [os.path.join(DATA, 'prior', 'manifest.jsonl'),
             os.path.join(DATA, 'prior', 'manifest_dyad.jsonl')]
OUT = os.path.join(DATA, 'nodshake')
GAP_S = 0.2
JUMP_RAD = 12 * np.pi / 180

# ── ground-truth parameters (offline, independent of the tracker under test) ──
LAB = dict(
    band=[0.6, 4.0],          # zero-phase Butterworth band-pass (order 2) → timing / turning points
    lp=4.0,                   # zero-phase 4 Hz low-pass of the raw angle (≈ 0.1 s smoothing) → true amplitudes
    min_seg_s=1.0,            # label only inside continuous face runs this long
    edge_s=0.1,               # swings must lie this far inside a run (filter transients)
    # a swing that passes everything except `dom` (but yaw/roll still < pitch) or exceeds amp_max (a look down/up
    # or a look-around, not a gesture) is an IGNORE interval: a detection there is neither a hit nor a false alarm.
    nod=dict(hyst=0.02, pp=0.07, dur=[0.25, 1.2], ret=0.5, merge_s=0.6, dom=0.6, amp_max=0.35),
    shake=dict(hyst=0.025, pp=0.10, dur=[0.4, 1.6], half_max_s=0.8, half_min=0.4, merge_s=0.6, dom=0.7, amp_max=0.6),
    frontal=0.6,              # a label is 'frontal' when |yaw| (low-passed raw) stays <= this through it
)


def parse_perception(path):
    src = open(path, encoding='utf8').read()
    m = re.search(r"const BS_NAMES = \[(.*?)\];", src, re.S)
    names = re.findall(r"'([^']+)'", m.group(1))
    s = re.search(r"const SIGN = \{([^}]*)\}", src).group(1)
    sign = {k: float(re.search(k + r":\s*(-?\d+(?:\.\d+)?)", s).group(1)) for k in ('yaw', 'pitch', 'roll')}
    f = re.search(r"const FILT = \{([^}]*)\}", src).group(1)
    filt = {k: [float(x) for x in re.search(k + r":\s*\[([^\]]*)\]", f).group(1).split(',')] for k in ('ang', 'blend')}
    assert len(names) == 52 and names[0] == '_neutral', names[:3]
    return names, sign, filt, os.path.getmtime(path)


class OneEuroVec:
    """perception.js OneEuro, vectorised over channels (dCutoff 1.0, dt capped 0.5 s, identical update order)."""
    def __init__(self, mc, beta, n, dc=1.0):
        self.mc, self.beta, self.dc, self.n = mc, beta, dc, n
        self.reset()

    def reset(self):
        self.x = None; self.d = np.zeros(self.n); self.t = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, v, t):
        if self.x is None or self.t is None or not (t > self.t):
            if self.x is None:
                self.x = np.array(v, float); self.d = np.zeros(self.n)
            self.t = t
            return self.x.copy()
        dt = min(0.5, t - self.t)
        dv = (v - self.x) / dt
        self.d = self.d + self.alpha(self.dc, dt) * (dv - self.d)
        cutoff = self.mc + self.beta * np.abs(self.d)
        self.x = self.x + self.alpha(cutoff, dt) * (v - self.x)
        self.t = t
        return self.x.copy()


def euler_from_rowmajor(mat, sign):
    """(N,16) Python row-major → JS column-major m (= transpose) → perception.js raw Euler → SIGN (viewer convention)."""
    M = mat.reshape(-1, 4, 4)
    m = np.transpose(M, (0, 2, 1)).reshape(-1, 16)          # m[c*4+r] = M[r][c]
    sx = np.sqrt(m[:, 0] ** 2 + m[:, 1] ** 2 + m[:, 2] ** 2)
    sy = np.sqrt(m[:, 4] ** 2 + m[:, 5] ** 2 + m[:, 6] ** 2)
    sz = np.sqrt(m[:, 8] ** 2 + m[:, 9] ** 2 + m[:, 10] ** 2)
    sx = np.where(sx > 0, sx, 1); sy = np.where(sy > 0, sy, 1); sz = np.where(sz > 0, sz, 1)
    pitch = np.arctan2(m[:, 6] / sy, m[:, 10] / sz)
    yaw = np.arcsin(np.clip(-m[:, 2] / sx, -1, 1))
    roll = np.arctan2(m[:, 1] / sx, m[:, 0] / sx)
    return sign['yaw'] * yaw, sign['pitch'] * pitch, sign['roll'] * roll, m


def cut_mask(cut):
    """motionlib.cut_mask: frame indices that start a new shot"""
    thr = max(18.0, np.percentile(cut, 99.5) if len(cut) else 18.0)
    return cut > min(thr, 30.0)


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def zigzag(x, h):
    """indices of alternating turning points (hysteresis h); first = start extreme, last = final running extreme"""
    n = len(x)
    if n < 3:
        return []
    pts, imax, imin, dirn, ie = [], 0, 0, 0, 0
    for i in range(1, n):
        v = x[i]
        if dirn == 0:
            if v > x[imax]: imax = i
            if v < x[imin]: imin = i
            if x[imax] - x[imin] >= h:
                if imax > imin: pts.append(imin); dirn = 1; ie = imax
                else: pts.append(imax); dirn = -1; ie = imin
        elif dirn == 1:
            if v > x[ie]: ie = i
            elif x[ie] - v >= h: pts.append(ie); dirn = -1; ie = i
        else:
            if v < x[ie]: ie = i
            elif v - x[ie] >= h: pts.append(ie); dirn = 1; ie = i
    if dirn != 0:
        pts.append(ie)
    return pts


def refine(lp, e, hi, w=4):
    """the low-passed raw angle's own extreme within ±w frames of a band-pass turning point"""
    a, b = max(0, e - w), min(len(lp), e + w + 1)
    seg = lp[a:b]
    return a + int(np.argmax(seg) if hi else np.argmin(seg))


def ptp(x):
    return float(np.max(x) - np.min(x)) if len(x) else 0.0


def onset_idx(lp, a, b, frac=0.2):
    """first frame after turning point a where the raw angle has moved frac of the way to b"""
    h = lp[b] - lp[a]
    for k in range(a, b + 1):
        if abs(lp[k] - lp[a]) >= frac * abs(h):
            return k
    return a


def label_segment(yaw, pitch, roll, fps, t0):
    sos_bp = butter(2, LAB['band'], btype='bandpass', fs=fps, output='sos')
    sos_lp = butter(2, LAB['lp'], btype='lowpass', fs=fps, output='sos')
    bp = {k: sosfiltfilt(sos_bp, v) for k, v in (('y', yaw), ('p', pitch), ('r', roll))}
    lp = {k: sosfiltfilt(sos_lp, v) for k, v in (('y', yaw), ('p', pitch), ('r', roll))}
    n, edge = len(yaw), int(round(LAB['edge_s'] * fps))
    T = lambda i: round(t0 + i / fps, 4)

    # ── nods: pitch swing a→b→c (down-then-up or up-then-down) ──
    P = LAB['nod']; E = zigzag(bp['p'], P['hyst']); swings = []; ign_n = []; i = 0
    while i + 2 < len(E):
        a, b, c = E[i], E[i + 1], E[i + 2]
        hi_b = bp['p'][b] > bp['p'][a]
        ra, rb, rc = refine(lp['p'], a, not hi_b), refine(lp['p'], b, hi_b), refine(lp['p'], c, not hi_b)
        h1, h2 = abs(lp['p'][rb] - lp['p'][ra]), abs(lp['p'][rc] - lp['p'][rb])
        pp = max(h1, h2); dur = (c - a) / fps
        bpp, ypp, rpp = ptp(bp['p'][a:c + 1]), ptp(bp['y'][a:c + 1]), ptp(bp['r'][a:c + 1])
        core = (P['dur'][0] <= dur <= P['dur'][1] and pp >= P['pp'] and min(h1, h2) >= P['ret'] * pp
                and a >= edge and c <= n - 1 - edge)
        if core and ypp < P['dom'] * bpp and rpp < P['dom'] * bpp and pp <= P['amp_max']:
            swings.append(dict(a=a, b=b, c=c, on=onset_idx(lp['p'], ra, rb), pp=pp, dir='down' if hi_b else 'up',
                               ypp=ypp, rpp=rpp, bpp=bpp))
            i += 2
        elif core and ((ypp < bpp and rpp < bpp) or pp > P['amp_max']):
            ign_n.append(dict(onset=T(onset_idx(lp['p'], ra, rb)), end=T(c), amp=round(pp, 4),
                              why='amp' if pp > P['amp_max'] else 'dom'))
            i += 2
        else:
            i += 1
    nods = []
    for s in swings:
        if nods and (s['a'] - nods[-1]['_c']) / fps <= P['merge_s']:
            g = nods[-1]; g['n_swings'] += 1; g['_c'] = s['c']; g['amp'] = max(g['amp'], round(s['pp'], 4)); g['end'] = T(s['c'])
        else:
            nods.append(dict(onset=T(s['on']), peak=T(s['b']), end=T(s['c']), n_swings=1, amp=round(s['pp'], 4),
                             dir=s['dir'], yaw_pp=round(s['ypp'], 4), roll_pp=round(s['rpp'], 4), _c=s['c']))
    F = lambda t_a, t_b: bool(np.max(np.abs(lp['y'][max(0, int(round((t_a - t0) * fps))):int(round((t_b - t0) * fps)) + 1])) <= LAB['frontal'])
    for g in nods:
        g['amp_deg'] = round(g['amp'] * 180 / np.pi, 2); g['dur'] = round(g['end'] - g['onset'], 3); g.pop('_c')
        g['frontal'] = F(g['onset'], g['end'])

    # ── shakes: >= 2 yaw reversals (3 half-swings) within 0.4–1.6 s, pp >= 0.1, yaw-dominant ──
    Q = LAB['shake']; E = zigzag(bp['y'], Q['hyst']); shakes = []; ign_s = []; i = 0
    while i + 3 < len(E):
        idx = E[i:i + 4]
        his = [bp['y'][idx[k]] > bp['y'][idx[k - 1]] if k else bp['y'][idx[0]] > bp['y'][idx[1]] for k in range(4)]
        r = [refine(lp['y'], e, h) for e, h in zip(idx, his)]
        halves = [abs(lp['y'][r[k + 1]] - lp['y'][r[k]]) for k in range(3)]
        hd = [(idx[k + 1] - idx[k]) / fps for k in range(3)]
        a, d = idx[0], idx[3]; dur = (d - a) / fps
        pp = ptp(lp['y'][a:d + 1]); yb, pb = ptp(bp['y'][a:d + 1]), ptp(bp['p'][a:d + 1])
        core = (Q['dur'][0] <= dur <= Q['dur'][1] and pp >= Q['pp'] and min(halves) >= Q['half_min'] * pp
                and max(hd) <= Q['half_max_s'] and a >= edge and d <= n - 1 - edge)
        if core and not (pb < Q['dom'] * yb and pp <= Q['amp_max']):
            if pb < yb or pp > Q['amp_max']:
                ign_s.append(dict(onset=T(onset_idx(lp['y'], r[0], r[1])), end=T(d), amp=round(pp, 4),
                                  why='amp' if pp > Q['amp_max'] else 'dom'))
                i += 3
            else:
                i += 1
            continue
        if core:
            j = i + 3
            while j + 1 < len(E):
                e0, e1 = E[j], E[j + 1]
                h = abs(lp['y'][e1] - lp['y'][e0])
                if (h >= Q['half_min'] * pp and (e1 - e0) / fps <= Q['half_max_s'] and e1 <= n - 1 - edge
                        and ptp(bp['p'][e0:e1 + 1]) < Q['dom'] * ptp(bp['y'][e0:e1 + 1])):
                    j += 1
                else:
                    break
            e_end = E[j]; amp = ptp(lp['y'][a:e_end + 1])
            ev = dict(onset=T(onset_idx(lp['y'], r[0], r[1])), end=T(e_end), reversals=j - i - 1, amp=round(amp, 4),
                      pitch_pp=round(pb, 4), _a=a, _e=e_end)
            if shakes and (a - shakes[-1]['_e']) / fps <= Q['merge_s']:
                g = shakes[-1]; g['reversals'] += ev['reversals'] + 1; g['end'] = ev['end']; g['amp'] = max(g['amp'], ev['amp']); g['_e'] = e_end
            else:
                shakes.append(ev)
            i = j
        else:
            i += 1
    for g in shakes:
        g['amp_deg'] = round(g['amp'] * 180 / np.pi, 2); g['dur'] = round(g['end'] - g['onset'], 3); g.pop('_a'); g.pop('_e')
        g['frontal'] = F(g['onset'], g['end'])
    return nods, shakes, ign_n, ign_s


def sign_check(lmk, yaw, pitch, roll, ok):
    """correlate the matrix angles with landmark proxies in VIEWER coords (detrended by a 1 s running mean)"""
    if ok.sum() < 90:
        return None
    L = lmk[ok]
    nose, eA, eB = L[:, 0], (L[:, 1] + L[:, 2]) / 2, (L[:, 3] + L[:, 4]) / 2   # A = 33/133 = raw image-left = his right eye
    mid = (eA + eB) / 2
    iod = np.maximum(1e-4, np.hypot(eA[:, 0] - eB[:, 0], eA[:, 1] - eB[:, 1]))
    yawP = (mid[:, 0] - nose[:, 0]) / iod            # viewer x = 1 - raw x → + = nose toward screen-right
    pitchP = (nose[:, 1] - mid[:, 1]) / iod          # + = nose tip lower than the eye line
    rollP = np.arctan2(eA[:, 1] - eB[:, 1], (1 - eA[:, 0]) - (1 - eB[:, 0]))   # eA is on screen-RIGHT; + = it is lower
    k = np.ones(31) / 31
    hp = lambda v: v - np.convolve(v, k, mode='same')
    c = lambda a, b: float(np.corrcoef(hp(a), hp(b))[0, 1])
    return dict(yaw=round(c(yaw[ok], yawP), 3), pitch=round(c(pitch[ok], pitchP), 3), roll=round(c(roll[ok], rollP), 3))


def process(job):
    path, ks, cat, out, names, sign, filt, raw_out = job
    clip = os.path.basename(path)[:-4]
    d = np.load(path, allow_pickle=False)
    fps = int(d['fps']); streams = [str(s) for s in d['streams']]
    cuts = cut_mask(d['cut'].astype(float))
    gapn = int(round(GAP_S * fps))
    res = []
    for k in ks:
        pres = d['present'][k].astype(bool); N = len(pres)
        yaw, pitch, roll, mcol = euler_from_rowmajor(d['mat'][k].astype(np.float64), sign)
        blend = d['blend'][k].astype(np.float64)
        # gaps: shot cuts and tracking jumps
        gap = np.zeros(N, dtype='U4')
        jump = np.zeros(N, bool)
        both = pres[1:] & pres[:-1]
        dang = np.max(np.abs(np.diff(np.stack([yaw, pitch, roll], 1), axis=0)), 1)
        jump[1:] = both & (dang > JUMP_RAD)
        for i in np.where(cuts[:N] & pres)[0]:
            gap[i:i + gapn] = np.where(gap[i:i + gapn] == '', 'cut', gap[i:i + gapn])
        for i in np.where(jump)[0]:
            gap[i:i + gapn] = np.where(gap[i:i + gapn] == '', 'jump', gap[i:i + gapn])
        eff = pres & (gap == '')
        # perception-style One-Euro filtering (reset while absent)
        fa = OneEuroVec(filt['ang'][0], filt['ang'][1], 3)
        fb = OneEuroVec(filt['blend'][0], filt['blend'][1], 51)
        fname = f'{clip}_s{k}'
        fpath = os.path.join(out, 'frames', fname + '.jsonl')
        bnames = names[1:]
        with open(fpath, 'w') as fh:
            for i in range(N):
                t = round(i / fps, 4)
                if not eff[i]:
                    fa.reset(); fb.reset()
                    face = {'present': False}
                    rec = {'t': t, 'face': face}
                    if gap[i]: rec['gap'] = str(gap[i])
                    fh.write(json.dumps(rec, separators=(',', ':')) + '\n'); continue
                ang = np.array([yaw[i], pitch[i], roll[i]])
                bs = blend[i, 1:]
                if not raw_out:
                    ang = fa.filter(ang, t)
                    bs = np.clip(fb.filter(bs, t), 0, 1)
                face = {'present': True, 'blend': {n: round(float(v), 4) for n, v in zip(bnames, bs)},
                        'yaw': round(float(ang[0]), 4), 'pitch': round(float(ang[1]), 4), 'roll': round(float(ang[2]), 4),
                        'matrix': [round(float(v), 4) for v in mcol[i]],
                        'raw': [round(float(yaw[i]), 4), round(float(pitch[i]), 4), round(float(roll[i]), 4)]}
                fh.write(json.dumps({'t': t, 'face': face}, separators=(',', ':')) + '\n')
        # labels on continuous valid runs (RAW angles)
        nods, shakes, ign_n, ign_s, segs = [], [], [], [], []
        for a, b in runs(eff):
            if (b - a) / fps < LAB['min_seg_s']:
                continue
            segs.append([round(a / fps, 3), round(b / fps, 3)])
            n_, s_, in_, is_ = label_segment(yaw[a:b], pitch[a:b], roll[a:b], fps, a / fps)
            nods += n_; shakes += s_; ign_n += in_; ign_s += is_
        valid_min = sum(b - a for a, b in segs) / 60.0
        sc = sign_check(d['lmk'][k], yaw, pitch, roll, eff)
        okv = eff
        stats = dict(pitch_p02_p98=[round(float(np.percentile(pitch[okv], 2)), 3), round(float(np.percentile(pitch[okv], 98)), 3)],
                     yaw_p02_p98=[round(float(np.percentile(yaw[okv], 2)), 3), round(float(np.percentile(yaw[okv], 98)), 3)],
                     roll_p02_p98=[round(float(np.percentile(roll[okv], 2)), 3), round(float(np.percentile(roll[okv], 98)), 3)],
                     frac_pitch_in_0p4=round(float(np.mean(np.abs(pitch[okv]) <= 0.4)), 3),
                     frac_yaw_in_0p6=round(float(np.mean(np.abs(yaw[okv]) <= 0.6)), 3))
        lab = dict(clip=clip, stream=streams[k], k=k, category=cat, fps=fps, n_frames=N, presence=round(float(pres.mean()), 3),
                   present_min=round(float(eff.sum()) / fps / 60, 3), valid_min=round(valid_min, 3), segments=segs,
                   gaps=dict(cut=int(np.sum(cuts[:N] & pres)), jump=int(jump.sum())), angle_stats=stats, sign_check=sc,
                   nods=nods, shakes=shakes, ignore=dict(nod=ign_n, head_shake=ign_s), params=LAB)
        with open(os.path.join(out, 'labels', fname + '.json'), 'w') as fh:
            json.dump(lab, fh, separators=(',', ':'))
        res.append(dict(clip=clip, stream=streams[k], k=k, category=cat, presence=lab['presence'],
                        present_min=lab['present_min'], valid_min=lab['valid_min'],
                        frames='frames/' + fname + '.jsonl', labels='labels/' + fname + '.json',
                        nods=len(nods), shakes=len(shakes), ignore_nod=len(ign_n), ignore_shake=len(ign_s),
                        sign_check=sc, angle_stats=stats))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--feats', default=FEATS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--raw', action='store_true', help='write unfiltered yaw/pitch/roll/blend (default: perception One-Euro)')
    A = ap.parse_args()
    t_start = time.time()
    names, sign, filt, pmt = parse_perception(PERC)
    cat = {}
    for mf in MANIFESTS:
        if os.path.exists(mf):
            for line in open(mf):
                j = json.loads(line); cat[os.path.basename(j['file'])[:-4]] = j.get('category')
    os.makedirs(os.path.join(A.out, 'frames'), exist_ok=True); os.makedirs(os.path.join(A.out, 'labels'), exist_ok=True)
    jobs = []
    for p in sorted(glob.glob(os.path.join(A.feats, '*.npz'))):
        if not A.all and len(jobs) >= A.n:
            break
        with np.load(p, allow_pickle=False) as d:
            pr = d['present'].mean(1)
        ks = [k for k in range(min(2, len(pr))) if pr[k] >= 0.5]
        if ks:
            clip = os.path.basename(p)[:-4]
            jobs.append((p, ks, cat.get(clip), A.out, names, sign, filt, A.raw))
    if A.workers > 1:
        from multiprocessing import Pool
        with Pool(A.workers) as pool:
            parts = pool.map(process, jobs, chunksize=1)
    else:
        parts = [process(j) for j in jobs]
    streams = [s for p in parts for s in p]
    tot_min = sum(s['valid_min'] for s in streams)
    by = {}
    for s in streams:
        c = by.setdefault(s['category'] or '?', dict(streams=0, valid_min=0.0, nods=0, shakes=0))
        c['streams'] += 1; c['valid_min'] += s['valid_min']; c['nods'] += s['nods']; c['shakes'] += s['shakes']
    for c in by.values():
        c['valid_min'] = round(c['valid_min'], 2)
        c['nods_per_min'] = round(c['nods'] / max(1e-9, c['valid_min']), 2); c['shakes_per_min'] = round(c['shakes'] / max(1e-9, c['valid_min']), 2)
    scs = [s['sign_check'] for s in streams if s['sign_check']]
    sc_med = {k: round(float(np.median([x[k] for x in scs])), 3) for k in ('yaw', 'pitch', 'roll')} if scs else None
    sc_pos = {k: f"{sum(x[k] > 0 for x in scs)}/{len(scs)}" for k in ('yaw', 'pitch', 'roll')} if scs else None
    allp = [s['angle_stats']['frac_pitch_in_0p4'] for s in streams]; ally = [s['angle_stats']['frac_yaw_in_0p6'] for s in streams]
    totals = dict(clips=len(jobs), streams=len(streams), valid_min=round(tot_min, 2),
                  nods=sum(s['nods'] for s in streams), shakes=sum(s['shakes'] for s in streams),
                  ignore_nod=sum(s['ignore_nod'] for s in streams), ignore_shake=sum(s['ignore_shake'] for s in streams))
    totals['nods_per_min'] = round(totals['nods'] / max(1e-9, tot_min), 2); totals['shakes_per_min'] = round(totals['shakes'] / max(1e-9, tot_min), 2)
    index = dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), perception_js=os.path.realpath(PERC),
                 perception_mtime=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(pmt)), sign=sign, filt=None if A.raw else filt,
                 fps=30, gap_s=GAP_S, label_params=LAB, totals=totals, by_category=by,
                 sign_check=dict(median_corr=sc_med, positive=sc_pos),
                 angle_range=dict(median_frac_pitch_in_0p4=round(float(np.median(allp)), 3), median_frac_yaw_in_0p6=round(float(np.median(ally)), 3)),
                 streams=streams)
    with open(os.path.join(A.out, 'index.json'), 'w') as fh:
        json.dump(index, fh, indent=1)
    print(f"{len(jobs)} clips, {len(streams)} streams, {tot_min:.1f} valid face-min, {time.time() - t_start:.1f} s → {A.out}")
    print(f"labels: nods {totals['nods']} ({totals['nods_per_min']}/min)   shakes {totals['shakes']} ({totals['shakes_per_min']}/min)"
          f"   ignore intervals: nod {totals['ignore_nod']}, shake {totals['ignore_shake']}")
    for c, v in sorted(by.items()):
        print(f"  {c:7s} streams {v['streams']:3d}  {v['valid_min']:7.1f} min  nods {v['nods']:4d} ({v['nods_per_min']:5.2f}/min)  shakes {v['shakes']:3d} ({v['shakes_per_min']:4.2f}/min)")
    print(f"sign check (corr with landmark proxies, median / #positive): {sc_med} {sc_pos}")
    print(f"angle range: median frac |pitch|<=0.4 {index['angle_range']['median_frac_pitch_in_0p4']}, |yaw|<=0.6 {index['angle_range']['median_frac_yaw_in_0p6']}")


if __name__ == '__main__':
    main()
