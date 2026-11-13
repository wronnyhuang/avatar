"""Compare a captured tap against a reference (both wav). Aligns (xcorr), resamples to 48 kHz, fits gain, reports:
SNR of residual (dB), THD+N-like ratio, residual clicks (|resid| jumps), clipped samples, gap count (runs of zeros in the
capture inside voiced reference), residual spectrum bands.  usage: python -I cmp.py ref.wav cap.wav [label]"""
import sys, wave, json
import numpy as np


def load(p):
    w = wave.open(p); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    return x, sr


def up(x, sr, to=48000):
    if sr == to:
        return x
    n = int(round(len(x) * to / sr))
    X = np.fft.rfft(x)
    Y = np.zeros(n // 2 + 1, complex); m = min(len(X), len(Y)); Y[:m] = X[:m]
    return np.fft.irfft(Y, n) * (n / len(x))


ref, rs = load(sys.argv[1]); cap, cs = load(sys.argv[2])
clip = int((np.abs(cap) >= 0.999).sum())
ref = up(ref, rs); cap = up(cap, cs); sr = 48000
# coarse align on envelopes
def env(x):
    E = 480; n = len(x) // E; return np.sqrt((x[:n * E].reshape(n, E) ** 2).mean(1))
er, ec = env(ref), env(cap)
L = 1 << int(np.ceil(np.log2(len(er) + len(ec))))
c = np.fft.irfft(np.fft.rfft(ec, L) * np.conj(np.fft.rfft(er, L)), L)
lag_c = int(np.argmax(c)); lag_c = lag_c if lag_c < L // 2 else lag_c - L
lag = lag_c * 480
# fine align +-960 samples
best = None
seg = slice(max(0, -lag) + 48000, max(0, -lag) + 48000 * 6)
r0 = ref[seg]
for d in range(lag - 960, lag + 961, 1):
    s = seg.start + d
    if s < 0 or s + len(r0) > len(cap):
        continue
    v = float(np.dot(r0, cap[s:s + len(r0)]))
    if best is None or v > best[0]:
        best = (v, d)
lag = best[1]
n = min(len(ref), len(cap) - lag) if lag >= 0 else min(len(ref) + lag, len(cap))
R = ref[max(0, -lag): max(0, -lag) + n]; C = cap[max(0, lag): max(0, lag) + n]
# blockwise gain fit (200 ms) so a slow gain/delay drift does not count as noise
B = 9600; res = np.zeros_like(C); vo = np.zeros(len(C), bool); prevd = None; BS = []; LAGS = []; STEPS = []
for i in range(0, n - B, B):
    r, cc = R[i:i + B], C[i:i + B]
    if np.sqrt((r ** 2).mean()) > 0.01:
        # allow +-2 samples re-alignment inside a block (delay changes)
        bb = None
        W = 1200  # +-25 ms search per block (the playout delay steps between replies)
        lo, hi = max(0, i - W), min(n, i + B + W)
        cw = C[lo:hi]
        xc = np.correlate(cw, r, 'valid')
        cands = {lo + int(np.argmax(xc))} | ({i + prevd} if prevd is not None else set())
        cands |= {lo + int(k) for k in np.argsort(xc)[-3:]}
        K = 128
        for j0 in cands:
            if j0 - K < 0 or j0 + B + K > n or i - K < 0 or i + B + K > n:
                continue
            c2 = C[j0:j0 + B]
            A = np.stack([R[i + s:i + s + B] for s in range(-K, K + 1)], 1)
            h, *_ = np.linalg.lstsq(A, c2, rcond=None)
            e = c2 - A @ h
            if bb is None or (e ** 2).sum() < bb[0]: bb = ((e ** 2).sum(), e, j0 - i)
        if bb is None:
            continue
        if prevd is not None and abs(bb[2] - prevd) > 24 and LAGS and LAGS[-1][0] == i - B: STEPS.append((round(i / 48000, 2), round((bb[2] - prevd) / 48, 2)))
        LAGS.append((i, bb[2])); prevd = bb[2]
        BS.append(float(10 * np.log10((r ** 2).sum() / (bb[0] + 1e-15))))
        res[i:i + B] = bb[1]; vo[i:i + B] = True
Pr = (R[vo] ** 2).mean(); Pe = (res[vo] ** 2).mean()
out = {"label": sys.argv[3] if len(sys.argv) > 3 else sys.argv[2].split('/')[-1], "lag_ms": round(lag / 48, 2),
       "voiced_s": round(vo.sum() / sr, 2), "snr_db": round(float(10 * np.log10(Pr / (Pe + 1e-15))), 1), "clipped": clip,
       "blk_snr_p10_p50": [round(float(np.percentile(BS, 10)), 1), round(float(np.percentile(BS, 50)), 1)], "blk_lt30": int((np.array(BS) < 30).sum()), "blocks": len(BS), "lag_steps_in_speech": len(STEPS), "steps": STEPS[:12]}
# clicks: residual sample jumps far above the residual's local level
d = np.abs(np.diff(res)); thr = max(0.02, 8 * np.median(d[vo[1:]]) if vo.any() else 0.02)
out["resid_clicks"] = int(((d > thr) & vo[1:]).sum())
# discontinuities in the capture itself: |x[n]-x[n-1]| far above what the reference does at that point
dc = np.abs(np.diff(C)); dr = np.abs(np.diff(R))
out["cap_jumps"] = int(((dc > 0.05) & (dc > 4 * dr + 0.02)).sum())
# gaps: >= 2 ms of exact silence in the capture where the reference is voiced
z = (np.abs(C) < 1e-4) & (np.abs(R) > 0.01)
runs = 0; k = 0
for v in z:
    k = k + 1 if v else 0
    if k == 96: runs += 1
out["gaps_2ms"] = runs
# residual spectrum
N = 2048; win = np.hanning(N); P = np.zeros(N // 2 + 1); Q = np.zeros(N // 2 + 1); m = 0
idx = np.where(vo)[0]
for i in range(idx[0] if len(idx) else 0, n - N, N):
    if vo[i] and vo[i + N - 1]:
        P += np.abs(np.fft.rfft(res[i:i + N] * win)) ** 2; Q += np.abs(np.fft.rfft(R[i:i + N] * win)) ** 2; m += 1
f = np.fft.rfftfreq(N, 1 / sr)
bands = {}
for a, b in [(50, 500), (500, 2000), (2000, 6000), (6000, 11500), (11500, 24000)]:
    s = (f >= a) & (f < b); bands[f"{a}-{b}"] = round(float(10 * np.log10((Q[s].sum() + 1e-15) / (P[s].sum() + 1e-15))), 1)
out["band_snr_db"] = bands
print(json.dumps(out))
