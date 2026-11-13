"""Buzz analysis of a mono wav: clipping, click count, HF energy, periodic-buzz (modulation) lines.
usage: python -I anal.py file.wav [t0 t1]"""
import sys, wave, json
import numpy as np


def load(p):
    w = wave.open(p)
    sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def analyse(x, sr):
    out = {"dur_s": round(len(x) / sr, 2), "peak": round(float(np.abs(x).max()), 4)}
    out["clip_ge_0.99"] = int((np.abs(x) >= 0.99).sum())
    out["clip_ge_0.95"] = int((np.abs(x) >= 0.95).sum())
    # voiced frames: 20 ms rms above -35 dBFS
    F = int(sr * 0.02)
    nf = len(x) // F
    fr = x[: nf * F].reshape(nf, F)
    rms = np.sqrt((fr ** 2).mean(1))
    voiced = rms > 0.018
    out["voiced_s"] = round(voiced.sum() * 0.02, 2)
    out["rms_voiced_dbfs"] = round(float(20 * np.log10(np.sqrt((fr[voiced] ** 2).mean()) + 1e-12)), 1) if voiced.any() else None
    # clicks: second difference outliers relative to local level, in voiced frames
    d2 = np.abs(np.diff(x, 2))
    d2f = d2[: (len(d2) // F) * F].reshape(-1, F)
    loc = np.median(d2f, 1, keepdims=True) + 1e-6
    k = (d2f > 12 * loc) & (d2f > 0.05)
    out["clicks_voiced"] = int(k[voiced[: len(k)]].sum())
    # spectrum of voiced audio (Welch, 4096)
    N = 4096
    seg = x[: nf * F]
    vm = np.repeat(voiced, F)
    xs = seg[vm]
    if len(xs) > N * 4:
        hop = N // 2
        win = np.hanning(N)
        P = np.zeros(N // 2 + 1)
        m = 0
        for i in range(0, len(xs) - N, hop):
            P += np.abs(np.fft.rfft(xs[i:i + N] * win)) ** 2
            m += 1
        P /= m
        f = np.fft.rfftfreq(N, 1 / sr)
        tot = P[(f > 80)].sum()
        bands = {}
        for a, b in [(80, 1000), (1000, 4000), (4000, 8000), (8000, 11500), (11500, 12500), (12500, 16000), (16000, 24000)]:
            sel = (f >= a) & (f < b)
            if sel.any() and b <= sr / 2 + 1:
                bands[f"{a}-{b}"] = round(float(10 * np.log10(P[sel].sum() / tot + 1e-15)), 1)
        out["band_db_rel"] = bands
        # spectral flatness between harmonics in 1-4 kHz ~ noise floor (buzz = raised noise)
        sel = (f > 1000) & (f < 4000)
        out["flatness_1_4k"] = round(float(np.exp(np.mean(np.log(P[sel] + 1e-20))) / np.mean(P[sel])), 4)
    # periodic buzz: modulation spectrum of the 1 ms energy envelope of the high band (>3 kHz) during voiced audio
    # (a chunk-rate discontinuity puts lines at 1/chunk and harmonics)
    hp = np.diff(x)  # crude highpass
    E = int(sr * 0.001)
    ne = len(hp) // E
    env = np.sqrt((hp[: ne * E].reshape(ne, E) ** 2).mean(1))
    vme = np.zeros(ne, bool); vv = np.repeat(voiced, 20)[:ne]; vme[:len(vv)] = vv
    if vme.sum() > 4000:
        e = env.copy()
        e[~vme] = 0
        e = e - e[vme].mean() * vme
        M = 1 << int(np.log2(len(e)))
        S = np.abs(np.fft.rfft(e[:M] * np.hanning(M))) ** 2
        fm = np.fft.rfftfreq(M, 0.001)
        lines = []
        for cand in [25, 31.25, 41.67, 50, 62.5, 93.75, 100, 125, 187.5, 200, 250, 375]:
            sel = np.abs(fm - cand) < 0.6
            nb = (np.abs(fm - cand) > 2) & (np.abs(fm - cand) < 8)
            if sel.any() and nb.any():
                lines.append([cand, round(float(10 * np.log10(S[sel].max() / np.median(S[nb]))), 1)])
        out["mod_lines_db"] = lines
    return out


if __name__ == "__main__":
    x, sr = load(sys.argv[1])
    if len(sys.argv) > 3:
        x = x[int(float(sys.argv[2]) * sr): int(float(sys.argv[3]) * sr)]
    r = analyse(x, sr)
    r["file"] = sys.argv[1].split("/")[-1]
    r["sr"] = sr
    print(json.dumps(r))
