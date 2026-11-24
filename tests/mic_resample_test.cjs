// The Gemini mic's context-rate -> 16 kHz resample (app/voice.js makeResampler + the Pcm16Tap worklet), no browser:
//  (a) frequency response: passband ripple, stopband / alias rejection (tones at 9, 10, 11, 11.9 kHz), OLD vs NEW
//  (b) streaming: random block splits give bit-identical output to one shot; the worklet (128-sample quanta) = the core
//  (c) long runs at 24 / 44.1 / 48 kHz: output count = round(n * 16000 / rate) +-1 at every checkpoint, no phase drift
//  (d) band SNR on speech vs an independent high-quality reference (direct long windowed-sinc), OLD vs NEW
//      speech WAVs: MIC_WAVS=<dir or comma list of .wav> (any rate, mono/stereo int16); (d) is skipped without them
//   node tests/mic_resample_test.cjs            MIC_JSON=<file> writes the measured numbers as JSON
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : '')); }
const OUT = {};

// ---- the code under test: voice.js as the page loads it (fake window), and the exact worklet module text
const W = {};
new Function('window', 'performance', 'setTimeout', 'clearTimeout', fs.readFileSync(path.join(ROOT, 'app/voice.js'), 'utf8'))(W, {now: () => 0}, () => 0, () => {});
const AV = W.AvatarVoice, makeResampler = AV.makeResampler;
function loadTap(rate, opts) {         // the REAL Pcm16Tap from the worklet source, run offline at `rate`
  const posts = []; let Tap = null;
  class AWP { constructor() { this.port = {postMessage: m => posts.push(m instanceof Int16Array ? m.slice(0) : m), onmessage: null}; } }
  const reg = (name, cls) => { if (name === 'avatar-pcm16-tap') Tap = cls; };
  Object.defineProperty(globalThis, 'currentTime', {get: () => 0, configurable: true});
  new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', 'currentFrame', AV.workletSource())(AWP, reg, rate, 0);
  const node = new Tap({processorOptions: opts || {}});
  return {node, posts, feed(x, q = 128) { for (let i = 0; i < x.length; i += q) { const b = x.subarray(i, Math.min(x.length, i + q)); node.process([[b]], [[new Float32Array(b.length)]]); } },
    pcm() { const c = posts.filter(p => p instanceof Int16Array); const o = new Int16Array(c.length * 320); c.forEach((p, i) => o.set(p, i * 320)); return o; },
    cfg: () => posts.find(p => p && p.t === 'cfg')};
}
function oldPick(x, rate) {            // the previous Pcm16Tap behaviour, line for line (float out, before the int16 store)
  const r = rate / 16000, o = new Float64Array(Math.ceil(x.length / r) + 1); let acc = 0, k = 0;
  for (let i = 0; i < x.length; i++) { acc += 1; if (acc >= r) { acc -= r; o[k++] = Math.max(-1, Math.min(1, x[i])); } }
  return o.subarray(0, k);
}
function newRs(x, rate) { const R = makeResampler(rate, 16000), y = new Float32Array(R.outMax(x.length)); const n = R.run(x, x.length, y); return {y: y.subarray(0, n), R}; }
const tone = (f, rate, n, a = 0.5, ph = 0) => { const y = new Float32Array(n), w = 2 * Math.PI * f / rate; for (let i = 0; i < n; i++) y[i] = a * Math.sin(w * i + ph); return y; };
function fitAmp(y, f, fs, from) {      // least-squares amplitude of a sinusoid at f (Hz) in y[from:]
  let ss = 0, cc = 0, sc = 0, ys = 0, yc = 0;
  for (let i = from; i < y.length; i++) { const s = Math.sin(2 * Math.PI * f * i / fs), c = Math.cos(2 * Math.PI * f * i / fs); ss += s * s; cc += c * c; sc += s * c; ys += y[i] * s; yc += y[i] * c; }
  const det = ss * cc - sc * sc, A = (ys * cc - yc * sc) / det, B = (yc * ss - ys * sc) / det;
  return {amp: Math.hypot(A, B), phase: Math.atan2(B, A)};
}
const rmsOf = (y, from) => { let e = 0; for (let i = from; i < y.length; i++) e += y[i] * y[i]; return Math.sqrt(e / (y.length - from)); };
const dB = v => 20 * Math.log10(Math.max(v, 1e-12));
const r2 = v => Math.round(v * 100) / 100;

// ================================================================== (a) frequency response
{ const res = {};
  for (const rate of [24000, 44100, 48000]) {
    const n = rate >> 2, warm = 200;
    let pmin = 1e9, pmax = -1e9;
    for (let f = 100; f <= 6800; f += 100) { const g = dB(fitAmp(newRs(tone(f, rate, n), rate).y, f, 16000, warm).amp / 0.5); pmin = Math.min(pmin, g); pmax = Math.max(pmax, g); }
    let worst = -1e9, worstF = 0;
    for (let f = 8600; f < rate / 2; f += 100) { const g = dB(rmsOf(newRs(tone(f, rate, n), rate).y, warm) / (0.5 / Math.SQRT2)); if (g > worst) { worst = g; worstF = f; } }
    const at = {}; for (const f of [7000, 7500, 7900]) at[f] = r2(dB(fitAmp(newRs(tone(f, rate, n), rate).y, f, 16000, warm).amp / 0.5));
    res[rate] = {pass_min_db: Math.round(pmin * 1000) / 1000, pass_max_db: Math.round(pmax * 1000) / 1000, stop_worst_db: r2(worst), stop_worst_hz: worstF, edge_db: at, taps: makeResampler(rate, 16000).taps, delay_ms: r2(makeResampler(rate, 16000).delaySec * 1000)};
    check(`(a) ${rate} Hz: passband 0.1-6.8 kHz flat within +-0.5 dB`, pmin >= -0.5 && pmax <= 0.5, res[rate]);
    check(`(a) ${rate} Hz: stopband >= 60 dB down from 8.6 kHz to Nyquist`, worst <= -60, {worst_db: r2(worst), at_hz: worstF});
    check(`(a) ${rate} Hz: group delay <= 2 ms`, makeResampler(rate, 16000).delaySec <= 0.002, res[rate].delay_ms);
  }
  const alias = {};
  for (const f of [9000, 10000, 11000, 11900]) {
    const x = tone(f, 24000, 24000), o = dB(rmsOf(oldPick(x, 24000), 200) / (0.5 / Math.SQRT2)), nw = dB(rmsOf(newRs(x, 24000).y, 200) / (0.5 / Math.SQRT2));
    alias[f] = {old_db: r2(o), new_db: r2(nw), lands_at_hz: 16000 - f};
    check(`(a) 24 kHz tone at ${f / 1000} kHz (alias lands at ${(16000 - f) / 1000} kHz): >= 60 dB rejection (old ${r2(o)} dB)`, nw <= -60, r2(nw));
  }
  res.alias_24k = alias; OUT.freq = res;
  // the worklet itself: 128-sample quanta, int16, cfg message
  const T = loadTap(24000); T.feed(tone(1000, 24000, 24000, 0.5));
  const p = T.pcm(), c = T.cfg();
  check('(a) worklet at 24 kHz: cfg says fir on, 118 taps, ~1.2 ms', c && c.fir === true && c.taps === 118 && Math.abs(c.delayMs - 1.219) < 0.01, c);
  check('(a) worklet: 20 ms chunks of 320 int16 (16000 in 1 s at 24 kHz -> 50 chunks)', p.length === 16000 && T.posts.filter(q => q instanceof Int16Array).every(q => q.length === 320), p.length);
  const pf = Float64Array.from(p, v => v / 32767);
  check('(a) worklet: 1 kHz tone comes out at the same level (int16, rounding)', Math.abs(dB(fitAmp(pf, 1000, 16000, 200).amp / 0.5)) < 0.05, r2(dB(fitAmp(pf, 1000, 16000, 200).amp / 0.5)));
  const clip = loadTap(24000); clip.feed(tone(500, 24000, 4800, 1.5));
  const cp = clip.pcm(); check('(a) worklet: overload clips to +-32767, no wrap-around', Math.max(...cp) === 32767 && Math.min(...cp) === -32767, [Math.max(...cp), Math.min(...cp)]);
  const L = loadTap(24000, {fir: false}); L.feed(tone(1000, 24000, 24000, 0.5));
  check('(a) processorOptions {fir:false} = the old picking, sample for sample', L.cfg().fir === false && L.pcm().every((v, i) => v === (oldPick(tone(1000, 24000, 24000, 0.5), 24000)[i] * 32767 | 0)), L.cfg());
  const P16 = loadTap(16000); const x16 = tone(440, 16000, 16000); P16.feed(x16);
  check('(a) a 16 kHz context passes through (no filter, no delay)', P16.cfg().taps === 0 && P16.pcm().every((v, i) => v === Math.round(x16[i] * 32767)), P16.cfg());
}

// ================================================================== (b) streaming equivalence
{ let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (const rate of [24000, 44100, 48000, 22050]) {
    const n = rate * 3, x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = 0.3 * Math.sin(i * 0.013) + 0.2 * (rnd() - 0.5) + 0.25 * Math.sin(2 * Math.PI * 9500 * i / rate);
    const one = newRs(x, rate).y;
    let same = true;
    for (let trial = 0; trial < 5 && same; trial++) {
      const R = makeResampler(rate, 16000), y = new Float32Array(R.outMax(n)), buf = new Float32Array(R.outMax(400)); let o = 0;
      for (let i = 0; i < n;) { const b = 1 + Math.floor(rnd() * (trial === 0 ? 3 : 300)), e = Math.min(n, i + b); const m = R.run(x.subarray(i, e), e - i, buf); y.set(buf.subarray(0, m), o); o += m; i = e; }
      same = o === one.length && y.subarray(0, o).every((v, i) => v === one[i]);
    }
    check(`(b) ${rate} Hz: random block splits (1..300 samples, 1..3 samples) are bit-identical to one shot`, same);
    const T = loadTap(rate); T.feed(x, 128); const p = T.pcm();
    const want = Int16Array.from(one.subarray(0, p.length), v => Math.round(Math.max(-1, Math.min(1, v)) * 32767));
    check(`(b) ${rate} Hz: the worklet over 128-sample quanta = the core, bit for bit (${p.length} samples)`, p.length > 0 && p.every((v, i) => v === want[i]));
  }
}

// ================================================================== (c) long runs: count + drift
{ const res = {};
  for (const rate of [24000, 44100, 48000]) {
    const R = makeResampler(rate, 16000), blk = new Float32Array(128), out = new Float32Array(R.outMax(128));
    const secs = +(process.env.MIC_LONG_S || 600), f = 1000, tab = tone(f, rate, rate); let n = 0, o = 0, worst = 0; const tail = new Float32Array(16000); let ti = 0; const tailFrom = Math.round((secs - 1) * 16000);
    while (n < secs * rate) {
      for (let i = 0; i < 128; i++) blk[i] = tab[(n + i) % rate];
      const m = R.run(blk, 128, out);
      for (let i = 0; i < m; i++, o++) if (o >= tailFrom && ti < tail.length) tail[ti++] = out[i];
      n += 128;
      if ((n / 128) % 997 === 0) worst = Math.max(worst, Math.abs(o - Math.round(n * 16000 / rate)));
    }
    worst = Math.max(worst, Math.abs(o - Math.round(n * 16000 / rate)));
    // drift: the output tone's phase 10 minutes in = the exact clock (input phase - the filter's fixed delay)
    const fit = fitAmp(tail.subarray(0, ti), f, 16000, 0), t0 = tailFrom / 16000;
    let dphi = fit.phase - (2 * Math.PI * f * (t0 - R.delaySec)); dphi = Math.atan2(Math.sin(dphi), Math.cos(dphi));
    const driftUs = dphi / (2 * Math.PI * f) * 1e6;
    res[rate] = {in: n, out: o, expected: Math.round(n * 16000 / rate), worst_dev: worst, phase_err_us: r2(driftUs)};
    check(`(c) ${rate} Hz, ${secs} s in 128-sample quanta: output count within +-1 of round(n*16000/rate) at every checkpoint`, worst <= 1, res[rate]);
    check(`(c) ${rate} Hz: no drift (1 kHz tone phase at the end within 5 us of the exact clock)`, Math.abs(driftUs) < 5, r2(driftUs));
  }
  OUT.longrun = res;
}

// ================================================================== (d) band SNR on speech vs a high-quality reference
function readWav(p) {
  const b = fs.readFileSync(p); let off = 12, sr = 0, ch = 1, bits = 16, data = null;
  while (off + 8 <= b.length) { const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(off + 10); sr = b.readUInt32LE(off + 12); bits = b.readUInt16LE(off + 22); }
    if (id === 'data') { data = b.subarray(off + 8, off + 8 + sz); break; } off += 8 + sz + (sz & 1); }
  if (!data || bits !== 16) return null;
  const n = Math.floor(data.length / 2 / ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < ch; c++) s += data.readInt16LE(2 * (i * ch + c)); x[i] = s / ch / 32768; }
  return {sr, x};
}
const KER = new Map();
function refAt(x, t, fc, rate, half) {  // ideal band-limited value of x at fractional input time t: direct Kaiser windowed sinc, beta 14 (independent of makeResampler)
  const a = Math.ceil(t - half), fr = Math.round((a - t) * 1e9) / 1e9, key = fr + '|' + fc + '|' + rate + '|' + half;
  let k = KER.get(key);
  if (!k) {
    const beta = 14, i0 = v => { let s = 1, q = 1; for (let m = 1; m < 80; m++) { q *= (v / (2 * m)) ** 2; s += q; if (q < 1e-14 * s) break; } return s; }, ib = i0(beta), c = 2 * fc / rate;
    k = new Float64Array(2 * half + 2);
    for (let m = 0; m < k.length; m++) { const d = -(fr + m), r = d / half; k[m] = r * r >= 1 ? 0 : (d === 0 ? c : Math.sin(Math.PI * c * d) / (Math.PI * d)) * i0(beta * Math.sqrt(1 - r * r)) / ib; }
    if (KER.size > 5000) KER.clear(); KER.set(key, k);
  }
  let acc = 0;
  for (let m = 0; m < k.length; m++) { const n = a + m; if (n >= 0 && n < x.length) acc += x[n] * k[m]; }
  return acc;
}
function fft(re, im) {                  // in-place radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let j = 0; j < len / 2; j++) {
      const ur = re[i + j], ui = im[i + j], vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci, vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
      re[i + j] = ur + vr; im[i + j] = ui + vi; re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
}
function welch(y, N = 1024) {          // power per bin (Hann, 50 % overlap), bins at k * 16000 / N
  const P = new Float64Array(N / 2 + 1), w = Float64Array.from({length: N}, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  for (let s = 0; s + N <= y.length; s += N / 2) { const re = new Float64Array(N), im = new Float64Array(N); for (let i = 0; i < N; i++) re[i] = y[s + i] * w[i]; fft(re, im); for (let k = 0; k <= N / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; }
  return P;
}
const BANDS = [[0, 1000], [1000, 3000], [3000, 5000], [5000, 8000], [5000, 7000], [7000, 8000]];   // 5-7 / 7-8: the NEW filter rolls off on purpose from 6.8 kHz (-3 dB at 7.5)
function bandSnr(ref, y) {
  const n = Math.min(ref.length, y.length), e = new Float64Array(n); for (let i = 0; i < n; i++) e[i] = y[i] - ref[i];
  const Pr = welch(ref.subarray ? ref.subarray(0, n) : ref.slice(0, n)), Pe = welch(e), N = 1024;
  const band = (lo, hi) => { let a = 0, b = 0; for (let k = 0; k <= N / 2; k++) { const f = k * 16000 / N; if (f >= lo && f < hi) { a += Pr[k]; b += Pe[k]; } } return 10 * Math.log10(a / Math.max(b, 1e-30)); };
  let a = 0, b = 0; for (let i = 0; i < n; i++) { a += ref[i] * ref[i]; b += e[i] * e[i]; }
  return {all: r2(10 * Math.log10(a / b)), b: BANDS.map(([lo, hi]) => r2(band(lo, hi)))};
}
function to24k(x, sr) {                // a clean conversion of the clip to the 24 kHz context rate (same independent reference)
  if (sr === 24000) return x; const n = Math.floor(x.length * 24000 / sr), y = new Float32Array(n), fc = Math.min(sr, 24000) * 0.49;
  for (let j = 0; j < n; j++) y[j] = refAt(x, j * sr / 24000, fc, sr, 96 * Math.max(1, sr / 24000)); return y;
}
{ const spec = process.env.MIC_WAVS || '';
  let files = [];
  for (const s of spec.split(',').filter(Boolean)) { if (fs.existsSync(s) && fs.statSync(s).isDirectory()) files.push(...fs.readdirSync(s).filter(f => /\.wav$/i.test(f)).sort().map(f => path.join(s, f))); else if (/\.wav$/i.test(s)) files.push(s); }
  if (!files.length) console.log('SKIP (d) band SNR on speech: set MIC_WAVS=<dir|a.wav,b.wav> (speech WAVs)');
  const rows = [], agg = {old: BANDS.concat([0]).map(() => []), new: BANDS.concat([0]).map(() => [])};
  for (const f of files) {
    const w = readWav(f); if (!w) { console.log('SKIP ' + f + ' (not 16-bit PCM WAV)'); continue; }
    const x = to24k(w.x, w.sr), hiE = (() => { const P = welch(x.length >= 1024 ? Float64Array.from(x) : new Float64Array(1024), 1024); let a = 0, b = 0; P.forEach((v, k) => { const fq = k * 24000 / 1024; a += v; if (fq >= 8000) b += v; }); return r2(10 * Math.log10(b / a)); })();
    const D = makeResampler(24000, 16000).delaySec * 24000, nOut = Math.floor((x.length - 1) / 1.5);
    const fc = 7800, half = 256;
    const refNew = new Float64Array(nOut), refOld = new Float64Array(nOut);
    for (let j = 0; j < nOut; j++) { refNew[j] = refAt(x, j * 1.5 - D, fc, 24000, half); refOld[j] = refAt(x, 0.75 + j * 1.5, fc, 24000, half); }
    const yNew = newRs(x, 24000).y, yOld = oldPick(x, 24000);
    const skip = 64;                    // the filter's start-up
    const sN = bandSnr(refNew.subarray(skip, nOut - skip), Float64Array.from(yNew.subarray(skip, nOut - skip)));
    const sO = bandSnr(refOld.subarray(skip, nOut - skip), yOld.subarray(skip, nOut - skip));
    rows.push({clip: path.basename(f), secs: r2(x.length / 24000), above8k_db: hiE, old: sO, new: sN});
    [sO, sN].forEach((s, k) => { const A = k ? agg.new : agg.old; A[0].push(s.all); s.b.forEach((v, i) => A[i + 1].push(v)); });
    console.log(`  ${path.basename(f)} (${r2(x.length / 24000)} s, energy above 8 kHz ${hiE} dB): SNR all/0-1/1-3/3-5/5-8/5-7/7-8 kHz  OLD ${sO.all} / ${sO.b.join(' / ')}   NEW ${sN.all} / ${sN.b.join(' / ')}`);
  }
  if (rows.length) {
    const med = a => { const s = a.slice().sort((p, q) => p - q); return r2(s[Math.floor(s.length / 2)]); };
    const mo = agg.old.map(med), mn = agg.new.map(med);
    console.log(`  median over ${rows.length} clips: OLD ${mo.join(' / ')}   NEW ${mn.join(' / ')}`);
    OUT.speech = {rows, median: {old: mo, new: mn}, bands: ['all', ...BANDS.map(b => b.join('-'))]};
    check('(d) speech: NEW beats OLD by >= 30 dB in 3-5 kHz and 5-7 kHz on every clip', rows.every(r => r.new.b[2] - r.old.b[2] >= 30 && r.new.b[4] - r.old.b[4] >= 30), rows.map(r => [r.clip, r2(r.new.b[2] - r.old.b[2]), r2(r.new.b[4] - r.old.b[4])]));
    check('(d) speech: NEW beats OLD in 5-8 kHz on every clip (the 7-8 kHz roll-off is deliberate)', rows.every(r => r.new.b[3] > r.old.b[3]), rows.map(r => [r.clip, r2(r.new.b[3] - r.old.b[3])]));
    check('(d) speech: NEW >= 60 dB SNR in 0-5 kHz and >= 45 dB in 5-7 kHz on every clip', rows.every(r => [0, 1, 2].every(i => r.new.b[i] >= 60) && r.new.b[4] >= 45), rows.map(r => r.new.b));
  }
}

// ================================================================== (e) page.html down16: the direct-mode mic to the hub's ASR (16 kHz mu-law)
{ const s = fs.readFileSync(path.join(ROOT, 'app/page.html'), 'utf8'), code = s.slice(s.indexOf('const DS = (() => {'), s.indexOf('function rigMsg('));
  const mk = fir => new Function('Q', 'window', code + '; return down16;')({get: k => (k === 'micfir' && !fir ? '0' : null)}, W);
  const level = (fir, f) => { const d = mk(fir), y = []; for (let c = 0; c < 50; c++) { const pcm = new Int16Array(480); for (let i = 0; i < 480; i++) pcm[i] = Math.round(16000 * Math.sin(2 * Math.PI * f * (c * 480 + i) / 24000)); y.push(...d(pcm)); }
    const t = y.slice(200); return r2(dB(Math.sqrt(t.reduce((p, v) => p + v * v, 0) / t.length) / (16000 / Math.SQRT2))); };
  const res = {};
  for (const f of [3000, 6000, 6800, 7500, 8600, 9000, 10000, 11000, 11900]) res[f] = {old: level(false, f), new: level(true, f)};
  OUT.down16 = res;
  check('(e) down16 NEW: flat to 6.8 kHz (old -3.9 dB there)', [3000, 6000, 6800].every(f => Math.abs(res[f].new) <= 0.5), res);
  check('(e) down16 NEW: >= 60 dB alias rejection from 8.6 kHz (old -20 dB at 8.6 kHz)', [8600, 9000, 10000, 11000, 11900].every(f => res[f].new <= -60), Object.fromEntries([8600, 9000, 10000].map(f => [f, res[f]])));
  const d = mk(true), lens = []; for (let c = 0; c < 20; c++) { const pcm = new Int16Array(480); for (let i = 0; i < 480; i++) pcm[i] = (c * 480 + i) % 97 * 300 - 14000; lens.push(d(pcm).length); }
  check('(e) down16 NEW: exactly 320 samples per 480-sample chunk from the first chunk', lens.every(n => n === 320), lens);
  const big = mk(true)(Int16Array.from({length: 480}, (_, i) => (i % 2 ? 32767 : -32768)));
  check('(e) down16 NEW: int16 range kept (clipped, no wrap)', big.every(v => v >= -32768 && v <= 32767));
}

if (process.env.MIC_JSON) fs.writeFileSync(process.env.MIC_JSON, JSON.stringify(OUT, null, 1));
console.log(`\n${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
