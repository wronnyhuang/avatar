// gaze split — the measurable check (a tool, not a suite test; tests/gaze_split_test.cjs is the suite test).
//   node tests/gaze_split_eval.cjs [PERCEPTION_DIR] [--json out.json]
// Replays (a) the real-clip perception logs (MediaPipe landmarks of real camera footage, tests/perception_smoke.cjs output:
// <clip>.jsonl) and (b) a synthetic sweep of face positions + a noisy still face through app/motion.js twice — the old
// split (config gazeSplit:false = ?gazesplit=0) and the new one — and reports, for the frames where it is following his face:
//   share of the aim carried by the head / by the eyes-in-head (least-squares slope of head angle on the eye-in-world aim),
//   head yaw/pitch travel (RMS about the mean, range), aim error vs his face (eye-in-world angle − gazeFollow × his offset),
//   jitter (RMS frame-to-frame acceleration of the head and of the eye aim; and their spread on a noisy still face).
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const M = load('app/motion.js');
let PERC = null; try { PERC = load('app/perception.js'); } catch (e) {}
const args = process.argv.slice(2), ji = args.indexOf('--json'), JSON_OUT = ji >= 0 ? args[ji + 1] : null;
const DIR = (args[0] && !args[0].startsWith('--') ? args[0] : null) || process.env.PERCEPTION_DIR || '';
const DT = 1 / 30, DEG = 180 / Math.PI;
const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const rms = a => Math.sqrt(mean(a.map(x => x * x)));
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const pct = (a, p) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : 0; };
const range = a => a.length ? Math.max(...a) - Math.min(...a) : 0;
const slope = (x, y) => { let sxy = 0, sxx = 0; for (let i = 0; i < x.length; i++) { sxy += x[i] * y[i]; sxx += x[i] * x[i]; } return sxx > 1e-9 ? sxy / sxx : NaN; };

// one run → per-frame rows {user (following his face), Wx, Wy (eye-in-world aim, rad), Hx, Hy (the head's FOLLOWING layer, rad,
// + = right / up: the gaze spring, before idle/speech head noise), Ox, Oy (the head as output: following + life), Tx, Ty (target aim)}
function runFrames(split, frames, seed) {
  const m = M.createMotion({seed: seed || 9, config: {gazeSplit: split}});
  const rows = [];
  for (const fr of frames) {
    const o = m.step(DT, fr), u = m.state.user, cfg = m.config || M.DEFAULTS;
    const c = Math.cos(o.head_roll), s = Math.sin(o.head_roll), lx = o.gaze_x * M.GAZE_RX, ly = o.gaze_y * M.GAZE_RY;
    rows.push({user: m.debug().mode === 'user' && u.present, Wx: c * lx + s * ly, Wy: -s * lx + c * ly, Hx: m.state.hy.x, Hy: m.state.hp.x, Ox: o.head_yaw, Oy: -o.head_pitch,
      Tx: M.DEFAULTS.gazeFollow * (u.ax || 0), Ty: M.DEFAULTS.gazeFollow * (u.ay || 0), gx: o.gaze_x, gy: o.gaze_y});
  }
  return rows;
}
function metrics(runs) {   // runs = [rows, rows, …] (one per seed / clip): accelerations never span two runs
  const rows = [].concat(...runs), U = rows.filter(r => r.user);
  const acc = k => [].concat(...runs.map(rw => rw.slice(2).map((r, i) => (r[k] - 2 * rw[i + 1][k] + rw[i][k]) / (DT * DT))));
  const Wx = U.map(r => r.Wx), Wy = U.map(r => r.Wy), Hx = U.map(r => r.Hx), Hy = U.map(r => r.Hy);
  const hsx = slope(Wx, Hx), hsy = slope(Wy, Hy), hs = slope(Wx.concat(Wy), Hx.concat(Hy));
  const err = U.map(r => Math.hypot(r.Wx - r.Tx, r.Wy - r.Ty) * DEG);
  const all = k => U.map(r => r[k]);   // travel: over the frames where it is following his face
  return {frames: rows.length, following: U.length,
    headShare: hs, eyeShare: 1 - hs, headShareYaw: hsx, headSharePitch: hsy,
    headYawRmsDeg: sd(all('Hx')) * DEG, headYawRangeDeg: range(all('Hx')) * DEG, headPitchRmsDeg: sd(all('Hy')) * DEG, headPitchRangeDeg: range(all('Hy')) * DEG,
    outYawRmsDeg: sd(all('Ox')) * DEG, outYawRangeDeg: range(all('Ox')) * DEG, outPitchRmsDeg: sd(all('Oy')) * DEG, outPitchRangeDeg: range(all('Oy')) * DEG,
    outHeadJitter: rms(acc('Ox').concat(acc('Oy'))),
    eyeInHeadMaxDeg: Math.max(0, ...U.map(r => Math.abs(r.Wx - r.Hx))) * DEG, eyeInHeadP95Deg: pct(U.map(r => Math.abs(r.Wx - r.Hx)), 0.95) * DEG,
    aimErrMeanDeg: mean(err), aimErrP95Deg: pct(err, 0.95),
    headJitter: rms(acc('Hx').concat(acc('Hy'))), eyeJitter: rms(acc('Wx').concat(acc('Wy'))),
    gazeDialMax: Math.max(0, ...rows.map(r => Math.max(Math.abs(r.gx), Math.abs(r.gy))))};
}

// ---- (a) the real clips
function clipFrames(fp) {
  const feats = [], evs = [];
  for (const line of fs.readFileSync(fp, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let j; try { j = JSON.parse(line); } catch (e) { continue; }
    if (j.event) { if (!/^model_/.test(j.event.type)) evs.push(j.event); } else if (j.t !== undefined && !j.stats) feats.push(j);
  }
  if (feats.length < 30) return null;
  const tr = PERC && PERC.createExpressionTracker ? PERC.createExpressionTracker() : null;
  const out = []; let fi = 0, ei = 0, f = null;
  for (let t = feats[0].t; t <= feats[feats.length - 1].t; t += DT) {
    const events = [];
    while (fi < feats.length && feats[fi].t <= t) {
      f = Object.assign({}, feats[fi++], {events: undefined});
      if (tr) { const r = tr.update(f.face, f.t); f.face = Object.assign({}, f.face, {xp: {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt}}); for (const e of r.events) events.push(e); }
    }
    while (ei < evs.length && evs[ei].t <= t) events.push(evs[ei++]);
    out.push({features: f, events});
  }
  return out;
}
// ---- (b) synthetic: a sweep of face positions (each held 5 s, x and y in the camera's normalized ±1), then a noisy still face
const face = (x, y) => ({present: true, x, y, z: 0.5, size: 0.2, roll: 0, lookX: 0, lookY: 0});
function sweepFrames() {
  const pos = [];
  for (const x of [-0.9, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9]) pos.push([x, 0.05]);
  for (const y of [-0.7, -0.4, 0.4, 0.7]) pos.push([0, y]);
  for (const p of [[-0.8, 0.6], [0.8, -0.6], [0.5, 0.5], [-0.5, -0.5]]) pos.push(p);
  const fr = []; let t = 0;
  for (const p of pos) for (let i = 0; i < 150; i++) { t += DT; fr.push({features: {t, face: face(p[0], p[1]), attending: true, attention: 0.9, motion: 0.05, hands: []}, pos: p}); }
  return fr;
}
function noisyFrames(x, y, sigma, seconds) {   // landmark noise: independent per perception frame (30 Hz), sd in normalized units
  let s = 12345; const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r());
  const fr = []; let t = 0;
  for (let i = 0; i < seconds * 30; i++) { t += DT; fr.push({features: {t, face: face(x + sigma * g(), y + sigma * g()), attending: true, attention: 0.9, motion: 0.05, hands: []}}); }
  return fr;
}

const R = {clips: {}, real: null, sweep: null, sweepTable: [], noisy: null};
const fmt = (m) => `head share ${(100 * m.headShare).toFixed(0)}% / eyes ${(100 * m.eyeShare).toFixed(0)}% · follow-head yaw rms ${m.headYawRmsDeg.toFixed(2)}° range ${m.headYawRangeDeg.toFixed(1)}° pitch rms ${m.headPitchRmsDeg.toFixed(2)}° range ${m.headPitchRangeDeg.toFixed(1)}° · output head (with idle noise) yaw rms ${m.outYawRmsDeg.toFixed(2)}° range ${m.outYawRangeDeg.toFixed(1)}° pitch rms ${m.outPitchRmsDeg.toFixed(2)}° range ${m.outPitchRangeDeg.toFixed(1)}° · aim err mean ${m.aimErrMeanDeg.toFixed(2)}° p95 ${m.aimErrP95Deg.toFixed(2)}° · jitter follow-head ${m.headJitter.toFixed(3)} output head ${m.outHeadJitter.toFixed(3)} eye ${m.eyeJitter.toFixed(2)} rad/s² · eye-in-head max ${m.eyeInHeadMaxDeg.toFixed(1)}°`;
// the two runs draw different random numbers once the head differs (blink / glance dice), so the transition frames differ by
// chance: pool SEEDS runs each way (rows concatenated; metrics() keeps the runs apart for the frame-to-frame terms)
const SEEDS = [9, 21, 33, 47, 58, 64, 77, 91];
const both = frames => ({before: SEEDS.map(sd => runFrames(false, frames, sd)), after: SEEDS.map(sd => runFrames(true, frames, sd))});

const clips = fs.existsSync(DIR) ? fs.readdirSync(DIR).filter(n => n.endsWith('.jsonl')).sort() : [];
const pool = {before: [], after: [], s: 0};
for (const n of clips) {
  const fr = clipFrames(path.join(DIR, n)); if (!fr) continue;
  const b = both(fr), mb = metrics(b.before), ma = metrics(b.after);
  if (mb.following < 60 * SEEDS.length) continue;
  R.clips[n.slice(0, -6)] = {before: mb, after: ma};
  pool.before.push(...b.before); pool.after.push(...b.after); pool.s += fr.length * DT;
  console.log(`-- ${n.slice(0, -6)} (${(fr.length * DT).toFixed(0)} s, following ${(100 * mb.following / mb.frames).toFixed(0)}%)\n   before: ${fmt(mb)}\n   after : ${fmt(ma)}`);
}
if (pool.before.length) {
  R.real = {before: metrics(pool.before), after: metrics(pool.after), clips: Object.keys(R.clips).length};
  console.log(`\n== REAL CLIPS pooled (${R.real.clips} clips × ${SEEDS.length} seeds, ${pool.s.toFixed(0)} s of footage)\n   before: ${fmt(R.real.before)}\n   after : ${fmt(R.real.after)}`);
} else console.log(`(no perception logs under ${DIR}: real-clip replay skipped)`);

{
  const fr = sweepFrames(), b = both(fr);
  R.sweep = {before: metrics(b.before), after: metrics(b.after)};
  console.log(`\n== SYNTHETIC SWEEP (${(fr.length * DT).toFixed(0)} s × ${SEEDS.length} seeds)\n   before: ${fmt(R.sweep.before)}\n   after : ${fmt(R.sweep.after)}`);
  console.log('   settled per position (deg):  face x,y → aim W | head before → after | eye-in-head before → after');
  for (let k = 0; k * 150 < fr.length; k++) {
    const sl = runs => [].concat(...runs.map(rows => rows.slice(k * 150 + 105, k * 150 + 150).filter(r => r.user))), med = (runs, key) => pct(sl(runs).map(r => r[key]), 0.5) * DEG;
    const p = fr[k * 150].pos, row = {x: p[0], y: p[1], aimBefore: [med(b.before, 'Wx'), med(b.before, 'Wy')], aimAfter: [med(b.after, 'Wx'), med(b.after, 'Wy')],
      headBefore: [med(b.before, 'Hx'), med(b.before, 'Hy')], headAfter: [med(b.after, 'Hx'), med(b.after, 'Hy')]};
    R.sweepTable.push(row);
    const f2 = a => `${a[0].toFixed(1).padStart(6)},${a[1].toFixed(1).padStart(6)}`;
    console.log(`   ${String(p[0]).padStart(5)},${String(p[1]).padStart(5)} → aim ${f2(row.aimBefore)} / ${f2(row.aimAfter)} | head ${f2(row.headBefore)} → ${f2(row.headAfter)} | eye-in-head ${f2([row.aimBefore[0] - row.headBefore[0], row.aimBefore[1] - row.headBefore[1]])} → ${f2([row.aimAfter[0] - row.headAfter[0], row.aimAfter[1] - row.headAfter[1]])}`);
  }
}
{
  // a still face off-centre with landmark noise (sd 0.006 of the half-frame ≈ 0.25° of camera angle, per frame at 30 Hz)
  const fr = noisyFrames(0.35, 0.1, 0.006, 40), b = both(fr), tail = runs => runs.map(rows => rows.slice(300)), flat = runs => [].concat(...tail(runs)).filter(r => r.user);
  const j = rows => ({headYawSdDeg: sd(flat(rows).map(r => r.Hx)) * DEG, headPitchSdDeg: sd(flat(rows).map(r => r.Hy)) * DEG, outYawSdDeg: sd(flat(rows).map(r => r.Ox)) * DEG,
    eyeSdDeg: sd(flat(rows).map(r => r.Wx)) * DEG, headJitter: metrics(tail(rows)).headJitter, eyeJitter: metrics(tail(rows)).eyeJitter});
  R.noisy = {before: j(b.before), after: j(b.after)};
  const f = q => `follow-head yaw sd ${q.headYawSdDeg.toFixed(3)}° pitch sd ${q.headPitchSdDeg.toFixed(3)}° (output head yaw sd ${q.outYawSdDeg.toFixed(2)}°) · eye aim sd ${q.eyeSdDeg.toFixed(3)}° · jitter follow-head ${q.headJitter.toFixed(3)} eye ${q.eyeJitter.toFixed(2)} rad/s²`;
  console.log(`\n== NOISY STILL FACE (x .35, landmark noise sd .006, 40 s)\n   before: ${f(R.noisy.before)}\n   after : ${f(R.noisy.after)}`);
}
if (JSON_OUT) { fs.writeFileSync(JSON_OUT, JSON.stringify(R, null, 1)); console.log('\nwrote ' + JSON_OUT); }
