// node tests/gaze_split_test.cjs — gaze split: following his face, the EYES carry most of the aim, the head less.
// Asserts (app/motion.js reflex path + app/behavior.js api.look('user')):
//   the aim (eye-in-world angle) is the same with the split on and off · the head's share of it drops from ≥ 0.85 to ≤ 0.5 ·
//   the eyes stay inside their range (eye-in-head ≤ eyeMax; the head takes the rest, so its share grows with the offset) ·
//   eyes still lead, no later than before · a noisy still face moves the head no more than before · the switches
//   (config gazeSplit / ?gazesplit=0, eyeShare, eyeMax) work · gazeSplit:false = the old head (headFollow × offset).
// Exit code = number of failures. The numbers for a report: node tests/gaze_split_eval.cjs
'use strict';
const load = require('./load.cjs');
const M = load('app/motion.js');
const SIM = require('../app/behavior_sim.cjs');
let fails = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const DT = 1 / 30, DEG = 180 / Math.PI;
const med = a => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const sd = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
const face = (x, y) => ({present: true, x, y, z: 0.5, size: 0.2, roll: 0, lookX: 0, lookY: 0});
const feat = (t, x, y) => ({features: {t, face: face(x, y), attending: true, attention: 0.9, motion: 0.05, hands: []}});

// hold his face at (x, y) for `sec` seconds → the settled (median of the last 1.5 s, frames following him) aim and head, radians
function settle(config, x, y, sec = 6, seed = 7) {
  const m = M.createMotion({seed, preset: 'default', config});
  const W = [[], []], H = [[], []];
  for (let i = 0; i < sec * 30; i++) {
    const o = m.step(DT, feat((i + 1) * DT, x, y));
    if (i >= (sec - 1.5) * 30 && m.debug().mode === 'user') {
      const c = Math.cos(o.head_roll), s = Math.sin(o.head_roll), lx = o.gaze_x * M.GAZE_RX, ly = o.gaze_y * M.GAZE_RY;
      W[0].push(c * lx + s * ly); W[1].push(-s * lx + c * ly); H[0].push(m.state.hy.x); H[1].push(m.state.hp.x);
    }
  }
  return {W: W.map(med), H: H.map(med), u: [m.state.user.ax, m.state.user.ay], cfg: m.config};
}

// ---------------------------------------------------------------- 1. the split rule itself
{
  const m = 0.2;
  let mono = true, cont = true, lim = true, prev = M.softSat(-1, m);
  for (let x = -1; x <= 1; x += 0.002) { const v = M.softSat(x, m); if (v < prev - 1e-12) mono = false; if (Math.abs(v - prev) > 0.0021) cont = false; if (Math.abs(v) > m) lim = false; prev = v; }
  check('softSat: monotonic, continuous, never past the limit, identity while small', mono && cont && lim && M.softSat(0.05, m) === 0.05 && M.softSat(-0.05, m) === -0.05, '');
  const h0 = M.splitHead([0.25, 0.1], 0, m), h7 = M.splitHead([0.1, 0.05], 0.7, m), hBig = M.splitHead([0.6, 0], 0.7, m);
  check('splitHead: share 0 → the head carries the whole aim', Math.abs(h0[0] - 0.25) < 1e-12 && Math.abs(h0[1] - 0.1) < 1e-12, h0.join(', '));
  check('splitHead: inside the eye range the head carries 1 − share', Math.abs(h7[0] - 0.03) < 1e-9 && Math.abs(h7[1] - 0.015) < 1e-9, h7.join(', '));
  check('splitHead: past the eye range the head takes the rest (eye-in-head ≤ eyeMax)', 0.6 - hBig[0] <= m + 1e-9 && 0.6 - hBig[0] > 0.9 * m, `eye-in-head ${(0.6 - hBig[0]).toFixed(3)}`);
  check('defaults: gazeSplit on, eyeShare > 0.5', M.DEFAULTS.gazeSplit === true && M.DEFAULTS.eyeShare > 0.5 && M.DEFAULTS.eyeMax > 0.1 && M.DEFAULTS.eyeMax <= 0.3, JSON.stringify({s: M.DEFAULTS.eyeShare, m: M.DEFAULTS.eyeMax}));
}

// ---------------------------------------------------------------- 2. a sweep of face positions: same aim, less head, eyes in range
{
  const POS = [[-0.9, 0.05], [-0.6, 0.05], [-0.3, 0.05], [0.3, 0.05], [0.6, 0.05], [0.9, 0.05], [0, 0.6], [0, -0.6], [0.7, 0.5], [-0.7, -0.5]];
  const EM = M.DEFAULTS.eyeMax, TOL = 0.03;   // micro-saccades / drift
  let aimOk = true, oldOk = true, newOk = true, rangeOk = true, lessOk = true; const lines = [], share = {};
  for (const [x, y] of POS) {
    const a = settle({gazeSplit: false}, x, y), b = settle({gazeSplit: true}, x, y);
    const ax = Math.abs(x) >= Math.abs(y) ? 0 : 1, Wb = b.W[ax], Wa = a.W[ax];
    if (Math.hypot(a.W[0] - b.W[0], a.W[1] - b.W[1]) > TOL) aimOk = false;                       // the aim did not move
    const want = M.DEFAULTS.gazeFollow * b.u[ax];
    if (Math.abs(Wb - want) > TOL) aimOk = false;                                                // … and still points at his face
    const so = a.H[ax] / Wa, sn = b.H[ax] / Wb; share[x + ',' + y] = sn;
    if (!(so >= 0.85)) oldOk = false;
    if (!(sn <= 0.5 && sn >= 0)) newOk = false;
    if (Math.abs(b.W[0] - b.H[0]) > EM + TOL || Math.abs(b.W[1] - b.H[1]) > 0.75 * EM + TOL) rangeOk = false;
    if (!(Math.abs(b.H[ax]) < 0.6 * Math.abs(a.H[ax]))) lessOk = false;
    lines.push(`(${x},${y}) aim ${(Wa * DEG).toFixed(1)}→${(Wb * DEG).toFixed(1)}° head ${(a.H[ax] * DEG).toFixed(1)}→${(b.H[ax] * DEG).toFixed(1)}° share ${so.toFixed(2)}→${sn.toFixed(2)}`);
  }
  console.log('   ' + lines.join('\n   '));
  check('sweep: the aim is unchanged by the split and points at his face (≤ 1.7°)', aimOk, '');
  check('sweep: old split — the head carries ≥ 85% of the aim (it out-turned the eyes sideways)', oldOk, '');
  check('sweep: new split — the head carries ≤ 50% of the aim, the eyes the rest', newOk, '');
  check('sweep: the head turns < 60% of what it did', lessOk, '');
  check('sweep: eye-in-head stays inside the eye range (yaw ≤ eyeMax, pitch ≤ 0.75 eyeMax)', rangeOk, '');
  check('sweep: the head takes a larger share for a larger offset', share['0.9,0.05'] > share['0.3,0.05'] + 0.03 && share['-0.9,0.05'] > share['-0.3,0.05'] + 0.03,
    `share at x .3 ${share['0.3,0.05'].toFixed(2)}, at x .9 ${share['0.9,0.05'].toFixed(2)}`);
}

// ---------------------------------------------------------------- 3. switches and tunables
{
  const off = settle({gazeSplit: false}, 0.6, 0.05), on = settle({}, 0.6, 0.05);
  check('gazeSplit:false = the old head (headFollow × his offset)', Math.abs(off.H[0] - M.DEFAULTS.headFollow * off.u[0]) < 0.01, `${off.H[0].toFixed(3)} vs ${(M.DEFAULTS.headFollow * off.u[0]).toFixed(3)}`);
  const s9 = settle({eyeShare: 0.9, eyeMax: 0.3}, 0.6, 0.05), s3 = settle({eyeShare: 0.3}, 0.6, 0.05), tight = settle({eyeMax: 0.05}, 0.9, 0.05);
  check('eyeShare tunes the split (0.9 → less head than 0.7 → less than 0.3)', Math.abs(s9.H[0]) < Math.abs(on.H[0]) - 0.01 && Math.abs(on.H[0]) < Math.abs(s3.H[0]) - 0.01, `${s9.H[0].toFixed(3)} < ${on.H[0].toFixed(3)} < ${s3.H[0].toFixed(3)}`);
  check('eyeMax caps the eye-in-head angle (0.05 rad → the head takes the rest)', Math.abs(tight.W[0] - tight.H[0]) < 0.05 + 0.03, `eye-in-head ${(tight.W[0] - tight.H[0]).toFixed(3)}`);
  const saved = globalThis.location;
  try {
    globalThis.location = {search: '?gazesplit=0'};
    const url0 = settle(undefined, 0.6, 0.05);
    check('?gazesplit=0 turns the split off (page URL)', url0.cfg.gazeSplit === false && Math.abs(url0.H[0] - off.H[0]) < 0.01, `head ${url0.H[0].toFixed(3)}`);
    globalThis.location = {search: '?eyeshare=0.4&eyemax=8'};
    const url1 = M.createMotion({seed: 1}).config;
    check('?eyeshare= / ?eyemax=<deg> set the two constants', url1.gazeSplit === true && url1.eyeShare === 0.4 && Math.abs(url1.eyeMax - 8 / DEG) < 1e-9, JSON.stringify({s: url1.eyeShare, m: url1.eyeMax}));
    globalThis.location = {search: '?gazesplit=0'};
    check('an explicit config value wins over the URL', M.createMotion({seed: 1, config: {gazeSplit: true}}).config.gazeSplit === true, '');
  } finally { if (saved === undefined) delete globalThis.location; else globalThis.location = saved; }
}

// ---------------------------------------------------------------- 4. dynamics: eyes lead, no later than before; the head arrives smoothly
{
  const run = (config, seed) => {
    const m = M.createMotion({seed, preset: 'default', config}), rows = [];
    for (let i = 0; i < 9 * 30; i++) { const t = (i + 1) * DT; const o = m.step(DT, feat(t, t < 5 ? 0.6 : -0.5, 0.05)); rows.push({t, g: o.gaze_x, h: m.state.hy.x, user: m.debug().mode === 'user'}); }
    return rows;
  };
  const onset = [[], []], lead = [], hstep = [];
  for (const seed of [3, 11, 19, 27, 35, 43]) {
    [false, true].forEach((sp, k) => {
      const r = run({gazeSplit: sp}, seed), g0 = r.find(q => q.t >= 4.9).g, g1 = r[r.length - 1].g, h0 = r.find(q => q.t >= 4.9).h, h1 = r[r.length - 1].h;
      const tg = r.find(q => q.t > 5 && q.g < g0 - 0.08), tgh = r.find(q => q.t > 5 && q.g < (g0 + g1) / 2), thh = r.find(q => q.t > 5 && q.h < (h0 + h1) / 2);
      if (tg) onset[k].push(tg.t - 5);
      if (sp && tgh && thh) lead.push(thh.t - tgh.t);
      if (sp) hstep.push(Math.max(...r.slice(1).map((q, i) => Math.abs(q.h - r[i].h))));
    });
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  check('step: eyes react 130–360 ms after he moves, split on (every seed)', onset[1].length === 6 && onset[1].every(v => v >= 0.13 && v <= 0.36), onset[1].map(v => v.toFixed(2)).join(' '));
  check('step: no added eye latency (mean onset within 40 ms of the old split)', Math.abs(mean(onset[1]) - mean(onset[0])) <= 0.04, `old ${mean(onset[0]).toFixed(3)} s, new ${mean(onset[1]).toFixed(3)} s`);
  check('step: eyes lead, the head follows (head 50% ≥ 80 ms after eyes 50%)', lead.length === 6 && lead.every(v => v >= 0.08), lead.map(v => v.toFixed(2)).join(' '));
  check('step: the head eases (no frame-to-frame jump > 1.2°)', Math.max(...hstep) * DEG < 1.2, `${(Math.max(...hstep) * DEG).toFixed(2)}°`);
}

// ---------------------------------------------------------------- 5. landmark noise on a still face: the head moves no more than before
{
  let s = 4914; const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }, g = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r());
  const frames = []; for (let i = 0; i < 30 * 30; i++) frames.push(feat((i + 1) * DT, 0.35 + 0.006 * g(), 0.1 + 0.006 * g()));
  const spread = config => { const m = M.createMotion({seed: 5, preset: 'default', config}), h = [], e = []; frames.forEach((f, i) => { m.step(DT, f); if (i > 240 && m.debug().mode === 'user') { h.push(m.state.hy.x); e.push(m.state.eye.W[0]); } }); return {h: sd(h) * DEG, e: sd(e) * DEG}; };
  const a = spread({gazeSplit: false}), b = spread({gazeSplit: true});
  check('noise: a noisy still face moves the following head no more than before', b.h <= a.h + 0.02, `head sd old ${a.h.toFixed(3)}°, new ${b.h.toFixed(3)}°`);
  check('noise: the eye aim stays steady (sd < 1°)', b.e < 1.0, `eye sd old ${a.e.toFixed(3)}°, new ${b.e.toFixed(3)}°`);
}

// ---------------------------------------------------------------- 6. behaviour programs: api.look('user') splits the same way; other targets do not change
{
  const saved = globalThis.location;
  const look = (code, at, search) => {
    if (search === undefined) delete globalThis.location; else globalThis.location = {search};
    try { const r = SIM.dryRun(code, {character: 'fox', script: {dur: 3, at}, trace: true}); return r.trace.find(x => x.t >= 2.5).sets; }
    finally { if (saved === undefined) delete globalThis.location; else globalThis.location = saved; }
  };
  const U = `({ per_frame(api) { api.look('user'); } })`, HND = `({ per_frame(api) { api.look('right_hand'); } })`;
  const on = look(U, [[0, 3, {x: 0.6}]]), off = look(U, [[0, 3, {x: 0.6}]], '?gazesplit=0');
  check("api.look('user'): same eye aim, the head turns < 60% of the old", Math.abs(on.gaze_x[0] - off.gaze_x[0]) < 1e-6 && on.gaze_x[0] > 0.5 && on.head_yaw[0] > 0.02 && on.head_yaw[0] < 0.6 * off.head_yaw[0],
    `gaze ${on.gaze_x[0].toFixed(2)} head ${off.head_yaw[0].toFixed(3)} → ${on.head_yaw[0].toFixed(3)}`);
  const eyeW = on.gaze_x[0] * Math.atan(0.30);
  check("api.look('user'): eye-in-head inside the eye range", Math.abs(eyeW - on.head_yaw[0]) <= M.DEFAULTS.eyeMax + 1e-6, `${(eyeW - on.head_yaw[0]).toFixed(3)} rad`);
  const h1 = look(HND, [[0, 3, {hand_right: {x: 0.6, y: 0.1, open: 1}}]]), h0 = look(HND, [[0, 3, {hand_right: {x: 0.6, y: 0.1, open: 1}}]], '?gazesplit=0');
  check("api.look(<a hand>): unchanged by the split", Math.abs(h1.head_yaw[0] - h0.head_yaw[0]) < 1e-9 && Math.abs(h1.gaze_x[0] - h0.gaze_x[0]) < 1e-9, `head ${h1.head_yaw[0].toFixed(3)}`);
}

console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails);
