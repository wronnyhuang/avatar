// node tests/sem_page_handover_test.cjs — the thinking lead-in -> move handover and
// the cue hold, measured end to end on one fake clock: app/perform.js + app/thinking.js + app/motion.js (muse, fox), a
// semantic plan whose first rig poses land 1.6 s after the call (the eval's fresh-plan median), no voice audio (the cue
// times out, as with the mock voice). BEFORE = how a rig plan ran before this round (the thinking paw allowed, 700 ms cue
// hold — reproduced with the planner set to 'current' and the same rig batches); AFTER = semantic mode now. Prints the
// worst per-frame step on the arm / head channels from the call to 1 s into the move, the thinking paw's excursion before
// the move, and call -> first motion. Checks AFTER is no worse on the steps, has no paw flick and starts sooner.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js'), Perf = load('app/perform.js'), Think = load('app/thinking.js');
const ROOT = path.join(__dirname, '..'), J = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), 'utf8'));
const BODY = {muse: J('renderer/animals/muse.body.json')};
const PROPS = {muse: J('renderer/animals/muse.props.json')};
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const smooth = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
const env = (t, a, b, r = 0.4) => smooth((t - a) / r) * (1 - smooth((t - (b - r)) / r));
// the move: a reach with the OTHER paw (screen-right = own left) + a head turn — the thinking paw is not in it
const POSES = []; for (let k = 0; k <= 60; k++) { const t = k / 20; POSES.push({t, hand_r_x: +(-0.3 * env(t, 0, 3)).toFixed(4), hand_r_y: +(0.8 * env(t, 0, 3)).toFixed(4), head_yaw: +(-0.2 * env(t, 0.2, 2.8)).toFixed(4)}); }
const CH = ['hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z', 'head_yaw', 'head_pitch', 'head_roll', 'body_lean'];
const DT = 1 / 30, PLAN_MS = 1600;

function run(name, mode, seed) {
  let tMs = 0; const timers = [];
  const T = {set: (f, ms) => { const x = {f, at: tMs + ms, done: false}; timers.push(x); return x; }, clear: x => { if (x) x.done = true; }};
  let rs = (seed * 7919 + 12345) % 2147483647; const rand = () => { rs = (rs * 16807) % 2147483647; return rs / 2147483647; };
  let P = null;
  const m = M.createMotion({seed, character: name, onAction: e => { if (P) P.onAction(e); }});
  if (BODY[name]) m.setProps(JSON.parse(JSON.stringify(PROPS[name])), JSON.parse(JSON.stringify(BODY[name])), name); else m.setProps(null, null, name);
  const K = Think.createThinker({motion: () => m, now: () => tMs, timers: T, rand, arms: () => true, fingers: () => name === 'muse'});
  P = Perf.createPerformer({motion: () => m, now: () => tMs, timers: T, thinker: () => K, character: () => name, prompt: () => null, sendCtl: () => {},
    planner: () => mode, semState: () => ({user: {present: true, ax: 0, ay: 0}}), cache: false});
  let t = 0; const rows = [];
  const step = () => { t += DT; tMs += DT * 1000; for (const x of timers.slice()) if (!x.done && x.at <= tMs) { x.done = true; x.f(); }
    rows.push(Object.assign({ms: tMs}, m.step(DT, {features: {t, face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}}))); };
  for (let i = 0; i < 45; i++) step();
  const r = P.call('reach out with your left paw and look over there'); const call = tMs;
  let sent = 0;
  while (tMs < call + PLAN_MS + 5000) {
    if (!sent && tMs >= call + PLAN_MS) { P.onKf({type: 'kf', id: r.id, space: 'rig', hz: 20, poses: POSES.slice(0, 20), done: false}); sent = 1; }
    if (sent === 1 && tMs >= call + PLAN_MS + 400) { P.onKf({type: 'kf', id: r.id, space: 'rig', poses: POSES.slice(20), done: true, ok: true, source: 'planner', planner: 'semantic'}); sent = 2; }
    step();
  }
  const job = P.list().find(x => x.id === r.id), fm = job.marks.first_motion_ms;
  const i0 = rows.findIndex(x => x.ms >= call), i1 = rows.findIndex(x => x.ms >= fm + 1000), iS = rows.findIndex(x => x.ms >= fm);
  let worst = 0, wc = '';
  for (let i = i0 + 1; i <= i1; i++) for (const c of CH) { const d = Math.abs(rows[i][c] - rows[i - 1][c]); if (d > worst) { worst = d; wc = c + ' @' + Math.round(rows[i].ms - call) + ' ms'; } }
  const base = rows[i0 - 1];
  let flick = 0; for (let i = i0; i < iS + 12; i++) for (const c of ['hand_l_y', 'hand_l_x', 'hand_l_z']) flick = Math.max(flick, Math.abs(rows[i][c] - base[c]));
  const th = K.list ? K.list().slice(-1)[0] : null;
  return {worst, wc, flick, ttm: Math.round(fm - call), variant: th && (th.variant || (th.variants || []).join('>')), outcome: job.outcome};
}
const res = {};
for (const name of ['muse', 'fox']) {
  const agg = {before: [], after: []};
  for (const seed of [11, 23, 37, 41, 59]) { agg.before.push(run(name, 'current', seed)); agg.after.push(run(name, 'semantic', seed)); }
  const mx = (a, k) => Math.max(...a.map(x => x[k])), md = (a, k) => { const v = a.map(x => x[k]).sort((p, q) => p - q); return v[v.length >> 1]; };
  const B = agg.before, A = agg.after;
  console.log(`  ${name}: BEFORE worst step ${mx(B, 'worst').toFixed(3)} (${B.find(x => x.worst === mx(B, 'worst')).wc}), thinking-paw excursion ${mx(B, 'flick').toFixed(2)}, call->first motion ${md(B, 'ttm')} ms [${B.map(x => x.variant).join(', ')}]`);
  console.log(`  ${name}: AFTER  worst step ${mx(A, 'worst').toFixed(3)} (${A.find(x => x.worst === mx(A, 'worst')).wc}), thinking-paw excursion ${mx(A, 'flick').toFixed(2)}, call->first motion ${md(A, 'ttm')} ms [${A.map(x => x.variant).join(', ')}]`);
  res[name] = {before: {worst: mx(B, 'worst'), flick: mx(B, 'flick'), ttm: md(B, 'ttm')}, after: {worst: mx(A, 'worst'), flick: mx(A, 'flick'), ttm: md(A, 'ttm')}};
  check(`${name}: no thinking-paw flick before a short semantic plan (excursion ${mx(A, 'flick').toFixed(2)} vs ${mx(B, 'flick').toFixed(2)}), worst step no worse (${mx(A, 'worst').toFixed(3)} vs ${mx(B, 'worst').toFixed(3)}), moves sooner (${md(A, 'ttm')} vs ${md(B, 'ttm')} ms), all done`,
    mx(A, 'flick') < 0.12 && mx(A, 'worst') <= mx(B, 'worst') + 0.005 && md(A, 'ttm') < md(B, 'ttm') - 300 && A.every(x => x.outcome === 'done'));
}
if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(res, null, 1));
console.log(`sem_page_handover_test: ${passes} passed, ${fails} failed`);
process.exit(fails);
