// node tests/reach_test.cjs — the per-body reach / midline / contact step (app/motion.js reachPass)
// on the real body files of the muse, a mascot (bonbon) and Juno (renderer/humans/junor): a clap has no one-frame seam at the end of
// its attack (paw acceleration bound), two paws sent past the midline end STACKED (one a paw's height over the other) instead of
// level, paws sent together meet at one height / depth, and reachFix:false leaves every paw number as it was. Exit code = failures.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js');
const ROOT = path.join(__dirname, '..');
let fails = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const BODY = {muse: 'renderer/animals/muse.body.json', bonbon: 'renderer/animals/bonbon.body.json', junor: 'renderer/humans/junor/junor.body.json'};
function mk(ch, reachFix) {
  const m = M.createMotion({seed: 7, character: ch, reachFix});
  const B = JSON.parse(fs.readFileSync(path.join(ROOT, BODY[ch]))), pp = path.join(ROOT, 'renderer/animals', ch + '.props.json');
  m.setProps(fs.existsSync(pp) ? JSON.parse(fs.readFileSync(pp)) : {}, B, ch);
  const rows = [], face = {present: true, x: 0.05, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
  return {m, rows, step(n) { for (let i = 0; i < n; i++) rows.push(Object.assign({}, m.step(1 / 30, {features: {t: m.now(), face, attending: true, hands: []}}))); }};
}
const H = ['hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z'];
function run(ch, rf, cmd, secs) { const t = mk(ch, rf); t.step(30); const n0 = t.rows.length; const r = t.m.command(cmd, {src: 'voice'}); t.step(Math.round(secs * 30)); return {r, rows: t.rows.slice(n0)}; }
const peakAcc = rows => { let a = 0; for (let i = 2; i < rows.length; i++) for (const p of H) a = Math.max(a, Math.abs(rows[i][p] - 2 * rows[i - 1][p] + rows[i - 2][p]) * 900); return a; };
const CROSS = {action: 'pose', name: 'arms_x', side: 'both', hold: true, ease_ms: 450, own: 'both', targets: {hand_l_x: -1, hand_l_y: 0.35, hand_l_z: 0.4, hand_r_x: -1, hand_r_y: 0.45, hand_r_z: 0.5}};
const TOGETHER = {action: 'pose', name: 'hands_together', side: 'both', hold: true, ease_ms: 450, own: 'both', targets: {hand_l_x: -0.75, hand_l_y: 0.45, hand_l_z: 0.7, hand_r_x: -0.8, hand_r_y: 0.55, hand_r_z: 0.8}};   // tools/kf_reach.py: paws together in front of the chest = x -0.75
for (const ch of ['muse', 'bonbon', 'junor']) {
  const on = run(ch, true, {action: 'gesture', name: 'clap', cycles: 4}, 3), off = run(ch, false, {action: 'gesture', name: 'clap', cycles: 4}, 3);
  const aOn = peakAcc(on.rows), aOff = peakAcc(off.rows);
  check(`${ch}: clap has no seam at the end of its attack (peak paw accel ${aOn.toFixed(0)} ≤ 150; off ${aOff.toFixed(0)})`, aOn <= 150);
  const minX = Math.min(...on.rows.map(o => o.hand_l_x));
  check(`${ch}: the clap's paws still reach the midline band (min hand_l_x ${minX.toFixed(2)})`, minX < (ch === 'junor' ? -0.38 : -0.75));
  const c = run(ch, true, CROSS, 4).rows.pop(), c0 = run(ch, false, CROSS, 4).rows.pop();
  check(`${ch}: both paws past the midline end stacked, not level (dy ${Math.abs(c.hand_l_y - c.hand_r_y).toFixed(2)} ≥ 0.12; off ${Math.abs(c0.hand_l_y - c0.hand_r_y).toFixed(2)})`, Math.abs(c.hand_l_y - c.hand_r_y) >= 0.12);
  check(`${ch}: reachFix:false leaves the crossing pose's numbers alone`, Math.abs(c0.hand_l_x + 1) < 0.02 && Math.abs(c0.hand_r_x + 1) < 0.02);
  if (ch !== 'junor') {   // on Juno -0.75 is past her midline (-0.39): a crossing, not paws together
    const g = run(ch, true, TOGETHER, 4).rows.pop();
    check(`${ch}: paws sent together meet at one height and depth (dy ${Math.abs(g.hand_l_y - g.hand_r_y).toFixed(3)}, dz ${Math.abs(g.hand_l_z - g.hand_r_z).toFixed(3)})`, Math.abs(g.hand_l_y - g.hand_r_y) < 0.02 && Math.abs(g.hand_l_z - g.hand_r_z) < 0.02);
  }
}
{ const a = run('muse', true, {action: 'gesture', name: 'laugh'}, 2).rows.pop(), b = run('muse', false, {action: 'gesture', name: 'laugh'}, 2).rows.pop();   // paws near (not at) the midline are not merged
  check(`muse: a laugh's paws by the mouth are left where the gesture puts them (dx ${Math.abs(a.hand_l_x - b.hand_l_x).toFixed(3)})`, Math.abs(a.hand_l_x - b.hand_l_x) < 0.01 && Math.abs(a.hand_l_y - b.hand_l_y) < 0.01); }
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails);
