// node tests/muse_moves_test.cjs — every whole-body move on the muse (real body/props JSON): accepted with a true `did`,
// ends where it should (posture, facing back to him unless held), no pops (per-frame velocity + acceleration bounds),
// planted feet do not slide, body never under the floor, eyes back on him 0.2–1.5 s after, a held prop stays in the
// paw through a spin, mid-move stop is graceful, step() cost. Exit code = failures.
'use strict';
const {makeMuse, Moves} = require('./muse_lib.cjs');
let fails = 0; const out = [];
const check = (n, ok, d) => { out.push((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const PER = p => /^root_(yaw|pitch|roll)$|^prop\d_r[xyz]$/.test(p);
const BODY = ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z', 'body_yaw', 'body_pitch', 'body_lean', 'hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z', 'head_yaw', 'head_pitch', 'cam_zoom', 'cam_x', 'cam_y'];
const VB = p => /^root_(x|z)$/.test(p) ? 1.8 : p === 'root_y' ? 5.5 : /^root_(yaw)$/.test(p) ? 2.6 : /^root_(pitch|roll)$/.test(p) ? 2.6 : p === 'squash' ? 14 : /^foot_/.test(p) ? 2.8 : /^body_/.test(p) ? 3.5 : /^hand_/.test(p) ? 6.5 : /^head_/.test(p) ? 4.5 : 3;
function rate(R, from, p, i) { let a = R[i - 2][p] || 0, b = R[i - 1][p] || 0, c = R[i][p] || 0; if (PER(p)) { b = a + (b - a - Math.round(b - a)); c = b + (c - b - Math.round(c - b)); } return [(c - b) * 30, (c - 2 * b + a) * 900]; }
// root_x/z while the body tips about a contact pivot (roll/pitch changing) is a rotation, judged by root_roll/pitch
function smooth(R, from, k = 1) { let w = {r: 0}; for (let i = Math.max(2, from); i < R.length; i++) { const piv = Math.abs(R[i].root_roll - R[i - 1].root_roll) + Math.abs(R[i].root_pitch - R[i - 1].root_pitch) > 0.002;
  for (const p of BODY) { if (piv && /^root_(x|z)$/.test(p)) continue; const [v, acc] = rate(R, from, p, i), r = Math.max(Math.abs(v) / (VB(p) * k), Math.abs(acc) / ((p === 'root_y' ? 160 : VB(p) * 40) * k)); if (r > w.r) w = {r, p, v, acc, t: R[i].t}; } } return w; }
function feet(R, from) { let slide = 0, low = 0; for (let i = Math.max(1, from); i < R.length; i++) { const a = R[i - 1], b = R[i]; if (!a._feet || b._ride > 0.01 || a._ride > 0.01) continue; for (const s of ['l', 'r']) { if (b._swing.includes(s) || a._swing.includes(s)) continue; slide = Math.max(slide, Math.hypot(b._feet[s][0] - a._feet[s][0], b._feet[s][2] - a._feet[s][2])); low = Math.min(low, b._feet[s][1]); } } return {slide, low}; }
// the body above the legs (belly bottom = the seat contact, body.json sitting.seat_contact_H) never goes under the floor
function floorPen(T, R, from) {
  const B = T.m.engine.body(), C = B.C, yb = 0.15, P = [];
  for (const x of [B.lo[0] + 0.06, B.hi[0] - 0.06]) for (const y of [yb, B.hi[1] - 0.05]) for (const z of [B.lo[2] + 0.06, B.hi[2] - 0.06]) P.push([x, y, z]);
  let pen = 0;
  for (let i = from; i < R.length; i++) { const r = R[i], M = Moves.rotm(r.root_yaw, r.root_pitch, r.root_roll);
    for (const p of P) { const d = [p[0] - C[0], p[1] - C[1], p[2] - C[2]], y = r.root_y + C[1] + M[1][0] * d[0] + M[1][1] * d[1] + M[1][2] * d[2]; pen = Math.max(pen, -y); } }
  return pen;
}
const MOVES = [
  ['spin', {count: 2, dir: 'left'}], ['spin', {count: 1, dir: 'right', speed: 'fast'}], ['twirl', {count: 3, speed: 'slow'}], ['turn', {dir: 'left'}],
  ['jump', {count: 2}], ['jump', {height: 1}], ['hop', {count: 3}], ['backflip', {}], ['frontflip', {}], ['walk', {count: 4}], ['walk', {count: 3, dir: 'back'}],
  ['walk', {count: 2, dir: 'screen_left'}], ['dance', {style: 'disco'}], ['dance', {style: 'robot'}], ['dance', {style: 'silly'}], ['dance', {style: 'happy', speed: 'fast'}],
  ['bow', {}], ['wiggle', {}], ['shake_off', {}], ['crouch', {}], ['sway', {}], ['shimmy', {}], ['cartwheel', {}], ['roll_over', {}],
  ['lie_down', {}], ['turn_away', {}], ['sit', {floor: true}]];
const tStep = [];
for (const [name, args] of MOVES) {
  const T = makeMuse(17); T.step(15); const n0 = T.rows.length;
  const r = T.m.command(Object.assign({action: 'move', name}, args), {src: 'test'});
  const a0 = process.hrtime.bigint(); const d = T.runUntilIdle(40); tStep.push(Number(process.hrtime.bigint() - a0) / 1000 / Math.max(1, d * 30));
  const tEnd = T.m.now(); T.step(60);
  const R = T.rows, w = smooth(R, n0, args.speed === 'fast' ? 1.3 : 1), f = feet(R, n0), pen = floorPen(T, R, n0), s = T.m.state();
  const endEv = T.ev.find(e => e.id === r.id && (e.phase === 'end' || e.phase === 'interrupted'));
  let back = null; for (let i = n0; i < R.length; i++) if (R[i].t >= tEnd - 0.4 && R[i].gaze_free < 0.3 && Math.abs(R[i].root_yaw) < 0.06) { back = R[i].t - tEnd; break; }
  const label = `${name} ${JSON.stringify(args)}`;
  const posOk = name === 'sit' ? s.posture === 'seated' : s.posture === 'standing';
  check(`${label}: accepted, true did, finished`, r.ok && /\w/.test(r.did) && !!endEv && r.did !== name, `${r.did || r.reason} → ${endEv ? endEv.did : 'no end event'} (${d.toFixed(1)} s)`);
  check(`${label}: ends ${name === 'sit' ? 'seated' : 'standing'}, facing him, eyes back ≤ 1.5 s after`, posOk && s.facing === 'camera' && back !== null && back <= 1.5, `posture ${s.posture}, facing ${s.facing}, gaze back at ${back === null ? 'never' : back.toFixed(2) + ' s'}`);
  check(`${label}: smooth (velocity + acceleration bounds)`, w.r <= 1, `worst ${w.p} v ${w.v && w.v.toFixed(2)} a ${w.acc && w.acc.toFixed(0)} (${(w.r * 100).toFixed(0)}%) at ${w.t && w.t.toFixed(2)}`);
  check(`${label}: planted feet stay put, nothing under the floor`, f.slide <= 0.003 && f.low >= -0.005 && pen <= 0.012, `slide ${(f.slide * 1000).toFixed(1)} mH/frame, lowest foot ${f.low.toFixed(3)}, body below floor ${pen.toFixed(3)}`);
}
// held prop through a spin; turn away held + gaze free; stop mid-backflip; refusal
{ const T = makeMuse(23); T.step(10); T.m.command({action: 'object', verb: 'pick_up', object: {name: 'ball'}}, {src: 'test'}); T.runUntilIdle(30);
  const n0 = T.rows.length; T.m.command({action: 'move', name: 'spin', count: 2}, {src: 'test'}); T.runUntilIdle(20); T.step(30);
  const att = T.rows.slice(n0).map(r => r._props.find(q => q.id === 'ball').att), s = T.m.state();
  check('a held ball stays in the paw through two spins', att.every(a => a > 0.99) && (s.holding.left === 'ball' || s.holding.right === 'ball'), `min attach ${Math.min(...att).toFixed(2)}`); }
{ const T = makeMuse(29); T.step(10); T.m.command({action: 'move', name: 'turn_away', hold: true}, {src: 'test'}); T.step(120);
  const s = T.m.state(), r = T.rows[T.rows.length - 1];
  check('turn away (hold): stays turned away, eyes free (not seeking him)', s.facing === 'away' && r.gaze_free > 0.7, `facing ${s.facing}, gaze_free ${r.gaze_free.toFixed(2)}`);
  const b = T.m.command({action: 'move', name: 'turn_back'}, {src: 'test'}); T.runUntilIdle(10); T.step(45);
  const s2 = T.m.state(), r2 = T.rows[T.rows.length - 1];
  check('turn back: faces him again, eyes on him', b.ok && s2.facing === 'camera' && r2.gaze_free < 0.3, `${b.did}; facing ${s2.facing}, gaze_free ${r2.gaze_free.toFixed(2)}`); }
{ const T = makeMuse(31); T.step(10); const n0 = T.rows.length; T.m.command({action: 'move', name: 'backflip', count: 2}, {src: 'test'}); T.step(20);
  const st = T.m.command({action: 'move', name: 'stop'}, {src: 'test'}); T.runUntilIdle(10); T.step(30);
  const w = smooth(T.rows, n0), ev = T.ev.find(e => e.kind === 'move' && e.phase === 'interrupted'), s = T.m.state(), pen = floorPen(T, T.rows, n0);
  check('stop mid-backflip: lands the flip it is in, upright, reported interrupted, smooth, not under the floor', st.ok && !!ev && s.posture === 'standing' && w.r <= 1 && pen <= 0.012, `${ev && ev.did}; worst ${w.p} ${(w.r * 100).toFixed(0)}%, posture ${s.posture}, pen ${pen.toFixed(3)}`); }
// "relax / okay / you can stop" (release) ends a RUNNING spin gracefully: it brakes, turns back to him, keeps what it holds
{ const T = makeMuse(17); T.step(15); T.m.command({action: 'object', verb: 'pick_up', object: {name: 'ball'}}, {src: 'test'}); T.runUntilIdle(30);
  const n0 = T.rows.length, e0 = T.ev.length; const r = T.m.command({action: 'move', name: 'spin', count: 3}, {src: 'test'}); T.step(40);
  const rel = T.m.command({action: 'release'}, {src: 'test'}); const tRel = T.m.now(); const d = T.runUntilIdle(20); T.step(45);
  const s = T.m.state(), ev = T.ev.slice(e0).find(e => e.id === r.id && e.kind === 'move' && (e.phase === 'interrupted' || e.phase === 'end')), w = smooth(T.rows, n0);
  check('release mid-spin: the spin ends (interrupted, ≤ 2.5 s), back to standing facing him, the ball still in its paw, smooth', rel.ok && !!ev && ev.phase === 'interrupted' && d <= 2.5 && s.posture === 'standing' && s.facing === 'camera' && !s.active.length && (s.holding.left === 'ball' || s.holding.right === 'ball') && w.r <= 1,
    `${ev ? ev.phase + ': ' + ev.did : 'no end event'}; ended ${d.toFixed(2)} s after release at ${tRel.toFixed(1)}; posture ${s.posture}, facing ${s.facing}, holding ${JSON.stringify(s.holding)}, worst ${w.p} ${(w.r * 100).toFixed(0)}%`);
  check('release mid-spin: did tells only what happened ("spun N of the 3 turns, then stopped …")', !!ev && /^spun [0-2] of the 3 turns, then stopped/.test(ev.did) && !/stopped early|turns done/.test(ev.did), ev && ev.did); }
// release a HELD move (turned away): it turns back to him; the move itself was complete, so did says so
{ const T = makeMuse(29); T.step(10); const r = T.m.command({action: 'move', name: 'turn_away', hold: true}, {src: 'test'}); T.step(120);
  T.m.command({action: 'release'}, {src: 'test'}); T.runUntilIdle(20); T.step(45);
  const s = T.m.state(), ev = T.ev.find(e => e.id === r.id && e.kind === 'move' && (e.phase === 'interrupted' || e.phase === 'end'));
  check('release a held turn-away: turns back to face him, did = turned away, held it, relaxed back (not "cut short")', s.facing === 'camera' && !s.active.length && !!ev && /held it/.test(ev.did) && !/cut short/.test(ev.did), `${ev && ev.did}; facing ${s.facing}`); }
// a stop mid-spin: the did names only the turns that happened, at several stop times
{ const dids = [];
  for (const f of [30, 50, 70]) { const T = makeMuse(17); T.step(15); const r = T.m.command({action: 'move', name: 'spin', count: 3}, {src: 'test'}); T.step(f); T.m.command({action: 'move', name: 'stop'}, {src: 'test'}); T.runUntilIdle(20);
    const ev = T.ev.find(e => e.id === r.id && e.kind === 'move' && (e.phase === 'interrupted' || e.phase === 'end')); dids.push(ev ? ev.did : 'none'); }
  check('stop mid-spin: did says "spun N of the 3 turns, then stopped" (or all 3 as it finished), never "3 turns … stopped early"', dids.every(d => /^spun ([0-2] of the 3 turns, then stopped$|all 3 of the 3 turns; the stop came as it was finishing$)|^started to spin \(3 turns asked\) but stopped before one was done$/.test(d)), dids.join(' | ')); }
{ const T = makeMuse(37); const r = T.m.command({action: 'move', name: 'moonwalk_on_the_ceiling'}, {src: 'test'});
  check('unknown move: refused truthfully', !r.ok && /don't know/.test(r.reason), r.reason); }
{ // timing without the test's own per-frame recording: the controller's step() alone, while moving
  tStep.length = 0; const T = makeMuse(41); T.step(10, false);
  for (const c of [{name: 'dance', style: 'disco'}, {name: 'walk', count: 6}, {name: 'spin', count: 2}, {name: 'backflip'}, {name: 'jump', count: 2}]) {
    T.m.command(Object.assign({action: 'move'}, c), {src: 'test'});
    for (let i = 0; i < 150; i++) { const a0 = process.hrtime.bigint(); T.step(1, false); tStep.push(Number(process.hrtime.bigint() - a0) / 1000); }
  }
  const I = makeMuse(43); I.step(10, false); const idle = []; for (let i = 0; i < 300; i++) { const a0 = process.hrtime.bigint(); I.step(1, false); idle.push(Number(process.hrtime.bigint() - a0) / 1000); }
  idle.sort((a, b) => a - b); out.push(`INFO idle muse step() median ${idle[150].toFixed(0)} µs (same box, same moment)`); }
tStep.sort((a, b) => a - b);
check('step() cost during moves ≤ 500 µs (median)', tStep[tStep.length >> 1] <= 500, `median ${tStep[tStep.length >> 1].toFixed(0)} µs, max ${tStep[tStep.length - 1].toFixed(0)} µs per frame`);
console.log(out.join('\n')); console.log(`\n${out.length - fails}/${out.length} passed`); process.exit(fails);
