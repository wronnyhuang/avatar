// in learned mode the page's reflexes + commands still fire at once (they bypass the planner): frames from the
// event to the first visible entry in the page's affine map (motion.js learnedParts), which the hub applies on the
// SAME tick (react_worker merge). Plus: rules output unchanged by learned mode, hold/release ease in/out.
//   node tests/react_reflex_test.cjs
'use strict';
const M = require('./load.cjs')('app/motion.js');
const DT = 1 / 30, P = M.RIG_PARAMS, I = n => P.indexOf(n);
let fails = 0;
const check = (name, ok, det) => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (det ? '  — ' + det : '')); if (!ok) fails++; };
const face = (t) => ({t, face: {present: true, x: 0.05, y: 0.05, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}}, attending: true, attention: 0.9, motion: 0.05, hands: []});
function mk(seed = 4) { const m = M.createMotion({seed, learned: true, preset: 'default'}); for (let k = 0; k < 60; k++) m.step(DT, {features: face(k * DT)}); return m; }
const entry = (lp, name) => { const e = lp.ab.find(x => x[0] === I(name)); return e ? {a: e[1], b: e[2]} : {a: 1, b: 0}; };
function framesUntil(m, fire, pred, max = 40) {
  fire(m);
  for (let k = 0; k <= max; k++) { m.step(DT, {features: face(2 + k * DT)}); if (pred(m.learnedParts(P))) return k + 1; }
  return null;
}
const res = {};
res.startle_blink = framesUntil(mk(), m => m.event({type: 'approach_fast', rate: 1.2, t: 1}), lp => lp.rb > 0.3);
res.startle_head = framesUntil(mk(), m => m.event({type: 'approach_fast', rate: 1.2, t: 1}), lp => entry(lp, 'head_pitch').b < -0.01 || entry(lp, 'head_z').b < -0.005);
res.startle_eyes = framesUntil(mk(), m => m.event({type: 'approach_fast', rate: 1.2, t: 1}), lp => entry(lp, 'eye_wide').b > 0.1);
res.gaze_snap_point = framesUntil(mk(), m => m.event({type: 'point', dir: {x: 0.8, y: 0}, t: 1}), lp => entry(lp, 'gaze_x').a < 0.5);
res.gaze_snap_lookback = framesUntil(mk(), m => m.event({type: 'look_back', away_s: 3, t: 1}), lp => entry(lp, 'gaze_x').a < 0.5 || entry(lp, 'brow_l').b > 0.05);
res.face_appear_perk = framesUntil(mk(), m => m.event({type: 'face_appear', t: 1}), lp => entry(lp, 'eye_wide').b > 0.05 || entry(lp, 'ear_l').b > 0.05);
res.blink_cmd = framesUntil(mk(), m => m.command({action: 'gesture', name: 'blink'}), lp => lp.rb > 0.3);
res.wave_cmd = framesUntil(mk(), m => m.command({action: 'gesture', name: 'wave', side: 'left'}), lp => entry(lp, 'hand_r_y').a < 0.999);   // present (its own min-jerk attack, as in rules)
res.look_cmd = framesUntil(mk(), m => m.command({action: 'look', target: 'left'}), lp => entry(lp, 'gaze_x').a < 0.5);
// nod-back: the page decides after a human-like 0.2-0.35 s beat (rules design); from that decision it is immediate
{ const m = mk(); m.event({type: 'nod', t: 1}); let dec = null, vis = null;
  for (let k = 0; k < 40 && vis === null; k++) { m.step(DT, {features: face(2 + k * DT)}); if (dec === null && m.state.gestures.some(g => g.name === 'nod' && g.src === 'reflex')) dec = k; if (dec !== null && entry(m.learnedParts(P), 'head_pitch').b > 1e-4) vis = k; }
  res.nod_back = vis === null ? null : vis - dec + 1; res.nod_back_beat_s = dec === null ? null : +((dec + 1) * DT).toFixed(3); }
for (const [k, v] of Object.entries(res)) if (k !== 'nod_back_beat_s') check(`${k}: in the affine map within 2 frames of the page's decision`, v !== null && v <= 2, `${v} frame(s)`);
console.log('nod-back beat (rules design, unchanged):', res.nod_back_beat_s, 's');
// the same gestures in the RULES vector start on the same frame (learned adds no frames to the reflex path)
{ const r = M.createMotion({seed: 4, preset: 'default'}); for (let k = 0; k < 60; k++) r.step(DT, {features: face(k * DT)});
  const base = r.step(DT, {features: face(2)}).hand_r_y; r.command({action: 'gesture', name: 'wave', side: 'left'}); let k0 = null;
  for (let k = 0; k < 10 && k0 === null; k++) if (Math.abs(r.step(DT, {features: face(2 + (k + 1) * DT)}).hand_r_y - base) > 1e-4) k0 = k + 1;
  check('rules: the wave starts on the same frame as in the learned map', k0 === res.wave_cmd, `rules ${k0} vs learned ${res.wave_cmd}`); }
// hold: alpha eases 1 -> 0 to the target, release eases back to 1 (never a snap)
{ const m = mk(); m.command({action: 'pose', targets: {head_yaw: 0.3}, hold: true}); const al = [];
  for (let k = 0; k < 30; k++) { m.step(DT, {features: face(2 + k * DT)}); al.push(entry(m.learnedParts(P), 'head_yaw')); }
  const last = al[al.length - 1];
  m.command({action: 'release', parts: 'all'}); const rel = [];
  for (let k = 0; k < 30; k++) { m.step(DT, {features: face(3 + k * DT)}); rel.push(entry(m.learnedParts(P), 'head_yaw').a); }
  const mono = al.every((x, i) => i === 0 || x.a <= al[i - 1].a + 1e-9);
  check('held pose: alpha eases to 0 with beta -> target (command keeps priority over the planner)', mono && last.a < 1e-3 && Math.abs(last.b - 0.3) < 1e-3 && al[0].a > 0.5, `alpha ${al.slice(0, 4).map(x => x.a.toFixed(2)).join(',')}.. ${last.a.toFixed(3)} beta ${last.b.toFixed(3)}`);
  check('release: alpha eases back to 1', rel[0] < 0.5 && rel[rel.length - 1] > 0.999 && rel.every((x, i) => i === 0 || x >= rel[i - 1] - 1e-9), rel.slice(0, 5).map(x => x.toFixed(2)).join(',') + '..'); }
// quiet: no reflex, no command -> (almost) nothing in the map, the planner owns everything it covers
{ const m = mk(); const n = []; for (let k = 0; k < 90; k++) { m.step(DT, {features: face(2 + k * DT)}); n.push(m.learnedParts(P).ab.length); }
  check('quiet: the page map stays empty (the learned layers own the slow motion)', Math.max(...n.slice(30)) === 0, `max entries ${Math.max(...n.slice(30))}`); }
// rules output identical whether learned is on or off
{ const a = M.createMotion({seed: 7, preset: 'default'}), b = M.createMotion({seed: 7, preset: 'default', learned: true}); let same = true;
  for (let k = 0; k < 400; k++) { const ev = k === 100 ? [{type: 'approach_fast', t: k}] : k === 200 ? [{type: 'wave', side: 'r', t: k}] : [];
    const oa = a.step(DT, {features: face(k * DT), events: ev}), ob = b.step(DT, {features: face(k * DT), events: ev});
    if (JSON.stringify(a.vector(oa)) !== JSON.stringify(b.vector(ob))) same = false; }
  check('rules vector identical with learned on/off (same seed, reflex events)', same); }
console.log(fails ? fails + ' FAILED' : 'ALL PASS');
process.exitCode = fails ? 1 : 0;
