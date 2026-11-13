// node tests/muse_test.cjs — the muse (scope change): the headline sequence step by step, its reverse,
// three other orders / variants, a mid-drag cancellation; checks every command result, the world state after each,
// the action events, smoothness (no pops), planted feet that do not slide, props never floating / duplicated.
// MUSE_BODY=… MUSE_PROPS=… point it at the real renderer/animals/muse.{body,props}.json. Exit code = failures.
'use strict';
const {makeMuse} = require('./muse_lib.cjs');
let fails = 0; const out = [];
const T00 = Date.now(); const check = (name, ok, detail) => { const l = (ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  — ' + detail : ''); out.push(l); if (!ok) fails++; if (process.env.MUSE_PROGRESS) console.error(((Date.now() - T00) / 1000).toFixed(1) + 's ' + l.slice(0, 160)); };
const O = (verb, name, extra = {}) => Object.assign({action: 'object', verb, object: name ? {name} : undefined}, extra);
const LIFE = /^(blink_|ear_flick|nose_wiggle|nostril|gaze_|breath$|brow|squint|smile|eye_wide|mouth_open|ear_|tail_|head_[xyz]$)/;
const PER = p => /^root_(yaw|pitch|roll)$|^prop\d_r[xyz]$/.test(p);
const VMAX = {hand: 6.5, root_x: 1.6, root_z: 1.6, root_y: 1.5, root_yaw: 1.6, foot: 2.6, body: 3.5, head: 4.5, prop_xyz: 9, prop_r: 4, w: 9.5, other: 8};
function vlim(p) { if (/^hand_.*_(x|y|z)$/.test(p)) return VMAX.hand; if (/^foot_/.test(p)) return VMAX.foot; if (/^body_/.test(p)) return VMAX.body; if (/^head_(yaw|pitch|roll)$/.test(p)) return VMAX.head;
  if (/^prop\d_[xyz]$/.test(p)) return VMAX.prop_xyz; if (/^prop\d_r/.test(p)) return VMAX.prop_r; if (/^prop\d_(hl|hr|hh|s)$|^(gaze_free|cam_zoom)$/.test(p)) return VMAX.w; return VMAX[p] || VMAX.other; }
function smooth(T, label, from = 0) {
  const R = T.rows, P = T.m.allParams; let worst = null;
  for (let i = Math.max(2, from); i < R.length; i++) for (const p of P) {
    if (LIFE.test(p)) continue;
    const pk = /^prop(\d)_(x|y|z|rx|ry|rz)$/.exec(p);
    if (pk) { const q1 = R[i]._props.find(q => q.k === +pk[1]), q0 = R[i - 1]._props.find(q => q.k === +pk[1]), q00 = R[i - 2]._props.find(q => q.k === +pk[1]);
      if (q1 && q0 && q00 && (Math.abs(q1.att - q0.att) + Math.abs(q1.hh - q0.hh) > 1e-6 || Math.abs(q0.att - q00.att) + Math.abs(q0.hh - q00.hh) > 1e-6)) continue; }
    let a = R[i - 2][p] || 0, b = R[i - 1][p] || 0, c = R[i][p] || 0;
    if (PER(p)) { b = a + (b - a - Math.round(b - a)); c = b + (c - b - Math.round(c - b)); }
    const v = Math.abs(c - b) * 30, r = v / vlim(p);
    if (!worst || r > worst.r) worst = {p, v, r, t: R[i].t};
  }
  check(`${label}: no pops (every non-life param under its velocity bound)`, worst.r <= 1, `worst ${worst.p} ${worst.v.toFixed(2)}/s at ${worst.t.toFixed(1)} s (${(worst.r * 100).toFixed(0)}% of bound)`);
}
function feet(T, label, from = 0) {
  let slide = 0, at = 0, below = 0;
  const R = T.rows;
  for (let i = Math.max(1, from); i < R.length; i++) {
    const a = R[i - 1], b = R[i]; if (!a._feet || !b._feet || b._ride > 0.01 || a._ride > 0.01) continue;
    for (const s of ['l', 'r']) {
      if (b._swing.includes(s) || a._swing.includes(s)) continue;
      const d = Math.hypot(b._feet[s][0] - a._feet[s][0], b._feet[s][2] - a._feet[s][2]);
      if (d > slide) { slide = d; at = b.t; }
      below = Math.min(below, b._feet[s][1]);
    }
  }
  check(`${label}: planted feet do not slide (≤ 3 mm/frame) and never go under the floor`, slide <= 0.003 && below >= -0.005, `max slide ${(slide * 1000).toFixed(1)} mH/frame at ${at.toFixed(1)} s, lowest foot ${below.toFixed(3)}`);
}
function props(T, label, from = 0) {
  const E = T.m.engine.E; let bad = '';
  for (let i = from; i < T.rows.length && !bad; i++) for (const q of T.rows[i]._props) {
    if (q.held && q.worn) bad = `${q.id} held and worn at ${T.rows[i].t.toFixed(1)}`;
    if (q.att + q.hh > 1.001) bad = `${q.id} attach ${q.att + q.hh}`;
  }
  // at rest: every free prop sits on a surface
  for (const p of E.P) {
    if (p.held || p.worn || p.F.furniture) continue;
    const pl = E.places.find(x => x.id === p.place) || E.places.find(x => x.id === 'floor');
    const want = (p.where === 'home' ? p.home[1] : pl.top_y + p.restH);
    if (Math.abs(p.pos[1] - want) > 0.01) bad = bad || `${p.id} floating at y ${p.pos[1].toFixed(3)} (rest ${want.toFixed(3)})`;
  }
  check(`${label}: props consistent (never held+worn, weights ≤ 1, free props rest on a surface)`, !bad, bad);
}
function run(T, cmds, label, opt = {}) {
  const res = [];
  for (const c of cmds) {
    const r = T.m.command(c, {src: 'test'}); res.push(r);
    if (!opt.batch) { if (opt.watch) { for (let i = 0; i < 1800; i++) { T.step(1); opt.watch(T); const E = T.m.engine.E; if (!E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue && i > 3) break; } } else T.runUntilIdle(60); if (opt.after) opt.after(c, T); }
  }
  if (opt.batch) T.runUntilIdle(120);
  T.step(opt.tail ?? 45);
  const bad = res.map((r, i) => r.ok ? null : `${cmds[i].verb || cmds[i].name}: ${r.reason}`).filter(Boolean);
  check(`${label}: every command accepted`, !bad.length, bad.join('; ') || res.map(r => r.did).join(' | ').slice(0, 300));
  return res;
}
const st = T => T.m.state();
const wrap = x => x - Math.round(x);
// ---------------------------------------------------------------- 1. the headline, step by step
const T = makeMuse(11);
T.step(10);
run(T, [O('pick_up', 'headphones'), O('put_on', 'headphones'), O('pull', 'chair', {target: 'me'}), O('pull', 'desk', {target: 'me'}),
  O('pull', 'chair', {target: {object: {name: 'desk'}}}), O('sit_in', 'chair'), O('pick_up', 'laptop'), O('slide', 'laptop', {target: 'closer'}),
  O('open', 'laptop'), O('type_on', 'laptop')], 'headline', {tail: 150});
let s = st(T);
check('headline: wearing the headphones, sitting at the desk, laptop open, typing', s.wearing.includes('headphones') && s.sittingOn === 'chair' && s.atDesk && s.open.includes('laptop') && s.activity === 'typing', JSON.stringify({w: s.wearing, sit: s.sittingOn, desk: s.atDesk, open: s.open, act: s.activity}));
const typingRows = T.rows.slice(-120), tapL = typingRows.map(r => r.hand_l_y), sd = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
check('headline: typing = small paw taps on the keys, eyes on the screen (gaze_free up)', sd(tapL) > 0.004 && sd(tapL) < 0.08 && typingRows.filter(r => r.gaze_free > 0.5).length > 60, `paw-y sd ${sd(tapL).toFixed(4)}, eyes-on-screen frames ${typingRows.filter(r => r.gaze_free > 0.5).length}/120`);
const n0 = T.rows.length;
// (c2) the seated turn back: the face (hips' root_yaw + 0.4·body_yaw + head_yaw, in turns) within ~30° of him by the time
// the command reports done (a keyframe taken then shows the face, not the profile), and still after
const faceYaw = r => (r.root_yaw || 0) + (0.4 * (r.body_yaw || 0) + (r.head_yaw || 0)) / (2 * Math.PI);
const FRONT = 30 / 360;
let tbAtEnd = null;
{ const r = T.m.command(O('turn_back'), {src: 'test'}); const k0 = T.ev.length;
  for (let i = 0; i < 300; i++) { T.step(1); if (T.ev.slice(k0).some(e => e.kind === 'object' && e.phase === 'end')) { tbAtEnd = Object.assign({dt: T.m.now()}, T.rows[T.rows.length - 1]); break; } }
  check('turn back: every command accepted', r.ok, r.did || r.reason); T.step(60); }
s = st(T);
const lr = T.rows[T.rows.length - 1];
check('turn back: upper body + head turned to him, still seated, typing paused, eyes on him', s.turnedBack && s.sittingOn === 'chair' && /paused/.test(s.activity) && lr.gaze_free < 0.3 && Math.abs(lr.body_yaw) > 0.3, `body_yaw ${lr.body_yaw.toFixed(2)} head_yaw ${lr.head_yaw.toFixed(2)} gaze_free ${lr.gaze_free.toFixed(2)} act ${s.activity}`);
void n0;
check('turn back: state().facing reflects it (facing him, turnedBack)', s.facing === 'camera' && s.turnedBack === true, `facing ${s.facing}, turnedBack ${s.turnedBack}, body facing ${s.facingBody}`);
check('seated turn back faces him: root+waist+head yaw within 30° of the camera when the command reports done, and after; hips still at the desk (profile ≥ 25° off)',
  !!tbAtEnd && Math.abs(faceYaw(tbAtEnd)) <= FRONT && Math.abs(faceYaw(lr)) <= FRONT && Math.abs(lr.root_yaw) * 360 >= 25 && s.sittingOn === 'chair',
  `at done: root ${tbAtEnd && (tbAtEnd.root_yaw * 360).toFixed(0)}° waist ${tbAtEnd && (0.4 * tbAtEnd.body_yaw * 57.3).toFixed(0)}° head ${tbAtEnd && (tbAtEnd.head_yaw * 57.3).toFixed(0)}° → face ${tbAtEnd && (faceYaw(tbAtEnd) * 360).toFixed(1)}°; settled face ${(faceYaw(lr) * 360).toFixed(1)}°`);
{ // while typing (before the turn back) it faced the laptop, not him: facing must not claim 'camera' then
  // (over the last 4 s of typing, not one frame — the typing activity glances at him every 4–8 s for ~1 s by
  // design, and a changed job sequence can land the single sampled frame inside a glance)
  const away = typingRows.filter(r => Math.abs(faceYaw(r)) > FRONT).length, typ = T.rows[n0 - 1];
  check('typing at the desk: the face is toward the laptop (state says not camera)', away > typingRows.length * 0.6, `face off the camera in ${away}/${typingRows.length} typing frames (last frame ${(faceYaw(typ) * 360).toFixed(1)}°)`); }
// "okay, relax" after the turn back: back to the laptop smoothly, the paused typing resumes; then turn back again
{ const k = T.rows.length; T.m.command({action: 'release'}, {src: 'test'}); T.step(75); const sR = st(T), rR = T.rows[T.rows.length - 1];
  check('release after the turn back: turns back to the laptop (typing again, face off him), smoothly', sR.activity === 'typing' && !sR.turnedBack && sR.facing !== 'camera' && Math.abs(faceYaw(rR)) > FRONT, `act ${sR.activity}, facing ${sR.facing}, face ${(faceYaw(rR) * 360).toFixed(1)}°`);
  smooth(T, 'release back to the laptop', k);
  const r2 = T.m.command(O('turn_back'), {src: 'test'}); T.runUntilIdle(10); T.step(45); const s2 = st(T);
  check('turn back again after relaxing: facing him', r2.ok && s2.facing === 'camera' && Math.abs(faceYaw(T.rows[T.rows.length - 1])) <= FRONT, `facing ${s2.facing}`); s = s2; }
smooth(T, 'headline'); feet(T, 'headline'); props(T, 'headline');
const evs = T.ev.filter(e => e.kind === 'object' && e.phase === 'end').map(e => e.did);
check('headline: action events tell what happened, in order', evs.length >= 10 && /put on the (\w+ )?headphones/.test(evs.join('|')) && /pulled the chair up to the desk/.test(evs.join('|')) && /opened the (\w+ )?laptop/.test(evs.join('|')), evs.slice(0, 12).join(' | ').slice(0, 400));
// ---------------------------------------------------------------- 2. the reverse
const r0 = T.rows.length;
const gripErr = {max: 0, frames: 0};
run(T, [O('take_off', 'headphones'), O('stand_up'), O('push', 'chair', {target: 'back'}), O('close', 'laptop')], 'reverse', {watch: (TT) => { const E = TT.m.engine.E; if (E.move && E.move.name === 'drag') { gripErr.frames++; gripErr.max = Math.max(gripErr.max, E.paw.l.err || 0, E.paw.r.err || 0); } }});
check('reverse: the push back runs from a stance where both paws really hold the chair handles (reach-checked grip)', gripErr.frames > 10 && gripErr.max <= 0.03, `${gripErr.frames} drag frames, worst paw-to-handle error ${gripErr.max.toFixed(4)} H`);
s = st(T);
check('reverse: headphones off (put down somewhere), standing, chair back from the desk, laptop closed', !s.wearing.length && s.posture === 'standing' && !s.chairAtDesk && !s.open.includes('laptop') && !s.activity, JSON.stringify({w: s.wearing, p: s.posture, chairAtDesk: s.chairAtDesk, open: s.open, hp: s.props.find(p => p.id === 'headphones').where}));
smooth(T, 'reverse', r0); feet(T, 'reverse', r0); props(T, 'reverse', r0);
// ---------------------------------------------------------------- 2b. (c2) the FULL reverse: headline forward, then everything undone —
// the headphones back on their stand (band on the saddle = the props JSON home pose), the laptop closed and put back
{ const R = makeMuse(13); R.step(10);
  run(R, [O('pick_up', 'headphones'), O('put_on', 'headphones'), O('pull', 'chair', {target: 'me'}), O('pull', 'desk', {target: 'me'}),
    O('pull', 'chair', {target: {object: {name: 'desk'}}}), O('sit_in', 'chair'), O('pick_up', 'laptop'), O('slide', 'laptop', {target: 'closer'}),
    O('open', 'laptop'), O('type_on', 'laptop'), O('turn_back')], 'full reverse — forward part', {tail: 30});
  const k0 = R.rows.length, e0 = R.ev.length;
  const res = run(R, [O('take_off', 'headphones'), O('put_back', 'headphones'), O('close', 'laptop'), O('put_back', 'laptop'), O('push', 'chair', {target: 'back'}), O('push', 'desk', {target: 'back'})], 'full reverse', {tail: 45});
  const E = R.m.engine.E, hp = E.P.find(p => p.id === 'headphones'), lap = E.P.find(p => p.id === 'laptop'), sF = st(R);
  const band = (hp.def.handlesFull || []).find(h => h.id === 'band'), yw = (hp.homeRot[1] + hp.rot[1]) * 2 * Math.PI;
  const bandW = band ? [hp.pos[0] + Math.cos(yw) * band.local[0] + Math.sin(yw) * band.local[2], hp.pos[1] + band.local[1], hp.pos[2] - Math.sin(yw) * band.local[0] + Math.cos(yw) * band.local[2]] : null;
  const bandErr = band && band.world_home ? Math.hypot(...bandW.map((v, i) => v - band.world_home[i])) : 0;
  check('full reverse: headphones hang on their stand again (home pose: pos, rot, the band on the saddle), nothing held',
    hp.where === 'home' && hp.place === 'side_table' && !hp.held && !hp.worn && Math.hypot(...hp.pos.map((v, i) => v - hp.home[i])) < 0.005 && Math.abs(wrap(hp.rot[1])) < 0.003 && bandErr < 0.01 && !sF.holding.left && !sF.holding.right,
    `where ${hp.where}/${hp.place}, pos err ${Math.hypot(...hp.pos.map((v, i) => v - hp.home[i])).toFixed(4)}, rot ${hp.rot.map(v => v.toFixed(3))}, band→saddle ${bandErr.toFixed(4)} H; did: ${res[1].did}`);
  check('full reverse: "put the laptop back" = closed, back at its home spot on the floor by the desk, and did says so',
    lap.where === 'home' && Math.hypot(lap.pos[0] - lap.home[0], lap.pos[2] - lap.home[2]) < 0.02 && (lap.s || 0) < 0.01 && /home spot on the floor/.test(res[3].did) && !lap.held,
    `where ${lap.where}, pos err ${Math.hypot(lap.pos[0] - lap.home[0], lap.pos[2] - lap.home[2]).toFixed(4)}, lid ${lap.s}; did: ${res[3].did}`);
  const ends = R.ev.slice(e0).filter(e => e.kind === 'object' && (e.phase === 'end' || e.phase === 'fail'));
  check('full reverse: every job ended (none failed), standing, chair away from the desk', ends.length === 6 && ends.every(e => e.phase === 'end') && sF.posture === 'standing' && !sF.chairAtDesk && !sF.wearing.length,
    ends.map(e => e.phase + ':' + (e.did || e.reason)).join(' | ').slice(0, 500));
  smooth(R, 'full reverse', k0); feet(R, 'full reverse', k0); props(R, 'full reverse', k0); }
// the other phrasings from the seat, holding the headphones it just took off: put them away / down where they were / return (no object)
{ const res = [];
  for (const [i, c] of [O('put_away', 'headphones'), O('put_down', 'headphones', {place: 'where they were'}), {action: 'object', verb: 'return'}].entries()) {
    const V = makeMuse(90 + i); V.step(10);
    for (const c0 of [O('put_on', 'headphones'), O('pull', 'chair', {target: {object: {name: 'desk'}}}), O('sit_in', 'chair'), O('take_off', 'headphones')]) { V.m.command(c0, {src: 'test'}); V.runUntilIdle(90); }
    const held0 = !!V.m.state().holding.left || !!V.m.state().holding.right, r = V.m.command(c, {src: 'test'}); V.runUntilIdle(90); V.step(20);
    const hp = V.m.engine.E.P.find(p => p.id === 'headphones'), sV = V.m.state();
    res.push({ok: held0 && r.ok && hp.where === 'home' && !hp.held && !sV.holding.left && !sV.holding.right, d: `${c.verb}${c.place ? ' ' + c.place : ''}: ${r.ok ? r.did : r.reason} → ${hp.where}`});
  }
  check('put them away / down where they were / return it (seated, holding them): back on the stand, nothing left held', res.every(x => x.ok), res.map(x => x.d).join(' | ').slice(0, 600)); }
// ---------------------------------------------------------------- 3. variant A: one request, three commands at once
const A = makeMuse(21); A.step(10);
run(A, [O('put_on', 'headphones'), O('sit_at', 'laptop'), O('work_on', 'laptop')], 'variant A (one-shot "put on your headphones and sit down at your laptop and get to work")', {batch: true, tail: 120});
s = st(A);
check('variant A: wearing headphones, seated at the desk, laptop open, typing', s.wearing.includes('headphones') && s.sittingOn === 'chair' && s.atDesk && s.activity === 'typing', JSON.stringify({w: s.wearing, sit: s.sittingOn, desk: s.atDesk, act: s.activity, open: s.open}));
smooth(A, 'variant A'); feet(A, 'variant A'); props(A, 'variant A');
// ---------------------------------------------------------------- 4. variant B: other order (sit first, then the desk, then work)
const Bm = makeMuse(31); Bm.step(10);
run(Bm, [{action: 'move', name: 'sit'}, O('pull', 'desk', {target: 'me'}), O('work_on', 'laptop'), O('put_on', 'headphones')], 'variant B (sit → pull the desk → work → headphones)', {tail: 90});
s = st(Bm);
check('variant B: ends seated at the desk typing... then (headphones last) wearing them; typing resumed or stopped truthfully', s.wearing.includes('headphones') && (s.sittingOn === null || s.sittingOn === 'chair'), JSON.stringify({w: s.wearing, sit: s.sittingOn, desk: s.atDesk, act: s.activity}));
smooth(Bm, 'variant B'); feet(Bm, 'variant B'); props(Bm, 'variant B');
// ---------------------------------------------------------------- 5. variant C: the laptop on the floor first, then "open the laptop and work"
const C = makeMuse(41); C.step(10);
run(C, [O('place', 'laptop', {place: 'floor'}), O('open', 'laptop'), O('work_on', 'laptop'), O('stop_typing'), O('put_back', 'laptop')], 'variant C (laptop to the floor, open it, work, stop, put it back)', {tail: 60});
s = st(C);
const lapP = C.m.engine.E.P.find(p => p.id === 'laptop');
check('variant C: laptop back home (lid state kept), not typing', Math.hypot(lapP.pos[0] - lapP.home[0], lapP.pos[2] - lapP.home[2]) < 0.02 && !s.activity, JSON.stringify({lap: s.props.find(p => p.id === 'laptop'), act: s.activity, open: s.open}));
smooth(C, 'variant C'); feet(C, 'variant C'); props(C, 'variant C');
// ---------------------------------------------------------------- 6. cancellation mid-drag
const D = makeMuse(51); D.step(10);
D.m.command(O('pull', 'chair', {target: {object: {name: 'desk'}}}), {src: 'test'});
for (let i = 0; i < 30 * 30 && !(D.m.engine.E.move && D.m.engine.E.move.drag); i++) D.step(1);
D.step(30);
const mid = D.m.engine.E.P.find(p => p.id === 'chair').pos.slice();
const rs = D.m.command({action: 'move', name: 'stop'}, {src: 'test'});
D.runUntilIdle(20); D.step(30);
s = st(D);
const paws = ['l', 'r'].map(x => D.m.engine.E.paw[x]);
check('cancel mid-drag: stops gracefully, lets go of the chair, chair left where it got to, job reported interrupted', rs.ok && paws.every(p => !p.keep && p.ctrl < 0.01) && D.ev.some(e => e.phase === 'interrupted' && e.kind === 'object') && Math.hypot(...[0, 2].map(k => D.m.engine.E.P.find(p => p.id === 'chair').pos[k] - mid[k])) < 0.3,
  `chair moved ${Math.hypot(...[0, 2].map(k => D.m.engine.E.P.find(p => p.id === 'chair').pos[k] - mid[k])).toFixed(2)} after stop; events: ${D.ev.filter(e => e.kind === 'object').map(e => e.phase + ':' + (e.did || e.reason)).join(' | ').slice(0, 260)}`);
smooth(D, 'cancel'); feet(D, 'cancel'); props(D, 'cancel');
// ---------------------------------------------------------------- 6b. variant D: ONE request "work on the laptop" from the start (lid closed, on the floor)
{ const W = makeMuse(81); W.step(10); const lapW = W.m.engine.E.P.find(p => p.id === 'laptop'); const s0 = lapW.s || 0;
  const r = W.m.command(O('work_on', 'laptop'), {src: 'test'}); let closedTyping = 0;
  for (let i = 0; i < 120 * 30; i++) { W.step(1); const E = W.m.engine.E; if (E.activity && !E.activity.paused && (lapW.s || 0) < 0.95) closedTyping++; if (!E.move && !E.chain && !E.jobs.length && i > 3) break; }
  W.step(60); const sD = W.m.state();
  const iOpen = W.ev.findIndex(e => e.kind === 'object_event' && e.name === 'opened'), iType = W.ev.findIndex(e => e.kind === 'activity' && e.name === 'typing' && e.phase === 'start');
  check('variant D (one "work on the laptop", lid closed): opens the lid with a paw BEFORE typing, never types on a closed lid', r.ok && s0 < 0.5 && /opening the (\w+ )?laptop/.test(r.did) && iOpen >= 0 && iType > iOpen && closedTyping === 0 && sD.open.includes('laptop') && sD.activity === 'typing' && lapW.s > 0.99,
    `lid s ${s0} → ${lapW.s}; opened event #${iOpen}, typing start #${iType}; frames typing on a closed lid ${closedTyping}; ${r.did.slice(0, 160)}`);
  smooth(W, 'variant D'); props(W, 'variant D'); }
// ---------------------------------------------------------------- 6c. the chair backed up against the desk: the grip search finds a stance that reaches both handles
{ const res = [];
  for (const [gap, ry] of [[0.05, 0], [0.1, 0.03], [0.0, -0.03]]) {
    const G = makeMuse(7); G.step(5); const E = G.m.engine.E, ch = E.P.find(p => p.id === 'chair'), dk = E.P.find(p => p.id === 'desk');
    const dyaw = dk.homeRot[1] + dk.rot[1], fr = [Math.sin(dyaw * 2 * Math.PI), 0, Math.cos(dyaw * 2 * Math.PI)], edge = [dk.pos[0] + fr[0] * 0.25, 0, dk.pos[2] + fr[2] * 0.25];
    ch.pos = [edge[0] + fr[0] * (0.22 + gap), ch.pos[1], edge[2] + fr[2] * (0.22 + gap)]; ch.rot = [0, dyaw + 0.5 + ry - ch.homeRot[1], 0];
    for (const c of [O('pull', 'chair', {target: 'me'}), O('push', 'chair', {target: 'back'})]) {
      let err = 0, n = 0; const n0 = G.ev.length, r = G.m.command(c, {src: 'test'});
      for (let i = 0; i < 1800; i++) { G.step(1, false); if (E.move && E.move.name === 'drag') { n++; err = Math.max(err, E.paw.l.err || 0, E.paw.r.err || 0); } if (!E.move && !E.chain && !E.jobs.length && i > 3) break; }
      const f = G.ev.slice(n0).find(e => e.kind === 'object' && (e.phase === 'fail' || e.phase === 'end'));
      res.push({ok: r.ok && f && f.phase === 'end' && (n === 0 || err <= 0.03), d: `${c.verb}@${gap}/${ry}: ${f ? f.phase : 'none'} ${n} drag frames err ${err.toFixed(3)}${f && f.phase === 'fail' ? ' ' + f.reason : ''}`});
    }
  }
  check('chair against the desk: pull / push it from a reach-checked grip (both handles, a stance with room)', res.every(x => x.ok), res.map(x => x.d).join(' | ')); }
// ---------------------------------------------------------------- 7. truthful failures
const F = makeMuse(61); F.step(5);
const f1 = F.m.command(O('pick_up', 'piano'), {src: 'test'}), f2 = F.m.command(O('pick_up', 'desk'), {src: 'test'}), f3 = F.m.command(O('take_off', 'headphones'), {src: 'test'});
check('failures are truthful: no such object / furniture is not picked up / not wearing it', !f1.ok && /no piano/.test(f1.reason) && !f2.ok && f3.ok && /not wearing/.test(f3.did), [f1.reason, f2.reason, f3.did].join(' | '));
// timing
// on a busy shared machine (load averages of 30+) the best of 3 identical runs is the
// code's own cost (contention only ever adds time); every run is listed
const runsUs = [];
for (let k = 0; k < 3; k++) { const TT = makeMuse(71); TT.step(5); TT.m.command(O('work_on', 'laptop'), {src: 'test'}); const tA = process.hrtime.bigint(); TT.step(900, false); const tB = process.hrtime.bigint(); runsUs.push(Number(tB - tA) / 900 / 1000); }
const us = Math.min(...runsUs);
check('step() cost while walking / dragging / typing ≤ 500 µs (best of 3 runs)', us <= 500, `${us.toFixed(0)} µs per frame (runs ${runsUs.map(x => x.toFixed(0)).join(' / ')} µs)`);
console.log(out.join('\n'));
console.log(`\n${out.length - fails}/${out.length} passed (files: ${T.files.src})`);
process.exit(fails);
