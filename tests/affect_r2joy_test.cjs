// joy: app/affect.js (the feeling's face + posture state) inside motion.js — general shape checks over the whole
// valence / arousal / dominance space, the time course (onset, hold through speech, decay, residue), layering (lips stay
// lip sync's, a busy body shows the feeling in the face only) and that nothing changes without a verdict.
'use strict';
const load = require('./load.cjs');
const AffLib = load('app/affect.js'); globalThis.AvatarAffectLib = AffLib;
const {makeMuse} = require('./muse_lib.cjs');
let fails = 0;
const check = (name, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (d !== undefined ? '  — ' + d : '')); if (!ok) fails++; };

// 1. the shape over a grid: bounded, finite
let bad = 0;
for (const v of [-1, -0.5, 0, 0.5, 1]) for (const a of [-1, -0.5, 0, 0.5, 1]) for (const d of [-1, 0, 1]) for (const k of [0.2, 0.6, 1]) {
  const {face, add} = AffLib.shape(v, a, d, k, {});
  for (const x of Object.values(face).concat(Object.values(add))) if (!Number.isFinite(x) || Math.abs(x) > 1.2) bad++;
}
check('shape finite and bounded over the VAD grid', bad === 0, bad);
const sh = (v, a, d, k = 0.9, F = {}) => AffLib.shape(v, a, d, k, F);
const joy = sh(0.85, 0.7, 0.3), sad = sh(-0.75, -0.4, -0.4), ang = sh(-0.7, 0.7, 0.6), fear = sh(-0.7, 0.8, -0.7), sur = sh(0.1, 0.85, -0.1);
check('joy: big smile, brows up, chest up', joy.face.smile > 0.8 && joy.face.brow > 0.3 && joy.add.body_pitch < -0.1, JSON.stringify(joy.face));
check('joy: eyes not squeezed sleepy (squint <= 0.3)', joy.face.squint <= 0.3, joy.face.squint);
check('sad: no smile, knit + raised brows (inner ends up), head down, sink', sad.face.smile < 0.1 && sad.face.brow_knit > 0.4 && sad.face.brow > 0.1 && sad.add.head_pitch > 0.05 && sad.add.body_bounce < -0.05, JSON.stringify(sad));
check('anger: knit + LOWERED brows', ang.face.brow_knit > 0.4 && ang.face.brow < -0.2, ang.face.brow);
check('fear: wide eyes, shoulders up, arms drawn in', fear.face.eye_wide > 0.4 && fear.add.shoulder > 0.2 && fear.add.hand_x < 0, JSON.stringify(fear.add));
check('surprise: wide eyes, brows up, open silent mouth', sur.face.eye_wide > 0.5 && sur.face.brow > 0.3 && sur.face.mouth > 0.2);
const mild = sh(0.85, 0.7, 0.3, 0.25);
check('a mild feeling is a smaller version of the same shape', mild.face.smile < joy.face.smile && mild.face.smile > 0.3 && Math.abs(mild.add.body_pitch) < Math.abs(joy.add.body_pitch));

// 2. in the motion controller
function run(verdict, opts = {}) {
  const T = makeMuse(7);
  const {m} = T;
  for (let i = 0; i < 30; i++) T.step(1, false);
  const base = m.step(1 / 30, {features: {t: m.now(), face: null, hands: []}});
  m.affect(Object.assign({hold_s: 5}, verdict));
  const out = [];
  for (let i = 0; i < 30 * (opts.secs || 25); i++) {
    const speaking = opts.speakS ? i / 30 < opts.speakS : false;
    const o = m.step(1 / 30, {features: {t: m.now(), face: null, hands: []}, speaking, avatarVoice: speaking ? {rms: 0.05, voiced: true} : {rms: 0, voiced: false}});
    out.push(Object.assign({t: i / 30, st: m.affectState()}, o));
  }
  return {m, base, out};
}
const J = run({v: 0.85, a: 0.7, d: 0.3, intensity: 0.9, label: 'delighted'}, {speakS: 8});
const at = s => J.out[Math.round(s * 30)];
check('onset: the face is clearly there 0.6 s after the verdict', at(0.6).smile > 0.6 && at(0.6).st.w > 0.75, `smile ${at(0.6).smile.toFixed(2)} w ${at(0.6).st.w}`);
check('held through the reply (still on at 7.5 s while it talks)', at(7.5).st.phase === 'on' && at(7.5).smile > 0.7, at(7.5).st.phase);
check('decays after the reply (released by 10 s, weaker)', at(10.5).st.phase !== 'on' && at(10.5).st.w < 0.6, `${at(10.5).st.phase} ${at(10.5).st.w}`);
check('gone back to idle within ~25 s', ['idle', 'residue'].includes(at(24.9).st.phase) && at(24.9).st.w < 0.08, `${at(24.9).st.phase} ${at(24.9).st.w}`);
const bmax = Math.max(...J.out.slice(0, 60).map(o => o.body_bounce)), bmin = Math.min(...J.out.slice(0, 60).map(o => o.body_bounce));
check('joy onset accent: a lift with bounces (body_bounce swings)', bmax - bmin > 0.12, (bmax - bmin).toFixed(3));
check('lips stay lip sync\'s: no held mouth_open while it talks', Math.max(...J.out.slice(20, 200).map(o => o.mouth_open)) < 0.05);
const camMax = Math.max(...J.out.map(o => o.cam_zoom || 0)), camMin = Math.min(...J.out.slice(30, 200).map(o => o.cam_zoom));
check('strong feeling eases the camera in a little (home framing)', camMin < 0.2 && camMin > 0.0, `min ${camMin.toFixed(3)} max ${camMax.toFixed(3)}`);
const S2 = run({v: -0.75, a: -0.4, d: -0.4, intensity: 0.8, label: 'sad'}, {speakS: 6});
const s3 = S2.out[90];
check('sad in the controller: knit, head down, body sinks', s3.brow_knit > 0.35 && s3.head_pitch > S2.base.head_pitch + 0.05 && s3.body_bounce < -0.04, `knit ${s3.brow_knit.toFixed(2)} hp ${s3.head_pitch.toFixed(2)} bb ${s3.body_bounce.toFixed(3)}`);
// 3. mild feelings stay natural: no camera move, small posture
const M2 = run({v: 0.5, a: 0.1, d: 0.1, intensity: 0.3, label: 'pleased'}, {speakS: 5});
check('mild: no camera push-in', Math.min(...M2.out.map(o => o.cam_zoom)) > 0.28);
check('mild: small posture change', Math.abs(M2.out[90].body_pitch - M2.base.body_pitch) < 0.12);
// 4. a busy body: the face shows, the posture does not
{
  const T = makeMuse(7), {m} = T;
  for (let i = 0; i < 30; i++) T.step(1, false);
  m.command({action: 'move', name: 'walk', steps: 3}, 'tool');
  for (let i = 0; i < 5; i++) T.step(1, false);
  const busy = m.engine.kit.busy();
  m.affect({v: -0.75, a: -0.4, d: -0.4, intensity: 0.9, hold_s: 5});
  let o; for (let i = 0; i < 30; i++) o = m.step(1 / 30, {features: {t: m.now(), face: null, hands: []}});
  const st = m.affectState();
  check('busy body (a move): feeling in the face, posture weight off', !busy || (st.wb < 0.2 && o.brow_knit > 0.3), `busy ${busy} wb ${st.wb} knit ${o.brow_knit.toFixed(2)}`);
}
// 5. no verdict: nothing changes (the state stays idle, never stepped)
{
  const A = makeMuse(3), B = makeMuse(3);
  for (let i = 0; i < 90; i++) { A.step(1); B.step(1); }
  const same = JSON.stringify(A.rows[89]) === JSON.stringify(B.rows[89]);
  check('no verdict: deterministic and untouched', same && A.m.affectState().phase === 'idle');
}
// 6. a refinement glides (no jump) and a plain verdict releases
{
  const T = makeMuse(7), {m} = T; for (let i = 0; i < 30; i++) T.step(1, false);
  m.affect({v: 0.6, a: 0.3, d: 0.2, intensity: 0.6, stage: 'pre', hold_s: 5});
  let prev = null, jump = 0;
  for (let i = 0; i < 60; i++) { const o = m.step(1 / 30, {features: {t: m.now(), face: null, hands: []}}); if (i === 20) m.affect({v: 0.9, a: 0.8, d: 0.3, intensity: 0.9, stage: 'reply', hold_s: 5}); if (prev) jump = Math.max(jump, Math.abs(o.smile - prev.smile), Math.abs(o.brow_l - prev.brow_l)); prev = o; }
  check('a refinement glides (per-frame change of smile / brow < 0.08)', jump < 0.08, jump.toFixed(3));
  m.affect({v: 0, a: 0, d: 0, intensity: 0.3, hold_s: 5});
  check('a plain verdict releases the feeling', m.affectState().phase !== 'on');
}
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
