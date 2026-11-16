// node tests/rigprobe_test.cjs [--verbose] — rig probe (docs/RIG-DEBUG.md §2) on app/motion.js:
//  (1) determinism: one seeded synthetic input stream (his face moving and lost, his expressions, a hand waving, him
//      speaking, the avatar speaking, wave / startle / look-back / nod / point events, commands, a held pose, a freeze,
//      a feeling, a thinking gesture, a co-speech stroke, a keyframe job, a whole-body move on the muse) for 70 s at 30 Hz
//      through step() with the probe OFF vs ON + probe() every frame (and toggled on / off mid-run) -> every output key of
//      every frame identical (Object.is), the action events identical; on 5 setups (fox, muse, fox + listener + program
//      layer, fox in learned mode, orb + the learned prior)
//      + the base decomposition: every frame, Σ of each param's base terms == the base value step() computed
//  (2) attribution scenarios: wave event / a waving hand -> 'reflex' on the paw; look left -> 'look'; a commanded gesture
//      -> 'cmd'; a hold -> 'hold'; the avatar speaking -> 'speech' on head / brows; him speaking -> 'listen'; a keyframe job
//      -> 'planner'; idle -> 'life'; his smile -> 'mirror'; a feeling -> 'feeling'; startle -> 'reflex'; listener / program /
//      learned / whole-body move layers; LIPSYNC always 'lips'; meta()
//  (3) cost: probe() µs per call, step() µs with the probe on vs off
// Exit code = number of failures.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const AffLib = load('app/affect.js'); globalThis.AvatarAffectLib = AffLib;
const M = load('app/motion.js');
const ANIM = path.join(__dirname, '..', 'renderer', 'animals');
const DT = 1 / 30, VERBOSE = process.argv.includes('--verbose');
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const PROPS = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), BODY = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json')));
const priorMod = (() => { const src = fs.readFileSync(path.join(__dirname, '..', 'app', 'prior.js'), 'utf8'), mod = {exports: {}};
  new Function('module', 'exports', 'globalThis', src)(mod, mod.exports, {}); return mod.exports.createPrior || mod.exports; })();
const PRIOR_W = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'app', 'prior_weights.json'), 'utf8'));

function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function face(x, y, blend = {}) {
  return {present: true, x, y, ax: Math.atan(x * M.TAN_HFOV2), ay: Math.atan(y * M.TAN_VFOV2), z: 0.45, size: 0.1, yaw: 0, pitch: 0, roll: 0.05 * x,
    lookX: 0, lookY: -0.1, blend: Object.assign({mouthSmileLeft: 0.04, mouthSmileRight: 0.05, browInnerUp: 0.05}, blend)};
}
const voice = (rnd, t, on) => on ? {rms: Math.max(0, 0.05 + 0.04 * Math.sin(2 * Math.PI * 4.3 * t) + 0.01 * (rnd() - 0.5)), pitch: 160 + 30 * Math.sin(2 * Math.PI * 0.7 * t), voiced: true}
  : {rms: 0.001 * rnd(), pitch: 0, voiced: false};

// the seeded input stream: frame i -> {features, userVoice, avatarVoice, act: [fn(m)]} (the same objects for every run)
function makeStream(secs, seed, o = {}) {
  const rnd = mulberry32(seed), N = Math.round(secs / DT), frames = [];
  for (let i = 0; i < N; i++) {
    const t = (i + 1) * DT, x = 0.25 * Math.sin(2 * Math.PI * t / 9) + 0.05 * (rnd() - 0.5), y = 0.06 * Math.sin(2 * Math.PI * t / 13);
    const blend = t > 10 && t < 14 ? {mouthSmileLeft: 0.75, mouthSmileRight: 0.7, cheekSquintLeft: 0.4, cheekSquintRight: 0.4}
      : t > 40 && t < 43 ? {browDownLeft: 0.7, browDownRight: 0.7, mouthFrownLeft: 0.5, mouthFrownRight: 0.5}
      : t > 54 && t < 55.5 ? {eyeWideLeft: 0.8, eyeWideRight: 0.8, browOuterUpLeft: 0.7, browOuterUpRight: 0.7, jawOpen: 0.5} : {};
    const present = !(t > 30 && t < 33);
    const hands = t > 18 && t < 20.5 ? [{side: 'l', x: x - 0.5 + 0.07 * Math.sin(2 * Math.PI * 2.5 * t), y: 0.1, open: 0.9, gesture: 'Open_Palm', age: 0, vx: 0.3, vy: 0}] : [];
    const features = {t, face: present ? face(x, y, blend) : {present: false}, attending: t < 36 || t > 37.5, attention: 0.9, motion: 0.03 + 0.02 * rnd(), hands};
    const userVoice = voice(rnd, t, (t > 5 && t < 9) || (t > 45 && t < 49));
    const avatarVoice = voice(rnd, t, (t > 22 && t < 28) || (t > 56 && t < 62));
    const act = [];
    const at = (s, fn) => { if (Math.abs(t - s) < DT / 2) act.push(fn); };
    at(3, m => m.setRestSmile && m.setRestSmile(1));
    at(12, m => m.affect({label: 'happy', v: 0.7, a: 0.5, d: 0.3, intensity: 0.8, hold_s: 6, source: 'reply'}));
    at(15, m => m.event({type: 'wave', side: 'r', t: 15}));
    at(16.5, m => m.think({variant: 'chin'}));
    at(17.8, m => m.thinkEnd('test', 0.4));
    at(23, m => m.playGesture({name: 'beat_test', dur: 1.2, attack: 0.25, release: 0.3, fn: (u) => ({set: {hand_l_y: 0.35, hand_l_x: 0.15}, add: {head_pitch: 0.02 * Math.sin(6 * u)}})}, {src: 'cospeech', side: 'l', clock: 'motion', delay: 0.2}));
    at(25, m => m.command({action: 'gesture', name: 'thumbs_up'}, {src: 'ui'}));
    at(29, m => m.command({action: 'look', target: 'left'}, {src: 'ui'}));
    at(31.5, m => m.event({type: 'face_appear', t: 31.5}));
    at(33, m => m.command({action: 'expression', name: 'happy'}, {src: 'ui'}));
    at(36, m => m.event({type: 'approach_fast', rate: 2.2, t: 36}));
    at(38, m => m.event({type: 'look_back', away_s: 2, t: 38}));
    at(41, m => m.command({action: 'pose', targets: {head_yaw: 0.25, hand_r_y: 0.6}, hold: true}, {src: 'ui'}));
    at(42, m => m.affect({label: 'sad', v: -0.6, a: -0.3, d: -0.3, intensity: 0.7, hold_s: 4, source: 'reply'}));
    at(44, m => m.command({action: 'release', parts: ['head']}, {src: 'ui'}));
    at(47, m => m.event({type: 'nod', t: 47}));
    at(50, m => m.command({action: 'hold'}, {src: 'ui'}));
    at(53, m => m.command({action: 'release'}, {src: 'ui'}));
    at(60, m => m.event({type: 'point', dir: {x: 0.6, y: 0.1}, t: 60}));
    at(63, m => m.command({action: 'keyframes', id: 'k1', description: 'look up and raise a paw', done: true, poses: [
      {t: 0, head_pitch: 0, right_paw_y: 0}, {t: 0.6, head_pitch: -0.2, right_paw_y: 0.7, smile: 0.5}, {t: 1.8, head_pitch: -0.2, right_paw_y: 0.7}, {t: 2.4, head_pitch: 0, right_paw_y: 0}]}, {src: 'tool'}));
    if (o.muse) at(67, m => m.command({action: 'move', name: 'jump'}, {src: 'tool'}));
    frames.push({t, features, userVoice, avatarVoice, act});
  }
  return frames;
}
function makeMotion(setup, seed) {
  const ev = [];
  const m = M.createMotion({seed, preset: 'default', onAction: e => ev.push(e)});
  if (setup.muse) m.setProps(JSON.parse(JSON.stringify(PROPS)), JSON.parse(JSON.stringify(BODY)), 'muse');
  else m.setProps(null, null, setup.char || 'fox');
  if (setup.listener) m.setListener(({t, state}) => ({smile: 0.3 + 0.1 * Math.sin(t), head_roll: 0.05 * Math.sin(0.7 * t), brow_l: state.speaking ? 0.3 : 0.1, mouth_open: 0.1}));
  if (setup.program) m.setProgramLayer(ctx => ({head_yaw: [0.15 * Math.sin(ctx.t), 0.6], ear_l: [0.4, 0.8], hand_r_x: [0.2, ctx.t > 30 ? 1 : 0]}));
  if (setup.learned) m.setLearned(true);
  if (setup.prior) m.setPrior(priorMod(PRIOR_W));
  if (setup.gest) m.setGesture({live: true, enabled: true, intensity: 0.8});
  return {m, ev};
}
// mode: 'off' | 'on' (probe() every frame) | 'cap' (probe on, probe() never called) | 'toggle' (on 20-45 s and 60 s+)
function runStream(setup, frames, mode, seed = 4553, hook) {
  const {m, ev} = makeMotion(setup, seed), outs = [], cost = {step: [], probe: []}, probes = [];
  if (mode === 'on' || mode === 'cap') m.setProbe(true);
  for (let i = 0; i < frames.length; i++) {
    const F = frames[i];
    if (mode === 'toggle') { if (Math.abs(F.t - 20) < DT / 2 || Math.abs(F.t - 60) < DT / 2) m.setProbe(true); if (Math.abs(F.t - 45) < DT / 2) m.setProbe(false); }
    for (const fn of F.act) fn(m);
    const t0 = process.hrtime.bigint();
    const o = m.step(DT, {features: F.features, userVoice: F.userVoice, avatarVoice: F.avatarVoice});
    const t1 = process.hrtime.bigint();
    cost.step.push(Number(t1 - t0) / 1000);
    outs.push(Object.assign({}, o));
    if (mode === 'on' || (mode === 'toggle' && m.state.dbg)) {
      const t2 = process.hrtime.bigint(), p = m.probe(), t3 = process.hrtime.bigint();
      cost.probe.push(Number(t3 - t2) / 1000);
      if (hook) hook(m, p, o, i, F);
      if (VERBOSE && i % 300 === 0) probes.push(p);
    }
  }
  return {outs, ev: ev.map(e => JSON.stringify(Object.assign({}, e, {wall: 0}))), cost, m, probes};
}
function sameOuts(A, B) {
  let diffs = 0, first = null, n = 0;
  for (let i = 0; i < Math.max(A.length, B.length); i++) {
    const a = A[i], b = B[i];
    if (!a || !b) { diffs++; if (!first) first = `frame ${i} missing`; continue; }
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) { diffs++; if (!first) first = `frame ${i} key count ${ka.length} vs ${kb.length}`; continue; }
    for (const k of ka) { n++; if (!Object.is(a[k], b[k])) { diffs++; if (!first) first = `frame ${i} ${k}: ${a[k]} vs ${b[k]}`; } }
  }
  return {diffs, first, n};
}
const stats = a => { const s = a.slice().sort((x, y) => x - y), mean = a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  return {mean, p50: s[Math.floor(s.length * 0.5)] || 0, p95: s[Math.floor(s.length * 0.95)] || 0}; };
const fmt = s => `mean ${s.mean.toFixed(1)} µs, p50 ${s.p50.toFixed(1)}, p95 ${s.p95.toFixed(1)}`;

// ------------------------------------------------------------------ (1) determinism + the base decomposition
const SETUPS = [
  {name: 'fox', char: 'fox'},
  {name: 'muse', muse: true},
  {name: 'fox+listener+program', char: 'fox', listener: true, program: true},
  {name: 'fox learned', char: 'fox', learned: true},
  {name: 'orb+prior', char: 'orb', prior: true},
];
const COST = {};
for (const setup of SETUPS) {
  const frames = makeStream(70, 7, {muse: !!setup.muse});
  const A = runStream(setup, frames, 'off');
  let baseChecks = 0, baseBad = 0, baseFirst = null, ownersBad = 0, layersSeen = new Set(), probeErr = null;
  const RIG = new Set(M.RIG_PARAMS), ALLP = M.createMotion({seed: 1}).meta().allParams, NL = M.createMotion({seed: 1}).meta().layers.length;
  const B = runStream(setup, frames, 'on', 4553, (m, p, o, i) => {
    try {
      const D = m.state.dbg;
      if (!p || !p.captured || !D || !D.base) { baseBad++; if (!baseFirst) baseFirst = `frame ${i}: no capture`; return; }
      // D.bt[param * NL + layer] = that layer's term of the param's base formula; D.o0 = the frame before the feeling
      for (let k = 0; k < ALLP.length; k++) {
        if (!D.bh[k]) continue;
        const q = ALLP[k]; let s = 0; for (let l = 0; l < NL; l++) s += D.bt[k * NL + l];
        baseChecks++;
        if (!(Math.abs(s - D.o0[k]) < 1e-9)) { baseBad++; if (!baseFirst) baseFirst = `frame ${i} ${q}: Σ terms ${s} vs base ${D.o0[k]}`; }
        if (D.fitK[k] === 1 && !/^shoulder_/.test(q)) {   // a clamped base param: its raw terms must clamp to the value step() wrote
          const hi = q === 'mouth_open' ? 0.5 : 1; baseChecks++;
          if (!(Math.abs(Math.min(hi, Math.max(0, D.fitRaw[k])) - D.o0[k]) < 1e-9)) { baseBad++; if (!baseFirst) baseFirst = `frame ${i} ${q}: clamp(raw ${D.fitRaw[k]}) vs ${D.o0[k]}`; }
        }
      }
      for (let k = 0; k < ALLP.length; k++) if (!D.bh[k] && D.o0[k] !== 0) { baseBad++; if (!baseFirst) baseFirst = `frame ${i} ${ALLP[k]}: base value ${D.o0[k]} with no terms`; }
      for (const q of M.RIG_PARAMS) if (!p.owners[q] || typeof p.owners[q].by !== 'string') { ownersBad++; break; }
      for (const q in p.owners) { if (!RIG.has(q) && !(o[q] !== 0 || p.owners[q].by !== 'life')) ownersBad++; layersSeen.add(p.owners[q].by); }
    } catch (e) { probeErr = probeErr || String(e.stack || e); }
  });
  const C = runStream(setup, frames, 'toggle');
  const ab = sameOuts(A.outs, B.outs), ac = sameOuts(A.outs, C.outs);
  check(`${setup.name}: probe on + probe() every frame -> step() outputs identical (${frames.length} frames, ${ab.n} values)`, ab.diffs === 0 && A.outs.length === frames.length, ab.first || '');
  check(`${setup.name}: probe toggled on/off mid-run -> identical`, ac.diffs === 0, ac.first || '');
  check(`${setup.name}: action events identical (${A.ev.length})`, A.ev.length === B.ev.length && A.ev.every((e, i) => e === B.ev[i]) && A.ev.every((e, i) => e === C.ev[i]));
  check(`${setup.name}: probe() never threw`, !probeErr, probeErr || '');
  check(`${setup.name}: base decomposition Σ terms == base value (${baseChecks} checks)`, baseBad === 0 && baseChecks > 0, baseFirst || '');
  check(`${setup.name}: owners cover the 77 rig params; extras only when nonzero or owned`, ownersBad === 0, `layers seen as 'by': ${[...layersSeen].join(' ')}`);
  COST[setup.name] = {off: stats(A.cost.step), on: stats(B.cost.step), probe: stats(B.cost.probe)};
  if (VERBOSE) for (const p of B.probes) console.log(JSON.stringify({t: +p.t.toFixed(2), gaze: p.gaze, by: Object.fromEntries(Object.entries(p.owners).filter(([, v]) => v.by !== 'life' && v.by !== 'lips').map(([k, v]) => [k, v.by]))}));
}

// ------------------------------------------------------------------ (2) attribution scenarios
function scene(setup = {}, seed = 11) {
  const {m, ev} = makeMotion(setup, seed); m.setProbe(true);
  let t = 0, f = {face: face(0, 0), attending: true, attention: 0.9, motion: 0.02, hands: []}, uv = null, av = null;
  const S = {m, ev,
    set(o) { if ('features' in o) f = o.features; if ('uv' in o) uv = o.uv; if ('av' in o) av = o.av; return S; },
    step(secs = DT, each) { const n = Math.max(1, Math.round(secs / DT)); let p = null; for (let i = 0; i < n; i++) { t += DT; const ff = typeof f === 'function' ? f(t) : f;
      m.step(DT, {features: ff, userVoice: typeof uv === 'function' ? uv(t) : uv, avatarVoice: typeof av === 'function' ? av(t) : av}); p = m.probe(); if (each) each(p, t); } return p; },
    now: () => t};
  return S;
}
const by = (p, q) => p.owners[q] ? p.owners[q].by : '(none)';
const has = (p, q, l) => !!p.owners[q] && (p.owners[q].by === l || p.owners[q].also.some(a => a[0] === l));
const show = (p, ps) => ps.map(q => `${q}=${by(p, q)}${p.owners[q] ? '@' + p.owners[q].w : ''}`).join(' ');
{   // idle -> life
  const S = scene().set({features: {face: {present: false}, hands: []}});
  let commanded = 0, seen = new Set();
  const p = S.step(8, p => { for (const q in p.owners) { const b = p.owners[q].by; seen.add(b); if (/^(cmd|look|hold|planner|move|program|learned|listen|speech|feeling|cospeech|think)$/.test(b)) commanded++; } });
  check('idle (nobody there, silent): breath / body lean / tail curl / paws / ear turn / nostril -> life', ['breath', 'body_lean', 'tail_curl', 'hand_l_x', 'hand_r_open', 'ear_l_turn', 'nostril'].every(q => by(p, q) === 'life'), show(p, ['breath', 'body_lean', 'tail_curl', 'hand_l_x', 'ear_l_turn', 'gaze_x']));
  check('idle: no commanded / speech / listening / feeling layer owns anything for 8 s', commanded === 0, `by values seen: ${[...seen].join(' ')}`);
  check('LIPSYNC params always lips', ['jaw', 'round', 'lip_open', 'tongue_back', 'voice'].every(q => by(p, q) === 'lips' && p.owners[q].w === 1));
}
{   // wave event -> reflex wave back on the paw
  const S = scene(); S.step(2);
  S.m.event({type: 'wave', side: 'r'});
  const p = S.step(0.7);
  const G = p.gestures.find(g => g.name === 'wave');
  check('event wave -> a reflex wave back owns the waving paw (hand_r_*)', G && G.src === 'reflex' && ['hand_r_x', 'hand_r_y', 'hand_r_open'].every(q => by(p, q) === 'reflex'), show(p, ['hand_r_x', 'hand_r_y', 'hand_r_open', 'smile']) + ' ' + JSON.stringify(G || null));
}
{   // a waving user (the greeting detector, no event) -> reflex
  const S = scene(); S.step(2);
  S.set({features: t => ({face: face(0.05, 0), attending: true, attention: 0.9, motion: 0.05, hands: [{side: 'l', x: -0.45 + 0.07 * Math.sin(2 * Math.PI * 2.5 * t), y: 0.1, open: 0.9, gesture: 'Open_Palm', age: 0, vx: 0.4, vy: 0}]})});
  let hit = null;
  S.step(3, (p, t) => { if (!hit && p.gestures.some(g => g.name === 'wave' && g.src === 'reflex' && g.w > 0.9) && ['hand_l_y', 'hand_r_y'].some(q => by(p, q) === 'reflex')) hit = {t: +t.toFixed(2), s: show(p, ['hand_l_y', 'hand_r_y'])}; });
  check('a waving hand beside his face -> wave back, the paw by reflex', !!hit, hit ? `at ${hit.t}s ${hit.s}` : 'no wave back seen');
}
{   // command look left -> look owns gaze / head
  const S = scene(); S.step(2);
  S.m.command({action: 'look', target: 'left'}, {src: 'ui'});
  const p = S.step(0.9);
  check("command look left -> 'look' owns gaze_x / gaze_y / head_yaw", ['gaze_x', 'gaze_y', 'head_yaw'].every(q => by(p, q) === 'look') && p.look === 'left' && p.gaze.src === 'look' && p.gaze.target === 'left', show(p, ['gaze_x', 'gaze_y', 'head_yaw', 'head_pitch']) + ` gaze=${JSON.stringify(p.gaze)}`);
}
{   // commanded gesture -> cmd
  const S = scene(); S.step(2);
  S.m.command({action: 'gesture', name: 'thumbs_up', side: 'screen_left'}, {src: 'ui'});
  const p = S.step(0.8);
  check("commanded thumbs-up -> 'cmd' owns thumb_l / hand_l_*", ['thumb_l', 'hand_l_x', 'hand_l_y', 'hand_l_z'].every(q => by(p, q) === 'cmd') && p.gestures.some(g => g.name === 'thumbs_up' && g.layer === 'cmd'), show(p, ['thumb_l', 'hand_l_y', 'smile']));
  S.m.command({action: 'expression', name: 'surprised'}, {src: 'ui'});
  const p2 = S.step(0.8);
  check("commanded expression -> 'cmd' owns eye_wide / mouth_open", by(p2, 'eye_wide') === 'cmd' && by(p2, 'mouth_open') === 'cmd', show(p2, ['eye_wide', 'mouth_open', 'brow_l']));
}
{   // hold -> hold
  const S = scene(); S.step(2);
  S.m.command({action: 'pose', targets: {head_yaw: 0.3, hand_r_y: 0.5}, hold: true}, {src: 'ui'});
  const p = S.step(1.2);
  check("held pose -> 'hold' owns head_yaw / hand_r_y (w 1)", by(p, 'head_yaw') === 'hold' && by(p, 'hand_r_y') === 'hold' && p.owners.head_yaw.w === 1 && p.holds.some(h => h.p === 'head_yaw'), show(p, ['head_yaw', 'hand_r_y']));
  S.m.command({action: 'release'}, {src: 'ui'});
  const p2 = S.step(1.5);
  check("released -> back to the live layers", by(p2, 'head_yaw') !== 'hold' && !p2.holds.length, show(p2, ['head_yaw', 'hand_r_y']));
  S.m.command({action: 'hold'}, {src: 'ui'});
  const p3 = S.step(0.5);
  check("'hold that' (freeze) -> posture params by 'hold'", ['head_pitch', 'body_lean', 'hand_l_x', 'shoulder_r'].every(q => by(p3, q) === 'hold') && by(p3, 'blink_l') !== 'hold' && by(p3, 'gaze_x') !== 'hold', show(p3, ['head_pitch', 'body_lean', 'hand_l_x', 'blink_l', 'gaze_x']));
}
{   // the avatar speaking -> speech on head / brows
  const S = scene(); S.step(2);
  const rnd = mulberry32(3); S.set({av: t => voice(rnd, t, true)});
  let n = 0, pitchSp = 0, pitchBy = 0, browSp = 0, browBy = 0, yawBy = 0;
  S.step(8, (p, t) => { if (t < 3.5) return; n++; if (has(p, 'head_pitch', 'speech')) pitchSp++; if (by(p, 'head_pitch') === 'speech') pitchBy++; if (by(p, 'head_yaw') === 'speech') yawBy++;
    if (has(p, 'brow_l', 'speech')) browSp++; if (by(p, 'brow_l') === 'speech' || by(p, 'brow_r') === 'speech') browBy++; });
  check("avatar speaking -> 'speech' on head_pitch (by or also) most frames, and owns head / brows at times", pitchSp / n > 0.8 && pitchBy / n > 0.3 && browBy > 0 && yawBy > 0,
    `head_pitch speech ${(100 * pitchSp / n).toFixed(0)}% (by ${(100 * pitchBy / n).toFixed(0)}%), head_yaw by ${(100 * yawBy / n).toFixed(0)}%, brow speech ${(100 * browSp / n).toFixed(0)}% (by ${(100 * browBy / n).toFixed(0)}%)`);
}
{   // him speaking -> listen
  const S = scene(); S.step(2);
  const rnd = mulberry32(5); S.set({uv: t => voice(rnd, t, t < 12)});
  let n = 0, z = 0, brow = 0, nod = 0;
  S.step(9, (p, t) => { if (t < 4) return; n++; if (by(p, 'head_z') === 'listen') z++; if (has(p, 'brow_l', 'listen')) brow++; if (p.gestures.some(g => g.src === 'listen')) nod++; });
  check("him speaking -> 'listen' owns head_z, shows on the brows", z / n > 0.8 && brow / n > 0.8, `head_z ${(100 * z / n).toFixed(0)}%, brow ${(100 * brow / n).toFixed(0)}%, listener-nod frames ${nod}`);
}
{   // keyframe job -> planner
  const S = scene(); S.step(2);
  const r = S.m.command({action: 'keyframes', id: 'kp', description: 'look up and raise a paw', done: true, poses: [{t: 0, head_pitch: 0, right_paw_y: 0}, {t: 0.6, head_pitch: -0.25, right_paw_y: 0.7}, {t: 1.8, head_pitch: -0.25, right_paw_y: 0.7}, {t: 2.4, head_pitch: 0, right_paw_y: 0}]}, {src: 'tool'});
  const p = S.step(1.3);
  check("keyframe job -> 'planner' owns head_pitch / hand_l_y (the character's right paw)", r.ok && by(p, 'head_pitch') === 'planner' && by(p, 'hand_l_y') === 'planner' && p.kf.length && p.kf[0].phase === 'playing', show(p, ['head_pitch', 'hand_l_y']) + ' ' + JSON.stringify(p.kf));
  const p2 = S.step(3);
  check('after the job blends back: planner gone', by(p2, 'head_pitch') !== 'planner' && !p2.kf.length, show(p2, ['head_pitch', 'hand_l_y']));
}
{   // his smile -> mirror; a feeling -> feeling
  const S = scene(); S.step(3);
  S.set({features: {face: face(0, 0, {mouthSmileLeft: 0.8, mouthSmileRight: 0.8, cheekSquintLeft: 0.4, cheekSquintRight: 0.4}), attending: true, attention: 0.9, motion: 0.02, hands: []}});
  const p = S.step(2);
  check("his smile -> 'mirror' owns smile / squint", by(p, 'smile') === 'mirror' && by(p, 'squint') === 'mirror', show(p, ['smile', 'squint', 'tail_wag']));
  S.m.affect({label: 'sad', v: -0.6, a: -0.3, d: -0.3, intensity: 0.9, hold_s: 6, source: 'reply'});
  const p2 = S.step(2);
  check("a feeling (sad) -> 'feeling' owns smile / brow_knit, affect reported", by(p2, 'smile') === 'feeling' && by(p2, 'brow_knit') === 'feeling' && p2.affect && p2.affect.w > 0.5, show(p2, ['smile', 'brow_knit', 'head_pitch']) + ' ' + JSON.stringify(p2.affect));
}
{   // startle -> reflex
  const S = scene(); S.step(3);
  S.m.event({type: 'approach_fast', rate: 2.2, source: 'hand'});
  const p = S.step(0.25);
  check("startle -> 'reflex' owns eye_wide / body_pitch / the flinching paws", by(p, 'eye_wide') === 'reflex' && by(p, 'body_pitch') === 'reflex' && by(p, 'hand_l_y') === 'reflex', show(p, ['eye_wide', 'body_pitch', 'hand_l_y', 'head_pitch']));
  S.m.event({type: 'look_back', away_s: 3});
  const p2 = S.step(0.3);
  check("look back -> the gaze source is the reflex", p2.gaze.src === 'reflex' && by(p2, 'gaze_x') === 'reflex', JSON.stringify(p2.gaze) + ' ' + show(p2, ['gaze_x', 'head_yaw']));
}
{   // listener / program layer / learned mode
  const S = scene({listener: true}); S.step(2);
  const p = S.step(0.5);
  check("listener network -> 'listen' owns smile / head_roll (rules base gone under it)", by(p, 'smile') === 'listen' && by(p, 'head_roll') === 'listen' && !has(p, 'smile', 'life'), show(p, ['smile', 'head_roll', 'brow_l']));
  S.m.command({action: 'pose', targets: {head_roll: 0.2}, hold: true}, {src: 'ui'});
  const p1 = S.step(1);
  check("a hold over the listener -> 'hold' (the listener keeps nothing there)", by(p1, 'head_roll') === 'hold' && !has(p1, 'head_roll', 'listen'), show(p1, ['head_roll']));
  const P = scene({program: true}); P.step(2);
  const q = P.step(0.5);
  check("program layer -> 'program' owns head_yaw / ear_l; a weight-0 column stays the rules'", by(q, 'head_yaw') === 'program' && by(q, 'ear_l') === 'program' && by(q, 'hand_r_x') === 'life', show(q, ['head_yaw', 'ear_l', 'hand_r_x']));
  const Lr = scene({learned: true}); Lr.step(2);
  const l = Lr.step(0.5);
  check("learned mode -> 'learned' on the face / gaze / posture groups, learned flag", l.learned && by(l, 'head_yaw') === 'learned' && by(l, 'smile') === 'learned' && by(l, 'tail_wag') !== 'learned', show(l, ['head_yaw', 'smile', 'tail_wag']));
  Lr.m.command({action: 'look', target: 'left'}, {src: 'ui'});
  const l2 = Lr.step(0.8);
  const lw = (l2.owners.gaze_x.also.find(a => a[0] === 'learned') || [0, 0])[1];
  check("learned mode + a commanded look -> the page owns the gaze (alpha -> 0: learned keeps < 0.1 of gaze_x)", by(l2, 'gaze_x') === 'look' && by(l2, 'head_yaw') === 'look' && lw < 0.1 && by(l2, 'smile') === 'learned', show(l2, ['gaze_x', 'head_yaw', 'smile']));
}
{   // a co-speech stroke, the thinking gesture
  const S = scene(); S.step(2);
  S.m.playGesture({name: 'beat', dur: 1.5, attack: 0.25, release: 0.3, fn: () => ({set: {hand_l_y: 0.4, hand_l_x: 0.15}})}, {src: 'cospeech', side: 'l', clock: 'motion'});
  const p = S.step(0.6);
  check("co-speech stroke -> 'cospeech' owns its paw", by(p, 'hand_l_y') === 'cospeech', show(p, ['hand_l_y', 'hand_l_x']));
  S.step(1.5);
  const r = S.m.think({variant: 'away'});
  const p2 = S.step(1);
  check("thinking -> 'think' owns gaze / brow_knit", r.ok && by(p2, 'brow_knit') === 'think' && by(p2, 'gaze_y') === 'think', show(p2, ['brow_knit', 'gaze_y', 'head_yaw']));
}
{   // the muse: a whole-body move -> move
  const S = scene({muse: true}); S.step(2);
  const r = S.m.command({action: 'move', name: 'jump'}, {src: 'tool'});
  let seen = null;
  S.step(2.5, (p, t) => { if (!seen && by(p, 'root_y') === 'move' && p.move) seen = {t: +t.toFixed(2), s: show(p, ['root_y', 'squash', 'body_pitch', 'cam_zoom']), move: p.move}; });
  check("muse jump -> 'move' owns root_y, probe.move named", r.ok && !!seen && !!seen.move.name, seen ? `${seen.t}s ${seen.s} ${JSON.stringify(seen.move)}` : 'never');
}
{   // meta()
  const m = M.createMotion({seed: 1}), d = m.meta();
  const L = d.layers;
  check('meta(): 77 params, allParams, ranges for every param, 19 lipsync, legacy, periodic, pri, layers in order', d.params.length === 77 && d.allParams.length === M.createMotion({seed: 2}).allParams.length &&
    d.allParams.every(p => Array.isArray(d.ranges[p]) && d.ranges[p].length === 2) && d.lipsync.length === 19 && d.legacy.length === 8 && d.periodic.length > 0 && d.pri.cmd === 4 &&
    L[0] === 'life' && L[L.length - 1] === 'lips' && L.indexOf('planner') > L.indexOf('hold') && L.indexOf('cmd') > L.indexOf('reflex') && L.indexOf('program') < L.indexOf('feeling'), JSON.stringify(L));
  d.params.push('x'); d.ranges.smile[1] = 9;
  check('meta() returns copies', m.meta().params.length === 77 && m.meta().ranges.smile[1] === 1);
  check('probe() is null with the probe off; setProbe(false) clears S.dbg', m.probe() === null && m.setProbe(true) === true && m.setProbe(false) === false && m.state.dbg === null);
}

// ------------------------------------------------------------------ (3) cost
// (a) inside the determinism runs above (the harness's own checks allocate between frames: GC lands in these numbers)
console.log('\ncost under the test harness (node, this machine; 70 s stream at 30 Hz, the checks run every frame):');
for (const [n, c] of Object.entries(COST)) console.log(`  ${n.padEnd(24)} step() probe off: ${fmt(c.off)} | probe on: ${fmt(c.on)} | probe(): ${fmt(c.probe)}`);
// (b) clean: no checks, off / capture only / capture + probe() interleaved, 3 runs each, median of the per-run means (frames 300+)
console.log('cost, clean (median of 3 interleaved runs, frames after the first 10 s):');
for (const setup of [SETUPS[0], SETUPS[1]]) {
  const frames = makeStream(70, 7, {muse: !!setup.muse}), R = {off: [], cap: [], on: [], probe: []};
  const mean = a => a.slice(300).reduce((x, y) => x + y, 0) / Math.max(1, a.length - 300);
  for (let rep = 0; rep < 3; rep++) for (const mode of ['off', 'cap', 'on']) { const r = runStream(setup, frames, mode); R[mode].push(mean(r.cost.step)); if (mode === 'on') R.probe.push(mean(r.cost.probe)); }
  const med = a => a.slice().sort((x, y) => x - y)[1];
  console.log(`  ${setup.name.padEnd(6)} step() probe off ${med(R.off).toFixed(1)} µs | probe on (capture only) ${med(R.cap).toFixed(1)} µs | probe on + probe() every frame: step ${med(R.on).toFixed(1)} µs + probe() ${med(R.probe).toFixed(1)} µs`);
}
console.log(`\nrigprobe_test: ${passes} passed, ${fails} failed`);
process.exit(fails);
