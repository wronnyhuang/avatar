// node tests/sem_page_offmode_test.cjs — OFF = IDENTICAL proof: this tree's app/motion.js and app/perform.js vs the
// commit before the semantic planner (git show <BASE>:app/...; BASE env, default 0403b74 = the first public commit) on a
// scripted session, exact equality (SKIP when this is not a git checkout or the commit is missing):
//  1. motion: muse + fox, 75 s at 30 fps — his face moving / lost / back, his voice and the avatar's voice in turns,
//     commanded gestures / expression / look / hold / release, two own-side keyframe jobs streamed in arrival order (one
//     released on its cue, one talked through), a thinking gesture handed over to a keyframe job, a whole-body move (muse),
//     'stop'. Every frame's full output object and every action event must be byte-identical (JSON).
//  2. perform.js: the same call / onKf / onAction / avatarAudio / stop / cache / library / failure script with the planner
//     option absent AND set to 'current': every control message, every prompt and every tool result byte-identical.
'use strict';
const fs = require('fs'), path = require('path'), {execFileSync} = require('child_process');
const ROOT = path.join(__dirname, '..'), BASE = process.env.BASE || '0403b74';
const loadSrc = src => { const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; };
const cur = rel => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const base = rel => execFileSync('git', ['show', `${BASE}:${rel}`], {cwd: ROOT, encoding: 'utf8', maxBuffer: 64 << 20});
try { base('app/motion.js'); } catch (e) { console.log(`SKIP: no git checkout with ${BASE} (${String(e.message).split('\n')[0]})`); process.exit(0); }
// the gaze split is a deliberate behaviour change of the rules path; this identity proof runs with its switch off
// (?gazesplit=0, which the base build does not read) — so it also proves that gazesplit=0 IS the old motion, byte for byte.
globalThis.location = {search: '?gazesplit=0'};
globalThis.AvatarMovesLib = loadSrc(cur('app/moves.js'));
if (cur('app/moves.js') !== base('app/moves.js')) console.log('note: app/moves.js differs from base');
const Mnew = loadSrc(cur('app/motion.js')), Mold = loadSrc(base('app/motion.js'));
const Pnew = loadSrc(cur('app/perform.js')), Pold = loadSrc(base('app/perform.js'));
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const J = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), 'utf8'));
const BODY = {muse: J('renderer/animals/muse.body.json')};
const PROPS = {muse: J('renderer/animals/muse.props.json')};
const DT = 1 / 30, N = 75 * 30;

function session(Lib, name) {
  const ev = [];
  const m = Lib.createMotion({seed: 4555, character: name, onAction: e => ev.push(JSON.stringify(Object.assign({}, e, {wall: undefined})))   /* wall = Date.now(): the only field that may differ */});
  if (BODY[name]) m.setProps(JSON.parse(JSON.stringify(PROPS[name])), JSON.parse(JSON.stringify(BODY[name])), name); else m.setProps(null, null, name);
  const frames = [];
  const kfA = [{t: 0.2, right_paw_y: 0.5, right_paw_x: 0.2, head_yaw: 0.1}, {t: 0.7, right_paw_y: 0.9, right_paw_z: 0.2, smile: 0.4}, {t: 1.2, right_paw_y: 0.6, mouth_open: 0.3},
    {t: 1.8, right_paw_y: 0.9, eyes_closed: 0.6}, {t: 2.4, right_paw_y: 0.2, left_paw_y: 0.4}, {t: 3.0, root_y: 0.2, squash: -0.2}, {t: 3.6, root_y: 0}];
  const kfB = [{t: 0.1, head_pitch: 0.2, body_lean: 0.3}, {t: 0.5, head_pitch: -0.15, left_paw_y: 0.7}, {t: 1.0, head_pitch: 0.1, root_yaw: 0.25}, {t: 1.6, head_pitch: 0, root_yaw: 0.5, left_paw_y: 0}];
  let t = 0;
  for (let i = 0; i < N; i++) {
    const s = i * DT;
    // scripted inputs
    const present = !(s > 30 && s < 33.5);
    const ax = 0.3 * Math.sin(s * 0.7), ay = 0.12 * Math.sin(s * 0.45 + 1);
    const av = (s > 8 && s < 12) || (s > 22 && s < 25) || (s > 40 && s < 44) || (s > 61 && s < 66);
    const uv = (s > 4 && s < 6.5) || (s > 18 && s < 20) || (s > 52 && s < 54);
    t += DT;
    const inp = {features: {t, face: present ? {present: true, x: ax, y: ay, ax, ay, z: .5, lookX: 0, lookY: 0, blend: {}} : {present: false}, attending: present, attention: present ? .9 : .1, motion: .02 + 0.02 * Math.abs(Math.sin(s)), hands: []},
      avatarVoice: {voiced: av, rms: av ? 0.03 + 0.02 * Math.abs(Math.sin(s * 9)) : 0}, userVoice: {voiced: uv, rms: uv ? 0.04 : 0}, speaking: av, listening: uv};
    // scripted commands
    const at = x => Math.abs(s - x) < DT / 2;
    if (at(5)) m.command({action: 'gesture', name: 'wave'}, {src: 'voice'});
    if (at(10)) m.command({action: 'expression', name: 'happy', intensity: 0.8}, {src: 'voice'});
    if (at(14)) m.command({action: 'look', target: 'left'}, {src: 'voice'});
    if (at(16)) m.command({action: 'release', parts: 'all'}, {src: 'voice'});
    if (at(20)) m.command({action: 'keyframes', id: 'kA', description: 'wave and hop', poses: kfA.slice(0, 2), done: false, hold: true}, {src: 'voice'});
    if (at(20.4)) m.kfFeed('kA', kfA.slice(2, 5), false);
    if (at(20.9)) m.kfFeed('kA', kfA.slice(5), true);
    if (at(21.2)) m.kfRelease('kA');
    if (at(27)) m.command({action: 'gesture', name: 'nod'}, {src: 'voice'});
    if (at(35)) { m.think({variant: 'chin'}); }
    if (at(36.2)) m.command({action: 'keyframes', id: 'kB', description: 'bow and turn', poses: kfB, done: true, hold: false}, {src: 'voice'});
    if (at(41)) m.command({action: 'keyframes', id: 'kA2', description: 'again', poses: kfA, done: true, hold: false}, {src: 'voice'});
    if (at(42)) m.command({action: 'hold'}, {src: 'voice'});
    if (at(43.5)) m.command({action: 'stop'}, {src: 'voice'});
    if (at(47) && name === 'muse') m.command({action: 'move', name: 'jump'}, {src: 'voice'});
    if (at(56)) m.command({action: 'pose', targets: {head: 'tilt'}, hold: true}, {src: 'voice'});
    if (at(58)) m.command({action: 'keyframes', id: 'kC', description: 'nod', poses: kfB, done: true, hold: false}, {src: 'voice'});
    if (at(68)) m.command({action: 'release', parts: 'all'}, {src: 'voice'});
    frames.push(JSON.stringify(m.step(DT, inp)));
  }
  return {frames, ev, st: JSON.stringify(m.state ? m.state() : null)};
}
console.log(`== 1. motion.js vs ${BASE}:app/motion.js, ${N} frames each`);
for (const name of ['muse', 'fox']) {
  const a = session(Mold, name), b = session(Mnew, name);
  let first = -1; for (let i = 0; i < N; i++) if (a.frames[i] !== b.frames[i]) { first = i; break; }
  const evSame = a.ev.length === b.ev.length && a.ev.every((x, i) => x === b.ev[i]);
  const kfEv = a.ev.filter(x => /"kind":"kf"/.test(x)).length;
  check(`${name}: every frame byte-identical (${N} frames, ${JSON.parse(a.frames[0]) ? Object.keys(JSON.parse(a.frames[0])).length : 0} params), every action event identical (${a.ev.length}, ${kfEv} keyframe-job events), state() identical`,
    first < 0 && evSame && a.st === b.st && kfEv >= 6, first >= 0 ? 'first difference at frame ' + first : '');
}

console.log(`== 2. perform.js vs ${BASE}:app/perform.js`);
function perfScript(Lib, plannerOpt) {
  let t = 1000; const timers = []; const out = [];
  const motion = {command: c => { out.push('cmd ' + JSON.stringify(c)); return {ok: true, id: c.id}; }, kfFeed: (id, p, d, info) => { out.push('feed ' + JSON.stringify([id, p, d, info])); return {ok: true}; },
    kfRelease: id => { out.push('rel ' + id); return {ok: true}; }, kfState: () => ({planned_s: 1.4}), state: () => ({posture: 'standing'})};
  const th = {begin: id => { out.push('think ' + id); return {ok: true}; }, known: id => out.push('known ' + id), end: (id, w) => out.push('tend ' + id + ' ' + w), cancel: w => out.push('tcancel ' + w)};
  const d = {motion: () => motion, prompt: (x, o) => { out.push('prompt ' + JSON.stringify([x, o && o.respond, o && o.maxWaitMs])); if (o && o.onAudio) out.push('(onAudio)'); return null; },
    sendCtl: o => out.push('ctl ' + JSON.stringify(o)), runCommand: c => { out.push('run ' + JSON.stringify(c)); return {ok: true, id: 'e' + out.length}; },
    character: () => 'muse', now: () => t, timers: {set: (f, ms) => { timers.push({f, at: t + ms}); return timers.length; }, clear: () => {}}, thinker: () => th,
    userSpeaking: () => false, lastUserSpeech: () => -1e12, log: r => out.push('log ' + JSON.stringify(r))};
  if (plannerOpt) Object.assign(d, plannerOpt);
  const P = Lib.createPerformer(d);
  const adv = ms => { t += ms; const due = timers.filter(x => x.at <= t && !x.done); for (const x of due) { x.done = true; x.f(); } };
  const R = x => out.push('ret ' + JSON.stringify(x));
  R(P.call('wave at me', {start_ms: 900, start_lat_ms: 300})); adv(800);
  P.onKf({type: 'kf', id: 'p1', poses: [{t: 0.2, right_paw_y: 0.5}], done: false}); adv(200);
  P.onKf({type: 'kf', id: 'p1', poses: [{t: 0.8, right_paw_y: 0.1}], done: true, ok: true, source: 'planner', ms_first_pose: 700, ms_done: 1000, usd: 0.001, summary: 'wave'});
  P.onAction({kind: 'kf', id: 'p1', phase: 'ready'}); adv(120); P.avatarAudio(); adv(30); P.onAction({kind: 'kf', id: 'p1', phase: 'start'}); adv(1500); P.onAction({kind: 'kf', id: 'p1', phase: 'end'});
  adv(20000);
  R(P.call('wave at me')); adv(50); P.onAction({kind: 'kf', id: 'p2', phase: 'ready'}); adv(800); P.onAction({kind: 'kf', id: 'p2', phase: 'start'}); P.onAction({kind: 'kf', id: 'p2', phase: 'end'});
  adv(20000);
  R(P.call('jump', {talk: 'through'})); P.onKf({type: 'kf', id: 'p3', source: 'library', engine: [{action: 'move', name: 'jump'}], done: true, ok: true}); P.onAction({kind: 'move', id: 'e' + 0, phase: 'end'}); adv(31000);
  adv(20000);
  R(P.call('do a backflip')); R(P.call('and then wave')); adv(300); P.onKf({type: 'kf', id: 'p4', poses: [], done: true, ok: false, error: 'nope'}); adv(16000);
  adv(20000);
  R(P.call('spin around')); adv(500); R(P.stop('he said stop')); adv(25000);
  R(P.call('touch your toes')); adv(21000);
  out.push('list ' + JSON.stringify(P.list())); out.push('stats ' + JSON.stringify(P.stats()));
  return out;
}
const o0 = perfScript(Pold, null), o1 = perfScript(Pnew, null), o2 = perfScript(Pnew, {planner: () => 'current', semState: () => ({user: {present: true, ax: 0.2}})});
const diff = (a, b) => { for (let i = 0; i < Math.max(a.length, b.length); i++) if (a[i] !== b[i]) return i + ': ' + String(a[i]).slice(0, 160) + ' | ' + String(b[i]).slice(0, 160); return null; };
const nctl = o0.filter(x => x.startsWith('ctl ')).length;
check(`perform.js without the planner option: ${o0.length} records (${nctl} control messages) byte-identical to base`, !diff(o0, o1), diff(o0, o1));
check(`perform.js with planner 'current' (+ a semState that must not be read): byte-identical to base`, !diff(o0, o2), diff(o0, o2));
console.log(`sem_page_offmode_test: ${passes} passed, ${fails} failed`);
process.exit(fails);
