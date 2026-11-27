// pose revert: every commanded pose / move that leaves a rig dial away from rest goes back on its own
// (app/motion.js poseRevertFrame, docs/POSE-REVERT.md).            node tests/pose_revert_test.cjs [--json out.json]
//
// Every case runs on the REAL motion controller + move engine (app/motion.js + app/moves.js). His words go through the REAL
// fast matcher (app/intent.js + intent_rules.json) to get the commands the page would run; planner cases feed keyframe jobs
// (a recorded planner stream from tests/fixtures/planner_streams, and adversarial scripts that never return to rest).
// For each case, N seconds after the command:
//   - no face / eyes / brows / mouth / ears / head / torso / shoulder / arm / hand / tail dial is owned by a command
//     (motion.owned(), motion.state().held, no live keyframe job, no held look), and
//   - every such dial sits where the idle character has it: its median over the next 3 s lies inside the range that same
//     statistic (a 3 s median) takes on idle twins (same character, same state, no pose command; 5 seeds x 50 s, plus the
//     same time after an interaction that poses nothing) +- EPS, and
//   - the eyes blink again (the reflex layer owns them).
// STATE must persist through it: sitting on the chair, the thing in its paw, where it stands (at the board), the view.
// The same cases run with the switch off (createMotion({poseRevert: false}) = the code before pose revert) for the before number.
// POSE_WT=<another worktree> runs this file against that tree's app/ (the shipped baseline has no switch: all "before").
'use strict';
const fs = require('fs'), path = require('path');
const WT = process.env.POSE_WT ? path.resolve(process.env.POSE_WT) : path.join(__dirname, '..');
function load(rel) { const src = fs.readFileSync(path.join(WT, rel), 'utf8'); const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; }
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const {Intent} = load('app/intent.js');
const RULES = JSON.parse(fs.readFileSync(path.join(WT, 'app/intent_rules.json'), 'utf8'));
const ANIM = path.join(WT, 'renderer', 'animals');
const MUSE = {props: JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), body: JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json')))};
const FLURRY = fs.existsSync(path.join(ANIM, 'flurry.body.json')) ? {props: JSON.parse(fs.readFileSync(path.join(ANIM, 'flurry.props.json'))), body: JSON.parse(fs.readFileSync(path.join(ANIM, 'flurry.body.json')))} : null;
let JUNO = null;
try {
  const {execFileSync} = require('child_process'), os = require('os');
  const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'pr_juno_')), 'juno.props.json');
  execFileSync(process.env.JUNO_PY || 'python3', ['-c', `import sys,json; sys.path.insert(0,'renderer'); from humans.charfiles import props_for; json.dump(props_for('juno'), open(${JSON.stringify(out)},'w'))`], {cwd: WT});
  JUNO = {body: JSON.parse(fs.readFileSync(path.join(WT, 'renderer/humans/juno/juno.body.json'))), props: JSON.parse(fs.readFileSync(out))};
} catch (e) { console.log('NOTE juno files unavailable (its cases are skipped):', String(e.message).slice(0, 100)); }

const DT = 1 / 30, N_S = 12, WIN_S = 3, EPS = 0.08;
const LIP = new Set(M.LIPSYNC), LEG = new Set(M.LEGACY);
// the dials the rule covers: everything of the face, head, torso, shoulders, arms, hands, tail (never the lip-sync mouth)
const DIALS = M.RIG_PARAMS.filter(p => !LIP.has(p) && !LEG.has(p) && p !== 'breath' && p !== 'voice');
const DIAL_SET = new Set(DIALS);
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : '')); }

function mk(ch, seed, revert) {
  const ev = [];
  const m = M.createMotion({seed, preset: 'default', onAction: e => ev.push(e), poseRevert: revert});
  if (ch === 'muse') m.setProps(JSON.parse(JSON.stringify(MUSE.props)), JSON.parse(JSON.stringify(MUSE.body)), 'muse');
  else if (ch === 'juno') m.setProps(JSON.parse(JSON.stringify(JUNO.props)), JSON.parse(JSON.stringify(JUNO.body)), 'juno');
  else if (ch === 'flurry' && FLURRY) m.setProps(JSON.parse(JSON.stringify(FLURRY.props)), JSON.parse(JSON.stringify(FLURRY.body)), 'flurry');
  else m.setProps(null, null, ch);
  const face = {present: true, x: 0.05, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
  const T = {m, ev, ch, rows: []};
  T.step = (n = 1, rec = false) => { for (let i = 0; i < n; i++) { const o = m.step(DT, {features: {t: m.now(), face, attending: true, hands: []}}); if (rec) T.rows.push(Object.assign({}, o)); } };
  T.idle = (maxS = 60) => { for (let i = 0; i < maxS * 30; i++) { T.step(1); const E = m.engine && m.engine.E; if (!E || (!E.move && !E.chain && !(E.jobs && E.jobs.length) && !E.queue && !E.oqueue && i > 3)) break; } };
  return T;
}
const O = (verb, name, extra = {}) => Object.assign({action: 'object', verb, object: name ? {name} : undefined}, extra);
// his words -> the commands the page's matcher runs
function said(text, act) { const h = Intent.create(RULES, {act: !!act}).parse(text, {final: true, nowMs: 0}); return h.map(x => x.cmd); }
function quant(a, q) { const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]; }

// ---- the setups (the state that must persist)
const SETUPS = {
  plain: {prep: () => {}, state: () => null},
  // seated on the chair with the mug in a paw (muse)
  seat_prop: {prep: T => { for (const c of [O('sit_in', 'chair'), O('pick_up', 'mug')]) { const r = T.m.command(c, {src: 'test'}); if (!r.ok) throw new Error('setup: ' + (r.reason || c.verb)); T.idle(60); } T.step(30); },
    state: T => { const s = T.m.state(); return {sittingOn: s.sittingOn || null, holding: JSON.stringify(s.holding), x: +s.root.x.toFixed(2), z: +s.root.z.toFixed(2)}; }},
  // standing at the chalkboard, the board view on (muse)
  board: {prep: T => { const r = T.m.command(O('go_to', 'board'), {src: 'test'}); if (!r.ok) throw new Error('setup: ' + r.reason); T.idle(60); try { T.m.engine.setView('board'); } catch (e) {} T.step(45); },
    state: T => { const s = T.m.state(), E = T.m.engine.E; return {posture: s.posture, view: E.view ? (E.view.name || E.view.id || 'on') : null, x: +s.root.x.toFixed(2), z: +s.root.z.toFixed(2), yaw: +s.root.yaw.toFixed(2)}; }},
};

// ---- planner jobs
const STREAM = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'planner_streams', 'stream_kf-sol__m06_jacks.json'), 'utf8'));
const FIX_POSES = STREAM.ref_parse.poses.map(p => Object.assign({t: p.t}, p.set));
const kf = (o) => T => T.m.command(Object.assign({action: 'keyframes', description: 'x', done: true, hold: false}, o), {src: 'voice'});
// adversarial: a dense rig-space program that ends away from rest on every group and says "keep the pose"
const ADV_RIG = []; for (let i = 0; i <= 20; i++) { const u = i / 20; ADV_RIG.push({t: +(i * 0.05).toFixed(2), blink_l: u, blink_r: 0, hand_r_x: 0.2 * u, hand_r_y: 1.1 * u, hand_r_z: 0.2 * u, hand_r_open: u, head_roll: 0.5 * u, tongue_out: u, brow_l: u, brow_r: u, body_lean: 0.6 * u, ear_l: -u, tail_raise: u}); }
// adversarial: the planner's pose language, last pose far from rest, the job told to wait for a cue that never comes
const ADV_KF = [{t: 0.4, right_paw_x: 0.2, right_paw_y: 1.0, head_roll: 0.5, eyes: 1, smile: 1}, {t: 1.2, right_paw_x: 0.3, right_paw_y: 1.1, left_paw_y: 1.0, head_roll: -0.5, eyes: 1}, {t: 17.5, right_paw_y: 1.1, left_paw_y: 1.1, eyes: 1, tongue: 1}];

// ---- the cases: [label, character, setup, run(T), opts]
const W = (text) => T => { const cs = said(text, T.ch === 'muse' || T.ch === 'juno'); if (!cs.length) throw new Error('the matcher read nothing from: ' + text); let ok = false; for (const c of cs) { const r = T.m.command(c, {src: 'matcher'}); ok = ok || !!(r && r.ok !== false); } if (!ok) throw new Error('no command ran for: ' + text); };
const CASES = [
  ['"close your left eye"', 'fox', 'plain', W('close your left eye')],
  ['"close your eyes"', 'fox', 'plain', W('close your eyes')],
  ['"stick your tongue out"', 'fox', 'plain', W('stick your tongue out')],
  ['"raise your hand"', 'fox', 'plain', W('raise your hand')],
  ['"cover your mouth"', 'fox', 'plain', W('cover your mouth')],
  ['"look left"', 'fox', 'plain', W('look left')],
  ['"tilt your head"', 'fox', 'plain', W('tilt your head')],
  ['"cross your arms"', 'fox', 'plain', W('cross your arms')],
  ['"cover your eyes"', 'fox', 'plain', W('cover your eyes')],
  ['"lean in"', 'fox', 'plain', W('lean in')],
  ['"put your ears down"', 'fox', 'plain', W('put your ears down')],
  ['"raise your eyebrows"', 'fox', 'plain', W('raise your eyebrows')],
  ['"look angry and hold it"', 'fox', 'plain', W('look angry and hold it')],
  ['"wave and hold it"', 'fox', 'plain', W('wave and hold it')],
  ['"point up" then "freeze"', 'fox', 'plain', T => { W('point up')(T); T.step(30); W('freeze')(T); }],
  ['"open your mouth" + "squint" + "flare your nostrils"', 'fox', 'plain', T => { W('open your mouth')(T); T.step(10); W('squint')(T); T.step(10); W('flare your nostrils')(T); }],
  ['tool pose: hold:true, 10 minutes asked (bounded)', 'fox', 'plain', T => T.m.command({action: 'pose', targets: {blink_l: 1, blink_r: 1, hand_l_y: 1.1, head_pitch: 0.5}, hold: true, duration_ms: 600000}, {src: 'tool'}), {nS: 66}],
  ['held gesture, no end (think, hold:true)', 'fox', 'plain', T => T.m.command({action: 'gesture', name: 'think', hold: true}, {src: 'tool'})],
  ['planner job: recorded stream (jumping jacks)', 'fox', 'plain', kf({id: 'k1', poses: FIX_POSES})],
  ['ADVERSARIAL planner job: ends far from rest, hold:true never released', 'fox', 'plain', kf({id: 'k2', poses: ADV_KF, hold: true}), {nS: 30}],
  ['ADVERSARIAL semantic program: keep_pose on every group', 'fox', 'plain', kf({id: 'k3', poses: ADV_RIG, space: 'rig', hz: 20, keep_pose: true})],
  ['planner job frozen mid-play ("hold that")', 'fox', 'plain', T => { kf({id: 'k4', poses: FIX_POSES})(T); T.step(45); T.m.command({action: 'hold', parts: 'all'}, {src: 'matcher'}); }, {nS: 20}],
  ['muse, seated + mug in paw: "close your eyes"', 'muse', 'seat_prop', W('close your eyes')],
  ['muse, seated + mug in paw: "tilt your head" + "stick your tongue out"', 'muse', 'seat_prop', T => { W('tilt your head')(T); T.step(8); W('stick your tongue out')(T); }],
  ['muse, seated + mug in paw: ADVERSARIAL keep_pose program (face + free arm)', 'muse', 'seat_prop', kf({id: 'k5', poses: ADV_RIG.map(p => ({t: p.t, blink_l: p.blink_l, head_roll: p.head_roll, tongue_out: p.tongue_out, brow_l: p.brow_l, brow_r: p.brow_r})), space: 'rig', hz: 20, keep_pose: true})],
  ['muse at the board: "raise your hand" + "look left"', 'muse', 'board', T => { W('raise your hand')(T); T.step(8); W('look left')(T); }],
  ['muse at the board: "cover your eyes"', 'muse', 'board', W('cover your eyes')],
];
if (JUNO) CASES.push(['juno: "close your left eye"', 'juno', 'plain', W('close your left eye')], ['juno: "cross your arms"', 'juno', 'plain', W('cross your arms')],
  ['juno: ADVERSARIAL keep_pose program', 'juno', 'plain', kf({id: 'k6', poses: ADV_RIG, space: 'rig', hz: 20, keep_pose: true})]);
if (FLURRY) CASES.push(['flurry: "stick your tongue out" + "raise your eyebrows"', 'flurry', 'plain', T => { W('stick your tongue out')(T); T.step(8); W('raise your eyebrows')(T); }]);

// the idle twins' band per (character, setup): same state, no pose command; the statistic is the one the case is judged by
// (the median over a 3 s window), over every window of 5 seeds x 50 s
const BANDS = new Map();
function band(ch, setup) {
  const key = ch + '|' + setup; if (BANDS.has(key)) return BANDS.get(key);
  const b = {}; for (const p of DIALS) b[p] = [Infinity, -Infinity];
  for (const seed of [101, 202, 303, 404, 23]) {
    const T = mk(ch, seed, true); T.step(90); SETUPS[setup].prep(T); T.step(60);
    T.rows = []; T.step(50 * 30, true);
    const W = WIN_S * 30;
    const take = rows => { for (const p of DIALS) { const a = rows.map(r => +r[p] || 0); for (let i = 0; i + W <= a.length; i += 15) { const m = quant(a.slice(i, i + W), 0.5); if (m < b[p][0]) b[p][0] = m; if (m > b[p][1]) b[p][1] = m; } } };
    take(T.rows);
    // ... and the same time after an interaction that poses nothing (a command resets its boredom: ears, tail, glances)
    for (let k = 0; k < 3; k++) { T.m.command({action: 'gesture', name: 'blink'}, {src: 'test'}); T.step(N_S * 30); T.rows = []; T.step(WIN_S * 30, true); take(T.rows); }
  }
  BANDS.set(key, b); return b;
}
function owning(T) {      // what a command still drives on a dial
  const out = [], ow = T.m.owned ? T.m.owned() : {};
  for (const p in ow) if (DIAL_SET.has(p) && ow[p] > 0.02) out.push(p);
  const s = T.m.state();
  for (const g of s.held || []) out.push('held:' + g);
  for (const a of s.active || []) if (a.kind === 'look' && a.held) out.push('look:' + a.name); else if (a.kind === 'kf') out.push('kf:' + a.id); else if (a.kind === 'gesture' || a.kind === 'expression') out.push(a.kind + ':' + a.name);
  const k = T.m.kfState ? T.m.kfState() : null; if (k && /^(holding|playing|buffering|ready)$/.test(k.phase)) out.push('kf-' + k.phase);
  return out;
}
function runCase(c, revert) {
  const [label, ch, setup, run, opt] = c, nS = (opt && opt.nS) || N_S;
  const T = mk(ch, 23, revert), B = band(ch, setup);
  T.step(90); SETUPS[setup].prep(T); T.step(60);
  const st0 = SETUPS[setup].state(T), own0 = new Set(owning(T));   // what the STATE itself drives (the paw carrying the mug)
  let err = null; try { run(T); } catch (e) { err = String(e.message || e); }
  if (err) return {label, ch, error: err, latched: true};
  // while it holds: how far the dials got (the case must really move something)
  let tRest = null, moved = 0; const base = {}; for (const p of DIALS) base[p] = (B[p][0] + B[p][1]) / 2;
  for (let i = 0; i < nS * 30; i++) {
    T.step(1, true); const o = T.rows[T.rows.length - 1];
    for (const p of DIALS) moved = Math.max(moved, Math.abs((+o[p] || 0) - base[p]) - (B[p][1] - B[p][0]) / 2);
    const ow = owning(T).filter(x => !own0.has(x)); if (ow.length) tRest = null; else if (tRest === null) tRest = (i + 1) / 30;
  }
  const own = owning(T).filter(x => !own0.has(x));
  T.rows = []; T.step(WIN_S * 30, true);
  const off = [];
  for (const p of DIALS) { const med = quant(T.rows.map(r => +r[p] || 0), 0.5); if (med < B[p][0] - EPS || med > B[p][1] + EPS) off.push(`${p} ${med.toFixed(2)} (idle ${B[p][0].toFixed(2)}..${B[p][1].toFixed(2)})`); }
  const blinkMin = Math.min(...T.rows.map(r => Math.max(r.blink_l, r.blink_r)));
  const st1 = SETUPS[setup].state(T);
  const stateKept = JSON.stringify(st0) === JSON.stringify(st1);
  const latched = own.length > 0 || off.length > 0 || blinkMin > 0.5;
  return {label, ch, setup, n_s: nS, moved: +moved.toFixed(2), latched, owned: own.slice(0, 8), off: off.slice(0, 6), blink_min: +blinkMin.toFixed(2), rest_s: latched ? null : tRest, state0: st0, state1: st1, stateKept,
    reverts: T.ev.filter(e => e.kind === 'release' && e.name === 'auto').length};
}

const hasSwitch = typeof M.createMotion({seed: 1}).poseRevert === 'function';
const before = CASES.map(c => runCase(c, false));
const after = hasSwitch ? CASES.map(c => runCase(c, true)) : null;
console.log(`\n${CASES.length} pose commands; every dial unowned and back in the idle band within N s (default ${N_S} s), state kept`);
console.log('case'.padEnd(74) + 'before (switch off)'.padEnd(22) + 'after');
for (let i = 0; i < CASES.length; i++) {
  const b = before[i], a = after ? after[i] : null;
  const f = r => r.error ? 'ERROR ' + r.error.slice(0, 40) : r.latched ? 'LATCHED' : 'rest in ' + (r.rest_s != null ? r.rest_s.toFixed(1) + ' s' : '?');
  console.log(('  ' + b.label).slice(0, 72).padEnd(74) + f(b).padEnd(22) + (a ? f(a) + (a.stateKept ? '' : '  STATE CHANGED') : '-'));
}
const nb = before.filter(r => r.latched).length, na = after ? after.filter(r => r.latched).length : null;
console.log(`\nLATCHED: before ${nb}/${CASES.length}` + (after ? `, after ${na}/${CASES.length}` : ' (this tree has no pose-revert switch: the before number only)'));
if (after) {
  for (const a of after) {
    check(`${a.label} [${a.ch}]: every dial back at rest, owned by the reflex layers, within ${a.n_s} s`, !a.error && !a.latched, a.error || {owned: a.owned, off: a.off, blink_min: a.blink_min});
    check(`${a.label} [${a.ch}]: the command really moved a dial (${a.moved})`, a.error || a.moved > 0.1, a.moved);
    if (a.setup !== 'plain') check(`${a.label}: state persists (${JSON.stringify(a.state1)})`, !!a.stateKept, {before: a.state0, after: a.state1});
  }
  check(`at least 12 varied pose commands (${CASES.length}), incl. adversarial scripts that never return`, CASES.length >= 12 && CASES.filter(c => /ADVERSARIAL/.test(c[0])).length >= 2);
  check(`with the switch off the old behaviour is back: ${nb} of ${CASES.length} stay latched`, nb >= 12, nb);
  const rs = after.filter(a => a.rest_s != null).map(a => a.rest_s);
  console.log(`time to rest (after): median ${quant(rs, 0.5).toFixed(1)} s, max ${Math.max(...rs).toFixed(1)} s`);

  // ---- a named duration is honoured (and bounded); an open-ended one is bounded
  {
    const T = mk('fox', 5, true); T.step(90);
    T.m.command({action: 'pose', targets: {blink_l: 1, blink_r: 1}, hold: true, duration_ms: 10000}, {src: 'matcher'});
    T.step(8 * 30, true); const at8 = Math.min(T.rows[T.rows.length - 1].blink_l, T.rows[T.rows.length - 1].blink_r);
    T.rows = []; T.step(5 * 30, true); const open13 = Math.min(...T.rows.slice(-30).map(r => Math.max(r.blink_l, r.blink_r)));
    check(`"for ten seconds": still shut at 8 s (${at8.toFixed(2)}), open again by 13 s (${open13.toFixed(2)})`, at8 > 0.9 && open13 < 0.3);
    check('askedHoldS reads a named duration from his words, and none from "a second" / no duration',
      M.askedHoldS('keep your eyes closed for ten seconds') === 10 && M.askedHoldS('hold it for 2 minutes') === 120 && M.askedHoldS('for half a minute') === 30 &&
      M.askedHoldS('close your eyes for a second') === null && M.askedHoldS('raise your hand') === null && M.askedHoldS('wait for me') === null && M.askedHoldS('stay like that until I say stop') > 60);
  }
  // ---- a behaviour program he started is not a pose: it keeps running (the program layer sits under commands)
  {
    const T = mk('fox', 9, true); T.step(60);
    let n = 0; T.m.setProgramLayer(() => { n++; return {set: {ear_l: -1, ear_r: -1}}; });
    T.step(20 * 30, true);
    check(`a running behaviour program is still being asked every frame after 20 s (${n} frames)`, n >= 20 * 30 - 2 && T.m.programLayer === true);
    T.m.setProgramLayer(null);
  }
  // ---- "relax" still works at once, and the action log hears the automatic release
  {
    const T = mk('fox', 6, true); T.step(60); W('close your eyes')(T); T.step(30);
    T.m.command({action: 'release', parts: 'all'}, {src: 'matcher'}); T.step(30, true);
    check('an explicit release still lets a held pose go at once', owning(T).length === 0);
    const T2 = mk('fox', 6, true); T2.step(60); W('close your eyes')(T2); T2.step(8 * 30);
    const ev = T2.ev.filter(e => e.kind === 'release' && e.name === 'auto');
    check(`the automatic release is reported once as an action event (${ev.length}: "${ev[0] && ev[0].did}")`, ev.length === 1 && ev[0].src === 'auto' && ev[0].phase === 'end');
  }
  // ---- the switch at run time
  {
    const T = mk('fox', 4, true); T.m.setPoseRevert(false); T.step(60); W('close your eyes')(T); T.step(15 * 30, true);
    check('setPoseRevert(false): the pose stays held (the old behaviour)', owning(T).length > 0);
    T.m.setPoseRevert(true); T.step(30, true);
    check('setPoseRevert(true): the overdue hold is let go at once', (T.step(60, true), owning(T).length === 0));
  }
}
const ji = process.argv.indexOf('--json');
if (ji > 0) fs.writeFileSync(process.argv[ji + 1], JSON.stringify({tree: WT, cases: CASES.length, latched_before: nb, latched_after: na, before, after}, null, 1));
console.log(`\npose_revert_test: ${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
