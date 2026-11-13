// node tools/plan_sim.cjs <plan.json> <outdir> [--character muse] [--seed 4286]
// Planner scorer, stage 2: play one planner answer on app/motion.js exactly as the page would and write
//   <outdir>/rig.jsonl   render_timeline format, 30 fps, already TRIMMED: starts PRE s before the motion starts, ends SETTLE s
//                        after it stops (header: rig_params + trim info)
//   <outdir>/sim.json    first_motion_ms (from the request), motion_s, energy, at_rest_end, unknown controls, ranges ...
// plan.json (tools/plan_eval.py): {source, ok, engine:[cmd...], poses:[{t, <control>: v, ms: arrival ms}], ms_done}
//   planner / cache: m.command({action:'keyframes', id, description, poses:[], done:false, hold:false}) at t = 0, then every
//                    pose fed with m.kfFeed at the frame its batch arrived, done at ms_done (motion.js buffer rule decides
//                    when playback starts)
//   library:         each engine command through m.command(cmd, {src:'tool'}) at ms_done
// Motion is measured against a REFERENCE run (same seed, a held keyframe job that never plays = the same boredom reset, the
// tests/kf_test.cjs trick), so idle life (blinks, breath, tail) does not count as motion.
'use strict';
const fs = require('fs'), path = require('path');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
// --root <dir>: the source tree (app/ + renderer/) that is being scored; default = this worktree
const ROOTDIR = path.resolve(opt('--root', path.join(__dirname, '..')));
const load = rel => { const mod = {exports: {}}; new Function('module', 'exports', fs.readFileSync(path.join(ROOTDIR, rel), 'utf8'))(mod, mod.exports); return mod.exports; };
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const ANIM = path.join(ROOTDIR, 'renderer', 'animals');
const VALOPT = new Set(['--root', '--character', '--seed']);
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && VALOPT.has(args[i - 1])));
if (pos.length < 2) { console.error('usage: node tools/plan_sim.cjs <plan.json> <outdir>'); process.exit(2); }
const plan = JSON.parse(fs.readFileSync(pos[0], 'utf8')), outdir = pos[1];
const CHAR = opt('--character', plan.character || 'muse'), SEED = +opt('--seed', 4286);
const FPS = 30, DT = 1 / FPS, WARM = 30, PRE = 0.3, SETTLE = 0.7, CAP_S = 40, QUIET_S = 1.0, MIN_AFTER_S = 1.5, IDLE_S = 3.5;
const EPS = 0.02;                       // a channel differs from the reference run by more than this = it is being moved
const VEL = 0.004;                      // per frame: below this (every channel) = still
const HELDKIND = /^(expression|pose|hold|release|look|stop|config)$/;   // commands that set a state and stay: never 'busy'
const HOLD_TAIL_S = 3.2;               // a clip that ends away from rest (a held pose / face) shows it held this long
const HIDE_Y = -40;                     // prop<k>_y of an absent thing
const NOCOUNT = /^(cam_|gaze_free$|prop\d|breath$)/;   // the camera follows; not body motion

// the row's scene: the objects that are there (null = the character's whole scene file; [] = an empty stage). The engine
// keeps its real configuration (the muse always has its room: an engine with no props is not a state the app has); a thing
// that is NOT there is (a) taken out of what the page sees (motion.props(), the matcher's scene, the aliases), (b) refused by
// name when an engine `object` command asks for it ("there is no chair here", the engine's own words for a missing thing),
// and (c) moved out of the picture.
const SCENE = Array.isArray(plan.scene) ? plan.scene.map(x => String(x).toLowerCase()) : null;
function sceneProps() {
  if (CHAR !== 'muse') return null;
  const full = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json')));
  const here = JSON.parse(JSON.stringify(full));
  let absent = [];
  if (SCENE) {
    here.props = full.props.filter(p => SCENE.includes(String(p.id).toLowerCase()));
    absent = full.props.filter(q => !here.props.some(p => p.id === q.id)).map((q, i) => ({id: q.id, name: q.name || q.id, slot: Number.isInteger(q.slot) ? q.slot : full.props.indexOf(q),
      words: [q.id, ...(q.aliases || [])].map(w => String(w).toLowerCase().replace(/_/g, ' ')).filter(w => w.length > 2 && !/ one$/.test(w))}));
    if (Array.isArray(here.places)) here.places = here.places.filter(pl => !absent.some(g => String(pl.id) === g.id || String(pl.id).startsWith(g.id + '_')));   // chair_seat, desk_top
  }
  return {full, here, absent};
}
const SP = sceneProps();
const ABSENT = SP ? SP.absent : [];
const ABSENT_SLOTS = ABSENT.map(a => a.slot);
// an engine command that names a thing which is not there -> the absent thing's id
function absentRef(cmd) {
  if (!ABSENT.length || !cmd || cmd.action !== 'object') return null;
  const strs = [];
  (function walk(v, k) { if (typeof v === 'string') { if (k !== 'action' && k !== 'verb') strs.push(' ' + v.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim() + ' '); } else if (v && typeof v === 'object') for (const [kk, vv] of Object.entries(v)) walk(vv, kk); })(cmd, '');
  // a thing that IS there wins when one phrase could be both ("table" = the desk)
  const here = SP.here.props.flatMap(p => [p.id, ...(p.aliases || [])].map(w => String(w).toLowerCase()));
  for (const a of ABSENT) for (const w of a.words) if (!here.includes(w) && strs.some(x => x.includes(' ' + w + ' '))) return a;
  return null;
}
function makeChar() {
  const ev = [];
  const m = M.createMotion({seed: SEED, character: CHAR, onAction: e => ev.push(e)});
  if (CHAR === 'muse') m.setProps(JSON.parse(JSON.stringify(SP.full)), JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json'))), 'muse');
  else m.setProps(null, null, CHAR);
  const cmd = (c, o) => { const a = absentRef(c); return a ? {ok: false, reason: `there is no ${a.id} here`, absent: a.id} : m.command(c, o); };
  let t = 0;
  const step = () => { t += DT; return Object.assign({}, m.step(DT, {features: {t, face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}})); };
  return {m, ev, step, cmd};
}

const cleanPoses = pl => ((pl && pl.poses) || []).filter(p => p && typeof p === 'object' && Number.isFinite(+p.t)).map(p => ({ms: +p.ms || 0, pose: Object.fromEntries(Object.entries(p).filter(([k]) => k !== 'ms'))}));
const poses = cleanPoses(plan);
const sim = {id: plan.id, source: plan.source || null, character: CHAR, root: ROOTDIR, scene: SCENE, hidden_slots: ABSENT_SLOTS, n_poses: poses.length, plan_ok: plan.ok !== false, errors: [], commands: []};
const HELD = [{ms: 0, fn: C => C.m.command({action: 'keyframes', id: 'ref', description: 'ref', poses: [], done: false, hold: true}, {src: 'tool'})}];

// one plan as timed actions on a character: {ms, fn(C)}; instant = every pose at once (a setup: not scored). -> {acts, endMs}
function actionsOf(pl, atMs, id, instant, log) {
  const acts = [], P = cleanPoses(pl);
  const dMs = instant ? 0 : (Number.isFinite(+pl.ms_done) ? +pl.ms_done : (P.length ? P[P.length - 1].ms : 0));
  if ((pl.source === 'library' || pl.source === 'page') && Array.isArray(pl.engine) && pl.engine.length) {
    acts.push({ms: atMs + dMs, fn: C => { for (const cmd of pl.engine) {
      let r; try { r = C.cmd(JSON.parse(JSON.stringify(cmd)), {src: 'tool'}); } catch (e) { r = {ok: false, reason: 'exception: ' + String(e && e.message || e).slice(0, 120)}; }
      if (log) { sim.commands.push({cmd, ok: !!(r && r.ok), reason: r && r.reason, did: r && r.did}); if (!r || !r.ok) sim.errors.push('command refused: ' + JSON.stringify(cmd).slice(0, 100) + ' — ' + (r && r.reason)); }
    } }});
  } else if (P.length) {
    let ok = true;
    acts.push({ms: atMs, fn: C => { const r = C.m.command({action: 'keyframes', id, description: String(pl.text || pl.description || 'move'), poses: [], done: false, hold: false}, {src: 'tool'});
      ok = !!(r && r.ok); if (log) { sim.commands.push({cmd: 'keyframes', ok, reason: r && r.reason}); if (!ok) sim.errors.push('keyframes refused: ' + (r && r.reason)); } }});
    const by = new Map(); for (const p of P) { const k = instant ? 0 : p.ms; if (!by.has(k)) by.set(k, []); by.get(k).push(p.pose); }
    const ks = [...by.keys()].sort((x, y) => x - y);
    ks.forEach((k, i) => acts.push({ms: atMs + k, fn: C => { if (ok) { try { C.m.kfFeed(id, by.get(k), i === ks.length - 1 && dMs <= k); } catch (e) { if (log) sim.errors.push('kfFeed: ' + String(e.message).slice(0, 120)); } } }}));
    if (dMs > ks[ks.length - 1]) acts.push({ms: atMs + dMs, fn: C => { if (ok) { try { C.m.kfFeed(id, [], true); } catch (e) {} } }});
  }
  return {acts, endMs: atMs + dMs};
}
// run a character and its reference in lockstep; motion = the character differs from / changes against the reference
function runPair(acts, refActs, o) {
  const C = makeChar(), R = makeChar();
  for (let i = 0; i < WARM; i++) { C.step(); R.step(); }
  const names = C.m.allParams, count = names.filter(n => !NOCOUNT.test(n));
  const A = acts.slice().sort((x, y) => x.ms - y.ms), B = refActs.slice().sort((x, y) => x.ms - y.ms);
  const rows = [], diffs = [], maxDiff = {};
  const open = new Set(); let evI = 0, lastBusy = 0;
  let ia = 0, ib = 0, lastActive = -1, firstActive = -1, energy = 0, prev = null, stillSince = 0, state = null;
  const N = CAP_S * FPS, from = Math.round((o.fromMs || 0) / 1000 * FPS), lastIn = A.length ? A[A.length - 1].ms : 0, minEnd = Math.max(lastIn + MIN_AFTER_S * 1000, o.minEndMs || 0);
  for (let i = 0; i < N; i++) {
    const ms = i * DT * 1000;
    if (o.stateAtMs != null && state === null && ms >= o.stateAtMs) { try { state = JSON.parse(JSON.stringify(C.m.state())); } catch (e) { state = {}; } }
    while (ia < A.length && A[ia].ms <= ms) A[ia++].fn(C);
    while (ib < B.length && B[ib].ms <= ms) B[ib++].fn(R);
    const row = C.step(), ref = R.step();
    rows.push(row);
    let d = 0, v = 0; const rel = {};
    for (const n of count) {
      const x = (row[n] || 0) - (ref[n] || 0); rel[n] = x;
      const ax = Math.abs(x); if (ax > d) d = ax;
      if (i >= from && ax > (maxDiff[n] || 0)) maxDiff[n] = ax;
      if (prev) { const dv = Math.abs(x - prev[n]); if (dv > v) v = dv; if (i >= from) energy += dv; }
    }
    prev = rel; diffs.push(d);
    if (i >= from) {
      if (firstActive < 0 && d > EPS) { firstActive = i; lastActive = i; }
      if (v > VEL) { if (firstActive >= 0) lastActive = i; stillSince = 0; } else stillSince++;
    }
    // busy = a keyframe job / engine job / commanded gesture is running (idle life later on must not keep the clip open)
    let live = false, heldNow = false; try { const act = C.m.state().active || []; heldNow = act.some(x => x.held || x.frozen); live = act.some(x => x.timed || (x.kind === 'kf' ? !x.frozen : !HELDKIND.test(x.kind || '') && x.kind !== 'gesture' && !x.held)); } catch (e) {}
    for (; evI < C.ev.length; evI++) { const e = C.ev[evI]; if ((e.src !== 'tool' && e.src !== 'cmd') || e.kind === 'kf' || HELDKIND.test(e.kind || '')) continue; const k = e.kind + ':' + e.name + ':' + (e.id || '');
      if (e.phase === 'start') open.add(k); else if (e.phase === 'end' || e.phase === 'interrupted' || e.phase === 'fail') { open.delete(k); for (const q of [...open]) if (q.startsWith(e.kind + ':' + e.name + ':')) open.delete(q); } }
    if (heldNow) open.clear();          // a frozen job never reports its end
    if (live || open.size || ia < A.length) lastBusy = i;
    const idle = (i - lastBusy) * DT;
    if (ia >= A.length && !live && ms >= minEnd && (stillSince >= (d > EPS * 2 ? HOLD_TAIL_S : QUIET_S) * FPS || idle >= IDLE_S) && (firstActive >= 0 || ms >= lastIn + 4000)) break;
  }
  // idle life (boredom gestures) differs between the two runs long after a command: motion ends with the last busy frame
  if (lastActive > lastBusy + Math.round(0.6 * FPS) && rows.length - 1 - lastBusy >= IDLE_S * FPS - 1) lastActive = lastBusy + Math.round(0.6 * FPS);
  if (process.env.SIM_DEBUG) console.error(JSON.stringify({active: C.m.state().active, open: [...open], lastBusy, lastActive, n: rows.length, stillSince}));
  return {C, rows, diffs, maxDiff, firstActive, lastActive, lastBusy, energy, names, state, truncated: rows.length >= N};
}

/* --route: what the PAGE does with the request before any server planner call. The real app/voicetools.js do_action route
   (sceneRoute / noThingHere / decline / page library / performer) on the real motion.js in the row's scene, in the state the
   setup left; description = the user's words = the utterance (no voice model in the loop). Every engine command the route runs is
   recorded (it is replayed in the scored run); a call that reaches the performer is the server planner's turn.
   -> {route: 'planner' | 'page', status, ok, reason, cmds: [...], state (what the page sends the server), caption, request_at_ms} */
function routeProbe() {
  const VTL = load('app/voicetools.js'), IL = load('app/intent.js'), Intent = IL.Intent || IL;
  const RULES = JSON.parse(fs.readFileSync(path.join(ROOTDIR, 'app/intent_rules.json'), 'utf8'));
  const text = String(plan.text || '').trim();
  const C = makeChar();
  for (let i = 0; i < WARM; i++) C.step();
  const A = setupActs.slice().sort((x, y) => x.ms - y.ms); let ia = 0;
  for (let i = 0; i * DT * 1000 < reqMs + 1; i++) { while (ia < A.length && A[ia].ms <= i * DT * 1000) A[ia++].fn(C); C.step(); }
  const act = CHAR === 'muse', P = SP ? SP.here : null;
  const hereIds = new Set(P ? P.props.map(p => p.id) : []);
  const mview = new Proxy(C.m, {get(t, k) { if (k === 'props') return () => (t.props() || []).filter(p => hereIds.has(p.id)); if (k === 'command') return (c, o) => C.cmd(c, o); return t[k]; }});
  let Mx = null; try { Mx = Intent.create(RULES, {act, scene: act && P ? P : null}); } catch (e) { Mx = null; }
  const cmds = [], perf = [], notes = [], logs = [];
  let t = 1e6;
  const run = (cmd, via) => { let r; try { r = C.cmd(JSON.parse(JSON.stringify(cmd)), {src: 'voice'}); } catch (e) { r = {ok: false, reason: 'exception: ' + String(e && e.message || e).slice(0, 120)}; }
    cmds.push({cmd, via, ok: !!(r && r.ok !== false), reason: r && r.reason, did: r && r.did}); return r; };
  const library = (desc, move) => {                       // app/page.html voiceLibrary
    let cs = [];
    if (move && move.name) return null;                    // no voice model here: never a named move
    if (!Mx) return null;
    const hits = Mx.parse(String(desc || ''), {final: true, nowMs: t});
    if (!hits.length || (Mx.understood && !Mx.understood(String(desc || '')))) return null;
    cs = hits.map(h => h.cmd);
    if (!act && cs.some(c => !/^(gesture|pose|expression|look|release)$/.test(c.action || ''))) return null;
    if (!cs.length) return null;
    let ran = 0, reason = ''; const did = [];
    for (const c of cs) { const r = run(c, 'library'); if (r && (r === true || r.ok !== false)) { ran++; if (r.did) did.push(r.did); } else reason = (r && r.reason) || reason; }
    return {ran, dup: 0, reason: ran ? '' : (reason || 'the body could not do it'), did: did.join(', '), held: !!ran && cs.some(c => c && (c.hold === true || c.hold === 'true'))};
  };
  const bodyState = () => { let s = null; try { s = C.m.state(); } catch (e) {} return s ? {posture: s.posture || null, holding: s.holding || null, wearing: s.wearing || [], seated: s.posture === 'seated' || !!s.sittingOn} : {}; };
  const state = bodyState();
  const vt = VTL.createVoiceTools({
    perf: () => ({call: (d, info) => { perf.push({description: d, words: info && info.words}); return {ok: true, status: 'preparing', id: 'p' + perf.length}; }}),
    motion: () => mview, runCommand: (cmd) => run(cmd, 'command'), runStep: (cmd) => run(cmd, 'step'),
    sceneMatcher: () => (act ? Mx : null), moveMatcher: () => (act ? Mx : null),
    sceneAliases: () => (P ? P.props.map(p => ({id: p.id, aliases: p.aliases || []})) : []), scenery: () => (P && Array.isArray(P.scenery) ? P.scenery : []),
    runLibrary: (desc, move) => library(desc, move), coveredByMatcher: () => false,
    prompt: (text, o) => { notes.push({text: String(text).slice(0, 300), respond: !(o && o.respond === false)}); return {id: 'n' + notes.length}; },
    sendCtl: () => {}, sendEvent: () => null, userSpeaking: () => false, avatarSpeaking: () => false, lastUserSpeech: () => t - 800,
    character: () => CHAR, log: r => logs.push(r), now: () => t, timers: {set: () => 1, clear: () => {}}});
  let res;
  try { vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText(text, true); t += 300; res = vt.call('do_action', {description: text}, 'done', {}); }
  catch (e) { res = {ok: false, status: 'route_error', reason: 'route exception: ' + String(e && e.stack || e).slice(0, 300)}; }
  res = res && typeof res === 'object' ? res : {ok: false, reason: 'no result'};
  const out = {request_at_ms: reqMs, setup: setupInfo, state, status: res.status || null, ok: res.ok !== false, reason: res.reason || null, did: res.did || null,
    cmds, perf, notes: notes.slice(0, 4), log: logs.slice(0, 8), props_here: P ? P.props.map(p => p.id) : []};
  if (perf.length) out.route = 'planner';
  else if (res.status === 'use_object' || res.status === 'route_error') out.route = 'hub';   // live: the voice model's `object` call; here the server's own library / planner reads the words
  else out.route = 'page';
  if (out.route === 'page' && res.ok === false) {
    // what the voice model is told, in short: the judge gets it as the words the character says
    const r = String(res.reason || '').replace(/\([^)]*\)/g, '').replace(/\s+/g, ' ');
    out.caption = r.split(/, so (?:nothing|that) was(?: not)? done|\. Tell him|: move it with/)[0].trim().slice(0, 200);
  }
  return out;
}

// ---- the run. A setup (hold / undo / "bigger" items) plays first, unscored; the request then comes while / after it.
let reqMs = 0, setupActs = [], setupInfo = null;
if (plan.setup && typeof plan.setup === 'object') {
  const S0 = actionsOf(plan.setup, 0, 'k0', true, false);
  const pre = runPair(S0.acts, HELD, {});
  const f0 = pre.firstActive, f1 = pre.lastActive, when = String(plan.setup.when || 'end');
  if (f0 < 0) { reqMs = 1500; sim.errors.push('setup produced no motion'); }
  else if (when === 'mid') reqMs = Math.round((f0 + Math.min(3.0 * FPS, Math.max(1.0 * FPS, 0.6 * (f1 - f0)))) * DT * 1000);
  else reqMs = Math.round((f1 + 0.3 * FPS) * DT * 1000);
  setupActs = S0.acts;
  setupInfo = {text: plan.setup.text, source: plan.setup.source, when, moved: f0 >= 0, motion_ms: f0 >= 0 ? [Math.round(f0 * DT * 1000), Math.round(f1 * DT * 1000)] : null, request_at_ms: reqMs};
  if (opt('--state-at-request', null) !== null || args.includes('--state-at-request')) {
    const st = runPair(S0.acts, HELD, {stateAtMs: reqMs, minEndMs: reqMs + 100}).state || {};
    console.log(JSON.stringify({request_at_ms: reqMs, setup: setupInfo, state: st}));
    process.exit(0);
  }
} else if (args.includes('--state-at-request')) { console.log(JSON.stringify({request_at_ms: 0, state: null})); process.exit(0); }
if (args.includes('--route')) { console.log(JSON.stringify(routeProbe())); process.exit(0); }
const main = actionsOf(plan, reqMs, 'k1', false, true);
const noInput = !main.acts.length;
const SETUP_MIN_S = 6.0;               // a setup item's clip runs at least this long after the request (holds must be seen held)
const run = runPair(setupActs.concat(noInput && !setupActs.length ? [{ms: 0, fn: C => C.m.command({action: 'keyframes', id: 'k1', description: 'none', poses: [], done: false, hold: true}, {src: 'tool'})}] : main.acts),
  setupActs.length ? setupActs : HELD, {fromMs: reqMs, minEndMs: setupActs.length ? reqMs + SETUP_MIN_S * 1000 : 0});
const {C, rows, diffs, maxDiff, names, energy} = run;
const reqF = Math.round(reqMs / 1000 * FPS);
const kfEv = C.ev.filter(e => e.kind === 'kf' && String(e.id) === 'k1');
const evMs = e => e ? Math.round((e.t - WARM * DT) * 1000 - reqMs) : null;
const startEv = kfEv.find(e => e.phase === 'start'), endEv = kfEv.find(e => ['end', 'interrupted', 'fail'].includes(e.phase));
const moved = run.firstActive >= 0 && Math.max(...diffs.slice(reqF)) > EPS * 2;
const f0 = moved ? run.firstActive : reqF, f1 = moved ? Math.min(rows.length - 1, run.lastActive) : Math.min(rows.length - 1, reqF + 60);
// the clip: PRE s before the motion starts (a setup item: 1 s before the request, so the state it starts from is seen)
const a = setupActs.length ? Math.max(0, reqF - FPS) : Math.max(0, f0 - Math.round(PRE * FPS));
const b = setupActs.length ? Math.max(Math.min(rows.length - 1, f1 + Math.round(SETTLE * FPS)), Math.min(rows.length - 1, reqF + Math.round(SETUP_MIN_S * FPS))) : Math.min(rows.length - 1, f1 + Math.round(SETTLE * FPS));
const endDiff = diffs[rows.length - 1];
const heldEnd = !setupActs.length && moved && endDiff >= EPS * 2;
const b2 = heldEnd ? Math.max(b, Math.min(rows.length - 1, f1 + Math.round(HOLD_TAIL_S * FPS))) : b;
// what the plan asked of each control
const KF_ALIAS = C.m.KF_ALIAS || {};
const known = new Set(Object.keys(KF_ALIAS)), ranges = {}, unknown = {};
for (const {pose} of poses) for (const [c, v] of Object.entries(pose)) {
  if (c === 't') continue;
  if (!known.has(c)) { unknown[c] = (unknown[c] || 0) + 1; continue; }
  if (!Number.isFinite(+v)) continue;
  const r = ranges[c] || (ranges[c] = [+v, +v]); r[0] = Math.min(r[0], +v); r[1] = Math.max(r[1], +v);
}
const ti = n => names.indexOf(n);
Object.assign(sim, {
  moved, first_motion_ms: moved ? Math.round(f0 * DT * 1000 - reqMs) : null, kf_start_ms: evMs(startEv), kf_end_ms: evMs(endEv), kf_end_phase: endEv ? endEv.phase : null,
  kf_fail_reason: endEv && endEv.phase !== 'end' ? endEv.reason : undefined,
  last_motion_ms: moved ? Math.round(f1 * DT * 1000 - reqMs) : null, motion_s: moved ? +((f1 - f0 + 1) * DT).toFixed(3) : 0,
  planned_s: poses.length ? Math.max(...poses.map(p => +p.pose.t || 0)) : null,
  energy: +energy.toFixed(3), peak_diff: +Math.max(...diffs).toFixed(4), at_rest_end: setupActs.length ? null : endDiff < EPS * 2, end_diff: +endDiff.toFixed(4),
  truncated: run.truncated, setup: setupInfo, sim_frames: rows.length,
  unknown_controls: unknown, control_ranges: Object.fromEntries(Object.entries(ranges).map(([c, r]) => [c, [+r[0].toFixed(3), +r[1].toFixed(3)]])),
  rig_moved: Object.fromEntries(Object.entries(maxDiff).filter(([, v]) => v > EPS).sort((x, y) => y[1] - x[1]).slice(0, 40).map(([c, v]) => [c, +v.toFixed(3)])),
  face_only: moved && !Object.keys(maxDiff).some(c => maxDiff[c] > 0.05 && /^(root_|foot_|squash|stand)/.test(c)),
  root_y_max: ti('root_y') >= 0 ? +Math.max(...rows.map(r => r.root_y || 0)).toFixed(3) : null,
  clip: {from_frame: a, to_frame: b2, held_end: heldEnd, fps: FPS, clip_s: +((b2 - a + 1) * DT).toFixed(3), motion_starts_at_s: moved ? +((f0 - a) * DT).toFixed(3) : null, request_at_s: +((reqF - a) * DT).toFixed(3), pre_s: PRE, settle_s: SETTLE},
  events: C.ev.filter(e => e.kind !== 'config' && evMs(e) >= -50).slice(0, 40).map(e => ({kind: e.kind, name: e.name, phase: e.phase, ms: evMs(e), did: e.did, reason: e.reason})),
});
fs.mkdirSync(outdir, {recursive: true});
const head = {rig_params: names, fps: FPS, time: 'seconds from the clip start (trimmed: the request was at t = ' + (-(a * DT)).toFixed(3) + ' s)', mode: 'plan_sim', trim_from_ms: Math.round(a * DT * 1000)};
const out = [JSON.stringify(head)];
const hideIx = new Set(ABSENT_SLOTS.map(k => names.indexOf('prop' + k + '_y')).filter(i => i >= 0));   // a thing that is not there: far below the floor
for (let i = a; i <= b2; i++) out.push(JSON.stringify({type: 'rig', t: +((i - a) / FPS).toFixed(4), a: names.map((n, j) => hideIx.has(j) ? HIDE_Y : Math.round((rows[i][n] || 0) * 1e4) / 1e4)}));
fs.writeFileSync(path.join(outdir, 'rig.jsonl'), out.join('\n') + '\n');
fs.writeFileSync(path.join(outdir, 'sim.json'), JSON.stringify(sim, null, 1));
console.log(JSON.stringify({id: sim.id, source: sim.source, moved, first_motion_ms: sim.first_motion_ms, motion_s: sim.motion_s, clip_s: sim.clip.clip_s, at_rest_end: sim.at_rest_end, unknown: Object.keys(unknown), errors: sim.errors}));
