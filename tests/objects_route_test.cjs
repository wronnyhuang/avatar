// node tests/objects_route_test.cjs — OBJECTS planner layer, the page side around the kf player:
//  R  voicetools sceneRoute: a do_action naming a scene thing that the scene matcher does not fully read goes to the
//     planner WITH things (perform.call info.things); objectPlans off / a motion without things() = the refusal;
//     a fully understood scene step still runs as object steps (fast)
//  P  perform.js with the REAL muse motion: the plan request carries state.things + root; the page cache key includes
//     where the things are (same state = page hit, a moved thing = a new plan); a failed "do" step makes the check-in
//     note say what did not work
'use strict';
const load = require('./load.cjs');
const {makeMuse} = require('./muse_lib.cjs');
const VTL = load('app/voicetools.js'), Perf = load('app/perform.js');
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d !== undefined && !ok ? '  — ' + (typeof d === 'string' ? d : JSON.stringify(d)) : '')); if (ok) passes++; else fails++; };
function makeClock() {
  const C = {t: 1000, q: [], n: 0};
  C.now = () => C.t;
  C.set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = ms => { const end = C.t + ms; for (;;) { C.q.sort((a, b) => a.at - b.at || a.id - b.id); const x = C.q[0]; if (!x || x.at > end) break; C.q.shift(); C.t = Math.max(C.t, x.at); x.f(); } C.t = end; };
  return C;
}

// ---------------------------------------------------------------- R: routing
function vt(o) {
  const clock = makeClock(), X = {perf: [], steps: [], prompts: []};
  X.P = {call: (d, info) => { X.perf.push({d, info}); return {ok: true, status: 'preparing', id: 'p' + X.perf.length}; }};
  const props = [{id: 'book', name: 'blue book', pos: [-0.8, 0.2, -0.24]}, {id: 'ball', name: 'yellow ball', pos: [0.31, 0.04, 0.27]}];
  X.motion = o.things === false ? {props: () => props} : {props: () => props, things: () => props};
  X.vt = VTL.createVoiceTools({perf: () => X.P, motion: () => X.motion, runCommand: () => ({ok: true}), sendCtl: () => {}, prompt: (t, opt) => { X.prompts.push({t, opt}); return {}; },
    sceneMatcher: () => o.M || ({parse: () => [], understood: () => false}), runStep: c => { X.steps.push(c); return {ok: true}; },
    objectPlans: o.objectPlans, userSpeaking: () => false, now: clock.now, timers: {set: clock.set, clear: clock.clear}});
  return X;
}
{
  let X = vt({});
  let r = X.vt.call('do_action', {description: 'spin the book on your finger'}, 'done', {});
  check('R a trick naming a thing (not understood by the scene matcher) -> the planner, with things', r.ok && r.status === 'preparing' && X.perf.length === 1 && X.perf[0].info.things === true && !X.steps.length, {r, perf: X.perf});
  X = vt({objectPlans: () => false});
  r = X.vt.call('do_action', {description: 'spin the book on your finger'}, 'done', {});
  check('R objectPlans off -> the refusal with the object hint', r.ok === false && r.status === 'use_object' && !X.perf.length, r);
  X = vt({things: false});
  r = X.vt.call('do_action', {description: 'throw the ball up and catch it behind you'}, 'done', {});
  check('R a motion layer without things() -> the refusal', r.status === 'use_object' && !X.perf.length, r);
  const M = {parse: t => /pick up the ball/.test(t) ? [{cmd: {action: 'object', verb: 'pick_up', object: {name: 'ball'}}, rule: 'pick'}] : [], understood: t => /pick up the ball/.test(t)};
  X = vt({M});
  r = X.vt.call('do_action', {description: 'pick up the ball'}, 'done', {});
  check('R a scene step the matcher fully reads still runs as an object step (fast, no plan)', r.ok && r.status === 'starting' && X.steps.length === 1 && !X.perf.length, {r, steps: X.steps});
  X = vt({});
  r = X.vt.call('do_action', {description: 'do three jumping jacks'}, 'done', {});
  check('R a request naming no thing -> the planner as before (no things flag)', r.ok && X.perf.length === 1 && !X.perf[0].info.things, X.perf);
}

// ---------------------------------------------------------------- P: perform.js + the real muse motion
function perfMuse() {
  const clock = makeClock(), X = {ctl: [], prompts: [], ev: []};
  let P = null;
  X.T = makeMuse(21, {onAction: e => { X.ev.push(e); if (P) P.onAction(e); }});
  X.T.step(20);
  X.T.m.engine.kfHeld = X.T.m.engine.kfHeld || (() => {}); X.T.m.engine.kfEvent = X.T.m.engine.kfEvent || (() => ({ok: true}));
  P = X.P = Perf.createPerformer({motion: () => X.T.m, sendCtl: m => X.ctl.push(JSON.parse(JSON.stringify(m))), character: () => 'muse',
    prompt: (t, opt) => { X.prompts.push({t, opt}); if (opt && opt.onAudio) opt.onAudio(); return {}; },
    userSpeaking: () => false, lastUserSpeech: () => -1e12, now: clock.now, timers: {set: clock.set, clear: clock.clear}});
  X.clock = clock;
  X.run = (s) => { for (let i = 0; i < s * 30; i++) { X.T.step(1); clock.advance(1000 / 30); } };
  return X;
}
{
  const X = perfMuse();
  const r = X.P.call('spin the book on your finger', {things: true});
  const msg = X.ctl.find(m => m.type === 'plan');
  check('P the plan request carries state.things (ids + positions) and the root', r.ok && msg && Array.isArray(msg.state.things) && msg.state.things.length >= 8 && msg.state.things.every(t => t.id) && msg.state.root && 'x' in msg.state.root, msg && msg.state);
  // the hub's answer: fetch, a failing step, poses
  X.P.onKf({type: 'kf', id: r.id, source: 'planner', done: true, ok: true, poses: [
    {t: 0.3, do: {verb: 'pick_up', object: 'book', side: 'right'}}, {t: 0.8, right_paw_y: 0.7, right_held_spin: 1}, {t: 1.4, right_held_spin: 2},
    {t: 1.8, do: {verb: 'pick_up', object: 'unicorn', side: 'left'}}, {t: 2.4, smile: 0.5}]});
  X.run(30);
  const end = X.ev.filter(e => e.kind === 'kf' && e.id === r.id && e.phase === 'end').pop();
  check('P the plan played to its end', !!end, X.ev.filter(e => e.kind === 'kf').map(e => e.phase));
  const note = X.prompts.find(p => /has finished on screen/.test(p.t));
  check('P the check-in note says what did not work (truthful)', note && /Part of it did not work/.test(note.t) && /did not work/.test(note.t), note ? note.t : X.prompts.map(p => p.t));
  // page cache: the same request with the things as they are now = a page hit; plan again after a thing moved
  const n0 = X.ctl.filter(m => m.type === 'plan').length;
  const st = X.T.m.things();
  const r2 = X.P.call('spin the book on your finger', {things: true});
  const n1 = X.ctl.filter(m => m.type === 'plan').length;
  void st; void r2;
  // the book is held now (it was not when the plan was made) -> a new plan, never a stale replay
  check('P the page cache key follows the things: a changed scene asks the hub again', n1 === n0 + 1, {n0, n1});
  X.run(2);
}
{
  const X = perfMuse();
  const r = X.P.call('wave the ball around', {things: true});
  X.P.onKf({type: 'kf', id: r.id, source: 'planner', done: true, ok: true, poses: [{t: 0.3, right_paw_y: 0.5}, {t: 0.9, right_paw_y: 0.2}, {t: 1.2, smile: 0.3}]});
  X.run(6);
  const n0 = X.ctl.filter(m => m.type === 'plan').length;
  const r2 = X.P.call('wave the ball around', {things: true});
  const n1 = X.ctl.filter(m => m.type === 'plan').length;
  check('P the same request with the things unchanged = a page cache hit (no new plan request)', r2.ok && r2.status === 'starting' && n1 === n0, {r2, n0, n1});
  const note = X.prompts.find(p => /has finished on screen/.test(p.t));
  check('P a plan without failed steps: the usual check-in note (no "did not work")', note && !/did not work/.test(note.t), note && note.t);
}
console.log(`${fails ? 'FAILED' : 'all'} ${fails ? fails + ' of ' + (fails + passes) : passes} ${fails ? 'failed' : 'passed'}`);
process.exit(fails);
