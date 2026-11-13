// voice layer + page wiring for perform_action (docs/PLANNER.md §2 page side, §4, §5). No browser,
// no vendor, no paid calls: app/perform.js with a mock motion (the §3 KF API + onAction events) and a mock voice
// (records prompt() calls); app/voice.js through its scripted transport (Realtime 2.1 / Gemini Live / GPT Live events);
// page.html's own onCall / onJSON / pageMatch / plannerOn code with stubs.   node tests/voice_planner_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const ROOT = path.join(__dirname, '..');
const Perf = load('app/perform.js');
let fails = 0, passes = 0;
function check(name, ok, detail) {
  if (ok) passes++; else fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : ''));
}

// ---------------------------------------------------------------------------------------------------- fake clock
function makeClock() {
  const C = {t: 1000, q: [], n: 0};
  C.now = () => C.t;
  C.set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = ms => {
    const end = C.t + ms;
    for (;;) {
      C.q.sort((a, b) => a.at - b.at || a.id - b.id);
      const x = C.q[0];
      if (!x || x.at > end) break;
      C.q.shift(); C.t = Math.max(C.t, x.at); x.f();
    }
    C.t = end;
  };
  return C;
}

// ---------------------------------------------------------------------------------------------------- mock motion (§3)
function mockMotion(clock, character) {
  const M = {character, jobs: new Map(), cmds: [], feeds: [], releases: [], stops: 0, listeners: [], channels: {}, onAction: null};
  const emit = ev => { if (M.onAction) M.onAction(Object.assign({t: clock.now() / 1000, src: 'voice'}, ev)); };
  function readyCheck(J) {
    const span = J.poses.length ? Math.max(...J.poses.map(p => +p.t)) : 0;
    if (J.phase === 'buffering' && ((J.poses.length >= 2 && span >= 1.0) || (J.done && J.poses.length))) { J.phase = 'ready'; emit({kind: 'kf', name: 'perform', id: J.id, phase: 'ready'}); }
    if (J.phase === 'ready' && J.released) startJob(J);
  }
  function startJob(J) {
    J.phase = 'playing'; J.t0 = clock.now();
    emit({kind: 'kf', name: 'perform', id: J.id, phase: 'start'});
    const dur = Math.max(...J.poses.map(p => +p.t)) * 1000;
    J.endTimer = clock.set(() => { if (J.phase === 'playing') { J.phase = 'ended'; emit({kind: 'kf', name: 'perform', id: J.id, phase: 'end', did: 'did the planned move'}); } }, dur + 150);
  }
  M.command = (cmd, o) => {
    M.cmds.push({cmd: JSON.parse(JSON.stringify(cmd)), o, t: clock.now()});
    if (cmd.action === 'keyframes') {
      for (const J of M.jobs.values()) if (J.phase === 'playing') { J.phase = 'interrupted'; clock.clear(J.endTimer); emit({kind: 'kf', name: 'perform', id: J.id, phase: 'interrupted'}); }
      const J = {id: cmd.id, poses: cmd.poses.slice(), done: !!cmd.done, phase: 'buffering', released: !cmd.hold};
      M.jobs.set(J.id, J); readyCheck(J);
      return {ok: true, id: cmd.id, did: 'getting ready to ' + cmd.description};
    }
    if (cmd.action === 'stop') { M.stops++; for (const J of M.jobs.values()) if (J.phase === 'playing' || J.phase === 'ready' || J.phase === 'buffering') { J.phase = 'interrupted'; clock.clear(J.endTimer); emit({kind: 'kf', name: 'perform', id: J.id, phase: 'interrupted'}); } return {ok: true}; }
    return {ok: true, id: 900 + M.cmds.length};
  };
  M.kfFeed = (id, poses, done) => { M.feeds.push({id, n: poses.length, done, t: clock.now()}); const J = M.jobs.get(id); if (!J) return; J.poses.push(...poses); if (done) J.done = true; readyCheck(J); };
  M.kfRelease = id => { M.releases.push({id, t: clock.now()}); const J = M.jobs.get(id); if (!J) return false; J.released = true; readyCheck(J); return true; };
  M.kfState = id => { const J = M.jobs.get(id); return J ? {phase: J.phase, poses: J.poses.length, planned_s: J.poses.length ? Math.max(...J.poses.map(p => +p.t)) : 0, played_s: 0} : null; };
  M.state = () => ({posture: character === 'muse' ? 'seated' : 'seated', holding: {left: null, right: null}, wearing: [], sittingOn: character === 'muse' ? 'chair' : null});
  // body channel the KF job drives (for the talk-while-moving check): a pure function of the job clock, never of speech
  M.bodyAt = (id, t) => { const J = M.jobs.get(id); if (!J || J.phase !== 'playing') return null; const s = (t - J.t0) / 1000; return Math.round(Math.sin(s * 3) * 1000) / 1000; };
  return M;
}

// ---------------------------------------------------------------------------------------------------- mock voice (records prompt())
function mockVoice(clock, opts) {
  const V = {prompts: [], speaking: false, userTalking: false, lastUser: -1e12, audioAfterMs: opts && opts.audioAfterMs, audio: []};
  V.prompt = (note, opt) => {
    const p = {note, opt: opt || {}, t: clock.now(), state: 'sent'};
    V.prompts.push(p);
    if (p.opt.respond && V.audioAfterMs != null && p.opt.onAudio) clock.set(() => { V.audio.push(clock.now()); p.opt.onAudio({id: 'x'}); }, V.audioAfterMs);
    return {id: 'v' + V.prompts.length, state: 'sent', cancel() {}};
  };
  return V;
}

function setup(character, o) {
  o = o || {};
  const clock = makeClock(), motion = mockMotion(clock, character), voice = mockVoice(clock, o);
  const ctl = [], ran = [];
  const alogCalls = [];
  const alog = o.alog === false ? null : {command: (c, r, s) => alogCalls.push(['command', c, r, s]), push: ev => alogCalls.push(['push', ev])};
  const P = Perf.createPerformer({
    motion: () => motion, prompt: voice.prompt, sendCtl: m => ctl.push(JSON.parse(JSON.stringify(m))),
    runCommand: (cmd, info) => { ran.push({cmd, info, t: clock.now()}); const r = motion.command(cmd, {src: 'voice'}); return r; },
    character: () => character, userSpeaking: () => voice.userTalking, lastUserSpeech: () => voice.lastUser,
    alog: () => alog, now: clock.now, timers: {set: clock.set, clear: clock.clear}, cache: o.cache,
  });
  motion.onAction = ev => P.onAction(ev);
  return {clock, motion, voice, ctl, ran, P, alogCalls};
}
const pose = (t, k) => Object.assign({t}, k || {right_paw_y: 0.5 + t / 4});
const plans = ctl => ctl.filter(m => m.type === 'plan');
const records = ctl => ctl.filter(m => m.type === 'perform');
const quoted = s => /(^|[\s(:;,])["“‘'][^"”’']{2,}["”’']/.test(s);

// ================================================================================================= perform.js page flow
function streamedPlan(character) {
  const X = setup(character, {audioAfterMs: 300});
  const r = X.P.call('wave with your left paw three times, slowly', {start_ms: 990, start_lat_ms: 400});
  const tag = '[' + character + '] ';
  check(tag + 'streamed: tool result at once, status preparing', r.ok && r.status === 'preparing' && /^p\d+$/.test(r.id) && !quoted(r.next), r);
  const pl = plans(X.ctl)[0];
  check(tag + 'streamed: plan sent with id, character, description, state', pl && pl.id === r.id && pl.character === character && pl.description === 'wave with your left paw three times, slowly' && pl.state && 'posture' in pl.state && 'holding' in pl.state && 'wearing' in pl.state && 'seated' in pl.state, pl);
  X.clock.advance(600);
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0)], done: false});
  const c0 = X.motion.cmds.find(c => c.cmd.action === 'keyframes');
  check(tag + 'streamed: first pose -> motion.command keyframes hold:true', c0 && c0.cmd.id === r.id && c0.cmd.hold === true && c0.cmd.done === false && c0.cmd.poses.length === 1 && c0.cmd.description, c0 && c0.cmd);
  X.clock.advance(200);
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0.6)], done: false});
  check(tag + 'streamed: later batches -> kfFeed', X.motion.feeds.length === 1 && X.motion.feeds[0].n === 1 && !X.motion.feeds[0].done);
  X.clock.advance(200);
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(1.2), pose(1.8)], done: false});   // >= 2 poses and >= 1.0 s -> ready
  const cue = X.voice.prompts.find(p => p.opt.respond && /about to start/.test(p.note));
  check(tag + 'ready -> prompt(about to start, planned length, respond:true)', cue && /1\.8 s/.test(cue.note) && !quoted(cue.note), X.voice.prompts.map(p => p.note));
  check(tag + 'ready: the job is HELD (no release before the cue audio)', X.motion.releases.length === 0 && X.motion.jobs.get(r.id).phase === 'ready');
  const tReady = X.clock.now();
  X.clock.advance(299);
  check(tag + 'still held 299 ms later (cue audio not yet)', X.motion.releases.length === 0);
  X.clock.advance(1);
  check(tag + 'released at the cue\'s first audio (300 ms)', X.motion.releases.length === 1 && X.motion.releases[0].t - tReady === 300, X.motion.releases);
  check(tag + "'start' -> a note without response", X.voice.prompts.some(p => /playing on screen now/.test(p.note) && !p.opt.respond));
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(2.4)], done: true, ok: true, source: 'planner', summary: 'a slow three-count wave', ms_first_pose: 600, ms_done: 1400, usd: 0.011});
  check(tag + 'done batch -> kfFeed(done:true)', X.motion.feeds.some(f => f.done));
  X.clock.advance(3000);
  const end = X.voice.prompts.find(p => /has finished/.test(p.note));
  check(tag + "'end' -> check-in note with response", end && end.opt.respond === true && /check in/.test(end.note) && !quoted(end.note), end);
  const rec = records(X.ctl)[0];
  const mk = rec && rec.marks || {};
  check(tag + 'perform record: source + all latency marks in order', rec && rec.id === r.id && rec.source === 'planner' && rec.outcome === 'done' &&
    mk.call_ms <= mk.plan_first_pose_ms && mk.plan_first_pose_ms <= mk.ready_ms && mk.ready_ms <= mk.cue_audio_ms && mk.cue_audio_ms <= mk.first_motion_ms && mk.first_motion_ms <= mk.end_ms && mk.user_end_ms === 590, rec);
  check(tag + 'action log: pending perform item with the job id (the motion events update it by id)', X.alogCalls.some(a => a[0] === 'command' && a[1].action === 'perform' && a[2].id === r.id));
  return X;
}

function talkWhileMoving(character) {
  const tag = '[' + character + '] talk-while-moving: ';
  const X = setup(character, {audioAfterMs: 200});
  const r = X.P.call('do a slow full spin while counting out loud to five', {});
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0), pose(1), pose(2), pose(3), pose(4)], done: true, ok: true, source: 'planner'});
  X.clock.advance(200);   // cue audio -> release -> start
  const J = X.motion.jobs.get(r.id);
  check(tag + 'the cue reply may run through the motion (released at its audio, not after it)', J.phase === 'playing' && X.motion.releases.length === 1);
  // the reference trajectory of the body channel, sampled with NO speech
  const ref = [], got = [];
  const n0 = X.voice.prompts.length;
  for (let i = 0; i < 20; i++) {
    const t = X.clock.now();
    ref.push(X.motion.bodyAt(r.id, t));
    // the avatar talks (counting): voiced audio onsets at the tap, the reply goes on — the performer must not react
    if (i % 4 === 0) X.P.avatarAudio();
    got.push(X.motion.bodyAt(r.id, t));
    X.clock.advance(150);
  }
  check(tag + 'body channel keeps its trajectory while speech plays', JSON.stringify(ref) === JSON.stringify(got) && ref.every(v => v !== null), {ref: ref.slice(0, 4), got: got.slice(0, 4)});
  const during = X.voice.prompts.slice(n0);
  check(tag + 'no reply requested, nothing held / stopped / re-released while it plays', during.every(p => !p.opt.respond) && X.motion.stops === 0 && X.motion.releases.length === 1, during.map(p => p.note));
  X.clock.advance(3000);
  check(tag + 'check-in only after the end', X.voice.prompts.slice(n0).filter(p => p.opt.respond).length === 1 && X.voice.prompts[X.voice.prompts.length - 1].opt.respond);
}

// library hit
(function () {
  const X = setup('fox');
  const r = X.P.call('wave', {});
  X.clock.advance(120);
  X.P.onKf({type: 'kf', id: r.id, poses: [], done: true, ok: true, source: 'library', engine: [{action: 'gesture', name: 'wave', side: 'right'}], summary: 'wave', ms_first_pose: 0, ms_done: 40});
  check('library hit: engine commands run at once through runCommand (source voice)', X.ran.length === 1 && X.ran[0].cmd.name === 'wave' && X.ran[0].info.perform === r.id);
  check('library hit: no keyframe job, no hold', !X.motion.cmds.some(c => c.cmd.action === 'keyframes') && X.motion.releases.length === 0);
  check('library hit: a start note without response', X.voice.prompts.length === 1 && !X.voice.prompts[0].opt.respond && /playing on screen now/.test(X.voice.prompts[0].note));
  X.clock.advance(2000);
  const rec = records(X.ctl)[0];
  check('library hit: check-in after the move, record source library', X.voice.prompts.some(p => p.opt.respond && /has finished/.test(p.note)) && rec && rec.source === 'library' && rec.outcome === 'done' && rec.marks.first_motion_ms === 1120, rec);
  // an engine move (whole-body, with an id) ends on its own end event
  const r2 = X.P.call('spin around', {});
  X.P.onKf({type: 'kf', id: r2.id, poses: [], done: true, ok: true, source: 'library', engine: [{action: 'move', name: 'spin'}]});
  const eid = X.ran[1] && X.ran[1].cmd.name === 'spin' ? 900 + X.motion.cmds.length : null;
  X.clock.advance(900);
  check('library engine move: no end before its end event', !records(X.ctl).some(m => m.id === r2.id));
  X.P.onAction({kind: 'move', name: 'spin', id: eid, phase: 'end', src: 'tool'});
  check('library engine move: ends on the engine job\'s end event', records(X.ctl).some(m => m.id === r2.id && m.outcome === 'done'));
  // a repeat of a library description is a page cache hit: 'starting' and instant
  const r3 = X.P.call('Wave!', {});
  check('library repeat: page cache hit -> status starting, no plan sent', r3.status === 'starting' && plans(X.ctl).length === 2 && X.ran.length === 3);
})();

// streamed plan + three characters + talk-while-moving
for (const ch of ['muse', 'fox', 'orb']) { streamedPlan(ch); talkWhileMoving(ch); }

// cache hits: the page cache (a repeat) and the hub's cache (all poses in the first message)
(function () {
  const X = setup('cube', {audioAfterMs: 250});
  const r1 = X.P.call('nod twice', {});
  X.P.onKf({type: 'kf', id: r1.id, poses: [pose(0), pose(0.5), pose(1.1)], done: true, ok: true, source: 'planner'});
  X.clock.advance(250 + 1500);
  const nPlans = plans(X.ctl).length, nPrompts = X.voice.prompts.length;
  const r2 = X.P.call('Nod twice.', {});
  const kf = X.motion.cmds.filter(c => c.cmd.action === 'keyframes').pop();
  check('page cache hit: status starting, no plan sent, all poses at once (done:true)', r2.status === 'starting' && plans(X.ctl).length === nPlans && kf.cmd.id === r2.id && kf.cmd.done === true && kf.cmd.poses.length === 3, r2);
  check('page cache hit: no extra cue prompt (the reply to "starting" is the cue)', !X.voice.prompts.slice(nPrompts).some(p => p.opt.respond));
  X.clock.advance(180);
  X.P.avatarAudio();      // the reply's first audio at the playback tap
  check('page cache hit: released at the reply\'s first audio (tap)', X.motion.releases.length === 2 && X.motion.releases[1].t - kf.t === 180, X.motion.releases);
  X.voice.lastUser = X.clock.now(); X.clock.advance(10);         // he asks for something new
  const r3 = X.P.call('hop on one foot', {});
  X.P.onKf({type: 'kf', id: r3.id, poses: [pose(0), pose(0.4), pose(0.9)], done: true, ok: true, source: 'cache'});
  check('hub cache hit (all poses, done in the first message): cue prompt + held', X.voice.prompts.some(p => p.opt.respond && /about to start/.test(p.note) && /hop on one foot/.test(p.note)) && X.motion.jobs.get(r3.id).phase === 'ready');
})();

// plan failure (hub says so) and planner silence
(function () {
  const X = setup('star', {});
  const r = X.P.call('juggle three balls', {});
  X.clock.advance(800);
  X.P.onKf({type: 'kf', id: r.id, poses: [], done: true, ok: false, source: 'planner', error: 'the planner timed out'});
  const p = X.voice.prompts[0];
  check('plan failure: a note asking for an honest reply (no example wording)', p && p.opt.respond && /could not be prepared/.test(p.note) && /timed out/.test(p.note) && !quoted(p.note), p);
  const rec = records(X.ctl)[0];
  check('plan failure: record outcome failed, nothing sent to the motion', rec && rec.outcome === 'failed' && !X.motion.cmds.length, rec);
  check('plan failure: the action log says it failed (truthful)', X.alogCalls.some(a => a[0] === 'command' && a[1].action === 'perform' && a[2].ok === false));
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0)], done: false});
  check('late batches of a finished plan are ignored', !X.motion.cmds.length);
  const r2 = X.P.call('moonwalk backwards', {});
  X.clock.advance(19999);
  check('planner silence: not failed before 20 s', !records(X.ctl).some(m => m.id === r2.id));
  X.clock.advance(2);
  check('planner silence: failed with an honest note at 20 s', records(X.ctl).some(m => m.id === r2.id && m.outcome === 'failed') && X.voice.prompts.some(q => q.opt.respond && /did not answer/.test(q.note)));
  // he is speaking when it fails: the note is context only
  X.voice.userTalking = true;
  const r3 = X.P.call('do a handstand', {});
  X.P.onKf({type: 'kf', id: r3.id, poses: [], done: true, ok: false, error: 'no pose'});
  check('plan failure while he speaks: note without response', !X.voice.prompts[X.voice.prompts.length - 1].opt.respond);
  // motion refuses the job (no KF API on this motion layer) = an honest failure, not silence
  const Y = setup('drop', {});
  Y.motion.command = () => ({ok: false, reason: 'unknown action "keyframes"'});
  const r4 = Y.P.call('shake', {});
  Y.P.onKf({type: 'kf', id: r4.id, poses: [pose(0), pose(1.1)], done: true, ok: true});
  check('motion refusal: honest failure note', Y.voice.prompts.some(q => q.opt.respond && /could not be played/.test(q.note)) && records(Y.ctl)[0].outcome === 'failed');
})();

// cue release: 700 ms fallback (no cue audio ever) and he is speaking at ready
(function () {
  const X = setup('pyramid', {});        // the mock voice never produces audio
  const r = X.P.call('lean left', {});
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0), pose(1.2)], done: true, ok: true});
  const tReady = X.clock.now();
  X.clock.advance(699);
  check('cue fallback: still held at 699 ms', X.motion.releases.length === 0);
  X.clock.advance(1);
  check('cue fallback: released at 700 ms', X.motion.releases.length === 1 && X.motion.releases[0].t - tReady === 700);
  check('cue fallback: record says why', (() => { X.clock.advance(2000); const m = records(X.ctl)[0]; return m && m.release === 'cue timeout'; })(), records(X.ctl));
  const Y = setup('blob', {});
  Y.voice.userTalking = true;
  const r2 = Y.P.call('bounce', {});
  Y.P.onKf({type: 'kf', id: r2.id, poses: [pose(0), pose(1.0)], done: true, ok: true});
  check('he is speaking at ready: no cue reply requested, released at once', !Y.voice.prompts.some(p => p.opt.respond) && Y.motion.releases.length === 1);
})();

// check-in skipped while he speaks / when he spoke during the action
(function () {
  const X = setup('rabbit', {audioAfterMs: 100});
  const r = X.P.call('wiggle your ears', {});
  X.P.onKf({type: 'kf', id: r.id, poses: [pose(0), pose(1.2)], done: true, ok: true});
  X.clock.advance(400);
  X.voice.userTalking = true;
  X.clock.advance(1500);
  const end = X.voice.prompts.find(p => /has finished/.test(p.note));
  check('check-in skipped while he speaks (note without response)', end && !end.opt.respond && !/check in/.test(end.note), end);
  const Y = setup('hippo', {audioAfterMs: 100});
  const r2 = Y.P.call('open your mouth wide', {});
  Y.P.onKf({type: 'kf', id: r2.id, poses: [pose(0), pose(1.2)], done: true, ok: true});
  Y.clock.advance(500);
  Y.voice.lastUser = Y.clock.now();      // he said something while it played
  Y.clock.advance(1500);
  const end2 = Y.voice.prompts.find(p => /has finished/.test(p.note));
  check('check-in skipped when he spoke since it started', end2 && !end2.opt.respond && records(Y.ctl)[0].checkin === 'skipped: he spoke');
})();

// one request, several calls (no speech in between) run in order; a newer request replaces one being prepared
(function () {
  const X = setup('muse', {audioAfterMs: 100});
  const a = X.P.call('raise both arms', {response_id: 'r1'});
  const b = X.P.call('then spin once', {response_id: 'r1'});
  check('same request: the second step says it follows', b.ok && b.status === 'preparing' && /follows/.test(b.next));
  X.P.onKf({type: 'kf', id: b.id, poses: [pose(0), pose(1.0)], done: true, ok: true});
  check('same request: the second step\'s poses wait for the first step', !X.motion.cmds.some(c => c.cmd.id === b.id));
  X.P.onKf({type: 'kf', id: a.id, poses: [pose(0), pose(1.0)], done: true, ok: true});
  X.clock.advance(100);
  check('same request: the first step plays', X.motion.jobs.get(a.id).phase === 'playing');
  X.clock.advance(1200);
  check('same request: no check-in between the steps, the second is commanded when the first ends',
    !X.voice.prompts.some(p => /has finished/.test(p.note) && /raise both arms/.test(p.note)) && X.motion.cmds.some(c => c.cmd.id === b.id && c.cmd.poses.length === 2));
  X.clock.advance(100);
  X.clock.advance(1300);
  check('same request: check-in after the last step', X.voice.prompts.some(p => p.opt.respond && /has finished/.test(p.note) && /spin once/.test(p.note)));
  // an object step after a body step of the same request waits for it
  const c = X.P.call('wave at me', {});
  let ranObj = 0;
  const q = X.P.engineCall({}, () => { ranObj++; return {ok: true, id: 77}; });
  check('object step of the same request: deferred, honest queued result', q.deferred && q.result.ok && q.result.queued && ranObj === 0);
  X.P.onKf({type: 'kf', id: c.id, poses: [pose(0), pose(1.0)], done: true, ok: true});
  X.clock.advance(100 + 1200);
  check('object step runs when the body step ends', ranObj === 1);
  // newer request (he spoke in between) replaces a plan still being prepared
  const d1 = X.P.call('jump', {});
  X.clock.advance(500); X.voice.lastUser = X.clock.now(); X.clock.advance(100);
  const d2 = X.P.call('no, a cartwheel', {});
  check('newer request replaces the one being prepared (record outcome replaced)', records(X.ctl).some(m => m.id === d1.id && m.outcome === 'replaced') && d2.ok);
  X.P.onKf({type: 'kf', id: d1.id, poses: [pose(0), pose(1)], done: true});
  check('the replaced plan\'s poses are ignored', !X.motion.cmds.some(cc => cc.cmd.id === d1.id));
  // stop
  X.P.onKf({type: 'kf', id: d2.id, poses: [pose(0), pose(3)], done: true, ok: true});
  X.clock.advance(200);
  const moving = X.P.stop('he asked to stop');
  check('stop: a playing job is stopped through the motion layer', moving && X.motion.stops === 1 && records(X.ctl).some(m => m.id === d2.id && m.outcome === 'interrupted'));
})();

// ================================================================================================= voice.js prompt() per back-end
function loadVoice(clock) {
  const src = fs.readFileSync(path.join(ROOT, 'app/voice.js'), 'utf8');
  const w = {};
  const perf = {now: clock.now};
  new Function('window', 'performance', 'setTimeout', 'clearTimeout', src)(w, perf, clock.set, clock.clear);
  return w.AvatarVoice;
}
const fakeCtx = {createGain: () => ({connect() {}, disconnect() {}, gain: {value: 1}}), currentTime: 0};
async function makeV(mode, extra) {
  const clock = makeClock(), AV = loadVoice(clock), sent = [], calls = [];
  const v = AV.createVoice(Object.assign({mode, ctx: fakeCtx, transport: {send: ev => sent.push(JSON.parse(JSON.stringify(ev)))},
    onCall: (name, args, phase, info) => { calls.push({name, args, phase, t: clock.now()}); if (name === 'perform_action' && phase === 'done') return {ok: true, status: 'preparing', id: 'p1'}; return {ok: true}; }}, extra || {}));
  await v.start();
  return {clock, v, sent, calls, AV};
}
const types = (sent, from) => sent.slice(from || 0).map(e => e.type || Object.keys(e)[0]);

(async function voiceTests() {
  // ---- Realtime 2.1 (primary)
  {
    const {clock, v, sent} = await makeV('realtime', {grabStill: () => 'data:image/jpeg;base64,AAAA'});
    check('realtime: SILENT prefix exported', typeof (await makeV('realtime')).AV.SILENT === 'string');
    // he asks; the model calls perform_action (tool-only response)
    v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 0});
    check('realtime: the camera still still rides his speech start (maybeStill)', sent.some(e => e.type === 'conversation.item.create' && e.item.content && e.item.content[0].type === 'input_image'));
    v.feed({type: 'input_audio_buffer.speech_stopped', audio_end_ms: 900});
    v.feed({type: 'response.created', response: {id: 'r1'}});
    v.feed({type: 'response.output_item.added', response_id: 'r1', item: {type: 'function_call', id: 'i1', call_id: 'c1', name: 'perform_action'}});
    const n0 = sent.length;
    v.feed({type: 'response.function_call_arguments.done', response_id: 'r1', call_id: 'c1', name: 'perform_action', arguments: JSON.stringify({description: 'spin twice'})});
    const out = sent.slice(n0).find(e => e.type === 'conversation.item.create' && e.item.type === 'function_call_output');
    check('realtime: perform_action result posted at once (status preparing)', out && JSON.parse(out.item.output).status === 'preparing' && out.item.call_id === 'c1');
    v.feed({type: 'response.done', response: {id: 'r1', status: 'completed', output: [{type: 'function_call', call_id: 'c1', name: 'perform_action', arguments: '{"description":"spin twice"}'}]}});
    check('realtime: the tool-only response gets its follow-up (the filler)', types(sent, n0).filter(t => t === 'response.create').length === 1);
    // ready: a cue prompt while the filler is being created -> note now, its response waits
    let audio = 0;
    const n1 = sent.length;
    const h = v.prompt('The action he asked for (spin twice) is about to start.', {respond: true, maxWaitMs: 1200, onAudio: () => audio++});
    check('realtime: prompt note is a SILENT system item at once', sent[n1] && sent[n1].type === 'conversation.item.create' && sent[n1].item.role === 'system' && sent[n1].item.content[0].text.startsWith('Silent context'));
    check('realtime: no response.create while the follow-up is being created', !types(sent, n1).includes('response.create') && h.state === 'queued');
    v.feed({type: 'response.created', response: {id: 'r2'}});
    clock.advance(200);
    check('realtime: no response.create while a response is active', !types(sent, n1).includes('response.create'));
    v.feed({type: 'response.done', response: {id: 'r2', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(1);
    const rc = sent.slice(n1).find(e => e.type === 'response.create');
    check('realtime: the cue response is created right after the active one ends, tagged', rc && rc.response && rc.response.metadata && rc.response.metadata.avatar_prompt === h.id && rc.event_id, rc);
    v.feed({type: 'response.created', response: {id: 'r3', metadata: {avatar_prompt: h.id}}});
    v.feed({type: 'output_audio_buffer.started', response_id: 'r3'});
    check('realtime: onAudio at the prompted reply\'s first audio', audio === 1 && h.state === 'speaking');
    // a context-only note while that reply plays: an item only, never a cancel / clear / new response (talk-while-moving)
    const n2 = sent.length;
    v.prompt('The action he asked for (spin twice) is playing on screen now.', {respond: false});
    check('realtime: a start note while the avatar talks = one item, no response.create / cancel / clear', JSON.stringify(types(sent, n2)) === '["conversation.item.create"]');
    v.feed({type: 'response.done', response: {id: 'r3', status: 'completed', output: [{type: 'message'}]}});
    check('realtime: prompted reply done', h.state === 'done');
    // check-in queued, he starts talking -> dropped ('spoke since')
    const n3 = sent.length;
    const h2 = v.prompt('The action he asked for (spin twice) has finished on screen.', {respond: true, maxWaitMs: 4000});
    check('realtime: an idle model -> the check-in response is created at once', types(sent, n3).includes('response.create') && h2.state === 'creating');
    v.feed({type: 'response.created', response: {id: 'r4', metadata: {avatar_prompt: h2.id}}});
    v.feed({type: 'response.done', response: {id: 'r4', status: 'completed', output: [{type: 'message'}]}});
    // a check-in queued while a reply is active; he starts talking -> dropped ('spoke since'); his own turn goes first
    v.feed({type: 'response.created', response: {id: 'r5'}});
    const h3 = v.prompt('x', {respond: true, maxWaitMs: 4000});
    clock.advance(100);
    check('realtime: queued while a reply is active', h3.state === 'queued');
    v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 5000});
    clock.advance(100);
    check('realtime: a prompt queued before he spoke is dropped once he speaks', h3.state === 'dropped' && h3.why === 'he spoke', h3.why);
    v.feed({type: 'response.done', response: {id: 'r5', status: 'cancelled', output: []}});
    const n4 = sent.length;
    const h3b = v.prompt('x2', {respond: true, maxWaitMs: 4000, dropIfUserSpoke: false});
    clock.advance(300);
    check('realtime: no response while he is speaking', !types(sent, n4).includes('response.create') && h3b.state === 'queued');
    v.feed({type: 'input_audio_buffer.speech_stopped', audio_end_ms: 6000});
    clock.advance(200);
    check('realtime: … nor right after his turn (the server creates its own reply)', !types(sent, n4).includes('response.create'));
    v.feed({type: 'response.created', response: {id: 'r6'}});
    v.feed({type: 'response.done', response: {id: 'r6', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(1);
    check('realtime: … then after his reply is done', h3b.state === 'creating');
    v.feed({type: 'response.created', response: {id: 'r7', metadata: {avatar_prompt: h3b.id}}});
    v.feed({type: 'response.done', response: {id: 'r7', status: 'completed', output: [{type: 'message'}]}});
    // stale: busy longer than maxWaitMs
    v.feed({type: 'response.created', response: {id: 'r8'}});
    const h4 = v.prompt('y', {respond: true, maxWaitMs: 500, dropIfUserSpoke: false});
    clock.advance(600);
    check('realtime: a prompt that cannot be spoken in time is dropped as stale', h4.state === 'dropped' && h4.why === 'stale', h4.why);
    v.feed({type: 'response.done', response: {id: 'r8', status: 'completed', output: []}});
    // the server refuses (a response already active): re-queued, then sent
    const h5 = v.prompt('z', {respond: true, maxWaitMs: 3000});
    check('realtime: idle -> sent at once', h5.state === 'creating', h5.state);
    const ev5 = sent.filter(e => e.type === 'response.create').pop();
    v.feed({type: 'error', error: {code: 'conversation_already_has_active_response', message: 'active', event_id: ev5.event_id}});
    check('realtime: an "active response" refusal re-queues the prompt', h5.state === 'queued');
    clock.advance(100);
    check('realtime: … and it is sent again', h5.state === 'creating');
    // metadata not accepted by this server: falls back to order matching
    v.feed({type: 'error', error: {code: 'unknown_parameter', param: 'response.metadata', message: 'Unknown parameter', event_id: sent.filter(e => e.type === 'response.create').pop().event_id}});
    clock.advance(100);
    const last = sent.filter(e => e.type === 'response.create').pop();
    check('realtime: metadata refused -> resent without it', h5.state === 'creating' && !last.response);
    v.feed({type: 'response.created', response: {id: 'r9'}});
    v.feed({type: 'output_audio_buffer.started', response_id: 'r9'});
    check('realtime: without metadata the next created response is matched by order', h5.state === 'speaking');
  }
  // ---- Gemini Live
  {
    const {clock, v, sent, calls} = await makeV('gemini');
    v.feed({toolCall: {functionCalls: [{id: 'f1', name: 'perform_action', args: {description: 'touch your nose'}}]}});
    const tr = sent.find(e => e.toolResponse);
    check('gemini: perform_action answered at once (status preparing)', tr && tr.toolResponse.functionResponses[0].response.status === 'preparing' && calls[0].args.description === 'touch your nose');
    // the model talks (filler): a cue prompt must not interrupt it (client content interrupts a generation)
    const pcm = Buffer.from(new Int16Array(240).buffer).toString('base64');
    v.feed({serverContent: {modelTurn: {parts: [{inlineData: {mimeType: 'audio/pcm;rate=24000', data: pcm}}]}}});
    let audio = 0;
    const n0 = sent.length;
    const h = v.prompt('about to start', {respond: true, maxWaitMs: 1500, onAudio: () => audio++});
    const h0 = v.prompt('playing now', {respond: false});
    clock.advance(300);
    check('gemini: nothing sent while the model is generating (never interrupts its speech)', sent.length === n0 && h.state === 'queued' && h0.state === 'queued');
    v.feed({serverContent: {turnComplete: true}});
    clock.advance(1);
    const cc = sent.slice(n0).find(e => e.clientContent);
    check('gemini: after turnComplete -> one clientContent turn, turnComplete:true, SILENT', cc && cc.clientContent.turnComplete === true && cc.clientContent.turns[0].parts[0].text.startsWith('Silent context'));
    v.feed({serverContent: {modelTurn: {parts: [{inlineData: {mimeType: 'audio/pcm;rate=24000', data: pcm}}]}}});
    check('gemini: onAudio at the prompted reply\'s first audio', audio === 1 && h.state === 'speaking');
    check('gemini: the context-only note waits for that reply too', h0.state === 'queued');
    v.feed({serverContent: {turnComplete: true}});
    clock.advance(1);
    const cc2 = sent.filter(e => e.clientContent).pop();
    check('gemini: context-only note sent when idle, turnComplete:false', h.state === 'done' && h0.state === 'sent' && cc2.clientContent.turnComplete === false);
    // he is talking (input transcription) -> waits
    v.feed({serverContent: {inputTranscription: {text: 'hey'}}});
    const h2 = v.prompt('finished', {respond: true, maxWaitMs: 4000});
    clock.advance(200);
    check('gemini: a prompt waits while he is speaking', h2.state === 'queued');
  }
  // ---- GPT Live
  {
    const {v, sent} = await makeV('live');
    v.prompt('about to start', {respond: true});
    v.prompt('playing', {respond: false});
    check('gpt live: respond -> instructions.append, context -> thinking.append', JSON.stringify(types(sent)) === '["session.instructions.append","session.thinking.append"]');
  }
  // ---- mock: never pretends
  {
    const {v} = await makeV('mock');
    const h = v.prompt('x', {respond: true});
    check('mock voice: prompt is dropped honestly (no vendor)', h.state === 'dropped' && h.why === 'mock');
  }
  pageTests();
  console.log(`${passes} passed, ${fails} failed`);
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });

// ================================================================================================= page.html's own code
function pageTests() {
  const page = fs.readFileSync(path.join(ROOT, 'app/page.html'), 'utf8');
  const fnSrc = name => { const a = page.indexOf('function ' + name + '('); const b = page.indexOf('\n}\n', a); return page.slice(a, b + 3); };
  const ttcSrc = page.slice(page.indexOf('const POSE_MAP'), page.indexOf('function toolToCommand')) + page.slice(page.indexOf('function toolToCommand'), page.indexOf('\n}\n', page.indexOf('function actToolCommand')) + 3);
  const a = page.indexOf('onCall: (name, args, phase, info) => {') + 'onCall: '.length, b = page.indexOf('\n    onMetric', a);
  const onCallSrc = page.slice(a, b).replace(/,\s*$/, '');
  const onJSONSrc = fnSrc('onJSON');
  const mk = (Q, cfg, mode) => {
    const env = {ran: [], ctl: [], perf: [], stops: 0, eng: 0};
    const D = {mode, cmds: [], perf: {call: (d, i) => { env.perf.push(d); return {ok: true, status: 'preparing', id: 'p9'}; }, onKf: j => env.perf.push(['kf', j.id]),
      stop: () => { env.stops++; }, engineCall: (info, run) => { env.eng++; return {deferred: false, result: run()}; }}, intentRes: null, alog: null, lastCmdRes: null};
    const S = {cfg, direct: true};
    const f = new Function('Q', 'S', 'D', 'env', ttcSrc + fnSrc('plannerReq') + fnSrc('plannerOn') + `
      const intentAct = () => false;
      function runCommand(cmd, source, info) { env.ran.push({cmd, source}); return {ok: true, id: 5, did: 'ok'}; }
      const sendCtl = m => env.ctl.push(m), echoOfAvatar = () => false, speechToolResult = r => r, REL_ALIAS = {};
      function intentToolResult(ran) { return ran && typeof ran === 'object' ? {ok: ran.ok !== false, did: ran.did} : {ok: !!ran}; }
      const onCall = ${onCallSrc};
      ${onJSONSrc}
      return {onCall, onJSON, plannerOn, plannerReq};`);
    return Object.assign(f(Q, S, D, env), {env, D});
  };
  const Qs = o => ({get: k => (o && k in o ? o[k] : null)});
  const on = mk(Qs({}), {character: 'fox'}, 'realtime');
  check('page: planner mode on by default', on.plannerOn() === true);
  check('page: ?planner=0 -> off', mk(Qs({planner: '0'}), {}, 'realtime').plannerOn() === false);
  check('page: config planner:false -> off (and the broker gets planner:false)', mk(Qs({}), {planner: false}, 'realtime').plannerOn() === false && mk(Qs({}), {planner: false}, 'realtime').plannerReq() === false);
  check('page: Gemini + thinking (no function calls) keeps the matcher path', mk(Qs({}), {}, 'gemini-think').plannerOn() === false);
  // planner on: perform_action -> performer, at once; matchers do not execute; kf -> performer
  const r = on.onCall('perform_action', {description: 'spin around twice'}, 'done', {start_lat_ms: 300});
  check('page planner on: perform_action -> D.perf.call, result returned to the model', r.status === 'preparing' && on.env.perf[0] === 'spin around twice' && on.env.ctl.some(m => m.type === 'call' && m.ran.status === 'preparing'));
  on.onJSON({type: 'intent', cmd: {action: 'gesture', name: 'wave'}, text: 'wave'});
  // hybrid (latency first): the fast matchers execute in planner mode too (what-if guarded)
  check('page planner on: the hub fast matcher executes (hybrid)', on.env.ran.some(x => x.cmd.name === 'wave' && x.source === 'matcher'));
  on.env.ran.length = 0;
  on.onJSON({type: 'kf', id: 'p9', poses: [], done: false});
  check('page planner on: kf -> D.perf.onKf', on.env.perf.some(x => Array.isArray(x) && x[0] === 'kf'));
  on.onJSON({type: 'act', id: 3, name: 'perform_action', args: {description: 'bow'}});
  check('page planner on: the hub GPT Live act path -> D.perf.call + act_result', on.env.perf.includes('bow') && on.env.ctl.some(m => m.type === 'act_result' && m.id === 3 && m.result.status === 'preparing'));
  on.onCall('release', {parts: 'all'}, 'done', {});
  check('page planner on: release all = stop the action in flight, then relax', on.env.stops === 1 && on.env.ran.some(x => x.cmd.action === 'release'));
  on.onCall('object', {verb: 'pick_up', object: {name: 'ball'}}, 'done', {});
  check('page planner on: object calls go through the performer\'s ordering', on.env.eng === 1 && on.env.ran.some(x => x.cmd.action === 'object'));
  // planner off: today's path exactly
  const off = mk(Qs({planner: '0'}), {character: 'muse'}, 'realtime');
  const r0 = off.onCall('perform_action', {description: 'spin'}, 'done', {});
  check('page planner=0: perform_action is an unknown tool (as before)', r0 && r0.ok === false && !off.env.perf.length);
  off.onCall('move', {name: 'spin', count: 2}, 'done', {});
  check('page planner=0: move tool runs on the motion (source voice)', off.env.ran.some(x => x.cmd.action === 'move' && x.cmd.name === 'spin' && x.cmd.count === 2 && x.source === 'voice'));
  off.onJSON({type: 'intent', cmd: {action: 'gesture', name: 'wave'}, text: 'wave', source: 'matcher'});
  check('page planner=0: the hub fast matcher executes', off.env.ran.some(x => x.cmd.name === 'wave' && x.source === 'matcher'));
  off.onJSON({type: 'act', id: 4, name: 'move', args: {name: 'jump'}});
  check('page planner=0: hub act move runs as before', off.env.ran.some(x => x.cmd.action === 'move' && x.cmd.name === 'jump') && off.env.ctl.some(m => m.type === 'act_result' && m.id === 4));
  check('page planner=0: release does not touch the performer', (off.onCall('release', {parts: 'all'}, 'done', {}), off.env.stops === 0));
  // the page caption matcher: returns before matching in planner mode
  const pm = fnSrc('pageMatch');
  check('page: the caption matcher runs in planner mode too, never on a what-if (hybrid)', /plannerOn\(\) && D\.vt && D\.vt\.whatIf/.test(pm) && pm.indexOf('whatIf') < pm.indexOf('runCommand('));
  // the harness still extracts toolToCommand / actToolCommand from page.html (engine.cjs)
  const {toolToCommand} = new Function(ttcSrc + '\nconst intentAct = () => false; return {toolToCommand};')();
  check('page: toolToCommand still extractable (harness)', toolToCommand('move', {name: 'spin'}).action === 'move');
}
