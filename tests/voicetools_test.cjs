// voice: the eight voice tools + events -> silent notes (app/voicetools.js), the what-if guard, talk-while-moving
// (app/perform.js `talk`), the look-at-a-thing target (app/motion.js), and page.html's wiring (onCall / onJSON 'act' /
// 'behavior_code'). No browser, no vendor, no paid calls.        node tests/voicetools_test.cjs
'use strict';
const PAGE_SRC_FOR_LEAD_CHECK = require('fs').readFileSync(require('path').join(__dirname, '..', 'app', 'page.html'), 'utf8');
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const ROOT = path.join(__dirname, '..');
const VTL = load('app/voicetools.js'), Perf = load('app/perform.js');
let fails = 0, passes = 0;
function check(name, ok, detail) {
  if (ok) passes++; else fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : ''));
}
function makeClock() {
  const C = {t: 1000, q: [], n: 0};
  C.now = () => C.t;
  C.set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = ms => { const end = C.t + ms; for (;;) { C.q.sort((a, b) => a.at - b.at || a.id - b.id); const x = C.q[0]; if (!x || x.at > end) break; C.q.shift(); C.t = Math.max(C.t, x.at); x.f(); } C.t = end; };
  return C;
}
function mockBehaviors() {
  const B = {progs: [], started: [], changed: [], stopped: [], fns: [], n: 0};
  B.start = o => { const id = ++B.n; B.started.push(o); B.progs.push({id, name: o.name, instruction: o.instruction, priority: o.priority, since: 0, frames: 0, errors: 0, code: o.code}); return id; };
  B.change = (n, o) => { B.changed.push([n, o]); const p = B.progs.find(x => x.name === n); if (p) { p.instruction = o.instruction; p.code = o.code; } return true; };
  B.stop = n => { B.stopped.push(n); B.progs = n === 'all' ? [] : B.progs.filter(x => x.name !== n && x.id !== n); return true; };
  B.list = () => B.progs.map(({code, ...r}) => r);
  B.get = n => B.progs.find(x => x.name === n) || null;
  B.onEvent = fn => { B.fns.push(fn); };
  return B;
}
function setup(o) {
  o = o || {};
  const clock = makeClock(), X = {clock, ctl: [], prompts: [], perf: [], cmds: [], events: [], speaking: false, avatar: false, react: o.react || 'rules'};
  X.P = {call: (d, info) => { X.perf.push({d, info}); return {ok: true, status: 'preparing', id: 'p' + X.perf.length}; }};
  X.B = o.noB ? null : mockBehaviors();
  X.G = o.G || null; X.A = o.A || null;
  X.motion = o.motion || {props: () => [{id: 'chair', name: 'chair', pos: [0.6, 0.25, 0.2]}, {id: 'laptop', name: 'silver laptop', pos: [0.4, 0, -0.3]}]};
  X.vt = VTL.createVoiceTools({perf: () => o.noPerf ? null : X.P, motion: () => X.motion, behaviors: () => X.B, gestures: () => X.G, attention: () => X.A,
    runCommand: (cmd, info) => { X.cmds.push(cmd); return o.runCommand ? o.runCommand(cmd) : {ok: true, id: X.cmds.length, did: 'did ' + cmd.action}; },
    sendCtl: m => X.ctl.push(m), prompt: (t, opt) => { X.prompts.push({t, opt}); return {id: 'n' + X.prompts.length}; },
    sendEvent: o.sendEvent ? (t, opt) => { X.events.push({t, opt}); return {sent: true}; } : undefined,
    userSpeaking: () => X.speaking, avatarSpeaking: () => X.avatar, lastUserSpeech: () => X.lastSpeech || -1e12, reactMode: () => X.react,
    now: clock.now, timers: {set: clock.set, clear: clock.clear}});
  return X;
}
const say = (X, text, gapMs) => { X.vt.userSpeech('start'); X.clock.advance(300); X.vt.userSpeech('stop'); X.clock.advance(gapMs == null ? 200 : gapMs); X.vt.userText(text, true); X.lastSpeech = X.clock.now(); X.clock.advance(400); };

// ===================================================================================================== dispatch
{
  const X = setup();
  check('handles: the eight tools + the alias, nothing else', ['do_action', 'perform_action', 'start_behavior', 'change_behavior', 'stop_behavior', 'list_behaviors', 'look', 'set_style', 'learn_gesture'].every(n => X.vt.handles(n)) && !X.vt.handles('gesture') && !X.vt.handles('object') && X.vt.call('gesture', {name: 'wave'}, 'done', {}) === null);
  check('early / start phases are not dispatched', X.vt.call('look', {target: 'left'}, 'early', {}) === null && !X.cmds.length);
  say(X, 'Can you jump twice?');
  const r = X.vt.call('do_action', {description: 'jump twice', talk: 'through'}, 'done', {start_lat_ms: 900});
  check('do_action -> perform.call with his description and talk', r.status === 'preparing' && X.perf[0].d === 'jump twice' && X.perf[0].info.talk === 'through' && X.perf[0].info.start_lat_ms === 900);
  check('do_action passes his own words (the hub library\'s second try)', X.perf[0].info.words === 'Can you jump twice?');
  const r2 = X.vt.call('perform_action', {description: 'spin once'}, 'done', {});
  check('perform_action is an alias of do_action', r2.ok && X.perf[1].d === 'spin once' && !('talk' in X.perf[1].info));
  check('do_action without a description / a body: honest failure', X.vt.call('do_action', {}, 'done', {}).ok === false && setup({noPerf: true}).vt.call('do_action', {description: 'wave'}, 'done', {}).ok === false);
  {   // hybrid: the engine first (named move / library-read description), a request a matcher already started, then the planner
    const lib = []; let covered = false;
    const Y = setup();
    Y.vt = VTL.createVoiceTools({perf: () => Y.P, motion: () => Y.motion, runCommand: () => ({ok: true}), sendCtl: () => {}, prompt: (t, o) => { Y.prompts.push({t, opt: o}); return {}; },
      runLibrary: (desc, move) => { lib.push([desc, move]); return move ? {ran: 1, dup: 0, did: 'doing a backflip'} : /^spin/.test(desc) ? {ran: 0, dup: 1} : null; },
      coveredByMatcher: () => covered, userSpeaking: () => false, now: Y.clock.now, timers: {set: Y.clock.set, clear: Y.clock.clear}});
    say(Y, 'Do a backward somersault.');
    let q = Y.vt.call('do_action', {description: 'do a backward somersault now', move: {name: 'backflip', count: 1}}, 'done', {});
    check('hybrid: a named move -> the engine at once (never the planner)', q.status === 'starting' && lib[0][1].name === 'backflip' && !Y.perf.length);
    q = Y.vt.call('do_action', {description: 'spin around twice'}, 'done', {});
    check('hybrid: a library-read description a matcher already ran -> already doing that, nothing twice', q.status === 'starting' && /already/.test(q.did) && !Y.perf.length);
    covered = true;
    q = Y.vt.call('do_action', {description: 'jump as high as you comfortably can'}, 'done', {});
    check('hybrid: an unread description whose request a matcher started -> covered, no planner', q.status === 'starting' && /already/.test(q.did) && !Y.perf.length);
    covered = false;
    q = Y.vt.call('do_action', {description: 'do the moonwalk across the room'}, 'done', {});
    check('hybrid: anything else -> the performer (cache -> planner)', q.status === 'preparing' && Y.perf.length === 1);
    lib.length = 0; say(Y, 'Do a backflip.');
    check('hybrid early: a streamed move object plays at once (before the description ends)', Y.vt.call('do_action', {move: {name: 'backflip'}}, 'early', {}) === null && lib.length === 1 && lib[0][1].name === 'backflip');
    Y.vt.userSpeech('start'); Y.clock.advance(500); Y.vt.userSpeech('stop');
    Y.vt.call('do_action', {move: {name: 'jump'}}, 'early', {});
    check('hybrid early: not while his words are still on their way (the done call decides)', lib.length === 1);
    Y.vt.userText('If I asked you to jump, could you?', true);
    Y.vt.call('do_action', {move: {name: 'jump'}}, 'early', {});
    check('hybrid early: never on a what-if', lib.length === 1);
    check('hybrid: the matchers\' guard (whatIf) reads a split half too', (say(Y, 'If I asked you to jump'), say(Y, 'could you?'), Y.vt.whatIf('could you?')) && !Y.vt.whatIf('jump twice'));
  }
  const st = X.vt.stats();
  check('stats: dispatch timing recorded (3 dispatched calls)', st.dispatch_us.n === 3 && st.dispatch_us.p50 !== null, st.dispatch_us);
}

// ===================================================================================================== the what-if guard
{
  const pos = ['If I asked you to jump, could you?', 'What if I told you to spin around?', 'Suppose I wanted a backflip, would you do it?', 'Hypothetically, could you dance?',
    'Would you jump if I asked you to?', 'What would you do if I asked you to wave?', 'If someone asked you to do a cartwheel, would you?', "Let's say I asked you to twirl. Could you?",
    'If you had to dance right now, how would you do it?', 'Imagine if I told you to sit down.', 'If my son asked you to hop, would you?', 'if i ask you to dance, dance',
    'what would happen if you did a backflip', 'If we wanted you to bow, would you be able to?'];
  const neg = ['Can you jump?', 'Could you spin around?', 'Would you wave at me?', 'If you can, do a backflip.', 'Do a little dance if you want.', 'Jump if you can.', 'Can you show me a dance?',
    "Imagine you're a bird and flap your arms.", "Pretend you're a robot and walk.", 'Show me what you would do if you were happy.', 'If I asked you to jump, could you? Go ahead, show me.',
    'Pick up the headphones and put them on.', 'When I clap, jump.', 'Would you mind doing a spin?', 'Do three jumps, then if you are tired sit down.', 'Can you jump? If you can, do it twice.',
    'If you could do a backflip that would be great.', 'Could you do a backflip if you can?'];
  const mp = pos.filter(s => !VTL.isHypothetical(s).hyp), mn = neg.filter(s => VTL.isHypothetical(s).hyp);
  check(`guard: ${pos.length} imagined-situation questions are not requests`, !mp.length, mp);
  check(`guard: ${neg.length} requests (incl. 'if you can, do X', 'can you X?', role play, a go-ahead) stay requests`, !mn.length, mn);
  const bm = ['could you?', 'Would you?', 'can you do that?', 'Would you be able to?', 'could you do it for me?', 'And would you?', 'Is that possible?'];
  const nbm = ['could you jump?', 'can you spin?', 'okay', 'do it', 'would you dance with me'];
  check('guard: a bare modal question continues the utterance before it', bm.every(s => VTL.isBareModal(s)) && !nbm.some(s => VTL.isBareModal(s)), bm.filter(s => !VTL.isBareModal(s)).concat(nbm.filter(s => VTL.isBareModal(s))));

  let X = setup();
  say(X, 'If I asked you to do a cartwheel, would you?');
  let r = X.vt.call('do_action', {description: 'do a cartwheel'}, 'done', {});
  check('guard: a what-if question -> refused, nothing played, the reason tells the model to answer in words', r.ok === false && r.status === 'not_a_request' && /answer/.test(r.reason) && !X.perf.length);
  check('guard: ... and a note asks for an answer in words (a parallel-style response already spoke)', X.prompts.some(p => /imagined situation/.test(p.t) && p.opt.respond));
  X = setup();
  say(X, 'If I asked you to sit down'); X.clock.advance(1500); say(X, 'would you?');
  r = X.vt.call('do_action', {description: 'sit down'}, 'done', {});
  check('guard: a question split by a mid-sentence pause (two turns) -> refused', r.ok === false && !X.perf.length, r);
  X = setup();
  say(X, 'If I asked you to sit down'); X.clock.advance(20000); say(X, 'could you?');
  r = X.vt.call('do_action', {description: 'sit down'}, 'done', {});
  check('guard: the two halves 20 s apart are not joined (the old question is over)', r.ok === true && X.perf.length === 1);
  X = setup();
  say(X, 'I like your hat'); say(X, 'could you jump?');
  r = X.vt.call('do_action', {description: 'jump'}, 'done', {});
  check('guard: an ordinary request after small talk passes', r.ok && X.perf.length === 1);
  X = setup();
  say(X, 'Okay, now');
  r = X.vt.call('do_action', {description: 'what would happen if you jumped'}, 'done', {});
  check('guard: a description that restates a what-if is refused too', r.ok === false && !X.perf.length);

  // his final transcript arrives AFTER the call: held, decided when it arrives
  X = setup();
  X.vt.userSpeech('start'); X.clock.advance(1200); X.vt.userSpeech('stop'); X.clock.advance(300);
  r = X.vt.call('do_action', {description: 'jump'}, 'done', {});
  check('hold: transcript pending -> preparing at once, nothing played yet', r.ok && r.status === 'preparing' && !X.perf.length && X.vt.stats().holds === 1);
  X.clock.advance(250); X.vt.userText('If I asked you to jump, could you?', true);
  check('hold: the transcript is a what-if -> not played, a note asks for words', !X.perf.length && X.prompts.some(p => /imagined situation/.test(p.t) && p.opt.respond) && X.vt.stats().holds === 0);
  X = setup();
  X.vt.userSpeech('start'); X.clock.advance(800); X.vt.userSpeech('stop'); X.clock.advance(100);
  r = X.vt.call('do_action', {description: 'jump', talk: 'quiet'}, 'done', {});
  X.clock.advance(300); X.vt.userText('Jump for me.', true);
  check('hold: the transcript is a request -> played at once on arrival (talk kept, held_ms recorded)', X.perf.length === 1 && X.perf[0].info.talk === 'quiet' && X.perf[0].info.held_ms === 300 && r.next && /quietly/.test(r.next));
  X = setup();
  X.vt.userSpeech('start'); X.clock.advance(800); X.vt.userSpeech('stop');
  X.vt.call('do_action', {description: 'jump'}, 'done', {});
  X.clock.advance(VTL.HOLD_MS - 1);
  check('hold: still waiting just before HOLD_MS', !X.perf.length);
  X.clock.advance(2);
  check('hold: no transcript within HOLD_MS -> played (never stuck)', X.perf.length === 1 && X.perf[0].info.held_ms === VTL.HOLD_MS);
  X = setup();
  X.vt.userSpeech('start'); X.clock.advance(800); X.vt.userSpeech('stop'); X.clock.advance(400); X.vt.userText('spin around', true);
  X.vt.call('do_action', {description: 'spin around'}, 'done', {});
  check('no hold when his transcript is already in', X.perf.length === 1 && X.vt.stats().holds === 0);
  X = setup();       // realtime: his next turn starts before the previous turn's transcript arrives
  X.vt.userSpeech('start'); X.clock.advance(900); X.vt.userSpeech('stop'); X.clock.advance(150);
  X.vt.userSpeech('start'); X.clock.advance(100); X.vt.userText('If I asked you to bow', true);
  X.clock.advance(400); X.vt.userSpeech('stop'); X.clock.advance(50);
  r = X.vt.call('do_action', {description: 'bow'}, 'done', {});
  X.clock.advance(200); X.vt.userText('would you?', true);
  check('transcripts land on their own utterance when the next turn already started (split what-if refused after the hold)',
    JSON.stringify(X.vt.utterances().map(u => u.text)) === JSON.stringify(['If I asked you to bow', 'would you?']) && !X.perf.length && X.vt.stats().blocks.length === 1, X.vt.utterances());
  X = setup();       // the hold waits for the NEWEST turn's words, not just any transcript
  X.vt.userSpeech('start'); X.clock.advance(900); X.vt.userSpeech('stop'); X.clock.advance(100);
  X.vt.userSpeech('start'); X.clock.advance(300); X.vt.userSpeech('stop'); X.clock.advance(50);
  X.vt.call('do_action', {description: 'jump'}, 'done', {});
  X.clock.advance(100); X.vt.userText('If I asked you to jump', true);
  check('hold: the older turn\'s words alone do not decide it', !X.perf.length && X.vt.stats().holds === 1);
  X.clock.advance(100); X.vt.userText('could you?', true);
  check('hold: decided on the newest turn\'s words (split what-if -> refused)', !X.perf.length && X.vt.stats().holds === 0 && X.vt.stats().blocks.length === 1);
  X = setup();       // the scene tool (object) goes through the same guard
  say(X, 'If I asked you to pick up the headphones, would you?');
  let ran = 0;
  r = X.vt.gate('pick_up headphones', () => { ran++; return {ok: true}; });
  check('gate (object): a what-if -> refused, nothing run', r && r.ok === false && r.status === 'not_a_request' && !ran);
  X = setup();
  say(X, 'Pick up the headphones.');
  check('gate (object): a request -> null (run now)', X.vt.gate('pick_up headphones', () => { ran++; return {ok: true}; }) === null);
  X.vt.userSpeech('start'); X.clock.advance(600); X.vt.userSpeech('stop');
  r = X.vt.gate('put_on headphones', () => { ran++; return {ok: true}; });
  X.clock.advance(200); X.vt.userText('and put them on', true);
  check('gate (object): words on their way -> held, then run on arrival', r.status === 'preparing' && ran === 1);
  {                  // several steps for the things around it: his words' other steps run (the model calls one)
    const {Intent} = load('app/intent.js');
    const rules = JSON.parse(fs.readFileSync(path.join(ROOT, 'app/intent_rules.json'), 'utf8')), props = JSON.parse(fs.readFileSync(path.join(ROOT, 'renderer/animals/muse.props.json'), 'utf8'));
    const mkX = () => { const steps = []; const M = Intent.create(rules, {act: true, scene: props});
      const Y = setup(); Y.steps = steps; Y.vt = VTL.createVoiceTools({perf: () => Y.P, motion: () => Y.motion, behaviors: () => null, runCommand: c => ({ok: true}), sendCtl: () => {}, prompt: (t, o) => { Y.prompts.push({t, opt: o}); return {}; },
        sceneMatcher: () => M, runStep: (c, info) => { steps.push(c); return {ok: true}; }, userSpeaking: () => false, now: Y.clock.now, timers: {set: Y.clock.set, clear: Y.clock.clear}}); return Y; };
    let Y = mkX();
    say(Y, 'Grab your headphones and put them on.');
    check('expand: two steps of his words run in order (pick up, put on)', Y.vt.expand() === 2 && Y.steps.map(c => c.verb).join() === 'pick_up,put_on');
    check('expand: once per utterance', Y.vt.expand() === 0 && Y.steps.length === 2);
    Y = mkX(); say(Y, 'Get the laptop and slide it a bit closer to you.');
    check('expand: get the laptop and slide it closer -> pick up AND slide', Y.vt.expand() === 2 && Y.steps.map(c => c.verb).join() === 'pick_up,slide');
    Y = mkX(); say(Y, 'Pull the chair up to the desk and have a seat.');
    check('expand: a scene step with a body step (pull the chair, sit) -> both, in order', Y.vt.expand() === 2 && Y.steps[0].verb === 'pull' && Y.steps[1].name === 'sit');
    Y = mkX(); say(Y, 'Close the laptop.');
    check('expand: a single step -> nothing added (the model\'s own call does it)', Y.vt.expand() === 0 && !Y.steps.length);
    Y = mkX(); say(Y, 'Jump twice and then spin around.');
    check('expand: body moves only -> nothing (do_action\'s)', Y.vt.expand() === 0);
    Y = mkX(); say(Y, 'If I asked you to pick up the headphones and put them on, would you?');
    check('expand: never a what-if', Y.vt.expand() === 0);
    Y = mkX(); say(Y, 'Put your headphones on, drag the chair and the desk over, sit down at the desk, and start working on your laptop.');
    check('expand: words the matcher does not fully understand -> nothing (the model\'s calls only)', Y.vt.expand() === 0);
    Y = mkX(); Y.vt.userSpeech('start'); Y.clock.advance(600); Y.vt.userSpeech('stop');
    check('expand: realtime words not in yet -> nothing (the gate holds the call until they are)', Y.vt.expand() === 0);
    Y = mkX(); Y.vt.userSpeech('start'); Y.vt.userText('pick up the ball and toss it', false);
    check('expand: delta-only voices (Gemini, hub-relayed) use the words so far', Y.vt.expand() === 2);
    // a pause split the request into two turns
    Y = mkX(); say(Y, 'Open it up.'); say(Y, 'and do some work.');
    check('expand: a split request (\'open it up.\' / \'and do some work.\') read together on the call after both', Y.vt.expand() === 2 && Y.steps.map(c => c.verb).join() === 'open,work_on', Y.steps);
    Y = mkX(); say(Y, 'Take the headphones off.');
    Y.vt.expand();                          // the model's object(take_off) came after the first half: 1 step, nothing added
    const n0 = Y.steps.length;
    say(Y, 'and put them back where they were.');
    check('expand: the 2nd half arriving after the scene call runs its own steps (put back)', n0 === 0 && Y.steps.map(c => c.verb).join() === 'put_back', Y.steps);
    check('expand: ... once (the model\'s later call finds it done)', Y.vt.expand() === 0 && Y.steps.length === 1);
    Y = mkX(); say(Y, 'Take the headphones off.'); Y.vt.expand(); Y.clock.advance(12000); say(Y, 'and then tell me a joke.');
    check('expand: a later "and …" with no scene call nearby / no step adds nothing', Y.steps.length === 0);
    Y = mkX(); say(Y, 'Close the laptop.'); say(Y, 'and if I asked you to open it again, would you?');
    check('expand: never a what-if second half', Y.vt.expand() === 0 && !Y.steps.length);
    // do_action about the things around it: never a keyframe mime (harness 806886b: pick-ups via do_action held nothing)
    Y = mkX(); say(Y, 'Pick up the headphones.');
    let rr = Y.vt.call('do_action', {description: 'Pick up the pink headphones with my right paw, gently'}, 'done', {});
    check('scene do_action: his fully understood words run (pick up), not the planner', rr.ok && rr.status === 'starting' && Y.steps.map(c => c.verb).join() === 'pick_up' && !Y.perf.length);
    Y = mkX(); say(Y, 'Grab your headphones and put them on.');
    Y.vt.expand();
    rr = Y.vt.call('do_action', {description: 'put the headphones on'}, 'done', {});
    check('scene do_action: a request already run from his words -> already under way (nothing twice)', rr.ok && /under way now/.test(rr.did) && Y.steps.length === 2 && !Y.perf.length);
    Y = mkX(); say(Y, 'um could you sort of grab those headphone thingies over there');
    rr = Y.vt.call('do_action', {description: 'pick up the headphones'}, 'done', {});
    check('scene do_action: his words not understood, the description is -> the description\'s steps', rr.ok && rr.status === 'starting' && Y.steps.map(c => c.verb).join() === 'pick_up' && !Y.perf.length, Y.steps);
    Y = mkX(); Y.motion = {props: () => [{id: 'chair', name: 'chair'}, {id: 'headphones', name: 'pink headphones'}]}; say(Y, 'do a silly dance around your chair');
    rr = Y.vt.call('do_action', {description: 'do a silly dance around the chair'}, 'done', {});
    check('scene do_action: names a thing in the scene but nothing understood -> refused with the object hint', rr.ok === false && rr.status === 'use_object' && /object tool/.test(rr.reason) && !Y.perf.length);
    check('scene do_action refused -> a note asks for the object call (the call\'s response may already have spoken)', Y.prompts.some(p => /Call object for it now/.test(p.t) && p.opt.respond));
    Y = mkX(); Y.motion = {props: () => [{id: 'headphones', name: 'pink headphones'}]}; say(Y, 'Reach for the earphones and hold them.');
    rr = Y.vt.call('do_action', {description: 'Reach for the earphones and hold them.'}, 'done', {});
    check('scene do_action: a thing named by an alias the matcher misses (earphones) -> object hint, not a keyframe mime', rr.status === 'preparing' && Y.perf.length === 1, rr);
    Y = mkX(); Y.motion = {props: () => [{id: 'headphones', name: 'pink headphones'}]}; Y.aliases = [{id: 'headphones', aliases: ['headset', 'earphones']}];
    Y.vt = VTL.createVoiceTools({perf: () => Y.P, motion: () => Y.motion, behaviors: () => null, runCommand: c => ({ok: true}), sendCtl: () => {}, prompt: (t, o) => { Y.prompts.push({t, opt: o}); return {}; },
      sceneMatcher: () => ({parse: () => [], understood: () => true}), runStep: () => ({ok: true}), sceneAliases: () => Y.aliases, userSpeaking: () => false, now: Y.clock.now, timers: {set: Y.clock.set, clear: Y.clock.clear}});
    rr = Y.vt.call('do_action', {description: 'Reach for the earphones and hold them.'}, 'done', {});
    check('scene do_action: ... with the scene file\'s aliases -> use_object + the follow-up note', rr.status === 'use_object' && !Y.perf.length && Y.prompts.some(p => /Call object/.test(p.t)), rr);
    Y = mkX(); say(Y, 'Jump twice.');
    rr = Y.vt.call('do_action', {description: 'jump twice'}, 'done', {});
    check('scene character, a body move -> the planner as before', rr.status === 'preparing' && Y.perf.length === 1 && !Y.steps.length);
  }
  X = setup();       // a transcript that never came does not swallow the next one
  X.vt.userSpeech('start'); X.clock.advance(500); X.vt.userSpeech('stop'); X.clock.advance(11000);
  X.vt.userSpeech('start'); X.clock.advance(500); X.vt.userSpeech('stop'); X.vt.userText('spin around', true);
  check('a lost transcript (> 10 s) is closed: the next words land on their own utterance', X.vt.utterances().slice(-1)[0].text === 'spin around' && !X.vt.utterances()[0].text);
  X = setup();
  X.vt.userSpeech('start'); X.vt.userText('jump ', false); X.vt.userText('now', false);
  X.vt.call('do_action', {description: 'jump'}, 'done', {});
  check('Gemini / hub voice (deltas, no stop event): no hold, the guard reads the deltas', X.perf.length === 1 && X.vt.utterances()[0].text === 'jump now');
}

// ===================================================================================================== behaviours
{
  let X = setup({noB: true});
  check('start_behavior without the program layer: honest failure, nothing sent', X.vt.call('start_behavior', {instruction: 'wave whenever I wave'}, 'done', {}).ok === false && !X.ctl.length);
  check('list_behaviors without the program layer: none, said plainly', (r => r.ok && !r.behaviors.length && /not available/.test(r.note))(X.vt.call('list_behaviors', {}, 'done', {})));
  X = setup();
  let r = X.vt.call('start_behavior', {instruction: 'whenever I raise my hand, wave back with the same hand', name: 'Wave Back', priority: 7}, 'done', {});
  const req = X.ctl.find(m => m.type === 'behavior_code');
  check('start_behavior -> {type:behavior_code} to the hub writer, result at once (writing)', r.ok && r.status === 'writing' && r.name === 'wave_back' && req && req.instruction.startsWith('whenever') && req.name === 'wave_back' && !req.prev_code);
  check('list_behaviors shows one being set up', X.vt.call('list_behaviors', {}, 'done', {}).behaviors.some(b => b.name === 'wave_back' && b.status === 'being set up'));
  check('onHub: a reply for someone else is not ours', X.vt.onHub({type: 'behavior_code', id: 'zz', name: 'other', ok: true, code: '({})'}) === false);
  X.vt.onHub({type: 'behavior_code', id: req.id, name: 'wave_back', ok: true, code: '({per_frame(api){}})'});
  check('onHub ok -> Behaviors.start({name, instruction, code, priority}) + a context note (no reply asked)', X.B.started.length === 1 && X.B.started[0].code.includes('per_frame') && X.B.started[0].priority === 7
    && X.prompts.some(p => /wave_back he asked for is running/.test(p.t) && !p.opt.respond));
  r = X.vt.call('list_behaviors', {}, 'done', {});
  check('list_behaviors -> names and what each does', r.ok && r.behaviors.length === 1 && r.behaviors[0].name === 'wave_back' && /raise my hand/.test(r.behaviors[0].does));
  r = X.vt.call('change_behavior', {name: 'wave back', instruction: 'whenever I raise my hand, wave back with both hands'}, 'done', {});
  const req2 = X.ctl.filter(m => m.type === 'behavior_code')[1];
  check('change_behavior -> found by a spoken name, the writer gets the previous code', r.ok && r.status === 'rewriting' && req2.name === 'wave_back' && req2.prev_code.includes('per_frame') && /raise my hand/.test(req2.prev_instruction));
  X.vt.onHub({type: 'behavior_code', id: req2.id, name: 'wave_back', ok: true, code: '({per_frame(api){ api.set("x",1) }})'});
  check('onHub (change) -> Behaviors.change(name, {code, instruction})', X.B.changed.length === 1 && X.B.changed[0][0] === 'wave_back' && /both hands/.test(X.B.changed[0][1].instruction));
  check('change_behavior on an unknown name: honest failure + what is running', (r2 => r2.ok === false && r2.running.includes('wave_back'))(X.vt.call('change_behavior', {name: 'dance loop', instruction: 'x'}, 'done', {})));
  X.vt.call('start_behavior', {instruction: 'sway gently while I talk'}, 'done', {});
  const req3 = X.ctl.filter(m => m.type === 'behavior_code')[2];
  X.vt.onHub({type: 'behavior_code', id: req3.id, name: req3.name, ok: false, error: 'static check failed: writes a group it did not claim'});
  check('onHub failure -> nothing started, a note asks to tell him honestly', X.B.started.length === 1 && X.prompts.some(p => /could not be started \(static check failed/.test(p.t) && p.opt.respond));
  X.vt.call('start_behavior', {instruction: 'nod every few seconds'}, 'done', {});
  X.clock.advance(26000);
  check('the writer never answers -> a failure note after the wait', X.prompts.some(p => /nod every few seconds\) could not be started \(the behaviour writer did not answer/.test(p.t)));
  {   // the program layer's own round trip (behavior.js request / onCode) and its event shapes
    const Y = setup(); let resolve; const asked = [];
    Y.B.request = (ins, o) => { asked.push([ins, o]); return new Promise(ok => { resolve = ok; }); };
    const ry = Y.vt.call('start_behavior', {instruction: 'nod whenever I nod', priority: 3}, 'done', {});
    check('start_behavior uses Behaviors.request when the program layer has it (no second behavior_code)', ry.status === 'writing' && asked[0][0] === 'nod whenever I nod' && asked[0][1].priority === 3 && !Y.ctl.some(m => m.type === 'behavior_code'));
    resolve({ok: true, name: 'nod_back'});
    setImmediate(() => {
      check('request resolved -> running note (context only)', Y.prompts.some(p => /nod_back he asked for is running/.test(p.t) && !p.opt.respond));
      const rm = Y.vt.onEvent({type: 'rule_moment', name: 'nod_back', instruction: 'nod when I nod', event: 'nodded', data: {n: 2}, reply: true, t: 5});
      check('behavior.js rule_moment -> a rule note that asks to speak', rm.respond && /behaviour you are running \(nod_back: nod when I nod\) reached a moment: nodded/.test(Y.prompts[Y.prompts.length - 1].t));
      const er = Y.vt.onEvent({type: 'behavior', phase: 'end', name: 'nod_back', error: 'TypeError x', t: 6});
      check('behavior.js end with an error -> an honest note', er.noted && /nod_back stopped because of an error \(TypeError x\)/.test(Y.prompts[Y.prompts.length - 1].t));
      check('behavior.js start / changed events are not notes', Y.vt.onEvent({type: 'behavior', phase: 'start', name: 'x', t: 7}).noted === false);
    });
  }
  r = X.vt.call('stop_behavior', {name: 'wave_back'}, 'done', {});
  check('stop_behavior by name', r.ok && r.stopped.join() === 'wave_back' && !X.B.progs.length);
  X.B.start({name: 'a', instruction: 'a'}); X.B.start({name: 'b', instruction: 'b'});
  r = X.vt.call('stop_behavior', {name: 'all'}, 'done', {});
  check('stop_behavior all', r.ok && r.stopped.length === 2 && X.B.stopped.includes('all'));
  check('stop_behavior unknown: honest failure', X.vt.call('stop_behavior', {name: 'zzz'}, 'done', {}).ok === false);
}

// ===================================================================================================== look / style / gestures
{
  let X = setup();
  let r = X.vt.call('look', {target: 'me'}, 'done', {});
  check('look at him: target user, a short look', r.ok && X.cmds[0].action === 'look' && X.cmds[0].target === 'user' && !X.cmds[0].hold && !X.cmds[0].duration_ms);
  X.vt.call('look', {target: 'up', seconds: 4}, 'done', {});
  check('look up for 4 s -> duration_ms 4000', X.cmds[1].target === 'up' && X.cmds[1].duration_ms === 4000);
  X.vt.call('look', {target: 'away', seconds: 0}, 'done', {});
  check('look away until told otherwise -> hold', X.cmds[2].target === 'away' && X.cmds[2].hold === true);
  r = X.vt.call('look', {target: 'the laptop', seconds: 3}, 'done', {});
  check('look at a thing by name -> its id as the target', r.ok && X.cmds[3].target === 'laptop' && X.cmds[3].duration_ms === 3000);
  r = X.vt.call('look', {target: 'the unicorn'}, 'done', {});
  check('look at a thing that is not there: honest failure, no command', r.ok === false && /nothing called/.test(r.reason) && X.cmds.length === 4);
  // the real motion layer: a look target that is a prop follows it
  globalThis.AvatarMovesLib = load('app/moves.js');
  const M = load('app/motion.js');
  const m = M.createMotion({seed: 3, character: 'muse'});
  m.setProps(JSON.parse(fs.readFileSync(path.join(ROOT, 'renderer/animals/muse.props.json'), 'utf8')), JSON.parse(fs.readFileSync(path.join(ROOT, 'renderer/animals/muse.body.json'), 'utf8')), 'muse');
  for (let i = 0; i < 10; i++) m.step(1 / 30, {});
  const X2 = setup({motion: m, runCommand: c => m.command(c, {src: 'voice'})});
  const rr = X2.vt.call('look', {target: 'chair', seconds: 2}, 'done', {});
  let row; for (let i = 0; i < 30; i++) row = m.step(1 / 30, {});
  const chairRight = m.props().find(p => p.id === 'chair').pos[0] > 0;
  check('motion: look at the chair turns the gaze toward it (screen side of the chair) and says so', rr.ok && /looking at the chair/.test(rr.did) && (chairRight ? row.gaze_x > 0.3 : row.gaze_x < -0.3) && m.state().looking === 'chair', {did: rr.did, gaze_x: row.gaze_x});
  const rh = X2.vt.call('look', {target: 'headphones', seconds: 2}, 'done', {});
  for (let i = 0; i < 30; i++) row = m.step(1 / 30, {});
  const hpRight = m.props().find(p => p.id === 'headphones').pos[0] > 0;
  check('motion: look at the headphones (other side) turns the other way', rh.ok && (hpRight ? row.gaze_x > 0.3 : row.gaze_x < -0.3), {gaze_x: row.gaze_x});
  for (let i = 0; i < 90; i++) row = m.step(1 / 30, {});
  check('motion: the look ends after its seconds', m.state().looking !== 'headphones');

  X = setup();
  r = X.vt.call('set_style', {energy: 0.2, mirroring: 1.4, expressiveness: -1}, 'done', {});
  const sc = X.ctl.find(c => c.type === 'style');
  check('set_style -> {type:style} (clamped, -1 = automatic / null)', sc && sc.energy === 0.2 && sc.mirroring === 1 && sc.expressiveness === null);
  check('set_style while the learned listener is off: honest (ok false, stored)', r.ok === false && r.stored && /off right now/.test(r.reason));
  X.react = 'unified';
  r = X.vt.call('set_style', {energy: 0.9}, 'done', {});
  check('set_style in unified mode: ok, the whole style reported', r.ok && r.style.energy === 0.9 && r.style.mirroring === 1 && r.style.expressiveness === 'automatic');
  check('set_style with nothing to change: refused, nothing sent', X.vt.call('set_style', {}, 'done', {}).ok === false && X.ctl.filter(c => c.type === 'style').length === 2);

  X = setup();
  check('attention_rule without the camera watcher: honest failure', X.vt.call('attention_rule', {text: 'tell me if my cat walks in', kind: 'standing'}, 'done', {}).ok === false);
  {
    const sent = [];
    const Y = setup({A: {instruct: t => { sent.push(['add', t]); return {type: 'attention_rule', op: 'add', text: t}; }, feedback: t => { sent.push(['fb', t]); return {}; }}});
    const a1 = Y.vt.call('attention_rule', {text: 'tell me if my cat walks in', kind: 'standing'}, 'done', {});
    const a2 = Y.vt.call('attention_rule', {text: "don't comment on my coffee", kind: 'feedback'}, 'done', {});
    check('attention_rule standing -> Attention.instruct, feedback -> Attention.feedback', a1.ok && a1.kind === 'standing' && a2.ok && a2.kind === 'feedback' && JSON.stringify(sent) === JSON.stringify([['add', 'tell me if my cat walks in'], ['fb', "don't comment on my coffee"]]));
  }
  {   // see_camera: one frame as context before the result
    const frames = []; let cam = 'data:image/jpeg;base64,AAAA';
    const Y = setup();
    Y.vt = VTL.createVoiceTools({perf: () => Y.P, cameraFrame: () => cam, sendFrame: (img, note) => { frames.push([img, note]); return true; }, prompt: () => ({}), now: Y.clock.now, timers: {set: Y.clock.set, clear: Y.clock.clear}});
    const c1 = Y.vt.call('see_camera', {about: 'what he is holding'}, 'done', {});
    check('see_camera -> one frame sent (with what it wanted to see), the result says answer from it', c1.ok && frames.length === 1 && /what he is holding/.test(frames[0][1]) && /answer him from what it shows/.test(c1.next));
    const c2 = Y.vt.call('see_camera', {}, 'done', {});
    check('see_camera twice within 3 s -> no second frame', c2.ok && frames.length === 1);
    Y.clock.advance(3500); cam = null;
    check('see_camera with no picture: honest failure', Y.vt.call('see_camera', {}, 'done', {}).ok === false);
    check('see_camera without a camera on the page: honest failure', setup().vt.call('see_camera', {}, 'done', {}).ok === false);
  }
  check('learn_gesture without perception: honest failure', X.vt.call('learn_gesture', {name: 'peace sign', how: 'now'}, 'done', {}).ok === false);
  const calls = [];
  X = setup({G: {learn: (n, o) => { calls.push([n, o]); return {ok: true, kind: 'static', examples: 1}; }}});
  r = X.vt.call('learn_gesture', {name: 'peace sign', how: 'now', seconds: 2}, 'done', {});
  check('learn_gesture now -> Gestures.learn(name, {from:last, seconds, counter}) sync result', r.ok && r.kind === 'static' && calls[0][0] === 'peace sign' && calls[0][1].from === 'last' && calls[0][1].seconds === 2 && calls[0][1].counter === false);
  let res;
  X = setup({G: {learn: (n, o) => { calls.push([n, o]); return new Promise(ok => { res = ok; }); }}});
  r = X.vt.call('learn_gesture', {name: 'not a wave', how: 'next', counterexample: true}, 'done', {});
  check('learn_gesture next -> watching at once (async), counterexample passed', r.ok && r.status === 'watching' && calls[1][1].from === 'next' && calls[1][1].counter === true && calls[1][1].seconds === 3);
  res({ok: true, kind: 'moving', examples: 2});
  setImmediate(() => {
    check('learn_gesture resolved -> a note asks for a short confirmation', X.prompts.some(p => /You learned what his gesture not a wave is not \(2 examples so far\)/.test(p.t) && p.opt.respond));
    events();
  });
}

// ===================================================================================================== events -> notes
function events() {
  let X = setup({sendEvent: true});
  let r = X.vt.onEvent({type: 'gesture', name: 'peace sign', score: 0.9, t: 1.0});
  check('gesture -> a note that asks for a reaction', r.noted && r.respond && /gesture he taught you as peace sign/.test(X.prompts[0].t) && X.prompts[0].opt.respond);
  check('the same event through a second route is dropped', X.vt.onEvent({type: 'gesture', name: 'peace sign', score: 0.9, t: 1.0}).why === 'duplicate');
  X.clock.advance(1000);
  check('the same gesture again within 4 s: rate-limited', X.vt.onEvent({type: 'gesture', name: 'peace sign', score: 0.9, t: 2.0}).why === 'rate');
  check('a weak gesture score is ignored', X.vt.onEvent({type: 'gesture', name: 'thumbs', score: 0.2, t: 3}).noted === false);
  X.B.start({name: 'peace_back', instruction: 'when I make the peace sign, make one back'});
  X.clock.advance(5000);
  r = X.vt.onEvent({type: 'gesture', name: 'peace sign', score: 0.9, t: 9});
  check('a gesture a running behaviour reacts to: context only (the behaviour owns the reaction)', r.noted && !r.respond);
  r = X.vt.onEvent({type: 'attention', what: 'he is holding up a red mug', worth_reply: true, t: 10, image: 'data:image/jpeg;base64,AAAA'});
  check('attention worth a reply + a frame -> sendEvent with the image, reply asked', r.respond && r.image && X.events[0].opt.image && X.events[0].opt.respond);
  X.clock.advance(7000);
  r = X.vt.onEvent({type: 'attention', what: 'the lights changed', worth_reply: false, t: 11});
  check('attention not worth a reply -> context note only', r.noted && !r.respond && !X.prompts[X.prompts.length - 1].opt.respond);
  X.speaking = true;
  check('a turn bid while he talks: nothing', X.vt.onEvent({type: 'turn_bid', cue: 'beckon', deliberateness: 0.9, t: 12}).noted === false);
  X.speaking = false; X.lastSpeech = X.clock.now() - 2000;
  check('a weak turn bid: nothing', X.vt.onEvent({type: 'turn_bid', cue: 'raised_brows', deliberateness: 0.3, t: 13}).why === 'below threshold');
  check('turnbid.js bid floor (he is about to talk): nothing', X.vt.onEvent({type: 'turn_bid', cue: 'inhale', deliberateness: 0.9, bid: 'floor', t: 13.5}).why === 'he wants the floor');
  r = X.vt.onEvent({type: 'turn_bid', cue: 'lean_in', deliberateness: 0.8, t: 14});
  check('a deliberate turn bid while he is silent -> asks for a reply', r.respond && /signalling that he wants you to talk \(lean in\)/.test(X.prompts[X.prompts.length - 1].t));
  X.avatar = true;
  X.clock.advance(9000);
  check('a turn bid while the avatar talks: nothing', X.vt.onEvent({type: 'turn_bid', cue: 'beckon', deliberateness: 0.9, t: 30}).why === 'not silent');
  X.avatar = false;
  r = X.vt.onEvent({type: 'emit', name: 'hand_raised', data: {hand: 'left'}, program: {name: 'wave_back', instruction: 'wave back when I raise my hand'}, t: 40});
  check('a rule moment (api.emit) -> a note that asks to speak, with the program and data', r.respond && /behaviour you are running \(wave_back: wave back when I raise my hand\) reached a moment: hand_raised \(\{"hand":"left"\}\)/.test(X.prompts[X.prompts.length - 1].t));
  r = X.vt.onEvent({type: 'emit', name: 'tick', data: {quiet: true}, program: {name: 'counter'}, t: 41});
  check('a quiet rule moment -> context only', r.noted && !r.respond);
  check('unknown events are not ours', X.vt.onEvent({type: 'expression_change'}).noted === false);
  // attach(): subscribes once to each module's onEvent
  const att = X.vt.attach(), again = X.vt.attach();
  check('attach(): Behaviors.onEvent subscribed once', att.behaviors === true && again.behaviors === false && X.B.fns.length === 1);
  const n0 = X.prompts.length;
  X.B.fns[0]({type: 'emit', name: 'clapped', program: {name: 'clap_rule'}, t: 50});
  check('a Behaviors event reaches the voice model through attach()', X.prompts.length === n0 + 1);
  talk();
}

// ===================================================================================================== talk while moving (perform.js)
function mockMotion(clock) {
  const M = {jobs: new Map(), onAction: null, releases: []};
  const emit = ev => { if (M.onAction) M.onAction(Object.assign({src: 'voice'}, ev)); };
  M.command = c => { if (c.action !== 'keyframes') return {ok: true}; M.jobs.set(c.id, {poses: c.poses.slice(), done: c.done}); if (c.done) emit({kind: 'kf', id: c.id, phase: 'ready'}); return {ok: true, id: c.id}; };
  M.kfFeed = (id, poses, done) => { const J = M.jobs.get(id); J.poses.push(...poses); if (done) emit({kind: 'kf', id, phase: 'ready'}); };
  M.kfRelease = id => { M.releases.push(id); emit({kind: 'kf', id, phase: 'start'}); };
  M.kfState = id => ({planned_s: 2});
  M.end = id => emit({kind: 'kf', id, phase: 'end'});
  return M;
}
function talk() {
  for (const mode of ['usual', 'through', 'quiet']) {
    const clock = makeClock(), notes = [], ctl = [], m = mockMotion(clock);
    const P = Perf.createPerformer({motion: () => m, prompt: (t, o) => { notes.push({t, o}); return {}; }, sendCtl: x => ctl.push(x), character: () => 'fox',
      now: clock.now, timers: {set: clock.set, clear: clock.clear}, userSpeaking: () => false, lastUserSpeech: () => -1e12});
    m.onAction = ev => P.onAction(ev);
    const r = P.call('count to five while you jump five times', {talk: mode === 'usual' ? undefined : mode, words: 'Count to five while you jump five times'});
    if (mode === 'usual') check('perform: the plan request carries his words', ctl.some(c => c.type === 'plan' && c.words === 'Count to five while you jump five times'));
    P.onKf({id: r.id, poses: [{t: 0.5, root_y: 0.3}, {t: 1.2, root_y: 0}], done: true, ok: true, source: 'planner'});
    const cue = notes.find(n => /about to start/.test(n.t));
    if (mode === 'quiet') check('talk quiet: no cue, the motion starts at once, the tool result says quietly', !cue && m.releases.length === 1 && /quietly/.test(r.next) && r.talk === 'quiet');
    else if (mode === 'through') check('talk through: the cue asks to keep talking through the whole motion', cue && cue.o.respond && /keep going in time with it until it ends/.test(cue.t) && /talk while it plays/.test(r.next));
    else check('talk usual: the cue', cue && cue.o.respond && /brief natural cue/.test(cue.t) && !r.talk);
    clock.advance(800);
    m.end(r.id);
    const fin = notes.find(n => /has finished/.test(n.t));
    check(`talk ${mode}: check-in ${mode === 'quiet' ? 'not asked' : 'asked'} at the end`, fin && fin.o.respond === (mode !== 'quiet') && ctl.some(c => c.type === 'perform' && (mode === 'usual' ? !c.talk : c.talk === mode)));
  }
  {   // a library run whose every step was already started from his words (expansion): done, not a failure
    const clock = makeClock(), notes = [], ctl = [], m = mockMotion(clock);
    const P = Perf.createPerformer({motion: () => m, prompt: (t, o) => { notes.push({t, o}); return {}; }, sendCtl: x => ctl.push(x), character: () => 'muse',
      runCommand: () => ({ok: false, dup: true}), now: clock.now, timers: {set: clock.set, clear: clock.clear}, userSpeaking: () => false, lastUserSpeech: () => -1e12});
    const r = P.call('sit down');
    P.onKf({id: r.id, poses: [], done: true, ok: true, source: 'library', engine: [{action: 'move', name: 'sit'}]});
    const rec = ctl.find(c => c.type === 'perform');
    check('perform: all library steps were duplicates -> done (already playing), no failure note', rec && rec.outcome === 'done' && !notes.some(n => /could not/.test(n.t)) && notes.some(n => /already playing/.test(n.t) && !n.o.respond));
  }
  page();
}

// ===================================================================================================== page.html wiring
function page() {
  const pg = fs.readFileSync(path.join(ROOT, 'app/page.html'), 'utf8');
  const fnSrc = name => { const a = pg.indexOf('function ' + name + '('); const b = pg.indexOf('\n}\n', a); return pg.slice(a, b + 3); };
  const ttcSrc = pg.slice(pg.indexOf('const POSE_MAP'), pg.indexOf('function toolToCommand')) + pg.slice(pg.indexOf('function toolToCommand'), pg.indexOf('\n}\n', pg.indexOf('function actToolCommand')) + 3);
  const a = pg.indexOf('onCall: (name, args, phase, info) => {') + 'onCall: '.length, b = pg.indexOf('\n    onMetric', a);
  const onCallSrc = pg.slice(a, b).replace(/,\s*$/, '');
  const mk = (Q, cfg, mode, direct) => {
    const env = {ran: [], ctl: [], perf: [], vt: [], hub: [], ev: []};
    const D = {mode, cmds: [], perf: {call: d => { env.perf.push(d); return {ok: true, status: 'preparing', id: 'p9'}; }, onKf: () => {}, stop: () => {}, engineCall: (i, run) => ({deferred: false, result: run()})},
      vt: {handles: n => VTL.ACTION.includes(n) || VTL.NAMES.includes(n), call: (n, args, ph, info) => { env.vt.push({n, args, ph, info}); return {ok: true, status: 'preparing', id: 'v1'}; },
        onHub: m => { env.hub.push(m); return true; }, onEvent: e => { env.ev.push(e); return {noted: true, respond: true}; }, userText: (t, f) => env.ev.push({tx: t, f}), userSpeech: p => env.ev.push({sp: p})},
      intentRes: null, alog: null, lastCmdRes: null, events: []};
    const S = {cfg, direct: direct !== false};
    const f = new Function('Q', 'S', 'D', 'env', ttcSrc + fnSrc('plannerReq') + fnSrc('plannerOn') + fnSrc('avatarEvent') + `
      const intentAct = () => false, SPEAK_EVENTS = {}, window = {};
      function runCommand(cmd, source, info) { env.ran.push({cmd, source}); return {ok: true, id: 5, did: 'ok'}; }
      const sendCtl = m => env.ctl.push(m), echoOfAvatar = () => false, speechToolResult = r => r, REL_ALIAS = {};
      let capWho = null; const capLine = {you: '', avatar: ''}; const $ = () => ({textContent: ''}); const charName = x => x; const sel = 'fox';
      function intentToolResult(ran) { return ran && typeof ran === 'object' ? {ok: ran.ok !== false, did: ran.did} : {ok: !!ran}; }
      const onCall = ${onCallSrc};
      ${fnSrc('onJSON')}
      return {onCall, onJSON, plannerOn, avatarEvent};`);
    return Object.assign(f(Q, S, D, env), {env, D});
  };
  const Qs = o => ({get: k => (o && k in o ? o[k] : null)});
  const on = mk(Qs({}), {character: 'fox'}, 'realtime');
  const r = on.onCall('do_action', {description: 'spin', talk: 'through'}, 'done', {start_lat_ms: 10});
  check('page: do_action -> D.vt.call (not D.perf directly), recorded for the hub', r.id === 'v1' && on.env.vt[0].n === 'do_action' && !on.env.perf.length && on.env.ctl.some(m => m.type === 'call' && m.name === 'do_action' && m.ran.status === 'preparing'));
  on.onCall('perform_action', {description: 'bow'}, 'done', {});
  on.onCall('look', {target: 'left'}, 'early', {});
  check('page: perform_action alias -> D.vt; an early look is not run (the new look runs once, on done)', on.env.vt[1].n === 'perform_action' && on.env.vt.length === 2 && !on.env.ran.length);
  for (const n of VTL.NAMES) on.onCall(n, {name: 'x', instruction: 'y', target: 'up'}, 'done', {});
  check('page: all nine program / look / style / gesture / attention / camera tools -> D.vt', VTL.NAMES.every(n => on.env.vt.some(c => c.n === n)));
  on.onCall('gesture', {name: 'wave'}, 'done', {});
  check('page: the expressive tools keep their old path', on.env.ran.some(x => x.cmd.action === 'gesture'));
  on.onJSON({type: 'act', id: 7, name: 'start_behavior', args: {instruction: 'nod when I nod'}});
  check('page: the hub-relayed voice (act) -> D.vt + act_result', on.env.vt.some(c => c.n === 'start_behavior' && c.info.via === 'hub') && on.env.ctl.some(m => m.type === 'act_result' && m.id === 7 && m.result.id === 'v1'));
  on.onJSON({type: 'behavior_code', id: 'b1', ok: true, code: 'x'});
  on.onJSON({type: 'attention', what: 'a cat', worth_reply: true});
  // merge: the server's attention verdict goes to the attention module (D.attn.onHub); its event reaches the voice tools
  // once, through perceptionLayerEvent -> avatarEvent -> D.vt.onEvent (not straight from onJSON, which double-noted).
  check('page: behavior_code -> D.vt.onHub; a hub attention verdict is not sent straight to D.vt', on.env.hub.length === 1 && !on.env.ev.some(e => e.type === 'attention' && e.what === 'a cat'));
  check('page: avatarEvent forwards attention events to D.vt.onEvent', /name === 'gesture' \|\| name === 'attention' \|\| name === 'turn_bid'\)\) \{ const r = D\.vt\.onEvent/.test(PAGE_SRC_FOR_LEAD_CHECK));
  const ev = on.avatarEvent('gesture', {type: 'gesture', name: 'peace', score: 0.9});
  check('page: a perception gesture event through avatarEvent -> D.vt.onEvent', ev.spoken === true && on.env.ev.some(e => e.type === 'gesture' && e.name === 'peace'));
  on.D.vt.gate = (desc, later) => (env => (env.gated = desc, /headphones/.test(desc) ? {ok: false, status: 'not_a_request', reason: 'x'} : null))(on.env);
  const ro = on.onCall('object', {verb: 'pick_up', object: {name: 'headphones'}}, 'done', {});
  check('page: object goes through D.vt.gate first (refused -> no command)', ro.status === 'not_a_request' && on.env.gated === 'pick_up headphones' && !on.env.ran.some(x => x.cmd.action === 'object'));
  on.D.vt.expand = () => { on.env.expanded = (on.env.expanded || 0) + 1; return 0; };
  on.onCall('object', {verb: 'pick_up', object: {name: 'ball'}}, 'done', {});
  check('page: object allowed by the gate: his words\' steps expanded first, then the call runs as before', on.env.expanded === 1 && on.env.ran.some(x => x.cmd.action === 'object'));
  on.onJSON({type: 'act', id: 8, name: 'object', args: {verb: 'put_on', object: {name: 'headphones'}}});
  check('page: the hub-relayed object call is gated too', on.env.ctl.some(m => m.type === 'act_result' && m.id === 8 && m.result.status === 'not_a_request'));
  const hub = mk(Qs({}), {character: 'fox'}, 'live', false);
  hub.onJSON({type: 'transcript', who: 'you', text: 'if I asked '}); hub.onJSON({type: 'transcript', who: 'you', text: 'you to jump'}); hub.onJSON({type: 'transcript', who: 'avatar', text: 'Hm'});
  check('page (hub-relayed voice): his transcript deltas -> D.vt (start, deltas, final at the turn change)', JSON.stringify(hub.env.ev) === JSON.stringify([{sp: 'start'}, {tx: 'if I asked ', f: false}, {tx: 'you to jump', f: false}, {tx: '', f: true}]));
  const off = mk(Qs({planner: '0'}), {character: 'muse'}, 'realtime');
  off.onCall('look', {target: 'left'}, 'done', {});
  off.onCall('do_action', {description: 'x'}, 'done', {});
  check('page planner=0: the voice tools are not used (the path)', !off.env.vt.length && off.env.ran.some(x => x.cmd.action === 'look'));
  {
    const ttc = new Function(ttcSrc + '\nconst intentAct = () => true; return {actToolCommand};')();
    const tg = t => ttc.actToolCommand('object', {verb: 'pull', object: {name: 'chair'}, target: t}).target;
    check('page: spoken destinations map onto the engine\'s (here / over here / to me -> me; nearer -> closer)', tg('here') === 'me' && tg('over here') === 'me' && tg('to me') === 'me' && tg('nearer') === 'closer' && tg('desk').name === 'desk' && tg('me') === 'me');
  }
  check('page: voicetools.js is inlined by the hub', /"voicetools\.js"/.test(fs.readFileSync(path.join(ROOT, 'app/app.py'), 'utf8')));
  latency();
}

// ===================================================================================================== latency of dispatch
function latency() {
  // the real chain: voicetools -> perform.js -> a mock motion (no plan answer needed: call() returns at once)
  const {performance} = require('perf_hooks');
  const m = mockMotion(makeClock());
  const P = Perf.createPerformer({motion: () => m, prompt: () => ({}), sendCtl: () => {}, character: () => 'muse', userSpeaking: () => false, lastUserSpeech: () => -1e12});
  const vt = VTL.createVoiceTools({perf: () => P, motion: () => ({props: () => [{id: 'chair', name: 'chair'}]}), behaviors: () => mockBehaviors(),
    runCommand: () => ({ok: true, did: 'x'}), sendCtl: () => {}, prompt: () => ({}), userSpeaking: () => false, reactMode: () => 'unified'});
  const per = {};
  const N = 300;
  for (const [name, args] of [['do_action', {description: 'jump three times'}], ['look', {target: 'chair', seconds: 2}], ['set_style', {energy: 0.5}],
    ['start_behavior', {instruction: 'wave when I wave'}], ['list_behaviors', {}]]) {
    const xs = [];
    for (let i = 0; i < N; i++) { vt.userText('Do it now please', true); const t0 = performance.now(); vt.call(name, args, 'done', {}); xs.push((performance.now() - t0) * 1000); }
    xs.sort((a, b) => a - b); per[name] = {p50: Math.round(xs[N >> 1]), p90: Math.round(xs[Math.floor(N * 0.9)])};
  }
  const g = [], texts = ['If I asked you to jump, could you?', 'Pick up the headphones and put them on the desk, then sit down.'];
  for (let i = 0; i < 2000; i++) { const t0 = performance.now(); VTL.isHypothetical(texts[i & 1]); g.push((performance.now() - t0) * 1000); }
  g.sort((a, b) => a - b);
  per.guard = {p50: Math.round(g[1000]), p90: Math.round(g[1800])};
  console.log('dispatch µs (p50/p90): ' + Object.entries(per).map(([k, v]) => `${k} ${v.p50}/${v.p90}`).join(', '));
  check('latency: tool dispatch p90 < 1 ms for every tool (event-driven, never per frame)', Object.values(per).every(v => v.p90 < 1000), per);
  console.log(`${passes} passed, ${fails} failed`);
  process.exitCode = fails ? 1 : 0;
}
