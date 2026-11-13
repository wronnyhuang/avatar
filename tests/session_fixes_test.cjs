// session fixes: the failures a session judge found in real sessions, as tests of the
// MECHANISMS with phrasings that are NOT the session's own.        node tests/session_fixes_test.cjs
//   1. the "no things here" backstop never refuses a body / face / eyelid / head / expression request, never asks for a word
//   2. a stale fast-matcher command never "covers" a different request; "more" never re-applies a superseded pose
//   3. every hold has a working release: a planned move takes over the held parts it drives; release / relax frees the rest
//   4. tool results are truthful (ran / not), and a spoken claim of a move with no action in the turn is corrected
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const ROOT = path.join(__dirname, '..');
const VTL = load('app/voicetools.js');
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

// ===================================================================================================== 1. the backstop
{
  const seated = () => { const clock = makeClock(), X = {perf: [], prompts: []};
    X.vt = VTL.createVoiceTools({perf: () => ({call: (d, info) => { X.perf.push(d); return {ok: true, status: 'preparing', id: 'p' + X.perf.length}; }}),
      motion: () => ({props: () => []}), prompt: (t, o) => { X.prompts.push({t, o}); return {}; }, now: clock.now, timers: {set: clock.set, clear: clock.clear}});
    return X; };
  const run = d => { const X = seated(); const r = X.vt.call('do_action', {description: d}, 'done', {}); return {r, planned: X.perf.length === 1}; };
  // body, face, eyelid, head, expression and acted-out requests: always planned, whatever words follow the body part
  const BODY = [
    'Slap him a high ten with both paws, then one more.',
    'Give him a big thumbs up with my left paw now.',
    'Give me a little wave and a huge grin.',
    'Raise my chin up as far as it goes, slowly.',
    'Lift my left arm up over my head and keep it there for a moment.',
    'Lift up my ears and perk them forward.',
    'Shut my eyes tight right now and keep them shut for three seconds.',
    'Close my mouth now, then open it wide in a yawn.',
    'Open my eyes wide in surprise, then relax.',
    'Open my mouth in a big yawn and stretch.',
    'Hold a calm drowsy look for a few seconds, then wake up.',
    'Hold that pose for two seconds and then shake it off.',
    'Hold my breath briefly and puff my cheeks.',
    'Hold up two fingers in a peace sign.',
    'Fold my arms together and rest my cheek on them like dozing off, then return to normal.',
    'Put my paws together under my chin and look dreamy in a gentle way.',
    'Put my hands on my hips and tap my foot.',
    'Put on a brave face and puff out my chest.',
    'Take a deep bow, then take a little step to the left.',
    'Take a deep breath in, hold it, and let it out slowly.',
    'Throw a kiss at him with my right paw.',
    'Throw my arms up in the air and cheer.',
    'Catch my breath after all that, panting a little.',
    'Bring my paws up to my cheeks in delight.',
    'Bring the energy up with a bouncy little groove.',
    'Wear a huge smile and wiggle.',
    'Pick up the pace and nod faster.',
    'Grab my tail and give it a tug.',
    'Play with my ears, flopping them back and forth.',
    'Show him a silly face with my tongue out.',
    'Use my right paw to scratch behind my ear.',
    'Tilt my head to the right and drift off, eyes closing, then perk back up.',
    'Go back to a cheerful face and let my eyes open again, in an easy way.',
    'Slide to my left a little and lean in.',
    'Give him five, up top, with the left paw.',
    'Sit up tall and straighten my back.',
  ];
  const badB = BODY.map(d => [d, run(d)]).filter(([d, x]) => !(x.r.ok && x.planned));
  check(`backstop: ${BODY.length} body / face / eyelid / head / expression / acted-out requests are never refused`, !badB.length, badB.map(([d, x]) => d + ' -> ' + (x.r.reason || '').slice(0, 60)));
  // a real thing on an object verb: still an honest refusal on a character with nothing around it (FOX muse-pickup declines)
  const THINGS = [
    ['Pick up the headphones from the desk and put them on.', 'headphones'],
    ['Grab the red ball and toss it to him.', 'red ball'],
    ['Put on the party hat now.', 'party hat'],
    ['Pick the umbrella up and open it.', 'umbrella'],
    ['Take off the scarf and fold it.', 'scarf'],
    ['Fetch his slippers and bring them over.', 'slippers'],
    ['Hand him the remote control.', 'remote control'],
    ['Hold the coffee mug in my left paw.', 'coffee mug'],
    ['Open the window behind me.', 'window'],
    ['Lift the box over my head.', 'box'],
    ['Give him the book now.', 'book'],
    ['Drink a glass of water slowly.', 'glass'],
    ['Wear the sunglasses and look cool.', 'sunglasses'],
    ['Sit on the chair next to me.', 'chair'],
    ['Carry the laptop to him.', 'laptop'],
    ['Put the cup down gently.', 'cup'],
  ];
  const badT = THINGS.map(([d, n]) => [d, n, run(d)]).filter(([d, n, x]) => !(x.r.ok === false && x.r.status === 'no_such_thing' && !x.planned && x.r.reason.includes('no ' + n + ' ')));
  check(`backstop: ${THINGS.length} requests to handle a real thing are declined honestly, naming the thing`, !badT.length, badT.map(([d, n, x]) => d + ' -> ' + JSON.stringify(x.r).slice(0, 90)));
  const one = run('Pick up the guitar and strum it.').r;
  check('backstop: the refusal says nothing moved and never asks him for a special word', one.ran === false && /nothing moved/.test(one.reason) && !/pretend|say the word|magic/i.test(one.reason), one.reason);
  check('backstop: acting it out is planned when the request says so (no word is required for body moves)', run('Mime picking up a heavy suitcase and carrying it.').planned && run('Act out drinking a cup of tea.').planned);
  // a character WITH things around it never reaches this check (the scene route decides)
  const clock = makeClock(), perf = [];
  const vt = VTL.createVoiceTools({perf: () => ({call: d => { perf.push(d); return {ok: true, status: 'preparing', id: 'p1'}; }}), motion: () => ({props: () => [{id: 'chair', name: 'chair'}]}), now: clock.now, timers: {set: clock.set, clear: clock.clear}});
  check('backstop: only for a character with nothing around it', vt.call('do_action', {description: 'Pick up the trumpet.'}, 'done', {}).ok === true && perf.length === 1);
}

// ============================================================ 2. a stale matcher command never covers a different request
{
  const {Intent, IntentDedupe} = require(path.join(ROOT, 'app', 'intent.js'));
  const RULES = JSON.parse(fs.readFileSync(path.join(ROOT, 'app', 'intent_rules.json'), 'utf8'));
  // (a) dedupe: the model's pose that drives the same part the opposite way is never dropped as the matcher's repeat
  IntentDedupe.reset();
  const up = {action: 'pose', name: 'a', targets: {head_yaw: -0.5}, parts: ['head'], hold: true}, down = {action: 'pose', name: 'b', targets: {head_yaw: 0.6}, parts: ['head'], hold: true};
  IntentDedupe.seen(up, 1000, 'matcher');
  const opp = IntentDedupe.seen(down, 2500, 'model');
  check('dedupe: an opposite pose on the same part from the model runs (and replaces the matcher record)', opp === false && IntentDedupe.last.conflict && IntentDedupe.recs.length === 1 && IntentDedupe.recs[0].source === 'model');
  IntentDedupe.reset(); IntentDedupe.seen(up, 1000, 'matcher');
  check('dedupe: the same-direction pose from the model is still one request', IntentDedupe.seen({action: 'pose', name: 'a2', targets: {head_yaw: -0.7}, parts: ['head'], hold: true}, 2500, 'model') === true);
  IntentDedupe.reset();
  // (b) "already started as he said it" needs the matcher's words to be in his newest utterance
  const mk = covered => { const clock = makeClock(), X = {perf: []};
    X.vt = VTL.createVoiceTools({perf: () => ({call: d => { X.perf.push(d); return {ok: true, status: 'preparing', id: 'p'}; }}), motion: () => ({props: () => []}),
      coveredByMatcher: () => covered, prompt: () => ({}), now: clock.now, timers: {set: clock.set, clear: clock.clear}});
    X.say = t => { X.vt.userSpeech('start'); X.vt.userSpeech('stop'); X.vt.userText(t, true); };
    return X; };
  let X = mk(['stick your tongue out']); X.say('Now puff your cheeks instead.');
  let r = X.vt.call('do_action', {description: 'Puff my cheeks out.'}, 'done', {});
  check('covered: a matcher command read from EARLIER words does not cover a new request (it is planned)', X.perf.length === 1 && !r.already, r);
  X = mk(['stick your tongue out']); X.say('Okay, stick your tongue out please.');
  r = X.vt.call('do_action', {description: 'Stick my tongue out.'}, 'done', {});
  check('covered: a matcher command read from THIS utterance covers the restated call, and the result says so', X.perf.length === 0 && r.ok === true && r.already === true, r);
  X = mk([null]); X.say('Wiggle your ears.');
  r = X.vt.call('do_action', {description: 'Wiggle my ears.'}, 'done', {});
  check('covered: a release never covers a request', X.perf.length === 1, r);
  X = mk(true); X.say('Wiggle your ears.');
  check('covered: a page that only answers yes / no behaves as before', X.vt.call('do_action', {description: 'Wiggle my ears.'}, 'done', {}).already === true && X.perf.length === 0);
  // (c) the matcher's "more" context: gone once it is superseded or released
  const M = Intent.create(RULES, {});
  const first = M.parse('raise your left paw', {final: true, nowMs: 1000});
  for (const h of first) M.commit(h, 1000);
  const more1 = M.parse('higher', {final: true, nowMs: 3000});
  M.forgetContext();
  const more2 = M.parse('higher', {final: true, nowMs: 3500});
  check('matcher: an adjustment word re-applies the last pose only while that pose is still the context', first.length > 0 && more1.length > 0 && more2.length === 0, {first: first.length, more1: more1.length, more2: more2.length});
}

// ================================================================================== 3. every hold has a working release
{
  const MO = load('app/motion.js'), DT = 1 / 30;
  const mk = name => { const m = MO.createMotion({seed: 7, character: name, onAction: () => {}}); m.setProps(null, null, name); let t = 0;
    const step = () => { t += DT; return Object.assign({}, m.step(DT, {features: {t, face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}})); };
    const run = s => { let o; for (let i = 0; i < Math.round(s * 30); i++) o = step(); return o; };
    return {m, run}; };
  for (const ch of ['fox', 'muse']) {
    // (a) a held head pose, then a planned move that drives the head the other way: it plays where the plan put it, and
    //     the head ends free at rest — not back in the held pose
    let C = mk(ch); C.run(1);
    const live = C.run(0.1).head_pitch;
    C.m.command({action: 'pose', name: 'x', targets: {head_pitch: -0.5}, parts: ['head'], hold: true, ease_ms: 300}, {src: 'matcher'});
    const held = C.run(2).head_pitch;
    C.m.command({action: 'keyframes', id: 'k1', description: 'the other way', poses: [{t: 0.5, head_pitch: 0.4}, {t: 1.6, head_pitch: 0.4}, {t: 2.3, head_pitch: 0}], done: true}, {src: 'voice'});
    const mid = C.run(1.3).head_pitch, st1 = C.m.state();
    const end = C.run(4).head_pitch, st2 = C.m.state();
    check(`hold (${ch}): a planned move takes over a held head — plays the plan's direction and ends at rest, hold gone`,
      held < live - 0.3 && mid > live + 0.2 && Math.abs(end - live) < 0.15 && !st2.held.includes('head'), {live, held, mid, end, held1: st1.held, held2: st2.held});
    // (b) held shut eyes, then a planned move that opens them
    C = mk(ch); C.run(1);
    C.m.command({action: 'pose', name: 'y', targets: {blink_l: 1, blink_r: 1}, parts: ['eyes'], hold: true}, {src: 'matcher'});
    const shut = C.run(2).blink_l;
    C.m.command({action: 'keyframes', id: 'k2', description: 'eyes open', poses: [{t: 0.4, eyes_closed: 0}, {t: 1.2, eyes_closed: 0}], done: true}, {src: 'voice'});
    C.run(3); let mx = 0; for (let i = 0; i < 20; i++) mx = Math.max(mx, C.run(0.1).blink_l < 0.5 ? 0 : 1);
    const st3 = C.m.state();
    check(`hold (${ch}): held shut eyes open with a planned move and stay open (blinks aside)`, shut > 0.8 && !st3.held.includes('eyes') && C.run(0.5).blink_l < 0.5 || mx === 0, {shut, held: st3.held});
    // (c) a part the plan does not touch stays held; release('all') frees it
    C = mk(ch); C.run(1);
    C.m.command({action: 'pose', name: 'x', targets: {head_pitch: -0.5}, parts: ['head'], hold: true}, {src: 'matcher'}); C.run(1.5);
    C.m.command({action: 'keyframes', id: 'k3', description: 'smile', poses: [{t: 0.4, smile: 0.8}, {t: 1.2, smile: 0}], done: true}, {src: 'voice'}); C.run(3);
    const keep = C.m.state().held.includes('head');
    C.m.command({action: 'release', parts: 'all'}, {src: 'voice'}); C.run(1.5);
    check(`hold (${ch}): a part the plan does not drive stays held, and release frees everything`, keep && C.m.state().held.length === 0, C.m.state().held);
    // (c2) a commanded gesture / expression on a held part: the hold lets go, the gesture shows
    C = mk(ch); C.run(1);
    C.m.command({action: 'pose', name: 'x', targets: {head_pitch: -0.5}, parts: ['head'], hold: true}, {src: 'matcher'}); C.run(1.5);
    const gOk = C.m.command({action: 'gesture', name: 'nod'}, {src: 'voice'});
    let lo = 9, hi = -9; for (let i = 0; i < 45; i++) { const v = C.run(1 / 30).head_pitch; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    C.run(3);
    check(`hold (${ch}): a commanded nod on a held head is visible and leaves the head free`, !!gOk && hi - lo > 0.08 && hi > -0.2 && !C.m.state().held.includes('head'), {gOk, lo, hi, held: C.m.state().held});
  }
  // (d) voice tools: "back to normal" in any wording is a release of everything, never a plan over the held pose
  const mkv = heldList => { const clock = makeClock(), X = {perf: [], cmds: [], rel: [], stops: 0};
    X.vt = VTL.createVoiceTools({perf: () => ({call: d => { X.perf.push(d); return {ok: true, status: 'preparing', id: 'p'}; }, stop: () => { X.stops++; }}),
      motion: () => ({props: () => [], state: () => ({held: heldList, active: []}), release: (parts, ms) => { X.rel.push(parts); }}),
      runCommand: c => { X.cmds.push(c); return {ok: true}; }, prompt: () => ({}), now: clock.now, timers: {set: clock.set, clear: clock.clear}});
    return X; };
  const REL = ['Go back to my normal standing pose now.', 'Relax completely.', 'Return to a neutral position and rest.', 'Stop posing.', 'Drop the pose and go back to normal.', 'Reset my pose.', 'Just be normal again.'];
  const relOk = REL.filter(d => { const X = mkv(['head', 'paw_l']); const r = X.vt.call('do_action', {description: d}, 'done', {}); return r.ok === true && X.perf.length === 0 && X.cmds.some(c => c.action === 'release' && c.parts === 'all') && X.stops === 1; });
  check('release: a description that only asks to go back to normal / relax releases every hold (no plan)', relOk.length === REL.length, REL.filter(d => !relOk.includes(d)));
  const NOTREL = ['Go back to normal and then wave with both paws.', 'Relax my shoulders and tilt my head to the left.', 'Stop and do a little spin.', 'Rest my chin on my paw.'];
  const notOk = NOTREL.filter(d => { const X = mkv(['head']); X.vt.call('do_action', {description: d}, 'done', {}); return X.perf.length === 1; });
  check('release: a request that asks for something more is still planned', notOk.length === NOTREL.length, NOTREL.filter(d => !notOk.includes(d)));
  // (e) voice tools: the newest request lets go of a hold on the parts it names, before the plan is even written
  const cases = [['Open my eyes wide and look around.', ['eyes'], ['eyes']], ['Bring my chin down to my chest.', ['head', 'paw_l'], ['head']], ['Put both paws down by my sides.', ['paw_l', 'paw_r', 'head'], ['paw_l', 'paw_r']],
    ['Wag my tail quickly.', ['head'], null]];
  const bad = cases.filter(([d, heldList, want]) => { const X = mkv(heldList); X.vt.call('do_action', {description: d}, 'done', {});
    const got = X.rel.length ? [...X.rel[0]].sort().join() : null; return !(X.perf.length === 1 && got === (want ? want.slice().sort().join() : null)); });
  check('supersede: a new request frees held parts it names (and only those)', bad.length === 0, bad);
}

// ================================================================================ 4. narration without motion is caught
{
  const YES = ['Sure thing, raising both paws now.', 'Okay, I am leaning to the left for you.', 'Here goes a big stretch!', 'Alright, spinning around slowly.', 'And that is it. How did that look?',
    'All done! Want another?', 'Tilting my head back now, nice and easy.', 'Closing my eyes now.', 'Okay, changing to a grumpy face now.', 'Getting into position now.', 'Watch this.'];
  const NO = ['Want me to spin around for you?', 'I can wave if you like.', 'That sounds fun, tell me more about your day.', 'I could not do that one, sorry.', 'Going to the store sounds like a plan.',
    'I love dancing, it is so much fun.', 'Should I lower my head now?', 'Nothing moved just now, sorry about that.', 'Here is what I think about it.', 'You are doing great.', 'I am thinking about it now.',
    'How was your day?', 'Right now I am just listening.', 'Going back to what you said, I agree.', 'You are raising a good point now.', 'Moving on, what else is on your mind?', 'Coming back to your question about work.'];
  check('claim: sentences that speak of its own move as starting / under way / done are recognised', YES.every(t => VTL.claimsMove(t)), YES.filter(t => !VTL.claimsMove(t)));
  check('claim: offers, questions, refusals and ordinary talk are not claims', NO.every(t => !VTL.claimsMove(t)), NO.filter(t => VTL.claimsMove(t)));
  const mk = o => { const clock = makeClock(), X = {prompts: [], perf: [], lastAction: -1e12, busy: false, speaking: false};
    X.clock = clock;
    X.vt = VTL.createVoiceTools(Object.assign({perf: () => ({call: d => { X.perf.push(d); return {ok: true, status: 'preparing', id: 'p'}; }, busy: () => X.busy, active: () => (X.busy ? 1 : 0)}),
      motion: () => ({props: () => [], state: () => ({held: [], active: []})}), prompt: (t, opt) => { X.prompts.push({t, opt}); return {}; }, lastActionAt: () => X.lastAction,
      avatarSpeaking: () => X.speaking, userSpeaking: () => false, now: clock.now, timers: {set: clock.set, clear: clock.clear}}, o || {}));
    X.say = t => { X.vt.userSpeech('start'); clock.advance(900); X.vt.userSpeech('stop'); X.vt.userText(t, true); clock.advance(600); };
    X.claims = () => X.prompts.filter(p => /Fact check/.test(p.t));
    return X; };
  // no call at all in the turn + a claim -> one note, with a chance to act, through the unprompted gate
  let X = mk(); X.say('Go on then, do the shimmy.');
  X.vt.avatarText('Okay! ', false); X.vt.avatarText('Shimmying side to side now. ', false); X.vt.avatarText('Okay! Shimmying side to side now. How was that?', true);
  X.clock.advance(1000); const early = X.claims().length; X.clock.advance(3000);
  check('claim: words of a move with no action in the turn -> one corrective note after a short wait', early === 0 && X.claims().length === 1 && X.claims()[0].opt.respond === true && X.claims()[0].opt.unprompted === true, X.prompts);
  X.vt.avatarText('All done, how did that look?', true); X.clock.advance(4000);
  check('claim: at most one note per utterance of his', X.claims().length === 1);
  // the call arrives right after the words (same response) -> no note
  X = mk(); X.say('Give me a big stretch.');
  X.vt.avatarText('Here goes a big stretch!', true); X.clock.advance(800);
  X.vt.call('do_action', {description: 'Do a big stretch with both arms up.'}, 'done', {}); X.clock.advance(4000);
  check('claim: a call that follows the words within the wait is not flagged', X.claims().length === 0 && X.perf.length === 1);
  // a matcher command ran for his words -> no note
  X = mk(); X.say('Stick your tongue out.'); X.lastAction = X.clock.now() - 300;
  X.vt.avatarText('Sticking my tongue out now.', true); X.clock.advance(4000);
  check('claim: no note when a fast matcher already acted in this turn', X.claims().length === 0);
  // a move is playing (its notes ask for a cue) -> no note
  X = mk(); X.say('Keep going.'); X.busy = true;
  X.vt.avatarText('Here it comes, nice and steady.', true); X.clock.advance(4000);
  check('claim: no note while a planned move is being prepared or playing', X.claims().length === 0);
  // ordinary talk -> nothing
  X = mk(); X.say('What do you think about the rain?');
  X.vt.avatarText('I like it, it makes everything smell fresh. How was your walk?', true); X.clock.advance(4000);
  check('claim: ordinary talk is never flagged', X.claims().length === 0);
  // still talking when the wait ends -> checked after it stops; he starts speaking -> dropped
  X = mk({userSpeaking: () => X.he}); X.say('Do a twirl.'); X.he = false;
  X.vt.avatarText('Twirling around now!', true); X.he = true; X.clock.advance(4000);
  check('claim: never interjects while he is speaking', X.claims().length === 0);
  // truthful results: covered -> already:true (2b), release -> held:[], backstop -> ran:false (1)
  const src = fs.readFileSync(path.join(ROOT, 'app', 'rtc.py'), 'utf8');
  check('prompt: the timing rules forbid speaking of a move without its call in the same turn', /unless you made its call in this same turn/.test(src));
}

console.log(`${passes} passed, ${fails} failed`);
process.exitCode = fails ? 1 : 0;
