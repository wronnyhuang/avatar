#!/usr/bin/env node
// speech — action-memory probes against the REAL voice model (gpt-realtime-2.1 over WebSocket, TEXT mode =
// the same model/persona/tools as the phone, cheapest modality). The page's logic runs in the loop: app/actionlog.js
// + a contract-shaped motion stand-in (tests/speech_fake_motion.cjs) in virtual time, the fast matcher's commands,
// the model's own tool calls (deduped like page.html runCommand), notes sent the way page.html sends them.
//
//   node tools/actionmem/rt_probe.cjs --mech none|notes|tool|both [--only S1,S5] [--out dir] [--replace 1]
// mech: none = the baseline (tool result {ok:true}, no log); notes = silent log notes + {ok,did,reason,now} results;
//       tool = recall_actions tool + rich results; both.
// Writes <out>/<mech>.json (every probe: answer, verdict, invented-action flags) and prints accuracy + estimated $.
// Env: OPENAI_API_KEY (required), PYTHON (interpreter for app/rtc.py, default python3), AVATAR_REALTIME_MODEL.
// --out default: <repo>/runs/actionmem
'use strict';
const fs = require('fs'), path = require('path'), {execFileSync} = require('child_process');
const ROOT = path.resolve(__dirname, '..', '..');
const load = require(path.join(ROOT, 'tests', 'load.cjs'));
const AL = load('app/actionlog.js');
const {createFakeMotion} = require(path.join(ROOT, 'tests', 'speech_fake_motion.cjs'));
// --engine real: the merged motion.js + moves.js with the real muse JSON (tests/muse_lib.cjs); default = the stand-in
const REAL = process.argv.includes('real');
const Muse = REAL ? require(path.join(ROOT, 'tests', 'muse_lib.cjs')) : null;
const RNAME = {cup: 'mug', flower: 'plant'};
function realCmd(c) {
  c = JSON.parse(JSON.stringify(c));
  if (c.object) { const o = c.object; if (o.name && RNAME[o.name]) o.name = RNAME[o.name]; if (o.color !== 'green') { delete o.color; delete o.category; } if (!o.name && o.category === 'toy') o.name = 'ball'; }
  if (c.action === 'release' && c._stop) return {action: 'stop'};
  if (c.action === 'turn_back') return {action: 'move', name: 'turn_back'};
  if (c.action === 'stand_up') return {action: 'move', name: 'stand_up'};
  return c;
}
function realWorld(onEv) {
  const X = Muse.makeMuse(Math.floor(Math.random() * 1000) + 1); let seen = 0;
  const sync = () => { for (; seen < X.ev.length; seen++) onEv(X.ev[seen]); };
  const st = () => { const s = X.m.state(); return Object.assign({}, s, {active: (s.active || []).map(a => ({name: a.name, kind: a.kind}))}); };
  const busy = () => { const E = X.m.engine.E; return !!(E.move || E.chain || E.jobs.length || E.queue || E.oqueue); };
  return {X, real: true, busy, command: (c, src) => { const r = X.m.command(realCmd(c), {src}); sync(); return r; },
    step: () => { X.step(1); sync(); }, state: st,
    reflex: name => { for (let k = 0; k < 6; k++) { const n0 = X.ev.length; X.m.event({type: name === 'nod_back' ? 'nod' : 'wave', side: 'l', x: 0.1, y: 0.2}); X.step(20); sync(); if (X.ev.slice(n0).some(e => e.kind === 'reflex')) return; X.step(130); sync(); } }};
}
const RATES = {text_in: 4.0, audio_in: 32.0, cached: 0.5, text_out: 24.0, audio_out: 64.0};

const args = {mech: 'notes', out: path.join(ROOT, 'runs', 'actionmem'), only: '', replace: '0', model: process.env.AVATAR_REALTIME_MODEL || 'gpt-realtime-2.1', rep: '1'};
for (let i = 2; i < process.argv.length; i++) { const m = /^--([a-z]+)$/.exec(process.argv[i]); if (m) args[m[1]] = process.argv[++i]; }
fs.mkdirSync(args.out, {recursive: true});

// ---- the session config from app/rtc.py (persona + tools exactly as the phone gets them)
const PY = process.env.PYTHON || 'python3';
const sess = JSON.parse(execFileSync(PY, ['-c', `import rtc, json; print(json.dumps(rtc.realtime_session('Jolly','plush muse','verse','server:300','','parallel', ${JSON.stringify(args.mech === 'none' ? 'off' : args.mech)}, 'muse')))`], {cwd: path.join(ROOT, 'app')}).toString());
// provisional move / object tools (the intent agent owns the real ones; contract §3 shapes)
const MOVES = ['spin', 'twirl', 'turn', 'turn_away', 'turn_back', 'backflip', 'flip', 'jump', 'hop', 'walk', 'dance', 'sit', 'stand', 'lie_down', 'roll_over', 'cartwheel', 'bow', 'wiggle'];
const VERBS = ['pick_up', 'put_down', 'put_back', 'place', 'hold_up', 'show', 'inspect', 'turn', 'flip', 'hand_over', 'toss', 'drop', 'shake', 'hug', 'offer',
  'put_on', 'take_off', 'pull', 'push', 'slide', 'bring', 'sit_at', 'sit_in', 'stand_up', 'open', 'close', 'type_on', 'work_on', 'turn_back'];
const silent = ' It happens silently on screen: never describe or announce it out loud.';
sess.tools.push(
  {type: 'function', name: 'move', description: 'Move your whole body: spin, turn away/back, backflip, jump, dance, sit, stand, lie down, walk, bow...' + silent,
    parameters: {type: 'object', properties: {name: {type: 'string', enum: MOVES}, count: {type: 'integer', minimum: 1, maximum: 10},
      dir: {type: 'string', enum: ['left', 'right', 'forward', 'back']}, style: {type: 'string'}, hold: {type: 'boolean'}}, required: ['name']}},
  {type: 'function', name: 'object', description: 'Do something with one of the things near you (pick up, put down, put back, show, toss, pass to the other paw, drop, inspect...).' + silent,
    parameters: {type: 'object', properties: {verb: {type: 'string', enum: VERBS}, object: {type: 'object', properties: {name: {type: 'string'}, color: {type: 'string'},
      category: {type: 'string'}, ref: {type: 'string', enum: ['it', 'that', 'other']}}}, side: {type: 'string', enum: ['left', 'right', 'both']},
      count: {type: 'integer'}, place: {type: 'string'}, target: {type: 'string', description: "where to: 'closer', 'back', 'away', 'me', or a thing/place"}}, required: ['verb']}});
// the stand-in's world, as config.props would tell the model (a scene line in the persona — the props agent's job)
if (REAL) { const X0 = Muse.makeMuse(1); const ns = X0.m.state().props.map(p => p.name); sess.instructions += ' Around you: ' + ns.slice(0, -1).map(n => 'a ' + n).join(', ') + ' and a ' + ns[ns.length - 1] + '.'; }
else sess.instructions += ' Around you: headphones on a side table, a chair and a desk by the wall, a laptop (closed) and a green mug on the desk, and a red ball, a blue cup, a yellow book and a pink flower on a little table.';

// ---- scenarios: user turns (with the fast matcher's command when it would fire), waits, reflexes, probes
const M = (action, o) => Object.assign({action}, o);
const NEG = '\\b(no|nope|not|didn\'?t|did not|haven\'?t|have not|never|wasn\'?t|couldn\'?t|could not|nothing)\\b';
const YESSTART = '^\\W*(yes|yeah|yep|yup|sure did|i did)\\b';
const SC = [
  {id: 'S1', cat: 'empty', steps: [
    {probe: "Hey Felix! What have you done so far, on the screen I mean? Any moves yet?", cat: 'empty', expect: () => ({none: true})},
    {probe: 'Did you wave at me a minute ago?', cat: 'empty/premise', expect: () => ({must: [NEG], not: [YESSTART]})},
    {probe: 'What are you holding right now?', cat: 'empty/holding', expect: () => ({must: ['(nothing|empty|not holding|no(t)? .{0,20}hold|don\'?t have|aren\'?t holding)'], not: ['\\b(ball|cup|book|flower)\\b(?! (on|is on|are on|sitting))']})}]},
  {id: 'S2', cat: 'completed/order', steps: [
    {user: 'Wave at me!', matcher: M('gesture', {name: 'wave', side: 'right'})}, {wait: 3},
    {probe: 'What did you just do?', cat: 'completed', expect: () => ({must: ['wav']})},
    {user: 'Now clap your paws.', matcher: M('gesture', {name: 'clap'})}, {wait: 2.5},
    {user: 'And nod.', matcher: M('gesture', {name: 'nod'})}, {wait: 2},
    {probe: 'What did you do right before you nodded?', cat: 'order', expect: () => ({must: ['clap']})},
    {probe: 'What was the very first thing you did?', cat: 'order', expect: () => ({must: ['wav'], not: ['\\bfirst\\b.{0,20}\\bclap']})}]},
  {id: 'S3', cat: 'counts', steps: [
    {user: 'Jump!', matcher: M('gesture', {name: 'jump'})}, {wait: 2},
    {user: 'Jump again!', matcher: M('gesture', {name: 'jump'})}, {wait: 2},
    {user: 'One more jump!', matcher: M('gesture', {name: 'jump'})}, {wait: 2},
    {probe: 'How many times did you jump?', cat: 'count', expect: c => ({must: ['\\b(' + c.countWord('jump') + ')\\b']})},
    {probe: 'Did you jump four times?', cat: 'count/premise', expect: c => ({must: [NEG + '|\\b(' + c.countWord('jump') + ')\\b'], not: [YESSTART]})}]},
  {id: 'S4', cat: 'failed', steps: [
    {user: 'Pick up the green ball.', matcher: M('object', {verb: 'pick_up', object: {color: 'green', category: 'ball'}})}, {wait: 2},
    {probe: 'Did you get the green ball?', cat: 'failed', expect: () => ({must: [NEG + '|there\'?s no|isn\'?t (a|any)'], not: [YESSTART, 'got (it|the green)']})},
    {probe: "So what's in your paws now?", cat: 'failed/holding', expect: () => ({must: ['(nothing|empty|not holding|no(t)? .{0,20}hold)']})}]},
  {id: 'S5', cat: 'interrupted', steps: [
    {user: 'Spin around three times!', matcher: M('move', {name: 'spin', count: 3})},
    {user: 'Stop! Stop spinning.', matcher: M('release', {parts: 'all', _stop: true})}, {wait: 1},
    {probe: 'Did you finish all three spins?', cat: 'interrupted', expect: c => c.progressN('spin') >= 3 ? ({must: ['\\b(yes|yeah|yep|did|all three)\\b']}) : ({must: [NEG + '|stopped|only'], not: [YESSTART]})},
    {probe: 'How many spins did you actually do?', cat: 'interrupted/count', expect: c => ({must: [c.progressWord('spin')]})}]},
  {id: 'S5b', cat: 'interrupted', steps: [
    {user: 'Spin around six times!', matcher: M('move', {name: 'spin', count: 6})},
    {user: 'Stop! Stop spinning.', matcher: M('release', {parts: 'all', _stop: true})}, {wait: 1},
    {probe: 'Did you finish all six spins?', cat: 'interrupted', expect: c => c.progressN('spin') >= 6 ? ({must: ['\\b(yes|yeah|did)\\b']}) : ({must: [NEG + '|stopped|only'], not: [YESSTART]})},
    {probe: 'How many spins did you actually do?', cat: 'interrupted/count', expect: c => ({must: [c.progressWord('spin')]})}]},
  {id: 'S6', cat: 'objects', steps: [
    {user: 'Pick up the red ball.', matcher: M('object', {verb: 'pick_up', object: {color: 'red', category: 'toy', name: 'ball'}})}, {wait: 2},
    {user: 'Now pick up the blue cup too.', matcher: M('object', {verb: 'pick_up', object: {color: 'blue', name: 'cup'}})}, {wait: 2},
    {probe: "What's in your left paw?", cat: 'ownership', expect: c => ({must: [c.inPaw('left')]})},
    {probe: 'And which paw has the ball?', cat: 'ownership', expect: c => ({must: [c.pawOf('ball')]})},
    {user: 'Switch the ball to your other paw.', matcher: M('object', {verb: 'hand_over', object: {name: 'ball'}})}, {wait: 2},
    {probe: 'Did you switch it?', cat: 'failed', expect: c => c.didLast('hand_over') ? ({must: ['\\b(yes|yeah|did|switched|passed)\\b']}) : ({must: [NEG + '|full|couldn\'?t'], not: [YESSTART]})}]},
  {id: 'S7', cat: 'premise', steps: [
    {user: 'Give me a thumbs up!', matcher: M('gesture', {name: 'thumbs_up'})}, {wait: 2.5},
    {probe: 'You waved at me earlier, right?', cat: 'premise', expect: () => ({must: [NEG + '|actually|thumbs'], not: [YESSTART]})},
    {probe: 'Thanks for that awesome backflip!', cat: 'premise', expect: () => ({must: [NEG + '|thumbs'], not: ['you(\'re| are) welcome[.!]?\\s*$']})},
    {probe: 'Why did you drop the cup?', cat: 'premise', expect: () => ({must: [NEG + '|haven\'?t']})}]},
  {id: 'S8', cat: 'in_progress', steps: [
    {user: 'Do a happy dance for me!', matcher: M('move', {name: 'dance', style: 'happy'})},
    {probe: 'Are you done dancing yet?', cat: 'in_progress', expect: c => c.active('dance') ? ({must: ['(not yet|still|no\\b|nope|almost)'], not: [YESSTART + '(?!.*still)', '\\b(all done|finished)\\b']}) : ({must: ['\\b(yes|yeah|yep|done|finished|all set)\\b']})},
    {probe: 'What are you doing right now?', cat: 'in_progress', expect: c => c.active('dance') ? ({must: ['danc']}) : ({none: true, allow: ['dance']})}]},
  {id: 'S9', cat: 'negated', steps: [
    {user: "Please don't wave, just tell me what two plus two is."},
    {probe: 'Did you wave just now?', cat: 'negated', expect: c => c.did('wave') ? ({must: ['\\b(yes|yeah|did|waved)\\b']}) : ({must: [NEG], not: [YESSTART]})},
    {user: "Don't pick anything up. What color is the sky?"},
    {probe: 'Are you holding anything?', cat: 'negated/holding', expect: c => c.holdingAny() ? ({must: ['\\b(yes|holding)\\b']}) : ({must: ['(nothing|empty|not holding|\\bno\\b|nope)']})}]},
  {id: 'S10', cat: 'tool_self', steps: [
    {user: 'Can you show me a thumbs up?'}, {wait: 2},
    {probe: 'What did you just do?', cat: 'completed', expect: c => c.did('thumbs_up') ? ({must: ['thumb']}) : ({none: true})}]},
  {id: 'S11', cat: 'relax_object', steps: [
    {user: 'Pick up the yellow book.', matcher: M('object', {verb: 'pick_up', object: {color: 'yellow', name: 'book'}})}, {wait: 2},
    {user: 'Now relax.', matcher: M('release', {parts: 'all'})}, {wait: 1},
    {probe: 'Are you still holding the book?', cat: 'ownership', expect: c => c.holding('book') ? ({must: ['\\b(yes|yeah|yep|still)\\b'], not: ['put it down|dropped it|not holding']}) : ({must: [NEG]})},
    {user: 'Put it back.', matcher: M('object', {verb: 'put_back', object: {ref: 'it'}})}, {wait: 2},
    {probe: "Where's the book now?", cat: 'objects/where', expect: c => c.holding('book') ? ({must: ['paw|holding']}) : ({must: ['(back|table|usual|spot|where it was|put it)']})}]},
  {id: 'S12', cat: 'reflex', steps: [
    {user: 'I had such a good day at work today.'}, {reflex: 'nod_back'}, {wait: 1},
    {probe: 'Did you just nod at me?', cat: 'reflex', expect: () => ({must: ['\\b(yes|yeah|yep|did|nodded|i nod)\\b'], not: ['^\\W*no\\b']})}]},
  {id: 'S13', cat: 'narration', steps: [
    {user: 'Wave hi!', matcher: M('gesture', {name: 'wave'})}, {wait: 2},
    {user: 'Spin once.', matcher: M('move', {name: 'spin', count: 1})}, {wait: 2},
    {probe: 'Tell me a short joke.', cat: 'narration', expect: () => ({none: true, narr: true})},
    {probe: "What's your favorite food?", cat: 'narration', expect: () => ({none: true, narr: true})}]},
  {id: 'S14', cat: 'reset', steps: [
    {user: 'Wave!', matcher: M('gesture', {name: 'wave'})}, {wait: 2},
    {user: 'Jump twice!', matcher: M('gesture', {name: 'jump', repeat: 2})}, {wait: 3},
    {reset: true},
    {probe: 'Hi again! What did you do before?', cat: 'reset', expect: () => ({none: true})},
    {probe: 'Did you jump?', cat: 'reset', expect: () => ({must: [NEG], not: [YESSTART]})}]},
  {id: 'S15', cat: 'redial', steps: [
    {user: 'Clap for me!', matcher: M('gesture', {name: 'clap'})}, {wait: 2},
    {user: 'Now bow.', matcher: M('gesture', {name: 'bow'})}, {wait: 3},
    {redial: true},
    {probe: 'What did you do a moment ago?', cat: 'redial', expect: () => ({must: ['(clap|bow)']})},
    {probe: 'Did you clap?', cat: 'redial', expect: () => ({must: ['\\b(yes|yeah|yep|did|clapped)\\b'], not: ['^\\W*no\\b']})}]},
  {id: 'S16', cat: 'object_chain', steps: [
    {user: 'Pick up the ball.', matcher: M('object', {verb: 'pick_up', object: {name: 'ball'}})}, {wait: 2},
    {user: 'Toss it up twice.', matcher: M('object', {verb: 'toss', object: {ref: 'it'}, count: 2})}, {wait: 3},
    {user: 'Put it down.', matcher: M('object', {verb: 'put_down', object: {ref: 'it'}})}, {wait: 2},
    {probe: 'What did you do with the ball?', cat: 'object_chain', expect: () => ({must: ['(pick|grab)', '(toss|threw|flip)', '(put|set|down|back)']})},
    {probe: 'How many times did you toss it?', cat: 'count', expect: () => ({must: ['\\b(twice|two|2)\\b']})}]},
  {id: 'S17', cat: 'premise', steps: [
    {user: 'Pick up the flower.', matcher: M('object', {verb: 'pick_up', object: {name: 'flower'}})}, {wait: 2},
    {probe: "You're holding the book, aren't you?", cat: 'premise/ownership', expect: () => ({must: [NEG + '|flower|actually'], not: [YESSTART + '(?!.*flower)']})},
    {probe: 'Which paw is it in?', cat: 'ownership', expect: c => ({must: [c.pawOf('flower')]})}]},
  {id: 'S18', cat: 'desk', steps: [
    {user: 'Put on the headphones.', matcher: M('object', {verb: 'put_on', object: {name: 'headphones'}})}, {wait: 2.5},
    {user: 'Pull the chair over.', matcher: M('object', {verb: 'pull', object: {name: 'chair'}})}, {wait: 2.5},
    {user: 'Now pull the desk over too.', matcher: M('object', {verb: 'pull', object: {name: 'desk'}})}, {wait: 2.5},
    {user: 'Sit down at the desk.', matcher: M('object', {verb: 'sit_at', object: {name: 'desk'}})}, {wait: 2},
    {user: 'Slide the laptop closer and open it.', matcher: M('object', {verb: 'slide', object: {name: 'laptop'}, target: 'closer'})}, {wait: 1.5},
    {act: M('object', {verb: 'open', object: {name: 'laptop'}}), src: 'tool'}, {wait: 1.5},
    {user: 'Type something on it.', matcher: M('object', {verb: 'type_on', object: {name: 'laptop'}})}, {wait: 3},
    {probe: 'What are you wearing right now?', cat: 'desk/worn', expect: c => c.worn('headphones') ? ({must: ['headphone']}) : ({must: [NEG]})},
    {probe: 'Are you sitting down?', cat: 'desk/posture', expect: c => c.posture() === 'seated' ? ({must: ['\\b(yes|yeah|yep|sitting|seated|i am)\\b'], not: ['^\\W*no\\b']}) : ({must: [NEG + '|standing']})},
    {user: 'Okay, turn back and talk to me.', matcher: M('turn_back', {})}, {wait: 1},
    {probe: 'What were you just doing on the laptop?', cat: 'desk/activity', expect: c => c.did('type_on') ? ({must: ['typ']}) : ({none: true})},
    {probe: 'Is the laptop open or closed?', cat: 'desk/lid', expect: c => ({must: [c.lid('laptop')]})},
    {probe: 'Tell me everything you did, in order.', cat: 'desk/order', expect: () => ({order: ['headphone', 'chair', 'desk', '(sat|sit)', '(slid|closer)', 'open', 'typ']})}]},
  {id: 'S19', cat: 'desk_reverse', steps: [
    {act: M('object', {verb: 'put_on', object: {name: 'headphones'}})}, {wait: 2.5}, {act: M('object', {verb: 'pull', object: {name: 'chair'}})}, {wait: 2.5},
    {act: M('object', {verb: 'pull', object: {name: 'desk'}})}, {wait: 2.5}, {act: M('object', {verb: 'sit_at', object: {name: 'desk'}})}, {wait: 2},
    {act: M('object', {verb: 'open', object: {name: 'laptop'}})}, {wait: 1.5},
    {user: 'Take off the headphones.', matcher: M('object', {verb: 'take_off', object: {name: 'headphones'}})}, {wait: 2},
    {user: 'Close the laptop.', matcher: M('object', {verb: 'close', object: {name: 'laptop'}})}, {wait: 1.5},
    {user: 'Stand up.', matcher: M('stand_up', {})}, {wait: 1.5},
    {user: 'And push the chair back.', matcher: M('object', {verb: 'push', object: {name: 'chair'}, target: 'back'})}, {wait: 2.5},
    {probe: 'Are you still wearing the headphones?', cat: 'desk/worn', expect: c => c.worn('headphones') ? ({must: ['\\b(yes|still)\\b']}) : ({must: [NEG + '|took (them|it) off'], not: [YESSTART]})},
    {probe: 'Is the laptop still open?', cat: 'desk/lid', expect: c => ({must: [c.lid('laptop') === 'closed' ? NEG + '|closed' : 'open']})},
    {probe: 'Did you push the desk back too?', cat: 'desk/premise', expect: c => c.did('push', 'desk') ? ({must: ['\\b(yes|did)\\b']}) : ({must: [NEG + '|just the chair|only the chair'], not: [YESSTART]})},
    {probe: 'What did you do last?', cat: 'desk/order', expect: () => ({must: ['chair'], not: ['^\\W*(i )?(took|closed)']})}]},
];

// ---- invented-action check: an action word in an answer that the log does not hold as done / doing / stopped
const LEX = [['wave', /\bwav(e|ed|ing)\b/], ['clap', /\bclap(s|ped|ping)?\b/], ['nod', /\bnod(s|ded|ding)?\b/], ['jump', /\bjump(s|ed|ing)?\b/],
  ['spin', /\b(spin|spun|spinning|twirl\w*)\b/], ['dance', /\bdanc(e|ed|ing)\b/], ['backflip', /\b(back ?flip\w*|flip(ped)?)\b/], ['bow', /\bbow(ed|ing)?\b/],
  ['thumbs_up', /\bthumbs?[- ]up\b/], ['pick_up', /\bpick(ed)? (it |them )?up\b|\bgrabbed\b|\bpicked\b/], ['toss', /\btoss(ed|ing)?\b|\bthrew\b/],
  ['drop', /\bdropp?(ed|ing)\b/], ['shrug', /\bshrug\w*\b/], ['wink', /\bwink\w*\b/], ['hand_over', /\b(switch\w*|pass(ed)?) (it|the ball)\b/],
  ['put_down', /\bput (it |the \w+ )?(down|back)\b/],
  ['put_on', /\bput (on )?(the |my )?headphones( on)?\b|\bwearing\b/], ['type_on', /\btyp(ed|ing)\b/], ['open', /\bopen(ed)? (the |my )?laptop\b/],
  ['sit_at', /\b(sat|sitting) (down )?(at|in)\b/], ['pull', /\bpull(ed)? (the |my )?(chair|desk)\b/]];
const NEGW = /\b(no|not|didn'?t|did not|haven'?t|have not|never|nothing|n't|without|wasn'?t|couldn'?t|instead of|rather than|yet)\b/;
function invented(answer, c) {
  const a = answer.toLowerCase(), out = [];
  for (const [name, re] of LEX) {
    const g = new RegExp(re.source, 'g'); let m;
    while ((m = g.exec(a))) {
      const before = a.slice(Math.max(0, m.index - 48), m.index);
      const clause = before.split(/[.!?;]|\bbut\b/).pop();
      if (NEGW.test(clause) || /\?\s*$/.test(a.slice(m.index, m.index + 60).split(/[.!]/)[0])) continue;   // negated / a question
      if (/\b(want|could|can|should|shall|let'?s|would|like to|if you|me to)\b/.test(clause)) continue;    // an offer, not a claim
      if (!c.mentionable(name)) out.push(name + ': "' + a.slice(Math.max(0, m.index - 30), m.index + 25).replace(/\s+/g, ' ') + '"');
    }
  }
  return out;
}
const NARR = /\b(i'?m|i am|i just|i'?ve)\b.{0,25}\b(wav|spin|spun|jump|clap|nod|danc|rais|hold|holding|pick|typ|sitting|wearing)/i;

// ---- one realtime WebSocket session (text mode)
function openWS() {
  return new Promise((res, rej) => {
    const ws = new WebSocket('wss://api.openai.com/v1/realtime?model=' + encodeURIComponent(args.model),
      ['realtime', 'openai-insecure-api-key.' + process.env.OPENAI_API_KEY]);
    const S = {ws, q: [], waiters: [], usd: 0, usage: []};
    ws.onmessage = e => { let m; try { m = JSON.parse(e.data); } catch (x) { return; } S.q.push(m); const w = S.waiters.slice(); S.waiters = []; w.forEach(f => f()); };
    ws.onerror = e => rej(new Error('ws error ' + (e.message || '')));
    ws.onopen = () => {
      ws.send(JSON.stringify({type: 'session.update', session: {type: 'realtime', instructions: sess.instructions, output_modalities: ['text'],
        tools: sess.tools, tool_choice: 'auto'}}));
      res(S);
    };
  });
}
async function next(S, pred, ms = 30000) {
  const t0 = Date.now();
  for (;;) {
    const i = S.q.findIndex(pred); if (i >= 0) return S.q.splice(i, 1)[0];
    if (Date.now() - t0 > ms) throw new Error('timeout');
    await new Promise(r => { S.waiters.push(r); setTimeout(r, 200); });
  }
}
const send = (S, ev) => S.ws.send(JSON.stringify(ev));
function usd(u) {
  const i = u.input_token_details || {}, o = u.output_token_details || {}, c = i.cached_tokens_details || {};
  const ct = c.text_tokens || 0, ca = c.audio_tokens || 0;
  return (Math.max(0, (i.text_tokens || 0) - ct) * RATES.text_in + Math.max(0, (i.audio_tokens || 0) - ca) * RATES.audio_in + (ct + ca) * RATES.cached +
    (o.text_tokens || 0) * RATES.text_out + (o.audio_tokens || 0) * RATES.audio_out) / 1e6;
}

// ---- the page's logic, in virtual time
async function runScenario(sc, mech) {
  let T = 0;
  const world = {};
  const mkWorld = () => {
    world.log = AL.createActionLog({now: () => T});
    const onEv = ev => world.log.push(Object.assign({}, ev, {t: undefined}));
    world.fake = REAL ? realWorld(onEv) : createFakeMotion({scene: 'muse', onAction: onEv});
    if (REAL) world.log = AL.createActionLog({now: () => world.fake.X.m.now(), limb: 'hand'});
    world.lastCmd = new Map(); world.noteVer = -1; world.transcript = [];
  };
  mkWorld();
  const advance = s => { const n = Math.round(s * 30); for (let i = 0; i < n; i++) { T += 1 / 30; world.fake.step(1 / 30); if (i % 3 === 0 || i === n - 1) world.log.observe(world.fake.state()); } };
  const settle = () => { if (!REAL) return; for (let i = 0; i < 90 * 30 && world.fake.busy(); i++) { T += 1 / 30; world.fake.step(); if (i % 3 === 0) world.log.observe(world.fake.state()); } advance(1); };
  const notes = mech === 'notes' || mech === 'both';
  let S = await openWS(); await next(S, m => m.type === 'session.updated');
  const sendNoteIfChanged = force => {
    if (!notes) return;
    const v = world.log.version();
    if (!force && v === world.noteVer) return;
    world.noteVer = v;
    const note = 'Silent context (not words he said) - never read aloud, quote, describe or mention it: ' + world.log.summary();
    const id = 'alog_' + Math.random().toString(36).slice(2, 10);
    if (args.replace === '1' && world.noteId) send(S, {type: 'conversation.item.delete', item_id: world.noteId});
    send(S, {type: 'conversation.item.create', item: {id, type: 'message', role: 'system', content: [{type: 'input_text', text: note}]}});
    world.noteId = id; world.notes = (world.notes || 0) + 1;
  };
  function runCmd(cmd, source) {
    // like page.html's IntentDedupe: the same act from another source within 3 s is one act (matcher gesture:jump = model move:jump)
    const key = JSON.stringify([cmd.name || cmd.verb || cmd.target || cmd.action, cmd.object ? (cmd.object.name || cmd.object.color || cmd.object.ref) : '']);
    const prev = world.lastCmd.get(key);
    if (prev && T - prev.t < 3 && prev.source !== source) return Object.assign({dup: true}, prev.res, {did: 'already ' + (prev.res.did || 'doing that') + ' (started a moment ago)'});
    const res = world.fake.command(cmd, source);
    world.log.command(cmd, res, source);
    world.lastCmd.set(key, {t: T, source, res});
    return res;
  }
  function toolCmd(name, a) {
    switch (name) {
      case 'gesture': return {action: 'gesture', name: a.name, side: a.side, repeat: a.repeat, hold: a.hold};
      case 'look': return {action: 'look', target: a.target, hold: a.hold};
      case 'expression': return {action: 'expression', name: a.name, hold: a.hold};
      case 'release': return {action: 'release', parts: a.parts || 'all'};
      case 'pose': return {action: 'pose', targets: a.targets || {}, hold: a.hold};
      case 'move': return Object.assign({action: 'move'}, a);
      case 'object': if (a.verb === 'turn_back' || a.verb === 'stand_up') return Object.assign({action: a.verb}, a); return Object.assign({action: 'object'}, a);
      case 'object': return Object.assign({action: 'object'}, a);
    }
    return null;
  }
  async function turn(text, matcher) {
    // he starts talking: the page flushes a changed log first (in context before his words end)
    sendNoteIfChanged(false);
    advance(Math.max(0.8, text.length / 14));                      // his speech
    if (matcher) runCmd(matcher, 'matcher');                       // the server's fast matcher acts at his speech end
    send(S, {type: 'conversation.item.create', item: {type: 'message', role: 'user', content: [{type: 'input_text', text}]}});
    world.transcript.push(['you', text]);
    let said = '', calls = [];
    send(S, {type: 'response.create'});
    for (let round = 0; round < 3; round++) {
      const d = await next(S, m => m.type === 'response.done' || m.type === 'error', 40000);
      if (d.type === 'error') throw new Error('api error ' + JSON.stringify(d.error).slice(0, 200));
      const r = d.response || {}; if (r.usage) { S.usd += usd(r.usage); S.usage.push(r.usage); }
      const outs = r.output || [];
      let txt = '';
      for (const it of outs) if (it.type === 'message') for (const c of it.content || []) if (c.text || c.transcript) txt += (c.text || c.transcript);
      said += (said && txt ? ' ' : '') + txt;
      const fcs = outs.filter(x => x.type === 'function_call');
      advance(round === 0 ? 1.1 : 0.9);                              // model latency to the call / words
      for (const fc of fcs) {
        let a = {}; try { a = JSON.parse(fc.arguments || '{}'); } catch (e) {}
        let out;
        if (fc.name === 'recall_actions') out = {log: world.log.summary({sinceS: a.since_s || undefined})};
        else {
          const cmd = toolCmd(fc.name, a);
          const res = cmd ? runCmd(cmd, 'tool') : {ok: false, reason: 'unknown tool'};
          out = mech === 'none' ? {ok: true} : Object.assign({ok: res.ok !== false}, res.did ? {did: res.did} : {}, res.reason ? {reason: res.reason} : {}, {now: world.log.brief()});
          if (out.dup) delete out.dup;
        }
        calls.push({name: fc.name, args: a, out});
        send(S, {type: 'conversation.item.create', item: {type: 'function_call_output', call_id: fc.call_id, output: JSON.stringify(out)}});
      }
      if (!fcs.length || (txt && round > 0)) break;
      if (fcs.length && !txt) { send(S, {type: 'response.create'}); continue; }
      break;
    }
    advance(Math.max(0.6, said.length / 15));                       // its reply plays
    world.transcript.push(['avatar', said]);
    return {said, calls};
  }
  // ground-truth helpers for the dynamic expectations
  const items = () => world.log.items();
  const ctx = {
    did: (n, obj) => items().some(it => it.name === n && (it.status === 'done' || it.status === 'doing') && (!obj || String(it.object || '').includes(obj))),
    worn: id => REAL ? (world.fake.state().wearing || []).includes(id) : world.fake.state().props.some(p => p.id === id && p.where === 'head'),
    posture: () => world.fake.state().posture,
    lid: id => REAL ? ((world.fake.state().open || []).includes(id) ? 'open' : 'closed') : (world.fake.state().props.find(p => p.id === id) || {}).open ? 'open' : 'closed',
    didLast: n => { const x = items().filter(it => it.name === n).pop(); return !!x && x.status === 'done'; },
    active: n => world.fake.state().active.some(a => a.name === n),
    holdingAny: () => { const h = world.fake.state().holding; return !!(h.left || h.right); },
    holding: id => { id = REAL && RNAME[id] || id; const h = world.fake.state().holding; return h.left === id || h.right === id; },
    inPaw: side => { const id = world.fake.state().holding[side]; return id ? id : '(nothing|empty|not holding)'; },
    pawOf: id => { id = REAL && RNAME[id] || id; const h = world.fake.state().holding; return h.left === id ? 'left' : h.right === id ? 'right' : '(not|nothing|neither|put)'; },
    countWord: n => { const k = items().filter(it => it.name === n && it.status === 'done').reduce((s, it) => s + (it.count || 1), 0); return [k, AL.VERBS ? ['zero', 'one', 'two', 'three', 'four', 'five', 'six'][k] : k, k === 1 ? 'once' : k === 2 ? 'twice' : 'x'].join('|'); },
    progressN: n => { const it = items().filter(x => x.name === n).pop(); return it ? (it.status === 'done' ? (it.count || 1) : it.progress || 0) : 0; },
    progressWord: n => { const it = items().filter(x => x.name === n).pop(); const k = it ? (it.status === 'done' ? (it.count || 1) : it.progress || 0) : 0;
      return k === 0 ? '(none|zero|didn\'?t (finish|complete|get)|not (even )?one|no full|barely|part)' : k === 1 ? '\\b(one|1|once|just one|only one)\\b' : '\\b(' + k + '|' + ['', '', 'two|twice', 'three', 'four', 'five', 'six'][k] + ')\\b'; },
    mentionable: n => (n === 'put_on' && world.fake.state().props.some(p => p.where === 'head')) || (n === 'sit_at' && world.fake.state().posture === 'seated') || items().some(it => (it.name === n || (n === 'pick_up' && it.name === 'pick_up') || (n === 'spin' && it.name === 'spin') || (n === 'put_down' && /put_(down|back)/.test(it.name)))
      && it.status !== 'failed' && it.status !== 'pending') || world.log._L.reflex.some(r => r.name.replace(/_back$/, '') === n),
  };
  const out = [];
  for (const st of sc.steps) {
    if (st.wait) { advance(st.wait); settle(); sendNoteIfChanged(false); }
    else if (st.act) { runCmd(st.act, st.src || 'matcher'); advance(0.1); }
    else if (st.reflex) { world.fake.reflex(st.reflex); advance(0.1); }
    else if (st.reset) { try { S.ws.close(); } catch (e) {} T = 0; mkWorld(); S = await openWS(); await next(S, m => m.type === 'session.updated'); }
    else if (st.redial) {
      const hist = world.transcript.slice(-20);
      const usdOld = S.usd; try { S.ws.close(); } catch (e) {}
      S = await openWS(); S.usd += usdOld; await next(S, m => m.type === 'session.updated');
      send(S, {type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: 'Silent context (not words he said) - never read aloud, quote, describe or mention it: the conversation so far (the call reconnected; carry on naturally, do not repeat or mention it):\n' + hist.map(([w, t]) => (w === 'you' ? 'Him: ' : 'You: ') + t).join('\n')}]}});
      world.noteVer = -1; sendNoteIfChanged(true);
    }
    else if (st.user) { const r = await turn(st.user, st.matcher); out.push({kind: 'turn', text: st.user, said: r.said, calls: r.calls, t: +T.toFixed(1)}); }
    else if (st.probe) {
      const exp0 = st.expect(ctx);                                  // the truth when he asks
      const r = await turn(st.probe, null);
      const exp = exp0, a = (r.said || '').replace(/[\u2018\u2019]/g, "'");
      const fails = [];
      for (const re of exp.must || []) if (!new RegExp(re, 'i').test(a)) fails.push('missing /' + re + '/');
      for (const re of exp.not || []) if (new RegExp(re, 'i').test(a)) fails.push('has /' + re + '/');
      if (exp.order) { let at = -1; for (const re of exp.order) { const m = new RegExp(re, 'i').exec(a.slice(at + 1)); if (!m) { fails.push('order: missing /' + re + '/ after pos ' + at); break; } at += 1 + m.index; } }
      const inv = invented(a, ctx);
      if (exp.none && inv.length) fails.push('claims: ' + inv.join('; '));
      if (/\b(action log|the log|my log|in the log|notes?\b|records?\b|context)\b/i.test(a)) fails.push('mentions the log / notes');
      const narr = exp.narr && NARR.test(a);
      if (narr) fails.push('narrates its movement');
      out.push({kind: 'probe', id: sc.id, cat: st.cat, q: st.probe, said: a, calls: r.calls, ok: !fails.length, fails, invented: inv, narr: !!narr, t: +T.toFixed(1),
        truth: world.log.summary()});
    }
  }
  try { S.ws.close(); } catch (e) {}
  return {out, usd: S.usd, notes: world.notes || 0};
}

(async () => {
  const only = args.only ? new Set(args.only.split(',')) : null;
  const res = {mech: args.mech, replace: args.replace, model: args.model, scenarios: [], usd: 0};
  for (let rep = 0; rep < +args.rep; rep++) for (const sc of SC) {
    if (only && !only.has(sc.id)) continue;
    let r;
    for (let tries = 0; tries < 2; tries++) { try { r = await runScenario(sc, args.mech); break; } catch (e) { console.error(sc.id, 'retry', e.message); } }
    if (!r) continue;
    res.scenarios.push({id: sc.id, rep, ...r}); res.usd += r.usd;
    for (const p of r.out.filter(x => x.kind === 'probe')) console.log((p.ok ? 'PASS ' : 'FAIL ') + p.id + ' [' + p.cat + '] ' + p.q + '\n     → ' + p.said.replace(/\s+/g, ' ').slice(0, 220) + (p.ok ? '' : '\n     ✗ ' + p.fails.join(' | ')));
  }
  const probes = res.scenarios.flatMap(s => s.out.filter(x => x.kind === 'probe'));
  res.n = probes.length; res.pass = probes.filter(p => p.ok).length; res.invented = probes.reduce((s, p) => s + p.invented.length, 0);
  res.narr = probes.filter(p => p.narr).length; res.usd = +res.usd.toFixed(4);
  fs.writeFileSync(path.join(args.out, args.mech + (REAL ? '-real' : '') + (args.replace === '1' ? '-replace' : '') + (args.tag ? '-' + args.tag : '') + '.json'), JSON.stringify(res, null, 1));
  console.log(`\nmech=${args.mech} replace=${args.replace}: ${res.pass}/${res.n} = ${(100 * res.pass / Math.max(1, res.n)).toFixed(1)}%  invented-flags=${res.invented}  narration=${res.narr}  $${res.usd}`);
})().catch(e => { console.error(e); process.exit(1); });
