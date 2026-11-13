// speech smoke test (node, no vendor): app/actionlog.js + app/cospeech.js against the REAL app/motion.js (the
// engine: actionlog observes its gestures; cospeech plays through the shim) and the contract-shaped stand-in.
//   node tests/speech_smoke.cjs [outdir]   → PASS/FAIL lines, exit code; writes <outdir>/cospeech_<id>.rig.jsonl for render_timeline.py
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const M = load('app/motion.js'), AL = load('app/actionlog.js'), CS = load('app/cospeech.js');
const {createFakeMotion} = require('./speech_fake_motion.cjs');
const OUT = process.argv[2] || path.join(path.join(require('os').tmpdir(), 'avatar-tests'), 'speech_smoke');
const TTS = path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'speech', 'tts');
fs.mkdirSync(OUT, {recursive: true});
let fails = 0;
const check = (name, ok, detail) => { if (!ok) fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  — ' + detail : '')); };
const DT = 1 / 30;

// ---- 1. the action log on the engine (observed gestures, not issued commands)
{
  let T = 0; const L = AL.createActionLog({now: () => T});
  const m = M.createMotion({seed: 3, onAction: e => L.push(Object.assign({}, e, {t: undefined}))});   // the real engine's events
  const step = s => { for (let i = 0; i < s * 30; i++) { T += DT; m.step(DT, {}); L.pollLegacy(m); } };
  const cmd = (c, src) => { const r = m.command(c, {src}); L.noteSource(src); L.command(c, r, src); return r; };
  step(0.5);
  check('empty log says nothing happened', /Nothing yet/.test(L.summary()));
  cmd({action: 'gesture', name: 'wave', side: 'right', repeat: 2}, 'matcher'); step(0.3);
  check('a running wave is "still going", never done', L.items().some(i => i.name === 'wave' && i.status === 'doing') && /STILL GOING/.test(L.summary()));
  step(4);
  check('the wave finished (observed end)', L.items().some(i => i.name === 'wave' && i.status === 'done'));
  cmd({action: 'gesture', name: 'clap'}, 'tool'); step(0.4); cmd({action: 'release', parts: 'all'}, 'matcher'); step(1);
  check('a released clap is interrupted', L.items().some(i => i.name === 'clap' && i.status === 'interrupted'), JSON.stringify(L.items().map(i => i.name + ':' + i.status)));
  cmd({action: 'gesture', name: 'moonwalk_backflip'}, 'tool'); step(0.2);
  check('an unknown gesture is logged as failed', L.items().some(i => i.status === 'failed'));
  const s = L.summary();
  check('summary in order with who asked', s.indexOf('waved') < s.indexOf('clap') && /\(asked\)/.test(s), s.slice(0, 300));
  L.reset(); check('session reset empties the log', /Nothing yet/.test(L.summary()));
}
// ---- 2. the desk scenario on the contract stand-in (worn / seated / lid / activity / order) + hand wording
{
  let T = 0; const L = AL.createActionLog({now: () => T, limb: 'hand'});
  const f = createFakeMotion({scene: 'muse', onAction: e => L.push(Object.assign({}, e, {t: undefined}))});
  const adv = s => { for (let i = 0; i < s * 30; i++) { T += DT; f.step(DT); L.observe(f.state()); } };
  const c = cmd => { const r = f.command(cmd, 'matcher'); L.command(cmd, r, 'matcher'); adv(2.6); return r; };
  for (const [v, o, x] of [['put_on', 'headphones'], ['pull', 'chair'], ['pull', 'desk'], ['sit_at', 'desk'], ['slide', 'laptop', 'closer'], ['open', 'laptop'], ['type_on', 'laptop']])
    c({action: 'object', verb: v, object: {name: o}, target: x});
  const mid = L.summary();
  check('desk: typing is in progress while it types', /typing on the laptop.*STILL GOING/.test(mid));
  c({action: 'turn_back'});
  const s = L.summary({maxChars: 1400});
  const order = ['put on the headphones', 'pulled the chair', 'pulled the desk', 'sat down at the desk', 'slid the laptop closer', 'opened the laptop', 'typed on the laptop', 'turned back'];
  let at = -1, ok = true; for (const k of order) { const i = s.indexOf(k); if (i <= at) { ok = false; break; } at = i; }
  check('desk: steps in order', ok, s);
  check('desk: state = wearing headphones, at the desk, laptop open, hands empty', /wearing the headphones/.test(s) && /at the desk/.test(s) && /laptop is open/.test(s) && /hands empty/.test(s) && !/paw/.test(s));
  const bad = {action: 'object', verb: 'put_on', object: {name: 'mug'}}; const r = f.command(bad, 'tool'); L.command(bad, r, 'tool');
  check('a refusal is logged once, as failed', L.items().filter(i => i.status === 'failed').length === 1 && /couldn't/.test(L.summary({maxChars: 1600})));
}
// ---- 3. co-speech through the engine (shim): strokes land, hands move, busy paws are respected, barge-in cancels
function wav(p) { const b = fs.readFileSync(p); let o = 12; while (b.toString('ascii', o, o + 4) !== 'data') o += 8 + b.readUInt32LE(o + 4); const n = b.readUInt32LE(o + 4) / 2; const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = b.readInt16LE(o + 8 + 2 * i) / 32768; return x; }
function speak(id, opts = {}) {
  const j = JSON.parse(fs.readFileSync(path.join(TTS, id + '.json'))), x = wav(path.join(TTS, id + '.wav'));
  const m = M.createMotion({seed: 5}); const ON = 0.8;
  const C = CS.createCospeech({motion: m, seed: 11, motionNow: () => m.state.t});
  if (opts.cmd) m.command(opts.cmd);
  const rows = [], names = M.RIG_PARAMS.concat(['fingers_l', 'fingers_r']);
  let ci = 0, textSent = false, cancelled = false;
  const N = Math.round((ON + x.length / 24000 + 1.2) * 30);
  for (let f = 0; f < N; f++) {
    const T = (f + 1) * DT;
    if (!textSent && T >= ON - 0.25) { for (const tk of j.text.match(/\S+\s*/g)) C.text(tk, {response_id: 'r'}); textSent = true; }
    let lv = 0;
    while (ON + (ci + 1) * 0.02 <= T + 1e-9) {
      const a = ci * 480, b = Math.min(x.length, a + 480), cut = opts.cut && ci * 0.02 >= opts.cut; let e = 0;
      for (let i = a; i < b; i++) e += x[i] * x[i];
      const r = b > a && !cut ? Math.sqrt(e / (b - a)) : 0; lv = r;
      C.audio(ON + ci * 0.02, r, r < 1e-3, 0.02, b > a && !cut ? x.subarray(a, b) : undefined); ci++;
      if (cut && !cancelled) { cancelled = T; C.cancel('barge-in'); }
    }
    C.tick(T);
    if (C.covering(T) && m.state.speech) m.state.speech.beatNext = m.state.t + 0.5;
    const o = m.step(DT, {avatarVoice: {rms: lv, voiced: lv > 0.008}});
    Object.assign(o, C.extras(m.state.t));
    rows.push({t: +T.toFixed(4), o: Object.assign({}, o)});
  }
  fs.writeFileSync(path.join(OUT, 'cospeech_' + id + (opts.tag || '') + '.rig.jsonl'), rows.map(r => JSON.stringify({t: +(r.t - ON).toFixed(4), names, a: names.map(n => Math.round((r.o[n] || 0) * 1e4) / 1e4)})).filter((_, i) => rows[i].t >= ON).join('\n') + '\n');
  return {C, rows, cancelled, ON};
}
{
  const {C, rows} = speak('three');
  const commits = C.commits();
  check('co-speech commits strokes into the engine', commits.length >= 2, commits.map(c => c.type + ':' + c.w).join(' '));
  // hand displacement from rest near each arm stroke's apex
  const moved = commits.filter(c => !/nod|brow|shake|shrug_small/.test(c.type)).map(c => {
    const r = rows.find(r => r.t >= c.apex) || rows[rows.length - 1]; const s = c.side || 'l';
    return Math.hypot(r.o['hand_' + s + '_x'], r.o['hand_' + s + '_y'] - 0, r.o['hand_' + s + '_z']);
  });
  check('the paw is out in gesture space at stroke apexes', moved.length && moved.every(d => d > 0.15), moved.map(d => d.toFixed(2)).join(' '));
  const fing = rows.some(r => (r.o.fingers_l || 0) > 2.5 || (r.o.fingers_r || 0) > 2.5);
  check('counting: "three things" raises fingers (fingers_* via the shim extras)', fing);
  // smoothness: max per-frame change of the hand targets (≤ 0.12 arm lengths per 33 ms frame ≈ 3.6 /s)
  let vmax = 0; for (let i = 1; i < rows.length; i++) for (const p of ['hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z']) vmax = Math.max(vmax, Math.abs(rows[i].o[p] - rows[i - 1].o[p]));
  check('no popping: max hand-target step per frame ≤ 0.12', vmax <= 0.12, vmax.toFixed(3));
}
{
  const {C, rows, cancelled} = speak('plan', {cut: 1.6, tag: '_bargein'});
  const after = C.commits().filter(c => c.commitT > cancelled).length;
  const live = rows.filter(r => r.t > cancelled + 0.5).every(r => Math.abs(r.o.hand_l_y) < 0.25 && Math.abs(r.o.hand_r_y) < 0.25);
  check('barge-in: nothing committed after the cancel, paws back down within 0.5 s', after === 0 && live, 'after=' + after);
}
{
  const {C} = speak('big', {cmd: {action: 'gesture', name: 'raise_paw', side: 'right', hold: true}, tag: '_busy'});
  const used = C.commits().filter(c => c.side === 'l' || (/big|open_palms|exclaim|beat_both|shrug$|small|long|count_both/.test(c.type)));
  check('a held paw (own right = screen-left) never gets a co-speech gesture', used.length === 0, C.commits().map(c => c.type + ':' + c.side).join(' '));
}
for (const id of ['ordinals', 'tower', 'ball']) speak(id);
// ---- 4. the REAL muse engine (moves.js + the real muse JSON): the desk headline, its log, co-speech while typing / turned back
{
  const {makeMuse} = require('./muse_lib.cjs');
  const X = makeMuse(17); let T = 0, seen = 0;
  const L = AL.createActionLog({now: () => X.m.now(), limb: 'hand'});
  const sync = () => { for (; seen < X.ev.length; seen++) L.push(Object.assign({}, X.ev[seen], {t: undefined})); L.observe(X.m.state()); };
  const step = n => { for (let i = 0; i < n; i++) { X.step(1); sync(); } };
  const cmd = c => { const r = X.m.command(c, {src: 'matcher'}); sync(); L.command(c, r, 'matcher'); return r; };
  const idle = (s = 60) => { for (let i = 0; i < s * 30; i++) { step(1); const E = X.m.engine.E; if (!E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue && i > 3) break; } };
  step(10);
  cmd({action: 'object', verb: 'put_on', object: {name: 'headphones'}}); step(20);
  check('muse: a running job is "STILL GOING" with only its finished steps', /STILL GOING/.test(L.summary({maxChars: 2000})) && !/put on the pink headphones/.test(L.summary({maxChars: 2000}).split('STILL GOING')[0]), L.summary({maxChars: 600}).slice(150, 420));
  idle(); cmd({action: 'object', verb: 'work_on', object: {name: 'laptop'}}); idle(); step(60);
  const s1 = L.summary({maxChars: 2400});
  check('muse: wearing the headphones, sitting on the chair at the desk, typing still going', /wearing the pink headphones/.test(s1) && /sitting on the chair at the desk/.test(s1) && /typing on the silver laptop.*STILL GOING/.test(s1) && !/paw/.test(s1), s1.slice(-420));
  const ord = ['picked up the pink headphones', 'put on the pink headphones', 'picked up the silver laptop', 'sat down on the chair at the desk'];
  let at = -1, ok = true; for (const k of ord) { const i = s1.indexOf(k); if (i <= at) { ok = false; break; } at = i; }
  check('muse: the steps that ran, in order', ok);
  // co-speech while typing: no stroke on a typing hand
  const C = CS.createCospeech({motion: X.m, seed: 3});
  const busy = X.m.armsBusy();
  const talk = (txt, secs) => { const t0 = X.m.now(); C.text(txt, {response_id: 'q' + t0}); const rows = [];
    for (let i = 0; i < secs * 30; i++) { const tt = X.m.now() - t0; for (let k = 0; k < 2; k++) C.audio(t0 + tt + 0.02 * k, 0.05 + 0.04 * Math.sin(tt * 25 + k), false, 0.02); C.tick(X.m.now()); step(1); }
    return C.commits().filter(c => c.commitT >= t0); };
  const typingCommits = talk('So, I think this is a really big project, and maybe we can finish it today!', 5);
  const armTyping = typingCommits.filter(c => !/nod|brow|head_shake|shrug_small/.test(c.type));
  check('co-speech while typing: no arm strokes on typing hands', (busy.l || busy.ctrlL) && (busy.r || busy.ctrlR) ? armTyping.length === 0 : true, `busy ${JSON.stringify(busy)}; strokes ${typingCommits.map(c => c.type).join(' ')}`);
  cmd({action: 'move', name: 'turn_back'}); step(90);
  const s2 = L.summary({maxChars: 2400});
  check('muse: turned back toward him, typing paused, still seated', /turned back toward him/.test(s2) && /paused/.test(s2) && /sitting on the chair/.test(s2), s2.slice(-300));
  C.reset(); const r0 = X.rows.length; const back = talk('Well, there are three things I love: the sun, the sea, and you!', 6);
  const lift = Math.max(...X.rows.slice(r0).map(r => Math.max(r.hand_l_y || 0, r.hand_r_y || 0))), lift0 = Math.max(...X.rows.slice(r0 - 60, r0).map(r => Math.max(r.hand_l_y || 0, r.hand_r_y || 0)));
  check('co-speech through the real playGesture moves a free hand (seated, turned back)', lift > lift0 + 0.12, `hand y max ${lift0.toFixed(2)} → ${lift.toFixed(2)}`);
  const armBack = back.filter(c => !/nod|brow|head_shake|shrug_small/.test(c.type));
  check('co-speech turned back (hands free): arm gestures again', armBack.length >= 1, back.map(c => c.type + ':' + c.side).join(' ') + ' busy ' + JSON.stringify(X.m.armsBusy()));
  const r = cmd({action: 'object', verb: 'put_on', object: {name: 'mug'}}); step(3);
  check('muse: a refusal is logged as failed with its reason', r.ok === false && L.items().some(i => i.status === 'failed' && /mug/.test(i.reason || i.text)));
}
// ---- 5. answer(q) (speech-c2): current-state questions from the OBSERVED state, history questions from the log, never
// inventing — on the REAL muse engine (fed exactly like section 4), then a few on the contract stand-in
{
  const {makeMuse} = require('./muse_lib.cjs');
  const X = makeMuse(17); let seen = 0;
  const L = AL.createActionLog({now: () => X.m.now(), limb: 'hand'});
  const sync = () => { for (; seen < X.ev.length; seen++) L.push(Object.assign({}, X.ev[seen], {t: undefined})); L.observe(X.m.state()); };
  const step = n => { for (let i = 0; i < n; i++) { X.step(1); sync(); } };
  const cmd = c => { const r = X.m.command(c, {src: 'matcher'}); sync(); L.command(c, r, 'matcher'); return r; };
  const idle = (s = 60) => { for (let i = 0; i < s * 30; i++) { step(1); const E = X.m.engine.E; if (!E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue && i > 3) break; } };
  const said = [];
  const qa = (q, pos, neg) => { const a = L.answer(q); said.push(a); check('answer: "' + q + '"', pos.every(r => r.test(a)) && !(neg || []).some(r => r.test(a)), a); return a; };
  step(10);
  qa('what did you just do?', [/^nothing yet\b/]);
  cmd({action: 'gesture', name: 'wave'}); step(120); cmd({action: 'gesture', name: 'wave'}); step(120);
  cmd({action: 'move', name: 'spin', count: 3}); step(45); cmd({action: 'stop'}); idle();
  cmd({action: 'object', verb: 'put_on', object: {name: 'headphones'}}); idle();
  cmd({action: 'object', verb: 'work_on', object: {name: 'laptop'}}); idle(); step(60);
  // typing: lid open, seated at the desk
  qa('what are you wearing?', [/^you're wearing the pink headphones$/], [/nothing|not wearing/]);
  qa('Are you wearing the headphones?', [/^yes\b/, /pink headphones/], [/^no\b/]);
  qa('are you wearing the mug?', [/^no\b/, /you're wearing the pink headphones/], [/^yes\b/]);
  qa('what are you doing?', [/^you're typing on the silver laptop, lid open, sitting on the chair at the desk$/], [/paused|nothing/]);
  qa('are you working?', [/^yes\b/, /typing on the silver laptop/, /sitting on the chair at the desk/]);
  qa('are you sitting?', [/^yes\b/, /sitting on the chair at the desk/]);
  qa('are you standing?', [/^no\b/, /sitting on the chair at the desk/], [/^yes/]);
  qa('where are you sitting?', [/on the chair at the desk/], [/not sitting/]);
  qa('are you at the desk?', [/^yes\b/, /at the desk/]);
  qa('is the laptop open?', [/^yes\b/, /silver laptop is open/], [/closed/]);
  qa('is the lid closed?', [/^no\b/, /silver laptop is open/]);
  qa('what are you holding?', [/^nothing\b/, /both hands are empty/], [/paw/]);
  qa('are you holding the book?', [/^no\b/, /not holding the blue book/, /both hands are empty/], [/^yes/]);
  qa("what's in your left hand?", [/^nothing\b/, /left hand is empty/]);
  qa('where is the laptop?', [/^the silver laptop is on the desk/]);
  qa('where are the headphones?', [/pink headphones are on your head/, /wearing them/]);
  qa('are you looking at me?', [/^no\b/, /screen of the silver laptop/]);
  qa('did you wave?', [/^yes\b/, /waved/, /twice/]);
  qa('how many times did you wave?', [/^twice\b/], [/once|3 times/]);
  qa('did you sit down?', [/^yes\b/, /sat down on the chair at the desk/]);
  qa('did you pick up the laptop?', [/^yes\b/, /picked up the silver laptop/]);
  qa('did you open the laptop?', [/^yes\b/, /opened the silver laptop/]);
  qa('did you drop the laptop?', [/^no\b/, /did not drop/], [/^yes|dropped/]);
  qa('did you do a backflip?', [/^no\b/], [/^yes|did a backflip/]);
  qa('why did you take the headphones off?', [/^you didn't take the headphones off/, /on your head/], [/because/]);
  qa('did you spin?', [/started to spin/, /stopped after 2 of 3/], [/^yes/]);
  qa('how many times did you spin?', [/^twice\b/, /2 of 3/]);
  qa('what did you do first?', [/^first, you waved/]);
  qa('what did you just do?', [/^you just /, /started typing on the silver laptop, and are still at it/]);
  // turned back: typing paused, facing him, still seated
  cmd({action: 'move', name: 'turn_back'}); step(90);
  qa('what are you doing?', [/typing on the silver laptop is paused/, /turned back toward him/, /still sitting on the chair at the desk/], [/^you're typing/]);
  qa('are you working?', [/^not right now\b/, /paused/, /still sitting/], [/^yes/]);
  qa('are you looking at me?', [/^yes\b/, /looking at him/]);
  qa('which way are you facing?', [/turned back toward him/]);
  qa('are you still sitting?', [/^yes\b/, /chair at the desk/]);
  qa('what did you just do?', [/^you just turned back to him/]);
  qa('what did you do before that?', [/^before that, you /, /typing on the silver laptop/]);
  qa('did you turn back?', [/^yes\b/, /turned back/]);
  // a refusal: tried but couldn't — never "yes"
  cmd({action: 'object', verb: 'put_on', object: {name: 'mug'}}); step(3);
  qa('did you put on the mug?', [/^no\b/, /tried to put on the red mug but couldn't/, /something to wear/], [/^yes/]);
  qa('what did you just do?', [/^you just tried to put on the red mug but couldn't/]);
  qa('did you put on the headphones?', [/^yes\b/, /put on the pink headphones/]);
  qa("why couldn't you put on the mug?", [/^because the red mug isn't something to wear/]);
  qa('how many times did you put something on?', [/^once\b/, /pink headphones/, /couldn't/]);
  check('answer: hand wording, never "paw" (the muse has hands)', said.every(a => !/\bpaws?\b/.test(a)), said.filter(a => /\bpaws?\b/.test(a)).join(' | '));
}
{
  let T = 0; const L = AL.createActionLog({now: () => T, limb: 'hand'});
  check('answer (stand-in): no snapshot yet → says it can\'t tell, never guesses', /can't tell/.test(L.answer('what are you holding?')) && /can't tell/.test(L.answer('what are you wearing?')), L.answer('what are you holding?'));
  const f = createFakeMotion({scene: 'muse', onAction: e => L.push(Object.assign({}, e, {t: undefined}))});
  const adv = s => { for (let i = 0; i < s * 30; i++) { T += DT; f.step(DT); L.observe(f.state()); } };
  const c = cmd => { const r = f.command(cmd, 'matcher'); L.command(cmd, r, 'matcher'); adv(2.6); return r; };
  const qa = (q, pos, neg) => { const a = L.answer(q); check('answer (stand-in): "' + q + '"', pos.every(r => r.test(a)) && !(neg || []).some(r => r.test(a)), a); };
  c({action: 'object', verb: 'pick_up', object: {name: 'ball'}, side: 'right'});
  qa("what's in your right hand?", [/red ball/], [/empty/]);
  qa('which hand is the ball in?', [/^your right hand\b/]);
  c({action: 'object', verb: 'put_down', object: {name: 'ball'}});
  for (const [v, o, x] of [['put_on', 'headphones'], ['pull', 'chair'], ['pull', 'desk'], ['sit_at', 'desk'], ['slide', 'laptop', 'closer'], ['open', 'laptop'], ['type_on', 'laptop']])
    c({action: 'object', verb: v, object: {name: o}, target: x});
  qa('what are you wearing?', [/^you're wearing the headphones$/]);
  qa('what are you doing?', [/^you're typing on the laptop, lid open, sitting at the desk$/]);
  c({action: 'turn_back'});
  qa('are you working?', [/^no\b/, /sitting at the desk, facing him/]);
  const bad = {action: 'object', verb: 'put_on', object: {name: 'mug'}}; L.command(bad, f.command(bad, 'tool'), 'tool');
  qa('did you put on the mug?', [/^no\b/, /tried to put on the green mug but couldn't/]);
  qa('did you drop the ball?', [/^no\b/, /red ball is on the table/]);
}
console.log(fails ? `FAIL (${fails})` : 'PASS all speech smoke checks');
process.exit(fails ? 1 : 0);
