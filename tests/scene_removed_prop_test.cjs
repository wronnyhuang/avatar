// node tests/scene_removed_prop_test.cjs — the headphones and their stand left the classroom
// scene; the page side on the REAL scene (renderer/animals/muse.props.json, the raw file the page receives):
//  (a) the move engine (moves.js via muse_lib) has no headphones; every other prop keeps its slot (prop{k}_* column) vs the
//      pre-removal scene (tests/fixtures/muse_props_with_headphones.json: chair 1 ... eraser 15);
//  (d) asking for them is declined honestly — the matcher / the object tool get the engine's "there is no headphones here"
//      (ok false, nothing moves, nothing held or worn), do_action never mimes it with the keyframe planner, no crash — while
//      "put on the hat" still works;
//  (e) the side table is a plain table now: "put the book on the side table" / "put it back" (the book's home is there)
//      work and never talk about a stand; on the fixture the headphones still go "back on the stand" (generic engine code).
// The hub / renderer side is tests/scene_removed_prop_test.py. Exit code = failures.
'use strict';
const fs = require('fs'), path = require('path');
const {makeMuse, museFiles, HP_FIXTURE} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent} = load('app/intent.js');
const VTL = load('app/voicetools.js');
let fails = 0, n = 0;
const check = (name, ok, d) => { n++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (d !== undefined && !ok ? '  — ' + (typeof d === 'string' ? d : JSON.stringify(d)) : '')); if (!ok) fails++; };
const RULES = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'app', 'intent_rules.json')));
const REAL = path.join(__dirname, '..', 'renderer', 'animals', 'muse.props.json');
const O = (verb, name, x) => Object.assign({action: 'object', verb, object: {name}}, x || {});
const idle = T => { const E = T.m.engine.E; return !E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue; };
const runIdle = (T, s = 60) => { for (let i = 0; i < s * 30; i++) { T.step(1, false); if (idle(T) && i > 3) return true; } return false; };
const where = (T, id) => (T.m.state().props.find(p => p.id === id) || {}).where;
const handsEmpty = T => { const h = T.m.state().holding; return !h.left && !h.right; };

// ---------------------------------------------------------------- (a) the engine's scene
{
  const files = museFiles();
  check('(a) the test runs on the real props file', /real props/.test(files.src) && files.props.props.some(p => p.id === 'headphones' && p.removed), files.src);
  const T = makeMuse(3); T.step(3, false);
  const E = T.m.engine.E, F = makeMuse(3, {propsPath: HP_FIXTURE}); F.step(3, false);
  check('(a) moves.js: no prop with id headphones (engine list and props())', !E.P.some(p => p.id === 'headphones') && !T.m.props().some(p => p.id === 'headphones') && !T.m.state().props.some(p => p.id === 'headphones'));
  check('(a) the fixture engine still has them in slot 0', F.m.engine.E.P.some(p => p.id === 'headphones' && p.k === 0));
  const slots = Object.fromEntries(E.P.map(p => [p.id, p.k])), want = Object.fromEntries(F.m.engine.E.P.filter(p => p.id !== 'headphones').map(p => [p.id, p.k]));
  check('(a) every other prop keeps its slot (chair 1 ... eraser 15)', JSON.stringify(slots) === JSON.stringify(want) && slots.chair === 1 && slots.eraser === 15 && Object.keys(slots).length === 15, {slots, want});
  const st = E.places.find(p => p.id === 'side_table');
  check('(a) the side table is a place without a stand', !!st && st.stand_top_y === undefined && !(st.aliases || []).some(a => /stand|headphone/.test(a)), st);
  // a scene with no removed prop is untouched (the filter only drops flagged entries)
  check('(a) the fixture scene loads whole (16 props)', F.m.engine.E.P.length === 16 && E.P.length === 15);
  // every row the page sends: slot 0 stays parked (no head / paw attach weight)
  let bad = 0; for (let i = 0; i < 30; i++) { const r = T.m.step(1 / 30, {}); if ((r.prop0_hh || 0) + (r.prop0_hl || 0) + (r.prop0_hr || 0) > 0) bad++; }
  check('(a) slot 0 is never attached in the rows', bad === 0, bad);
}

// ---------------------------------------------------------------- (d) asking for the headphones
const matcher = Intent.create(RULES, {scene: JSON.parse(fs.readFileSync(REAL, 'utf8'))});
for (const text of ['put on your headphones', 'pick up the headset', 'take off your headphones', 'grab your headphones and put them on']) {
  const T = makeMuse(4); T.step(5, false); const k0 = T.ev.length;
  let hits = [], rs = [], err = null;
  try { hits = matcher.parse(text, {final: true, stable: true}); rs = hits.map(h => T.m.command(h.cmd, {src: 'matcher'})); } catch (e) { err = e; }
  const started = !idle(T); runIdle(T, 10); T.step(10, false);
  const s = T.m.state();
  check(`(d) matcher "${text}": declined honestly (ok false, "there is no … here"), nothing started, nothing held or worn`,
    // (a pronoun step after it — "put them on" — has nothing to refer to: "which one do you mean?", also ok false)
    !err && rs.length >= 1 && /there (is|are) no (headphones|headset) here/.test(rs[0].reason) && rs.every(r => r.ok === false && /there (is|are) no (headphones|headset) here|which one/.test(r.reason)) && !started && handsEmpty(T) && !s.wearing.length && !T.ev.slice(k0).some(e => e.kind === 'object' && e.phase === 'start'),
    err ? String(err) : rs.map(r => r.ok + ':' + (r.reason || r.did)).join(' | '));
}
{ // the voice model's object tool (page onCall('object') -> runCommand -> motion.command, src voice)
  const T = makeMuse(5); T.step(5, false);
  const r = T.m.command({action: 'object', verb: 'put_on', object: {name: 'headphones'}}, {src: 'voice'});
  const r2 = T.m.command({action: 'object', verb: 'put_on', object: {name: 'pink headphones', color: 'pink'}}, {src: 'voice'});
  runIdle(T, 5);
  check('(d) object tool put_on {headphones} / {pink headphones}: ok false with the reason, nothing moves', r.ok === false && /no headphones/.test(r.reason) && r2.ok === false && /no .*headphones/.test(r2.reason) && handsEmpty(T) && !T.m.state().wearing.length, [r.reason, r2.reason]);
}
// do_action (the voice tools): never planned as a keyframe mime of a thing that is not there
function vtHarness(seed) {
  const T = makeMuse(seed); T.step(5, false); let t = 1000; const perf = [], steps = [], logs = [];
  const M = Intent.create(RULES, {scene: T.files.props});
  const vt = VTL.createVoiceTools({perf: () => ({call: d => { perf.push(d); return {ok: true, status: 'preparing'}; }}), motion: () => T.m, sceneMatcher: () => M,
    runStep: cmd => { const r = T.m.command(cmd, {src: 'caption'}); steps.push({cmd, r}); return r; }, runCommand: cmd => T.m.command(cmd, {src: 'voice'}),
    sendCtl: () => {}, prompt: () => ({}), log: e => logs.push(e), userSpeaking: () => false, now: () => t, timers: {set: () => 0, clear: () => {}}});
  const say = text => { vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText(text, true); t += 400; };
  return {T, vt, perf, steps, say};
}
for (const text of ['put on your headphones', 'pick up the headset']) {
  const H = vtHarness(6); H.say(text); let r = null, err = null;
  try { r = H.vt.call('do_action', {description: text}, 'done', {}); } catch (e) { err = e; }
  runIdle(H.T, 5);
  // (known gap: the do_action result itself still says "doing what he asked" for a refused step — the voice tools do not
  // read runStep's result; generic for any thing that is not in the scene, not asserted here)
  check(`(d) do_action "${text}": no keyframe mime (planner not called), no crash, the step the engine ran was refused, nothing moved`,
    !err && r && H.perf.length === 0 && H.steps.length >= 1 && H.steps.every(s => s.r.ok === false && /there (is|are) no/.test(s.r.reason)) && handsEmpty(H.T) && !H.T.m.state().wearing.length,
    err ? String(err) : {r, perf: H.perf, steps: H.steps.map(s => [s.cmd.verb, s.r.ok, s.r.reason])});
}
{ // the scene-less refusal (noThingHere) is unchanged: a character with no things declines "put on your headphones"
  let t = 1000; const perf = [];
  const vt = VTL.createVoiceTools({perf: () => ({call: d => { perf.push(d); return {ok: true}; }}), motion: () => ({props: () => []}), runCommand: () => ({ok: true}),
    sendCtl: () => {}, prompt: () => ({}), userSpeaking: () => false, now: () => t, timers: {set: () => 0, clear: () => {}}});
  vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText('Put on your headphones.', true); t += 400;
  const r = vt.call('do_action', {description: 'put on your headphones'}, 'done', {});
  check('(d) scene-less character: do_action "put on your headphones" -> no_such_thing (unchanged)', r && r.ok === false && r.status === 'no_such_thing' && !perf.length, r);
}
{ // "put on the hat" still works (matcher -> engine; do_action -> the same steps)
  const T = makeMuse(7); T.step(5, false);
  const rs = matcher.parse('put on the hat', {final: true, stable: true}).map(h => T.m.command(h.cmd, {src: 'matcher'})); runIdle(T, 60); T.step(20, false);
  check('(d) matcher "put on the hat": wearing the hat, paws empty', rs.length === 1 && rs[0].ok && T.m.state().wearing.includes('hat') && handsEmpty(T), rs.map(r => r.did || r.reason));
  const H = vtHarness(8); H.say('Put on the hat.'); const r = H.vt.call('do_action', {description: 'put on the hat'}, 'done', {}); runIdle(H.T, 60); H.T.step(20, false);
  check('(d) do_action "put on the hat": the step runs, wearing it, no planner', r.ok && H.steps.length === 1 && H.steps[0].r.ok && !H.perf.length && H.T.m.state().wearing.includes('hat'), {r, steps: H.steps.map(s => s.r.did || s.r.reason)});
  const r2 = T.m.command(O('take_off', 'hat'), {src: 'matcher'}); runIdle(T, 60);
  const r3 = T.m.command(O('put_back', 'hat'), {src: 'matcher'}); runIdle(T, 60); T.step(10, false);
  check('(d) take the hat off and put it back: home, nothing held, never "stand"', r2.ok && r3.ok && where(T, 'hat') === 'home' && handsEmpty(T) && !/stand/.test(r3.did), [r2.did, r3.did, where(T, 'hat')]);
}

// ---------------------------------------------------------------- (e) the side table is a plain table
{
  const T = makeMuse(9); T.step(5, false);
  const r0 = T.m.command(O('place', 'book', {place: 'floor'}), {src: 'matcher'}); runIdle(T, 60);
  const atDesk = where(T, 'book');
  const rs = matcher.parse('put the book on the side table', {final: true, stable: true}).map(h => T.m.command(h.cmd, {src: 'matcher'})); runIdle(T, 60); T.step(10, false);
  check('(e) "put the book on the side table" (from the floor): on the side table, paws empty', r0.ok && atDesk === 'floor' && rs.length === 1 && rs[0].ok && /^(side_table|home)$/.test(where(T, 'book')) && handsEmpty(T) && !/stand/.test(rs[0].did), [r0.did, atDesk, rs.map(r => r.did || r.reason), where(T, 'book')]);
  const r1 = T.m.command(O('place', 'book', {place: 'desk_top'}), {src: 'matcher'}); runIdle(T, 60);
  const rb = matcher.parse('pick up the book and put it back', {final: true, stable: true}).map(h => T.m.command(h.cmd, {src: 'matcher'})); runIdle(T, 60); T.step(10, false);
  const bk = T.m.engine.E.P.find(p => p.id === 'book');
  check('(e) "put it back" (the book\'s home is the side table): home, said as "back in its spot on the side table", no stand',
    r1.ok && rb.length >= 2 && rb.every(r => r.ok) && where(T, 'book') === 'home' && Math.hypot(...bk.pos.map((v, i) => v - bk.home[i])) < 0.01 && handsEmpty(T),
    [rb.map(r => r.did || r.reason), where(T, 'book')]);
  const ev = T.ev.filter(e => e.kind === 'object' && e.phase === 'end').map(e => e.did).join(' | ');
  check('(e) the action events say "side table", never "stand"', /side table/.test(ev) && !/stand\b/.test(ev.replace(/stand(ing|s up| up)/g, '')), ev.slice(-400));
  const T2 = makeMuse(10); T2.step(5, false);
  const rr = matcher.parse('put the ball on the side table', {final: true, stable: true}).map(h => T2.m.command(h.cmd, {src: 'matcher'})); runIdle(T2, 60); T2.step(10, false);
  const rp = T2.m.command(O('put_back', 'ball'), {src: 'matcher'}); runIdle(T2, 60);
  check('(e) "put the ball on the side table", then back: on the table, then home (floor)', rr.length === 1 && rr[0].ok && rp.ok && where(T2, 'ball') === 'home' && handsEmpty(T2) && !/stand/.test(rp.did), [rr.map(r => r.did || r.reason), rp.did]);
}
{ // the engine's wearable-band + stand code is kept (generic): on the pre-removal scene the headphones still hang back on the stand
  const T = makeMuse(11, {propsPath: HP_FIXTURE}); T.step(5, false);
  const a = T.m.command(O('put_on', 'headphones'), {src: 'test'}); runIdle(T, 60);
  const b = T.m.command(O('take_off', 'headphones'), {src: 'test'}); runIdle(T, 60);
  const c = T.m.command(O('put_back', 'headphones'), {src: 'test'}); runIdle(T, 60);
  check('(e) fixture: headphones on, off, put back -> "back on the stand on the side table"', a.ok && b.ok && c.ok && where(T, 'headphones') === 'home' && /back on the stand/.test(c.did), [a.did, b.did, c.did]);
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails);
