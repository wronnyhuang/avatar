// do_action's result says what each step REALLY did on the real muse engine + scene (the headphones left the
// scene: "put on your headphones" through do_action answered "doing what he asked, 1 step in order" over a refused step).
// all refused -> ok:false with the engine's reason; partly refused -> partial, which ran and which did not; a step already
// running from his words -> "already", never a refusal nor a new success; a thing not in the scene is never planned as a mime.
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const {makeMuse} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent} = load('app/intent.js');
const VTL = load('app/voicetools.js');
const RULES = JSON.parse(fs.readFileSync(path.join(ROOT, 'app', 'intent_rules.json')));
let passes = 0, fails = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok || detail === undefined ? '' : '  — ' + JSON.stringify(detail).slice(0, 300))); }

// pre: commands run (and remembered as already running) before the call, as a fast matcher would; opts.absent: the switch
function run(text, {pre = null, absent} = {}) {
  const X = makeMuse(3); X.step(5, false); let t = 1000; const steps = [], perf = [], prompts = [], seen = new Set();
  const M = Intent.create(RULES, {scene: X.files.props});
  const key = c => JSON.stringify([c.verb, c.object, c.place || c.target, c.name]);
  const runStep = cmd => { if (seen.has(key(cmd))) { steps.push([cmd, 'dup']); return false; }
    const r = X.m.command(cmd, {src: 'caption'}); if (r && r.ok !== false) seen.add(key(cmd)); steps.push([cmd, r]); return r; };
  if (pre) { for (const h of M.parse(pre, {final: true})) runStep(h.cmd); steps.length = 0; }
  const d = {perf: () => ({call: desc => { perf.push(desc); return {ok: true, status: 'preparing', id: 'p1'}; }}), motion: () => X.m, sceneMatcher: () => M,
    runStep, runCommand: cmd => X.m.command(cmd, {src: 'voice'}), sendCtl: () => {}, prompt: n => { prompts.push(n); return {}; }, userSpeaking: () => false,
    now: () => t, timers: {set: () => 0, clear: () => {}}};
  if (absent !== undefined) d.absentThings = () => absent;
  const vt = VTL.createVoiceTools(d);
  vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText(text, true); t += 400;
  const r = vt.call('do_action', {description: text}, 'done', {});
  const r2 = vt.call('do_action', {description: text}, 'done', {});   // the model's second call of the same request
  return {r, r2, steps, perf, X};
}
const refused = s => s.filter(([, r]) => r && r !== 'dup' && r.ok === false).length;
const moved = X => X.m.engine.E.jobs.length || X.m.engine.E.move || X.m.engine.E.chain || X.m.engine.E.queue || X.m.engine.E.oqueue;

for (const [text, thing] of [['put on your headphones', 'headphones'], ['pick up the headset', 'headset'], ['put on your scarf', 'scarf'], ['grab the umbrella', 'umbrella']]) {
  const o = run(text);
  check(`all refused: "${text}" -> ok:false, the engine's reason, nothing planned`, o.r.ok === false && new RegExp('no ' + thing).test(o.r.reason) && !o.perf.length && refused(o.steps) === o.steps.length && o.steps.length >= 1, o.r);
  check(`all refused: "${text}" -> the model's second call is not "under way"`, o.r2.ok === false && !o.perf.length, o.r2);
}
{ // the composed sentence — plural agreement, and no dangling "which one do you mean?" for the absent thing's "them"
  const o = run('grab the headphones and put them on');
  check('all refused, two steps: "there are no headphones here, so nothing moved…" (no clarification question, no "is no headphones")',
    o.r.ok === false && /^there are no headphones here, so nothing moved/.test(o.r.reason) && !/which one|\?/.test(o.r.reason) && !/is no headphones/.test(o.r.reason), o.r);
  const k = run('grab the kite');
  check('singular thing keeps "there is no kite …"', k.r.ok === false && /there is no kite/.test(k.r.reason), k.r);
}
{
  const o = run('put on your headphones and sit at the desk');
  check('partly refused: partial, says which step did not run and why, and that the other runs', o.r.ok === true && o.r.partial === true && /only 1 step of 2/.test(o.r.did) &&
    /put on headphones: NOT done \(there are no headphones here\)/.test(o.r.did) && /sit at desk: walking/.test(o.r.did) && /no headphones/.test(o.r.next), o.r);
  check('partly refused: the present step really runs on the engine', !!moved(o.X));
}
{
  const o = run('put on the hat and then put on your headphones');
  check('partly refused (the refused step last): partial', o.r.partial === true && /put on headphones: NOT done/.test(o.r.did), o.r);
}
{
  const o = run('put on the hat');
  check('all ran: ok, not partial, the step says what it does', o.r.ok === true && !o.r.partial && !o.r.already && /1 step in order: put on hat: walking over to the straw sun hat/.test(o.r.did), o.r);
}
{
  const o = run('put the book on the side table');
  check('a no-op step is reported as such, not as a fresh move', o.r.ok === true && /nothing to do — it already is that way/.test(o.r.did), o.r);
}
{
  const o = run('put on the hat', {pre: 'put on the hat'});
  check('duplicate only: "already", not a refusal, not a new success', o.r.ok === true && o.r.already === true && !o.r.partial, o.r);
}
{
  const o = run('put on your headphones and sit at the desk', {pre: 'sit at the desk'});
  check('duplicate + refused: partial; the duplicate is "already under way", not a refusal', o.r.partial === true && /sit at desk: already under way/.test(o.r.did) && /only 1 step of 2/.test(o.r.did), o.r);
}
{
  const o = run('put on your scarf and your gloves');
  check('a thing not in the scene, words the matcher does not read -> refused, not planned as a mime', o.r.ok === false && o.r.status === 'no_such_thing' && /no scarf/.test(o.r.reason) && !o.perf.length, o.r);
  const off = run('put on your scarf and your gloves', {absent: false});
  check('?absent=0 (absentThings false): the old path (the planner)', off.perf.length >= 1 && off.r.ok !== false, off.r);
}
for (const text of ['pretend to put on a scarf', 'put on a big smile', 'take a deep breath', 'hold up your paw', 'put your hands on your head', 'pick up the pace', 'put on the sunglasses and the hat']) {
  const o = run(text);
  check(`never refused as a missing thing: "${text}"`, o.r.ok !== false && o.r.status !== 'no_such_thing', o.r);
}
console.log(`${passes} passed, ${fails} failed`);
process.exitCode = fails ? 1 : 0;
