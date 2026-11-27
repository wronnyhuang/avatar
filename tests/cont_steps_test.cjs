// the second half of a request split across two turns ("put on the hat." / "and sit at the desk") runs its
// own steps from his words (contExpand) — and each step's REAL outcome reaches the voice model, as sceneRoute's result does
// (runSteps / stepsResult): a refused step is said as NOT done (a note asking for a reply, and the model's own do_action
// call of it answers ok:false), an all-done one only informs (no second reply), the model's call of it then says what ran.
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const {makeMuse} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent} = load('app/intent.js');
const VTL = load('app/voicetools.js');
const RULES = JSON.parse(fs.readFileSync(path.join(ROOT, 'app', 'intent_rules.json')));
let passes = 0, fails = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok || detail === undefined ? '' : '  — ' + JSON.stringify(detail).slice(0, 400))); }

function run(first, second, {desc2} = {}) {
  const X = makeMuse(3); X.step(5, false); let t = 1000; const steps = [], prompts = [], perf = [];
  const M = Intent.create(RULES, {scene: X.files.props});
  const runStep = cmd => { const r = X.m.command(cmd, {src: 'caption'}); steps.push([cmd, r]); return r; };
  const vt = VTL.createVoiceTools({perf: () => ({call: dd => { perf.push(dd); return {ok: true, status: 'preparing', id: 'p1'}; }}), motion: () => X.m, sceneMatcher: () => M,
    runStep, runCommand: cmd => X.m.command(cmd, {src: 'voice'}), sendCtl: () => {}, prompt: (n, o) => { prompts.push({n, o: o || {}}); return {}; }, userSpeaking: () => false,
    now: () => t, timers: {set: () => 0, clear: () => {}}});
  const say = text => { vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText(text, true); t += 400; };
  say(first);
  const r1 = vt.call('do_action', {description: first}, 'done', {});
  const n1 = steps.length; t += 600;
  say(second);
  const contSteps = steps.slice(n1), notes = prompts.slice();
  const r2 = vt.call('do_action', {description: desc2 || second.replace(/^and (then )?/, '')}, 'done', {});
  const r3 = vt.call('do_action', {description: desc2 || second.replace(/^and (then )?/, '')}, 'done', {});
  return {r1, r2, r3, contSteps, notes, perf, X};
}
const refused = s => s.filter(([, r]) => r && r.ok === false).length;

{
  const o = run('put on the hat.', 'and put on your headphones');
  check('refused 2nd half: its step ran through the engine and was refused', o.contSteps.length === 1 && refused(o.contSteps) === 1, o.contSteps);
  check('refused 2nd half: a note tells the model it is NOT done, with the reason, and asks for a reply', o.notes.length === 1 && /NOT done/.test(o.notes[0].n) && /no headphones/.test(o.notes[0].n) && o.notes[0].o.respond === true, o.notes);
  check('refused 2nd half: the model\'s do_action of it answers ok:false with the reason (never "under way")', o.r2.ok === false && /no headphones/.test(o.r2.reason || '') && !o.perf.length, o.r2);
  check('refused 2nd half: a second call is still not "under way"', o.r3.ok === false, o.r3);
}
{
  const o = run('put on the hat.', 'and then sit at the desk');
  check('done 2nd half: its step ran', o.contSteps.length >= 1 && refused(o.contSteps) === 0, o.contSteps);
  check('done 2nd half: a note says what each step does, informs only (no second reply)', o.notes.length === 1 && /sit at desk/.test(o.notes[0].n) && o.notes[0].o.respond === false, o.notes);
  check('done 2nd half: the model\'s do_action of it reports the steps (what ran), nothing run twice', o.r2.ok === true && /sit at desk/.test(o.r2.did || '') && !o.perf.length, o.r2);
  check('done 2nd half: its next call is "under way", not a second report', o.r3.ok === true && /under way/.test(o.r3.did || ''), o.r3);
}
{
  const o = run('put on the hat.', 'and then sit at the desk and put on your headphones');
  check('partly refused 2nd half: note says which step is NOT done and asks for a reply', o.notes.length === 1 && /NOT done/.test(o.notes[0].n) && /sit at desk/.test(o.notes[0].n) && o.notes[0].o.respond === true, o.notes);
  check('partly refused 2nd half: do_action of it is partial', o.r2.ok === true && o.r2.partial === true, o.r2);
}
console.log(`${passes} passed, ${fails} failed`);
process.exitCode = fails ? 1 : 0;
