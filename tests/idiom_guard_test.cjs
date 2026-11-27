// the absent-thing refusal (noThingHere, with the scene's words for the muse; ?absent=0 = off) never refuses an
// IDIOM ("put on your thinking cap", "put on a brave face", "pick up the pace") — those are body / mood requests the library
// or planner acts out — while a real request for a thing that is not around is still refused, with a scene (the muse) and
// without one (a character with no things: motion.props() empty).
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const {makeMuse} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent} = load('app/intent.js');
const VTL = load('app/voicetools.js');
const RULES = JSON.parse(fs.readFileSync(path.join(ROOT, 'app', 'intent_rules.json')));
let passes = 0, fails = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok || detail === undefined ? '' : '  — ' + JSON.stringify(detail).slice(0, 240))); }

let X = null;
function run(text, scene, guard) {
  let t = 1000; const perf = [];
  if (scene && !X) { X = makeMuse(3); X.step(5, false); }
  const M = scene ? Intent.create(RULES, {scene: X.files.props}) : null;
  const motion = scene ? X.m : {props: () => []};
  const vt = VTL.createVoiceTools({perf: () => ({call: d => { perf.push(d); return {ok: true, status: 'preparing', id: 'p1'}; }}), motion: () => motion, sceneMatcher: scene ? () => M : undefined,
    runStep: scene ? (cmd => ({ok: false, reason: 'test: not run'})) : undefined, runCommand: () => ({ok: true}), sendCtl: () => {}, prompt: () => ({}), userSpeaking: () => false,
    now: () => t, timers: {set: () => 0, clear: () => {}}, idiomGuard: guard === undefined ? undefined : () => guard});
  vt.userSpeech('start'); t += 300; vt.userSpeech('stop'); t += 200; vt.userText(text, true); t += 400;
  return vt.call('do_action', {description: text}, 'done', {});
}
const IDIOMS = ['put on your thinking cap', 'put on a brave face', 'pick up the pace', 'put on a happy face', 'put on a show for me', 'put on a little show',
  'pick up the tempo', 'put your best foot forward', 'take a bow', 'hold your horses', 'put on your game face', 'put on a smile', 'take a breather',
  'pick up the slack', 'hold the fort', 'take the stage', 'put on your dancing shoes', 'take it easy', 'grab a seat', 'take a seat', 'hold that thought',
  'pick up where you left off', 'take a chill pill', 'put on a serious look', 'put on your listening ears', 'take a load off', 'take five', 'hold your head high',
  'put on some swagger', 'grab some shut eye', 'put on a poker face', 'pick up the beat', 'carry a tune', 'take a stand', 'put on airs'];
const ABSENT = ['put on your scarf', 'pick up the umbrella', 'grab the banana', 'put on your gloves', 'wear a crown', 'take off your jacket', 'fetch me the remote',
  'pick up the guitar', 'hold the flashlight', 'drink some coffee', 'eat the apple', 'carry the bucket', 'bring me the pillow', 'put on the rain boots',
  'grab your backpack', 'put on your coat', 'pick up the phone', 'hand me the scissors', 'wear your helmet', 'grab a cookie', 'play with the kite',
  'put on your shoes', 'take off your mittens and your scarf', 'pick up the teddy bear', 'eat a sandwich', 'drink the juice', 'grab the broom',
  'put on your necklace', 'fetch the newspaper', 'bring me a towel'];
const THE_SCENE = ['put on the hat', 'pick up the ball', 'grab the mug', 'put on the sunglasses'];
const res = {idiom_refused: [], absent_missed: []};
for (const scene of [true, false]) {
  const tag = scene ? 'muse (scene)' : 'no scene';
  for (const s of IDIOMS) { const r = run(s, scene); const bad = r.status === 'no_such_thing'; if (bad) res.idiom_refused.push([tag, s]); check(`${tag}: idiom not refused as a missing thing: "${s}"`, !bad, r); }
  for (const s of ABSENT) { const r = run(s, scene); const ok = r.ok === false; if (!ok) res.absent_missed.push([tag, s]); check(`${tag}: a real absent thing is refused: "${s}"`, ok, r); }
}
for (const s of THE_SCENE) { const r = run(s, true); check(`muse: a thing in the scene is never "no such thing": "${s}"`, r.status !== 'no_such_thing', r); }
for (const scene of [true, false]) { const r = run('pick up the slack', scene, false); check(`?idiomguard=0: the old any-noun rule (${scene ? 'muse' : 'no scene'}: "pick up the slack" refused)`, r.status === 'no_such_thing', r); }
if (process.env.SUMMARY) console.log(JSON.stringify(res));
console.log(`${passes} passed, ${fails} failed`);
process.exitCode = fails ? 1 : 0;
