// intent: the voice tools' page mapping (page.html toolToCommand -> actToolCommand, intentToolResult) speaks the
// same command language as the fast matcher. node tests/intent_tools_test.mjs [--print]  (exit 1 on a failure)
import {createRequire} from 'module';
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const mod = require(path.join(here, '..', 'app', 'intent.js')) || {};
const Intent = mod.Intent || globalThis.Intent, IntentDedupe = mod.IntentDedupe || globalThis.IntentDedupe;
const rules = JSON.parse(fs.readFileSync(path.join(here, '..', 'app', 'intent_rules.json'), 'utf8'));
const scene = JSON.parse(fs.readFileSync(path.join(here, 'props_mock', 'muse.json'), 'utf8'));
const PRINT = process.argv.includes('--print');

// the page's mapping code, verbatim: from "const POSE_MAP" to "// intent end"
const html = fs.readFileSync(path.join(here, '..', 'app', 'page.html'), 'utf8');
const a0 = html.indexOf('const POSE_MAP'), a1 = html.indexOf('// intent end');
if (a0 < 0 || a1 < 0) { console.log('page.html: mapping block not found'); process.exit(1); }
const D = {}, Q = new URLSearchParams(''), S = {cfg: {character: 'muse'}}, sel = 'muse';
const page = new Function('D', 'Q', 'S', 'sel', 'IntentDedupe', html.slice(a0, a1) +
  '\nreturn {toolToCommand, actToolCommand, intentToolResult, intentAct, intentOpts};')(D, Q, S, sel, IntentDedupe);

const m = Intent.create(rules, {act: true, scene});
let fails = 0, n = 0;
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sortObj = (o) => (o && typeof o === 'object' && !Array.isArray(o)) ? Object.fromEntries(Object.keys(o).sort().map(k => [k, sortObj(o[k])])) : o;
function check(name, got, want) {
  n++;
  if (!eq(sortObj(got), sortObj(want))) { fails++; console.log('FAIL', name, '\n  got ', JSON.stringify(got), '\n  want', JSON.stringify(want)); }
  else if (PRINT) console.log('ok  ', name, JSON.stringify(got));
}

// [tool name, tool args (as the model sends them), the phrase whose matcher command must be identical]
const SAME = [
  ['move', {name: 'spin', count: 3, dir: 'left'}, 'spin around three times to your left'],
  ['move', {name: 'backflip', count: 2}, 'do two backflips'],
  ['move', {name: 'sit'}, 'sit down'],
  ['move', {name: 'dance', style: 'silly'}, 'do a silly dance'],
  ['move', {name: 'turn_back'}, 'turn back to me'],
  ['move', {name: 'cartwheel'}, 'muse can you do a cartwheel'],
  ['object', {verb: 'pick_up', object: {name: 'ball', color: 'red'}}, 'pick up the red ball'],
  ['object', {verb: 'put_back', object: {ref: 'it'}}, 'put it back'],
  ['object', {verb: 'put_on', object: {name: 'headphones'}}, 'put on your headphones'],
  ['object', {verb: 'take_off', object: {name: 'headphones'}}, 'take off your headphones'],
  ['object', {verb: 'pull', object: {name: 'chair'}, target: 'desk'}, 'pull the chair up to the desk'],
  ['object', {verb: 'push', object: {name: 'chair'}, target: 'back'}, 'push the chair back'],
  ['object', {verb: 'slide', object: {name: 'laptop'}, target: 'closer'}, 'slide the laptop closer'],
  ['object', {verb: 'sit_in', object: {name: 'chair'}}, 'sit in the chair'],
  ['object', {verb: 'sit_at', object: {name: 'desk'}}, 'sit at the desk'],
  ['object', {verb: 'open', object: {name: 'laptop'}}, 'open the laptop'],
  ['object', {verb: 'type_on', object: {name: 'laptop'}}, 'type on the laptop'],
];
for (const [tool, args, phrase] of SAME) {
  const hits = m.parse(phrase, {final: true});
  const want = hits.length === 1 ? hits[0].cmd : {phrase_hits: hits.map(h => h.cmd)};
  check(`${tool}${JSON.stringify(args)} == "${phrase}"`, page.toolToCommand(tool, args), want);
}

// normalisation of sloppy model arguments
check('count clamps + string numbers', page.actToolCommand('move', {name: 'jump', count: '25', height: 3}), {action: 'move', name: 'jump', count: 10, height: 1});
check('object as a bare string', page.actToolCommand('object', {verb: 'pick_up', object: 'laptop'}), {action: 'object', verb: 'pick_up', object: {name: 'laptop'}});
check('ref that -> it', page.actToolCommand('object', {verb: 'drop', object: {ref: 'that'}}), {action: 'object', verb: 'drop', object: {ref: 'it'}});
check('target "the desk" -> {name}', page.actToolCommand('object', {verb: 'bring', object: {name: 'chair'}, target: 'the desk'}), {action: 'object', verb: 'bring', object: {name: 'chair'}, target: {name: 'desk'}});
check('place + side', page.actToolCommand('object', {verb: 'place', object: {name: 'mug'}, place: 'the desk', side: 'left'}), {action: 'object', verb: 'place', object: {name: 'mug'}, place: 'desk', side: 'left'});
check('empty name -> null', page.actToolCommand('move', {}), null);
check('legacy tools untouched', page.toolToCommand('gesture', {name: 'wave'}), {action: 'gesture', name: 'wave', side: 'right', side_frame: 'character', repeat: 1, hold: false});

// tool output = motion.command()'s result
D.intentRes = {ok: true, id: 4, did: 'spinning three times to its left'};
check('result ok', page.intentToolResult(true), {ok: true, did: 'spinning three times to its left'});
D.intentRes = {ok: false, id: 5, reason: 'there is no red one here'};
check('result refusal', page.intentToolResult(false), {ok: false, reason: 'there is no red one here'});
D.intentRes = {dup: true, first: {ok: true, did: 'picking up the red ball'}};
check('duplicate answers with the first', page.intentToolResult(false), {ok: true, already: true, did: 'picking up the red ball'});
D.intentRes = null;
check('fallback bool', page.intentToolResult(true), {ok: true});
check('fallback bool false', page.intentToolResult(false), {ok: false});

// page matcher options: the muse gets the act grammar + its scene; another character the matcher
S.cfg = {character: 'muse', props: scene};
check('muse page matcher', [page.intentAct(), !!page.intentOpts().scene], [true, true]);
S.cfg = {character: 'fox'};
check('fox page matcher', page.intentOpts(), {act: false, scene: null});

// dedupe: the matcher's command and the model's tool call for the same request = one action; a new utterance passes
IntentDedupe.reset();
const mcmd = m.parse('spin around three times', {final: true})[0].cmd;
check('matcher spin runs', IntentDedupe.seen(mcmd, 1000, 'matcher'), false);
IntentDedupe.attach({ok: true, did: 'spinning three times'});
check('model twirl after it = dup', IntentDedupe.seen(page.toolToCommand('move', {name: 'twirl', count: 3}), 2100, 'model'), true);
check('dup carries the first result', IntentDedupe.last.rec.res, {ok: true, did: 'spinning three times'});
IntentDedupe.userSpoke(2300);
check('new utterance: model jump passes', IntentDedupe.seen(page.toolToCommand('move', {name: 'jump'}), 2600, 'model'), false);
check('new utterance: model spin again = same request, dup', IntentDedupe.seen(page.toolToCommand('move', {name: 'spin'}), 2700, 'model'), true);

// intent c2: the model's call that DISAGREES with the matcher's command for the same utterance replaces it
// (not dropped as a duplicate); a model call that agrees stays deduped; the model is never overridden by a later read
const mc = (p) => m.parse(p, {final: true})[0].cmd;
IntentDedupe.reset();
check('matcher frontflip runs', IntentDedupe.seen(mc('do a front flip'), 1000, 'matcher'), false);
IntentDedupe.attach({ok: true, id: 7, did: 'doing a front flip'});
check('model backflip vs matcher frontflip = conflict, not dup', IntentDedupe.seen(page.toolToCommand('move', {name: 'backflip'}), 1900, 'model'), false);
check('conflict names the replaced matcher job', IntentDedupe.last.conflict && IntentDedupe.last.conflict.res, {ok: true, id: 7, did: 'doing a front flip'});
check('model backflip again = dup', IntentDedupe.seen(page.toolToCommand('move', {name: 'backflip'}), 2100, 'model'), true);
check('caption frontflip after the model = dup (model wins)', IntentDedupe.seen(mc('do a front flip'), 2300, 'caption'), true);
IntentDedupe.reset();
IntentDedupe.seen(mc('spin around twice'), 1000, 'matcher');
check('model twirl without count agrees = dup', IntentDedupe.seen(page.toolToCommand('move', {name: 'twirl'}), 1800, 'model'), true);
check('model spin count 3 vs count 2 = conflict', IntentDedupe.seen(page.toolToCommand('move', {name: 'spin', count: 3}), 1900, 'model'), false);
IntentDedupe.reset();
IntentDedupe.seen(mc('pick up the yellow ball'), 1000, 'matcher');
check('model pick_up ball (no colour) agrees = dup', IntentDedupe.seen(page.toolToCommand('object', {verb: 'pick_up', object: {name: 'ball'}}), 1800, 'model'), true);
check('model pick_up it agrees = dup', IntentDedupe.seen(page.toolToCommand('object', {verb: 'pick_up', object: {ref: 'it'}}), 1850, 'model'), true);
check('model pick_up book vs ball = conflict', IntentDedupe.seen(page.toolToCommand('object', {verb: 'pick_up', object: {name: 'book'}}), 1900, 'model'), false);
IntentDedupe.reset();
IntentDedupe.seen(mc('sit down'), 1000, 'matcher');
check('model sit_in chair vs matcher sit = conflict', IntentDedupe.seen(page.toolToCommand('object', {verb: 'sit_in', object: {name: 'chair'}}), 1800, 'model'), false);
IntentDedupe.reset();
IntentDedupe.seen(mc('pull the chair over here'), 1000, 'matcher');
check('model pull chair target desk vs me = conflict', IntentDedupe.seen(page.toolToCommand('object', {verb: 'pull', object: {name: 'chair'}, target: 'desk'}), 1800, 'model'), false);
IntentDedupe.reset();
IntentDedupe.seen(mc('pull the chair over here'), 1000, 'matcher');
check('model pull chair (no target) agrees = dup', IntentDedupe.seen(page.toolToCommand('object', {verb: 'pull', object: {name: 'chair'}}), 1800, 'model'), true);

// the page's runCommand (verbatim from page.html) with a stub motion engine: the conflicting model call reaches
// motion.command with replaces = the matcher job's id; an agreeing call never reaches it and answers with the first result
const r0 = html.indexOf('function runCommand('), r1 = html.indexOf('// Until motion.js lands');
const sent = [];
let clock = 0, nextId = 1;
const D2 = {cmds: [], lastCmd: new Map(), motion: {command: (c) => { sent.push(c); return {ok: true, id: nextId++, did: 'doing ' + (c.name || c.verb)}; }}};
const run = new Function('D', 'window', 'IntentDedupe', 'performance', 'REL_ALIAS', 'legacyCommand', html.slice(r0, r1) + '\nreturn runCommand;')(
  D2, {IntentDedupe}, IntentDedupe, {now: () => clock}, {}, () => {});
IntentDedupe.reset();
clock = 1000; const ra = run(mc('do a front flip'), 'matcher', {});
clock = 1900; const rb = run(page.toolToCommand('move', {name: 'backflip'}), 'voice', {});
check('runCommand: conflicting model call is sent as a replacement', sent.map(c => [c.name, c.replaces]), [['frontflip', undefined], ['backflip', ra.id]]);
check('runCommand: conflict logged', D2.cmds.some(x => x.conflict && x.replaces === ra.id), true);
clock = 2100; const rc = run(page.toolToCommand('move', {name: 'backflip'}), 'voice', {});
check('runCommand: the agreeing repeat is dropped', [rc, sent.length], [false, 2]);
D.intentRes = D2.intentRes;
check('runCommand: dropped repeat answers with what the replacement is doing', page.intentToolResult(rc), {ok: true, already: true, did: 'doing backflip'});

console.log(`${n} checks, ${fails} failures`);
process.exit(fails ? 1 : 0);
