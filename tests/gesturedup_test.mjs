// one dedupe for EVERY tool kind, keyed on the utterance and the body parts the command drives. The fast
// matcher starts a gesture from his words; the voice model's own gesture call for the same request ~1 s later named the
// side in words ('right') and the act its own way ('shake', 'tongue_out', 'brow_raise', 'point') -> different
// IntentDedupe keys -> a second motion. Now: one motion, the tool answers ok:true/already with the non-apologetic note;
// a different body part, a new utterance, a different hand still run; ?dupparts=0 = the old keys.
// Real modules: app/intent.js (matcher + IntentDedupe), app/voicetools.js (look), page.html's runCommand / vtRunCommand /
// toolToCommand / intentToolResult verbatim.  node tests/gesturedup_test.mjs [--print]
import {createRequire} from 'module';
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const {Intent, IntentDedupe} = require(path.join(here, '..', 'app', 'intent.js'));
const VTL = require(path.join(here, '..', 'app', 'voicetools.js'));
const rules = JSON.parse(fs.readFileSync(path.join(here, '..', 'app', 'intent_rules.json'), 'utf8'));
const scene = JSON.parse(fs.readFileSync(path.join(here, 'props_mock', 'muse.json'), 'utf8'));
const PRINT = process.argv.includes('--print');
const html = fs.readFileSync(path.join(here, '..', 'app', 'page.html'), 'utf8');
const a0 = html.indexOf('const POSE_MAP'), a1 = html.indexOf('// intent end');
const r0 = html.indexOf('function vtRunCommand(') >= 0 ? html.indexOf('function vtRunCommand(') : html.indexOf('function runCommand('), r1 = html.indexOf('// Until motion.js lands');
let pass = 0, fail = 0;
const ok = (name, cond, extra) => { if (cond) { pass++; if (PRINT) console.log('PASS', name); } else { fail++; console.log('FAIL', name, extra === undefined ? '' : JSON.stringify(extra)); } };
const SORRY = /\b(sorry|apolog|could ?n[o']t|can ?not|can't|unable|fail)/i;   // guidance may say "never apologize": strip that clause first
const apologetic = (r) => SORRY.test(JSON.stringify(r).replace(/never apologize or say it could not be done/g, '').replace(/never say nothing new started/g, ''));

function world(query, motionFail) {
  if (IntentDedupe.setBodyKeys) IntentDedupe.setBodyKeys(new URLSearchParams(query || '').get('dupparts') !== '0');
  const D = {cmds: [], lastCmd: new Map()}, Q = new URLSearchParams(query || ''), S = {cfg: {character: 'muse'}};
  const sent = []; let clock = 0, nextId = 1;
  D.motion = {command: (c) => { if (motionFail && motionFail(c)) return {ok: false, reason: 'that part cannot move like that'}; sent.push(c); return {ok: true, id: nextId++, did: 'doing ' + (c.name || c.target || c.action)}; }, props: () => []};
  const perf = {now: () => clock};
  const page = new Function('D', 'Q', 'S', 'sel', 'IntentDedupe', html.slice(a0, a1) + '\nreturn {toolToCommand, intentToolResult};')(D, Q, S, 'muse', IntentDedupe);
  const rc = new Function('D', 'window', 'IntentDedupe', 'performance', 'REL_ALIAS', 'legacyCommand', html.slice(r0, r1) +
    '\nreturn {runCommand, vtRunCommand: typeof vtRunCommand === "function" ? vtRunCommand : null};')(D, {IntentDedupe}, IntentDedupe, perf, {}, () => {});
  // the page's wiring of the voice tools' runCommand (page.html createVoiceTools deps), read from the page itself
  const wired = /runCommand: \(cmd, info\) => vtRunCommand\(cmd, info\)/.test(html) && rc.vtRunCommand;
  const vt = VTL.createVoiceTools({motion: () => D.motion, perf: () => null, runCommand: wired ? (c, i) => rc.vtRunCommand(c, i) : (c, i) => rc.runCommand(c, 'voice', i),
    dupOk: () => Q.get('dupok') !== '0', sendCtl: () => {}, prompt: () => ({}), now: perf.now, timers: {set: () => 0, clear: () => {}}});
  const m = Intent.create(rules, {act: true, scene});
  IntentDedupe.reset();
  return {D, page, rc, vt, m, sent, at: (t) => { clock = t; },
    // the model's tool call as the page's onCall routes it: look -> the voice tools (planner on); the others -> toolToCommand
    tool(name, args) { if (name === 'look') return vt.call('look', args, 'done', {});
      const cmd = page.toolToCommand(name, args); if (!cmd) return {ok: false, error: 'unknown tool'}; const ran = rc.runCommand(cmd, 'voice', {}); return page.intentToolResult(ran); }};
}


// [what he says (matcher first), the model's own gesture/pose/... call for the same request]
const CASES = [
  ['wave', 'gesture', {name: 'wave'}], ['wave at me', 'gesture', {name: 'wave', repeat: 3}], ['wave goodbye', 'gesture', {name: 'wave'}],
  ['nod your head', 'gesture', {name: 'nod'}], ['nod yes', 'gesture', {name: 'nod', repeat: 2}],
  ['shake your head', 'gesture', {name: 'shake'}], ['shake your head no', 'gesture', {name: 'shake_head'}],
  ['wink', 'gesture', {name: 'wink'}], ['wink at me', 'gesture', {name: 'wink'}],
  ['give me a thumbs up', 'gesture', {name: 'thumbs_up'}], ['stick out your tongue', 'gesture', {name: 'tongue_out'}],
  ['clap your hands', 'gesture', {name: 'clap'}], ['shrug', 'gesture', {name: 'shrug'}], ['blow a kiss', 'gesture', {name: 'blow_kiss'}],
  ['point at me', 'gesture', {name: 'point'}], ['blink', 'gesture', {name: 'blink'}], ['wiggle your ears', 'gesture', {name: 'ear_wiggle'}],
  ['raise your eyebrows', 'gesture', {name: 'brow_raise'}], ['high five', 'gesture', {name: 'high_five'}], ['bow', 'gesture', {name: 'bow'}],
  ['wave with your left hand', 'gesture', {name: 'wave', side: 'left'}],
  // the other tool kinds through the same keys
  ['raise your right hand', 'pose', {targets: {right_paw_y: 1}, hold: true}], ['look sad', 'expression', {name: 'sad'}],
  ['look to the left', 'look', {target: 'left'}], ['spin around', 'move', {name: 'spin'}],
];
const rows = []; let gdup = 0, gtot = 0;
for (const [phrase, name, args] of CASES) {
  const W = world('');
  const hits = W.m.parse(phrase, {final: true});
  if (!hits.length) { rows.push([phrase, name, 'matcher has no rule']); continue; }
  W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of hits) W.rc.runCommand(h.cmd, 'matcher', {text: phrase});
  const n0 = W.sent.length;
  W.at(2100); const r = W.tool(name, args);
  const one = W.sent.length === n0;
  if (name === 'gesture') { gtot++; if (one) gdup++; }
  rows.push([phrase, name, one ? 'one motion' : 'SECOND MOTION', JSON.stringify(r)]);
  ok(`"${phrase}" then ${name}(${JSON.stringify(args)}): one motion`, one, W.sent.slice(n0));
  ok(`"${phrase}" then ${name}: ok:true, already`, r && r.ok === true && r.already === true, r);
  ok(`"${phrase}" then ${name}: note present, not apologetic`, r && /already/.test(String(r.did || '') + String(r.note || '')) && !apologetic(r), r);
}
ok('>= 10 gestures checked', gtot >= 10, gtot);
// not duplicates: another body part in the same utterance, the other hand, a new utterance, no matcher at all
const runs = (phrase, name, args, gap) => { const W = world(''); W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of W.m.parse(phrase, {final: true})) W.rc.runCommand(h.cmd, 'matcher', {text: phrase});
  if (gap) { W.at(9000); IntentDedupe.userSpoke(9000, 9000); }
  const n0 = W.sent.length; W.at(gap ? 10500 : 2100); const r = W.tool(name, args); return {ran: W.sent.length === n0 + 1, r}; };
for (const [p, n, a, gap, why] of [['wave', 'gesture', {name: 'nod'}, 0, 'another body part'], ['nod your head', 'gesture', {name: 'wave'}, 0, 'another body part'],
  ['wave with your left hand', 'gesture', {name: 'wave', side: 'right'}, 0, 'the other hand'], ['wave', 'gesture', {name: 'wave'}, 1, 'a new utterance'],
  ['wink', 'gesture', {name: 'wink'}, 1, 'a new utterance'], ['look sad', 'gesture', {name: 'wave'}, 0, 'face vs hand']]) {
  const {ran, r} = runs(p, n, a, gap); ok(`"${p}" then ${n} ${JSON.stringify(a)} (${why}): runs`, ran && r.ok === true && !r.already, r);
}
{ const W = world(''); W.at(1000); const n0 = W.sent.length; const r = W.tool('gesture', {name: 'salute'}); ok('a gesture with no matcher record runs', W.sent.length === n0 + 1 && r.ok, r); }
// the switch: ?dupparts=0 = the old keys (the model's wave ran a second time)
{ const W = world('dupparts=0'); W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of W.m.parse('wave', {final: true})) W.rc.runCommand(h.cmd, 'matcher', {});
  const n0 = W.sent.length; W.at(2100); W.tool('gesture', {name: 'wave'}); ok('?dupparts=0 restores the old keys (second wave)', W.sent.length === n0 + 1); }
if (IntentDedupe.setBodyKeys) IntentDedupe.setBodyKeys(true);
if (PRINT || process.argv.includes('--table')) for (const r of rows) console.log(r.join(' | '));
console.log(`gestures deduplicated: ${gdup}/${gtot}`);
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
