// his words start a command through the fast matcher; the voice model's own tool call for the SAME
// request is dropped as a duplicate (IntentDedupe) — its tool result must be success-shaped (ok:true, already, guidance
// that never apologizes), never ok:false; no second motion; a genuinely failed tool still answers ok:false.
// Real modules: app/intent.js (matcher + IntentDedupe), app/voicetools.js (look), page.html's runCommand / vtRunCommand /
// toolToCommand / intentToolResult verbatim.  node tests/lookdup_test.mjs [--print]
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

// [what he says (the fast matcher starts it), the model's own tool call for the same request]
const CASES = [
  ['look to the left', 'look', {target: 'left'}],
  ['look to the right', 'look', {target: 'right'}],
  ['look up', 'look', {target: 'up'}],
  ['look down', 'look', {target: 'down'}],
  ['look away', 'look', {target: 'away'}],
  ['look at me', 'look', {target: 'user'}],
  ['raise your right hand', 'pose', {targets: {right_paw_y: 1}, hold: true}],
  ['open your mouth', 'pose', {targets: {mouth_open: 0.7}, hold: true}],
  ['shrug', 'pose', {targets: {right_shoulder: 1, left_shoulder: 1}}],
  ['smile', 'expression', {name: 'happy'}],
  ['look sad', 'expression', {name: 'sad'}],
  ['wave', 'gesture', {name: 'wave'}],
  ['nod your head', 'gesture', {name: 'nod'}],
  ['wink', 'gesture', {name: 'wink'}],
  ['give me a thumbs up', 'gesture', {name: 'thumbs_up'}],
  ['look surprised', 'expression', {name: 'surprised'}],
  ['make a happy face', 'expression', {name: 'happy'}],
  ['spin around', 'move', {name: 'spin'}],
  ['sit down', 'move', {name: 'sit'}],
  ['stick out your tongue', 'gesture', {name: 'tongue_out'}],
];
const rows = [];
for (const [phrase, name, args] of CASES) {
  const W = world('');
  const hits = W.m.parse(phrase, {final: true});
  if (!hits.length) { rows.push([phrase, name, 'matcher has no rule', '']); continue; }
  W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of hits) W.rc.runCommand(h.cmd, 'matcher', {text: phrase});
  const n0 = W.sent.length;
  W.at(2100); const r = W.tool(name, args);
  const dup = W.sent.length === n0;
  rows.push([phrase, name, dup ? 'dup' : 'ran (model version differs from the matcher: its own motion)', JSON.stringify(r)]);
  if (!dup) continue;                                    // not this class: the model's call ran as its own command
  ok(`"${phrase}" then ${name}: ok:true`, r && r.ok === true, r);
  ok(`"${phrase}" then ${name}: already`, r && r.already === true, r);
  ok(`"${phrase}" then ${name}: guidance present, not apologetic`, r && /already/.test(String(r.did || '') + String(r.note || '')) && !apologetic(r), r);
  ok(`"${phrase}" then ${name}: no second motion`, W.sent.length === n0);
}
const dupKinds = rows.filter(r => r[2] === 'dup');
ok('at least 8 matcher-first-then-tool cases are deduplicated and checked', dupKinds.length >= 8, dupKinds.length);
ok('the cases cover look, pose, expression and move', ['look', 'pose', 'expression', 'move'].every(k => dupKinds.some(r => r[1] === k)), dupKinds.map(r => r[1]));

// a genuinely failed tool is still ok:false
{ const W = world(''); W.at(1000);
  const r = W.tool('look', {target: 'the purple elephant'});
  ok('look at a thing that is not there: ok:false', r && r.ok === false && /nothing called/.test(r.reason || ''), r); }
{ const W = world('', (c) => c.action === 'look'); W.at(1000);
  const r = W.tool('look', {target: 'left'});
  ok('look the engine refuses (no matcher first): ok:false', r && r.ok === false && !r.already, r); }
{ const W = world('', (c) => c.action === 'look'); W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of W.m.parse('look to the left', {final: true})) W.rc.runCommand(h.cmd, 'matcher', {});
  W.at(2100); const r = W.tool('look', {target: 'left'});
  ok('matcher look failed, then the tool fails too: ok:false (never a false "already")', r && r.ok === false && !r.already, r); }
{ const W = world('', (c) => c.action === 'gesture'); W.at(1000);
  const r = W.tool('gesture', {name: 'wave'});
  ok('gesture the engine refuses: ok:false', r && r.ok === false, r); }
// a new request (he spoke again) is not "already": it runs
{ const W = world(''); W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of W.m.parse('look to the left', {final: true})) W.rc.runCommand(h.cmd, 'matcher', {});
  W.at(9000); IntentDedupe.userSpoke(9000, 9000); W.at(10500);
  const n0 = W.sent.length, r = W.tool('look', {target: 'left'});
  ok('a later utterance\'s look runs as a new command', r && r.ok === true && !r.already && W.sent.length === n0 + 1, r); }
// the switch: ?dupok=0 = the old answer
{ const W = world('dupok=0'); W.at(1000); IntentDedupe.userSpoke(900, 900);
  for (const h of W.m.parse('look to the left', {final: true})) W.rc.runCommand(h.cmd, 'matcher', {});
  W.at(2100); const r = W.tool('look', {target: 'left'});
  ok('?dupok=0 restores the old look answer', r && r.ok === false, r); }

if (PRINT || process.argv.includes('--table')) for (const r of rows) console.log(r.join(' | '));
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
