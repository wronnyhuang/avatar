// node tests/intent_test.mjs — runs app/intent.js on the shared vectors (same files as tests/test_intent.py)
import {createRequire} from 'module';
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
// app/ may sit under a package.json with "type":"module"; then the IIFE attaches to globalThis instead of module.exports
const mod = require(path.join(here, '..', 'app', 'intent.js')) || {};
const Intent = mod.Intent || globalThis.Intent, IntentDedupe = mod.IntentDedupe || globalThis.IntentDedupe;
const rules = JSON.parse(fs.readFileSync(path.join(here, '..', 'app', 'intent_rules.json'), 'utf8'));
const vec = JSON.parse(fs.readFileSync(path.join(here, 'intent_vectors.json'), 'utf8'));
const sv = JSON.parse(fs.readFileSync(path.join(here, 'stream_vectors.json'), 'utf8'));
Intent.setDefaultRules(rules);
function vs(v) { return typeof v === 'boolean' ? String(v) : String(v); }
function summAct(c) {   // == tests/check_cases.py summ_act
  const kv = [];
  for (const k of Object.keys(c).sort()) {
    if (['action', 'name', 'verb'].includes(k)) continue;
    const v = c[k];
    if (v && typeof v === 'object') for (const k2 of Object.keys(v).sort()) kv.push((k === 'object' ? 'obj' : k) + '.' + k2 + '=' + vs(v[k2]));
    else kv.push(k + '=' + vs(v));
  }
  return [c.action + '/' + (c.action === 'move' ? c.name : c.verb)].concat(kv).join(' ');
}
function summ(c) {
  const a = c.action;
  if (a === 'move' || a === 'object') return summAct(c);
  if (a === 'release') return 'release/' + (c.parts === 'all' ? 'all' : [...c.parts].sort().join('+'));
  if (a === 'hold') return 'hold/all';
  if (a === 'look') return 'look/' + c.target;
  let s = a + '/' + (c.name || '');
  if (c.side && (a === 'pose' || a === 'gesture')) s += '/' + c.side;
  return s;
}
function deq(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((x, i) => deq(x, b[i]));
  if (a && typeof a === 'object') {
    if (!b || typeof b !== 'object') return false;
    const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
    return deq(ka, kb) && ka.every((k) => deq(a[k], b[k]));
  }
  return a === b;
}
let bad = 0, n = 0;
for (const cs of vec.cases) {
  n++;
  const m = Intent.create(rules);
  const cmds = m.parse(cs.text).map((h) => h.cmd);
  const got = cmds.map(summ);
  if (!deq(got, cs.expect)) { bad++; console.log('SUMMARY MISMATCH', cs.text, cs.expect, got); continue; }
  if (!deq(cmds, cs.cmds)) { bad++; console.log('CMD MISMATCH vs python', cs.text, JSON.stringify(cmds), JSON.stringify(cs.cmds)); }
}
for (const s of sv.streams) {
  n++;
  const st = Intent.create(rules).stream();
  const outs = s.steps.map(([t, fin, ms]) => st.update(t, fin, ms));
  const got = outs.map((o) => o.map((h) => summ(h.cmd)));
  if (!deq(got, s.expect)) { bad++; console.log('STREAM MISMATCH', s.name, JSON.stringify(s.expect), JSON.stringify(got)); }
  if (s.replaces) {
    const rp = outs.map((o) => (o.length && o[0].replaces ? summ(o[0].replaces) : null));
    if (!deq(rp, s.replaces)) { bad++; console.log('REPLACES MISMATCH', s.name, JSON.stringify(s.replaces), JSON.stringify(rp)); }
  }
}
// intent: the act grammar on every phrase + streaming replays, against the Python matcher's output
const av = JSON.parse(fs.readFileSync(path.join(here, 'intent_act_vectors.json'), 'utf8'));
const scene = fs.readdirSync(path.join(here, 'props_mock')).sort().map((f) => JSON.parse(fs.readFileSync(path.join(here, 'props_mock', f), 'utf8')));
const mScene = Intent.create(rules, {scene});
let actBad = 0;
const t0 = process.hrtime.bigint();
for (const cs of av.cases) {
  n++;
  const got = Intent.create(rules, {scene}).parse(cs.text).map((h) => h.cmd);
  if (!deq(got, cs.cmds)) { bad++; if (actBad++ < 15) console.log('ACT MISMATCH vs python', cs.text, JSON.stringify(got), JSON.stringify(cs.cmds)); }
}
let tParse = 0n;
for (const cs of av.cases) { const a = process.hrtime.bigint(); mScene.parse(cs.text); tParse += process.hrtime.bigint() - a; }
for (const s of av.streams) {
  n++;
  const st = Intent.create(rules, {scene}).stream();
  const outs = s.steps.map(([t, fin, ms]) => st.update(t, fin, ms).map((h) => h.cmd));
  if (!deq(outs, s.outs)) { bad++; if (actBad++ < 25) console.log('ACT STREAM MISMATCH', s.text, JSON.stringify(outs), JSON.stringify(s.outs)); }
}
console.log(`act parity: ${av.cases.length} phrases + ${av.streams.length} streams; JS parse ${(Number(tParse) / 1000 / av.cases.length).toFixed(1)} us/utterance`);
// dedupe: matcher hit then the realtime model's function call for the same part ~1 s later
const D = IntentDedupe; D.reset();
const hit = Intent.create(rules).parse('raise your left paw')[0].cmd;
const checks = [
  [D.seen(hit, 1000, 'matcher'), false, 'first matcher hit passes'],
  [D.seen({action: 'pose', targets: {hand_r_y: 1, hand_r_x: 0.2}}, 2100, 'model'), true, 'model pose same paw dropped'],
  [D.seen({action: 'gesture', name: 'wave', side: 'l'}, 2200, 'model'), false, 'model on the other paw passes'],
  [D.seen({action: 'gesture', name: 'point', side: 'l', targets: {point_l: 1}}, 2600, 'model'), false, 'model chaining on its own paw passes'],
  [D.seen({action: 'pose', targets: {hand_r_y: 1}}, 6000, 'model'), false, 'after window passes'],
  [D.seen({action: 'look', target: 'user', hold: true}, 7000, 'matcher'), false, 'look passes'],
  [D.seen({action: 'look', target: 'user', hold: true}, 7500, 'caption'), true, 'caption re-read of the same look dropped'],
  [D.seen({action: 'gesture', name: 'nod'}, 9000, 'matcher'), false, 'nod by name'],
  [D.seen({action: 'gesture', name: 'nod', repeat: 2}, 9500, 'model'), true, 'model nod after matcher nod dropped'],
  [D.seen(hit, 20000, 'matcher'), false, 'matcher again'],
  [D.seen(Object.assign({}, hit, {targets: {hand_r_y: 0.5}}), 20300, 'matcher'), false, 'matcher refinement passes'],
  [D.seen(Object.assign({}, hit, {targets: {hand_r_y: 0.5}}), 20400, 'matcher'), true, 'identical matcher repeat dropped'],
  [D.seen({action: 'release', parts: 'all'}, 30000, 'matcher'), false, 'release passes'],
  [D.seen({action: 'release', parts: 'all'}, 30900, 'model'), true, 'model release after matcher release dropped'],
  [D.seen({action: 'expression', name: 'frown', intensity: 0.8, hold: true}, 40000, 'matcher'), false, 'matcher frown'],
  [D.forget({action: 'expression', name: 'frown', intensity: 0.8, hold: true}, 'matcher'), true, 'motion could not run it: forget'],
  [D.seen({action: 'expression', name: 'sad', intensity: 0.8}, 40900, 'model'), false, 'model expression passes after forget'],
];
// intent: moves/objects + a new utterance (userSpoke)
D.reset();
const spin = {action: 'move', name: 'spin', count: 2}, pick = {action: 'object', verb: 'pick_up', object: {name: 'ball'}};
checks.push(
  [D.seen(spin, 1000, 'matcher'), false, 'matcher spin runs'],
  // intent c2: a model call that DISAGREES (count 3 vs 2) is not dropped: it replaces the matcher's job
  [D.seen({action: 'move', name: 'spin', count: 3}, 2000, 'model'), false, 'c2: model spin count 3 vs matcher count 2 = conflict, replaces'],
  [!!(D.last.conflict && D.last.conflict.source === 'matcher'), true, 'c2: the conflict names the matcher record'],
);
D.reset(); D.seen(spin, 1000, 'matcher');
checks.push(
  [D.seen({action: 'move', name: 'spin'}, 2000, 'model'), true, 'model spin without args = same request: dropped'],
  [D.seen({action: 'move', name: 'twirl'}, 2100, 'model'), true, 'model twirl = same family, same utterance: dropped'],
  [D.seen({action: 'gesture', name: 'spin'}, 2200, 'model'), true, 'legacy gesture spin = same request: dropped'],
  [D.seen({action: 'expression', name: 'happy'}, 2300, 'model'), false, 'model adds a face: runs'],
  [D.seen(pick, 2400, 'matcher'), false, 'matcher pick up runs'],
  [D.seen({action: 'object', verb: 'show', object: {ref: 'it'}}, 2500, 'model'), false, 'model show (the clause the matcher missed): runs'],
  [D.seen({action: 'object', verb: 'pick_up', object: {name: 'ball'}}, 2600, 'model'), true, 'model pick up of the same ball: dropped'],
);
D.reset(); D.seen(spin, 1000, 'matcher'); D.userSpoke(1800);
checks.push(
  [D.seen({action: 'move', name: 'jump'}, 2400, 'model'), false, 'new utterance after the hit: different family runs'],
  [D.seen({action: 'move', name: 'spin'}, 2500, 'model'), true, 'new utterance but the same request as the hit: dropped'],
  [D.seen({action: 'object', verb: 'sit_at', object: {name: 'desk'}}, 2600, 'matcher'), false, 'matcher sit at desk'],
  [D.seen({action: 'move', name: 'sit'}, 2900, 'model'), true, 'model move sit after matcher sit_at: same posture, dropped'],
);
D.reset(); D.seen(pick, 1000, 'matcher'); D.attach({ok: false, reason: 'there is no ball here'});
checks.push([D.seen(pick, 1500, 'model') && D.last.rec.res.reason, 'there is no ball here', 'a dropped duplicate sees the first result']);
for (const [got, want, what] of checks) { n++; if (got !== want) { bad++; console.log('DEDUPE', what, 'want', want, 'got', got); } }
console.log(`${n} checks, ${bad} failures`);
process.exit(bad ? 1 : 0);
