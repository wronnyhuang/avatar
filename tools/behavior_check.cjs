#!/usr/bin/env node
/* tools/behavior_check.cjs — Program layer: the server's static check + dry run for a coder-written program.
 * Persistent: one JSON object per line on stdin → one JSON line on stdout (app/behavior_coder.py keeps one process).
 *   {id, op:'doc', character}                     → {id, ok, toolkit, character_doc}
 *   {id, op:'check', code, character, instruction, gestures?}
 *        → {id, ok, error, errors:[...], fix:[...] (what the coder must fix), notes:[...] (soft), stats, ms}
 *          = compile (tokenize, refused syntax / names, guard instrumentation) + a dry run on the synthetic tour (every
 *          sense cue, taught gestures / heard words the code asks for) and the recorded perception clips (15 fps),
 *          + literal checks: event types, gesture names, face names, touch parts, look targets, controls, groups.
 *   {id, op:'dev', item, code}                    → {id, ok, detail, ...} (tests/behavior_dev.cjs item check, for the bench)
 * One-shot: node tools/behavior_check.cjs --check prog.js [--character fox]
 */
'use strict';
const fs = require('fs'), path = require('path'), readline = require('readline');
const SIM = require('../app/behavior_sim.cjs');
const B = SIM.B;
let DEVSET = null;

const EVENT_TYPES = new Set(['blink', 'wink', 'eyes_closed', 'eyes_open', 'smile_start', 'smile_end', 'mouth_open', 'mouth_close', 'brows_up', 'brows_down',
  'frown', 'nod', 'head_shake', 'head_tilt', 'head_straight', 'head_turn', 'look_away', 'look_back', 'face_appear', 'face_lost', 'lean_in', 'lean_back',
  'speech_start', 'speech_end', 'hand_appear', 'hand_gone', 'hand_up', 'hand_down', 'hand_gesture', 'touch', 'touch_end', 'hands_together',
  'hands_apart', 'wave', 'point', 'thumbs_up', 'object_up', 'approach_fast', 'shrug', 'expression_change', 'gesture', 'said', 'self_speech_start',
  'self_speech_end', 'action_start', 'action_end', 'attention', 'turn_bid']);
const BUILTIN_GESTURES = new Set(['wave', 'nod', 'head_shake', 'shake', 'thumbs_up', 'thumbs_down', 'fist', 'open_palm', 'victory', 'peace', 'i_love_you',
  'point', 'pointing', 'hands_together', 'clap', 'shrug', 'smile', 'blink', 'wink', 'hand_up', 'raise_hand', 'head_tilt', 'look_away']);
const PARTS = new Set(['nose', 'mouth', 'lips', 'lip', 'chin', 'jaw', 'forehead', 'head', 'hair', 'eye', 'eyes', 'right_eye', 'left_eye', 'cheek', 'cheeks',
  'right_cheek', 'left_cheek', 'ear', 'ears', 'right_ear', 'left_ear', 'face']);
const LOOKS = new Set(['user', 'him', 'me', 'camera', 'face', 'you', 'eyes', 'gaze', 'his_gaze', 'where_he_looks', 'where_i_look', 'look', 'same_way',
  'where_he_is_looking', 'left_hand', 'right_hand', 'his_left_hand', 'his_right_hand', 'hand', 'hands', 'his_hands', 'his_hand', 'screen_left', 'screen_right',
  'left', 'right', 'up', 'down', 'ahead', 'forward', 'straight', 'center', 'away', 'my_hands', 'my_hand', 'my_left_hand', 'my_right_hand', 'my_gaze', 'the_user']);
const FACE_OK = name => {
  const k = String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
  const friendly = ['smile', 'happy', 'frown', 'sad', 'surprise', 'surprised', 'angry', 'mouthopen', 'jawopen', 'open', 'browup', 'browsup', 'raisedbrows',
    'eyebrowsup', 'browdown', 'browsdown', 'squint', 'eyewide', 'eyeswide', 'blink', 'eyesclosed', 'blinkleft', 'blinkright', 'lefteyeclosed', 'righteyeclosed',
    'wink', 'pucker', 'kiss', 'cheekpuff', 'puff', 'sneer', 'tongue', 'tongueout'];
  const bs = ['browdownleft', 'browdownright', 'browinnerup', 'browouterupleft', 'browouterupright', 'cheekpuff', 'cheeksquintleft', 'cheeksquintright',
    'eyeblinkleft', 'eyeblinkright', 'eyelookdownleft', 'eyelookdownright', 'eyelookinleft', 'eyelookinright', 'eyelookoutleft', 'eyelookoutright',
    'eyelookupleft', 'eyelookupright', 'eyesquintleft', 'eyesquintright', 'eyewideleft', 'eyewideright', 'jawforward', 'jawleft', 'jawopen', 'jawright',
    'mouthclose', 'mouthdimpleleft', 'mouthdimpleright', 'mouthfrownleft', 'mouthfrownright', 'mouthfunnel', 'mouthleft', 'mouthlowerdownleft',
    'mouthlowerdownright', 'mouthpressleft', 'mouthpressright', 'mouthpucker', 'mouthright', 'mouthrolllower', 'mouthrollupper', 'mouthshruglower',
    'mouthshrugupper', 'mouthsmileleft', 'mouthsmileright', 'mouthstretchleft', 'mouthstretchright', 'mouthupperupleft', 'mouthupperupright',
    'nosesneerleft', 'nosesneerright', 'blinkl', 'blinkr', 'cheekpuff'];
  return friendly.includes(k) || bs.includes(k);
};

// literal arguments of api calls (strings only): api.events('x'), api.events(['x', 'y']), api.recent('x'), api.gesture('x'), ...
function literals(code, fn) {
  const out = [];
  const re = new RegExp('\\b(?:api\\.)?(?:user\\.)?' + fn + '\\(\\s*(\\[[^\\]]{0,400}\\]|[\'"`][^\'"`]{0,60}[\'"`])', 'g');
  for (const m of String(code).matchAll(re)) for (const q of m[1].matchAll(/['"`]([^'"`]{0,60})['"`]/g)) out.push(q[1]);
  return out;
}
function literalChecks(code, character, gestures) {
  const fix = [], notes = [];
  const taught = new Set((gestures || []).map(g => String(g).toLowerCase()));
  for (const t of literals(code, 'events').concat(literals(code, 'recent'))) if (!EVENT_TYPES.has(t)) fix.push(`'${t}' is not an event type (see the event list)`);
  for (const g of literals(code, 'gesture')) { const k = g.toLowerCase().replace(/[\s-]+/g, '_'); if (!BUILTIN_GESTURES.has(k) && !/^touch(ing)?_/.test(k) && !taught.has(k)) notes.push(`api.gesture('${g}') is neither built in nor a gesture he taught${taught.size ? ' (taught: ' + [...taught].join(', ') + ')' : ''}: it stays off until he teaches it`); }
  for (const f of literals(code, 'face')) if (!FACE_OK(f)) fix.push(`api.user.face('${f}') is not a face name`);
  for (const p of literals(code, 'touching').concat(literals(code, 'touching_hand'))) if (!PARTS.has(p.toLowerCase().replace(/[\s-]+/g, '_'))) fix.push(`'${p}' is not a face part for touching()`);
  for (const l of literals(code, 'look')) if (!LOOKS.has(l.toLowerCase().replace(/[\s-]+/g, '_'))) fix.push(`'${l}' is not a look target`);
  for (const c of literals(code, 'set')) if (!B.CH[c]) fix.push(`'${c}' is not a control`); else if (!B.hasChannel(character, c)) notes.push(`${B.TRAITS[character] ? B.TRAITS[character].name : character} has no ${c}`);
  for (const g of literals(code, 'claim')) { const r = B.expandGroups([g]); if (r.bad.length) fix.push(`'${g}' is not a group`); }
  return {fix: [...new Set(fix)], notes: [...new Set(notes)]};
}

function check(req) {
  const t0 = process.hrtime.bigint();
  const character = B.TRAITS[req.character] ? req.character : 'fox';
  const code = String(req.code || '');
  const comp = B.compile(code);
  if (!comp.ok) return {ok: false, error: comp.error, errors: comp.errors.map(e => (e.line ? `line ${e.line}: ` : '') + e.msg), fix: comp.errors.map(e => (e.line ? `line ${e.line}: ` : '') + e.msg), notes: [], stats: null, ms: ms(t0)};
  const lit = literalChecks(code, character, req.gestures);
  const r = SIM.dryRun(code, {character, recorded: req.recorded !== false, fps: req.fps || 15, instruction: req.instruction || ''});
  const fix = lit.fix.slice(), notes = lit.notes.slice();
  for (const e of r.errors) fix.push(e);
  for (const w of r.warnings) {
    if (/has no /.test(w)) notes.push(w);
    else if (/needs api\.claim|unknown|non-number/.test(w)) fix.push(w);
    else notes.push(w);
  }
  const P = r.runtime && r.runtime._perc && r.runtime._perc.P;
  if (P && P.unknownFace) fix.push(`api.user.face('${P.unknownFace}') is not a face name`);
  if (P && P.unknownPart) fix.push(`'${P.unknownPart}' is not a face part for touching()`);
  const s = r.stats || {};
  const acted = (s.sets || 0) + (s.emits || 0) + (s.actions || 0) + (s.styles || 0) > 0 || !!s.done;
  if (!acted && !fix.length) notes.push('it never acted on the test frames (every cue: blinks, smiles, tilts, turns, raised hands, touches, waves, nods, looking away, talking, thumbs-up, open mouth, brows, frown, leaving, leaning in, shrug)');
  if (s.ms_mean !== undefined && s.ms_mean > 0.3) notes.push(`per_frame averages ${s.ms_mean} ms (budget 1 ms)`);
  const uniq = [...new Set(fix)];
  return {ok: !uniq.length, error: uniq.join('; ') || null, errors: r.errors, fix: uniq, notes: [...new Set(notes)], stats: Object.assign({acted, level: comp.level}, s), ms: ms(t0)};
}
const ms = t0 => Math.round(Number(process.hrtime.bigint() - t0) / 1e5) / 10;

function handle(req) {
  try {
    if (req.op === 'doc') return {ok: true, toolkit: B.toolkitDoc(), character_doc: B.characterDoc(B.TRAITS[req.character] ? req.character : 'fox')};
    if (req.op === 'check') return check(req);
    if (req.op === 'dev') {
      DEVSET = DEVSET || require('../tests/behavior_dev.cjs');
      const item = DEVSET.byId(req.item);
      if (!item) return {ok: false, error: 'no such dev item'};
      return DEVSET.runItem(item, String(req.code || ''));
    }
    if (req.op === 'ping') return {ok: true};
    return {ok: false, error: 'unknown op'};
  } catch (e) { return {ok: false, error: 'checker error: ' + String(e && e.stack || e).slice(0, 400)}; }
}

if (require.main === module) {
  const i = process.argv.indexOf('--check');
  if (i > 0) {
    const ci = process.argv.indexOf('--character');
    const r = check({code: fs.readFileSync(process.argv[i + 1], 'utf8'), character: ci > 0 ? process.argv[ci + 1] : 'fox'});
    console.log(JSON.stringify(r, null, 1)); process.exit(r.ok ? 0 : 1);
  }
  SIM.tour(); B.compile('({ per_frame(api) {} })');   // warm the caches
  SIM.dryRun('({ per_frame(api) {} })', {character: 'fox', fps: 15, recorded: true});
  const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
  rl.on('line', line => {
    let req = null;
    try { req = JSON.parse(line); } catch (e) { process.stdout.write(JSON.stringify({ok: false, error: 'bad json'}) + '\n'); return; }
    const res = Object.assign({}, handle(req), {id: req.id});
    process.stdout.write(JSON.stringify(res) + '\n');
  });
  process.stdout.write(JSON.stringify({ready: true}) + '\n');
}
module.exports = {check, literalChecks, handle};
