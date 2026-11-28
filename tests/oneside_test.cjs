// ONE side of a paired part. The fast matcher's rules collapsed "close one eye" into both eyes: the
// quantifier 'one' was read as a possessive (class poss, like 'your') and a rule whose default side is 'both' ignored it.
// app/intent.js oneSide / app/intent.py _one_side resolve 'one' / 'an' / 'either' / 'single' ->
// one side (the rule's own one-sided default, else 'l' = the character's own right), 'other' -> the opposite, a named
// side wins, plural / 'both' -> both. Real matcher + REAL motion controller + move engine: the commanded side moves, the
// other side stays near rest, and after the short hold both sides are back (the pose-revert rule).
// ?oneside=0 / create(.., {oneSide:false}) / AVATAR_INTENT_ONESIDE=0 = the old sides.   node tests/oneside_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const WT = path.join(__dirname, '..');
function load(rel) { const src = fs.readFileSync(path.join(WT, rel), 'utf8'); const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; }
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const {Intent} = load('app/intent.js');
const RULES = JSON.parse(fs.readFileSync(path.join(WT, 'app/intent_rules.json'), 'utf8'));
const ANIM = path.join(WT, 'renderer', 'animals');
const MUSE = {props: JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), body: JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json')))};
let pass = 0, fail = 0;
const ok = (n, c, d) => { if (c) pass++; else { fail++; console.log('FAIL', n, d === undefined ? '' : JSON.stringify(d)); } };
const said = (t, o) => Intent.create(RULES, Object.assign({act: false}, o || {})).parse(t, {final: true, nowMs: 0}).map(h => h.cmd);
// [his words, the dial pair, the side that must move ('l' = own right, 'r' = own left, 'both')]
const CASES = [
  ['close one eye', 'blink', 'l'], ['shut one eye', 'blink', 'l'], ['close an eye', 'blink', 'l'], ['close your left eye', 'blink', 'r'],
  ['close your right eye', 'blink', 'l'], ['close your eyes', 'blink', 'both'], ['wink', null, 'l'], ['wink your left eye', null, 'r'],
  ['lift one eyebrow', 'brow', 'l'], ['raise one eyebrow', 'brow', 'l'], ['lift an eyebrow', 'brow', 'l'], ['raise your left eyebrow', 'brow', 'r'],
  ['raise your eyebrows', 'brow', 'both'], ['raise both eyebrows', 'brow', 'both'],
  ['wiggle one ear', 'ear', 'l'], ['perk up one ear', 'ear', 'l'], ['raise your left ear', 'ear', 'r'], ['wiggle your ears', 'ear', 'both'],
  ['raise one hand', 'hand', 'l'], ['raise your other hand', 'hand', 'r'], ['raise one paw', 'hand', 'l'], ['raise both hands', 'hand', 'both'],
];
for (const [t, , side] of CASES) {
  const c = said(t);
  ok(`"${t}" -> side ${side}`, c.length === 1 && c[0].side === side, c);
}
// the switch
ok('oneSide:false restores both eyes for "close one eye"', (said('close one eye', {oneSide: false})[0] || {}).side === 'both');

// the real motion engine: the asked side moves, the other side does not, and it reverts after the short hold
const DT = 1 / 30;
function run(ch, cmds) {
  const m = M.createMotion({seed: 7, preset: 'default'});
  if (ch === 'muse') m.setProps(JSON.parse(JSON.stringify(MUSE.props)), JSON.parse(JSON.stringify(MUSE.body)), 'muse'); else m.setProps(null, null, ch);
  const face = {present: true, x: 0.05, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
  const rows = [];
  for (const c of cmds) m.command(c, {src: 'matcher'});
  for (let i = 0; i < 16 * 30; i++) rows.push(Object.assign({t: i * DT}, m.step(DT, {features: {t: m.now(), face, attending: true, hands: []}})));
  return rows;
}
const dial = (k, s) => k === 'hand' ? `hand_${s}_y` : `${k}_${s}`;
const mx = (rows, k, a, b) => Math.max(...rows.filter(r => r.t >= a && r.t < b).map(r => +r[k] || 0));
const med = (rows, k, a, b) => { const v = rows.filter(r => r.t >= a && r.t < b).map(r => +r[k] || 0).sort((x, y) => x - y); return v[v.length >> 1]; };
const out = [];
for (const ch of ['muse', 'fox', 'bonbon']) {
  for (const [t, k, side] of CASES.filter(c => c[1] && c[2] !== 'both' && c[1] !== 'hand' || (c[1] === 'hand' && c[2] !== 'both'))) {
    const rows = run(ch, said(t)), o = side === 'l' ? 'r' : 'l';
    const on = mx(rows, dial(k, side), 0.4, 1.6), off = med(rows, dial(k, o), 0.4, 1.6), late = med(rows, dial(k, side), 13, 16), lateIdle = med(run(ch, []), dial(k, side), 13, 16);
    out.push(`${ch} | ${t} | ${dial(k, side)} peak ${on.toFixed(2)} | ${dial(k, o)} ${off.toFixed(2)} | after hold ${late.toFixed(2)} (idle ${lateIdle.toFixed(2)})`);
    ok(`${ch} "${t}": ${dial(k, side)} moves`, on > 0.5, on);
    ok(`${ch} "${t}": ${dial(k, o)} stays near rest`, off < 0.35 && off < on - 0.3, {on, off});
    ok(`${ch} "${t}": back to rest after the hold`, Math.abs(late - lateIdle) < 0.15, {late, lateIdle});
  }
  // the wink gesture: one eye's lid only
  for (const [t, side] of [['wink', 'l'], ['wink your left eye', 'r']]) {
    const rows = run(ch, said(t)), o = side === 'l' ? 'r' : 'l';
    const on = mx(rows, 'blink_' + side, 0, 1.2), off = mx(rows, 'blink_' + o, 0.05, 0.6);
    out.push(`${ch} | ${t} | blink_${side} peak ${on.toFixed(2)} | blink_${o} ${off.toFixed(2)}`);
    ok(`${ch} "${t}": one lid`, on > 0.6 && off < on - 0.3, {on, off});
  }
}
if (process.argv.includes('--table')) console.log(out.join('\n'));
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
