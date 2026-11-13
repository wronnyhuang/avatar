// node tests/muse_regressions_test.cjs — regressions (B2–B7, B9–B12): pronouns across one request, an explicit paw,
// replacement never leaves a job in limbo, scene-aware dedupe, full circles, count refinements, the reverse-burst stall,
// furniture slide, unknown place → own spot, seated fetch. Exit code = failures.
'use strict';
const fs = require('fs'), path = require('path');
const {makeMuse, museFiles} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent, IntentDedupe} = load('app/intent.js');
let fails = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const O = (verb, o, x) => Object.assign({action: 'object', verb, object: typeof o === 'string' ? {id: o} : o}, x || {});
const props = museFiles().props;
const matcher = Intent.create(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'app', 'intent_rules.json'))), {scene: props});
const idle = T => { const E = T.m.engine.E; return !E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue; };
const runIdle = (T, s = 60) => { for (let i = 0; i < s * 30; i++) { T.step(1, false); if (idle(T) && i > 3) return true; } return false; };
const where = (T, id) => (T.m.state().props.find(p => p.id === id) || {}).where;
const winding = (T, from) => { let acc = 0; for (let i = from + 1; i < T.rows.length; i++) { let d = (T.rows[i].root_yaw || 0) - (T.rows[i - 1].root_yaw || 0); d -= Math.round(d); acc += d; } return acc; };
const say = (T, text) => matcher.parse(text, {final: true, stable: true}).map(h => T.m.command(h.cmd, {src: 'matcher'}));

// ---- B2: pronouns bind to the step before them (my own phrasings)
for (const [pre, text, check2] of [
  [[O('put_on', 'headphones')], 'please take the headphones off and then put them back where they go', T => where(T, 'headphones') === 'home' && !T.m.state().wearing.length],
  [[O('put_on', 'headphones')], 'remove the headphones and put it away', T => where(T, 'headphones') === 'home'],
  [[], 'grab the yellow ball and set it down', T => where(T, 'ball') === 'floor' && !T.m.state().holding.left && !T.m.state().holding.right],
  [[O('pick_up', 'book')], 'grab the mug and then put it on the desk', T => /desk/.test(where(T, 'mug')) && T.m.state().holding.left !== 'mug' && T.m.state().holding.right !== 'mug'],
]) {
  const T = makeMuse(7); T.step(5, false); for (const c of pre) { T.m.command(c); runIdle(T, 30); }
  const rs = say(T, text); runIdle(T, 60);
  check(`B2 "${text}": every step accepted, the pronoun means the thing named before it`, rs.length >= 2 && rs.every(r => r.ok) && check2(T), rs.map(r => r.ok + ':' + String(r.did || r.reason).slice(0, 60)).join(' | ') + ' | ' + JSON.stringify(T.m.state().props.filter(p => ['headphones', 'ball', 'mug'].includes(p.id)).map(p => p.id + ':' + p.where)));
}
{ // the voice model's two calls in sequence with ref 'it' while the first is still queued
  const T = makeMuse(8); T.step(5, false);
  const a = T.m.command(O('pick_up', {name: 'plant'}), {src: 'voice'}), b = T.m.command(O('put_back', {ref: 'it'}), {src: 'voice'}); runIdle(T, 60);
  check('B2 model calls pick_up plant → put_back {ref:it} (queued): binds to the plant', a.ok && b.ok && where(T, 'plant') === 'home' && !T.m.state().holding.left && !T.m.state().holding.right, (b.did || b.reason));
}
{ const ps = matcher.parse('take off the headphones and put them down', {final: true}).map(h => h.cmd);
  check('B2 matcher: "them" after one named thing is {ref:it}, not {ref:all}', ps.length === 2 && JSON.stringify(ps[1].object) === '{"ref":"it"}', JSON.stringify(ps));
  const pa = matcher.parse('put them all back', {final: true}).map(h => h.cmd);
  check('B2 matcher: an explicit "all" stays all', pa.length === 1 && pa[0].object.ref === 'all', JSON.stringify(pa)); }

// ---- B3: an explicit paw is honored (character's own frame), two-pawed props unaffected
for (const id of ['ball', 'headphones', 'plant']) for (const side of ['left', 'right']) {
  const T = makeMuse(9); T.step(5, false); const r = T.m.command(O('pick_up', id, {side})); runIdle(T, 30);
  check(`B3 pick up the ${id} with the ${side} paw → state().holding.${side}`, r.ok && T.m.state().holding[side] === id, JSON.stringify(T.m.state().holding));
}
{ const T = makeMuse(9); T.step(5, false); T.m.command(O('pick_up', 'laptop', {side: 'right'})); runIdle(T, 30);
  const h = T.m.state().holding; check('B3 a two-pawed prop with a side: still both paws', h.left === 'laptop' && h.right === 'laptop', JSON.stringify(h)); }
{ const T = makeMuse(9); T.step(5, false); T.m.command(O('put_on', 'headphones')); runIdle(T, 30); T.m.command(O('take_off', 'headphones', {side: 'right'})); runIdle(T, 30);
  check('B3 take off the headphones with the right paw', T.m.state().holding.right === 'headphones', JSON.stringify(T.m.state().holding)); }
{ const T = makeMuse(9); T.step(5, false); T.m.command(O('pick_up', 'ball', {side: 'right'})); runIdle(T, 30); const r = T.m.command(O('pick_up', 'mug', {side: 'right'})); runIdle(T, 30);
  check('B3 asked paw busy: the other paw, said truthfully', r.ok && /right paw is holding something/.test(r.did) && T.m.state().holding.left === 'mug', r.did); }

// ---- B4: a replaced in-progress object job always ends with the replacement executed
for (const [pre, first, second, delay] of [
  [O('pick_up', 'headphones'), O('put_back', {name: 'headset'}), O('put_back', {name: 'headphones'}), 0.95],
  [O('pick_up', 'headphones'), O('put_back', {name: 'headphones'}), O('put_back', {name: 'headphones'}, {place: 'stand'}), 0.95],
  [O('pick_up', 'ball'), O('put_back', {name: 'ball'}), O('put_back', {name: 'yellow ball'}), 1.1],
  [O('pick_up', 'mug'), O('place', {name: 'mug'}, {place: 'desk'}), O('place', {name: 'mug'}, {place: 'floor'}), 0.8],
]) {
  const T = makeMuse(10); T.step(5, false); T.m.command(pre); runIdle(T, 30); const k0 = T.ev.length;
  const a = T.m.command(first); T.step(Math.round(delay * 30), false); const b = T.m.command(Object.assign({}, second, {replaces: a.id})); const done = runIdle(T, 60);
  const ends = T.ev.slice(k0).filter(e => e.kind === 'object' && /end|interrupted|fail/.test(e.phase));
  const id2 = b.replaced !== undefined ? b.id : a.id, h = T.m.state().holding;
  check(`B4 ${first.verb} ${JSON.stringify(first.object)} replaced by ${JSON.stringify(second.object)}${second.place ? ' place ' + second.place : ''}: queue drains, the replacement ran, paws empty`,
    done && ends.some(e => e.id === id2 && e.phase === 'end') && !h.left && !h.right, ends.map(e => e.id + ':' + e.phase).join(',') + ' ' + JSON.stringify(h));
}

// ---- B5: same thing under two names is not a disagreement
IntentDedupe.setScene(props);
check('B5 headset vs pink headphones agree; computer vs laptop agree; ball vs mug disagree',
  IntentDedupe.agree(O('put_back', {name: 'headset'}), O('put_back', {name: 'pink headphones'})) && IntentDedupe.agree(O('pick_up', {name: 'computer'}), O('pick_up', {name: 'laptop'})) &&
  !IntentDedupe.agree(O('pick_up', {name: 'ball'}), O('pick_up', {name: 'mug'})) && IntentDedupe.agree(O('place', 'ball', {place: 'side_table'}), O('place', 'ball', {place: 'side table'})));
{ IntentDedupe.reset(); IntentDedupe.userSpoke(1000);
  IntentDedupe.seen(O('put_back', {name: 'headset'}), 1100, 'matcher'); IntentDedupe.attach({ok: true, id: 3});
  const dup = IntentDedupe.seen(O('put_back', {name: 'headphones'}), 2000, 'model');
  check('B5 the model\'s "headphones" after the matcher\'s "headset": deduped, no replacement', dup === true, JSON.stringify(IntentDedupe.last && IntentDedupe.last.conflict ? 'conflict' : 'ok')); }

// ---- B6 / B7: full circles and count refinements (root yaw winding)
for (const [cmds, want, label] of [
  [[[{action: 'move', name: 'turn', degrees: 360, dir: 'cw'}, 0]], -1, 'turn degrees 360 cw = one full circle'],
  [[[{action: 'move', name: 'spin', degrees: 720}, 0]], 2, 'spin degrees 720 = two'],
  [[[{action: 'move', name: 'turn', count: 2, dir: 'left'}, 0]], 2, 'turn x2 = two spins'],
  [[[{action: 'move', name: 'spin'}, 0.5], [{action: 'move', name: 'spin', count: 2, replaces: 'prev'}, 0]], 2, 'B7 spin → refined "spin twice" at 0.5 s = two turns'],
  [[[{action: 'move', name: 'spin'}, 0.5], [{action: 'move', name: 'spin', count: 3, replaces: 'prev'}, 0]], 3, 'B7 spin → refined count 3 = three turns'],
]) {
  const T = makeMuse(12); T.step(10); const n0 = T.rows.length; let prev = null; const rs = [];
  for (const [c, d] of cmds) { const cc = Object.assign({}, c); if (cc.replaces === 'prev') cc.replaces = prev; const r = T.m.command(cc); rs.push(r); prev = r.id; T.step(Math.round(d * 30)); }
  T.runUntilIdle(30); const w = winding(T, n0);
  check(`B6/B7 ${label}`, Math.abs(w - want) < 0.1 && rs.every(r => r.ok), `winding ${w.toFixed(2)} turns; ${rs.map(r => r.did).join(' | ')}`);
}
{ const T = makeMuse(12); T.step(10); const n0 = T.rows.length; T.m.command({action: 'move', name: 'turn', degrees: 180}); T.runUntilIdle(20);
  let mn = 0, acc = 0; for (let i = n0 + 1; i < T.rows.length; i++) { let d = T.rows[i].root_yaw - T.rows[i - 1].root_yaw; d -= Math.round(d); acc += d; mn = Math.max(mn, Math.abs(acc)); }
  check('B6 turn degrees 180 = a half turn (turn_away)', mn > 0.4 && mn < 0.6, `max |winding| ${mn.toFixed(2)}`); }

// ---- B9: the reverse burst from the seated-working state never stalls (matcher burst, then the model's refinements
// merged like page.html runCommand: IntentDedupe.seen → conflict → replaces)
{ const T = makeMuse(13); T.step(5, false);
  for (const c of [O('pick_up', 'headphones'), O('put_on', 'headphones'), O('pull', 'chair', {target: 'closer'}), O('pull', 'desk', {target: 'closer'}), O('pull', 'chair', {target: {object: 'desk'}}),
    O('sit_at', 'desk'), O('pick_up', 'laptop'), O('place', 'laptop', {place: 'desk'}), O('slide', 'laptop', {target: 'closer'}), O('open', 'laptop'), O('work_on', 'laptop')]) { T.m.command(c); runIdle(T, 40); }
  T.step(60, false);
  IntentDedupe.reset(); let tms = 50000; IntentDedupe.userSpoke(tms); const res = [];
  const run = (cmd, source) => { if (IntentDedupe.seen(cmd, tms, source)) { const fr = IntentDedupe.last.rec && IntentDedupe.last.rec.res; if (fr && fr.id !== undefined) { const ok = T.m.anchor(fr.id, source === "model" ? "voice" : source); if (process.env.DBG) console.log("   anchor", fr.id, ok, JSON.stringify(cmd).slice(0, 80)); } res.push({dup: true}); return; } const cf = IntentDedupe.last.conflict; const c2 = cf && cf.res && cf.res.id !== undefined ? Object.assign({}, cmd, {replaces: cf.res.id}) : cmd;
    const r = T.m.command(c2, {src: source === 'model' ? 'voice' : source}); IntentDedupe.attach(r); res.push(r); };
  const wait = s => { T.step(Math.round(s * 30), false); tms += s * 1000; };
  for (const c of [O('stop', null), O('close', {name: 'laptop'}), O('put_back', {ref: 'it'}, {place: 'floor'}), O('take_off', {name: 'headphones'}), {action: 'move', name: 'stand'}, O('push', {name: 'chair'}, {target: 'back'})]) { if (!c.object) delete c.object; run(c, 'matcher'); }
  wait(0.95);
  for (const c of [O('stop', null), O('close', {name: 'silver laptop'}), O('pick_up', {name: 'silver laptop'}), O('place', {name: 'silver laptop'}, {place: 'floor'}), O('take_off', {name: 'pink headphones'}),
    O('place', {name: 'pink headphones'}, {place: 'hook'}), {action: 'move', name: 'stand'}, O('push', {name: 'chair'}, {target: 'back'}), O('put_back', {name: 'desk'})]) { if (!c.object) delete c.object; run(c, 'model'); wait(0.35); }
  const done = runIdle(T, 150), s = T.m.state();
  if (process.env.DBG) for (const e of T.ev) if (e.id >= 12 && e.phase !== "start") console.log("   ev", e.id, e.name, e.phase, String(e.did || e.reason).slice(0, 140));
  check('B9 reverse burst drains: not typing, lid closed, headphones off and on the stand, laptop on the floor, standing, chair back',
    done && !s.activity && !s.open.length && !s.wearing.length && where(T, 'headphones') === 'home' && /floor|home/.test(where(T, 'laptop')) && s.posture === 'standing' && where(T, 'chair') === 'home',
    JSON.stringify({done, activity: s.activity, open: s.open, wearing: s.wearing, posture: s.posture, props: s.props.filter(p => ['headphones', 'laptop', 'chair', 'desk'].includes(p.id)).map(p => p.id + ':' + p.where)}));
  check('B9 no tool result said ok for a job that never ran', res.every(r => r.dup || r.ok === false || r.already || T.ev.some(e => e.id === r.id && /end|interrupted|fail/.test(e.phase))), res.map(r => r.dup ? 'dup' : (r.ok ? 'ok' : 'no') + '#' + r.id).join(' ')); }

// ---- B10: slide / scoot furniture = push / pull toward what it names
{ const T = makeMuse(14); T.step(5, false); const E = T.m.engine.E, desk = E.P.find(p => p.id === 'desk'), chair = E.P.find(p => p.id === 'chair');
  const d0 = Math.hypot(desk.pos[0] - chair.pos[0], desk.pos[2] - chair.pos[2]);
  const r = T.m.command(O('slide', {name: 'desk'}, {target: {name: 'chair'}})); runIdle(T, 40);
  const d1 = Math.hypot(desk.pos[0] - chair.pos[0], desk.pos[2] - chair.pos[2]);
  check('B10 slide the desk to the chair: the desk moves toward the chair', r.ok && d1 < d0 - 0.15, `${(r.did || r.reason)} | ${d0.toFixed(2)} → ${d1.toFixed(2)}`); }

// ---- B11: an unknown place for a thing with its own spot → its own spot, said truthfully
{ const T = makeMuse(15); T.step(5, false); T.m.command(O('pick_up', 'headphones')); runIdle(T, 30); const r = T.m.command(O('place', {name: 'headphones'}, {place: 'hook'})); runIdle(T, 40);
  check('B11 headphones "on the hook" (no hook): back on the stand, and it says there is no hook', r.ok && /no hook/.test(r.did) && where(T, 'headphones') === 'home', r.did || r.reason); }

// ---- B12: fetching from the seat at the desk ends seated again
{ const T = makeMuse(16); T.step(5, false); T.m.command(O('sit_at', 'desk')); runIdle(T, 40);
  T.m.command(O('pick_up', {name: 'laptop'})); T.m.command(O('bring', {ref: 'it'}, {target: 'closer'})); runIdle(T, 60); const s = T.m.state();
  check('B12 seated: pick up the laptop + bring it closer → seated at the desk, laptop on the desk', s.posture === 'seated' && s.atDesk && /desk/.test(where(T, 'laptop')) && !s.holding.left && !s.holding.right, JSON.stringify({posture: s.posture, atDesk: s.atDesk, laptop: where(T, 'laptop'), holding: s.holding})); }
{ const T = makeMuse(16); T.step(5, false); T.m.command(O('pick_up', {name: 'laptop'})); runIdle(T, 40); const s = T.m.state();
  check('B12 standing pick-up unchanged: holding the laptop, standing', s.posture === 'standing' && s.holding.left === 'laptop', JSON.stringify({posture: s.posture, holding: s.holding})); }

console.log(`${fails} failures`); process.exit(fails);
