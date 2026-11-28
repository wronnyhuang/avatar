// a job that needs a paw that is holding something frees it first (the engine's paw claim:
// planGoals 'paws' / claimPaws -> pawsFree, the one put-back / set-down path), said in the action log — unless the held
// thing is the job's own tool (chalk, eraser). The real engine + teach director, on the muse and on Juno (her own body
// and scene; the rule was not tuned on her). Switch: scene config handfree:false / ?handfree=0 / createMotion({handFree:false})
// (here E.handFree = false) = the old behaviour (the job works around what its paws hold).
// Metric printed at the end: "a non-tool thing still in a paw when the task's motion starts" on N cases, on vs off.
'use strict';
const path = require('path');
const A = path.join(__dirname, '..', 'renderer', 'animals');
let pass = 0, fail = 0;
function check(name, ok, info) { if (ok) pass++; else fail++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok || info === undefined ? '' : '  — ' + JSON.stringify(info).slice(0, 400))); }
const metric = {on: {n: 0, held: 0}, off: {n: 0, held: 0}};
const tally = (off, stillHeld) => { const m = off ? metric.off : metric.on; m.n++; if (stillHeld) m.held++; };
const CHARS = (process.env.HF_CHARS || 'muse junor').split(/\s+/).filter(Boolean);
const O = (verb, name, x) => Object.assign({action: 'object', verb, object: {name}}, x || {});

// Juno's own body + scene the way the hub builds them (renderer/humans/charfiles.props_for — as gesture_layer_test does); the
// mascots ship byte-identical copies of the muse's body + props files, so they add nothing as cases
const fs = require('fs'), os = require('os'), {execFileSync} = require('child_process');
const ROOT = path.join(__dirname, '..'), FILES = {};
function filesFor(c) {
  if (FILES[c]) return FILES[c];
  if (c === 'juno' || c === 'junor') {
    const PY = process.env.JUNO_PY || 'python3';
    const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'handfree_' + c + '_')), c + '.props.json');
    execFileSync(PY, ['-c', `import sys,json; sys.path.insert(0,'renderer'); from humans.charfiles import props_for; json.dump(props_for(${JSON.stringify(c)}), open(${JSON.stringify(out)},'w'))`], {cwd: ROOT});
    return (FILES[c] = {body: path.join(ROOT, 'renderer/humans', c, c + '.body.json'), props: out});
  }
  return (FILES[c] = {body: path.join(A, c + '.body.json'), props: path.join(A, c + '.props.json')});
}
function rigFor(c, off) {
  const F = filesFor(c); process.env.MUSE_BODY = F.body; process.env.MUSE_PROPS = F.props;
  delete require.cache[require.resolve('./teach_lib.cjs')];
  const {makeRig} = require('./teach_lib.cjs');
  const R = makeRig(); R.tick(5);
  if (off) R.X.m.engine.E.handFree = false;
  return R;
}
const E = R => R.X.m.engine.E, P = (R, id) => E(R).P.find(p => p.id === id);
const idle = R => { for (let i = 0; i < 30 * 40; i++) { R.tick(1); const e = E(R); if (!e.move && !e.chain && !e.jobs.length && i > 10) break; } };
const inks = R => R.sent.filter(m => m.type === 'board_ink').length;
// run a board lesson; at the first chalk mark: what each id is doing
function writeAndWatch(R, ids) {
  R.teach.call('teach_board', {request: 'write cat'});
  let at = null;
  for (let i = 0; i < 30 * 45 && !at; i++) { const k = inks(R); R.tick(1); if (inks(R) > k) at = Object.fromEntries(ids.map(id => [id, P(R, id) ? P(R, id).held || null : 'n/a'])); }
  return at;
}
const logTxt = R => R.X.ev.filter(e => e.did && e.phase === 'start').map(e => e.did).join(' || ');

for (const c of CHARS) {
  // 1. one-pawed things, then a board write
  for (const thing of ['ball', 'mug', 'book']) for (const off of [false, true]) {
    const R = rigFor(c, off), tag = `${c}/${thing}${off ? ' [handfree off]' : ''}`;
    const r = R.X.m.command(O('pick_up', thing), {src: 'voice'}); idle(R);
    const had = P(R, thing) && P(R, thing).held;
    if (!r.ok || !had) { check(`${tag}: picked up first`, false, {r: r.reason || r.did, had}); continue; }
    const at = writeAndWatch(R, [thing, 'chalk']);
    if (!at) { check(`${tag}: a chalk mark was made`, false, logTxt(R)); continue; }
    tally(off, !!at[thing]);
    if (!off) {
      check(`${tag}: the ${thing} is out of its paws before the first chalk mark, the chalk in a paw`, !at[thing] && !!at.chalk, at);
      const nm = P(R, thing).name;
      check(`${tag}: the action log says it put the ${thing} away to free its paw`, new RegExp(`(putting|setting) the ${nm}[^|]*to free its paw`).test(logTxt(R)), logTxt(R).slice(0, 300));
      check(`${tag}: the ${thing} rests somewhere (home / a surface), not flying`, ['home', 'placed'].includes(P(R, thing).where) && !P(R, thing).flying, {where: P(R, thing).where});
    } else check(`${tag}: switch off = the old behaviour (still held while it writes)`, !!at[thing], at);
  }
  // 2. a two-pawed carry, then a board write
  for (const off of [false, true]) {
    const R = rigFor(c, off), tag = `${c}/gift_box (two paws)${off ? ' [handfree off]' : ''}`;
    const r = R.X.m.command(O('pick_up', 'gift_box'), {src: 'voice'}); idle(R);
    if (!r.ok || P(R, 'gift_box').held !== 'both') { check(`${tag}: carried in both paws first`, false, {r: r.reason || r.did, held: P(R, 'gift_box').held}); continue; }
    const at = writeAndWatch(R, ['gift_box', 'chalk']);
    if (!off && !at) { check(`${tag}: a chalk mark was made`, false, logTxt(R)); continue; }
    tally(off, !at || !!at.gift_box);
    if (!off) check(`${tag}: the box is set down before the first chalk mark`, !at.gift_box && !!at.chalk && /to free its paws/.test(logTxt(R)), {at, log: logTxt(R).slice(0, 300)});
    else console.log(`NOTE ${tag}: old behaviour at the first mark: ${JSON.stringify(at)} (the muse keeps the box in both paws; Juno's chalk fetch already set it down through the planner's count rule)`);
  }
  // 3. exemption: the held thing IS the task's tool (he handed it the chalk / the eraser first)
  for (const tool of ['chalk', 'eraser']) {
    const R = rigFor(c, false), tag = `${c}/exempt ${tool}`;
    const r = R.X.m.command(O('pick_up', tool), {src: 'voice'}); idle(R);
    if (!r.ok || !P(R, tool).held) { check(`${tag}: picked up first`, false, r.reason || r.did); continue; }
    const at = writeAndWatch(R, [tool, 'chalk']);
    check(`${tag}: never put away to free a paw (it is the board's own tool)`, !!at && !/to free its paw/.test(logTxt(R)) && !!at.chalk, {at, log: logTxt(R).slice(0, 300)});
  }
  // 4. a board erase with the ball in the duster's paw
  for (const off of [false, true]) {
    const R = rigFor(c, off), tag = `${c}/erase${off ? ' [handfree off]' : ''}`;
    R.teach.call('teach_board', {request: 'two plus two'});
    for (let i = 0; i < 30 * 40; i++) { R.tick(1); if (R.teach._T.phase === 'idle' && !R.teach._T.req && i > 90) break; }
    R.teach.call('go_home', {}); idle(R); for (let i = 0; i < 30 * 6; i++) R.tick(1); idle(R);
    const r = R.X.m.command(O('pick_up', 'ball'), {src: 'voice'}); idle(R);
    if (!r.ok || !P(R, 'ball').held) { check(`${tag}: ball picked up`, false, r.reason || r.did); continue; }
    R.teach.call('board_erase', {what: 'everything'});
    let at = null;
    for (let i = 0; i < 30 * 40 && !at; i++) { R.tick(1); if (P(R, 'eraser').held && R.teach._T.phase === 'erasing') at = {ball: P(R, 'ball').held || null, eraser: P(R, 'eraser').held}; }
    tally(off, !at || !!(at.ball && at.ball === at.eraser) || !!at.ball);
    if (!off) check(`${tag}: the ball is out of its paws while it wipes`, !!at && !at.ball && /to free its (\w+ )?paw/.test(logTxt(R)), {at, log: logTxt(R).slice(-300)});
    else check(`${tag}: switch off = old (ball still in a paw while wiping)`, !at || !!at.ball, at);
  }
  // 5. a move (a dance) while holding the ball: NOT a claim — the move engine's own rule stands (a busy paw sits the move out
  //    and keeps its grasp; muse_moves_test pins "a held ball stays in the paw through two spins")
  {
    const R = rigFor(c, false), tag = `${c}/dance`;
    R.X.m.command(O('pick_up', 'ball'), {src: 'voice'}); idle(R);
    const r = R.X.m.command({action: 'move', name: 'dance'}, {src: 'voice'});
    check(`${tag}: a move keeps the held ball (no put-down, as before)`, r.ok && !/to free its paw/.test(r.did || '') && !!P(R, 'ball').held, {did: r.did});
  }
  // 6. typing on the laptop while holding the ball (the planner's own claim: it already put things down; kept)
  {
    const R = rigFor(c, false), tag = `${c}/type`;
    R.X.m.command(O('pick_up', 'ball'), {src: 'voice'}); idle(R);
    const r = R.X.m.command(O('type_on', 'laptop'), {src: 'voice'}); let typed = false, ballWhileTyping = 'never typed';
    for (let i = 0; i < 30 * 90 && !typed; i++) { R.tick(1); if (E(R).activity && !E(R).activity.paused) { typed = true; ballWhileTyping = P(R, 'ball').held || null; } }
    check(`${tag}: types with both paws free (ball put away first)`, r.ok !== false && typed && ballWhileTyping === null, {r: (r.reason || r.did || '').slice(0, 80), ballWhileTyping});
  }
  // 7. both paws full (ball + mug), then a two-pawed carry / a third thing: the planner's own claim (pawsFree) frees them
  for (const [want, n] of [['gift_box', 2], ['book', 1]]) {
    const R = rigFor(c, false), tag = `${c}/full paws -> ${want}`;
    R.X.m.command(O('pick_up', 'ball'), {src: 'voice'}); idle(R); R.X.m.command(O('pick_up', 'mug'), {src: 'voice'}); idle(R);
    if (!P(R, 'ball').held || !P(R, 'mug').held) { check(`${tag}: ball + mug in paws first`, false, {ball: P(R, 'ball').held, mug: P(R, 'mug').held}); continue; }
    const r = R.X.m.command(O('pick_up', want), {src: 'voice'}); idle(R);
    const freed = ['ball', 'mug'].filter(id => !P(R, id).held).length;
    check(`${tag}: ${n} thing(s) put down first, then it holds the ${want}`, r.ok !== false && !!P(R, want).held && freed >= n, {r: (r.reason || r.did || '').slice(0, 200), freed, held: P(R, want).held});
  }
}
const pct = m => m.n ? `${m.held}/${m.n}` : '0/0';
console.log(`\nMETRIC non-tool thing still in a paw when the task's motion starts: handfree off ${pct(metric.off)} -> on ${pct(metric.on)}`);
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
