// erase: the quick wipe (design target: "erase in a few seconds, not 20-23 s, and a new lesson on a full board shouldn't
// take 33-35 s to its first mark"). Node, headless, the real scene: timings from the tool call, the chalk left on the
// simulated server canvas, the duster's paw, the lazy page turn, abort mid-wipe, the free-board first mark unchanged.
'use strict';
const {makeRig, stubItems, boardResidue} = require('./teach_lib.cjs');
let fails = 0, n = 0;
function ok(c, name, info) { n++; if (c) console.log('PASS', name, info !== undefined ? JSON.stringify(info) : ''); else { fails++; console.log('FAIL', name, info !== undefined ? '— ' + JSON.stringify(info) : ''); } }
const segsOf = R => R.sent.filter(m => m.type === 'board_ink').flatMap(m => m.segs);
const props = R => Object.fromEntries(R.X.m.engine.props().map(p => [p.id, p.where]));
function fullItems() {   // three columns of four lines (the brain's layout flows down the reach band, column by column)
  const L = [];
  for (const [ci, u] of [0.04, 0.37, 0.70].entries()) for (let r = 0; r < 4; r++) L.push({text: ['3x + 2 = 11', '3x = 9', 'x = 3', '9 - 4 = 5'][(r + ci) % 4], u, v: 0.36 + 0.145 * r, h: 0.1, say: 'Line ' + (ci * 4 + r) + '.'});
  return stubItems(L).map((x, i) => Object.assign(x, {id: 'f' + i}));
}
function fullBoard(opt = {}) {
  const R = makeRig(Object.assign({items: fullItems(), brainLatency: 0.4, itemGap: 0.1}, opt));
  R.tick(5); R.teach.call('teach_board', {request: 'fill the board'});
  R.until(() => R.teach.stats().items >= 12 && R.teach._T.phase === 'idle' && !R.X.m.engine.E.jobs.length, 400);
  R.tick(30);
  return R;
}
const newLesson = () => [{id: 'e1', kind: 'erase', say_before: 'Let me clear some space.', bbox: [0.03, 0.34, 0.97, 0.92], target: ['all']}]
  .concat(stubItems([{text: '5 + 5', say: 'Five and five.', u: 0.06}, {text: '= 10', say: 'Ten.', u: 0.06}]).map((x, i) => Object.assign(x, {id: 'n' + i})));

// ---- 1. free board: the first mark is where it was (the duster is not fetched before the first item)
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'two plus two'});
  R.until(() => R.teach._T.firstMark, 30);
  const fm = R.teach._T.firstMark && R.teach._T.firstMark.s_after_call;
  ok(fm !== null && fm <= 5.6, 'free board: first mark <= 5.6 s after the call (no duster fetch before it)', {s: fm});
  R.until(() => R.teach._T.phase === 'idle' && !R.teach._T.req && !R.X.m.engine.E.jobs.length, 40);
  const P = props(R);
  ok(/paw/.test(P.eraser) && P.eraser !== P.chalk, 'a lesson by the tray: the free paw takes the duster between its lines', P);
  // ---- 2. "erase that" at the board: the duster is in hand, one column
  R.hub.eraseRegions = [[0.02, 0.3, 0.4, 0.75]];   // what the brain names for "that" (the sum it just wrote)
  const t0 = R.now(); R.teach.call('board_erase', {what: 'the sum'});
  R.until(() => R.sent.some(m => m.type === 'board_erased'), 20);
  const s = R.now() - t0;
  ok(s <= 5, 'erase a region at the board: done <= 5 s after the call (brain 1.0 s incl.)', {s: +s.toFixed(2)});
  ok(boardResidue(segsOf(R)).residue < 0.02, 'erase a region: < 2% of its chalk left', boardResidue(segsOf(R)));
  ok(/paw/.test(props(R).chalk) && /paw/.test(props(R).eraser), 'no tray trips: chalk and duster both still in the paws', props(R));
}
// ---- 3. erase a completely full board: <= 8 s, clean, the felt really passes (no clean-up needed)
{
  const R = fullBoard();
  const n0 = R.sent.length, t0 = R.now();
  R.teach.call('board_erase', {what: 'all'});
  R.until(() => R.sent.some(m => m.type === 'board_erased'), 30);
  const s = R.now() - t0, res = boardResidue(segsOf(R));
  ok(s <= 8, 'erase a full board: done <= 8 s after the call', {s: +s.toFixed(2)});
  ok(res.residue < 0.02, 'erase a full board: < 2% of the chalk left by the felt itself', res);
  const er = R.sent.slice(n0).filter(m => m.type === 'board_ink').flatMap(m => m.segs).filter(x => x.erase);
  ok(er.length > 30 && er.filter(x => x.sweep).length > er.length * 0.8, 'the felt is stamped along its track (swept stamps)', {n: er.length, sweep: er.filter(x => x.sweep).length});
}
// ---- 4. a new lesson on a full board: a lazy page turn — first mark <= 7 s, the rest wiped by the lesson's end
{
  const R = fullBoard();
  const old = new Set(segsOf(R).filter(x => !x.erase));
  R.hub.items = newLesson(); R.hub.latency = 1.5;
  const t0 = R.now(), p0 = R.prompts.length; R.teach.call('teach_board', {request: 'something new'});
  R.until(() => R.teach._T.firstMark && R.teach._T.firstMark.id === (R.teach._T.req || {}).id, 30);
  const fm = R.now() - t0;
  ok(fm <= 7, 'full board, new lesson: first mark <= 7 s after the call (brain 1.5 s incl.)', {s: +fm.toFixed(2)});
  const partial = R.sent.filter(m => m.type === 'board_erased' && m.partial);
  ok(partial.length === 1, 'only the first item\'s columns were wiped before it was written', partial.map(m => m.regions));
  R.until(() => R.teach._T.phase === 'idle' && !R.teach._T.req && !R.X.m.engine.E.jobs.length, 60);
  const all = segsOf(R), oldLeft = boardResidue(all.filter(x => x.erase || old.has(x)), x => old.has(x)), newLeft = boardResidue(all, x => !old.has(x));
  ok(oldLeft.residue < 0.02, 'by the lesson\'s end the old chalk is gone (< 2% left)', oldLeft);
  ok(newLeft.residue > 0.95, 'the new lesson\'s chalk is untouched by the later wipes', newLeft);
  const fin = R.sent.filter(m => m.type === 'board_erased' && !m.partial);
  ok(fin.length === 1 && R.teach.stats().items >= 14, 'the page turn ends with one final board_erased; every item written', {fin: fin.length, items: R.teach.stats().items});
  const notes = R.prompts.slice(p0).filter(P => /finished writing/.test(P.text));
  ok(notes.length === 1, 'one hand-back line for the lesson (said while it wipes the rest)', notes.map(P => P.text.slice(0, 60)));
}
// ---- 5. stop during the page turn's wipe: halts at once, tools back on the tray, home
{
  const R = fullBoard();
  R.hub.items = newLesson(); R.hub.latency = 1.5;
  R.teach.call('teach_board', {request: 'something new'});
  const got = R.until(() => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && C.cur.name === 'erase'); }, 20);
  const t0 = R.now();
  R.teach.userText('stop', true);
  let lastWipe = null;
  for (let i = 0; i < 30 * 25; i++) { const k = R.sent.length; R.tick(1); if (R.sent.slice(k).some(m => m.type === 'board_ink' && m.segs.some(x => x.erase))) lastWipe = R.now(); if (R.teach.state().at === 'home' && !R.X.m.engine.E.jobs.length && i > 30) break; }
  ok(got && (lastWipe === null || lastWipe - t0 <= 0.3), 'stop mid-wipe: no felt stamp later than 0.3 s after it', {got, late: lastWipe === null ? 0 : +(lastWipe - t0).toFixed(2)});
  ok(R.teach.state().at === 'home' && props(R).chalk === 'home' && props(R).eraser === 'home', 'stop mid-wipe: home, chalk and duster on the tray', Object.assign({at: R.teach.state().at}, props(R)));
  ok(!R.teach._T.wipe && !R.teach._T.wiping, 'stop mid-wipe: no wipe left pending', {});
}
// ---- 6. the duster is not taken while he has asked for a thing (his object request owns the paws)
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.state().phase === 'writing', 40);
  R.X.m.engine.command({action: 'object', verb: 'pick_up', object: {name: 'mug'}}, 'tool', 91);
  R.until(() => R.teach._T.phase === 'idle' && !R.teach._T.req, 60);
  ok(!/eraser/.test(JSON.stringify(R.X.m.engine.state().holding)), 'his object request mid-lesson: the duster stays on the tray', R.X.m.engine.state().holding);
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails ? 1 : 0);
