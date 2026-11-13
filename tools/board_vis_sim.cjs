// Board visibility: one lesson through the REAL teach director + motion engine (node, no server, no renderer), recorded
// for the board-visibility metric (renderer/tools/board_vis_render.py + tools/board_vis_metric.py).
//   node tools/board_vis_sim.cjs --lesson CASE.json --out DIR [--every 3] [--cfg '{"windowU":0.15}'] [--tail 2.0]
// CASE.json: a board-brain replay case ({items: [...]}) or {items}. Writes DIR/rows.jsonl (rig params every --every
// frames), DIR/ink.jsonl (board_ink messages with their playback t), DIR/phases.jsonl (director phase / item each
// frame), DIR/meta.json (times, stats).
'use strict';
const fs = require('fs'), path = require('path');
const L = require(path.join(__dirname, '..', 'tests', 'teach_lib.cjs'));
const a = {every: 3, tail: 2.0, maxS: 400};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const C = JSON.parse(fs.readFileSync(a.lesson, 'utf8'));
let items = (C.items || []).filter(Boolean);
items = items.map(it => Object.assign({}, it, {strokes: (it.strokes || []).map(st => st.map(q => [+q[0], +q[1]]))}));
const cfg = a.cfg ? JSON.parse(a.cfg) : undefined;
const hook = a.hook ? require(path.resolve(a.hook))(a.hookArgs ? JSON.parse(a.hookArgs) : {}) : undefined;
const tlog = [];
const R = L.makeRig({items, cfg, rows: true, brainLatency: 1.0, itemGap: 0.4, stanceHook: hook, log: r => tlog.push(Object.assign({t: +R.now().toFixed(3)}, r))});
fs.mkdirSync(a.out, {recursive: true});
const T = R.teach._T;
const ph = [], phase = () => ({t: +R.now().toFixed(4), ph: T.phase, item: T.item ? T.item.id : null, at: T.at, say: T.say ? T.say.item : null, req: !!T.req, aud: !!R.voice.audible});
const t0 = R.now();
R.teach.call('teach_board', {request: (C.case && C.case.request) || 'lesson'});
let done = null; const ex = [];
for (let i = 0; i < +a.maxS * 30; i++) {
  R.tick(1); ph.push(phase());
  { const k = R.X.m.engine.kit, E = k.E; ex[R.X.rows.length - 1] = {pl: k.fwdW('l', E.paw.l.h).map(x => +x.toFixed(4)), pr: k.fwdW('r', E.paw.r.h).map(x => +x.toFixed(4)), bp: +((E.bend && E.bend.p) || 0).toFixed(4), by: +((E.bend && E.bend.y) || 0).toFixed(4)}; }
  if (done === null && !T.req && T.phase === 'idle' && !T.jobs.size && R.now() - t0 > 2) done = R.now();   // (its own jobs too: the end-of-lesson duster grab + the turn to him)
  if (done !== null && R.now() - done >= +a.tail) break;
}
const rows = R.X.rows, keep = [];
for (let i = 0; i < rows.length; i += +a.every) {
  const r = rows[i], p = {};
  for (const k of Object.keys(r)) if (!k.startsWith('_') && k !== 't' && typeof r[k] === 'number') p[k] = +r[k].toFixed(5);
  keep.push(JSON.stringify({t: +r.t.toFixed(4), p, x: ex[i] || null}));
}
fs.writeFileSync(path.join(a.out, 'rows.jsonl'), keep.join('\n') + '\n');
fs.writeFileSync(path.join(a.out, 'ink.jsonl'), R.sent.filter(m => m.type === 'board_ink' || m.type === 'board_erased').map(m => JSON.stringify(m.type === 'board_ink' ? {t: m.t, at: m.at, segs: m.segs} : {t: m.t, at: m.at, erased: m.regions, all: !!m.all})).join('\n') + '\n');
fs.writeFileSync(path.join(a.out, 'phases.jsonl'), ph.map(x => JSON.stringify(x)).join('\n') + '\n');
fs.writeFileSync(path.join(a.out, 'teach_log.jsonl'), tlog.map(x => JSON.stringify(x)).join('\n') + '\n');
const meta = {lesson: a.lesson, id: C.case ? C.case.id : path.basename(a.lesson), t0, done, end: R.now(), items: items.length, stats: R.teach.stats(),
  firstMark: T.firstMark, cfg: cfg || null, every: +a.every};
fs.writeFileSync(path.join(a.out, 'meta.json'), JSON.stringify(meta, null, 1));
console.log(JSON.stringify({id: meta.id, dur: +(R.now() - t0).toFixed(1), done: done && +(done - t0).toFixed(1), stats: meta.stats, firstMark: T.firstMark && T.firstMark.s_after_call}));
