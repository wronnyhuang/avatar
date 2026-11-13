// node tests/muse_refine_test.cjs — streaming refinements (hub matcher: a stable partial fires, then the final refines it
// with `replaces: <id>`): one action, never two. Exit code = failures.
'use strict';
const {makeMuse} = require('./muse_lib.cjs');
let fails = 0; const out = [];
const check = (n, ok, d) => { out.push((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const apexes = (rows, from) => { let n = 0; for (let i = from + 1; i < rows.length - 1; i++) { const y = rows[i].root_y; if (y > 0.04 && y >= rows[i - 1].root_y && y > rows[i + 1].root_y) n++; } return n; };
// 1. jump → (0.1 s later) jump higher: restarted with the new height, one jump
{ const T = makeMuse(3); T.step(10); const n0 = T.rows.length;
  const a = T.m.command({action: 'move', name: 'jump'}, {src: 'matcher'}); T.step(3);
  const b = T.m.command({action: 'move', name: 'jump', height: 1, replaces: a.id}, {src: 'matcher'}); T.runUntilIdle(10);
  const k = apexes(T.rows, n0), ev = T.ev.filter(e => e.kind === 'move' && (e.phase === 'end' || e.phase === 'interrupted'));
  let vmax = 0; for (let i = n0 + 1; i < T.rows.length; i++) for (const p of ['root_y', 'root_x', 'root_z', 'squash', 'foot_l_y', 'hand_l_y']) vmax = Math.max(vmax, Math.abs((T.rows[i][p] || 0) - (T.rows[i - 1][p] || 0)) * 30);
  const P1 = makeMuse(3); P1.step(10); const p0 = P1.rows.length; P1.m.command({action: 'move', name: 'jump', height: 1}, {src: 'matcher'}); P1.runUntilIdle(10);
  let vref = 0; for (let i = p0 + 1; i < P1.rows.length; i++) for (const p of ['root_y', 'root_x', 'root_z', 'squash', 'foot_l_y', 'hand_l_y']) vref = Math.max(vref, Math.abs((P1.rows[i][p] || 0) - (P1.rows[i - 1][p] || 0)) * 30);
  check('the restart is as smooth as a plain high jump (max per-frame rate of root/squash/feet/paws within 15%)', vmax <= vref * 1.15, `refined ${vmax.toFixed(2)}/s vs plain ${vref.toFixed(2)}/s`);
  check('jump → "jump higher" within 0.1 s: ONE jump, the first reported replaced', b.ok && b.replaced === a.id && k === 1 && ev.some(e => e.id === a.id && e.phase === 'interrupted'), `apexes ${k}, b=${JSON.stringify(b).slice(0, 120)}, events ${ev.map(e => e.id + ':' + e.phase).join(',')}`); }
// 2. jump → (0.6 s later, airborne) the same jump again: merged, still one jump
{ const T = makeMuse(4); T.step(10); const n0 = T.rows.length;
  const a = T.m.command({action: 'move', name: 'jump'}, {src: 'matcher'}); T.step(18);
  const b = T.m.command({action: 'move', name: 'jump', height: 1, replaces: a.id}, {src: 'matcher'}); T.runUntilIdle(10);
  const k = apexes(T.rows, n0);
  check('jump → refinement 0.6 s later (already in the air): merged, one jump', b.ok && b.already && k === 1, `apexes ${k}, b=${JSON.stringify(b).slice(0, 140)}`); }
// 3. spin → (0.4 s) "a silly dance": the spin stops gracefully, the dance runs
{ const T = makeMuse(5); T.step(10);
  const a = T.m.command({action: 'move', name: 'spin'}, {src: 'matcher'}); T.step(12);
  const b = T.m.command({action: 'move', name: 'dance', style: 'silly', replaces: a.id}, {src: 'matcher'}); T.runUntilIdle(20);
  const e1 = T.ev.find(e => e.id === a.id && (e.phase === 'interrupted' || e.phase === 'end')), e2 = T.ev.find(e => e.id === b.id && e.phase === 'end');
  check('spin → "silly dance": spin interrupted (says why), dance done', e1 && e1.phase === 'interrupted' && /replaced/.test(e1.did + ' ' + (e1.reason || '')) && !!e2, `${e1 && e1.phase}:${e1 && e1.did} | ${e2 && e2.did}`); }
// 4. after it is over, the identical request is not done twice
{ const T = makeMuse(6); T.step(10); const n0 = T.rows.length;
  const a = T.m.command({action: 'move', name: 'hop'}, {src: 'matcher'}); T.runUntilIdle(10);
  const b = T.m.command({action: 'move', name: 'hop', replaces: a.id}, {src: 'matcher'}); T.runUntilIdle(10);
  check('hop finished, then the identical refinement: dropped as already done', b.ok && b.already && apexes(T.rows, n0) === 1, JSON.stringify(b)); }
// 5. objects: pick up the ball → (0.3 s) "… with your left paw": one pickup, in the left paw
{ const T = makeMuse(7); T.step(10);
  const a = T.m.command({action: 'object', verb: 'pick_up', object: {name: 'ball'}}, {src: 'matcher'}); T.step(9);
  const b = T.m.command({action: 'object', verb: 'pick_up', object: {name: 'ball'}, side: 'left', replaces: a.id}, {src: 'matcher'}); T.runUntilIdle(30);
  const s = T.m.state(), picks = T.ev.filter(e => e.kind === 'object_event' && e.name === 'picked_up').length;
  check('pick up the ball → "with your left paw": one pickup, held in the left paw', b.ok && picks === 1 && s.holding.left === 'ball', `picks ${picks}, holding ${JSON.stringify(s.holding)}, ${b.did || b.reason}`); }
console.log(out.join('\n')); console.log(`\n${out.length - fails}/${out.length} passed`); process.exit(fails);
