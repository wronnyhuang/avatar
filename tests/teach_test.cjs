// teach: the teaching director + performer on the muse (node, headless): walk to the board, chalk from the tray,
// items written stroke by stroke with marks only where the paw got, the say -> write cadence and its timing rules,
// erase, the TV station, and ABORT at every phase (walking, picking up, mid-stroke, between items, erasing, at the TV)
// -> the stroke stops within 0.3 s, the tools go back on the tray, it walks home and faces him. Exit code = failures.
'use strict';
const {makeRig, stubItems, boardResidue} = require('./teach_lib.cjs');
let fails = 0, n = 0;
function ok(c, name, info) { n++; if (c) console.log('PASS', name); else { fails++; console.log('FAIL', name, info !== undefined ? '— ' + JSON.stringify(info) : ''); } }
const inkMsgs = R => R.sent.filter(m => m.type === 'board_ink');
const inkSegs = R => inkMsgs(R).reduce((a, m) => a.concat(m.segs), []);
const props = R => Object.fromEntries(R.X.m.engine.props().map(p => [p.id, p.where]));
function atHome(R) {
  const st = R.X.m.engine.state(), E = R.X.m.engine.E;
  return Math.hypot(st.root.x, st.root.z) < 0.04 && Math.abs(st.root.yaw) < 0.03 && !E.jobs.length && !E.move && !E.chain;
}
function abortAt(label, cond, opt = {}) {
  const R = makeRig(opt.rig || {});
  R.tick(5);
  if (opt.start) opt.start(R); else R.teach.call('teach_board', {request: 'show two plus two'});
  const got = R.until(() => cond(R), 40);
  ok(got, `abort/${label}: reached the phase`, R.teach.state());
  const t0 = R.now(), n0 = inkMsgs(R).length;
  if (opt.how === 'tool') R.teach.call('go_home', {});
  else if (opt.how === 'stopcmd') R.X.m.command({action: 'move', name: 'stop'});
  else R.teach.userText(opt.words || 'never mind', true);
  let lastInk = null;
  for (let i = 0; i < 30 * 30; i++) { const k = inkMsgs(R).length; R.tick(1); if (inkMsgs(R).length > k) lastInk = R.now(); if (i > 30 && atHome(R)) break; }
  const late = lastInk === null ? 0 : lastInk - t0;
  ok(late <= 0.3, `abort/${label}: no mark later than 0.3 s after the abort`, {late: +late.toFixed(3), before: n0});
  ok(atHome(R), `abort/${label}: home, facing him, idle`, R.X.m.engine.state().root);
  const P = props(R);
  ok(P.chalk === 'home' && P.eraser === 'home', `abort/${label}: chalk + eraser back on the tray`, P);
  ok(R.sent.some(m => m.type === 'teach_cancel') || label === 'tv' || label === 'idle', `abort/${label}: the brain call is cancelled`);
  ok(R.teach.state().at === 'home' && !R.teach.state().req, `abort/${label}: director idle at home`, R.teach.state());
  const t1 = R.now();
  return {R, home_s: +(t1 - t0).toFixed(2)};
}

// ---- 1. a full worked sum: walk, chalk, three items, cadence, marks under the tip, turn to him at the end
{
  const items = stubItems([{text: '3x + 2 = 11', say: "Let's write the equation."}, {text: '3x = 9', say: 'Take two from both sides.'}, {text: 'x = 3', say: 'And divide by three.'}]);
  const R = makeRig({items, brainLatency: 1.2, itemGap: 0.5});
  R.tick(5);
  const t0 = R.now();
  const r = R.teach.call('teach_board', {request: 'solve 3x + 2 = 11 step by step'});
  ok(r.ok && /board/.test(r.did) && /line/.test(r.next), 'teach_board answers at once with what it is doing + how to talk', r);
  ok(R.sent.some(m => m.type === 'teach_plan' && m.request && Array.isArray(m.context)), 'the hub is asked for a plan with the conversation');
  const T = R.teach._T, starts = {}, ends = {};
  let sawBoardWhere = false, turnedYaw = null, writeYaw = [];
  for (let i = 0; i < 30 * 60; i++) {
    R.tick(1);
    if (T.item && !starts[T.item.id]) starts[T.item.id] = R.now();
    for (const it of Object.keys(starts)) if (!ends[it] && T.item && T.item.id !== it) ends[it] = R.now();
    if (R.X.m.state().station === 'at the board') sawBoardWhere = true;
    if (T.phase === 'writing') writeYaw.push(R.X.m.engine.state().root.yaw);
    if (T.phase === 'idle' && !T.req && i > 90) break;
  }
  R.tick(60); for (let i = 0; i < 8 * 30 && T.jobs.size; i++) R.tick(1);   // (a step aside off the text comes first)
  turnedYaw = R.X.m.engine.state().root.yaw;
  { // teach: the body stands clear of what it wrote when it turns to him (seen from the render camera)
    const bx = Object.values(T.drawn).reduce((a, q) => [Math.min(a[0], q[0]), Math.min(a[1], q[1]), Math.max(a[2], q[2]), Math.max(a[3], q[3])], [1, 1, 0, 0]);
    const rt = R.X.m.engine.state().root, h = R.teach._hiddenU(rt.x, rt.z);
    ok(h && (h[0] > bx[2] || h[1] < bx[0]), 'turned to him, its body does not hide the text it wrote (camera view)', {hidden: h && h.map(x => +x.toFixed(3)), text: bx.map(x => +x.toFixed(3))});
  }
  const st = R.teach.stats();
  ok(st.items === 3 && st.strokes > 10, 'all three items written', st);
  const segs = inkSegs(R).filter(s => !s.erase);
  ok(segs.length > 40, 'marks reported along the strokes', segs.length);
  ok(new Set(segs.map(s => s.item)).size === 3, 'marks for every item');
  ok(inkMsgs(R).every(m => typeof m.t === 'number'), 'every mark carries a playback stamp');
  // each stroke's marks advance monotonically (never ahead, never backwards) and reach its end
  const by = {}; for (const s of segs) (by[s.item + '/' + s.stroke] = by[s.item + '/' + s.stroke] || []).push(s);
  const mono = Object.values(by).every(L => L.every((s, i) => i === 0 || s.s0 >= L[i - 1].s1 - 1e-6) && L[L.length - 1].s1 > 0.999);
  ok(mono, 'per stroke: marks continue where the last one ended and finish the stroke');
  // the chalk tip is ON the board where the marks are: grasp point back along the chalk by its length lands on the mark
  const B = R.teach.surface('board'), cfg = R.teach.cfg;
  // (boardvis: the stick also leans chalkSide toward the side away from the body: -U for the screen-left paw)
  const dir = (() => { const sk = -(cfg.chalkSide || 0), d = [-B.n[0] + sk * B.U[0], cfg.chalkUp - B.n[1], -B.n[2] + sk * B.U[2]], l = Math.hypot(...d); return d.map(x => x / l); })();
  let worstPlane = 0, worstAlong = 0;
  for (const m of R.inks) for (const s of m.segs) {
    if (s.erase) continue;
    const g = m.paw.l, tip = [g[0] + dir[0] * cfg.tipLen, g[1] + dir[1] * cfg.tipLen, g[2] + dir[2] * cfg.tipLen];
    const pl = (tip[0] - B.O[0]) * B.n[0] + (tip[1] - B.O[1]) * B.n[1] + (tip[2] - B.O[2]) * B.n[2];
    worstPlane = Math.max(worstPlane, Math.abs(pl));
    const q = s.pts[s.pts.length - 1], W = B.world(q[0], q[1]);
    worstAlong = Math.max(worstAlong, Math.hypot(tip[0] - W[0], tip[1] - W[1], tip[2] - W[2]));
  }
  ok(worstPlane < 0.025, 'chalk tip within 0.025 H of the board plane at every mark', +worstPlane.toFixed(4));
  ok(worstAlong < 0.036, 'chalk tip within 0.036 H of the newest mark (one frame of travel)', +worstAlong.toFixed(4));
  ok(st.firstMark !== null && st.firstMark < 14, 'first mark time recorded (walk + chalk + first line)', st.firstMark);
  // cadence: one line per item, each asked before that item is written, writing starts by the timing rule
  const lines = R.prompts.filter(p => /only this/.test(p.text));
  ok(lines.length === 3, 'one "say this line" note per item', lines.map(p => p.text.slice(-40)));
  let cad = true; const info = [];
  items.forEach((it, i) => {
    const p = lines[i], s = starts[it.id];
    if (!p || !s) { cad = false; return; }
    const est = Math.max(0.6, it.say_before.length / cfg.charsPerS), need = p.audioAt + Math.min(cfg.sayCapS, cfg.sayFrac * est);
    info.push({i, prompt: +p.at.toFixed(2), audio: +p.audioAt.toFixed(2), write: +s.toFixed(2), need: +need.toFixed(2)});
    if (s < need - 0.05 && s < p.endAt - 0.05) cad = false;               // not before ~70 % of the line (or its end)
    if (s - Math.max(need, R.now() * 0) > 6) cad = false;
    if (i > 0 && p.at < starts[items[i - 1].id]) cad = false;             // never two lines ahead of the writing
  });
  ok(cad, 'say -> write -> say -> write: each line before its item, writing from ~70 % of the line', info);
  ok(lines.every((p, i) => i === 0 || p.at - lines[i - 1].at > 1.0), 'lines spaced by the writing (no narration in one go)');
  ok(sawBoardWhere, 'action memory: "at the board" while there');
  ok(JSON.stringify(R.X.m.engine.state().holding).includes('chalk'), 'still holding the chalk after the lesson');
  const wy = writeYaw.length ? writeYaw.reduce((a, b) => a + b, 0) / writeYaw.length : 0;
  ok(turnedYaw !== null && Math.abs(turnedYaw) < Math.abs(wy) - 0.02, 'turns toward him when done', {writeYaw: +wy.toFixed(3), endYaw: turnedYaw});
  ok(R.teach._T.look === 'user', 'eyes / head on him when done');
  ok(R.prompts.some(p => /finished writing/.test(p.text)), 'hands the turn back with a note when done');
  // idle at the board -> drifts home
  R.teach.cfg.idleHomeS = 3;
  R.until(() => atHome(R), 30);
  ok(atHome(R) && props(R).chalk === 'home', 'idle at the board -> chalk back, walks home', props(R));
}

// ---- 2. no voice: motion never waits on speech
{
  const R = makeRig({noVoice: true});
  R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.stats().items === 2, 40);
  ok(R.teach.stats().items === 2, 'no voice: it still writes every item', R.teach.stats());
}

// ---- 3. a diagram wider than one reach: steps along the board between stroke groups, every stroke drawn
{
  const box = (u0, v0, u1, v1) => [[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]];
  const arrow = (u0, u1, v) => [[u0, v], [u1, v]];
  const strokes = [box(0.03, 0.3, 0.2, 0.55), arrow(0.21, 0.4, 0.42), box(0.41, 0.3, 0.58, 0.55), arrow(0.59, 0.78, 0.42), box(0.79, 0.3, 0.97, 0.55)];
  const R = makeRig({items: [{id: 'd1', kind: 'diagram', say_before: 'Here is the flow.', strokes, bbox: [0.03, 0.3, 0.97, 0.55]}]});
  R.tick(5); R.teach.call('teach_board', {request: 'draw the flow'});
  const xs = []; let steps = 0, lastMove = null;
  R.until(() => { const E = R.X.m.engine.E; if (E.move && E.move !== lastMove && R.teach._T.phase === 'writing') { steps++; lastMove = E.move; } if (R.teach._T.phase === 'writing') xs.push(E.base.x); return R.teach.stats().items === 1; }, 60);
  const segs = inkSegs(R).filter(s => !s.erase);
  ok(R.teach.stats().items === 1 && new Set(segs.map(s => s.stroke)).size === 5, 'diagram: all 5 strokes drawn', {strokes: new Set(segs.map(s => s.stroke)).size, misses: R.teach.stats().misses});
  ok(steps >= 2 && Math.max(...xs) - Math.min(...xs) > 0.12, 'diagram: it steps along the board (several stances)', {steps, span: +(Math.max(...xs) - Math.min(...xs)).toFixed(2)});
}

// ---- 4. erase: the eraser from the tray, wipes (erase marks), back on the tray, board state updated
{
  const R = makeRig();
  R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.stats().items === 2 && R.teach._T.phase === 'idle', 40);
  const r = R.teach.call('board_erase', {what: 'all'});
  ok(r.ok, 'board_erase answers', r);
  let sawEraserHeld = false;
  R.until(() => { if (JSON.stringify(R.X.m.engine.state().holding).includes('eraser')) sawEraserHeld = true; return R.teach.stats().erases >= 1 && R.teach._T.phase === 'idle' && !R.X.m.engine.E.jobs.length; }, 40);
  const er = inkSegs(R).filter(s => s.erase);
  ok(sawEraserHeld, 'erase: picks up the eraser');
  ok(er.length > 10, 'erase: wipe marks along the path', er.length);
  const us = er.map(s => s.pts[0][0]), vs = er.map(s => s.pts[0][1]);
  const res = boardResidue(inkSegs(R));   // erase: the felt's real stamps (swept) on the simulated hub canvas
  ok(res.residue < 0.02, 'erase all: the felt passes over every chalk mark (< 2% of the chalk left)', Object.assign(res, {u: [Math.min(...us), Math.max(...us)], v: [Math.min(...vs), Math.max(...vs)]}));
  // erase: the duster stays in the paw that does not write while it is at the board; home puts it back
  const P0 = props(R);
  ok(/paw/.test(P0.eraser) && P0.eraser !== P0.chalk, 'erase: the duster stays in the other paw at the board', P0);
  R.teach.call('go_home', {}); R.until(() => R.teach.state().at === 'home' && !R.X.m.engine.E.jobs.length, 30);
  ok(props(R).eraser === 'home' && props(R).chalk === 'home', 'erase, then home: the duster and the chalk are back on the tray', props(R));
  ok(R.sent.some(m => m.type === 'board_erased'), 'erase: the hub is told (board state)');
}

// ---- 5. abort at every phase
const res = {};
res.walking = abortAt('walking', R => R.now() > 6.5 && !!R.X.m.engine.E.move);
res.pickup = abortAt('picking up the chalk', R => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && /grasp|move paw/.test(C.cur.name) && R.teach._T.phase === 'going'); });
res.stroke = abortAt('mid-stroke', R => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && C.cur.name === 'write' && R.teach.stats().strokes >= 2); }, {words: 'stop'});
res.between = abortAt('between items', R => R.teach._T.phase === 'saying' && R.teach.stats().items >= 1, {how: 'tool'});
res.erasing = abortAt('erasing', R => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && C.cur.name === 'erase'); },
  {start: R => { R.teach.call('teach_board', {request: 'x'}); R.until(() => R.teach.stats().items === 2, 40); R.until(() => R.teach._T.phase === 'idle', 10); R.teach.call('board_erase', {what: 'all'}); }, words: 'forget it'});
res.stopcmd = abortAt('the stop command (fast matcher path)', R => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && C.cur.name === 'write'); }, {how: 'stopcmd'});
res.tv = abortAt('tv', R => R.teach._T.phase === 'presenting', {start: R => R.teach.goStation('tv'), words: 'come back'});
ok(true, 'abort -> home times (s): ' + JSON.stringify(Object.fromEntries(Object.entries(res).map(([k, v]) => [k, v.home_s]))));

// ---- 6. a stop word inside a longer sentence that is not a request to stop, and "don't stop"
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'}); R.tick(60);
  ok(R.teach.userText("don't stop, keep going", true) === false, '"don\'t stop" is not an abort');
  ok(R.teach.userText('so the train has to stop at every station on the way and then it goes on', false) === false, 'a long partial with "stop" in it is not an abort');
  ok(!!R.teach.state().req, 'still teaching');
}
// ---- 6b. he starts talking between items: nothing new starts until he stops (a teacher listens)
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.stats().items === 1, 40);
  R.voice.userTalking = true; const n0 = R.prompts.length, i0 = R.teach.stats().items; R.tick(90);
  const held = R.teach.stats().items === i0 && R.prompts.length === n0 && !R.teach._T.item;
  R.voice.userTalking = false; R.until(() => R.teach.stats().items === 2, 40);
  ok(held && R.teach.stats().items === 2, 'he talks between items: it holds (no line, no new item), then carries on', {held, items: R.teach.stats().items});
}
// ---- 7. cost: the director's frame is cheap
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  let worst = 0, sum = 0, k = 0; const f = R.teach.frame;
  R.teach.frame = (...a) => { const t0 = process.hrtime.bigint(); f(...a); const us = Number(process.hrtime.bigint() - t0) / 1000; sum += us; k++; worst = Math.max(worst, us); };
  R.until(() => R.teach.stats().items === 2, 40);
  ok(sum / k < 400, 'director frame mean < 400 µs (stance search included)', {mean_us: Math.round(sum / k), worst_us: Math.round(worst)});
}
// ---- 8. integration (field bugs): a request right after a stop / after tv_show is not stalled by the older job's end
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.stats().items >= 1, 40);
  R.teach.userText('stop', true);                         // walking home ...
  R.tick(20);
  R.teach.call('teach_board', {request: 'y'});              // ... and a new lesson before it got there
  const done = R.until(() => R.teach.state().req === null && R.teach.stats().items >= 3, 80);
  ok(done, 'a lesson asked for while it walks home after a stop is written (no stall)', {state: R.teach.state(), items: R.teach.stats().items});
}
{
  const R = makeRig(); R.tick(5);
  R.teach.goStation('tv', {present: true});
  R.teach.call('teach_board', {request: 'x'});
  const done = R.until(() => R.teach.stats().items >= 2, 80);
  const presAtBoard = R.X.ev.filter(e => /present/.test(e.name || '')).length;
  ok(done, 'tv_show + teach_board together: TV first, then the lesson is written (no stall)', {state: R.teach.state(), items: R.teach.stats().items, presAtBoard});
}
// ---- 9. a change of subject is soft: it stays at the board, a new request carries on from there; idle -> home
{
  const R = makeRig({cfg: {softHomeS: 3}}); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.state().phase === 'writing', 40);
  R.teach.onHub({type: 'teach_abort', why: 'he moved on: test'});
  const st = R.teach.state();
  ok(st.at === 'board' && st.req === null, 'topic change: lesson ends, it stays at the board', st);
  R.until(() => R.teach.state().at === 'home', 20);
  ok(R.teach.state().at === 'home', 'nothing new within softHomeS: it walks home', R.teach.state());
}
// ---- 10. his speech start lifts the chalk at once; the item resumes from its first unwritten stroke
{
  const R = makeRig({items: stubItems([{text: '12 + 30 = 42', say: 'Add them.'}])}); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.state().phase === 'writing' && R.teach.stats().strokes >= 2, 60);
  const s0 = R.teach.stats().strokes, ink0 = R.inks.length;
  R.voice.userTalking = true;
  const paused = R.teach.userSpeech('start');
  R.tick(30);
  const inkDuring = R.inks.length - ink0;
  R.voice.userTalking = false;
  R.until(() => R.teach.state().req === null, 60);
  const total = R.hub.items[0].strokes.length;
  ok(paused && inkDuring <= 1 && R.teach.stats().items === 1, 'speech start pauses writing within a frame, then the item completes', {paused, inkDuring, s0, strokes: R.teach.stats().strokes, total});
  ok(R.teach.stats().strokes <= total + 1, 'resume does not rewrite finished strokes', {strokes: R.teach.stats().strokes, total});
}
// ---- 11. integration: stop words + the voice's go_home both send it home: it ends AT home (was at=null)
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.state().phase === 'writing', 40);
  R.teach.userText('stop', true); R.tick(10); R.teach.call('go_home', {});
  R.until(() => !R.teach.state().jobs.length, 40);
  ok(R.teach.state().at === 'home', 'stop + go_home: it ends at home', R.teach.state());
}
// ---- 12. integration: idle at the TV, he asks for an object elsewhere: the station camera is let go
{
  const R = makeRig(); R.tick(5); R.teach.goStation('tv', {present: true});
  R.until(() => R.teach.state().at === 'tv' && !R.teach.state().jobs.length, 40);
  const views = []; const e = R.X.m.engine, sv = e.setView; e.setView = v => { views.push(v); return sv ? sv(v) : {ok: true}; };
  const res = e.command({action: 'object', verb: 'pick_up', object: {name: 'hat'}}, 'tool', 77);
  R.tick(30);
  ok(res && res.ok && R.teach.state().at === null && views.includes(null), 'object job away from the TV releases the station view', {at: R.teach.state().at, views});
}
// ---- 13. dev loop: mid-lesson he asks to hold the eraser up: the lesson yields, the eraser stays in the
// paw (no walk home that tidies it away), and nothing walks it home 40 s later either
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'x'});
  R.until(() => R.teach.state().phase === 'writing', 40);
  const e = R.X.m.engine;
  const r1 = e.command({action: 'object', verb: 'show', object: {name: 'eraser'}}, 'tool', 78);
  R.until(() => JSON.stringify(e.state().holding).includes('eraser'), 30);
  R.tick(30 * 50);
  const P = props(R), st = R.teach.state();
  ok(r1 && r1.ok && /paw/.test(P.eraser) && st.at !== 'home' && !String(st.at || '').startsWith('going'), 'his object request mid-lesson: the eraser stays in the paw, no tidy-up walk home', {eraser: P.eraser, at: st.at, r1});
  R.teach.call('go_home', {}); R.until(() => !R.teach.state().jobs.length, 40);
  ok(/paw/.test(props(R).eraser) && props(R).chalk === 'home', 'then home: the chalk goes back to the tray, HIS eraser stays in the paw', props(R));
  // a board_erase from the voice while a fresh lesson is in progress, without his words asking for it, is refused
  const R2 = makeRig(); R2.tick(5); R2.teach.call('teach_board', {request: 'draw a diagram'}); R2.tick(90);   // the plan is fully streamed (req.done) but not yet written
  const re = R2.teach.call('board_erase', {what: 'all'});
  ok(re && re.ok === false, 'erase right behind a fresh lesson (he did not ask) is refused', re);
}
// ---- 14. dev loop (chemistry stall): a new lesson asked for while the last one's page-turn wipe runs is written
{
  const items = stubItems([{text: '2 + 2', say: 'First.'}, {text: '= 4', say: 'Then.'}]);
  items.push({id: 'e1', kind: 'erase', say_before: 'Let me make some room.', regions: [[0, 0.3, 1, 0.8]]});
  items.push(...stubItems([{text: '3 + 3', say: 'Next.'}]).map(x => Object.assign(x, {id: 'x1'})));
  const R = makeRig({items}); R.tick(5); R.teach.call('teach_board', {request: 'a'});
  R.until(() => { const C = R.X.m.engine.E.chain; return !!(C && C.cur && C.cur.name === 'erase'); }, 120);
  R.voice.userTalking = true; if (R.teach.userSpeech) R.teach.userSpeech('start'); R.tick(60); R.voice.userTalking = false;
  R.hub.items = stubItems([{text: '5 + 5', say: 'Now five.'}, {text: '= 10', say: 'Ten.'}]);
  const n0 = R.teach.stats().items;
  R.teach.call('teach_board', {request: 'continue'});
  const done = R.until(() => R.teach.state().phase === 'idle' && !R.teach.state().req && !R.X.m.engine.E.jobs.length, 90);
  const hd = R.X.m.engine.state().holding;   // erase: the duster may stay in the other paw, never in the chalk's
  ok(done && R.teach.stats().items >= n0 + 2 && /chalk/.test(JSON.stringify(hd)) && !(hd.left === 'eraser' && hd.right === 'eraser') && Object.values(hd).filter(x => x === 'chalk').length === 1,
    'a lesson asked for during the last one\'s wipe: written, chalk in the paw, the duster not in the chalk paw', {done, items: R.teach.stats().items, n0, holding: hd});
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails ? 1 : 0);
