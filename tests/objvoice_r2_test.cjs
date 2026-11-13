// objvoice: object requests from any state (after a lesson at the board, at the TV, seated) through the
// engine + the teach director (node, headless): his object request is his (no "did not work" note from the lesson's
// own tidy-up it cut), the lesson's tools go back before the hand is needed, a shown thing is presented big and toward the
// camera with the camera pushed in, and the action memory stays truthful. Exit code = failures.
'use strict';
const {makeRig} = require('./teach_lib.cjs');
let fails = 0, n = 0;
const VERBOSE = !!process.env.V;
function ok(c, name, info) { n++; if (c) console.log('PASS', name); else { fails++; console.log('FAIL', name, info !== undefined ? '— ' + JSON.stringify(info) : ''); } }
const props = R => Object.fromEntries(R.X.m.engine.props().map(p => [p.id, p.where]));
// his voice object call, the way page.html runs it (teach hook first, then the engine command)
function hisObject(R, cmd) {
  const c = Object.assign({action: 'object'}, cmd);
  if (R.teach.yieldTools) { try { R.teach.yieldTools(c); } catch (e) {} }
  return R.X.m.command(c, {src: 'voice'});
}
// sample the show segment: where the shown thing is (world), the camera, the head
function watch(R, id, maxS) {
  const k = R.X.m.engine.kit, E = k.E, out = {held: false, showing: 0, maxY: -1, zoomMin: 1, frontZ: -9, heldBy: null, chainNames: new Set()};
  for (let i = 0; i < maxS * 30; i++) {
    R.tick(1);
    const p = E.P.find(q => q.id === id), C = E.chain;
    if (C && C.cur) out.chainNames.add(C.cur.name);
    if (p && p.held) { out.held = true; out.heldBy = p.held; }
    const showing = !!(C && C.cur && /^(show|offer)$/.test(C.cur.name));
    if (showing && p) {
      const sd = p.held === 'both' ? 'l' : p.held, P = sd ? k.fwdW(sd, E.paw[sd].h) : p.pos;
      out.showing++; out.maxY = Math.max(out.maxY, P[1] - E.base.y); out.frontZ = Math.max(out.frontZ, P[2] - E.base.z);
      const L = R.X.m.lastRig ? R.X.m.lastRig() : null; void L;
      const o = E.lp || []; const z = (o.find(x => x[0] === 'cam_zoom') || [])[2];
      if (Number.isFinite(z)) out.zoomMin = Math.min(out.zoomMin, z);
    }
    if (i > 60 && !E.jobs.length && !E.chain && !E.move) break;
  }
  out.chainNames = [...out.chainNames];
  return out;
}
const bad = R => R.prompts.filter(P => /did not work|could not|didn't work/i.test(P.text) && P.o && P.o.respond).map(P => P.text.slice(0, 120));

// ---- 1. a lesson finished at the board (chalk in the paw), then "hold up the eraser so I can see it"
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'two plus two'});
  R.until(() => R.teach.state().phase === 'idle' && !R.teach.state().req && !R.X.m.engine.E.jobs.length, 90);
  const before = JSON.stringify(R.X.m.engine.state().holding);
  const p0 = R.prompts.length;
  const r = hisObject(R, {verb: 'show', object: {name: 'board eraser'}});
  const w = watch(R, 'eraser', 40);
  const head = R.X.m.engine.kit.E.body.head;
  const newBad = bad({prompts: R.prompts.slice(p0)});
  if (VERBOSE) console.log({before, r, w, head, props: props(R), prompts: R.prompts.slice(p0).map(P => P.text.slice(0, 100))});
  ok(r && r.ok !== false, 'board: show the eraser accepted', r);
  ok(!newBad.length, 'board: no "did not work" note while his request runs', newBad);
  ok(w.held && w.showing > 30, 'board: the eraser is held and shown for > 1 s', w);
  ok(props(R).chalk === 'home' || props(R).chalk === 'board_tray' || /tray|home/.test(props(R).chalk), 'board: the lesson\'s chalk went back to the tray first', props(R));
  ok(w.zoomMin < -0.3, 'board: the camera pushes in while it shows', w.zoomMin);
}
// ---- 2. at the TV after a lesson (chalk carried), "go grab the yellow ball and show it to me"
{
  const R = makeRig(); R.tick(5); R.teach.call('teach_board', {request: 'two plus two'});
  R.until(() => R.teach.state().phase === 'idle' && !R.teach.state().req && !R.X.m.engine.E.jobs.length, 90);
  R.teach.goStation('tv', {present: true});
  R.until(() => R.teach.state().at === 'tv' && !R.teach.state().jobs.length, 40);
  const p0 = R.prompts.length;
  const r = hisObject(R, {verb: 'show', object: {name: 'ball'}});
  const w = watch(R, 'ball', 40);
  if (VERBOSE) console.log({r, w, props: props(R), at: R.teach.state().at, prompts: R.prompts.slice(p0).map(P => P.text.slice(0, 100))});
  ok(r && r.ok !== false && w.held && w.showing > 30, 'tv: the ball is fetched and shown', {r, w});
  ok(!bad({prompts: R.prompts.slice(p0)}).length, 'tv: no "did not work" note', bad({prompts: R.prompts.slice(p0)}));
}
// ---- 3. seated, "show me the mug"
{
  const R = makeRig(); R.tick(5);
  R.X.m.command({action: 'object', verb: 'sit_in', object: {name: 'chair'}}, {src: 'voice'});
  R.until(() => R.X.m.engine.state().posture === 'seated' && !R.X.m.engine.E.jobs.length, 40);
  const r = hisObject(R, {verb: 'show', object: {name: 'mug'}});
  const w = watch(R, 'mug', 40);
  if (VERBOSE) console.log({r, w, props: props(R), posture: R.X.m.engine.state().posture});
  ok(r && r.ok !== false && w.held && w.showing > 30, 'seated: the mug is fetched and shown', {r, w});
}
// ---- 4. home, show a small thing: presented high and toward the camera
{
  const R = makeRig(); R.tick(5);
  const r = hisObject(R, {verb: 'show', object: {name: 'chalk'}});
  const w = watch(R, 'chalk', 40);
  const E = R.X.m.engine.kit.E, Hh = E.body.hi[1] - E.body.lo[1];
  if (VERBOSE) console.log({r, w, body: {lo: E.body.lo, hi: E.body.hi, head: E.body.head, front: E.body.front}});
  ok(w.showing > 30 && w.maxY > E.body.lo[1] + 0.62 * Hh, 'home: a small thing is held up high (shoulder / face height)', {maxY: w.maxY, lo: E.body.lo[1], Hh});
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails ? 1 : 0);
