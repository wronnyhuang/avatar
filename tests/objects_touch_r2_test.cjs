// objvoice: CONTACT with a thing (verb touch + styles: tap / knock / pat / poke / press / stroke / lean) and
// GO-TO, on the real engine (motion.js + moves.js, the real muse body + props JSON). Checks the world, not the words alone:
//   - the paw really touches: the DRAWN paw's surface (grasp point from the hand columns actually output, palm normal)
//     within 0.015 H of the thing's top / face at >= 2 separate moments (one for a single tap),
//   - the thing touched does not move (furniture never; a light loose thing is not knocked away),
//   - where it ends up standing, and the truthful `did` / refusal.
// Run: node tests/objects_touch_r2_test.cjs   (MUSE_ONLY=<substring> runs the matching scenarios only)
const {makeMuse} = require('./muse_lib.cjs');

let pass = 0, fail = 0;
const only = process.env.MUSE_ONLY || '';
function ok(name, cond, info) { if (cond) pass++; else fail++; console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${info !== undefined ? '  — ' + (typeof info === 'string' ? info : JSON.stringify(info)) : ''}`); }
const sc = (name, fn) => { if (only && !name.includes(only)) return; try { fn(name); } catch (e) { ok(name + ' (threw)', false, String(e && e.stack || e).slice(0, 400)); } };
const TOUCH = 0.015;

// run one command; `probe(X, E)` -> a signed gap (H) of the paw nearest the contact surface, or null when no paw is over it
function run(c, probe, maxS = 60, prep) {
  const T = makeMuse(7); T.step(20);
  const E = T.m.engine, X = E.E;
  if (prep) prep(X, E);
  const pos0 = Object.fromEntries(X.P.map(p => [p.id, {pos: p.pos.slice(), rot: p.rot.slice()}]));
  const n0 = T.ev.length, r = T.m.command(Object.assign({action: 'object'}, c), {src: 'test'});
  const touches = []; let inTouch = false, minGap = Infinity;
  for (let i = 0; i < maxS * 30; i++) {
    T.step(1);
    const g = probe ? probe(X, E) : null;
    if (g !== null && g !== undefined) { minGap = Math.min(minGap, Math.abs(g)); if (Math.abs(g) <= TOUCH) { if (!inTouch) touches.push(T.m.now()); inTouch = true; } else if (g > TOUCH + 0.005) inTouch = false; }
    if (!X.move && !X.chain && !X.jobs.length && i > 5) break;
  }
  for (let i = 0; i < 30; i++) T.step(1);
  const evs = T.ev.slice(n0), fin = evs.filter(e => e.kind === 'object' && /^(end|fail|interrupted)$/.test(e.phase)).pop() || null;
  const words = fin ? (fin.phase === 'fail' ? 'FAIL: ' + fin.reason : fin.did) : r && r.ok === false ? 'FAIL: ' + r.reason : '(no final event)';
  const moved = X.P.filter(p => Math.hypot(...p.pos.map((v, k) => v - pos0[p.id].pos[k])) > 1e-4 || Math.abs(p.rot[1] - pos0[p.id].rot[1]) > 1e-4).map(p => [p.id, +Math.hypot(...p.pos.map((v, k) => v - pos0[p.id].pos[k])).toFixed(4)]);
  return {T, E, X, r, fin, words, failed: (r && r.ok === false) || (fin && fin.phase === 'fail'), touches, minGap, moved, root: {x: X.base.x, z: X.base.z, yaw: X.base.yaw}, posture: X.posture};
}
// the paws' gap to a horizontal top (y) over a footprint test inside(x, z)
const topProbe = (yTop, inside) => (X, E) => {
  let best = null;
  for (const s of ['l', 'r']) { const g = X.pawGW && X.pawGW[s]; if (!g || !inside(g[0], g[2])) continue; const d = E._obj.pawSurfGap(s, [g[0], yTop, g[2]], [0, 1, 0]); if (d !== null && (best === null || Math.abs(d) < Math.abs(best))) best = d; }
  return best;
};
const rectInside = (c, w, d, yawT) => (x, z) => { const a = -yawT * 2 * Math.PI, dx = x - c[0], dz = z - c[2], lx = dx * Math.cos(a) + dz * Math.sin(a), lz = -dx * Math.sin(a) + dz * Math.cos(a); return Math.abs(lx) <= w / 2 && Math.abs(lz) <= d / 2; };
const deskOf = X => { const d = X.P.find(p => p.id === 'desk'), pl = X.places.find(q => q.id === 'desk_top'); return {d, pl, inside: rectInside(pl.center, pl.size[0], pl.size[1], d.homeRot[1] + d.rot[1])}; };
const distToRect = (c, w, dp, yawT, x, z) => { const a = -yawT * 2 * Math.PI, dx = x - c[0], dz = z - c[2], lx = dx * Math.cos(a) + dz * Math.sin(a), lz = -dx * Math.sin(a) + dz * Math.cos(a); return Math.hypot(Math.max(0, Math.abs(lx) - w / 2), Math.max(0, Math.abs(lz) - dp / 2)); };

sc('T1 knock on the desk from home: walks over, raps the top, the desk stays', name => {
  let D = null;
  const R = run({verb: 'knock', object: {name: 'desk'}}, (X, E) => { D = D || deskOf(X); return topProbe(D.pl.top_y, D.inside)(X, E); });
  const {d, pl} = deskOf(R.X);
  ok(`${name}: said truthfully`, !R.failed && /walked over to the desk/.test(R.words) && /knocked on the desk 3 times/.test(R.words), R.words);
  ok(`${name}: it walked (not knocking empty air at home)`, Math.hypot(R.root.x, R.root.z) > 0.3, R.root);
  ok(`${name}: the paw on the desk top (≤ ${TOUCH} H) at >= 2 moments`, R.touches.length >= 2, {touches: R.touches.map(v => +v.toFixed(2)), minGap: +R.minGap.toFixed(4)});
  ok(`${name}: nothing moved (the desk least of all)`, R.moved.length === 0, R.moved);
  ok(`${name}: ends standing near the desk`, R.posture === 'standing' && distToRect(pl.center, pl.size[0], pl.size[1], d.homeRot[1] + d.rot[1], R.root.x, R.root.z) < 0.45, R.root);
  ok(`${name}: a knock = quick raps (3 contacts, under 0.45 s apart)`, R.touches.length === 3 && R.touches.slice(1).every((t, i) => t - R.touches[i] < 0.45), R.touches);
});
sc('T2 pat the ball: the paw on its top, the ball stays put', name => {
  let b = null;
  const R = run({verb: 'pat', object: {name: 'ball'}}, (X, E) => { b = b || X.P.find(p => p.id === 'ball'); const c = E._obj.P(b); return topProbe(c[1] + b.hH, (x, z) => Math.hypot(x - c[0], z - c[2]) < b.hx + 0.02)(X, E); });
  ok(`${name}: said`, !R.failed && /patted the yellow ball 3 times/.test(R.words), R.words);
  ok(`${name}: the paw on the ball (≤ ${TOUCH} H) at >= 2 moments`, R.touches.length >= 2, {touches: R.touches.length, minGap: +R.minGap.toFixed(4)});
  const mv = R.moved.find(m => m[0] === 'ball');
  ok(`${name}: the ball did not fly off (moved ≤ 0.02 H)`, !mv || mv[1] <= 0.02, R.moved);
  ok(`${name}: nothing else moved`, R.moved.filter(m => m[0] !== 'ball').length === 0, R.moved);
});
sc('T3 tap the laptop (on the floor): it bends down, a finger-tap on the top', name => {
  let L = null;
  const R = run({verb: 'tap', object: {name: 'laptop'}}, (X, E) => { L = L || X.P.find(p => p.id === 'laptop'); const c = E._obj.P(L); return topProbe(c[1] + L.hH, (x, z) => Math.hypot(x - c[0], z - c[2]) < L.size)(X, E); });
  ok(`${name}: said`, !R.failed && /tapped the silver laptop twice/.test(R.words), R.words);
  ok(`${name}: the paw on it (≤ ${TOUCH} H) at >= 2 moments`, R.touches.length >= 2, {touches: R.touches.length, minGap: +R.minGap.toFixed(4)});
  ok(`${name}: nothing moved`, R.moved.length === 0, R.moved);
});
sc('T4 go to the board: at its station, facing it', name => {
  const R = run({verb: 'go_to', object: {name: 'board'}});
  const st = R.X.props.places.find(q => q.id === 'board').station;
  ok(`${name}: said`, !R.failed && /walked over to the chalkboard/.test(R.words), R.words);
  ok(`${name}: stands at the station`, Math.hypot(R.root.x - st.pos[0], R.root.z - st.pos[2]) < 0.04 && Math.abs(R.root.yaw - st.yaw) < 0.03, {root: R.root, st});
  ok(`${name}: nothing moved`, R.moved.length === 0, R.moved);
  const R2 = run({verb: 'approach', object: {name: 'desk'}});
  const dk = R2.X.P.find(p => p.id === 'desk').pos, fw = [Math.sin(2 * Math.PI * R2.root.yaw), Math.cos(2 * Math.PI * R2.root.yaw)], to = [dk[0] - R2.root.x, dk[2] - R2.root.z], tl = Math.hypot(...to);
  ok(`${name}: approach the desk = stand by it, facing it`, !R2.failed && /walked over to the desk/.test(R2.words) && (fw[0] * to[0] + fw[1] * to[1]) / tl > 0.8 && tl < 0.7 && R2.moved.length === 0, {w: R2.words, root: R2.root, facing: +((fw[0] * to[0] + fw[1] * to[1]) / tl).toFixed(2)});
  const R3 = run({verb: 'go_to', object: 'home'});
  ok(`${name}: go home while home = already there (truthful)`, !R3.failed && /already/.test(R3.words), R3.words);
});
sc('T5 knock over still topples; plain knock of a small loose thing still topples', name => {
  const R = run({verb: 'knock_over', object: {id: 'block_a'}});
  ok(`${name}: knock over the red block: it fell`, !R.failed && /knocked the red block over/.test(R.words) && R.moved.some(m => m[0] === 'block_a' && m[1] > 0.05), {w: R.words, moved: R.moved});
  const R2 = run({verb: 'knock', object: {id: 'block_b'}});
  ok(`${name}: knock the blue block: knocked over`, !R2.failed && /knocked the blue block over/.test(R2.words), R2.words);
  const R3 = run({verb: 'knock', object: {id: 'block_a'}, relation: 'on'}, null);
  ok(`${name}: knock ON the red block = knocking on it, nothing falls`, !R3.failed && /knocked on the red block/.test(R3.words) && R3.moved.length === 0, {w: R3.words, moved: R3.moved});
});
sc('T6 honest refusals: nothing there, out of reach, held', name => {
  const R = run({verb: 'touch', object: {name: 'piano'}});
  ok(`${name}: no piano: refused, it never moved`, R.failed && /no piano/.test(R.words) && Math.hypot(R.root.x, R.root.z) < 0.01, {w: R.words, root: R.root});
  const R2 = run({verb: 'tap', object: {name: 'tv'}}, null, 60, X => { const q = X.props.places.find(p => p.id === 'tv'); q.screen = Object.assign({}, q.screen, {origin: [q.screen.origin[0], 2.2, q.screen.origin[2]]}); });
  ok(`${name}: a screen far above any paw: refused (can't reach), nothing moved`, R2.failed && /can't reach/.test(R2.words) && R2.moved.length === 0, R2.words);
  const T = makeMuse(7); T.step(20); T.m.command({action: 'object', verb: 'pick_up', object: {id: 'ball'}}, {src: 'test'}); T.runUntilIdle(30);
  const n0 = T.ev.length, r = T.m.command({action: 'object', verb: 'pat', object: {id: 'ball'}}, {src: 'test'}); T.runUntilIdle(30);
  const fin = T.ev.slice(n0).filter(e => e.kind === 'object' && /^(end|fail)$/.test(e.phase)).pop();
  ok(`${name}: pat the ball it is holding: refused (in my paw)`, (r.ok === false && /in my paw/.test(r.reason)) || (fin && fin.phase === 'fail' && /in my paw/.test(fin.reason)), r.reason || (fin && fin.reason));
});
sc('T7 knock on the chalkboard: raps its face (a vertical surface)', name => {
  const pl = (() => { const T = makeMuse(7); return T.m.engine.E.props.places.find(q => q.id === 'board').plane; })();
  const n = pl.n, o = pl.origin;
  const R = run({verb: 'knock', object: {name: 'chalkboard'}, relation: 'on'}, (X, E) => {
    let best = null;
    for (const s of ['l', 'r']) { const g = X.pawGW && X.pawGW[s]; if (!g || g[1] < o[1] - pl.size[1] || g[1] > o[1]) continue; const d = E._obj.pawSurfGap(s, o, n); if (d !== null && Math.abs(d) < 0.08 && (best === null || Math.abs(d) < Math.abs(best))) best = d; }
    return best;
  });
  ok(`${name}: said`, !R.failed && /knocked on the chalkboard 3 times/.test(R.words), R.words);
  ok(`${name}: the paw on the board face (≤ ${TOUCH} H) at >= 2 moments`, R.touches.length >= 2, {touches: R.touches.length, minGap: +R.minGap.toFixed(4)});
  ok(`${name}: nothing moved`, R.moved.length === 0, R.moved);
});
sc('T8 styles + counts + a side: touch {style, count, side}', name => {
  let D = null;
  const R = run({verb: 'touch', object: {name: 'desk'}, style: 'pat', count: 2, side: 'left'}, (X, E) => { D = D || deskOf(X); return topProbe(D.pl.top_y, D.inside)(X, E); });
  ok(`${name}: pat twice, said so`, !R.failed && /patted the desk twice/.test(R.words) && R.touches.length === 2, {w: R.words, touches: R.touches.length});
  const R2 = run({verb: 'lean_on', object: {name: 'desk'}}, (X, E) => { D = deskOf(X); return topProbe(D.pl.top_y, D.inside)(X, E); });
  ok(`${name}: lean on the desk: a paw rests on it, the desk stays`, !R2.failed && /leaned on the desk/.test(R2.words) && R2.touches.length >= 1 && R2.moved.length === 0, {w: R2.words, touches: R2.touches.length, moved: R2.moved});
  const R3 = run({verb: 'stroke', object: {name: 'book'}});
  ok(`${name}: stroke the book: said, nothing moved`, !R3.failed && /stroked the blue book/.test(R3.words) && R3.moved.length === 0, {w: R3.words, moved: R3.moved});
});
sc('T9 a tool stays a tool: tap the plant WITH the watering can = use', name => {
  const R = run({verb: 'tap', object: {id: 'plant'}, with: 'watering can'});
  ok(`${name}: used the can on the plant`, !R.failed && /tapped the green watering can on the little green plant/.test(R.words), R.words);
});

sc('T10 a keyframe plan: do touch (knock on the desk), poses, do go_to the board — through motion.js', name => {
  const T = makeMuse(7); T.step(20);
  const E = T.m.engine, X = E.E;
  const r = T.m.command({action: 'keyframes', id: 'kt', description: 'knock on the desk then go to the board', done: true, hold: false, poses: [
    {t: 0.3, do: {verb: 'touch', object: 'desk', style: 'knock', count: 2}}, {t: 0.8, head_pitch: -0.2}, {t: 1.2, head_pitch: 0},
    {t: 1.5, do: 'go to board'}, {t: 2.0, right_paw_y: 0.5, right_paw_z: 0.4}, {t: 2.6, right_paw_y: 0}]}, {src: 'tool'});
  ok(`${name}: accepted`, r && r.ok !== false, r);
  let D = null; const touches = []; let inT = false, deskMoved = 0, root1 = null;
  for (let i = 0; i < 70 * 30; i++) {
    T.step(1);
    D = D || deskOf(X);
    const g = topProbe(D.pl.top_y, D.inside)(X, E);
    if (g !== null && Math.abs(g) <= TOUCH) { if (!inT) touches.push(T.m.now()); inT = true; } else if (g === null || g > TOUCH + 0.005) inT = false;
    const st = T.m.kfState('kt'); if (st && st.steps && st.steps.length === 1 && !root1) root1 = {x: X.base.x, z: X.base.z};
    if (st && /ended|failed/.test(st.phase) && !(st.w > 1e-4)) break;
  }
  deskMoved = Math.hypot(...D.d.pos.map((v, k) => v - D.d.home[k]));
  const end = T.ev.filter(e => e.kind === 'kf' && e.id === 'kt' && /^(end|fail|interrupted)$/.test(e.phase)).pop();
  const steps = end && end.steps || [];
  ok(`${name}: both steps ran and are reported ok`, steps.length === 2 && steps.every(s => s.ok) && /knocked on the desk twice/.test(steps[0].did || '') && /chalkboard/.test(steps[1].did || ''), steps.map(s => [s.verb, s.ok, s.did || s.reason]));
  ok(`${name}: the paw rapped the desk top twice, the desk stayed`, touches.length === 2 && deskMoved < 1e-4, {touches: touches.length, deskMoved});
  const st = X.props.places.find(q => q.id === 'board').station;
  ok(`${name}: ends at the board's station`, Math.hypot(X.base.x - st.pos[0], X.base.z - st.pos[2]) < 0.05, {x: X.base.x, z: X.base.z});
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
