// erase: timing probe (node) — a full board, then (A) "erase the board" and (B) a new lesson whose plan starts
// with a page-turn erase. Prints the step timeline and the numbers. `node tests/teach_erase_probe.cjs [A|B] [json cfg]`
'use strict';
const {makeRig, stubItems} = require('./teach_lib.cjs');
const which = process.argv[2] || 'AB', cfg = process.argv[3] ? JSON.parse(process.argv[3]) : {};
function fullItems() {   // three columns of four lines each (the brain's layout: columns flow down the reach band)
  const L = [];
  for (const [ci, u] of [0.04, 0.37, 0.70].entries()) for (let r = 0; r < 4; r++) L.push({text: ['3x + 2 = 11', '3x = 9', 'x = 3', '9 - 4 = 5'][(r + ci) % 4], u, v: 0.36 + 0.145 * r, h: 0.1, say: 'Line ' + (ci * 4 + r) + '.'});
  return stubItems(L).map((x, i) => Object.assign(x, {id: 'f' + i}));
}
const inkSegs = R => R.sent.filter(m => m.type === 'board_ink').reduce((a, m) => a.concat(m.segs), []);
function fill(R) {
  R.tick(5); R.teach.call('teach_board', {request: 'fill'});
  R.until(() => R.teach.stats().items >= 12 && R.teach._T.phase === 'idle' && !R.X.m.engine.E.jobs.length, 400);
  R.tick(30);
}
const HID = {n: 0, hid: 0};
function timeline(R, until, maxS) {
  const tl = []; let last = '', t0 = R.now(), si = R.sent.length;
  for (let i = 0; i < maxS * 30; i++) {
    R.tick(1);
    // how much of the wipe the camera sees: felt stamps behind the body (the body's shadow on the board, from the camera)
    const b = R.X.m.engine.E.base, h = R.teach._hiddenU(b.x, b.z);
    for (; si < R.sent.length; si++) { const m = R.sent[si]; if (m.type !== 'board_ink') continue; for (const sg of m.segs) if (sg.erase) { HID.n++; if (h && sg.pts[0][0] > h[0] && sg.pts[0][0] < h[1]) HID.hid++; } }
    const E = R.X.m.engine.E, nm = (E.chain && E.chain.cur ? E.chain.cur.name : '-') + (E.move ? ' [walk]' : '') + ' ' + R.teach._T.phase;
    if (nm !== last) { tl.push([+(R.now() - t0).toFixed(2), nm]); last = nm; }
    if (until()) break;
  }
  return {tl, s: +(R.now() - t0).toFixed(2)};
}
// the hub's board canvas, simulated (app/teach_hub.py _ink + surfaces): chalk lines, then the felt's stamps (a swept
// stamp = rects along its segment); residue = the chalk left after the wipe / the chalk before, by pixel coverage
function residue(R, region) {
  const W = 512, H = 256, A = new Float32Array(W * H), segs = inkSegs(R);
  const rect = (u0, v0, u1, v1, f) => { for (let y = Math.max(0, Math.floor(v0 * H)); y <= Math.min(H - 1, Math.ceil(v1 * H)); y++) for (let x = Math.max(0, Math.floor(u0 * W)); x <= Math.min(W - 1, Math.ceil(u1 * W)); x++) f(y * W + x); };
  for (const sg of segs) {
    if (sg.erase) {
      const hu = sg.ru * 0.8, hv = sg.rv, P = sg.pts;
      const pts = P.length >= 2 && sg.sweep ? (() => { const out = []; const n = Math.max(1, Math.ceil(Math.max(Math.abs(P[1][0] - P[0][0]) / hu, Math.abs(P[1][1] - P[0][1]) / hv) * 1.25)); for (let i = 0; i <= n; i++) out.push([P[0][0] + (P[1][0] - P[0][0]) * i / n, P[0][1] + (P[1][1] - P[0][1]) * i / n]); return out; })() : P;
      for (const c of pts) rect(c[0] - hu, c[1] - hv, c[0] + hu, c[1] + hv, i => { A[i] *= 0.03; });
    } else {
      for (let i = 0; i + 1 < sg.pts.length || i === 0; i++) {
        const a = sg.pts[i], b = sg.pts[Math.min(i + 1, sg.pts.length - 1)], n = Math.max(1, Math.ceil(Math.hypot((b[0] - a[0]) * W, (b[1] - a[1]) * H)));
        for (let j = 0; j <= n; j++) { const u = a[0] + (b[0] - a[0]) * j / n, v = a[1] + (b[1] - a[1]) * j / n; rect(u - 0.0055 / 2, v - 0.0055, u + 0.0055 / 2, v + 0.0055, q => { A[q] = 1; }); }
        if (sg.pts.length < 2) break;
      }
    }
  }
  let left = 0, all = 0; const miss = [];
  // chalk ever drawn (alpha set) vs left: count pixels drawn at all by re-running lines only
  const D = new Uint8Array(W * H);
  for (const sg of segs) if (!sg.erase) for (let i = 0; i + 1 < sg.pts.length || i === 0; i++) { const a = sg.pts[i], b = sg.pts[Math.min(i + 1, sg.pts.length - 1)], n = Math.max(1, Math.ceil(Math.hypot((b[0] - a[0]) * W, (b[1] - a[1]) * H))); for (let j = 0; j <= n; j++) { const u = a[0] + (b[0] - a[0]) * j / n, v = a[1] + (b[1] - a[1]) * j / n; rect(u - 0.0055 / 2, v - 0.0055, u + 0.0055 / 2, v + 0.0055, q => { D[q] = 1; }); } if (sg.pts.length < 2) break; }
  for (let i = 0; i < W * H; i++) if (D[i]) { const u = (i % W) / W, v = Math.floor(i / W) / H; if (region && !(u >= region[0] && u <= region[2] && v >= region[1] && v <= region[3])) continue; all++; left += A[i]; if (A[i] > 0.5 && miss.length < 2000) miss.push([+u.toFixed(2), +v.toFixed(2)]); }
  const cells = {}; for (const [u, v] of miss) { const key = (Math.floor(u * 10) / 10).toFixed(1) + ',' + (Math.floor(v * 10) / 10).toFixed(1); cells[key] = (cells[key] || 0) + 1; }
  return {residue: +(left / Math.max(1, all)).toFixed(4), px: all, cells};
}
function cover(R, since) {   // fraction of chalk points the felt passed over (pad = ru x rv half-sizes, as the hub stamps)
  const segs = inkSegs(R), er = segs.filter(s => s.erase && s._i >= since), wrote = segs.filter(s => !s.erase).flatMap(s => s.pts);
  const hit = wrote.filter(([u, v]) => er.some(e => Math.abs(e.pts[0][0] - u) < e.ru * 0.8 && Math.abs(e.pts[0][1] - v) < e.rv)).length;
  return +(hit / Math.max(1, wrote.length)).toFixed(3);
}
if (which.includes('A')) {
  const R = makeRig({items: fullItems(), brainLatency: 0.4, itemGap: 0.1, cfg});
  fill(R);
  inkSegs(R).forEach((s, i) => s._i = 0);
  const t0 = R.now();
  R.teach.call('board_erase', {what: 'all'});
  const r = timeline(R, () => R.sent.some(m => m.type === 'board_erased') && R.teach._T.phase === 'idle', 60);
  const er = R.sent.find(m => m.type === 'board_erased');
  inkSegs(R).forEach(s => { if (s._i === undefined) s._i = 1; });
  console.log("stats", JSON.stringify(R.teach.stats()));
  console.log('A erase-all: board_erased at', er ? +(er.at - t0).toFixed(2) : null, 's; idle at', r.s, 's; residue', JSON.stringify(residue(R)), 'stamps hidden', HID.hid + '/' + HID.n);
  for (const x of r.tl) console.log('  ', x[0], x[1]);
}
if (which.includes('B')) {
  HID.n = 0; HID.hid = 0;
  const R = makeRig({items: fullItems(), brainLatency: 0.4, itemGap: 0.1, cfg});
  fill(R);
  const items = [{id: 'e1', kind: 'erase', say_before: 'Let me clear some space.', bbox: [0.03, 0.34, 0.97, 0.92], target: ['all']}].concat(stubItems([{text: '5 + 5', say: 'Five and five.', u: 0.06}, {text: '= 10', say: 'Ten.', u: 0.06}]).map((x, i) => Object.assign(x, {id: 'n' + i})));
  R.hub.items = items; R.hub.latency = 1.5;
  R.teach._T.stats.firstMark = null;
  const t0 = R.now();
  R.teach.call('teach_board', {request: 'a new subject'});
  const r = timeline(R, () => R.teach._T.firstMark && R.teach._T.firstMark.id === R.teach._T.req?.id, 60);
  console.log('B new lesson on a full board: first mark', R.teach._T.firstMark, 'after', +(R.now() - t0).toFixed(2));
  for (const x of r.tl) console.log('  ', x[0], x[1]);
  const n0 = R.sent.length, oldSegs = new Set(R.sent.filter(m => m.type === 'board_ink').flatMap(m => m.segs).filter(x => !x.erase));
  const r2 = timeline(R, () => R.teach._T.phase === 'idle' && !R.teach._T.req, 60);
  const {boardResidue} = require('./teach_lib.cjs');
  const all = R.sent.filter(m => m.type === 'board_ink').flatMap(m => m.segs);
  console.log('B lesson done at', +(R.now() - t0).toFixed(2), 's; old chalk left', JSON.stringify(boardResidue(all.filter(x => x.erase || oldSegs.has(x)), x => oldSegs.has(x))),
    'new chalk left', JSON.stringify(boardResidue(all, x => !oldSegs.has(x))), 'stamps hidden', HID.hid + '/' + HID.n, 'erased msgs', JSON.stringify(R.sent.filter(m => m.type === 'board_erased').map(m => [m.item, m.partial || false, m.regions.length])));
  for (const x of r2.tl) console.log('  ', x[0], x[1]);
}
