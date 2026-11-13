// Board visibility: stance study. For each (stance, pen point) the real teach director + motion engine walks there,
// takes the chalk and writes one short tick at the pen point; the rig frame mid-tick is kept (plus the IK error and the
// bend it needed). Render the rows with renderer/tools/board_vis_render.py and score with tools/board_stance_score.py.
//   node tools/board_stance_probe.cjs --out DIR [--grid '{"lat":[..],"d":[..],"yaw":[..],"u":[..],"v":[..]}']
'use strict';
const fs = require('fs'), path = require('path');
const L = require(path.join(__dirname, '..', 'tests', 'teach_lib.cjs'));
const a = {};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const G = Object.assign({lat: [0.2, 0.26, 0.32, 0.38], d: [0.18, 0.24, 0.3], yaw: [0, -0.04, -0.083, -0.13], u: [0.3], v: [0.38, 0.55, 0.75, 0.92]}, a.grid ? JSON.parse(a.grid) : {});
fs.mkdirSync(a.out, {recursive: true});
const rowsOut = [], meta = [];
for (const u of G.u) for (const v of G.v) for (const yaw of G.yaw) for (const d of G.d) for (const lat of G.lat) {
  let forced = null;
  const hook = (B, uvs, s, cur, kind, X) => {
    if (kind === 'eraser') return null;
    const P0 = B.world(u, v), b = [P0[0] + X.nf[0] * d + X.Uf[0] * lat, 0, P0[2] + X.nf[2] * d + X.Uf[2] * lat];
    const r = {x: b[0], y: 0, z: b[2], yaw};
    const q = X.reach(r, X.gripOf(B, P0, 0));
    forced = {x: r.x, z: r.z, yaw, cost: 0, bend: q.b.p, e: q.e};
    return forced;
  };
  const items = [{id: 'p', kind: 'math', say_before: null, strokes: [[[u, v], [u + 0.004, v + 0.01], [u + 0.008, v + 0.02], [u + 0.012, v + 0.03]]], bbox: [u, v, u + 0.012, v + 0.03], est_s: 1}];
  const R = L.makeRig({items, rows: true, cfg: {visTol: -1}, stanceHook: hook, brainLatency: 0.5});
  const T = R.teach._T;
  R.teach.call('teach_board', {request: 'probe'});
  let got = null, inkAt = null;
  for (let i = 0; i < 40 * 30; i++) {
    R.tick(1);
    if (inkAt === null && T.inkPts > 0) inkAt = R.now();
    if (inkAt !== null && R.now() - inkAt >= 0.12) {
      const k = R.X.m.engine.kit, E = k.E, r = R.X.rows[R.X.rows.length - 1], p = {};
      for (const kk of Object.keys(r)) if (!kk.startsWith('_') && kk !== 't' && typeof r[kk] === 'number') p[kk] = +r[kk].toFixed(5);
      got = {t: +r.t.toFixed(4), p, x: {pl: k.fwdW('l', E.paw.l.h), pr: k.fwdW('r', E.paw.r.h), bp: E.bend.p}};
      break;
    }
  }
  const m = {u, v, yaw, d, lat, ok: !!got, forced, root: got ? [got.p.root_x, got.p.root_z, got.p.root_yaw] : null};
  meta.push(m);
  if (got) rowsOut.push(JSON.stringify(got));
  else rowsOut.push(JSON.stringify({t: -1, p: {}, x: null}));
}
fs.writeFileSync(path.join(a.out, 'rows.jsonl'), rowsOut.join('\n') + '\n');
fs.writeFileSync(path.join(a.out, 'meta.json'), JSON.stringify(meta));
console.log('probes', meta.length, 'ok', meta.filter(m => m.ok).length);
