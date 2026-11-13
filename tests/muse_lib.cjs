// shared helpers for the muse tests: a motion controller with the move engine + the muse mocks (or the real JSON)
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const FIX = path.join(__dirname, 'fixtures');
function museFiles() {
  // default: the REAL files (renderer/animals); MUSE_MOCK=1 = the mocks
  const out = {body: JSON.parse(fs.readFileSync(path.join(FIX, 'muse_body_mock.json'))), props: JSON.parse(fs.readFileSync(path.join(FIX, 'muse_props_mock.json'))), src: 'mock'};
  const first = l => l.find(p => p && fs.existsSync(p));
  const A = path.join(__dirname, '..', 'renderer', 'animals');
  const bp = process.env.MUSE_BODY || (process.env.MUSE_MOCK ? null : first([path.join(A, 'muse.body.json')]));
  const pp = process.env.MUSE_PROPS || (process.env.MUSE_MOCK ? null : first([path.join(A, 'muse.props.json')]));
  if (bp && fs.existsSync(bp)) { out.body = JSON.parse(fs.readFileSync(bp)); out.src = 'real body'; }
  if (pp && fs.existsSync(pp)) { out.props = JSON.parse(fs.readFileSync(pp)); out.src += '+real props'; }
  return out;
}
function makeMuse(seed = 5, opts = {}) {
  const ev = [];
  const m = M.createMotion(Object.assign({seed, preset: 'default', onAction: e => ev.push(e)}, opts));
  const f = museFiles();
  m.setProps(f.props, f.body, 'muse');
  const face = {present: true, x: 0.05, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
  const rows = [];
  const DT = 1 / 30;
  function step(n = 1, rec = true) { for (let i = 0; i < n; i++) { const o = m.step(DT, {features: {t: m.now(), face, attending: true, hands: []}}); if (rec) { const E = m.engine.E; rows.push(Object.assign({t: m.now(), _feet: E.footNow ? {l: E.footNow.l.slice(), r: E.footNow.r.slice()} : null, _swing: E.move && E.move.steps ? E.move.steps.filter(st => m.now() - E.move.t0 >= st.t0 - 1e-6 && m.now() - E.move.t0 <= st.t1 + 1e-6).map(st => st.s) : [], _ride: Math.max(E.ride || 0, E.seatW || 0), _chain: E.chain && E.chain.cur ? E.chain.cur.name : null, _lw: E.look ? E.look.w : 0, _tb: E.tbW || 0, _act: E.activity ? (E.activity.paused ? 'paused' : 'typing') : null, _move: E.move ? E.move.name : null, _props: E.P.map(p => ({id: p.id, k: p.k, pos: p.pos.slice(), held: p.held, worn: !!p.worn, flying: !!p.flying, place: p.place, att: p.att.l + p.att.r, hh: p.hh || 0}))}, o)); } } }
  function runUntilIdle(maxS = 60) { const n0 = rows.length; for (let i = 0; i < maxS * 30; i++) { step(1); const E = m.engine.E; if (!E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue && i > 3) break; } return (rows.length - n0) / 30; }
  return {m, ev, rows, step, runUntilIdle, files: f, DT};
}
module.exports = {M, Moves, makeMuse, museFiles};
