#!/usr/bin/env node
/* Tests for app/react_feat.js + perception.js canon (learned-reaction mode).   node tests/react_feat_test.cjs [--live DIR] [--quick]
 *  1. synthetic edge cases: JS canonFrame vs the Python spec reference (tests/react_feat_ref.py canon), every column
 *  2. live dumps (tests/react_feat_dump.cjs live -> DIR/live/<clip>.json, real MediaPipe JS on real clips): the browser's
 *     canon frames vs the Python reference from the SAME raw worker vectors (+ the dataset's mp_raw_to_canon itself),
 *     node canonFrame == browser canonFrame bit-for-bit, and the page's 'H' bytes decoded in Python
 *  3. packHuman / unpackHuman round trip, header layout, clamping
 *  4. replay of the recorded worker messages through perception.js in node: v1.4.1 perception == canon:false ==
 *     canon:true minus f.canon (byte-identical JSON feature streams), and replayed canon frames == the browser's
 *  5. cost of canonFrame (+ bsToSpec) and packHuman per call
 * Python: $PYTHON -I (numpy + react/spec.py from a learned-reaction checkout: REACT_ROOT or SPEC_DIR; without it the test
 * SKIPs). Exit code 1 on any failure. */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process');
const load = require('./load.cjs');
const ROOT = path.resolve(__dirname, '..');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const SCRATCH = arg('out', path.join(require('os').tmpdir(), 'avatar-tests', 'canon'));
const LIVE = arg('live', path.join(SCRATCH, 'live'));
const PY = process.env.PYTHON || 'python3';
const SPEC_DIR = process.env.SPEC_DIR || (process.env.REACT_ROOT ? path.join(process.env.REACT_ROOT, 'react') : '');
if (!SPEC_DIR || !fs.existsSync(path.join(SPEC_DIR, 'spec.py'))) { console.log('SKIP react_feat_test: needs the learned-reaction spec (set REACT_ROOT or SPEC_DIR)'); process.exit(0); }
const BASE_REF = process.env.BASE_REF || 'v1.4.1';
const RF = load('app/react_feat.js');
const fails = [];
const check = (name, ok, info) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info != null ? '  ' + (typeof info === 'string' ? info : JSON.stringify(info)) : ''}`); if (!ok) fails.push(name); };
const TOL = 1e-4;
fs.mkdirSync(SCRATCH, {recursive: true});

function py(mode, files) {
  const r = cp.spawnSync('nice', ['-n', '10', PY, '-I', path.join(__dirname, 'react_feat_ref.py'), mode, ...files],
    {encoding: 'utf8', env: Object.assign({}, process.env, {OMP_NUM_THREADS: '3', OPENBLAS_NUM_THREADS: '3', MKL_NUM_THREADS: '3'}), maxBuffer: 1 << 28});
  const line = (r.stdout || '').split('\n').find(l => l.startsWith('RESULT '));
  if (!line) throw Error('python ref failed: ' + (r.stderr || '').slice(-2000));
  return JSON.parse(line.slice(7));
}
const groupsOk = (mx, tol) => Object.values(mx).every(v => typeof v === 'number' && v < tol);

// ── deterministic PRNG + synthetic raw MediaPipe-like inputs ──
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const U = (a, b) => a + (b - a) * rnd();
function rotColMajor(yaw, pitch, roll, t) {      // R = Ry(yaw) Rx(pitch) Rz(roll), stored column-major like MediaPipe JS
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch), cz = Math.cos(roll), sz = Math.sin(roll);
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  const mul = (A, B) => A.map((r, i) => B[0].map((_, j) => r.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
  const R = mul(mul(Ry, Rx), Rz), M = new Float32Array(16);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[4 * j + i] = R[i][j];
  M[12] = t[0]; M[13] = t[1]; M[14] = t[2]; M[15] = 1;
  return M;
}
const NAMES_SHUF = RF.BS_NAMES.slice().sort((a, b) => (a.length * 7 + a.charCodeAt(2)) % 13 - (b.length * 7 + b.charCodeAt(2)) % 13 || (a < b ? -1 : 1));
function face(opts = {}) {
  const bs = new Float32Array(52); for (let i = 0; i < 52; i++) bs[i] = rnd() ** 2;
  return {bs, mat: opts.noMat ? null : rotColMajor(U(-0.8, 0.8), U(-0.6, 0.6), U(-0.5, 0.5), [U(-8, 8), U(-6, 6), U(-90, -25)])};
}
function pose(opts = {}) {      // 33*4 normalized; his left shoulder (11) at LARGER raw x unless opts.mirror
  const lm = new Float32Array(132);
  for (let i = 0; i < 33; i++) { lm[4 * i] = U(0.2, 0.8); lm[4 * i + 1] = U(0.2, 0.9); lm[4 * i + 2] = U(-0.4, 0.4); lm[4 * i + 3] = U(0.6, 1); }
  const cx = U(0.4, 0.6), cy = U(0.55, 0.75), hw = U(0.08, 0.2) * (opts.mirror ? -1 : 1);
  const set = (i, x, y) => { lm[4 * i] = x; lm[4 * i + 1] = y; };
  set(11, cx + hw, cy); set(12, cx - hw, cy + U(-0.02, 0.02));
  set(13, cx + 1.3 * hw, cy + U(0.1, 0.2)); set(14, cx - 1.3 * hw, cy + U(0.1, 0.2));
  set(15, cx + U(0.5, 1.5) * hw, cy + U(-0.3, 0.3)); set(16, cx - U(0.5, 1.5) * hw, cy + U(-0.3, 0.3));
  if (opts.degenerate) { set(12, lm[44], lm[45]); lm[50] = lm[46]; }
  for (const [i, v] of opts.vis || []) lm[4 * i + 3] = v;
  return lm;
}
function hand(wx, wy, opts = {}) {      // 21*3 normalized, wrist at (wx, wy)
  const lm = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    if (opts.collapsed) { lm[3 * i] = wx; lm[3 * i + 1] = wy; lm[3 * i + 2] = 0; continue; }
    const f = i === 0 ? 0 : Math.floor((i - 1) / 4), k = i === 0 ? 0 : (i - 1) % 4 + 1, open = opts.open == null ? rnd() : opts.open;
    lm[3 * i] = wx + (f - 2) * 0.012 + U(-0.003, 0.003); lm[3 * i + 1] = wy - k * 0.02 * (0.4 + open) + U(-0.003, 0.003); lm[3 * i + 2] = U(-0.05, 0.05);
  }
  return lm;
}
function mkCase(name, w, h, f, p, hs) {
  const names = NAMES_SHUF;
  const wbs = f ? Float32Array.from(names, n => f.bs[RF.BS_NAMES.indexOf(n)]) : null;    // worker order
  const inp = {face: f && f.mat ? {bs: RF.bsToSpec(wbs, names), matColMajor: f.mat} : null, pose: p ? {lm: p} : null, hands: hs.map(lm => ({lm})), w, h};
  const r = RF.canonFrame(inp), t = U(0, 5000);
  return {name, w, h, t, face: f ? {bs: Array.from(wbs), mat: f.mat ? Array.from(f.mat) : null} : null, pose: p ? {lm: Array.from(p)} : null,
    hands: hs.map(lm => ({lm: Array.from(lm)})), js: {motion: Array.from(r.motion), mask: Array.from(r.mask), tcm: r.tcm},
    packed: Buffer.from(RF.packHuman(t, r.motion, r.mask)).toString('base64')};
}
function syntheticCases() {
  const C = [];
  const wristPx = (p, i) => [p[4 * i], p[4 * i + 1]];
  for (let k = 0; k < 40; k++) { const p = pose(), [lx, ly] = wristPx(p, 15), [rx, ry] = wristPx(p, 16); C.push(mkCase('random_full_' + k, 320, 180, face(), p, [hand(lx + 0.01, ly), hand(rx - 0.01, ry)])); }
  C.push(mkCase('nothing', 320, 180, null, null, []));
  C.push(mkCase('face_only', 320, 240, face(), null, []));
  C.push(mkCase('face_no_matrix', 320, 180, face({noMat: true}), pose(), []));
  C.push(mkCase('no_pose_two_hands_ignored', 320, 180, face(), null, [hand(0.3, 0.5), hand(0.7, 0.5)]));
  C.push(mkCase('pose_only_no_face', 320, 180, null, pose(), []));
  { const p = pose(), [lx, ly] = wristPx(p, 15); C.push(mkCase('one_hand_at_left_wrist', 320, 180, face(), p, [hand(lx, ly)])); }
  { const p = pose(), [rx, ry] = wristPx(p, 16); C.push(mkCase('one_hand_at_right_wrist', 320, 180, face(), p, [hand(rx, ry)])); }
  { const p = pose(), [lx, ly] = wristPx(p, 15), [rx, ry] = wristPx(p, 16); C.push(mkCase('two_hands_listed_r_then_l', 320, 180, face(), p, [hand(rx, ry), hand(lx, ly)])); }
  { const p = pose(), [lx, ly] = wristPx(p, 15), [rx, ry] = wristPx(p, 16);
    p[60] = rx; p[61] = ry; p[64] = lx; p[65] = ly;   // arms crossed: pose wrists swapped across the body
    C.push(mkCase('two_hands_crossed', 320, 180, face(), p, [hand(lx, ly), hand(rx, ry)])); }
  { const p = pose(), [lx, ly] = wristPx(p, 15), [rx, ry] = wristPx(p, 16); C.push(mkCase('three_hands_first_two_used', 320, 180, face(), p, [hand(lx, ly), hand(rx, ry), hand(0.5, 0.5)])); }
  C.push(mkCase('invisible_elbows_wrists', 320, 180, face(), pose({vis: [[13, 0.2], [14, 0.49], [15, 0.1], [16, 0.0]]}), []));
  C.push(mkCase('vis_exactly_half', 320, 180, face(), pose({vis: [[11, 0.5], [12, 0.5], [13, 0.5]]}), []));
  C.push(mkCase('one_shoulder_invisible', 320, 180, face(), pose({vis: [[12, 0.3]]}), []));
  C.push(mkCase('one_shoulder_invisible_mirrored', 320, 180, face(), pose({vis: [[11, 0.3]], mirror: true}), []));
  C.push(mkCase('mirrored_person', 320, 180, face(), pose({mirror: true}), []));
  C.push(mkCase('degenerate_shoulders', 320, 180, face(), pose({degenerate: true}), [hand(0.5, 0.5)]));
  { const p = pose(), [lx, ly] = wristPx(p, 15); C.push(mkCase('collapsed_hand', 320, 180, face(), p, [hand(lx, ly, {collapsed: true})])); }
  { const p = pose(), [lx, ly] = wristPx(p, 15); C.push(mkCase('portrait_frame', 180, 320, face(), p, [hand(lx, ly, {open: 1})])); }
  { const p = pose(), [lx, ly] = wristPx(p, 15), [rx, ry] = wristPx(p, 16); C.push(mkCase('fist_and_open', 640, 360, face(), p, [hand(lx, ly, {open: 0}), hand(rx, ry, {open: 1})])); }
  { const f = face(); f.bs.fill(0); f.mat = rotColMajor(0, 0, 0, [0, 0, -50]); C.push(mkCase('identity_head_zero_bs', 320, 180, f, pose(), [])); }
  { const f = face(); f.mat = rotColMajor(Math.PI * 0.49, 0.1, 0.2, [3, -2, -40]); C.push(mkCase('yaw_near_90deg', 320, 180, f, null, [])); }
  return C;
}

// ── 1. synthetic edge cases vs Python ──
{
  const cases = syntheticCases(), fp = path.join(SCRATCH, 'synthetic_cases.json');
  fs.writeFileSync(fp, JSON.stringify({names: NAMES_SHUF, cases}));
  const R = py('canon', [fp]).total;
  check(`synthetic ${cases.length} cases: JS vs Python spec, max abs diff < ${TOL} per group`, groupsOk(R.max, TOL), R.max);
  check('synthetic: masks + tcm equal', R.mask_mismatch === 0, {mask_mismatch: R.mask_mismatch, mask_counts: R.mask_counts});
  check('synthetic: H decoded in Python, err <= 5e-5 (cm cols <= 0.005 cm), flags exact', R.h_max <= 5e-5 + 1e-9 && R.h_cm_max <= 0.005 + 1e-9 && R.h_flags_bad === 0 && R.h_t_err === 0,
    {max_err: R.h_max, max_err_cm: R.h_cm_max, clamped: R.h_clamped, t_err: R.h_t_err});
  const byName = Object.fromEntries(cases.map(c => [c.name, c.js]));
  check('edge semantics: no face -> cols 0-59 zero + mask 0; no matrix -> face absent; no pose -> hands ignored + rest body',
    byName.nothing.motion.slice(0, 60).every(v => v === 0) && byName.nothing.mask.join('') === '0000' && byName.face_no_matrix.mask[0] === 0 &&
    byName.no_pose_two_hands_ignored.mask.join('') === '1000' && byName.nothing.motion[60] === 0.5 && byName.nothing.tcm === null);
  check('edge semantics: crossed arms assign by pose wrists (hand 0 -> r, hand 1 -> l); listed r-then-l handled',
    JSON.stringify(RF.assignHandsToWrists([[10, 0], [0, 0]], [0, 0], [10, 0])) === '{"l":1,"r":0}' && byName.two_hands_crossed.mask.join('') === '1111' &&
    byName.two_hands_listed_r_then_l.mask.join('') === '1111');
}

// ── 2. live dumps ──
const dumps = fs.existsSync(LIVE) ? fs.readdirSync(LIVE).filter(f => /\.json$/.test(f) && !/nocanon/.test(f)).map(f => path.join(LIVE, f)) : [];
let perfInputs = [];
if (!dumps.length) console.log(`SKIP live dumps (none in ${LIVE}; run tests/react_feat_dump.cjs live)`);
else {
  const R = py('canon', dumps), T = R.total;
  const nHands = T.mask_counts[2] + T.mask_counts[3];
  check(`live: ${dumps.length} clips, ${T.frames} canon frames (>= 300), hand frames ${nHands} (> 0)`, T.frames >= 300 && nHands > 0 && dumps.length >= 2, {mask_counts: T.mask_counts});
  check(`live: browser canonFrame vs Python spec on the same raw vectors, max abs diff < ${TOL} per group`, groupsOk(T.max, TOL), T.max);
  check('live: masks + tcm equal', T.mask_mismatch === 0, {mask_mismatch: T.mask_mismatch});
  const vs = Object.values(R.files).map(f => f.vs_mp_raw_to_canon);
  if (vs.every(v => typeof v === 'string')) console.log('SKIP live vs mp_raw_to_canon: ' + vs[0]);
  else check(`live: vs the dataset's own mp_raw_to_canon (head_t through its median/10), < ${TOL}`, vs.every(v => typeof v === 'object' && groupsOk(v, TOL)) &&
    Object.values(R.files).every(f => f.vs_mp_raw_to_canon_mask_mismatch === 0), vs);
  check('live: page H bytes decoded in Python (err <= 5e-5, cm <= 0.005, flags + t exact)', T.h_max <= 5e-5 + 1e-9 && T.h_cm_max <= 0.005 + 1e-9 && T.h_flags_bad === 0 && T.h_t_err < 1e-9,
    {max_err: T.h_max, max_err_cm: T.h_cm_max, clamped: T.h_clamped, t_err: T.h_t_err});
  for (const fp of dumps) console.log('   ' + path.basename(fp) + ' ' + JSON.stringify(R.files[path.basename(fp)].max_abs_diff));
  // node canonFrame on the same raw inputs == the browser's output (bit-exact)
  let nd = 0, n = 0;
  for (const fp of dumps) {
    const d = JSON.parse(fs.readFileSync(fp, 'utf8'));
    for (const c of d.canon) {
      const s = c.src, src = s.pose || s.hand || s.face;
      const inp = {face: s.face ? {bs: RF.bsToSpec(Float32Array.from(s.face.bs), d.names), matColMajor: Float32Array.from(s.face.mat)} : null,
        pose: s.pose ? {lm: Float32Array.from(s.pose.lm)} : null, hands: s.hand ? s.hand.hands.map(h => ({lm: Float32Array.from(h.lm)})) : [], w: src.w, h: src.h};
      perfInputs.push({raw: s.face ? Float32Array.from(s.face.bs) : null, names: d.names, inp});
      const r = RF.canonFrame(inp);
      n++; if (r.motion.some((v, i) => !Object.is(v, c.motion[i])) || r.mask.some((v, i) => v !== c.mask[i])) nd++;
    }
  }
  check('live: node canonFrame == browser canonFrame bit-for-bit', nd === 0, `${n - nd}/${n}`);
}

// ── 3. pack / unpack ──
{
  let maxE = 0, maxCm = 0, bad = 0;
  for (let k = 0; k < 2000; k++) {
    const m = new Float32Array(86); for (let i = 0; i < 86; i++) m[i] = U(-3.2, 3.2);
    m[55] = U(-30, 30); m[56] = U(-30, 30); m[57] = U(-200, -20);
    const mask = Uint8Array.from([rnd() < 0.5, rnd() < 0.5, rnd() < 0.5, rnd() < 0.5].map(Number)), t = U(0, 1e5);
    const b = RF.packHuman(t, m, mask), u = RF.unpackHuman(b);
    if (b.length !== 184 || b[0] !== 0x48 || u.n !== 86 || u.t !== t || u.mask.join('') !== mask.join('') || b[1] !== (mask[0] | mask[1] << 1 | mask[2] << 2 | mask[3] << 3)) bad++;
    for (let i = 0; i < 86; i++) { const e = Math.abs(u.motion[i] - m[i]); if (i >= 55 && i <= 57) maxCm = Math.max(maxCm, e); else maxE = Math.max(maxE, i >= 60 && i <= 77 ? e / 10 : e); }   // body x1e3
  }
  check('pack/unpack: 2000 random frames, 184 B, header exact, err <= 5e-5, cm <= 0.005', bad === 0 && maxE <= 5e-5 + 1e-7 && maxCm <= 0.005 + 1e-5, {maxE, maxCm, bad});
  const m = new Float32Array(86); m[0] = 99; m[1] = -99; m[2] = NaN; m[3] = Infinity; m[55] = 1e6; m[56] = -1e6;
  const u = RF.unpackHuman(RF.packHuman(1.5, m, [1, 0, 0, 1]));
  check('pack: clamp +-32767, NaN/Inf -> 0, cm clamp +-327.67', u.motion[0] === Math.fround(3.2767) && u.motion[1] === Math.fround(-3.2767) && u.motion[2] === 0 && u.motion[3] === 0 &&
    Math.abs(u.motion[55] - 327.67) < 1e-4 && Math.abs(u.motion[56] + 327.67) < 1e-4 && u.flags === 9, Array.from(u.motion.slice(0, 4)).concat([u.motion[55], u.motion[56], u.flags]));
  const dv = new DataView(RF.packHuman(2.25, m, [0, 1, 0, 0]).buffer);
  check('pack: header <BBHd> = 0x48, flags, uint16 86, float64 t', dv.getUint8(0) === 0x48 && dv.getUint8(1) === 2 && dv.getUint16(2, true) === 86 && dv.getFloat64(4, true) === 2.25);
}

// ── 4. replay: canon:false leaves the feature stream byte-identical; canon:true adds only f.canon ──
function replay(src, wmsg, popts) {
  const FAKES = [];
  class FakeWorker { constructor(u) { this.url = u; this.onmessage = null; this.onerror = null; FAKES.push(this); } postMessage(m) { if (m && m.type === 'init') this.model = m.model; } terminate() {} }
  const video = {muted: true, playsInline: true, srcObject: null, readyState: 4, videoWidth: 640, videoHeight: 360, currentTime: 0, play: () => Promise.resolve()};
  const G = {Worker: FakeWorker, location: {href: 'http://replay.local/x.html'}, document: {createElement: () => video},
    requestAnimationFrame: () => 0, cancelAnimationFrame: () => {}, fetch: async () => ({ok: true, arrayBuffer: async () => new ArrayBuffer(4)}),
    navigator: {hardwareConcurrency: 8, mediaDevices: {getUserMedia: async () => ({getTracks: () => [{stop() {}}]})}}};
  for (const k in G) Object.defineProperty(globalThis, k, {value: G[k], configurable: true, writable: true});
  URL.createObjectURL = () => 'blob:replay'; URL.revokeObjectURL = () => {};
  const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports);
  const feats = [], canon = [];
  const rep = (k, v) => ArrayBuffer.isView(v) ? Array.from(v) : v;
  const P = mod.exports.createPerception(Object.assign({pose: true, preview: false, onFeatures: f => feats.push(f.canon ? {s: JSON.stringify(Object.assign({}, f, {canon: undefined}), rep), seq: f.canon.seq} : {s: JSON.stringify(f, rep), seq: null}),
    onCanon: c => canon.push({t: c.t, seq: c.seq, motion: Array.from(c.motion), mask: Array.from(c.mask)})}, popts));
  return (async () => {
    await P.start();
    for (let i = 0; i < 2000 && FAKES.length < 3; i++) await new Promise(r => setImmediate(r));
    const de = v => v && typeof v === 'object' && v.__t ? new globalThis[v.__t](v.a) : v;
    for (const m0 of wmsg) {
      const m = {}; for (const k in m0) m[k] = de(m0[k]);
      if (m.hands) m.hands = m.hands.map(h => Object.assign({}, h, {lm: de(h.lm)}));
      const w = FAKES.find(x => x.model === m.model);
      if (w && w.onmessage) w.onmessage({data: m});
    }
    return {feats, canon, stats: P.stats()};
  })();
}
(async () => {
  if (!dumps.length) console.log('SKIP replay (no live dumps)');
  else {
    const base = cp.execFileSync('git', ['-C', ROOT, 'show', BASE_REF + ':app/perception.js']).toString();
    const cur = fs.readFileSync(path.join(ROOT, 'app/perception.js'), 'utf8');
    for (const fp of dumps) {
      const d = JSON.parse(fs.readFileSync(fp, 'utf8'));
      const A = await replay(base, d.wmsg, {}), B = await replay(cur, d.wmsg, {}), Cc = await replay(cur, d.wmsg, {canon: true, reactFeat: RF});
      const same = (x, y) => x.length === y.length && x.every((f, i) => f.s === y[i].s);
      const nm = path.basename(fp, '.json');
      check(`replay ${nm}: ${BASE_REF} perception == canon:false (${A.feats.length} feature frames, byte-identical JSON)`, A.feats.length > 100 && same(A.feats, B.feats));
      check(`replay ${nm}: canon:true minus f.canon == canon:false; no f.canon / stats.canon when off`, same(B.feats, Cc.feats) && B.feats.every(f => f.seq === null) && !('canon' in B.stats) && Cc.stats.canon.n > 0);
      const bc = d.canon, rc = Cc.canon;
      const eq = bc.length === rc.length && bc.every((c, i) => c.seq === rc[i].seq && c.t === rc[i].t && c.mask.join() === rc[i].mask.join() && c.motion.every((v, j) => Object.is(Math.fround(v), Math.fround(rc[i].motion[j]))));
      check(`replay ${nm}: replayed canon frames == the browser's (${rc.length} vs ${bc.length})`, eq);
      const lastSeq = Cc.feats.filter(f => f.seq != null).length;
      const mono = rc.every((c, i) => i === 0 || c.t > rc[i - 1].t);
      check(`replay ${nm}: f.canon on ${lastSeq}/${Cc.feats.length} feature frames, canon t strictly increasing`, lastSeq >= Cc.feats.length - 5 && mono);
    }
  }
  // ── 5. cost ──
  {
    if (!perfInputs.length) perfInputs = syntheticCases().map(c => ({raw: c.face ? Float32Array.from(c.face.bs) : null, names: NAMES_SHUF,
      inp: {face: c.face && c.face.mat ? {bs: null, matColMajor: Float32Array.from(c.face.mat)} : null, pose: c.pose ? {lm: Float32Array.from(c.pose.lm)} : null, hands: c.hands.map(h => ({lm: Float32Array.from(h.lm)})), w: c.w, h: c.h}}));
    const N = 50000, L = perfInputs.length;
    for (let i = 0; i < 2000; i++) { const x = perfInputs[i % L]; RF.canonFrame(x.inp); }
    let t0 = process.hrtime.bigint(), sink = 0;
    for (let i = 0; i < N; i++) { const x = perfInputs[i % L]; if (x.inp.face) x.inp.face.bs = RF.bsToSpec(x.raw, x.names); sink += RF.canonFrame(x.inp).motion[60]; }
    const us = Number(process.hrtime.bigint() - t0) / 1e3 / N;
    t0 = process.hrtime.bigint();
    const mm = new Float32Array(86).fill(0.3), mk = [1, 1, 0, 1];
    for (let i = 0; i < N; i++) sink += RF.packHuman(i, mm, mk)[5];
    const usP = Number(process.hrtime.bigint() - t0) / 1e3 / N;
    check(`cost: bsToSpec + canonFrame ${us.toFixed(1)} us/call (< 200 us), packHuman ${usP.toFixed(1)} us/call`, us < 200, `${L} distinct inputs, ${N} calls${sink ? '' : ''}`);
  }
  console.log(`\n${fails.length ? 'FAILED: ' + fails.join('; ') : 'all passed'}`);
  process.exit(fails.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
