// node tests/kf_test.cjs [--verbose] — streamed keyframe jobs (docs/PLANNER.md §3, §5) on app/motion.js.
// Recorded REAL Sol streams (<KF_SOL>/<id>/stream.json: every pose with the ms it arrived) are fed to
// m.kfFeed() in their real arrival timing at 30 fps, exactly as the planner bench's exec_stream.cjs simulated them:
//  (a) parity: the muse's rig vs the recorded rig.jsonl (exec_stream.cjs output, buffer 2:1, same seed / features / engine)
//  (b) the same streams on fox, orb and cube (no root, no legs): no exceptions, motion on the channels the poses touch,
//      byte-identical to a run without the job once it has blended back
//  (c) interruption by 'stop', by a new move, by a newer keyframe job; 'hold' pauses it
//  (d) ownership weights rise / fall; a listener's values win on face / gaze / posture only where no commanded job owns them
//  (e) talking while it moves (muse, fox, orb): body channels keep their trajectory, the mouth follows speech
// Exit code = number of failures. KF_SOL=<dir> overrides the recordings folder.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const ANIM = path.join(__dirname, '..', 'renderer', 'animals');
const SOL = process.env.KF_SOL || path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'planner_runs', 'stream_kf-sol');
const KFW_OUT = 0.4, DT = 1 / 30, VERBOSE = process.argv.includes('--verbose');
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const PROPS = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), BODY = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json')));
const wrap = x => x - Math.round(x);
const PER = /^root_(yaw|pitch|roll)$/;

// a character as the page runs it: the muse with its body + scene, every other character without (seated, no engine)
function makeChar(name, seed = 4286) {
  const ev = [];
  const m = M.createMotion({seed, character: name, onAction: e => ev.push(e)});
  if (name === 'muse') m.setProps(JSON.parse(JSON.stringify(PROPS)), JSON.parse(JSON.stringify(BODY)), 'muse');
  else m.setProps(null, null, name);
  let t = 0;
  // exec_stream.cjs harness(): a still, attentive face; extra inputs (voices) per frame
  const step = (extra = {}) => { t += DT; return Object.assign({}, m.step(DT, Object.assign({features: {t, face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}}, extra))); };
  return {m, ev, step, name, now: () => t};
}
function loadStream(id) {
  const S = JSON.parse(fs.readFileSync(path.join(SOL, id, 'stream.json'), 'utf8'));
  const P = S.parse.poses.map(p => ({ms: p.ms, pose: Object.assign({t: p.t}, p.set)}));
  let rig = null;
  const rp = path.join(SOL, id, 'rig.jsonl');
  if (fs.existsSync(rp)) {
    const L = fs.readFileSync(rp, 'utf8').trim().split('\n').map(l => JSON.parse(l));
    rig = {names: L[0].rig_params, buffer: L[0].buffer, rows: L.slice(1).map(r => r.a), metrics: S.rig && S.rig.metrics};
  }
  return {id, request: S.text_request, P, complete: S.parse.complete_ms, rig};
}
// warm 1 s, command at t = 0 (hold:false = start as soon as the 2:1 rule holds), feed by arrival ms, step n frames
function runStream(C, st, n, o = {}) {
  for (let i = 0; i < 30; i++) C.step(o.voice ? o.voice(-1) : {});
  const id = o.id || 'k1';
  if (o.kf !== false) {
    const r = C.m.command({action: 'keyframes', id, description: st.request, poses: [], done: false, hold: !!o.hold}, {src: 'tool'});
    if (!r.ok) throw Error('command refused: ' + JSON.stringify(r));
  }
  const rows = []; let k = 0, done = false;
  for (let i = 0; i < n; i++) {
    const ms = i * DT * 1000;
    if (o.kf !== false) {
      const batch = []; while (k < st.P.length && st.P[k].ms <= ms) batch.push(st.P[k++].pose);
      if (batch.length) C.m.kfFeed(id, batch, false);
      if (!done && ms >= st.complete) { C.m.kfFeed(id, [], true); done = true; }
    }
    if (o.at) o.at(i, C);
    rows.push(C.step(o.voice ? o.voice(i) : {}));
  }
  return rows;
}
const kfEvents = C => C.ev.filter(e => e.kind === 'kf');
// frame index of an event: the command is at motion t = 1.0 s (30 warm frames) and frame i is stepped to t = 1 + (i + 1)·DT
const frameOf = e => e ? Math.round((e.t - 1.0) / DT) - 1 : null;
const startFrame = C => frameOf(kfEvents(C).find(x => x.phase === 'start'));
const fmt = x => (Math.abs(x) < 5e-5 ? '0' : x.toFixed(4));

const PARITY = ['n02_jump7', 'f01_cartwheel', 'm12_runplace', 'n14_tango', 'f03_pushups'];
const KEY = ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z',
  'hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z', 'body_pitch', 'body_lean', 'body_yaw', 'head_yaw', 'head_pitch', 'head_roll', 'smile',
  'mouth_open', 'cam_zoom', 'cam_x', 'cam_y', 'gaze_free'];
const BODY_CH = KEY.filter(c => !/^(smile|mouth_open|cam_|gaze_free)/.test(c));
if (!fs.existsSync(SOL)) { console.log('SKIP recorded Sol streams missing (set KF_SOL=<dir>): ' + SOL); process.exit(0); }
const streams = {};
for (const id of PARITY.concat(['n10_chain3'])) streams[id] = loadStream(id);

// ------------------------------------------------------------------ (a) parity with exec_stream.cjs on the muse
// raw = |live − recorded|. The live job's command resets the boredom clock (every command counts as interaction) while
// exec_stream.cjs never commanded its engine, so the idle layers drift apart after the command; the KF composition itself is
// compared as a CONTRIBUTION: (live − the same run with a held job that never plays) vs (recorded − the run with no command;
// that run reproduces the recorded idle rows exactly on every channel the poses do not touch, checked below).
console.log('== (a) parity: muse KF job vs the recorded rig.jsonl (exec_stream.cjs, buffer 2:1, seed 4286)');
const parityAgg = {};
for (const id of PARITY) {
  const st = streams[id]; if (!st.rig) { check(`parity ${id}: rig.jsonl present`, false); continue; }
  const n = st.rig.rows.length, ix = n2 => st.rig.names.indexOf(n2);
  const C = makeChar('muse'), rows = runStream(C, st, n);
  const Cr = makeChar('muse'), ref = runStream(Cr, Object.assign({}, st, {P: [], complete: 1e12}), n, {hold: true});   // boredom reset, never plays
  const Cb = makeChar('muse'), base = runStream(Cb, st, n, {kf: false});
  const rec = st.rig.rows.map(r => Object.fromEntries(st.rig.names.map((k, i) => [k, r[i]])));
  const untouched = ['blink_l', 'tail_wag', 'tail_raise', 'ear_l_turn', 'breath'].filter(c => ix(c) >= 0);
  const baseDiff = Math.max(...untouched.map(c => Math.max(...base.map((r, i) => Math.abs((r[c] || 0) - rec[i][c])))));
  check(`parity ${id}: harness reproduces the recorded idle base (untouched channels max |diff| ${baseDiff.toExponential(1)})`, baseDiff < 2e-4);
  const s0 = startFrame(C), recStart = st.rig.metrics ? Math.round(st.rig.metrics.t_start_ms / 1000 * 30) : null;
  const endEv = kfEvents(C).find(e => e.phase === 'end'), endF = endEv ? frameOf(endEv) : n;
  check(`parity ${id}: playback starts on the recorded frame (${recStart}) and ends`, s0 === recStart && !!endEv, `live start frame ${s0}, end frame ${endF}`);
  const df = (a, b, c) => PER.test(c) ? Math.abs(wrap(a - b)) : Math.abs(a - b);
  let worst = 0, worstC = '';
  for (const c of KEY) {
    const raw = rows.map((r, i) => df(r[c] || 0, rec[i][c], c));
    const con = rows.map((r, i) => df(PER.test(c) ? wrap((r[c] || 0) - (ref[i][c] || 0)) : (r[c] || 0) - (ref[i][c] || 0), PER.test(c) ? wrap(rec[i][c] - (base[i][c] || 0)) : rec[i][c] - (base[i][c] || 0), c));
    const seg = (v, a, b) => v.slice(Math.max(0, a), Math.max(0, Math.min(v.length, b)));
    const mx = v => v.length ? Math.max(...v) : 0;
    const A = parityAgg[c] || (parityAgg[c] = {raw: 0, rawSum: 0, nn: 0, con: 0, conSum: 0, play: 0, tail: 0});
    A.raw = Math.max(A.raw, mx(raw)); A.rawSum += raw.reduce((x, y) => x + y, 0); A.nn += raw.length;
    A.con = Math.max(A.con, mx(con)); A.conSum += con.reduce((x, y) => x + y, 0);
    const play = mx(seg(con, s0 ?? 0, endF + 1)), tail = mx(seg(con, endF + 1, n));
    A.play = Math.max(A.play, play); A.tail = Math.max(A.tail, tail);
    if (c !== 'cam_zoom' && play > worst) { worst = play; worstC = c; }
    if (VERBOSE) console.log(`   ${id} ${c.padEnd(11)} raw max ${fmt(mx(raw))}  contribution max ${fmt(mx(con))} (playback ${fmt(play)}, after the end ${fmt(tail)})`);
  }
  check(`parity ${id}: KF contribution matches exec_stream during playback on every body channel (max |diff| ${worst.toExponential(2)} on ${worstC || '-'})`, worst < 2e-3);
}
console.log('   per channel over the ' + PARITY.length + ' streams (rig.jsonl rounded to 1e-4): raw max / raw mean | contribution max / mean | during playback | after the end');
for (const c of KEY) { const A = parityAgg[c]; if (A) console.log(`   ${c.padEnd(11)} ${fmt(A.raw).padStart(7)} ${fmt(A.rawSum / A.nn).padStart(7)} | ${fmt(A.con).padStart(7)} ${fmt(A.conSum / A.nn).padStart(7)} | ${fmt(A.play).padStart(7)} | ${fmt(A.tail).padStart(7)}`); }
check('parity: camera zoom within 0.03 of exec_stream during playback (live: the engine camera is the not-zooming target; exec: the rest snapshot)', parityAgg.cam_zoom && parityAgg.cam_zoom.play < 0.03, `max ${parityAgg.cam_zoom && parityAgg.cam_zoom.play.toFixed(4)}`);

// ------------------------------------------------------------------ helpers for (b)–(e)
// the avatar's voice level as the page feeds it (syllables at 4 / s) for frames a..b-1
const talk = (a, b) => i => i >= a && i < b ? {avatarVoice: {rms: 0.02 + 0.06 * Math.max(0, Math.sin(2 * Math.PI * 4 * i * DT)), voiced: true, pitch: 180}} : {};
const ZERO = c => /^(root_|squash$|foot_|cam_|gaze_free$|stand$)/.test(c);
// rig channels a stream drives on a character: its direct columns (+ the seated body mapping of root controls)
function touchedOf(st, legs) {
  const T = new Set();
  for (const {pose} of st.P) for (const k of Object.keys(pose)) {
    if (k === 't' || !M.KF_ALIAS[k]) continue;
    for (const c of M.KF_ALIAS[k]) {
      if (c[0] === '@') { if (legs) T.add(c.slice(1)); continue; }
      if (!legs && /^root_|^squash$/.test(c)) { const m = {root_y: 'body_bounce', squash: 'body_bounce', root_yaw: 'body_yaw', root_pitch: 'body_pitch', root_roll: 'body_lean'}[c]; if (m) T.add(m); continue; }
      T.add(c);
    }
  }
  return [...T];
}
// a reference run: the same command (so the boredom clock resets the same way) but a job that never gets poses / plays
const refRun = (name, st, n, o = {}) => { const C = makeChar(name, o.seed); return {C, rows: runStream(C, Object.assign({}, st, {P: [], complete: 1e12}), n, Object.assign({}, o, {hold: true, id: 'ref'}))}; };
const maxAbs = (A, B, c, a = 0, b = A.length) => { let m = 0; for (let i = Math.max(0, a); i < Math.min(b, A.length); i++) { const d = PER.test(c) ? Math.abs(wrap((A[i][c] || 0) - (B[i][c] || 0))) : Math.abs((A[i][c] || 0) - (B[i][c] || 0)); if (d > m) m = d; } return m; };
const ALLCH = M.RIG_PARAMS.concat(['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'gaze_free', 'cam_zoom', 'cam_x', 'cam_y', 'foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z', 'fingers_l', 'fingers_r']);

// ------------------------------------------------------------------ (b) the same streams on characters without legs / root
console.log('== (b) the recorded muse streams on fox, orb, cube (seated: no root, no legs) + every character smoke');
const B_STREAMS = PARITY.concat(['n10_chain3']);
for (const name of ['fox', 'orb', 'cube']) {
  for (const id of B_STREAMS) {
    const st = streams[id], n = Math.round(((st.rig ? st.rig.rows.length / 30 : 12) + 1.5) * 30);
    let rows, err = null, C;
    try { C = makeChar(name); rows = runStream(C, st, n); } catch (e) { err = e; }
    if (err) { check(`${name} ${id}: runs without an exception`, false, String(err && err.stack || err).slice(0, 200)); continue; }
    const R = refRun(name, st, n).rows;
    const s0 = startFrame(C), endE = kfEvents(C).find(e => e.phase === 'end'), e0 = frameOf(endE);
    const tch = touchedOf(st, false), moved = tch.filter(c => maxAbs(rows, R, c, s0 ?? 0, e0 ?? n) > 0.01);
    const lacks = Object.keys(st.P.reduce((a, p) => Object.assign(a, p.pose), {})).filter(k => /_foot_|^root_[xz]$/.test(k));
    const zeroOk = ALLCH.filter(ZERO).every(c => rows.every(r => !r[c]));
    const back = e0 !== null ? Math.max(...ALLCH.map(c => maxAbs(rows, R, c, e0 + Math.ceil(0.45 * 30) + 1, n))) : 1;
    check(`${name} ${id}: start + end, moves ${moved.length}/${tch.length} touched channels, ignores ${lacks.length ? lacks.join(' ') : 'nothing'} (root/feet/camera stay 0), back to the no-job frame after the blend`,
      s0 !== null && !!endE && moved.length >= Math.ceil(0.75 * tch.length) && zeroOk && back === 0,
      `start ${s0} end ${e0}; not moved: ${tch.filter(c => !moved.includes(c)).join(' ') || '-'}; max |diff| after the blend ${back}`);
  }
}
{ // every character: one stream with feet, travel, turns and face (n14_tango) — no exception, it plays and ends
  const st = streams.n14_tango, n = 600, bad = [];
  for (const name of ['fox', 'bear', 'rabbit', 'hippo', 'monkey', 'orb', 'cube', 'drop', 'star', 'pyramid', 'blob', 'muse']) {
    try { const C = makeChar(name); runStream(C, st, n); const ph = kfEvents(C).map(e => e.phase).join(','); if (ph !== 'ready,start,end') bad.push(name + ':' + ph); }
    catch (e) { bad.push(name + ': ' + String(e).slice(0, 80)); }
  }
  check('all 12 characters play n14_tango: ready,start,end, no exception', !bad.length, bad.join('; '));
}

// ------------------------------------------------------------------ (c) interruption
console.log('== (c) interruption: stop, a new move, a newer keyframe job, hold');
for (const name of ['muse', 'fox']) {
  const st = streams.n02_jump7, n = 330;
  const probe = makeChar(name); runStream(probe, st, n); const s0 = startFrame(probe);
  const at = s0 + 60;
  // stop
  const C = makeChar(name), rows = runStream(C, st, n, {at: (i, X) => { if (i === at) X.m.command({action: 'stop'}, {src: 'tool'}); }});
  const R = refRun(name, st, n, {at: (i, X) => { if (i === at) X.m.command({action: 'stop'}, {src: 'tool'}); }}).rows;
  const iv = kfEvents(C).find(e => e.phase === 'interrupted');
  const skip = c => name === 'muse' && /^cam_/.test(c);   // the engine camera continues from the job's (its own spring)
  const back = Math.max(...ALLCH.filter(c => !skip(c)).map(c => maxAbs(rows, R, c, at + Math.ceil(0.45 * 30) + 1, n)));
  const during = Math.max(...['hand_l_y', 'smile', name === 'muse' ? 'root_y' : 'body_bounce'].map(c => maxAbs(rows, R, c, at - 30, at)));
  const camBack = name === 'muse' ? Math.max(...['cam_zoom', 'cam_x', 'cam_y'].map(c => maxAbs(rows, R, c, at + 90, n))) : 0;
  check(`${name}: 'stop' interrupts the job (event + reason), it blends back within ${KFW_OUT} s to exactly the no-job frame`, !!iv && iv.reason === 'stopped' && during > 0.05 && back === 0 && camBack < 0.02,
    `interrupted at frame ${frameOf(iv)}: "${iv && iv.did}"; max |diff| after the blend ${back}${name === 'muse' ? `, camera ${camBack.toFixed(4)} after 3 s` : ''}`);
  // a new move
  const C2 = makeChar(name); runStream(C2, st, at + 60, {at: (i, X) => { if (i === at) X.lastR = X.m.command({action: 'move', name: 'jump'}, {src: 'tool'}); }});
  const iv2 = kfEvents(C2).find(e => e.phase === 'interrupted'), mv = C2.ev.find(e => e.kind === 'move' && (e.phase === 'start' || e.phase === 'queued'));
  const gone = !C2.m.debug().kf.length || C2.m.debug().kf.every(j => j.w === 0);
  check(`${name}: a new move interrupts it and runs`, !!iv2 && /new move/.test(iv2.reason) && !!C2.lastR && C2.lastR.ok && gone && (name !== 'muse' || !!mv),
    `reason "${iv2 && iv2.reason}"; move result ${JSON.stringify(C2.lastR && {ok: C2.lastR.ok, did: C2.lastR.did})}`);
}
{ // a newer keyframe job: one still buffering is replaced at once; one playing goes on until the newer one starts
  const C = makeChar('fox'), st = streams.n02_jump7;
  for (let i = 0; i < 30; i++) C.step();
  C.m.command({action: 'keyframes', id: 'a', description: 'jump seven times', poses: [], hold: false});
  C.m.command({action: 'keyframes', id: 'b', description: 'wave', poses: [], hold: false});
  const ia = kfEvents(C).find(e => e.id === 'a' && e.phase === 'interrupted');
  check('a newer request replaces a job still being written ("did not start")', !!ia && /did not start/.test(ia.did), ia && ia.did);
  C.m.kfFeed('b', st.P.map(p => p.pose), true);
  for (let i = 0; i < 90; i++) C.step();
  const before = C.m.kfState('b').phase;
  C.m.command({action: 'keyframes', id: 'c', description: 'nod twice', poses: [{t: 0.3, head_pitch: 0.2}, {t: 0.6, head_pitch: -0.1}, {t: 0.9, head_pitch: 0.2}, {t: 1.2, head_pitch: 0}], done: true, hold: true});
  for (let i = 0; i < 10; i++) C.step();
  const stillB = C.m.kfState('b').phase;
  C.m.kfRelease('c'); C.step(); C.step();
  const ib = kfEvents(C).find(e => e.id === 'b' && e.phase === 'interrupted'), sc = kfEvents(C).find(e => e.id === 'c' && e.phase === 'start');
  check('a playing job goes on while the newer one waits (held), and is interrupted when the newer one starts', before === 'playing' && stillB === 'playing' && !!ib && !!sc && ib.t === sc.t,
    `b ${before} -> ${stillB}; b interrupted at ${ib && ib.t}, c started at ${sc && sc.t}`);
}
{ // hold = pause; release = end (blend back)
  const C = makeChar('fox'), st = streams.n02_jump7;
  for (let i = 0; i < 30; i++) C.step();
  C.m.command({action: 'keyframes', id: 'h', description: 'jump seven times', poses: st.P.map(p => p.pose), done: true, hold: false});
  for (let i = 0; i < 45; i++) C.step();
  const o0 = C.step();
  C.m.command({action: 'hold'}); const p0 = C.m.kfState('h').played_s; const held = []; for (let i = 0; i < 30; i++) held.push(C.step()); const p1 = C.m.kfState('h').played_s;
  const jump = Math.max(...['hand_l_y', 'hand_r_y', 'hand_l_x', 'body_bounce', 'smile'].map(c => Math.max(...held.map(o => Math.abs(o[c] - o0[c])))));
  C.m.command({action: 'release', parts: 'all'}); const ph = C.m.kfState('h');
  check("'hold that' pauses the job's clock where it is (its channels stay put); 'relax' ends it", p1 - p0 < 0.03 && jump < 0.08 && ph.phase === 'ended' && ph.interrupted && ph.reason === 'relaxed',
    `played ${p0} -> ${p1} s over 1 s held, its channels moved at most ${jump.toFixed(3)} (idle life); then ${ph.phase} (${ph.reason})`);
}

// ------------------------------------------------------------------ (d) ownership + the listener hook
console.log('== (d) ownership weights, the listener hook');
{ // muse: a KF job started while an engine move runs blends in (W_IN); its weights fall to 0 over W_OUT after the end
  const C = makeChar('muse');
  for (let i = 0; i < 30; i++) C.step();
  C.m.command({action: 'move', name: 'spin'}, {src: 'tool'}); for (let i = 0; i < 15; i++) C.step();
  const ow0 = C.m.owned();
  C.m.command({action: 'keyframes', id: 'w', description: 'jump twice', poses: [{t: 0.3, root_y: -0.05, squash: 0.12, right_paw_y: 0.9}, {t: 0.6, root_y: 0.4, squash: -0.08}, {t: 0.9, root_y: 0, squash: 0.12}, {t: 1.4, root_y: 0.4}, {t: 1.7, root_y: 0}], done: true, hold: false});
  const rise = [], fall = []; let endT = null;
  for (let i = 0; i < 120; i++) {
    C.step(); const o = C.m.owned(), st = C.m.kfState('w');
    if (st.phase === 'playing' && rise.length < 8) rise.push(st.w);
    if (st.phase === 'ended' && endT === null) endT = i;
    if (endT !== null && fall.length < 16) fall.push(o.hand_l_y || 0);
  }
  const mono = (v, up) => v.every((x, i) => !i || (up ? x >= v[i - 1] - 1e-9 : x <= v[i - 1] + 1e-9));
  check('owned(): an engine move owns the root before the job (its move-layer weight)', (ow0.root_yaw || 0) > 0.5, `root_yaw ${ow0.root_yaw}`);
  check('a job started over a running move blends in (its weight rises 0 -> 1 over W_IN)', rise[0] < 0.5 && rise[rise.length - 1] > 0.99 && mono(rise, true), rise.map(x => x.toFixed(2)).join(' '));
  check('owned(): after the end its channels fall smoothly to 0 (gone) over W_OUT', fall[0] > 0.5 && fall[fall.length - 1] === 0 && mono(fall, false), fall.map(x => x.toFixed(2)).join(' '));
  const g = makeChar('fox'); for (let i = 0; i < 5; i++) g.step(); g.m.command({action: 'gesture', name: 'wave', side: 'right'}); g.step(); g.step(); const a = g.m.owned().hand_l_y || 0; for (let i = 0; i < 20; i++) g.step(); const b = g.m.owned().hand_l_y || 0;
  g.m.command({action: 'look', target: 'left'}); for (let i = 0; i < 10; i++) g.step(); const lk = g.m.owned().gaze_x || 0;
  check('owned(): commanded gestures rise with their envelope, a commanded look owns the gaze; idle owns nothing', a > 0 && a < b && b > 0.9 && lk > 0.9 && !Object.keys(makeChar('fox').m.owned()).length, `wave ${a.toFixed(2)} -> ${b.toFixed(2)}, look ${lk.toFixed(2)}`);
}
for (const name of ['fox', 'muse']) {
  const LV = {smile: 0.9, gaze_x: 0.5, head_yaw: 0.3, body_lean: 0.4, brow_l: 0.6, mouth_open: 0.3, ear_l: -0.5};
  const C = makeChar(name); let calls = 0, lastState = null;
  C.m.setListener(a => { calls++; lastState = a.state; return LV; });
  const st = streams.n02_jump7, n = 360;
  const rows = runStream(C, st, n);
  const s0 = startFrame(C), e0 = frameOf(kfEvents(C).find(e => e.phase === 'end'));
  const near = (i, c) => Math.abs(rows[i][c] - LV[c]);
  const pre = Math.max(...Object.keys(LV).map(c => near(s0 - 5, c)));
  const mid = Math.round((s0 + e0) / 2);
  const freeMid = Math.max(...['gaze_x', 'head_yaw', 'body_lean', 'brow_l', 'ear_l'].map(c => near(mid, c)));
  const smileMid = rows[mid].smile, plain = runStream(makeChar(name), st, n)[mid].smile;   // the same job without a listener
  const post = Math.max(...Object.keys(LV).map(c => near(e0 + 20, c)));
  check(`${name}: listener values are the frame where nothing commanded owns them (before / after the job: max |diff| ${Math.max(pre, post).toExponential(1)})`, pre < 1e-9 && post < 1e-9 && calls > n);
  check(`${name}: during the job its channels (smile) are the job's; gaze / head / body_lean / brow / ear stay the listener's`, freeMid < 1e-9 && Math.abs(smileMid - plain) < 1e-9, `smile ${smileMid.toFixed(3)} = the job's ${plain.toFixed(3)} without a listener (listener 0.9), others max |diff| ${freeMid.toExponential(1)}; state keys ${Object.keys(lastState || {}).join(',')}`);
  C.m.setListener(null);
}
{ // the listener's mouth yields to speech too; silent it is the listener's
  const C = makeChar('fox'); C.m.setListener(() => ({mouth_open: 0.5, smile: 0.6}));
  const vz = talk(30, 75), mo = [];
  for (let i = 0; i < 120; i++) mo.push(C.step(vz(i)).mouth_open);
  check("listener: its mouth_open yields while voiced (speech's), applies when silent", Math.max(...mo.slice(36, 74)) < 0.01 && Math.min(...mo.slice(105, 120)) > 0.49 && Math.min(...mo.slice(5, 29)) > 0.49,
    `voiced max ${Math.max(...mo.slice(36, 74)).toFixed(3)}, silent again ${Math.min(...mo.slice(105, 120)).toFixed(3)}`);
}
{ // contract shapes: the command result, kfState, fail
  const C = makeChar('orb'); for (let i = 0; i < 5; i++) C.step();
  const r = C.m.command({action: 'keyframes', id: 'p7', description: 'wave hello', poses: [{t: 0.3, right_paw_y: 0.9}], done: false, hold: true});
  const s1 = C.m.kfState('p7');
  C.m.kfFeed('p7', [{t: 0.8, right_paw_x: 0.3}, {t: 1.3, right_paw_x: 0}], false); const s2 = C.m.kfState('p7');
  C.step(); const s3 = C.m.kfState('p7');
  C.m.kfRelease('p7'); C.step(); const s4 = C.m.kfState('p7');
  const ok = r.ok && r.id === 'p7' && s1.phase === 'buffering' && s1.poses === 1 && s2.phase === 'ready' && s3.phase === 'ready' && s4.phase === 'playing' && s4.played_s > 0 && s2.planned_s === 1.3;
  check("command → {ok,id,did}; kfState buffering → ready (2 poses, 1.3 s) → held → playing after kfRelease", ok, JSON.stringify({r, s1: s1.phase, s2: [s2.phase, s2.poses, s2.planned_s], s3: s3.phase, s4: [s4.phase, s4.played_s]}));
  const f1 = C.m.command({action: 'keyframes', id: 'p8', description: 'do a thing', poses: [], done: true});
  C.m.command({action: 'keyframes', id: 'p9', description: 'another', poses: []}); const f2 = C.m.kfFeed('p9', [], true, {ok: false, error: 'planner timed out'});
  const fe = kfEvents(C).filter(e => e.phase === 'fail');
  check("a job that ends with no poses fails (event 'fail' + the planner's error)", !f1.ok && !f2.ok && fe.length === 2 && fe[1].reason === 'planner timed out' && C.m.kfState('p9').phase === 'failed', JSON.stringify(fe.map(e => e.reason)));
  for (let i = 0; i < 60; i++) C.step();
  const stalled = C.m.kfState('p7');   // no done yet: the clock brakes at the last parsed pose and waits there
  C.m.kfFeed('p7', [], true); for (let i = 0; i < 60; i++) C.step();
  check('an open stream waits at its last parsed pose (never runs past it); done lets it finish', stalled.phase === 'playing' && Math.abs(stalled.played_s - 1.3) < 0.02 && stalled.stalls === 1, JSON.stringify(stalled));
  const ph = kfEvents(C).filter(e => e.id === 'p7').map(e => e.phase).join(',');
  check("events: kind 'kf', name 'perform', phases ready, start, end (a failing newer job leaves the playing one alone), each with did", ph === 'ready,start,end' && kfEvents(C).every(e => e.name === 'perform' && (e.did || e.phase === 'fail')), ph);
}
{ // the learned-reaction (hub) map gets the job as a commanded value: alpha 0 on its channels
  const C = makeChar('muse'); C.m.setLearned(true);
  const st = streams.n02_jump7; let alpha = null;
  runStream(C, st, 200, {at: (i, X) => { if (X.m.kfState('k1') && X.m.kfState('k1').phase === 'playing' && alpha === null && X.m.kfState('k1').played_s > 1) { const names = ['smile', 'hand_l_y', 'head_yaw']; const lp = X.m.learnedParts(names); alpha = Object.fromEntries(names.map((nm, k) => [nm, (lp.ab.find(q => q[0] === k) || [k, 1, 0])[1]])); } }});
  check('learned mode: learnedParts gives the job its channels (alpha 0 on smile / paw), untouched head stays learned (alpha 1)', alpha && alpha.smile === 0 && alpha.hand_l_y === 0 && alpha.head_yaw === 1, JSON.stringify(alpha));
}

// ------------------------------------------------------------------ (e) talking while it moves
console.log('== (e) talk while moving: body channels keep their trajectory, the mouth follows speech (muse, fox, orb)');
const lipJaw = i => 0.5 * Math.max(0, Math.sin(2 * Math.PI * 4 * i * DT));   // the renderer's viseme jaw, 4 syllables / s
for (const name of ['muse', 'fox', 'orb']) {
  // (1) a real stream: the job's contribution on every body channel is the same with and without speech
  const st = streams.n10_chain3, n = 330;
  const probe = makeChar(name); probe.m.setCospeech(true); runStream(probe, st, n); const s0 = startFrame(probe);
  const run = (voice, kf) => { const C = makeChar(name); C.m.setCospeech(true); return kf ? {C, rows: runStream(C, st, n, {voice})} : refRun(name, st, n, {voice, at: (i, X) => { if (i === 0) X.m.setCospeech(true); }}); };
  const V = talk(s0 - 15, s0 + 120), Q = () => ({});
  const S1 = run(Q, true).rows, R1 = run(Q, false).rows, S2r = run(V, true), S2 = S2r.rows, R2 = run(V, false).rows;
  const contrib = (A, B, i, c) => PER.test(c) ? wrap((A[i][c] || 0) - (B[i][c] || 0)) : (A[i][c] || 0) - (B[i][c] || 0);
  let worst = 0, wc = '';
  const bodyCh = (name === 'muse' ? BODY_CH : BODY_CH.filter(c => !ZERO(c)).concat(['body_bounce']));
  for (const c of bodyCh) for (let i = 0; i < n; i++) { const d = Math.abs(contrib(S2, R2, i, c) - contrib(S1, R1, i, c)); if (d > worst) { worst = d; wc = c; } }
  const sameStart = startFrame(S2r.C) === s0, sameEnd = frameOf(kfEvents(S2r.C).find(e => e.phase === 'end')) === frameOf(kfEvents(probe).find(e => e.phase === 'end'));
  check(`${name}: speech during the job changes none of its body channels (n10_chain3, max |contribution diff| ${worst.toExponential(1)}${wc ? ' ' + wc : ''}), same start / end frames`, worst < 1e-9 && sameStart && sameEnd);
  // (2) the mouth: a job holding the mouth open 0.7 from 0.5 s to 4.5 s, speech from 1.0 s to 2.5 s
  const C = makeChar(name); for (let i = 0; i < 30; i++) C.step();
  C.m.command({action: 'keyframes', id: 'm', description: 'open your mouth wide and wave', done: true, hold: false,
    poses: [{t: 0.5, mouth_open: 0.7, right_paw_y: 0.8}, {t: 1.5, right_paw_x: 0.3}, {t: 2.5, right_paw_x: 0}, {t: 3.5, right_paw_x: 0.3}, {t: 4.5, mouth_open: 0.7, right_paw_x: 0}]});
  const mo = [], vz = talk(30, 75);   // job starts at frame 1 (t 0 = frame 0): voiced plan time 1.0 – 2.5 s
  for (let i = 0; i < 180; i++) mo.push(C.step(vz(i)).mouth_open);
  const voicedW = mo.slice(40, 74), silentW = mo.slice(100, 130);
  const jawErr = Math.max(...voicedW.map((m, k) => Math.abs(Math.max(lipJaw(40 + k), m) - lipJaw(40 + k))));
  check(`${name}: while voiced the job's mouth_open yields (max ${Math.max(...voicedW).toFixed(3)}; rendered jaw = the visemes, max err ${jawErr.toFixed(3)}); silent it applies (min ${Math.min(...silentW).toFixed(3)})`,
    Math.max(...voicedW) < 0.02 && jawErr < 0.02 && Math.min(...silentW) > 0.6);
}

if (require.main === module) { console.log(fails ? `${fails} failed, ${passes} passed` : `all ${passes} passed`); process.exit(fails); }
