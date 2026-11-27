// node tests/sem_page_test.cjs [--verbose] — (docs/SEMANTIC.md §4): the page side of the semantic API planner.
//  M. app/motion.js space:'rig' keyframe jobs (dense rig-space tracks keyed by rig names) on muse / fox (+ orb):
//     plays the track within tolerance on the channels it lists, never touches the others (identical to a run with the same
//     command that never plays), crossfades in from the live value (no pop, also out of the thinking gesture), per-channel
//     release, blends back, mouth yields to speech, stop / newer request interrupt, stall braking, the 0.4 s ready rule,
//     keep_pose holds, unknown channels counted, root group (muse floor contact, fox seated mapping).
//  P. app/perform.js: semantic mode only when asked (plan message planner:'semantic' + state.sem, off = the old message),
//     mode-keyed page cache, space:'rig' kf batches reach the motion as a rig job (keep_pose too), sem messages stored,
//     timings, thinking lead-in, sem_unavailable.
// Exit code = number of failures.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const Perf = load('app/perform.js');
const ROOT = path.join(__dirname, '..'), ANIM = path.join(ROOT, 'renderer', 'animals');
const DT = 1 / 30, VERBOSE = process.argv.includes('--verbose');
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const J = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const BODY = {muse: J(path.join(ANIM, 'muse.body.json'))};
const PROPS = {muse: J(path.join(ANIM, 'muse.props.json'))};

function makeChar(name, seed = 4555, face) {
  const ev = [];
  const m = M.createMotion({seed, character: name, onAction: e => ev.push(e)});
  if (BODY[name]) m.setProps(JSON.parse(JSON.stringify(PROPS[name])), JSON.parse(JSON.stringify(BODY[name])), name);
  else m.setProps(null, null, name);
  let t = 0;
  const f0 = face || {ax: 0, ay: 0};
  const step = (extra = {}) => { t += DT; const fc = typeof f0 === 'function' ? f0(t) : f0;
    return Object.assign({}, m.step(DT, Object.assign({features: {t, face: {present: true, x: fc.ax, y: fc.ay, ax: fc.ax, ay: fc.ay, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}}, extra))); };
  return {m, ev, step, name, now: () => t};
}
// a dense track like app/semantic emits: 20 Hz, only the channels the program drives, each listed over its window
// chans: {name: {w: [t0, t1], f: t -> value}}
function track(chans, dur, hz = 20) {
  const P = [];
  for (let k = 0; k <= Math.round(dur * hz); k++) {
    const t = +(k / hz).toFixed(4), p = {t};
    for (const [c, s] of Object.entries(chans)) if (t >= s.w[0] - 1e-9 && t <= s.w[1] + 1e-9) p[c] = +s.f(t).toFixed(5);
    if (Object.keys(p).length > 1) P.push(p);
  }
  return P;
}
const smooth = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
const env = (t, a, b, r = 0.4) => smooth((t - a) / r) * (1 - smooth((t - (b - r)) / r));   // 0 -> 1 -> 0 over [a, b]
// the example program: raise the own-right paw (rig _l) and wave, a head turn, a smile, mouth, a wink (released early)
const WAVE = {
  hand_l_x: {w: [0, 3], f: t => 0.35 * env(t, 0, 3)},
  hand_l_y: {w: [0, 3], f: t => 0.9 * env(t, 0, 3)},
  hand_l_z: {w: [0, 3], f: t => 0.2 * env(t, 0, 3) + 0.12 * Math.sin(2 * Math.PI * 1.6 * t) * env(t, 0.5, 2.6, 0.3)},
  hand_l_open: {w: [0, 3], f: t => 0.2 + 0.8 * env(t, 0, 3)},
  head_yaw: {w: [0, 3], f: t => 0.25 * env(t, 0.2, 2.8)},
  smile: {w: [0.3, 2.6], f: t => 0.1 + 0.5 * env(t, 0.3, 2.6)},
  mouth_open: {w: [0.5, 2.0], f: t => 0.5 * env(t, 0.5, 2.0, 0.3)},
  blink_r: {w: [1.0, 1.4], f: t => env(t, 1.0, 1.4, 0.15)},
};
const WAVE_P = track(WAVE, 3);
const lin = (P, c, x) => {   // the track value at plan time x (linear between dense samples; null outside its window)
  let a = null, b = null;
  for (const p of P) { if (!(c in p)) continue; if (p.t <= x + 1e-9) a = p; else { b = p; break; } }
  if (!a) return null; if (!b) return a[c];
  const u = (x - a.t) / (b.t - a.t); return a[c] + (b[c] - a[c]) * u;
};
// stream the poses: `ahead` s of poses parsed before the start, then they arrive faster than real time (one 0.25 s batch
// every 0.15 s); done when the last batch is in. Returns per-frame {o, kf: kfState}.
function runRig(C, P, n, o = {}) {
  for (let i = 0; i < 30; i++) C.step();
  if (o.before) o.before(C);
  const id = o.id || 's1', hz = 20;
  const cmd = {action: 'keyframes', id, description: o.desc || 'wave with the right paw', poses: [], done: false, hold: false, space: o.space === undefined ? 'rig' : o.space, hz};
  if (o.kf !== false) { const r = C.m.command(cmd, {src: 'tool'}); if (!r.ok) throw Error('refused ' + JSON.stringify(r)); }
  const out = []; let k = 0, sent = false;
  const batchS = o.batchS || 0.25, everyS = o.everyS || 0.15;
  for (let i = 0; i < n; i++) {
    const tt = i * DT;
    if (o.kf !== false && !o.noFeed) {
      const upto = o.gap && tt >= o.gap[0] && tt < o.gap[1] ? o.gapUpTo : (o.ahead ?? 0.5) + Math.floor(tt / everyS) * batchS;
      const b = []; while (k < P.length && P[k].t <= upto + 1e-9) b.push(P[k++]);
      if (b.length || (k >= P.length && !sent)) { const fin = k >= P.length && !sent; if (fin) sent = true; C.m.kfFeed(id, b, fin, fin && o.keep ? {keep_pose: true} : null); }
    }
    if (o.at) o.at(i, C, id);
    const row = C.step(o.voice ? o.voice(i) : {});
    out.push({o: row, kf: C.m.kfState(id), fed: k ? P[k - 1].t : -1});
  }
  return out;
}
const kfEv = C => C.ev.filter(e => e.kind === 'kf');
const RIGP = M.RIG_PARAMS || null;
const allNames = r => Object.keys(r).filter(k => typeof r[k] === 'number');

console.log('== M. space:\'rig\' keyframe jobs (app/motion.js)');
for (const name of ['muse', 'fox']) {
  const N = 150;
  const C = makeChar(name), rows = runRig(C, WAVE_P, N);
  const Cb = makeChar(name), base = runRig(Cb, WAVE_P, N, {noFeed: true});   // the same command (boredom reset), never plays
  const evs = kfEv(C).map(e => e.phase);
  check(`${name}: ready -> start -> end events, as a planner move`, evs.join() === 'ready,start,end', evs.join());
  const st = rows[rows.length - 1].kf;
  check(`${name}: kfState says space rig, 8 channels, nothing unknown`, st.space === 'rig' && st.channels.length === 8 && !st.unknown.length, JSON.stringify({space: st.space, ch: st.channels, unk: st.unknown}));
  // tracking: owned channels (from touch + 0.25 s to release / the end) follow the track; the mouth only while silent (it is)
  let worst = 0, wc = '', nchk = 0, over = 0, oc = '', sum = 0;
  for (const r of rows) {
    const x = r.kf ? r.kf.played_s : 0; if (!(x > 0) || r.kf.phase !== 'playing') continue;
    for (const [c, s] of Object.entries(WAVE)) {
      if (x < s.w[0] + 0.25 || x > s.w[1] - 0.02) continue;
      if (name !== 'muse' && name !== 'juno' && /^(smile|mouth_open|blink_r)$/.test(c) && false) continue;
      const v = lin(WAVE_P, c, x); if (v === null) continue;
      const d = Math.abs(r.o[c] - v); nchk++; sum += d; if (d > worst) { worst = d; wc = c + ' @' + x.toFixed(2); }
      const k = WAVE_P.findIndex(p => p.t > x + 1e-9), a = WAVE_P[k - 1], b = WAVE_P[k];   // no overshoot between samples
      if (a && b && c in a && c in b) { const ov = Math.max(0, r.o[c] - Math.max(a[c], b[c]), Math.min(a[c], b[c]) - r.o[c]); if (ov > over) { over = ov; oc = c + ' @' + x.toFixed(2); } }
    }
  }
  check(`${name}: owned channels play the dense track (mean |out - linear track| ${(sum / nchk).toFixed(4)}, max ${worst.toFixed(4)} ${wc} — monotone cubic vs linear at a blink's corner; ${nchk} samples)`, sum / nchk < 0.01 && worst < 0.08 && nchk > 300);
  check(`${name}: no overshoot between dense samples (max ${over.toFixed(4)} ${oc})`, over < 0.003);
  // unlisted channels: identical to the same command that never plays, every frame
  const listed = new Set(Object.keys(WAVE)), legs = name !== 'fox';
  let diffN = 0, diffC = new Set();
  for (let i = 0; i < N; i++) for (const c of allNames(rows[i].o)) {
    if (listed.has(c) || (legs && /^cam_/.test(c))) continue;   // the camera follows every keyframe job on a character with legs (c2)
    if (rows[i].o[c] !== base[i].o[c]) { diffN++; diffC.add(c); }
  }
  check(`${name}: channels the track never lists are identical to a no-play run (all ${allNames(rows[0].o).length - listed.size - (legs ? 3 : 0)} others${legs ? ' but the camera' : ''}, ${N} frames)`, diffN === 0, [...diffC].slice(0, 12).join(','));
  // blend back: after the end + W_OUT the job is gone and every channel equals the no-play run
  const endI = rows.findIndex(r => r.kf && r.kf.phase === 'ended');
  let backD = 0;
  for (let i = Math.min(N - 1, endI + 15); i < N; i++) for (const c of listed) backD = Math.max(backD, Math.abs(rows[i].o[c] - base[i].o[c]));
  check(`${name}: blends back — 0.5 s after the end every listed channel equals the no-play run (max |diff| ${backD.toExponential(1)})`, endI > 0 && backD < 1e-9);
  // per-channel release: the wink (listed 1.0-1.4 s) is the live layer's again 0.45 s after its last pose, mid-move
  let relD = 0, relN = 0;
  for (let i = 0; i < N; i++) { const x = rows[i].kf ? rows[i].kf.played_s : 0; if (rows[i].kf && rows[i].kf.phase === 'playing' && x > 1.85 && x < 2.9) { relD = Math.max(relD, Math.abs(rows[i].o.blink_r - base[i].o.blink_r)); relN++; } }
  check(`${name}: per-channel release (blink_r back on the live layer mid-move, max |diff| ${relD.toExponential(1)} over ${relN} frames)`, relN > 10 && relD < 1e-9);
  // no pops: no frame step on a listed channel is much bigger than the track's own steepest step
  let pop = 0, pc = '';
  for (const c of listed) {
    let trackMax = 0; for (let k = 1; k < WAVE_P.length; k++) if (c in WAVE_P[k] && c in WAVE_P[k - 1]) trackMax = Math.max(trackMax, Math.abs(WAVE_P[k][c] - WAVE_P[k - 1][c]) * (20 / 30));
    for (let i = 1; i < N; i++) { const d = Math.abs(rows[i].o[c] - rows[i - 1].o[c]) - Math.abs(base[i].o[c] - base[i - 1].o[c]); if (d - 1.5 * trackMax > pop) { pop = d - 1.5 * trackMax; pc = c + ' f' + i; } }
  }
  check(`${name}: no pops (largest frame step beyond 1.5x the track's own steepest + the live layer's: ${pop.toFixed(4)})`, pop < 0.03, pc);
}

// a live value far from the track's start (gaze following turned the head): the job crossfades from it (no jump)
{
  const face = {ax: 0.45, ay: 0.1};
  const P = track({head_yaw: {w: [0, 2], f: t => -0.3 * env(t, 0, 2)}, gaze_x: {w: [0, 1.2], f: t => -0.5 * env(t, 0, 1.2, 0.3)}}, 2);
  const C = makeChar('muse', 4555, face), rows = runRig(C, P, 90);
  const Cb = makeChar('muse', 4555, face), base = runRig(Cb, P, 90, {noFeed: true});
  const s = rows.findIndex(r => r.kf && r.kf.phase === 'playing');
  const live0 = base[s].o.head_yaw, out0 = rows[s].o.head_yaw;
  let mx = 0; for (let i = Math.max(1, s); i < s + 10; i++) mx = Math.max(mx, Math.abs(rows[i].o.head_yaw - rows[i - 1].o.head_yaw));
  check(`crossfade from the live value: first frame ${out0.toFixed(3)} ≈ live ${live0.toFixed(3)} (track starts at 0), steepest step ${mx.toFixed(4)}`, Math.abs(out0 - live0) < 0.05 && mx < 0.11 && Math.abs(live0) > 0.08);
  const gz = rows.slice(s).map(r => r.o.gaze_x), bz = base.slice(s).map(r => r.o.gaze_x);
  const g1 = rows.findIndex((r, i) => i > s && r.kf && r.kf.played_s > 0.6);
  check(`gaze_x owned while listed (at 0.6 s ${rows[g1].o.gaze_x.toFixed(3)} vs track ${lin(P, 'gaze_x', rows[g1].kf.played_s).toFixed(3)}), the gaze-follow layer's again after release`,
    Math.abs(rows[g1].o.gaze_x - lin(P, 'gaze_x', rows[g1].kf.played_s)) < 0.03 && Math.abs(gz[gz.length - 1] - bz[bz.length - 1]) < 1e-9);
}

// speech: the mouth yields to lip sync while voiced (SPEECH_GROUP), body channels unchanged
{
  const voice = i => ({avatarVoice: {voiced: i >= 20 && i < 52, rms: i >= 20 && i < 52 ? 0.05 : 0}});
  const C = makeChar('muse'), rows = runRig(C, WAVE_P, 120, {voice});
  const Cs = makeChar('muse'), silent = runRig(Cs, WAVE_P, 120);
  const Cb = makeChar('muse'), base = runRig(Cb, WAVE_P, 120, {noFeed: true, voice});
  let yv = 0, yn = 0, body = 0;
  for (let i = 0; i < 120; i++) {
    const x = rows[i].kf ? rows[i].kf.played_s : 0;
    if (i >= 26 && i < 50 && x > 0.75 && x < 1.8) { yv = Math.max(yv, Math.abs(rows[i].o.mouth_open - base[i].o.mouth_open)); yn++; }
    if (rows[i].kf && rows[i].kf.phase === 'playing' && x > 0.5 && x < 2.6) for (const c of ['hand_l_x', 'hand_l_y', 'hand_l_z', 'head_yaw']) body = Math.max(body, Math.abs(rows[i].o[c] - silent[i].o[c]));
  }
  check(`speech keeps the mouth while voiced (max |mouth - the no-job mouth| ${yv.toFixed(4)} over ${yn} frames); fully owned body channels as silent (max ${body.toExponential(1)})`, yn > 5 && yv < 0.02 && body < 1e-9);
  const silentMouth = silent.filter(r => r.kf && r.kf.played_s > 0.9 && r.kf.played_s < 1.6).map(r => Math.abs(r.o.mouth_open - lin(WAVE_P, 'mouth_open', r.kf.played_s)));
  check(`silent: the track's mouth plays (max err ${Math.max(...silentMouth).toFixed(4)})`, silentMouth.length > 5 && Math.max(...silentMouth) < 0.03);
}

// interruption: 'stop' mid-move, a newer keyframe job (planner move of either space), a whole-body move
{
  const C = makeChar('muse');
  const rows = runRig(C, WAVE_P, 90, {at: (i, C2) => { if (i === 40) C2.m.command({action: 'stop'}, {src: 'voice'}); }});
  const Cb = makeChar('muse'), base = runRig(Cb, WAVE_P, 90, {noFeed: true, at: (i, C2) => { if (i === 40) C2.m.command({action: 'stop'}, {src: 'voice'}); }});
  const e = kfEv(C).map(x => x.phase);
  let d = 0; for (let i = 60; i < 90; i++) for (const c of Object.keys(WAVE)) d = Math.max(d, Math.abs(rows[i].o[c] - base[i].o[c]));
  check(`'stop' interrupts it (events ${e.join()}) and it has blended back 0.6 s later (max |diff| ${d.toExponential(1)})`, e.join() === 'ready,start,interrupted' && d < 1e-9);
  const C2 = makeChar('muse');
  runRig(C2, WAVE_P, 60, {at: (i, X) => { if (i === 30) { X.m.command({action: 'keyframes', id: 'old2', description: 'nod', poses: [{t: 0.3, head_pitch: 0.2}, {t: 1.2, head_pitch: 0}], done: true, hold: false}, {src: 'tool'}); } }});
  const e2 = C2.ev.filter(x => x.kind === 'kf').map(x => x.id + ':' + x.phase);
  check(`a newer keyframe job (own-side) replaces it when it starts (${e2.join(' ')})`, e2.includes('s1:interrupted') && e2.includes('old2:start'));
  const C3 = makeChar('muse');
  runRig(C3, WAVE_P, 60, {at: (i, X) => { if (i === 30) X.m.command({action: 'move', name: 'jump'}, {src: 'tool'}); }});
  const e3 = kfEv(C3).map(x => x.phase);
  check(`a whole-body move interrupts it (${e3.join()})`, e3.includes('interrupted'));
}

// stall braking: the stream runs dry at 1.0 s for 1.5 s, then the rest arrives
{
  const C = makeChar('muse');
  const rows = runRig(C, WAVE_P, 170, {ahead: 0.5, gap: [0.3, 2.1], gapUpTo: 1.0});
  const Cf = makeChar('muse'), free = runRig(Cf, WAVE_P, 170);
  const st = rows[rows.length - 1].kf;
  let past = 0, acc = 0;
  let accF = 0;
  for (let i = 0; i < rows.length; i++) { const k = rows[i].kf; if (k && k.phase === 'playing' && !k.complete && k.played_s > rows[i].fed + 1e-6) past++; }
  for (let i = 2; i < rows.length; i++) for (const c of ['hand_l_y', 'hand_l_x']) { acc = Math.max(acc, Math.abs(rows[i].o[c] - 2 * rows[i - 1].o[c] + rows[i - 2].o[c])); accF = Math.max(accF, Math.abs(free[i].o[c] - 2 * free[i - 1].o[c] + free[i - 2].o[c])); }
  const e = kfEv(C).map(x => x.phase);
  check(`stall braking: never plays past the last pose it has (${past} frames past), counts a stall (${st && st.stalls}), resumes and ends (${e.join()}), smooth (max 2nd diff ${acc.toFixed(4)} vs ${accF.toFixed(4)} unstalled)`, past === 0 && st && st.stalls >= 1 && e.join() === 'ready,start,end' && acc < 1.5 * accF + 0.005);
}

// ready rule: >= 0.4 s of poses parsed (or done)
{
  const C = makeChar('fox'); for (let i = 0; i < 5; i++) C.step();
  C.m.command({action: 'keyframes', id: 'r1', description: 'x', poses: WAVE_P.filter(p => p.t <= 0.3), done: false, hold: true, space: 'rig', hz: 20}, {src: 'tool'});
  C.step(); const a = C.m.kfState('r1').phase;
  C.m.kfFeed('r1', WAVE_P.filter(p => p.t > 0.3 && p.t <= 0.4), false); C.step(); const b = C.m.kfState('r1').phase;
  const C2 = makeChar('fox'); for (let i = 0; i < 5; i++) C2.step();
  C2.m.command({action: 'keyframes', id: 'r2', description: 'x', poses: [{t: 0, smile: 0.1}, {t: 0.1, smile: 0.3}], done: true, hold: true, space: 'rig', hz: 20}, {src: 'tool'});
  C2.step(); const c = C2.m.kfState('r2').phase;
  check(`ready rule: 0.3 s parsed = ${a}, 0.4 s = ${b}; a short complete program = ${c}; hold:true waits for kfRelease`, a === 'buffering' && b === 'ready' && c === 'ready');
}

// keep_pose: the pose is held after the end (commanded-pose hold) until released
{
  const P = track({hand_r_y: {w: [0, 1.2], f: t => 0.8 * smooth(t / 0.8)}, hand_r_x: {w: [0, 1.2], f: t => -0.3 * smooth(t / 0.8)}}, 1.2);
  const C = makeChar('muse');
  const rows = runRig(C, P, 150, {keep: true, at: (i, X) => { if (i === 120) X.m.command({action: 'release', parts: 'all'}, {src: 'voice'}); }});
  const endI = rows.findIndex(r => r.kf && r.kf.phase === 'holding');
  const held = rows.slice(endI + 20, 119).map(r => r.o.hand_r_y);
  const after = rows[149].o.hand_r_y, evs = kfEv(C).map(e => e.phase).join();
  check(`keep_pose: the end is reported (${evs}) and the arm stays up (hand_r_y ${Math.min(...held).toFixed(3)}..${Math.max(...held).toFixed(3)} for ${held.length} frames), 'release' lets it go (${after.toFixed(3)})`,
    endI > 0 && held.length > 40 && Math.min(...held) > 0.7 && after < 0.3 && evs === 'ready,start,end');
  // root (a crouch) and the mouth are kept too; a newer move replaces the kept pose when it starts
  const P2 = track({root_y: {w: [0, 1.0], f: t => -0.25 * smooth(t / 0.6)}, squash: {w: [0, 1.0], f: t => 0.3 * smooth(t / 0.6)}, mouth_open: {w: [0, 1.0], f: t => 0.4 * smooth(t / 0.6)}}, 1.0);
  const C2 = makeChar('muse');
  const r2 = runRig(C2, P2, 150, {keep: true, at: (i, X) => { if (i === 110) X.m.command({action: 'keyframes', id: 'n2', description: 'nod', poses: [{t: 0.2, head_pitch: 0.2}, {t: 0.8, head_pitch: 0}], done: true, hold: false}, {src: 'tool'}); }});
  const h2 = r2.slice(60, 105), e2 = C2.ev.filter(x => x.kind === 'kf').map(x => x.id + ':' + x.phase).join(' ');
  check(`keep_pose holds the root (root_y ${Math.max(...h2.map(r => r.o.root_y)).toFixed(3)}) and the mouth (${Math.min(...h2.map(r => r.o.mouth_open)).toFixed(3)}) until a newer move starts (${e2}; root_y after ${r2[149].o.root_y.toFixed(3)})`,
    Math.max(...h2.map(r => r.o.root_y)) < -0.15 && Math.min(...h2.map(r => r.o.mouth_open)) > 0.3 && Math.abs(r2[149].o.root_y) < 0.03 && !/s1:interrupted/.test(e2) && /n2:start/.test(e2));
}

// unknown channels / other characters: dropped and counted; nothing breaks
{
  for (const name of ['orb', 'cube', 'bear']) {
    const C = makeChar(name); let err = null;
    try { runRig(C, WAVE_P.map(p => Object.assign({jaw: 0.5, right_paw_x: 0.2, foo_bar: 1}, p)), 120); } catch (e) { err = String(e); }
    const st = C.m.kfState('s1');
    check(`${name}: rig job plays, unknown channels counted (${st && st.unknown.join(',')})`, !err && st && ['jaw', 'right_paw_x', 'foo_bar'].every(k => st.unknown.includes(k)) && kfEv(C).some(e => e.phase === 'end'), err || '');
  }
}

// root group: muse jumps (floor contact through the engine), fox shows it on body_bounce
{
  const P = track({root_y: {w: [0, 1.6], f: t => 0.35 * Math.max(0, Math.sin(Math.PI * (t - 0.4) / 0.8)) * (t > 0.4 && t < 1.2 ? 1 : 0)}, squash: {w: [0, 1.6], f: t => -0.3 * env(t, 0.1, 0.5, 0.15)}}, 1.6);
  const C = makeChar('muse'), rows = runRig(C, P, 90);
  const ry = Math.max(...rows.map(r => r.o.root_y || 0));
  check(`muse: a rig jump lifts the root (max root_y ${ry.toFixed(3)}) and returns`, ry > 0.25 && Math.abs(rows[89].o.root_y || 0) < 0.02);
  const F = makeChar('fox'), fr = runRig(F, P, 90), Fb = makeChar('fox'), fb = runRig(Fb, P, 90, {noFeed: true});
  const bb = Math.max(...fr.map((r, i) => Math.abs(r.o.body_bounce - fb[i].o.body_bounce)));
  check(`fox (no root): the same jump shows on body_bounce (max offset ${bb.toFixed(3)})`, bb > 0.3);
  for (const name of ['muse']) {
    const Cn = makeChar(name), rn = runRig(Cn, WAVE_P, 150), Cnb = makeChar(name), rnb = runRig(Cnb, WAVE_P, 150, {noFeed: true});
    const OWN = WAVE_P.map(p => ({t: p.t + 0.3, right_paw_x: p.hand_l_x, right_paw_y: p.hand_l_y}));
    const Co = makeChar(name), ro = runRig(Co, OWN, 150, {space: null});
    const rootD = Math.max(...rn.map((r, i) => Math.max(...['root_x', 'root_y', 'root_z', 'root_yaw', 'squash', 'foot_l_y', 'gaze_free'].map(c => Math.abs((r.o[c] || 0) - (rnb[i].o[c] || 0))))));
    const mid = [45, 60, 75], z = mid.map(i => rn[i].o.cam_zoom), zo = mid.map(i => ro[i].o.cam_zoom), zb = mid.map(i => rnb[i].o.cam_zoom);
    const back = Math.abs(rn[149].o.cam_zoom - rnb[149].o.cam_zoom);
    check(`${name}: the camera follows a rig job like today's keyframe job (mid-move zoom ${z.map(x => x.toFixed(2)).join('/')} vs own-side job ${zo.map(x => x.toFixed(2)).join('/')}, live ${zb.map(x => x.toFixed(2)).join('/')}), back after (|diff| ${back.toExponential(1)}); root / feet untouched (max |diff| ${rootD.toExponential(1)})`,
      z.every((x, k) => Math.abs(x - zo[k]) < 0.05) && back < 1e-6 && rootD < 1e-9);
  }
}

// the hand heights may use the renderer's wire limit (1.4) in a rig job, e.g. a hand on the face; an own-side job still clamps at 1.2
{
  const P = track({hand_r_y: {w: [0, 2], f: t => 1.35 * env(t, 0, 2, 0.5)}}, 2);
  const C = makeChar('muse'), rows = runRig(C, P, 70);
  const Co = makeChar('muse'), ro = runRig(Co, P.map(p => ({t: p.t + 0.3, left_paw_y: p.hand_r_y})), 70, {space: null});
  const a = Math.max(...rows.map(r => r.o.hand_r_y)), b = Math.max(...ro.map(r => r.o.hand_r_y));
  check(`muse: a rig job's hand_r_y 1.35 reaches ${a.toFixed(3)} (wire limit 1.4); the same through an own-side job clamps at ${b.toFixed(3)}`, a >= 1.3 && Math.abs(b - 1.2) < 1e-6);
}

// thinking lead-in: the thinking gesture is on when the rig job starts: no snap, the thinking fades under the move — compared
// with today's own-side keyframe job playing the same arm path out of the same thinking pose
{
  const OWN = WAVE_P.map(p => ({t: p.t + 0.3, right_paw_x: p.hand_l_x, right_paw_y: p.hand_l_y, right_paw_z: p.hand_l_z, right_paw_open: p.hand_l_open, head_yaw: p.head_yaw}));
  const steep = rows => { let mx = 0, mc = ''; for (let i = 1; i < 40; i++) for (const c of ['hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_y', 'head_yaw', 'head_pitch', 'gaze_y', 'brow_knit']) { const d = Math.abs(rows[i].o[c] - rows[i - 1].o[c]); if (d > mx) { mx = d; mc = c + ' f' + i; } } return [mx, mc]; };
  for (const name of ['muse', 'fox']) {
    const before = X => { const r = X.m.think({variant: 'chin'}); for (let i = 0; i < 25; i++) X.step(); X.thinkOk = r && r.ok; };
    const C = makeChar(name), rows = runRig(C, WAVE_P, 90, {before});
    const Co = makeChar(name), rowsO = runRig(Co, OWN, 90, {before, space: null});
    const [mx, mc] = steep(rows), [mo, moc] = steep(rowsO);
    const th = C.m.thinking ? C.m.thinking() : null;
    check(`${name}: thinking -> rig move hands over without a snap (steepest step ${mx.toFixed(4)} ${mc}; today's own-side job out of the same pose ${mo.toFixed(4)} ${moc}); thinking off after (${th && th.on})`,
      C.thinkOk && mx <= 1.2 * mo + 0.02 && !(th && th.on));
  }
}

// own-side keyframe jobs are untouched: the same poses as a space:'rig' track vs the keyframe language
{
  const C = makeChar('fox'), rows = runRig(C, [{t: 0.2, right_paw_y: 0.6}, {t: 1.0, right_paw_y: 0}], 60, {space: null});
  const st = C.m.kfState('s1');
  check(`own-side job still speaks the keyframe language (right_paw_y -> hand_l_y, peak ${Math.max(...rows.map(r => r.o.hand_l_y)).toFixed(2)}, no space field)`, st && st.space === undefined && Math.max(...rows.map(r => r.o.hand_l_y)) > 0.4);
}

// probe: a rig job shows as 'planner move' on the channels it drives only
{
  const C = makeChar('muse'); C.m.setProbe(true);
  const rows = runRig(C, WAVE_P, 50); const pr = C.m.probe();
  const o = pr.owners;
  check(`probe: rig job owns hand_l_y as planner (${o.hand_l_y && o.hand_l_y.by}), not hand_r_y (${o.hand_r_y && o.hand_r_y.by}) nor gaze_x (${o.gaze_x && o.gaze_x.by})`, o.hand_l_y.by === 'planner' && o.hand_r_y.by !== 'planner' && o.gaze_x.by !== 'planner' && pr.kf.length === 1);
}

// ================================================================== P. perform.js
console.log('== P. perform.js semantic plumbing');
function makePerf(mode, extra = {}) {
  let t = 1000; const timers = []; const sent = [], prompts = [], cmds = [], feeds = [];
  const motion = {command: (c, o) => { cmds.push(c); return {ok: true, id: c.id}; }, kfFeed: (id, p, d, info) => { feeds.push({id, n: p.length, d, info}); return {ok: true}; },
    kfRelease: () => ({ok: true}), kfState: () => null, state: () => ({posture: 'standing'})};
  const th = {begun: [], known: [], ended: [], begin(id) { this.begun.push(id); return {ok: true}; }, known(id) { this.known.push(id); }, end(id, w) { this.ended.push([id, w]); }, cancel() {}};
  const P = Perf.createPerformer(Object.assign({moveSync: () => false /* the flow without move sync */, motion: () => motion, prompt: (x, o) => { prompts.push(x); return null; }, sendCtl: o => sent.push(JSON.parse(JSON.stringify(o))),
    character: () => 'muse', now: () => t, timers: {set: (f, ms) => { timers.push({f, at: t + ms}); return timers.length; }, clear: () => {}}, thinker: () => th,
    planner: mode === undefined ? undefined : () => mode, semState: () => ({user: {present: true, ax: 0.1, ay: -0.05}})}, extra));
  return {P, sent, prompts, cmds, feeds, th, timers, adv: ms => { t += ms; }, now: () => t};
}
{
  const A = makePerf(undefined), B = makePerf('current'), Cs = makePerf('semantic');
  A.P.call('wave at me'); B.P.call('wave at me'); Cs.P.call('wave at me');
  check(`current mode: the plan message is the old one byte for byte (no planner / sem keys)`, JSON.stringify(A.sent[0]) === JSON.stringify(B.sent[0]) && !('planner' in B.sent[0]) && !('sem' in B.sent[0].state), JSON.stringify(B.sent[0]));
  const m = Cs.sent[0];
  check(`semantic mode: plan message carries planner:'semantic' and state.sem (his face)`, m.type === 'plan' && m.planner === 'semantic' && m.state.sem && m.state.sem.user && m.state.sem.user.present === true, JSON.stringify(m));
  check(`thinking lead-in begins for a semantic plan too`, Cs.th.begun.length === 1);
  // rig batches -> one rig keyframe job, later batches fed, keep_pose on the final
  Cs.P.onSem({type: 'sem', id: 'p1', phase: 'start', program: '', source: 'planner', model: 'sonnet', ms: {recv: 0}});
  Cs.adv(900);
  Cs.P.onKf({type: 'kf', id: 'p1', space: 'rig', hz: 20, poses: WAVE_P.slice(0, 10), done: false});
  Cs.P.onSem({type: 'sem', id: 'p1', phase: 'stmt', program: 'reach("right", "up")\n', ms: {recv: 0, first_token: 400, first_stmt: 600, first_pose: 650}, exec_ms: [12], warnings: [], errors: [], targets: {camera: [0, 30, 200]}});
  Cs.adv(300);
  Cs.P.onKf({type: 'kf', id: 'p1', space: 'rig', poses: WAVE_P.slice(10), done: true, ok: true, source: 'planner', planner: 'semantic', keep_pose: true, ms_first_pose: 650, ms_done: 1400});
  Cs.P.onSem({type: 'sem', id: 'p1', phase: 'done', program: 'reach("right", "up")\nwave("right")\n', ms: {recv: 0, first_token: 400, first_stmt: 600, first_pose: 650, done: 1400}, exec_ms: [12, 9], warnings: ['w1'], errors: [], targets: {camera: [0, 30, 200]}, source: 'planner', model: 'sonnet', cached: false});
  const c0 = Cs.cmds[0];
  check(`the first rig batch becomes motion.command({action:'keyframes', space:'rig', hz:20, hold:true})`, c0 && c0.action === 'keyframes' && c0.space === 'rig' && c0.hz === 20 && c0.hold === true && c0.poses.length === 10);
  const f1 = Cs.feeds[Cs.feeds.length - 1];
  check(`later batches go to kfFeed, the final one with keep_pose`, f1 && f1.d === true && f1.info && f1.info.keep_pose === true && f1.n === WAVE_P.length - 10, JSON.stringify(f1));
  const L = Cs.P.semLast();
  check(`semLast(): mode, program, timings, warnings, source (${L && L.source}, ${L && L.program.split('\n').length - 1} statements)`, L && L.mode === 'semantic' && /wave/.test(L.program) && L.hub_ms.first_stmt === 600 && L.warnings[0] === 'w1' && L.page_ms.call === 1000 && L.page_ms.first_pose === 1900);
  // the motion events -> marks + motion length
  Cs.P.onAction({kind: 'kf', id: 'p1', phase: 'ready'}); Cs.adv(100); Cs.P.avatarAudio(); Cs.adv(50); Cs.P.onAction({kind: 'kf', id: 'p1', phase: 'start'}); Cs.adv(3000); Cs.P.onAction({kind: 'kf', id: 'p1', phase: 'end', planned_s: 3});
  const L2 = Cs.P.semLast(), perf = Cs.sent.find(x => x.type === 'perform');
  check(`timings complete after the end (first motion ${L2.page_ms.first_motion}, end ${L2.page_ms.end}); the perform record says planner:'semantic'`, L2.page_ms.first_motion === 2350 && L2.page_ms.end === 5350 && perf && perf.planner === 'semantic');
  const Bp = B.sent.length;
  B.P.onKf({type: 'kf', id: 'p1', poses: [{t: 0.2, right_paw_y: 0.5}, {t: 1, right_paw_y: 0}], done: true, ok: true, source: 'planner'});
  check(`current mode: an own-side kf message plays as today (no space key in the command)`, B.cmds[0] && !('space' in B.cmds[0]) && !('hz' in B.cmds[0]));
  // the page cache is keyed by mode
  const X = makePerf('semantic'); let mode = 'semantic';
  const Y = makePerf(undefined, {planner: () => mode});
  Y.P.call('do a little dance'); Y.P.onKf({type: 'kf', id: 'p1', space: 'rig', poses: WAVE_P, done: true, ok: true, source: 'planner'});
  Y.P.onSem({type: 'sem', id: 'p1', phase: 'done', program: 'sway()\n', ms: {}});
  Y.P.onAction({kind: 'kf', id: 'p1', phase: 'end'});
  Y.adv(20000); mode = 'current';
  const r2 = Y.P.call('do a little dance');
  check(`page cache keyed by mode: a semantic plan never serves the current planner (2nd call ${r2.status}, plan sent ${Y.sent.filter(x => x.type === 'plan').length}x)`, r2.status === 'preparing' && Y.sent.filter(x => x.type === 'plan').length === 2 && !('planner' in Y.sent.filter(x => x.type === 'plan')[1]));
  Y.adv(20000); mode = 'semantic';
  const r3 = Y.P.call('do a little dance');
  check(`... and the semantic repeat is served from its own cache as a rig job (${r3.status})`, r3.status === 'starting' && Y.cmds[Y.cmds.length - 1].space === 'rig');
  // a program aimed at his live face is not replayed from the page cache (the hub re-runs it with the fresh face)
  const Z = makePerf('semantic');
  Z.P.call('point at me'); Z.P.onSem({type: 'sem', id: 'p1', phase: 'done', program: 'point("right", at="user")\n', ms: {}});
  Z.P.onKf({type: 'kf', id: 'p1', space: 'rig', poses: WAVE_P, done: true, ok: true, source: 'planner'}); Z.P.onAction({kind: 'kf', id: 'p1', phase: 'end'});
  Z.adv(20000); const r4 = Z.P.call('point at me');
  check(`a program with a live target ('user') is not replayed from the page cache (${r4.status})`, r4.status === 'preparing');
  // sem_unavailable
  const U = makePerf('semantic'); U.P.call('wave'); U.P.onKf({type: 'kf', id: 'p1', poses: [{t: 0.2, right_paw_y: 0.5}, {t: 1, right_paw_y: 0}], done: true, ok: true, source: 'planner', sem_unavailable: true});
  check(`sem_unavailable:true is stored for the panel and the move plays on today's path`, U.P.semLast().sem_unavailable === true && U.cmds[0] && !('space' in U.cmds[0]));
  // c3 cue hold: semantic = the move starts 250 ms after ready (the cue is still asked for and spoken over it); current = 700
  for (const [md, want] of [['semantic', 250], ['current', 700]]) {
    const Q = makePerf(md); let rel = null; const mo = Q.P; Q.P.call('wave');
    Q.P.onKf({type: 'kf', id: 'p1', space: 'rig', poses: WAVE_P, done: true, ok: true, source: 'planner'});
    const tReady = Q.now(); Q.P.onAction({kind: 'kf', id: 'p1', phase: 'ready'});
    const cueAsked = Q.prompts.some(x => /about to start/.test(x));
    const st = Q.P.list()[0], cueT = Q.timers[Q.timers.length - 1].at - tReady;
    check(`${md}: cue prompt asked (${cueAsked}), the move is released ${cueT} ms after ready without cue audio (want ${want})`, cueAsked && st.phase === 'ready' && cueT === want);
  }
  // c1: the thinking paw is skipped while semantic plans are fast, back once they run slow (running estimate)
  {
    const Q = makePerf('semantic'), th = [];
    Q.th.begin = (id, info) => { th.push(info); return {ok: true}; };
    Q.P.call('a'); Q.adv(1500); Q.P.onKf({type: 'kf', id: 'p1', space: 'rig', poses: WAVE_P, done: true, ok: true, source: 'planner'}); Q.P.stop('x');
    for (let k = 2; k <= 6; k++) { Q.adv(20000); Q.P.call('b' + k); Q.adv(4500); Q.P.onKf({type: 'kf', id: 'p' + k, space: 'rig', poses: WAVE_P, done: true, ok: true, source: 'planner'}); Q.P.stop('x'); }
    Q.adv(20000); Q.P.call('z');
    const C0 = makePerf('current'), th0 = []; C0.th.begin = (id, info) => { th0.push(info); return {ok: true}; }; C0.P.call('a');
    check(`thinking paw: fast semantic plans -> {arms:false} (${JSON.stringify(th[0])}); after slow ones (est ${Q.P.stats().sem_est_ms} ms) -> paw allowed (${JSON.stringify(th[th.length - 1])}); current mode unchanged (${JSON.stringify(th0[0])})`,
      th[0].arms === false && th[1].arms === false && !('arms' in th[th.length - 1]) && !('arms' in th0[0]) && !('sem_est_ms' in C0.P.stats()));
  }
  check(`perform.js without the planner option behaves as before (mode 'current')`, A.P.mode() === 'current' && X.P.mode() === 'semantic');
}

console.log(`sem_page_test: ${passes} passed, ${fails} failed`);
process.exit(fails);
