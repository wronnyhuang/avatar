// EVAL: offline replay of today's dev controller (avatar repo 8e8f606 = what avatar-dev runs) on one clip.
// No browser, no network, no future frames. One job JSON (argv[2]) -> rig.jsonl (+ .stats.json).
//
// mode 'rules'     = the page's motion stack as live on the Muse: motion.js createMotion (moves engine + muse body/props
//                    via setProps, as page.html startMotion) + setPrior(prior.js GRU, prior_weights.json) + cospeech.js
//                    (muse speech feature; fed the avatar reply's words when the clip has a transcript), stepped at 30 Hz
//                    with the page's inputs: perception features (+ their events), userVoice {rms, voiced = the mic VAD
//                    of voice.js: TH 0.012, 120 ms onset, 450 ms hangover, echo-gated while the avatar is audible},
//                    avatarVoice = the playout tap {rms, voiced: rms > 0.008, pitch: page pitchOf()}, speaking/listening,
//                    playT. Rig vector rounded to 1e-4 like the wire.
// mode 'gru-prior' = prior.js ALONE: the same frames the motion controller builds for it (self/partner voice, the user's face
//                    blend+matrix+yaw/pitch deg, hands, user presence) -> its OWN rig-unit outputs (head yaw/pitch/roll,
//                    head x/y/z, gaze x/y, blink l/r, brow l/r, smile, squint, eye_wide) written directly to those rig
//                    channels; every other rig channel 0 (no rule reactions, no attention/gaze-following, no breathing,
//                    no blink clock, no gestures, no cospeech). Mouth = the renderer's lip sync (same for every system).
'use strict';
const fs = require('fs'), path = require('path');
const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const AV = job.avatar;                                        // avatar repo root (8e8f606)
const load = rel => { const src = fs.readFileSync(path.join(AV, rel), 'utf8'); const mod = {exports: {}};
  new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; };
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const priorLib = load('app/prior.js');
const createPrior = priorLib.createPrior || globalThis.createPrior;
const Cos = load('app/cospeech.js');
const weights = JSON.parse(fs.readFileSync(path.join(AV, 'app/prior_weights.json'), 'utf8'));

// ---- inputs
const rows = fs.readFileSync(job.perception, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
function readWav(p) {   // 16-bit PCM mono/stereo -> Int16Array mono, rate
  const b = fs.readFileSync(p); let o = 12, fmt = null, data = null;
  while (o + 8 <= b.length) { const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = {ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22)};
    if (id === 'data') data = b.subarray(o + 8, o + 8 + n); o += 8 + n + (n & 1); }
  if (!fmt || fmt.bits !== 16) throw Error('wav: 16-bit PCM only ' + p);
  const all = new Int16Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length - (data.length % 2)));
  if (fmt.ch === 1) return {x: all, sr: fmt.sr};
  const x = new Int16Array(all.length / fmt.ch); for (let i = 0; i < x.length; i++) { let s = 0; for (let c = 0; c < fmt.ch; c++) s += all[i * fmt.ch + c]; x[i] = s / fmt.ch; }
  return {x, sr: fmt.sr};
}
const mic = readWav(job.mic), voice = readWav(job.voice);
if (mic.sr !== 24000 || voice.sr !== 24000) throw Error('expect 24 kHz wavs');
const SR = 24000, N = Math.round(job.duration * 30), DT = 1 / 30;
const utts = job.voice_utts || [];                            // [[t0, t1, text]] the avatar's replies (playout spans)
const inUtt = t => utts.some(u => t >= u[0] && t <= u[1]);

// page.html pitchOf (20 ms int16 chunk @ 24 kHz)
function pitchOf(a) {
  const n = a.length >> 1; if (n < 160) return null;
  const x = new Float32Array(n); let e = 0; for (let i = 0; i < n; i++) { const v = (a[2 * i] + a[2 * i + 1]) * 0.5; x[i] = v; e += v * v; }
  if (e < n * 300 * 300) return null;
  let best = 0, bl = 0; const lo = Math.floor(12000 / 400), hi = Math.min(n - 40, Math.ceil(12000 / 70));
  for (let l = lo; l <= hi; l++) { let c = 0; for (let i = 0; i + l < n; i++) c += x[i] * x[i + l]; c /= (n - l); if (c > best) { best = c; bl = l; } }
  return best > 0.45 * e / n && bl ? Math.round(12000 / bl) : null;
}
const rmsOf = (x, a, b) => { let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, b - a)); };

// ---- controller
let m = null, prior = null, cos = null;
const actions = [];
if (job.mode === 'rules') {
  m = M.createMotion({seed: job.seed, prior: null, onAction: ev => actions.push(ev)});
  if (job.body || job.props) m.setProps(job.props ? JSON.parse(fs.readFileSync(job.props, 'utf8')) : null,
    job.body ? JSON.parse(fs.readFileSync(job.body, 'utf8')) : null, job.character || 'muse');
  m.setPrior(createPrior(weights));                            // page: fetch prior_weights.json -> setPrior(createPrior(w))
  if (job.cospeech !== false && utts.some(u => u[2])) cos = Cos.createCospeech({motion: m, seed: job.seed + 1, motionNow: () => m.state.t});
} else if (job.mode === 'gru-prior') {
  prior = createPrior(weights, {seed: job.seed});
} else throw Error('mode ' + job.mode);

const meta = {rig_params: M.RIG_PARAMS, mode: job.mode, seed: job.seed, clip: job.clip, perception: job.perception,
  ablation: job.ablation || 'none', controller: 'avatar ' + (job.avatar_rev || '?'), cospeech: !!cos};
const lines = [JSON.stringify(meta)], times = [];
let ri = 0, f = null, micI = 0, vI = 0, userLvl = 0, micSpeaking = false, voicedRun = 0, silentRun = 0;
let lastLevel = 0, vpitch = null, cosCov = null;
const textAt = [];                                            // cospeech text schedule: [t, delta, info]
utts.forEach((u, k) => { if (!u[2]) return; textAt.push([Math.max(0, u[0] - 0.4), u[2] + ' ', {response_id: 'r' + k}]);
  textAt.push([Math.max(0, u[0] - 0.3), u[2], {response_id: 'r' + k, final: true}]); });
textAt.sort((a, b) => a[0] - b[0]);
let ti = 0;
const pstate = {present: false};
for (let i = 0; i < N; i++) {
  const t = i * DT;
  // avatar playout tap: 20 ms chunks up to t (silent between replies when the clip has reply spans, else digital silence)
  while ((vI + 1) * 480 <= Math.round(t * SR) + 480 && vI * 480 < voice.x.length) {
    const a = vI * 480, b = Math.min(voice.x.length, a + 480), t0 = a / SR, pcm = voice.x.subarray(a, b);
    const r = rmsOf(voice.x, a, b);
    const silent = utts.length ? !inUtt(t0 + 0.01) : r < 30;
    lastLevel = silent ? 0 : r / 32768;
    if (!silent) { const f0 = pitchOf(pcm); if (f0) vpitch = {t: t0, f0}; }
    if (cos) cos.audio(t0, lastLevel, silent, 0.02, silent ? undefined : pcm);
    vI++;
  }
  const av = {rms: lastLevel, voiced: lastLevel > 0.008, pitch: null};
  if (av.voiced && vpitch && t - vpitch.t < 0.12) av.pitch = vpitch.f0;
  // mic meter (voice.js): 10 ms blocks, TH 0.012, onset 120 ms, end 450 ms, echo-gated while the avatar is audible
  while ((micI + 1) * 240 <= Math.round(t * SR) + 240 && micI * 240 < mic.x.length) {
    const a = micI * 240, b = Math.min(mic.x.length, a + 240), r = rmsOf(mic.x, a, b) / 32768;
    userLvl = r;
    if (r > 0.012 && !av.voiced) { voicedRun += 10; silentRun = 0; if (!micSpeaking && voicedRun >= 120) micSpeaking = true; }
    else { silentRun += 10; if (silentRun > 60) voicedRun = 0; if (micSpeaking && silentRun >= 450) micSpeaking = false; }
    micI++;
  }
  while (ti < textAt.length && textAt[ti][0] <= t) { if (cos) cos.text(textAt[ti][1], textAt[ti][2]); ti++; }
  while (ri < rows.length && rows[ri].t <= t + 1e-9) f = rows[ri++].features;
  const speaking = utts.length ? inUtt(t) : av.voiced;
  const st = performance.now();
  let o;
  if (m) {
    o = m.step(DT, {features: f, userVoice: {rms: userLvl, voiced: micSpeaking, pitch: null}, avatarVoice: av,
      speaking, listening: micSpeaking, playT: t}) || {};
    if (cos) { cos.tick(t); const cov = cos.covering(t); if (cov !== cosCov) { cosCov = cov; m.setCospeech(cov); } }
  } else {
    const fc = f && f.face && f.face.present ? f.face : null;
    const pr = prior.step({
      self: {rms: av.rms, pitch: av.pitch || 0, voiced: av.voiced, speaking},
      partner: {rms: userLvl, pitch: 0, voiced: micSpeaking, speaking: micSpeaking},
      face: fc ? {present: true, blend: fc.blend || {}, matrix: fc.matrix || undefined, yaw: (+fc.yaw || 0) * 57.2958, pitch: (+fc.pitch || 0) * 57.2958} : {present: false},
      hands: ((f && f.hands) || []).map(h => ({x: (+h.x || 0) * 0.5 + 0.5, y: 0.5 - (+h.y || 0) * 0.5})),
      t, dt: DT, user: {present: !!fc, ax: fc ? fc.ax : 0, ay: fc ? fc.ay : 0, attending: !!(f && f.attending)}}) || {};
    o = {};
    for (const k of M.RIG_PARAMS) o[k] = 0;
    for (const k of ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'blink_l', 'blink_r',
      'brow_l', 'brow_r', 'smile', 'squint', 'eye_wide']) if (Number.isFinite(+pr[k])) o[k] = +pr[k];
  }
  times.push(performance.now() - st);
  lines.push(JSON.stringify({type: 'rig', t: +t.toFixed(6), a: M.RIG_PARAMS.map(k => { const x = +o[k]; return Number.isFinite(x) ? Math.round(x * 1e4) / 1e4 : 0; }),
    speaking, listening: micSpeaking, f0: av.pitch || 0}));
}
fs.writeFileSync(job.rig, lines.join('\n') + '\n');
const ts = times.slice().sort((a, b) => a - b);
fs.writeFileSync(job.rig + '.stats.json', JSON.stringify({mode: job.mode, frames: N, step_ms: {p50: ts[Math.floor(ts.length * .5)],
  p99: ts[Math.floor(ts.length * .99)], max: ts[ts.length - 1]}, events: m ? m.state.log : [], actions: actions.slice(0, 400),
  cospeech: cos ? {stats: cos.stats(), commits: cos.commits().length} : null}, null, 1));
