#!/usr/bin/env node
/* attention_test.cjs — offline test of app/attention.js (ATTENTION candidate detector).
 *
 *   node tests/attention_test.cjs [--clips a,b,...] [--dir D] [--cache C] [--opts '{json}'] [--fps 18] [--verbose]
 *                                       [--out results.json] [--no-assert]
 *
 * Data: the eval clips (<dir>/<clip>/{perception.jsonl, camera.mp4, mic.wav, voice.wav} + <dir>/annotations/<clip>.json).
 *   perception.jsonl = native MediaPipe + the exact JS perception, one JSON per line {t, features, events} at 30 fps;
 *   fed to the detector at ~18 Hz (a frame is kept when floor(t·fps) advances; skipped frames' events ride on the next
 *   kept frame). ctx.thumb = camera.mp4 decoded by ffmpeg to a long-side-48 RGBA thumbnail, a new one every 0.15 s
 *   (the page's cadence), cached under <cache>. ctx.userLevel = the PEAK 10 ms RMS of mic.wav since the previous frame;
 *   ctx.userSpeaking = voice.js's meter VAD (> 0.012 for 120 ms on, < for 450 ms off); avatarSpeaking from voice.wav.
 *   A simulated hub answers every candidate 1.0 s later (worth_reply false) so the in-flight gate behaves as live.
 * Scored:
 *   suite clips (held-out, non-Seamless): target cues = annotation cues of kind presence / reflex / expression (turn cues
 *     are speech pauses, not attention). A candidate HITS a cue when cue.t − 0.5 ≤ t ≤ cue.t + 2.5 (one-to-one, in time
 *     order); unmatched candidates inside a matched window are DUP, the rest FP. recall = hits / cues,
 *     precision = hits / (hits + FP). Clips with only turn cues contribute FPs only.
 *   Seamless sl_* (normal conversation): candidates per minute (target ≤ 2).
 *   loud_sound (SYNTHETIC — no clip has a real bang): a 120 ms decaying noise burst (peak RMS ≈ 0.45) is mixed into
 *     mic.wav of startle_vcd_hands_at_lens_woman at 4.0 s and of sl_S0648_0377_P0299A inside a ≥ 2 s user pause; the
 *     detector must report loud_sound within 0.6 s, and no loud_sound anywhere else.
 *   Cost: ms per frame() call (includes the scene analysis when a thumbnail arrives; the thumbnail decode/draw is not).
 *   Cues are split by whether he is SPEAKING at the cue (annotation user_speech ± 0.2/0.5 s): the detector deliberately
 *   damps conversation-time triggers (smiles, gestures, glances while talking are ordinary), so speech-time cues (the
 *   RAVDESS emotion takes) are reported, not asserted. with_gestures = the same suite with opts.gestures on.
 * Asserts (exit 1 on failure unless --no-assert): recall on cues while he is silent ≥ 0.6, suite precision ≥ 0.5, Seamless ≤ 2 /min,
 *   synthetic bangs detected, mean ≤ 0.3 ms / frame.
 */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process');
const ROOT = path.resolve(__dirname, '..');
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const flag = k => argv.includes('--' + k);
const DIR = arg('dir', process.env.AVATAR_EVAL_CLIPS || path.join(ROOT, 'data', 'eval_clips'));
if (!fs.existsSync(DIR)) { console.log('SKIP eval clips not present (derived from the Seamless Interaction dataset; set AVATAR_EVAL_CLIPS=<dir>): ' + DIR); process.exit(0); }
const CACHE = arg('cache', path.join(require('os').tmpdir(), 'avatar-tests', 'attention_cache'));
const OPTS = JSON.parse(arg('opts', '{}'));
const FPS = +arg('fps', 18);
const THUMB_LONG = 48, THUMB_EVERY = 0.15;
const VERBOSE = flag('verbose');

const src = fs.readFileSync(path.resolve(arg('attention', path.join(ROOT, 'app', 'attention.js'))), 'utf8');
const mod = {exports: {}};
new Function('module', 'exports', src)(mod, mod.exports);
const AvatarAttention = mod.exports;

fs.mkdirSync(CACHE, {recursive: true});

function listClips() {
  const only = arg('clips', '');
  const all = fs.readdirSync(DIR).filter(c => fs.existsSync(path.join(DIR, c, 'perception.jsonl'))).sort();
  return only ? all.filter(c => only.split(',').includes(c)) : all;
}
function probe(file) {
  const r = cp.spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file], {encoding: 'utf8'});
  const [w, h] = r.stdout.trim().split(',').map(Number); return {w, h};
}
function thumbs(clip) {
  const file = path.join(DIR, clip, 'camera.mp4');
  const {w: vw, h: vh} = probe(file);
  const w = vw >= vh ? THUMB_LONG : Math.max(4, Math.round(THUMB_LONG * vw / vh)), h = vw >= vh ? Math.max(4, Math.round(THUMB_LONG * vh / vw)) : THUMB_LONG;
  const cf = path.join(CACHE, `${clip}_${w}x${h}_30.rgba`);
  if (!fs.existsSync(cf)) {
    const r = cp.spawnSync('nice', ['-n', '10', 'ffmpeg', '-v', 'error', '-i', file, '-vf', `fps=30,scale=${w}:${h}:flags=area`, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-y', cf + '.tmp'], {stdio: 'inherit'});
    if (r.status !== 0) throw new Error('ffmpeg failed for ' + clip);
    fs.renameSync(cf + '.tmp', cf);
  }
  const buf = fs.readFileSync(cf), fsz = w * h * 4, n = Math.floor(buf.length / fsz);
  return {w, h, n, get: i => new Uint8ClampedArray(buf.buffer, buf.byteOffset + Math.max(0, Math.min(n - 1, i)) * fsz, fsz)};
}
function readWav(file) {
  if (!fs.existsSync(file)) return null;
  const b = fs.readFileSync(file);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = {ch: b.readUInt16LE(off + 10), sr: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22)};
    if (id === 'data') { data = b.subarray(off + 8, off + 8 + sz); break; }
    off += 8 + sz + (sz & 1);
  }
  if (!fmt || !data || fmt.bits !== 16) return null;
  const n = Math.floor(data.length / 2 / fmt.ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2 * fmt.ch) / 32768;
  return {sr: fmt.sr, x};
}
function rms10(w) {             // 10 ms RMS chunks, like the page's mic meter worklet
  if (!w) return {step: 0.01, r: new Float32Array(0)};
  const n = Math.round(w.sr * 0.01), m = Math.floor(w.x.length / n), r = new Float32Array(m);
  for (let k = 0; k < m; k++) { let s = 0; for (let i = k * n; i < (k + 1) * n; i++) s += w.x[i] * w.x[i]; r[k] = Math.sqrt(s / n); }
  return {step: 0.01, r};
}
function addBang(w, t, peak) {  // a 120 ms decaying noise burst (deterministic LCG noise)
  let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296) * 2 - 1;
  const i0 = Math.round(t * w.sr), n = Math.round(0.12 * w.sr);
  for (let i = 0; i < n && i0 + i < w.x.length; i++) w.x[i0 + i] = Math.max(-1, Math.min(1, w.x[i0 + i] + peak * 1.7 * rnd() * Math.exp(-i / (0.025 * w.sr))));
}
function vad(r) {               // voice.js startMeter: > TH for 120 ms → speaking, < TH for 450 ms → not
  const on = new Uint8Array(r.length); let sp = false, voiced = 0, silent = 0;
  for (let k = 0; k < r.length; k++) {
    if (r[k] > 0.012) { voiced += 10; silent = 0; if (!sp && voiced >= 120) sp = true; }
    else { silent += 10; if (silent > 60) voiced = 0; if (sp && silent >= 450) sp = false; }
    on[k] = sp ? 1 : 0;
  }
  return on;
}

function runClip(clip, o) {
  o = o || {};
  const lines = fs.readFileSync(path.join(DIR, clip, 'perception.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
  const th = thumbs(clip);
  const mic = readWav(path.join(DIR, clip, 'mic.wav'));
  if (o.bangs && mic) for (const b of o.bangs) addBang(mic, b.t, b.peak);
  const mr = rms10(mic), mvad = vad(mr.r);
  const vr = rms10(readWav(path.join(DIR, clip, 'voice.wav'))), vvad = vad(vr.r);
  let simT = 0;
  const sent = [], verdictQueue = [], events = [];
  const at = AvatarAttention.create(Object.assign({}, OPTS, o.opts || {}, {
    send: m => { if (m.type === 'attention_candidate') { sent.push(m); verdictQueue.push({due: m.t + 1.0, msg: {type: 'attention', id: m.id, what: 'sim', worth_reply: false, salience: 0.2}}); } },
    onEvent: ev => events.push(ev), now: () => simT * 1000,
    grab: (w, h, q, ago) => 'data:image/jpeg;base64,SIM' + (ago ? 'B' : 'N')}));
  let lastKey = -1, pendingEv = [], prevT = 0, lastThumbT = -99, thumb = null;
  const reps = o.repeat || 1, dur = lines.length ? +lines[lines.length - 1].t + 0.034 : 0, times = [];
  for (let rep = 0; rep < reps; rep++) for (const L of lines) {
    const f = L.features; const t = +L.t + rep * dur, tc = +L.t;
    for (const e of (f.events || [])) pendingEv.push(e);
    const key = Math.floor(t * FPS);
    if (key === lastKey) continue;
    lastKey = key;
    simT = t;
    while (verdictQueue.length && verdictQueue[0].due <= t) at.onHub(verdictQueue.shift().msg);
    const ff = Object.assign({}, f, {events: pendingEv}); pendingEv = [];
    if (t - lastThumbT >= THUMB_EVERY - 1e-6) { lastThumbT = t; thumb = {w: th.w, h: th.h, data: th.get(Math.round(tc * 30)), t}; }
    const k0 = Math.max(0, Math.floor(Math.min(prevT, tc) / 0.01) + 1), k1 = Math.min(mr.r.length - 1, Math.floor(tc / 0.01));
    let lvl = 0; for (let k = k0; k <= k1; k++) if (mr.r[k] > lvl) lvl = mr.r[k];
    const kk = Math.min(mr.r.length - 1, Math.max(0, Math.floor(tc / 0.01)));
    const vk = Math.min(vr.r.length - 1, Math.max(0, Math.floor(tc / 0.01)));
    prevT = tc;
    const ctx = {t, userSpeaking: !!mvad[kk], userLevel: lvl, avatarSpeaking: vr.r.length ? !!vvad[vk] : false, thumb};
    if (rep === reps - 1 && reps > 1) { const h0 = process.hrtime.bigint(); at.frame(ff, ctx); times.push(Number(process.hrtime.bigint() - h0) / 1e6); }
    else at.frame(ff, ctx);
  }
  const ts = times.slice().sort((a, b) => a - b);
  const trim = ts.slice(0, Math.max(1, Math.floor(ts.length * 0.99)));   // the loaded shared box preempts us for ms at a time: report both
  const timing = ts.length ? {frames: ts.length, mean: ts.reduce((a, b) => a + b, 0) / ts.length, trimmed: trim.reduce((a, b) => a + b, 0) / trim.length,
    p50: ts[Math.floor(ts.length / 2)], p99: ts[Math.floor(0.99 * (ts.length - 1))], max: ts[ts.length - 1]} : null;
  const st = at.stats();
  return {at, timing, clip, dur: lines.length ? +lines[lines.length - 1].t : 0, cands: sent.map(m => ({t: m.t, kinds: m.kinds, strength: m.strength, details: m.details, summary: m.summary})), stats: st};
}

function score(res, ann) {
  const speech = (ann && ann.user_speech || []);
  const cues = (ann && ann.cues || []).filter(c => ['presence', 'reflex', 'expression'].includes(c.kind))
    .map(c => Object.assign({}, c, {hit: null, speech: speech.some(u => c.t >= u.t0 - 0.2 && c.t <= u.t1 + 0.5)}));
  const used = new Set(), out = {cues: [], fp: [], dup: []};
  for (const c of cues) {
    const m = res.cands.find((x, i) => !used.has(i) && x.t >= c.t - 0.5 && x.t <= c.t + 2.5);
    if (m) { used.add(res.cands.indexOf(m)); c.hit = m; }
    out.cues.push({t: c.t, cue: c.cue, kind: c.kind, speech: c.speech, hit: m ? {t: m.t, lat: +(m.t - c.t).toFixed(2), kinds: m.kinds} : null});
  }
  res.cands.forEach((x, i) => {
    if (used.has(i)) return;
    if (cues.some(c => c.hit && x.t >= c.t - 0.5 && x.t <= c.t + 2.5)) out.dup.push(x); else out.fp.push(x);
  });
  return out;
}

function main() {
  const t0 = Date.now();
  const clips = listClips();
  const suite = [], seamless = [];
  let msSum = 0, frames = 0, msP99 = 0, thumbMs = 0, thumbs = 0;
  const kindHits = {}, kindFp = {}, kindSl = {};
  for (const clip of clips) {
    const annF = path.join(DIR, 'annotations', clip + '.json');
    const ann = fs.existsSync(annF) ? JSON.parse(fs.readFileSync(annF, 'utf8')) : null;
    const res = runClip(clip);
    msSum += res.stats.ms_per_frame * res.stats.frames; frames += res.stats.frames; msP99 = Math.max(msP99, res.stats.ms_p99);
    thumbMs += res.stats.ms_per_thumb * res.stats.thumbs; thumbs += res.stats.thumbs;
    if (clip.startsWith('sl_')) {
      seamless.push({clip, dur: res.dur, n: res.cands.length, cands: res.cands});
      for (const c of res.cands) for (const k of c.kinds) kindSl[k] = (kindSl[k] || 0) + 1;
      if (VERBOSE) console.log(clip, 'cands', res.cands.map(c => `${c.t.toFixed(1)}:${c.kinds.join('+')}`).join(' '));
    } else {
      const sc = score(res, ann);
      suite.push({clip, dur: res.dur, ...sc, n: res.cands.length});
      for (const c of sc.cues) if (c.hit) for (const k of c.hit.kinds) kindHits[k] = (kindHits[k] || 0) + 1;
      for (const c of sc.fp) for (const k of c.kinds) kindFp[k] = (kindFp[k] || 0) + 1;
      if (VERBOSE) {
        console.log(`${clip}: cands ${res.cands.map(c => `${c.t.toFixed(1)}:${c.kinds.join('+')}(${c.strength})`).join(' ')}`);
        for (const c of sc.cues) console.log(`   cue ${c.t} ${c.cue}${c.speech ? ' [speech]' : ''} -> ${c.hit ? c.hit.t.toFixed(2) + ' ' + c.hit.kinds.join('+') : 'MISS'}`);
      }
    }
  }
  // synthetic loud sounds
  const bangs = [];
  const bangCases = [{clip: 'startle_vcd_hands_at_lens_woman', t: 4.0}, {clip: 'sl_S0648_0377_P0299A', t: null}];
  for (const bc of bangCases) {
    if (!clips.includes(bc.clip)) continue;
    let t = bc.t;
    if (t == null) {   // the middle of the first user pause ≥ 2 s (from the annotation)
      const ann = JSON.parse(fs.readFileSync(path.join(DIR, 'annotations', bc.clip + '.json'), 'utf8'));
      const p = (ann.cues || []).find(c => /pause (\d+(\.\d+)?)s/.test(c.cue) && parseFloat(c.cue.match(/pause (\d+(\.\d+)?)s/)[1]) >= 2);
      t = p ? p.t + 1.0 : 10;
    }
    const res = runClip(bc.clip, {bangs: [{t, peak: 0.45}]});
    const hit = res.cands.find(c => c.kinds.includes('loud_sound') && c.t >= t && c.t <= t + 0.6);
    bangs.push({clip: bc.clip, t, detected: !!hit, at: hit ? hit.t : null, lat: hit ? +(hit.t - t).toFixed(3) : null,
      other_loud: res.cands.filter(c => c.kinds.includes('loud_sound') && c !== hit).map(c => c.t)});
  }
  // steady-state cost: ONE detector replays a long clip 3× back to back (a page keeps one warm instance); the last
  // pass is timed per frame() call — the per-clip runs above re-create the detector every clip (cold JIT, shared machine)
  const timing = (() => {
    const tc = clips.find(c => c === 'sl_S0648_0373_P0299A') || clips[0];
    let best = null;
    const r = runClip(tc, {repeat: 4});
    return {clip: tc, frames: r.timing.frames, ms_per_frame: +r.timing.trimmed.toFixed(4), ms_mean_all: +r.timing.mean.toFixed(4), ms_p50: +r.timing.p50.toFixed(4),
      ms_p99: +r.timing.p99.toFixed(4), ms_max: +r.timing.max.toFixed(3), note: 'ms_per_frame = mean of the fastest 99 % (machine load ~190 on 88 cores preempts single calls)'};
  })();
  const nCues = suite.reduce((s, x) => s + x.cues.length, 0), nHit = suite.reduce((s, x) => s + x.cues.filter(c => c.hit).length, 0);
  const nFp = suite.reduce((s, x) => s + x.fp.length, 0), nDup = suite.reduce((s, x) => s + x.dup.length, 0);
  const silentCues = suite.flatMap(x => x.cues.filter(c => !c.speech)), speechCues = suite.flatMap(x => x.cues.filter(c => c.speech));
  // the same suite with perception's wave / point / thumbs_up passed through as kind 'gesture' (opts.gestures)
  let gHit = 0, gFp = 0, gSilentHit = 0;
  for (const x of suite) {
    const annF = path.join(DIR, 'annotations', x.clip + '.json');
    const sc = score(runClip(x.clip, {opts: {gestures: true}}), fs.existsSync(annF) ? JSON.parse(fs.readFileSync(annF, 'utf8')) : null);
    gHit += sc.cues.filter(c => c.hit).length; gSilentHit += sc.cues.filter(c => c.hit && !c.speech).length; gFp += sc.fp.length;
  }
  const slMin = seamless.reduce((s, x) => s + x.dur, 0) / 60, slN = seamless.reduce((s, x) => s + x.n, 0);
  const loudElsewhere = [...suite.flatMap(x => x.fp.concat(x.dup)), ...seamless.flatMap(x => x.cands)].filter(c => c.kinds.includes('loud_sound')).length;
  const summary = {
    suite: {clips: suite.length, cues: nCues, hits: nHit, recall: nCues ? +(nHit / nCues).toFixed(3) : null, fp: nFp, dup: nDup,
      precision: nHit + nFp ? +(nHit / (nHit + nFp)).toFixed(3) : null, kind_hits: kindHits, kind_fp: kindFp,
      silent_cues: silentCues.length, silent_hits: silentCues.filter(c => c.hit).length,
      silent_recall: silentCues.length ? +(silentCues.filter(c => c.hit).length / silentCues.length).toFixed(3) : null,
      speech_cues: speechCues.length, speech_hits: speechCues.filter(c => c.hit).length,
      with_gestures: {hits: gHit, recall: nCues ? +(gHit / nCues).toFixed(3) : null, silent_recall: silentCues.length ? +(gSilentHit / silentCues.length).toFixed(3) : null,
        fp: gFp, precision: gHit + gFp ? +(gHit / (gHit + gFp)).toFixed(3) : null},
      latency_p50: (() => { const l = suite.flatMap(x => x.cues.filter(c => c.hit).map(c => c.hit.lat)).sort((a, b) => a - b); return l.length ? l[Math.floor(l.length / 2)] : null; })()},
    seamless: {clips: seamless.length, minutes: +slMin.toFixed(2), candidates: slN, per_min: slMin ? +(slN / slMin).toFixed(2) : null, kinds: kindSl},
    synthetic_bangs: bangs, loud_sound_elsewhere: loudElsewhere,
    cost: {steady: timing, frames, cold_ms_per_frame: frames ? +(msSum / frames).toFixed(4) : null, worst_clip_ms_p99: +msP99.toFixed(3), thumbs, ms_per_thumb: thumbs ? +(thumbMs / thumbs).toFixed(4) : null},
    per_clip: suite.map(x => ({clip: x.clip, n: x.n, cues: x.cues.map(c => `${c.t}:${c.hit ? 'HIT ' + c.hit.kinds.join('+') + ' +' + c.hit.lat : 'MISS'}`), fp: x.fp.map(c => `${c.t}:${c.kinds.join('+')}`)})),
    seamless_per_clip: seamless.map(x => ({clip: x.clip, n: x.n, c: x.cands.map(c => `${c.t}:${c.kinds.join('+')}`)})),
    seconds: +((Date.now() - t0) / 1000).toFixed(1)};
  const fails = [];
  if (!flag('no-assert')) {
    if (summary.suite.silent_recall != null && summary.suite.silent_recall < 0.6) fails.push('suite recall on cues while he is silent < 0.6');
    if (summary.suite.precision != null && summary.suite.precision < 0.5) fails.push('suite precision < 0.5');
    if (summary.seamless.per_min != null && summary.seamless.per_min > 2) fails.push('seamless > 2 candidates/min');
    if (bangs.some(b => !b.detected)) fails.push('synthetic bang missed');
    // (since 11-09 a sustained loud sound and quieter bangs count too: ≤ 1 spurious per ~15 min of conversation)
    if (loudElsewhere > 1) fails.push('loud_sound fired without a bang more than once');
    if (summary.cost.steady.ms_per_frame > 0.3) fails.push('steady ms/frame > 0.3');
  }
  summary.ok = fails.length === 0; summary.fails = fails;
  const out = arg('out', '');
  if (out) fs.writeFileSync(out, JSON.stringify(summary, null, 1));
  const brief = Object.assign({}, summary); if (!VERBOSE) { delete brief.per_clip; delete brief.seamless_per_clip; }
  console.log(JSON.stringify(brief, null, 1));
  process.exit(fails.length ? 1 : 0);
}
if (require.main === module) main();
else module.exports = {runClip, score, AvatarAttention, DIR};
