/* turnbid_lib.cjs — shared data helpers for the turn-bid tests (node, CommonJS).
 *   loadUMD(path)            → module.exports of a UMD file (package.json is "type":"module", so app/*.js can't be require()d)
 *   readWav(path)            → {sr, x: Float32Array} (16-bit PCM mono, as the clip extractor's write_wav)
 *   rms10(x, sr)             → Float32Array of 10 ms RMS blocks (the page's mic meter block)
 *   micState(rms, opts)      → per-10ms {level, speaking} like voice.js startMeter: speaking on after 120 ms voiced
 *                              (> 0.012), off after 450 ms of silence
 *   avatarState(rms)         → per-10ms speaking like voice.js onAvatarChunk (voiced chunk → on; off after 400 ms quiet)
 *   wordsToSegs(words, gap)  → merged speech segments [[t0,t1]] from [[t0,t1,w]] words (gap ≤ gap s merged)
 *   loadClip(dir, opts)      → {id, frames:[f] (the app's feature frames, decimated to ~fps), mic, voice (per-10ms states),
 *                               dur}  — perception.jsonl lines {t, features, events}
 *   ctxAt(clip, t, opts)     → {t, userSpeaking, userLevel, avatarSpeaking}
 */
'use strict';
const fs = require('fs'), path = require('path');

function loadUMD(p) {
  const src = fs.readFileSync(p, 'utf8');
  const mod = {exports: {}};
  new Function('module', 'exports', src)(mod, mod.exports);
  return mod.exports;
}

function readWav(p) {
  const b = fs.readFileSync(p);
  let o = 12, sr = 24000, bits = 16, ch = 1, data = null;
  while (o + 8 <= b.length) {
    const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); bits = b.readUInt16LE(o + 22); }
    else if (id === 'data') { data = b.subarray(o + 8, o + 8 + n); break; }
    o += 8 + n + (n & 1);
  }
  if (!data || bits !== 16) throw Error('unsupported wav ' + p);
  const n = Math.floor(data.length / 2 / ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2 * ch) / 32768;
  return {sr, x};
}

function rms10(x, sr) {
  const n = Math.round(sr * 0.01), m = Math.floor(x.length / n), r = new Float32Array(m);
  for (let i = 0; i < m; i++) { let s = 0; for (let j = i * n; j < (i + 1) * n; j++) s += x[j] * x[j]; r[i] = Math.sqrt(s / n); }
  return r;
}

function micState(rms, opts) {
  const TH = (opts && opts.th) || 0.012, END = (opts && opts.endMs) || 450;
  const sp = new Uint8Array(rms.length);
  let voicedRun = 0, silentRun = 0, on = false;
  for (let i = 0; i < rms.length; i++) {
    if (rms[i] > TH) { voicedRun += 10; silentRun = 0; if (!on && voicedRun >= 120) on = true; }
    else { silentRun += 10; if (silentRun > 60) voicedRun = 0; if (on && silentRun >= END) on = false; }
    sp[i] = on ? 1 : 0;
  }
  return {level: rms, speaking: sp};
}

function avatarState(rms, opts) {
  const TH = (opts && opts.th) || 0.012, OFF = (opts && opts.offMs) || 400;
  const sp = new Uint8Array(rms.length);
  let on = false, quiet = 0;
  for (let i = 0; i < rms.length; i++) {
    if (rms[i] > TH) { on = true; quiet = 0; } else { quiet += 10; if (on && quiet > OFF) on = false; }
    sp[i] = on ? 1 : 0;
  }
  return {level: rms, speaking: sp};
}

function wordsToSegs(words, gap) {
  const s = [];
  for (const w of (words || []).slice().sort((a, b) => a[0] - b[0])) {
    const a = +w[0], b = +w[1];
    if (s.length && a - s[s.length - 1][1] <= gap) s[s.length - 1][1] = Math.max(s[s.length - 1][1], b);
    else s.push([a, b]);
  }
  return s;
}

// voiced segments of a per-10ms state array: [[t0,t1]] where speaking==1
function segsOf(sp) {
  const out = []; let a = -1;
  for (let i = 0; i <= sp.length; i++) {
    const v = i < sp.length ? sp[i] : 0;
    if (v && a < 0) a = i; else if (!v && a >= 0) { out.push([a * 0.01, i * 0.01]); a = -1; }
  }
  return out;
}

function loadClip(dir, opts) {
  opts = opts || {};
  const fps = opts.fps || 18;
  const lines = fs.readFileSync(path.join(dir, 'perception.jsonl'), 'utf8').split('\n').filter(Boolean);
  const frames = [];
  let lastK = -1;
  for (const l of lines) {
    const r = JSON.parse(l), f = r.features;
    if (!f) continue;
    const k = Math.floor(f.t * fps + 1e-6);
    if (k === lastK) continue;   // keep one frame per 1/fps slot: the page emits features at the face rate
    lastK = k;
    frames.push(f);
  }
  const micW = readWav(path.join(dir, 'mic.wav')), voiceW = fs.existsSync(path.join(dir, 'voice.wav')) ? readWav(path.join(dir, 'voice.wav')) : null;
  const mic = micState(rms10(micW.x, micW.sr));
  const voice = voiceW ? avatarState(rms10(voiceW.x, voiceW.sr)) : {level: new Float32Array(mic.level.length), speaking: new Uint8Array(mic.level.length)};
  return {id: path.basename(dir), dir, frames, mic, voice, dur: micW.x.length / micW.sr};
}

function ctxAt(clip, t) {
  const i = Math.max(0, Math.min(clip.mic.level.length - 1, Math.floor(t * 100)));
  const j = Math.max(0, Math.min(clip.voice.level.length - 1, Math.floor(t * 100)));
  return {t, userSpeaking: !!clip.mic.speaking[i], userLevel: clip.mic.level[i], avatarSpeaking: !!clip.voice.speaking[j]};
}

module.exports = {loadUMD, readWav, rms10, micState, avatarState, wordsToSegs, segsOf, loadClip, ctxAt};
