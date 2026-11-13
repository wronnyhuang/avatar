/* Shared helpers for the perception tests: load the page's UMD modules in node, read recorded perception
 * streams (eval clips: native MediaPipe at the page's rates through the real perception.js) and the
 * nod/shake face streams. */
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');
const CLIPS = process.env.AVATAR_EVAL_CLIPS || path.join(ROOT, 'data', 'eval_clips');
const NODSHAKE = process.env.AVATAR_NODSHAKE || path.join(process.env.AVATAR_DATA || path.join(ROOT, 'data'), 'nodshake');

function loadUmd(rel) {
  const src = fs.readFileSync(path.join(ROOT, rel), 'utf8');
  const mod = {exports: {}};
  new Function('module', 'exports', src)(mod, mod.exports);
  return mod.exports;
}
function clipIds(filter) {
  return fs.readdirSync(CLIPS).filter(d => fs.existsSync(path.join(CLIPS, d, 'perception.jsonl'))).filter(d => !filter || filter(d)).sort();
}
function receipt(id) { try { return JSON.parse(fs.readFileSync(path.join(CLIPS, id, 'receipt.json'), 'utf8')); } catch (e) { return {}; } }
function annotation(id) { try { return JSON.parse(fs.readFileSync(path.join(CLIPS, 'annotations', id + '.json'), 'utf8')); } catch (e) { return null; } }
// feature frames of a clip, with aspect filled from the receipt when the stream predates the aspect field
function loadStream(id) {
  const rc = receipt(id), wh = rc.wh || [640, 360], k = wh[1] / wh[0];
  const out = [];
  for (const line of fs.readFileSync(path.join(CLIPS, id, 'perception.jsonl'), 'utf8').split('\n')) {
    if (!line) continue;
    const d = JSON.parse(line), f = d.features;
    if (f.aspect == null) f.aspect = k;
    out.push(f);
  }
  return out;
}
// the nod / shake face-only streams (30 fps): {t, face:{present, blend, yaw, pitch, roll, matrix}} + labels
function nodshakeStreams(minNods) {
  const idx = JSON.parse(fs.readFileSync(path.join(NODSHAKE, 'index.json'), 'utf8'));
  const out = [];
  for (const s of idx.streams) {
    const name = `${s.clip}_s${s.k}`;
    const lab = JSON.parse(fs.readFileSync(path.join(NODSHAKE, 'labels', name + '.json'), 'utf8'));
    if ((lab.nods || []).length < (minNods || 0)) continue;
    out.push({name, lab, file: path.join(NODSHAKE, 'frames', name + '.jsonl')});
  }
  return out;
}
function loadNodshakeFrames(file) {
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).map(r => ({t: r.t, face: r.face, hands: [], pose: null, motion: 0, aspect: 0.5625}));
}
// mono 16-bit PCM WAV → RMS per `hop` seconds (linear, 0..1)
function wavRms(file, hop) {
  hop = hop || 0.02;
  const b = fs.readFileSync(file);
  let off = 12, sr = 24000, ch = 1, bits = 16, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(off + 10); sr = b.readUInt32LE(off + 12); bits = b.readUInt16LE(off + 22); }
    else if (id === 'data') { data = b.subarray(off + 8, off + 8 + sz); break; }
    off += 8 + sz + (sz & 1);
  }
  if (!data || bits !== 16) return {hop, rms: new Float32Array(0)};
  const n = Math.floor(data.length / 2 / ch), step = Math.round(sr * hop), out = new Float32Array(Math.floor(n / step));
  for (let k = 0; k < out.length; k++) { let s = 0; for (let i = k * step; i < (k + 1) * step; i++) { const x = data.readInt16LE(i * 2 * ch) / 32768; s += x * x; } out[k] = Math.sqrt(s / step); }
  return {hop, rms: out};
}
const pct = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const r3 = v => v == null || !isFinite(v) ? null : Math.round(v * 1000) / 1000;
function rng(seed) { let s = seed >>> 0 || 1; const r = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; r.gauss = () => { const u = Math.max(1e-9, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; return r; }
module.exports = {ROOT, CLIPS, NODSHAKE, loadUmd, clipIds, receipt, annotation, loadStream, nodshakeStreams, loadNodshakeFrames, wavRms, pct, r3, rng};
