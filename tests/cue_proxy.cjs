// node tests/cue_proxy.cjs [--clips a,b] [--preset default] [--seeds 3] — free, fast stand-in for the eval's cue-hit metric on
// the TRAIN clips: their perception logs (tests/perception_smoke.cjs with PERC_CLIPS_DIR/PERC_OUT) + mic envelope →
// motion.js (commands injected 0.6 s after the spoken command, mock replies after his speech) → per-region motion
// energy from the rig (|Δparam| per frame, weighted like visible pixels) → the eval's onset rule (avmetrics.py: baseline
// = median of the 2 s before the cue, thr = base + max(4·MAD·1.4826, 0.05 + 0.3·base), sustained 0.15 s, in window).
'use strict';
const fs = require('fs'), path = require('path');
const M = require('./load.cjs')('app/motion.js'), PERC = require('./load.cjs')('app/perception.js');
const XP = !process.argv.includes('--no-xp');   // run perception's expression tracker over the logged faces (as the browser does)
const T = (process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), D = T + '/train', ANN = T + '/eval_clips/annotations';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const ids = (arg('clips', '') || fs.readFileSync(D + '/train_ids.txt', 'utf8')).split(/[,\s]+/).filter(Boolean);
const PRESET = arg('preset', 'default'), SEEDS = +arg('seeds', 3), DT = 1 / 30;
const CMD = [[/look at me/, {action: 'look', target: 'user'}], [/big smile/, {action: 'expression', name: 'big_smile', intensity: 0.9}],
  [/nod/, {action: 'gesture', name: 'nod', repeat: 1}], [/turn around/, {action: 'gesture', name: 'turn_around'}],
  [/wave/, {action: 'gesture', name: 'wave', side: 'r', repeat: 1}], [/ears|request ends/, {action: 'gesture', name: 'ear_wiggle', repeat: 1}],
  [/wink/, {action: 'gesture', name: 'wink', side: 'r'}]];
const W = {   // visible-motion weights per region (rig units → roughly comparable pixel motion)
  head: {head_yaw: 8, head_pitch: 8, head_roll: 6, gaze_x: 0.6, gaze_y: 0.6, blink_l: 0.25, blink_r: 0.25, head_z: 6},
  face: {head_yaw: 4, head_pitch: 4, head_roll: 3, smile: 2, squint: 1.2, eye_wide: 1.5, brow_l: 1, brow_r: 1, brow_knit: 1, mouth_open: 2, ear_l: 0.5, ear_r: 0.5, blink_l: 0.2, blink_r: 0.2},
  mouth: {smile: 3, mouth_open: 3, tongue_out: 3, head_pitch: 2},
  ears: {ear_l: 1, ear_r: 1, ear_l_turn: 0.6, ear_r_turn: 0.6, ear_flick_l: 0.6, ear_flick_r: 0.6},
  arms: {hand_l_x: 1, hand_l_y: 1, hand_l_z: 1, hand_r_x: 1, hand_r_y: 1, hand_r_z: 1, hand_l_open: 0.3, hand_r_open: 0.3, shoulder_l: 0.5, shoulder_r: 0.5},
  body: {body_yaw: 1, body_lean: 1, body_pitch: 1, body_bounce: 1, head_z: 6, head_yaw: 2},
};
W.eyes = W.head; W.any = Object.assign({}, W.head, W.face, W.arms, W.body, W.ears);
const smooth = (a, n) => a.map((_, i) => { let s = 0, c = 0; for (let j = Math.max(0, i - (n >> 1)); j <= Math.min(a.length - 1, i + (n >> 1)); j++) { s += a[j]; c++; } return s / c; });
const median = a => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0; };

function load(id) {
  const fp = path.join(D, 'perc', id + '.jsonl');
  if (!fs.existsSync(fp)) return null;
  const feats = [], evs = [];
  for (const l of fs.readFileSync(fp, 'utf8').split('\n')) {
    if (!l.trim()) continue; let j; try { j = JSON.parse(l); } catch (e) { continue; }
    if (j.event) { if (!/^model_/.test(j.event.type)) evs.push(j.event); } else if (j.t !== undefined && !j.stats) feats.push(j);
  }
  if (!feats.length) return null;
  const mic = JSON.parse(fs.readFileSync(path.join(D, 'mic', id + '.json'), 'utf8')).rms;
  const ann = JSON.parse(fs.readFileSync(path.join(ANN, id + '.json'), 'utf8'));
  return {feats, evs, mic, ann};
}
// his speech spans from the mic (for mock replies)
function spans(mic) {
  const peak = Math.max(...mic, 1e-4), thr = Math.max(0.01, 0.08 * peak), out = []; let s = null, quiet = 0;
  mic.forEach((v, i) => { if (v > thr) { if (s === null) s = i; quiet = 0; } else if (s !== null && ++quiet > 15) { out.push([s / 30, (i - quiet) / 30]); s = null; quiet = 0; } });
  if (s !== null) out.push([s / 30, mic.length / 30]);
  return out.filter(([a, b]) => b - a > 0.4);
}
function runClip(c, seed) {
  const m = M.createMotion({seed, preset: PRESET});
  const t0 = c.feats[0].t, dur = Math.min(c.feats[c.feats.length - 1].t - t0, c.mic.length / 30);
  const sp = spans(c.mic), replies = sp.map(([a, b]) => [b + 1.4, b + 1.4 + 2.5]);
  const cmds = (c.ann.cues || []).filter(q => q.kind === 'command').map(q => { const k = CMD.find(([re]) => re.test(String(q.cue).toLowerCase())); return k ? {t: q.t + 0.6, cmd: k[1]} : null; }).filter(Boolean);
  let fi = 0, ei = 0, f = null, ci = 0; const rows = [], tr = PERC.createExpressionTracker();
  for (let t = 0; t <= dur; t += DT) {
    const events = [];
    while (fi < c.feats.length && c.feats[fi].t - t0 <= t) {
      f = Object.assign({}, c.feats[fi++], {events: undefined});
      if (XP) { const r = tr.update(f.face, f.t - t0); f.face = Object.assign({}, f.face, {xp: {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt}}); events.push(...r.events.filter(e => e.type !== 'expression_change')); }
    }
    while (ei < c.evs.length && c.evs[ei].t - t0 <= t) events.push(c.evs[ei++]);
    while (ci < cmds.length && cmds[ci].t <= t) m.command(cmds[ci++].cmd);
    const mi = Math.min(c.mic.length - 1, Math.round(t * 30)), urms = c.mic[mi] || 0;
    const talking = replies.some(([a, b]) => t >= a && t <= b);
    const arms = talking ? 0.05 * Math.max(0, Math.sin(t * 27)) + 0.01 : 0;
    rows.push(m.step(DT, {features: f, events, userVoice: {rms: urms}, avatarVoice: {rms: arms}}));
  }
  return rows;
}
function cueHits(rows, cues) {
  const out = [];
  for (const q of cues || []) {
    const reg = ({head: 'head', face: 'face', eyes: 'eyes', mouth: 'mouth', ears: 'ears', arms: 'arms', body: 'body'})[String(q.region || 'any').toLowerCase()] || 'any';
    const w = W[reg], e = rows.map((r, i) => { if (!i) return 0; let s = 0; for (const k in w) s += w[k] * Math.abs(r[k] - rows[i - 1][k]); return s * 30; });
    const s = smooth(e, 3), it = Math.round(q.t * 30), b0 = Math.max(0, Math.round((q.t - 2) * 30));
    const base = median(s.slice(b0, it)), mad = median(s.slice(b0, it).map(v => Math.abs(v - base)));
    const thr = base + Math.max(4 * mad * 1.4826, 0.05 + 0.3 * base);
    const [w0, w1] = q.window_s || [0.2, 1.5], end = Math.min(s.length, Math.round((q.t + w1 + 1) * 30)), need = 5;
    let onset = null;
    for (let i = Math.max(0, it); i <= end - need; i++) { let ok = true; for (let j = 0; j < need; j++) if (!(s[i + j] > thr)) { ok = false; break; } if (ok) { onset = (i - 0.5) / 30; break; } }
    const lat = onset === null ? null : Math.max(0, onset - q.t);
    out.push({cue: q.cue, kind: q.kind, region: reg, t: q.t, lat, hit: lat !== null && lat >= w0 - 0.1 && lat <= w1});
  }
  return out;
}
const agg = {}, perClip = [];
for (const id of ids) {
  const c = load(id); if (!c) { console.log('skip (no perception log)', id); continue; }
  const hits = []; let ev = {};
  for (const e of c.evs) ev[e.type] = (ev[e.type] || 0) + 1;
  for (let s = 1; s <= SEEDS; s++) hits.push(...cueHits(runClip(c, s * 7 + 1), c.ann.cues));
  const by = {};
  for (const h of hits) { const k = h.kind || 'other'; (by[k] = by[k] || []).push(h.hit); (agg[k] = agg[k] || []).push(h.hit); }
  const rate = hits.length ? hits.filter(h => h.hit).length / hits.length : null;
  perClip.push({id, rate, n: hits.length / SEEDS, ev});
  console.log(`${id.padEnd(44)} hit ${rate === null ? '  – ' : (rate * 100).toFixed(0).padStart(3) + '%'}  ${JSON.stringify(ev)}  ` +
    hits.slice(0, hits.length / SEEDS).map((h, i) => `${h.cue.slice(0, 18)}:${[0, 1, 2].slice(0, SEEDS).map(s => hits[i + s * hits.length / SEEDS].hit ? '✓' : '·').join('')}`).join(' '));
}
const all = Object.values(agg).flat();
console.log('\nby kind: ' + Object.entries(agg).map(([k, v]) => `${k} ${(100 * v.filter(Boolean).length / v.length).toFixed(0)}% (${v.length})`).join(' · '));
console.log(`ALL cue-hit proxy ${(100 * all.filter(Boolean).length / all.length).toFixed(1)}% over ${all.length} cue×seed (preset ${PRESET})`);
fs.writeFileSync(path.join(D, 'proxy_' + PRESET + '.json'), JSON.stringify({perClip, agg}, null, 1));
