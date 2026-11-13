// node tests/rules_golden.cjs [--dump FILE] — deterministic rules-mode fingerprint of motion.js.
// Replays the recorded perception logs (read-only, scratch) + a scripted event/command sequence through
// createMotion({seed}) at 30 Hz and prints one sha256 over every output vector. Run before and after a change:
// the hash must not move (rules mode byte-for-byte unchanged). Writes nothing unless --dump is given.
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const M = require('./load.cjs')('app/motion.js');
const PERC = require('./load.cjs')('app/perception.js');
const DIR = path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'perception');
const DT = 1 / 30;
const H = crypto.createHash('sha256');
const dumpAt = process.argv.indexOf('--dump');
const dump = dumpAt > 0 ? [] : null;
const opts = JSON.parse(process.env.GOLDEN_OPTS || '{}');   // e.g. {"learned":true} to fingerprint the learned build's rules output

function feed(name, rows) {
  for (const o of rows) H.update(JSON.stringify(o));
  if (dump) dump.push({name, rows});
}
function replay(clip) {
  const feats = [], evs = [];
  for (const line of fs.readFileSync(path.join(DIR, clip + '.jsonl'), 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let j; try { j = JSON.parse(line); } catch (e) { continue; }
    if (j.event) { if (!/^model_/.test(j.event.type)) evs.push(j.event); } else if (j.t !== undefined && !j.stats) feats.push(j);
  }
  if (!feats.length) return;
  const t0 = feats[0].t, t1 = Math.min(feats[feats.length - 1].t, t0 + 40);
  const m = M.createMotion(Object.assign({seed: 9, preset: 'default'}, opts)), tr = PERC.createExpressionTracker();
  let fi = 0, ei = 0, f = null; const rows = [];
  for (let t = t0, k = 0; t <= t1; t += DT, k++) {
    const events = [];
    while (fi < feats.length && feats[fi].t <= t) {
      f = Object.assign({}, feats[fi++], {events: undefined});
      const r = tr.update(f.face, f.t); f.face = Object.assign({}, f.face, {xp: {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt}});
      for (const e of r.events) events.push(e);
    }
    while (ei < evs.length && evs[ei].t <= t) events.push(evs[ei++]);
    const sp = Math.sin(k / 7) > 0.3, uv = {rms: Math.cos(k / 11) > 0.5 ? 0.05 : 0.001, pitch: 140};
    if (k === 200) m.command({action: 'gesture', name: 'wave', side: 'left'});
    if (k === 400) m.command({action: 'pose', targets: {hand_l_y: 0.8, head_yaw: 0.3}, hold: true});
    if (k === 600) m.command({action: 'release', parts: 'all'});
    if (k === 700) m.command({action: 'expression', name: 'surprised', intensity: 1});
    if (k === 800) m.command({action: 'look', target: 'left'});
    if (k === 900) m.command({action: 'hold'});
    if (k === 1000) m.command({action: 'release', parts: 'all'});
    const o = m.step(DT, {features: f, events, avatarVoice: {rms: sp ? 0.04 : 0, voiced: sp, pitch: sp ? 180 : 0}, userVoice: uv, speaking: sp, listening: !sp});
    rows.push(m.vector(o));
  }
  feed(clip, rows);
}
// a synthetic run with every reflex event
{
  const m = M.createMotion(Object.assign({seed: 3, preset: 'default'}, opts)); const rows = [];
  const EV = {30: {type: 'face_appear'}, 90: {type: 'wave', side: 'r'}, 200: {type: 'approach_fast', rate: 1.2}, 300: {type: 'look_back', away_s: 3},
    360: {type: 'point', dir: {x: 0.5, y: 0.1}}, 420: {type: 'object_up', x: 0.2, y: 0.1}, 480: {type: 'thumbs_up', side: 'l'},
    540: {type: 'nod'}, 600: {type: 'head_shake'}, 700: {type: 'face_lost'}};
  for (let k = 0; k < 800; k++) {
    const t = k * DT, present = k < 700;
    const face = present ? {present: true, x: 0.1 * Math.sin(t), y: 0.05, z: 0.45, yaw: 0.1 * Math.sin(t / 2), pitch: 0, roll: 0.05, blend: {mouthSmileLeft: k > 100 && k < 250 ? 0.6 : 0.05, mouthSmileRight: k > 100 && k < 250 ? 0.6 : 0.05}} : {present: false};
    const ev = EV[k] ? [Object.assign({t: k}, EV[k])] : [];
    rows.push(m.vector(m.step(DT, {features: {t, face, attending: true, attention: 0.9, motion: 0.05, hands: []}, events: ev})));
  }
  feed('synthetic_events', rows);
}
const clips = fs.readdirSync(DIR).filter(n => n.endsWith('.jsonl')).map(n => n.slice(0, -6)).sort().slice(0, 8);
for (const c of clips) replay(c);
console.log('rules golden sha256', H.digest('hex'), 'clips', clips.length + 1);
if (dump) fs.writeFileSync(process.argv[dumpAt + 1], JSON.stringify(dump));
