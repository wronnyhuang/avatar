// node tests/motion_replay.cjs [dir] — replays the perception feature logs of the real clips (data/perception/<clip>.jsonl,
// written by tests/perception_smoke.cjs) through motion.js at 30 Hz and checks clip-level behaviour.
'use strict';
const fs = require('fs'), path = require('path');
const M = require('./load.cjs')('app/motion.js');
const PERC = require('./load.cjs')('app/perception.js');   // logs predate face.xp: run the same tracker the browser runs
const XP = process.argv.includes('--no-xp') ? false : true;
const DIR = process.argv[2] || path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'perception');
const OUT = path.join(path.join(require('os').tmpdir(), 'avatar-tests'), 'replay');
fs.mkdirSync(OUT, {recursive: true});
const DT = 1 / 30;
const results = [];
const check = (name, ok, detail) => { results.push({name, ok: !!ok, detail}); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  — ' + detail : '')); };

function replay(clip) {
  const fp = path.join(DIR, clip + '.jsonl');
  if (!fs.existsSync(fp)) return null;
  const feats = [], evs = [];
  for (const line of fs.readFileSync(fp, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let j; try { j = JSON.parse(line); } catch (e) { continue; }
    if (j.event) { if (!/^model_/.test(j.event.type)) evs.push(j.event); }
    else if (j.t !== undefined && !j.stats) feats.push(j);
  }
  if (!feats.length) return null;
  const t0 = feats[0].t, t1 = feats[feats.length - 1].t;
  const m = M.createMotion({seed: 9}), tr = PERC.createExpressionTracker(), xev = {};
  let fi = 0, ei = 0, f = null;
  const rows = [], rig = [];
  for (let t = t0; t <= t1; t += DT) {
    const events = [];
    while (fi < feats.length && feats[fi].t <= t) {
      f = Object.assign({}, feats[fi++], {events: undefined});
      if (XP) { const r = tr.update(f.face, f.t); f.face = Object.assign({}, f.face, {xp: {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt}}); for (const e of r.events) { events.push(e); xev[e.type] = (xev[e.type] || 0) + 1; } }
    }
    while (ei < evs.length && evs[ei].t <= t) events.push(evs[ei++]);
    const o = m.step(DT, {features: f, events});
    rows.push(Object.assign({t: t - t0, mode: m.debug().mode}, o));
    rig.push(JSON.stringify({type: 'rig', t: +(t - t0).toFixed(4), a: m.vector(o)}));
  }
  fs.writeFileSync(path.join(OUT, clip + '.rig.jsonl'), JSON.stringify({rig_params: M.RIG_PARAMS}) + '\n' + rig.join('\n') + '\n');
  const evCount = {}; for (const e of evs) evCount[e.type] = (evCount[e.type] || 0) + 1;
  const pres = feats.filter(x => x.face && x.face.present);
  return {rows, log: m.state.log, evCount: Object.assign(evCount, xev), faceRatio: pres.length / feats.length, attending: pres.filter(x => x.attending).length / Math.max(1, pres.length)};
}
const std = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
const wavebacks = r => r.log.filter(e => e.gesture === 'wave' && e.src === 'reflex').length;

const clips = fs.readdirSync(DIR).filter(n => n.endsWith('.jsonl')).map(n => n.slice(0, -6));
for (const clip of clips) {
  const r = replay(clip); if (!r) continue;
  const hy = r.rows.map(q => q.head_yaw), gx = r.rows.map(q => q.gaze_x), up = r.rows.filter(q => q.hand_l_y > 0.5 || q.hand_r_y > 0.5).length * DT;
  const startles = r.rows.filter((q, i) => i && q.head_z < -0.04 && r.rows[i - 1].head_z >= -0.04).length;
  console.log(`-- ${clip}: ${r.rows.length} frames, face ${(r.faceRatio * 100).toFixed(0)}%, attending ${(r.attending * 100).toFixed(0)}%, events ${JSON.stringify(r.evCount)}, wave-backs ${wavebacks(r)}, paw raised ${up.toFixed(1)} s, startles ${startles}, head std ${std(hy).toFixed(3)}, gaze std ${std(gx).toFixed(2)}, smile max ${Math.max(...r.rows.map(q => q.smile)).toFixed(2)}`);
  if (r.rows.length >= 300) check(`${clip}: alive (head std ≥ 0.02, gaze moves)`, std(hy) >= 0.02 && std(gx) > 0.03, '');
  if (/wave/.test(clip)) check(`${clip}: waves back, once per greeting (paw raised ≤ 40% of the clip)`, wavebacks(r) >= 1 && up <= 0.4 * r.rows.length * DT, `${wavebacks(r)} wave-backs, ${up.toFixed(1)} s raised`);
  else check(`${clip}: no wave back without a wave`, wavebacks(r) === 0 || (r.evCount.wave || 0) > 0, `${wavebacks(r)} wave-backs`);
  // attention: people talking into the lens (YouTube) and at the screen count as attending; a man looking at his laptop does not
  if (/^(sam_|emotion_|vcd_face)/.test(clip)) check(`${clip}: attending while he looks into the lens/screen (≥ 0.8)`, r.attending >= 0.8, r.attending.toFixed(2));
  if (/^watch_object/.test(clip)) check(`${clip}: not attending while he looks at his laptop (≤ 0.4)`, r.attending <= 0.4, r.attending.toFixed(2));
  if (/^(sam_|emotion_)/.test(clip)) check(`${clip}: mostly mutual gaze (avatar mode 'user' ≥ 70%)`, r.rows.filter(q => q.mode === 'user').length / r.rows.length >= 0.7,
    (r.rows.filter(q => q.mode === 'user').length / r.rows.length).toFixed(2));
  if (/happy/.test(clip)) check(`${clip}: mirrors his smile`, Math.max(...r.rows.map(q => q.smile)) > 0.35, Math.max(...r.rows.map(q => q.smile)).toFixed(2));
  check(`${clip}: no startle without approach_fast`, startles === 0 || (r.evCount.approach_fast || 0) > 0, `${startles}`);
  const mins = r.rows.length * DT / 60, nodBacks = r.log.filter(e => e.nodback).length, quiz = r.log.filter(e => e.gesture === 'quizzical').length;
  console.log(`   reactions: nod-backs ${nodBacks} (${(nodBacks / mins).toFixed(1)}/min), quizzical ${quiz}, expression_change ${(r.evCount.expression_change || 0)} (${((r.evCount.expression_change || 0) / mins).toFixed(1)}/min)`);
  check(`${clip}: not trigger-happy (per ~21 s clip: nod-backs ≤ 3, quizzical ≤ 1, expression_change ≤ 3)`, nodBacks <= 3 && quiz <= 1 && (r.evCount.expression_change || 0) <= 3,
    `${nodBacks} / ${quiz} / ${r.evCount.expression_change || 0}`);
}
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 1));
const nf = results.filter(r => !r.ok).length;
console.log(`\n${results.length - nf}/${results.length} passed (rig logs → ${OUT})`);
process.exitCode = nf ? 1 : 0;
