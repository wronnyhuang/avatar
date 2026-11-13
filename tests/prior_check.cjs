// node tests/prior_check.cjs [prior_dir] — the learned prior (branch prior: app/prior.js + app/prior_weights.json) through
// motion.setPrior: synthetic speaking/listening/idle + a real-clip replay; prints motion stats, blink rate and cost.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const M = load('app/motion.js');
const PD = process.argv[2] || path.join(__dirname, '..', 'app');
const src = fs.readFileSync(path.join(PD, 'prior.js'), 'utf8'), mod = {exports: {}};
new Function('module', 'exports', 'globalThis', src)(mod, mod.exports, {});
const createPrior = mod.exports.createPrior || mod.exports;
const weights = JSON.parse(fs.readFileSync(path.join(PD, 'prior_weights.json'), 'utf8'));
const DT = 1 / 30, results = [];
const check = (n, ok, d) => { results.push({n, ok: !!ok}); console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); };
const std = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
function syll(t, t0, t1) {
  if (t < t0 || t > t1) return {rms: 0, voiced: false};
  const u = t - t0, pu = u % 2.6; if (pu > 2.1) return {rms: 0.001, voiced: false};
  const syl = Math.floor(pu * 4.3), sp = pu * 4.3 - syl, stress = (syl * 7 + Math.floor(u / 2.6) * 3) % 5 === 0 ? 1 : 0.45;
  const env = Math.sin(Math.PI * Math.min(1, sp / 0.8)) ** 1.5; return {rms: 0.06 * stress * env + 0.004, voiced: env > 0.15};
}
function scenario(name, secs, inp, usePrior) {
  const m = M.createMotion({seed: 5});
  if (usePrior) check(name + ': setPrior(createPrior(weights)) accepted', m.setPrior(createPrior(weights)));
  const rows = []; let ms = 0;
  for (let i = 0; i < secs * 30; i++) { const t = (i + 1) * DT, a = process.hrtime.bigint(); const o = m.step(DT, inp(t)); ms += Number(process.hrtime.bigint() - a) / 1e6; rows.push(o); }
  const d = m.debug(), bl = rows.filter((q, i) => i && q.blink_l > 0.8 && rows[i - 1].blink_l <= 0.8).length / (secs / 60);
  const fin = rows.every(o => Object.values(o).every(Number.isFinite));
  const st = {yaw: std(rows.map(q => q.head_yaw)), pitch: std(rows.map(q => q.head_pitch)), roll: std(rows.map(q => q.head_roll))};
  let jump = 0; for (let i = 1; i < rows.length; i++) jump = Math.max(jump, Math.abs(rows[i].head_yaw - rows[i - 1].head_yaw), Math.abs(rows[i].head_pitch - rows[i - 1].head_pitch));
  console.log(`  ${name}${usePrior ? ' +prior' : ''}: head std yaw ${st.yaw.toFixed(3)} pitch ${st.pitch.toFixed(3)} roll ${st.roll.toFixed(3)} rad · blinks ${bl.toFixed(1)}/min · max head Δ/frame ${jump.toFixed(3)} · ${(ms / rows.length).toFixed(3)} ms/frame · priorErr ${d.priorErr}`);
  return {fin, st, bl, jump, err: d.priorErr, prior: d.prior};
}
const face = {present: true, x: 0.05, y: 0.05, ax: 0.035, ay: 0.026, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {mouthSmileLeft: 0.1, mouthSmileRight: 0.1}};
const sets = {
  speaking: t => ({features: {t, face, attending: true, hands: []}, avatarVoice: syll(t, 1, 29)}),
  listening: t => ({features: {t, face, attending: true, hands: []}, userVoice: syll(t, 1, 29)}),
  idle: t => ({features: {t, face, attending: true, hands: []}}),
};
for (const [n, f] of Object.entries(sets)) {
  const base = scenario(n, 30, f, false), pr = scenario(n, 30, f, true);
  check(`${n}: prior runs, finite, no error`, pr.fin && !pr.err && pr.prior, pr.err || '');
  check(`${n}: with the prior the head still moves (yaw+pitch std ≥ 0.02) and never snaps (Δ ≤ 0.09)`, pr.st.yaw + pr.st.pitch >= 0.02 && pr.jump <= 0.09, '');
  check(`${n}: blink rate stays human (8–35/min)`, pr.bl >= 8 && pr.bl <= 35, pr.bl.toFixed(1));
}
// real clip: Sam telling a story (perception log) with the prior on
const fp = path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'perception', 'sam_story.jsonl');
if (fs.existsSync(fp)) {
  const feats = fs.readFileSync(fp, 'utf8').split('\n').filter(l => l.startsWith('{"t"')).map(l => JSON.parse(l));
  let fi = 0, f = null; const t0 = feats[0].t;
  const r = scenario('sam_story replay', Math.floor(feats[feats.length - 1].t - t0), t => { while (fi < feats.length && feats[fi].t - t0 <= t) f = feats[fi++]; return {features: f}; }, true);
  check('sam_story replay: prior runs on real perception frames', r.fin && !r.err, r.err || '');
}
const nf = results.filter(r => !r.ok).length; console.log(`\n${results.length - nf}/${results.length} passed`); process.exitCode = nf ? 1 : 0;
