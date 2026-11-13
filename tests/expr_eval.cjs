// node tests/expr_eval.cjs [--motion app/motion.js] — expression detection accuracy + avatar response latency on RAVDESS
// "sessions" (data/expr: per actor, 17 clips of 7 emotions concatenated; labels from the RAVDESS filenames; perception
// logs from tests/perception_smoke.cjs run on them). DETECTION = the tracker's state (perception.createExpressionTracker) or,
// for the old controller (no tracker), motion's own mirror channels argmax ≥ 0.15. RESPONSE = the avatar's rig: the channel
// that should move for that emotion rises ≥ thr over its level just before the clip; latency = his expression onset (his
// key blendshape deviation reaching 30% of its in-clip peak) → avatar response reaching 50% of its in-clip peak.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const M = load(arg('motion', 'app/motion.js')), P = load('app/perception.js');
const D = path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'expr'), DT = 1 / 30;
const ACCEPT = {happy: ['happy'], surprised: ['surprised', 'raised_brows'], angry: ['frown'], disgust: ['frown'], sad: ['sad', 'frown'],
  fearful: ['frown', 'sad', 'surprised'], neutral: ['neutral']};
// avatar channel that should respond, and its threshold
const RESP = {happy: [o => o.smile + 0.5 * o.squint, 0.15], surprised: [o => o.eye_wide + 0.5 * (o.brow_l + o.brow_r) / 2, 0.15],
  angry: [o => o.brow_knit - 0.5 * (o.brow_l + o.brow_r) / 2, 0.1], disgust: [o => o.brow_knit - 0.5 * (o.brow_l + o.brow_r) / 2, 0.1],
  sad: [o => o.brow_knit - 0.3 * (o.ear_l + o.ear_r) / 2 + o.head_pitch, 0.1], fearful: [o => o.brow_knit + o.eye_wide, 0.1]};
const KEY = {happy: d => d.smile, surprised: d => Math.max(d.innerUp, d.outerUp, -d.browDown), angry: d => d.browDown, disgust: d => d.browDown,
  sad: d => Math.max(d.browDown, -d.smile), fearful: d => Math.max(d.browDown, -d.smile, d.innerUp)};
const BL = ['mouthSmileLeft', 'mouthSmileRight', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'eyeSquintLeft', 'eyeSquintRight'];
const old = arg('old', '0') === '1';
const res = {}, lat = {}, resp = {};
for (const a of ['01', '02']) {
  const lines = fs.readFileSync(path.join(D, 'perc', 'ravdess_a' + a + '.jsonl'), 'utf8').split('\n').filter(l => l.startsWith('{"t"')).map(JSON.parse);
  const segs = JSON.parse(fs.readFileSync(path.join(D, 'ravdess_a' + a + '.labels.json'), 'utf8'));
  const t0 = lines[0].t, tr = P.createExpressionTracker(), m = M.createMotion({seed: 3});
  let fi = 0, f = null; const rows = [];
  // the raw deviation of his key channels vs his first-2-s mean (for the onset only)
  let base = null, nb = 0; const acc = {};
  for (let t = 0; t <= lines[lines.length - 1].t - t0; t += DT) {
    let fresh = false;
    while (fi < lines.length && lines[fi].t - t0 <= t) { f = Object.assign({}, lines[fi++]); fresh = true; }
    let xp = null;
    if (f && fresh) { xp = tr.update(f.face, f.t - t0); if (!old) f.face = Object.assign({}, f.face, {xp}); }
    const o = m.step(DT, {features: f && !old ? f : f});
    const b = (f && f.face && f.face.blend) || {}, avg = (x, y) => ((+b[x] || 0) + (+b[y] || 0)) / 2;
    const raw = {smile: avg('mouthSmileLeft', 'mouthSmileRight'), browDown: avg('browDownLeft', 'browDownRight'), innerUp: +b.browInnerUp || 0,
      outerUp: avg('browOuterUpLeft', 'browOuterUpRight')};
    if (t < 2 && f && f.face && f.face.present) { base = base || {smile: 0, browDown: 0, innerUp: 0, outerUp: 0}; for (const k in raw) base[k] += raw[k]; nb++; }
    const dev = base && nb ? Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v - base[k] / nb])) : {smile: 0, browDown: 0, innerUp: 0, outerUp: 0};
    let state;
    if (old) {
      const e = m.state.expr.m, c = {happy: e.smile, surprised: e.surprise ?? 0, frown: Math.max(e.anger ?? 0, e.browDown * 0.8), sad: e.sad, raised_brows: e.browUp * 0.8};
      let best = 'neutral', bv = 0.15; for (const k in c) if (c[k] > bv) { bv = c[k]; best = k; }
      state = best;
    } else state = tr.state.state;
    rows.push({t, state, o, dev});
  }
  for (const s of segs) {
    const inSeg = rows.filter(r => r.t >= s.t0 + 0.5 && r.t <= s.t1);
    const cnt = {}; for (const r of inSeg) cnt[r.state] = (cnt[r.state] || 0) + 1;
    const nonNeu = Object.entries(cnt).filter(([k]) => k !== 'neutral').sort((x, y) => y[1] - x[1]);
    const nn = nonNeu.reduce((q, [, v]) => q + v, 0);
    const pred = nn >= 0.25 * inSeg.length && nonNeu.length ? nonNeu[0][0] : 'neutral';
    const ok = ACCEPT[s.label].includes(pred);
    (res[s.label] = res[s.label] || []).push({ok, pred});
    if (s.label === 'neutral') continue;
    // response + latency
    const [fn, thr] = RESP[s.label], pre = rows.filter(r => r.t >= s.t0 - 0.4 && r.t < s.t0 + 0.1), seg = rows.filter(r => r.t >= s.t0 && r.t <= s.t1 + 0.6);
    const preV = pre.length ? pre.reduce((q, r) => q + fn(r.o), 0) / pre.length : fn(seg[0].o);
    const sig = seg.map(r => fn(r.o) - preV), pk = Math.max(...sig);
    const keyS = seg.map(r => KEY[s.label](r.dev)), kpk = Math.max(...keyS), k0 = keyS[0];
    const onIdx = keyS.findIndex(v => v - k0 >= 0.3 * (kpk - k0));
    const respIdx = pk >= thr ? sig.findIndex(v => v >= 0.5 * pk) : -1;
    (resp[s.label] = resp[s.label] || []).push(pk >= thr);
    if (respIdx >= 0 && onIdx >= 0 && kpk - k0 > 0.05) (lat[s.label] = lat[s.label] || []).push(Math.max(0, (respIdx - onIdx) * DT));
  }
}
const med = a => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
const table = {};
let okAll = 0, nAll = 0, rAll = 0, nR = 0; const latAll = [];
for (const lab of Object.keys(ACCEPT)) {
  const r = res[lab] || [], ok = r.filter(x => x.ok).length; okAll += ok; nAll += r.length;
  const rr = resp[lab] || [], ro = rr.filter(Boolean).length; rAll += ro; nR += rr.length; latAll.push(...(lat[lab] || []));
  table[lab] = {detect: `${ok}/${r.length}`, preds: r.map(x => x.pred).join(','), respond: rr.length ? `${ro}/${rr.length}` : '–', latency_med_s: med(lat[lab] || [])};
}
console.log((old ? 'OLD (mirror channels)' : 'NEW (tracker)') + ' — ' + arg('motion', 'app/motion.js'));
for (const [k, v] of Object.entries(table)) console.log(`  ${k.padEnd(9)} detect ${v.detect.padEnd(5)} respond ${String(v.respond).padEnd(5)} latency ${v.latency_med_s === null ? '–' : v.latency_med_s.toFixed(2) + ' s'}   [${v.preds}]`);
console.log(`  ALL detect ${okAll}/${nAll} (${(100 * okAll / nAll).toFixed(0)}%) · avatar responds ${rAll}/${nR} (${(100 * rAll / nR).toFixed(0)}%) · median latency ${med(latAll) === null ? '–' : med(latAll).toFixed(2) + ' s'} (n=${latAll.length})`);
fs.writeFileSync(path.join(D, 'eval_' + (old ? 'old' : 'new') + '.json'), JSON.stringify({table, okAll, nAll, rAll, nR, latAll}, null, 1));
