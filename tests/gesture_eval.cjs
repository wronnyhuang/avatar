#!/usr/bin/env node
/* gesture_eval.cjs — learn-from-a-few-examples evaluation of app/gestures.js (perception).
 *
 *   node tests/gesture_eval.cjs [--suite synth|nods|counter|dropout|all] [--quick] [--out FILE]
 *
 * synth   : scripted gestures (tests/gesture_synth.cjs) injected into REAL Seamless conversation streams of one person
 *           (his own normal motion, his own gesticulating hands): ~60–90 s of baseline, then 3 examples taught with
 *           learn() (per-example 'last' calls, or one 'next' call covering two repeats + one 'last'), then the rest of
 *           his clips with 8 held-out repeats (new place / angle / finger curl / speed / size) + 6 confusers
 *           (other gestures of the same body part). hit = an event for the gesture between the repeat's start and
 *           1 s after its end; false alarm = an event outside every repeat of that gesture (inside a confuser =
 *           confusion; otherwise = during his normal motion → per minute of normal motion).
 * nods    : REAL nods (offline labels on 34 YouTube talking/listening streams): learn from the stream's first 3
 *           labelled nods, detect the rest; FA = events farther than 0.5 s from any labelled nod / shake / ignore
 *           interval. Reference: the hand-built nod rule in perception.js on the same held-out span.
 * counter : 'peace' learned, confusion with 'three' / 'open_palm' before and after a counterexample ("not like this").
 * dropout : the synth suite on a subset with heavy MediaPipe dropouts (25 % random + bursts) and handedness flips.
 */
'use strict';
const fs = require('fs'), path = require('path');
const L = require('./perc_lib.cjs');
const SY = require('./gesture_synth.cjs');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const SUITE = arg('suite', 'all'), QUICK = process.argv.includes('--quick');
const OUT = arg('out', path.join(path.join(require('os').tmpdir(), 'avatar-tests'), 'gesture_eval.json'));
const GM = L.loadUmd('app/gestures.js');

const PEOPLE = {P0299A: 'sl_S0644_0578_P0299A,sl_S0644_0582_P0299A,sl_S0648_0371_P0299A,sl_S0648_0372_P0299A,sl_S0648_0373_P0299A,sl_S0648_0375_P0299A,sl_S0648_0377_P0299A,sl_S0648_0382_P0299A,sl_S0648_0502_P0299A,sl_S0648_0543_P0299A,sl_S0648_0545_P0299A',
  P1281A: 'sl_S2025_1079_P1281A,sl_S2025_1080_P1281A,sl_S2025_1081_P1281A,sl_S2025_1083_P1281A,sl_S2089_1165_P1281A',
  P1282A: 'sl_S2091_1095_P1282A,sl_S2091_1100_P1282A,sl_S2091_1102_P1282A,sl_S2091_1103_P1282A',
  P1346A: 'sl_S2090_1010_P1346A,sl_S2090_1011_P1346A,sl_S2090_1013_P1346A'};
const CONFUSE = {
  hand_static: ['peace', 'thumbs_up', 'ok', 'rock', 'point_up', 'fist', 'call_me', 'three', 'open_palm', 'finger_gun'],
  hand_moving: ['wave', 'beckon', 'circle', 'swipe', 'chop'],
  hands: ['raise_roof', 'clap', 'wave'],
  face_static: ['wink', 'kiss', 'puff', 'brows_hold'],
  face_moving: ['head_tilt_rock', 'brow_flash', 'kiss'],
};
function family(g) { const G = SY.GESTURES[g]; return G.part === 'face' ? 'face_' + G.kind : G.part === 'hands' ? 'hands' : 'hand_' + G.kind; }

const streamCache = {};
function personStream(p) {
  if (streamCache[p]) return streamCache[p];
  const out = []; let off = 0;
  for (const id of PEOPLE[p].split(',')) {
    const fr = L.loadStream(id); if (!fr.length) continue;
    const t0 = fr[0].t;
    for (const f of fr) out.push(Object.assign({}, f, {t: Math.round((f.t - t0 + off) * 1e4) / 1e4, events: []}));
    off = out[out.length - 1].t + 0.5;
  }
  return (streamCache[p] = out);
}

// run one gesture on one person: returns metrics
function runOne(person, g, o) {
  o = o || {};
  const R = L.rng(o.seed || (hash(person + g) + 1));
  const base = personStream(person);
  const T = base[base.length - 1].t;
  const B = o.baseline_s || (T > 300 ? 90 : 60);
  const hand = R() < 0.25 ? 'left' : 'right';
  const insts = [], sched = [];
  // 3 teaching examples
  const style = o.style || (R() < 0.5 ? 'each' : 'next');
  let t = B + 1.5;
  const ex = [];
  for (let k = 0; k < 3; k++) {
    const inst = {g, t0: t, hold: 0.9 + R() * 0.6, hand, var: SY.variation(R, 0.5), role: 'example'};
    inst.cycles = SY.GESTURES[g].kind === 'moving' && SY.GESTURES[g].motion !== 'swipe' && SY.GESTURES[g].motion !== 'circle' ? 2 + Math.floor(R() * 2) : undefined;
    const d = SY.instDur(inst); ex.push(inst); insts.push(inst); t += d + 2.5 + R() * 1.5;
  }
  if (style === 'each') for (const e of ex) { const d = SY.instDur(e); sched.push({t: e.t0 + d + 0.8, act: 'last', seconds: d + 1.8}); }
  else { const d1 = SY.instDur(ex[0]); sched.push({t: ex[0].t0 - 0.4, act: 'next', seconds: ex[1].t0 + SY.instDur(ex[1]) - ex[0].t0 + 0.8}); const d3 = SY.instDur(ex[2]); sched.push({t: ex[2].t0 + d3 + 0.8, act: 'last', seconds: d3 + 1.8}); }
  const testStart = t + 3;
  // held-out repeats + confusers, ≥ 6 s apart, in the rest of his stream
  const conf = CONFUSE[family(g)].filter(x => x !== g);
  const slots = [];
  const nRep = o.repeats || 8, nConf = o.confusers == null ? 6 : o.confusers;
  const span = T - testStart - 4;
  const need = nRep + nConf, gap = span / need;
  const order = [];
  for (let k = 0; k < nRep; k++) order.push('rep'); for (let k = 0; k < nConf; k++) order.push('conf');
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  order.forEach((kind, i) => {
    const t0 = testStart + i * gap + R() * Math.max(0.1, gap - 4);
    const gg = kind === 'rep' ? g : conf[Math.floor(R() * conf.length)];
    const inst = {g: gg, t0, hold: 0.7 + R() * 1.0, hand: kind === 'rep' ? (R() < 0.85 ? hand : (hand === 'right' ? 'left' : 'right')) : hand, var: SY.variation(R, 1), role: kind};
    if (SY.GESTURES[gg].kind === 'moving' && SY.GESTURES[gg].motion !== 'swipe' && SY.GESTURES[gg].motion !== 'circle') inst.cycles = 2 + Math.floor(R() * 2);
    inst.otherHand = kind === 'rep' && inst.hand !== hand;
    insts.push(inst);
  });
  const frames = SY.inject(base, insts, {seed: (o.seed || 1) * 31 + 7, dropout: o.dropout, burst: o.burst, flip: o.flip});
  // feed
  const det = GM.create({faTarget: 0.25});
  const events = [], learnRes = [], pend = [];
  let si = 0; const ms = [];
  for (const f of frames) {
    while (si < sched.length && f.t >= sched[si].t) {
      const s = sched[si++];
      const r = det.learn(g, {from: s.act, seconds: s.seconds});
      if (r && typeof r.then === 'function') pend.push(r.then(x => learnRes.push(x))); else learnRes.push(r);
    }
    const t0 = process.hrtime.bigint();
    const ev = det.frame(f);
    ms.push(Number(process.hrtime.bigint() - t0) / 1e6);
    for (const e of ev) if (e.type === 'gesture') events.push(e);
  }
  // score on the test span only
  const testEv = events.filter(e => e.t >= testStart);
  const reps = insts.filter(i => i.role === 'rep'), confs = insts.filter(i => i.role === 'conf');
  const within = (e, i, pre, post) => e.t >= i.t0 - (pre || 0) && e.t <= i.t1 + (post || 1.0);
  let hits = 0, hitsOther = 0, nOther = 0; const lat = [];
  for (const r of reps) {
    const e = testEv.find(e => e.name === g && within(e, r));
    if (r.otherHand) { nOther++; if (e) hitsOther++; } else if (e) { hits++; lat.push(e.t - r.t0); }
  }
  let confusion = 0, fa = 0;
  for (const e of testEv) {
    if (e.name !== g) continue;
    if (reps.some(r => within(e, r, 0.3, 1.5))) continue;
    if (confs.some(c => within(e, c, 0.3, 1.0))) confusion++; else fa++;
  }
  const busy = insts.filter(i => i.t0 >= testStart).reduce((s, i) => s + i.dur + 1.3, 0);
  const normalMin = Math.max(0.1, (T - testStart - busy) / 60);
  const model = det.export(g) || {};
  const sm = ms.slice().sort((a, b) => a - b);
  const dbg = {style, examples: insts.filter(i => i.role === 'example').map(i => [L.r3(i.t0), L.r3(i.t1)]), sched, segs: (model.examples || []).map(e => [e.t, e.t1, e.vote, e.groups, e.why]), groups: model.groups,
    dims: model.dims, w: model.weights, thr: model.thr, thr_pos: model.thr_pos, thr_fa: model.thr_fa, loo: model.loo, in_sample: model.in_sample,
    test: reps.map(r => [L.r3(r.t0), L.r3(r.t1), r.otherHand ? 'other' : '']), ev: testEv.filter(e => e.name === g).map(e => [e.t, e.score])};
  return {person, g, kind: SY.GESTURES[g].kind, family: family(g), style, hand, learned: learnRes.map(r => r && {ok: r.ok, kind: r.kind, examples: r.examples, quality: r.quality, segs: r.segments && r.segments.length, groups: r.groups}),
    model_kind: model.kind, thr: model.thr, fa_est: model.fa_per_min, quality: model.quality,
    reps: reps.length - nOther, hits, other_hand: {n: nOther, hits: hitsOther}, confusers: confs.length, confusion, fa, normal_min: L.r3(normalMin), fa_per_min: L.r3(fa / normalMin),
    dbg, latency_s: lat.length ? L.r3(L.pct(lat, 0.5)) : null, ms_p50: L.r3(L.pct(sm, 0.5)), ms_p95: L.r3(L.pct(sm, 0.95))};
}
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

function summarize(rows, label) {
  const by = {};
  for (const r of rows) { const k = r.family; (by[k] = by[k] || []).push(r); }
  const agg = rs => {
    const reps = rs.reduce((s, r) => s + r.reps, 0), hits = rs.reduce((s, r) => s + r.hits, 0), mins = rs.reduce((s, r) => s + r.normal_min, 0);
    const fa = rs.reduce((s, r) => s + r.fa, 0), conf = rs.reduce((s, r) => s + r.confusion, 0), nconf = rs.reduce((s, r) => s + r.confusers, 0);
    const oh = rs.reduce((s, r) => s + r.other_hand.n, 0), ohh = rs.reduce((s, r) => s + r.other_hand.hits, 0);
    return {runs: rs.length, hit_rate: L.r3(hits / Math.max(1, reps)), hits, reps, fa_per_min: L.r3(fa / Math.max(0.01, mins)), fa, normal_min: L.r3(mins),
      confusion_rate: L.r3(conf / Math.max(1, nconf)), other_hand_hits: `${ohh}/${oh}`, kind_ok: rs.filter(r => r.model_kind === r.kind).length + '/' + rs.length,
      latency_s_p50: L.r3(L.pct(rs.map(r => r.latency_s).filter(x => x != null), 0.5)), ms_p95: L.r3(Math.max(...rs.map(r => r.ms_p95)))};
  };
  const out = {label, all: agg(rows)};
  for (const k in by) out[k] = agg(by[k]);
  return out;
}

function suiteSynth(o) {
  o = o || {};
  const persons = QUICK ? ['P0299A'] : (o.persons || (arg('persons') ? arg('persons').split(',') : null) || ['P0299A', 'P1281A', 'P1282A']);
  const gs = o.gestures || (arg('only') ? arg('only').split(',') : null) || (QUICK ? ['peace', 'thumbs_up', 'wave', 'beckon', 'wink', 'head_tilt_rock'] : Object.keys(SY.GESTURES).filter(g => g !== 'open_palm'));
  const rows = [];
  // face gestures need a face: P1282A's clips are mostly faceless (face found in 0-31 % of frames) → P1346A instead
  const FACE_OK = {P0299A: 1, P1281A: 1, P1346A: 1};
  for (const p0 of persons) for (const g of gs) {
    const p = SY.GESTURES[g].part === 'face' && !FACE_OK[p0] ? 'P1346A' : p0;
    const r = runOne(p, g, o);
    rows.push(r);
    if (process.argv.includes('--verbose')) console.error(JSON.stringify({learned: r.learned, dbg: r.dbg}, null, 0));
    console.error(`${o.tag || 'synth'} ${p} ${g.padEnd(15)} ${r.model_kind}/${r.kind} q=${r.quality} hits ${r.hits}/${r.reps} other ${r.other_hand.hits}/${r.other_hand.n} conf ${r.confusion}/${r.confusers} fa ${r.fa} (${r.fa_per_min}/min) thr ${r.thr} ms95 ${r.ms_p95}`);
  }
  return {rows, summary: summarize(rows, o.tag || 'synth')};
}

function suiteNods() {
  const streams = L.nodshakeStreams(8).slice(0, QUICK ? 6 : 40);
  const PERC = L.loadUmd('app/perception.js');
  const rows = [];
  for (const s of streams) {
    const frames = L.loadNodshakeFrames(s.file);
    const nods = s.lab.nods.slice().sort((a, b) => a.onset - b.onset);
    const learnN = 3, teach = nods.slice(0, learnN), held = nods.slice(learnN);
    if (held.length < 4) continue;
    const testStart = teach[teach.length - 1].end + 1.5;
    const det = GM.create({faTarget: 0.25});
    const sched = teach.map(n => ({t: n.end + 0.6, seconds: n.end - n.onset + 1.1}));
    let si = 0; const evs = [];
    const xt = PERC.createExpressionTracker(); const ruleEv = [];
    for (const f of frames) {
      while (si < sched.length && f.t >= sched[si].t) { const q = sched[si++]; det.learn('nod', {from: 'last', seconds: q.seconds}); }
      for (const e of det.frame(f)) if (e.type === 'gesture') evs.push(e);
      const xr = xt.update(f.face && f.face.present ? f.face : {present: false}, f.t);
      for (const e of xr.events) if (e.type === 'nod') ruleEv.push(e);
    }
    const ign = [].concat(s.lab.shakes || [], (s.lab.ignore && s.lab.ignore.nod) || [], (s.lab.ignore && s.lab.ignore.head_shake) || []);
    const score = list => {
      const te = list.filter(e => e.t >= testStart);
      let hits = 0; const used = new Set();
      for (const n of held) { const e = te.find(e => e.t >= n.onset - 0.2 && e.t <= n.end + 0.6); if (e) { hits++; used.add(e); } }
      const fa = te.filter(e => !used.has(e) && !nods.some(n => e.t >= n.onset - 0.5 && e.t <= n.end + 0.8) && !ign.some(n => e.t >= n.onset - 0.5 && e.t <= n.end + 0.8)).length;
      return {hits, fa};
    };
    const T = frames[frames.length - 1].t, mins = Math.max(0.1, (T - testStart) / 60);
    const a = score(evs), b = score(ruleEv);
    const m = det.export('nod') || {};
    rows.push({stream: s.name, held: held.length, minutes: L.r3(mins), learned: {kind: m.kind, thr: m.thr, quality: m.quality, dims: m.dims},
      learned_hits: a.hits, learned_fa: a.fa, rule_hits: b.hits, rule_fa: b.fa});
    console.error(`nods ${s.name.padEnd(26)} held ${held.length} learned ${a.hits} fa ${a.fa} | rule ${b.hits} fa ${b.fa} | ${m.kind} ${m.quality} dims ${(m.dims || []).join(',')}`);
  }
  const sum = k => rows.reduce((s, r) => s + r[k], 0), mins = rows.reduce((s, r) => s + r.minutes, 0);
  return {rows, summary: {streams: rows.length, held: sum('held'), minutes: L.r3(mins),
    learned: {hit_rate: L.r3(sum('learned_hits') / Math.max(1, sum('held'))), fa_per_min: L.r3(sum('learned_fa') / mins)},
    rule_perception_js: {hit_rate: L.r3(sum('rule_hits') / Math.max(1, sum('held'))), fa_per_min: L.r3(sum('rule_fa') / mins)}}};
}

function suiteCounter() {
  // learn peace (3 examples), then: confusion with 'three' before / after one counterexample of 'three'
  const person = 'P0299A', base = personStream(person), R = L.rng(99);
  const B = 90, insts = [], sched = [];
  let t = B + 1.5;
  for (let k = 0; k < 3; k++) { const i = {g: 'peace', t0: t, hold: 1.1, hand: 'right', var: SY.variation(R, 0.5), role: 'example'}; insts.push(i); const d = SY.instDur(i); sched.push({t: t + d + 0.8, opt: {from: 'last', seconds: d + 1.8}}); t += d + 3; }
  // phase 1 test: 4 peace + 4 three
  const phase = (t0, n) => { const out = []; for (let k = 0; k < n; k++) { out.push({g: k % 2 ? 'three' : 'peace', t0: t0 + k * 7, hold: 1.0, hand: 'right', var: SY.variation(R, 1), role: k % 2 ? 'conf' : 'rep'}); } return out; };
  const p1 = phase(t + 2, 8); insts.push(...p1); t = p1[p1.length - 1].t0 + 7;
  // the counterexample: one 'three', taught with counter:true
  const c = {g: 'three', t0: t, hold: 1.1, hand: 'right', var: SY.variation(R, 0.5), role: 'counter'}; insts.push(c); const dc = SY.instDur(c);
  sched.push({t: t + dc + 0.8, opt: {from: 'last', seconds: dc + 1.8, counter: true}}); t += dc + 3;
  const p2 = phase(t + 2, 12); insts.push(...p2);
  const frames = SY.inject(base, insts, {seed: 5});
  const det = GM.create({}); const ev = []; let si = 0; const res = [];
  for (const f of frames) {
    while (si < sched.length && f.t >= sched[si].t) { const s = sched[si++]; res.push(det.learn('peace', s.opt)); }
    for (const e of det.frame(f)) if (e.type === 'gesture') ev.push(e);
  }
  const score = list => ({peace_hits: list.filter(i => i.g === 'peace').filter(i => ev.some(e => e.t >= i.t0 && e.t <= i.t1 + 1)).length + '/' + list.filter(i => i.g === 'peace').length,
    three_confusions: list.filter(i => i.g === 'three').filter(i => ev.some(e => e.t >= i.t0 && e.t <= i.t1 + 1)).length + '/' + list.filter(i => i.g === 'three').length});
  const m = det.export('peace') || {};
  return {before: score(p1), after: score(p2), learn: res.map(r => ({ok: r.ok, examples: r.examples, counters: r.counters, quality: r.quality, thr: r.threshold, why: r.why})),
    model: {thr: m.thr, thr_pos: m.thr_pos, thr_fa: m.thr_fa, loo: m.loo, in_sample: m.in_sample, groups: m.groups, counter_llr: m.counter_llr, cseg: (m.counters || []).map(c => [c.t, c.t1, c.vote])},
    events: ev.map(e => [e.t, e.score]), p2: p2.map(i => [i.g, +i.t0.toFixed(2), +i.t1.toFixed(2)]), cinst: [c.t0, c.t1]};
}

// invariance: a clean hands-only stream at 30 fps (no face / pose), ONE taught example ('last'), then the gesture used
// at 0.8–1.3× the size, moved up to ±0.3 of the frame and tilted up to ±15°, plus near-miss shapes; FA on the idle hand
function suiteInvariance() {
  const statics = ['peace', 'thumbs_up', 'ok', 'rock', 'point_up', 'call_me', 'three', 'fist', 'finger_gun'];
  const moving = ['wave', 'circle', 'beckon'];
  const NEAR = {peace: ['three', 'point_up'], thumbs_up: ['fist', 'finger_gun'], ok: ['open_palm', 'three'], rock: ['point_up', 'call_me'], point_up: ['peace', 'fist'],
    call_me: ['thumbs_up', 'rock'], three: ['peace', 'open_palm'], fist: ['thumbs_up', 'point_up'], finger_gun: ['thumbs_up', 'point_up'], wave: ['swipe', 'open_palm'], circle: ['point_up', 'swipe'], beckon: ['point_up', 'wave']};
  const rows = [];
  for (const g of (arg('only') ? arg('only').split(',') : statics.concat(moving))) {
    const R = L.rng(hash('inv' + g));
    const base = SY.idleHandStream(200, 30, hash(g) % 1000 + 1);
    const insts = [], DS = 0.6;   // the synth's body unit without a body = 0.6 x-units = 0.3 frame widths
    const ex = {g, t0: 61, hold: 1.2, hand: 'right', cycles: SY.GESTURES[g].kind === 'moving' && !/circle|swipe/.test(SY.GESTURES[g].motion) ? 3 : undefined, var: {}, role: 'example'};
    insts.push(ex);
    const testStart = 61 + SY.instDur(ex) + 4;
    for (let k = 0; k < 8; k++) {
      const v = {size: 0.8 + 0.5 * R(), dx: (R() * 2 - 1) * 0.3 * 2 / DS, dy: (R() * 2 - 1) * 0.3 * 2 * 0.75 / DS, roll: (R() * 2 - 1) * 15};
      insts.push({g, t0: testStart + k * 13, hold: 0.8 + R() * 0.8, hand: 'right', cycles: ex.cycles ? 2 + Math.floor(R() * 2) : undefined, var: v, role: 'rep'});
    }
    NEAR[g].forEach((c, k) => insts.push({g: c, t0: testStart + 8 * 13 + k * 13, hold: 1.0, hand: 'right', cycles: SY.GESTURES[c].kind === 'moving' ? 2 : undefined, var: {roll: (R() * 2 - 1) * 10}, role: 'conf'}));
    // the injection's rest pose is below the frame; the idle hand is replaced while a gesture is up
    const frames = SY.inject(base, insts, {seed: 3, dropout: 0, burst: 0});
    const det = GM.create({}); const ev = []; let learned = null;
    const tl = ex.t0 + SY.instDur(ex) + 0.8;
    for (const f of frames) { if (!learned && f.t >= tl) learned = det.learn(g, {from: 'last', seconds: 4}); for (const e of det.frame(f)) if (e.type === 'gesture') ev.push(e); }
    const reps = insts.filter(i => i.role === 'rep'), confs = insts.filter(i => i.role === 'conf');
    const hit = reps.filter(r => ev.some(e => e.t >= r.t0 && e.t <= r.t1 + 1)).length;
    const near = confs.filter(c => ev.some(e => e.t >= c.t0 && e.t <= c.t1 + 1)).length;
    const busy = insts.filter(i => i.t0 >= testStart);
    const fa = ev.filter(e => e.t >= testStart && !busy.some(i => e.t >= i.t0 - 0.3 && e.t <= i.t1 + 1.5)).length;
    const idleMin = (200 - testStart - busy.reduce((s, i) => s + i.dur + 1.8, 0)) / 60;
    const r = {g, kind: SY.GESTURES[g].kind, learned_kind: learned && learned.kind, quality: learned && learned.quality, hits: hit, reps: reps.length, near, nears: confs.length, fa, idle_min: L.r3(idleMin), fa_per_min: L.r3(fa / idleMin)};
    rows.push(r);
    if (process.argv.includes('--verbose')) { const m = det.export(g) || {}; console.error(JSON.stringify({learned, thr: m.thr, pos: m.thr_pos, fa: m.thr_fa, loo: m.loo, groups: m.groups, ex: (m.examples || []).map(e => [e.t, e.t1, e.vote, e.why]), ev: ev.map(e => [e.t, e.score]), reps: reps.map(q => [L.r3(q.t0), L.r3(q.t1), q.var])})); }
    console.error(`inv ${g.padEnd(11)} ${r.learned_kind}/${r.kind} q=${r.quality} hits ${hit}/${reps.length} near ${near}/${confs.length} fa ${fa} (${r.fa_per_min}/min)`);
  }
  const sum = k => rows.reduce((s, r) => s + r[k], 0);
  return {rows, summary: {gestures: rows.length, kind_ok: rows.filter(r => r.learned_kind === r.kind).length, hit_rate: L.r3(sum('hits') / sum('reps')),
    near_miss_rate: L.r3(sum('near') / sum('nears')), fa_per_min: L.r3(sum('fa') / Math.max(0.01, sum('idle_min'))), fa: sum('fa'), idle_min: L.r3(sum('idle_min'))}};
}

const out = {when: new Date().toISOString(), gestures_sha: require('crypto').createHash('sha1').update(fs.readFileSync(path.join(L.ROOT, 'app/gestures.js'))).digest('hex').slice(0, 12)};
if (SUITE === 'synth' || SUITE === 'all') { const r = suiteSynth({}); out.synth = r.summary; out.synth_rows = r.rows; console.error(JSON.stringify(r.summary)); }
if (SUITE === 'nods' || SUITE === 'all') { const r = suiteNods(); out.nods = r.summary; out.nod_rows = r.rows; console.error(JSON.stringify(r.summary)); }
if (SUITE === 'invariance' || SUITE === 'all') { const r = suiteInvariance(); out.invariance = r.summary; out.invariance_rows = r.rows; console.error(JSON.stringify(r.summary)); }
if (SUITE === 'counter' || SUITE === 'all') { out.counter = suiteCounter(); console.error(JSON.stringify(out.counter)); }
if (SUITE === 'dropout' || SUITE === 'all') {
  const r = suiteSynth({tag: 'dropout', dropout: 0.25, burst: 0.06, flip: 0.05, persons: ['P0299A'], gestures: ['peace', 'thumbs_up', 'call_me', 'wave', 'beckon', 'circle', 'raise_roof']});
  out.dropout = r.summary; out.dropout_rows = r.rows; console.error(JSON.stringify(r.summary));
}
fs.mkdirSync(path.dirname(OUT), {recursive: true});
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify({synth: out.synth && out.synth.all, nods: out.nods, counter: out.counter, dropout: out.dropout && out.dropout.all}, null, 1));
