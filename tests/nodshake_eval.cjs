#!/usr/bin/env node
/* nodshake_eval.cjs — offline eval of app/perception.js createExpressionTracker on real conversation clips.
 *
 *   node tests/nodshake_eval.cjs [--clips N] [--fps F] [--opts '{"big":0.6}'] [--dir D] [--out F] [--per-stream]
 *
 * Data: <dir>/index.json + frames/*.jsonl + labels/*.json written by tests/npz_to_frames.py (30 fps perception-style
 * face frames from 133 YouTube clips; offline non-causal nod / head-shake labels from the raw head angles).
 * --clips N  first N clips (sorted by name) of the index (default 40; 0 = all — the index holds 63 after --all)
 * --fps F    feed the tracker at ~F Hz (keeps a frame when floor(t·F) advances; the app's face rate is 18) — default 30
 * --opts     JSON passed to createExpressionTracker(opts)
 * --dets     also write <out dir>/dets.json: every detection (hit / fp / dup / ign), miss and expression_change per stream
 * --per-stream  print one line per stream
 * Each stream gets a fresh tracker, fed every frame causally: tr.update(face, t) with face = the frame's face object
 * ({present:false} while absent). 'nod' / 'head_shake' events are matched one-to-one to labels of the same type: a
 * detection hits a label when onset − 0.2 s ≤ t ≤ onset + 1.5 s (labels in onset order take the first free detection).
 * Unmatched detections are DUP (inside a matched label's span, onset − 0.2 .. end + 0.5 — a multi-swing nod reported
 * twice), IGN (inside a label file's ignore interval: ambiguous / oversized swings) or FP (everything else; `conf` =
 * FPs that hit a label of the OTHER type). precision = TP/(TP+FP), recall = TP/labels; latency = t − onset (hits).
 * Rates are per minute of face-present time. 'frontal' = labels with |yaw| ≤ 0.6 throughout, detections whose frame
 * has |raw yaw| ≤ 0.6. expression_change: count, per-minute rate (> ~2/min = too trigger-happy), `to` / from→to
 * distribution, intensity, shortest gap between two events of one stream (the ≥ 3 s debounce).
 * Writes <dir>/results.json (or --out) and prints a compact table. */
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const ROOT = path.resolve(__dirname, '..');
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const flag = k => argv.includes('--' + k);
const DIR = arg('dir', path.join(process.env.AVATAR_DATA || path.join(ROOT, 'data'), 'nodshake'));
const NCLIPS = +arg('clips', 40);   // 0 = every clip in the index
const FEED_FPS = +arg('fps', 30);
const OPTS = JSON.parse(arg('opts', '{}'));
const OUTF = arg('out', path.join(DIR, 'results.json'));
const PERC = path.resolve(arg('perception', path.join(ROOT, 'app', 'perception.js')));
const WIN = [-0.2, 1.5], DUP_TAIL = 0.5, FRONTAL = 0.6, TYPES = ['nod', 'head_shake'];
const t0 = Date.now();

// ── tracker under test (same trick as tests/load.cjs) ──
const src = fs.readFileSync(PERC, 'utf8');
const mod = {exports: {}};
new Function('module', 'exports', src)(mod, mod.exports);
const P = mod.exports && mod.exports.createExpressionTracker ? mod.exports : (globalThis.AvatarPerception || {});
const percInfo = {path: PERC, mtime: fs.statSync(PERC).mtime.toISOString(), mtime_local: fmtLocal(fs.statSync(PERC).mtime),
  sha1: crypto.createHash('sha1').update(src).digest('hex').slice(0, 12)};
if (typeof P.createExpressionTracker !== 'function') {
  console.error(`createExpressionTracker not exported by ${PERC} (mtime ${percInfo.mtime_local})`); process.exit(2);
}

// ── data ──
const index = JSON.parse(fs.readFileSync(path.join(DIR, 'index.json'), 'utf8'));
let streams = index.streams.slice().sort((a, b) => a.clip < b.clip ? -1 : a.clip > b.clip ? 1 : a.k - b.k);
const clipNames = [...new Set(streams.map(s => s.clip))];
const useClips = new Set(NCLIPS > 0 ? clipNames.slice(0, NCLIPS) : clipNames);
streams = streams.filter(s => useClips.has(s.clip));

const pct = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const median = a => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const r3 = v => v == null || !isFinite(v) ? null : Math.round(v * 1000) / 1000;
function fmtLocal(d) { const p = n => String(n).padStart(2, '0'); return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; }

function match(labels, ign, other, dets, frontAt) {
  const L = labels.slice().sort((a, b) => a.onset - b.onset), D = dets.slice().sort((a, b) => a.t - b.t);
  const used = new Array(D.length).fill(false), hits = [], miss = [];
  for (const l of L) {
    let j = -1;
    for (let i = 0; i < D.length; i++) { if (used[i]) continue; const dt = D[i].t - l.onset; if (dt < WIN[0]) continue; if (dt > WIN[1]) break; j = i; break; }
    if (j >= 0) { used[j] = true; hits.push({label: l, det: D[j], lat: D[j].t - l.onset}); } else miss.push(l);
  }
  const inSpan = (t, l) => t >= l.onset + WIN[0] && t <= Math.max(l.onset + WIN[1], (l.end != null ? l.end : l.onset) + DUP_TAIL);
  const fp = [], dup = [], ig = [];
  for (let i = 0; i < D.length; i++) {
    if (used[i]) continue;
    const t = D[i].t;
    if (hits.some(h => inSpan(t, h.label))) dup.push(D[i]);
    else if (ign.some(l => inSpan(t, l))) ig.push(D[i]);
    else fp.push(Object.assign({conf: other.some(l => t >= l.onset + WIN[0] && t <= l.onset + WIN[1]), frontal: frontAt(t)}, D[i]));
  }
  return {hits, miss, fp, dup, ig};
}

function summarize(acc, minutes) {
  const tp = acc.hits.length, fp = acc.fp.length, fn = acc.miss.length, nl = tp + fn, nd = tp + fp + acc.dup.length + acc.ig.length;
  const P_ = tp + fp ? tp / (tp + fp) : null, R_ = nl ? tp / nl : null;
  const F1 = P_ != null && R_ != null && P_ + R_ > 0 ? 2 * P_ * R_ / (P_ + R_) : (nl || tp + fp ? 0 : null);
  const lat = acc.hits.map(h => h.lat);
  return {labels: nl, detections: nd, tp, fp, fn, dup: acc.dup.length, ignored: acc.ig.length, confused: acc.fp.filter(x => x.conf).length,
    precision: r3(P_), recall: r3(R_), f1: r3(F1), latency_median: r3(median(lat)), latency_p90: r3(pct(lat, 0.9)),
    fp_per_min: r3(fp / minutes), det_per_min: r3(nd / minutes), labels_per_min: r3(nl / minutes)};
}
const newAcc = () => ({hits: [], miss: [], fp: [], dup: [], ig: []});
const addAcc = (A, m) => { for (const k of ['hits', 'miss', 'fp', 'dup', 'ig']) A[k].push(...m[k]); };

const tot = {min: 0, nod: newAcc(), head_shake: newAcc(), fr: {nod: newAcc(), head_shake: newAcc()}, frMin: 0};
const byCat = {};
const expr = [], otherEv = {}, stateT = {}, perStream = [], errors = [];
const examples = {nod: {fp: [], miss: []}, head_shake: {fp: [], miss: []}};   // first 40 of each, for tuning
const detDump = {};   // --dets: every detection / label with its verdict, per stream → <dir>/dets.json
let framesFed = 0;

for (const s of streams) {
  const lab = JSON.parse(fs.readFileSync(path.join(DIR, s.labels), 'utf8'));
  const lines = fs.readFileSync(path.join(DIR, s.frames), 'utf8').split('\n');
  const tr = P.createExpressionTracker(OPTS);
  const dets = {nod: [], head_shake: []}, ex = [], yawAt = [];
  let presentN = 0, frontN = 0, bucket = -1, err = null, lastState = null;
  const dtFrame = 1 / (index.fps || 30);
  for (const line of lines) {
    if (!line) continue;
    const fr = JSON.parse(line), t = fr.t, face = fr.face;
    if (face.present) { presentN++; if (Math.abs(face.raw ? face.raw[0] : face.yaw) <= FRONTAL) frontN++; }
    yawAt.push(face.present ? Math.abs(face.raw ? face.raw[0] : face.yaw) : null);
    if (FEED_FPS < 30) { const b = Math.floor(t * FEED_FPS + 1e-6); if (b <= bucket) continue; bucket = b; }
    let r;
    try { r = tr.update(face, t); } catch (e) { err = String(e && e.stack || e).split('\n').slice(0, 3).join(' | '); break; }
    framesFed++;
    if (r && face.present && r.state != null) { stateT[r.state] = (stateT[r.state] || 0) + 1; lastState = r.state; }
    const evs = r && Array.isArray(r.events) ? r.events : [];
    for (const ev of evs) {
      const et = typeof ev.t === 'number' ? ev.t : t;
      if (ev.type === 'nod' || ev.type === 'head_shake') dets[ev.type].push(Object.assign({}, ev, {t: et}));
      else if (ev.type === 'expression_change') ex.push(Object.assign({}, ev, {t: et}));
      else otherEv[ev.type] = (otherEv[ev.type] || 0) + 1;
    }
  }
  if (err) errors.push({clip: s.clip, k: s.k, error: err});
  const minutes = presentN * dtFrame / 60, frMin = frontN * dtFrame / 60;
  const frontAt = tt => { const i = Math.min(yawAt.length - 1, Math.max(0, Math.round(tt / dtFrame))); return yawAt[i] != null && yawAt[i] <= FRONTAL; };
  const ign = (lab.ignore || {});
  const lt = {nod: lab.nods || [], head_shake: lab.shakes || []};
  const cat = s.category || '?';
  const C = byCat[cat] || (byCat[cat] = {min: 0, streams: 0, nod: newAcc(), head_shake: newAcc(), expr: 0});
  C.min += minutes; C.streams++; C.expr += ex.length;
  tot.min += minutes; tot.frMin += frMin;
  const ps = {clip: s.clip, k: s.k, category: cat, present_min: r3(minutes), expression_change: ex.length};
  for (const ty of TYPES) {
    const m = match(lt[ty], ign[ty] || [], lt[ty === 'nod' ? 'head_shake' : 'nod'], dets[ty], frontAt);
    addAcc(tot[ty], m); addAcc(C[ty], m);
    addAcc(tot.fr[ty], {hits: m.hits.filter(h => h.label.frontal), miss: m.miss.filter(l => l.frontal), fp: m.fp.filter(x => x.frontal), dup: [], ig: []});
    ps[ty] = {labels: lt[ty].length, det: dets[ty].length, tp: m.hits.length, fp: m.fp.length, fn: m.miss.length};
    const X = examples[ty], id = `${s.clip}_s${s.k}`;
    if (flag('dets')) {
      const D = detDump[id] || (detDump[id] = {});
      D[ty] = {hit: m.hits.map(h => ({t: h.det.t, onset: h.label.onset, lat: r3(h.lat), amp: h.det.amp})), fp: m.fp.map(d => ({t: d.t, amp: d.amp, conf: d.conf})),
        dup: m.dup.map(d => d.t), ign: m.ig.map(d => d.t), miss: m.miss.map(l => ({onset: l.onset, amp_deg: l.amp_deg}))};
      D.expression_change = ex.map(e => ({t: e.t, from: e.from, to: e.to, intensity: e.intensity}));
    }
    for (const d of m.fp) if (X.fp.length < 40) X.fp.push({stream: id, t: d.t, amp: d.amp, n: d.n, conf: d.conf});
    for (const l of m.miss) if (X.miss.length < 40) X.miss.push({stream: id, onset: l.onset, amp_deg: l.amp_deg, n: l.n_swings || l.reversals});
  }
  for (const e of ex) expr.push(Object.assign({clip: s.clip, k: s.k, min: minutes}, e));
  ps.expr_min_gap = ex.length > 1 ? r3(Math.min(...ex.slice(1).map((e, i) => e.t - ex[i].t))) : null;
  ps.expr_per_min = r3(ex.length / Math.max(1e-9, minutes));
  perStream.push(ps);
}

// ── expression_change summary ──
const toDist = {}, pairDist = {};
for (const e of expr) { toDist[e.to] = (toDist[e.to] || 0) + 1; const k = `${e.from}>${e.to}`; pairDist[k] = (pairDist[k] || 0) + 1; }
const sortObj = o => Object.fromEntries(Object.entries(o).sort((a, b) => b[1] - a[1]));
const rates = perStream.map(p => p.expr_per_min).filter(v => v != null);
const gaps = perStream.map(p => p.expr_min_gap).filter(v => v != null);
const inten = expr.map(e => e.intensity).filter(v => typeof v === 'number');
const stTot = Object.values(stateT).reduce((a, b) => a + b, 0) || 1;
const exprSum = {count: expr.length, per_min: r3(expr.length / tot.min), to: sortObj(toDist), from_to: sortObj(pairDist),
  to_frac: Object.fromEntries(Object.entries(sortObj(toDist)).map(([k, v]) => [k, r3(v / Math.max(1, expr.length))])),
  stream_rate_median: r3(median(rates)), stream_rate_p90: r3(pct(rates, 0.9)), stream_rate_max: r3(rates.length ? Math.max(...rates) : null),
  streams_over_2_per_min: rates.filter(v => v > 2).length, min_gap_s: r3(gaps.length ? Math.min(...gaps) : null),
  intensity_median: r3(median(inten)), intensity_p10: r3(pct(inten, 0.1)),
  state_time_frac: sortObj(Object.fromEntries(Object.entries(stateT).map(([k, v]) => [k, r3(v / stTot)])))};

const results = {generated: new Date().toISOString(), generated_local: fmtLocal(new Date()), perception: percInfo,
  args: {clips: NCLIPS || null, fps: FEED_FPS, opts: OPTS, window_s: WIN, dup_tail_s: DUP_TAIL},
  data: {dir: DIR, index_generated: index.generated, clips: useClips.size, streams: streams.length, present_min: r3(tot.min),
    frontal_min: r3(tot.frMin), frames_fed: framesFed, label_params: index.label_params},
  nod: summarize(tot.nod, tot.min), head_shake: summarize(tot.head_shake, tot.min),
  frontal: {nod: summarize(tot.fr.nod, tot.frMin), head_shake: summarize(tot.fr.head_shake, tot.frMin)},
  expression_change: exprSum, other_events: otherEv,
  by_category: Object.fromEntries(Object.entries(byCat).sort().map(([k, C]) => [k, {streams: C.streams, present_min: r3(C.min),
    nod: summarize(C.nod, C.min), head_shake: summarize(C.head_shake, C.min), expression_change_per_min: r3(C.expr / C.min)}])),
  errors, runtime_s: null, examples, per_stream: perStream};
results.runtime_s = r3((Date.now() - t0) / 1000);
fs.writeFileSync(OUTF, JSON.stringify(results, null, 1));
if (flag('dets')) fs.writeFileSync(path.join(path.dirname(OUTF), 'dets.json'), JSON.stringify(detDump));

// ── table ──
const f = (v, w = 6, d = 3) => (v == null ? '-' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v)).padStart(w);
const row = (name, m) => `${name.padEnd(22)}${f(m.labels, 5)}${f(m.detections, 5)}${f(m.tp, 5)}${f(m.fp, 5)}${f(m.fn, 5)}${f(m.dup, 4)}${f(m.ignored, 4)}${f(m.confused, 5)}` +
  `${f(m.precision)}${f(m.recall)}${f(m.f1)}${f(m.latency_median, 7, 2)}${f(m.latency_p90, 6, 2)}${f(m.fp_per_min, 7, 2)}${f(m.det_per_min, 7, 2)}${f(m.labels_per_min, 7, 2)}`;
console.log(`perception.js ${percInfo.mtime_local} sha1 ${percInfo.sha1} | ${useClips.size} clips, ${streams.length} streams, ${tot.min.toFixed(1)} face-min | feed ${FEED_FPS} fps | opts ${JSON.stringify(OPTS)}`);
console.log(`${''.padEnd(22)}${'lab'.padStart(5)}${'det'.padStart(5)}${'TP'.padStart(5)}${'FP'.padStart(5)}${'FN'.padStart(5)}${'dup'.padStart(4)}${'ign'.padStart(4)}${'conf'.padStart(5)}` +
  `${'P'.padStart(6)}${'R'.padStart(6)}${'F1'.padStart(6)}${'latMed'.padStart(7)}${'p90'.padStart(6)}${'FP/min'.padStart(7)}${'det/m'.padStart(7)}${'lab/m'.padStart(7)}`);
for (const ty of TYPES) console.log(row(ty, results[ty]));
for (const ty of TYPES) console.log(row(ty + ' (frontal)', results.frontal[ty]));
for (const [c, v] of Object.entries(results.by_category)) for (const ty of TYPES) console.log(row(`${ty} [${c}]`, v[ty]));
const E = exprSum;
console.log(`expression_change: ${E.count} events = ${E.per_min}/min (per stream median ${E.stream_rate_median}, p90 ${E.stream_rate_p90}, max ${E.stream_rate_max}; ` +
  `${E.streams_over_2_per_min}/${streams.length} streams > 2/min; min gap ${E.min_gap_s} s; intensity median ${E.intensity_median})`);
console.log(`  to: ${Object.entries(E.to).map(([k, v]) => `${k} ${v}`).join(', ') || '-'}   | by category /min: ${Object.entries(results.by_category).map(([c, v]) => `${c} ${v.expression_change_per_min}`).join(', ')}`);
console.log(`  state time (face present): ${Object.entries(E.state_time_frac).map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(', ')}`);
if (Object.keys(otherEv).length) console.log(`other events: ${JSON.stringify(otherEv)}`);
if (errors.length) console.log(`ERRORS in ${errors.length} streams: ${errors[0].clip}_s${errors[0].k}: ${errors[0].error}`);
if (flag('per-stream')) for (const p of perStream) console.log(`  ${(p.clip + '_s' + p.k).padEnd(22)} ${p.category.padEnd(6)} ${f(p.present_min, 6, 2)} min  nod ${p.nod.tp}/${p.nod.labels} fp ${p.nod.fp}  shake ${p.head_shake.tp}/${p.head_shake.labels} fp ${p.head_shake.fp}  expr ${p.expression_change} (${p.expr_per_min}/min)`);
console.log(`→ ${OUTF}  (${results.runtime_s} s)`);
