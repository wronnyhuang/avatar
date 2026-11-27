// the measurable check: his look-aways replayed through app/motion.js, old timing (idle2 off) vs new.
//   node tests/idle_away_replay.cjs [--seeds 200] [--real DIR] [--real-seeds 40] [--base PATH/motion.js] [--json OUT] [--jobs 12]
// Synthetic timelines (looking at the screen 20 s, away 60 s, back 20 s; three variants) and, with --real DIR, real
// recorded sessions: DIR/perc/<name>/<name>.jsonl (tests/perception_smoke.cjs on his camera picture cut out of a
// recording) + DIR/clips/<name>.wav (the recording's audio = his mic + the avatar's voice).
// --base: a baseline motion.js (e.g. git show <older commit>:app/motion.js) — checks that idle2 off replays it idle for idle.
'use strict';
const fs = require('fs'), path = require('path'), WT = require('worker_threads');
const L = require('./idle_away_lib.cjs');
const loadAt = fp => { const mod = {exports: {}}; new Function('module', 'exports', fs.readFileSync(fp, 'utf8'))(mod, mod.exports); return mod.exports; };
const timeline = spec => spec.synth !== undefined ? L.SYNTH()[spec.synth] : L.realSession(spec.name, spec.jsonl, spec.wav);

if (!WT.isMainThread) {   // a worker: one timeline, one variant, a list of seeds
  const {spec, seeds, idle2, motion} = WT.workerData, M = motion ? loadAt(motion) : L.load('app/motion.js'), tl = timeline(spec);
  WT.parentPort.postMessage(seeds.map(seed => L.run(M, tl, {seed, idle2})));
  return;
}
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const M = L.load('app/motion.js');
const SEEDS = +arg('seeds', 200), RSEEDS = +arg('real-seeds', 40), REAL = arg('real', null), BASE = arg('base', null), OUT = arg('json', null), JOBS = +arg('jobs', 12);
const queue = []; let active = 0;
function pump() { while (active < JOBS && queue.length) { const q = queue.shift(); active++; const w = new WT.Worker(__filename, {workerData: q.data}); w.once('message', r => { active--; q.res(r); pump(); }); w.once('error', e => { active--; q.rej(e); pump(); }); } }
const job = data => new Promise((res, rej) => { queue.push({data, res, rej}); pump(); });
async function many(spec, n, idle2, motion) {
  const per = Math.max(1, Math.ceil(n / Math.min(JOBS, n))), parts = [];
  for (let s = 1; s <= n; s += per) parts.push(job({spec, idle2, motion, seeds: Array.from({length: Math.min(per, n - s + 1)}, (_, k) => s + k)}));
  return (await Promise.all(parts)).flat();
}
const report = {constants: {IDLE_AWAY_ONSET: M.IDLE_AWAY_ONSET, IDLE_AWAY_RATE: M.IDLE_AWAY_RATE, IDLE_AWAY: M.IDLE_AWAY}, synthetic: {}, real: {}};
function show(group, name, pool, n) {
  const o = {old: L.measure(pool.old), new: L.measure(pool.new)};
  o.ratio = {onset_median: +(o.new.onset_median_s / o.old.onset_median_s).toFixed(2), onset_p90: +(o.new.onset_p90_s / o.old.onset_p90_s).toFixed(2),
    idles_per_min_away: +(o.new.idles_per_min_away / o.old.idles_per_min_away).toFixed(2)};
  report[group][name] = o;
  console.log(`\n== ${group} / ${name}  (${n} seeds)`);
  for (const k of ['old', 'new']) console.log(k.toUpperCase().padEnd(4), JSON.stringify(o[k]));
  console.log('new/old', JSON.stringify(o.ratio));
}
async function group(g, specs, names, n) {
  const all = {old: [], new: []};
  const pools = await Promise.all(specs.map(async spec => ({old: await many(spec, n, false), new: await many(spec, n, true)})));
  pools.forEach((p, i) => { show(g, names[i], p, n); all.old.push(...p.old); all.new.push(...p.new); });
  if (specs.length > 1) show(g, 'ALL', all, n);
}
(async () => {
  const synth = L.SYNTH();
  await group('synthetic', synth.map((_, i) => ({synth: i})), synth.map(t => t.name), SEEDS);
  if (BASE) {
    let same = 0, n = 0;
    for (let i = 0; i < synth.length; i++) {
      const a = await many({synth: i}, 24, undefined, path.resolve(BASE)), b = await many({synth: i}, 24, false);
      for (let k = 0; k < a.length; k++) { n++; if (JSON.stringify(a[k].idles) === JSON.stringify(b[k].idles)) same++; }
    }
    report.base_equal = `${same}/${n}`;
    console.log(`\nidle2 off vs baseline motion.js: identical idle logs in ${same}/${n} runs`);
  }
  if (REAL) {
    const names = fs.readdirSync(path.join(REAL, 'perc')).filter(n => fs.existsSync(path.join(REAL, 'perc', n, n + '.jsonl'))).sort(), specs = [], keep = [];
    for (const n of names) {
      const spec = {name: n, jsonl: path.join(REAL, 'perc', n, n + '.jsonl'), wav: path.join(REAL, 'clips', n + '.wav')}, tl = timeline(spec);
      if (tl.dur < 30) continue;
      const st = L.stretches(L.run(M, tl, {seed: 1, idle2: false}).att);
      console.log(`-- ${n}: ${tl.dur.toFixed(0)} s, someone speaking ${(100 * (tl.voicedShare || 0)).toFixed(0)} % of it, ${st.length} look-aways ≥ 3 s (${st.reduce((s, x) => s + x[1] - x[0], 0).toFixed(0)} s in all; longest ${Math.max(0, ...st.map(x => x[1] - x[0])).toFixed(0)} s)`);
      report.real[n + '_info'] = {dur_s: +tl.dur.toFixed(0), speaking_share: tl.voicedShare, look_aways: st.map(x => [+x[0].toFixed(1), +x[1].toFixed(1)])};
      specs.push(spec); keep.push(n);
    }
    await group('real', specs, keep, RSEEDS);
  }
  if (OUT) fs.writeFileSync(OUT, JSON.stringify(report, null, 1));
})();
