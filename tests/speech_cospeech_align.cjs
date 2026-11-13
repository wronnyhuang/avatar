// speech — offline co-speech timing test against TTS ground truth (ElevenLabs with-timestamps, tools/cospeech/gen_tts.py).
// c2 rebuild: the independent tester's text-arrival modes, three real TTS voices (one tuning voice, two held-out voices
// never tuned on), several planner seeds per utterance, a fresh co-speech instance per utterance (no call-level rate
// learning: the hard case). For each utterance the 24 kHz audio is fed through app/cospeech.js in 20 ms tap chunks on
// the playback timeline, frames tick at 30 Hz, and every committed stroke apex is compared with the TTS's stressed-
// syllable nucleus of the word it was planned on. Text arrival modes:
//   early    — the WHOLE reply text arrives 0.5–3.0 s before the voice (a burst of deltas)
//   stream2x — the text streams at 2× the audio rate (word i at start + t_i/2, start 0–0.4 s before the voice)
//   jitter   — each word arrives 0.15–0.6 s before its audio, ± 80 ms jitter (kept in order)
// Gates (exit 1 on failure): apex within ±250 ms ≥ 90% in every mode × voice; ≥ 4 distinct gesture types per mode;
// no identical (type + side) back-to-back strokes; no stroke committed after a barge-in cancel and no stroke apex on
// cancelled audio; busy / holding paws never used.
//   node tests/speech_cospeech_align.cjs [out.json]        COS_ALIGNER=legacy → the c1 tracker (for the before/after table)
'use strict';
const fs = require('fs'), path = require('path');
const CS = require('./load.cjs')('app/cospeech.js');
const ROOT = path.join((process.env.AVATAR_DATA || path.join(__dirname, '..', 'data')), 'speech');
const SETS = [['tune', ROOT + '/tts'], ['heldout', ROOT + '/tts_heldout'], ['heldout2', ROOT + '/tts_heldout2']].filter(([, d]) => fs.existsSync(d));
const OUT = process.argv[2] || ROOT + '/align.json';
const MODES = ['early', 'stream2x', 'jitter'];
const SEEDS = (process.env.COS_SEEDS || '7,21,99').split(',').map(Number);
const ALIGNER = process.env.COS_ALIGNER || undefined;
const pct = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function wav(p) { const b = fs.readFileSync(p); let o = 12; while (b.toString('ascii', o, o + 4) !== 'data') o += 8 + b.readUInt32LE(o + 4); const n = b.readUInt32LE(o + 4) / 2; const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = b.readInt16LE(o + 8 + 2 * i) / 32768; return x; }
// ground truth: words (same tokenizer) → char spans → start / stressed-nucleus time
function truth(j) {
  const text = j.text, low = text.toLowerCase(), toks = CS.tokenize(text);
  let pos = 0;
  return toks.map(W => {
    const k = low.indexOf(W.raw.toLowerCase(), pos); pos = k + W.raw.length;
    const t0 = j.start[k], t1 = j.end[k + W.raw.length - 1];
    const word = W.raw.toLowerCase(); const groups = []; const re = /[aeiouy]+/g; let m;
    while ((m = re.exec(word))) if (!(m.index === 0 && m[0] === 'y' && word.length > 1)) groups.push([m.index, m.index + m[0].length - 1]);
    if (groups.length > 1 && /[^aeiouy]e$/.test(word) && !/[^aeiouy]le$/.test(word)) groups.pop();
    const g = groups[Math.min(W.stress, groups.length - 1)] || [0, word.length - 1];
    const nuc = (j.start[k + g[0]] + j.end[k + g[1]]) / 2;
    return {w: W.raw, t0, t1, nuc};
  });
}
const cache = new Map();
function load(dir, id) { const k = dir + id; if (!cache.has(k)) { const j = JSON.parse(fs.readFileSync(path.join(dir, id + '.json'))); cache.set(k, {j, x: wav(path.join(dir, id + '.wav')), tr: truth(j)}); } return cache.get(k); }
function arrivals(mode, j, tr, ON, rnd) {
  const toks = j.text.match(/\S+\s*/g);
  // token i ↔ the tokenizer's word for its first word (numbers expand into several words; take the first one's time)
  let wi = 0; const tt = toks.map(tk => { const n = Math.max(1, CS.tokenize(tk).length); const t = tr[Math.min(wi, tr.length - 1)].t0; wi += n; return t; });
  if (mode === 'early') { const t0 = ON - (0.5 + 2.5 * rnd()); return toks.map((tk, i) => [t0 + 0.004 * i, tk]); }
  if (mode === 'stream2x') { const t0 = ON - 0.4 * rnd(); return toks.map((tk, i) => [t0 + tt[i] / 2, tk]); }
  let last = -Infinity;
  return toks.map((tk, i) => { const t = Math.max(last + 0.001, ON + tt[i] - (0.15 + 0.45 * rnd()) + 0.16 * (rnd() - 0.5)); last = t; return [t, tk]; });
}
function run(dir, id, mode, opts = {}) {
  const {j, x, tr} = load(dir, id);
  const rec = []; let gid = 0, T = 0;
  const live = new Map();
  const motion = {playGesture: (def, o) => { const r = {name: def.name, at: o.at, side: o.side, id: ++gid}; rec.push(r); live.set(r.id, r); return r.id; },
    armsBusy: () => (opts.busy || {}), state: () => ({holding: opts.holding || {}}),
    cancelGestures: f => { if (f && f.id) { const r = live.get(f.id); if (r) r.cancelT = T; return; } rec.cancelAt = rec.cancelAt ?? T; for (const r of rec) if (r.cancelT == null) r.cancelT = T; }};
  const seed = opts.seed || 7, rnd = mulberry32(seed * 7919 + id.length * 31 + mode.length);
  const C = CS.createCospeech(Object.assign({motion, seed, aligner: ALIGNER}, JSON.parse(process.env.COS_OPTS || '{}')));
  const ON = 3.2;                                    // the voice reaches the tap at 3.2 s (early text may come 3 s before)
  const dur = x.length / 24000, cut = opts.cut ?? Infinity;
  const arr = arrivals(mode, j, tr, ON, rnd);
  let ai = 0, ci = 0, nextFrame = 0;
  for (T = 0; T < ON + dur + 1.0; T += 0.005) {
    while (ai < arr.length && arr[ai][0] <= T) { C.text(arr[ai][1], {response_id: 'r1'}); ai++; }
    while (ON + (ci + 1) * 0.02 <= T + 1e-9) {          // 20 ms chunk [ON + ci*0.02, +0.02) has fully passed the tap
      const a = ci * 480, b = Math.min(x.length, a + 480); let e = 0;
      const t0 = ON + ci * 0.02, cutNow = t0 >= ON + cut;
      for (let i = a; i < b; i++) e += x[i] * x[i];
      const rms = b > a && !cutNow ? Math.sqrt(e / (b - a)) : 0;
      C.audio(t0, rms, rms < 24 / 32768, 0.02, opts.nopcm || cutNow || b <= a ? undefined : x.subarray(a, b));
      ci++;
      if (cutNow && !rec.cancelled) { rec.cancelled = true; rec.cancelT = T; C.cancel('barge-in'); }
    }
    if (T >= nextFrame) { nextFrame += 1 / 30; C.tick(T); }
  }
  const P = C.plan();
  const strokes = P.gestures.filter(g => g.committed).map(g => ({type: g.type, w: g.w, word: g.word, apex: g.apex_t, at: g.at,
    off: 1000 * (g.apex_t - (ON + tr[g.word].nuc)), side: g.side, id: g.id}));
  // a stroke "on cancelled audio": its apex is after the cut was known (the chunk past the cut reached the tap) and the
  // motion side did not cancel it before the apex
  const onCut = rec.cancelled ? strokes.filter(s => s.apex > Math.max(ON + cut, rec.cancelT) && !(live.get(s.id) && live.get(s.id).cancelT != null && live.get(s.id).cancelT < s.apex)).length : 0;
  const afterCancel = rec.cancelled ? C.commits().filter(c => c.commitT > rec.cancelT).length : 0;
  return {id, mode, dur, strokes, planned: P.gestures.length, dropped: P.gestures.filter(g => g.dropped).map(g => String(g.dropped).replace(/ \(.*/, '')), afterCancel, onCut, stats: C.stats()};
}
const res = {aligner: ALIGNER || 'default', seeds: SEEDS, sets: {}, modes: {}};
const gateFail = [];
const table = [];
for (const mode of MODES) {
  const all = [];
  for (const [set, dir] of SETS) {
    const ids = fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => f.replace('.json', '')).sort();
    const runs = []; for (const id of ids) for (const seed of SEEDS) runs.push(run(dir, id, mode, {seed}));
    if (process.env.COS_MISSES) for (const r of runs) for (const s of r.strokes) if (Math.abs(s.off) > 250) console.log(`  miss ${mode}/${set} ${r.id} ${s.type} "${s.w}" ${Math.round(s.off)} ms`);
    const offs = runs.flatMap(r => r.strokes.map(s => s.off));
    const within = offs.filter(o => Math.abs(o) <= 250).length, n = offs.length, speech = runs.reduce((s, r) => s + r.dur, 0);
    const m = {voice: load(dir, ids[0]).j.voice, utts: ids.length, strokes: n, per_s: +(n / speech).toFixed(2), within_250: +(within / Math.max(1, n)).toFixed(3),
      median_ms: Math.round(pct(offs, .5)), abs_p90_ms: Math.round(pct(offs.map(Math.abs), .9))};
    (res.sets[set] = res.sets[set] || {})[mode] = m;
    table.push([mode, set, m]);
    all.push(...runs);
    if (m.within_250 < 0.9) gateFail.push(`${mode}/${set} within±250 ${m.within_250}`);
    if (m.per_s < 0.2) gateFail.push(`${mode}/${set} too few strokes ${m.per_s}/s`);
  }
  const types = {}; for (const r of all) for (const s of r.strokes) types[s.type] = (types[s.type] || 0) + 1;
  let rep = 0, pairs = 0; for (const r of all) { const st = r.strokes.slice().sort((a, b) => a.apex - b.apex); for (let i = 1; i < st.length; i++) { pairs++; if (st[i].type === st[i - 1].type && st[i].side === st[i - 1].side) rep++; } }
  const offs = all.flatMap(r => r.strokes.map(s => s.off));
  const dropped = all.flatMap(r => r.dropped).reduce((a, d) => (a[d] = (a[d] || 0) + 1, a), {});
  res.modes[mode] = {strokes: offs.length, within_250: +(offs.filter(o => Math.abs(o) <= 250).length / Math.max(1, offs.length)).toFixed(3), types, distinct: Object.keys(types).length, same_back_to_back: rep, pairs, dropped};
  if (Object.keys(types).length < 4) gateFail.push(mode + ' distinct types ' + Object.keys(types).length);
  if (rep) gateFail.push(`${mode} identical back-to-back ${rep}/${pairs}`);
  console.log(`${mode.padEnd(8)} all voices: strokes ${offs.length} within ±250 ${(100 * res.modes[mode].within_250).toFixed(1)}%  types ${Object.keys(types).length} ${JSON.stringify(types)}  identical back-to-back ${rep}/${pairs}  dropped ${JSON.stringify(dropped)}`);
}
console.log('\nmode      voice set  (voice)                         strokes  /s    within±250  median  |.|p90');
for (const [mode, set, m] of table) console.log(`${mode.padEnd(9)} ${set.padEnd(10)} ${String(m.voice).slice(0, 30).padEnd(31)} ${String(m.strokes).padStart(6)}  ${m.per_s.toFixed(2)}  ${(100 * m.within_250).toFixed(1).padStart(8)}%  ${String(m.median_ms).padStart(6)}  ${String(m.abs_p90_ms).padStart(5)}`);
// barge-in: every utterance cut at 45% of its length, in every mode: nothing committed after the cancel, no apex on cut audio
let bAfter = 0, bOnCut = 0, bN = 0;
for (const mode of MODES) for (const [, dir] of SETS) for (const id of fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => f.replace('.json', ''))) {
  const {x} = load(dir, id); const r = run(dir, id, mode, {cut: 0.45 * x.length / 24000, seed: 5}); bAfter += r.afterCancel; bOnCut += r.onCut; bN++;
}
console.log(`\nbarge-in (${bN} cut replies): strokes committed after the cancel ${bAfter}; stroke apexes on cancelled audio ${bOnCut}`);
if (bAfter || bOnCut) gateFail.push(`barge-in after=${bAfter} onCut=${bOnCut}`);
// busy paws: the right (screen-left) paw holding something, both busy
const TD = SETS[0][1], HB = fs.existsSync(path.join(TD, 'three.json')) ? 'three' : 'plan';
const hb = run(TD, HB, 'early', {holding: {right: 'cup'}}), bb = run(TD, HB, 'early', {busy: {l: true, r: true}});
const onHeld = hb.strokes.filter(s => s.side === 'l' && !/nod|brow|shake|shrug_small/.test(s.type)).length;
const armsWhenBusy = bb.strokes.filter(s => !/nod|brow|head_shake|shrug_small/.test(s.type)).length;
console.log(`holding in its right paw: arm strokes on that paw ${onHeld} (of ${hb.strokes.length}); both paws busy: arm strokes ${armsWhenBusy} (of ${bb.strokes.length}, head/face-only ok)`);
if (onHeld || armsWhenBusy) gateFail.push('busy paws used');
res.bargein = {replies: bN, afterCancel: bAfter, onCut: bOnCut}; res.busy = {onHeld, armsWhenBusy};
fs.writeFileSync(OUT, JSON.stringify(res, null, 1));
console.log(gateFail.length ? 'FAIL ' + gateFail.join('; ') : 'PASS all co-speech timing gates');
process.exit(gateFail.length ? 1 : 0);
