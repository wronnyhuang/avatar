#!/usr/bin/env node
/* gesture_prior.cjs — measures gestures.js's PRIOR (per-dimension mean / spread of normal conversational motion) on
 * the recorded eval clips (every clip's perception.jsonl through gestures.Features). Prints the JSON table that
 * is pasted into app/gestures.js. Trimmed: mean and sd of values between the 2nd and 98th percentile. */
'use strict';
const L = require('./perc_lib.cjs');
const G = L.loadUmd('app/gestures.js');
const ids = L.clipIds();
const D = G.LAYOUT.length, cols = Array.from({length: D}, () => []);
for (const id of ids) {
  const F = G.Features({});
  for (const f of L.loadStream(id)) { const v = F.compute(f); for (let i = 0; i < D; i++) if (v[i] === v[i]) cols[i].push(v[i]); }
}
const out = {};
G.LAYOUT.forEach((l, i) => {
  const a = cols[i]; if (!a.length) return;
  if (/^p_/.test(l.name)) { const m = a.reduce((x, y) => x + y, 0) / a.length; out[l.name] = [+m.toFixed(3), +Math.sqrt(m * (1 - m)).toFixed(3)]; return; }
  const lo = L.pct(a, 0.02), hi = L.pct(a, 0.98), b = a.filter(x => x >= lo && x <= hi);
  const m = b.reduce((x, y) => x + y, 0) / b.length, sd = Math.sqrt(b.reduce((x, y) => x + (y - m) * (y - m), 0) / b.length);
  out[l.name] = [+m.toFixed(3), +sd.toFixed(3)];
});
// symmetric pairs: average the spreads of left/right twins, mirror the means (so the prior has no handedness)
for (const l of G.LAYOUT) {
  if (l.mirror === l.name || !out[l.name] || !out[l.mirror]) continue;
  const a = out[l.name], b = out[l.mirror], sd = +((a[1] + b[1]) / 2).toFixed(3);
  const m = l.flip ? +((a[0] - b[0]) / 2).toFixed(3) : +((a[0] + b[0]) / 2).toFixed(3);
  out[l.name] = [m, sd]; out[l.mirror] = [l.flip ? -m : m, sd];
}
console.error('clips', ids.length, 'dims', Object.keys(out).length);
console.log(JSON.stringify(out));
