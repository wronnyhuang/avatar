// node tools/prior/test_prior_js.mjs <weights.json> <testvec.json>
// JS vs Python (PyTorch) parity: (1) raw network head on identical inputs, (2) a 450-frame mean-mode rollout.
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url));
await import(pathToFileURL(path.join(here, '../../app/prior.js')).href);   // classic/UMD script: defines globalThis.createPrior
const { createPrior } = globalThis;
const [wp, tp] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(wp, 'utf8')), V = JSON.parse(fs.readFileSync(tp, 'utf8'));
const sub = W.models ? { models: [W.models[V.model]], defaults: W.defaults } : W;
let maxRaw = 0, maxY = 0;
const q = createPrior(sub, { mode: 'mean', leak: 0, anchor: 0, smoothHz: 0, blinkSaccade: 1 });
for (const n of V.nets) {
  const raw = q._forward(n.x);
  n.raw.forEach((v, i) => { maxRaw = Math.max(maxRaw, Math.abs(v - raw[i])); });
}
const p = createPrior(sub, { mode: 'mean', leak: 0, anchor: 0, smoothHz: 0, blinkSaccade: 1 });
const t0 = process.hrtime.bigint();
V.ext.forEach((e, t) => {
  p.stepExt(e);
  V.y[t].forEach((v, i) => { maxY = Math.max(maxY, Math.abs(v - p._y[i])); });
});
const ms = Number(process.hrtime.bigint() - t0) / 1e6 / V.ext.length;
let maxRx = null;
if (V.rx_in && W.reaction) {
  const r = createPrior({ models: [W.models[V.model]], reaction: W.reaction }, { mode: 'mean' });
  maxRx = 0;
  V.rx_in.forEach((xi, t) => { const o = r._rxForward(xi); V.rx_out[t].forEach((v, i) => { maxRx = Math.max(maxRx, Math.abs(v - o[i])); }); });
}
console.log(JSON.stringify({ frames: V.ext.length, net_steps: V.nets.length, max_abs_diff_raw_head: maxRaw,
  max_abs_diff_rollout_y_normalized: maxY, max_abs_diff_reaction_model: maxRx, ms_per_frame: +ms.toFixed(4) }));
if (maxRaw > 1e-3 || maxY > 1e-2 || (maxRx !== null && maxRx > 1e-3)) { console.error('PARITY FAIL'); process.exit(1); }
console.log('PARITY OK');
