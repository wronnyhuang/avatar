// node tools/prior/sim_live.mjs <prior_weights.json> <live_audio.json> <out.jsonl>
// Real voices through the LIVE feature path: each clip's audio is (a) the avatar's own voice (speaking, no explicit
// speaking flag -> the JS VAD decides) and then (b) the user's voice while the avatar listens with a neutral user face.
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url));
await import(pathToFileURL(path.join(here, '../../app/prior.js')).href);
const [wp, ap_, outp] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(wp, 'utf8')), S = JSON.parse(fs.readFileSync(ap_, 'utf8'));
const p = globalThis.createPrior(W, { seed: 3 });
const params = ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y', 'blink_l', 'blink_r', 'brow_l', 'brow_r', 'smile', 'squint', 'eye_wide'];
const lines = [JSON.stringify({ params })]; let t = 0; let spkFrames = 0, n = 0;
const face = { present: true, blend: { mouthSmileLeft: 0.05, mouthSmileRight: 0.05, jawOpen: 0.05 }, yaw: 0, pitch: 0 };
for (const s of S) {
  for (const role of ['speaking', 'listening']) {
    for (let i = 0; i < s.rms.length; i++) {
      const v = { rms: s.rms[i], pitch: s.pitch[i] };
      const o = role === 'speaking' ? p.step({ self: v, partner: { rms: 0.001, pitch: 0 }, face })
                                    : p.step({ self: { rms: 0, pitch: 0 }, partner: v, face });
      const f = o._features; if (role === 'speaking') { spkFrames += f[3]; n++; }
      const out = {}; for (const k of params) out[k] = +o[k].toFixed(5);
      lines.push(JSON.stringify({ t: +t.toFixed(4), v: out, state: role, f0: role === 'speaking' ? s.pitch[i] : 0 }));
      t += 1 / 30;
    }
  }
}
fs.writeFileSync(outp, lines.join('\n') + '\n');
console.log(`wrote ${outp}: ${lines.length - 1} frames; JS VAD says the avatar speaks ${(100 * spkFrames / n).toFixed(0)}% of its speaking turns`);
