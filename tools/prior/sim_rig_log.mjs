// node tools/prior/sim_rig_log.mjs <prior_weights.json> <out.jsonl> [minutes=6]
// Drives app/prior.js through step() with synthetic RAW live inputs (avatar TTS rms/pitch, user mic, a user face)
// cycling speaking 20 s / listening 20 s / idle 10 s, and writes a rig log for tools/motion_stats.py.
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url));
await import(pathToFileURL(path.join(here, '../../app/prior.js')).href);
const [wp, outp, minArg] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(wp, 'utf8'));
const p = globalThis.createPrior(W, { seed: 7 });
let s = 12345; const u = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
function talker() {           // syllables ~4.5/s with 0.2-0.8 s pauses between phrases
  let t = 0, syl = 0, pause = 0, f0 = 180;
  return (on) => {
    if (!on) return { rms: 0, pitch: 0 };
    t += 1 / 30;
    if (pause > 0) { pause -= 1 / 30; return { rms: 0.002, pitch: 0 }; }
    syl += 4.5 / 30; if (u() < 0.012) pause = 0.2 + 0.6 * u();
    const env = Math.max(0, Math.sin(Math.PI * (syl % 1)));
    f0 += (180 + 30 * Math.sin(t * 1.3) - f0) * 0.1 + (u() - 0.5) * 4;
    return { rms: 0.15 * env + 0.003, pitch: env > 0.3 ? f0 : 0 };
  };
}
const me = talker(), you = talker();
const minutes = +(minArg || 6), N = minutes * 60 * 30;
const lines = [JSON.stringify({ params: ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z', 'gaze_x', 'gaze_y',
  'blink_l', 'blink_r', 'brow_l', 'brow_r', 'smile', 'squint', 'eye_wide'] })];
let smile = 0;
for (let i = 0; i < N; i++) {
  const t = i / 30, ph = t % 50, state = ph < 20 ? 'speaking' : ph < 40 ? 'listening' : 'idle';
  smile += ((u() < 0.01 ? 0.6 : 0) - smile) * 0.05;
  const face = { present: true, blend: { mouthSmileLeft: smile, mouthSmileRight: smile, jawOpen: state === 'listening' ? 0.2 * u() : 0.02 },
    yaw: 3 * Math.sin(t * 0.4), pitch: 2 * Math.sin(t * 0.7) };
  const o = p.step({ self: me(state === 'speaking'), partner: you(state === 'listening'), face });
  const v = {}; for (const k of JSON.parse(lines[0]).params) v[k] = +o[k].toFixed(5);
  lines.push(JSON.stringify({ t: +t.toFixed(4), v, state }));
}
fs.writeFileSync(outp, lines.join('\n') + '\n');
console.log(`wrote ${outp}: ${N} frames`);
