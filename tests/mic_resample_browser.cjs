// The Gemini mic resampler in a REAL browser: headless Chrome loads the exact AudioWorklet module the page loads
// (AvatarVoice.workletSource() from app/voice.js), runs the 'avatar-pcm16-tap' processor in an OfflineAudioContext at
// 24 / 44.1 / 48 kHz on a sweep + a speech clip, and the 16 kHz int16 it posts must equal the node run of the same
// code bit for bit (proves the worklet wiring, block handling and int16 conversion, not just the math). Also ?micfir=0's
// processorOptions {fir:false} path. Needs playwright-core + a Chrome:
//   PLAYWRIGHT_CORE=<.../node_modules/playwright-core> CHROME_PATH=<chrome binary> node tests/mic_resample_browser.cjs [speech.wav]
'use strict';
const fs = require('fs'), path = require('path'), os = require('os');
const ROOT = path.join(__dirname, '..');
let chromium;
try { ({chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core')); } catch (e) { console.log('SKIP: playwright-core not found (set PLAYWRIGHT_CORE=)'); process.exit(0); }
const CHROME = [process.env.CHROME_PATH, '/opt/google/chrome/chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find(p => p && fs.existsSync(p));
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined ? '  — ' + JSON.stringify(detail) : '')); }

const SRC = fs.readFileSync(path.join(ROOT, 'app/voice.js'), 'utf8');
const W = {}; new Function('window', 'performance', 'setTimeout', 'clearTimeout', SRC)(W, {now: () => 0}, () => 0, () => {});
function nodeTap(rate, x, fir) {        // the same worklet code under node, 128-sample quanta
  const posts = []; let Tap = null;
  class AWP { constructor() { this.port = {postMessage: m => posts.push(m instanceof Int16Array ? m.slice(0) : m)}; } }
  Object.defineProperty(globalThis, 'currentTime', {get: () => 0, configurable: true});
  new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', 'currentFrame', W.AvatarVoice.workletSource())(AWP, (n, c) => { if (n === 'avatar-pcm16-tap') Tap = c; }, rate, 0);
  const t = new Tap({processorOptions: {fir}});
  for (let i = 0; i < x.length; i += 128) { const b = new Float32Array(128); b.set(x.subarray(i, Math.min(x.length, i + 128))); t.process([[b]], [[new Float32Array(128)]]); }
  return {chunks: posts.filter(p => p instanceof Int16Array), cfg: posts.find(p => p && p.t === 'cfg')};
}
function readWav(p) {
  const b = fs.readFileSync(p); let off = 12, sr = 0, ch = 1, data = null;
  while (off + 8 <= b.length) { const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4); if (id === 'fmt ') { ch = b.readUInt16LE(off + 10); sr = b.readUInt32LE(off + 12); } if (id === 'data') { data = b.subarray(off + 8, off + 8 + sz); break; } off += 8 + sz + (sz & 1); }
  const n = Math.floor(data.length / 2 / ch), x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = data.readInt16LE(2 * i * ch) / 32768; return {sr, x};
}
function input(rate, speech) {          // 3 s log sweep 50 Hz -> Nyquist at 0.5 + 0.2 s silence + the speech clip (taken as samples at `rate`)
  const ns = 3 * rate, f0 = 50, f1 = rate / 2 * 0.995, k = Math.log(f1 / f0), sp = speech ? speech.x : new Float32Array(0);
  const x = new Float32Array(ns + Math.round(0.2 * rate) + sp.length);
  for (let i = 0; i < ns; i++) { const t = i / rate, T = 3; x[i] = 0.5 * Math.sin(2 * Math.PI * f0 * T / k * (Math.exp(t / T * k) - 1)); }
  x.set(sp, ns + Math.round(0.2 * rate));
  return x;
}

(async () => {
  if (!CHROME) { console.log('SKIP: no Chrome (set CHROME_PATH=)'); process.exit(0); }
  const speech = process.argv[2] ? readWav(process.argv[2]) : null;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'micrs-'));
  fs.writeFileSync(path.join(dir, 'voice.js'), SRC);
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><meta charset=utf-8><script src="voice.js"></script><body>ok</body>');
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--allow-file-access-from-files', '--autoplay-policy=no-user-gesture-required']});
  try {
    const page = await browser.newPage();
    await page.goto('file://' + path.join(dir, 'index.html'));
    for (const rate of [24000, 44100, 48000]) for (const fir of [true, false]) {
      const x = input(rate, speech);
      const got = await page.evaluate(async ({xs, rate, fir}) => {
        const ctx = new OfflineAudioContext(1, xs.length + 2 * 128, rate);
        await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([AvatarVoice.workletSource()], {type: 'application/javascript'})));
        const buf = ctx.createBuffer(1, xs.length, rate); buf.getChannelData(0).set(Float32Array.from(xs));
        const src = ctx.createBufferSource(); src.buffer = buf;
        const tap = new AudioWorkletNode(ctx, 'avatar-pcm16-tap', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: {fir}});
        const chunks = []; let cfg = null;
        tap.port.onmessage = ev => { const d = ev.data; if (d && d.t === 'cfg') cfg = d; else chunks.push(Array.from(d)); };
        src.connect(tap); tap.connect(ctx.destination); src.start(0);
        await ctx.startRendering();
        await new Promise(r => setTimeout(r, 800));
        return {chunks, cfg, rate: ctx.sampleRate};
      }, {xs: Array.from(x), rate, fir});
      const want = nodeTap(rate, x, fir);
      // the browser renders whole quanta past the end of the input (silence) — compare the chunks both produced
      const n = Math.min(got.chunks.length, want.chunks.length);
      let diff = 0, first = -1;
      for (let c = 0; c < n; c++) for (let i = 0; i < 320; i++) if (got.chunks[c][i] !== want.chunks[c][i]) { diff++; if (first < 0) first = c * 320 + i; }
      const expect = Math.floor(x.length * 16000 / rate / 320);
      check(`browser ${rate} Hz ${fir ? 'NEW (fir)' : 'OLD (?micfir=0)'}: ${n} chunks x 320 bit-identical to node${speech ? ' (sweep + speech)' : ' (sweep)'}`,
        diff === 0 && n >= expect - 1 && got.chunks.every(c => c.length === 320), {browser_chunks: got.chunks.length, node_chunks: want.chunks.length, expect, diff, first});
      check(`browser ${rate} Hz ${fir ? 'NEW' : 'OLD'}: cfg message`, got.cfg && got.cfg.fir === fir && got.cfg.rate === rate && got.cfg.taps === want.cfg.taps, got.cfg);
    }
  } finally { await browser.close(); fs.rmSync(dir, {recursive: true, force: true}); }
  console.log(`\n${passes} passed, ${fails} failed`);
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
