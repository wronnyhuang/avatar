// buzz: the REAL avatar page (a running server, through its URL) in Chrome whose speaker output goes to the
// null sink $BUZZ_SINK (default avatarbuzz); the page runs the mock voice through a real WebRTC track (voice=loopback, $0). Taps in the page:
// 'out' = D.play.out (what the recorder records), 'heard' = the echo-canceller loopback's stream (what the <audio> plays);
// the sink monitor (sinkdev.sh) = what the machine's speaker got. Also returns D.play.stats() (aec_neteq after the fix).
//   node devcheck.cjs --url U --key K --out DIR --label L [--seconds 60] [--voice loopback] [--sink NAME]
// --url/--key default to $AVATAR_URL / $AVATAR_KEY. Env: PLAYWRIGHT_CORE, CHROME_PATH (default /opt/google/chrome/chrome
// when present), BUZZ_SINK.
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const a = {seconds: 60, voice: 'loopback', label: 'dev', 'boot-s': 300, url: process.env.AVATAR_URL, key: process.env.AVATAR_KEY, sink: process.env.BUZZ_SINK || 'avatarbuzz'};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!a.url || !a.key || !a.out) { console.error('usage: node devcheck.cjs --url U --key K --out DIR --label L (or AVATAR_URL / AVATAR_KEY)'); process.exit(2); }
fs.mkdirSync(a.out, {recursive: true});
const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', a.voice); u.searchParams.set('char', 'muse');
u.searchParams.set('attention', '0'); if (a.mock) u.searchParams.set('mock', a.mock); u.searchParams.set('hud', '1');
function wav(i16buf, sr) {
  const h = Buffer.alloc(44); h.write('RIFF', 0); h.writeUInt32LE(36 + i16buf.length, 4); h.write('WAVEfmt ', 8); h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(sr, 24); h.writeUInt32LE(sr * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(i16buf.length, 40); return Buffer.concat([h, i16buf]);
}
(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || ['/opt/google/chrome/chrome'].find(p => fs.existsSync(p)), headless: true, ignoreDefaultArgs: ['--mute-audio'],
    env: {...process.env, PULSE_SINK: a.sink, PULSE_PROP: 'application.name=avatarbuzz media.role=test'},
    args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows']});
  const ctx = await browser.newContext({viewport: {width: 412, height: 915}, permissions: ['camera', 'microphone']});
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 200)));
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await page.waitForTimeout(1500);
  try { await page.click('#go', {timeout: 5000}); } catch (e) {}
  try { await page.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live && window.__avatar.D && window.__avatar.D.play && window.__avatar.D.play.aec,
    null, {timeout: 1000 * +a['boot-s']}); }
  catch (e) { console.log(JSON.stringify({ok: false, error: 'never live', errors})); await browser.close(); process.exit(2); }
  await page.evaluate(async () => {
    const CAP = `class Cap extends AudioWorkletProcessor { constructor() { super(); this.on = true; this.port.onmessage = e => { if (e.data === 'stop') this.on = false; }; }
      process(i, o) { const x = i[0] && i[0][0]; if (this.on) this.port.postMessage(x ? x.slice(0) : new Float32Array(128)); return true; } }
      registerProcessor('bzcap', Cap);`;
    const mk = async (c) => { await c.audioWorklet.addModule(URL.createObjectURL(new Blob([CAP], {type: 'application/javascript'})));
      const n = new AudioWorkletNode(c, 'bzcap', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]}); const parts = [];
      n.port.onmessage = e => parts.push(e.data); const z = c.createGain(); z.gain.value = 0; n.connect(z).connect(c.destination); return {n, parts, sr: c.sampleRate}; };
    const {S, D} = window.__avatar;
    const out = await mk(S.ctx); D.play.out.connect(out.n);
    const c2 = new AudioContext({sampleRate: 48000}); await c2.resume();
    const heard = await mk(c2); c2.createMediaStreamSource(D.play.aec.el.srcObject).connect(heard.n);
    window.__bz = {out, heard};
  });
  // mock replies on a fixed cadence (the voice's own test seam = his end of turn), unless --drive 0 (a real voice)
  const t0 = Date.now(); let replies = 0;
  while (Date.now() - t0 < 1000 * +a.seconds) {
    if (a.drive !== '0') { const ok = await page.evaluate(() => { const v = window.__avatar.D.voice; if (!v || !v.feed) return false; v.feed({type: 'test.mock_user_end'}); return true; }).catch(() => false); if (ok) replies++; }
    await page.waitForTimeout(6500);
  }
  const res = await page.evaluate(() => {
    const b64 = f32 => { const i16 = new Int16Array(f32.length); for (let i = 0; i < f32.length; i++) i16[i] = Math.round(Math.max(-1, Math.min(1, f32[i])) * 32767);
      const u = new Uint8Array(i16.buffer); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
    const r = {stats: window.__avatar.D.play.stats(), taps: {}};
    for (const [k, t] of Object.entries(window.__bz)) { t.n.port.postMessage('stop'); let L = 0; for (const p of t.parts) L += p.length; const f = new Float32Array(L); let o = 0; for (const p of t.parts) { f.set(p, o); o += p.length; }
      let clip = 0; for (const v of f) if (Math.abs(v) >= 0.999) clip++; r.taps[k] = {sr: t.sr, clip, b64: b64(f)}; }
    return r;
  });
  try { await page.click('#go', {timeout: 3000}); } catch (e) {}     // Stop: ends the session cleanly
  await page.waitForTimeout(2000);
  const meta = {label: a.label, url: a.url.replace(/([?&]k=)[^&]+/, '$1…'), stats: res.stats, errors, taps: {}};
  for (const [k, t] of Object.entries(res.taps)) { const f = path.join(a.out, `${a.label}.${k}.wav`); fs.writeFileSync(f, wav(Buffer.from(t.b64, 'base64'), t.sr)); meta.taps[k] = {file: f, sr: t.sr, clip: t.clip}; }
  fs.writeFileSync(path.join(a.out, a.label + '.json'), JSON.stringify(meta, null, 1));
  console.log(JSON.stringify({label: a.label, replies, stats: res.stats, taps: meta.taps, errors: errors.slice(0, 3)}));
  await browser.close();
})();
