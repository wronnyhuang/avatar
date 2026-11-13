// node run.cjs --app <app dir with voice.js> --out DIR --label L --wavs a.wav,b.wav [--src buffer|rtc] [--aec 1] [--sr 24000] [--need 400] [--sink NAME]
// --wavs: the reply clips to play (e.g. the server's mock voice WAVs). Env: PLAYWRIGHT_CORE, CHROME_PATH (default
// /opt/google/chrome/chrome when present, else Playwright's Chromium).
'use strict';
const fs = require('fs'), path = require('path'), http = require('http');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const a = {src: 'buffer', aec: '1', sr: '24000', need: '', label: 'run', headful: '0'};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!a.out || !a.wavs) { console.error('usage: node run.cjs --app DIR --out DIR --label L --wavs a.wav,b.wav [...]'); process.exit(2); }
fs.mkdirSync(a.out, {recursive: true});
const wavs = a.wavs.split(',');
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(q.url.split('?')[0]);
  let f = null;
  if (u === '/h.html') f = path.join(__dirname, "h.html");
  else if (u === '/voice.js') f = path.join(a.app, 'voice.js');
  else if (u.startsWith('/w/')) f = wavs[+u.slice(3)];
  if (!f || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, {'content-type': f.endsWith('.html') ? 'text/html' : f.endsWith('.js') ? 'application/javascript' : 'audio/wav'});
  fs.createReadStream(f).pipe(r);
});
function wav(i16buf, sr) {
  const h = Buffer.alloc(44); h.write('RIFF', 0); h.writeUInt32LE(36 + i16buf.length, 4); h.write('WAVEfmt ', 8); h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(sr, 24); h.writeUInt32LE(sr * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(i16buf.length, 40); return Buffer.concat([h, i16buf]);
}
srv.listen(0, '127.0.0.1', async () => {
  const port = srv.address().port;
  const CHROME = process.env.CHROME_PATH || ['/opt/google/chrome/chrome'].find(p => fs.existsSync(p));
  const browser = await chromium.launch({executablePath: CHROME, headless: a.headful !== '1', env: a.sink ? {...process.env, PULSE_SINK: a.sink, PULSE_PROP: 'application.name=avatarbuzz media.role=test'} : process.env, ignoreDefaultArgs: ['--mute-audio'], args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--use-fake-ui-for-media-stream']});
  const page = await browser.newPage();
  const logs = []; page.on('console', m => logs.push(m.type() + ': ' + m.text().slice(0, 200))); page.on('pageerror', e => logs.push('pageerror: ' + e.message));
  await page.goto(`http://127.0.0.1:${port}/h.html`);
  const res = await page.evaluate(([n, o]) => window.run(Array.from({length: n}, (_, i) => '/w/' + i), o),
    [wavs.length, {src: a.src, aec: a.aec === '1', sr: a.sr, need: a.need || null, needJit: a.needjit, maxMs: a.maxms, gap: a.gap, stress: a.stress, stressEvery: a.every, hogMs: a.hog, hogEvery: a.hogevery}]);
  const meta = {label: a.label, src: a.src, aec: a.aec, sr: res.sr, starts: res.starts, stats: res.stats, loopRtp: res.loopRtp, vendRtp: res.vendRtp, wavs, taps: {}};
  for (const [k, t] of Object.entries(res.taps)) {
    const f = path.join(a.out, `${a.label}.${k}.wav`); fs.writeFileSync(f, wav(Buffer.from(t.b64, 'base64'), t.sr));
    meta.taps[k] = {file: f, sr: t.sr, n: t.n, over: t.over};
  }
  meta.logs = logs.slice(0, 30);
  fs.writeFileSync(path.join(a.out, a.label + '.json'), JSON.stringify(meta, null, 1));
  console.log(JSON.stringify({label: a.label, loopRtp: res.loopRtp, vendRtp: res.vendRtp}));
  await browser.close(); srv.close();
});
