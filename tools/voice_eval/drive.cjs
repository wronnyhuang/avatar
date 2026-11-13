// node drive.cjs <url> <micwav> <seconds> <out.json>  (headed, on $DISPLAY with $PULSE_SERVER)
// Env: PLAYWRIGHT_CORE (module path, default 'playwright-core'), CHROME_PATH (browser binary, default Playwright's)
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
(async () => {
  const [url, mic, secs, out] = process.argv.slice(2);
  const b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: false,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-audio-capture=' + mic + '%noloop',
      '--autoplay-policy=no-user-gesture-required', '--no-sandbox', '--window-position=0,0', '--window-size=800,1000', '--no-first-run']});
  const ctx = await b.newContext({permissions: ['microphone', 'camera'], viewport: {width: 780, height: 860}});
  await ctx.addInitScript(() => {   // a 'desynchronized' 2-D canvas is never composited into Xvfb screen grabs (eval feed_init.js)
    const g = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, opts) {
      if (opts && typeof opts === 'object' && opts.desynchronized) { opts = Object.assign({}, opts); delete opts.desynchronized; }
      return g.call(this, type, opts); };
  });
  const p = await ctx.newPage();
  const logs = []; p.on('console', m => logs.push(m.type() + ': ' + m.text())); p.on('pageerror', e => logs.push('pageerror ' + e.message));
  await p.goto(url.replace('&auto=1', ''));
  // A/V capture calibration (same idea as the eval's feed_init.js): a tone on its own AudioContext and a white
  // marker square flipped at the tone's scheduled output time; the recorder finds both
  await p.evaluate(async () => {
    const m = document.createElement('div'); m.style.cssText = 'position:fixed;left:0;bottom:0;width:40px;height:40px;z-index:2147483647;background:#000';
    document.body.appendChild(m);
    const ac = new AudioContext(); await ac.resume();
    const o = ac.createOscillator(), g = ac.createGain(); o.frequency.value = 1000; g.gain.value = 0.5; o.connect(g); g.connect(ac.destination);
    await new Promise(r => setTimeout(r, 500));
    const at = ac.currentTime + 0.2; o.start(at); o.stop(at + 0.15);
    const ts = ac.getOutputTimestamp(); const heard = ts.performanceTime + (at - ts.contextTime) * 1000;
    await new Promise(r => setTimeout(r, Math.max(0, heard - performance.now())));
    m.style.background = '#fff'; await new Promise(r => setTimeout(r, 150)); m.style.background = '#000';
    await new Promise(r => setTimeout(r, 400)); ac.close();
  });
  if (url.includes('&auto=1')) await p.click('#go');
  await p.waitForTimeout(+secs * 1000);
  const rep = await p.evaluate(() => ({r: window.__avatar.voiceReport(), flashes: window.__avatar.D.flashes || [], st: window.__avatar.stats()})).catch(e => ({err: e.message}));
  require('fs').writeFileSync(out, JSON.stringify({rep, logs}, null, 1));
  await p.evaluate(() => window.__avatar.stop()).catch(() => {});
  await p.waitForTimeout(1000); await b.close();
})();
