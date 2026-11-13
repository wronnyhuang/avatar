// node drive_life.cjs <url> <micwav> <seconds> <out.json> <plan: "t:action[:arg],...">  actions: drop, offline:<s>, freeze:<s>, hide:<s>, event:<name>
// Env: PLAYWRIGHT_CORE, CHROME_PATH, VOICE_EVAL_CAM (optional fake-camera y4m; default = Chrome's built-in pattern)
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
(async () => {
  const [url, mic, secs, out, plan] = process.argv.slice(2);
  const b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-audio-capture=' + mic,
      ...(process.env.VOICE_EVAL_CAM ? ['--use-file-for-fake-video-capture=' + process.env.VOICE_EVAL_CAM] : []),
      '--autoplay-policy=no-user-gesture-required', '--no-sandbox', '--mute-audio']});
  const ctx = await b.newContext({permissions: ['microphone', 'camera']});
  const p = await ctx.newPage();
  const cdp = await ctx.newCDPSession(p);
  const logs = []; p.on('console', m => { if (/error|warn/i.test(m.type())) logs.push(m.type() + ': ' + m.text().slice(0, 200)); });
  p.on('pageerror', e => logs.push('pageerror ' + e.message));
  await p.goto(url);
  const steps = (plan || '').split(',').filter(Boolean).map(x => { const [t, a, ...rest] = x.split(':'); const raw = rest.join(':'); return {t: +t, a, arg: +raw || 0, raw, done: false}; });
  const t0 = Date.now(), log = [], snap = async () => p.evaluate(() => ({life: __avatar.life(), turns: (__avatar.voiceReport().turns || []).length,
    drawn: __avatar.S.drawn, state: __avatar.D.voice ? __avatar.D.voice.state() : null, status: document.getElementById('status').textContent.slice(0, 80)})).catch(e => ({err: e.message}));
  while (Date.now() - t0 < +secs * 1000) {
    await new Promise(r => setTimeout(r, 1000));
    const el = (Date.now() - t0) / 1000;
    for (const s of steps) if (!s.done && el >= s.t) {
      s.done = true; const at = Math.round(el);
      if (s.a === 'drop') log.push({at, a: 'drop', r: await p.evaluate(() => __avatar.testDrop()).catch(e => e.message)});
      else if (s.a === 'offline') { await ctx.setOffline(true); log.push({at, a: 'offline', s: s.arg}); await new Promise(r => setTimeout(r, s.arg * 1000)); await ctx.setOffline(false); }
      else if (s.a === 'freeze') { await cdp.send('Page.setWebLifecycleState', {state: 'frozen'}); log.push({at, a: 'freeze', s: s.arg}); await new Promise(r => setTimeout(r, s.arg * 1000)); await cdp.send('Page.setWebLifecycleState', {state: 'active'}); }
      else if (s.a === 'hide') {
        await p.evaluate(() => { Object.defineProperty(document, 'visibilityState', {value: 'hidden', configurable: true}); Object.defineProperty(document, 'hidden', {value: true, configurable: true}); document.dispatchEvent(new Event('visibilitychange')); });
        log.push({at, a: 'hide', s: s.arg}); await new Promise(r => setTimeout(r, s.arg * 1000));
        await p.evaluate(() => { Object.defineProperty(document, 'visibilityState', {value: 'visible', configurable: true}); Object.defineProperty(document, 'hidden', {value: false, configurable: true}); document.dispatchEvent(new Event('visibilitychange')); });
      } else if (s.a === 'event') log.push({at, a: 'event', name: s.raw, r: await p.evaluate(n => __avatar.event(n, {face: true, faces: 1}), s.raw || 'wave').catch(e => e.message)});
      else if (s.a === 'ctx') log.push({at, a: 'ctx', r: await p.evaluate(t => __avatar.context(t), s.raw.replace(/_/g, ' ')).catch(e => e.message)});
      log.push({at, after: s.a, snap: await snap()});
    }
    if (Math.round(el) % 30 === 0) { const sn = await snap(); log.push({at: Math.round(el), snap: sn}); console.log(Math.round(el) + 's', JSON.stringify({sid: sn.life && sn.life.sid && sn.life.sid.slice(0, 6), live: sn.life && sn.life.live, rec: sn.life && sn.life.recoveries.length, rst: sn.life && sn.life.restarts.length, turns: sn.turns, drawn: sn.drawn, st: sn.state, status: sn.status})); }
  }
  const fin = await snap(); const rep = await p.evaluate(() => __avatar.voiceReport()).catch(e => ({err: e.message}));
  require('fs').writeFileSync(out, JSON.stringify({plan, log, fin, rep, logs: logs.slice(-50)}, null, 1));
  await b.close();
})();
