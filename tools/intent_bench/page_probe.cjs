#!/usr/bin/env node
// Page-hook probe: headless Chromium, fake mic = a WAV of spoken commands, voice=mock (free). Verifies the page
// uploads 'M' to the server, the server's matcher answers with {"type":"intent"}, and runCommand(cmd,'matcher') acts.
//   node page_probe.cjs <url-with-k> <mic.wav> <manifest.json> <seconds>
// Env: PLAYWRIGHT_CORE (module path, default 'playwright-core'), CHROME_PATH (browser binary, default Playwright's own),
//      PROBE_CAM (a .y4m clip for the fake camera; default: Chromium's synthetic test pattern).
'use strict';
const fs = require('fs');
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || undefined;
const CAM = process.env.PROBE_CAM || '';
(async () => {
  const [u, mic, manPath, secs] = process.argv.slice(2);
  const url = new URL(u); url.searchParams.set('voice', 'mock'); url.searchParams.set('auto', '1'); url.searchParams.set('char', 'fox');
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${mic}%noloop`, ...(CAM ? [`--use-file-for-fake-video-capture=${CAM}`] : []),
    '--autoplay-policy=no-user-gesture-required', '--mute-audio'] });
  const ctx = await browser.newContext();
  await ctx.grantPermissions(['microphone', 'camera'], { origin: url.origin });
  await ctx.addInitScript(() => {
    const g = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => { const s = await g(c); if (c && c.audio && !window.__gum) window.__gum = performance.now(); return s; };
  });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(url.toString(), { waitUntil: 'load' });
  // instrument: page time at which each uploaded mic chunk (seq) was captured, to split the latency
  for (let i = 0; i < 40; i++) {
    const ok = await page.evaluate(() => {
      const A = window.__avatar; if (!A || !A.S.node) return false;
      if (window.__wrapped) return true;
      const port = A.S.node.port, orig = port.onmessage; window.__seqT = [];
      port.onmessage = (e) => { const before = A.S.micSeq; orig(e); if (A.S.micSeq > before) window.__seqT[before] = performance.now(); };
      window.__wrapped = true; return true;
    });
    if (ok) break;
    await page.waitForTimeout(250);
  }
  await page.waitForTimeout(Number(secs) * 1000);
  const split = await page.evaluate(() => (window.__avatar.D.cmds.filter(c => c.intent_rx)).map(c => ({rx: c.intent_rx, text: c.text})));
  const seqT = await page.evaluate(() => window.__seqT || []);
  const r = await page.evaluate(() => ({ gum: window.__gum, cfg: window.__avatar.S.cfg && window.__avatar.S.cfg.intent_mic,
    stage: window.__avatar.S.stage, micSeq: window.__avatar.S.micSeq, cmds: window.__avatar.D.cmds }));
  await page.evaluate(() => window.__avatar.stop && window.__avatar.stop()).catch(() => {});
  await page.waitForTimeout(800);
  await browser.close();
  const man = JSON.parse(fs.readFileSync(manPath, 'utf8'));
  const rx = r.cmds.filter((c) => c.intent_rx);
  // server audio position of each intent -> the page time that audio was captured -> transport+ASR share
  const xs = [];
  for (const c of rx) {
    const ent = r.cmds.find((d) => d.source === 'matcher' && Math.abs(d.t - c.intent_rx) < 2);
    const a = c.audio_s;
    if (a !== undefined && seqT[Math.floor(a * 50) - 1]) xs.push(Math.round(c.intent_rx - seqT[Math.floor(a * 50) - 1]));
  }
  xs.sort((a, b) => a - b);
  console.log('capture of the decisive audio -> intent at page (ms): ' + JSON.stringify(xs));
  const ran = r.cmds.filter((c) => c.source === 'matcher' && !c.dup);
  const lats = [];
  for (const it of man) {
    const end = r.gum + it.end_s * 1000;
    const hit = r.cmds.find((c) => c.source === 'matcher' && c.t >= r.gum + it.start_s * 1000 && c.t < end + 2000);
    const lat = hit ? Math.round(hit.t - end) : null;
    if (lat !== null) lats.push(lat);
    console.log(JSON.stringify(it.text).padEnd(40), hit ? `${hit.cmd.action}/${hit.cmd.name || hit.cmd.target || ''}/${hit.cmd.side || ''} ${hit.dup ? 'DUP' : ''}` : 'MISS', lat === null ? '' : `${lat} ms after speech end (page clock)`);
  }
  lats.sort((a, b) => a - b);
  console.log(JSON.stringify({ intent_mic: r.cfg, stage: r.stage, mic_chunks_sent: r.micSeq, intents_received: rx.length, commands_run: ran.length,
    lat_p50: lats[Math.floor(lats.length / 2)], lat_max: lats[lats.length - 1], page_errors: errs.slice(0, 3) }));
})();
