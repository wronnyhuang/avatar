// joy: the feeling state's held look per (v, a, d, intensity, facets) point, silent, on a dev server instance (tools/devinst.sh) (?voice=mock):
// per pose the state is set directly (no judge) and the stage is shot at the given times after onset.
//   node tests/affect_poses.cjs --url http://127.0.0.1:PORT/ --key K --poses poses.json --out DIR [--at 0.4,1.2,3] [--only id,id]
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || process.env.CHROME || undefined;   // undefined = playwright's own chromium
const a = {q: '540p', w: 960, h: 540, at: '0.5,1.5,3.5', gapS: 3};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
fs.mkdirSync(a.out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
let poses = JSON.parse(fs.readFileSync(a.poses, 'utf8'));
if (a.only) { const o = a.only.split(','); poses = poses.filter(s => o.includes(s.id)); }
(async () => {
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader']});
  const ctx = await browser.newContext({viewport: {width: +a.w, height: +a.h + 160}, permissions: ['camera', 'microphone']});
  await ctx.addInitScript(q => { try { localStorage.setItem('avatar.q', q); } catch (e) {} }, a.q);
  const page = await ctx.newPage();
  const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', 'mock'); u.searchParams.set('char', 'muse'); u.searchParams.set('camera', '0');
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await sleep(1500);
  await page.click('#go', {timeout: 15000});
  await page.waitForFunction(() => window.__avatar && window.__avatar.D && window.__avatar.D.voice && window.__avatar.S.live, null, {timeout: 240000});
  for (let i = 0; i < 240; i++) { const fps = await page.evaluate(() => (window.__avatar.S.lastMsg || {}).fps || 0).catch(() => 0); if (fps >= 3) break; await sleep(500); }
  await sleep(3000);
  const stage = await page.$('#stage');
  const ats = a.at.split(',').map(Number);
  for (const p of poses) {
    await page.evaluate(() => { const M = window.__avatar.D.motion; if (M && M.affectReset) M.affectReset(); });
    await sleep(+a.gapS * 1000);
    const t0 = Date.now();
    if (!p.none) await page.evaluate(j => window.__avatar.D.motion.affect(Object.assign({hold_s: 14, label: j.id}, j)), p);
    for (const s of ats) {
      const w = s * 1000 - (Date.now() - t0); if (w > 0) await sleep(w);
      await stage.screenshot({path: path.join(a.out, `${p.id}_${s}.png`)});
      const rig = await page.evaluate(() => { const L = window.__avatar.D.lastRig || {}; const r = {}; for (const k of ['smile', 'brow_knit', 'brow_l', 'eye_wide', 'squint', 'body_pitch', 'hand_l_y']) r[k] = L[k] === undefined ? null : +(+L[k]).toFixed(3); return r; });
      console.log(p.id, s, JSON.stringify(rig));
    }
    console.log(p.id);
  }
  await page.click('#go', {timeout: 5000}).catch(() => {});
  await sleep(1000); await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
