// end-to-end: the real page on an isolated app server instance (CPU renderer), headless Chromium, fake camera clip +
// fake mic wav, mock voice ($0), ?react=learned|rules. Records the page (video), screenshots, every drawn frame's
// timeline stamp + draw time, the page's rig vectors with the perception frame they used, the canon frames sent ('H').
//   node tests/e2e_react.cjs --url http://127.0.0.1:PORT/ --key K --cam clip.y4m --mic clip.wav --out DIR
//        --home <the server's AVATAR_HOME: mock wavs go to <home>/mock> [--mock-src <dir of mock wavs>]
//        [--react learned|rules] [--seconds 30] [--char fox] [--mock mock_01_hi.wav,...] [--shots 6]
// ?react=learned needs a server configured for learned mode (react.root = a learned-reaction checkout); else use rules.
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || undefined;   // undefined = playwright's own chromium
const a = {seconds: 30, inst: 'e2e', home: process.env.AVATAR_HOME || '', 'mock-src': process.env.AVATAR_MOCK_WAVS || path.join(__dirname, '..', 'data', 'mock_voice'), mock: 'mock_01_hi.wav,mock_04_fact.wav,mock_02_food.wav', char: 'fox', react: 'learned', shots: 6, extra: ''};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
for (const r of ['url', 'key', 'cam', 'mic', 'out', 'home']) if (!a[r]) throw new Error('--' + r + ' required');
fs.mkdirSync(a.out, {recursive: true});
const mockDir = path.join(a.home, 'mock'); fs.mkdirSync(mockDir, {recursive: true});
const names = a.mock.split(',').filter(Boolean);
for (const n of names) if (!fs.existsSync(path.join(mockDir, n))) fs.copyFileSync(path.join(a['mock-src'], n), path.join(mockDir, n));   // read-only source
const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', 'mock'); u.searchParams.set('mock', names.join(','));
u.searchParams.set('auto', '1'); u.searchParams.set('char', a.char); u.searchParams.set('hud', '1'); u.searchParams.set('react', a.react);
for (const kv of a.extra.split(',').filter(Boolean)) { const [k, v] = kv.split('='); u.searchParams.set(k, v); }

(async () => {
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${a.cam}`,
    `--use-file-for-fake-audio-capture=${a.mic}`, '--mute-audio', '--autoplay-policy=no-user-gesture-required',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows']});
  const ctx = await browser.newContext({viewport: {width: 412, height: 915}, permissions: ['camera', 'microphone']});
  const page = await ctx.newPage(), errors = [], consoleLines = [];
  // the avatar canvas, ~15 fps, as JPEGs with their draw time (playwright's own video needs an ffmpeg build we lack)
  const fdir = path.join(a.out, 'frames'); fs.mkdirSync(fdir, {recursive: true});
  const frames = [];
  await page.exposeFunction('__saveFrame', (i, perf, tl, dataUrl) => {
    fs.writeFileSync(path.join(fdir, `f${String(i).padStart(5, '0')}.jpg`), Buffer.from(dataUrl.split(',')[1], 'base64'));
    frames.push({i, perf, tl});
  });
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { if (['error', 'warning'].includes(m.type()) || /react/.test(m.text())) consoleLines.push(m.type() + ': ' + m.text().slice(0, 300)); });
  const t0 = Date.now();
  const say = m => console.error(`[e2e ${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
  say('goto');
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 30000});
  say('loaded');
  await page.waitForTimeout(1500);
  const live = await page.evaluate(() => !!(window.__avatar && window.__avatar.S && window.__avatar.S.live));
  if (!live) { try { await page.click('#go', {timeout: 3000}); } catch (e) {} }
  await page.waitForFunction(() => window.__avatar && window.__avatar.D && window.__avatar.D.motion && window.__avatar.D.motion.step, null, {timeout: 180000});
  say('motion up');
  await page.evaluate(() => {
    const {S, D} = window.__avatar, m = D.motion, orig = m.step.bind(m);
    window.__rig = []; window.__draw = []; window.__map = []; window.__canon = [];
    // uplink composition: binary messages by type/flags (+ sizes), control JSON by type
    window.__up = {};
    const qpush = S.upQ.push.bind(S.upQ);
    S.upQ.push = (...ms) => { for (const m of ms) { const k = m && m[0] !== undefined ? String.fromCharCode(m[0]) + (m[0] === 0x52 ? ':' + m[1] : '') : 'other'; const e = window.__up[k] || (window.__up[k] = {n: 0, bytes: 0}); e.n++; e.bytes += m.length || 0; } return qpush(...ms); };
    if (S.ctlQ) { const cpush = S.ctlQ.push.bind(S.ctlQ); S.ctlQ.push = (...ms) => { for (const m of ms) { let t = 'json'; try { t = 'J:' + (typeof m === 'string' ? JSON.parse(m).type : m.type); } catch (e) {} const e = window.__up[t] || (window.__up[t] = {n: 0, bytes: 0}); e.n++; e.bytes += typeof m === 'string' ? m.length : JSON.stringify(m).length; } return cpush(...ms); }; }
    m.step = (dt, inp) => {
      const o = orig(dt, inp), f = inp.features;
      const rt = D.play && D.play.startCt !== null ? S.ctx.currentTime - D.play.startCt : null;
      window.__rig.push({perf: performance.now(), rt, ft: f ? f.t : null, a: m.vector(o), lp: D.learned ? m.learnedParts(m.params) : null});
      const c = f && f.canon;
      if (c && c.seq !== window.__lastCanon) { window.__lastCanon = c.seq; window.__canon.push({seq: c.seq, t: c.t, mask: Array.from(c.mask), smile: (c.motion[44] + c.motion[45]) / 2, yaw: c.motion[52], pitch: c.motion[53], roll: c.motion[54]}); }
      return o;
    };
    const loop = () => {
      if (S.drawnT !== undefined && S.drawnT !== null && S.drawnT !== window.__lastDrawn) { window.__lastDrawn = S.drawnT; window.__draw.push({t: S.drawnT, perf: S.lastDrawPerf || performance.now()}); }
      if (D.play && D.play.startCt !== null && window.__map.length < 4000 && (window.__map.length === 0 || performance.now() - window.__map[window.__map.length - 1][0] > 500))
        window.__map.push([performance.now(), S.ctx.currentTime - D.play.startCt]);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
    let fi = 0;
    setInterval(() => {
      const cv = document.querySelector('canvas#cv') || document.querySelector('canvas');
      if (!cv || !S.drawn) return;
      try { window.__saveFrame(fi++, performance.now(), S.drawnT, cv.toDataURL('image/jpeg', 0.85)); } catch (e) {}
    }, 66);
  });
  const tStart = Date.now(), nShots = +a.shots;
  for (let i = 0; i < nShots; i++) {
    await page.waitForTimeout(+a.seconds * 1000 / nShots);
    try { await page.screenshot({path: path.join(a.out, `shot_${String(i).padStart(2, '0')}.png`)}); } catch (e) {}
  }
  say('dumping');
  const dump = await page.evaluate(() => {
    const {S, D} = window.__avatar, cv = document.querySelector('canvas#cv') || document.querySelector('canvas');
    const r = cv ? cv.getBoundingClientRect() : null;
    return {up: window.__up, rig: window.__rig, draw: window.__draw, map: window.__map, canon: window.__canon, events: D.events, learned: !!D.learned, rx: D.rx || null,
      rig_params: (S.cfg || {}).rig_params || null, motion_params: D.motion.params, canvas: r ? [r.left, r.top, r.width, r.height] : null,
      perc: D.perc && D.perc.stats ? D.perc.stats() : null, delay: S.D, react_mode: (S.cfg || {}).react_mode};
  });
  let stats = null;
  try {
    const su = new URL('api/stats', a.url); su.searchParams.set('k', a.key);
    stats = await page.evaluate(async (s) => (await fetch(s, {headers: {'x-requested-with': 'avatar-live'}})).json(), su.toString());
  } catch (e) { stats = {error: String(e)}; }
  dump.frames = frames;
  fs.writeFileSync(path.join(a.out, 'dump.json'), JSON.stringify(dump));
  fs.writeFileSync(path.join(a.out, 'hub_stats.json'), JSON.stringify(stats && stats.live, null, 1));
  const sum = {react: a.react, uplink: dump.up, learned_on_page: dump.learned, rx: dump.rx, rig_rows: dump.rig.length, drawn: dump.draw.length,
    seconds: (Date.now() - tStart) / 1000, load_to_motion_s: (tStart - t0) / 1000, canon_frames: dump.canon.length,
    hub_react: stats && stats.live ? stats.live.react : null, fps_shown: stats && stats.live ? stats.live.fps_shown : null,
    delay_s: dump.delay, errors, console: consoleLines.slice(0, 30), canvas: dump.canvas};
  fs.writeFileSync(path.join(a.out, 'summary.json'), JSON.stringify(sum, null, 1));
  console.log(JSON.stringify(sum, null, 1).slice(0, 3000));
  try { await page.click('#go', {timeout: 2000}); } catch (e) {}
  await page.waitForTimeout(800);
  await ctx.close();
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
