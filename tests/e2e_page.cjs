// End-to-end: the real page (hub instance) in headless Chromium with a fake camera clip + fake mic wav, mock voice ($0).
// Logs every rig vector the page's 30 Hz timer produces (motion.step wrapped), the perception features and events.
//   node tests/e2e_page.cjs --url http://127.0.0.1:18921/ --key <k> --cam clip.y4m --mic clip.wav --out dir [--seconds 35]
//        [--mock mock_01_hi.wav,...] [--home <the hub's AVATAR_HOME: mock wavs go to <home>/mock>] [--mock-src <dir of mock wavs>]
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || undefined;   // undefined = playwright's own chromium
const a = {seconds: 35, inst: 'motion', home: process.env.AVATAR_HOME || '', 'mock-src': process.env.AVATAR_MOCK_WAVS || path.join(__dirname, '..', 'data', 'mock_voice'), mock: 'mock_01_hi.wav,mock_04_fact.wav,mock_02_food.wav', char: 'fox'};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
for (const r of ['url', 'key', 'cam', 'mic', 'out', 'home']) if (!a[r]) throw new Error('--' + r + ' required');
fs.mkdirSync(a.out, {recursive: true});
const mockDir = path.join(a.home, 'mock'); fs.mkdirSync(mockDir, {recursive: true});
const names = a.mock.split(',').filter(Boolean);
for (const n of names) fs.copyFileSync(path.join(a['mock-src'], n), path.join(mockDir, n));
const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', 'mock'); u.searchParams.set('mock', names.join(','));
u.searchParams.set('auto', '1'); u.searchParams.set('char', a.char); u.searchParams.set('hud', '1');

(async () => {
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${a.cam}`,
    `--use-file-for-fake-audio-capture=${a.mic}%noloop`, '--mute-audio', '--autoplay-policy=no-user-gesture-required',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows']});
  const ctx = await browser.newContext({viewport: {width: 412, height: 915}, permissions: ['camera', 'microphone']});
  const page = await ctx.newPage(), errors = [], consoleLines = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) consoleLines.push(m.type() + ': ' + m.text().slice(0, 300)); });
  const t0 = Date.now();
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 30000});
  // the page auto-starts with auto=1; press Start if it did not
  await page.waitForTimeout(1500);
  const live = await page.evaluate(() => !!(window.__avatar && window.__avatar.S && window.__avatar.S.live));
  if (!live) { try { await page.click('#go', {timeout: 3000}); } catch (e) {} }
  await page.waitForFunction(() => window.__avatar && window.__avatar.D && window.__avatar.D.motion && window.__avatar.D.motion.step, null, {timeout: 120000});
  await page.evaluate(() => {
    const D = window.__avatar.D, m = D.motion, orig = m.step.bind(m);
    window.__rig = []; window.__feat = []; window.__dbg = [];
    m.step = (dt, inp) => {
      const o = orig(dt, inp), d = m.debug(), t = performance.now() / 1000;
      window.__rig.push({type: 'rig', t: +t.toFixed(4), a: m.vector(o)});
      window.__dbg.push({t: +t.toFixed(3), mode: d.mode, spk: d.speaking, lis: d.listening, bored: +d.bored.toFixed(3), avr: inp.avatarVoice ? +(+inp.avatarVoice.rms || 0).toFixed(4) : null,
        ur: inp.userVoice ? +(+inp.userVoice.rms || 0).toFixed(4) : null, g: d.gestures.join(','), smile: +d.expr.smile.toFixed(3)});
      const f = inp.features;
      if (f && f !== window.__lastF) {
        window.__lastF = f;
        const fc = f.face || {};
        window.__feat.push({t: f.t, face: fc.present ? {x: fc.x, y: fc.y, ax: fc.ax, ay: fc.ay, z: fc.z, yaw: fc.yaw, pitch: fc.pitch, roll: fc.roll,
          gazeX: fc.gazeX, gazeY: fc.gazeY, lookX: fc.lookX, lookY: fc.lookY, expr: fc.expr, xp: fc.xp} : null,
          hands: (f.hands || []).map(h => ({side: h.side, handed: h.handed, x: h.x, y: h.y, gesture: h.gesture, open: h.open, scale: h.scale, vscale: h.vscale})),
          motion: f.motion, approach: f.approach, attention: f.attention, attending: f.attending, events: (f.events || []).map(e => e.type)});
      }
      return o;
    };
  });
  const tStart = Date.now();
  await page.waitForTimeout(+a.seconds * 1000);
  const dump = await page.evaluate(() => {
    const D = window.__avatar.D;
    return {rig: window.__rig, feat: window.__feat, dbg: window.__dbg, events: D.events, cmds: D.cmds,
      perc: D.perc && D.perc.stats ? D.perc.stats() : null, rig_params: (window.AvatarMotionLib || {}).RIG_PARAMS,
      cfg_rig_params: (window.__avatar.S.cfg || {}).rig_params || null, voice: window.__avatar.voiceReport ? window.__avatar.voiceReport() : null};
  });
  await page.screenshot({path: path.join(a.out, 'screenshot.png')});
  const rt0 = dump.rig.length ? dump.rig[0].t : 0;   // timeline from 0 (render_timeline.py input)
  fs.writeFileSync(path.join(a.out, 'rig.jsonl'), JSON.stringify({rig_params: dump.rig_params}) + '\n' + dump.rig.map(r => JSON.stringify({type: 'rig', t: +(r.t - rt0).toFixed(4), a: r.a})).join('\n') + '\n');
  fs.writeFileSync(path.join(a.out, 'events.json'), JSON.stringify({events: dump.events, cmds: dump.cmds, rig_t0_perf_s: rt0}, null, 1));
  fs.writeFileSync(path.join(a.out, 'features.jsonl'), dump.feat.map(r => JSON.stringify(r)).join('\n') + '\n');
  fs.writeFileSync(path.join(a.out, 'debug.jsonl'), dump.dbg.map(r => JSON.stringify(r)).join('\n') + '\n');
  // summary
  const P = dump.rig_params || [], idx = n => P.indexOf(n), col = n => dump.rig.map(r => r.a[idx(n)]);
  const std = x => { if (!x.length) return 0; const m = x.reduce((p, q) => p + q, 0) / x.length; return Math.sqrt(x.reduce((p, q) => p + (q - m) ** 2, 0) / x.length); };
  const faces = dump.feat.filter(f => f.face), ev = {};
  for (const e of dump.events || []) ev[e.name] = (ev[e.name] || 0) + 1;
  const modes = {}; for (const d of dump.dbg) modes[d.mode] = (modes[d.mode] || 0) + 1;
  const span = dump.rig.length > 1 ? dump.rig[dump.rig.length - 1].t - dump.rig[0].t : 0;
  const summary = {url_inst: a.inst, cam: path.basename(a.cam), mic: path.basename(a.mic), seconds: +a.seconds, load_to_motion_s: (tStart - t0) / 1000,
    rig_rows: dump.rig.length, rig_hz: span ? +(dump.rig.length / span).toFixed(1) : 0, hub_rig_params: dump.cfg_rig_params ? dump.cfg_rig_params.length : null,
    feat_rows: dump.feat.length, feat_hz: dump.feat.length > 1 ? +(dump.feat.length / (dump.feat[dump.feat.length - 1].t - dump.feat[0].t)).toFixed(1) : 0,
    face_ratio: dump.feat.length ? +(faces.length / dump.feat.length).toFixed(2) : 0,
    face_x: faces.length ? [Math.min(...faces.map(f => f.face.x)).toFixed(2), Math.max(...faces.map(f => f.face.x)).toFixed(2)] : null,
    attending_ratio: dump.feat.length ? +(dump.feat.filter(f => f.attending).length / dump.feat.length).toFixed(2) : 0,
    events: ev, modes, speaking_frames: dump.dbg.filter(d => d.spk).length, listening_frames: dump.dbg.filter(d => d.lis).length,
    std: Object.fromEntries(['head_yaw', 'head_pitch', 'head_roll', 'gaze_x', 'gaze_y', 'smile', 'body_yaw', 'hand_l_y', 'hand_r_y', 'blink_l'].map(n => [n, +std(col(n)).toFixed(3)])),
    perception: dump.perc, errors, console: consoleLines.slice(0, 20)};
  fs.writeFileSync(path.join(a.out, 'summary.json'), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(Object.assign({}, summary, {perception: summary.perception && summary.perception.models}), null, 1));
  try { await page.click('#go', {timeout: 2000}); } catch (e) {}
  await page.waitForTimeout(800);
  await browser.close();
})().catch(e => { console.error(e); process.exitCode = 1; });
