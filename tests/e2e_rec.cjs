// End-to-end recording test (rec): the real page on a scratch server instance in headless Chrome with fake media
// devices, mock voice ($0); the page records the stage + both voices and uploads it; the server assembles the mp4 into
// $AVATAR_REC_ROOT or <repo>/recordings/avatar-sessions (variant from the scratch config — "test"). Optional mid-session device switches.
//   node tests/e2e_rec.cjs --url http://127.0.0.1:PORT/ --key K --out DIR [--cam clip.y4m --mic clip.wav] [--seconds 30]
//        [--char muse] [--switch 1 (needs no --cam: two fake cameras + fake mics)] [--root <recordings root>]
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process'), os = require('os');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const a = {seconds: 30, char: 'muse', mock: 'mock_01_hi.wav,mock_04_fact.wav,mock_02_food.wav', chrome: process.env.CHROME_PATH || undefined,
  root: process.env.AVATAR_REC_ROOT || path.join(__dirname, '..', 'recordings', 'avatar-sessions'), switch: '0'};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
for (const r of ['url', 'key', 'out']) if (!a[r]) throw new Error('--' + r + ' required');
fs.mkdirSync(a.out, {recursive: true});
const u = new URL(a.url);
for (const [k, v] of Object.entries({k: a.key, voice: 'mock', mock: a.mock, auto: '1', char: a.char, rec: '1', hud: '1', attention: '0'})) u.searchParams.set(k, v);
const log = (...x) => { const s = new Date().toISOString().slice(11, 19) + ' ' + x.join(' '); console.log(s); fs.appendFileSync(path.join(a.out, 'run.log'), s + '\n'); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const probe = f => JSON.parse(cp.execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,duration,sample_rate', '-of', 'json', f]).toString());

(async () => {
  const args = ['--no-sandbox', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required', '--mute-audio',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'];
  if (a.cam) args.push('--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${a.cam}`);
  else args.push('--use-fake-device-for-media-stream=device-count=2');
  if (a.mic) args.push(`--use-file-for-fake-audio-capture=${a.mic}`);
  const browser = await chromium.launch({executablePath: a.chrome, headless: true, args});
  const ctx = await browser.newContext({viewport: {width: 412, height: 915}, deviceScaleFactor: 2, permissions: ['camera', 'microphone']});
  const page = await ctx.newPage(), errors = [], consoleLines = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { const t = m.text(); if (['error', 'warning'].includes(m.type()) || /^rec:/.test(t)) consoleLines.push(m.type() + ': ' + t.slice(0, 300)); });
  const out = {args: a, errors, console: consoleLines, samples: [], switches: []};
  try {
    await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
    await page.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live, null, {timeout: 60000});
    await page.waitForFunction(() => { const r = window.__avatar.rec && window.__avatar.rec(); return r && r.cur && r.cur.rid; }, null, {timeout: 60000});
    const t0 = Date.now();
    const rec0 = await page.evaluate(() => window.__avatar.rec());
    log('recording', rec0.cur.rid, rec0.cur.mime, rec0.cur.w + 'x' + rec0.cur.h);
    await page.evaluate(() => { window.__v0 = window.__avatar.D.voice; window.__p0 = window.__avatar.D.perc; window.__nf = 0;
      const p = window.__avatar.D.perc; if (p && p._state) { const S = p._state; setInterval(() => { if (S.latest && S.latest !== window.__lastF) { window.__lastF = S.latest; window.__nf++; } }, 20); } });
    const sample = async tag => {
      const s = await page.evaluate(() => {
        const A = window.__avatar, D = A.D, r = A.rec(), ps = D.perc && D.perc.stats ? D.perc.stats() : null;
        return {t: Math.round(performance.now()), voice_same: D.voice === window.__v0, voice_state: D.voice && D.voice.state ? D.voice.state() : null,
          perc_same: D.perc === window.__p0, perc_running: !!(D.perc && D.perc.running), features: window.__nf,
          face_fps: ps && ps.models.face ? ps.models.face.fps : null, cam: ps ? ps.camera : null, devices: A.devices ? A.devices() : {},
          rec: r.cur ? {rid: r.cur.rid, part: r.cur.part, state: r.cur.state, frames: r.cur.frames, fps: r.cur.fps, sent: r.cur.up && r.cur.up.sent,
            queued: r.cur.up && r.cur.up.queued, retries: r.cur.up && r.cur.up.retries, draw_p50: r.cur.draw_ms_p50, draw_p95: r.cur.draw_ms_p95} : null,
          turns: D.voice && D.voice.stats ? D.voice.stats().turns.length : null, mic_lvl: D.voice && D.voice.userVoice ? D.voice.userVoice().rms : null};
      });
      s.tag = tag; s.at_s = Math.round((Date.now() - t0) / 100) / 10; out.samples.push(s);
      log(tag, JSON.stringify({voice: s.voice_state, same: s.voice_same && s.perc_same, feat: s.features, face_fps: s.face_fps, rec: s.rec, mic: s.devices.micTrack, cam: s.devices.camTrack}));
      return s;
    };
    const secs = +a.seconds;
    if (a.switch === '1') {
      const ids = await page.evaluate(async () => { const ds = await navigator.mediaDevices.enumerateDevices();
        return {mics: ds.filter(d => d.kind === 'audioinput' && d.deviceId !== 'default').map(d => [d.deviceId, d.label]),
          cams: ds.filter(d => d.kind === 'videoinput').map(d => [d.deviceId, d.label]),
          micOpts: [...document.querySelectorAll('#micsel option')].map(o => o.textContent), camOpts: [...document.querySelectorAll('#camsel option')].map(o => o.textContent)}; });
      out.devices = ids; log('devices', JSON.stringify(ids));
      await sleep(secs / 3 * 1000);
      const before = await sample('before-mic-switch');
      const m = ids.mics.find(x => x[1] !== before.devices.micTrack) || ids.mics[ids.mics.length - 1];
      const t1 = Date.now(); await page.evaluate(id => window.__avatar.switchMic(id), m[0]);
      out.switches.push({kind: 'mic', to: m[1], ms: Date.now() - t1});
      await sleep(3000);
      await sample('after-mic-switch');
      await sleep(Math.max(0, secs / 3 * 1000 - 3000));
      const bc = await sample('before-cam-switch');
      const c = ids.cams.find(x => x[1] !== bc.devices.camTrack) || ids.cams[ids.cams.length - 1];
      const t2 = Date.now(); await page.evaluate(id => window.__avatar.switchCam(id), c[0]);
      out.switches.push({kind: 'cam', to: c[1], ms: Date.now() - t2});
      await sleep(3000);
      await sample('after-cam-switch');
      await sleep(Math.max(0, secs / 3 * 1000 - 3000));
    } else {
      for (let k = 0; k < Math.ceil(secs / 10); k++) { await sleep(Math.min(10, secs - 10 * k) * 1000); await sample('t' + (k + 1)); }
    }
    await page.screenshot({path: path.join(a.out, 'page.png')});
    const last = await sample('before-stop');
    const recStats = await page.evaluate(() => window.__avatar.rec());
    out.rec_before_stop = recStats;
    const tStop = Date.now();
    await page.click('#go');
    log('stopped after', ((tStop - t0) / 1000).toFixed(1), 's of recording');
    // the page drains its last chunk, posts 'end'; the hub assembles
    const rid = last.rec.rid;
    let rec = null;
    for (let i = 0; i < 180 && !rec; i++) {
      await sleep(1000);
      try { const idx = JSON.parse(fs.readFileSync(path.join(a.root, 'index.json'))); rec = idx.sessions.find(s => s.rid === rid) || null; } catch (e) {}
    }
    out.assembly_wait_s = Math.round((Date.now() - tStop) / 1000);
    out.after = await page.evaluate(() => window.__avatar.rec());
    if (!rec) { log('NOT assembled within 180 s'); out.ok = false; }
    else {
      const mp4 = path.join(a.root, rec.file);
      const pr = probe(mp4);
      const vol = cp.spawnSync('ffmpeg', ['-hide_banner', '-nostats', '-i', mp4, '-af', 'volumedetect', '-vn', '-f', 'null', '-'], {encoding: 'utf8'}).stderr;
      const mv = /mean_volume: (-?[\d.]+)/.exec(vol), xv = /max_volume: (-?[\d.]+)/.exec(vol);
      cp.spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-ss', String(Math.max(1, (rec.duration_s || 10) * 0.6)), '-i', mp4, '-frames:v', '1', path.join(a.out, 'frame.png')]);
      const recorded_s = (tStop - t0) / 1000;
      out.record = rec; out.ffprobe = pr; out.mean_volume_db = mv ? +mv[1] : null; out.max_volume_db = xv ? +xv[1] : null; out.recorded_s = recorded_s;
      const v = pr.streams.find(s => s.codec_type === 'video'), au = pr.streams.find(s => s.codec_type === 'audio');
      // the recording starts once the voice is up (before t0 = the rid was known) and ends at Stop
      out.ok = !!(v && au && Math.abs(+pr.format.duration - recorded_s) < 4 && out.mean_volume_db > -60 && !errors.length);
      log('assembled', rec.file, 'dur', pr.format.duration, 'vs', recorded_s.toFixed(1), v && v.codec_name, v && v.width + 'x' + v.height, v && v.avg_frame_rate,
        au && au.codec_name, 'vol', out.mean_volume_db, 'dB; chunks', rec.chunks + '/' + rec.chunks_expected, 'missing', rec.missing.length, 'wait', out.assembly_wait_s + 's');
    }
  } catch (e) {
    out.ok = false; out.error = String(e && e.stack || e).slice(0, 600); log('ERROR', out.error);
    try { await page.screenshot({path: path.join(a.out, 'error.png')}); } catch (x) {}
  }
  fs.writeFileSync(path.join(a.out, 'result.json'), JSON.stringify(out, null, 1));
  log('ok =', out.ok, 'errors', errors.length);
  await browser.close();
  process.exit(out.ok ? 0 : 1);
})();
