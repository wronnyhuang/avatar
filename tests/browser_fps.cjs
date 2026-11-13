// the fps the BROWSER actually shows.  Opens the avatar page through its link (public or local) in headless
// Chromium, starts a session with the mock voice ($0) and a fake camera / mic, runs a scripted 3-5 minutes (talking,
// full-body moves, the desk scene) and records the time of EVERY frame the page puts on its canvas (the page's own
// draw loop: one entry per new video frame drawn).  Reports, overall and per segment: shown fps, the frame-interval
// distribution, stutters (an interval over 1.5 frame periods: a picture held twice) and freezes (over 250 ms), plus what
// the page and the render server say about themselves (render ms / fps, late, dropped, the quality tier trail is in the
// render box's log as "[rt adaptive]").
//
//   node tests/browser_fps.cjs --url 'https://<public link>/' --key <k> --out DIR [--minutes 4] [--char muse]
//        [--quality auto] [--cam clip.y4m --mic clip.wav] [--extra 'attention=0&react=rules'] [--label after]
//
// choppy additions: --latency <ms> adds that much latency to every request the page makes (Chrome devtools network
// emulation; a phone on cellular / a far Wi-Fi gave 220-370 ms per stream request, the server's own machine 78 ms), so a
// run from the server machine reproduces what a phone sees; --page-file <page.html> serves that file for the page itself (the
// app's other files still come from the link), so a page change is measured against a live backend before it is
// deployed. The report adds the page's direct-voice numbers (need/late/video lag) and per-30 s shown fps.
//
// recording: --rec 1 records the session (?rec=1; automated -> the judge skips it), --dpr N sets the device
// pixel ratio (a phone is 2-3; the recording is CSS px x min(dpr, 1.5)); at the end the page is stopped with its own
// Stop button, the upload drains, and the hub's assembled mp4 is analysed (tools/rec_fps_probe.py): container fps,
// distinct-frame rate, longest frame gap -> out.recording.
//
// Exit 0 when the run is SMOOTH: shown fps >= --min-fps (default 23.0 of the stream's 24) and stutters under
// --max-stutter-pct (default 1.0 % of frames) and no freeze; 1 otherwise; 2 = the session never went live.
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
// Google Chrome first: it decodes H.264 (WebCodecs) the way users' browsers do; a bare Chromium build may not
const CHROME = [process.env.CHROME_PATH, process.env.CHROME, '/opt/google/chrome/chrome', '/usr/bin/google-chrome']
  .find(p => p && fs.existsSync(p));   // none found = playwright's own chromium
const a = {minutes: 4, char: 'muse', label: 'run', 'min-fps': 23.0, 'max-stutter-pct': 1.0, extra: 'attention=0', mock: ''};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
for (const r of ['url', 'out']) if (!a[r]) { console.error('--' + r + ' required'); process.exit(2); }
fs.mkdirSync(a.out, {recursive: true});
const u = new URL(a.url);
if (a.key) u.searchParams.set('k', a.key);
u.searchParams.set('voice', 'mock'); u.searchParams.set('char', a.char); u.searchParams.set('hud', '1');
if (a.mock) u.searchParams.set('mock', a.mock);
if (a.rec === '1') u.searchParams.set('rec', '1');
for (const kv of String(a.extra || '').split('&').filter(Boolean)) { const [k, v] = kv.split('='); u.searchParams.set(k, v === undefined ? '1' : v); }

// the script: [seconds from the start of the measured window, label, command or null].  It loops until --minutes.
const O = (verb, name, extra = {}) => Object.assign({action: 'object', verb, object: name ? {name} : undefined}, extra);
const M = (name, extra = {}) => Object.assign({action: 'move', name}, extra);
const SCRIPT = [
  [0, 'talk', null],
  [15, 'gesture', M('wave')], [19, 'gesture', M('clap')], [23, 'gesture', M('bow')],
  [28, 'walk', M('walk', {dir: 'left'})], [33, 'walk', M('walk', {dir: 'right'})], [38, 'walk', M('come_back')],
  [43, 'spin', M('spin', {count: 2})], [49, 'jump', M('jump', {count: 2})], [54, 'flip', M('backflip')],
  [60, 'dance', M('dance')], [72, 'dance', M('stop')],
  [75, 'desk', O('put_on', 'headphones')], [82, 'desk', O('pull', 'chair', {target: {object: {name: 'desk'}}})],
  [90, 'desk', O('sit_in', 'chair')], [98, 'desk', O('open', 'laptop')], [112, 'desk', M('stand')],
  [118, 'desk', O('take_off', 'headphones')], [124, 'talk', M('come_back')],
];
const LOOP_S = 135;

function pct(xs, p) { if (!xs.length) return null; const s = xs.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
const r1 = x => x === null || x === undefined ? null : Math.round(x * 10) / 10;
function summarize(times, period) {
  if (times.length < 3) return {frames: times.length};
  const iv = []; for (let i = 1; i < times.length; i++) iv.push(times[i] - times[i - 1]);
  const dur = (times[times.length - 1] - times[0]) / 1000;
  // per-second shown fps
  const sec = new Map(); for (const t of times) { const k = Math.floor((t - times[0]) / 1000); sec.set(k, (sec.get(k) || 0) + 1); }
  const full = [...sec.entries()].filter(([k]) => k < Math.floor(dur)).map(([, v]) => v);
  const st = iv.filter(x => x > 1.5 * period).length, fr = iv.filter(x => x > 250);
  return {frames: times.length, seconds: r1(dur), fps: r1((times.length - 1) / Math.max(dur, 1e-3)), fps_p5_sec: pct(full, 0.05), fps_min_sec: full.length ? Math.min(...full) : null,
    interval_ms: {p50: r1(pct(iv, 0.5)), p95: r1(pct(iv, 0.95)), p99: r1(pct(iv, 0.99)), max: r1(Math.max(...iv))},
    stutters: st, stutter_pct: r1(100 * st / iv.length), freezes: fr.length, freeze_ms: r1(fr.reduce((s, x) => s + x, 0))};
}

(async () => {
  const args = ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--mute-audio',
    '--autoplay-policy=no-user-gesture-required', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows', '--disable-frame-rate-limit=0'];
  if (a.cam) args.push(`--use-file-for-fake-video-capture=${a.cam}`);
  if (a.mic) args.push(`--use-file-for-fake-audio-capture=${a.mic}`);
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args});
  const ctx = await browser.newContext({viewport: {width: 412, height: 915}, deviceScaleFactor: +(a.dpr || 1), permissions: ['camera', 'microphone']});
  const page = await ctx.newPage(), errors = [];
  if (a['page-file']) {
    // the app inlines its modules into the page when it serves it: keep the served page, swap in only the page's own
    // (last) <script> from the file
    const mine = fs.readFileSync(a['page-file'], 'utf8'), mi = mine.lastIndexOf('<script>'), mj = mine.indexOf('</script>', mi);
    const myScript = mine.slice(mi, mj + 9);
    await page.route(r => r.origin + r.pathname === u.origin + u.pathname, async route => {
      const res = await route.fetch(); const html = await res.text();
      const head = myScript.slice(0, 80), i = html.lastIndexOf(head), j = i < 0 ? -1 : html.indexOf('</script>', i);
      if (i < 0 || j < 0) { console.error('page-file: main script not found in the served page'); return route.fulfill({response: res}); }
      route.fulfill({response: res, body: html.slice(0, i) + myScript + html.slice(j + 9), headers: {...res.headers(), 'content-type': 'text/html; charset=utf-8'}});
    });
  }
  if (+a.latency > 0 && a['page-file']) {
    // request interception (page.route) bypasses devtools network emulation (a run showed poll rtt under the emulated
    // latency): with --page-file every other request is held for --latency ms before it goes out instead
    const L = +a.latency;
    await page.route(r => !(r.origin + r.pathname === u.origin + u.pathname), route => setTimeout(() => route.continue().catch(() => {}), L));
  } else if (+a.latency > 0) {
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', {offline: false, latency: +a.latency, downloadThroughput: -1, uploadThroughput: -1});
  }
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await page.waitForTimeout(1500);
  // the page's own quality picker (--quality auto | 270p | 360p | 540p | 720p; default: what the page has), then Start
  if (a.quality) await page.evaluate(q => { const e = document.getElementById('ql'); if (e) { e.value = q; e.dispatchEvent(new Event('change')); } }, a.quality);
  try { await page.click('#go', {timeout: 5000}); } catch (e) {}
  try {
    // live AND pictures arriving (a cold cloud box can take minutes to boot: --boot-s)
    await page.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live && window.__avatar.S.drawn > 24, null,
      {timeout: 1000 * +(a['boot-s'] || 420)});
  } catch (e) {
    const st = await page.evaluate(() => ({live: !!(window.__avatar && window.__avatar.S && window.__avatar.S.live), drawn: window.__avatar && window.__avatar.S ? window.__avatar.S.drawn : null,
      status: (document.getElementById('st') || {}).textContent || null})).catch(() => null);
    fs.writeFileSync(path.join(a.out, `browser_fps_${a.label}.json`), JSON.stringify({ok: false, error: 'the session never showed video', state: st, errors}, null, 1));
    console.log(JSON.stringify({ok: false, error: 'the session never showed video', state: st, errors: errors.slice(0, 3)}));
    await browser.close(); process.exit(2);
  }
  await page.waitForTimeout(3000);              // settle (first key frame, the adaptive tier)
  // the avatar itself, upright and facing the camera (before any move): to LOOK at the renderer
  // (the canvas's own pixels at the stream's size: no HUD on top, no page scaling)
  try {
    const png = await page.evaluate(() => { const c = document.getElementById('cv') || document.querySelector('canvas'); return c.toDataURL('image/png'); });
    fs.writeFileSync(path.join(a.out, `browser_fps_${a.label}_face.png`), Buffer.from(png.split(',')[1], 'base64'));
  } catch (e) {}
  // record the time of every frame the page draws: its draw loop pushes one entry to S.shownTimes per new frame
  await page.evaluate(() => {
    const S = window.__avatar.S; window.__shown = []; window.__stats = [];
    const hook = arr => { const p = arr.push.bind(arr); arr.push = (...x) => { window.__shown.push(performance.now()); return p(...x); }; return arr; };
    let cur = hook(S.shownTimes);
    // the stats loop replaces the array every 2 s (filter): re-hook, and keep each stats message
    setInterval(() => { if (S.shownTimes !== cur) cur = hook(S.shownTimes);
      const m = window.__avatar.stats && window.__avatar.stats(); if (m && m !== window.__lastM) { window.__lastM = m; window.__stats.push({t: performance.now(), m}); } }, 50);
    window.__t0 = performance.now();
  });
  const total = Math.round(+a.minutes * 60), marks = [], cmdLog = [];
  const tStart = Date.now();
  let next = 0, loop = 0;
  while ((Date.now() - tStart) / 1000 < total) {
    const el = (Date.now() - tStart) / 1000, inLoop = el - loop * LOOP_S;
    if (inLoop >= LOOP_S) { loop++; next = 0; continue; }
    if (next < SCRIPT.length && inLoop >= SCRIPT[next][0]) {
      const [, label, cmd] = SCRIPT[next++];
      const tp = await page.evaluate(c => { let res = null; if (c) { try { res = window.__avatar.command(c, 'ui'); } catch (e) { res = 'error: ' + e.message; } } return {t: performance.now() - window.__t0, res}; }, cmd);
      marks.push({t: tp.t, label}); cmdLog.push({t: r1(tp.t / 1000), label, cmd, res: tp.res});
    }
    await page.waitForTimeout(200);
  }
  const dump = await page.evaluate(() => ({shown: window.__shown.map(t => t - window.__t0), stats: window.__stats.map(s => ({t: s.t - window.__t0, m: s.m})),
    cfg: window.__avatar.S.cfg || null, st: window.__avatar.S.st, canvas: (() => { const c = document.querySelector('canvas'); return c ? [c.width, c.height] : null; })()}));
  await page.screenshot({path: path.join(a.out, `browser_fps_${a.label}.png`)}).catch(() => {});
  let recording = null;
  if (a.rec === '1') {
    // stop with the page's own button (the recorder drains its backlog and posts 'end'), then wait for the hub's mp4
    const r0 = await page.evaluate(() => window.__avatar.rec && window.__avatar.rec()).catch(() => null);
    const rid = r0 && r0.cur && r0.cur.rid;
    recording = {rid, client_before_stop: r0 && r0.cur};
    try { await page.click('#go', {timeout: 5000}); } catch (e) {}
    const root = a['rec-root'] || path.join(require('os').homedir(), 'recordings/avatar-sessions');
    let rec = null; const tw = Date.now();
    for (let i = 0; i < 240 && !rec && rid; i++) {
      await page.waitForTimeout(1000);
      try { const idx = JSON.parse(fs.readFileSync(path.join(root, 'index.json'))); rec = idx.sessions.find(x => x.rid === rid) || null; } catch (e) {}
    }
    recording.assembly_wait_s = Math.round((Date.now() - tw) / 1000);
    if (rec) {
      const mp4 = path.join(root, rec.file); recording.file = mp4; recording.record = rec;
      try {
        const cp = require('child_process');
        recording.probe = JSON.parse(cp.execFileSync('python3', [path.join(__dirname, '../tools/rec_fps_probe.py'), mp4], {maxBuffer: 1 << 26}).toString());
      } catch (e) { recording.probe_error = String(e.message || e).slice(0, 300); }
    } else recording.error = 'not assembled';
  } else {
    try { await page.evaluate(() => window.__avatar.stop && window.__avatar.stop()); await page.waitForTimeout(1200); } catch (e) {}
  }
  await browser.close();

  const fps = (dump.cfg && dump.cfg.fps) || 24, period = 1000 / fps;
  const out = {ok: true, label: a.label, url: u.origin + u.pathname, char: a.char, minutes: +a.minutes, stream_fps: fps, canvas: dump.canvas,
    overall: summarize(dump.shown, period), segments: {}, page_counters: dump.st, errors: errors.slice(0, 10), commands: cmdLog};
  const labels = [...new Set(marks.map(m => m.label))];
  for (const L of labels) {
    let ts = [], n = 0, st = 0, ivAll = [];
    for (let i = 0; i < marks.length; i++) {
      if (marks[i].label !== L) continue;
      const t0 = marks[i].t, t1 = i + 1 < marks.length ? marks[i + 1].t : total * 1000;
      const seg = dump.shown.filter(t => t >= t0 && t < t1);
      for (let k = 1; k < seg.length; k++) ivAll.push(seg[k] - seg[k - 1]);
      n += seg.length; ts.push((t1 - t0) / 1000);
    }
    const secs = ts.reduce((s, x) => s + x, 0);
    out.segments[L] = {seconds: r1(secs), fps: r1(n / Math.max(secs, 1e-3)), interval_p95: r1(pct(ivAll, 0.95)), interval_max: ivAll.length ? r1(Math.max(...ivAll)) : null,
      stutters: ivAll.filter(x => x > 1.5 * period).length};
  }
  // what the render side reported (the page forwards the server's stats in its own message)
  const rs = dump.stats.map(s => s.m && (s.m.render || s.m.r || null)).filter(Boolean);
  const num = k => rs.map(r => +r[k]).filter(x => isFinite(x));
  out.render_side = rs.length ? {samples: rs.length, fps_p50: r1(pct(num('fps'), 0.5)), fps_min: num('fps').length ? Math.min(...num('fps')) : null,
    render_ms_p50: r1(pct(num('render_ms'), 0.5)), render_ms_max: num('render_ms').length ? Math.max(...num('render_ms')) : null,
    encode_p95_max: num('encode_p95').length ? Math.max(...num('encode_p95')) : null, last: rs[rs.length - 1]} : null;
  const pg = dump.stats.map(s => s.m).filter(m => m && typeof m.fps === 'number');
  out.page_fps = pg.length ? {p50: pct(pg.map(m => m.fps), 0.5), p5: pct(pg.map(m => m.fps), 0.05), min: Math.min(...pg.map(m => m.fps))} : null;
  // direct voice: how late frames came (need = frame decoded - its audio heard undelayed), the delay, the video lag
  const lastV = [...pg].reverse().find(m => m.voice) || null;
  out.voice = lastV ? {delay_ms: lastV.voice.delay_ms, need_ms: lastV.voice.need_ms, video_lag_ms: lastV.voice.video_lag_ms || null,
    poll_rtt_ms: lastV.poll_rtt_ms, xchg: lastV.xchg} : null;
  out.recording = recording;
  out.latency_ms = +a.latency || 0; out.page_file = a['page-file'] || null; out.extra = a.extra;
  { const s30 = new Map(); for (const t of dump.shown) { const k = Math.floor(t / 30000); s30.set(k, (s30.get(k) || 0) + 1); }
    out.fps_per_30s = [...s30.entries()].sort((x, y) => x[0] - y[0]).map(([, n]) => r1(n / 30)); }
  const o = out.overall;
  out.smooth = !!(o.fps >= +a['min-fps'] && o.stutter_pct <= +a['max-stutter-pct'] && o.freezes === 0);
  fs.writeFileSync(path.join(a.out, `browser_fps_${a.label}.json`), JSON.stringify(out, null, 1));
  fs.writeFileSync(path.join(a.out, `browser_fps_${a.label}_shown.json`), JSON.stringify(dump.shown.map(x => Math.round(x * 10) / 10)));
  console.log(JSON.stringify({ok: true, smooth: out.smooth, recording: recording && {file: recording.file, probe: recording.probe, client: recording.client_before_stop}, voice: out.voice, fps_per_30s: out.fps_per_30s, page_counters: out.page_counters, overall: o, segments: out.segments, render_side: out.render_side, page_fps: out.page_fps,
    errors: out.errors.length, file: path.join(a.out, `browser_fps_${a.label}.json`)}));
  process.exit(out.smooth ? 0 : 1);
})().catch(e => { console.error(e); process.exit(2); });
