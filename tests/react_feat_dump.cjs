#!/usr/bin/env node
/* Headless-Chrome dumps for the react_feat tests. Kills only its own children. Never writes next to the clips.
 *   node tests/react_feat_dump.cjs live   [--clips a,b,c] [--secs N] [--out DIR]
 *        perception canon:true (tests/canon_page.html) on the fake camera (y4m copies of the real clips) ->
 *        DIR/live/<clip>.json = {clip, names, canon:[{t,seq,motion,mask,tcm,packed,src}], wmsg:[raw worker messages], feats, stats}
 *   node tests/react_feat_dump.cjs layout [--clips a,b] [--n 150] [--out DIR]
 *        the page's percept_worker.js on fixed PNG frames (tests/mp_frames_page.html) -> DIR/layout/<clip>.js.json;
 *        tests/react_feat_ref.py layout runs Python MediaPipe on the same PNGs and compares.
 * DIR defaults to <tmpdir>/avatar-tests/canon; --clips-dir defaults to data/eval_clips. Run under nice / taskset (<= 3 cores). */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process'), http = require('http');
const pw = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const ROOT = path.resolve(__dirname, '..');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const MODE = process.argv[2] || 'live';
const OUT = arg('out', path.join(require('os').tmpdir(), 'avatar-tests', 'canon'));
const CLIPS_DIR = arg('clips-dir', process.env.AVATAR_EVAL_CLIPS || path.join(ROOT, 'data', 'eval_clips'));
const CLIPS = arg('clips', MODE === 'live' ? 'wave_mixkit_selfie_kitchen_man,emotion_happy_actor02,startle_vcd_arm_sweep_man'
  : 'talk_john_intro,wave_mixkit_selfie_kitchen_man').split(',').filter(Boolean);
const SECS = arg('secs', null);
const NFR = +arg('n', 150);
const CANON = arg('canon', '1') !== '0';
const EXE = process.env.CHROME_PATH || (fs.existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined);   // undefined = playwright's own chromium
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.png': 'image/png', '.task': 'application/octet-stream', '.json': 'application/json'};

function serve() {      // / -> the repo, /scratch/ -> OUT (frames)
  const srv = http.createServer((req, res) => {
    const u = decodeURIComponent(req.url.split('?')[0]);
    const base = u.startsWith('/scratch/') ? OUT : ROOT, rel = u.startsWith('/scratch/') ? u.slice(9) : u.slice(1);
    const fp = path.resolve(base, rel);
    if (!fp.startsWith(base) || !fs.existsSync(fp) || !fs.statSync(fp).isFile()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, {'content-type': TYPES[path.extname(fp)] || 'application/octet-stream'});
    fs.createReadStream(fp).pipe(res);
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r(srv)));
}
const ff = a => cp.execFileSync('nice', ['-n', '10', 'ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-y'].concat(a));
function y4m(clip) {
  const dst = path.join(OUT, 'y4m', clip + '.y4m');
  if (!fs.existsSync(dst)) { fs.mkdirSync(path.dirname(dst), {recursive: true}); ff(['-i', path.join(CLIPS_DIR, clip, 'camera.mp4'), '-pix_fmt', 'yuv420p', '-r', '30', dst]); }
  return dst;
}
function pngs(clip, n) {
  const dir = path.join(OUT, 'frames', clip);
  if (!fs.existsSync(path.join(dir, String(n).padStart(4, '0') + '.png'))) {
    fs.mkdirSync(dir, {recursive: true});
    ff(['-i', path.join(CLIPS_DIR, clip, 'camera.mp4'), '-frames:v', String(n), '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', path.join(dir, '%04d.png')]);
  }
  return dir;
}
const dur = clip => +cp.execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path.join(CLIPS_DIR, clip, 'camera.mp4')]).toString().trim();

async function browser(extra) {
  return pw.chromium.launch({executablePath: EXE, headless: true, args: ['--no-sandbox', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--autoplay-policy=no-user-gesture-required'].concat(extra || [])});
}

async function live(port) {
  fs.mkdirSync(path.join(OUT, 'live'), {recursive: true});
  for (const clip of CLIPS) {
    const secs = SECS ? +SECS : Math.min(30, dur(clip));
    const b = await browser(['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + y4m(clip)]);
    const L = {clip, names: null, canon: [], wmsg: [], feats: [], stats: null, error: null, consoleErr: []};
    try {
      const page = await (await b.newContext({viewport: {width: 960, height: 540}})).newPage();
      page.on('console', m => { if (m.type() === 'error') L.consoleErr.push(m.text().slice(0, 200)); });
      page.on('pageerror', e => L.consoleErr.push('pageerror ' + String(e).slice(0, 200)));
      await page.goto(`http://127.0.0.1:${port}/tests/canon_page.html?pose=1&canon=${CANON ? 1 : 0}`);
      const t0 = Date.now(); let first = null;
      while (Date.now() - t0 < (secs + 90) * 1000) {
        await sleep(1000);
        const r = await page.evaluate(() => { const g = window.__log; return {c: g.canon.splice(0), w: g.wmsg.splice(0), f: g.feats.splice(0), names: g.names, error: g.error}; });
        L.canon.push(...r.c); L.wmsg.push(...r.w); L.feats.push(...r.f); if (r.names) L.names = r.names;
        if (r.error) { L.error = r.error; break; }
        if (first == null && L.feats.length) first = Date.now();
        if (first != null && Date.now() - first > secs * 1000) break;
      }
      L.stats = await page.evaluate(() => window.__P.stats());
      await page.evaluate(() => window.__P.stop());
      const r = await page.evaluate(() => { const g = window.__log; return {c: g.canon.splice(0), w: g.wmsg.splice(0), f: g.feats.splice(0)}; });
      L.canon.push(...r.c); L.wmsg.push(...r.w); L.feats.push(...r.f);
    } catch (e) { L.error = String(e && e.stack || e); }
    finally { await b.close().catch(() => {}); }
    const fp = path.join(OUT, 'live', clip + (CANON ? '' : '.nocanon') + '.json');
    fs.writeFileSync(fp, JSON.stringify(L));
    const sm = L.stats && L.stats.models || {};
    console.log(`${clip}: canon ${L.canon.length} frames, face ${L.canon.filter(c => c.mask[0]).length}, body ${L.canon.filter(c => c.mask[1]).length}, ` +
      `hand_l ${L.canon.filter(c => c.mask[2]).length}, hand_r ${L.canon.filter(c => c.mask[3]).length}; feats ${L.feats.length}; wmsg ${L.wmsg.length}; ` +
      `fps face ${sm.face && sm.face.fps} hand ${sm.hand && sm.hand.fps} pose ${sm.pose && sm.pose.fps}; canon ${JSON.stringify(L.stats && L.stats.canon)}; ` +
      `err ${L.error || '-'} ${L.consoleErr.slice(0, 3).join(' | ')} -> ${fp}`);
  }
}

async function layout(port) {
  fs.mkdirSync(path.join(OUT, 'layout'), {recursive: true});
  for (const clip of CLIPS) {
    pngs(clip, NFR);
    const b = await browser();
    let out = null;
    try {
      const page = await (await b.newContext()).newPage();
      page.on('console', m => { if (m.type() === 'error') console.log('console', m.text().slice(0, 200)); });
      await page.goto(`http://127.0.0.1:${port}/tests/mp_frames_page.html?dir=/scratch/frames/${clip}/&n=${NFR}&fps=30`);
      const t0 = Date.now();
      while (Date.now() - t0 < 600000) { await sleep(1000); if (await page.evaluate(() => window.__out.done)) break; }
      out = await page.evaluate(() => window.__out);
    } finally { await b.close().catch(() => {}); }
    out.clip = clip; out.n = NFR;
    const fp = path.join(OUT, 'layout', clip + '.js.json');
    fs.writeFileSync(fp, JSON.stringify(out));
    console.log(`${clip}: ${out.frames.length} frames, face ${out.frames.filter(f => f.face).length}, pose ${out.frames.filter(f => f.pose).length}, ` +
      `hands ${out.frames.reduce((a, f) => a + (f.hand ? f.hand.hands.length : 0), 0)}, err ${out.error || '-'} -> ${fp}`);
  }
}

(async () => {
  const srv = await serve(), port = srv.address().port;
  try { await (MODE === 'layout' ? layout(port) : live(port)); }
  finally { srv.close(); }
})().catch(e => { console.error(e); process.exit(1); });
