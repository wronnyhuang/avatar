#!/usr/bin/env node
/* Headless smoke test for app/perception.js against the y4m fixture clips (fake camera).
 *   node tests/perception_smoke.cjs [--clips a,b] [--secs 20] [--hflip vcd_face_motion,vcd_hand_motion] [--shot vcd_face_motion]
 * Writes <OUT>/<clip>.jsonl (features without mesh/lm, plus {"event":..} lines and a final {"stats":..} line),
 * <OUT>/summary.json, screenshots, and prints a summary incl. the geometry sign checks (landmark-proxy correlations
 * and the hflip comparison). Kills only its own child processes. */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process');
const pw = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const ROOT = path.resolve(__dirname, '..');
const CLIPS_DIR = process.env.PERC_CLIPS_DIR || path.join(process.env.AVATAR_DATA || path.join(ROOT, 'data'), 'clips');
const OUT = process.env.PERC_OUT || path.join(process.env.AVATAR_DATA || path.join(ROOT, 'data'), 'perception');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const CLIPS = arg('clips', 'vcd_face_motion,vcd_hand_motion,wave_silent,watch_object_silent,sam_story,emotion_happy,emotion_surprise').split(',').filter(Boolean);
const SECS = +arg('secs', 20);
const HFLIP = arg('hflip', 'vcd_face_motion,vcd_hand_motion').split(',').filter(Boolean);
const SHOT = arg('shot', 'vcd_face_motion');
const PORT = +arg('port', 18926);
const EXE = process.env.CHROME_PATH || (fs.existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined);   // undefined = playwright's own chromium
fs.mkdirSync(OUT, {recursive: true});

const sleep = ms => new Promise(r => setTimeout(r, ms));
const pct = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const corr = (a, b) => { const n = Math.min(a.length, b.length); if (n < 5) return null; const ma = mean(a.slice(0, n)), mb = mean(b.slice(0, n)); let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; } return s / Math.sqrt(sa * sb + 1e-12); };
const f3 = v => v == null ? 'n/a' : (+v).toFixed(3);
const rng = a => a.length ? `[${f3(pct(a, 0.02))} .. ${f3(pct(a, 0.98))}] med ${f3(pct(a, 0.5))}` : 'n/a';

function hflipClip(name) {
  const src = path.join(CLIPS_DIR, name + '.y4m'), dst = path.join(OUT, name + '_hflip.y4m');
  if (!fs.existsSync(dst)) cp.execFileSync('nice', ['-n', '10', 'ffmpeg', '-v', 'error', '-y', '-i', src, '-vf', 'hflip', '-pix_fmt', 'yuv420p', dst]);
  return dst;
}

async function runClip(label, y4m, opts) {
  const browser = await pw.chromium.launch({executablePath: EXE, headless: true, args: ['--no-sandbox',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + y4m,
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--autoplay-policy=no-user-gesture-required']});
  const out = fs.createWriteStream(path.join(OUT, label + '.jsonl'));
  const frames = [], events = [], consoleErr = [];
  let stats = null, err = null;
  try {
    const ctx = await browser.newContext({viewport: {width: 960, height: 540}, deviceScaleFactor: 2});
    const page = await ctx.newPage();
    page.on('console', m => { if (m.type() === 'error') consoleErr.push(m.text().slice(0, 200)); });
    page.on('pageerror', e => consoleErr.push('pageerror ' + String(e).slice(0, 200)));
    await page.goto(`http://127.0.0.1:${PORT}/tests/perception_page.html?debug=1`);
    const t0 = Date.now(); let firstFrame = null, shotDone = false;
    while (Date.now() - t0 < (SECS + 60) * 1000) {
      await sleep(1000);
      const r = await page.evaluate(() => { const L = window.__log; return {f: L.frames.splice(0), e: L.events.splice(0), error: L.error}; });
      if (r.error) { err = r.error; break; }
      for (const f of r.f) { frames.push(f); out.write(JSON.stringify(f) + '\n'); }
      for (const e of r.e) { events.push(e); out.write(JSON.stringify({event: e}) + '\n'); }
      if (frames.length && firstFrame == null) firstFrame = Date.now();
      if (opts.shot && !shotDone && firstFrame && Date.now() - firstFrame > 6000 && frames.length && frames[frames.length - 1].face.present && (!/hand/.test(label) || frames[frames.length - 1].hands.length)) {
        await page.screenshot({path: path.join(OUT, label + '_page.png')});
        const box = await page.$('.avp-preview');
        if (box) await box.screenshot({path: path.join(OUT, label + '_preview.png')});
        shotDone = true;
      }
      if (firstFrame && Date.now() - firstFrame > SECS * 1000) break;
    }
    stats = await page.evaluate(() => window.__P.stats());
    if (opts.toggle) {   // setPreview(false) must keep tracking; tap toggles mesh
      const n0 = await page.evaluate(() => { window.__P.setPreview(false); return window.__log.frames.length; });
      await sleep(2500);
      const n1 = await page.evaluate(() => window.__log.frames.length);
      await page.evaluate(() => window.__P.setPreview(true));
      stats.previewOffFrames = n1 - n0;
    }
    await page.evaluate(() => window.__P.stop());
    await ctx.close();
  } catch (e) { err = String(e && e.stack || e); }
  finally { await browser.close().catch(() => {}); }
  out.write(JSON.stringify({stats, error: err, consoleErr: consoleErr.slice(0, 20)}) + '\n'); out.end();
  return {label, frames, events, stats, err, consoleErr};
}

function summarize(r) {
  const F = r.frames, faces = F.filter(f => f.face.present).map(f => f.face);
  const s = {clip: r.label, frames: F.length, error: r.err, consoleErr: r.consoleErr.slice(0, 5), models: {}};
  for (const [m, v] of Object.entries((r.stats && r.stats.models) || {})) s.models[m] = {fps: v.fps, p50: v.infer_p50 && +v.infer_p50.toFixed(1), p95: v.infer_p95 && +v.infer_p95.toFixed(1), lat50: v.latency_p50 && +v.latency_p50.toFixed(1), failed: v.failed};
  if (F.length > 1) s.emitFps = +((F.length - 1) / (F[F.length - 1].t - F[0].t)).toFixed(2);
  s.facePresent = F.length ? +(faces.length / F.length).toFixed(3) : 0;
  s.x = rng(faces.map(f => f.x)); s.ax = rng(faces.map(f => f.ax)); s.yaw = rng(faces.map(f => f.yaw));
  s.pitch = rng(faces.map(f => f.pitch)); s.roll = rng(faces.map(f => f.roll)); s.gazeX = rng(faces.map(f => f.gazeX)); s.gazeY = rng(faces.map(f => f.gazeY));
  s.z = rng(faces.map(f => f.z));
  s.attention = {mean: f3(mean(F.map(f => f.attention))), attendingRatio: f3(mean(F.map(f => f.attending ? 1 : 0)))};
  s.motion = rng(F.map(f => f.motion)); s.approach = rng(F.map(f => f.approach));
  const ev = {}; for (const e of r.events) if (!/^model_/.test(e.type)) ev[e.type + (e.continuing ? '(cont)' : '')] = (ev[e.type + (e.continuing ? '(cont)' : '')] || 0) + 1;
  s.events = ev;
  s.eventList = r.events.filter(e => !/^model_/.test(e.type)).map(e => `${e.type}@${e.t.toFixed(1)}${e.side ? ':' + e.side : ''}${e.handed ? '/' + e.handed : ''}${e.at ? '>' + e.at : ''}${e.source ? '(' + e.source + ')' : ''}`).slice(0, 30);
  const hands = F.flatMap(f => f.hands.map(h => Object.assign({fx: f.face.present ? f.face.x : 0}, h)));
  s.hands = {n: hands.length, sideL: hands.filter(h => h.side === 'l').length, sideR: hands.filter(h => h.side === 'r').length,
    gestures: hands.reduce((a, h) => (a[h.gesture] = (a[h.gesture] || 0) + 1, a), {})};
  const unc = hands.filter(h => Math.abs(h.x - h.fx) > 0.2 && h.handed);
  s.hands.handedMatchesSide = unc.length ? +(unc.filter(h => (h.side === 'l') === (h.handed === 'left')).length / unc.length).toFixed(3) : null;
  s.hands.uncrossedN = unc.length;
  // landmark-proxy sign checks (viewer convention) — + means the matrix/blendshape value agrees with the geometry
  const D = faces.filter(f => f.dbg);
  if (D.length > 10) s.signCheck = {
    yaw_vs_noseOffset: f3(corr(D.map(f => f.yaw), D.map(f => f.dbg.yawP))), rawYaw_vs_noseOffset: f3(corr(D.map(f => f.dbg.yawR), D.map(f => f.dbg.yawP))),
    pitch_vs_noseDrop: f3(corr(D.map(f => f.pitch), D.map(f => f.dbg.pitchP))), rawPitch_vs_noseDrop: f3(corr(D.map(f => f.dbg.pitchR), D.map(f => f.dbg.pitchP))),
    roll_vs_eyeLine: f3(corr(D.map(f => f.roll), D.map(f => f.dbg.rollP))), rawRoll_vs_eyeLine: f3(corr(D.map(f => f.dbg.rollR), D.map(f => f.dbg.rollP))),
    gazeX_vs_iris: f3(corr(D.map(f => f.gazeX), D.map(f => f.dbg.gazeP))), rawA_vs_iris: f3(corr(D.map(f => f.dbg.gxA), D.map(f => f.dbg.gazeP))),
    gazeY_raw_med: f3(pct(D.map(f => f.dbg.gyA), 0.5)), eye33screen: D.filter(f => f.dbg.eyeAscreen === 'r').length / D.length};
  return s;
}

// align two runs of the same clip (normal vs hflip) on flip-invariant signals, then correlate the flip-odd ones
function hflipCompare(a, b) {
  const grid = (F) => { const fs_ = F.filter(f => f.face.present); if (!fs_.length) return null; const t0 = fs_[0].t, t1 = fs_[fs_.length - 1].t, n = Math.floor((t1 - t0) * 15); const out = []; let j = 0;
    for (let i = 0; i < n; i++) { const t = t0 + i / 15; while (j + 1 < fs_.length && fs_[j + 1].t <= t) j++; out.push(fs_[j].face); } return out; };
  const A = grid(a.frames), B = grid(b.frames); if (!A || !B) return {error: 'no faces'};
  let best = {c: -2, lag: 0};
  for (let lag = -90; lag <= 90; lag++) {
    const pa = [], pb = [];
    for (let i = 0; i < A.length; i++) { const j = i + lag; if (j < 0 || j >= B.length) continue; pa.push(A[i].y + 3 * A[i].z); pb.push(B[j].y + 3 * B[j].z); }
    if (pa.length < 60) continue; const c = corr(pa, pb); if (c > best.c) best = {c, lag};
  }
  const pairs = []; for (let i = 0; i < A.length; i++) { const j = i + best.lag; if (j >= 0 && j < B.length) pairs.push([A[i], B[j]]); }
  const C = k => f3(corr(pairs.map(p => p[0][k]), pairs.map(p => p[1][k])));
  const M = k => `${f3(mean(pairs.map(p => p[0][k])))} vs ${f3(mean(pairs.map(p => p[1][k])))}`;
  const hs = r => { const h = r.frames.flatMap(f => f.hands); return {l: h.filter(x => x.side === 'l').length, r: h.filter(x => x.side === 'r').length,
    left: h.filter(x => x.handed === 'left').length, right: h.filter(x => x.handed === 'right').length}; };
  return {alignCorr: f3(best.c), lagFrames: best.lag, n: pairs.length,
    corr_normal_vs_flipped: {x: C('x'), yaw: C('yaw'), roll: C('roll'), gazeX: C('gazeX'), lookX: C('lookX'), y: C('y'), pitch: C('pitch'), gazeY: C('gazeY'), z: C('z')},
    means_normal_vs_flipped: {x: M('x'), yaw: M('yaw'), roll: M('roll'), gazeX: M('gazeX')},
    hands_normal: hs(a), hands_flipped: hs(b)};
}

(async () => {
  const srv = cp.spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', ROOT], {stdio: 'ignore'});
  process.on('exit', () => { try { process.kill(srv.pid); } catch (_) {} });
  await sleep(800);
  const results = {}, summaries = [];
  try {
    for (const c of CLIPS) {
      const r = await runClip(c, path.join(CLIPS_DIR, c + '.y4m'), {shot: SHOT.split(',').includes(c), toggle: c === SHOT.split(',')[0]});
      results[c] = r; const s = summarize(r); if (r.stats && r.stats.previewOffFrames != null) s.previewOffFrames = r.stats.previewOffFrames;
      summaries.push(s); console.log(JSON.stringify(s, null, 1));
    }
    const flips = {};
    for (const c of HFLIP) {
      const r = await runClip(c + '_hflip', hflipClip(c), {shot: SHOT.split(',').includes(c)});
      const s = summarize(r); summaries.push(s); console.log(JSON.stringify(s, null, 1));
      if (!results[c]) results[c] = await runClip(c, path.join(CLIPS_DIR, c + '.y4m'), {});
      flips[c] = hflipCompare(results[c], r);
      console.log('HFLIP ' + c + ' ' + JSON.stringify(flips[c], null, 1));
    }
    fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify({summaries, flips}, null, 1));
  } finally { try { process.kill(srv.pid); } catch (_) {} }
})().catch(e => { console.error(e); process.exit(1); });
