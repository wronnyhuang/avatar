#!/usr/bin/env node
// Headless-Chromium voice-path eval runner.
//
//   node run_eval.js --url 'http://127.0.0.1:<port>/?k=<key>' --mode realtime|live|mock|hub [--vad server|semantic]
//        --mic <wav> [--cam <y4m>] --out <dir> [--seconds N] [--mock a.wav,b.wav] [--mock-lat-ms 300]
//        [--inst <instname> | --mock-dir <dir>] [--char fox] [--extra 'a=1&b=2'] [--loop] [--audible] [--headed]
//
// Feeds <wav> as the fake mic and <y4m> as the fake camera, opens url + &voice=&vad=&auto=1&hud=1, polls
// window.__avatar.voiceReport() every 5 s into <out>/report.json, then screenshot + final report + Stop.
//
// Timing: the run clock starts when the page's first audio getUserMedia() resolves (an init script records it in
// window.__evalGum; that is when the fake device starts reading the WAV), and lasts --seconds (default = WAV
// duration + 12). The WAV is fed with Chrome's '%noloop' suffix, so after its end the mic is silent (validated
// on Chrome 156); --loop feeds it looping instead, and analyze.py then labels turns past the first pass as
// loop-2 extra data.
//
// Mock mode: --mock names (bare names resolve against $VOICE_EVAL_DIR/mock/, default <repo>/runs/voice_eval/mock/)
// are copied into the instance's mock/ dir (<inst>/mock, instance found from --inst, --mock-dir, or by matching the
// URL's k= against $AVATAR_DEV_ROOT/inst/*/state/secret; AVATAR_DEV_ROOT defaults to <repo>/runs/devinst, the
// tools/devinst.sh layout) and passed as mock=<basenames>.
//
// Env: PLAYWRIGHT_CORE (module path, default 'playwright-core'), CHROME_PATH (browser binary, default Playwright's own),
//      VOICE_EVAL_CAM (default --cam y4m; without one Chrome's built-in fake camera pattern is used).
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');

const REPO = path.resolve(__dirname, '..', '..');
const CHROME = process.env.CHROME_PATH || undefined;
const DEV_ROOT = process.env.AVATAR_DEV_ROOT || path.join(REPO, 'runs', 'devinst');
const MOCK_SRC = path.join(process.env.VOICE_EVAL_DIR || path.join(REPO, 'runs', 'voice_eval'), 'mock');
const DEFAULT_CAM = process.env.VOICE_EVAL_CAM || null;

function parseArgs(argv) {
  const a = { vad: 'server', cam: DEFAULT_CAM, char: 'fox', poll: 5 };
  const flags = new Set(['loop', 'audible', 'headed']);
  for (let i = 0; i < argv.length; i++) {
    const m = /^--([a-z-]+)(?:=(.*))?$/.exec(argv[i]);
    if (!m) throw new Error(`bad arg ${argv[i]}`);
    const k = m[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (flags.has(m[1])) { a[k] = true; continue; }
    a[k] = m[2] !== undefined ? m[2] : argv[++i];
  }
  for (const r of ['url', 'mode', 'mic', 'out']) if (!a[r]) throw new Error(`--${r} is required`);
  return a;
}

function wavInfo(p) {
  const b = fs.readFileSync(p);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`${p}: not a WAV`);
  let off = 12, fmt = null, dataLen = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), len = b.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = { fmt: b.readUInt16LE(off + 8), ch: b.readUInt16LE(off + 10), sr: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === 'data') { dataLen = Math.min(len, b.length - off - 8); break; }
    off += 8 + len + (len & 1);
  }
  if (!fmt || dataLen === null) throw new Error(`${p}: no fmt/data chunk`);
  return { ...fmt, duration_s: dataLen / (fmt.sr * fmt.ch * fmt.bits / 8) };
}

function findInstance(a, url) {
  if (a.mockDir) return { name: null, mockDir: a.mockDir };
  if (a.inst) return { name: a.inst, mockDir: `${DEV_ROOT}/inst/${a.inst}/mock` };
  const k = url.searchParams.get('k');
  if (!k) return null;
  if (!fs.existsSync(`${DEV_ROOT}/inst`)) return null;
  for (const name of fs.readdirSync(`${DEV_ROOT}/inst`)) {
    try {
      if (fs.readFileSync(`${DEV_ROOT}/inst/${name}/state/secret`, 'utf8').trim() === k) return { name, mockDir: `${DEV_ROOT}/inst/${name}/mock` };
    } catch (_) { /* not an instance */ }
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function withTimeout(p, ms, label) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out after ${ms} ms`)), ms))]);
}

// Records every getUserMedia call (performance.now() at request and resolve) before any page script runs.
const INIT_SCRIPT = `(() => {
  const md = navigator.mediaDevices; if (!md || !md.getUserMedia || md.__evalWrapped) return;
  const orig = md.getUserMedia.bind(md); window.__evalGum = [];
  md.getUserMedia = async (c) => {
    const rec = { t_req_ms: performance.now(), audio: !!(c && c.audio), video: !!(c && c.video), constraints: JSON.stringify(c) };
    window.__evalGum.push(rec);
    try {
      const s = await orig(c); rec.t_res_ms = performance.now();
      rec.tracks = s.getTracks().map((t) => ({ kind: t.kind, label: t.label, settings: t.getSettings ? t.getSettings() : null }));
      return s;
    } catch (e) { rec.t_err_ms = performance.now(); rec.error = String(e); throw e; }
  };
  md.__evalWrapped = true;
})();`;

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const out = path.resolve(a.out);
  fs.mkdirSync(out, { recursive: true });
  const mic = path.resolve(a.mic), cam = a.cam ? path.resolve(a.cam) : null;
  const wi = wavInfo(mic);
  if (wi.fmt !== 1 || wi.bits !== 16) console.warn(`warning: ${mic} is fmt=${wi.fmt} bits=${wi.bits}; Chrome expects PCM s16`);
  if (cam && !fs.existsSync(cam)) throw new Error(`no such camera clip ${cam}`);
  const manifestPath = mic.replace(/\.wav$/i, '.json');
  const seconds = a.seconds ? Number(a.seconds) : Math.ceil(wi.duration_s + 12);

  const url = new URL(a.url);
  url.searchParams.set('voice', a.mode);
  if (a.mode === 'realtime' && a.vad) url.searchParams.set('vad', a.vad);
  url.searchParams.set('auto', '1');
  url.searchParams.set('hud', '1');
  if (a.char && !url.searchParams.has('char')) url.searchParams.set('char', a.char);
  let mockInfo = null;
  if (a.mock) {
    const names = [];
    const inst = findInstance(a, url);
    if (!inst) throw new Error('--mock: cannot find the instance mock dir (give --inst or --mock-dir)');
    fs.mkdirSync(inst.mockDir, { recursive: true });
    for (const m of a.mock.split(',').map((s) => s.trim()).filter(Boolean)) {
      const src = fs.existsSync(m) ? path.resolve(m) : path.join(MOCK_SRC, m);
      if (!fs.existsSync(src)) throw new Error(`mock file not found: ${m}`);
      fs.copyFileSync(src, path.join(inst.mockDir, path.basename(src)));
      names.push(path.basename(src));
    }
    url.searchParams.set('mock', names.join(','));
    mockInfo = { instance: inst.name, mock_dir: inst.mockDir, files: names };
  }
  if (a.mockLatMs) url.searchParams.set('mock_lat_ms', String(a.mockLatMs));
  if (a.extra) for (const [k, v] of new URLSearchParams(a.extra)) url.searchParams.set(k, v);
  const redacted = new URL(url); if (redacted.searchParams.has('k')) redacted.searchParams.set('k', '<redacted>');

  const logf = fs.openSync(path.join(out, 'console.log'), 'w');
  const t0 = Date.now();
  const log = (kind, text) => fs.writeSync(logf, `[+${((Date.now() - t0) / 1000).toFixed(3)}s] [${kind}] ${text}\n`);
  const say = (s) => { console.log(s); log('runner', s); };

  const meta = {
    started_at: new Date(t0).toISOString(), url: redacted.toString(), mode: a.mode, vad: a.mode === 'realtime' ? a.vad : null,
    mic, mic_info: wi, manifest: fs.existsSync(manifestPath) ? manifestPath : null, cam, seconds,
    mic_loop: !!a.loop, muted: !a.audible, mock: mockInfo, chrome: CHROME || 'playwright default',
  };
  const metaPath = path.join(out, 'run_meta.json');
  const saveMeta = () => fs.writeFileSync(metaPath, JSON.stringify(meta, null, 1));
  saveMeta();

  const args = [
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${mic}${a.loop ? '' : '%noloop'}`,
    '--autoplay-policy=no-user-gesture-required',
  ];
  if (cam) args.push(`--use-file-for-fake-video-capture=${cam}`);
  if (!a.audible) args.push('--mute-audio');   // silent by default: never play the avatar out loud
  const browser = await chromium.launch({ executablePath: CHROME, headless: !a.headed, args });
  let closed = false;
  const shutdown = async (why) => {
    if (closed) return; closed = true;
    say(`shutdown: ${why}`);
    try { await browser.close(); } catch (_) {}
    fs.closeSync(logf);
  };
  process.on('SIGINT', () => { shutdown('SIGINT').then(() => process.exit(130)); });
  process.on('SIGTERM', () => { shutdown('SIGTERM').then(() => process.exit(143)); });

  const ctx = await browser.newContext({ viewport: { width: 720, height: 1000 } });
  await ctx.grantPermissions(['microphone', 'camera'], { origin: url.origin });
  await ctx.addInitScript(INIT_SCRIPT);
  const page = await ctx.newPage();
  page.on('console', (m) => log(m.type(), m.text()));
  page.on('pageerror', (e) => log('pageerror', String(e && e.stack || e)));
  page.on('requestfailed', (r) => log('requestfailed', `${r.method()} ${r.url().replace(/([?&]k=)[^&]+/, '$1<redacted>')} ${r.failure() && r.failure().errorText}`));
  page.on('crash', () => log('crash', 'page crashed'));

  say(`open ${redacted} (mic ${path.basename(mic)} ${wi.duration_s.toFixed(1)}s${a.loop ? ' looping' : ''}, cam ${cam ? path.basename(cam) : 'default'}, run ${seconds}s after mic start)`);
  await page.goto(url.toString(), { waitUntil: 'load', timeout: 30000 });
  meta.page_loaded_at_ms = Date.now() - t0;

  // Wait (≤ 30 s) for the page's audio getUserMedia to resolve: that is t=0 of the WAV.
  let gum = null;
  for (let i = 0; i < 60 && !gum; i++) {
    gum = await page.evaluate(() => (window.__evalGum || []).find((g) => g.audio && g.t_res_ms) || null).catch(() => null);
    if (!gum) await sleep(500);
  }
  const tStart = Date.now();
  if (gum) {
    meta.gum_audio = gum; meta.mic_started_at_ms = tStart - t0;
    say(`mic getUserMedia resolved at page t=${gum.t_res_ms.toFixed(0)} ms (${gum.tracks ? gum.tracks.map((t) => t.label).join(', ') : ''})`);
  } else {
    meta.gum_audio = null;
    say('WARNING: no audio getUserMedia within 30 s of load (auto=1 not honoured?) — timing the run from now');
  }
  saveMeta();

  const pollReport = async () => {
    try {
      let r = await withTimeout(page.evaluate(async () => {
        const A = window.__avatar;
        if (A && typeof A.voiceReport === 'function') {
          let v = await A.voiceReport();
          if (typeof v === 'string') { try { v = JSON.parse(v); } catch (_) {} }
          return { ok: true, report: v };
        }
        return { ok: false, why: A ? 'window.__avatar.voiceReport missing' : 'window.__avatar missing',
                 stats: A && typeof A.stats === 'function' ? A.stats() : null };
      }), 4000, 'voiceReport');
      return r;
    } catch (e) { return { ok: false, why: String(e) }; }
  };
  const extras = async () => page.evaluate(() => ({ gum: window.__evalGum || [], now_ms: performance.now(), time_origin: performance.timeOrigin })).catch(() => null);

  let polls = 0, lastOk = null, lastWhy = null;
  const savePoll = async (file) => {
    const r = await pollReport(); polls++;
    if (r.ok) {
      lastOk = r.report;
      fs.writeFileSync(path.join(out, file), JSON.stringify(r.report, null, 1));
    } else {
      if (r.why !== lastWhy) say(`voiceReport unavailable: ${r.why}`);
      lastWhy = r.why;
      if (r.stats) fs.writeFileSync(path.join(out, 'stats_fallback.json'), JSON.stringify(r.stats, null, 1));
    }
    return r;
  };

  const deadline = tStart + seconds * 1000;
  while (Date.now() < deadline && !closed) {
    await sleep(Math.min(a.poll * 1000, Math.max(0, deadline - Date.now())));
    const r = await savePoll('report.json');
    if (r.ok) {
      const rep = r.report || {};
      const nt = Array.isArray(rep.turns) ? rep.turns.length : '?', nc = Array.isArray(rep.calls) ? rep.calls.length : '?';
      say(`t+${((Date.now() - tStart) / 1000).toFixed(0)}s state=${rep.state} turns=${nt} calls=${nc} usd=${rep.usd_est}`);
    }
  }
  if (closed) return;

  try { await page.screenshot({ path: path.join(out, 'screenshot.png') }); } catch (e) { say(`screenshot failed: ${e}`); }
  await savePoll('report.json');
  meta.ended_at_ms = Date.now() - t0;
  meta.session_s = (Date.now() - tStart) / 1000;
  const ex = await extras();
  if (ex) { meta.gum_all = ex.gum; meta.page_time_origin = ex.time_origin; meta.page_now_at_end_ms = ex.now_ms; }

  // Stop: the Start/Stop toggle (#go.stop) if it is in its stop state, else window.__avatar.stop().
  let how = 'none';
  try {
    how = await withTimeout(page.evaluate(async () => {
      const g = document.querySelector('#go');
      if (g && g.classList.contains('stop')) { g.click(); return 'clicked #go'; }
      if (window.__avatar && typeof window.__avatar.stop === 'function') { await window.__avatar.stop(); return '__avatar.stop()'; }
      return 'no stop control';
    }), 5000, 'stop');
  } catch (e) { how = `stop failed: ${e}`; }
  say(`stop: ${how}`);
  meta.stop = how;
  await sleep(2000);
  await savePoll('report_after_stop.json');
  meta.polls = polls;
  meta.report_ok = !!lastOk;
  saveMeta();
  say(`done: ${out}/report.json (${lastOk ? 'ok' : 'NO voiceReport — see console.log'})`);
  await shutdown('done');
}

main().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
