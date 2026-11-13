// Realtime-voice end-to-end check: an isolated app server from <tree> (stub renderer), headless Chromium with a REAL
// realtime dial (needs OPENAI_API_KEY in the env), then voice-path commands on the page.
//   node tools/realtime_voice_e2e.cjs <tree> <label> <char0> [switchTo]
// Env: AVATAR_BASE_CONFIG  base state/config.json to copy (default <repo>/deploy/config.local.example.json)
//      E2E_SCRATCH         instance scratch root (default <repo>/runs/realtime_voice_e2e)
//      PYTHON              interpreter for app/app.py (default <tree>/.venv/bin/python, else python3)
//      PLAYWRIGHT_CORE / CHROME_PATH  playwright-core module / Chromium binary
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs'), path = require('path'), net = require('net'), {spawn} = require('child_process');
const [tree, label, char0, switchTo] = process.argv.slice(2);
if (!tree || !label || !char0) { console.error('usage: node tools/realtime_voice_e2e.cjs <tree> <label> <char0> [switchTo]'); process.exit(2); }
const REPO = path.resolve(__dirname, '..');
const SCRATCH = process.env.E2E_SCRATCH || path.join(REPO, 'runs', 'realtime_voice_e2e');
const inst = path.join(SCRATCH, `inst-${label}`), out = {label, tree, char0, switchTo: switchTo || null};
const freePort = () => new Promise(r => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  fs.rmSync(inst, {recursive: true, force: true}); fs.mkdirSync(inst + '/state', {recursive: true}); fs.mkdirSync(inst + '/cache', {recursive: true});
  const cfg = JSON.parse(fs.readFileSync(process.env.AVATAR_BASE_CONFIG || path.join(REPO, 'deploy', 'config.local.example.json')));
  cfg.local = cfg.local || {}; cfg.local.env = cfg.local.env || {};
  cfg.local.render_root = path.join(tree, 'renderer'); cfg.local.env.TA_OUT = inst + '/cache'; cfg.local.force_stub = true; cfg.daily_cap_usd = 1;
  fs.writeFileSync(inst + '/state/config.json', JSON.stringify(cfg, null, 2));
  const key = 'e2e-isolated-' + label; fs.writeFileSync(inst + '/state/secret', key, {mode: 0o600});
  const port = await freePort();
  const env = Object.assign({}, process.env, {AVATAR_HOME: inst, PORT: String(port), PYTHONDONTWRITEBYTECODE: '1', OPENBLAS_NUM_THREADS: '2', OMP_NUM_THREADS: '2'});
  const log = fs.openSync(inst + '/server.log', 'w');
  const venvPy = path.join(tree, '.venv', 'bin', 'python');
  const py = process.env.PYTHON || (fs.existsSync(venvPy) ? venvPy : 'python3');
  const hub = spawn(py, [path.join(tree, 'app/app.py')], {env, detached: true, stdio: ['ignore', log, log]});
  fs.writeFileSync(inst + '/owned-pid.json', JSON.stringify({pid: hub.pid}));
  let b = null;
  try {
    for (let i = 0; i < 100; i++) { try { const r = await fetch(`http://127.0.0.1:${port}/health`); if (r.ok) break; } catch (e) {} await sleep(150); }
    b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: true,
      args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required', '--no-sandbox']});
    const ctx = await b.newContext({permissions: ['microphone', 'camera'], viewport: {width: 480, height: 860}});
    const p = await ctx.newPage(); const errs = [];
    p.on('pageerror', e => errs.push('pageerror ' + e.message)); p.on('console', m => { if (m.type() === 'error') errs.push('console ' + m.text().slice(0, 160)); });
    await p.goto(`http://127.0.0.1:${port}/?k=${key}&char=${char0}&voice=realtime&auto=1`);
    const t0 = Date.now(); let live = false;
    while (Date.now() - t0 < 60000) { live = await p.evaluate(() => !!(window.__avatar && window.__avatar.S && window.__avatar.S.live && window.__avatar.D.voice && window.__avatar.D.motion)); if (live) break; await sleep(1000); }
    out.live = live; out.live_after_s = Math.round((Date.now() - t0) / 1000); await sleep(2500);
    if (switchTo) { await p.evaluate(k => pick(k), switchTo); await sleep(9000); out.switched = await p.evaluate(() => ({sel, cfgChar: window.__avatar.S.cfg && window.__avatar.S.cfg.character, propsFor: window.__avatar.D.propsFor})); }
    const st = () => p.evaluate(() => { const s = window.__avatar.D.motion.state ? window.__avatar.D.motion.state() : null; return s ? {posture: s.posture, sittingOn: s.sittingOn, props: (s.props || []).length, wearing: s.wearing, holding: s.holding} : null; });
    out.state0 = await st();
    const vcall = (name, args) => p.evaluate(([name, args]) => { const cmd = toolToCommand(name, args); const r = runCommand(cmd, 'voice'); return r && typeof r === 'object' ? {ok: r.ok, did: r.did, reason: r.reason} : r; }, [name, args]);
    out.backflip = await vcall('move', {name: 'backflip'}); await sleep(4500); out.state_bf = await st();
    out.put_on = await vcall('object', {verb: 'put_on', object: {name: 'headphones'}}); await sleep(14000); out.state_po = await st();
    // a gesture through the exact voice path: early phase, then done (as voice.js onCall does), recorded at the server
    if (label.startsWith('c2')) await p.evaluate(() => { window.__rawRan = true; });
    out.nod = await p.evaluate(() => {

      const args = {name: 'nod', repeat: 1};
      const early = runCommand(toolToCommand('gesture', args), 'voice-early');
      const ran = runCommand(toolToCommand('gesture', args), 'voice');
      const dupOf = ran === false && D.intentRes && D.intentRes.dup ? (D.intentRes.first && typeof D.intentRes.first === 'object' ? D.intentRes.first : {ok: true}) : null;
      const ranRec = dupOf ? {ok: dupOf.ok !== false, dup: true, did: dupOf.did, reason: dupOf.reason} : ran && typeof ran === 'object' ? {ok: ran.ok !== false, id: ran.id, did: ran.did, reason: ran.reason} : !!ran;
      sendCtl({type: 'call', name: 'gesture', args, ran: window.__rawRan ? ran : ranRec, e2eprobe: true});
      return {early: early && typeof early === 'object' ? {ok: early.ok, did: early.did} : early, done: ran && typeof ran === 'object' ? {ok: ran.ok} : ran, recorded: ranRec};
    });
    // did the head actually nod? sample head_pitch from the rig over 1.5 s
    out.nod_head_pitch_range = await p.evaluate(async () => { const v = []; for (let i = 0; i < 30; i++) { const s = window.__avatar.D.motion.debug ? window.__avatar.D.motion.debug() : null; const o = D.lastRig || null; v.push(o && o.head_pitch !== undefined ? o.head_pitch : (s && s.params && s.params.head_pitch) || 0); await new Promise(r => setTimeout(r, 50)); } return [Math.min(...v), Math.max(...v)]; });
    out.rig_sample = await p.evaluate(() => { const r = D.lastRig; return r ? (Array.isArray(r) ? 'array ' + r.length : Object.keys(r).slice(0, 12).join(',')) : null; });
    out.usd_page = await p.evaluate(() => window.__avatar.D.usd);
    out.errs = errs.slice(0, 8);
    await p.close(); await sleep(2500);
  } catch (e) { out.fatal = String(e && e.stack || e).slice(0, 400); }
  finally {
    if (b) await b.close().catch(() => {});
    try { process.kill(-hub.pid, 'SIGTERM'); } catch (e) {}
    await sleep(1500);
    try { const files = fs.readdirSync(inst + '/data').filter(f => f.startsWith('session-')).sort(); const S = JSON.parse(fs.readFileSync(inst + '/data/' + files[files.length - 1]));
      out.server_calls_page = (S.calls_page || []).map(c => ({name: c.name, ran: c.ran, probe: !!c.e2eprobe})); out.server_usd = S.usd; } catch (e) { out.server_session = 'n/a ' + e.message; }
    console.log(JSON.stringify(out, null, 1));
  }
})();
