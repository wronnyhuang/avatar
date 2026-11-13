// node tests/think_page_test.cjs — the REAL page (page.html + the modules inlined as app.py does) in headless Chrome,
// no hub / no session / $0: D.think is built, a perform call shows the thinking within ~200 ms on real timers (a page-side
// fox motion stepped by the test), voiceReport carries it, stop cancels it, ?think=0 turns it off, no page errors.
const path = require('path'), APP = path.join(__dirname, '../app');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const http = require('http'), fs = require('fs');
const MODS = ['voice.js', 'react_feat.js', 'perception.js', 'prior.js', 'moves.js', 'motion.js', 'intent.js', 'actionlog.js', 'cospeech.js', 'thinking.js', 'perform.js', 'behavior.js', 'gestures.js', 'attention.js', 'turnbid.js', 'voicetools.js', 'recorder.js'];
const HTML = fs.readFileSync(path.join(APP, 'page.html'), 'utf8').replace('<!--MODULES-->', () => MODS.filter(n => fs.existsSync(path.join(APP, n))).map(n => `<script>/* ${n} */\n${fs.readFileSync(path.join(APP, n), 'utf8')}\n</script>`).join('\n') + '\n<script>window.INTENT_RULES=' + fs.readFileSync(path.join(APP, 'intent_rules.json'), 'utf8') + ';</script>');
(async () => {
  const srv = http.createServer((q, r) => { if (q.url.split('?')[0] === '/') { r.setHeader('content-type', 'text/html'); r.end(HTML); } else { r.statusCode = 404; r.end(); } }).listen(0, '127.0.0.1');
  await new Promise(r => srv.on('listening', r)); const port = srv.address().port;
  const b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});
  const out = {};
  for (const [tag, qs] of [['on', '?char=fox'], ['off', '?char=fox&think=0']]) {
    const p = await (await b.newContext()).newPage(); const errs = [];
    p.on('pageerror', e => errs.push(String(e.message).slice(0, 200)));
    await p.goto(`http://127.0.0.1:${port}/${qs}`, {waitUntil: 'load'}); await p.waitForTimeout(1500);
    out[tag] = await p.evaluate(async () => {
      const R = {D: typeof D, think: typeof D !== 'undefined' && !!D.think, perf: typeof D !== 'undefined' && !!D.perf, motion: typeof D !== 'undefined' && !!D.motion, vt: typeof D !== 'undefined' && !!D.vt};
      if (!D.motion) { const lib = window.AvatarMotionLib; R.lib = Object.keys(window).filter(k => /motion/i.test(k)).slice(0, 6);
        const mk = (lib && lib.createMotion) || window.createMotion; if (!mk) return R;
        D.motion = mk({seed: 1, character: 'fox', onAction() {}}); D.motion.setProps(null, null, 'fox'); R.madeMotion = true;
        const feat = () => ({t: D.motion.now(), face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []});
        let last = performance.now(); R.rows = []; window.__tick = setInterval(() => { const n = performance.now(); const row = D.motion.step((n - last) / 1000, {features: feat()}); last = n; R.rows.push([Math.round(n), row.gaze_y, row.brow_knit]); }, 16); }
      await new Promise(q => setTimeout(q, 500)); const g0 = R.rows ? R.rows[R.rows.length - 1] : null;
      const t0 = performance.now(); let r;
      try { r = D.perf.call('do a stiff robot dance', {}); } catch (e) { R.callErr = String(e); return R; }
      R.call = r && {ok: r.ok, status: r.status};
      let seen = null;
      while (performance.now() - t0 < 1200) { await new Promise(q => setTimeout(q, 10)); const th = D.motion.thinking ? D.motion.thinking() : null; if (th && th.on && seen == null) seen = Math.round(performance.now() - t0); }
      if (R.rows) { const v = R.rows.find(x => x[0] >= t0 && (Math.abs(x[1] - g0[1]) > 0.1 || Math.abs(x[2] - g0[2]) > 0.05)); R.first_visible_ms = v ? Math.round(v[0] - t0) : null; clearInterval(window.__tick); delete R.rows; }
      R.thinking_on_ms = seen; R.stats = D.think ? D.think.stats() : null; R.thinking = D.motion.thinking ? D.motion.thinking() : null;
      R.report = typeof voiceReport === 'function' ? (() => { try { return voiceReport().think; } catch (e) { return 'ERR ' + e; } })() : 'n/a';
      try { D.perf.stop('test'); } catch (e) { R.stopErr = String(e); }
      await new Promise(q => setTimeout(q, 100)); R.after_stop = D.think ? D.think.stats().active : null; R.cmds = (D.cmds || []).filter(x => x.think).slice(-3);
      return R;
    });
    out[tag].errors = errs.slice(0, 5);
  }
  await b.close(); srv.close();
  const on = out.on, off = out.off; let fails = 0;
  const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
  check('page loads with the thinker (D.think), no page errors', on.think && on.perf && !on.errors.length, JSON.stringify(on.errors));
  check('a perform call shows the thinking on real timers: on <= 200 ms, visible <= 300 ms', on.thinking_on_ms != null && on.thinking_on_ms <= 200 && on.first_visible_ms != null && on.first_visible_ms <= 300, `on +${on.thinking_on_ms} ms, visible +${on.first_visible_ms} ms (${on.stats && on.stats.active && on.stats.active.variant})`);
  check('voiceReport().think carries the stats; stop cancels it', on.report && on.report.shown === 1 && on.after_stop === null && on.cmds.some(c => c.end === 'cancelled'), JSON.stringify(on.cmds));
  check('?think=0: no thinker, nothing shown, no page errors', !off.think && off.thinking_on_ms == null && off.first_visible_ms == null && !off.errors.length, JSON.stringify(off.errors));
  console.log(`think_page_test: ${4 - fails} passed, ${fails} failed`); process.exit(fails ? 1 : 0);
})().catch(e => { console.error('FAILED', e); process.exit(1); });
