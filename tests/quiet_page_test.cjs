// quiet: the page wiring in a real browser, no server and no vendor — page.html with the modules inlined the way
// app.py does it, opened from a file. The controller exists (window.Quiet), the cue line sits after the captions (outside
// the stage), counts down, and goes away; a level shows as "Talk: brief".   node tests/quiet_page_test.cjs [--chrome <path>]
'use strict';
const fs = require('fs'), os = require('os'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const chrome = process.argv[3] || process.env.CHROME_PATH || undefined;   // undefined = playwright's own chromium
const APP = path.join(__dirname, '..', 'app');
(async () => {
  const names = /for name in \(([\s\S]*?)\):/.exec(fs.readFileSync(path.join(APP, 'app.py'), 'utf8').split('browser modules inlined BEFORE')[1])[1].match(/"[a-z_]+\.js"/g).map(s => s.slice(1, -1));
  const mods = names.filter(n => fs.existsSync(path.join(APP, n))).map(n => '<script>/* ' + n + ' */\n' + fs.readFileSync(path.join(APP, n), 'utf8') + '\n</script>').join('\n');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'quietpage-')), file = path.join(dir, 'page.html');
  fs.writeFileSync(file, fs.readFileSync(path.join(APP, 'page.html'), 'utf8').replace('<!--MODULES-->', () => mods));
  const b = await chromium.launch({executablePath: chrome, headless: true, args: ['--no-sandbox', '--allow-file-access-from-files']});
  const p = await b.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(String(e.message || e)));
  await p.goto('file://' + file); await p.waitForTimeout(800);
  const r = await p.evaluate(async () => {
    const out = {has: !!(window.Quiet && window.AvatarQuietCtl === window.Quiet), mods: !!window.AvatarQuiet};
    if (!out.has) return out;
    const el = document.getElementById('quietcue'), caps = document.getElementById('caps'), stage = document.getElementById('stage');
    out.el = !!el; out.afterCaps = !!el && caps.nextElementSibling === el; out.outsideStage = !!el && !stage.contains(el); out.hidden0 = !!el && getComputedStyle(el).display === 'none';
    out.call = Quiet.call('be_quiet', {duration_s: 5, wake_on_speech: false});
    out.shown = getComputedStyle(el).display !== 'none'; out.text1 = el.textContent;
    await new Promise(r => setTimeout(r, 2200)); out.text2 = el.textContent;
    await new Promise(r => setTimeout(r, 3200)); out.after = {quiet: Quiet.state().quiet, text: el.textContent, display: getComputedStyle(el).display};
    Quiet.call('set_talkativeness', {level: 'less'}); out.talk = el.textContent; Quiet.call('set_talkativeness', {level: 'normal'}); out.talk0 = el.textContent;
    return out;
  });
  await b.close(); fs.rmSync(dir, {recursive: true, force: true});
  const mine = errs.filter(e => /quiet/i.test(e));
  const checks = [
    ['quiet.js is inlined and the page creates the controller (window.Quiet)', r.mods && r.has],
    ['the cue is its own line after the captions, outside the stage, hidden by default', r.el && r.afterCaps && r.outsideStage && r.hidden0],
    ['be_quiet shows "Quiet · 0:05 left"', r.call && r.call.ok && r.shown && /Quiet/.test(r.text1) && /0:05 left/.test(r.text1), r.text1],
    ['the cue counts down', /0:0[23] left/.test(r.text2), r.text2],
    ['the timer ends it and the cue goes away', r.after && r.after.quiet === false && r.after.text === '' && r.after.display === 'none', r.after],
    ['a level other than normal shows as "Talk: brief"; normal shows nothing', r.talk === 'Talk: brief' && r.talk0 === '', [r.talk, r.talk0]],
    ['no page error from the quiet code', mine.length === 0, mine],
  ];
  let fail = 0;
  for (const [n, ok, d] of checks) { if (!ok) fail++; console.log((ok ? 'PASS ' : 'FAIL ') + n + (ok || d === undefined ? '' : '  — ' + JSON.stringify(d))); }
  if (errs.length) console.log('     (page errors without a server: ' + errs.length + ' — ' + errs.slice(0, 3).map(e => e.slice(0, 80)).join(' | ') + ')');
  console.log(fail ? `${fail} FAILED` : `${checks.length}/${checks.length} passed`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
