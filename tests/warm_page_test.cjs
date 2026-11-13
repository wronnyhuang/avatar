// node tests/warm_page_test.cjs — honest warm-up: the REAL page (page.html + the modules inlined as app.py does) in
// headless Chrome, no hub / no session / $0. The hub's status messages are fed to the page's own onStatus with a fake
// direct voice: #phtx shows the estimate while there is one, an elapsed counter (ticking on the page's 1 s timer) once the
// hub says the estimate is overrun, "taking longer than usual" after a long overrun; the voice gets ONE silent body note
// while warming and ONE when the picture is live (a redialled voice is told again); the timer stops after warming.
const path = require('path'), APP = path.join(__dirname, '../app');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const http = require('http'), fs = require('fs');
const MODS = ['voice.js', 'react_feat.js', 'perception.js', 'prior.js', 'moves.js', 'motion.js', 'intent.js', 'actionlog.js', 'cospeech.js', 'thinking.js', 'perform.js', 'behavior.js', 'gestures.js', 'attention.js', 'turnbid.js', 'voicetools.js', 'recorder.js'];
const HTML = fs.readFileSync(path.join(APP, 'page.html'), 'utf8').replace('<!--MODULES-->', () => MODS.filter(n => fs.existsSync(path.join(APP, n))).map(n => `<script>/* ${n} */\n${fs.readFileSync(path.join(APP, n), 'utf8')}\n</script>`).join('\n') + '\n<script>window.INTENT_RULES=' + fs.readFileSync(path.join(APP, 'intent_rules.json'), 'utf8') + ';</script>');
(async () => {
  const srv = http.createServer((q, r) => { if (q.url.split('?')[0] === '/') { r.setHeader('content-type', 'text/html'); r.end(HTML); } else { r.statusCode = 404; r.end(); } }).listen(0, '127.0.0.1');
  await new Promise(r => srv.on('listening', r)); const port = srv.address().port;
  const b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});
  const p = await (await b.newContext()).newPage(); const errs = [];
  p.on('pageerror', e => errs.push(String(e.message).slice(0, 200)));
  await p.goto(`http://127.0.0.1:${port}/?char=fox`, {waitUntil: 'load'}); await p.waitForTimeout(1200);
  const R = await p.evaluate(async () => {
    const R = {}, tx = () => document.getElementById('phtx').textContent, sleep = ms => new Promise(q => setTimeout(q, ms));
    R.pure = [warmText(60, 30, 90), warmText(null, null, 0), warmText(null, 130, 90), warmText(null, 239, 90), warmText(null, 240, 90), warmText(null, 373, 120), warmText(0, 95, 90)];
    const mkVoice = () => { const v = {ready: false, notes: [], sendNote(t, o) { if (!v.ready) return false; v.notes.push({t, o: o || null}); return true; }}; return v; };
    const v1 = mkVoice();
    S.live = true; S.direct = true; S.stage = 'warming'; D.voice = v1;
    onStatus({type: 'status', stage: 'warming', detail: 'starting the render box', eta_s: 60, elapsed_s: 30, usual_s: 90});
    R.within = tx(); R.timerOn = !!WS.timer;
    await sleep(1200); R.notesBeforeVoiceLive = v1.notes.length; R.withinLater = tx();
    v1.ready = true;                                                       // the voice becomes live while the stage is still warming
    onStatus({type: 'status', stage: 'warming', detail: 'booting', eta_s: null, elapsed_s: 130, usual_s: 90});
    R.over0 = tx(); R.statusLine = document.getElementById('status').textContent;
    await sleep(2300); R.over2 = tx();
    onStatus({type: 'status', stage: 'warming', detail: 'booting', eta_s: null, elapsed_s: 133, usual_s: 90});   // the hub's 3 s relay: no reset, no second note
    await sleep(1100);
    R.notesWarming = v1.notes.slice();
    onStatus({type: 'status', stage: 'warming', detail: 'booting', eta_s: null, elapsed_s: 373, usual_s: 90});
    R.long = tx();
    const v2 = mkVoice(); v2.ready = true; D.voice = v2;                   // a redial while warming: the new voice session is told too
    await sleep(1200); R.notesRedial = v2.notes.slice();
    onStatus({type: 'status', stage: 'warming', detail: 'loading Fox', eta_s: 20});   // box up, character loading: an estimate again
    R.loading = tx();
    onStatus({type: 'status', stage: 'live', detail: 'Fox is listening', eta_s: null});
    R.live = tx(); R.timerAfterLive = !!WS.timer;
    await sleep(1200); R.liveLater = tx(); R.notesLive = v2.notes.slice(); R.v1Final = v1.notes.length;
    onStatus({type: 'status', stage: 'live', detail: 'video lost — voice continues', eta_s: null});
    R.notesAfterRepeat = v2.notes.length;
    // hub voice (not direct): no notes at all, the overlay still honest
    const v3 = mkVoice(); v3.ready = true; D.voice = v3; S.direct = false; S.stage = 'warming';
    onStatus({type: 'status', stage: 'warming', detail: 'booting', eta_s: null, elapsed_s: 100, usual_s: 90});
    await sleep(1100); R.hubOver = tx(); R.hubNotes = v3.notes.length;
    S.live = false; await sleep(1100); R.timerAfterStop = !!WS.timer;      // a local stop: the timer notices and stops
    D.voice = null;
    return R;
  });
  await b.close(); srv.close();
  let fails = 0, n = 0;
  const check = (name, ok, d) => { n++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (d ? '  — ' + d : '')); if (!ok) fails++; };
  const sec = s => { const m = /(\d+):(\d\d) so far/.exec(s || ''); return m ? +m[1] * 60 + +m[2] : null; };
  check('no page errors', !errs.length, JSON.stringify(errs));
  check('warmText: estimate while there is one', R.pure[0] === 'Warming up (about 60 s)' && R.pure[1] === 'Warming up', JSON.stringify(R.pure.slice(0, 2)));
  check('warmText: elapsed form once overrun (no fake ETA)', R.pure[2] === 'Still starting the renderer — 2:10 so far (usually about 90 s)' && R.pure[6] === 'Still starting the renderer — 1:35 so far (usually about 90 s)', R.pure[2] + ' | ' + R.pure[6]);
  check('warmText: "taking longer than usual" from 150 s over the usual', !/longer/.test(R.pure[3]) && /3:59 so far/.test(R.pure[3]) && R.pure[4] === 'Still starting the renderer — 4:00 so far (usually about 90 s), taking longer than usual' && R.pure[5] === 'Still starting the renderer — 6:13 so far (usually about 2 min), taking longer than usual', R.pure.slice(3, 6).join(' | '));
  check('#phtx within the estimate', R.within === 'Warming up (about 60 s)…' && R.withinLater === R.within && R.timerOn, R.within);
  check('#phtx switches to the elapsed form after the estimate', sec(R.over0) === 130 && /usually about 90 s/.test(R.over0) && !/about \d+ s\)…$/.test(R.over0.replace('(usually about 90 s)', '')), R.over0);
  check('the counter ticks on the page between hub messages', sec(R.over2) >= 131 && sec(R.over2) <= 133, R.over0 + ' -> ' + R.over2);
  check('status line keeps the hub detail', /Still starting the renderer — 2:1\d so far.* — booting$/.test(R.statusLine || ''), R.statusLine);
  check('long overrun says so', /^Still starting the renderer — 6:13 so far \(usually about 90 s\), taking longer than usual…$/.test(R.long), R.long);
  check('no note before the voice is live; ONE silent loading note once it is', R.notesBeforeVoiceLive === 0 && R.notesWarming.length === 1 && /not on his screen yet/.test(R.notesWarming[0].t) && R.notesWarming[0].o === null, JSON.stringify(R.notesWarming));
  check('a redialled voice is told once too', R.notesRedial.length === 1 && /not on his screen yet/.test(R.notesRedial[0].t), JSON.stringify(R.notesRedial.map(x => x.t.slice(0, 40))));
  check('an estimate again when the hub has one (character loading)', R.loading === 'Warming up (about 20 s)…', R.loading);
  check('live: overlay text is the hub detail and stays (timer stopped)', R.live === 'Fox is listening' && R.liveLater === R.live && R.timerAfterLive === false, R.live + ' / ' + R.liveLater);
  check('ONE "on his screen now" note at live, none repeated, old voice untouched', R.notesLive.length === 2 && /on his screen now/.test(R.notesLive[1].t) && R.notesAfterRepeat === 2 && R.v1Final === 1, JSON.stringify(R.notesLive.map(x => x.t.slice(0, 40))));
  check('hub voice: honest overlay, no notes; a stopped session stops the timer', sec(R.hubOver) >= 100 && sec(R.hubOver) <= 102 && R.hubNotes === 0 && R.timerAfterStop === false, R.hubOver);
  console.log(`warm_page_test: ${n - fails} passed, ${fails} failed`); process.exit(fails ? 1 : 0);
})().catch(e => { console.error('FAILED', e); process.exit(1); });
