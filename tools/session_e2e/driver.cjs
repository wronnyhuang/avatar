// driver.cjs — the browser half of run_session.py: opens the real page in headless Playwright Chromium (fake camera clip,
// injected controllable microphone), plays the user's turns reactively, writes the raw timeline. No judging here.
//   node driver.cjs <cfg.json>
// cfg: {url, out, chrome, cam, viewport:{w,h}, dsf, quality, params:{...}, turns:[{id, wav, say, wait, delay_s, quiet_s, max_s, settle_s}],
//       start_wait_s, max_total_s, rec, hold_open_s}
// Writes out/raw.jsonl (every page log row + {kind:'sample'} rows), out/driver.json (summary), out/page_console.log.
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
fs.mkdirSync(cfg.out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
const rawf = path.join(cfg.out, 'raw.jsonl');
const logf = path.join(cfg.out, 'driver.log');
const say = (...x) => { const s = new Date().toISOString().slice(11, 23) + ' ' + x.join(' '); console.log(s); fs.appendFileSync(logf, s + '\n'); };
const DURING = {
  // 'writing' = real chalk writing (was any 'board' word: the teach_board call itself or the page-turn erase)
  writing: /"phase":\s*"writing"|\bwriting [a-z]+\d|\bstroke\b/i,
  walking: /walk|station|goto|go_to|move_to|step|head/i,
  tv: /\btv\b|screen|scene|show_tv|tv_show|illustrat/i, moving: null, speaking: null,
};

(async () => {
  const injected = fs.readFileSync(path.join(__dirname, 'e2e_inject.js'), 'utf8');
  const args = ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    '--autoplay-policy=no-user-gesture-required', '--mute-audio', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader'];
  if (cfg.cam) args.push(`--use-file-for-fake-video-capture=${cfg.cam}`);
  const browser = await chromium.launch({executablePath: cfg.chrome || undefined, headless: true, args});
  const ctx = await browser.newContext({viewport: {width: cfg.viewport.w, height: cfg.viewport.h}, deviceScaleFactor: cfg.dsf || 1.5,
    permissions: ['camera', 'microphone']});
  await ctx.addInitScript(injected);
  const page = await ctx.newPage();
  const cons = [];
  const raw = row => fs.appendFileSync(rawf, JSON.stringify(row) + '\n');
  page.on('pageerror', e => { raw({w: Date.now(), kind: 'page_error', msg: String(e.message).slice(0, 300)}); });
  page.on('console', m => {
    const t = m.text(), ty = m.type();
    const line = {w: Date.now(), kind: 'console', type: ty, text: t.slice(0, 300)};
    if (ty === 'error' || ty === 'warning' || /board|chalk|\btv\b|station|teach|stroke|program|abort|scene|e2e|rec:/i.test(t)) raw(line);
    cons.push(ty + ': ' + t.slice(0, 300)); if (cons.length > 3000) cons.splice(0, 1000);
  });
  const u = new URL(cfg.url);
  for (const [k, v] of Object.entries(cfg.params || {})) if (v !== null && v !== undefined) u.searchParams.set(k, String(v));
  const state = {lastActive: 0, drained: 0};
  const evalSafe = async (fn, arg) => { try { return await page.evaluate(fn, arg); } catch (e) { return null; } };
  async function drain() {
    const d = await evalSafe(() => window.__e2e ? window.__e2e.drain() : null);
    if (!d) return 0;
    let fresh = 0;
    for (const r of d.log) { raw(r); fresh++; }
    const rows = [];
    for (const s of d.samples) {
      if (s[2] || s[3] || s[5] || s[6]) state.lastActive = Math.max(state.lastActive, s[0]);
      rows.push(s);
    }
    if (rows.length) fs.appendFileSync(path.join(cfg.out, 'samples.jsonl'), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
    for (const r of d.log) { if (['tx_start', 'tool', 'motion', 'state', 'response'].includes(r.kind) || /^state:/.test(r.kind)) state.lastActive = Math.max(state.lastActive, r.w); }
    state.recent = (state.recent || []).concat(d.log).slice(-400);
    return fresh;
  }
  const peek = () => evalSafe(() => window.__e2e && window.__e2e.peek());

  say('goto', u.origin + u.pathname);
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  for (let i = 0; i < 40 && !(await evalSafe(() => window.__e2e.tap())); i++) await sleep(250);
  const tapped = await evalSafe(() => window.__e2e.tapped);
  say('tap installed:', tapped);
  if (cfg.quality) await evalSafe(q => { const s = document.getElementById('ql'); if (s) s.value = q; }, cfg.quality);
  await page.waitForTimeout(600);
  await page.click('#go', {timeout: 15000});
  const tClick = Date.now();
  raw({w: tClick, kind: 'go_click'});
  try { await page.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live, null, {timeout: 150000}); }
  catch (e) {
    const st = await evalSafe(() => ({status: (document.getElementById('status') || {}).textContent, err: (document.getElementById('err') || {}).textContent}));
    say('NOT LIVE', JSON.stringify(st)); raw({w: Date.now(), kind: 'fatal', msg: 'session never went live', st});
    await page.screenshot({path: path.join(cfg.out, 'not_live.png')}).catch(() => {});
    fs.writeFileSync(path.join(cfg.out, 'driver.json'), JSON.stringify({ok: false, error: 'not live', st}, null, 1));
    await browser.close(); process.exit(2);
  }
  let tLive = Date.now();
  raw({w: tLive, kind: 'live', after_click_ms: tLive - tClick});
  say('live after', tLive - tClick, 'ms');
  // the page is 'live' before the renderer has produced a picture and before the voice call is up: wait for both
  const ready = {};
  for (let i = 0; i < 240; i++) {
    await drain();
    const st = await evalSafe(() => { const A = window.__avatar, S = A.S, m = S.lastMsg || {}; return {fps: m.fps || 0, render: (S.renderStats || {}).fps || 0, vstate: A.D && A.D.voice && A.D.voice.stats ? A.D.voice.stats().state : null}; });
    const verr = state.recent && state.recent.find(r => r.kind === 'voice_state' && r.state === 'error');
    if (verr) { say('VOICE ERROR', verr.detail); raw({w: Date.now(), kind: 'fatal', msg: 'voice error: ' + verr.detail});
      fs.writeFileSync(path.join(cfg.out, 'driver.json'), JSON.stringify({ok: false, error: 'voice error: ' + verr.detail}, null, 1)); await browser.close(); process.exit(2); }
    ready.last = st;
    if (st && st.fps >= 3 && (st.vstate === 'live' || cfg.params.voice === 'mock')) break;
    await sleep(500);
  }
  if (!(ready.last && ready.last.fps >= 3)) { say('NO VIDEO', JSON.stringify(ready.last)); raw({w: Date.now(), kind: 'fatal', msg: 'no video frames', ready: ready.last});
    fs.writeFileSync(path.join(cfg.out, 'driver.json'), JSON.stringify({ok: false, error: 'no video (renderer not producing frames)', ready: ready.last}, null, 1)); await page.click('#go').catch(() => {}); await browser.close(); process.exit(2); }
  tLive = Date.now(); raw({w: tLive, kind: 'live', ready: ready.last, after_click_ms: tLive - tClick});
  say('ready after', tLive - tClick, 'ms', JSON.stringify(ready.last));
  let rid = null;
  if (cfg.rec) {
    for (let i = 0; i < 160 && !rid; i++) { rid = await evalSafe(() => { const r = window.__avatar.rec && window.__avatar.rec(); return r && r.cur && r.cur.rid; }); if (!rid) await sleep(500); }
    raw({w: Date.now(), kind: 'rec', rid});
    say('recording rid', rid);
  }

  // ---------------------------------------------------------------- turns
  const result = {turns: []};
  let prevEnd = tLive, nTurn = 0;
  const hardStop = tLive + (cfg.max_total_s || 900) * 1000;
  async function waitStart(t, tPrevEnd, mark) {
    const w = t.wait === undefined ? 'idle' : t.wait;
    const lim = Date.now() + (t.wait_max_s || 60) * 1000;
    if (w && typeof w === 'object' && w.after_s !== undefined) { await sleepDrain(Math.max(0, tPrevEnd + w.after_s * 1000 - Date.now())); return 'after_s'; }
    if (w === 'immediate' || w === 'now') return 'immediate';
    if (w && typeof w === 'object' && w.during !== undefined) {
      const name = typeof w.during === 'string' ? w.during : (w.during.regex || ''); const rx = DURING[name] !== undefined ? DURING[name] : new RegExp(name, 'i');
      const since = Date.now();
      while (Date.now() < lim && Date.now() < hardStop) {
        await drain(); const p = await peek(); let hit = false;
        if (name === 'speaking') hit = !!(p && (p.voiced || p.spk));
        else if (name === 'moving') hit = !!(p && p.busy);
        else {
          const hay = (p ? p.active.join(' ') : '') + ' ' + state.recent.filter(r => r.w >= since - 200).map(r => [r.kind, r.name || '', r.args || '', r.v || '', r.text || ''].join(' ')).join(' ');
          hit = rx.test(hay) && (!p || true);
        }
        if (hit) { await sleepDrain(Math.max(0, (w.delay_s !== undefined ? w.delay_s : 1.5) * 1000)); return 'during'; }
        await sleep(150);
      }
      mark.during_missed = true; return 'during_missed';
    }
    // idle: avatar silent + no motion + no pending tool for quiet_s, after it reacted (or no_reply_s passed)
    const quiet = (t.quiet_s !== undefined ? t.quiet_s : (cfg.quiet_s || 1.8)) * 1000, noReply = (cfg.no_reply_s || 12) * 1000;
    const minGap = (t.min_gap_s !== undefined ? t.min_gap_s : 1.0) * 1000;
    while (Date.now() < lim && Date.now() < hardStop) {
      await drain(); const n = Date.now();
      const responded = state.lastActive > tPrevEnd + 200;
      const p = await peek();
      const active = p && (p.voiced || p.spk || p.busy || p.pending);
      if (n - tPrevEnd >= minGap && !active && n - state.lastActive >= quiet && (responded || n - tPrevEnd >= noReply)) { if (!responded) mark.no_reply = true; return 'idle'; }
      await sleep(150);
    }
    mark.idle_timeout = true; return 'idle_timeout';
  }
  async function sleepDrain(ms) { const end = Date.now() + ms; while (Date.now() < end) { await drain(); await sleep(Math.min(200, Math.max(1, end - Date.now()))); } }

  await sleepDrain((cfg.start_wait_s !== undefined ? cfg.start_wait_s : 4) * 1000);
  for (const t of cfg.turns) {
    if (Date.now() > hardStop) { say('max_total_s reached; stopping turns'); break; }
    nTurn++;
    const mark = {id: t.id, say: t.say};
    // an idle wait before this line lasts as long as the PREVIOUS turn may take (its max_s); was the
    // default 60 s, so a 70-90 s lesson was cut off by the next line ("idle_timeout") whatever max_s it was given
    const prevT = cfg.turns[nTurn - 2];
    const how = await waitStart((prevT && prevT.max_s && t.wait_max_s === undefined) ? Object.assign({}, t, {wait_max_s: prevT.max_s}) : t, prevEnd, mark);
    const wav = fs.readFileSync(t.wav);              // 16-bit mono pcm wav, 24 kHz (44-byte header)
    const sr = wav.readUInt32LE(24), pcm = wav.subarray(44);
    const rec = await page.evaluate(([b64, sr, label]) => window.__e2e.say(b64, sr, label), [pcm.toString('base64'), sr, t.id]);
    mark.started_how = how; mark.user_start_w = rec.start_w; mark.user_end_w = rec.start_w + rec.dur_ms;
    raw({w: rec.start_w, kind: 'user_turn_start', id: t.id, say: t.say, how, dur_ms: rec.dur_ms, wait: t.wait === undefined ? 'idle' : t.wait});
    say(`turn ${nTurn}/${cfg.turns.length} "${(t.say || '').slice(0, 60)}" how=${how}`);
    await sleepDrain(rec.dur_ms + 150);
    raw({w: Date.now(), kind: 'user_turn_end', id: t.id});
    prevEnd = mark.user_end_w;
    // the next turn's wait decides how long we wait; but the LAST turn waits for idle + settle here
    const nextT = cfg.turns[nTurn];
    if (!nextT) {
      const m2 = {}; await waitStart({wait: 'idle', wait_max_s: t.max_s || 90, quiet_s: t.quiet_s}, prevEnd, m2);
      Object.assign(mark, m2); await sleepDrain((t.settle_s !== undefined ? t.settle_s : 3) * 1000);
    } else if (t.settle_s) {
      await sleepDrain(t.settle_s * 1000);
    }
    mark.done_w = Date.now();
    result.turns.push(mark);
    const p = await peek(); raw({w: Date.now(), kind: 'turn_check', id: t.id, peek: p});
  }
  await drain();

  // ---------------------------------------------------------------- wrap up
  const rep = await evalSafe(() => { const A = window.__avatar; let vr = null; try { vr = A.voiceReport ? A.voiceReport() : null; } catch (e) {}
    return {voice: vr, rec: A.rec && A.rec(), last: A.S && A.S.lastMsg ? {fps: A.S.lastMsg.fps, render: A.S.renderStats, voice: A.S.lastMsg.voice && {mode: A.S.lastMsg.voice.mode}} : null, devices: A.devices && A.devices(), teachLog: A.teachLog ? A.teachLog() : null, actionLog: (window.AvatarActionLog && window.AvatarActionLog.tail) ? window.AvatarActionLog.tail(300) : null}; });   // the director's own log
  fs.writeFileSync(path.join(cfg.out, 'page_report.json'), JSON.stringify(rep, null, 1));
  await page.screenshot({path: path.join(cfg.out, 'final_page.png')}).catch(() => {});
  const tStop = Date.now();
  await page.click('#go', {timeout: 5000}).catch(() => {});
  raw({w: tStop, kind: 'stop_click'});
  // wait for the recorder to drain and say 'end'
  for (let i = 0; i < 90; i++) {
    const r = await evalSafe(() => { const s = window.AvatarRec && AvatarRec.stats(); return s ? {cur: s.cur ? {state: s.cur.state, q: s.cur.up && s.cur.up.queued} : null, active: s.active} : null; });
    if (!cfg.rec || !r || !r.cur || r.cur.state === 'ended' || r.cur.state === 'done' || (r.active === false && !(r.cur.q > 0))) break;
    await sleep(500);
  }
  await sleep(1500); await drain();
  fs.writeFileSync(path.join(cfg.out, 'page_console.log'), cons.join('\n'));
  result.ok = true; result.t_live = tLive; result.t_stop = tStop; result.rid = rid; result.tap = tapped;
  fs.writeFileSync(path.join(cfg.out, 'driver.json'), JSON.stringify(result, null, 1));
  await browser.close();
  say('done');
})().catch(e => { console.error(e); try { fs.writeFileSync(path.join(cfg.out, 'driver.json'), JSON.stringify({ok: false, error: String(e && e.stack || e)}, null, 1)); } catch (x) {} process.exit(1); });
