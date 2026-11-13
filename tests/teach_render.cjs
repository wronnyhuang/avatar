// teach: real-render look check. The page on a dev server instance (tools/devinst.sh) in headless Chromium (?voice=mock), the muse driven
// through the teaching scenario with window.__avatar.teachCall (the same code path as the voice tools), the stage
// screenshotted every --every ms into --out (frames f00000.png ... + timeline.jsonl with e2eState per frame).
//   node tests/teach_render.cjs --url http://127.0.0.1:18464/ --key K --out DIR [--q 540p] [--every 200] [--scenario full|abort]
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || process.env.CHROME || undefined;   // undefined = playwright's own chromium
const a = {q: '540p', every: 200, scenario: 'full', w: 960, h: 540};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
fs.mkdirSync(a.out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader']});
  const ctx = await browser.newContext({viewport: {width: +a.w, height: +a.h + 160}, permissions: ['camera', 'microphone']});
  await ctx.addInitScript(q => { try { localStorage.setItem('avatar.q', q); } catch (e) {} }, a.q);
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { if (m.type() === 'error' || (m.type() === 'warning' && /teach/.test(m.text()))) errors.push('console: ' + m.text().slice(0, 300)); });
  const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', 'mock'); u.searchParams.set('char', 'muse'); u.searchParams.set('camera', '0'); if (a.rec) u.searchParams.set('rec', '1');
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await sleep(1500);
  await page.click('#go', {timeout: 15000});   // one start (as the e2e driver does: ?auto=1 + a click opened two sessions)
  await page.waitForFunction(() => window.__avatar && window.__avatar.D && window.__avatar.D.teach && window.__avatar.S.live, null, {timeout: 240000});
  for (let i = 0; i < 240; i++) {   // live != picture: wait for the renderer's frames (as the e2e driver does)
    const fps = await page.evaluate(() => { const S = window.__avatar.S, m = S.lastMsg || {}; return m.fps || 0; }).catch(() => 0);
    if (fps >= 3) break; await sleep(500);
    if (i === 239) { console.error('NO VIDEO (renderer not producing frames)'); await page.click('#go').catch(() => {}); await sleep(1500); await browser.close(); process.exit(2); }
  }
  await sleep(2000);
  const stage = await page.$('#stage');
  const tl = fs.createWriteStream(path.join(a.out, 'timeline.jsonl'));
  let n = 0, t0 = Date.now(), stop = false;
  const shooter = (async () => {
    while (!stop) {
      const t = Date.now();
      try {
        await stage.screenshot({path: path.join(a.out, `f${String(n).padStart(5, '0')}.png`)});
        const st = await page.evaluate(() => { const e = window.__avatar.e2eState(); const m = window.__avatar.mstate(); let cam = null; try { const E = window.__avatar.D.motion.engine.kit.E, L = window.__avatar.D.lastRig || {}; cam = {view: E.view, w: +(E.viewW || 0).toFixed(3), zoom: L.cam_zoom, x: L.cam_x, y: L.cam_y, chain: E.chain ? [E.chain.verb, E.chain.cur && E.chain.cur.name, E.chain.args && E.chain.args.verb] : null, job: E.jobs && E.jobs[0] ? E.jobs[0].name : null}; } catch (x) {} return {e, root: m && m.root, holding: m && m.holding, cam}; });
        tl.write(JSON.stringify({n, t: (t - t0) / 1000, ...st}) + '\n'); n++;
      } catch (e) { errors.push('shot: ' + e.message.slice(0, 120)); }
      const w = +a.every - (Date.now() - t); if (w > 0) await sleep(w);
    }
  })();
  const call = async (name, args) => { const r = await page.evaluate(([n2, a2]) => window.__avatar.teachCall(n2, a2), [name, args]); tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, call: name, args, r}) + '\n'); return r; };
  const until = async (pred, maxS) => { const T = Date.now(); while (Date.now() - T < maxS * 1000) { const ok = await page.evaluate(pred); if (ok) return true; await sleep(150); } return false; };
  const idle = () => { const e = window.__avatar.e2eState(); return !e.busy; };
  await sleep(1500);
  if (a.scenario === 'full') {
    await call('teach_board', {request: a.request || 'solve 3x + 2 = 11 step by step, one line per step'});
    await until(idle, 120); await sleep(2500);
    await call('board_erase', {what: a.erase || 'the last line'});
    await until(idle, 90); await sleep(1500);
    const s0 = await page.evaluate(() => window.__avatar.e2eState().stats.strokes);
    await call('teach_board', {request: a.request2 || 'draw a small flowchart: question, then think, then answer'});
    await until(`(() => { const e = window.__avatar.e2eState(); return e.phase === 'writing' && e.stats.strokes > ${s0} + ${+(a.abortAfter || 8)}; })()`, 90);
    await sleep(600);
    const ab = await page.evaluate(() => { const T = window.__avatar.D.teach, ink0 = T._T.inkSent, ph = T.state().phase; T.userText('stop', true); return {ink0, ph, at: performance.now()}; });   // abort mid-stroke
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, abort: 'stop', ...ab}) + '\n');
    const tA = Date.now(); let inkStopS = null;
    while (Date.now() - tA < 3000) { const k = await page.evaluate(() => window.__avatar.D.teach._T.inkSent); if (k === ab.ink0) { if (inkStopS === null) inkStopS = (Date.now() - tA) / 1000; } else inkStopS = null; await sleep(50); }
    await until(idle, 60);
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, abort_done: true, home_s: (Date.now() - tA) / 1000, ink_after_abort: (await page.evaluate(() => window.__avatar.D.teach._T.inkSent)) - ab.ink0}) + '\n');
    await until(idle, 60); await sleep(2500);
  } else if (a.scenario === 'lessons') {   // boardvis: lessons one after another (--requests 'a|b|c'), each to its end
    for (const rq of String(a.requests || 'solve 2x + 3 = 9').split('|')) {
      await call('teach_board', {request: rq});
      await until(idle, 150); await sleep(3000);
    }
  } else if (a.scenario === 'abort') {
    await call('teach_board', {request: a.request || 'write the times table of 7 up to 7 x 5'});
    await until(() => window.__avatar.e2eState().phase === 'writing', 60); await sleep(1200);
    await page.evaluate(() => window.__avatar.D.teach.userText('never mind', true));
    await until(idle, 60); await sleep(2000);
  } else if (a.scenario === 'probe') {   // what the director holds a few seconds after the call
    await call('teach_board', {request: a.request || 'solve 3x + 2 = 11 step by step, one line per step'});
    for (let q = 0, T0 = Date.now(); Date.now() - T0 < +(a.probeS || 9) * 1000; q++) {   // the hub's board canvas every 2 s
      if (a.snaps) { try { const r = await page.request.get(`${u.origin}/debug/surface?k=${a.key}&surface=board`); fs.writeFileSync(path.join(a.out, `hub_board_${String(q).padStart(3, '0')}_${((Date.now() - t0) / 1000).toFixed(1)}.png`), await r.body()); } catch (e) {} }
      await sleep(2000);
    }
    const P = await page.evaluate(() => { const T = window.__avatar.D.teach._T; return {state: window.__avatar.D.teach.state(), goJob: T.goJob && {done: T.goJob.done, ok: T.goJob.ok, name: T.goJob.name},
      jobs: [...T.jobs.values()].map(J => J.name), engineJobs: (() => { try { const k = window.__avatar.D.motion.engine.kit; return k.E.jobs.map(j => ({id: j.id, name: j.name, st: j.state || j.status})); } catch (e) { return String(e); } })(),
      items: T.req ? T.req.items.map(it => ({id: it.id, kind: it.kind, say: it.say_before, n: (it.strokes || []).length, bbox: it.bbox, said: it.said})) : null}; });
    P.frameProbe = await page.evaluate(async () => { const D = window.__avatar.D, T = D.teach; let n = 0; const f0 = T.frame; T.frame = (...x) => { n++; return f0(...x); };
      await new Promise(r => setTimeout(r, 1000)); T.frame = f0; let err = null; try { f0(0.01, 0); } catch (e) { err = String(e && e.stack || e).slice(0, 600); }
      return {callsPerS: n, err, live: !!window.__avatar.S.live, startCt: D.play && D.play.startCt, names: ((window.__avatar.S.cfg || {}).rig_params || []).length, phase: T.state().phase}; });
    fs.writeFileSync(path.join(a.out, 'probe.json'), JSON.stringify(P, null, 1));
  } else if (a.scenario === 'erase') {   // write, erase a part, hub canvas before/after
    await call('teach_board', {request: a.request || 'solve 3x + 2 = 11 step by step, one line per step'});
    await until(idle, 120); await sleep(1500);
    const snap = async nm => { try { const r = await page.request.get(`${u.origin}/debug/surface?k=${a.key}&surface=board`); fs.writeFileSync(path.join(a.out, nm), await r.body()); } catch (e) {} };
    await snap('hub_board_written.png');
    await call('board_erase', {what: a.erase || 'the last line'});
    await until(idle, 90); await sleep(1500);
    await snap('hub_board_erased.png');
  } else if (a.scenario === 'both') {   // dev loop: a TV picture up, then a board lesson: the framing holds both
    await page.evaluate(() => window.AvatarTeachHook.goStation('tv'));
    await until(idle, 60); await sleep(1000);
    await call('teach_board', {request: a.request || 'write 1/f = (n-1)(1/R1 - 1/R2)'});
    await until(idle, 120); await sleep(2000);
  } else if (a.scenario === 'show') {   // dev loop: at the TV, he asks to see a thing: it fetches it and shows it close
    if (a.chalk) { await call('teach_board', {request: 'write 2 + 2 = 4'}); await until(idle, 120); await sleep(1000); }
    await page.evaluate(() => window.AvatarTeachHook.goStation('tv'));
    await until(idle, 60); await sleep(1000);
    const r = await page.evaluate(o => window.__avatar.D.motion.engine.command({action: 'object', verb: 'show', object: {name: o}}, 'tool', 901), a.obj || 'eraser');
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, call: 'object show', r}) + '\n');
    await until(idle, 60); await sleep(2000);
  } else if (a.scenario === 'affect') {   // dev loop: the affect verdicts on the face / body, one after another
    for (const f of String(a.feels || 'excited:clap,sad:hand_on_heart,surprised:none,worried:none,laughing:none,angry:none').split(',')) {
      const [feel, gesture] = f.split(':');
      tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, affect: feel, gesture}) + '\n');
      await page.evaluate(([fe, g]) => onAffect({feel: fe, intensity: 0.9, gesture: g || 'none', hold_s: 5}), [feel, gesture]);
      await sleep(6000);
    }
  } else if (a.scenario === 'steps') {   // erase: a list of tool calls, timed: --steps '[["teach_board","..."],["board_erase","all"]]'
    const steps = JSON.parse(a.steps), snap = async nm => { try { const r = await page.request.get(`${u.origin}/debug/surface?k=${a.key}&surface=board`); fs.writeFileSync(path.join(a.out, nm), await r.body()); } catch (e) {} };
    const res = [];
    for (const [qi, [name, arg]] of steps.entries()) {
      const before = await page.evaluate(() => { const T = window.__avatar.D.teach._T; return {erases: T.stats.erases, items: T.stats.items}; });
      const tc = Date.now(), r = await call(name, name === 'teach_board' ? {request: arg} : name === 'board_erase' ? {what: arg} : {});
      const R = {step: qi, name, arg, ok: r && r.ok, t_call: (tc - t0) / 1000, first_mark_s: null, erase_s: null, idle_s: null, phases: []};
      let last = null, sawErase = false;
      for (const T = Date.now(); Date.now() - T < 150000;) {
        const e = await page.evaluate(() => { const D = window.__avatar.D.teach, T = D._T, e = window.__avatar.e2eState(); return {ph: e.phase, busy: e.busy, fm: e.first_mark, req: T.req && T.req.id, wipe: !!T.wipe, erases: T.stats.erases}; });
        const now = (Date.now() - tc) / 1000;
        if (e.ph !== last) { R.phases.push([+now.toFixed(2), e.ph]); last = e.ph; }
        if (e.ph === 'erasing') sawErase = true;
        if (name === 'teach_board' && R.first_mark_s === null && e.fm && e.fm.id === e.req) R.first_mark_s = e.fm.s_after_call;
        if (sawErase && R.erase_s === null && e.ph !== 'erasing' && !e.wipe) R.erase_s = +now.toFixed(2);
        if (!e.busy && now > 1.5) { R.idle_s = +now.toFixed(2); break; }
        await sleep(100);
      }
      res.push(R); tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, step: R}) + '\n');
      console.error('step', JSON.stringify(R));
      await sleep(1500); await snap(`hub_board_step${qi}.png`); await sleep(1000);
    }
    fs.writeFileSync(path.join(a.out, 'steps.json'), JSON.stringify(res, null, 1));
  } else if (a.scenario === 'combo') {   // final: lesson -> erase -> TV -> object show -> emotion, one session
    await call('teach_board', {request: a.request || 'solve 2x + 5 = 13 step by step'});
    await until(idle, 120); await sleep(2000);
    await call('board_erase', {what: 'all'});
    await until(idle, 60); await sleep(1500);
    const tv = await page.evaluate(c => window.__avatar.D.vt.call('tv_show', {concept: c}, 'done', {}), a.concept || 'a convex lens focusing parallel light rays to a focal point');
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, call: 'tv_show', r: tv}) + '\n');
    await until(idle, 60); await sleep(9000);
    const r = await page.evaluate(o => window.__avatar.D.motion.engine.command({action: 'object', verb: 'show', object: {name: o}}, 'tool', 902), a.obj || 'eraser');
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, call: 'object show', r}) + '\n');
    await until(idle, 60); await sleep(3000);
    tl.write(JSON.stringify({t: (Date.now() - t0) / 1000, affect: 'excited'}) + '\n');
    await page.evaluate(() => onAffect({feel: 'excited', intensity: 0.9, gesture: 'clap', hold_s: 5})).catch(e => errors.push('affect: ' + e.message));
    await sleep(6000);
    await call('go_home', {}); await until(idle, 60); await sleep(1500);
  } else if (a.scenario === 'tv') {
    await page.evaluate(() => window.AvatarTeachHook.goStation('tv'));
    await until(idle, 60); await sleep(2000);
    await call('go_home', {});
    await until(idle, 60); await sleep(1500);
  }
  stop = true; await shooter;
  const rec = await page.evaluate(() => ({teachLog: window.__avatar.teachLog().slice(-200), state: window.__avatar.e2eState(), log: window.__avatar.summary ? window.__avatar.summary() : null}));
  await page.click('#go', {timeout: 5000}).catch(() => {});   // end the session cleanly (a lingering one is 'replaced' by the next run, whose render link then closed)
  for (let i = 0; a.rec && i < 90; i++) {   // the recorder drains its upload (the hub then assembles the mp4)
    const r = await page.evaluate(() => { const s = window.AvatarRec && AvatarRec.stats(); return s ? {cur: s.cur ? {state: s.cur.state, q: s.cur.up && s.cur.up.queued} : null, active: s.active} : null; }).catch(() => null);
    if (!r || !r.cur || r.cur.state === 'ended' || r.cur.state === 'done' || (r.active === false && !(r.cur.q > 0))) break;
    await sleep(500);
  }
  await sleep(2000);
  fs.writeFileSync(path.join(a.out, 'final.json'), JSON.stringify({errors, frames: n, ...rec}, null, 1));
  console.log(JSON.stringify({frames: n, errors: [...new Set(errors)].slice(0, 8), first_mark: rec.state.first_mark, stats: rec.state.stats}));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
