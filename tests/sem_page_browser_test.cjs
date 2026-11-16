// node tests/sem_page_browser_test.cjs [--shots DIR] — the REAL page (page.html + the modules inlined as app.py does) in
// headless Chrome, no hub / $0: the planner switch (?planner=semantic, ?sem=1, default current, setPlanner, the rig panel's
// Planner row), the plan message (planner:'semantic' + state.sem only when on; off = the old message), synthetic hub
// 'sem' / 'kf' (space:'rig') messages through onJSON -> a rig job on a page-side muse motion, the panel's Planner section
// (program, timings, warnings, targets; phone bottom sheet + desktop drawer, screenshots), semLast() / rigdbg().planner with
// the panel closed, sem_unavailable notice, no page errors.
const path = require('path'), APP = path.join(__dirname, '../app');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const http = require('http'), fs = require('fs');
const MODS = ['voice.js', 'react_feat.js', 'perception.js', 'prior.js', 'moves.js', 'motion.js', 'intent.js', 'actionlog.js', 'cospeech.js', 'thinking.js', 'perform.js', 'behavior.js', 'gestures.js', 'attention.js', 'turnbid.js', 'voicetools.js', 'recorder.js', 'rigdebug.js'];
const HTML = fs.readFileSync(path.join(APP, 'page.html'), 'utf8').replace('<!--MODULES-->', () => MODS.filter(n => fs.existsSync(path.join(APP, n))).map(n => `<script>/* ${n} */\n${fs.readFileSync(path.join(APP, n), 'utf8')}\n</script>`).join('\n') + '\n<script>window.INTENT_RULES=' + fs.readFileSync(path.join(APP, 'intent_rules.json'), 'utf8') + ';</script>');
const BODY = fs.readFileSync(path.join(__dirname, '../renderer/animals/muse.body.json'), 'utf8'), PROPS = fs.readFileSync(path.join(__dirname, '../renderer/animals/muse.props.json'), 'utf8');
const ai = process.argv.indexOf('--shots'), SHOTS = ai > 0 ? process.argv[ai + 1] : null;
let fails = 0, passes = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };

const PROGRAM = ['look_at("user", dur=0.4)', 'reach("right", target="top_of_head", dur=0.6)', 'scratch("right", cycles=3)', 'expression("puzzled", 0.6)', 'rest_arm("right")'];
// the page-side session stand-in: a muse motion stepped at 30 Hz by the test (no hub, no renderer)
const SETUP = `(() => {
  const mk = window.AvatarMotionLib && AvatarMotionLib.createMotion; if (!mk) return 'no motion lib';
  D.motion = mk({seed: 1, character: 'muse', onAction: e => { if (D.perf) D.perf.onAction(e); }}); D.motion.setProps(${PROPS}, ${BODY}, 'muse');
  const feat = () => ({t: D.motion.now(), face: {present: true, x: 0.1, y: 0, ax: 0.12, ay: -0.04, z: .5, size: .2, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []});
  D.features = feat();
  let last = performance.now(); window.__rows = [];
  window.__tick = setInterval(() => { const n = performance.now(); const row = D.motion.step(Math.min(0.1, (n - last) / 1000), {features: feat()}); last = n;
    window.__rows.push({t: n, hand_l_y: row.hand_l_y, head_yaw: row.head_yaw}); if (D.rdbg) D.rdbg.tick(row, n / 1000); }, 33);
  return 'ok';
})()`;
async function page(b, qs, vp) {
  const ctx = await b.newContext({viewport: vp || {width: 1280, height: 860}, deviceScaleFactor: vp && vp.width < 600 ? 2 : 1});
  const p = await ctx.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(String(e.message).slice(0, 200)));
  await p.goto(`http://127.0.0.1:${PORT}/${qs}`, {waitUntil: 'load'}); await p.waitForTimeout(700);
  return {p, errs};
}
let PORT = 0;
(async () => {
  const srv = http.createServer((q, r) => { if (q.url.split('?')[0] === '/') { r.setHeader('content-type', 'text/html'); r.end(HTML); } else { r.statusCode = 404; r.end(); } }).listen(0, '127.0.0.1');
  await new Promise(r => srv.on('listening', r)); PORT = srv.address().port;
  const b = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});

  // 1. the switch
  const modes = {};
  for (const [tag, qs] of [['default', '?char=muse'], ['sem', '?char=muse&planner=semantic'], ['alias', '?char=muse&sem=1'], ['off', '?char=muse&planner=0']]) {
    const {p, errs} = await page(b, qs);
    modes[tag] = await p.evaluate(() => window.__avatar.planner());
    if (errs.length) check(`page ${tag}: no page errors`, false, errs.join(' | '));
    await p.context().close();
  }
  check(`switch: default ${modes.default}, ?planner=semantic ${modes.sem}, ?sem=1 ${modes.alias}, ?planner=0 ${modes.off}`, modes.default === 'current' && modes.sem === 'semantic' && modes.alias === 'semantic' && modes.off === false);

  // 2. plan messages per mode (the page's control queue: nothing reaches a hub here)
  {
    const {p, errs} = await page(b, '?char=muse');
    const r = await p.evaluate(SETUP);
    const out = await p.evaluate(async () => {
      const q = () => window.__avatar.S.ctlQ.filter(x => x.type === 'plan').slice(-1)[0];
      window.__avatar.performAction('touch your nose'); const a = JSON.parse(JSON.stringify(q()));
      window.__avatar.D.perf.stop('t');
      const m2 = window.__avatar.setPlanner('semantic');
      window.__avatar.performAction('touch your nose'); const b2 = JSON.parse(JSON.stringify(q()));
      window.__avatar.D.perf.stop('t');
      const m3 = window.__avatar.setPlanner('current');
      window.__avatar.performAction('touch your nose'); const c = JSON.parse(JSON.stringify(q()));
      return {a, b: b2, c, m2, m3, warm: window.__avatar.S.ctlQ.filter(x => x.type === 'semwarm').length};
    });
    const strip = m => JSON.stringify(Object.assign({}, m, {id: 'x'}));
    check(`current: plan message has no planner / state.sem; back to current after semantic = the same message`, r === 'ok' && !('planner' in out.a) && !('sem' in out.a.state) && strip(out.a) === strip(out.c), JSON.stringify(out.a).slice(0, 200));
    check(`semantic: planner:'semantic' + state.sem.user from perception (${JSON.stringify(out.b.state && out.b.state.sem)})`, out.m2 === 'semantic' && out.b.planner === 'semantic' && out.b.state.sem.user.present === true && Math.abs(out.b.state.sem.user.ax - 0.12) < 1e-6);
    check(`no session: no semwarm queued (sent only while a session runs) (${out.warm})`, out.warm === 0);
    check(`no page errors`, !errs.length, errs.join(' | '));
    await p.context().close();
  }

  // 3. synthetic hub stream -> rig job + panel, phone + desktop
  for (const [lay, vp] of [['desktop', {width: 1280, height: 860}], ['phone', {width: 390, height: 844}]]) {
    const {p, errs} = await page(b, '?char=muse&planner=semantic&rig=1', vp);
    await p.evaluate(SETUP);
    const res = await p.evaluate(async (PROGRAM) => {
      const A = window.__avatar, wait = ms => new Promise(r => setTimeout(r, ms));
      const r = A.performAction('scratch your head like you are puzzled');
      const id = r.id, prog = [];
      A.onHub({type: 'sem', id, phase: 'start', program: '', source: 'planner', model: 'claude-haiku', cached: false, ms: {recv: 0}, targets: {camera: [0, 32, 210], user: [-14, 30, 205]}});
      await wait(300);
      const poses = []; for (let k = 0; k <= 60; k++) { const t = k / 20, e = Math.sin(Math.PI * Math.min(1, t / 3)); poses.push({t: +t.toFixed(3), hand_l_x: +(0.3 * e).toFixed(4), hand_l_y: +(1.0 * e).toFixed(4), hand_l_z: +(0.35 * e).toFixed(4), head_roll: +(0.12 * e).toFixed(4), brow_knit: +(0.5 * e).toFixed(4)}); }
      for (let s = 0; s < PROGRAM.length; s++) {
        prog.push(PROGRAM[s]);
        A.onHub({type: 'sem', id, phase: 'stmt', program: prog.join('\n') + '\n', source: 'planner', model: 'claude-haiku', ms: {recv: 0, first_token: 610, first_stmt: 820, first_pose: 850}, exec_ms: [14, 11, 9, 6, 5].slice(0, s + 1), warnings: s >= 3 ? ['expression "puzzled" read as "confused"'] : [], errors: [], targets: {camera: [0, 32, 210], user: [-14, 30, 205]}});
        A.onHub({type: 'kf', id, space: 'rig', hz: 20, poses: poses.slice(s * 12, s * 12 + 12), done: false});
        await wait(120);
      }
      A.onHub({type: 'kf', id, space: 'rig', hz: 20, poses: poses.slice(60), done: true, ok: true, source: 'planner', planner: 'semantic', engine: 'claude-haiku', summary: 'scratch head', ms_first_pose: 850, ms_done: 1900, usd: 0, channels: ['hand_l_x', 'hand_l_y', 'hand_l_z', 'head_roll', 'brow_knit']});
      A.onHub({type: 'sem', id, phase: 'done', program: prog.join('\n') + '\n', source: 'planner', model: 'claude-haiku', cached: false, ms: {recv: 0, first_token: 610, first_stmt: 820, first_pose: 850, done: 1900}, exec_ms: [14, 11, 9, 6, 5], warnings: ['expression "puzzled" read as "confused"'], errors: [], targets: {camera: [0, 32, 210], user: [-14, 30, 205]}});
      await wait(900);
      const st = A.D.motion.kfState(id), snapOpen = A.rigdbg();
      const peak = Math.max(...window.__rows.map(x => x.hand_l_y));
      return {id, st: st && {space: st.space, phase: st.phase, poses: st.poses}, peak, snap: snapOpen.planner, layout: snapOpen.layout,
        pre: (document.querySelector('#rigpanel .rd-prog') || {}).textContent || '', kv: (document.querySelector('#rigpanel .rd-pkv') || {}).textContent || '',
        seg: [...document.querySelectorAll('#rigpanel .rd-seg button')].map(x => x.dataset.v + ':' + x.getAttribute('aria-pressed')).join(',')};
    }, PROGRAM);
    check(`${lay}: synthetic rig stream plays as a rig job (${JSON.stringify(res.st)}, hand_l_y peak ${res.peak.toFixed(2)})`, res.st && res.st.space === 'rig' && res.st.poses === 61 && res.peak > 0.6);
    check(`${lay}: panel (${res.layout}) Planner row = ${res.seg}; program shown (${res.pre.split('\n').filter(Boolean).length} statements); timings + warnings + targets`,
      res.layout === (lay === 'phone' ? 'sheet' : 'drawer') && res.seg === 'current:false,semantic:true' && res.pre.includes('scratch("right"') && /hub first statement820 ms/.test(res.kv) && /puzzled/.test(res.kv) && /user \(/.test(res.kv), res.kv.slice(0, 300));
    check(`${lay}: rigdbg().planner = {mode, source, program, ms}`, res.snap && res.snap.mode === 'semantic' && res.snap.source === 'planner' && /rest_arm/.test(res.snap.program) && res.snap.ms.hub_first_stmt === 820 && res.snap.ms.page_first_pose > 0);
    // program text is selectable (the panel's drag / tap handlers do not swallow it)
    const sel = await p.evaluate(() => { const pre = document.querySelector('#rigpanel .rd-prog'); const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); return {txt: String(s).length, us: getComputedStyle(pre).userSelect}; });
    check(`${lay}: program text selectable / copyable (${sel.txt} chars, user-select ${sel.us})`, sel.txt > 50 && sel.us === 'text');
    if (SHOTS) {
      fs.mkdirSync(SHOTS, {recursive: true});
      await p.evaluate(() => { getSelection().removeAllRanges(); const m = document.querySelector('#rigpanel .rd-main'); if (m) m.scrollTop = 0; });
      await p.waitForTimeout(300);
      await p.screenshot({path: path.join(SHOTS, `panel-${lay}.png`)});
    }
    // the Planner row's buttons switch the mode (next command)
    const sw = await p.evaluate(async () => { document.querySelector('#rigpanel .rd-seg button[data-v=current]').click(); await new Promise(r => setTimeout(r, 400));
      return {mode: window.__avatar.planner(), seg: [...document.querySelectorAll('#rigpanel .rd-seg button')].map(x => x.getAttribute('aria-pressed')).join(','), log: window.__avatar.rigdbg().log.slice(0, 3).map(x => x.text)}; });
    check(`${lay}: tapping "current" switches (mode ${sw.mode}, buttons ${sw.seg}, log "${sw.log[0]}")`, sw.mode === 'current' && sw.seg === 'true,false' && /planner → current/.test(sw.log.join(' ')));
    if (lay === 'desktop') {
      // closed panel: semLast / rigdbg().planner still answer; current-mode command shows timings without a program
      const cl = await p.evaluate(async () => {
        const A = window.__avatar; window.AvatarRigDebug.close();
        const r = A.performAction('nod twice'); A.onHub({type: 'kf', id: r.id, poses: [{t: 0.2, head_pitch: 0.2}, {t: 0.6, head_pitch: -0.1}, {t: 1.0, head_pitch: 0}], done: true, ok: true, source: 'planner', summary: 'nod', ms_first_pose: 700, ms_done: 900});
        await new Promise(q => setTimeout(q, 300));
        const L = A.semLast(), s = A.rigdbg();
        window.AvatarRigDebug.open(); await new Promise(q => setTimeout(q, 500));
        return {open: s.open, mode: s.planner.mode, cmode: s.planner.command_mode, prog: s.planner.program, L: {mode: L.mode, source: L.source, first_pose: L.page_ms.first_pose},
          kv: document.querySelector('#rigpanel .rd-pkv').textContent, preHidden: document.querySelector('#rigpanel .rd-prog').hidden, rdbgClosed: null};
      });
      check(`panel closed: semLast() (${JSON.stringify(cl.L)}) and rigdbg().planner (${cl.cmode}) answer; current mode shows timings + summary, no program`, !cl.open && cl.L.mode === 'current' && cl.cmode === 'current' && cl.prog === '' && /summarynod/.test(cl.kv) && cl.preHidden);
      if (SHOTS) await p.screenshot({path: path.join(SHOTS, `panel-desktop-current.png`)});
      // sem_unavailable
      const un = await p.evaluate(async () => { const A = window.__avatar; A.setPlanner('semantic'); const r = A.performAction('wave');
        A.onHub({type: 'kf', id: r.id, poses: [{t: 0.2, right_paw_y: 0.5}, {t: 1, right_paw_y: 0}], done: true, ok: true, source: 'planner', planner: 'semantic', sem_unavailable: true});
        await new Promise(q => setTimeout(q, 600)); return document.querySelector('#rigpanel .rd-pnote').textContent; });
      check(`sem_unavailable -> "${un}"`, /not available for this character/.test(un));
    }
    check(`${lay}: no page errors`, !errs.length, errs.join(' | '));
    await p.context().close();
  }
  await b.close(); srv.close();
  console.log(`sem_page_browser_test: ${passes} passed, ${fails} failed`); process.exit(fails ? 1 : 0);
})().catch(e => { console.error('FAILED', e); process.exit(1); });
