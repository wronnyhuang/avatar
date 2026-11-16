// run_live.cjs — live-pipeline runner: ONE character x ONE planner mode x N commands through the REAL page path
// (perform.call -> {type:'plan'} -> hub planner -> {type:'kf'} (+ {type:'sem'} in semantic mode) -> motion KF layer),
// against a running app server (any renderer; the stub is fastest), headless Chromium, mock voice ($0 voice). Camera:
// Chrome's built-in fake device by default; --cam <clip.y4m> feeds a face clip of your own (gaze following, mirrored
// expression and the live `user` target then have a face to follow). No clip ships with this repo.
//   node tools/sem_eval/run_live.cjs --url http://127.0.0.1:PORT/ --key K --char muse --mode current|semantic \
//        --cmds tests/sem_eval/commands.jsonl --out DIR [--only id,id] [--pass fresh|cached] [--cam clip.y4m]
//        [--reload 1|0 (fresh page per command, default 1)] [--warm-s 3] [--video DIR (record the page as shown)] [--max-s 25] [--settle-s 1.5] [--pre-s 1] [--post-s 1] [--wait-sem-s 30] [--param k=v ...] [--ctl-all 1]
// Per command -> DIR/<id>.json: {id, utterance, char, mode, pass, call (perform.call result), job (perf.list() row: marks,
//   source, outcome, error), kf: [hub->page kf msgs for the id, poses trimmed to counts unless --keep-poses 1],
//   sem: [hub->page sem msgs for the id], semLast, rigdbg_planner, ctl: [page->hub control msgs during the window],
//   timings: {plan_sent_ms, first_kf_ms, first_pose_ms, hub_first_pose_ms, hub_done_ms, first_motion_ms, any_motion_ms,
//   end_ms, dur_s} (page ms relative to the call), active: [motion active-set changes], errors}
// and DIR/<id>.rig.jsonl: render_timeline.py's format ({"rig_params":[...]} then {"t","a"} @ the page's ~30 Hz tick,
// t = s from window start = pre_s before the call). DIR/run.json = session summary (page errors, ctl totals).
// The page must expose window.__avatar.D.perf. Semantic mode = ?planner=semantic on the URL (page agent's switch)
// plus window.__avatar.setPlanner('semantic') when that hook exists.
'use strict';
const fs = require('fs'), path = require('path');
const PW = process.env.PLAYWRIGHT_CORE || 'playwright-core';
const CHROME = process.env.CHROME_PATH || process.env.E2E_CHROME || undefined;   // undefined = playwright's own chromium
const DEF_CAM = process.env.SEM_EVAL_CAM || '';   // '' = Chrome's synthetic fake camera
const sleep = ms => new Promise(r => setTimeout(r, ms));

function args() {
  const A = process.argv.slice(2), o = {params: {}, mode: 'current', pass: 'fresh', cam: DEF_CAM, max_s: 25, settle_s: 1.5,
    pre_s: 1, post_s: 1, wait_sem_s: 30, keep_poses: '1', ctl_all: '0'};
  for (let i = 0; i < A.length; i++) {
    const k = A[i].replace(/^--/, '').replace(/-/g, '_'), v = A[++i];
    if (k === 'param') { const j = v.indexOf('='); o.params[v.slice(0, j)] = v.slice(j + 1); } else o[k] = v;
  }
  for (const r of ['url', 'key', 'char', 'cmds', 'out']) if (!o[r]) { console.error('--' + r + ' required'); process.exit(2); }
  for (const k of ['max_s', 'settle_s', 'pre_s', 'post_s', 'wait_sem_s']) o[k] = +o[k];
  return o;
}

// page->hub control messages (api/up + api/exchange bodies: [u32 LE len][msg]..., control = 0x4A + JSON), full JSON kept
// for everything except the per-tick rig/stats traffic (counted). Same decoding as the rig panel's control tap.
const INIT = `(() => {
  if (window.__se) return;
  const SE = window.__se = {ctl: [], counts: {}, down: [], bad: 0};
  const of = window.fetch, dec = new TextDecoder();
  const SKIP = new Set(['rig', 'stats', 'rx_sync']);
  window.fetch = function (input, init) {
    try {
      const u = typeof input === 'string' ? input : (input && input.url) || '';
      const body = init && init.body;
      if (/api\\/(up|exchange)/.test(u) && body) {
        const b = body instanceof Uint8Array ? body : body instanceof ArrayBuffer ? new Uint8Array(body) : null;
        if (b) {
          const dv = new DataView(b.buffer, b.byteOffset, b.byteLength); let o = 0;
          while (o + 4 <= b.byteLength) {
            const n = dv.getUint32(o, true); o += 4; if (o + n > b.byteLength) { SE.bad++; break; }
            const m = b.subarray(o, o + n); o += n;
            if (n && m[0] === 0x4A) {
              let j = null; try { j = JSON.parse(dec.decode(m.subarray(1))); } catch (e) { SE.bad++; }
              if (j) { const t = String(j.type); SE.counts[t] = (SE.counts[t] || 0) + 1;
                if (!SKIP.has(t) || window.__seCtlAll) SE.ctl.push({pt: performance.now(), j}); if (SE.ctl.length > 20000) SE.ctl.splice(0, 10000); }
            }
          }
        }
      }
    } catch (e) { SE.bad++; }
    return of.apply(this, arguments);
  };
})();`;

// after load: wrap the global onJSON (hub->page JSON) and the motion step (rig output per tick)
const HOOK = `(() => {
  const SE = window.__se, A = window.__avatar;
  if (SE.hooked) return true;
  if (typeof window.onJSON === 'function') {
    const oj = window.onJSON;
    window.onJSON = function (j) {
      try { if (j && (j.type === 'kf' || j.type === 'sem' || j.type === 'plan' || j.type === 'semwarm')) SE.down.push({pt: performance.now(), j: JSON.parse(JSON.stringify(j))});
        if (SE.down.length > 4000) SE.down.splice(0, 2000); } catch (e) {}
      return oj.apply(this, arguments);
    };
    SE.onjson = true;
  }
  SE.rig = []; SE.act = []; SE.lastAct = '';
  SE.hookMotion = () => {
    const m = A.D.motion; if (!m || m.__seWrapped) return;
    const orig = m.step.bind(m);
    const mt = m.meta ? m.meta() : {}, names = mt.allParams || m.allParams || mt.params || m.params;   // the superset: the renderer keeps the names it knows
    SE.names = names.slice();
    m.step = (dt, inp) => {
      const o = orig(dt, inp);
      try {
        const t = performance.now();
        if (SE.rec) SE.rig.push([+(t).toFixed(1), names.map(n => { const v = o[n]; return Number.isFinite(v) ? Math.round(v * 1e4) / 1e4 : 0; })]);
        if (SE.rig.length > 3000) SE.rig.splice(0, 1000);
        const st = A.mstate && A.mstate(), act = st && st.active ? JSON.stringify(st.active) : '';
        if (act !== SE.lastAct) { SE.lastAct = act; SE.act.push([+t.toFixed(1), act]); if (SE.act.length > 2000) SE.act.splice(0, 1000); }
      } catch (e) {}
      return o;
    };
    m.__seWrapped = true;
  };
  SE.hookMotion();
  SE.hooked = true;
  return true;
})()`;

(async () => {
  const o = args();
  fs.mkdirSync(o.out, {recursive: true});
  const cmds = fs.readFileSync(o.cmds, 'utf8').split('\n').filter(s => s.trim()).map(s => JSON.parse(s))
    .filter(c => !o.only || o.only.split(',').includes(c.id));
  const log = (...x) => { const s = new Date().toISOString().slice(11, 19) + ` [${o.char}/${o.mode}/${o.pass}] ` + x.join(' ');
    console.log(s); fs.appendFileSync(path.join(o.out, 'run.log'), s + '\n'); };
  const {chromium} = require(PW);
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', ...(o.cam ? [`--use-file-for-fake-video-capture=${o.cam}`] : []), '--mute-audio', '--autoplay-policy=no-user-gesture-required',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader']});
  const vidT0 = Date.now();   // --video DIR: Playwright records what the page SHOWS (the live renderer's stream) -> true live clips
  const ctx = await browser.newContext(Object.assign({viewport: {width: 412, height: 915}, permissions: ['camera', 'microphone']},
    o.video ? {recordVideo: {dir: o.video, size: {width: 412, height: 915}}} : {}));
  const page = await ctx.newPage(), errors = [], consoleLines = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) consoleLines.push(m.type() + ': ' + m.text().slice(0, 300)); });
  await page.addInitScript(INIT);
  if (o.ctl_all === '1') await page.addInitScript('window.__seCtlAll = true;');
  const u = new URL(o.url);
  const P = Object.assign({k: o.key, voice: 'mock', mock: '', auto: '1', char: o.char, attention: '0'}, o.params);
  if (o.mode === 'semantic') P.planner = 'semantic';
  for (const [k, v] of Object.entries(P)) u.searchParams.set(k, v);
  const run = {args: o, url: u.toString().replace(/k=[^&]+/, 'k=…'), errors, console: consoleLines, started: new Date().toISOString(), cmds: []};
  const t0 = Date.now();
  // boot(mode): (re)load the page = a fresh session with the character at home, no held pose, empty page cache, then the
  // planner mode. --reload 1 (default) boots before EVERY command so commands never leak into each other (a walk or a
  // keep_pose of the previous command would otherwise be the next clip's starting state).
  let hooks = null, names = null, nBoot = 0;
  async function boot(mode) {
  const first = !nBoot++;
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await sleep(1500);
  if (!(await page.evaluate(() => !!(window.__avatar && window.__avatar.S && window.__avatar.S.live)))) { try { await page.click('#go', {timeout: 3000}); } catch (e) {} }
  await page.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live && window.__avatar.D.motion && window.__avatar.D.perf, null, {timeout: 120000});
  run.ready_ms = Date.now() - t0;
  await page.evaluate(HOOK);
  hooks = await page.evaluate(mode => {
    const A = window.__avatar, h = {onjson: !!window.__se.onjson, setPlanner: typeof A.setPlanner === 'function', semLast: typeof A.semLast === 'function',
      plannerHook: typeof A.planner === 'function' ? A.planner() : null, rigdbg: typeof A.rigdbg === 'function', character: A.S.cfg && A.S.cfg.character};
    if (h.setPlanner) { try { A.setPlanner(mode); h.set = A.planner(); } catch (e) { h.setErr = String(e); } }
    return h;
  }, mode);
  run.hooks = hooks; log('ready', run.ready_ms, 'ms', JSON.stringify(hooks));
  if (!hooks.onjson) log('WARN: onJSON not wrappable: kf/sem messages will be missing');
  if (mode === 'semantic' && (hooks.set || hooks.plannerHook) !== 'semantic') log('WARN: page does not report semantic mode (planner() =', JSON.stringify(hooks.set || hooks.plannerHook) + ')');
  await sleep(first ? 4000 : +(o.warm_s || 3) * 1000);   // perception warm-up (face model) + semwarm
  names = await page.evaluate(() => window.__se.names);
  return hooks;
  }
  await boot(o.modes ? o.modes.split(',')[0] : o.mode);

  // --modes current,semantic = INTERLEAVED on one page: per command both modes back to back (order alternating per
  // command), switched with window.__avatar.setPlanner(); --out may hold '{mode}' (default <out>/<mode>).
  const MODES = (o.modes || o.mode).split(',');
  const outOf = m => o.out.includes('{mode}') ? o.out.replace('{mode}', m) : (o.modes ? path.join(o.out, m) : o.out);
  const jobs = []; cmds.forEach((c, k) => { const ms = k % 2 ? MODES.slice().reverse() : MODES; for (const m of ms) jobs.push([c, m]); });
  for (const [c, MODE] of jobs) {
    const OUT = outOf(MODE); fs.mkdirSync(OUT, {recursive: true});
    if (o.reload !== '0' && jobs.indexOf(jobs.find(j => j[0] === c && j[1] === MODE)) > 0) await boot(MODE);
    else if (MODES.length > 1 || hooks.setPlanner) { const got = await page.evaluate(m => { try { return window.__avatar.setPlanner(m); } catch (e) { return 'err ' + e; } }, MODE); if (MODES.length > 1) await sleep(300); }
    const R = {id: c.id, utterance: c.utterance, category: c.category, char: o.char, mode: MODE, pass: o.pass, errors: []};
    try {
      // idle: no active perform job, no planner-owned motion
      const tIdle = Date.now();
      await page.waitForFunction(() => { const A = window.__avatar, p = A.D.perf; return !(p && p.busy && p.busy()) && !(p && p.active && p.active()); }, null, {timeout: 30000}).catch(() => R.errors.push('not idle before call'));
      await sleep(o.settle_s * 1000);
      R.idle_wait_ms = Date.now() - tIdle;
      const callInfo = await page.evaluate(async ([desc, pre]) => {
        const SE = window.__se, A = window.__avatar;
        SE.hookMotion();
        SE.rig = []; SE.rec = true; SE.down = []; SE.ctl = []; SE.act = SE.act.slice(-1);
        await new Promise(r => setTimeout(r, pre * 1000));
        const t = performance.now();
        let res; try { res = A.D.perf.call(desc, {via: 'sem_eval'}); } catch (e) { res = {ok: false, reason: 'call threw ' + e}; }
        return {t, epoch: performance.timeOrigin + t, res: JSON.parse(JSON.stringify(res || null))};
      }, [c.utterance, o.pre_s]);
      R.call = callInfo.res; const tc = callInfo.t; R.call_epoch_ms = Math.round(callInfo.epoch); if (o.video) R.video_call_s = +((callInfo.epoch - vidT0) / 1000).toFixed(2);
      const id = R.call && R.call.id;
      // wait for the job to close (or max_s)
      const tW = Date.now();
      let job = null;
      while (Date.now() - tW < o.max_s * 1000) {
        job = await page.evaluate(id => { const L = window.__avatar.D.perf.list(); return L.find(j => j.id === id) || null; }, id);
        if (!job || job.closed) break;
        await sleep(150);
      }
      if (job && !job.closed) R.errors.push('job not closed after ' + o.max_s + ' s');
      // the motion layer may still be finishing (library runs report their end on a fixed estimate): wait for it, max 12 s
      { const tA = Date.now();
        while (Date.now() - tA < 12000) { const busy = await page.evaluate(() => { const st = window.__avatar.mstate(); return !!(st && st.active && st.active.length); }); if (!busy) break; await sleep(150); }
        R.motion_tail_ms = Date.now() - tA; }
      // semantic: let the final sem message arrive
      if (MODE === 'semantic') {
        const tS = Date.now();
        while (Date.now() - tS < o.wait_sem_s * 1000) {
          const done = await page.evaluate(id => window.__se.down.some(d => d.j.type === 'sem' && d.j.id === id && d.j.phase === 'done') ||
            window.__se.down.some(d => d.j.type === 'kf' && d.j.id === id && d.j.done && d.j.source === 'library'), id);
          if (done) break;
          await sleep(200);
        }
      }
      await sleep(o.post_s * 1000);
      const dump = await page.evaluate(([id, tc]) => {
        const SE = window.__se, A = window.__avatar;
        SE.rec = false;
        const job = A.D.perf.list().find(j => j.id === id) || null;
        let semLast = null, rp = null;
        try { semLast = typeof A.semLast === 'function' ? A.semLast() : null; } catch (e) {}
        try { const r = typeof A.rigdbg === 'function' ? A.rigdbg() : null; rp = r && r.planner ? r.planner : null; } catch (e) {}
        return {job, rig: SE.rig, act: SE.act, down: SE.down.filter(d => d.j.id === id), ctl: SE.ctl, semLast, rp, tc};
      }, [id, tc]);
      R.job = dump.job;
      const rel = pt => pt == null ? null : Math.round(pt - tc);
      R.kf = dump.down.filter(d => d.j.type === 'kf').map(d => Object.assign({pt: rel(d.pt)}, d.j, {n_poses: (d.j.poses || []).length},
        o.keep_poses === '1' ? {} : {poses: undefined}));
      R.sem = dump.down.filter(d => d.j.type === 'sem').map(d => Object.assign({pt: rel(d.pt)}, d.j));
      R.ctl = dump.ctl.map(d => ({pt: rel(d.pt), j: d.j}));
      R.semLast = dump.semLast; R.rigdbg_planner = dump.rp;
      R.active = dump.act.map(([t, a]) => [rel(t), a]);
      const marks = (R.job && R.job.marks) || {};
      const mk = k => marks[k] != null ? Math.round(marks[k] - tc) : null;
      const planCtl = R.ctl.find(x => x.j.type === 'plan' && x.j.id === id);
      const firstKf = R.kf.find(k => true), firstPose = R.kf.find(k => k.n_poses > 0 || (k.engine && k.engine.length));
      const done = R.kf.find(k => k.done);
      // any motion: first tick after the call where any non-face channel moved >= 0.08 from its value at the call — includes
      // the thinking lead-in gesture (that is what he sees first); first_motion_ms = the planned move's own start (perform marks)
      const rig = dump.rig, iCall = rig.findIndex(r => r[0] >= tc);
      let onset = null;
      if (iCall >= 0) {
        const base = rig[iCall][1];
        for (let i = iCall + 1; i < rig.length && onset == null; i++) {
          const a = rig[i][1];
          for (let k = 0; k < a.length; k++) if (Math.abs(a[k] - base[k]) >= 0.08 && !/^(eye|blink|jaw|mouth|lip|viseme|breath)/i.test(names[k])) { onset = rel(rig[i][0]); break; }
        }
      }
      R.timings = {plan_sent_ms: planCtl ? planCtl.pt : null, first_kf_ms: firstKf ? firstKf.pt : null, first_pose_ms: mk('plan_first_pose_ms') ?? (firstPose ? firstPose.pt : null),
        hub_first_pose_ms: done ? done.ms_first_pose ?? null : null, hub_done_ms: done ? done.ms_done ?? null : null,
        ready_ms: mk('ready_ms'), first_motion_ms: mk('first_motion_ms'), any_motion_ms: onset, end_ms: mk('end_ms'),
        source: (R.job && R.job.source) || (done && done.source) || null, outcome: R.job ? R.job.outcome : null};
      if (R.sem.length) { const sd = R.sem.filter(s => s.phase === 'done').pop() || R.sem[R.sem.length - 1];
        R.program = sd.program || null; R.sem_ms = sd.ms || null; R.timings.sem = sd.ms || null; }
      else if (R.semLast && R.semLast.program) R.program = R.semLast.program;
      if (done) R.summary = done.summary || null;
      // rig file: t from the window start (pre_s before the call)
      const tStart = tc - o.pre_s * 1000;
      const lines = [JSON.stringify({rig_params: names, t_call_s: o.pre_s, t_end_s: R.timings.end_ms != null ? o.pre_s + R.timings.end_ms / 1000 : null})];
      for (const [pt, a] of rig) if (pt >= tStart) lines.push(JSON.stringify({t: +((pt - tStart) / 1000).toFixed(4), a}));
      fs.writeFileSync(path.join(OUT, c.id + '.rig.jsonl'), lines.join('\n') + '\n');
      R.n_frames = lines.length - 1; R.dur_s = R.n_frames ? +JSON.parse(lines[lines.length - 1]).t.toFixed(2) : 0;
      if (R.job && R.job.error) R.errors.push('job: ' + R.job.error);
      if (done && done.error) R.errors.push('hub: ' + done.error);
      if (MODE === 'semantic' && done && done.planner !== 'semantic') { R.errors.push('NOT SEMANTIC: the hub answered without planner:semantic'); R.not_semantic = true; }
      if (done && done.sem_unavailable) R.errors.push('sem_unavailable: semantic API not available for this character');
    } catch (e) { R.errors.push('harness: ' + String(e && e.stack || e).slice(0, 400)); }
    fs.writeFileSync(path.join(OUT, c.id + '.json'), JSON.stringify(R, null, 1));
    const T = R.timings || {};
    log(MODE, c.id, 'src', T.source, 'out', T.outcome, 'plan', T.plan_sent_ms, 'pose', T.first_pose_ms, 'hub1', T.hub_first_pose_ms,
      'motion', T.first_motion_ms, 'any', T.any_motion_ms, 'end', T.end_ms, R.errors.length ? 'ERR ' + R.errors.join(' | ') : '');
    run.cmds.push({id: c.id, mode: MODE, timings: R.timings || null, errors: R.errors});
    if (R.not_semantic && run.cmds.filter(x => x.mode === 'semantic').length === 1 && o.force !== '1') { log('ABORT: semantic mode requested but the page/hub plans with the current planner (this tree has no switch?) — --force 1 to continue'); run.aborted = 'not semantic'; break; }
  }
  run.ctl_counts = await page.evaluate(() => window.__se.counts);
  if (P.rec === '1') {   // ?rec=1: the page's own session recorder (the stage as shown): stop -> the hub assembles the mp4
    try { run.rec = await page.evaluate(() => { const r = window.__avatar.rec(); return r && r.cur ? {rid: r.cur.rid, client_ts: r.cur.client_ts, w: r.cur.w, h: r.cur.h} : null; }); } catch (e) {}
    try { await page.click('#go', {timeout: 3000}); } catch (e) {}
    await sleep(8000);
  }
  run.wall_s = Math.round((Date.now() - t0) / 100) / 10;
  for (const m of MODES) fs.writeFileSync(path.join(outOf(m), 'run.json'), JSON.stringify(Object.assign({}, run, {mode: m, cmds: run.cmds.filter(x => !x.mode || x.mode === m)}), null, 1));
  log('done', run.wall_s, 's, page errors', errors.length);
  if (o.video) { await ctx.close(); try { run.video = await page.video().path(); run.video_t0_epoch_ms = vidT0; fs.writeFileSync(path.join(o.out, 'run.json'), JSON.stringify(run, null, 1)); } catch (e) {} }
  await browser.close();
})().catch(e => { console.error(e); process.exit(2); });
