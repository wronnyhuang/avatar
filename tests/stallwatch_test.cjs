// stall watchdog (app/stallwatch.js): detection, classification, expected waits and the escalation ladder on synthetic
// timelines, plus a no-false-positive replay of recorded page-stats timelines (numbers only, tests/fixtures/stall_timelines.json).
//   node tests/stallwatch_test.cjs
'use strict';
const path = require('path');
const SW = require(path.join(__dirname, '..', 'app', 'stallwatch.js'));
let fails = 0, passes = 0;
function check(name, ok, detail) {
  if (ok) passes++; else fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (!ok && detail !== undefined ? '  — ' + JSON.stringify(detail) : ''));
}

/* a simulated page: 24 fps source, a 250 ms watchdog tick; `plan(t)` (seconds) returns what is happening at t:
   {stage, detail, hidden, live, video, recv, dec, draw, same} — recv/dec/draw: frames flow through that stage (bool);
   same: the drawn picture is identical */
function run(plan, secs, cfg) {
  const calls = [], reports = [], alerts = [];
  const actions = {};
  for (const k of ['keyframe', 'resync', 'reopen', 'hub']) actions[k] = st => calls.push([k, +(T / 1000).toFixed(2), st.cls]);
  actions.report = r => reports.push(r);
  actions.alert = (on, st) => alerts.push([on, +(T / 1000).toFixed(2), st.cls]);
  const w = SW.create({cfg, actions});
  const c = {recv: 0, dec: 0, drawn: 0};
  let T = 0, phase = 0, sig = null;
  for (; T <= secs * 1000; T += 250) {
    const p = Object.assign({live: true, stage: 'live', recv: true, dec: true, draw: true}, plan(T / 1000) || {});
    if (p.recv) c.recv += 6;
    if (p.recv && p.dec) c.dec += 6;
    if (p.recv && p.dec && p.draw) { c.drawn += 6; if (!p.same) phase++; }
    sig = new Uint8Array(576); for (let i = 0; i < 576; i++) sig[i] = (i * 7 + phase * 3) & 255;
    w.tick(T, {live: p.live, stage: p.stage, detail: p.detail || '', hidden: !!p.hidden, video: p.video,
      recv: c.recv, dec: c.dec, drawn: c.drawn, sig: p.hidden ? null : sig});
  }
  w.end(T);
  return {calls, reports, alerts, st: w.stats(T), rec: w.records()};
}

// 1. healthy: no stalls, no actions
{
  const r = run(() => ({}), 120);
  check('healthy 120 s: no stall, no action', r.st.stalls === 0 && r.calls.length === 0, r.st);
}
// 2. frames stop arriving at 20 s for 8 s (hub stops forwarding) -> no_frames, ladder, recovered
{
  const r = run(t => (t >= 20 && t < 28 ? {recv: false} : {}), 60);
  const s = r.rec.stalls[0] || {};
  check('no_frames: one stall', r.st.stalls === 1 && s.cls === 'no_frames', r.rec.stalls);
  check('no_frames: detected 3 s after the last frame', s.detect_ms >= 3000 && s.detect_ms <= 3250, s);
  check('no_frames: keyframe, resync, reopen in order, no hub (recovered at 8 s)', JSON.stringify(r.calls.map(x => x[0])) === '["keyframe","resync","reopen"]', r.calls);
  check('no_frames: recovered, fixed_by = the last step, ~8 s', s.outcome === 'recovered' && s.fixed_by === 'reopen' && s.dur_ms >= 7750 && s.dur_ms <= 8250, s);
  check('no_frames: reports open, one update (every 5 s), close', r.reports.map(x => x.ev).join() === 'open,update,close', r.reports.map(x => x.ev));
  check('stats: stall_ms / stalls / longest / recoveries', r.st.stalls === 1 && r.st.recoveries === 1 && Math.abs(r.st.longest_stall_ms - s.dur_ms) < 1 && r.st.stall_ms === r.st.longest_stall_ms, r.st);
}
// 3. frames arrive but never decode (decoder wedged / waiting for a keyframe)
{
  const r = run(t => (t >= 10 && t < 16 ? {dec: false} : {}), 40);
  const s = r.rec.stalls[0] || {};
  check('not_decoded: classified', s.cls === 'not_decoded', r.rec.stalls);
  check('not_decoded: the first step is a keyframe with the class (page resets the decoder)', r.calls[0] && r.calls[0][0] === 'keyframe' && r.calls[0][2] === 'not_decoded', r.calls);
  check('not_decoded: recovered by resync', s.fixed_by === 'resync', s);
}
// 4. decoded but never drawn (every frame judged late: clock model wrong)
{
  const r = run(t => (t >= 10 && t < 30 ? {draw: false} : {}), 60);
  const s = r.rec.stalls[0] || {};
  check('not_drawn: classified', s.cls === 'not_drawn', r.rec.stalls);
  check('not_drawn: alert at 10 s of stall, cleared on recovery', r.alerts.length === 2 && r.alerts[0][0] === true && Math.abs(r.alerts[0][1] - 20) <= 0.25 && r.alerts[1][0] === false, r.alerts);
  check('not_drawn: the hub step ran once (7 s after detection)', r.calls.filter(x => x[0] === 'hub').length === 1 && Math.abs(r.calls.find(x => x[0] === 'hub')[1] - 20) <= 0.25, r.calls);
  check('not_drawn: an update report at the alert', r.reports.some(x => x.ev === 'update' && x.alerted), r.reports);
}
// 5. drawn but identical (renderer sends the same picture)
{
  const r = run(t => (t >= 10 && t < 25 ? {same: true} : {}), 40);
  const s = r.rec.stalls[0] || {};
  check('static: classified after 8 s of an identical picture', s.cls === 'static' && s.detect_ms >= 8000 && s.detect_ms <= 8250, s);
  check('static: recovered when the picture changes', s.outcome === 'recovered', s);
  check('static: its own ladder (keyframe, then the hub 2 s later)', JSON.stringify(r.calls.map(x => x[0])) === '["keyframe","hub"]' && Math.abs(r.calls[1][1] - r.calls[0][1] - 2) <= 0.25, r.calls);
}
// 5b. the hub's own re-attach is recorded in the open stall and named as the fix
{
  const w = SW.create({actions: {}});
  let c = 0;
  for (let T = 0; T <= 20000; T += 250) {
    const flowing = T < 5000 || T >= 12000;
    if (flowing) c += 6;
    if (T === 11000) w.hubAction(T, 'reattach', true);
    w.tick(T, {live: true, stage: 'live', recv: c, dec: c, drawn: c});
  }
  const s = w.records().stalls[0] || {};
  check('hubAction: fixed_by hub:reattach', s.fixed_by === 'hub:reattach' && s.steps.includes('hub:reattach'), s);
  check('hubAction: no page step in the 3 s after it (reopen at 9.75 s moved past the recovery)', !s.steps.includes('reopen'), s);
}
// 6. engine init during a character switch: 50 s with no frames is an expected wait, not a stall
{
  const r = run(t => (t >= 10 && t < 60 ? {stage: 'switching', detail: 'init: gekkie 1280x720 rt', recv: false} : (t >= 60 && t < 62 ? {recv: false} : {})), 80);
  check('switch: no stall, no action', r.st.stalls === 0 && r.calls.length === 0, {st: r.st, calls: r.calls});
  const w = r.rec.waits.find(x => x.reason === 'switch');
  check('switch: wait recorded with its duration', w && Math.abs(w.ms - 50000) <= 250, r.rec.waits);
  const f = r.rec.waits.find(x => x.reason === 'first_frame');
  check('switch: the first-frame wait after it recorded (~2 s)', f && f.ms >= 1750 && f.ms <= 2500, r.rec.waits);
}
// 7. cold box start then engine load
{
  const r = run(t => (t < 40 ? {stage: 'warming', detail: 'container RUNNING; render server not answering /health yet', recv: false}
    : t < 70 ? {stage: 'warming', detail: 'loading Gekkie', recv: false} : {}), 90);
  check('box + engine: no stall', r.st.stalls === 0 && r.calls.length === 0, r.st);
  check('box + engine: waits box ~40 s, engine ~30 s', Math.abs((r.st.wait_ms.box || 0) - 40000) <= 250 && Math.abs((r.st.wait_ms.engine || 0) - 30000) <= 250, r.st.wait_ms);
}
// 8. hidden page: nothing drawn for 30 s
{
  const r = run(t => (t >= 10 && t < 40 ? {hidden: true, draw: false} : {}), 60);
  check('hidden: no stall', r.st.stalls === 0 && r.calls.length === 0, r.st);
  check('hidden: wait recorded', Math.abs((r.st.wait_ms.hidden || 0) - 30000) <= 250, r.st.wait_ms);
}
// 9. no session
{
  const r = run(() => ({live: false, recv: false}), 30);
  check('no session: nothing', r.st.stalls === 0 && r.calls.length === 0 && r.rec.waits.length === 0, r);
}
// 10. live but the first frame never comes -> stall after first_ms
{
  const r = run(t => (t < 20 ? {recv: false} : {}), 40);
  const s = r.rec.stalls[0] || {};
  check('never a first frame: no_frames after first_ms (6 s)', s.cls === 'no_frames' && s.detect_ms >= 6000 && s.detect_ms <= 6250, s);
}
// 11. the hub says the render link is gone -> straight to the hub step
{
  const r = run(t => (t >= 10 && t < 16 ? {video: false, recv: false} : {}), 30);
  const s = r.rec.stalls[0] || {};
  check('video_lost: immediate hub step only', s.cls === 'video_lost' && r.calls.length === 1 && r.calls[0][0] === 'hub' && Math.abs(r.calls[0][1] - 10) <= 0.25, {s, calls: r.calls});
  check('video_lost: recovered by hub', s.fixed_by === 'hub' && s.outcome === 'recovered', s);
}
// 12. a long, unrecovered stall: the hub step repeats at most hub_max times, nothing else loops
{
  const r = run(t => (t >= 10 ? {recv: false} : {}), 200);
  const hubs = r.calls.filter(x => x[0] === 'hub');
  check('long stall: hub step 3 times, 30 s apart', hubs.length === 3 && Math.abs(hubs[1][1] - hubs[0][1] - 30) <= 0.25, hubs);
  check('long stall: 6 actions in total', r.calls.length === 6, r.calls);
  check('long stall: closed as session_end, counted unrecovered', r.rec.stalls[0].outcome === 'session_end' && r.st.unrecovered === 1, r.rec.stalls);
  check('long stall: periodic update reports', r.reports.filter(x => x.ev === 'update').length >= 30, r.reports.length);
}
// 13. a flapping stream: budget caps the actions
{
  const r = run(t => ((t % 8) >= 2 && (t % 8) < 5.75 ? {recv: false} : {}), 120, {budget_per_min: 4});
  check('flapping: actions limited to the budget', r.calls.length <= 4 * 3 && r.st.rate_limited > 0, {calls: r.calls.length, st: r.st});
}
// 14. a stall interrupted by hiding the page is closed as a wait, not counted unrecovered
{
  const r = run(t => (t >= 10 && t < 15 ? {recv: false} : t >= 15 && t < 30 ? {hidden: true, recv: false} : {}), 50);
  check('stall then hidden: closed as wait:hidden', r.rec.stalls[0] && r.rec.stalls[0].outcome === 'wait:hidden' && r.st.unrecovered === 0, r.rec.stalls);
}
// 15. waitReason
check('waitReason', SW.waitReason({live: false}) === 'no_session' && SW.waitReason({live: true, hidden: true}) === 'hidden' &&
  SW.waitReason({live: true, stage: 'warming', detail: 'starting the render box'}) === 'box' &&
  SW.waitReason({live: true, stage: 'warming', detail: 'loading Muse'}) === 'engine' && SW.waitReason({live: true, stage: 'live'}) === null);
check('sigDiff', SW.sigDiff(new Uint8Array([1, 2]), new Uint8Array([1, 2])) === 0 && SW.sigDiff(new Uint8Array([0, 0]), new Uint8Array([2, 4])) === 3);

// 16. recorded healthy timelines (2-s page stats of real sessions, numbers only): replay at 250 ms, no stall may fire.
//     Old stats carry no visibility flag: a window with nothing drawn but frames decoded is the hidden page (rAF stops;
//     proved on the recordings by the recorder's own frame gaps) and is replayed as hidden.
{
  const TL = require('./fixtures/stall_timelines.json');
  let windows = 0, falsePos = 0;
  for (const s of TL) {
    const st = s.stats, stage = t => { let x = ['warming', '']; for (const e of s.status) if (e[0] <= t) x = [e[1], e[2]]; return x; };
    const w = SW.create({actions: {}});
    let recv = 0, dec = 0, drawn = 0, phase = 0;
    for (let i = 1; i < st.length; i++) {
      const [t0, , , in0] = st[i - 1], [t1, fps, decf, in1] = st[i];
      const n = Math.max(1, Math.round((t1 - t0) / 0.25)); windows++;
      for (let k = 1; k <= n; k++) {
        const t = t0 + (t1 - t0) * k / n, [sg, det] = stage(t);
        recv = in0 + (in1 - in0) * k / n; dec += decf * (t1 - t0) / n; const dd = fps * (t1 - t0) / n; drawn += dd; if (dd > 0) phase++;
        const sig = new Uint8Array(576); for (let j = 0; j < 576; j++) sig[j] = (j + phase) & 255;
        const hidden = fps === 0 && decf > 0;
        w.tick(t * 1000, {live: true, stage: sg, detail: det === 'box' ? 'render box' : '', hidden, recv: Math.floor(recv), dec: Math.floor(dec), drawn: Math.floor(drawn), sig: hidden ? null : sig});
      }
    }
    falsePos += w.stats(st[st.length - 1][0] * 1000).stalls;
  }
  check('recorded healthy timelines: ' + TL.length + ' sessions, ' + windows + ' windows of 2 s, 0 false stalls', falsePos === 0, falsePos);
}

console.log(`\n${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
