/* stallwatch.js — the page's video stall watchdog: detects a stalled or frozen avatar picture, drives an escalating
 * recovery, and reports every stall and every expected no-video wait to the hub.
 *
 * window.AvatarStall = {create, signature, sigDiff, waitReason, DEFAULTS}   (also module.exports, for node tests)
 *
 * SIGNALS (read from the page's own counters on a 250 ms timer, never from the per-frame path): time since a video
 *   frame was RECEIVED (S.st.framesIn), DECODED (S.decOut), DRAWN to the canvas (S.drawn), and since the drawn CONTENT
 *   last changed (a 32x18 grey signature of the canvas, sampled on the same timer; an idle avatar still breathes and
 *   blinks, so a living picture changes in every 250 ms sample: p1 of the change 0.2-0.34 grey levels on recorded runs,
 *   a frozen canvas gives exactly 0).
 * EXPECTED STILLNESS (never a stall, but recorded as a wait with its reason and duration, because the viewer sees no
 *   moving avatar then too): no session; box starting ('box'); character / engine loading ('engine'); voice connecting
 *   ('voice'); character switch ('switch'); page hidden ('hidden'); the first frame after any of these ('first_frame',
 *   up to first_ms).
 * A STALL = live, visible, video expected, and one signal older than detect_ms (first_ms before the first frame):
 *   no_frames (nothing received: renderer, hub or socket), not_decoded (received, not decoded), not_drawn (decoded,
 *   never drawn: all late / clock), static (drawn, but the picture is identical for static_ms), video_lost (the hub
 *   says the render link is gone).
 * RECOVERY LADDER, from the detection, each step once per stall, all steps under one per-minute budget:
 *   keyframe (+ a fresh decoder when not_decoded) -> resync (flush the reorder buffer, reset the clock / lag, hub
 *   skip-to-live) -> reopen (a separate video socket, for a page that has one; a page without one leaves the step out
 *   of cfg.steps) -> hub (the hub checks its render link and re-attaches it). static: keyframe -> hub; video_lost: hub. The hub step repeats at most hub_max times,
 *   hub_repeat_ms apart. Nothing else loops. The hub's own re-attach (hubAction) is logged into the open stall.
 * ALERT: alert(true) once a stall is alert_ms old (the page puts it on its status line), alert(false) on recovery.
 * REPORT: report({ev: 'open'|'update'|'close', ...}) at detection, at the alert, every update_ms while open, at the end.
 */
'use strict';
(function () {
  const G = typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : {});

  const DEFAULTS = {
    tick_ms: 250,
    detect_ms: 3000,        // > the longest picture gap seen on visible healthy sessions (1.57 s at an ABR size switch)
    first_ms: 6000,         // first frame after 'live': p99 3.8 s over 206 recorded sessions
    static_ms: 8000,        // identical picture while frames flow; healthy runs never hold one 250 ms sample
    static_eps: 0.05,       // mean |grey diff| on the 32x18 signature that counts as "changed"
    steps: [['keyframe', 0], ['resync', 2000], ['reopen', 4500], ['hub', 7000]],
    // per class: an identical picture is the renderer's (nothing on the page or the link can change it): keyframe, then
    // the hub at once; a lost render link is the hub's alone
    ladders: {static: [['keyframe', 0], ['hub', 2000]], video_lost: [['hub', 0]]},
    alert_ms: 10000,        // stall age at which the status line says so (the ladder has run by then)
    update_ms: 5000,
    hub_repeat_ms: 30000, hub_max: 3,
    budget_per_min: 10,
    after_hub_ms: 3000,     // the hub just re-attached the renderer: its first picture needs ~2 s; no page step meanwhile
    max_records: 40,
  };

  /* expected no-video wait for the page state x, or null when video is expected */
  function waitReason(x) {
    if (!x.live) return 'no_session';
    if (x.hidden) return 'hidden';
    const st = x.stage || '';
    if (st === 'warming') return /render box|container|bootstrap|pod|gpu|box/i.test(x.detail || '') ? 'box' : 'engine';
    if (st === 'connecting') return 'voice';
    if (st === 'switching') return 'switch';
    if (st && st !== 'live') return st;
    return null;
  }

  /* 32 x 18 grey signature of a canvas-like source (a canvas, ImageBitmap or VideoFrame); ctx = a small 2d context */
  function signature(src, ctx, w, h) {
    w = w || 32; h = h || 18;
    ctx.drawImage(src, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data, out = new Uint8Array(w * h);
    for (let i = 0, k = 0; k < out.length; i += 4, k++) out[k] = (d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8;
    return out;
  }
  function sigDiff(a, b) {
    if (!a || !b || a.length !== b.length) return Infinity;
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  }
  function sigLuma(a) { if (!a) return null; let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }

  function create(opts) {
    opts = opts || {};
    const C = Object.assign({}, DEFAULTS, opts.cfg || {});
    const act = opts.actions || {};
    const W = {
      t0: null, last: {recv: null, dec: null, drawn: null}, at: {recv: 0, dec: 0, drawn: 0, content: 0},
      sig: null, firstPending: true, base: null, firstWait: null,
      wait: null, waits: [], waitMs: {},
      stall: null, stalls: [], nStalls: 0, stallMs: 0, longest: 0, recoveries: 0, unrecovered: 0, steps: 0,
      budget: [], rateLimited: 0, seq: 0, lastLuma: null,
    };
    const rel = now => Math.round((now - W.t0) / 100) / 10;   // seconds since the watch started (session start)

    function call(name, ...a) { const f = act[name]; if (typeof f === 'function') { try { return f(...a); } catch (e) { /* never break the page */ } } return undefined; }
    function spend(now) {
      while (W.budget.length && now - W.budget[0] > 60000) W.budget.shift();
      if (W.budget.length >= C.budget_per_min) { W.rateLimited++; return false; }
      W.budget.push(now); return true;
    }
    function baseline(now) {
      W.at.recv = W.at.dec = W.at.drawn = W.at.content = now; W.base = now;
    }
    function openWait(now, reason) {
      if (W.wait && W.wait.reason === reason) return;
      closeWait(now);
      W.wait = {reason, start: now};
    }
    function closeWait(now) {
      const w = W.wait; if (!w) return;
      W.wait = null;
      const ms = Math.round(now - w.start);
      if (w.reason === 'no_session' || ms < C.tick_ms) return;
      W.waitMs[w.reason] = (W.waitMs[w.reason] || 0) + ms;
      W.waits.push({reason: w.reason, t0_s: rel(w.start), ms});
      if (W.waits.length > C.max_records) W.waits.shift();
    }
    function record(st, now) {
      return {id: st.id, cls: st.cls, t0_s: rel(st.start), dur_ms: Math.round(now - st.start), detect_ms: Math.round(st.detected - st.start),
        steps: st.steps.map(s => s[0]), fixed_by: st.fixed_by || null, outcome: st.outcome || 'open', alerted: st.alerted,
        luma: st.luma === null ? null : Math.round(st.luma)};
    }
    function openStall(now, cls, start) {
      const st = {id: ++W.seq, cls, start, detected: now, shift: 0, steps: [], next: 0, hubN: 0, lastHub: 0, alerted: false, lastUpd: now,
        luma: W.lastLuma};
      W.stall = st; W.nStalls++;
      call('log', 'stall ' + cls + ' detected after ' + Math.round(now - start) + ' ms');
      call('report', Object.assign({ev: 'open'}, record(st, now)));
      return st;
    }
    function closeStall(now, outcome) {
      const st = W.stall; if (!st) return;
      W.stall = null;
      st.outcome = outcome;
      const ms = now - st.start;
      W.stallMs += ms; W.longest = Math.max(W.longest, ms);
      if (outcome === 'recovered') { W.recoveries++; st.fixed_by = st.steps.length ? st.steps[st.steps.length - 1][0] : 'self'; }
      else if (!/^wait:/.test(outcome)) W.unrecovered++;
      if (st.alerted) call('alert', false, st);
      const r = record(st, now);
      W.stalls.push(r); if (W.stalls.length > C.max_records) W.stalls.shift();
      call('log', 'stall ' + st.cls + ' ' + outcome + ' after ' + Math.round(ms) + ' ms' + (st.fixed_by ? ' by ' + st.fixed_by : ''));
      call('report', Object.assign({ev: 'close'}, r));
      baseline(now);
    }
    function doStep(st, name, now) {
      if (!spend(now)) { st.steps.push([name + ':rate_limited', Math.round(now - st.start)]); return; }
      st.steps.push([name, Math.round(now - st.start)]); W.steps++;
      if (name === 'hub') { st.hubN++; st.lastHub = now; }
      call(name, st);
    }
    function escalate(st, now) {
      const ladder = (C.ladders && C.ladders[st.cls]) || C.steps;
      if (st.hubFixAt && now - st.hubFixAt < C.after_hub_ms) { st.shift += C.tick_ms; return; }   // shift the ladder
      while (st.next < ladder.length && now - st.detected - st.shift >= ladder[st.next][1]) { doStep(st, ladder[st.next][0], now); st.next++; }
      if (st.next >= ladder.length && st.hubN > 0 && st.hubN < C.hub_max && now - st.lastHub >= C.hub_repeat_ms) doStep(st, 'hub', now);
      if (!st.alerted && now - st.start >= C.alert_ms) {
        st.alerted = true; call('alert', true, st);
        call('report', Object.assign({ev: 'update'}, record(st, now))); st.lastUpd = now;
      } else if (now - st.lastUpd >= C.update_ms) { st.lastUpd = now; call('report', Object.assign({ev: 'update'}, record(st, now))); }
    }

    /* x = {live, stage, detail, hidden, video (false = the hub lost the render link), recv, dec, drawn (counters),
       sig (Uint8Array signature of the drawn picture, or null when not sampled)} */
    function tick(now, x) {
      if (W.t0 === null) W.t0 = now;
      // counters -> last-change times
      let drewNow = false, changed = false;
      for (const k of ['recv', 'dec', 'drawn']) {
        if (x[k] !== undefined && x[k] !== W.last[k]) {
          if (W.last[k] !== null && x[k] > W.last[k]) { W.at[k] = now; if (k === 'drawn') drewNow = true; }
          W.last[k] = x[k];
        }
      }
      if (x.sig) {
        W.lastLuma = sigLuma(x.sig);
        if (!W.sig || sigDiff(x.sig, W.sig) >= C.static_eps) { changed = true; W.at.content = now; }
        W.sig = x.sig;
      }
      if (drewNow) W.firstPending = false;
      // expected waits
      const wr = waitReason(x);
      if (wr) {
        if (W.stall) closeStall(now, 'wait:' + wr);
        openWait(now, wr);
        W.firstPending = true; W.sig = null;
        return;
      }
      if (W.wait) { closeWait(now); baseline(now); W.firstPending = true; }
      if (W.base === null) baseline(now);
      if (W.firstPending && !W.stall) {      // the first picture after a wait: a short expected wait of its own
        if (now - W.base < C.first_ms) { if (!W.firstWait) W.firstWait = W.base; return; }
      }
      if (W.firstWait && !W.firstPending) {   // it came in time
        const ms = Math.round(W.at.drawn - W.firstWait);
        if (ms >= C.tick_ms) { W.waitMs.first_frame = (W.waitMs.first_frame || 0) + ms; W.waits.push({reason: 'first_frame', t0_s: rel(W.firstWait), ms}); if (W.waits.length > C.max_records) W.waits.shift(); }
        W.firstWait = null;
      }
      const st0 = W.stall;
      if (st0) {
        const ok = st0.cls === 'video_lost' ? (x.video !== false && drewNow)
          : st0.cls === 'static' ? (drewNow && changed) : drewNow;
        if (ok) { closeStall(now, 'recovered'); W.firstWait = null; return; }
        escalate(st0, now);
        return;
      }
      W.firstWait = null;
      const thr = W.firstPending ? C.first_ms : C.detect_ms;
      let cls = null, start = 0;
      if (x.video === false) { cls = 'video_lost'; start = now; }
      else if (now - W.at.recv >= thr) { cls = 'no_frames'; start = W.at.recv; }
      else if (now - W.at.dec >= thr) { cls = 'not_decoded'; start = W.at.dec; }
      else if (now - W.at.drawn >= thr) { cls = 'not_drawn'; start = W.at.drawn; }
      else if (C.static_ms > 0 && x.sig !== undefined && now - W.at.content >= C.static_ms) { cls = 'static'; start = W.at.content; }
      if (cls) escalate(openStall(now, cls, start), now);
    }

    /* the hub's own recovery action (it re-attached the render link, on its own watch or on the page's ask) joins the
       open stall's steps, so 'fixed_by' names it when the picture comes back after it */
    function hubAction(now, name, ok) {
      const st = W.stall; if (!st) return;
      st.steps.push(['hub:' + String(name || 'action') + (ok === false ? ':failed' : ''), Math.round(now - st.start)]);
      if (ok !== false) st.hubFixAt = now;
    }
    function stats(now) {
      const st = W.stall, open = st ? now - st.start : 0;
      return {stall_ms: Math.round(W.stallMs + open), stalls: W.nStalls, longest_stall_ms: Math.round(Math.max(W.longest, open)),
        recoveries: W.recoveries, unrecovered: W.unrecovered, steps: W.steps, rate_limited: W.rateLimited,
        open: st ? {cls: st.cls, ms: Math.round(open), steps: st.steps.map(s => s[0])} : null,
        wait: W.wait ? {reason: W.wait.reason, ms: Math.round(now - W.wait.start)} : null,
        wait_ms: Object.assign({}, W.waitMs)};
    }
    function end(now) { if (W.stall) closeStall(now, 'session_end'); closeWait(now); }
    return {tick, stats, end, hubAction, records: () => ({stalls: W.stalls.slice(), waits: W.waits.slice()}), cfg: C, _W: W};
  }

  const api = {create, signature, sigDiff, sigLuma, waitReason, DEFAULTS};
  G.AvatarStall = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
