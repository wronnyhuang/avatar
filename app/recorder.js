/* recorder.js — records every avatar session on the phone and uploads it to the app server.
 *
 * window.AvatarRec = {begin(opts), end(reason), note({character}), setMic(stream), stats(), get active}
 * window.AvatarRecLib = {createUploader, createRecorder, pickMime, fitRect, coverRect}   (also module.exports)
 *
 * WHAT: the stage exactly as shown — the avatar video canvas + the user's self-view video in its corner (mirrored, rounded,
 *   as the page draws it) + the placeholder text while it connects — composited onto an off-screen canvas at the
 *   stage's displayed size (CSS px x min(dpr, 1.5), capped at 960 px wide) at 30 fps (a steady due-time
 *   cadence, half a refresh of slack, so a 60 / 90 / 120 Hz display all give 30 — the old ">= 40 ms since the last
 *   draw" rule landed on every 3rd refresh of a 60 Hz phone = 20 fps; each draw is pushed with requestFrame(), so the
 *   recording holds exactly the frames drawn); audio = a WebAudio mix (same
 *   AudioContext) of the user's mic stream and the avatar's voice exactly as the user hears it (the playout's output node, after
 *   the A/V delay line). MediaRecorder: webm VP8/VP9 + Opus (Chrome/Android), mp4 (Safari).
 * HOW IT REACHES THE HUB (app/recorder_hub.py + recstore.py): POST api/rec/start -> rid; MediaRecorder timeslice =
 *   one CHUNK every 3 s, numbered 0, 1, 2 … (chunk 0 = the container header) -> POST api/rec/chunk?rid&seq, ONE at a
 *   time, low priority, each with a timeout; failures retry with back-off (1 s doubling to 30 s, jitter) forever while
 *   the page lives; after a failure GET api/rec/status skips chunks the server already has (resumable by sequence
 *   number); 410 = the server closed the rid (idle-assembled after a long outage) -> a NEW part (fresh MediaRecorder =
 *   fresh header, same group). Backlog bounded (48 MB): the oldest unsent chunks (never chunk 0) are dropped and named
 *   in 'end'. 'end' = POST api/rec/end {n, dropped, duration_ms, reason, stats} once the backlog drained; at pagehide
 *   a sendBeacon 'end' (the server also assembles on its own after ~2.5 min without chunks — a phone that dies).
 * COST ON THE SESSION: nothing on the voice/render path (own rAF callback, ~1 ms canvas copies on a GPU canvas —
 *   stats().draw_ms; a slow canvas lowers the frame rate to keep drawing under 250 ms/s (an Android phone drew a
 *   960x540 composite in 5-8 ms: the old 120 ms/s budget held it at ~13 fps), >= 8 fps;
 *   the upload is one low-priority request per 3 s); OPT-IN: only with ?rec=1 (off by default). Visible: a small red "● rec" after the live status line (no new UI).
 */
'use strict';
(function () {
  const G = typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : {});

  const MIMES = ['video/webm;codecs=vp8,opus', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=h264,opus', 'video/webm',
    'video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=avc1,mp4a', 'video/mp4'];
  function pickMime(MR, prefer) {
    if (!MR || typeof MR.isTypeSupported !== 'function') return '';
    const list = prefer ? [prefer].concat(MIMES) : MIMES;
    for (const m of list) { try { if (MR.isTypeSupported(m)) return m; } catch (e) {} }
    return '';
  }
  /* the cheapest supported codec: MediaCapabilities 'record' says which encoders are hardware (powerEfficient) —
     a hardware VP8 / H.264 / VP9 encoder costs the phone almost nothing; else the first supported (VP8) */
  async function chooseMime(MR, prefer, cfg) {
    const first = pickMime(MR, prefer);
    const mc = G.navigator && G.navigator.mediaCapabilities;
    if (prefer || !mc || !mc.encodingInfo || !MR || !MR.isTypeSupported) return first;
    const cands = MIMES.filter(m => /codecs=/.test(m)).filter(m => { try { return MR.isTypeSupported(m); } catch (e) { return false; } });
    const vtype = m => m.replace(/codecs=([^,]+).*$/, (x, v) => 'codecs=' + (v === 'h264' ? 'avc1.42E01E' : v));
    try {
      const infos = await Promise.race([Promise.all(cands.map(m => mc.encodingInfo({type: 'record', video: {contentType: vtype(m),
        width: cfg.w, height: cfg.h, bitrate: cfg.bitrate, framerate: cfg.fps}}).then(i => ({m, i}), () => ({m, i: null})))),
        new Promise(res => setTimeout(() => res(null), 400))]);
      const hw = infos && infos.find(x => x.i && x.i.supported && x.i.powerEfficient);
      return hw ? hw.m : first;
    } catch (e) { return first; }
  }
  // object-fit: contain / cover of a (sw x sh) source in a (dw x dh) box -> [sx, sy, sw, sh, dx, dy, dw, dh] for drawImage
  function fitRect(sw, sh, dx, dy, dw, dh) {
    if (!(sw > 0 && sh > 0)) return null;
    const s = Math.min(dw / sw, dh / sh), w = sw * s, h = sh * s;
    return [0, 0, sw, sh, dx + (dw - w) / 2, dy + (dh - h) / 2, w, h];
  }
  function coverRect(sw, sh, dx, dy, dw, dh) {
    if (!(sw > 0 && sh > 0)) return null;
    const s = Math.max(dw / sw, dh / sh), cw = dw / s, ch = dh / s;
    return [(sw - cw) / 2, (sh - ch) / 2, cw, ch, dx, dy, dw, dh];
  }
  const pct = (xs, p) => { if (!xs.length) return null; const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };
  const r2 = x => Math.round(x * 100) / 100;

  // ------------------------------------------------------------------------------------------- the uploader
  /* createUploader({fetch, url(path, params), headers, maxBytes, sleep(ms), now(), random(), log, onClosed(),
   *   timeoutMs(bytes), beacon(url, blob)})
   *   .setRid(rid)  (chunks pushed before the rid is known wait)  .push(seq, blob)  .finish(info) -> Promise
   *   .beaconEnd(info)  .stats()  .closed  .idle()  */
  function createUploader(o) {
    const F = o.fetch || (G.fetch && G.fetch.bind(G));
    const sleep = o.sleep || (ms => new Promise(r => setTimeout(r, ms)));
    const now = o.now || (() => Date.now());
    const rnd = o.random || Math.random;
    const log = o.log || (() => {});
    const U = {rid: null, q: [], bytes: 0, n: 0, sent: 0, sentBytes: 0, dup: 0, retries: 0, fails: 0, dropped: [],
      closed: false, ended: false, running: false, lastErr: '', maxBytes: o.maxBytes || 48e6, maxQ: 0, waiters: [],
      wake: null, upMs: []};
    const tmo = o.timeoutMs || (bytes => 15000 + bytes / 25);     // 15 s + 40 ms per KB (~200 kbit/s floor)
    function signal(ms) {
      ms = Math.max(1, Math.round(ms));
      if (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) return AbortSignal.timeout(ms);
      if (typeof AbortController === 'undefined') return undefined;
      const c = new AbortController(); setTimeout(() => c.abort(), ms); return c.signal;
    }
    function push(seq, blob) {
      if (U.closed || U.ended) return false;
      const size = blob && (blob.size != null ? blob.size : blob.byteLength) || 0;
      if (!size) return false;
      U.q.push({seq, blob, size}); U.bytes += size; U.n = Math.max(U.n, seq + 1);
      while (U.bytes > U.maxBytes && U.q.length > 1) {          // bounded memory: drop the oldest unsent (never chunk 0)
        const i = U.q[0].seq === 0 ? 1 : 0, d = U.q.splice(i, 1)[0];
        if (!d) break;
        U.bytes -= d.size; U.dropped.push(d.seq);
      }
      U.maxQ = Math.max(U.maxQ, U.bytes);
      kick();
      return true;
    }
    function kick() { if (U.wake) { const w = U.wake; U.wake = null; w(); } if (!U.running) pump(); }
    function setRid(rid) { U.rid = rid; kick(); }
    async function post(path, params, body, ctype, ms) {
      const r = await F(o.url(path, params), {method: 'POST', headers: Object.assign({}, o.headers || {}, ctype ? {'content-type': ctype} : {}),
        body, signal: signal(ms), cache: 'no-store', priority: 'low'});
      let j = null; try { j = await r.json(); } catch (e) {}
      return {status: r.status, ok: r.ok, j};
    }
    async function skipHad() {          // after a failure: which chunks did the server get anyway (response lost)?
      try {
        const r = await F(o.url('api/rec/status', {rid: U.rid}), {method: 'GET', headers: o.headers || {}, signal: signal(8000), cache: 'no-store'});
        if (r.status === 410) return 'closed';
        if (!r.ok) return;
        const j = await r.json(), have = new Set((j && j.have) || []);
        for (let i = U.q.length - 1; i >= 0; i--) if (have.has(U.q[i].seq)) { U.bytes -= U.q[i].size; U.sent++; U.q.splice(i, 1); }
      } catch (e) {}
    }
    function closedNow(why) {
      if (U.closed) return;
      U.closed = true; U.lastErr = why || 'closed'; U.q = []; U.bytes = 0;
      log('rec: hub closed ' + U.rid + ' (' + U.lastErr + ')');
      for (const w of U.waiters.splice(0)) w();
      if (o.onClosed) { try { o.onClosed(); } catch (e) {} }
    }
    async function pump() {
      if (U.running) return;
      U.running = true;
      try {
        let back = 1000;
        while (!U.closed) {
          if (!U.rid || !U.q.length) {
            for (const w of U.waiters.splice(0)) w();
            await new Promise(r => { U.wake = r; });
            continue;
          }
          const c = U.q[0], t0 = now();
          let res = null, e1 = null;
          try { res = await post('api/rec/chunk', {rid: U.rid, seq: c.seq}, c.blob, 'application/octet-stream', tmo(c.size)); }
          catch (e) { e1 = e; }
          if (res && res.status === 410) { closedNow('closed'); break; }
          if (res && res.ok) {
            if (U.q[0] === c) { U.q.shift(); U.bytes -= c.size; }
            U.sent++; U.sentBytes += c.size; if (res.j && res.j.dup) U.dup++;
            U.upMs.push(now() - t0); if (U.upMs.length > 100) U.upMs.shift();
            U.fails = 0; back = 1000;
            continue;
          }
          if (res && (res.status === 413 || res.status === 400)) {     // the server will never take this one: skip it
            if (U.q[0] === c) { U.q.shift(); U.bytes -= c.size; U.dropped.push(c.seq); }
            U.lastErr = 'HTTP ' + res.status; continue;
          }
          U.retries++; U.fails++;
          U.lastErr = e1 ? String(e1 && e1.name === 'TimeoutError' ? 'timeout' : (e1 && e1.message) || e1).slice(0, 60) : 'HTTP ' + res.status;
          await sleep(back * (0.75 + 0.5 * rnd()));
          back = Math.min(30000, back * 2);
          if (U.fails >= 2 && (await skipHad()) === 'closed') { closedNow('closed'); break; }
        }
      } finally { U.running = false; for (const w of U.waiters.splice(0)) w(); }
    }
    function idle() { return U.closed || (!U.q.length && !!U.rid); }
    /* wait for the backlog to drain (no deadline while the page lives, or opts.maxWaitMs), then POST 'end' */
    async function finish(info, opts) {
      const maxWait = opts && opts.maxWaitMs;
      const t0 = now();
      while (U.rid && !idle() && !(maxWait && now() - t0 > maxWait)) {
        let tm; await new Promise(r => { U.waiters.push(r); tm = setTimeout(r, 1000); }); clearTimeout(tm);
      }
      if (U.closed || !U.rid || U.ended) { U.ended = true; if (!U.rid) { U.q = []; U.bytes = 0; } return {ok: false, closed: U.closed, norid: !U.rid}; }
      U.ended = true;
      const body = Object.assign({n: U.n, dropped: U.dropped.concat(U.q.map(c => c.seq))}, info || {});
      for (let i = 0; i < 4; i++) {
        try {
          const r = await post('api/rec/end', {rid: U.rid}, JSON.stringify(body), 'application/json', 8000);
          if (r.ok || r.status === 410) return {ok: r.ok};
        } catch (e) {}
        await sleep(1000 * (i + 1));
      }
      return {ok: false};
    }
    function beaconEnd(info) {
      if (!U.rid || U.ended || U.closed) return false;
      U.ended = true;
      const body = Object.assign({n: U.n, dropped: U.dropped.concat(U.q.map(c => c.seq))}, info || {});
      const u = o.url('api/rec/end', {rid: U.rid, n: U.n, reason: (info && info.reason) || 'pagehide'});
      try {
        if (o.beacon) return o.beacon(u, JSON.stringify(body));
        if (G.navigator && G.navigator.sendBeacon) return G.navigator.sendBeacon(u, new Blob([JSON.stringify(body)], {type: 'text/plain'}));
      } catch (e) {}
      return false;
    }
    function stats() {
      return {rid: U.rid, n: U.n, sent: U.sent, sent_kb: Math.round(U.sentBytes / 1024), queued: U.q.length,
        queued_kb: Math.round(U.bytes / 1024), max_queue_kb: Math.round(U.maxQ / 1024), retries: U.retries, dup: U.dup,
        dropped: U.dropped.length, closed: U.closed, err: U.lastErr, up_ms_p50: pct(U.upMs, 0.5)};
    }
    return {push, setRid, finish, beaconEnd, stats, idle, get closed() { return U.closed; }, get rid() { return U.rid; }, _U: U};
  }

  // ------------------------------------------------------------------------------------------- the recorder
  /* createRecorder(deps) — deps (tests) override the browser: {document, MediaRecorder, MediaStream, raf, caf, now,
   *   fetch, navigator, devicePixelRatio, timeslice, fps, maxW, log} */
  function createRecorder(deps) {
    const X = deps || {};
    const doc = X.document || G.document;
    const now = X.now || (() => (G.performance ? G.performance.now() : Date.now()));
    const raf = X.raf || (G.requestAnimationFrame ? G.requestAnimationFrame.bind(G) : (f => setTimeout(() => f(now()), 40)));
    const caf = X.caf || (G.cancelAnimationFrame ? G.cancelAnimationFrame.bind(G) : clearTimeout);
    const log = X.log || ((...a) => { try { console.info(...a); } catch (e) {} });
    const TIMESLICE = X.timeslice || 3000, MAXW = X.maxW || 960, BUDGET = X.budgetMsPerS || 250;
    let FPS = X.fps || 30;
    const R = {active: false, off: '', parts: 0, group: '', lastEnd: null, cur: null, totals: {chunks: 0, kb: 0, parts: 0}};

    function styleOnce() {
      if (!doc || R.styled || !doc.head) return; R.styled = true;
      const s = doc.createElement('style');
      s.textContent = 'body.av-rec #status.live::after{content:"  \\25CF  rec";color:#e5534b;font-size:11px;letter-spacing:.5px;white-space:pre}';
      doc.head.appendChild(s);
    }
    function setDot(on) { try { if (doc && doc.body) doc.body.classList.toggle('av-rec', !!on); } catch (e) {} }

    function begin(o) {
      if (R.cur) end('restart');
      const Q = new URLSearchParams((G.location && G.location.search) || '');
      const want = Q.get('rec');
      const MR = X.MediaRecorder || G.MediaRecorder, MS = X.MediaStream || G.MediaStream, nav = X.navigator || G.navigator || {};
      if (want !== '1' && !o.force) { R.off = 'off (add ?rec=1 to record)'; return null; }   // opt-in: nothing is recorded by default
      if (!MR || !MS || !o.stage || !o.ctx || !doc) { R.off = 'unsupported (no MediaRecorder)'; return null; }
      const prefMime = Q.get('recmime') || '';
      if (!X.fps && +Q.get('recfps') > 0) FPS = Math.min(60, Math.max(5, +Q.get('recfps')));
      const cv = doc.createElement('canvas');
      if (!cv.captureStream && !cv.mozCaptureStream) { R.off = 'unsupported (no canvas capture)'; return null; }
      R.off = '';
      // output size: the stage as displayed (CSS px x dpr capped at 1.5 — sharp on a phone, light to encode), at most
      // MAXW wide, even numbers
      const sr = o.stage.getBoundingClientRect(), dpr = Math.min(1.5, X.devicePixelRatio || G.devicePixelRatio || 1);
      const maxW = +(Q.get('recw') || 0) || MAXW;
      let W = Math.round((sr.width || 640) * dpr), H = Math.round((sr.height || 360) * dpr);
      if (W > maxW) { H = Math.round(H * maxW / W); W = maxW; }
      if (W < 320) { H = Math.round(H * 320 / W); W = 320; }
      W -= W % 2; H -= H % 2;
      cv.width = W; cv.height = H;
      const g = cv.getContext('2d', {alpha: false});
      // captureStream(0) + requestFrame() after every draw = one recorded frame per composite drawn (a rate-capped
      // capture drops a frame whenever two draws land a little closer than 1/fps apart); older engines: rate-capped
      let vstream = null, pushFrame = null;
      if (cv.captureStream) {
        try {
          const s0 = cv.captureStream(0), t0 = s0.getVideoTracks()[0];
          if (t0 && typeof t0.requestFrame === 'function') { vstream = s0; pushFrame = () => { try { t0.requestFrame(); } catch (e) {} }; }
          else { try { s0.getTracks().forEach(t => t.stop()); } catch (e) {} }
        } catch (e) {}
      }
      if (!vstream) vstream = cv.captureStream ? cv.captureStream(FPS) : cv.mozCaptureStream(FPS);
      // audio: the user's mic + the avatar's voice as heard, mixed in the page's own AudioContext (mono)
      const ctx = o.ctx, A = {dest: null, mic: null, micGain: null, voiceGain: null, voiceNode: null};
      try {
        A.dest = ctx.createMediaStreamDestination(); try { A.dest.channelCount = 1; } catch (e) {}
        A.voiceGain = ctx.createGain(); A.voiceGain.gain.value = 1; A.voiceGain.connect(A.dest);
        A.micGain = ctx.createGain(); A.micGain.gain.value = 1; A.micGain.connect(A.dest);
      } catch (e) { A.dest = null; }
      const tracks = vstream.getVideoTracks().concat(A.dest ? A.dest.stream.getAudioTracks() : []);
      const C = {o, mime: pickMime(MR, prefMime), W, H, cv, g, vstream, pushFrame, due: 0, rafDt: 0, lastRaf: 0, A, stream: new MS(tracks), mr: null, up: null, seq: 0, t0: now(), wall0: Date.now(),
        raf: 0, lastDraw: -1e9, lastLayout: -1e9, L: null, draws: [], ewma: 0, frames: 0, ended: false, part: 0, gen: 0,
        bitrate: Math.round(Math.max(300e3, Math.min(1.5e6, W * H * FPS * 0.09)))};
      R.cur = C; R.active = true;
      const cont = R.lastEnd && R.lastEnd.reason === 'reconnect' && now() - R.lastEnd.t < 120000;
      if (!cont || !R.group) { R.group = Math.random().toString(36).slice(2, 10); R.parts = 0; }
      wireAudio(C);
      C.mimeP = chooseMime(MR, prefMime, {w: W, h: H, fps: FPS, bitrate: C.bitrate}).then(m => {
        C.mime = m;
        if (!C.ended) startPart(C);
        log('rec: recording ' + W + 'x' + H + ' ' + (m || 'default') + ' ' + Math.round(C.bitrate / 1000) + ' kbit/s');
      });
      styleOnce(); setDot(true);
      C.raf = raf(t => frame(C, t));
      return {w: W, h: H, mime: C.mime, ready: C.mimeP};
    }

    function wireAudio(C) {
      const A = C.A, o = C.o;
      if (!A.dest) return;
      if (o.mic && !A.mic) { try { A.mic = o.ctx.createMediaStreamSource(o.mic); A.mic.connect(A.micGain); } catch (e) { A.mic = null; } }
      let vn = null; try { vn = typeof o.voiceOut === 'function' ? o.voiceOut() : o.voiceOut; } catch (e) {}
      if (vn !== A.voiceNode) {
        if (A.voiceNode) { try { A.voiceNode.disconnect(A.voiceGain); } catch (e) {} }
        A.voiceNode = vn || null;
        if (vn) { try { vn.connect(A.voiceGain); } catch (e) { A.voiceNode = null; } }
      }
    }
    function unwireAudio(C) {
      const A = C.A;
      if (A.voiceNode) { try { A.voiceNode.disconnect(A.voiceGain); } catch (e) {} A.voiceNode = null; }
      if (A.mic) { try { A.mic.disconnect(); } catch (e) {} A.mic = null; }
      for (const n of [A.micGain, A.voiceGain]) { if (n) { try { n.disconnect(); } catch (e) {} } }
      if (A.dest) { try { A.dest.stream.getTracks().forEach(t => t.stop()); } catch (e) {} }
    }

    // one recording part = one MediaRecorder (its own container header) + one rid + one uploader
    function startPart(C) {
      const MR = X.MediaRecorder || G.MediaRecorder, o = C.o, my = ++C.gen;
      C.part = ++R.parts; C.seq = 0;
      const up = createUploader({fetch: X.fetch || o.fetch, url: o.url, headers: o.headers, log, beacon: X.beacon,
        onClosed: () => { if (R.cur === C && !C.ended && C.gen === my) { log('rec: new part'); try { C.mr && C.mr.stop(); } catch (e) {} startPart(C); } }});
      C.up = up;
      let mr;
      const opt = {videoBitsPerSecond: C.bitrate, audioBitsPerSecond: 48000};
      if (C.mime) opt.mimeType = C.mime;
      try { mr = new MR(C.stream, opt); } catch (e) { try { mr = new MR(C.stream); } catch (e2) { R.off = 'MediaRecorder failed: ' + (e2 && e2.message); stopAll(C); return; } }
      C.mr = mr;
      mr.ondataavailable = ev => { if (C.gen === my && ev.data && ev.data.size) { up.push(C.seq++, ev.data); R.totals.chunks++; R.totals.kb += ev.data.size / 1024; } };
      mr.onerror = ev => { log('rec: MediaRecorder error ' + (ev && ev.error && ev.error.name || '')); if (R.cur === C) end('error'); };
      const meta = {character: o.character || '', character_name: o.characterName || '', voice: o.voice || '', voice_label: o.voiceLabel || '',
        mime: mr.mimeType || C.mime || '', width: C.W, height: C.H, fps: FPS, timeslice_ms: TIMESLICE, group: R.group, part: C.part,
        sid: (o.sid || '').slice(0, 6), client_ts: Date.now(), automated: !!((X.navigator || G.navigator || {}).webdriver),
        ua: ((X.navigator || G.navigator || {}).userAgent || '').slice(0, 120)};
      C.ridP = startRid(C, up, meta, my);
      try { mr.start(TIMESLICE); } catch (e) { R.off = 'MediaRecorder.start failed: ' + (e && e.message); stopAll(C); }
      R.totals.parts++;
    }
    async function startRid(C, up, meta, my) {
      let back = 1000;
      for (let i = 0; i < 12 && C.gen === my; i++) {           // (keeps trying after a quick Stop: the chunks wait for it)
        try {
          const F = X.fetch || C.o.fetch || G.fetch.bind(G);
          const r = await F(C.o.url('api/rec/start'), {method: 'POST', headers: Object.assign({}, C.o.headers || {}, {'content-type': 'application/json'}),
            body: JSON.stringify(meta), cache: 'no-store', signal: (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) ? AbortSignal.timeout(10000) : undefined});
          let j = null; try { j = await r.json(); } catch (e) {}
          if (r.ok && j && j.rid) { up.setRid(j.rid); C.rid = j.rid; log('rec: ' + j.rid); return; }
          if (r.status === 503 || r.status === 403 || r.status === 404 || r.status === 507) {     // recording off / not here / disk full
            R.off = 'hub: ' + ((j && j.error) || ('HTTP ' + r.status)); up.setRid(null); if (R.cur === C) end('off', true); return;
          }
        } catch (e) {}
        await new Promise(res => setTimeout(res, back)); back = Math.min(30000, back * 2);
      }
      if (C.gen === my) { R.off = 'hub unreachable for recording'; if (R.cur === C) end('off', true); }
    }

    // ---- the composite (own rAF callback; ~25 fps)
    function layout(C) {
      const o = C.o, st = o.stage, sr = st.getBoundingClientRect(), k = C.W / Math.max(1, sr.width);
      const rel = el => { const r = el.getBoundingClientRect(); return [(r.left - sr.left) * k, (r.top - sr.top) * k, r.width * k, r.height * k]; };
      const shown = el => { if (!el || !el.isConnected) return false; const cs = G.getComputedStyle ? G.getComputedStyle(el) : {}; return cs.display !== 'none' && cs.visibility !== 'hidden' && +(cs.opacity || 1) > 0.05; };
      const L = {k, canvas: null, vids: [], ph: null};
      const cvs = o.canvas || st.querySelector('canvas');
      if (cvs && shown(cvs)) L.canvas = {el: cvs, r: rel(cvs)};
      for (const v of st.querySelectorAll('video')) {
        const box = v.parentElement && v.parentElement !== st ? v.parentElement : v;
        if (!shown(box) || !shown(v)) continue;
        const cs = G.getComputedStyle ? G.getComputedStyle(box) : {}, vcs = G.getComputedStyle ? G.getComputedStyle(v) : {};
        L.vids.push({el: v, r: rel(box), rad: (parseFloat(cs.borderTopLeftRadius) || 0) * k, bw: (parseFloat(cs.borderTopWidth) || 0) * k,
          bc: cs.borderTopColor || 'rgba(255,255,255,.55)', mirror: /matrix\(-1|scaleX\(-1|scale\(-1/.test(String(vcs.transform || ''))});
      }
      const ph = o.placeholder || st.querySelector('#ph');
      if (ph && shown(ph)) L.ph = {text: (ph.innerText || ph.textContent || '').trim().split(/\n+/).slice(0, 3)};
      C.L = L;
    }
    function rr(g, x, y, w, h, r) {
      if (g.roundRect) { g.roundRect(x, y, w, h, r); return; }
      g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
    }
    function draw(C) {
      const g = C.g, L = C.L, W = C.W, H = C.H;
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      if (L.canvas) {
        const c = L.canvas.el, f = fitRect(c.width, c.height, ...L.canvas.r);
        if (f) g.drawImage(c, ...f);
      }
      if (L.ph && L.ph.text.length) {
        g.fillStyle = '#93a0b0'; g.textAlign = 'center'; g.textBaseline = 'middle';
        const lines = L.ph.text, big = Math.round(H * 0.16), small = Math.max(10, Math.round(H * 0.04));
        let y = H / 2 - (lines.length > 1 ? big * 0.35 : 0);
        lines.forEach((t, i) => { g.font = (i === 0 && lines.length > 1 ? big : small) + 'px system-ui,sans-serif'; g.fillText(t, W / 2, y); y += i === 0 ? big * 0.75 : small * 1.4; });
      }
      for (const v of L.vids) {
        const el = v.el;
        if (!(el.readyState >= 2) || !el.videoWidth) continue;
        const [x, y, w, h] = v.r, f = coverRect(el.videoWidth, el.videoHeight, 0, 0, w, h);
        if (!f) continue;
        g.save(); g.beginPath(); rr(g, x, y, w, h, v.rad); g.clip();
        if (v.mirror) { g.translate(x + w, y); g.scale(-1, 1); } else g.translate(x, y);
        g.drawImage(el, f[0], f[1], f[2], f[3], 0, 0, w, h);
        g.restore();
        if (v.bw > 0) { g.save(); g.lineWidth = v.bw; g.strokeStyle = v.bc; g.beginPath(); rr(g, x + v.bw / 2, y + v.bw / 2, w - v.bw, h - v.bw, Math.max(0, v.rad - v.bw / 2)); g.stroke(); g.restore(); }
      }
    }
    function frame(C, t) {
      if (C.ended) return;
      C.raf = raf(tt => frame(C, tt));
      const tn = now();
      // FPS on any display rate: a due time advanced by the frame gap, drawn on the refresh nearest to it (half a refresh
      // of slack) -> 30 fps on 60 / 90 / 120 Hz; a device whose canvas copies are slow records fewer frames instead of
      // taking the session's main thread: at most BUDGET ms of drawing per second (>= 8 fps)
      if (C.lastRaf) { const d = tn - C.lastRaf; if (d > 2 && d < 100) C.rafDt = C.rafDt ? 0.9 * C.rafDt + 0.1 * d : d; }
      C.lastRaf = tn;
      const gap = Math.min(125, Math.max(1000 / FPS, C.ewma * 1000 / BUDGET));
      const slack = Math.min(gap / 2, (C.rafDt || 16.7) / 2);
      if (tn < C.due - slack) return;
      C.due = (C.due && tn - C.due < gap) ? C.due + gap : tn + gap;    // keep the cadence; resync after a stall
      C.lastDraw = tn;
      const t0 = now();
      try {
        if (tn - C.lastLayout > 500) { C.lastLayout = tn; layout(C); wireAudio(C); }
        draw(C); C.frames++;
        if (C.pushFrame) C.pushFrame();
      } catch (e) { C.drawErr = String(e && e.message || e).slice(0, 80); }
      const ms = now() - t0; C.draws.push(ms); if (C.draws.length > 300) C.draws.shift();
      C.ewma = C.ewma ? 0.9 * C.ewma + 0.1 * ms : ms;
    }

    function stopAll(C) {
      C.ended = true; caf(C.raf);
      try { C.vstream.getTracks().forEach(t => t.stop()); } catch (e) {}
      unwireAudio(C);
      if (R.cur === C) { R.cur = null; R.active = false; setDot(false); }
    }
    /* end the recording: reason 'stop' (the user pressed Stop), 'reconnect' (the page restarts the server session — the next
       begin() continues the same group as part n+1), 'pagehide', 'error', 'restart' */
    function end(reason, silent) {
      const C = R.cur;
      if (!C) return null;
      R.lastEnd = {reason, t: now()};
      const up = C.up, mr = C.mr, dur = Math.round(now() - C.t0);
      const info = {reason, duration_ms: dur, stats: statsOf(C)};
      C.ended = true; caf(C.raf); setDot(false);
      R.cur = null; R.active = false;
      if (reason === 'pagehide') {
        try { if (mr && mr.state !== 'inactive') { C.gen++; mr.stop(); } } catch (e) {}
        if (up) up.beaconEnd(info);
        stopAll(C);
        return info;
      }
      const done = () => {
        stopAll(C);
        if (!up) return;
        Promise.resolve(C.ridP).then(() => up.finish(info)).then(r => { if (!silent) log('rec: ended ' + (up.rid || '') + ' ' + JSON.stringify(r)); });
      };
      if (mr && mr.state !== 'inactive') {
        let fired = false;
        const fin = () => { if (!fired) { fired = true; done(); } };
        mr.addEventListener ? mr.addEventListener('stop', fin) : (mr.onstop = fin);
        setTimeout(fin, 3000);                                 // a recorder that never fires 'stop'
        try { mr.stop(); } catch (e) { fin(); }
      } else done();
      return info;
    }
    /* a new microphone mid-session (the page's picker): the mix takes its audio from the new stream */
    function setMic(stream) {
      const C = R.cur;
      if (!C || !stream) return false;
      C.o.mic = stream;
      if (C.A.mic) { try { C.A.mic.disconnect(); } catch (e) {} C.A.mic = null; }
      wireAudio(C);
      return true;
    }
    function note(n) {
      const C = R.cur;
      if (!C || !C.up || !C.up.rid || !n) return;
      const F = X.fetch || C.o.fetch || G.fetch.bind(G);
      const body = Object.assign({t_s: Math.round(now() - C.t0) / 1000}, n);
      F(C.o.url('api/rec/note', {rid: C.up.rid}), {method: 'POST', headers: Object.assign({}, C.o.headers || {}, {'content-type': 'application/json'}),
        body: JSON.stringify(body), cache: 'no-store', priority: 'low'}).catch(() => {});
    }
    function statsOf(C) {
      return {w: C.W, h: C.H, frames: C.frames, fps: r2(C.frames / Math.max(0.001, (now() - C.t0) / 1000)), part: C.part,
        draw_ms_p50: C.draws.length ? r2(pct(C.draws, 0.5)) : null, draw_ms_p95: C.draws.length ? r2(pct(C.draws, 0.95)) : null,
        draw_ms_max: C.draws.length ? r2(Math.max(...C.draws)) : null, kbps: Math.round(C.bitrate / 1000),
        ...(C.up ? (({sent, retries, dropped, max_queue_kb}) => ({sent, retries, dropped, max_queue_kb}))(C.up.stats()) : {})};
    }
    function stats() {
      const C = R.cur;
      return {active: R.active, off: R.off, group: R.group, totals: {...R.totals, kb: Math.round(R.totals.kb)},
        cur: C ? {mime: C.mime, state: C.mr && C.mr.state, rid: C.up && C.up.rid, ...statsOf(C), up: C.up && C.up.stats(), drawErr: C.drawErr || null} : null};
    }
    // leaving the page: the last chunk cannot be uploaded any more — tell the server where the recording ends
    if (!X.noUnload && G.addEventListener) G.addEventListener('pagehide', () => { if (R.cur) end('pagehide'); });
    return {begin, end, note, setMic, stats, get active() { return R.active; }, _R: R};
  }

  const lib = {createUploader, createRecorder, pickMime, fitRect, coverRect, MIMES};
  if (typeof module === 'object' && module.exports) module.exports = lib;
  if (typeof window !== 'undefined') { window.AvatarRecLib = lib; window.AvatarRec = createRecorder(); }
})();
