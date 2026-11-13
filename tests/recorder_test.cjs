// recorder.js unit tests (rec): the chunk uploader against an in-memory hub, and the recorder's lifecycle with
// fake browser objects (MediaRecorder, canvas, AudioContext).   node tests/recorder_test.cjs
'use strict';
const assert = require('assert');
const L = require('../app/recorder.js');

let pass = 0, fail = 0;
async function t(name, fn) {
  let timer;
  try { await Promise.race([fn(), new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('timed out after 10 s')), 10000); })]); pass++; console.log('PASS ' + name); }
  catch (e) { fail++; console.log('FAIL ' + name + '\n   ' + String(e && e.stack || e).split('\n').slice(0, 4).join('\n   ')); }
  finally { clearTimeout(timer); }
}
const tick = (n = 5) => new Promise(r => { let i = 0; const f = () => (++i >= n ? r() : setImmediate(f)); f(); });

// a fake hub speaking recorder_hub.py's protocol; knobs make it misbehave
function fakeHub() {
  const H = {rids: {}, closed: new Set(), calls: [], failNext: 0, loseResponseNext: 0, status503: 0, n: 0, maxChunk: 1e9};
  H.url = (path, params) => path + '?' + new URLSearchParams(params || {}).toString();
  H.fetch = async (u, init) => {
    const [path, qs] = u.split('?'), q = new URLSearchParams(qs), m = (init && init.method) || 'GET';
    H.calls.push({path, q: Object.fromEntries(q), m});
    const reply = (status, j) => ({status, ok: status >= 200 && status < 300, json: async () => j});
    if (path === 'api/rec/start') { const rid = 'rid' + (++H.n); H.rids[rid] = {chunks: new Map(), meta: JSON.parse(init.body), end: null}; return reply(200, {ok: true, rid}); }
    const rid = q.get('rid'), R = H.rids[rid];
    if (path === 'api/rec/chunk') {
      if (H.failNext > 0) { H.failNext--; throw new TypeError('Failed to fetch'); }
      if (H.status503 > 0) { H.status503--; return reply(503, {ok: false}); }
      if (!R || H.closed.has(rid)) return reply(410, {ok: false, closed: true});
      const body = init.body, size = body.size != null ? body.size : body.length;
      if (size > H.maxChunk) return reply(413, {ok: false});
      const seq = +q.get('seq'), dup = R.chunks.has(seq);
      R.chunks.set(seq, body);
      if (H.loseResponseNext > 0) { H.loseResponseNext--; throw new TypeError('network changed'); }
      return reply(200, {ok: true, seq, dup});
    }
    if (path === 'api/rec/status') {
      if (!R || H.closed.has(rid)) return reply(410, {});
      return reply(200, {ok: true, have: [...R.chunks.keys()].sort((a, b) => a - b)});
    }
    if (path === 'api/rec/end') { if (!R) return reply(410, {}); R.end = JSON.parse(init.body); return reply(200, {ok: true}); }
    if (path === 'api/rec/note') { if (R) (R.notes = R.notes || []).push(JSON.parse(init.body)); return reply(200, {ok: true}); }
    return reply(404, {});
  };
  return H;
}
const blob = (n, fill = 1) => { const b = Buffer.alloc(n, fill); return b; };
const rblob = n => new Blob([Buffer.alloc(n, 2)]);        // what MediaRecorder hands out (has .size)

(async () => {
  // ------------------------------------------------------------------------------------------- uploader
  await t('chunks pushed before the rid is known go out in sequence once it arrives; end carries n', async () => {
    const H = fakeHub(), sleeps = [];
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: ms => { sleeps.push(ms); return new Promise(r => setImmediate(r)); }});
    for (let i = 0; i < 5; i++) up.push(i, blob(100 + i));
    await tick(); assert.strictEqual(H.calls.length, 0);
    up.setRid('rid9'); H.rids.rid9 = {chunks: new Map()};
    const r = await up.finish({reason: 'stop', duration_ms: 1500});
    assert.ok(r.ok);
    const seqs = H.calls.filter(c => c.path === 'api/rec/chunk').map(c => +c.q.seq);
    assert.deepStrictEqual(seqs, [0, 1, 2, 3, 4]);
    assert.deepStrictEqual(H.rids.rid9.end.n, 5); assert.deepStrictEqual(H.rids.rid9.end.dropped, []);
    assert.strictEqual(H.rids.rid9.end.reason, 'stop'); assert.strictEqual(sleeps.length, 0);
  });

  await t('network errors + 503 retry with doubling back-off (1 s .. 30 s cap, jitter), nothing lost or reordered', async () => {
    const H = fakeHub(), sleeps = [];
    H.rids.r = {chunks: new Map()};
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: ms => { sleeps.push(ms); return new Promise(r => setImmediate(r)); }, random: () => 0.5});
    up.setRid('r'); H.failNext = 7; H.status503 = 2;
    for (let i = 0; i < 3; i++) up.push(i, blob(50));
    await up.finish({reason: 'stop'});
    assert.deepStrictEqual([...H.rids.r.chunks.keys()], [0, 1, 2]);
    assert.deepStrictEqual(sleeps, [1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000, 30000]);
    assert.strictEqual(up.stats().retries, 9);
  });

  await t('a lost response: the status call skips what the hub already has (resumable by sequence number)', async () => {
    const H = fakeHub();
    H.rids.r = {chunks: new Map()};
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r))});
    up.setRid('r');
    H.loseResponseNext = 2;          // chunk 0 stored but the reply never came, twice -> status -> skip
    up.push(0, blob(10)); up.push(1, blob(10));
    await up.finish({});
    const sent = H.calls.filter(c => c.path === 'api/rec/chunk').map(c => +c.q.seq);
    assert.ok(H.calls.some(c => c.path === 'api/rec/status'), 'status asked');
    assert.deepStrictEqual(sent, [0, 0, 1], 'chunk 0 not re-sent after status: ' + sent);
    assert.strictEqual(H.rids.r.end.n, 2);
  });

  await t('410 closed -> onClosed (new part), queue cleared, no end', async () => {
    const H = fakeHub();
    let closed = 0;
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r)), onClosed: () => closed++});
    H.rids.r = {chunks: new Map()}; H.closed.add('r');
    up.setRid('r'); up.push(0, blob(10)); up.push(1, blob(10));
    const r = await up.finish({});
    assert.strictEqual(closed, 1); assert.ok(up.closed); assert.strictEqual(r.ok, false);
    assert.ok(!H.calls.some(c => c.path === 'api/rec/end'));
    assert.strictEqual(up.push(2, blob(10)), false);
  });

  await t('bounded memory: over the cap the oldest unsent chunks are dropped (never chunk 0) and named in end', async () => {
    const H = fakeHub();
    H.rids.r = {chunks: new Map()};
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r)), maxBytes: 1000});
    for (let i = 0; i < 10; i++) up.push(i, blob(300));      // rid unknown: everything waits
    assert.ok(up.stats().queued_kb <= 1, 'bounded: ' + JSON.stringify(up.stats()));
    up.setRid('r');
    await up.finish({});
    const got = [...H.rids.r.chunks.keys()].sort((a, b) => a - b);
    assert.ok(got.includes(0) && got.includes(9), 'kept header + newest: ' + got);
    assert.deepStrictEqual(H.rids.r.end.dropped, [1, 2, 3, 4, 5, 6, 7], 'dropped: ' + H.rids.r.end.dropped);
    assert.strictEqual(H.rids.r.end.n, 10);
  });

  await t('413 (hub will never take it) -> skipped + reported dropped; the rest continues', async () => {
    const H = fakeHub(); H.maxChunk = 100;
    H.rids.r = {chunks: new Map()};
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r))});
    up.setRid('r'); up.push(0, blob(50)); up.push(1, blob(500)); up.push(2, blob(50));
    await up.finish({});
    assert.deepStrictEqual([...H.rids.r.chunks.keys()], [0, 2]); assert.deepStrictEqual(H.rids.r.end.dropped, [1]);
  });

  await t('pagehide beacon: end with rid, n, reason in the query (no headers possible)', async () => {
    const H = fakeHub(); let got = null;
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r)), beacon: (u, b) => { got = {u, b: JSON.parse(b)}; return true; }});
    H.rids.r = {chunks: new Map()};
    up.setRid('r'); up.push(0, blob(10)); up.push(1, blob(10));
    assert.ok(up.beaconEnd({reason: 'pagehide'}));
    assert.ok(/rid=r/.test(got.u) && /n=2/.test(got.u) && /reason=pagehide/.test(got.u), got.u);
    assert.strictEqual(up.beaconEnd({}), false, 'once');
  });

  await t('finish with no rid (hub refused / unreachable) returns at once and frees the queue', async () => {
    const H = fakeHub();
    const up = L.createUploader({fetch: H.fetch, url: H.url, sleep: () => new Promise(r => setImmediate(r))});
    up.push(0, blob(10));
    const r = await up.finish({});
    assert.ok(r.norid && up.stats().queued === 0);
  });

  // ------------------------------------------------------------------------------------------- geometry
  await t('fitRect / coverRect (object-fit contain / cover)', async () => {
    assert.deepStrictEqual(L.fitRect(1280, 720, 0, 0, 640, 480), [0, 0, 1280, 720, 0, 60, 640, 360]);
    assert.deepStrictEqual(L.coverRect(640, 480, 0, 0, 160, 90), [0, 60, 640, 360, 0, 0, 160, 90]);
    assert.strictEqual(L.pickMime({isTypeSupported: m => m === 'video/mp4'}), 'video/mp4');
    assert.strictEqual(L.pickMime({isTypeSupported: m => /vp8|vp9/.test(m)}), 'video/webm;codecs=vp8,opus');
  });

  // ------------------------------------------------------------------------------------------- recorder lifecycle (fakes)
  function fakes(H, opt = {}) {
    const ops = [];
    const node = name => ({name, conns: [], connect(n) { this.conns.push(n); ops.push(name + '->' + (n && n.name)); return n; },
      disconnect(n) { ops.push(name + '-x' + (n ? '>' + n.name : '')); this.conns = n ? this.conns.filter(x => x !== n) : []; }});
    const track = kind => ({kind, stopped: false, stop() { this.stopped = true; }});
    const ctx2d = {calls: [], fillRect() {}, drawImage(...a) { this.calls.push(['drawImage', a[0] && a[0].tag]); }, save() {}, restore() {}, beginPath() {}, roundRect() {}, clip() {},
      translate(...a) { this.calls.push(['translate', ...a]); }, scale(...a) { this.calls.push(['scale', ...a]); }, stroke() {}, fillText() {}};
    const vtrack = track('video'); vtrack.pushed = 0; if (!opt.noRequestFrame) vtrack.requestFrame = () => { vtrack.pushed++; };
    const canvasOut = {tag: 'rec', width: 0, height: 0, getContext: () => ctx2d, captureStream: fps => { canvasOut.fps = fps; return {fps, getVideoTracks: () => [vtrack], getTracks: () => [vtrack]}; }};
    const rect = (l, t, w, h) => ({left: l, top: t, width: w, height: h});
    const avatarCanvas = {tag: 'avatar', width: 480, height: 270, isConnected: true, getBoundingClientRect: () => rect(0, 0, 400, 225)};
    const video = {tag: 'self', readyState: 4, videoWidth: 640, videoHeight: 480, isConnected: true, getBoundingClientRect: () => rect(298, 147, 96, 72)};
    const box = {isConnected: true, getBoundingClientRect: () => rect(298, 147, 96, 72)};
    video.parentElement = box;
    const stage = {isConnected: true, getBoundingClientRect: () => rect(0, 0, 400, 225), querySelectorAll: () => [video], querySelector: () => null};
    const styles = new Map([[box, {display: 'block', opacity: '1', borderTopLeftRadius: '10px', borderTopWidth: '1.5px', borderTopColor: 'rgba(255,255,255,.55)'}],
      [video, {display: 'block', opacity: '1', transform: 'matrix(-1, 0, 0, 1, 0, 0)'}], [avatarCanvas, {display: 'block', opacity: '1'}]]);
    global.window = undefined;
    const G = {getComputedStyle: el => styles.get(el) || {display: 'block', opacity: '1'}, location: {search: opt.search == null ? '?rec=1' : opt.search}};
    Object.assign(global, G);
    const recs = [];
    class MR {
      constructor(stream, o) { this.stream = stream; this.o = o; this.state = 'inactive'; this.mimeType = o.mimeType; this.l = {}; recs.push(this); }
      static isTypeSupported(m) { return m === 'video/webm;codecs=vp8,opus'; }
      start(ts) { this.ts = ts; this.state = 'recording'; }
      stop() { this.state = 'inactive'; setImmediate(() => { this.ondataavailable({data: rblob(77)}); (this.l.stop || []).forEach(f => f()); }); }
      addEventListener(n, f) { (this.l[n] = this.l[n] || []).push(f); }
      emit(n) { this.ondataavailable({data: rblob(n)}); }
    }
    class MS { constructor(tracks) { this.tracks = tracks; } getTracks() { return this.tracks; } }
    const dest = Object.assign(node('dest'), {stream: {getAudioTracks: () => [track('audio')], getTracks: () => []}});
    const actx = {createMediaStreamDestination: () => dest, createGain: () => Object.assign(node('gain' + ops.length), {gain: {value: 0}}),
      createMediaStreamSource: s => node('src:' + s.id)};
    const doc = {createElement: tag => tag === 'canvas' ? canvasOut : {}, head: {appendChild() {}}, body: {classList: {toggle() {}}}};
    const frames = [];
    let tnow = 0;
    const deps = {document: doc, MediaRecorder: MR, MediaStream: MS, fetch: H.fetch, navigator: {webdriver: !!opt.webdriver, userAgent: 'test'},
      raf: f => { frames.push(f); return frames.length; }, caf: () => {}, now: () => tnow, devicePixelRatio: 2, noUnload: true, log: () => {}};
    return {deps, recs, ops, ctx2d, actx, canvasOut, stage, avatarCanvas, video, vtrack, frames, step: ms => { tnow += ms; const f = frames.splice(0); f.forEach(fn => fn(tnow)); }};
  }

  await t('recorder: begin -> start (meta) -> 3 s timeslice chunks -> end(stop) -> final chunk + end; mirrored self-view drawn', async () => {
    const H = fakeHub(), F = fakes(H);
    const rec = L.createRecorder(F.deps);
    const voiceOut = {name: 'voiceOut', connect(n) { F.ops.push('voiceOut->' + n.name); }, disconnect() { F.ops.push('voiceOut-x'); }};
    const r = rec.begin({url: H.url, headers: {}, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx, mic: {id: 'mic1'}, voiceOut: () => voiceOut,
      character: 'muse', characterName: 'Muse', voice: 'realtime', voiceLabel: 'Realtime 2.1', sid: 'abcdefgh'});
    assert.deepStrictEqual([r.w, r.h, r.mime], [600, 338, 'video/webm;codecs=vp8,opus']);   // 400x225 css x dpr 1.5 (capped)
    await r.ready;
    assert.strictEqual(F.recs[0].ts, 3000);
    await tick(10);
    const st = H.calls.find(c => c.path === 'api/rec/start');
    assert.ok(st); const meta = H.rids.rid1.meta;
    assert.strictEqual(meta.character, 'muse'); assert.strictEqual(meta.sid, 'abcdef'); assert.strictEqual(meta.part, 1); assert.strictEqual(meta.width, 600);
    assert.ok(F.ops.includes('voiceOut->' + F.ops.find(o => o.startsWith('voiceOut->')).split('->')[1]), 'avatar voice into the mix');
    assert.ok(F.ops.some(o => o.startsWith('src:mic1->')), 'mic into the mix');
    F.step(40); F.step(40);
    assert.ok(F.ctx2d.calls.some(c => c[0] === 'drawImage' && c[1] === 'avatar'), 'avatar canvas drawn');
    assert.ok(F.ctx2d.calls.some(c => c[0] === 'drawImage' && c[1] === 'self'), 'self-view drawn');
    assert.ok(F.ctx2d.calls.some(c => c[0] === 'scale' && c[1] === -1), 'self-view mirrored like the page');
    F.recs[0].emit(1000); F.recs[0].emit(1200);
    rec.end('stop');
    for (let i = 0; i < 40 && !H.rids.rid1.end; i++) await tick(5);
    assert.deepStrictEqual([...H.rids.rid1.chunks.keys()], [0, 1, 2], 'two timeslices + the final one');
    assert.strictEqual(H.rids.rid1.end.n, 3); assert.strictEqual(H.rids.rid1.end.reason, 'stop');
    assert.ok(F.vtrack.stopped, 'canvas track stopped'); assert.ok(!rec.active);
    assert.ok(H.rids.rid1.end.stats && H.rids.rid1.end.stats.w === 600, 'client stats ride the end');
  });

  await t('recorder: 410 mid-recording -> new part (part 2, same group, fresh MediaRecorder)', async () => {
    const H = fakeHub(), F = fakes(H);
    const rec = L.createRecorder(F.deps);
    rec.begin({url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx, mic: {id: 'm'}, voiceOut: null, character: 'fox'});
    await tick(10);
    F.recs[0].emit(500); await tick(10);
    H.closed.add('rid1');
    F.recs[0].emit(500); await tick(20);
    assert.strictEqual(F.recs.length, 2, 'a second MediaRecorder');
    await tick(10);
    const m2 = H.rids.rid2 && H.rids.rid2.meta;
    assert.ok(m2 && m2.part === 2 && m2.group === H.rids.rid1.meta.group, JSON.stringify(m2));
    F.recs[1].emit(300); await tick(10);
    assert.deepStrictEqual([...H.rids.rid2.chunks.keys()], [0]);
    rec.end('stop'); for (let i = 0; i < 40 && !H.rids.rid2.end; i++) await tick(5);
    assert.ok(H.rids.rid2.end && !H.rids.rid1.end);
  });

  await t('recorder: a reconnect (quiet restart) continues the group as the next part', async () => {
    const H = fakeHub(), F = fakes(H);
    const rec = L.createRecorder(F.deps);
    const o = {url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx, mic: {id: 'm'}, character: 'fox'};
    rec.begin(o); await tick(10); rec.end('reconnect'); await tick(20);
    rec.begin(o); await tick(10);
    assert.strictEqual(H.rids.rid2.meta.part, 2); assert.strictEqual(H.rids.rid2.meta.group, H.rids.rid1.meta.group);
    rec.end('stop'); await tick(20);
    rec.begin(o); await tick(10);
    assert.strictEqual(H.rids.rid3.meta.part, 1); assert.notStrictEqual(H.rids.rid3.meta.group, H.rids.rid1.meta.group);
    rec.end('stop');
  });

  await t('recorder: off by default and with ?rec=0; on only with ?rec=1; server 503 -> stops quietly', async () => {
    let H = fakeHub(), F = fakes(H, {search: ''});
    let rec = L.createRecorder(F.deps);
    assert.strictEqual(rec.begin({url: H.url, stage: F.stage, ctx: F.actx}), null); assert.ok(/rec=1/.test(rec.stats().off));
    H = fakeHub(); F = fakes(H, {search: '?rec=0'}); rec = L.createRecorder(F.deps);
    assert.strictEqual(rec.begin({url: H.url, stage: F.stage, ctx: F.actx}), null); assert.ok(/rec=1/.test(rec.stats().off));
    H = fakeHub(); F = fakes(H, {webdriver: true, search: '?rec=1'}); rec = L.createRecorder(F.deps);
    assert.ok(rec.begin({url: H.url, stage: F.stage, ctx: F.actx, canvas: F.avatarCanvas})); rec.end('stop');
    H = fakeHub(); const f0 = H.fetch;
    H.fetch = async (u, i) => (u.startsWith('api/rec/start') ? {status: 503, ok: false, json: async () => ({ok: false, error: 'recording is off here', off: true})} : f0(u, i));
    F = fakes(H); rec = L.createRecorder(F.deps);
    rec.begin({url: H.url, stage: F.stage, ctx: F.actx, canvas: F.avatarCanvas}); await tick(30);
    assert.ok(!rec.active && /off here/.test(rec.stats().off), JSON.stringify(rec.stats()));
  });

  await t('recorder: setMic mid-session re-feeds the mix from the new stream; note() posts the character switch', async () => {
    const H = fakeHub(), F = fakes(H);
    const rec = L.createRecorder(F.deps);
    rec.begin({url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx, mic: {id: 'micA'}, character: 'fox'});
    await tick(10);
    assert.ok(rec.setMic({id: 'micB'}));
    assert.ok(F.ops.includes('src:micA-x') && F.ops.some(o => o.startsWith('src:micB->')), F.ops.join(' '));
    rec.note({character: 'muse'}); await tick(10);
    assert.strictEqual(H.rids.rid1.notes[0].character, 'muse');
    rec.end('stop');
  });

  await t('recorder: 30 fps on 60 / 90 / 120 Hz displays, one requestFrame per draw; slow canvas -> budget', async () => {
    for (const [hz, jit] of [[60, 0], [90, 0], [120, 0], [60, 1.5], [144, 0.5]]) {
      const H = fakeHub(), F = fakes(H);
      const rec = L.createRecorder(F.deps);
      rec.begin({url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx, character: 'muse'});
      assert.strictEqual(F.canvasOut.fps, 0, 'captureStream(0): frames pushed by requestFrame');
      let k = 0;
      for (let i = 0; i < hz * 4; i++) F.step(1000 / hz + (jit ? ((k++ % 2) ? jit : -jit) : 0));
      const n = F.vtrack.pushed, fps = n / 4;
      assert.ok(fps >= 29 && fps <= 31, hz + ' Hz jitter ' + jit + ': ' + fps + ' fps');
      assert.strictEqual(n, rec.stats().cur.frames, 'one pushed frame per draw');
      rec.end('stop');
    }
    { // no requestFrame (older engine): rate-capped captureStream(30)
      const H = fakeHub(), F = fakes(H, {noRequestFrame: true});
      const rec = L.createRecorder(F.deps);
      rec.begin({url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx});
      assert.strictEqual(F.canvasOut.fps, 30); rec.end('stop');
    }
    { // a slow canvas (each draw costs 12 ms): <= 250 ms/s of drawing -> ~20 fps, never above budget
      const H = fakeHub(), F = fakes(H);
      let tn = 0; F.deps.now = () => tn;                                                // own clock for this case
      const rec = L.createRecorder(F.deps);
      rec.begin({url: H.url, stage: F.stage, canvas: F.avatarCanvas, ctx: F.actx});
      const d = F.ctx2d.drawImage;
      F.ctx2d.drawImage = function (...a) { tn += 6; return d.apply(this, a); };     // 2 drawImage per composite = 12 ms
      const raf = F.frames;
      for (let i = 0; i < 240; i++) { tn += 1000 / 60; raf.splice(0).forEach(fn => fn(tn)); }
      const st = rec.stats().cur, fps = st.fps;   // frames per elapsed second (the clock includes the draw time)
      assert.ok(fps >= 15 && fps <= 23, 'slow canvas ' + fps + ' fps');
      rec.end('stop');
    }
  });

  console.log(`\n${pass}/${pass + fail} passed`);
  process.exit(fail ? 1 : 0);
})();
