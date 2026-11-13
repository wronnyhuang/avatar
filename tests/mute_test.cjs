// mute: the real Gate worklet offline (a muted mic is silent and teaches the echo gate nothing; unmuting restores it; a muted
// speaker removes the reference) + voice.js on the wire (gate messages, the vendor's input buffer is cleared, the shield stands down).
//   node tests/mute_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Q = load('app/quiet.js');
const src = fs.readFileSync(path.join(__dirname, '..', 'app/voice.js'), 'utf8');
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + JSON.stringify(detail) : '')); }
// ------------------------------------------------------------------------------------------------ the gate worklet
const raw = src.slice(src.indexOf('const WORKLET = `') + 'const WORKLET = `'.length); const code = eval('`' + raw.slice(0, raw.indexOf('\n`;')) + '`');
const SR = 48000, QN = 128;
function makeGate() {
  const G = {t: 0, posts: []}; let Gate = null;
  Object.defineProperty(globalThis, 'currentTime', {get: () => G.t, configurable: true});
  class AWP { constructor() { this.port = {postMessage: m => G.posts.push(m), onmessage: null}; } }
  new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', 'currentFrame', code)(AWP, (n, c) => { if (n === 'avatar-mic-gate') Gate = c; }, SR, 0);
  G.node = new Gate(); G.msg = m => G.node.port.onmessage({data: m}); return G;
}
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const tone = (n, a, f) => { const bl = []; let ph = 0; for (let i = 0; i < n; i++) { const b = new Float32Array(QN); for (let j = 0; j < QN; j++) { b[j] = a * Math.sin(ph); ph += 2 * Math.PI * f / SR; } bl.push(b); } return bl; };
const noise = (n, a) => Array.from({length: n}, () => Float32Array.from({length: QN}, () => rnd() * a));
const ms = x => Math.round(x * SR / 1000 / QN);
const rms = b => Math.sqrt(b.reduce((a, v) => a + v * v, 0) / b.length);
function run(G, mic, ref) { const out = []; mic.forEach((b, i) => { const o = new Float32Array(QN); G.node.process([[b], [ref ? ref[i] : new Float32Array(QN)]], [[o]]); out.push(o); G.t += QN / SR; }); return out; }
{ const G = makeGate();
  // the avatar talks (reference) and the speaker echo reaches the mic at a steady ratio; train the gate on 3 s of that
  const N = ms(3000), ref = tone(N, 0.2, 180), echo = tone(N, 0.04, 180);
  run(G, echo, ref); const learned = {c: G.node.c, em: G.node.echoMax, bg: G.node.bg};
  G.msg({mute: true});
  const speech = tone(ms(2000), 0.15, 230);                                    // he talks (the page has silenced the track, but even if it did not)
  const outM = run(G, speech.map((b, i) => Float32Array.from(b)), tone(ms(2000), 0.2, 180));
  check('muted: nothing of his mic leaves the gate (silence out)', outM.every(b => rms(b) === 0));
  // the page's track is silent while muted: the avatar keeps talking for 10 s over a silent mic - the echo estimates must not drift
  run(G, noise(ms(10000), 0), tone(ms(10000), 0.2, 180));
  check('muted for 10 s while the avatar talks: the gate learns nothing (echo coupling / ceiling / background unchanged)', G.node.c === learned.c && G.node.echoMax === learned.em && G.node.bg === learned.bg, {learned, now: {c: G.node.c, em: G.node.echoMax, bg: G.node.bg}});
  check('muted frames are counted (stats)', G.node.st.mutedFrames > 500, G.node.st.mutedFrames);
  const barge = G.node.st.bargeIns; run(G, tone(ms(1500), 0.3, 230), tone(ms(1500), 0.2, 180));
  check('muted: no barge-in is ever counted', G.node.st.bargeIns === barge);
  // unmute with the avatar silent: his speech passes untouched
  G.msg({mute: false}); const sp = tone(ms(1500), 0.12, 230);
  const outU = run(G, sp, null); const loud = outU.filter(b => rms(b) > 0.05).length;
  check('unmuted: his speech passes again (at the same level once the gate settles)', loud >= sp.length - 8, loud + '/' + sp.length);
}
{ const G = makeGate(); G.msg({shield: true}); run(G, tone(ms(800), 0.12, 230), tone(ms(800), 0.2, 180));
  check('speech was being held for the shield...', G.node.q.length > 0 && G.node.sh === true);
  G.msg({mute: true}); check('...mute drops it and ends the hold (it must not replay after unmuting)', G.node.q.length === 0 && G.node.sh === false);
  G.msg({mute: false}); const out = run(G, noise(ms(500), 0.001), null); check('unmute afterwards: nothing is replayed', out.every(b => rms(b) < 0.01));
}
{ const G = makeGate(); const N = ms(1500); G.msg({refOff: true});
  run(G, tone(N, 0.12, 230), tone(N, 0.3, 180));
  check('speaker muted (refOff): the avatar\'s reference is ignored (the gate is never "in a reply")', G.node.inReply === false && G.node.st.refFrames === 0, {inReply: G.node.inReply, refFrames: G.node.st.refFrames});
  G.msg({refOff: false}); run(G, tone(N, 0.04, 230), tone(N, 0.3, 180)); check('speaker unmuted: the reference counts again', G.node.st.refFrames > 0);
}
// ------------------------------------------------------------------------------------------------ voice.js on the wire
function makeClock() { const C = {t: 1000, q: [], n: 0}; C.now = () => C.t; C.set = (f, m) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +m || 0), f}); return id; }; C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = m => { const end = C.t + m; for (;;) { C.q.sort((a, b) => a.at - b.at || a.id - b.id); const x = C.q[0]; if (!x || x.at > end) break; C.q.shift(); C.t = Math.max(C.t, x.at); x.f(); } C.t = end; };
  C.api = {now: C.now, setTimeout: C.set, clearTimeout: C.clear}; return C; }
function fakeCtx() { const X = {gains: [], sources: [], currentTime: 0}; X.createGain = () => { const g = {connect() { return g; }, disconnect() {}, gain: {value: 1}}; X.gains.push(g); return g; }; X.createBufferSource = () => { const s = {connect() {}, start() { X.sources.push(s); }, stop() {}}; return s; }; return X; }
const TD = {type: 'server_vad', threshold: 0.6, prefix_padding_ms: 300, silence_duration_ms: 300, create_response: true, interrupt_response: true};
async function makeV() {
  const clock = makeClock(), sent = [], gate = []; const w = {};
  new Function('window', 'performance', 'setTimeout', 'clearTimeout', src)(w, {now: clock.now}, clock.set, clock.clear);
  const ctl = Q.create({clock: clock.api, voice: () => X.v, name: () => 'Muse'}); const X = {clock, sent, gate};
  X.v = w.AvatarVoice.createVoice({mode: 'realtime', ctx: fakeCtx(), quiet: ctl, shieldPort: {postMessage: m => gate.push(m)}, transport: {send: ev => sent.push(JSON.parse(JSON.stringify(ev)))}, onCall: () => ({ok: true})});
  await X.v.start(); X.v.feed({type: 'session.created', session: {instructions: 'You are Muse.', audio: {input: {turn_detection: TD}}}}); return X;
}
(async () => {
  { const X = await makeV(); const {v, sent, gate} = X; const n0 = sent.length;
    v.setMicMuted(true);
    check('setMicMuted(true): the gate is told to mute', gate.some(m => m.mute === true), gate);
    check('...and the vendor\'s uncommitted input buffer is cleared (half a sentence cannot become a reply)', sent.slice(n0).some(e => e.type === 'input_audio_buffer.clear'), sent.slice(n0));
    check('...stats say muted', v.stats().mic_muted === true && v.micMuted() === true);
    v.feed({type: 'response.created', response: {id: 'r1'}}); v.feed({type: 'output_audio_buffer.started', response_id: 'r1'});
    check('a reply plays while muted: no shield (nothing of his to hold)', !gate.some(m => m.shield === true), gate);
    const n1 = sent.length; v.setMicMuted(true); check('setMicMuted(true) again is a no-op (no duplicate clear)', sent.length === n1);
    v.setMicMuted(false); check('unmute: the gate is told, the session was never touched (no cancel, no hang-up)', gate.some(m => m.mute === false) && !sent.some(e => e.type === 'response.cancel') && v.state() === 'live', {state: v.state()});
    v.feed({type: 'output_audio_buffer.stopped', response_id: 'r1'});
    v.feed({type: 'response.created', response: {id: 'r2'}}); v.feed({type: 'output_audio_buffer.started', response_id: 'r2'});
    check('unmuted: the shield works as before', gate.some(m => m.shield === true));
  }
  { const X = await makeV(); const {v, gate} = X; v.setSpeakerMuted(true);
    check('setSpeakerMuted(true): the gate drops the avatar reference', gate.some(m => m.refOff === true));
    v.feed({type: 'response.created', response: {id: 'r1'}}); v.feed({type: 'output_audio_buffer.started', response_id: 'r1'});
    check('speaker muted: no shield (there is nothing audible to protect, his words must not wait)', !gate.some(m => m.shield === true), gate);
    v.setSpeakerMuted(false); check('speaker unmuted: reference back', gate.some(m => m.refOff === false) && v.stats().spk_muted === false);
  }
  { const X = await makeV(); const {v, gate} = X; v.feed({type: 'response.created', response: {id: 'r1'}}); v.feed({type: 'output_audio_buffer.started', response_id: 'r1'});
    v.setMicMuted(true); check('muting during a shielded reply releases the shield', gate.map(m => m.shield).filter(x => x !== undefined).join() === 'true,false', gate);
  }
  console.log(`mute_test: ${passes} passed, ${fails} failed`); process.exit(fails ? 1 : 0);
})();
