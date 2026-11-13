// shield: his next words must not cut the reply that is audible. app/voice.js through its scripted transport (no browser,
// no vendor): what goes out on the wire around a reply's audio.   node tests/shield_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const Q = load('app/quiet.js');
const ROOT = path.join(__dirname, '..');
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + JSON.stringify(detail) : '')); }
function makeClock() {
  const C = {t: 1000, q: [], n: 0};
  C.now = () => C.t; C.set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = ms => { const end = C.t + ms; for (;;) { C.q.sort((a, b) => a.at - b.at || a.id - b.id); const x = C.q[0]; if (!x || x.at > end) break; C.q.shift(); C.t = Math.max(C.t, x.at); x.f(); } C.t = end; };
  C.api = {now: C.now, setTimeout: C.set, clearTimeout: C.clear}; return C;
}
function loadVoice(clock) {
  const src = fs.readFileSync(path.join(ROOT, 'app/voice.js'), 'utf8'); const w = {};
  new Function('window', 'performance', 'setTimeout', 'clearTimeout', src)(w, {now: clock.now}, clock.set, clock.clear); return w.AvatarVoice;
}
function fakeCtx() {
  const X = {gains: [], sources: [], currentTime: 0};
  X.createGain = () => { const g = {connect() { return g; }, disconnect() {}, gain: {value: 1}}; X.gains.push(g); return g; };
  X.createBufferSource = () => { const s = {connect() {}, start() { X.sources.push(s); }, stop() {}}; return s; };
  return X;
}
const TD = {type: 'server_vad', threshold: 0.6, prefix_padding_ms: 300, silence_duration_ms: 300, create_response: true, interrupt_response: true};
async function makeV() {
  const clock = makeClock(), ctx = fakeCtx(), sent = [];
  const AV = loadVoice(clock);
  const ctl = Q.create({clock: clock.api, voice: () => X.v, name: () => 'Muse'});
  const X = {clock, sent, ctl}; GATE = [];
  X.v = AV.createVoice({mode: 'realtime', ctx, quiet: ctl, shieldPort: {postMessage: m => GATE.push(m)}, transport: {send: ev => sent.push(JSON.parse(JSON.stringify(ev)))}, onCall: () => ({ok: true})});
  await X.v.start();
  X.v.feed({type: 'session.created', session: {instructions: 'You are Muse.', audio: {input: {turn_detection: TD}}}});
  return X;
}
const updates = s => s.filter(e => e.type === 'session.update' && e.session.audio && e.session.audio.input && e.session.audio.input.turn_detection).map(e => e.session.audio.input.turn_detection);
let GATE = [];                                   // what the page's gate worklet is told (the test seam o.shieldPort)
const flags = () => GATE.filter(m => 'shield' in m).map(m => m.shield);
let rn = 0;
const created = (X, id, md) => X.v.feed({type: 'response.created', response: {id, metadata: md}});
const started = (X, id) => X.v.feed({type: 'output_audio_buffer.started', response_id: id});
const stopped = (X, id) => X.v.feed({type: 'output_audio_buffer.stopped', response_id: id});
const speech = X => X.v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 0, item_id: 'u' + (++rn)});

(async () => {
  // ---- a reply to a command: shield up at its audio, his speech does not cut it, down when the audio ends
  { const X = await makeV(); const {v, sent, clock} = X;
    created(X, 'r1'); clock.advance(300); started(X, 'r1');
    check('audible reply: the mic is held (shield message to the gate), the VAD settings are not touched', flags().join() === 'true' && updates(sent).length === 0, {g: GATE, u: updates(sent)});
    const n0 = sent.length; speech(X); clock.advance(1200);
    check('his speech during it: nothing of ours cancels or clears the reply', !sent.slice(n0).some(e => e.type === 'response.cancel' || e.type === 'output_audio_buffer.clear'), sent.slice(n0));
    check('shield counters: one shield, one held', v.stats().shield.n === 1 && v.stats().shield.held === 1 && v.stats().shield.on === true, v.stats().shield);
    stopped(X, 'r1');
    check('audio ended: the held mic is released', flags().join() === 'true,false', flags());
    check('shield off after the audio', v.stats().shield.on === false);
  }
  // ---- the cap: a long reply still yields to him
  { const X = await makeV(); const {sent, clock} = X;
    created(X, 'r1'); started(X, 'r1'); clock.advance(11900);
    check('cap: still shielded at 11.9 s', flags().join() === 'true', flags());
    clock.advance(200);
    check('cap: released at 12 s (the valve for a lost audio-end event)', flags().join() === 'true,false', flags());
  }
  // ---- a chain (reply, cue, done): each further reply restarts the cap; no started event between them
  { const X = await makeV(); const {sent, clock} = X;
    created(X, 'r1'); started(X, 'r1'); clock.advance(8000);
    created(X, 'r2', {avatar_prompt: 'avp2'}); clock.advance(8000);
    check('chain: a reply created while one is audible keeps the shield (cap restarted)', flags().join() === 'true', flags());
    clock.advance(4100);
    check('chain: released 12 s after the last reply of the chain', flags().join() === 'true,false', flags());
  }
  // ---- an unprompted line is shielded too ("Nice to see you" was cut to "Nice to")
  { const X = await makeV(); const {v, sent, clock} = X;
    const e = v.sendEvent('He is nonverbally inviting you to talk. Take your turn.', {respond: true});
    created(X, 'r1'); started(X, 'r1');
    check('unprompted event reply (a turn bid\'s greeting) is shielded too', e.responded && flags().join() === 'true', {e, f: flags()});
  }
  // ---- a reply being made from half a sentence of his is still cancelled when he goes on
  { const X = await makeV(); const {sent, clock} = X;
    created(X, 'r1'); started(X, 'r1');
    created(X, 'r2');                             // the server's reply to his first half-sentence: no audio yet
    const n0 = sent.length; speech(X);
    const c = sent.slice(n0).filter(e => e.type === 'response.cancel');
    check('shield: the reply to a half sentence (no audio yet) is cancelled, the audible one is not', c.length === 1 && c[0].response_id === 'r2', c);
  }
  // ---- the cue our own prompt asked for is never cancelled by his speech
  { const X = await makeV(); const {v, sent, clock} = X;
    created(X, 'r1'); started(X, 'r1'); X.v.feed({type: 'response.done', response: {id: 'r1', status: 'completed', output: []}});
    v.prompt('The action he asked for is about to start. Give him one brief cue.', {respond: true}); clock.advance(50);
    const pc = sent.filter(e => e.type === 'response.create'); const md = pc.length && pc[pc.length - 1].response && pc[pc.length - 1].response.metadata;
    created(X, 'r2', md);                          // the cue (ours, by metadata): no audio yet
    const n0 = sent.length; speech(X);
    check('shield: the cue our prompt asked for (no audio yet) is not cancelled', pc.length === 1 && !sent.slice(n0).some(e => e.type === 'response.cancel'), {pc, c: sent.slice(n0)});
  }
  // ---- tap-to-interrupt releases it at once
  { const X = await makeV(); const {v} = X;
    created(X, 'r1'); started(X, 'r1'); v.openMic(2500);
    check('tap-to-interrupt: shield released', flags().join() === 'true,false', GATE);
  }
  // ---- a dead / stopped session never leaves his mic held
  { const X = await makeV(); const {v} = X;
    created(X, 'r1'); started(X, 'r1'); v.stop();
    check('stop: the hold is released', flags().join() === 'true,false', flags());
  }
  // ---- quiet mode: no shield (the gate owns the wire)
  { const X = await makeV(); const {v, sent} = X;
    v.setQuiet(true, {wake: false, graceMs: 0});
    const n0 = flags().length;
    created(X, 'r1'); started(X, 'r1');
    check('quiet: no shield', !flags().slice(n0).includes(true), flags());
  }
  // ---- no session.update before the server's turn_detection is known; mock/other modes never touch it
  { const clock = makeClock(), ctx = fakeCtx(), sent = []; const AV = loadVoice(clock);
    const v = AV.createVoice({mode: 'realtime', ctx, transport: {send: ev => sent.push(ev)}, onCall: () => ({ok: true})}); await v.start();
    v.feed({type: 'response.created', response: {id: 'x'}}); v.feed({type: 'output_audio_buffer.started', response_id: 'x'});
    check('no gate (raw mic / no worklet): no shield, nothing sent', updates(sent).length === 0 && sent.length === 0, sent);
  }
  console.log(`shield_test: ${passes} passed, ${fails} failed`);
  process.exit(fails ? 1 : 0);
})();
