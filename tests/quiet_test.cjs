// quiet: talkativeness + quiet mode. No browser, no vendor, no paid calls: app/quiet.js (the controller) on a fake
// clock, and app/voice.js through its scripted transport (Realtime 2.1 / Gemini Live / GPT Live / mock) with the real
// controller attached, so every check is on what would go out on the wire.   node tests/quiet_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const ROOT = path.join(__dirname, '..');
const Q = load('app/quiet.js');
let fails = 0, passes = 0;
function check(name, ok, detail) {
  if (ok) passes++; else fails++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : ''));
}
function makeClock() {
  const C = {t: 1000, q: [], n: 0};
  C.now = () => C.t;
  C.set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  C.clear = id => { C.q = C.q.filter(x => x.id !== id); };
  C.advance = ms => {
    const end = C.t + ms;
    for (;;) {
      C.q.sort((a, b) => a.at - b.at || a.id - b.id);
      const x = C.q[0];
      if (!x || x.at > end) break;
      C.q.shift(); C.t = Math.max(C.t, x.at); x.f();
    }
    C.t = end;
  };
  C.api = {now: C.now, setTimeout: C.set, clearTimeout: C.clear};
  return C;
}
const words = s => String(s).split(/\s+/).filter(Boolean).length;

// =============================================================================================== the controller alone
(function controller() {
  const clock = makeClock();
  const calls = [], el = {textContent: '', style: {}}, nods = [], th = [];
  const voice = {setTalk: t => calls.push(['talk', t]), setQuiet: (on, o) => calls.push(['quiet', on, o]), prompt: (t, o) => calls.push(['prompt', t, o])};
  const c = Q.create({clock: clock.api, voice: () => voice, el, nod: k => nods.push(k), turnBid: () => ({setThreshold: x => th.push(x)}), name: () => 'Muse'});

  // ---- be_quiet holds the full duration
  const r = c.call('be_quiet', {duration_s: 120, wake_on_speech: false});
  check('be_quiet: tool result says how long and that the voice is muted', r.ok && r.quiet_for_s === 120 && r.wake_on_speech === false && /muted/.test(r.note));
  check('be_quiet: the voice gate is closed at once (setQuiet true, a short grace for its "okay")', calls.some(x => x[0] === 'quiet' && x[1] === true && x[2].graceMs > 0 && x[2].graceMs <= 4000));
  check('be_quiet: acknowledged with a nod, not words', nods.length === 1);
  check('cue: shows quiet mode and the time left, on its own element', /Quiet/.test(el.textContent) && /2:00 left/.test(el.textContent) && el.style.display === '');
  clock.advance(30500);
  check('cue: counts down', /1:30 left/.test(el.textContent), el.textContent);
  clock.advance(89000);       // 119.5 s
  check('quiet holds to the last second (119.5 of 120 s)', c.state().quiet === true && !calls.some(x => x[0] === 'quiet' && x[1] === false));
  // wake_on_speech=false: his speech (even to it) does not end it
  check('wake_on_speech=false: addressed speech is held', c.onQuietTurn({text: 'Hey Muse, what do you think about this?', level: 0.08, durMs: 2000}) === 'hold' && c.state().quiet);
  clock.advance(600);
  const off = calls.filter(x => x[0] === 'quiet' && x[1] === false);
  check('timer end: the gate opens exactly once, with a calm period', !c.state().quiet && off.length === 1 && off[0][2].calmMs === Q.CALM_MS);
  const pr = calls.filter(x => x[0] === 'prompt');
  check('timer end: only a context note, never a request to speak (no blurting)', pr.length === 1 && pr[0][2].respond === false && /quiet time is over/i.test(pr[0][1]));
  check('cue: hidden again after the quiet period', el.textContent === '' && el.style.display === 'none');
  check('stats: one timer end', c.stats().n.timer === 1 && c.stats().n.quiet === 1);

  // ---- duration clamps / default
  check('be_quiet: no duration = 5 minutes', c.call('be_quiet', {}).quiet_for_s === 300); c.end('page');
  check('be_quiet: clamped to 5 s .. 2 h', c.call('be_quiet', {duration_s: 1}).quiet_for_s === 5 && c.call('be_quiet', {duration_s: 1e6}).quiet_for_s === 7200); c.end('page');

  // ---- early end by phrase (wake=false)
  c.call('be_quiet', {duration_s: 600, wake_on_speech: false});
  check('early end: unrelated words keep it quiet', c.onQuietTurn({text: 'I need to finish this email first', level: 0.07, durMs: 1800}) === 'hold');
  check('early end: "do not talk yet" keeps it quiet', c.onQuietTurn({text: "no don't talk yet", level: 0.07, durMs: 900}) === 'hold' && c.state().quiet);
  check('early end: "okay, you can talk again" ends it and gets an answer', c.onQuietTurn({text: 'Okay, you can talk again.', level: 0.07, durMs: 1200}) === 'wake' && !c.state().quiet);
  clock.advance(700000);
  check('early end: the old timer is gone (no second end)', c.stats().n.timer === 1 && c.stats().n.ended >= 1);
  c.call('be_quiet', {duration_s: 60, wake_on_speech: false});
  check('end_quiet tool ends it', c.call('end_quiet', {}).ok && !c.state().quiet);

  // ---- wake_on_speech=true: only real addressed speech
  const held = [
    ['ASR filler on noise', {text: 'Thank you.', level: 0.05, durMs: 500}],
    ['one stray word', {text: 'Okay.', level: 0.05, durMs: 400}],
    ['no words (a cough, a door)', {text: '', level: 0.09, durMs: 700}],
    ['its own echo', {text: 'so that is how the fox got home', level: 0.06, durMs: 1500, echo: true}],
    ['faint background speech (a TV)', {text: 'and the forecast for tomorrow is rain in the north', level: 0.009, durMs: 3000}],
    ['a click-length blip', {text: 'what is', level: 0.05, durMs: 200}],
    ['normal speech with no face in view (talking to someone else)', {text: 'yeah I will send that over after lunch', level: 0.03, durMs: 2200, attention: 0.1}],
  ];
  c.call('be_quiet', {duration_s: 600, wake_on_speech: true});
  for (const [name, seg] of held) check('wake_on_speech=true holds: ' + name, c.onQuietTurn(seg) === 'hold' && c.state().quiet, c.stats().last);
  check('wake_on_speech=true: held turns are counted, none woke it', c.stats().n.held >= held.length && c.stats().n.woke === 0);
  check('wake_on_speech=true: plain speech to it wakes it', c.onQuietTurn({text: 'What time is it in Tokyo right now?', level: 0.06, durMs: 1800, attention: 0.6}) === 'wake' && !c.state().quiet && c.stats().n.woke === 1);
  check('documented choice: the quiet period ENDS on that wake (it does not resume)', c.onQuietTurn({text: 'thanks'}) === 'wake' && c.state().left_s === 0);
  c.call('be_quiet', {duration_s: 600, wake_on_speech: true});
  check('wake_on_speech=true: its name wakes it even when faint / not facing', c.onQuietTurn({text: 'Muse?', level: 0.01, durMs: 500, attention: 0.1}) === 'wake');
  c.call('be_quiet', {duration_s: 600, wake_on_speech: true});
  check('wake_on_speech=true: loud speech wakes it without a face in view', c.onQuietTurn({text: 'can anybody tell me the capital of Peru', level: 0.07, durMs: 2000, attention: 0.1}) === 'wake');
  // a second be_quiet while quiet = a new period (extends), no second nod
  c.call('be_quiet', {duration_s: 10, wake_on_speech: false}); const n0 = nods.length; clock.advance(8000);
  c.call('be_quiet', {duration_s: 10, wake_on_speech: false}); clock.advance(8000);
  check('be_quiet again while quiet restarts the timer (still quiet 16 s in), no second nod', c.state().quiet && nods.length === n0);
  clock.advance(2500); check('…and ends 10 s after the second call', !c.state().quiet);

  // ---- end phrases
  for (const t of ['okay you can talk again', 'You may speak now.', 'alright, talk to me', 'quiet time is over', "ok, stop being quiet", 'unmute', 'you can talk now'])
    check('end phrase: "' + t + '"', Q.endPhrase(t));
  for (const t of ["don't talk", 'you can not talk yet', 'stay quiet please', 'I was talking to my sister', 'we can talk about it later', ''])
    check('not an end phrase: "' + t + '"', !Q.endPhrase(t));

  // ---- talkativeness
  check('levels: five, normal in the middle', Q.LEVELS.length === 5 && Q.LEVELS[2] === 'normal' && c.level() === 'normal');
  const less = c.call('set_talkativeness', {level: 'less'});
  check('talk less = one step down, with the rule in the tool result', less.ok && less.level === 'brief' && /one short sentence/.test(less.note));
  const lt = calls.filter(x => x[0] === 'talk').pop()[1];
  check('talk less: the voice gets a suffix, a hard token cap and a slower unprompted rate', lt.level === 'brief' && /one short sentence/.test(lt.suffix) && lt.maxTokens > 0 && lt.gapMs > Q.POLICY.normal.gapMs && lt.perMin < Q.POLICY.normal.perMin);
  check('talk less: nonverbal turn bids need a stronger cue', th[th.length - 1] > Q.POLICY.normal.turnTh);
  c.call('set_talkativeness', {level: 'less'}); c.call('set_talkativeness', {level: 'less'});
  check('talk less bottoms out at minimal (unprompted speech off)', c.level() === 'minimal' && c.policy().off === true && c.policy().perMin === 0);
  c.call('set_talkativeness', {level: 'talkative'});
  check('a level by name', c.level() === 'talkative' && c.policy().maxTokens === null && c.policy().gapMs < Q.POLICY.normal.gapMs);
  check('talk more tops out', c.call('set_talkativeness', {level: 'more'}).level === 'talkative');
  check('unknown level is refused, level kept', c.call('set_talkativeness', {level: 'loudest'}).ok === false && c.level() === 'talkative');
  const P = Q.POLICY, L = Q.LEVELS;
  check('policy is monotonic: caps grow, gaps shrink, per-minute grows, bid threshold falls',
    L.every((l, i) => !i || ((P[l].maxTokens || 1e9) >= (P[L[i - 1]].maxTokens || 1e9) && P[l].gapMs <= P[L[i - 1]].gapMs && P[l].perMin >= P[L[i - 1]].perMin && P[l].turnTh <= P[L[i - 1]].turnTh)));
  check('the asked length shrinks with the level (instruction text)', /few words/.test(P.minimal.suffix) && /one short sentence/.test(P.brief.suffix) && P.normal.suffix === '' && /two to four/.test(P.chatty.suffix) && /four to six/.test(P.talkative.suffix));
  check('the level lasts the session: a re-dialed call gets it again (attach)', (() => { const got = []; c.attach({setTalk: t => got.push(t.level), setQuiet: () => got.push('q')}); return got.join() === 'talkative'; })());
  c.call('be_quiet', {duration_s: 60, wake_on_speech: true});
  check('…and a quiet period still running', (() => { const got = []; c.attach({setTalk: () => {}, setQuiet: (on, o) => got.push([on, o.wake])}); return got.length === 1 && got[0][0] === true && got[0][1] === true; })());
  c.end('page');
})();

// =============================================================================================== voice.js on the wire
function loadVoice(clock, win) {
  const src = fs.readFileSync(path.join(ROOT, 'app/voice.js'), 'utf8');
  const w = win || {};
  new Function('window', 'performance', 'setTimeout', 'clearTimeout', src)(w, {now: clock.now}, clock.set, clock.clear);
  return w.AvatarVoice;
}
function fakeCtx() {
  const X = {gains: [], sources: [], currentTime: 0};
  X.createGain = () => { const g = {connect() { return g; }, disconnect() {}, gain: {value: 1}}; X.gains.push(g); return g; };
  X.createBufferSource = () => { const s = {connect() {}, start() { X.sources.push(s); }, stop() {}}; return s; };
  return X;
}
async function makeV(mode, extra) {
  const clock = makeClock(), ctx = fakeCtx(), sent = [], calls = [], said = [], notes = [];
  const AV = loadVoice(clock);
  const ctl = Q.create({clock: clock.api, voice: () => X.v, name: () => 'Muse', log: r => notes.push(r)});
  const X = {clock, ctx, sent, calls, said, ctl, notes, AV};
  X.v = AV.createVoice(Object.assign({mode, ctx, quiet: ctl, transport: {send: ev => sent.push(JSON.parse(JSON.stringify(ev)))},
    onCall: (name, args, phase) => { calls.push({name, args, phase}); return {ok: true}; },
    onTranscript: (who, text, fin) => { if (who === 'avatar') said.push({text, fin, t: clock.now()}); }}, extra || {}));
  await X.v.start();
  X.bus = ctx.gains[0];
  return X;
}
const TD = {type: 'server_vad', threshold: 0.6, prefix_padding_ms: 300, silence_duration_ms: 300, create_response: true, interrupt_response: true};
const BASE = 'You are Muse, a plush companion.';
const creates = s => s.filter(e => e.type === 'response.create');
const cancels = s => s.filter(e => e.type === 'response.cancel');
const tdUpdates = s => s.filter(e => e.type === 'session.update' && e.session.audio && e.session.audio.input && e.session.audio.input.turn_detection).map(e => e.session.audio.input.turn_detection.create_response);
let rid = 0, iid = 0;
function rtTool(X, name, args, spokeFirst) {       // the model's response that calls a tool
  const id = 'r' + (++rid);
  X.v.feed({type: 'response.created', response: {id}});
  if (spokeFirst) X.v.feed({type: 'response.output_audio_transcript.done', response_id: id, transcript: 'Okay.'});
  X.v.feed({type: 'response.done', response: {id, status: 'completed', output: (spokeFirst ? [{type: 'message'}] : []).concat([{type: 'function_call', name, call_id: 'c' + id, arguments: JSON.stringify(args)}])}});
  return 'c' + id;
}
function rtSay(X, text, o) {                         // he says something: VAD events, then its transcript
  o = o || {}; const item = 'it' + (++iid);
  X.v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 0, item_id: item});
  X.clock.advance(o.durMs || 1500);
  X.v.feed({type: 'test.mic_seg', seg: {level: o.level === undefined ? 0.06 : o.level, durMs: o.durMs || 1500}});
  X.v.feed({type: 'input_audio_buffer.speech_stopped', audio_end_ms: o.durMs || 1500, item_id: item});
  X.clock.advance(o.asrMs || 400);
  X.v.feed({type: 'conversation.item.input_audio_transcription.completed', item_id: item, transcript: text});
}

(async function wire() {
  // ------------------------------------------------------------------------------------------- Realtime 2.1
  {
    const X = await makeV('realtime'), {v, sent, clock} = X;
    v.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
    check('realtime: nothing is sent for the default level (sessions start exactly as before)', !sent.some(e => e.type === 'session.update'));
    // he: "be quiet for two minutes" -> the model calls be_quiet (tool-only response)
    const item0 = 'it' + (++iid);
    v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 0, item_id: item0}); clock.advance(1400);
    v.feed({type: 'input_audio_buffer.speech_stopped', audio_end_ms: 1400, item_id: item0}); clock.advance(300);
    const n0 = sent.length, call = rtTool(X, 'be_quiet', {duration_s: 120, wake_on_speech: true});
    const out = sent.find(e => e.type === 'conversation.item.create' && e.item.type === 'function_call_output' && e.item.call_id === call);
    check('realtime: be_quiet is answered by the harness (not the page onCall), result = muted for 120 s', out && JSON.parse(out.item.output).quiet_for_s === 120 && !X.calls.some(c => c.name === 'be_quiet' && c.phase === 'done'));
    check('realtime: API control — turn_detection.create_response=false, the rest of turn detection unchanged', (() => { const u = sent.slice(n0).find(e => e.type === 'session.update' && e.session.audio); const td = u && u.session.audio.input.turn_detection;
      return td && td.create_response === false && td.type === 'server_vad' && td.threshold === 0.6 && td.silence_duration_ms === 300 && td.interrupt_response === true; })());
    check('realtime: the tool-only follow-up reply is NOT created (it would be the model saying "okay I will be quiet")', creates(sent.slice(n0)).length === 0);
    check('realtime: the voice bus is still open during the short grace', X.bus.gain.value === 1 && v.quiet().on === true);
    clock.advance(900); v.feed({type: 'output_audio_buffer.stopped', response_id: 'r1'});
    check('realtime: muted by the harness the moment its "okay" has played out', X.bus.gain.value === 0 && v.quiet().muted === true);
    // the request's own transcript arrives late: it must not wake the quiet period it asked for
    v.feed({type: 'conversation.item.input_audio_transcription.completed', item_id: item0, transcript: 'Hey Muse, can you be quiet for two minutes until I talk to you?'});
    check('realtime: the late transcript of the request itself does not wake it', v.quiet().on && X.ctl.state().quiet && creates(sent.slice(n0)).length === 0);
    // everything that asks for a reply while quiet
    const n1 = sent.length;
    const p1 = v.prompt('The action he asked for is about to start. Say something short.', {respond: true});
    const p2 = v.prompt('He is signalling that he wants you to talk. Take the turn now.', {respond: true, unprompted: true});
    const e1 = v.sendEvent('He is nonverbally inviting you to talk. Take your turn.', {respond: true});
    const e2 = v.sendEvent('The camera shows: he holds up a mug (frame attached).', {respond: true, image: 'data:image/jpeg;base64,AAAA'});
    clock.advance(3000);
    check('realtime quiet: a prompted cue becomes a context note, no response.create', p1.state === 'sent' && creates(sent.slice(n1)).length === 0 && sent.slice(n1).some(e => e.type === 'conversation.item.create' && /about to start/.test(JSON.stringify(e))));
    check('realtime quiet: a cue to speak up (turn bid) is dropped entirely', p2.state === 'dropped' && p2.why === 'quiet' && e1.sent === false && e1.why === 'quiet' && !sent.slice(n1).some(e => /Take (the|your) turn/.test(JSON.stringify(e))));
    check('realtime quiet: a camera event still reaches it as context (it keeps watching), no reply', e2.sent === true && e2.responded === false && sent.slice(n1).some(e => /holds up a mug/.test(JSON.stringify(e))));
    // a response leaks (created by the server before create_response=false landed)
    const n2 = sent.length;
    v.feed({type: 'response.created', response: {id: 'leak1'}});
    v.feed({type: 'response.output_audio_transcript.delta', response_id: 'leak1', delta: 'Sure, I was just'});
    check('realtime quiet: a leaked response is cancelled and its audio buffer cleared', cancels(sent.slice(n2)).some(e => e.response_id === 'leak1') && sent.slice(n2).some(e => e.type === 'output_audio_buffer.clear'));
    check('realtime quiet: its words never reach the captions / gestures', X.said.length === 0);
    v.feed({type: 'response.done', response: {id: 'leak1', status: 'cancelled', output: []}});
    // background: noise the ASR turned into words, a far TV, a stray "okay"
    rtSay(X, 'Thank you.', {level: 0.05, durMs: 500});
    rtSay(X, 'and in sports tonight the home side took the lead early', {level: 0.008, durMs: 3200});
    rtSay(X, '', {level: 0.09, durMs: 600});
    check('realtime wake_on_speech: background noise / a TV / empty transcripts do not wake it', v.quiet().on && X.ctl.state().quiet && creates(sent.slice(n0)).length === 0, X.ctl.stats().last);
    clock.advance(60000);
    check('realtime quiet: still silent a minute in — zero response.create since be_quiet', v.quiet().on && creates(sent.slice(n0)).length === 0 && X.bus.gain.value === 0);
    // he talks to it
    const n3 = sent.length;
    rtSay(X, 'Muse, what is the tallest mountain in Japan?', {level: 0.07, durMs: 2100});
    const after = sent.slice(n3);
    check('realtime wake_on_speech: addressed speech ends the quiet period', !v.quiet().on && !X.ctl.state().quiet && X.bus.gain.value === 1);
    check('realtime wake: create_response back on, then ONE response.create answers him', tdUpdates(after).join() === 'true' && creates(after).length === 1 && after.findIndex(e => e.type === 'session.update') < after.findIndex(e => e.type === 'response.create'));
    v.feed({type: 'response.created', response: {id: 'ans1'}});
    v.feed({type: 'response.output_audio_transcript.done', response_id: 'ans1', transcript: 'Mount Fuji.'});
    check('realtime wake: that answer is not cancelled and is heard', cancels(sent.slice(n3)).length === 0 && X.said.some(s => s.text === 'Mount Fuji.'));
    v.feed({type: 'response.done', response: {id: 'ans1', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(200000);
    check('realtime wake: the old timer does not fire later', X.ctl.stats().n.timer === 0 && !v.quiet().on);
  }
  // ---- wake_on_speech=false: holds the whole time, early end by phrase; timer end without blurting
  {
    const X = await makeV('realtime'), {v, sent, clock} = X;
    v.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
    rtTool(X, 'be_quiet', {duration_s: 300, wake_on_speech: false}, true);
    clock.advance(Q.GRACE_MS + 100);
    check('realtime: with no playout event the voice is cut after the grace', X.bus.gain.value === 0);
    const n0 = sent.length;
    rtSay(X, 'Hey Muse, are you still there? What do you think?', {level: 0.08, durMs: 2500});
    check('realtime wake_on_speech=false: silent even when he talks to it', v.quiet().on && creates(sent.slice(n0)).length === 0);
    rtSay(X, "no, don't talk yet", {level: 0.08, durMs: 900});
    check('realtime: "don\'t talk yet" keeps it quiet', v.quiet().on && creates(sent.slice(n0)).length === 0);
    clock.advance(120000);
    rtSay(X, 'Okay, you can talk again.', {level: 0.08, durMs: 1200});
    check('realtime early end: "okay, you can talk again" (local transcript keyword) ends it and it answers', !v.quiet().on && creates(sent.slice(n0)).length === 1 && tdUpdates(sent.slice(n0)).join() === 'true' && X.bus.gain.value === 1);
    check('realtime early end: it is told the quiet time ended, as context', sent.slice(n0).some(e => e.type === 'conversation.item.create' && /quiet time is over/i.test(JSON.stringify(e))));
    v.feed({type: 'response.created', response: {id: 'back1'}}); v.feed({type: 'response.done', response: {id: 'back1', status: 'completed', output: [{type: 'message'}]}});
    // a full period to the timer
    clock.advance(5000);
    rtTool(X, 'be_quiet', {duration_s: 90, wake_on_speech: false}, true);
    const n1 = sent.length;
    for (let i = 0; i < 89; i++) { clock.advance(1000); if (i % 20 === 5) v.prompt('A behaviour you are running reached a moment: wave. Say something.', {respond: true, unprompted: true}); }
    check('realtime: quiet holds the full duration (89 s of 90: on, muted, zero response.create)', v.quiet().on && X.bus.gain.value === 0 && creates(sent.slice(n1)).length === 0);
    clock.advance(1100);
    const end = sent.slice(n1);
    check('realtime timer end: unmuted, create_response back on', !v.quiet().on && X.bus.gain.value === 1 && tdUpdates(end).pop() === true);
    check('realtime timer end: NO response is created (no blurting), only a context note', creates(end).length === 0 && end.some(e => e.type === 'conversation.item.create' && /may talk again when he talks to you/.test(JSON.stringify(e))));
    const pc = v.prompt('He is signalling that he wants you to talk. Take the turn now.', {respond: true, unprompted: true});
    const ec = v.sendEvent('He is nonverbally inviting you to talk. Take your turn.', {respond: true});
    check('realtime timer end: unprompted speech stays blocked for the calm period', pc.state === 'dropped' && pc.why === 'resume calm' && ec.why === 'resume calm' && creates(sent.slice(n1)).length === 0);
    const pa = v.prompt('The action he asked for could not be started. Tell him honestly.', {respond: true}); clock.advance(200);
    check('…but a reply he caused (an action note) is allowed again', creates(sent.slice(n1)).length === 1 && pa.state !== 'dropped');
    v.feed({type: 'response.created', response: {id: 'pa1'}}); v.feed({type: 'response.done', response: {id: 'pa1', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(Q.CALM_MS + 1000);
    const pl = v.prompt('He is signalling that he wants you to talk. Take the turn now.', {respond: true, unprompted: true}); clock.advance(200);
    check('after the calm period unprompted speech works as before', pl.state !== 'dropped' && creates(sent.slice(n1)).length === 2);
  }
  // ---- talkativeness on the wire (Realtime)
  {
    const X = await makeV('realtime'), {v, sent, clock} = X;
    v.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
    const n0 = sent.length;
    rtTool(X, 'set_talkativeness', {level: 'less'});
    const u = sent.slice(n0).find(e => e.type === 'session.update');
    check('talk less: session.update = the session\'s own instructions + the rule, and max_output_tokens (hard cap)', u && u.session.instructions.startsWith(BASE) && /one short sentence/.test(u.session.instructions) && u.session.max_output_tokens === Q.POLICY.brief.maxTokens);
    check('talk less: the tool-only follow-up IS created (it may say "sure")', creates(sent.slice(n0)).length === 1);
    v.feed({type: 'response.created', response: {id: 'f1'}}); v.feed({type: 'response.done', response: {id: 'f1', status: 'completed', output: [{type: 'message'}]}});
    rtTool(X, 'set_talkativeness', {level: 'minimal'});
    const u2 = sent.filter(e => e.type === 'session.update').pop();
    check('minimal: a tighter cap and the few-words rule replace the earlier one (not stacked)', u2.session.max_output_tokens === Q.POLICY.minimal.maxTokens && u2.session.max_output_tokens < Q.POLICY.brief.maxTokens && /few words/.test(u2.session.instructions) && !/one short sentence;/.test(u2.session.instructions) && u2.session.instructions.split('How much you talk').length === 2);
    v.feed({type: 'response.created', response: {id: 'f2'}}); v.feed({type: 'response.done', response: {id: 'f2', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(30000);
    const n1 = sent.length;
    const p = v.prompt('He is signalling that he wants you to talk. Take the turn now.', {respond: true, unprompted: true});
    const e = v.sendEvent('He is nonverbally inviting you to talk. Take your turn.', {respond: true});
    check('minimal: unprompted speech is off (turn bids, behaviour moments, camera remarks)', p.state === 'dropped' && p.why === 'talkativeness' && e.sent === false && e.why === 'talkativeness' && creates(sent.slice(n1)).length === 0);
    const pa = v.prompt('The action he asked for could not be started. Tell him honestly.', {respond: true}); clock.advance(200);
    check('minimal: replies he caused still happen', creates(sent.slice(n1)).length === 1 && pa.state !== 'dropped');
    v.feed({type: 'response.created', response: {id: 'f3'}}); v.feed({type: 'response.done', response: {id: 'f3', status: 'completed', output: [{type: 'message'}]}});
    // the unprompted rate per level: how many of 12 cues a minute apart by 5 s get a spoken reply
    const rate = async level => { const Y = await makeV('realtime'); Y.v.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
      Y.ctl.setTalk(level); let n = 0;
      for (let i = 0; i < 24; i++) { Y.clock.advance(5000); const r = Y.v.sendEvent('He is nonverbally inviting you to talk. Take your turn.', {respond: true}); if (r.responded) { n++; const id = 'u' + i; Y.v.feed({type: 'response.created', response: {id}}); Y.v.feed({type: 'response.done', response: {id, status: 'completed', output: [{type: 'message'}]}}); } }
      return n; };
    const R = {}; for (const l of Q.LEVELS) R[l] = await rate(l);
    console.log('     unprompted replies to 24 cues in 2 min, per level: ' + JSON.stringify(R));
    check('unprompted-speech rate gate rises with the level (minimal 0 < brief < normal < chatty <= talkative)', R.minimal === 0 && R.brief > 0 && R.brief < R.normal && R.normal < R.chatty && R.chatty <= R.talkative, R);
    rtTool(X, 'set_talkativeness', {level: 'normal'});
    const u3 = sent.filter(e => e.type === 'session.update').pop();
    check('back to normal: the original instructions, no cap', u3.session.instructions === BASE && u3.session.max_output_tokens === 'inf');
    // unit-level reply length: the cap bounds the words of one reply (>= ~10 output tokens a spoken word, measured live)
    const capWords = l => (Q.POLICY[l].maxTokens ? Q.POLICY[l].maxTokens / 10 : Infinity);
    check('reply-length caps: minimal < brief < normal (hard bound in words of one reply)', capWords('minimal') < capWords('brief') && capWords('brief') < capWords('normal') && capWords('minimal') <= 35);
    // a re-dialed call (renew / recovery) carries the level and a running quiet period
    X.ctl.setTalk('brief'); X.ctl.quiet(60, false);
    const AV2 = X.AV, sent2 = [];
    const v2 = AV2.createVoice({mode: 'realtime', ctx: fakeCtx(), quiet: X.ctl, transport: {send: ev => sent2.push(JSON.parse(JSON.stringify(ev)))}});
    await v2.start();
    v2.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
    const us = sent2.filter(e => e.type === 'session.update');
    check('re-dial: the new call gets the level (instructions + cap) and the quiet gate (create_response=false, muted)', us.some(e => /one short sentence/.test(e.session.instructions || '') && e.session.max_output_tokens === Q.POLICY.brief.maxTokens) && tdUpdates(sent2).join() === 'false' && v2.quiet().on && v2.quiet().muted);
    X.ctl.end('page');
    check('re-dial: ending the quiet period opens both calls', !v2.quiet().on && !v.quiet().on && tdUpdates(sent2).pop() === true);
  }
  // ---- real-session failures (a session judge's "unsolicited_speech" finding): the requirements behind them
  {
    const LOOK_BACK = 'He just looked back at you after looking away. Say something tiny and warm to him, in your own words.';   // page.html SPEAK_EVENTS
    const X = await makeV('realtime'), {v, sent, clock} = X;
    v.feed({type: 'session.created', session: {instructions: BASE, audio: {input: {turn_detection: TD}}}});
    clock.advance(20000);
    // never a second utterance over one still playing or being made
    v.feed({type: 'response.created', response: {id: 'long1'}});
    const b1 = v.sendEvent(LOOK_BACK, {respond: true});
    check('no overlap: an unprompted check-in is dropped while a reply is being generated', b1.sent === false && b1.why === 'reply in progress' && creates(sent).length === 0);
    v.feed({type: 'output_audio_buffer.started', response_id: 'long1'});
    v.feed({type: 'response.done', response: {id: 'long1', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(4000);       // generated, still playing out
    const b2 = v.sendEvent(LOOK_BACK, {respond: true});
    const b3 = v.prompt('He is signalling that he wants you to talk. Take the turn now.', {respond: true, unprompted: true, maxWaitMs: 1500});
    clock.advance(2000);
    check('no overlap: …and while its audio is still playing out (look_back dropped, a turn bid goes stale)', b2.sent === false && b2.why === 'avatar speaking' && b3.state === 'dropped' && creates(sent).length === 0, [b2, b3.state, b3.why]);
    v.feed({type: 'output_audio_buffer.stopped', response_id: 'long1'}); clock.advance(12000);
    const b4 = v.sendEvent(LOOK_BACK, {respond: true});
    check('…once it has finished, a look_back may be answered again (normal level)', b4.responded === true && creates(sent).length === 1);
    v.feed({type: 'response.created', response: {id: 'lb1'}}); v.feed({type: 'response.done', response: {id: 'lb1', status: 'completed', output: [{type: 'message'}]}});
    // quiet: the look_back check-ins that broke "be quiet for two minutes" at 4:17 / 5:12 / 6:09
    clock.advance(15000);
    rtTool(X, 'be_quiet', {duration_s: 120, wake_on_speech: false});
    const n0 = sent.length; let spoke = 0;
    for (let i = 0; i < 11; i++) { clock.advance(10000); const r = v.sendEvent(LOOK_BACK, {respond: true}); if (r.responded || r.sent) spoke++; }
    check('quiet gates proactive look_back: 11 look-backs in 110 s of a 2-minute quiet -> none spoken, none left as context', spoke === 0 && creates(sent.slice(n0)).length === 0 && !sent.slice(n0).some(e => /looked back/.test(JSON.stringify(e))));
    clock.advance(11000);
    // stale replies (41-61 s late in his session): audio that starts > 5 s after its reply was created, after he spoke again
    v.feed({type: 'response.created', response: {id: 'old1'}});
    v.feed({type: 'response.done', response: {id: 'old1', status: 'completed', output: [{type: 'message'}]}});
    clock.advance(3000); v.feed({type: 'input_audio_buffer.speech_started', audio_start_ms: 0, item_id: 'itx'}); clock.advance(800);
    v.feed({type: 'input_audio_buffer.speech_stopped', audio_end_ms: 800, item_id: 'itx'}); clock.advance(3000);
    const n1 = sent.length;
    v.feed({type: 'output_audio_buffer.started', response_id: 'old1'});
    check('stale reply: audio starting 6.8 s after its reply was made, after he spoke again, is cancelled and cleared', cancels(sent.slice(n1)).some(e => e.response_id === 'old1') && sent.slice(n1).some(e => e.type === 'output_audio_buffer.clear') && v.quiet().stale === 1);
    v.feed({type: 'response.created', response: {id: 'ok1'}}); clock.advance(700);
    const n2 = sent.length; v.feed({type: 'output_audio_buffer.started', response_id: 'ok1'});
    check('a reply that starts on time is never touched', cancels(sent.slice(n2)).length === 0 && v.quiet().stale === 1);
    v.feed({type: 'response.done', response: {id: 'ok1', status: 'completed', output: [{type: 'message'}]}}); v.feed({type: 'output_audio_buffer.stopped', response_id: 'ok1'});
    v.feed({type: 'response.created', response: {id: 'slow1'}}); clock.advance(7000);
    const n3 = sent.length; v.feed({type: 'output_audio_buffer.started', response_id: 'slow1'});
    check('a slow reply he has not talked over is not stale', cancels(sent.slice(n3)).length === 0);
    v.feed({type: 'response.done', response: {id: 'slow1', status: 'completed', output: [{type: 'message'}]}}); v.feed({type: 'output_audio_buffer.stopped', response_id: 'slow1'});
    // "Be quiet for a—" (cut off before the time) … "Two minutes."
    clock.advance(5000);
    rtTool(X, 'be_quiet', {duration_s: 300, wake_on_speech: false});
    const n4 = sent.length;
    rtSay(X, 'Two minutes.', {level: 0.07, durMs: 700});
    check('a bare duration right after be_quiet sets the period (5:00 -> 2:00), silently', X.ctl.state().quiet && Math.abs(X.ctl.state().left_s - 120) < 1 && creates(sent.slice(n4)).length === 0, X.ctl.state());
    clock.advance(30000); rtSay(X, 'ten minutes', {level: 0.07, durMs: 700});
    check('…but not later in the period (that is just talk)', X.ctl.state().left_s < 100);
    X.ctl.end('page');
    // "stay quiet until requested": stray fragments are not requests
    X.ctl.quiet(600, true); clock.advance(3000);
    const n5 = sent.length;
    for (const t of ['Thank you. This is the kind of', "It's not really good.", 'I have been nonstop upgrading', 'Okay, I guess.', 'eh'])
      rtSay(X, t, {level: 0.06, durMs: 1200});
    check('wake_on_speech: statements not said to it (his own fragments, noise turned into words) do not wake it', X.ctl.state().quiet && creates(sent.slice(n5)).length === 0, X.ctl.stats().last);
    rtSay(X, 'Teach me about quantum mechanics.', {level: 0.06, durMs: 1500});
    check('wake_on_speech: a request does', !X.ctl.state().quiet && creates(sent.slice(n5)).length === 1);
    check('durations: parsed', Q.durationOf('Two minutes.') === 120 && Q.durationOf('for 90 seconds') === 90 && Q.durationOf('half an hour') === 1800 && Q.durationOf('a couple of minutes') === 120 && Q.durationOf('two minutes ago I saw it') === 0 && Q.durationOf('minutes') === 0);
  }
  // ------------------------------------------------------------------------------------------- Gemini Live
  {
    const X = await makeV('gemini'), {v, sent, clock} = X;
    const audio = {inlineData: {mimeType: 'audio/pcm;rate=24000', data: Buffer.from(new Int16Array(480).buffer).toString('base64')}};
    v.feed({serverContent: {inputTranscription: {text: 'please be quiet for a minute'}}});
    v.feed({toolCall: {functionCalls: [{id: 'g1', name: 'be_quiet', args: {duration_s: 60, wake_on_speech: true}}]}});
    const tr = sent.find(e => e.toolResponse);
    check('gemini: be_quiet answered by the harness', tr && tr.toolResponse.functionResponses[0].response.quiet_for_s === 60 && v.quiet().on);
    clock.advance(1500);
    // the model talks anyway (after the tool result): dropped + interrupted once
    const n0 = sent.length;
    v.feed({serverContent: {modelTurn: {parts: [audio]}}});
    v.feed({serverContent: {modelTurn: {parts: [audio]}, outputTranscription: {text: 'Okay, I will be quiet now.'}}});
    v.feed({serverContent: {modelTurn: {parts: [audio]}}});
    const ints = sent.slice(n0).filter(e => e.clientContent && e.clientContent.turnComplete === false && /Quiet mode/.test(JSON.stringify(e)));
    check('gemini quiet: its turn audio is dropped (3 of 3 chunks) and never captioned', v.quiet().audioDropped === 3 && X.said.length === 0);
    check('gemini quiet: the generation is interrupted with ONE silent note', ints.length === 1 && v.quiet().interrupts === 1);
    v.feed({serverContent: {interrupted: true}}); v.feed({serverContent: {turnComplete: true}});
    // a page cue that asks for a reply: context only
    const p = v.prompt('The action he asked for is about to start. Say something short.', {respond: true}); clock.advance(300);
    check('gemini quiet: a prompted cue goes out with turnComplete:false (no reply asked)', sent.some(e => e.clientContent && /about to start/.test(JSON.stringify(e)) && e.clientContent.turnComplete === false) && !sent.slice(n0).some(e => e.clientContent && e.clientContent.turnComplete === true), p.state);
    // background words, model stays silent: the harness decides after the pause
    v.feed({type: 'test.mic_seg', seg: {level: 0.007, durMs: 2500}});
    v.feed({serverContent: {inputTranscription: {text: ' and now the weather'}}}); v.feed({serverContent: {inputTranscription: {text: ' for the weekend'}}});
    clock.advance(1000);
    check('gemini wake_on_speech: faint background speech does not wake it', v.quiet().on && X.ctl.state().quiet && !sent.slice(n0).some(e => e.clientContent && e.clientContent.turnComplete === true));
    clock.advance(20000);
    const n1 = sent.length;
    v.feed({type: 'test.mic_seg', seg: {level: 0.07, durMs: 1800}});
    v.feed({serverContent: {inputTranscription: {text: 'Muse, how do you'}}}); clock.advance(300);
    v.feed({serverContent: {inputTranscription: {text: ' say hello in French?'}}});
    check('gemini: no decision mid-utterance', v.quiet().on);
    clock.advance(950);
    check('gemini wake_on_speech: addressed speech ends the quiet period and asks for the answer (the model had stayed silent)', !v.quiet().on && !X.ctl.state().quiet && sent.slice(n1).filter(e => e.clientContent && e.clientContent.turnComplete === true).length === 1);
    v.feed({serverContent: {modelTurn: {parts: [audio]}, outputTranscription: {text: 'Bonjour!'}}}); v.feed({serverContent: {turnComplete: true}});
    check('gemini wake: the answer is played and captioned', v.quiet().audioDropped === 3 && X.said.some(s => /Bonjour/.test(s.text)));
    // wake=false + early end phrase when the model replies at once (decision at the reply's first chunk)
    clock.advance(3000);
    v.feed({toolCall: {functionCalls: [{id: 'g2', name: 'be_quiet', args: {duration_s: 60, wake_on_speech: false}}]}}); clock.advance(1500);
    v.feed({serverContent: {inputTranscription: {text: 'hey Muse are you there'}}});
    const d0 = v.quiet().audioDropped;
    v.feed({serverContent: {modelTurn: {parts: [audio]}}}); v.feed({serverContent: {turnComplete: true}});
    check('gemini wake_on_speech=false: its reply to him is dropped', v.quiet().on && v.quiet().audioDropped === d0 + 1);
    clock.advance(1000);
    v.feed({serverContent: {inputTranscription: {text: 'okay you can talk again'}}});
    v.feed({serverContent: {modelTurn: {parts: [audio]}, outputTranscription: {text: 'I am back!'}}});
    check('gemini early end: the phrase opens the gate at the reply\'s first chunk — that reply is heard', !v.quiet().on && v.quiet().audioDropped === d0 + 1 && X.said.some(s => /I am back/.test(s.text)));
    v.feed({serverContent: {turnComplete: true}});
    // end_quiet as a tool call
    v.feed({toolCall: {functionCalls: [{id: 'g3', name: 'be_quiet', args: {duration_s: 60, wake_on_speech: false}}]}}); clock.advance(1500);
    v.feed({toolCall: {functionCalls: [{id: 'g4', name: 'end_quiet', args: {}}]}});
    check('gemini early end: end_quiet (a tool-only turn passes the gate)', !v.quiet().on && !X.ctl.state().quiet);
    // timer
    v.feed({toolCall: {functionCalls: [{id: 'g5', name: 'be_quiet', args: {duration_s: 30, wake_on_speech: false}}]}});
    clock.advance(29000); const on29 = v.quiet().on; const n2 = sent.length; clock.advance(1500);
    check('gemini: holds 29 of 30 s, then the timer ends it with no reply asked', on29 && !v.quiet().on && !sent.slice(n2).some(e => e.clientContent && e.clientContent.turnComplete === true));
    // talk less: a silent note + the harness length backstop
    clock.advance(30000);
    const n3 = sent.length;
    v.feed({toolCall: {functionCalls: [{id: 'g6', name: 'set_talkativeness', args: {level: 'brief'}}]}}); clock.advance(500);
    check('gemini talk less: the rule goes in as a silent note (turnComplete:false)', sent.slice(n3).some(e => e.clientContent && e.clientContent.turnComplete === false && /one short sentence/.test(JSON.stringify(e))));
    const d1 = v.quiet().audioDropped; let long = '';
    for (let i = 0; i < 12; i++) { const t = ' word'.repeat(5); long += t; v.feed({serverContent: {modelTurn: {parts: [audio]}, outputTranscription: {text: t}}}); }
    check('gemini talk less: a reply far past the asked length is cut by the harness (audio after ' + Q.POLICY.brief.gemWords + ' words dropped)', v.quiet().capped === 1 && v.quiet().audioDropped - d1 >= 4 && v.quiet().audioDropped - d1 <= 6, v.quiet());
    v.feed({serverContent: {turnComplete: true}});
    v.feed({serverContent: {modelTurn: {parts: [audio]}, outputTranscription: {text: 'Short one.'}}});
    check('gemini talk less: the next (short) reply plays', v.quiet().audioDropped - d1 <= 6 && X.said.some(s => /Short one/.test(s.text)));
  }
  // ------------------------------------------------------------------------------------------- GPT Live 1 (muted + notes)
  {
    const X = await makeV('live'), {v, sent, clock} = X;
    X.ctl.quiet(60, true); clock.advance(Q.GRACE_MS + 100);
    const n0 = sent.length;
    v.prompt('The action he asked for is about to start. Say something short.', {respond: true}); clock.advance(300);
    check('gpt live quiet: muted; a spoken note becomes context-only (thinking.append)', X.bus.gain.value === 0 && !sent.slice(n0).some(e => e.type === 'session.instructions.append') && sent.slice(n0).some(e => e.type === 'session.thinking.append'));
    v.feed({type: 'test.mic_seg', seg: {level: 0.07, durMs: 1500}});
    v.feed({type: 'session.input_transcript.delta', delta: 'Muse, can you hear me?'}); clock.advance(1000);
    check('gpt live wake_on_speech: addressed speech unmutes and asks for the answer', !v.quiet().on && X.bus.gain.value === 1 && sent.slice(n0).some(e => e.type === 'session.instructions.append'));
  }
  // ------------------------------------------------------------------------------------------- mock voice
  {
    const X = await makeV('mock', {mock: {latMs: 100}}), {v, clock, ctx} = X;
    const bufs = [{url: 'mock/reply1.wav', buf: {}}];
    v.feed({type: 'test.mock_user_end', bufs}); clock.advance(200);
    check('mock: replies normally before quiet', ctx.sources.length === 1);
    X.ctl.quiet(30, false); clock.advance(100);
    check('mock quiet: muted at once when the page (not the model) starts it… after the grace', v.quiet().on);
    clock.advance(Q.GRACE_MS);
    v.feed({type: 'test.mic_seg', seg: {level: 0.07, durMs: 1500}}); v.feed({type: 'test.mock_user_end'}); clock.advance(500);
    v.feed({type: 'test.mock_user_end'}); clock.advance(500);
    check('mock quiet (wake_on_speech=false): no reply WAV while quiet', ctx.sources.length === 1 && X.bus.gain.value === 0);
    clock.advance(28000);
    check('mock: the timer ends it, no reply by itself', !v.quiet().on && ctx.sources.length === 1 && X.bus.gain.value === 1);
    v.feed({type: 'test.mock_user_end'}); clock.advance(200);
    check('mock: replies again on his next turn', ctx.sources.length === 2);
    X.ctl.quiet(30, true); clock.advance(Q.GRACE_MS);
    v.feed({type: 'test.mic_seg', seg: {level: 0.004, durMs: 300}}); v.feed({type: 'test.mock_user_end'}); clock.advance(300);
    check('mock wake_on_speech: a faint blip does not wake it', v.quiet().on && ctx.sources.length === 2);
    v.feed({type: 'test.mic_seg', seg: {level: 0.06, durMs: 1400}}); v.feed({type: 'test.mock_user_end'}); clock.advance(300);
    check('mock wake_on_speech: a real utterance wakes it and gets the reply', !v.quiet().on && ctx.sources.length === 3);
  }
  // ------------------------------------------------------------------------------------------- the live measurement (recorded)
  {
    // tests/quiet_live.py against gpt-realtime (paid, run by hand): the real session, the exact session.update of each level
    const fx = path.join(__dirname, 'fixtures', 'quiet_live.json');
    if (fs.existsSync(fx)) {
      const L = JSON.parse(fs.readFileSync(fx, 'utf8')), m = l => L.levels[l] && L.levels[l].mean_words;
      console.log('     live (' + L.model + '): mean reply words ' + Object.keys(L.levels).map(l => l + ' ' + m(l)).join(', ') + '; ' + L.out_tokens_per_word.mean + ' output tokens a word');
      check('live record: talk less / more measurably changes reply length (brief < 0.7 x normal, talkative > 1.5 x normal)', m('brief') < 0.7 * m('normal') && m('talkative') > 1.5 * m('normal'), L.levels);
      check('live record: no reply at a capped level was cut by its cap (the cap is a backstop)', Object.values(L.levels).every(v => v.rows.every(r => r.status === 'completed')));
      check('live record: the caps leave >= 2x room over the measured replies', ['minimal', 'brief'].every(l => !L.levels[l] || L.levels[l].rows.every(r => r.output_tokens * 2 <= Q.POLICY[l].maxTokens)));
      check('live record: asked in words, the model calls be_quiet / set_talkativeness', L.tools.length >= 2 && L.tools.every(t => t.ok));
    } else console.log('     (no live record: tests/fixtures/quiet_live.json)');
  }
  // ------------------------------------------------------------------------------------------- the session builders + page wiring
  {
    const rtc = fs.readFileSync(path.join(ROOT, 'app/rtc.py'), 'utf8'), page = fs.readFileSync(path.join(ROOT, 'app/page.html'), 'utf8'),
      app = fs.readFileSync(path.join(ROOT, 'app/app.py'), 'utf8'), vt = fs.readFileSync(path.join(ROOT, 'app/voicetools.js'), 'utf8');
    check('rtc.py: the three tools are declared for every realtime_session (Gemini Live builds on it)', Q.TOOLS.every(n => new RegExp('"name": "' + n + '"').test(rtc)) && /def realtime_session\(name: str, what: str, \*a, \*\*k\) -> dict:\s+# noqa: F811/.test(rtc));
    check('rtc.py: be_quiet is acknowledged silently (the call is the answer; the harness nods)', /for be_quiet say nothing at all/.test(rtc) && /without saying anything/.test(rtc));
    check('rtc.py: the tool levels are the controller\'s levels', Q.LEVELS.every(l => rtc.includes('"' + l + '"')));
    check('page: the controller is created for every character and voice, with the cue outside the stage canvas', /window\.AvatarQuietCtl = D\.quiet = AvatarQuiet\.create\(/.test(page) && /AvatarQuiet\.cue\(document, 'caps'\)/.test(page) && !/quietcue/.test(page.slice(page.indexOf('<div id="stage">'), page.indexOf('<div id="caps">'))));
    check('app.py: quiet.js is inlined with the page modules', /"quiet\.js",/.test(app));
    check('voicetools: its cues to speak up are marked unprompted (turn bid, taught gesture, attention, behaviour moment)', (vt.match(/unprompted: true/g) || []).length === 5);   // + fixes: the claim-check note (narration without motion)
    // the cue element
    const made = []; const doc = {getElementById: id => (id === 'caps' ? {parentNode: {insertBefore: (el, ref) => made.push(el)}, nextSibling: null} : null),
      createElement: () => ({style: {}, setAttribute() {}})};
    const el = Q.cue(doc, 'caps');
    check('cue element: inserted after the captions, hidden until needed', made.length === 1 && el.id === 'quietcue' && /display:none/.test(el.style.cssText));
  }
  console.log(`\n${passes} passed, ${fails} failed`);
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
