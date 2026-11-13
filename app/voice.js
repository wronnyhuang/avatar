/* voice.js — the avatar's DIRECT voice: one WebRTC session from this page straight to OpenAI
   (gpt-realtime-2.1 by default, gpt-live-1 as the switch-back), or 'mock' (scripted reply WAVs, no vendor).
   The app server only brokers the SDP (/api/rtc — the key never reaches the browser) and meters spend.

   window.AvatarVoice.createVoice({mode, character, vad, ctx, micStream, rtcUrl, rtcHeaders, mock:{urls, latMs},
       onCall(name, args, phase, info), onState(state, detail), onMetric(m), onEvent(ev), onTranscript(who, text, final),
       grabStill() -> dataURL|null})
     -> {start(), stop(), sendEvent(text, {image, respond}), sendToolResult(id, out), audioNode, stats(), userVoice,
         speaking(), listening(), prompt(note, {respond, maxWaitMs, dropIfUserSpoke, onSent, onAudio, onDone, onDrop})}
   prompt(): a SILENT note (never read aloud: the SILENT prefix) that may ask for a spoken response — the page's
     cue / check-in / honest-failure notes around a planned action. Realtime: the note item at once, its response.create
     only when no response is active or being created and the user is not speaking (queued, dropped when stale); Gemini Live:
     one clientContent turn (turnComplete = respond) once the model is idle (client content interrupts a generation);
     GPT Live: instructions.append (respond) / thinking.append. onAudio fires at the prompted reply's first audio.
     Nothing here ever holds, gates or delays the avatar's own audio.
   audioNode = a GainNode carrying the avatar's voice (remote track or mock WAVs). The page graph puts it through
   AvatarVoice.createPlayout(): an AudioWorklet tap (24 kHz int16, 20 ms chunks, silent runs flagged, on the playback
   timeline that starts at 0 with the voice session) -> the app server, and a DelayNode (= measured video round trip) -> speakers.
*/
(() => {
  'use strict';
  const TSR = 24000;
  const AEC_JB_MS = 60;   // buzz: the echo-canceller loopback's jitter-buffer target (see aecOutput)
  const WORKLET = `
const TSR = 24000;
class Tap extends AudioWorkletProcessor {
  // The avatar's voice as it enters the delay line, resampled to 24 kHz, 20 ms chunks on the playback timeline.
  constructor() {
    super(); this.on = false; this.s = 0; this.buf = new Int16Array(480); this.n = 0; this.ph = 0; this.prev = 0;
    this.step = sampleRate / TSR; this.peak = 0; this.e = 0; this.first = -1;
    this.port.onmessage = e => { const m = e.data;
      if (m.t === 'start') { this.on = true; this.s = 0; this.n = 0; this.startFrame = currentFrame; this.port.postMessage({t: 'started', ct: currentTime, frame: currentFrame}); }
      else if (m.t === 'stop') this.on = false; };
  }
  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0] && outputs[0][0]; if (out) out.fill(0);
    if (!this.on) return true;
    const L = 128;
    for (let i = 0; i < L; i++) {
      const x = inp ? inp[i] : 0;
      if (this.step === 1) this.push(x);
      else { while (this.ph <= 1) { this.push(this.prev + (x - this.prev) * this.ph); this.ph += this.step; } this.ph -= 1; this.prev = x; }
    }
    return true;
  }
  push(x) {
    const v = x > 1 ? 1 : x < -1 ? -1 : x; const q = Math.round(v * 32767);
    const a = q < 0 ? -q : q; if (a > this.peak) this.peak = a; if (a > 40 && this.first < 0) this.first = this.n;
    this.e += q * q; this.buf[this.n++] = q;
    if (this.n === 480) {
      const silent = this.peak < 24;
      const msg = {t: 'c', s0: this.s, silent, rms: Math.sqrt(this.e / 480), first: this.first, ct: currentTime};
      if (!silent) { msg.pcm = this.buf; this.port.postMessage(msg, [this.buf.buffer]); this.buf = new Int16Array(480); }
      else this.port.postMessage(msg);
      this.s += 480; this.n = 0; this.peak = 0; this.e = 0; this.first = -1;
    }
  }
}
class Meter extends AudioWorkletProcessor {
  // mic level every 10 ms (local end-of-speech for metrics + the mock vendor)
  constructor() { super(); this.e = 0; this.n = 0; this.N = Math.round(sampleRate / 100); }
  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0]; const out = outputs[0] && outputs[0][0]; if (out) out.fill(0);
    if (!inp) return true;
    for (let i = 0; i < inp.length; i++) { this.e += inp[i] * inp[i]; if (++this.n === this.N) {
      this.port.postMessage({rms: Math.sqrt(this.e / this.N), ct: currentTime + (i + 1) / sampleRate}); this.e = 0; this.n = 0; } }
    return true;
  }
}
// Gemini Live: the (gated) mic as 16 kHz PCM16, 20 ms per message
class Pcm16Tap extends AudioWorkletProcessor {
  constructor() { super(); this.r = sampleRate / 16000; this.acc = 0; this.buf = new Int16Array(320); this.n = 0; }
  process(inputs, outputs) {
    const x = inputs[0] && inputs[0][0]; const o = outputs[0] && outputs[0][0]; if (o) o.fill(0);
    if (!x) return true;
    for (let i = 0; i < x.length; i++) { this.acc += 1; if (this.acc >= this.r) { this.acc -= this.r;
      const v = Math.max(-1, Math.min(1, x[i])); this.buf[this.n++] = v * 32767;
      if (this.n === 320) { this.port.postMessage(this.buf.slice(0)); this.n = 0; } } }
    return true;
  }
}
// Gemini Live: plays 24 kHz PCM16 chunks (port) at the context rate; {clear:true} drops what is queued (barge-in)
class PcmPlayer extends AudioWorkletProcessor {
  constructor() { super(); this.q = []; this.cur = null; this.pos = 0; this.step = 24000 / sampleRate; this.frac = 0;
    this.port.onmessage = e => { const d = e.data; if (d && d.clear) { this.q = []; this.cur = null; this.pos = 0; return; } if (d && d.length) this.q.push(d); }; }
  process(inputs, outputs) {
    const o = outputs[0] && outputs[0][0]; if (!o) return true;
    for (let i = 0; i < o.length; i++) {
      if (!this.cur || this.pos >= this.cur.length - 1) { if (this.cur && this.pos >= this.cur.length - 1) { this.pos -= this.cur.length - 1; } this.cur = this.q.shift() || null; if (!this.cur) { o.fill(0, i); this.pos = 0; return true; } }
      const k = Math.floor(this.pos), f = this.pos - k; o[i] = ((this.cur[k] * (1 - f)) + (this.cur[k + 1] || this.cur[k]) * f) / 32768; this.pos += this.step;
    }
    return true;
  }
}
registerProcessor('avatar-pcm16-tap', Pcm16Tap);
registerProcessor('avatar-pcm-player', PcmPlayer);
registerProcessor('avatar-voice-tap', Tap);
registerProcessor('avatar-mic-meter', Meter);
class Gate extends AudioWorkletProcessor {
  // Half-duplex / barge-in gate on the (already echo-cancelled) mic, referenced to the avatar's own output.
  // input 0 = mic, input 1 = the avatar voice as it leaves for the speaker. 10 ms frames:
  //  refEnv = max avatar level over the last 450 ms (covers speaker + acoustic + capture lag; plus the tail below);
  //  c      = learned echo coupling (mic / refEnv while only the avatar talks; lower-envelope tracker);
  //  while the avatar is audible (or 450 ms after), the mic passes only if it is well above the predicted
  //  echo (k x c x refEnv, and an absolute floor) for 30 ms -> barge-in; otherwise it is attenuated (-30 dB,
  //  or silence in hard half-duplex mode). Gain moves smoothly (attack 3 ms, release 80 ms).
  // (phone speakerphone): the reference is tapped BEFORE the WebRTC loopback, the <audio>
  //  element and the phone's output path, so on Android the echo can reach the mic more than 450 ms later — the gate
  //  re-opened during the echo of the last words ("Yes." / "Sure." / "OK." came back as user turns). Now:
  //  - after the reference stops the gate stays closed until the mic is back to its background level (the echo has
  //    arrived and died away; strict max 1.2 s), not a fixed 450 ms;
  //  - a barge-in must beat the loudest echo heard during this reply (the old coupling estimate tracked the quietest
  //    moments and collapsed to ~0.01, so any loud speaker echo counted as the user talking), and none in the first second;
  //  - strict (hard) mode: silence while the avatar talks + tail, except a LOUD barge-in (well above the predicted echo
  //    and an absolute floor, sustained 150 ms) and an explicit open ({openMs}) for the page's tap-to-interrupt.
  constructor() {
    super(); this.N = Math.round(sampleRate / 100); this.n = 0; this.em = 0; this.er = 0;
    this.ring = new Float32Array(45); this.ri = 0; this.c = 0.35; this.open = 0; this.hold = 0; this.g = 1; this.target = 1;
    this.hard = false; this.k = 2.5; this.floor = 0.012; this.refOn = 0.006; this.closedGain = 0.003;
    // tail = closed after the reference stops until the mic itself is back to its background level for 150 ms
    // (min / max: normal 100 / 600 ms, strict 250 / 1000 ms); echoMax = the loudest mic level heard during this reply
    // while closed — a barge-in must beat it (normal 1.5x, strict 2.5x) as well as k x c x refEnv and a floor
    this.quiet = 1e9; this.idle = 1e9; this.calib = 0; this.calm = 0; this.bg = 0.002; this.echoMax = 0; this.inReply = false;
    this.hardFloor = 0.05; this.hardN = 15; this.hardBarge = true; this.force = 0; this.tailMax = 100;
    this.st = {frames: 0, refFrames: 0, closedFrames: 0, bargeIns: 0, tailFrames: 0, forced: 0, held: 0, replayed: 0, skipped: 0, maxq: 0, mutedFrames: 0}; this.lastReport = 0; this.wasClosed = false;
    // hold: while the avatar's sentence is audible the (gated) mic is queued, not sent (the vendor clears its audio the instant it
    // hears speech); when the sentence has ended the queue is replayed in real time and its silence is dropped, so the delay does not stay
    this.muted = false; this.refOff = false;   // mic muted (frozen, silent) / avatar voice muted at the speaker (no reference)
    this.rq = 0; this.refSeen = false; this.hLoud = false; this.PAUSEQ = 30; this.LONGQ = 70; this.MINWAIT = 2; this.hSince = 0;   // avatar-voice quiet frames (10 ms): pause / long pause; the user's words wait >= 2 s before an ordinary pause releases
    this.sh = false; this.q = []; this.silRun = 0; this.SILQ = Math.round(0.35 * sampleRate / 128); this.MAXQ = Math.round(15 * sampleRate / 128);
    this.port.onmessage = e => { const d = e.data || {}; if ('hard' in d) this.hard = !!d.hard; if (d.k) this.k = d.k; if (d.floor) this.floor = d.floor;
      if ('hardBarge' in d) this.hardBarge = !!d.hardBarge; if (d.tailMs) this.tailMax = Math.round(d.tailMs / 10);
      if (d.openMs) { this.force = Math.round(d.openMs / 10); this.open = 0; this.hold = 0; this.sh = false; this.q.length = 0; }   // tap-to-interrupt: now, nothing held
      if ('mute' in d) { this.muted = !!d.mute; this.q.length = 0; this.sh = false; this.hold = 0; this.open = 0; this.g = 0; this.target = this.muted ? 0 : 1; }
      if ('refOff' in d) { this.refOff = !!d.refOff; if (this.refOff) { this.q.length = 0; this.sh = false; } }
      if ('shield' in d) { this.sh = !!d.shield; if (this.sh) { this.silRun = this.SILQ; this.refSeen = false; this.hLoud = false; this.rq = 0; this.hSince = 0; } } };
  }
  frame(m, r) {
    this.ring[this.ri] = r; this.ri = (this.ri + 1) % this.ring.length;
    if (this.muted) { this.st.mutedFrames++; this.target = 0; return; }   // nothing is learned from a muted (silent) mic: the echo coupling / background estimates stay as they were
    let env = 0; for (let i = 0; i < this.ring.length; i++) if (this.ring[i] > env) env = this.ring[i];
    const st = this.st; st.frames++;
    if (this.force > 0) { this.force--; st.forced++; this.target = 1; this.inReply = false; return; }   // tap-to-interrupt
    if (r >= this.refOn) { this.refSeen = true; this.rq = 0; } else this.rq++;   // hold: the avatar's audible voice, frame by frame
    const refActive = env > this.refOn;
    if (refActive) { if (!this.inReply) { this.calib = 100; this.echoMax = 0; } this.inReply = true; this.quiet = 0; this.calm = 0; }
    else this.quiet++;
    if (this.inReply && !refActive) {   // the tail: is the mic back to its background (no more echo arriving)?
      const minT = this.hard ? 25 : 10, maxT = this.hard ? this.tailMax : 60;
      this.calm = m < Math.max(3 * this.bg, 0.004) ? this.calm + 1 : 0;
      if ((this.quiet >= minT && this.calm >= 15) || this.quiet >= maxT) this.inReply = false;
      else { st.tailFrames++; env = Math.max(env, this.refOn); }
    }
    if (!this.inReply) {
      if (m < 3 * this.bg + 0.004) this.bg += (m - this.bg) * (m < this.bg ? 0.05 : 0.002);   // background mic level
      this.open = 0; this.hold = 0; this.target = 1; this.wasClosed = false; return;
    }
    st.refFrames++;
    if (this.calib > 0) this.calib--;
    // in the tail the speaker is already silent: the echo can only fade from what this reply measured, so the user's voice
    // clearly above that (2x, 50 ms) passes at once instead of waiting for the tail to run out
    const tailNow = !refActive;
    const thr = tailNow ? Math.max(this.floor, 2 * this.echoMax)
      : this.hard ? Math.max(this.hardFloor, 2.5 * this.echoMax, 6 * this.c * env)
                  : Math.max(this.floor, 1.5 * this.echoMax, this.k * this.c * env);
    const need = tailNow ? 5 : this.hard ? this.hardN : 3;
    if ((!this.hard || this.hardBarge) && this.calib <= 0 && m > thr) { if (++this.open >= need) { if (this.hold <= 0) st.bargeIns++; this.hold = 30; } }
    else { this.open = 0; if (this.hold > 0 && m < 0.6 * thr) this.hold--; }
    const passing = this.hold > 0;
    // the echo level: freely while calibrating (reply onset), afterwards only from frames that are not a barge-in candidate
    if (!passing && (this.calib > 0 || m <= thr)) { if (m > this.echoMax) this.echoMax = m; else this.echoMax *= 0.9995; }
    // learn the coupling only from frames that look like echo alone (ignore probable double-talk)
    if (this.hold <= 0 && r > this.refOn) {
      const ratio = m / Math.max(env, 1e-4);
      if (ratio < 2.5 * this.c) this.c += (Math.min(2, Math.max(0.005, ratio)) - this.c) * (ratio < this.c ? 0.03 : 0.01);
    }
    this.target = passing ? 1 : (this.hard ? 0 : this.closedGain);
    if (!passing) st.closedFrames++;
  }
  process(inputs, outputs) {
    const mic = inputs[0] && inputs[0][0]; const ref = inputs[1] && inputs[1][0]; const out = outputs[0] && outputs[0][0];
    if (!out) return true;
    if (!mic) { out.fill(0); return true; }
    const aUp = 1 - Math.exp(-1 / (0.003 * sampleRate)), aDn = 1 - Math.exp(-1 / (0.08 * sampleRate));
    for (let i = 0; i < mic.length; i++) {
      const x = mic[i], y = ref && !this.refOff ? ref[i] : 0; this.em += x * x; this.er += y * y;
      if (++this.n === this.N) { this.frame(Math.sqrt(this.em / this.N), Math.sqrt(this.er / this.N)); this.em = 0; this.er = 0; this.n = 0; }
      this.g += (this.target - this.g) * (this.target > this.g ? aUp : aDn);
      out[i] = this.muted ? 0 : x * this.g;
    }
    if (!this.muted && (this.sh || this.q.length)) {                      // hold / replay (a delay line that drops silence)
      let e = 0; for (let i = 0; i < out.length; i++) e += out[i] * out[i];
      const loud = Math.sqrt(e / out.length) > 0.006; if (this.sh && loud) { if (!this.hLoud) this.hSince = currentTime; this.hLoud = true; }
      this.q.push({b: out.slice(0), loud});
      // the sentence the user talked over has ended (the avatar's audible voice paused >= 300 ms after being heard during the hold):
      // the user is heard now, the vendor yields to them at the sentence boundary, not mid-word
      if (this.sh && this.hLoud && this.refSeen && (this.rq >= this.LONGQ || (this.rq >= this.PAUSEQ && currentTime - this.hSince >= this.MINWAIT))) { this.sh = false; this.port.postMessage({t: 'shield_released', why: 'pause', queued: this.q.length}); }
      if (this.sh) { out.fill(0); this.st.held++; this.silRun = this.SILQ; if (this.q.length > this.MAXQ) this.q.shift(); }
      else {
        while (this.q.length > 1 && !this.q[0].loud && this.silRun >= this.SILQ) { this.q.shift(); this.st.skipped++; }
        const f = this.q.shift(); out.set(f.b); this.st.replayed++; this.silRun = f.loud ? 0 : this.silRun + 1;
      }
      if (this.q.length > this.st.maxq) this.st.maxq = this.q.length;
    }
    if (currentTime - this.lastReport > 1) { this.lastReport = currentTime; this.port.postMessage({...this.st, c: this.c}); }
    return true;
  }
}
registerProcessor('avatar-mic-gate', Gate);
`;
  const loaded = new WeakMap();
  async function ensureWorklet(ctx) {
    if (!loaded.has(ctx)) {
      const u = URL.createObjectURL(new Blob([WORKLET], {type: 'application/javascript'}));
      loaded.set(ctx, ctx.audioWorklet.addModule(u));
    }
    return loaded.get(ctx);
  }
  const now = () => performance.now();
  const SILENT = 'Silent context (not words he said) - never read aloud, quote, describe or mention it: ';
  const pct = (xs, p) => { if (!xs.length) return null; const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };


  /* aecOutput(ctx, node): play `node` through a local RTCPeerConnection pair into an <audio> element, so the
     browser's echo canceller (which references WebRTC playout, not WebAudio's ctx.destination) sees the avatar's
     voice. Returns {el, latencyMs(), stop()} or null (then the caller falls back to ctx.destination). */
  async function aecOutput(ctx, node) {
    if (typeof RTCPeerConnection === 'undefined' || !ctx.createMediaStreamDestination) return null;
    const dest = ctx.createMediaStreamDestination(); node.connect(dest);
    const a = new RTCPeerConnection(), b = new RTCPeerConnection();
    const R = {el: null, lat: 40, recv: null, stop() { try { a.close(); b.close(); } catch (e) {} try { node.disconnect(dest); } catch (e) {}
      if (R.el) { R.el.pause(); R.el.srcObject = null; } }};
    a.onicecandidate = e => { if (e.candidate) b.addIceCandidate(e.candidate).catch(() => {}); };
    b.onicecandidate = e => { if (e.candidate) a.addIceCandidate(e.candidate).catch(() => {}); };
    const got = new Promise(res => { b.ontrack = e => res(e); });
    a.addTrack(dest.stream.getAudioTracks()[0], dest.stream);
    // buzz: this hop re-encodes the (already Opus-coded) vendor voice — a second generation of Opus. At 96 kbit/s
    // VBR it added audible coding noise (measured on the page's real output: 2-6 kHz only 19.8 dB above the noise,
    // 6-11.5 kHz 11.7 dB, ~120 clicks per 9 s of speech); at 256 kbit/s it is 31.8 / 23.1 dB and click-free. A local
    // loopback costs no network, so the bitrate is the codec's ceiling.
    const hq = sdp => sdp.replace(/a=fmtp:(\d+) (.*useinbandfec=1.*)/g, (m0, pt, f) => 'a=fmtp:' + pt + ' ' +
      f.replace(/;?(maxaveragebitrate|usedtx|stereo|maxplaybackrate)=\d+/g, '') + ';maxaveragebitrate=256000;maxplaybackrate=48000;stereo=0;usedtx=0');
    const off = await a.createOffer(); await a.setLocalDescription(off); await b.setRemoteDescription({type: 'offer', sdp: hq(off.sdp)});
    const ans = await b.createAnswer(); await b.setLocalDescription(ans); await a.setRemoteDescription({type: 'answer', sdp: hq(ans.sdp)});
    const ev = await Promise.race([got, new Promise((_, rej) => setTimeout(() => rej(Error('loopback track timeout')), 5000))]);
    R.recv = ev.receiver;
    // buzz: a jitter-buffer target of 0 kept NetEq squeezing the buffer to nothing, so it time-compressed and
    // stretched the voice while it played (a 17 ms jump inside a word on the page's real output; under load it
    // conceals) = the warbly / buzzy sound. A small fixed target keeps it steady; the latency it adds is measured below
    // and taken off the DelayNode, so the voice still reaches the speaker on the lips' timeline.
    try { if ('jitterBufferTarget' in R.recv) R.recv.jitterBufferTarget = AEC_JB_MS; else if ('playoutDelayHint' in R.recv) R.recv.playoutDelayHint = AEC_JB_MS / 1000; } catch (e) {}
    const el = new Audio(); el.autoplay = true; el.playsInline = true; el.srcObject = ev.streams[0] || new MediaStream([ev.track]);
    await el.play().catch(() => {}); R.el = el;
    // the loopback's own playout latency (jitter buffer + 20 ms frame), so the caller can take it off its DelayNode
    R.timer = setInterval(async () => {
      try { const st = await b.getStats(); st.forEach(x => { if (x.type === 'inbound-rtp' && x.kind === 'audio' && x.jitterBufferEmittedCount > 0) {
        const jb = 1000 * x.jitterBufferDelay / x.jitterBufferEmittedCount; R.lat = Math.max(20, Math.min(200, jb + 20));
        // buzz: what the loopback's jitter buffer did to the voice (ms of audio concealed / time-compressed /
        // stretched; silent concealment excluded) -> stats() -> the page's report
        const sr = 48000;   // NetEq counts samples at the Opus clock
        R.neteq = {jb_ms: Math.round(jb), conceal_ms: Math.round(1000 * Math.max(0, (x.concealedSamples || 0) - (x.silentConcealedSamples || 0)) / sr),
          accel_ms: Math.round(1000 * (x.removedSamplesForAcceleration || 0) / sr), decel_ms: Math.round(1000 * (x.insertedSamplesForDeceleration || 0) / sr),
          conceal_events: x.concealmentEvents || 0}; } }); } catch (e) {}
    }, 2000);
    const stop0 = R.stop; R.stop = () => { clearInterval(R.timer); stop0(); };
    R.latencyMs = () => R.lat;
    return R;
  }

  // ---------------------------------------------------------------------------------------------- playout (tap + delay)
  /* createPlayout(ctx, srcNode, {onChunk(s0, pcm|null, silent, ct), delayMs, minMs, maxMs, marginMs})
     -> {start(), stop(), setNeed(ms) (one video-round-trip sample), delayMs(), flush(), tapStartCt, avatarVoice(), stats()}
     The delay changes only while the delay line holds silence. */
  async function createPlayout(ctx, src, o) {
    await ensureWorklet(ctx);
    const tap = new AudioWorkletNode(ctx, 'avatar-voice-tap', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
    const sink = ctx.createGain(); sink.gain.value = 0; tap.connect(sink).connect(ctx.destination);  // keeps the tap clocked
    const delay = ctx.createDelay(1.0), out = ctx.createGain();
    const P = {minMs: o.minMs ?? 60, maxMs: o.maxMs ?? 400, marginMs: o.marginMs ?? 30, D: (o.delayMs ?? 250) / 1000,
      want: null, need: [], silentRun: 0, lastVoicedCt: -1, startCt: null, levels: [], changes: 0, flushes: 0,
      chunks: 0, silentChunks: 0, onChunk: o.onChunk || (() => {})};
    P.D = Math.min(P.maxMs, Math.max(P.minMs, P.D * 1000)) / 1000;
    src.connect(tap); src.connect(delay); delay.connect(out);
    // speakers: through the echo canceller's reference path (WebRTC loopback -> <audio>), unless ?aecout=0
    P.aec = (o.aecOut === false) ? null : await aecOutput(ctx, out).catch(() => null);
    const spk = ctx.createGain();            // the speaker-mute switch for the non-loopback path
    if (!P.aec) out.connect(spk).connect(ctx.destination);
    // buzz: a whole number of samples — a fractional DelayNode time is a linear interpolation between samples, a
    // low-pass that changed with every new delay (at 0.42 of a sample: -2.9 dB at 6 kHz, -12 dB at 10 kHz)
    const nodeDelay = () => Math.round(Math.max(0, P.D - (P.aec ? P.aec.latencyMs() / 1000 : 0)) * ctx.sampleRate) / ctx.sampleRate;
    delay.delayTime.value = nodeDelay();
    tap.port.onmessage = e => {
      const m = e.data;
      if (m.t === 'started') { P.startCt = m.ct; if (o.onStart) o.onStart(m.ct); return; }
      if (m.t !== 'c') return;
      P.chunks++;
      if (m.silent) { P.silentChunks++; P.silentRun += 480; } else { P.silentRun = 0; P.lastVoicedCt = m.ct; }
      P.levels.push([P.startCt + m.s0 / TSR, m.silent ? 0 : m.rms / 32768]); if (P.levels.length > 400) P.levels.splice(0, 200);
      // change the delay only when the whole delay line is silence (inaudible), never mid-word
      // a stable delay: change it only for a real shift (>= 20 ms), at most every 4 s, and only inside a long silence
      // (>= 0.6 s and > D + 0.1 s at the tap, so the delay line holds silence) — the eval saw 24-49 changes a clip
      const nowCt = m.ct;
      if (P.want !== null && Math.abs(P.want - P.D) >= 0.02 && P.silentRun / TSR > Math.max(0.6, P.D + 0.1)
          && !(nowCt - (P.lastChangeCt ?? -1e9) < 4)) {
        P.D = P.want; delay.delayTime.value = nodeDelay(); P.changes++; P.lastChangeCt = nowCt; if (o.onDelay) o.onDelay(P.D);
      }
      if (P.aec && P.silentRun / TSR > Math.max(0.6, P.D + 0.1) && Math.abs(delay.delayTime.value - nodeDelay()) > 0.01) delay.delayTime.value = nodeDelay();
      P.onChunk(m.s0, m.silent ? null : m.pcm, m.silent, m.ct, m.first, m.rms);
    };
    const api = {
      tap, delay, out, get aec() { return P.aec; },
      start() { tap.port.postMessage({t: 'start'}); },
      stop() { tap.port.postMessage({t: 'stop'}); if (P.aec) { P.aec.stop(); P.aec = null; } },
      get startCt() { return P.startCt; },
      delayMs: () => P.D * 1000,
      /* speaker mute: nothing of the avatar's voice reaches the speakers (the tap, the lips, the recording and the transcript are untouched) */
      setMuted(on) { P.spkMuted = !!on; spk.gain.value = on ? 0 : 1; if (P.aec && P.aec.el) P.aec.el.muted = !!on; },
      get muted() { return !!P.spkMuted; },
      /* one sample of "how much delay this frame needed" (ms): frame decoded time - when its audio would be heard undelayed */
      setNeed(ms) {
        P.need.push(ms); if (P.need.length > 240) P.need.splice(0, 120);
        if (P.need.length >= 12) {
          const p90 = pct(P.need.slice(-120), 0.9);
          P.want = Math.min(P.maxMs, Math.max(P.minMs, p90 + P.marginMs)) / 1000;
        }
      },
      /* interrupted: the server cut the reply; drop what is still inside the delay line */
      flush() {
        const t = ctx.currentTime; P.flushes++;
        out.gain.cancelScheduledValues(t); out.gain.setValueAtTime(out.gain.value, t);
        out.gain.linearRampToValueAtTime(0, t + 0.015); out.gain.setValueAtTime(0, t + P.D + 0.02);
        out.gain.linearRampToValueAtTime(1, t + P.D + 0.04);
      },
      /* the avatar's level at the NEWEST tapped audio = the timeline the rig vector is stamped with (motion sync) */
      avatarVoiceTap() {
        const lv = P.levels.length ? P.levels[P.levels.length - 1][1] : 0;
        return {rms: lv, voiced: lv > 0.008, pitch: null};
      },
      /* the avatar's level as HEARD now (delayed) */
      avatarVoice() {
        const at = ctx.currentTime - P.D; let lv = 0;
        for (let i = P.levels.length - 1; i >= 0; i--) { if (P.levels[i][0] <= at) { lv = P.levels[i][1]; break; } }
        return {rms: lv, voiced: lv > 0.008, pitch: null};
      },
      stats: () => ({aec_out: !!P.aec, aec_loop_ms: P.aec ? Math.round(P.aec.latencyMs()) : null, aec_neteq: P.aec && P.aec.neteq ? P.aec.neteq : null, node_delay_ms: Math.round(delay.delayTime.value * 1000), delay_ms: Math.round(P.D * 1000), want_ms: P.want === null ? null : Math.round(P.want * 1000),
        need_p50: pct(P.need.slice(-120), .5), need_p90: pct(P.need.slice(-120), .9), need_n: P.need.length,
        delay_changes: P.changes, flushes: P.flushes, chunks: P.chunks, silent_chunks: P.silentChunks}),
    };
    return api;
  }

  // ---------------------------------------------------------------------------------------------- the voice session
  function createVoice(o) {
    const mode = o.mode || 'realtime';
    // 'loopback' = the mock vendor's WAVs sent over a local RTCPeerConnection pair, so the voice arrives as a REAL
    // remote WebRTC track (same receive path as realtime/live, no vendor, no cost) — for sync tests
    const MOCK = mode === 'mock' || mode === 'loopback';
    const GEM = /^gemini/.test(mode);   // Gemini 3.8 Live over a WebSocket (server-minted one-use token)
    const ctx = o.ctx;
    const bus = ctx.createGain();            // the avatar's voice enters here (remote track or mock WAVs)
    const V = {
      state: 'idle', pc: null, dc: null, audioEl: null, remote: null, ready: false, startedAt: 0,
      micSpeaking: false, micVoicedMs: 0, lastMicVoiceCt: -1, userEnds: [], turnOpen: null, turns: [], calls: [],
      pendingCalls: new Map(), itemToCall: new Map(), respHasAudio: new Map(), respCalls: new Map(), events: {}, evlog: [],
      lastStill: -1e9, lastEventResp: -1e9, eventResps: [], usage: [], usd: 0, serverSpeaking: false,
      avatarSpeaking: false, lastAvatarVoicePerf: -1e9, transcriptYou: '', mockIdx: 0, mockBufs: [], mockSrc: null,
      liveSeconds: 0, meter: null, userLvl: 0, startupMs: null, createMs: null,
      // response bookkeeping for prompt() (realtime: never response.create while one is active / being created)
      respActive: new Map(), respWantedAt: 0, lastRespCreatedAt: -1e9, lastUserSpeechAt: -1e9, pq: [], pqTimer: 0, promptN: 0,
      pCreating: null, noMeta: false, prompts: [],
      // shield (see "shield" below)
      sh: {on: false, rid: null, timer: 0, n: 0, held: 0, other: 0}, unprompted: new Set(), evRespAt: 0,
      micMuted: false, spkMuted: false,
      // quiet: the harness speech gate (app/quiet.js owns the policy; see "quiet" below)
      q: {on: false, wake: false, muted: false, graceT: 0, calmUntil: 0, bypass: 0, cancelled: new Set(), segs: new Map(), respAt: new Map(), since: -1e12, heard: '', heardT: 0, dropTurn: false,
        turnAsked: false, tdPending: false, seg: null, n: {blocked: 0, cancelled: 0, audioDropped: 0, interrupts: 0, notesOnly: 0,
        turns: 0, woke: 0, capped: 0, stale: 0}},
      talk: {level: 'normal', gapMs: 10000, perMin: 4, off: false, gemWords: null, suffix: '', maxTokens: null}, sess: null, baseInstr: null,
    };
    const emitState = (s, d) => { V.state = s; if (o.onState) o.onState(s, d || ''); if (s === 'live') quietReady(); };   // pending level / quiet go out
    const metric = m => { if (o.onMetric) o.onMetric(m); };
    // the event log keeps every state change of a session (counts of the high-rate delta events stay in V.events only),
    // each with the ids that tie a cut audio buffer to the response / speech / prompt that caused it
    const NOISY = /\.delta$|rate_limits\.updated$|conversation\.item\.(added|done)$|content_part\.|output_item\.(added|done)$/;
    const log = (type, extra) => { V.events[type] = (V.events[type] || 0) + 1; if (V.evlog.length < 4000 && !NOISY.test(type)) V.evlog.push({t: Math.round(now()), type, ...(extra || {})}); };
    const evExtra = ev => {
      const t = ev.type, r = ev.response || {};
      if (t === 'response.created') return {rid: r.id, md: r.metadata && r.metadata.avatar_prompt || undefined};
      if (t === 'response.done') return {rid: r.id, status: r.status, reason: r.status_details && (r.status_details.reason || r.status_details.type) || undefined};
      if (/^output_audio_buffer\./.test(t)) return {rid: ev.response_id};
      if (t === 'conversation.item.truncated') return {item: ev.item_id, audio_end_ms: ev.audio_end_ms};
      if (t === 'input_audio_buffer.speech_started') return {audio_start_ms: ev.audio_start_ms, avatar_audible: !!(o.audible && o.audible())};
      if (t === 'input_audio_buffer.speech_stopped') return {audio_end_ms: ev.audio_end_ms};
      if (t === 'response.output_audio_transcript.done') return {rid: ev.response_id, text: String(ev.transcript || '').slice(0, 200)};
      if (t === 'response.function_call_arguments.done') return {rid: ev.response_id, name: ev.name};
      if (t === 'conversation.item.input_audio_transcription.completed') return {text: String(ev.transcript || '').slice(0, 120)};
      return undefined;
    };
    const ctToPerf = ct => now() - (ctx.currentTime - ct) * 1000;      // context time -> perf time (same instant)

    // ---- the user's mic level (local end-of-speech; the mock vendor's turn-taking)
    async function startMeter() {
      if (!o.micStream) return;
      await ensureWorklet(ctx);
      const src = ctx.createMediaStreamSource(o.micStream);
      const m = new AudioWorkletNode(ctx, 'avatar-mic-meter', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
      const sink = ctx.createGain(); sink.gain.value = 0; src.connect(m).connect(sink).connect(ctx.destination);
      V.meter = {src, m, sink};
      const TH = o.micThreshold || 0.012, END_MS = MOCK ? (o.mock && o.mock.endMs) || 600 : 450;
      let voicedRun = 0, silentRun = 0;
      m.port.onmessage = e => {
        const {rms, ct} = e.data; V.userLvl = rms;
        // the avatar is audible (delayed playout, via the page) -> not the user's speech for our metrics (AEC is not guaranteed)
        const echo = o.audible ? o.audible() : (V.avatarSpeaking && now() - V.lastAvatarVoicePerf < 250);
        if (rms > TH && !echo) {
          voicedRun += 10; silentRun = 0; V.lastMicVoiceCt = ct;
          const sg = V.uSeg || (V.uSeg = {sum: 0, n: 0, pk: 0, echo: 0}); sg.sum += rms; sg.n++; if (rms > sg.pk) sg.pk = rms;
          if (!V.micSpeaking && voicedRun >= 120) { V.micSpeaking = true; V.micStartCt = ct - voicedRun / 1000; onUserStart(); }
        } else {
          if (rms > TH && echo) { const sg = V.uSeg || (V.uSeg = {sum: 0, n: 0, pk: 0, echo: 0}); sg.echo++; }      // the user's mic hears the avatar
          silentRun += 10; if (silentRun > 60) voicedRun = 0;
          if (V.micSpeaking && silentRun >= END_MS) { V.micSpeaking = false; onUserEnd(V.lastMicVoiceCt); }
          if (!V.micSpeaking && silentRun >= END_MS && V.uSeg) { V.lastSeg = segOf(V.uSeg); V.uSeg = null; }
        }
      };
    }
    function onUserStart() {
      V.lastUserSpeechAt = now();
      if (o.onActivity) o.onActivity('speech');
      if (MOCK && V.mockSrc) { try { V.mockSrc.stop(); } catch (e) {} V.mockSrc = null; if (o.onInterrupt) o.onInterrupt(); }
    }
    function onUserEnd(ct) {
      const perf = ctToPerf(ct);
      V.userEnds.push({ct, perf}); if (V.userEnds.length > 200) V.userEnds.splice(0, 100);
      V.turnOpen = {user_end_ct: ct, user_end_ms: Math.round(perf), text: '', events: {}};
      if (V._stopPending && now() - V._stopPending < 1500) V.turnOpen.events.speech_stopped = Math.round(V._stopPending - perf);
      V._stopPending = 0;
      log('user_end');
      if (V.uSeg) { V.lastSeg = segOf(V.uSeg); V.uSeg = null; }
      if (MOCK) { if (V.q.on) { if (quietTurn({text: '', durMs: V.lastSeg ? V.lastSeg.durMs : null}) !== 'wake') return; } mockReply(); }
    }
    // the playout tap tells us when the avatar's voice starts (first voiced 24 kHz sample, playback timeline)
    function onAvatarChunk(s0, silent, ct, first, startCt, delayMs, outLatMs) {
      if (silent) { if (V.avatarSpeaking && now() - V.lastAvatarVoicePerf > 400) V.avatarSpeaking = false; return; }
      V.avatarSpeaking = true; V.lastAvatarVoicePerf = now();
      const T = V.turnOpen;
      if (T && T.first_tap_ct === undefined) {
        const tapCt = startCt + (s0 + Math.max(0, first)) / TSR;
        if (tapCt > T.user_end_ct) {
          T.first_tap_ct = tapCt;
          T.tap_lat_ms = Math.round((tapCt - T.user_end_ct) * 1000);
          T.delay_ms = Math.round(delayMs); T.out_latency_ms = Math.round(outLatMs);
          T.audible_lat_ms = T.tap_lat_ms + T.delay_ms + T.out_latency_ms;
          T.first_tap_ms = Math.round(ctToPerf(tapCt)); T.first_audible_ms = T.first_tap_ms + T.delay_ms + T.out_latency_ms;
          T.text = V.transcriptYou.slice(-160); V.transcriptYou = '';
          V.turns.push(T); V.turnOpen = null;
          metric({type: 'turn', ...T});
        }
      }
    }

    // ---- data channel
    function send(ev) {
      if (V.q.on && !V.q.bypass) { ev = quietFilter(ev); if (!ev) return false; }
      if (/^(response\.create|response\.cancel|output_audio_buffer\.clear|conversation\.item\.truncate)$/.test(ev.type))   // what we ask the vendor to do
        log('tx:' + ev.type, {rid: ev.response_id, md: ev.response && ev.response.metadata && ev.response.metadata.avatar_prompt || undefined});
      if (o.transport) { o.transport.send(ev); return true; }   // test seam
      if (GEM) { if (V.ws && V.ws.readyState === 1) { V.ws.send(JSON.stringify(ev)); return true; } return false; }
      if (V.dc && V.dc.readyState === 'open') { V.dc.send(JSON.stringify(ev)); return true; } return false; }
    function lastUserEnd() { return V.userEnds.length ? V.userEnds[V.userEnds.length - 1] : null; }
    function callStart(name, callId, extra) {
      if (V.pendingCalls.has(callId)) return;
      const ue = lastUserEnd(), t = now();
      const c = {name, call_id: callId, start_ms: Math.round(t), start_lat_ms: ue && t - ue.perf < 20000 ? Math.round(t - ue.perf) : null, ...extra};
      V.pendingCalls.set(callId, c);
      if (o.onCall) o.onCall(name, null, 'start', c);
    }
    /* act before the arguments finish: once the call's string fields are complete (a later key has started, or the
       object closed), the page can move; 'done' then only confirms (the page dedupes identical commands) */
    function partialArgs(s) {
      s = String(s || '').trim(); if (!s.startsWith('{')) return null;
      for (const c of [s, s + '}', s.slice(0, s.lastIndexOf(',')) + '}']) {
        try { const v = JSON.parse(c); if (v && typeof v === 'object') return v; } catch (e) {}
      }
      return null;
    }
    // the complete object value of "key" in a streamed JSON argument buffer (string-aware brace match), or null
    function closedObj(buf, key) {
      const m = new RegExp('"' + key + '"\\s*:\\s*\\{').exec(buf); if (!m) return null;
      let i = m.index + m[0].length - 1, depth = 0, str = false, esc = false;
      for (let j = i; j < buf.length; j++) {
        const ch = buf[j];
        if (str) { if (esc) esc = false; else if (ch === '\\') esc = true; else if (ch === '"') str = false; continue; }
        if (ch === '"') str = true; else if (ch === '{') depth++;
        else if (ch === '}' && --depth === 0) { try { return JSON.parse(buf.slice(i, j + 1)); } catch (e) { return null; } }
      }
      return null;
    }
    function callDelta(callId, delta) {
      const c = V.pendingCalls.get(callId); if (!c || c.early_ms) return;
      c.argBuf = (c.argBuf || '') + (delta || '');
      if (c.name === 'do_action' || c.name === 'perform_action') {   // a named move plays before the description ends
        const mv = closedObj(c.argBuf, 'move');
        if (!mv || !mv.name) return;
        const t = now(), ue = lastUserEnd();
        c.early_ms = Math.round(t); c.early_lat_ms = ue && t - ue.perf < 20000 ? Math.round(t - ue.perf) : null; c.early_args = {move: mv};
        if (o.onCall) { try { o.onCall(c.name, {move: mv}, 'early', c); } catch (e) {} }
        return;
      }
      if (!/,/.test(c.argBuf)) return;                    // only trust fields that a comma has closed
      const a = partialArgs(c.argBuf.slice(0, c.argBuf.lastIndexOf(',')) + '}');
      const key = {gesture: 'name', look: 'target', expression: 'name'}[c.name];
      if (!a || !key || typeof a[key] !== 'string') return;
      if (c.name === 'gesture' && !/"side"|"repeat"|"hold"/.test(c.argBuf.slice(c.argBuf.indexOf('"name"') + 6)) ) return;
      const t = now(), ue = lastUserEnd();
      c.early_ms = Math.round(t); c.early_lat_ms = ue && t - ue.perf < 20000 ? Math.round(t - ue.perf) : null; c.early_args = a;
      if (o.onCall) { try { o.onCall(c.name, a, 'early', c); } catch (e) {} }
    }
    function callDone(name, callId, argStr, extra) {
      let args = {}; try { args = JSON.parse(argStr || '{}') || {}; } catch (e) { args = {_bad: String(argStr).slice(0, 200)}; }
      if (!V.pendingCalls.has(callId)) callStart(name, callId, extra);
      const c = V.pendingCalls.get(callId); V.pendingCalls.delete(callId);
      const t = now(), ue = lastUserEnd();
      c.args = args; c.done_ms = Math.round(t); c.done_lat_ms = ue && t - ue.perf < 20000 ? Math.round(t - ue.perf) : null;
      V.calls.push(c); if (V.calls.length > 300) V.calls.splice(0, 100);
      let out = {ok: true};
      const qo = quietTool(name, args, c);
      if (qo) out = qo;
      else if (o.onCall) { try { const r = o.onCall(name, args, 'done', c); if (r && typeof r === 'object') out = r; } catch (e) { out = {ok: false, error: String(e.message || e)}; } }
      delete c.argBuf;
      metric({type: 'call', name, args, start_lat_ms: c.start_lat_ms, early_lat_ms: c.early_lat_ms, done_lat_ms: c.done_lat_ms, mode});
      api.sendToolResult(callId, out);
    }

    function onRealtime(ev) {
      const t = ev.type; log(t, evExtra(ev));
      switch (t) {
        case 'session.updated': rtSession(ev.session, false); break;
        case 'session.created':
          rtSession(ev.session, true);
          if (V.audioBase == null) V.audioBase = now();   // the server's audio clock starts about now (refined below)
          if (!V.ready) { V.ready = true; V.startupMs = Math.round(now() - V.startedAt); seedHistory(); emitState('live'); }
          armRenew(ev.session && ev.session.expires_at);
          break;
        case 'input_audio_buffer.speech_started': {
          V.serverSpeaking = true; V.lastUserSpeechAt = now(); if (V.turnOpen) V.turnOpen.events.speech_started = Math.round(now());
          V.q.seg = {start: now(), audio_start_ms: ev.audio_start_ms, echo: !!(o.audible && o.audible())};
          if (ev.item_id) { V.q.segs.set(ev.item_id, V.q.seg); if (V.q.segs.size > 40) V.q.segs.delete(V.q.segs.keys().next().value); }
          // when the user's speech really began, on this page's clock. The event can arrive seconds late (seen: 11 s); audio_start_ms is on the server's audio clock, whose start is the smallest (arrival - audio ms) seen.
          let startedAt;
          if (Number.isFinite(ev.audio_start_ms)) {
            const off = now() - ev.audio_start_ms; V.audioBase = V.audioBase == null ? off : Math.min(V.audioBase, off);
            startedAt = Math.min(now(), V.audioBase + ev.audio_start_ms);
          }
          if (o.onActivity) o.onActivity('speech', {startedAt});
          shieldSpeech();
          if (o.onUserSpeech) { try { o.onUserSpeech('start', ev); } catch (e) {} }    // speech: note flush, barge-in
          maybeStill('speech');
          break; }
        case 'input_audio_buffer.speech_stopped':
          V.serverSpeaking = false;
          if (V.q.seg) { V.q.seg.durMs = Number.isFinite(ev.audio_end_ms) && Number.isFinite(V.q.seg.audio_start_ms) ? ev.audio_end_ms - V.q.seg.audio_start_ms : now() - V.q.seg.start; }
          if (Number.isFinite(ev.audio_end_ms)) { const off = now() - ev.audio_end_ms; V.audioBase = V.audioBase == null ? off : Math.min(V.audioBase, off); }
          if (o.onUserSpeech) { try { o.onUserSpeech('stop', ev); } catch (e) {} }     // speech
          V._lastStop = {perf: now(), audio_end_ms: ev.audio_end_ms}; pqKick();
          if (V.turnOpen) V.turnOpen.events.speech_stopped = Math.round(now() - V.turnOpen.user_end_ms);
          else V._stopPending = now();
          break;
        case 'conversation.item.input_audio_transcription.delta':
          V.transcriptYou += ev.delta || ''; if (o.onTranscript) o.onTranscript('you', ev.delta || '', false); break;
        case 'conversation.item.input_audio_transcription.completed':
          if (o.onTranscript) o.onTranscript('you', ev.transcript || '', true);
          // while quiet nothing answers the user (create_response off); the controller decides whether this wakes it
          // (an utterance that began before the quiet period — the very request for it, transcribed late — never wakes it)
          if (V.q.on) { const sg = (ev.item_id && V.q.segs.get(ev.item_id)) || V.q.seg || null;
            if (sg ? sg.start < V.q.since : now() - V.q.since < 2500) log('quiet_turn', {r: 'before quiet'});
            else if (quietTurn({text: ev.transcript || '', durMs: sg ? sg.durMs : null, echo: sg ? sg.echo : false}) === 'wake') respondNow(); }
          break;
        case 'response.created': {
          promptCreated(ev.response || {});
          if (ev.response && ev.response.id) { V.q.respAt.set(ev.response.id, now()); if (V.q.respAt.size > 30) V.q.respAt.delete(V.q.respAt.keys().next().value); }
          if (V.q.on && !V.q.allowResp && ev.response && ev.response.id) quietCancel(ev.response.id);                     // a leak
          if (V.evRespAt && ev.response && ev.response.id && now() - V.evRespAt < 3000) { V.unprompted.add(ev.response.id); V.evRespAt = 0; if (V.unprompted.size > 60) V.unprompted.delete(V.unprompted.values().next().value); }
          if (V.sh.on && ev.response && ev.response.id) shieldArm(ev.response.id);   // a chained reply
          V.q.allowResp = false;
          if (o.onResponse) { try { o.onResponse('created', {response_id: ev.response && ev.response.id}); } catch (e) {} }   // speech
          const ue = lastUserEnd(); V.respHasAudio.set(ev.response && ev.response.id, false);
          if (V.turnOpen && V.turnOpen.events.response_created === undefined && ue) V.turnOpen.events.response_created = Math.round(now() - ue.perf);
          break; }
        case 'output_audio_buffer.started': {
          V.outPlaying = now();
          // a reply whose audio only starts > 5 s after it was created, when the user has spoken since = stale (it sat
          // behind other audio; the user has moved on): dropped, never played a turn late
          { const at = V.respActive.get(ev.response_id) || V.q.respAt.get(ev.response_id);
            if (at && now() - at > STALE_MS && V.lastUserSpeechAt > at + 500) { V.q.n.stale++; log('stale_reply', {id: ev.response_id, late_ms: Math.round(now() - at)}); quietCancel(ev.response_id); break; } }
          const ue = lastUserEnd(); if (V.turnOpen && ue && V.turnOpen.events.audio_started === undefined) V.turnOpen.events.audio_started = Math.round(now() - ue.perf);
          if (ev.response_id) V.respHasAudio.set(ev.response_id, true);
          promptAudio(ev.response_id);
          shieldOn(ev.response_id);
          break; }
        case 'output_audio_buffer.cleared': V.outPlaying = 0; shieldOff('cleared'); if (o.onInterrupt) o.onInterrupt(); if (o.onResponse) { try { o.onResponse('cleared', {response_id: ev.response_id}); } catch (e) {} } break;
        // speech: the 4th argument carries the response / item ids (co-speech gestures follow each reply)
        case 'response.output_audio_transcript.delta': if (V.q.cancelled.has(ev.response_id) || V.q.muted) break; if (o.onTranscript) o.onTranscript('avatar', ev.delta || '', false, {response_id: ev.response_id, item_id: ev.item_id}); break;
        case 'response.output_audio_transcript.done': if (V.q.cancelled.has(ev.response_id) || V.q.muted) break; if (o.onTranscript) o.onTranscript('avatar', ev.transcript || '', true, {response_id: ev.response_id, item_id: ev.item_id}); break;
        case 'output_audio_buffer.stopped': V.outPlaying = 0; shieldOff('stopped'); if (V.q.on && V.q.graceT && V.q.cut) V.q.cut(); if (o.onResponse) { try { o.onResponse('audio_stopped', {response_id: ev.response_id}); } catch (e) {} } break;
        case 'conversation.item.truncated': if (o.onResponse) { try { o.onResponse('truncated', {item_id: ev.item_id, audio_end_ms: ev.audio_end_ms}); } catch (e) {} } break;
        case 'response.output_item.added': {
          const it = ev.item || {};
          if (it.type === 'function_call') { V.itemToCall.set(it.id, it.call_id); callStart(it.name, it.call_id, {response_id: ev.response_id}); }
          else if (it.type === 'message') V.respHasAudio.set(ev.response_id, true);
          break; }
        case 'response.function_call_arguments.delta': if (o.early !== false) callDelta(ev.call_id || V.itemToCall.get(ev.item_id), ev.delta); break;
        case 'response.function_call_arguments.done': callDone(ev.name, ev.call_id, ev.arguments, {response_id: ev.response_id}); break;
        case 'response.done': {
          const r = ev.response || {}, outs = r.output || [];
          const calls = outs.filter(x => x.type === 'function_call');
          for (const c of calls) if (V.pendingCalls.has(c.call_id) || !V.calls.some(x => x.call_id === c.call_id)) callDone(c.name, c.call_id, c.arguments, {response_id: r.id});
          const spoke = outs.some(x => x.type === 'message') || V.respHasAudio.get(r.id);
          // a tool-only response: let the character say its few words once (never a second time if it already spoke)
          V.respActive.delete(r.id);                                                  // (before the follow-up: it re-arms)
          if (calls.length && !spoke && r.status === 'completed') { wantResponse({type: 'response.create'}); log('tool_followup'); }
          if (V.turnOpen && V.turnOpen.events.first_response_done === undefined) V.turnOpen.events.first_response_done = Math.round(now() - V.turnOpen.user_end_ms);
          V.respHasAudio.delete(r.id);
          if (o.onResponse) { try { o.onResponse('done', {response_id: r.id, status: r.status, reason: r.status_details && r.status_details.reason}); } catch (e) {} }   // speech
          promptDone(r.id, r.status);
          if (r.usage) { V.usage.push(r.usage); metric({type: 'usage', mode: 'realtime', usage: r.usage}); }
          break; }
        case 'error': {
          const code = String((ev.error && ev.error.code) || ''), msg = String((ev.error && ev.error.message) || '');
          log('error_detail', {code, message: msg.slice(0, 160)});
          promptError(ev.error || {});
          if (/session_expired|expired|maximum (session )?duration/i.test(code + ' ' + msg)) drop('session expired');
          break; }
      }
    }
    function onLive(ev) {
      const t = ev.type; log(t);
      switch (t) {
        case 'session.started':
          if (!V.ready) { V.ready = true; V.startupMs = Math.round(now() - V.startedAt); emitState('live'); }
          armRenew(ev.session && ev.session.expires_at);
          break;
        case 'session.input_transcript.delta':
          V.transcriptYou += ev.delta || ''; if (o.onTranscript) o.onTranscript('you', ev.delta || '', false); V.lastUserSpeechAt = now(); quietHeard(ev.delta);
          if (o.onUserSpeech && !V._liveUserTalking) { V._liveUserTalking = true; try { o.onUserSpeech('start', ev); } catch (e) {} }   // speech
          clearTimeout(V._liveUserT); V._liveUserT = setTimeout(() => { V._liveUserTalking = false; }, 1200);
          if (o.onActivity) o.onActivity('speech'); break;
        case 'session.output_transcript.delta': if (o.onTranscript) o.onTranscript('avatar', ev.delta || '', false, {response_id: ev.response_id || ev.item_id || null, item_id: ev.item_id || null}); break;
        case 'session.usage.updated': {
          const s = +((ev.usage || {}).seconds || 0); if (s > V.liveSeconds) { V.liveSeconds = s; metric({type: 'usage', mode: 'live', seconds: s}); }
          break; }
        case 'response.event': {
          const n = ev.event || {}, it = n.item || {};
          log('re:' + n.type);
          if (n.type === 'response.output_item.added' && it.type === 'function_call') { V.itemToCall.set(it.id, it.call_id); callStart(it.name, it.call_id, {delegation_id: ev.delegation_id}); }
          else if (n.type === 'response.function_call_arguments.delta' && o.early !== false) callDelta(n.call_id || V.itemToCall.get(n.item_id), n.delta);
          else if (n.type === 'response.output_item.done' && it.type === 'function_call') {
            callDone(it.name, it.call_id, it.arguments, {delegation_id: ev.delegation_id});
            const rid = n.response_id || (n.response && n.response.id) || ev.delegation_id || '';
            V.respCalls.set(rid, (V.respCalls.get(rid) || 0) + 1); V._lastRid = rid;
          } else if (['response.completed', 'response.done', 'response.incomplete', 'response.failed'].includes(n.type)) {
            const rid = (n.response && n.response.id) || ev.delegation_id || '';
            const had = V.respCalls.get(rid) || (V._lastRid && V.respCalls.get(V._lastRid));
            if (had) { V.respCalls.delete(rid); V.respCalls.delete(V._lastRid); send({type: 'response.create'}); }   // results in: continue
            const u = n.response && n.response.usage; if (u) metric({type: 'usage', mode: 'live', usage: u});
          }
          break; }
        case 'session.closed': if (!V.stopping) drop('vendor closed'); emitState('ended', 'vendor closed'); break;
        case 'error': {
          const code = String((ev.error && ev.error.code) || ''), msg = String((ev.error && ev.error.message) || '');
          log('error_detail', {code, message: msg.slice(0, 160)});
          if (/session_expired|expired|maximum (session )?duration/i.test(code + ' ' + msg)) drop('session expired');
          break; }
      }
    }

    // ---- camera stills: one small JPEG when the user starts talking (<= 1 per 4 s, never its own response)
    function maybeStill(why) {
      if (mode !== 'realtime' || !o.grabStill || now() - V.lastStill < 4000) return;
      const img = o.grabStill(); if (!img) return;
      V.lastStill = now();
      send({type: 'conversation.item.create', item: {type: 'message', role: 'user', content: [{type: 'input_image', image_url: img, detail: 'low'}]}});
      log('still_sent', {why, kb: Math.round(img.length / 1024)});
    }

    // ---- mock vendor: scripted reply WAVs through the same graph, after a mic-energy end of speech
    async function loadMock() {
      const urls = (o.mock && o.mock.urls) || [];
      V.mockBufs = [];
      for (const u of urls) {
        try { const r = await fetch(u, {cache: 'no-store'}); V.mockBufs.push({url: u, buf: await ctx.decodeAudioData(await r.arrayBuffer())}); }
        catch (e) { log('mock_load_failed', {u: String(u).slice(-40)}); }
      }
    }
    function mockReply() {
      if (!V.mockBufs.length) return;
      const item = V.mockBufs[V.mockIdx++ % V.mockBufs.length];
      const lat = (o.mock && o.mock.latMs != null) ? o.mock.latMs : 300;
      const m = /call_([a-z_]+)/.exec(item.url);
      setTimeout(() => {
        if (V.state !== 'live') return;
        if (V.q.on) { V.q.n.blocked++; log('quiet_blocked', {what: 'mock reply'}); return; }
        if (m) { const id = 'mock_' + Math.random().toString(36).slice(2, 8); callStart('gesture', id); callDone('gesture', id, JSON.stringify({name: m[1]})); }
        const s = ctx.createBufferSource(); s.buffer = item.buf; s.connect(V.mockOut || bus); s.start();
        V.mockSrc = s; s.onended = () => { if (V.mockSrc === s) V.mockSrc = null; };
      }, lat);
    }

    function attachRemote(ms) {
      // Chrome quirk: a remote WebRTC stream only flows into WebAudio while a media element holds it
      const el = new Audio(); el.muted = true; el.autoplay = true; el.playsInline = true; el.srcObject = ms;
      el.play().catch(() => {}); V.audioEl = el;
      V.remote = ctx.createMediaStreamSource(ms); V.remote.connect(bus);
      log('track');
    }
    async function startLoopback() {
      V.mockOut = ctx.createMediaStreamDestination();
      const a = new RTCPeerConnection(), b = new RTCPeerConnection(); V.loop = [a, b];
      a.onicecandidate = e => { if (e.candidate) b.addIceCandidate(e.candidate).catch(() => {}); };
      b.onicecandidate = e => { if (e.candidate) a.addIceCandidate(e.candidate).catch(() => {}); };
      b.ontrack = e => { if (e.track.kind === 'audio') attachRemote(e.streams[0] || new MediaStream([e.track])); };
      watchPc(b);
      a.addTrack(V.mockOut.stream.getAudioTracks()[0], V.mockOut.stream);
      const off = await a.createOffer(); await a.setLocalDescription(off); await b.setRemoteDescription(off);
      const ans = await b.createAnswer(); await b.setLocalDescription(ans); await a.setRemoteDescription(ans);
    }

    // ---- the mic the VENDOR hears: the echo-cancelled mic through the half-duplex / barge-in gate (?gate=0 = raw mic)
    async function micForVendor() {
      if (o.gate === false || !o.micStream || MOCK) return o.micStream;
      try {
        await ensureWorklet(ctx);
        const src = ctx.createMediaStreamSource(o.micStream);
        const g = new AudioWorkletNode(ctx, 'avatar-mic-gate', {numberOfInputs: 2, numberOfOutputs: 1, outputChannelCount: [1]});
        src.connect(g, 0, 0);
        if (o.gateRef) o.gateRef.connect(g, 0, 1);
        const dest = ctx.createMediaStreamDestination(); g.connect(dest);
        g.port.onmessage = e => { const d = e.data; if (d && d.t === 'shield_released') { shieldOff('pause', true); return; } V.gate = d; };
        if (o.hardDuplex) g.port.postMessage({hard: true});
        V.gateNodes = {src, g, dest};
        if (V.micMuted) g.port.postMessage({mute: true}); if (V.spkMuted) g.port.postMessage({refOff: true});   // muted before the gate existed
        try { const t = o.micStream.getAudioTracks()[0]; V.micSettings = t && t.getSettings ? t.getSettings() : null; } catch (e) {}
        return dest.stream;
      } catch (e) { log('gate_failed', {why: String(e && e.message || e).slice(0, 80)}); return o.micStream; }
    }

    /* rec: a new microphone mid-session — every consumer of the old stream is re-fed from the new one, the
       session itself (WebRTC / WebSocket, its state, the gate's state) stays: the level meter, the gate's input (whose
       output stream is what the vendor hears), and without a gate the vendor's own sender (replaceTrack) or Gemini's tap */
    function setMic(stream) {
      if (!stream || stream === o.micStream) return false;
      const old = o.micStream; o.micStream = stream;
      const resrc = (holder, connect) => {
        if (!holder) return;
        try { holder.src.disconnect(); } catch (e) {}
        holder.src = ctx.createMediaStreamSource(stream); connect(holder.src);
      };
      resrc(V.meter, s => s.connect(V.meter.m));
      if (V.gateNodes) {
        resrc(V.gateNodes, s => s.connect(V.gateNodes.g, 0, 0));
        try { const t = stream.getAudioTracks()[0]; V.micSettings = t && t.getSettings ? t.getSettings() : null; } catch (e) {}
      } else {
        const tr = stream.getAudioTracks()[0];
        if (V.pc && tr) for (const sd of V.pc.getSenders()) { if (sd.track && sd.track.kind === 'audio') sd.replaceTrack(tr).catch(() => {}); }
        if (V.gemTap) resrc(V.gemTap, s => s.connect(V.gemTap.tap));
      }
      log('mic_switch', {had: !!old});
      return true;
    }

    // ---- lifecycle
    async function start() {
      V.startedAt = now(); emitState('connecting');
      await startMeter();
      if (mode === 'loopback') await startLoopback();
      // test seam: a scripted transport instead of WebRTC / WebSocket (node tests drive api.feed(ev))
      if (o.transport) { V.ready = true; V.startupMs = 0; emitState('live'); return; }
      if (GEM) return startGemini();
      if (MOCK) { await loadMock(); V.ready = true; V.startupMs = Math.round(now() - V.startedAt); if (o.renewMs) armRenew(null); emitState('live'); return; }
      const pc = new RTCPeerConnection(); V.pc = pc;
      pc.ontrack = e => { if (e.track.kind === 'audio') attachRemote(e.streams[0] || new MediaStream([e.track])); };
      watchPc(pc);
      const sendStream = await micForVendor();
      for (const tr of sendStream.getAudioTracks()) pc.addTrack(tr, sendStream);
      const dc = pc.createDataChannel('oai-events'); V.dc = dc;
      dc.onmessage = e => { let ev; try { ev = JSON.parse(e.data); } catch (x) { return; } (mode === 'live' ? onLive : onRealtime)(ev); };
      dc.onopen = () => log('dc_open');
      dc.onclose = () => { log('dc_close'); if (!V.stopping && V.ready) drop('data channel closed'); };
      const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
      const t0 = now();
      const r = await fetch(o.rtcUrl, {method: 'POST', headers: {...(o.rtcHeaders || {}), 'content-type': 'application/json'},
        body: JSON.stringify({sdp: pc.localDescription.sdp, mode, character: o.character, vad: o.vad, effort: o.effort || undefined, style: o.style || undefined,
          planner: o.planner === false ? false : undefined,   // ?planner=0 = the planner-off session
          history: mode === 'live' && o.history && o.history.length ? o.history.slice(-20) : undefined}), cache: 'no-store'});
      let j = null; try { j = await r.json(); } catch (e) {}
      if (!r.ok || !j || !j.sdp) { const msg = (j && j.error) || ('HTTP ' + r.status); stop(); emitState('error', msg); throw new Error(msg); }
      V.createMs = Math.round(now() - t0); V.model = j.model;
      await pc.setRemoteDescription({type: 'answer', sdp: j.sdp});
      // realtime: session.created flips us live; live: session.started (the HTTP request already started it)
      setTimeout(() => { if (!V.ready && V.state === 'connecting') { V.ready = true; emitState('live', 'no session event'); } }, 6000);
    }
    // ---- Gemini 3.8 Live (direct WebSocket; same gate, playout bus, tools and notes as gpt-realtime)
    const b64 = (buf) => { const u = new Uint8Array(buf.buffer || buf); let t = ''; for (let i = 0; i < u.length; i += 0x8000) t += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(t); };
    async function startGemini() {
      const t0 = now();
      const r = await fetch(o.geminiUrl, {method: 'POST', headers: {...(o.rtcHeaders || {}), 'content-type': 'application/json'}, cache: 'no-store',
        body: JSON.stringify({mode, character: o.character, history: o.history && o.history.length ? o.history.slice(-20) : undefined,
          planner: o.planner === false ? false : undefined})});
      let j = null; try { j = await r.json(); } catch (e) {}
      if (!r.ok || !j || !j.token) { const msg = (j && j.error) || ('HTTP ' + r.status); stop(); emitState('error', msg); throw new Error(msg); }
      V.createMs = Math.round(now() - t0); V.model = j.model; V.gemTools = !!j.tools;
      await ensureWorklet(ctx);
      const player = new AudioWorkletNode(ctx, 'avatar-pcm-player', {numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1]});
      player.connect(bus); V.player = player;
      const ws = new WebSocket(j.url + '?access_token=' + encodeURIComponent(j.token)); V.ws = ws; V.respN = 0; V.frames = 0;
      ws.onopen = () => { log('ws_open'); ws.send(JSON.stringify({setup: {}})); };
      ws.onmessage = async e => { let m; try { m = JSON.parse(typeof e.data === 'string' ? e.data : await e.data.text()); } catch (x) { return; } onGemini(m); };
      ws.onerror = () => log('ws_error');
      ws.onclose = ev => { log('ws_close', {code: ev.code, reason: String(ev.reason || '').slice(0, 120)});
        if (V.stopping) return; if (V.ready) drop('gemini closed ' + ev.code); else emitState('error', 'Gemini: ' + (String(ev.reason || '').slice(0, 100) || ('closed ' + ev.code))); };
      const ms = await micForVendor();
      if (ms) { const src = ctx.createMediaStreamSource(ms), tap = new AudioWorkletNode(ctx, 'avatar-pcm16-tap', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
        const sink = ctx.createGain(); sink.gain.value = 0; src.connect(tap); tap.connect(sink).connect(ctx.destination);
        tap.port.onmessage = ev => { if (V.ready && ws.readyState === 1) ws.send('{"realtimeInput":{"audio":{"mimeType":"audio/pcm;rate=16000","data":"' + b64(ev.data) + '"}}}'); };
        V.gemTap = {src, tap, sink}; }
      setTimeout(() => { if (!V.ready && V.state === 'connecting') { emitState('error', 'Gemini setup timeout'); stop(); } }, 12000);
    }
    function startVision() {          // a low, steady camera stream (~1 fps, small frames); ?vfps= overrides
      clearInterval(V.visTimer); const fps = o.visionFps === undefined ? 2 : +o.visionFps;
      if (!(fps > 0) || !o.grabStill) return;
      V.visTimer = setInterval(() => { if (!V.ready) return; const url = o.grabStill(); if (!url) return;
        if (send({realtimeInput: {video: {mimeType: 'image/jpeg', data: url.slice(url.indexOf(',') + 1)}}})) V.frames++; }, Math.round(1000 / fps));
    }
    function onGemini(m) {
      if (m.setupComplete) { V.ready = true; V.startupMs = Math.round(now() - V.startedAt); log('setup_complete'); emitState('live'); armRenew(null); startVision(); return; }
      const sc = m.serverContent;
      if (sc) {
        if (sc.interrupted) { log('interrupted'); V.q.turnAsked = false; V.q.dropTurn = false; V.q.turnWords = 0; if (V.player) V.player.port.postMessage({clear: true}); if (o.onInterrupt) o.onInterrupt(); if (o.onResponse) o.onResponse('truncated', {response_id: V.respId}); promptDone(null, 'interrupted'); }
        if (sc.inputTranscription && sc.inputTranscription.text) quietHeard(sc.inputTranscription.text);
        if (V.q.on && !V.q.turnAsked && ((sc.modelTurn && sc.modelTurn.parts && sc.modelTurn.parts.length) || (sc.outputTranscription && sc.outputTranscription.text))) {
          // a reply began while quiet — the user's words since the quiet began decide (wake: it plays; else its audio is
          // dropped + interrupted). No user words = the model talking on its own (a tool result, a note): dropped.
          V.q.turnAsked = true;
          if (!V.q.heard.trim() || quietAsk('reply') !== 'wake') { V.q.dropTurn = true; gemInterrupt(); }
        }
        if (sc.outputTranscription && sc.outputTranscription.text && V.talk.gemWords && !V.q.dropTurn) {      // length backstop
          V.q.turnWords = (V.q.turnWords || 0) + sc.outputTranscription.text.split(/\s+/).filter(Boolean).length;
          if (V.q.turnWords > V.talk.gemWords) { V.q.dropTurn = true; V.q.n.capped++; log('talk_capped', {words: V.q.turnWords}); gemInterrupt(true); }
        }
        const qdrop = V.q.on || V.q.dropTurn;
        for (const p of (sc.modelTurn && sc.modelTurn.parts) || []) {
          if (qdrop && p.inlineData && /audio/.test(p.inlineData.mimeType || '')) { V.q.n.audioDropped++; continue; }
          if (p.inlineData && /audio/.test(p.inlineData.mimeType || '')) { const b = atob(p.inlineData.data), a = new Int16Array(b.length >> 1);
            for (let i = 0; i < a.length; i++) a[i] = ((b.charCodeAt(2 * i) | (b.charCodeAt(2 * i + 1) << 8)) << 16) >> 16;
            if (V.player) V.player.port.postMessage(a, [a.buffer]); V.serverSpeaking = true; if (!V.respId) { V.respId = 'g' + (++V.respN); promptCreated({id: V.respId}); } promptAudio(V.respId); }
        }
        if (sc.inputTranscription && sc.inputTranscription.text) { const t = sc.inputTranscription.text;
          V.lastUserSpeechAt = now();
          if (!V._gemUser) { V._gemUser = true; if (o.onActivity) o.onActivity('speech'); if (o.onUserSpeech) { try { o.onUserSpeech('start', {}); } catch (e) {} }
            if (o.grabStill && now() - V.lastStill > 1500) { const u = o.grabStill(); if (u) { send({realtimeInput: {video: {mimeType: 'image/jpeg', data: u.slice(u.indexOf(',') + 1)}}}); V.lastStill = now(); } } }   // the still gpt-realtime gets at speech start
          V.transcriptYou += t; if (o.onTranscript) o.onTranscript('you', t, false); }
        if (sc.outputTranscription && sc.outputTranscription.text) { if (V._gemUser) { V._gemUser = false; if (o.onTranscript) o.onTranscript('you', V.transcriptYou, true); V.transcriptYou = ''; }
          if (!V.respId) { V.respId = 'g' + (++V.respN); promptCreated({id: V.respId}); } if (!qdrop) { V.avText = (V.avText || '') + sc.outputTranscription.text; if (o.onTranscript) o.onTranscript('avatar', sc.outputTranscription.text, false, {response_id: V.respId}); } }
        if (sc.turnComplete) { if (V._gemUser) { V._gemUser = false; if (o.onTranscript) o.onTranscript('you', V.transcriptYou, true); V.transcriptYou = ''; }
          if (o.onTranscript && (V.avText || !(V.q.on || V.q.dropTurn))) o.onTranscript('avatar', V.avText || '', true, {response_id: V.respId}); V.avText = ''; if (o.onResponse) o.onResponse('done', {status: 'completed', response_id: V.respId}); const rid = V.respId; V.respId = null; V.serverSpeaking = false; promptDone(rid, 'completed');
          gemTurnEnd(); }
      }
      const batch = m.toolCall ? 'gtc' + (V.tcN = (V.tcN || 0) + 1) : null;            // the calls of one toolCall = one request
      if (m.toolCall) for (const fc of m.toolCall.functionCalls || []) {
        const c = {name: fc.name, t: Math.round(now()), start_ms: Math.round(now()), batch, start_lat_ms: V.turnOpen && V.turnOpen.user_end_perf ? Math.round(now() - V.turnOpen.user_end_perf) : null};
        let out = {ok: true};
        const qo = quietTool(fc.name, fc.args || {}, c);
        if (qo) out = qo;
        else if (o.onCall) { try { const r = o.onCall(fc.name, fc.args || {}, 'done', c); if (r && typeof r === 'object') out = r; } catch (e) { out = {ok: false, error: String(e.message || e)}; } }
        V.calls.push(c); send({toolResponse: {functionResponses: [{id: fc.id, name: fc.name, response: out}]}});
      }
      if (m.usageMetadata) { V.usage.push(m.usageMetadata); metric({type: 'usage', usage: m.usageMetadata, vendor: 'gemini'}); }
      if (m.goAway) { log('goaway'); if (!V.stopping && o.onDrop) o.onDrop('renew', true); }
    }
    /* ---- auto-recovery hooks: the page re-dials in the background and swaps audio when the new call is live */
    function drop(reason) {
      if (V.dropped || V.stopping) return;
      V.dropped = true; log('drop', {reason}); clearTimeout(V.renewTimer); clearTimeout(V.discTimer);
      shieldOff('drop');   // never leave the user's mic held on a dead session
      if (o.onDrop) o.onDrop(reason, false);
    }
    function watchPc(pc) {
      pc.addEventListener('connectionstatechange', () => {
        const st = pc.connectionState; log('pc:' + st);
        if (st === 'connected') { clearTimeout(V.discTimer); V.discTimer = 0; }
        else if (st === 'failed' || (st === 'closed' && !V.stopping)) drop('connection ' + st);
        else if (st === 'disconnected' && !V.discTimer) V.discTimer = setTimeout(() => { V.discTimer = 0;
          if (pc.connectionState !== 'connected') drop('disconnected > 3 s'); }, 3000);
      });
    }
    // renew before the vendor's maximum session length (expires_at, seconds since epoch); default 28 min
    function armRenew(expiresAt) {
      clearTimeout(V.renewTimer);
      let ms = (+expiresAt > 1e9 ? +expiresAt * 1000 - Date.now() - 90000 : 28 * 60000);
      if (o.renewMs) ms = o.renewMs;
      ms = Math.max(30000, Math.min(ms, 2 ** 31 - 1));
      V.renewAt = Date.now() + ms;
      V.renewTimer = setTimeout(() => { if (!V.stopping && !V.dropped && o.onDrop) { log('renew'); o.onDrop('renew', true); } }, ms);
    }
    // realtime: the new call starts knowing the conversation so far (one system note; GPT-Live gets it as session.input)
    function seedHistory() {
      const h = (o.history || []).slice(-20).filter(x => x && x.text);
      if (mode !== 'realtime' || !h.length) return;
      const lines = h.map(x => (x.role === 'user' ? 'Him: ' : 'You: ') + String(x.text).slice(0, 300)).join('\n');
      send({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text',
        text: SILENT + 'the conversation so far (the call reconnected; carry on naturally, do not repeat or mention it):\n' + lines}]}});
      log('history_seeded', {n: h.length});
    }
    function testDrop() { const pc = V.pc || (V.loop && V.loop[1]); if (pc) pc.close(); if (!V.pc) drop('test: loopback pc closed'); }

    function stop() {
      shieldOff('stop');
      if (V.state === 'stopped') return;
      V.stopping = true; clearTimeout(V.renewTimer); clearTimeout(V.discTimer); clearTimeout(V.ctxTimer); clearTimeout(V.pqTimer);
      if (mode === 'live' && V.dc && V.dc.readyState === 'open') { try { send({type: 'session.close'}); } catch (e) {} }
      if (GEM) { clearInterval(V.visTimer); const w = V.ws; V.ws = null; try { w && w.close(); } catch (e) {} if (V.gemTap) { try { V.gemTap.src.disconnect(); V.gemTap.tap.disconnect(); } catch (e) {} V.gemTap = null; } if (V.player) { try { V.player.disconnect(); } catch (e) {} V.player = null; } }
      const pc = V.pc, dc = V.dc; V.pc = null; V.dc = null;
      if (V.loop) { for (const x of V.loop) { try { x.close(); } catch (e) {} } V.loop = null; }
      setTimeout(() => { try { dc && dc.close(); } catch (e) {} try { pc && pc.close(); } catch (e) {} }, mode === 'live' ? 1200 : 0);
      if (V.remote) { try { V.remote.disconnect(); } catch (e) {} V.remote = null; }
      if (V.audioEl) { V.audioEl.srcObject = null; V.audioEl = null; }
      if (V.mockSrc) { try { V.mockSrc.stop(); } catch (e) {} V.mockSrc = null; }
      if (V.meter) { try { V.meter.src.disconnect(); V.meter.m.disconnect(); } catch (e) {} V.meter = null; }
      if (V.gateNodes) { const G = V.gateNodes; try { G.src.disconnect(); G.g.disconnect(); if (o.gateRef) o.gateRef.disconnect(G.g); } catch (e) {} V.gateNodes = null; }
      V.ready = false; emitState('stopped');
    }

    // ---- camera / perception events: speak only on real events, rate-limited
    function canRespond() {
      const t = now();
      V.eventResps = V.eventResps.filter(x => t - x < 60000);
      const why = unpromptedWhy(); if (why) return why;
      if (t - V.lastEventResp < V.talk.gapMs || V.eventResps.length >= V.talk.perMin) return 'rate';
      if (V.micSpeaking || V.serverSpeaking) return 'user speaking';
      if (V.avatarSpeaking && t - V.lastAvatarVoicePerf < 1500) return 'avatar speaking';
      if (avatarAudible()) return 'avatar speaking';             // never a second utterance over one still playing
      if (modelBusy()) return 'reply in progress';               // … or behind one being made (it would play late)
      if (V.pendingCalls.size) return 'busy';
      return '';
    }
    function sendEvent(text, opt) {
      opt = opt || {};
      if (!V.ready || MOCK) return {sent: false, responded: false, why: MOCK ? 'mock' : 'not ready'};
      const why = opt.respond ? canRespond() : 'no';
      const respond = !!opt.respond && !why;
      // a spoken event that is rate-limited is dropped, not queued as context (stale context made later replies
      // mention it: "tell me what you're showing"); respond:false callers get context-only on purpose
      // while quiet a camera event (it brings its frame) still reaches the model as context (it keeps watching), it
      // just gets no reply; a cue to speak (a turn bid) is dropped like any rate-limited event
      if (opt.respond && !respond && !((why === 'quiet' || why === 'resume calm') && opt.image)) { log('event_dropped', {why}); return {sent: false, responded: false, why}; }
      // never a tag the model could read out ('[camera]' was): one plainly-marked silent note
      const note = SILENT + String(text).slice(0, 200);
      const img = opt.image && /^data:image\/jpeg;base64,/.test(opt.image) && opt.image.length < 150000 ? opt.image : null;
      if (GEM) {
        if (img) send({realtimeInput: {video: {mimeType: 'image/jpeg', data: img.slice(img.indexOf(',') + 1)}}});
        send({clientContent: {turns: [{role: 'user', parts: [{text: note}]}], turnComplete: respond}});
      } else if (mode === 'realtime') {
        // images ride a user item; text-only notes are system items (not words the user said)
        const item = img ? {type: 'message', role: 'user', content: [{type: 'input_text', text: note}, {type: 'input_image', image_url: img, detail: 'low'}]}
          : {type: 'message', role: 'system', content: [{type: 'input_text', text: note}]};
        send({type: 'conversation.item.create', item});
        if (respond) { V.evRespAt = now(); wantResponse({type: 'response.create'}); }   // its reply is unprompted (never shielded)
      } else {
        // GPT-Live: the voice model gets it directly (instructions.append speaks, thinking.append is context only)
        send({type: respond ? 'session.instructions.append' : 'session.thinking.append', delegation_id: null, content: note});
      }
      if (respond) { V.lastEventResp = now(); V.eventResps.push(now()); }
      if (opt.image) V.lastStill = now();
      log(respond ? 'event_spoken' : 'event_context', {why: why || undefined});
      return {sent: true, responded: respond, why};
    }
    /* sendContext(text): a silent context item for the model (expression changes, what the camera sees, …): never
       read aloud, never described, no response of its own; at most one per 5 s (the newest pending note wins) */
    function sendContext(text) {
      if (!V.ready || MOCK || !text) return false;
      V.ctxPending = String(text).slice(0, 240);
      const go = () => {
        V.ctxTimer = 0;
        if (!V.ctxPending || !V.ready) return;
        const t = now();
        if (t - (V.lastCtx || -1e9) < 5000) { V.ctxTimer = setTimeout(go, 5000 - (t - V.lastCtx)); return; }
        const note = SILENT + V.ctxPending; V.ctxPending = null; V.lastCtx = t;
        if (GEM) send({clientContent: {turns: [{role: 'user', parts: [{text: note}]}], turnComplete: false}});
        else if (mode === 'realtime') send({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: note}]}});
        else send({type: 'session.thinking.append', delegation_id: null, content: note});
        log('context_sent');
      };
      if (!V.ctxTimer) go();
      return true;
    }
    /* speech — sendNote(text, {replace}): the action log's own silent channel (no 5 s pacing; the page coalesces).
       realtime: a system item with our own id (replace: the previous note item is deleted, so only the newest log is in
       context); GPT Live: session.thinking.append (context only, never spoken). */
    function sendNote(text, opt) {
      opt = opt || {};
      if (!V.ready || MOCK || !text) return false;
      const note = SILENT + String(text).slice(0, opt.maxChars || 1400);
      if (GEM) send({clientContent: {turns: [{role: 'user', parts: [{text: note}]}], turnComplete: false}});
      else if (mode === 'realtime') {
        const id = 'alog_' + (V.noteN = (V.noteN || 0) + 1) + '_' + Math.random().toString(36).slice(2, 7);
        if (opt.replace && V.noteId) send({type: 'conversation.item.delete', item_id: V.noteId});
        send({type: 'conversation.item.create', item: {id, type: 'message', role: 'system', content: [{type: 'input_text', text: note}]}});
        V.noteId = id;
      } else send({type: 'session.thinking.append', delegation_id: null, content: note});
      log('note_sent', {chars: note.length});
      return true;
    }
    // ---- prompt(): a silent note that may ask for a spoken reply (the page's cue / check-in / honest failure
    // around a planned action). Never a hold on the avatar's own audio: only WHEN a reply is requested is managed here.
    function wantResponse(ev) { const ok = send(ev); if (ok) V.respWantedAt = now(); return ok; }   // a blocked create is not wanted
    function userTalking() {
      if (V.micSpeaking) return true;
      if (GEM) return !!V._gemUser && now() - V.lastUserSpeechAt < 2500;   // (no reply to the user's words: not forever)
      if (mode === 'live') return !!V._liveUserTalking;
      if (mode !== 'realtime') return false;
      if (V.serverSpeaking) return true;
      // the user's turn just ended: the server's own reply to it is about to be created (server_vad create_response)
      return !!(V._lastStop && now() - V._lastStop.perf < 1500 && V.lastRespCreatedAt < V._lastStop.perf);
    }
    function modelBusy() {
      const t = now();
      if (V.pCreating && t - (V.pCreating.firedAt || t) > 5000) { V.pCreating.state = 'lost'; V.pCreating = null; }   // never matched
      if (GEM) return !!V.respId || !!V.pCreating;
      if (mode !== 'realtime') return false;
      for (const [id, at] of V.respActive) if (t - at > 30000) V.respActive.delete(id);              // a lost response.done
      return V.respActive.size > 0 || !!V.pCreating || (V.respWantedAt > 0 && t - V.respWantedAt < 3000);
    }
    const pview = p => ({id: p.id, get state() { return p.state; }, get why() { return p.why; }, cancel: () => { if (p.state === 'queued') pdrop(p, 'cancelled'); }});
    function pdrop(p, why) {
      p.state = 'dropped'; p.why = why; log('prompt_dropped', {id: p.id, why});
      if (p.o.onDrop) { try { p.o.onDrop(why); } catch (e) {} }
      return pview(p);
    }
    function prompt(text, opt) {
      opt = opt || {};
      const p = {id: 'avp' + (++V.promptN), note: SILENT + String(text || '').slice(0, opt.maxChars || 700), respond: !!opt.respond,
        at: now(), maxWaitMs: opt.maxWaitMs != null ? +opt.maxWaitMs : (opt.respond ? 2500 : 5000),
        dropIfUserSpoke: opt.dropIfUserSpoke != null ? !!opt.dropIfUserSpoke : !!opt.respond, o: opt, state: 'queued', why: ''};
      V.prompts.push(p); if (V.prompts.length > 60) V.prompts.splice(0, 30);
      // quiet = context only, no reply; an unprompted reply (turn bid, gesture, camera, behaviour moment) also obeys
      // the talkativeness gate (off at minimal, min gap / per-minute cap) and the calm after a quiet period
      if (p.respond) { const qw = V.q.on ? 'quiet' : opt.unprompted ? (unpromptedWhy() || unpromptedRate()) : '';
        if (qw && opt.unprompted) { V.q.n.blocked++; return pdrop(p, qw); }      // a cue to speak up: dropped, not left as stale context
        if (qw) { p.respond = false; p.why = qw; V.q.n.notesOnly++; log('prompt_note_only', {id: p.id, why: qw}); } else if (opt.unprompted) V.eventResps.push(now()); }
      if (!V.ready || !text) return pdrop(p, V.ready ? 'empty' : 'not ready');
      if (MOCK) return pdrop(p, 'mock');
      if (!GEM && (mode === 'realtime' || !p.respond)) {
        // realtime: the note is context at once (an item never interrupts a reply); GPT Live: a context-only note at once
        if (mode === 'realtime') send({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: p.note}]}});
        else send({type: 'session.thinking.append', delegation_id: null, content: p.note});
        if (!p.respond) { p.state = 'sent'; p.firedAt = now(); log('prompt_note', {id: p.id}); return pview(p); }
      }
      V.pq.push(p); pump();
      return pview(p);
    }
    function pqKick() { if (V.pq.length) { clearTimeout(V.pqTimer); V.pqTimer = setTimeout(pump, 0); } }   // re-check now (not at the next poll)
    function pump() {
      clearTimeout(V.pqTimer); V.pqTimer = 0;
      while (V.pq.length) {
        const p = V.pq[0];
        if (p.state !== 'queued') { V.pq.shift(); continue; }
        if (!V.ready || V.stopping) { V.pq.shift(); pdrop(p, 'not ready'); continue; }
        if (p.dropIfUserSpoke && V.lastUserSpeechAt > p.at) { V.pq.shift(); pdrop(p, 'he spoke'); continue; }
        if (now() - p.at > p.maxWaitMs) { V.pq.shift(); pdrop(p, 'stale'); continue; }
        const why = userTalking() ? 'he is speaking' : modelBusy() ? 'reply in progress' : (p.respond && p.o.unprompted && avatarAudible()) ? 'avatar speaking' : '';   // unprompted never over its own voice
        if (why) { p.why = why; break; }
        V.pq.shift(); fire(p);
        if (p.respond) break;                                    // one prompted reply at a time
      }
      if (V.pq.length) V.pqTimer = setTimeout(pump, 80);
    }
    function fire(p) {
      if (V.q.on && p.respond) { pdrop(p, 'quiet'); return; }
      p.firedAt = now(); p.why = '';
      if (GEM) {      // a clientContent message interrupts a generation: sent only while the model is idle (pump)
        send({clientContent: {turns: [{role: 'user', parts: [{text: p.note}]}], turnComplete: p.respond}});
        if (p.respond) { p.state = 'creating'; V.pCreating = p; V.respWantedAt = now(); } else p.state = 'sent';
      } else if (mode === 'realtime') {
        p.state = 'creating'; V.pCreating = p; p.eventId = 'evp_' + p.id;
        wantResponse(V.noMeta ? {type: 'response.create', event_id: p.eventId}
          : {type: 'response.create', event_id: p.eventId, response: {metadata: {avatar_prompt: p.id}}});
      } else {
        send({type: 'session.instructions.append', delegation_id: null, content: p.note});
        p.state = 'sent';
      }
      log(p.respond ? 'prompt_respond' : 'prompt_note', {id: p.id, wait_ms: Math.round(p.firedAt - p.at)});
      if (p.o.onSent) { try { p.o.onSent({id: p.id, wait_ms: Math.round(p.firedAt - p.at)}); } catch (e) {} }
    }
    // a reply was created: realtime response.created (ours by metadata), Gemini a new model turn
    function promptCreated(r) {
      const t = now(); V.lastRespCreatedAt = t; V.respWantedAt = 0;
      if (!GEM && r.id) V.respActive.set(r.id, t);
      const p = V.pCreating; if (!p || p.state !== 'creating') return;
      const md = r.metadata && typeof r.metadata === 'object' ? r.metadata : null;
      const heSpoke = V.lastUserSpeechAt > p.firedAt || !!(V._lastStop && V._lastStop.perf > p.firedAt);
      const ours = md && md.avatar_prompt ? md.avatar_prompt === p.id : (GEM || V.noMeta || t - p.firedAt < 2000) && !heSpoke;
      if (!ours) return;
      p.state = 'speaking'; p.respId = r.id || null; p.createdAt = t; V.pCreating = null;
      if (p.o.unprompted && r.id) V.unprompted.add(r.id);
    }
    function promptAudio(respId) {
      for (let i = V.prompts.length - 1; i >= 0 && i >= V.prompts.length - 8; i--) {
        const p = V.prompts[i];
        if (p.state === 'speaking' && !p.audioAt && (!respId || !p.respId || p.respId === respId)) {
          p.audioAt = now(); log('prompt_audio', {id: p.id, ms: Math.round(p.audioAt - p.at)});
          if (p.o.onAudio) { try { p.o.onAudio({id: p.id, response_id: respId || null, wait_ms: Math.round(p.audioAt - p.at)}); } catch (e) {} }
        }
      }
    }
    function promptDone(respId, status) {
      if (!GEM && respId) V.respActive.delete(respId);
      for (let i = V.prompts.length - 1; i >= 0 && i >= V.prompts.length - 8; i--) {
        const p = V.prompts[i];
        const mine = (p.state === 'speaking' && (!respId || !p.respId || p.respId === respId)) ||
          (GEM && p.state === 'creating' && p === V.pCreating && !(V.lastUserSpeechAt > p.firedAt));   // a turn with no audio
        if (!mine) continue;
        if (p === V.pCreating) V.pCreating = null;
        p.state = 'done'; p.status = status;
        if (p.o.onDone) { try { p.o.onDone({id: p.id, status}); } catch (e) {} }
      }
      pqKick();
    }
    function promptError(err) {
      const p = V.pCreating, code = String(err.code || '');
      if (!p || !err.event_id || err.event_id !== p.eventId) {
        if (code === 'conversation_already_has_active_response') V.respWantedAt = 0;   // someone else's create lost the race
        return;
      }
      V.pCreating = null; V.respWantedAt = 0;
      const meta = /metadata/i.test(String(err.param || '') + ' ' + String(err.message || ''));
      if (meta) V.noMeta = true;                               // this server does not take response.metadata: match by order
      if (meta || code === 'conversation_already_has_active_response') { p.state = 'queued'; V.pq.unshift(p); pqKick(); }
      else pdrop(p, 'vendor error ' + code);
    }
    // ---- shield: the sentence the user asked for is not cut by their next words. The vendor clears its audio the instant its
    // server VAD hears speech: a reply to a command ("Sure, here it comes") or the cue / done line around the move was cut
    // mid-word whenever the user began the next command, coughed, or said "ok" while it played (field tests, the Muse / Felix
    // montages; the replay logs: all 14 cut-offs sat right behind input_audio_buffer.speech_started, none behind a
    // cancel / clear of ours). turn_detection.interrupt_response=false does NOT stop it for the WebRTC audio buffer (measured:
    // the server reports it false and still sends output_audio_buffer.cleared + conversation.item.truncated on the same
    // speech_started), so the vendor must not HEAR the user while the sentence is audible: the gate worklet queues the user's (gated)
    // mic from the moment a reply's audio starts and replays it in real time when that audio ends (output_audio_buffer.stopped /
    // cleared), at the next pause of the avatar's audible voice (>= 300 ms once the user's words have waited 2 s, >= 700 ms at once:
    // the sentence the user talked over has ended and the rest of the reply yields to them) or after SHIELD_MS, dropping the silence of the backlog so the delay does
    // not stay: the user is heard a little late (the rest of the sentence), never over it. The cap restarts with each further reply of a chain (reply, cue, done);
    // every reply of ours is shielded (a turn bid's greeting too: "Nice to see you" was cut to "Nice to"); tap-to-interrupt (openMic) releases it at once.
    const SHIELD_MS = 12000;                      // safety valve only (a lost event); the audio's own end releases it
    const shieldPort = () => (V.gateNodes && V.gateNodes.g.port) || o.shieldPort || null;
    function shieldWire(on) {                     // the gate worklet queues the gated mic while on, replays it when off
      const port = shieldPort(); if (mode !== 'realtime' || MOCK || !port) return false;
      port.postMessage({shield: !!on}); return true;
    }
    function shieldArm(rid) {
      V.sh.rid = rid; clearTimeout(V.sh.timer); V.sh.timer = setTimeout(() => shieldOff('cap'), SHIELD_MS);
    }
    function shieldOn(rid) {                      // the user's words must not cut the reply that is audible now
      if (V.q.on || !rid || V.micMuted || V.spkMuted) return;   // a muted mic has nothing to hold; a muted speaker has no sentence to protect
      if (V.sh.on) { shieldArm(rid); return; }
      if (!shieldWire(true)) return;
      V.sh.on = true; V.sh.n++; shieldArm(rid);
      log('shield_on', {rid});
    }
    function shieldOff(why, fromGate) {
      if (!V.sh.on) return;
      clearTimeout(V.sh.timer); V.sh.timer = 0; V.sh.on = false; V.sh.rid = null;
      if (!fromGate) shieldWire(false); log('shield_off', {why});
    }
    function shieldSpeech() {                     // the user began speaking while the shield is up
      if (!V.sh.on) return;
      V.sh.held++; log('shield_held');
      // the server's reply to a half sentence of the user's (no audio yet, not asked for by one of our prompts / events): cancelled, as the
      // server's own interrupt would have; the cue / done lines our prompts asked for are kept (they are the sentence the user asked for)
      const ours = id => V.unprompted.has(id) || V.prompts.some(p => p.respId === id);
      for (const id of Array.from(V.respActive.keys())) if (!V.respHasAudio.get(id) && !ours(id)) { V.sh.other++; send({type: 'response.cancel', response_id: id}); log('shield_cancel_other', {rid: id}); }
    }
    // ---- quiet: the harness speech gate. app/quiet.js owns the policy (level, timer, addressed speech); here it is
    // enforced on the wire: while quiet nothing creates a response (Realtime: turn_detection.create_response=false and
    // every response.create dropped, a leaked response cancelled; Gemini Live: its turn audio dropped + interrupted; GPT
    // Live: respond notes become thinking; mock: no reply WAV) and the voice bus is muted. It keeps listening: the user's
    // transcripts and items stay in the conversation, and o.onQuietTurn(seg) (the controller) may end the quiet period.
    const STALE_MS = 5000;
    /* its voice is still coming out: the vendor's output buffer has not played out (Realtime), or the page hears it */
    function avatarAudible() {
      if (V.outPlaying && now() - V.outPlaying < 120000) return true;
      if (o.audible) { try { if (o.audible()) return true; } catch (e) {} }
      return false;
    }
    function segOf(g) { return {at: now(), level: g.n ? g.sum / g.n : null, peak: g.pk || null, durMs: g.n * 10, echoFrames: g.echo || 0}; }
    function quietTurn(seg) {
      V.q.n.turns++;
      const ls = V.uSeg && V.uSeg.n >= 20 ? segOf(V.uSeg) : V.lastSeg && now() - V.lastSeg.at < 8000 ? V.lastSeg : null;   // the user's newest utterance at the mic
      const s2 = Object.assign({mode: GEM ? 'gemini' : MOCK ? 'mock' : mode}, seg);
      if (ls) { if (s2.level === undefined) s2.level = ls.level; if (s2.durMs == null) s2.durMs = ls.durMs; if (ls.echoFrames > ls.durMs / 20) s2.echo = true; }
      let r = 'hold';
      const qc = quietCtl();
      if (o.onQuietTurn || (qc && qc.onQuietTurn)) { try { r = (o.onQuietTurn ? o.onQuietTurn(s2) : qc.onQuietTurn(s2)) || 'hold'; } catch (e) { r = 'hold'; } }
      if (r === 'wake' && V.q.on) setQuiet(false, {});    // the controller ended it (or none is attached): the gate opens
      if (r === 'wake') V.q.n.woke++;
      log('quiet_turn', {r, chars: String(seg.text || '').length});
      return r;
    }
    function quietFilter(ev) {
      const t = ev && ev.type;
      if (t === 'response.create') { V.q.n.blocked++; log('quiet_blocked', {what: 'response.create'}); return null; }
      if (ev && ev.clientContent && ev.clientContent.turnComplete) {      // Gemini: context yes, a reply no
        V.q.n.notesOnly++; return {clientContent: Object.assign({}, ev.clientContent, {turnComplete: false})};
      }
      if (t === 'session.instructions.append') { V.q.n.notesOnly++; return Object.assign({}, ev, {type: 'session.thinking.append'}); }
      return ev;
    }
    function qsend(ev) { V.q.bypass++; try { return send(ev); } finally { V.q.bypass--; } }
    function unpromptedWhy() {
      if (V.q.on) return 'quiet';
      if (now() < V.q.calmUntil) return 'resume calm';
      if (V.talk.off) return 'talkativeness';
      return '';
    }
    function unpromptedRate() {           // prompt({unprompted}) only below the normal level (normal keeps its old behaviour)
      if (V.talk.level === 'normal' || V.talk.gapMs <= 10000) return '';
      const t = now(); V.eventResps = V.eventResps.filter(x => t - x < 60000);
      const last = Math.max(V.lastEventResp, V.eventResps.length ? V.eventResps[V.eventResps.length - 1] : -1e12);
      return t - last < V.talk.gapMs || V.eventResps.length >= V.talk.perMin ? 'rate' : '';
    }
    function muteOut(on) {
      V.q.muted = !!on;
      const g = bus.gain, v = on ? 0 : 1;
      try { if (g.setTargetAtTime) { if (g.cancelScheduledValues) g.cancelScheduledValues(ctx.currentTime); g.setTargetAtTime(v, ctx.currentTime, 0.015); } else g.value = v; } catch (e) { try { g.value = v; } catch (x) {} }
      if (on && V.player) V.player.port.postMessage({clear: true});
      if (on && V.mockSrc) { try { V.mockSrc.stop(); } catch (e) {} V.mockSrc = null; }
    }
    function quietCancel(respId) {        // Realtime: stop a response that is running while quiet (no more tokens, no audio)
      V.q.cancelled.add(respId); if (V.q.cancelled.size > 50) V.q.cancelled = new Set([...V.q.cancelled].slice(-20));
      V.q.n.cancelled++; log('quiet_cancel', {id: respId});
      qsend({type: 'response.cancel', response_id: respId});
      if (!o.transport || mode === 'realtime') qsend({type: 'output_audio_buffer.clear'});
      if (o.onInterrupt) { try { o.onInterrupt(); } catch (e) {} }
    }
    function gemInterrupt(keepQueued) {   // Gemini: a client turn interrupts the generation (one per model turn)
      V.q.n.interrupts++; log('quiet_interrupt');
      if (V.player && !keepQueued) V.player.port.postMessage({clear: true});   // (a length cut lets the words already queued finish)
      qsend({clientContent: {turns: [{role: 'user', parts: [{text: SILENT + (V.q.on ? 'Quiet mode: your voice is muted by the app and nothing you say is heard. Do not reply until the app tells you he is talking to you again.'
        : 'Your reply ran past the length he asked for and was cut off; keep the next one much shorter.')}]}], turnComplete: false}});
    }
    function gemTurnEnd() {
      const asked = V.q.turnAsked; V.q.turnAsked = false; V.q.dropTurn = false; V.q.turnWords = 0;
      // the user's words finished after the reply had already been dropped (the transcript lagged): decide once more on all of them
      if (V.q.on && asked && V.q.heard.trim() && V.q.heard.length > (V.q.askedLen || 0) && quietAsk('final') === 'wake') respondNow();
    }
    /* Gemini Live / GPT Live: the user's words while quiet arrive as transcript pieces with no end-of-utterance event, and the
       model may not reply at all (it was told it is muted) — so the harness decides 900 ms after the last piece. */
    function quietHeard(piece) {
      if (!V.q.on || !piece || now() - V.q.since < 800) return;      // (the tail of the request for quiet itself)
      V.q.heard += piece;
      clearTimeout(V.q.heardT);
      V.q.heardT = setTimeout(() => { V.q.heardT = 0; if (!V.q.on || !V.q.heard.trim()) return;
        const r = V.q.heard.length > (V.q.askedLen || 0) ? quietAsk('pause') : 'hold';
        if (r === 'wake') respondNow(); else { V.q.heard = ''; V.q.askedLen = 0; } }, 900);
    }
    function quietAsk(at) {
      V.q.askedLen = V.q.heard.length;
      const r = quietTurn({text: V.q.heard, at});
      if (r === 'wake') { V.q.heard = ''; V.q.askedLen = 0; clearTimeout(V.q.heardT); V.q.heardT = 0; }
      return r;
    }
    /* set_talkativeness / be_quiet / end_quiet are answered by the quiet controller (app/quiet.js), not the page's onCall */
    function quietCtl() { return o.quiet || (typeof window !== 'undefined' && window && window.AvatarQuietCtl) || null; }
    function quietTool(name, args, c) {
      const qc = quietCtl(); if (!qc || !qc.handles || !qc.handles(name)) return null;
      let out; try { out = qc.call(name, args || {}, c) || {ok: true}; } catch (e) { out = {ok: false, error: String(e.message || e)}; }
      log('quiet_tool', {name}); return out;
    }
    function quietReady() {               // the session is live: a level / quiet period set before it (or on the last call) goes out
      if (V.talk.pending) applyTalkWire();
      if (V.q.tdPending || V.q.on) rtTurnDetection();
    }
    function rtSession(sess, created) {
      if (!sess || typeof sess !== 'object') return;
      const td = sess.audio && sess.audio.input && sess.audio.input.turn_detection;
      if (td && typeof td === 'object') { V.sess = {td: Object.assign({}, td)}; log('session_td', {td}); }   // what the server reports (interrupt_response ...)
      if (created && V.baseInstr == null && typeof sess.instructions === 'string') V.baseInstr = sess.instructions;
      // (session.created after the call already counted as live — the 6 s fallback: what was sent blind goes out properly)
      if (created && V.ready) { if (V.talkSent || V.talk.level !== 'normal') V.talk.pending = true; quietReady(); }
    }
    function rtTurnDetection() {
      if (mode !== 'realtime' || MOCK) return;
      if (!V.sess || !V.sess.td || !V.sess.td.type) { V.q.tdPending = true; return; }
      V.q.tdPending = false;
      const want = !V.q.on;
      if (V.sess.td.create_response === want && V.q.tdSent === want) return;
      const td = Object.assign({}, V.sess.td, {create_response: want});
      V.q.tdSent = want; V.sess.td.create_response = want;
      qsend({type: 'session.update', session: {type: 'realtime', audio: {input: {turn_detection: td}}}});
      log('quiet_td', {create_response: want});
    }
    function setQuiet(on, opt) {
      opt = opt || {}; on = !!on;
      const was = V.q.on; V.q.on = on; V.q.wake = !!opt.wake;
      clearTimeout(V.q.graceT); V.q.graceT = 0;
      V.q.heard = ''; V.q.askedLen = 0; clearTimeout(V.q.heardT); V.q.heardT = 0;
      if (on) {
        if (!was) V.q.since = now();
        V.q.allowResp = false;
        // the response that asked for quiet may finish its few words ("okay, going quiet"): the voice is cut when that
        // audio has played out (Realtime: output_audio_buffer.stopped) or after graceMs, whichever is first
        const cut = V.q.cut = () => { clearTimeout(V.q.graceT); V.q.graceT = 0; if (!V.q.on || V.q.muted) return; muteOut(true);
          if (mode === 'realtime') for (const id of V.respActive.keys()) if (!V.q.cancelled.has(id)) quietCancel(id);
          if (GEM && V.respId) { V.q.dropTurn = true; gemInterrupt(); } };
        if (opt.graceMs > 0) V.q.graceT = setTimeout(cut, opt.graceMs); else cut();
        for (const p of V.pq) if (p.state === 'queued' && p.respond) pdrop(p, 'quiet');
        rtTurnDetection();
      } else {
        muteOut(false); V.q.dropTurn = false; V.q.turnAsked = false;
        V.q.calmUntil = opt.calmMs > 0 ? now() + opt.calmMs : 0;
        rtTurnDetection();
      }
      if (was !== on) log(on ? 'quiet_on' : 'quiet_off', {wake: V.q.wake || undefined});
      return true;
    }
    /* the voice answers the user now (the quiet period just ended on their words): bypasses the gate */
    function respondNow(note) {
      if (!V.ready) return false;
      if (GEM) { qsend({clientContent: {turns: [{role: 'user', parts: [{text: SILENT + (note || 'He is talking to you again: answer what he just said.')}]}], turnComplete: true}}); return true; }
      if (MOCK) { mockReply(); return true; }
      if (mode === 'realtime') { if (note) qsend({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: SILENT + note}]}});
        V.q.allowResp = true; V.respWantedAt = now(); return qsend({type: 'response.create'}); }
      return qsend({type: 'session.instructions.append', delegation_id: null, content: SILENT + (note || 'Answer what he just said.')});
    }
    /* talkativeness: {level, suffix, maxTokens, gemWords, gapMs, perMin, off} */
    function setTalk(t) {
      t = t || {};
      V.talk = {level: t.level || 'normal', suffix: t.suffix || '', maxTokens: t.maxTokens || null, gemWords: t.gemWords || null,
        gapMs: Number.isFinite(t.gapMs) || t.gapMs === Infinity ? t.gapMs : 10000, perMin: Number.isFinite(t.perMin) ? t.perMin : 4, off: !!t.off};
      applyTalkWire();
      log('talk', {level: V.talk.level});
      return true;
    }
    function applyTalkWire() {
      const T = V.talk;
      if (!V.ready) { T.pending = true; return; }
      T.pending = false;
      if (MOCK) return;
      if (T.level === 'normal' && !V.talkSent) return;      // never changed on this call: the session's own instructions stand
      V.talkSent = true;
      const text = T.suffix || 'How much you talk: as usual again (he chose this; it overrides any earlier guidance on reply length).';
      if (mode === 'realtime') {
        // the session's own instructions + the level's suffix, and the hard cap on one response's output
        const sess = {type: 'realtime', max_output_tokens: T.maxTokens || 'inf'};
        if (V.baseInstr != null) sess.instructions = V.baseInstr + (T.suffix ? '\n\n' + T.suffix : '');
        else qsend({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: SILENT + text}]}});
        qsend({type: 'session.update', session: sess});
      } else prompt(text, {respond: false, maxWaitMs: 20000});      // Gemini (sent while the model is idle) / GPT Live: a silent note
    }
    function sendToolResult(id, out) {
      const item = {type: 'function_call_output', call_id: id, output: JSON.stringify(out || {ok: true})};
      if (String(id).startsWith('mock_')) return;
      if (GEM) { send({toolResponse: {functionResponses: [{id, response: out || {ok: true}}]}}); return; }
      send(mode === 'live' ? {type: 'response.item.create', item} : {type: 'conversation.item.create', item});
    }
    const api = {
      start, stop, sendEvent, sendContext, sendNote, sendToolResult, audioNode: bus, mode, testDrop,   // speech: sendNote
      prompt, userTalking, lastUserSpeechAt: () => V.lastUserSpeechAt,
      feed: ev => {                                                                                    // test seam
        // test seams: the user's newest utterance as the mic meter measured it; the mock vendor's end of the user's turn
        if (ev && ev.type === 'test.mic_seg') { V.uSeg = null; V.lastSeg = Object.assign({at: now(), level: null, durMs: null, echoFrames: 0}, ev.seg || {}); return; }
        if (ev && ev.type === 'test.mock_user_end') { if (ev.bufs) V.mockBufs = ev.bufs; return onUserEnd(ctx.currentTime); }
        return (GEM ? onGemini : mode === 'live' ? onLive : onRealtime)(ev); },
      renewAt: () => V.renewAt || null, dropped: () => !!V.dropped,
      onAvatarChunk, // called by the page's playout tap
      userVoice: () => ({rms: V.userLvl, voiced: V.micSpeaking, pitch: null}),
      speaking: () => V.avatarSpeaking, listening: () => V.micSpeaking || V.serverSpeaking,
      send, state: () => V.state,
      setHardDuplex(on) { if (V.gateNodes) V.gateNodes.g.port.postMessage({hard: !!on}); },
      openMic(ms) { shieldOff('tap'); if (V.gateNodes) V.gateNodes.g.port.postMessage({openMs: ms || 2500}); return !!V.gateNodes; },   // tap-to-interrupt
      get gateNode() { return V.gateNodes ? V.gateNodes.g : null; },
      setMic,        // rec: switch microphones mid-session (no re-dial)
      /* mute. setMicMuted: the page has silenced the mic track; here the gate freezes (learns nothing from silence), anything
         held for the shield is dropped, and the vendor's uncommitted input buffer is cleared so half a sentence the user spoke just before
         tapping mute cannot turn into a reply. The session stays up; the avatar keeps talking. setSpeakerMuted: the avatar's voice is
         silenced at the speaker (the page), so there is nothing audible to gate against or to protect with the shield. */
      setMicMuted(on) {
        on = !!on; if (V.micMuted === on) return; V.micMuted = on; log('mic_muted', {on});
        const port = shieldPort(); if (port) port.postMessage({mute: on});
        if (on) { shieldOff('mute'); V.micSpeaking = false; V.serverSpeaking = false; if (mode === 'realtime' && !MOCK && V.ready) send({type: 'input_audio_buffer.clear'}); }
      },
      setSpeakerMuted(on) {
        on = !!on; if (V.spkMuted === on) return; V.spkMuted = on; log('spk_muted', {on});
        const port = shieldPort(); if (port) port.postMessage({refOff: on});
        if (on) shieldOff('speaker muted');
      },
      micMuted: () => V.micMuted, speakerMuted: () => V.spkMuted,
      setQuiet, setTalk, respondNow, quiet: () => ({on: V.q.on, wake: V.q.wake, muted: V.q.muted, ...V.q.n}), canRespond, talk: () => ({...V.talk}),
      lastSegment: () => V.lastSeg || null,
      stats: () => ({vision_frames: V.frames || 0, gem_tools: GEM ? !!V.gemTools : undefined, gate: V.gate || null, gate_on: !!V.gateNodes, mic_settings: V.micSettings || null, mode, model: V.model || (MOCK ? 'mock' : null), vad: o.vad || null, state: V.state,
        startup_ms: V.startupMs, create_ms: V.createMs, turns: V.turns.slice(-100), calls: V.calls.slice(-100),
        events: V.events, evlog: V.evlog.slice(window.__avEvlogAll ? 0 : -200), live_seconds: V.liveSeconds, usage_n: V.usage.length,
        mic_muted: V.micMuted, spk_muted: V.spkMuted, quiet: {on: V.q.on, wake: V.q.wake, muted: V.q.muted, ...V.q.n}, shield: {on: V.sh.on, n: V.sh.n, held: V.sh.held, other: V.sh.other}, talk_level: V.talk.level,
        prompts: V.prompts.slice(-30).map(p => ({id: p.id, respond: p.respond, state: p.state, why: p.why || undefined,
          wait_ms: p.firedAt ? Math.round(p.firedAt - p.at) : null, audio_ms: p.audioAt ? Math.round(p.audioAt - p.at) : null})),
        usage_tokens: V.usage.reduce((a, u) => [a[0] + (u.input_tokens || 0), a[1] + (u.output_tokens || 0)], [0, 0])}),
    };
    { const qc = quietCtl(); if (qc && qc.attach) { try { qc.attach(api); } catch (e) {} } }   // the level / quiet period carry over a re-dial
    return api;
  }

  window.AvatarVoice = {createVoice, createPlayout, TSR, SILENT};   // SILENT for the server-relayed notes
})();
