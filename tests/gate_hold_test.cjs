// hold: the REAL Gate worklet code from app/voice.js, run offline. While a reply is audible the gated mic is queued, not sent;
// on release it is replayed in real time (his words arrive intact, in order, a little late) and the silence of the backlog is dropped
// so the delay does not stay.   node tests/gate_hold_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'app/voice.js'), 'utf8');
const raw = src.slice(src.indexOf('const WORKLET = `') + 'const WORKLET = `'.length); const code = eval('`' + raw.slice(0, raw.indexOf('\n`;')) + '`');
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + JSON.stringify(detail) : '')); }
function makeGate() {
  const G = {t: 0, posts: []};
  let Gate = null;
  const reg = (name, cls) => { if (name === 'avatar-mic-gate') Gate = cls; };
  class AWP { constructor() { this.port = {postMessage: m => G.posts.push(m), onmessage: null}; } }
  Object.defineProperty(globalThis, 'currentTime', {get: () => G.t, configurable: true});
  new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', 'currentFrame', code)(AWP, reg, 48000, 0);
  G.node = new Gate(); G.msg = m => G.node.port.onmessage({data: m});
  return G;
}
const Q = 128, SR = 48000;
function run(G, blocks, refs) {                     // blocks: array of Float32Array(128) mic quanta (refs: the avatar's voice, same length) -> output quanta
  const out = [];
  blocks.forEach((b, i) => { const o = new Float32Array(Q); G.node.process([[b], [refs ? refs[i] : new Float32Array(Q)]], [[o]]); out.push(o); G.t += Q / SR; });
  return out;
}
let seed = 1; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const noise = n => Array.from({length: n}, () => Float32Array.from({length: Q}, () => rnd() * 0.004));
function speech(n, f0) { const bl = []; let ph = 0; for (let i = 0; i < n; i++) { const b = new Float32Array(Q); for (let j = 0; j < Q; j++) { b[j] = 0.12 * Math.sin(ph) * (0.6 + 0.4 * Math.sin(ph / 37)); ph += 2 * Math.PI * f0 / SR; } bl.push(b); } return bl; }
const ms = x => Math.round(x * SR / 1000 / Q);
const rms = b => Math.sqrt(b.reduce((a, v) => a + v * v, 0) / b.length);
const loudIdx = o => o.map((b, i) => rms(b) > 0.009 ? i : -1).filter(i => i >= 0);

// ---- 1: speech during the hold is replayed intact after the release; the backlog's silence is dropped; no lasting delay
{ const G = makeGate();
  const pre = run(G, noise(ms(500)));
  check('before the hold: the mic passes untouched (no delay)', pre.every(b => rms(b) < 0.01) && G.node.q.length === 0);
  G.msg({shield: true});
  const sp = speech(ms(900), 220);
  const held = run(G, noise(ms(400)).concat(sp, noise(ms(300))));
  check('during the hold: nothing reaches the vendor (silence out)', held.every(b => rms(b) === 0), held.filter(b => rms(b) > 0).length);
  G.msg({shield: false});
  const post = run(G, noise(ms(4000)));
  const li = loudIdx(post);
  check('after the release: his speech comes back, all of it (no loss)', li.length >= sp.length - 2 && li.length <= sp.length + 2, {got: li.length, want: sp.length});
  check('…contiguous and starting at once (the leading silence of the backlog is dropped)', li[0] <= 2 && li[li.length - 1] - li[0] + 1 === li.length, {first: li[0], span: li[li.length - 1] - li[0] + 1, n: li.length});
  const first = post.slice(li[0], li[0] + sp.length).flatMap(b => Array.from(b)), want = sp.flatMap(b => Array.from(b));
  let maxd = 0; for (let i = 0; i < want.length; i++) maxd = Math.max(maxd, Math.abs(first[i] - want[i]));
  check('…sample-exact, in order', maxd < 1e-6, maxd);
  check('the delay does not stay: the queue is empty again and the output is live', G.node.q.length === 0 && !G.node.sh, G.node.q.length);
  const live = noise(1)[0]; live[5] = 0.3; const o = new Float32Array(Q); G.node.process([[live], [new Float32Array(Q)]], [[o]]);
  check('live again: a quantum goes straight through', o[5] === 0.3 * G.node.g || Math.abs(o[5] - 0.3) < 1e-3, o[5]);
  check('stats: held / replayed / skipped counted', G.node.st.held === held.length && G.node.st.replayed > 0 && G.node.st.skipped > 0, G.node.st);
}
// ---- 2: nothing said during the hold: the replay is silent and the queue empties at once
{ const G = makeGate(); run(G, noise(10)); G.msg({shield: true}); run(G, noise(ms(1500))); G.msg({shield: false});
  const post = run(G, noise(5)); check('nothing said: the backlog is dropped at the release, no delay', G.node.q.length === 0, G.node.q.length); }
// ---- 3: speech still going on at the release: the delay is the rest of the hold, then it catches up through his pauses
{ const G = makeGate(); G.msg({shield: true}); const a = speech(ms(1200), 180); run(G, a.slice(0, ms(700))); G.msg({shield: false});
  const rest = a.slice(ms(700)).concat(noise(ms(800)), speech(ms(500), 260), noise(ms(1500)));
  const post = run(G, rest);
  check('speech ongoing at the release: all of it is replayed (700 ms held + 500 ms live)', loudIdx(post).length >= ms(1200) - 3, loudIdx(post).length);
  check('…and the second utterance after a pause is replayed too', loudIdx(post).filter(i => i > ms(1300)).length >= ms(500) - 3, loudIdx(post).filter(i => i > ms(1300)).length);
  check('…and the line is live again at the end', G.node.q.length === 0, G.node.q.length); }
// ---- 4: tap-to-interrupt: nothing held, nothing replayed
{ const G = makeGate(); G.msg({shield: true}); run(G, speech(ms(600), 200)); G.msg({openMs: 2500});
  check('openMs (tap-to-interrupt) drops the hold and the queue', G.node.sh === false && G.node.q.length === 0); }
// ---- 5: memory bound
{ const G = makeGate(); G.msg({shield: true}); run(G, noise(G.node.MAXQ + 500)); check('the queue is bounded (15 s)', G.node.q.length <= G.node.MAXQ, G.node.q.length); }
// ---- 6: the sentence boundary: released at the avatar's pause once his words have waited 2 s (or at a long pause), not before, not for silence
// (the gate's own rule: no barge-in is passed in the first second of a reply while it calibrates - his words come later than that)
{ const av = (n) => speech(n, 140).map(b => b.map(v => v * 0.8));           // the avatar's voice as it leaves for the speaker
  const quiet = n => Array.from({length: n}, () => new Float32Array(Q));
  const at = (out, thr) => { const i = out.findIndex(b => rms(b) > (thr || 0.009)); return Math.round(i * Q / SR * 1000); };
  // avatar: 3.5 s sentence, 400 ms pause, 1.5 s sentence. He speaks 1.2 .. 1.7 s (waits 2.3 s at the pause)
  const refs = av(ms(3500)).concat(quiet(ms(400)), av(ms(1500)), quiet(ms(500)));
  const mic = noise(ms(1200)).concat(speech(ms(500), 230), noise(ms(4500)));
  const G = makeGate(); G.msg({shield: true});
  const out = run(G, mic.slice(0, refs.length), refs);
  const rel = G.posts.find(m => m.t === 'shield_released');
  check('his words waited 2 s: the hold is released at the avatar\'s pause (told to the page)', !!rel && rel.why === 'pause', G.posts.filter(m => m.t));
  const a1 = at(out);
  check('…his held words come out right after (3.5 s + ~300 ms, not at the end of the 5.4 s chain)', a1 >= 3750 && a1 <= 4000, {a1});
  check('…and not before the sentence ended (silence out until then)', out.slice(0, ms(3500)).every(b => rms(b) === 0));
  // a short overlap (he waited 1.3 s): an ordinary 400 ms pause does not release; the audio end does
  const refsB = av(ms(2500)).concat(quiet(ms(400)), av(ms(1500)), quiet(ms(500)));
  const GB = makeGate(); GB.msg({shield: true}); const outB = run(GB, noise(ms(1200)).concat(speech(ms(500), 230), noise(ms(4500))).slice(0, refsB.length), refsB);
  check('a short overlap (waited 1.3 s): the 400 ms pause after the first sentence does not release the hold (that sentence is finished first)', outB.slice(0, ms(4300)).every(b => rms(b) === 0));
  // …but a long pause (750 ms) is a real boundary at once
  const refsC = av(ms(2500)).concat(quiet(ms(750)), av(ms(1500)), quiet(ms(500)));
  const GC = makeGate(); GC.msg({shield: true}); const outC = run(GC, noise(ms(1200)).concat(speech(ms(500), 230), noise(ms(4500))).slice(0, refsC.length), refsC);
  const aC = at(outC);
  check('a long pause (700 ms+) releases at once', GC.posts.some(m => m.t === 'shield_released') && aC >= 3150 && aC <= 3350, {aC});
  // nothing said: no early release at the pause
  const G2 = makeGate(); G2.msg({shield: true}); run(G2, noise(refs.length), refs);
  check('he said nothing: the pause does not release the hold (the audio end does)', !G2.posts.some(m => m.t === 'shield_released') && G2.node.sh === true);
  // a 120 ms comma pause inside a sentence is not a boundary even after 2 s
  const refs3 = av(ms(1800)).concat(quiet(ms(120)), av(ms(1800)), quiet(ms(600)));
  const G3 = makeGate(); G3.msg({shield: true});
  const o3 = run(G3, noise(ms(1100)).concat(speech(ms(400), 200), noise(refs3.length)).slice(0, refs3.length), refs3);
  const f3 = at(o3);
  check('a comma pause (120 ms) is not a sentence boundary: held through it, released at the real pause after', G3.posts.some(m => m.t === 'shield_released') && f3 >= 3700, {f3});
}
console.log(`gate_hold_test: ${passes} passed, ${fails} failed`); process.exit(fails ? 1 : 0);
