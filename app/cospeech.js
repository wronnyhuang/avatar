/* cospeech.js — co-speech gestures from the avatar's own reply (muse only, see page.html).
 *
 * The reply text streams in (transcript deltas, usually ahead of the audio); the audio passes the page's playout tap in
 * 20 ms chunks on the PLAYBACK timeline (the timeline the rig vector is stamped with — a frame stamped t is shown when
 * the audio at t is heard, so a gesture apex placed at the tap time of a syllable lands on what he HEARS; the DelayNode
 * is the renderer's round trip, not slack for us). We:
 *   1. split the text into words → syllables (+ lexical stress, function/content words, expected pauses at punctuation),
 *   2. track where the voice is in that syllable stream (voiced time × an adaptive syllable rate; re-anchored at every
 *      pause to the nearest punctuation / word boundary; the rate re-fit at each anchor),
 *   3. plan gestures per intonational phrase (one main stroke + occasional beats; rests; triggers: counting, deixis,
 *      iconic size/shape/direction, shrug, negation, open palms, exclamation, question offer, affirm nod, beats on the
 *      nuclear-stress word) with variety (no identical back-to-back, alternating hands, amplitude jitter),
 *   4. commit each gesture JUST IN TIME (≈ its preparation start) through motion.playGesture(def, {src:'cospeech', at,
 *      side, intensity}) so its stroke apex lands 0–150 ms before the stressed syllable's nucleus as estimated then;
 *      never on a busy / holding paw (motion.armsBusy(), state().holding — only holdSafe defs there), never while the
 *      voice is not flowing, and cancel() drops everything not yet performed (barge-in / cancelled / truncated reply).
 *
 * c2 (tester verdict: text arriving early / at 2× made strokes early): the word clock is AUDIO-driven — two readings of
 * the tapped audio (the pause-anchored voiced-time clock and a particle-filter forced alignment on voicing, syllable
 * nuclei and pauses; createDual) whose mean times each syllable and whose spread + disagreement is the confidence. A
 * stroke commits only when that is tight (else it waits, then is skipped, with a plain beat on a later content word as
 * the fallback), re-aims its apex at the newest estimate until its rise begins, and is aborted if still unsure then.
 * Text arrival times never enter the clock. `aligner: 'legacy'` = the c1 tracker, 'pf' = the filter alone.
 *
 *   const C = AvatarCospeech.createCospeech({motion, seed, rate, onCommit, aligner})
 *   C.text(delta, {response_id, item_id, final})   C.audio(t0, rms, silent)   C.tick(tNow)   C.cancel(why)
 *   C.plan() → {words:[{w, i, t_est, stress_t}], gestures:[{type, side, word, apex_t, committed, at, id, dropped}]}
 *   C.covering(tNow) → true while a planned utterance is playing (the page then turns motion's own prosody beats off)
 * Works in node (module.exports) for the offline alignment test (tests/speech_cospeech_align.cjs).
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarCospeech = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));
  const bump = (u, c, w) => { const x = (u - c) / w; return Math.abs(x) >= 1 ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * x); };   // raised cosine
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  // ------------------------------------------------------------------ words, syllables, stress
  const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
  const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
  function numWords(n) {
    n = Math.floor(n);
    if (n < 20) return ONES[n];
    if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ' ' + ONES[n % 10] : '');
    if (n < 1000) return ONES[Math.floor(n / 100)] + ' hundred' + (n % 100 ? ' ' + numWords(n % 100) : '');
    if (n < 10000) return numWords(Math.floor(n / 1000)) + ' thousand' + (n % 1000 ? ' ' + numWords(n % 1000) : '');
    return 'a lot';
  }
  const SYL_EXC = {every: 2, everyone: 3, everybody: 4, everything: 3, different: 3, family: 3, fire: 1, hour: 1, our: 1, really: 2, idea: 3,
    area: 3, people: 2, business: 2, orange: 2, beautiful: 3, being: 2, going: 2, doing: 2, seeing: 2, create: 2, science: 2, quiet: 2,
    lion: 2, poem: 2, piano: 3, video: 3, radio: 3, museum: 3, usually: 4, actually: 4, probably: 3, interesting: 3, chocolate: 3,
    camera: 3, favorite: 3, evening: 2, anything: 3, something: 2, sometimes: 2, maybe: 2, okay: 2, ok: 2, hmm: 1, hm: 1, wow: 1, uh: 1,
    um: 1, oh: 1, ah: 1, tv: 2, one: 1, once: 1, someone: 2, anyone: 3, cereal: 3, real: 1, ideas: 3, whatever: 3, whenever: 3,
    forever: 3, eyes: 1, fly: 1, why: 1, by: 1, my: 1, sky: 1, day: 1, play: 1, they: 1, the: 1, cool: 1, through: 1, sure: 1, yes: 1,
    hello: 2, hi: 1, cookie: 2, movie: 2, prairie: 2, toy: 1, boy: 1, hey: 1, whoa: 1, yay: 1, huge: 1, ice: 1, nice: 1, little: 2,
    giant: 2, tiny: 2, quite: 1, while: 1, smile: 1, purple: 2, more: 1, score: 1, sure_: 1, squirrel: 2, world: 1, girl: 1, oil: 1,
    naive: 2, coordinate: 4, cooperate: 4, recreate: 3, react: 2, reality: 4, area_: 3, being_: 2, iron: 2, flour: 1, flower: 2, tower: 2,
    power: 2, every_: 2, mostly: 2, lovely: 2, likely: 2, safely: 2, nicely: 2, lately: 2, extremely: 3, completely: 3, absolutely: 4};
  function sylCount(w) {
    w = String(w).toLowerCase().replace(/[^a-z']/g, '');
    if (!w) return 0;
    if (SYL_EXC[w]) return SYL_EXC[w];
    let extra = 0;
    if (/n't$/.test(w)) { const base = w.slice(0, -3); if (/[^aeiouy]$/.test(base) && !/^(ca|wo|do|ai)$/.test(base)) extra = 1; w = base; }
    w = w.replace(/'(s|re|ll|ve|d|m)$/, '').replace(/'/g, '');
    if (SYL_EXC[w]) return SYL_EXC[w] + extra;
    if (w.length <= 3) return 1 + extra;
    let s = w;
    if (/[^aeiouy]le$/.test(s)) s = s.slice(0, -2) + 'el';                     // ta-ble, lit-tle
    else if (/[^aeiouy]e$/.test(s) && !/[aeiouy][^aeiouy]e$/.test(s.slice(0, -1) + 'x') ) s = s.slice(0, -1);
    else if (/e$/.test(s) && !/(ee|ie|ye|oe)$/.test(s)) s = s.slice(0, -1);
    if (/[^tdaeiouy]ed$/.test(s)) s = s.slice(0, -2);                          // jumped (1), wanted (2)
    if (/[^aeiouysxzh]es$/.test(s) && !/(ces|ges|ses|zes)$/.test(w)) s = s.slice(0, -2) + 's';   // makes (1), boxes (2)
    s = s.replace(/^y/, '');
    const g = s.match(/[aeiouy]+/g) || [];
    let n = g.length;
    for (const x of g) if (/^(ia|io|eo|ua|uo|ii|iu)$/.test(x) && !/(tion|sion|cious|tious|cial|tial|cian|geo)/.test(w)) n++;   // gi-ant, vi-o-lin
    if (/[aeiou]y[aeiou]/.test(s)) n++;
    return Math.max(1, n) + extra;
  }
  const PREFIX2 = /^(a|be|de|re|pre|ex|in|un|en|em|com|con|to|for|mis|ad|ob|sub|sup|sur|per|pro|dis|im|il|ir|al)(?=[a-z]{3,})/;
  const NOT_PREFIX = /^(every|any|some|after|under|over|happy|pretty|really|little|very|ever|even|never|other|only|open|ocean|often|enter|into|always|also|answer|animal|apple|area|army|art|able|actual|absolute|average|awesome|awful|dinner|doctor|perfect|prob|product|program|problem|present|record|pretty|comfort|common|compan|concert|conver|cons|interest|instant|incred|infor|impor|imag|enem|entr|en$|ex$|expert|excellent|exerc)/;
  function stressIdx(w, n) {
    w = String(w).toLowerCase().replace(/[^a-z]/g, '');
    if (n <= 1) return 0;
    if (/(tion|sion|cian|cious|tious|cial|tial|ic|ics|ian|ient|ience|ual)s?$/.test(w)) return Math.max(0, n - 2);
    if (/(ity|ical|graphy|ogy|ology|ify|ible|able)$/.test(w) && n >= 3) return Math.max(0, n - 3);
    if (/(ee|eer|ese|ique|oo|oon)$/.test(w)) return n - 1;
    if (PREFIX2.test(w) && !NOT_PREFIX.test(w)) return 1;
    if (/^(to|be|a)(day|night|morrow|gether|bout|round|way|side|cause|fore|hind|neath|tween|yond|lone|long|gain|ware|go|head|live|like|maz|mus|ttack|ttent|ward)/.test(w)) return 1;
    return 0;
  }
  const FUNC = new Set(('a an the and or but so to of in on at by for with from up as is am are was were be been being it its it\'s this that these those ' +
    'i me my mine you your yours he him his she her we us our they them their do does did have has had will would can could shall should may might ' +
    'must just if then than when what who how which there here not no oh um uh well like very really too also all some any each every').split(' '));
  const PAUSE = {',': 0.12, ';': 0.2, ':': 0.2, '.': 0.3, '!': 0.3, '?': 0.3, '—': 0.18, '–': 0.15, '-': 0.0, '…': 0.4, '...': 0.4};   // TTS-fit
  // tokenize one chunk of reply text → words with punctuation that follows them
  function tokenize(text) {
    const out = [], re = /([A-Za-z0-9'’]+(?:[-.][A-Za-z0-9'’]+)*)([^A-Za-z0-9'’]*)/g;
    let m;
    while ((m = re.exec(text))) {
      let w = m[1].replace(/’/g, "'"), tail = m[2] || '';
      if (/^\d+(\.\d+)?$/.test(w)) {                                 // numbers → their spoken words (their own syllables)
        const ws = numWords(+w).split(' ');
        ws.forEach((x, i) => out.push(mkWord(x, i === ws.length - 1 ? tail : '', w)));
        continue;
      }
      out.push(mkWord(w, tail));
    }
    return out;
  }
  function mkWord(w, tail, raw) {
    const low = w.toLowerCase(), n = sylCount(low);
    const punct = (tail.match(/\.\.\.|…|[,.;:!?—–]/g) || []);
    let pause = 0; for (const p of punct) pause = Math.max(pause, PAUSE[p] || 0);
    return {w, low, raw: raw || w, syl: n, stress: stressIdx(low, n), func: FUNC.has(low), punct: punct.join(''), pause,
      excl: /!/.test(tail), quest: /\?/.test(tail), caps: w.length > 1 && w === w.toUpperCase() && /[A-Z]/.test(w)};
  }

  // ------------------------------------------------------------------ the timing tracker (one reply stream)
  // weights per syllable: function words short, the stressed syllable of a content word long, phrase-final lengthened
  // word duration model, least squares on the 195 TTS words (median error 28 ms vs 77 ms for a constant):
  //   dur ≈ 0.058 + 0.0358·letters − 0.053·(function word) + 0.10·(phrase-final); split over its syllables (stress × 1.3)
  function wordDur(W) { const n = W.low.replace(/[^a-z]/g, '').length || 2; return Math.max(0.07, 0.058 + 0.0358 * n - 0.053 * (W.func ? 1 : 0) + 0.10 * (W.pause > 0 ? 1 : 0)) * (W.emph || 1); }
  // emphasis stretches speech: an intensifier ("so", "really", "way") and the word it lifts; shouted words; '!' phrases
  const INTENS = /^(so|really|very|super|such|too|way|totally|absolutely|incredibly|extremely)$/;
  function markEmphasis(ws, prev) {
    for (let i = 0; i < ws.length; i++) {
      const W = ws[i], P = i ? ws[i - 1] : prev;
      if (INTENS.test(W.low)) W.emph = 1.3;
      if (P && INTENS.test(P.low) && !W.func) W.emph = 1.45;
      if (W.caps) W.emph = Math.max(W.emph || 1, 1.4);
    }
  }
  function sylWeights(W) {
    const k = []; for (let i = 0; i < W.syl; i++) k.push(!W.func && i === W.stress && W.syl > 1 ? 1.3 : 1);
    const sum = k.reduce((a, b) => a + b, 0), d = wordDur(W);
    return k.map(x => d * x / sum);
  }
  function createTiming(o = {}) {
    const R0 = o.rate || 0.86;                      // model-seconds per second of speech (pauses >= 100 ms out); TTS-set median 0.86 (0.74-0.99)
    const T = {words: [], U: [0], wordU: [], bounds: [], rate: R0, P: 0, vt: 0, anchorP: 0, anchorVt: 0, onset: null, tNow: 0,
      silentRun: 0, lastVoiced: -1, peak: 0.02, env: 0, voicedRuns: 0, ended: false, hist: [], pauseK: 1,
      nz: {y: 0, pz: 0, sm: [0, 0, 0], pk: 1e-4, trough: 1e9, last: -9, n: 0, vt: 0, on: false},
      RG: o.rateGain != null ? o.rateGain : 0.6, NZ: !!o.nuclei, fwdReach: o.legacy ? Infinity : 0.15};
    function addWords(ws) {
      markEmphasis(ws, T.words[T.words.length - 1]);
      for (const W of ws) {
        const i = T.words.length; W.i = i;
        const sw = sylWeights(W), u0 = T.U[T.U.length - 1];
        W.u0 = u0; W.sylU = []; let u = u0;
        for (const x of sw) { W.sylU.push([u, x]); u += x; }
        W.u1 = u; T.U.push(u); T.words.push(W);
        // a boundary after this word (where a pause may come): strength = its expected pause
        T.bounds.push({u, pause: W.pause, word: i});
      }
    }
    const totalU = () => T.U[T.U.length - 1];
    // syllable nuclei from the tapped PCM (10 ms pre-emphasized envelope peaks; ratio to true syllables 0.85-1.16 on the
    // TTS set) → the speaker's own syllable rate, which keeps the voiced-time clock honest between pause anchors
    function nuclei(t0, pcm, dur) {
      const Z = T.nz, n = pcm.length, h = n >> 1, sc = pcm instanceof Int16Array ? 1 / 32768 : 1;
      for (let part = 0; part < 2; part++) {
        let e = 0;
        for (let i = part * h; i < (part + 1) * h; i++) { const x = pcm[i] * sc, d = x - 0.95 * Z.pz; Z.pz = x; Z.y = 0.6 * Z.y + 0.4 * d; e += Z.y * Z.y; }
        const v = 0.6 * Z.sm[2] + 0.4 * Math.sqrt(e / h);
        Z.sm = [Z.sm[1], Z.sm[2], v]; Z.pk = Math.max(Z.pk * Math.exp(-dur / 2 / 4), v);
        const [a, b, c] = Z.sm, tt = t0 + (part + 0.5) * dur / 2 - dur / 2;
        Z.trough = Math.min(Z.trough, b);
        if (b > a && b >= c && b > 0.08 * Z.pk && b - Z.trough > 0.2 * b && tt - Z.last > 0.1) { Z.n++; Z.last = tt; Z.trough = b; }
      }
    }
    // one 20 ms chunk at the tap (t0 = playback-timeline seconds of its first sample); pcm optional (Int16 or Float32)
    function audio(t0, rms, silent, dur = 0.02, pcm) {
      T.tNow = t0 + dur;
      if (pcm && pcm.length >= 64 && !silent) { nuclei(t0, pcm, dur); T.nz.on = true; }
      T.peak = Math.max(0.02, Math.max(T.peak * Math.exp(-dur / 3), rms || 0));
      const voiced = !silent && rms > Math.max(0.006, 0.09 * T.peak);
      // hysteresis: inside a pause, one voiced chunk (a breath, a click) does not end it — two in a row do
      if (voiced && T.silentRun >= 0.06 && !T.pv) { T.pv = 1; T.silentRun += dur; T.pvDur = dur; return; }
      if (!voiced && T.pv) T.pv = 0;
      if (voiced) {
        if (T.onset === null) { T.onset = t0; T.anchorVt = 0; T.vt = 0; }
        if (T.pv) { T.silentRun -= T.pvDur; T.pv = 0; T.vt += T.pvDur; }
        if (T.silentRun >= 0.1 && T.onset !== null && T.vt > 0) reanchor(T.silentRun);
        else if (T.silentRun > 0) T.vt += T.silentRun;          // a gap < 100 ms (a stop closure) is still speech
        T.silentRun = 0; T.vt += dur; T.lastVoiced = T.tNow;
        if (T.nz.on && T.NZ) {
          // syllables per voiced second (prior worth 1.2 s of speech) × the text's mean weight per syllable
          const W = T.words; let su = 0, sn = 0; for (const w of W) { su += w.u1 - w.u0; sn += w.syl; }
          const wpS = sn ? su / sn : 1, rs = (T.nz.n + 1.2 * R0 / wpS) / (T.vt + 1.2);
          T.rate = clamp(0.85 * T.rate + 0.15 * rs * wpS, R0 * 0.6, R0 * 1.6);
        }
        T.P = T.anchorP + T.rate * (T.vt - T.anchorVt);
      } else if (T.onset !== null) T.silentRun += dur;
      if (T.onset !== null && T.hist.length < 4000) T.hist.push([T.tNow, T.P]);
    }
    // a pause just ended: snap to the nearest plausible boundary (punctuation strongly, any word end weakly), re-fit the rate
    function reanchor(L) {
      const P = T.P, span = Math.max(0.5, P - T.anchorP);
      // a punctuation boundary in reach wins (nearest); otherwise the nearest plain word end in a tighter reach
      let best = null, bestD = Infinity, bestW = null, bestWD = Infinity;
      for (const b of T.bounds) {
        if (b.u <= T.anchorP + 0.05) continue;
        const d = Math.abs(b.u - P);
        // c2: a short pause (a hesitation, a stop closure) never jumps the clock far AHEAD to a comma it has not reached
        if (b.u > P + T.fwdReach && L < 0.2) continue;
        if (b.pause > 0) { if (d <= Math.max(0.3, 0.32 * span) && d < bestD) { bestD = d; best = b; } }
        else if (d <= (L > 0.22 ? Math.max(0.16, 0.15 * span) : 0.12) && d < bestWD) { bestWD = d; bestW = b; }
      }
      best = best || bestW;
      if (!best) return;
      const vtSpan = T.vt - T.anchorVt;
      if (vtSpan > 0.35) {
        const obs = (best.u - T.anchorP) / vtSpan;
        T.rate = clamp((1 - T.RG) * T.rate + T.RG * obs, R0 * 0.6, R0 * 1.6);
      }
      if (best.pause > 0) T.pauseK = clamp(0.5 * T.pauseK + 0.5 * L / best.pause, 0.6, 3);   // this speaker's pause length
      T.anchorP = best.u; T.anchorVt = T.vt; T.P = best.u;
      best.used = true;
    }
    // predicted tap time of weighted position X (future); null if the voice has not started
    function timeOf(X) {
      if (T.onset === null) return null;
      if (X <= T.P) { for (let i = T.hist.length - 1; i >= 0; i--) if (T.hist[i][1] <= X) return T.hist[i][0]; return T.onset; }
      let dt = (X - T.P) / T.rate;
      for (const b of T.bounds) if (b.u > T.P + 0.05 && b.u < X - 0.05) dt += b.pause * T.pauseK;
      // inside a pause now: the voice resumes after what is left of the expected pause there
      if (T.silentRun > 0) {
        let here = 0; for (const b of T.bounds) if (Math.abs(b.u - T.P) < 0.6) here = Math.max(here, b.pause);
        dt += Math.max(0.03, here * T.pauseK - T.silentRun);
      }
      return T.tNow + dt;
    }
    // the stressed syllable's nucleus of word i (weighted position) and its predicted time
    function stressU(i) { const W = T.words[i]; if (!W) return null; const s = W.sylU[Math.min(W.stress, W.sylU.length - 1)]; return s[0] + 0.42 * s[1]; }
    return {T, addWords, audio, timeOf, stressU, totalU, playing: tNow => T.onset !== null && (tNow - T.lastVoiced < 0.45 || (T.silentRun < 0.9 && T.P < totalU() - 0.3))};
  }

  // ------------------------------------------------------------------ c2: the AUDIO-DRIVEN aligner (default)
  // An online forced alignment of the reply's syllable stream to the tapped audio, as a particle filter over
  // (position u in the text's model-time, speaking rate r, speaking|pausing). Per 20 ms tap chunk:
  //   predict — a speaking particle advances u by r·dt; crossing a word boundary it may enter a pause (likely at
  //             punctuation, rare between plain words), whose length is drawn around the expected pause;
  //   weigh   — voicing (speaking ↔ voiced, pausing ↔ silent), syllable nuclei detected in the PCM (a nucleus lands
  //             near a syllable centre of the text), the end of the known text (no voice past a final reply's end);
  //   resample when the effective sample size drops.
  // The text only gives ORDER (the syllable sequence) and an upper bound (no gesture is planned on a word before it has
  // arrived; a particle past the known text gets no syllable evidence); its arrival TIMES are never used as time.
  // timeOf(X) = the weighted median of each particle's time to reach X (rest of its pause + (X − u)/r + the expected
  // pauses in between); spread(X) = the weighted inter-quartile half-width of the same — the scheduler's confidence.
  function createAligner(o = {}) {
    const R0 = o.rate || 0.86, N = o.particles || 192;
    const K = Object.assign({plain: 0.03, silSpeak: 0.6, silLong: 0.3, voPause: 0.06, exitP: 0.85, snapP: 0.3, snapReach: 0.25, snapFwd: 0.12, nucFloor: 0.25, nucSig: 0.3, rateSd: 0.24, walk: 0.012, nuc: 1}, (typeof globalThis !== 'undefined' && globalThis.__PF) || {}, o.pf || {});
    const rnd = mulberry32((o.seed || 99) * 2654435761 >>> 0);
    const gauss = () => { let u = 0, v = 0; while (u === 0) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const T = {words: [], U: [0], bounds: [], rate: R0, P: 0, vt: 0, onset: null, tNow: 0, silentRun: 0, lastVoiced: -1, peak: 0.02,
      ended: false, hist: [], pauseK: 1, final: false, spreadNow: 0,
      nz: {y: 0, pz: 0, sm: [0, 0, 0], pk: 1e-4, trough: 1e9, last: -9, n: 0, on: false}};
    const pu = new Float64Array(N), pr = new Float64Array(N), pm = new Uint8Array(N), prem = new Float64Array(N), pw = new Float64Array(N).fill(1 / N);
    const SC = [], SW = [];                              // syllable centres / widths (model-time), in order
    const BU = [], BP = [], BPI = [], CUM = [0];         // boundary u, expected pause, P(pause), cumulative expected pause
    function addWords(ws) {
      markEmphasis(ws, T.words[T.words.length - 1]);
      for (const W of ws) {
        const i = T.words.length; W.i = i;
        const sw = sylWeights(W), u0 = T.U[T.U.length - 1];
        W.u0 = u0; W.sylU = []; let u = u0;
        for (const x of sw) { W.sylU.push([u, x]); SC.push(u + 0.45 * x); SW.push(x); u += x; }
        W.u1 = u; T.U.push(u); T.words.push(W);
        T.bounds.push({u, pause: W.pause, word: i});
        const pi = W.pause >= 0.25 ? 0.9 : W.pause >= 0.15 ? 0.8 : W.pause > 0 ? 0.7 : K.plain;
        const ep = W.pause > 0 ? W.pause : 0.12;
        BU.push(u); BP.push(ep); BPI.push(pi); CUM.push(CUM[CUM.length - 1] + pi * ep);
      }
    }
    const totalU = () => T.U[T.U.length - 1];
    // first boundary index with BU > u
    function bAfter(u) { let a = 0, b = BU.length; while (a < b) { const m = (a + b) >> 1; if (BU[m] > u + 1e-9) b = m; else a = m + 1; } return a; }
    function nearestSyl(u) { let a = 0, b = SC.length; while (a < b) { const m = (a + b) >> 1; if (SC[m] < u) a = m + 1; else b = m; } let best = -1, bd = Infinity; for (const k of [a - 1, a]) if (k >= 0 && k < SC.length) { const d = Math.abs(SC[k] - u); if (d < bd) { bd = d; best = k; } } return best; }
    function init(t0) {
      for (let k = 0; k < N; k++) { pu[k] = 0.02 * rnd(); pr[k] = R0 * Math.exp(K.rateSd * gauss()); pm[k] = 0; prem[k] = 0; pw[k] = 1 / N; }
    }
    // syllable nuclei from the tapped PCM: 10 ms pre-emphasized envelope peaks (ratio to true syllables 0.85-1.16 on the TTS set)
    function nuclei(t0, pcm, dur) {
      const Z = T.nz, n = pcm.length, h = n >> 1, sc = pcm instanceof Int16Array ? 1 / 32768 : 1; let hits = 0;
      for (let part = 0; part < 2; part++) {
        let e = 0;
        for (let i = part * h; i < (part + 1) * h; i++) { const x = pcm[i] * sc, d = x - 0.95 * Z.pz; Z.pz = x; Z.y = 0.6 * Z.y + 0.4 * d; e += Z.y * Z.y; }
        const v = 0.6 * Z.sm[2] + 0.4 * Math.sqrt(e / h);
        Z.sm = [Z.sm[1], Z.sm[2], v]; Z.pk = Math.max(Z.pk * Math.exp(-dur / 2 / 4), v);
        const [a, b, c] = Z.sm, tt = t0 + (part + 0.5) * dur / 2 - dur / 2;
        Z.trough = Math.min(Z.trough, b);
        if (b > a && b >= c && b > 0.08 * Z.pk && b - Z.trough > 0.2 * b && tt - Z.last > 0.1) { Z.n++; Z.last = tt; Z.trough = b; hits++; }
      }
      return hits;
    }
    function step(dt, voiced, nuc) {
      const tot = totalU(), haveText = SC.length > 0;
      let wsum = 0;
      for (let k = 0; k < N; k++) {
        // predict
        if (pm[k] === 1) {
          prem[k] -= dt;
          if (prem[k] <= 0 || (voiced && rnd() < K.exitP)) pm[k] = 0;          // the voice is back: most pausing particles resume now
        } else {
          const u1 = pu[k] + pr[k] * dt;
          const bi = bAfter(pu[k]);
          if (bi < BU.length && BU[bi] <= u1 && rnd() < BPI[bi]) {
            pu[k] = BU[bi]; pm[k] = 1;
            const ep = BP[bi] * T.pauseK; prem[k] = Math.max(0.05, ep * Math.exp(0.5 * gauss()) - (u1 - BU[bi]) / pr[k]);
          } else pu[k] = u1;
          // a long silence where this particle has no boundary: re-anchor it at the nearest boundary in reach (a pause there)
          if (!voiced && T.silentRun >= 0.08 && pm[k] === 0 && BU.length && rnd() < K.snapP) {
            const j = bAfter(pu[k]), cand = [j - 1, j].filter(x => x >= 0 && x < BU.length);
            let bj = -1, bd = Infinity; for (const x of cand) { const d = Math.abs(BU[x] - pu[k]) / (BPI[x] > 0.5 ? 1.6 : 1); if (d < bd) { bd = d; bj = x; } }
            if (bj >= 0 && Math.abs(BU[bj] - pu[k]) < (BU[bj] > pu[k] && T.silentRun < 0.2 ? K.snapFwd : K.snapReach)) { pu[k] = BU[bj]; pm[k] = 1; prem[k] = Math.max(0.04, (BP[bj] || 0.12) * T.pauseK * Math.exp(0.4 * gauss()) - T.silentRun); }
          }
        }
        pr[k] = clamp(pr[k] * Math.exp(K.walk * gauss()), R0 * 0.45, R0 * 1.9);
        // weigh
        let w = pm[k] === 1 ? (voiced ? K.voPause : 0.97) : (voiced ? 0.9 : (T.silentRun >= 0.1 ? K.silLong : K.silSpeak));
        if (pm[k] === 0 && T.final && pu[k] > tot + 0.08) w *= voiced ? 0.3 : 1;
        if (nuc && haveText && K.nuc) {
          if (pm[k] === 1) w *= 0.35;
          else if (pu[k] <= tot + 0.05) { const s = nearestSyl(pu[k]); const d = Math.abs(SC[s] - pu[k]) / Math.max(0.05, SW[s]); w *= K.nucFloor + Math.exp(-0.5 * (d / K.nucSig) * (d / K.nucSig)); }
        }
        pw[k] *= w; wsum += pw[k];
      }
      if (!(wsum > 0)) { for (let k = 0; k < N; k++) pw[k] = 1 / N; wsum = 1; }
      let ess = 0; for (let k = 0; k < N; k++) { pw[k] /= wsum; ess += pw[k] * pw[k]; }
      if (1 / ess < N * 0.5) resample();
    }
    function resample() {
      const cu = new Float64Array(N), cr = new Float64Array(N), cm = new Uint8Array(N), cp = new Float64Array(N);
      let c = pw[0], i = 0; const u0 = rnd() / N;
      for (let k = 0; k < N; k++) { const x = u0 + k / N; while (x > c && i < N - 1) { i++; c += pw[i]; } cu[k] = pu[i]; cr[k] = pr[i]; cm[k] = pm[i]; cp[k] = prem[i]; }
      pu.set(cu); pr.set(cr); pm.set(cm); prem.set(cp); pw.fill(1 / N);
    }
    function estimate() {
      let su = 0, sr = 0; for (let k = 0; k < N; k++) { su += pw[k] * pu[k]; sr += pw[k] * pr[k]; }
      T.P = su; T.rate = sr;
    }
    function audio(t0, rms, silent, dur = 0.02, pcm) {
      T.tNow = t0 + dur;
      const nuc = pcm && pcm.length >= 64 && !silent ? nuclei(t0, pcm, dur) : 0;
      T.peak = Math.max(0.02, Math.max(T.peak * Math.exp(-dur / 3), rms || 0));
      const voiced = !silent && rms > Math.max(0.006, 0.09 * T.peak);
      if (T.onset === null) { if (!voiced) return; T.onset = t0; init(t0); }
      if (voiced) {
        if (T.silentRun >= 0.1) {        // a pause ended: learn this speaker's pause length at the boundary the particles sit on
          const bi = bAfter(T.P - 0.02) - 1, b = bi >= 0 && Math.abs(BU[bi] - T.P) < 0.25 ? bi : -1;
          if (b >= 0 && BP[b] > 0 && BPI[b] > 0.5) T.pauseK = clamp(0.6 * T.pauseK + 0.4 * T.silentRun / BP[b], 0.6, 2);
        }
        if (T.silentRun > 0 && T.silentRun < 0.1) T.vt += T.silentRun;
        T.silentRun = 0; T.vt += dur; T.lastVoiced = T.tNow;
      } else T.silentRun += dur;
      step(dur, voiced, nuc);
      estimate();
      if (T.hist.length < 6000) T.hist.push([T.tNow, T.P]);
    }
    // per-particle time to reach X (s from now)
    function ttr(k, X) {
      if (X <= pu[k]) return 0;
      const a = bAfter(pu[k]), b = bAfter(X - 0.03);     // boundaries strictly between
      let pz = b > a ? CUM[b] - CUM[a] : 0;
      return (pm[k] === 1 ? Math.max(0, prem[k]) : 0) + (X - pu[k]) / pr[k] + pz * T.pauseK;
    }
    const tmpT = new Float64Array(N), idx = new Int32Array(N);
    function quant(X) {
      for (let k = 0; k < N; k++) { tmpT[k] = ttr(k, X); idx[k] = k; }
      const ia = Array.from(idx).sort((p, q) => tmpT[p] - tmpT[q]);
      let c = 0, q25 = null, q50 = null, q75 = null;
      for (const k of ia) { c += pw[k]; if (q25 === null && c >= 0.25) q25 = tmpT[k]; if (q50 === null && c >= 0.5) q50 = tmpT[k]; if (q75 === null && c >= 0.75) { q75 = tmpT[k]; break; } }
      return [q25, q50, q75 ?? q50];
    }
    function timeOf(X) {
      if (T.onset === null) return null;
      if (X <= T.P - 0.02) { for (let i = T.hist.length - 1; i >= 0; i--) if (T.hist[i][1] <= X) return T.hist[i][0]; return T.onset; }
      return T.tNow + quant(X)[1];
    }
    // the half-width of the middle 50% of predicted times for X (s): how sure the alignment is
    function spread(X) { if (T.onset === null) return Infinity; const q = quant(X); return (q[2] - q[0]) / 2; }
    function stressU(i) { const W = T.words[i]; if (!W) return null; const s = W.sylU[Math.min(W.stress, W.sylU.length - 1)]; return s[0] + 0.45 * s[1]; }
    function markFinal() { T.final = true; }
    let pausingFrac = () => { let s = 0; for (let k = 0; k < N; k++) if (pm[k] === 1) s += pw[k]; return s; };
    return {T, addWords, audio, timeOf, spread, stressU, totalU, markFinal, pausing: pausingFrac, kind: 'audio',
      playing: tNow => T.onset !== null && (tNow - T.lastVoiced < 0.45 || (T.silentRun < 0.9 && T.P < totalU() - 0.3))};
  }

  // c2: the two audio-driven estimators side by side — the voiced-time clock re-anchored at pauses (createTiming) and
  // the particle-filter alignment (createAligner). The time of a syllable is their mean; the confidence is the filter's
  // own spread AND their agreement (two different readings of the same audio that disagree mean nobody knows).
  function createDual(o = {}) {
    const L = createTiming(o), A = createAligner(o);
    const AGREE = o.agreeK || 1.3;
    function both(X) { const a = A.timeOf(X), l = L.timeOf(X); return a == null || l == null ? null : [a, l]; }
    return {T: L.T, A, L, kind: 'dual', totalU: L.totalU, stressU: L.stressU, playing: L.playing,
      addWords(ws) { L.addWords(ws); A.addWords(ws); },
      audio(t0, rms, silent, dur, pcm) { L.audio(t0, rms, silent, dur, pcm); A.audio(t0, rms, silent, dur, pcm); },
      markFinal() { A.markFinal(); },
      timeOf(X) { if (X <= L.T.P) return L.timeOf(X); const b = both(X); return b ? (b[0] + b[1]) / 2 : null; },
      spread(X) { if (X <= L.T.P) return 0; const b = both(X); return b ? Math.max(A.spread(X), Math.abs(b[0] - b[1]) * AGREE) : Infinity; }};
  }

  // ------------------------------------------------------------------ the gesture vocabulary (motion.playGesture defs)
  // hand targets = contract hand_<s>_x/y/z (arm lengths, x outward), open, turn (+1 palm to the camera); s = screen side.
  // Each def: prep (attack) → stroke peaking at `apex` (s from start) → post-stroke hold → retraction (release).
  const HP = (s, x, y, z, open, turn) => { const o = {}; o['hand_' + s + '_x'] = x; o['hand_' + s + '_y'] = y; o['hand_' + s + '_z'] = z; o['hand_' + s + '_open'] = open; o['hand_' + s + '_turn'] = turn; return o; };
  const oth = s => s === 'l' ? 'r' : 'l';
  const yawSign = s => s === 'l' ? -1 : 1;      // a screen-left paw points toward screen-left (head_yaw < 0)
  function mk(name, prep, hold, release, pose, holdSafe, parts) {
    // pose(u, g, k) → {set, add}; k = stroke envelope (0..1 peaking at the apex)
    return {name, attack: prep, release, holdSafe: !!holdSafe, parts: parts || 'arms', apex: prep + 0.06,
      dur: g => (g.hold != null ? g.hold : hold) + prep + 0.06 + release,
      fn(u, g) { const k = bump(u, g.apex != null ? g.apex : prep + 0.06, g.sw || 0.16); return pose(u, g, k, g.amp || 1); }};
  }
  const G = {
    beat: mk('beat', 0.26, 0.28, 0.32, (u, g, k, a) => {
      const s = g.side, palmUp = g.variant !== 'down';
      const set = HP(s, 0.16 + 0.05 * k * a, 0.30 - 0.11 * k * a, 0.30 + 0.03 * k, palmUp ? 0.6 + 0.15 * k : 0.45, palmUp ? 0.5 - 0.1 * k : -0.25);
      set['shoulder_' + s] = 0.06 + 0.06 * k * a;
      return {set, add: {head_pitch: 0.025 * k * a, brow_l: 0.08 * k, brow_r: 0.08 * k}};
    }),
    beat_both: mk('beat_both', 0.28, 0.3, 0.34, (u, g, k, a) => {
      const set = Object.assign(HP('l', 0.2 + 0.04 * k, 0.3 - 0.1 * k * a, 0.3, 0.6, 0.45), HP('r', 0.2 + 0.04 * k, 0.3 - 0.1 * k * a, 0.3, 0.6, 0.45));
      return {set, add: {head_pitch: 0.03 * k * a, shoulder_l: 0.05 * k, shoulder_r: 0.05 * k, brow_l: 0.1 * k, brow_r: 0.1 * k}};
    }, false, 'both'),
    open_palms: mk('open_palms', 0.32, 0.4, 0.38, (u, g, k, a) => {
      const set = Object.assign(HP('l', 0.32 + 0.08 * k * a, 0.30 + 0.03 * k, 0.30, 1, 0.95), HP('r', 0.32 + 0.08 * k * a, 0.30 + 0.03 * k, 0.30, 1, 0.95));
      return {set, add: {shoulder_l: 0.08 + 0.05 * k, shoulder_r: 0.08 + 0.05 * k, brow_l: 0.15 * k, brow_r: 0.15 * k, head_roll: 0.03 * k * (g.dir || 1)}};
    }, false, 'both'),
    self: mk('self', 0.3, 0.32, 0.36, (u, g, k, a) => {
      const s = g.side; const set = HP(s, -0.42, 0.33 + 0.02 * k, 0.26 - 0.05 * k * a, 1, 0.2);
      return {set, add: {head_pitch: 0.03 * k, smile: 0.08 * k}};
    }),
    you: mk('you', 0.3, 0.32, 0.38, (u, g, k, a) => {
      const s = g.side; const set = HP(s, 0.02, 0.42 + 0.02 * k, 0.68 + 0.08 * k * a, 1, 0.85);
      return {set, add: {head_pitch: 0.03 * k, brow_l: 0.15 * k, brow_r: 0.15 * k, smile: 0.06 * k}};
    }),
    point_side: mk('point_side', 0.32, 0.45, 0.4, (u, g, k, a) => {
      const s = g.side, set = HP(s, 0.58 + 0.06 * k * a, 0.58, 0.3, 0.3, 0.3); set['point_' + s] = 1;
      const d = yawSign(s), look = mj(clamp(u / 0.3, 0, 1)) * (1 - mj(clamp((u - 0.75) / 0.35, 0, 1)));
      return {set, add: {head_yaw: 0.12 * d * look, gaze_x: 0.45 * d * look}};
    }),
    present: mk('present', 0.3, 0.38, 0.36, (u, g, k, a) => {     // "this / here": palm up, low and forward
      const s = g.side; return {set: HP(s, 0.14, 0.24 - 0.06 * k * a, 0.55, 1, 0.95), add: {head_pitch: 0.04 * k, gaze_y: -0.2 * k}};
    }),
    big: mk('big', 0.36, 0.42, 0.42, (u, g, k, a) => {
      const set = Object.assign(HP('l', 0.56 + 0.12 * k * a, 0.48, 0.24, 1, 0.55), HP('r', 0.56 + 0.12 * k * a, 0.48, 0.24, 1, 0.55));
      return {set, add: {eye_wide: 0.25 * k, brow_l: 0.3 * k, brow_r: 0.3 * k, shoulder_l: 0.1, shoulder_r: 0.1, body_pitch: -0.08 * k}};
    }, false, 'both'),
    small: mk('small', 0.32, 0.42, 0.38, (u, g, k, a) => {
      const set = Object.assign(HP('l', -0.2 - 0.05 * k * a, 0.42, 0.42, 0.25, 0.3), HP('r', -0.2 - 0.05 * k * a, 0.42, 0.42, 0.25, 0.3));
      return {set, add: {squint: 0.2 * k, head_roll: 0.04 * k, head_pitch: 0.03 * k}};
    }, false, 'both'),
    up: mk('up', 0.34, 0.4, 0.55, (u, g, k, a) => {
      const s = g.side; return {set: HP(s, 0.22, 0.92 + 0.1 * k * a, 0.22, 1, -0.5), add: {head_pitch: -0.05 * k, gaze_y: 0.45 * k, brow_l: 0.15 * k, brow_r: 0.15 * k}};
    }),
    down: mk('down', 0.3, 0.38, 0.36, (u, g, k, a) => {
      const s = g.side; return {set: HP(s, 0.26, 0.08 - 0.1 * k * a, 0.42, 1, -0.85), add: {head_pitch: 0.05 * k, gaze_y: -0.3 * k}};
    }),
    long: Object.assign(mk('long', 0.3, 0.3, 0.4, (u, g, k, a) => {   // both paws pull apart over the stroke
      const s = mj(clamp((u - 0.18) / 0.45, 0, 1));
      const x = 0.02 + 0.55 * s * a;
      return {set: Object.assign(HP('l', x, 0.4, 0.36, 0.8, 0.0), HP('r', x, 0.4, 0.36, 0.8, 0.0)), add: {brow_l: 0.12 * s, brow_r: 0.12 * s}};
    }, false, 'both'), {apex: 0.5}),
    round: Object.assign(mk('round', 0.28, 0.25, 0.38, (u, g, k, a) => {   // the paw traces a circle in front
      const s = g.side, ph = 2 * Math.PI * clamp((u - 0.2) / 0.75, 0, 1), r = 0.11 * a;
      const set = HP(s, 0.16 + r * Math.cos(ph), 0.46 + r * Math.sin(ph), 0.4, 0.55, 0.4); set['point_' + s] = 0.6;
      return {set};
    }), {apex: 0.58}),
    count: mk('count', 0.32, 0.55, 0.42, (u, g, k, a) => {
      const s = g.side, n = clamp(g.n || 1, 1, 5);
      const set = HP(s, 0.14, 0.5 + 0.03 * k, 0.36 + 0.05 * k * a, 1, 0.95);
      set['fingers_' + s] = n * mj(clamp((u - 0.12) / 0.2, 0, 1));
      return {set, add: {brow_l: 0.12 * k, brow_r: 0.12 * k, head_pitch: 0.02 * k}};
    }),
    count_both: mk('count_both', 0.34, 0.6, 0.44, (u, g, k, a) => {     // 6..10: one full hand + the other
      const n = clamp(g.n || 6, 6, 10), e = mj(clamp((u - 0.12) / 0.2, 0, 1));
      const set = Object.assign(HP('l', 0.18, 0.6, 0.36 + 0.04 * k, 1, 0.95), HP('r', 0.18, 0.6, 0.36 + 0.04 * k, 1, 0.95));
      set.fingers_l = 5 * e; set.fingers_r = (n - 5) * e;
      return {set, add: {brow_l: 0.12 * k, brow_r: 0.12 * k}};
    }, false, 'both'),
    shrug: Object.assign(mk('shrug', 0.24, 0.32, 0.42, (u, g, k, a) => {
      const p = mj(clamp(u / 0.26, 0, 1)) * (1 - mj(clamp((u - 0.56) / 0.3, 0, 1)));
      const set = Object.assign({shoulder_l: 0.75 * p * a, shoulder_r: 0.75 * p * a}, HP('l', 0.34, 0.28, 0.2, 1, 1), HP('r', 0.34, 0.28, 0.2, 1, 1));
      return {set, add: {head_roll: 0.08 * p * (g.dir || 1), brow_l: 0.4 * p, brow_r: 0.4 * p, head_pitch: -0.02 * p}};
    }, false, 'both'), {apex: 0.27}),
    shrug_small: Object.assign(mk('shrug_small', 0.22, 0.25, 0.35, (u, g, k, a) => {   // holdSafe: shoulders + face only
      const p = mj(clamp(u / 0.26, 0, 1)) * (1 - mj(clamp((u - 0.5) / 0.3, 0, 1)));
      return {set: {shoulder_l: 0.55 * p * a, shoulder_r: 0.55 * p * a}, add: {head_roll: 0.07 * p * (g.dir || 1), brow_l: 0.35 * p, brow_r: 0.35 * p}};
    }, true, 'shoulders'), {apex: 0.28}),
    negate: mk('negate', 0.24, 0.3, 0.36, (u, g, k, a) => {     // palm-down sweep out + a small head shake
      const s = g.side, sw = mj(clamp((u - 0.12) / 0.3, 0, 1));
      const set = HP(s, 0.02 + 0.5 * sw * a, 0.36 - 0.04 * sw, 0.42, 1, -0.85);
      const sh = Math.sin(2 * Math.PI * 2.6 * clamp(u - 0.1, 0, 0.77)) * (1 - mj(clamp((u - 0.5) / 0.4, 0, 1)));
      return {set, add: {head_yaw: 0.07 * sh * a, brow_knit: 0.15 * k}};
    }),
    head_shake: Object.assign(mk('head_shake', 0.12, 0.45, 0.25, (u, g, k, a) => {
      const sh = Math.sin(2 * Math.PI * 2.6 * clamp(u - 0.05, 0, 0.77)) * (1 - mj(clamp((u - 0.5) / 0.35, 0, 1)));
      return {add: {head_yaw: 0.08 * sh * a, brow_knit: 0.12 * k}};
    }, true, 'head'), {apex: 0.2}),
    exclaim: mk('exclaim', 0.26, 0.32, 0.38, (u, g, k, a) => {
      const set = Object.assign(HP('l', 0.3, 0.58 + 0.08 * k * a, 0.3, 1, 0.85), HP('r', 0.3, 0.58 + 0.08 * k * a, 0.3, 1, 0.85));
      return {set, add: {eye_wide: 0.25 * k, brow_l: 0.4 * k, brow_r: 0.4 * k, smile: 0.2 * k, body_bounce: 0.04 * k}};
    }, false, 'both'),
    offer: mk('offer', 0.3, 0.45, 0.4, (u, g, k, a) => {       // a question: palm up toward him, head tilt
      const s = g.side; return {set: HP(s, 0.12, 0.33, 0.6 + 0.05 * k * a, 1, 1), add: {head_roll: 0.06 * (g.dir || 1) * mj(clamp(u / 0.3, 0, 1)), brow_l: 0.3 * k, brow_r: 0.3 * k}};
    }),
    nod: Object.assign(mk('nod', 0.1, 0.2, 0.25, (u, g, k, a) => ({add: {head_pitch: 0.1 * k * a, head_y: -0.006 * k}}), true, 'head'), {apex: 0.16}),
    brow_beat: Object.assign(mk('brow_beat', 0.08, 0.15, 0.2, (u, g, k, a) => ({add: {brow_l: 0.35 * k * a, brow_r: 0.35 * k * a, head_pitch: 0.035 * k * a}}), true, 'head'), {apex: 0.12}),
  };
  // two phases (McNeill/Kendon): an early READY phase (the paw rises into gesture space and waits = the pre-stroke hold),
  // then the STROKE, committed just in time with a short travel (strokeIn) so its apex needs only a ~0.15-0.25 s forecast
  const STROKE_IN = {beat: 0.12, beat_both: 0.14, open_palms: 0.16, self: 0.18, you: 0.18, point_side: 0.2, present: 0.16, big: 0.22, small: 0.2,
    up: 0.22, down: 0.18, count: 0.18, count_both: 0.2, negate: 0.14, exclaim: 0.16, offer: 0.18};
  for (const k in STROKE_IN) {
    const d = G[k], hold = d.dur({}) - d.attack - 0.06 - d.release;
    d.attack = STROKE_IN[k]; d.apex = STROKE_IN[k] + 0.04;
    d.dur = g => d.apex + (g.hold != null ? g.hold : hold) + d.release;
  }
  G.ready = {name: 'ready', attack: 0.34, release: 0.3, parts: 'arms', apex: 0.34, holdSafe: false, dur: g => g.hold || 1.0,
    fn: (u, g) => ({set: HP(g.side, 0.15, 0.3, 0.3, 0.5, 0.4)})};
  for (const k in G) { G[k].dur0 = G[k].dur({}); }

  // ------------------------------------------------------------------ lexical triggers
  const L = s => new Set(s.split(' '));
  const LEX = {
    big: L('big huge giant enormous massive gigantic wide lots tons everything whole entire vast humongous bigger biggest'),
    small: L('small tiny little bit teeny mini smaller smallest tad wee'),
    up: L('up above high tall sky top rise rising climb climbed taller tallest higher upward'),
    down: L('down below low under bottom fall falling fell drop dropped downward lower underneath'),
    long: L('long far forever ages distance longer farther further'),
    round: L('round around circle ball world spin spinning rolls rolling everywhere globe'),
    there: L('there over yonder'), here: L('this these here'), that: L('that those'),
    self: L('i me my myself mine i\'m i\'ll i\'ve i\'d'), you: L('you your yours yourself you\'re you\'ll you\'ve'),
    neg: L('no not never nothing nobody none nope nah can\'t don\'t won\'t isn\'t didn\'t doesn\'t wasn\'t aren\'t couldn\'t shouldn\'t wouldn\'t neither nor'),
    affirm: L('yes yeah yep exactly right sure absolutely definitely totally okay ok'),
    open: L('well so look see basically actually imagine because example means thing idea explain'),
    excl: L('wow whoa yay awesome amazing incredible fantastic wonderful hooray great love perfect brilliant'),
    ord: {first: 1, second: 2, third: 3, fourth: 4, fifth: 5, firstly: 1, secondly: 2, thirdly: 3, last: 0, finally: 0, next: 0},
    num: {one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, couple: 2, both: 2, pair: 2},
    side: {left: 'left', right: 'right'},
  };
  const SHRUG_RE = /\b(maybe|perhaps|i (don'?t|do not) know|not sure|who knows|i guess|kind of|sort of|dunno|i wonder|hard to say|it depends|whatever)\b/;

  // ------------------------------------------------------------------ the planner + scheduler
  function createCospeech(o = {}) {
    const rnd = mulberry32(o.seed || 1234567);
    const U = (a, b) => a + (b - a) * rnd();
    const motion = o.motion || null;
    const TO = {rate: o.rate, nuclei: o.nuclei, rateGain: o.rateGain, seed: o.seed, particles: o.particles, agreeK: o.agreeK, pf: o.pf};
    const LEGACY = o.aligner === 'legacy';          // the c1 voiced-time × rate tracker (kept for comparison)
    const mkTm = () => LEGACY ? createTiming(Object.assign({legacy: true}, TO)) : o.aligner === 'pf' ? createAligner(TO) : createDual(TO);
    const C = {tm: mkTm(), buf: '', words: [], planned: 0, gestures: [], cancelled: false, last: [], lastApex: -9,
      lastSide: 'r', stats: {committed: 0, dropped: 0, refused: 0, cancelled: 0, late: 0}, respId: null, enabled: o.enabled !== false,
      plans: [], commits: [], textLog: []};
    const nowT = () => C.tm.T.tNow;
    function newStream(why) {
      // a reply spoken to the end teaches this call's voice rate (model-seconds per second of speech) to the next one
      const T0 = C.tm.T;
      if (C.words.length && !C.cancelled && T0.vt > 1.0 && T0.P >= C.tm.totalU() - 0.5) {
        const obs = C.tm.totalU() / T0.vt; if (obs > 0.5 && obs < 1.4) { C.R0 = C.R0 ? 0.5 * C.R0 + 0.5 * obs : obs; TO.rate = C.R0; }
      }
      if (C.words.length) C.plans.push(snapshot());
      if (C.plans.length > 8) C.plans.shift();
      C.tm = mkTm(); C.buf = ''; C.words = []; C.planned = 0; C.gestures = []; C.cancelled = false; C.endT = null;
    }
    // ---- text in
    function text(delta, info = {}) {
      if (!C.enabled) return;
      const rid = info.response_id || info.item_id || null;
      const T = C.tm.T;
      // a new reply: start a new stream when the old one is done (its audio ended) or was cancelled
      if (rid && rid !== C.respId) {
        const oldDone = !C.words.length || C.cancelled || (T.onset !== null && nowT() - T.lastVoiced > 0.6) || T.onset === null && !C.words.length;
        if (oldDone) newStream('response');
        else if (C.words.length) { const lw = C.words[C.words.length - 1]; lw.pause = Math.max(lw.pause, 0.5); C.tm.T.bounds[C.tm.T.bounds.length - 1].pause = lw.pause; }
        C.respId = rid; C.cancelled = false;
      }
      if (C.cancelled) return;
      C.textLog.push([nowT(), String(delta).length]);
      if (info.final) {                       // the full transcript: only words we have not seen yet
        const all = tokenize(String(delta));
        if (all.length > C.words.length) { const add = all.slice(C.words.length); C.tm.addWords(add); C.words.push(...add); }
        C.buf = ''; C.final = true; if (C.tm.markFinal) C.tm.markFinal(); planMore(true); return;
      }
      C.buf += delta;
      // complete words only (the last token may still grow)
      const m = /^([\s\S]*[\s,.;:!?—…])([^\s,.;:!?—…]*)$/.exec(C.buf);
      if (!m) return;
      const ws = tokenize(m[1]); C.buf = m[2];
      if (ws.length) { for (const W of ws) W.arrived = nowT(); C.tm.addWords(ws); C.words.push(...ws); planMore(false); }
    }
    // ---- phrases → gestures (only phrases whose end has arrived, so "three things" / "over there" are seen whole)
    function planMore(final) {
      const W = C.words;
      while (C.planned < W.length) {
        let end = -1;
        for (let i = C.planned; i < W.length; i++) {
          const long = i - C.planned >= 6 && /^(and|but|so|because|then|or|which|when)$/.test((W[i + 1] || {}).low || '');
          if (W[i].pause > 0 || long) { end = i; break; }
        }
        if (end < 0) { if (final || W.length - C.planned > 9) end = W.length - 1; else return; }
        planPhrase(C.planned, end);
        C.planned = end + 1;
      }
    }
    function pickSide(pref) {
      if (pref) return pref;
      // dominant = its own right = screen-left 'l' (~60%), alternate after a same-side run
      const s = rnd() < 0.6 ? 'l' : 'r';
      const l2 = C.last.slice(-2);
      if (l2.length === 2 && l2.every(g => g.side === s)) return oth(s);
      return s;
    }
    function add(type, word, extra) {
      C.gestures.push(Object.assign({type, word, side: null, committed: false, dropped: false, id: null, amp: U(0.85, 1.15)}, extra || {}));
    }
    function planPhrase(a, b) {
      const W = C.words.slice(a, b + 1), txt = W.map(w => w.low).join(' ');
      const excl = W.some(w => w.excl) || W.some(w => w.caps), quest = W[W.length - 1].quest;
      const calm = /\b(sorry|sad|miss|lost|hurt|tired|sleepy|rough)\b/.test(txt);
      const ampK = (excl ? 1.2 : 1) * (calm ? 0.7 : 1);
      const cands = [];   // [priority, type, word index, extra]
      const shrugM = SHRUG_RE.exec(txt);
      W.forEach((w, j) => {
        const i = a + j, nx = W[j + 1] ? W[j + 1].low : '';
        if (LEX.ord[w.low] > 0) cands.push([9, 'count', i, {n: LEX.ord[w.low]}]);
        else if (LEX.num[w.low]) {
          const n = LEX.num[w.low];
          if (n >= 2 || /^(thing|way|time|more|step|reason|idea|word)s?$/.test(nx)) cands.push([8.5, n > 5 ? 'count_both' : 'count', i, {n}]);
        }
        if (LEX.side[w.low] && /^(the|your|my|to|on|his|her|its)$/.test(W[j - 1] ? W[j - 1].low : '')) cands.push([8, 'point_side', i, {own: LEX.side[w.low]}]);
        if (LEX.there.has(w.low) && (w.low === 'there' ? !/^(is|are|was|were|'s)$/.test(nx) || /\bover\b/.test(txt) : nx === 'there')) cands.push([7.5, 'point_side', w.low === 'over' ? i + 1 : i, {}]);
        if (LEX.that.has(w.low) && (/^(one|ones|way|thing)$/.test(nx) || W[j - 1] && /^(at|see|look)$/.test(W[j - 1].low))) cands.push([6, 'point_side', i, {}]);
        if (LEX.here.has(w.low) && (/^(one|is|small|big|much|way)$/.test(nx) || w.low === 'here')) cands.push([5.5, 'present', i, {}]);
        for (const t of ['big', 'small', 'up', 'down', 'long', 'round']) if (LEX[t].has(w.low)) {
          if (t === 'up' && /^(come|came|show|wake|end|give|pick|set|make|look)/.test(W[j - 1] ? W[j - 1].low : '')) continue;   // phrasal verbs
          if (t === 'down' && /^(sit|calm|slow|settle|write|put)/.test(W[j - 1] ? W[j - 1].low : '')) continue;
          if (t === 'small' && w.low === 'little' && nx && !/^(bit|one|thing|guy)$/.test(nx) && rnd() < 0.5) continue;
          cands.push([7, t, i, {}]);
        }
        if (LEX.neg.has(w.low) && !(shrugM && shrugM.index <= txt.indexOf(w.low) + 2 && txt.indexOf(w.low) < shrugM.index + shrugM[0].length)) cands.push([6.5, 'negate', i, {}]);
        if (LEX.excl.has(w.low)) cands.push([5, 'exclaim', i, {}]);
        if (LEX.affirm.has(w.low) && j <= 1) cands.push([4.5, 'nod', i, {}]);
        if (LEX.you.has(w.low) && rnd() < 0.55) cands.push([4, 'you', i, {}]);
        if (LEX.self.has(w.low) && (w.low !== 'i' || /^(love|think|believe|promise|bet|hope|feel|really|totally|mean)$/.test(nx)) && rnd() < 0.55) cands.push([3.8, 'self', i, {}]);
        if (LEX.open.has(w.low) && (j === 0 || /^(the|an?)$/.test(W[j - 1] ? W[j - 1].low : ''))) cands.push([3.5, 'open_palms', i, {}]);
      });
      if (shrugM) {   // the stroke goes on the phrase's content word ("not SURE", "KIND of", "don't KNOW", "MAYBE")
        const wi = a + Math.max(0, txt.slice(0, shrugM.index).split(' ').length - 1), mw = shrugM[0].split(' ');
        let k = mw.length - 1; while (k > 0 && FUNC.has(mw[k]) && !/^(know|sure)$/.test(mw[k])) k--;
        if (/^(kind|sort) of$/.test(shrugM[0])) k = 0;
        cands.push([7.8, 'shrug', Math.min(b, wi + k), {}]);
      }
      // the nuclear stress = the last content word (a beat there), the prominence of emphatic modifiers
      let nuc = -1; for (let j = W.length - 1; j >= 0; j--) if (!W[j].func && W[j].syl > 0) { nuc = a + j; break; }
      for (let j = 0; j < W.length - 1; j++) if (/^(so|really|very|super|totally)$/.test(W[j].low) && !W[j + 1].func) nuc = a + j + 1;
      if (quest && nuc >= 0) cands.push([3, 'offer', nuc, {}]);
      cands.sort((x, y) => y[0] - x[0] || x[2] - y[2]);
      // rest: some phrases get no hand gesture (more often when nothing in them asks for one, or right after a busy one)
      const strong = cands.length && cands[0][0] >= 6;
      const recentBusy = C.gestures.slice(-2).filter(g => !/nod|brow/.test(g.type)).length >= 2;
      const pRest = strong ? 0.06 : (calm ? 0.45 : 0.24) + (recentBusy ? 0.15 : 0);
      const nWords = W.length;
      if (!cands.length && nuc < 0) return;
      if (rnd() < pRest) { if (nuc >= 0 && rnd() < 0.5) add('brow_beat', nuc, {amp: ampK}); return; }
      let main = cands[0];
      if (!main || (main[0] < 4 && rnd() < 0.5)) main = [2, excl && rnd() < 0.5 ? 'beat_both' : 'beat', nuc >= 0 ? nuc : b, {variant: rnd() < 0.3 ? 'down' : 'up'}];
      add(main[1], main[2], Object.assign({amp: ampK * U(0.85, 1.15)}, main[3]));
      // a second stroke in a long phrase: a beat on the nuclear stress if it is far from the main one (or the next trigger)
      const second = cands.slice(1).find(c => Math.abs(c[2] - main[2]) >= 3 && c[0] >= 5);
      if (second) add(second[1], second[2], Object.assign({amp: ampK}, second[3]));
      else if (nWords >= 6 && nuc >= 0 && Math.abs(nuc - main[2]) >= 3 && rnd() < 0.6) add('beat', nuc, {amp: ampK * 0.9, variant: rnd() < 0.4 ? 'down' : 'up', follow: true});
    }

    // ---- motion adapter: the motion module's API when present, else a shim over the motion.js internals
    function busyNow() {
      const b = motion && typeof motion.armsBusy === 'function' ? motion.armsBusy() || {} : shimBusy();
      const st = motion && typeof motion.state === 'function' ? motion.state() || {} : null;
      const hold = {l: false, r: false};
      if (st && st.holding) { if (st.holding.right) hold.l = true; if (st.holding.left) hold.r = true; }   // own right paw = screen-left
      return {l: !!b.l, r: !!b.r, body: !!b.body, head: !!b.head, hold};
    }
    function shimBusy() {
      const S = motion && motion.state && typeof motion.state === 'object' ? motion.state : null; if (!S) return {};
      const busy = {l: false, r: false, body: false, head: false};
      for (const p in S.holds || {}) { const h = S.holds[p]; if (h.out) continue;
        if (/_l(_|$)/.test(p) && /^(hand|thumb|point|shoulder)/.test(p)) busy.l = true; if (/_r(_|$)/.test(p) && /^(hand|thumb|point|shoulder)/.test(p)) busy.r = true;
        if (/^head_/.test(p)) busy.head = true; if (/^body_/.test(p)) busy.body = true; }
      for (const g of S.gestures || []) { if (g.rel || (g.src !== 'cmd' && g.src !== 'reflex')) continue;
        for (const p of g.params || []) { if (/hand_l|point_l|thumb_l/.test(p)) busy.l = true; if (/hand_r|point_r|thumb_r/.test(p)) busy.r = true; if (/^head_(yaw|pitch)/.test(p) && g.src === 'cmd') busy.head = true; } }
      return busy;
    }
    function play(def, g, at, side) {
      if (motion && typeof motion.playGesture === 'function') {
        try { return motion.playGesture(def, {src: 'cospeech', side, at, intensity: g.amp, parts: def.parts, g}); } catch (e) { return null; }
      }
      // shim: a gesture record in the motion.js engine (speech priority, so commands / reflexes / holds win), clocked on its S.t
      const S = motion && motion.state && typeof motion.state === 'object' ? motion.state : null; if (!S) return null;
      const off = (o.motionNow ? o.motionNow() : S.t) - nowT();
      const t0 = at + off, dur = typeof def.dur === 'function' ? def.dur(g) : def.dur;
      const r0 = def.fn(def.apex, g), params = Object.keys(Object.assign({}, r0.set || {}, r0.add || {}));
      // hand over softly: the older one fades out across the new one's attack (never cut short below "now")
      const mNow = o.motionNow ? o.motionNow() : S.t;
      for (const x of S.gestures) if (x.cospeech && !x.rel && x.params.some(p => params.includes(p)) && x.t0 + x.dur > t0) {
        const u = mNow - x.t0, w = u < 0 ? 0 : x.attack > 0 ? mj(u / x.attack) : 1;
        x.rel = {t0: Math.max(mNow, t0 + def.attack * 0.3), w: Math.max(w, x.t0 < t0 ? 1 : 0)}; x.release = Math.max(0.3, x.release);   // its own retraction speed
      }
      S.gid = (S.gid || 0) + 1;
      const G0 = {name: 'cs:' + def.name, g, side, t0, dur, attack: def.attack, release: def.release, fn: def.fn, src: 'speech', rel: null, params, id: S.gid, cospeech: true, extra: {}};
      S.gestures.push(G0);
      return 'cs' + G0.id;
    }
    function cancelMotion() {
      if (motion && typeof motion.cancelGestures === 'function') { try { motion.cancelGestures({src: 'cospeech'}); } catch (e) {} return; }
      const S = motion && motion.state && typeof motion.state === 'object' ? motion.state : null; if (!S) return;
      const off = (o.motionNow ? o.motionNow() : S.t);
      S.gestures = S.gestures.filter(x => !(x.cospeech && x.t0 > off));                     // not started: gone
      for (const x of S.gestures) if (x.cospeech && !x.rel) { x.rel = {t0: off, w: 1}; x.release = 0.28; }   // in flight: retract now
    }
    // shim only: params the motion.js engine does not carry (fingers_*) — the page merges them into the rig vector
    function extras(motionT) {
      const S = motion && motion.state && typeof motion.state === 'object' ? motion.state : null; if (!S) return {};
      const out = {};
      for (const x of S.gestures || []) {
        if (!x.cospeech) continue;
        const u = motionT - x.t0; if (u < 0 || u > x.dur) continue;
        let w = x.attack > 0 ? mj(u / x.attack) : 1; if (u > x.dur - x.release) w = Math.min(w, 1 - mj((u - (x.dur - x.release)) / x.release));
        if (x.rel) w = Math.min(w, 1 - mj((motionT - x.rel.t0) / 0.28));
        const r = x.fn(u, x.g).set || {};
        for (const k in r) if (/^fingers_/.test(k)) out[k] = Math.max(out[k] || 0, r[k] * clamp(w, 0, 1));
      }
      return out;
    }

    // ---- every frame: commit what is due
    const HORIZON = 0.05, LEAD = o.lead || [0.0, 0.08], READY_LEAD = 0.75;   // apex 0-80 ms before the stressed syllable
    // c2 confidence (audio aligner): a stroke is committed only when the middle half of the particles' predicted times for
    // its syllable is within ±SPREAD_COMMIT (else it waits while there is time, then is skipped); a committed k-driven
    // stroke keeps re-aiming its apex at the newest estimate until its rise begins (FREEZE before the peak), and if the
    // estimate is still unsure then (> SPREAD_ABORT) the stroke is aborted (the paw goes back down unstroked)
    const SPREAD_COMMIT = o.spreadCommit || 0.11, SPREAD_ABORT = o.spreadAbort || 0.09, FREEZE = 0.17, LATE_SLACK = 0.18;
    const REAIM = new Set(['beat', 'beat_both', 'open_palms', 'self', 'you', 'point_side', 'present', 'big', 'small', 'up', 'down', 'count', 'count_both', 'exclaim', 'offer', 'nod', 'brow_beat']);
    const sideEnd = {l: -9, r: -9};             // when our last committed gesture on that paw lets go
    const sideUp = {l: -9, r: -9};              // when that paw arrives in gesture space (end of the ready / stroke attack)
    function tick(tNow) {
      if (!C.enabled || C.cancelled) return;
      const tm = C.tm, T = tm.T;
      if (T.onset === null) return;
      const flowing = tm.playing(tNow);
      if (tm.spread) reaim(tNow);
      // streaming text: a phrase whose end has not arrived yet is planned as far as it has, once its voice is near
      if (!LEGACY && C.planned < C.words.length && !C.final) { const tw = tm.timeOf(C.words[C.planned].u0); if (tw != null && tw - tNow < 0.9) { planPhrase(C.planned, C.words.length - 1); C.planned = C.words.length; } }
      for (let gi = 0; gi < C.gestures.length; gi++) {
        const g = C.gestures[gi];
        if (g.committed || g.dropped) continue;
        const Xs = tm.stressU(g.word); if (Xs === null) continue;
        const tA = tm.timeOf(Xs); if (tA === null) continue;
        const def = G[g.type];
        const lead = g.lead != null ? g.lead : (g.lead = U(LEAD[0], LEAD[1]));
        const apexT = tA - lead;
        if (!g.readyDone && apexT - tNow <= READY_LEAD) ready(g, def, apexT, tNow, flowing);
        // a paw that is not up yet (no ready phase fitted) needs the longer travel: commit it earlier
        const armDef = def.parts === 'arms' || def.parts === 'both';
        const start = apexT - (armDef && !(g.readyIds && g.readyIds.length) ? Math.max(def.apex, 0.33) : def.apex);
        if (start - tNow > HORIZON) continue;
        if (!flowing) { g.dropped = 'voice not flowing'; C.stats.dropped++; continue; }
        const isReady = !!(g.readyIds && g.readyIds.length) || def.parts === 'head' || def.parts === 'shoulders';
        // in a pause the next word's time is a guess: hold the stroke until the voice is back (the paw waits, ready)
        if (T.silentRun >= 0.06 && Xs > T.P + 0.05 && apexT > tNow + 0.02) continue;
        if (apexT < tNow + (isReady ? 0.07 : 0.12)) { g.dropped = 'late'; C.stats.late++; continue; }     // its word is (nearly) gone: never a late stroke
        if (apexT - C.lastApex < 0.42) { g.dropped = 'too close'; C.stats.dropped++; continue; }
        if (tm.spread) {
          const sp = tm.spread(Xs), need = REAIM.has(g.type) ? SPREAD_COMMIT : SPREAD_ABORT;
          if (!(sp <= need)) {
            if (apexT - tNow > (isReady ? 0.16 : 0.34)) continue;                 // there is still time: it may firm up
            g.dropped = 'unsure'; C.stats.unsure = (C.stats.unsure || 0) + 1; fallback(g, tNow); continue;   // no gesture beats a mistimed one
          }
        }
        commit(g, def, Math.max(tNow, start), apexT, tNow);
      }
    }
    // late binding: a committed stroke whose envelope is driven by its apex re-aims at the newest audio-aligned estimate
    // while its rise has not begun (k is still 0 there, so the move is invisible); frozen FREEZE before the peak
    function reaim(tNow) {
      const tm = C.tm;
      for (const g of C.gestures) {
        if (!g.committed || g.frozen || !g.gg) continue;
        if (tNow >= g.apex_t - FREEZE) {
          g.frozen = true;
          const sp = tm.spread(tm.stressU(g.word));
          // counting carries content (the fingers say "three"): it keeps a slightly wider margin before it is given up
          const lim = /^count/.test(g.played || g.type) ? SPREAD_ABORT + 0.04 : SPREAD_ABORT;
          if (g.reaim && sp > lim && tm.timeOf(tm.stressU(g.word)) > tNow + 0.02) abort(g, 'unsure at the stroke');
          continue;
        }
        if (!g.reaim) continue;
        const tA = tm.timeOf(tm.stressU(g.word)); if (tA == null) continue;
        let ap = tA - g.lead;
        const lo = Math.max(tNow + FREEZE + 0.005, g.at + Math.max(0.16, g.def2.attack * 0.8)), hi = g.at + g.def2.dur - g.def2.release - 0.1;
        ap = clamp(ap, lo, Math.max(lo, hi));
        if (Math.abs(ap - g.apex_t) < 0.004) continue;
        g.apex_t = ap; g.gg.apex = ap - g.at; if (g.rec) g.rec.apex = +ap.toFixed(3);
      }
    }
    // an unsure stroke → a plain beat on the next content word of the reply that has nothing planned (once)
    function fallback(g, tNow) {
      if (g.fallback || LEGACY) return;
      for (let j = g.word + 1; j < Math.min(C.words.length, g.word + 6); j++) {
        const W = C.words[j]; if (W.func || C.gestures.some(x => x.word === j && !x.dropped)) continue;
        const t = C.tm.timeOf(C.tm.stressU(j)); if (t == null || t - tNow < 0.35) continue;
        C.gestures.push({type: 'beat', word: j, side: null, committed: false, dropped: false, id: null, amp: g.amp * 0.9, fallback: true, variant: rnd() < 0.4 ? 'down' : 'up'});
        C.stats.fallback = (C.stats.fallback || 0) + 1; return;
      }
    }
    function abort(g, why) {
      g.committed = false; g.dropped = why; C.stats.unsure = (C.stats.unsure || 0) + 1; C.stats.committed--;
      g.gg.amp = 1e-4;
      if (motion && typeof motion.cancelGestures === 'function') { try { motion.cancelGestures({src: 'cospeech', id: g.id}); } catch (e) {} }
      else { const S = motion && motion.state && typeof motion.state === 'object' ? motion.state : null;
        if (S) for (const x of S.gestures) if (x.cospeech && 'cs' + x.id === g.id && !x.rel) { x.rel = {t0: o.motionNow ? o.motionNow() : S.t, w: 1}; x.release = 0.3; } }
      const i = C.commits.indexOf(g.rec); if (i >= 0) C.commits.splice(i, 1);
      const j = C.last.indexOf(g.lastRec); if (j >= 0) C.last.splice(j, 1);
    }
    const HEAD_ONLY = {negate: 'head_shake', shrug: 'shrug_small', exclaim: 'brow_beat', beat: 'brow_beat', beat_both: 'brow_beat', open_palms: 'brow_beat',
      offer: 'brow_beat', you: 'brow_beat', self: 'brow_beat', big: 'brow_beat', small: 'brow_beat', up: 'brow_beat', down: 'nod', count: null, count_both: null,
      point_side: null, present: null, long: null, round: null};
    // choose type + paw for g under the current busy state → {type, side} or null
    function choose(g, def, busy) {
      let type = g.type, side = g.side0 || null;
      const both = def.parts === 'both', oneHand = def.parts === 'arms';
      if (g.type === 'point_side' && g.own) side = g.own === 'left' ? 'r' : 'l';     // its own left = screen-right
      if (busy.body) return {why: 'body busy'};
      const free = s => !busy[s] && !busy.hold[s];
      if (oneHand) {
        side = side || pickSide(null);
        if (!free(side)) side = free(oth(side)) && g.type !== 'point_side' ? oth(side) : null;
        if (!side) type = HEAD_ONLY[g.type];
      } else if (both) {
        if (!free('l') || !free('r')) {
          // one free paw: the one-handed cousin; none: a head/shoulder-only version
          const one = {beat_both: 'beat', open_palms: 'beat', exclaim: 'beat', big: 'beat', small: 'present', shrug: 'shrug_small', long: 'beat', count_both: null}[g.type];
          const fs = free('l') ? 'l' : free('r') ? 'r' : null;
          if (fs && one && G[one].parts === 'arms') { type = one; side = fs; } else type = HEAD_ONLY[g.type] || (one && G[one].holdSafe ? one : null);
        }
      }
      if (!type) return {why: 'paws busy'};
      if (G[type].parts === 'head' && busy.head) return {why: 'head busy'};
      return {type, side, free};
    }
    function ready(g, def, apexT, tNow, flowing) {
      g.readyDone = true;
      if (!flowing || C.cancelled) return;
      const ch = choose(g, def, busyNow()); if (!ch.type) return;
      const d = G[ch.type]; if (d.parts !== 'arms' && d.parts !== 'both') return;
      g.side0 = ch.side;
      const sides = d.parts === 'both' ? ['l', 'r'] : [ch.side];
      const strokeAt = apexT - d.apex;
      for (const sd of sides) {
        if (sideEnd[sd] > strokeAt - 0.05) continue;        // that paw is still up from the last gesture: it waits there
        const lead = strokeAt - tNow; if (lead < 0.3) continue;
        const gg = {side: sd, hold: lead + 0.3};       // total duration: up, wait for the stroke, overlap its attack
        const def2 = {name: 'ready', attack: Math.min(0.34, lead - 0.05), release: 0.3, parts: 'arms', apex: 0.34, dur: G.ready.dur(gg), fn: u => G.ready.fn(u, gg)};
        if (play(def2, gg, tNow, sd) != null) { sideEnd[sd] = Math.max(sideEnd[sd], tNow + def2.dur); sideUp[sd] = tNow + def2.attack; g.readyIds = (g.readyIds || []).concat(sd); }
      }
    }
    function commit(g, def, at, apexT, tNow) {
      const busy = busyNow();
      const ch = choose(g, def, busy);
      if (!ch.type) { g.dropped = ch.why; C.stats.refused++; return; }
      let type = ch.type, side = ch.side; const free = ch.free;
      const d0 = G[type];
      // variety: never the identical gesture twice in a row (same type + side + variant) — swap side / variant
      const prev = C.last[C.last.length - 1];
      const variant = g.variant || (type === 'beat' ? (rnd() < 0.3 ? 'down' : 'up') : null);
      let v = variant;
      // c2: identical = same type on the same paw (whatever the variant): swap paw, else a different beat, else skip
      if (prev && prev.type === type && prev.side === (side || null)) {
        if (d0.parts === 'arms' && free(oth(side)) && g.type !== 'point_side') side = oth(side);
        else if (type === 'beat' && free('l') && free('r') && prev.type !== 'beat_both') { type = 'beat_both'; side = null; }
        else if (type !== 'brow_beat' && !busy.head && (/beat|open_palms|exclaim|offer|big|small|up|you|self/.test(type))) { type = 'brow_beat'; side = null; }
        else { g.dropped = 'repeat'; C.stats.dropped++; return; }
      }
      const d = G[type];
      const reaim = !LEGACY && REAIM.has(type);
      // apex = the stroke peak, s after the gesture starts (a stroke committed late gets a shorter preparation)
      const gg = {side: side || (ch.side || 'l'), amp: clamp(g.amp, 0.6, 1.4), n: g.n, variant: v, dir: rnd() < 0.5 ? -1 : 1, apex: apexT - at};
      // hold into the next planned gesture on the same paw if it is close (no drop to rest in between)
      const nx = C.gestures.find(x => !x.committed && !x.dropped && x !== g);
      if (nx) { const tn = C.tm.timeOf(C.tm.stressU(nx.word)); if (tn && tn - apexT < 1.1) gg.hold = Math.max(0.15, tn - apexT - 0.25); }
      // the def handed to motion closes over its own parameters (whatever g the engine passes is ignored)
      // an unprepared paw (no ready phase, not up from the last gesture) needs >= 0.26 s to get there — else skip the stroke
      const sidesUsed = d.parts === 'both' ? ['l', 'r'] : d.parts === 'arms' ? [gg.side] : [];
      const unready = sidesUsed.some(sd => !(sideUp[sd] <= at + 0.1 && sideEnd[sd] > at - 0.05));
      if (unready && gg.apex < 0.3) { g.dropped = 'no time to prepare'; C.stats.late++; return; }
      // a paw still on its way up (ready phase not finished) keeps travelling at that pace into the stroke
      const still = sidesUsed.reduce((mx, sd) => Math.max(mx, sideUp[sd] - at), 0);
      const att0 = unready ? Math.max(0.26, d.attack) : Math.max(d.attack, still > 0 ? still + 0.12 : 0);
      // a re-aimable stroke gets LATE_SLACK more hold, the room its apex may move later into
      const holdN = gg.hold != null ? gg.hold : Math.max(0.12, d.dur0 - d.apex - d.release);
      const dur = reaim ? gg.apex + holdN + LATE_SLACK + d.release : d.dur(gg);
      const def2 = {name: d.name, attack: Math.max(unready ? 0.26 : 0.08, Math.min(att0, gg.apex - 0.04)), release: d.release, holdSafe: d.holdSafe, parts: d.parts,
        apex: gg.apex, dur, fn: u => d.fn(u, gg)};
      const id = play(def2, gg, at, gg.side);
      if (id === null || id === undefined) { g.dropped = 'refused'; C.stats.refused++; return; }
      g.committed = true; g.id = id; g.at = at; g.apex_t = apexT; g.side = side; g.played = type; g.commitT = tNow; g.gg = gg; g.def2 = def2; g.reaim = reaim;
      for (const sd of sidesUsed) { sideEnd[sd] = Math.max(sideEnd[sd], at + def2.dur - def2.release); sideUp[sd] = Math.max(sideUp[sd], at + def2.attack); }
      C.lastApex = apexT; g.lastRec = {type, side: side || null, variant: v}; C.last.push(g.lastRec); if (C.last.length > 12) C.last.shift();
      C.stats.committed++;
      g.rec = {type, side, word: g.word, w: (C.words[g.word] || {}).w, at: +at.toFixed(3), apex: +apexT.toFixed(3), commitT: +tNow.toFixed(3)};
      C.commits.push(g.rec);
      if (C.commits.length > 400) C.commits.splice(0, 200);
      if (o.onCommit) o.onCommit({type, side, word: g.word, at, apex: apexT});
    }
    function cancel(why) {
      if (C.cancelled) return;
      C.cancelled = true; C.cancelWhy = why || 'cancel';
      for (const g of C.gestures) if (!g.committed && !g.dropped) { g.dropped = 'cancelled'; C.stats.cancelled++; }
      // the motion side: drop what has not started, retract what is in flight
      cancelMotion();
    }
    function snapshot() {
      const tm = C.tm;
      return {onset: tm.T.onset, rate: +tm.T.rate.toFixed(2), cancelled: C.cancelled ? C.cancelWhy : false,
        words: C.words.map((w, i) => ({w: w.w, i, syl: w.syl, stress: w.stress, t_est: tm.timeOf(w.u0), stress_t: tm.timeOf(tm.stressU(i)), arrived: w.arrived ?? null})),
        gestures: C.gestures.map(g => ({type: g.played || g.type, planned: g.type, side: g.side, word: g.word, w: (C.words[g.word] || {}).w, committed: g.committed, dropped: g.dropped || false,
          at: g.at ?? null, apex_t: g.apex_t ?? null, id: g.id}))};
    }
    return {
      text, tick, cancel, extras,
      audio: (t0, rms, silent, dur, pcm) => { if (C.enabled) C.tm.audio(t0, rms, silent, dur, pcm); },
      plan: () => snapshot(), plans: () => C.plans.concat([snapshot()]), rate: () => C.R0 || null, commits: () => C.commits.slice(), stats: () => Object.assign({}, C.stats),
      covering: tNow => C.enabled && !C.cancelled && C.words.length > 0 && C.tm.playing(tNow),
      setEnabled(on) { C.enabled = !!on; if (!on) cancel('disabled'); },
      reset() { newStream('reset'); C.respId = null; },
      _C: C,
    };
  }
  return {createCospeech, tokenize, sylCount, stressIdx, createTiming, createAligner, createDual, GESTURES: G, numWords};
});
