/* turnbid.js — nonverbal TURN-TAKING bids for the live talking avatar (perception, turn-taking piece).
 *
 * window.AvatarTurnBid = {create, DEFAULTS, CUES}   (also module.exports when `module` exists; UMD, no dependencies)
 *
 * create({onEvent(ev), threshold, refractory, ...}) → {frame(f, ctx), setThreshold(x), stats(), state(), reset()}
 *   f   = a perception.js feature frame (~18 Hz; see the header of app/perception.js): face {present, z, yaw, blend{…},
 *         xp{talk}, age}, hands[] {id, t, main, x, y, scale, ext[4], …}, attention (0..1 looking at the camera/screen),
 *         attending. Missing pieces are tolerated (face dropouts, no hands, no pose).
 *   ctx = {t (s, same clock as f.t; defaults to f.t), userSpeaking: bool, userLevel: linear mic RMS (the page's 10 ms
 *         meter; voiced > 0.012), avatarSpeaking: bool}.  userLevel is the primary voice signal (it has no
 *         end-of-speech hangover); userSpeaking is used alone only when no level is given.
 *   Returns the frame's result {t, d (max deliberateness this frame), cue (its cue), bid, scores{cue: D}, events[]}.
 *
 * EVENT (onEvent + the returned events[]): {type:'turn_bid', cue, deliberateness 0..1, t, bid:'reply'|'floor', parts:{…}}
 *   bid 'reply' = the user invites the avatar to talk — only while the user is silent and the avatar is not speaking:
 *     beckon          repeated finger curl / come-here hand (≥ 3 curl–extend swings of the finger extension in ~3 s,
 *                     the hand itself roughly still — a lateral oscillation of an open hand is a wave, not a beckon)
 *     lean_in         the user's face distance falls vs THEIR running baseline (ln z, τ 10 s, frozen while away from it) and holds
 *     raised_brows    brows up vs THE USER'S resting brows (lower-envelope baseline) and held — an expectant look
 *     gaze_after_pause  the user spoke (≥ 0.6 s voiced in their last turn), stopped, and holds eye contact through the pause
 *   bid 'floor' = the user wants to speak while the AVATAR is speaking (the avatar should yield):
 *     inhale          the mouth opens from THE USER'S closed baseline and holds open without voice (not the 3–6 Hz jaw of talk,
 *                     not a smile/laugh)
 *     talk_over       the user is voiced AND their jaw moves like speech for a sustained stretch (longer than a backchannel)
 * DELIBERATENESS (per cue, each frame) = a soft AND of parts in 0..1:
 *     D = amp^½ · hold^½ · (e0 + (1−e0)·eye) · quiet · avatar            (reply cues; gaze_after_pause: see code)
 *     amp   = amplitude above THE USER'S baseline, normalised per cue        hold  = how long it has been held
 *     eye   = eye contact (f.attention, smoothed)                      quiet = the user is silent (and for how long)
 *     avatar= reply: the avatar is silent (and for how long);  floor: the avatar is speaking
 *   A bid is emitted when D ≥ threshold, once per episode (the cue's D must fall below 0.5·threshold for 0.4 s before
 *   that cue can bid again) and at most once per `refractory` s per bid type.
 * Inhale while the avatar is silent never bids (the user is about to take the turn anyway) but vetoes reply bids for 1 s.
 * Cost: a few dozen arithmetic ops per frame (tests/turnbid_test.cjs reports ms/frame).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarTurnBid = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';

  const CUES = {beckon: 'reply', lean_in: 'reply', raised_brows: 'reply', gaze_after_pause: 'reply', inhale: 'floor', talk_over: 'floor'};
  const DEFAULTS = {
    threshold: 0.6,          // calibrated in tests/turnbid_test.cjs (held out by person)
    refractory: {reply: 4.0, floor: 3.0},   // s between two bids of one type (reset when either side speaks again)
    rearm: 0.4,              // s below 0.5·threshold before the same cue may bid again
    voiceTh: 0.012,          // the page's mic threshold
    warm: 2.0,               // s of face before face cues may bid (the user's resting face must be known)
    cues: null,              // null = all; or a list of cue names to enable
    // per-cue shapes. amp: 0→1 between a0..a1 ABOVE THE USER'S BASELINE, where a0 is raised to k × their own typical excursion
    // (the running 90th percentile of their deviation while silent) — an expressive or noisy face needs a bigger move.
    brows: {a0: 0.05, a1: 0.22, k: 1.25, h0: 0.25, h1: 1.0, on: 0.05},
    lean: {a0: 0.04, a1: 0.14, k: 1.5, h0: 0.25, h1: 0.9, on: 0.035, maxHold: 6.0},
    inhale: {a0: 0.06, a1: 0.20, k: 1.25, h0: 0.12, h1: 0.45, on: 0.05, vel: 1.0, smile: 0.2, stale: 1.6, rise: 0.8},
    talkOver: {h0: 0.8, h1: 1.6, talk: 0.3, gap: 0.3},
    beckon: {swing: 0.28, a0: 0.25, a1: 0.6, win: 3.0, minSwings: 3},
    gaze: {s0: 0.9, s1: 1.8, minTurn: 0.6, turnGap: 1.2, maxPause: 8.0},
    quiet: {s0: 0.6, s1: 1.4},
    avatarGap: {s0: 0.2, s1: 1.0},
    eta: {face: 0.06, warm: 0.6},   // quantile-tracker speeds (blendshape units / s)
  };

  const cl = (v, a, b) => v < (a === undefined ? 0 : a) ? (a === undefined ? 0 : a) : v > (b === undefined ? 1 : b) ? (b === undefined ? 1 : b) : v;
  const ramp = (v, a, b) => cl((v - a) / Math.max(1e-6, b - a));
  const r3 = v => Math.round(v * 1000) / 1000;
  const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : () => Date.now();

  function merge(a, b) {
    const o = Object.assign({}, a);
    for (const k in b || {}) o[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') ? Object.assign({}, a[k], b[k]) : b[k];
    return o;
  }
  // running quantile (stochastic approximation): moves at most eta·dt per update, robust to the very episodes we detect
  function qtrack(q, x, dt, q0, eta) { return q === null ? x : q + eta * dt * (x > q ? q0 : -(1 - q0)); }

  function create(opts) {
    const O = merge(DEFAULTS, opts || {});
    const onEvent = (opts && opts.onEvent) || null;
    let threshold = +O.threshold;
    const enabled = {};
    for (const c in CUES) enabled[c] = !O.cues || O.cues.indexOf(c) >= 0;

    const S = {};
    const ST = {frames: 0, ms: 0, msMax: 0, bids: {}, byType: {reply: 0, floor: 0}, maxD: {}};
    const P = {};   // the current frame's parts (numbers; rounded copies only go out with an event / state())
    function reset() {
      Object.assign(S, {
        lastT: null, lastVoiceT: -1e9, runStart: -1e9, turnVoiced: 0, avLastT: -1e9, avWas: false,
        // face: quantile baselines (rest = 40th pct) and the user's typical excursion (90th pct) while silent
        faceSeen: 0, faceLastT: -1e9, bLo: null, bHi: null, jLo: null, jHi: null, sLo: null, lzBase: null, lzHi: 0.03, lzAway: 0,
        jawPrev: null, jawVel: 0, talk: 0, eye: 0,
        browHeld: 0, leanHeld: 0, openHeld: 0, openStart: -1e9, openSmile: 0, sharp: -1, jring: [], preSilent: 0, inhaleVetoT: -1e9,
        // gaze after pause
        pauseEyeSum: 0, pauseEyeN: 0, pauseFired: false,
        // hands
        tracks: {},
        // emission
        epi: {}, lastBid: {reply: -1e9, floor: -1e9}, scores: {},
      });
      for (const c in CUES) S.epi[c] = {active: false, low: 0};
    }
    reset();

    // ── hands: beckon = repeated curl/extend swings of the finger extension, the hand roughly still ──
    function updateHands(f, t, faceOk) {
      const B = O.beckon;
      let best = 0, bestParts = null;
      const hands = Array.isArray(f.hands) ? f.hands : [];
      const seen = {};
      const fc = faceOk ? f.face : null, fh = fc && fc.topY > fc.chinY ? fc.topY - fc.chinY : 0;
      for (const h of hands) {
        if (!h || !h.ext || h.ext.length < 4) continue;
        if (faceOk && h.main === false) continue;             // someone else's hand
        // two hands together (clasped / fidgeting with each other) are not a one-hand beckon
        let clasped = false;
        for (const o of hands) if (o !== h && o && Math.hypot((+o.x || 0) - (+h.x || 0), (+o.y || 0) - (+h.y || 0)) < 2.5 * Math.max(+h.scale || 0.05, +o.scale || 0.05)) clasped = true;
        if (clasped) continue;
        // a beckon is made in view, at chest height or above — not with hands resting low (lap, waist, desk)
        const low = fh > 0 ? ramp((fc.chinY - (+h.y || 0)) / fh, 3.0, 4.5) : 0;
        const id = h.id != null ? h.id : (h.side || 'x');
        seen[id] = 1;
        let tr = S.tracks[id];
        if (!tr) tr = S.tracks[id] = {lastHT: -1, sw: [], dir: 0, ext: null, extT: 0, pos: [], lastSeen: t};
        tr.lastSeen = t;
        const ht = h.t != null ? +h.t : t;
        if (ht === tr.lastHT) continue;                      // sample-and-hold: only fresh detections
        tr.lastHT = ht;
        // curl signal: the index finger and the mean of all four (a whole-hand beckon curls all of them)
        const e = h.ext, c = Math.max(e[0], (e[0] + e[1] + e[2] + e[3]) / 4);
        const sc = Math.max(0.02, +h.scale || 0.1);
        tr.pos.push([ht, +h.x || 0, +h.y || 0, sc]);
        while (tr.pos.length && ht - tr.pos[0][0] > B.win) tr.pos.shift();
        // turning points with hysteresis 0.1 (the running extreme becomes a turning point once the signal leaves it)
        if (tr.ext === null || ht - tr.extT > 1.2) { tr.ext = null; tr.dir = 0; tr.lo = [ht, c]; tr.hi = [ht, c]; tr.sw = []; tr.ext = c; tr.extT = ht; }
        if (tr.dir === 0) {
          if (c < tr.lo[1]) tr.lo = [ht, c];
          if (c > tr.hi[1]) tr.hi = [ht, c];
          if (c > tr.lo[1] + 0.1) { tr.sw.push(tr.lo); tr.dir = 1; tr.ext = c; tr.extT = ht; }
          else if (c < tr.hi[1] - 0.1) { tr.sw.push(tr.hi); tr.dir = -1; tr.ext = c; tr.extT = ht; }
        } else if (tr.dir === 1) {
          if (c >= tr.ext) { tr.ext = c; tr.extT = ht; }
          else if (c < tr.ext - 0.1) { tr.sw.push([tr.extT, tr.ext]); tr.dir = -1; tr.ext = c; tr.extT = ht; }
        } else {
          if (c <= tr.ext) { tr.ext = c; tr.extT = ht; }
          else if (c > tr.ext + 0.1) { tr.sw.push([tr.extT, tr.ext]); tr.dir = 1; tr.ext = c; tr.extT = ht; }
        }
        while (tr.sw.length && ht - tr.sw[0][0] > B.win) tr.sw.shift();
        // swings = consecutive turning points; count those big enough and quick enough (0.1–0.9 s per half cycle)
        let n = 0, ampSum = 0;
        const pts = tr.sw.concat([[tr.extT, tr.ext]]);
        for (let i = 1; i < pts.length; i++) {
          const a = Math.abs(pts[i][1] - pts[i - 1][1]), d = pts[i][0] - pts[i - 1][0];
          if (a >= B.swing && d >= 0.08 && d <= 0.9) { n++; ampSum += a; } else if (d > 0.9) { n = 0; ampSum = 0; }
        }
        if (n < 2) continue;
        // hand travel over the window vs its size: a beckon keeps the hand in place; a lateral sweep = a wave/gesture
        let xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9, scm = 0;
        for (const p of tr.pos) { xmin = Math.min(xmin, p[1]); xmax = Math.max(xmax, p[1]); ymin = Math.min(ymin, p[2]); ymax = Math.max(ymax, p[2]); scm += p[3]; }
        scm /= Math.max(1, tr.pos.length);
        const travel = Math.max(xmax - xmin, 0.6 * (ymax - ymin)) / (2 * Math.max(0.03, scm));   // in palm lengths (x,y are ±1 units = 2 frame widths)
        const still = 1 - ramp(travel, 1.0, 2.2);
        const amp = ramp(ampSum / n, B.a0, B.a1);
        const cyc = ramp(n, B.minSwings - 1, B.minSwings + 1);   // 2 swings → 0, 3 → 0.5, 4+ → 1
        const v = Math.sqrt(amp * cyc) * still * (1 - low);
        if (v > best) { best = v; bestParts = {amp: r3(amp), swings: n, still: r3(still)}; }
      }
      for (const id in S.tracks) if (!seen[id] && t - S.tracks[id].lastSeen > 0.8) delete S.tracks[id];
      return {v: best, parts: bestParts};
    }

    // eye contact 0..1: perception's f.attention when the frame has it; else f.attending; else from the head pose and
    // the eye gaze (looking into the lens / at the screen ≈ yaw, pitch and gaze near 0)
    function eyeContact(f, face) {
      if (typeof f.attention === 'number' && isFinite(f.attention)) return cl(f.attention);
      if (typeof f.attending === 'boolean') return f.attending ? 0.9 : 0.1;
      const y = +face.yaw || 0, p = +face.pitch || 0, gx = +face.gazeX || 0, gy = +face.gazeY || 0;
      const lx = typeof face.lookX === 'number' ? face.lookX : y + 0.75 * gx, ly = typeof face.lookY === 'number' ? face.lookY : -p + 0.75 * gy;
      return Math.exp(-Math.pow(lx / 0.25, 2) - Math.pow((ly + 0.05) / 0.32, 2));
    }
    function frame(f, ctx) {
      const t0 = now();
      ctx = ctx || {};
      const t = ctx.t != null ? +ctx.t : +(f && f.t) || 0;
      const dt = S.lastT === null ? 1 / 18 : cl(t - S.lastT, 0, 0.25);
      S.lastT = t;
      const lp = (p, x, tau) => p + (x - p) * (1 - Math.exp(-dt / Math.max(1e-3, tau)));
      // ── voice context ──
      const lvl = ctx.userLevel;
      const hasLevel = typeof lvl === 'number' && isFinite(lvl);
      const voicedNow = hasLevel ? lvl > O.voiceTh : !!ctx.userSpeaking;
      if (voicedNow) {
        if (t - S.lastVoiceT > O.gaze.turnGap) S.turnVoiced = 0;
        if (t - S.lastVoiceT > O.talkOver.gap) S.runStart = t;
        S.turnVoiced += dt; S.lastVoiceT = t; S.pauseEyeSum = 0; S.pauseEyeN = 0; S.pauseFired = false;
        if (S.turnVoiced > 0.4) S.lastBid.reply = -1e9;   // the user took a turn: an earlier reply bid is spent
      }
      const silentFor = t - S.lastVoiceT;
      const voicedRun = silentFor <= O.talkOver.gap ? S.lastVoiceT - S.runStart : 0;
      const heTalking = hasLevel ? silentFor < 0.3 : !!ctx.userSpeaking;
      const av = !!ctx.avatarSpeaking;
      if (av) { S.avLastT = t; if (!S.avWas) S.lastBid.reply = -1e9; }   // the avatar answered: the reply bid is spent
      if (!av && S.avWas) S.lastBid.floor = -1e9;                        // the avatar stopped: the floor bid is spent
      S.avWas = av;
      const avSilentFor = av ? 0 : t - S.avLastT;

      // ── face ──
      const face = f && f.face && f.face.present && (f.face.blend || f.face.expr) ? f.face : null;
      const fresh = !!face && !(face.age > 0.35);
      let browV = 0, leanV = 0, inhaleV = 0, talkOverV = 0, eye = S.eye;
      if (fresh) {
        if (t - S.faceLastT > 3.0) {   // a new face (or back after a long absence): re-learn the resting face
          S.faceSeen = 0; S.bLo = S.bHi = S.jLo = S.jHi = S.sLo = S.lzBase = null; S.lzHi = 0.03; S.jawPrev = null; S.talk = 0;
        }
        S.faceLastT = t; S.faceSeen += dt;
        const X = face.expr || {};   // perception's expr summary stands in when a stream carries no raw blendshapes
        const B = face.blend || {browInnerUp: X.browUp, browOuterUpLeft: X.browUp, browOuterUpRight: X.browUp, jawOpen: X.jawOpen,
          mouthSmileLeft: X.smile, mouthSmileRight: X.smile};
        const brow = 0.5 * (+B.browInnerUp || 0) + 0.25 * ((+B.browOuterUpLeft || 0) + (+B.browOuterUpRight || 0));
        const jaw = +B.jawOpen || 0;
        const smile = 0.5 * ((+B.mouthSmileLeft || 0) + (+B.mouthSmileRight || 0));
        // distance: perception's z (m); without it, the face size (distance ∝ 1 / size; only changes matter)
        const z = +face.z, sz = +face.size, lz = z > 0.05 ? Math.log(z) : sz > 0.01 ? Math.log(0.1 / sz) : null;
        const warming = S.faceSeen < O.warm, eta = warming ? O.eta.warm : O.eta.face;
        // jaw activity (talk): speech moves the jaw at 3–6 Hz
        // (measured over ≥ 50 ms so a 30 fps stream's jitter does not read as fast jaw motion — the 18 Hz page rate is ~56 ms)
        if (S.jawPrev === null) { S.jawPrev = jaw; S.jawPrevT = t; }
        if (t - S.jawPrevT >= 0.05) {
          const jdt = t - S.jawPrevT, jv = Math.abs(jaw - S.jawPrev) / jdt;
          S.jawPrev = jaw; S.jawPrevT = t; S.jawVel = lp(S.jawVel, jv, 0.3); S.talk = lp(S.talk, cl(jv / 1.2), 0.35);
        }
        // baselines learn only while the user is silent (speech moves brows and jaw): rest = 40th percentile, their typical
        // excursion = 90th percentile of the deviation
        if (S.bLo === null) { S.bLo = brow; S.jLo = jaw; S.sLo = smile; S.bHi = 0.02; S.jHi = 0.02; }
        if (silentFor > 0.3 || warming) {
          S.bLo = qtrack(S.bLo, brow, dt, 0.4, eta); S.bHi = Math.max(0.01, qtrack(S.bHi, brow - S.bLo, dt, 0.9, 0.5 * eta));
          S.jLo = qtrack(S.jLo, jaw, dt, 0.4, eta); S.jHi = Math.max(0.01, qtrack(S.jHi, jaw - S.jLo, dt, 0.9, 0.5 * eta));
          S.sLo = qtrack(S.sLo, smile, dt, 0.4, eta);
        }
        if (lz !== null) {
          if (S.lzBase === null) S.lzBase = lz;
          const ldv = S.lzBase - lz, away = Math.abs(ldv) > O.lean.on;
          S.lzAway = away ? S.lzAway + dt : 0;
          if (!away || S.lzAway > O.lean.maxHold || warming) S.lzBase = lp(S.lzBase, lz, warming ? 0.5 : 10);
          if (!warming) S.lzHi = Math.max(0.01, qtrack(S.lzHi, Math.abs(ldv), dt, 0.9, 0.01));
        }
        eye = S.eye = lp(S.eye, eyeContact(f, face), 0.15);
        const warm = !warming;
        // raised brows
        const bdev = brow - S.bLo, bA0 = Math.max(O.brows.a0, O.brows.k * S.bHi);
        S.browHeld = bdev >= Math.max(O.brows.on, 0.7 * bA0) ? S.browHeld + dt : 0;
        const bAmp = ramp(bdev, bA0, bA0 + O.brows.a1 - O.brows.a0), bHold = ramp(S.browHeld, O.brows.h0, O.brows.h1);
        browV = warm ? Math.sqrt(bAmp * bHold) * (1 - 0.7 * ramp(S.talk, 0.25, 0.5)) : 0;
        P.b_amp = bAmp; P.b_hold = bHold; P.b_dev = bdev; P.b_a0 = bA0;
        // lean in (frontal faces only: the matrix z is stable through turns, but a big turn is not a lean)
        const ldev = lz !== null && S.lzBase !== null ? S.lzBase - lz : 0, lA0 = Math.max(O.lean.a0, O.lean.k * S.lzHi);
        S.leanHeld = ldev >= Math.max(O.lean.on, 0.7 * lA0) && Math.abs(+face.yaw || 0) < 0.6 ? S.leanHeld + dt : 0;
        const lAmp = ramp(ldev, lA0, lA0 + O.lean.a1 - O.lean.a0), lHold = ramp(S.leanHeld, O.lean.h0, O.lean.h1) * (S.leanHeld > O.lean.maxHold ? 0 : 1);
        leanV = warm ? Math.sqrt(lAmp * lHold) : 0;
        P.l_amp = lAmp; P.l_hold = lHold; P.l_dev = ldev; P.l_a0 = lA0;
        // inhale: the mouth opens QUICKLY from the user's rest, from silence, holds steadily (not talk), not a smile/laugh,
        // and not for long (a mouth left open for seconds is not an intake of breath)
        const jdev = jaw - S.jLo, sdev = smile - S.sLo, jA0 = Math.max(O.inhale.a0, O.inhale.k * S.jHi);
        // the jaw over the last ~0.8 s (ring of 16): an intake of breath is a QUICK opening — most of the opening
        // must have happened within that window when the jaw first reaches the threshold
        const R = S.jring; R.push(t, jaw, smile); while (R.length > 90 || (R.length && t - R[0] > 1.6)) R.splice(0, 3);
        let mn = jaw; for (let i = 0; i < R.length; i += 3) if (t - R[i] <= O.inhale.rise) mn = Math.min(mn, R[i + 1]);
        const gain = jaw - mn;   // how much the jaw opened within the last 0.8 s
        if (jdev >= Math.max(O.inhale.on, 0.6 * jA0) && !voicedNow) {
          // a NEW quick opening restarts the episode even if the mouth was already a little open
          if (S.openHeld > O.inhale.rise && gain >= 0.6 * jA0 && S.sharp >= 0) S.openHeld = 0;
          if (S.openHeld === 0) { S.openStart = t; S.preSilent = t - S.lastVoiceT; S.sharp = -1; S.openSmile = 0; }
          S.openHeld += dt;
          // a smile that RISES with the opening (vs its own low point in the last 1.5 s) = a (silent) laugh;
          // a smile the user already wore is not
          let smn = smile; for (let i = 0; i < R.length; i += 3) smn = Math.min(smn, R[i + 2]);
          S.openSmile = Math.max(S.openSmile, smile - smn);
          if (S.sharp < 0 && jdev >= jA0) S.sharp = ramp(gain / jA0, 0.5, 0.9);   // reached the threshold by a quick opening
        } else S.openHeld = 0;
        const iAmp = ramp(jdev, jA0, jA0 + O.inhale.a1 - O.inhale.a0);
        const iHold = ramp(S.openHeld, O.inhale.h0, O.inhale.h1) * (1 - ramp(S.openHeld, O.inhale.stale, O.inhale.stale + 0.6));
        const sharp = S.openHeld > 0 && S.sharp >= 0 ? S.sharp : 0;
        const steady = (1 - ramp(S.jawVel, 0.5 * O.inhale.vel, O.inhale.vel)) * (1 - ramp(S.talk, 0.2, 0.4));   // laughing / chewing / mouthing move the jaw on
        const noSmile = 1 - 0.7 * ramp(S.openHeld > 0 ? S.openSmile : 0, O.inhale.smile * 0.4, O.inhale.smile);
        const fromSilence = 0.3 + 0.7 * ramp(S.preSilent, 0.3, 0.8);
        inhaleV = warm ? Math.sqrt(iAmp * iHold) * sharp * steady * noSmile * fromSilence : 0;
        P.i_amp = iAmp; P.i_hold = iHold; P.i_sharp = sharp; P.i_steady = steady; P.i_smile = 1 - noSmile; P.i_dev = jdev; P.i_a0 = jA0;
        if (inhaleV >= 0.35 && !av) S.inhaleVetoT = t;
        // talk over: voiced + a talking jaw, sustained (a backchannel is short), not laughter
        const toHold = ramp(voicedRun, O.talkOver.h0, O.talkOver.h1), toJaw = ramp(S.talk, 0.5 * O.talkOver.talk, O.talkOver.talk);
        talkOverV = Math.sqrt(toHold * toJaw) * (1 - 0.6 * ramp(sdev, 0.15, 0.35));
        P.o_hold = toHold; P.o_jaw = toJaw;
      } else {
        // dropout: holds survive a short gap (MediaPipe flicker), then reset
        if (t - S.faceLastT > 0.3) { S.browHeld = 0; S.leanHeld = 0; S.openHeld = 0; }
        eye = S.eye = lp(S.eye, 0, 0.4);
        const toHold = ramp(voicedRun, O.talkOver.h0 + 0.4, O.talkOver.h1 + 0.4);
        talkOverV = 0.6 * toHold;   // no face: voice alone is weaker evidence (echo, noise)
        P.o_hold = toHold; P.o_jaw = -1;
      }
      // hands
      const hb = enabled.beckon ? updateHands(f || {}, t, fresh) : {v: 0, parts: null};

      // ── context factors ──
      const quiet = heTalking ? 0 : ramp(silentFor, O.quiet.s0, O.quiet.s1);
      const avGap = av ? 0 : ramp(avSilentFor, O.avatarGap.s0, O.avatarGap.s1);
      const vetoed = t - S.inhaleVetoT < 1.0;
      const replyCtx = quiet * avGap * (vetoed ? 0.3 : 1) * (1 - 0.8 * ramp(S.talk, 0.3, 0.55));   // mouthing words = talking
      const eyeF = e0 => e0 + (1 - e0) * ramp(eye, 0.3, 0.8);
      const sc = {};
      sc.raised_brows = browV * eyeF(0.2) * replyCtx;
      sc.lean_in = leanV * eyeF(0.35) * replyCtx;
      sc.beckon = hb.v * eyeF(0.5) * replyCtx;
      // gaze after a pause: the user spoke, stopped, keeps eye contact through the pause
      let gz = 0;
      P.g = null;
      if (!voicedNow && S.turnVoiced >= O.gaze.minTurn * 0.5 && silentFor < O.gaze.maxPause && !av) {
        if (fresh) { S.pauseEyeSum += eye; S.pauseEyeN++; }
        const eyeMean = S.pauseEyeN ? S.pauseEyeSum / S.pauseEyeN : 0;
        const sN = ramp(silentFor, O.gaze.s0, O.gaze.s1), eN = ramp(Math.min(eyeMean, eye), 0.45, 0.85), tN = 0.5 + 0.5 * ramp(S.turnVoiced, O.gaze.minTurn * 0.5, O.gaze.minTurn * 2);
        const avN = S.avLastT > S.lastVoiceT ? ramp(avSilentFor, 0.5, 1.5) : 1;   // the avatar spoke during the user's pause → its reply, not the user's yield
        gz = Math.sqrt(sN * eN) * tN * avN * (vetoed ? 0.3 : 1) * (1 - 0.8 * ramp(S.talk, 0.3, 0.55)) * (fresh ? 1 : 0);
        P.g = [sN, eN, tN, avN];
      }
      sc.gaze_after_pause = gz;
      // floor bids: only while the avatar is speaking
      sc.inhale = av ? inhaleV * eyeF(0.3) : 0;
      sc.talk_over = av ? talkOverV * eyeF(0.6) : 0;

      // ── emission ──
      let events = null;
      let dmax = 0, cmax = null;
      for (const c in CUES) {
        if (!enabled[c]) { sc[c] = 0; continue; }
        const D = cl(sc[c]);
        sc[c] = D;
        if (D > dmax) { dmax = D; cmax = c; }
        if (!(ST.maxD[c] >= D)) ST.maxD[c] = D;
        const E = S.epi[c], type = CUES[c];
        if (D >= threshold) {
          E.low = 0;
          if (!E.active) {
            E.active = true;
            if (t - S.lastBid[type] >= O.refractory[type] && !(c === 'gaze_after_pause' && S.pauseFired)) {
              S.lastBid[type] = t;
              if (c === 'gaze_after_pause') S.pauseFired = true;
              const ev = {type: 'turn_bid', cue: c, deliberateness: r3(D), t: r3(t), bid: type,
                parts: Object.assign(partsOf(c, hb), {eye: r3(eye), quiet: r3(quiet), silent_s: r3(Math.min(silentFor, 99)), avatar_gap: r3(avGap), avatar_speaking: av})};
              (events || (events = [])).push(ev);
              ST.bids[c] = (ST.bids[c] || 0) + 1; ST.byType[type]++;
            }
          }
        } else if (E.active) {
          if (D < 0.5 * threshold) { E.low += dt; if (E.low >= O.rearm) { E.active = false; E.low = 0; } } else E.low = 0;
        }
      }
      S.scores = sc; S.hb = hb;
      events = events || [];
      if (onEvent) for (const ev of events) { try { onEvent(ev); } catch (e) { /* the page's handler must not break perception */ } }
      const ms = now() - t0;
      ST.frames++; ST.ms += ms; if (ms > ST.msMax) ST.msMax = ms;
      return {t, d: r3(dmax), cue: cmax, bid: cmax ? CUES[cmax] : null, scores: sc, events};
    }

    function partsOf(c, hb) {
      switch (c) {
        case 'raised_brows': return {amp: r3(P.b_amp), hold: r3(P.b_hold), dev: r3(P.b_dev), his_a0: r3(P.b_a0)};
        case 'lean_in': return {amp: r3(P.l_amp), hold: r3(P.l_hold), dev: r3(P.l_dev), his_a0: r3(P.l_a0)};
        case 'inhale': return {amp: r3(P.i_amp), hold: r3(P.i_hold), sharp: r3(P.i_sharp), steady: r3(P.i_steady), smile: r3(P.i_smile), dev: r3(P.i_dev), his_a0: r3(P.i_a0)};
        case 'talk_over': return {hold: r3(P.o_hold), jaw: P.o_jaw < 0 ? null : r3(P.o_jaw)};
        case 'beckon': return hb && hb.parts ? Object.assign({}, hb.parts) : {};
        case 'gaze_after_pause': return P.g ? {silence: r3(P.g[0]), eye: r3(P.g[1]), turn: r3(P.g[2]), avatar: r3(P.g[3])} : {};
      }
      return {};
    }

    return {
      frame,
      setThreshold(x) { if (isFinite(+x)) threshold = cl(+x, 0.05, 0.99); return threshold; },
      threshold: () => threshold,
      reset() { reset(); },
      stats() {
        return {frames: ST.frames, ms_per_frame: ST.frames ? r3(ST.ms / ST.frames * 1000) / 1000 : 0, ms_max: r3(ST.msMax),
          bids: Object.assign({}, ST.bids), by_type: Object.assign({}, ST.byType), max_d: Object.assign({}, ST.maxD), threshold};
      },
      state() {
        const parts = {};
        for (const c in CUES) parts[c] = partsOf(c, S.hb);
        return {t: S.lastT, threshold, scores: Object.assign({}, S.scores), parts, silent_s: S.lastT === null ? null : r3(S.lastT - S.lastVoiceT),
          eye: r3(S.eye), talk: r3(S.talk), base: {brow: S.bLo, brow_p90: S.bHi, jaw: S.jLo, jaw_p90: S.jHi, smile: S.sLo, lnz: S.lzBase, lnz_p90: S.lzHi},
          last_bid: Object.assign({}, S.lastBid)};
      },
    };
  }

  return {create, DEFAULTS, CUES};
});
