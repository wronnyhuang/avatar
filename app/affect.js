/* affect.js — the avatar's own FEELING as a sustained expressive state of face AND body (any feeling).
 *
 * The server's affect judge (app/affect.py) names the feeling of each reply as continuous dimensions — valence v, arousal a,
 * dominance d (-1..1), an intensity k (0.2..1) — plus optional FACETS (features the three dimensions cannot carry:
 * laugh, tilt, look_down, look_away, one_brow, nose, heavy_lids, tense) and a free-text label. Nothing here knows a
 * feeling by name: joy, grief, bashful pride, mock outrage or wistful awe all come out of the same formulas, so feelings
 * nobody designed for still get a coherent face and posture.
 *
 *   const A = AvatarAffectLib.create({rnd})
 *   A.set({v, a, d, intensity, facets:[...], label, hold_s, stage}, now)   // a new verdict (retargets smoothly)
 *   A.step(dt, {now, speaking, silentT, bodyFree}) → {w, wb, face:{p: value}, add:{p: offset}, energy, cam}
 *   A.state()  A.reset()
 *
 * Layering (motion.js): the state is applied AFTER the rules pipeline (idle life, the user's mirrored face, speech beats) and
 * BEFORE gestures, co-speech strokes, commands, held poses, keyframe jobs and the whole-body engine, so every one of
 * those plays over it; the lips stay lip sync's (a silent-only mouth). While the body works (a lesson at the board, an
 * object job, a move) only the face and the energy show (bodyFree = 0 fades the posture channels out).
 *
 * Time course: onset eases in over ~0.4 s (with an accent sized by intensity: a lift with a few happy bounces, a sink, a
 * recoil); the state HOLDS while the reply is spoken (at least hold_s), then decays over a few seconds to a faint mood
 * residue that fades over ~10 s — never a blink, never a snap. A newer verdict glides the dimensions to the new point.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarAffectLib = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));
  const lp = (x, y, dt, tau) => x + (y - x) * (1 - Math.exp(-dt / Math.max(1e-3, tau)));
  const P = x => Math.max(0, x), N = x => Math.max(0, -x);
  const FACETS = ['laugh', 'surprise', 'tilt', 'look_down', 'look_away', 'one_brow', 'nose', 'heavy_lids', 'tense'];

  // the readable face + posture for a point (v, a, d) at intensity k. Values are ABSOLUTE targets for the face's set
  // channels and OFFSETS for posture channels. Amplitudes were sized on renders of the home framing at 960x540 and
  // 480x270 (r2/joy/stills): what reads there, not more.
  function shape(v, a, d, k, F) {
    const pos = P(v), neg = Math.pow(N(v), 0.75), ar = P(a), calm = N(a), dom = P(d), sub = N(d);
    const g = 0.45 + 0.75 * k;                        // a mild feeling is a smaller version of the same shape (r2 a1: 0.3 + 0.7k read 'content' at 540p)
    const surpr = ar * (1 - Math.abs(v));            // aroused, not clearly good or bad: surprise / alarm / awe
    // R2: agency = dominance pushed up by arousal; a negative feeling either fights or yields (see brows)
    const ag = clamp(d + 0.5 * ar - 0.1, -1, 1), fight = neg * P(ag), yld = neg * P(0.15 - ag);
    const face = {}, add = {};
    // mouth corners: joy up (a clear curve at stream size), negative feelings flatten / turn down
    face.smile = clamp(0.1 + g * (1.05 * Math.pow(P(v - 0.12), 1.3) * (0.75 + 0.25 * P(1 - calm)) - 0.35 * neg) + (F.laugh ? 0.15 : 0) - (F.nose ? 0.12 : 0), 0, 1);
    // cheeks: a little lower-lid lift for warm joy (a big squint on bead eyes reads sleepy / smug); anger glares
    face.squint = clamp(g * (0.3 * pos * (0.6 + 0.4 * ar) + 0.45 * fight) + (F.laugh ? 0.4 : 0) + (F.heavy_lids ? 0.15 : 0) + (F.nose ? 0.35 : 0) - (F.surprise ? 0.2 : 0), 0, 0.75);
    // eyes open wide with arousal that is surprise / fear / delight
    face.eye_wide = clamp(g * (0.75 * surpr + 0.8 * ar * neg * (1 - dom) * (0.5 + sub)) + (F.surprise ? 0.6 : 0) - (F.laugh ? 0.3 : 0), 0, 1);
    // brows: up with arousal / joy / pleading (inner ends up with the knit = sad, worried); down + knit = anger
    // R2: a negative feeling either FIGHTS (anger, annoyance, disgust, outrage on the user's behalf: brows down, glare) or YIELDS
    // (sadness, worry, fear, shame: inner ends up) — agency = dominance pushed up by arousal (a4: 'sympathetic outrage' at
    // d 0.1 read as worried)
    const up = 0.5 * ar * (1 - Math.min(1, 2 * fight)) * (1 - 0.5 * neg) + 0.35 * pos + 0.6 * yld * (0.6 + sub) + (F.tense ? 0.1 : 0);
    const down = 2.6 * fight * (0.6 + 0.4 * ar);
    face.brow = clamp(g * (up - down) + (F.surprise ? 0.4 : 0), -0.8, 1.0);
    face.brow_knit = clamp(g * (1.25 * neg + (F.tense ? 0.25 : 0)) + (F.nose ? 0.35 : 0), 0, 0.95);   // R2: the knit is also Muse's frown (corners down)
    // upper lids: heavy when low-arousal (sadness, tiredness, contentment a little) or a facet
    face.lid = clamp(g * (0.22 * calm * (0.4 + neg) + 0.1 * neg * calm) + (F.heavy_lids ? 0.3 : 0), 0, 0.5);
    // a silent mouth: an open smile for big joy / laughter, a dropped jaw for surprise (lip sync owns it while talking)
    face.mouth = clamp(g * (0.22 * pos * ar + 0.35 * surpr * (1 - pos)) + (F.laugh ? 0.18 : 0) + (F.surprise ? 0.3 : 0), 0, 0.5);
    // gaze: down / away for submissive feelings (shy, sheepish, sad) and facets
    add.gaze_y = -g * (0.35 * sub * (0.4 + neg + 0.5 * pos) * (1 - 0.8 * ar)) - (F.look_down ? 0.45 : 0);
    add.gaze_x = F.look_away ? 0.35 : 0;
    // head: down with sadness / submission, up with pride / dominance and big joy; a tilt for warmth / curiosity / coyness
    add.head_pitch = g * (0.2 * yld * (0.5 + calm + sub) * (1 - 0.7 * ar) + 0.08 * fight + 0.1 * sub * (1 - ar) - 0.1 * dom * (1 - neg) - 0.06 * pos * ar) + (F.look_down ? 0.08 : 0) - (F.nose ? 0.06 : 0) - (F.surprise ? 0.05 : 0);
    add.head_roll = g * (0.05 * pos * sub + 0.04 * P(-Math.abs(a) + 0.5) * pos) + (F.tilt ? 0.13 : 0);
    // body (R2: sized on home-framing renders — the body is what reads at stream size, the face is ~60 px):
    // chest up / lean back with joy and pride, slump forward with sadness, lean in with anger, shrink back with fear
    const angr = fight * (0.4 + 0.6 * ar), fear = neg * sub * P(a + 0.2), low = neg * (0.3 + calm + 0.5 * sub) * (1 - 0.6 * ar);
    add.body_pitch = g * (-0.3 * pos * (0.3 + ar) - 0.28 * dom * (1 - neg) + 0.45 * low + 0.35 * angr - 0.35 * fear - 0.3 * surpr * (neg + 0.4));
    add.body_bounce = g * (0.06 * pos * ar - 0.45 * low - 0.25 * fear);   // sink into the knees
    add.shoulder = clamp(g * (0.9 * fear + 0.35 * angr + 0.2 * surpr * neg + 0.15 * neg * sub) + (F.tense ? 0.3 : 0), 0, 0.9);
    // arms: open, lifted and palms out for warm / excited feelings; on the hips for pride; drawn to the chest for fear and
    // tenderness; fists low and forward for anger; hanging for sadness
    const open = pos * (0.4 + 0.6 * ar) * (1 - 0.6 * dom * P(0.6 - ar));
    const akimbo = pos * dom * P(0.7 - ar), hug = pos * calm * (1 - dom) + 0.5 * neg * sub * calm;
    add.hand_x = g * (0.45 * open + 0.5 * akimbo - 0.5 * fear - 0.45 * hug + 0.15 * angr - 0.15 * low);
    add.hand_y = g * (0.5 * open + 0.05 * akimbo + 0.75 * fear + 0.4 * hug + 0.2 * angr - 0.15 * low + 0.35 * surpr);
    add.hand_z = g * (0.25 * open - 0.45 * akimbo + 0.45 * fear + 0.3 * hug + 0.3 * angr + 0.15 * surpr);
    add.hand_open = g * (0.6 * open + 0.4 * surpr - 0.25 * angr - 0.2 * akimbo + 0.2 * fear);
    // ears + tail: perk with arousal / joy, droop with sadness / fear, flatten back with anger
    add.ear = clamp(g * (0.55 * pos + 0.45 * ar * (1 - neg) - 0.9 * neg * (0.4 + calm + sub) - 0.4 * angr), -1, 1);
    add.tail_raise = g * (0.4 * pos + 0.2 * ar - 0.45 * neg);
    add.nose = F.nose ? 0.5 : 0;
    add.one_brow = F.one_brow ? 0.35 : 0;
    return {face, add};
  }

  function create(opt = {}) {
    const rnd = opt.rnd || Math.random;
    const S = {on: false, v: 0, a: 0, d: 0, k: 0, T: null, facets: {}, label: '', t0: -1e9, holdUntil: -1e9, w: 0, wb: 1, phase: 'idle',
      released: -1e9, accent: null, side: 1, n: 0, cam: 0, energy: 0, spokeSince: false};
    function set(j, now) {
      if (!j || typeof j !== 'object') return false;
      const v = clamp(+j.v || 0, -1, 1), a = clamp(+j.a || 0, -1, 1), d = clamp(+j.d || 0, -1, 1);
      const k = clamp(+j.intensity || 0.5, 0.2, 1);
      const facets = {}; for (const f of (Array.isArray(j.facets) ? j.facets : String(j.facets || '').split(/[\s,]+/))) if (FACETS.includes(f)) facets[f] = 1;
      const plain = Math.abs(v) < 0.15 && Math.abs(a) < 0.25 && !Object.keys(facets).length;
      const fresh = S.phase === 'idle' || S.phase === 'residue' || now - S.t0 > 20;
      S.T = {v, a, d, k};
      if (fresh) { S.v = v * 0.6; S.a = a * 0.6; S.d = d * 0.6; S.k = k * 0.8; }   // glide from near neutral (the residue's own dims are smaller)
      S.facets = facets; S.label = String(j.label || j.feel || '').slice(0, 40);
      S.t0 = now; S.n++; S.on = !plain;
      S.holdUntil = now + clamp(+j.hold_s || 5, 2.5, 14);
      S.phase = plain ? 'release' : 'on'; if (plain) S.released = now;
      S.side = rnd() < 0.5 ? -1 : 1;
      S.spokeSince = false;
      // the onset accent (one per new feeling, not per refinement of the same one): sized by intensity and arousal
      const big = k * (0.5 + 0.5 * Math.abs(a));
      const pos = P(v), neg = N(v);
      if (!plain && (fresh || j.stage !== 'reply' || !S.accent)) {
        if (pos > 0.35 && a > 0.25 && !facets.surprise) S.accent = {kind: 'lift', t0: now, amp: big, n: k >= 0.8 ? 3 : k >= 0.55 ? 2 : 1};
        else if (a > 0.45 && (pos < 0.35 || facets.surprise)) S.accent = {kind: 'recoil', t0: now, amp: big};
        else if (neg > 0.3 && a < 0.2) S.accent = {kind: 'sink', t0: now, amp: k};
        else S.accent = null;
      }
      return true;
    }
    function reset() { S.on = false; S.phase = 'idle'; S.w = 0; S.k = 0; S.accent = null; S.cam = 0; S.energy = 0; S.T = null; }
    function step(dt, c = {}) {
      const now = +c.now || 0;
      if (S.phase === 'idle' && S.w < 1e-4) return null;
      const T = S.T || {v: 0, a: 0, d: 0, k: 0};
      // dims glide to the newest verdict (a refinement never jumps)
      S.v = lp(S.v, T.v, dt, 0.25); S.a = lp(S.a, T.a, dt, 0.25); S.d = lp(S.d, T.d, dt, 0.25); S.k = lp(S.k, T.k, dt, 0.25);
      if (c.speaking) S.spokeSince = true;
      // phases: on (eases in, holds while it talks and for hold_s) -> release (a few seconds) -> residue (a faint mood) -> idle
      let wT;
      if (S.phase === 'on') {
        wT = 1;
        const talking = c.speaking || (+c.silentT || 0) < 1.2;
        if (now > S.holdUntil && !talking) { S.phase = 'release'; S.released = now; }
      }
      if (S.phase === 'release') { wT = 0.3; if (now - S.released > 4) S.phase = 'residue'; }
      if (S.phase === 'residue') { wT = 0; }
      const tau = S.phase === 'on' ? 0.2 : S.phase === 'release' ? 1.3 : 6;
      S.w = lp(S.w, wT, dt, tau);
      if (S.phase === 'residue' && S.w < 0.01) { S.phase = 'idle'; S.w = 0; }
      S.wb = lp(S.wb, c.bodyFree === false ? 0 : 1, dt, 0.3);
      const {face, add} = shape(S.v, S.a, S.d, S.k, S.facets);
      // the onset accent (offsets on top, its own envelope)
      const acc = {body_bounce: 0, body_pitch: 0, head_pitch: 0, brow: 0, eye_wide: 0, ear: 0, hand_y: 0, hand_x: 0, hand_z: 0, hand_open: 0};
      const A = S.accent;
      if (A) {
        const u = now - A.t0;
        if (A.kind === 'lift') {   // a rise onto the toes and a few small happy bounces (~2.4 Hz), brows flash
          const dur = 0.35 + A.n * 0.42;
          if (u < dur + 0.6) {
            const env = mj(u / 0.15) * (1 - mj((u - dur) / 0.3));
            const b = u < 0.35 ? Math.sin(Math.PI * u / 0.35) : 0.55 * Math.abs(Math.sin(Math.PI * (u - 0.35) / 0.42));
            acc.body_bounce = env * A.amp * 0.32 * b; acc.body_pitch = -env * A.amp * 0.12; acc.brow = env * A.amp * 0.3;
            acc.ear = env * A.amp * 0.5; acc.head_pitch = -env * A.amp * 0.05;
            // a big one throws the arms up (R2: joy has to be unmistakable at stream size); a smaller one lifts the paws
            const up = A.n >= 3 ? 1 : A.n === 2 ? 0.45 : 0.15, hold = mj(u / 0.3) * (1 - mj((u - dur - 0.1) / 0.45));
            acc.hand_y = A.amp * (0.15 * env * b + 0.75 * up * hold); acc.hand_x = 0.3 * A.amp * up * hold; acc.hand_open = 0.7 * A.amp * up * hold;
          } else S.accent = null;
        } else if (A.kind === 'recoil') {   // a quick draw back, eyes and brows flash, paws up, then it settles into the feeling
          if (u < 1.6) { const e = u < 0.18 ? mj(u / 0.18) : 1 - mj((u - 0.4) / 1.2); acc.body_pitch = -0.45 * A.amp * e; acc.head_pitch = -0.08 * A.amp * e;
            acc.eye_wide = 0.4 * A.amp * e; acc.brow = 0.4 * A.amp * e; acc.body_bounce = 0.1 * A.amp * e;
            acc.hand_y = 0.3 * A.amp * e; acc.hand_z = 0.15 * A.amp * e; acc.hand_open = 0.5 * A.amp * e; } else S.accent = null;
        } else if (A.kind === 'sink') {   // a slow sag a little past the held posture, then it settles back to it
          if (u < 3.2) { const e = u < 1.2 ? mj(u / 1.2) : 1 - mj((u - 1.2) / 2.0); acc.body_bounce = -0.15 * A.amp * e; acc.head_pitch = 0.07 * A.amp * e; acc.body_pitch = 0.1 * A.amp * e; }
          else S.accent = null;
        }
      }
      // a sustained lively bob for strong happy excitement, held through the reply and settling (never for mild feelings:
      // a gentle smile does not bounce), with a slowly varying amplitude so it is not a metronome
      const bobK = P(S.v) * P(S.a - 0.3) * P(S.k - 0.45) * 2.5;
      if (bobK > 0 && S.phase === 'on') {
        const age = now - S.t0, env = mj(age / 0.8) * (0.55 + 0.45 * Math.exp(-age / 4)) * (c.speaking ? 1 : 0.6);
        S.bobPh = (S.bobPh || 0) + dt * 2 * Math.PI * 1.7;
        const amp = Math.min(1, bobK) * env * (0.75 + 0.25 * Math.sin(0.37 * 2 * Math.PI * age));
        acc.body_bounce += 0.09 * amp * Math.abs(Math.sin(S.bobPh)); acc.ear += 0.15 * amp * Math.sin(S.bobPh);
        acc.hand_y += 0.06 * amp * Math.abs(Math.sin(S.bobPh + 0.6));
      }
      // energy: how lively the whole body moves (head motion, co-speech strokes); low for sad / calm, high for excited
      S.energy = S.w * S.k * (0.8 * S.a + 0.2 * S.v);
      // the camera eases in a little for strong feelings so the face reads at stream sizes (home framing only)
      S.cam = S.phase === 'on' ? clamp((S.k - 0.55) / 0.4, 0, 1) * clamp(Math.abs(S.v) + P(S.a), 0, 1) : 0;
      return {w: S.w, wb: S.wb, face, add, acc, energy: S.energy, cam: S.cam, side: S.side};
    }
    function state() {
      return {phase: S.phase, w: +S.w.toFixed(3), wb: +S.wb.toFixed(3), v: +S.v.toFixed(2), a: +S.a.toFixed(2), d: +S.d.toFixed(2), k: +S.k.toFixed(2),
        label: S.label, facets: Object.keys(S.facets), accent: S.accent ? S.accent.kind : null, energy: +S.energy.toFixed(3), cam: +S.cam.toFixed(3), n: S.n};
    }
    return {set, step, state, reset};
  }
  return {create, shape, FACETS};
});
