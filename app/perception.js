/* perception.js — phone-side perception for the live talking avatar.
 *
 * window.AvatarPerception = {createPerception, OneEuro, GEOM}   (also module.exports when `module` exists)
 *
 * createPerception({assetUrl:(name)=>url, container, onFeatures(f), onEvent(ev), rates:{face:18,hand:10,pose:6},
 *                   pose:true|false|'auto', preview:true, mesh:true, debug:false, canon:false, onCanon(c, src)})
 *   → {start():Promise, stop(), setPreview(bool), setMesh(bool), setCamera(deviceId):Promise, currentDevice(), latest(),
 *      stats(), events:[last 50], running}      (opts.deviceId = the camera to open; '' = the front camera)
 *
 * CANON (opt-in, learned reaction): canon:true needs app/react_feat.js loaded first (window.AvatarReactFeat, or
 *   opts.reactFeat). Computed from the RAW worker results (unfiltered, UNMIRRORED: bs mapped to the spec order by
 *   category name, the column-major matrix, pose lm + visibility, hand lm, the worker's bitmap w/h) — never from the
 *   filtered / mirrored values below. A canon frame is computed on every face result (present or not) and, while no
 *   face is present (no face result with a face within 0.5 s, or the face model not ready), on every pose / hand
 *   result; a pose or hand result more than 0.5 s from the frame's capture time counts as absent; frames whose capture
 *   time is not after the previous canon frame's are skipped (the timeline stays monotonic).
 *   c = {t: capture time ms (performance.now() timeline, the triggering frame's own stamp), seq (+1 per canon frame),
 *        motion: Float32Array(86) (cols 55-57 = RAW head translation in cm), mask: Uint8Array(4), tcm: [x,y,z] cm | null}
 *   → f.canon = the latest c on every emitted feature frame (dedupe by seq), and opts.onCanon(c, src) on every new one
 *     (src = the raw worker messages used: {face, pose, hand, names}; read-only, for tests). stats().canon = {n, ms}.
 *   canon:false (default): nothing of this runs and the feature stream is unchanged.
 * RAW (opt-in, unified mode): rawmp:true asks the workers for WORLD landmarks too (pose wlm, hand wlm) and calls
 *   opts.onRaw(m, msg) with every raw worker result (m = 'face'|'pose'|'hand'; msg as percept_worker.js posts it,
 *   read-only). rawmp:false (default): workers and features unchanged.
 *
 * Camera → ONE worker per model (vision/percept_worker.js: face = FaceLandmarker, hand = GestureRecognizer,
 * pose = PoseLandmarker lite), each fed its own ImageBitmap (320 px on the long side, aspect kept → 320×240 for
 * a 4:3 camera) at its own rate cap, skipping frames while that worker is busy. Results are merged here,
 * filtered (One-Euro), turned into a feature frame `f` (emitted at the face rate) + debounced discrete events.
 *
 * ───────────────────────────── GEOMETRY (all emitted values follow this) ─────────────────────────────
 * Input: the RAW front-camera frame (unmirrored). MediaPipe landmark x/y are raw-image coords in 0..1. The user
 * faces the phone; his own LEFT appears at LARGER raw x.
 * The corner preview is MIRRORED (video CSS scaleX(-1)) like a mirror: he sees his left on screen-left.
 * Every emitted coordinate is a VIEWER-SCREEN coordinate in that mirrored view:
 *     vx = 1 - rawx,  vy = rawy                          (0..1, used for mesh / lm / drawing)
 *     x  = (vx - 0.5) * 2 ∈ [-1,1]   -1 = screen-left = HIS OWN LEFT side
 *     y  = (0.5 - rawy) * 2 ∈ [-1,1] +1 = up
 *     ax = atan(x * tan(HFOV/2)), ay = atan(y * tan(VFOV/2))   angular position off the camera axis (rad),
 *          HFOV = 70°, VFOV = 55° for a 4:3 landscape frame (long side 70°; short side scaled by the aspect).
 *     "frame-width units" = distances measured in raw-image widths (y distances × H/W).
 * Head (from the facial transformation matrix, column-major 4×4; raw MediaPipe Euler:
 *     pitch = atan2(m6/sy, m10/sz), yaw = asin(-m2/sx), roll = atan2(m1/sx, m0/sx)), converted to the MIRRORED
 *     viewer convention: yaw > 0 = his nose turns toward SCREEN-RIGHT (= his own right);
 *     pitch > 0 = nose DOWN; roll > 0 = crown tilts toward SCREEN-RIGHT.
 *     Signs fixed EMPIRICALLY (tests/perception_smoke.cjs): each matrix angle is correlated with a landmark proxy in
 *     viewer coords (nose-tip offset from the eye midpoint for yaw/pitch, the screen-left→screen-right eye line
 *     angle for roll, iris offset inside the eye for gaze), and an hflip'ed copy of a clip must flip x, yaw, roll,
 *     gazeX and hand side. Result: SIGN.yaw = -1, SIGN.pitch = +1, SIGN.roll = +1 (raw MediaPipe → viewer).
 * Gaze (eye-in-head, ≈ rad) from blendshapes: gazeX > 0 = eyes toward screen-right, gazeY > 0 = up.
 *     MediaPipe's blendshape "Left"/"Right" (raw frame input) = his ANATOMICAL left/right eye (= the eye at larger /
 *     smaller raw x) — measured: gazeX_raw = ((InLeft+OutRight)-(OutLeft+InRight))/2 correlates +0.74 with the
 *     iris-in-eye offset toward screen-right (SIGN.bsLeftIsHisRight = false). In the mirror his left eye is on
 *     SCREEN-LEFT, so expr.blinkL = eyeBlinkLeft, expr.blinkR = eyeBlinkRight (blinkL/R are per SCREEN side).
 * Look direction: lookX = yaw + gazeX, lookY = -pitch + gazeY (rad, + = screen-right / up).
 * Attention: the camera is ABOVE the screen, so looking at the screen centre reads ≈ 8° down; the face's own
 *     off-axis position is compensated (to look at the camera from ax he must look -ax):
 *     expected look while attending = toward the lens..screen from where he sits: ex = -ax, ey = -ay - 0.07 (halfway
 *     between the lens and the screen centre 0.14 below it, so BOTH count). MediaPipe head pitch/yaw and blendshape
 *     gaze carry per-person offsets of ±0.1–0.2 rad (YouTube talkers into the lens read chin-up + eyes-down), so a
 *     per-person bias (bx, by) is learned from frames already near the expected look (τ 1.5 s for the first 3 s of
 *     a face, then 10 s; |bx| ≤ 0.08 (head yaw is reliable: a real head turn must not be absorbed), |by| ≤ 0.2;
 *     frames > 0.32 rad off are not learned from). Blendshape eye gaze (biased per person) is weighted 0.75:
 *     dx = yaw + 0.75 gazeX - ex - bx, dy = -pitch + 0.75 gazeY - ey - by,
 *     attention = exp(-(dx/0.20)² - (dy/0.28)²); face absent → 0. Tuned on the clips (scratch/motion/attn_sim.py):
 *     Sam talking into the lens 0.90–0.95 attending, emotion_happy 0.98, vcd_face_motion 1.0, a man looking at his
 *     laptop (watch_object_silent, head turned ~25°) 0.12.
 *     attending: on after > 0.55 for 250 ms, off after < 0.30 for 400 ms.
 * Hands: `side` = SCREEN side ('l'|'r') where the hand appears in the mirrored view, relative to his face midline
 *     when a face is tracked (else the screen centre); `handed` = his anatomical hand. MEASURED: this tasks-vision
 *     GestureRecognizer labels UNMIRRORED input anatomically (despite the old "assumes mirrored" doc note): on
 *     camera-filmed clips the hand on his left (screen-left 'l') is labelled 'Left' — sam_story 99 % of 246,
 *     sam_excited 98 % of 235 uncrossed hands, vcd_hand_motion 80 % — so handed = label (SIGN.handLabelInverted = false),
 *     smoothed per track by a score-weighted vote. Under an hflip both `side` and `handed` swap.
 *     `main` = the hand plausibly belongs to the MAIN face (FaceLandmarker numFaces:1 keeps tracking one face —
 *     sticky): within 2.5 face widths horizontally and palm ≥ 0.45 × IOD. thumbs_up / point / object_up need main;
 *     wave needs main when a face is tracked, and is allowed without any face (distant wavers).
 * Pose: l/r = SCREEN side of the SHOULDER of that arm (each arm keeps its own shoulder's side); pose.handed.l/r
 *     = which anatomical arm that is (landmark 11/13/15 = his left).
 * Distance z (m): matrix translation × 0.0105 (calibrated against the IOD estimate 0.063 m on frontal frames of 4
 *     clips; stays stable through head turns, where IOD collapses); IOD-based only when no matrix.
 * hflip check (vcd_face_motion, normal vs ffmpeg hflip, time-aligned on y+z): corr x -0.998, yaw -0.992, roll -0.951,
 *     gazeX -0.740, lookX -0.932; y +0.995, pitch +0.975, z +0.995. vcd_hand_motion: handed right 89/89 → left 149/151.
 *
 * FEATURE FRAME f (onFeatures, at the face rate; at the hand rate while the face model is not ready):
 *   {t, dt, face:{present:false} | {present, x, y, ax, ay, z, size (face width, frame widths), iod, yaw, pitch, roll,
 *     gazeX, gazeY, lookX, lookY, blend{51 names}, expr{smile, squint, browUp, browDown, eyeWide, sad, jawOpen, blink,
 *     blinkL, blinkR, pucker, cheekPuff}, vx, vy, chinY, topY, mesh:Float32Array(956) vx,vy 0..1, age},
 *    hands:[{side, handed, main, x, y, ax, ay, vx, vy, scale, vscale, open, ext[4], gesture, gscore, point:{dx,dy}|null,
 *     fore (index MCP→tip length / palm length; < 0.4 = pointing at the camera), lm:[21×[vx,vy,z]] (vx,vy 0..1 viewer,
 *     z MediaPipe relative depth), id, t, age}],
 *    pose:{present, t, shoulders, elbows, wrists:{l,r:{x,y,v}}, lean, width, handed:{l,r}, age} | null,
 *    motion, approach, attention, attending, events:[...], aspect (H/W of the inference frame)}
 * EVENTS (f.events + onEvent; ring `events` keeps the last 50): expression_change{from,to,intensity} (BIG changes, ≥ 3 s
 *     apart), nod{amp}, head_shake{amp} (createExpressionTracker below; face.xp = {state, intensity, scores, tilt, talk}),
 *     face_appear{away_s}, face_lost, attention_lost,
 *   look_back{away_s} (debounce 6 s), wave{side, handed, source:'hand'|'pose', main, continuing?, x, y, ax, ay},
 *   point{side, handed, dir{x,y}, at, x, y, ax, ay}, thumbs_up{side, handed, x, y, ax, ay}, object_up{side, handed, x, y,
 *   ax, ay}, approach_fast{source:'face'|'hand', rate (+ side, x, y, ax, ay for hand)}. x,y,ax,ay = the hand (palm centre)
 *   or, for pose waves, the waving wrist, in the same viewer convention as hands[].
 *   Status only (onEvent + ring, NOT in f.events): model_ready{model}, model_error{model, message}.
 * ────────────────────────────────────────────────────────────────────────────────────────────────────────────
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarPerception = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';

  const DEG = Math.PI / 180;
  const GEOM = {HFOV: 70 * DEG, VFOV: 55 * DEG, IOD_M: 0.063, SCREEN_BELOW_CAMERA: 0.14, TZ_TO_M: 0.0105};
  const TAN_LONG = Math.tan(GEOM.HFOV / 2);
  // raw MediaPipe → mirrored viewer convention (measured; see header + tests/perception_smoke.cjs)
  const SIGN = {yaw: -1, pitch: 1, roll: 1, gazeX: 1, bsLeftIsHisRight: false, handLabelInverted: false};
  const GAZE_GAIN = {x: 0.55, y: 0.45};      // blendshape units → ≈ radians
  const GAZE_Y_BIAS = 0;                     // measured rest bias is subject-dependent (+0.10 up on vcd_face_motion); none applied
  const CONN = {hand: null, face: null, pose: null};
  const BS_NAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
    'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
    'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'];
  const FILT = {pos: [1.0, 0.6], dist: [0.6, 0.3], ang: [1.2, 0.4], gaze: [1.5, 0.5], blend: [2.5, 0.2], hand: [1.5, 1.0]};
  const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
  // angular position (rad) of a viewer-normalized point for a frame of aspect k = H/W (long side = HFOV 70°)
  const angXY = (x, y, k) => { const tanH = k <= 1 ? Math.tan(35 * Math.PI / 180) : Math.tan(35 * Math.PI / 180) / k, tanV = k <= 1 ? Math.tan(35 * Math.PI / 180) * k : Math.tan(35 * Math.PI / 180);
    return [Math.round(Math.atan(x * tanH) * 1e4) / 1e4, Math.round(Math.atan(y * tanV) * 1e4) / 1e4]; };
  const clamp01 = v => clamp(v, 0, 1);
  const r4 = v => Math.round(v * 1e4) / 1e4;

  // ── One-Euro filter (Casiez et al. 2012). t in seconds. `.d` = low-passed derivative (units/s). ──
  function OneEuro(minCutoff, beta, dCutoff) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff || 1.0;
    this.x = null; this.d = 0; this.t = null;
  }
  OneEuro.prototype.alpha = function (cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };
  OneEuro.prototype.filter = function (v, t) {
    if (this.x === null || this.t === null || !(t > this.t)) {
      if (this.x === null) { this.x = v; this.d = 0; }
      this.t = t; return this.x;
    }
    const dt = Math.min(0.5, t - this.t);
    const dv = (v - this.x) / dt;
    this.d += this.alpha(this.dCutoff, dt) * (dv - this.d);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.d);
    this.x += this.alpha(cutoff, dt) * (v - this.x);
    this.t = t;
    return this.x;
  };
  OneEuro.prototype.reset = function () { this.x = null; this.d = 0; this.t = null; };
  function bank(kind) {          // lazily-created named filters of one kind
    const p = FILT[kind], m = {};
    return {get: k => m[k] || (m[k] = new OneEuro(p[0], p[1])), reset: () => { for (const k in m) m[k].reset(); }};
  }
  function lowpass(prev, v, dt, cutoff) {
    if (prev == null || !isFinite(prev)) return v;
    const a = 1 - Math.exp(-2 * Math.PI * cutoff * Math.max(1e-3, dt));
    return prev + a * (v - prev);
  }
  function pct(arr, p) {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
  }
  function corr(a, b) {
    const n = Math.min(a.length, b.length); if (n < 3) return 0;
    let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return sab / Math.sqrt(saa * sbb + 1e-12);
  }


  // ── his expressions, relative to HIS resting face ──────────────────────────────────────────────────────────────
  // MediaPipe blendshapes are biased per person / lighting / glasses (measured on RAVDESS: one actor rests at smile 0.28
  // and browDown 0.61, another at 0.00 / 0.14; eyeWide, mouthFrown, noseSneer, cheekSquint barely move on real faces), so
  // every channel is measured as a SIGNED deviation from a per-person neutral baseline: learned fast in the first 2 s
  // the face is seen (τ 0.7 s), then slowly: τ 6 s while he is neutral, τ 15 s once an expression has been held > 3 s
  // (never during the first 3 s of one, so a smile is not absorbed while it is being mirrored). States (readable, what the avatar mirrors): happy (smile up), frown (brows down, smile below rest,
  // mouth press/frown), sad (smile below rest + inner brows / brows down, no squint-up), surprised (brows up AND eyes
  // opening or jaw drop out of speech), raised_brows (brows up alone), neutral. Talking (jaw activity) gates the jaw and
  // mouth-shape cues. Head: tilt = roll − his slow roll baseline; nod / head_shake from pitch / yaw swings in 0.25–1.5 s.
  // Events (debounced): expression_change {from, to, intensity} on BIG arrivals (≥ 0.7; ≥ 5 s apart; the same one not
  // within 12 s), nod {amp}, head_shake {amp} (not while he is talking — speech moves the head). Causal; no look-ahead.
  function createExpressionTracker(opts) {
    opts = Object.assign({warm: 2.0, tauWarm: 0.7, tauSlow: 6, tauHeld: 15, enter: 0.45, exit: 0.25, big: 0.85, evGap: 8.0, sameGap: 20, evHold: 0.6, evStates: ['happy', 'sad', 'frown', 'surprised']}, opts || {});
    const CH = {smile: ['mouthSmileLeft', 'mouthSmileRight'], eyeSq: ['eyeSquintLeft', 'eyeSquintRight'], cheekSq: ['cheekSquintLeft', 'cheekSquintRight'],
      jaw: ['jawOpen'], browDown: ['browDownLeft', 'browDownRight'], innerUp: ['browInnerUp'], outerUp: ['browOuterUpLeft', 'browOuterUpRight'],
      eyeWide: ['eyeWideLeft', 'eyeWideRight'], frown: ['mouthFrownLeft', 'mouthFrownRight'], press: ['mouthPressLeft', 'mouthPressRight'],
      upperUp: ['mouthUpperUpLeft', 'mouthUpperUpRight'], sneer: ['noseSneerLeft', 'noseSneerRight']};
    const cl = (v, a = 0, b = 1) => v < a ? a : v > b ? b : v;
    const lpf = (p, x, dt, tau) => p + (x - p) * (1 - Math.exp(-dt / Math.max(1e-3, tau)));
    const T = {base: {}, dev: {}, seenT: 0, lastT: null, state: 'neutral', since: 0, peak: 0, cand: null, candT: 0, lowT: 0,
      lastEv: -99, sc: {}, jawPrev: null, talk: 0, rollBase: null, tilt: 0, hp: {p: null, y: null, r: null}, swings: {p: [], y: [], r: []},
      lastNod: -99, lastShake: -99, ext: {p: null, y: null, r: null}};
    const STATES = ['happy', 'frown', 'sad', 'surprised', 'raised_brows'];
    function update(face, t) {
      const events = [];
      const dt = T.lastT === null ? 1 / 15 : Math.max(0, Math.min(0.25, t - T.lastT)); T.lastT = t;
      if (!face || !face.present || !face.blend) {
        T.seenT = 0; T.jawPrev = null;
        T.hp = {p: null, y: null, r: null}; T.ext = {p: null, y: null, r: null}; T.swings = {p: [], y: [], r: []};   // re-acquisition is not a swing
        for (const k of STATES) T.sc[k] = lpf(T.sc[k] || 0, 0, dt, 0.25);
        if (T.state !== 'neutral') { T.state = 'neutral'; T.since = t; }
        return {state: T.state, intensity: 0, scores: Object.assign({}, T.sc), tilt: 0, talk: 0, events};
      }
      const B = face.blend, raw = {};
      for (const k in CH) { let v = 0; for (const n of CH[k]) v += +B[n] || 0; raw[k] = v / CH[k].length; }
      T.seenT += dt;
      // talking: jaw activity (speech moves the jaw 3–6 Hz; a held surprise jaw does not)
      const dj = T.jawPrev === null ? 0 : Math.abs(raw.jaw - T.jawPrev) / Math.max(dt, 1e-3); T.jawPrev = raw.jaw;
      T.talk = lpf(T.talk, cl(dj / 1.2), dt, 0.35);
      const talking = T.talk > 0.35;
      // the resting face drifts (head pose, light, the phone moving): adapt τ 6 s while neutral, τ 15 s once an
      // expression has been held > 3 s (people rarely hold one that long; a "stuck" reading is a new neutral)
      const learning = T.seenT < opts.warm ? opts.tauWarm : T.state === 'neutral' ? opts.tauSlow : (t - T.since > 3 ? opts.tauHeld : 0);
      for (const k in CH) {
        if (T.base[k] === undefined) T.base[k] = k === 'smile' ? Math.min(raw[k], 0.22) : raw[k];
        if (k === 'smile') {
          // smiles are upward excursions and people often START smiling (hello!): his resting smile is the LOWER envelope
          // (falls fast, rises slowly even in the warm-up) and never above 0.22 (neutral faces read 0.0–0.28 on RAVDESS)
          const tau = raw[k] < T.base[k] ? 0.5 : Math.max(learning || 0, 4) * (learning ? 1 : 0);
          if (tau) T.base[k] = lpf(T.base[k], raw[k], dt, tau);
          T.base[k] = Math.min(T.base[k], 0.22);
        } else if (learning) T.base[k] = lpf(T.base[k], raw[k], dt, learning);
        T.dev[k] = lpf(T.dev[k] ?? 0, raw[k] - T.base[k], dt, 0.08);
      }
      const d = T.dev, warm = cl((T.seenT - 0.5) / 1.0);   // no states before his resting face is known
      // while he talks, lips spread with the vowels: a smile then needs a bigger corner rise or the eyes squinting with it
      const smileDz = talking && d.eyeSq < 0.05 ? 0.16 : 0.08;
      const smileUp = cl((d.smile - smileDz) / 0.3), smileDown = cl((-d.smile - 0.06) / 0.18);
      const browsDown = cl((d.browDown - 0.05) / 0.18);
      const browsUp = cl((Math.max(d.innerUp, d.outerUp) + 0.8 * Math.max(0, -d.browDown) - 0.06) / 0.22);
      const eyesOpen = cl((-d.eyeSq - 0.04) / 0.14) * 0.6 + cl((d.eyeWide - 0.04) / 0.2) * 0.4;
      const jawDrop = talking ? 0 : cl((d.jaw - 0.08) / 0.25);
      const mouthNeg = cl(Math.max(d.frown / 0.15, d.press / 0.2, 0));
      const sc = {
        happy: smileUp,
        frown: cl(0.65 * browsDown + 0.3 * smileDown + 0.25 * mouthNeg + 0.15 * cl((d.eyeSq - 0.05) / 0.2) - 0.6 * smileUp - 0.5 * browsUp),
        sad: cl(0.6 * smileDown + 0.25 * browsDown + 0.25 * cl((d.innerUp - 0.05) / 0.2) - 0.2 * cl((d.eyeSq - 0.1) / 0.2) - 0.6 * smileUp),
        surprised: cl(browsUp * (0.55 + 0.45 * cl(eyesOpen + jawDrop)) * cl(eyesOpen * 1.5 + jawDrop * 1.5 + 0.25) - 0.5 * smileUp - 0.6 * browsDown),
        raised_brows: cl(browsUp - 0.6 * cl(eyesOpen + jawDrop) - 0.5 * browsDown - 0.3 * smileUp),
      };
      for (const k of STATES) T.sc[k] = lpf(T.sc[k] || 0, sc[k] * warm, dt, 0.1);
      // state machine: enter ≥ 0.45 for 0.15 s, leave < 0.25 for 0.3 s, switch when another is 0.2 higher for 0.2 s
      let best = null, bv = 0; for (const k of STATES) if (T.sc[k] > bv) { bv = T.sc[k]; best = k; }
      const cur = T.state, cv = cur === 'neutral' ? 0 : T.sc[cur];
      if (cur === 'neutral') {
        if (bv >= opts.enter) { if (T.cand !== best) { T.cand = best; T.candT = t; } else if (t - T.candT >= 0.15) setState(best, t, events); }
        else T.cand = null;
      } else {
        if (cv < opts.exit) { T.lowT += dt; if (T.lowT >= 0.3) setState('neutral', t, events); } else T.lowT = 0;
        if (T.state === cur && best !== cur && bv >= opts.enter && bv > cv + 0.2) {
          if (T.cand !== best) { T.cand = best; T.candT = t; } else if (t - T.candT >= 0.2) setState(best, t, events);
        }
        if (T.state !== 'neutral') T.peak = Math.max(T.peak, T.sc[T.state]);
      }
      // expression_change: a BIG arrival (peak ≥ 0.85) HELD ≥ 0.6 s (no flicker), ≥ 8 s apart, the same one not within 20 s;
      // raised brows (constant in speech) and returns to neutral are not events
      const P = T.evPending;
      if (P && T.state === P.to && t - P.t0 >= opts.evHold) {
        T.lastTo = T.lastTo || {};
        if (T.peak >= opts.big && t - T.lastEv >= opts.evGap && t - (T.lastTo[P.to] ?? -99) >= opts.sameGap) {
          T.lastEv = t; T.lastTo[P.to] = t;
          events.push({type: 'expression_change', t: Math.round(t * 1e4) / 1e4, from: P.from, to: P.to, intensity: Math.round(T.peak * 100) / 100});
        }
        T.evPending = null;
      } else if (P && T.state !== P.to) T.evPending = null;
      // head: tilt relative to his slow roll baseline; nods / shakes from swings
      const roll = +face.roll || 0, pitch = +face.pitch || 0, yaw = +face.yaw || 0; T.rollRaw = roll;
      T.rollBase = T.rollBase === null ? roll : lpf(T.rollBase, roll, dt, T.seenT < opts.warm ? 0.7 : 8);
      T.tilt = lpf(T.tilt, roll - T.rollBase, dt, 0.12);
      headSwings(pitch, yaw, t, dt, events);
      const inten = T.state === 'neutral' ? 0 : T.sc[T.state];
      return {state: T.state, intensity: Math.round(inten * 1000) / 1000, scores: Object.assign({}, T.sc), tilt: T.tilt, talk: T.talk, events,
        dev: Object.assign({}, d)};
    }
    function setState(to, t, events) {
      const from = T.state, fromPeak = T.peak, inten = to === 'neutral' ? 0 : T.sc[to];
      T.state = to; T.since = t; T.cand = null; T.lowT = 0; T.peak = inten;
      T.evPending = to !== 'neutral' && opts.evStates.includes(to) ? {to, from, t0: t} : null;
    }
    // swings: high-pass each angle (minus a 0.8 s mean), track turning points; a nod = pitch swing(s) of ≥ 0.06 rad
    // peak-to-peak, each half 0.12–0.75 s, pitch-dominant; a shake = ≥ 2 yaw reversals ≥ 0.08 rad within 1.5 s
    function headSwings(pitch, yaw, t, dt, events) {
      for (const [k, v] of [['p', pitch], ['y', yaw], ['r', T.rollRaw || 0]]) {
        const H = T.hp;
        H[k] = H[k] === null ? v : lpf(H[k], v, dt, 0.8);
        const x = v - H[k], E = T.ext[k] || (T.ext[k] = {dir: 0, last: x, lastT: t, ext: x, extT: t});
        const dir = x > E.ext + 0.012 ? 1 : x < E.ext - 0.012 ? -1 : 0;
        if (E.dir === 0) { if (dir) { E.dir = dir; } E.ext = dir ? E.ext : x; }
        if (E.dir === 1 && x > E.ext) { E.ext = x; E.extT = t; }
        else if (E.dir === -1 && x < E.ext) { E.ext = x; E.extT = t; }
        else if (E.dir && (E.dir === 1 ? x < E.ext - 0.02 : x > E.ext + 0.02)) {   // turning point confirmed
          T.swings[k].push({t: E.extT, v: E.ext}); E.dir = -E.dir; E.ext = x; E.extT = t;
          while (T.swings[k].length > 6) T.swings[k].shift();
        }
      }
      const P = T.swings.p, Y = T.swings.y, Rr = T.swings.r;
      // motion of the other axes over the same window (peak-to-peak of their turning points within it, or 0)
      const other = (arr, t0) => { const q = arr.filter(e => e.t >= t0 - 0.1); return q.length >= 2 ? Math.max(...q.map(e => e.v)) - Math.min(...q.map(e => e.v)) : 0; };
      if (P.length >= 3 && t - T.lastNod > 1.0 && T.seenT > 0.8) {
        // a nod goes AND comes back: three turning points, the first half ≥ 0.07 rad, the return ≥ half of it, each 0.12–0.75 s
        const a = P[P.length - 3], b = P[P.length - 2], c = P[P.length - 1], amp = Math.abs(b.v - a.v), back = Math.abs(c.v - b.v);
        const h1 = b.t - a.t, h2 = c.t - b.t, dom = Math.max(other(Y, a.t), other(Rr, a.t));
        if (amp >= 0.06 && amp <= 0.35 && back >= 0.4 * amp && h1 >= 0.12 && h1 <= 0.75 && h2 >= 0.1 && h2 <= 0.75 && t - c.t < 0.4 &&
            dom < 0.7 * amp && c.t > T.lastNodSwing && T.talk < 0.4) {
          T.lastNod = t; T.lastNodSwing = c.t;
          events.push({type: 'nod', t: Math.round(t * 1e4) / 1e4, amp: Math.round(amp * 1000) / 1000});
        }
      }
      if (Y.length >= 3 && t - T.lastShake > 1.5) {
        const c = Y.slice(-3), s1 = Math.abs(c[1].v - c[0].v), s2 = Math.abs(c[2].v - c[1].v), amp = Math.min(s1, s2), pp = Math.max(s1, s2), span = c[2].t - c[0].t;
        const dom = Math.max(other(P, c[0].t), 0.7 * other(Rr, c[0].t));
        // ≥ 2 reversals, balanced halves (each ≥ 0.4 × the bigger), each ≤ 0.8 s, ≥ 0.1 rad, ≤ 0.6 rad, yaw-dominant
        if (amp >= 0.1 && pp <= 0.6 && amp >= 0.4 * pp && span <= 1.5 && span >= 0.3 && c[1].t - c[0].t <= 0.8 && c[2].t - c[1].t <= 0.8 &&
            t - c[2].t < 0.4 && dom < 0.7 * pp && T.talk < 0.4 && (T.sc.happy || 0) < 0.3 && T.seenT > 0.8) {   // laughing shakes the head too
          T.lastShake = t; T.lastNod = Math.max(T.lastNod, t - 0.2);
          events.push({type: 'head_shake', t: Math.round(t * 1e4) / 1e4, amp: Math.round(amp * 1000) / 1000});
        }
      }
    }
    T.lastNodSwing = -99;
    return {update, state: T};
  }

  function createPerception(opts) {
    opts = opts || {};
    const assetUrl = opts.assetUrl || (n => 'vision/' + n);
    const abs = n => new URL(assetUrl(n), (typeof location !== 'undefined' ? location.href : undefined)).href;
    const rates = Object.assign({face: 18, hand: 10, pose: 6}, opts.rates || {});
    const onFeatures = opts.onFeatures || (() => {});
    const onEvent = opts.onEvent || (() => {});
    const debug = !!opts.debug;
    const poseOn = opts.pose === 'auto' || opts.pose == null ? (typeof navigator !== 'undefined' && (navigator.hardwareConcurrency || 0) >= 8) : !!opts.pose;
    const models = ['face', 'hand'].concat(poseOn ? ['pose'] : []);

    const S = {
      running: false, stream: null, video: null, box: null, canvas: null, ctx: null, previewOn: opts.preview !== false,
      meshOn: opts.mesh !== false, workers: {}, ready: {}, failed: {}, busy: {}, busySince: {}, credit: {}, lastTick: null,
      blobUrls: [], frameId: 0, lastFrameT: -1, lastRvfc: 0, rvfcHandle: 0, rafHandle: 0, drawHandle: 0, watchdog: 0,
      res: {face: [], hand: [], pose: []}, errors: [], events: [], pendingEvents: [],
      latest: null, lastEmitT: null, lastDraw: 0, w: 0, h: 0, vw: 0, vh: 0,
      // perception state
      face: null, faceRaw: null, faceT: -1, faceAbsentSince: null, faceSeenOnce: false, faceLostSent: true, facePresentSince: null,
      handRaw: null, handT: -1, tracks: [], nextTrack: 1, poseRaw: null, poseT: -1, pose: null,
      fp: bank('pos'), fd: bank('dist'), fa: bank('ang'), fg: bank('gaze'), fb: bank('blend'), fpose: bank('pos'),
      prevLnZ: null, prevZT: null, approach: 0, approachHi: null, vx: 0, vy: 0, prevFX: null, prevFY: null,
      attention: 0, attending: false, attHiSince: null, attLoSince: null, notAttSince: 0,
      motionCanvas: null, motionCtx: null, motionPrev: null, motionT: 0, motionFrame: 0, motionLm: 0, motion: 0,
      gesture: {}, debounce: {}, edges: {face: null, hand: null, pose: null}, poseWave: {l: null, r: null}
    };

    // ── events ──
    function emit(ev, cue) {
      S.events.push(ev); if (S.events.length > 50) S.events.shift();
      if (cue !== false) S.pendingEvents.push(ev);
      try { onEvent(ev); } catch (e) { noteErr('onEvent: ' + e.message); }
    }
    function debounced(key, t, gap) {
      if (S.debounce[key] != null && t - S.debounce[key] < gap) return false;
      S.debounce[key] = t; return true;
    }
    function noteErr(msg) { S.errors.push({t: now() / 1000, msg: String(msg).slice(0, 240)}); if (S.errors.length > 30) S.errors.shift(); }
    const now = () => performance.now();

    // ── canonical human numbers (opt-in; see CANON in the header). C === null → none of this runs ──
    const CANON_STALE_MS = 500;
    const C = opts.canon ? canonInit() : null;
    function canonInit() {
      const g = typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : {});
      let RF = opts.reactFeat || g.AvatarReactFeat || null;
      if (!RF && typeof require === 'function') { try { RF = require('./react_feat.js'); } catch (_) { RF = null; } }   // node / CommonJS
      if (!RF || typeof RF.canonFrame !== 'function') { noteErr('canon: react_feat.js (AvatarReactFeat) is not loaded — canon off'); return null; }
      return {RF, onCanon: typeof opts.onCanon === 'function' ? opts.onCanon : null, names: null, face: null, pose: null, hand: null,
        seq: 0, lastT: -Infinity, last: null, ms: 0, n: 0};
    }
    function canonReset() { C.face = null; C.pose = null; C.hand = null; C.lastT = -Infinity; C.last = null; }
    function canonStep(m, msg) {
      try {
        const t = msg.t;
        if (m === 'face') { if (msg.names) C.names = msg.names; C.face = msg; }
        else if (m === 'pose') C.pose = msg;
        else C.hand = msg;
        // pose / hand results trigger a frame only while no face is present (face frames carry the rest)
        if (m !== 'face' && S.ready.face && C.face && C.face.present && Math.abs(t - C.face.t) <= CANON_STALE_MS) return;
        if (!(t > C.lastT)) return;
        const t0 = now();
        const face = m === 'face' && msg.present && msg.bs && msg.mat ? msg : null;
        const pose = C.pose && C.pose.present && C.pose.lm && Math.abs(t - C.pose.t) <= CANON_STALE_MS ? C.pose : null;
        const hand = C.hand && Math.abs(t - C.hand.t) <= CANON_STALE_MS ? C.hand : null;
        const src = pose || hand || msg;
        const r = C.RF.canonFrame({face: face ? {bs: C.RF.bsToSpec(face.bs, C.names), matColMajor: face.mat} : null,
          pose: pose ? {lm: pose.lm} : null, hands: hand ? hand.hands || [] : [], w: src.w, h: src.h});
        C.seq++; C.lastT = t;
        const c = {t, seq: C.seq, motion: r.motion, mask: r.mask, tcm: r.tcm};
        C.last = c; C.ms += now() - t0; C.n++;
        if (C.onCanon) { try { C.onCanon(c, {face, pose, hand, names: C.names}); } catch (e) { noteErr('onCanon: ' + e.message); } }
      } catch (e) { noteErr('canon(' + m + '): ' + (e && e.message || e)); }
    }

    // ── preview DOM ──
    function buildPreview() {
      const c = opts.container;
      if (!c || typeof document === 'undefined') return;
      if (getComputedStyle(c).position === 'static') c.style.position = 'relative';
      const box = document.createElement('div');
      box.className = 'avp-preview';
      box.style.cssText = 'position:absolute;right:1.5%;bottom:2.5%;width:24%;aspect-ratio:4/3;border-radius:10px;' +
        'border:1.5px solid rgba(255,255,255,.55);overflow:hidden;background:#111;z-index:20;cursor:pointer;' +
        'box-shadow:0 2px 10px rgba(0,0,0,.35);touch-action:manipulation;-webkit-tap-highlight-color:transparent;';
      const v = document.createElement('video');
      v.muted = true; v.playsInline = true; v.autoplay = true; v.setAttribute('playsinline', ''); v.setAttribute('muted', '');
      v.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);';
      box.appendChild(v); c.appendChild(box);
      S.box = box; S.video = v;
      if (opts.mesh !== false) {        // mesh:false = the plain self-view (no landmark wireframe, no tap toggle)
        const cv = document.createElement('canvas');
        cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
        box.appendChild(cv); S.canvas = cv; S.ctx = cv.getContext('2d');
        box.addEventListener('click', e => { e.stopPropagation(); setMesh(!S.meshOn); });
      }
      applyPreview();
    }
    function applyPreview() {
      if (!S.box) return;
      // opacity, not display:none — some browsers stop video-frame callbacks for undisplayed videos
      S.box.style.opacity = S.previewOn && S.running ? '1' : '0';
      S.box.style.pointerEvents = S.previewOn && S.running ? 'auto' : 'none';
      if (!S.running) S.box.style.display = 'none'; else S.box.style.display = 'block';
    }
    function setPreview(on) { S.previewOn = !!on; applyPreview(); }
    function setMesh(on) { S.meshOn = !!on; if (!S.meshOn && S.ctx) S.ctx.clearRect(0, 0, S.canvas.width, S.canvas.height); }

    // ── assets ──
    function fetchBlobUrl(name, type, ms) {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), ms);
      return fetch(abs(name), {signal: ctl.signal, cache: 'force-cache'}).then(r => {
        if (!r.ok) throw Error(name + ' HTTP ' + r.status);
        return r.arrayBuffer();
      }).then(buf => {
        const u = URL.createObjectURL(new Blob([buf], {type}));
        S.blobUrls.push(u); return u;
      }).catch(e => { throw Error(e && e.name === 'AbortError' ? name + ' timed out after ' + ms / 1000 + ' s' : String(e && e.message || e)); })
        .finally(() => clearTimeout(timer));
    }
    async function startWorkers() {
      let shared;
      try {
        const [bundle, loader, wasm] = await Promise.all([
          fetchBlobUrl('vision_bundle.js', 'text/javascript', 120000),
          fetchBlobUrl('vision_wasm_internal.js', 'text/javascript', 120000),
          fetchBlobUrl('vision_wasm_internal.wasm', 'application/wasm', 120000)]);
        shared = {bundle, loader, wasm};
      } catch (e) {
        for (const m of models) markFailed(m, 'runtime: ' + e.message);
        return;
      }
      if (!S.running) return;
      const file = {face: 'face_landmarker.task', hand: 'gesture_recognizer.task', pose: 'pose_landmarker_lite.task'};
      for (const m of models) {
        let w;
        try { w = new Worker(abs('percept_worker.js')); } catch (e) { markFailed(m, 'worker: ' + e.message); continue; }
        S.workers[m] = w; S.busy[m] = false; S.credit[m] = 1;
        const guard = setTimeout(() => { if (!S.ready[m] && S.workers[m] === w) { markFailed(m, 'model load exceeded 180 s'); } }, 180000);
        w.onmessage = e => onWorker(m, e.data, guard);
        w.onerror = e => { e.preventDefault && e.preventDefault(); if (!S.ready[m]) { clearTimeout(guard); markFailed(m, 'worker error: ' + (e.message || 'load failed')); } else { S.busy[m] = false; noteErr(m + ': ' + (e.message || 'worker error')); } };
        w.postMessage(Object.assign({type: 'init', model: m, assets: {bundle: shared.bundle, loader: shared.loader, wasm: shared.wasm, model: abs(file[m])}}, opts.rawmp ? {world: true} : {}));
      }
    }
    function markFailed(m, msg) {
      S.failed[m] = msg; S.ready[m] = false; noteErr(m + ': ' + msg);
      if (S.workers[m]) { try { S.workers[m].terminate(); } catch (_) {} delete S.workers[m]; }
      emit({type: 'model_error', model: m, message: msg, t: now() / 1000}, false);
    }
    function onWorker(m, msg, guard) {
      if (!S.running) return;
      if (msg.type === 'ready') {
        clearTimeout(guard); S.ready[m] = true; S.edges[m] = msg.edges || null;
        emit({type: 'model_ready', model: m, t: now() / 1000}, false);
        if (models.every(x => S.ready[x] || S.failed[x])) for (const u of S.blobUrls.splice(0)) URL.revokeObjectURL(u);
        return;
      }
      if (msg.type === 'error') {
        if (msg.fatal) { clearTimeout(guard); markFailed(m, msg.message); } else { S.busy[m] = false; noteErr(m + ': ' + msg.message); }
        return;
      }
      if (msg.type !== 'result') return;
      S.busy[m] = false;
      if (msg.skipped) return;
      const recv = now();
      const q = S.res[m]; q.push({recv, infer: msg.infer, lat: recv - msg.t}); if (q.length > 150) q.shift();
      if (C) canonStep(m, msg);
      if (opts.rawmp && typeof opts.onRaw === 'function') { try { opts.onRaw(m, msg); } catch (e) { noteErr('onRaw(' + m + '): ' + (e && e.message || e)); } }
      try {
        if (m === 'face') onFace(msg);
        else if (m === 'hand') onHand(msg);
        else onPose(msg);
        const emitter = S.ready.face ? 'face' : S.ready.hand ? 'hand' : 'pose';
        if (m === emitter) emitFrame(msg.t / 1000);
      } catch (e) { noteErr('features(' + m + '): ' + (e && e.stack || e)); }
    }

    // ── frame pump ──
    function inferSize() {
      const vw = S.video.videoWidth, vh = S.video.videoHeight;
      if (!vw || !vh) return null;
      if (vw >= vh) return [320, Math.max(2, Math.round(320 * vh / vw))];
      return [Math.max(2, Math.round(320 * vw / vh)), 320];
    }
    function tick(ts) {
      if (!S.running) return;
      if (!(ts > S.lastFrameT)) ts = S.lastFrameT + 0.5;
      S.lastFrameT = ts;
      const v = S.video;
      if (v.readyState < 2) return;
      if (S.vw !== v.videoWidth || S.vh !== v.videoHeight) {
        S.vw = v.videoWidth; S.vh = v.videoHeight;
        if (S.box && S.vw && S.vh) S.box.style.aspectRatio = S.vw + '/' + S.vh;
      }
      const sz = inferSize(); if (!sz) return;
      const dtf = S.lastTick == null ? 0 : (ts - S.lastTick) / 1000; S.lastTick = ts;
      for (const m of models) {
        if (!S.ready[m]) continue;
        S.credit[m] = Math.min(1.5, (S.credit[m] || 0) + dtf * rates[m]);
        if (S.busy[m]) { if (now() - S.busySince[m] > 4000) { S.busy[m] = false; noteErr(m + ': busy watchdog reset'); } else continue; }
        if (S.credit[m] < 1) continue;
        S.credit[m] -= 1; S.busy[m] = true; S.busySince[m] = now();
        const id = ++S.frameId, w = S.workers[m];
        createImageBitmap(v, {resizeWidth: sz[0], resizeHeight: sz[1], resizeQuality: 'low'}).then(bmp => {
          if (!S.running || S.workers[m] !== w) { bmp.close(); S.busy[m] = false; return; }
          w.postMessage({type: 'frame', bmp, t: ts, id}, [bmp]);
        }).catch(e => { S.busy[m] = false; noteErr(m + ' frame: ' + e.message); });
      }
      motionTick(ts);
    }
    function onVideoFrame(nowTs, meta) {
      if (!S.running) return;
      S.lastRvfc = now();
      S.rvfcHandle = S.video.requestVideoFrameCallback(onVideoFrame);
      const ts = meta && isFinite(meta.captureTime) && meta.captureTime > 0 && Math.abs(meta.captureTime - nowTs) < 1000 ? meta.captureTime : nowTs;
      tick(ts);
    }
    let rafLastVT = -1;
    function rafPump() {      // fallback when requestVideoFrameCallback is missing or stalls
      if (!S.running) return;
      S.rafHandle = requestAnimationFrame(rafPump);
      if (S.video.requestVideoFrameCallback && now() - S.lastRvfc < 400) return;
      const vt = S.video.currentTime;
      if (vt === rafLastVT) return;
      rafLastVT = vt; tick(now());
    }

    // ── motion energy (32×24 gray thumbnail, ≤ 15 fps) ──
    function motionTick(ts) {
      if (ts - S.motionT < 66) return;
      const dt = S.motionT ? Math.min(0.5, (ts - S.motionT) / 1000) : 0.066; S.motionT = ts;
      if (!S.motionCanvas) {
        S.motionCanvas = document.createElement('canvas'); S.motionCanvas.width = 32; S.motionCanvas.height = 24;
        S.motionCtx = S.motionCanvas.getContext('2d', {willReadFrequently: true});
      }
      try {
        S.motionCtx.drawImage(S.video, 0, 0, 32, 24);
        const d = S.motionCtx.getImageData(0, 0, 32, 24).data, g = new Float32Array(768);
        for (let i = 0, j = 0; i < 768; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
        if (S.motionPrev) {
          let s = 0; for (let i = 0; i < 768; i++) s += Math.abs(g[i] - S.motionPrev[i]);
          const mad = s / 768;                                   // mean abs diff, 0..255 gray levels
          const raw = clamp01((mad - 1.2) / 14);                 // ~1 level = sensor noise; 15 = whole-frame motion
          S.motionFrame = lowpass(S.motionFrame, raw, dt, 3.0);
        }
        S.motionPrev = g;
      } catch (e) { /* tainted / not ready */ }
    }

    // ── face ──
    function onFace(msg) {
      const t = msg.t / 1000;
      S.faceT = t; S.w = msg.w; S.h = msg.h;
      if (msg.names) S.bsNames = msg.names;
      if (!msg.present) {
        S.faceRaw = null; S.face = {present: false};
        if (S.faceAbsentSince == null) { S.faceAbsentSince = t; if (S.xtrack) S.xtrack.update({present: false}, t); }
        if (!S.faceLostSent && t - S.faceAbsentSince >= 0.6) { S.faceLostSent = true; emit({type: 'face_lost', t: r4(t)}); }
        S.facePresentSince = null;
        resetFaceFilters();
        updateAttention(t, 0);
        return;
      }
      const W = msg.w, H = msg.h, k = H / W, lm = msg.lm, P = i => lm[3 * i], Q = i => lm[3 * i + 1];
      const landscape = W >= H, tanH = landscape ? TAN_LONG : TAN_LONG * W / H, tanV = landscape ? TAN_LONG * H / W : TAN_LONG;
      if (S.faceAbsentSince != null || !S.faceSeenOnce) {
        const away = S.faceSeenOnce ? t - S.faceAbsentSince : Infinity;
        if (!S.faceSeenOnce || away >= 1.0) emit({type: 'face_appear', t: r4(t), away_s: isFinite(away) ? r4(away) : null});
        S.faceSeenOnce = true;
      }
      S.faceAbsentSince = null; S.faceLostSent = false;
      if (S.facePresentSince == null) S.facePresentSince = t;
      // eye centres from the corners (gaze-independent); A = landmarks 33/133 (raw image-left eye = his RIGHT eye)
      const ax_ = (P(33) + P(133)) / 2, ay_ = (Q(33) + Q(133)) / 2, bx_ = (P(362) + P(263)) / 2, by_ = (Q(362) + Q(263)) / 2;
      const iod = Math.hypot(ax_ - bx_, (ay_ - by_) * k);                 // frame-width units
      const rawCX = (ax_ + bx_) / 2, rawCY = (ay_ + by_) / 2;
      let xmin = 1, xmax = 0; for (let i = 0; i < 468; i++) { const v = P(i); if (v < xmin) xmin = v; if (v > xmax) xmax = v; }
      // head pose
      const m = msg.mat;
      let yawR = 0, pitchR = 0, rollR = 0, tz = null;
      if (m) {
        const sx = Math.hypot(m[0], m[1], m[2]), sy = Math.hypot(m[4], m[5], m[6]), sz = Math.hypot(m[8], m[9], m[10]);
        pitchR = Math.atan2(m[6] / sy, m[10] / sz);
        yawR = Math.asin(clamp(-m[2] / sx, -1, 1));
        rollR = Math.atan2(m[1] / sx, m[0] / sx);
        tz = -m[14];
      }
      const yaw0 = SIGN.yaw * yawR, pitch0 = SIGN.pitch * pitchR, roll0 = SIGN.roll * rollR;
      const iodC = iod / Math.max(0.5, Math.cos(yawR));
      // distance: the matrix translation (cm in MediaPipe's virtual camera) is stable through head turns; the IOD estimate
      // collapses at |yaw| > 45° (fake "approach" when the head turns back — watch_object_silent). Calibrated on frontal
      // frames of 4 clips: IOD-distance ≈ tz × 0.0103..0.0111 → z = 0.0105 × tz; IOD only as the fallback.
      const z0 = tz != null && tz > 1 ? GEOM.TZ_TO_M * tz : GEOM.IOD_M * (0.5 / tanH) / Math.max(1e-3, iodC);
      // blendshapes
      const names = S.bsNames || BS_NAMES, bs = msg.bs, blend = {};
      for (let i = 0; i < bs.length && i < names.length; i++) { if (names[i] === '_neutral') continue; blend[names[i]] = clamp01(S.fb.get(names[i]).filter(bs[i], t)); }
      const B = n => blend[n] || 0;
      // gaze: A = (InLeft + OutRight) - (OutLeft + InRight) > 0 = eyes toward HIS right if "Left" = his left eye
      const gxA = ((B('eyeLookInLeft') + B('eyeLookOutRight')) - (B('eyeLookOutLeft') + B('eyeLookInRight'))) / 2;
      const gyA = ((B('eyeLookUpLeft') + B('eyeLookUpRight')) - (B('eyeLookDownLeft') + B('eyeLookDownRight'))) / 2 + GAZE_Y_BIAS;
      const blinkAvg = (B('eyeBlinkLeft') + B('eyeBlinkRight')) / 2;
      let gx0 = SIGN.gazeX * (SIGN.bsLeftIsHisRight ? -1 : 1) * gxA * GAZE_GAIN.x;
      let gy0 = gyA * GAZE_GAIN.y;
      const prevFace = S.face && S.face.present ? S.face : null;
      if (blinkAvg > 0.45 && prevFace) { gy0 = prevFace.gazeY; gx0 = prevFace.gazeX; }   // lids closing fake "look down"
      // filtered values
      const x = S.fp.get('x').filter((0.5 - rawCX) * 2, t), y = S.fp.get('y').filter((0.5 - rawCY) * 2, t);
      const z = S.fd.get('z').filter(z0, t);
      const yaw = S.fa.get('yaw').filter(yaw0, t), pitch = S.fa.get('pitch').filter(pitch0, t), roll = S.fa.get('roll').filter(roll0, t);
      const gazeX = S.fg.get('gx').filter(gx0, t), gazeY = S.fg.get('gy').filter(gy0, t);
      const dtF = S.prevZT == null ? 0 : t - S.prevZT;
      // velocities from filtered values (light 4 Hz smoothing)
      if (S.prevFX != null && dtF > 0) {
        S.vx = lowpass(S.vx, (x - S.prevFX) / dtF, dtF, 4); S.vy = lowpass(S.vy, (y - S.prevFY) / dtF, dtF, 4);
      } else { S.vx = 0; S.vy = 0; }
      S.prevFX = x; S.prevFY = y;
      const lnz = Math.log(Math.max(0.05, z));
      if (S.prevLnZ != null && dtF > 0) S.approach = lowpass(S.approach, -(lnz - S.prevLnZ) / dtF, dtF, 3);
      else S.approach = 0;
      S.prevLnZ = lnz; S.prevZT = t;
      const ax = Math.atan(x * tanH), ay = Math.atan(y * tanV);
      const lookX = yaw + gazeX, lookY = -pitch + gazeY;
      const L = SIGN.bsLeftIsHisRight;   // MediaPipe "Left" eye = his right = SCREEN-RIGHT in the mirror
      const expr = {
        smile: clamp01((B('mouthSmileLeft') + B('mouthSmileRight')) / 2),
        squint: clamp01((B('eyeSquintLeft') + B('eyeSquintRight')) / 2),
        browUp: clamp01(Math.max(B('browInnerUp'), (B('browOuterUpLeft') + B('browOuterUpRight')) / 2)),
        browDown: clamp01((B('browDownLeft') + B('browDownRight')) / 2),
        eyeWide: clamp01((B('eyeWideLeft') + B('eyeWideRight')) / 2),
        sad: 0, jawOpen: clamp01(B('jawOpen')), blink: clamp01(blinkAvg),
        blinkL: clamp01(L ? B('eyeBlinkRight') : B('eyeBlinkLeft')), blinkR: clamp01(L ? B('eyeBlinkLeft') : B('eyeBlinkRight')),
        pucker: clamp01(B('mouthPucker')), cheekPuff: clamp01(B('cheekPuff'))
      };
      expr.sad = clamp01(0.8 * (B('mouthFrownLeft') + B('mouthFrownRight')) / 2 + 0.5 * B('browInnerUp') * (1 - expr.browUp * 0.5) - 0.6 * expr.smile - 0.05);
      const mesh = new Float32Array(478 * 2);
      const nPts = Math.min(478, lm.length / 3);
      for (let i = 0; i < nPts; i++) { mesh[2 * i] = 1 - lm[3 * i]; mesh[2 * i + 1] = lm[3 * i + 1]; }
      const face = {present: true, x: r4(x), y: r4(y), ax: r4(ax), ay: r4(ay), z: r4(z), size: r4(xmax - xmin), iod: r4(iod),
        yaw: r4(yaw), pitch: r4(pitch), roll: r4(roll), gazeX: r4(gazeX), gazeY: r4(gazeY), lookX: r4(lookX), lookY: r4(lookY),
        blend, expr, vx: r4(S.vx), vy: r4(S.vy), chinY: r4((0.5 - Q(152)) * 2), topY: r4((0.5 - Q(10)) * 2),
        matrix: m && m.length >= 16 ? Array.from(m, v => Math.round(v * 1e4) / 1e4) : null,   // raw MediaPipe 4×4 (the learned prior's input)
        mesh, age: 0};
      if (debug) face.dbg = faceProxies(lm, k, {yawR, pitchR, rollR, tz, gxA, gyA});
      // his expression relative to his resting face (+ nod / head_shake / expression_change events, debounced)
      S.xtrack = S.xtrack || createExpressionTracker();
      const xr = S.xtrack.update(face, t);
      face.xp = {state: xr.state, intensity: xr.intensity, scores: xr.scores, tilt: r4(xr.tilt), talk: r4(xr.talk)};
      for (const ev of xr.events) emit(ev);
      S.face = face; S.faceRaw = mesh;
      {
        // eye gaze from blendshapes is the least reliable part → weighted 0.75 inside attention
        const ex = -ax, ey = -ay - 0.5 * GEOM.SCREEN_BELOW_CAMERA, dx0 = yaw + 0.75 * gazeX - ex, dy0 = -pitch + 0.75 * gazeY - ey;
        const B0 = S.attBias || (S.attBias = {x: 0, y: 0});
        const seen = S.facePresentSince != null ? t - S.facePresentSince : 0, dtA = Math.max(0, Math.min(0.2, t - (S.attBiasT ?? t)));
        S.attBias.t = t; S.attBiasT = t;
        if (Math.hypot(dx0 - B0.x, dy0 - B0.y) < 0.32 && dtA > 0) {
          const tau = seen < 3 ? 1.5 : 10, k = 1 - Math.exp(-dtA / tau);
          B0.x = Math.max(-0.08, Math.min(0.08, B0.x + (dx0 - B0.x) * k)); B0.y = Math.max(-0.2, Math.min(0.2, B0.y + (dy0 - B0.y) * k));
        }
        updateAttention(t, Math.exp(-Math.pow((dx0 - B0.x) / 0.20, 2) - Math.pow((dy0 - B0.y) / 0.28, 2)));
      }
      // approach_fast (face): sustained > 0.9/s for 120 ms, face tracked ≥ 0.35 s
      if (S.approach > 0.9 && t - S.facePresentSince > 0.35 && Math.abs(yaw) < 0.7) {
        if (S.approachHi == null) S.approachHi = t;
        if (t - S.approachHi >= 0.12 && debounced('approach_fast', t, 1.5)) emit({type: 'approach_fast', t: r4(t), source: 'face', rate: r4(S.approach)});
      } else S.approachHi = null;
    }
    function faceProxies(lm, k, raw) {     // landmark-only cross-checks in VIEWER coords (debug / sign tests)
      const V = i => [1 - lm[3 * i], lm[3 * i + 1] * k];   // vx, vy-down in frame-width units
      const eA = [(V(33)[0] + V(133)[0]) / 2, (V(33)[1] + V(133)[1]) / 2], eB = [(V(362)[0] + V(263)[0]) / 2, (V(362)[1] + V(263)[1]) / 2];
      const [eL, eR] = eA[0] < eB[0] ? [eA, eB] : [eB, eA];         // screen-left / screen-right eye
      const iod = Math.hypot(eR[0] - eL[0], eR[1] - eL[1]) || 1e-3, mid = [(eL[0] + eR[0]) / 2, (eL[1] + eR[1]) / 2], nose = V(1);
      const irisOff = (ir, a, b) => { const c = [(V(a)[0] + V(b)[0]) / 2]; return (V(ir)[0] - c[0]) / (Math.abs(V(a)[0] - V(b)[0]) || 1e-3); };
      return {
        yawP: r4((nose[0] - mid[0]) / iod),                         // + = nose toward screen-right
        pitchP: r4((nose[1] - mid[1]) / iod),                       // + = nose lower (down)
        rollP: r4(Math.atan2(eR[1] - eL[1], eR[0] - eL[0])),        // + = screen-right eye lower = crown to screen-right
        gazeP: r4((irisOff(468, 33, 133) + irisOff(473, 362, 263)) / 2),   // + = irises toward screen-right
        eyeAscreen: eA[0] < eB[0] ? 'l' : 'r',                       // screen side of landmarks 33/133 (his right eye)
        yawR: r4(raw.yawR), pitchR: r4(raw.pitchR), rollR: r4(raw.rollR), tz: raw.tz == null ? null : r4(raw.tz), gxA: r4(raw.gxA), gyA: r4(raw.gyA)
      };
    }
    function resetFaceFilters() {
      S.fp.reset(); S.fd.reset(); S.fa.reset(); S.fg.reset(); S.fb.reset();
      S.prevLnZ = null; S.prevZT = null; S.approach = 0; S.approachHi = null; S.vx = 0; S.vy = 0; S.prevFX = null; S.prevFY = null;
    }
    function updateAttention(t, a) {
      S.attention = a;
      if (a > 0.55) { S.attLoSince = null; if (S.attHiSince == null) S.attHiSince = t; } else S.attHiSince = null;
      if (a < 0.30) { if (S.attLoSince == null) S.attLoSince = t; } else if (a > 0.30) S.attLoSince = null;
      if (!S.attending && S.attHiSince != null && t - S.attHiSince >= 0.25) {
        S.attending = true;
        const away = t - (S.notAttSince || t);
        if (S.notAttSince && away >= 2.0 && S.everAttended && debounced('look_back', t, 6.0)) emit({type: 'look_back', t: r4(t), away_s: r4(away)});
        S.everAttended = true;
      } else if (S.attending && S.attLoSince != null && t - S.attLoSince >= 0.4) {
        S.attending = false; S.notAttSince = t;
        emit({type: 'attention_lost', t: r4(t)});
      }
      if (!S.attending && !S.notAttSince) S.notAttSince = t;
    }

    // ── hands ──
    const FINGERS = [[5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]];
    const TRACK_KEEP = 0.6;   // s a hand track survives missed detections (hands run 6-10 fps and flicker)
    function onHand(msg) {
      const t = msg.t / 1000, W = msg.w, H = msg.h, k = H / W;
      const landscape = W >= H, tanH = landscape ? TAN_LONG : TAN_LONG * W / H, tanV = landscape ? TAN_LONG * H / W : TAN_LONG;
      S.handT = t;
      if (!(S.w > 0) && W > 0 && H > 0) { S.w = W; S.h = H; }   // aspect before the face model is ready
      const faceFresh = S.face && S.face.present && Math.abs(S.faceT - t) < 0.4 ? S.face : null;
      const dets = (msg.hands || []).map(h => {
        const lm = h.lm, X = i => lm[3 * i], Y = i => lm[3 * i + 1], Z = i => lm[3 * i + 2];
        const d3 = (a, b) => Math.hypot(X(a) - X(b), (Y(a) - Y(b)) * k, Z(a) - Z(b));
        const pcx = (X(0) + X(5) + X(9) + X(17)) / 4, pcy = (Y(0) + Y(5) + Y(9) + Y(17)) / 4;
        const scale = Math.hypot(X(9) - X(0), (Y(9) - Y(0)) * k);
        const ext = FINGERS.map(([mcp, pip, tip]) => clamp01((d3(tip, 0) / Math.max(1e-4, d3(mcp, 0)) - 1.15) / 0.75));
        const handed = SIGN.handLabelInverted ? (h.label === 'Left' ? 'right' : h.label === 'Right' ? 'left' : '') : (h.label || '').toLowerCase();
        const pdx = -(X(8) - X(5)), pdy = -(Y(8) - Y(5)) * k, plen = Math.hypot(pdx, pdy);
        return {x: (0.5 - pcx) * 2, y: (0.5 - pcy) * 2, wx: 1 - X(0), scale, ext, open: (ext[0] + ext[1] + ext[2] + ext[3]) / 4,
          handed, lscore: h.lscore, gesture: h.gscore >= 0.5 ? (h.gesture || 'None') : 'None', gscore: h.gscore, graw: h.gesture,
          pdir: plen > 1e-4 ? [pdx / plen, pdy / plen] : [0, 1], pfore: plen / Math.max(1e-4, scale), lm, k};
      });
      // track association (greedy nearest, handedness as a tie-breaker)
      const used = new Set(), out = [];
      const tracks = S.tracks.filter(tr => t - tr.lastT < TRACK_KEEP);
      const pairsList = [];
      dets.forEach((d, di) => tracks.forEach((tr, ti) => pairsList.push([Math.hypot(d.x - tr.x, d.y - tr.y) + (d.handed && tr.handed && d.handed !== tr.handed ? 0.25 : 0), di, ti])));
      pairsList.sort((a, b) => a[0] - b[0]);
      const assign = {}, takenT = new Set();
      for (const [c, di, ti] of pairsList) { if (c > 0.6 || used.has(di) || takenT.has(ti)) continue; assign[di] = tracks[ti]; used.add(di); takenT.add(ti); }
      const nextTracks = [];
      dets.forEach((d, di) => {
        let tr = assign[di];
        if (!tr) tr = {id: S.nextTrack++, f: bank('hand'), n: 0, born: t, wave: {dir: 0, ext: null, revs: [], lastOk: -1, waving: false},
          lnS: null, vscale: 0, vsHi: 0, hold: {}, holdN: {}, hv: 0};
        const dt = tr.lastT != null ? t - tr.lastT : 0;
        const x = tr.f.get('x').filter(d.x, t), y = tr.f.get('y').filter(d.y, t), sc = tr.f.get('s').filter(d.scale, t), open = tr.f.get('o').filter(d.open, t);
        tr.vx = tr.n ? lowpass(tr.vx, (x - tr.x) / Math.max(1e-3, dt), dt, 4) : 0;
        tr.vy = tr.n ? lowpass(tr.vy, (y - tr.y) / Math.max(1e-3, dt), dt, 4) : 0;
        const ln = Math.log(Math.max(1e-3, sc));
        tr.vscale = tr.lnS != null && dt > 0 ? lowpass(tr.vscale, (ln - tr.lnS) / dt, dt, 3) : 0;
        tr.lnS = ln; tr.x = x; tr.y = y; tr.scale = sc; tr.open = open; tr.lastT = t; tr.n++;
        if (d.handed) tr.hv = 0.6 * tr.hv + 0.4 * (d.handed === 'left' ? 1 : -1) * Math.max(0.3, d.lscore || 0.5);
        tr.handed = tr.hv > 0.05 ? 'left' : tr.hv < -0.05 ? 'right' : (d.handed || tr.handed);
        const mid = faceFresh ? faceFresh.x : 0;
        const side = x < mid ? 'l' : 'r';
        const curled3 = Math.max(d.ext[1], d.ext[2], d.ext[3]);
        const pointing = d.ext[0] > 0.55 && curled3 < 0.45 && ((d.gesture === 'Pointing_Up' && d.gscore >= 0.7) || (d.ext[0] > 0.7 && curled3 < 0.35));
        const main = !!faceFresh && Math.abs(x - faceFresh.x) <= 2.5 * 2 * faceFresh.size && sc >= 0.45 * faceFresh.iod;
        const hand = {side, handed: tr.handed || '', x: r4(x), y: r4(y), ax: r4(Math.atan(x * tanH)), ay: r4(Math.atan(y * tanV)),
          vx: r4(tr.vx), vy: r4(tr.vy), scale: r4(sc), vscale: r4(tr.vscale), open: r4(clamp01(open)), gesture: d.gesture, gscore: r4(d.gscore),
          point: pointing ? {dx: r4(d.pdir[0]), dy: r4(d.pdir[1])} : null, fore: r4(d.pfore), ext: d.ext.map(r4), main, id: tr.id, t: r4(t), age: 0,
          lm: Array.from({length: 21}, (_, i) => [r4(1 - d.lm[3 * i]), r4(d.lm[3 * i + 1]), r4(d.lm[3 * i + 2])])};
        tr.hand = hand;
        handEvents(tr, hand, d, t, faceFresh);
        nextTracks.push(tr); out.push(hand);
      });
      S.tracks = nextTracks.concat(tracks.filter(tr => !nextTracks.includes(tr) && t - tr.lastT < TRACK_KEEP));
      out.sort((a, b) => a.x - b.x);
      S.hands = out; S.handRaw = out;
    }
    function holdFor(tr, key, cond, t) {   // seconds the condition has held on CONSECUTIVE hand frames (0 if false); tr.holdN[key] = frames
      if (!cond) { tr.hold[key] = null; tr.holdN[key] = 0; return 0; }
      if (tr.hold[key] == null) { tr.hold[key] = t; tr.holdN[key] = 0; }
      tr.holdN[key]++;
      return t - tr.hold[key];
    }
    function handEvents(tr, h, d, t, face) {
      const chin = face ? face.chinY : -0.25;
      // wave (hand model): open hand (≥ 3 fingers extended, not Fist/Thumb/Pointing), palm centre at/above the main
      // face's chin, ≥ 2 x reversals within 1.6 s of amplitude ≥ AMP frame widths: 0.5 × palm (clamped 0.02..0.06) and,
      // with a face, ≥ 0.4 face widths (≈ 0.2 shoulder widths). UNFILTERED palm centre (filtering eats the swing at
      // 6-10 fps). With a face tracked the hand must belong to it; with no face (distant people) any raised open hand.
      const W = tr.wave, px = 0.5 + d.x / 2;
      const open = d.ext.filter(e => e > 0.6).length >= 3 && h.gesture !== 'Closed_Fist' && h.gesture !== 'Thumb_Up' && h.gesture !== 'Pointing_Up';
      const ok = open && (face ? h.main && h.y >= chin : distant(t) && h.y > -0.35 && h.scale > 0.012);
      revStep(W, px, Math.max(clamp(0.5 * d.scale, 0.02, 0.06), face ? 0.4 * face.size : 0), ok, t, 1.6, 0.5);
      maybeWave(W, ok, t, h.side, h.handed, 'hand', h.main, h.x, h.y, d.k);
      // point: main hand, index out + others curled (+ Pointing_Up ≥ 0.7 or a clear geometric point), ≥ 300 ms and ≥ 3 frames
      if (holdFor(tr, 'point', !!h.point && h.main, t) >= 0.3 && tr.holdN.point >= 3) {
        const dx = h.point.dx, dy = h.point.dy;
        const at = h.fore < 0.4 ? 'screen' : Math.abs(dy) >= Math.abs(dx) ? (dy > 0 ? 'up' : 'down') : (dx < 0 ? 'left' : 'right');
        if (debounced('point', t, 2.0)) emit({type: 'point', t: r4(t), side: h.side, handed: h.handed, dir: {x: dx, y: dy}, at, x: h.x, y: h.y, ax: h.ax, ay: h.ay});
      }
      // thumbs up: main hand, Thumb_Up ≥ 0.7 with the four fingers curled, ≥ 350 ms and ≥ 3 frames
      const thumb = h.main && h.gesture === 'Thumb_Up' && h.gscore >= 0.7 && Math.max(d.ext[0], d.ext[1], d.ext[2], d.ext[3]) < 0.5;
      if (holdFor(tr, 'thumb', thumb, t) >= 0.35 && tr.holdN.thumb >= 3 && debounced('thumbs_up', t, 3.0)) emit({type: 'thumbs_up', t: r4(t), side: h.side, handed: h.handed, x: h.x, y: h.y, ax: h.ax, ay: h.ay});
      // object_up: main hand raised to around face height (chin-0.12 .. top of head), curled-but-not-a-sign, not resting on
      // the face, near-stationary ≥ 0.6 s and ≥ 4 frames
      const sign = ['Open_Palm', 'Pointing_Up', 'Thumb_Up', 'Victory', 'ILoveYou'].includes(h.gesture);
      const onFace = face && Math.abs(h.x - face.x) < face.size * 1.1 && h.y < face.topY && h.y > face.chinY - 0.05;
      const objCond = !!face && h.main && !sign && h.open > 0.08 && h.open < 0.75 && h.y > chin - 0.12 && h.y <= face.topY + 0.05 &&
        !onFace && Math.hypot(h.vx, h.vy) < 0.35 && tr.n >= 3;
      if (holdFor(tr, 'obj', objCond, t) >= 0.6 && tr.holdN.obj >= 4 && debounced('object_up', t, 4.0)) emit({type: 'object_up', t: r4(t), side: h.side, handed: h.handed, x: h.x, y: h.y, ax: h.ax, ay: h.ay});
      // approach_fast (hand): d ln(scale)/dt > 2.2/s on 2 consecutive samples, scale > 0.12, track ≥ 3 samples, main hand if a face is tracked
      if (h.vscale > 2.2 && h.scale > 0.12 && tr.n >= 3 && (!face || h.main)) tr.vsHi++; else tr.vsHi = 0;
      if (tr.vsHi >= 2 && debounced('approach_fast', t, 1.5)) emit({type: 'approach_fast', t: r4(t), source: 'hand', side: h.side, rate: h.vscale, x: h.x, y: h.y, ax: h.ax, ay: h.ay});
    }

    // ── pose ──
    function onPose(msg) {
      const t = msg.t / 1000, k = msg.h / msg.w;
      S.poseT = t;
      if (!msg.present) { S.pose = {present: false, t}; S.fpose.reset(); return; }
      const lm = msg.lm, P = i => ({x: (0.5 - lm[4 * i]) * 2, y: (0.5 - lm[4 * i + 1]) * 2, v: lm[4 * i + 3]});
      const sh = [P(11), P(12)], el = [P(13), P(14)], wr = [P(15), P(16)];
      const present = sh[0].v > 0.5 && sh[1].v > 0.5;
      const li = sh[0].x <= sh[1].x ? 0 : 1, ri = 1 - li;   // arm whose shoulder is on screen-left
      const F = (name, p) => ({x: r4(S.fpose.get(name + 'x').filter(p.x, t)), y: r4(S.fpose.get(name + 'y').filter(p.y, t)), v: r4(p.v)});
      const shoulders = {l: F('sl', sh[li]), r: F('sr', sh[ri])}, elbows = {l: F('el', el[li]), r: F('er', el[ri])}, wrists = {l: F('wl', wr[li]), r: F('wr', wr[ri])};
      const dxw = (shoulders.r.x - shoulders.l.x) / 2, dyw = (shoulders.l.y - shoulders.r.y) / 2 * k;
      S.pose = {present, t, shoulders, elbows, wrists, lean: r4(Math.atan2(dyw, dxw)), width: r4(Math.hypot(dxw, dyw)), age: 0,
        handed: {l: li === 0 ? 'left' : 'right', r: li === 0 ? 'right' : 'left'}};
      if (present) poseWave(t, k, {l: [sh[li], el[li], wr[li]], r: [sh[ri], el[ri], wr[ri]]}, Math.hypot(dxw, dyw), S.pose.handed);
      S.poseRaw = {sh: [shoulders.l, shoulders.r], el: [elbows.l, elbows.r], wr: [wrists.l, wrists.r]};
    }

    // Pose wave (distant / small hands: GestureRecognizer drops tiny moving hands, pose keeps the wrist at 6 fps):
    // forearm up (wrist above the elbow), ≥ 2 reversals of the RAW wrist x within 1.6 s, amplitude ≥ max(0.012, 0.2 ×
    // shoulder width) frame widths. No face tracked (distant people): wrist no lower than 0.35 shoulder widths below the
    // shoulder. Face tracked: the pose must be that person (shoulder midpoint within 2 face widths), wrist at/above the
    // shoulder, AND an open hand from the hand model near that wrist with its palm at/above the chin (storytelling
    // gestures in sam_story swing the wrist 0.05 frame widths at chest height — rejected by this).
    function poseWave(t, k, arms, widthW, handed) {
      const face = S.face && S.face.present && Math.abs(S.faceT - t) < 0.4 ? S.face : null;
      if (!face && !distant(t)) return;
      const midX = (arms.l[0].x + arms.r[0].x) / 2;
      if (face && Math.abs(midX - face.x) > 2 * 2 * face.size) return;
      const wy = widthW * 2 / k;                         // shoulder width in y-units
      for (const side of ['l', 'r']) {
        const [shp, elp, wrp] = arms[side];
        let ok = wrp.v > 0.5 && elp.v > 0.4 && wrp.y > elp.y + 0.1 * wy;
        if (face) {
          // close-range main person: wrist at/above the shoulder AND the hand model must corroborate — an open hand
          // (≥ 3 fingers extended) near that wrist, palm at/above the chin, seen in the last 0.5 s
          ok = ok && wrp.y >= shp.y && S.tracks.some(tr => tr.hand && t - tr.lastT < 0.5 && Math.hypot(tr.hand.x - wrp.x, tr.hand.y - wrp.y) < 0.5 &&
            tr.hand.ext.filter(e => e > 0.6).length >= 3 && tr.hand.y >= face.chinY);
        } else ok = ok && wrp.y > shp.y - 0.35 * wy;   // distant people (no face found): forearm up, wrist ~shoulder high
        const W = S.poseWave[side] || (S.poseWave[side] = {dir: 0, ext: null, revs: [], lastOk: -1, waving: false});
        revStep(W, 0.5 + wrp.x / 2, Math.max(0.012, 0.2 * widthW), ok, t, 1.6, 0.6);
        maybeWave(W, ok, t, side, handed[side], 'pose', !!face, wrp.x, wrp.y, k);
      }
    }
    // "distant" = no face now and none in the last 1.5 s (a face lost a moment ago is still the close main person)
    const distant = t => S.ready.face ? (!S.faceSeenOnce || (S.faceAbsentSince != null && t - S.faceAbsentSince >= 1.5)) : !!S.failed.face;
    // reversal counter shared by the hand and pose wave detectors (x in frame-width units)
    function revStep(W, px, amp, ok, t, win, okGrace) {
      if (ok) W.lastOk = t; else if (t - W.lastOk > okGrace) { W.revs = []; W.waving = false; }
      if (W.ext == null) { W.ext = px; W.dir = 0; }
      else if (W.dir === 0) { if (px - W.ext >= amp) { W.dir = 1; W.ext = px; } else if (W.ext - px >= amp) { W.dir = -1; W.ext = px; } }
      else if (W.dir > 0) { if (px > W.ext) W.ext = px; else if (W.ext - px >= amp) { if (ok) W.revs.push(t); W.dir = -1; W.ext = px; } }
      else { if (px < W.ext) W.ext = px; else if (px - W.ext >= amp) { if (ok) W.revs.push(t); W.dir = 1; W.ext = px; } }
      W.revs = W.revs.filter(rt => t - rt <= win);
      if (W.waving && (!W.revs.length || t - W.revs[W.revs.length - 1] > 1.2)) W.waving = false;
    }
    function maybeWave(W, ok, t, side, handed, source, main, x, y, k) {
      if (!(ok && W.revs.length >= 2)) return;
      // a fresh wave needs ≥ 4 s since the previous wave event (any source); inside that, re-emit every ≥ 1.5 s as continuing
      const last = S.debounce.wave;
      const [ax, ay] = angXY(x, y, k), pos = {x: r4(x), y: r4(y), ax, ay};
      if (last == null || t - last >= 4.0) { S.debounce.wave = t; W.waving = true; emit(Object.assign({type: 'wave', t: r4(t), side, handed, source, main}, pos)); }
      else if (t - last >= 1.5) { S.debounce.wave = t; W.waving = true; emit(Object.assign({type: 'wave', t: r4(t), side, handed, source, main, continuing: true}, pos)); }
    }

    // ── feature frame ──
    function emitFrame(t) {
      const dt = S.lastEmitT == null ? 0 : t - S.lastEmitT;
      S.lastEmitT = t;
      const faceOk = S.face && S.face.present && t - S.faceT < 0.5;
      const face = faceOk ? Object.assign({}, S.face, {age: r4(Math.max(0, t - S.faceT))}) : {present: false};
      if (!faceOk && S.ready.face === false && !S.failed.face) face.loading = true;
      const handsFresh = S.hands && t - S.handT < 0.5 ? S.hands : [];
      const hands = handsFresh.map(h => Object.assign({}, h, {age: r4(Math.max(0, t - h.t))}));
      const pose = S.pose && t - S.poseT < 1.0 ? Object.assign({}, S.pose, {age: r4(Math.max(0, t - S.poseT))}) : null;
      // motion = frame-difference energy + landmark speed
      const lmSpeed = Math.max(faceOk ? Math.hypot(face.vx, face.vy) / 1.2 : 0, ...hands.map(h => Math.hypot(h.vx, h.vy) / 2.5));
      S.motionLm = lowpass(S.motionLm, clamp01(lmSpeed), Math.max(0.02, dt), 3);
      S.motion = clamp01(0.7 * S.motionFrame + 0.3 * S.motionLm);
      if (!faceOk && t - S.faceT > 0.5 && S.face && S.face.present) { S.face = {present: false}; updateAttention(t, 0); }
      const f = {t: r4(t), dt: r4(dt), face, hands, pose, motion: r4(S.motion), approach: r4(faceOk ? S.approach : 0),
        attention: r4(faceOk ? S.attention : 0), attending: S.attending, events: S.pendingEvents.splice(0),
        aspect: S.w > 0 && S.h > 0 ? r4(S.h / S.w) : null};   // H/W of the inference frame (isotropic geometry for gestures.js)
      if (C && C.last) f.canon = C.last;
      S.latest = f;
      try { onFeatures(f); } catch (e) { noteErr('onFeatures: ' + e.message); }
    }

    // ── preview drawing (≤ 20 fps, only on the preview canvas) ──
    function drawLoop() {
      if (!S.running) return;
      S.drawHandle = requestAnimationFrame(drawLoop);
      const tnow = now();
      if (tnow - S.lastDraw < 50) return;
      S.lastDraw = tnow;
      if (!S.ctx || !S.previewOn) return;
      const cv = S.canvas, dpr = Math.min(2, (typeof devicePixelRatio !== 'undefined' && devicePixelRatio) || 1);
      const bw = Math.max(1, Math.round(cv.clientWidth * dpr)), bh = Math.max(1, Math.round(cv.clientHeight * dpr));
      if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
      const c = S.ctx;
      c.clearRect(0, 0, bw, bh);
      if (!S.meshOn) return;
      // object-fit:cover mapping (mirrored video content at box X = ox + vx*vw*s)
      const vw = S.vw || 4, vh = S.vh || 3, s = Math.max(bw / vw, bh / vh), ox = (bw - vw * s) / 2, oy = (bh - vh * s) / 2;
      const VX = vx => ox + vx * vw * s, VY = vy => oy + vy * vh * s;
      const tsec = tnow / 1000;
      const face = S.face;
      if (face && face.present && S.faceRaw && tsec - S.faceT < 0.4) {
        const m = S.faceRaw, E = S.edges.face;
        if (E) {
          c.strokeStyle = 'rgba(140,235,255,0.30)'; c.lineWidth = Math.max(0.5, 0.55 * dpr); c.beginPath();
          for (let i = 0; i < E.length; i += 2) { const a = E[i], b = E[i + 1]; c.moveTo(VX(m[2 * a]), VY(m[2 * a + 1])); c.lineTo(VX(m[2 * b]), VY(m[2 * b + 1])); }
          c.stroke();
        }
        c.fillStyle = 'rgba(210,250,255,0.85)'; const ps = Math.max(1, 0.9 * dpr);
        for (let i = 0; i < m.length / 2; i++) c.fillRect(VX(m[2 * i]) - ps / 2, VY(m[2 * i + 1]) - ps / 2, ps, ps);
      }
      if (S.hands && tsec - S.handT < 0.4) {
        const E = S.edges.hand || HAND_EDGES;
        for (const h of S.hands) {
          c.strokeStyle = h.side === 'l' ? 'rgba(255,200,120,0.95)' : 'rgba(255,150,190,0.95)'; c.lineWidth = Math.max(1, 1.4 * dpr); c.beginPath();
          for (let i = 0; i < E.length; i += 2) { const a = h.lm[E[i]], b = h.lm[E[i + 1]]; c.moveTo(VX(a[0]), VY(a[1])); c.lineTo(VX(b[0]), VY(b[1])); }
          c.stroke(); c.fillStyle = '#fff2d0';
          for (const p of h.lm) c.fillRect(VX(p[0]) - 1.5 * dpr, VY(p[1]) - 1.5 * dpr, 3 * dpr, 3 * dpr);
        }
      }
      const pose = S.pose;
      if (pose && pose.present && S.poseRaw && tsec - S.poseT < 0.6) {
        const toV = p => [VX((p.x / 2) + 0.5), VY(0.5 - p.y / 2)];
        c.strokeStyle = 'rgba(190,160,255,0.9)'; c.lineWidth = Math.max(1, 1.6 * dpr); c.beginPath();
        const [sl, sr] = S.poseRaw.sh.map(toV); c.moveTo(sl[0], sl[1]); c.lineTo(sr[0], sr[1]);
        for (let i = 0; i < 2; i++) {
          const sp = S.poseRaw.sh[i], ep = S.poseRaw.el[i], wp = S.poseRaw.wr[i];
          const a = toV(sp); c.moveTo(a[0], a[1]);
          if (ep.v > 0.4) { const b = toV(ep); c.lineTo(b[0], b[1]); if (wp.v > 0.4) { const w = toV(wp); c.lineTo(w[0], w[1]); } }
        }
        c.stroke();
      }
    }
    const HAND_EDGES = [0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 5, 9, 9, 10, 10, 11, 11, 12, 9, 13, 13, 14, 14, 15, 15, 16, 13, 17, 0, 17, 17, 18, 18, 19, 19, 20];

    // ── lifecycle ──
    // opts.deviceId picks the camera (a missing / unplugged one falls back to the front camera);
    // setCamera(deviceId) switches it mid-session — same <video> element, so the workers, the self-view and anything
    // drawing that element (the session recorder) just carry on
    function camGum(deviceId, ms) {
      const v = {width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 30, max: 30}};
      if (deviceId) v.deviceId = {exact: deviceId}; else v.facingMode = 'user';
      let expired = false, timer;
      const req = navigator.mediaDevices.getUserMedia({video: v, audio: false})
        .then(st => { if (expired) { st.getTracks().forEach(x => x.stop()); throw Error('camera arrived after timeout'); } return st; });
      return Promise.race([req, new Promise((_, rej) => { timer = setTimeout(() => { expired = true; rej(Error('camera permission/start timed out after ' + Math.round(ms / 1000) + ' s')); }, ms); })])
        .finally(() => clearTimeout(timer));
    }
    async function openCamera(deviceId) {
      if (!deviceId) return camGum('', 30000);
      try { return await camGum(deviceId, 30000); }
      catch (e) { if (e && /timed out/.test(e.message)) throw e; noteErr('camera ' + deviceId.slice(0, 8) + ' unavailable (' + (e && e.name || e) + '): default camera'); return camGum('', 30000); }
    }
    function armVideo() {
      if (S.video.requestVideoFrameCallback) {
        try { if (S.rvfcHandle && S.video.cancelVideoFrameCallback) S.video.cancelVideoFrameCallback(S.rvfcHandle); } catch (_) {}
        S.rvfcHandle = S.video.requestVideoFrameCallback(onVideoFrame);
      }
    }
    async function setCamera(deviceId) {
      opts.deviceId = deviceId || '';
      if (!S.running || !S.video) return {ok: true, later: true};
      let st;
      try { st = await camGum(deviceId || '', 15000); }
      catch (e) {                 // phones that cannot open two cameras at once: release the old one, then retry
        if (S.stream) S.stream.getTracks().forEach(x => x.stop());
        try { st = await camGum(deviceId || '', 15000); }
        catch (e2) { st = await camGum('', 15000).catch(() => null); if (st && S.running) { S.stream = st; S.video.srcObject = st; S.video.play().catch(() => {}); armVideo(); } throw e2; }
      }
      if (!S.running) { st.getTracks().forEach(x => x.stop()); return {ok: false, stopped: true}; }
      const old = S.stream; S.stream = st; S.video.srcObject = st;
      try { await Promise.race([S.video.play(), new Promise((_, rej) => setTimeout(() => rej(Error('camera playback timed out')), 10000))]); }
      catch (e) { noteErr('setCamera play: ' + e.message); }
      if (old && old !== st) old.getTracks().forEach(x => x.stop());
      S.lastTick = null; S.switches = (S.switches || 0) + 1;
      armVideo();
      return {ok: true, camera: {w: S.video.videoWidth, h: S.video.videoHeight}, deviceId: currentDevice()};
    }
    function currentDevice() {
      try { const t = S.stream && S.stream.getVideoTracks()[0], st = t && t.getSettings ? t.getSettings() : null; return (st && st.deviceId) || ''; } catch (e) { return ''; }
    }
    async function start() {
      if (S.running) return {ok: true, already: true};
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw Error('camera needs a secure page (getUserMedia unavailable)');
      S.stream = await openCamera(opts.deviceId || '');
      if (!S.box) buildPreview();
      if (!S.video) { S.video = document.createElement('video'); S.video.muted = true; S.video.playsInline = true; }
      S.video.srcObject = S.stream;
      S.running = true; applyPreview();
      try { await Promise.race([S.video.play(), new Promise((_, rej) => setTimeout(() => rej(Error('camera playback timed out')), 10000))]); }
      catch (e) { stop(); throw e; }
      S.ready = {}; S.failed = {}; S.tracks = []; S.hands = []; S.face = null; S.pose = null;
      S.faceSeenOnce = false; S.faceAbsentSince = null; S.faceLostSent = true; S.attending = false; S.notAttSince = 0; S.everAttended = false;
      if (C) canonReset();
      armVideo();
      S.rafHandle = requestAnimationFrame(rafPump);
      S.drawHandle = requestAnimationFrame(drawLoop);
      startWorkers();
      return {ok: true, camera: {w: S.video.videoWidth, h: S.video.videoHeight}, models: models.slice()};
    }
    function stop() {
      S.running = false;
      if (S.stream) { S.stream.getTracks().forEach(x => x.stop()); S.stream = null; }
      if (S.video) { try { if (S.rvfcHandle && S.video.cancelVideoFrameCallback) S.video.cancelVideoFrameCallback(S.rvfcHandle); } catch (_) {} S.video.srcObject = null; }
      cancelAnimationFrame(S.rafHandle); cancelAnimationFrame(S.drawHandle);
      for (const m in S.workers) { try { S.workers[m].terminate(); } catch (_) {} }
      S.workers = {}; S.ready = {}; S.busy = {};
      for (const u of S.blobUrls.splice(0)) URL.revokeObjectURL(u);
      applyPreview();
    }
    function stats() {
      const out = {models: {}, errors: S.errors.slice(-10), camera: {w: S.vw, h: S.vh}, infer: {w: S.w, h: S.h}, pose: poseOn};
      const tn = now();
      for (const m of models) {
        const q = S.res[m].filter(r => tn - r.recv < 5000);
        out.models[m] = {ready: !!S.ready[m], failed: S.failed[m] || null, n: S.res[m].length,
          fps: q.length > 1 ? r4((q.length - 1) * 1000 / (q[q.length - 1].recv - q[0].recv)) : 0,
          infer_p50: pct(q.map(r => r.infer), 0.5), infer_p95: pct(q.map(r => r.infer), 0.95),
          latency_p50: pct(q.map(r => r.lat), 0.5), latency_p95: pct(q.map(r => r.lat), 0.95)};
      }
      if (C) out.canon = {n: C.n, seq: C.seq, ms: C.n ? r4(C.ms / C.n) : 0};
      return out;
    }
    return {start, stop, setPreview, setMesh, setCamera, currentDevice, latest: () => S.latest, stats, events: S.events,
      get running() { return S.running; }, _state: S};
  }

  return {createPerception, createExpressionTracker, OneEuro, GEOM, SIGN, corr};
});
