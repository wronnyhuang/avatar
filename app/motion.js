/* motion.js — the avatar's continuous motion controller.
 *
 * createMotion({prior, seed, config}) → {step(dt, inputs) → {param:value}, command(cmd), release(parts), event(ev), ...}
 * Runs every display frame (~30 Hz) on the phone and produces the FULL rig vector (every contract param except the
 * lipsync mouth, which the renderer takes from lipstream). Works in the browser (window.AvatarMotionLib) and in node
 * (module.exports) so the tests can drive it with synthetic feature streams.
 *
 * GEOMETRY (all in the VIEWER-SCREEN frame, the same frame as the renderer's params):
 *   renderer: head_yaw > 0 turns toward screen-right; head_pitch > 0 = nose down; head_roll > 0 = crown toward
 *   screen-right; gaze_x > 0 = eyes toward screen-right, gaze_y > 0 = up. gaze is an offset FROM THE CAMERA
 *   DIRECTION added in head-local space (fox.glsl: eyeF = normalize(toCam + (0.30 gx, 0.24 gy, 0))), so gaze 0 is
 *   eye contact whatever the head does, and the eye-in-world angle off the camera is ≈ atan(0.30 gx) — at most ~17°.
 *   perception: face.x/ax etc. are MIRRORED viewer coordinates (x < 0 = screen-left as he sees himself = his own
 *   left). He moves to his left → he is on the screen-left side → the avatar must look toward screen-left →
 *   gaze_x < 0, head_yaw < 0. So the mapping is the identity in this frame (an earlier version used the raw camera x: mirror bug).
 *   Eyes LEAD (saccade after a 150–250 ms reaction, 45–150 ms long, min-jerk), the head FOLLOWS through a
 *   second-order spring; since gaze is world-referenced the eyes stay on target while the head arrives (VOR for free).
 *   Looking AT him = eye world angle ≈ gazeFollow × his angular offset (default 0.5: visible following while staying
 *   inside the mutual-gaze cone), head ≈ headFollow × offset (0.6).
 *
 * LAYERS (base → top): life (breath, blinks, micro-saccades, springs on ears/tail/paws, weight shifts) · attention
 *   and gaze · emotion mirroring (delayed, attenuated) · reflexes (startle, wave back, perk up, look back, point,
 *   object) · speaking (prosody → nods, tilts, brow raises, beat gestures, gaze aversion) · listening (backchannel
 *   nods at his pauses, tilts) · boredom (look around, fidgets — a no-repeat bag) · learned prior (plug-in) ·
 *   gestures (keyframed, weighted) · commands (holds with min-jerk ease in/out — held params win, the rest lives).
 *
 * PAGE NOTES — loading the learned prior (branch `prior`, v1: app/prior.js + app/prior_weights.json, 1.3 MB JSON):
 *   1. app.py: inline prior.js before motion.js (integ already does) and serve app/prior_weights.json (allowlisted
 *      GET route like vision/, gzip + Cache-Control private max-age=86400; 1.3 MB → ~0.5 MB gzipped).
 *   2. page.html, right after startMotion() created D.motion (works whenever the weights arrive; until then: procedural):
 *        if (window.createPrior && D.motion && D.motion.setPrior)
 *          fetch(url('prior_weights.json'), {cache: 'force-cache', signal: tsignal(60000)})
 *            .then(r => r.ok ? r.json() : null)
 *            .then(w => { if (w && D.motion) D.motion.setPrior(window.createPrior(w)); })
 *            .catch(() => {});
 *   (`createMotion({prior: window.AvatarPrior})` stays harmless: nothing defines AvatarPrior.)
 *   Verified in node (tests/prior_check.cjs): +0.1 ms/frame; its head angles are smoothed (3 Hz) and scaled ×0.75 on
 *   top of the attention head; when it blinks it owns the spontaneous blink clock (reflex blinks stay ours).
 *   PRESETS for the eval loop: ?mset=calm|default|lively (read from the page URL unless createMotion({preset}) is given;
 *   default output is byte-identical to no preset). Multipliers (calm / lively): free head motion + per-phrase postures
 *   + tail wag 0.8 / 1.3; additive gesture channels + speech nods/sways/brow raises 0.85 / 1.25; fidget + ear/nose
 *   twitch rate 0.7 / 1.4; emotion mirroring 0.8 / 1.15; beat gestures (chance, rate, threshold) 0.7 / 1.6;
 *   weight shifts + body noise + body follow 0.8 / 1.4. ?prior=0 (page) turns the learned prior off.
 *   {action:'hold'} ("hold that / freeze / stay like that"): freezes the posture + whatever a gesture is moving, until release.
 *   ALSO: feed motion the avatar level at ctx.currentTime (newest tapped chunk), not D.play.avatarVoice() (heard level,
 *   D behind) — the rig vector is stamped with the newest timeline time, so heard-level prosody lands D late.
 *
 * SIDES: `_l` params = SCREEN-LEFT (character's anatomical right). In commands, 'left'/'right' mean the CHARACTER'S
 *   OWN left/right ("wink your left eye" → blink_r); 'screen_left'/'screen_right' (or 'l'/'r') name screen sides.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarMotionLib = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ------------------------------------------------------------------ params (anim/schema.py order + contract appends)
  const SCHEMA_PARAMS = ['jaw', 'round', 'spread', 'press', 'labdent', 'tongue', 'smile',
    'head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z', 'breath', 'blink_l', 'blink_r',
    'gaze_x', 'gaze_y', 'brow_l', 'brow_r', 'brow_knit', 'ear_l', 'ear_r', 'ear_flick_l', 'ear_flick_r', 'cheek',
    'nostril', 'voice', 'lip_open', 'lip_upper', 'lip_lower', 'lip_width', 'lip_protrude', 'lip_press', 'lip_tuck',
    'teeth_gap', 'tongue_tip', 'tongue_dental', 'tongue_back', 'arm_l', 'arm_r', 'wave_l', 'wave_r', 'point_l',
    'point_r', 'paw_open_l', 'paw_open_r', 'clap', 'thumbs_up', 'body_lean', 'body_bounce', 'tail_wag',
    'tail_raise', 'body_yaw'];
  const NEW_PARAMS = ['hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_r_x', 'hand_r_y', 'hand_r_z', 'hand_l_open',
    'hand_r_open', 'hand_l_turn', 'hand_r_turn', 'thumb_l', 'thumb_r', 'shoulder_l', 'shoulder_r', 'ear_l_turn',
    'ear_r_turn', 'nose_wiggle', 'tongue_out', 'mouth_open', 'eye_wide', 'squint', 'body_pitch', 'tail_curl'];
  const RIG_PARAMS = SCHEMA_PARAMS.concat(NEW_PARAMS);
  // whole-body moves (docs/INTENT-RULES.md): root motion, camera, counting, props, feet — produced by app/moves.js; they stay 0
  // (and vector() without an order keeps the base columns only) unless a move / object command runs
  const BODY_PARAMS = ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'gaze_free', 'cam_zoom',
    'cam_x', 'cam_y', 'stand', 'fingers_l', 'fingers_r'];
  for (let k = 0; k < 8; k++) for (const a of ['x', 'y', 'z', 'rx', 'ry', 'rz', 'hl', 'hr', 'hh', 's']) BODY_PARAMS.push('prop' + k + '_' + a);
  BODY_PARAMS.push('foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z');
  // objects: prop slots 8-15, appended after the feet (anim/schema.py order)
  for (let k = 8; k < 16; k++) for (const a of ['x', 'y', 'z', 'rx', 'ry', 'rz', 'hl', 'hr', 'hh', 's']) BODY_PARAMS.push('prop' + k + '_' + a);
  const ALL_PARAMS = RIG_PARAMS.concat(BODY_PARAMS);
  const PERIODIC = new Set(BODY_PARAMS.filter(p => /^root_(yaw|pitch|roll)$|^prop\d+_r[xyz]$/.test(p)));
  // owned by lipstream at the renderer — the controller leaves them 0
  const LIPSYNC = new Set(['jaw', 'round', 'spread', 'press', 'labdent', 'tongue', 'cheek', 'voice', 'lip_open',
    'lip_upper', 'lip_lower', 'lip_width', 'lip_protrude', 'lip_press', 'lip_tuck', 'teeth_gap', 'tongue_tip',
    'tongue_dental', 'tongue_back']);
  const LEGACY = new Set(['arm_l', 'arm_r', 'wave_l', 'wave_r', 'paw_open_l', 'paw_open_r', 'clap', 'thumbs_up']);
  const R = (lo, hi) => [lo, hi];
  const RANGE = {
    smile: R(0, 1), head_yaw: R(-0.6, 0.6), head_pitch: R(-0.45, 0.45), head_roll: R(-0.4, 0.4),
    head_x: R(-0.12, 0.12), head_y: R(-0.12, 0.12), head_z: R(-0.12, 0.12), breath: R(-1, 1),
    blink_l: R(0, 1), blink_r: R(0, 1), gaze_x: R(-1, 1), gaze_y: R(-1, 1), brow_l: R(-1, 1), brow_r: R(-1, 1),
    brow_knit: R(0, 1), ear_l: R(-1, 1), ear_r: R(-1, 1), ear_flick_l: R(0, 1), ear_flick_r: R(0, 1),
    nostril: R(0, 1), point_l: R(0, 1), point_r: R(0, 1), body_lean: R(-1, 1), body_bounce: R(-1, 1),
    tail_wag: R(-1, 1), tail_raise: R(0, 1), body_yaw: R(-1, 1),
    hand_l_x: R(-1, 1.2), hand_l_y: R(-1, 1.2), hand_l_z: R(-1, 1.2), hand_r_x: R(-1, 1.2), hand_r_y: R(-1, 1.2),
    hand_r_z: R(-1, 1.2), hand_l_open: R(0, 1), hand_r_open: R(0, 1), hand_l_turn: R(-1, 1), hand_r_turn: R(-1, 1),
    thumb_l: R(0, 1), thumb_r: R(0, 1), shoulder_l: R(0, 1), shoulder_r: R(0, 1), ear_l_turn: R(-1, 1),
    ear_r_turn: R(-1, 1), nose_wiggle: R(-1, 1), tongue_out: R(0, 1), mouth_open: R(0, 1), eye_wide: R(0, 1),
    squint: R(0, 1), body_pitch: R(-1, 1), tail_curl: R(0, 1),
    root_x: R(-3, 3), root_y: R(-0.6, 3), root_z: R(-3, 1.5), squash: R(-1, 1), gaze_free: R(0, 1), cam_zoom: R(-1, 1),
    cam_x: R(-3, 3), cam_y: R(-1, 3), stand: R(0, 1), fingers_l: R(0, 5), fingers_r: R(0, 5),
    foot_l_x: R(-2, 2), foot_l_y: R(-0.3, 1.5), foot_l_z: R(-2, 2), foot_r_x: R(-2, 2), foot_r_y: R(-0.3, 1.5), foot_r_z: R(-2, 2),
  };
  for (let k = 0; k < 16; k++) { for (const a of ['x', 'y', 'z']) RANGE['prop' + k + '_' + a] = R(-3, 3); for (const a of ['hl', 'hr', 'hh', 's']) RANGE['prop' + k + '_' + a] = R(0, 1); }
  // the arm rest pose (contract REST): offsets 0, open 0.2, turn 0
  const REST_OPEN = 0.2;
  const GAZE_RX = Math.atan(0.30), GAZE_RY = Math.atan(0.24);   // eye-in-world radians at gaze ±1 (fox/bear/monkey)
  const TAN_HFOV2 = Math.tan(35 * Math.PI / 180), TAN_VFOV2 = Math.tan(27.5 * Math.PI / 180);

  // planner (docs/PLANNER.md §1): the planner's pose controls (the character's OWN-side names of
  // the keyframe-stream prompt) → rig columns, as the keyframe-stream player's KF_ALIAS. '@foot_*' = WORLD
  // foot offsets (characters with legs only). body_bounce = the seated characters' vertical channel (their renderer has no
  // root), passed straight through like every other direct column. A control not listed here is ignored.
  const KF_ALIAS = {eyes_closed: ['blink_l', 'blink_r']};
  for (const [own, s] of [['right', 'l'], ['left', 'r']]) {
    for (const a of ['x', 'y', 'z', 'open', 'turn']) KF_ALIAS[`${own}_paw_${a}`] = ['hand_' + s + '_' + a];
    KF_ALIAS[own + '_thumb'] = ['thumb_' + s]; KF_ALIAS[own + '_shoulder'] = ['shoulder_' + s]; KF_ALIAS[own + '_fingers'] = ['fingers_' + s];
    KF_ALIAS[own + '_ear'] = ['ear_' + s]; KF_ALIAS[own + '_brow'] = ['brow_' + s]; KF_ALIAS[own + '_point'] = ['point_' + s];
    for (const a of ['x', 'y', 'z']) KF_ALIAS[`${own}_foot_${a}`] = ['@foot_' + s + '_' + a];
  }
  for (const c of ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'body_lean', 'body_pitch', 'body_yaw', 'head_yaw', 'head_pitch',
    'head_roll', 'smile', 'mouth_open', 'eye_wide', 'squint', 'tongue_out', 'brow_knit', 'tail_raise', 'tail_wag', 'tail_curl', 'nose_wiggle', 'body_bounce']) KF_ALIAS[c] = [c];
  // objects (docs/OBJECTS.md §6): the keyframe language WITH THINGS. Held-object controls -> pseudo
  // columns '%held_<engine side>_<axis>' (interpolated like every control, never rig columns: handed to engine.kfHeld each
  // frame); paw events (fire once when playback crosses the pose) and "do" object steps are read off the pose itself.
  // OWN right paw = engine side 'l' (screen-left), as KF_ALIAS.
  const KF_OBJ = {}, KF_EV = {};
  for (const [own, s] of [['right', 'l'], ['left', 'r']]) {
    for (const a of ['x', 'y', 'z', 'spin', 'tilt', 'roll']) KF_OBJ[`${own}_held_${a}`] = ['%held_' + s + '_' + a];
    KF_EV[own + '_let_go'] = {type: 'release', side: s}; KF_EV[own + '_toss_up'] = {type: 'release', side: s, toss: true};
    KF_EV[own + '_catch'] = {type: 'catch', side: s}; KF_EV[own + '_grab'] = {type: 'grab', side: s};
  }
  // planner contract §3: the channel groups a learned listener may drive through motion.setListener(fn) — docs/LISTENER-INTERFACE.md
  const LISTENER_GROUPS = {
    face: ['smile', 'squint', 'eye_wide', 'blink_l', 'blink_r', 'brow_l', 'brow_r', 'brow_knit', 'mouth_open', 'nostril', 'nose_wiggle',
      'ear_l', 'ear_r', 'ear_l_turn', 'ear_r_turn', 'ear_flick_l', 'ear_flick_r'],
    gaze: ['gaze_x', 'gaze_y'],
    posture: ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z', 'body_yaw', 'body_lean', 'body_pitch', 'body_bounce',
      'shoulder_l', 'shoulder_r', 'breath'],
  };
  // planner contract §5: the rig mouth channels lip-sync owns while the avatar is voiced (the renderer opens the jaw to max(lipsync,
  // mouth_open), so a held mouth_open / tongue_out would cover the visemes): a KF pose's or the listener's values on these
  // yield to speech while voiced and apply when silent. (The lip columns themselves, LIPSYNC, are always lipstream's.)
  const SPEECH_GROUP = ['mouth_open', 'tongue_out'];

  const DEFAULTS = {
    gazeFollow: 0.5,        // eye world angle per radian of his angular offset
    headFollow: 0.6,        // head yaw/pitch per radian of his offset
    headMax: 0.5,
    reactLatency: [0.15, 0.25],   // eyes start this long after he moves (s)
    mirrorLag: 0.45,        // pure delay of emotion mirroring (s) ...
    mirrorTau: 0.45,         // ... then a first-order lag (s)
    mirrorGain: {smile: 0.75, squint: 0.5, browUp: 0.7, browDown: 0.65, eyeWide: 0.7, sad: 0.85, jawOpen: 0.35, surprise: 0.9, fear: 0.8, anger: 0.85},
    blinkIBI: {idle: 3.4, speaking: 2.5, listening: 4.0, bored: 2.7},   // mean inter-blink interval (s)
    headNoise: {speak: [0.06, 0.032, 0.032], listen: [0.04, 0.022, 0.025], idle: [0.052, 0.022, 0.022]},   // yaw, pitch, roll rad (std)
    boredAfter: 9,          // seconds of no interaction before boredom starts building
    priorWeight: 1.0,
    priorHead: 0.75,        // the prior's head angles are scaled by this on top of attention (speaking yaw SD ≈ 6–7°)
    legacyArms: false,      // also fill arm_l/wave_l/... from the hand targets for an old renderer
    handFollowHz: 5.5,
    xpLag: 0.25,            // with perception's expression tracker (face.xp): pure delay … (his face → ours ≈ 0.4–0.6 s)
    xpTau: 0.2,             // … and the first-order ease after it
    xpGain: {smile: 0.85, squint: 0.45, sad: 0.8, anger: 0.75, browDown: 0.45, surprise: 0.9, eyeWide: 0.5, browUp: 0.8},
    avatarVoiceTh: 0.008,   // rms (float, 0..1) above which the avatar's own voice counts as voiced
    userVoiceTh: 0.012,
  };

  // motion presets for the eval loop (?mset=calm|default|lively): amp = free head motion + per-phrase postures + tail,
  // gest = additive gesture channels and speech kicks (nods, sways, brow raises), fidget = boredom/twitch rate,
  // mirror = emotion-mirroring gain, beat = beat-gesture chance/frequency, sway = weight shifts + body noise + body follow
  const PRESETS = {
    calm: {amp: 0.8, gest: 0.85, fidget: 0.7, mirror: 0.8, beat: 0.7, sway: 0.8},
    default: {amp: 1, gest: 1, fidget: 1, mirror: 1, beat: 1, sway: 1},
    lively: {amp: 1.3, gest: 1.25, fidget: 1.4, mirror: 1.15, beat: 1.6, sway: 1.4},
  };

  // ------------------------------------------------------------------ small math
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const lerp = (a, b, s) => a + (b - a) * s;
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));   // min-jerk
  const smooth01 = (x) => mj(clamp(x, 0, 1));
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  class Spring {   // second-order, semi-implicit, sub-stepped
    constructor(hz, zeta, x = 0) { this.w = 2 * Math.PI * hz; this.z = zeta; this.x = x; this.v = 0; }
    step(target, dt) {
      const n = Math.max(1, Math.ceil(dt / 0.008)), h = dt / n, w = this.w, z = this.z;
      for (let i = 0; i < n; i++) { this.v += (w * w * (target - this.x) - 2 * z * w * this.v) * h; this.x += this.v * h; }
      return this.x;
    }
    hit(dv) { this.v += dv; }
  }
  const lp = (prev, x, dt, tau) => prev + (x - prev) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));

  // ------------------------------------------------------------------ the controller
  function createMotion(opts = {}) {
    const cfg = Object.assign({}, DEFAULTS, opts.config || {});
    // preset: createMotion({preset}) or the page URL's ?mset= (so page.html needs no change); unknown → default
    let presetName = opts.preset;
    if (presetName === undefined) { try { presetName = new URLSearchParams(globalThis.location.search).get('mset') || 'default'; } catch (e) { presetName = 'default'; } }
    if (!PRESETS[presetName]) presetName = 'default';
    const K = PRESETS[presetName];
    cfg.mirrorGain = Object.assign({}, DEFAULTS.mirrorGain, (opts.config || {}).mirrorGain || {});
    const rnd = mulberry32((opts.seed ?? (Date.now() ^ 0x5bd1e995)) >>> 0);
    const U = (a = 0, b = 1) => a + (b - a) * rnd();
    let spare = null;
    const N = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0; while (u < 1e-9) u = rnd(); v = rnd();
      const r = Math.sqrt(-2 * Math.log(u)); spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v);
    };
    const logn = (mean, sigma) => mean * Math.exp(sigma * N() - sigma * sigma / 2);
    const pick = (arr) => arr[Math.floor(rnd() * arr.length) % arr.length];
    const chance = p => rnd() < p;

    // 1/f-ish noise: sum of OU processes (corners 0.1–2 Hz, amplitude ∝ 1/sqrt(f)), then a 2nd-order smoother
    class Pink {
      constructor(bands = [[0.10, 1.0], [0.25, 0.7], [0.6, 0.5], [1.4, 0.32], [2.2, 0.2]]) {
        let v = 0; for (const b of bands) v += b[1] * b[1];
        const k = 1 / Math.sqrt(v);
        this.b = bands.map(([f, a]) => ({tau: 1 / (2 * Math.PI * f), a: a * k, x: N() * a * k}));
        this.sm = new Spring(2.6, 1.0, this.b.reduce((s, b) => s + b.x, 0));
      }
      step(dt) {
        let s = 0;
        for (const b of this.b) { b.x += -b.x * dt / b.tau + b.a * Math.sqrt(2 * dt / b.tau) * N(); s += b.x; }
        return this.sm.step(s, dt);
      }
    }

    let prior = opts.prior && typeof opts.prior.step === 'function' ? opts.prior : null;   // a prior instance ({step}); else procedural
    let learned = !!opts.learned;   // learned-reaction mode: the server's planner owns the slow layers (see learnedPass)
    const S = {
      t: 0, frame: 0, out: {}, lastOut: null,
      // perception-derived user state
      user: {present: false, presentT: 0, absentT: 99, ax: 0, ay: 0, attending: false, attT: 0, awayT: 0,
        energy: 0, boredom: 0, x: 0, y: 0, z: 0.5, hands: [], lookX: 0, lookY: 0},
      expr: {buf: [], base: {}, m: {smile: 0, squint: 0, browUp: 0, browDown: 0, eyeWide: 0, sad: 0, jawOpen: 0, surprise: 0, fear: 0, anger: 0}},
      // prosody trackers
      av: mkPros(), uv: mkPros(),
      // gaze
      eye: {W: [0, 0], from: [0, 0], to: [0, 0], sacT0: -1, sacDur: 0, goal: [0, 0], pending: null, micro: [0, 0],
        microNext: 0.6, lastMode: 'user', fixStart: 0},
      headT: [0, 0], headGain: 0.6, headLastSet: 0, headPending: null,
      hy: new Spring(1.55, 0.9), hp: new Spring(1.7, 0.92),
      focus: null,          // reflex focus {kind, W | fn, until, gain}
      avert: null,          // speaking/thinking/listening glance {W, until, gain}
      around: null,         // boredom / wander look-around {seq:[{W,dur}], i, t0, gain}
      cmdLook: null,        // {target, hold, until}
      joint: {done: false},
      // blink
      blink: {phase: 0, t0: -1, amp: 1, next: 1.2, refr: 0, dbl: false, forced: false},
      // life
      breath: {ph: U(0, 6.28), rate: 0.25, amp: 0.6, v: 0, sigh: -1, sp: new Spring(1.4, 1.0)},
      nz: {yaw: new Pink(), pitch: new Pink(), roll: new Pink(), lean: new Pink([[0.04, 1], [0.12, 0.5]]),
        byaw: new Pink([[0.03, 1], [0.1, 0.4]]), earL: new Pink([[0.2, 1], [0.6, 0.5]]), earR: new Pink([[0.2, 1], [0.6, 0.5]]),
        brow: new Pink([[0.15, 1], [0.5, 0.5]]), hand: new Pink([[0.1, 1], [0.3, 0.5]]), tail: new Pink([[0.08, 1], [0.3, 0.6]]),
        smile: new Pink([[0.05, 1], [0.15, 0.4]])},
      nodK: new Spring(2.3, 0.45), browK: new Spring(2.6, 0.7), tiltK: new Spring(1.4, 0.6), swayK: new Spring(1.3, 0.6),
      startleK: new Spring(1.8, 0.5), perkK: new Spring(1.2, 0.55),
      earSL: new Spring(3.2, 0.32), earSR: new Spring(3.0, 0.32), tailS: new Spring(1.6, 0.5),
      tilt: {target: 0, next: 3, v: 0}, shift: {lean: 0, yaw: 0, next: 6},
      speech: {aversNext: 3, beatSide: 'l', beatUntil: -1, beatNext: 0, lastNod: -1, stressNodWait: 0},
      listen: {lastNod: -9, tiltSide: 1},
      startle: {t0: -9, amp: 0, eyeWide: 0},
      perk: {t0: -9, amp: 0},
      greet: {t0: -9},
      flash: {t0: -9, amp: 0},
      mood: {val: 0, arousal: 0.3},
      idleT: 0, bored: 0, fidget: {next: 6, last: [], lastName: ''},
      earTw: {l: 4, r: 6}, noseTw: 9, sniff: {t0: -9},
      gestures: [], holds: {}, log: [], events: [], lastWaveBack: -9, lastStartle: -9,
      handF: {}, prevHead: [0, 0, 0], headVel: [0, 0, 0],
      priorOut: null,
    };
    // ---------------------------------------------------------------- action events, the move/object engine
    const MovesLib = opts.moves || (typeof globalThis !== 'undefined' && globalThis.AvatarMovesLib) || null;
    S.actions = []; S.cosp = false; S.pendingG = [];
    function emit(ev) {
      ev = Object.assign({t: +S.t.toFixed(3), wall: Date.now()}, ev);
      if (ev.src === undefined) ev.src = 'auto';
      S.actions.push(ev); if (S.actions.length > 400) S.actions.splice(0, 200);
      if (S.kf && S.kf.length && ev.id != null && (ev.kind === 'object' || ev.kind === 'move')) kfEngineEvent(ev);   // a "do" step's job
      if (typeof opts.onAction === 'function') { try { opts.onAction(ev); } catch (e) { /* the page's problem */ } }
      return ev;
    }
    const engine = MovesLib && MovesLib.createEngine ? MovesLib.createEngine({rnd, now: () => S.t, emit}) : null;
    let engineOn = false;   // only a character with a body file (config.body: the muse) gets whole-body moves / props
    // affect: the feeling's sustained face + posture state (app/affect.js); nothing runs until a verdict arrives
    const AffLib = opts.affect || (typeof globalThis !== 'undefined' && globalThis.AvatarAffectLib) || null;
    let affSide = 0;
    const AFF = AffLib && AffLib.create ? AffLib.create({rnd: () => (affSide++ % 2) * 0.99}) : null;
    function mkPros() {
      return {env: 0, peak: 0.03, vmean: 0.01, silentT: 9, voicedT: 0, speaking: false, phraseT: 0, prevLog: -9,
        lastOnset: -9, onset: 0, stress: 0, pref: Math.log(180), prise: 0, phraseStart: false, phraseEnd: 0,
        lvl: 0, everVoiced: false};
    }

    // ---------------------------------------------------------------- prosody (avatar voice + user voice)
    function prosody(P, v, dt, gateOff, th) {
      let rms = Math.max(0, +(v && v.rms) || 0);
      // a level that has not changed at all for 1 s is a stalled meter (no audio frames), not a voice
      if (rms > 0 && rms === P.lastRms) { P.sameT = (P.sameT || 0) + dt; if (P.sameT > 1.0) rms = 0; } else P.sameT = 0;
      P.lastRms = v ? Math.max(0, +(v.rms) || 0) : 0;
      // the level decides (a module's voiced flag carries its own long hangover); the flag alone if no level
      const voicedIn = v && (rms > 0 ? rms > th : v.voiced === true);
      const voiced = !!(!gateOff && voicedIn);
      if (v && v.pitch) P.hasPitch = true;
      P.env = rms > P.env ? lp(P.env, rms, dt, 0.025) : lp(P.env, rms, dt, 0.09);
      P.peak = Math.max(0.02, Math.max(P.peak * Math.exp(-dt / 10), P.env));
      P.lvl = clamp(P.env / P.peak, 0, 1);
      P.phraseStart = false; P.phraseEnd = 0; P.onset = 0;
      const was = P.speaking;
      if (voiced) { P.silentT = 0; P.voicedT += dt; P.everVoiced = true; P.vmean = lp(P.vmean, P.env, dt, 2.5); }
      else { P.silentT += dt; }
      P.speaking = P.everVoiced && P.silentT < 0.32;
      if (P.speaking && !was) { P.phraseStart = true; P.phraseT = 0; }
      if (P.speaking) P.phraseT += dt;
      if (!P.speaking && was) { P.phraseEnd = Math.max(0.05, P.phraseT); P.voicedT = 0; }
      const lg = Math.log(P.env + 1e-4);
      const rise = (lg - P.prevLog) / Math.max(dt, 1e-3);
      P.prevLog = lg;
      if (voiced && rise > 5.5 && P.lvl > 0.22 && S.t - P.lastOnset > 0.14) {
        P.lastOnset = S.t; P.onset = clamp(rise / 14, 0.2, 1);
        P.stress = clamp(P.env / Math.max(1e-3, P.vmean * 1.6), 0, 1.5);
      }
      const f0 = +(v && v.pitch) || 0;
      if (voiced && f0 > 60 && f0 < 700) {
        const l = Math.log(f0);
        P.prise = 12 * (l - P.pref) / Math.LN2;
        P.pref = lp(P.pref, l, dt, 3.5);
      } else P.prise = lp(P.prise, 0, dt, 0.2);
    }

    // ---------------------------------------------------------------- perception → user state
    function perceive(f, dt) {
      const u = S.user;
      const face = f && f.face && f.face.present ? f.face : null;
      if (face) {
        u.presentT += dt; u.absentT = 0;
        if (!u.present && u.presentT > 0.15) u.present = true;
        const ax = Number.isFinite(face.ax) ? face.ax : Math.atan((+face.x || 0) * TAN_HFOV2);
        const ayRaw = Number.isFinite(face.ay) ? face.ay : Math.atan((+face.y || 0) * TAN_VFOV2);
        // the phone camera sits above the screen and selfie framing is rarely centred: adapt the vertical centre slowly
        u.ayBase = clamp(u.ayBase === undefined ? ayRaw : lp(u.ayBase, ayRaw, dt, 20), -0.2, 0.2);
        const ay = ayRaw - 0.8 * u.ayBase;
        u.ax = ax; u.ay = ay; u.x = +face.x || 0; u.y = +face.y || 0; u.z = +face.z || u.z;
        u.lookX = +face.lookX || 0; u.lookY = +face.lookY || 0;
        // postural mirroring: his head tilt, slowly and partly (1.2 s lag); his distance → personal-space response
        if (!face.xp) u.rollM = lp(u.rollM || 0, clamp(+face.roll || 0, -0.4, 0.4), dt, 1.2);
        u.zS = lp(u.zS ?? (+face.z || 0.45), clamp(+face.z || u.zS || 0.45, 0.1, 2), dt, 0.6);
        u.zBase = lp(u.zBase ?? u.zS, u.zS, dt, 25);   // his usual distance on this device
        const att = f.attending !== undefined ? !!f.attending : (f.attention === undefined ? true : f.attention > 0.5);
        if (att) { u.attT += dt; u.awayT = 0; } else { u.awayT += dt; u.attT = 0; }
        u.attending = att;
        mirrorIn(face, dt);
      } else {
        u.absentT += dt; u.presentT = 0;
        if (u.present && u.absentT > 0.6) u.present = false;
        u.attending = false; u.awayT += dt; u.attT = 0;
        u.rollM = lp(u.rollM || 0, 0, dt, 1.5);
        mirrorIn(null, dt);
      }
      u.hands = (f && f.hands) || [];
      greetDetect(f, face, dt);
      const energy = f ? clamp(+f.motion || 0, 0, 1) : 0;
      u.energy = lp(u.energy, energy, dt, 1.2);
      // his boredom: still, not attending, little expression, yawning
      const ex = S.expr.m;
      const expressive = ex.smile + ex.browUp + ex.eyeWide + ex.sad;
      const yawn = face && face.expr ? clamp(((face.expr.jawOpen || 0) - 0.5) * 3, 0, 1) * clamp((face.expr.blink || 0) * 1.5, 0, 1) : 0;
      const drive = (u.present ? 0 : 0.3) + (u.attending ? 0 : 0.35) + clamp(0.25 - u.energy * 4, 0, 0.25) +
        clamp(0.15 - expressive, 0, 0.15) + 0.6 * yawn - (S.uv.speaking ? 0.6 : 0);
      u.boredom = lp(u.boredom, clamp(drive, 0, 1), dt, drive > u.boredom ? 8 : 2);
    }

    // perception's tracker already measures his face against HIS resting face (scores 0..1 per readable state):
    // mirror those, attenuated, after a human reaction lag — never an instant copy
    function mirrorXp(xp, dt) {
      const E = S.expr; E.xpMode = true;
      const s = xp ? xp.scores : {}, g = cfg.xpGain, k = (x) => clamp(+x || 0, 0, 1);
      const target = {smile: k(s.happy) * g.smile, squint: k(s.happy) * g.squint, sad: k(s.sad) * g.sad, anger: k(s.frown) * g.anger,
        browDown: k(s.frown) * g.browDown, surprise: k(s.surprised) * g.surprise, eyeWide: k(s.surprised) * g.eyeWide,
        browUp: Math.max(k(s.raised_brows), 0.5 * k(s.surprised)) * g.browUp, fear: 0, jawOpen: 0};
      E.xbuf = E.xbuf || []; E.xbuf.push([S.t, target]);
      while (E.xbuf.length > 2 && E.xbuf[1][0] <= S.t - cfg.xpLag) E.xbuf.shift();
      const delayed = E.xbuf[0][0] <= S.t - cfg.xpLag ? E.xbuf[0][1] : null;
      for (const c in E.m) {
        const v = delayed && delayed[c] !== undefined ? delayed[c] * K.mirror : 0;
        E.m[c] = lp(E.m[c], v, dt, v > E.m[c] ? cfg.xpTau : cfg.xpTau * 1.8);
      }
      // his head tilt, relative to his own rest, mirrored (same screen side) — faster and fuller than the absolute-roll path
      S.user.rollM = lp(S.user.rollM || 0, clamp(xp ? 0.5 / 0.35 * (+xp.tilt || 0) : 0, -0.4, 0.4), dt, 0.35);
    }

    // A hand raised in greeting, robust to 6–8 fps hand tracking that flickers (perception's own 'wave' event needs
    // wrist reversals it rarely sees): an OPEN hand of his beside his face (1.4–4.5 face widths out — a hand on or near
    // the face is a chin rest, a scratch or a phone) at about chin height or above, or his pose wrist above shoulder and
    // elbow there; evidence accumulates while seen (decays at a third of that while not); it fires once ≥ 0.25 s with
    // the hand moving ≥ 0.08 RELATIVE TO HIS FACE (walking carries the hand along); then it re-arms only after the hand
    // has been down ≥ 1.5 s (one wave back per greeting, never a 6 s loop on a hand that stays up), ≥ 10 s apart, and
    // not in his first second on camera. The wave back follows after a human reaction delay (0.2–0.35 s).
    function greetDetect(f, face, dt) {
      const G = S.greetD || (S.greetD = {ev: 0, xs: [], fired: -99, armed: true, downT: 9});
      let cand = null;
      if (face && f && S.user.presentT > 1.0) {
        const fx = +face.x || 0, chin = Number.isFinite(+face.chinY) ? +face.chinY : (+face.y || 0) - 0.25, fw = 2 * Math.max(0.04, (+face.size || 0.1));
        const beside = x => Math.abs(x - fx) >= 1.4 * fw && Math.abs(x - fx) <= 4.5 * fw;
        for (const h of f.hands || []) {
          if ((h.age ?? 0) > 0.35 || h.main === false) continue;
          if (['Pointing_Up', 'Thumb_Up', 'Closed_Fist', 'Victory', 'ILoveYou'].includes(h.gesture)) continue;
          if ((+h.open || 0) >= 0.6 && (+h.y || 0) >= chin - 0.15 * fw && beside(+h.x || 0)) { cand = {x: +h.x, y: +h.y, rx: (+h.x || 0) - fx, side: h.side}; break; }
        }
        const P = f.pose;
        if (!cand && P && P.present && P.shoulders && P.wrists && P.elbows) for (const sd of ['l', 'r']) {
          const sh = P.shoulders[sd], wr = P.wrists[sd], el = P.elbows[sd];
          if (!sh || !wr || !el || Math.abs(+wr.y) > 1.0 || Math.abs(+wr.x) > 1.0) continue;
          if (+wr.y > +sh.y + 0.05 && +wr.y > +el.y + 0.05 && +wr.y >= chin && beside(+wr.x)) { cand = {x: +wr.x, y: +wr.y, rx: +wr.x - fx, side: sd}; break; }
        }
      }
      if (cand) { G.ev = Math.min(1, G.ev + dt); G.downT = 0; G.xs.push([S.t, cand.rx]); G.last = cand; }
      else { G.ev = Math.max(0, G.ev - 0.35 * dt); G.downT += dt; }
      if (G.downT > 1.5) G.armed = true;
      while (G.xs.length && G.xs[0][0] < S.t - 1.2) G.xs.shift();
      const xs = G.xs.map(q => q[1]), moved = xs.length ? Math.max(...xs) - Math.min(...xs) : 0;
      if (G.armed && G.ev >= 0.25 && moved >= 0.08 && S.t - G.fired > 10 && S.user.attending !== false) {
        G.fired = S.t; G.ev = 0; G.armed = false;
        S.greetAt = {at: S.t + U(0.2, 0.35), side: G.last.side, x: G.last.x, y: G.last.y};
      }
    }

    function mirrorIn(face, dt) {
      if (face && face.xp && face.xp.scores) return mirrorXp(face.xp, dt);
      if (!face && S.expr.xpMode) return mirrorXp(null, dt);
      const E = S.expr, B = face && face.blend ? face.blend : null, x = face && face.expr ? face.expr : null;
      const g = (k) => B ? (+B[k] || 0) : 0;
      let raw;
      if (face && (B || x)) {
        const avg = (a, b) => (g(a) + g(b)) / 2;
        raw = {
          smile: x && x.smile !== undefined ? x.smile : avg('mouthSmileLeft', 'mouthSmileRight'),
          squint: x && x.squint !== undefined ? x.squint : 0.6 * avg('eyeSquintLeft', 'eyeSquintRight') + 0.4 * avg('cheekSquintLeft', 'cheekSquintRight'),
          browUp: x && x.browUp !== undefined ? x.browUp : Math.max(g('browInnerUp'), avg('browOuterUpLeft', 'browOuterUpRight')),
          browDown: x && x.browDown !== undefined ? x.browDown : avg('browDownLeft', 'browDownRight'),
          eyeWide: x && x.eyeWide !== undefined ? x.eyeWide : avg('eyeWideLeft', 'eyeWideRight'),
          sad: x && x.sad !== undefined ? x.sad : clamp(0.7 * avg('mouthFrownLeft', 'mouthFrownRight') + 0.3 * g('browInnerUp') * (1 - avg('mouthSmileLeft', 'mouthSmileRight')), 0, 1),
          jawOpen: x && x.jawOpen !== undefined ? x.jawOpen : g('jawOpen'),
        };
        // composites the single channels miss (they drive brows, ears, eyes, head and body together)
        const sm = raw.smile, jo = S.uv.speaking ? 0 : raw.jawOpen;
        // surprise raises the OUTER brows and drops the jaw; fear draws the brows together and stretches the mouth
        const outerUp = avg('browOuterUpLeft', 'browOuterUpRight'), stretch = avg('mouthStretchLeft', 'mouthStretchRight');
        raw.surprise = clamp(0.55 * raw.eyeWide + 0.45 * raw.browUp + 0.4 * jo - 0.35 * sm - 0.4 * raw.browDown - 0.4 * stretch, 0, 1);
        raw.fear = clamp(0.4 * raw.eyeWide + 0.3 * g('browInnerUp') + 0.35 * raw.browDown + 0.5 * stretch - 0.45 * outerUp - 0.4 * sm, 0, 1);
        raw.anger = clamp(0.85 * raw.browDown + 0.45 * avg('noseSneerLeft', 'noseSneerRight') + 0.25 * raw.squint + 0.3 * avg('mouthPressLeft', 'mouthPressRight') - 0.45 * sm, 0, 1);
      } else raw = null;
      // per-person neutral baseline: follows downward fast, upward very slowly (a long smile is not "neutral")
      const out = {};
      for (const k in E.m) {
        if (!raw) { out[k] = 0; continue; }
        const r = clamp(+raw[k] || 0, 0, 1);
        if (E.base[k] === undefined) E.base[k] = Math.min(r, 0.15);
        E.base[k] = r < E.base[k] ? lp(E.base[k], r, dt, 1.0) : lp(E.base[k], r, dt, 40);
        E.base[k] = Math.min(E.base[k], 0.3);
        const v = clamp((r - E.base[k]) / (1 - E.base[k]), 0, 1);
        out[k] = clamp((v - 0.06) / 0.94, 0, 1);   // dead zone: blendshape noise does not make the face chatter
      }
      E.buf.push([S.t, out]);
      while (E.buf.length > 2 && E.buf[1][0] <= S.t - cfg.mirrorLag) E.buf.shift();
      const delayed = E.buf[0][0] <= S.t - cfg.mirrorLag ? E.buf[0][1] : null;
      for (const k in E.m) {
        const target = delayed ? delayed[k] * (cfg.mirrorGain[k] ?? 0.5) * K.mirror : 0;
        E.m[k] = lp(E.m[k], target, dt, target > E.m[k] ? cfg.mirrorTau : cfg.mirrorTau * 1.4);
      }
      // speaking jaw is not surprise
      if (S.uv.speaking) E.m.jawOpen = lp(E.m.jawOpen, 0, dt, 0.15);
    }

    // ---------------------------------------------------------------- events from perception (reflexes)
    function event(a, b) {
      // accepts event({type, ...}) or event(type, detail) (the page's avatarEvent); the same perception event may
      // arrive both ways (features.events and onEvent) — deduplicated by (type, t)
      const ev = typeof a === 'string' ? Object.assign({}, b || {}, {type: a}) : a;
      if (!ev || !ev.type) return;
      if (ev.t !== undefined) {
        const key = ev.type + ':' + ev.t;
        S.seenEv = S.seenEv || [];
        if (S.seenEv.includes(key)) return;
        S.seenEv.push(key); if (S.seenEv.length > 40) S.seenEv.shift();
      }
      const t = S.t;
      S.events.push({t, type: ev.type});
      if (S.events.length > 200) S.events.splice(0, 100);
      const interact = () => { S.idleT = 0; S.bored = Math.max(0, S.bored - 0.5); };
      switch (ev.type) {
        case 'wave': {
          // a wave meant for us comes from someone looking at us; one wave back per greeting (extend ≤ 2 s), then a pause
          if (S.user.present && S.user.awayT > 1.0) break;
          // a wave with no face tracked counts only from people who have been out of face range for a while (distant
          // wavers); a face lost a moment ago is someone walking, turning or leaving — their arm swing is not a greeting
          if (!S.user.present && ev.source !== 'raise' && S.user.absentT < 3.0) break;
          // ... and only once it is confirmed: a single pose-only swing (someone walking, putting something on) is not a
          // wave; real distant waving keeps coming (perception re-emits it every ~1.5 s), so wait for a second event ≤ 3 s
          if (!S.user.present && ev.source !== 'raise') {
            if (!(t - (S.farWaveAt ?? -99) <= 3.0)) { S.farWaveAt = t; break; }
          }
          interact();
          const side = ev.side === 'l' || ev.side === 'r' ? ev.side : 'r';   // mirror: wave the paw on HIS screen side
          // no face tracked (someone waving from afar): look toward the waving hand
          if (!S.user.present) {
            let W = null;
            if (Number.isFinite(ev.ax) && Number.isFinite(ev.ay)) W = [ev.ax * 0.8, ev.ay * 0.8];
            else if (Number.isFinite(ev.x) && Number.isFinite(ev.y)) W = [Math.atan(ev.x * TAN_HFOV2) * 0.8, Math.atan(ev.y * TAN_VFOV2) * 0.8];
            else W = [(side === 'r' ? 1 : -1) * 0.12, 0.02];
            S.around = null; S.focus = {kind: 'object', W, until: t + 3.0, gain: 0.7};
          }
          const episode = t - (S.lastWaveEv ?? -99) < 5.0;   // waves keep coming = the same greeting
          S.lastWaveEv = t;
          const running = S.gestures.find(g => g.name === 'wave' && !g.rel && g.src === 'reflex');
          if (running) { if (!running.ext) { running.ext = true; running.dur = Math.min(running.dur + 2, Math.max(running.dur, (t - running.t0) + 1.6)); } break; }
          if (episode || t - S.lastWaveBack < 6.0) {   // already waved back for this greeting: a smile and a little nod
            if (t - (S.lastGreetNod ?? -99) > 4) { S.lastGreetNod = t; S.greet.t0 = t; startGesture('nod', {cycles: 1, amp: 0.4, src: 'reflex'}); }
            break;
          }
          S.lastWaveBack = t;
          if (!heldBy('hand_' + side + '_y')) startGesture('wave', {side, cycles: 4, src: 'reflex'});
          emit({kind: 'reflex', name: 'wave_back', src: 'reflex', phase: 'end', did: 'waved back at him (he waved)'});
          S.around = null; S.avert = null;
          S.greet.t0 = t; S.focus = {kind: 'user', until: t + 2.5, gain: cfg.headFollow};
          if (chance(0.5)) S.perkK.hit(4);
          break;
        }
        case 'approach_fast': {
          if (t - S.lastStartle < 2.0) break;
          interact();
          S.lastStartle = t;
          const amp = clamp(0.75 + 0.25 * ((+ev.rate || 1) - 0.9), 0.7, 1.15);
          S.startle = {t0: t, amp, eyeWide: 0};
          S.startleK.hit(21 * amp * ((engineOn && engine.wholeBody()) || (S.kf.length && kfBusy().body) ? 0.3 : 1));
          forceBlink(1.0, true, true, true);
          emit({kind: 'reflex', name: 'startle', src: 'reflex', phase: 'end', did: 'flinched (something came at the camera fast)'});
          S.focus = {kind: 'user', until: t + 1.5, gain: 0.3};
          S.around = null; S.avert = null;
          if (!heldBy('hand_l_y') && !(engineOn && engine.wholeBody()) && !(S.kf.length && kfBusy().any)) startGesture('startle_hands', {src: 'reflex'});
          break;
        }
        case 'face_appear': {
          interact();
          S.perk = {t0: t, amp: 1}; S.perkK.hit(7);
          emit({kind: 'reflex', name: 'perk', src: 'reflex', phase: 'end', did: 'perked up when he came into view'});
          S.focus = {kind: 'user', until: t + 2.0, gain: cfg.headFollow};
          S.around = null;
          if (chance(0.6)) forceBlink(1, false, false, true);
          S.flash = {t0: t + 0.15, amp: 0.45};
          break;
        }
        case 'face_lost': break;
        case 'look_back': {
          interact();
          S.avert = null; S.around = null;
          S.focus = {kind: 'user', until: t + 1.5, gain: cfg.headFollow, fast: true};
          emit({kind: 'reflex', name: 'look_back', src: 'reflex', phase: 'end', did: 'looked back at him when he looked back'});
          S.flash = {t0: t + 0.1, amp: 0.35 + 0.1 * clamp((+ev.away_s || 2) / 6, 0, 1)};
          if (chance(0.35)) forceBlink(1, false, false, true);
          S.joint.done = false;
          break;
        }
        case 'point': {
          interact();
          const d = ev.dir || {x: 0, y: 0};
          let W;
          if (ev.at === 'screen') {
            const h = (S.user.hands || []).find(h => h.side === ev.side) || S.user.hands[0];
            W = h ? [Math.atan((+h.x || 0) * TAN_HFOV2) * 0.8, Math.atan((+h.y || 0) * TAN_VFOV2) * 0.8 - 0.05] : [0, -0.1];
          } else W = [clamp((+d.x || 0) * 0.45, -0.45, 0.45), clamp((+d.y || 0) * 0.3, -0.25, 0.3)];
          S.focus = {kind: 'point', W, until: t + U(1.6, 2.4), gain: 0.85, then: 'user'};
          emit({kind: 'reflex', name: 'look_where_pointed', src: 'reflex', phase: 'end', did: 'looked where he pointed'});
          S.avert = null; S.around = null;
          // ears turn toward the pointed side
          S.earPoint = {t0: t, side: W[0]};
          break;
        }
        case 'object_up': {
          interact();
          const hx = +ev.x || 0, hy = +ev.y || 0;
          S.focus = {kind: 'object', W: [Math.atan(hx * TAN_HFOV2) * 0.9, Math.atan(hy * TAN_VFOV2) * 0.9], until: t + U(2.0, 3.0), gain: 0.7,
            curious: true, t0: t};
          S.perk = {t0: t, amp: 0.7}; S.perkK.hit(5);
          emit({kind: 'reflex', name: 'curious_look', src: 'reflex', phase: 'end', did: 'looked curiously at what he held up'});
          break;
        }
        case 'thumbs_up': {
          interact();
          S.greet.t0 = t;
          if (chance(0.6)) startGesture('nod', {cycles: 1, amp: 0.6, src: 'reflex'});
          else if (!heldBy('hand_l_y')) startGesture('thumbs_up', {side: ev.side === 'l' ? 'l' : 'r', dur: 1.6, src: 'reflex'});
          break;
        }
        case 'attention_lost': S.joint.done = false; break;
        case 'nod': {   // he nods → nod back after a beat (a small one if we are talking: agreement noted)
          if (t - (S.lastNodBack ?? -9) < 4 || S.uv.speaking || !S.user.attending || !chance(0.75)) break;
          S.lastNodBack = t; interact(); S.log.push({t: +t.toFixed(2), nodback: 1});
          S.nodBackAt = {at: t + U(0.2, 0.35), amp: S.av.speaking ? 0.35 : 0.6};
          break;
        }
        case 'head_shake': {   // he shakes his head → a quizzical tilt with one brow up, a touch of concern
          if (t - (S.lastShakeBack ?? -9) < 4 || !S.user.attending) break;
          S.lastShakeBack = t; interact();
          S.shakeBackAt = t + U(0.2, 0.35);
          break;
        }
      }
    }

    // ---------------------------------------------------------------- blinks
    function priorBlinks() { return !!(prior && cfg.priorWeight > 0 && S.priorOut && S.priorOut.blink_l !== undefined); }
    function forceBlink(amp = 1, now = false, override = false, rx = false) {
      // rx: a reflex / commanded blink (startle, face appears, look back, "blink", a saccade the reflex gaze makes) —
      // learned mode keeps those on the page and floors the learned lids with them; rules mode ignores the flag
      const b = S.blink;
      if (b.phase !== 0 && override) {   // restart a blink in progress from its current closure (no jump)
        const cur = S.lastOut ? S.lastOut.blink_l : 0;
        b.phase = 1; b.t0 = S.t - 0.075 * clamp(cur, 0, 1) * 0.6; b.amp = Math.max(amp, cur); b.dbl = false; b.rx = !!rx; return;
      }
      if (b.phase !== 0) return;
      if (!now && S.t < b.refr) return;
      b.phase = 1; b.t0 = S.t; b.amp = amp; b.dbl = chance(0.07); b.rx = !!rx;
    }
    function blinkStep(dt) {
      const b = S.blink;
      const ibi = S.av.speaking ? cfg.blinkIBI.speaking : S.bored > 0.5 ? cfg.blinkIBI.bored :
        S.uv.speaking ? cfg.blinkIBI.listening : cfg.blinkIBI.idle;
      if (b.phase === 0) {
        b.next -= dt * (priorBlinks() ? 0 : 1);   // a prior that blinks owns the spontaneous blink clock
        if (b.next <= 0 && S.t >= b.refr) { forceBlink(chance(0.12) ? U(0.6, 0.85) : 1, true); }
        S.rxBlink = 0;
        return 0;
      }
      const u = S.t - b.t0, close = 0.075, hold = 0.035, open = 0.16;
      let v;
      if (u < close) v = mj(u / close);
      else if (u < close + hold) v = 1;
      else if (u < close + hold + open) v = 1 - mj((u - close - hold) / open);
      else {
        v = 0; b.phase = 0; b.refr = S.t + 0.3;
        if (b.dbl) { b.dbl = false; b.phase = 1; b.t0 = S.t + 0.06; b.amp = U(0.7, 1); }
        else b.next = clamp(logn(ibi, 0.55), 0.5, 11);
      }
      S.rxBlink = b.rx ? v * b.amp : 0;
      return v * b.amp;
    }

    // ---------------------------------------------------------------- gaze target selection
    function chooseTarget(dt) {
      const t = S.t, u = S.user;
      const userW = () => [cfg.gazeFollow * u.ax, cfg.gazeFollow * u.ay];
      const userH = (g = cfg.headFollow) => [g * u.ax, 0.8 * g * u.ay];
      // 1. a commanded look
      if (S.cmdLook) {
        const c = S.cmdLook;
        if (!c.hold && t > c.until) S.cmdLook = null;
        else if (c.target === 'user' || c.target === 'me' || c.target === 'camera') return {mode: 'cmd', W: u.present ? userW() : [0, 0], H: u.present ? userH() : [0, 0], gain: cfg.headFollow, hold: true};
        else return {mode: 'cmd', W: lookTargetW(c.target, c), gain: 0.85, hold: true};
      }
      // 2. scripted look-around (command gesture or boredom)
      if (S.around) {
        const a = S.around, seg = a.seq[a.i];
        if (t - a.t0 > seg.dur) { a.i++; a.t0 = t; if (a.i >= a.seq.length) S.around = null; }
        if (S.around) return {mode: 'around', W: S.around.seq[S.around.i].W, gain: S.around.gain};
      }
      // 3. reflex focus (point / object / greeting)
      if (S.focus) {
        const fo = S.focus;
        if (t > fo.until) S.focus = null;
        else if (fo.kind === 'user') { if (u.present) return {mode: 'user', W: userW(), H: userH(fo.gain), gain: fo.gain, fast: fo.fast}; }
        else return {mode: fo.kind, W: fo.W, gain: fo.gain};
      }
      // 4. a glance away (speaking aversion, thinking, listener glance, mutual-gaze break)
      if (S.avert) {
        if (t > S.avert.until) S.avert = null;
        else return {mode: 'avert', W: S.avert.W, gain: S.avert.gain};
      }
      if (u.present) {
        // 5. joint attention: he looks away → sometimes look where he looks, briefly
        if (!u.attending && u.awayT > 0.7 && !S.joint.done) {
          S.joint.done = true;
          if (chance(0.5) && (Math.abs(u.lookX) > 0.25 || Math.abs(u.lookY) > 0.25)) {
            const W = [clamp(u.lookX * 0.9, -0.4, 0.4), clamp(u.lookY * 0.7, -0.25, 0.25)];
            S.avert = {W, until: t + U(0.8, 1.6), gain: 0.7};
            return {mode: 'avert', W, gain: 0.7};
          }
        }
        return {mode: 'user', W: userW(), H: userH(), gain: cfg.headFollow};
      }
      // 6. nobody there: wander, with glances back to the middle
      if (u.absentT > 1.5 && !S.around) {
        const seq = [];
        for (let i = 0; i < 3; i++) seq.push({W: [U(-0.32, 0.32), U(-0.12, 0.16)], dur: U(1.0, 2.6)});
        seq.push({W: [0, 0], dur: U(1.2, 2.5)});
        S.around = {seq, i: 0, t0: t, gain: 0.7, kind: 'wander'};
        return {mode: 'around', W: seq[0].W, gain: 0.7};
      }
      return {mode: 'user', W: [0, 0], gain: cfg.headFollow};
    }
    function lookTargetW(target, c) {
      const u = S.user;
      switch (target) {
        case 'user': case 'me': case 'camera': return u.present ? [cfg.gazeFollow * u.ax, cfg.gazeFollow * u.ay] : [0, 0];
        case 'screen_left': case 'l': return [-0.42, 0.02];
        case 'screen_right': case 'r': return [0.42, 0.02];
        case 'left': return [0.42, 0.02];     // the character's own left = screen-right
        case 'right': return [-0.42, 0.02];
        case 'up': return [0, 0.3];
        case 'down': return [0, -0.28];
        case 'up_left': return [0.35, 0.25];
        case 'up_right': return [-0.35, 0.25];
        case 'down_left': return [0.35, -0.25];
        case 'down_right': return [-0.35, -0.25];
        case 'away': if (!c.W) c.W = [pick([-1, 1]) * U(0.3, 0.42), U(-0.15, 0.12)]; return c.W;
        case 'point_dir': case 'point': {
          const last = S.lastPoint; return last ? last : [0, 0];
        }
        default: {
          if (Array.isArray(target)) return [clamp(+target[0] || 0, -0.5, 0.5), clamp(+target[1] || 0, -0.35, 0.35)];
          const P = lookProp(target);                    // a thing in the scene, followed while it moves
          if (P) return P.W;
          return [0, 0];
        }
      }
    }
    // look(target): a prop by id or name ('chair', 'the laptop', 'pink headphones') -> gaze target on the screen
    // plane (+x = screen right = the character's own left, as lookTargetW 'left'), from its position relative to the body
    function lookProp(target) {
      if (!engineOn || typeof target !== 'string' || !target) return null;
      const q = target.toLowerCase().replace(/^(at |the |my |your |that |this )+/, '').replace(/_/g, ' ').trim();
      if (!q) return null;
      let best = null;
      for (const p of engine.props()) {
        const id = String(p.id || '').toLowerCase(), nm = String(p.name || '').toLowerCase();
        const score = id === q || nm === q ? 3 : nm.split(/\s+/).includes(q) || q.split(/\s+/).includes(id) ? 2 : 0;
        if (score && (!best || score > best.score)) best = {p, score};
      }
      if (!best) return null;
      const r = engine.state().root || {x: 0, y: 0, z: 0}, pos = best.p.pos || [0, 0, 0];
      return {id: best.p.id, name: best.p.name, W: [clamp(0.75 * (pos[0] - (r.x || 0)), -0.45, 0.45), clamp(0.6 * (pos[1] - 0.45 - (r.y || 0)), -0.3, 0.25)]};
    }

    // ---------------------------------------------------------------- gaze dynamics: eyes lead, head follows
    function gazeStep(dt) {
      const t = S.t, E = S.eye;
      const tg = chooseTarget(dt);
      if (tg.mode === 'point' || tg.mode === 'object') S.lastPoint = tg.W;
      const W = [tg.W[0], tg.W[1]];
      // micro-saccades + drift while fixating
      E.microNext -= dt;
      if (E.microNext <= 0) { E.microNext = U(0.45, 1.4); E.micro = [N() * 0.009, N() * 0.007]; }
      const Wm = [W[0] + E.micro[0], W[1] + E.micro[1]];
      const changed = tg.mode !== E.lastMode;
      E.lastMode = tg.mode;
      const inSac = E.sacT0 >= 0 && t < E.sacT0 + E.sacDur;
      const err = Math.hypot(W[0] - E.goal[0], W[1] - E.goal[1]);
      if (!inSac) {
        if (E.pending && t >= E.pending.at) {
          fireSaccade(W, tg); E.pending = null;
        } else if (!E.pending && err > 0.032) {
          // perception-driven retarget (he moved): human reaction latency; internal decisions go at once
          if (tg.mode === 'user' && !changed && !tg.fast) E.pending = {at: t + U(cfg.reactLatency[0], cfg.reactLatency[1])};
          else fireSaccade(W, tg);
        } else if (!E.pending) {
          E.goal = [lp(E.goal[0], W[0], dt, 0.12), lp(E.goal[1], W[1], dt, 0.12)];   // smooth pursuit of small drifts
        }
      }
      // micro offsets jump (they are saccades too) — apply with a 40 ms min-jerk of their own
      if (E.sacT0 >= 0 && t < E.sacT0 + E.sacDur) {
        const s = mj((t - E.sacT0) / E.sacDur);
        E.W = [lerp(E.from[0], E.to[0], s), lerp(E.from[1], E.to[1], s)];
      } else {
        const g = [E.goal[0] + E.micro[0], E.goal[1] + E.micro[1]];
        E.W = [lp(E.W[0], g[0], dt, 0.03), lp(E.W[1], g[1], dt, 0.03)];
      }
      // head: follows the eye goal, a beat later
      if (E.headPending && t >= E.headPending.at) { S.headT = E.headPending.T; S.headGain = E.headPending.g; E.headPending = null; S.headLastSet = t; }
      if (!E.headPending && !E.pending && !inSac) {
        const want = tg.H ? tg.H : [tg.gain * E.goal[0], tg.gain * 0.75 * E.goal[1]];
        if (Math.hypot(want[0] - S.headT[0], want[1] - S.headT[1]) > 0.03 || t - S.headLastSet > 1.0) { S.headT = want; S.headLastSet = t; }
      }
      const hx = S.hy.step(clamp(S.headT[0], -cfg.headMax, cfg.headMax), dt);
      const hyv = S.hp.step(clamp(S.headT[1], -0.3, 0.3), dt);
      return {mode: tg.mode, head: [hx, hyv]};
    }
    function fireSaccade(W, tg) {
      const E = S.eye, t = S.t;
      const amp = Math.hypot(W[0] - E.W[0], W[1] - E.W[1]);
      E.from = [E.W[0], E.W[1]]; E.to = [W[0] + E.micro[0], W[1] + E.micro[1]]; E.goal = [W[0], W[1]];
      E.sacT0 = t; E.sacDur = clamp(0.045 + 0.0029 * amp * 57.3, 0.045, 0.15);
      const g = tg.gain;
      const T = tg.H ? [tg.H[0], tg.H[1]] : [g * W[0], g * 0.75 * W[1]];
      const headShift = Math.hypot(T[0] - S.headT[0], T[1] - S.headT[1]);
      E.headPending = {at: t + 0.03 + 0.02 * rnd(), T, g};
      if (headShift > 0.16 && chance(0.55)) forceBlink(1, false, false, gazeOwnedNow());   // blinks ride big gaze shifts
      else if (amp > 0.2 && chance(0.3)) forceBlink(1, false, false, gazeOwnedNow());
    }

    // ---------------------------------------------------------------- behaviour schedulers (speaking / listening / boredom)
    function behaviours(dt) {
      const t = S.t, av = S.av, uv = S.uv, u = S.user, sp = S.speech;
      if (S.nodBackAt && t >= S.nodBackAt.at) { const a = S.nodBackAt.amp; S.nodBackAt = null; if (!heldBy('head_pitch')) { startGesture('nod', {cycles: 1, amp: a, src: 'reflex'}); emit({kind: 'reflex', name: 'nod_back', src: 'reflex', phase: 'end', did: 'nodded back when he nodded'}); } }
      if (S.shakeBackAt && t >= S.shakeBackAt) { S.shakeBackAt = null; if (!heldBy('head_roll')) { startGesture('quizzical', {src: 'reflex'}); emit({kind: 'reflex', name: 'quizzical', src: 'reflex', phase: 'end', did: 'tilted its head, puzzled, when he shook his head'}); } }
      if (S.greetAt && t >= S.greetAt.at) { const g = S.greetAt; S.greetAt = null; event({type: 'wave', side: g.side, x: g.x, y: g.y, source: 'raise'}); }
      // -- the avatar speaking
      if (av.phraseStart) {
        S.idleT = 0;
        if (!S.cmdLook && av.silentT === 0 && chance(0.62)) {
          const side = pick([-1, 1]);
          S.avert = {W: [side * U(0.16, 0.3), U(0.06, 0.2)], until: t + U(0.7, 1.8), gain: 0.4};   // thinking glance at a turn start
        }
        sp.aversNext = t + U(3, 6);
        S.nodK.hit(-0.25 * U(0.6, 1.2) * K.gest);   // small lift at a phrase start
        // speakers settle into a new head posture per phrase (Hadar et al.): hold it, nods ride on top
        S.phrasePose = {yaw: N() * 0.085 * K.amp, pitch: N() * 0.035 * K.amp, roll: N() * 0.045 * K.amp, next: t + U(1.8, 3.2)};
        S.tilt.target = clamp(S.tilt.target + N() * 0.04, -0.09, 0.09);
        S.breath.inhaleAt = t;
      }
      if (av.speaking) {
        S.idleT = 0;
        if (S.phrasePose && t > S.phrasePose.next && chance(0.45)) S.phrasePose = {yaw: lerp(S.phrasePose.yaw, N() * 0.085 * K.amp, 0.7), pitch: N() * 0.03 * K.amp, roll: N() * 0.04 * K.amp, next: t + U(1.8, 3.2)};
        else if (S.phrasePose && t > S.phrasePose.next) S.phrasePose.next = t + U(1.2, 2.4);
        if (t > sp.aversNext && !S.avert && !S.cmdLook) {
          sp.aversNext = t + U(3.5, 7);
          if (chance(0.55)) S.avert = {W: [cfg.gazeFollow * u.ax + pick([-1, 1]) * U(0.1, 0.22), U(-0.08, 0.14)], until: t + U(0.45, 1.1), gain: 0.35};
        }
        if (av.onset > 0) {
          const st = av.stress;
          if (st > 0.55 && t - sp.lastNod > 0.32 && chance(0.6)) {
            sp.lastNod = t;
            S.nodK.hit((0.55 + 0.9 * st * st) * U(0.7, 1.2) * K.gest);          // nod on the stressed syllable (nose down)
            S.swayK.hit(N() * 0.5 * st * K.gest);
          }
          if (av.prise > 1.5 && st > 0.45 && chance(0.55)) S.browK.hit(12 * clamp(av.prise / 5, 0.35, 1) * K.gest);   // brow raise on a pitch accent
          else if (st > (av.hasPitch ? 0.95 : 0.7) && chance(av.hasPitch ? 0.25 : 0.4)) S.browK.hit((6 + 3 * clamp(st - 0.7, 0, 1)) * K.gest);
          // beat gestures with a paw on strong accents, once he has been talking a while
          if (!S.cosp && st > (K.beat > 1 ? 0.7 : 0.8) && av.phraseT > 1.2 / K.beat && t > sp.beatNext && chance(Math.min(0.9, 0.35 * K.beat))) {
            sp.beatNext = t + U(0.9, 2.2) / K.beat;
            if (sp.beatUntil < t) { sp.beatSide = chance(0.65) ? 'l' : 'r'; sp.beatBoth = chance(0.3); }
            sp.beatUntil = t + U(1.6, 2.8);
            sp.beatHit = t;
          }
          if (chance(0.08)) S.earTw[chance(0.5) ? 'l' : 'r'] = 0;
        }
      }
      if (av.phraseEnd > 0) {
        if (S.avert && S.avert.gain < 0.5) S.avert = null;          // look back at him at the phrase end
        if (!priorBlinks() && chance(0.5)) forceBlink(1, false);
        S.nodK.hit(0.35 * K.gest);                                  // phrase-final lowering
        if (av.phraseEnd > 1.5) S.breath.inhaleAt = t;
      }
      // -- him speaking: listening
      if (uv.speaking) { S.idleT = 0; S.listen.speakT = (S.listen.speakT || 0) + dt; }
      if (uv.phraseEnd > 0) {
        const run = uv.phraseEnd;
        if (run > 1.0 && t - S.listen.lastNod > 2.2 && !av.speaking) {
          S.listen.lastNod = t;
          if (chance(0.55)) startGesture('nod', {cycles: chance(0.35) ? 2 : 1, amp: U(0.35, 0.6), src: 'listen'});
          if (chance(0.3)) S.greet.t0 = t - 0.3;   // tiny smile pulse
        }
        if (!priorBlinks() && chance(0.3)) forceBlink(1, false);   // listeners blink at the speaker's pauses
        S.listen.speakT = 0;
      }
      if (uv.speaking && t > S.tilt.next) {
        S.tilt.next = t + U(4, 9);
        S.listen.tiltSide = chance(0.7) ? S.listen.tiltSide : -S.listen.tiltSide;
        S.tilt.target = S.listen.tiltSide * U(0.05, 0.11);
      } else if (!uv.speaking && !av.speaking && t > S.tilt.next) {
        S.tilt.next = t + U(5, 12);
        S.tilt.target = chance(0.5) ? N() * 0.035 : S.tilt.target * 0.4;
      }
      // salient motion: his hand moving fast catches the eye — a brief glance at it (not while a reflex/command owns the gaze)
      if (u.present && !S.focus && !S.cmdLook && !S.around && t - (S.handGlanceAt ?? -9) > 4) {
        for (const h of u.hands || []) {
          const sp = Math.hypot(+h.vx || 0, +h.vy || 0);
          if (h.main !== false && sp > 1.1 && (h.age === undefined || h.age < 0.3) && chance(0.4)) {
            S.handGlanceAt = t;
            const W = [Math.atan((+h.x || 0) * TAN_HFOV2) * 0.85, Math.atan((+h.y || 0) * TAN_VFOV2) * 0.85];
            S.focus = {kind: 'hand', W, until: t + U(0.5, 0.9), gain: 0.35};
            break;
          } else if (sp > 1.1) S.handGlanceAt = t - 3.6;   // checked; try again in 0.4 s
        }
      }
      // a listener's occasional glance away; in a silent stare, break mutual gaze now and then
      if (u.present && !S.avert && !S.cmdLook && !S.focus && !av.speaking) {
        S.mgNext = S.mgNext ?? t + U(3, 6);
        if (t > S.mgNext) {
          S.mgNext = t + (uv.speaking ? U(5, 11) : u.attending ? U(3.5, 7) : U(2, 4));
          if (chance(uv.speaking ? 0.35 : (u.attending ? 0.5 : 0.8)))
            S.avert = {W: [cfg.gazeFollow * u.ax + pick([-1, 1]) * U(0.06, 0.17), cfg.gazeFollow * u.ay + U(-0.15, 0.03)], until: t + U(0.35, 0.8), gain: 0.3};
        }
      }
      // -- boredom
      if (!av.speaking && !uv.speaking) S.idleT += dt;
      const bTarget = clamp((S.idleT - cfg.boredAfter) / 22, 0, 1) * (u.attending ? 0.7 : 1) + 0.5 * u.boredom * clamp(S.idleT / 6, 0, 1);
      S.bored = lp(S.bored, clamp(bTarget, 0, 1), dt, bTarget > S.bored ? 5 : 1.2);
      const fid = S.fidget;
      fid.next -= dt * (0.3 + 0.9 * S.bored) * K.fidget;
      if (fid.next <= 0) {
        fid.next = logn(lerp(10, 5, S.bored), 0.35);
        // no idle fidget over a whole-body / object job (live v7: a 'sigh' fidget mid-offer left the ball
        // held at the belly; v3: a head_tilt mid-give)
        if (!av.speaking && !uv.speaking && !S.cmdLook && !(engineOn && engine && ((engine.kit && engine.kit.busy && engine.kit.busy()) || (engine.wholeBody && engine.wholeBody())))) fidget();
      }
      // idle ear twitches, nose twitches
      for (const s of ['l', 'r']) {
        S.earTw[s] -= dt;
        if (S.earTw[s] <= 0) { S.earTw[s] = logn(lerp(9, 5, S.bored), 0.5) / K.fidget; startGesture('ear_flick', {side: s, src: 'life', amp: U(0.5, 1)}); }
      }
      S.noseTw -= dt;
      if (S.noseTw <= 0) { S.noseTw = logn(14, 0.5) / K.fidget; if (chance(0.6)) startGesture('nose_wiggle', {src: 'life', dur: U(0.5, 1.0), amp: U(0.3, 0.6)}); }
      // weight shifts
      if (t > S.shift.next) {
        S.shift.next = t + logn(lerp(12, 6, S.bored), 0.35);
        S.shift.from = {lean: S.shift.lean, yaw: S.shift.yaw, t0: t, dur: U(1.0, 1.8)};
        S.shift.to = {lean: N() * 0.14 * (1 + S.bored) * K.sway, yaw: N() * 0.09 * (1 + S.bored) * K.sway};
      }
    }

    const FIDGETS = [
      ['look_around', 3], ['weight_shift', 2], ['sigh', 1.4], ['ear_wiggle', 1], ['tail_swish', 1.2], ['paw_fidget', 1.6],
      ['head_tilt', 1.5], ['sniff', 1], ['stretch', 0.5], ['yawn', 0.45], ['chin_rest', 0.9], ['shoulder_roll', 0.8],
    ];
    function fidget() {
      const fid = S.fidget, b = S.bored;
      const ok = FIDGETS.filter(([n]) => !fid.last.includes(n) &&
        !((n === 'stretch' || n === 'yawn') && (b < 0.55 || S.t - (fid.bigAt || -99) < 35)) &&
        !(n === 'chin_rest' && b < 0.35));
      let tot = 0; for (const [, w] of ok) tot += w;
      let r = rnd() * tot, name = ok.length ? ok[0][0] : 'look_around';
      for (const [n, w] of ok) { r -= w; if (r <= 0) { name = n; break; } }
      fid.last.push(name); if (fid.last.length > 3) fid.last.shift();
      fid.lastName = name;
      if (name === 'stretch' || name === 'yawn') fid.bigAt = S.t;
      S.log.push({t: +S.t.toFixed(2), fidget: name, bored: +b.toFixed(2)});
      if (S.log.length > 400) S.log.splice(0, 200);
      switch (name) {
        case 'look_around': lookAround(2 + Math.floor(rnd() * 3), 'bored'); break;
        case 'weight_shift': S.shift.next = S.t; break;
        case 'sigh': S.breath.sigh = S.t; startGesture('sigh', {src: 'fidget'}); break;
        default: startGesture(name, {src: 'fidget'});
      }
    }
    function lookAround(n, kind) {
      const seq = []; let side = pick([-1, 1]);
      for (let i = 0; i < n; i++) { seq.push({W: [side * U(0.18, 0.42), U(-0.14, 0.22)], dur: U(0.55, 1.3)}); side = chance(0.75) ? -side : side; }
      if (S.user.present) seq.push({W: [cfg.gazeFollow * S.user.ax, cfg.gazeFollow * S.user.ay], dur: 0.3});
      S.around = {seq, i: 0, t0: S.t, gain: 0.75, kind};
    }

    // ---------------------------------------------------------------- gestures (keyframed, weighted)
    // each: (u, g) → {set:{param:v}, add:{param:v}}; the executor adds attack/release weights (min-jerk)
    const HP = (s, x, y, z, open, turn) => {
      const o = {}; o['hand_' + s + '_x'] = x; o['hand_' + s + '_y'] = y; o['hand_' + s + '_z'] = z;
      if (open !== undefined) o['hand_' + s + '_open'] = open; if (turn !== undefined) o['hand_' + s + '_turn'] = turn; return o;
    };
    const other = s => s === 'l' ? 'r' : 'l';
    // ---------------------------------------------------------------- face touches for a body-file character
    // The muse's head (radii 0.70 0.64 0.62) is wider than its arm is long (0.57) and sits in front of its shoulders, so
    // the contract's face numbers (COVER_EYES, CHIN, a mouth touch) bury its paw inside the head. makeFace(body) turns
    // a face point of the body file (body.face, head space, x outward on the paw's side) and a head pose into the hand
    // target that puts the palm there, through the renderer's own hand mapping (body.hand_map); the shader rests the
    // palm on the surface in front of it. S.face = null for the old characters: their gestures keep their numbers.
    // head space -> upper-body / world axes, as the shaders: F = rotY(yaw) rotX(pitch) rotZ(-roll)
    const rotHead = (h, v) => {
      const r = -(h.roll || 0), p = h.pitch || 0, y = h.yaw || 0;
      let [x, yy, z] = [Math.cos(r) * v[0] - Math.sin(r) * v[1], Math.sin(r) * v[0] + Math.cos(r) * v[1], v[2]];
      [yy, z] = [Math.cos(p) * yy - Math.sin(p) * z, Math.sin(p) * yy + Math.cos(p) * z];
      return [Math.cos(y) * x + Math.sin(y) * z, yy, -Math.sin(y) * x + Math.cos(y) * z];
    };
    // ---------------------------------------------------------------- a character's own chin
    // The contract's CHIN numbers (-0.35, 0.75, 0.35) are a pose the fox / bear / rabbit / hippo shaders turn into
    // "knuckles under my chin" themselves. The others need the target worked out from their own geometry:
    //  - a SHAPE (orb, cube, drop, star, pyramid, blob) is all face: those numbers put its mitten on an eye. Its chin is
    //    the spot just under its own lower lip (the shader's mouth frame LIP_O / LIP_W / LIP_HL, head space), reached
    //    through engine/shape_arms.glsl's mapping: wrist = shoulder(head pose) + SA_REST + (xin*hx | SA_LEN*hx,
    //    SA_LEN*hy, SA_LEN*hz), xin = (SA_SHO.x + SA_REST.x)/0.45; the shader then rests the hand on the host's surface.
    //    A shape's arm is short: the spot slides along the under-lip line toward the paw's own mouth corner until the
    //    arm reaches it; a shape whose arm cannot reach even there (the pyramid: its mouth is a full arm length in
    //    front of its shoulders) has no chin pose (makeChin -> null: the thinking's main pose runs without the paw).
    //  - an ARM character with a linear hand mapping and no chin pose in its shader (the monkey: the numbers put its
    //    palm over an eye): the spot under its muzzle, through that mapping (tools/think_geom.cjs linearArm).
    // The proportions are shared by every character of a kind; the numbers are each character's own shader constants.
    // <think-geom> generated by tools/think_geom.cjs --write from renderer/animals/*.glsl — do not edit by hand
    const CHIN_GEOM = {
      blob: {kind: "shape", lip: [0, -0.25, 0.73], lipW: 0.322, lipHL: 0.0978, sho: [0.7, -0.22, 0.1], rest: [0.16, -0.3, 0.25], len: 0.85, armR: 0.09, hand: 0.18, pivot: [0, -0.8, 0], yaw0: 0},
      cube: {kind: "shape", lip: [0, -0.272, 0.7], lipW: 0.318, lipHL: 0.066, sho: [0.66, -0.15, 0.15], rest: [0.12, -0.36, 0.22], len: 0.85, armR: 0.09, hand: 0.18, pivot: [0, -0.7, 0], yaw0: -0.34},
      drop: {kind: "shape", lip: [0, -0.257, 0.6079], lipW: 0.262, lipHL: 0.0699, sho: [0.57, -0.28, 0.05], rest: [0.14, -0.34, 0.22], len: 0.78, armR: 0.082, hand: 0.16, pivot: [0, -0.575, 0], yaw0: 0},
      monkey: {kind: "arm", wrist0: [0.62, -1.32, 0.54], len: 1.3616, zk: 0.85, zup: 0.1, neck: [0, -0.6, -0.14], byaw: 0.2, yaw0: -0.23, muzC: [0, -0.22, 0.5], muzR: [0.34, 0.3, 0.48]},
      orb: {kind: "shape", lip: [0, -0.316, 0.6246], lipW: 0.3, lipHL: 0.0733, sho: [0.6, -0.3, 0.05], rest: [0.14, -0.36, 0.22], len: 0.8, armR: 0.086, hand: 0.17, pivot: [0, -0.7, 0], yaw0: 0},
      pyramid: {kind: "shape", lip: [0, -0.155, 0.808], lipW: 0.352, lipHL: 0.0939, sho: [0.78, -0.25, 0.05], rest: [0.14, -0.3, 0.25], len: 0.85, armR: 0.09, hand: 0.18, pivot: [0, -0.72, 0], yaw0: -0.33},
      star: {kind: "shape", lip: [0, -0.278, 0.495], lipW: 0.292, lipHL: 0.0692, sho: [0.52, -0.17, 0.1], rest: [0.2, -0.38, 0.25], len: 0.8, armR: 0.084, hand: 0.17, pivot: [0, 0, 0], yaw0: 0},
    };
    // </think-geom>
    const CHIN_DX = 0.35, CHIN_DX_MAX = 1.0;   // shape: the wrist sits this much of the mouth's half width toward the paw's own side (.. up to the mouth corner)
    const CHIN_DROP = 0.55;   // ... this many palm lengths under the lower lip's edge (the mitten's fingers end at the lip)
    const CHIN_OUT = 1.0;     // ... and this many arm radii in front of the mouth's surface point
    const CHIN_REACH = 1.06, CHIN_REACH_MAX = 1.08;   // of the arm's length. Up to ~1.06 the shader's own soft reach cap still lands the mitten under the mouth (seen on renders: orb, cube, drop); further the spot slides toward the mouth corner; still past the max at the corner = no chin pose (the pyramid, 1.12: only a knob of the mitten showed)
    const CHIN_AX = 0.35, CHIN_ADROP = -0.03, CHIN_AZ = 0.25;   // arm kind: the wrist this much of the muzzle's half width to the paw's side, level with the muzzle's underside (minus = a touch above: its hand hangs from the wrist, so the fingers lie under the chin), this much of its depth forward of its centre
    function makeChin(G) {
      if (!G) return null;
      const rg = (k, v) => { const r = (S.rng && S.rng[k]) || RANGE[k]; return r ? clamp(v, r[0], r[1]) : v; };
      const out = (side, h) => [rg('hand_' + side + '_x', h[0]), rg('hand_' + side + '_y', h[1]), rg('hand_' + side + '_z', h[2])];
      if (G.kind === 'arm') {
        const cb = Math.cos(G.byaw), sb = Math.sin(G.byaw), nk = G.neck;
        return (side, head = {}, d = {}) => {
          const sg = side === 'r' ? 1 : -1;
          const q = [sg * (CHIN_AX * G.muzR[0] + (d.dx || 0)), G.muzC[1] - G.muzR[1] - CHIN_ADROP * G.len + (d.dy || 0), G.muzC[2] + CHIN_AZ * G.muzR[2] + (d.dz || 0)];
          const o = rotHead({yaw: G.yaw0 + (head.yaw || 0), pitch: head.pitch || 0, roll: head.roll || 0}, [q[0] - nk[0], q[1] - nk[1], q[2] - nk[2]]);
          const W = [nk[0] + o[0], nk[1] + o[1], nk[2] + o[2]];
          const T = [sg * (cb * W[0] - sb * W[2]), W[1], sb * W[0] + cb * W[2]];   // world -> this side's (mirrored) torso frame
          const hy = (T[1] - G.wrist0[1]) / G.len;
          return out(side, [(T[0] - G.wrist0[0]) / G.len, hy, (T[2] - G.wrist0[2] - G.zup * smooth01(hy)) / (G.zk * G.len)]);
        };
      }
      const xin = (G.sho[0] + G.rest[0]) / 0.45, pv = G.pivot;
      const spot = dx => [dx * G.lipW, G.lip[1] - G.lipHL - CHIN_DROP * G.hand, G.lip[2] + CHIN_OUT * G.armR];
      const reach = dx => { const q = spot(dx); return Math.hypot(q[0] - G.sho[0], q[1] - G.sho[1], q[2] - G.sho[2]) / G.len; };
      let DX = CHIN_DX;
      while (DX < CHIN_DX_MAX && reach(DX) > CHIN_REACH) DX = Math.min(CHIN_DX_MAX, DX + 0.05);
      if (reach(DX) > CHIN_REACH_MAX) return null;
      // [hx, hy, hz] for side 'l' (screen-left) / 'r'; head = {yaw, pitch, roll} the gesture holds; d = {dx (outward), dy, dz}
      return (side, head = {}, d = {}) => {
        const sg = side === 'r' ? 1 : -1, q0 = spot(DX);
        const q = [sg * (q0[0] + (d.dx || 0)), q0[1] + (d.dy || 0), q0[2] + (d.dz || 0)];
        const hd = {yaw: G.yaw0 + (head.yaw || 0), pitch: head.pitch || 0, roll: head.roll || 0};
        const C = rotHead(hd, [q[0] - pv[0], q[1] - pv[1], q[2] - pv[2]]), Sw = rotHead(hd, [sg * G.sho[0] - pv[0], G.sho[1] - pv[1], G.sho[2] - pv[2]]);
        const ox = sg * (C[0] - Sw[0]) - G.rest[0], oy = C[1] - Sw[1] - G.rest[1], oz = C[2] - Sw[2] - G.rest[2];
        return out(side, [ox / (ox >= 0 ? G.len : xin), oy / G.len, oz / G.len]);
      };
    }
    function makeFace(body) {
      const f = body && body.face, hm = body && body.hand_map;
      if (!f || !hm || !Array.isArray(body.head_scene) || !Array.isArray(hm.rest_wrist_scene)) return null;
      const nk = f.neck_pivot_scene || (body.joints_scene && body.joints_scene.neck) || [0, 1.12, 0];
      const hc0 = body.head_scene, rw = hm.rest_wrist_scene, L = +hm.L_scene || 0.57, up = +hm.palm_above_wrist_scene || 0.09;
      const gxo = +hm.gain_x_out || 0.95, gxi = +hm.gain_x_in || 1.05, gy = +hm.gain_y || 1.4, gz = +hm.gain_z || 0.8;
      const rotF = rotHead;   // head space -> upper-body space about the neck
      const rg = (k, v) => { const r = (S.rng && S.rng[k]) || RANGE[k]; return r ? clamp(v, r[0], r[1]) : v; };
      return {
        has: n => Array.isArray(f[n]),
        // [hx, hy, hz] for side 'l' (screen-left) / 'r'; head = {yaw, pitch, roll} the gesture holds; d = {dx (outward), dy, dz}
        target(name, side, head = {}, d = {}) {
          const q = f[name] || f.mouth, sg = side === 'r' ? 1 : -1;
          const qv = [sg * (q[0] + (d.dx || 0)), q[1] + (d.dy || 0), q[2] + (d.dz || 0)];
          const c = rotF(head, [hc0[0] - nk[0], hc0[1] - nk[1], hc0[2] - nk[2]]), o = rotF(head, qv);
          const W = [nk[0] + c[0] + o[0], nk[1] + c[1] + o[1] - up, nk[2] + c[2] + o[2]];
          const ox = sg * W[0] - rw[0];
          return [rg('hand_' + side + '_x', ox / ((ox > 0 ? gxo : gxi) * L)), rg('hand_' + side + '_y', (W[1] - rw[1]) / (gy * L)),
            rg('hand_' + side + '_z', (W[2] - rw[2]) / (gz * L))];
        },
      };
    }
    const FP = (s, h, open, turn) => HP(s, h[0], h[1], h[2], open, turn);
    const pulse = (u, a, b, c, d) => u < a ? 0 : u < b ? mj((u - a) / (b - a)) : u < c ? 1 : u < d ? 1 - mj((u - c) / (d - c)) : 0;
    const GEST = {
      wave: {dur: g => 0.5 + (g.cycles || 4) * 0.4 + 0.2, attack: 0.38, release: 0.5, fn(u, g) {
        const s = g.side, e = smooth01((u - 0.25) / 0.3), w = Math.sin(2 * Math.PI * 2.5 * (u - 0.25));
        const set = HP(s, 0.20 + 0.12 * w * e, 1.0 + 0.03 * Math.sin(2 * Math.PI * 5 * u) * e, 0.20, 1, 1 - 0.3 * Math.abs(w) * e);
        set['shoulder_' + s] = 0.12; set['point_' + s] = 0; set['thumb_' + s] = 0.3;
        return {set, add: {head_roll: (s === 'r' ? 1 : -1) * 0.05, smile: 0.25, brow_l: 0.25, brow_r: 0.25, ear_l: 0.2, ear_r: 0.2}};
      }},
      raise_paw: {dur: 2.2, attack: 0.4, release: 0.45, fn(u, g) { const s = g.side; return {set: Object.assign(HP(s, 0.25, 0.95, 0.15, 1, 1), {['shoulder_' + s]: 0.1})}; }},
      raise_both: {dur: 2.2, attack: 0.45, release: 0.5, fn(u) {
        const k = 0.03 * Math.sin(2 * Math.PI * 3 * u);
        return {set: Object.assign(HP('l', 0.3 + k, 1.05, 0.15, 1, 1), HP('r', 0.3 - k, 1.05, 0.15, 1, 1)), add: {smile: 0.5, eye_wide: 0.2, body_bounce: 0.12 * Math.sin(2 * Math.PI * 2 * u)}};
      }},
      nod: {dur: g => 0.15 + 0.48 * (g.cycles || 2), attack: 0.08, release: 0.2, fn(u, g) {
        // g.tune: the character's body-file tuning (muse.body.json gesture_tuning.nod: a bigger head pitch per dip and
        // less decay, so a full-body framing still reads the nod); none for the old characters (identical numbers)
        const T = g.tune || {}, A = 0.11 * (g.amp ?? 1) * (T.amp ?? 1), c = u / 0.48, k = Math.pow(T.decay ?? 0.7, Math.floor(c));
        const v = c < (g.cycles || 2) ? Math.sin(Math.PI * (c % 1)) : 0;
        return {add: {head_pitch: A * k * v * v, head_y: -0.01 * k * v}};
      }},
      shake_head: {dur: g => 0.2 + 0.52 * (g.cycles || 3), attack: 0.12, release: 0.25, fn(u, g) {
        const c = u / 0.52, k = Math.pow(0.8, c), v = Math.sin(2 * Math.PI * c);
        return {add: {head_yaw: 0.17 * k * v, head_roll: -0.03 * k * v, brow_knit: 0.15}};
      }},
      wink: {dur: 0.75, attack: 0.05, release: 0.15, fn(u, g) {
        const s = g.side, v = pulse(u, 0, 0.1, 0.38, 0.55);
        const set = {['blink_' + s]: v}; const add = {smile: 0.3 * v, head_roll: (s === 'r' ? 1 : -1) * 0.05 * v, squint: 0.2 * v};
        add['brow_' + s] = -0.25 * v; return {set, add};
      }},
      ear_wiggle: {dur: g => 0.3 + 0.45 * (g.cycles || 3), attack: 0.1, release: 0.2, fn(u) {
        const v = Math.sin(2 * Math.PI * 2.2 * u);
        return {add: {ear_l: 0.4 * v, ear_r: -0.4 * v}, set: {ear_flick_l: 0.5 * Math.max(0, v), ear_flick_r: 0.5 * Math.max(0, -v)}};
      }},
      ear_flick: {dur: 0.5, attack: 0.02, release: 0.08, fn(u, g) {
        const a = g.amp ?? 1, v = pulse(u, 0, 0.06, 0.1, 0.24) + 0.6 * pulse(u, 0.24, 0.3, 0.33, 0.45) * (g.double ? 1 : 0);
        return {add: {['ear_flick_' + g.side]: a * v}};
      }},
      nose_wiggle: {dur: g => g.dur || 1.3, attack: 0.1, release: 0.15, fn(u, g) {
        const a = g.amp ?? 0.8; return {set: {nose_wiggle: a * Math.sin(2 * Math.PI * 7 * u)}, add: {nostril: 0.25 * a}};
      }},
      sniff: {dur: 1.4, attack: 0.15, release: 0.3, fn(u) {
        const v = Math.max(0, Math.sin(2 * Math.PI * 4 * u));
        return {set: {nose_wiggle: 0.5 * v}, add: {nostril: 0.5 * v, head_pitch: -0.04, head_z: 0.01, eye_wide: 0.1}};
      }},
      tail_wag: {dur: g => g.dur || 2.6, attack: 0.25, release: 0.4, fn(u) {
        return {set: {tail_wag: 0.9 * Math.sin(2 * Math.PI * 2.2 * u)}, add: {tail_raise: 0.3, smile: 0.15}};
      }},
      tail_swish: {dur: 3.2, attack: 0.5, release: 0.6, fn(u) { return {set: {tail_wag: 0.55 * Math.sin(2 * Math.PI * 0.55 * u)}, add: {tail_curl: 0.2}}; }},
      clap: {dur: g => 0.5 + 0.42 * (g.cycles || 4), attack: 0.35, release: 0.45, fn(u) {
        const k = u < 0.35 ? 0 : 0.5 + 0.5 * Math.cos(2 * Math.PI * 2.4 * (u - 0.35));
        const x = -0.45 + 0.2 * k;
        return {set: Object.assign(HP('l', x, 0.45, 0.45, 1, 0), HP('r', x, 0.45, 0.45, 1, 0)), add: {smile: 0.45, squint: 0.2, body_bounce: 0.05 * (1 - k)}};
      }},
      thumbs_up: {dur: g => g.dur || 2.2, attack: 0.38, release: 0.45, fn(u, g) {
        const s = g.side; const set = HP(s, 0.10, 0.45, 0.45, 0, 0); set['thumb_' + s] = 1;
        return {set, add: {smile: 0.35, head_pitch: 0.04 * pulse(u, 0.35, 0.5, 0.6, 0.8), brow_l: 0.15, brow_r: 0.15}};
      }},
      shrug: {dur: 1.7, attack: 0.25, release: 0.45, fn(u) {
        const p = pulse(u, 0, 0.3, 1.0, 1.4);
        return {set: Object.assign({shoulder_l: p, shoulder_r: p}, HP('l', 0.35, 0.30, 0.20, 1, 1), HP('r', 0.35, 0.30, 0.20, 1, 1)),
          add: {head_roll: 0.08 * p, brow_l: 0.5 * p, brow_r: 0.4 * p, head_pitch: -0.03 * p, mouth_open: 0.05 * p}};
      }},
      bow: {dur: 2.2, attack: 0.45, release: 0.6, fn(u) {
        const p = pulse(u, 0, 0.6, 1.2, 1.9);
        return {set: Object.assign({body_pitch: 0.65 * p}, HP('l', -0.45, 0.35, 0.25, 1, 0)), add: {head_pitch: 0.32 * p, gaze_y: -0.7 * p, smile: 0.2}};
      }},
      point: {dur: g => g.dur || 2.4, attack: 0.38, release: 0.45, fn(u, g) {
        const s = g.side, side = g.dir === 'side' || g.dir === 'left' || g.dir === 'right' || g.dir === 'screen_left' || g.dir === 'screen_right';
        const set = side ? HP(s, 0.60, 0.60, 0.30, 0, 0) : HP(s, -0.10, 0.55, 0.85, 0, 0);
        set['point_' + s] = 1;
        return {set};
      }},
      stick_tongue_out: {dur: g => g.dur || 1.9, attack: 0.25, release: 0.3, fn(u) {
        return {set: {tongue_out: 1, mouth_open: 0.32}, add: {squint: 0.35, head_roll: 0.06, smile: 0.15}};
      }},
      jump: {dur: 1.15, attack: 0.04, release: 0.15, fn(u) {
        let b;
        if (u < 0.2) b = -0.4 * mj(u / 0.2); else if (u < 0.45) b = lerp(-0.4, 1, mj((u - 0.2) / 0.25));
        else if (u < 0.7) b = lerp(1, -0.3, mj((u - 0.45) / 0.25)); else b = lerp(-0.3, 0, mj((u - 0.7) / 0.4));
        const up = clamp(b, 0, 1);
        return {set: {body_bounce: b}, add: {hand_l_y: 0.3 * up, hand_r_y: 0.3 * up, hand_l_x: 0.1 * up, hand_r_x: 0.1 * up,
          ear_l: -0.4 * clamp(-b, 0, 1) + 0.2 * up, ear_r: -0.4 * clamp(-b, 0, 1) + 0.2 * up, eye_wide: 0.3 * up, smile: 0.3, tail_raise: 0.4 * up}};
      }},
      turn: {dur: g => g.dur || 2.6, attack: 0.55, release: 0.7, fn(u, g) {
        const d = g.dir || 1;   // +1 = toward screen-right
        return {set: {body_yaw: 0.85 * d}, add: {head_yaw: 0.25 * d, head_roll: 0.03 * d}};
      }},
      turn_around: {dur: 3.0, attack: 0.8, release: 0.9, fn(u, g) {
        const d = g.dir || 1;
        return {set: {body_yaw: 1.0 * d}, add: {head_yaw: 0.45 * d * pulse(u, 0.4, 1.0, 2.0, 2.6), tail_wag: 0.3 * Math.sin(2 * Math.PI * 1.5 * u)}};
      }},
      lean_in: {dur: g => g.dur || 2.5, attack: 0.5, release: 0.6, fn() { return {set: {body_pitch: 0.7}, add: {head_z: 0.035, eye_wide: 0.1, brow_l: 0.15, brow_r: 0.15}}; }},
      lean_back: {dur: g => g.dur || 2.5, attack: 0.5, release: 0.6, fn() { return {set: {body_pitch: -0.6}, add: {head_z: -0.03, head_pitch: -0.04}}; }},
      lean_side: {dur: g => g.dur || 2.5, attack: 0.5, release: 0.6, fn(u, g) { return {set: {body_lean: 0.7 * (g.dir || 1)}, add: {head_roll: 0.06 * (g.dir || 1)}}; }},
      tilt_head: {dur: g => g.dur || 2.4, attack: 0.35, release: 0.5, fn(u, g) { return {add: {head_roll: 0.17 * (g.dir || 1), brow_l: 0.12, brow_r: 0.12}}; }},
      head_tilt: {dur: g => U(2.0, 3.5), attack: 0.6, release: 0.7, fn(u, g) { g.dir = g.dir || pick([-1, 1]); return {add: {head_roll: 0.11 * g.dir}}; }},
      cover_eyes: {dur: g => g.dur || 2.6, attack: 0.4, release: 0.35, fn(u, g) {
        const peek = g.peekaboo ? pulse(u, (g.dur || 2.6) - 0.7, (g.dur || 2.6) - 0.5, 9, 10) : 0;
        if (S.face) {   // muse: both palms over its eyes, the head bowed into them a little; peekaboo opens them aside
          const hd = {pitch: 0.12 * (1 - peek)}, d = {dx: 0.28 * peek, dy: -0.10 * peek};
          const set = Object.assign(FP('l', S.face.target('eye', 'l', hd, d), 1, 1), FP('r', S.face.target('eye', 'r', hd, d), 1, 1));
          set.head_yaw = 0; set.head_roll = 0; set.head_pitch = hd.pitch;
          return {set, add: {smile: 0.2 + 0.5 * peek, eye_wide: 0.4 * peek}};
        }
        const x = -0.35 + 0.35 * peek;
        return {set: Object.assign(HP('l', x, 1.0, 0.45, 1, 0), HP('r', x, 1.0, 0.45, 1, 0)), add: {smile: 0.2 + 0.5 * peek, eye_wide: 0.4 * peek}};
      }},
      hand_on_heart: {dur: g => g.dur || 2.4, attack: 0.4, release: 0.45, fn(u, g) { return {set: HP(g.side, -0.45, 0.35, 0.25, 1, 0), add: {smile: 0.25, head_roll: 0.05}}; }},
      think: {dur: g => g.dur || 3.0, attack: 0.45, release: 0.5, fn(u, g) {
        if (S.face) return {set: Object.assign(FP(g.side, S.face.target('chin', g.side, {roll: (g.side === 'r' ? 1 : -1) * 0.07}), 0.3, 0), {head_yaw: 0, head_roll: (g.side === 'r' ? 1 : -1) * 0.07}), add: {brow_knit: 0.3, brow_l: 0.15}};
        // a body with a chin of its own worked out (a shape, the monkey: makeChin) uses it; else the contract's CHIN
        return {set: S.chin ? FP(g.side, S.chin(g.side, {roll: 0.07}), 0.3, 0) : HP(g.side, -0.35, 0.75, 0.35, 0.3, 0), add: {brow_knit: 0.3, head_roll: 0.07, brow_l: 0.15}};
      }},
      chin_rest: {dur: () => U(3.5, 6), attack: 0.6, release: 0.7, fn(u, g) {
        g.sd = g.sd || pick(['l', 'r']);
        if (S.face) return {set: FP(g.sd, S.face.target('chin', g.sd, {roll: (g.sd === 'l' ? -1 : 1) * 0.07, pitch: 0.03}), 0.3, 0), add: {head_roll: (g.sd === 'l' ? -1 : 1) * 0.07, head_pitch: 0.03, blink_l: 0.12, blink_r: 0.12}};
        return {set: S.chin ? FP(g.sd, S.chin(g.sd, {roll: (g.sd === 'l' ? -1 : 1) * 0.07, pitch: 0.03}), 0.3, 0) : HP(g.sd, -0.35, 0.75, 0.35, 0.3, 0), add: {head_roll: (g.sd === 'l' ? -1 : 1) * 0.07, head_pitch: 0.03, blink_l: 0.12, blink_r: 0.12}};
      }},
      paw_fidget: {dur: () => U(2.0, 3.2), attack: 0.5, release: 0.6, fn(u) {
        const k = 0.04 * Math.sin(2 * Math.PI * 1.6 * u);
        return {set: Object.assign(HP('l', -0.32 + k, 0.28, 0.32, 0.4, 0.1), HP('r', -0.32 - k, 0.28, 0.32, 0.4, 0.1)), add: {gaze_y: -0.35, head_pitch: 0.06}};
      }},
      shoulder_roll: {dur: 1.4, attack: 0.2, release: 0.4, fn(u) {
        return {set: {shoulder_l: pulse(u, 0, 0.35, 0.5, 0.9), shoulder_r: pulse(u, 0.2, 0.55, 0.7, 1.1)}, add: {head_roll: 0.03 * Math.sin(2 * Math.PI * 0.8 * u)}};
      }},
      sigh: {dur: 2.2, attack: 0.3, release: 0.6, fn(u) {
        const up = pulse(u, 0, 0.7, 0.9, 2.0);
        return {set: {shoulder_l: 0.45 * up, shoulder_r: 0.45 * up}, add: {head_pitch: -0.05 * up + 0.07 * pulse(u, 0.9, 1.4, 1.7, 2.2), blink_l: 0.2, blink_r: 0.2, ear_l: -0.2, ear_r: -0.2}};
      }},
      stretch: {dur: 3.0, attack: 0.7, release: 0.8, fn(u) {
        const p = pulse(u, 0, 0.9, 1.8, 2.8);
        return {set: Object.assign(HP('l', 0.25, 1.15, 0.0, 1, 0.5), HP('r', 0.25, 1.15, 0.0, 1, 0.5), {shoulder_l: 0.5 * p, shoulder_r: 0.5 * p, body_pitch: -0.2 * p}),
          add: {head_pitch: -0.1 * p, squint: 0.4 * p, blink_l: 0.35 * p, blink_r: 0.35 * p}};
      }},
      yawn: {dur: 2.6, attack: 0.3, release: 0.6, fn(u) {
        const m = pulse(u, 0.1, 0.9, 1.5, 2.3);
        return {set: {mouth_open: 0.78 * m}, add: {squint: 0.6 * m, blink_l: 0.55 * m, blink_r: 0.55 * m, head_pitch: -0.11 * m, ear_l: -0.25 * m, ear_r: -0.25 * m, brow_l: 0.25 * m, brow_r: 0.25 * m}};
      }},
      dance: {dur: g => g.dur || 4.5, attack: 0.4, release: 0.6, fn(u) {
        const b = Math.sin(2 * Math.PI * 2 * u), s = Math.sin(2 * Math.PI * 1 * u);
        return {set: {body_bounce: 0.3 * Math.abs(b), body_lean: 0.35 * s, tail_wag: 0.8 * Math.sin(2 * Math.PI * 2 * u)},
          add: {head_roll: 0.08 * s, hand_l_y: 0.35 + 0.1 * b, hand_r_y: 0.35 - 0.1 * b, hand_l_x: 0.15, hand_r_x: 0.15, smile: 0.5}};
      }},
      laugh: {dur: 1.9, attack: 0.15, release: 0.45, fn(u) {
        const v = 0.5 + 0.5 * Math.sin(2 * Math.PI * 5 * u);
        return {set: {mouth_open: 0.2 + 0.18 * v}, add: {smile: 0.7, squint: 0.65, head_pitch: -0.06 + 0.035 * v, body_bounce: 0.05 * v, shoulder_l: 0.2 * v, shoulder_r: 0.2 * v, ear_l: 0.2, ear_r: 0.2}};
      }},
      high_five: {dur: g => g.dur || 2.2, attack: 0.38, release: 0.45, fn(u, g) { return {set: HP(g.side, 0.15, 0.85, 0.6, 1, 1), add: {smile: 0.4}}; }},
      blow_kiss: {dur: 1.8, attack: 0.3, release: 0.4, fn(u, g) {
        const s = g.side, out = mj(clamp((u - 0.7) / 0.45, 0, 1));
        if (S.face) {   // muse: the paw's fingertips on its lips first (head steady), then thrown out toward him
          const h = S.face.target('mouth', s, {}, {dx: -0.03});
          const set = HP(s, lerp(h[0], 0.12, out), lerp(h[1], 0.62, out), lerp(h[2], 0.8, out), lerp(0.5, 1, out), 1);
          set.head_yaw = 0; set.head_roll = 0;
          return {set, add: {smile: 0.35 + 0.2 * out, squint: 0.3 + 0.2 * (1 - out), head_pitch: -0.03 * out}};
        }
        const set = HP(s, lerp(-0.28, 0.12, out), lerp(0.72, 0.62, out), lerp(0.38, 0.8, out), lerp(0.3, 1, out), lerp(0, 1, out));
        return {set, add: {smile: 0.35 + 0.2 * out, squint: 0.3, head_roll: (s === 'r' ? 1 : -1) * 0.06, head_pitch: -0.03 * out}};
      }},
      brow_wiggle: {dur: 1.4, attack: 0.1, release: 0.2, fn(u) { const v = Math.sin(2 * Math.PI * 3 * u); return {add: {brow_l: 0.55 * v, brow_r: 0.55 * v, smile: 0.25, squint: 0.15}}; }},
      eye_roll: {dur: 1.3, attack: 0.1, release: 0.3, fn(u) {
        const a = Math.PI * clamp(u / 0.9, 0, 1);   // a half circle: side → up → other side
        return {add: {gaze_x: -0.9 * Math.cos(a), gaze_y: 0.9 * Math.sin(a), head_pitch: -0.05 * Math.sin(a), head_roll: 0.04, blink_l: 0.15, blink_r: 0.15}};
      }},
      flap_arms: {dur: g => 0.5 + 0.33 * (g.cycles || 5), attack: 0.3, release: 0.4, fn(u) {
        const v = Math.sin(2 * Math.PI * 3 * u);
        return {set: Object.assign(HP('l', 0.55, 0.35 + 0.2 * v, 0.1, 1, 0), HP('r', 0.55, 0.35 + 0.2 * v, 0.1, 1, 0)), add: {smile: 0.3, shoulder_l: 0.2 * Math.max(0, v), shoulder_r: 0.2 * Math.max(0, v), body_bounce: 0.05 * v}};
      }},
      lick_lips: {dur: 1.2, attack: 0.15, release: 0.25, fn(u) { const v = pulse(u, 0.1, 0.3, 0.7, 0.95); return {set: {tongue_out: 0.45 * v, mouth_open: 0.15 * v}, add: {smile: 0.2, squint: 0.2 * v, head_roll: 0.03}}; }},
      rub_belly: {dur: g => g.dur || 2.6, attack: 0.4, release: 0.45, fn(u, g) {
        const a = 2 * Math.PI * 1.4 * u;
        return {set: HP(g.side, -0.35 + 0.06 * Math.cos(a), 0.05 + 0.05 * Math.sin(a), 0.35, 1, 0), add: {smile: 0.45, squint: 0.3}};
      }},
      scratch_head: {dur: g => g.dur || 2.2, attack: 0.4, release: 0.45, fn(u, g) {
        const v = Math.sin(2 * Math.PI * 4 * u);
        if (S.face) {   // muse: the paw on the upper side of its head (all its short arm reaches), the head tilted into it, scratching
          const sg = g.side === 'r' ? 1 : -1, hd = {roll: 0.24 * sg, pitch: 0.06};
          const set = FP(g.side, S.face.target('temple', g.side, hd, {dy: 0.03 * v, dz: 0.02 * v}), 0.45 + 0.3 * v, 0.3);
          set.head_yaw = 0; set.head_roll = hd.roll; set.head_pitch = hd.pitch;
          return {set, add: {brow_knit: 0.3, brow_l: 0.2, gaze_y: 0.3}};
        }
        return {set: HP(g.side, 0.1, 1.18 + 0.02 * v, 0.0, 0.5, 0.3), add: {head_roll: (g.side === 'r' ? 1 : -1) * 0.08, brow_knit: 0.3, brow_l: 0.2, gaze_y: 0.3}};
      }},
      spin: {dur: 2.2, attack: 0.15, release: 0.3, fn(u) {
        const k = Math.sin(Math.PI * clamp(u / 1.8, 0, 1));
        return {set: {body_yaw: k}, add: {head_yaw: 0.35 * Math.sin(2 * Math.PI * clamp(u / 1.8, 0, 1)), tail_wag: 0.6 * Math.sin(2 * Math.PI * 1.5 * u), smile: 0.3}};
      }},
      quizzical: {dur: 1.8, attack: 0.3, release: 0.5, fn(u, g) {
        g.dir = g.dir || pick([-1, 1]);
        return {add: {head_roll: 0.11 * g.dir, brow_l: g.dir > 0 ? 0.45 : -0.1, brow_r: g.dir > 0 ? -0.1 : 0.45, brow_knit: 0.2, ear_l: 0.15 * g.dir, ear_r: -0.15 * g.dir}};
      }},
      startle_hands: {dur: 1.0, attack: 0.06, release: 0.6, fn() { return {add: {hand_l_y: 0.18, hand_r_y: 0.18, hand_l_z: 0.12, hand_r_z: 0.12, hand_l_open: 0.5, hand_r_open: 0.5}}; }},
    };
    const GEST_ALIAS = {wave_hello: 'wave', hello: 'wave', bye: 'wave', nod_yes: 'nod', yes: 'nod', no: 'shake_head', shake: 'shake_head',
      wink_left: ['wink', {own: 'left'}], wink_right: ['wink', {own: 'right'}], ear_flick_left: ['ear_flick', {own: 'left'}],
      ear_flick_right: ['ear_flick', {own: 'right'}], wiggle_ears: 'ear_wiggle', wag_tail: 'tail_wag', wiggle_nose: 'nose_wiggle',
      tongue_out: 'stick_tongue_out', raise_hand: 'raise_paw', raise_arm: 'raise_paw', hooray: 'raise_both', celebrate: 'raise_both',
      raise_both_paws: 'raise_both', turn_left: ['turn', {dir: 1}], turn_right: ['turn', {dir: -1}],   // own left = screen right
      turn_around: 'turn_around', lean_left: ['lean_side', {dir: 1}], lean_right: ['lean_side', {dir: -1}], lean_forward: 'lean_in',
      peekaboo: ['cover_eyes', {peekaboo: true}], heart: 'hand_on_heart', thinking: 'think', hmm: 'think', giggle: 'laugh',
      tilt_head_left: ['tilt_head', {dir: 1}], tilt_head_right: ['tilt_head', {dir: -1}], bounce: 'jump', hop: 'jump'};
    const GEST_PARAMS = {};   // gesture name → params it writes (for release-by-part); filled lazily

    // ---------------------------------------------------------------- THINKING while a commanded move is planned
    // think({variant, avoid, side, dir}) starts one of these looped, subtle "let me think" motions on the gesture layer (src
    // 'think', priority between the listening / speech layer and co-speech gestures); thinkEnd(why, fadeS) fades it out.
    // Real rig motion only. Never reported: no onAction events, nothing in S.log / state().active, so the action log
    // never itemizes it ("what did you just do?" stays the commanded move). A keyframe job's start and any other
    // successful command fade it on their own (the planned move takes over through the normal additive composition; the
    // job's rest snapshot excludes the thinking offsets, so its poses land where the planner put them). Arm variants
    // need arms (think({arms: false}) = an armless character: the head variants only) and a free paw (no held prop / pose / commanded gesture / co-speech gesture / playing job on it); head variants a
    // free head (no commanded look / held head / playing job). Each variant is a held loop (dur = THINK_MAX_S safety cap).
    // `dir` (+1 / -1) = which way it looks / tilts; no motion rng is used (the idle layers' random stream is untouched).
    const THINK_MAX_S = 20;
    const THINK_LEAD = 0.25;   // the eyes / brows (the 'add' channels) lead with this attack; the head eases in itself (eh), a paw on the gesture's attack
    // the head's part of a head-only variant, scaled by the character's body-file tuning (gesture_tuning.think.amp: the
    // muse is framed full-body, its face a fifth of the frame's height, so the shared angles barely show); 1 without one
    const eh = u => mj(clamp(u / 0.4, 0, 1)) * thinkAmp();
    const thinkAmp = () => { const a = S.gtune && S.gtune.think && +S.gtune.think.amp; return a > 0 ? clamp(a, 0.5, 3) : 1; };
    const tapAt = (u, T, n, f) => { const x = u % T; return x < n / f ? Math.pow(Math.sin(Math.PI * f * x), 2) : 0; };   // n taps at f Hz every T s
    const THINK = {
      chin: {arm: true, can: () => !S.chinNone, attack: 0.4, release: 0.45, fn(u, g) {               // MAIN: eyes break contact DOWN and AWAY,
        // the head turns a little away and down, a paw comes to the chin (a fingertip tapping it now and then), brows knit
        const s = g.side, sg = s === 'r' ? 1 : -1, tap = u > 0.8 ? tapAt(u - 0.8, 2.6, 2, 3) : 0, e = mj(clamp(u / 0.4, 0, 1));
        const yaw = -sg * 0.08, pitch = 0.06, roll = sg * 0.05;   // away from the paw's side, nose down a little, into the paw
        let set;
        if (S.face) {
          const hd = {yaw: yaw * e, roll: roll * e, pitch: pitch * e};
          set = FP(s, S.face.target('chin', s, hd, {dy: 0.012 * tap, dz: 0.006 * tap}), 0.3, 0);
          set.head_yaw = hd.yaw; set.head_roll = hd.roll; set.head_pitch = hd.pitch;
          set['point_' + s] = 0.45 + 0.3 * tap;   // a fingertip (the muse's point column is its index finger only)
        } else if (S.chin) {   // its own chin, from its geometry (a shape: under its lower lip; the monkey: under its muzzle)
          set = FP(s, S.chin(s, {yaw: yaw * e, roll: roll * e, pitch: pitch * e}, {dy: 0.015 * tap, dz: 0.01 * tap}), 0.3, 0);
        } else set = HP(s, -0.35, 0.75 + 0.02 * tap, 0.35, 0.3, 0);   // the contract's CHIN: the animal shaders rest the knuckles under their own chin
        // (no point_<s> off the muse: for every other character the renderer's legacy map drags a pointing hand to
        // the camera-point pose at the chest, anim/rig.py map_legacy)
        const drift = Math.sin(2 * Math.PI * 0.22 * u);
        const add = {gaze_y: -0.5 + 0.05 * drift, gaze_x: -sg * (0.45 + 0.05 * drift), brow_knit: 0.3, squint: 0.08};
        add['brow_' + s] = 0.12;
        if (!S.face) { add.head_yaw = yaw * e; add.head_pitch = pitch * e; add.head_roll = roll * e; }
        return {set, add};
      }},
      away: {attack: 0.3, release: 0.4, fn(u, g) {                           // the main pose without a paw: down-away gaze, head turn + tilt, brows
        const d = g.dir, e = eh(u), drift = Math.sin(2 * Math.PI * 0.22 * u);
        const add = {gaze_y: -0.5 + 0.05 * drift, gaze_x: d * (0.5 + 0.05 * drift), head_yaw: d * 0.09 * e, head_pitch: 0.06 * e, head_roll: -d * 0.06 * e,
          brow_knit: 0.32, squint: 0.08};
        add[d > 0 ? 'brow_l' : 'brow_r'] = 0.12;
        return {add};
      }},
      up: {attack: 0.3, release: 0.4, fn(u, g) {                             // eyes up and to the side, the head follows a little
        const d = g.dir, drift = Math.sin(2 * Math.PI * 0.3 * u);
        const add = {gaze_x: d * (0.55 + 0.05 * drift), gaze_y: 0.6 + 0.04 * drift, head_yaw: d * 0.07 * eh(u), head_pitch: -0.05 * eh(u), brow_knit: 0.1, squint: 0.08};
        add[d > 0 ? 'brow_r' : 'brow_l'] = 0.28; add[d > 0 ? 'brow_l' : 'brow_r'] = 0.15;
        return {add};
      }},
      tilt: {attack: 0.3, release: 0.4, fn(u, g) {                           // a slow, puzzled head tilt, one brow up
        const d = g.dir, sw = 0.015 * Math.sin(2 * Math.PI * 0.25 * u);
        const add = {head_roll: (d * 0.11 + sw) * eh(u), head_pitch: 0.02 * eh(u), brow_knit: 0.15, gaze_y: 0.12, gaze_x: d * 0.12, ear_l: 0.15 * d, ear_r: -0.15 * d};
        add[d > 0 ? 'brow_r' : 'brow_l'] = 0.35; add[d > 0 ? 'brow_l' : 'brow_r'] = -0.05;
        return {add};
      }},
      hmm: {attack: 0.25, release: 0.4, fn(u, g) {                           // a small "hmm" nod (two dips, a pause), brows knit
        const x = u % 2.6, nod = x < 0.55 ? Math.pow(Math.sin(Math.PI * x / 0.55), 2) : x < 1.1 ? 0.7 * Math.pow(Math.sin(Math.PI * (x - 0.55) / 0.55), 2) : 0;
        return {add: {head_pitch: (0.045 * nod + 0.015) * eh(u), brow_knit: 0.3, gaze_y: -0.12, gaze_x: g.dir * 0.1, squint: 0.12, ear_l: -0.1, ear_r: -0.1}};
      }},
      finger: {arm: true, can: () => !!S.face, attack: 0.4, release: 0.45, fn(u, g) {             // one finger raised by the face ("wait, I know...")
        const s = g.side, sg = s === 'r' ? 1 : -1, bob = 0.015 * Math.sin(2 * Math.PI * 0.5 * u);
        const set = HP(s, 0.12, 0.9 + bob, 0.3, 0.2, 0.6);
        set['point_' + s] = 1; set['thumb_' + s] = 0;   // (finger-only point: `can` above)
        const add = {brow_l: 0.28, brow_r: 0.28, eye_wide: 0.12, gaze_y: 0.3, gaze_x: sg * 0.1, head_pitch: -0.03 * eh(u), head_roll: sg * 0.03 * eh(u)};
        return {set, add};
      }},
    };
    const THINK_HEAD = ['away', 'up', 'tilt', 'hmm'];   // the paw-less fallbacks, the main pose's own first
    function thinkParams(def, g) {
      const keys = new Set();
      for (const u of [0.05, 0.6, 1.3, 2.0]) { const r = def.fn(u, g) || {}; for (const k in (r.set || {})) keys.add(k); for (const k in (r.add || {})) keys.add(k); }
      return [...keys].filter(p => p in RANGE || ALL_PARAMS.includes(p));
    }
    function thinkFree(params) {   // null = free, else why not
      const busy = armsBusy();
      for (const p of params) {
        if (heldBy(p)) return 'held pose on ' + p;
        if (/^(hand_|thumb_|point_|fingers_|shoulder_)/.test(p)) { const s = /_l(_|$)/.test(p) ? 'l' : 'r'; if (s === 'l' ? busy.l : busy.r) return 'paw busy'; }
        if (/^head_/.test(p) && busy.head) return 'head busy';
        if (/^gaze_/.test(p) && S.cmdLook) return 'commanded look';
      }
      for (const J of S.kf) if (!J.term && J.phase === 'playing') return 'a planned move is playing';
      for (const G of S.gestures) if (!G.rel && G.src !== 'think' && priorityOf(G) >= PRI.think && G.t0 <= S.t + 0.5 && G.params && G.params.some(q => params.includes(q))) return G.src + ' gesture on the same parts';
      return null;
    }
    function think(o = {}) {
      const want = THINK[o.variant] ? o.variant : null;
      const order = [];
      if (want) order.push(want);
      // an arm variant that cannot run falls back to the head variants (in the given order, the avoided one last)
      const heads = (Array.isArray(o.fallback) ? o.fallback : THINK_HEAD).filter(v => THINK[v] && v !== want);
      for (const v of heads) if (v !== o.avoid) order.push(v);
      if (o.avoid && THINK[o.avoid] && !order.includes(o.avoid)) order.push(o.avoid);
      const why = [];
      for (const v of order) {
        const def = THINK[v];
        if (def.arm && o.arms === false) { why.push(v + ': no arms'); continue; }   // an armless character: gaze + head + brows only
        // what this character's body cannot do: a raised index finger needs a finger-only point column (off the muse the
        // renderer drags a pointing hand to the camera-point pose at the chest); a chin paw needs an arm that reaches its chin
        if (def.can && !def.can()) { why.push(v + ': not this body'); continue; }
        let side = o.side === 'l' || o.side === 'r' ? o.side : (S.user.present && S.user.ax > 0.15 ? 'r' : 'l');
        if (def.arm && engineOn && engine.heldBy && engine.heldBy(side) && !engine.heldBy(side === 'l' ? 'r' : 'l')) side = side === 'l' ? 'r' : 'l';
        const g = {side, dir: o.dir === -1 || o.dir === 1 ? o.dir : (side === 'r' ? 1 : -1), src: 'think'};
        const params = thinkParams(def, g);
        const no = thinkFree(params);
        if (no) { why.push(v + ': ' + no); continue; }
        thinkEnd('next variant', 0.45);
        // what it lays over: lower layers (fidgets, listening tilts) on the same parts give way softly
        for (const G of S.gestures) if (!G.rel && G.src !== 'think' && priorityOf(G) < PRI.think && G.params && G.params.some(q => params.includes(q)) && /^(hand_|thumb_|point_)/.test(G.params.find(q => params.includes(q)) || '')) G.rel = {t0: S.t, w: weightOf(G)};
        const G = {name: 'think:' + v, g, side, t0: S.t, dur: THINK_MAX_S, attack: def.attack, release: def.release, fn: def.fn, src: 'think', rel: null, params, id: ++S.gid, think: v, addAttack: THINK_LEAD};
        S.gestures.push(G);
        S.thinkN = (S.thinkN || 0) + 1;
        return {ok: true, variant: v, id: G.id, side, dir: g.dir, fallback: v !== want && !!want};
      }
      return {ok: false, reason: why.join('; ') || 'no variant'};
    }
    function thinkEnd(why, fadeS) {
      let n = 0;
      for (const G of S.gestures) if (G.src === 'think' && !G.rel) {
        G.rel = {t0: S.t, w: weightOf(G)}; if (Number.isFinite(+fadeS) && +fadeS > 0) G.release = +fadeS; G.endWhy = why || 'end'; n++;
      }
      return n;
    }
    function thinking() {
      const G = S.gestures.find(x => x.src === 'think' && !x.rel);
      return G ? {on: true, variant: G.think, id: G.id, since: +G.t0.toFixed(3), w: +weightOf(G).toFixed(3)}
        : {on: false, fading: S.gestures.some(x => x.src === 'think'), n: S.thinkN || 0};
    }
    function gestParams(name, g) {
      const def = GEST[name]; if (!def) return [];
      const r = def.fn(0.6, g), keys = new Set();
      for (const k in (r.set || {})) keys.add(k); for (const k in (r.add || {})) keys.add(k);
      return [...keys];
    }
    function resolveSide(side, dflt) {
      // commands: 'left'/'right' = the character's own side; screen sides explicit
      if (side === 'l' || side === 'screen_left') return 'l';
      if (side === 'r' || side === 'screen_right') return 'r';
      if (side === 'left' || side === 'own_left') return 'r';
      if (side === 'right' || side === 'own_right') return 'l';
      return dflt;
    }
    function startGesture(name, g = {}) {
      let def = GEST[name], extra = {};
      if (!def && GEST_ALIAS[name] !== undefined) {
        const a = GEST_ALIAS[name];
        if (Array.isArray(a)) { name = a[0]; extra = a[1]; } else name = a;
        def = GEST[name];
      }
      if (!def) return null;
      g = Object.assign({}, extra, g);
      // per-character tuning from the body file (muse.body.json gesture_tuning.<name>): its defaults (cycles) unless the
      // caller gave them, the rest read by the gesture (g.tune). Characters without a body file have none.
      const tu = S.gtune && S.gtune[name];
      if (tu && typeof tu === 'object') { g.tune = tu; if (g.cycles === undefined && +tu.cycles > 0) g.cycles = +tu.cycles; }
      if (extra.own) g.side = resolveSide(extra.own, 'l');
      g.side = g.side === 'l' || g.side === 'r' ? g.side : resolveSide(g.side, S.user.present && S.user.ax > 0.15 ? 'r' : 'l');
      const dur = typeof def.dur === 'function' ? def.dur(g) : def.dur;
      const G = {name, g, side: g.side, t0: S.t, dur: g.hold ? Infinity : (g.duration_s || dur), attack: def.attack, release: def.release,
        fn: def.fn, src: g.src || 'cmd', rel: null, params: null, id: ++S.gid};
      G.params = gestParams(name, g);
      // a new gesture on the same params replaces older ones softly (they release)
      for (const o of S.gestures) if (!o.rel && o.params.some(p => G.params.includes(p)) && priorityOf(o) <= priorityOf(G)) o.rel = {t0: S.t, w: weightOf(o)};
      S.gestures.push(G);
      if (REPORT_SRC.has(G.src)) { G.rep = true; emit({kind: 'gesture', name, src: G.src, phase: 'start', args: {side: G.side, via: g.via}, did: gestDid(name, G, 0)}); }
      if (G.src !== 'life') { S.log.push({t: +S.t.toFixed(2), gesture: name, side: G.side, src: G.src}); if (S.log.length > 400) S.log.splice(0, 200); }
      return G;
    }
    S.gid = 0;
    const PRI = {life: 0, fidget: 1, listen: 2, speech: 2, think: 2.2, cospeech: 2.5, reflex: 3, cmd: 4};   // think
    const REPORT_SRC = new Set(['cmd', 'reflex', 'fidget', 'cospeech']);
    const GDID = {wave: ['waving', 'waved'], nod: ['nodding', 'nodded'], shake_head: ['shaking its head', 'shook its head'], wink: ['winking', 'winked'],
      ear_wiggle: ['wiggling its ears', 'wiggled its ears'], nose_wiggle: ['wiggling its nose', 'wiggled its nose'], tail_wag: ['wagging its tail', 'wagged its tail'],
      clap: ['clapping', 'clapped'], thumbs_up: ['giving a thumbs-up', 'gave a thumbs-up'], shrug: ['shrugging', 'shrugged'], bow: ['bowing', 'bowed'],
      point: ['pointing', 'pointed'], stick_tongue_out: ['sticking its tongue out', 'stuck its tongue out'], jump: ['bouncing in its seat', 'bounced in its seat'],
      turn: ['turning its body', 'turned its body'], turn_around: ['turning partway around', 'turned partway around'], spin: ['swivelling around in its seat', 'swivelled around in its seat'],
      dance: ['dancing in its seat', 'danced in its seat'], laugh: ['laughing', 'laughed'], cover_eyes: ['covering its eyes', 'covered its eyes'],
      raise_paw: ['raising a paw', 'raised a paw'], raise_both: ['raising both paws', 'raised both paws'], hand_on_heart: ['putting a paw on its heart', 'put a paw on its heart'],
      think: ['thinking, paw on its chin', 'thought, paw on its chin'], stretch: ['stretching', 'stretched'], yawn: ['yawning', 'yawned'], sigh: ['sighing', 'sighed'],
      blow_kiss: ['blowing a kiss', 'blew a kiss'], high_five: ['holding up a paw for a high five', 'held up a paw for a high five'], quizzical: ['tilting its head, puzzled', 'tilted its head, puzzled'],
      look_around: ['looking around', 'looked around'], scratch_head: ['scratching its head', 'scratched its head'], flap_arms: ['flapping its arms', 'flapped its arms']};
    function gestDid(name, G, past) {
      const d = GDID[name] || ['doing a ' + name.replace(/_/g, ' '), 'did a ' + name.replace(/_/g, ' ')];
      const sideW = /^(wave|point|thumbs_up|raise_paw|hand_on_heart|think|high_five|blow_kiss|scratch_head|wink)$/.test(name) && G && G.side ? ` with its ${name === 'wink' ? (G.side === 'l' ? 'right eye' : 'left eye') : (G.side === 'l' ? 'right paw' : 'left paw')}` : '';
      return d[past ? 1 : 0] + sideW;
    }
    const priorityOf = G => PRI[G.src] ?? 4;
    function weightOf(G, att) {   // att: another attack for the same envelope (a thinking gesture's face channels lead)
      const u = S.t - G.t0, a = att != null ? att : G.attack;
      let w = a > 0 ? mj(u / a) : 1;
      if (Number.isFinite(G.dur) && u > G.dur - G.release) w = Math.min(w, 1 - mj((u - (G.dur - G.release)) / G.release));
      if (G.rel) w = Math.min(w, (att != null ? (a > 0 ? mj((G.rel.t0 - G.t0) / a) : 1) : G.rel.w) * (1 - mj((S.t - G.rel.t0) / Math.max(0.15, G.release))));
      return clamp(w, 0, 1);
    }
    function applyGestures(o) {
      S.thinkD = null;
      S.gestures.sort((a, b) => priorityOf(a) - priorityOf(b) || a.id - b.id);
      const keep = [];
      for (const G of S.gestures) {
        const u = S.t - G.t0;
        const done = (Number.isFinite(G.dur) && u >= G.dur) || (G.rel && S.t - G.rel.t0 >= Math.max(0.15, G.release));
        if (done) { if (G.rep) emit({kind: G.isExpr ? 'expression' : 'gesture', name: G.name.replace(/^expr:/, ''), src: G.src, phase: G.rel && u < G.dur - G.release - 0.05 ? 'interrupted' : 'end', args: {side: G.side}, did: G.isExpr ? 'stopped looking ' + G.name.slice(5) : gestDid(G.name, G, 1) + (G.rel && u < G.dur - G.release - 0.05 ? ' (cut short)' : '')}); continue; }
        if (G.t0 > S.t) { keep.push(G); continue; }   // scheduled (co-speech, a future start)
        keep.push(G);
        const w = weightOf(G), wa = G.addAttack != null ? weightOf(G, G.addAttack) : w; if (w <= 0 && wa <= 0) continue;
        const r = G.fn(u, G.g), ki = Number.isFinite(+G.g.intensity) && G.g.intensity !== undefined ? clamp(+G.g.intensity, 0.2, 1.5) : 1;
        const th = G.src === 'think' ? {} : null;   // what the thinking layer adds this frame (a job's rest excludes it)
        if (th) { for (const k in (r.set || {})) if (k in o) th[k] = o[k]; for (const k in (r.add || {})) if (k in o && !(k in th)) th[k] = o[k]; }
        if (r.set) for (const k in r.set) if (k in o) o[k] = lerp(o[k], r.set[k], w * Math.min(1, ki));
        if (r.add) for (const k in r.add) if (k in o) o[k] += wa * ki * K.gest * r.add[k];
        if (th) { S.thinkD = S.thinkD || {}; for (const k in th) S.thinkD[k] = (S.thinkD[k] || 0) + o[k] - th[k]; }
        if (S.gRec) S.gRec.push({G, w, ki, r});
      }
      S.gestures = keep;
    }
    function heldBy(param) { return S.holds[param] && !S.holds[param].out; }

    // ---------------------------------------------------------------- expressions
    const EXPR = {
      happy: {set: {smile: 0.8, squint: 0.35}, add: {brow_l: 0.2, brow_r: 0.2, ear_l: 0.3, ear_r: 0.3, tail_raise: 0.3}},
      smile: {set: {smile: 0.75, squint: 0.25}, add: {brow_l: 0.12, brow_r: 0.12}},
      big_smile: {set: {smile: 1, squint: 0.5}, add: {brow_l: 0.25, brow_r: 0.25, ear_l: 0.35, ear_r: 0.35}},
      sad: {set: {smile: 0, brow_knit: 0.5, eye_wide: 0}, add: {brow_l: 0.4, brow_r: 0.4, ear_l: -0.6, ear_r: -0.6, head_pitch: 0.12, blink_l: 0.2, blink_r: 0.2, tail_raise: -0.3}},   // brows 0.2 -> 0.4 (with the knit they read angry below ~0.3 raise)
      surprised: {set: {eye_wide: 0.9, mouth_open: 0.4}, add: {brow_l: 0.75, brow_r: 0.75, ear_l: 0.55, ear_r: 0.55, head_z: -0.02, body_pitch: -0.12}},
      angry: {set: {brow_knit: 0.85, smile: 0, squint: 0.4}, add: {brow_l: -0.6, brow_r: -0.6, ear_l: -0.5, ear_r: -0.5, nostril: 0.5, head_pitch: 0.06}},
      scared: {set: {eye_wide: 0.85, brow_knit: 0.45, smile: 0}, add: {brow_l: 0.6, brow_r: 0.6, ear_l: -0.7, ear_r: -0.7, body_pitch: -0.3, shoulder_l: 0.4, shoulder_r: 0.4}},
      disgusted: {set: {squint: 0.45, brow_knit: 0.5, smile: 0}, add: {nostril: 0.6, nose_wiggle: 0.3, head_pitch: -0.05, head_z: -0.02, ear_l: -0.3, ear_r: -0.3}},
      thinking: {set: {brow_knit: 0.3}, add: {head_roll: 0.08, brow_l: 0.2}},
      confused: {set: {brow_knit: 0.3}, add: {head_roll: 0.14, brow_l: 0.5, brow_r: -0.2}},
      excited: {set: {smile: 0.9, eye_wide: 0.4}, add: {brow_l: 0.4, brow_r: 0.4, ear_l: 0.7, ear_r: 0.7, tail_raise: 0.5}},
      sleepy: {set: {blink_l: 0.5, blink_r: 0.5, eye_wide: 0}, add: {ear_l: -0.35, ear_r: -0.35, head_pitch: 0.1, head_roll: 0.06}},
      shy: {set: {smile: 0.5}, add: {gaze_y: -0.5, head_roll: 0.1, head_pitch: 0.08, ear_l: -0.2, ear_r: -0.2}},
      curious: {set: {eye_wide: 0.25}, add: {head_roll: 0.12, brow_l: 0.35, brow_r: 0.25, ear_l: 0.5, ear_r: 0.5, body_pitch: 0.15}},
      smug: {set: {smile: 0.45, squint: 0.4}, add: {brow_l: 0.35, brow_r: -0.1, head_pitch: -0.05, head_roll: 0.05}},
      love: {set: {smile: 0.85, squint: 0.55}, add: {head_roll: 0.1, ear_l: 0.2, ear_r: 0.2, brow_l: 0.25, brow_r: 0.25}},
      proud: {set: {smile: 0.7, squint: 0.2}, add: {head_pitch: -0.07, body_pitch: -0.1, brow_l: 0.2, brow_r: 0.2, ear_l: 0.3, ear_r: 0.3, tail_raise: 0.4}},
      silly: {set: {tongue_out: 0.85, mouth_open: 0.3, squint: 0.45}, add: {head_roll: 0.12, brow_l: 0.35, brow_r: -0.1}},
      embarrassed: {set: {smile: 0.4}, add: {gaze_y: -0.45, gaze_x: 0.3, head_roll: 0.09, head_pitch: 0.07, ear_l: -0.35, ear_r: -0.35, shoulder_l: 0.25, shoulder_r: 0.25}},
      calm: {set: {smile: 0.25}, add: {blink_l: 0.08, blink_r: 0.08}},
      worried: {set: {brow_knit: 0.5, smile: 0}, add: {brow_l: 0.3, brow_r: 0.3, ear_l: -0.3, ear_r: -0.3, shoulder_l: 0.2, shoulder_r: 0.2, head_pitch: 0.04}},
      suspicious: {set: {squint: 0.5, brow_knit: 0.3, smile: 0.05}, add: {brow_l: -0.25, brow_r: 0.3, head_yaw: 0.06, gaze_x: 0.35, head_pitch: 0.03}},
      serious: {set: {smile: 0, brow_knit: 0.25, squint: 0.15}, add: {brow_l: -0.1, brow_r: -0.1}},
      frown: {set: {smile: 0, brow_knit: 0.55}, add: {brow_l: -0.35, brow_r: -0.35, ear_l: -0.2, ear_r: -0.2}},
      pout: {set: {smile: 0, brow_knit: 0.25}, add: {head_pitch: 0.06, gaze_y: -0.3, brow_l: 0.15, brow_r: 0.15, ear_l: -0.3, ear_r: -0.3}},
      laughing: {set: {smile: 1, squint: 0.7, mouth_open: 0.3}, add: {head_pitch: -0.05, ear_l: 0.2, ear_r: 0.2}},
      evil: {set: {smile: 0.6, squint: 0.55, brow_knit: 0.6}, add: {brow_l: -0.4, brow_r: -0.4, head_pitch: 0.08}},
    };
    EXPR.surprise = EXPR.surprised; EXPR.fear = EXPR.scared; EXPR.afraid = EXPR.scared; EXPR.disgust = EXPR.disgusted;
    EXPR.mad = EXPR.angry; EXPR.joy = EXPR.happy; EXPR.tired = EXPR.sleepy; EXPR.bored = EXPR.sleepy; EXPR.think = EXPR.thinking;
    const ALIAS_EXPR = {surprised: 'shocked amazed astonished surprising', sad: 'unhappy upset gloomy sorry', angry: 'grumpy furious cross annoyed angrily',
      happy: 'glad joyful cheerful cheer cute', scared: 'frightened terrified fearful scary', worried: 'nervous anxious', confused: 'puzzled',
      disgusted: 'grossed gross', excited: 'thrilled', silly: 'goofy funny', thinking: 'thoughtful', love: 'loving', laughing: 'laugh'};
    for (const [k, v] of Object.entries(ALIAS_EXPR)) for (const a of v.split(' ')) EXPR[a] = EXPR[k];
    function startExpression(name, intensity = 1, hold = false, durS = 2.6, ease = 0.35) {
      const e = EXPR[name]; if (!e) return false;
      const k = clamp(+intensity || 1, 0, 1.5);
      // expressions ride the gesture machinery with their own name so a new one replaces the old one
      for (const o of S.gestures) if (o.isExpr && !o.rel) o.rel = {t0: S.t, w: weightOf(o)};
      const G = {name: 'expr:' + name, g: {}, t0: S.t, dur: hold ? Infinity : durS, attack: ease, release: Math.max(0.3, ease), src: 'cmd', rel: null,
        isExpr: true, id: ++S.gid,
        fn: () => {
          const set = {}, add = {};
          for (const p in (e.set || {})) set[p] = (e.set[p]) * Math.min(1, k) + (k > 1 ? (e.set[p]) * (k - 1) * 0.3 : 0);
          for (const p in (e.add || {})) add[p] = e.add[p] * k;
          return {set, add};
        }};
      G.params = [...Object.keys(e.set || {}), ...Object.keys(e.add || {})];
      S.gestures.push(G); G.rep = true;
      S.log.push({t: +S.t.toFixed(2), expression: name, hold});
      return true;
    }

    // ---------------------------------------------------------------- commands: holds with min-jerk ease
    const GROUPS = {
      head: ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z'],
      eyes: ['gaze_x', 'gaze_y', 'blink_l', 'blink_r', 'eye_wide', 'squint'], gaze: ['gaze_x', 'gaze_y'], look: ['gaze_x', 'gaze_y'],
      brows: ['brow_l', 'brow_r', 'brow_knit'], ears: ['ear_l', 'ear_r', 'ear_flick_l', 'ear_flick_r', 'ear_l_turn', 'ear_r_turn'],
      mouth: ['mouth_open', 'tongue_out', 'smile'], tongue: ['tongue_out'], nose: ['nose_wiggle', 'nostril'],
      tail: ['tail_wag', 'tail_raise', 'tail_curl'], body: ['body_yaw', 'body_lean', 'body_bounce', 'body_pitch'],
      paws: [], arms: [], hands: [], left_paw: [], right_paw: [], shoulders: ['shoulder_l', 'shoulder_r'],
      expression: ['smile', 'squint', 'eye_wide', 'brow_l', 'brow_r', 'brow_knit', 'mouth_open'],
    };
    for (const s of ['l', 'r']) {
      const ps = ['hand_' + s + '_x', 'hand_' + s + '_y', 'hand_' + s + '_z', 'hand_' + s + '_open', 'hand_' + s + '_turn', 'thumb_' + s, 'point_' + s, 'shoulder_' + s];
      GROUPS['paw_' + s] = ps; GROUPS.paws.push(...ps);
      GROUPS[s === 'l' ? 'right_paw' : 'left_paw'].push(...ps);   // own right paw = screen-left
    }
    GROUPS.arms = GROUPS.hands = GROUPS.paws;
    GROUPS.left_ear = ['ear_r', 'ear_flick_r', 'ear_r_turn']; GROUPS.right_ear = ['ear_l', 'ear_flick_l', 'ear_l_turn'];
    GROUPS.left_eye = ['blink_r']; GROUPS.right_eye = ['blink_l'];
    GROUPS.hand_l = GROUPS.paw_l; GROUPS.hand_r = GROUPS.paw_r;
    GROUPS.face = [...new Set([...GROUPS.expression, ...GROUPS.eyes, ...GROUPS.brows, 'nostril', 'nose_wiggle', 'tongue_out'])];
    GROUPS.everything = RIG_PARAMS.filter(p => !LIPSYNC.has(p));
    function expandParts(parts) {
      if (parts === 'all' || parts === undefined || parts === null) return 'all';
      const list = Array.isArray(parts) ? parts : [parts], out = new Set();
      for (const p of list) {
        if (GROUPS[p]) GROUPS[p].forEach(x => out.add(x));
        else if (RANGE[p] || RIG_PARAMS.includes(p)) out.add(p);
        else if (typeof p === 'string') for (const q of RIG_PARAMS) if (q.startsWith(p + '_')) out.add(q);
      }
      return out;
    }
    // natural top speeds: an ease is never shorter than distance × 1.875 / vmax (min-jerk peak velocity)
    const vmax = p => /^head_(yaw|pitch|roll)/.test(p) ? 1.6 : /^head_/.test(p) ? 0.25 : /^body_/.test(p) ? 2.2 :
      /^hand_.*_(x|y|z)$/.test(p) ? 3.0 : /^(gaze|blink)/.test(p) ? 12 : 4;
    const minEase = (p, d) => 1.875 * Math.abs(d) / vmax(p);
    function hold(param, value, easeS, isHold, durS) {
      if (!(param in RANGE) && !RIG_PARAMS.includes(param)) return false;
      if (LIPSYNC.has(param)) return false;
      const [lo, hi] = (S.rng && S.rng[param]) || RANGE[param] || [-1, 1];
      const SO = learned && S.shownOut ? S.shownOut : null;
      const cur = SO && Number.isFinite(SO[param]) ? SO[param] : S.lastOut ? S.lastOut[param] : 0;
      const prev = S.holds[param];
      const from = prev && !prev.out ? holdValue(prev) : cur, to = clamp(+value, lo, hi);
      const ease = Math.max(easeS, minEase(param, to - from));
      S.holds[param] = {from, to, t0: S.t, ease, until: isHold ? Infinity : S.t + ease + (durS ?? 2.5), out: null};
      return true;
    }
    function holdValue(h) { return lerp(h.from, h.to, mj((S.t - h.t0) / h.ease)); }
    function releaseHold(param, easeS) {
      const h = S.holds[param]; if (!h || h.out) return;
      const v = holdValue(h), live = S.react && Number.isFinite(S.react[param]) ? S.react[param] : 0;
      h.out = {t0: S.t, ease: Math.max(easeS || 0.4, minEase(param, v - live)), v};
    }
    // a commanded gesture / expression owns the channels it drives — a pose HELD on one of them is let go (holds
    // are laid over gestures: the nod / wave / face he asked for was invisible under an earlier held pose, and said "done")
    function ownOverHolds(params) {
      const freed = [];
      for (const p of Array.isArray(params) ? params : []) if (heldBy(p)) { releaseHold(p, 0.3); freed.push(p); }
      if (freed.length) S.log.push({t: +S.t.toFixed(2), release: freed, why: 'a commanded gesture drives these'});
    }
    function applyHolds(o) {
      for (const p in S.holds) {
        const h = S.holds[p];
        if (!h.out && S.t >= h.until) releaseHold(p, Math.max(0.3, h.ease));
        if (h.out) {
          const s = mj((S.t - h.out.t0) / h.out.ease);
          if (s >= 1) { delete S.holds[p]; continue; }
          o[p] = lerp(h.out.v, o[p], s);
        } else o[p] = holdValue(h);
      }
    }
    function release(parts, easeMs) {
      const set = expandParts(parts), ease = (easeMs ?? 400) / 1000;
      for (const p in S.holds) if (set === 'all' || set.has(p)) releaseHold(p, ease);
      for (const G of S.gestures) {
        if (G.rel || G.src === 'life') continue;
        if (set === 'all' ? G.src === 'cmd' || G.src === 'reflex' : G.params.some(p => set.has(p))) {
          G.rel = {t0: S.t, w: weightOf(G)}; G.release = Math.max(G.release, ease);
        }
      }
      if (set === 'all' || set.has('gaze_x')) { S.cmdLook = null; if (S.around && S.around.kind === 'cmd') S.around = null; }
      S.log.push({t: +S.t.toFixed(2), release: set === 'all' ? 'all' : [...set]});
    }
    // "hold that / freeze / stay like that": the posture (head, body, paws, shoulders) and everything a gesture,
    // expression or hold is moving right now become held poses at their CURRENT values (no jump), until release.
    // Blinks, gaze, breathing, nostrils and ear twitches keep living; idle ears/tail/face keep living unless a gesture
    // is moving them. The gestures that were running are released so they do not resume when the hold ends.
    const POSTURE = /^(head_(yaw|pitch|roll)|body_(yaw|lean|pitch|bounce)|hand_[lr]_|thumb_|point_|shoulder_)/;
    const NEVER_FREEZE = /^(blink_|gaze_|breath$|nostril$|ear_flick_|head_[xyz]$)/;
    function freeze(parts, durS) {
      if (!S.lastOut) return false;
      const sel = parts && parts !== 'all' ? expandParts(parts) : null;
      const moving = new Set();
      for (const G of S.gestures) if (!G.rel && G.src !== 'life') for (const p of G.params || []) moving.add(p);
      for (const p in S.holds) if (!S.holds[p].out) moving.add(p);
      let n = 0;
      for (const p of RIG_PARAMS) {
        if (LIPSYNC.has(p) || LEGACY.has(p) || NEVER_FREEZE.test(p)) continue;
        if (!(POSTURE.test(p) || moving.has(p))) continue;
        if (sel && sel !== 'all' && !sel.has(p)) continue;
        if (S.kf.length && S.kf.some(J => !J.term && J.wc && J.wc[p] > 0.5)) continue;   // a keyframe job pauses its own channels
        const SO = learned && S.shownOut && Number.isFinite(S.shownOut[p]) ? S.shownOut[p] : S.lastOut[p];
        if (hold(p, SO, 0.12, durS === undefined, durS)) n++;
        if (S.handF[p]) { S.handF[p].x = S.lastOut[p]; S.handF[p].v = 0; }   // stop the arm's follow-through too
      }
      for (const G of S.gestures) if (!G.rel && G.src !== 'life' && (!sel || G.params.some(p => sel.has(p)))) G.rel = {t0: S.t, w: weightOf(G)};
      S.log.push({t: +S.t.toFixed(2), freeze: n});
      return n > 0;
    }
    // a pose aimed at the face, for a character with face points (the muse). Named matcher poses get the face
    // point outright (cover_eyes, cover_mouth, thinking / paw on chin, hand_on_head); any other pose whose paw target
    // is near one of the contract's face poses (mouth, chin, eyes — e.g. a pose-tool "touch your mouth") is drawn onto
    // that face point, fully within 0.15 of it, not at all past 0.30 (raise_paw, wave, clap, point... stay as they are).
    const FACE_REFS = [[[-0.30, 0.76, 0.52], 'mouth'], [[-0.60, 0.72, 0.95], 'mouth_mid'], [[-0.35, 0.72, 0.35], 'chin'], [[-0.35, 1.00, 0.45], 'eye']];
    const FACE_NAMED = {cover_eyes: ['eye', 1, 1], cover_mouth: ['mouth', 1, 1], thinking: ['chin', 0.3, 0], hand_on_head: ['temple', 0.8, 0.3]};
    function facePose(name, targets) {
      const out = Object.assign({}, targets);
      for (const s of ['l', 'r']) {
        const k = a => 'hand_' + s + '_' + a;
        if (!(k('x') in out || k('y') in out || k('z') in out)) continue;
        const nm = FACE_NAMED[name];
        if (nm && S.face.has(nm[0])) {
          const h = S.face.target(nm[0], s, name === 'hand_on_head' ? {roll: (s === 'r' ? 1 : -1) * 0.2} : {});
          [out[k('x')], out[k('y')], out[k('z')]] = h;
          if (!(k('open') in targets)) out[k('open')] = nm[1];
          if (!(k('turn') in targets)) out[k('turn')] = nm[2];
          continue;
        }
        const cur = S.lastOut || {};
        const h0 = ['x', 'y', 'z'].map(a => Number.isFinite(+out[k(a)]) ? +out[k(a)] : (cur[k(a)] || 0));
        let wsum = 0, wmax = 0; const acc = [0, 0, 0];
        for (const [ref, anchor] of FACE_REFS) {
          const d = Math.hypot(h0[0] - ref[0], h0[1] - ref[1], h0[2] - ref[2]);
          const w = smooth01((0.30 - d) / 0.15);
          if (w <= 0) continue;
          const wi = w / (d * d + 1e-3), t = S.face.target(anchor, s);
          for (let i = 0; i < 3; i++) acc[i] += wi * t[i];
          wsum += wi; wmax = Math.max(wmax, w);
        }
        if (wsum > 0) {
          ['x', 'y', 'z'].forEach((a, i) => { out[k(a)] = lerp(h0[i], acc[i] / wsum, wmax); });
          if (!(k('turn') in targets)) out[k('turn')] = lerp(cur[k('turn')] || 0, 1, wmax);
        }
      }
      return out;
    }
    function command0(cmd, src) {
      if (!cmd || typeof cmd !== 'object') return false;
      S.idleT = 0; S.bored = Math.max(0, S.bored - 0.4);
      const ease = clamp((+cmd.ease_ms || 0) / 1000 || 0.33, 0.12, 3);
      const holdIt = cmd.hold === true || cmd.hold === 'true';
      const durS = cmd.duration_ms ? clamp(+cmd.duration_ms / 1000, 0.2, 120) : undefined;
      S.log.push({t: +S.t.toFixed(2), cmd: cmd.action, name: cmd.name || cmd.target, hold: holdIt});
      switch (cmd.action) {
        case 'pose': {
          let n = 0;
          const tg = S.face ? facePose(cmd.name, cmd.targets || {}) : (cmd.targets || {});
          for (const [k, v] of Object.entries(tg)) if (hold(k, v, ease, holdIt, durS)) n++;
          return n > 0;
        }
        case 'gesture': {
          const name = String(cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_');
          if (name === 'blink') { forceBlink(1, true, false, true); return true; }
          if (name === 'look_around') { lookAround(clamp(+cmd.repeat || 3, 2, 6), 'cmd'); return true; }
          if (cmd.side === 'both' && !['clap', 'shrug', 'cover_eyes', 'raise_both', 'dance', 'stretch'].includes(name)) {
            if (name === 'wave' || name === 'raise_paw' || name === 'raise_hand') return !!startGesture('raise_both', {src: 'cmd', hold: holdIt, duration_s: durS});
            const a = command0(Object.assign({}, cmd, {side: 'screen_left'}), src), b = command0(Object.assign({}, cmd, {side: 'screen_right'}), src);
            return a || b;
          }
          const g = {side: cmd.side, src: 'cmd', hold: holdIt, via: src};
          // a paw that holds a prop: use the free paw for a one-paw gesture when there is one (the held prop stays put)
          if (engineOn && /^(wave|point|thumbs_up|raise_paw|raise_hand|raise_arm|high_five|blow_kiss|scratch_head|hand_on_heart|think|wave_hello|hello|bye)$/.test(name)) {
            const sd = cmd.side === 'l' || cmd.side === 'r' ? cmd.side : resolveSide(cmd.side, S.user.present && S.user.ax > 0.15 ? 'r' : 'l');
            if (engine.heldBy(sd) && !engine.heldBy(sd === 'l' ? 'r' : 'l')) { g.side = sd === 'l' ? 'r' : 'l'; S.swapNote = ` (its ${sd === 'l' ? 'right' : 'left'} paw is holding something)`; }
          }
          // repeat: 1 (the tool default) = the gesture's natural count; more = that many cycles
          if (+cmd.repeat > 1) g.cycles = clamp(+cmd.repeat * (name === 'wave' ? 2 : 1), 1, 14);
          if (durS) g.duration_s = durS;
          if (cmd.dir) g.dir = cmd.dir;
          if (cmd.intensity !== undefined) g.intensity = +cmd.intensity;
          if (name === 'point' && cmd.target) g.dir = cmd.target;
          const G = startGesture(name, g);
          if (G) ownOverHolds(G.params);
          if (G && G.name === 'point') {   // look where it points
            const side = G.g.dir && /side|left|right/.test(G.g.dir) ? (G.side === 'r' ? 1 : -1) : 0;
            S.focus = {kind: 'point', W: side ? [side * 0.35, 0.08] : [0, 0], until: S.t + (holdIt ? 9e9 : 2.0), gain: 0.7};
          }
          return !!G;
        }
        case 'look': {
          let target = cmd.target || 'user';
          if (cmd.screen_side === 'l' || cmd.screen_side === 'r') target = cmd.screen_side === 'l' ? 'screen_left' : 'screen_right';
          if (target === 'around') { lookAround(clamp(+cmd.repeat || 3, 2, 6), 'cmd'); return true; }
          S.cmdLook = {target, hold: holdIt, until: S.t + (durS ?? 2.5)};
          S.avert = null; S.around = null; S.focus = null;
          return true;
        }
        case 'expression': {
          const name = String(cmd.name || '').toLowerCase();
          if (name === 'neutral' || name === 'none') { release('expression', cmd.ease_ms); for (const o of S.gestures) if (o.isExpr && !o.rel) o.rel = {t0: S.t, w: weightOf(o)}; return true; }
          const okE = startExpression(name, cmd.intensity ?? 1, holdIt, durS ?? 2.6, ease);
          if (okE) { const GE = S.gestures[S.gestures.length - 1]; if (GE && GE.isExpr) ownOverHolds(GE.params); }
          return okE;
        }
        case 'release': {
          release(cmd.parts ?? 'all', cmd.ease_ms); if (engineOn) engine.release(cmd.parts === undefined || cmd.parts === 'all' ? 'all' : expandParts(cmd.parts));
          if (cmd.parts === undefined || cmd.parts === 'all') kfInterruptAll('relaxed');   // "relax" ends a keyframe job too
          return true;
        }
        case 'hold': {
          const a = freeze(cmd.parts, durS); const b = engineOn && (cmd.parts === undefined || cmd.parts === 'all') ? engine.freeze() : false;
          let c = false;   // "hold that" pauses a playing keyframe job where it is (until release / stop)
          if (cmd.parts === undefined || cmd.parts === 'all') for (const J of S.kf) if (!J.term && J.phase === 'playing') { J.pl.st.frozen = true; c = true; }
          return a || b || c;
        }
        case 'stop': if (engineOn) engine.stopAll('stopped'); release('all', cmd.ease_ms); kfInterruptAll('stopped'); return true;
        default: return false;
      }
    }

    // command(cmd, {src}) → {ok, id, did, reason}: one shape for every source (tools, matcher, page UI, tests)
    const SRC = {voice: 'tool', 'voice-early': 'tool', model: 'tool', tool: 'tool', matcher: 'matcher', caption: 'matcher', ui: 'ui', test: 'ui'};
    function command(cmd, o2) {
      if (!cmd || typeof cmd !== 'object') return {ok: false, reason: 'not a command'};
      const src = SRC[(o2 && o2.src) || cmd.src || cmd.source] || (o2 && o2.src) || 'tool';
      if (cmd.action === 'keyframes') return kfCommand(cmd, src);   // planner contract §3: a streamed keyframe job
      if (cmd.action === 'move' || cmd.action === 'object') {
        const r = engineCommand(cmd, src);
        if (r && r.ok) thinkEnd('a commanded ' + cmd.action, 0.3);
        // a new whole-body move / object action (or the engine's 'stop' move) interrupts a keyframe job
        if (r && r.ok && S.kf.length) kfInterruptAll(cmd.action === 'move' && /^stop$/i.test(String(cmd.name || '')) ? 'stopped' : 'replaced by a new ' + (cmd.action === 'move' ? 'move' : 'action'));
        return r;
      }
      S.swapNote = '';
      const id = ++S.cmdN;
      let ok = false, err = null;
      try { ok = command0(cmd, src) !== false; } catch (e) { err = e; }
      if (ok) thinkEnd('a commanded ' + (cmd.action || 'command'), 0.3);   // any commanded action replaces the thinking
      const did = ok ? cmdDid(cmd) + (S.swapNote || '') : null;
      const reason = ok ? undefined : err ? 'error: ' + String(err).slice(0, 80) : cmdWhyNot(cmd);
      const kind = cmd.action || 'command';
      if (kind !== 'gesture' || !ok) emit({kind, name: cmd.name || cmd.target || (cmd.parts !== undefined ? String(cmd.parts) : undefined), id, src, phase: ok ? (kind === 'release' || kind === 'hold' || kind === 'stop' ? 'end' : 'start') : 'fail', args: cmd, did, reason});
      return ok ? {ok: true, id, did} : {ok: false, id, reason};
    }
    S.cmdN = 0;
    function cmdDid(c) {
      const hold = c.hold === true || c.hold === 'true' ? ' and holding it' : '';
      switch (c.action) {
        case 'gesture': { const n = String(c.name || '').toLowerCase().replace(/[\s-]+/g, '_'); const G = S.gestures[S.gestures.length - 1]; return gestDid(G && G.src === 'cmd' ? G.name : n, G, 0) + (+c.repeat > 1 ? ` ${c.repeat} times` : '') + hold; }
        case 'pose': return 'moving into the pose it was asked for' + hold;
        case 'look': { const lp = typeof c.target === 'string' ? lookProp(c.target) : null;   // at a thing by name
          return `looking ${c.target === 'user' || c.target === 'me' || c.target === 'camera' ? 'at him' : lp ? 'at the ' + lp.name : String(c.target || 'away').replace(/_/g, ' ')}` + hold; }
        case 'expression': return `making a ${String(c.name || '')} face` + hold;
        case 'release': return 'relaxing (letting go of held poses; anything in its paws stays there)';
        case 'hold': return 'freezing in place';
        case 'stop': return 'stopping';
      }
      return 'doing it';
    }
    function cmdWhyNot(c) {
      if (c.action === 'gesture') return `it doesn't know the gesture "${c.name}"`;
      if (c.action === 'expression') return `it doesn't know the expression "${c.name}"`;
      if (c.action === 'pose') return 'none of those pose targets exist';
      if (c.action === 'hold') return 'nothing to hold yet';
      return `unknown action "${c.action}"`;
    }
    // whole-body moves + objects: the muse (a character with a body file) gets the engine; the others keep today's
    // seated gesture approximations (truthfully described) or a refusal
    const APPROX = {spin: 'spin', twirl: 'spin', spin_around: 'spin', turn: 'turn', turn_left: 'turn_left', turn_right: 'turn_right', turn_away: 'turn_around',
      turn_around: 'turn_around', jump: 'jump', hop: 'jump', bounce: 'jump', dance: 'dance', bow: 'bow', wiggle: 'dance', shimmy: 'shoulder_roll', sway: 'lean_side'};
    function engineCommand(cmd, src) {
      if (engineOn) {
        const r = engine.command(cmd, src, ++S.cmdN);   // one id space with the rules commands
        S.idleT = 0; S.bored = Math.max(0, S.bored - 0.4);
        S.log.push({t: +S.t.toFixed(2), cmd: cmd.action, name: cmd.name || cmd.verb, ok: r.ok});
        return r;
      }
      const id = ++S.cmdN;
      if (cmd.action === 'object') { const reason = 'there are no objects in this scene'; emit({kind: 'object', name: cmd.verb, id, src, phase: 'fail', args: cmd, reason}); return {ok: false, id, reason}; }
      const n = String(cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_'), g = APPROX[n];
      if (g) {
        const ok = command0({action: 'gesture', name: g, side: cmd.dir, dir: n === 'turn' ? (/right/.test(String(cmd.dir)) ? -1 : 1) : undefined, hold: cmd.hold}, src);
        if (ok) { const did = gestDid(g, null, 0) + ' (this character stays in its seat — an approximation)'; emit({kind: 'move', name: n, id, src, phase: 'start', args: cmd, did}); return {ok: true, id, did, approx: true}; }
      }
      const reason = `this character can't ${n.replace(/_/g, ' ') || 'do that'} — it stays in its seat`;
      emit({kind: 'move', name: n, id, src, phase: 'fail', args: cmd, reason});
      return {ok: false, id, reason};
    }
    // co-speech gestures from outside (app/cospeech.js): def = {name, dur, attack, release, fn(u, g) → {set, add}, holdSafe}
    // opts = {src:'cospeech', side, at (motion clock s; may be in the future — motion.now()), delay (s from now), intensity,
    // parts}. Refused (null) when a commanded move, a held pose, a commanded gesture, or a held prop owns those parts.
    function defParams(def, g) {
      const keys = new Set();
      for (const u of [0.1, 0.35, 0.6, 0.85]) { let r; try { r = def.fn(u * (def.dur || 1), g) || {}; } catch (e) { r = {}; } for (const k in (r.set || {})) keys.add(k); for (const k in (r.add || {})) keys.add(k); }
      return [...keys];
    }
    function playGesture(def, o = {}) {
      if (!def || typeof def.fn !== 'function') return null;
      const side = o.side === 'l' || o.side === 'r' ? o.side : resolveSide(o.side, 'l');
      const g = Object.assign({}, o.g || {}, {side, intensity: o.intensity, src: o.src || 'cospeech'});
      if ((S.affE || 0) !== 0 && g.src === 'cospeech') g.intensity = clamp((Number.isFinite(+g.intensity) ? +g.intensity : 1) * (1 + 0.4 * S.affE), 0.4, 1.4);   // affect: livelier strokes when excited, smaller when low
      const params = (o.parts ? [...expandParts(o.parts)] : defParams(def, g)).filter(p => p in RANGE || ALL_PARAMS.includes(p));
      const busy = armsBusy();
      for (const p of params) {
        if (heldBy(p)) return null;
        if (/^(hand_|thumb_|point_|fingers_)/.test(p)) {
          const s = /_l(_|$)/.test(p) ? 'l' : 'r';
          if ((s === 'l' ? busy.l : busy.r) && !(def.holdSafe && (s === 'l' ? busy.holdL && !busy.ctrlL : busy.holdR && !busy.ctrlR))) return null;
        }
        if (/^body_/.test(p) && busy.body) return null;
        if (/^head_(yaw|pitch)/.test(p) && busy.head && !def.holdSafe) return null;
      }
      for (const G of S.gestures) if (G.src === 'cmd' && !G.rel && G.params.some(q => params.includes(q))) return null;
      // `at`: on the PLAYBACK timeline (the audio tap / rig stamp clock, seconds) when o.clock === 'playback', or by default
      // for src 'cospeech' once the page feeds step(dt, {playT}); o.clock === 'motion' (or no playT yet) = motion.now()
      const pb = o.clock === 'playback' || (o.clock !== 'motion' && (o.src || 'cospeech') === 'cospeech' && Number.isFinite(S.playOff));
      const at = Number.isFinite(+o.at) && o.at !== null ? +o.at + (pb && Number.isFinite(S.playOff) ? S.playOff : 0) : S.t + (Number.isFinite(+o.delay) ? Math.max(0, +o.delay) : 0);
      const dur = typeof def.dur === 'function' ? def.dur(g) : (+def.dur || 1);
      const G = {name: String(def.name || 'cospeech'), g, side, t0: Math.max(S.t, at), dur, attack: +def.attack || 0.15, release: +def.release || 0.25,
        fn: def.fn, src: g.src, rel: null, params, id: ++S.gid, holdSafe: !!def.holdSafe, ext: true};
      S.gestures.push(G);
      G.rep = true; emit({kind: 'gesture', name: G.name, src: G.src, phase: 'start', args: {side, at: +G.t0.toFixed(3)}, did: gestDid(G.name, G, 0)});
      return G.id;
    }
    function cancelGestures(f = {}) {
      let n = 0;
      for (const G of S.gestures) {
        if (G.rel || (f.src && G.src !== f.src) || (f.id && G.id !== f.id)) continue;
        if (G.t0 > S.t) { G.dur = 0; G.rel = {t0: S.t, w: 0}; G.rep = false; emit({kind: 'gesture', name: G.name, src: G.src, phase: 'interrupted', did: 'cancelled before it started'}); n++; continue; }
        G.rel = {t0: S.t, w: weightOf(G)}; G.release = Math.min(G.release, 0.2); n++;
      }
      return n;
    }
    function armsBusy() {
      const b = engineOn ? engine.busy() : {l: false, r: false, body: false, head: false, holdL: false, holdR: false};
      const cmdOn = s => S.gestures.some(G => G.src === 'cmd' && !G.rel && G.params.some(p => p.startsWith('hand_' + s + '_')));
      const holdOn = s => Object.keys(S.holds).some(p => p.startsWith('hand_' + s + '_') && !S.holds[p].out);
      const kb = S.kf && S.kf.length ? kfBusy() : {l: false, r: false, body: false, head: false};   // a playing keyframe job
      return {l: b.l || b.holdL || cmdOn('l') || holdOn('l') || kb.l, r: b.r || b.holdR || cmdOn('r') || holdOn('r') || kb.r,
        body: b.body || Object.keys(S.holds).some(p => /^body_/.test(p) && !S.holds[p].out) || kb.body,
        head: b.head || !!S.cmdLook || Object.keys(S.holds).some(p => /^head_/.test(p) && !S.holds[p].out) || kb.head,
        holdL: b.holdL, holdR: b.holdR, ctrlL: b.l, ctrlR: b.r};
    }
    function stateOut() {
      const es = engineOn ? engine.state() : {posture: 'seated', lying: null, facing: 'camera', gazeFree: false, holding: {left: null, right: null}, props: [], active: []};
      const held = new Set();
      for (const p in S.holds) if (!S.holds[p].out) for (const [g, ps] of Object.entries(GROUPS)) if (ps.length && ps.length < 20 && ps.includes(p) && !/^(everything|face|expression|arms|hands|paws)$/.test(g)) held.add(g);
      const active = es.active.slice();
      for (const G of S.gestures) if (!G.rel && (G.src === 'cmd' || G.src === 'cospeech') && G.t0 <= S.t) active.push({kind: G.isExpr ? 'expression' : 'gesture', name: G.name.replace(/^expr:/, ''), since: +G.t0.toFixed(2), src: G.src});
      if (S.cmdLook) active.push({kind: 'look', name: String(S.cmdLook.target), since: null, held: !!S.cmdLook.hold});
      // a keyframe job — 'name' is what he asked for (the action log reads name), so "in progress: <his request>"
      const kfJ = S.kf.length ? S.kf.filter(J => !J.term).slice(-1)[0] || null : null;
      for (const J of S.kf) if (!J.term && J.phase === 'playing') active.push({kind: 'kf', name: J.description, id: J.id, since: +J.startT.toFixed(2), frozen: !!J.pl.st.frozen});
      const mode = S.mode || 'user';
      const looking = es.gazeFree && engineOn && engine.E.look.w > 0.5 ? 'object' : es.facing === 'away' ? 'away' : S.cmdLook ? String(S.cmdLook.target) : mode === 'user' || mode === 'cmd' ? (S.user.present ? 'user' : 'around') : mode;
      const st = Object.assign({}, es, {t: +S.t.toFixed(3), looking, held: [...held], active, character: S.charName || null});
      st.kf = kfJ ? kfSummary(kfJ) : null;
      return st;
    }
    // ---------------------------------------------------------------- streamed keyframe jobs (contract §3, §5)
    // m.command({action:'keyframes', id, description, poses, done, hold}) / kfFeed / kfRelease / kfState play planner poses
    // ({t, <control>: value} — the keyframe stream format, KF_ALIAS control names) LIVE, for every character, on top of whatever
    // the rules / engine / gestures produce this frame. Player = the keyframe-stream player (makePlayer), real time at the
    // page frame rate: per rig column keys at every pose (an unlisted column holds), a monotone cubic between keys, a provisional
    // ease into a pose whose successor is not parsed yet and a C1 re-plan when it lands, the playback clock braking within
    // KF.HEAD s of the last parsed pose and returning to real time with KF.TAU_UP, never past the last pose, and after the last
    // pose every column eases back to rest (rotations to the nearest whole turn; root_x / root_z stay). Start rule ("2:1"):
    // >= 2 poses AND >= 1.0 s parsed, or the whole list (done) — and released (hold:true waits for kfRelease).
    // Composition = exec_stream.cjs compose / plan/exec.cjs runKf: direct columns additive on this frame's output (pose − rest
    // snapshot taken at the command), clamped to RANGE. A character with legs (the muse) also gets the root with floor contact
    // (body.lift), automatic / blended WORLD feet, camera zoom from the poses parsed so far + follow, gaze_free; the root is
    // relative to where it stood at the start (x / z / yaw) and the final root_x / root_z are handed back to the engine. The
    // seated characters (no root in their renderer) ignore feet and root_x / root_z and show root_y + squash / root_yaw /
    // root_pitch / root_roll on body_bounce / body_yaw / body_pitch / body_lean. A job owns its channels with weight w (min-jerk
    // in over KF.W_IN at the start, out over KF.W_OUT at its end / interruption = the blend back); SPEECH_GROUP channels get
    // w·(1 − speech), so speech keeps the mouth while voiced.
    const KF = {HEAD: 0.35, TAU_UP: 0.15, FOOT_IN: 0.25, FOOT_OUT: 0.35, FOOT_GAP: 1.0, PLAN_MAX: 18, BUF_N: 2, BUF_H: 1.0,
      W_IN: 0.15, W_OUT: 0.4, TIMEOUT: 45, AUTO_RELEASE: 4,
      DO_OUT: 0.35, DO_IN: 0.45, DO_MAX: 25, G: 12, HELD_XYZ: 0.4, CATCH_WIN: 0.6};   // "do" steps, held things, tosses
    const KF_PERIODIC = /^root_(yaw|pitch|roll)$|^%held_[lr]_(spin|tilt|roll)$/, KF_ROOTCOL = /^root_|^squash$/;
    const KF_PAWCOL = /^(hand|thumb|point|fingers)_[lr](_|$)/;   // absolute in a plan with things (J.absW)
    const SPEECH_SET = new Set(SPEECH_GROUP);
    const kwrapH = x => { const y = x - Math.floor(x + 0.5); return y >= 0.5 ? y - 1 : y; };
    const kwrap = x => x - Math.round(x);
    const kmv = (Rm, v) => [0, 1, 2].map(i => Rm[i][0] * v[0] + Rm[i][1] * v[1] + Rm[i][2] * v[2]);
    const ksp2 = (st, target, w, dt) => { const a = w * w * (target - st.x) - 2 * w * st.v; st.v += a * dt; st.x += st.v * dt; return st.x; };
    const ksstep = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
    const kswivel = f => { const q = f - Math.trunc(f); return Math.sign(q) * Math.sin(Math.PI * Math.abs(q)); };
    S.kf = []; S.kfHist = new Map(); S.kfSv = 0; S.kfBase = null; S.kfPaw = {l: 0, r: 0};

    function kfPlayer(J) {
      const P = [], cols = {};
      const st = {complete: false, started: false, tau: 0, r: 0, Tknown: 0, tEnd: Infinity, ext: 0, stalls: 0, stall_s: 0, inStall: false, minRate: 1, replans: 0, frozen: false,
        stopAt: Infinity, paused: false};   // the clock stops at the next "do" step (stopAt) and waits there (paused)
      const isFoot = c => c.charCodeAt(0) === 64;
      const restOf = c => J.restOf(c);
      const tangent = (k, j) => {          // tangent at key j, or null if key j+1 is not known yet (and the stream is still open)
        if (j === 0) return 0;
        if (j + 1 >= k.length) return st.complete ? 0 : null;
        const d0 = (k[j][1] - k[j - 1][1]) / Math.max(1e-6, k[j][0] - k[j - 1][0]), d1 = (k[j + 1][1] - k[j][1]) / Math.max(1e-6, k[j + 1][0] - k[j][0]);
        if (d0 * d1 <= 0) return 0;
        const m = (d0 + d1) / 2, lim = 3 * Math.min(Math.abs(d0), Math.abs(d1));
        return Math.sign(m) * Math.min(Math.abs(m), lim);
      };
      const herm = (s, x) => {
        const h = s.tb - s.ta; if (h <= 1e-9) return [s.xb, 0];
        const u = clamp((x - s.ta) / h, 0, 1), u2 = u * u, u3 = u2 * u;
        const v = (2 * u3 - 3 * u2 + 1) * s.xa + (u3 - 2 * u2 + u) * h * s.ma + (-2 * u3 + 3 * u2) * s.xb + (u3 - u2) * h * s.mb;
        const dv = ((6 * u2 - 6 * u) * s.xa + (3 * u2 - 4 * u + 1) * h * s.ma + (-6 * u2 + 6 * u) * s.xb + (3 * u2 - 2 * u) * h * s.mb) / h;
        return [v, dv];
      };
      const rangeClamp = (c, v) => { const R = RANGE[isFoot(c) ? c.slice(1) : c]; return (R && !KF_PERIODIC.test(c) && !isFoot(c)) ? clamp(v, R[0], R[1]) : v; };
      function addPose(p) {
        const touched = new Set();
        for (const [ctl, v] of Object.entries(p.set)) for (const c of KF_ALIAS[ctl] || KF_OBJ[ctl]) {
          if (isFoot(c) && !J.legs) continue;            // no legs: foot controls are ignored
          touched.add(c);
          if (!cols[c]) {   // first time listed: history = rest at every earlier pose (what was played); feet: flat from this value
            const keys = isFoot(c) ? [] : [[0, restOf(c)], ...P.map(q => [q.t, restOf(c)])];
            cols[c] = {keys, seg: null, listed: [], act: []};
          }
          const C = cols[c]; C.keys.push([p.t, rangeClamp(c, v)]); C.listed.push(p.t);
          if (isFoot(c)) {   // (re)activation of a world-foot axis: blend in over FOOT_IN before this pose (never before the current tau)
            const a = C.act[C.act.length - 1];
            if (!a || a.out0 !== null) { const in1 = p.t, in0 = Math.min(Math.max(p.t - KF.FOOT_IN, st.tau), in1 - 0.05); C.act.push({in0, in1, out0: null}); }
          }
        }
        for (const [c, C] of Object.entries(cols)) if (!touched.has(c) && C.keys.length) C.keys.push([p.t, C.after !== undefined && C.after < p.t && C.listed[C.listed.length - 1] <= C.after ? restOf(c) : C.keys[C.keys.length - 1][1]]);
        // extent for the camera zoom (exec.cjs: max |root_x|, |root_z|, 1.2 root_y, 0.5 if tipped, 2 |foot offset|)
        for (const [ctl, v] of Object.entries(p.set)) {
          if (ctl === 'root_x' || ctl === 'root_z') st.ext = Math.max(st.ext, Math.abs(v));
          else if (ctl === 'root_y') st.ext = Math.max(st.ext, 1.2 * Math.max(0, v));
          else if ((ctl === 'root_pitch' || ctl === 'root_roll') && Math.abs(v) > 0.06) st.ext = Math.max(st.ext, 0.5);
          else if (/_foot_/.test(ctl)) st.ext = Math.max(st.ext, 2 * Math.abs(v));
        }
        // world-foot auto release: an axis not listed for FOOT_GAP s of plan time goes back to automatic
        for (const [c, C] of Object.entries(cols)) if (isFoot(c)) {
          const a = C.act[C.act.length - 1], tl = C.listed[C.listed.length - 1];
          if (a && a.out0 === null && p.t > tl + KF.FOOT_GAP) a.out0 = Math.max(tl + KF.FOOT_GAP, st.tau);
        }
        P.push(p);
      }
      function finish() {   // the list is complete: every column eases back to rest after the last pose
        if (st.complete) return;
        st.complete = true;
        const tN = P.length ? P[P.length - 1].t : 0;
        for (const [c, C] of Object.entries(cols)) {
          if (isFoot(c)) { const a = C.act[C.act.length - 1]; if (a && a.out0 === null) a.out0 = Math.max(C.listed[C.listed.length - 1], st.tau); continue; }
          if (c === 'root_x' || c === 'root_z') continue;
          const last = C.keys[C.keys.length - 1], target = KF_PERIODIC.test(c) ? Math.round(last[1]) : restOf(c);
          if (Math.abs(last[1] - target) > 0.005) {
            const span = KF_PERIODIC.test(c) ? 4 * Math.abs(last[1] - target) : /^root_/.test(c) ? 2.5 * Math.abs(last[1] - target) : 0.6 * Math.abs(last[1] - target);
            C.keys.push([Math.max(last[0], tN) + clamp(0.35 + span, 0.4, 1.2), target]);
          }
        }
        st.tEnd = Math.max(tN, ...Object.entries(cols).map(([c, C]) => isFoot(c) ? (C.act.length ? C.act[C.act.length - 1].out0 + KF.FOOT_OUT : 0) : C.keys[C.keys.length - 1][0]));
      }
      function evalCol(C, x) {     // advance / re-plan the column's segment at plan time x, return [value, slope]
        const k = C.keys; if (!k.length) return [0, 0];
        if (x <= k[0][0] && !C.seg) return [k[0][1], 0];
        if (!C.seg) C.seg = {j: -1, ta: k[0][0], xa: k[0][1], ma: 0, tb: k[0][0], xb: k[0][1], mb: 0, prov: false};
        let s = C.seg;
        if (s.prov) {           // end tangent was provisional: the key after it may have landed -> re-plan the rest from the current state
          const m = tangent(k, s.j + 1);
          const t0 = Math.max(C.lastX ?? s.ta, s.ta);
          if (m !== null && m !== 0 && t0 < s.tb - 1e-3) {
            const [v0, d0] = herm(s, t0), d = (s.xb - v0) / (s.tb - t0);
            let mb = m; if (d * mb <= 0) mb = 0; else mb = Math.sign(mb) * Math.min(Math.abs(mb), 3 * Math.abs(d));
            C.seg = s = {j: s.j, ta: t0, xa: v0, ma: d0, tb: s.tb, xb: s.xb, mb, prov: false}; st.replans++;
          } else if (m !== null) s.prov = false;
        }
        while (x >= s.tb && s.j + 1 < k.length) {   // next segment: start tangent = the committed end tangent of the previous one
          const j = s.j + 1; if (j + 1 >= k.length) break;
          const m = tangent(k, j + 1), prov = m === null;
          C.seg = s = {j, ta: k[j][0], xa: k[j][1], ma: s.mb, tb: k[j + 1][0], xb: k[j + 1][1], mb: prov ? 0 : m, prov};
        }
        C.lastX = x;
        if (x >= s.tb) return [s.xb, 0];
        return herm(s, x);
      }
      const footW = (C, x) => Math.max(0, ...C.act.map(a => Math.min(ksstep((x - a.in0) / Math.max(1e-3, a.in1 - a.in0)), a.out0 === null ? 1 : 1 - ksstep((x - a.out0) / KF.FOOT_OUT))));
      const ready = () => P.length >= 1 && ((P.length >= KF.BUF_N && P[P.length - 1].t >= KF.BUF_H) || st.complete);
      function advance(dt, run) {   // one display frame: start rule, playback clock, every column at the clock
        st.Tknown = st.complete ? Infinity : (P.length ? P[P.length - 1].t : 0);
        let first = false;
        if (!st.started && run && ready()) { st.started = true; st.r = 0; first = true; }
        if (st.started && st.tau < st.tEnd + 0.35) {
          const rt = st.frozen || st.paused ? 0 : st.complete ? 1 : clamp((st.Tknown - st.tau) / KF.HEAD, 0, 1);
          st.r = rt < st.r ? rt : st.r + (rt - st.r) * (1 - Math.exp(-dt / KF.TAU_UP));
          if (first) st.r = rt;        // first frame of playback: start at the rate the buffer allows
          st.tau = Math.min(st.tau + st.r * dt, st.Tknown, st.stopAt);
          if (!st.frozen && !st.paused) {
            st.stall_s += (1 - st.r) * dt; st.minRate = Math.min(st.minRate, st.r);
            if (st.r < 0.5 && !st.inStall) { st.stalls++; st.inStall = true; } else if (st.r >= 0.75) st.inStall = false;
          }
        }
        const vals = {}, slopes = {}, wts = {};
        for (const [c, C] of Object.entries(cols)) { const [v, d] = evalCol(C, st.tau); vals[c] = v; slopes[c] = d; if (isFoot(c)) wts[c] = footW(C, st.tau); }
        return {vals, slopes, wts, r: st.started ? st.r : 0, started: st.started, ended: st.complete && st.tau >= st.tEnd,
          done: st.complete && (!P.length || (st.started && st.tau >= st.tEnd + 0.3))};
      }
      // a "do" step at plan time t is a fresh start — every column is at rest at t (its key there and the hold copies
      // after it up to its next listed key; later hold copies follow from addPose), the world feet go back to automatic
      function rebase(t) {
        for (const [c, C] of Object.entries(cols)) {
          if (isFoot(c)) { for (const a of C.act) if (a.out0 === null && a.in1 <= t + 1e-6) a.out0 = t - KF.FOOT_OUT; continue; }
          const nextListed = C.listed.find(x => x > t + 1e-6);
          for (const k of C.keys) if (k[0] >= t - 1e-6 && (nextListed === undefined || k[0] < nextListed - 1e-6)) k[1] = restOf(c);
          C.after = t; C.seg = null; C.lastX = undefined;
        }
      }
      return {P, cols, st, addPose, finish, advance, ready, rebase};
    }
    // a pose as the planner wrote it ({t, ctl: v} | [t, {ctl: v}] | {t, set: {...}}) -> {t (plan s, shifted), set} or null
    // (parseStream's take(): unknown controls dropped, an early first key shifts the whole program 0.3 s so it eases in from
    // rest, times strictly increasing, the last pose at most PLAN_MAX)
    function kfNorm(J, raw) {
      let o = raw;
      if (typeof o === 'string') { try { o = JSON.parse(o); } catch (e) { J.dropped++; return null; } }
      let t, set;
      if (Array.isArray(o)) { t = +o[0]; set = o[1]; }
      else if (o && typeof o === 'object') { t = +(o.t ?? o.time); set = o.set && typeof o.set === 'object' ? o.set : Object.fromEntries(Object.entries(o).filter(([k]) => k !== 't' && k !== 'time' && k !== 'ms')); }
      if (!Number.isFinite(t) || !set || typeof set !== 'object') { J.dropped++; return null; }
      const clean = {}, ev = [];
      let step = null;
      for (const [k, v] of Object.entries(set)) {
        if (k === 'do') { step = kfNormDo(v); if (!step && J.unknown.length < 20) J.unknown.push('do'); continue; }
        if (!KF_ALIAS[k] && !KF_OBJ[k] && !KF_EV[k]) { if (J.unknown.length < 20 && !J.unknown.includes(k)) J.unknown.push(k); continue; }
        const x = typeof v === 'number' ? v : (v !== null && v !== '' && v !== true && v !== false ? +v : NaN);
        if (!Number.isFinite(x)) continue;
        if (KF_EV[k]) { const e = KF_EV[k]; if (e.toss ? x > 0 : x >= 0.5) ev.push(Object.assign({key: k}, e, e.toss ? {height: Math.min(3, x)} : {})); continue; }
        clean[k] = x;
      }
      if (J.shift === null) J.shift = (t < 0.2 && (Object.keys(clean).length || ev.length || step)) ? 0.3 : 0;
      t = Math.max(0, t) + J.shift;
      if (t <= J.lastT + 1e-6 || t > KF.PLAN_MAX + J.shift + 1e-6) { J.dropped++; return null; }
      J.lastT = t;
      const p = {t, set: clean};
      if (ev.length) p.ev = ev;
      if (step) p.do = step;
      return p;
    }
    // a pose's "do" -> the object step {verb, object, ...} (the object tool's shape) or null; "pick_up ball" /
    // "put ball on block_a" strings read as verb + object [+ relation + target] (planner.py norm_do, same rules)
    const DO_REL = ['next to', 'onto', 'into', 'on top of', 'under', 'behind', 'in front of', 'beside', 'on', 'in', 'at', 'to'];
    const DO_VERB_ALIAS = {put: 'place', set: 'place', take: 'pick_up', grab: 'pick_up', wear: 'put_on'};
    function kfNormDo(v) {
      if (typeof v === 'string') {
        const w = v.trim().toLowerCase().replace(/-/g, '_').replace(/^(pick|put|take|hand|turn|sit|stand|work|type) (up|down|back|on|off|over|in|at)\b/, '$1_$2')
          .replace(/^(go|walk|knock|rap|lean|tap|pat|press) (over to|up to|to|on|against)\b/, (m, a, b) => a + '_' + b.replace(/ /g, '_')); if (!w) return null;   // go to / knock on (planner.py norm_do)
        const i = w.indexOf(' '), o = {verb: i < 0 ? w : w.slice(0, i)}, rest = i < 0 ? '' : w.slice(i + 1).trim();
        if (rest) {
          let done = false;
          for (const rel of DO_REL) { const j = (' ' + rest + ' ').indexOf(' ' + rel + ' '); if (j > 0) { const a = rest.slice(0, j - 1).trim(), b = rest.slice(j + rel.length).trim(); if (a && b) { Object.assign(o, {object: a, relation: rel.replace(/ /g, '_'), target: b}); done = true; break; } } }
          if (!done) o.object = rest;
        }
        v = o;
      }
      if (!v || typeof v !== 'object' || Array.isArray(v)) return null;
      const verb = typeof v.verb === 'string' ? v.verb : typeof v.action === 'string' ? v.action : null;
      if (!verb || !/^[a-z_ ]{2,30}$/.test(verb.trim().toLowerCase())) return null;
      let V = verb.trim().toLowerCase().replace(/ /g, '_'); V = DO_VERB_ALIAS[V] || V;
      const out = {verb: V};
      for (const k of ['object', 'target', 'relation', 'place', 'side', 'to', 'count', 'amount', 'dir', 'speed', 'height', 'attach', 'anchor', 'style']) {
        const x = v[k];
        if (x === undefined || x === null || x === '' || typeof x === 'boolean') continue;
        if (typeof x === 'number') { if (Number.isFinite(x)) out[k] = x; }
        else if (typeof x === 'string') out[k] = x.trim().slice(0, 60);
        else if (typeof x === 'object' && !Array.isArray(x)) { const d = {}; for (const kk of ['id', 'name', 'color', 'ref', 'place']) if (typeof x[kk] === 'string' || typeof x[kk] === 'number') d[kk] = String(x[kk]).slice(0, 40); if (Object.keys(d).length) out[k] = d; }
      }
      return out;
    }
    const kfPlanned = J => J.pl.P.length ? J.pl.P[J.pl.P.length - 1].t : 0;
    const kfArgs = J => ({action: 'keyframes', description: J.description});
    function kfSummary(J) {
      if (!J) return null;
      return {id: J.id, description: J.description, phase: J.phase, interrupted: J.term === 'interrupted', reason: J.reason || undefined,
        poses: J.pl.P.length, planned_s: +kfPlanned(J).toFixed(2), played_s: +J.pl.st.tau.toFixed(2), complete: J.pl.st.complete,
        released: J.released, w: +J.w.toFixed(3), stalls: J.pl.st.stalls, dropped: J.dropped, unknown: J.unknown.slice(),
        steps: J.steps.slice(), doing: J.doing ? J.doing.d.verb : null, paw_events: J.events.slice()};
    }
    function kfNew(cmd, src) {
      const id = cmd.id !== undefined && cmd.id !== null && cmd.id !== '' ? String(cmd.id) : 'kf' + (++S.cmdN);
      const holdIt = cmd.hold === true || cmd.hold === 'true';
      const legs = !!(engineOn && engine && engine.E && engine.E.body && engine.E.body.legs);
      // rest snapshot = this frame's output BEFORE any keyframe overlay (exec_stream.cjs: the idle row before the command)
      const snap = Object.assign({}, S.kfBase || S.lastOut || {});
      if (!S.kfBase && S.thinkD) for (const k in S.thinkD) if (k in snap) snap[k] -= S.thinkD[k];   // rest without the thinking
      const J = {id, description: String(cmd.description || cmd.name || 'the move he asked for').replace(/\s+/g, ' ').trim().slice(0, 200) || 'the move he asked for',
        src, hold: holdIt, released: !holdIt, legs, phase: 'buffering', term: null, reason: null, t0: S.t, readyAt: null, startT: null, endT: null,
        rest: snap, shift: null, lastT: -1, dropped: 0, unknown: [], w: 0, wIn: null, wOut: null, fr: null, T: null, wc: null,
        ext: 0, ride: {l: 0, r: 0}, base: {x: 0, y: 0, z: 0, yaw: 0}, seated: false,
        evI: 0, doing: null, doW: null, steps: [], events: []};   // next pose to fire, the running "do" step, its weight
      J.restOf = c => c.charCodeAt(0) === 64 || KF_ROOTCOL.test(c) ? 0 : (Number.isFinite(J.rest[c]) ? J.rest[c] : 0);
      J.pl = kfPlayer(J);
      return J;
    }
    // the planned move owns the channels its poses write. A pose HELD on one of them (a matcher's "hold" pose:
    // head up, eyes closed, paws over the eyes) is let go when the move starts, and the move's rest for that channel is the
    // live unheld value — it used to start from the held pose and ease back INTO it, so the hold outlived every later request
    // for that part ("lower your head" played and returned to head-up; eyes stayed shut; "back to normal" changed nothing).
    function kfOwn(J, p) {
      for (const ctl of Object.keys(p.set)) for (const c of KF_ALIAS[ctl] || []) {
        if (c.charCodeAt(0) === 64 || !heldBy(c) || (J.freed && J.freed.has(c))) continue;
        (J.freed || (J.freed = new Set())).add(c);
        if (S.react && Number.isFinite(S.react[c])) J.rest[c] = S.react[c];
        if (J.phase === 'playing') releaseHold(c, 0.3);
      }
    }
    function kfCheckReady(J) {
      if (J.term || J.phase !== 'buffering' || !J.pl.ready()) return;
      J.phase = 'ready'; J.readyAt = S.t;
      emit({kind: 'kf', name: 'perform', id: J.id, src: J.src, phase: 'ready', args: kfArgs(J), poses: J.pl.P.length, planned_s: +kfPlanned(J).toFixed(2),
        complete: J.pl.st.complete, did: `ready to start: ${J.description} (${kfPlanned(J).toFixed(1)} s planned so far)`});
    }
    function kfFeedJ(J, poses, done, info) {
      if (J.term) return {ok: false, id: J.id, reason: 'that move already ended', phase: J.phase};
      let n = 0;
      for (const raw of Array.isArray(poses) ? poses : poses ? [poses] : []) {
        const p = kfNorm(J, raw);
        if (p) { kfOwn(J, p); J.pl.addPose(p); n++; if (!J.objMode && (p.do || p.ev || Object.keys(p.set).some(k => KF_OBJ[k]))) J.objMode = S.t; }
      }
      if (done) {
        J.pl.finish();
        if (!J.pl.P.length) {
          const why = info && info.error ? String(info.error).slice(0, 120) : 'the planner returned no usable poses';
          kfEnd(J, 'fail', why); return {ok: false, id: J.id, reason: why, phase: J.phase};
        }
      }
      kfCheckReady(J);
      return {ok: true, id: J.id, added: n, poses: J.pl.P.length, phase: J.phase};
    }
    function kfCommand(cmd, src) {
      const id0 = cmd.id !== undefined && cmd.id !== null && cmd.id !== '' ? String(cmd.id) : null;
      const ex = id0 !== null ? S.kf.find(J => J.id === id0 && !J.term) : null;
      if (ex) {   // the same job again: more poses (and / or the hold flag)
        if (cmd.hold === false || cmd.hold === 'false') ex.released = true;
        const r = kfFeedJ(ex, cmd.poses, cmd.done === true || cmd.done === 'true', cmd);
        return r.ok ? {ok: true, id: ex.id, did: `getting ready to ${ex.description}`, phase: ex.phase} : {ok: false, id: ex.id, reason: r.reason};
      }
      // a newer request replaces one still being written / waiting; one already playing goes on until this one starts
      for (const J of S.kf) if (!J.term && J.phase !== 'playing') kfEnd(J, 'interrupted', 'replaced by a newer request');
      const J = kfNew(cmd, src);
      S.kf.push(J);
      S.idleT = 0; S.bored = Math.max(0, S.bored - 0.4);   // a command is interaction (as command0)
      S.log.push({t: +S.t.toFixed(2), cmd: 'keyframes', name: J.description, id: J.id});
      const r = kfFeedJ(J, cmd.poses, cmd.done === true || cmd.done === 'true', cmd);
      if (!r.ok) return {ok: false, id: J.id, reason: r.reason};
      return {ok: true, id: J.id, did: `getting ready to ${J.description}`, phase: J.phase};
    }
    function kfStart(J) {   // the first frame of playback
      J.phase = 'playing'; J.wIn = null; J.startT = S.t;
      if (J.freed) { for (const c of J.freed) releaseHold(c, 0.3); S.log.push({t: +S.t.toFixed(2), release: [...J.freed], why: 'a planned move drives these'}); }
      thinkEnd('planned move started', 0.4);   // the thinking hands over to the planned move (min-jerk fade under it)
      for (const K of S.kf) if (K !== J && !K.term) kfEnd(K, 'interrupted', 'replaced by a newer request');
      const lo = S.lastOut || {};
      if (J.legs && engineOn && engine) {
        const E = engine.E;
        // a whole-body job: a running / queued engine move and the laptop activity end gracefully ("relax": what it holds and
        // what it wears stay); the keyframes play from where it stands
        // a whole-body job: the engine's running / queued moves, object work and the laptop activity stop gracefully (the
        // engine's 'stop': what it holds and what it wears stay). The keyframes start from rest on every direct column (no jump
        // there); only the root / feet / camera can differ from what the engine shows now: then the job blends in over W_IN
        if ((E.move && !E.move.silent) || E.queue || E.oqueue || E.chain || E.activity || (E.jobs && E.jobs.length)) { engine.stopAll('replaced by a planned move'); J.wIn = S.t; }
        else if (E.move || E.sitOn || (+E.w || 0) > 0.02 || Math.abs(+E.cur.y || 0) > 0.01 || Math.abs(kwrap(+E.cur.pitch || 0)) > 0.01 || Math.abs(kwrap(+E.cur.roll || 0)) > 0.01) J.wIn = S.t;
        const c = E.cur || {};
        J.base = {x: +c.x || 0, y: E.sitOn ? Math.max(0, +c.y || 0) : 0, z: +c.z || 0, yaw: kwrapH(+c.yaw || 0)};
        J.seated = J.base.y > 0.05;   // on a chair: the legs ride along with the body
      }
      // a listener / the learned reaction shows other values than the rules frame the job composes on: blend in
      if (listener || learned) J.wIn = S.t;
      J.cz = {x: +lo.cam_zoom || 0, v: 0}; J.cx = {x: +lo.cam_x || 0, v: 0}; J.cy = {x: +lo.cam_y || 0, v: 0}; J.gf = {x: +lo.gaze_free || 0, v: 0};
      emit({kind: 'kf', name: 'perform', id: J.id, src: J.src, phase: 'start', args: kfArgs(J), poses: J.pl.P.length, planned_s: +kfPlanned(J).toFixed(2),
        complete: J.pl.st.complete, did: `doing the move he asked for: ${J.description}`});
    }
    function kfEnd(J, phase, reason) {   // phase 'end' | 'interrupted' | 'fail'; the weight eases out from here (blend back)
      if (J.term) return;
      const was = J.phase, played = J.pl.st.tau, planned = kfPlanned(J);
      if (J.doing) {   // cut short while a "do" step ran (the engine's own job goes on unless it is stopped too)
        const d = J.doing.d; J.steps.push({t: +J.doing.planT.toFixed(2), verb: d.verb, what: d.verb.replace(/_/g, ' '), ok: false, reason: 'cut short: ' + (reason || phase)}); J.doing = null;
      }
      J.term = phase; J.reason = reason || null; J.phase = phase === 'fail' ? 'failed' : 'ended'; J.endT = S.t;
      J.wOut = {t0: S.t, from: J.w};
      // the floor position stays where the keyframes left it (exec.cjs: root_x / root_z stay): hand it to the engine
      if (was === 'playing' && J.legs && engineOn && engine && J.T) {
        const E = engine.E, x = J.T.root_x, z = J.T.root_z;
        if (!E.move && Number.isFinite(x) && Number.isFinite(z)) { E.base.x = x; E.base.z = z; if (E.cur) { E.cur.x = x; E.cur.z = z; } E.feet = null; }
        if (E.cam && J.cz) Object.assign(E.cam, {zoom: J.cz.x, zv: J.cz.v, x: J.cx.x, xv: J.cx.v, y: J.cy.x, yv: J.cy.v});
      }
      // the object steps' outcomes, truthfully (a failed step did not stop the plan)
      const bad = J.steps.filter(x => !x.ok), good = J.steps.filter(x => x.ok);
      const partial = bad.length ? bad.map(x => `${x.what} did not work (${x.reason})`).join('; ') : '';
      const stepsDid = J.steps.length ? ` [object steps: ${good.length ? good.map(x => x.what).join(', ') + ' done' : ''}${good.length && bad.length ? '; ' : ''}${partial}]` : '';
      const did = phase === 'end' ? `did the move he asked for: ${J.description} (${planned.toFixed(1)} s)${stepsDid}`
        : phase === 'fail' ? `could not do the move he asked for: ${J.description}`
        : was === 'playing' ? `the first ${played.toFixed(1)} of its ${planned.toFixed(1)} s (${J.description}), then it was cut short`
        : `did not start the move he asked for (${J.description})`;
      emit(Object.assign({kind: 'kf', name: 'perform', id: J.id, src: J.src, phase, args: kfArgs(J), did, reason: phase === 'end' ? undefined : (reason || undefined),
        poses: J.pl.P.length, played_s: +played.toFixed(2), planned_s: +planned.toFixed(2)}, J.steps.length ? {steps: J.steps.slice(), partial: partial || undefined} : {},
        J.events.length ? {paw_events: J.events.slice()} : {}));
      S.kfHist.set(J.id, kfSummary(J)); if (S.kfHist.size > 40) S.kfHist.delete(S.kfHist.keys().next().value);
    }
    function kfInterruptAll(reason) { let n = 0; for (const J of S.kf) if (!J.term) { kfEnd(J, 'interrupted', reason); n++; } return n; }
    // the composed target value of every channel the job drives this frame
    function kfComposeLegs(J, o, fr, dt) {
      const V = fr.vals, T = {}, E = engine.E, body = E.body, Cc = body.C, legs = body.legs, rotm = MovesLib.rotm;
      const val = c => c in V ? V[c] : 0;
      // direct columns: additive on this frame's output (pose − rest), clamped to RANGE
      const aw = J.absW || 0;   // a plan WITH things owns its paws outright (the engine's carry pose may change under it)
      for (const c in V) {
        if (c.charCodeAt(0) === 64 || KF_ROOTCOL.test(c) || !(c in o)) continue;
        const r = RANGE[c] || [-1, 1], add = o[c] + (V[c] - J.restOf(c));
        T[c] = clamp(aw > 0 && KF_PAWCOL.test(c) ? add + (V[c] - add) * aw : add, r[0], r[1]);
      }
      // root: the program's height is above floor contact; the lift keeps a tipped body on the floor
      const B = J.base, yaw = B.yaw + val('root_yaw'), pitch = val('root_pitch'), roll = val('root_roll'), Rm = rotm(yaw, pitch, roll);
      const yPlan = val('root_y'), root = [B.x + val('root_x'), B.y + yPlan + body.lift(Rm), B.z + val('root_z')];
      T.root_x = clamp(root[0], -3, 3); T.root_y = clamp(root[1], -0.6, 3); T.root_z = clamp(root[2], -3, 1.5);
      T.root_yaw = kwrapH(yaw); T.root_pitch = kwrapH(pitch); T.root_roll = kwrapH(roll); T.squash = clamp(val('squash'), -1, 1);
      // feet: WORLD targets (listed axes, blended in / out) or automatic (planted under the body / riding along when airborne or tipped)
      const tipped = J.seated || yPlan > 0.015 || Math.abs(kwrap(pitch)) > 0.02 || Math.abs(kwrap(roll)) > 0.02;
      const Ry = rotm(yaw, 0, 0), bOff = [B.x, 0, B.z];
      for (const s of ['l', 'r']) {
        J.ride[s] += ((tipped ? 1 : 0) - J.ride[s]) * (1 - Math.exp(-dt / (tipped ? 0.03 : 0.05)));
        const f0 = legs.foot[s], d = f0.map((v, j) => v - Cc[j]);
        const restW = kmv(Rm, d).map((v, j) => v + Cc[j] + root[j]);
        const gnd = kmv(Ry, d).map((v, j) => v + Cc[j] + root[j]); gnd[1] = f0[1];
        const auto = gnd.map((v, j) => v + (restW[j] - v) * J.ride[s]);
        ['x', 'y', 'z'].forEach((a, j) => {
          const key = '@foot_' + s + '_' + a;
          const Fw = key in V ? auto[j] + (fr.wts[key] || 0) * (f0[j] + bOff[j] + V[key] - auto[j]) : auto[j];
          const rr = RANGE['foot_' + s + '_' + a];
          T['foot_' + s + '_' + a] = clamp(Fw - restW[j], rr[0], rr[1]);
        });
      }
      // camera: zoom target from the extent of the poses parsed so far, active from playback start to the end of the motion
      J.ext = Math.max(J.ext, J.pl.st.ext);
      const zBig = clamp(0.3 + 0.8 * J.ext, 0.3, 0.85), active = fr.started && !fr.ended && !J.term;
      T.cam_zoom = clamp(ksp2(J.cz, active ? zBig : (+o.cam_zoom || 0), 5, dt), -1, 1);   // scene: < 0 = a station view
      T.cam_x = ksp2(J.cx, 0.75 * root[0], 6, dt);
      T.cam_y = ksp2(J.cy, 0.75 * Math.max(0, root[1] - 0.04), 7, dt);
      // scene: a station view (engine.setView) holds the camera during a planned move too (no zoom-out pop at the board)
      if (engineOn && engine && engine.E && engine.E.viewCam && engine.E.viewW > 0) {
        const Vc = engine.E.viewCam, k = clamp(engine.E.viewW, 0, 1);
        T.cam_zoom = clamp(T.cam_zoom + (Vc.zoom - T.cam_zoom) * k, -1, 1); T.cam_x += (Vc.x - T.cam_x) * k; T.cam_y += (Vc.y - T.cam_y) * k;
      }
      const away = Math.abs(kwrap(yaw)) > 0.12 || Math.abs(kwrap(pitch)) > 0.1 || Math.abs(kwrap(roll)) > 0.1 || ('root_yaw' in V && Math.abs((fr.slopes.root_yaw || 0) * fr.r * 0.1) > 0.03);
      T.gaze_free = Math.max(clamp(ksp2(J.gf, away ? 1 : 0, 10, dt), 0, 1), +o.gaze_free || 0);
      return T;
    }
    function kfComposeSeated(J, o, fr) {
      const V = fr.vals, T = {};
      const aw = J.absW || 0;   // a plan WITH things owns its paws outright (the engine's carry pose may change under it)
      for (const c in V) {
        if (c.charCodeAt(0) === 64 || KF_ROOTCOL.test(c) || !(c in o)) continue;
        const r = RANGE[c] || [-1, 1], add = o[c] + (V[c] - J.restOf(c));
        T[c] = clamp(aw > 0 && KF_PAWCOL.test(c) ? add + (V[c] - add) * aw : add, r[0], r[1]);
      }
      // no root in its renderer: the vertical / turning / tipping intent on the body channels it does render (the seated
      // gesture approximations' channels: jump = body_bounce, spin = a body_yaw swivel, bow = body_pitch); root_x / z are ignored
      const add = (c, d) => { const r = RANGE[c]; T[c] = clamp((c in T ? T[c] : o[c]) + d, r[0], r[1]); };
      if ('root_y' in V || 'squash' in V) add('body_bounce', clamp(2 * (V.root_y || 0) - 2 * (V.squash || 0), -1, 1));
      if ('root_yaw' in V) add('body_yaw', kswivel(V.root_yaw));
      if ('root_pitch' in V) add('body_pitch', clamp(2.5 * kswivel(V.root_pitch), -1, 1));
      if ('root_roll' in V) add('body_lean', clamp(2.5 * kswivel(V.root_roll), -1, 1));
      return T;
    }
    function kfBlend(J, o, T) {
      const wc = {}, sv = S.kfSv;
      for (const c in T) {
        const w = J.w * (SPEECH_SET.has(c) ? 1 - sv : 1); wc[c] = w;
        if (!(w > 0)) continue;
        if (KF_PERIODIC.test(c)) o[c] = w >= 1 ? T[c] : kwrapH((+o[c] || 0) + kwrap(T[c] - (+o[c] || 0)) * w);
        else o[c] = w >= 1 ? T[c] : lerp(o[c], T[c], w);
      }
      J.T = T; J.wc = wc;
    }
    // every frame, after the rules / engine / gestures / clamps: advance the jobs and lay their channels over the frame
    function kfFrame(o, dt) {
      const spk = S.av.speaking ? 1 : 0;
      S.kfSv = lp(S.kfSv || 0, spk, dt, spk ? 0.05 : 0.15);
      // the frame before any keyframe overlay and before the listener (a new job's rest snapshot)
      S.kfBase = S.kf.length || listener ? Object.assign({}, o) : null;
      if (S.kfBase && S.thinkD) for (const k in S.thinkD) if (k in S.kfBase) S.kfBase[k] -= S.thinkD[k];   // rest without the thinking
      if (!S.kf.length) { S.kfPaw.l = S.kfPaw.r = 0; if (S.kfHeldOn.l || S.kfHeldOn.r) kfHeldOut(); return; }
      for (const J of S.kf.slice()) {
        if (!J.term) {
          if (J.phase === 'buffering' && !J.pl.st.complete && S.t - J.t0 > KF.TIMEOUT) kfEnd(J, 'fail', 'the planner sent no usable poses in time');
          else if (J.phase === 'ready' && !J.released && S.t - J.readyAt > KF.AUTO_RELEASE) J.released = true;
        }
        if (!J.term) {
          if (J.doing) kfDoCheck(J, o);                     // a "do" step running / just ended (resume here)
          J.pl.st.stopAt = kfNextDoT(J);
          const was = J.pl.st.started, fr = J.pl.advance(dt, J.released);
          if (!was && fr.started) kfStart(J);
          J.fr = fr;
          if (fr.started && !J.term) kfFire(J);             // paw events / do steps the clock has reached
          if (fr.done && J.phase === 'playing' && !J.doing) kfEnd(J, 'end');
        }
        if (J.objMode !== undefined && J.objMode !== false) J.absW = J.pl.st.started ? Math.min(1, (J.absW || 0) + dt / 0.3) : 1;
        if (J.phase === 'playing') J.w = (J.wIn === null ? 1 : mj((S.t - J.wIn) / KF.W_IN)) * kfDoFactor(J);
        else if (J.wOut) J.w = J.wOut.from * (1 - mj((S.t - J.wOut.t0) / KF.W_OUT));
        else J.w = 0;
        if (J.fr && J.fr.started && J.w > 0) kfBlend(J, o, J.legs && engineOn && engine ? kfComposeLegs(J, o, J.fr, dt) : kfComposeSeated(J, o, J.fr));
        else { J.T = null; J.wc = null; }
        if (J.term && !(J.w > 1e-4)) { S.kfHist.set(J.id, kfSummary(J)); S.kf.splice(S.kf.indexOf(J), 1); }
      }
      kfHeldOut();                                          // what the paws hold follows the held controls
      // the paw weight the prosody beats yield to (next frame)
      S.kfPaw.l = S.kfPaw.r = 0;
      for (const J of S.kf) if (J.wc) for (const c in J.wc) { const m = /^(hand|thumb|point|fingers)_([lr])(_|$)/.exec(c); if (m) S.kfPaw[m[2]] = Math.max(S.kfPaw[m[2]], J.wc[c]); }
    }
    // ---------------------------------------------------------------- things in the keyframe plan (docs/OBJECTS.md §6)
    // Paw events fire once when the playback clock crosses their pose (engine.kfEvent). A "do" step: the clock stops at its
    // pose (stopAt), the job's weight eases out (KF.DO_OUT) so the engine's own motion shows, the engine runs the object
    // command (engineCommand: never m.command, which would end this very job), and when the engine's job ends — or fails:
    // then the plan goes on and the failure is reported in the end event — the plan resumes from where the body now is
    // (rest snapshot + root base re-taken, every column restarts from rest at the step: kfPlayer.rebase) and the weight
    // eases back in (KF.DO_IN). Held controls go to engine.kfHeld(side, {x, y, z, rx, ry, rz}) every frame, scaled by the
    // job's weight. A missing engine method degrades: the event / step is recorded as not supported, the plan plays on.
    S.kfHeldOn = {l: false, r: false};
    const kfDoFactor = J => { const d = J.doW; if (!d) return 1; const u = (S.t - d.t0) / (d.mode === 'out' ? KF.DO_OUT : KF.DO_IN);
      return d.mode === 'out' ? d.from * (1 - mj(u)) : d.from + (1 - d.from) * mj(u); };
    function kfNextDoT(J) { const P = J.pl.P; for (let i = J.evI; i < P.length; i++) if (P[i].do) return P[i].t; return Infinity; }
    function kfFire(J) {
      const P = J.pl.P, tau = J.pl.st.tau;
      while (!J.doing && J.evI < P.length && P[J.evI].t <= tau + 1e-6) {
        const p = P[J.evI++];
        for (const e of p.ev || []) kfPawEvent(J, e, p.t);
        if (p.do) kfDoStart(J, p);
      }
    }
    const sideWordOf = s => (s === 'l' ? 'right' : 'left');          // engine letter -> the character's own side
    function kfPawEvent(J, e, t) {
      const ev = {type: e.type, side: e.side};
      // a toss straight up h: the release velocity is exactly (0, sqrt(2Gh), 0) — the impulse cancels the paw's own measured
      // velocity (the held prop's wvel, when the engine keeps one), so the planner's airtime sqrt(8h/G) holds
      if (e.toss) {
        ev.height = e.height; const vy = Math.sqrt(2 * KF.G * Math.max(0.01, e.height));
        let wv = [0, 0, 0];
        try { const q = engineOn && engine.E && engine.E.P ? engine.E.P.find(x => x.held === e.side) : null; if (q && Array.isArray(q.wvel) && q.wvel.every(Number.isFinite)) wv = q.wvel; } catch (err) {}
        ev.impulse = [-wv[0], vy - wv[1], -wv[2]];   // world H/s (engine.kfEvent, contract §5)
      }
      if (e.type === 'catch') ev.window = KF.CATCH_WIN;
      let r = null;
      if (engineOn && engine && typeof engine.kfEvent === 'function') { try { r = engine.kfEvent(ev); } catch (err) { r = {ok: false, reason: 'error: ' + String(err && err.message || err).slice(0, 60)}; } }
      else r = {ok: false, reason: engineOn ? 'paw events are not supported here' : 'there are no things here'};
      const rec = {t: +t.toFixed(2), key: e.key, ok: !r || r.ok !== false, reason: r && r.ok === false ? String(r.reason || '') : undefined};
      J.events.push(rec); if (J.events.length > 40) J.events.splice(0, 20);
      S.log.push({t: +S.t.toFixed(2), kf_event: e.key, ok: rec.ok});
    }
    function kfDoCmd(d) {
      const cmd = {action: 'object', verb: d.verb};
      for (const k in d) if (k !== 'verb' && k !== 'object') cmd[k] = d[k];
      const ob = d.object;
      if (typeof ob === 'string') {
        const w = ob.trim().toLowerCase().replace(/^the /, '');
        const ids = engineOn ? engine.props().map(p => p.id) : [];
        cmd.object = /^(all|everything|them all|all of them)$/.test(w) ? {ref: 'all'} : /^(it|that|this)$/.test(w) ? {ref: 'it'}
          : ids.includes(w.replace(/ /g, '_')) ? {id: w.replace(/ /g, '_')} : {name: w};
      } else if (ob && typeof ob === 'object') cmd.object = ob;
      return cmd;
    }
    function kfDoStart(J, p) {
      const d = p.do;
      J.doing = {d, t: S.t, planT: p.t, id: null, early: [], result: null};
      J.pl.st.paused = true;
      J.doW = {mode: 'out', t0: S.t, from: kfDoFactor(J)};
      if (d.verb === 'attach') {          // the held thing onto the head / a foot / a thing — the engine's attach event, no job
        const side = /left/.test(String(d.side || '')) ? 'r' : 'l', to0 = String(d.to || d.target || d.place || 'head').toLowerCase().replace(/ /g, '_');
        const to = to0 === 'right_foot' ? 'foot_l' : to0 === 'left_foot' ? 'foot_r' : to0;
        let r;
        if (engineOn && engine && typeof engine.kfEvent === 'function') { try { r = engine.kfEvent({type: 'attach', side, to}); } catch (err) { r = {ok: false, reason: 'error'}; } }
        else r = {ok: false, reason: engineOn ? 'attaching is not supported here' : 'there are no things here'};
        J.doing.result = {ok: !r || r.ok !== false, reason: r && r.ok === false ? String(r.reason || 'it could not attach it') : null, did: r && r.did};
        return;
      }
      // hand the floor position to the engine (as kfEnd): the engine's step starts where the body stands now
      if (J.legs && engineOn && engine && J.T) {
        const E = engine.E, x = J.T.root_x, z = J.T.root_z;
        if (!E.move && Number.isFinite(x) && Number.isFinite(z)) { E.base.x = x; E.base.z = z; if (E.cur) { E.cur.x = x; E.cur.z = z; } E.feet = null; }
      }
      let r;
      try { r = engineCommand(kfDoCmd(d), J.src || 'tool'); } catch (err) { r = {ok: false, reason: 'error: ' + String(err && err.message || err).slice(0, 80)}; }
      if (!r || r.ok === false) { J.doing.result = {ok: false, reason: String((r && r.reason) || 'it could not be done')}; return; }
      J.doing.id = r.id;
      const early = J.doing.early.find(e => String(e.id) === String(r.id) && /^(end|fail|failed|interrupted)$/.test(e.phase));
      if (early) J.doing.result = {ok: early.phase === 'end', reason: early.phase === 'end' ? null : String(early.reason || early.phase), did: early.did};
      else if (r.already && !r.queued) J.doing.result = {ok: true, reason: null, did: r.did};
    }
    // engine events (from emit): the end of a running "do" step's job
    function kfEngineEvent(ev) {
      if (!ev || ev.id == null || (ev.kind !== 'object' && ev.kind !== 'move')) return;
      for (const J of S.kf) {
        const D = J.doing; if (!D || D.result) continue;
        if (D.id === null) { D.early.push(ev); continue; }
        if (String(D.id) !== String(ev.id)) continue;
        if (ev.phase === 'end') D.result = {ok: true, reason: null, did: ev.did};
        else if (/^(fail|failed|interrupted)$/.test(ev.phase)) D.result = {ok: false, reason: String(ev.reason || (ev.phase === 'interrupted' ? 'it was stopped' : 'it did not work')), did: ev.did};
      }
    }
    function kfDoCheck(J, o) {
      const D = J.doing;
      if (!D.result && S.t - D.t > KF.DO_MAX) D.result = {ok: false, reason: 'it took too long'};
      if (!D.result) return;
      if (D.result.ok === false && D.id !== null && engineOn && engine && engine.E && engine.E.jobs && engine.E.jobs.some(j => String(j.id) === String(D.id))) engine.stopAll('the planned move goes on');
      const d = D.d, what = d.verb.replace(/_/g, ' ') + (d.object ? ' ' + (typeof d.object === 'string' ? d.object : d.object.name || d.object.id || d.object.ref || '') : '');
      J.steps.push({t: +D.planT.toFixed(2), verb: d.verb, object: typeof d.object === 'string' ? d.object : d.object ? (d.object.id || d.object.name || d.object.ref) : undefined,
        what: what.trim(), ok: D.result.ok !== false, reason: D.result.ok === false ? D.result.reason : undefined, did: D.result.did || undefined, s: +(S.t - D.t).toFixed(2)});
      S.log.push({t: +S.t.toFixed(2), kf_do: d.verb, ok: D.result.ok !== false, reason: D.result.reason || undefined});
      // fresh start from where the body is now: the rest snapshot (this frame before any keyframe overlay), the root base
      const snap = Object.assign({}, S.kfBase || o);
      if (S.thinkD) for (const k in S.thinkD) if (k in snap) snap[k] -= S.thinkD[k];
      J.rest = snap;
      if (J.legs && engineOn && engine) { const c = engine.E.cur || {}; J.base = {x: +c.x || 0, y: engine.E.sitOn ? Math.max(0, +c.y || 0) : 0, z: +c.z || 0, yaw: kwrapH(+c.yaw || 0)}; J.seated = J.base.y > 0.05; }
      J.pl.rebase(D.planT);
      J.doing = null; J.pl.st.paused = false;
      J.doW = {mode: 'in', t0: S.t, from: kfDoFactor(J)};
    }
    function kfHeldOut() {
      const acc = {l: null, r: null};
      for (const J of S.kf) {
        if (!J.fr || !J.fr.started || !(J.w > 0)) continue;
        const V = J.fr.vals;
        for (const sd of ['l', 'r']) {
          let any = false; const v = {x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0};
          for (const [a, k, turn] of [['x', 'x'], ['y', 'y'], ['z', 'z'], ['tilt', 'rx', 1], ['spin', 'ry', 1], ['roll', 'rz', 1]]) {
            const c = '%held_' + sd + '_' + a; if (!(c in V)) continue;
            any = true; v[k] = (turn ? kwrap(V[c]) : clamp(V[c], -KF.HELD_XYZ, KF.HELD_XYZ)) * J.w;
          }
          if (any) acc[sd] = acc[sd] ? Object.fromEntries(Object.keys(v).map(k => [k, acc[sd][k] + v[k]])) : v;
        }
      }
      const can = engineOn && engine && typeof engine.kfHeld === 'function';
      for (const sd of ['l', 'r']) {
        if (acc[sd]) { if (can) { try { engine.kfHeld(sd, acc[sd]); } catch (e) {} } S.kfHeldOn[sd] = true; }
        else if (S.kfHeldOn[sd]) { if (can) { try { engine.kfHeld(sd, null); } catch (e) {} } S.kfHeldOn[sd] = false; }
      }
      S.kfHeldNow = acc;
    }
    // the live things list a plan WITH things is written from (page -> server state.things): the engine's own
    // things() when it has one, else the props list (no affordances: the server reads those from the props file)
    function thingsNow() {
      if (!engineOn || !engine) return [];
      try { if (typeof engine.things === 'function') return engine.things() || []; } catch (e) { return []; }
      try { return engine.props().map(p => ({id: p.id, name: p.name, where: p.where, held: p.held || null, pos: p.pos})); } catch (e) { return []; }
    }
    function kfBusy() {   // for armsBusy / the startle damping: what a playing keyframe job is moving
      const b = {l: false, r: false, body: false, head: false, any: false};
      for (const J of S.kf) {
        if (!J.wc || !(J.w > 0.05)) continue;
        b.any = true;
        for (const c in J.wc) {
          if (!(J.wc[c] > 0.05)) continue;
          if (/^(hand|thumb|point|fingers|shoulder)_l(_|$)/.test(c)) b.l = true;
          else if (/^(hand|thumb|point|fingers|shoulder)_r(_|$)/.test(c)) b.r = true;
          else if (/^(body_|root_|squash$|foot_)/.test(c)) b.body = true;
          else if (/^head_/.test(c)) b.head = true;
        }
      }
      return b;
    }
    function kfFeed(id, poses, done, info) {
      const J = S.kf.find(j => j.id === String(id) && !j.term);
      if (!J) return {ok: false, id, reason: S.kfHist.has(String(id)) ? 'that move already ended' : 'no such keyframe job'};
      if (info && (info.hold === false || info.hold === 'false')) J.released = true;
      return kfFeedJ(J, poses, done === true || done === 'true', info || null);
    }
    function kfRelease(id) {
      const J = S.kf.find(j => j.id === String(id) && !j.term);
      if (!J) return {ok: false, id, reason: 'no such keyframe job'};
      J.released = true;
      return {ok: true, id: J.id, phase: J.phase, ready: J.pl.ready()};
    }
    function kfState(id) {
      if (id === undefined || id === null) { const J = S.kf[S.kf.length - 1]; return J ? kfSummary(J) : null; }
      const J = S.kf.find(j => j.id === String(id));
      return J ? kfSummary(J) : (S.kfHist.get(String(id)) || null);
    }
    // ---------------------------------------------------------------- channel ownership + the listener hook (§3)
    // owned(): {channel: weight 0..1} of what COMMANDED actions drive right now — keyframe jobs (their weight, speech-yielded on
    // SPEECH_GROUP), whole-body moves / object work of the engine (its move-layer weight on root / feet / camera, its recorded
    // set-weights elsewhere), commanded gestures / expressions (their envelope), held poses (their ease) and commanded looks.
    // Reflexes and the idle / speaking / listening layers are not commanded and never appear.
    const ENG_ROOT = ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'stand', 'cam_zoom', 'cam_x', 'cam_y',
      'foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z'];
    function ownedNow() {
      const W = {};
      const up = (p, w) => { if (w > 1e-3 && !(W[p] >= w)) W[p] = Math.min(1, w); };
      for (const J of S.kf) if (J.wc) for (const c in J.wc) up(c, J.wc[c]);
      for (const G of S.gestures) if (G.src === 'cmd' && G.t0 <= S.t && G.params) { const w = weightOf(G); for (const p of G.params) up(p, w); }
      for (const p in S.holds) { const h = S.holds[p]; up(p, h.out ? 1 - mj((S.t - h.out.t0) / h.out.ease) : mj((S.t - h.t0) / h.ease)); }
      if (S.ownLook > 1e-3) for (const p of ['gaze_x', 'gaze_y', 'head_yaw', 'head_pitch']) up(p, S.ownLook);
      if (engineOn && engine) {
        const E = engine.E, mw = clamp(+E.w || 0, 0, 1), rec = engine.lp ? engine.lp() : null;
        if (mw > 1e-3) for (const p of ENG_ROOT) if (p.startsWith('foot_') ? !!E.body.legs : true) up(p, mw);
        if (rec) for (const [p, w, , a] of rec) { if (ENG_ROOT.includes(p) || p === 'gaze_free') continue; if (w > 0) up(p, w); else if (a) up(p, mw); }
        if (E.gf > 1e-3 && (E.look && E.look.w > 1e-3 || (E.move && !E.move.silent))) up('gaze_free', E.gf);
      }
      return W;
    }
    let listener = null;
    const LISTEN_LIST = [...new Set([].concat(LISTENER_GROUPS.face, LISTENER_GROUPS.gaze, LISTENER_GROUPS.posture))];
    // the listener's values replace the rules / idle base on its groups and sit UNDER every commanded layer and the reflexes,
    // exactly as the server composes the learned reaction: final = alpha * listener + beta with learnedPass's per-channel
    // map (alpha = what survives of the base under commands, beta = reflex offsets + commanded values; blinks floor at the
    // reflex blink). Keyframe jobs are in that map, so per channel final = commanded·w + listener·(1 − w).
    function listenerPass(o, dt) {
      const L = S.lp; if (!L || !listener) return;
      let out = null;
      const rules = {}; for (const p of LISTEN_LIST) rules[p] = o[p];
      try {
        out = listener({t: S.t, dt, state: {speaking: !!S.av.speaking, voiced: +S.kfSv.toFixed(3), listening: !!S.uv.speaking,
          user: {present: !!S.user.present, ax: S.user.ax, ay: S.user.ay, attending: !!S.user.attending}, owned: ownedNow(), rules,
          kf: kfSummary(S.kf.find(J => !J.term) || null), character: S.charName || null}});
      } catch (e) { S.listenerErr = String(e && e.message || e).slice(0, 120); out = null; }
      if (!out || typeof out !== 'object') return;
      const sv = S.kfSv;
      for (const p of LISTEN_LIST) {
        if (!(p in out)) continue;
        const x = +out[p]; if (!Number.isFinite(x)) continue;
        const q = L.ab[p] || [1, 0];
        let v = p === 'blink_l' || p === 'blink_r' ? q[0] * Math.max(x, L.rb || 0) + q[1] : q[0] * x + q[1];
        if (SPEECH_SET.has(p) && sv > 0) v = lerp(v, rules[p], sv);   // voiced: the mouth is speech's
        const r = (S.rng && S.rng[p]) || RANGE[p]; if (r) v = clamp(v, r[0], r[1]);
        o[p] = v;
      }
    }
    // ---------------------------------------------------------------- the PROGRAM layer (app/behavior.js)
    // setProgramLayer(fn): fn(ctx) runs the standing-behaviour programs once per step and returns {<rig column>: [value, weight]}.
    // Programs sit UNDER every commanded layer (keyframe jobs, engine moves, commanded gestures / holds / looks), reflex offsets
    // and lip sync, and OVER the listener / learned network / rules: per column, with learnedPass's affine map (alpha, beta),
    //   final = lerp(below, alpha·v + beta, w)      and the map becomes [alpha·(1 − w), alpha·w·v + beta]
    // so the server (learned / unified mode) sees the program's columns as owned (the unified worker → set_override) and applies
    // the same composition. SPEECH_GROUP columns use w·(1 − speech) while voiced. Nothing changes while no layer is set.
    let programLayer = null;
    function programPass(o, dt, inputs) {
      const fn = programLayer; if (!fn) return;
      let out = null;
      try {
        out = fn({t: S.t, dt, features: inputs.features || null, out: o, character: S.charName || null, actions: S.actions,
          voice: {user: {speaking: !!S.uv.speaking, level: S.uv.lvl}, avatar: {speaking: !!S.av.speaking, level: S.av.lvl}},
          owned: ownedNow, state: stateOut});
      } catch (e) { S.programErr = String(e && e.message || e).slice(0, 120); out = null; }
      if (!out || typeof out !== 'object') return;
      const L = S.lp, sv = S.kfSv;
      for (const p in out) {
        if (!(p in RANGE) || LIPSYNC.has(p)) continue;
        const a = out[p]; if (!a) continue;
        let v = +a[0], w = clamp(+a[1] || 0, 0, 1);
        if (!Number.isFinite(v) || !(w > 0)) continue;
        if (SPEECH_SET.has(p)) w *= 1 - sv;
        const r = (S.rng && S.rng[p]) || RANGE[p]; v = clamp(v, r[0], r[1]);
        const q = L ? (L.ab[p] || [1, 0]) : [1, 0];
        o[p] = clamp(lerp(o[p], q[0] * v + q[1], w), r[0], r[1]);
        if (L) L.ab[p] = [q[0] * (1 - w), q[0] * w * v + q[1]];
      }
    }
    // ---------------------------------------------------------------- learned-reaction mode
    // The server's planner (react worker) owns the slow layers (life, attention, mirroring, speaking, listening, boredom,
    // prior) on the channels its retarget covers. The page keeps what must be instant or explicit: reflexes (startle,
    // reflex blinks, gaze snaps, nod-back, wave back, perk/flash) and commands (gestures, expressions, held poses,
    // commanded looks). Per channel the page sends an AFFINE map final = alpha * learned + beta (alpha = how much of the
    // learned value survives, beta = reflex offsets + commanded values) built by running the reflex/command part of the
    // pipeline symbolically on the unknown learned value; blinks: final = alpha * max(learned, rxBlink) + beta.
    // Rules output (step's return value) is untouched by any of this.
    function gazeOwnedNow() {
      const t = S.t;
      return !!(S.cmdLook || (S.around && S.around.kind === 'cmd') ||
        (S.focus && t <= S.focus.until && (S.focus.kind !== 'user' || S.user.present)));
    }
    const LP_CH = ['head_yaw', 'head_pitch', 'head_roll', 'head_z', 'eye_wide', 'brow_l', 'brow_r', 'smile', 'shoulder_l',
      'shoulder_r', 'ear_l', 'ear_r', 'ear_l_turn', 'ear_r_turn', 'body_pitch', 'hand_l_y', 'hand_r_y', 'hand_l_z',
      'hand_r_z', 'hand_l_open', 'hand_r_open'];
    function learnedPass(dt, preG, e) {
      const t = S.t, AB = {};
      const ab = p => AB[p] || (AB[p] = [1, 0]);
      const add = (p, v) => { if (v) ab(p)[1] += v; };
      // gaze snaps / commanded looks: the page's attention system owns gaze + head direction while one is active
      const own = gazeOwnedNow();
      S.wf = lp(S.wf || 0, own ? 1 : 0, dt, own ? 0.035 : 0.25);
      if (S.wf > 1e-3) for (const p of ['gaze_x', 'gaze_y', 'head_yaw', 'head_pitch']) AB[p] = [1 - S.wf, S.wf * preG[p]];
      // reflex envelopes as additive offsets (the same terms the rules pipeline adds)
      const fl = e.fl, flp = Math.max(0, fl), sgn = e.ax >= 0 ? 1 : -1;
      add('head_pitch', -0.13 * fl); add('head_roll', 0.12 * e.curious * sgn);
      add('head_z', -0.055 * fl + 0.02 * e.curious + 0.015 * e.perkEnv);
      add('eye_wide', 0.9 * e.ewStart + 0.3 * e.perkEnv + 0.12 * e.curious + 0.2 * e.flashEnv);
      const br = 0.6 * e.flashEnv + 0.7 * e.ewStart + 0.25 * e.perkEnv + 0.25 * e.greetEnv + 0.3 * e.curious;
      add('brow_l', br); add('brow_r', br);
      add('smile', 0.25 * e.greetEnv + 0.1 * e.perkEnv);
      add('shoulder_l', 0.5 * flp); add('shoulder_r', 0.5 * flp);
      const ea = 0.35 * e.pk + 0.45 * e.perkEnv - 0.6 * flp + 0.2 * e.curious;
      add('ear_l', ea); add('ear_r', ea);
      add('ear_l_turn', -0.2 * e.curious - 0.3 * e.ep); add('ear_r_turn', -0.2 * e.curious + 0.3 * e.ep);
      add('body_pitch', -0.5 * fl + 0.12 * e.curious + 0.05 * e.perkEnv);
      for (const sd of ['l', 'r']) { add('hand_' + sd + '_y', 0.2 * flp); add('hand_' + sd + '_z', 0.12 * flp); add('hand_' + sd + '_open', 0.3 * flp); }
      // affect: the feeling sits under the reflexes / commands, over the learned slow layers
      if (S.affRec) for (const [p, kind, v, w] of S.affRec) { if (!(p in RANGE)) continue; const q = ab(p); if (kind === 'set') { q[0] *= 1 - w; q[1] = (1 - w) * q[1] + w * v; } else q[1] += v; }
      // reflex + command gestures / expressions, in applyGestures' order (life, fidget, listening, speech = slow layers)
      for (const {G, w, ki, r} of (S.gRec || [])) {
        if (priorityOf(G) < 3) continue;
        if (r.set) { const w2 = w * Math.min(1, ki); for (const k in r.set) { if (!(k in RANGE)) continue; const q = ab(k); q[0] *= 1 - w2; q[1] = (1 - w2) * q[1] + w2 * r.set[k]; } }
        if (r.add) for (const k in r.add) if (k in RANGE) ab(k)[1] += w * ki * K.gest * r.add[k];
      }
      // holds (commands): ease in from what is on screen (the live learned value), ease out back to it
      for (const p in S.holds) {
        const h = S.holds[p], q = ab(p);
        if (h.out) { const s2 = mj((t - h.out.t0) / h.out.ease); q[0] *= s2; q[1] = (1 - s2) * h.out.v + s2 * q[1]; }
        else { const k = mj((t - h.t0) / h.ease); q[0] *= 1 - k; q[1] = (1 - k) * q[1] + k * h.to; }
      }
      // whole-body moves / object actions: their sets and adds, in the order the engine applied them
      const er = engineOn && engine.lp ? engine.lp() : null;
      if (er) for (const [p, w, v, a] of er) { const q = ab(p); if (w > 0) { q[0] *= 1 - w; q[1] = (1 - w) * q[1] + w * v; } if (a) q[1] += a; }
      // keyframe jobs (applied last, over everything): commanded·w + below·(1 − w) on every channel they drive
      for (const J of S.kf) if (J.wc && J.T) for (const c in J.wc) { const w = J.wc[c]; if (!(w > 0) || !(c in RANGE)) continue; const q = ab(c); q[0] *= 1 - w; q[1] = (1 - w) * q[1] + w * J.T[c]; }
      S.lp = {ab: AB, rb: S.rxBlink || 0, own: S.wf};
    }
    // the learned parts of the last step, in a vector order (the renderer's rig_params):
    // {ab: [[index, alpha, beta], ...] only where (alpha, beta) != (1, 0), rb: reflex blink floor 0..1}
    function learnedParts(names) {
      const L = S.lp; if (!L) return {ab: [], rb: 0};
      const out = [];
      names.forEach((n, i) => { const q = L.ab[n]; if (q && (Math.abs(q[0] - 1) > 1e-4 || Math.abs(q[1]) > 1e-4)) out.push([i, clamp(q[0], 0, 1), q[1]]); });
      return {ab: out, rb: clamp(L.rb, 0, 1)};
    }

    // ---------------------------------------------------------------- affect: the feeling (app/affect.js)
    // Face channels are SET toward the feeling's targets (20 % of the live variation — idle noise, his mirrored face,
    // greetings — stays on top so it never freezes); posture channels get OFFSETS. Everything after this (gestures,
    // co-speech, commands, holds, keyframe jobs, the whole-body engine) plays over it. The lips stay lip sync's: the
    // silent-mouth target fades out while it talks. While the body works (an engine job, a lesson, a station view, a
    // keyframe job) only the face, ears and energy show.
    const AFF_REST = {smile: 0.1};
    function affectPass(o, dt) {
      if (!AFF) return;
      let busy = false;
      if (engineOn && engine) { try { busy = !!((engine.kit && engine.kit.busy && engine.kit.busy()) || engine.view); } catch (e) { busy = false; } }
      if (!busy && S.kf.some(J => !J.term)) busy = true;
      const r = AFF.step(dt, {now: S.t, speaking: !!S.av.speaking, silentT: S.av.silentT || 0, bodyFree: !busy});
      S.affOut = r;
      if (!r || r.w < 1e-4) { S.affE = 0; S.affRec = null; if (engineOn && engine && engine.affectCam) engine.affectCam(0); return; }
      const w = r.w, wb = w * r.wb, F = r.face, A = r.add, X = r.acc, keep = 0.2;
      const rec = S.affRec = [];   // for learnedPass (react learned / unified mode): [param, 'set' | 'add', value, weight]
      const setP = (p, T, ww) => { if (p in o) { ww = clamp(ww, 0, 1); o[p] = lerp(o[p], T + keep * (o[p] - (AFF_REST[p] || 0)), ww); rec.push([p, 'set', T, ww * (1 - keep)]); } };
      const o0 = {}; for (const p of AFF_ADD) o0[p] = o[p];
      setP('smile', F.smile, w); setP('squint', F.squint, w); setP('eye_wide', clamp(F.eye_wide + X.eye_wide, 0, 1), w); setP('brow_knit', F.brow_knit, w);
      const ob = A.one_brow * r.side;
      setP('brow_l', F.brow + X.brow + ob, w); setP('brow_r', F.brow + X.brow - ob, w);
      if (F.lid > 0) { o.blink_l = Math.max(o.blink_l, F.lid * w); o.blink_r = Math.max(o.blink_r, F.lid * w); }
      S.affMouth = lp(S.affMouth || 0, S.av.speaking ? 0 : 1, dt, S.av.speaking ? 0.06 : 0.35);
      if (F.mouth > 0) o.mouth_open = Math.max(o.mouth_open, F.mouth * w * S.affMouth);
      o.gaze_x += w * A.gaze_x * r.side; o.gaze_y += w * A.gaze_y;
      o.head_pitch += w * (A.head_pitch + X.head_pitch); o.head_roll += w * A.head_roll * r.side;
      for (const s2 of ['l', 'r']) {
        o['ear_' + s2] += w * (A.ear + X.ear);
        o['shoulder_' + s2] += wb * A.shoulder;
        o['hand_' + s2 + '_x'] += wb * (A.hand_x + (X.hand_x || 0)); o['hand_' + s2 + '_y'] += wb * (A.hand_y + X.hand_y); o['hand_' + s2 + '_z'] += wb * (A.hand_z + (X.hand_z || 0));
        o['hand_' + s2 + '_open'] += wb * (A.hand_open + (X.hand_open || 0));
      }
      o.tail_raise += w * A.tail_raise;
      if (A.nose) { o.nostril += w * A.nose; o.nose_wiggle += w * 0.3 * A.nose; }
      o.body_pitch += wb * (A.body_pitch + X.body_pitch); o.body_bounce += wb * (A.body_bounce + X.body_bounce);
      for (const p of AFF_ADD) if (o[p] !== o0[p]) rec.push([p, 'add', o[p] - o0[p], 1]);
      S.affE = r.energy;
      if (engineOn && engine && engine.affectCam) engine.affectCam(busy ? 0 : r.cam);
    }
    const AFF_ADD = ['gaze_x', 'gaze_y', 'head_pitch', 'head_roll', 'ear_l', 'ear_r', 'shoulder_l', 'shoulder_r', 'hand_l_x', 'hand_l_y', 'hand_l_z',
      'hand_r_x', 'hand_r_y', 'hand_r_z', 'hand_l_open', 'hand_r_open', 'tail_raise', 'nostril', 'nose_wiggle', 'body_pitch', 'body_bounce', 'blink_l', 'blink_r', 'mouth_open'];
    function affectSet(j) { if (!AFF) return false; const ok = AFF.set(j, S.t); if (ok) { S.log.push({t: +S.t.toFixed(2), affect: String(j.label || j.feel || ''), v: j.v, a: j.a, d: j.d, k: j.intensity}); if (S.log.length > 400) S.log.splice(0, 200); } return ok; }

    // ---------------------------------------------------------------- the frame
    function step(dt, inputs = {}) {
      dt = clamp(+dt || 0, 0, 0.1);
      S.t += dt; S.frame++;
      // the page's PLAYBACK (audio tap) time for this frame → the offset between the two clocks (playGesture at)
      if (Number.isFinite(+inputs.playT) && inputs.playT !== null) S.playOff = S.t - +inputs.playT;
      const t = S.t;
      const f = inputs.features || null;
      if (inputs.events) for (const ev of inputs.events) event(ev);
      if (f && f.events && f.events !== S._lastEvArr) { S._lastEvArr = f.events; for (const ev of f.events) event(ev); }
      // voices: the avatar's own voice gates the user's (echo)
      const avIn = inputs.avatarVoice && (inputs.avatarVoice.rms > 0 || inputs.avatarVoice.voiced) ? inputs.avatarVoice :
        (inputs.speaking ? {rms: 0.05, voiced: true} : inputs.avatarVoice);
      prosody(S.av, avIn, dt, false, cfg.avatarVoiceTh);
      prosody(S.uv, inputs.userVoice, dt, S.av.speaking && !(inputs.userVoice && inputs.userVoice.rms > 0.08), cfg.userVoiceTh);
      perceive(f, dt);
      behaviours(dt);
      const gz = gazeStep(dt);
      const blink = blinkStep(dt);
      const u = S.user, ex = S.expr.m, av = S.av, uv = S.uv;
      const spk = av.speaking ? 1 : 0, lis = uv.speaking ? 1 : 0;
      // mood: valence from his (mirrored) smile, arousal from events/energy/speech
      S.mood.val = lp(S.mood.val, clamp(ex.smile * 1.2 - ex.sad, -1, 1), dt, 2.5);
      // his loud/animated speech raises arousal (ears up, livelier tail, wider eyes); quiet speech does not
      const uLoud = lis ? clamp(uv.env / Math.max(0.02, uv.vmean * 1.5) - 0.6, 0, 1) : 0;
      S.mood.arousal = lp(S.mood.arousal, clamp(0.25 + 0.5 * u.energy * 3 + 0.25 * spk + 0.15 * lis + 0.3 * uLoud - 0.3 * S.bored, 0, 1), dt, 2);
      S.uLoud = lp(S.uLoud || 0, uLoud, dt, 0.6);

      // ---- prior (learned) suggestions: replace the procedural free head motion / brows when present
      let pr = null;
      if (prior && cfg.priorWeight > 0) {
        try {
          // the prior's frame (app/prior.js on branch prior): self = the avatar's voice, partner = his mic, face = his
          // MediaPipe face (blend + yaw/pitch in DEGREES), hands = wrists. Extra fields are ignored by it.
          const fc = f && f.face && f.face.present ? f.face : null, avv = inputs.avatarVoice || {}, uvv = inputs.userVoice || {};
          pr = prior.step({
            self: {rms: +avv.rms || 0, pitch: +avv.pitch || 0, voiced: !!av.speaking && av.silentT === 0, speaking: !!av.speaking},
            partner: {rms: +uvv.rms || 0, pitch: +uvv.pitch || 0, voiced: !!uv.speaking && uv.silentT === 0, speaking: !!uv.speaking},
            face: fc ? {present: true, blend: fc.blend || {}, matrix: fc.matrix || undefined,   // raw matrix = its training convention
              yaw: (+fc.yaw || 0) * 57.2958, pitch: (+fc.pitch || 0) * 57.2958} : {present: false},
            hands: ((f && f.hands) || []).map(h => ({x: (+h.x || 0) * 0.5 + 0.5, y: 0.5 - (+h.y || 0) * 0.5})),
            t, dt, user: {present: u.present, ax: u.ax, ay: u.ay, attending: u.attending}});
          if (pr && typeof pr === 'object') {   // its smile is absolute: use the deviation from its own running level
            S.priorSmile = S.priorSmile === undefined ? (+pr.smile || 0) : lp(S.priorSmile, +pr.smile || 0, dt, 8);
            pr.smile_dev = (+pr.smile || 0) - S.priorSmile;
          }
        } catch (e) { pr = null; S.priorErr = String(e).slice(0, 120); }
        S.priorOut = pr;
      }
      const usePrior = pr && typeof pr === 'object';
      const pw = usePrior ? clamp(cfg.priorWeight, 0, 1) : 0;

      const o = {};
      for (const p of ALL_PARAMS) o[p] = 0;

      // ---- head: attention + free continuous motion + nods + tilts
      const amp = av.speaking ? cfg.headNoise.speak : uv.speaking ? cfg.headNoise.listen : cfg.headNoise.idle;
      S.noiseAmp = S.noiseAmp || [amp[0], amp[1], amp[2]];
      for (let i = 0; i < 3; i++) S.noiseAmp[i] = lp(S.noiseAmp[i], amp[i] * (1 + 0.6 * av.lvl * spk) * (1 + 0.5 * S.bored) * K.amp * (1 + 0.7 * (S.affE || 0)), dt, 0.8);   // affect: a feeling's energy
      const nzY = S.nz.yaw.step(dt) * S.noiseAmp[0], nzP = S.nz.pitch.step(dt) * S.noiseAmp[1], nzR = S.nz.roll.step(dt) * S.noiseAmp[2];
      const nod = S.nodK.step(0, dt), sway = S.swayK.step(0, dt);
      const tilt = S.tiltK.step(S.tilt.target, dt);
      const pp = S.phrasePose && (av.speaking || av.silentT < 1.2) ? S.phrasePose : {yaw: 0, pitch: 0, roll: 0};
      S.ppY = S.ppY || new Spring(1.1, 0.95); S.ppP = S.ppP || new Spring(1.1, 0.95); S.ppR = S.ppR || new Spring(1.0, 0.95);
      const ppy = S.ppY.step(pp.yaw, dt), ppp = S.ppP.step(pp.pitch, dt), ppr = S.ppR.step(pp.roll, dt);
      let hYaw = gz.head[0] + (1 - pw) * (nzY + sway + ppy);
      let hPitch = -gz.head[1] + (1 - pw) * (nzP + nod + ppp) + 0.09 * ex.sad - 0.05 * ex.surprise + 0.03 * ex.anger;
      let hRoll = tilt + (1 - pw) * (nzR + ppr) + 0.22 * gz.head[0] * 0.5 + 0.35 * (u.present ? (u.rollM || 0) : 0);
      if (usePrior) {
        // its samples can step (a GRU/MDN chunk every 6 frames): a 3.5 Hz critically damped spring keeps its 0.1–2 Hz
        // motion and removes the steps
        // (×priorHead: it rides on top of the attention-driven head turns, which it never saw in training)
        S.prY = S.prY || new Spring(3, 1, 0); S.prP = S.prP || new Spring(3, 1, 0); S.prR = S.prR || new Spring(3, 1, 0);
        const ph = pw * cfg.priorHead;
        hYaw += ph * S.prY.step(clamp(+pr.head_yaw || 0, -0.3, 0.3), dt);
        hPitch += ph * S.prP.step(clamp(+pr.head_pitch || 0, -0.22, 0.22), dt);
        hRoll += ph * S.prR.step(clamp(+pr.head_roll || 0, -0.22, 0.22), dt);
      }
      // reflex envelopes
      const st = S.startle, su = t - st.t0;
      const fl = S.startleK.step(0, dt);   // flinch: jerks back fast, recovers ~1 s, slight rebound
      const ewStart = su < 3 ? st.amp * clamp(su / 0.06, 0, 1) * Math.exp(-Math.max(0, su - 0.15) / 0.75) : 0;
      if (su > 0.9 && su < 0.9 + dt * 1.5 && st.amp > 0) S.breath.sigh = t;   // the "phew" after a fright
      const pk = S.perkK.step(0, dt);
      const perkEnv = t - S.perk.t0 < 4 ? S.perk.amp * Math.exp(-(t - S.perk.t0) / 1.3) : 0;
      const greetEnv = pulse(t - S.greet.t0, 0, 0.5, 1.3, 2.4);
      const flashEnv = pulse(t - S.flash.t0, 0, 0.12, 0.3, 0.6) * S.flash.amp;
      const fo = S.focus, curious = fo && fo.curious ? pulse(t - fo.t0, 0, 0.5, 1.8, 2.6) : 0;
      hPitch += -0.13 * fl; hRoll += 0.12 * curious * (S.user.ax >= 0 ? 1 : -1) + 0.06 * ex.browUp * (S.user.ax >= 0 ? 1 : -1);
      o.head_yaw = hYaw; o.head_pitch = hPitch; o.head_roll = hRoll;
      // head velocity (for secondary motion)
      const hv = [(hYaw - S.prevHead[0]) / Math.max(dt, 1e-3), (hPitch - S.prevHead[1]) / Math.max(dt, 1e-3), (hRoll - S.prevHead[2]) / Math.max(dt, 1e-3)];
      S.prevHead = [hYaw, hPitch, hRoll];
      S.headVel = hv;
      o.head_x = 0.16 * hYaw + 0.01 * S.nz.lean.step(dt) * 0.5;
      o.head_y = -0.08 * hPitch;
 o.head_z = -0.025 * ex.fear - 0.015 * ex.surprise - 0.055 * fl + 0.02 * lis + 0.015 * spk * av.lvl + 0.02 * curious + 0.015 * perkEnv;

      // ---- eyes
      const roll = o.head_roll, c = Math.cos(roll), s = Math.sin(roll);
      const W = S.eye.W;
      const lx = c * W[0] - s * W[1], ly = s * W[0] + c * W[1];   // world offset → head-local (the shader adds it there)
      o.gaze_x = lx / GAZE_RX; o.gaze_y = ly / GAZE_RY;
      if (usePrior) { o.gaze_x += 0.3 * pw * (+pr.gaze_x || 0); o.gaze_y += 0.3 * pw * (+pr.gaze_y || 0); }
      const lid = 0.03 + 0.02 * S.bored + 0.06 * clamp(ex.sad, 0, 1);
      let bl = blink;
      const prBlink = usePrior ? Math.max(+pr.blink || 0, +pr.blink_l || 0) : 0;
      if (prBlink > 0.4 && !S.prBlinkWas && S.blink.phase === 0) forceBlink(1, false);   // its blink onsets trigger ours
      S.prBlinkWas = prBlink > 0.4;
      o.blink_l = Math.max(bl, lid); o.blink_r = Math.max(bl, lid);
      o.eye_wide = clamp(0.6 * ex.surprise + 0.5 * ex.fear + 0.9 * ewStart + 0.15 * (S.uLoud || 0) + 0.3 * perkEnv + ex.eyeWide + 0.12 * curious + 0.2 * flashEnv + (usePrior ? 0.5 * pw * (+pr.eye_wide || 0) : 0), 0, 1);
      o.squint = clamp(0.4 * ex.smile + ex.squint * 0.8 + 0.1 * S.mood.val + (usePrior ? 0.5 * pw * (+pr.squint || 0) : 0), 0, 1);

      // ---- brows
      const browNoise = S.nz.brow.step(dt) * (0.05 + 0.04 * spk);
      const bk = S.browK.step(0, dt);
      let brow = 0.6 * ex.surprise + 0.25 * ex.fear - 0.45 * ex.anger + browNoise + (1 - pw) * bk + 0.6 * flashEnv + 0.7 * ewStart + 0.25 * perkEnv + 0.12 * lis + 0.25 * greetEnv +
        ex.browUp * 0.9 - ex.browDown * 0.9 + 0.3 * curious;
      if (usePrior) brow += pw * 0.5 * ((+pr.brow_l || 0) + (+pr.brow_r || 0));
      const skew = 0.15 * Math.sin(t * 0.37 + 1.3) * (0.3 + bk);
      o.brow_l = brow * (1 + skew); o.brow_r = brow * (1 - skew);
      o.brow_knit = clamp(ex.browDown * 0.6 + ex.sad * 0.6 + 0.5 * ex.fear + 0.5 * ex.anger + (S.avert && av.silentT > 0.4 && !av.speaking ? 0.12 : 0), 0, 1);

      // ---- mouth corners
      const smNoise = 0.04 * S.nz.smile.step(dt);
      const neg = Math.max(ex.sad, ex.fear, ex.anger);   // his negative expression takes the resting smile away
      o.smile = clamp((0.12 + smNoise) * (1 - 0.8 * neg) + ex.smile * 1.0 + 0.25 * greetEnv + 0.1 * perkEnv + 0.12 * S.mood.val + (usePrior ? 0.6 * pw * (+pr.smile_dev || 0) : 0), 0, 1);
      o.mouth_open = clamp((ex.jawOpen * 0.6 + 0.2 * ex.surprise) * (av.speaking ? 0 : 1), 0, 0.5);

      // ---- breathing (rate varies; deeper sighs; inhale before phrases)
      const B = S.breath;
      B.rate = lp(B.rate, (0.22 + 0.06 * S.mood.arousal) * (av.speaking ? 0.8 : 1), dt, 3);
      let breathT;
      if (av.speaking) { B.speakEx = (B.speakEx ?? 0.4) - 0.25 * dt; breathT = Math.max(-0.8, B.speakEx); }
      else { B.ph += 2 * Math.PI * B.rate * dt; const sn = Math.sin(B.ph); breathT = 0.6 * sn + 0.12 * sn * Math.abs(sn); B.speakEx = 0.5; }
      if (B.inhaleAt !== undefined && t - B.inhaleAt < 0.45) breathT = 0.85;
      const sg = t - B.sigh; if (sg >= 0 && sg < 2.4) breathT = sg < 0.8 ? lerp(breathT, 1, mj(sg / 0.8)) : lerp(1, -0.7, mj((sg - 0.8) / 1.6));
      o.breath = B.sp.step(breathT, dt);
      o.shoulder_l = o.shoulder_r = clamp(0.05 * Math.max(0, o.breath) + 0.5 * Math.max(0, fl) + 0.3 * ex.fear, 0, 1);

      // ---- ears: pose from mood/attention, springs from head motion, turn toward his voice
 const earEmo = 0.5 * ex.surprise - 0.5 * ex.fear - 0.35 * ex.anger - 0.15 * ex.sad;
      const earBase = earEmo + 0.12 + 0.2 * (S.mood.arousal - 0.3) + 0.15 * (S.uLoud || 0) + 0.35 * pk + 0.45 * perkEnv + 0.3 * ex.smile - 0.55 * ex.sad - 0.35 * S.bored + 0.15 * lis - 0.6 * Math.max(0, fl) + 0.2 * curious;
      const acc = clamp((hv[1] - (S.prevHv1 ?? hv[1])) / Math.max(dt, 1e-3), -60, 60); S.prevHv1 = hv[1];
      S.earSL.v += -1.0 * acc * dt; S.earSR.v += -1.0 * acc * dt;   // ears swing with nods (inertia)
      const eL = S.earSL.step(earBase + 0.12 * S.nz.earL.step(dt), dt);
      const eR = S.earSR.step(earBase + 0.12 * S.nz.earR.step(dt), dt);
      o.ear_l = eL - 0.08 * hv[2]; o.ear_r = eR + 0.08 * hv[2];
      const turnBase = -0.25 * lis - 0.2 * curious + 0.15 * S.bored;
      const ep = S.earPoint && t - S.earPoint.t0 < 2.5 ? pulse(t - S.earPoint.t0, 0, 0.3, 1.8, 2.5) * Math.sign(S.earPoint.side || 0) : 0;
      o.ear_l_turn = turnBase + 0.15 * Math.sin(t * 0.23) * (u.present ? 0.3 : 1) - 0.3 * ep;
      o.ear_r_turn = turnBase + 0.15 * Math.sin(t * 0.19 + 2) * (u.present ? 0.3 : 1) + 0.3 * ep;
      o.ear_flick_l = 0; o.ear_flick_r = 0;

      // ---- body: weight shifts, turn with sustained looks, lean toward him when engaged, flinch back
      const sh = S.shift;
      if (sh.from) { const k = mj((t - sh.from.t0) / sh.from.dur); sh.lean = lerp(sh.from.lean, sh.to.lean, k); sh.yaw = lerp(sh.from.yaw, sh.to.yaw, k); }
      S.bodyFollow = lp(S.bodyFollow || 0, gz.head[0] * 0.45 * Math.min(1.25, K.sway) / 0.38, dt, 1.4);
      o.body_yaw = sh.yaw + S.bodyFollow + 0.08 * K.sway * S.nz.byaw.step(dt);
      o.body_lean = sh.lean + 0.06 * K.sway * S.nz.lean.step(dt) * 0.6;
      // personal space: very close (< 30 cm) → lean back a little; far (> 75 cm) → lean in to him
      // (relative to his usual distance, so a phone at 35 cm and a laptop at 65 cm both read as normal)
      const zz = u.present ? (u.zS || 0.45) : 0.45, rel = u.present && u.zBase ? zz / u.zBase : 1;
      const space = -0.35 * Math.max(clamp((0.72 - rel) / 0.22, 0, 1), clamp((0.2 - zz) / 0.06, 0, 1)) + 0.15 * clamp((rel - 1.35) / 0.4, 0, 1);
      S.space = lp(S.space || 0, space, dt, 0.8);
 o.body_pitch = -0.25 * ex.fear - 0.1 * ex.surprise - 0.06 * ex.sad + 0.1 * lis + 0.08 * spk * av.lvl + 0.12 * curious - 0.5 * fl - 0.1 * S.bored + 0.05 * perkEnv + S.space;
      o.body_bounce = 0;

      // ---- tail: wag with happiness/excitement, slow swish idle, raise with arousal
      const happy = clamp(ex.smile * 1.3 + 0.6 * greetEnv + 0.4 * perkEnv, 0, 1);
      S.tailPh = (S.tailPh || 0) + 2 * Math.PI * (0.5 + 1.8 * happy) * dt;
      const tailTarget = (0.1 + 0.5 * happy) * K.amp * Math.sin(S.tailPh) + 0.12 * S.nz.tail.step(dt);
      o.tail_wag = S.tailS.step(tailTarget, dt);
      o.tail_raise = clamp(0.3 + 0.35 * S.mood.arousal + 0.3 * happy - 0.3 * S.bored - 0.3 * ex.sad, 0, 1);
      o.tail_curl = clamp(0.15 + 0.15 * Math.sin(t * 0.13), 0, 1);
      o.nose_wiggle = 0; o.nostril = clamp(0.05 + 0.15 * Math.max(0, o.breath) * 0.3, 0, 1);

      // ---- paws: rest with drift; speech beats in gesture space; startle flinch
      const sp = S.speech;
      const beatOn = av.speaking && t < sp.beatUntil ? 1 : 0;
      S.beatW = lp(S.beatW || 0, beatOn, dt, beatOn ? 0.25 : 0.5);
      const hn = S.nz.hand.step(dt);
      // idle paw resettling: now and then one paw shifts to a new rest spot (min-jerk 0.9 s)
      const PR = S.pawRest || (S.pawRest = {l: {x: 0, y: 0, z: 0, open: REST_OPEN}, r: {x: 0, y: 0, z: 0, open: REST_OPEN}, next: 4, mv: null});
      if (t > PR.next) {
        PR.next = t + logn(lerp(9, 5, S.bored), 0.4);
        const sd = chance(0.5) ? 'l' : 'r', from = Object.assign({}, PR[sd]);
        const to = chance(0.3) ? {x: 0, y: 0, z: 0, open: REST_OPEN} : {x: U(-0.12, 0.08), y: U(-0.03, 0.1), z: U(0, 0.12), open: U(0.1, 0.5)};
        PR.mv = {sd, from, to, t0: t, dur: U(0.7, 1.1)};
      }
      if (PR.mv) { const k = mj((t - PR.mv.t0) / PR.mv.dur), q = PR[PR.mv.sd]; for (const c of ['x', 'y', 'z', 'open']) q[c] = lerp(PR.mv.from[c], PR.mv.to[c], k); if (k >= 1) PR.mv = null; }
      let torso = 0;
      for (const sd of ['l', 'r']) {
        const isBeat = sd === sp.beatSide || sp.beatBoth;
        // prep (rise into gesture space) → stroke (quick down-and-out on the accent) → hold → retraction (beatW decays)
        const stroke = isBeat && sp.beatHit ? pulse(t - sp.beatHit, 0, 0.1, 0.14, 0.45) : 0;
        const bw = isBeat ? S.beatW * (1 - S.kfPaw[sd]) : 0, rest = PR[sd];   // a keyframe job's paw keeps its trajectory
        o['hand_' + sd + '_x'] = rest.x + 0.02 * hn * (sd === 'l' ? 1 : -1) + bw * (0.12 + 0.04 * stroke);
        o['hand_' + sd + '_y'] = rest.y + 0.01 * hn + bw * (0.34 - 0.1 * stroke) + 0.2 * Math.max(0, fl);
        o['hand_' + sd + '_z'] = rest.z + bw * (0.3 + 0.05 * stroke) + 0.12 * Math.max(0, fl);
        o['hand_' + sd + '_open'] = rest.open + bw * (0.45 + 0.15 * stroke) + 0.3 * Math.max(0, fl);
        o['hand_' + sd + '_turn'] = bw * (0.45 - 0.15 * stroke);
        o['thumb_' + sd] = 0; o['point_' + sd] = 0;
        o['shoulder_' + sd] += 0.14 * bw + 0.1 * stroke * bw;
        if (!sp.beatBoth) torso += (sd === 'l' ? 1 : -1) * 0.1 * bw;   // the gesturing shoulder comes forward
      }
      o.body_yaw += torso;
      o.body_bounce += 0.04 * (sp.beatHit ? pulse(t - sp.beatHit, 0, 0.1, 0.14, 0.4) : 0) * S.beatW;
      affectPass(o, dt);   // affect: the feeling, under gestures / co-speech / commands / the engine
      if (engineOn) engine.base(o, dt);

      // ---- gestures (life, fidgets, reflexes, commands) and holds
      const lpOn = learned || !!listener || !!programLayer;   // the listener hook (and the program layer) compose through the learned-reaction map
      const preG = lpOn ? {gaze_x: o.gaze_x, gaze_y: o.gaze_y, head_yaw: gz.head[0], head_pitch: -gz.head[1]} : null;
      S.gRec = lpOn ? [] : null;
      applyGestures(o);
      S.react = Object.assign({}, o);   // the live (unheld) values: release eases toward these
      applyHolds(o);
      // ---- whole-body moves + object actions (top priority; nothing happens here unless one ran)
      const eng = engineOn ? engine.apply(o, dt, true) : null, pins = eng && eng.pins;   // (its channel record feeds owned())

      // ---- arm follow-through (light 2nd order on the hand targets; gestures already ease)
      for (const p of RIG_PARAMS) if (p.startsWith('hand_') && !p.endsWith('_open') && !p.endsWith('_turn')) {
        const F = S.handF[p] || (S.handF[p] = new Spring(cfg.handFollowHz, 0.85, o[p]));
        const raw = o[p];
        o[p] = F.step(raw, dt);
        if (pins && pins[p]) { const v = lerp(o[p], raw, pins[p]); F.v = (v - F.x) / Math.max(dt, 1e-3) * 0 + F.v * (1 - pins[p]); F.x = v; o[p] = v; }   // an object action drives the paw exactly
      }
      if (eng) engine.post(o);
      // ---- clamp, legacy mirror, lipsync left alone
      for (const p of RIG_PARAMS) {
        if (LIPSYNC.has(p)) { o[p] = 0; continue; }
        const r = (S.rng && S.rng[p]) || RANGE[p]; if (r) o[p] = clamp(o[p], r[0], r[1]);
        if (!Number.isFinite(o[p])) o[p] = 0;
      }
      if (eng) for (const p of BODY_PARAMS) { if (PERIODIC.has(p)) { if (!Number.isFinite(o[p])) o[p] = 0; continue; } const r = RANGE[p]; o[p] = Number.isFinite(o[p]) ? (r ? clamp(o[p], r[0], r[1]) : o[p]) : 0; }
      if (cfg.legacyArms) legacy(o); else for (const p of LEGACY) o[p] = 0;
      // ---- streamed keyframe jobs over everything above (nothing changes unless one runs)
      kfFrame(o, dt);
      if (eng && engine.postKf) engine.postKf(o, S.kf.length > 0);   // objects engine: the hands as drawn, held-prop offsets (§5)
      { const ownL = !!(S.cmdLook || (S.around && S.around.kind === 'cmd')); S.ownLook = lp(S.ownLook || 0, ownL ? 1 : 0, dt, ownL ? 0.035 : 0.25); }
      if (lpOn) learnedPass(dt, preG, {fl, ewStart, perkEnv, pk, greetEnv, flashEnv, curious, ep, ax: S.user.ax});
      if (listener) listenerPass(o, dt);   // planner contract §3: the listener's face / gaze / posture under the commanded layers
      if (programLayer) programPass(o, dt, inputs);   // standing-behaviour programs over the listener, under commands
      S.lastOut = o; S.mode = gz.mode;
      return o;
    }
    function legacy(o) {
      for (const s of ['l', 'r']) {
        o['arm_' + s] = clamp(o['hand_' + s + '_y'], 0, 1);
        o['paw_open_' + s] = o['hand_' + s + '_open'];
      }
      o.thumbs_up = Math.max(o.thumb_l, o.thumb_r);
    }

    function vector(o, order) {
      const names = order && order.length ? order : RIG_PARAMS;
      return names.map(n => { const v = o[n]; return Number.isFinite(v) ? Math.round(v * 1000) / 1000 : 0; });
    }
    function debug() {
      return {t: S.t, mode: S.mode, bored: S.bored, idleT: S.idleT, user: Object.assign({}, S.user, {hands: undefined}), expr: Object.assign({}, S.expr.m),
        speaking: S.av.speaking, listening: S.uv.speaking, holds: Object.keys(S.holds), gestures: S.gestures.map(g => g.name + ':' + g.src),
        cmdLook: S.cmdLook && S.cmdLook.target, focus: S.focus && S.focus.kind, avert: !!S.avert, around: S.around && S.around.kind,
        eye: S.eye.W.slice(), head: S.headT.slice(), prior: !!prior, priorErr: S.priorErr || null, preset: presetName,
        kf: S.kf.map(kfSummary), listener: !!listener, listenerErr: S.listenerErr || null, program: !!programLayer, programErr: S.programErr || null};
    }
    // motion.state() → the action state (contract §4); motion.state.<x> still reads the internal state (old tests)
    const stateApi = new Proxy(function () { return stateOut(); }, {
      get(f, k) { return k in S ? S[k] : f[k]; }, set(f, k, v) { S[k] = v; return true; }, has(f, k) { return k in S; }});
    return {step, command, release: (parts, ms) => release(parts, ms), event, vector, debug, state: stateApi, params: RIG_PARAMS, allParams: ALL_PARAMS,
      // a voice-model call that duplicated a queued job: its next calls follow that job (engine burst cursor)
      anchor: (id, src) => !!(engineOn && engine && engine.anchor && engine.anchor(id, SRC[src] || src || 'tool')),
      // scene: the station camera (docs/SCENE.md 6): 'board' | 'tv' | null (home / follow framing)
      setView: name => (engineOn && engine && engine.setView) ? engine.setView(name) : {ok: false, reason: 'no whole-body engine for this character'},
      get view() { return engineOn && engine ? engine.view || null : null; },
      // whole-body move APIs
      setProps(props, body, character) {
        S.charName = character || (props && props.character) || (body && body.character) || null;
        const has = !!(engine && body && typeof body === 'object');
        S.gtune = has && body.gesture_tuning && typeof body.gesture_tuning === 'object' ? body.gesture_tuning : null;
        S.face = has ? makeFace(body) : null;
        // the character's own chin (see makeChin). S.chinNone: its arm cannot reach its chin at all
        const cg = !S.face && Object.prototype.hasOwnProperty.call(CHIN_GEOM, S.charName) ? CHIN_GEOM[S.charName] : null;
        S.chin = makeChin(cg); S.chinNone = !!cg && !S.chin;
        // per-character output ranges (muse.body.json rig_ranges: its short arms reach the head only with hand_*_y up to
        // the wire limit 1.4); only rig columns already in RANGE, only [lo, hi] pairs
        S.rng = null;
        if (has && body.rig_ranges && typeof body.rig_ranges === 'object') for (const [k, v] of Object.entries(body.rig_ranges)) {
          if (RANGE[k] && Array.isArray(v) && v.length === 2 && v.every(Number.isFinite) && v[0] < v[1]) (S.rng = S.rng || {})[k] = [v[0], v[1]];
        }
        if (S.kf.length) { kfInterruptAll('character switch'); S.kf = []; }   // a keyframe job belongs to its character
        if (engine) {
          if (engineOn && !has) engine.stopAll('character switch');
          engineOn = has;
          if (has) { const r = engine.setConfig(props || null, body); emit({kind: 'config', name: 'props', src: 'auto', phase: 'end', did: `scene ready: ${r.props} objects`, args: r}); return r; }
        }
        return {props: 0, places: 0, off: true};
      },
      events(sinceS) { const t0 = sinceS === undefined ? -Infinity : S.t - +sinceS; return S.actions.filter(e => e.t >= t0); },
      playGesture, cancelGestures, armsBusy, think, thinkEnd, thinking, thinkVariants: Object.keys(THINK), thinkCan: v => !!THINK[v] && (!THINK[v].can || THINK[v].can()), chinTarget: (side, head, d) => S.face ? S.face.target('chin', side, head, d) : S.chin ? S.chin(side, head, d) : null, setCospeech(on) { S.cosp = !!on; if (S.cosp) { S.speech.beatUntil = -1; } return S.cosp; },
      now: () => S.t, props: () => engineOn ? engine.props() : [], things: thingsNow, get moves() { return engineOn ? engine.moveNames : []; }, get verbs() { return engineOn ? engine.verbs : []; },
      engine,
      setLearned(on) { learned = !!on; if (!learned && !listener && !programLayer) { S.lp = null; S.shownOut = null; } else if (!learned) S.shownOut = null; return learned; },
      // planner contract §3 / §5: streamed keyframe jobs, channel ownership, the listener hook (docs/LISTENER-INTERFACE.md)
      kfFeed, kfRelease, kfState, owned: ownedNow, LISTENER_GROUPS, SPEECH_GROUP, KF_ALIAS,
      setListener(fn) { listener = typeof fn === 'function' ? fn : null; if (!listener && !learned && !programLayer) S.lp = null; S.listenerErr = null; return !!listener; },
      // the program layer (app/behavior.js Behaviors installs / removes it as programs start and end)
      setProgramLayer(fn) { programLayer = typeof fn === 'function' ? fn : null; if (!programLayer && !learned && !listener) S.lp = null; S.programErr = null; return !!programLayer; },
      get programLayer() { return !!programLayer; },
      get listener() { return !!listener; },
      get learned() { return learned; }, learnedParts,
      setShown(names, a) { if (!learned || !names || !a) return; const o = {}; for (let i = 0; i < names.length && i < a.length; i++) if (Number.isFinite(+a[i])) o[names[i]] = +a[i]; S.shownOut = o; },
      setPrior(p, weights) {
        if (p && typeof p.step !== 'function' && typeof p.createPrior === 'function' && weights) p = p.createPrior(weights);
        else if (typeof p === 'function' && weights) p = p(weights);
        prior = p && typeof p.step === 'function' ? p : null; S.priorOut = null; return !!prior;
      }, gestures: Object.keys(GEST).concat(Object.keys(GEST_ALIAS)), expressions: Object.keys(EXPR), config: cfg,
      // affect: the feeling state (app/affect.js): affect(verdict) sets it, affectState() reads it, affectReset() drops it
      affect: j => affectSet(j), affectState: () => AFF ? AFF.state() : null, affectReset: () => { if (AFF) AFF.reset(); S.affE = 0; return !!AFF; }};
  }

  return {createMotion, RIG_PARAMS, SCHEMA_PARAMS, NEW_PARAMS, LIPSYNC: [...LIPSYNC], LEGACY: [...LEGACY], RANGE, DEFAULTS,
    GAZE_RX, GAZE_RY, TAN_HFOV2, TAN_VFOV2, mj, KF_ALIAS, LISTENER_GROUPS, SPEECH_GROUP};
});
