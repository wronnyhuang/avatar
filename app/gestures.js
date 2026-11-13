/* gestures.js — gestures HE teaches the avatar, learned from a few examples of his own landmarks.
 *
 * window.AvatarGestures = {create, Features, LAYOUT, VERSION}   (also module.exports when `module` exists)
 *
 * G = AvatarGestures.create({onEvent(ev), send(msg), faTarget: 0.25 (false alarms / min of his normal motion),
 *                            historyS: 360, aspect: 0.75 (fallback H/W when frames carry none)})
 *   G.frame(f, ctx?)          one perception.js feature frame (app/perception.js header); returns the gesture events of
 *                             this frame (also sent to onEvent). ctx = {t?, userSpeaking?} (optional).
 *   G.learn(name, {from:'last'|'next', seconds, counter, kind:'static'|'moving'|'auto', mirror:bool|'auto'})
 *       'last' → the result object at once; 'next' → a Promise of it (resolves when `seconds` of FRAME time have passed).
 *       result = {ok, name, kind, examples, counters, threshold, fa_per_min, quality:'good'|'weak', why?, segments}
 *       Every call ADDS examples to the named gesture (repeat it to give 2–3 examples); counter:true adds
 *       counterexamples ("not like this"). Saved by name: {type:'gesture_save', name, model} goes to the server
 *       (AVATAR_HOME/state/gestures/<name>.json); events gesture_learned / gesture_learn_failed go to onEvent.
 *   G.list() → [{name, kind, examples, counters, threshold, fa_per_min, quality, mirror, created, updated}]
 *   G.forget(name) → bool  (+ {type:'gesture_forget', name} to the server)
 *   G.state(name) → {on, score}           (the behavior layer's api.gesture(name))
 *   G.load(models) / G.onHub(msg)         models from the server ({type:'gestures', items:[model…]})
 *   G.export(name) → the model JSON; G.stats() → {frames, ms_p50, ms_p95, ms_max, baseline_s, gestures}
 *   Detection events: {type:'gesture', name, score (0..1), t, kind, hand?:'right'|'left', mirrored?}
 *
 * ── WHAT A FRAME BECOMES (Features) ───────────────────────────────────────────────────────────────────────────────
 * One vector of ~105 named numbers (LAYOUT), NaN where unseen, in HIS BODY FRAME: isotropic viewer coordinates
 * (x = perception's x, y = perception's y × aspect, so a unit is the same length both ways), translated to his shoulder
 * midpoint, divided by his shoulder width and rotated by the shoulder line (|angle| ≤ 0.5 rad). Shoulders come from the
 * pose model; without them they are inferred from the face (shoulder width ≈ 2.1 face widths, midpoint ≈ 1.33 face
 * widths below the eyes — measured on 34k frames of the eval clips — refined per person while both are seen).
 *   presence   face / his right hand / his left hand / shoulders (0/1; Bernoulli terms)
 *   head       yaw, pitch, roll; gaze gx, gy; head position hx, hy (body frame); lean = ln(face size / his 30 s mean)
 *   face       brows (5), eyes (6: blink / squint / wide, L and R), mouth (19) — MediaPipe blendshapes, raw
 *   body       shoulder tilt, shoulder width / face width (a body turn), elbows and wrists (pose) per ANATOMICAL arm
 *   hands      per anatomical hand (MediaPipe handedness, smoothed by perception.js; screen side as the fallback):
 *              palm centre (body frame) + palm-to-face vector; shape = 8 landmarks relative to the wrist in palm
 *              lengths, in the body orientation (so thumbs-up ≠ thumbs-down); finger extension ×4, thumb out, pinch,
 *              fingertip depth. A hand missing for < 0.35 s keeps its last values (MediaPipe drops hands often).
 * MIRROR maps a vector to the other side (left↔right hands / arms / L-R blendshapes, x / yaw / roll / gaze x negated).
 *
 * ── LEARNING ──────────────────────────────────────────────────────────────────────────────────────────────────────
 * Baseline = his normal motion: every frame of the session buffer (historyS) outside learning windows and outside
 * detections, plus the per-dimension prior (PRIOR, measured offline on 40 real conversation clips) with a 30 s
 * pseudo-count. In the learning window each frame gets a per-group activity = RMS z-score vs that baseline,
 * normalised by the same group's activity spread on the baseline; frames above it are the gesture, contiguous runs
 * (gaps < 0.3 s merged, ≥ 0.2 s) are the repeats. Kind: a run that holds still for ≥ 0.3 s (speed in z units/s
 * below its own median) for ≥ 35 % of its length is STATIC (its stillest stretch is the example), else MOVING.
 * STATIC model: per-dimension Gaussians from the examples (between-example spread + within-example jitter +
 *   0.35 × his baseline spread as the floor) vs the baseline Gaussian (and one Gaussian per counterexample, mixed);
 *   score = group-normalised log-likelihood ratio (each dimension's z clipped at 4, each group divided by √dims, so
 *   one glitching landmark cannot fire or veto it alone). A detection = LLR ≥ threshold held ≥ hold_s (≈ 0.3 s).
 * MOVING model: the K (≤ 10) dimensions that move most in the examples relative to his baseline spread (+ ≤ 4 that
 *   are held far from baseline, e.g. an open hand while waving), each example a template sequence at 15 Hz;
 *   detection = streaming subsequence DTW (SPRING, open begin) per template, cost = mean per-step RMS z-distance,
 *   duration 0.4–2.5 × the template's; missing dimensions cost a fixed penalty.
 * THRESHOLD: leave-one-out scores of the examples vs the baseline's would-be detections per minute: the threshold is
 *   the lowest one whose false-alarm rate on his baseline is ≤ faTarget, moved halfway toward the examples' worst
 *   leave-one-out score; counterexamples must stay below it. Short baselines (< 30 s) are padded with draws from the
 *   baseline Gaussian. Every 60 s of new normal motion the false-alarm side is re-checked (raises the threshold only).
 * Cost: features ≈ 0.02 ms, each static gesture ≈ 0.01 ms, each moving gesture ≈ 0.02 ms per frame (node, this box).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarGestures = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';
  const VERSION = 1;
  const NaN_ = NaN;
  const isNum = v => typeof v === 'number' && v === v && v !== Infinity && v !== -Infinity;
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const r4 = v => Math.round(v * 1e4) / 1e4;
  const median = a => { if (!a.length) return NaN; const s = Float64Array.from(a).sort(); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const quant = (a, p) => { if (!a.length) return NaN; const s = Float64Array.from(a).sort(); return s[Math.min(s.length - 1, Math.max(0, Math.floor(p * s.length)))]; };

  // ───────────────────────────────────────────── layout ─────────────────────────────────────────────
  const LAYOUT = [], IDX = {};
  function add(name, group, mirror, flip) { IDX[name] = LAYOUT.length; LAYOUT.push({name, group, mirror: mirror || name, flip: !!flip}); }
  add('p_face', 'pres'); add('p_hR', 'pres', 'p_hL'); add('p_hL', 'pres', 'p_hR'); add('p_pose', 'pres');
  add('yaw', 'head', null, true); add('pitch', 'head'); add('roll', 'head', null, true);
  add('gx', 'gaze', null, true); add('gy', 'gaze');
  add('hx', 'headpos', null, true); add('hy', 'headpos'); add('lean', 'headpos');
  const LR = (base, group) => { add(base + 'Left', group, base + 'Right'); add(base + 'Right', group, base + 'Left'); };
  LR('browDown', 'brow'); add('browInnerUp', 'brow'); LR('browOuterUp', 'brow');
  LR('eyeBlink', 'eye'); LR('eyeSquint', 'eye'); LR('eyeWide', 'eye');
  add('jawOpen', 'mouth'); LR('mouthSmile', 'mouth'); LR('mouthFrown', 'mouth'); add('mouthPucker', 'mouth'); add('mouthFunnel', 'mouth');
  add('cheekPuff', 'mouth'); add('mouthLeft', 'mouth', 'mouthRight'); add('mouthRight', 'mouth', 'mouthLeft'); add('mouthRollLower', 'mouth');
  add('mouthRollUpper', 'mouth'); LR('mouthPress', 'mouth'); LR('mouthStretch', 'mouth'); LR('noseSneer', 'mouth');
  add('jawLeft', 'mouth', 'jawRight'); add('jawRight', 'mouth', 'jawLeft');
  add('tilt', 'body', null, true); add('bturn', 'body');
  for (const j of ['elb', 'wri']) for (const s of ['R', 'L']) { const o = s === 'R' ? 'L' : 'R'; add(`${j}${s}_x`, 'body', `${j}${o}_x`, true); add(`${j}${s}_y`, 'body', `${j}${o}_y`); }
  // hand SHAPE is measured in the hand's own frame (wrist → middle-finger base = up), one group per finger, so the same
  // shape held 15° tilted, bigger / smaller or elsewhere in the frame reads the same; the tilt itself is the `dir` group
  const SHAPE_LM = [3, 4, 5, 8, 12, 16, 17, 20];
  const FINGER_OF = {3: 'th', 4: 'th', 5: 'ix', 8: 'ix', 12: 'mi', 16: 'ri', 17: 'pi', 20: 'pi'};
  const HAND_GROUPS = ['loc', 'mov', 'dir', 'th', 'ix', 'mi', 'ri', 'pi'];
  for (const s of ['R', 'L']) {
    const o = s === 'R' ? 'L' : 'R', g = 'h' + s;
    add(`${s}_px`, g + '_loc', `${o}_px`, true); add(`${s}_py`, g + '_loc', `${o}_py`);
    add(`${s}_fx`, g + '_loc', `${o}_fx`, true); add(`${s}_fy`, g + '_loc', `${o}_fy`);
    // how the hand MOVES, in palm lengths / s (no body needed: a wave reads the same anywhere in the frame, at any size)
    add(`${s}_vx`, g + '_mov', `${o}_vx`, true); add(`${s}_vy`, g + '_mov', `${o}_vy`);
    add(`${s}_ux`, g + '_dir', `${o}_ux`, true); add(`${s}_uy`, g + '_dir', `${o}_uy`); add(`${s}_face`, g + '_dir', `${o}_face`, true);
    add(`${s}_depth`, g + '_dir', `${o}_depth`);
    for (const i of SHAPE_LM) { add(`${s}_l${i}x`, g + '_' + FINGER_OF[i], `${o}_l${i}x`, true); add(`${s}_l${i}y`, g + '_' + FINGER_OF[i], `${o}_l${i}y`); }
    ['ix', 'mi', 'ri', 'pi'].forEach((fg, i) => add(`${s}_e${i}`, g + '_' + fg, `${o}_e${i}`));
    add(`${s}_thumb`, g + '_th', `${o}_thumb`); add(`${s}_pinch`, g + '_th', `${o}_pinch`);
  }
  const D = LAYOUT.length;
  const NAMES = LAYOUT.map(l => l.name);
  const PRES = [IDX.p_face, IDX.p_hR, IDX.p_hL, IDX.p_pose];
  const IS_PRES = new Uint8Array(D); for (const i of PRES) IS_PRES[i] = 1;
  const GROUP_NAMES = [...new Set(LAYOUT.map(l => l.group))].filter(g => g !== 'pres');
  const GROUP_OF = new Int16Array(D).fill(-1);
  LAYOUT.forEach((l, i) => { GROUP_OF[i] = l.group === 'pres' ? -1 : GROUP_NAMES.indexOf(l.group); });
  const MIRROR_PERM = new Int16Array(D), MIRROR_SIGN = new Float32Array(D);
  LAYOUT.forEach((l, i) => { MIRROR_PERM[i] = IDX[l.mirror]; MIRROR_SIGN[i] = l.flip ? -1 : 1; });
  function mirrorVec(v) { const o = new Float32Array(D); for (let i = 0; i < D; i++) o[MIRROR_PERM[i]] = v[i] * MIRROR_SIGN[i]; return o; }
  // the slot (hand / arm) groups a presence dim governs: a hand's features are only compared while that hand is seen
  const SLOT_OF = new Int8Array(D).fill(-1);    // 0 = face dims, 1 = right hand, 2 = left hand, 3 = pose dims
  LAYOUT.forEach((l, i) => { if (/^R_/.test(l.name)) SLOT_OF[i] = 1; else if (/^L_/.test(l.name)) SLOT_OF[i] = 2; });

  // PRIOR: per-dimension mean / spread of normal conversational motion (measured offline on 40 real clips of the
  // eval set (50 clips, 43k frames) through the real perception.js: Seamless conversations + the held-out suite;
  // tests/gesture_prior.cjs, trimmed 2–98 %, left/right twins symmetrised).
  // Presence entries = the probability of being seen. Replaced by the session's own baseline as it accumulates.
  const PRIOR = {"p_face":[0.79,0.407],"p_hR":[0.682,0.464],"p_hL":[0.682,0.464],"p_pose":[0.997,0.058],"yaw":[-0.017,0.134],"pitch":[0.216,0.178],"roll":[0.035,0.087],"gx":[-0.048,0.129],"gy":[-0.077,0.104],"hx":[0.011,0.074],"hy":[0.609,0.063],"lean":[-0.017,0.034],"browDownLeft":[0.157,0.175],"browDownRight":[0.157,0.175],"browInnerUp":[0.089,0.163],"browOuterUpLeft":[0.099,0.152],"browOuterUpRight":[0.099,0.152],"eyeBlinkLeft":[0.208,0.162],"eyeBlinkRight":[0.208,0.162],"eyeSquintLeft":[0.399,0.167],"eyeSquintRight":[0.399,0.167],"eyeWideLeft":[0.007,0.007],"eyeWideRight":[0.007,0.007],"jawOpen":[0.09,0.118],"mouthSmileLeft":[0.259,0.283],"mouthSmileRight":[0.259,0.283],"mouthFrownLeft":[0.002,0.006],"mouthFrownRight":[0.002,0.006],"mouthPucker":[0.049,0.089],"mouthFunnel":[0.027,0.033],"cheekPuff":[0,0],"mouthLeft":[0.002,0.003],"mouthRight":[0.002,0.003],"mouthRollLower":[0.009,0.019],"mouthRollUpper":[0.019,0.027],"mouthPressLeft":[0.046,0.064],"mouthPressRight":[0.046,0.064],"mouthStretchLeft":[0.021,0.028],"mouthStretchRight":[0.021,0.028],"noseSneerLeft":[0,0],"noseSneerRight":[0,0],"jawLeft":[0.002,0.002],"jawRight":[0.002,0.002],"tilt":[-0.021,0.047],"bturn":[2.097,0.209],"elbR_x":[0.644,0.083],"elbR_y":[-0.781,0.09],"elbL_x":[-0.644,0.083],"elbL_y":[-0.781,0.09],"wriR_x":[0.381,0.211],"wriR_y":[-1.227,0.395],"wriL_x":[-0.381,0.211],"wriL_y":[-1.227,0.395],"R_px":[0.28,0.295],"R_py":[-1.353,0.492],"R_fx":[0.292,0.265],"R_fy":[-1.988,0.479],"R_vx":[0.059,1.787],"R_vy":[0.015,1.669],"R_ux":[-0.347,0.473],"R_uy":[-0.481,0.619],"R_face":[0.23,0.811],"R_depth":[-0.462,0.595],"R_l3x":[0.33,0.628],"R_l3y":[0.975,0.447],"R_l4x":[0.372,0.748],"R_l4y":[1.209,0.531],"R_l5x":[0.098,0.243],"R_l5y":[0.996,0.112],"R_l8x":[0.181,0.532],"R_l8y":[1.825,0.475],"R_l12x":[0.189,0.513],"R_l12y":[1.772,0.54],"R_l16x":[0.153,0.569],"R_l16y":[1.599,0.534],"R_l17x":[-0.084,0.38],"R_l17y":[0.851,0.198],"R_l20x":[0.108,0.65],"R_l20y":[1.361,0.504],"R_e0":[0.82,0.286],"R_e1":[0.764,0.318],"R_e2":[0.682,0.338],"R_e3":[0.556,0.323],"R_thumb":[0.85,0.511],"R_pinch":[0.954,0.53],"L_px":[-0.28,0.295],"L_py":[-1.353,0.492],"L_fx":[-0.292,0.265],"L_fy":[-1.988,0.479],"L_vx":[-0.059,1.787],"L_vy":[0.015,1.669],"L_ux":[0.347,0.473],"L_uy":[-0.481,0.619],"L_face":[-0.23,0.811],"L_depth":[-0.462,0.595],"L_l3x":[-0.33,0.628],"L_l3y":[0.975,0.447],"L_l4x":[-0.372,0.748],"L_l4y":[1.209,0.531],"L_l5x":[-0.098,0.243],"L_l5y":[0.996,0.112],"L_l8x":[-0.181,0.532],"L_l8y":[1.825,0.475],"L_l12x":[-0.189,0.513],"L_l12y":[1.772,0.54],"L_l16x":[-0.153,0.569],"L_l16y":[1.599,0.534],"L_l17x":[0.084,0.38],"L_l17y":[0.851,0.198],"L_l20x":[-0.108,0.65],"L_l20y":[1.361,0.504],"L_e0":[0.82,0.286],"L_e1":[0.764,0.318],"L_e2":[0.682,0.338],"L_e3":[0.556,0.323],"L_thumb":[0.85,0.511],"L_pinch":[0.954,0.53]};
  const PRIOR_SHAPE = [0, 0.6];   // hand shape coordinates (palm lengths) / extension values without a measured prior
  function priorOf(name) {
    if (PRIOR[name]) return PRIOR[name];
    if (/_l\d+[xy]$/.test(name)) return [0, 0.6];
    if (/_e\d$/.test(name)) return [0.5, 0.35];
    if (/_thumb$/.test(name)) return [1.0, 0.35];
    if (/_pinch$/.test(name)) return [0.9, 0.4];
    if (/_depth$/.test(name)) return [-0.1, 0.2];
    return PRIOR_SHAPE;
  }
  const PRIOR_MEAN = new Float32Array(D), PRIOR_SD = new Float32Array(D);
  NAMES.forEach((n, i) => { const p = priorOf(n); PRIOR_MEAN[i] = p[0]; PRIOR_SD[i] = Math.max(1e-3, p[1]); });
  // noise floors: the smallest spread a dimension can honestly have (MediaPipe jitter)
  const FLOOR = new Float32Array(D);
  NAMES.forEach((n, i) => {
    FLOOR[i] = IS_PRES[i] ? 0.05 : /^(yaw|pitch|roll)$/.test(n) ? 0.015 : /^g[xy]$/.test(n) ? 0.02 : /_l\d+[xy]$/.test(n) ? 0.06
      : /_e\d$|_thumb|_pinch/.test(n) ? 0.06 : /_depth/.test(n) ? 0.05 : /^[RL]_[pf][xy]$|^(elb|wri)/.test(n) ? 0.03
      : /^h[xy]$|lean|tilt/.test(n) ? 0.015 : n === 'bturn' ? 0.05 : /_v[xy]$/.test(n) ? 0.6 : /_u[xy]$/.test(n) ? 0.3 : /_face$/.test(n) ? 0.25 : 0.02;   // blendshapes 0.02
  });

  // ───────────────────────────────────────────── features ─────────────────────────────────────────────
  const BODY_PER_FACE = 2.1, SH_BELOW_FACE = 1.33, HOLD_HAND_S = 0.35;
  function Features(opts) {
    opts = opts || {};
    const st = {ref: null, refT: -1, ratio: BODY_PER_FACE, below: SH_BELOW_FACE, sizeBase: null, hands: {R: null, L: null}, lastT: null};
    function compute(f) {
      const v = new Float32Array(D).fill(NaN_);
      const t = +f.t || 0, dt = st.lastT == null ? 0.05 : clamp(t - st.lastT, 0, 0.5); st.lastT = t;
      const k = isNum(f.aspect) && f.aspect > 0 ? f.aspect : (opts.aspect || 0.75);
      const face = f.face && f.face.present ? f.face : null;
      const pose = f.pose && f.pose.present && f.pose.shoulders && f.pose.shoulders.l && f.pose.shoulders.r &&
        f.pose.shoulders.l.v >= 0.5 && f.pose.shoulders.r.v >= 0.5 ? f.pose : null;
      const faceW = face && isNum(face.size) && face.size > 0 ? face.size * 2 : null;   // face width in x units
      // ── body reference (O, S, angle) ──
      let O = null, S = null, ang = 0;
      if (pose) {
        const sl = pose.shoulders.l, sr = pose.shoulders.r, dx = sr.x - sl.x, dy = (sr.y - sl.y) * k;
        S = Math.hypot(dx, dy); O = [(sl.x + sr.x) / 2, (sl.y + sr.y) / 2 * k]; ang = clamp(Math.atan2(dy, dx), -0.5, 0.5);
        if (faceW && S > 0.05 && isNum(face.x) && isNum(face.y)) {          // learn his face → body relation while both are seen
          const a = 1 - Math.exp(-dt / 3);
          st.ratio += a * (clamp(S / faceW, 1.4, 3.2) - st.ratio);
          st.below += a * (clamp((face.y * k - O[1]) / faceW, 0.6, 2.2) - st.below);
        }
        if (!(S > 0.05)) { O = null; S = null; }
      }
      if (!O && faceW && isNum(face.x) && isNum(face.y)) { S = st.ratio * faceW; O = [face.x, face.y * k - st.below * faceW]; ang = 0; }
      if (O) {
        if (st.ref && t - st.refT < 1.0) {                                   // smooth (τ 0.25 s) unless it jumped
          const a = 1 - Math.exp(-dt / 0.25), R = st.ref, jump = Math.hypot(O[0] - R.O[0], O[1] - R.O[1]) > 0.5 * R.S;
          if (!jump) { O = [R.O[0] + a * (O[0] - R.O[0]), R.O[1] + a * (O[1] - R.O[1])]; S = R.S + a * (S - R.S); ang = R.ang + a * (ang - R.ang); }
        }
        st.ref = {O, S, ang}; st.refT = t;
      } else if (st.ref && t - st.refT < 2.0) ({O, S, ang} = st.ref);
      const ca = Math.cos(-ang), sa = Math.sin(-ang);
      const toBody = (X, Y) => { const x = (X - O[0]) / S, y = (Y - O[1]) / S; return [x * ca - y * sa, x * sa + y * ca]; };
      const rot = (x, y) => [x * ca - y * sa, x * sa + y * ca];
      // ── presence ──
      v[IDX.p_face] = face ? 1 : 0; v[IDX.p_pose] = pose ? 1 : 0;
      // ── face ──
      if (face) {
        if (isNum(face.yaw)) v[IDX.yaw] = face.yaw; if (isNum(face.pitch)) v[IDX.pitch] = face.pitch; if (isNum(face.roll)) v[IDX.roll] = face.roll;
        if (isNum(face.gazeX)) v[IDX.gx] = face.gazeX; if (isNum(face.gazeY)) v[IDX.gy] = face.gazeY;
        if (pose && O && isNum(face.x) && isNum(face.y)) { const b = toBody(face.x, face.y * k); v[IDX.hx] = b[0]; v[IDX.hy] = b[1]; }
        if (isNum(face.size) && face.size > 0) {
          const ls = Math.log(face.size);
          st.sizeBase = st.sizeBase == null ? ls : st.sizeBase + (1 - Math.exp(-dt / 30)) * (ls - st.sizeBase);
          v[IDX.lean] = ls - st.sizeBase;
        }
        const B = face.blend || {};
        for (let i = IDX.browDownLeft; i <= IDX.jawRight; i++) { const x = B[NAMES[i]]; if (isNum(x)) v[i] = x; }
      }
      // ── body (pose) ──
      if (pose && O) {
        v[IDX.tilt] = ang;
        if (faceW) v[IDX.bturn] = S / faceW;
        for (const scr of ['l', 'r']) {
          const anat = pose.handed && pose.handed[scr] === 'left' ? 'L' : pose.handed && pose.handed[scr] === 'right' ? 'R' : (scr === 'l' ? 'L' : 'R');
          for (const [j, src] of [['elb', pose.elbows], ['wri', pose.wrists]]) {
            const p = src && src[scr];
            if (p && p.v >= 0.5 && isNum(p.x) && isNum(p.y)) { const b = toBody(p.x, p.y * k); v[IDX[`${j}${anat}_x`]] = b[0]; v[IDX[`${j}${anat}_y`]] = b[1]; }
          }
        }
      }
      // ── hands ──
      const seen = {R: null, L: null};
      for (const h of f.hands || []) {
        if (!h || !h.lm || h.lm.length < 21) continue;
        let s = h.handed === 'right' ? 'R' : h.handed === 'left' ? 'L' : (h.side === 'l' ? 'L' : 'R');
        if (seen[s]) { const o = s === 'R' ? 'L' : 'R'; if (!seen[o]) s = (h.side === 'l') ? 'L' : 'R'; if (seen[s]) s = o; if (seen[s]) continue; }
        seen[s] = h;
      }
      for (const s of ['R', 'L']) {
        const h = seen[s];
        if (h) { const prev = st.hands[s], vals = handVals(h, s, prev); st.hands[s] = {t, vals, mv: vals.mv}; }
        const H = st.hands[s];
        if (H && t - H.t <= HOLD_HAND_S) { v[IDX['p_h' + s]] = 1; for (const [i, x] of H.vals) v[i] = x; }
        else { v[IDX['p_h' + s]] = 0; if (H && t - H.t > HOLD_HAND_S) st.hands[s] = null; }
      }
      function handVals(h, s, prev) {
        const P = i => [(h.lm[i][0] - 0.5) * 2, (0.5 - h.lm[i][1]) * 2 * k, (+h.lm[i][2] || 0) * 2];
        const w = P(0), m5 = P(5), m9 = P(9), m17 = P(17);
        const palm = Math.max(1e-3, Math.hypot(m9[0] - w[0], m9[1] - w[1]));
        const cx = (w[0] + m5[0] + m9[0] + m17[0]) / 4, cy = (w[1] + m5[1] + m9[1] + m17[1]) / 4;
        const out = [];
        // velocity of the palm centre between fresh detections ≥ 70 ms apart (the hand model's sample-and-hold), low-passed
        const ht = isNum(h.t) ? +h.t : t, mv = prev && prev.mv ? Object.assign({}, prev.mv) : {x: cx, y: cy, t: ht, vx: 0, vy: 0, palm};
        if (ht - mv.t >= 0.07 && ht - mv.t < 0.6) {
          const ivx = (cx - mv.x) / ((ht - mv.t) * palm), ivy = (cy - mv.y) / ((ht - mv.t) * palm), a = 1 - Math.exp(-(ht - mv.t) / 0.12);
          mv.vx += a * (clamp(ivx, -30, 30) - mv.vx); mv.vy += a * (clamp(ivy, -30, 30) - mv.vy); mv.x = cx; mv.y = cy; mv.t = ht;
        } else if (ht - mv.t >= 0.6) { mv.x = cx; mv.y = cy; mv.t = ht; mv.vx = 0; mv.vy = 0; }
        const vr = rot(mv.vx, mv.vy); out.push([IDX[s + '_vx'], vr[0]], [IDX[s + '_vy'], vr[1]]);
        out.mv = mv;
        if (O) {   // where the hand is: only relative to HIM (no face / shoulders seen → unknown, not frame coordinates)
          const b = toBody(cx, cy); out.push([IDX[s + '_px'], b[0]], [IDX[s + '_py'], b[1]]);
          if (face && isNum(face.x) && isNum(face.y)) { const r = rot((cx - face.x) / S, (cy - face.y * k) / S); out.push([IDX[s + '_fx'], r[0]], [IDX[s + '_fy'], r[1]]); }
        }
        // palm direction (wrist → middle base) in his body orientation, and which side of the palm faces the camera
        const ux = (m9[0] - w[0]) / palm, uy = (m9[1] - w[1]) / palm, ud = rot(ux, uy);
        out.push([IDX[s + '_ux'], ud[0]], [IDX[s + '_uy'], ud[1]]);
        const ax_ = m17[0] - m5[0], ay_ = m17[1] - m5[1], crs = (ux * ay_ - uy * ax_) / Math.max(1e-4, Math.hypot(ax_, ay_));
        out.push([IDX[s + '_face'], clamp(crs, -1, 1)]);
        // shape in the hand's own frame: rotate so the palm direction points up (+y)
        const hc = uy, hs = -ux;   // rotation by -(angle of u) + 90°: (ux, uy) → (0, 1)
        for (const i of SHAPE_LM) { const p = P(i), dx = (p[0] - w[0]) / palm, dy = (p[1] - w[1]) / palm; out.push([IDX[`${s}_l${i}x`], dx * hc + dy * hs], [IDX[`${s}_l${i}y`], -dx * hs + dy * hc]); }
        const ext = Array.isArray(h.ext) && h.ext.length === 4 ? h.ext : null;
        const d3 = (a, c) => Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
        if (ext) for (let i = 0; i < 4; i++) out.push([IDX[`${s}_e${i}`], +ext[i] || 0]);
        else for (let i = 0; i < 4; i++) { const tip = P([8, 12, 16, 20][i]), mcp = P([5, 9, 13, 17][i]); out.push([IDX[`${s}_e${i}`], clamp((d3(tip, w) / Math.max(1e-4, d3(mcp, w)) - 1.15) / 0.75, 0, 1)]); }
        const t4 = P(4), t8 = P(8);
        out.push([IDX[s + '_thumb'], d3(t4, m5) / palm], [IDX[s + '_pinch'], d3(t4, t8) / palm]);
        const tz = (P(8)[2] + P(12)[2] + P(16)[2] + P(20)[2]) / 4 - w[2];
        out.push([IDX[s + '_depth'], tz / palm]);
        return out;
      }
      return v;
    }
    return {compute, state: st, reset() { st.ref = null; st.hands = {R: null, L: null}; st.lastT = null; }};
  }

  // ───────────────────────────────────────────── statistics ─────────────────────────────────────────────
  // baseline = (session frames) + PRIOR with a pseudo-count; presence dims are Bernoulli (sd from p)
  function baselineStats(frames, pseudo) {
    pseudo = pseudo == null ? 600 : pseudo;
    const n = new Float64Array(D), s1 = new Float64Array(D), s2 = new Float64Array(D);
    for (const v of frames) for (let i = 0; i < D; i++) { const x = v[i]; if (x === x) { n[i]++; s1[i] += x; s2[i] += x * x; } }
    const mean = new Float32Array(D), sd = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      const m0 = PRIOR_MEAN[i], v0 = PRIOR_SD[i] * PRIOR_SD[i];
      const N = n[i] + pseudo, m = (s1[i] + pseudo * m0) / N;
      const varr = (s2[i] + pseudo * (v0 + m0 * m0)) / N - m * m;
      mean[i] = m;
      sd[i] = IS_PRES[i] ? Math.sqrt(clamp(m, 0.03, 0.97) * (1 - clamp(m, 0.03, 0.97))) : Math.max(FLOOR[i], Math.sqrt(Math.max(0, varr)));
    }
    return {mean, sd, n: frames.length};
  }
  function meanSd(vecs) {     // per-dimension mean / sd over present values (NaN where never present)
    const n = new Float64Array(D), s1 = new Float64Array(D), s2 = new Float64Array(D);
    for (const v of vecs) for (let i = 0; i < D; i++) { const x = v[i]; if (x === x) { n[i]++; s1[i] += x; s2[i] += x * x; } }
    const mean = new Float32Array(D).fill(NaN_), sd = new Float32Array(D).fill(NaN_), cnt = new Float32Array(D);
    for (let i = 0; i < D; i++) if (n[i] > 0) { mean[i] = s1[i] / n[i]; sd[i] = Math.sqrt(Math.max(0, s2[i] / n[i] - mean[i] * mean[i])); cnt[i] = n[i]; }
    return {mean, sd, n: cnt, frames: vecs.length};
  }

  // ───────────────────────────────────────────── static model ─────────────────────────────────────────────
  const BETA = 0.35, ZCLIP = 4;
  // how much of his baseline spread a template tolerates per dim: WHERE the hand is matters least (the same shape a
  // bit elsewhere is the same gesture), its tilt somewhat (±15° must pass; upside-down must not)
  const BETA_DIM = new Float32Array(D).fill(BETA);
  NAMES.forEach((n, i) => { if (/^[RL]_[pf][xy]$|^(elb|wri)/.test(n) || n === 'hx' || n === 'hy') BETA_DIM[i] = 1.2; else if (/_u[xy]$|_face$/.test(n)) BETA_DIM[i] = 0.5; });
  // compile: examples [{mean, sd, n}] (+ counters) + base {mean, sd} → evaluator
  function compileStatic(examples, counters, base, mask) {
    const pos = mix(examples, base, true, mask);
    // a counterexample shares the gesture's spreads (only its MEANS differ, like LDA): otherwise a component with wider
    // spreads loses on every dim that does not matter, and the few that do (one finger) cannot separate them
    const negs = counters.map(c => { const n = mix([c], base, false, mask); for (let i = 0; i < D; i++) if (!IS_PRES[i] && pos.sg[i] === pos.sg[i] && n.mu[i] === n.mu[i]) n.sg[i] = pos.sg[i]; return n; });
    return {pos, negs, base};
  }
  // which dims a model looks at: its groups, plus the presence of the hands / face those groups need
  function maskOf(groups) {
    const m = new Uint8Array(D);
    if (!groups || !groups.length) { m.fill(1); return m; }
    const gs = new Set(groups);
    for (let i = 0; i < D; i++) if (!IS_PRES[i] && gs.has(LAYOUT[i].group)) m[i] = 1;
    if ([...gs].some(g => /^hR_/.test(g))) m[IDX.p_hR] = 1;
    if ([...gs].some(g => /^hL_/.test(g))) m[IDX.p_hL] = 1;
    if ([...gs].some(g => /^(head|gaze|headpos|brow|eye|mouth)$/.test(g))) m[IDX.p_face] = 1;
    return m;
  }
  // a component from example summaries: per-dim mean / sigma; presence probabilities (Laplace)
  function mix(exs, base, isPos, mask) {
    const mu = new Float32Array(D).fill(NaN_), sg = new Float32Array(D).fill(NaN_);
    for (let i = 0; i < D; i++) {
      if (mask && !mask[i]) continue;
      if (IS_PRES[i]) { let k = 0; for (const e of exs) k += e.mean[i] === e.mean[i] ? clamp(e.mean[i], 0, 1) : 0; mu[i] = (k + 0.5) / (exs.length + 1); continue; }
      const xs = [], within = [];
      for (const e of exs) if (e.mean[i] === e.mean[i]) { xs.push(e.mean[i]); within.push(e.sd[i] === e.sd[i] ? e.sd[i] : 0); }
      if (!xs.length) continue;
      let m = 0; for (const x of xs) m += x; m /= xs.length;
      let vb = 0; for (const x of xs) vb += (x - m) * (x - m); vb = xs.length > 1 ? vb / (xs.length - 1) : 0;
      let vw = 0; for (const w of within) vw += w * w; vw /= within.length;
      const fl = BETA_DIM[i] * base.sd[i];
      mu[i] = m; sg[i] = Math.sqrt(vb + vw + fl * fl + FLOOR[i] * FLOOR[i]);
    }
    return {mu, sg};
  }
  // log-likelihood of x under a component, group-normalised; dims = those present in x and defined in `ref`
  const GN = GROUP_NAMES.length;
  const _ll = new Float64Array(GN), _cnt = new Float64Array(GN);
  function llComp(x, comp, ref, base) {
    _ll.fill(0); _cnt.fill(0);
    let pres = 0;
    for (let i = 0; i < D; i++) {
      const xi = x[i]; if (xi !== xi) continue;
      if (ref.mu[i] !== ref.mu[i]) continue;                        // the gesture says nothing about this dim
      if (IS_PRES[i]) {
        const p = comp && comp.mu[i] === comp.mu[i] ? comp.mu[i] : clamp(base.mean[i], 0.03, 0.97);
        pres += Math.log(xi > 0.5 ? p : 1 - p);
        continue;
      }
      let m, s;
      if (comp && comp.mu[i] === comp.mu[i]) { m = comp.mu[i]; s = comp.sg[i]; } else { m = base.mean[i]; s = base.sd[i]; }
      let z = (xi - m) / s; if (z > ZCLIP) z = ZCLIP; else if (z < -ZCLIP) z = -ZCLIP;
      const g = GROUP_OF[i];
      _ll[g] += -0.5 * z * z - Math.log(s); _cnt[g]++;
    }
    let L = pres;
    for (let g = 0; g < GN; g++) if (_cnt[g] > 0) L += _ll[g] / Math.sqrt(_cnt[g]);
    return L;
  }
  function staticLLR(x, M) {
    const lp = llComp(x, M.pos, M.pos, M.base);
    let ln = llComp(x, null, M.pos, M.base);
    for (const c of M.negs) { const l = llComp(x, c, M.pos, M.base); const mx = Math.max(ln, l); ln = mx + Math.log(Math.exp(ln - mx) + Math.exp(l - mx)); }
    return lp - ln;
  }

  // ───────────────────────────────────────────── moving model (SPRING DTW) ─────────────────────────────────────────────
  const DTW_HZ = 15, MISS_PEN = 1.5;
  function seqOf(frames, dims, base) {        // frames [{t, v}] → resampled 15 Hz sequence of standardized dims
    const out = []; if (!frames.length) return out;
    const t0 = frames[0].t, t1 = frames[frames.length - 1].t; let j = 0;
    for (let t = t0; t <= t1 + 1e-6; t += 1 / DTW_HZ) {
      while (j + 1 < frames.length && frames[j + 1].t <= t + 1e-6) j++;
      out.push(stdz(frames[j].v, dims, base));
    }
    return out;
  }
  function stdz(v, dims, base) { const o = new Float32Array(dims.length); for (let q = 0; q < dims.length; q++) { const i = dims[q]; o[q] = (v[i] - base.mean[i]) / base.sd[i]; } return o; }
  function stepCost(x, T, w) {
    let s = 0, ws = 0;
    for (let q = 0; q < T.length; q++) {
      const a = T[q]; if (a !== a) continue;
      const b = x[q]; const wq = w[q]; ws += wq;
      if (b !== b) { s += wq * MISS_PEN * MISS_PEN; continue; }
      let d = b - a; if (d > 4) d = 4; else if (d < -4) d = -4;
      s += wq * d * d;
    }
    return ws > 0 ? Math.sqrt(s / ws) : MISS_PEN;
  }
  // a streaming SPRING matcher for one template: push(x, t) → {cost, start, dur} of the best match ending now
  function Spring(T, w, tdur) {
    const m = T.length; let Dp = new Float64Array(m + 1).fill(Infinity), Sp = new Float64Array(m + 1), Lp = new Int32Array(m + 1);
    let Dc = new Float64Array(m + 1), Sc = new Float64Array(m + 1), Lc = new Int32Array(m + 1);
    return {
      push(x, t) {
        Dc[0] = 0; Sc[0] = t; Lc[0] = 0;
        for (let i = 1; i <= m; i++) {
          const c = stepCost(x, T[i - 1], w);
          // predecessors: (i-1, now) vertical, (i, prev) horizontal, (i-1, prev) diagonal
          let best = Dc[i - 1], bs = Sc[i - 1], bl = Lc[i - 1];
          if (Dp[i] < best) { best = Dp[i]; bs = Sp[i]; bl = Lp[i]; }
          if (Dp[i - 1] <= best) { best = Dp[i - 1]; bs = i - 1 === 0 ? t : Sp[i - 1]; bl = Lp[i - 1]; }
          Dc[i] = best + c; Sc[i] = bs; Lc[i] = bl + 1;
        }
        const res = {cost: Dc[m] / Math.max(m, Lc[m] * 0.5), start: Sc[m], dur: t - Sc[m]};
        let tmp = Dp; Dp = Dc; Dc = tmp; tmp = Sp; Sp = Sc; Sc = tmp; const tl = Lp; Lp = Lc; Lc = tl;
        if (res.dur < 0.5 * tdur || res.dur > 2.0 * tdur + 0.2) res.cost = Infinity;
        return res;
      },
      reset() { Dp.fill(Infinity); }
    };
  }
  function compileMoving(model) {
    const base = {mean: Float32Array.from(model.base.mean), sd: Float32Array.from(model.base.sd)};
    const dims = model.dims.map(n => IDX[n]).filter(i => i != null);
    const w = Float32Array.from(model.weights || dims.map(() => 1));
    const temps = model.templates.map(tp => ({seq: tp.seq.map(r => Float32Array.from(r.map(x => x == null ? NaN : x))), dur: tp.dur}));
    return {base, dims, w, temps};
  }

  // ───────────────────────────────────────────── the detector ─────────────────────────────────────────────
  function create(opts) {
    opts = opts || {};
    const onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : () => {};
    const send = typeof opts.send === 'function' ? opts.send : () => {};
    const FA_TARGET = opts.faTarget != null ? +opts.faTarget : 0.25;
    const HIST_S = opts.historyS || 360, MIN_BASE_S = 30;
    const feats = Features({aspect: opts.aspect});
    const H = {t: [], v: [], excl: []};           // the session buffer (≤ 20 Hz), excl = inside a learn window / a detection
    const G = new Map();                           // name → {model, run}
    const pending = [];                            // 'next' learns waiting for frames
    const ms = []; let nFrames = 0, lastT = null, lastBufT = -1, nextRecal = 0, recalGap = 20, baseCache = null, baseCacheN = -1;
    const exclWins = [];                           // [t0, t1] windows never used as baseline (learned examples)
    let lastDebug = null;

    // ── buffer ──
    function push(t, v) {
      if (t - lastBufT < 0.045) return;
      lastBufT = t; H.t.push(t); H.v.push(v); H.excl.push(0);
      const cut = t - HIST_S; let k = 0; while (k < H.t.length && H.t[k] < cut) k++;
      if (k > 200) { H.t.splice(0, k); H.v.splice(0, k); H.excl.splice(0, k); }
    }
    function baselineFrames(skip0, skip1, withDetections) {
      const out = [];
      for (let i = 0; i < H.t.length; i++) {
        const t = H.t[i]; if (H.excl[i] && !withDetections) continue;
        if (skip0 != null && t >= skip0 && t <= skip1) continue;
        let bad = false; for (const w of exclWins) if (t >= w[0] && t <= w[1]) { bad = true; break; }
        if (!bad) out.push(i);
      }
      return out;
    }
    function baselineNow() {
      if (baseCache && H.t.length - baseCacheN < 100) return baseCache;
      const idx = baselineFrames(); baseCache = baselineStats(idx.map(i => H.v[i])); baseCache.idx = idx; baseCacheN = H.t.length;
      return baseCache;
    }

    // ── per frame ──
    function frame(f, ctx) {
      if (!f) return [];
      const t0 = now();
      const t = isNum(f.t) ? f.t : ctx && isNum(ctx.t) ? ctx.t : (lastT == null ? 0 : lastT + 0.055);
      if (lastT != null && t <= lastT) return [];
      lastT = t; nFrames++;
      const v = feats.compute(Object.assign({}, f, {t}));
      push(t, v);
      const evs = [];
      let anyOn = false;
      for (const [name, g] of G) { const e = step(g, v, t); if (e) evs.push(e); if (g.run.on || t - g.run.lastFire < 1.0) anyOn = true; }
      if (anyOn && H.excl.length) H.excl[H.excl.length - 1] = 1;
      if (evs.length) for (let i = H.t.length - 1; i >= 0 && H.t[i] >= t - 1.0; i--) H.excl[i] = 1;   // its build-up too
      for (const p of pending.slice()) if (t >= p.t1) { pending.splice(pending.indexOf(p), 1); p.resolve(finishLearn(p)); }
      if (G.size && t >= nextRecal) { recalGap = Math.min(60, recalGap * 1.25); nextRecal = t + recalGap; try { recalibrate(); } catch (e) {} }
      ms.push(now() - t0); if (ms.length > 600) ms.shift();
      for (const e of evs) { try { onEvent(e); } catch (_) {} }
      return evs;
    }
    function step(g, v, t) {
      const M = g.c, R = g.run, model = g.model;
      if (model.kind === 'static') {
        let llr = staticLLR(v, M), mir = false;
        if (g.cm) { const l2 = staticLLR(mirrorVec(v), g.cm); if (l2 > llr) { llr = l2; mir = true; } }
        const thr = g.thr, sc = 1 / (1 + Math.exp(-(llr - thr) / model.scale));
        R.score = sc; R.llr = llr;
        if (llr >= thr) { if (R.hiSince == null) R.hiSince = t; R.loSince = null; }
        else if (llr < thr - model.hyst) { R.hiSince = null; if (R.loSince == null) R.loSince = t; }
        if (!R.on && R.hiSince != null && t - R.hiSince >= model.hold_s && t - R.lastOff >= 0.4) {
          R.on = true; R.lastFire = t;
          return ev(model, sc, t, mir, v);
        }
        if (R.on && R.loSince != null && t - R.loSince >= 0.25) { R.on = false; R.lastOff = t; }
        return null;
      }
      // moving: feed the 15 Hz grid
      if (R.nextT != null && t < R.nextT) { R.score *= 0.97; return null; }
      R.nextT = (R.nextT == null ? t : R.nextT) + 1 / DTW_HZ; if (R.nextT < t) R.nextT = t + 1 / DTW_HZ;
      let best = Infinity, mir = false;
      const x = stdz(v, M.dims, M.base);
      // a match that began before the last detection is the same occurrence (SPRING's non-overlap rule)
      for (const s of g.springs) { const r = s.push(x, t); if (r.cost < best && r.start > R.lastFire) best = r.cost; }
      if (g.mirSprings) { const xm = stdz(mirrorVec(v), M.dims, M.base); for (const s of g.mirSprings) { const r = s.push(xm, t); if (r.cost < best && r.start > R.lastFire) { best = r.cost; mir = true; } } }
      const thr = g.thr, sc = 1 / (1 + Math.exp(-(thr - best) / model.scale));
      R.score = isFinite(best) ? sc : 0; R.cost = best;
      R.on = t - R.lastFire < 0.5;
      if (best <= thr && t - R.lastFire >= model.refr_s) { R.lastFire = t; R.on = true; return ev(model, sc, t, mir, v); }
      return null;
    }
    function ev(model, sc, t, mir, v) {
      const e = {type: 'gesture', name: model.name, score: r4(sc), t: r4(t), kind: model.kind};
      if (mir) e.mirrored = true;
      if (model.hand) e.hand = mir ? (model.hand === 'right' ? 'left' : 'right') : model.hand;
      return e;
    }

    // ── learning ──
    function learn(name, o) {
      o = o || {};
      name = String(name || '').trim().slice(0, 60);
      if (!name) return {ok: false, why: 'no name'};
      const from = o.from === 'next' || o.how === 'next' ? 'next' : 'last';
      const secs = clamp(+o.seconds || 5, 0.8, 20);
      const t = lastT == null ? 0 : lastT;
      if (from === 'next') {
        return new Promise(resolve => { pending.push({name, o, t0: t, t1: t + secs, resolve}); });
      }
      return finishLearn({name, o, t0: t - secs, t1: t});
    }
    function finishLearn(p) {
      let r;
      try { r = doLearn(p.name, p.o || {}, p.t0, p.t1); } catch (e) { r = {ok: false, name: p.name, why: 'error: ' + (e && e.message || e)}; }
      try { onEvent(Object.assign({type: r.ok ? 'gesture_learned' : 'gesture_learn_failed', t: r4(lastT || 0)}, r)); } catch (_) {}
      return r;
    }
    function windowFrames(t0, t1) { const out = []; for (let i = 0; i < H.t.length; i++) if (H.t[i] >= t0 - 1e-6 && H.t[i] <= t1 + 1e-6) out.push({t: H.t[i], v: H.v[i], i}); return out; }

    // ── what in a window is the gesture ──
    // per-group activity of a frame = RMS z-score of the group's dims vs his baseline; RARITY = -log10 of the share of
    // his baseline frames at least that active (scale-free across groups: a brow raise and a raised hand compare);
    // presence of a hand he rarely shows counts as its own group
    const NG = GN + 2;
    // per-frame squared z per group (+ hands he rarely shows); then an energy ENVELOPE over ±0.2 s (an oscillation is
    // active through its zero crossings)
    function groupZ2(v, base) {
      const a = new Float32Array(NG), n = new Float32Array(GN);
      for (let i = 0; i < D; i++) {
        const x = v[i]; if (x !== x || IS_PRES[i]) continue;
        const z = clamp((x - base.mean[i]) / base.sd[i], -6, 6), g = GROUP_OF[i]; a[g] += z * z; n[g]++;
      }
      for (let g = 0; g < GN; g++) a[g] = n[g] ? a[g] / n[g] : 0;
      a[GN] = v[IDX.p_hR] > 0.5 ? 1 - clamp(base.mean[IDX.p_hR], 0, 1) : 0; a[GN + 1] = v[IDX.p_hL] > 0.5 ? 1 - clamp(base.mean[IDX.p_hL], 0, 1) : 0;
      return a;
    }
    function envelopes(frames, base) {
      const Z = frames.map(w => groupZ2(w.v, base));
      return frames.map((w, j) => {
        const o = new Float32Array(NG); let n = 0;
        for (let k = j; k >= 0 && w.t - frames[k].t <= 0.2; k--) { for (let g = 0; g < NG; g++) o[g] += Z[k][g]; n++; }
        for (let k = j + 1; k < frames.length && frames[k].t - w.t <= 0.2; k++) { for (let g = 0; g < NG; g++) o[g] += Z[k][g]; n++; }
        for (let g = 0; g < GN; g++) o[g] = Math.sqrt(o[g] / n); for (let g = GN; g < NG; g++) o[g] /= n;
        return o;
      });
    }
    function rarityTables(base, frames) {   // frames: time-ordered {t, v} of his baseline
      const E = envelopes(frames, base), step = Math.max(1, Math.floor(E.length / 2000));
      const cols = Array.from({length: NG}, () => []);
      for (let j = 0; j < E.length; j += step) for (let g = 0; g < NG; g++) cols[g].push(E[j][g]);
      return cols.map(c => Float64Array.from(c).sort());
    }
    function rarity(tab, x) {   // -log10(share of baseline values ≥ x), capped by the table size
      const n = tab.length; if (!n) return 0;
      let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (tab[m] < x) lo = m + 1; else hi = m; }
      return -Math.log10(Math.max(1 / (n + 1), (n - lo) / n));
    }
    function levels(frames, base, tabs, bg) {        // per-frame group rarity Rs + one activity level (top group + half the next)
      const E = envelopes(frames, base);
      const Rs = E.map(e => Float32Array.from(e, (x, g) => g < GN ? Math.max(0, rarity(tabs[g], x) - (bg ? 0.8 * (bg[g] || 0) : 0)) : x * 2.5));
      const lvl = Rs.map(r => { const s = Array.from(r).sort((x, y) => y - x); return s[0] + 0.5 * s[1]; });
      return {Rs, lvl};
    }
    function candidates(frames, base, tabs, LV) {   // frames [{t, v}] → segments with per-frame group rarity
      const {Rs, lvl} = LV || levels(frames, base, tabs);
      // runs ≥ 1.3 (gaps ≤ 0.3 s) that reach 2.0 somewhere (hysteresis: a gesture's ramps belong to it)
      const segs = []; let cur = null;
      frames.forEach((w, j) => {
        if (lvl[j] >= 1.3) { if (cur && w.t - cur.t1 <= 0.3) { cur.t1 = w.t; cur.j1 = j; } else { cur = {t0: w.t, t1: w.t, j0: j, j1: j, peak: 0}; segs.push(cur); } cur.peak = Math.max(cur.peak, lvl[j]); }
      });
      const out = [];
      for (const sg of segs) {
        if (sg.t1 - sg.t0 < 0.2 || sg.peak < 2.0) continue;
        let area = 0; const gm = new Float64Array(NG);
        for (let j = sg.j0; j <= sg.j1; j++) { const dt = j > sg.j0 ? frames[j].t - frames[j - 1].t : 0.05; area += Math.min(4, lvl[j]) * dt; for (let g = 0; g < NG; g++) gm[g] += Rs[j][g]; }
        for (let g = 0; g < NG; g++) gm[g] /= (sg.j1 - sg.j0 + 1);
        out.push({t0: sg.t0, t1: sg.t1, area, gm, fr: frames.slice(sg.j0, sg.j1 + 1), R: Rs.slice(sg.j0, sg.j1 + 1)});
      }
      out.sort((x, y) => y.area - x.area);
      return out.slice(0, 4);
    }
    // a candidate's signature: mean + spread of every dim over its frames, and which hands are in it
    function signature(c) { if (!c.sig) { const s = meanSd(c.fr.map(w => w.v)); c.sig = s; } return c.sig; }
    function candDist(c1, c2, base) {
      const a = signature(c1), b = signature(c2);
      let s = 0, n = 0, pen = 0;
      for (let i = 0; i < D; i++) {
        if (IS_PRES[i]) { const pa = a.mean[i] === a.mean[i] ? a.mean[i] : 0, pb = b.mean[i] === b.mean[i] ? b.mean[i] : 0; if ((pa > 0.5) !== (pb > 0.5) && i !== IDX.p_face && i !== IDX.p_pose) pen += 1.5; continue; }
        const ma = a.mean[i], mb = b.mean[i]; if (ma !== ma || mb !== mb) continue;
        const sd = base.sd[i], za = Math.abs(ma - base.mean[i]) / sd, zb = Math.abs(mb - base.mean[i]) / sd, va = a.sd[i] / sd, vb = b.sd[i] / sd;
        if (za < 1.5 && zb < 1.5 && va < 1 && vb < 1) continue;            // only what is unusual in either counts
        const dm = (ma - mb) / sd, dv = va - vb; s += Math.min(16, dm * dm + dv * dv); n++;
      }
      return (n ? Math.sqrt(s / n) : 3) + pen;
    }
    // ── cross-example support: a frame belongs to the gesture when it is unusual for him AND the other examples have a
    // frame like it (the shared part of 2+ demonstrations; his other moves are not repeated in every one) ──
    function zvec(v, base) { const z = new Float32Array(D); for (let i = 0; i < D; i++) { const x = v[i]; z[i] = x === x ? clamp((x - base.mean[i]) / base.sd[i], -6, 6) : NaN; } return z; }
    function frameDist(a, b, va, vb) {
      let s = 0, n = 0;
      for (let i = 0; i < D; i++) {
        if (IS_PRES[i]) continue;
        const x = a[i], y = b[i]; if (x !== x || y !== y) continue;
        if (x < 1.5 && x > -1.5 && y < 1.5 && y > -1.5) continue;
        let d = x - y; if (d > 4) d = 4; else if (d < -4) d = -4; s += d * d; n++;
      }
      let pen = 0; if ((va[IDX.p_hR] > 0.5) !== (vb[IDX.p_hR] > 0.5)) pen += 1; if ((va[IDX.p_hL] > 0.5) !== (vb[IDX.p_hL] > 0.5)) pen += 1;
      return n >= 3 ? Math.sqrt(s / n) + pen : Infinity;
    }
    function supported(winFr, LVs, base) {
      const Z = winFr.map(fr => fr.map(f => zvec(f.v, base)));
      const K = winFr.length, out = [];
      for (let a = 0; a < K; a++) {
        const S = new Float32Array(winFr[a].length).fill(Infinity);
        for (let i = 0; i < winFr[a].length; i++) {
          if (LVs[a].lvl[i] < 1.3) continue;
          const mins = [];
          for (let b = 0; b < K; b++) {
            let m = Infinity;
            for (let j = 0; j < winFr[b].length; j++) {
              if (LVs[b].lvl[j] < 1.0) continue;
              if (b === a && Math.abs(winFr[b][j].t - winFr[a][i].t) < 0.8) continue;   // the same repeat
              const d = frameDist(Z[a][i], Z[b][j], winFr[a][i].v, winFr[b][j].v); if (d < m) m = d;
            }
            if (b !== a || m < Infinity) mins.push(m);
          }
          mins.sort((x, y) => x - y);
          const need = Math.max(1, Math.ceil((mins.length) / 2)), best = mins.slice(0, need);
          S[i] = best.length ? best.reduce((x, y) => x + y, 0) / best.length : Infinity;
        }
        out.push(S);
      }
      return out;
    }
    function supportedSegments(winFr, LVs, S) {
      const res = [];
      winFr.forEach((fr, a) => {
        const lvl = LVs[a].lvl, q = fr.map((_, i) => lvl[i] >= 1.3 && S[a][i] <= 1.3);
        let cur = null; const segs = [];
        fr.forEach((w, i) => { if (q[i]) { if (cur && w.t - fr[cur.j1].t <= 0.3) cur.j1 = i; else { cur = {j0: i, j1: i}; segs.push(cur); } } });
        for (const sg of segs) {
          // extend over the ramps (level ≥ 1.0, ≤ 0.3 s each side)
          while (sg.j0 > 0 && lvl[sg.j0 - 1] >= 1.0 && fr[sg.j0].t - fr[sg.j0 - 1].t < 0.3 && fr[sg.j1].t - fr[sg.j0 - 1].t < 6) { const t0 = fr[sg.j0].t; sg.j0--; if (t0 - fr[sg.j0].t > 0.3) break; }
          while (sg.j1 < fr.length - 1 && lvl[sg.j1 + 1] >= 1.0 && fr[sg.j1 + 1].t - fr[sg.j1].t < 0.3) { sg.j1++; }
          const t0 = fr[sg.j0].t, t1 = fr[sg.j1].t; if (t1 - t0 < 0.2) continue;
          const Rs = LVs[a].Rs.slice(sg.j0, sg.j1 + 1), gm = new Float64Array(NG); let area = 0;
          for (let j = sg.j0; j <= sg.j1; j++) { area += Math.min(4, lvl[j]) * (j > sg.j0 ? fr[j].t - fr[j - 1].t : 0.05); }
          Rs.forEach(r => { for (let g = 0; g < NG; g++) gm[g] += r[g]; }); for (let g = 0; g < NG; g++) gm[g] /= Rs.length;
          const sup = []; for (let j = sg.j0; j <= sg.j1; j++) if (S[a][j] < Infinity) sup.push(S[a][j]);
          res.push({win: a, t0, t1, area, gm, fr: fr.slice(sg.j0, sg.j1 + 1), R: Rs, support: median(sup)});
        }
      });
      // merge pieces of one window closer than 0.5 s (a gesture whose middle was less supported)
      res.sort((x, y) => x.win - y.win || x.t0 - y.t0);
      const out = [];
      for (const c of res) {
        const p = out[out.length - 1];
        if (p && p.win === c.win && c.t0 - p.t1 <= 0.5) {
          const fr = winFr[c.win], j0 = fr.indexOf(p.fr[0]), j1 = fr.indexOf(c.fr[c.fr.length - 1]);
          const R = LVs[c.win].Rs.slice(j0, j1 + 1), gm = new Float64Array(NG); R.forEach(r => { for (let g = 0; g < NG; g++) gm[g] += r[g]; }); for (let g = 0; g < NG; g++) gm[g] /= R.length;
          out[out.length - 1] = {win: c.win, t0: p.t0, t1: c.t1, area: p.area + c.area, gm, fr: fr.slice(j0, j1 + 1), R, support: Math.min(p.support, c.support)};
        } else out.push(c);
      }
      return out.filter(c => c.t1 - c.t0 >= 0.3);
    }
    // pick one candidate per window so the picks agree with each other (the gesture is what the examples share)
    function consensus(wins, base) {
      const live = wins.filter(w => w.cands.length);
      if (!live.length) return [];
      const k = live.length;
      let best = null, bestCost = Infinity;
      const combos = live.reduce((n, w) => n * w.cands.length, 1);
      const pick = new Array(k).fill(0);
      const cost = p => { let c = 0; for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) c += candDist(live[i].cands[p[i]], live[j].cands[p[j]], base); return c; };
      if (k === 1) best = [0];
      else if (combos <= 4096) {
        const rec = i => { if (i === k) { const c = cost(pick); if (c < bestCost) { bestCost = c; best = pick.slice(); } return; } for (let q = 0; q < live[i].cands.length; q++) { pick[i] = q; rec(i + 1); } };
        rec(0);
      } else { best = new Array(k).fill(0); }
      const chosen = live.map((w, i) => w.cands[best[i]]);
      // repeats inside one window: other candidates close to the window's pick join as examples
      const ref = chosen.length > 1 ? median(chosen.flatMap((c, i) => chosen.filter((_, j) => j > i).map(d => candDist(c, d, base)))) : 1.0;
      const extra = [];
      live.forEach((w, i) => w.cands.forEach((c, q) => { if (q !== best[i] && candDist(c, chosen[i], base) <= Math.max(0.8, 1.25 * ref) && c.area >= 0.3 * chosen[i].area) extra.push(c); }));
      return chosen.concat(extra);
    }
    function closestTo(cands, refs, base) {   // a counterexample window: the candidate most like the gesture
      let best = null, bd = Infinity;
      for (const c of cands) { const d = median(refs.map(r => candDist(c, r, base))); if (d < bd) { bd = d; best = c; } }
      return best;
    }
    // still-stretch analysis of one example → {kind vote, still frames}
    function analyse(c, base, rel) {
      const fr = c.fr;
      const dims = []; for (let i = 0; i < D; i++) if (!IS_PRES[i] && rel.includes(GROUP_OF[i])) dims.push(i);
      const Z = fr.map(w => dims.map(i => { const x = w.v[i]; return x === x ? (x - base.mean[i]) / base.sd[i] : NaN; }));
      const Zs = fr.map((w, j) => dims.map((_, q) => { let s = 0, n = 0; for (let k = j; k >= 0 && w.t - fr[k].t <= 0.12; k--) { const x = Z[k][q]; if (x === x) { s += x; n++; } }
        for (let k = j + 1; k < fr.length && fr[k].t - w.t <= 0.12; k++) { const x = Z[k][q]; if (x === x) { s += x; n++; } } return n ? s / n : NaN; }));
      // STILL = the longest stretch in which every dim that makes it (smoothed ±0.12 s) stays inside a band of 1.2 of his
      // baseline spreads, relative to its own offset (2nd widest dim: one jittery landmark does not break it) — a band,
      // not a speed threshold, so MediaPipe's sample-and-hold jumps and 30 fps jitter cannot split a held pose
      let bestRun = null;
      const lo = new Float64Array(dims.length), hi = new Float64Array(dims.length);
      // ...and only where its groups are unusual for him (rarity ≥ 1): his idle hand before / after is still too
      const act = fr.map((_, j) => { if (!c.R || !c.R[j]) return true; let m = 0; for (const g of rel) m = Math.max(m, c.R[j][g]); return m >= 1.5; });
      for (let a0 = 0; a0 < fr.length; a0++) {
        if (bestRun && fr.length - a0 <= bestRun.b - bestRun.a) break;
        if (!act[a0]) continue;
        lo.fill(Infinity); hi.fill(-Infinity);
        let b0 = a0;
        for (; b0 < fr.length; b0++) {
          if (!act[b0]) break;
          let w1 = 0, w2 = 0;
          for (let q = 0; q < dims.length; q++) {
            const x = Zs[b0][q]; if (x !== x) continue;
            if (x < lo[q]) lo[q] = x; if (x > hi[q]) hi[q] = x;
            const r = hi[q] - lo[q]; if (r > w1) { w2 = w1; w1 = r; } else if (r > w2) w2 = r;
          }
          if (w2 > 1.2) break;
        }
        if (b0 - 1 > a0 && (!bestRun || b0 - 1 - a0 > bestRun.b - bestRun.a)) bestRun = {a: a0, b: b0 - 1};
      }
      const vthr = 1.2;
      const stillDur = bestRun ? fr[bestRun.b].t - fr[bestRun.a].t : 0, dur = c.t1 - c.t0;
      // a still stretch whose dims still swing (a curling finger, a rocking head) is a moving gesture held in place
      // swing = how much the dims that make it still vary inside the still stretch, relative to how far they are from his
      // baseline (2nd largest ratio): a held pose varies little around a big offset; a beckon / a head rock swings
      let swing = 0;
      if (bestRun) {
        const per = [];
        dims.forEach((_, q) => { const xs = []; for (let j = bestRun.a; j <= bestRun.b; j++) { const x = Z[j][q]; if (x === x) xs.push(x); } if (xs.length < 3) return;
          const m = xs.reduce((s, x) => s + x, 0) / xs.length, sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) * (x - m), 0) / xs.length);
          if (Math.abs(m) >= 1.5 || sd >= 0.5) per.push(sd / (Math.abs(m) + 1)); });
        per.sort((x, y) => y - x); swing = per[1] || per[0] || 0;
      }
      const kind = stillDur >= 0.3 && stillDur >= 0.3 * dur ? 'static' : 'moving';   // (the band already rules out a swing)
      // the ACTIVE part (for a moving template): where the 3 fastest dims move ≥ 30 % of their 90th-percentile speed
      const sp = fr.map((w, j) => {
        let a = j, b = j; while (a > 0 && w.t - fr[a].t < 0.1) a--; while (b < fr.length - 1 && fr[b].t - w.t < 0.1) b++;
        const dt = Math.max(0.05, fr[b].t - fr[a].t), d = [];
        for (let q = 0; q < dims.length; q++) { const x = Zs[b][q], y = Zs[a][q]; if (x === x && y === y) d.push(Math.abs(x - y) / dt); }
        d.sort((x, y) => y - x); const top = d.slice(0, 3); return top.length ? Math.sqrt(top.reduce((s, x) => s + x * x, 0) / top.length) : 0;
      });
      const p90 = quant(sp, 0.9); let ja = -1, jb = -1;
      sp.forEach((v, j) => { if (v >= 0.3 * p90) { if (ja < 0) ja = j; jb = j; } });
      const core = ja >= 0 && fr[jb].t - fr[ja].t >= 0.3 ? [fr[ja].t, fr[jb].t] : null;
      return {kind, core, still: bestRun ? fr.slice(bestRun.a, bestRun.b + 1) : fr, stillDur, groups: rel.map(g => GROUP_NAMES[g]), why: {still: r4(stillDur), dur: r4(dur), swing: r4(swing), dims: dims.length, run: bestRun ? [r4(fr[bestRun.a].t), r4(fr[bestRun.b].t)] : null, act: act.filter(Boolean).length, n: fr.length}};
    }
    function doLearn(name, o, t0, t1) {
      const W = windowFrames(t0, t1);
      if (W.length < 6) return {ok: false, name, why: 'not enough camera frames in the window (' + W.length + ')'};
      const counter = !!o.counter;
      const prev = G.get(name);
      if (counter && !prev) return {ok: false, name, why: 'no gesture called "' + name + '" to add a counterexample to'};
      const bidx = baselineFrames(t0 - 1.0, t1 + 1.0);
      const baseVecs = bidx.map(i => H.v[i]);
      const base = baselineStats(baseVecs);
      const bfr = bidx.map(i => ({t: H.t[i], v: H.v[i]}));
      const tabs = rarityTables(base, bfr.length >= 200 ? bfr : bfr.concat(synthBaseline(base, 400).map((v, k) => ({t: 1e6 + k * 0.5, v}))));
      exclWins.push([t0 - 0.5, t1 + 0.5]); baseCache = null;     // never baseline again
      const model = prev ? JSON.parse(JSON.stringify(prev.model)) : {name, v: VERSION, names: NAMES, windows: [], created: Date.now()};
      model.windows = model.windows || [];
      // what he was ALREADY doing just before the window (a group busy in ≥ 70 % of [t0-1.5, t0-0.2]) is background, not it
      const pre = windowFrames(t0 - 1.5, t0 - 0.2), bg = new Array(NG).fill(0);
      if (pre.length >= 6) { const LVp = levels(pre, base, tabs); for (let g = 0; g < GN; g++) { const busy = LVp.Rs.filter(r => r[g] >= 1.0).length / pre.length; if (busy >= 0.7) bg[g] = r4(median(LVp.Rs.map(r => r[g]))); } }
      model.windows.push({t0: r4(t0), t1: r4(t1), counter, bg, frames: W.map(w => ({t: r4(w.t), v: arr(w.v)}))});
      const pos = model.windows.filter(w => !w.counter).slice(-6), neg = model.windows.filter(w => w.counter).slice(-3);
      model.windows = pos.concat(neg);
      model.updated = Date.now();
      // candidates in every window (re-derived each time: a later example can correct which part of an earlier window it was)
      const asFrames = w => w.frames.map(fr => ({t: fr.t, v: fromArr(fr.v)}));
      const wins = pos.map(w => ({w, cands: candidates(asFrames(w), base, tabs, levels(asFrames(w), base, tabs, w.bg))}));
      let fallback = false;
      for (const x of wins) if (!x.cands.length) {   // nothing stands out: the 1.2 s around its most unusual frame
        const fr = asFrames(x.w); let bj = 0, bv = -1;
        const E = envelopes(fr, base);
        fr.forEach((f, j) => { let m = 0; for (let g = 0; g < GN; g++) m = Math.max(m, rarity(tabs[g], E[j][g])); if (m > bv) { bv = m; bj = j; } });
        const tc = fr[bj].t, sel = fr.filter(f => Math.abs(f.t - tc) <= 0.6);
        const gm = new Float64Array(NG); x.cands = [{t0: sel[0].t, t1: sel[sel.length - 1].t, area: 0, gm, fr: sel}]; fallback = true;
      }
      const winFr = pos.map(asFrames), LVs = winFr.map((fr, a) => levels(fr, base, tabs, pos[a].bg));
      let chosen = [];
      const S = supported(winFr, LVs, base), sup = supportedSegments(winFr, LVs, S);
      // every window should contribute; a window without a supported segment falls back to its best-matching candidate
      if (sup.length >= 2) {
        chosen = sup.slice();
        wins.forEach((x, a) => { if (!sup.some(c => c.win === a) && x.cands.length) chosen.push(closestTo(x.cands, sup, base)); });
      } else chosen = consensus(wins, base);
      if (opts.debug) lastDebug = {wins: wins.map(x => x.cands.map(c => ({t0: r4(c.t0), t1: r4(c.t1), area: r4(c.area), gm: Array.from(c.gm, r4)}))),
        chosen: chosen.map(c => [r4(c.t0), r4(c.t1)]), dist: wins.map(x => x.cands.map(c => wins.map(y => y.cands.map(d => r4(candDist(c, d, base))))))};
      const negs = neg.map(w => closestTo(candidates(asFrames(w), base, tabs), chosen, base)).filter(Boolean);
      chosen.forEach(c => { if (c.win == null) { const k = wins.findIndex(x => x.cands.includes(c)); c.win = k; } });
      // the groups the examples SHARE are the gesture (his other hand, his talking face are not)
      const nEx = chosen.length, CG = [];
      for (let g = 0; g < GN; g++) { const k = chosen.filter(c => c.gm[g] >= 1.0).length; if (nEx > 1 ? k >= Math.ceil(0.6 * nEx) : k) CG.push(g); }
      if (!CG.length) { const c = chosen[0]; CG.push([...Array(GN).keys()].sort((a, b) => c.gm[b] - c.gm[a])[0]); }
      if (CG.length > 5) { const sc = g => chosen.reduce((s, c) => s + c.gm[g], 0); CG.sort((a, b) => sc(b) - sc(a)); CG.length = 5; }
      // trim every example to where those groups are active (≥ 1.0, gaps ≤ 0.3 s)
      const trim = c => {
        if (!c.R) return c;
        let best = null, cur = null;
        c.fr.forEach((w, j) => { let m = 0; for (const g of CG) m = Math.max(m, c.R[j][g]); if (m >= 1.0) { if (cur && w.t - c.fr[cur.b].t <= 0.3) cur.b = j; else cur = {a: j, b: j}; if (!best || c.fr[cur.b].t - c.fr[cur.a].t > c.fr[best.b].t - c.fr[best.a].t) best = {a: cur.a, b: cur.b}; } });
        if (!best || c.fr[best.b].t - c.fr[best.a].t < 0.2) return c;
        return Object.assign({}, c, {t0: c.fr[best.a].t, t1: c.fr[best.b].t, fr: c.fr.slice(best.a, best.b + 1), R: c.R.slice(best.a, best.b + 1)});
      };
      const mk = c0 => {
        const c = trim(c0);
        const an = analyse(c, base, CG), st = meanSd(an.still.map(w => w.v)), k = Math.max(1, Math.floor(an.still.length / 6));
        const pad = 0.15, src = c0.win != null && c0.win >= 0 ? wins[c0.win] : (wins.find(x => x.cands.includes(c0)) || null);
        const all = src ? asFrames(src.w) : c.fr;
        const fr = all.filter(w => w.t >= c.t0 - pad && w.t <= c.t1 + pad);
        let hR = 0, hL = 0; for (const w of c.fr) { if (w.v[IDX.p_hR] > 0.5) hR++; if (w.v[IDX.p_hL] > 0.5) hL++; }
        return {vote: an.kind, why: an.why, groups: an.groups, core: an.core ? an.core.map(r4) : null, mean: arr(st.mean), sd: arr(st.sd), n: an.still.length, hold: r4(an.stillDur || 0),
          samples: an.still.filter((_, j) => j % k === 0).slice(0, 6).map(w => arr(w.v)),
          frames: fr.map(w => ({t: r4(w.t), v: arr(w.v)})), dur: r4(c.t1 - c.t0 + 2 * pad), t: r4(c.t0), t1: r4(c.t1),
          hand: hR > hL * 1.5 && hR > 0.3 * c.fr.length ? 'right' : hL > hR * 1.5 && hL > 0.3 * c.fr.length ? 'left' : null};
      };
      let fitBase = base;
      if (counter && prev.model.examples && prev.model.examples.length && prev.model.base) {
        // a counterexample keeps the gesture as it was taught (examples, groups, the baseline then): only "not like this"
        // is new — re-deriving everything against today's baseline (which by now holds his repeats of it) would blur it
        fitBase = {mean: fromArr(prev.model.base.mean), sd: fromArr(prev.model.base.sd)};
        CG.length = 0; for (const gname of prev.model.groups || []) { const gi = GROUP_NAMES.indexOf(gname); if (gi >= 0) CG.push(gi); }
        const refs = prev.model.examples.map(e => ({fr: (e.frames || []).map(f => ({t: f.t, v: fromArr(f.v)}))})).filter(r => r.fr.length);
        const W2 = asFrames(model.windows.filter(w => w.counter).slice(-1)[0]);
        const LV2 = levels(W2, fitBase, tabs);
        let c = closestTo(candidates(W2, fitBase, tabs, LV2), refs, fitBase);
        if (!c) c = {t0: W2[0].t, t1: W2[W2.length - 1].t, fr: W2, R: LV2.Rs, gm: new Float64Array(NG), area: 0};
        model.examples = prev.model.examples; model.groups = prev.model.groups;
        model.counters = (prev.model.counters || []).concat([mk(c)]).slice(-3);
      } else {
        model.groups = CG.map(g => GROUP_NAMES[g]);
        // a hand that is part of it brings its whole description (where, shape, fingers) and the arms
        for (const h of ['hR', 'hL']) if (model.groups.some(g => g.startsWith(h + '_'))) for (const g of HAND_GROUPS.map(x => h + '_' + x).concat(['body'])) if (!model.groups.includes(g)) model.groups.push(g);
        model.examples = chosen.map(mk); model.counters = negs.map(mk);
      }
      if (o.kind === 'static' || o.kind === 'moving') model.kindFixed = o.kind;
      const votes = model.examples.filter(e => e.vote === 'static').length;
      const kind = model.kindFixed || (votes * 2 >= model.examples.length ? 'static' : 'moving');
      if (model.kind !== kind) { delete model.templates; delete model.dims; delete model.weights; }
      model.kind = kind;
      if (o.mirror === true || o.mirror === false) { model.mirror = o.mirror; model.mirrorAuto = false; }
      const hands = new Set(model.examples.map(e => e.hand).filter(Boolean));
      model.hand = hands.size === 1 ? [...hands][0] : null;
      if (model.mirror == null || model.mirrorAuto !== false) { model.mirror = hands.size > 1; model.mirrorAuto = true; }
      if (!model.examples.length) return {ok: false, name, why: 'no examples'};
      fit(model, fitBase, baseVecs, bidx);
      const g = install(model);
      recalGap = 16; nextRecal = (lastT || 0) + 15;
      send({type: 'gesture_save', name, model});
      const groups = [...new Set(model.examples.flatMap(e => e.groups || []))];
      return {ok: true, name, kind, examples: model.examples.length, counters: model.counters.length, threshold: r4(g.thr),
        fa_per_min: model.fa_per_min, quality: model.quality, why: model.why || undefined, fallback: fallback || undefined,
        segments: model.examples.map(e => ({t0: e.t, t1: e.t1, kind: e.vote, hold_s: e.hold})), groups};
    }
    function mergeHands(a, h) { const s = new Set(a || []); if (h) s.add(h); return [...s]; }
    function arr(fa) { return Array.from(fa, x => x === x ? r4(x) : null); }
    function fromArr(a) { return Float32Array.from(a, x => x == null ? NaN : x); }
    function synthBaseline(base, n) {   // draws from the baseline Gaussian (presence as Bernoulli) when he has no history yet
      let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296;
      const gauss = () => { const u = Math.max(1e-9, rnd()), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      const out = [];
      for (let k = 0; k < n; k++) {
        const v = new Float32Array(D);
        for (let i = 0; i < D; i++) v[i] = IS_PRES[i] ? (rnd() < base.mean[i] ? 1 : 0) : base.mean[i] + base.sd[i] * gauss();
        for (const s of ['R', 'L']) if (v[IDX['p_h' + s]] < 0.5) for (let i = 0; i < D; i++) if (SLOT_OF[i] === (s === 'R' ? 1 : 2)) v[i] = NaN;
        out.push(v);
      }
      return out;
    }

    // ── fitting a model: compile + threshold vs his baseline ──
    function fit(model, base, baseVecs, bidx) {
      fit1(model, base, baseVecs, bidx);
      // an example unlike the others (a wrong segment, a different move) is dropped when ≥ 3 remain, then refit
      const loo = model.loo || [];
      if (model.examples.length >= 4 && loo.length === model.examples.length) {
        const med = median(loo);
        const bad = loo.map((l, k) => model.kind === 'static' ? l < med - Math.max(6, 0.5 * Math.abs(med)) : l > Math.max(1.6 * med, med + 0.4)).map((b, k) => b ? k : -1).filter(k => k >= 0);
        if (bad.length && model.examples.length - bad.length >= 3) {
          model.dropped = (model.dropped || 0) + bad.length;
          model.examples = model.examples.filter((_, k) => !bad.includes(k));
          delete model.templates; delete model.dims; delete model.weights;
          fit1(model, base, baseVecs, bidx);
        }
      } else if (model.examples.length === 3 && loo.length === 3) {
        const med = median(loo);
        const bad = loo.map((l, k) => model.kind === 'static' ? l < med - Math.max(10, Math.abs(med)) : l > Math.max(2 * med, med + 0.6)).map((b, k) => b ? k : -1).filter(k => k >= 0);
        if (bad.length === 1) { model.dropped = (model.dropped || 0) + 1; model.examples = model.examples.filter((_, k) => k !== bad[0]); delete model.templates; delete model.dims; delete model.weights; fit1(model, base, baseVecs, bidx); }
      }
    }
    function fit1(model, base, baseVecs, bidx) {
      model.base = {mean: arr(base.mean), sd: arr(base.sd), n: base.n};
      const bTimes = bidx.map(i => H.t[i]);
      let bv = baseVecs, bt = bTimes;
      model.base_real_s = r4(spanMinutes(bTimes) * 60);
      if (bv.length < MIN_BASE_S * 18) {     // pad a short baseline with Gaussian draws (spaced like 18 Hz frames, 5 s apart episodes)
        const syn = synthBaseline(base, 900);
        bv = bv.concat(syn); const tEnd = bt.length ? bt[bt.length - 1] : 0;
        bt = bt.concat(syn.map((_, k) => tEnd + 10 + k * 0.4));   // independent draws: 0.4 s apart → no hold across draws
      }
      const minutes = Math.max(0.5, spanMinutes(bt));
      if (model.kind === 'static') fitStatic(model, base, bv, bt, minutes);
      else fitMoving(model, base, bv, bt, minutes);
      model.base_minutes = r4(minutes);
    }
    function spanMinutes(ts) {   // total covered time (gaps > 1 s count as their frames' share only)
      let s = 0; for (let i = 1; i < ts.length; i++) { const d = ts[i] - ts[i - 1]; s += d > 1 ? 0.05 : d; } return s / 60;
    }
    function compiledStatic(model, mirror) {
      const ex = model.examples.map(e => ({mean: fromArr(e.mean), sd: fromArr(e.sd)}));
      const co = model.counters.map(e => ({mean: fromArr(e.mean), sd: fromArr(e.sd)}));
      const base = {mean: fromArr(model.base.mean), sd: fromArr(model.base.sd)};
      const mask = maskOf(model.groups);
      if (!mirror) return compileStatic(ex, co, base, mask);
      const mv = e => ({mean: mirrorVec(e.mean), sd: mirrorVec(e.sd).map(Math.abs)});
      const mm = new Uint8Array(D); for (let i = 0; i < D; i++) mm[MIRROR_PERM[i]] = mask[i];
      return compileStatic(ex.map(mv), co.map(mv), {mean: mirrorVec(base.mean), sd: mirrorVec(base.sd).map(Math.abs)}, mm);
    }
    // would-be static detections on a frame series at threshold thr (hold + re-arm like the live detector)
    function countStatic(llr, ts, thr, hold, hyst) {
      let n = 0, hi = null, on = false, lo = null, lastOff = -1e9; hyst = hyst == null ? 0.5 : hyst;
      for (let j = 0; j < llr.length; j++) {
        const t = ts[j];
        if (j && t - ts[j - 1] > 1.0) { hi = null; on = false; lo = null; }
        if (llr[j] >= thr) { if (hi == null) hi = t; lo = null; } else if (llr[j] < thr - hyst) { hi = null; if (lo == null) lo = t; }
        if (!on && hi != null && t - hi >= hold && t - lastOff >= 0.4) { on = true; n++; }
        if (on && lo != null && t - lo >= 0.25) { on = false; lastOff = t; }
      }
      return n;
    }
    function fitStatic(model, base, bv, bt, minutes) {
      const durs = model.examples.map(e => e.hold || e.dur || 0.5);
      model.hold_s = r4(clamp(0.5 * median(durs), 0.15, 0.35)); model.hyst = 0.75;
      const M = compiledStatic(model, false);
      const llrB = bv.map(v => staticLLR(v, M));
      // leave-one-out scores of the examples (the median frame of each example's still stretch)
      const loo = model.examples.map((e, k) => {
        const others = model.examples.filter((_, j) => j !== k);
        const m2 = Object.assign({}, model, {examples: others.length ? others : [inflate(e)]});
        const c = compiledStatic(m2, false);
        const xs = e.samples && e.samples.length ? e.samples.map(fromArr) : [fromArr(e.mean)];
        if (!others.length) return 0.65 * median(xs.map(x => staticLLR(x, M)));   // one example: 65 % of its own fit (measured: a
        return median(xs.map(x => staticLLR(x, c)));                            // repeat 0.8–1.3× size, ±15°, moved scores 0.8–1.0)
      });
      const inS = model.examples.map(e => staticLLR(fromArr(e.mean), M));
      // the examples' worst leave-one-out score (the 2nd worst with ≥ 3 examples: one odd demonstration may not set it)
      const lsort = loo.slice().sort((a, b) => a - b), thrPos = lsort.length >= 3 ? lsort[1] : lsort[0];
      // the lowest threshold with ≤ FA_TARGET would-be detections per minute on his baseline
      const allowed = Math.floor(FA_TARGET * minutes);
      let lo = -50, hi = 200;
      if (countStatic(llrB, bt, lo, model.hold_s, model.hyst) > allowed) {
        for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (countStatic(llrB, bt, mid, model.hold_s, model.hyst) > allowed) lo = mid; else hi = mid; }
      } else hi = lo;
      const thrFa = hi + 0.5;
      let thr, quality = 'good', why;
      if (thrPos > thrFa) thr = thrFa + 0.5 * (thrPos - thrFa);
      else { thr = thrFa; quality = 'weak'; why = 'it looks a lot like your normal motion — show it a bit bigger or add more examples'; }
      // little of HIS motion seen yet (< 60 s): the false-alarm side is a guess → never below 55 % of the examples' own fit
      if ((model.base_real_s || 0) < 60) thr = Math.max(thr, 0.55 * median(inS));
      // counterexamples must stay below
      const coS = model.counters.map(c => staticLLR(fromArr(c.mean), M)); model.counter_llr = coS.map(r4);
      if (coS.length) { const cm = Math.max(...coS); if (cm + 0.5 > thr) { thr = Math.min(cm + 0.5, Math.max(thrPos, thr)); if (thr <= cm) { quality = 'weak'; why = 'the counterexample is very close to the gesture'; } } }
      model.thr = r4(thr); model.thr_learned = r4(thr); model.thr_pos = r4(thrPos); model.thr_fa = r4(thrFa);
      model.scale = r4(Math.max(0.75, Math.abs(thrPos - thrFa) / 4));
      model.loo = loo.map(r4); model.in_sample = inS.map(r4);
      model.fa_per_min = r4(countStatic(llrB, bt, thr, model.hold_s, model.hyst) / minutes);
      model.quality = quality; model.why = why;
    }
    function inflate(e) { return {mean: e.mean, sd: e.sd.map(s => s == null ? null : s * 1.5 + 0.05), n: e.n, dur: e.dur}; }
    function selectDims(model, base) {
      // motion relevance: per-dim temporal sd (in baseline sd units) inside each example, averaged
      const mot = new Float64Array(D), held = new Float64Array(D), cnt = new Float64Array(D);
      for (const ex of model.examples) {
        const c0 = ex.core ? ex.core[0] : ex.t + 0.15 * (ex.t1 - ex.t), c1 = ex.core ? ex.core[1] : ex.t1 - 0.15 * (ex.t1 - ex.t);   // the core: the active part
        let vs = ex.frames.filter(fr => fr.t >= c0 && fr.t <= c1).map(fr => fromArr(fr.v));
        if (vs.length < 4) vs = ex.frames.map(fr => fromArr(fr.v));
        const s = meanSd(vs);
        for (let i = 0; i < D; i++) {
          if (IS_PRES[i] || !(s.n[i] >= Math.max(3, 0.5 * vs.length))) continue;
          mot[i] += s.sd[i] / base.sd[i]; held[i] += Math.abs(s.mean[i] - base.mean[i]) / base.sd[i]; cnt[i]++;
        }
      }
      const n = model.examples.length, need = Math.max(1, Math.ceil(n * 0.67)), mask = maskOf(model.groups);
      const cand = []; for (let i = 0; i < D; i++) if (cnt[i] >= need && mask[i]) cand.push(i);
      const mv = cand.map(i => [mot[i] / cnt[i], i]).sort((a, b) => b[0] - a[0]);
      const top = mv.length ? mv[0][0] : 0;
      // every dim that moves (≥ 0.3 of its baseline spread, ≥ 25 % of the top one), weighted by how much (max 16)
      let sel = mv.filter(([m]) => m >= 0.25 * top && m >= 0.3).slice(0, 16);
      if (!sel.length) sel = mv.slice(0, 4);
      const dims = sel.map(x => x[1]), weights = sel.map(([m]) => r4(Math.max(0.2, m / Math.max(1e-6, top))));
      // + what is held far from his baseline while it moves (a raised, open hand while waving), weight 0.4 (max 6)
      const ctx = cand.filter(i => !dims.includes(i)).map(i => [held[i] / cnt[i], i]).filter(([h]) => h >= 2.0).sort((a, b) => b[0] - a[0]).slice(0, 6).map(x => x[1]);
      return {dims: dims.concat(ctx), weights: weights.concat(ctx.map(() => 0.4))};
    }
    function springRun(base, frames, dims, w, temps) {   // the best match cost at every 15 Hz step of a frame series
      const springs = temps.map(tp => Spring(tp.seq, w, tp.dur));
      const costs = []; let nextT = null, lastT_ = null;
      for (const fr of frames) {
        if (lastT_ != null && fr.t - lastT_ > 1.0) { springs.forEach(s => s.reset()); nextT = null; }
        lastT_ = fr.t;
        if (nextT != null && fr.t < nextT) continue;
        nextT = (nextT == null ? fr.t : nextT) + 1 / DTW_HZ; if (nextT < fr.t) nextT = fr.t + 1 / DTW_HZ;
        const x = stdz(fr.v, dims, base); let best = Infinity;
        let st = null;
        for (const s of springs) { const r = s.push(x, fr.t); if (r.cost < best) { best = r.cost; st = r.start; } }
        costs.push([fr.t, best, st]);
      }
      return costs;
    }
    // would-be detections (after a detection the live matcher restarts: costs of matches that began before it are ignored)
    function countMoving(costs, thr, refr) { let n = 0, last = -1e9; for (const [t, c, st] of costs) if (c <= thr && t - last >= refr && !(st <= last)) { n++; last = t; } return n; }
    function fitMoving(model, base, bv, bt, minutes) {
      const {dims, weights} = selectDims(model, base);
      model.dims = dims.map(i => NAMES[i]); model.weights = weights;
      const B = {mean: base.mean, sd: base.sd};
      const core = ex => { const c0 = (ex.core ? ex.core[0] : ex.t + 0.15 * (ex.t1 - ex.t)) - 0.25, c1 = (ex.core ? ex.core[1] : ex.t1 - 0.15 * (ex.t1 - ex.t)) + 0.25; const fr = ex.frames.filter(f => f.t >= c0 && f.t <= c1); return fr.length >= 4 ? fr : ex.frames; };
      const mkT = ex => { const fr = core(ex); return {seq: seqOf(fr.map(f => ({t: f.t, v: fromArr(f.v)})), dims, B).map(r => Array.from(r, x => x === x ? r4(x) : null)), dur: r4(fr[fr.length - 1].t - fr[0].t)}; };
      model.templates = model.examples.map(mkT);
      const comp = compileMoving(model);
      const durs = model.templates.map(tp => tp.dur);
      model.refr_s = r4(clamp(0.9 * median(durs), 0.6, 3));
      const bframes = bv.map((v, j) => ({t: bt[j], v}));
      const costsB = springRun(comp.base, bframes, comp.dims, comp.w, comp.temps);
      // leave-one-out: each example against the others' templates (or itself, time-stretched, when alone)
      const loo = model.examples.map((ex, k) => {
        const others = comp.temps.filter((_, j) => j !== k);
        const temps = others.length ? others : [stretch(comp.temps[k], 1.25)];
        const fr = padFrames(ex.frames.map(f => ({t: f.t, v: fromArr(f.v)})));
        const c = springRun(comp.base, fr, comp.dims, comp.w, temps);
        return Math.min(...c.map(x => x[1]));
      });
      const lf = loo.filter(isFinite).sort((a, b) => b - a);
      // one example: matching it against itself stretched says little — a real repeat costs ~0.5–1.3 (measured) → ≥ 1.0
      const thrPos = lf.length >= 3 ? lf[1] : lf.length >= 2 ? lf[0] : Math.max(1.0, lf.length ? lf[0] : 1.0);
      const allowed = Math.floor(FA_TARGET * minutes);
      const finite = costsB.map(c => c[1]).filter(isFinite).sort((a, b) => a - b);
      let lo = 0, hi = finite.length ? finite[finite.length - 1] : 5;
      if (countMoving(costsB, hi, model.refr_s) <= allowed) lo = hi;
      else for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (countMoving(costsB, mid, model.refr_s) > allowed) hi = mid; else lo = mid; }
      const thrFa = lo * 0.95;
      let thr, quality = 'good', why;
      if (thrPos < thrFa) thr = thrFa - 0.5 * (thrFa - thrPos);
      else { thr = thrFa; quality = 'weak'; why = 'it looks a lot like your normal motion — make it bigger or add more examples'; }
      if (model.counters.length) {
        const cc = model.counters.map(ex => Math.min(...springRun(comp.base, padFrames(ex.frames.map(f => ({t: f.t, v: fromArr(f.v)}))), comp.dims, comp.w, comp.temps).map(x => x[1])));
        const cm = Math.min(...cc);
        if (isFinite(cm) && thr >= cm * 0.95) { thr = cm * 0.92; if (thr < thrPos) { quality = 'weak'; why = 'the counterexample moves very much like the gesture'; } }
        model.counter_costs = cc.map(r4);
      }
      model.thr = r4(thr); model.thr_learned = r4(thr); model.thr_pos = r4(thrPos); model.thr_fa = r4(thrFa);
      model.scale = r4(Math.max(0.05, Math.abs(thrFa - thrPos) / 4)); model.loo = loo.map(r4);
      model.fa_per_min = r4(countMoving(costsB, thr, model.refr_s) / minutes);
      model.quality = quality; model.why = why;
    }
    function stretch(tp, f) { const n = Math.max(2, Math.round(tp.seq.length * f)), out = []; for (let i = 0; i < n; i++) out.push(tp.seq[Math.min(tp.seq.length - 1, Math.floor(i / f))]); return {seq: out, dur: tp.dur * f}; }
    function padFrames(fr) {   // 0.5 s of the example's own end frames on both sides, so the open-begin matcher has room
      if (!fr.length) return fr; const a = fr[0], b = fr[fr.length - 1], pre = [], post = [];
      for (let k = 8; k >= 1; k--) pre.push({t: a.t - k / 16, v: a.v});
      for (let k = 1; k <= 8; k++) post.push({t: b.t + k / 16, v: b.v});
      return pre.concat(fr, post);
    }

    // ── install / load / recalibrate ──
    function install(model) {
      const g = {model, run: {on: false, score: 0, hiSince: null, loSince: null, lastOff: -1e9, lastFire: -1e9, nextT: null}, thr: model.thr};
      if (model.base_real_s != null && model.base_real_s < 45 && !model.loaded) {   // provisional: little of his motion seen yet
        g.provisional = true;
        g.thr = model.kind === 'static' ? model.thr + 2 * (model.scale || 1) : model.thr * 0.9;
      }
      if (model.kind === 'static') { g.c = compiledStatic(model, false); if (model.mirror) g.cm = compiledStatic(model, true); }
      else {
        g.c = compileMoving(model);
        g.springs = g.c.temps.map(tp => Spring(tp.seq, g.c.w, tp.dur));
        if (model.mirror) {
          // mirrored input against the same templates: x → mirror(x) happens in step(); dims stay
          g.mirSprings = g.c.temps.map(tp => Spring(tp.seq, g.c.w, tp.dur));
        }
      }
      G.set(model.name, g);
      return g;
    }
    // RE-CHECK against his normal motion: 20 s after a gesture is learned, then every 30 → 60 s. Normal motion = the
    // buffered frames outside the learning windows and outside detections (± 1 s); the last ≤ 5 min. Too many would-be
    // detections (> faTarget / min) → the threshold rises until they stop — past the examples' own scores if need be
    // (then quality 'weak': "fires on your normal motion"). A model fit on < 45 s of his motion starts PROVISIONAL (a
    // stricter threshold) until 60 s of normal motion confirm it.
    function recalibrate() {
      const all = baselineFrames(null, null, true); if (all.length < 20 * 18) return;
      const idx = all.slice(-Math.min(all.length, 18 * 300));
      const vs = idx.map(i => H.v[i]), ts = idx.map(i => H.t[i]), minutes = Math.max(0.33, spanMinutes(ts));
      const allowed = Math.max(1, Math.floor(FA_TARGET * minutes));
      for (const [, g] of G) {
        const m = g.model;
        if (g.provisional && minutes >= 1.0) { g.provisional = false; g.thr = Math.max(g.thr, m.thr); }
        // SUSPECTS = episodes that only just cleared the threshold (peak < thr + margin): a real use of a taught gesture
        // matches it clearly, a false alarm on his ordinary motion scrapes past. Too many suspects → the threshold rises
        // above them — never above a clear match, so frequent genuine use is not punished.
        if (m.kind === 'static') {
          const llr = vs.map(v => { let l = staticLLR(v, g.c); if (g.cm) l = Math.max(l, staticLLR(mirrorVec(v), g.cm)); return l; });
          const margin = Math.max(2, 1.5 * (m.scale || 1)), peaks = [];
          let cur = null;
          for (let j = 0; j < llr.length; j++) {
            if (llr[j] >= g.thr && !(j && ts[j] - ts[j - 1] > 1)) { if (!cur) cur = {t0: ts[j], pk: -1e9}; cur.pk = Math.max(cur.pk, llr[j]); cur.t1 = ts[j]; }
            else if (cur) { if (cur.t1 - cur.t0 >= m.hold_s) peaks.push(cur.pk); cur = null; }
          }
          const sus = peaks.filter(p => p < g.thr + margin).sort((x, y) => y - x);
          if (sus.length > allowed) { g.thr = sus[allowed] + 0.25; g.recal = r4(g.thr); if (m.thr_pos != null && g.thr > m.thr_pos) g.weak = 'fires on your normal motion — show it bigger or teach it again'; }
        } else {
          const costs = springRun(g.c.base, vs.map((v, j) => ({t: ts[j], v})), g.c.dims, g.c.w, g.c.temps);
          const margin = 0.15 * g.thr, mins = []; let cur = null, last = -1e9;
          for (const [t, c] of costs) {
            if (c <= g.thr) { if (!cur) cur = {t0: t, mn: c}; cur.mn = Math.min(cur.mn, c); }
            else if (cur) { if (cur.t0 - last >= m.refr_s) { mins.push(cur.mn); last = cur.t0; } cur = null; }
          }
          const sus = mins.filter(c => c > g.thr - margin).sort((x, y) => x - y);
          if (sus.length > allowed) { g.thr = sus[allowed] - 0.01; g.recal = r4(g.thr); if (m.thr_pos != null && g.thr < m.thr_pos) g.weak = 'fires on your normal motion — make it bigger or teach it again'; }
        }
      }
    }

    function load(models) {
      let n = 0;
      for (const m of models || []) {
        try {
          if (!m || !m.name || !m.kind || !m.examples || !m.base) continue;
          const mm = remap(m); if (!mm) continue;
          install(mm); n++;
        } catch (e) { /* a broken file never takes the page down */ }
      }
      return n;
    }
    function remap(m) {     // a model saved under another layout: re-index its vectors by name
      if (!m.names || (m.names.length === D && m.names.every((x, i) => x === NAMES[i]))) return m;
      const map = NAMES.map(n => m.names.indexOf(n));
      const re = a => a ? map.map(j => j >= 0 ? a[j] : null) : a;
      const out = JSON.parse(JSON.stringify(m)); out.names = NAMES;
      out.base = {mean: re(m.base.mean), sd: re(m.base.sd), n: m.base.n};
      out.base.mean = out.base.mean.map((x, i) => x == null ? PRIOR_MEAN[i] : x); out.base.sd = out.base.sd.map((x, i) => x == null ? PRIOR_SD[i] : x);
      const ex = e => e.mean ? Object.assign({}, e, {mean: re(e.mean), sd: re(e.sd)}) : Object.assign({}, e, {frames: e.frames.map(fr => ({t: fr.t, v: re(fr.v)}))});
      out.examples = m.examples.map(ex); out.counters = (m.counters || []).map(ex);
      return out;
    }
    function onHub(msg) {
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'gestures' && Array.isArray(msg.items)) load(msg.items);
    }
    function list() {
      return [...G.values()].map(g => ({name: g.model.name, kind: g.model.kind, examples: g.model.examples.length, counters: g.model.counters.length,
        threshold: r4(g.thr), fa_per_min: g.model.fa_per_min, quality: g.weak ? 'weak' : g.model.quality, why: g.weak || g.model.why || undefined, mirror: !!g.model.mirror, hand: g.model.hand || null,
        created: g.model.created, updated: g.model.updated}));
    }
    function forget(name) {
      if (name === 'all') { const n = G.size; for (const k of [...G.keys()]) { G.delete(k); send({type: 'gesture_forget', name: k}); } return n > 0; }
      const ok = G.delete(name); if (ok) send({type: 'gesture_forget', name}); return ok;
    }
    function state(name) { const g = G.get(name); return g ? {on: !!g.run.on, score: r4(g.run.score || 0)} : {on: false, score: 0}; }
    function stats() {
      const s = ms.slice().sort((a, b) => a - b);
      return {frames: nFrames, ms_p50: r4(quant(s, 0.5) || 0), ms_p95: r4(quant(s, 0.95) || 0), ms_max: r4(s.length ? s[s.length - 1] : 0),
        baseline_s: r4(H.t.length ? H.t[H.t.length - 1] - H.t[0] : 0), gestures: G.size,
        thresholds: Object.fromEntries([...G].map(([k, g]) => [k, {thr: r4(g.thr), learned: g.model.thr_learned, recal: g.recal || null}]))};
    }
    const now = typeof performance !== 'undefined' && performance.now ? () => performance.now() : () => Date.now();
    return {frame, learn, list, forget, state, load, onHub, stats, export: name => G.get(name) ? JSON.parse(JSON.stringify(G.get(name).model)) : null,
      _H: H, _G: G, _features: feats, _debug: () => lastDebug, GROUPS: GROUP_NAMES};
  }

  return {create, Features, LAYOUT, NAMES, IDX, VERSION, mirrorVec, baselineStats, PRIOR};
});
