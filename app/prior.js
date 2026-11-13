/* prior.js — the learned causal motion prior ("prior"). Pure JS, no dependencies.
 *
 *   const p = createPrior(weights);           // weights = JSON.parse(app/prior_weights.json)
 *   const out = p.step(frame);                // call once per 30 Hz frame
 *   // out = {head_yaw, head_pitch, head_roll, head_x, head_y, head_z, gaze_x, gaze_y, blink_l, blink_r,
 *   //        brow_l, brow_r, smile, squint, eye_wide, _arm, _human:{...}, _model}
 *
 * frame (every field optional; missing = silence / no face):
 *   self:    {rms, pitch, voiced, speaking}  the AVATAR's own voice (TTS): rms linear full-scale 0..1,
 *            pitch Hz (0 = unvoiced), speaking = the TTS is in an utterance (if omitted: VAD on rms)
 *   partner: {rms, pitch, voiced, speaking}  the USER's mic, same units
 *   face:    the USER's face from MediaPipe: {present, blend: {name: score} | number[52] (category order),
 *            matrix: number[16] (facialTransformationMatrixes[0].data) | yaw, pitch (deg, see docs/PRIOR.md)}
 *   hands:   optional user arm energy: {speed} wrist speed in shoulder widths / s, or [{x,y}] wrists (normalized)
 *
 * Everything the network sees at step t is available at t (strictly causal). Docs: docs/PRIOR.md.
 * Works as a classic <script> (defines globalThis.createPrior), an ES-module side-effect import
 * (`import './prior.js'; const {createPrior} = globalThis;`) and CommonJS (node tests).
 */
(function (root) {
  'use strict';
  const BN = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
    'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
    'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'];
  const DEG = Math.PI / 180;
  const clip = (x, a, b) => (x < a ? a : x > b ? b : x);

  // ------------------------------------------------------------------ small seedable RNG
  function rng(seed) {
    let s = (seed >>> 0) || 0x9e3779b9;
    const u = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    const n = () => { if (spare !== null) { const v = spare; spare = null; return v; }
      let a = 0, b = 0; while (a === 0) a = u(); b = u();
      const r = Math.sqrt(-2 * Math.log(a)); spare = r * Math.sin(2 * Math.PI * b); return r * Math.cos(2 * Math.PI * b); };
    return { u, n };
  }

  // ------------------------------------------------------------------ the network (GRU + MDN head)
  function tensor(a) {                  // number[] or "b64f32:<base64 little-endian float32>"
    if (typeof a !== 'string') return Float32Array.from(a);
    const b = a.slice(7);
    let bytes;
    if (typeof atob === 'function') { const s = atob(b); bytes = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i); }
    else bytes = Uint8Array.from(Buffer.from(b, 'base64'));
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), out = new Float32Array(bytes.byteLength / 4);
    for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(4 * i, true);
    return out;
  }

  function makeNet(W) {
    const A = W.arch, C = A.C || 1, D = A.D || 0, DO = D * C, H = A.H, K = A.K || 0, E = A.E, I = A.nin,
      NO = A.type === 'gru-reaction' ? 8 : K + 2 * K * DO + 1 + (A.nod ? 1 : 0);
    const f = tensor;
    const inW = f(W.inp_w), inB = f(W.inp_b), wih = f(W.w_ih), whh = f(W.w_hh), bih = f(W.b_ih), bhh = f(W.b_hh);
    const oW = f(W.out_w), oB = f(W.out_b);
    const z = new Float32Array(E), gi = new Float32Array(3 * H), gh = new Float32Array(3 * H), o = new Float32Array(NO);
    function forward(x, h) {          // x: Float32Array(I), h: Float32Array(H) updated in place; returns raw head
      for (let i = 0; i < E; i++) { let s = inB[i]; const r = i * I; for (let j = 0; j < I; j++) s += inW[r + j] * x[j]; z[i] = Math.tanh(s); }
      for (let i = 0; i < 3 * H; i++) {
        let s = bih[i]; const r = i * E; for (let j = 0; j < E; j++) s += wih[r + j] * z[j]; gi[i] = s;
        let t = bhh[i]; const q = i * H; for (let j = 0; j < H; j++) t += whh[q + j] * h[j]; gh[i] = t;
      }
      for (let i = 0; i < H; i++) {     // torch GRU gate order: r, z, n
        const r = 1 / (1 + Math.exp(-(gi[i] + gh[i])));
        const u = 1 / (1 + Math.exp(-(gi[H + i] + gh[H + i])));
        const n = Math.tanh(gi[2 * H + i] + r * gh[2 * H + i]);
        h[i] = (1 - u) * n + u * h[i];
      }
      for (let i = 0; i < NO; i++) { let s = oB[i]; const r = i * H; for (let j = 0; j < H; j++) s += oW[r + j] * h[j]; o[i] = s; }
      return o;
    }
    return { forward, D, H, K, I, NO, C, DO, nod: !!A.nod };
  }

  // ------------------------------------------------------------------ live feature computation
  function voiceTracker(f0init) {
    let peak = -20, floor = -60, hang = 0, lf0 = Math.log2(f0init), nf0 = 0;
    return function (v) {
      v = v || {};
      const db = 20 * Math.log10(Math.max(v.rms || 0, 1e-5));
      if (db > peak) peak = 0.7 * peak + 0.3 * db; else peak = Math.max(peak - 0.02, -45);   // ~0.6 dB/s release
      if (db < floor) floor = db; else floor += 0.01;
      const fl = Math.max(floor, peak - 30);
      const level = clip((db - fl) / Math.max(peak - fl, 6), 0, 1.2);
      const pitch = v.pitch || 0;
      const voiced = (v.voiced !== undefined ? !!v.voiced : pitch > 0) && pitch > 0;
      let speaking;
      if (v.speaking !== undefined) speaking = !!v.speaking;
      else { if (level > 0.45 || (voiced && level > 0.25)) hang = 9; else hang--; speaking = hang > 0; }
      let semi = 0;
      if (voiced && speaking) {
        const l = Math.log2(pitch); nf0++; const a = Math.max(0.005, 1 / nf0);
        lf0 += a * (l - lf0); semi = clip(l - lf0, -1, 1);       // semitones/12 == log2 ratio
      }
      const g = speaking ? 1 : 0;
      return [level * g, semi * g, (voiced ? 1 : 0) * g, g];
    };
  }

  function poseFromMatrix(m) {        // MediaPipe 4x4; detects row- vs column-major by where the translation sits
    const colMajor = Math.abs(m[14]) > Math.abs(m[11]);
    const R = colMajor ? (r, c) => m[c * 4 + r] : (r, c) => m[r * 4 + c];
    const yaw = Math.atan2(R(0, 2), R(2, 2)) / DEG;
    const pitch = Math.atan2(-R(1, 2), Math.hypot(R(0, 2), R(2, 2))) / DEG;
    const roll = Math.atan2(R(0, 1), R(1, 1)) / DEG;
    return { yaw, pitch, roll };
  }

  function faceTracker() {
    let sp = null, sy = null, pp = 0, py = 0, mp = 0, my = 0, n = 0, armE = 0, lastW = null;
    const a6 = 1 - Math.exp(-2 * Math.PI * 6 / 30);       // ~6 Hz one-pole, like the training low-pass
    return function (face, hands) {
      if (!face || !face.present) { sp = null; return [0, 0, 0, 0, 0, 0, 0, 0, 0]; }
      const b = face.blend || {};
      const g = Array.isArray(b) || ArrayBuffer.isView(b) ? (k) => b[BN.indexOf(k)] || 0 : (k) => b[k] || 0;
      const smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2;
      const jaw = g('jawOpen');
      const brow = 0.25 * (g('browOuterUpLeft') + g('browOuterUpRight')) + 0.5 * g('browInnerUp') -
        0.5 * (g('browDownLeft') + g('browDownRight'));
      let yaw = 0, pitch = 0;
      if (face.matrix && face.matrix.length >= 16) ({ yaw, pitch } = poseFromMatrix(face.matrix));
      else { yaw = face.yaw || 0; pitch = face.pitch || 0; }
      n++; const am = Math.max(1 / 600, 1 / n);                // running neutral (~20 s), like the clip median
      mp += am * (pitch - mp); my += am * (yaw - my);
      const p = pitch - mp, y = yaw - my;
      if (sp === null) { sp = p; sy = y; pp = p; py = y; }
      sp += a6 * (p - sp); sy += a6 * (y - sy);
      const vp = (sp - pp) * 30, vy = (sy - py) * 30; pp = sp; py = sy;
      let arm = 0;
      if (hands && typeof hands.speed === 'number') { armE += (hands.speed - armE) / 30; arm = Math.log1p(Math.max(armE, 0)); }
      else if (Array.isArray(hands) && hands.length) {
        const w = hands.map((h) => [h.x || 0, h.y || 0]);
        if (lastW && lastW.length === w.length) {
          let s = 0; for (let i = 0; i < w.length; i++) s += Math.hypot(w[i][0] - lastW[i][0], w[i][1] - lastW[i][1]);
          armE += ((s / w.length) * 30 * 4 - armE) / 30;        // normalized image units -> ~shoulder widths
        }
        lastW = w; arm = Math.log1p(Math.max(armE, 0));
      }
      return [1, smile, jaw, brow, clip(vp / 30, -3, 3), clip(vy / 30, -3, 3), sp / 10, sy / 20, arm];
    };
  }

  // ------------------------------------------------------------------ the prior
  function createPrior(weights, opts) {
    const dflt = weights.defaults || {};
    opts = Object.assign({ temp: dflt.temp !== undefined ? dflt.temp : 1.0, speakHead: dflt.speakHead || 0,
      idleHead: dflt.idleHead || 0, blinkSaccade: dflt.blinkSaccade || 1, smileHz: dflt.smileHz || 0,
      leak: dflt.leak !== undefined ? dflt.leak : 0, anchor: dflt.anchor || 0, smoothHz: dflt.smoothHz || 0,
      seed: 1, gainHead: 1.3, gainGaze: 1.0, gainBrow: 1.3, gainTrans: 0.5,
      gazeDegPerUnit: 30, sceneUnitsPerCm: 0.085, maxNorm: 5, mode: 'sample', maxHeadRad: 0.5, maxTrans: 0.25,
      nodGain: dflt.nodGain !== undefined ? dflt.nodGain : 1.0,
      react: dflt.react || { smile: 1.0, nod: 1.0, gaze: 1.0, energy: 0.5 } }, opts || {});
    const models = (weights.models || [weights]).map((W) => ({ W, net: makeNet(W), h: new Float32Array(W.arch.H),
      role: (W.meta && W.meta.role) || 'general' }));
    // the dyadic reaction model: partner-only inputs -> expected reaction (smile, head energy, nod, aversion, ...)
    const RX = weights.reaction ? { W: weights.reaction, net: makeNet(weights.reaction), h: new Float32Array(weights.reaction.arch.H) } : null;
    const rxIn = new Float32Array(14); let rxOut = null;
    const rxBase = RX ? RX.W.meta.eval.base_rates : null;
    const W0 = models[0].W, D = W0.arch.D, OUT = W0.arch.out;
    const nm = W0.norm, posC = nm.pos_c, posS = nm.pos_s, dS = nm.d_s;
    const perCh = (v) => (Array.isArray(v) ? Float32Array.from(v) : new Float32Array(D).fill(v || 0));
    const tempV = perCh(opts.temp), leakV = perCh(opts.leak);
    const aSm = opts.smoothHz > 0 ? 1 - Math.exp(-2 * Math.PI * opts.smoothHz / 30) : 1;
    const aSmile = opts.smileHz > 0 ? 1 - Math.exp(-2 * Math.PI * opts.smileHz / 30) : 1;
    const SMOOTH = ['yaw', 'pitch', 'roll', 'tx', 'ty', 'tz', 'smile'];
    let hs = null;                                                 // output-smoothed human-unit channels
    const R = rng(opts.seed);
    const selfV = voiceTracker(180), partV = voiceTracker(140), faceF = faceTracker();
    const y = new Float32Array(D); let clock = 1.0, blinkT = -1, frameN = 0, active = 0;
    const x = new Float32Array(models[0].net.I);
    const BLINK = [0.45, 0.92, 1.0, 1.0, 0.72, 0.42, 0.18, 0.05];      // 8 frames = 267 ms, fast close, slow open
    const idx = {}; OUT.forEach((n, i) => { idx[n] = i; });
    const neutral = {};                                                  // human-units neutral for level channels
    OUT.forEach((n, i) => { neutral[n] = posC[i]; });

    function pick(ext) {                  // reaction model while the avatar is not speaking and a face is present
      if (models.length < 2) return 0;
      const speaking = ext[3] > 0, face = ext[8] > 0;
      const k = models.findIndex((m) => m.role === 'listener');
      return (!speaking && face && k >= 0) ? k : models.findIndex((m) => m.role !== 'listener');
    }

    function step(frame) {
      frame = frame || {};
      const ext = selfV(frame.self).concat(partV(frame.partner), faceF(frame.face, frame.hands));
      return stepExt(ext);
    }

    const C = models[0].net.C, DO = D * C, K = W0.arch.K;
    const queue = [];                      // pending per-frame deltas (normalized) from the last network step
    let extHist = [];                      // the last C feature frames (the step input is their mean)
    let pendingBlink = false;

    function netStep() {
      const n = extHist.length;
      for (let i = 0; i < 17; i++) { let s = 0; for (let j = 0; j < n; j++) s += extHist[j][i]; x[i] = s / n; }
      for (let i = 0; i < D; i++) x[17 + i] = y[i];
      x[17 + D] = 1; x[18 + D] = clock;
      active = pick(extHist[n - 1]);
      let o = null;
      for (let m = 0; m < models.length; m++) {     // every model keeps its state warm; the active one drives
        const r = models[m].net.forward(x, models[m].h);
        if (m === active) o = Float32Array.from(r);
      }
      let k = 0;
      if (opts.mode === 'mean') { for (let i = 1; i < K; i++) if (o[i] > o[k]) k = i; }
      else {
        let mx = -1e9; for (let i = 0; i < K; i++) mx = Math.max(mx, o[i]);
        let s = 0; const p = []; for (let i = 0; i < K; i++) { p.push(Math.exp(o[i] - mx)); s += p[i]; }
        let u = R.u() * s; k = K - 1; for (let i = 0; i < K; i++) { u -= p[i]; if (u <= 0) { k = i; break; } }
      }
      const spk = x[3] > 0.5, idle = !spk && x[7] < 0.5;     // nobody speaking -> calmer head
      // reaction modulation (only with a visible partner): head energy scales the head temperature
      let eMul = 1;
      if (rxOut && rxOut.present && opts.mode !== 'mean') {
        const base = Math.expm1(RX.W.norm.ymu[1]);
        eMul = clip(Math.pow(rxOut.head_speed / Math.max(base, 0.5), opts.react.energy), 0.6, 1.6);
      }
      for (let c = 0; c < C; c++) {
        const dl = new Float32Array(D);
        for (let d = 0; d < D; d++) {
          const j = c * D + d;
          const mu = o[K + k * DO + j];
          const ls = clip(o[K + K * DO + k * DO + j], -5, 2);
          let tp = tempV[d];
          if (d < 6 && idle && opts.idleHead > 0) tp = opts.idleHead;
          if (d < 6 && spk && opts.speakHead > 0) tp = opts.speakHead;
          tp *= d < 6 ? eMul : 1;
          dl[d] = opts.mode === 'mean' ? mu : mu + tp * Math.exp(ls) * R.n();
        }
        queue.push(dl);
      }
      let pb = 1 / (1 + Math.exp(-o[K + 2 * K * DO]));
      if (opts.blinkSaccade !== 1) {        // blinks ride on gaze shifts: world-gaze jump >= 8 deg in this chunk
        let jx = 0, jy = 0;
        for (const dl of queue) { jx += dl[6] * dS[6] * posS[6] + dl[0] * dS[0] * posS[0]; jy += dl[7] * dS[7] * posS[7] - dl[1] * dS[1] * posS[1]; }
        if (Math.hypot(jx, jy) >= 8) pb = Math.min(pb * opts.blinkSaccade, 0.95);
      }
      const blinkNow = opts.mode === 'mean' ? pb > 0.5 : R.u() < pb;
      if (models[active].net.nod && opts.mode !== 'mean') {        // explicit big nod (backchannel / beat)
        let pn = opts.nodGain / (1 + Math.exp(-o[K + 2 * K * DO + 1]));
        if (rxOut && rxOut.present) pn *= Math.pow(clip(rxOut.bignod / Math.max(rxBase.bignod, 1e-3), 0.3, 3), opts.react.nod);
        if (R.u() < pn && (nodT < 0 || nodT > 18)) {
          nodT = 0; nodA = clip(Math.exp(Math.log(6) + 0.3 * R.n()), 4, 12); nodDouble = R.u() < 0.3;
        }
      }
      if (blinkNow && (blinkT < 0 || blinkT >= 6)) { pendingBlink = true; clock = 0; }
      else clock = Math.min(clock + C / 30 / 4, 1);
      lastRaw = o;
    }
    let lastRaw = null, nodT = -1, nodA = 0, nodDouble = false, nodOff = 0;
    function nodTemplate(k) {             // deg, + = nose down: a 0.45 s dip, sometimes a smaller second one
      const T1 = 14;
      if (k < T1) return nodA * Math.sin(Math.PI * k / T1);
      if (nodDouble && k < 2 * T1) return 0.6 * nodA * Math.sin(Math.PI * (k - T1) / T1);
      return 0;
    }

    function rxStep(ext) {                // partner face (9) + partner voice (4) + self speaking (1)
      for (let i = 0; i < 9; i++) rxIn[i] = ext[8 + i];
      for (let i = 0; i < 4; i++) rxIn[9 + i] = ext[4 + i];
      rxIn[13] = ext[3];
      const o = RX.net.forward(rxIn, RX.h), nm = RX.W.norm, sg = (v) => 1 / (1 + Math.exp(-v));
      rxOut = { smile_rel: o[0] * nm.ysd[0] + nm.ymu[0], head_speed: Math.expm1(o[1] * nm.ysd[1] + nm.ymu[1]),
        bignod: sg(o[4]), avert: sg(o[5]), brow: sg(o[6]), smile_rise: sg(o[7]), present: ext[8] > 0 };
    }

    function stepExt(ext) {               // ext = the 17 model features (sv 4, pv 4, pf 9) — tests feed these directly
      if (RX) rxStep(ext);
      extHist.push(ext.slice ? ext.slice() : Array.from(ext));
      if (extHist.length > C) extHist.shift();
      if (queue.length === 0) netStep();
      const dl = queue.shift();
      for (let d = 0; d < D; d++) y[d] = clip(y[d] * (1 - leakV[d]) + dl[d] * dS[d], -opts.maxNorm, opts.maxNorm);
      if (opts.anchor > 0) {                // pull WORLD gaze (eye + head) toward the partner, through the eyes
        let an = opts.anchor;
        if (rxOut && rxOut.present) an *= clip(Math.pow((1 - rxOut.avert) / Math.max(1 - rxBase.avert, 0.05), 2 * opts.react.gaze), 0.3, 2);
        const wx = y[6] * posS[6] + y[0] * posS[0], wy = y[7] * posS[7] - y[1] * posS[1];
        y[6] -= an * wx / posS[6]; y[7] -= an * wy / posS[7];
      }
      let bl = 0;
      if (pendingBlink) { pendingBlink = false; blinkT = 0; }
      nodOff = 0;
      if (nodT >= 0) { nodOff = nodTemplate(nodT); nodT++; if (nodT > 30) nodT = -1; }
      if (blinkT >= 0) { bl = BLINK[blinkT] || 0; blinkT++; if (blinkT >= BLINK.length) blinkT = -1; }
      frameN++;
      return toRig(bl, ext);
    }

    function human() { const h = {}; OUT.forEach((n, i) => { h[n] = y[i] * posS[i] + posC[i]; }); return h; }

    function toRig(bl, ext) {
      const h = human();
      h.pitch += nodOff; h.nod = nodOff;
      if (rxOut && rxOut.present) h.smile += opts.react.smile * Math.max(-0.1, rxOut.smile_rel);   // mirroring
      // output smoothing AFTER the nod / mirroring layers (exactly what eval_prior.py measured)
      if (!hs) { hs = {}; SMOOTH.forEach((k) => { hs[k] = h[k]; }); }
      SMOOTH.forEach((k) => { hs[k] += (k === 'smile' ? aSmile : aSm) * (h[k] - hs[k]); h[k] = hs[k]; });
      const gH = opts.gainHead * DEG, gT = opts.gainTrans * opts.sceneUnitsPerCm, gG = opts.gainGaze / opts.gazeDegPerUnit;
      const soft = (v, L) => L * Math.tanh(v / L);       // soft limits: no flat-topped clamps
      return {
        head_yaw: soft(h.yaw * gH, opts.maxHeadRad), head_pitch: soft(h.pitch * gH, opts.maxHeadRad),
        head_roll: soft(h.roll * gH, opts.maxHeadRad * 0.7),
        head_x: soft(h.tx * gT, opts.maxTrans), head_y: soft(h.ty * gT, opts.maxTrans), head_z: soft(h.tz * gT, opts.maxTrans),
        gaze_x: clip(h.gx * gG, -1, 1), gaze_y: clip(h.gy * gG, -1, 1),
        blink_l: bl, blink_r: bl,
        brow_l: clip((h.brow_l - neutral.brow_l) * opts.gainBrow, -1, 1),
        brow_r: clip((h.brow_r - neutral.brow_r) * opts.gainBrow, -1, 1),
        smile: clip(h.smile, 0, 1),
        squint: clip(h.squint - neutral.squint, 0, 1), eye_wide: clip(h.wide - neutral.wide, 0, 1),
        _arm: clip(Math.expm1(h.arm) / 2, 0, 1),
        _human: h, _model: models[active].role, _features: ext, _reaction: rxOut,
      };
    }

    function reset() { y.fill(0); hs = null; nodT = -1; nodOff = 0; clock = 1; blinkT = -1; queue.length = 0; extHist = []; pendingBlink = false;
      models.forEach((m) => m.h.fill(0)); if (RX) { RX.h.fill(0); rxOut = null; } }
    // raw network access for tests: forward one input vector through model m (state kept per model)
    function _forward(xin, m) { return Array.from(models[m || 0].net.forward(Float32Array.from(xin), models[m || 0].h)); }
    function _rxForward(xin) { return Array.from(RX.net.forward(Float32Array.from(xin), RX.h)); }
    return { step, stepExt, reset, human, _forward, _rxForward, _models: models, _y: y, outputs: OUT, get lastRaw() { return lastRaw; },
      get active() { return models[active].role; } };
  }

  root.createPrior = createPrior;
  root.priorPoseFromMatrix = poseFromMatrix;
  if (typeof module !== 'undefined' && module.exports) module.exports = { createPrior, poseFromMatrix };
})(typeof globalThis !== 'undefined' ? globalThis : this);
