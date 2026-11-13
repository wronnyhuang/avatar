/* gesture_synth.cjs — scripted gestures injected into REAL recorded perception streams (perception tests).
 *
 * A kinematic hand (21 MediaPipe landmarks: palm + 4 fingers with 3 flexing segments + a thumb) posed by finger flexion
 * and a 3-D orientation, placed relative to the person's own shoulders (pose) or face, raised from rest, held or moved,
 * lowered; landmarks get MediaPipe-like jitter, the hand model's ~10 Hz sample-and-hold, missed detections (random +
 * bursts) and occasional handedness flips. Face gestures add blendshape / head-angle deltas with ramps. The base
 * stream's own hands, face and body keep moving underneath (his normal motion), so every confuser is real.
 *
 *   inject(frames, instances, {seed, dropout, burst, flip}) → new frames (deep-copied where modified)
 *   instance = {g: <gesture name in GESTURES>, t0, hold (s, static) | cycles (moving), hand: 'right'|'left', var: {...}}
 *   GESTURES[name] = {kind:'static'|'moving', part:'hand'|'hands'|'face', ...}
 */
'use strict';
const {rng} = require('./perc_lib.cjs');
const DEG = Math.PI / 180;

// canonical RIGHT hand, palm toward the camera, fingers up (hand frame: x toward his right in the mirrored view, y up,
// z toward the camera is NEGATIVE like MediaPipe's depth); units = palm length (wrist → middle MCP)
const MCP = {index: [-0.30, 0.95], middle: [-0.08, 1.0], ring: [0.13, 0.95], pinky: [0.31, 0.83]};
const SEG = {index: [0.45, 0.27, 0.22], middle: [0.50, 0.30, 0.24], ring: [0.47, 0.28, 0.22], pinky: [0.36, 0.22, 0.20]};
const FING = ['index', 'middle', 'ring', 'pinky'];
const LMI = {index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20]};
function handPoints(spec) {
  // spec: {flex: [4] 0..1, spread: rad, thumb: 'out'|'tucked'|'up'|'touch'|'side', thumbFlex}
  const P = Array.from({length: 21}, () => [0, 0, 0]);
  P[0] = [0, 0, 0];
  FING.forEach((fn, k) => {
    const [mx, my] = MCP[fn], fl = spec.flex[k];
    const spr = (k - 1.5) * (spec.spread == null ? 0.12 : spec.spread);
    let p = [mx, my, 0]; P[LMI[fn][0]] = p.slice();
    const ang = [fl * 75 * DEG, fl * 95 * DEG, fl * 65 * DEG];
    let phi = 0;
    for (let s = 0; s < 3; s++) {
      phi += ang[s];
      const L = SEG[fn][s], dir = [Math.sin(spr) * Math.cos(phi), Math.cos(spr) * Math.cos(phi), -Math.sin(phi)];
      p = [p[0] + L * dir[0], p[1] + L * dir[1], p[2] + L * dir[2]];
      P[LMI[fn][s + 1]] = p.slice();
    }
  });
  // thumb chain 1..4
  const th = spec.thumb || 'out';
  const T = {out: [[-0.22, 0.22, -0.05], [-0.48, 0.42, -0.08], [-0.66, 0.62, -0.1], [-0.80, 0.82, -0.1]],
    tucked: [[-0.22, 0.22, -0.08], [-0.34, 0.42, -0.2], [-0.22, 0.58, -0.3], [-0.05, 0.62, -0.32]],
    up: [[-0.22, 0.22, -0.05], [-0.30, 0.50, -0.25], [-0.32, 0.78, -0.35], [-0.33, 1.02, -0.42]],
    side: [[-0.22, 0.22, -0.05], [-0.52, 0.30, -0.06], [-0.78, 0.38, -0.06], [-1.0, 0.45, -0.06]]};
  if (th === 'touch') {   // thumb tip meets the index tip (OK sign)
    const it = P[8];
    T.touch = [[-0.22, 0.22, -0.06], [-0.42, 0.42, -0.15], [(it[0] - 0.42) / 2 - 0.05, (it[1] + 0.42) / 2, (it[2] - 0.15) / 2], [it[0] + 0.02, it[1] - 0.03, it[2]]];
  }
  (T[th] || T.out).forEach((p, i) => { P[i + 1] = p.slice(); });
  return P;
}
function rotate(P, roll, yaw, pitch) {   // roll about the view axis, yaw about the hand's up axis, pitch about x
  const cr = Math.cos(roll), sr = Math.sin(roll), cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return P.map(([x, y, z]) => {
    let y1 = y * cp - z * sp, z1 = y * sp + z * cp; y = y1; z = z1;            // pitch (x axis)
    let x1 = x * cy + z * sy; z1 = -x * sy + z * cy; x = x1; z = z1;          // yaw (y axis)
    x1 = x * cr - y * sr; y1 = x * sr + y * cr; return [x1, y1, z];            // roll (z axis)
  });
}

// gesture library: hand shapes / motions / face patterns. pos = palm target in his body frame (shoulder widths from the
// shoulder midpoint; x toward the hand's own side, y up). orient degrees.
const GESTURES = {
  peace: {kind: 'static', part: 'hand', shape: {flex: [0, 0, 1, 1], spread: 0.22, thumb: 'tucked'}, orient: [0, 0, 0], pos: [0.55, 0.55]},
  thumbs_up: {kind: 'static', part: 'hand', shape: {flex: [1, 1, 1, 1], thumb: 'up'}, orient: [0, 75, 0], pos: [0.5, 0.25]},
  thumbs_down: {kind: 'static', part: 'hand', shape: {flex: [1, 1, 1, 1], thumb: 'up'}, orient: [180, 75, 0], pos: [0.5, 0.15]},
  ok: {kind: 'static', part: 'hand', shape: {flex: [0.55, 0, 0.05, 0.1], thumb: 'touch', spread: 0.14}, orient: [0, 0, 0], pos: [0.5, 0.5]},
  rock: {kind: 'static', part: 'hand', shape: {flex: [0, 1, 1, 0], thumb: 'tucked', spread: 0.16}, orient: [0, 0, 0], pos: [0.55, 0.6]},
  point_up: {kind: 'static', part: 'hand', shape: {flex: [0, 1, 1, 1], thumb: 'tucked'}, orient: [0, 0, 0], pos: [0.5, 0.6]},
  fist: {kind: 'static', part: 'hand', shape: {flex: [1, 1, 1, 1], thumb: 'tucked'}, orient: [0, 0, 0], pos: [0.55, 0.55]},
  call_me: {kind: 'static', part: 'hand', shape: {flex: [1, 1, 1, 0], thumb: 'side'}, orient: [-35, 0, 0], pos: [0.45, 0.85]},
  three: {kind: 'static', part: 'hand', shape: {flex: [0, 0, 0, 1], thumb: 'tucked', spread: 0.15}, orient: [0, 0, 0], pos: [0.5, 0.55]},
  open_palm: {kind: 'static', part: 'hand', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.14}, orient: [0, 0, 0], pos: [0.55, 0.55]},
  finger_gun: {kind: 'static', part: 'hand', shape: {flex: [0, 1, 1, 1], thumb: 'up'}, orient: [90, 60, 0], pos: [0.45, 0.2]},
  wave: {kind: 'moving', part: 'hand', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.14}, orient: [0, 0, 0], pos: [0.55, 0.6], motion: 'wave', hz: 2.2},
  beckon: {kind: 'moving', part: 'hand', shape: {flex: [0, 1, 1, 1], thumb: 'tucked'}, orient: [0, 0, -60], pos: [0.35, 0.05], motion: 'beckon', hz: 1.8},
  circle: {kind: 'moving', part: 'hand', shape: {flex: [0, 1, 1, 1], thumb: 'tucked'}, orient: [0, 0, 0], pos: [0.5, 0.45], motion: 'circle', hz: 0.8},
  swipe: {kind: 'moving', part: 'hand', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.1}, orient: [0, 60, 0], pos: [0.1, 0.35], motion: 'swipe', hz: 1.6},
  chop: {kind: 'moving', part: 'hand', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.05}, orient: [0, 80, 0], pos: [0.45, 0.4], motion: 'chop', hz: 1.4},
  raise_roof: {kind: 'moving', part: 'hands', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.14}, orient: [0, 0, -70], pos: [0.7, 0.9], motion: 'pump', hz: 1.5},
  clap: {kind: 'moving', part: 'hands', shape: {flex: [0, 0, 0, 0], thumb: 'out', spread: 0.06}, orient: [0, 80, 0], pos: [0.18, 0.1], motion: 'clap', hz: 2.0},
  wink: {kind: 'static', part: 'face', blend: {eyeBlinkLeft: 0.75, eyeSquintLeft: 0.35, cheekSquintLeft: 0.3, mouthSmileLeft: 0.12}},
  kiss: {kind: 'static', part: 'face', blend: {mouthPucker: 0.75, mouthFunnel: 0.25, mouthRollLower: 0.1}},
  puff: {kind: 'static', part: 'face', blend: {cheekPuff: 0.55, mouthRollLower: 0.25, mouthPressLeft: 0.2, mouthPressRight: 0.2}},
  brows_hold: {kind: 'static', part: 'face', blend: {browInnerUp: 0.55, browOuterUpLeft: 0.5, browOuterUpRight: 0.5, eyeWideLeft: 0.15, eyeWideRight: 0.15}},
  head_tilt_rock: {kind: 'moving', part: 'face', head: 'roll', amp: 0.22, hz: 1.4},
  brow_flash: {kind: 'moving', part: 'face', blend: {browInnerUp: 0.6, browOuterUpLeft: 0.55, browOuterUpRight: 0.55}, flash: true, hz: 2.0},
};

const ease = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
function bodyRef(f) {
  const k = f.aspect || 0.5625, p = f.pose;
  if (p && p.present && p.shoulders && p.shoulders.l.v >= 0.5 && p.shoulders.r.v >= 0.5) {
    const sl = p.shoulders.l, sr = p.shoulders.r, S = Math.hypot(sr.x - sl.x, (sr.y - sl.y) * k);
    if (S > 0.05) return {O: [(sl.x + sr.x) / 2, (sl.y + sr.y) / 2 * k], S, k};
  }
  if (f.face && f.face.present && f.face.size) { const fw = f.face.size * 2; return {O: [f.face.x, f.face.y * k - 1.33 * fw], S: 2.1 * fw, k}; }
  return {O: [0, -0.5 * k], S: 0.6, k};
}
// instance timeline: static → raise 0.35 s, hold, lower 0.35; moving → raise, `cycles` cycles at hz, lower
function instDur(inst) {
  const G = GESTURES[inst.g], sp = inst.var && inst.var.speed || 1;
  if (G.kind === 'static') return 0.35 + (inst.hold || 1.0) + 0.35;
  const cyc = inst.cycles || (G.motion === 'swipe' ? 1 : G.motion === 'circle' ? 1 : 3);
  return (G.part === 'face' ? 0.15 : 0.35) * 2 + cyc / (G.hz * sp);
}

function inject(frames, instances, opt) {
  opt = opt || {};
  const R = rng(opt.seed || 7), dropout = opt.dropout == null ? 0.08 : opt.dropout, burstP = opt.burst == null ? 0.02 : opt.burst;
  const out = frames.slice();
  for (const inst of instances) {
    const G = GESTURES[inst.g]; if (!G) throw Error('unknown gesture ' + inst.g);
    const V = Object.assign({dx: 0, dy: 0, roll: 0, yaw: 0, pitch: 0, flex: 0, amp: 1, speed: 1, size: 1}, inst.var || {});
    const dur = instDur(inst), t0 = inst.t0, t1 = t0 + dur;
    inst.t1 = t1; inst.dur = dur;
    let lastHandUpd = -1, held = null, burst = 0;
    const sideSign = (inst.hand || 'right') === 'right' ? 1 : -1;
    const flexJ = [0, 1, 2, 3].map(() => V.flex * (R() * 2 - 1));
    for (let j = 0; j < out.length; j++) {
      const f0 = out[j], t = f0.t;
      if (t < t0 - 0.05 || t > t1 + 0.05) continue;
      const f = JSON.parse(JSON.stringify(f0)); out[j] = f;
      const raiseD = G.part === 'face' ? 0.15 : 0.35;
      const env = ease((t - t0) / raiseD) * (1 - ease((t - (t1 - raiseD)) / raiseD));
      const ph = Math.max(0, t - t0 - raiseD) * G.hz * V.speed * 2 * Math.PI;   // motion phase
      if (G.part === 'face') {
        if (!f.face || !f.face.present) continue;
        f.face.blend = f.face.blend || {};
        let a = env * V.amp;
        if (G.flash) a = env * V.amp * Math.max(0, Math.sin(ph));
        if (G.blend) for (const [k, d] of Object.entries(G.blend)) f.face.blend[k] = Math.min(1, (f.face.blend[k] || 0) + d * a);
        if (G.head) f.face[G.head] = (f.face[G.head] || 0) + G.amp * V.amp * env * Math.sin(ph);
        continue;
      }
      // hands
      const ref = bodyRef(f), k = ref.k;
      const hands = G.part === 'hands' ? ['right', 'left'] : [inst.hand || 'right'];
      // the hand model runs ~10 Hz: landmarks update every 0.1 s and are held in between
      const upd = t - lastHandUpd >= 0.095;
      if (upd) {
        lastHandUpd = t;
        if (burst > 0) burst--; else if (R() < burstP) burst = 2 + Math.floor(R() * 4);
        held = [];
        for (const hn of hands) {
          const sgn = hn === 'right' ? 1 : -1;
          let px = G.pos[0] * sgn + V.dx * sgn, py = G.pos[1] + V.dy, roll = (G.orient[0] + V.roll) * DEG, yaw = (G.orient[1] + V.yaw) * DEG, pitch = (G.orient[2] + V.pitch) * DEG;
          const shape = Object.assign({}, G.shape, {flex: G.shape.flex.map((x, i) => Math.max(0, Math.min(1, x + flexJ[i])))});
          const A = V.amp;
          if (G.motion === 'wave') { px += 0.22 * A * Math.sin(ph); roll += 0.25 * A * Math.sin(ph); }
          else if (G.motion === 'beckon') { const c = 0.5 - 0.5 * Math.cos(ph); shape.flex = [c, 1, 1, 1]; }
          else if (G.motion === 'circle') { px += 0.2 * A * Math.cos(ph); py += 0.2 * A * Math.sin(ph); }
          else if (G.motion === 'swipe') { px += sgn * (-0.3 + 0.9 * A * Math.min(1, ph / (2 * Math.PI))); }
          else if (G.motion === 'chop') { py += -0.35 * A * (0.5 - 0.5 * Math.cos(ph)); }
          else if (G.motion === 'pump') { py += 0.18 * A * Math.sin(ph); }
          else if (G.motion === 'clap') { px = sgn * (0.06 + 0.16 * A * (0.5 + 0.5 * Math.cos(ph))); }
          // rest: below the frame on his side
          const rest = [0.45 * sgn, -1.9];
          const bx = rest[0] + env * (px - rest[0]), by = rest[1] + env * (py - rest[1]);
          // palm size: ~0.28 shoulder widths × size
          const palm = 0.27 * ref.S * V.size;
          let P = handPoints(shape);
          if (hn === 'left') P = P.map(([x, y, z]) => [-x, y, z]);
          const r = hn === 'left' ? -1 : 1;
          P = rotate(P, roll * r, yaw * r, pitch);
          // palm centre of the posed hand (wrist + MCPs) sits at the target
          const cx = (P[0][0] + P[5][0] + P[9][0] + P[17][0]) / 4, cy = (P[0][1] + P[5][1] + P[9][1] + P[17][1]) / 4;
          const X = ref.O[0] + bx * ref.S, Y = ref.O[1] + by * ref.S;
          const lm = P.map(([x, y, z]) => {
            const ix = X + (x - cx) * palm, iy = Y + (y - cy) * palm;
            return [0.5 + ix / 2 + R.gauss() * 0.004, 0.5 - iy / (2 * k) + R.gauss() * 0.004, z * palm / 2 + R.gauss() * 0.003];
          });
          const vx = (lm[0][0] + lm[5][0] + lm[9][0] + lm[17][0]) / 4, vy = (lm[0][1] + lm[5][1] + lm[9][1] + lm[17][1]) / 4;
          const inFrame = vx > 0.02 && vx < 0.98 && vy > 0.02 && vy < 0.98;
          const detected = inFrame && burst === 0 && R() >= dropout;
          if (!detected) { held.push({hn, gone: true, wx: X, wy: Y}); continue; }
          let handed = hn; if (opt.flip && R() < opt.flip) handed = hn === 'right' ? 'left' : 'right';
          // finger extension exactly as perception.js computes it (tip / MCP distance from the wrist)
          const iso = lm.map(p => [p[0], p[1] * k, p[2]]), d3 = (a, b) => Math.hypot(iso[a][0] - iso[b][0], iso[a][1] - iso[b][1], iso[a][2] - iso[b][2]);
          const ext = [[5, 8], [9, 12], [13, 16], [17, 20]].map(([m, tp]) => Math.max(0, Math.min(1, (d3(tp, 0) / Math.max(1e-4, d3(m, 0)) - 1.15) / 0.75)));
          held.push({hn, wx: X, wy: Y, hand: {side: vx < 0.5 ? 'l' : 'r', handed, x: (vx - 0.5) * 2, y: (0.5 - vy) * 2, ax: 0, ay: 0, vx: 0, vy: 0,
            scale: Math.hypot(lm[9][0] - lm[0][0], (lm[9][1] - lm[0][1]) * k), vscale: 0, open: ext.reduce((a, b) => a + b, 0) / 4,
            gesture: 'None', gscore: 0, point: null, fore: 1, ext, main: true, id: 900 + (hn === 'right' ? 1 : 2), t, age: 0,
            lm: lm.map(p => p.map(x => Math.round(x * 1e4) / 1e4))}});
        }
      }
      if (!held) continue;
      // replace the base stream's hand(s) of the same handedness; keep the others
      const mine = new Set(held.map(h => h.hn));
      f.hands = (f.hands || []).filter(h => !(mine.has(h.handed) && env > 0.15));
      for (const h of held) if (!h.gone && env > 0.05) f.hands.push(h.hand);
      // the arm in the pose follows: wrist at the palm, elbow by 2-link IK (upper arm 0.62, forearm 0.58 shoulder widths,
      // elbow bent outward / down)
      if (f.pose && f.pose.present && f.pose.wrists) {
        for (const h of held) {
          const scr = f.pose.handed && f.pose.handed.l === h.hn ? 'l' : 'r';
          const sh = f.pose.shoulders[scr], sx = sh.x, sy = sh.y * k;
          const wx = h.wx, wy = h.wy, dx = wx - sx, dy = wy - sy, d = Math.max(1e-3, Math.hypot(dx, dy));
          const L1 = 0.62 * ref.S, L2 = 0.58 * ref.S, dd = Math.min(d, (L1 + L2) * 0.999);
          const a = (L1 * L1 - L2 * L2 + dd * dd) / (2 * dd), hgt = Math.sqrt(Math.max(0, L1 * L1 - a * a));
          const ux = dx / d, uy = dy / d, out_ = (scr === 'l' ? -1 : 1);
          let ex = sx + a * ux + hgt * -uy, ey = sy + a * uy + hgt * ux;
          const ex2 = sx + a * ux - hgt * -uy, ey2 = sy + a * uy - hgt * ux;
          if ((ex2 - sx) * out_ - ey2 > (ex - sx) * out_ - ey) { ex = ex2; ey = ey2; }   // the outward / lower solution
          f.pose.wrists[scr] = {x: Math.round(wx * 1e4) / 1e4, y: Math.round(wy / k * 1e4) / 1e4, v: 0.9};
          f.pose.elbows[scr] = {x: Math.round(ex * 1e4) / 1e4, y: Math.round(ey / k * 1e4) / 1e4, v: 0.85};
        }
      }
    }
  }
  return out;
}
// a variation draw for a repeat (held-out repeats differ in place, angle, finger curl, speed, size)
function variation(R, scale) {
  scale = scale == null ? 1 : scale;
  return {dx: (R() * 2 - 1) * 0.12 * scale, dy: (R() * 2 - 1) * 0.12 * scale, roll: (R() * 2 - 1) * 12 * scale, yaw: (R() * 2 - 1) * 12 * scale,
    pitch: (R() * 2 - 1) * 10 * scale, flex: 0.1 * scale, amp: 1 + (R() * 2 - 1) * 0.2 * scale, speed: 1 + (R() * 2 - 1) * 0.25 * scale, size: 1 + (R() * 2 - 1) * 0.1 * scale};
}
// a clean hands-only "normal motion" stream (no face, no pose): his hand drifts around relaxed — curls, opens, tilts,
// leaves and re-enters the frame — at `fps`; landmarks exactly as perception.js emits them (lm 21×[vx,vy,z], ext[4]).
function idleHandStream(T, fps, seed) {
  const R = rng(seed || 1), out = [], k = 0.75;
  let x = 0.2, y = -0.2, rollv = 0, fl = [0.3, 0.3, 0.3, 0.3], curl = 0.3, off = [0, 0, 0, 0], present = true, nextFlip = 6 + R() * 10;
  for (let i = 0; i < T * fps; i++) {
    const t = i / fps;
    if (t > nextFlip) { present = !present; nextFlip = t + (present ? 6 + R() * 12 : 1 + R() * 3); }
    x = Math.max(-0.7, Math.min(0.7, x + (R() - 0.5) * 0.04)); y = Math.max(-0.7, Math.min(0.5, y + (R() - 0.5) * 0.04));
    rollv = Math.max(-0.6, Math.min(0.6, rollv + (R() - 0.5) * 0.05));
    // relaxed hands curl their fingers TOGETHER (one shared curl + small per-finger offsets), not independently
    curl = Math.max(0.05, Math.min(0.8, curl + (R() - 0.5) * 0.05)); off = off.map(v => Math.max(-0.12, Math.min(0.12, v + (R() - 0.5) * 0.02)));
    fl = off.map((o, i) => Math.max(0, Math.min(0.9, curl + o + 0.05 * i)));
    const f = {t: Math.round(t * 1e4) / 1e4, face: {present: false}, hands: [], pose: null, motion: 0.05, aspect: k, events: []};
    if (present) {
      let P = handPoints({flex: fl, thumb: 'out', spread: 0.1});
      P = rotate(P, rollv, 0.3 * Math.sin(t / 3), 0);
      const palm = 0.17, cx = (P[0][0] + P[5][0] + P[9][0] + P[17][0]) / 4, cy = (P[0][1] + P[5][1] + P[9][1] + P[17][1]) / 4;
      const lm = P.map(([px, py, pz]) => [Math.round((0.5 + (x + (px - cx) * palm) / 2) * 1e4) / 1e4, Math.round((0.5 - (y + (py - cy) * palm) / (2 * k)) * 1e4) / 1e4, Math.round(pz * palm / 2 * 1e4) / 1e4]);
      const iso = lm.map(p => [p[0], p[1] * k, p[2]]), d3 = (a, b) => Math.hypot(iso[a][0] - iso[b][0], iso[a][1] - iso[b][1], iso[a][2] - iso[b][2]);
      const ext = [[5, 8], [9, 12], [13, 16], [17, 20]].map(([m, tp]) => Math.max(0, Math.min(1, (d3(tp, 0) / Math.max(1e-4, d3(m, 0)) - 1.15) / 0.75)));
      f.hands.push({side: x < 0 ? 'l' : 'r', handed: 'right', x, y, scale: 0.08, open: ext.reduce((a, b) => a + b) / 4, ext, gesture: 'None', gscore: 0, main: true, id: 1, t, lm});
    }
    out.push(f);
  }
  return out;
}
module.exports = {GESTURES, inject, instDur, variation, handPoints, rotate, idleHandStream};
