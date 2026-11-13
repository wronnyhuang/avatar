/* react_feat.js — page-side canonical "human numbers" for the learned reaction model.
 *
 * window.AvatarReactFeat (browser) / module.exports (node). No dependencies.
 * A faithful port of react/spec.py (the ONE definition) as applied per frame by the dataset path
 * (mp_raw_to_canon in data/seamless_lib.py). Coordinates = the person's OWN camera, UNMIRRORED: feed the RAW
 * MediaPipe worker results (vision/percept_worker.js), never the mirrored / One-Euro-filtered perception values.
 *
 *   BS_NAMES (52, spec order), MOTION_COLS (86), D_MOTION, MASK_COLS
 *   eulerFromMat(m16RowMajor)         -> {rot:[yaw,pitch,roll] rad, t:[x,y,z] cm}        (spec.euler_from_mat)
 *   colToRow(m16)                     -> Float64Array(16) row-major (MediaPipe JS matrix .data is COLUMN-major)
 *   bsToSpec(bs, names)               -> Float32Array(52) in spec order, mapped by category NAME
 *   gazeFromBs(bs52)                  -> [gx, gy]                                           (spec.gaze_from_bs)
 *   bodyFromPose(lm33x4|null, w, h, visMin=0.5) -> {body: Float32Array(18), ok}            (spec.body_from_pose)
 *   handDescriptor(lm21x3Pixels)      -> [open, thumb, point, palm]                        (spec.hand_descriptor)
 *   handDescriptorSide(lm, isLeft)    -> same, palm sign flipped for the right hand         (spec.hand_descriptor_side)
 *   assignHandsToWrists(handWristsXY, poseWristL, poseWristR) -> {l, r} (index or null)   (spec.assign_hands_to_wrists)
 *   canonFrame({face:{bs (spec order), matColMajor}|null, pose:{lm 33*4 normalized x,y,z,vis}|null,
 *               hands:[{lm: 63 normalized x,y,z}], w, h})
 *        -> {motion: Float32Array(86), mask: Uint8Array(4), tcm: [x,y,z] cm | null (face absent)}
 *        motion cols 55-57 = the RAW MediaPipe head translation in cm (NOT median-subtracted, NOT /10: the server applies
 *        the running median, spec.head_translation_feature). Everything else exactly as the spec / mp_raw_to_canon:
 *        face absent -> cols 0-59 = 0, mask[0] = 0; pose absent -> body = the spec's rest pose (side +1), mask[1] = 0;
 *        hands only while a pose is present (assigned to the POSE wrists 15/16, never by MediaPipe handedness), at most
 *        the first two hands; a face with no matrix or no blendshapes counts as absent.
 *   packHuman(t, motion, mask) -> Uint8Array(184)   uplink message 'H' (little-endian):
 *        header <BBHd> (12 B): 0x48, flags (bit0 face, bit1 body, bit2 hand_l, bit3 hand_r), uint16 n = 86, float64 t (s);
 *        then 86 x int16 in MOTION_COLS order = round(value * 1e4) clamped to +-32767, EXCEPT cols 55-57 (head
 *        translation, cm) = round(cm / 100 * 1e4) (metres * 1e4) and cols 60-77 (body, shoulder widths) = round(v * 1e3).
 *        Non-finite values encode as 0.
 *   unpackHuman(bytes) -> {t, flags, n, mask: Uint8Array(4), motion: Float32Array(86) (cols 55-57 back in cm)}
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarReactFeat = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const BS_NAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
    'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'];
  const BS = {};
  BS_NAMES.forEach((n, i) => { BS[n] = i; });
  const BODY_JOINTS = ['shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r'];
  const POSE_IDX = [11, 12, 13, 14, 15, 16];
  const MOTION_COLS = BS_NAMES.map(n => 'bs_' + n)
    .concat(['head_yaw', 'head_pitch', 'head_roll', 'head_tx', 'head_ty', 'head_tz', 'gaze_x', 'gaze_y'])
    .concat([].concat(...BODY_JOINTS.map(j => ['x', 'y', 'z'].map(a => j + '_' + a))))
    .concat([].concat(...['l', 'r'].map(s => ['open', 'thumb', 'point', 'palm'].map(k => 'hand_' + s + '_' + k))));
  const D_MOTION = 86, D_MASK = 4;
  if (MOTION_COLS.length !== D_MOTION) throw Error('react_feat: MOTION_COLS != 86');
  const MASK_COLS = ['face', 'body', 'hand_l', 'hand_r'];
  const SL = {bs: 0, headRot: 52, headT: 55, gaze: 58, body: 60, handL: 78, handR: 82};
  const CM_COLS = [55, 56, 57];
  // the spec's neutral "arm hanging" guess (x sign follows the image side of the person's left shoulder)
  const REST = [[0.5, 0, 0], [-0.5, 0, 0], [0.6, -1.0, 0], [-0.6, -1.0, 0], [0.6, -1.9, 0.1], [-0.6, -1.9, 0.1]];
  const clip01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
  const norm3 = (x, y, z) => Math.sqrt(x * x + y * y + z * z);

  function colToRow(m) {
    const o = new Float64Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) o[4 * i + j] = m[4 * j + i];
    return o;
  }

  // spec.euler_from_mat (row-major 4x4): yaw = atan2(R02, R22), pitch = atan2(-R12, hypot(R02, R22)),
  // roll = atan2(R01, R11); translation = column 3 (cm)
  function eulerFromMat(m) {
    const r02 = +m[2], r22 = +m[10], r12 = +m[6];
    return {rot: [Math.atan2(r02, r22), Math.atan2(-r12, Math.hypot(r02, r22)), Math.atan2(+m[1], +m[5])],
      t: [+m[3], +m[7], +m[11]]};
  }

  let mapNames = null, mapIdx = null;
  function bsToSpec(bs, names) {
    const out = new Float32Array(52);
    if (!bs) return out;
    if (!names) { for (let i = 0; i < 52 && i < bs.length; i++) out[i] = bs[i]; return out; }
    if (names !== mapNames) {
      const at = {};
      for (let i = 0; i < names.length; i++) at[names[i]] = i;
      mapIdx = BS_NAMES.map(n => at[n] == null ? -1 : at[n]); mapNames = names;
    }
    for (let i = 0; i < 52; i++) { const k = mapIdx[i]; out[i] = k >= 0 && k < bs.length ? bs[k] : 0; }
    return out;
  }

  function gazeFromBs(b) {
    const gx = 0.5 * ((b[BS.eyeLookOutLeft] - b[BS.eyeLookInLeft]) + (b[BS.eyeLookInRight] - b[BS.eyeLookOutRight]));
    const gy = 0.5 * ((b[BS.eyeLookUpLeft] - b[BS.eyeLookDownLeft]) + (b[BS.eyeLookUpRight] - b[BS.eyeLookDownRight]));
    return [gx, gy];
  }

  const ZERO_POSE = new Float32Array(33 * 4);
  // lm: 33*4 normalized x,y,z,visibility (or null = no pose -> zeros, as the dataset stores an absent pose)
  function bodyFromPose(lm, w, h, visMin) {
    if (visMin == null) visMin = 0.5;
    const a = lm || ZERO_POSE, stride = a.length >= 33 * 4 ? 4 : 3;
    const px = [0, 0, 0, 0, 0, 0], py = px.slice(), pz = px.slice(), vis = px.slice();
    for (let j = 0; j < 6; j++) {
      const o = POSE_IDX[j] * stride;
      px[j] = a[o] * w; py[j] = a[o + 1] * h; pz[j] = a[o + 2] * w; vis[j] = stride === 4 ? +a[o + 3] : 1;
    }
    const cx = 0.5 * (px[0] + px[1]), cy = 0.5 * (py[0] + py[1]), cz = 0.5 * (pz[0] + pz[1]);
    let s = Math.hypot(px[0] - px[1], py[0] - py[1]);
    const ok = vis[0] >= visMin && vis[1] >= visMin && s > 1e-3;
    if (!ok) s = 1.0;
    const body = new Float32Array(18);
    let side = Math.sign((px[0] - cx) / s);
    if (side === 0) side = 1;                          // numpy: sign 0 (or -0) -> +1
    for (let j = 0; j < 6; j++) {
      const o = 3 * j;
      if (!ok || vis[j] < visMin) {
        body[o] = REST[j][0] * side; body[o + 1] = REST[j][1]; body[o + 2] = REST[j][2];
      } else {
        body[o] = (px[j] - cx) / s; body[o + 1] = -(py[j] - cy) / s; body[o + 2] = -(pz[j] - cz) / s;
      }
    }
    return {body, ok};
  }

  // h: 21*3 (flat) MediaPipe hand landmarks in consistent metric units (pixels: x*w, y*h, z*w)
  function handDescriptor(h) {
    const d = (i, j) => norm3(h[3 * i] - h[3 * j], h[3 * i + 1] - h[3 * j + 1], h[3 * i + 2] - h[3 * j + 2]);
    const ext = (tip, mcp) => clip01((d(tip, 0) / Math.max(d(mcp, 0), 1e-6) - 1.1) / (1.9 - 1.1));
    const fi = ext(8, 5), fm = ext(12, 9), fr = ext(16, 13), fp = ext(20, 17);
    const op = (fi + fm + fr + fp) / 4;
    const th = clip01((d(4, 17) / Math.max(d(5, 17), 1e-6) - 0.9) / 0.7);
    const point = clip01(fi - (fm + fr + fp) / 3);
    const ax = h[15] - h[0], ay = h[16] - h[1], az = h[17] - h[2];      // 5 - 0
    const bx = h[51] - h[0], by = h[52] - h[1], bz = h[53] - h[2];      // 17 - 0
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const palm = -nz / Math.max(norm3(nx, ny, nz), 1e-6);
    return [op, th, point, palm];
  }
  function handDescriptorSide(h, isLeft) {
    const d = handDescriptor(h);
    if (!isLeft) d[3] = -d[3];
    return d;
  }

  function assignHandsToWrists(P, wl, wr) {
    const out = {l: null, r: null};
    if (!P || !P.length) return out;
    const dl = i => Math.hypot(P[i][0] - wl[0], P[i][1] - wl[1]), dr = i => Math.hypot(P[i][0] - wr[0], P[i][1] - wr[1]);
    if (P.length === 1) out[dl(0) <= dr(0) ? 'l' : 'r'] = 0;
    else {
      const a = dl(0) + dr(1), b = dl(1) + dr(0);
      if (a <= b) { out.l = 0; out.r = 1; } else { out.l = 1; out.r = 0; }
    }
    return out;
  }

  const REST_BODY = bodyFromPose(null, 1, 1).body;   // pose absent (side +1)

  function canonFrame(inp) {
    const motion = new Float32Array(D_MOTION), mask = new Uint8Array(D_MASK);
    const w = +inp.w || 1, h = +inp.h || 1;
    let tcm = null;
    const face = inp.face;
    if (face && face.bs && face.matColMajor && face.matColMajor.length >= 16) {
      const bs = face.bs;
      for (let i = 0; i < 52; i++) motion[i] = i < bs.length ? bs[i] : 0;
      const b = motion.subarray(0, 52);          // float32 values, as the dataset stores them
      const e = eulerFromMat(colToRow(face.matColMajor));
      motion[52] = e.rot[0]; motion[53] = e.rot[1]; motion[54] = e.rot[2];
      motion[55] = e.t[0]; motion[56] = e.t[1]; motion[57] = e.t[2];
      const g = gazeFromBs(b);
      motion[58] = g[0]; motion[59] = g[1];
      mask[0] = 1; tcm = e.t;
    }
    const pose = inp.pose && inp.pose.lm ? inp.pose.lm : null;
    if (pose) {
      const r = bodyFromPose(pose, w, h, 0.5);
      motion.set(r.body, SL.body); mask[1] = r.ok ? 1 : 0;
      const hands = inp.hands || [];
      const n = Math.min(2, hands.length);
      if (n) {
        const wl = [pose[15 * 4] * w, pose[15 * 4 + 1] * h], wr = [pose[16 * 4] * w, pose[16 * 4 + 1] * h];
        const hw = [];
        for (let k = 0; k < n; k++) hw.push([hands[k].lm[0] * w, hands[k].lm[1] * h]);
        const a = assignHandsToWrists(hw, wl, wr);
        for (const [side, col, mi] of [['l', SL.handL, 2], ['r', SL.handR, 3]]) {
          const k = a[side];
          if (k == null) continue;
          const lm = hands[k].lm, H = new Float64Array(63);
          for (let i = 0; i < 21; i++) { H[3 * i] = lm[3 * i] * w; H[3 * i + 1] = lm[3 * i + 1] * h; H[3 * i + 2] = lm[3 * i + 2] * w; }
          const d = handDescriptorSide(H, side === 'l');
          motion[col] = d[0]; motion[col + 1] = d[1]; motion[col + 2] = d[2]; motion[col + 3] = d[3];
          mask[mi] = 1;
        }
      }
    } else motion.set(REST_BODY, SL.body);
    return {motion, mask, tcm};
  }

  // ── uplink 'H' ──
  const H_TYPE = 0x48, H_HDR = 12, H_SCALE = 10000, H_BODY = 1000;   // body cols: wrist z reaches 4.4 shoulder widths
  function packHuman(t, motion, mask) {
    const n = D_MOTION, m = new Uint8Array(H_HDR + 2 * n), dv = new DataView(m.buffer);
    let flags = 0;
    for (let i = 0; i < D_MASK; i++) if (mask && mask[i]) flags |= 1 << i;
    m[0] = H_TYPE; m[1] = flags; dv.setUint16(2, n, true); dv.setFloat64(4, +t, true);
    for (let i = 0; i < n; i++) {
      let v = +motion[i];
      v = i >= 55 && i <= 57 ? v / 100 * H_SCALE : i >= 60 && i <= 77 ? v * H_BODY : v * H_SCALE;
      v = v === v && Math.abs(v) !== Infinity ? Math.round(v) : 0;
      dv.setInt16(H_HDR + 2 * i, v > 32767 ? 32767 : v < -32767 ? -32767 : v, true);
    }
    return m;
  }
  function unpackHuman(bytes) {
    const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    if (b.length < H_HDR || b[0] !== H_TYPE) throw Error('not an H message');
    const flags = b[1], n = dv.getUint16(2, true), t = dv.getFloat64(4, true);
    if (b.length < H_HDR + 2 * n) throw Error('short H message');
    const motion = new Float32Array(n), mask = new Uint8Array(D_MASK);
    for (let i = 0; i < n; i++) {
      const q = dv.getInt16(H_HDR + 2 * i, true);
      motion[i] = i >= 55 && i <= 57 ? q / H_SCALE * 100 : i >= 60 && i <= 77 ? q / H_BODY : q / H_SCALE;
    }
    for (let i = 0; i < D_MASK; i++) mask[i] = (flags >> i) & 1;
    return {t, flags, n, mask, motion};
  }

  return {BS_NAMES, MOTION_COLS, D_MOTION, MASK_COLS, POSE_IDX, SL, CM_COLS, eulerFromMat, colToRow, bsToSpec, gazeFromBs,
    bodyFromPose, handDescriptor, handDescriptorSide, assignHandsToWrists, canonFrame, packHuman, unpackHuman,
    H_TYPE, H_BYTES: H_HDR + 2 * D_MOTION};
});
