/* moves.js — the avatar's whole-body move engine and object (prop) sequencer.
 *
 * Loaded BEFORE motion.js (window.AvatarMovesLib in the browser, module.exports under node); motion.js creates one
 * engine per controller: AvatarMovesLib.createEngine({rnd, now, emit}) and calls it from step():
 *   engine.base(o, dt)   — after the paw rest layer, before gestures: a paw that HOLDS a prop rests at its carry pose
 *   engine.apply(o, dt)  — after gestures + held poses (top priority): root motion, squash, camera, gaze_free, the move's
 *                          full-body channels, secondary motion, object actions (paws, grasp, props)
 * Nothing here runs (and the RNG is never touched) until a move/object command arrives, so the rules output with no new
 * command and every prop at home is byte-identical to before (tests/rules_golden.cjs).
 *
 * UNITS (docs/INTENT-RULES.md §1): angles in TURNS, distances in body heights H; world = H units, origin on the floor
 * under the character's centre at rest, x screen-right, y up, z toward the camera. Root: pw = T + C + R(pl − C),
 * R = Ry(yaw)·Rx(pitch)·Rz(roll), yaw + = front swings to screen-right (= the character's OWN left), pitch + = top tips
 * toward the camera (backflip = −1), roll + = top to screen-right.
 *
 * PRIMITIVES: Track = per-channel piecewise curve (min-jerk eases, quintic Hermite blends that match position, velocity
 * and acceleration, relative procedural segments: cruise rotations, ballistic arcs, gait cycles, damped wobbles);
 * every move is composed from them, starts from the CURRENT pose (+ a decaying residual that carries the current
 * velocity, so there is never a kink), ends in an eased settle. Floor contact: the lowest point of a rounded body box
 * (renderer/body/<c>.json extents) under R stays on the floor (root_y = lift(R) + any airborne height), so rotations
 * roll/tip on the floor instead of sinking into it. Ballistic arcs use one gravity per character G = 12·speed² H/s²
 * (airtime ↔ height consistent; speed scales time, heights stay). Secondary motion: ears / tail / paws lag the root's
 * accelerations through springs; the head leads every turn (it looks where the body will be 0.18 s later).
 *
 * OBJECTS: props/<c>.json (places, props, reach maps). The paw is driven in WORLD space along smooth paths (an arc
 * over whatever is on the table) and turned into hand targets every frame with the inverse of the calibrated reach map
 * (trilinear grid, damped Gauss-Newton, warm started). Grasp = the attach weight eases 0→1 only once the paw's grasp
 * point is at the object (within tolerance); release = 1→0 at the rest spot. Grasp, release, the flight + catch of a
 * toss and a drop's fall are ATOMIC: a new command or "stop" waits for them, so a prop is always either held or at a
 * valid rest spot — never floating, never duplicated.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarMovesLib = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ------------------------------------------------------------------ math
  const TAU = 2 * Math.PI;
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const lerp = (a, b, s) => a + (b - a) * s;
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));
  const sstep = (a, b, x) => { const u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };
  const wrap = x => x - Math.round(x);            // → [-0.5, 0.5]
  const wrapH = x => { const y = x - Math.floor(x + 0.5); return y >= 0.5 ? y - 1 : y; };   // [-0.5, 0.5)
  const pulse = (u, a, b, c, d) => u < a ? 0 : u < b ? mj((u - a) / (b - a)) : u < c ? 1 : u < d ? 1 - mj((u - c) / (d - c)) : 0;
  const MJI = s => s <= 0 ? 0 : s >= 1 ? 0.5 : s * s * s * s * (2.5 + s * (-3 + s));   // ∫ mj
  // smooth start, constant middle velocity, smooth stop (0 → 1); a = ramp fraction each end
  function cruise(u, a = 0.25) {
    if (u <= 0) return 0; if (u >= 1) return 1;
    a = clamp(a, 0.01, 0.5); const V = 1 / (1 - a);
    if (u < a) return V * a * MJI(u / a);
    if (u > 1 - a) return 1 - V * a * MJI((1 - u) / a);
    return V * (a * 0.5 + (u - a));
  }
  // quintic Hermite (p, v, a at both ends) → f(t), t in [0, T]
  function quintic(p0, v0, a0, p1, v1, a1, T) {
    T = Math.max(1e-3, T);
    const c2 = a0 / 2, d = p1 - (p0 + v0 * T + c2 * T * T), dv = v1 - (v0 + a0 * T), da = a1 - a0;
    const k3 = 10 * d - 4 * dv * T + 0.5 * da * T * T, k4 = -15 * d + 7 * dv * T - da * T * T, k5 = 6 * d - 3 * dv * T + 0.5 * da * T * T;
    return t => {
      if (t <= 0) return p0; if (t >= T) return p1 + v1 * (t - T);
      const s = t / T;
      return p0 + v0 * t + c2 * t * t + s * s * s * (k3 + s * (k4 + s * k5));
    };
  }
  const v3 = (a) => Array.isArray(a) && a.length >= 3 && a.every(Number.isFinite) ? [+a[0], +a[1], +a[2]] : null;
  const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const len3 = a => Math.hypot(a[0], a[1], a[2]);
  const lerp3 = (a, b, s) => [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
  const bez3 = (a, m, b, s) => { const u = 1 - s; return [u * u * a[0] + 2 * u * s * m[0] + s * s * b[0], u * u * a[1] + 2 * u * s * m[1] + s * s * b[1], u * u * a[2] + 2 * u * s * m[2] + s * s * b[2]]; };
  function rotm(yaw, pitch, roll) {   // turns → R = Ry·Rx·Rz (roll + = top to screen-right)
    const a = yaw * TAU, b = pitch * TAU, c = roll * TAU;
    const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
    // Rx·Rz
    const xz = [[cc, sc, 0], [-cb * sc, cb * cc, -sb], [-sb * sc, sb * cc, cb]];
    return [[ca * xz[0][0] + sa * xz[2][0], ca * xz[0][1] + sa * xz[2][1], ca * xz[0][2] + sa * xz[2][2]],
      xz[1], [-sa * xz[0][0] + ca * xz[2][0], -sa * xz[0][1] + ca * xz[2][1], -sa * xz[0][2] + ca * xz[2][2]]];
  }
  const mv = (R, p) => [R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]];

  // ------------------------------------------------------------------ Track: a channel's piecewise curve over move time
  class Track {
    constructor(v, ang = false) { this.s = []; this.te = 0; this.v = v; this.ang = ang; this.i = 0; }
    _push(t1, f, vEnd) { this.s.push({t0: this.te, t1, f}); this.te = t1; this.v = vEnd; return this; }
    hold(d) { if (d > 1e-6) { const v = this.v; this._push(this.te + d, () => v, v); } return this; }
    until(t) { return this.hold(t - this.te); }
    to(v1, d, prof = mj) { if (d <= 1e-6) return this; const v0 = this.v, t0 = this.te; return this._push(t0 + d, t => v0 + (v1 - v0) * prof((t - t0) / d), v1); }
    rel(d, f) { if (d <= 1e-6) return this; const v0 = this.v, t0 = this.te; return this._push(t0 + d, t => v0 + f(clamp((t - t0) / d, 0, 1), t - t0), v0 + f(1, d)); }
    fn(d, f) { if (d <= 1e-6) return this; const t0 = this.te; return this._push(t0 + d, t => f(t - t0), f(d)); }   // absolute values
    quint(v1, d, vel0, acc0, vel1 = 0, acc1 = 0) { if (d <= 1e-6) return this; const t0 = this.te, q = quintic(this.v, vel0, acc0, v1, vel1, acc1, d); return this._push(t0 + d, t => q(t - t0), v1); }
    truncate(t) {   // keep the curve up to t; the end value becomes the value at t
      const v = this.at(t); this.s = this.s.filter(g => g.t0 < t); if (this.s.length) this.s[this.s.length - 1].t1 = t;
      this.te = t; this.v = v; this.i = 0; return this;
    }
    at(t) {
      const s = this.s; if (!s.length || t >= this.te) return this.v;
      if (t <= s[0].t0) return s[0].f(s[0].t0);
      let i = Math.min(this.i, s.length - 1);
      while (i > 0 && t < s[i].t0) i--;
      while (i < s.length - 1 && t >= s[i].t1) i++;
      this.i = i; return s[i].f(t);
    }
    vel(t, h = 1 / 120) { return (this.at(t + h) - this.at(t - h)) / (2 * h); }
    acc(t, h = 1 / 60) { return (this.at(t + h) - 2 * this.at(t) + this.at(t - h)) / (h * h); }
  }
  const ROOT = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'squash', 'stand'];
  const ANG = new Set(['yaw', 'pitch', 'roll']);

  // ------------------------------------------------------------------ the body (floor support, head, stand)
  function collectBoxes(o, out, depth = 0) {
    if (!o || typeof o !== 'object' || depth > 4) return;
    const lo = v3(o.min || o.lo || o.lower), hi = v3(o.max || o.hi || o.upper);
    if (lo && hi) { out.push([lo, hi]); return; }
    if (Array.isArray(o) && o.length === 2 && v3(o[0]) && v3(o[1])) { out.push([v3(o[0]), v3(o[1])]); return; }
    if (Array.isArray(o.x) && Array.isArray(o.y) && Array.isArray(o.z) && o.x.length === 2) {
      out.push([[+o.x[0], +o.y[0], +o.z[0]], [+o.x[1], +o.y[1], +o.z[1]]]); return;
    }
    for (const k in o) if (o[k] && typeof o[k] === 'object') collectBoxes(o[k], out, depth + 1);
  }
  function bodyModel(b) {
    b = b && typeof b === 'object' ? b : {};
    const boxes = []; collectBoxes(b.extents, boxes);
    let lo = [-0.42, 0, -0.36], hi = [0.42, 1.0, 0.36];
    if (boxes.length) {
      lo = [Infinity, Infinity, Infinity]; hi = [-Infinity, -Infinity, -Infinity];
      for (const [a, c] of boxes) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], a[i], c[i]); hi[i] = Math.max(hi[i], a[i], c[i]); }
    }
    const baseY = Number.isFinite(+b.base_y) ? +b.base_y : 0;
    if (!(hi[1] > lo[1] + 0.2)) { lo = [-0.42, baseY, -0.36]; hi = [0.42, baseY + 1, 0.36]; }
    const E = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], s = [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2];
    const C = v3(b.com) || [E[0], lo[1] + 0.42 * (hi[1] - lo[1]), E[2]];
    const rr = 0.4 * Math.min(s[0], s[1], s[2]);
    const head = v3(b.head) || v3(b.eyes) || [0, lo[1] + 0.8 * (hi[1] - lo[1]), Math.min(hi[2], E[2] + 0.12)];
    const eps = 0.04, sab = x => Math.sqrt(x * x + eps * eps) - eps;
    const low = R => C[1] + R[1][0] * (E[0] - C[0]) + R[1][1] * (E[1] - C[1]) + R[1][2] * (E[2] - C[2]) -
      (s[0] - rr) * sab(R[1][0]) - (s[1] - rr) * sab(R[1][1]) - (s[2] - rr) * sab(R[1][2]) - rr;
    const low0 = low(rotm(0, 0, 0));
    // the same lowest point with a much softer corner (in the air the root follows this, no kink per quarter turn)
    const sabK = (x, k) => Math.sqrt(x * x + k * k) - k;
    const lowK = (R, k) => C[1] + R[1][0] * (E[0] - C[0]) + R[1][1] * (E[1] - C[1]) + R[1][2] * (E[2] - C[2]) -
      (s[0] - rr) * sabK(R[1][0], k) - (s[1] - rr) * sabK(R[1][1], k) - (s[2] - rr) * sabK(R[1][2], k) - rr;
    const L = b.legs && typeof b.legs === 'object' ? b.legs : null;
    let legs = null;
    if (L) {
      const hipL = v3(L.hip_l) || [-0.09, 0.36, 0], hipR = v3(L.hip_r) || [0.09, 0.36, 0];
      const fl = v3(L.foot_l) || [hipL[0], lo[1], 0.02], fr = v3(L.foot_r) || [hipR[0], lo[1], 0.02];
      const len = (+L.thigh || 0) + (+L.shin || 0) || +L.length || (hipL[1] - fl[1]);
      legs = {hip: {l: hipL, r: hipR}, foot: {l: fl, r: fr}, thigh: +L.thigh || 0.52 * len, shin: +L.shin || 0.48 * len, len,
        hipY: (hipL[1] + hipR[1]) / 2, belowHip: b.seat && +b.seat.below_hip || 0.06, maxStep: +b.max_step || +L.max_step || 0.22};
    }
    return {lo, hi, E, s, C, rr, head, stand: b.stand === true || !!legs, legs, H: +b.H || 1, raw: b, waist: v3(b.waist) || null, seatFeet: b.seatFeet || null,
      lift: R => low0 - low(R),                         // root_y that keeps the lowest point on the floor
      liftSoft: (R, k) => lowK(rotm(0, 0, 0), k) - lowK(R, k),
      pivot(R, pe) { const d = sub3(pe, C); return sub3(d, mv(R, d)); },   // T for a rotation about the floor point pe
      front: lo[2] < hi[2] ? hi[2] : 0.35};
  }

  // ------------------------------------------------------------------ adapters: the body / props files' real JSON → the engine's shapes
  // renderer/animals/<c>.body.json: joints in SCENE units (÷ H), extents relative to the com, leg/step/sitting
  // blocks. The engine wants H units: legs {hip_l/r, thigh, shin, foot_l/r (sole points at rest)}, seat.below_hip, head.
  function adaptBody(b) {
    if (!b || typeof b !== 'object' || b.legs || !b.joints_scene) return b;
    const H = +b.H || 1, o = Array.isArray(b.origin_scene) ? b.origin_scene : [0, 0, 0], J = b.joints_scene;
    const sc = v => Array.isArray(v) ? [(v[0] - o[0]) / H, (v[1] - o[1]) / H, (v[2] - o[2]) / H] : null;
    const c = v3(b.com) || [0, 0.5, 0], ex = b.extents || {}, leg = b.leg || {}, st = b.step || {}, sit = b.sitting || {};
    const out = Object.assign({}, b, {sceneH: H});
    if (Number.isFinite(+ex.x)) out.extents = {min: [c[0] - ex.x, c[1] - ex.y_down, c[2] - ex.z_back], max: [c[0] + ex.x, c[1] + ex.y_up, c[2] + ex.z_front]};
    const hipL = sc(J.hip_l), hipR = sc(J.hip_r), aL = sc(J.ankle_rest_l), aR = sc(J.ankle_rest_r), sole = (+leg.sole_below_ankle_scene || 0) / H;
    if (hipL && hipR && aL && aR) {
      out.legs = {hip_l: hipL, hip_r: hipR, thigh: (+leg.thigh || 0.2) / H, shin: (+leg.shin || 0.2) / H,
        foot_l: [aL[0], Math.max(0, aL[1] - sole), aL[2]], foot_r: [aR[0], Math.max(0, aR[1] - sole), aR[2]],
        max_step: +st.max_comfortable_H || 0.13, lift: +st.foot_lift_H || 0.05, cadence: +st.cadence_hz || 1.8};
      const hipY = (hipL[1] + hipR[1]) / 2;
      out.seat = {below_hip: Number.isFinite(+sit.seat_contact_H) ? hipY - sit.seat_contact_H : 0.06};
      const fh = sit.feet_for_thighs_horizontal_H; if (fh && fh.foot_l && fh.foot_r) out.seatFeet = {l: v3(fh.foot_l), r: v3(fh.foot_r)};
    }
    const hd = sc(b.head_scene); if (hd) out.head = hd;
    const wj = sc(J.waist); if (wj) out.waist = wj;
    return out;
  }
  // renderer/animals/<c>.props.json: handles as {id, side, local}, chair.seat.sit_point_local, desk.sit_point_local,
  // laptop keyboard/screen blocks, grip_point_local, places with on_prop. Normalised into what feat() and the steps read.
  function adaptProps(P) {
    if (!P || typeof P !== 'object' || !Array.isArray(P.props)) return P;
    const out = Object.assign({}, P);
    out.places = (P.places || []).map(pl => pl && pl.on_prop && !pl.prop ? Object.assign({}, pl, {prop: pl.on_prop}) : pl);
    out.props = P.props.map(d0 => {
      if (!d0 || typeof d0 !== 'object') return d0;
      const d = Object.assign({}, d0), H = Array.isArray(d0.handles) && d0.handles.length && typeof d0.handles[0] === 'object' && !Array.isArray(d0.handles[0]) ? d0.handles : null;
      if (H) {
        d.handlesFull = H;
        const pick = s => H.find(h => h.side === s && Array.isArray(h.local));
        const hl = pick('l'), hr = pick('r');
        d.handles = hl && hr ? [hl.local.slice(), hr.local.slice()] : H.filter(h => Array.isArray(h.local)).map(h => h.local.slice());
        const one = H.find(h => h.side === 'either' && !/lid/.test(h.id || ''));
        if (one && !d.grip_point_local) d.grip_point_local = one.local.slice();
        const lid = H.find(h => /lid/.test(h.id || '')); if (lid) d.lidEdge = lid.local.slice();
      }
      const home = d0.home && Array.isArray(d0.home.pos) ? d0.home.pos : [0, 0, 0];
      if (d0.seat && typeof d0.seat === 'object' && d0.seat.sit_point_local) {
        const sp = d0.seat.sit_point_local, h = +d0.seat.height || home[1];
        d.seat = Object.assign({}, d0.seat, {center: [sp[0], h - home[1], sp[2]], height: h, front: d0.seat.sit_facing_local || [0, 0, 1]});
      }
      if (d0.furniture && d0.top && d0.sit_point_local) { d.chair_spot = d0.sit_point_local.slice(); d.chairSpotIsSit = true; }
      if (d0.keyboard && typeof d0.keyboard === 'object' && !Array.isArray(d0.keyboard)) {
        const K = d0.keyboard, kc = K.center_local || [0, 0, 0], kl = K.keys_left_local, kr = K.keys_right_local;
        d.keyboard = kl && kr ? [(kl[0] + kr[0]) / 2, (kl[1] + kr[1]) / 2, (kl[2] + kr[2]) / 2] : kc.slice();
        const half = kl && kr ? Math.abs(kr[0] - kl[0]) / 2 / 0.4 : null;
        const S = d0.screen || {}, sc0 = S.center_open_local, edge = d.lidEdge;
        if (sc0 && edge) {   // hinge at the back edge: the open panel's centre sits half a lid length up from it, tipped back
          const deg = 112, th = deg * Math.PI / 180, Lz = edge[2] - sc0[2], len = Lz / (1 + 0.5 * -Math.cos(th));
          const hinge = [0, edge[1], edge[2] - len];
          d.lid = {hinge, length: len, open_deg: deg, half_width: half || undefined};
        } else if (half) d.lid = Object.assign({}, d0.lid || {}, {half_width: half});
        if (sc0) d.screen = sc0.slice();
      }
      return d;
    });
    return out;
  }

  // ------------------------------------------------------------------ the calibrated reach map and its inverse
  // Accepted shapes for props.reach.<l|r> (the props file's choice): {axes:{x,y,z}, pts | grid | world} (pts flat [x,y,z]*
  // in x-major order unless order:'zyx', or nested [ix][iy][iz] = [x,y,z]); {lo, hi, n, pts}; {origin|o, J} (affine).
  function ReachMap(def, side, body) {
    this.side = side; this.ok = false; this.kind = 'analytic';
    const sg = side === 'l' ? -1 : 1, B = body;
    // analytic fallback: a paw at rest beside the tummy, one arm unit ≈ 0.42 H
    const rest = [sg * Math.max(0.12, 0.45 * B.s[0]), B.lo[1] + 0.3 * (B.hi[1] - B.lo[1]), Math.min(B.hi[2] - 0.05, B.E[2] + 0.22)], L = 0.42 * (B.hi[1] - B.lo[1]);
    this.fallback = h => [rest[0] + sg * L * h[0], rest[1] + L * h[1] * (1 - 0.08 * h[0] * h[0]), rest[2] + L * 0.85 * h[2] * (1 - 0.12 * Math.max(0, h[1])) + 0.04 * L * h[1]];
    this.lo = [-1, -1, -1]; this.hi = [1.2, 1.2, 1.2];
    try { this._parse(def); } catch (e) { this.ok = false; }
    this.h0 = [0, 0, 0];
  }
  ReachMap.prototype._parse = function (d) {
    if (!d || typeof d !== 'object') return;
    const A = d.axes || d.grid && !Array.isArray(d.grid) && d.grid.x && d.grid || null;
    let ax = Array.isArray(A) && A.length === 3 ? A : A && [A.x || A.hand_x, A.y || A.hand_y, A.z || A.hand_z];
    if (!ax && Array.isArray(d.lo) && Array.isArray(d.hi) && Array.isArray(d.n)) ax = [0, 1, 2].map(i => { const n = Math.max(2, d.n[i] | 0), a = []; for (let k = 0; k < n; k++) a.push(d.lo[i] + (d.hi[i] - d.lo[i]) * k / (n - 1)); return a; });
    const P = d.pts || d.world || d.points || d.pos || d.p || (Array.isArray(d.grid) ? d.grid : null);
    if (ax && ax.every(a => Array.isArray(a) && a.length >= 2) && P) {
      const nx = ax[0].length, ny = ax[1].length, nz = ax[2].length, N = nx * ny * nz, flat = new Float64Array(N * 3);
      const zyx = d.order === 'zyx';
      if (Array.isArray(P[0]) && Array.isArray(P[0][0]) && Array.isArray(P[0][0][0])) {   // nested [ix][iy][iz][3]
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) { const q = P[i][j][k]; const o = ((i * ny + j) * nz + k) * 3; flat[o] = q[0]; flat[o + 1] = q[1]; flat[o + 2] = q[2]; }
      } else {
        const F = Array.isArray(P[0]) ? P.flat() : P;
        if (F.length < N * 3) return;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
          const src = zyx ? ((k * ny + j) * nx + i) * 3 : ((i * ny + j) * nz + k) * 3, o = ((i * ny + j) * nz + k) * 3;
          flat[o] = +F[src]; flat[o + 1] = +F[src + 1]; flat[o + 2] = +F[src + 2];
        }
      }
      if (!flat.every(Number.isFinite)) return;
      this.ax = ax.map(a => a.map(Number)); this.n = [nx, ny, nz]; this.g = flat; this.ok = true; this.kind = 'grid';
      // objects engine: the palm normals on the same grid (where a held thing's anchor sits off the palm)
      const PN = d.palm_normal, FN = Array.isArray(PN) ? (Array.isArray(PN[0]) ? PN.flat() : PN) : null;
      if (FN && FN.length >= N * 3 && !zyx) { const gn = new Float64Array(N * 3); for (let i = 0; i < N * 3; i++) gn[i] = +FN[i]; if (gn.every(Number.isFinite)) this.gn = gn; }
      this.lo = this.ax.map(a => a[0]); this.hi = this.ax.map(a => a[a.length - 1]);
      return;
    }
    const o = v3(d.origin || d.o), J = d.J || d.jacobian;
    if (o && Array.isArray(J) && J.length === 3) { this.aff = {o, J}; this.ok = true; this.kind = 'affine'; }
  };
  ReachMap.prototype.fwd = function (h) {
    if (this.kind === 'grid') {
      const ax = this.ax, n = this.n, idx = [0, 0, 0], fr = [0, 0, 0];
      for (let a = 0; a < 3; a++) {
        const A = ax[a], x = h[a]; let i = 0;
        while (i < n[a] - 2 && x > A[i + 1]) i++;
        idx[a] = i; fr[a] = (x - A[i]) / (A[i + 1] - A[i]);   // linear extrapolation outside
      }
      const out = [0, 0, 0], g = this.g;
      for (let c = 0; c < 8; c++) {
        const di = c >> 2 & 1, dj = c >> 1 & 1, dk = c & 1;
        const w = (di ? fr[0] : 1 - fr[0]) * (dj ? fr[1] : 1 - fr[1]) * (dk ? fr[2] : 1 - fr[2]);
        if (w === 0) continue;
        const o = (((idx[0] + di) * n[1] + idx[1] + dj) * n[2] + idx[2] + dk) * 3;
        out[0] += w * g[o]; out[1] += w * g[o + 1]; out[2] += w * g[o + 2];
      }
      return out;
    }
    if (this.kind === 'affine') { const {o, J} = this.aff; return [0, 1, 2].map(i => o[i] + J[i][0] * h[0] + J[i][1] * h[1] + J[i][2] * h[2]); }
    return this.fallback(h);
  };
  ReachMap.prototype.normal = function (h) {   // objects engine: the palm normal (rest body frame) for hand target h, or null
    if (this.kind !== 'grid' || !this.gn) return null;
    const ax = this.ax, n = this.n, idx = [0, 0, 0], fr = [0, 0, 0];
    for (let a = 0; a < 3; a++) { const A = ax[a], x = clamp(h[a], A[0], A[A.length - 1]); let i = 0; while (i < n[a] - 2 && x > A[i + 1]) i++; idx[a] = i; fr[a] = (x - A[i]) / (A[i + 1] - A[i]); }
    const out = [0, 0, 0], g = this.gn;
    for (let c = 0; c < 8; c++) {
      const di = c >> 2 & 1, dj = c >> 1 & 1, dk = c & 1, w = (di ? fr[0] : 1 - fr[0]) * (dj ? fr[1] : 1 - fr[1]) * (dk ? fr[2] : 1 - fr[2]);
      if (w === 0) continue; const o = (((idx[0] + di) * n[1] + idx[1] + dj) * n[2] + idx[2] + dk) * 3; out[0] += w * g[o]; out[1] += w * g[o + 1]; out[2] += w * g[o + 2];
    }
    const L = len3(out); return L > 1e-6 ? mul3(out, 1 / L) : null;
  };
  // hand target that puts the grasp point at P (damped Gauss-Newton, clamped to the calibrated box) → {h, err}
  ReachMap.prototype.inv = function (P, h0, iters = 8) {
    let h = (h0 || this.h0).slice(), lo = this.lo, hi = this.hi;
    for (let a = 0; a < 3; a++) h[a] = clamp(h[a], lo[a], hi[a]);
    let f = this.fwd(h), r = sub3(P, f), e = len3(r);
    for (let it = 0; it < iters && e > 2e-4; it++) {
      const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dh = 1e-3;
      for (let a = 0; a < 3; a++) { const hp = h.slice(); hp[a] += dh; const fp = this.fwd(hp); for (let b = 0; b < 3; b++) J[b][a] = (fp[b] - f[b]) / dh; }
      // (JᵀJ + λI) δ = Jᵀ r
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0], lam = 1e-4;
      for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += J[k][i] * J[k][j]; A[i][j] = s + (i === j ? lam : 0); } let s = 0; for (let k = 0; k < 3; k++) s += J[k][i] * r[k]; g[i] = s; }
      const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
      if (Math.abs(det) < 1e-12) break;
      const inv = [[A[1][1] * A[2][2] - A[1][2] * A[2][1], A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][1] * A[1][2] - A[0][2] * A[1][1]],
        [A[1][2] * A[2][0] - A[1][0] * A[2][2], A[0][0] * A[2][2] - A[0][2] * A[2][0], A[0][2] * A[1][0] - A[0][0] * A[1][2]],
        [A[1][0] * A[2][1] - A[1][1] * A[2][0], A[0][1] * A[2][0] - A[0][0] * A[2][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]]];
      let step = [0, 1, 2].map(i => (inv[i][0] * g[0] + inv[i][1] * g[1] + inv[i][2] * g[2]) / det);
      const sl = len3(step); if (sl > 0.6) step = mul3(step, 0.6 / sl);
      let k = 1, hn, fn, en;
      for (let ls = 0; ls < 4; ls++) {   // backtracking
        hn = [0, 1, 2].map(a => clamp(h[a] + k * step[a], lo[a], hi[a])); fn = this.fwd(hn); en = len3(sub3(P, fn));
        if (en < e) break; k *= 0.5;
      }
      if (!(en < e)) break;
      h = hn; f = fn; r = sub3(P, f); e = en;
    }
    return {h, err: e, p: f};
  };

  // ------------------------------------------------------------------ words → numbers
  const NUMW = {once: 1, one: 1, a: 1, an: 1, single: 1, twice: 2, two: 2, double: 2, couple: 2, thrice: 3, three: 3, triple: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, few: 3, several: 3, half: 0.5};
  function num(x, dflt) {
    if (x === undefined || x === null || x === '') return dflt;
    if (Number.isFinite(+x)) return +x;
    const w = String(x).toLowerCase().match(/[a-z]+|\d+(\.\d+)?/g) || [];
    for (const t of w) { if (Number.isFinite(+t)) return +t; if (NUMW[t] !== undefined) return NUMW[t]; }
    return dflt;
  }
  function speedOf(x) {
    if (x === undefined || x === null || x === '') return 1;
    if (Number.isFinite(+x)) return clamp(+x, 0.3, 3);
    const s = String(x).toLowerCase();
    if (/very\s*slow|super\s*slow/.test(s)) return 0.4; if (/slow/.test(s)) return 0.6;
    if (/very\s*fast|super\s*fast|really\s*fast/.test(s)) return 2.2; if (/fast|quick|rapid/.test(s)) return 1.6;
    return 1;
  }
  function heightOf(x, dflt) {
    if (x === undefined || x === null || x === '') return dflt;
    if (Number.isFinite(+x)) return clamp(+x, 0, 1);
    const s = String(x).toLowerCase();
    if (/tiny|little|small|low|short/.test(s)) return 0.2; if (/huge|very high|really high|giant/.test(s)) return 1; if (/big|high|tall/.test(s)) return 0.8;
    return dflt;
  }
  const normDir = d => String(d || '').toLowerCase().replace(/[\s-]+/g, '_');
  // how far to rotate, in degrees ("all the way around" = 360, "half way" = 180, "a quarter turn" = 90), for the
  // rotating moves: >= 300 is that many full circles (a spin), 135..300 a half turn (turn_away), less a quarter turn
  // (turn, its side kept); "turn … N times" (N >= 2) is N spins. Nothing else changes.
  const ROTATING = /^(turn|spin|twirl|turn_away)$/;
  function rotateAmount(name, a) {
    if (!ROTATING.test(name)) return name;
    const deg = a.degrees !== undefined && a.degrees !== null && isFinite(+a.degrees) ? Math.abs(+a.degrees) : null;
    if (deg === null) { if (name === 'turn' && +a.count >= 2) return 'spin'; return name; }
    delete a.degrees;
    if (deg >= 300) { a.count = Math.max(1, Math.min(10, Math.round(deg / 360))); return name === 'twirl' ? 'twirl' : 'spin'; }
    if (deg >= 135) { delete a.count; return 'turn_away'; }
    delete a.count; if (!a.dir || /^(around|back|away)$/.test(normDir(a.dir))) a.dir = 'left'; return 'turn';
  }
  // yaw sign: + = the front swings to screen-right (= the character's own left; counter-clockwise seen from above)
  function yawSign(dir, dflt = 1) {
    const d = normDir(dir);
    if (/^(left|own_left|ccw|counter_?clockwise|anti_?clockwise|l_own)$/.test(d)) return 1;
    if (/^(right|own_right|cw|clockwise)$/.test(d)) return -1;
    if (/^(screen_left|l|your_left|viewer_left)$/.test(d)) return -1;
    if (/^(screen_right|r|your_right|viewer_right)$/.test(d)) return 1;
    return dflt;
  }
  // lateral world-x sign for travel / tipping: own left = screen-right (+x)
  const sideSign = (dir, dflt) => yawSign(dir, dflt);
  const ownWord = s => s > 0 ? 'its left' : 'its right';            // in yaw-sign terms
  const plural = (n, one, many) => n === 1 ? one : (Number.isInteger(n) ? n : +n.toFixed(2)) + ' ' + many;
  const times = n => n === 1 ? 'once' : n === 2 ? 'twice' : (Number.isInteger(n) ? n : +n.toFixed(2)) + ' times';
  const turnsWord = n => Math.abs(n - 0.5) < 1e-6 ? 'half a turn' : n === 1 ? 'one full turn' : (Number.isInteger(n) ? n : +n.toFixed(2)) + ' turns';

  // ------------------------------------------------------------------ the move library (names → builder)
  const MOVE_ALIAS = {twirl: 'twirl', spin: 'spin', spin_around: 'spin', pirouette: 'twirl', whirl: 'spin', turn_around: 'turn_away',
    turn_away: 'turn_away', turn_your_back: 'turn_away', look_away_body: 'turn_away', turn_back: 'turn_back', face_me: 'turn_back',
    face_camera: 'turn_back', face_front: 'turn_back', turn_to_me: 'turn_back', turn: 'turn', turn_left: ['turn', {dir: 'left'}],
    turn_right: ['turn', {dir: 'right'}], backflip: 'backflip', back_flip: 'backflip', flip: 'flip', frontflip: 'frontflip',
    front_flip: 'frontflip', somersault: 'frontflip', jump: 'jump', leap: 'jump', hop: 'hop', bounce: 'bounce', walk: 'walk',
    step: 'step', steps: 'walk', scoot: 'scoot', shuffle: 'scoot', come_closer: ['walk', {dir: 'forward', hold: true}],
    come_here: ['walk', {dir: 'forward'}], approach: ['walk', {dir: 'forward'}], back_up: ['walk', {dir: 'back'}], dance: 'dance',
    boogie: 'dance', groove: 'dance', sit: 'sit', sit_down: 'sit', settle: 'sit', sit_up: 'sit', stand: 'stand', stand_up: 'stand',
    rise: 'stand', stand_tall: 'stand', get_up: 'get_up', lie_down: 'lie_down', lay_down: 'lie_down', lie: 'lie_down',
    play_dead: ['lie_down', {dir: 'back', dead: true}], roll_over: 'roll_over', roll: 'roll_over', cartwheel: 'cartwheel',
    bow: 'bow', curtsy: 'bow', wiggle: 'wiggle', wriggle: 'wiggle', shake_off: 'shake_off', shake: 'shake_off', shake_it_off: 'shake_off',
    crouch: 'crouch', duck: 'crouch', squat: 'crouch', hide: ['crouch', {hide: true}], sway: 'sway', rock: 'sway', shimmy: 'shimmy',
    stretch_up: 'stand', stop: 'stop', freeze: 'stop', come_back: 'come_back', go_back: 'come_back', return: 'come_back'};
  const MOVE_NAMES = ['spin', 'twirl', 'turn', 'turn_away', 'turn_back', 'backflip', 'frontflip', 'flip', 'jump', 'hop', 'bounce', 'walk',
    'step', 'scoot', 'come_back', 'dance', 'sit', 'stand', 'get_up', 'lie_down', 'roll_over', 'cartwheel', 'bow', 'wiggle', 'shake_off',
    'crouch', 'sway', 'shimmy', 'stop'];

  // ------------------------------------------------------------------ dance: styles from composable steps
  // every step: (b = beats into the step, k = intensity) → {root:{x,y,z,yaw,pitch,roll,squash}, set:{}, add:{}}; periodic in
  // its length (beats) so steps loop and cross-fade (angles blended on the short arc)
  const HPs = (o, s, x, y, z, open, turn) => { o['hand_' + s + '_x'] = x; o['hand_' + s + '_y'] = y; o['hand_' + s + '_z'] = z; if (open !== undefined) o['hand_' + s + '_open'] = open; if (turn !== undefined) o['hand_' + s + '_turn'] = turn; return o; };
  const bump = b => { const f = b - Math.floor(b); return Math.exp(-f * 7) * (1 - Math.exp(-f * 60)); };   // sharp rise on the beat, decay
  const DSTEP = {
    bounce: {len: 2, f(b, k) { const d = 0.5 + 0.5 * Math.cos(TAU * b); return {root: {squash: 0.12 * k * d, y: 0.035 * k * Math.max(0, Math.sin(TAU * b - 1.2)) ** 2}, set: HPs(HPs({}, 'l', 0.18, 0.32 - 0.06 * d, 0.3, 0.6, 0.3), 'r', 0.18, 0.32 - 0.06 * d, 0.3, 0.6, 0.3), add: {head_pitch: 0.06 * k * d, smile: 0.4}}; }},
    sway: {len: 4, f(b, k) { const s = Math.sin(TAU * b / 2); return {root: {roll: 0.022 * k * s, x: 0.05 * k * s, squash: 0.05 * k * (0.5 + 0.5 * Math.cos(TAU * b))}, set: HPs(HPs({}, 'l', 0.35 - 0.12 * s, 0.45 + 0.12 * s, 0.2, 0.9, 0.6), 'r', 0.35 + 0.12 * s, 0.45 - 0.12 * s, 0.2, 0.9, 0.6), add: {head_roll: -0.08 * k * s, smile: 0.35}}; }},
    twist: {len: 2, f(b, k) { const s = Math.sin(TAU * b / 2); return {root: {yaw: 0.05 * k * s, squash: 0.1 * k * Math.abs(Math.cos(TAU * b / 2))}, set: HPs(HPs({}, 'l', 0.2, 0.28, 0.3 + 0.25 * s, 0.4, 0), 'r', 0.2, 0.28, 0.3 - 0.25 * s, 0.4, 0), add: {head_yaw: -0.12 * k * s, smile: 0.4}}; }},
    arm_pump: {len: 2, f(b, k) { const s = Math.sin(TAU * b / 2), l = Math.max(0, s), r = Math.max(0, -s); return {root: {squash: 0.1 * k * (0.5 + 0.5 * Math.cos(TAU * b))}, set: HPs(HPs({}, 'l', 0.18, 0.3 + 0.7 * l, 0.25, 0.1, 0.2), 'r', 0.18, 0.3 + 0.7 * r, 0.25, 0.1, 0.2), add: {head_pitch: 0.05 * k * Math.cos(TAU * b), smile: 0.5, eye_wide: 0.1}}; }},
    disco: {len: 4, f(b, k) { const ph = mj(clamp((b % 4) / 1.2, 0, 1)) - mj(clamp(((b % 4) - 2) / 1.2, 0, 1)); const hip = Math.sin(TAU * b / 2); const o = HPs(HPs({}, 'r', lerp(-0.1, 0.55, ph), lerp(0.05, 1.15, ph), 0.25, 0, 0), 'l', 0.15, 0.28, 0.3, 0.3, 0); o.point_r = 1; return {root: {roll: 0.018 * k * hip, x: 0.04 * k * hip, squash: 0.06 * k * (0.5 + 0.5 * Math.cos(TAU * b))}, set: o, add: {head_roll: 0.06 * (ph - 0.5) * k, smile: 0.45, squint: 0.2}}; }},
    robot: {len: 4, f(b, k) {   // staccato: hold a pose, snap to the next on each beat
      const i = ((Math.floor(b) % 4) + 4) % 4, u = b - Math.floor(b), q = mj(clamp(u / 0.3, 0, 1));
      const P = [[0.5, 0.45, 0.1, 0.05, 0.2, 0.4], [0.05, 0.2, 0.4, 0.5, 0.45, 0.1], [0.45, 0.9, 0.0, 0.45, 0.9, 0.0], [0.05, 0.3, 0.45, 0.05, 0.3, 0.45]];
      const a = P[(i + 3) % 4], c = P[i], v = j => lerp(a[j], c[j], q);
      const hy = [0.12, -0.12, 0.0, 0.06], ry = [0.03, -0.03, 0, 0];
      const y0 = hy[(i + 3) % 4], y1 = hy[i], r0 = ry[(i + 3) % 4], r1 = ry[i];
      return {root: {yaw: k * lerp(r0, r1, q), squash: 0.04 * k * (1 - q)}, set: HPs(HPs({}, 'l', v(0), v(1), v(2), 0.8, 0), 'r', v(3), v(4), v(5), 0.8, 0), add: {head_yaw: k * lerp(y0, y1, q), head_roll: 0}}; }},
    wave_arms: {len: 4, f(b, k) { const s = Math.sin(TAU * b / 4); return {root: {roll: 0.02 * k * s, squash: 0.06 * k * (0.5 + 0.5 * Math.cos(TAU * b))}, set: HPs(HPs({}, 'l', 0.2 - 0.2 * s, 1.05, 0.15, 1, 1), 'r', 0.2 + 0.2 * s, 1.05, 0.15, 1, 1), add: {head_roll: 0.07 * k * s, smile: 0.6, eye_wide: 0.15}}; }},
    shimmy: {len: 2, f(b, k) { const s = Math.sin(TAU * b * 2); return {root: {yaw: 0.012 * k * s, squash: 0.04 * k}, set: Object.assign(HPs(HPs({}, 'l', 0.35, 0.3, 0.2, 1, 0.5), 'r', 0.35, 0.3, 0.2, 1, 0.5), {shoulder_l: 0.5 + 0.5 * s, shoulder_r: 0.5 - 0.5 * s}), add: {body_pitch: -0.15, smile: 0.5, squint: 0.3}}; }},
    spin: {len: 4, f(b, k) { const u = clamp(b / 3, 0, 1); return {root: {yaw: cruise(u, 0.35), squash: -0.08 * Math.sin(Math.PI * u)}, set: HPs(HPs({}, 'l', 0.45, 0.55, 0.1, 1, 0.5), 'r', 0.45, 0.55, 0.1, 1, 0.5), add: {smile: 0.5}}; }},
    hop_turn: {len: 4, f(b, k) { const i = Math.floor(b / 2) % 2, u = (b % 2) / 2, sg = i ? -1 : 1; const air = Math.sin(Math.PI * clamp(u / 0.6, 0, 1)); return {root: {y: 0.09 * k * air, yaw: 0.07 * sg * Math.sin(Math.PI * clamp(u / 0.6, 0, 1)), squash: -0.08 * air + 0.1 * pulse(u, 0.55, 0.62, 0.7, 0.95)}, set: HPs(HPs({}, 'l', 0.3, 0.5 + 0.3 * air, 0.2, 1, 0.6), 'r', 0.3, 0.5 + 0.3 * air, 0.2, 1, 0.6), add: {smile: 0.55, eye_wide: 0.1 * air}}; }},
    head_bob: {len: 2, f(b, k) { const d = bump(b); const s = Math.sin(TAU * b / 2); return {root: {squash: 0.1 * k * d, roll: 0.008 * k * s}, set: HPs(HPs({}, 'l', 0.15 + 0.1 * s, 0.25, 0.3, 0.2, 0.1), 'r', 0.15 - 0.1 * s, 0.25, 0.3, 0.2, 0.1), add: {head_pitch: 0.12 * k * d, head_z: 0.01 * d, smile: 0.25, squint: 0.25}}; }},
    clap: {len: 2, f(b, k) { const c = 0.5 + 0.5 * Math.cos(TAU * b); const x = -0.45 + 0.22 * (1 - c); return {root: {squash: 0.08 * k * c}, set: HPs(HPs({}, 'l', x, 0.5, 0.45, 1, 0), 'r', x, 0.5, 0.45, 1, 0), add: {smile: 0.55, squint: 0.2}}; }},
    chicken: {len: 2, f(b, k) { const s = Math.sin(TAU * b), p = Math.max(0, Math.sin(TAU * b / 2)); return {root: {squash: 0.08 * k * (0.5 + 0.5 * Math.cos(TAU * b)), yaw: 0.01 * Math.sin(TAU * b / 2)}, set: Object.assign(HPs(HPs({}, 'l', 0.45 + 0.15 * s, 0.25 + 0.1 * s, -0.05, 0.2, 0), 'r', 0.45 + 0.15 * s, 0.25 + 0.1 * s, -0.05, 0.2, 0), {shoulder_l: 0.3 + 0.3 * s, shoulder_r: 0.3 + 0.3 * s}), add: {head_z: 0.03 * p, head_pitch: 0.12 * p, smile: 0.4, eye_wide: 0.2}}; }},
    hip_shake: {len: 2, f(b, k) { const s = Math.sin(TAU * b * 2); return {root: {roll: 0.02 * k * s, yaw: 0.015 * k * Math.sin(TAU * b), squash: 0.05 * k}, set: HPs(HPs({}, 'l', 0.25, 0.4, 0.25, 0.8, 0.3), 'r', 0.25, 0.4, 0.25, 0.8, 0.3), add: {tail_wag: 0.9 * s, head_roll: -0.05 * s, smile: 0.5}}; }},
    side_step: {len: 4, f(b, k) { const s = Math.sin(TAU * b / 4), u = b % 1, air = Math.sin(Math.PI * clamp(u / 0.5, 0, 1)); return {root: {x: 0.13 * k * s, y: 0.03 * k * air, roll: -0.012 * k * Math.cos(TAU * b / 4), squash: 0.08 * k * pulse(u, 0.45, 0.55, 0.6, 0.95)}, set: HPs(HPs({}, 'l', 0.25 + 0.15 * s, 0.35, 0.25, 0.6, 0.3), 'r', 0.25 - 0.15 * s, 0.35, 0.25, 0.6, 0.3), add: {smile: 0.45}}; }},
    ballet: {len: 8, f(b, k) { const u = (b % 8) / 8, up = 0.5 - 0.5 * Math.cos(TAU * u); return {root: {squash: -0.15 * k * up, roll: 0.01 * Math.sin(TAU * u)}, set: HPs(HPs({}, 'l', lerp(0.2, -0.05, up), lerp(0.4, 1.15, up), lerp(0.35, 0.15, up), 0.85, 0.4), 'r', lerp(0.2, -0.05, up), lerp(0.4, 1.15, up), lerp(0.35, 0.15, up), 0.85, 0.4), add: {head_roll: 0.1 * Math.sin(TAU * u), head_pitch: -0.06 * up, smile: 0.35, squint: 0.15}}; }},
    body_roll: {len: 4, f(b, k) { const u = (b % 4) / 4, w = Math.sin(TAU * u); return {root: {pitch: 0.015 * k * w, squash: 0.06 * k * (0.5 + 0.5 * Math.cos(TAU * u))}, set: HPs(HPs({}, 'l', 0.2, 0.3 + 0.08 * w, 0.3, 0.5, 0.2), 'r', 0.2, 0.3 + 0.08 * w, 0.3, 0.5, 0.2), add: {body_pitch: 0.25 * w, head_pitch: -0.08 * w, smile: 0.3}}; }},
    jump: {len: 2, f(b, k) { const u = (b % 2) / 2, air = Math.sin(Math.PI * clamp((u - 0.1) / 0.45, 0, 1)); return {root: {y: 0.14 * k * air, squash: 0.14 * pulse(u, 0, 0.08, 0.1, 0.18) - 0.12 * air + 0.14 * pulse(u, 0.52, 0.58, 0.62, 0.85)}, set: HPs(HPs({}, 'l', 0.3, 0.4 + 0.6 * air, 0.15, 1, 0.8), 'r', 0.3, 0.4 + 0.6 * air, 0.15, 1, 0.8), add: {smile: 0.6, eye_wide: 0.2 * air}}; }},
  };
  const DSTYLE = {
    happy: {bpm: 120, steps: [['bounce', 2], ['arm_pump', 2], ['sway', 1.5], ['wave_arms', 1.5], ['hop_turn', 1], ['clap', 1], ['side_step', 1], ['spin', 0.6], ['jump', 0.8]], expr: {smile: 0.25}},
    disco: {bpm: 116, steps: [['disco', 3], ['hip_shake', 1.5], ['sway', 1], ['side_step', 1.2], ['spin', 1]], expr: {smile: 0.2, squint: 0.15}},
    robot: {bpm: 100, steps: [['robot', 4]], feel: 'staccato', expr: {smile: 0}},
    silly: {bpm: 132, steps: [['chicken', 2], ['hip_shake', 1.5], ['wave_arms', 1], ['hop_turn', 1.2], ['shimmy', 1.2], ['jump', 1], ['spin', 0.8]], expr: {smile: 0.3, tongue_out: 0.5, squint: 0.3}},
    hiphop: {bpm: 92, steps: [['head_bob', 3], ['arm_pump', 1.2], ['side_step', 1.2], ['body_roll', 1.5], ['twist', 1]], expr: {smile: 0.1, squint: 0.25}},
    ballet: {bpm: 66, steps: [['ballet', 3], ['spin', 1.2], ['sway', 1]], expr: {smile: 0.2}, k: 0.8},
    salsa: {bpm: 104, steps: [['side_step', 2], ['hip_shake', 2], ['twist', 1.2], ['spin', 1], ['sway', 1]], expr: {smile: 0.3}},
    twist: {bpm: 120, steps: [['twist', 4], ['bounce', 1]], expr: {smile: 0.3}},
    chicken: {bpm: 126, steps: [['chicken', 4], ['bounce', 1]], expr: {smile: 0.3, eye_wide: 0.2}},
    shimmy: {bpm: 112, steps: [['shimmy', 3], ['hip_shake', 1], ['sway', 1]], expr: {smile: 0.35}},
  };
  const DSTYLE_WORDS = [[/robot|mechanic|machine|android/, 'robot'], [/disco|funk|groov|70|seventies|boogie/, 'disco'],
    [/silly|goof|crazy|funny|wild|wacky|weird/, 'silly'], [/hip ?hop|rap|break|street|bboy|b-boy/, 'hiphop'],
    [/ballet|grace|elegant|swan|gentle|slow|waltz/, 'ballet'], [/salsa|latin|cha|samba|tango|rumba|mambo|merengue|bachata/, 'salsa'],
    [/twist/, 'twist'], [/chicken|bird|flap/, 'chicken'], [/shimmy|shake/, 'shimmy'], [/happy|fun|party|cheer|joy|celebrat|excit|bouncy/, 'happy']];

  // ------------------------------------------------------------------ the engine
  const MAX_SLOTS = 16;   // objects engine: prop0 .. prop15 (anim/schema.py)
  function createEngine(ctx) {
    const rnd = ctx.rnd || Math.random, now = ctx.now, emit = ctx.emit || (() => {});
    const U = (a, b) => a + (b - a) * rnd();
    const E = {
      body: bodyModel(null), props: null, places: [], reach: {l: null, r: null},
      cur: {x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0, squash: 0, stand: 0},     // last root output (unwrapped)
      prevRoot: null, vel: {}, acc: {},
      base: {x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0, squash: 0, stand: 0},
      posture: 'seated', lying: null, held: false, facingHold: null,
      move: null, queue: null, w: 0, wHead: 0, gf: 0, gfT: 0, cam: {zoom: 0, zv: 0, x: 0, xv: 0, y: 0, yv: 0},
      view: null, viewW: 0, viewWv: 0, viewCam: null,      // scene: the station camera (setView)
      sec: {earL: 0, earLv: 0, earR: 0, earRv: 0, tail: 0, tailv: 0, hand: 0, handv: 0, tailR: 0, tailRv: 0},
      frozen: false, active: false, cmdId: 0,
      // objects
      P: [], chain: null, oqueue: null, paw: {l: mkPaw(), r: mkPaw()}, look: {P: null, w: 0, wT: 0, user: false},
      lastRef: null, objW: 0, lp: null,
    };
    function mkPaw() { return {ctrl: 0, ctrlT: 0, h: [0, 0, 0], tgt: null, open: 0.2, turn: 0, thumb: 0, point: 0, carryW: 0, hPrev: [0, 0, 0], grip: 0}; }
    const t = () => now();
    function crit(o, k, target, tau, dt) {   // 2nd-order critically damped follower (smooth start and stop)
      const w = 2.2 / Math.max(1e-3, tau), kv = k + 'V'; let x = o[k] || 0, v = o[kv] || 0;
      const n = Math.max(1, Math.ceil(dt / 0.008)), h = dt / n;
      for (let i = 0; i < n; i++) { v += (w * w * (target - x) - 2 * w * v) * h; x += v * h; }
      if (Math.abs(target - x) < 1e-4 && Math.abs(v) < 1e-3) { x = target; v = 0; }
      o[k] = x; o[kv] = v; return x;
    }
    const calmPosture = () => E.posture === (E.body.legs ? 'standing' : 'seated') || (E.body.legs && E.posture === 'seated');

    // ---------------------------------------------------------------- config
    function setConfig(props, body) {
      props = adaptProps(props); body = adaptBody(body || (props && props.body) || null);
      E.body = bodyModel(body);
      E.props = props && typeof props === 'object' && Array.isArray(props.props) ? props : null;
      E.places = E.props && Array.isArray(E.props.places) ? E.props.places.map(normPlace).filter(Boolean) : [];
      if (!E.places.some(p => p.id === 'floor')) E.places.push({id: 'floor', name: 'the floor', aliases: ['floor', 'ground'], center: [0, 0, 0.5], size: [3, 2], top_y: 0, spots: [[-0.28, 0, 0.55], [0.28, 0, 0.55], [0, 0, 0.62], [-0.5, 0, 0.45], [0.5, 0, 0.45]], floor: true});
      const R = E.props && E.props.reach || {};
      E.reach = {l: new ReachMap(R.l || R.left || R.screen_left, 'l', E.body), r: new ReachMap(R.r || R.right || R.screen_right, 'r', E.body)};
      E.P = [];
      if (E.props) for (const d of E.props.props.slice(0, MAX_SLOTS)) {   // objects engine: 16 slots (rig schema prop0-15)
        const k = Number.isInteger(d.slot) ? d.slot : E.P.length; if (k < 0 || k >= MAX_SLOTS || E.P.some(p => p.k === k)) continue;   // slots 8-13 objects, 14-15 chalk / eraser
        const home = d.home && v3(d.home.pos) || [0.5, 0.3, 0.3];
        const hp = E.places.find(p => p.id === (d.home && d.home.place));
        E.P.push({k, id: String(d.id || 'prop' + k), def: d, name: String(d.name || d.id || 'thing'), home, homeRot: v3(d.home && d.home.rot) || [0, 0, 0],
          homePlace: hp ? hp.id : (d.home && d.home.place) || null, restH: hp ? home[1] - hp.top_y : Math.max(0.02, +d.size || 0.06),
          size: clamp(+d.size || 0.06, 0.01, 0.5), pos: home.slice(), rot: [0, 0, 0], where: 'home', place: hp ? hp.id : null,
          held: null, att: {l: 0, r: 0}, hoff: [0, 0, 0], hrot: [0, 0, 0], free: true});
      }
      E.chain = null; E.oqueue = null; E.lastRef = null; E.move = null; E.queue = null;
      E.view = null; E.viewW = 0; E.viewWv = 0; E.viewCam = null;      // scene
      for (const s of ['l', 'r']) { E.paw[s] = mkPaw(); }
      E.base = restBase(); E.cur = restBase(); E.posture = E.body.legs ? 'standing' : 'seated'; E.lying = null; E.sitOn = null; E.activity = null; E.turnBack = false;
      E.jobs = []; E.feet = null; E.rootOut = null; E.frozen = false; E.bend = {y: 0, p: 0}; E.bendT = {y: 0, p: 0};
      for (const p of E.P) { p.s = +(p.def.articulation ?? p.def.s ?? 0) || 0; p.hh = 0; p.worn = false; feat(p); }
      for (const pl of E.places) { pl.c0 = null; }
      syncPlaces();
      objInit();   // objects engine
      return {props: E.P.length, places: E.places.length, reach: {l: E.reach.l.kind, r: E.reach.r.kind}, stand: E.body.stand};
    }
    function normPlace(p) {
      if (!p || !p.id) return null;
      if (p.placeable === false) return null;     // scene: the TV is a place to stand by, never to put things on
      const c = v3(p.center) || [0, 0, 0];
      let sz = Array.isArray(p.size) ? p.size.map(Number) : [0.4, 0.4];
      if (p.plane || p.screen) sz = [sz[0], 0.06];   // a board / screen place's size is its picture (w, h); its footprint is thin
      const ty = Number.isFinite(+p.top_y) ? +p.top_y : c[1];
      const spots = Array.isArray(p.spots) && p.spots.length ? p.spots.map(v3).filter(Boolean) : [[c[0], ty, c[2]]];
      return {id: String(p.id), name: p.name || 'the ' + p.id, aliases: (p.aliases || []).map(a => String(a).toLowerCase()), center: c, size: sz, top_y: ty, spots, floor: p.id === 'floor',
        prop: p.prop || p.on_prop || p.on || p.attach || p.furniture || null, yaw: Number.isFinite(+p.yaw) ? +p.yaw : 0 /* objects: a turned place (the board tray) */, stand_top_y: Number.isFinite(+p.stand_top_y) ? +p.stand_top_y : undefined,
        yaw0: +p.yaw || 0, sub: !!(p.on_place || p.plane || p.screen)};   // a tray / a board / a TV is no place to set things down (only its own tools go back there)
    }

    // ---------------------------------------------------------------- root state helpers
    function curPose() { return Object.assign({}, E.cur); }
    function rootVel(c) { return E.vel[c] || 0; }
    function rootAcc(c) { return E.acc[c] || 0; }
    function R_of(p) { return rotm(p.yaw, p.pitch, p.roll); }
    function atRest(tol = 0.01) {
      const b = E.base, c = E.cur;
      return !E.move && Math.abs(wrap(c.yaw)) < tol && Math.abs(wrap(c.pitch)) < tol && Math.abs(wrap(c.roll)) < tol && Math.abs(c.x) < tol && Math.abs(c.z) < tol && Math.abs(c.stand) < tol &&
        Math.abs(b.yaw) < tol && Math.abs(b.pitch) < tol && Math.abs(b.roll) < tol && Math.abs(b.x) < tol && Math.abs(b.z) < tol && !b.stand;
    }
    const restBase = () => ({x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0, squash: 0, stand: 0});

    // ---------------------------------------------------------------- Plan: tracks + extras + phases
    function Plan(name, args, src) {
      const st = curPose();
      // whole turns out of the start angles (invisible), base kept consistent
      for (const c of ANG) { const k = Math.round(st[c]); st[c] -= k; E.cur[c] -= k; }
      this.name = name; this.args = args; this.src = src; this.t0 = t(); this.start = st;
      this.T = {}; for (const c of ROOT) this.T[c] = new Track(st[c], ANG.has(c));
      this.x = []; this.ph = []; this.zoom = 0; this.gaze = args.gaze || 'user'; this.headW = 0.75; this.whole = true; this.dwellHold = false;
      this.vel0 = {}; this.acc0 = {}; for (const c of ROOT) { this.vel0[c] = rootVel(c); this.acc0[c] = rootAcc(c); }
      this.count = {done: 0, want: 0, unit: ''}; this.stopFn = null; this.endBase = null; this.endPosture = null;
      this.pivot = null; this.kappa = null;
      if (E.body.legs) { if (!E.feet) initFeet(); this.footStart = {l: E.feet.l.slice(), r: E.feet.r.slice()}; }
    }
    Plan.prototype.te = function () { let m = 0; for (const c of ROOT) m = Math.max(m, this.T[c].te); return m; };
    Plan.prototype.sync = function (tt) { const m = tt === undefined ? this.te() : tt; for (const c of ROOT) this.T[c].until(m); return m; };
    Plan.prototype.extra = function (t0, t1, fn, opt = {}) { this.x.push({t0, t1, fn, att: opt.att ?? 0.25, rel: opt.rel ?? 0.35}); };
    Plan.prototype.phase = function (tt, name, did) { this.ph.push({t: tt, name, did, done: false}); };

    // ---------------------------------------------------------------- building blocks (append at the plan's current end)
    // settle every root channel to `b` (the end base) with min-jerk; angles to the nearest equivalent turn
    function settleTo(P, b, d) {
      const tt = P.sync();
      for (const c of ROOT) {
        const tr = P.T[c]; let target = b[c];
        if (ANG.has(c)) target = b[c] + Math.round(tr.v - b[c]);
        const dd = Math.max(0.05, d * (Math.abs(target - tr.v) > 1e-4 ? 1 : 0.2));
        if (Math.abs(target - tr.v) > 1e-5) tr.to(target, d); else tr.hold(dd);
      }
      return P.sync(tt + d);
    }
    // ballistic hop of the lowest point: crouch → launch → flight (gravity G) → land → recover; rotations added by caller
    function jumpOnce(P, h, sp, opt = {}) {
      const legs = !!E.body.legs; if (legs) h *= 0.7;   // legs (the muse): a small plush — stronger gravity in H units, lower hops, the knees take the landing
      const G = 12 * sp * sp, Tair = 2 * Math.sqrt(2 * Math.max(0.01, h) / G), v0 = G * Tair / 2;
      const crouch = (opt.crouch ?? 0.17) / sp, ext = 0.07 / sp, land = 0.07 / sp, rec = (opt.rec ?? 0.32) / sp;
      const t0 = P.sync();
      const sq = P.T.squash, y = P.T.y;
      const cA = opt.chain ? sq.v : 0;   // chained from a landing: already squashed
      const sk = legs ? 0.55 : 1;
      sq.to(Math.max(cA, (opt.crouchSq ?? 0.22) * sk), crouch * (opt.chain ? 0.6 : 1)).to(-0.22 * sk * clamp(h / 0.6, 0.4, 1.2), ext);
      if (E.body.legs) { const t1 = sq.te, D = t1 - y.te; y.to(-0.06 * clamp(h / 0.6, 0.5, 1.2), D * 0.72).to(0, D * 0.28); y.until(t1); }
      y.until(sq.te);
      const tTake = sq.te;
      y.fn(Tair, tau => Math.max(0, v0 * tau - 0.5 * G * tau * tau));
      sq.to(0, Tair * 0.45).to(-0.08, Tair * 0.45).hold(Tair * 0.1 - 0.0001);
      const tLand = tTake + Tair;
      sq.until(tLand);
      const imp = clamp(0.12 + 0.045 * v0, 0.1, 0.42) * (opt.impact ?? 1) * (legs ? 0.5 : 1);
      if (!opt.next) {
        sq.to(imp, land).rel(rec, (u, tau) => -imp * u + imp * (1 - u) * 0 + 0.0 * tau).to(0, 0.001);
        // damped follow-through wobble
        const tW = sq.te; sq.rel(0.4 / sp, (u) => 0.05 * imp / 0.3 * Math.sin(TAU * 2.2 * u) * (1 - u) * (1 - u));
        void tW;
      } else sq.to(imp * 0.8, land);
      if (E.body.legs) { const t2 = sq.te, D = Math.max(0.15, t2 - y.te); y.to(-0.05 * clamp(v0 / 3, 0.4, 1.2), D * 0.35).to(0, D * 0.65); }
      y.until(sq.te);
      // opt.keepRoot: leave the other root tracks where they are, so a caller can still lay its rotation onto
      // the airborne window (a sync here pushed the flip's pitch track past the landing: it then tumbled on the rug)
      return {tTake, tLand, Tair, v0, end: opt.keepRoot ? Math.max(sq.te, y.te) : P.sync(), t0};
    }
    // a little shuffle-hop while turning in place / a hop-waddle step: lowest-point arc + squash pulse
    function shuffleHops(P, t0, dur, n, hh, sp) {
      if (E.body.legs) return;   // legs: the feet step (planSteps) instead of hopping
      const y = P.T.y, sq = P.T.squash;
      y.until(t0); sq.until(t0);
      const each = dur / Math.max(1, n);
      for (let i = 0; i < n; i++) {
        y.rel(each, u => hh * Math.sin(Math.PI * clamp((u - 0.12) / 0.62, 0, 1)) ** 1.4);
        sq.rel(each, u => 0.07 * pulse(u, 0, 0.1, 0.12, 0.22) - 0.04 * Math.sin(Math.PI * clamp((u - 0.12) / 0.62, 0, 1)) + 0.09 * pulse(u, 0.7, 0.78, 0.82, 1.0));
      }
    }
    // the head looks where the body will be in 0.18 s (lead), and tucks into flips; arms tuck/extend — a generic extra
    function airArms(P, t0, t1, tuck) {
      P.extra(t0, t1, (tt) => {
        const u = clamp((tt - t0) / Math.max(1e-3, t1 - t0), 0, 1), out = {};
        const up = Math.sin(Math.PI * u);
        if (tuck) HPs(HPs(out, 'l', lerp(0.35, -0.3, up), lerp(0.55, 0.35, up), 0.3, lerp(1, 0.2, up), 0.5), 'r', lerp(0.35, -0.3, up), lerp(0.55, 0.35, up), 0.3, lerp(1, 0.2, up), 0.5);
        else HPs(HPs(out, 'l', 0.45, 0.4 + 0.5 * up, 0.15, 1, 0.8), 'r', 0.45, 0.4 + 0.5 * up, 0.15, 1, 0.8);
        return {set: out, add: {eye_wide: 0.35 * up, smile: 0.35, mouth_open: 0.12 * up}};
      }, {att: 0.12, rel: 0.3});
    }

    // ---------------------------------------------------------------- the move builders
    // each: (P, a) → description object {did:[present, past], ...} or {error}
    const B = {};
    B.spin = function (P, a, twirl) {
      const n = clamp(Math.abs(num(a.count, twirl ? 1 : 1)), 0.25, 10), sg = yawSign(a.dir, 1), sp = speedOf(a.speed);
      P.count = {want: n, unit: 'turn'};
      const rate = (twirl ? 1.7 : 1.25) * sp;                    // turns per second at cruise
      const wind = 0.22 / sp, dur = (n / rate) + 0.45 / sp, settle = 0.4 / sp;
      const t0 = P.sync();
      P.T.yaw.to(P.T.yaw.v - 0.025 * sg, wind);
      P.T.squash.to(twirl ? -0.05 : 0.07, wind);
      const tS = P.T.yaw.te, y0 = P.T.yaw.v;
      const frac = n - Math.floor(n), endsTurned = frac > 1e-3;
      P.T.yaw.rel(dur, u => sg * (n + 0.025 + 0.02) * cruise(u, Math.min(0.4, 0.35 / Math.max(1, n) + 0.12)));
      P.T.squash.to(twirl ? -0.2 : -0.06, dur * 0.25).hold(dur * 0.5).to(0.04, dur * 0.25);
      if (twirl) shuffleHops(P, tS, 0.001, 0, 0, sp);
      const tE = P.T.yaw.te;
      P.T.yaw.to(y0 + sg * n, settle);                              // follow-through: a hair past, back
      P.T.squash.until(tE).rel(settle, u => 0.04 * (1 - u) * Math.cos(TAU * 1.2 * u) - 0.04 * u);
      P.sync();
      P.zoom = twirl ? 0.25 : 0.15;
      P.extra(tS, tE + settle, (tt) => {
        const u = clamp((tt - tS) / (tE - tS), 0, 1), e = Math.sin(Math.PI * u);
        const o = twirl ? HPs(HPs({}, 'l', 0.0, 1.12, 0.12, 0.8, 0.6), 'r', 0.0, 1.12, 0.12, 0.8, 0.6) : HPs(HPs({}, 'l', 0.3 + 0.2 * e, 0.3 + 0.15 * e, 0.1, 0.7, 0.4), 'r', 0.3 + 0.2 * e, 0.3 + 0.15 * e, 0.1, 0.7, 0.4);
        return {set: o, add: {smile: 0.35, ear_l: -0.25 * e, ear_r: -0.25 * e, tail_raise: 0.3 * e, squint: 0.15 * e}};
      }, {att: 0.25, rel: 0.45});
      P.prog = () => Math.abs(P.T.yaw.at(t() - P.t0) - P.start.yaw);
      P.stopFn = (tt) => brakeRotation(P, tt, 'yaw', 1);
      if (endsTurned) { P.endBase = Object.assign({}, E.base, {yaw: wrapH(E.base.yaw + sg * frac)}); P.dwell = a.hold ? Infinity : 2.5; }
      const what = twirl ? 'twirling' : 'spinning';
      const desc = n === 1 ? 'around once' : n === 0.5 ? 'half a turn' : turnsWord(n);
      return {did: [`${what} ${desc} to ${ownWord(sg)}`, `${twirl ? 'twirled' : 'spun'} ${desc} to ${ownWord(sg)}`], sg, n};
    };
    B.twirl = (P, a) => B.spin(P, a, true);
    function brakeRotation(P, tt, c, unit) {   // decelerate (matching velocity) to the nearest whole `unit` ahead, then settle
      const tr = P.T[c], p = tr.at(tt), v = tr.vel(tt), aa = 0;
      const stopAt = p + v * 0.22, target = Math.round(stopAt / unit) * unit + (P.start[c] - Math.round(P.start[c]));
      const d = clamp(2 * Math.abs(target - p) / Math.max(0.4, Math.abs(v)), 0.3, 0.9);
      for (const k of ROOT) P.T[k].truncate(tt);
      tr.quint(target, d, v, aa);
      P.T.squash.to(0, d);
      return settleTo(P, P.endBase || E.base, 0.3);
    }
    // turn to a side / away and (unless hold) back after a dwell
    B.turn = function (P, a, away) {
      const sp = speedOf(a.speed);
      let sg = yawSign(a.dir, 0), amt;
      if (away) { amt = 0.47; if (!sg) sg = rnd() < 0.5 ? 1 : -1; }
      else {
        const d = normDir(a.dir);
        if (/^(back|around|away)$/.test(d)) return B.turn(P, Object.assign({}, a, {dir: undefined}), true);
        if (/^(forward|front|camera|me|user)$/.test(d)) return B.turn_back(P, a);
        if (!sg) sg = 1;
        amt = Number.isFinite(+a.amount) ? clamp(Math.abs(+a.amount), 0.05, 0.5) : 0.25;
      }
      const target = sg * amt, cur = P.T.yaw.v, dist = Math.abs(target - cur);
      const dur = (0.45 + 1.1 * dist) / sp;
      const t0 = P.sync();
      if (E.body.legs) legWalk(P, {x: P.T.x.v, z: P.T.z.v, yaw: target}, sp);
      else {
        P.T.yaw.to(target, dur, u => cruise(u, 0.42));
        shuffleHops(P, t0, dur, Math.max(1, Math.round(dist / 0.13)), 0.018, sp);
        P.T.roll.to(-0.008 * sg, dur * 0.5).to(0, dur * 0.5);
      }
      P.sync();
      if (away) { P.gaze = a.gaze === 'user' ? 'user' : 'free'; P.whole = true; }
      P.phase(t0 + dur, away ? 'turned_away' : 'turned', away ? 'turned away from him' : `turned to ${ownWord(sg)}`);
      P.endBase = Object.assign({}, E.base, {yaw: target});
      P.dwell = a.hold ? Infinity : num(a.duration_ms, 0) > 0 ? num(a.duration_ms, 0) / 1000 : 2.5;
      P.facing = away ? 'away' : (sg > 0 ? 'left' : 'right');
      P.extra(t0, t0 + dur, tt => ({add: {ear_l: -0.1, ear_r: -0.1, tail_raise: 0.15}}), {att: 0.2, rel: 0.3});
      P.stopFn = (tt) => { for (const k of ROOT) P.T[k].truncate(tt); return settleTo(P, P.endBase, 0.35); };
      return away ? {did: ['turning away from him', 'turned away from him'], back: true} : {did: [`turning to ${ownWord(sg)}`, `turned to ${ownWord(sg)}`]};
    };
    B.turn_away = (P, a) => B.turn(P, a, true);
    B.turn_back = function (P, a) {
      const sp = speedOf(a.speed), cur = P.T.yaw.v, dist = Math.abs(wrap(cur));
      const t0 = P.sync();
      if (dist < 0.02 && !E.base.yaw) { P.T.squash.to(0.04, 0.2).to(0, 0.3); P.sync(); return {did: ['facing him (it already was)', 'faced him (it already was)'], already: true}; }
      const dur = (0.45 + 1.1 * dist) / sp, target = Math.round(cur);
      if (E.body.legs) legWalk(P, {x: P.T.x.v, z: P.T.z.v, yaw: target}, sp);
      else { P.T.yaw.to(target, dur, u => cruise(u, 0.42)); shuffleHops(P, t0, dur, Math.max(1, Math.round(dist / 0.13)), 0.018, sp); }
      P.sync();
      P.endBase = Object.assign({}, E.base, {yaw: 0, x: P.T.x.v, z: P.T.z.v});
      return {did: ['turning back to face him', 'turned back to face him']};
    };
    // jumps / hops / bounces / flips
    B.jump = function (P, a, kind = 'jump') {
      const sp = speedOf(a.speed);
      const n = clamp(Math.round(num(a.count, 1)), 1, kind === 'jump' ? 8 : 12);
      const hk = heightOf(a.height, kind === 'jump' ? 0.55 : kind === 'hop' ? 0.25 : 0.3);
      // on legs (the muse) jumpOnce lowers every jump to 0.7x, which left a hop at ~0.11 — below its own bounce, and
      // read as a little bob, not a jump (eval jump-01 / jump-06). A hop there is a real small jump: both feet clearly
      // leave the ground, still well under a jump. Characters without legs keep their hop exactly.
      const legs = !!(E.body && E.body.legs);
      const h = kind === 'jump' ? lerp(0.25, 1.1, hk) : kind === 'hop' ? (legs ? lerp(0.24, 0.6, hk) : lerp(0.08, 0.4, hk)) : lerp(0.08, 0.45, hk);
      P.count = {want: n, unit: kind};
      P.zoom = clamp(0.2 + 0.7 * h, 0, 0.9);
      let jj = null; const lands = [];
      for (let i = 0; i < n; i++) {
        jj = jumpOnce(P, h, sp, {chain: i > 0, next: i < n - 1, crouch: i === 0 ? 0.18 : 0.08, crouchSq: kind === 'bounce' ? 0.16 : 0.22, rec: kind === 'jump' ? 0.32 : 0.22});
        lands.push(jj.tLand);
        airArms(P, jj.tTake - 0.05, jj.tLand + 0.1, false);
        P.phase(jj.tLand, 'landed', null);
      }
      P.lands = lands;
      P.prog = () => lands.filter(x => x <= t() - P.t0).length;
      P.stopFn = (tt) => stopAirborne(P, tt);
      const word = kind === 'jump' ? ['jumping', 'jumped'] : kind === 'hop' ? ['hopping', 'hopped'] : ['bouncing', 'bounced'];
      const hw = a.height !== undefined ? (hk < 0.35 ? ' a little' : hk > 0.75 ? ' high' : '') : '';
      return {did: [`${word[0]}${hw} ${times(n)}`, `${word[1]}${hw} ${times(n)}`], n};
    };
    B.hop = (P, a) => B.jump(P, a, 'hop');
    B.bounce = (P, a) => B.jump(P, a, 'bounce');
    function stopAirborne(P, tt) {   // finish the jump it is in (can't stop in the air), drop the rest, recover
      const L = P.lands || [], next = L.find(x => x >= tt - 0.02);
      if (next === undefined) return P.te();
      const end = next + 0.05;
      if (end >= P.te() - 0.1) return P.te();
      for (const k of ROOT) P.T[k].truncate(end);
      P.lands = L.filter(x => x <= next);
      P.T.squash.to(0.25, 0.06).to(0, 0.32);
      return settleTo(P, P.endBase || E.base, 0.25);
    }
    B.flip = function (P, a, fwd) {
      const sp = speedOf(a.speed), d = normDir(a.dir);
      if (fwd === undefined) fwd = /^(forward|front|frontflip)$/.test(d);
      const n = clamp(Math.round(num(a.count, 1)), 1, 3);
      const hk = heightOf(a.height, 0.6), h = lerp(0.95, 1.35, hk) + 0.4 * (n - 1);   // a little more air for the turn
      P.count = {want: n, unit: 'flip'}; P.zoom = 0.9; P.flipAir = true;
      const sg = fwd ? 1 : -1;
      // anticipation: a crouch, the arms swing back, a little lean opposite the rotation
      const j = jumpOnce(P, h, sp, {crouch: 0.24, crouchSq: 0.26, rec: 0.38, impact: 0.85, keepRoot: true});
      const p0 = P.start.pitch, tA = j.t0, lead = 0.06 / sp;
      const tr = P.T.pitch; tr.until(tA);
      tr.to(p0 - 0.012 * sg, j.tTake - lead - tA);
      const spinT = j.Tair + lead + 0.05 / sp;
      tr.rel(spinT, u => sg * (n + 0.012 + 0.02) * cruise(u, n > 1 ? 0.14 : 0.22));   // several turns: a steadier spin (peak rate)
      tr.to(p0 + sg * n, 0.32 / sp);
      P.sync();
      airArms(P, j.tTake - 0.1, j.tLand + 0.12, true);
      P.extra(j.tTake, j.tLand, () => ({add: {ear_l: -0.4, ear_r: -0.4, tail_curl: 0.5, eye_wide: 0.3}}), {att: 0.1, rel: 0.25});
      P.lands = [j.tLand]; P.phase(j.tTake, 'airborne', null); P.phase(j.tLand, 'landed', null);
      P.prog = () => Math.abs(P.T.pitch.at(t() - P.t0) - p0);
      P.stopFn = (tt) => stopAirborne(P, tt);
      P.gaze = 'user';
      const w = fwd ? ['doing a front flip', 'did a front flip'] : ['doing a backflip', 'did a backflip'];
      if (n > 1) { w[0] = `doing ${n} ${fwd ? 'front flips' : 'backflips'}`; w[1] = `did ${n} ${fwd ? 'front flips' : 'backflips'}`; w[0] = w[0] + ' in one jump'; w[1] += ' in one jump'; }
      return {did: w, n};
    };
    B.backflip = (P, a) => B.flip(P, a, false);
    B.frontflip = (P, a) => B.flip(P, a, true);
    // walking: hop-waddle steps (seated characters) or upright steps (renderer stand), out and (unless hold) back
    B.walk = function (P, a, kind = 'walk') {
      const sp = speedOf(a.speed);
      if (E.body.legs) {
        const n = clamp(Math.round(num(a.distance, num(a.count, kind === 'step' ? 1 : 3))), 1, 12), d = normDir(a.dir);
        const fwd = /^(forward|front|closer|toward|towards|camera|me|in)$/.test(d), back = /^(back|backward|backwards|away|out)$/.test(d);
        const yaw = P.T.yaw.v, SL = Math.min(E.body.legs.maxStep, 0.2) * (kind === 'scoot' ? 0.6 : 1);
        let dir;
        if (fwd) dir = [0, 1]; else if (back) dir = [0, -1];
        else { const sg = sideSign(a.dir, 0); dir = sg ? [sg, 0] : [Math.sin(yaw * TAU), Math.cos(yaw * TAU)]; }
        const goal = {x: P.T.x.v + dir[0] * SL * n, z: P.T.z.v + dir[1] * SL * n};
        goal.yaw = back || kind === 'scoot' || fwd ? yaw : Math.round(yaw);
        legWalk(P, goal, sp, {keepFacing: back || kind === 'scoot' || fwd});
        P.endBase = Object.assign({}, E.base, {x: P.T.x.v, z: P.T.z.v, yaw: P.T.yaw.v}); P.zoom = clamp(0.25 + 0.05 * n, 0, 0.7);
        P.count = {want: n, unit: 'step'}; P.prog = () => (P.steps || []).filter(x => x.t1 <= t() - P.t0).length;
        P.stopFn = (tt) => stopLegWalk(P, tt);
        const where = fwd ? 'toward him' : back ? 'backward' : sideSign(a.dir, 0) ? `to ${sideSign(a.dir, 0) > 0 ? 'its left' : 'its right'}` : 'forward';
        return {did: [`walking ${plural(n, 'a step', 'steps')} ${where}`, `walked ${plural(n, 'a step', 'steps')} ${where}`], n};
      }
      const n = clamp(Math.round(num(a.distance, num(a.count, kind === 'step' ? 1 : 3))), 1, 8);
      const d = normDir(a.dir), fwd = /^(forward|front|closer|toward|towards|camera|me|in)$/.test(d), back = /^(back|backward|backwards|away|out)$/.test(d);
      const sg = fwd || back ? 0 : sideSign(a.dir, rnd() < 0.5 ? 1 : -1);
      const upright = E.body.stand && kind !== 'scoot';
      const stepL = (upright ? 0.2 : 0.17) * (kind === 'scoot' ? 0.8 : 1), stepT = (upright ? 0.5 : 0.42) / sp;
      const yawT = kind === 'scoot' || fwd || back ? 0 : sg * 0.2;   // 3/4 profile toward the way it goes
      P.count = {want: n, unit: 'step'}; P.zoom = clamp(0.35 + 0.07 * n, 0, 0.85);
      if (upright && !E.cur.stand) standUp(P, sp);
      const steps = [];
      const go = (dirx, dirz, k, facing) => {
        if (Math.abs(P.T.yaw.v - facing) > 0.01) { const tt = P.sync(); P.T.yaw.to(facing, 0.45 / sp, u => cruise(u, 0.42)); shuffleHops(P, tt, 0.45 / sp, 1, 0.015, sp); P.sync(); }
        for (let i = 0; i < k; i++) {
          const tt = P.sync(), x0 = P.T.x.v, z0 = P.T.z.v, side = i % 2 ? 1 : -1;
          P.T.x.to(x0 + dirx * stepL, stepT, u => mj(clamp((u - 0.1) / 0.75, 0, 1)));
          P.T.z.to(z0 + dirz * stepL, stepT, u => mj(clamp((u - 0.1) / 0.75, 0, 1)));
          if (upright) { P.T.y.rel(stepT, u => 0.025 * Math.sin(Math.PI * u)); P.T.squash.rel(stepT, u => -0.03 * Math.sin(Math.PI * u)); }
          else shuffleHops(P, tt, stepT, 1, 0.06, sp);
          P.T.roll.to(0.014 * side, stepT * 0.5).to(0, stepT * 0.5);
          P.T.yaw.rel(stepT, u => 0.012 * side * Math.sin(Math.PI * u));
          steps.push(tt + stepT);
        }
        P.sync();
      };
      const dx = fwd || back ? 0 : sg, dz = fwd ? 1 : back ? -1 : 0;
      go(dx, dz, n, P.T.yaw.v + yawT);
      P.outEnd = P.te(); P.steps = steps.slice();
      const tOut = P.sync(); P.T.y.hold(0.4 / sp); P.T.squash.hold(0.4 / sp); P.sync();
      P.phase(tOut, 'arrived', null);
      const stay = !!a.hold;
      if (!stay) {
        P.retStart = P.te();
        go(-dx, -dz, n, Math.round(P.T.yaw.v) - yawT);
        const tt = P.sync(); P.T.yaw.to(Math.round(P.T.yaw.v), 0.45 / sp, u => cruise(u, 0.42)); P.T.x.to(0, 0.45 / sp); P.T.z.to(0, 0.45 / sp); shuffleHops(P, tt, 0.45 / sp, 1, 0.015, sp);
      } else P.endBase = Object.assign({}, E.base, {x: P.T.x.v, z: P.T.z.v, yaw: P.T.yaw.v});
      if (upright && !stay) { P.sync(); sitDown(P, sp); }
      P.sync();
      P.extra(0, P.te(), (tt) => ({add: {smile: 0.2, tail_raise: 0.2}, set: HPs(HPs({}, 'l', 0.1, 0.12 + 0.05 * Math.sin(TAU * tt / (2 * stepT)), 0.2, 0.4, 0.1), 'r', 0.1, 0.12 - 0.05 * Math.sin(TAU * tt / (2 * stepT)), 0.2, 0.4, 0.1)}), {att: 0.3, rel: 0.4});
      P.prog = () => steps.filter(x => x <= t() - P.t0).length;
      P.stopFn = (tt) => stopWalk(P, tt, dx, dz, yawT, stepL, stepT, sp);
      const where = fwd ? 'toward him' : back ? 'away from him' : `to ${sg > 0 ? 'its left' : 'its right'}`;
      const how = upright ? 'walking' : kind === 'scoot' ? 'scooting' : 'hopping';
      const verb = kind === 'scoot' ? ['scooting', 'scooted'] : upright ? ['walking', 'walked'] : ['hop-walking', 'hop-walked'];
      void how;
      return {did: [`${verb[0]} ${plural(n, 'a step', 'steps')} ${where}${stay ? ' and staying there' : ' and back'}`, `${verb[1]} ${plural(n, 'a step', 'steps')} ${where}${stay ? ' and stayed there' : ' and came back'}`], n};
    };
    B.step = (P, a) => B.walk(P, a, 'step');
    B.scoot = (P, a) => B.walk(P, Object.assign({}, a, {dir: a.dir || 'screen_right'}), 'scoot');
    function stopWalk(P, tt, dx, dz, yawT, stepL, stepT, sp) {
      const S = P.steps || [], next = S.find(x => x >= tt - 0.01);
      if (P.retStart !== undefined && tt >= P.retStart) return P.te();   // already coming back
      const cut = next !== undefined ? next : tt;
      for (const k of ROOT) P.T[k].truncate(cut);
      P.steps = S.filter(x => x <= cut);
      // come straight back to its spot facing him
      const tt2 = P.sync(), dist = Math.hypot(P.T.x.v, P.T.z.v), k = Math.max(1, Math.round(dist / stepL));
      const dur = k * stepT;
      P.T.x.to(0, dur); P.T.z.to(0, dur); P.T.yaw.to(Math.round(P.T.yaw.v), dur, u => cruise(u, 0.4));
      shuffleHops(P, tt2, dur, k, 0.05, sp);
      P.sync();
      if (P.T.stand.v > 0.01) sitDown(P, sp);
      P.endBase = null;
      return settleTo(P, E.base.x || E.base.z ? restBase() : E.base, 0.25);
    }
    B.come_back = function (P, a) {
      const sp = speedOf(a.speed), dist = Math.hypot(P.T.x.v, P.T.z.v);
      if (E.body.legs) {
        if (dist < 0.03 && Math.abs(wrap(P.T.yaw.v)) < 0.02) return {did: ['staying in its spot (it is already there)', 'stayed in its spot'], already: true};
        legWalk(P, {x: 0, z: 0, yaw: 0}, sp); P.endBase = Object.assign({}, E.base, {x: 0, z: 0, yaw: 0}); P.zoom = 0.3;
        P.stopFn = (tt) => stopLegWalk(P, tt);
        return {did: ['walking back to the middle, facing him', 'walked back to the middle, facing him']};
      }
      if (dist < 0.02 && Math.abs(wrap(P.T.yaw.v)) < 0.02) return {did: ['staying in its spot (it is already there)', 'stayed in its spot (it was already there)'], already: true};
      const k = Math.max(1, Math.round(dist / 0.17)), dur = k * 0.42 / sp, tt = P.sync();
      P.T.x.to(0, dur); P.T.z.to(0, dur); P.T.yaw.to(Math.round(P.T.yaw.v), dur, u => cruise(u, 0.4));
      shuffleHops(P, tt, dur, k, 0.05, sp); P.sync();
      P.endBase = restBase(); P.zoom = 0.4;
      return {did: ['coming back to its spot', 'came back to its spot']};
    };
    function standUp(P, sp) {
      const tt = P.sync();
      P.T.squash.to(0.12, 0.25 / sp).to(-0.06, 0.55 / sp).to(0, 0.3 / sp);
      P.T.stand.until(tt + 0.2 / sp).to(1, 0.75 / sp);
      P.T.pitch.to(0.02, 0.3 / sp).to(0, 0.5 / sp);
      P.sync();
    }
    function sitDown(P, sp) {
      P.sync();
      P.T.pitch.to(0.015, 0.3 / sp).to(0, 0.5 / sp);
      P.T.stand.to(0, 0.7 / sp);
      P.T.squash.hold(0.45 / sp).to(0.12, 0.15 / sp).to(0, 0.35 / sp);
      P.sync();
    }
    B.stand = function (P, a) {
      const sp = speedOf(a.speed);
      if (E.posture === 'standing' || P.T.stand.v > 0.9) return {did: ['standing (it already is)', 'stood (it already was)'], already: true};
      const t0 = P.sync();
      if (E.body.stand) {
        standUp(P, sp); P.zoom = 0.5;
        P.endBase = Object.assign({}, E.base, {stand: 1}); P.endPosture = 'standing';
        P.dwell = a.hold ? Infinity : 5;
        P.phase(P.te(), 'standing', 'stood up');
        return {did: ['standing up', 'stood up']};
      }
      // no legs to stand on: rise up tall in its seat (stretch, chest out, chin up), held a while
      P.T.squash.to(0.1, 0.22 / sp).to(-0.26, 0.5 / sp);
      P.sync();
      P.endBase = Object.assign({}, E.base, {squash: -0.26}); P.endPosture = 'tall';
      P.dwell = a.hold ? Infinity : 3;
      P.extra(t0, Infinity, () => ({set: HPs(HPs({}, 'l', 0.12, 0.2, 0.15, 0.6, 0.2), 'r', 0.12, 0.2, 0.15, 0.6, 0.2), add: {body_pitch: -0.15, head_pitch: -0.07, ear_l: 0.35, ear_r: 0.35, eye_wide: 0.15, smile: 0.2}}), {att: 0.4, rel: 0.4});
      P.whole = true;
      return {did: ['rising up tall in its seat (it has no legs to stand on)', 'rose up tall in its seat (it has no legs to stand on)'], approx: true};
    };
    B.sit = function (P, a) {
      const sp = speedOf(a.speed);
      if (E.posture === 'lying') return B.get_up(P, a);
      if (E.posture === 'standing' || P.T.stand.v > 0.05) {
        sitDown(P, sp); P.endBase = Object.assign({}, E.base, {stand: 0}); P.endPosture = 'seated';
        return {did: ['sitting back down', 'sat back down']};
      }
      if (E.posture === 'tall' || E.base.squash) { P.T.squash.to(0, 0.4 / sp); P.sync(); P.endBase = Object.assign({}, E.base, {squash: 0}); P.endPosture = 'seated'; }
      // already sitting: settle in (a little bounce and wiggle into the seat)
      P.T.squash.to(0.1, 0.18 / sp).to(-0.03, 0.22 / sp).to(0.03, 0.2 / sp).to(0, 0.25 / sp);
      P.T.roll.to(0.01, 0.2 / sp).to(-0.01, 0.25 / sp).to(0, 0.3 / sp);
      P.sync();
      P.whole = false; P.headW = 0.2;
      P.extra(0, P.te(), () => ({add: {smile: 0.25, squint: 0.2}}));
      return {did: ['settling into its seat (it is already sitting)', 'settled into its seat (it was already sitting)']};
    };
    // lying down: tips over onto its side (about the base edge), or onto its back / front; holds or gets up again
    B.lie_down = function (P, a) {
      const sp = speedOf(a.speed), d = normDir(a.dir);
      const how = /back|belly_up|up/.test(d) || a.dead ? 'back' : /front|tummy|belly|stomach|face/.test(d) ? 'front' : 'side';
      const sg = how === 'side' ? sideSign(a.dir, rnd() < 0.5 ? 1 : -1) : 0;
      if (E.posture === 'lying' && E.lying === how && (how !== 'side' || Math.sign(E.base.roll) === sg)) return {did: ['lying down (it already is)', 'lay down (it already was)'], already: true};
      if (E.posture === 'lying') getUp(P, sp);
      if (P.T.stand.v > 0.05) { sitDown(P, sp); }
      const t0 = P.sync();
      const tgt = how === 'side' ? {roll: 0.25 * sg} : how === 'back' ? {pitch: -0.23} : {pitch: 0.2};
      const k = Object.keys(tgt)[0], v = tgt[k];
      const tr = P.T[k];
      tr.to(-0.012 * Math.sign(v), 0.3 / sp);                      // lean the other way first
      P.T.squash.to(0.08, 0.3 / sp);
      tr.to(v * 0.985, 0.8 / sp, u => u < 0.7 ? 0.82 * MJI(u / 0.7) / 0.5 * 0.5 * (u / 0.7) ** 0 * 1 : 0);   // placeholder, replaced below
      // the fall: slow start (tipping), gravity speeds it up, a soft landing with a small bounce
      tr.s.pop(); tr.te = tr.s.length ? tr.s[tr.s.length - 1].t1 : 0; tr.v = -0.012 * Math.sign(v);
      const vS = tr.v; tr.rel(0.75 / sp, u => (v * 1.006 - vS) * (u * u * (3 - 2 * u) * 0.35 + 0.65 * mj(u)));
      tr.to(v, 0.35 / sp, u => u + 0.0 * Math.sin(Math.PI * u));
      P.T.squash.to(0, 0.4 / sp).to(0.1, 0.12 / sp).to(0, 0.3 / sp);
      P.sync();
      if (how === 'side') P.pivot = {pe: [sg * E.body.s[0] * 0.9, 0, E.body.E[2]], until: Infinity};
      if (how !== 'side') P.pivot = {pe: [0, 0, E.body.E[2] + (how === 'back' ? -1 : 1) * E.body.s[2] * 0.9], until: Infinity};
      P.endBase = Object.assign({}, E.base, {[k]: v}); P.endPosture = 'lying'; P.lyingHow = how;
      P.zoom = 0.55;
      P.dwell = a.hold ? Infinity : num(a.duration_ms, 0) > 0 ? num(a.duration_ms, 0) / 1000 : 5;
      P.phase(P.te(), 'lying', null);
      const sgHead = how === 'side' ? -sg : 0;
      P.extra(t0 + 0.3 / sp, Infinity, () => {
        const o = how === 'back' ? HPs(HPs({}, 'l', 0.15, a.dead ? 0.85 : 0.5, 0.25, a.dead ? 0.3 : 0.7, 0.5), 'r', 0.15, a.dead ? 0.85 : 0.5, 0.25, a.dead ? 0.3 : 0.7, 0.5) : HPs(HPs({}, 'l', -0.25, 0.12, 0.3, 0.5, 0.1), 'r', -0.25, 0.12, 0.3, 0.5, 0.1);
        const add = {head_roll: 0.28 * sgHead, smile: a.dead ? 0 : 0.25, ear_l: -0.25, ear_r: -0.25, tail_raise: -0.2};
        if (how === 'front') add.head_pitch = -0.4;
        if (a.dead) Object.assign(add, {blink_l: 0.9, blink_r: 0.9, tongue_out: 0.7, mouth_open: 0.25});
        return {set: o, add};
      }, {att: 0.35, rel: 0.4});
      P.gaze = how === 'side' ? (a.gaze || 'user') : 'free';
      const words = how === 'side' ? `lying down on ${sg > 0 ? 'its left' : 'its right'} side` : how === 'back' ? (a.dead ? 'playing dead on its back' : 'lying down on its back') : 'lying down on its tummy';
      const past = words.replace('lying down', 'lay down').replace('playing dead', 'played dead');
      return {did: [words, past]};
    };
    function getUp(P, sp) {
      const b = E.base, k = Math.abs(b.roll) > 0.01 ? 'roll' : 'pitch', v = P.T[k].v, t0 = P.sync();
      P.T[k].to(v * 1.04, 0.25 / sp).to(-0.02 * Math.sign(v), 0.65 / sp, u => cruise(u, 0.45)).to(0, 0.3 / sp);
      P.T.squash.hold(0.8 / sp).to(0.1, 0.12 / sp).to(0, 0.3 / sp);
      P.sync();
      if (!P.pivot) P.pivot = {pe: k === 'roll' ? [Math.sign(b.roll) * E.body.s[0] * 0.9, 0, E.body.E[2]] : [0, 0, E.body.E[2] + Math.sign(b.pitch) * E.body.s[2] * 0.9], until: P.te()};
      P.extra(t0, P.te(), () => ({set: HPs(HPs({}, 'l', 0.2, 0.0, 0.3, 0.8, 0), 'r', 0.2, 0.0, 0.3, 0.8, 0), add: {smile: 0.15}}), {att: 0.2, rel: 0.35});
      P.endBase = Object.assign({}, E.base, {roll: 0, pitch: 0}); P.endPosture = 'seated';
    }
    B.get_up = function (P, a) {
      const sp = speedOf(a.speed);
      if (E.posture === 'lying' || Math.abs(E.base.roll) > 0.01 || Math.abs(E.base.pitch) > 0.01) { getUp(P, sp); P.zoom = 0.45; return {did: ['getting up', 'got up']}; }
      if (E.body.stand) return B.stand(P, a);
      if (E.posture === 'tall') return B.sit(P, a);
      return B.stand(P, a);
    };
    // rolling over: a full sideways roll on the floor (lowest point kept on the floor), then scoots back to its spot
    B.roll_over = function (P, a) {
      const sp = speedOf(a.speed), sg = sideSign(a.dir, E.base.roll ? Math.sign(E.base.roll) : (rnd() < 0.5 ? 1 : -1));
      const n = clamp(Math.round(num(a.count, 1)), 1, 3);
      const t0 = P.sync(), from = P.T.roll.v, total = sg * n - (Math.abs(from) > 0.01 ? from : 0);
      const dur = (1.1 + 1.0 * Math.abs(total)) / sp;
      P.T.roll.rel(dur, u => total * cruise(u, 0.3));
      P.T.x.rel(dur, u => sg * 0.55 * n * cruise(u, 0.3));
      P.T.squash.to(0.06, 0.2 / sp).to(-0.04, dur - 0.5 / sp).to(0.1, 0.12 / sp).to(0, 0.3 / sp);
      P.sync(); P.zoom = 0.7; P.count = {want: n, unit: 'roll'};
      P.extra(t0, t0 + dur, () => ({set: HPs(HPs({}, 'l', -0.3, 0.35, 0.3, 0.3, 0), 'r', -0.3, 0.35, 0.3, 0.3, 0), add: {smile: 0.45, squint: 0.4, ear_l: -0.3, ear_r: -0.3}}), {att: 0.2, rel: 0.4});
      const back = !a.hold;
      if (back) { const tt = P.sync(), k = Math.max(2, Math.round(Math.abs(P.T.x.v) / 0.17)), d = k * 0.4 / sp; P.T.x.to(0, d); shuffleHops(P, tt, d, k, 0.05, sp); P.sync(); }
      else P.endBase = Object.assign({}, E.base, {x: P.T.x.v, roll: 0});
      P.endPosture = 'seated';
      if (E.posture === 'lying') P.endBase = Object.assign({}, P.endBase || E.base, {roll: 0, pitch: 0});
      P.prog = () => Math.abs(P.T.roll.at(t() - P.t0) - from);
      P.stopFn = (tt) => brakeRotation(P, tt, 'roll', 1);
      return {did: [`rolling over ${times(n)}${back ? ' and scooting back' : ''}`, `rolled over ${times(n)}${back ? ' and scooted back' : ''}`], n};
    };
    B.cartwheel = function (P, a) {
      const sp = speedOf(a.speed), sg = sideSign(a.dir, rnd() < 0.5 ? 1 : -1), n = clamp(Math.round(num(a.count, 1)), 1, 3);
      P.count = {want: n, unit: 'cartwheel'}; P.zoom = 0.85;
      const lands = [];
      for (let i = 0; i < n; i++) {
        const t0 = P.sync(), dur = 1.0 / sp;
        P.T.squash.to(0.15, 0.18 / sp).to(-0.12, 0.08 / sp);
        const tS = P.T.squash.te;
        P.T.roll.until(t0).to(P.T.roll.v - 0.01 * sg, tS - t0).rel(dur, u => sg * 1.01 * cruise(u, 0.2)); P.T.roll.to(Math.round(P.T.roll.v), 0.25 / sp);
        P.T.x.until(tS).rel(dur, u => sg * 0.75 * cruise(u, 0.25));
        P.T.y.until(tS).rel(dur, u => 0.18 * Math.sin(Math.PI * u));
        P.T.squash.to(-0.1, dur * 0.8).to(0.15, dur * 0.2).to(0, 0.3 / sp);
        P.sync(); lands.push(tS + dur);
        P.extra(t0, tS + dur + 0.1, (tt) => ({set: HPs(HPs({}, 'l', 0.55, 1.05, 0.05, 1, 0.8), 'r', 0.55, 1.05, 0.05, 1, 0.8), add: {smile: 0.5, eye_wide: 0.25}}), {att: 0.15, rel: 0.35});
      }
      P.lands = lands;
      if (!a.hold) { const tt = P.sync(), k = Math.max(2, Math.round(Math.abs(P.T.x.v) / 0.17)), d = k * 0.4 / sp; P.T.x.to(0, d); shuffleHops(P, tt, d, k, 0.05, sp); P.sync(); }
      else P.endBase = Object.assign({}, E.base, {x: P.T.x.v});
      P.prog = () => lands.filter(x => x <= t() - P.t0).length;
      P.stopFn = (tt) => brakeRotation(P, tt, 'roll', 1);
      return {did: [`doing ${n === 1 ? 'a cartwheel' : n + ' cartwheels'} to ${sg > 0 ? 'its left' : 'its right'}`, `did ${n === 1 ? 'a cartwheel' : n + ' cartwheels'} to ${sg > 0 ? 'its left' : 'its right'}`], n};
    };
    B.bow = function (P, a) {
      const sp = speedOf(a.speed), n = clamp(Math.round(num(a.count, 1)), 1, 4), deep = /deep|low/.test(normDir(a.style || a.dir));
      const t0 = P.sync();
      for (let i = 0; i < n; i++) { P.T.pitch.to(deep ? 0.085 : 0.06, 0.6 / sp).hold(0.6 / sp).to(0, 0.7 / sp); }
      P.sync();
      P.pivot = {pe: [0, 0, E.body.hi[2] * 0.85], until: P.te()};
      P.extra(t0, P.te(), (tt) => {
        const u = (tt - t0) / ((P.te() - t0) / n) % 1, p = pulse(u, 0, 0.3, 0.6, 0.95);
        return {set: Object.assign({body_pitch: 0.5 * p}, HPs({}, 'l', -0.45, 0.35, 0.25, 1, 0)), add: {head_pitch: 0.25 * p, gaze_y: -0.6 * p, smile: 0.25}};
      }, {att: 0.3, rel: 0.4});
      P.whole = true; P.headW = 0.3;
      return {did: [n === 1 ? 'taking a bow' : `bowing ${times(n)}`, n === 1 ? 'took a bow' : `bowed ${times(n)}`]};
    };
    B.wiggle = function (P, a) {
      const sp = speedOf(a.speed), dur = clamp(num(a.duration_ms, 1600) / 1000, 0.6, 8) / sp, t0 = P.sync();
      P.T.roll.rel(dur, u => 0.022 * Math.sin(TAU * 3.2 * u * dur * sp) * Math.sin(Math.PI * u));
      P.T.squash.rel(dur, u => 0.05 * Math.sin(TAU * 6.4 * u * dur * sp) * Math.sin(Math.PI * u));
      P.T.yaw.rel(dur, u => 0.012 * Math.sin(TAU * 1.6 * u * dur * sp + 1) * Math.sin(Math.PI * u));
      P.sync(); P.whole = true; P.headW = 0.3;
      P.extra(t0, t0 + dur, (tt) => { const s = Math.sin(TAU * 3.2 * (tt - t0) * sp); return {add: {tail_wag: 0.8 * s, ear_l: 0.3 * s, ear_r: -0.3 * s, smile: 0.5, squint: 0.3}, set: HPs(HPs({}, 'l', 0.2 + 0.08 * s, 0.25, 0.25, 0.7, 0.2), 'r', 0.2 - 0.08 * s, 0.25, 0.25, 0.7, 0.2)}; });
      return {did: ['wiggling', 'wiggled']};
    };
    B.shimmy = function (P, a) {
      const sp = speedOf(a.speed), dur = clamp(num(a.duration_ms, 2000) / 1000, 0.6, 8) / sp, t0 = P.sync(), bpm = 112 * sp;
      P.T.yaw.rel(dur, u => 0.012 * Math.sin(TAU * 2 * (u * dur) * bpm / 60) * Math.sin(Math.PI * u));
      P.sync(); P.whole = true; P.headW = 0.2;
      P.extra(t0, t0 + dur, (tt) => { const r = DSTEP.shimmy.f((tt - t0) * bpm / 60, 1); return {set: r.set, add: r.add}; });
      return {did: ['doing a shimmy', 'did a shimmy']};
    };
    B.sway = function (P, a) {
      const sp = speedOf(a.speed), dur = clamp(num(a.duration_ms, 4000) / 1000, 1, 20) / sp, t0 = P.sync();
      P.T.roll.rel(dur, u => 0.02 * Math.sin(TAU * 0.5 * u * dur * sp) * Math.sin(Math.PI * u) ** 0.5);
      P.T.x.rel(dur, u => 0.03 * Math.sin(TAU * 0.5 * u * dur * sp) * Math.sin(Math.PI * u) ** 0.5);
      P.sync(); P.whole = true; P.headW = 0.3;
      P.extra(t0, t0 + dur, (tt) => ({add: {head_roll: -0.06 * Math.sin(TAU * 0.5 * (tt - t0) * sp), smile: 0.3}}));
      return {did: ['swaying from side to side', 'swayed from side to side']};
    };
    B.shake_off = function (P, a) {
      const sp = speedOf(a.speed), dur = 1.5 / sp, t0 = P.sync(), f = 4.5 * sp;
      const env = u => Math.sin(Math.PI * clamp(u * 1.15, 0, 1)) * (1 - 0.3 * u);
      P.T.yaw.rel(dur, u => 0.035 * Math.sin(TAU * f * u * dur) * env(u));
      P.T.roll.rel(dur, u => 0.012 * Math.sin(TAU * f * u * dur + 0.8) * env(u));
      P.T.squash.to(0.06, 0.2 / sp).hold(dur - 0.5 / sp).to(-0.04, 0.15 / sp).to(0, 0.25 / sp);
      P.sync(); P.whole = true; P.headW = 0.5;
      P.extra(t0, P.te(), (tt) => {
        const u = clamp((tt - t0) / dur, 0, 1), ph = TAU * f * (tt - t0), e = env(u);
        return {add: {head_yaw: 0.18 * Math.sin(ph + 1.3) * e, head_roll: 0.06 * Math.sin(ph + 1.8) * e, tail_wag: 0.9 * Math.sin(ph - 1.2) * e, ear_flick_l: 0.6 * Math.max(0, Math.sin(ph + 2)) * e, ear_flick_r: 0.6 * Math.max(0, Math.sin(ph + 2.4)) * e, ear_l: 0.3 * Math.sin(ph + 2) * e, ear_r: 0.3 * Math.sin(ph + 2.3) * e, squint: 0.7 * e, smile: 0.2}};
      });
      return {did: ['shaking itself off', 'shook itself off']};
    };
    B.crouch = function (P, a) {
      const sp = speedOf(a.speed), t0 = P.sync();
      P.T.squash.to(0.34, 0.3 / sp); P.sync();
      P.endBase = Object.assign({}, E.base, {squash: 0.34}); P.endPosture = 'crouched';
      P.dwell = a.hold ? Infinity : 2.5;
      P.extra(t0, Infinity, () => a.hide ? {set: HPs(HPs({}, 'l', -0.35, 1.0, 0.45, 1, 0), 'r', -0.35, 1.0, 0.45, 1, 0), add: {ear_l: -0.6, ear_r: -0.6, smile: 0.2}} : {set: HPs(HPs({}, 'l', 0.0, 0.1, 0.3, 0.6, 0), 'r', 0.0, 0.1, 0.3, 0.6, 0), add: {ear_l: -0.5, ear_r: -0.5, eye_wide: 0.3}}, {att: 0.3, rel: 0.35});
      return a.hide ? {did: ['crouching down and hiding behind its paws', 'crouched down and hid behind its paws']} : {did: ['crouching down low', 'crouched down low']};
    };
    // dance: a style = a weighted bag of composable steps on a beat grid; varied (no immediate repeats), cross-faded
    B.dance = function (P, a) {
      const sp = speedOf(a.speed), sw = String(a.style || '').toLowerCase();
      let style = null; for (const [re, k] of DSTYLE_WORDS) if (re.test(sw)) { style = k; break; }
      const unknownStyle = !!sw && !style; if (!style) style = sw ? 'happy' : ['happy', 'disco', 'silly', 'hiphop', 'salsa'][Math.floor(rnd() * 5) % 5];
      const S0 = DSTYLE[style], bpm = S0.bpm * sp, beat = 60 / bpm, k = S0.k || 1;
      const dur = clamp(num(a.duration_ms, 7000) / 1000, 2.5, 30);
      const nb = Math.max(4, Math.round(dur / beat / 2) * 2);
      // choreography: pick steps until the beats are filled
      const seq = []; let b = 0, last = '';
      while (b < nb) {
        const opts = S0.steps.filter(([n]) => n !== last || S0.steps.length === 1);
        let tot = 0; for (const [, w] of opts) tot += w; let r = rnd() * tot, pick = opts[0][0];
        for (const [n, w] of opts) { r -= w; if (r <= 0) { pick = n; break; } }
        const L = DSTEP[pick].len * (pick === 'spin' ? 1 : (rnd() < 0.5 ? 2 : 1));
        seq.push({name: pick, b0: b, len: Math.min(L, nb - b)}); b += L; last = pick;
      }
      const t0 = P.sync(), T = nb * beat, intro = 0.6 * beat, outro = beat * 1.5;
      P.seq = seq; P.beat = beat; P.style = style;
      const start = {x: P.T.x.v, y: P.T.y.v, z: P.T.z.v, yaw: P.T.yaw.v, pitch: P.T.pitch.v, roll: P.T.roll.v, squash: P.T.squash.v};
      // root channels are a function of the beat clock with turn-accumulation for spin steps
      const spinsBefore = (i) => seq.slice(0, i).filter(s => s.name === 'spin').length;
      const evalAt = (tt) => {   // tt = seconds into the dance
        const bb = tt / beat; let i = 0; while (i < seq.length - 1 && bb >= seq[i].b0 + seq[i].len) i++;
        const s = seq[i], r = DSTEP[s.name].f(bb - s.b0, k), xf = 0.5;
        let out = r, w = 1;
        if (i + 1 < seq.length && bb > s.b0 + s.len - xf) {   // cross-fade into the next step
          const nx = seq[i + 1], r2 = DSTEP[nx.name].f(Math.max(0, bb - nx.b0), k); w = mj((bb - (s.b0 + s.len - xf)) / xf);
          out = {root: {}, set: {}, add: {}};
          for (const c of ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'squash']) { const p = (r.root[c] || 0) + (c === 'yaw' && s.name === 'spin' ? 0 : 0), q = r2.root[c] || 0; out.root[c] = ANG.has(c) ? p + wrap(q - p) * w : lerp(p, q, w); }
          const ks = new Set([...Object.keys(r.set || {}), ...Object.keys(r2.set || {})]);
          for (const kk of ks) out.set[kk] = lerp((r.set || {})[kk] ?? 0, (r2.set || {})[kk] ?? 0, w);
          const ka = new Set([...Object.keys(r.add || {}), ...Object.keys(r2.add || {})]);
          for (const kk of ka) out.add[kk] = lerp((r.add || {})[kk] ?? 0, (r2.add || {})[kk] ?? 0, w);
        }
        out.turns = spinsBefore(i) + (s.name === 'spin' ? 0 : 0);
        return out;
      };
      P.danceEval = evalAt;
      const env = tt => mj(tt / intro) * (1 - mj((tt - (T - outro)) / outro));
      for (const c of ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'squash']) {
        const s0 = start[c];
        P.T[c].fn(T, tt => {
          const r = evalAt(tt), e = env(tt);
          let v = (r.root[c] || 0);
          if (c === 'yaw') v += r.turns;   // completed spin steps (whole turns, invisible)
          return s0 + e * v + (c === 'yaw' ? (1 - e) * r.turns : 0) * 0;
        });
      }
      // a little "ta-da" at the end, then settle
      P.sync(); settleTo(P, E.base, 0.45 / sp);
      P.zoom = 0.45;
      P.extra(t0, t0 + T + 0.3, (tt) => {
        const r = evalAt(Math.min(tt - t0, T - 1e-3)), e = env(tt - t0);
        const add = Object.assign({}, r.add); for (const kk in add) add[kk] *= e;
        for (const kk in (S0.expr || {})) add[kk] = (add[kk] || 0) + S0.expr[kk] * e;
        return {set: r.set, add};
      }, {att: intro, rel: 0.45});
      P.extra(t0 + T - outro, t0 + T + 0.6, (tt) => { const p = pulse(tt - (t0 + T - outro), 0, 0.25, 0.9, 1.6); return {set: HPs(HPs({}, 'l', 0.35, 0.95, 0.2, 1, 1), 'r', 0.35, 0.95, 0.2, 1, 1), add: {smile: 0.5 * p, eye_wide: 0.15 * p}, w: p}; }, {att: 0.01, rel: 0.01});
      P.prog = () => clamp((t() - P.t0 - t0) / T, 0, 1);
      P.stopFn = (tt) => {   // finish on the next beat with the outro
        const into = tt - t0; if (into >= T - outro) return P.te();
        const nb2 = Math.ceil((into + 0.5 * beat) / beat) * beat, cut = t0 + nb2;
        for (const c of ROOT) P.T[c].truncate(Math.min(cut, P.T[c].te));
        P.sync(cut); return settleTo(P, E.base, 0.5);
      };
      const sd = unknownStyle ? `a happy dance (it has no ${sw} routine)` : `a ${style === 'hiphop' ? 'hip-hop' : style} dance`;
      return {did: [`doing ${sd}`, `did ${sd}`], style, steps: seq.map(s => s.name)};
    };
    B.stop = null;   // handled by the engine

    // ---------------------------------------------------------------- run / interrupt moves
    function startMove(name, args, src, id) {
      const P = new Plan(name, args, src);
      // lying / standing / turned: compose a prelude where the move needs it
      const keepsPosture = {get_up: 1, sit: 1, stand: 1, lie_down: 1, roll_over: 1, wiggle: 1, turn_back: 1, come_back: 1, crouch: 1, shake_off: 0, sway: 1, shimmy: 1};
      let pre = '';
      if (E.posture === 'lying' && !keepsPosture[name]) { getUp(P, 1); pre = 'getting up, then '; E.posture = 'seated'; }
      else if (E.posture === 'standing' && /^(roll_over|cartwheel|lie_down)$/.test(name)) { sitDown(P, 1); pre = 'sitting down, then '; }
      else if ((E.posture === 'tall' || E.posture === 'crouched') && name !== 'stand' && name !== 'crouch') { P.T.squash.to(0, 0.3); P.sync(); E.base.squash = 0; }
      if (!E.body.legs && (E.base.x || E.base.z) && /^(walk|step|scoot|roll_over|cartwheel)$/.test(name) === false && name !== 'come_back' && !/^(turn|turn_away|turn_back|spin|twirl|sit|wiggle|sway|shimmy|bow|crouch)$/.test(name)) {
        // displaced (walked away with hold): come back first
        const r = B.come_back(P, {}); if (!r.already) pre += 'coming back to its spot, then ';
      }
      const f = B[name]; const r = f(P, args);
      if (!r || r.error) return {ok: false, reason: (r && r.error) || 'cannot do that move'};
      if (!P.endBase) P.endBase = Object.assign({}, E.base);
      // posture-changing moves that do not hold: dwell, then return
      if (P.dwell !== undefined && P.dwell !== Infinity && P.endBase) {
        const tEnd = P.sync();
        for (const c of ROOT) P.T[c].hold(P.dwell);
        P.sync(tEnd + P.dwell);
        P.returnAt = tEnd + P.dwell;
        retPlan(P, name, args);
      } else if (P.dwell === Infinity) P.holdEnd = true;
      if (!P.holdEnd && !r.keepBase) {
        // non-holding moves end in the posture they started in (or the explicit endBase they set)
        const eb = P.endBase || E.base; settleTo(P, eb, 0.001);
      }
      P.id = id; P.did = r.did; P.desc = r; P.pre = pre; P.dur = P.te();
      P.env = {a: 0.25, r: 0.4};
      // end of an interrupted / previous move: its last values are our start (the residual carries the velocity)
      E.move = P; E.active = true;
      return {ok: true, did: pre + r.did[0], plan: P};
    }
    function movePreview(name, args) {   // the builder's own words for a move (dry run on a throwaway plan; builders do not touch E)
      try { const f = B[name]; if (!f) return null; const r = f(new Plan(name, args || {}, 'preview'), args || {}); return r && !r.error && r.did ? r.did.slice() : null; } catch (e) { return null; }
    }
    function retPlan(P, name, args) {   // after a dwell: undo the posture change (turn back, get up, sit, un-crouch)
      const sp = speedOf(args.speed), eb = P.endBase;
      if (name === 'lie_down') { E.base = Object.assign({}, eb); getUp(P, sp); P.endBase = Object.assign({}, eb, {roll: 0, pitch: 0}); P.endPosture = 'seated'; if (P.pivot) P.pivot.until = Infinity; }
      else if (name === 'stand' && E.body.stand) { sitDown(P, sp); P.endBase = Object.assign({}, eb, {stand: 0}); P.endPosture = 'seated'; }
      else if (name === 'stand' || name === 'crouch') { P.T.squash.to(0, 0.45 / sp); P.sync(); P.endBase = Object.assign({}, eb, {squash: 0}); P.endPosture = 'seated'; P.xEnd = P.te(); }
      else if (E.body.legs) { legWalk(P, {x: P.T.x.v, z: P.T.z.v, yaw: Math.round(P.T.yaw.v)}, sp); P.endBase = Object.assign({}, eb, {yaw: 0, x: P.T.x.v, z: P.T.z.v}); P.endPosture = E.posture; P.returned = true; }
      else { const tt = P.sync(), cur = P.T.yaw.v, tgt = Math.round(cur), dist = Math.abs(cur - tgt), dur = (0.45 + 1.1 * dist) / sp; P.T.yaw.to(tgt, dur, u => cruise(u, 0.42)); shuffleHops(P, tt, dur, Math.max(1, Math.round(dist / 0.13)), 0.018, sp); P.sync(); P.endBase = Object.assign({}, eb, {yaw: 0}); P.endPosture = 'seated'; P.returned = true; }
      P.didReturn = true;
      for (const x of P.x) if (x.t1 === Infinity) x.t1 = P.returnAt + 0.4;
      P.facingEnd = 'camera';
    }
    function stopMove(reason) {   // graceful end of the active move (keeps whatever it holds in its paws)
      const P = E.move; if (!P || P.stopping) return;
      const tt = t() - P.t0;
      P.stopping = true; P.interrupted = reason || 'stopped';
      if (P.holdEnd && tt >= P.te() - 1e-3) {   // a held end state: release it (turn back, get up, sit, relax)
        P.holdEnd = false; P.dwell = 0; P.returnAt = tt;
        for (const c of ROOT) P.T[c].truncate(tt);
        P.sync(tt);
        retPlan(P, P.name, P.args);
        P.released = true; P.heldDone = true; P.dur = P.te();
        for (const x of P.x) if (x.t1 === Infinity || x.t1 > P.dur) x.t1 = Math.max(tt + 0.4, x.t1 === Infinity ? tt + 0.4 : x.t1);
        return;
      }
      if (P.returnAt !== undefined && tt >= P.returnAt) { P.stopping = 'finishing'; return; }   // already returning
      if (P.returnAt !== undefined && tt < P.returnAt && tt >= (P.returnAt - P.dwell)) {   // in the dwell: return now
        for (const c of ROOT) P.T[c].truncate(tt);
        P.sync(tt); P.returnAt = tt; retPlan(P, P.name, P.args); P.dur = P.te();
        for (const x of P.x) if (x.t1 > P.dur) x.t1 = P.dur;
        return;
      }
      let end;
      if (P.stopFn) end = P.stopFn(tt);
      else { for (const c of ROOT) P.T[c].truncate(tt); end = settleTo(P, P.returnAt !== undefined ? Object.assign({}, E.base) : (P.endBase || E.base), 0.45); }
      P.dwell = undefined; P.holdEnd = false;
      if (P.returnAt !== undefined && P.returnAt > tt) { P.returnAt = undefined; P.endBase = Object.assign({}, E.base); P.endPosture = E.posture; }
      P.dur = P.te();
      for (const x of P.x) if (x.t1 > P.dur) x.t1 = P.dur;
    }
    function finishMove(P, phase) {
      const tt = t() - P.t0;
      // commit the end state
      const eb = P.holdEnd ? (P.endBase || E.base) : (P.endBase || E.base);
      E.base = Object.assign({}, eb);
      for (const c of ANG) E.base[c] = wrapH(E.base[c]);
      if (P.endPosture) E.posture = P.endPosture;
      if (P.endPosture === 'lying') E.lying = P.lyingHow; else if (P.endPosture) E.lying = null;
      if (!P.holdEnd && (P.didReturn || P.endPosture === 'seated')) { if (P.name === 'lie_down' || P.name === 'get_up') { E.posture = 'seated'; E.lying = null; } }
      E.facingHold = P.holdEnd && P.facing ? P.facing : (Math.abs(E.base.yaw) > 0.05 ? (Math.abs(Math.abs(E.base.yaw) - 0.47) < 0.08 ? 'away' : E.base.yaw > 0 ? 'left' : 'right') : null);
      const prog = P.prog ? P.prog() : null, want = P.count && P.count.want;
      let did = P.did[1];
      if (phase === 'interrupted' && want && prog !== null && P.count.unit) {
        const got = P.count.unit === 'turn' || P.count.unit === 'flip' || P.count.unit === 'roll' ? Math.round(prog * 4) / 4 : Math.round(prog * 100) / 100;
        // only what happened: "spun 2 of the 3 turns, then stopped" / "started to spin (3 turns asked) but stopped before one was done"
        const u = P.count.unit, n = Math.max(0, u === 'turn' ? Math.floor(got + 0.05) : Math.floor(got)), uw = k => k === 1 ? u : u + 's';
        const V = {spin: ['spun', 'spin'], twirl: ['twirled', 'twirl'], walk: ['walked', 'walk'], step: ['stepped', 'step'], scoot: ['scooted', 'scoot'], roll_over: ['rolled', 'roll over'], come_back: ['walked', 'walk back']}[P.name] || ['did', P.name.replace(/_/g, ' ')];
        const what = u === 'turn' || u === 'step' ? uw(want) : `${P.name === 'cartwheel' ? 'cartwheel' : P.name === 'roll_over' ? 'roll' : (P.name.replace(/_/g, ' ') || u)}${want === 1 ? '' : 's'}`;
        const rel = P.released || P.interrupted === 'released';
        did = n >= want ? `${V[0]} all ${want} of the ${want} ${what}; the stop came as it was finishing` + (rel ? ', then it relaxed back' : '')
          : n > 0 ? `${V[0]} ${n} of the ${want} ${what}, then stopped` + (rel ? ' and relaxed back' : '')
          : `started to ${V[1]} (${want} ${what} asked) but stopped before one was done` + (rel ? ', then relaxed back' : '');
      } else if (phase === 'interrupted' && P.heldDone) did = P.did[1] + ', held it, then relaxed back out of it';   // the move itself was complete: only the hold ended
      else if (phase === 'interrupted') did = P.did[1] + ' (cut short)' + (P.released ? ', then relaxed back' : '');
      else if (P.released) did += ', then relaxed back';
      if (E.body.legs) {
        if (P.steps && P.steps.length) E.feet = {l: footAtPlan(P, 'l', 1e9), r: footAtPlan(P, 'r', 1e9)};
        if (P.sitOn) { E.sitOn = P.sitOn; E.posture = 'seated'; }
        if (P.unsit) { E.sitOn = null; E.posture = 'standing'; }
        if (E.posture === 'seated' && !E.sitOn) E.posture = 'standing';   // legs: "seated" without a seat = back on its feet
        if (P.drag) dragFrame(P, Object.assign({}, E.cur));
      }
      if (P.opDid) { P.opDid[1] = did; if (phase === 'interrupted' && P.opOf) P.opOf.stopped = true; E.move = null; return; }
      if (P.opOf) { if (phase === 'interrupted') P.opOf.stopped = true; E.move = null; return; }
      emit({kind: 'move', name: P.name, id: P.id, src: P.src, phase, args: P.args, did: (P.pre ? P.pre.replace(/ing\b/g, 'ed').replace('getted', 'got').replace('sitted', 'sat').replace('comed', 'came') : '') + did});
      E.move = null;
    }
    // per-frame: the root pose + move channels
    function moveFrame(o, dt) {
      const P = E.move, out = {};
      let tt = 0;
      if (P) {
        tt = t() - P.t0;
        for (const ph of P.ph) if (!ph.done && tt >= ph.t) { ph.done = true; if (ph.did) emit({kind: 'move_phase', name: P.name, id: P.id, src: P.src, phase: 'end', step: ph.name, did: ph.did}); }
        for (const c of ROOT) out[c] = P.T[c].at(tt);
        // residual: the velocity / acceleration it had when the move started decays over 0.3 s (C2 start)
        const Tb = 0.3;
        if (tt < Tb) for (const c of ROOT) { const v0 = P.vel0[c] || 0, a0 = P.acc0[c] || 0; if (Math.abs(v0) > 1e-4 || Math.abs(a0) > 1e-3) { if (!P.res) P.res = {}; const q = P.res[c] || (P.res[c] = quintic(0, v0 - P.T[c].vel(0), clamp(a0, -60, 60), 0, 0, 0, Tb)); out[c] += q(tt); } }
        const doneAt = P.holdEnd ? Infinity : P.dur;
        if (tt >= doneAt) { finishMove(P, P.interrupted ? 'interrupted' : 'end'); }
      } else {
        for (const c of ROOT) out[c] = E.base[c] + (ANG.has(c) ? Math.round(E.cur[c] - E.base[c]) : 0);
      }
      return {out, P: E.move === P ? P : null, tt};
    }

    // ---------------------------------------------------------------- objects
    const sideWord = s => s === 'l' ? 'right' : 'left';          // screen-left paw = its own right
    const otherSide = s => s === 'l' ? 'r' : 'l';
    function holdingSide(p) { return p && p.held; }
    function heldBy(s) { return E.P.find(p => p.held === s || p.held === 'both'); }
    // a requested paw in the character's OWN frame ("your right paw") -> the internal paw ('l' = its right, screen-left)
    const sideIn = v => v ? (/both/.test(String(v)) ? 'both' : ({left: 'r', right: 'l', own_left: 'r', own_right: 'l', screen_left: 'l', screen_right: 'r', l: 'l', r: 'r'})[String(v).toLowerCase().replace(/[\s-]+/g, '_')] || null) : null;
    const supAway = (q, d = 0) => !!q && d < 8 && (!!q.held || !!q.worn || !!q.bal || !!q.fly || q.where !== 'home' || (!!q.sup && supAway(byId(q.sup.id), d + 1)));
    function propWhere(p) {
      if (p.worn) return 'worn on its head';
      if (p.bal) return 'balanced on its head';   // objects engine: the general attachments
      if (p.anc && !p.held) return 'on ' + anchorWord(p.anc.name);
      if (p.fly) return 'in the air';
      if (p.sup && !p.held && (p.where !== 'home' || supAway(byId(p.sup.id)))) return supName(p);   // (on a thing that is not home itself: not home)
      if (p.F && p.F.furniture) return p.where === 'home' || len3(sub3(p.pos, p.home)) < 0.02 ? 'home' : 'moved';
      if (p.held) return p.held === 'both' ? 'both paws' : sideWord(p.held) + ' paw';
      if (p.flying) return 'air';
      if (p.where === 'home') return 'home';
      if (p.place === 'floor') return 'floor';
      return p.place || 'floor';
    }
    function placeAt(x, z, y) {   // the highest place under (x, z) below y
      let best = null;
      for (const pl of E.places) {
        if (pl.floor) continue;
        const c = pl.center, sx = (pl.size[0] || 0.4) / 2, sz = (pl.size[1] || 0.4) / 2;
        if (Math.abs(x - c[0]) <= sx && Math.abs(z - c[2]) <= sz && pl.top_y <= y + 1e-3 && (!best || pl.top_y > best.top_y)) best = pl;
      }
      return best || E.places.find(p => p.floor);
    }
    function restYFor(p, pl) { return (pl ? pl.top_y : 0) + p.restH; }
    const footR = p => Math.min(p.size, 0.15);   // a resting prop's footprint radius (a big bounding radius over-reserves a desk)
    function spotFree(pt, self) { return !E.P.some(q => q !== self && !q.held && !q.flying && !q.worn && !(q.F && q.F.furniture) && Math.hypot(q.pos[0] - pt[0], q.pos[2] - pt[2]) < (footR(q) + footR(self)) * 1.05); }
    function reachable(s, P) { let r = invW(s, P, E.paw[s].hPrev, 14); if (r.err >= 0.035) { const r0 = invW(s, P, [0, 0, 0], 14); if (r0.err < r.err) r = r0; }   // a fresh start when the last hand pose sits in a local minimum
       if (r.err >= 0.035 && E.body.legs) { const e = reachE(s, P); if (e < 0.035) return {ok: true, err: e, h: r.h, p: P}; } return {ok: r.err < 0.035, ...r}; }
    function reachE(s, P) { if (!E.body.legs) return invW(s, P, E.paw[s].hPrev, 14).err; const r = rootNow(); return bendSolve({x: r.x, y: E.sitOn ? r.y - (E.bend ? E.bend.y : 0) : 0, z: r.z, yaw: r.yaw}, P, [s], !!E.sitOn).e; }
    function rootNow() { return E.rootKfNow || E.rootOut || {x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0}; }
    function toWorldR(r, pl) { const R = rotm(r.yaw, r.pitch || 0, r.roll || 0), C = E.body.C; return add3(add3([r.x, r.y || 0, r.z], C), mv(R, sub3(pl, C))); }
    // the waist bend (legs): body_pitch tips the upper body (arms, head) about the waist, 0.28 rad per unit (muse.glsl)
    const WAIST = () => E.body.waist || [0, 0.3, 0];
    function bendUp(u, bp) { if (!bp) return u; const W = WAIST(), a = 0.28 * bp, c = Math.cos(a), sn = Math.sin(a), y = u[1] - W[1], z = u[2] - W[2]; return [u[0], W[1] + y * c - z * sn, W[2] + y * sn + z * c]; }
    function unbendUp(q, bp) { return bendUp(q, -bp); }
    function toWorld(pl) { return toWorldR(rootNow(), bendUp(pl, E.bend ? E.bend.p : 0)); }
    function toLocal(pw) { const r = rootNow(), R = rotm(r.yaw, r.pitch, r.roll), C = E.body.C, d = sub3(pw, add3([r.x, r.y, r.z], C)); return unbendUp(add3(C, [R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2], R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2], R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]]), E.bend ? E.bend.p : 0); }
    // a squat + forward lean for low targets (the floor): how much for a target at world height y
    function bendFor(y, seated) { const k = clamp((0.2 - y) / 0.17, 0, 1); return {y: seated ? 0 : -0.15 * k, p: k}; }
    // the least bend that reaches Pw from root pose r (low: squat + lean; far: lean forward up to 16°)
    function bendSolve(r, Pw, sides, seated) {
      const b0 = bendFor(Pw[1], seated); let best = null;
      for (const p of [b0.p, Math.max(b0.p, 0.5), 1]) {
        const b = {y: b0.y, p}, lo = toLocalR(r, Pw, b), e = Math.min(...sides.map(sd => E.reach[sd].inv(lo, [0, 0, 0], 14).err));
        if (!best || e < best.e - 1e-4) best = {b, e};
        if (e < 0.02) break;
      }
      return best;
    }
    function bendStep(Pw) {
      return {name: 'bend', dur: 0.5, start() { const r = rootNow(); this.b = Pw ? bendSolve({x: r.x, y: E.base.y || 0, z: r.z, yaw: r.yaw}, Pw, ['l', 'r'], !!E.sitOn).b : {y: 0, p: 0}; E.bendT = this.b; this.dur = Math.abs(this.b.p - (E.bend.p || 0)) < 0.02 ? 0.001 : 0.55; }, update(tt) { return tt >= this.dur; }};
    }
    const fwdW = (s, h) => toWorld(E.reach[s].fwd(h));
    function invW(s, P, h0, it) { const r = E.reach[s].inv(toLocal(P), h0, it); r.p = toWorld(r.p); return r; }
    const restHand = s => E.restH ? E.restH[s] : [0, 0, 0];
    // ---- grips: the paw closes on a handle (prop-local H); the renderer draws a held prop with ITS grip anchor
    // (grip_point_local) in the palm, so a hold by another handle carries the held offset that puts that handle there
    const gaOf = p => v3(p.def.grip_point_local) || [0, 0, 0];
    const gripLocal = p => p.grip || gaOf(p);
    const heldYawL = p => p.homeRot[1] + p.hrot[1];                 // the held prop's yaw in the body frame
    const restYawW = p => p.homeRot[1] + p.hrot[1] + rootNow().yaw;  // its world yaw if let go now
    function gripCands(p) {
      const H = p.def.handlesFull; if (!H) return [gaOf(p)];
      const c = H.filter(h => Array.isArray(h.local) && !/lid/.test(h.id || '')).map(h => h.local.slice());
      return c.length ? c : [gaOf(p)];
    }
    function hoffFor(p, g) { return rotYv(sub3(gaOf(p), g), heldYawL(p)); }
    function centreFromPaw(p, P0, yawW) { return sub3(P0, rotYv(gripLocal(p), yawW ?? restYawW(p))); }
    function pawAtRest(p, rest, yawW) { return add3(rest, rotYv(gripLocal(p), yawW ?? restYawW(p))); }
    // the handle a paw takes from where the body is now: reachable first, then the renderer's own anchor, then a
    // handle on the paw's inner side (an off-centre prop then hangs outward, clear of the body)
    function gripPick(p, s) {
      const C = gripCands(p), ga = gaOf(p); let best = null;
      for (const g of C) {
        const P = fW(p, g), e = reachE(s, P), lx = toLocal(P)[0] - toLocal(p.pos)[0];
        const cost = (e > 0.03 ? 10 + e : e) + (len3(sub3(g, ga)) < 1e-6 ? 0 : 0.02) + ((s === 'l' ? -1 : 1) * lx > 0 ? 0.01 : 0);
        if (!best || cost < best.cost) best = {g, cost, e};
      }
      return best;
    }
    // resolve a descriptor against the props + state
    const toks = s => String(s || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(Boolean).map(w => w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w);
    function propNames(p) {
      const d = p.def, set = new Set([...toks(p.id), ...toks(p.name), ...toks(d.category)]);
      for (const a of d.aliases || []) toks(a).forEach(w => set.add(w));
      return set;
    }
    function resolveObject(desc, verb, opt = {}) {
      if (!E.P.length) return {err: 'there are no objects here'};
      let d = desc;
      if (typeof d === 'string') d = {name: d};
      d = d || {};
      const ref = String(d.ref || '').toLowerCase();
      let C = E.P.slice();
      const filters = [];
      if (d.id) { const x = C.filter(p => p.id === String(d.id) || p.id.toLowerCase() === String(d.id).toLowerCase()); if (!x.length) return {err: `there is no ${d.id} here`}; C = x; }
      const nm = toks(d.name || d.category || '').filter(w => !/^(the|a|an|one|thing|object|it|that|this|other|item|stuff)$/.test(w));
      if (nm.length) {
        const ex = C.filter(p => { const N = propNames(p); return nm.every(w => N.has(w)); });   // an exact word wins ("sunglasses" is not the "sun hat")
        const x = ex.length ? ex : C.filter(p => { const N = propNames(p); return nm.every(w => N.has(w) || [...N].some(n => n.startsWith(w) || w.startsWith(n) && n.length > 2)); });
        if (!x.length) { const y = C.filter(p => { const N = propNames(p); return nm.some(w => N.has(w)); }); if (!y.length) return {err: `there is no ${d.name || d.category} here`}; C = y; } else C = x;
        filters.push(d.name || d.category);
      }
      if (d.category && d.name) { const x = C.filter(p => toks(p.def.category).some(w => toks(d.category).includes(w))); if (x.length) C = x; }
      if (d.color) {
        const c = String(d.color).toLowerCase();
        const x = C.filter(p => [p.def.color, ...(p.def.colors || [])].filter(Boolean).some(k => String(k).toLowerCase() === c || String(k).toLowerCase().includes(c)));
        if (!x.length) return {err: `there is no ${c} ${nm.length ? (d.name || d.category) : 'one'} here`};
        C = x; filters.push(c);
      }
      if (/^(it|that|this|same|them|they|those|these|em)$/.test(ref)) {
        if (!filters.length) {
          // "pick up the ball and put it down" — the step before it in the same request (still queued or running,
          // so nothing is in a paw yet) is what "it" means
          const pj = pendingProp(); if (pj) return {p: pj};
          const held = E.P.filter(p => p.held);
          if (held.length === 1) return {p: held[0]};
          if (E.lastRef && E.P.includes(E.lastRef)) return {p: E.lastRef};
          if (held.length > 1) return {err: `which one — ${held.map(p => 'the ' + p.name).join(' or ')}?`, ask: true};
          if (E.P.length === 1) return {p: E.P[0]};
          return {err: 'which one do you mean?', ask: true};
        }
      }
      if (ref === 'all' && !filters.length) { const ps = pendingProps(); if (ps.length === 1) return {p: ps[0]}; }   // "… and put them back": the one thing the request is busy with
      if (ref === 'other' || ref === 'another' || ref === 'the_other') {
        const lr = E.lastRef || E.P.find(p => p.held);
        const x = C.filter(p => p !== lr && (!opt.notHeld || !p.held));
        if (!x.length) return {err: 'there is no other one here'};
        C = x;
      }
      if (d.side) {
        const sg = sideSign(d.side, 0);   // own left = screen right = +x
        if (sg) { C = C.slice().sort((a, b) => sg * (b.pos[0] - a.pos[0])); C = [C[0]]; }
      }
      if (d.ordinal !== undefined && d.ordinal !== null) {
        const n = num(d.ordinal, 1), order = C.slice().sort((a, b) => a.pos[0] - b.pos[0]);   // left → right as he sees it
        const i = n < 0 ? order.length + n : n - 1;
        if (i < 0 || i >= order.length) return {err: `there are only ${order.length} of those`};
        C = [order[i]];
      }
      if (C.length > 1 && opt.preferFree) { const x = C.filter(p => !p.held); if (x.length >= 1) C = x; }
      if (C.length > 1 && opt.preferHeld) { const x = C.filter(p => p.held); if (x.length >= 1) C = x; }
      if (C.length > 1 && (ref === 'any' || ref === 'a' || ref === 'all' && opt.single)) C = [C.slice().sort((a, b) => Math.abs(a.pos[0]) - Math.abs(b.pos[0]))[0]];
      if (C.length > 1) return {err: `which one — ${C.map(p => 'the ' + p.name).join(' or ')}?`, ask: true, cands: C};
      return {p: C[0]};
    }
    function resolvePlace(name, p) {
      if (!name) return null;
      const w = toks(name).filter(x => !/^(the|a|on|onto|in|into|at|to|top|of)$/.test(x));
      if (!w.length) return null;
      for (const pl of E.places) {
        const N = new Set([...toks(pl.id), ...toks(pl.name), ...pl.aliases.flatMap(toks)]);
        if (w.some(x => N.has(x))) return pl;
      }
      return {err: `there is no ${name} here`};
    }
    // a free rest spot on a place for prop p, nearest to the paw's side; null if all taken
    function freeSpot(pl, p, s) {
      const y = pl.top_y + p.restH, near = s === 'l' ? -1 : 1;
      const cand = pl.spots.map(q => [q[0], y, q[2]]).filter(q => spotFree(q, p));
      if (!cand.length) return null;
      cand.sort((a, b) => (Math.abs(a[0] - near * 0.3) + Math.abs(a[2] - 0.35)) - (Math.abs(b[0] - near * 0.3) + Math.abs(b[2] - 0.35)));
      return cand[0];
    }
    // the carry pose: in front of the tummy on its own side (both paws: the middle)
    function carryPoint(s, p) {
      const B0 = E.body, sz = p ? p.size : 0.06, two = p && p.held === 'both', g = p && !two ? gripLocal(p) : [0, 0, 0], off = Math.hypot(g[0], g[2]);
      const x = two ? (s === 'l' ? -1 : 1) * Math.min(sz * 0.95, 0.4) : (s === 'l' ? -1 : 1) * Math.max(0.14, 0.32 * B0.s[0] + 0.02 + (off > 0.15 ? 0.14 : 0));
      return [x, B0.lo[1] + 0.4 * (B0.hi[1] - B0.lo[1]) + (off > 0.15 ? 0.06 : 0), B0.front + Math.min(sz, off > 0.15 ? 0.02 : 0.12) + 0.03];
    }
    function headPoint() { return E.body.head; }
    function graspValues(p, s) {   // hand shape while holding p
      const g = (p.def.grip || 'power');
      return g === 'pinch' ? {open: 0.35, turn: 0.1, thumb: 0.3, point: 0} : g === 'two_hand' ? {open: 0.7, turn: 0, thumb: 0, point: 0} : {open: 0.12, turn: 0.15, thumb: 0, point: 0};
    }
    // the prop's free offsets (world, from home) for the current state
    function freeOff(p) { return sub3(p.pos, p.home); }

    // --- object steps: {name, atomic, start(), update(tt) → done:boolean, end()}; tt = seconds since start
    function pawPath(s, Q1, dur, opt = {}) {   // a step moving paw s (grasp point, world) from where it is to Q1
      return {name: opt.name || 'move paw', atomic: !!opt.atomic, dur, s,
        start() {
          const pw = E.paw[s];
          this.h0 = pw.hPrev.slice(); const r = invW(s, Q1, pw.hPrev, 14); this.h1 = r.h; this.Q1 = r.err > 0.02 ? r.p : Q1;
          this.world = opt.arc === 0; this.Q0 = fwdW(s, this.h0);
          const dh = len3(sub3(this.h1, this.h0));
          this.dur = Math.max(dur, this.world ? 0 : 0.3 + 0.38 * dh);
          this.arc = this.world ? 0 : Math.min(0.25, (opt.arc ?? Math.min(0.14, 0.25 * len3(sub3(this.Q1, this.Q0)))) / 0.35);
          this.open0 = pw.open; this.turn0 = pw.turn; this.thumb0 = pw.thumb;
          pw.ctrlT = 1;
        },
        update(tt) {
          const pw = E.paw[s], u = clamp(tt / this.dur, 0, 1), m = mj(u);
          if (this.world) { pw.tgt = lerp3(this.Q0, this.Q1, m); pw.hd = null; }
          else { const R2 = E.reach[s], h = lerp3(this.h0, this.h1, m); h[1] = Math.min(R2.hi[1], h[1] + this.arc * Math.sin(Math.PI * u)); pw.hd = h; pw.tgt = fwdW(s, h); pw.hdFor = pw.tgt; }
          if (opt.open !== undefined) pw.open = lerp(this.open0, opt.open, mj(clamp(u / 0.7, 0, 1)));
          if (opt.turn !== undefined) pw.turn = lerp(this.turn0, opt.turn, m);
          if (opt.thumb !== undefined) pw.thumb = lerp(this.thumb0, opt.thumb, m);
          if (opt.look) E.look.P = opt.look === 'paw' ? pw.tgt : opt.look;
          if (opt.lookUser) E.look.P = null;
          if (opt.each) opt.each(u, tt);
          return u >= 1;
        }};
    }
    function stepFn(name, dur, fn, opt = {}) { return {name, dur, atomic: !!opt.atomic, start: opt.start || (() => {}), update(tt) { const u = clamp(tt / dur, 0, 1); fn(u, tt); return u >= 1; }}; }
    function lookStep(Pt, on) { return stepFn('look', 0.001, () => { if (on) { E.look.wT = 1; E.look.P = Pt; } else { E.look.wT = 0; } }); }
    // grasp: the paw is at the object; attach eases 0 → 1 (the free offsets ramp to the held ones: no slide)
    function graspStep(s, p, two, grip) {
      return {name: 'grasp', atomic: true, dur: 0.28, start() {
        const pw = E.paw[s]; this.o0 = pw.open; this.g = graspValues(p, s);
        // where it was resting (c2): "put it back where it was / where you got it" goes back here
        if (!p.held && !p.worn && !p.flying) p.lastRest = {pos: p.pos.slice(), rot: p.rot.slice(), place: p.where === 'home' ? p.homePlace : (p.place || 'floor'), home: p.where === 'home'};
        this.fo = freeOff(p); this.fr = p.rot.slice(); p.graspFrom = {fo: this.fo, fr: this.fr};
        if (p.fly) { const F = p.fly; if (F.settle) setP(p, F.settle.to); p.fly = null; p.flying = false; }
        p.sup = null; p.anc = null; p.bal = null;   // whatever it rested on / rode on, it is in a paw now
        p.held = two ? 'both' : s; p.free = false; p.where = 'held';
        p.hrot = p.rot.slice(); p.hrot[1] = wrapH(p.rot[1] - rootNow().yaw);
        const hs = p.def.handles; p.grip = two ? (hs && hs.length >= 2 ? lerp3(v3(hs[0]), v3(hs[1]), 0.5) : [0, 0, 0]) : (grip || gaOf(p)).slice();
        p.hoff = hoffFor(p, p.grip);
        this.fr = p.rot.slice();
      }, update(tt) {
        const u = clamp(tt / this.dur, 0, 1), w = mj(u), pw = E.paw[s];
        pw.open = lerp(this.o0, this.g.open, w); pw.turn = lerp(pw.turn, this.g.turn, w * 0.3);
        if (two) { p.att.l = 0.5 * w; p.att.r = 0.5 * w; E.paw[otherSide(s)].open = lerp(E.paw[otherSide(s)].open, this.g.open, w); }
        else p.att[s] = w;
        return u >= 1;
      }, end() {
        emit({kind: 'object_event', name: 'picked_up', phase: 'end', src: 'auto', args: {object: p.id, side: two ? 'both' : sideWord(s)}, did: `picked up the ${p.name} with ${two ? 'both paws' : 'its ' + sideWord(s) + ' paw'}`});
        p.place = null;
      }};
    }
    // release at a rest spot: attach 1 → 0 while the offsets ramp from the held frame to the world offsets of the spot
    function releaseStep(s, p, rest, how = 'put') {
      return {name: 'release', atomic: true, dur: 0.28, start() {
        this.from = {l: p.att.l, r: p.att.r};
        p.pos = rest.slice(); p.rot = p.hrot.slice(); p.rot[1] = wrapH(p.hrot[1] + rootNow().yaw);
      }, update(tt) {
        const u = clamp(tt / this.dur, 0, 1), w = 1 - mj(u);
        p.att.l = this.from.l * w; p.att.r = this.from.r * w;
        for (const sd of ['l', 'r']) if (this.from[sd] > 0) E.paw[sd].open = lerp(E.paw[sd].open, 0.75, mj(u) * 0.5);
        return u >= 1;
      }, end() {
        p.att.l = 0; p.att.r = 0; p.held = null; p.free = true; p.grip = null; p.hoff = [0, 0, 0];
        const pl = placeAt(rest[0], rest[2], rest[1]);
        const home = len3(sub3(rest, p.home)) < 0.01;
        if (home && Math.abs(wrap(p.rot[1])) < 2e-3 && Math.abs(p.rot[0]) + Math.abs(p.rot[2]) < 2e-3) p.rot = [0, 0, 0];   // back in its home pose exactly (the headphones' band on the saddle)
        p.where = home ? 'home' : 'placed'; p.place = home ? p.homePlace : pl ? pl.id : 'floor';
        if (E.kfh) settleSupport(p);   // resting on another prop's top / in it: kept as a relation
        // VISIBILITY: a thing he asked to put on / in something: the camera eases in on the result a moment
        // (live v4: the mug sat on the book, small behind the headphones; "the end state is not clearly framed")
        if (how !== 'drop' && !home && p.sup) { const Cc = E.chain, vb = Cc ? String((Cc.args && Cc.args.verb) || (Cc.op && Cc.op.cmd && Cc.op.cmd.verb) || Cc.verb || '') : '';
          if (/^(place|put|put_in|set_on|set|stack|move|drop_in|insert|put_on)$/.test(vb)) E.resultView = {id: p.id, until: t() + 2.6}; }
        if (how !== 'drop') emit({kind: 'object_event', name: p.sup ? (p.sup.inside ? 'put_in' : 'stacked') : 'put_down', phase: 'end', src: 'auto', args: {object: p.id, where: home ? 'home' : p.sup ? p.sup.id : p.place, inside: !!(p.sup && p.sup.inside)}, did: home ? `put the ${p.name} back where it belongs` : p.sup ? `put the ${p.name} ${supName(p)}` : `put the ${p.name} down on ${placeName(p.place)}`});
      }};
    }
    const placeName = id => { const pl = E.places.find(x => x.id === id); return pl ? pl.name : 'the floor'; };
    function retractStep(s, toCarry) {   // the paw back to its carry pose (holding) or rest, then the base layer takes it
      return {name: 'retract', dur: 0.5, start() {
        const pw = E.paw[s], p = heldBy(s);
        this.h0 = pw.hPrev.slice(); this.h1 = p ? carryHand(s, p) : restHand(s);
        this.dur = clamp(0.35 + 0.4 * len3(sub3(this.h1, this.h0)), 0.45, 1.1);
        this.o0 = pw.open; this.t0 = pw.turn; this.g = p ? graspValues(p, s) : {open: 0.2, turn: 0, thumb: 0};
      }, update(tt) {
        const pw = E.paw[s], u = clamp(tt / this.dur, 0, 1), m = mj(u);
        const h = lerp3(this.h0, this.h1, m); h[1] += 0.06 * Math.sin(Math.PI * u) * (this.h1[1] > this.h0[1] ? 1 : 0);
        pw.hd = h; pw.tgt = fwdW(s, h); pw.hdFor = pw.tgt; pw.open = lerp(this.o0, this.g.open, m); pw.turn = lerp(this.t0, this.g.turn, m);
        if (u >= 1) { pw.ctrlT = 0; pw.hd = null; }
        return u >= 1;
      }};
    }
    function carryHand(s, p) {
      const r = E.reach[s].inv(carryPoint(s, p), E.paw[s].hPrev, 14);   // the carry pose is body-relative (rest frame)
      return r.h;
    }
    // a hold: keep the paws where the previous step left them for `dur`, with an optional procedural motion
    function dwellStep(name, dur, fn) { return stepFn(name, dur, fn || (() => {})); }

    // --- verb planners → list of steps (built lazily: each verb's steps read the state when they start)
    function chainPickUp(p, s, two) {
      const out = [];
      if (two) {
        const o = otherSide(s), off = p.size * 0.95, hs = p.def.handles && p.def.handles.length >= 2 ? p.def.handles.slice(0, 2).map(h => fW(p, h)) : null;
        let Ql = add3(p.pos, [-off, 0, 0]), Qr = add3(p.pos, [off, 0, 0]);
        if (hs) { const sw = toLocal(hs[0])[0] > toLocal(hs[1])[0]; Ql = sw ? hs[1] : hs[0]; Qr = sw ? hs[0] : hs[1]; }
        const Al = add3(Ql, [-0.05, 0.1, 0.06]), Ar = add3(Qr, [0.05, 0.1, 0.06]);
        out.push(lookStep(p.pos, true));
        if (E.body.legs) out.push(bendStep(lerp3(Ql, Qr, 0.5)));
        out.push(par([pawPath('l', Al, 0.65, {open: 0.85, turn: 0}), pawPath('r', Ar, 0.65, {open: 0.85, turn: 0})]));
        out.push(par([pawPath('l', Ql, 0.35, {arc: 0.0}), pawPath('r', Qr, 0.35, {arc: 0.0})]));
        out.push(graspStep('l', p, true));
        out.push(par([pawPath('l', add3(Ql, [0, 0.12, 0.02]), 0.35, {arc: 0}), pawPath('r', add3(Qr, [0, 0.12, 0.02]), 0.35, {arc: 0})]));
        if (E.body.legs) out.push(bendStep(null));
        out.push(lookStep(null, false));
        out.push(par([retractStep('l', true), retractStep('r', true)]));
        void o;
        return out;
      }
      const gp = gripPick(p, s), G = fW(p, gp.g), A1 = add3(G, [0, 0.1 + Math.min(p.size, 0.1), 0.05]);
      out.push(lookStep(p.pos, true));
      if (E.body.legs) out.push(bendStep(G));
      out.push(pawPath(s, A1, clamp(0.45 + 0.8 * len3(sub3(A1, fwdW(s, E.paw[s].hPrev))), 0.5, 1.1), {open: 0.85, turn: -0.35, thumb: 0.2, look: p.pos}));
      out.push(pawPath(s, G, 0.32, {arc: 0.0, arcZ: 0, open: 0.9, look: p.pos}));
      out.push(graspStep(s, p, false, gp.g));
      out.push(pawPath(s, add3(G, [0, 0.13, 0.03]), 0.3, {arc: 0.0, look: 'paw'}));
      if (E.body.legs) out.push(bendStep(null));
      out.push(lookStep(null, false));
      out.push(retractStep(s, true));
      return out;
    }
    function par(steps) {   // run steps together; done when all are
      return {name: steps.map(x => x.name).join('+'), atomic: steps.some(x => x.atomic), dur: Math.max(...steps.map(x => x.dur || 0)),
        start() { for (const x of steps) x.start && x.start(); this.d = steps.map(() => false); },
        update(tt) { let all = true; steps.forEach((x, i) => { if (!this.d[i]) { this.d[i] = x.update(tt); if (this.d[i] && x.end) x.end(); } all = all && this.d[i]; }); return all; },
        end() {}};
    }
    function chainPutAt(p, s, rest, how, opt = {}) {
      const out = [], two = p.held === 'both';
      const yW = opt.yawW;
      const reachOk = s && (two ? reachable('l', rest).ok : reachable(s, pawAtRest(p, rest, yW)).ok);
      let rel = rest.slice(), drop = false;
      if (!two && !reachOk) {   // not reachable (the floor): lower it as far as it goes above the spot, let go, it drops
        if (yW !== undefined && rest[1] > 0.05) return null;   // an exact pose on a surface (its stand): never dropped onto it
        for (let y = rest[1]; y < rest[1] + 0.8; y += 0.03) { const q = [rest[0], y, rest[2]]; if (reachable(s, pawAtRest(p, q, yW)).ok) { rel = q; drop = true; break; } }
        if (!drop) return null;
      }
      out.push(lookStep(rest, true));
      if (E.body.legs) out.push(bendStep(two ? rel : pawAtRest(p, rel, yW)));   // bend so the PAW gets there (a handle off the centre)
      if (two) {
        const off = p.size * 0.95;
        out.push(par([pawPath('l', add3(rel, [-off, 0.12, 0.04]), 0.6), pawPath('r', add3(rel, [off, 0.12, 0.04]), 0.6)]));
        out.push(par([pawPath('l', add3(rel, [-off, 0, 0]), 0.35, {arc: 0}), pawPath('r', add3(rel, [off, 0, 0]), 0.35, {arc: 0})]));
        out.push(releaseStep('l', p, rest));
        out.push(par([pawPath('l', add3(rel, [-off - 0.04, 0.1, 0.04]), 0.3, {arc: 0, open: 0.4}), pawPath('r', add3(rel, [off + 0.04, 0.1, 0.04]), 0.3, {arc: 0, open: 0.4})]));
        if (E.body.legs) out.push(bendStep(null));
        out.push(lookStep(null, false));
        out.push(par([retractStep('l'), retractStep('r')]));
        return out;
      }
      const relP = pawAtRest(p, rel, yW);
      out.push(pawPath(s, add3(relP, [0, 0.12 + Math.min(p.size, 0.1), 0.04]), 0.6, {turn: -0.3, look: rest}));
      out.push(pawPath(s, relP, 0.32, {arc: 0, look: rest}));
      if (drop) out.push(dropStep(s, p, how === 'drop' ? 'drop' : 'let go'));
      else out.push(releaseStep(s, p, rest, how));
      out.push(pawPath(s, add3(relP, [0, 0.1, 0.05]), 0.28, {arc: 0, open: 0.45}));
      if (E.body.legs) out.push(bendStep(null));
      out.push(lookStep(null, false));
      out.push(retractStep(s));
      return out;
    }
    // drop: let go where the paw is; it falls under gravity onto what is below, bounces, settles (atomic)
    function dropStep(s, p, how = 'drop') {
      const G = 12;
      return {name: 'drop', atomic: true, dur: 3, start() {
        const pw = E.paw[s], P0 = centreFromPaw(p, fwdW(s, pw.hPrev));
        this.from = {l: p.att.l, r: p.att.r}; this.T0 = fwdW(s, pw.hPrev);
        p.pos = P0.slice(); p.rot = p.hrot.slice(); p.rot[1] = wrapH(p.hrot[1] + rootNow().yaw); p.flying = true;
        let pl = placeAt(P0[0], P0[2], P0[1]), y1 = restYFor(p, pl);
        if (E.kfh) { const W0 = P(p, P0), sp = supportAt(W0[0], W0[2], W0[1] - p.hH + 0.004, p); y1 = toR(p, [W0[0], sp.top + p.hH, W0[2]])[1]; pl = sp.kind === 'place' ? E.places.find(x => x.id === sp.id) : E.places.find(x => x.id === supPlace(byId(sp.id))); }   // onto whatever is under it
        this.y0 = P0[1]; this.y1 = Math.min(y1, P0[1]); this.P0 = P0; this.pl = pl;
        const e = p.def.mass === 'heavy' ? 0.1 : p.def.mass === 'medium' ? 0.25 : 0.45;
        // bounce schedule: fall, then arcs of decreasing height
        const segs = []; let tt = Math.sqrt(2 * Math.max(0, this.y0 - this.y1) / G), v = G * tt;
        segs.push({t0: 0, t1: tt, kind: 'fall'});
        let vv = v * e, tc = tt;
        for (let i = 0; i < 3 && vv > 0.35; i++) { const d = 2 * vv / G; segs.push({t0: tc, t1: tc + d, v: vv}); tc += d; vv *= e; }
        this.segs = segs; this.dur = tc + 0.25; this.spin = (rnd() - 0.5) * 0.3; this.dz = 0.05 * (rnd() - 0.2);
      }, update(tt) {
        // the paw lets go over 0.22 s (min-jerk; an attach weight never jumps); while it does, the paw drops with the prop
        const u = clamp(tt / 0.22, 0, 1), w = 1 - mj(u);
        p.att.l = this.from.l * w; p.att.r = this.from.r * w;
        E.paw[s].open = lerp(E.paw[s].open, 0.8, mj(u) * 0.4);
        let y = this.y1;
        for (const g of this.segs) if (tt >= g.t0 && tt < g.t1) { const q = tt - g.t0; y = g.kind === 'fall' ? this.y0 - 0.5 * G * q * q : this.y1 + g.v * q - 0.5 * G * q * q; }
        const k = clamp(tt / Math.max(0.2, this.segs[this.segs.length - 1].t1), 0, 1), roll = 1 - (1 - k) * (1 - k);
        p.pos = [this.P0[0], Math.max(this.y1, y), this.P0[2] + this.dz * roll];
        p.rot = [p.hrot[0] + this.spin * roll, p.hrot[1], p.hrot[2]];
        if (w > 0) E.paw[s].tgt = lerp3(this.T0, add3(p.pos, sub3(this.T0, this.P0)), w);
        if (tt > this.segs[0].t1 && !this.hit) { this.hit = true; }
        return tt >= this.dur;
      }, end() {
        p.att.l = 0; p.att.r = 0; p.held = null; p.free = true; p.flying = false;
        p.pos = [p.pos[0], this.y1, p.pos[2]];
        p.where = 'placed'; p.place = this.pl ? this.pl.id : 'floor';
        if (E.kfh) settleSupport(p);
        emit({kind: 'object_event', name: how === 'drop' ? 'dropped' : 'put_down', phase: 'end', src: 'auto', args: {object: p.id, where: p.place}, did: how === 'drop' ? `dropped the ${p.name}; it landed ${p.sup ? supName(p) : 'on ' + placeName(p.place)}` : `let the ${p.name} down ${p.sup ? supName(p).replace(/^on /, 'onto ') : 'onto ' + placeName(p.place)}`});
      }};
    }
    // toss: wind-up, flick, release at the top with the flick's velocity, a ballistic arc, the catching paw meets it
    // under the arc (inverse reach every frame), velocity-matched catch, absorb; atomic from release to catch
    function tossSteps(p, s, toS, h) {
      const G = 12, out = [];
      const Sx = (s === 'l' ? -1 : 1) * 0.16, B0 = E.body;
      const S0 = [Sx, B0.lo[1] + 0.45 * (B0.hi[1] - B0.lo[1]), B0.front + p.size + 0.08];
      out.push(pawPath(s, S0, 0.4, {look: 'paw'}));
      const toss = {name: 'toss', atomic: true, dur: 4, start() {
        const pw = E.paw[s];
        this.Q = fwdW(s, pw.hPrev);
        const Wd = add3(this.Q, [0, -0.07, -0.01]), Rl = add3(this.Q, [0, 0.07, 0.01]);
        const tw = 0.2, tf = 0.11;
        const catchS = toS, Sc = [(catchS === 'l' ? -1 : 1) * 0.16, this.Q[1] - 0.02, this.Q[2]];
        if (!reachable(catchS, Sc).ok) Sc[1] = this.Q[1];
        const vy = Math.sqrt(2 * G * h), Tf = (vy + Math.sqrt(vy * vy + 2 * G * (Rl[1] - Sc[1]))) / G;
        this.vx = (Sc[0] - Rl[0]) / Tf; this.vz = (Sc[2] - Rl[2]) / Tf; this.vy = vy;
        this.W = Wd; this.R = Rl; this.tw = tw; this.tf = tf; this.Tf = Tf; this.Sc = Sc; this.cs = catchS;
        this.flip = p.def.throwable === false ? 0 : (rnd() < 0.5 ? 1 : 0.5);
        this.relDone = false; this.caught = false; this.r0 = p.hrot.slice();
        // the flick: wind-up dip, then up to the release point arriving with the arc's velocity (quintic)
        const qy = quintic(Wd[1], 0, 0, Rl[1], vy * 0.55, 0, tf);
        this.flickY = qy; this.dur = tw + tf + Tf + 0.45;
        E.paw[catchS].ctrlT = 1;
        E.look.wT = 1;
      }, update(tt) {
        const pw = E.paw[s], tw = this.tw, tf = this.tf, Tf = this.Tf;
        if (tt < tw) { pw.tgt = lerp3(this.Q, this.W, mj(tt / tw)); E.look.P = pw.tgt; return false; }
        if (tt < tw + tf) { const q = tt - tw; pw.tgt = [lerp(this.W[0], this.R[0], mj(q / tf)), this.flickY(q), lerp(this.W[2], this.R[2], mj(q / tf))]; E.look.P = pw.tgt; return false; }
        const q = tt - tw - tf;
        const pos = [this.R[0] + this.vx * q, this.R[1] + this.vy * q - 0.5 * G * q * q, this.R[2] + this.vz * q];
        if (!this.relDone) {   // release: attach eases out over 3 frames from the flick's top
          this.relDone = true; p.flying = true; p.pos = this.R.slice();
          this.att0 = {l: p.att.l, r: p.att.r};
          emit({kind: 'object_event', name: 'tossed', phase: 'end', src: 'auto', args: {object: p.id, side: sideWord(s)}, did: `tossed the ${p.name} up`});
        }
        if (q < Tf) {
          const wr = 1 - mj(clamp(q / Math.min(0.21, Tf * 0.8), 0, 1));   // the attach eases out over ≥ 0.2 s (no rig-column jump)
          p.att.l = this.att0.l * wr; p.att.r = this.att0.r * wr;
          p.pos = pos; p.rot = [this.r0[0] + this.flip * mj(q / Tf), this.r0[1], this.r0[2]];
          if (wr <= 0) { p.held = null; p.where = 'air'; }
          // the throwing paw follows through and drops back; the catching paw goes under the falling prop
          if (this.cs !== s) { const back = mj(clamp(q / 0.4, 0, 1)); pw.tgt = lerp3(this.R, fwdW(s, [0, 0, 0]), back); }
          const cp = E.paw[this.cs], lead = Tf - q;
          if (this.cs === s || q > 0.1) {
            // meet the prop: ahead of contact, ease toward the arc's catch point; in the last 0.12 s, track the prop itself
            const k = clamp((0.12 - lead) / 0.12, 0, 1);
            const wait = add3(this.Sc, [0, 0.0, 0]);
            const from = this.cs === s ? (q < 0.25 ? lerp3(this.R, wait, mj(q / 0.25)) : wait) : lerp3(fwdW(this.cs, cp.hPrev), wait, clamp(q / 0.3, 0, 1));
            cp.tgt = lerp3(from, pos, mj(k)); cp.open = lerp(cp.open, 0.85, 0.2);
          }
          if (wr > 0) pw.tgt = lerp3(pw.tgt, pos, wr);   // still attached: the throwing paw stays with the prop as it lets go
          E.look.P = pos;
          return false;
        }
        // catch: from contact, attach eases in while paw and prop move together and decelerate (absorb 0.1 H)
        const qc = q - Tf;
        if (!this.caught) {
          this.caught = true; p.flying = false; p.held = this.cs; p.where = 'held';
          this.cP = pos; this.cV = [this.vx, this.vy - G * Tf, this.vz];
          this.ax = [0, 1, 2].map(i => quintic(this.cP[i], this.cV[i], 0, this.cP[i] + this.cV[i] * 0.1 + (i === 1 ? 0 : 0), 0, 0, 0.32));
          this.ay = quintic(this.cP[1], this.cV[1], 0, this.cP[1] - 0.09, 0, 0, 0.32);
          emit({kind: 'object_event', name: 'caught', phase: 'end', src: 'auto', args: {object: p.id, side: sideWord(this.cs)}, did: `caught the ${p.name} in its ${sideWord(this.cs)} paw`});
        }
        const cp = E.paw[this.cs], w = mj(clamp(qc / 0.21, 0, 1));   // paw and prop move together from contact; the attach eases in over 0.21 s
        const P1 = [this.ax[0](Math.min(qc, 0.32)), this.ay(Math.min(qc, 0.32)), this.ax[2](Math.min(qc, 0.32))];
        cp.tgt = P1; p.pos = P1; cp.open = lerp(cp.open, graspValues(p, this.cs).open, w);
        if (this.attO === undefined) this.attO = p.att[otherSide(this.cs)] || 0;
        p.att[this.cs] = w; p.att[otherSide(this.cs)] = this.attO * (1 - w);
        p.hrot = p.rot.slice();
        if (qc >= 0.45) return true;
        return false;
      }, end() { p.att[this.cs] = 1; p.held = this.cs; p.free = false; p.flying = false; p.where = 'held'; p.hoff = [0, 0, 0]; }};
      out.push(toss);
      return out;
    }
    // poses with the held prop: show (to the camera, its front facing the camera), inspect (by the face, eyes on it),
    // offer (held out to him), hug (to the chest), shake, turn / flip (rotate in the grasp frame)
    function frontFacing(p) { const f = v3(p.def.facing) || [0, 0, 1]; return -Math.atan2(f[0], f[2]) / TAU; }
    function showPoint(s, p, kind) {
      const B0 = E.body, sg = s === 'l' ? -1 : 1, Hh = B0.hi[1] - B0.lo[1];
      // a thing held up to show is raised beside the face toward the camera (the arm's highest reach,
      // ~0.63 H — the chest-height show read as nothing in the live frames); a big one stays below the face (it would hide it)
      if (kind === 'show') return p.size < 0.12 ? [sg * 0.3, B0.lo[1] + 0.7 * Hh, B0.front + 0.5 * p.size] : [sg * 0.26, B0.lo[1] + 0.6 * Hh, B0.front + p.size + 0.18];
      if (kind === 'inspect') { const hp = headPoint(); return [sg * 0.1, hp[1] - 0.08, hp[2] + 0.3 + p.size]; }
      if (kind === 'offer') return [sg * 0.18, B0.lo[1] + 0.68 * Hh, B0.front + 0.05];   // its short arms never reach out past its front (reach grid: forward targets clamp to belly height, 0.40 H) — handed over high at its front (~0.6 H), not "held low at the belly"
      if (kind === 'shake') return showPoint(s, p, 'show');   // shaken where he sees it (at the carry point it read as 'held low at the waist, no shake')
      if (kind === 'pour') return [sg * 0.08, B0.lo[1] + 0.5 * Hh, B0.front + p.size + 0.32];   // (pouring onto the floor keeps the old held-out point)
      if (kind === 'hug') return [0, B0.lo[1] + 0.48 * Hh, B0.front + p.size * 0.6 + 0.01];
      return carryPoint(s, p);
    }
    function reachClamp(s, P) {   // the reachable point nearest P (grasp frame of a held prop)
      const r = invW(s, P, E.paw[s].hPrev, 14);
      return r.p;
    }
    function poseChain(p, kind, a) {
      const s = p.held === 'both' ? 'l' : p.held, two = p.held === 'both', out = [];
      const target = (sd) => reachClamp(sd, toWorld(two ? add3(showPoint('l', p, kind), [(sd === 'l' ? -1 : 1) * p.size * 0.95, 0, 0]) : showPoint(sd, p, kind)));   // showPoint is body-relative (it aimed at the room's origin away from home)
      const sides = two ? ['l', 'r'] : [s];
      const dur = kind === 'inspect' ? 0.7 : 0.6;
      const turnFor = kind === 'show' ? 0.85 : kind === 'offer' ? 1 : kind === 'inspect' ? 0.35 : 0;
      out.push(par(sides.map(sd => pawPath(sd, target(sd), dur, {turn: turnFor, look: kind === 'inspect' ? 'paw' : undefined}))));
      const r0 = () => p.hrot.slice();
      const hold = a.hold ? Infinity : kind === 'inspect' ? 3.2 : kind === 'hug' ? 2.6 : kind === 'shake' ? 2.2 : kind === 'show' ? 5 : kind === 'offer' ? 6 : 2.4;   // a readable moment — shown 5 s, held out to him 6 s (3.4 s fell between the frames he saw; live v2/v3: the close-up caught the lowered paw)   // offer 3.4 s, shake 2.2 s (the push-in arrived as it went down)   // show 2.4 -> 3.4 s (the camera eases in on it)
      if (kind === 'show') {
        out.push(stepFn('face the camera', 0.45, (u) => { p.hrot[1] = lerp(this0(p), frontFacing(p), mj(u)); }, {start() { p.showFrom = p.hrot[1]; }}));
      }
      let base = null;
      const pres = presenter(p, kind, two, s, hold);   // a small thing shown / given / shaken sits OUT of the fist
      out.push({name: kind, dur: hold, start() { base = sides.map(sd => fwdW(sd, E.paw[sd].hPrev)); this.r = r0(); if (kind === 'inspect') { E.look.wT = 1; } if (kind === 'show' || kind === 'offer') { E.look.wT = 0; } pres.start(); },
        update(tt) {
          const sp = 1; pres.update(tt);
          sides.forEach((sd, i) => {
            const pw = E.paw[sd]; let P = base[i];
            if (kind === 'inspect') { const ph = tt * 1.1; P = add3(P, [0.015 * Math.sin(ph * 1.3), 0.012 * Math.sin(ph), 0]); p.hrot = [this.r[0] + 0.05 * Math.sin(ph * 0.9), this.r[1] + 0.14 * Math.sin(ph * 0.7), this.r[2]]; E.look.P = P; }
            if (kind === 'shake') { const k = Math.sin(Math.PI * clamp(tt / hold, 0, 1)); P = add3(P, [0, 0.045 * Math.sin(TAU * 5 * tt) * k, 0.01 * Math.sin(TAU * 5 * tt + 1) * k]); p.hrot = [this.r[0], this.r[1], this.r[2] + 0.04 * Math.sin(TAU * 5 * tt + 0.5) * k]; }
            if (kind === 'hug') { const k = Math.sin(Math.PI * clamp(tt / Math.min(hold, 2.6), 0, 1)); P = add3(P, [0.02 * Math.sin(TAU * 0.6 * tt) * k, 0, 0]); }
            pw.tgt = P; void sp;
          });
          return tt >= hold;
        }, end() { if (kind === 'inspect' || kind === 'shake') p.hrot = this.r.slice(); pres.end(); }});
      if (kind === 'hug') {   // the free paw comes over it
        const o = otherSide(s);
        if (!two && !heldBy(o)) out.splice(1, 0, pawPath(o, toWorld(add3(showPoint(s, p, 'hug'), [(o === 'l' ? -1 : 1) * (p.size + 0.03), 0.02, 0.03])), 0.5, {open: 0.8, turn: 0}));
      }
      out.push(lookStep(null, false));
      out.push(par(sides.map(sd => retractStep(sd, true)).concat(kind === 'hug' && !two && !heldBy(otherSide(s)) ? [retractStep(otherSide(s))] : [])));
      return out;
    }
    function this0(p) { return p.showFrom ?? p.hrot[1]; }
    // VISIBILITY: a small thing (chalk, eraser, ball, mug...) held to be SEEN — shown, given, waved,
    // shaken — was swallowed by the fist (live: "a tiny sliver tucked in a fist", the chalk "a white speck"). While it is
    // presented it rides up out of the paw toward the camera (the held frame faces the camera: +y up, +z toward him) and
    // the paw opens a little around it, eased in and out inside the hold so a put-down later starts from the normal grip.
    const PRESENT = /^(show|offer|shake|wave)$/;
    function presenter(p, kind, two, s, hold) {
      const on = !two && PRESENT.test(kind) && p.size < 0.12 && p.held && p.held !== 'both';
      let o0 = null, op0 = 0;
      const lift = clamp(1.5 * p.size + 0.012, 0.02, 0.08), fwd = clamp(0.5 * p.size, 0.008, 0.03);
      const kAt = tt => { const kin = mj(clamp(tt / 0.4, 0, 1)); return Number.isFinite(hold) ? kin * (1 - mj(clamp((tt - (hold - 0.35)) / 0.35, 0, 1))) : kin; };
      return {
        start() { if (!on) return; o0 = (p.hoff || [0, 0, 0]).slice(); op0 = graspValues(p, s).open; },
        update(tt) { if (!on || !o0) return; const k = kAt(tt); p.hoff = add3(o0, [0, lift * k, fwd * k]); E.paw[s].open = lerp(op0, Math.max(op0, 0.42), k); },
        end() { if (!on || !o0) return; p.hoff = o0; E.paw[s].open = op0; o0 = null; },
      };
    }
    function rotateChain(p, axis, amount, a) {
      const s = p.held === 'both' ? 'l' : p.held, two = p.held === 'both', sides = two ? ['l', 'r'] : [s], out = [];
      out.push(par(sides.map(sd => pawPath(sd, reachClamp(sd, toWorld(two ? add3(showPoint('l', p, 'show'), [(sd === 'l' ? -1 : 1) * p.size * 0.95, -0.1, 0]) : add3(showPoint(sd, p, 'show'), [0, -0.1, 0]))), 0.55, {turn: 0.5}))));
      const dur = clamp(0.6 + 0.9 * Math.abs(amount), 0.6, 3);
      out.push({name: 'rotate', dur, start() { this.r0 = p.hrot.slice(); this.t0 = sides.map(sd => E.paw[sd].turn); }, update(tt) {
        const u = clamp(tt / dur, 0, 1), m = mj(u);
        p.hrot = this.r0.slice(); p.hrot[axis] = this.r0[axis] + amount * m;
        sides.forEach((sd, i) => { E.paw[sd].turn = clamp(this.t0[i] + 0.45 * Math.sin(Math.PI * u) * Math.sign(amount), -1, 1); });
        return u >= 1;
      }, end() { p.hrot[axis] = wrapH(p.hrot[axis]); emit({kind: 'object_event', name: axis === 1 ? 'turned' : 'flipped', phase: 'end', src: 'auto', args: {object: p.id, amount}, did: axis === 1 ? `turned the ${p.name} ${turnsWord(Math.abs(amount))} in its paw` : `flipped the ${p.name} over`}); }});
      out.push(dwellStep('look at it', a.hold ? Infinity : 1.0, () => {}));
      out.push(par(sides.map(sd => retractStep(sd, true))));
      return out;
    }
    function handOverChain(p, from, to) {
      const out = [], B0 = E.body;
      let M = [0, B0.lo[1] + 0.45 * (B0.hi[1] - B0.lo[1]), B0.front + p.size + 0.08];
      for (let k = 0; k < 6 && !(reachable('l', M).ok && reachable('r', M).ok); k++) M = add3(M, [0, 0.02, 0.03]);
      out.push(lookStep(M, true));
      out.push(par([pawPath(from, M, 0.55, {look: M}), pawPath(to, add3(M, [(to === 'l' ? -1 : 1) * 0.0, 0, 0]), 0.6, {open: 0.85, turn: 0})]));
      out.push({name: 'hand over', atomic: true, dur: 0.32, start() { this.g = graspValues(p, to); }, update(tt) {
        const u = clamp(tt / this.dur, 0, 1), w = mj(u);
        p.att[from] = 1 - w; p.att[to] = w;
        E.paw[to].open = lerp(0.85, this.g.open, w); E.paw[from].open = lerp(graspValues(p, from).open, 0.6, w);
        return u >= 1;
      }, end() { p.att[from] = 0; p.att[to] = 1; p.held = to; emit({kind: 'object_event', name: 'handed_over', phase: 'end', src: 'auto', args: {object: p.id, from: sideWord(from), to: sideWord(to)}, did: `passed the ${p.name} from its ${sideWord(from)} paw to its ${sideWord(to)} paw`}); }});
      out.push(lookStep(null, false));
      out.push(par([retractStep(from), retractStep(to, true)]));
      return out;
    }

    // --- object command → chain
    const VERB_ALIAS = {pick_up: 'pick_up', pickup: 'pick_up', grab: 'pick_up', take: 'pick_up', get: 'pick_up', lift: 'pick_up', hold: 'pick_up',
      put_down: 'put_down', set_down: 'put_down', put: 'put_down', release: 'put_down', put_back: 'put_back', return: 'put_back', replace: 'put_back', tidy: 'put_back', clean_up: 'put_back',
      place: 'place', put_on: 'place', set_on: 'place', move: 'place', hold_up: 'show', show: 'show', present: 'show', display: 'show',
      inspect: 'inspect', look_at: 'inspect', examine: 'inspect', check: 'inspect', study: 'inspect', smell: 'inspect', sniff: 'inspect',
      turn: 'turn', rotate: 'turn', spin: 'spin', twirl: 'twirl', spin_around: 'spin', turn_around: 'turn', flip: 'flip',
      tilt: 'tilt', tip: 'tilt', lean: 'tilt', wave: 'wave', wave_around: 'wave', raise: 'raise', raise_up: 'raise', lift_up: 'raise', hold_up_high: 'raise', overhead: 'raise', hold_overhead: 'raise',
      hide: 'hide', hide_behind_back: 'hide', conceal: 'hide', drink: 'drink', sip: 'drink', drink_from: 'drink', pour: 'pour', pour_out: 'pour', empty: 'pour',   // objects engine turn_over: 'flip', upside_down: 'flip',
      hand_over: 'hand_over', pass: 'hand_over', switch_paws: 'hand_over', swap: 'hand_over', switch_hands: 'hand_over', other_paw: 'hand_over',
      toss: 'toss', throw: 'toss', juggle: 'toss', catch: 'toss', flip_up: 'toss', drop: 'drop', let_go: 'drop', shake: 'shake', rattle: 'shake',
      hug: 'hug', cuddle: 'hug', squeeze: 'hug', offer: 'offer', give: 'offer', hand: 'offer', hold_out: 'offer'};
    const OBJECT_VERBS = ['pick_up', 'put_down', 'put_back', 'place', 'show', 'inspect', 'turn', 'flip', 'hand_over', 'toss', 'drop', 'shake', 'hug', 'offer',
      'spin', 'twirl', 'tilt', 'wave', 'raise', 'hide', 'drink', 'pour', 'stack', 'put_in', 'balance', 'throw', 'juggle', 'use', 'water', 'wear', 'take_off', 'open', 'close', 'touch', 'go_to'];   // (the muse's general verbs too; touch, go_to)
    function objectCommand(cmd, src, id) {
      const vraw = String(cmd.verb || cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_');
      const verb = VERB_ALIAS[vraw];
      if (!verb) return {ok: false, reason: `I don't know how to "${vraw.replace(/_/g, ' ') || '?'}" an object`};
      if (!E.P.length) return {ok: false, reason: 'there are no objects here'};
      const holdVerb = !/^(pick_up)$/.test(verb);
      // which prop
      let p, res;
      const desc = cmd.object;
      if (verb === 'put_back' && desc && (desc.ref === 'all' || desc === 'all' || desc.name === 'all' || desc.name === 'everything')) return putBackAll(src, id, cmd);
      if (!desc || (typeof desc === 'object' && !Object.keys(desc).some(k => desc[k] !== undefined && desc[k] !== null && desc[k] !== ''))) {
        const held = E.P.filter(q => q.held);
        if (holdVerb && held.length === 1) p = held[0];
        else if (holdVerb && held.length > 1) return {ok: false, reason: `which one — ${held.map(q => 'the ' + q.name).join(' or ')}?`, ask: true};
        else if (E.P.length === 1) p = E.P[0];
        else if (verb === 'pick_up') { const free = E.P.filter(q => !q.held); if (free.length === 1) p = free[0]; else return {ok: false, reason: `which one — ${free.map(q => 'the ' + q.name).join(', ')}?`, ask: true}; }
        else if (holdVerb) return {ok: false, reason: 'I am not holding anything'};
      } else {
        res = resolveObject(desc, verb, {preferHeld: holdVerb && verb !== 'put_back', preferFree: verb === 'pick_up'});
        if (res.err) return {ok: false, reason: res.err, ask: res.ask};
        p = res.p;
      }
      E.lastRef = p;
      const side0 = sideIn(cmd.side);
      const steps = [], did = [];
      const two = p.def.grip === 'two_hand';
      // pick it up first if the verb needs it in a paw
      let s = p.held === 'both' ? 'l' : p.held;
      const needHeld = !/^(pick_up|put_back|put_down|place|drop)$/.test(verb) || (/^(put_back|put_down|place|drop)$/.test(verb) && false);
      const pickSide = () => {
        if (two) { if (heldBy('l') || heldBy('r')) return {err: `I need both paws free for the ${p.name}`}; return {s: 'l'}; }
        const pref = side0 && side0 !== 'both' ? side0 : null;
        const order = pref ? [pref] : (p.pos[0] >= 0 ? ['r', 'l'] : ['l', 'r']);
        const free = order.filter(x => !heldBy(x));
        if (!free.length) return {err: 'my paws are full'};
        if (pref && heldBy(pref)) return {err: `my ${sideWord(pref)} paw is holding the ${heldBy(pref).name}`};
        for (const x of free) if (reachable(x, p.pos).ok) return {s: x};
        return {err: `the ${p.name} is out of my reach`};
      };
      if (verb === 'pick_up') {
        if (p.held) {
          if (side0 && side0 !== 'both' && p.held !== side0 && p.held !== 'both' && !heldBy(side0)) { steps.push(...handOverChain(p, p.held, side0)); did.push(`moving the ${p.name} to its ${sideWord(side0)} paw`); }
          else return {ok: true, did: `already holding the ${p.name} in ${p.held === 'both' ? 'both paws' : 'its ' + sideWord(p.held) + ' paw'}`, already: true, noop: true};
        } else {
          const ps = pickSide(); if (ps.err) return {ok: false, reason: ps.err};
          s = ps.s; steps.push(...chainPickUp(p, s, two)); did.push(`picking up the ${p.name} with ${two ? 'both paws' : 'its ' + sideWord(s) + ' paw'}`);
          if (cmd.hold === true) { /* holding is the end state anyway */ }
        }
      } else if (needHeld && !p.held) {
        if (p.flying) return {ok: false, reason: `the ${p.name} is in the air`};
        const ps = pickSide(); if (ps.err) return {ok: false, reason: ps.err};
        s = ps.s; steps.push(...chainPickUp(p, s, two)); did.push(`picking up the ${p.name}`);
      }
      const willHold = () => p.held || steps.length;
      switch (verb) {
        case 'pick_up': break;
        case 'put_back': case 'put_down': case 'place': case 'drop': {
          if (!p.held) {
            if (verb === 'drop') return {ok: false, reason: `I am not holding the ${p.name}`};
            // move a prop that is not in a paw: pick it up, then put it there
            if (verb === 'put_back' && p.where === 'home') return {ok: true, did: `the ${p.name} is already where it belongs`, already: true, noop: true};
            const ps = pickSide(); if (ps.err) return {ok: false, reason: ps.err};
            s = ps.s; steps.push(...chainPickUp(p, s, two)); did.push(`picking up the ${p.name}`);
          }
          const hs = () => p.held === 'both' ? 'l' : p.held || s;
          if (verb === 'drop') {
            steps.push(dropStep(hs(), p, 'drop'));
            steps.push(retractStep(hs())); did.push(`dropping the ${p.name}`); break;
          }
          let rest = null, pl = null;
          if (verb === 'put_back') { rest = p.home.slice(); if (!spotFree(rest, p)) { const q = E.P.find(o => o !== p && !o.held && Math.hypot(o.pos[0] - rest[0], o.pos[2] - rest[2]) < (o.size + p.size)); return {ok: false, reason: `the ${q ? q.name : 'other thing'} is in its spot`}; } did.push(`putting the ${p.name} back where it belongs`); }
          else {
            if (verb === 'place' || cmd.place) { pl = resolvePlace(cmd.place, p); if (pl && pl.err) return {ok: false, reason: pl.err}; }
            if (!pl) pl = E.places.find(x => x.id === (p.place || p.homePlace)) || E.places.find(x => x.id === p.homePlace) || E.places.find(x => x.floor);
            rest = freeSpot(pl, p, hs() || 'r');
            if (!rest && !pl.floor) return {ok: false, reason: `there is no room on ${pl.name}`};
            if (!rest) rest = [((hs() || 'r') === 'l' ? -1 : 1) * 0.25, pl.top_y + p.restH, 0.55];
            did.push(`putting the ${p.name} down on ${pl.name}`);
          }
          const sideNow = p.held ? (p.held === 'both' ? 'l' : p.held) : s;
          const ch = chainPutAt(p, sideNow, rest, verb);
          if (!ch) return {ok: false, reason: `I can't reach ${pl ? pl.name : 'there'}`};
          if (!p.held && !steps.length) return {ok: false, reason: `I am not holding the ${p.name}`};
          steps.push(...ch);
          break;
        }
        case 'show': case 'inspect': case 'offer': case 'hug': case 'shake': {
          steps.push(lazy(() => poseChain(p, verb, cmd)));
          did.push({show: `holding the ${p.name} up to show him`, inspect: `looking closely at the ${p.name}`, offer: `holding the ${p.name} out to him`, hug: `hugging the ${p.name}`, shake: `shaking the ${p.name}`}[verb]);
          break;
        }
        case 'turn': case 'flip': {
          const sg = yawSign(cmd.dir, 1);
          const amt = verb === 'flip' ? 0.5 : Number.isFinite(+cmd.amount) ? clamp(+cmd.amount, -3, 3) : num(cmd.amount, 0.5);
          steps.push(lazy(() => rotateChain(p, verb === 'flip' ? 0 : 1, verb === 'flip' ? 0.5 : sg * Math.abs(amt), cmd)));
          did.push(verb === 'flip' ? `flipping the ${p.name} upside down` : `turning the ${p.name} ${turnsWord(Math.abs(amt))}${amt === 0.5 ? ' (around)' : ''}`);
          break;
        }
        case 'spin': case 'twirl': {   // objects engine: in-paw manipulations from parameters
          const n = clamp(Math.abs(num(cmd.amount ?? cmd.count, verb === 'twirl' ? 1 : 2)), 0.25, 12), fing = /finger|fingertip/.test([cmd.style, cmd.on, cmd.how, cmd.where, cmd.target && (cmd.target.name || cmd.target)].filter(x => typeof x === 'string').join(' ').toLowerCase());
          if (fing && two) return {ok: false, reason: `the ${p.name} is too big to spin on a finger`};
          steps.push(lazy(() => spinChain(p, cmd, verb === 'twirl'))); did.push(`spinning the ${p.name} ${turnsWord(n)}${fing ? ' on its fingertip' : ''}`); break;
        }
        case 'tilt': steps.push(lazy(() => tiltChain(p, cmd))); did.push(`tilting the ${p.name}`); break;
        case 'wave': steps.push(lazy(() => waveChain(p, cmd))); did.push(`waving the ${p.name} around`); break;
        case 'raise': steps.push(lazy(() => raiseChain(p, cmd))); did.push(`raising the ${p.name} over its head`); break;
        case 'hide': if (two) return {ok: false, reason: `the ${p.name} is too big to hide behind its back`}; steps.push(lazy(() => hideChain(p, cmd))); did.push(`hiding the ${p.name} behind its back`); break;
        case 'drink': if (!AFF(p).drink) return {ok: false, reason: `I can't drink from the ${p.name}`}; steps.push(lazy(() => drinkChain(p, cmd))); did.push(`taking a sip from the ${p.name}`); break;
        case 'pour': if (!AFF(p).pour) return {ok: false, reason: `the ${p.name} has nothing in it to pour`}; steps.push(lazy(() => pourChain(p, cmd))); did.push(`pouring ${AFF(p).pour.what || 'it'} out of the ${p.name}`); break;
        case 'hand_over': {
          if (two) return {ok: false, reason: `the ${p.name} takes both paws`};
          steps.push(lazy(() => { const fr = p.held, to = otherSide(fr); if (heldBy(to)) return null; return handOverChain(p, fr, to); }));
          const fr = p.held || s, to = otherSide(fr);
          if (heldBy(to) && heldBy(to) !== p) return {ok: false, reason: `my ${sideWord(to)} paw is holding the ${heldBy(to).name}`};
          did.push(`passing the ${p.name} to its ${sideWord(to)} paw`);
          break;
        }
        case 'toss': {
          if (p.def.throwable === false) return {ok: false, reason: `the ${p.name} isn't something to toss`};
          if (two) return {ok: false, reason: `the ${p.name} is too big to toss`};
          const n = clamp(Math.round(num(cmd.count, 1)), 1, 6), hh = lerp(0.3, 0.75, heightOf(cmd.height, 0.5));
          const alternate = side0 === 'both' || /other|between|juggle/.test(String(cmd.style || cmd.dir || '') + vraw);
          for (let i = 0; i < n; i++) steps.push(lazy(() => { const fr = p.held; if (!fr || fr === 'both') return null; const to = alternate && !heldBy(otherSide(fr)) ? otherSide(fr) : fr; return tossSteps(p, fr, to, hh); }));
          steps.push(lazy(() => [lookStep(null, false), retractStep(p.held || 'r', true)].concat(alternate ? [retractStep(otherSide(p.held || 'r'))] : [])));
          did.push(`tossing the ${p.name} up and catching it${n > 1 ? ' ' + times(n) : ''}${alternate ? ' from paw to paw' : ''}`);
          break;
        }
      }
      void willHold;
      if (!steps.length) return {ok: false, reason: 'nothing to do'};
      return startChain(verb, p, steps, did.join(', then '), src, id, cmd);
    }
    function lazy(f) { return {name: 'lazy', lazy: f}; }
    function putBackAll(src, id, cmd) {
      const away = E.P.filter(p => p.where !== 'home' || p.held);
      if (!away.length) return {ok: true, did: 'everything is already where it belongs', already: true, noop: true};
      const steps = [];
      for (const p of away) steps.push(lazy(() => {
        const out = [];
        let s = p.held === 'both' ? 'l' : p.held;
        if (!s) { const two = p.def.grip === 'two_hand'; s = two ? 'l' : (['r', 'l'].find(x => !heldBy(x) && reachable(x, p.pos).ok)); if (!s) return null; out.push(...chainPickUp(p, s, two)); }
        out.push(lazy(() => chainPutAt(p, p.held === 'both' ? 'l' : p.held, p.home.slice(), 'put_back')));
        return out;
      }));
      return startChain('put_back', away[0], steps, `putting ${away.length === 1 ? 'the ' + away[0].name : 'everything'} back where it belongs`, src, id, cmd);
    }
    function startChain(verb, p, steps, did, src, id, cmd) {
      E.chain = {verb, p, steps, i: -1, cur: null, t0: 0, did, src, id, args: cmd, started: t(), hold: !!cmd.hold};
      return {ok: true, did, chain: E.chain};
    }
    // per-frame object chain update
    function chainFrame() {
      const C = E.chain; if (!C) return;
      // a stopped chain only finishes its atomic step and brings the paws home: never longer than a few seconds (
      // a stopped chain that kept going held its job — and every job queued behind it — forever)
      if (C.stopped && C.stopT !== undefined && t() - C.stopT > 6 && !(C.cur && C.cur.atomic)) { if (C.cur && C.cur.end) C.cur.end(); C.cur = null; endChain(C, 'interrupted'); return; }
      for (let guard = 0; guard < 8; guard++) {
        if (!C.cur) {
          C.i++;
          if (C.i >= C.steps.length) { endChain(C, C.stopped ? 'interrupted' : 'end'); return; }
          let st = C.steps[C.i];
          if (st.lazy) { const r = st.lazy(); C.steps.splice(C.i, 1, ...(r || [])); C.i--; continue; }
          C.cur = st; C.t0 = t(); st.start && st.start();
        }
        const st = C.cur, tt = t() - C.t0;
        const done = st.update(tt);
        if (done) { st.end && st.end(); C.cur = null; if (C.stopped && !C.aborted && !st.atomic && C.i < C.steps.length - 1) { abortTail(C); } continue; }
        return;
      }
    }
    function abortTail(C) {   // after an atomic section: drop the rest, bring the paws home with the prop still held
      const keep = [];
      keep.push(lookStep(null, false));
      for (const s of ['l', 'r']) if (E.paw[s].ctrlT > 0 || E.paw[s].ctrl > 0) keep.push(retractStep(s, true));
      C.steps = C.steps.slice(0, C.i + 1).concat([par(keep.filter(x => x.name === 'retract')), keep[0]]);
      C.aborted = true;
    }
    function stopChain(reason) {
      const C = E.chain; if (!C || C.stopped) return;
      C.stopped = reason || 'stopped'; C.stopT = t();
      if (C.cur && C.cur.atomic) return;   // finishes the grasp / release / flight first (chainFrame aborts after it)
      if (C.cur) { C.cur = null; }
      abortTail(C);
      C.i = Math.min(C.i, C.steps.length - 1);
      // abortTail kept steps up to i; continue right after i
      C.steps.splice(C.i, 0); C.i = C.i;   // (no-op: clarity)
      if (C.i >= 0 && C.steps[C.i] && !C.steps[C.i].done) { /* the current one was cut: skip it */ }
    }
    function pastOf(d) {
      return String(d || '').replace(/\bpicking up\b/g, 'picked up').replace(/\bputting\b/g, 'put').replace(/\bholding\b/g, 'held').replace(/\blooking\b/g, 'looked')
        .replace(/\bhugging\b/g, 'hugged').replace(/\bshaking\b/g, 'shook').replace(/\bturning\b/g, 'turned').replace(/\bflipping\b/g, 'flipped').replace(/\bpassing\b/g, 'passed')
        .replace(/\btossing\b/g, 'tossed').replace(/\bcatching\b/g, 'caught').replace(/\bdropping\b/g, 'dropped').replace(/\bmoving\b/g, 'moved')
        .replace(/\bspinning\b/g, 'spun').replace(/\btilting\b/g, 'tilted').replace(/\bwaving\b/g, 'waved').replace(/\braising\b/g, 'raised').replace(/\bhiding\b/g, 'hid').replace(/\btaking a sip\b/g, 'took a sip').replace(/\bpouring\b/g, 'poured');
    }
    function endChain(C, phase) {
      E.chain = null;
      for (const s of ['l', 'r']) if (!E.paw[s].keep) E.paw[s].ctrlT = 0;
      E.look.wT = 0;
      const p = C.p;
      let did = pastOf(C.did);
      if (phase === 'interrupted') did = `${did} — stopped partway; the ${p.name} is ${p.held ? 'still in its ' + (p.held === 'both' ? 'paws' : sideWord(p.held) + ' paw') : 'on ' + placeName(p.place || p.homePlace)}`;
      // a step that found it could not do its part (C.err): that op is NOT done — the job fails truthfully with it
      if (C.err && C.opOf) { const J = C.opOf, i = J.done ? J.done.lastIndexOf(C.op) : -1; if (i >= 0) J.done.splice(i, 1); J.failErr = C.err; }
      if (C.silent) { if (C.opOf && phase === 'interrupted') C.opOf.stopped = true; return; }
      emit({kind: 'object', name: C.verb, id: C.id, src: C.src, phase, args: C.args, did, state: {object: p.id, where: propWhere(p)}});
    }

    // ---------------------------------------------------------------- commands
    function command(cmd, src, idIn) {
      const id = idIn || ++E.cmdId;
      if (E.body.legs) {
        const rid = cmd && (cmd.replaces ?? (cmd.refine && E.lastJob ? E.lastJob.id : undefined));
        if (rid !== undefined && rid !== null) { const r = refineJob(cmd, src, id, rid); if (r) return r; }
        const r = museCommand(cmd, src, id);
        if (r && r.ok) E.lastJob = {id, t: t(), key: cmdKey(cmd)};
        // the same source's next step, right after it refined an earlier step of the request, follows that step
        // (the voice model restates the whole request call by call) instead of jumping to the back of the queue
        const B = E.burst, k = r && r.ok ? E.jobs.findIndex(j => j.id === id) : -1;
        if (B && k >= 0 && B.src === src && t() - B.t < 2) { const b = E.jobs.findIndex(j => j.id === B.id); if (b >= 0 && k > b + 1) E.jobs.splice(b + 1, 0, E.jobs.splice(k, 1)[0]); if (b >= 0) E.burst = {src, id, t: t()}; }
        return r;
      }
      if (cmd.action === 'object') {
        const r = objectCommand(cmd, src, id);
        return finishCmd(r, 'object', String(cmd.verb || cmd.name || ''), cmd, src, id);
      }
      // moves
      const raw = String(cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_');
      let name = MOVE_ALIAS[raw], extra = {};
      if (Array.isArray(name)) { extra = name[1]; name = name[0]; }
      if (!name) return {ok: false, id, reason: `I don't know the move "${raw.replace(/_/g, ' ') || '?'}"`};
      if (name === 'stop') { const had = !!(E.move || E.chain); stopAll('stopped'); return {ok: true, id, did: had ? 'stopping what it was doing' : 'nothing to stop (it was not moving)'}; }
      const a = Object.assign({}, extra, cmd);
      if (name === 'flip' && /forward|front/.test(normDir(a.dir))) name = 'frontflip'; else if (name === 'flip') name = 'backflip';
      name = rotateAmount(name, a);
      if (name === 'turn' && /^(around|back|away)$/.test(normDir(a.dir))) name = 'turn_away';
      if ((name === 'jump' || name === 'hop') && /(forward|back|left|right|side)/.test(normDir(a.dir))) { name = 'walk'; }
      if (name === 'stand' && E.posture === 'lying') name = 'get_up';
      const r = {name, a};
      return finishCmd(r, 'move', name, a, src, id);
    }
    function finishCmd(r, kind, name, args, src, id) {
      if (r.ok === false) { emit({kind, name, id, src, phase: 'fail', args, reason: r.reason}); return {ok: false, id, reason: r.reason, ask: r.ask}; }
      if (r.noop) { emit({kind, name, id, src, phase: 'end', args, did: r.did}); return {ok: true, id, did: r.did, already: true}; }
      // a new command waits for the current one to end gracefully (atomic object sections, airborne moves, braking)
      const q = {kind, r, name, args, src, id};
      const busyMove = E.move && !(E.move.holdEnd && t() - E.move.t0 >= E.move.dur);
      if (kind === 'move') {
        // validate now (so the tool result is truthful): build into a scratch plan when idle
        if (E.chain) { stopChain('superseded'); E.queue = q; return {ok: true, id, did: `${moveDidPreview(r)} (after it finishes with its paws)`, queued: true}; }
        if (E.move) { stopMove('superseded'); E.queue = q; return {ok: true, id, did: `${moveDidPreview(r)} (after it finishes the current move)`, queued: true}; }
        return runQueued(q);
      }
      // object chain
      if (E.chain) { stopChain('superseded'); E.oqueue = q; emit({kind, name, id, src, phase: 'start', args, did: r.did + ' (next)'}); return {ok: true, id, did: r.did + ' (after the current one)', queued: true}; }
      if (busyMove || E.move || !atRest(0.02)) {
        // objects need it seated and facing him (the reach map is calibrated there): settle first
        if (E.move) stopMove('object');
        if (!E.move && !atRest(0.02)) settleHome();
        E.oqueue = q; E.chain = null;
        const pre = E.posture === 'lying' ? 'getting up, then ' : Math.abs(E.base.yaw) > 0.02 || Math.abs(E.cur.yaw) > 0.02 ? 'turning back to him, then ' : '';
        emit({kind, name, id, src, phase: 'start', args, did: pre + r.did});
        return {ok: true, id, did: pre + r.did, queued: true};
      }
      emit({kind, name, id, src, phase: 'start', args, did: r.did});
      E.chain.t0 = t();
      return {ok: true, id, did: r.did};
    }
    function moveDidPreview(r) { return (r.a && r.name ? r.name.replace(/_/g, ' ') : 'the move'); }
    function runQueued(q) {
      if (q.kind === 'move') {
        const r = startMove(q.r.name, q.r.a, q.src, q.id);
        if (!r.ok) { emit({kind: 'move', name: q.r.name, id: q.id, src: q.src, phase: 'fail', args: q.args, reason: r.reason}); return {ok: false, id: q.id, reason: r.reason}; }
        emit({kind: 'move', name: q.r.name, id: q.id, src: q.src, phase: 'start', args: q.args, did: r.did});
        return {ok: true, id: q.id, did: r.did};
      }
      // an object command validated earlier: re-validate against the state now (things may have moved)
      const r = objectCommand(q.args, q.src, q.id);
      if (!r.ok) { emit({kind: 'object', name: q.name, id: q.id, src: q.src, phase: 'fail', args: q.args, reason: r.reason}); return r; }
      if (r.noop) { emit({kind: 'object', name: q.name, id: q.id, src: q.src, phase: 'end', args: q.args, did: r.did}); return r; }
      E.chain.t0 = t(); return r;
    }
    function settleHome() {   // ease back to the seated rest pose (used before object work)
      const P = new Plan('settle', {}, 'auto');
      if (E.posture === 'lying') getUp(P, 1.2);
      else if (P.T.stand.v > 0.05) sitDown(P, 1.2);
      settleTo(P, restBase(), clamp(0.45 + 1.0 * Math.abs(wrap(P.T.yaw.v)) + 0.8 * Math.hypot(P.T.x.v, P.T.z.v), 0.4, 1.4));
      P.endBase = restBase(); P.endPosture = 'seated'; P.did = ['getting back to its seat', 'got back to its seat']; P.id = 0; P.dur = P.te(); P.whole = false; P.headW = 0.3;
      P.silent = true; E.move = P;
    }
    function stopAll(reason) {
      if (E.body.legs) {
        const J = E.jobs[0]; if (J) { J.stopped = true; J.ops = J.ops ? J.ops.filter(o => o.op === '_letgo') : []; }
        for (const J2 of E.jobs.slice(1)) emit({kind: J2.kind, name: J2.name, id: J2.id, src: J2.src, phase: 'interrupted', args: J2.args, did: 'not started (stopped)'});
        E.jobs = J ? [J] : [];
        if (E.activity) { const A = E.activity; E.activity = null; E.turnBack = false; for (const sd of ['l', 'r']) E.paw[sd].act = false; emit({kind: 'activity', name: A.kind, phase: 'end', src: 'auto', did: 'stopped typing'}); if (!E.chain) E.jobs.push({kind: 'move', name: 'relax', goals: [], ops: [{op: '_letgo', sub: true}], src: 'auto', id: 0, started: true, done: []}); }
        if (['l', 'r'].some(sd => E.paw[sd].keep) && !(J && J.ops.length)) E.jobs.push({kind: 'move', name: 'let go', goals: [], ops: [{op: '_letgo', sub: true}], src: 'auto', id: 0, started: true, done: []});
      }
      if (E.chain) stopChain(reason);
      if (E.move) stopMove(reason);
      E.queue = null; E.oqueue = null;
    }
    // "relax / okay / you can stop": ends held moves and held object poses; never drops what it holds
    function release(parts) {
      const all = parts === 'all' || !parts;
      const touches = all || ['root', 'body', 'everything'].some(k => (parts instanceof Set ? parts.has(k) : false));
      if (E.frozen && (all || touches)) { E.frozen = false; if (!E.move && !atRest(0.01)) { settleHomeFrom(); } }
      if (E.body.legs && (all || touches)) {
        // legs: "relax / okay / you can stop" ends running and held MOVES gracefully (a spin brakes to face him, a held
        // turn-away turns back, a lie-down gets up); object work, what it holds and what it wears are left alone
        for (const J2 of E.jobs.slice(1)) if (J2.kind === 'move') { E.jobs.splice(E.jobs.indexOf(J2), 1); emit({kind: J2.kind, name: J2.name, id: J2.id, src: J2.src, phase: 'interrupted', args: J2.args, did: 'not started (relaxed)', reason: 'released'}); }
        const J = E.jobs[0];
        if (E.move && !E.move.silent !== undefined && (!J || J.kind === 'move' || E.move.opOf !== J)) {
          if (J && J.kind === 'move' && E.move.opOf === J) { J.stopped = true; J.replacedWhy = 'relaxed'; J.ops = J.ops ? J.ops.filter(o => o.op === '_letgo') : []; }
          if (!E.move.stopping) { stopMove('released'); E.move && (E.move.released = true); }
        }
        // …and back to neutral: standing (not at work), it turns to face him again
        const busyObj = E.jobs.some(j => j.kind !== 'move') || E.chain || E.activity;
        if (!busyObj && !E.sitOn && E.posture !== 'lying' && Math.abs(wrap(E.move && E.move.endBase ? E.move.endBase.yaw : E.base.yaw)) > 0.04 && !E.jobs.some(j => j.name === 'face him'))
          E.jobs.push({kind: 'move', name: 'face him', goals: [{g: 'face'}], args: {action: 'move', name: 'turn_back'}, src: 'auto', id: 0, ops: null});
      } else if (E.move && (all || touches)) { if (E.move.holdEnd || E.move.dwell) stopMove('released'); }
      else if (!E.move && (all || touches) && !E.body.legs && (!atRest(0.01) || E.posture !== 'seated')) settleHomeFrom();
      if (E.body.legs && all && E.turnBack) {   // (c2) "relax / okay" after a turn back: back to the laptop, smoothly
        E.turnBack = false; if (E.chain && E.chain.verb === 'turn_back') stopChain('released');
        if (E.activity && E.activity.paused && !E.jobs.some(j => j.name === 'back to work'))
          E.jobs.push({kind: 'move', name: 'back to work', goals: [], ops: [{op: 'resume', auto: true, did: ['turning back to the laptop and getting back to typing', 'turned back to the laptop and got back to typing']}], src: 'auto', id: 0, started: true, done: []});
      } else if (E.body.legs && all && E.activity) stopActivity(null);
      if (E.chain && E.chain.hold && all) stopChain('released');
    }
    function settleHomeFrom() {
      const was = E.posture, P = new Plan('relax', {}, 'auto');
      if (was === 'lying') getUp(P, 1);
      else if (P.T.stand.v > 0.05) sitDown(P, 1);
      else if (was === 'tall' || was === 'crouched') { P.T.squash.to(0, 0.4); P.sync(); }
      settleTo(P, restBase(), clamp(0.45 + 1.1 * Math.abs(wrap(P.T.yaw.v)) + 0.8 * Math.hypot(P.T.x.v, P.T.z.v), 0.4, 1.6));
      P.endBase = restBase(); P.endPosture = 'seated'; P.id = 0; P.dur = P.te(); P.did = ['relaxing back into its seat', 'relaxed back into its seat'];
      P.silent = false; E.move = P;
      emit({kind: 'move', name: 'relax', id: 0, src: 'auto', phase: 'start', did: P.did[0]});
    }
    // "hold that / freeze": the root stops where it is (a quick brake; airborne → after it lands) and holds there
    function freeze() {
      const P = E.move;
      if (P) {
        const tt = t() - P.t0;
        const air = P.lands && P.lands.find(x => x >= tt - 0.02);
        if (air !== undefined) { stopMove('freeze'); P.freezeAfter = true; return true; }
        for (const c of ROOT) P.T[c].truncate(tt);
        for (const c of ROOT) { const v = P.T[c].vel(Math.max(0, tt - 0.02)); P.T[c].quint(P.T[c].v + v * 0.08, 0.2, v, 0); }
        P.sync(); P.dur = P.te(); P.holdEnd = true; P.dwell = Infinity; P.frozenHere = true;
        P.endBase = {}; for (const c of ROOT) P.endBase[c] = P.T[c].v;
        P.endPosture = E.posture; P.returnAt = undefined; P.interrupted = 'frozen';
        for (const x of P.x) if (x.t1 > tt) { const fn = x.fn, tf = tt; x.fn = () => fn(tf); x.t1 = Infinity; }
        E.frozen = true; return true;
      }
      E.frozen = true; return false;
    }

    // ================================================================ objects engine: the general object state model
    // (docs/OBJECTS.md §3/§5). Every geometric number comes from the props JSON (size, home, afford.*), so
    // an object added later with only its JSON works the same. A prop's attachment is one of:
    //   free at a pose (p.pos / p.rot) · resting on a place (p.place) · resting ON / IN another prop (p.sup = {id, rel, rr,
    //   inside}: a persistent pose relative to its support, re-derived every frame, also while the support is held / slid /
    //   dragged / carried; a support that tips over > 0.12 turn, flies or drops sheds it: it falls) · held (p.held, the paw
    //   attach weights) · worn (p.worn, hh = 1, hoff = afford.wear.offset) · balanced on the head (p.bal, hh = 1, hoff = the
    //   head top + its half height, a live wobble) · on a tracked body anchor (p.anc: foot_l / foot_r / back / lap /
    //   shoulder_l / shoulder_r — the renderer draws it free, the engine drives its free columns from the body pose) ·
    //   flying (p.fly: analytic ballistic segments under the engine's gravity; it bounces and settles onto the highest support
    //   under it: a place, another prop's top, or INTO an open container through its opening).
    // p.pos keeps the engine's render-space convention: for a prop whose home is on the floor the renderer adds the rug step
    // (floor_steps) between where it is and its home, so p.pos excludes that step. P(p) / setP(p, w) convert to and from the
    // PHYSICAL world position (H) that every query below uses (wpos, supports, flights, riders are physical).
    const GRAV = 12;                                         // H/s² (the engine's ballistic gravity, dropStep / tossSteps)
    const AFF = p => (p && p.def && p.def.afford) || {};
    const isCont = p => !!AFF(p).container;
    const contOpen = p => isCont(p) && (!AFF(p).container.needs_open || (p.s || 0) > 0.5);
    const hasArt = p => !!AFF(p).articulation && !feat(p).laptop;
    const topOff = p => Number.isFinite(+(AFF(p).surface || {}).top_y) ? +AFF(p).surface.top_y : p.hH;   // its top above its centre
    const loose = q => !q.held && !q.worn && !q.bal && !q.fly && !q.flying && !isFurn(q);   // resting somewhere (or on a body anchor)
    function floorTopAt(x, z) { let y = 0; for (const r of E.rug || []) if (x >= r.x0 && x <= r.x1 && z >= r.z0 && z <= r.z1) y = Math.max(y, r.top); return y; }
    const rugK = p => p.homePlace === 'floor' && !p.homeOn ? 1 : 0;
    function P(p, pos) { const q = pos || p.pos; return rugK(p) ? [q[0], q[1] + floorTopAt(q[0], q[2]) - floorTopAt(p.home[0], p.home[2]), q[2]] : q.slice(); }
    function toR(p, w) { return rugK(p) ? [w[0], w[1] - (floorTopAt(w[0], w[2]) - floorTopAt(p.home[0], p.home[2])), w[2]] : w.slice(); }
    function setP(p, w) { p.pos = toR(p, w); }
    function objInit() {   // per-prop geometry from the JSON (after setConfig built E.P)
      const H = E.props && E.props.units && +E.props.units.H || 2.3;
      E.rug = ((E.props && E.props.floor_steps) || []).filter(r => Array.isArray(r.rect_scene)).map(r => ({x0: r.rect_scene[0] / H, z0: r.rect_scene[1] / H, x1: r.rect_scene[2] / H, z1: r.rect_scene[3] / H, top: (+r.top_scene || 0) / H}));
      const rd = E.body.raw || {}, bH = +rd.H || H, hr = Array.isArray(rd.head_radii_scene) ? rd.head_radii_scene[1] / bH : 0.24;
      E.headTop = Number.isFinite(+(E.props && E.props.renderer || {}).head_top) ? +E.props.renderer.head_top : hr * 0.92;   // head frame → top of the head (H)
      E.footTop = rd.leg && Number.isFinite(+rd.leg.sole_below_ankle_scene) ? rd.leg.sole_below_ankle_scene / bH : 0.05;
      E.kfh = {l: null, r: null}; E.pawGW = {l: null, r: null}; E.pawGN = {l: null, r: null}; E.pawGV = {l: [0, 0, 0], r: [0, 0, 0]}; E.catchArm = {l: 0, r: 0}; E.lastDt = 1 / 30;
      for (const p of E.P) {
        const d = p.def, home = p.home, on = d.home && d.home.on ? byId(d.home.on) : null;
        p.homeOn = on ? on.id : null;
        const S = AFF(p).surface, fr = Math.min(p.size, 0.15) * 0.9;
        p.hx = S && Array.isArray(S.size) ? S.size[0] / 2 : fr; p.hz = S && Array.isArray(S.size) ? S.size[1] / 2 : fr;
        p.sup = null; p.bal = null; p.anc = null; p.fly = null; p.attE = null; p.wpos = home.slice(); p.wvel = [0, 0, 0];
      }
      for (const p of E.P) {
        const home = p.home, on = p.homeOn ? byId(p.homeOn) : null;
        if (on) p.hH = Math.max(0.005, home[1] - (on.home[1] + (Number.isFinite(+(AFF(on).surface || {}).top_y) ? +AFF(on).surface.top_y : 0.5 * on.size)));
        else if (p.homePlace === 'floor') p.hH = Math.max(0.005, home[1] - floorTopAt(home[0], home[2]));
        else p.hH = Math.max(0.005, p.restH);
      }
      for (const p of E.P) if (p.homeOn) { const q = byId(p.homeOn); p.sup = {id: q.id, rel: rotYv(sub3(P(p), P(q)), -(q.homeRot[1])), rr: p.homeRot[1] - q.homeRot[1], inside: false}; p.place = supPlace(q); }
    }
    const supPlace = q => { for (let i = 0; q && q.sup && i < 8; i++) q = byId(q.sup.id); return q ? (q.place || q.homePlace || 'floor') : 'floor'; };
    function tiltOf(r, k) { const a = Math.abs(wrap((r[0] || 0) + (k ? k.rx || 0 : 0))), b = Math.abs(wrap((r[2] || 0) + (k ? k.rz || 0 : 0))); return Math.max(a, b); }
    function chainOf(p) { const out = []; for (let q = p, i = 0; q && q.sup && i < 8; i++) { q = byId(q.sup.id); if (q) out.push(q); } return out; }
    const ridersOf = q => E.P.filter(r => r.sup && r.sup.id === q.id);
    function stackAbove(q) { const out = []; const go = x => { for (const r of ridersOf(x)) if (!out.includes(r)) { out.push(r); go(r); } }; go(q); return out; }
    // where a prop is in the world (physical) and how it is turned / tipped
    function worldOf(q) {
      if (q.held) return {pos: q.wpos, yaw: heldYawW(q), tilt: tiltOf(q.hrot, E.kfh[q.held === 'both' ? 'l' : q.held])};
      if (q.worn || q.bal) { const r = rootNow(); return {pos: q.wpos, yaw: r.yaw + q.homeRot[1], tilt: Math.max(Math.abs(wrap(r.pitch)), Math.abs(wrap(r.roll)))}; }
      return {pos: P(q), yaw: ryOf(q), tilt: tiltOf(q.rot)};
    }
    // the highest support under (x, z) whose top is at or below y (self, and whatever rides on self, excluded)
    function supportAt(x, z, y, self, opt = {}) {
      let best = {kind: 'place', id: 'floor', top: floorTopAt(x, z)};
      const lv = opt.margin ?? 0.002;
      for (const pl of E.places) {
        if (pl.floor) continue;
        const c = pl.center, l = rotYv([x - c[0], 0, z - c[2]], -(pl.yaw || 0)), sx = (pl.size[0] || 0.4) / 2, sz = (pl.size[1] || 0.4) / 2;
        if (Math.abs(l[0]) <= sx && Math.abs(l[2]) <= sz && pl.top_y <= y + lv && pl.top_y > best.top) best = {kind: 'place', id: pl.id, top: pl.top_y};
      }
      const up = self ? stackAbove(self) : [];
      for (const q of E.P) {
        if (q === self || up.includes(q) || !loose(q) || q.anc) continue;
        const Q = P(q), l = rotYv([x - Q[0], 0, z - Q[2]], -ryOf(q));
        if (Math.abs(l[0]) > q.hx || Math.abs(l[2]) > q.hz) continue;
        const C = AFF(q).container;
        if (C && contOpen(q) && self && self.hx <= C.inner[0] + 1e-3 && self.hz <= C.inner[2] + 1e-3 && Math.abs(l[0]) <= C.inner[0] && Math.abs(l[2]) <= C.inner[2]) {   // through the opening (if it fits): its floor inside
          const fl = Q[1] + (Number.isFinite(+C.floor_y) ? +C.floor_y : -q.hH * 0.8);
          const occ = E.P.filter(r => r !== self && r.sup && r.sup.id === q.id && r.sup.inside).reduce((m, r) => Math.max(m, P(r)[1] + topOff(r)), fl);
          if (occ <= y + lv && occ > best.top - 1e-6) best = {kind: 'prop', id: q.id, top: occ, inside: true};
          continue;
        }
        const tp = Q[1] + topOff(q);
        if (tp <= y + lv && tp > best.top) best = {kind: 'prop', id: q.id, top: tp};
      }
      return best;
    }
    // come to rest on support s at the physical pose w: the relation is kept from now on (it moves with its support)
    function restOn(p, s, w) {
      p.fly = null; p.flying = false; p.held = null; p.free = true; p.bal = null; p.anc = null; p.att.l = 0; p.att.r = 0; p.hh = p.worn ? p.hh : 0;
      if (w) setP(p, w);
      const W = P(p), home = len3(sub3(p.pos, p.home)) < 0.012;
      if (s && s.kind === 'prop') { const q = byId(s.id), Q = P(q); p.sup = {id: q.id, rel: rotYv(sub3(W, Q), -ryOf(q)), rr: ryOf(p) - ryOf(q), inside: !!s.inside}; p.place = supPlace(q); }
      else { p.sup = null; p.place = s ? s.id : 'floor'; }
      p.where = home && (p.sup ? p.sup.id === p.homeOn && !p.sup.inside : !p.homeOn) ? 'home' : 'placed';
      if (p.where === 'home') { if (!p.sup) p.place = p.homePlace; if (Math.abs(wrap(p.rot[1])) < 2e-3 && Math.abs(p.rot[0]) + Math.abs(p.rot[2]) < 2e-3) p.rot = [0, 0, 0]; }
      p.wpos = W.slice(); p.wvel = [0, 0, 0];
    }
    // after an engine step let a prop go at a rest spot: what is it resting on? (a prop top under it wins over the place)
    function settleSupport(p) {
      const W = P(p), s = supportAt(W[0], W[2], W[1] - p.hH + 0.012, p);
      if (s.kind === 'prop' && Math.abs(s.top - (W[1] - p.hH)) < 0.02) { const q = byId(s.id), Q = P(q); p.sup = {id: q.id, rel: rotYv(sub3(W, Q), -ryOf(q)), rr: ryOf(p) - ryOf(q), inside: !!s.inside}; p.place = supPlace(q); if (p.where === 'home' && !(q.id === p.homeOn && !s.inside)) p.where = 'placed'; }
      else { p.sup = null; if (p.where === 'home' && p.homeOn) p.where = 'placed'; }
      p.wpos = W.slice();
    }
    function supName(p) {   // what it rests on, in words
      if (p.sup) { const q = byId(p.sup.id); return q ? (p.sup.inside ? `in the ${q.name}` : `on the ${q.name}`) : 'on something'; }
      return `on ${placeName(p.place || p.homePlace || 'floor')}`;
    }
    // ---- flights: analytic ballistic segments (p0, v0, t0) under GRAV, contact with the highest support under it, bounce
    // (afford.bouncy, else by mass), a short settle, then rest. The segment starts where it is drawn (continuous)
    const bounceOf = p => Number.isFinite(+AFF(p).bouncy) ? clamp(+AFF(p).bouncy, 0, 0.95) : p.def.mass === 'heavy' ? 0.08 : p.def.mass === 'medium' ? 0.2 : 0.3;
    function startFlight(p, w, vel, why, opt = {}) {
      p.fly = {p0: w.slice(), v0: vel.slice(), t0: t(), tPrev: 0, why: why || 'let go', n: 0, spin: opt.spin || [0, 0, 0], r0: p.rot.slice(), lastB: w[1] - p.hH, target: opt.target || null, src: opt.src || 'auto', flipOK: !!opt.flipOK, done: opt.done || null};
      p.flying = true; p.where = 'air'; p.place = null; p.sup = null; p.bal = null; p.anc = null; setP(p, w); p.wpos = w.slice();
    }
    function flyFrame(p) {
      const F = p.fly, now = t();
      if (F.settle) {   // the last few mm: ease into the rest pose (no float, no snap)
        const S = F.settle, u = clamp((now - S.t0) / S.dur, 0, 1), m = mj(u);
        setP(p, lerp3(S.from, S.to, m)); p.rot = [lerp(S.r0[0], S.r1[0], m), lerp(S.r0[1], S.r1[1], m), lerp(S.r0[2], S.r1[2], m)];
        if (u >= 1) { const s = S.sup, fly = p.fly; restOn(p, s, S.to); landed(p, fly, s); }
        return;
      }
      const tt = now - F.t0, ta = F.tPrev || 0, n = Math.max(1, Math.ceil((tt - ta) * 240));
      for (let i = 1; i <= n; i++) {
        const q = ta + (tt - ta) * i / n, w = [F.p0[0] + F.v0[0] * q, F.p0[1] + F.v0[1] * q - 0.5 * GRAV * q * q, F.p0[2] + F.v0[2] * q];
        const vy = F.v0[1] - GRAV * q, bottom = w[1] - p.hH;
        // the sides of resting props: pushed back out, that velocity component reversed and damped (a new segment)
        for (const o of E.P) {
          if (o === p || !loose(o) || o.anc || F.inside === o.id) continue;
          const O = P(o), l = rotYv(sub3(w, O), -ryOf(o)), ox = o.hx + p.hx * 0.7, oz = o.hz + p.hz * 0.7, otop = O[1] + topOff(o), obot = O[1] - o.hH;
          if (Math.abs(l[0]) < ox && Math.abs(l[2]) < oz && bottom < otop - 0.01 && w[1] + p.hH > obot && F.lastB < otop - 0.01) {
            const vl = rotYv([F.v0[0], 0, F.v0[2]], -ryOf(o));
            if (ox - Math.abs(l[0]) < oz - Math.abs(l[2])) { l[0] = Math.sign(l[0] || 1) * (ox + 1e-3); vl[0] = -0.35 * vl[0]; } else { l[2] = Math.sign(l[2] || 1) * (oz + 1e-3); vl[2] = -0.35 * vl[2]; }
            const np = add3(O, rotYv([l[0], 0, l[2]], ryOf(o))), vn = rotYv(vl, ryOf(o));
            F.p0 = [np[0], w[1], np[2]]; F.v0 = [vn[0], vy, vn[2]]; F.t0 += q; F.tPrev = 0; F.n++; F.r0 = p.rot.slice();
            setP(p, F.p0); return;
          }
        }
        if (F.inside) {   // inside a container: its walls keep it in
          const o = byId(F.inside), C = o && AFF(o).container;
          if (C) { const O = P(o), l = rotYv(sub3(w, O), -ryOf(o)), mx = Math.max(0, C.inner[0] - p.hx * 0.8), mz = Math.max(0, C.inner[2] - p.hz * 0.8);
            if (Math.abs(l[0]) > mx || Math.abs(l[2]) > mz) { const lc = [clamp(l[0], -mx, mx), 0, clamp(l[2], -mz, mz)], np = add3(O, rotYv(lc, ryOf(o))); w[0] = np[0]; w[2] = np[2]; F.v0[0] *= 0.3; F.v0[2] *= 0.3; F.p0[0] = w[0] - F.v0[0] * q; F.p0[2] = w[2] - F.v0[2] * q; } }
        }
        const s = supportAt(w[0], w[2], Math.max(F.lastB, bottom) + 0.004, p);
        if (s.kind === 'prop' && s.inside && !F.inside) F.inside = s.id;
        if (vy < 0 && bottom <= s.top) {
          const y = s.top + p.hH, e = bounceOf(p), vyi = -vy, hv = [F.v0[0], 0, F.v0[2]];
          F.n++; if (!F.firstHit) F.firstHit = s;
          if (vyi * e > 0.45 && F.n < 6 && !s.inside) {   // bounce: a new segment from the contact point
            F.p0 = [w[0], y, w[2]]; F.v0 = [hv[0] * 0.65, vyi * e, hv[2] * 0.65]; F.t0 += q; F.tPrev = 0; F.lastB = y - p.hH; F.spin = F.spin.map(v => v * 0.6); F.r0 = p.rot.slice();
            setP(p, F.p0); return;
          }
          // settle: a little roll / slide the way it was going (never off an edge), clear of its neighbours, then rest
          const roll = (+AFF(p).bouncy || 0) > 0.5 ? 0.08 : 0.015, hs = Math.hypot(hv[0], hv[2]);
          let to = [w[0] + (hs > 1e-3 ? hv[0] / hs * Math.min(roll, hs * 0.06) : 0), y, w[2] + (hs > 1e-3 ? hv[2] / hs * Math.min(roll, hs * 0.06) : 0)];
          const s2 = s.inside ? s : supportAt(to[0], to[2], y - p.hH + 0.004, p);
          if (s2.id !== s.id || Math.abs(s2.top - s.top) > 0.004) to = [w[0], y, w[2]];
          if (!s.inside) to = clearSpot(p, to, s);
          const r1 = F.flipOK ? [Math.round(p.rot[0] * 2) / 2, p.rot[1], Math.round(p.rot[2] * 2) / 2] : [Math.round(p.rot[0]), p.rot[1], Math.round(p.rot[2])];
          F.settle = {t0: now, dur: clamp(0.2 + 1.5 * Math.hypot(to[0] - w[0], to[2] - w[2]), 0.2, 0.5), from: [w[0], y, w[2]], to, r0: p.rot.slice(), r1, sup: s};
          setP(p, [w[0], y, w[2]]);
          return;
        }
        F.lastB = bottom; setP(p, w);
        p.rot = [F.r0[0] + F.spin[0] * q, F.r0[1] + F.spin[1] * q, F.r0[2] + F.spin[2] * q];
      }
      F.tPrev = tt;
    }
    // do two footprints (rectangles hx x hz turned by their yaws, turns) overlap? 2-D separating axes
    function footOverlap(a, ya, A, b, yb, B) {
      const ax = [rotYv([1, 0, 0], ya), rotYv([0, 0, 1], ya)], bx = [rotYv([1, 0, 0], yb), rotYv([0, 0, 1], yb)], d = sub3(b, a);
      const ea = [A.hx, A.hz], eb = [B.hx, B.hz], dot = (u, v) => u[0] * v[0] + u[2] * v[2];
      for (const n of [...ax, ...bx]) {
        const ra = ea[0] * Math.abs(dot(ax[0], n)) + ea[1] * Math.abs(dot(ax[1], n)), rb = eb[0] * Math.abs(dot(bx[0], n)) + eb[1] * Math.abs(dot(bx[1], n));
        if (Math.abs(dot(d, n)) >= ra + rb - 1e-4) return false;
      }
      return true;
    }
    // a rest spot clear of other resting props at the same level (slid aside along the shortest way)
    function clearSpot(p, w, s) {
      let q = w.slice();
      for (let it = 0; it < 6; it++) {
        // (a neighbour still settling counts at the spot it is settling to — two things landing together)
        const at = o2 => o2.fly && o2.fly.settle ? o2.fly.settle.to : P(o2);
        const o = E.P.find(o2 => o2 !== p && (loose(o2) || (o2.fly && o2.fly.settle)) && !o2.anc && !(s && s.kind === 'prop' && s.id === o2.id) && !stackAbove(p).includes(o2) && (() => {
          const O = at(o2); return Math.abs(O[1] - q[1]) < o2.hH + p.hH - 0.004 && footOverlap(q, ryOf(p), p, O, ryOf(o2), o2); })());
        if (!o) break;
        const O = at(o), d = sub3(flat(q), flat(O)), L = Math.hypot(d[0], d[2]) || 1, need = Math.hypot(o.hx, o.hz) + Math.hypot(p.hx, p.hz) * 0.8 + 0.004;
        q = [O[0] + d[0] / L * need, q[1], O[2] + d[2] / L * need];
        const s2 = supportAt(q[0], q[2], q[1] - p.hH + 0.004, p); q[1] = s2.top + p.hH;
      }
      return q;
    }
    function landed(p, F, s) {
      const q0 = F.firstHit && F.firstHit.id !== s.id ? F.firstHit : null;
      const first = q0 ? `bounced off ${q0.kind === 'prop' ? 'the ' + byId(q0.id).name : placeName(q0.id)} and ` : F.n > 1 ? 'bounced and ' : '';
      const T = F.target, hit = !T || !T.id || s.id === T.id || chainOf(p).some(q => q.id === T.id);
      emit({kind: 'object_event', name: s.inside ? 'landed_in' : 'landed', phase: 'end', src: F.src || 'auto', args: {object: p.id, where: s.id, inside: !!s.inside, hit},
        did: `the ${p.name} ${first}landed ${supName(p)}${hit ? '' : ` (it missed ${T.name || 'what it was aimed at'})`}`});
      if (F.done) F.done(p, s, hit);
    }
    // a support that is gone (tipped, flying): what rode on it falls from where it is, with the support's velocity
    function shed(p, why) {
      const q = p.sup ? byId(p.sup.id) : null, v = q && q.wvel ? q.wvel.slice() : [0, 0, 0];
      p.sup = null;
      emit({kind: 'object_event', name: 'fell_off', phase: 'end', src: 'auto', args: {object: p.id, from: q ? q.id : null}, did: `the ${p.name} fell off the ${q ? q.name : 'thing it was on'}${why ? ' (' + why + ')' : ''}`});
      startFlight(p, P(p), [v[0], clamp(v[1], -0.5, 1), v[2]], 'fell', {spin: [0.5 * (rnd() - 0.5), 0, 0.5 * (rnd() - 0.5)]});
    }
    // ---- tracked body anchors (character-local points → world every frame)
    const ANCHORS = ['foot_l', 'foot_r', 'back', 'lap', 'shoulder_l', 'shoulder_r'];
    function anchorWorld(name, p) {
      const B = E.body, hh = p ? p.hH : 0.03;
      if (/^foot_[lr]$/.test(name)) { if (!B.legs) return null; const s = name.slice(-1), F = (E.footNow && E.footNow[s]) || toWorld(footRest(s)); return add3(F, add3([0, E.footTop + hh, 0], rotYv([0, 0, 0.03], rootNow().yaw))); }
      if (name === 'back') return toWorld([0, B.lo[1] + 0.5 * (B.hi[1] - B.lo[1]), B.lo[2] - (p ? p.hz : 0.03) - 0.004]);
      if (name === 'lap') { if (!E.sitOn) return null; const L = B.legs; return toWorld([0, (L ? L.hipY : 0.3) + 0.03 + hh, L ? 0.1 : 0.15]); }
      if (/^shoulder_[lr]$/.test(name)) { const sg = name.slice(-1) === 'l' ? -1 : 1, j = B.raw && B.raw.joints_scene && B.raw.joints_scene['shoulder_' + name.slice(-1)], H = +(B.raw && B.raw.H) || 2.3; return toWorld(j ? [j[0] / H, j[1] / H + 0.04 + hh, j[2] / H] : [sg * 0.17, B.lo[1] + 0.47 * (B.hi[1] - B.lo[1]) + hh, 0]); }
      return null;
    }
    const anchorWord = n => ({foot_l: 'its right foot', foot_r: 'its left foot', back: 'its back', lap: 'its lap', shoulder_l: 'its right shoulder', shoulder_r: 'its left shoulder', head: 'its head'})[n] || n;
    // ---- the head: worn (afford.wear.offset) or balanced (on top of the head / of what is worn on its crown)
    function wearOff(p) { const W = AFF(p).wear; return W && v3(W.offset) || [0, 0, 0]; }
    function headClash(p, off, bal) {   // a worn / balanced thing whose region overlaps (0.45 = how much of its size is solid there)
      // (bal: p is balanced at off — resting ON TOP of a worn / balanced thing is no clash, balanceOff stacks it there)
      for (const q of E.P) { if (q === p || !(q.worn || q.bal)) continue; const o2 = q.bal ? q.bal.off : wearOff(q); if (bal && off[1] - p.hH >= o2[1] + topOff(q) - 0.02) continue; if (len3(sub3(off, o2)) < 0.45 * (p.size + q.size)) return q; }
      return null;
    }
    function balanceOff(p) {
      let y = E.headTop;
      for (const q of E.P) if ((q.worn || q.bal) && q !== p) { const o = q.bal ? q.bal.off : wearOff(q); if (Math.hypot(o[0], o[2]) < 0.12 && o[1] > 0.05) y = Math.max(y, o[1] + topOff(q)); }
      return [0, y + p.hH, 0];
    }
    function headPose() { return {pos: toWorld(E.body.head), yaw: rootNow().yaw}; }
    // the centre of a held prop from the tracked grasp point(s) (+ the keyframe layer's per-frame offset), blended with its
    // free pose by the attach weight as the renderer does (continuous through a grasp / a release)
    // (the renderer's held frame faces the camera: the held prop's world yaw is its home turn + hrot[1]; its grip anchor sits
    // off the palm along the palm normal by its extent that way — engine/props.glsl propHeldIn)
    const heldYawW = p => p.homeRot[1] + p.hrot[1];
    function heldOne(p, s, yawW, gaL) {   // (yawW / gaL given: where p WOULD sit in paw s held that way — the catch test)
      const g = E.pawGW[s]; if (!g) return null;
      const yaw = yawW ?? heldYawW(p), ga = rotYv(gaL || gripLocal(p), yaw), n = pawN(s);
      let c = sub3(g, ga);
      if (n) { const ext = Math.max(Math.abs(n[0]) * p.hx + Math.abs(n[1]) * p.hH + Math.abs(n[2]) * p.hz - (n[0] * ga[0] + n[1] * ga[1] + n[2] * ga[2]), 0.04 * p.size) - (p.def.grip === 'pinch' ? 0 : 0.03 * p.size); c = add3(c, mul3(n, ext)); }
      const k = E.kfh[s]; if (k) c = add3(c, [k.x, k.y, k.z]);
      if (yawW === undefined && p.hoff && p.held) c = add3(c, p.hoff);   // (its held offset — the renderer adds the columns' offs in the camera-facing held frame)
      return c;
    }
    function heldWorld(p) {
      let c = null;
      if (p.held === 'both') { const a = heldOne(p, 'l'), b = heldOne(p, 'r'); c = a && b ? lerp3(a, b, 0.5) : a || b; }
      else if (p.held) c = heldOne(p, p.held);
      if (!c) return null;
      const w = clamp(p.att.l + p.att.r, 0, 1);
      return w >= 1 ? c : lerp3(P(p), c, w);
    }
    // per frame, in apply() (after the chain step, before the prop columns): eases, flights, the head, catches
    function objFrame(dt) {
      if (!E.P.length || !E.kfh) return;
      const now = t(); E.lastDt = dt;
      for (const p of E.P) if (p.attE) {   // attach-weight eases (kfEvent release / catch / grab / attach): never a jump
        const A = p.attE, u = clamp((now - A.t0) / A.dur, 0, 1), m = mj(u);
        for (const sd of ['l', 'r']) p.att[sd] = lerp(A.from[sd], A.to[sd], m);
        if (A.hh !== undefined) p.hh = lerp(A.hh0, A.hh, m);
        if (A.off) p.hoff = lerp3(A.off0, A.off, m);
        if (A.each) A.each(m, u);
        if (u >= 1) { p.attE = null; if (A.end) A.end(); }
      }
      for (const p of E.P) if (p.fly) flyFrame(p);
      for (const p of E.P) if (p.bal) {   // balanced on the head: a small live wobble; it falls if the head tips over
        const r = rootNow(), tip = Math.max(Math.abs(wrap(r.pitch)), Math.abs(wrap(r.roll)));
        if (tip > 0.12 && !p.attE) {
          const hp = (p.wpos || P(p)).slice(); p.bal = null; p.hh = 0; p.hoff = [0, 0, 0]; p.hrot = [0, 0, 0]; p.rot = [0, rootNow().yaw, 0];
          emit({kind: 'object_event', name: 'fell_off', phase: 'end', src: 'auto', args: {object: p.id, from: 'head'}, did: `the ${p.name} fell off its head`});
          startFlight(p, hp, [0, 0.3, 0.25], 'fell'); continue;
        }
        if (!p.attE) { const ph = now - p.bal.t0; p.hoff = add3(p.bal.off, [0.004 * Math.sin(ph * 2.3), 0, 0.003 * Math.sin(ph * 1.7 + 1)]); p.hrot = [0.012 * Math.sin(ph * 1.7 + 1), 0, -0.015 * Math.sin(ph * 2.3)]; }
      }
      for (const sd of ['l', 'r']) {   // an armed paw catches a flying prop that comes within reach of its grasp point
        const A = E.catchArm[sd]; if (!A) continue;
        if (now > A.until) { E.catchArm[sd] = 0; emit({kind: 'object_event', name: 'missed', phase: 'end', src: A.src || 'auto', args: {side: sideWord(sd)}, did: `its ${sideWord(sd)} paw was ready to catch, but nothing came within reach`}); continue; }
        const g = E.pawGW[sd]; if (heldBy(sd) || !g) continue;
        // (measured to where it WOULD sit if this paw held it — its centre off the grasp point by its grip — or to the grasp
        // point itself, whichever is nearer: a straight-up toss that comes back to where it left the paw is caught)
        const dC = p => { const h = heldOne(p, sd, p.homeRot[1] + p.rot[1], gaOf(p)), Q = P(p); return Math.min(len3(sub3(Q, g)), h ? len3(sub3(Q, h)) : 1e9); };
        // (only while it is coming TOWARD the paw: a thing this paw has just let go of is never caught again on its way out)
        const inbound = p => { const F = p.fly, q = now - F.t0, v = [F.v0[0], F.v0[1] - GRAV * q, F.v0[2]], d = sub3(P(p), g), gv = E.pawGV[sd] || [0, 0, 0]; return (d[0] * (v[0] - gv[0]) + d[1] * (v[1] - gv[1]) + d[2] * (v[2] - gv[2])) < 0; };
        const fresh = p => p.fly.from === sd && now - p.fly.tRel < 0.25;
        const c = E.P.filter(p => p.fly && !p.fly.settle && !fresh(p) && dC(p) < 0.06 + p.size && inbound(p)).sort((a, b) => dC(a) - dC(b))[0];
        if (c) { E.catchArm[sd] = 0; catchNow(c, sd, A.src); }
      }
    }
    // after the arm follow-through (post) and after the keyframe layer (postKf): where things really are this frame —
    // held / worn props from the drawn paws and head, then what rides on them, then the body anchors; their columns rewritten
    function trackFrame(o) {
      if (!E.kfh) return;
      const dt = E.lastDt || 1 / 30;
      const upd = (p, w) => { if (!w) return; if (p.wpos && dt > 0) p.wvel = lerp3(p.wvel || [0, 0, 0], mul3(sub3(w, p.wpos), 1 / dt), 0.5); p.wpos = w.slice(); };
      for (const p of E.P) {   // (blended as the renderer blends: free -> the paw(s) -> the head, by hl + hr + hh — a hand-off
        // between a paw and the head (put on / take off) is one continuous path, never a sweep through its old rest spot)
        if (!p.held && !p.worn && !p.bal) continue;
        const wa = clamp(p.att.l + p.att.r, 0, 1), wh = clamp(p.hh || 0, 0, 1), w = clamp(wa + wh, 0, 1);
        let cH = null;
        if (p.held && wa > 5e-4) { const a = p.held === 'both' ? heldOne(p, 'l') : heldOne(p, p.held), b = p.held === 'both' ? heldOne(p, 'r') : null; cH = a && b ? lerp3(a, b, 0.5) : a || b; }
        if (wh > 5e-4) { const H = headPose(), hw = add3(H.pos, rotYv(p.hoff || [0, 0, 0], H.yaw)); cH = cH ? lerp3(cH, hw, wh / (wa + wh)) : hw; }
        upd(p, !cH ? P(p) : w >= 1 ? cH : lerp3(P(p), cH, w));
      }
      const done = new Set(), touched = [];
      const ride = p => {
        if (done.has(p)) return; done.add(p);
        if (!p.sup) return;
        if (p.held || p.worn || p.bal || p.anc || p.fly || p.flying) { p.sup = null; return; }
        const q = byId(p.sup.id); if (!q) { p.sup = null; return; }
        ride(q);
        const W = worldOf(q);
        const tipped = W.tilt > 0.12 && !(p.sup.inside && !contOpen(q));   // (shut in a closed box it stays in, however it is turned)
        if (q.fly || q.flying || tipped) { shed(p, tipped ? `the ${q.name} tipped over` : `the ${q.name} went flying`); touched.push(p); return; }
        if (!p.sup.inside && hasArt(q) && (q.s || 0) > 0.15) { shed(p, `the ${q.name}'s lid opened under it`); touched.push(p); return; }
        setP(p, add3(W.pos, rotYv(p.sup.rel, W.yaw))); p.rot[1] = W.yaw + p.sup.rr - p.homeRot[1]; touched.push(p);
      };
      for (const p of E.P) ride(p);
      for (const p of E.P) if (p.anc && !p.held) {
        const A = anchorWorld(p.anc.name, p);
        if (!A) { const w = P(p); p.anc = null; startFlight(p, w, [0, 0, 0], 'fell'); continue; }
        const e = p.anc.ease, k = e ? mj(clamp((t() - e.t0) / e.dur, 0, 1)) : 1;
        setP(p, k < 1 ? lerp3(e.from, A, k) : A); if (k >= 1) p.anc.ease = null;
        p.rot[1] = rootNow().yaw + (p.anc.rr || 0) - p.homeRot[1]; touched.push(p);
      }
      for (const p of E.P) if (!p.held && !p.worn && !p.bal) {   // (at rest and unmoved: nothing to update — no allocation)
        const w = P(p), q = p.wpos, v = p.wvel;
        if (q && v && w[0] === q[0] && w[1] === q[1] && w[2] === q[2] && v[0] === 0 && v[1] === 0 && v[2] === 0) continue;
        if (q && v && w[0] === q[0] && w[1] === q[1] && w[2] === q[2] && Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]) < 1e-6) { p.wvel = [0, 0, 0]; continue; }
        upd(p, w);
      }
      if (o) for (const p of touched) writeCols(o, p);
    }
    function writeCols(o, p) {   // the prop's columns (same formula as apply's prop loop)
      const w = clamp(p.att.l + p.att.r + (p.hh || 0), 0, 1), fo = freeOff(p), k = 'prop' + p.k + '_';
      if (!(k + 'x' in o)) return;
      const hx = p.hoff; o[k + 'x'] = (1 - w) * fo[0] + w * hx[0]; o[k + 'y'] = (1 - w) * fo[1] + w * hx[1]; o[k + 'z'] = (1 - w) * fo[2] + w * hx[2];
      const rf = p.rot, rh = p.hrot;
      o[k + 'rx'] = wrapH(rf[0] + wrap(rh[0] - rf[0]) * w); o[k + 'ry'] = wrapH(rf[1] + wrap(rh[1] - rf[1]) * w); o[k + 'rz'] = wrapH(rf[2] + wrap(rh[2] - rf[2]) * w);
    }
    // a paw closes on a flying prop: the free position glides onto the paw while the attach weight eases in
    function catchNow(p, sd, src) {
      const F = p.fly, q0 = t() - F.t0, P0 = [F.p0[0] + F.v0[0] * q0, F.p0[1] + F.v0[1] * q0 - 0.5 * GRAV * q0 * q0, F.p0[2] + F.v0[2] * q0], V0 = [F.v0[0], F.v0[1] - GRAV * q0, F.v0[2]];
      p.fly = null; p.flying = false; p.held = sd; p.where = 'held'; p.place = null; p.sup = null; p.free = false;
      p.hrot = p.rot.slice(); p.grip = gaOf(p).slice(); p.hoff = [0, 0, 0];   // (the held frame faces the camera: same world yaw)
      // the paw gives with it: the offset from the held pose (and the velocity relative to the paw) decays as a critically
      // damped spring — position and velocity continuous at the catch, a give of about |v_rel| / (k e) (3 cm at 3.5 H/s)
      const t0 = t(), D = 0.2, k = 40, h0 = heldOne(p, sd) || P0, e0 = sub3(P0, h0), vr = sub3(V0, E.pawGV[sd] || [0, 0, 0]), b = add3(vr, mul3(e0, k));
      p.attE = {t0, dur: D, from: {l: p.att.l, r: p.att.r}, to: {l: sd === 'l' ? 1 : 0, r: sd === 'r' ? 1 : 0},
        each(m, u) { const q = u * D, c = heldOne(p, sd); if (!c) return; setP(p, add3(c, mul3(add3(e0, mul3(b, q)), Math.exp(-k * q) * (1 - u * u * u)))); },
        end() { p.att[sd] = 1; p.att[otherSide(sd)] = 0; }};
      setP(p, P0);
      E.paw[sd].open = graspValues(p, sd).open;
      emit({kind: 'object_event', name: 'caught', phase: 'end', src: src || 'auto', args: {object: p.id, side: sideWord(sd)}, did: `caught the ${p.name} in its ${sideWord(sd)} paw`});
    }
    function foldKf(sd) {   // the keyframe layer's held offset becomes part of the held pose (no pop)
      if (!sd || sd === 'both' || !E.kfh) return; const k = E.kfh[sd]; if (!k) return; const p = heldBy(sd); E.kfh[sd] = null; if (!p || p.held === 'both') return;
      p.hoff = add3(p.hoff || [0, 0, 0], [k.x, k.y, k.z]); p.hrot = [p.hrot[0] + k.rx, p.hrot[1] + k.ry, p.hrot[2] + k.rz];
    }
    // let go of what paw sd holds NOW, with its measured world velocity (+ impulse, H/s world): it flies, lands, bounces, rests
    function letGoNow(sd, impulse, src, opt = {}) {
      const p = heldBy(sd); if (!p) return null;
      foldKf(sd);
      const w = (heldWorld(p) || p.wpos || P(p)).slice(), v = add3(p.wvel || [0, 0, 0], v3(impulse) || [0, 0, 0]);
      const rotW = p.hrot.slice();
      p.held = null; p.grip = null; p.free = true; p.rot = rotW; p.hoff = [0, 0, 0];
      p.attE = {t0: t(), dur: 0.14, from: {l: p.att.l, r: p.att.r}, to: {l: 0, r: 0}};
      startFlight(p, w, v, opt.why || 'let go', {spin: opt.spin, target: opt.target, src, flipOK: opt.flipOK, done: opt.done});
      if (p.fly) { p.fly.from = sd; p.fly.tRel = t(); }   // (the paw that let go: it does not catch it again on its way out)
      return p;
    }
    // ---- the planner hooks (docs/OBJECTS.md §5)
    const pawOf = side => { if (side === 'l' || side === 'r') return side; const s = sideIn(side); return s === 'both' ? null : s; };
    function kfHeld(side, off) {
      const s = pawOf(side); if (!s || !E.kfh) return {ok: false, reason: 'which paw?'};
      if (!off) { foldKf(s); return {ok: true}; }
      E.kfh[s] = {x: +off.x || 0, y: +off.y || 0, z: +off.z || 0, rx: +off.rx || 0, ry: +off.ry || 0, rz: +off.rz || 0}; return {ok: true};
    }
    function kfEvent(ev) {
      ev = ev || {}; const s = pawOf(ev.side), src = ev.src || 'kf';
      if (!E.kfh) return {ok: false, reason: 'no objects here'};
      if (!s) return {ok: false, reason: 'which paw?'};
      if (ev.type === 'release') {
        const p = heldBy(s); if (!p) return {ok: false, reason: `its ${sideWord(s)} paw is not holding anything`};
        letGoNow(s, ev.impulse, src, {flipOK: true});
        const sp = len3(p.fly.v0), thrown = sp > 0.6;
        emit({kind: 'object_event', name: thrown ? 'thrown' : 'let_go', phase: 'end', src, args: {object: p.id, side: sideWord(s), speed: +sp.toFixed(2)}, did: thrown ? `threw the ${p.name}` : `let go of the ${p.name}`});
        return {ok: true, did: `${thrown ? 'threw' : 'let go of'} the ${p.name} (it is in the air)`, object: p.id};
      }
      if (ev.type === 'catch') {
        if (heldBy(s)) return {ok: false, reason: `its ${sideWord(s)} paw is holding the ${heldBy(s).name}`};
        E.catchArm[s] = {until: t() + (Number.isFinite(+ev.window) ? clamp(+ev.window, 0.05, 3) : 0.5), src};
        return {ok: true, did: `its ${sideWord(s)} paw is ready to catch`};
      }
      if (ev.type === 'grab') {
        if (heldBy(s)) return {ok: false, reason: `its ${sideWord(s)} paw is already holding the ${heldBy(s).name}`};
        const g = E.pawGW[s]; if (!g) return {ok: false, reason: 'that paw is not tracked yet'};
        const c = E.P.filter(p => !isFurn(p) && !p.held && !p.worn && !p.bal && p.def.grip !== 'two_hand' && len3(sub3(P(p), g)) < 0.05 + p.size).sort((a, b) => len3(sub3(P(a), g)) - len3(sub3(P(b), g)))[0];
        if (!c) return {ok: false, reason: 'nothing is within reach of that paw'};
        if (c.fly && !c.fly.settle) { catchNow(c, s, src); return {ok: true, did: `caught the ${c.name}`, object: c.id}; }
        if (c.fly) { const F = c.fly; restOn(c, F.settle.sup, F.settle.to); }
        c.lastRest = {pos: c.pos.slice(), rot: c.rot.slice(), place: c.place || 'floor', home: c.where === 'home'};
        const W0 = P(c);
        c.held = s; c.where = 'held'; c.place = null; c.sup = null; c.anc = null; c.free = false; c.hrot = c.rot.slice(); c.grip = gaOf(c).slice(); c.hoff = [0, 0, 0];
        c.attE = {t0: t(), dur: 0.2, from: {l: c.att.l, r: c.att.r}, to: {l: s === 'l' ? 1 : 0, r: s === 'r' ? 1 : 0}, each(m) { const h1 = heldOne(c, s); if (h1) setP(c, lerp3(W0, h1, m)); }};
        emit({kind: 'object_event', name: 'picked_up', phase: 'end', src, args: {object: c.id, side: sideWord(s)}, did: `picked up the ${c.name} with its ${sideWord(s)} paw`});
        return {ok: true, did: `grabbed the ${c.name}`, object: c.id};
      }
      if (ev.type === 'attach') {
        const p = heldBy(s); if (!p) return {ok: false, reason: `its ${sideWord(s)} paw is not holding anything`};
        if (p.held === 'both') return {ok: false, reason: `the ${p.name} is held in both paws`};
        const to = String(ev.to || '').toLowerCase().replace(/[\s-]+/g, '_'), c = heldWorld(p) || p.wpos;
        if (to === 'head') {
          const wear = !!AFF(p).wear, off = wear ? wearOff(p) : balanceOff(p), H = headPose(), Pt = add3(H.pos, rotYv(off, H.yaw));
          if (len3(sub3(c, Pt)) > 0.12 + p.size) return {ok: false, reason: `the ${p.name} is not up at its head`};
          const q = headClash(p, off, !wear); if (q) return {ok: false, reason: `the ${q.name} is in the way on its head`};
          headAttach(p, wear, off, 0.3, src); return {ok: true, did: wear ? `put on the ${p.name}` : `balanced the ${p.name} on its head`};
        }
        if (ANCHORS.includes(to)) {
          const A = anchorWorld(to, p); if (!A) return {ok: false, reason: `there is no ${to.replace('_', ' ')} to put it on now`};
          if (len3(sub3(c, A)) > 0.08 + p.size) return {ok: false, reason: `the ${p.name} is not at ${anchorWord(to)}`};
          anchorAttach(p, to, 0.25, src); return {ok: true, did: `put the ${p.name} on ${anchorWord(to)}`};
        }
        const q = byId(to) || (E.P.find(x => propNames(x).has(to)) || null), pl = !q ? E.places.find(x => x.id === to || x.aliases.includes(to.replace(/_/g, ' '))) : null;
        if (q && q !== p) {
          const R = restPose(p, {rel: isCont(q) && contOpen(q) ? 'in' : 'on', prop: q});
          if (R.err) return {ok: false, reason: R.err};
          if (len3(sub3(c, R.pos)) > 0.08 + p.size) return {ok: false, reason: `the ${p.name} is not at the ${q.name}`};
          easeDown(p, s, R, src); return {ok: true, did: `put the ${p.name} ${R.sup.inside ? 'in' : 'on'} the ${q.name}`};
        }
        if (pl) {
          const s2 = supportAt(c[0], c[2], c[1], p);
          if (!(s2.kind === 'place' && s2.id === pl.id)) return {ok: false, reason: `the ${p.name} is not over ${pl.name}`};
          const R = {pos: [c[0], s2.top + p.hH, c[2]], sup: s2}; if (c[1] - R.pos[1] > 0.08 + p.size) return {ok: false, reason: `the ${p.name} is too high above ${pl.name}`};
          easeDown(p, s, R, src); return {ok: true, did: `put the ${p.name} down on ${pl.name}`};
        }
        return {ok: false, reason: `there is no ${to.replace(/_/g, ' ')} here`};
      }
      return {ok: false, reason: `unknown event "${ev.type}"`};
    }
    // eased hand-offs (attach weights never jump; the pose glides from where it is drawn now)
    function headAttach(p, wear, off, dur, src, done) {
      const sd = p.held === 'both' ? 'l' : p.held; foldKf(sd);
      const hr0 = p.hrot.slice(), r1 = wear ? (v3(AFF(p).wear.rot) || [0, 0, 0]) : [0, 0, 0];
      p.attE = {t0: t(), dur, from: {l: p.att.l, r: p.att.r}, to: {l: 0, r: 0}, hh0: p.hh || 0, hh: 1, off0: (p.hoff || [0, 0, 0]).slice(), off,
        each(m) { p.hrot = hr0.map((v, i) => v + (r1[i] - v) * m); },
        end() { p.held = null; p.grip = null; p.att.l = 0; p.att.r = 0; p.hh = 1; p.hoff = off.slice(); p.hrot = r1.slice(); p.rot = [0, 0, 0];
          if (wear) { p.worn = true; p.where = 'worn'; } else { p.bal = {off: off.slice(), t0: t()}; p.where = 'head'; }
          emit({kind: 'object_event', name: wear ? 'put_on' : 'balanced', phase: 'end', src: src || 'auto', args: {object: p.id}, did: wear ? `put on the ${p.name}` : `balanced the ${p.name} on its head`});
          if (done) done(); }};
    }
    function anchorAttach(p, name, dur, src) {
      const sd = p.held === 'both' ? 'l' : p.held; foldKf(sd);
      const from = (heldWorld(p) || p.wpos).slice(), rotW = p.hrot.slice();
      p.held = null; p.grip = null; p.free = true; p.rot = [0, rotW[1], 0]; p.hoff = [0, 0, 0]; setP(p, from);
      p.anc = {name, rr: wrapH(rotW[1] - rootNow().yaw + p.homeRot[1]), ease: {t0: t(), dur, from}};
      p.attE = {t0: t(), dur, from: {l: p.att.l, r: p.att.r}, to: {l: 0, r: 0}};
      p.where = 'anchor'; p.place = null; p.sup = null;
      emit({kind: 'object_event', name: 'put_on_body', phase: 'end', src: src || 'auto', args: {object: p.id, anchor: name}, did: `put the ${p.name} on ${anchorWord(name)}`});
    }
    function easeDown(p, sd, R, src) {   // a held prop set onto a rest pose from wherever the paw holds it now
      foldKf(sd);
      const from = (heldWorld(p) || p.wpos).slice(), rotW = p.hrot.slice();
      p.held = null; p.grip = null; p.free = true; p.rot = [0, rotW[1], 0]; p.hoff = [0, 0, 0]; setP(p, from); p.where = 'placed';
      p.attE = {t0: t(), dur: 0.22, from: {l: p.att.l, r: p.att.r}, to: {l: 0, r: 0}, each(m) { setP(p, lerp3(from, R.pos, m)); },
        end() { restOn(p, R.sup, R.pos); emit({kind: 'object_event', name: R.sup.inside ? 'put_in' : 'put_down', phase: 'end', src: src || 'auto', args: {object: p.id, where: R.sup.id}, did: `put the ${p.name} ${supName(p)}`}); }};
    }
    // the planner prompt's compact list of things
    function things() {
      return E.P.map(p => {
        const A = AFF(p), w = p.wpos || P(p);
        return {id: p.id, name: p.name, kind: feat(p).kind, where: propWhere(p), held: p.held ? (p.held === 'both' ? 'both' : sideWord(p.held)) : null, worn: !!p.worn,
          on: p.sup ? p.sup.id : (p.place || null), inside: !!(p.sup && p.sup.inside), pos: w.map(v => +v.toFixed(3)), size: +p.size.toFixed(3),
          afford: Object.keys(A).concat(isFurn(p) ? ['furniture'] : []).concat(p.def.throwable ? ['throwable'] : []),
          ...(hasArt(p) || feat(p).laptop ? {open: (p.s || 0) > 0.5} : {})};
      });
    }
    // grasp points of both paws from the hand columns actually drawn this frame (+ their world velocities)
    function trackPaws(o, rootO, bendP) {
      const dt = E.lastDt || 1 / 30;
      for (const s of ['l', 'r']) {
        const h = [o['hand_' + s + '_x'], o['hand_' + s + '_y'], o['hand_' + s + '_z']]; if (!h.every(Number.isFinite) || !E.reach[s]) continue;
        const g = toWorldR(rootO, bendUp(E.reach[s].fwd(h), bendP));
        if (E.pawGW[s] && dt > 0) E.pawGV[s] = lerp3(E.pawGV[s], mul3(sub3(g, E.pawGW[s]), 1 / dt), 0.5);
        E.pawGW[s] = g;
        E.pawGN[s] = undefined; (E.pawNq || (E.pawNq = {}))[s] = {h, rootO, bendP};   // the palm normal: computed when asked (pawN)
      }
    }
    function pawN(s) {   // the palm normal → world (the root's rotation and the waist bend), lazily per frame (step cost)
      if (E.pawGN[s] !== undefined) return E.pawGN[s];
      const q = E.pawNq && E.pawNq[s]; if (!q) return (E.pawGN[s] = null);
      const nl = E.reach[s].normal(q.h);
      E.pawGN[s] = nl ? sub3(toWorldR(q.rootO, bendUp(add3([0, 0.5, 0], nl), q.bendP)), toWorldR(q.rootO, bendUp([0, 0.5, 0], q.bendP))) : null;
      return E.pawGN[s];
    }
    function postKf(o, kfOn) {   // after the keyframe layer (motion.js): the hands really drawn, the held offsets it asks for
      if (!E.kfh || !E.P.length) { E.kfLive = false; return; }
      const was = E.kfLive; E.kfLive = !!kfOn;
      if (!kfOn && was) { trackPaws(o, rootNow(), E.bend ? E.bend.p : 0); trackFrame(o); }   // (the frame it ended: post() skipped)
      if (kfOn) {
        const r = {x: +o.root_x || 0, y: +o.root_y || 0, z: +o.root_z || 0, yaw: +o.root_yaw || 0, pitch: +o.root_pitch || 0, roll: +o.root_roll || 0};
        E.rootKfNow = r;
        try { trackPaws(o, r, E.bend ? E.bend.p : 0); for (const s of ['l', 'r']) { const pw = E.paw[s]; pw.hPrev = [o['hand_' + s + '_x'], o['hand_' + s + '_y'], o['hand_' + s + '_z']]; if (!(pw.ctrl > 0)) pw.h = pw.hPrev.slice(); } trackFrame(o); } finally { E.rootKfNow = null; }
      }
      for (const s of ['l', 'r']) {
        const k = E.kfh[s], p = k && heldBy(s); if (!p || p.held === 'both') continue;
        const c = 'prop' + p.k + '_', w = clamp(p.att[s], 0, 1); if (!(c + 'x' in o)) continue;
        o[c + 'x'] += w * k.x; o[c + 'y'] += w * k.y; o[c + 'z'] += w * k.z;
        o[c + 'rx'] = wrapH(o[c + 'rx'] + w * k.rx); o[c + 'ry'] = wrapH(o[c + 'ry'] + w * k.ry); o[c + 'rz'] = wrapH(o[c + 'rz'] + w * k.rz);
      }
    }

    // ---------------------------------------------------------------- objects engine: rest poses for relations
    // relation words → on | in | under | behind | in_front_of | next_to | left_of | right_of. LEFT / RIGHT are from the
    // USER's viewpoint (the camera): left_of = toward screen-left (−x world), right_of = toward screen-right (+x); behind =
    // away from the camera (−z), in front of = toward it (+z). (A PAW / FOOT named by the user is the character's OWN side.)
    const REL_WORD = {on: 'on', in: 'in', under: 'under', behind: 'behind', in_front_of: 'in front of', next_to: 'next to', left_of: 'to the left of', right_of: 'to the right of'};
    function normRel(s) {
      const w = String(s || '').toLowerCase().replace(/[\s-]+/g, '_');
      if (!w) return null;
      if (/^(on|onto|on_to|on_top|on_top_of|top|upon|atop|over|stack|stacked_on)$/.test(w)) return 'on';
      if (/^(in|into|in_to|inside|inside_of|within)$/.test(w)) return 'in';
      if (/^(under|underneath|below|beneath)$/.test(w)) return 'under';
      if (/^(behind|back_of|in_back_of|in_behind)$/.test(w)) return 'behind';
      if (/^(in_front_of|front|in_front|before|front_of)$/.test(w)) return 'in_front_of';
      if (/^(next_to|beside|by|near|next|alongside|close_to|by_the_side_of|nearby)$/.test(w)) return 'next_to';
      if (/^(left_of|left|to_the_left_of|to_the_left|left_side)$/.test(w)) return 'left_of';
      if (/^(right_of|right|to_the_right_of|to_the_right|right_side)$/.test(w)) return 'right_of';
      return null;
    }
    // overlap with any resting prop (its box), except those in `ign` and what rides on p
    function boxHit(p, w, ign = []) {
      const up = stackAbove(p);
      for (const q of E.P) {
        if (q === p || ign.includes(q) || up.includes(q) || !loose(q)) continue;
        const Q = P(q), l = rotYv(sub3(w, Q), -ryOf(q)), pr = Math.hypot(p.hx, p.hz) * 0.85;
        if (Math.abs(w[1] - Q[1]) < q.hH + p.hH - 0.003 && Math.abs(l[0]) < q.hx + pr - 0.002 && Math.abs(l[2]) < q.hz + pr - 0.002) return q;
      }
      return null;
    }
    function stackTop(q, ex) { let cur = q; for (let i = 0; i < 10; i++) { const r = ridersOf(cur).find(x => !x.sup.inside && loose(x) && x !== ex); if (!r) break; cur = r; } return cur; }
    const halfAlong = (hx, hz, yaw, d) => { const ax = rotYv([1, 0, 0], yaw), az = rotYv([0, 0, 1], yaw); return Math.abs(d[0] * ax[0] + d[2] * ax[2]) * hx + Math.abs(d[0] * az[0] + d[2] * az[2]) * hz; };
    const REL_DIR = {behind: [0, 0, -1], in_front_of: [0, 0, 1], left_of: [-1, 0, 0], right_of: [1, 0, 0]};
    // the rest pose (physical) of p for a relation to a target {rel, prop | place | me}; {err} when there is none
    function restPose(p, T) {
      const rel = T.rel || 'on';
      if (T.prop) {
        const q = T.prop;
        if (q === p) return {err: `I can't put the ${p.name} ${REL_WORD[rel] || rel} itself`};
        if (stackAbove(p).includes(q)) return {err: `the ${q.name} is resting on the ${p.name}`};
        if (isFurn(q)) {
          const pl = placeOfFurn(q);
          if (rel === 'on') return pl ? restPose(p, {rel: 'on', place: pl, near: T.near}) : {err: `there is no room on the ${q.name}`};
          if (rel === 'in') return {err: `the ${q.name} isn't something things go in`};
          const R = footOf(q, q.pos, ryOf(q));
          if (rel === 'under') return underSpot(p, R, q.name);
          return besideSpot(p, {c: q.pos, hx: R.hw, hz: R.hd, yaw: ryOf(q), bottom: 0, name: q.name}, rel, T);
        }
        if (rel === 'in') {
          const C = AFF(q).container; if (!C) return {err: `the ${q.name} isn't something things go in`};
          if (p.hx > C.inner[0] + 1e-3 || p.hz > C.inner[2] + 1e-3 || (isCont(p) && p.size >= q.size)) return {err: `the ${p.name} is too big to fit in the ${q.name}`};
          const Q = P(q), fl = Q[1] + (Number.isFinite(+C.floor_y) ? +C.floor_y : -q.hH * 0.8), mx = Math.max(0, C.inner[0] - p.hx), mz = Math.max(0, C.inner[2] - p.hz);
          const ins = E.P.filter(r => r !== p && r.sup && r.sup.id === q.id && r.sup.inside && loose(r));
          for (const [a, b] of [[0, 0], [-mx, -mz], [mx, -mz], [-mx, mz], [mx, mz], [0, -mz], [0, mz], [-mx, 0], [mx, 0]]) {
            const w = add3([Q[0], fl + p.hH, Q[2]], rotYv([a, 0, b], ryOf(q)));
            if (!boxHit(p, w, [q])) return {pos: w, sup: {kind: 'prop', id: q.id, inside: true}, placeId: supPlace(q)};
          }
          const top = ins.reduce((m, r) => Math.max(m, P(r)[1] + topOff(r)), fl), w = [Q[0], top + p.hH, Q[2]];
          if (C.needs_open && w[1] + p.hH > Q[1] + topOff(q) + 0.005) return {err: `there is no room left in the ${q.name}`};
          return {pos: w, sup: {kind: 'prop', id: q.id, inside: true}, placeId: supPlace(q)};
        }
        if (rel === 'on') {
          const top = stackTop(q, p), Tq = P(top), w = [Tq[0], Tq[1] + topOff(top) + p.hH, Tq[2]];
          return {pos: w, sup: {kind: 'prop', id: top.id}, placeId: supPlace(top), onTop: top !== q ? top : null};
        }
        if (rel === 'under') {   // under a small thing: the honest approximation is beside it
          const r = besideSpot(p, {c: P(q), hx: q.hx, hz: q.hz, yaw: ryOf(q), bottom: P(q)[1] - q.hH, name: q.name}, 'next_to', T);
          if (!r.err) r.note = `the ${q.name} is too small to go under, so it is next to it`;
          return r;
        }
        return besideSpot(p, {c: P(q), hx: q.hx, hz: q.hz, yaw: ryOf(q), bottom: P(q)[1] - q.hH, name: q.name, prop: q}, rel, T);
      }
      if (T.place) {
        const pl = T.place;
        if (rel === 'under' && !pl.floor) { const f = pl.prop ? byId(pl.prop) : null; return f ? underSpot(p, footOf(f, f.pos, ryOf(f)), f.name) : underSpot(p, {c: pl.center, hw: pl.size[0] / 2, hd: pl.size[1] / 2, ry: pl.yaw || 0}, pl.name.replace(/^the /, '')); }
        if (rel !== 'on' && rel !== 'in' && !pl.floor) return besideSpot(p, {c: pl.center, hx: pl.size[0] / 2, hz: pl.size[1] / 2, yaw: pl.yaw || 0, bottom: 0, name: pl.name.replace(/^the /, '')}, rel, T);
        const near = T.near || [rootNow().x, 0, rootNow().z];
        const c = pl.center, sx = (pl.size[0] || 0.4) / 2 - p.hx - 0.01, sz = (pl.size[1] || 0.4) / 2 - p.hz - 0.01;
        const cand = pl.floor ? [] : pl.spots.map(q => [q[0], 0, q[2]]);
        if (!pl.floor) for (const a of [-0.66, -0.33, 0, 0.33, 0.66]) for (const b of [-0.6, 0, 0.6]) cand.push(add3(c, rotYv([a * Math.max(0, sx), 0, b * Math.max(0, sz)], pl.yaw || 0)));
        else for (let r = 0.25; r <= 0.75; r += 0.1) for (let k = 0; k < 12; k++) { const a = k / 12 * TAU; cand.push([near[0] + r * Math.sin(a), 0, near[2] + r * Math.cos(a)]); }
        cand.sort((u, v) => d2(u, near) - d2(v, near));
        for (const q of cand) {
          const s = supportAt(q[0], q[2], pl.floor ? 0.05 : pl.top_y + 0.01, p); if (s.kind !== 'place' || s.id !== pl.id) continue;
          const w = [q[0], s.top + p.hH, q[2]]; if (boxHit(p, w)) continue;
          if (pl.floor && standClear(q[0], q[2], null) === false && !T.anyFloor) continue;
          return {pos: w, sup: s, placeId: pl.id};
        }
        return {err: `there is no room on ${pl.name}`};
      }
      if (T.me) {   // toward the camera: the front of the rug, in front of where it stands
        const r = rootNow(), x0 = clamp(r.x, -0.6, 0.6);
        for (const [dx, dz] of [[0, 0], [0.12, 0], [-0.12, 0], [0, -0.1], [0.24, 0], [-0.24, 0]]) {
          const q = [x0 + dx, 0, Math.max(r.z + 0.32, 0.55) + dz], s = supportAt(q[0], q[2], 0.05, p), w = [q[0], s.top + p.hH, q[2]];
          if (s.kind === 'place' && s.id === 'floor' && !boxHit(p, w)) return {pos: w, sup: s, placeId: 'floor', me: true};
        }
        return {err: 'there is no clear spot in front of you'};
      }
      return {err: 'where to?'};
    }
    function besideSpot(p, B, rel, T) {   // on the same support as B, offset along the relation by both sizes
      const me = [rootNow().x, 0, rootNow().z];
      let dirs = REL_DIR[rel] ? [REL_DIR[rel]] : [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
      if (!REL_DIR[rel]) { const v = sub3(flat(me), flat(B.c)); dirs.sort((a, b) => -(a[0] * v[0] + a[2] * v[2]) + (b[0] * v[0] + b[2] * v[2])); if (T && T.side) { const sg = sideSign(T.side, 0); if (sg) dirs = [[sg, 0, 0]].concat(dirs); } }
      for (const d of dirs) for (const extra of [0.012, 0.04, 0.08]) {
        const gap = halfAlong(B.hx, B.hz, B.yaw, d) + halfAlong(p.hx, p.hz, 0, d) + extra, q = add3(flat(B.c), mul3(d, gap));
        const s = supportAt(q[0], q[2], B.bottom + 0.01, p);
        if (Math.abs(s.top - B.bottom) > 0.015 && !(B.bottom === 0 && s.kind === 'place' && s.id === 'floor')) continue;   // off the edge of what it stands on
        const w = [q[0], s.top + p.hH, q[2]];
        if (boxHit(p, w, B.prop ? [B.prop] : [])) continue;
        if (B.bottom < 0.02 && !standClearPt(q)) continue;
        return {pos: w, sup: s, placeId: s.kind === 'place' ? s.id : supPlace(byId(s.id))};
      }
      return {err: `there is no room ${REL_WORD[rel] || rel} the ${B.name}`};
    }
    const standClearPt = q => !E.P.some(f => isFurn(f) && inFoot(footOf(f, f.pos, ryOf(f)), q, 0.005));   // the floor spot is not inside a piece of furniture
    function underSpot(p, R, name) {
      for (const [a, b] of [[0, 0.3], [0, 0], [-0.4, 0.3], [0.4, 0.3], [-0.4, 0], [0.4, 0]]) {
        const q = add3(R.c, rotYv([a * R.hw, 0, b * R.hd], R.ry || 0)), s = supportAt(q[0], q[2], 0.05, p), w = [q[0], s.top + p.hH, q[2]];
        if (s.kind === 'place' && s.id === 'floor' && !boxHit(p, w)) return {pos: w, sup: s, placeId: 'floor', under: true};
      }
      return {err: `there is no room under the ${name}`};
    }
    // what a command's target names: the character's own body, him, a prop, a place, a piece of furniture
    function resolveTarget(cmd, p) {
      const tg = cmd.target ?? cmd.place ?? cmd.to ?? cmd.onto ?? cmd.into ?? cmd.on;
      if (tg === undefined || tg === null || tg === '') return null;
      const parts = typeof tg === 'object' ? [tg.anchor, tg.ref, tg.name, tg.id, tg.place, tg.object && typeof tg.object === 'object' ? tg.object.name || tg.object.id : tg.object] : [tg];
      const w = parts.filter(x => typeof x === 'string').join(' ').toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
      if (!w) return null;
      const own = /\b(your|my|its|his|her|own)\b/.test(w);
      if (/\b(head|crown|noggin)\b/.test(w)) return {anchor: 'head'};
      if (/\b(foot|feet|toe|toes)\b/.test(w)) { const sd = /\bleft\b/.test(w) ? 'r' : /\bright\b/.test(w) ? 'l' : (sideIn(cmd.side) === 'l' || sideIn(cmd.side) === 'r' ? sideIn(cmd.side) : 'r'); return {anchor: 'foot_' + sd}; }
      if (/\blap\b/.test(w)) return {anchor: 'lap'};
      if (/\bshoulder/.test(w)) return {anchor: 'shoulder_' + (/\bleft\b/.test(w) ? 'r' : 'l')};
      if (/\bback\b/.test(w) && own) return {anchor: 'back'};
      if (/^(me|him|you|user|camera|us|here|the user|the camera)$/.test(w) || /\bto me\b/.test(w)) return {me: true};
      const od = typeof tg === 'object' && (tg.id || tg.object) ? (tg.object && typeof tg.object === 'object' ? tg.object : tg.id ? {id: tg.id} : {name: w}) : {name: w};
      const r = resolveObject(od, 'to', {});
      if (r.p && r.p === p && r.ask === undefined) return {err: `I can't put the ${p.name} on itself`, self: true};
      let q = r.p || null;
      if (!q && r.cands) { const c = r.cands.filter(x => x !== p); if (c.length === 1) q = c[0]; else if (!r.cands.includes(p)) return {err: r.err, ask: true}; else return {err: r.err, ask: true}; }
      if (q && !isFurn(q)) return {prop: q};
      const pl = resolvePlace(w, p);
      if (pl && !pl.err) return {place: pl};
      if (q) return {prop: q};
      return {err: r.err || `there is no ${w} here`};
    }

    // ---------------------------------------------------------------- objects engine: chains for the new verbs
    const gripW = (p, c, yawW) => add3(c, rotYv(gripLocal(p), yawW ?? heldYawW(p)));   // the paw for a centre at c
    function waitStep(name, maxS, pred, opt = {}) { return {name, dur: maxS, atomic: !!opt.atomic, start() {}, update(tt) { return pred(tt) || tt >= maxS; }}; }
    // onto the head: worn (afford.wear) or balanced; one paw brings it there from above (or from the front for the face)
    function headSteps(p, wear) {
      const s = p.held === 'both' ? 'l' : p.held, out = [], off = () => wear ? wearOff(p) : balanceOff(p);
      const spot = () => { const H = headPose(); return add3(H.pos, rotYv(off(), H.yaw)); };
      const front = Math.abs(off()[2]) > Math.abs(off()[1]);
      out.push(lookStep(null, false));
      out.push(pawPath(s, gripW(p, add3(spot(), front ? rotYv([0, 0.01, 0.1], rootNow().yaw) : [0, 0.09, 0.03])), 0.75, {turn: 0.2}));
      out.push(pawPath(s, gripW(p, spot()), 0.35, {arc: 0}));
      out.push({name: wear ? 'put on' : 'balance', atomic: true, dur: 0.5, start() { const q = headClash(p, off(), !wear); this.ok = !q; if (this.ok) headAttach(p, wear, off(), 0.35, 'auto'); else if (E.chain) E.chain.err = `the ${q.name} is in the way on its head`; },
        update(tt) { if (this.ok) E.paw[s].tgt = gripW(p, spot()); return !p.attE || !this.ok || tt > 1; }});
      out.push(pawPath(s, gripW(p, add3(spot(), front ? rotYv([0.06, -0.03, 0.08], rootNow().yaw) : [0.05, 0.06, 0.05])), 0.3, {arc: 0, open: 0.6}));
      out.push(retractStep(s));
      return out;
    }
    function headOffSteps(p, side) {   // a one-paw worn / balanced thing back into a paw
      const s = side && !heldBy(side) ? side : !heldBy('r') ? 'r' : 'l', out = [];
      const spot = () => { const H = headPose(); return add3(H.pos, rotYv(p.hoff || [0, 0, 0], H.yaw)); }, yawW = () => p.homeRot[1];
      out.push(pawPath(s, gripW(p, add3(spot(), [0, 0.05, 0.05]), yawW()), 0.6, {open: 0.8}));
      out.push(pawPath(s, gripW(p, spot(), yawW()), 0.3, {arc: 0}));
      out.push({name: 'take off', atomic: true, dur: 0.4, start() {
        const was = p.worn ? 'took off' : 'took down', off0 = (p.hoff || [0, 0, 0]).slice();
        p.worn = false; p.bal = null; p.held = s; p.where = 'held'; p.grip = gaOf(p).slice(); p.hrot = [0, 0, 0]; p.place = null;
        p.attE = {t0: t(), dur: 0.32, from: {l: p.att.l, r: p.att.r}, to: {l: s === 'l' ? 1 : 0, r: s === 'r' ? 1 : 0}, hh0: p.hh || 1, hh: 0, off0, off: [0, 0, 0],
          end() { p.hh = 0; p.hoff = [0, 0, 0]; emit({kind: 'object_event', name: 'took_off', phase: 'end', src: 'auto', args: {object: p.id, side: sideWord(s)}, did: `${was} the ${p.name} (holding it in its ${sideWord(s)} paw)`}); }};
      }, update(tt) { E.paw[s].tgt = gripW(p, spot(), yawW()); E.paw[s].open = lerp(0.8, graspValues(p, s).open, mj(clamp(tt / 0.3, 0, 1))); return !p.attE; }});
      out.push(retractStep(s, true));
      return out;
    }
    function anchorSteps(p, name) {   // a body anchor: the paw sets it there, the engine carries it with the body from then on
      const s = p.held === 'both' ? 'l' : p.held, out = [], A = () => anchorWorld(name, p);
      const low = A() && A()[1] < 0.2;
      if (low && E.body.legs) out.push(bendStep(gripW(p, A())));
      out.push(pawPath(s, gripW(p, add3(A(), [0, 0.08, name === 'back' ? -0.04 : 0.04])), 0.7, {turn: -0.2}));
      out.push(pawPath(s, gripW(p, A()), 0.3, {arc: 0}));
      out.push({name: 'set on body', atomic: true, dur: 0.4, start() { anchorAttach(p, name, 0.3, 'auto'); }, update() { E.paw[s].tgt = gripW(p, A(), p.anc ? rootNow().yaw + p.anc.rr : undefined); return !p.attE; }});
      if (low && E.body.legs) out.push(bendStep(null));
      out.push(retractStep(s));
      return out;
    }
    // a hinged lid (afford.articulation): one paw on its handle, along the hinge arc while prop{k}_s eases
    function artHandle(q, sv) {
      const A = AFF(q).articulation, h = v3(A.handle_shut) || [0, q.hH, q.hz], g = v3(A.hinge) || [0, q.hH, -q.hz], d = sub3(h, g), th = (+A.open_turns || 0.3) * sv * TAU;
      return add3(P(q), rotYv(add3(g, [d[0], d[1] * Math.cos(th) + d[2] * Math.sin(th), -d[1] * Math.sin(th) + d[2] * Math.cos(th)]), ryOf(q)));
    }
    function artSteps(q, open) {
      const s0 = q.s || 0, s1 = open ? 1 : 0, sd = toLocal(q.pos)[0] < 0 ? 'l' : 'r', out = [lookStep(artHandle(q, s0), true)];
      if (E.body.legs) out.push(bendStep(artHandle(q, s0)));
      out.push(pawPath(sd, add3(artHandle(q, s0), [0, 0.04, 0.03]), 0.55, {open: 0.5, turn: -0.3}));
      out.push(pawPath(sd, artHandle(q, s0), 0.22, {arc: 0}));
      out.push({name: open ? 'open' : 'close', dur: 0.9, start() {}, update(tt) { const u = clamp(tt / 0.9, 0, 1); q.s = lerp(s0, s1, mj(u)); E.paw[sd].tgt = artHandle(q, q.s); E.look.P = artHandle(q, q.s); return u >= 1; },
        end() { q.s = s1; emit({kind: 'object_event', name: open ? 'opened' : 'closed', phase: 'end', src: 'auto', args: {object: q.id}, did: `${open ? 'opened' : 'closed'} the ${q.name}`}); }});
      out.push(pawPath(sd, add3(artHandle(q, s1), [0, 0.05, 0.04]), 0.25, {arc: 0, open: 0.6}));
      if (E.body.legs) out.push(bendStep(null));
      out.push(lookStep(null, false));
      out.push(retractStep(sd));
      return out;
    }
    // throw at a target: wind-up, a flick that arrives at the release point with the ballistic velocity that reaches the
    // target, let go (physics from there: it lands / bounces / goes in, or misses — reported as it happened)
    function throwSteps(p, aim, op) {
      const s = p.held === 'both' ? 'l' : p.held, sg = s === 'l' ? -1 : 1, B0 = E.body, out = [], Hh = B0.hi[1] - B0.lo[1];
      out.push(lookStep(aim.pos, true));
      const Wl = [sg * 0.2, B0.lo[1] + 0.55 * Hh, B0.front - 0.02], Rl = [sg * 0.14, B0.lo[1] + 0.62 * Hh, B0.front + 0.16];
      out.push(pawPath(s, reachClamp(s, toWorld(Wl)), 0.45, {look: aim.pos}));
      out.push({name: 'throw', atomic: true, dur: 0.6, start() {
        const pw = E.paw[s]; this.W = fwdW(s, pw.hPrev); this.R = reachClamp(s, toWorld(Rl)); this.tf = 0.16;
        const c0 = sub3(this.R, rotYv(gripLocal(p), heldYawW(p))), d = d2(aim.pos, c0);
        const T = clamp(0.3 + 0.3 * d, 0.32, 1.1); this.v = [(aim.pos[0] - c0[0]) / T, (aim.pos[1] - c0[1] + 0.5 * GRAV * T * T) / T, (aim.pos[2] - c0[2]) / T];
        this.q = [0, 1, 2].map(i => quintic(this.W[i], 0, 0, this.R[i], this.v[i] * 0.6, 0, this.tf)); this.rel = false;
      }, update(tt) {
        const pw = E.paw[s];
        if (tt < this.tf) { pw.tgt = this.q.map(f => f(tt)); return false; }
        if (!this.rel) {
          this.rel = true; const thrown = letGoNow(s, sub3(this.v, p.wvel || [0, 0, 0]), 'auto', {why: 'thrown', target: {id: aim.id, name: aim.name}, spin: p.def.afford && AFF(p).throw === 'frisbee' ? [0, 2.5, 0] : [rnd() - 0.5, 0, rnd() - 0.5], flipOK: false,
            done: (pp, s2, hit) => { if (op) op.did[1] = `threw the ${p.name} ${aim.word}` + (hit ? (aim.after ? '; ' + aim.after(pp) : '') : ` — it missed and landed ${supName(pp)}`); }});
          if (thrown) { thrown.fly.v0 = this.v.slice(); thrown.fly.p0 = (heldWorld(thrown) || thrown.fly.p0).slice(); }
          emit({kind: 'object_event', name: 'thrown', phase: 'end', src: 'auto', args: {object: p.id, at: aim.id || 'there'}, did: `threw the ${p.name} ${aim.word}`});
          this.F = sub3(this.R, this.W);
        }
        const k = mj(clamp((tt - this.tf) / 0.3, 0, 1)); pw.tgt = add3(this.R, mul3(norm2(this.F), 0.05 * k)); pw.open = lerp(pw.open, 0.85, 0.3);
        return tt >= this.tf + 0.3;
      }});
      out.push(waitStep('watch it land', 3.5, () => !p.fly));
      out.push(lookStep(null, false));
      out.push(retractStep(s));
      return out;
    }
    // juggle two things: one per paw, alternate throws to the other paw (two flights at once), each caught by an armed paw
    function juggleSteps(a, b, n, op) {
      const B0 = E.body, Hh = B0.hi[1] - B0.lo[1], wait = sd => toWorld([(sd === 'l' ? -1 : 1) * 0.15, B0.lo[1] + 0.47 * Hh, B0.front + 0.14]);
      const out = [lookStep(toWorld([0, B0.lo[1] + 0.9 * Hh, B0.front + 0.2]), true)];
      out.push(par(['l', 'r'].map(sd => pawPath(sd, wait(sd), 0.5))));
      const slot = 0.3, Tf = 2 * slot;
      out.push({name: 'juggle', atomic: true, dur: n * slot + Tf + 1.5, start() {
        this.s0 = a.held === 'l' || a.held === 'r' ? a.held : 'l'; this.k = 0; this.inAir = {}; this.miss = 0; this.caught = 0; this.t0 = t();
        for (const sd of ['l', 'r']) E.paw[sd].ctrlT = 1;
      }, update(tt) {
        const C = E.chain, stop = !!(C && C.stopped);
        // the two-object exchange: a paw throws to the other paw when the thing coming TO it has passed its apex (so it is
        // empty again before that one lands); with nothing in the air the paws alternate (the first throw: s0)
        const air = Object.values(this.inAir), due = this.k < n && !stop && tt >= 0.15 && air.length < 2 && (!air.length || tt - (air[0].t - Tf) >= Tf * 0.5);
        if (due) {
          const sd = air.length ? air[0].to : heldBy(this.s0) ? this.s0 : otherSide(this.s0), p = heldBy(sd);
          if (p && !p.attE && (p === a || p === b)) {
            this.lastS = sd;
            // (aimed at where it WOULD sit in the other paw as that paw really is now — the commanded wait point can lie
            // beyond what the arm reaches)
            const o = otherSide(sd), Cw = wait(o), c0 = heldWorld(p) || p.wpos, tgt = heldOne(p, o, heldYawW(p), gripLocal(p)) || sub3(Cw, rotYv(gripLocal(p), heldYawW(p)));
            const v = [(tgt[0] - c0[0]) / Tf, (tgt[1] - c0[1] + 0.5 * GRAV * Tf * Tf) / Tf, (tgt[2] - c0[2]) / Tf];
            letGoNow(sd, sub3(v, p.wvel || [0, 0, 0]), 'auto', {why: 'juggled', spin: [0.8, 0, 0]});
            if (p.fly) { p.fly.v0 = v; p.fly.p0 = c0.slice(); p.fly.t0 = t(); }
            E.catchArm[o] = {until: t() + Tf + 0.3, src: 'auto', juggle: true}; this.inAir[p.id] = {to: o, t: tt + Tf, p};
            this.k++;
          } else if (!p && !Object.keys(this.inAir).length) this.k = n;   // nothing left to throw
        }
        for (const id in this.inAir) { const A = this.inAir[id]; if (A.p.held === A.to) { this.caught++; delete this.inAir[id]; } else if (!A.p.fly && !A.p.held) { this.miss++; delete this.inAir[id]; } }
        for (const sd of ['l', 'r']) {   // the paws: wait, dip after a catch, rise into the throw; meet what is coming
          const pw = E.paw[sd], W = wait(sd), inc = Object.values(this.inAir).find(A => A.to === sd);
          let T = W;
          if (inc && inc.p.fly) { const lead = inc.t - tt; if (lead < 0.14) T = lerp3(W, sub3(P(inc.p), [0, 0, 0]), mj(clamp((0.14 - lead) / 0.14, 0, 1))); }
          else if (heldBy(sd)) { const ph = ((tt - 0.15) / slot) % 2; T = add3(W, [0, -0.04 * Math.sin(Math.PI * clamp(ph / 2, 0, 1)), 0]); }
          pw.tgt = T; pw.hd = null;
        }
        if (this.k >= n && !Object.keys(this.inAir).length) return true;
        return tt > n * slot + Tf + 1.4;
      }, end() { this.res = {caught: this.caught, miss: this.miss, throws: this.k}; JUGGLE_LAST.res = this.res;
        const lost = [a, b].filter(q => !q.held);
        if (op) op.did[1] = `juggled the ${a.name} and the ${b.name} (${this.k} throws${this.miss || lost.length ? '' : ', every one caught'})` + (lost.length ? ` — it missed ${lost.map(q => 'the ' + q.name + ' (it landed ' + supName(q) + ')').join(' and ')}` : '');
        emit({kind: 'object_event', name: 'juggled', phase: 'end', src: 'auto', args: {objects: [a.id, b.id], throws: this.k, missed: lost.map(q => q.id)}, did: op ? op.did[1] : 'juggled'}); }});
      out.push(lookStep(null, false));
      out.push(par([retractStep('l', true), retractStep('r', true)]));
      return out;
    }
    const JUGGLE_LAST = {res: null};
    // use a tool on a target (afford.use.motion): pour (tilt the spout over its top, flow on), rub / brush (strokes on its
    // top), tap / press (dabs) — then back
    function useSteps(tool, tgt, motion, op) {
      const s = tool.held === 'both' ? 'l' : tool.held, out = [], A = AFF(tool), pour = motion === 'pour', Pd = A.pour || {};
      const top = () => add3(P(tgt), [0, topOff(tgt) + (pour ? 0.09 : 0.012), 0]);
      out.push(lookStep(top(), true));
      if (E.body.legs && top()[1] < 0.25) out.push(bendStep(top()));
      if (pour) {
        const sp = v3(Pd.spout) || [tool.hx, tool.hH * 0.5, 0], tilt = Number.isFinite(+Pd.tilt_turns) ? +Pd.tilt_turns : 0.12, ax = /^x$/i.test(Pd.axis || '') ? 0 : 2;
        let yaw1 = 0, c1 = null;
        out.push({name: 'aim the spout', dur: 0.8, start() {
          const T = top(), c = heldWorld(tool) || tool.wpos, d = sub3(flat(T), flat(c)), dl = Math.hypot(d[0], d[2]) || 1;
          // the spout (prop-local +x side) points at the target: its world yaw, then the paw where the spout is above it
          const yW = (Math.atan2(-d[2], d[0]) - Math.atan2(-sp[2], sp[0])) / TAU; void dl;   // heading(rotYv(v, a)) = heading(v) + a
          this.h0 = tool.hrot[1]; yaw1 = this.h0 + wrap(wrap(yW - tool.homeRot[1]) - this.h0);
          c1 = sub3(T, rotYv(sp, yW)); this.path = pawPath(s, reachClamp(s, gripW(tool, c1, yW)), 0.7, {turn: 0}); this.path.start();
        }, update(tt) { const u = clamp(tt / 0.8, 0, 1); tool.hrot[1] = lerp(this.h0, yaw1, mj(u)); this.path.update(tt); return u >= 1 && tt >= this.path.dur; }});
        out.push({name: 'pour', dur: 2.6, start() { this.r0 = tool.hrot[ax]; this.flow = !A.articulation; }, update(tt) {
          const up = mj(clamp(tt / 0.5, 0, 1)), down = mj(clamp((tt - 2.0) / 0.5, 0, 1)), k = up * (1 - down);
          tool.hrot[ax] = this.r0 + (ax === 2 ? -1 : 1) * tilt * k;
          if (this.flow) tool.s = clamp((k - 0.6) / 0.4, 0, 1);
          if (tt > 0.6 && !this.said) { this.said = true; emit({kind: 'object_event', name: 'poured', phase: 'end', src: 'auto', args: {object: tool.id, on: tgt.id, what: Pd.what || 'it'}, did: `poured ${Pd.what || 'it'} from the ${tool.name} on the ${tgt.name}`}); if (AFF(tgt).water_target && /water/.test(Pd.what || '')) tgt.watered = t(); }
          return tt >= 2.5;
        }, end() { tool.hrot[ax] = this.r0; if (this.flow) tool.s = 0; }});
      } else {
        out.push(pawPath(s, reachClamp(s, gripW(tool, add3(top(), [0, tool.hH + 0.04, 0]))), 0.65, {}));
        const strokes = motion === 'tap' || motion === 'press' ? 3 : 4, per = motion === 'press' ? 0.6 : motion === 'tap' ? 0.35 : 0.5;
        out.push({name: motion, dur: strokes * per, start() { this.c = add3(top(), [0, tool.hH, 0]); }, update(tt) {
          const ph = tt / per, k = Math.sin(Math.PI * clamp(tt / (strokes * per), 0, 1));
          const off = motion === 'rub' || motion === 'brush' ? [0.035 * Math.sin(TAU * ph) * k, 0.004 * Math.abs(Math.sin(TAU * ph)), 0.015 * Math.cos(TAU * ph) * k] : [0, 0.035 * (1 - Math.abs(Math.sin(Math.PI * ph))) , 0];
          E.paw[s].tgt = reachClamp(s, gripW(tool, add3(this.c, off)));
          if (tt > per && !this.said) { this.said = true; emit({kind: 'object_event', name: 'used', phase: 'end', src: 'auto', args: {object: tool.id, on: tgt.id, motion}, did: `${{rub: 'rubbed', brush: 'brushed', tap: 'tapped', press: 'pressed'}[motion] || 'used'} the ${tool.name} on the ${tgt.name}`}); }
          return tt >= strokes * per;
        }});
      }
      if (E.body.legs && top()[1] < 0.25) out.push(bendStep(null));
      out.push(lookStep(null, false));
      out.push(retractStep(s, true));
      void op; return out;
    }
    // ---- in-paw manipulations from parameters (generic verbs: the thing is held first)
    function sideOfHeld(p) { return p.held === 'both' ? 'l' : p.held; }
    function spinChain(p, a, twirl) {   // spin about an axis, `amount` turns at a speed; "on your finger": a point pose, the thing up on the fingertip
      const s = sideOfHeld(p), two = p.held === 'both', out = [], ax = /^x/i.test(a.axis || '') ? 0 : /^z/i.test(a.axis || '') ? 2 : 1;
      const finger = !two && /finger|fingertip|tip/.test([a.style, a.on, a.target && (a.target.name || a.target), a.how, a.where].filter(x => typeof x === 'string').join(' ').toLowerCase());
      const n = clamp(Math.abs(num(a.amount ?? a.count, twirl ? 1 : 2)), 0.25, 12), sp = speedOf(a.speed) * (finger ? 2.2 : 1.2), dir = yawSign(a.dir, 1);
      const B0 = E.body, Hh = B0.hi[1] - B0.lo[1], up = [(s === 'l' ? -1 : 1) * 0.15, B0.lo[1] + (finger || twirl ? 0.72 : 0.58) * Hh, B0.front + 0.18];
      out.push(two ? par(['l', 'r'].map(sd => pawPath(sd, reachClamp(sd, toWorld(add3(up, [(sd === 'l' ? -1 : 1) * p.size * 0.95 - (s === 'l' ? -1 : 1) * 0.15, 0, 0]))), 0.55, {}))) : pawPath(s, reachClamp(s, toWorld(up)), 0.55, {turn: finger ? 0.6 : 0.3}));
      const dur = clamp(n / sp, 0.5, 8), lift = finger ? p.hH + p.size * 0.6 + 0.02 : 0;
      out.push({name: finger ? 'spin on its finger' : twirl ? 'twirl' : 'spin', dur: dur + 0.6, start() { this.r0 = p.hrot.slice(); this.o0 = p.hoff.slice(); },
        update(tt) {
          const u = clamp((tt - 0.3) / dur, 0, 1), k = mj(clamp(tt / 0.3, 0, 1)) * (1 - mj(clamp((tt - 0.3 - dur) / 0.3, 0, 1)));
          p.hrot = this.r0.slice(); p.hrot[ax] = this.r0[ax] + dir * n * cruise(u, 0.15);
          if (finger) { p.hoff = add3(this.o0, [0, lift * k, 0]); E.paw[s].point = k; E.paw[s].open = lerp(graspValues(p, s).open, 0.5, k); }
          return tt >= dur + 0.6;
        }, end() { p.hrot[ax] = wrapH(p.hrot[ax]); p.hoff = this.o0; if (finger) E.paw[s].point = 0;
          emit({kind: 'object_event', name: 'spun', phase: 'end', src: 'auto', args: {object: p.id, turns: n, axis: 'xyz'[ax], finger}, did: `spun the ${p.name} ${turnsWord(n)}${finger ? ' on its fingertip' : twirl ? ' in a twirl' : ''}`}); }});
      out.push(par((two ? ['l', 'r'] : [s]).map(sd => retractStep(sd, true))));
      return out;
    }
    function tiltChain(p, a) {
      const s = sideOfHeld(p), out = [], ax = /^z|side|roll|left|right/i.test(String(a.axis || a.dir || '')) ? 2 : 0, amt = clamp(num(a.amount, 0.12), -0.5, 0.5) * (ax === 2 ? yawSign(a.dir, 1) : 1);
      out.push(pawPath(s, reachClamp(s, toWorld(showPoint(s, p, 'show'))), 0.55, {}));
      out.push(stepFn('tilt', 0.6, u => { p.hrot[ax] = this0t(p, ax) + amt * mj(u); }, {start() { p.tiltFrom = p.hrot[ax]; }}));
      out.push(dwellStep('hold the tilt', a.hold ? Infinity : 1.0));
      if (!a.keep) out.push(stepFn('untilt', 0.5, u => { p.hrot[ax] = p.tiltFrom + amt * (1 - mj(u)); }, {}));
      out.push(stepFn('said', 0.001, () => emit({kind: 'object_event', name: 'tilted', phase: 'end', src: 'auto', args: {object: p.id, turns: amt}, did: `tilted the ${p.name}${a.keep ? ' (and kept it tilted)' : ' and set it straight again'}`})));
      out.push(retractStep(s, true));
      return out;
    }
    const this0t = (p, ax) => p.tiltFrom ?? p.hrot[ax];
    function waveChain(p, a) {
      const s = sideOfHeld(p), out = [], B0 = E.body, Hh = B0.hi[1] - B0.lo[1], up = toWorld([(s === 'l' ? -1 : 1) * 0.2, B0.lo[1] + 0.85 * Hh, B0.front + 0.1]);
      const n = clamp(Math.round(num(a.count, 4)), 1, 8);   // 4 by default (3 x 0.55 s fell between the frames he saw)
      out.push(pawPath(s, reachClamp(s, up), 0.6, {turn: 0.4}));
      const pres = presenter(p, 'wave', false, s, n * 0.55);
      out.push({name: 'wave it', dur: n * 0.55, start() { this.c = fwdW(s, E.paw[s].hPrev); this.r0 = p.hrot.slice(); pres.start(); }, update(tt) {
        pres.update(tt); const ph = Math.sin(TAU * tt / 0.55), k = Math.sin(Math.PI * clamp(tt / (n * 0.55), 0, 1)) ** 0.5; E.paw[s].tgt = add3(this.c, rotYv([0.07 * ph * k, 0.01 * Math.abs(ph), 0], rootNow().yaw)); p.hrot[2] = this.r0[2] + 0.05 * ph * k; return tt >= n * 0.55;
      }, end() { p.hrot = this.r0; pres.end(); emit({kind: 'object_event', name: 'waved', phase: 'end', src: 'auto', args: {object: p.id}, did: `waved the ${p.name} in the air`}); }});
      out.push(retractStep(s, true));
      return out;
    }
    function raiseChain(p, a) {   // over its head (both paws for a two-pawed thing)
      const two = p.held === 'both', s = sideOfHeld(p), out = [], hp = headPoint();
      const tg = sd => reachClamp(sd, toWorld([two ? (sd === 'l' ? -1 : 1) * p.size * 0.95 : (s === 'l' ? -0.06 : 0.06), hp[1] + E.headTop + p.hH + 0.06, hp[2] + 0.04]));
      out.push(par((two ? ['l', 'r'] : [s]).map(sd => pawPath(sd, tg(sd), 0.7, {turn: 0.2}))));
      // (short arms — say how high it really got: over the head only when the thing's bottom clears the head top)
      out.push(stepFn('said', 0.001, () => { const sd0 = two ? 'l' : s, g = E.pawGW && E.pawGW[sd0] ? E.pawGW[sd0] : fwdW(sd0, E.paw[sd0].hPrev), top = toWorld([hp[0], hp[1] + E.headTop, hp[2]])[1], over = g[1] - p.hH > top - 0.02;
        const dd = over ? `raised the ${p.name} over its head` : `held the ${p.name} up as high as its short arms reach (not over its head)`;
        { const C = E.chain, J = C && C.opOf, op = J && J.done && J.done[J.done.length - 1]; if (op && op.op === 'generic') op.did = [dd, dd]; if (C) C.did = dd; }   // the job's words = what happened
        emit({kind: 'object_event', name: 'raised', phase: 'end', src: 'auto', args: {object: p.id, over}, did: over ? `raised the ${p.name} up over its head` : `held the ${p.name} up as high as its short arms reach (about ${g[1] + p.hH > top ? 'head' : 'chin'} height, not over its head)`}); }));
      out.push(dwellStep('hold it up', a.hold ? Infinity : 1.4));
      out.push(par((two ? ['l', 'r'] : [s]).map(sd => retractStep(sd, true))));
      return out;
    }
    function hideChain(p, a) {   // behind its back: the paw goes behind the body if the reach map allows, else onto the back anchor
      const s = sideOfHeld(p), out = [], B0 = E.body;
      if (p.held === 'both') return null;
      const Bp = [(s === 'l' ? -1 : 1) * (B0.s[0] * 0.7), B0.lo[1] + 0.42 * (B0.hi[1] - B0.lo[1]), B0.lo[2] - p.size - 0.02], r = E.reach[s].inv(Bp, E.paw[s].hPrev, 14);
      if (r.err < 0.03) {
        out.push(pawPath(s, toWorld(Bp), 0.8, {turn: -0.3}));
        out.push(stepFn('said', 0.001, () => emit({kind: 'object_event', name: 'hid', phase: 'end', src: 'auto', args: {object: p.id, where: 'behind its back'}, did: `hid the ${p.name} behind its back`})));
        out.push(dwellStep('keep it hidden', a.hold === false ? 2 : Infinity));
        out.push(retractStep(s, true));
        return out;
      }
      return anchorSteps(p, 'back').concat([stepFn('said', 0.001, () => { const d = `its paw can't reach behind its back, so it tucked the ${p.name} onto its back`;
        if (E.chain && E.chain.op && E.chain.op.did) E.chain.op.did[1] = `tucked the ${p.name} onto its back (its paw can't reach behind it)`;   // (the job's words say the approximation too)
        emit({kind: 'object_event', name: 'hid', phase: 'end', src: 'auto', args: {object: p.id, where: 'on its back'}, did: d}); })]);
    }
    function drinkChain(p, a) {   // to the mouth and tip it back a little (afford.drink)
      const s = sideOfHeld(p), out = [], hp = headPoint(), mouth = [s === 'l' ? -0.05 : 0.05, hp[1] - 0.12, hp[2] + 0.27 + p.size];
      out.push(pawPath(s, reachClamp(s, toWorld(mouth)), 0.7, {turn: 0.3}));
      out.push(stepFn('sip', 1.6, u => { const k = Math.sin(Math.PI * u); p.hrot[0] = this0d(p) - 0.12 * k; E.look.wT = 0; }, {start() { p.drinkFrom = p.hrot[0]; }}));
      out.push(stepFn('said', 0.001, () => { p.hrot[0] = p.drinkFrom; emit({kind: 'object_event', name: 'drank', phase: 'end', src: 'auto', args: {object: p.id}, did: `took a sip from the ${p.name}`}); }));
      out.push(retractStep(s, true));
      void a; return out;
    }
    const this0d = p => p.drinkFrom ?? p.hrot[0];
    function pourChain(p, a) {   // pour with no target: in front of it, onto the floor
      const s = sideOfHeld(p), out = [], Pd = AFF(p).pour, ax = /^x$/i.test(Pd.axis || '') ? 0 : 2, tilt = +Pd.tilt_turns || 0.12;
      out.push(pawPath(s, reachClamp(s, toWorld(showPoint(s, p, 'pour'))), 0.6, {}));
      out.push({name: 'pour', dur: 2.2, start() { this.r0 = p.hrot[ax]; this.flow = !AFF(p).articulation; }, update(tt) {
        const k = mj(clamp(tt / 0.5, 0, 1)) * (1 - mj(clamp((tt - 1.6) / 0.5, 0, 1))); p.hrot[ax] = this.r0 + (ax === 2 ? -1 : 1) * tilt * k; if (this.flow) p.s = clamp((k - 0.6) / 0.4, 0, 1); return tt >= 2.1;
      }, end() { p.hrot[ax] = this.r0; if (this.flow) p.s = 0; emit({kind: 'object_event', name: 'poured', phase: 'end', src: 'auto', args: {object: p.id, on: 'floor'}, did: `poured ${Pd.what || 'it'} from the ${p.name} onto the floor`}); }});
      out.push(retractStep(s, true));
      void a; return out;
    }
    const twoPawWear = p => { const H = p.def.handlesFull; return !!(H && H.some(h => h.side === 'l') && H.some(h => h.side === 'r') && p.def.grip !== 'two_hand'); };   // a band with a cup each side (the headphones)
    const serT = T => T ? {rel: T.rel, prop: T.prop ? T.prop.id : null, place: T.place ? T.place.id : null, anchor: T.anchor || null, me: !!T.me} : null;
    const deT = o => o ? {rel: o.rel, prop: o.prop ? byId(o.prop) : null, place: o.place ? E.places.find(x => x.id === o.place) : null, anchor: o.anchor || null, me: !!o.me} : null;
    function aimPoint(p, T) {   // where a throw should come down (physical): into an open container, on a top, a free spot, in front of him
      if (!T || T.me) { const R0 = restPose(p, {me: true}); return R0.err ? R0 : {pos: R0.pos, id: 'floor', name: 'him', word: 'to him', sup: R0.sup, placeId: 'floor'}; }
      if (T.anchor) return {err: `I can't throw the ${p.name} onto my own body`};
      const rl = T.rel || 'on';
      if (T.prop && rl === 'in' && !contOpen(T.prop)) return {err: `the ${T.prop.name} is shut`};
      const R0 = restPose(p, Object.assign({}, T, {rel: rl}));
      if (R0.err) return R0;
      const nm = T.prop ? 'the ' + T.prop.name : T.place ? T.place.name : 'there';
      return {pos: R0.pos, id: T.prop ? T.prop.id : T.place ? T.place.id : null, name: nm, word: `${rl === 'in' ? 'into' : rl === 'on' ? 'onto' : REL_WORD[rl] || 'at'} ${nm}`, sup: R0.sup, placeId: R0.placeId};
    }
    // run a helper against the simulated world with its attachments too (supports, worn / balanced / anchored, paws)
    function withW2(W, fn) {
      return withW(W, () => {
        const sv = E.P.map(p => [p, p.sup, p.worn, p.bal, p.anc, p.held]);
        for (const p of E.P) { const w = W.P[p.id]; if (!w) continue; p.sup = w.on ? {id: w.on, inside: !!w.inside, rel: [0, 0, 0], rr: 0} : null; p.worn = !!w.worn; p.bal = w.bal ? (p.bal || {off: [0, 0.3, 0], t0: 0}) : null; p.anc = w.anc ? {name: w.anc} : null;
          p.held = W.held.l === p.id && W.held.r === p.id ? 'both' : W.held.l === p.id ? 'l' : W.held.r === p.id ? 'r' : null; }
        try { return fn(); } finally { for (const [p, a, b, c, d, e] of sv) { p.sup = a; p.worn = b; p.bal = c; p.anc = d; p.held = e; } }
      });
    }

    // ---------------------------------------------------------------- per frame
    let lastOutRoot = null;
    function base(o, dt) {   // before gestures: holding paws rest at their carry pose (gestures may move them, holding)
      if (!E.P.length) return;
      for (const s of ['l', 'r']) {
        const p = heldBy(s), pw = E.paw[s];
        const want = p && !p.flying ? 1 : 0;
        crit(pw, 'carryW', want, want ? 0.1 : 0.22, dt); pw.carryW = clamp(pw.carryW, 0, 1);
        if (pw.carryW < 1e-4 && !want) { pw.carryW = 0; continue; }
        const h = p ? carryHandCached(s, p) : pw.carryH || [0, 0, 0]; pw.carryH = h;
        const g = p ? graspValues(p, s) : {open: 0.2, turn: 0};
        const k = pw.carryW;
        o['hand_' + s + '_x'] = lerp(o['hand_' + s + '_x'], h[0] + 0.3 * o['hand_' + s + '_x'], k);
        o['hand_' + s + '_y'] = lerp(o['hand_' + s + '_y'], h[1] + 0.3 * o['hand_' + s + '_y'], k);
        o['hand_' + s + '_z'] = lerp(o['hand_' + s + '_z'], h[2] + 0.3 * o['hand_' + s + '_z'], k);
        o['hand_' + s + '_open'] = lerp(o['hand_' + s + '_open'], g.open, k);
        o['hand_' + s + '_turn'] = lerp(o['hand_' + s + '_turn'], g.turn, k);
      }
    }
    const carryCache = {};
    function carryHandCached(s, p) { const key = s + p.id + p.held; if (!carryCache[key]) carryCache[key] = carryHand(s, p); return carryCache[key]; }

    function apply(o, dt, lpRec) {
      const has = E.move || E.chain || E.queue || E.oqueue || E.active || E.P.length;
      if (!has) return null;
      const rec = lpRec ? [] : null;
      const setW = (p, v, w) => { if (!(p in o) || !(w > 0)) return; o[p] = lerp(o[p], v, w); if (rec) rec.push([p, w, v, 0]); };
      const addW = (p, v) => { if (!(p in o) || !v) return; o[p] += v; if (rec) rec.push([p, 0, 0, v]); };
      // queued commands start when nothing blocks them
      if (E.body.legs) museFrame(dt);
      if (!E.move && E.queue && !(E.chain)) { const q = E.queue; E.queue = null; runQueued(q); }
      if (!E.move && !E.chain && E.oqueue) { if (atRest(0.02)) { const q = E.oqueue; E.oqueue = null; runQueued(q); } else if (!E.move) settleHome(); }
      chainFrame();
      // ---- root
      const fr = moveFrame(o, dt), P = fr.P, tt = fr.tt;
      const r = fr.out;
      if (E.move === null && P === null && E.queue) { /* next frame */ }
      const R = rotm(r.yaw, r.pitch, r.roll);
      let lift = E.body.lift(R);
      // in the air a flip's root follows a soft-cornered lowest point (the near-hard corner kinks root_y at every
      // quarter turn); the true corner lift still applies if the body would touch the floor
      if (P && P.flipAir && E.body.liftSoft) lift = Math.max(E.body.liftSoft(R, 0.35), lift - Math.max(0, r.y));
      let px = 0, pz = 0;
      if (P && P.pivot && tt <= P.pivot.until) { const T = E.body.pivot(R, P.pivot.pe); px = T[0]; pz = T[2]; }
      else if (!P && (E.posture === 'lying')) { const pe = E.lying === 'side' ? [Math.sign(E.base.roll) * E.body.s[0] * 0.9, 0, E.body.E[2]] : [0, 0, E.body.E[2] + Math.sign(E.base.pitch) * E.body.s[2] * 0.9]; const T = E.body.pivot(R, pe); px = T[0]; pz = T[2]; }
      // the pivot shift fades in/out with the rotation itself, so it is continuous at both ends
      const bob = P && P.steps && P.steps.length && !P.sitOn && !P.unsit ? bobAt(P, tt) : 0;
      if (!E.bend) { E.bend = {y: 0, p: 0}; E.bendT = {y: 0, p: 0}; }
      if (E.body.legs) { crit(E.bend, 'y', E.bendT.y, 0.32, dt); crit(E.bend, 'p', E.bendT.p, 0.32, dt); }
      const rootOut = {x: r.x + px, y: r.y + lift + bob + (E.body.legs ? E.bend.y : 0), z: r.z + pz, yaw: r.yaw, pitch: r.pitch, roll: r.roll, squash: r.squash, stand: E.body.legs ? 0 : clamp(E.body.stand ? r.stand : 0, 0, 1)};
      E.rootOut = rootOut;
      if (P && P.drag) dragFrame(P, r);
      // velocities / accelerations (for residual starts and secondary motion)
      if (lastOutRoot && dt > 0) {
        for (const c of ROOT) {
          const v = (r[c] - lastOutRoot[c]) / dt;
          E.acc[c] = lerp(E.acc[c] || 0, (v - (E.vel[c] || 0)) / dt, 0.5); E.vel[c] = v;
        }
      }
      lastOutRoot = Object.assign({}, r);
      E.cur = Object.assign({}, r);
      const moving = !!P || !calmPosture() || !!E.body.legs || Math.abs(r.yaw) > 1e-4 || Math.abs(r.pitch) > 1e-4 || Math.abs(r.roll) > 1e-4 || Math.abs(r.x) > 1e-4 || Math.abs(r.z) > 1e-4 || Math.abs(r.squash) > 1e-4;
      if (!moving && !E.chain && !E.P.some(p => p.held || p.where !== 'home' || p.att.l || p.att.r) && E.w < 1e-4 && E.gf < 1e-4 && Math.abs(E.cam.zoom) < 1e-4 && Math.abs(E.cam.x) < 1e-4 && Math.abs(E.cam.y) < 1e-4 && E.objW < 1e-4) {
        E.active = false; E.lp = rec; return {active: false, pins: null};
      }
      E.active = true;
      o.root_x = rootOut.x; o.root_y = rootOut.y; o.root_z = rootOut.z;
      o.root_yaw = wrapH(rootOut.yaw); o.root_pitch = wrapH(rootOut.pitch); o.root_roll = wrapH(rootOut.roll);
      o.squash = clamp(rootOut.squash, -1, 1); o.stand = rootOut.stand;
      // ---- feet (legs): planted where they stand / stepping on the plan's footsteps / riding the body in the air
      if (E.body.legs) {
        if (!E.feet) initFeet();
        const wr = P && !(P.steps && P.steps.length) && !P.seatRide && !(E.sitOn && E.sitOn !== 'floor') ? ((r.y > 0.015 || Math.abs(wrap(r.pitch)) > 0.02 || Math.abs(wrap(r.roll)) > 0.02 || Math.abs(E.vel.yaw || 0) > 0.35) ? 1 : 0) : 0;
        E.ride = lerp(E.ride || 0, wr, 1 - Math.exp(-dt / (wr ? 0.03 : 0.05)));
        if (E.ride < 1e-3) E.ride = 0;
        E.footNow = E.footNow || {};
        const chairSit = (!P && E.sitOn && E.sitOn !== 'floor') || (P && P.seatRide && tt >= P.seatRide[0]) || (P && !P.seatOff && !P.seatRide && E.sitOn && E.sitOn !== 'floor');
        const sT = !chairSit ? 0 : P && P.seatRide ? mj(clamp((tt - P.seatRide[0]) / (P.seatRide[1] - P.seatRide[0]), 0, 1)) : 1;
        E.seatW = sT;
        for (const sd of ['l', 'r']) {
          const restW = toWorldR(rootOut, footRest(sd));
          let F = P && P.steps && P.steps.length ? footAtPlan(P, sd, tt) : E.feet[sd];
          if (E.seatW > 0) {   // seated on a chair: the thighs along the seat, the shins hang (body.json sitting pose)
            const sf = (E.body.seatFeet && E.body.seatFeet[sd]) || [0, 0.07, 0.08], hang = toWorldR(rootOut, add3(footRest(sd), sf));
            F = lerp3(F, hang, E.seatW); E.feet[sd] = F.slice();
          }
          if (E.ride > 0) { F = lerp3(F, restW, E.ride); if (!(P && P.steps && P.steps.length)) E.feet[sd] = [F[0], Math.max(0, F[1] - (r.y > 0.015 ? r.y : 0)), F[2]]; }
          if (E.ride > 0 && !(P && P.steps && P.steps.length) && r.y <= 0.015) E.feet[sd][1] = 0;
          E.footNow[sd] = F;
          o['foot_' + sd + '_x'] = F[0] - restW[0]; o['foot_' + sd + '_y'] = F[1] - restW[1]; o['foot_' + sd + '_z'] = F[2] - restW[2];
        }
      }
      // ---- move layer weight (full-body channels) and head weight
      const wT = P && !P.silent ? 1 : P ? 0.5 : (!calmPosture() ? 1 : 0);
      crit(E, 'w', wT, wT > E.w ? 0.12 : 0.25, dt);
      const hT = P ? (P.headW ?? 0.75) * (P.whole ? 1 : 0.6) : (!calmPosture() || E.facingHold ? 0.75 : 0);
      crit(E, 'wHead', hT, hT > E.wHead ? 0.1 : 0.3, dt);
      // ---- move extras (full-body sets/adds); paws busy with a prop keep their grasp/carry
      const pawBusy = s => E.paw[s].ctrlT > 0 || E.paw[s].ctrl > 0.01 || !!heldBy(s);
      if (P) {
        for (const x of P.x) {
          // legs (the muse): every extra has faded out by the plan's end, so nothing drops when the move layer lets go
          const xr = E.body.legs ? Math.max(x.rel, 0.4) : x.rel, xt1 = E.body.legs && !P.holdEnd && P.dur ? Math.max(x.t0 + 0.05, Math.min(x.t1, P.dur - xr)) : x.t1;
          if (tt < x.t0 - 1e-6 || tt > xt1 + xr) continue;
          const xa = E.body.legs && x.att > 0 ? Math.max(x.att, 0.3) : x.att;   // legs: arms rise over ≥ 0.3 s (no flicks)
          let w = xa > 0 ? mj((tt - x.t0) / xa) : 1;
          if (xt1 !== Infinity && tt > xt1) w = Math.min(w, 1 - mj((tt - xt1) / Math.max(1e-3, xr)));
          if (w <= 0) continue;
          const v = x.fn(Math.min(tt, x.t1 === Infinity ? tt : x.t1));
          if (!v) continue;
          const ww = w * (v.w ?? 1);
          for (const k in (v.set || {})) { const sd = /^(hand_|thumb_|point_|shoulder_|fingers_)([lr])/.exec(k) || /_([lr])$/.exec(k); const s = sd && (sd[2] || sd[1]); if (s && /^(hand_|thumb_|point_|fingers_)/.test(k) && pawBusy(s)) continue; setW(k, v.set[k], ww); }
          for (const k in (v.add || {})) addW(k, v.add[k] * ww);
        }
      } else if (E.posture === 'lying' || E.posture === 'tall' || E.posture === 'crouched') {
        // a held posture keeps its full-body pose (the extras of the move that made it are kept by the plan; nothing else)
      }
      // whole-body moves hold the torso still (body reflexes — flinch, lean-in, weight shifts — are suppressed)
      const wb = E.w * (P ? (P.whole === false ? 0.3 : 1) : 0.9);
      if (wb > 0) for (const k of ['body_yaw', 'body_lean', 'body_pitch', 'body_bounce']) setW(k, 0, 0.85 * wb * (P && P.x.length ? 1 : 1));
      // ---- secondary motion: ears / tail / paws lag the root's accelerations; the head leads turns
      const sec = E.sec, ay = clamp(rootAcc('y') * 1 + 4 * rootAcc('squash') * 0.1, -80, 80), ayaw = clamp(rootAcc('yaw') * TAU, -200, 200), aroll = clamp(rootAcc('roll') * TAU, -200, 200);
      const sp2 = (x, v, target, hz, z) => { const w = TAU * hz; const n = Math.max(1, Math.ceil(dt / 0.008)), h = dt / n; for (let i = 0; i < n; i++) { v += (w * w * (target - x) - 2 * z * w * v) * h; x += v * h; } return [x, v]; };
      if (E.w > 1e-3 || Math.abs(sec.earL) > 1e-4 || Math.abs(sec.tail) > 1e-4) {
        [sec.earL, sec.earLv] = sp2(sec.earL, sec.earLv, clamp(-0.018 * ay - 0.004 * aroll, -0.8, 0.8), 2.4, 0.32);
        [sec.earR, sec.earRv] = sp2(sec.earR, sec.earRv, clamp(-0.018 * ay + 0.004 * aroll, -0.8, 0.8), 2.2, 0.32);
        [sec.tail, sec.tailv] = sp2(sec.tail, sec.tailv, clamp(-0.006 * ayaw - 0.004 * aroll, -0.9, 0.9), 1.6, 0.35);
        [sec.hand, sec.handv] = sp2(sec.hand, sec.handv, clamp(-0.006 * ay, -0.35, 0.35), 3.0, 0.5);
        [sec.tailR, sec.tailRv] = sp2(sec.tailR, sec.tailRv, clamp(-0.008 * ay, -0.5, 0.5), 1.8, 0.4);
        const k = Math.max(E.w, 0.0);
        addW('ear_l', sec.earL * k); addW('ear_r', sec.earR * k); addW('tail_wag', sec.tail * k); addW('tail_raise', sec.tailR * k);
        for (const s of ['l', 'r']) if (!pawBusy(s)) addW('hand_' + s + '_y', sec.hand * k);
      }
      // head: attention suppressed while the whole body moves; it leads turns (looks where the body will be)
      if (!(E.wHead > 1e-3)) { E.hLeadY = 0; E.hLeadYV = 0; }
      if (E.wHead > 1e-3) {
        const k = E.wHead;
        o.head_yaw *= 1 - 0.85 * k; o.head_pitch *= 1 - 0.6 * k; o.head_roll *= 1 - 0.4 * k;
        if (P) {
          const la = 0.18, yF = P.T.yaw.at(tt + la) - P.T.yaw.at(tt), pF = P.T.pitch.at(tt + la) - P.T.pitch.at(tt), rF = P.T.roll.at(tt + la) - P.T.roll.at(tt);
          // the yaw lead runs through a critically damped follower: a turn that starts hard (a dance's spin step) never
          // whips the head faster than the turn itself
          o.head_yaw += crit(E, 'hLeadY', clamp(0.9 * TAU * yF, -0.5, 0.5), 0.15, dt) * k;
          o.head_pitch += clamp(0.5 * TAU * pF, -0.3, 0.3) * k;
          o.head_roll += clamp(0.4 * TAU * rF, -0.25, 0.25) * k;
        }
      }
      // turn back (seated / at work): the upper body + head twist back toward the camera; a glance up while typing = the head only
      const A0 = E.activity, glancing = A0 && !A0.paused && t() < (A0.glanceEnd || 0);
      const tbT = E.turnBack ? 1 : glancing ? 1 : 0;
      // (c2) a turn back takes ~1 s (it was ~3 s, so a keyframe at the end of the command still showed the profile); the
      // waist takes up to its full twist, the head the rest (to ±0.75 rad), the eyes whatever is left
      crit(E, 'tbW', tbT, glancing ? 0.32 : E.turnBack ? 0.28 : 0.4, dt); if (Math.abs(E.tbW) < 1e-4 && !tbT) E.tbW = 0;
      if (tbT) {
        const need = -wrap(rootOut.yaw) * TAU, by = E.turnBack ? clamp(need * 0.55 / 0.4, -1, 1) : 0, aim = {by, hy: clamp(need - by * 0.4, -0.75, 0.75), k: glancing && !E.turnBack ? 0.75 : 1};
        // a turn back that starts during a glance changes the aim under a weight that is already up — ease the aim
        // itself (it jumped: a one-frame head_yaw pop)
        if (!E.tbAim || E.tbW < 0.05) E.tbAim = aim; else { const g = 1 - Math.exp(-dt / 0.18); for (const q of ['by', 'hy', 'k']) E.tbAim[q] += (aim[q] - E.tbAim[q]) * g; }
      }
      if (E.tbW > 0 && E.tbAim) {   // (the aim is kept while it eases back, so nothing snaps)
        if (E.tbAim.by) setW('body_yaw', E.tbAim.by, clamp(E.tbW, 0, 1));
        setW('head_yaw', E.tbAim.hy + 0.3 * o.head_yaw, clamp(E.tbW, 0, 1) * E.tbAim.k);
      }
      // ---- gaze: eyes stop seeking the camera when the face can't see it (facing), or when told (free / turned away)
      const eY = (o.body_yaw || 0) * 0.35 + o.head_yaw, fz = R[2][0] * Math.sin(eY) + R[2][2] * Math.cos(eY);   // ≈ how much the face points at the camera
      let gfT = sstep(0.82, 0.5, fz);
      const gm = P ? P.gaze : (E.facingHold === 'away' ? 'free' : 'user');
      if (gm === 'free' || gm === 'forward') gfT = Math.max(gfT, P ? mj(clamp(tt / 0.3, 0, 1)) * (P.holdEnd || tt < P.dur - 0.35 ? 1 : 0) : 1);
      // looking at a prop (pick up, inspect, put down …)
      { const lT = E.look.wT && E.look.P ? 1 : 0; crit(E.look, 'w', lT, lT > E.look.w ? (E.activity ? 0.3 : 0.16) : 0.32, dt); }
      gfT = Math.max(gfT, E.look.w);
      crit(E, 'gf', gfT, gfT > E.gf ? 0.09 : 0.2, dt); E.gf = clamp(E.gf, 0, 1);
      if (E.gf < 1e-4 && !gfT) E.gf = 0;
      let gx = 0, gy = 0;
      if (E.look.w > 1e-3 && E.look.P) {
        const hp = headPoint(), d = sub3(E.look.P, hp);
        const yaw = Math.atan2(d[0], Math.max(0.02, d[2])), pitchDown = Math.atan2(-d[1], Math.hypot(d[0], d[2]));
        const hy = clamp(0.75 * yaw, -0.6, 0.6), hpch = clamp(0.7 * pitchDown, -0.4, 0.45);
        const lw = E.look.w;
        setW('head_yaw', hy, 0.9 * lw); setW('head_pitch', hpch, 0.9 * lw);
        gx = clamp(Math.tan(yaw - hy) / 0.3, -1, 1); gy = clamp(-Math.tan(pitchDown - hpch) / 0.24, -1, 1);
        addW('brow_l', 0.15 * lw); addW('brow_r', 0.15 * lw); addW('ear_l', 0.15 * lw); addW('ear_r', 0.15 * lw);
      }
      if (E.gf > 0) {
        const lw = E.look.w;
        const fx = lerp(0, gx, lw), fy = lerp(0, gy, lw);
        o.gaze_x = lerp(o.gaze_x, fx + 0.05 * o.gaze_x, E.gf); o.gaze_y = lerp(o.gaze_y, fy + 0.05 * o.gaze_y, E.gf);
        if (rec) { rec.push(['gaze_x', E.gf, fx, 0]); rec.push(['gaze_y', E.gf, fy, 0]); }
      }
      o.gaze_free = E.gf;
      const ctrlMax = Math.max(E.paw.l.ctrl, E.paw.r.ctrl);
      crit(E, 'bodyLock', ctrlMax > 0 ? 0.9 * ctrlMax : 0, 0.22, dt); if (E.bodyLock < 1e-4 && !(ctrlMax > 0)) E.bodyLock = 0;   // eased, never a snap
      if (E.bodyLock > 0) for (const k of ['body_yaw', 'body_lean', 'body_pitch', 'body_bounce']) setW(k, 0, clamp(E.bodyLock, 0, 0.9));
      if (E.body.legs && E.bend && Math.abs(E.bend.p) > 1e-5) o.body_pitch = (o.body_pitch || 0) + E.bend.p;
      // ---- paws driven by object steps (world target → hand target through the inverse reach map)
      const pins = {};
      for (const s of ['l', 'r']) {
        const pw = E.paw[s];
        crit(pw, 'ctrl', pw.ctrlT, pw.ctrlT > pw.ctrl ? 0.08 : 0.14, dt); pw.ctrl = clamp(pw.ctrl, 0, 1);
        if (pw.ctrlT === 0 && pw.ctrl < 1e-3) { pw.ctrl = 0; pw.ctrlV = 0;  pw.hLast = null; }
        const held = heldBy(s);
        if (pw.ctrl > 0 && pw.tgt) {
          if (pw.follow) pw.tgt = pw.follow();
          if (pw.hd && pw.hdFor === pw.tgt) { pw.h = pw.hd.slice(); pw.err = 0; }
          else { const r2 = invW(s, pw.tgt, pw.h, 6); pw.h = r2.h; pw.err = r2.err; }
          if (pw.hLast && dt > 0) {   // slew limit: an inverse that hops to another branch / a target out of reach never pops the arm
            const lim = 4.5 * dt; for (let a = 0; a < 3; a++) { const d = pw.h[a] - pw.hLast[a]; if (Math.abs(d) > lim) pw.h[a] = pw.hLast[a] + Math.sign(d) * lim; }
          }
          pw.hLast = pw.h.slice();
          const k = pw.ctrl;
          setW('hand_' + s + '_x', pw.h[0], k); setW('hand_' + s + '_y', pw.h[1], k); setW('hand_' + s + '_z', pw.h[2], k);
          setW('hand_' + s + '_open', pw.open, k); setW('hand_' + s + '_turn', pw.turn, k); setW('thumb_' + s, pw.thumb, k); setW('point_' + s, pw.point || 0, k);
          pins['hand_' + s + '_x'] = k; pins['hand_' + s + '_y'] = k; pins['hand_' + s + '_z'] = k;
        } else if (!pw.ctrl) { pw.open = o['hand_' + s + '_open']; pw.turn = o['hand_' + s + '_turn']; pw.thumb = o['thumb_' + s]; }
        if (held && !held.flying) {   // the grasp closure wins over any gesture on a holding paw
          // eased in as the step control hands the paw back (1 − ctrl), never a one-frame switch when ctrl reaches 0
          const g = graspValues(held, s), k = clamp(1 - pw.ctrl, 0, 1);
          if (k > 0) { setW('hand_' + s + '_open', g.open, k); setW('point_' + s, 0, k); setW('thumb_' + s, g.thumb, k); setW('fingers_' + s, 0, k); }
        }
        pw.hPrev = [o['hand_' + s + '_x'], o['hand_' + s + '_y'], o['hand_' + s + '_z']];   // refined in post()
      }
      objFrame(dt);   // objects engine: eases, flights, the head, catches
      // ---- props
      let anyProp = 0;
      for (const p of E.P) {
        const w = clamp(p.att.l + p.att.r + (p.hh || 0), 0, 1), fo = freeOff(p), k = 'prop' + p.k + '_';
        o[k + 'hh'] = p.hh || 0; o[k + 's'] = p.s || 0;
        const hx = p.hoff;
        o[k + 'x'] = (1 - w) * fo[0] + w * hx[0]; o[k + 'y'] = (1 - w) * fo[1] + w * hx[1]; o[k + 'z'] = (1 - w) * fo[2] + w * hx[2];
        const rf = p.rot, rh = p.hrot;
        o[k + 'rx'] = wrapH(rf[0] + wrap(rh[0] - rf[0]) * w); o[k + 'ry'] = wrapH(rf[1] + wrap(rh[1] - rf[1]) * w); o[k + 'rz'] = wrapH(rf[2] + wrap(rh[2] - rf[2]) * w);
        o[k + 'hl'] = p.att.l; o[k + 'hr'] = p.att.r;
        anyProp += Math.abs(o[k + 'x']) + Math.abs(o[k + 'y']) + Math.abs(o[k + 'z']) + w + (p.s || 0);
      }
      E.objW = anyProp;
      E.twOut = ((o.body_yaw || 0) * 0.4 + (o.head_yaw || 0)) / TAU;   // the upper body + head twist over the hips (turns), as drawn
      // ---- camera: pull out for big moves, follow the root, come back after
      const zT = P ? (P.zoom || 0) * (tt < P.dur - 0.2 || P.holdEnd ? 1 : 0.3) : (E.posture === 'lying' ? 0.5 : !calmPosture() ? 0.3 : E.body.legs ? 0.3 : 0);
      const C = E.cam;
      [C.zoom, C.zv] = sp2(C.zoom, C.zv, zT, 1.1, 1.0);
      [C.x, C.xv] = sp2(C.x, C.xv, 0.75 * rootOut.x, 1.0, 1.0);
      [C.y, C.yv] = sp2(C.y, C.yv, (E.body.legs ? 0.75 : 0.45) * Math.max(0, rootOut.y - 0.04) + 0.25 * rootOut.stand, 1.2, 1.0);
      if (Math.abs(C.zoom) < 1e-5 && Math.abs(C.zv) < 1e-5) { C.zoom = 0; C.zv = 0; }
      if (Math.abs(C.x) < 1e-5 && Math.abs(C.xv) < 1e-5) { C.x = 0; C.xv = 0; }
      if (Math.abs(C.y) < 1e-5 && Math.abs(C.yv) < 1e-5) { C.y = 0; C.yv = 0; }
      o.cam_zoom = clamp(C.zoom, 0, 1); o.cam_x = C.x; o.cam_y = C.y;
      // scene: a station view (setView) blends the camera toward that place's camera (critically damped, ~1.5 s);
      // the follow camera above keeps running underneath, so letting go eases back to it
      // (an early eval run: "hold the eraser up so I can see it" was shown in a wide shot across the room — a 4 cm eraser
      // invisible at 540p): while it holds a thing up to show / offer / look at, a temporary station view eases in on it
      // (the same view blend the board / TV use, so every camera path honours it); the previous view comes back after
      { const Cc = E.chain, cn = Cc && Cc.cur ? String(Cc.cur.name || '') : '', vb = Cc ? String((Cc.args && Cc.args.verb) || (Cc.op && Cc.op.cmd && Cc.op.cmd.verb) || Cc.verb || '') : '';
        const showing = !!Cc && (SHOWY.test(cn) || (SHOWY.test(vb) && cn !== 'retract'));
        if (showing) E.showUntil = t() + 0.5;   // held a moment after the thing goes down (1.5 -> 0.5 s: the close-up then showed only the lowered fist)
        const RV = !showing && E.resultView && t() < E.resultView.until ? E.resultView : null;   // the placed result
        if (RV) {
          const q = RV.id ? byId(RV.id) : null, Pw = RV.P || (q && (q.wpos || q.pos));
          if (Pw && Number.isFinite(Pw[0])) {
            if (E.view !== '__show') { E.viewPrev = {view: E.view, cam: E.viewCam}; E.view = '__show'; }
            const pv = E.viewCam && E.view === '__show' ? E.viewCam : null, ke = pv ? 0.12 : 1, ax = 0.9 * Pw[0], ay = Pw[1] - 0.4;   // (the show aim: Y - 0.45 at the paw; a little above the thing so its support shows)
            E.viewCam = {zoom: RV.P ? -0.6 : -0.75, x: pv ? pv.x + (ax - pv.x) * ke : ax, y: pv ? pv.y + (ay - pv.y) * ke : ay};
          }
        } else if (showing || t() < (E.showUntil || 0)) {
          if (E.view !== '__show') { E.viewPrev = {view: E.view, cam: E.viewCam}; E.view = '__show'; }
          // aim between the face and the shown thing and push in harder for a small one (the eraser at
          // -0.6 aimed at the body's middle stayed a grey sliver at 540p)
          let aimX = rootOut.x, aimY = 0.12 + 0.45 * Math.max(0, rootOut.y - 0.04), zoom = SHOW_ZOOM;
          try {
            const J = Cc && Cc.opOf, q = (J && jobProp(J)) || E.P.find(r => r.held), sd = q && q.held ? (q.held === 'both' ? 'l' : q.held) : null;
            if (q && sd) {
              const Pw = fwdW(sd, E.paw[sd].h), Hw = toWorld(headPoint());
              // a tiny thing (chalk, eraser, ball: a fist's size) is the subject — the aim leans to the paw
              // and the push-in goes all the way (between face and paw at -0.85 it stayed "a sliver tucked in a fist")
              const wP = q.size < 0.06 ? 0.7 : 0.5;
              aimX = wP * Pw[0] + (1 - wP) * Hw[0]; aimY = wP * Pw[1] + (1 - wP) * Hw[1] - 0.45;
              zoom = q.size < 0.06 ? SHOW_ZOOM_TINY : q.size < 0.12 ? SHOW_ZOOM_SMALL : q.size < 0.25 ? SHOW_ZOOM : -0.4;
            }
          } catch (e) { /* the default show framing */ }
          if (!Number.isFinite(aimX) || !Number.isFinite(aimY)) { aimX = rootOut.x; aimY = 0.12; }
          const pv = E.viewCam && E.view === '__show' ? E.viewCam : null, ke = pv ? 0.12 : 1;   // the aim follows the paw smoothly
          E.viewCam = {zoom, x: pv ? pv.x + (0.9 * aimX - pv.x) * ke : 0.9 * aimX, y: pv ? pv.y + (aimY - pv.y) * ke : aimY};
        } else if (E.view === '__show') { const pv = E.viewPrev || {}; E.view = pv.view || null; E.viewCam = pv.cam || null; E.viewPrev = null; }
      }
      [E.viewW, E.viewWv] = sp2(E.viewW, E.viewWv, E.view ? 1 : 0, 0.55, 1.0);
      if (E.viewW < 1e-4 && !E.view) { E.viewW = 0; E.viewWv = 0; }
      if (E.viewW > 0 && E.viewCam) {
        const V = E.viewCam, k = clamp(E.viewW, 0, 1);
        o.cam_zoom = clamp(o.cam_zoom + (V.zoom - o.cam_zoom) * k, -1, 1); o.cam_x += (V.x - o.cam_x) * k; o.cam_y += (V.y - o.cam_y) * k;
      }
      // affect: a strong feeling eases the home camera in a little so the face reads at stream sizes (never during
      // a move, a station / show view or object work: affectCam(0) then); slow both ways (0.5 Hz critically damped, ~1.5 s), at most AFF_ZOOM
      { const free = !P && !E.view && !E.chain && !E.jobs.length;
        [E.affZ, E.affZv] = sp2(E.affZ || 0, E.affZv || 0, free ? clamp(+E.affCamT || 0, 0, 1) : 0, 0.5, 1.0);
        if (Math.abs(E.affZ) < 1e-5 && Math.abs(E.affZv) < 1e-5) { E.affZ = 0; E.affZv = 0; }
        if (E.affZ) { o.cam_zoom = clamp(o.cam_zoom - AFF_ZOOM * E.affZ * (1 - clamp(E.viewW, 0, 1)), -1, 1); o.cam_y += AFF_CAM_Y * E.affZ * (1 - clamp(E.viewW, 0, 1)); } }
      if (rec) for (const kk of ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'stand', 'gaze_free', 'cam_zoom', 'cam_x', 'cam_y']) if (o[kk]) rec.push([kk, 1, o[kk], 0]);
      E.lp = rec;
      return {active: true, pins};
    }

    // ---------------------------------------------------------------- state / busy
    function stateOut() {
      const P = E.move, tt = P ? t() - P.t0 : 0;
      let posture = E.posture === 'tall' || E.posture === 'crouched' ? 'seated' : E.posture;
      if (P && P.lands && P.lands.some(x => x > tt) && E.cur.y > 0.02) posture = 'airborne';
      else if (P && !P.holdEnd && !P.silent) posture = E.posture === 'lying' ? 'lying' : (P.name === 'walk' || P.name === 'step' || P.name === 'scoot' || P.name === 'come_back') ? 'moving' : (P.name === 'dance' ? 'dancing' : (Math.abs(E.cur.yaw) > 0.03 && (P.name === 'spin' || P.name === 'twirl') ? 'spinning' : posture));
      // facing = where its upper body + face point (a seated turn-back twists them toward him over the hips)
      // seated, the face = the hips' yaw + the drawn waist twist + head yaw ('camera' only when that is within ~30° of him)
      const yB = wrap(E.cur.yaw), tw = E.sitOn && E.body.legs ? (E.twOut || 0) : 0;
      const y = wrap(yB + tw), ay = Math.abs(y);
      // lying on the rug while the lie-down / roll-over is still running (the posture used to change only when
      // the move ended — never, while held — so the action memory read 'standing')
      if (P && /^(lie_down|roll_over)$/.test(P.name) && posture !== 'airborne' && (Math.abs(wrap(E.cur.roll)) > 0.12 || Math.abs(wrap(E.cur.pitch)) > 0.12)) posture = 'lying';
      const facing = Math.abs(wrap(E.cur.pitch)) > 0.15 && posture !== 'lying' || (Math.abs(wrap(E.cur.roll)) > 0.15 && E.posture !== 'lying' && posture !== 'lying') ? 'tumbling' : ay < 0.08 ? 'camera' : ay > 0.38 ? 'away' : (y > 0 ? 'left' : 'right');
      const holding = {left: null, right: null};
      for (const p of E.P) { if (p.held === 'r' || p.held === 'both') holding.left = p.id; if (p.held === 'l' || p.held === 'both') holding.right = p.id; }
      const active = [];
      if (P && !P.silent) active.push({kind: 'move', name: P.name, since: +P.t0.toFixed(2), held: !!P.holdEnd && tt >= P.dur});
      if (E.chain) active.push({kind: 'object', name: E.chain.verb, since: +E.chain.started.toFixed(2), object: E.chain.p.id});
      const desk = E.P.find(q => q.F && q.F.desk), chair = E.P.find(q => q.F && q.F.chair);
      const chairAtDesk = desk && chair ? d2(fW(desk, desk.F.desk.chairSpot), chair.pos) < 0.12 : false;
      const A = E.activity;
      const muse = E.body.legs ? {sittingOn: E.sitOn || null, atDesk: !!(E.sitOn && chair && E.sitOn === chair.id && chairAtDesk), chairAtDesk,
        wearing: E.P.filter(q => q.worn).map(q => q.id), open: E.P.filter(q => q.F && q.F.articulated && (q.s || 0) > 0.5).map(q => q.id),
        activity: A ? (A.paused ? `${A.kind} (paused — turned back to him)` : A.kind) : null, turnedBack: !!E.turnBack, queued: E.jobs.length,
        station: E.teachWhere || null, teaching: E.teachDoing || null,   // teach mode: "at the board", "writing on the board"
        facingBody: Math.abs(yB) < 0.08 ? 'camera' : Math.abs(yB) > 0.38 ? 'away' : (yB > 0 ? 'left' : 'right')} : {};
      return {...muse, posture, lying: E.posture === 'lying' ? E.lying : null, standing: E.posture === 'standing', tall: E.posture === 'tall', facing, gazeFree: E.gf > 0.5,
        holding, props: E.P.map(p => ({id: p.id, name: p.name, slot: p.k, where: propWhere(p), held: p.held ? (p.held === 'both' ? 'both' : sideWord(p.held)) : null})), active,
        frozen: E.frozen, root: {x: +E.cur.x.toFixed(3), y: +E.cur.y.toFixed(3), z: +E.cur.z.toFixed(3), yaw: +wrap(E.cur.yaw).toFixed(3), pitch: +wrap(E.cur.pitch).toFixed(3), roll: +wrap(E.cur.roll).toFixed(3)}};
    }
    function busy() {
      const P = E.move, mv = !!(P && !P.silent && (!P.holdEnd || true));
      const pb = s => E.paw[s].ctrlT > 0 || E.paw[s].ctrl > 0.01;
      return {l: pb('l') || (mv && P.x.length > 0), r: pb('r') || (mv && P.x.length > 0), body: mv || !calmPosture() || E.frozen || !!(E.activity && !E.activity.paused), head: mv && (P.whole !== false) || E.look.w > 0.2,
        holdL: !!heldBy('l'), holdR: !!heldBy('r')};
    }
    function wholeBody() { const P = E.move; return !!(P && !P.silent && P.whole !== false) || E.posture === 'lying' || !!(P && P.legs); }

    // ================================================================ the muse (a character with legs): gait, furniture,
    // worn items, the laptop, an activity loop, and a small planner over the world state (contract §0)
    const rotYv = (v, a) => { const c = Math.cos(a * TAU), sn = Math.sin(a * TAU); return [v[0] * c + v[2] * sn, v[1], -v[0] * sn + v[2] * c]; };
    const yawOfDir = d => Math.atan2(d[0], d[2]) / TAU;
    const flat = v => [v[0], 0, v[2]];
    const norm2 = v => { const l = Math.hypot(v[0], v[2]) || 1; return [v[0] / l, 0, v[2] / l]; };
    const d2 = (a, b) => Math.hypot(a[0] - b[0], a[2] - b[2]);
    // ---- prop features: explicit JSON fields first, else defaults from the renderer catalog kind + dims
    function feat(p) {
      if (p.F) return p.F;
      const d = p.def, kind = String(d.kind || d.category || '').toLowerCase(), key = kind + ' ' + p.id, dims = Array.isArray(d.dims) ? d.dims.map(Number) : null;
      const F = {kind};
      F.furniture = !!(d.furniture || d.free === true || /^(chair|desk|table|furniture|stool|sofa)$/.test(kind));
      F.wearable = !!(d.worn || d.wearable || d.worn_pose || /headphone|headset|hat|cap|crown|glasses/.test(key));
      if (F.furniture && /chair|stool|seat/.test(key)) {
        const w = dims ? dims[0] : 0.3, dep = dims ? dims[1] : 0.3, sd = d.seat || {}, sh = +sd.top_y || +sd.height || (dims ? dims[2] : 0.2), th = dims && dims[3] ? dims[3] : sh * 2.2;
        F.chair = {seat: v3(sd.center) || [0, sh, 0], top: sh, front: v3(sd.front) || v3(d.front) || [0, 0, 1], w, dep};
        F.handles = Array.isArray(d.handles) && d.handles.length >= 2 ? d.handles.map(v3) : [[-0.35 * w, th, -0.5 * dep], [0.35 * w, th, -0.5 * dep]];
      } else if (F.furniture) {
        const w = +d.width || (dims ? dims[0] : 0.7), dep = +d.depth || (dims ? dims[1] : 0.4), h = dims ? dims[2] : 0.42;
        F.desk = {w, dep, h, front: v3(d.front) || [0, 0, 1], chairSpot: v3(d.chair_spot) || [0, 0, dep / 2 + 0.26]};
        F.handles = Array.isArray(d.handles) && d.handles.length >= 2 ? d.handles.map(v3) : [[-0.4 * w, h, 0.5 * dep], [0.4 * w, h, 0.5 * dep]];
      }
      if (/laptop|computer|notebook/.test(key)) {
        const sc = p.size, Ld = d.lid || {};
        F.laptop = {hinge: v3(Ld.hinge) || [0, 0.08 * sc, -0.7 * sc], len: +Ld.length || 1.4 * sc, open: (+Ld.open_deg || 110) * Math.PI / 180,
          kb: v3(d.keyboard) || [0, 0.09 * sc, 0.15 * sc], screen: v3(d.screen) || null, half: +Ld.half_width || 0.9 * sc};
        F.articulated = true;
      }
      if (d.articulation || d.lid || /box|book/.test(kind)) F.articulated = true;
      return (p.F = F);
    }
    const ryOf = p => p.homeRot[1] + p.rot[1];
    const fW = (p, pl) => add3(p.pos, rotYv(pl, ryOf(p)));     // prop-local → world (about its centre / floor point)
    const byId = id => E.P.find(p => p.id === id);
    const isFurn = p => feat(p).furniture;
    // places that sit on a piece of furniture move with it
    function syncPlaces() {
      for (const pl of E.places) {
        const fid = pl.prop || pl.on || pl.attach; if (!fid) continue;
        const f = byId(fid); if (!f) continue;
        if (!pl.c0) { pl.c0 = pl.center.slice(); pl.s0 = pl.spots.map(q => q.slice()); }
        const dr = f.rot[1], tr = q => add3(f.pos, rotYv(sub3(q, f.home), dr));
        pl.center = tr(pl.c0); pl.spots = pl.s0.map(tr); pl.yaw = dr;
      }
    }
    function placeOfFurn(f) { return E.places.find(pl => (pl.prop || pl.on || pl.attach) === f.id); }
    function onPlace(pl) { return E.P.filter(q => !q.held && !q.worn && !q.flying && q.place === pl.id && !isFurn(q)); }
    // ---- feet (world) and the gait
    const footRest = s => E.body.legs.foot[s];
    function idealFoot(s, pose) { return toWorldR({x: pose.x, y: 0, z: pose.z, yaw: pose.yaw, pitch: 0, roll: 0}, footRest(s)); }
    function initFeet() { const b = E.base; E.feet = {l: idealFoot('l', b), r: idealFoot('r', b)}; E.footNow = {l: E.feet.l.slice(), r: E.feet.r.slice()}; }
    function planSteps(P, ta, tb, Ts) {
      if (!P.steps) P.steps = [];
      const cur = P.footCur || (P.footCur = {l: (P.footStart || E.feet).l.slice(), r: (P.footStart || E.feet).r.slice()});
      const pose = tt => ({x: P.T.x.at(tt), z: P.T.z.at(tt), yaw: P.T.yaw.at(tt)});
      let tt = ta;
      const want = (s, at) => idealFoot(s, pose(Math.min(tb, at)));
      let side = len3(sub3(want('l', ta + Ts), cur.l)) >= len3(sub3(want('r', ta + Ts), cur.r)) ? 'l' : 'r';
      for (let n = 0; tt < tb - 0.02 && n < 120; tt += Ts, n++) {
        const tLand = tt + 0.72 * Ts, tgt = want(side, tLand + 0.55 * Ts), dd = len3(sub3(tgt, cur[side]));
        if (dd > 0.006) { P.steps.push({s: side, t0: tt, t1: tLand, from: cur[side].slice(), to: tgt, h: clamp(0.022 + 0.22 * dd, 0.02, 0.07)}); cur[side] = tgt; }
        side = side === 'l' ? 'r' : 'l';
      }
      for (let k = 0; k < 2; k++) {   // close the stance under the final pose
        const tgt = want(side, tb), dd = len3(sub3(tgt, cur[side]));
        if (dd > 0.008) { P.steps.push({s: side, t0: tt, t1: tt + 0.72 * Ts, from: cur[side].slice(), to: tgt, h: clamp(0.02 + 0.2 * dd, 0.02, 0.05)}); cur[side] = tgt; tt += Ts; }
        side = side === 'l' ? 'r' : 'l';
      }
      return Math.max(tb, tt);
    }
    function footAtPlan(P, s, tt) {
      let pos = (P.footStart || E.feet)[s];
      for (const st of P.steps) {
        if (st.s !== s || st.t0 > tt) continue;
        if (tt >= st.t1) { pos = st.to; continue; }
        const u = (tt - st.t0) / (st.t1 - st.t0), m = mj(u), q = lerp3(st.from, st.to, m);
        pos = [q[0], q[1] + st.h * Math.sin(Math.PI * u) ** 1.2, q[2]];
      }
      return pos;
    }
    // walk to a pose: turn toward the way (if it is far), go, turn to face; the feet plant step by step
    function legWalk(P, goal, sp, opt = {}) {
      const L = E.body.legs, Ts = (opt.Ts || 0.46) / sp, SL = Math.min(L ? L.maxStep : 0.2, 0.2);
      const t0 = P.sync(), dx = goal.x - P.T.x.v, dz = goal.z - P.T.z.v, dist = Math.hypot(dx, dz);
      const yawTo = a => { const cur = P.T.yaw.v, tgt = cur + wrap(a - cur), d = Math.abs(tgt - cur); if (d < 0.008) return; const n = Math.max(1, Math.ceil(d / (opt.turnStep || 0.1))); P.sync(); P.T.yaw.to(tgt, n * Ts, u => cruise(u, 0.4)); P.sync(); };
      const face = goal.yaw ?? P.T.yaw.v;
      const turnFirst = dist > 0.35 && !opt.keepFacing && !opt.blendYaw;
      if (turnFirst) yawTo(yawOfDir([dx, 0, dz]));
      if (dist > 0.004) {
        const n = Math.max(1, Math.ceil(dist / SL)), dur = (n + 0.5) * Ts, a = Math.min(0.45, 0.7 / n);
        P.sync(); P.T.x.to(goal.x, dur, u => cruise(u, a)); P.T.z.to(goal.z, dur, u => cruise(u, a));
        if (opt.blendYaw || (!turnFirst && !opt.keepFacing)) { const cur = P.T.yaw.v; P.T.yaw.to(cur + wrap(face - cur), dur, u => cruise(u, 0.4)); }
        P.sync();
      }
      if (!opt.keepFacing) yawTo(face);
      const t1 = P.sync();
      if (L) { const end = planSteps(P, t0, t1, Ts); P.sync(end); P.gaitT = [t0, end]; P.Ts = Ts; }
      P.whole = true; P.headW = 0.35; P.legs = true;
      if (!P.walkArms) { P.walkArms = true; P.extra(0, Infinity, tt => walkArms(P, tt), {att: 0.2, rel: 0.3}); }
      return P.te();
    }
    function walkArms(P, tt) {   // arms swing opposite the feet; the pelvis bobs at each landing (root y, in moveFrame)
      let sw = 0; for (const st of P.steps || []) if (tt >= st.t0 && tt < st.t1) sw += (st.s === 'l' ? 1 : -1) * Math.sin(Math.PI * (tt - st.t0) / (st.t1 - st.t0));
      if (!sw) return null;
      return {add: {hand_l_z: -0.12 * sw, hand_r_z: 0.12 * sw, hand_l_x: 0.03 * Math.abs(sw), hand_r_x: 0.03 * Math.abs(sw), body_yaw: 0.06 * sw}};
    }
    function bobAt(P, tt) { let y = 0; for (const st of P.steps || []) { const q = (tt - st.t1) / 0.09; if (q > -3 && q < 3) y -= 0.014 * Math.exp(-q * q); } return y; }
    function stopLegWalk(P, tt) {   // finish the steps in the air, come to a stop where it is, both feet under it
      for (const k of ROOT) P.T[k].truncate(tt);
      const v = [P.T.x.vel(tt), P.T.z.vel(tt)], stop = {x: P.T.x.v + v[0] * 0.2, z: P.T.z.v + v[1] * 0.2, yaw: P.T.yaw.v};
      P.T.x.quint(stop.x, 0.4, v[0], 0); P.T.z.quint(stop.z, 0.4, v[1], 0); P.T.yaw.hold(0.4);
      P.steps = (P.steps || []).filter(s2 => s2.t0 <= tt);
      P.footCur = {l: footAtPlan(P, 'l', 1e9), r: footAtPlan(P, 'r', 1e9)};
      const end = planSteps(P, tt, P.sync(), P.Ts || 0.46);
      P.sync(end); P.endBase = Object.assign({}, E.base, {x: P.T.x.v, z: P.T.z.v, yaw: P.T.yaw.v}); P.dwell = undefined;
      if (P.drag) P.dragEnd = P.te();
      return P.te();
    }
    // ---- approach poses (world): where to stand to reach something, facing it
    function approachFor(p, W, side) {
      const r = W ? W.root : E.base, here = [r.x, 0, r.z];
      if (isFurn(p)) {
        const F = feat(p), h = F.handles.map(q => fW(p, q)), mid = lerp3(h[0], h[1], 0.5), n = norm2(sub3(flat(mid), flat(p.pos)));
        if (!E.body.legs || h.length < 2) { const st = add3(flat(mid), mul3(n, 0.24)); return {x: st[0], z: st[2], yaw: yawOfDir(mul3(n, -1)), kind: 'handles'}; }
        // behind the handles, facing them: the nearest stance (distance, a little to either side, a slight turn) where each
        // paw reaches its own handle and the body has room (the piece itself excepted)
        const tn = [-n[2], 0, n[0]]; let best = null;
        for (const d of [0.24, 0.2, 0.28, 0.17, 0.32, 0.36]) for (const lat of [0, 0.04, -0.04, 0.08, -0.08]) for (const a of [0, 0.03, -0.03]) {
          const st = add3(add3(flat(mid), mul3(n, d)), mul3(tn, lat)), yaw = yawOfDir(mul3(n, -1)) + a, rr = {x: st[0], y: 0, z: st[2], yaw};
          const sides = toLocalR(rr, h[0])[0] < toLocalR(rr, h[1])[0] ? ['l', 'r'] : ['r', 'l'];
          const e = Math.max(bendSolve(rr, h[0], [sides[0]], false).e, bendSolve(rr, h[1], [sides[1]], false).e);
          const cost = Math.abs(d - 0.24) + Math.abs(lat) + Math.abs(a) + (e > 0.03 ? 10 + e : e) + (standClear(st[0], st[2], p) ? 0 : 5);
          if (!best || cost < best.cost) best = {x: st[0], z: st[2], yaw, kind: 'handles', cost, e};
          if (cost < 0.02) break;
        }
        if (best && best.cost < 5) return best;
        // no clear stance straight behind the handles (blocked, or a paw short): search the poses around the handle pair
        // (either side, nearer / further, turned) for one where both paws reach and the body has room
        const A = approachReach([{P: h[0], s: null}, {P: h[1], s: null}], add3(flat(mid), mul3(n, 0.3)), {self: p});
        if (A) {
          const rr = {x: A.x, y: 0, z: A.z, yaw: A.yaw}, sides = toLocalR(rr, h[0])[0] < toLocalR(rr, h[1])[0] ? ['l', 'r'] : ['r', 'l'];
          const e = Math.max(bendSolve(rr, h[0], [sides[0]], false).e, bendSolve(rr, h[1], [sides[1]], false).e);
          if (e <= 0.03) return {x: A.x, z: A.z, yaw: A.yaw, kind: 'handles', cost: A.cost, e};
        }
        return best && best.e <= 0.03 ? best : null;
      }
      const two = p.def.grip === 'two_hand' && p.def.handles && p.def.handles.length >= 2;
      if (two) return approachReach(p.def.handles.slice(0, 2).map(h => ({P: fW(p, h), s: null})), here) || null;
      let best = null;
      for (const g of gripCands(p)) { const A = approachReach([{P: fW(p, g), s: side || null}], here); if (A && (!best || A.cost < best.cost)) best = A; }
      return best;
    }
    // where to stand to reach world points (each {P, s: paw or null}): a search over poses around the target (side of
    // approach, distance, how far off its nose the target is) scored by the reach map's inverse error, the walk, the
    // turn, and room to stand (clear of tables and furniture). null = nowhere it can stand reaches it.
    function standClear(x, z, self, things) {
      const r = 0.21;
      if (things) for (const q of E.P) { if (q === self || isFurn(q) || q.held || q.worn || q.bal || q.flying || q.anc) continue; const c = P(q); if (c[1] - q.hH < 0.06 && Math.hypot(x - c[0], z - c[2]) < footR(q) + 0.13) return false; }   // never stand on a thing on the floor
      for (const pl of E.places) { if (pl.floor || pl.prop) continue; const c = pl.center, l = pl.yaw0 ? rotYv([x - c[0], 0, z - c[2]], -pl.yaw0) : [x - c[0], 0, z - c[2]];   // a turned place (the board) by its own yaw
        const dx = Math.max(0, Math.abs(l[0]) - pl.size[0] / 2), dz = Math.max(0, Math.abs(l[2]) - pl.size[1] / 2); if (Math.hypot(dx, dz) < r) return false; }
      for (const f of E.P) {
        if (!isFurn(f) || f === self) continue;
        const F = feat(f), w = F.chair ? F.chair.w : F.desk ? F.desk.w : 0.4, dp = F.chair ? F.chair.dep : F.desk ? F.desk.dep : 0.4, l = rotYv([x - f.pos[0], 0, z - f.pos[2]], -ryOf(f));
        if (Math.hypot(Math.max(0, Math.abs(l[0]) - w / 2), Math.max(0, Math.abs(l[2]) - dp / 2)) < r) return false;
      }
      return true;
    }
    function approachReach(reqs, here, opt = {}) {
      const tgt = mul3(reqs.reduce((a, q) => add3(a, q.P), [0, 0, 0]), 1 / reqs.length);
      const hv = sub3(here, flat(tgt)), phi0 = Math.hypot(hv[0], hv[2]) > 1e-3 ? Math.atan2(hv[0], hv[2]) / TAU : 0;
      const cands = [];
      for (const dphi of [0, 0.04, -0.04, 0.08, -0.08, 0.13, -0.13, 0.19, -0.19, 0.25, -0.25, 0.35, -0.35, 0.5])
        for (const d of [0.22, 0.27, 0.32, 0.37, 0.42, 0.47])
          for (const a of [0, 0.05, -0.05, 0.1, -0.1, 0.17, -0.17, 0.24, -0.24]) {
            const phi = (phi0 + dphi) * TAU, v = [Math.sin(phi), 0, Math.cos(phi)], x = tgt[0] + v[0] * d, z = tgt[2] + v[2] * d;
            // a stance that keeps its face toward the camera (+z) is preferred where the geometry allows — the hands
            // work three-quarters to him rather than with its back to him
            const yw = yawOfDir(mul3(v, -1)) - a, away = (1 - Math.cos(TAU * yw)) / 2;
            cands.push({x, z, v, a, cheap: 0.6 * Math.hypot(x - here[0], z - here[2]) + 0.25 * Math.abs(dphi) + 0.6 * Math.abs(a) + 0.05 * d + (opt.camW ?? 0.35) * away});
          }
      cands.sort((p, q) => p.cheap - q.cheap);
      let best = null, n = 0;
      for (const c of cands) {
        if (best && c.cheap >= best.cost) break;
        if (!standClear(c.x, c.z, opt.self, opt.things)) continue;
        const r = {x: c.x, y: 0, z: c.z, yaw: yawOfDir(mul3(c.v, -1)) - c.a}; let e = 0;
        if (++n > 400) break;
        const tol = opt.tol ?? 0.03;   // (a touch asks for a tighter reach)
        for (const q of reqs) { e = Math.max(e, bendSolve(r, q.Pf ? q.Pf(r) : q.P, q.s ? [q.s] : ['l', 'r'], false).e); if (e > tol) break; }
        if (e > tol) continue;
        const cost = c.cheap + 2 * e;
        if (!best || cost < best.cost) best = {x: c.x, z: c.z, yaw: r.yaw, cost, err: e};
      }
      return best;
    }
    function approachPoint(P0, placeId, here, p, yawW) {
      const one = p && p.held !== 'both' && !(p.def.grip === 'two_hand' && !p.held);
      if (yawW !== undefined && one) {   // an exact rest pose: the handle in the paw is at a fixed world point
        const PW = add3(P0, rotYv(gripLocal(p), yawW)), A0 = approachReach([{P: PW, s: p.held || null}], here);
        if (A0) return A0;
        const pl0 = E.places.find(x => x.id === placeId); if (!(pl0 && pl0.floor)) return null;   // a surface: no stance reaches that pose
      }
      const offL = p && p.held && p.held !== 'both' ? rotYv(gripLocal(p), p.homeRot[1] + p.hrot[1]) : null;   // the handle's offset, body frame
      const A = approachReach([offL ? {P: add3(P0, [0, offL[1], 0]), Pf: r => add3(P0, rotYv(offL, r.yaw)), s: p.held} : {P: add3(P0, [0, p ? gripLocal(p)[1] : 0.03, 0]), s: null}], here);
      if (A) return A;
      const pl = E.places.find(x => x.id === placeId), f = pl && byId(pl.prop || pl.on || ''), F = f && feat(f);
      if (E.body.legs && E.reach.l.ok && !(pl && pl.floor)) return null;   // a surface spot nowhere to stand reaches: the caller tries the next spot (the floor keeps its stand-off pose)
      let n;
      if (F && F.desk) { n = norm2(rotYv(F.desk.front, ryOf(f))); const edge = Math.abs((P0[0] - f.pos[0]) * n[0] + (P0[2] - f.pos[2]) * n[2]); const st = add3(flat(P0), mul3(n, Math.max(0.05, F.desk.dep / 2 - edge * 0) + 0.2)); void edge; return {x: st[0], z: st[2], yaw: yawOfDir(mul3(n, -1))}; }
      n = norm2(sub3(here, flat(P0))); if (!Number.isFinite(n[0]) || Math.hypot(here[0] - P0[0], here[2] - P0[2]) < 1e-3) n = [0, 0, 1];
      const st = add3(flat(P0), mul3(n, 0.27)); return {x: st[0], z: st[2], yaw: yawOfDir(mul3(n, -1))};
    }
    // what "put it back" did, in words: back on its stand / in its spot on the side table / where it came from (its
    // spot on the floor by the desk) / back where it was, on the desk
    function backPhrase(p, placeId, how) {
      const pl = E.places.find(x => x.id === placeId), nm = pl ? pl.name : 'its place';
      if (how === 'last') return pl && pl.floor ? 'back down where it was, on the floor' : `back where it was, on ${nm}`;
      if (pl && pl.stand_top_y !== undefined && feat(p).wearable) return `back on the stand on ${nm}`;
      if (pl && pl.floor) {
        const f = E.P.filter(q => isFurn(q)).map(q => ({q, d: d2(q.pos, p.home) - (feat(q).desk ? 0.25 : 0)})).sort((a, b) => a.d - b.d)[0];
        return `back where it came from — its home spot on the floor${f && f.d < 0.9 ? ' by the ' + f.q.name : ''}`;
      }
      return `back in its spot on ${nm}`;
    }
    function chairApproach(c) {
      const F = feat(c).chair, front = norm2(rotYv(F.front, ryOf(c))), seat = fW(c, F.seat);
      const st = add3(flat(seat), mul3(front, F.dep * 0.5 + 0.09));
      return {x: st[0], z: st[2], yaw: yawOfDir(front), seat, front};
    }
    // ---- sitting (on a chair, or on the floor) and standing up
    function sitPlan(P, c, sp) {
      const L = E.body.legs, t0 = P.sync(), dur = 1.05 / sp;
      if (c) {
        const A = chairApproach(c), ry = A.seat[1] + L.belowHip - L.hipY, back = add3(flat(A.seat), mul3(A.front, 0.03)), D = 1.25 / sp;
        P.T.yaw.to(P.T.yaw.v + wrap(A.yaw - P.T.yaw.v), D * 0.45, u => cruise(u, 0.4));   // turn its back to the seat
        P.sync();
        const t1 = P.te(), up = Math.max(0.05, ry + 0.06);
        P.T.x.to(back[0], D, u => mj(clamp((u - 0.05) / 0.8, 0, 1))); P.T.z.to(back[2], D, u => mj(clamp((u - 0.05) / 0.8, 0, 1)));
        P.T.y.to(up, D * 0.5, u => mj(u)).to(ry, D * 0.5, u => mj(u));                  // up onto the seat, then settle
        P.sync();
        P.seatRide = [t1 + 0.1 * D, t1 + 0.55 * D];                                         // the feet leave the floor
        P.sitOn = c.id;
      } else {   // the floor: hips down, legs out in front (the feet slide forward as it lowers)
        const ry = -(L.hipY - L.belowHip) + 0.01; P.T.y.to(ry, dur, u => mj(clamp((u - 0.1) / 0.9, 0, 1)));
        P.steps = P.steps || []; const yaw = P.T.yaw.v, fwd = [Math.sin(yaw * TAU), 0, Math.cos(yaw * TAU)];
        for (const s of ['l', 'r']) { const from = (P.footCur || E.feet)[s]; P.steps.push({s, t0: t0 + 0.2 * dur, t1: t0 + dur, from, to: add3(from, mul3(fwd, L.thigh * 0.9)), h: 0.01}); }
        P.sitOn = 'floor';
      }
      P.sync();
      P.extra(t0, t0 + dur + 0.3, tt => { const u = clamp((tt - t0) / dur, 0, 1), k = Math.sin(Math.PI * u); return {add: {body_pitch: 0.4 * k, head_pitch: -0.1 * k}, set: HPs(HPs({}, 'l', 0.25, -0.15 * k, -0.15 * k, 0.8, -0.3), 'r', 0.25, -0.15 * k, -0.15 * k, 0.8, -0.3), w: k}; }, {att: 0.1, rel: 0.3});
      P.endBase = Object.assign({}, E.base, {x: P.T.x.v, y: P.T.y.v, z: P.T.z.v, yaw: P.T.yaw.v}); P.endPosture = 'seated'; P.whole = true; P.headW = 0.3;
    }
    function standPlan(P, sp) {
      const t0 = P.sync(), dur = 1.0 / sp, c = byId(E.sitOn);
      let goal = {x: P.T.x.v, z: P.T.z.v};
      if (c && c.F && c.F.chair) { const A = chairApproach(c); goal = {x: A.x, z: A.z}; }
      P.T.x.to(goal.x, dur, u => mj(clamp((u - 0.15) / 0.85, 0, 1))); P.T.z.to(goal.z, dur, u => mj(clamp((u - 0.15) / 0.85, 0, 1)));
      P.T.y.to(0, dur, u => mj(clamp((u - 0.2) / 0.8, 0, 1)));
      if (E.sitOn === 'floor') { P.steps = P.steps || []; const yaw = P.T.yaw.v; for (const s of ['l', 'r']) { const from = E.feet[s]; P.steps.push({s, t0: t0, t1: t0 + 0.6 * dur, from, to: idealFoot(s, {x: goal.x, z: goal.z, yaw}), h: 0.03}); } }
      else if (c && E.body.legs) { P.steps = P.steps || []; const yaw = P.T.yaw.v; P.seatOff = [t0, t0 + 0.55 * dur]; for (const s of ['l', 'r']) { const from = E.feet[s]; P.steps.push({s, t0: t0 + 0.05 * dur, t1: t0 + 0.6 * dur, from, to: idealFoot(s, {x: goal.x, z: goal.z, yaw}), h: 0.0}); } }
      P.sync();
      P.extra(t0, t0 + dur + 0.2, tt => { const k = Math.sin(Math.PI * clamp((tt - t0) / dur, 0, 1)); return {add: {body_pitch: 0.45 * k}, w: 1}; }, {att: 0.1, rel: 0.3});
      P.endBase = Object.assign({}, E.base, {x: goal.x, y: 0, z: goal.z, yaw: P.T.yaw.v}); P.endPosture = 'standing'; P.unsit = true; P.whole = true; P.headW = 0.3;
    }
    // ---- moving furniture: hands on its handles, walk; it moves in lockstep with the body (and what is on it)
    // footprints: a piece of furniture at (pos, ry) must not land on a prop on the floor or on other furniture
    function footOf(f, pos, ry) { const F = feat(f), w = F.chair ? F.chair.w : F.desk ? F.desk.w : 0.4, dp = F.chair ? F.chair.dep : F.desk ? F.desk.dep : 0.4; return {c: pos, hw: w / 2, hd: dp / 2, ry}; }
    function inFoot(R, pt, m = 0) { const l = rotYv([pt[0] - R.c[0], 0, pt[2] - R.c[2]], -R.ry); return Math.abs(l[0]) <= R.hw + m && Math.abs(l[2]) <= R.hd + m; }
    function cornersOf(R) { return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => add3(R.c, rotYv([a * R.hw, 0, b * R.hd], R.ry))); }
    function footHits(f, pos, ry, ignore) {
      const R = footOf(f, pos, ry);
      for (const q of E.P) {
        if (q === f || q === ignore || q.held || q.worn) continue;
        if (isFurn(q)) { const Q = footOf(q, q.pos, ryOf(q)); if (cornersOf(R).some(c => inFoot(Q, c, 0.02)) || cornersOf(Q).some(c => inFoot(R, c, 0.02)) || inFoot(Q, R.c)) return q; continue; }
        const pl = E.places.find(x => x.id === q.place); if (pl && pl.prop) continue;   // riding on furniture
        if (q.pos[1] < 0.2 && inFoot(R, q.pos, q.size * 0.6 + 0.03)) return q;
      }
      for (const pl of E.places) { if (pl.floor || pl.prop) continue; const Q = {c: pl.center, hw: pl.size[0] / 2, hd: pl.size[1] / 2, ry: 0}; if (cornersOf(R).some(c => inFoot(Q, c, 0.02)) || cornersOf(Q).some(c => inFoot(R, c, 0.02))) return pl; }
      return null;
    }
    function clearGoal(f, g, ignore) {   // the nearest pose to g whose footprint is clear (slide it along / around)
      if (!footHits(f, g.pos, g.ry, ignore)) return g;
      let best = null;
      for (const r of [0.08, 0.16, 0.24, 0.32, 0.4, 0.5]) for (let k = 0; k < 12; k++) {
        const a = k / 12 * TAU, p2 = [g.pos[0] + r * Math.sin(a), g.pos[1], g.pos[2] + r * Math.cos(a)];
        if (footHits(f, p2, g.ry, ignore)) continue;
        const cost = r + 0.3 * Math.max(0, Math.hypot(p2[0], p2[2]) - Math.hypot(g.pos[0], g.pos[2]));
        if (!best || cost < best.cost) best = {cost, pos: p2};
      }
      return best ? Object.assign({}, g, {pos: best.pos}) : g;
    }
    function furnGoal(f, target, W) { const g = furnGoal0(f, target, W); return g.toObj ? g : clearGoal(f, g); }
    function furnGoal0(f, target, W) {
      const F = feat(f), tgt = target && typeof target === 'object' ? target : {to: target};
      const ref = tgt.object || tgt.place || tgt.id || (typeof tgt.to === 'string' && byId(tgt.to) ? tgt.to : null);
      const other = ref ? byId(typeof ref === 'object' ? ref.id || ref.name : ref) || resolveObject(ref).p : null;
      if (other && other !== f && feat(other).desk && F.chair) {   // the chair goes to the desk's chair spot, facing the desk
        const D = feat(other).desk, spot = fW(other, D.chairSpot), dfront = norm2(rotYv(D.front, ryOf(other)));
        const want = yawOfDir(mul3(dfront, -1)) - yawOfDir(F.chair.front);
        const ry = ryOf(f) + wrap(want - ryOf(f)), so = other.def.chairSpotIsSit ? rotYv([F.chair.seat[0], 0, F.chair.seat[2]], ry) : [0, 0, 0];
        return {pos: [spot[0] - so[0], f.pos[1], spot[2] - so[2]], ry, why: `up to ${other.name === 'desk' || /desk/.test(other.id) ? 'the desk' : 'the ' + other.name}`, toObj: true};
      }
      // toward another thing ("slide the desk over to the chair"): a desk brings its chair spot to the chair;
      // anything else stops beside it (clearGoal keeps it off everything else)
      if (other && other !== f) {
        if (F.desk && feat(other).chair) { const off = sub3(flat(fW(f, F.desk.chairSpot)), flat(f.pos)); return {pos: [other.pos[0] - off[0], f.pos[1], other.pos[2] - off[2]], ry: ryOf(f), why: `up to the ${other.name}`}; }
        const half = q => { const G = feat(q); return G.desk ? Math.max(G.desk.w, G.desk.dep) / 2 : G.chair ? Math.max(G.chair.w, G.chair.dep) / 2 : Math.max(0.08, q.size || 0.1); };
        const v0 = sub3(flat(f.pos), flat(other.pos)), n = Math.hypot(v0[0], v0[2]) > 1e-3 ? norm2(v0) : [1, 0, 0], gap = half(f) + half(other) + 0.06;
        return {pos: [other.pos[0] + n[0] * gap, f.pos[1], other.pos[2] + n[2] * gap], ry: ryOf(f), why: `over to the ${other.name}`};
      }
      const to = String(tgt.to || 'me').toLowerCase();
      const here = W ? [W.root.x, 0, W.root.z] : [E.base.x, 0, E.base.z];
      if (to === 'work' && F.desk) {   // turned in place just enough that its sitter is three-quarters to the camera
        const cur = ryOf(f) + yawOfDir(F.desk.front), sit = wrap(cur + 0.5), lim = 0.12;
        if (Math.abs(sit) <= lim + 0.04) return {pos: f.pos.slice(), ry: ryOf(f), why: 'as it is'};
        const want = (sit !== 0 ? Math.sign(sit) : (f.pos[0] >= 0 ? 1 : -1)) * lim;
        return {pos: f.pos.slice(), ry: ryOf(f) + wrap(want - sit), why: 'round to face the room'};
      }
      if (/back|away|home|further|farther/.test(to)) {
        const desk = E.P.find(q => q !== f && feat(q).desk && F.chair && d2(q.pos, f.pos) < 0.9);
        if (desk && !/home/.test(to)) { const n = norm2(sub3(flat(f.pos), flat(desk.pos))); const p2 = add3(f.pos, mul3(n, 0.45)); return {pos: p2, ry: ryOf(f), why: 'back from the desk'}; }
        return {pos: f.home.slice(), ry: f.homeRot[1], why: 'back to where it was'};
      }
      // "over" / "me" / "closer": toward the middle of the room, near where it stands
      const c = [0, 0, 0.0], v = sub3(flat(f.pos), c), dist = Math.hypot(v[0], v[2]), keep = Math.max(0.55, (F.desk ? F.desk.w * 0.6 : 0.3) + 0.3);
      // a pull is a visible one — at least ~0.25 toward the middle (it stops 0.45 from it at the nearest);
      // only a piece already that close is pulled toward him instead
      const goTo = Math.min(keep, Math.max(0.45, dist - 0.25));
      if (dist <= goTo + 0.05) { const p2 = add3(f.pos, [0, 0, 0.25]); return {pos: p2, ry: ryOf(f), why: 'closer'}; }
      const n = norm2(v); const p2 = add3(c, mul3(n, goTo)); void here;
      let ry = ryOf(f);
      if (F.desk) {   // a desk is set so that whoever works at it sits three-quarters to the camera (its face turned to him)
        const wantFront = p2[0] >= 0 ? -0.41 : 0.41, cur = ryOf(f) + yawOfDir(F.desk.front);   // the sitter ~0.09 turn off facing him
        ry = ryOf(f) + wrap(wantFront - cur);
      }
      return {pos: [p2[0], f.pos[1], p2[2]], ry, why: 'over'};
    }
    function dragPlan(P, f, goalF, sp) {   // the body walks so that the furniture ends at goalF (rigid hold)
      const r = {x: P.T.x.v, z: P.T.z.v, yaw: P.T.yaw.v};
      const rel = toLocalR(r, f.pos), relYaw = ryOf(f) - r.yaw;
      const myYaw = goalF.ry - relYaw, R = rotm(myYaw, 0, 0), C = E.body.C, q = mv(R, sub3(rel, C));
      const goal = {x: goalF.pos[0] - C[0] - q[0], z: goalF.pos[2] - C[2] - q[2], yaw: myYaw};
      const riders = [];
      const pl = placeOfFurn(f); if (pl) for (const q2 of onPlace(pl)) riders.push({p: q2, rel: rotYv(sub3(q2.pos, f.pos), -ryOf(f)), rr: q2.rot[1] - ryOf(f)});
      P.drag = {f, rel, relYaw, riders};
      legWalk(P, goal, sp * 0.8, {blendYaw: true, Ts: 0.5});
      P.endBase = Object.assign({}, E.base, {x: P.T.x.v, z: P.T.z.v, yaw: P.T.yaw.v});
      P.stopFn = (tt) => stopLegWalk(P, tt);
      P.zoom = 0.35;
    }
    function toLocalR(r, pw, b) { const R = rotm(r.yaw, 0, 0), C = E.body.C, d = sub3(pw, add3([r.x, (r.y || 0) + (b ? b.y : 0), r.z], C)); return unbendUp(add3(C, [R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2], R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2], R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]]), b ? b.p : 0); }
    function dragFrame(P, r) {   // lockstep: furniture pose = body pose ⊕ the grip
      const D = P.drag, f = D.f, pose = {x: r.x, y: 0, z: r.z, yaw: r.yaw};
      const pw = toWorldR(pose, D.rel); f.pos = [pw[0], f.pos[1], pw[2]]; f.rot[1] = r.yaw + D.relYaw - f.homeRot[1];
      for (const q of D.riders) { q.p.pos = add3(f.pos, rotYv(q.rel, ryOf(f))); q.p.rot[1] = ryOf(f) + q.rr; }
      f.where = len3(sub3(f.pos, f.home)) < 0.01 ? 'home' : 'moved';
      syncPlaces();
    }
    // ---- chains for the muse's object verbs (built from the generic paw steps)
    function gripSteps(f) {   // both paws onto its handles; they stay there (follow) until letGo
      const F = feat(f), out = [lookStep(fW(f, F.handles[0]), true)];
      const sides = F.handles.length >= 2 ? (toLocal(fW(f, F.handles[0]))[0] < toLocal(fW(f, F.handles[1]))[0] ? ['l', 'r'] : ['r', 'l']) : ['r'];
      out.push(par(sides.map((sd, i) => pawPath(sd, fW(f, F.handles[i]), 0.6, {open: 0.7, turn: -0.4}))));
      out.push(stepFn('grip', 0.2, (u) => { sides.forEach((sd, i) => { const pw = E.paw[sd]; pw.open = lerp(0.7, 0.15, mj(u)); pw.keep = true; pw.ctrlT = 1; pw.follow = () => fW(f, F.handles[i]); }); }));
      out.push(lookStep(null, false));
      return out;
    }
    function letGoSteps() {
      const out = [];
      out.push(stepFn('let go', 0.2, (u) => { for (const sd of ['l', 'r']) if (E.paw[sd].keep) { E.paw[sd].open = lerp(0.15, 0.6, mj(u)); } }));
      out.push(stepFn('release', 0.001, () => { for (const sd of ['l', 'r']) { const pw = E.paw[sd]; if (pw.keep) { pw.tgt = pw.follow ? pw.follow() : pw.tgt; pw.follow = null; pw.keep = false; } } }));
      out.push(par([retractStep('l'), retractStep('r')]));
      return out;
    }
    function lidEdge(p, sv) { const L = feat(p).laptop, th = L.open * sv; return fW(p, add3(L.hinge, [0, L.len * Math.sin(th), L.len * Math.cos(th)])); }
    function lidSteps(p, open) {
      const L = feat(p).laptop, s0 = p.s || 0, s1 = open ? 1 : 0;
      const sd = toLocal(p.pos)[0] < 0 ? 'l' : 'r', out = [lookStep(lidEdge(p, s0), true)];
      if (E.body.legs) out.push(bendStep(lidEdge(p, s0)));
      out.push(pawPath(sd, add3(lidEdge(p, s0), [0, 0.03, 0.02]), 0.55, {open: 0.5, turn: -0.3}));
      out.push(pawPath(sd, lidEdge(p, s0), 0.2, {arc: 0}));
      out.push({name: open ? 'open' : 'close', dur: 0.9, start() {}, update(tt) {
        const u = clamp(tt / 0.9, 0, 1), v = lerp(s0, s1, mj(u)); p.s = v; E.paw[sd].tgt = lidEdge(p, v); E.look.P = lidEdge(p, v); return u >= 1;
      }, end() { p.s = s1; emit({kind: 'object_event', name: open ? 'opened' : 'closed', phase: 'end', src: 'auto', args: {object: p.id}, did: `${open ? 'opened' : 'closed'} the ${p.name}`}); }});
      if (E.body.legs) out.push(bendStep(null));
      out.push(lookStep(null, false));
      out.push(retractStep(sd));
      void L; return out;
    }
    function slideSteps(p, dir) {   // both paws on its sides, slide it along the surface toward / away from the body
      const pl = E.places.find(x => x.id === p.place), out = [];
      const me = flat(toWorld(E.body.C)), v = norm2(sub3(me, flat(p.pos)));
      let dist = /further|farther|away|back/.test(dir) ? -0.12 : 0.12;
      if (/closer|me|toward/.test(dir) || !dir) { const gap = d2(me, p.pos); dist = clamp(gap - 0.3, 0.04, 0.2); }
      let to = add3(p.pos, mul3(v, dist));
      if (pl && !pl.floor) { const c = pl.center, sx = pl.size[0] / 2 - p.size * 0.6, sz = pl.size[1] / 2 - p.size * 0.6, lr = rotYv(sub3(to, c), -(pl.yaw || 0)); lr[0] = clamp(lr[0], -sx, sx); lr[2] = clamp(lr[2], -sz, sz); to = add3(c, rotYv(lr, pl.yaw || 0)); to[1] = p.pos[1]; }
      const side = (sd) => { const lx = rotYv([1, 0, 0], ryOf(p)); return add3(p.pos, mul3(lx, (sd === 'l' ? -1 : 1) * (feat(p).laptop ? feat(p).laptop.half : p.size) * (toLocal(add3(p.pos, lx))[0] > toLocal(p.pos)[0] ? 1 : -1))); };
      out.push(lookStep(p.pos, true));
      out.push(par(['l', 'r'].map(sd => pawPath(sd, side(sd), 0.55, {open: 0.6, turn: 0}))));
      const from = p.pos.slice(), offs = {l: sub3(side('l'), p.pos), r: sub3(side('r'), p.pos)};
      out.push({name: 'slide', dur: 0.8, start() {}, update(tt) { const u = clamp(tt / 0.8, 0, 1); p.pos = lerp3(from, to, mj(u)); for (const sd of ['l', 'r']) E.paw[sd].tgt = add3(p.pos, offs[sd]); E.look.P = p.pos; return u >= 1; },
        end() { emit({kind: 'object_event', name: 'slid', phase: 'end', src: 'auto', args: {object: p.id}, did: `slid the ${p.name} ${dist > 0 ? 'closer' : 'further away'}`}); }});
      out.push(lookStep(null, false));
      out.push(par([retractStep('l'), retractStep('r')]));
      return out;
    }
    function wearSteps(p, on, side) {
      const hp = headPoint(), crown = [hp[0], hp[1] + 0.2, hp[2]], out = [];
      const cupW = sd => toWorld(add3(hp, [(sd === 'l' ? -1 : 1) * 0.16, 0.02, 0]));
      if (on) {
        const s = p.held === 'both' ? 'l' : p.held, o = otherSide(s);
        out.push(pawPath(s, toWorld(crown), 0.7, {look: null, turn: 0.2}));
        if (!heldBy(o)) out.push(pawPath(o, cupW(o), 0.45, {open: 0.8}));
        out.push({name: 'put on', atomic: true, dur: 0.45, start() { this.a = {l: p.att.l, r: p.att.r}; }, update(tt) {
          const w = mj(clamp(tt / 0.45, 0, 1)); p.att.l = this.a.l * (1 - w); p.att.r = this.a.r * (1 - w); p.hh = w; E.paw[s].tgt = lerp3(toWorld(crown), cupW(s), w);
          if (!this.hr0) this.hr0 = p.hrot.slice(); p.hrot = this.hr0.map(v => v * (1 - w)); return tt >= 0.45;
        }, end() { p.att.l = 0; p.att.r = 0; p.hh = 1; p.held = null; p.worn = true; p.where = 'worn'; p.hoff = [0, 0, 0]; p.hrot = [0, 0, 0]; p.rot = [0, 0, 0]; p.grip = null;
          emit({kind: 'object_event', name: 'put_on', phase: 'end', src: 'auto', args: {object: p.id}, did: `put on the ${p.name}`}); }});
        out.push(par(['l', 'r'].map(sd => ({name: 'adjust', dur: 0.8, start() { E.paw[sd].ctrlT = 1; this.o0 = E.paw[sd].open; this.Q0 = fwdW(sd, E.paw[sd].hPrev); }, update(tt) { const k = Math.sin(Math.PI * clamp(tt / 0.8, 0, 1)), m = mj(clamp(tt / 0.3, 0, 1)); E.paw[sd].tgt = lerp3(this.Q0, add3(cupW(sd), [0, 0.012 * Math.sin(TAU * 2 * tt) * k, 0.01 * k]), m); E.paw[sd].open = lerp(this.o0, 0.6, m); return tt >= 0.8; }}))));
        out.push(par([retractStep('l'), retractStep('r')]));
        return out;
      }
      const s = side && !heldBy(side) ? side : !heldBy('r') ? 'r' : 'l', o = otherSide(s);
      out.push(par([pawPath(s, cupW(s), 0.55, {open: 0.7}), ...(heldBy(o) ? [] : [pawPath(o, cupW(o), 0.55, {open: 0.7})])]));
      out.push({name: 'take off', atomic: true, dur: 0.45, start() {}, update(tt) {
        const w = mj(clamp(tt / 0.45, 0, 1)); p.hh = 1 - w; p.att[s] = w; E.paw[s].tgt = lerp3(cupW(s), toWorld(crown), w); E.paw[s].open = lerp(0.7, 0.15, w); return tt >= 0.45;
      }, end() { p.hh = 0; p.att[s] = 1; p.held = s; p.worn = false; p.where = 'held'; p.hrot = [0, 0, 0]; p.grip = gaOf(p).slice(); p.hoff = [0, 0, 0];
        emit({kind: 'object_event', name: 'took_off', phase: 'end', src: 'auto', args: {object: p.id, side: sideWord(s)}, did: `took off the ${p.name} (holding it in its ${sideWord(s)} paw)`}); }});
      out.push(par([retractStep(s, true), retractStep(o)]));
      return out;
    }
    // ---- the typing activity (until told otherwise); "turn back" pauses it and turns the upper body to him
    function kbPoint(p, sd) { const L = feat(p).laptop, lx = rotYv([1, 0, 0], ryOf(p)), sgn = toLocal(add3(p.pos, lx))[0] > toLocal(p.pos)[0] ? 1 : -1; return add3(fW(p, L.kb), mul3(lx, (sd === 'l' ? -1 : 1) * sgn * L.half * 0.4)); }
    function screenPoint(p) { const L = feat(p).laptop; return L.screen ? fW(p, L.screen) : lerp3(fW(p, L.hinge), lidEdge(p, p.s || 0), 0.55); }
    function activityFrame(dt) {
      const A = E.activity; if (!A || A.kind !== 'typing') return;
      const p = A.p, now = t();
      if (A.paused) { for (const sd of ['l', 'r']) if (E.paw[sd].act) { E.paw[sd].act = false; E.paw[sd].keep = false; E.paw[sd].ctrlT = 0; } E.look.wT = 0; return; }
      for (const sd of ['l', 'r']) {
        const pw = E.paw[sd]; if (!pw.act) { pw.act = true; pw.keep = true; } pw.ctrlT = 1; pw.follow = null;
        const T = A.taps[sd]; if (now >= T.next) { T.t0 = now; T.next = now + (rnd() < 0.08 ? U(0.5, 1.3) : U(0.1, 0.28)); T.dx = U(-0.02, 0.02); }
        const u = (now - T.t0) / 0.1, dip = u >= 0 && u < 1 ? Math.sin(Math.PI * u) : 0;
        T.x = lerp(T.x || 0, T.dx || 0, 1 - Math.exp(-dt / 0.15));
        const k = kbPoint(p, sd), lx = rotYv([1, 0, 0], ryOf(p));
        pw.tgt = add3(add3(k, mul3(lx, T.x)), [0, 0.02 - 0.012 * dip, 0]); pw.open = 0.4 + 0.08 * dip; pw.turn = -0.5;
      }
      if (now >= A.glance) { A.glanceEnd = now + U(0.7, 1.3); A.glance = now + U(4, 8); }
      const glancing = now < (A.glanceEnd || 0);
      E.look.wT = glancing ? 0 : 1; E.look.P = screenPoint(p);
      A.ink = (A.ink || 0) + dt;
    }
    // ---- the planner: goals → primitive ops over a simulated world state (preconditions / effects)
    const Pw = (p, pos) => P(p, pos);
    function snapW() {
      const P = {};
      for (const p of E.P) P[p.id] = {pos: (p.fly && p.fly.settle ? toR(p, p.fly.settle.to) : p.pos).slice(), ry: p.rot[1], place: p.place, held: p.held, worn: !!p.worn, s: p.s || 0,
        on: p.sup ? p.sup.id : null, inside: !!(p.sup && p.sup.inside), bal: !!p.bal, anc: p.anc ? p.anc.name : null};   // (attachments)
      // (a rider on a thing that is held / worn sits, in the simulated world, at its place relative to that thing's
      // simulated pose — so moving the support there (wMove) brings it along exactly)
      const riderAt = (p, d = 0) => { const q = p.sup && byId(p.sup.id); if (!q || d > 8) return P[p.id].pos; if (q.sup) P[q.id].pos = riderAt(q, d + 1);
        return toR(p, add3(Pw(q, P[q.id].pos), rotYv(p.sup.rel, q.homeRot[1] + P[q.id].ry))); };
      for (const p of E.P) if (p.sup && !p.held) { let q = byId(p.sup.id), k = 0; while (q && q.sup && !q.held && k++ < 8) q = byId(q.sup.id); if (q && (q.held || q.worn || q.bal)) P[p.id].pos = riderAt(p); }
      return {root: {x: E.base.x, z: E.base.z, yaw: E.base.yaw}, sit: E.sitOn || null, held: {l: (heldBy('l') || {}).id || null, r: (heldBy('r') || {}).id || null},
        P, typing: !!(E.activity && !E.activity.paused), activity: !!E.activity, turned: !!E.turnBack};
    }
    // run a geometry helper against the simulated state (temporarily swap the live poses in)
    function withW(W, fn) {
      const save = E.P.map(p => [p, p.pos, p.rot.slice(), p.place, p.held, p.s]), b = E.base, sv = Object.assign({}, b);
      for (const p of E.P) { const q = W.P[p.id]; p.pos = q.pos; p.rot = [p.rot[0], q.ry, p.rot[2]]; p.place = q.place; p.s = q.s; }
      Object.assign(E.base, W.root); syncPlaces();
      try { return fn(); } finally { for (const [p, pos, rot, pl, h, s0] of save) { p.pos = pos; p.rot = rot; p.place = pl; p.held = h; p.s = s0; } Object.assign(E.base, sv); syncPlaces(); }
    }
    const freePaws = W => ['l', 'r'].filter(s => !W.held[s]);
    function reachFromW(W, Pw, sides) {   // can a paw (or the one named) reach world point Pw from the simulated root pose (seated or standing)?
      const r = Object.assign({}, W.root, {y: W.sit ? W.sitY || 0 : 0});
      return bendSolve(r, Pw, sides || ['l', 'r'], !!W.sit).e < 0.04;
    }
    function nearW(W, p, side) {
      if (side && !isFurn(p) && p.def.grip !== 'two_hand') return withW(W, () => gripCands(p).some(g => reachFromW(W, fW(p, g), [side])));
      if (isFurn(p)) { const A = withW(W, () => approachFor(p, W)); return !!A && Math.hypot(W.root.x - A.x, W.root.z - A.z) < 0.08 && Math.abs(wrap(W.root.yaw - A.yaw)) < 0.06; }
      return withW(W, () => (p.def.grip === 'two_hand' && p.def.handles && p.def.handles.length >= 2 ? [p.def.handles.slice(0, 2)] : gripCands(p).map(g => [g])).some(gs => gs.every(g => reachFromW(W, fW(p, g)))));
    }
    function wMove(w, id, pos) {   // a thing moved in the simulated world carries what rests on it
      const o = w.P[id], d = sub3(pos, o.pos); o.pos = pos.slice();
      for (const k in w.P) if (k !== id && w.P[k].on === id) wMove(w, k, add3(w.P[k].pos, d));
    }
    function planGoals(goals) {
      const W = snapW(), ops = [];
      const SP = goals.reduce((a, g) => Math.max(a, +(g && g.speed) || 0), 0) || undefined;   // teach mode: a job's walking pace (default 1)
      const fail = (why) => ({ok: false, reason: why});
      const push = (op, eff) => { ops.push(op); eff(W); };
      function standing() { if (W.activity) push({op: 'stop_typing', did: ['stopping typing', 'stopped typing']}, w => { w.typing = false; w.activity = false; }); if (W.sit) push({op: 'stand_up', did: ['standing up', 'stood up']}, w => { const c = byId(w.sit); if (c && feat(c).chair) { const A = withW(w, () => chairApproach(c)); w.root = {x: A.x, z: A.z, yaw: A.yaw}; } w.sit = null; w.turned = false; }); return true; }
      function walkTo(A, why) { if (Math.hypot(W.root.x - A.x, W.root.z - A.z) < (SP ? 0.015 : 0.06) && Math.abs(wrap(W.root.yaw - A.yaw)) < (SP ? 0.015 : 0.05)) return; /* a paced (teach) job stands where the plan says */ standing(); push({op: 'walk_to', pose: {x: A.x, z: A.z, yaw: A.yaw}, sp: SP, did: [`walking over to ${why}`, `walked over to ${why}`]}, w => { w.root = {x: A.x, z: A.z, yaw: A.yaw}; }); }
      function nearOrSit(p) {   // on a desk with its chair pulled up: if no standing spot reaches it, sit at the desk
        if (nearW(W, p)) return true;
        const pl = E.places.find(x => x.id === W.P[p.id].place), desk = pl && pl.prop ? byId(pl.prop) : null, chair = E.P.find(q => feat(q).chair);
        if (!W.sit) { const A = withW(W, () => approachFor(p, W)); if (A) { walkTo(A, 'the ' + p.name); return true; } }
        if (desk && feat(desk).desk && chair) { if (sitting(chair, desk) === false) return false; if (nearW(W, p)) return true; }
        return near(p);
      }
      function near(p, why, opt = {}) {
        if (opt.side && !isFurn(p)) {   // a paw was asked for: stand where THAT paw reaches it, if there is such a spot
          if (nearW(W, p, opt.side)) return true;
          const As = withW(W, () => approachFor(p, W, opt.side)); if (As) { walkTo(As, why || 'the ' + p.name); return true; }
          opt.sideLost = 'reach';
        }
        if (nearW(W, p)) return true; if (W.sit) { const W2 = Object.assign({}, W, {sit: null}); void W2; } let A = withW(W, () => approachFor(p, W));
        if (!A && !isFurn(p) && opt.pushChair) {   // a chair pulled up to the desk is in the way: stand, push it back, step in
          const ch = E.P.find(q => feat(q).chair && q !== p), dk = E.P.find(q => feat(q).desk);
          if (ch && dk && d2(W.P[ch.id].pos, W.P[dk.id].pos) < 0.9 && freePaws(W).length === 2) { const n0 = ops.length, r = moveFurn(ch, 'back'); if (r === true) A = withW(W, () => approachFor(p, W)); if (!A) ops.length = n0; }
        }
        if (!A) return fail(`the ${p.name} is out of my reach`); walkTo(A, why || 'the ' + p.name); return true; }
      function pawsFree(n, keep) {
        if (W.activity) push({op: 'stop_typing', did: ['stopping typing', 'stopped typing']}, w => { w.typing = false; w.activity = false; });
        for (let guard = 0; freePaws(W).length < n; guard++) {
          const s = ['l', 'r'].find(x => W.held[x] && W.held[x] !== keep); if (!s || guard > 3) return false;
          const q = byId(W.held[s]);
          let done = false;
          if (q.homePlace && q.where !== 'home' && d2(q.home, [W.root.x, 0, W.root.z]) < (q.size < 0.12 ? 2.6 : 1.2) && !takenBy(W, q, q.home)) {   // its own spot nearby: put it back there (a small thing goes back to its own spot anywhere in the room — the chalk planned 'on the chalk tray' landed on the floor)
            const n0 = ops.length, sv = JSON.stringify(W), r = on(q, q.homePlace, true, q.home.slice(), {rot: [0, 0, 0], how: 'home'});
            if (r === true && W.held[s] !== q.id) done = true; else { ops.length = n0; Object.assign(W, JSON.parse(sv)); delete ops.failed; }
          }
          if (!done) for (const plId of placesByNear(q)) {
            // (a live run: the lesson's chalk was dropped on the FLOOR to free a paw for the hat): before the
            // floor, a thing with its own spot in the room goes back there (a short walk), like the tray's chalk / eraser
            if (plId === 'floor' && q.homePlace && q.where !== 'home' && d2(q.home, [W.root.x, 0, W.root.z]) < 2.6 && !takenBy(W, q, q.home)) {
              const n1 = ops.length, sv = JSON.stringify(W), r1 = on(q, q.homePlace, true, q.home.slice(), {rot: [0, 0, 0], how: 'home'});
              if (r1 === true && W.held[s] !== q.id) { done = true; break; } ops.length = n1; Object.assign(W, JSON.parse(sv)); delete ops.failed;
            }
            const n0 = ops.length, r = on(q, plId, true); if (r === true && W.held[s] !== q.id) { done = true; break; } ops.length = n0; }
          if (!done) return false;
        }
        return true;
      }
      function placesByNear(q) {   // where to put something down to free a paw: surfaces near it first, the floor last
        const here = [W.root.x, 0, W.root.z];
        const c = withW(W, () => E.places.filter(pl => !pl.floor && pl.prop !== q.id && (!pl.sub || pl.id === q.homePlace) && !(byId(pl.prop) && feat(byId(pl.prop)).chair)).map(pl => ({id: pl.id, d: d2(pl.center, here)}))).sort((a, b) => a.d - b.d).filter(x => x.d < 1.2).map(x => x.id);
        return [...c, 'floor'];
      }
      function nearestPlace(q) {   // the nearest place (not the floor if a surface is close) to put something down
        const here = [W.root.x, 0, W.root.z];
        const cands = withW(W, () => E.places.filter(pl => !pl.floor && pl.id !== q.id && (pl.prop !== q.id) && (!pl.sub || pl.id === q.homePlace)).map(pl => ({pl, d: d2(pl.center, here)}))).sort((a, b) => a.d - b.d);
        return cands.length && cands[0].d < 0.9 ? cands[0].pl.id : 'floor';
      }
      function holding(p, want) {
        const two = p.def.grip === 'two_hand';
        if (two || want === 'both') want = null;   // a two-pawed thing takes both anyway; "both paws" on a small one = either
        if (W.held.l === p.id || W.held.r === p.id) {
          // held, but in the other paw than the one asked for: pass it over when that paw is free
          if (want && W.held[want] !== p.id && !W.held[want]) {
            const fr = otherSide(want);
            push({op: 'generic', cmd: {action: 'object', verb: 'hand_over', object: {id: p.id}}, did: [`moving the ${p.name} to its ${sideWord(want)} paw`, `moved the ${p.name} to its ${sideWord(want)} paw`]}, w => { w.held[want] = p.id; w.held[fr] = null; });
          }
          return true;
        }
        if (isFurn(p)) return fail(`the ${p.name} is too big to pick up — I can push or pull it`);
        if (W.P[p.id].anc && !W.P[p.id].worn) { if (!pawsFree(1)) return fail('my paws are full'); const s1 = want && !W.held[want] ? want : (!W.held.r ? 'r' : 'l'); push({op: 'pick_up', id: p.id, side: s1, did: [`taking the ${p.name} off ${anchorWord(W.P[p.id].anc)}`, `took the ${p.name} off ${anchorWord(W.P[p.id].anc)}`]}, w => { w.held[s1] = p.id; w.P[p.id].anc = null; w.P[p.id].place = null; }); return true; }
        if (W.P[p.id].worn || W.P[p.id].bal) { if (!pawsFree(1)) return false; const s0 = want && !W.held[want] ? want : (!W.held.r ? 'r' : 'l'); push({op: 'take_off', id: p.id, side: want && !W.held[want] ? want : undefined, did: [`taking ${W.P[p.id].bal ? 'down' : 'off'} the ${p.name}`, `took ${W.P[p.id].bal ? 'down' : 'off'} the ${p.name}`]}, w => { w.held[s0] = p.id; w.P[p.id].worn = false; w.P[p.id].bal = false; w.P[p.id].place = null; }); return true; }
        if (!pawsFree(two ? 2 : 1, p.id)) return fail('my paws are full');
        const busy = want && W.held[want] ? 'busy' : null, free1 = !two && freePaws(W).length === 1 ? freePaws(W)[0] : null;
        const nopt = {pushChair: true, side: want && !busy ? want : free1};   // one paw free: stand where that paw reaches
        const nr = near(p, null, nopt); if (nr && nr.ok === false) return nr;
        const pw = W.P[p.id].pos, sv0 = withW(W, () => ['r', 'l'].filter(x => !W.held[x]).sort((a, b) => Math.abs(toLocalR(W.root, pw)[0] - (a === 'l' ? -0.2 : 0.2)) - Math.abs(toLocalR(W.root, pw)[0] - (b === 'l' ? -0.2 : 0.2)))[0]);
        // the paw asked for, when it is free and reaches from where it stands; otherwise the nearer free paw, said truthfully
        const sv = want && !busy && !nopt.sideLost ? want : sv0, why = want && sv !== want ? (busy ? ` (its ${sideWord(want)} paw is holding something)` : ` (its ${sideWord(want)} paw can't reach it)`) : '';
        const from = W.P[p.id].place, paw = want && !two ? ` with its ${sideWord(sv)} paw` : '';
        push({op: 'pick_up', id: p.id, side: two ? 'both' : sv, want: want || undefined, did: [`picking up the ${p.name}${paw}${why}${from && from !== 'floor' ? '' : ''}`, `picked up the ${p.name}${paw}${why}`]}, w => { if (two) { w.held.l = p.id; w.held.r = p.id; } else w.held[sv] = p.id; w.P[p.id].place = null; w.P[p.id].on = null; w.P[p.id].inside = false; });
        return true;
      }
      function on(p, placeId, quiet, at, opt = {}) {
        const pl = E.places.find(x => x.id === placeId); if (!pl) return fail(`there is no ${placeId} here`);
        if (!opt.force && W.held.l !== p.id && W.held.r !== p.id && W.P[p.id].place === placeId && !W.P[p.id].worn && !W.P[p.id].on && !(at && opt.rot && d2(W.P[p.id].pos, at) > 0.01)) return true;
        { const h = holding(p); if (h === false || (h && h.ok === false) || ops.failed) return h || false; }
        // an exact rest pose (back home / where it was): the prop's world yaw is fixed, so the paw goes to the handle there
        const two = W.held.l === p.id && W.held.r === p.id, yawW = at && opt.rot ? p.homeRot[1] + opt.rot[1] : undefined;
        // (the paw goes to the HANDLE: a prop held off its centre — the headphones by the band — is set down by that handle)
        const heldOne = !two && (W.held.l === p.id || W.held.r === p.id) && p.held && p.held !== 'both';
        const pawPt = q => two ? q : yawW !== undefined ? add3(q, rotYv(gripLocal(p), yawW)) : heldOne ? add3(q, rotYv(gripLocal(p), p.homeRot[1] + p.hrot[1] + W.root.yaw)) : q;
        // the free spots (nearest first): the first one it can reach from the seat, or from somewhere it can stand
        const spots = at ? [at.slice()] : withW(W, () => { syncPlaces(); const pl2 = E.places.find(x => x.id === placeId); return pl2.floor ? [add3([W.root.x, 0, W.root.z], rotYv([0.12, p.restH, 0.3], W.root.yaw))] : freeSpotNear(pl2, p, W, true); });
        if (!spots.length) return fail(`there is no room on ${pl.name}`);
        let spot = null, A = null;
        for (const q of spots.slice(0, 6)) {
          if (W.sit && reachFromW(W, pawPt(q))) { spot = {pos: q}; break; }
          A = (opt.force ? withW2 : withW)(W, () => approachPoint(q, placeId, [W.root.x, 0, W.root.z], p, yawW)); if (A) { spot = {pos: q}; break; }
        }
        if (!spot && !opt.noSit && !W.sit && pl.prop) {   // a desk top no standing spot reaches (its chair pulled up): sit at it and set it down from the seat
          const dk = byId(pl.prop), ch = E.P.find(q => feat(q).chair);
          if (dk && feat(dk).desk && ch && withW(W, () => d2(fW(dk, feat(dk).desk.chairSpot), W.P[ch.id].pos)) < 0.12) {
            const n0 = ops.length, sv = JSON.stringify(W), rs = sitting(ch, dk);
            if (rs === true) { const r2 = on(p, placeId, quiet, at, Object.assign({}, opt, {noSit: true})); if (r2 === true) return true; }
            ops.length = n0; Object.assign(W, JSON.parse(sv));
          }
        }
        if (!spot) return fail(`I can't reach ${pl.name}`);
        if (A) walkTo(A, opt.why ? opt.why : at && opt.how ? (pl.floor ? `the ${p.name}'s spot` : pl.name) : pl.floor ? 'a clear spot' : pl.name);
        const back = at ? withW(W, () => backPhrase(p, placeId, opt.how || 'home')) : null;
        push({op: 'place', id: p.id, place: placeId, at: spot.pos, rot: yawW !== undefined ? opt.rot.slice() : undefined, did: opt.did ? opt.did.slice() : at ? [`putting the ${p.name} ${back}`, `put the ${p.name} ${back}`] : pl.floor ? [`putting the ${p.name} down on the floor`, `put the ${p.name} down on the floor`] : [`putting the ${p.name} on ${pl.name}`, `put the ${p.name} on ${pl.name}`], quiet}, w => { if (w.held.l === p.id) w.held.l = null; if (w.held.r === p.id) w.held.r = null; w.P[p.id].place = placeId; wMove(w, p.id, spot.pos);
          w.P[p.id].on = opt.sup && opt.sup.kind === 'prop' ? opt.sup.id : null; w.P[p.id].inside = !!(opt.sup && opt.sup.inside); w.P[p.id].anc = null; w.P[p.id].bal = false;
          if (yawW !== undefined) w.P[p.id].ry = opt.rot[1];
          else if (feat(p).laptop) w.P[p.id].ry = wrap(withW(w, () => laptopFaceYaw(spot.pos, placeId, [w.root.x, 0, w.root.z])) - p.homeRot[1]); });
        return true;
      }
      // "put it down" with no place named: the nearest surface it can really reach (from the seat, or a few steps away), the floor last
      function onAny(p) {
        const cands = placesByNear(p);
        for (const plId of cands) { const n0 = ops.length, sv = JSON.stringify(W), r = on(p, plId, false); if (r === true) return true; ops.length = n0; Object.assign(W, JSON.parse(sv)); delete ops.failed; }
        return fail(`there is nowhere in reach to put the ${p.name} down`);
      }
      // put back: its home pose (or where it last rested); its spot taken → beside it on the same place, said so
      function back(p, how, at, rot, placeId) {
        // taken = something else (not resting in its OWN home spot — the homes are laid out to sit side by side) on that spot
        const q = takenBy(W, p, at);
        if (!q) return on(p, placeId, false, at, {rot, how});
        const pl = E.places.find(x => x.id === placeId);
        const n0 = ops.length, r = on(p, placeId, false);
        if (r === true && ops.length > n0) { const op = ops[ops.length - 1]; if (op.op === 'place') op.did = op.did.map(d => d + ` (its own spot is taken by the ${q ? q.name : 'something else'})`); }
        return r === true ? true : fail(`its spot on ${pl ? pl.name : 'its place'} is taken${q ? ' by the ' + q.name : ''}, and there is no room beside it`);
      }
      function worn(p, want, side, legacy) {
        if (want && !legacy && AFF(p).wear && !twoPawWear(p)) return headG(p, 'wear');   // hat / glasses — one paw, afford.wear
        if (want ? W.P[p.id].worn : !(W.P[p.id].worn || W.P[p.id].bal)) return true;
        if (side === 'both') side = null;
        if (want) { if (!feat(p).wearable) return fail(`the ${p.name} isn't something to wear`); if (holding(p) === false || ops.failed) return false; push({op: 'put_on', id: p.id, did: [`putting on the ${p.name}`, `put on the ${p.name}`]}, w => { if (w.held.l === p.id) w.held.l = null; if (w.held.r === p.id) w.held.r = null; w.P[p.id].worn = true; }); return true; }
        if (!pawsFree(1)) return fail('my paws are full');
        { const sk = side && !W.held[side] ? side : null, paw = side ? ` with its ${sideWord(sk || (!W.held.r ? 'r' : 'l'))} paw` : '';
          push({op: 'take_off', id: p.id, side: sk || undefined, did: [`taking ${W.P[p.id].bal ? 'down' : 'off'} the ${p.name}${paw}`, `took ${W.P[p.id].bal ? 'down' : 'off'} the ${p.name}${paw}`]}, w => { const s = sk || (!w.held.r ? 'r' : 'l'); w.held[s] = p.id; w.P[p.id].worn = false; w.P[p.id].bal = false; w.P[p.id].place = null; }); }
        if (W.sit && W.sit !== 'floor') {   // at the desk: set them down on it (within reach of the seat), else keep them in the paw
          const d0 = E.P.find(q => feat(q).desk), dp = d0 && placeOfFurn(d0);
          if (dp) { const n0 = ops.length, sv = JSON.stringify(W), r = on(p, dp.id, false); if (r !== true || !W.sit) { ops.length = n0; Object.assign(W, JSON.parse(sv)); } }
        }
        return true;
      }
      function moveFurn(f, target) {
        const g = withW(W, () => furnGoal(f, target, W));
        if (d2(g.pos, W.P[f.id].pos) < 0.05 && Math.abs(wrap(g.ry - (f.homeRot[1] + W.P[f.id].ry))) < 0.03) return true;
        if (W.sit === f.id || W.sit) standing();
        if (!pawsFree(2)) return fail('my paws are full');
        { const nr = near(f); if (nr && nr.ok === false) return fail(`I can't get a grip on the ${f.name} from anywhere I can stand`); }
        push({op: 'move_furniture', id: f.id, goal: g, did: [`${target === 'away' || /back/.test(String(target)) ? 'pushing' : 'pulling'} the ${f.name} ${g.why}`, `${target === 'away' || /back/.test(String(target)) ? 'pushed' : 'pulled'} the ${f.name} ${g.why}`]}, w => {
          const rr = w.root, q = W.P[f.id], rel = withW(w, () => toLocalR(rr, q.pos)), relYaw = f.homeRot[1] + q.ry - rr.yaw, my = g.ry - relYaw, C = E.body.C, v = mv(rotm(my, 0, 0), sub3(rel, C));
          const pl = placeOfFurn(f), dp = sub3(g.pos, q.pos), dr = g.ry - (f.homeRot[1] + q.ry);
          if (pl) for (const id in w.P) if (w.P[id].place === pl.id) { const o = w.P[id]; o.pos = add3(g.pos, rotYv(sub3(o.pos, q.pos), dr)); o.ry += dr; }
          q.pos = g.pos.slice(); q.ry = g.ry - f.homeRot[1]; void dp;
          w.root = {x: g.pos[0] - C[0] - v[0], z: g.pos[2] - C[2] - v[2], yaw: my};
        });
        return true;
      }
      function sitting(c, deskFor) {
        if (deskFor) { const ok = atDesk(c, deskFor); if (ok === false) return false; }
        if (W.sit === (c ? c.id : 'floor')) return true;
        standing();
        if (c) { const A = withW(W, () => chairApproach(c)); walkTo(A, 'the ' + c.name); }
        push({op: 'sit', id: c ? c.id : null, did: c ? [`sitting down on the ${c.name}${deskFor ? ' at the ' + deskFor.name : ''}`, `sat down on the ${c.name}${deskFor ? ' at the ' + deskFor.name : ''}`] : ['sitting down on the floor', 'sat down on the floor']},
          w => { w.sit = c ? c.id : 'floor'; if (c) { const A = withW(w, () => chairApproach(c)); w.root = {x: A.seat[0] + A.front[0] * 0.03, z: A.seat[2] + A.front[2] * 0.03, yaw: A.yaw}; w.sitY = A.seat[1] + E.body.legs.belowHip - E.body.legs.hipY; } else w.sitY = -(E.body.legs.hipY - E.body.legs.belowHip); });
        return true;
      }
      function atDesk(c, desk) {
        const D = feat(desk).desk; if (!D) return true;
        // a desk that would seat it with its back to him is first turned to the working pose (three-quarters to
        // the camera, as the pulled-over layout sets it), so opening / typing / turning back read from the camera
        { const g = withW(W, () => furnGoal0(desk, 'work', W)); if (g && Math.abs(wrap(g.ry - (desk.homeRot[1] + W.P[desk.id].ry))) > 0.03) { const r0 = moveFurn(desk, 'work'); if (bad(r0)) return r0; } }
        const spot = withW(W, () => fW(desk, D.chairSpot)), cp = W.P[c.id].pos;
        if (d2(spot, cp) < 0.12) return true;
        return moveFurn(c, {object: desk.id});
      }
      const bad = r => r === false || (r && r.ok === false);   // a step that failed (false or a {ok:false} reason)
      function lid(p, open) {
        if ((W.P[p.id].s > 0.5) === open) return true;
        if (!feat(p).laptop) return hasArt(p) ? lidArt(p, open) : fail(`the ${p.name} doesn't open`);   // any hinged lid
        if (W.held.l === p.id || W.held.r === p.id) { const pl = nearestPlace(p), r0 = on(p, pl); if (bad(r0)) return r0 || false; }
        if (!pawsFree(1)) return fail('my paws are full');
        // one paw on the lid's front edge: from the seat if it reaches, else go where it does
        const edge = withW(W, () => lidEdge(byId(p.id), W.P[p.id].s));
        if (!(W.sit && reachFromW(W, edge))) { const nr = nearOrSit(p); if (bad(nr)) return nr || false; }
        push({op: open ? 'open' : 'close', id: p.id, did: [`${open ? 'opening' : 'closing'} the ${p.name}`, `${open ? 'opened' : 'closed'} the ${p.name}`]}, w => { w.P[p.id].s = open ? 1 : 0; });
        return true;
      }
      function slide(p, dir) {
        if (W.held.l === p.id || W.held.r === p.id) { const d0 = E.P.find(q => feat(q).desk), dp = d0 && placeOfFurn(d0); if (!on(p, dp ? dp.id : nearestPlace(p))) return false; }
        const pl = W.P[p.id].place; if (!pl || pl === 'floor') return fail(`the ${p.name} isn't on a table to slide`);
        if (!pawsFree(2)) return fail('my paws are full');
        { const nr = nearOrSit(p); if (nr === false || (nr && nr.ok === false)) return nr; }
        push({op: 'slide', id: p.id, dir, did: [`sliding the ${p.name} ${/further|away|back/.test(dir) ? 'further away' : 'closer'}`, `slid the ${p.name} ${/further|away|back/.test(dir) ? 'further away' : 'closer'}`]}, w => {
          const me = [w.root.x, 0, w.root.z], v = norm2(sub3(me, flat(w.P[p.id].pos))), dd = /further|away|back/.test(dir) ? -0.12 : clamp(d2(me, w.P[p.id].pos) - 0.3, 0.04, 0.2); w.P[p.id].pos = add3(w.P[p.id].pos, mul3(v, dd)); });
        return true;
      }
      function typing(p, wantSit) {
        if (W.typing) return true;
        if (W.activity && !W.typing) { push({op: 'resume', did: ['getting back to typing', 'got back to typing']}, w => { w.typing = true; w.turned = false; }); return true; }
        if (!feat(p).laptop) return fail(`the ${p.name} isn't something to type on`);
        // it works at a desk: the laptop on a desk, a chair pulled up to it, sitting
        let desk = null;
        const plId = W.P[p.id].place, pl = E.places.find(x => x.id === plId);
        if (pl && (pl.prop || pl.on)) desk = byId(pl.prop || pl.on);
        if (!desk || !feat(desk).desk) {
          const d0 = E.P.find(q => feat(q).desk), ch0 = E.P.find(q => feat(q).chair);
          if (d0) {
            const n0 = ops.length, sv = JSON.stringify(W); let r0 = on(p, placeOfFurn(d0).id);
            if (bad(r0) && ch0) {   // the chair is in the way: carry it to the seat, sit, set it down from there
              ops.length = n0; Object.assign(W, JSON.parse(sv));
              const rh = holding(p); if (bad(rh)) return rh || false;
              const rs = sitting(ch0, d0); if (bad(rs)) return rs || false;
              r0 = on(p, placeOfFurn(d0).id);
            }
            if (bad(r0)) return r0 || false; desk = d0;
          }
        }
        const chair = E.P.find(q => feat(q).chair);
        if (desk && chair && wantSit !== false) { const r1 = sitting(chair, desk); if (bad(r1)) return r1 || false; }
        if (!pawsFree(2)) return fail('my paws are full');
        // within reach first (slide it closer from the seat), then the lid up with a paw, then the keys
        // (both paws' key points comfortably inside reach — at the edge of reach the taps saturate and stop)
        const kbOk = withW(W, () => reachFromW(W, fW(byId(p.id), feat(p).laptop.kb)) && ['l', 'r'].every(sd => bendSolve(Object.assign({}, W.root, {y: W.sit ? W.sitY || 0 : 0}), kbPoint(byId(p.id), sd), [sd], !!W.sit).e < 0.012));
        if (!kbOk) { const r2 = slide(p, 'closer'); if (bad(r2)) return r2 || false; }
        { const r3 = lid(p, true); if (bad(r3)) return r3 || false; }
        if (!(W.P[p.id].s > 0.5)) return fail(`I couldn't get the ${p.name} open`);
        push({op: 'type', id: p.id, did: [`typing on the ${p.name}`, `started typing on the ${p.name} (still at it)`]}, w => { w.typing = true; w.activity = true; w.turned = false; });
        return true;
      }
      // ---- objects engine: relations, the head, body anchors, lids, throws, juggling, tools, putting it all back
      const heldW = id => W.held.l === id || W.held.r === id;
      const wv = fn => withW2(W, fn);
      const tName = T => T.prop ? 'the ' + T.prop.name : T.place ? T.place.name : T.me ? 'him' : T.anchor ? anchorWord(T.anchor) : 'there';
      // set p at a computed rest pose (physical) through on(): pick up, approach / walk, the 'place' op
      function onto(p, R, did, why) {
        const at = wv(() => toR(p, R.pos));
        return on(p, R.placeId || 'floor', false, at, {force: true, did, sup: R.sup, why});
      }
      function rel(p, T, verbWord) {
        if (isFurn(p)) {
          const tid = T.prop ? T.prop.id : T.place && T.place.prop ? T.place.prop : null;
          if (tid && /^(next_to|behind|in_front_of|left_of|right_of)$/.test(T.rel) && tid !== p.id) return moveFurn(p, {object: tid});
          return fail(`the ${p.name} is furniture — far too big to ${T.rel === 'in' ? 'fit in' : 'put'} ${T.rel === 'in' ? tName(T) : REL_WORD[T.rel] + ' ' + tName(T)}; I can push or pull it`);
        }
        if (T.anchor) return T.anchor === 'head' ? headG(p, AFF(p).wear ? 'wear' : 'balance') : anchorG(p, T.anchor);
        const q = T.prop;
        if (q === p) return fail(`I can't put the ${p.name} ${T.rel === 'in' ? 'in' : REL_WORD[T.rel] || 'on'} itself`);
        if (q && q !== p && heldW(q.id)) { const r0 = onAny(q); if (bad(r0)) return r0; }   // the target is in a paw: set it down first
        if (q && T.rel === 'in') {
          if (!isCont(q)) return fail(`the ${q.name} isn't something things go in`);
          const C = AFF(q).container;
          if (p.hx > C.inner[0] + 1e-3 || p.hz > C.inner[2] + 1e-3 || (isCont(p) && p.size >= q.size)) {
            // too big to go IN — it rests ON it instead (the opening's rim / the lid), said so; a closed-lid box is shut first
            if (isFurn(q) || p.size > 2.2 * q.size) return fail(`the ${p.name} is too big to fit in the ${q.name}`);
            T = Object.assign({}, T, {rel: 'on', note0: `too big to go in the ${q.name}, so it sits on top`});
          }
          if (T.rel === 'in' && C.needs_open && !(W.P[q.id].s > 0.5)) { const r1 = lidArt(q, true); if (bad(r1)) return r1; }
        }
        if (q && T.rel === 'on' && q !== p) {   // an open lid is no surface: shut it first
          const top = wv(() => stackTop(q, p));
          if (top !== p && hasArt(top) && isCont(top) && AFF(top).container.needs_open && W.P[top.id].s > 0.5) { const r1 = lidArt(top, false); if (bad(r1)) return r1; }
        }
        const R = wv(() => restPose(p, Object.assign({}, T, {near: [W.root.x, 0, W.root.z]})));
        if (R.err) return fail(R.err);
        if (T.note0) R.note = R.note ? T.note0 + '; ' + R.note : T.note0;
        const w = (T.me ? 'down in front of him' : `${REL_WORD[T.rel] || 'on'} ${tName(T)}`) + (R.onTop ? ` (on top of the ${R.onTop.name} already there)` : ''), vb = verbWord || (T.rel === 'in' ? 'putting' : 'putting');
        const past = {putting: 'put', stacking: 'stacked', setting: 'set', moving: 'moved', hiding: 'hid'}[vb] || 'put';
        const n0 = ops.length, r = onto(p, R, [`${vb} the ${p.name} ${w}${R.note ? ' (' + R.note + ')' : ''}`, `${past} the ${p.name} ${w}${R.note ? ' (' + R.note + ')' : ''}`], T.me ? 'a spot in front of him' : tName(T));
        void n0; return r;
      }
      // a hinged lid (afford.articulation): what rests on it is set aside first; one paw on its handle
      function lidArt(q, open) {
        if ((W.P[q.id].s > 0.5) === open) return true;
        if (!hasArt(q)) return fail(`the ${q.name} doesn't open`);
        if (heldW(q.id)) { const r0 = onAny(q); if (bad(r0)) return r0; }
        if (open) for (const r of E.P.filter(x => W.P[x.id].on === q.id && !W.P[x.id].inside && !heldW(x.id))) {
          const R = wv(() => restPose(r, {rel: 'next_to', prop: q, near: [W.root.x, 0, W.root.z]}));
          if (R.err) return fail(`the ${r.name} is on the ${q.name}'s lid and there is no room to set it aside`);
          const rr = onto(r, R, [`moving the ${r.name} off the ${q.name}'s lid first`, `moved the ${r.name} off the ${q.name}'s lid first`]); if (bad(rr)) return rr;
        }
        if (!pawsFree(1)) return fail('my paws are full');
        { const nr = near(q); if (bad(nr)) return nr; }
        push({op: 'art', id: q.id, open, did: [`${open ? 'opening' : 'closing'} the ${q.name}`, `${open ? 'opened' : 'closed'} the ${q.name}`]}, w => { w.P[q.id].s = open ? 1 : 0; });
        return true;
      }
      // what rests ON p is set down beside it first (before p is thrown / juggled: a stack would come apart in the air)
      function clearRiders(p, why) {
        for (const r of E.P.filter(x => x !== p && W.P[x.id].on === p.id && !W.P[x.id].inside && !heldW(x.id))) {
          const R = wv(() => restPose(r, {rel: 'next_to', prop: p, near: [W.root.x, 0, W.root.z]}));
          if (R.err) return fail(`the ${r.name} is on the ${p.name} and there is no room to set it aside`);
          const rr = onto(r, R, [`taking the ${r.name} off the ${p.name} first${why ? ' (' + why + ')' : ''}`, `took the ${r.name} off the ${p.name} first${why ? ' (' + why + ')' : ''}`]); if (bad(rr)) return rr;
        }
        return true;
      }
      function headG(p, mode) {
        const wear = mode === 'wear';
        if (isFurn(p)) return fail(`the ${p.name} is furniture — I can't ${wear ? 'wear' : 'lift'} it`);
        if (wear && !AFF(p).wear && !twoPawWear(p)) return fail(`the ${p.name} isn't something to wear${p.size <= 0.2 ? ' (I could balance it on my head instead)' : ''}`);
        if (wear && W.P[p.id].worn) return true;
        if (!wear && W.P[p.id].bal) return true;
        if (!wear && p.size > 0.2) return fail(`the ${p.name} is too big to balance on my head`);
        if (wear && twoPawWear(p)) return worn(p, true, null, true);
        if (wear) {
          const off = wearOff(p), clash = E.P.find(q => q !== p && W.P[q.id].worn && len3(sub3(off, wearOff(q))) < 0.45 * (p.size + q.size));
          if (clash) return fail(`I can't wear the ${p.name} with the ${clash.name} on — they'd sit in the same spot on my head (take the ${clash.name} off first?)`);
        }
        if (W.P[p.id].worn || W.P[p.id].bal) { const r0 = worn(p, false); if (bad(r0)) return r0; }
        const h = holding(p); if (bad(h) || ops.failed) return h || false;
        push({op: 'head', id: p.id, wear, did: wear ? [`putting on the ${p.name}`, `put on the ${p.name}`] : [`balancing the ${p.name} on its head`, `balanced the ${p.name} on its head`]},
          w => { if (w.held.l === p.id) w.held.l = null; if (w.held.r === p.id) w.held.r = null; w.P[p.id].worn = wear; w.P[p.id].bal = !wear; w.P[p.id].on = null; w.P[p.id].place = null; });
        return true;
      }
      function anchorG(p, name) {
        if (isFurn(p)) return fail(`the ${p.name} is furniture — far too big for that`);
        if (p.size > 0.2) return fail(`the ${p.name} is too big to put on ${anchorWord(name)}`);
        if (W.P[p.id].anc === name) return true;
        if (name === 'lap' && !W.sit) { const ch = E.P.find(q => feat(q).chair), r0 = sitting(ch || null); if (bad(r0)) return r0; }
        const h = holding(p); if (bad(h) || ops.failed) return h || false;
        push({op: 'anchor', id: p.id, name, did: [`putting the ${p.name} on ${anchorWord(name)}`, `put the ${p.name} on ${anchorWord(name)}`]},
          w => { if (w.held.l === p.id) w.held.l = null; if (w.held.r === p.id) w.held.r = null; w.P[p.id].anc = name; w.P[p.id].on = null; w.P[p.id].place = null; });
        return true;
      }
      function throwG(p, T) {
        if (isFurn(p)) return fail(`the ${p.name} is far too heavy to throw`);
        if (p.def.grip === 'two_hand') return fail(`the ${p.name} is too big to throw`);
        if (p.def.throwable === false && !AFF(p).throw) return fail(`the ${p.name} isn't something to throw`);
        if (T && T.anchor) return fail(`I can't throw the ${p.name} onto my own ${T.anchor === 'head' ? 'head' : 'body'}`);
        if (T && T.prop === p) return fail(`I can't throw the ${p.name} at itself`);
        const rl = T && T.prop ? (T.rel && T.rel !== 'on' && T.rel !== 'in' ? T.rel : isCont(T.prop) && (T.rel === 'in' || !AFF(T.prop).surface) ? 'in' : 'on') : T && T.rel || 'on';
        if (T && T.prop && rl === 'in') {
          if (!isCont(T.prop)) return fail(`the ${T.prop.name} isn't something things go in`);
          const C = AFF(T.prop).container; if (p.hx > C.inner[0] + 1e-3 || p.hz > C.inner[2] + 1e-3) return fail(`the ${p.name} is too big to fit in the ${T.prop.name}`);
          if (C.needs_open && !(W.P[T.prop.id].s > 0.5)) { const r1 = lidArt(T.prop, true); if (bad(r1)) return r1; }
        }
        const T2 = Object.assign({}, T || {me: true}, {rel: rl});
        { const r0 = clearRiders(p); if (bad(r0)) return r0; }
        const h = holding(p); if (bad(h) || ops.failed) return h || false;
        const aim = wv(() => aimPoint(p, T2)); if (aim.err) return fail(aim.err);
        const here = [W.root.x, 0, W.root.z], d = d2(aim.pos, here);
        if (d > 1.7 && !W.sit) { const v = norm2(sub3(here, flat(aim.pos))), st = add3(flat(aim.pos), mul3(v, 1.1)); walkTo({x: st[0], z: st[2], yaw: yawOfDir(mul3(v, -1))}, `within throwing range of ${aim.name}`); }
        else if (!W.sit) { const yawT = yawOfDir(sub3(flat(aim.pos), here)); if (Math.abs(wrap(yawT - W.root.yaw)) > 0.1) { standing(); push({op: 'walk_to', pose: {x: W.root.x, z: W.root.z, yaw: yawT}, did: [`turning toward ${aim.name}`, `turned toward ${aim.name}`]}, w => { w.root = Object.assign({}, w.root, {yaw: yawT}); }); } }
        push({op: 'throw', id: p.id, target: serT(T2), did: [`throwing the ${p.name} ${aim.word}`, `threw the ${p.name} ${aim.word}`]},
          w => { if (w.held.l === p.id) w.held.l = null; if (w.held.r === p.id) w.held.r = null; w.P[p.id].pos = wv(() => toR(p, aim.pos)); w.P[p.id].place = aim.placeId || 'floor'; w.P[p.id].on = aim.sup && aim.sup.kind === 'prop' ? aim.sup.id : null; w.P[p.id].inside = !!(aim.sup && aim.sup.inside); });
        return true;
      }
      function juggleG(a, b, n) {
        if (a === b) return fail('juggling needs two things — which other one?');
        for (const p of [a, b]) if (isFurn(p) || p.def.grip === 'two_hand' || (p.def.throwable === false && !AFF(p).throw)) return fail(`the ${p.name} isn't something to juggle`);
        for (const sd of ['l', 'r']) { const h = W.held[sd]; if (h && h !== a.id && h !== b.id) { const r0 = onAny(byId(h)); if (bad(r0)) return r0; } }   // other things in its paws are set down first
        for (const q of [a, b]) { const r0 = clearRiders(q); if (bad(r0)) return r0; }
        let r = holding(a, W.held.r === b.id ? 'l' : W.held.l === b.id ? 'r' : null); if (bad(r) || ops.failed) return r || false;
        const sa = W.held.l === a.id ? 'l' : 'r';
        r = holding(b, otherSide(sa)); if (bad(r) || ops.failed) return r || false;
        if (!(W.held.l && W.held.r && W.held.l !== W.held.r)) return fail('it needs one thing in each paw to juggle');
        if (!W.sit && Math.abs(wrap(W.root.yaw)) > 0.08) { standing(); push({op: 'walk_to', pose: {x: W.root.x, z: W.root.z, yaw: 0}, did: ['turning to face him', 'turned to face him']}, w => { w.root = Object.assign({}, w.root, {yaw: 0}); }); }
        push({op: 'juggle', ids: [a.id, b.id], n, did: [`juggling the ${a.name} and the ${b.name}`, `juggled the ${a.name} and the ${b.name}`]}, w => {});
        return true;
      }
      function useG(tool, tgt, motion) {
        if (!tool || !tgt) return fail('use what on what?');
        if (tool === tgt) return fail(`I can't use the ${tool.name} on itself`);
        if (isFurn(tool)) return fail(`the ${tool.name} is furniture — too big to use like that`);
        const U = AFF(tool).use || {}; motion = motion || U.motion || (AFF(tool).pour ? 'pour' : 'tap');
        if (motion === 'pour' && !AFF(tool).pour) return fail(`the ${tool.name} has nothing in it to pour`);
        if (heldW(tgt.id)) { const r0 = onAny(tgt); if (bad(r0)) return r0; }
        const had = heldW(tool.id), h = holding(tool); if (bad(h) || ops.failed) return h || false;
        const sd = W.held.l === tool.id ? 'l' : 'r', T0 = wv(() => add3(P(tgt), [0, topOff(tgt) + 0.12, 0]));
        if (!reachFromW(W, T0, [sd])) {
          if (W.sit) standing();
          const A = wv(() => approachReach([{P: T0, s: sd}], [W.root.x, 0, W.root.z]));
          if (!A) return fail(`I can't reach the ${tgt.name} with the ${tool.name}`);
          walkTo(A, 'the ' + tgt.name);
        }
        const Pd = AFF(tool).pour || {}, water = motion === 'pour' && /water/.test(Pd.what || '') && AFF(tgt).water_target;
        const word = motion === 'pour' ? (water ? [`watering the ${tgt.name} with the ${tool.name}`, `watered the ${tgt.name} with the ${tool.name}`] : [`pouring ${Pd.what || 'it'} from the ${tool.name} on the ${tgt.name}`, `poured ${Pd.what || 'it'} from the ${tool.name} on the ${tgt.name}`])
          : [`${{rub: 'rubbing', brush: 'brushing', tap: 'tapping', press: 'pressing'}[motion] || 'using'} the ${tool.name} on the ${tgt.name}`, `${{rub: 'rubbed', brush: 'brushed', tap: 'tapped', press: 'pressed'}[motion] || 'used'} the ${tool.name} on the ${tgt.name}`];
        push({op: 'use', id: tool.id, target: tgt.id, motion, did: word}, w => {});
        if (!had && tool.homePlace) { const n0 = ops.length, sv = JSON.stringify(W), r2 = putHome(tool); if (bad(r2)) { ops.length = n0; Object.assign(W, JSON.parse(sv)); delete ops.failed; } }   // the tool fetched for it goes back
        return true;
      }
      // put everything back: unstack what is on moved things top-first, home stacks bottom-first, worn things off, shut
      // what is open that is shut at home, furniture last
      const atHomeW = q => { const w = W.P[q.id]; return !heldW(q.id) && !w.worn && !w.bal && !w.anc && Math.hypot(w.pos[0] - q.home[0], w.pos[1] - q.home[1], w.pos[2] - q.home[2]) < 0.015 && (w.on || null) === (q.homeOn || null) && !w.inside; };
      function putHome(q) {
        const w = W.P[q.id];
        if (w.worn || w.bal) { const r0 = worn(q, false); if (bad(r0)) return r0; }
        if (q.homeOn) {
          const s = byId(q.homeOn);
          if (!atHomeW(s)) return fail(`the ${s.name} it belongs on is not back in its place`);
          if (hasArt(s) && W.P[s.id].s > 0.5 && !((+(s.def.articulation ?? 0) || 0) > 0.5)) { const r1 = lidArt(s, false); if (bad(r1)) return r1; }   // its lid shut first (it rests on it)
          return on(q, supPlace(s), false, q.home.slice(), {rot: [0, 0, 0], how: 'home', force: true, sup: {kind: 'prop', id: s.id}, did: [`putting the ${q.name} back on the ${s.name}`, `put the ${q.name} back on the ${s.name}`]});
        }
        const hit = wv(() => boxHit(q, P(q, q.home), [q]));   // its home spot physically taken by something out of place: that goes aside first
        if (hit && !atHomeW(hit)) {
          const fl = E.places.find(x => x.floor), Rr = wv(() => restPose(hit, {place: fl, near: W.P[hit.id].pos, anyFloor: true}));
          if (!Rr.err) { const r0 = onto(hit, Rr, [`moving the ${hit.name} out of the ${q.name}'s spot`, `moved the ${hit.name} out of the ${q.name}'s spot`]); if (bad(r0)) return r0; }
        }
        return on(q, q.homePlace || 'floor', false, q.home.slice(), {rot: [0, 0, 0], how: 'home', force: true});
      }
      function tidy() {
        if (W.activity) push({op: 'stop_typing', did: ['stopping typing', 'stopped typing']}, w => { w.typing = false; w.activity = false; });
        for (let guard = 0; guard < 30; guard++) {
          const A = E.P.filter(q => !isFurn(q) && !atHomeW(q)); if (!A.length) break;
          // (a rider whose home is ON q rides back with it: no blocker)
          const el = A.filter(q => !E.P.some(r => r !== q && W.P[r.id].on === q.id && !atHomeW(r) && r.homeOn !== q.id) && (!q.homeOn || atHomeW(byId(q.homeOn))));
          if (!el.length) return fail('it could not work out an order to put everything back');
          const here = [W.root.x, 0, W.root.z];
          el.sort((a, b) => (heldW(b.id) - heldW(a.id)) || ((W.P[b.id].worn || W.P[b.id].bal ? 1 : 0) - (W.P[a.id].worn || W.P[a.id].bal ? 1 : 0)) || (d2(W.P[a.id].pos, here) - d2(W.P[b.id].pos, here)));
          const q = el[0], w = W.P[q.id], cont = w.inside && w.on ? byId(w.on) : null;
          if (cont && AFF(cont).container.needs_open && !(W.P[cont.id].s > 0.5)) { const r1 = lidArt(cont, true); if (bad(r1)) return r1; }
          const r = putHome(q); if (bad(r)) return r;
          if (!atHomeW(q)) return fail(`it could not put the ${q.name} back`);
        }
        for (const q of E.P) {
          const home0 = (+(q.def.articulation ?? q.def.s ?? 0) || 0) > 0.5;
          if (hasArt(q) && (W.P[q.id].s > 0.5) !== home0) { const r = lidArt(q, home0); if (bad(r)) return r; }
          else if (feat(q).laptop && (W.P[q.id].s > 0.5) !== home0) { const r = lid(q, home0); if (bad(r)) return r; }
        }
        for (const f of E.P) if (isFurn(f) && (d2(W.P[f.id].pos, f.home) > 0.05 || Math.abs(wrap(W.P[f.id].ry)) > 0.03)) { const r = moveFurn(f, 'home'); if (bad(r)) return r; }
        return true;
      }
      // objects: push a loose thing so it tips / flies (paw), or kick it (foot): stand where that limb reaches
      // it, then the engine's own flight physics takes over (what rode on it is shed, it lands where it lands — said truthfully)
      function knock(p, foot) {
        if (isFurn(p)) return fail(`the ${p.name} is furniture — I can push or pull it`);
        if (W.P[p.id].worn || W.held.l === p.id || W.held.r === p.id) return fail(`I'd have to put the ${p.name} down first`);
        const q = W.P[p.id].pos, me = [W.root.x, 0, W.root.z];
        if (foot) {
          if (!E.body.legs) return fail('I have no feet to kick with');
          if (q[1] - p.restH > 0.05) return fail(`the ${p.name} is up on ${placeName(W.P[p.id].place || 'floor')}, too high to kick`);
          const d = norm2(sub3(flat(q), me)), back = 0.17 + footR(p), A = {x: q[0] - d[0] * back, z: q[2] - d[2] * back, yaw: yawOfDir(d)};
          walkTo(A, 'the ' + p.name);
        } else { const nr = near(p); if (nr && nr.ok === false) return nr; }
        push({op: 'knock', id: p.id, foot: !!foot, did: foot ? [`kicking the ${p.name}`, `kicked the ${p.name}`] : [`knocking the ${p.name} over`, `knocked the ${p.name} over`]}, w => { w.P[p.id].place = null; });
        return true;
      }
      // stand within reach of a thing (or place) and touch it with a paw in a style; go and stand by one
      function touchG(g) {
        const T = g.T, nm = targetName(T), here = [W.root.x, 0, W.root.z];
        if (T.prop && (heldW(T.prop) || W.P[T.prop].worn || W.P[T.prop].bal)) return fail(`the ${byId(T.prop).name} is ${W.P[T.prop].worn || W.P[T.prop].bal ? 'on my head' : 'in my paw'} — I can't ${g.style === 'knock' ? 'knock on' : g.style} it there`);
        const S = withW(W, () => touchSpot(T, here)); if (S.err) return fail(S.err);
        if (W.held.l && W.held.r) { if (!pawsFree(1)) return fail('my paws are full'); }
        const free = ['l', 'r'].filter(x => !W.held[x]), want = g.side && free.includes(g.side) ? [g.side] : free;
        const G0 = add3(S.C, mul3(S.n, TOUCH_DEEP));   // the lowest the grasp point goes (palm facing the surface)
        const okHere = !W.sit && withW(W, () => { const r = Object.assign({}, W.root, {y: 0}); return bendSolve(r, G0, want, false).e < 0.008; });
        if (!okHere) {
          const self = T.prop && !isFurn(byId(T.prop)) ? byId(T.prop) : undefined, A = withW(W, () => approachReach([{P: G0, s: want.length === 1 ? want[0] : null}], here, {things: true, self, tol: 0.008, camW: S.n[1] > 0.7 ? 0.9 : 1.2}));
          if (!A) return fail(`I can't reach ${nm} from anywhere I can stand`);
          walkTo(A, nm);
        }
        standing();
        const Sw = withW(W, () => touchSpot(T, [W.root.x, 0, W.root.z]));   // the spot as seen from where it will stand
        push({op: 'touch', T, S: Sw.err ? S : Sw, id: T.prop || undefined, style: g.style, count: g.count, side: g.side || null, did: [`${TOUCH_WORDS[g.style][0]} ${nm}`, `${TOUCH_WORDS[g.style][1]} ${nm}`]}, w => {});
        return true;
      }
      function goToG(g) {
        const T = g.T, nm = targetName(T), here = [W.root.x, 0, W.root.z];
        const n0 = ops.length;
        let A = null;
        if (T.home) A = {x: E.home0 ? E.home0.x : 0, z: E.home0 ? E.home0.z : 0, yaw: 0};
        else if (T.place) {
          const q = rawPlace(T.place), st = q && q.station;
          if (st && Array.isArray(st.pos)) A = {x: +st.pos[0], z: +st.pos[2], yaw: +st.yaw || 0};
          else if (q && q.on_place && rawPlace(q.on_place) && rawPlace(q.on_place).station) { const s2 = rawPlace(q.on_place).station; A = {x: +s2.pos[0], z: +s2.pos[2], yaw: +s2.yaw || 0}; }
          else if (q) { const c = v3(q.center), y = Number.isFinite(+q.top_y) ? +q.top_y : c && c[1]; if (!c) return fail(`I can't tell where ${nm} is`);   // a top: within reach of its middle
            A = withW(W, () => approachReach([{P: [c[0], y + 0.06, c[2]], s: null}], here, {things: true})); }
        } else {
          const p = byId(T.prop); if (heldW(p.id) || W.P[p.id].worn || W.P[p.id].bal) return fail(`the ${p.name} is ${heldW(p.id) ? 'in my paw' : 'on my head'} — it goes where I go`);
          A = withW(W, () => approachFor(p, W));
          if (!A) { const S = withW(W, () => touchSpot(T, here)); if (!S.err) A = withW(W, () => approachReach([{P: add3(S.C, mul3(S.n, 0.06)), s: null}], here, {things: true, self: p})); }
        }
        if (!A) return fail(`I can't find a spot to stand by ${nm}`);
        if (W.sit) standing();
        walkTo(A, nm);
        if (ops.length === n0) push({op: 'noop', did: [`already standing by ${nm}`, `was already standing by ${nm}`]}, w => {});
        return true;
      }
      for (const g of goals) {
        let r = true;
        const p = g.id ? byId(g.id) : null;
        switch (g.g) {
          case 'touch': r = touchG(g); break;
          case 'go_to': r = goToG(g); break;
          case 'present': {   // VISIBILITY: a thing shown / given / waved to him is brought over to him first
            // (live + eval: shown at the desk / tray / chair it was hidden behind furniture or the big headphones, and small
            // across the room); standing only, and only when it is away from its spot in front of him
            const hx = E.home0 ? E.home0.x : 0, hz = E.home0 ? E.home0.z : 0;
            if (!W.sit && Math.hypot(W.root.x - hx, W.root.z - hz) > (g.far ?? 0.45)) walkTo({x: hx, z: hz, yaw: 0}, 'him');
            break;
          }
          case 'bring':   // "bring it closer": on a table → slide it over; otherwise fetch it (decided from the state when it runs)
            if (W.P[p.id].place && W.P[p.id].place !== 'floor' && W.held.l !== p.id && W.held.r !== p.id && !W.P[p.id].worn) { r = slide(p, 'closer'); break; }
          // falls through
          case 'holding': {
            const W0 = {sit: W.sit};
            r = holding(p, g.side);
            // fetched from the seat at the desk (the last thing asked): back to the seat with it — a two-pawed
            // thing (the laptop) set down on the desk first — instead of standing there holding it
            if (r === true && (g.fetch || g.g === 'bring') && W0.sit && W0.sit !== 'floor' && !W.sit && g === goals[goals.length - 1]) {
              const ch = byId(W0.sit), dk = E.P.find(q => feat(q).desk), dp = dk && placeOfFurn(dk);
              if (ch && feat(ch).chair) {
                const n0 = ops.length, sv = JSON.stringify(W);
                let r2 = true;
                if (p.def.grip === 'two_hand' && dp) r2 = on(p, dp.id, false);
                if (r2 === true && !W.sit) r2 = sitting(ch, dk && d2(W.P[ch.id].pos, W.P[dk.id].pos) < 0.9 ? dk : null);
                if (bad(r2)) { ops.length = n0; Object.assign(W, JSON.parse(sv)); delete ops.failed; }
              }
            }
            break;
          }
          case 'worn': r = worn(p, true); break;
          case 'unworn': r = worn(p, false, g.side); break;
          case 'on': {
            const W0 = {sit: W.sit, activity: W.activity, typing: W.typing};
            r = g.any ? onAny(p) : g.back ? back(p, g.back, g.at, g.rot, g.place) : on(p, g.place, false, g.at);
            // it got up from work at the desk to put it away: it comes back and carries on (the last thing asked)
            if (r === true && W0.sit && W0.sit !== 'floor' && W0.activity && !W.sit && g === goals[goals.length - 1]) {
              const lap = E.P.find(q => feat(q).laptop && W.P[q.id].place && W.P[q.id].place !== 'floor'), ch = byId(W0.sit);
              if (lap && W0.typing) { const n0 = ops.length, sv = JSON.stringify(W), r2 = typing(lap); if (bad(r2)) { ops.length = n0; Object.assign(W, JSON.parse(sv)); } }
              else if (ch && feat(ch).chair) { const dk = E.P.find(q => feat(q).desk), n0 = ops.length, sv = JSON.stringify(W), r2 = sitting(ch, null); if (bad(r2)) { ops.length = n0; Object.assign(W, JSON.parse(sv)); } void dk; }
            } else if (r === true && W0.sit && W0.sit !== 'floor' && !W.sit && g === goals[goals.length - 1]) {
              // got up from the desk to put something ON that desk: back into the seat (as a seated fetch does)
              const ch = byId(W0.sit), dk = E.P.find(q => feat(q).desk), dp = dk && placeOfFurn(dk);
              if (ch && feat(ch).chair && dp && g.place === dp.id) { const n0 = ops.length, sv = JSON.stringify(W), r2 = sitting(ch, null); if (bad(r2)) { ops.length = n0; Object.assign(W, JSON.parse(sv)); } }
            }
            break;
          }
          case 'furn': r = moveFurn(p, g.target); break;
          case 'sit': r = sitting(p, g.desk ? byId(g.desk) : null); break;
          case 'stand': standing(); break;
          case 'lid': r = lid(p, g.open); break;
          case 'slide': r = slide(p, g.dir); break;
          case 'type': r = typing(p, g.sit); break;
          case 'near': near(p); break;
          case 'knock': r = knock(p, g.foot); break;   // objects: knock over / tip over / kick / flick a loose thing
          case 'face': push({op: 'turn_back', did: ['turning to face him', 'turned to face him']}, w => { w.root = Object.assign({}, w.root, {yaw: 0}); }); break;
          case 'turn_back': push({op: 'turn_back', did: ['turning back to him (keeping its place)', 'turned back to him']}, w => { w.turned = true; if (w.activity) w.typing = false; }); break;
          case 'stop_activity': if (W.activity) push({op: 'stop_typing', did: ['stopping typing', 'stopped typing']}, w => { w.typing = false; w.activity = false; }); break;
          case 'generic': { const vb = String(g.cmd.verb).replace(/_/g, ' '), nm = byId(g.cmd.object.id).name;   // a real progressive / past (the action memory read "then show the eraser")
            const GD = {show: ['holding the N up to show him', 'held the N up to show him'], hold_up: ['holding the N up to show him', 'held the N up to show him'], raise: ['raising the N as high as its short arms reach (about chin height — not over its head)', 'held the N up as high as its short arms reach (not over its head)'], offer: ['holding the N out to him', 'held the N out to him'], inspect: ['looking closely at the N', 'looked closely at the N'], hug: ['hugging the N', 'hugged the N'], shake: ['shaking the N', 'shook the N'], wave: ['waving the N', 'waved the N']}[String(g.cmd.verb)];
            push({op: 'generic', cmd: g.cmd, did: GD ? GD.map(x => x.replace('N', nm)) : [vb + ' the ' + nm, vb + ' the ' + nm]}, w => {}); break; }
          // teach mode: stand at an exact pose (stations, steps along the board, a half-turn to him) and a chain of steps
          // built from the live state when it runs (writing, erasing, presenting) — app/teach.js
          case 'stand_at': { const A = g.pose, tol = g.tol || 0.012; if (Math.hypot(W.root.x - A.x, W.root.z - A.z) < tol && Math.abs(wrap(W.root.yaw - A.yaw)) < (g.ytol || 0.01)) break; standing();
            push({op: 'walk_to', pose: {x: A.x, z: A.z, yaw: A.yaw}, sp: g.speed, Ts: g.Ts, keepFacing: g.keepFacing, blendYaw: g.blendYaw, turnStep: g.turnStep, did: g.did || [`walking over to ${g.why || 'there'}`, `walked over to ${g.why || 'there'}`], quiet: g.quiet}, w => { w.root = {x: A.x, z: A.z, yaw: A.yaw}; }); break; }
          case 'steps': if (g.ran) break; push({op: 'steps', g, build: g.build, id: g.id, did: g.did || ['', '']}, w => { if (g.eff) g.eff(w); }); break;   // (a re-plan never repeats a chain that ran)
          case 'rel': r = rel(p, deT(g.T), g.vb); break;   // objects engine
          case 'head': r = headG(p, g.mode); break;
          case 'throw': r = throwG(p, deT(g.T)); break;
          case 'juggle': r = juggleG(byId(g.ids[0]), byId(g.ids[1]), g.n); break;
          case 'use': r = useG(byId(g.tool), byId(g.target), g.motion); break;
          case 'tidy': r = tidy(); break;
          case 'home1': r = putHome(p); break;
          case 'move': { if ((W.sit || W.activity) && !SEATED_OK.test(g.name)) standing(); const dv = movePreview(g.name, g.args); push({op: 'move', name: g.name, args: g.args, did: dv || [g.name.replace(/_/g, ' '), g.name.replace(/_/g, ' ')]}, w => {}); break; }
        }
        if (r && r.ok === false) return r;
        if (r === false) return fail('it could not work out how to do that');
      }
      return {ok: true, ops};
    }
    function takenBy(W, p, at) {
      return E.P.find(o => o !== p && !isFurn(o) && W.P[o.id] && !W.P[o.id].worn && W.held.l !== o.id && W.held.r !== o.id && d2(W.P[o.id].pos, o.home) > 0.01
        && Math.hypot(W.P[o.id].pos[0] - at[0], W.P[o.id].pos[2] - at[2]) < (footR(o) + footR(p)) * 1.05 && Math.abs(W.P[o.id].pos[1] - at[1]) < 0.3) || null;
    }
    function freeSpotNear(pl, p, W, all) {   // a free rest spot on the place, nearest to the body (the laptop: facing the seat)
      const y = pl.top_y + p.restH, here = [W.root.x, 0, W.root.z], f = pl.prop ? byId(pl.prop) : null, F = f && feat(f);
      const extra = F && F.desk ? [-0.26, -0.13, 0, 0.13, 0.26].map(x => add3(pl.center, rotYv([x * F.desk.w / 0.7, 0, F.desk.dep / 2 - Math.max(0.06, footR(p) * 0.8)], ryOf(f)))) : [];   // along its near edge
      const cand = [...pl.spots, ...extra].map(q => [q[0], y, q[2]]).filter(q => !E.P.some(o => o !== p && !isFurn(o) && W.P[o.id] && !W.P[o.id].worn && W.P[o.id].place === pl.id && W.held.l !== o.id && W.held.r !== o.id && d2(W.P[o.id].pos, q) < (footR(o) + footR(p)) * 1.05));
      cand.sort((a, b) => d2(a, here) - d2(b, here));
      return all ? cand : cand[0] || null;
    }
    // a laptop set on a desk faces the desk's sitting side (its keys toward the chair), wherever it was put down from;
    // anywhere else it faces the body
    function laptopFaceYaw(rest, placeId, bodyAt) {
      const pl = E.places.find(x => x.id === placeId), f = pl && byId(pl.prop || pl.on || ''), F = f && feat(f);
      if (F && F.desk && F.desk.chairSpot) return yawOfDir(sub3(flat(fW(f, F.desk.chairSpot)), flat(rest)));
      return yawOfDir(sub3(bodyAt, flat(rest)));
    }
    // ---- execute one op from the live state (→ E.move / E.chain, or an instant change)
    function runOp(op, J) {
      const p = op.id ? byId(op.id) : null, src = J.src;
      switch (op.op) {
        case 'steps': { if (op.g) op.g.ran = true; const st = op.build ? op.build() : null; if (st && st.err) return {err: st.err}; if (!st || !st.length) return true; return chainOp(st, op, J); }   // teach mode
        case 'walk_to': { const P = new Plan('walk_to', {}, src); legWalk(P, op.pose, op.sp || 1, {Ts: op.Ts, keepFacing: op.keepFacing, blendYaw: op.blendYaw, turnStep: op.turnStep}); P.endBase = Object.assign({}, E.base, {x: op.pose.x, z: op.pose.z, yaw: op.pose.yaw}); P.endPosture = 'standing'; finishPlan(P, op, J); return true; }
        case 'stand_up': { const P = new Plan('stand_up', {}, src); standPlan(P, 1); finishPlan(P, op, J); return true; }
        case 'sit': { const P = new Plan('sit', {}, src); sitPlan(P, p, 1); finishPlan(P, op, J); return true; }
        case 'move_furniture': {
          J.ops.unshift({op: '_drag', id: op.id, goal: op.goal, did: op.did, sub: true}, {op: '_letgo', sub: true});
          return chainOp(gripSteps(p), op, J);
        }
        case '_drag': { const P = new Plan('drag', {}, src); dragPlan(P, p, op.goal, 1); finishPlan(P, op, J); return true; }
        case '_letgo': return chainOp(letGoSteps(), op, J);
        case 'pick_up': {
          if (p.held) return true;
          const two = op.side === 'both', s = two ? 'l' : (heldBy(op.side) ? otherSide(op.side) : op.side);
          if (!two && heldBy(s)) return {err: 'my paws are full'};
          if (!two && gripPick(p, s).e > 0.035) { const o = otherSide(s); if (!heldBy(o) && gripPick(p, o).e <= 0.035) return chainOp(chainPickUp(p, o, false), op, J); return {err: `the ${p.name} is out of my reach`}; }
          return chainOp(chainPickUp(p, s, two), op, J);
        }
        case 'place': {
          if (!p.held) return {err: `I am not holding the ${p.name}`};
          const rest = op.at && op.at.length ? op.at.slice() : null; if (!rest) return {err: 'no spot'};
          let turn = null, yawW;
          if (op.rot) { yawW = p.homeRot[1] + op.rot[1]; const h0 = p.hrot[1]; turn = h0 + wrap(wrap(op.rot[1] - rootNow().yaw) - h0); }   // an exact rest pose (back home): its own yaw there
          else if (feat(p).laptop) { const want = laptopFaceYaw(rest, op.place, flat(toWorld(E.body.C))), h0 = p.hrot[1]; turn = h0 + wrap(wrap(want - p.homeRot[1] - rootNow().yaw) - h0); }
          const ch = chainPutAt(p, p.held === 'both' ? 'l' : p.held, rest, 'place', {yawW: p.held === 'both' ? undefined : yawW});
          if (!ch) return {err: `I can't reach there`};
          if (turn !== null && Math.abs(turn - p.hrot[1]) > 1e-4) {   // turned in the paws, never snapped (a held-by-a-handle offset turns with it)
            const h0 = p.hrot[1], r0 = [p.hrot[0], p.hrot[2]], r1 = op.rot ? [op.rot[0], op.rot[2]] : r0, dur = clamp(0.45 + 0.9 * Math.abs(turn - h0), 0.6, 1.0);
            ch.unshift(stepFn('turn it to face', dur, u => { const m = mj(u); p.hrot[1] = lerp(h0, turn, m); p.hrot[0] = lerp(r0[0], r1[0], m); p.hrot[2] = lerp(r0[1], r1[1], m); if (p.grip && p.held !== 'both') p.hoff = hoffFor(p, p.grip); }));
          }
          return chainOp(ch, op, J);
        }
        case 'put_on': if (!p.held) return {err: `I am not holding the ${p.name}`}; return chainOp(wearSteps(p, true), op, J);
        case 'take_off': if (!p.worn && !p.bal) return true; if (!twoPawWear(p)) return chainOp(headOffSteps(p, op.side), op, J); return chainOp(wearSteps(p, false, op.side), op, J);
        // objects engine
        case 'head': { if (!p.held) return {err: `I am not holding the ${p.name}`}; const q = headClash(p, op.wear ? wearOff(p) : balanceOff(p), !op.wear); if (q) return {err: `the ${q.name} is in the way on its head`}; return chainOp(headSteps(p, op.wear), op, J); }
        case 'anchor': if (!p.held) return {err: `I am not holding the ${p.name}`}; if (!anchorWorld(op.name, p)) return {err: `there is no ${anchorWord(op.name).replace(/^its /, '')} to put it on`}; return chainOp(anchorSteps(p, op.name), op, J);
        case 'art': return chainOp(artSteps(p, op.open), op, J);
        case 'throw': { if (!p.held) return {err: `I am not holding the ${p.name}`}; const aim = aimPoint(p, deT(op.target)); if (aim.err) return {err: aim.err}; return chainOp(throwSteps(p, aim, op), op, J); }
        case 'juggle': { const a = byId(op.ids[0]), b = byId(op.ids[1]); if (!a.held || !b.held || a.held === b.held) return {err: 'it needs one thing in each paw to juggle'}; return chainOp(juggleSteps(a, b, op.n, op), op, J); }
        case 'use': { const q = byId(op.target); if (!p.held) return {err: `I am not holding the ${p.name}`}; if (!q) return {err: 'use it on what?'}; return chainOp(useSteps(p, q, op.motion, op), op, J); }
        case 'open': case 'close': return chainOp(lidSteps(p, op.op === 'open'), op, J);
        case 'slide': return chainOp(slideSteps(p, op.dir || 'closer'), op, J);
        case 'type': {
          const go = [lookStep(screenPoint(p), true), par(['l', 'r'].map(sd => pawPath(sd, add3(kbPoint(p, sd), [0, 0.02, 0]), 0.6, {open: 0.4, turn: -0.5, arc: 0.05})))];
          go.push(stepFn('start typing', 0.001, () => { E.activity = {kind: 'typing', p, t0: t(), glance: t() + U(3, 6), taps: {l: {next: t() + 0.15}, r: {next: t() + 0.25}}, paused: false}; E.turnBack = false;
            emit({kind: 'activity', name: 'typing', phase: 'start', src, args: {object: p.id}, did: `typing on the ${p.name}, eyes on the screen`}); }));
          for (const sd of ['l', 'r']) E.paw[sd].keep = true;
          return chainOp(go, op, J);
        }
        case 'resume': {   // back to the keys: the twist unwinds while both paws travel back to the keys (no jump), then the taps resume
          const A = E.activity; if (!A) return true;
          E.turnBack = false; const p2 = A.p;
          if (!A.paused) return true;
          return chainOp([lookStep(screenPoint(p2), true), par(['l', 'r'].map(sd => pawPath(sd, add3(kbPoint(p2, sd), [0, 0.02, 0]), 0.6, {open: 0.4, turn: -0.5, arc: 0.05}))),
            stepFn('resume typing', 0.001, () => { if (E.activity !== A) return; A.paused = false; A.glance = t() + U(3, 6); for (const sd of ['l', 'r']) E.paw[sd].keep = true;
              emit({kind: 'activity', name: A.kind, phase: 'start', src, did: op.auto ? 'turned back to the laptop and got back to typing' : 'back to typing'}); })], op, J);
        }
        case 'stop_typing': return stopActivity(J);
        case 'knock': return chainOp(knockSteps(p, op.foot), op, J);   // objects
        case 'touch': {
          if (op.T.prop && (!byId(op.T.prop) || byId(op.T.prop).held)) return {err: `the ${byId(op.T.prop) ? byId(op.T.prop).name : 'thing'} is not there to touch`};
          const st = touchSteps(op.T, op.S, op.style, op.count, op.side, op); if (st.err) return {err: st.err};
          return chainOp(st, op, J);
        }
        case 'noop': return true;
        case 'generic': { const r = objectCommand(op.cmd, J.src, J.id); if (!r.ok) return {err: r.reason}; if (r.noop) return true; if (E.chain) { E.chain.silent = true; E.chain.opOf = J; E.chain.op = op; op.did = [r.did, pastOf(r.did)]; } return true; }
        case 'move': { const r = startMove(op.name, op.args, J.src, J.id); if (!r.ok) return {err: r.reason}; op.did = [r.did, r.plan.did[1]]; E.move.opDid = op.did; E.move.opOf = J; return true; }
        case 'face': {   // turn on the spot to face him again, eyes first
          if (E.sitOn || E.posture === 'lying' || Math.abs(wrap(E.base.yaw)) <= 0.06) return true;
          const P = new Plan('turn_back', {}, src); legWalk(P, {x: E.base.x, z: E.base.z, yaw: 0}, 1); P.endBase = Object.assign({}, E.base, {yaw: 0}); P.silent = true; finishPlan(P, op, J); E.move.silent = true; return true;
        }
        case 'turn_back': {
          if (E.activity) E.activity.paused = true;
          if (E.posture === 'seated' || E.sitOn) {   // the job lasts until the twist has really arrived (its face to him)
            E.turnBack = true;
            return chainOp([{name: 'turn back', dur: 1.8, start() {}, update(tt) { return (E.tbW > 0.95 && tt > 0.3) || tt >= 1.8 || !E.turnBack; },
              end() { emit({kind: 'activity', name: 'turn_back', phase: 'end', src, did: 'turned its upper body and head back to him (still in its seat)'}); }}], op, J);
          }
          const P = new Plan('turn_back', {}, src); legWalk(P, {x: E.base.x, z: E.base.z, yaw: 0}, 1); P.endBase = Object.assign({}, E.base, {yaw: 0}); finishPlan(P, op, J); return true;
        }
      }
      return {err: 'unknown step ' + op.op};
    }
    // objects: the limb swings through the thing; at contact it leaves under the engine's flight physics
    function knockSteps(p, foot) {
      const me = flat(toWorld(E.body.C)), c0 = P(p), d = norm2(sub3(flat(c0), me)), out = [lookStep(c0, true)];
      const launch = (spd, up) => { const w = P(p); p.wvel = [d[0] * spd * 0.6, up * 0.5, d[2] * spd * 0.6]; startFlight(p, w, [d[0] * spd, up, d[2] * spd], foot ? 'kicked' : 'knocked over', {spin: [0.6 * d[2], 0, -0.6 * d[0]], flipOK: true}); for (const q of E.P) if (q.sup && q.sup.id === p.id) shed(q, `the ${p.name} was ${foot ? 'kicked' : 'knocked over'}`); };
      if (foot && E.body.legs) {
        const toS = sd => d2(E.footNow ? E.footNow[sd] || c0 : c0, c0), s = toS('l') <= toS('r') ? 'l' : 'r';
        out.push({name: 'kick', atomic: true, dur: 0.9, start() { this.F0 = (E.footNow && E.footNow[s] ? E.footNow[s] : E.feet[s]).slice(); this.hit = false; },
          update(tt) {
            const F0 = this.F0, behind = add3(F0, [-d[0] * 0.08, 0.05, -d[2] * 0.08]), hitP = [c0[0] - d[0] * (p.hx + 0.02), Math.max(0.02, c0[1] - 0.2 * p.hH), c0[2] - d[2] * (p.hx + 0.02)], thru = add3(hitP, [d[0] * 0.05, 0.03, d[2] * 0.05]);
            let F;
            if (tt < 0.3) F = lerp3(F0, behind, mj(tt / 0.3));
            else if (tt < 0.42) F = lerp3(behind, hitP, mj((tt - 0.3) / 0.12));
            else if (tt < 0.55) F = lerp3(hitP, thru, mj((tt - 0.42) / 0.13));
            else F = lerp3(thru, F0, mj((tt - 0.55) / 0.35));
            if (!this.hit && tt >= 0.42) { this.hit = true; launch(1.6, 0.5); emit({kind: 'object_event', name: 'kicked', phase: 'end', src: 'auto', args: {object: p.id}, did: `kicked the ${p.name}`}); }
            E.feet[s] = F; return tt >= 0.9;
          }, end() { E.feet[s] = this.F0.slice(); }});
      } else {
        const s = toLocal(c0)[0] < 0 ? 'l' : 'r', near = [c0[0] - d[0] * (p.hx + 0.03), c0[1] + 0.4 * p.hH, c0[2] - d[2] * (p.hx + 0.03)];
        if (E.body.legs) out.push(bendStep(near));
        out.push(pawPath(s, add3(near, [-d[0] * 0.06, 0.04, -d[2] * 0.06]), 0.55, {open: 0.8, turn: 0, look: c0}));
        out.push(pawPath(s, near, 0.18, {arc: 0}));
        out.push(stepFn('push', 0.001, () => { launch(1.3, 1.1); emit({kind: 'object_event', name: 'knocked_over', phase: 'end', src: 'auto', args: {object: p.id}, did: `knocked the ${p.name} over`}); }, {atomic: true}));
        out.push(pawPath(s, add3(near, [d[0] * 0.06, 0.02, d[2] * 0.06]), 0.2, {arc: 0}));
        if (E.body.legs) out.push(bendStep(null));
        out.push(retractStep(s));
      }
      out.push(dwellStep('watch it land', 1.2, () => { E.look.P = P(p); }));
      out.push(lookStep(null, false));
      return out;
    }
    // ---- CONTACT with a thing (touch / tap / knock on / pat / poke / press / stroke / lean on) and GO-TO.
    // General: the contact spot comes from the thing's geometry (a furniture top from its place rect, a loose thing's top
    // from afford.surface / its size, a board / screen face from its plane), never from the thing's name.
    const TOUCH_STYLES = {tap: 1, knock: 1, pat: 1, poke: 1, press: 1, stroke: 1, lean: 1, touch: 1};
    const TOUCH_N = {tap: 2, knock: 3, pat: 3, poke: 1, press: 1, stroke: 2, lean: 1, touch: 1};
    const TOUCH_WORDS = {tap: ['tapping', 'tapped'], knock: ['knocking on', 'knocked on'], pat: ['patting', 'patted'], poke: ['poking', 'poked'], press: ['pressing on', 'pressed on'],
      stroke: ['stroking', 'stroked'], lean: ['leaning on', 'leaned on'], touch: ['touching', 'touched']};
    const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], norm3 = a => { const l = len3(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const GRASP_N = 0.07 / 2.3, PAW_R = 0.022, TOUCH_DEEP = 0.0;   // TOUCH_DEEP: where the planner checks the reach (the grasp point at the surface)   // the grasp point sits 0.07 scene off the palm (muse.body.json arm.grasp); the paw's half-thickness
    const timesWord = n => n === 1 ? 'once' : n === 2 ? 'twice' : `${n} times`;
    function rawPlace(id) { const a = E.props && Array.isArray(E.props.places) ? E.props.places : []; return a.find(q => q && q.id === id) || null; }
    // a thing or a place by words: {prop} | {place} | {home} | {err}. Places the planner can't put things on (the TV) count.
    const PREP_RE = /^(?:(?:over|up|right|next|close|on|onto|to|at|by|toward|towards|against|the|in front of|front of)\s+)+/;   // "to the board", "on desk"
    function touchTarget(desc, cmd) {
      if (typeof desc === 'string') desc = desc.toLowerCase().replace(PREP_RE, '');
      else if (desc && typeof desc === 'object' && typeof desc.name === 'string' && !desc.id) desc = Object.assign({}, desc, {name: desc.name.toLowerCase().replace(PREP_RE, '')});
      const nm = desc && typeof desc === 'object' ? (desc.id || desc.name || desc.place || '') : String(desc || '');
      const pl0 = cmd && (typeof cmd.place === 'string' ? cmd.place : cmd.place && (cmd.place.id || cmd.place.name));
      const words = String(nm || pl0 || '').toLowerCase().replace(/[_-]+/g, ' ').trim();
      if (/^(home|back|your spot|its spot|the middle|me|him|you|user|here|over here|the camera|camera)$/.test(words)) return {home: true};
      // a place named exactly ("the board") beats a thing whose name only contains the word ("board eraser")
      const bare = x => String(x || '').toLowerCase().replace(/[_-]+/g, ' ').replace(/^the /, '').trim(), wb = bare(words);
      const exactP = wb && (E.props && Array.isArray(E.props.places) ? E.props.places : []).find(q => q && q.id && q.id !== 'floor' && [q.id, q.name, ...(q.aliases || [])].some(a => bare(a) === wb));
      const exactT = wb && E.P.find(q => [q.id, q.name, ...(q.def.aliases || [])].some(a => bare(a) === wb));
      if (exactP && !exactT && !(desc && typeof desc === 'object' && (desc.id || desc.color))) { const f = exactP.on_prop || exactP.prop; return f && byId(f) ? {prop: f} : {place: exactP.id}; }
      if (desc && !(typeof desc === 'object' && desc.place && !desc.id && !desc.name)) {
        const r = resolveObject(desc, 'touch', {});
        if (r.p) return {prop: r.p.id};
        if (r.ask) return {err: r.err, ask: true};
      }
      const raws = E.props && Array.isArray(E.props.places) ? E.props.places : [];
      const w = toks(words).filter(x => !/^(the|a|on|onto|in|into|at|to|top|of|over|by)$/.test(x));
      if (w.length) for (const q of raws) {
        if (!q || !q.id || q.id === 'floor') continue;
        const N = new Set([...toks(q.id), ...toks(q.name), ...(q.aliases || []).flatMap(toks)]);
        if (w.every(x => N.has(x)) || (w.length === 1 && N.has(w[0]))) {
          const f = q.on_prop || q.prop; if (f && byId(f)) return {prop: f};   // the desk's top = the desk
          return {place: q.id};
        }
      }
      if (w.length && /^(floor|ground|rug|carpet)$/.test(w.join(' '))) return {err: "I'd rather not paw at the floor"};
      return {err: words ? `there is no ${words} here` : 'which thing?', ask: !words};
    }
    const targetName = T => T.prop ? 'the ' + byId(T.prop).name : T.place ? (() => { const q = rawPlace(T.place); const n = q && q.name || T.place; return /^the /.test(n) ? n : 'the ' + n; })() : 'its spot';
    // the best spot on a horizontal rect (centre c, top y, size [w, d], yaw a): nearest the body, clear of what rests there
    function topSpot(c, y, w, d, a, here, placeId) {
      const m = Math.min(0.06, w / 4, d / 4), out = [];
      const lying = E.P.filter(q => !q.held && !q.worn && !q.flying && !isFurn(q) && (q.place === placeId || Math.abs(P(q)[1] - q.hH - y) < 0.02));
      for (let i = 0; i <= 8; i++) for (let j = 0; j <= 4; j++) {
        const lx = -w / 2 + m + (w - 2 * m) * i / 8, lz = -d / 2 + m + (d - 2 * m) * j / 4, Q = add3([c[0], y, c[2]], rotYv([lx, 0, lz], a));
        if (lying.some(q => { const qp = P(q); return Math.hypot(qp[0] - Q[0], qp[2] - Q[2]) < footR(q) + 0.07; })) continue;
        out.push({C: Q, cost: Math.hypot(Q[0] - here[0], Q[2] - here[2])});
      }
      out.sort((u, v) => u.cost - v.cost);
      return out.length ? {C: out[0].C, n: [0, 1, 0], t: norm2(rotYv([1, 0, 0], a)), amp: Math.min(0.07, w / 4)} : null;
    }
    // where a paw touches T, seen from `here`: {C world, n outward normal, t stroke direction, amp, rel (prop-local C)} | {err}
    function touchSpot(T, here) {
      if (T.prop) {
        const p = byId(T.prop); if (!p) return {err: 'there is no such thing here'};
        if (p.held) return {err: `the ${p.name} is in my paw`};
        if (p.worn || p.bal) return {err: `the ${p.name} is on my head`};
        if (isFurn(p)) {
          const F = feat(p), pl = placeOfFurn(p);
          let S = null;
          if (F.desk && pl) S = topSpot(pl.center, pl.top_y, pl.size[0], pl.size[1], ryOf(p), here, pl.id);
          else if (F.chair) { const st = fW(p, F.chair.seat); S = topSpot(st, st[1], F.chair.w * 0.6, F.chair.dep * 0.6, ryOf(p), here, pl ? pl.id : null); }
          else { const top = F.desk ? F.desk.h : 0.4; S = topSpot(p.pos, top, 0.3, 0.3, ryOf(p), here, pl ? pl.id : null); }
          if (!S) return {err: `there is no free spot on the ${p.name} to touch`};
          return Object.assign(S, {rel: rotYv(sub3(S.C, p.pos), -ryOf(p))});
        }
        const c = P(p), riders = E.P.filter(q => q.sup && q.sup.id === p.id && !q.held && !q.inside), d = norm2(sub3(here, flat(c)));
        if (riders.length) { const C = add3(c, mul3(d, p.hx)); return {C, n: d, t: [0, 1, 0], amp: Math.min(0.03, p.hH * 0.5), rel: rotYv(sub3(C, c), -ryOf(p)), side: true}; }   // its top is taken: its near side
        const C = add3(c, [0, topOff(p), 0]);
        return {C, n: [0, 1, 0], t: norm2([-d[2], 0, d[0]]), amp: Math.min(0.05, Math.max(0.015, p.hx * 0.6)), rel: rotYv(sub3(C, c), -ryOf(p))};
      }
      if (T.place) {
        const q = rawPlace(T.place); if (!q) return {err: 'there is no such place here'};
        const Q = q.plane || q.screen;
        if (Q && Array.isArray(Q.origin) && Array.isArray(Q.u) && Array.isArray(Q.n) && Array.isArray(Q.size)) {   // a face (board, screen): the paw goes against it
          const o = Q.origin, u = Q.u, n = norm2(Q.n), sz = Q.size, v = Q.v || [0, -1, 0];
          const a = clamp(((here[0] - o[0]) * u[0] + (here[2] - o[2]) * u[2]) / sz[0], 0.12, 0.88);
          const yTop = o[1], yBot = o[1] + v[1] * sz[1], yT = clamp(0.5, Math.min(yTop, yBot) + 0.04, Math.max(yTop, yBot) - 0.04);
          const b = clamp((yT - o[1]) / (v[1] * sz[1] || -1), 0, 1);
          const C = [o[0] + u[0] * a * sz[0] + v[0] * b * sz[1], o[1] + u[1] * a * sz[0] + v[1] * b * sz[1], o[2] + u[2] * a * sz[0] + v[2] * b * sz[1]];
          return {C, n, t: norm2(u), amp: 0.06, face: true};
        }
        const pl = E.places.find(x => x.id === T.place), c = v3(q.center) || (pl && pl.center), y = Number.isFinite(+q.top_y) ? +q.top_y : c && c[1];
        if (!c) return {err: `I can't tell where ${targetName(T)} is`};
        const sz = Array.isArray(q.size) ? q.size.map(Number) : [0.3, 0.3];
        const S = topSpot(c, y, sz[0], sz[1], +q.yaw || 0, here, q.id);
        return S || {err: `there is no free spot on ${targetName(T)} to touch`};
      }
      return {err: 'touch what?'};
    }
    function spotNow(T, S) {   // a spot chosen on a thing: where it is now (the thing may have moved since)
      if (T.prop && S.rel) { const p = byId(T.prop); if (p && !p.held) { const c = isFurn(p) ? p.pos : P(p); return Object.assign({}, S, {C: add3(c, rotYv(S.rel, ryOf(p)))}); } }
      return S;
    }
    // the grasp point that puts the paw's touching surface on C (normal n), given the palm normal pn (world)
    function graspFor(C, n, pn) { const k = pn ? Math.max(0, -dot3(pn, n)) : 0; return add3(add3(C, mul3(n, PAW_R * (1 - k))), pn ? mul3(pn, GRASP_N) : mul3(n, -GRASP_N * 0)); }
    function pawSurfGap(s, C, n) {   // the drawn paw's touching surface vs the contact plane (H, + = above it)
      const g = E.pawGW && E.pawGW[s]; if (!g) return null;
      const pn = pawN(s), k = pn ? Math.max(0, -dot3(pn, n)) : 0, pc = pn ? sub3(g, mul3(pn, GRASP_N)) : g;
      return dot3(sub3(pc, C), n) - PAW_R * (1 - k);
    }
    function palmNormalW(s, h) { const nl = E.reach[s].normal(h); return nl ? norm3(sub3(toWorld(add3([0, 0.5, 0], nl)), toWorld([0, 0.5, 0]))) : null; }
    function touchSteps(T, S0, style, count, side, op) {
      const S = spotNow(T, S0), C = S.C, n = S.n, tdir = S.t || [1, 0, 0];
      let s = side || (toLocal(C)[0] < 0 ? 'l' : 'r');
      if (heldBy(s)) s = otherSide(s);
      if (heldBy(s)) return {err: 'my paws are full'};
      const shape = {knock: {open: 0.0, turn: 0.1, point: 0}, tap: {open: 0.35, turn: 0, point: 0.8}, poke: {open: 0.3, turn: 0, point: 1}, pat: {open: 0.95, turn: -0.3, point: 0},
        press: {open: 0.9, turn: -0.3, point: 0}, stroke: {open: 0.9, turn: -0.3, point: 0}, lean: {open: 0.95, turn: -0.3, point: 0}, touch: {open: 0.75, turn: -0.2, point: 0}}[style];
      const lift = {knock: 0.045, tap: 0.035, poke: 0.06, pat: 0.055, press: 0.04, stroke: 0.04, lean: 0.05, touch: 0.04}[style];
      const per = {knock: 0.3, tap: 0.34, poke: 0.6, pat: 0.5, press: 1.1, stroke: 1.1, lean: 1.8, touch: 0.8}[style];
      const nm = targetName(T), out = [lookStep(C, true)];
      const G0 = add3(C, mul3(n, PAW_R + GRASP_N));
      if (E.body.legs) out.push(bendStep(add3(C, mul3(n, TOUCH_DEEP))));
      let G = G0, pn = null;
      const fix = () => { const r = invW(s, G0, E.paw[s].hPrev, 14); pn = palmNormalW(s, r.h); G = graspFor(C, n, pn); const r2 = invW(s, G, r.h, 14); const pn2 = palmNormalW(s, r2.h); if (pn2) { pn = pn2; G = graspFor(C, n, pn); } };
      const above = () => add3(G, mul3(n, lift));
      // a top (normal up): the paw rises beside the body first and comes in OVER the thing, and leaves the same way (never
      // through its edge); a face (a board): straight in from the front
      const upN = n[1] > 0.7, Q0s = {};
      if (upN) out.push({name: 'raise paw', dur: 0.45, start() { fix(); const Q0 = fwdW(s, E.paw[s].hPrev); Q0s.rest = Q0; this.Q0 = Q0; this.Q1 = [Q0[0], Math.max(Q0[1], above()[1] + 0.04), Q0[2]]; this.p0 = E.paw[s].point || 0; this.o0 = E.paw[s].open; },
        update(tt) { const pw = E.paw[s], u = clamp(tt / 0.45, 0, 1), m = mj(u); pw.ctrlT = 1; pw.tgt = lerp3(this.Q0, this.Q1, m); pw.hd = null; pw.point = lerp(this.p0, shape.point, m); pw.open = lerp(this.o0, shape.open, m); return u >= 1; }});
      out.push({name: 'reach to touch', dur: 0.7, start() { fix(); const a = add3(above(), mul3(n, 0.03)); this.p = upN ? null : pawPath(s, a, 0.6, {open: shape.open, turn: shape.turn, look: C}); if (this.p) this.p.start();
        this.Q0 = fwdW(s, E.paw[s].hPrev); this.Q1 = [a[0], Math.max(a[1], this.Q0[1]), a[2]]; this.pt0 = E.paw[s].point || 0; this.t0 = E.paw[s].turn; },
        update(tt) { const pw = E.paw[s]; let d; if (this.p) d = this.p.update(tt); else { const u = clamp(tt / 0.6, 0, 1); pw.ctrlT = 1; pw.tgt = lerp3(this.Q0, this.Q1, mj(u)); pw.hd = null; pw.turn = lerp(this.t0, shape.turn, mj(u)); d = u >= 1; }
          pw.point = lerp(this.pt0, shape.point, mj(clamp(tt / 0.5, 0, 1))); return d; }});
      out.push({name: 'line up', dur: 0.25, start() { fix(); this.Q0 = fwdW(s, E.paw[s].hPrev); }, update(tt) {   // straight in over the spot (the palm normal now known)
        const pw = E.paw[s], u = clamp(tt / 0.25, 0, 1); pw.tgt = lerp3(this.Q0, above(), mj(u)); pw.hd = null; pw.ctrlT = 1; return u >= 1; }});
      const contacts = [];
      out.push({name: style, atomic: false, dur: count * per + 0.2, start() { this.n = 0; this.b0 = E.bendT ? Object.assign({}, E.bendT) : {y: 0, p: 0}; },
        update(tt) {
          const pw = E.paw[s], i = Math.floor(tt / per), ph = clamp(tt / per - i, 0, 1), up = above();
          let Q = up, onIt = false;
          if (i < count) {
            if (style === 'knock' || style === 'tap' || style === 'pat') {   // rap: down quick, a short touch, back up
              const dn = style === 'pat' ? 0.4 : 0.3, hold = style === 'pat' ? 0.25 : 0.2;
              const k = ph < dn ? mj(ph / dn) : ph < dn + hold ? 1 : 1 - mj((ph - dn - hold) / (1 - dn - hold));
              Q = lerp3(up, G, k); onIt = ph >= dn && ph < dn + hold;
            } else if (style === 'poke') {
              const k = ph < 0.3 ? mj(ph / 0.3) : ph < 0.55 ? 1 : 1 - mj((ph - 0.55) / 0.45);
              Q = add3(lerp3(up, G, k), mul3(n, -0.004 * (ph >= 0.3 && ph < 0.55 ? 1 : 0))); onIt = ph >= 0.3 && ph < 0.55;
            } else if (style === 'stroke') {   // down, slide along and back, up
              const k = ph < 0.18 ? mj(ph / 0.18) : ph < 0.82 ? 1 : 1 - mj((ph - 0.82) / 0.18), sl = ph < 0.18 ? -1 : ph < 0.82 ? -Math.cos(TAU * (ph - 0.18) / 0.64) : -1;
              Q = add3(lerp3(up, G, k), mul3(tdir, S.amp * sl * (count > 0 ? 1 : 0))); onIt = ph >= 0.18 && ph < 0.82;
            } else {   // press / lean / touch: down, rest there, up
              const k = ph < 0.25 ? mj(ph / 0.25) : ph < 0.8 ? 1 : 1 - mj((ph - 0.8) / 0.2);
              Q = add3(lerp3(up, G, k), mul3(n, style === 'press' ? -0.003 * (ph >= 0.25 && ph < 0.8 ? 1 : 0) : 0)); onIt = ph >= 0.25 && ph < 0.8;
              if (style === 'lean' && E.body.legs) { const lk = ph < 0.25 ? 0 : ph < 0.8 ? mj(clamp((ph - 0.25) / 0.2, 0, 1)) : 1 - mj((ph - 0.8) / 0.2); E.bendT = {y: this.b0.y, p: this.b0.p + 0.3 * lk}; }
            }
          }
          pw.tgt = Q; pw.hd = null; pw.ctrlT = 1;
          pw.open = lerp(pw.open, shape.open, 0.3); pw.turn = lerp(pw.turn, shape.turn, 0.3); pw.point = lerp(pw.point || 0, shape.point, 0.3);
          E.look.P = C;
          if (onIt && this.n <= i && i < count) { this.n = i + 1; contacts.push(t()); if (i === 0) emit({kind: 'object_event', name: 'touched', phase: 'start', src: 'auto', args: T.prop ? {object: T.prop, style} : {place: T.place, style}, did: `${TOUCH_WORDS[style][0]} ${nm}`}); }
          E.touchNow = onIt ? {s, C: C.slice(), n: n.slice(), style} : null;
          E.resultView = {P: C.slice(), until: t() + 0.9};   // the camera eases in on the contact (live v2-v5: the knock on the desk read as 'paw hidden')
          return tt >= count * per + 0.2;
        }, end() { E.touchNow = null; if (style === 'lean' && E.body.legs) E.bendT = Object.assign({}, this.b0); }});
      out.push(stepFn('touched', 0.001, () => { op.did = [op.did[0], `${TOUCH_WORDS[style][1]} ${nm}${count > 1 || /knock|tap|pat|poke/.test(style) ? ' ' + timesWord(contacts.length || count) : ''}`]; }));
      if (upN) out.push({name: 'lift off', dur: 0.5, start() { this.Q0 = fwdW(s, E.paw[s].hPrev); const R = Q0s.rest || this.Q0; this.Q1 = [R[0], Math.max(this.Q0[1], above()[1] + 0.04), R[2]]; },
        update(tt) { const pw = E.paw[s], u = clamp(tt / 0.5, 0, 1); pw.ctrlT = 1; pw.tgt = lerp3(this.Q0, this.Q1, mj(u)); pw.hd = null; pw.point = lerp(pw.point || 0, 0, u); return u >= 1; }});
      if (E.body.legs) out.push(bendStep(null));
      out.push(stepFn('point back', 0.25, u => { E.paw[s].point = lerp(E.paw[s].point || 0, 0, u); }));
      out.push(lookStep(null, false));
      out.push(retractStep(s));
      return out;
    }
    function stopActivity(J) {
      const A = E.activity; if (!A) return true;
      for (const sd of ['l', 'r']) if (!E.paw[sd].follow) E.paw[sd].keep = false;
      E.activity = null; E.turnBack = false; E.look.wT = 0;
      for (const sd of ['l', 'r']) if (E.paw[sd].act) { E.paw[sd].act = false; }
      emit({kind: 'activity', name: A.kind, phase: 'end', src: J ? J.src : 'auto', did: `stopped typing (after ${Math.round(t() - A.t0)} s)`});
      return chainOp([par([retractStep('l'), retractStep('r')])], {op: 'stop_typing', did: ['', ''], sub: true}, J || {src: 'auto', ops: []});
    }
    function finishPlan(P, op, J) { P.id = J.id; P.did = op.did; P.dur = P.te(); P.silent = !!op.sub; P.opOf = J; P.desc = {}; E.move = P; E.active = true; }
    function chainOp(steps, op, J) { E.chain = {verb: op.op, p: op.id ? byId(op.id) : (E.P[0] || {name: ''}), steps, i: -1, cur: null, t0: t(), did: (op.did || [''])[0], src: J.src, id: J.id, args: {}, started: t(), silent: true, opOf: J, op}; return true; }
    // the job queue (one per command; FIFO so step-by-step instructions run in order)
    E.jobs = [];
    // the prop a job acts on (its first goal naming one), and the props of the object jobs still queued or running —
    // newest first — so a pronoun in the next step binds to it before it has happened
    function jobProp(J) { for (const g of (J && J.goals) || []) { const q = g && g.id ? byId(g.id) : null; if (q) return q; } return null; }
    function pendingProps() { const out = []; for (let i = E.jobs.length - 1; i >= 0; i--) { const J = E.jobs[i]; if (J.stopped || J.kind !== 'object') continue; const q = jobProp(J); if (q && !out.includes(q)) out.push(q); } return out; }
    function pendingProp() { return E.jobs && E.jobs.length ? pendingProps()[0] || null : null; }
    function jobFrame() {
      // a held end state (turned away, lying, …) gives way to the next job once its own motion is over
      if (E.move && E.move.holdEnd && !E.chain && E.jobs.some(j => j !== E.move.opOf) && t() - E.move.t0 >= E.move.dur) finishMove(E.move, 'end');
      if (E.move || E.chain) return;
      const J = E.jobs[0]; if (!J) return;
      if (!J.ops) {
        const r = planGoals(J.goals);
        if (!r.ok) { E.jobs.shift(); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'fail', args: J.args, reason: r.reason}); J.result = r; return; }
        J.ops = r.ops; if (!J.done) J.done = [];   // (a re-plan keeps what was already done: the result stays complete)
        if (!J.started) { J.started = true; J.t0 = t(); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'start', args: J.args, did: describe(r.ops, 0) || J.noopDid || 'nothing to do — it already is'}); }
      }
      // the last job done standing with its back (or side) to him: it turns back to face him (eyes lead) — unless
      // the request was to turn away / hold a pose, or it is lying, seated or working
      if (!J.ops.length && !J.faced && !J.stopped && J.kind === 'object' && E.jobs.length === 1 && !E.sitOn && E.posture !== 'lying' && !E.activity &&
          Math.abs(wrap(E.base.yaw)) > 0.06 && !(J.args && J.args.hold) && !(J.goals || []).some(g => g && (g.g === 'touch' || g.g === 'go_to'))) { /* at a thing it stays facing it */ J.faced = true; J.ops.push({op: 'face', sub: true, did: ['', '']}); }
      if (J.failErr) { E.jobs.shift(); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'fail', args: J.args, reason: J.failErr, did: describe(J.done, 1)}); J.result = {ok: false, reason: J.failErr}; return; }
      const op = J.ops.shift();
      if (!op) { E.jobs.shift(); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: J.stopped ? 'interrupted' : 'end', args: J.args, did: ((dd) => dd + (J.stopped && !/\bstop(ped)?\b|cut short|relaxed/.test(dd) ? (J.replacedWhy ? ` (then stopped: ${J.replacedWhy})` : ' (then stopped)') : J.stopped && J.replacedWhy && !/replaced|relax/.test(dd) ? ` (${J.replacedWhy})` : ''))(describe(J.done, 1) || J.noopDid || (J.stopped ? 'nothing yet' : 'it already was that way')), reason: J.replacedWhy, state: stateBrief()}); return; }
      const r = runOp(op, J);
      if (E.dbg) E.dbg(op, r);
      if (r && r.err) {
        if (!J.replanned) { J.replanned = true; J.ops = null; return; }
        E.jobs.shift(); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'fail', args: J.args, reason: r.err, did: describe(J.done, 1)}); return;
      }
      if (!op.sub || op.op === '_drag') J.done.push(op);
    }
    function describe(ops, past) {
      const ds = ops.filter(o => o.did && o.did[past] && !(past && o.op === 'stop_typing' && false)).map(o => o.did[past]);
      const out = []; for (const d of ds) if (out[out.length - 1] !== d) out.push(d);
      return out.length <= 1 ? out.join('') : out.slice(0, -1).join(', ') + ', then ' + out[out.length - 1];
    }
    function stateBrief() { const s = stateOut(); return {posture: s.posture, sitting: s.sittingOn, wearing: s.wearing, holding: s.holding, activity: s.activity}; }
    // object command (muse) → goals
    // "put it back / away / down where it was": where to? 'last' = where it last rested, 'home' = its own spot, null = not said
    function backWhere(cmd) {
      const words = [cmd.place, cmd.target, cmd.where, cmd.to, cmd.back, cmd.from].map(v => v && typeof v === 'object' ? [v.ref, v.name, v.id, v.place].filter(Boolean).join(' ') : v).filter(v => typeof v === 'string').join(' ').toLowerCase().replace(/[_-]+/g, ' ');
      if (!words) return null;
      if (/where (it|they|you|i|that|this|those|these)( \w+)? (was|were|got|found|picked|came|had)|came from|got (it|them) from|original|previous|last|before|same (place|spot)/.test(words)) return 'last';
      if (/^\s*(home|back|away|its (place|spot|home)|their (place|spot|home)|where (it|they) belongs?|(its|their|the) stand|in place)\s*$/.test(words)) return 'home';
      return null;
    }
    // ---- objects engine: the general object verbs → goals (examples are never special-cased: words map to primitives)
    const XVERB = {stack: 'rel', stack_up: 'rel', pile: 'rel', put_in: 'rel', put_inside: 'rel', put_into: 'rel', insert: 'rel', drop_in: 'rel', place_in: 'rel', pack: 'rel', store: 'rel', set_on: 'rel', lay: 'rel',
      balance: 'balance', throw: 'throw', toss_to: 'throw', throw_to: 'throw', chuck: 'throw', lob: 'throw', fling: 'throw', hurl: 'throw', pitch: 'throw', toss: 'throw',
      juggle: 'juggle', water: 'water', pour_on: 'use', apply: 'use', rub: 'use', wipe: 'use', polish: 'use', brush: 'use', tap: 'use', press: 'use', poke: 'use', dab: 'use',
      eat: 'nonsense', bite: 'nonsense', chew: 'nonsense', swallow: 'nonsense', lick: 'nonsense', break: 'nonsense', smash: 'nonsense', cut: 'nonsense', burn: 'nonsense', tear: 'nonsense'};
    const targetText = cmd => { const tg = cmd.target ?? cmd.place ?? cmd.to ?? cmd.onto ?? cmd.into ?? cmd.on; return (typeof tg === 'object' && tg ? [tg.ref, tg.name, tg.id, tg.place, tg.anchor].filter(x => typeof x === 'string').join(' ') : String(tg ?? '')).toLowerCase().replace(/[_-]+/g, ' ').trim(); };
    const relFromText = w => { const m = /^(on top of|on to|onto|on|into|in to|inside|in|underneath|under|beneath|below|behind|in front of|next to|beside|to the left of|left of|to the right of|right of)\b/.exec(w || ''); return m ? normRel(m[1]) : null; };
    const UPWORDS = /^(up|up high|high|the air|air|in the air|up in the air|into the air|ceiling|the ceiling|sky|the sky|overhead)$/;
    const hasTarget = cmd => [cmd.target, cmd.to, cmd.onto, cmd.into, cmd.on].some(v => v !== undefined && v !== null && v !== '') || (cmd.place !== undefined && cmd.place !== null && cmd.place !== '');
    function xVerb(V, vraw, cmd, p) {
      let X = XVERB[vraw] || null;
      const rl = normRel(cmd.relation ?? cmd.rel ?? cmd.prep ?? cmd.preposition) || relFromText(targetText(cmd)), tg = hasTarget(cmd) && !UPWORDS.test(targetText(cmd));
      if ((X === 'throw') && !tg) return null;   // "toss it / throw it up and catch it": the toss-and-catch chain
      if (X === 'use' && !tg && !(p && AFF(p).use) && !(cmd.with || cmd.tool || cmd.using || cmd.instrument)) return null;   // ("tap the plant WITH the can" = the tool path)
      if (/^(pour|pour_out|empty|sprinkle|spray)$/.test(vraw) && tg) X = 'use';
      if (/^hide$/.test(vraw) && tg) { const T = resolveTarget(cmd, p); if (T && !T.err && !T.anchor) X = 'rel'; }
      if (!X && V === 'work' && p && !feat(p).laptop && (tg || AFF(p).use)) X = 'use';
      if (!X && (V === 'place' || V === 'put_on' || V === 'pull' || V === 'push' || V === 'slide') && (rl || tg)) {
        if (V !== 'place' && V !== 'put_on' && p && isFurn(p) && !rl) return null;   // furniture: pushed / pulled toward what it names (existing)
        const T = tg ? resolveTarget(cmd, p) : null;
        if ((rl && rl !== 'on' && rl !== 'in') || (T && T.self)) X = 'rel';   // (a thing onto itself: refused there, truthfully)
        else if (T && (T.err && !T.ask ? false : true) && (T.prop || T.anchor || T.me || T.err || (T.place && rl))) X = 'rel';
        else if (rl === 'in') X = 'rel';
        if (X === 'rel' && T && T.place && !rl && V !== 'put_on') X = null;   // "put it on the desk": the place path it always took
      }
      return X;
    }
    function xObject(X, p, cmd, vraw) {
      const rl = normRel(cmd.relation ?? cmd.rel ?? cmd.prep ?? cmd.preposition) || relFromText(targetText(cmd));
      if (!p && X !== 'juggle' && X !== 'water') {
        // "put it in the box" with the thing omitted = what it holds / is fetching, never the target itself
        let tq = null; try { const T0 = hasTarget(cmd) ? resolveTarget(cmd, null) : null; tq = T0 && T0.prop || null; } catch (e) { tq = null; }
        const cand = [...E.P.filter(q => q.held), ...pendingProps()].filter(q => q !== tq && !isFurn(q));
        p = cand[0] || pendingProp() || (E.P.filter(q => q.held).length === 1 ? E.P.find(q => q.held) : null);
      }
      if (X === 'nonsense') return {ok: false, reason: p ? `I can't ${vraw.replace(/_/g, ' ')} the ${p.name}${/eat|bite|chew|swallow|lick/.test(vraw) ? ` — it isn't food` : ` — I'd rather keep it in one piece`}` : `I can't ${vraw.replace(/_/g, ' ')} that`};
      if (X === 'rel') {
        if (!p) return {ok: false, reason: 'which thing?', ask: true};
        const T = resolveTarget(cmd, p) || (rl === 'in' ? {err: 'in what?', ask: true} : {err: 'where should it go?', ask: true});
        if (T.err) return {ok: false, reason: T.err, ask: T.ask};
        let r = T.anchor ? null : rl || (/in$|inside|into|insert|pack|store/.test(vraw) ? 'in' : 'on');
        if (!rl && r === 'on' && T.prop && isCont(T.prop) && T.prop !== p) {   // onto an OPEN (or being-opened) container = into it, if it fits
          const q = T.prop, C = AFF(q).container, opening = E.jobs.some(J => !J.stopped && (J.goals || []).some(g => g.g === 'lid' && g.id === q.id && g.open));
          if ((contOpen(q) || opening || !C.needs_open) && p.hx <= C.inner[0] + 1e-3 && p.hz <= C.inner[2] + 1e-3) r = 'in';
        }
        const vb = /^stack|pile/.test(vraw) ? 'stacking' : /^hide/.test(vraw) ? 'hiding' : 'putting';
        return {ok: true, goals: [{g: 'rel', id: p.id, T: serT(Object.assign({}, T, {rel: r})), vb}], did: null};
      }
      if (X === 'balance') {
        if (!p) return {ok: false, reason: 'balance what?', ask: true};
        const T = hasTarget(cmd) ? resolveTarget(cmd, p) : {anchor: 'head'};
        if (T && T.err) return {ok: false, reason: T.err, ask: T.ask};
        if (T && T.anchor && T.anchor !== 'head') return {ok: true, goals: [{g: 'rel', id: p.id, T: serT(T)}], did: null};
        if (T && (T.prop || T.place)) return {ok: true, goals: [{g: 'rel', id: p.id, T: serT(Object.assign({}, T, {rel: 'on'})), vb: 'putting'}], did: null};
        return {ok: true, goals: [{g: 'head', id: p.id, mode: 'balance'}], did: null};
      }
      if (X === 'throw') {
        if (!p) return {ok: false, reason: 'throw what?', ask: true};
        const T = resolveTarget(cmd, p); if (T && T.err) return {ok: false, reason: T.err, ask: T.ask};
        return {ok: true, goals: [{g: 'throw', id: p.id, T: serT(T ? Object.assign({}, T, {rel: rl || T.rel}) : {me: true})}], did: null};
      }
      if (X === 'juggle') {
        const list = [];
        const add = d => { if (!d) return null; const r = resolveObject(d, 'juggle', {}); if (r.p) { if (!list.includes(r.p)) list.push(r.p); return null; } if (r.cands) { for (const c of r.cands) if (!list.includes(c) && list.length < 2) list.push(c); return null; } return r.err; };
        for (const d of [].concat(Array.isArray(cmd.objects) ? cmd.objects : [], cmd.object || [], cmd.target || [])) { const e = add(d); if (e) return {ok: false, reason: e}; }
        for (const q of E.P.filter(q => q.held)) if (list.length < 2 && !list.includes(q)) list.push(q);
        if (list.length < 2) { const here = [E.base.x, 0, E.base.z]; for (const q of E.P.filter(q => !isFurn(q) && !list.includes(q) && q.def.grip !== 'two_hand' && q.def.throwable !== false).sort((a, b) => d2(a.pos, here) - d2(b.pos, here))) if (list.length < 2) list.push(q); }
        if (list.length < 2) return {ok: false, reason: 'juggling needs two things to throw'};
        if (list.length > 2) return {ok: false, reason: `it can juggle two things at once — which two of ${list.map(q => 'the ' + q.name).join(', ')}?`, ask: true};
        const n = clamp(Math.round(2 * num(cmd.count, 3)), 2, 16);
        return {ok: true, goals: [{g: 'juggle', ids: list.map(q => q.id), n, id: list[0].id}], did: null};
      }
      if (X === 'water' || X === 'use') {
        // water X = pour water on X with a tool that pours water (afford.pour.what) and lists X (afford.use.targets) if any does
        let tool = null, tgt = null;
        const wd = cmd.with || cmd.tool || cmd.using || cmd.instrument;
        if (X === 'water') {
          tgt = p || (hasTarget(cmd) ? (resolveTarget(cmd, null) || {}).prop : null) || E.P.find(q => AFF(q).water_target) || null;
          if (!tgt) return {ok: false, reason: 'water what?', ask: true};
          if (wd) { const r = resolveObject(wd, 'with', {}); if (r.err) return {ok: false, reason: r.err, ask: r.ask}; tool = r.p; }
          else { const ws = E.P.filter(q => AFF(q).pour && /water/.test(AFF(q).pour.what || '')); tool = ws.find(q => (AFF(q).use && AFF(q).use.targets || []).includes(tgt.id)) || ws[0] || null; }
          if (!tool) return {ok: false, reason: `there is nothing here to water the ${tgt.name} with`};
          if (tool === tgt) return {ok: false, reason: `I can't water the ${tgt.name} with itself`};
          return {ok: true, goals: [{g: 'use', id: tool.id, tool: tool.id, target: tgt.id, motion: 'pour'}], did: null};
        }
        if (wd && p) { const r = resolveObject(wd, 'with', {}); if (r.err) return {ok: false, reason: r.err, ask: r.ask}; tool = r.p; tgt = p; }
        else { tool = p; const T = hasTarget(cmd) ? resolveTarget(cmd, p) : null; if (T && T.err) return {ok: false, reason: T.err, ask: T.ask}; tgt = T && T.prop || null;
          if (!tgt && tool && AFF(tool).use && (AFF(tool).use.targets || []).length) tgt = byId(AFF(tool).use.targets[0]) || null; }
        if (!tool) return {ok: false, reason: 'use what?', ask: true};
        if (!tgt) return {ok: false, reason: `use the ${tool.name} on what?`, ask: true};
        const mot = {rub: 'rub', wipe: 'rub', polish: 'rub', brush: 'brush', tap: 'tap', poke: 'tap', dab: 'tap', press: 'press', pour: 'pour', pour_on: 'pour', pour_out: 'pour', sprinkle: 'pour', spray: 'pour', empty: 'pour'}[vraw] || cmd.motion || null;
        return {ok: true, goals: [{g: 'use', id: tool.id, tool: tool.id, target: tgt.id, motion: mot}], did: null};
      }
      return null;
    }
    // the contact verbs (a paw ON a thing, no tool) and go-to. Returns null when the words mean something
    // else: a tool used on a thing ("tap it with the chalk", "rub the can on the plant") stays `use`; plain "knock" on a
    // small loose thing (no "on", no contact style) stays knock-over; "lean" without "on" + a big thing stays the held tilt.
    const TOUCH_V = {touch: 'touch', touch_lightly: 'touch', put_paw_on: 'touch', rest_paw_on: 'touch', lay_paw_on: 'touch', tap: 'tap', tap_on: 'tap', pat: 'pat', pat_on: 'pat',
      poke: 'poke', prod: 'poke', press: 'press', press_on: 'press', push_down_on: 'press', stroke: 'stroke', pet: 'stroke', rub: 'stroke', caress: 'stroke',
      knock_on: 'knock', rap: 'knock', rap_on: 'knock', lean_on: 'lean', lean_against: 'lean', rest_on: 'lean'};
    const GO_V = {go_to: 1, approach: 1, walk_to: 1, walk_up_to: 1, go_over_to: 1, walk_over_to: 1, go_up_to: 1, stand_by: 1, stand_at: 1, stand_next_to: 1, go_near: 1, move_to: 1, head_to: 1, go: 1, come_to: 1};
    function touchOrGo(cmd, vraw) {
      const desc = cmd.object ?? null, tgD = cmd.target ?? cmd.place ?? cmd.to ?? null;
      if (GO_V[vraw]) {
        const T = touchTarget(desc || tgD, cmd); if (T.err) return {ok: false, reason: T.err, ask: T.ask};
        return {ok: true, goals: [{g: 'go_to', T, id: T.prop}], did: null};
      }
      const rl = normRel(cmd.relation ?? cmd.rel ?? cmd.prep ?? cmd.preposition), st = String(cmd.style || '').toLowerCase().replace(/[\s-]+/g, '_');
      let style = TOUCH_V[vraw] || null;
      if (!style && (vraw === 'knock' || vraw === 'lean')) {
        const T0 = desc || tgD ? touchTarget(desc || tgD, cmd) : null, p0 = T0 && T0.prop ? byId(T0.prop) : null;
        const contact = rl === 'on' || TOUCH_STYLES[st] || (T0 && T0.place) || (p0 && (isFurn(p0) || p0.def.grip === 'two_hand'));
        if (!contact || (T0 && T0.err && !T0.ask && vraw === 'lean')) return null;
        style = vraw === 'knock' ? 'knock' : 'lean';
      }
      if (!style) return null;
      if (cmd.with || cmd.tool || cmd.using || cmd.instrument) return null;   // with a tool: the use path
      if (TOUCH_STYLES[st]) style = st;
      const d0 = desc || tgD;
      const ask = `${{knock: 'knock on', lean: 'lean on', press: 'press on'}[style] || style} what?`;
      if (!d0) return {ok: false, reason: ask, ask: true};
      if (desc && tgD && rl !== 'on') { const r0 = resolveObject(desc, 'use', {}); if (r0.p && AFF(r0.p).use) return null; }   // "tap the can on the plant": the tool path
      const T = touchTarget(d0, cmd); if (T.err) return {ok: false, reason: T.err, ask: T.ask};
      if (T.home) return {ok: false, reason: ask, ask: true};
      const count = clamp(Math.round(num(cmd.count, TOUCH_N[style])), 1, 8), sd = sideIn(cmd.side);
      return {ok: true, goals: [{g: 'touch', T, id: T.prop, style, count, side: sd && sd !== 'both' ? sd : null}], did: null};
    }
    function museObject(cmd, src, id) {
      const vraw = String(cmd.verb || cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_');
      let V = {kick: 'kick', punt: 'kick', knock_over: 'knock', knock_down: 'knock', knock: 'knock', tip_over: 'knock', topple: 'knock', push_over: 'knock', flick: 'knock', scoot: 'slide', nudge: 'slide', shift: 'slide', shove: 'push', put_on: 'put_on', wear: 'put_on', don: 'put_on', take_off: 'take_off', remove: 'take_off', pull: 'pull', drag: 'pull', push: 'push', bring: 'pull', move: 'pull',
        slide: 'slide', sit: 'sit', sit_in: 'sit', sit_on: 'sit', sit_at: 'sit_at', sit_down: 'sit', stand_up: 'stand', stand: 'stand', get_up: 'stand', open: 'open', close: 'close', shut: 'close',
        use: 'work', work_on: 'work', type_on: 'work', type: 'work', work: 'work', get_to_work: 'work', turn_back: 'turn_back', stop: 'stop_activity', stop_typing: 'stop_activity', stop_working: 'stop_activity',
        pick_up: 'pick_up', grab: 'pick_up', take: 'pick_up', get: 'pick_up', lift: 'pick_up', hold: 'pick_up', put_down: 'place', place: 'place', put: 'place', set_down: 'place', put_back: 'put_back', return: 'put_back',
        put_away: 'put_back', put_it_away: 'put_back', replace: 'put_back', hang_up: 'put_back', hang_back: 'put_back', hang: 'put_back', tidy: 'put_back', tidy_up: 'put_back', clean_up: 'put_back', give_back: 'put_back', return_it: 'put_back', set_back: 'put_back'}[vraw];
      const desc = cmd.object, tg = cmd.target;
      { const r = touchOrGo(cmd, vraw); if (r) return r; }   // contact / go-to
      // "put them all down / back" with several things in its paws (or, for put back, out of place): one job, one
      // thing after another (a single one, or the one thing the request is busy with, goes the usual way)
      if ((V === 'place' || V === 'put_back') && desc && typeof desc === 'object' && desc.ref === 'all' && !desc.name && !desc.id && !desc.color && !desc.category && pendingProps().length !== 1) {
        let set = E.P.filter(q => q.held || (V === 'put_back' && q.worn));
        if (V === 'put_back' && !set.length) set = E.P.filter(q => !isFurn(q) && q.where !== 'home');
        if (set.length > 1) {
          const goals = [];
          for (const q of set) { const r1 = museObject(Object.assign({}, cmd, {object: {id: q.id}}), src, id); if (r1 && r1.ok && r1.goals) goals.push(...r1.goals); }
          if (goals.length) return {ok: true, goals, did: null};
        }
      }
      // objects engine: "put everything back" = the dependency-aware tidy (stacks, lids, worn things, furniture)
      const allRef = d => !d || (typeof d === 'string' ? /^(all|everything|every ?thing|it all|them all|all of it|all of them)$/i.test(d) : (d.ref === 'all' || /^(all|everything)$/i.test(String(d.name || ''))) && !d.id && !d.color && !d.category && (!d.name || /^(all|everything)$/i.test(d.name)));
      if (V === 'put_back' && allRef(desc) && (desc || /everything|all/.test(vraw)) && (typeof desc === 'string' || /^(everything|all)$/i.test(String(desc && desc.name || '')) || E.P.filter(q => !isFurn(q) && (q.where !== 'home' || q.held || q.worn || q.bal)).length > 1 || E.P.some(q => isFurn(q) && q.where !== 'home'))) return {ok: true, goals: [{g: 'tidy'}], did: null};
      const res = desc ? resolveObject(desc, vraw, {preferHeld: /^(put_on|place|put_back)$/.test(V)}) : null;
      // (juggling takes two — an ambiguous 'the blocks' names both, xObject resolves the pair)
      // (the matcher reads "juggle the blocks" as a toss of them all: two or more tossed things at once = a juggle)
      if (res && res.cands && /^(toss|throw|juggle)$/.test(vraw) && (XVERB[vraw] === 'juggle' || (desc && desc.ref === 'all')) && !hasTarget(cmd)) return xObject('juggle', null, cmd, 'juggle');
      if (res && res.err && !/^(stand|turn_back|stop_activity|sit|work)$/.test(V) && !(XVERB[vraw] === 'juggle' && res.cands)) return {ok: false, reason: res.err, ask: res.ask};
      { const X = xVerb(V, vraw, cmd, res && res.p); if (X) return xObject(X, res && res.p, cmd, vraw); }
      let p = res && res.p;
      const pick = (pred, what) => { const c = E.P.filter(pred); return c.length ? c[0] : null; };
      const goals = [];
      // sliding / scooting a piece of FURNITURE is pushing or pulling it (toward what it names); the tabletop
      // slide stays for things on a table
      if (V === 'slide' && p && isFurn(p)) V = /away|back|further|farther/.test(String((tg && typeof tg === 'object' ? '' : tg) || cmd.dir || '')) ? 'push' : 'pull';
      switch (V) {
        case 'put_on': if (!desc && p && p.worn) p = null;   // "put it on" while the last thing is already worn = the next wearable
          p = p || E.P.find(q => feat(q).wearable && !q.worn && (q.held || pendingProps().includes(q))) || pick(q => feat(q).wearable && !q.worn) || pick(q => feat(q).wearable); if (!p) return {ok: false, reason: 'there is nothing here to wear'};
          // "put the ball on (your head)" for a small thing that is not worn = balance it there (said so)
          if (vraw === 'put_on' && !feat(p).wearable && !isFurn(p) && p.def.grip !== 'two_hand' && !hasTarget(cmd)) return xObject('balance', p, cmd, 'balance');
          goals.push({g: 'worn', id: p.id}); break;
        case 'take_off': p = p || pick(q => q.worn || q.bal); if (!p) return {ok: false, reason: "I'm not wearing anything"}; if (!p.worn && !p.bal && !pendingProps().includes(p)) return {ok: true, noop: true, pid: p.id, did: `not wearing the ${p.name} (nothing to take off)`}; { const sd = sideIn(cmd.side); goals.push(sd && sd !== 'both' ? {g: 'unworn', id: p.id, side: sd} : {g: 'unworn', id: p.id}); } break;
        case 'pull': case 'push': {
          if (!p) return {ok: false, reason: 'which thing should I move?', ask: true};
          if (!isFurn(p)) { // a small thing: carry it there
            const pl = tg && (typeof tg === 'object' ? (tg.place || tg.object) : tg); const r2 = pl ? resolvePlace(typeof pl === 'object' ? pl.name || pl.id : pl, p) : null;
            if (r2 && !r2.err) { goals.push({g: 'on', id: p.id, place: r2.id}); break; }
            if (V !== 'push' && /^(me|him|you|user|here|to me|over here)$/.test(String((tg && typeof tg === 'object' ? tg.name || tg.ref : tg) || '').toLowerCase().replace(/_/g, ' '))) {   // "bring it to me" = carry it and hold it out
              goals.push({g: 'holding', id: p.id}, {g: 'generic', cmd: {action: 'object', verb: 'offer', object: {id: p.id}}}); break;
            }
            if (V !== 'push') { goals.push({g: 'bring', id: p.id}); break; }
            if (p.place && p.place !== 'floor') { goals.push({g: 'slide', id: p.id, dir: 'away'}); break; }
            goals.push({g: 'holding', id: p.id}); break;
          }
          let target = tg || (V === 'push' ? 'away' : 'me');
          if (typeof target === 'object' && (target.name || target.id)) { const r3 = resolveObject(target, 'to'); if (r3.p) target = {object: r3.p.id}; else { const r4 = resolvePlace(target.name || target.id); if (r4 && !r4.err && r4.prop) target = {object: r4.prop}; else return {ok: false, reason: r3.err || 'where to?'}; } }
          goals.push({g: 'furn', id: p.id, target}); break;
        }
        case 'slide': if (!p) p = pick(q => feat(q).laptop); if (!p) return {ok: false, reason: 'slide what?', ask: true}; goals.push({g: 'slide', id: p.id, dir: String(tg || cmd.dir || 'closer')}); break;
        case 'sit': case 'sit_at': {
          const chair = p && feat(p).chair ? p : pick(q => feat(q).chair);
          let desk = null;
          if (V === 'sit_at' || (p && !feat(p).chair)) { const ref = p && !feat(p).chair ? p : null; desk = ref ? (feat(ref).desk ? ref : (() => { const pl = E.places.find(x => x.id === ref.place); return pl && byId(pl.prop || pl.on || ''); })()) : pick(q => feat(q).desk); if (ref && feat(ref).laptop && !desk) desk = pick(q => feat(q).desk); }
          if (tg && typeof tg === 'object') { const r5 = resolveObject(tg, 'at'); if (r5.p && feat(r5.p).desk) desk = r5.p; }
          if (!chair && (V === 'sit_at' || p)) return {ok: false, reason: 'there is no chair here'};
          goals.push({g: 'sit', id: chair ? chair.id : null, desk: desk ? desk.id : null}); break;
        }
        case 'stand': goals.push({g: 'stand'}); break;
        case 'knock': case 'kick': {   // objects: a loose thing pushed over / kicked (physics does the rest)
          if (!p) return {ok: false, reason: V === 'kick' ? 'kick what?' : 'knock over what?', ask: true};
          const top = V === 'knock' ? (E.P.filter(q => q.sup && q.sup.id === p.id)[0] || null) : null;   // "knock the tower over" names its base: fine
          void top; goals.push({g: 'knock', id: p.id, foot: V === 'kick'}); break;
        }
        case 'open': case 'close': p = p || pick(q => feat(q).laptop) || pick(q => hasArt(q)); if (!p) return {ok: false, reason: 'there is nothing here to open'}; if (!feat(p).laptop && !hasArt(p)) return {ok: false, reason: isFurn(p) ? `the ${p.name} doesn't open` : `the ${p.name} doesn't open — it has no lid`}; goals.push({g: 'lid', id: p.id, open: V === 'open'}); break;
        case 'work': p = p || pick(q => feat(q).laptop); if (!p) return {ok: false, reason: 'there is no laptop here'}; goals.push({g: 'type', id: p.id, sit: cmd.sit}); break;
        case 'turn_back': goals.push({g: 'turn_back'}); break;
        case 'stop_activity': goals.push({g: 'stop_activity'}); break;
        case 'pick_up': { if (!p) return {ok: false, reason: 'which one?', ask: true}; const sd = sideIn(cmd.side); goals.push(Object.assign({g: 'holding', id: p.id}, sd && sd !== 'both' ? {side: sd} : {}, cmd.stay || cmd.stand ? {} : {fetch: true})); break; }
        case 'place': {
          if (!p) p = pendingProp();   // "pick up the ball and set down": the thing the step before it is getting
          if (!p) { const h = E.P.filter(q => q.held); if (h.length !== 1) return {ok: false, reason: h.length ? 'which one?' : 'I am not holding anything', ask: !!h.length}; p = h[0]; }
          if (backWhere(cmd)) return museBack(p, backWhere(cmd), cmd);   // "put it down where it was" = put it back
          let pl = cmd.place || (tg && (typeof tg === 'object' ? tg.place || tg.name : tg));
          if (pl === 'near' || (pl && typeof pl === 'object' && pl.ref === 'near')) pl = null;
          let r6 = pl ? resolvePlace(typeof pl === 'object' ? pl.name || pl.id : pl, p) : null;
          // a place that isn't here ("hang them on the hook") for a thing with a spot of its own: its own spot
          if (r6 && r6.err && p.homePlace && p.home && !isFurn(p) && !(p.where === 'home' && !p.held && !p.worn)) { const rb = museBack(p, 'home', {}); if (rb && rb.ok) return Object.assign(rb, {note: `${r6.err}, so its own spot`}); }
          if (r6 && r6.err) return {ok: false, reason: r6.err};
          if (r6) { goals.push({g: 'on', id: p.id, place: r6.id}); break; }
          if (p.place && p.place !== 'floor' && !p.held && !p.worn) { goals.push({g: 'on', id: p.id, place: p.place}); break; }
          goals.push({g: 'on', id: p.id, any: true}); break;   // "put it down": the nearest surface it can reach
        }
        case 'put_back': {
          if (!p) p = pendingProp();
          if (!p) { const h = E.P.filter(q => q.held || q.worn); if (h.length !== 1) return {ok: false, reason: h.length ? 'which one?' : 'put what back?', ask: true}; p = h[0]; }
          return museBack(p, backWhere(cmd) || 'home', cmd);
        }
        default: return null;   // the generic verbs (show, inspect, toss …) → the seated-character chains
      }
      return {ok: true, goals, did: null};
    }
    // put back / return / put away: its home spot (the headphones: on their stand), or where it last rested ("where it
    // was"); a place the request names wins ("put the laptop back on the desk"); furniture is pushed back home
    function museBack(p, how, cmd) {
      if (isFurn(p)) return {ok: true, goals: [{g: 'furn', id: p.id, target: 'home'}], did: null};
      const tg = cmd.place || cmd.target, nm = tg && (typeof tg === 'object' ? tg.place || tg.name || tg.id : tg);
      let note = null;
      if (nm && !backWhere({place: nm})) { const r = resolvePlace(nm, p); if (r && !r.err) {
        if (r.id === p.homePlace) how = 'home';
        else return {ok: true, goals: [{g: 'on', id: p.id, place: r.id}], did: null};
      } else if (r && r.err) { note = `${r.err}, so its own spot`; how = 'home'; } }   // said truthfully
      const L = p.lastRest, busy = pendingProps().includes(p);   // a queued step will move it — judge when it runs
      if (how === 'last' && L && !L.home) {
        if (!busy && !p.held && !p.worn && d2(p.pos, L.pos) < 0.02) return {ok: true, noop: true, pid: p.id, did: `the ${p.name} is already where it was`};
        return {ok: true, goals: [{g: 'on', id: p.id, place: L.place, at: L.pos.slice(), rot: L.rot.slice(), back: 'last'}], did: null};
      }
      if (!busy && p.where === 'home' && !p.held && !p.worn && !p.bal) return {ok: true, noop: true, pid: p.id, did: `the ${p.name} is already where it belongs`};
      if (p.homeOn || p.worn || p.bal || p.anc) return {ok: true, goals: [{g: 'home1', id: p.id}], did: null, note};   // its home is ON another thing / it is on the body
      return {ok: true, goals: [{g: 'on', id: p.id, place: p.homePlace || 'floor', at: p.home.slice(), rot: [0, 0, 0], back: 'home'}], did: null, note};
    }
    function enqueueJob(kind, name, goals, args, src, id) {
      // preview the plan now so the tool result is truthful (re-planned from the live state when it starts)
      const prev = E.jobs.length || E.move || E.chain ? null : planGoals(goals);
      if (prev && !prev.ok) { emit({kind, name, id, src, phase: 'fail', args, reason: prev.reason}); return {ok: false, id, reason: prev.reason}; }
      const J = {kind, name, goals, args, src, id, ops: null};
      if (prev && !prev.ops.length) { emit({kind, name, id, src, phase: 'end', args, did: 'nothing to do — it already is that way'}); return {ok: true, id, did: 'nothing to do — it already is that way', already: true}; }
      E.jobs.push(J);
      if (kind === 'object') { const q = jobProp(J); if (q) E.lastRef = q; }
      const did = prev ? describe(prev.ops, 0) : `${name.replace(/_/g, ' ')} (after what it is doing now)`; J.did0 = did;
      return {ok: true, id, did, queued: !prev};
    }
    function museFrame(dt) { activityFrame(dt); jobFrame(); }
    // ---- streaming refinements (the server matcher fires on a stable partial, then sends the final as a refinement of
    // the same utterance): a command carrying `replaces: <id>` (or `refine: true` = the last job) REPLACES that job
    // while it is young instead of queueing a second one. Same move, just started (< 0.35 s): restarted with the new
    // parameters from the current pose; same move further along: kept (the refinement is merged, nothing doubles);
    // anything else within REFINE_S: the old job stops gracefully (its paws/feet finish what is atomic) and the new one
    // starts from wherever it got to. Too late, or the old one already finished: an identical request is dropped as
    // already done, a different one runs as a new command.
    const REFINE_S = 1.2;
    const cmdKey = c => JSON.stringify([c.action, String(c.name || c.verb || '').toLowerCase(), c.object && (c.object.id || c.object.name || c.object.ref) || '', c.place || '', c.target && (c.target.object || c.target.place || c.target) || '']);
    const cmdName = c => String(c.name || c.verb || '').toLowerCase().replace(/[\s-]+/g, '_');
    function refineJob(cmd, src, id, rid) {
      const J = E.jobs.find(j => j.id === rid);
      const clean = Object.assign({}, cmd); delete clean.replaces; delete clean.refine;
      if (!J) {   // already over (or failed): an identical request is not done twice
        const L = E.lastJob && E.lastJob.id === rid ? E.lastJob : null;
        if (L && L.key === cmdKey(clean) && t() - L.t < 6) return {ok: true, id: rid, did: 'already did that (same request)', already: true, refined: rid};
        return null;
      }
      const iJ = E.jobs.indexOf(J), age = J.started ? t() - (J.t0 || t()) : 0, same = J.args && cmdName(J.args) === cmdName(clean) && J.args.action === clean.action;
      if (J.started && age > REFINE_S) return same ? {ok: true, id: rid, did: 'already doing that', already: true, refined: rid} : null;
      if (J.started && same && age > 0.35 && clean.action === 'move') {
        // the refinement asks for MORE of the same move ("spin" → "spin twice"): the difference follows right after
        // it (unless the refinement also changes how — direction, speed, height, style); anything else is merged as is
        const n0 = Math.max(1, Math.round(+J.args.count || 1)), n1 = Math.max(1, Math.round(+clean.count || 1));
        const ARGS = ['dir', 'speed', 'height', 'style', 'distance'], argv = (o, k) => (o[k] === undefined || o[k] === null ? null : String(o[k]).toLowerCase());
        const changed = ARGS.some(k => argv(clean, k) !== null && argv(clean, k) !== argv(J.args, k));
        if (!changed && clean.count !== undefined && n1 > n0) {
          const r2 = museCommand(Object.assign({}, J.args, clean, {count: n1 - n0}), src, id);
          if (r2 && r2.ok) {
            const k = E.jobs.findIndex(j => j.id === id), at = E.jobs.indexOf(J) + 1;
            if (k > at && at > 0) E.jobs.splice(at, 0, E.jobs.splice(k, 1)[0]);
            E.lastJob = {id, t: t(), key: cmdKey(clean)}; E.burst = {src, id, t: t()};
            return {ok: true, id, did: `${J.did0 || J.name}, then ${n1 - n0} more (${n1} in all; refinement merged)`, refined: rid, replaced: undefined};
          }
        }
        return {ok: true, id: rid, did: (J.did0 || 'already doing that') + ' (refinement merged)', already: true, refined: rid};
      }
      if (J.started && same && clean.action === 'move' && E.move && E.move.opOf === J && !E.chain && E.jobs[0] === J) {
        // just started, same move: retarget — drop the young plan; the new one starts from the current pose and velocity
        E.jobs.shift(); E.move = null;
        emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'interrupted', args: J.args, did: `restarted with the refined request (#${id})`, reason: `replaced by the refined request (#${id})`});
      } else cancelJob(J, `replaced by the refined request (#${id})`);
      const i0 = E.jobs.indexOf(J) >= 0 ? E.jobs.indexOf(J) + 1 : Math.max(0, iJ);
      const r = museCommand(clean, src, id);
      if (r && r.ok) { E.lastJob = {id, t: t(), key: cmdKey(clean)}; r.replaced = rid; }
      // the replacement takes the replaced job's turn in the queue (not the back of it, behind the steps he
      // asked for after it): "close the laptop, put it away, stand" stays in that order when the model refines a step
      const k = E.jobs.findIndex(j => j.id === id);
      if (k > i0) E.jobs.splice(i0, 0, E.jobs.splice(k, 1)[0]);
      if (k >= 0) E.burst = {src, id, t: t()};
      return r;
    }
    function cancelJob(J, why) {
      const i = E.jobs.indexOf(J); if (i < 0) return;
      if (!J.started) { E.jobs.splice(i, 1); emit({kind: J.kind, name: J.name, id: J.id, src: J.src, phase: 'interrupted', args: J.args, did: 'not started (' + why + ')', reason: why}); return; }
      J.stopped = true; J.replacedWhy = why; J.ops = J.ops ? J.ops.filter(o => o.op === '_letgo') : [];
      if (E.chain && E.chain.opOf === J) stopChain(why);
      if (E.move && E.move.opOf === J) stopMove(why);
      if (E.activity && J.ops.length === 0 && /type/.test(J.name)) stopActivity(null);
    }
    const SEATED_OK = /^(turn_back|wiggle|sway|shimmy|bow|dance|stop|sit|get_up|stand|crouch)$/;
    function museCommand(cmd, src, id) {
      if (cmd.action === 'object') {
        // "hand me X" / "pass it to me" — a hand-over / give / pass whose target is HIM is an offer (eval o14:
        // the voice sent hand_over target "me", the engine passed the block to its other paw)
        { const vb0 = String(cmd.verb || '').toLowerCase().replace(/[\s-]+/g, '_'), tg0 = String(cmd.target || cmd.to || cmd.place || (cmd.object && cmd.object.to) || '').toLowerCase().trim();
          if (/^(hand_over|hand|pass|give)$/.test(vb0) && /^(me|him|the user|user|you|the camera|camera|viewer|us|her|them)$/.test(tg0)) { cmd = Object.assign({}, cmd, {verb: 'offer'}); delete cmd.target; delete cmd.to; if (/^(me|him|the user|user|you|the camera|camera|viewer|us|her|them)$/.test(String(cmd.place || '').toLowerCase().trim())) delete cmd.place; } }
        const r = museObject(cmd, src, id), vraw = String(cmd.verb || cmd.name || '');
        if (r && !r.ok) { emit({kind: 'object', name: vraw, id, src, phase: 'fail', args: cmd, reason: r.reason}); return {ok: false, id, reason: r.reason, ask: r.ask}; }
        if (r && r.noop) { if (r.pid && byId(r.pid)) E.lastRef = byId(r.pid); emit({kind: 'object', name: vraw, id, src, phase: 'end', args: cmd, did: r.did}); return {ok: true, id, did: r.did, already: true}; }
        if (r) { const rr = enqueueJob('object', vraw, r.goals, cmd, src, id); if (r.note && rr && rr.ok) rr.did = `${r.note}: ${rr.did}`; return rr; }
        // show / inspect / toss / shake / hug / offer / turn / flip / hand over / drop: hold it first, then the paw chain
        const verb = VERB_ALIAS[vraw.toLowerCase().replace(/[\s-]+/g, '_')];
        if (!verb) { const reason = `I don't know how to "${vraw}" something`; emit({kind: 'object', name: vraw, id, src, phase: 'fail', args: cmd, reason}); return {ok: false, id, reason}; }
        const res = cmd.object ? resolveObject(cmd.object, verb, {preferHeld: true}) : null;
        let p = res && res.p;
        if (res && res.err) { emit({kind: 'object', name: vraw, id, src, phase: 'fail', args: cmd, reason: res.err}); return {ok: false, id, reason: res.err, ask: res.ask}; }
        if (!p) p = pendingProp();
        if (!p) { const h = E.P.filter(q => q.held); if (h.length === 1) p = h[0]; else { const reason = h.length ? 'which one?' : 'I am not holding anything'; emit({kind: 'object', name: vraw, id, src, phase: 'fail', args: cmd, reason}); return {ok: false, id, reason, ask: true}; } }
        if (isFurn(p)) { const reason = `the ${p.name} is furniture — I can push or pull it`; emit({kind: 'object', name: vraw, id, src, phase: 'fail', args: cmd, reason}); return {ok: false, id, reason}; }
        const sdg = sideIn(cmd.side);
        // a thing shown / offered to him is presented FACING him (r6: shown at the tray turned 36° away)
        const faceG = /^(show|offer|shake|wave|raise)$/.test(verb) && E.posture !== 'seated' && !E.sitOn ? [{g: 'present'}, {g: 'face'}] : [];
        return enqueueJob('object', vraw, [sdg && sdg !== 'both' ? {g: 'holding', id: p.id, side: sdg} : {g: 'holding', id: p.id}].concat(faceG, [{g: 'generic', cmd: Object.assign({}, cmd, {object: {id: p.id}})}]), cmd, src, id);
      }
      const raw = String(cmd.name || '').toLowerCase().replace(/[\s-]+/g, '_');
      let name = MOVE_ALIAS[raw], extra = {};
      if (Array.isArray(name)) { extra = name[1]; name = name[0]; }
      if (!name) { const reason = `I don't know the move "${raw.replace(/_/g, ' ') || '?'}"`; emit({kind: 'move', name: raw, id, src, phase: 'fail', args: cmd, reason}); return {ok: false, id, reason}; }
      if (name === 'stop') { const had = !!(E.move || E.chain || E.jobs.length || E.activity); stopAll('stopped'); return {ok: true, id, did: had ? 'stopping what it was doing' : 'nothing to stop (it was not moving)'}; }
      const a = Object.assign({}, extra, cmd);
      if (name === 'flip') name = /forward|front/.test(normDir(a.dir)) ? 'frontflip' : 'backflip';
      name = rotateAmount(name, a);
      if (name === 'lie_down' && !a.dir && !a.dead) a.dir = rnd() < 0.5 ? 'left' : 'right';   // one side for the preview, the plan and the result
      if (name === 'turn' && /^(around|back|away)$/.test(normDir(a.dir))) name = 'turn_away';
      if (name === 'sit' || (name === 'get_up' && E.sitOn) || name === 'stand') {
        const goals = name === 'sit' ? [{g: 'sit', id: (a.object ? (resolveObject(a.object, 'sit').p || {}).id : null) || (E.P.find(q => feat(q).chair) || {}).id || null}] : [{g: 'stand'}];
        if (name === 'sit' && a.floor) goals[0].id = null;
        return enqueueJob('move', name, goals, a, src, id);
      }
      if (name === 'turn_back' && E.sitOn) return enqueueJob('move', name, [{g: 'turn_back'}], a, src, id);
      return enqueueJob('move', name, [{g: 'move', name, args: a}], a, src, id);
    }
    function post(o) {   // after the arm follow-through: where each paw actually is (paths start from here)
      for (const s of ['l', 'r']) { const pw = E.paw[s]; pw.hPrev = [o['hand_' + s + '_x'], o['hand_' + s + '_y'], o['hand_' + s + '_z']]; if (!(pw.ctrl > 0)) pw.h = pw.hPrev.slice(); }
      // objects engine. While the keyframe layer draws the body, postKf tracks the paws / things once, from the hands
      // really drawn: tracking here too would (a) leave the engine's own carry pose in pawGW / wpos while kfFrame fires its
      // events (a release then started the flight from where the paw was NOT drawn) and (b) measure the velocities from two
      // different poses in one frame.
      if (E.kfh && E.P.length && !E.kfLive) { trackPaws(o, rootNow(), E.bend ? E.bend.p : 0); trackFrame(o); }
    }
    // a source's call that was dropped as the same request as job `id` (page IntentDedupe): its following new
    // steps queue right after that job, like after a refinement (the voice model restates the request in order)
    // scene: setView('board' | 'tv' | <any place with a "camera"> | null): ease the camera to that station's
    // framing (renderer/animals/muse.props.json places[].camera {cam_zoom, cam_x, cam_y}; cam_zoom < 0 pushes in) and,
    // with null, back to the normal follow / home framing.  Returns {ok, view, reason}.
    const VIEW_BOTH_K = 0.75;   // zoom-out per unit of cam_x span between the two stations
    const SHOWY = /^(show|offer|inspect|hold_up|shake|wave|wave it)$/, SHOW_ZOOM = -0.6, SHOW_ZOOM_SMALL = -0.85, SHOW_ZOOM_TINY = -1;   // small things closer   // showing a thing pushes in
    const AFF_ZOOM = 0.24, AFF_CAM_Y = 0.0;   // affect: the feeling push-in (home framing 0.3 -> 0.06 at most)
    function setView(name) {
      if (name == null || name === '' || name === 'home') { E.view = null; return {ok: true, view: null}; }
      // 'a+b' = a framing that holds both stations (the cameras' mean, pushed in less)
      const camOf = nm => { const r = E.props && Array.isArray(E.props.places) ? E.props.places.find(p => p && p.id === nm && p.camera) : null; return r ? r.camera : null; };
      let c = camOf(name);
      if (!c && String(name).includes('+')) {
        const cs = String(name).split('+').map(camOf).filter(Boolean);
        if (cs.length >= 2) { const m = f => cs.reduce((a, q) => a + (+q[f] || 0), 0) / cs.length, span = Math.max(...cs.map(q => +q.cam_x || 0)) - Math.min(...cs.map(q => +q.cam_x || 0));
          c = {cam_zoom: Math.min(0, m('cam_zoom') + VIEW_BOTH_K * span), cam_x: m('cam_x'), cam_y: m('cam_y')}; }
      }
      if (!c) return {ok: false, view: E.view, reason: `no station camera for '${name}'`};
      E.viewCam = {zoom: clamp(+c.cam_zoom || 0, -1, 1), x: +c.cam_x || 0, y: +c.cam_y || 0};
      E.view = String(name);
      return {ok: true, view: E.view};
    }
    function anchor(id, src) { if (!E.jobs || !E.jobs.some(j => j.id === id && !j.stopped)) return false; E.burst = {src, id, t: t()}; return true; }
    // teach mode: the narrow kit app/teach.js drives (stations, chalk writing / erasing, presenting) through the job queue
    let teachN = 0;
    const kit = {E, t, emit, pawPath, stepFn, lookStep, par, retractStep, dwellStep, graspStep, bendStep, gripPick, invW, fwdW, toWorld, toLocal, toLocalR, toWorldR, rootNow, bendSolve,
      standClear, byId, heldBy, feat, fW, gripLocal, rotYv, yawOfDir, wrap, mj, stopAll, state: () => stateOut(), stopChain,
      enqueue(name, goals, src, args) { const id = 'tch' + (++teachN); return enqueueJob('object', name, goals, Object.assign({action: 'teach', name}, args || {}), src || 'tool', id); },
      cancel(id, why) { const J = E.jobs.find(j => j.id === id); if (J) cancelJob(J, why || 'stopped'); return !!J; },
      jobs: () => E.jobs.map(j => ({id: j.id, name: j.name, started: !!j.started, stopped: !!j.stopped})),
      busy: () => !!(E.move || E.chain || E.jobs.length)};
    function affectCam(x) { E.affCamT = clamp(+x || 0, 0, 1); }   // affect
    return {setConfig, command, apply, base, post, release, freeze, stopAll, state: stateOut, busy, wholeBody, anchor, kit, setView, affectCam,
      get view() { return E.view; },
      kfHeld, kfEvent, things, postKf,   // objects engine (docs/OBJECTS.md §5)
      _obj: {supportAt, P: p => P(p), boxHit, restPose, worldOf: p => worldOf(p), pawSurfGap, touchSpot, touchTarget},   // (contact queries)   // (tests: physics queries)
      get active() { return E.active || !!E.move || !!E.chain; }, heldBy: s => heldBy(s), lp: () => E.lp, E,
      props: () => E.P.map(p => ({id: p.id, slot: p.k, name: p.name, where: propWhere(p), held: p.held, pos: p.pos.map(v => +v.toFixed(4)), rot: p.rot.slice(), att: Object.assign({}, p.att), home: p.home.slice()})),
      hasProps: () => E.P.length > 0, moveNames: MOVE_NAMES, verbs: OBJECT_VERBS, reach: s => E.reach[s], body: () => E.body};
  }

  return {createEngine, ReachMap, bodyModel, Track, quintic, cruise, rotm, wrapH, MOVE_NAMES, MOVE_ALIAS, DSTYLE, DSTEP, mj};
});
