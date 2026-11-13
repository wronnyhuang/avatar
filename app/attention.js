/* attention.js — ATTENTION (page side): a CHEAP candidate detector that decides when the server's vision judge
 * should look at the camera. The realtime voice gets camera frames ONLY through this path (never a steady stream).
 *
 * window.AvatarAttention = {create, KINDS, DEFAULTS, _test}   (also module.exports when `module` exists)
 *
 * const at = AvatarAttention.create({send(msg), onEvent(ev), video?, grab?, now?, rate?, ...opts})
 *   at.frame(f, ctx)     every perception feature frame (~18 Hz). f = app/perception.js feature frame;
 *                        ctx = {t (s, f.t's clock), userSpeaking, userLevel (PEAK linear mic RMS since the previous call,
 *                        else the latest; speech ≈ > 0.012), avatarSpeaking, thumb?: {w, h, data: RGBA Uint8ClampedArray,
 *                        t?}}. No ctx.thumb + opts.video → the page draws its own thumbnail (long side 48 px) every
 *                        0.15 s (that draw is timed separately: stats().thumb_draw_ms). Returns the candidate message
 *                        sent on this frame, or null.
 *   at.onHub(msg)        server → page: {type:'attention', id, what, worth_reply, salience, reason, t, ms, model, usd}
 *                        → onEvent({type:'attention', id, what, worth_reply, salience, reason, t, kinds, age_s, stale,
 *                        image (the "now" JPEG, only when worth_reply)}); {type:'attention_rules', rules} → rules().
 *   at.instruct(text)    → send({type:'attention_rule', op:'add', text})           a standing instruction
 *   at.feedback(text, {id}?) → send({type:'attention_rule', op:'feedback', text, id}) (id defaults to the last verdict)
 *   at.forget(idOrAll)   → send({type:'attention_rule', op:'remove', id}) | op:'clear' for 'all'
 *   at.rules()           the last rules list the server sent
 *   at.stats()           {frames, ms_per_frame, ms_p99, thumbs, ms_per_thumb, candidates, sent, by_kind, suppressed,
 *                         verdicts, worth_reply, inflight, thumb_draw_ms, last:[recent candidates]}
 *   at.reset()           forget the scene model / baselines (e.g. after a camera switch)
 *
 * CANDIDATE KINDS (general mechanisms; each trigger = {kind, strength 0..1, detail}):
 *   scene_change     the camera thumbnail vs a slow-adapting BACKGROUND MODEL, outside THE USER'S region (face box, body below
 *                    the face, hands, pose arms; a pixel stays excluded 1.2 s after it was last covered — head turns,
 *                    detection dropouts). A compact blob of change that persists ≥ ~0.3 s (someone/something entering,
 *                    a door), or a whole-frame change not explained by a small camera shift (lights, the lens covered,
 *                    the camera turned). A global shift that IS a camera translation re-learns the background instead.
 *   new_object       a blob of change outside the user's region that stays put ≥ 1.2 s (something placed / left in view),
 *                    or perception's object_up (the user holds something up near their face).
 *   new_face         face_appear after an absence (≥ 2.5 s, or a first face once the detector is past its warm-up),
 *                    stronger when the new face's size / position differs from the last one (a different person).
 *   expression_spike perception's expression_change; a fast jump in an xp score (≥ 0.55 within 0.8 s); a sustained
 *                    wide-open jaw while the user is silent (yawn, gasp). Weaker while the user is talking.
 *   motion_burst     f.motion well above the user's running baseline (median-like EMA + deviation), or approach_fast.
 *   gaze_to_camera   look_back, or the first look at the camera after ≥ 1.5 s of a face that never attended.
 *                    Weak while the user is talking (looking away and back is ordinary speech behaviour).
 *   loud_sound       an impulsive mic peak (≥ max(0.25, 2× the user's speech peaks)) after ≥ 0.8 s of silence, short (falls
 *                    below 30 % within 0.4 s) — a bang, a clap, a dropped object. Confirmed ≤ 0.45 s after onset.
 * RATE LIMITS: per-kind cooldown (habituating: base × (1 + same kind sent in the last 30 s)), a global minimum gap
 *   (3 s) and a sliding cap (10 / min), at most one request in flight (8 s timeout). A trigger blocked only by the
 *   global gap waits ≤ 1 s and is merged with whatever else triggers meanwhile. Weak triggers (< minStrength, or <
 *   speakingMin while the user talks, except scene_change / new_face / loud_sound) are dropped.
 * ON A CANDIDATE: 1–2 small JPEGs (~1 s before from a ring of canvases + now, 320 px long side, quality 0.6, re-encoded
 *   at 0.45 when > 40 KB) → send({type:'attention_candidate', id, t, kinds:[...], strength, details:{kind:detail},
 *   images:[dataURL...], summary:{speaking, avatar_speaking, expression, attending, face, hands, motion, level,
 *   recent:[event types in the last 3 s]}}).
 * Cost (node, tests/attention_test.cjs): see stats(); the thumbnail analysis runs only when a new thumbnail
 *   arrives (~7 Hz), the per-frame path is a handful of comparisons.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarAttention = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';

  const KINDS = ['scene_change', 'new_object', 'new_face', 'expression_spike', 'motion_burst', 'distance_change', 'gaze_to_camera', 'loud_sound', 'gesture'];
  const DEFAULTS = {
    rate: {minGap: 3.0, perMin: 10, habituate: 30, pendingMax: 1.0, inflightMax: 8, maxAge: 5, minStrength: 0.35, speakingMin: 0.65,
      cooldown: {scene_change: 5, new_object: 8, new_face: 6, expression_spike: 6, motion_burst: 5, distance_change: 5, gaze_to_camera: 8, loud_sound: 4, gesture: 5},
      // in CONVERSATION (the user spoke ≤ 1.5 s ago or the avatar ≤ 1.0 s ago) a trigger's strength is scaled by this factor and must
      // reach speakingMin: smiles, gestures, glances and lean-ins are ordinary conversation behaviour, not news
      conv: {scene_change: 1, new_object: 0.95, new_face: 0.7, expression_spike: 0.9, jump: 0.7, happy: 0.55, raised_brows: 0.55, motion_burst: 0.6, distance_change: 0.8,
        gaze_to_camera: 0.5, loud_sound: 1, gesture: 0.8},
      urgent: 0.7, hardGap: 1.0},       // urgent = loud_sound, a different face, a whole-frame scene_change ≥ urgent: skips minGap + in-flight
    gestures: false,                    // perception's wave / point / thumbs_up as kind 'gesture' (off: the page's SPEAK_EVENTS answer waves)
    warmup: 1.0,                        // s of scene model learning before scene candidates
    thumb: {long: 48, every: 0.15},     // page-side thumbnail (when ctx.thumb is absent)
    ring: {every: 0.25, size: 6, long: 320, quality: 0.6, maxBytes: 40000, before: 1.0},
    scene: {tauBg: 5, tauFg: 30, tauFast: 0.3, thMin: 14, thK: 3, maskHold: 1.2, minBlob: 0.03, minBlobPx: 6, persist: 2,
      global: 0.45, objectAge: 1.2, objectMax: 0.25, shift: 2},
    motion: {tau: 15, minRise: 0.10, k: 4, abs: 0.15, hold: 0.2, sizeTau: 6, closer: 1.45, farther: 0.7, sizeHold: 0.3},
    photo: {tau: 3, dMean: 30, sigLo: 0.4, sigHi: 2.2},   // whole-frame brightness / contrast jumps (lights, lens covered)
    expr: {jumpWin: 0.8, jump: 0.55, jumpTo: 0.7, jaw: 0.33, jawHold: 0.4},
    sound: {abs: 0.18, rel: 2.0, quiet: 0.8, decay: 0.3, confirm: 0.4},
  };
  const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
  const clamp01 = v => clamp(v, 0, 1);
  const r3 = v => Math.round(v * 1000) / 1000;
  const nowMs = () => (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  function merge(a, b) {
    const o = Object.assign({}, a);
    for (const k in (b || {})) o[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') ? merge(a[k], b[k]) : b[k];
    return o;
  }

  // ───────────────────────────── scene model on a tiny gray thumbnail (raw, unmirrored camera pixels) ─────────────────────────────
  function createScene(P) {
    const S = {W: 0, H: 0, N: 0, B: null, V: null, age: null, maskT: null, mask: null, t0: null, lastT: null, globalN: 0,
      fastUntil: -1, tdt: null, lastPerson: null, lastPersonT: -99, stack: null, comp: null};
    function init(W, H, g, t) {
      S.W = W; S.H = H; S.N = W * H; S.B = Float32Array.from(g); S.V = new Float32Array(S.N).fill(16);
      S.age = new Uint8Array(S.N); S.maskT = new Float32Array(S.N).fill(-99); S.mask = new Uint8Array(S.N);
      S.stack = new Int32Array(S.N); S.comp = new Int32Array(S.N); S.valid = new Uint8Array(S.N); S.fg = new Uint8Array(S.N); S.t0 = t; S.lastT = t; S.globalN = 0;
    }
    // the user's region → S.mask (1 = person) and S.maskT (last time covered), from the feature frame (viewer coords → raw px)
    function person(f, t) {
      const W = S.W, H = S.H, m = S.mask; m.fill(0);
      const PX = x => (0.5 - x / 2) * W, PY = y => (0.5 - y / 2) * H;
      let any = false;
      const box = (x0, x1, y0, y1) => {
        const a = Math.max(0, Math.floor(Math.min(x0, x1))), b = Math.min(W - 1, Math.ceil(Math.max(x0, x1)));
        const c = Math.max(0, Math.floor(Math.min(y0, y1))), d = Math.min(H - 1, Math.ceil(Math.max(y0, y1)));
        for (let y = c; y <= d; y++) for (let x = a; x <= b; x++) m[y * W + x] = 1;
        any = true;
      };
      const disc = (cx, cy, r) => box(cx - r, cx + r, cy - r, cy + r);
      const seg = (p, q, r) => { const n = Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / Math.max(1, r))); for (let i = 0; i <= n; i++) disc(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n, r); };
      const face = f && f.face && f.face.present ? f.face : null;
      let fw = 0.08 * W;
      if (face) {
        fw = Math.max(2, (face.size || 0.15) * W);
        const cx = PX(face.x), top = face.topY != null ? PY(face.topY) : PY(face.y) - 0.6 * fw, chin = face.chinY != null ? PY(face.chinY) : PY(face.y) + 0.6 * fw;
        const fh = Math.max(2, chin - top);
        box(cx - 0.9 * fw, cx + 0.9 * fw, top - 0.5 * fh, chin + 0.3 * fh);         // head + hair
        box(cx - 1.5 * fw, cx + 1.5 * fw, chin, H);                                   // body below the face (shoulders ≈ 3 face widths)
      }
      const pose = f && f.pose && f.pose.present ? f.pose : null;
      if (pose && pose.shoulders) {
        const sh = pose.shoulders, el = pose.elbows || {}, wr = pose.wrists || {};
        const sw = sh.l && sh.r ? Math.abs(PX(sh.l.x) - PX(sh.r.x)) : fw * 2;
        const r = Math.max(1.5, 0.22 * sw);
        if (sh.l && sh.r) {
          box(Math.min(PX(sh.l.x), PX(sh.r.x)) - r, Math.max(PX(sh.l.x), PX(sh.r.x)) + r, Math.min(PY(sh.l.y), PY(sh.r.y)) - r, H);
          if (!face) { const hx = (PX(sh.l.x) + PX(sh.r.x)) / 2, hy = Math.min(PY(sh.l.y), PY(sh.r.y)); box(hx - 0.45 * sw, hx + 0.45 * sw, hy - 0.9 * sw, hy); }
        }
        for (const s of ['l', 'r']) {
          const a = sh[s], b = el[s], c = wr[s];
          if (a && b && (b.v == null || b.v > 0.3)) seg([PX(a.x), PY(a.y)], [PX(b.x), PY(b.y)], r);
          if (b && c && (c.v == null || c.v > 0.3)) seg([PX(b.x), PY(b.y)], [PX(c.x), PY(c.y)], r);
        }
      }
      for (const h of (f && f.hands) || []) {
        const r = Math.max(2, 1.4 * (h.scale || 0.06) * W);
        disc(PX(h.x), PY(h.y), r);
        if (face) seg([PX(h.x), PY(h.y)], [PX(face.x), Math.min(H - 1, PY(face.chinY != null ? face.chinY : face.y) + fw)], Math.max(1.5, 0.5 * r));   // the forearm toward the body
      }
      if (any) { S.lastPerson = Uint8Array.from(m); S.lastPersonT = t; }
      else if (S.lastPerson && t - S.lastPersonT < 2.0) m.set(S.lastPerson);    // tracking dropout: keep the last person region
      for (let i = 0; i < S.N; i++) if (m[i]) S.maskT[i] = t;
    }
    // best integer shift (camera translation) of g vs B over valid pixels: returns {dx, dy, res0, res}
    function shiftTest(g, valid, R) {
      const W = S.W, H = S.H; let best = {dx: 0, dy: 0, res: Infinity}, res0 = 0;
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        let s = 0, n = 0;
        for (let y = R; y < H - R; y++) for (let x = R; x < W - R; x++) {
          const i = y * W + x; if (!valid[i]) continue;
          s += Math.abs(g[(y + dy) * W + x + dx] - S.B[i]); n++;
        }
        const r = n ? s / n : 0;
        if (dx === 0 && dy === 0) res0 = r;
        if (r < best.res) best = {dx, dy, res: r};
      }
      best.res0 = res0; return best;
    }
    // returns triggers [{kind, strength, detail}] for this thumbnail
    function update(thumb, f, t) {
      const W = thumb.w, H = thumb.h, d = thumb.data, N = W * H;
      const g = new Float32Array(N);
      for (let i = 0, j = 0; i < N; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
      let mu = 0; for (let i = 0; i < N; i++) mu += g[i]; mu /= N;
      let va = 0; for (let i = 0; i < N; i++) va += (g[i] - mu) * (g[i] - mu); const sig = Math.sqrt(va / N) + 1;
      if (!S.B || S.W !== W || S.H !== H) { init(W, H, g, t); S.mu = mu; S.sig = sig; S.photoN = 0; S.hist = [[t, g]]; return []; }
      let g1 = null; for (const h of S.hist) if (t - h[0] >= 0.8) g1 = h[1];          // the thumbnail ≈ 1 s ago (stationarity)
      if (t - S.hist[S.hist.length - 1][0] >= 0.25) S.hist.push([t, g]);
      while (S.hist.length > 2 && t - S.hist[1][0] >= 1.0) S.hist.shift();
      const dt = clamp(t - S.lastT, 0.01, 1.0); S.lastT = t;
      S.tdt = S.tdt == null ? dt : S.tdt + (dt - S.tdt) * 0.2;      // the thumbnail interval (ages are counted in thumbnails)
      person(f, t);
      const warm = t - S.t0 < P.warmup || t < S.fastUntil;
      const aBg = 1 - Math.exp(-dt / P.scene.tauBg), aFg = 1 - Math.exp(-dt / P.scene.tauFg), aFast = 1 - Math.exp(-dt / P.scene.tauFast);
      const valid = S.valid, fg = S.fg; valid.fill(0); fg.fill(0); let nValid = 0, nFg = 0, sumSigned = 0;
      for (let i = 0; i < N; i++) {
        const ex = t - S.maskT[i] < P.scene.maskHold;
        if (ex) { S.age[i] = 0; if (!S.mask[i]) S.B[i] += (g[i] - S.B[i]) * aFast; continue; }   // the person's trail: re-learn fast, never judged
        valid[i] = 1; nValid++;
        const diff = g[i] - S.B[i], th = Math.max(P.scene.thMin, P.scene.thK * Math.sqrt(S.V[i]));
        if (Math.abs(diff) > th) { fg[i] = 1; nFg++; sumSigned += diff; }
      }
      const out = [];
      // whole frame, mask-free: a brightness or contrast jump (lights, the lens covered, the camera turned to a wall)
      const Ph = P.photo, aPh = 1 - Math.exp(-dt / Ph.tau);
      const photo = Math.abs(mu - S.mu) > Ph.dMean || sig < Ph.sigLo * S.sig || sig > Ph.sigHi * S.sig;
      if (photo && !warm) {
        S.photoN++;
        if (S.photoN === 2) {
          out.push({kind: 'scene_change', strength: 0.85, detail: {what: sig < Ph.sigLo * S.sig ? 'covered_or_blurred' : mu < S.mu - Ph.dMean ? 'darker' : mu > S.mu + Ph.dMean ? 'brighter' : 'contrast', mean: r3(mu), was: r3(S.mu)}});
          S.fastUntil = t + 1.5;
        }
        if (S.photoN > 12) { S.mu = mu; S.sig = sig; }                         // a new normal after ~2 s
      } else { S.photoN = 0; S.mu += (mu - S.mu) * aPh; S.sig += (sig - S.sig) * aPh; }
      if (photo && S.photoN >= 2) return out;
      const fgFrac = nValid ? nFg / nValid : 0;
      let cam = false;
      if (!warm && fgFrac >= 0.15 && W > 2 * P.scene.shift + 4 && H > 2 * P.scene.shift + 4) {
        const s = shiftTest(g, valid, P.scene.shift);
        if ((s.dx || s.dy) && s.res < 0.6 * s.res0) cam = true;               // the camera moved: re-learn, no candidate
      }
      if (warm || cam) {
        if (P.debug) (S.dbg = S.dbg || []).push({t: r3(t), fg: r3(fgFrac), valid: nValid, warm, cam});
        for (let i = 0; i < N; i++) if (valid[i]) { S.B[i] += (g[i] - S.B[i]) * (warm ? aFast : 0.6); S.age[i] = 0; }
        S.globalN = 0;
        return out;
      }
      // global change (lights, lens covered, camera turned) — persists ≥ 2 thumbs
      if (fgFrac >= P.scene.global) {
        S.globalN++;
        if (S.globalN === 2) {
          const mean = nFg ? sumSigned / nFg : 0;
          out.push({kind: 'scene_change', strength: clamp01(0.8 + 0.4 * (fgFrac - P.scene.global)), detail: {what: mean < -25 ? 'darker' : mean > 25 ? 'brighter' : 'global', frac: r3(fgFrac)}});
          S.fastUntil = t + 1.5;                                                 // then re-learn the new scene
        }
      } else S.globalN = 0;
      // blobs (4-connected components of fg pixels)
      const comp = S.comp; comp.fill(-1); const st = S.stack; let best = null; const objects = [];
      for (let i = 0; i < N; i++) {
        if (fg[i]) S.age[i] = Math.min(255, S.age[i] + 1); else S.age[i] = 0;
      }
      if (fgFrac < P.scene.global) {
        let id = 0;
        for (let i = 0; i < N; i++) {
          if (!fg[i] || comp[i] >= 0) continue;
          let sp = 0, area = 0, ageSum = 0, x0 = W, x1 = 0, y0 = H, y1 = 0, sx = 0, sy = 0, oldN = 0, stillN = 0;
          const stillAge = P.scene.objectAge / Math.max(0.02, S.tdt);
          st[sp++] = i; comp[i] = id;
          while (sp) {
            const k = st[--sp], x = k % W, y = (k - x) / W;
            area++; ageSum += S.age[k]; sx += x; sy += y; if (S.age[k] >= P.scene.persist) oldN++; if (S.age[k] >= stillAge) stillN++;
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
            if (x > 0 && fg[k - 1] && comp[k - 1] < 0) { comp[k - 1] = id; st[sp++] = k - 1; }
            if (x < W - 1 && fg[k + 1] && comp[k + 1] < 0) { comp[k + 1] = id; st[sp++] = k + 1; }
            if (y > 0 && fg[k - W] && comp[k - W] < 0) { comp[k - W] = id; st[sp++] = k - W; }
            if (y < H - 1 && fg[k + W] && comp[k + W] < 0) { comp[k + W] = id; st[sp++] = k + W; }
          }
          const c = {id, area, frac: area / Math.max(1, nValid), age: ageSum / area, old: oldN / area, still: stillN / area, cx: sx / area / W, cy: sy / area / H,
            fill: area / ((x1 - x0 + 1) * (y1 - y0 + 1)), bw: (x1 - x0 + 1) / W, bh: (y1 - y0 + 1) / H};
          if (!best || c.area > best.area) best = c;
          if (c.still >= 0.7 && c.frac <= P.scene.objectMax && c.area >= P.scene.minBlobPx && c.fill >= 0.3 && g1) {
            let ch = 0; for (let k = 0; k < N; k++) if (comp[k] === id) ch += Math.abs(g[k] - g1[k]);
            c.motion = ch / area; if (c.motion < P.scene.thMin * 0.6) objects.push(c);   // it stays put: unchanged for ~1 s
          }
          id++;
        }
      }
      const minArea = Math.max(P.scene.minBlobPx, P.scene.minBlob * nValid);
      if (best && best.area >= minArea && best.old >= 0.5) {
        out.push({kind: 'scene_change', strength: clamp01(0.3 + 2.5 * best.frac), detail: {what: 'region', frac: r3(best.frac), x: r3(best.cx), y: r3(best.cy), age_s: r3(best.age * S.tdt)}});
      }
      for (const o of objects) {
        out.push({kind: 'new_object', strength: clamp01(0.4 + 3 * o.frac), detail: {what: 'blob', frac: r3(o.frac), x: r3(o.cx), y: r3(o.cy)}});
        for (let i = 0; i < N; i++) if (comp[i] === o.id) { S.B[i] = g[i]; S.age[i] = 0; }   // absorbed: it is part of the scene now
      }
      // background update
      for (let i = 0; i < N; i++) {
        if (!valid[i]) continue;
        const diff = g[i] - S.B[i];
        if (fg[i]) S.B[i] += diff * aFg;
        else { S.B[i] += diff * aBg; S.V[i] += (diff * diff - S.V[i]) * aBg; if (S.V[i] < 4) S.V[i] = 4; }
      }
      S.lastFg = fgFrac;
      if (P.debug) (S.dbg = S.dbg || []).push({t: r3(t), mu: r3(mu), sig: r3(sig), rmu: r3(S.mu), rsig: r3(S.sig), fg: r3(fgFrac), valid: nValid, best: best ? {frac: r3(best.frac), old: r3(best.old), x: r3(best.cx), y: r3(best.cy)} : null, g: S.globalN, out: out.map(x => x.kind)});
      return out;
    }
    return {update, reset: () => { S.B = null; S.lastPerson = null; }, state: S};
  }

  // ─────────────────────────────────────────── the detector ───────────────────────────────────────────
  function create(opts) {
    opts = opts || {};
    const P = merge(DEFAULTS, opts);
    if (opts.rate) P.rate = merge(DEFAULTS.rate, opts.rate);
    const send = opts.send || (() => {}), onEvent = opts.onEvent || (() => {});
    const clock = opts.now || nowMs;
    const scene = createScene(P);
    const A = {t0: null, lastT: null, frames: 0, msSum: 0, msArr: [], thumbs: 0, thumbMs: 0, drawMs: 0, drawN: 0,
      lastThumbRef: null, lastThumbT: -99, sent: [], kindSent: {}, pending: [], inflight: null, cand: 0,
      byKind: {}, supp: {weak: 0, cooldown: 0, global: 0, inflight: 0, stale: 0}, verdicts: 0, worth: 0, rules: [], lastVerdictId: null,
      byId: new Map(), recentEv: [], last: [],
      // motion
      mBase: null, mDev: 0.02, mHiSince: null, mArmed: true,
      // expression
      xpHist: [], jawBase: null, jawHiSince: null, jawArmed: true, lastSpeakT: -99, levelHist: [],
      // faces / gaze
      faceWas: false, lastFace: null, lastFaceT: -99, faceSince: null, attendedSinceAppear: false, attWas: false, attOffSince: null,
      // sound
      quietSince: null, quietRun: 0, lastQuietT: null, floor: null, sndT: null, peaks: [], snd: null, lastAvT: -99, szBase: null, szArmed: true, szHiSince: null,
      // page thumbnail / ring
      tcv: null, tcx: null, ring: [], ringI: 0, ringT: -99};

    function noteEv(type, t) { A.recentEv.push([t, type]); while (A.recentEv.length && t - A.recentEv[0][0] > 3) A.recentEv.shift(); }

    // ── page: thumbnail + JPEG ring from opts.video ──
    function pageThumb(t) {
      const v = opts.video; if (!v || typeof document === 'undefined' || !v.videoWidth || v.readyState < 2) return null;
      if (t - A.lastThumbT < P.thumb.every) return null;
      const t0 = nowMs();
      const vw = v.videoWidth, vh = v.videoHeight, L = P.thumb.long, w = vw >= vh ? L : Math.max(4, Math.round(L * vw / vh)), h = vw >= vh ? Math.max(4, Math.round(L * vh / vw)) : L;
      if (!A.tcv || A.tcv.width !== w || A.tcv.height !== h) { A.tcv = document.createElement('canvas'); A.tcv.width = w; A.tcv.height = h; A.tcx = A.tcv.getContext('2d', {willReadFrequently: true}); }
      try { A.tcx.drawImage(v, 0, 0, w, h); const data = A.tcx.getImageData(0, 0, w, h).data; A.drawMs += nowMs() - t0; A.drawN++; return {w, h, data, t}; }
      catch (e) { return null; }
    }
    function ringTick(t) {
      const v = opts.video; if (!v || typeof document === 'undefined' || !v.videoWidth || v.readyState < 2) return;
      if (t - A.ringT < P.ring.every) return;
      A.ringT = t;
      const vw = v.videoWidth, vh = v.videoHeight, L = P.ring.long, w = vw >= vh ? L : Math.round(L * vw / vh), h = vw >= vh ? Math.round(L * vh / vw) : L;
      let slot = A.ring[A.ringI];
      if (!slot || slot.c.width !== w || slot.c.height !== h) { const c = document.createElement('canvas'); c.width = w; c.height = h; slot = A.ring[A.ringI] = {c, x: c.getContext('2d'), t: -99}; }
      try { const t0 = nowMs(); slot.x.drawImage(v, 0, 0, w, h); slot.t = t; A.drawMs += nowMs() - t0; } catch (e) { slot.t = -99; }
      A.ringI = (A.ringI + 1) % P.ring.size;
    }
    function encode(canvas) {
      try {
        let u = canvas.toDataURL('image/jpeg', P.ring.quality);
        if (u.length * 0.75 > P.ring.maxBytes) u = canvas.toDataURL('image/jpeg', 0.45);
        return u;
      } catch (e) { return null; }
    }
    function grabNow() {
      if (opts.grab) { try { return opts.grab(P.ring.long, null, P.ring.quality, 0) || null; } catch (e) { return null; } }
      const v = opts.video; if (!v || typeof document === 'undefined' || !v.videoWidth) return null;
      const vw = v.videoWidth, vh = v.videoHeight, L = P.ring.long;
      const c = document.createElement('canvas'); c.width = vw >= vh ? L : Math.round(L * vw / vh); c.height = vw >= vh ? Math.round(L * vh / vw) : L;
      try { c.getContext('2d').drawImage(v, 0, 0, c.width, c.height); } catch (e) { return null; }
      return encode(c);
    }
    function grabBefore(t) {
      if (opts.grab) { try { return opts.grab(P.ring.long, null, P.ring.quality, P.ring.before) || null; } catch (e) { return null; } }
      let best = null;
      for (const s of A.ring) if (s && s.t > -99 && t - s.t >= 0.5 && (!best || Math.abs(t - s.t - P.ring.before) < Math.abs(t - best.t - P.ring.before))) best = s;
      return best ? encode(best.c) : null;
    }

    // ── triggers ──
    function motionTriggers(f, t, dt, out) {
      const m = +f.motion || 0;
      if (A.mBase == null) { A.mBase = m; return; }
      const age = t - A.t0, M = P.motion;
      const th = A.mBase + Math.max(M.minRise, M.k * A.mDev);
      const hi = m > th && m > M.abs;
      const a = 1 - Math.exp(-dt / (hi ? M.tau * 4 : M.tau));
      A.mBase += (m - A.mBase) * a; A.mDev += (Math.abs(m - A.mBase) - A.mDev) * a; if (A.mDev < 0.01) A.mDev = 0.01;
      if (age < 2) return;
      if (hi) {
        if (A.mHiSince == null) A.mHiSince = t;
        if (A.mArmed && t - A.mHiSince >= M.hold) { A.mArmed = false; out.push({kind: 'motion_burst', strength: clamp01(0.2 + (m - A.mBase) / 0.5), detail: {what: 'motion', level: r3(m), base: r3(A.mBase)}}); }
      } else { A.mHiSince = null; if (m < A.mBase + 0.5 * (th - A.mBase)) A.mArmed = true; }
    }
    // the user's distance: face size vs its running baseline (leaning in close / moving back)
    function distanceTriggers(f, t, dt, out) {
      const face = f.face && f.face.present ? f.face : null, M = P.motion;
      if (!face || !(face.size > 0)) { A.szHiSince = null; return; }
      if (A.szBase == null || t - A.lastFaceT > 1.0) { A.szBase = face.size; A.szArmed = true; A.szHiSince = null; return; }
      const ratio = face.size / A.szBase, far = ratio < M.farther, near = ratio > M.closer;
      if ((near || far) && A.szArmed) {
        if (A.szHiSince == null) A.szHiSince = t;
        if (t - A.szHiSince >= M.sizeHold) { A.szArmed = false; out.push({kind: 'distance_change', strength: clamp01(0.45 + Math.abs(Math.log(ratio))), detail: {what: near ? 'closer' : 'farther', ratio: r3(ratio)}}); }
      } else A.szHiSince = null;
      if (ratio > 0.85 && ratio < 1.18) A.szArmed = true;
      A.szBase += (face.size - A.szBase) * (1 - Math.exp(-dt / M.sizeTau));
    }
    function exprTriggers(f, t, dt, ctx, out) {
      const face = f.face && f.face.present ? f.face : null;
      if (!face) { A.xpHist.length = 0; A.jawHiSince = null; return; }
      const xp = face.xp || {}, sc = xp.scores || {};
      A.xpHist.push([t, sc]); while (A.xpHist.length && t - A.xpHist[0][0] > P.expr.jumpWin) A.xpHist.shift();
      for (const k in sc) {
        const v = +sc[k] || 0; if (v < P.expr.jumpTo) continue;
        let lo = v; for (const [, s] of A.xpHist) lo = Math.min(lo, +s[k] || 0);
        if (v - lo >= P.expr.jump) { A.xpHist.length = 0; out.push({kind: 'expression_spike', strength: clamp01(0.3 + 0.6 * (v - lo)), detail: {what: k, src: 'jump', from: r3(lo), to: r3(v)}}); break; }
      }
      // wide-open jaw held while silent (yawn, gasp)
      const jaw = face.expr ? +face.expr.jawOpen || 0 : 0;
      if (A.jawBase == null) A.jawBase = jaw;
      const silent = !ctx.userSpeaking && t - A.lastSpeakT > 0.5;
      const jhi = jaw - A.jawBase >= P.expr.jaw;
      A.jawBase += (jaw - A.jawBase) * (1 - Math.exp(-dt / (jhi ? 30 : 4)));
      if (jhi && silent) {
        if (A.jawHiSince == null) A.jawHiSince = t;
        if (A.jawArmed && t - A.jawHiSince >= P.expr.jawHold) { A.jawArmed = false; out.push({kind: 'expression_spike', strength: clamp01(0.45 + jaw - A.jawBase), detail: {what: 'jaw_open', jaw: r3(jaw)}}); }
      } else { A.jawHiSince = null; if (jaw - A.jawBase < 0.5 * P.expr.jaw) A.jawArmed = true; }
    }
    function faceTriggers(f, t, ctx, out) {
      const face = f.face && f.face.present ? f.face : null;
      for (const ev of f.events || []) {
        noteEv(ev.type, t);
        if (ev.type === 'face_appear') {
          const away = ev.away_s == null ? null : +ev.away_s;
          const first = away == null;
          if ((first && t - A.t0 >= 2.5) || (!first && away >= 2.5)) {
            let s = first ? 0.5 : clamp01(0.3 + 0.04 * Math.min(away, 10)), diff = false;
            if (A.lastFace && face) {
              const ratio = (face.size || 0.1) / (A.lastFace.size || 0.1), jump = Math.hypot(face.x - A.lastFace.x, face.y - A.lastFace.y);
              if (ratio > 1.7 || ratio < 0.6 || jump > 0.6) { diff = true; s = clamp01(s + 0.25); }
            }
            out.push({kind: 'new_face', strength: s, detail: {what: diff ? 'different' : first ? 'first' : 'returned', away_s: away}});
          }
        } else if (ev.type === 'look_back') {
          const away = +ev.away_s || 2;
          out.push({kind: 'gaze_to_camera', strength: clamp01(0.35 + 0.12 * (away - 2)), detail: {what: 'look_back', away_s: r3(away)}});
        } else if (ev.type === 'expression_change') {
          out.push({kind: 'expression_spike', strength: clamp01(+ev.intensity || 0.7), detail: {what: ev.to, from: ev.from}});
        } else if (ev.type === 'object_up') {
          out.push({kind: 'new_object', strength: 0.7, detail: {what: 'held_up', side: ev.side}});
        } else if (P.gestures && (ev.type === 'wave' || ev.type === 'point' || ev.type === 'thumbs_up')) {
          out.push({kind: 'gesture', strength: ev.type === 'wave' ? 0.7 : 0.6, detail: {what: ev.type, side: ev.side}});
        } else if (ev.type === 'approach_fast') {
          out.push({kind: 'motion_burst', strength: 0.55, detail: {what: 'approach', source: ev.source}});
        }
      }
      // first look at the camera of a face that has not attended since it appeared
      if (face) {
        if (!A.faceWas) { A.faceSince = t; A.attendedSinceAppear = false; }
        if (f.attending && !A.attWas && !A.attendedSinceAppear && t - A.faceSince >= 1.5 && t - A.t0 >= 2.0)
          out.push({kind: 'gaze_to_camera', strength: 0.5, detail: {what: 'first_look', after_s: r3(t - A.faceSince)}});
        if (f.attending) A.attendedSinceAppear = true;
        A.lastFace = {x: face.x, y: face.y, size: face.size}; A.lastFaceT = t;
      }
      A.faceWas = !!face; A.attWas = !!(face && f.attending);
    }
    // loud sound: QUIET is relative to the room's noise floor (a slow low tracker), not an absolute level; the onset may
    // build over a few frames (≤ 0.3 s after the quiet ends); an impulse (falls to 30 % within 0.4 s, then mostly quiet up to
    // 0.45 s after the onset — a first syllable is followed by more speech) fires, a
    // sustained loud sound fires too while the user is not speaking and the avatar is silent (a shout, a crash, music blaring)
    function soundTriggers(t, ctx, out) {
      const L = +ctx.userLevel || 0, S = P.sound, dt = A.sndT == null ? 0.05 : clamp(t - A.sndT, 0.001, 0.5); A.sndT = t;
      A.floor = A.floor == null ? L : A.floor + (L - A.floor) * Math.min(1, dt / (L < A.floor ? 0.5 : 20));
      const qth = Math.max(0.012, 2.5 * A.floor), loud = L > qth;
      if (!loud) { if (A.quietSince == null) A.quietSince = t; A.lastQuietT = t; }
      else if (A.quietSince != null) { A.quietRun = t - A.quietSince; A.quietSince = null; }
      // the user's speech peaks (a slow, robust reference)
      if (ctx.userSpeaking && L > qth && !A.snd) { A.peaks.push(L); if (A.peaks.length > 400) A.peaks.splice(0, 200); }
      if (A.snd) {                                               // confirming an onset
        const s = A.snd; if (s.decayT == null) s.max = Math.max(s.max, L); if (ctx.userSpeaking) s.speaking = true;   // the user's own voice is not 'a loud sound'
        if (s.decayT != null) {
          // after an impulse: a bang is followed by quiet; a first syllable is followed by more speech
          s.n++; if (L > qth) s.nl++;
          if (t - s.t >= 0.45) { A.snd = null; if (s.nl / Math.max(1, s.n) < 0.4) out.push({kind: 'loud_sound', strength: clamp01(0.5 + (s.max - s.th) * 2), detail: {what: 'impulse', peak: r3(s.max)}}); }
        } else if (L < S.decay * s.max) { s.decayT = t; s.n = 0; s.nl = 0; }
        else if (t - s.t > S.confirm) {
          A.snd = null;
          if (!ctx.avatarSpeaking && !s.speaking) out.push({kind: 'loud_sound', strength: clamp01(0.45 + (s.max - s.th) * 2), detail: {what: 'sustained', peak: r3(s.max)}});
        }
      } else if (loud && (A.quietRun || 0) >= S.quiet && t - (A.lastQuietT == null ? -99 : A.lastQuietT) <= 0.3) {
        let ref = 0; if (A.peaks.length >= 20) { const a = A.peaks.slice().sort((x, y) => x - y); ref = a[Math.floor(0.9 * (a.length - 1))]; }
        const th = Math.max(ref > 0 ? S.abs : 0.25, S.rel * ref, 6 * qth);   // the user's speech level unknown yet → a higher bar
        if (L >= th) A.snd = {t, max: L, th, speaking: false, decayT: null};
      }
    }

    // ── gating, merging, sending ──
    function gate(trs, t, ctx) {
      const R = P.rate, conv = !!ctx.userSpeaking || t - A.lastSpeakT < 1.5 || !!ctx.avatarSpeaking || t - A.lastAvT < 1.0;
      for (const tr of trs) {
        if (P.debug) (A.dbgTr = A.dbgTr || []).push({t: r3(t), kind: tr.kind, s: r3(tr.strength), d: tr.detail, conv});
        A.byKind[tr.kind] = A.byKind[tr.kind] || {triggers: 0, sent: 0};
        A.byKind[tr.kind].triggers++;
        if (conv) {
          const w = tr.detail && tr.detail.what, C = R.conv;
          // a fast xp jump while talking is mostly speech moving the brows / jaw: only perception's debounced expression_change counts
          const k = tr.kind === 'expression_spike' && tr.detail && tr.detail.src === 'jump' ? Math.min(C.jump, C[w] != null ? C[w] : 1)
            : tr.kind === 'expression_spike' && C[w] != null ? C[w] : tr.kind === 'new_face' && w === 'different' ? 1 : (C[tr.kind] != null ? C[tr.kind] : 1);
          tr.strength = r3(tr.strength * k); tr.conv = true;
        }
        if (tr.strength < (conv ? Math.max(R.minStrength, R.speakingMin) : R.minStrength)) { A.supp.weak++; continue; }
        const key = tr.kind === 'expression_spike' ? tr.kind + ':' + (tr.detail && tr.detail.what) : tr.kind;   // each expression has its own cooldown
        const ks = A.kindSent[key] || [];
        const nRecent = ks.filter(x => t - x < R.habituate).length;
        const cd = (R.cooldown[tr.kind] || 6) * (1 + nRecent);
        if (ks.length && t - ks[ks.length - 1] < cd) { A.supp.cooldown++; continue; }
        const p = A.pending.find(x => x.kind === tr.kind);
        if (p) { if (tr.strength > p.strength) Object.assign(p, tr, {t: p.t}); }
        else A.pending.push(Object.assign({t}, tr));
      }
      A.pending = A.pending.filter(p => t - p.t <= R.pendingMax || (A.supp.stale++, false));
      if (!A.pending.length) return null;
      const urgent = A.pending.some(p => p.kind === 'loud_sound' || (p.kind === 'new_face' && p.detail && p.detail.what === 'different') ||
        (p.kind === 'scene_change' && p.detail && p.detail.what !== 'region' && p.strength >= R.urgent));
      if (A.inflight && t - A.inflight.t >= R.inflightMax) A.inflight = null;
      if (A.inflight && !urgent) return null;
      const last = A.sent.length ? A.sent[A.sent.length - 1] : -99;
      if (t - last < (urgent ? R.hardGap : R.minGap)) return null;
      while (A.sent.length && t - A.sent[0] > 60) A.sent.shift();
      if (A.sent.length >= R.perMin) return null;
      return fire(t, ctx);
    }
    function fire(t, ctx) {
      const items = A.pending.splice(0).sort((a, b) => b.strength - a.strength);
      const id = 'at' + (++A.cand) + '-' + Math.floor(t * 1000).toString(36);
      const kinds = items.map(x => x.kind), details = {};
      for (const x of items) {
        const key = x.kind === 'expression_spike' ? x.kind + ':' + (x.detail && x.detail.what) : x.kind;
        details[x.kind] = x.detail; (A.kindSent[key] = A.kindSent[key] || []).push(t); A.byKind[x.kind].sent++; if (A.kindSent[key].length > 20) A.kindSent[key].shift();
      }
      A.sent.push(t);
      const f = A.lastF || {}, face = f.face && f.face.present ? f.face : null;
      const imgNow = grabNow(), imgBefore = grabBefore(t);
      const images = [imgBefore, imgNow].filter(Boolean);
      const summary = {speaking: !!ctx.userSpeaking, avatar_speaking: !!ctx.avatarSpeaking, expression: face && face.xp ? face.xp.state : null,
        attending: !!f.attending, face: !!face, hands: (f.hands || []).length, motion: r3(+f.motion || 0), level: r3(+ctx.userLevel || 0),
        recent: [...new Set(A.recentEv.map(x => x[1]))]};
      const msg = {type: 'attention_candidate', id, t: r3(t), kinds, reason: familyText(kinds, details), strength: r3(items[0].strength), details, images, summary};
      A.inflight = {id, t};
      const rec = {id, t, kinds, details, strength: items[0].strength, now: imgNow, sentAt: clock()};
      A.byId.set(id, rec); if (A.byId.size > 50) A.byId.delete(A.byId.keys().next().value);
      A.last.push({id, t: r3(t), kinds, strength: r3(items[0].strength)}); if (A.last.length > 30) A.last.shift();
      try { send(msg); } catch (e) {}
      return msg;
    }

    function frame(f, ctx) {
      const c0 = nowMs();
      ctx = ctx || {};
      const t = ctx.t != null ? +ctx.t : +f.t;
      if (A.t0 == null) A.t0 = t;
      const dt = A.lastT == null ? 0.055 : clamp(t - A.lastT, 0.001, 1.0); A.lastT = t;
      A.lastF = f;
      if (ctx.userSpeaking) A.lastSpeakT = t;
      if (ctx.avatarSpeaking) A.lastAvT = t;
      const trs = [];
      motionTriggers(f, t, dt, trs);
      distanceTriggers(f, t, dt, trs);
      exprTriggers(f, t, dt, ctx, trs);
      faceTriggers(f, t, ctx, trs);
      soundTriggers(t, ctx, trs);
      let thumb = ctx.thumb || null;
      if (!thumb && opts.video) thumb = pageThumb(t);
      if (thumb && thumb !== A.lastThumbRef && (thumb.t == null || thumb.t !== A.lastThumbT)) {
        A.lastThumbRef = thumb; A.lastThumbT = thumb.t != null ? thumb.t : t;
        const s0 = nowMs();
        for (const x of scene.update(thumb, f, t)) trs.push(x);
        A.thumbMs += nowMs() - s0; A.thumbs++;
      }
      if (opts.video) ringTick(t);
      const msg = gate(trs, t, ctx);
      const ms = nowMs() - c0;
      A.frames++; A.msSum += ms; A.msArr.push(ms); if (A.msArr.length > 2000) A.msArr.splice(0, 1000);
      return msg;
    }

    // the event family in words ('loud sound', 'new face', 'motion burst', 'scene change', 'gaze to camera', 'new object', …)
    function familyText(kinds, details) {
      return [...new Set((kinds || []).map(k => String(k).replace(/_/g, ' ')))].join(' + ');
    }
    function onHub(msg) {
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'attention_rules') { A.rules = Array.isArray(msg.rules) ? msg.rules : []; try { onEvent({type: 'attention_rules', rules: A.rules}); } catch (e) {} return; }
      if (msg.type !== 'attention') return;
      const rec = A.byId.get(msg.id);
      if (A.inflight && A.inflight.id === msg.id) A.inflight = null;
      A.verdicts++;
      const age = rec ? (clock() - rec.sentAt) / 1000 : null;
      const stale = age != null && age > P.rate.maxAge;
      const worth = !!msg.worth_reply && !stale && !msg.skipped;
      if (worth) A.worth++;
      if (msg.id) A.lastVerdictId = msg.id;
      const fam = familyText(rec ? rec.kinds : (msg.kinds || []), rec ? rec.details : null), mr = String(msg.reason || '');
      const ev = {type: 'attention', id: msg.id, what: String(msg.what || ''), worth_reply: worth, salience: +msg.salience || 0,
        reason: fam && mr.toLowerCase().indexOf(fam.split(' + ')[0]) !== 0 ? fam + (mr ? ': ' + mr : '') : mr || fam, family: fam,
        t: rec ? r3(rec.t) : msg.t, kinds: rec ? rec.kinds : (msg.kinds || []), age_s: age == null ? null : r3(age), stale, image: worth && rec ? rec.now : null,
        ms: msg.ms, model: msg.model};
      if (msg.skipped) ev.skipped = msg.skipped;
      try { onEvent(ev); } catch (e) {}
      return ev;
    }
    function instruct(text) { const m = {type: 'attention_rule', op: 'add', text: String(text || '').slice(0, 300)}; send(m); return m; }
    function feedback(text, o) { const m = {type: 'attention_rule', op: 'feedback', text: String(text || '').slice(0, 300), id: (o && o.id) || A.lastVerdictId}; send(m); return m; }
    function forget(id) { const m = id === 'all' ? {type: 'attention_rule', op: 'clear'} : {type: 'attention_rule', op: 'remove', id}; send(m); return m; }
    function stats() {
      const a = A.msArr.slice().sort((x, y) => x - y);
      return {frames: A.frames, ms_per_frame: A.frames ? r3(A.msSum / A.frames) : 0, ms_p99: a.length ? r3(a[Math.floor(0.99 * (a.length - 1))]) : 0,
        thumbs: A.thumbs, ms_per_thumb: A.thumbs ? r3(A.thumbMs / A.thumbs) : 0, thumb_draw_ms: A.drawN ? r3(A.drawMs / A.drawN) : 0,
        candidates: A.cand, sent: A.sent.length, by_kind: A.byKind, suppressed: Object.assign({}, A.supp), verdicts: A.verdicts, worth_reply: A.worth,
        inflight: A.inflight ? A.inflight.id : null, last: A.last.slice(-10)};
    }
    function reset() { scene.reset(); A.mBase = null; A.jawBase = null; A.xpHist.length = 0; A.peaks.length = 0; A.snd = null; A.floor = null; A.quietSince = null; A.quietRun = 0; A.t0 = null; A.lastT = null; }
    return {frame, onHub, instruct, feedback, forget, rules: () => A.rules.slice(), stats, reset, _scene: scene, _A: A};
  }

  return {create, KINDS, DEFAULTS, _test: {createScene, merge}};
});
