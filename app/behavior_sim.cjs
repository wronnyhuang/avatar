/* behavior_sim.js — program layer: synthetic perception + dry-run hosts for app/behavior.js (node only: the server's
 * checker tools/behavior_check.cjs, tests/behavior_test.cjs, tests/behavior_dev.cjs). Not inlined into the page.
 *
 *   synth(script, {fps=18}) → {frames: [feature frame f as perception.js emits it], voice(t) → {user, avatar}, dur}
 *     script = {dur, base: {...state}, at: [[t0, t1, {...state}], ...], events: [[t, type, data], ...]}
 *     state keys (perception conventions): face (bool), x, y, z (m), size, yaw, pitch (+ = nose DOWN), roll, lookX, lookY
 *     (+ = right / up), attending (bool), expr {smile, browUp, browDown, eyeWide, sad, jawOpen, squint, pucker, cheekPuff},
 *     blink (both eyes closed), blinkL / blinkR (THE USER'S left / right eye), xp {state, scores}, hand_left / hand_right
 *     {x, y, open, gesture, touch: part, point, scale}, pose {lean, shrug}, speaking, level (the user's voice), self_speaking.
 *   TOUR = the generic dry-run script (every cue twice-ish in ~32 s); tour(extra) adds taught-gesture / speech events.
 *   dryRun(code, {character, script | frames, recorded: bool, fps=30, host: 'sim' | 'motion', trace}) →
 *     {ok, error, errors, warnings, events (out: rule moments, behavior events), actions, stats {frames, ms_p50, ms_p95,
 *      ms_max, sets, emits, actions, acted, claims, done, status}, trace: [{t, sets}] (trace:true), runtime}
 */
'use strict';
const path = require('path'), fs = require('fs');
const load = rel => { const src = fs.readFileSync(path.join(__dirname, rel), 'utf8'); const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; };
const B = load('behavior.js');
let Motion = null, Moves = null;
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;

// ------------------------------------------------------------------ a synthetic face (the mesh points behavior.js reads)
const MESH_PTS = {1: [0, 0.62], 4: [0, 0.6], 5: [0, 0.57], 13: [0, 0.78], 14: [0, 0.82], 0: [0, 0.76], 17: [0, 0.86], 152: [0, 1.0], 175: [0, 0.99],
  10: [0, 0.02], 151: [0, 0.12], 9: [0, 0.3], 33: [0.3, 0.44], 133: [0.12, 0.44], 159: [0.2, 0.41], 145: [0.2, 0.47], 263: [-0.3, 0.44],
  362: [-0.12, 0.44], 386: [-0.2, 0.41], 374: [-0.2, 0.47], 50: [0.3, 0.64], 205: [0.27, 0.68], 187: [0.36, 0.62], 280: [-0.3, 0.64],
  425: [-0.27, 0.68], 411: [-0.36, 0.62], 234: [0.5, 0.5], 127: [0.5, 0.42], 93: [0.48, 0.58], 454: [-0.5, 0.5], 356: [-0.5, 0.42], 323: [-0.48, 0.58]};
const PART_PT = {nose: 1, mouth: 13, chin: 152, forehead: 151, right_eye: 159, left_eye: 386, right_cheek: 50, left_cheek: 280, right_ear: 234, left_ear: 454,
  eyes: 159, eye: 159, ear: 234, cheek: 50, head: 151, lips: 13};
// face geometry in perception units: x + = screen-right (= THE USER'S right in the mirror), y + = up. dx = THE USER'S right → + x.
function faceGeom(st) {
  const fx = st.x, fw = 2 * st.size, top = st.y + 0.62 * fw * 1.25, fh = 1.25 * fw * 1.15;
  const chin = top - fh;
  const P = i => { const [dx, fy] = MESH_PTS[i]; return [fx + dx * fw + 0.35 * fw * Math.sin(st.yaw) * (1 - Math.abs(dx) * 1.5), top - fy * fh]; };
  return {fx, fw, top, chin, fh, P};
}
function meshOf(g) {
  const m = new Float32Array(956);
  for (const i of Object.keys(MESH_PTS)) { const [x, y] = g.P(+i); m[2 * i] = x / 2 + 0.5; m[2 * i + 1] = 0.5 - y / 2; }
  return m;
}
// a hand: palm centre (x, y) or fingertips on a face part; lm = 21 × [vx, vy, z] (viewer 0..1)
function handOf(side, h, g, faceOn) {
  let x = +h.x || 0, y = +h.y || 0;
  const touch = h.touch && faceOn ? PART_PT[h.touch] : null;
  let tip = null;
  if (touch !== null && touch !== undefined) { const p = g.P(touch); tip = p; x = p[0] + (side === 'right' ? 0.06 : -0.06); y = p[1] - 0.18; }
  const sc = h.scale || 0.12, open = h.open === undefined ? 0.7 : h.open;
  const lm = [];
  for (let i = 0; i < 21; i++) {
    const f = Math.floor((i - 1) / 4), k = (i - 1) % 4;
    let px = x, py = y;
    if (i > 0) { const ang = (f - 2) * 0.35; const len = (k + 1) * 0.045 * (0.4 + 0.6 * open); px = x + Math.sin(ang) * len; py = y + 0.08 + Math.cos(ang) * len; }
    if (tip && (i === 8 || i === 12)) { px = tip[0] + (i === 12 ? 0.01 : 0); py = tip[1]; }
    if (h.pinch && (i === 4 || i === 8)) { px = x + 0.03; py = y + 0.1; }
    lm.push([px / 2 + 0.5, 0.5 - py / 2, 0]);
  }
  const g2 = h.gesture || (open > 0.6 ? 'Open_Palm' : open < 0.2 ? 'Closed_Fist' : 'None');
  const ext = [open, open, open, open].map(v => clamp(v, 0, 1));
  if (h.fingers !== undefined) for (let i = 0; i < 4; i++) ext[i] = i < h.fingers ? 1 : 0;
  return {side: x < (faceOn ? g.fx : 0) ? 'l' : 'r', handed: side, x, y, ax: Math.atan(x * 0.7), ay: Math.atan(y * 0.52), scale: sc, open,
    gesture: g2, gscore: 0.9, ext, point: h.point ? {dx: h.point[0] || 0, dy: h.point[1] || 1} : null, lm};
}
const BASE = {face: true, x: 0.0, y: 0.15, z: 0.5, size: 0.16, yaw: 0, pitch: 0, roll: 0, lookX: 0, lookY: -0.05, attending: true,
  expr: {}, blink: 0, blinkL: 0, blinkR: 0, speaking: false, level: 0, self_speaking: false};
function stateAt(script, t) {
  const st = Object.assign({}, BASE, script.base || {});
  st.expr = Object.assign({}, (script.base && script.base.expr) || {});
  for (const [t0, t1, s] of script.at || []) {
    if (t < t0 || t >= t1) continue;
    for (const [k, v] of Object.entries(s)) {
      if (k === 'expr') Object.assign(st.expr, v);
      else st[k] = v;
    }
  }
  return st;
}
function frameAt(script, t, events) {
  const st = stateAt(script, t);
  const g = faceGeom(st);
  const f = {t, dt: 0, face: {present: false}, hands: [], pose: null, motion: 0.02, approach: 0, attention: st.attending ? 0.9 : 0.1, attending: st.face ? !!st.attending : false, events};
  if (st.face) {
    const e = Object.assign({smile: 0.04, squint: 0.03, browUp: 0.08, browDown: 0.03, eyeWide: 0.04, sad: 0.02, jawOpen: 0.04, pucker: 0.02, cheekPuff: 0}, st.expr);
    const bl = Math.max(st.blink || 0, st.blinkL || 0) ? 0.9 : 0.06, br = Math.max(st.blink || 0, st.blinkR || 0) ? 0.9 : 0.06;
    e.blinkL = bl; e.blinkR = br; e.blink = (bl + br) / 2;
    if (st.speaking) e.jawOpen = Math.max(e.jawOpen, 0.15 + 0.2 * Math.abs(Math.sin(t * 9)));
    const blend = {eyeBlinkLeft: bl, eyeBlinkRight: br, jawOpen: e.jawOpen, mouthSmileLeft: e.smile, mouthSmileRight: e.smile, browInnerUp: e.browUp,
      browDownLeft: e.browDown, browDownRight: e.browDown, mouthFrownLeft: e.sad * 0.5, mouthFrownRight: e.sad * 0.5, mouthPucker: e.pucker, cheekPuff: e.cheekPuff,
      eyeWideLeft: e.eyeWide, eyeWideRight: e.eyeWide};
    f.face = {present: true, x: st.x, y: st.y, ax: Math.atan(st.x * 0.7), ay: Math.atan(st.y * 0.52), z: st.z, size: st.size, iod: st.size * 0.45,
      yaw: st.yaw, pitch: st.pitch, roll: st.roll, lookX: st.lookX, lookY: st.lookY, blend, expr: e, chinY: g.chin, topY: g.top, mesh: st.noMesh ? undefined : meshOf(g),
      xp: st.xp || (e.smile > 0.5 ? {state: 'happy', scores: {happy: e.smile}} : {state: 'neutral', scores: {}})};
  }
  for (const side of ['left', 'right']) { const h = st['hand_' + side]; if (h) f.hands.push(handOf(side, h, g, st.face)); }
  f.hands.sort((a, b) => a.x - b.x);
  if (st.pose || st.face) {
    const lean = st.pose ? +st.pose.lean || 0 : 0, shrug = st.pose ? +st.pose.shrug || 0 : 0;
    const sy = g.chin - 0.35 + 0.2 * shrug;
    f.pose = {present: true, shoulders: {l: {x: st.x - 0.35, y: sy + 0.1 * lean, v: 0.9}, r: {x: st.x + 0.35, y: sy - 0.1 * lean, v: 0.9}},
      lean: Math.atan2(0.1 * lean, 0.35), width: 0.35, handed: {l: 'left', r: 'right'}};
  }
  return {f, st};
}
function synth(script, o = {}) {
  const fps = o.fps || 18, dur = script.dur || 10, frames = [], evs = (script.events || []).slice().sort((a, b) => a[0] - b[0]);
  let ei = 0;
  for (let k = 0, t = 0; t <= dur + 1e-9; k++, t = k / fps) {
    const events = [];
    while (ei < evs.length && evs[ei][0] <= t + 1e-9) { const [, type, data] = evs[ei++]; events.push(Object.assign({type, t}, data || {})); }
    frames.push(frameAt(script, t, events.length ? events : undefined).f);
  }
  const vt = frames.map(f => { const st = stateAt(script, f.t); return Object.freeze({user: Object.freeze({speaking: !!st.speaking, level: st.speaking ? (st.level || 0.5) : 0}), avatar: Object.freeze({speaking: !!st.self_speaking, level: st.self_speaking ? 0.5 : 0})}); });
  const voice = t => vt[clamp(Math.floor(t * fps + 1e-6), 0, vt.length - 1)];
  return {frames, voice, dur, script};
}

// ------------------------------------------------------------------ the generic tour (every cue, ~32 s) for the server's dry run
const TOUR = {dur: 32, at: [
  [1.0, 1.15, {blink: 1}], [3.4, 3.55, {blink: 1}], [6.3, 6.45, {blink: 1}], [9.6, 9.75, {blink: 1}], [14.1, 14.25, {blink: 1}],
  [20.4, 20.55, {blink: 1}], [24.0, 24.15, {blink: 1}], [29.2, 29.35, {blink: 1}], [12.2, 12.6, {blinkR: 1}],
  [2.0, 4.6, {expr: {smile: 0.85}}],
  [4.8, 6.0, {roll: 0.32}], [6.6, 7.8, {roll: -0.32}], [8.0, 9.0, {yaw: 0.45}],
  [9.2, 11.6, {hand_right: {x: 0.35, y: 0.45, open: 1}}], [9.4, 11.0, {hand_left: {x: -0.4, y: 0.3, open: 0.9}}],
  [12.8, 14.6, {hand_right: {touch: 'nose', open: 0.5}}], [15.0, 16.0, {hand_left: {touch: 'left_ear', open: 0.6}}],
  [16.2, 17.2, {hand_right: {touch: 'mouth', open: 0.4}}],
  [17.5, 19.5, {attending: false, lookX: 0.5, lookY: 0.05, yaw: 0.45}],
  [19.8, 22.5, {speaking: true, level: 0.7}], [21.0, 22.2, {pitch: -0.15}],
  [22.6, 23.4, {hand_right: {x: 0.05, y: -0.5, open: 1}, hand_left: {x: -0.05, y: -0.5, open: 1}}],
  [23.6, 24.6, {hand_right: {x: 0.3, y: 0.0, open: 0.1, gesture: 'Thumb_Up'}}],
  [24.6, 25.6, {expr: {jawOpen: 0.75}}], [25.6, 26.6, {expr: {browUp: 0.8, eyeWide: 0.6}, xp: {state: 'surprised', scores: {surprised: 0.8, raised_brows: 0.7}}}],
  [26.6, 27.6, {expr: {browDown: 0.7, sad: 0.4}, xp: {state: 'frown', scores: {frown: 0.8}}}],
  [27.8, 29.0, {face: false}], [29.0, 30.2, {z: 0.32}], [30.2, 31.0, {pose: {shrug: 1}}], [30.0, 32.0, {self_speaking: true}],
  [31.0, 32.0, {hand_right: {x: 0.3, y: 0.2, open: 0.2, fingers: 2, gesture: 'Victory'}}]],
events: [[10.0, 'wave', {side: 'r', handed: 'right'}], [10.8, 'wave', {side: 'r', handed: 'right', continuing: true}], [20.0, 'nod', {amp: 0.1}],
  [21.5, 'nod', {amp: 0.12}], [23.0, 'head_shake', {amp: 0.2}], [24.0, 'thumbs_up', {side: 'r', handed: 'right'}], [11.0, 'point', {side: 'r', handed: 'right'}],
  [25.0, 'expression_change', {from: 'neutral', to: 'surprised', intensity: 0.8}], [9.8, 'object_up', {side: 'l', handed: 'left'}]]};
function tour() { return TOUR; }
const TOUR_CACHE = {};
function tourCached(fps) { return TOUR_CACHE[fps] || (TOUR_CACHE[fps] = synth(TOUR, {fps})); }

// ------------------------------------------------------------------ dry run
const REST_OUT = {smile: 0.12, ear_l: 0.12, ear_r: 0.12, tail_raise: 0.4, tail_curl: 0.15, hand_l_open: 0.2, hand_r_open: 0.2, nostril: 0.05};
function codeRefs(code) {   // literal names a program asks for: taught gestures, heard words
  const g = new Set(), w = new Set();
  for (const m of String(code).matchAll(/gesture\(\s*['"`]([^'"`]{1,40})['"`]/g)) g.add(m[1].toLowerCase());
  for (const m of String(code).matchAll(/heard\(\s*(\[[^\]]{0,300}\]|['"`][^'"`]{1,40}['"`])/g)) for (const q of m[1].matchAll(/['"`]([^'"`]{1,40})['"`]/g)) w.add(q[1].toLowerCase());
  return {gestures: [...g], words: [...w]};
}
function loadRecorded() {
  if (loadRecorded.c) return loadRecorded.c;
  const fp = path.join(__dirname, '..', 'tests', 'fixtures', 'behavior_rec.json');
  const out = [];
  if (fs.existsSync(fp)) {
    const d = JSON.parse(fs.readFileSync(fp, 'utf8'));
    for (const clip of Object.keys(d)) { const rows = d[clip]; if (!rows.length) continue; const t0 = rows[0].t; out.push({clip, frames: rows.map(f => Object.assign({}, f, {t: f.t - t0}))}); }
  }
  return (loadRecorded.c = out);
}
const RV = {};
function recVoice(clip) { const v = RV[clip] || (RV[clip] = Object.freeze({user: Object.freeze({speaking: clip === 'sam_story', level: clip === 'sam_story' ? 0.5 : 0}), avatar: Object.freeze({speaking: false, level: 0})})); return () => v; }
function percentile(a, p) { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }

/** run a program on frames with the light 'sim' host (no motion.js) or the full 'motion' host */
function dryRun(code, o = {}) {
  const character = o.character || 'fox', fps = o.fps || 30, dt = 1 / fps;
  const R = B.createBehaviors({character});
  const out = [], actions = [];
  let simState = {posture: o.posture || (character === 'muse' ? 'standing' : 'seated'), active: [], kf: null, holding: {left: null, right: null}, wearing: []};
  let simBusyUntil = -1;
  R.attach({character: () => character, doAction: (d) => { actions.push({t: R._R.clock, description: d}); simBusyUntil = R._R.clock + 2; simState = Object.assign({}, simState, {kf: {description: d}, active: [{kind: 'kf', name: d}]}); return {ok: true, status: 'preparing'}; },
    setStyle: s => out.push({type: 'style', t: R._R.clock, style: s})});
  R.onEvent(ev => out.push(ev));
  const segs = [];
  const refs = codeRefs(code);
  if (o.frames) segs.push({frames: o.frames, voice: o.voice || (() => ({user: {speaking: false, level: 0}, avatar: {speaking: false, level: 0}})), dur: o.frames.length ? o.frames[o.frames.length - 1].t : 0});
  else {
    if (o.script) segs.push(synth(o.script, {fps: o.featFps || 18}));
    else {   // the generic tour (cached) + the taught gestures / words this program asks for
      const base = tourCached(o.featFps || 18);
      const extra = [];
      for (const [i, gname] of refs.gestures.entries()) extra.push([12 + i * 0.3, {type: 'gesture', name: gname, score: 0.9}], [12.2 + i * 0.3, {type: 'gesture', name: gname, score: 0.9}], [22 + i * 0.3, {type: 'gesture', name: gname, score: 0.9}]);
      if (refs.words.length) extra.push([18.5, {type: 'said', text: 'okay ' + refs.words.join(' and ') + ' right'}], [26, {type: 'said', text: refs.words[0]}]);
      let frames = base.frames;
      if (extra.length) {
        frames = frames.slice();
        for (const [t, ev] of extra) {
          const i = frames.findIndex(f => f.t >= t - 1e-9); if (i < 0) continue;
          frames[i] = Object.assign({}, frames[i], {events: (frames[i].events || []).concat([Object.assign({t: frames[i].t}, ev)])});
        }
      }
      segs.push({frames, voice: base.voice, dur: base.dur});
    }
  }
  if (o.recorded) for (const r of loadRecorded()) segs.push({frames: r.frames, voice: recVoice(r.clip), dur: r.frames[r.frames.length - 1].t, clip: r.clip});
  let motion = null, ms = [], id = null, startErr = null;
  if (o.host === 'motion') {
    if (!Motion) { Moves = load('moves.js'); globalThis.AvatarMovesLib = Moves; Motion = load('motion.js'); }
    motion = Motion.createMotion({seed: 7, preset: 'default'});
    if (character === 'muse') {
      const anim = path.join(__dirname, '..', 'renderer', 'animals');
      motion.setProps(JSON.parse(fs.readFileSync(path.join(anim, 'muse.props.json'))), JSON.parse(fs.readFileSync(path.join(anim, 'muse.body.json'))), 'muse');
    } else motion.setProps(null, null, character);
    R.attach({motion});
  }
  id = R.start({name: o.name || 'dry', instruction: o.instruction || '', code, priority: o.priority});
  if (!id) startErr = R.lastError;
  const rigTrace = o.trace && motion ? [] : null;
  const progTrace = [];
  let tGlobal = 0;
  const PREC = id ? R._R.progs.find(p => p.id === id) : null;   // kept even after the program ends (done / stop)
  if (id) {
    const P = PREC; if (P && o.trace) P.trace = [];
    outer: for (const seg of segs) {
      let fi = 0, f = null;
      const n = Math.ceil(seg.dur * fps) + 1;
      for (let k = 0; k < n; k++) {
        const t = k * dt;
        while (fi < seg.frames.length && seg.frames[fi].t <= t + 1e-9) f = seg.frames[fi++];
        const ff = f;
        const v = seg.voice(t);
        if (R._R.clock > simBusyUntil && simState.kf) simState = Object.assign({}, simState, {kf: null, active: []});
        const t0 = Date.now();
        if (motion) {
          const rig = motion.step(dt, {features: ff, userVoice: {rms: v.user.speaking ? 0.06 : 0.001, voiced: v.user.speaking}, avatarVoice: {rms: v.avatar.speaking ? 0.05 : 0, voiced: v.avatar.speaking}});
          if (rigTrace) rigTrace.push({t: tGlobal, rig});
          if (!motion.programLayer && !R._R.progs.length) { tGlobal += dt; break outer; }
        } else {
          R.tick({dt, features: ff, voice: v, out: REST_OUT, state: () => simState, owned: () => ({}), actions: null, character});
        }
        ms.push(Date.now() - t0);
        tGlobal += dt;
        if (!R._R.progs.length) break outer;
        if (o.maxS && tGlobal >= o.maxS) break outer;
      }
    }
  }
  const info = id ? (R.info(id) || null) : null;
  const behEv = out.filter(e => e.type === 'behavior');
  const errEv = behEv.filter(e => e.phase === 'error' || e.phase === 'paused');
  const warnings = [...new Set(behEv.filter(e => e.phase === 'warning' || e.phase === 'unsupported').map(e => e.msg || e.what))];
  const endEv = behEv.find(e => e.phase === 'end');
  const progRec = PREC || null, running = !!(PREC && R._R.progs.includes(PREC));
  const recs = R._R.progs;
  const stats = {frames: progRec ? progRec.frames : 0, status: running ? progRec.status : (endEv ? 'ended' : (startErr ? 'failed' : 'unknown')),
    sim_s: +tGlobal.toFixed(2), emits: out.filter(e => e.type === 'rule_moment').length, actions: actions.length, styles: out.filter(e => e.type === 'style').length,
    done: endEv ? endEv.reason : null, claims: progRec ? [...progRec.claims] : null, sets: progRec ? progRec.nSets : null, unclaimed: progRec ? progRec.unclaimed : null};
  if (progRec) { const a = Array.from(progRec.ring.slice(0, Math.min(progRec.ringN, 30))); stats.ms_max = +progRec.maxMs.toFixed(3); stats.ms_mean = +(progRec.totMs / Math.max(1, progRec.frames)).toFixed(4); }
  const errors = [];
  if (startErr) errors.push(startErr);
  for (const e of errEv) errors.push(`${e.phase}${e.where ? ' in ' + e.where : ''}: ${e.error || e.reason}`);
  return {ok: !!id && !errEv.length, error: errors.join('; ') || null, errors, warnings, events: out, actions, stats,
    trace: o.trace && progRec ? progRec.trace : null, rig: rigTrace, runtime: R, id, logs: progRec ? progRec.logs.slice() : []};
}

module.exports = {synth, frameAt, stateAt, TOUR, tour, dryRun, codeRefs, loadRecorded, percentile, B, REST_OUT};
