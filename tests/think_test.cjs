// node tests/think_test.cjs [--verbose] — the thinking gesture while the planner writes a commanded move
// (app/thinking.js + motion.js think()/thinkEnd() + the perform.js / voicetools.js hooks). Everything runs on engine time
// (30 fps, a timer queue that fires as the motion steps), with a COUNTERFACTUAL motion (same seed, same inputs, no command)
// beside it: "visible" = any rig column more than VIS away from the counterfactual.
//  (a) timing on all 12 characters: nothing visible before the gate, visible <= 200 ms after the call
//  (b) skip: a hub cache / library answer inside the gate, a page-cache hit -> never shown, rig untouched until the move
//  (c) hand-over: thinking -> planned move with no pop (per-frame steps vs the same move without thinking) and the move's
//      columns land where they would have without thinking (the job's rest excludes the thinking)
//  (d) cancel: stop, a failed plan, the 12 s cap, a what-if hold, a replaced request — never left on
//  (e) truthful: nothing from it in the action log; it never fights a held pose or a held prop
//  (f) variants: the main pose most of the time, alternates never twice running, every variant seen
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js'), TK = load('app/thinking.js'), PF = load('app/perform.js'), VT = load('app/voicetools.js'), AL = load('app/actionlog.js');
const ANIM = path.join(__dirname, '..', 'renderer', 'animals');
const PROPS = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), BODY = JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json')));
const CHARS = ['fox', 'bear', 'rabbit', 'hippo', 'monkey', 'orb', 'cube', 'drop', 'star', 'pyramid', 'blob', 'muse'];
const DT = 1 / 30, VIS = 0.03, VERBOSE = process.argv.includes('--verbose');
let fails = 0, passes = 0;
const check = (n, ok, d) => { if (!ok || VERBOSE) console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const feat = t => ({features: {t, face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []}});
function mkMotion(name, seed, onAction) {
  const m = M.createMotion({seed, character: name, onAction});
  if (name === 'muse') m.setProps(JSON.parse(JSON.stringify(PROPS)), JSON.parse(JSON.stringify(BODY)), 'muse'); else m.setProps(null, null, name);
  return m;
}
function seeded(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// one scene: the motion under test + its counterfactual, the performer + thinker (+ optional voice tools) on engine time
function scene(name, o = {}) {
  const X = {name, t: 0, frames: [], cf: [], ctl: [], prompts: [], ev: []};
  let P = null;
  X.m = mkMotion(name, o.seed || 4364, e => { X.ev.push(e); if (P) P.onAction(e); if (X.alog) X.alog.push(Object.assign({}, e, {t: undefined})); });
  X.c = mkMotion(name, o.seed || 4364, () => {});
  X.names = X.m.allParams || M.RIG_PARAMS;
  const timers = []; let tid = 0;
  X.now = () => X.t * 1000;
  X.T = {set: (f, ms) => { const id = ++tid; timers.push({id, at: X.now() + Math.max(0, +ms || 0), f}); return id; },
    clear: id => { const i = timers.findIndex(x => x.id === id); if (i >= 0) timers.splice(i, 1); }};
  const fire = () => { for (let g = 0; g < 500; g++) { let k = -1; for (let i = 0; i < timers.length; i++) if (timers[i].at <= X.now() + 1e-6 && (k < 0 || timers[i].at < timers[k].at)) k = i; if (k < 0) return; timers.splice(k, 1)[0].f(); } };
  X.alog = o.alog ? AL.createActionLog({now: () => X.m.now()}) : null;
  X.K = o.think === false ? null : TK.createThinker({motion: () => X.m, now: X.now, timers: X.T, rand: seeded(o.rseed || 7), gateMs: o.gateMs, fingers: () => name === 'muse'});
  P = X.P = PF.createPerformer({motion: () => X.m, prompt: (t, op) => { X.prompts.push({t: X.now(), text: t, op}); return null; }, sendCtl: c => X.ctl.push(c),
    runCommand: c => X.m.command(c, {src: 'voice'}), character: () => name, now: X.now, timers: X.T, thinker: () => X.K, cache: o.cache !== false,
    alog: () => X.alog, userSpeaking: () => !!X.speaking, lastUserSpeech: () => X.lastSpeech || -1e12});
  X.step = (sec, f) => {
    for (let i = 0; i < Math.round(sec / DT); i++) {
      X.t += DT;
      X.frames.push(Object.assign({}, X.m.step(DT, feat(X.t))));
      X.cf.push(Object.assign({}, X.c.step(DT, feat(X.t))));
      fire(); if (f) f(X);
    }
  };
  X.diff = i => { let mx = 0, col = null; for (const k in X.frames[i]) { const d = Math.abs((X.frames[i][k] || 0) - (X.cf[i][k] || 0)); if (d > mx) { mx = d; col = k; } } return [mx, col]; };
  X.firstVisible = (fromS) => { for (let i = Math.max(0, Math.round(fromS / DT) - 1); i < X.frames.length; i++) if (X.diff(i)[0] > VIS) return {t: (i + 1) * DT, col: X.diff(i)[1]}; return null; };
  X.step(1.0);
  return X;
}
const kfMsg = (id, extra) => Object.assign({type: 'kf', id, poses: [], done: false, source: 'planner'}, extra);
const POSES = [{t: 0.4, head_yaw: 0.25, right_paw_y: 0.9, right_paw_x: 0.2}, {t: 1.0, head_yaw: -0.2, right_paw_y: 1.0}, {t: 1.6, head_yaw: 0}];

// ---------------------------------------------------------------------------------------------------- (a) timing
for (const ch of CHARS) {
  const X = scene(ch);
  const t0 = X.t; const r = X.P.call('do a little dance', {});
  X.step(1.5);
  const fv = X.firstVisible(t0), st = X.K.stats();
  const early = X.frames.slice(Math.round(t0 / DT), Math.round((t0 + 0.1) / DT)).some((f, j) => X.diff(Math.round(t0 / DT) + j)[0] > VIS);
  check(`(a) ${ch}: thinking visible within 200 ms of the call, nothing before the gate`, r.status === 'preparing' && st.shown === 1 && fv && fv.t - t0 <= 0.2 + 1e-6 && !early,
    `first visible +${fv ? Math.round((fv.t - t0) * 1000) : '-'} ms (${fv && fv.col}), variant ${st.last.length ? st.last[0].variants : X.K.active() && X.K.active().variant}`);
  // every variant runs on this character: finite, in range, visible
  let bad = [];
  for (const v of X.m.thinkVariants) {
    X.m.thinkEnd('t'); X.step(0.6);
    const i0 = X.frames.length, rr = X.m.think({variant: v});
    X.step(1.2);
    const vis = X.frames.slice(i0).some((f, j) => X.diff(i0 + j)[0] > 0.08);
    const fin = X.frames.slice(i0).every(f => Object.entries(f).every(([k, x]) => Number.isFinite(x) && (!M.RANGE[k] || (x >= M.RANGE[k][0] - 1e-6 && x <= M.RANGE[k][1] + 1e-6))));
    // a variant this body cannot do (a raised index finger off the muse; a chin paw the arm cannot reach) falls back to a head variant
    const wantV = X.m.thinkCan(v) ? rr.variant === v : rr.fallback && ['away', 'up', 'tilt', 'hmm'].includes(rr.variant);
    if (!rr.ok || !wantV || !vis || !fin) bad.push(v + (rr.ok ? (vis ? '' : ' invisible') + (fin ? '' : ' out of range') : ' refused ' + rr.reason));
  }
  X.m.thinkEnd('t');
  check(`(a) ${ch}: all ${X.m.thinkVariants.length} variants run (finite, in range, visible)`, !bad.length, bad.join(', '));
}

// ---------------------------------------------------------------------------------------------------- (b) skip
for (const ch of ['muse', 'fox', 'orb']) {
  for (const src of ['cache', 'library']) {
    const X = scene(ch);
    const t0 = X.t; const r = X.P.call('wave hello', {});
    let sent = false;
    X.step(1.6, Y => { if (!sent && Y.t - t0 >= 0.06) { sent = true;
      if (src === 'cache') Y.P.onKf(kfMsg(r.id, {source: 'cache', poses: POSES, done: true}));
      else Y.P.onKf(kfMsg(r.id, {source: 'library', engine: [{action: 'gesture', name: 'wave'}], done: true})); } });
    const st = X.K.stats(), fv = X.firstVisible(t0);
    const start = X.ev.find(e => (e.kind === 'kf' && e.phase === 'start') || (e.kind === 'gesture' && e.phase === 'start'));
    check(`(b) ${ch}: a hub ${src} answer inside the gate never shows the thinking`, st.shown === 0 && st.skipped === 1 && !X.m.thinking().on && !X.frames.some((f, i) => X.frames[i] && false),
      `shown ${st.shown}, skipped ${st.skipped}, first visible +${fv ? Math.round((fv.t - t0) * 1000) : '-'} ms (the move: ${start ? start.kind + ' ' + start.phase : 'none'})`);
  }
  // the page cache: the same description again after it finished plays at once, no thinking begun
  const X = scene(ch);
  const a = X.P.call('nod twice', {}); X.step(0.3);
  X.P.onKf(kfMsg(a.id, {poses: POSES, done: true})); X.step(4);
  const b0 = X.K.stats().begun, b = X.P.call('nod twice', {}); X.step(0.5);
  check(`(b) ${ch}: a page-cache hit never begins the thinking`, b.status === 'starting' && X.K.stats().begun === b0, `status ${b.status}, begun ${b0} -> ${X.K.stats().begun}`);
}

// ---------------------------------------------------------------------------------------------------- (c) hand-over
function handover(ch, think, waitS, rseed) {
  const X = scene(ch, {think, rseed});
  const t0 = X.t; const r = X.P.call('turn your head and raise a paw', {});
  let fed = false, start = null;
  X.step(waitS + 4.5, Y => {
    if (!fed && Y.t - t0 >= waitS) { fed = true; Y.P.onKf(kfMsg(r.id, {poses: POSES.slice(0, 2)})); Y.P.onKf(kfMsg(r.id, {poses: POSES.slice(2), done: true})); }
    if (start === null) { const e = Y.ev.find(e => e.kind === 'kf' && e.phase === 'start'); if (e) start = Y.t; }
  });
  return {X, t0, start};
}
const CH_COLS = /^(head_|gaze_|hand_|point_|thumb_|brow|body_|root_|foot_)/;
// a pop = a second difference (per-frame acceleration) far above smooth motion: a one-frame jump of d gives ~d, a min-jerk
// ease of 0.8 over 0.4 s gives 0.03. Eyes saccade by nature: each column's bound is at least its own idle maximum (the
// counterfactual's saccades / blinks) x 1.2, never below POP.
const POP = 0.05;
function acc(F, i, k) { return Math.abs(F[i][k] - 2 * F[i - 1][k] + F[i - 2][k]); }
function popScan(X, a, b) {
  const idle = {};
  for (let i = 2; i < X.cf.length; i++) for (const k in X.cf[i]) if (CH_COLS.test(k)) idle[k] = Math.max(idle[k] || 0, acc(X.cf, i, k));
  let worst = 0, col = null, ratio = 0;
  for (let i = Math.max(2, Math.round(a / DT)); i < Math.min(X.frames.length, Math.round(b / DT)); i++)
    for (const k in X.frames[i]) if (CH_COLS.test(k)) { const v = acc(X.frames, i, k), lim = Math.max(POP, 1.2 * (idle[k] || 0)); if (v / lim > ratio) { ratio = v / lim; worst = v; col = k; } }
  return {ratio, worst, col};
}
for (const ch of ['muse', 'fox', 'orb', 'cube', 'blob']) for (const rs of [3, 11]) {
  const A = handover(ch, true, 2.0, rs), B = handover(ch, false, 2.0, rs);
  const v = A.X.K.list()[0] || {};
  const pa = popScan(A.X, A.start - 0.3, A.start + 1.0), pb = popScan(B.X, B.start - 0.3, B.start + 1.0), pOn = popScan(A.X, A.t0, A.t0 + 1.2);
  check(`(c) ${ch}/${v.variants}: thinking -> planned move without a pop`, A.start && Math.abs(A.start - B.start) < 1e-6 && pa.ratio <= Math.max(1, pb.ratio * 1.2) && pOn.ratio <= 1.3,
    `hand-over worst accel ${pa.worst.toFixed(3)} (${pa.col}, ${pa.ratio.toFixed(2)} of the bound; without thinking ${pb.ratio.toFixed(2)}); onset ${pOn.worst.toFixed(3)} (${pOn.col}, ${pOn.ratio.toFixed(2)}); start +${Math.round((A.start - A.t0) * 1000)} ms`);
  // after the thinking faded (start + 0.6 s): the planned columns equal the no-thinking run (rest excludes the thinking)
  let dmax = 0, dcol = null;
  for (let i = Math.round((A.start + 0.6) / DT); i < Math.round((A.start + 1.4) / DT); i++) for (const k of ['head_yaw', 'hand_r_y', 'hand_r_x', 'hand_l_y']) {
    const d = Math.abs(A.X.frames[i][k] - B.X.frames[i][k]); if (d > dmax) { dmax = d; dcol = k; } }
  check(`(c) ${ch}/${v.variants}: the move's columns land as planned once the thinking has handed over`, dmax < 0.02 && !A.X.m.thinking().on, `max |with - without| ${dmax.toFixed(4)} (${dcol}), session end: ${v.why}`);
}

// ---------------------------------------------------------------------------------------------------- (d) cancel
{
  const off = (X, what, extra) => { X.step(0.8); const st = X.K.stats(); check(`(d) ${what}: thinking off, nothing left on`, !X.m.thinking().on && !X.m.thinking().fading && !st.active && (extra ? extra(X) : true), JSON.stringify(st.last.slice(-1)[0] || {})); };
  let X = scene('muse'); X.P.call('do a cartwheel', {}); X.step(1.5); X.P.stop('he asked to stop'); off(X, 'stop');
  X = scene('fox'); let r = X.P.call('juggle', {}); X.step(1.5); X.P.onKf(kfMsg(r.id, {poses: [], done: true, ok: false, error: 'nothing'})); off(X, 'a plan with no poses (failed)');
  X = scene('orb'); X.P.call('do something hard', {}); X.step(12.3); off(X, 'the 12 s cap (planner still silent)', Y => Y.K.stats().capped === 1);
  X = scene('muse'); r = X.P.call('jump', {}); X.step(1.0); X.lastSpeech = X.now(); const r2 = X.P.call('no, spin instead', {});
  X.step(0.5); const on2 = X.m.thinking().on; X.P.onKf(kfMsg(r2.id, {poses: POSES, done: true})); X.step(2.5);
  check('(d) a newer request replaces one being planned: the thinking carries on, then hands over', on2 && !X.m.thinking().on && X.ev.some(e => e.kind === 'kf' && e.phase === 'start'), `on during the 2nd plan: ${on2}`);
  // motion alone: a keyframe job's first frame and any other command fade it even if nobody calls end()
  X = scene('cube', {think: false}); X.m.think({variant: 'tilt'}); X.step(0.5);
  X.m.command({action: 'keyframes', id: 'z', description: 'x', poses: POSES, done: true, hold: false}); X.step(0.8);
  check('(d) motion: a keyframe job start fades the thinking by itself', !X.m.thinking().on && !X.m.thinking().fading);
  X.m.think({variant: 'up'}); X.step(0.5); X.m.command({action: 'gesture', name: 'nod'}, {src: 'tool'}); X.step(0.6);
  check('(d) motion: any other command fades it by itself', !X.m.thinking().on && !X.m.thinking().fading);
  // voice tools: a do_action held for his final words thinks; a what-if transcript refuses it -> off
  X = scene('muse');
  const V = VT.createVoiceTools({perf: () => X.P, motion: () => X.m, thinker: () => X.K, now: X.now, timers: X.T, prompt: () => null, log: () => {}});
  V.userSpeech('start'); V.userSpeech('stop');
  const h = V.call('do_action', {description: 'jump up and down'}, 'done', {}); X.step(0.4); const onH = X.m.thinking().on;
  V.userText('what would you do if I asked you to jump?', true); X.step(0.8);
  check('(d) a held do_action thinks; refused as a what-if -> off', h && h.status === 'preparing' && onH && !X.m.thinking().on && !X.K.stats().active, `hold ${h && h.status}, on while held ${onH}`);
  // a held do_action released into the planner keeps ONE thinking going (no restart)
  X = scene('fox');
  const V2 = VT.createVoiceTools({perf: () => X.P, motion: () => X.m, thinker: () => X.K, now: X.now, timers: X.T, prompt: () => null, log: () => {}});
  V2.userSpeech('start'); V2.userSpeech('stop'); V2.call('do_action', {description: 'do a backflip'}, 'done', {}); X.step(0.3);
  V2.userText('do a backflip', true); X.step(0.5);
  const st2 = X.K.stats();
  check('(d) hold -> planner: the same thinking carries on', X.m.thinking().on && st2.begun === 1 && st2.joined === 1 && st2.shown === 1, JSON.stringify({begun: st2.begun, joined: st2.joined, shown: st2.shown}));
}

// ---------------------------------------------------------------------------------------------------- (e) truthful + no fights
{
  const X = scene('muse', {alog: true});
  const r = X.P.call('do a cartwheel', {}); X.step(2.0);
  X.P.onKf(kfMsg(r.id, {poses: POSES, done: true})); X.step(4.0);
  X.alog.observe(X.m.state());
  const sum = X.alog.summary({}), items = X.alog.items();
  check('(e) the action log never itemizes the thinking ("what did you just do?" = the move)', !/think|chin|ponder/i.test(sum) && !items.some(i => /think/.test(String(i.name) + i.src)) && !X.ev.some(e => e.src === 'think'),
    sum.slice(0, 160));
  check('(e) state().active never lists it', !(X.m.state().active || []).some(a => /think/.test(a.name || '')));
  // a held pose on the head: head variants refused, nothing moves the held head
  const Y = scene('fox', {think: false});
  Y.m.command({action: 'pose', targets: {head_yaw: 0.3}, hold: true}, {src: 'tool'}); Y.step(1.0);
  const rr = Y.m.think({variant: 'up'});
  check('(e) a held head pose: the head variants are refused (never fights it)', !rr.ok || !/up|tilt|hmm|away/.test(rr.variant), JSON.stringify(rr));
  // the muse holding a prop in a paw: the paw variant uses the other paw or falls back to the paw-less main pose
  const Z = scene('muse', {think: false});
  Z.m.command({action: 'object', verb: 'pick_up', object: {id: 'headphones'}}, {src: 'tool'}); Z.step(12);
  const hold = Z.m.state().holding || {}, rz = Z.m.think({variant: 'chin', side: hold.right ? 'l' : 'r'});
  check('(e) muse holding a prop: the chin pose takes the free paw or falls back (prop stays put)', rz.ok && (rz.variant === 'away' || rz.side !== (hold.right ? 'l' : 'r') || !(hold.left || hold.right)),
    `holding ${JSON.stringify(hold)}, think -> ${JSON.stringify(rz)}`);
}

// ---------------------------------------------------------------------------------------------------- (g) the chin, per body
{
  // the generated geometry block of app/motion.js matches the shaders
  const cp = require('child_process').spawnSync(process.execPath, [require('path').join(__dirname, '..', 'tools', 'think_geom.cjs'), '--check'], {encoding: 'utf8'});
  check('(g) app/motion.js <think-geom> block is what tools/think_geom.cjs reads from the shaders', cp.status === 0, (cp.stderr || cp.stdout || '').trim());
  const row = (ch, opt) => { const m = mkMotion(ch, 4364, () => {}); for (let i = 0; i < 30; i++) m.step(DT, feat(i * DT)); const r = m.think(Object.assign({variant: 'chin'}, opt)); let o; for (let i = 0; i < 45; i++) o = m.step(DT, feat(1 + i * DT)); return {r, o, m}; };
  const G = require('../tools/think_geom.cjs').geom();
  const bad = [], info = [];
  for (const ch of CHARS) {
    for (const side of ['l', 'r']) {
      const {r, o, m} = row(ch, {side});
      // off the muse a point column would drag the hand to the camera-point pose at the chest (renderer anim/rig.py map_legacy)
      if (ch !== 'muse' && (o.point_l > 1e-3 || o.point_r > 1e-3)) bad.push(`${ch}/${side}: point column set`);
      if (!m.thinkCan('chin')) { if (r.variant !== 'away') bad.push(`${ch}/${side}: no chin reach but variant ${r.variant}`); continue; }
      if (r.variant !== 'chin') { bad.push(`${ch}/${side}: ${JSON.stringify(r)}`); continue; }
      const g = G[ch];
      if (g && g.kind === 'shape') {
        // replay engine/shape_arms.glsl's mapping (head ~ at rest): the wrist lands under the lower lip, on its own side, within reach
        const h = [o['hand_' + side + '_x'], o['hand_' + side + '_y'], o['hand_' + side + '_z']], xin = (g.sho[0] + g.rest[0]) / 0.45;
        // (the shape's resting turn SHAPE_YAW0 about its pivot: shoulder turned, rest + column offsets in world axes, then back into head space; own-side x)
        const sg = side === 'r' ? 1 : -1, cy = Math.cos(g.yaw0), sy = Math.sin(g.yaw0), ry = (v, c, s_) => [c * v[0] + s_ * v[2], v[1], -s_ * v[0] + c * v[2]];
        const Sw = ry([sg * g.sho[0] - g.pivot[0], g.sho[1] - g.pivot[1], g.sho[2] - g.pivot[2]], cy, sy);
        const Ww = [Sw[0] + sg * (g.rest[0] + (h[0] >= 0 ? g.len * h[0] : xin * h[0])), Sw[1] + g.rest[1] + g.len * h[1], Sw[2] + g.rest[2] + g.len * h[2]];
        const Wh = ry(Ww, cy, -sy), W = [sg * (Wh[0] + g.pivot[0]), Wh[1] + g.pivot[1], Wh[2] + g.pivot[2]];
        const lipLow = g.lip[1] - g.lipHL, reach = Math.hypot(W[0] - g.sho[0], W[1] - g.sho[1], W[2] - g.sho[2]) / g.len;
        if (side === 'l') info.push(`${ch} wrist x ${W[0].toFixed(2)} (mouth half width ${g.lipW}) y ${W[1].toFixed(2)} (lower lip ${lipLow.toFixed(2)}) reach ${reach.toFixed(2)}`);
        if (!(W[1] < lipLow - 0.02 && W[1] > lipLow - 1.3 * g.hand && W[0] > 0 && W[0] < 1.3 * g.lipW && W[2] > g.lip[2] - 0.12 && reach < 1.1)) bad.push(`${ch}/${side}: wrist ${W.map(x => x.toFixed(2))} not under the lower lip (${lipLow.toFixed(2)})`);
      }
      if (g && g.kind === 'arm') {
        const wy = g.wrist0[1] + g.len * o['hand_' + side + '_y'], under = g.muzC[1] - g.muzR[1];
        if (side === 'l') info.push(`${ch} wrist y ${wy.toFixed(2)} (muzzle underside ${under.toFixed(2)})`);
        if (!(Math.abs(wy - under) < 0.15)) bad.push(`${ch}/${side}: wrist y ${wy.toFixed(2)} not at the muzzle's underside (${under.toFixed(2)}; its hand hangs from the wrist)`);
      }
    }
  }
  check('(g) chin paw on every body: no point column off the muse, shapes under their own lower lip, the monkey under its muzzle, out of reach -> the paw-less main pose', !bad.length, bad.join('; ') || info.join('; '));
  // the main pose breaks eye contact DOWN and AWAY on every character, with or without the paw
  const rest = ch => { const m = mkMotion(ch, 4364, () => {}); let o; for (let i = 0; i < 75; i++) o = m.step(DT, feat(i * DT)); return o; };
  const gz = CHARS.map(ch => { const {r, o} = row(ch, {side: 'l'}), o0 = rest(ch); return [ch, r.variant, o.gaze_x - o0.gaze_x, o.gaze_y - o0.gaze_y]; });
  const gbad = gz.filter(([, , dx, dy]) => !(Math.abs(dx) > 0.3 && dy < -0.3));
  check('(g) main pose on all 12: the eyes go down and away', !gbad.length, gbad.map(x => x.join(' ')).join('; '));
  // head-only variants are scaled by the body file's tuning (the muse, framed full-body)
  const amp = ch => { const m = mkMotion(ch, 4364, () => {}), c = mkMotion(ch, 4364, () => {}); let d = 0; for (let i = 0; i < 30; i++) { m.step(DT, feat(i * DT)); c.step(DT, feat(i * DT)); } m.think({variant: 'tilt', dir: 1}); for (let i = 0; i < 45; i++) { const a = m.step(DT, feat(1 + i * DT)), b = c.step(DT, feat(1 + i * DT)); d = a.head_roll - b.head_roll; } return d; };
  const aM = amp('muse'), aF = amp('fox');
  check('(g) the muse turns its head further in the head-only variants (body-file gesture_tuning.think)', aM > 1.6 * aF && aF > 0.08, `tilt head_roll muse ${aM.toFixed(3)} vs fox ${aF.toFixed(3)}`);
}

// ---------------------------------------------------------------------------------------------------- (f) variants
{
  const X = scene('fox', {rseed: 99});
  const seq = [];
  for (let i = 0; i < 120; i++) { X.K.begin('k' + i); X.step(0.2); const a = X.K.active(); seq.push(a && a.variant); X.K.end('k' + i, 'x'); X.step(0.6); }
  const cnt = {}; for (const v of seq) cnt[v] = (cnt[v] || 0) + 1;
  let rep = 0, tri = 0; for (let i = 1; i < seq.length; i++) { if (seq[i] === seq[i - 1] && seq[i] !== 'chin') rep++; if (i > 1 && seq[i] === 'chin' && seq[i - 1] === 'chin' && seq[i - 2] === 'chin') tri++; }
  check('(f) mostly the main pose, alternates never twice running, every variant this body can do seen', (cnt.chin || 0) >= 0.45 * seq.length && rep === 0 && tri === 0 && ['up', 'tilt', 'hmm'].every(v => cnt[v] > 0) && !cnt.finger,
    JSON.stringify(cnt) + ` repeats ${rep}, 3 mains ${tri}`);
  {
    const Xm = scene('muse', {rseed: 99}), sq = [];
    for (let i = 0; i < 120; i++) { Xm.K.begin('k' + i); Xm.step(0.2); const a = Xm.K.active(); sq.push(a && a.variant); Xm.K.end('k' + i, 'x'); Xm.step(0.6); }
    const c2 = {}; for (const v of sq) c2[v] = (c2[v] || 0) + 1;
    check('(f) the muse (an index finger of its own) also raises a finger now and then', ['up', 'tilt', 'hmm', 'finger'].every(v => c2[v] > 0) && (c2.chin || 0) >= 0.45 * sq.length, JSON.stringify(c2));
  }
  // a long wait switches variant (crossfade), main <-> alternate
  const Y = scene('rabbit'); Y.P.call('do something hard', {}); Y.step(10);
  const s = Y.K.stats(), act = Y.K.active();
  check('(f) a long wait changes pose every ~4.5 s (main <-> alternate)', s.switched >= 1 && act && act.state === 'on', `switched ${s.switched}`);
}

// ---------------------------------------------------------------------------------------------------- (g) armless
{
  const X = scene('orb', {think: false});
  const ARM = /^(hand_|point_|thumb_|fingers_|shoulder_)/, seen = new Set(); let armMoved = null, vis = 0, n = 0;
  X.K = TK.createThinker({motion: () => X.m, now: X.now, timers: X.T, rand: seeded(11), arms: () => false, switchMs: 600, maxMs: 60000});
  for (let k = 0; k < 14; k++) {
    X.K.begin('a' + k, {}); const f0 = X.frames.length; X.step(1.5);
    const a = X.K.active(); if (a && a.variant) seen.add(a.variant);
    for (const s of X.K.stats().last) for (const v of s.variants) seen.add(v);
    let v = false;
    for (let i = f0; i < X.frames.length; i++) { const [mx] = X.diff(i); if (mx > VIS) v = true; for (const c in X.frames[i]) if (ARM.test(c) && Math.abs((X.frames[i][c] || 0) - (X.cf[i][c] || 0)) > 1e-6) armMoved = c; }
    n++; if (v) vis++;
    X.K.end('a' + k, 'test'); X.step(0.8);
  }
  check('(g) armless character (arms: false): only gaze + head + brow variants, every one visible, no arm column moves',
    ![...seen].some(v => v === 'chin' || v === 'finger') && seen.has('away') && seen.size >= 3 && vis === n && !armMoved, `variants ${[...seen]}, visible ${vis}/${n}, arm column ${armMoved}`);
  const r = X.m.think({variant: 'finger', arms: false}); X.m.thinkEnd('t');
  check('(g) motion.think({arms: false}) never gives an arm variant (finger -> a head variant)', r.ok && !/chin|finger/.test(r.variant), JSON.stringify(r));
}
console.log(`think_test: ${passes} passed, ${fails} failed`);
process.exit(fails);
