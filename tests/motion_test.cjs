// node tests/motion_test.cjs [outdir] — synthetic feature streams through app/motion.js; asserts + traces (+ PNGs via plot_traces.py)
'use strict';
const fs = require('fs'), path = require('path'), {execFileSync} = require('child_process');
const M = require('./load.cjs')('app/motion.js');
const OUT = process.argv[2] || path.join(path.join(require('os').tmpdir(), 'avatar-tests'), 'motion');
fs.mkdirSync(OUT, {recursive: true});
const FPS = 30, DT = 1 / FPS;
const results = [];
function check(name, ok, detail) { results.push({name, ok: !!ok, detail}); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  — ' + detail : '')); }

const BLEND0 = {mouthSmileLeft: 0.04, mouthSmileRight: 0.05, browInnerUp: 0.05, eyeWideLeft: 0.02, eyeWideRight: 0.02};
function face(x, y, o = {}) {
  return Object.assign({present: true, x, y, ax: Math.atan(x * M.TAN_HFOV2), ay: Math.atan(y * M.TAN_VFOV2), z: 0.45,
    yaw: 0, pitch: 0, roll: 0, lookX: 0, lookY: -0.14, blend: Object.assign({}, BLEND0, o.blend || {})}, o.face || {});
}
// run a scenario: script(t) → {features, events, userVoice, avatarVoice, commands}
function run(name, secs, script, seed = 7, track = []) {
  const m = M.createMotion({seed});
  const rows = [], rig = [], modes = [];
  let f = null;
  for (let i = 0; i < Math.round(secs * FPS); i++) {
    const t = (i + 1) * DT;
    const s = script(t, m) || {};
    if (s.features !== undefined) f = s.features;
    if (s.commands) for (const c of s.commands) m.command(c);
    if (s.events) for (const e of s.events) m.event(e);
    const o = m.step(DT, {features: f, userVoice: s.userVoice || null, avatarVoice: s.avatarVoice || null});
    rows.push(Object.assign({t}, o));
    modes.push(m.debug().mode);
    rig.push(JSON.stringify({type: 'rig', t: +t.toFixed(4), a: m.vector(o)}));
  }
  fs.writeFileSync(path.join(OUT, name + '.rig.jsonl'), JSON.stringify({rig_params: M.RIG_PARAMS}) + '\n' + rig.join('\n') + '\n');
  return {rows, modes, m, log: m.state.log};
}
const col = (rows, k) => rows.map(r => r[k]);
const at = (rows, t) => rows[Math.min(rows.length - 1, Math.max(0, Math.round(t * FPS) - 1))];
const firstT = (rows, pred, from = 0) => { for (const r of rows) if (r.t >= from && pred(r)) return r.t; return null; };
const std = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
const plots = [];
function plot(name, title, rows, series, marks = []) {
  const step = 1;
  plots.push({file: path.join(OUT, name + '.png'), title, t: rows.filter((_, i) => i % step === 0).map(r => +r.t.toFixed(3)),
    panels: series.map(p => ({label: p.label, lines: p.keys.map(k => ({name: k, y: rows.filter((_, i) => i % step === 0).map(r => +(r[k] ?? 0).toFixed(4))}))})), marks});
}
// smoothness: no param jumps more than its limit in one frame (blinks/gaze saccades/the commanded tongue excepted)
const JUMP = {head_yaw: 0.09, head_pitch: 0.08, head_roll: 0.06, body_yaw: 0.08, body_pitch: 0.12, body_lean: 0.08, smile: 0.1,
  hand_l_y: 0.2, hand_r_y: 0.2, hand_l_x: 0.2, hand_r_x: 0.2, ear_l: 0.3, ear_r: 0.3, breath: 0.15, tail_wag: 0.3, brow_l: 0.2, brow_r: 0.2};
function jumps(rows) {
  const bad = [];
  for (let i = 1; i < rows.length; i++) for (const k in JUMP) {
    const d = Math.abs(rows[i][k] - rows[i - 1][k]); if (d > JUMP[k]) bad.push(k + '@' + rows[i].t.toFixed(2) + ' Δ' + d.toFixed(3));
  }
  return bad;
}

// ---------------------------------------------------------------- 1. face moves left → right; a jump; eyes lead, head follows
{
  const sc = run('track', 12, t => {
    let x;
    if (t < 1) x = -0.6; else if (t < 4) x = -0.6 + 1.2 * (t - 1) / 3; else if (t < 7) x = 0.6; else if (t < 7.0 + DT) x = -0.5; else x = -0.5;
    return {features: {t, face: face(x, 0.1), attending: true, attention: 0.9, motion: 0.05, hands: []}};
  });
  const r = sc.rows;
  const g1 = at(r, 0.95).gaze_x, h1 = at(r, 0.95).head_yaw, g2 = at(r, 6.9).gaze_x, h2 = at(r, 6.9).head_yaw;
  check('track: he is screen-left → avatar looks screen-left (gaze_x<0, head_yaw<0)', g1 < -0.3 && h1 < -0.1, `gaze ${g1.toFixed(2)} head ${h1.toFixed(2)}`);
  check('track: he is screen-right → avatar looks screen-right', g2 > 0.3 && h2 > 0.1, `gaze ${g2.toFixed(2)} head ${h2.toFixed(2)}`);
  // step at t=7: x 0.6 → -0.5
  const gEnd = at(r, 9).gaze_x, hEnd = at(r, 9).head_yaw;
  const gHalf = firstT(r, q => q.gaze_x < (g2 + gEnd) / 2, 7), hHalf = firstT(r, q => q.head_yaw < (h2 + hEnd) / 2, 7);
  const gStart = firstT(r, q => q.gaze_x < g2 - 0.08, 7);
  check('track: eyes react 150–300 ms after he moves', gStart !== null && gStart - 7 >= 0.13 && gStart - 7 <= 0.33, `eye onset ${(gStart - 7).toFixed(3)} s`);
  check('track: eyes lead, head follows (head 50% ≥ 120 ms after eyes 50%)', gHalf !== null && hHalf !== null && hHalf - gHalf >= 0.12,
    `eyes50 ${(gHalf - 7).toFixed(2)} s, head50 ${(hHalf - 7).toFixed(2)} s`);
  const bad = jumps(r);
  check('track: no snapping', bad.length === 0, bad.slice(0, 4).join(', '));
  plot('track', 'face moves L→R (1–4 s), jumps R→L at 7 s: eyes lead, head follows', r,
    [{label: 'gaze', keys: ['gaze_x', 'gaze_y']}, {label: 'head (rad)', keys: ['head_yaw', 'head_pitch', 'head_roll']}, {label: 'body', keys: ['body_yaw']},
      {label: 'blink', keys: ['blink_l']}], [1, 4, 7]);
}

// ---------------------------------------------------------------- 2. he looks away, then back: mutual gaze returns
{
  const sc = run('lookback', 12, t => {
    const away = t >= 3 && t < 7;
    const ev = (t >= 7 && t < 7 + DT) ? [{type: 'look_back', t: 7, away_s: 4}] : (t >= 3 && t < 3 + DT ? [{type: 'attention_lost', t: 3}] : null);
    return {features: {t, face: face(0.1, 0.05, {face: {lookX: away ? 0.55 : 0, lookY: away ? 0.1 : -0.14, yaw: away ? 0.5 : 0}}),
      attending: !away, attention: away ? 0.05 : 0.9, motion: 0.05, hands: []}, events: ev};
  });
  const r = sc.rows, md = sc.modes;
  const back = md.slice(Math.round(7.05 * FPS), Math.round(7.6 * FPS));
  const userW = 0.5 * Math.atan(0.1 * M.TAN_HFOV2) / M.GAZE_RX;
  const tBack = firstT(r, q => Math.abs(q.gaze_x - userW) < 0.12, 7);
  check('lookback: mutual gaze after he looks back (mode user, eyes on him ≤ 0.6 s)', back.every(x => x === 'user') && tBack !== null && tBack - 7 < 0.6,
    `eyes on him at +${tBack !== null ? (tBack - 7).toFixed(2) : '–'} s; modes ${[...new Set(back)].join(',')}`);
  const awayModes = md.slice(Math.round(3 * FPS), Math.round(7 * FPS));
  check('lookback: while he is away the avatar does something other than stare (joint attention / glance) at least sometimes',
    awayModes.some(x => x !== 'user'), [...new Set(awayModes)].join(','));
  const flash = Math.max(...r.filter(q => q.t > 7 && q.t < 8).map(q => q.brow_l)) - at(r, 6.9).brow_l;
  check('lookback: eyebrow flash greets him', flash > 0.15, `brow +${flash.toFixed(2)}`);
  plot('lookback', 'he looks away 3–7 s (to his right), looks back at 7 s', r,
    [{label: 'gaze', keys: ['gaze_x', 'gaze_y']}, {label: 'head', keys: ['head_yaw', 'head_pitch']}, {label: 'brows', keys: ['brow_l', 'eye_wide']}], [3, 7]);
}

// ---------------------------------------------------------------- 3. wave → wave back within 300 ms
{
  const sc = run('wave', 7, t => {
    const ev = t >= 2 && t < 2 + DT ? [{type: 'wave', side: 'r', handed: 'right', t: 2}] : null;
    return {features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.2, hands: []}, events: ev};
  });
  const r = sc.rows, y0 = at(r, 1.95).hand_r_y;
  const tUp = firstT(r, q => q.hand_r_y > y0 + 0.05, 2);
  check('wave: wave-back visibly starts ≤ 300 ms after the event (hand_r_y +0.05)', tUp !== null && tUp - 2 <= 0.3, `+${tUp !== null ? Math.round((tUp - 2) * 1000) : '–'} ms`);
  const xs = r.filter(q => q.t > 2.8 && q.t < 3.8).map(q => q.hand_r_x);
  check('wave: paw oscillates (wave amplitude ≥ 0.12 peak-to-peak)', Math.max(...xs) - Math.min(...xs) >= 0.12, `p2p ${(Math.max(...xs) - Math.min(...xs)).toFixed(2)}`);
  check('wave: reaches WAVE_UP height (hand_r_y ≥ 0.9)', Math.max(...col(r, 'hand_r_y')) >= 0.9, `max ${Math.max(...col(r, 'hand_r_y')).toFixed(2)}`);
  check('wave: back to rest by the end', Math.abs(at(r, 7).hand_r_y) < 0.08, `end ${at(r, 7).hand_r_y.toFixed(3)}`);
  const bad = jumps(r); check('wave: no snapping', bad.length === 0, bad.slice(0, 4).join(', '));
  plot('wave', 'he waves (screen-right hand) at 2 s → avatar waves back with its screen-right paw', r,
    [{label: 'right paw xyz', keys: ['hand_r_x', 'hand_r_y', 'hand_r_z']}, {label: 'open/turn', keys: ['hand_r_open', 'hand_r_turn']}, {label: 'face', keys: ['smile', 'brow_l', 'ear_l']}], [2]);
}

// ---------------------------------------------------------------- 3b. wave false-positive guards
{
  const sc = run('wave_guard', 20, t => {
    const away = t >= 1 && t < 6;
    const ev = [];
    for (const w of [3, 8, 9.5, 11, 12.5, 14, 15.5, 17, 18.5]) if (t >= w && t < w + DT) ev.push({type: 'wave', side: 'l', t: w});
    return {features: {t, face: face(0, 0.05, {face: {lookX: away ? 0.6 : 0}}), attending: !away, attention: away ? 0.1 : 0.9, motion: 0.2, hands: []}, events: ev};
  });
  const r = sc.rows, waves = sc.log.filter(e => e.gesture === 'wave');
  check('wave guard: no wave back while he looks away', r.filter(q => q.t > 2.9 && q.t < 6).every(q => q.hand_l_y < 0.1), '');
  const upFrames = r.filter(q => q.hand_l_y > 0.5).length / FPS;
  check('wave guard: 10 s of continuous waving → one wave back (≤ 4.5 s raised), then smiles/nods', waves.length <= 1 && upFrames <= 4.5, `${waves.length} wave-backs, ${upFrames.toFixed(1)} s raised`);
}

// ---------------------------------------------------------------- 4. startle
{
  const sc = run('startle', 5, t => {
    const ev = t >= 1.5 && t < 1.5 + DT ? [{type: 'approach_fast', source: 'hand', rate: 2.5, t: 1.5}] : null;
    return {features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.1, hands: []}, events: ev};
  });
  const r = sc.rows;
  const tBlink = firstT(r, q => q.blink_l > 0.8, 1.5);
  const zMin = Math.min(...r.filter(q => q.t > 1.5 && q.t < 2.0).map(q => q.head_z)), tz = r.find(q => q.head_z === zMin).t;
  const ew = Math.max(...r.filter(q => q.t > 1.5 && q.t < 2.2).map(q => q.eye_wide));
  check('startle: blink within 120 ms', tBlink !== null && tBlink - 1.5 <= 0.12, `+${tBlink !== null ? Math.round((tBlink - 1.5) * 1000) : '–'} ms`);
  check('startle: head jerks back (head_z ≤ -0.04, peak ≤ 250 ms)', zMin <= -0.04 && tz - 1.5 <= 0.25, `head_z ${zMin.toFixed(3)} at +${Math.round((tz - 1.5) * 1000)} ms`);
  check('startle: eyes wide', ew > 0.6, ew.toFixed(2));
  const rec = r.filter(q => q.t > 2.8 && q.t < 3.2);
  check('startle: recovers by ~1.3 s', rec.every(q => Math.abs(q.head_z) < 0.015 && Math.abs(q.body_pitch) < 0.12), `head_z ${at(r, 3).head_z.toFixed(3)} body_pitch ${at(r, 3).body_pitch.toFixed(3)}`);
  plot('startle', 'quick hand toward the camera at 1.5 s: blink + head jerks back + eyes wide, recovers', r,
    [{label: 'head', keys: ['head_z', 'head_pitch']}, {label: 'body/shoulders', keys: ['body_pitch', 'shoulder_l']}, {label: 'eyes', keys: ['blink_l', 'eye_wide', 'brow_l']}, {label: 'ears', keys: ['ear_l', 'ear_r']}], [1.5]);
}

// ---------------------------------------------------------------- 5. sustained smile → mirrored with lag + attenuation
{
  const sc = run('smile', 12, t => {
    const s = t >= 2 && t < 8 ? 0.8 : 0.05;
    return {features: {t, face: face(0, 0.05, {blend: {mouthSmileLeft: s, mouthSmileRight: s, eyeSquintLeft: s * 0.4, eyeSquintRight: s * 0.4, cheekSquintLeft: s * 0.5, cheekSquintRight: s * 0.5}}),
      attending: true, attention: 0.9, motion: 0.05, hands: []}};
  });
  const r = sc.rows, base = at(r, 1.9).smile, top = Math.max(...r.filter(q => q.t > 5 && q.t < 8).map(q => q.smile));
  const half = firstT(r, q => q.smile > base + (top - base) / 2, 2);
  check('smile: mirrored (avatar smile rises ≥ 0.4)', top - base >= 0.4, `base ${base.toFixed(2)} → ${top.toFixed(2)}`);
  check('smile: with a lag (50% reached 0.6–1.6 s after he smiles), never instant', half !== null && half - 2 >= 0.6 && half - 2 <= 1.6 && at(r, 2.3).smile < base + 0.05,
    `50% at +${half !== null ? (half - 2).toFixed(2) : '–'} s; at +0.3 s ${(at(r, 2.3).smile - base).toFixed(3)}`);
  check('smile: attenuated (not a 1:1 copy)', top < 0.95, top.toFixed(2));
  check('smile: relaxes after he stops', at(r, 11.5).smile < base + 0.15, at(r, 11.5).smile.toFixed(2));
  { const a = std(r.filter(q => q.t > 4 && q.t < 8).map(q => q.tail_wag)), b = std(r.filter(q => q.t < 2).map(q => q.tail_wag)); check('smile: tail wags while happy', a > 1.5 * b && a > 0.15, `std ${b.toFixed(2)} → ${a.toFixed(2)}`); }
  plot('smile', 'he smiles 2–8 s (0.8): mirrored with ~1 s lag, attenuated', r,
    [{label: 'mouth', keys: ['smile', 'squint']}, {label: 'ears/tail', keys: ['ear_l', 'tail_wag']}, {label: 'brows', keys: ['brow_l']}], [2, 8]);
}

// ---------------------------------------------------------------- 6. 40 s of silence: boredom appears, varied
{
  const sc = run('silence', 45, t => ({features: {t, face: face(0.05, 0.05), attending: t < 15 || t > 30 ? true : false, attention: 0.8, motion: 0.01, hands: []}}), 11);
  const r = sc.rows, fid = sc.log.filter(e => e.fidget);
  const names = fid.map(e => e.fidget), distinct = new Set(names);
  let rep = 0; for (let i = 1; i < names.length; i++) if (names[i] === names[i - 1]) rep++;
  check('silence: boredom behaviours appear in 45 s (≥ 4)', fid.length >= 4, names.map((n, i) => n + '@' + fid[i].t.toFixed(0)).join(' '));
  check('silence: varied (≥ 4 distinct, no back-to-back repeats)', distinct.size >= 4 && rep === 0, `${distinct.size} distinct, ${rep} repeats`);
  const hy = col(r, 'head_yaw');
  check('silence: never a fixed pose (head moves: std ≥ 0.03 rad)', std(hy) >= 0.03, `std ${std(hy).toFixed(3)}`);
  const blinks = r.filter((q, i) => i && q.blink_l > 0.8 && r[i - 1].blink_l <= 0.8).length;
  check('silence: blink rate human (12–28/min)', blinks / 0.75 >= 12 && blinks / 0.75 <= 28, `${(blinks / 0.75).toFixed(1)}/min`);
  const bad = jumps(r); check('silence: no snapping', bad.length === 0, bad.slice(0, 4).join(', '));
  plot('silence', '45 s silence (he looks away 15–30 s): boredom builds, varied fidgets', r,
    [{label: 'head', keys: ['head_yaw', 'head_pitch', 'head_roll']}, {label: 'gaze', keys: ['gaze_x', 'gaze_y']}, {label: 'body', keys: ['body_lean', 'body_yaw', 'body_pitch']},
      {label: 'paws', keys: ['hand_l_y', 'hand_r_y', 'shoulder_l', 'mouth_open']}, {label: 'blink/ears', keys: ['blink_l', 'ear_l', 'tail_wag']}], fid.map(e => e.t));
}

// ---------------------------------------------------------------- 7. commands: hold + release, the rest keeps living
{
  const sc = run('commands', 16, t => {
    const c = [];
    const once = (a) => t >= a && t < a + DT;
    if (once(1)) c.push({action: 'pose', targets: {head_yaw: 0.4, hand_l_y: 0.9, hand_l_x: 0.25, hand_l_open: 1}, hold: true, ease_ms: 350});
    if (once(6)) c.push({action: 'release', parts: ['head', 'hand_l']});
    if (once(8)) c.push({action: 'look', target: 'left', hold: true});
    if (once(11)) c.push({action: 'release', parts: 'all'});
    if (once(12.5)) c.push({action: 'gesture', name: 'nod', side: 'right', repeat: 1});
    if (once(13.5)) c.push({action: 'expression', name: 'surprised', intensity: 0.8});
    return {features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []}, commands: c};
  });
  const r = sc.rows;
  const held = r.filter(q => q.t > 2.0 && q.t < 6);
  check('commands: pose eases in (no frame jump > 0.06 rad)', r.slice(29, 45).every((q, i, a) => !i || Math.abs(q.head_yaw - a[i - 1].head_yaw) < 0.06), '');
  check('commands: held exactly while held', held.every(q => Math.abs(q.head_yaw - 0.4) < 1e-6 && Math.abs(q.hand_l_y - 0.9) < 0.01), `${held[0].head_yaw.toFixed(3)} … ${held[held.length - 1].head_yaw.toFixed(3)}`);
  check('commands: the rest keeps living during the hold (brow/breath/gaze move)', std(held.map(q => q.breath)) > 0.05 && std(held.map(q => q.brow_l)) > 0.01 && held.some(q => q.blink_l > 0.8),
    `breath std ${std(held.map(q => q.breath)).toFixed(3)}, brow std ${std(held.map(q => q.brow_l)).toFixed(3)}`);
  const rel = r.filter(q => q.t > 6 && q.t < 6.6);
  check('commands: release eases out (≥ 250 ms, no jump)', rel.every((q, i, a) => !i || Math.abs(q.head_yaw - a[i - 1].head_yaw) < 0.06) && at(r, 6.2).head_yaw > 0.15, `at +0.2 s ${at(r, 6.2).head_yaw.toFixed(3)}`);
  check('commands: after release the head is free again', Math.abs(at(r, 7.5).head_yaw) < 0.2, at(r, 7.5).head_yaw.toFixed(3));
  const lk = r.filter(q => q.t > 9 && q.t < 11);
  check("commands: 'look left' (character's own left = screen-right) holds", lk.every(q => q.gaze_x > 0.6 && q.head_yaw > 0.15), `gaze ${lk[0].gaze_x.toFixed(2)} head ${lk[0].head_yaw.toFixed(2)}`);
  check('commands: look released → back to him', Math.abs(at(r, 12.3).gaze_x) < 0.3, at(r, 12.3).gaze_x.toFixed(2));
  const np = r.filter(q => q.t > 12.5 && q.t < 13.4).map(q => q.head_pitch);
  check('commands: nod gesture visible', Math.max(...np) - at(r, 12.45).head_pitch > 0.06, `+${(Math.max(...np) - at(r, 12.45).head_pitch).toFixed(3)}`);
  check('commands: expression eases in and auto-releases', Math.max(...r.filter(q => q.t > 13.8 && q.t < 15.5).map(q => q.eye_wide)) > 0.6 && at(r, 16).eye_wide < 0.2, '');
  const bad = jumps(r); check('commands: no snapping', bad.length === 0, bad.slice(0, 4).join(', '));
  plot('commands', 'pose hold 1→6 s (head_yaw .4 + paw up) · look left hold 8→11 s · nod 12.5 · surprised 13.5', r,
    [{label: 'head', keys: ['head_yaw', 'head_pitch']}, {label: 'paw', keys: ['hand_l_y', 'hand_l_x', 'hand_l_open']}, {label: 'gaze', keys: ['gaze_x']},
      {label: 'living', keys: ['breath', 'brow_l', 'blink_l', 'eye_wide']}], [1, 6, 8, 11, 12.5, 13.5]);
}

// ---------------------------------------------------------------- 8. speaking: prosody-driven head motion, never fixed
function syllables(t, t0, t1, seed = 3) {
  if (t < t0 || t > t1) return {rms: 0.0, voiced: false};
  const u = t - t0, phrase = Math.floor(u / 2.6), pu = u % 2.6;
  if (pu > 2.1) return {rms: 0.001, voiced: false};   // pauses between phrases
  const syl = Math.floor(pu * 4.3), sp = pu * 4.3 - syl;
  const stress = ((syl * 7 + phrase * 3 + seed) % 5 === 0) ? 1.0 : 0.45;
  const env = Math.sin(Math.PI * Math.min(1, sp / 0.8)) ** 1.5;
  return {rms: 0.06 * stress * env + 0.004, voiced: env > 0.15, pitch: 180 * (1 + 0.15 * stress * env)};
}
{
  const sc = run('speaking', 18, t => ({features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []},
    avatarVoice: syllables(t, 4, 14)}));
  const r = sc.rows, sp = r.filter(q => q.t > 5 && q.t < 14), qu = r.filter(q => q.t > 0.5 && q.t < 4);
  const vel = a => { const v = []; for (let i = 1; i < a.length; i++) v.push(Math.abs(a[i] - a[i - 1]) * FPS); return v; };
  const hv = vel(sp.map(q => q.head_pitch)), mvSp = std(sp.map(q => q.head_pitch)) + std(sp.map(q => q.head_yaw)) + std(sp.map(q => q.head_roll));
  const mvQu = std(qu.map(q => q.head_pitch)) + std(qu.map(q => q.head_yaw)) + std(qu.map(q => q.head_roll));
  check('speaking: more head motion while talking than idle', mvSp > 1.3 * mvQu, `speak ${mvSp.toFixed(3)} vs idle ${mvQu.toFixed(3)}`);
  check('speaking: never a fixed pose (head moving > 0.01 rad/s in ≥ 90% of frames)', hv.filter(v => v > 0.01).length / hv.length >= 0.9, (hv.filter(v => v > 0.01).length / hv.length).toFixed(2));
  const avertFrames = sc.modes.slice(5 * FPS, 14 * FPS).filter(x => x === 'avert').length / (9 * FPS);
  check('speaking: natural gaze aversion while talking (10–50% of the time)', avertFrames > 0.1 && avertFrames < 0.5, (avertFrames * 100).toFixed(0) + '%');
  const bl = r.filter((q, i) => i && q.t > 4 && q.t < 14 && q.blink_l > 0.8 && r[i - 1].blink_l <= 0.8).length * 6;
  check('speaking: blink rate higher while talking (≥ 15/min)', bl >= 15, bl + '/min');
  check('speaking: beat gestures with a paw', sp.some(q => q.hand_l_y > 0.15 || q.hand_r_y > 0.15), `max paw y ${Math.max(...sp.map(q => Math.max(q.hand_l_y, q.hand_r_y))).toFixed(2)}`);
  const bad = jumps(r); check('speaking: no snapping', bad.length === 0, bad.slice(0, 4).join(', '));
  const rows2 = r.map(q => Object.assign({}, q, {voice_rms: syllables(q.t, 4, 14).rms * 10}));
  plot('speaking', 'avatar speaks 4–14 s (synthetic syllables, stressed beats): nods, tilts, brow raises, aversion, beats', rows2,
    [{label: 'voice ×10', keys: ['voice_rms']}, {label: 'head', keys: ['head_pitch', 'head_yaw', 'head_roll']}, {label: 'brows/gaze', keys: ['brow_l', 'gaze_x', 'gaze_y']},
      {label: 'paws', keys: ['hand_l_y', 'hand_r_y']}, {label: 'blink', keys: ['blink_l']}], [4, 14]);
}

// ---------------------------------------------------------------- 9. listening: backchannel nods at his pauses
{
  const sc = run('listening', 20, t => ({features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []},
    userVoice: syllables(t, 2, 18, 1)}), 5);
  const r = sc.rows, nods = sc.log.filter(e => e.gesture === 'nod' && e.src === 'listen');
  check('listening: backchannel nods at his pauses (≥ 2 in 16 s)', nods.length >= 2, nods.map(e => e.t.toFixed(1)).join(' '));
  const rolls = r.filter(q => q.t > 4 && q.t < 18).map(q => q.head_roll);
  check('listening: head tilts while listening (|roll| mean ≥ 0.03)', rolls.reduce((a, b) => a + Math.abs(b), 0) / rolls.length >= 0.03, (rolls.reduce((a, b) => a + Math.abs(b), 0) / rolls.length).toFixed(3));
  check('listening: ears turn toward his voice', r.filter(q => q.t > 4 && q.t < 18).reduce((a, q) => a + q.ear_l_turn, 0) / (14 * FPS) < -0.1, '');
  const rows2 = r.map(q => Object.assign({}, q, {user_rms: syllables(q.t, 2, 18, 1).rms * 10}));
  plot('listening', 'he talks 2–18 s with pauses: backchannel nods at pauses, tilt, ears forward', rows2,
    [{label: 'his voice ×10', keys: ['user_rms']}, {label: 'head', keys: ['head_pitch', 'head_roll', 'head_yaw']}, {label: 'ears', keys: ['ear_l_turn', 'ear_l']}], nods.map(e => e.t));
}

// ---------------------------------------------------------------- 10. pointing + object + no face
{
  const sc = run('point', 10, t => {
    const ev = t >= 2 && t < 2 + DT ? [{type: 'point', side: 'r', dir: {x: 1, y: 0.2}, at: 'right', t: 2}] :
      t >= 6 && t < 6 + DT ? [{type: 'object_up', side: 'l', x: -0.3, y: 0.0, t: 6}] : null;
    return {features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []}, events: ev};
  });
  const r = sc.rows;
  check('point: he points to his right (screen-right) → avatar looks screen-right', at(r, 2.8).gaze_x > 0.6 && at(r, 2.8).head_yaw > 0.15, `gaze ${at(r, 2.8).gaze_x.toFixed(2)} head ${at(r, 2.8).head_yaw.toFixed(2)}`);
  check('point: then back to him', Math.abs(at(r, 5.5).gaze_x) < 0.3, at(r, 5.5).gaze_x.toFixed(2));
  check('object: looks at the object (screen-left, curious)', at(r, 6.8).gaze_x < -0.25 && at(r, 6.8).eye_wide > 0.1, `gaze ${at(r, 6.8).gaze_x.toFixed(2)} wide ${at(r, 6.8).eye_wide.toFixed(2)}`);
  plot('point', 'he points to his right at 2 s; holds an object up screen-left at 6 s', r, [{label: 'gaze', keys: ['gaze_x', 'gaze_y']}, {label: 'head', keys: ['head_yaw', 'head_roll']}, {label: 'ears', keys: ['ear_l_turn', 'ear_r_turn', 'eye_wide']}], [2, 6]);
}
{
  const sc = run('noface', 20, t => ({features: t < 4 || t > 14 ? {t, face: face(0.2, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []} : {t, face: {present: false}, hands: [], motion: 0},
    events: t >= 14 && t < 14 + DT ? [{type: 'face_appear', t: 14}] : null}));
  const r = sc.rows, gz = r.filter(q => q.t > 6 && q.t < 14).map(q => q.gaze_x);
  check('noface: wanders while nobody is there', std(gz) > 0.15, `gaze std ${std(gz).toFixed(2)}`);
  const ears = Math.max(...r.filter(q => q.t > 14 && q.t < 15.5).map(q => q.ear_l)) - at(r, 13.9).ear_l;
  check('noface: perks up when he appears (ears up, eyes widen)', ears > 0.2 && Math.max(...r.filter(q => q.t > 14 && q.t < 15.5).map(q => q.eye_wide)) > 0.2, `ears +${ears.toFixed(2)}`);
  plot('noface', 'face gone 4–14 s (wander), reappears at 14 s (perk up)', r, [{label: 'gaze', keys: ['gaze_x', 'gaze_y']}, {label: 'ears/eyes', keys: ['ear_l', 'eye_wide', 'ear_l_turn']}, {label: 'head', keys: ['head_yaw']}], [4, 14]);
}

// ---------------------------------------------------------------- postural mirroring + personal space
{
  const sc = run('posture', 14, t => {
    const roll = t >= 2 && t < 7 ? 0.25 : 0, z = t >= 8 && t < 11 ? 0.22 : 0.45;
    return {features: {t, face: face(0, 0.05, {face: {roll, z}}), attending: true, attention: 0.9, motion: 0.05, hands: []}};
  });
  const r = sc.rows, avgRoll = (a, b) => { const q = r.filter(x => x.t > a && x.t < b); return q.reduce((s, x) => s + x.head_roll, 0) / q.length; };
  check('posture: mirrors his head tilt slowly and partly (same screen side)', avgRoll(4.5, 7) - avgRoll(0.5, 2) > 0.05 && at(r, 2.3).head_roll - at(r, 1.9).head_roll < 0.04,
    `roll ${avgRoll(0.5, 2).toFixed(3)} → ${avgRoll(4.5, 7).toFixed(3)}`);
  check('posture: leans back when he comes very close, then returns', at(r, 10.5).body_pitch < at(r, 7.5).body_pitch - 0.15 && Math.abs(at(r, 13.9).body_pitch - at(r, 7.5).body_pitch) < 0.12,
    `body_pitch ${at(r, 7.5).body_pitch.toFixed(2)} → ${at(r, 10.5).body_pitch.toFixed(2)} → ${at(r, 13.9).body_pitch.toFixed(2)}`);
  plot('posture', 'he tilts his head 2–7 s (mirrored slowly), comes close 8–11 s (lean back)', r, [{label: 'head', keys: ['head_roll', 'head_yaw']}, {label: 'body', keys: ['body_pitch']}], [2, 7, 8, 11]);
}

// ---------------------------------------------------------------- salient motion: a fast hand catches the eye
{
  let glances = 0;
  for (let seed = 1; seed <= 6; seed++) {
    const sc = run('handglance' + seed, 6, t => {
      const moving = t > 2 && t < 2.6;
      return {features: {t, face: face(0, 0.05), attending: true, attention: 0.9, motion: 0.2,
        hands: moving ? [{side: 'l', x: -0.5, y: -0.2, vx: -1.8, vy: 0.4, main: true, age: 0.05, gesture: 'None'}] : []}};
    }, seed);
    if (sc.modes.slice(Math.round(2 * 30), Math.round(3.6 * 30)).includes('hand')) glances++;
  }
  check('salient motion: a fast hand sometimes draws a glance (1–5 of 6 seeds)', glances >= 1 && glances <= 5, `${glances}/6`);
}

// ---------------------------------------------------------------- "hold that / freeze": the moving pose freezes until release
{
  const sc = run('freeze', 9, t => {
    const c = [], once = a => t >= a && t < a + DT;
    if (once(1)) c.push({action: 'gesture', name: 'wave', side: 'right', repeat: 3});
    if (once(1.9)) c.push({action: 'hold', parts: 'all'});
    if (once(6)) c.push({action: 'release', parts: 'all'});
    return {features: {t, face: face(0.1, 0.05), attending: true, attention: 0.9, motion: 0.05, hands: []}, commands: c};
  });
  const r = sc.rows, held = r.filter(q => q.t > 2.0 && q.t < 6);
  const fr = at(r, 1.9), still = k => held.every(q => Math.abs(q[k] - held[0][k]) < 1e-6);
  check('freeze: the waving paw and the head stop exactly where they were', still('hand_l_y') && still('hand_l_x') && still('head_yaw') && still('head_pitch') && Math.abs(held[0].hand_l_y - fr.hand_l_y) < 0.05 && held[0].hand_l_y > 0.6,
    `paw y ${fr.hand_l_y.toFixed(2)} → ${held[0].hand_l_y.toFixed(2)} … ${held[held.length - 1].hand_l_y.toFixed(2)}`);
  check('freeze: eyes, blinks and breathing keep living', held.some(q => q.blink_l > 0.8) && std(held.map(q => q.breath)) > 0.05, '');
  check('freeze: release eases back to rest (no jump)', at(r, 8.5).hand_l_y < 0.1 && r.filter(q => q.t > 6 && q.t < 7).every((q, i, a) => !i || Math.abs(q.hand_l_y - a[i - 1].hand_l_y) < 0.2), `paw y at 8.5 s ${at(r, 8.5).hand_l_y.toFixed(2)}`);
  plot('freeze', 'wave at 1 s · "hold that" at 1.9 s (frozen mid-wave) · release at 6 s', r, [{label: 'paw', keys: ['hand_l_y', 'hand_l_x']}, {label: 'head', keys: ['head_yaw', 'head_pitch']}, {label: 'living', keys: ['blink_l', 'breath', 'gaze_x']}], [1, 1.9, 6]);
}
// ---------------------------------------------------------------- presets: default identical, calm < default < lively
{
  const amp = {};
  for (const ps of ['calm', 'default', 'lively']) {
    const m = M.createMotion({seed: 21, preset: ps}); const ys = [], bl = [];
    let beats = 0, prevBeat = 0;
    for (let i = 0; i < 40 * FPS; i++) { const t = (i + 1) * DT; const o = m.step(DT, {features: {t, face: face(0, 0.05), attending: true, hands: []}, avatarVoice: syllables(t, 2, 38)}); ys.push(o.head_yaw); bl.push(o.body_lean);
      const b = Math.max(o.hand_l_y, o.hand_r_y) > 0.15 ? 1 : 0; if (b && !prevBeat) beats++; prevBeat = b; }
    amp[ps] = {head: std(ys), body: std(bl), beats};
  }
  check('presets: calm < default < lively (speaking head motion, body sway, beat gestures)', amp.calm.head < amp.default.head && amp.default.head < amp.lively.head && amp.lively.beats >= amp.default.beats && amp.calm.body < amp.lively.body,
    Object.entries(amp).map(([k, v]) => `${k}: head ${v.head.toFixed(3)} body ${v.body.toFixed(3)} beats ${v.beats}`).join(' · '));
}

// ---------------------------------------------------------------- negative emotions: clear, lagged, distinct (c2)
{
  const EMO = {
    sad: {mouthFrownLeft: 0.6, mouthFrownRight: 0.6, browInnerUp: 0.6},
    fear: {eyeWideLeft: 0.7, eyeWideRight: 0.7, browInnerUp: 0.6, browDownLeft: 0.3, browDownRight: 0.3, mouthStretchLeft: 0.5, mouthStretchRight: 0.5},
    surprise: {eyeWideLeft: 0.8, eyeWideRight: 0.8, browInnerUp: 0.7, browOuterUpLeft: 0.7, browOuterUpRight: 0.7, jawOpen: 0.5},
  };
  const res = {};
  for (const [name, bl] of Object.entries(EMO)) {
    const sc = run('emo_' + name, 7, t => ({features: {t, face: face(0, 0.05, {blend: t >= 2 && t < 5.5 ? bl : {}}), attending: true, attention: 0.9, motion: 0.05, hands: []}}), 4);
    const r = sc.rows, b = at(r, 1.9), p = r.filter(q => q.t > 3.2 && q.t < 5.3), mean = k => p.reduce((a, q) => a + q[k], 0) / p.length;
    res[name] = {early: Math.abs(at(r, 2.3).eye_wide - b.eye_wide) + Math.abs(at(r, 2.3).ear_l - b.ear_l), ew: mean('eye_wide') - b.eye_wide, ear: mean('ear_l') - b.ear_l,
      knit: mean('brow_knit') - b.brow_knit, pitch: mean('head_pitch') - b.head_pitch, brow: mean('brow_l') - b.brow_l, bp: mean('body_pitch') - b.body_pitch, mouth: mean('mouth_open') - b.mouth_open};
  }
  const f2 = o => Object.entries(o).map(([k, v]) => k + ' ' + v.toFixed(2)).join(', ');
  check('emotion: sad → ears droop, brows knit, head lowers', res.sad.ear < -0.15 && res.sad.knit > 0.15 && res.sad.pitch > 0.02, f2(res.sad));
  check('emotion: fear → eyes wide, ears back, brows knit, leans back', res.fear.ew > 0.25 && res.fear.ear < -0.15 && res.fear.knit > 0.15 && res.fear.bp < -0.05, f2(res.fear));
  check('emotion: surprise → eyes wide, brows up, ears up, mouth opens', res.surprise.ew > 0.3 && res.surprise.brow > 0.2 && res.surprise.ear > 0.1 && res.surprise.mouth > 0.05, f2(res.surprise));
  check('emotion: still lagged (nothing visible 0.3 s after his change)', Object.values(res).every(v => v.early < 0.06), Object.entries(res).map(([k, v]) => k + ' ' + v.early.toFixed(3)).join(', '));
}

// ---------------------------------------------------------------- his face → perception's tracker → avatar: human-like latency
{
  const PER = require('./load.cjs')('app/perception.js');
  const lat = {};
  const cases = {
    smile: {blend: {mouthSmileLeft: 0.6, mouthSmileRight: 0.6, eyeSquintLeft: 0.2, eyeSquintRight: 0.2}, out: o => o.smile},
    surprise: {blend: {browInnerUp: 0.5, browOuterUpLeft: 0.5, browOuterUpRight: 0.5, eyeWideLeft: 0.4, eyeWideRight: 0.4, jawOpen: 0.4}, out: o => o.eye_wide},
    frown: {blend: {browDownLeft: 0.5, browDownRight: 0.5, mouthPressLeft: 0.3, mouthPressRight: 0.3}, out: o => o.brow_knit},
    tilt: {roll: 0.2, out: o => o.head_roll},
  };
  for (const [name, c] of Object.entries(cases)) {
    const tr = PER.createExpressionTracker(), m = M.createMotion({seed: 6}), rows = [];
    for (let i = 0; i < 7 * FPS; i++) {
      const t = (i + 1) * DT, on = t >= 3 && t < 6;
      const fc = Object.assign(face(0, 0.05, {blend: on && c.blend ? c.blend : {}}), {roll: on && c.roll ? c.roll : 0});
      if (i % 2 === 0) { const r = tr.update(fc, t); fc.xp = {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt}; this.lastXp = fc.xp; } else fc.xp = this.lastXp;
      rows.push(Object.assign({t}, m.step(DT, {features: {t, face: fc, attending: true, attention: 0.9, motion: 0.05, hands: []}})));
    }
    const b = c.out(at(rows, 2.9)), pk = Math.max(...rows.filter(q => q.t > 3 && q.t < 6).map(c.out)) - b;
    const t50 = firstT(rows, q => c.out(q) - b >= 0.5 * pk, 3);
    lat[name] = {lat: t50 === null ? null : t50 - 3, amp: pk};
  }
  // a nod of his (pitch swing) → a nod back
  for (const sd of [6, 7, 8, 9, 10]) { if (lat.nod && lat.nod.lat !== null) break;   // it answers ~75% of his nods
    const tr = PER.createExpressionTracker(), m = M.createMotion({seed: sd}); let nodAt = null, backAt = null;
    for (let i = 0; i < 7 * FPS; i++) { const t = (i + 1) * DT, ph = t - 3, pitch = ph > 0 && ph < 0.6 ? 0.09 * Math.sin(Math.PI * ph / 0.3) ** 2 * (ph < 0.3 ? 1 : 1) : 0;
      const fc = Object.assign(face(0, 0.05), {pitch}); const r = tr.update(fc, t); fc.xp = {state: r.state, intensity: r.intensity, scores: r.scores, tilt: r.tilt};
      for (const e of r.events) if (e.type === 'nod' && nodAt === null) nodAt = t;
      m.step(DT, {features: {t, face: fc, attending: true, attention: 0.9, motion: 0.05, hands: []}, events: r.events});
      if (backAt === null && m.state.log.some(e => e.nodback)) backAt = t; }
    lat.nod = {lat: backAt === null ? null : backAt - 3, amp: nodAt === null ? 0 : 1};
  }
  const txt = Object.entries(lat).map(([k, v]) => `${k} ${v.lat === null ? '–' : v.lat.toFixed(2) + ' s'} (amp ${v.amp.toFixed(2)})`).join(' · ');
  check('his face → avatar: smile/surprise/frown/tilt answered in 0.3–0.8 s (50% of the response), visibly', ['smile', 'surprise', 'frown', 'tilt'].every(k => lat[k].lat !== null && lat[k].lat >= 0.3 && lat[k].lat <= 0.8) &&
    lat.smile.amp > 0.3 && lat.surprise.amp > 0.3 && lat.frown.amp > 0.15 && lat.tilt.amp > 0.04, txt);
  check('his nod → a nod back (0.3–1.2 s after his nod starts)', lat.nod.lat !== null && lat.nod.lat >= 0.3 && lat.nod.lat <= 1.2, txt);
}

// ---------------------------------------------------------------- every gesture & expression runs, eases, and ends
{
  const m = M.createMotion({seed: 2}); const errs = [];
  for (const g of m.gestures) {
    try { m.command({action: 'gesture', name: g, side: 'right', repeat: 1}); for (let i = 0; i < 230; i++) m.step(DT, {}); if (m.debug().gestures.some(x => x.endsWith(':cmd'))) errs.push(g + ' did not end'); }
    catch (e) { errs.push(g + ': ' + e.message); }
  }
  for (const e of m.expressions) { try { m.command({action: 'expression', name: e, intensity: 0.8}); for (let i = 0; i < 120; i++) m.step(DT, {}); } catch (x) { errs.push(e + ': ' + x.message); } }
  check(`library: ${m.gestures.length} gestures + ${m.expressions.length} expressions run and end`, errs.length === 0, errs.join('; '));
}

// ---------------------------------------------------------------- prior plug-in (a stand-in with prior.js's interface)
{
  const frames = [];
  const fake = {step(fr) { frames.push(fr); const k = frames.length; return {head_yaw: 0.15 * Math.sin(k / 30 * 2 * Math.PI * 0.3), head_pitch: 0, head_roll: 0,
    gaze_x: 0, gaze_y: 0, blink_l: k % 90 < 3 ? 1 : 0, blink_r: k % 90 < 3 ? 1 : 0, brow_l: 0, brow_r: 0, smile: 0.2, squint: 0, eye_wide: 0}; }};
  const m = M.createMotion({seed: 4, prior: fake}); const ys = [], bl = [];
  for (let i = 0; i < 300; i++) { const o = m.step(DT, {features: {t: i * DT, face: face(0, 0.05), attending: true, hands: []}, avatarVoice: {rms: 0.02}}); ys.push(o.head_yaw); bl.push(o.blink_l); }
  const fr = frames[frames.length - 1];
  const okFrame = fr && fr.self && fr.partner && fr.face && fr.face.present && typeof fr.face.yaw === 'number';
  const blinks = bl.filter((v, i) => i && v > 0.8 && bl[i - 1] <= 0.8).length;
  check('prior: frames in prior.js shape, its head motion drives the free head motion, its blinks trigger ours', okFrame && Math.max(...ys) - Math.min(...ys) > 0.2 && blinks >= 3,
    `yaw p2p ${(Math.max(...ys) - Math.min(...ys)).toFixed(2)}, blinks ${blinks}`);
  const bad = M.createMotion({seed: 4, prior: {step() { throw new Error('boom'); }}}); let ok = true;
  try { for (let i = 0; i < 60; i++) bad.step(DT, {}); } catch (e) { ok = false; }
  check('prior: a failing prior falls back to procedural (no throw)', ok && bad.debug().priorErr, bad.debug().priorErr);
}

fs.writeFileSync(path.join(OUT, 'plots.json'), JSON.stringify(plots));
try { execFileSync('python3', [path.join(__dirname, 'plot_traces.py'), path.join(OUT, 'plots.json')], {stdio: 'inherit'}); } catch (e) { console.log('plot failed', e.message); }
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 1));
const nf = results.filter(r => !r.ok).length;
console.log(`\n${results.length - nf}/${results.length} passed → ${OUT}`);
process.exitCode = nf ? 1 : 0;
