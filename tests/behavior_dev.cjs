// tests/behavior_dev.cjs — program layer DEV set: 25 standing instructions (written by the builder, varied over sense
// cues × acts × timing × counting / state × stop conditions × characters) + 2 live edits, each with a scripted user and a
// behaviour check over the dry-run trace. Used by tools/behavior_bench.py (coder model choice) through
// tools/behavior_check.cjs op 'dev'. NOT the sealed test set (never read here).
//   node tests/behavior_dev.cjs --list                 the items
//   node tests/behavior_dev.cjs --selftest             reference programs (hand-written) must pass every check
'use strict';
const SIM = require('../app/behavior_sim.cjs');

// ---- trace helpers (trace = [{t, sets: {control: [v, w]}}], t = runtime clock; the program starts at t≈0)
const series = (r, name) => (r.trace || []).map(x => [x.t, x.sets && x.sets[name] ? x.sets[name][0] : null]);
const win = (r, name, t0, t1) => series(r, name).filter(([t]) => t >= t0 && t <= t1);
const vals = (r, name, t0, t1) => win(r, name, t0, t1).map(([, v]) => v).filter(v => v !== null);
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
const range = a => a.length ? Math.max(...a) - Math.min(...a) : 0;
const avgIn = (r, name, t0, t1) => { const w = win(r, name, t0, t1); if (!w.length) return null; return mean(w.map(([, v]) => v === null ? 0 : v)); };
const setFrac = (r, name, t0, t1) => { const w = win(r, name, t0, t1); return w.length ? w.filter(([, v]) => v !== null).length / w.length : 0; };
const rangeIn = (r, name, t0, t1) => range(vals(r, name, t0, t1));
const maxOf = (r, names, t0, t1, f = Math.max) => { const a = []; for (const n of [].concat(names)) a.push(...vals(r, n, t0, t1)); return a.length ? f(...a) : null; };
// "quiet": the program leaves the control alone (not set) or holds it near its resting value / still
const quiet = (r, name, t0, t1, tol = 0.15, rest = 0) => { const v = vals(r, name, t0, t1); return !v.length || (range(v) < tol && Math.abs(mean(v) - rest) < tol + 0.1); };
const emits = (r, t0 = -1, t1 = 1e9) => (r.events || []).filter(e => e.type === 'rule_moment' && e.t >= t0 && e.t <= t1);
const acts = (r, t0 = -1, t1 = 1e9) => (r.actions || []).filter(a => a.t >= t0 && a.t <= t1);
const zc = (r, name, t0, t1) => { const v = vals(r, name, t0, t1); const m = mean(v) || 0; let n = 0; for (let i = 1; i < v.length; i++) if ((v[i - 1] - m) * (v[i] - m) < 0) n++; return n / Math.max(1e-3, t1 - t0); };
const doneAt = r => { const e = (r.events || []).find(x => x.type === 'behavior' && x.phase === 'end'); return e ? e.t : null; };
const lastSetT = r => { let t = null; for (const x of r.trace || []) if (x.sets && Object.keys(x.sets).length) t = x.t; return t; };
const has = (s, re) => re.test(JSON.stringify(s || ''));
const R = (ok, detail) => ({ok: !!ok, detail});
const f2 = v => v === null || v === undefined ? 'null' : (+v).toFixed(2);
const blinkAt = ts => ts.map(t => [t, t + 0.15, {blink: 1}]);

const DEV = [
  {id: 'd01_blink_nose', character: 'fox', instruction: 'Every time I blink, wiggle your nose.',
    script: {dur: 14, at: blinkAt([2, 6, 10])},
    check: r => { const a = [2, 6, 10].map(t => rangeIn(r, 'nose_wiggle', t, t + 1.0)); const q = rangeIn(r, 'nose_wiggle', 4.0, 5.8) + rangeIn(r, 'nose_wiggle', 12.5, 14);
      return R(a.every(x => x > 0.3) && q < 0.15, `wiggle ${a.map(f2)} quiet ${f2(q)}`); }},
  {id: 'd02_hand_paw_side', character: 'bear', instruction: 'Whenever I raise my right hand, raise your paw on the same side of the screen.',
    script: {dur: 13, at: [[3, 6, {hand_right: {x: 0.4, y: 0.45, open: 1}}], [9, 11, {hand_right: {x: 0.45, y: 0.5, open: 1}}]]},
    check: r => { const up1 = avgIn(r, 'left_paw_y', 3.6, 6), up2 = avgIn(r, 'left_paw_y', 9.6, 11), wrong = maxOf(r, 'right_paw_y', 3, 11);
      const dn = maxOf(r, 'left_paw_y', 7.2, 8.6) ?? 0;
      return R(up1 > 0.55 && up2 > 0.55 && dn < 0.35 && (wrong === null || wrong < 0.3), `left paw up ${f2(up1)} ${f2(up2)} between ${f2(dn)} right paw max ${f2(wrong)}`); }},
  {id: 'd03_mirror_tilt_timed', character: 'monkey', instruction: 'Mirror my head tilt for the next 20 seconds.',
    script: {dur: 26, at: [[2, 5, {roll: 0.3}], [7, 10, {roll: -0.3}], [22, 25, {roll: 0.3}]]},
    check: r => { const a = avgIn(r, 'head_roll', 3, 5), b = avgIn(r, 'head_roll', 8, 10), late = maxOf(r, 'head_roll', 22.5, 25, (...x) => Math.max(...x.map(Math.abs)));
      return R(a > 0.12 && b < -0.12 && (late === null || late < 0.08), `tilt ${f2(a)} / ${f2(b)} after 20 s ${f2(late)} done ${f2(doneAt(r))}`); }},
  {id: 'd04_count_nods', character: 'bear', instruction: 'Count how many times I nod, and when I reach five nods let me know.',
    script: {dur: 14, events: [2, 4, 6, 8, 10, 12].map(t => [t, 'nod', {amp: 0.1}])},
    check: r => { const e = emits(r); return R(e.length === 1 && e[0].t >= 9.9 && e[0].t <= 11.5 && has(e[0].data, /5|five/i), `emits ${e.map(x => f2(x.t) + JSON.stringify(x.data)).join(' ')}`); }},
  {id: 'd05_eyes_on_hands', character: 'fox', instruction: "Keep your eyes on my hands whenever they're in view.",
    script: {dur: 12, at: [[2, 5, {hand_right: {x: 0.6, y: 0.2, open: 1}}], [7, 10, {hand_left: {x: -0.6, y: -0.2, open: 1}}]]},
    check: r => { const a = avgIn(r, 'gaze_x', 3, 5), b = avgIn(r, 'gaze_x', 8, 10), z = maxOf(r, 'gaze_x', 11, 12, (...x) => Math.max(...x.map(Math.abs)));
      return R(a > 0.2 && b < -0.2 && (z === null || z < 0.35), `gaze ${f2(a)} / ${f2(b)} no hands ${f2(z)}`); }},
  {id: 'd06_look_where_he_looks', character: 'rabbit', instruction: "When I look away from the screen, look the same way I'm looking.",
    script: {dur: 14, at: [[3, 6, {attending: false, lookX: 0.5, yaw: 0.4}], [9, 12, {attending: false, lookX: -0.5, yaw: -0.4}]]},
    check: r => { const d = (t0, t1) => (avgIn(r, 'gaze_x', t0, t1) || 0) + (avgIn(r, 'head_yaw', t0, t1) || 0);
      const a = d(4, 6), b = d(10, 12), q = d(7.2, 8.6);
      return R(a > 0.15 && b < -0.15 && Math.abs(q) < 0.2, `look ${f2(a)} / ${f2(b)} back ${f2(q)}`); }},
  {id: 'd07_smile_frown', character: 'fox', instruction: 'If I smile, smile back big; if I frown, droop your ears.',
    script: {dur: 13, at: [[2, 5, {expr: {smile: 0.85}}], [8, 11, {expr: {browDown: 0.7, sad: 0.4}, xp: {state: 'frown', scores: {frown: 0.8}}}]]},
    check: r => { const s = avgIn(r, 'smile', 3, 5), ea = Math.min(avgIn(r, 'right_ear', 9, 11) ?? 1, avgIn(r, 'left_ear', 9, 11) ?? 1), s2 = maxOf(r, 'smile', 9, 11);
      return R(s > 0.6 && ea < -0.2 && (s2 === null || s2 < 0.5), `smile ${f2(s)} ears ${f2(ea)} smile while frown ${f2(s2)}`); }},
  {id: 'd08_tail_loudness', character: 'fox', instruction: 'Wag your tail faster the louder I talk.',
    script: {dur: 13, at: [[2, 6, {speaking: true, level: 0.2}], [8, 12, {speaking: true, level: 0.9}]]},
    check: r => { const lo = zc(r, 'tail_wag', 2.5, 6), hi = zc(r, 'tail_wag', 8.5, 12);
      return R(hi > 1 && hi > lo * 1.3 && rangeIn(r, 'tail_wag', 8.5, 12) > 0.3, `crossings/s quiet ${f2(lo)} loud ${f2(hi)}`); }},
  {id: 'd09_touch_nose_ears', character: 'fox', instruction: 'Every time I touch my nose, wiggle your ears for two seconds.',
    script: {dur: 12, at: [[2, 2.6, {hand_right: {touch: 'nose', open: 0.5}}], [7, 7.6, {hand_right: {touch: 'nose', open: 0.5}}]]},
    check: r => { const e = t => Math.max(rangeIn(r, 'right_ear', t, t + 1.4), rangeIn(r, 'left_ear', t, t + 1.4));
      const q = Math.max(rangeIn(r, 'right_ear', 5, 6.5), rangeIn(r, 'left_ear', 5, 6.5), rangeIn(r, 'right_ear', 10.6, 12));
      return R(e(2.4) > 0.4 && e(7.4) > 0.4 && q < 0.15, `wiggle ${f2(e(2.4))} ${f2(e(7.4))} quiet ${f2(q)}`); }},
  {id: 'd10_cover_eyes', character: 'hippo', instruction: 'Close your eyes whenever I cover my eyes with my hands.',
    script: {dur: 10, at: [[3, 6, {hand_left: {touch: 'left_eye', open: 0.8}, hand_right: {touch: 'right_eye', open: 0.8}}]]},
    check: r => { const c = Math.max(avgIn(r, 'eyes_closed', 3.6, 6) ?? 0, Math.min(avgIn(r, 'right_eye_closed', 3.6, 6) ?? 0, avgIn(r, 'left_eye_closed', 3.6, 6) ?? 0));
      const o = Math.max(maxOf(r, 'eyes_closed', 7.5, 10) ?? 0, maxOf(r, 'right_eye_closed', 7.5, 10) ?? 0);
      return R(c > 0.7 && o < 0.3, `closed ${f2(c)} after ${f2(o)}`); }},
  {id: 'd11_mouth_tongue', character: 'fox', instruction: 'When I open my mouth wide, open yours too, and if I keep it open for more than two seconds, stick your tongue out.',
    script: {dur: 12, at: [[2, 3, {expr: {jawOpen: 0.8}}], [6, 10, {expr: {jawOpen: 0.8}}]]},
    check: r => { const m = avgIn(r, 'mouth_open', 2.4, 3), t1 = maxOf(r, 'tongue_out', 2, 3.5), t2 = avgIn(r, 'tongue_out', 8.5, 10);
      return R(m > 0.4 && (t1 === null || t1 < 0.2) && t2 > 0.4, `mouth ${f2(m)} tongue early ${f2(t1)} late ${f2(t2)}`); }},
  {id: 'd12_lean_distance', character: 'orb', instruction: 'Lean toward me when I come closer to the camera and lean back when I move away.',
    script: {dur: 12, at: [[3, 6, {z: 0.3}], [8, 11, {z: 0.8}]]},
    check: r => { const a = avgIn(r, 'body_pitch', 4.2, 6), b = avgIn(r, 'body_pitch', 9.2, 11);
      return R(a > 0.15 && b < -0.15, `lean ${f2(a)} / ${f2(b)}`); }},
  {id: 'd13_hands_together_bow', character: 'cube', instruction: 'Whenever I put my hands together, bow your head.',
    script: {dur: 11, at: [[3, 5, {hand_right: {x: 0.05, y: -0.5, open: 1}, hand_left: {x: -0.05, y: -0.5, open: 1}}], [8, 9.5, {hand_right: {x: 0.05, y: -0.5, open: 1}, hand_left: {x: -0.05, y: -0.5, open: 1}}]]},
    check: r => { const a = maxOf(r, 'head_pitch', 3, 5.5, Math.min), b = maxOf(r, 'head_pitch', 8, 10, Math.min), act = acts(r, 3, 5.5).filter(x => /bow/i.test(x.description)).length;
      const q = maxOf(r, 'head_pitch', 6.4, 7.6, Math.min);
      return R(((a !== null && a < -0.12 && b !== null && b < -0.12) || act > 0) && (q === null || q > -0.08), `pitch ${f2(a)} ${f2(b)} between ${f2(q)} bow actions ${act}`); }},
  {id: 'd14_thumbs_jump', character: 'muse', instruction: 'Every time I give you a thumbs up, do a little jump.',
    script: {dur: 11, at: [[3, 4, {hand_right: {x: 0.3, y: 0.0, open: 0.1, gesture: 'Thumb_Up'}}], [8, 9, {hand_right: {x: 0.3, y: 0.0, open: 0.1, gesture: 'Thumb_Up'}}]],
      events: [[3.2, 'thumbs_up', {side: 'r', handed: 'right'}], [8.2, 'thumbs_up', {side: 'r', handed: 'right'}]]},
    check: r => { const a = acts(r).filter(x => /jump|hop/i.test(x.description)); const b = rangeIn(r, 'body_bounce', 3, 5) > 0.2 && rangeIn(r, 'body_bounce', 8, 10) > 0.2;
      return R((a.length === 2 && a[0].t >= 3 && a[0].t < 4.6 && a[1].t >= 8 && a[1].t < 9.6) || b, `actions ${a.map(x => f2(x.t) + ' ' + x.description).join(' | ')}`); }},
  {id: 'd15_count_blinks_timer', character: 'fox', instruction: 'Count my blinks for 15 seconds, then tell me how many I did.',
    script: {dur: 19, at: blinkAt([1, 3, 5, 7, 9, 11, 13, 16.5])},
    check: r => { const e = emits(r); return R(e.length === 1 && e[0].t >= 14.8 && e[0].t <= 16.3 && has(e[0].data, /\b7\b|seven/i), `emits ${e.map(x => f2(x.t) + JSON.stringify(x.data)).join(' ')}`); }},
  {id: 'd16_wave_back_side', character: 'star', instruction: 'When I wave, wave back with the hand on the same side of the screen.',
    script: {dur: 9, at: [[3, 5, {hand_right: {x: 0.4, y: 0.4, open: 1}}]], events: [[3.5, 'wave', {side: 'r', handed: 'right'}], [4.2, 'wave', {side: 'r', handed: 'right', continuing: true}]]},
    check: r => { const p = maxOf(r, 'left_paw_y', 3.5, 5.5), w = rangeIn(r, 'left_paw_x', 3.5, 5.5) + rangeIn(r, 'left_paw_turn', 3.5, 5.5), wr = maxOf(r, 'right_paw_y', 3, 6);
      const act = acts(r, 3, 6).filter(x => /wave/i.test(x.description) && /left/i.test(x.description)).length;
      return R((p > 0.6 && w > 0.15 && (wr === null || wr < 0.3)) || act > 0, `left paw max ${f2(p)} swing ${f2(w)} right paw ${f2(wr)} actions ${act}`); }},
  {id: 'd17_nod_while_talking', character: 'fox', instruction: "Nod slowly along while I'm talking, and stay still while you're talking.",
    script: {dur: 14, at: [[2, 7, {speaking: true, level: 0.6}], [9, 13, {self_speaking: true}]]},
    check: r => { const a = rangeIn(r, 'head_pitch', 3, 7), b = rangeIn(r, 'head_pitch', 9.5, 13), z = zc(r, 'head_pitch', 3, 7);
      return R(a > 0.08 && b < 0.05 && z < 6, `nod range talking ${f2(a)} (${f2(z)} crossings/s) while it talks ${f2(b)}`); }},
  {id: 'd18_look_for_him', character: 'bear', instruction: 'If I disappear from view for more than three seconds, look around for me until I come back.',
    script: {dur: 13, at: [[3, 10, {face: false}]]},
    check: r => { const early = Math.max(rangeIn(r, 'head_yaw', 3, 5.6), rangeIn(r, 'gaze_x', 3, 5.6)), mid = Math.max(rangeIn(r, 'head_yaw', 7, 10), rangeIn(r, 'gaze_x', 7, 10));
      const back = Math.max(rangeIn(r, 'head_yaw', 11.5, 13), rangeIn(r, 'gaze_x', 11.5, 13));
      return R(early < 0.1 && mid > 0.3 && back < 0.2, `early ${f2(early)} searching ${f2(mid)} after he is back ${f2(back)}`); }},
  {id: 'd19_laugh_bounce', character: 'monkey', instruction: 'Every time I laugh, bounce in your seat.',
    script: {dur: 11, at: [[3, 4.5, {expr: {smile: 0.9, jawOpen: 0.6}, speaking: true, level: 0.8}], [5.5, 7, {speaking: true, level: 0.6}], [8, 9.5, {expr: {smile: 0.9, jawOpen: 0.6}, speaking: true, level: 0.8}]]},
    check: r => { const a = rangeIn(r, 'body_bounce', 3.2, 5), b = rangeIn(r, 'body_bounce', 8.2, 10), q = rangeIn(r, 'body_bounce', 5.8, 7);
      return R(a > 0.2 && b > 0.2 && q < 0.1, `bounce ${f2(a)} ${f2(b)} talking only ${f2(q)}`); }},
  {id: 'd20_point_at_hand', character: 'fox', instruction: 'Point at my hand whenever I hold it up.',
    script: {dur: 9, at: [[3, 6, {hand_right: {x: 0.5, y: 0.5, open: 1}}]]},
    check: r => { const p = Math.max(avgIn(r, 'left_point', 3.6, 6) ?? 0, avgIn(r, 'right_point', 3.6, 6) ?? 0), y = Math.max(avgIn(r, 'left_paw_y', 3.6, 6) ?? 0, avgIn(r, 'right_paw_y', 3.6, 6) ?? 0);
      const after = Math.max(maxOf(r, 'left_point', 7.2, 9) ?? 0, maxOf(r, 'right_point', 7.2, 9) ?? 0);
      return R(p > 0.5 && y > 0.3 && after < 0.3, `point ${f2(p)} paw ${f2(y)} after ${f2(after)}`); }},
  {id: 'd21_squint_lean_in', character: 'rabbit', instruction: 'Squint at me suspiciously whenever I lean in close.',
    script: {dur: 10, at: [[4, 7, {z: 0.28}]]},
    check: r => { const a = avgIn(r, 'squint', 4.6, 7), b = maxOf(r, 'squint', 1, 3.5) ?? 0, c = maxOf(r, 'squint', 8.5, 10) ?? 0;
      return R(a > 0.4 && b < 0.2 && c < 0.2, `squint ${f2(a)} before ${f2(b)} after ${f2(c)}`); }},
  {id: 'd22_anti_mirror_minute', character: 'muse', instruction: 'For the next minute, every time I turn my head to one side, turn yours the other way.',
    script: {dur: 66, at: [[3, 5, {yaw: 0.4}], [7, 9, {yaw: -0.4}], [62, 64, {yaw: 0.4}]]},
    check: r => { const a = avgIn(r, 'head_yaw', 3.6, 5), b = avgIn(r, 'head_yaw', 7.6, 9), late = maxOf(r, 'head_yaw', 62.5, 64, Math.min);
      return R(a < -0.12 && b > 0.12 && (late === null || late > -0.05), `turn ${f2(a)} / ${f2(b)} after the minute ${f2(late)} done ${f2(doneAt(r))}`); }},
  {id: 'd23_ears_follow_hands', character: 'fox', instruction: 'Make your ears follow my hands: left ear up when my left hand is up, right ear up when my right hand is up.',
    script: {dur: 10, at: [[2, 4, {hand_left: {x: -0.4, y: 0.45, open: 1}}], [6, 8, {hand_right: {x: 0.4, y: 0.45, open: 1}}]]},
    check: r => { const L1 = avgIn(r, 'left_ear', 2.6, 4) ?? 0, R1 = avgIn(r, 'right_ear', 2.6, 4) ?? 0, L2 = avgIn(r, 'left_ear', 6.6, 8) ?? 0, R2 = avgIn(r, 'right_ear', 6.6, 8) ?? 0;
      const lit = L1 - R1 > 0.3 && R2 - L2 > 0.3, mir = R1 - L1 > 0.3 && L2 - R2 > 0.3;
      return R(lit || mir, `left hand: L ${f2(L1)} R ${f2(R1)}; right hand: L ${f2(L2)} R ${f2(R2)}`); }},
  {id: 'd24_word_bounce', character: 'drop', instruction: "Whenever I say 'bounce', bounce up and down three times.",
    script: {dur: 13, events: [[3, 'said', {text: 'okay bounce now'}], [9, 'said', {text: 'bounce'}]]},
    check: r => { const a = rangeIn(r, 'body_bounce', 3, 5.2), b = rangeIn(r, 'body_bounce', 9, 11.2), q = rangeIn(r, 'body_bounce', 6.5, 8.5);
      const z = zc(r, 'body_bounce', 3, 5.5);
      return R(a > 0.2 && b > 0.2 && q < 0.08, `bounce ${f2(a)} ${f2(b)} quiet ${f2(q)} (${f2(z)} crossings/s)`); }},
  {id: 'd25_peace_fingers', character: 'muse', instruction: 'When I hold up a peace sign, hold up two fingers back.',
    script: {dur: 9, at: [[3, 6, {hand_right: {x: 0.35, y: 0.35, open: 0.5, fingers: 2, gesture: 'Victory'}}]]},
    check: r => { const f = Math.max(avgIn(r, 'left_fingers', 3.6, 6) ?? 0, avgIn(r, 'right_fingers', 3.6, 6) ?? 0), y = Math.max(avgIn(r, 'left_paw_y', 3.6, 6) ?? 0, avgIn(r, 'right_paw_y', 3.6, 6) ?? 0);
      const after = Math.max(maxOf(r, 'left_fingers', 7.5, 9) ?? 0, maxOf(r, 'right_fingers', 7.5, 9) ?? 0);
      return R(Math.abs(f - 2) < 0.3 && y > 0.3 && after < 1, `fingers ${f2(f)} paw ${f2(y)} after ${f2(after)}`); }},
];
// live edits: the program from `base` is changed by `edit`; the check runs on the edited program (setup reruns with its state)
const EDITS = [
  {id: 'e01_one_ear_longer', base: 'd09_touch_nose_ears', instruction: 'Only wiggle your left ear, and make it last four seconds.',
    script: {dur: 14, at: [[2, 2.6, {hand_right: {touch: 'nose', open: 0.5}}], [9, 9.6, {hand_right: {touch: 'nose', open: 0.5}}]]},
    check: r => { const l = rangeIn(r, 'left_ear', 5, 6.2), rr = rangeIn(r, 'right_ear', 2, 7), q = rangeIn(r, 'left_ear', 7.2, 8.8), l2 = rangeIn(r, 'left_ear', 9.4, 10.5);
      return R(l > 0.4 && rr < 0.1 && q < 0.15 && l2 > 0.4, `left ear at 5-6 s ${f2(l)}, right ear ${f2(rr)}, quiet ${f2(q)}, again ${f2(l2)}`); }},
  {id: 'e02_ten_nods', base: 'd04_count_nods', instruction: 'Make it ten nods instead of five.',
    script: {dur: 24, events: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22].map(t => [t, 'nod', {amp: 0.1}])},
    check: r => { const e = emits(r); return R(e.length === 1 && e[0].t >= 19.9 && e[0].t <= 21.5 && has(e[0].data, /10|ten/i), `emits ${e.map(x => f2(x.t) + JSON.stringify(x.data)).join(' ')}`); }},
];
const ALL = DEV.concat(EDITS);
const byId = id => ALL.find(x => x.id === id);

function runItem(item, code) {
  const r = SIM.dryRun(code, {character: item.character || (byId(item.base) || {}).character || 'fox', script: item.script, fps: 30, trace: true, instruction: item.instruction});
  let res;
  try { res = r.id ? item.check(r) : R(false, 'did not start: ' + r.error); } catch (e) { res = R(false, 'check error: ' + String(e && e.message || e)); }
  return {item: item.id, ok: res.ok && !r.errors.length, detail: res.detail + (r.errors.length ? ' | errors: ' + r.errors.join('; ').slice(0, 200) : ''), warnings: r.warnings, stats: r.stats};
}

// reference programs (hand-written) — the self-test proves every check is passable and fair
const REF = {
  d01_blink_nose: `({ setup(api) { api.claim(['nose']); }, per_frame(api) { const s = api.state; if (api.events('blink').length) s.t0 = api.t; if (s.t0 !== undefined && api.t - s.t0 < 0.8) api.set('nose_wiggle', 0.7 * Math.sin(2 * Math.PI * 6 * (api.t - s.t0))); } })`,
  d02_hand_paw_side: `({ setup(api) { api.claim(['left_arm']); }, per_frame(api) { const h = api.user.hands.right; if (h.present && h.raised) { api.set('left_paw_y', 0.95); api.set('left_paw_x', 0.25); } } })`,
  d03_mirror_tilt_timed: `({ setup(api) { api.claim(['head']); }, per_frame(api) { if (api.t > 20) return api.done('time'); if (api.user.present) api.set('head_roll', api.user.head.roll); } })`,
  d04_count_nods: `({ setup(api) { api.state.n = api.state.n || 0; api.state.goal = 5; }, per_frame(api) { for (const e of api.events('nod')) { api.state.n++; if (api.state.n === api.state.goal) api.emit('nods_reached', {count: api.state.n}); } } })`,
  d05_eyes_on_hands: `({ per_frame(api) { const H = [api.user.hands.left, api.user.hands.right].filter(h => h.present); if (H.length) api.look('hands'); } })`,
  d06_look_where_he_looks: `({ per_frame(api) { if (api.user.present && !api.user.gaze.atCamera) api.look('gaze'); } })`,
  d07_smile_frown: `({ setup(api) { api.claim(['mouth', 'ears']); }, per_frame(api) { const u = api.user; if (u.face('smile') > 0.4) api.set('smile', 1); if (u.face('frown') > 0.4) { api.set('right_ear', -0.8); api.set('left_ear', -0.8); } } })`,
  d08_tail_loudness: `({ setup(api) { api.claim(['tail']); api.state.ph = 0; }, per_frame(api) { const lv = api.user.speaking ? api.user.level : 0; api.state.ph += api.dt * 2 * Math.PI * (0.5 + 4 * lv); api.set('tail_wag', 0.8 * Math.sin(api.state.ph)); } })`,
  d09_touch_nose_ears: `({ setup(api) { api.claim(['ears']); api.state.until = -1; }, per_frame(api) { for (const e of api.events('touch')) if (e.part === 'nose') api.state.until = api.t + 2; if (api.t < api.state.until) { const v = 0.7 * Math.sin(2 * Math.PI * 4 * api.t); api.set('right_ear', v); api.set('left_ear', -v); } } })`,
  d10_cover_eyes: `({ setup(api) { api.claim(['eyes']); }, per_frame(api) { if (api.user.touching('eyes')) api.set('eyes_closed', 1); } })`,
  d11_mouth_tongue: `({ setup(api) { api.claim(['mouth']); }, per_frame(api) { const s = api.state, open = api.user.face('mouth_open') > 0.5; if (open && !s.was) s.t0 = api.t; s.was = open; if (open) { api.set('mouth_open', 0.7); if (api.t - s.t0 > 2) api.set('tongue_out', 1); } } })`,
  d12_lean_distance: `({ setup(api) { api.claim(['body']); }, per_frame(api) { const s = api.state, d = api.user.distance; if (!d) return; s.base = s.base || d; const k = (s.base - d) / s.base; s.v = (s.v || 0) + (Math.max(-1, Math.min(1, 2 * k)) - (s.v || 0)) * Math.min(1, api.dt * 5); api.set('body_pitch', s.v); } })`,
  d13_hands_together_bow: `({ setup(api) { api.claim(['head']); }, per_frame(api) { if (api.gesture('hands_together').on) api.set('head_pitch', -0.35); } })`,
  d14_thumbs_jump: `({ per_frame(api) { for (const e of api.events(['thumbs_up'])) api.do_action('do a little jump'); } })`,
  d15_count_blinks_timer: `({ setup(api) { api.state.n = 0; }, per_frame(api) { if (api.t <= 15) api.state.n += api.events('blink').length; else { api.emit('blink_count', {count: api.state.n}); api.done('told'); } } })`,
  d16_wave_back_side: `({ setup(api) { api.claim(['left_arm']); }, per_frame(api) { const s = api.state; if (api.events('wave').length) s.until = api.t + 1.5; if (api.t < (s.until || 0)) { api.set('left_paw_y', 0.95); api.set('left_paw_x', 0.25 + 0.2 * Math.sin(2 * Math.PI * 2 * api.t)); api.set('left_paw_open', 1); } } })`,
  d17_nod_while_talking: `({ setup(api) { api.claim(['head']); }, per_frame(api) { if (api.user.speaking && !api.self.speaking) api.set('head_pitch', -0.08 - 0.08 * Math.sin(2 * Math.PI * 0.8 * api.t)); } })`,
  d18_look_for_him: `({ setup(api) { api.claim(['head']); api.state.gone = null; }, per_frame(api) { const s = api.state; if (api.user.present) { s.gone = null; return; } if (s.gone === null) s.gone = api.t; if (api.t - s.gone > 3) { api.set('head_yaw', 0.45 * Math.sin(2 * Math.PI * 0.3 * (api.t - s.gone))); } } })`,
  d19_laugh_bounce: `({ setup(api) { api.claim(['body']); }, per_frame(api) { const u = api.user, laugh = u.face('smile') > 0.5 && u.speaking; const s = api.state; if (laugh) s.until = api.t + 0.8; if (api.t < (s.until || 0)) api.set('body_bounce', 0.5 * Math.abs(Math.sin(2 * Math.PI * 2 * api.t))); } })`,
  d20_point_at_hand: `({ setup(api) { api.claim(['left_arm']); }, per_frame(api) { const h = api.user.hands.right; if (h.present && h.raised) { api.paw_to('left', h.x, h.y); api.set('left_point', 1); } } })`,
  d21_squint_lean_in: `({ setup(api) { api.claim(['eyes']); }, per_frame(api) { const s = api.state, d = api.user.distance; if (!d) return; s.base = s.base || d; if (d < 0.75 * s.base) api.set('squint', 0.7); } })`,
  d22_anti_mirror_minute: `({ setup(api) { api.claim(['head']); }, per_frame(api) { if (api.t > 60) return api.done('minute'); const y = api.user.head.yaw; if (Math.abs(y) > 0.15) api.set('head_yaw', -Math.sign(y) * 0.35); } })`,
  d23_ears_follow_hands: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { const H = api.user.hands; api.set('left_ear', H.left.present && H.left.raised ? 0.9 : 0); api.set('right_ear', H.right.present && H.right.raised ? 0.9 : 0); } })`,
  d24_word_bounce: `({ setup(api) { api.claim(['body']); }, per_frame(api) { const s = api.state; if (api.heard('bounce')) s.t0 = api.t; if (s.t0 !== undefined && api.t - s.t0 < 1.5) api.set('body_bounce', 0.6 * Math.abs(Math.sin(Math.PI * 2 * (api.t - s.t0)))); } })`,
  d25_peace_fingers: `({ setup(api) { api.claim(['left_arm']); }, per_frame(api) { const h = api.user.hands.right; if (h.present && h.gesture === 'victory') { api.set('left_paw_y', 0.7); api.set('left_fingers', 2); } } })`,
  e01_one_ear_longer: `({ setup(api) { api.claim(['ears']); api.state.until = -1; }, per_frame(api) { for (const e of api.events('touch')) if (e.part === 'nose') api.state.until = api.t + 4; if (api.t < api.state.until) api.set('left_ear', 0.7 * Math.sin(2 * Math.PI * 4 * api.t)); } })`,
  e02_ten_nods: `({ setup(api) { api.state.n = api.state.n || 0; api.state.goal = 10; }, per_frame(api) { for (const e of api.events('nod')) { api.state.n++; if (api.state.n === api.state.goal) api.emit('nods_reached', {count: api.state.n}); } } })`,
};

module.exports = {DEV, EDITS, ALL, byId, runItem, REF};

if (require.main === module) {
  if (process.argv.includes('--list')) { for (const x of ALL) console.log(x.id.padEnd(26), (x.character || '(edit)').padEnd(8), x.instruction); process.exit(0); }
  let bad = 0;
  for (const x of ALL) { const r = runItem(x, REF[x.id]); if (!r.ok) bad++; console.log((r.ok ? 'PASS ' : 'FAIL ') + x.id.padEnd(26) + r.detail); }
  // the checks also reject wrong programs: a do-nothing program fails every item
  let fp = 0;
  for (const x of ALL) { const r = runItem(x, `({ per_frame(api) {} })`); if (r.ok) { fp++; console.log('NULL-PASS ' + x.id); } }
  console.log(`dev self-test: ${ALL.length - bad}/${ALL.length} reference programs pass; a do-nothing program passes ${fp}`);
  process.exit(bad + fp);
}
