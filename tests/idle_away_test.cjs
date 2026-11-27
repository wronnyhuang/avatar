// idle while the user looks away (look-away / yawn / head-scratch style idles start somewhat
// sooner and come a bit more often): the look-away idle timing of app/motion.js (IDLE_AWAY_ONSET / IDLE_AWAY_RATE, switch
// idle2 / ?idle2=0), old vs new on seeded synthetic timelines (looking at the screen 20 s, away 60 s, back 20 s).
//   node tests/idle_away_test.cjs        (the full before/after tables + real sessions: tests/idle_away_replay.cjs)
'use strict';
const L = require('./idle_away_lib.cjs');
const M = L.load('app/motion.js');
let fails = 0;
const check = (name, ok, detail) => { if (!ok) fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined ? '  — ' + detail : '')); };
const SEEDS = +(process.env.IDLE_SEEDS || 20);
const many = (tl, idle2, n = SEEDS) => { const r = []; for (let s = 1; s <= n; s++) r.push(L.run(M, tl, {seed: s, idle2})); return r; };

// the pair of named constants, in the owner's "moderate" band
check('constants: IDLE_AWAY_ONSET shortens the wait (0.4 .. 0.75)', M.IDLE_AWAY_ONSET >= 0.4 && M.IDLE_AWAY_ONSET <= 0.75, M.IDLE_AWAY_ONSET);
check('constants: IDLE_AWAY_RATE is a bit more often (1.25 .. 1.55)', M.IDLE_AWAY_RATE >= 1.25 && M.IDLE_AWAY_RATE <= 1.55, M.IDLE_AWAY_RATE);

// the switch: default on; createMotion({idle2: false}) and the page URL's ?idle2=0 = the old timing
check('switch: on by default', M.createMotion({seed: 1}).debug().idle2 === true);
check('switch: createMotion({idle2: false}) is off', M.createMotion({seed: 1, idle2: false}).debug().idle2 === false);
{ const had = 'location' in globalThis, was = globalThis.location;
  try { globalThis.location = {search: '?k=x&idle2=0'}; check('switch: ?idle2=0 is off', M.createMotion({seed: 1}).debug().idle2 === false);
    globalThis.location = {search: '?k=x'}; check('switch: no flag is on', M.createMotion({seed: 1}).debug().idle2 === true); }
  finally { if (had) globalThis.location = was; else delete globalThis.location; } }

const [talkThenAway, silent, awayTalking] = L.SYNTH();
const quiet = {old: [...many(talkThenAway, false), ...many(silent, false)], new: [...many(talkThenAway, true), ...many(silent, true)]};
const qo = L.measure(quiet.old), qn = L.measure(quiet.new);
console.log('  quiet look-aways  old', JSON.stringify(qo)); console.log('  quiet look-aways  new', JSON.stringify(qn));
const rOn = qn.onset_median_s / qo.onset_median_s, rP90 = qn.onset_p90_s / qo.onset_p90_s, rF = qn.idles_per_min_away / qo.idles_per_min_away;
check('onset: the first idle after he looks away comes 20–50 % sooner (median)', rOn >= 0.5 && rOn <= 0.8, `${qo.onset_median_s} s -> ${qn.onset_median_s} s (x${rOn.toFixed(2)})`);
check('onset: p90 sooner too (x0.5 .. x0.85)', rP90 >= 0.5 && rP90 <= 0.85, `${qo.onset_p90_s} s -> ${qn.onset_p90_s} s (x${rP90.toFixed(2)})`);
check('frequency: idles per minute while he is away x1.15 .. x1.6', rF >= 1.15 && rF <= 1.6, `${qo.idles_per_min_away} -> ${qn.idles_per_min_away} /min (x${rF.toFixed(2)})`);
check('frequency: not a constant fidget (≤ 17 / min in a dead-silent minute, starts ≥ minGapS apart while away)', qn.idles_per_min_away <= 17 &&
  quiet.new.every(r => { const st = L.stretches(r.att)[0], a = r.idles.filter(e => e.t >= st[0] + 1 && e.t < st[1]); return a.every((e, i) => !i || e.t - a[i - 1].t >= M.IDLE_AWAY.minGapS - 0.05); }), qn.idles_per_min_away);
check('while he looks at the screen the rate is the old one (±15 %)', Math.abs(qn.idles_per_min_attending / qo.idles_per_min_attending - 1) <= 0.15, `${qo.idles_per_min_attending} -> ${qn.idles_per_min_attending} /min`);
check('variety: never the same idle twice running', qo.repeats_in_a_row === 0 && qn.repeats_in_a_row === 0, `old ${qo.repeats_in_a_row}, new ${qn.repeats_in_a_row}`);
check('variety: ≥ 8 kinds, none over 25 %', Object.keys(qn.kinds).length >= 8 && Math.max(...Object.values(qn.kinds)) <= 0.25, JSON.stringify(qn.kinds));
check('yawn / stretch stay rare (≤ 8 % together; ≥ 35 s apart)', (qn.kinds.yawn || 0) + (qn.kinds.stretch || 0) <= 0.08 &&
  quiet.new.every(r => { const b = r.idles.filter(e => e.name === 'yawn' || e.name === 'stretch'); return b.every((e, i) => !i || e.t - b[i - 1].t >= 35); }));

// the conversation goes on while he looks away; a thinking stall and a commanded wave fall inside
const to = L.measure(many(awayTalking, false)), tn = L.measure(many(awayTalking, true));
console.log('  away while talking  old', JSON.stringify(to)); console.log('  away while talking  new', JSON.stringify(tn));
check('never while someone is speaking (flags)', tn.fired_while_speaking_flag === 0, tn.fired_while_speaking_flag);
check('never within 350 ms of a voiced frame (a pause between words is not a pause)', tn.fired_within_350ms_of_voice === 0 && qn.fired_within_350ms_of_voice === 0, `new ${tn.fired_within_350ms_of_voice} (old ${to.fired_within_350ms_of_voice})`);
check('never over a commanded gesture or a thinking stall', tn.fired_over_commanded_or_thinking === 0, `new ${tn.fired_over_commanded_or_thinking} (old ${to.fired_over_commanded_or_thinking})`);
check('away while talking: still more often, moderately (≤ +1.5 / min)', tn.idles_per_min_away >= to.idles_per_min_away && tn.idles_per_min_away - to.idles_per_min_away <= 1.5, `${to.idles_per_min_away} -> ${tn.idles_per_min_away} /min`);

// nothing changes where there is no look-away: he watches the whole time; no camera at all (no face ever seen)
const same = (tl, n = 6) => { for (let s = 1; s <= n; s++) if (JSON.stringify(L.run(M, tl, {seed: s, idle2: false}).idles) !== JSON.stringify(L.run(M, tl, {seed: s, idle2: true}).idles)) return false; return true; };
check('he watches the whole time (silent): idle for idle the old schedule', same(L.synthetic('watching', {pre: 60, away: 0, post: 0})));
check('no camera (no face ever seen): idle for idle the old schedule', same({name: 'nocam', dur: 60, at: () => ({features: null, user: 0, avatar: 0})}));
// a glance aside (0.4 s) is not a look-away
{ const tl = {name: 'glance', dur: 40, at: t => ({features: L.synthetic('x', {pre: 0, away: 0, post: 40}).at(t).features, user: 0, avatar: 0})};
  const g = {name: 'glance', dur: 40, at: t => { const x = tl.at(t); if (t % 8 < 0.4) { x.features.attending = false; x.features.attention = 0.1; } return x; }};
  check('a 0.4 s glance aside does not speed the clock', same(g, 4)); }

// the head scratch joins the away bag only where the paw is aimed at the character's own head (a body file
// with a temple); the shader animals (no body file) keep the old bag
{ const ML = require('./muse_lib.cjs'), f = ML.museFiles(), tl = silent;
  const kinds = (body, seed) => { const m = ML.M.createMotion({seed, preset: 'default', idle2: true}); if (body) m.setProps(f.props, f.body, 'muse');
    const out = []; let seen = new Set();
    for (let i = 0; i < Math.floor(tl.dur / L.DT); i++) { const x = tl.at(i * L.DT); m.step(L.DT, {features: x.features, userVoice: {rms: 0}, avatarVoice: {rms: 0}});
      for (const e of m.state.log) if (e.fidget && !seen.has(e)) { seen.add(e); out.push(e.fidget); } }
    return out; };
  let withBody = 0, without = 0;
  for (let s = 1; s <= 8; s++) { withBody += kinds(true, s).filter(n => n === 'scratch_head').length; without += kinds(false, s).filter(n => n === 'scratch_head').length; }
  check('head scratch while away: on a character with a body (paw aimed at its temple), never on a body-less one', withBody > 0 && without === 0, `body ${withBody}, no body ${without} (8 seeds)`); }

console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
