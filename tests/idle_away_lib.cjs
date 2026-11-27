// replay a timeline (synthetic or a real recorded session) through app/motion.js and measure the idle
// (boredom fidget) scheduler around his look-aways. Used by tests/idle_away_test.cjs and tests/idle_away_replay.cjs.
'use strict';
const fs = require('fs');
const load = require('./load.cjs');
const DT = 1 / 30, VOICE_TH = 0.012;

// ------------------------------------------------------------------ timelines
// a timeline = {name, dur, at(t) -> {features, user: rms, avatar: rms}, acts: [[t, fn(motion)]]}
function feat(t, o) {
  const away = !o.attending;
  return {t, dt: DT, face: o.face === false ? {present: false} : {present: true, x: 0.02, y: 0.03, z: 0.5, size: 0.2, yaw: away ? 0.6 : 0, pitch: 0, roll: 0,
    lookX: away ? 0.7 : 0, lookY: 0, expr: {}}, hands: [], motion: 0.02, approach: 0, attention: o.attending ? 0.9 : 0.1, attending: !!o.attending, events: []};
}
const inSpan = (t, spans) => spans.some(([a, b]) => t >= a && t < b);
// he looks at the screen `pre` s, away `away` s, back `post` s; speech spans (seconds) for him and for the avatar
function synthetic(name, o = {}) {
  const pre = o.pre ?? 20, away = o.away ?? 60, post = o.post ?? 20, he = o.he || [], av = o.av || [];
  return {name, dur: pre + away + post, acts: o.acts || [],
    at: t => ({features: feat(t, {attending: t < pre || t >= pre + away}),
      user: inSpan(t, he) ? 0.06 + 0.03 * Math.sin(40 * t) : 0, avatar: inSpan(t, av) ? 0.07 + 0.03 * Math.sin(37 * t) : 0})};
}
const SYNTH = () => [
  // a short exchange, then he looks away in silence
  synthetic('talk-then-away', {he: [[3, 8]], av: [[9, 16]]}),
  // silence throughout (the idle clock is already running when he looks away)
  synthetic('silent', {}),
  // he looks away and the conversation goes on in bursts; a thinking stall and a commanded wave fall inside the look-away
  synthetic('away-while-talking', {he: [[3, 8], [30, 33.2], [33.6, 37], [37.5, 41]], av: [[9, 16], [43, 47.3], [47.9, 52]],
    acts: [[41.2, m => m.think && m.think({variant: 'scratch'})], [42.9, m => m.thinkEnd && m.thinkEnd()], [62, m => m.command({action: 'gesture', name: 'wave'})]]}),
];

// a real session: perception's feature log (tests/perception_smoke.cjs on his camera picture) + the recording's audio
// (his mic + the avatar's voice, mixed: fed as "someone is speaking")
function wavRms(fp) {
  const b = fs.readFileSync(fp);
  let p = 12, rate = 16000, data = null, ch = 1;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), n = b.readUInt32LE(p + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(p + 10); rate = b.readUInt32LE(p + 12); }
    if (id === 'data') { data = b.subarray(p + 8, Math.min(b.length, p + 8 + n)); break; }
    p += 8 + n + (n & 1);
  }
  const hop = Math.round(rate * DT) * ch, out = [];
  for (let i = 0; i + hop * 2 <= data.length; i += hop * 2) { let s = 0; for (let k = 0; k < hop; k++) { const v = data.readInt16LE(i + 2 * k) / 32768; s += v * v; } out.push(Math.sqrt(s / hop)); }
  return out;
}
function realSession(name, jsonl, wav) {
  const feats = [], evs = [];
  for (const line of fs.readFileSync(jsonl, 'utf8').split('\n')) {
    if (!line) continue;
    let j; try { j = JSON.parse(line); } catch (e) { continue; }
    if (j.event) { if (!/^model_/.test(j.event.type)) evs.push(j.event); continue; }
    if (j.t === undefined || j.stats !== undefined) continue;
    if (j.face) { delete j.face.dbg; delete j.face.mesh; }
    delete j.pose; j.events = undefined; feats.push(j);
  }
  const rms = wav && fs.existsSync(wav) ? wavRms(wav) : [];
  const floor = rms.length ? rms.slice().sort((a, b) => a - b)[Math.floor(rms.length * 0.1)] : 0, th = Math.max(VOICE_TH, 3 * floor);
  const t0 = feats.length ? feats[0].t : 0, dur = feats.length ? feats[feats.length - 1].t - t0 : 0;
  let fi = 0, ei = 0, cur = null;
  return {name, dur, acts: [], voicedShare: rms.length ? rms.filter(x => x > th).length / rms.length : null,
    reset() { fi = 0; ei = 0; cur = null; },
    at(t) {
      const events = [];
      while (fi < feats.length && feats[fi].t - t0 <= t) cur = feats[fi++];
      while (ei < evs.length && evs[ei].t - t0 <= t) events.push(evs[ei++]);
      const r = rms[Math.floor(t / DT)] || 0;
      return {features: cur, events, user: r > th ? r : 0, avatar: 0};
    }};
}

// ------------------------------------------------------------------ one run
// -> {idles: [{t, name, bored, speakingFlag, voicedRecently, busy}], att: Uint8Array per frame, voiced: Uint8Array, dur}
function run(M, tl, {seed, idle2}) {
  const m = M.createMotion({seed, idle2, preset: 'default'});
  if (tl.reset) tl.reset();
  const n = Math.floor(tl.dur / DT), att = new Uint8Array(n), voiced = new Uint8Array(n), seen = new Set(), idles = [];
  const acts = (tl.acts || []).slice().sort((a, b) => a[0] - b[0]); let ai = 0, lastVoiced = -9;
  for (let i = 0; i < n; i++) {
    const t = i * DT, x = tl.at(t), f = x.features;
    while (ai < acts.length && acts[ai][0] <= t) acts[ai++][1](m);
    m.step(DT, {features: f, events: x.events, userVoice: {rms: x.user}, avatarVoice: {rms: x.avatar}});
    att[i] = f && f.face && f.face.present && f.attending ? 1 : 0;
    if (x.user > VOICE_TH || x.avatar > 0.008) lastVoiced = t;
    voiced[i] = t - lastVoiced < 1e-9 ? 1 : 0;
    const L = m.state.log;
    for (let k = L.length - 1; k >= 0 && k >= L.length - 6; k--) {
      const e = L[k]; if (!e.fidget || seen.has(e)) continue;
      seen.add(e);
      const before = m.debug();   // this frame's voice flags / live gestures = what the scheduler saw (its own fidget is src 'fidget')
      idles.push({t, name: e.fidget, bored: e.bored, speakingFlag: !!(before.speaking || before.listening), sinceVoiced: t - lastVoiced,
        busy: before.gestures.filter(g => /:(cmd|think|cospeech)$/.test(g)).join(',')});
    }
  }
  idles.sort((a, b) => a.t - b.t);
  return {idles, att, voiced, dur: n * DT};
}

// his look-aways in a run: not attending for >= minS (attending blips < 0.5 s bridged), cut at 20 s of no face at all
function stretches(att, minS = 3) {
  const out = [], n = att.length, bridge = Math.round(0.5 / DT);
  let i = 0;
  while (i < n) {
    if (att[i]) { i++; continue; }
    let j = i, gap = 0, end = i;
    while (j < n) { if (!att[j]) { gap = 0; end = j + 1; } else if (++gap > bridge) break; j++; }
    if ((end - i) * DT >= minS && i > 0) out.push([i * DT, end * DT]);   // i > 0: a stretch he was seen before (not the start of the log)
    i = Math.max(j, end);
  }
  return out;
}
const pct = (a, p) => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

// pooled metrics over many runs of one timeline (or several)
function measure(runs) {
  const onset = [], kinds = {}; let awayS = 0, awayQuietS = 0, nAway = 0, nStretch = 0, nStretchIdle = 0, repeats = 0, total = 0, totalS = 0,
    flagViol = 0, voicedViol = 0, busyViol = 0, attS = 0, nAtt = 0, n10 = 0, in10 = 0;
  for (const r of runs) {
    const st = stretches(r.att);
    totalS += r.dur; total += r.idles.length;
    for (let k = 1; k < r.idles.length; k++) if (r.idles[k].name === r.idles[k - 1].name) repeats++;
    for (const e of r.idles) {
      kinds[e.name] = (kinds[e.name] || 0) + 1;
      if (e.speakingFlag) flagViol++;
      if (e.sinceVoiced < 0.35) voicedViol++;
      if (e.busy) busyViol++;
    }
    let inAway = 0;
    for (const [a, b] of st) {
      nStretch++; awayS += b - a;
      for (let i = Math.floor(a / DT); i < Math.floor(b / DT); i++) if (!r.voiced[i]) awayQuietS += DT;
      const inside = r.idles.filter(e => e.t >= a && e.t < b);
      inAway += inside.length;
      if (inside.length) { nStretchIdle++; onset.push(inside[0].t - a); }
      if (b - a >= 10) { n10++; if (inside.length && inside[0].t - a <= 10) in10++; }
    }
    nAway += inAway;
    const attFrames = r.att.reduce((s, v) => s + v, 0); attS += attFrames * DT;
    nAtt += r.idles.filter(e => r.att[Math.min(r.att.length - 1, Math.floor(e.t / DT))]).length;
  }
  const r2 = x => x == null ? null : Math.round(x * 100) / 100;
  return {runs: runs.length, stretches: nStretch, stretches_with_idle: nStretchIdle, away_min: r2(awayS / 60),
    onset_median_s: r2(pct(onset, 0.5)), onset_p90_s: r2(pct(onset, 0.9)),
    first_idle_within_10s: n10 ? r2(in10 / n10) : null,   // share of the look-aways lasting ≥ 10 s (no censoring by the look-away's end)
    idles_per_min_away: awayS ? r2(nAway / (awayS / 60)) : null, idles_per_quiet_min_away: awayQuietS ? r2(nAway / (awayQuietS / 60)) : null,
    idles_per_min_attending: attS ? r2(nAtt / (attS / 60)) : null,
    idles_total: total, kinds: Object.fromEntries(Object.entries(kinds).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, r2(v / Math.max(1, total))])),
    repeats_in_a_row: repeats, fired_while_speaking_flag: flagViol, fired_within_350ms_of_voice: voicedViol, fired_over_commanded_or_thinking: busyViol};
}

module.exports = {DT, load, SYNTH, synthetic, realSession, run, stretches, measure, pct};
