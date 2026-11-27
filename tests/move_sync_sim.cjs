// move sync — the instrumented replay (docs/MOVE-SYNC.md).
//
//   const {replay} = require('./move_sync_sim.cjs');   replay({sync: true|false, performPath?}) -> {cases, stats}
//   node tests/move_sync_sim.cjs [--json out.json]     prints the before / after table
//
// What is REAL here: app/perform.js (the code under test), app/motion.js + app/moves.js (the move engine plays every job:
// "motion start" and "motion end" are read off the engine, never assumed), and every case's timeline up to the motion:
// one case per commanded move in the demo's session logs (a fixture of numbers built from the recorded sessions; not part of
// this repository — point MOVE_SYNC_REAL at one to run the replay; voice sessions only): that case's own
// tool call -> first pose, -> plan ready, its move's length, and how long the voice model took to start speaking after the
// tool result. The planner's poses are recorded planner streams (tests/fixtures/planner_streams), stretched to the case's
// real length and delivered at the case's real times.
// What is MODELLED (no session log has it): the voice model's reply to a silent note (the cue, the check-in). Its latency is
// drawn from the same real distribution (tool result -> first audio, n = 140), its length from the real lengths of the
// avatar's lines in those sessions, and it obeys voice.js's rule that a note waits while a reply is being generated or
// spoken. Seeded: the same draws for before and after.
//
// Timeline marks per case (ms after the tool call): call 0 · announcement (the avatar starts its reply to the tool result) ·
// cue (it starts the "about to start" line, when asked) · motion start · motion end (every dial back at rest: the planner
// job's clock reached its end / the commanded gesture's envelope closed) · done note sent · done line (its audio starts).
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
function loadFrom(file) { const mod = {exports: {}}; new Function('module', 'exports', fs.readFileSync(file, 'utf8'))(mod, mod.exports); return mod.exports; }
const Moves = loadFrom(path.join(ROOT, 'app/moves.js')); globalThis.AvatarMovesLib = Moves;
const M = loadFrom(path.join(ROOT, 'app/motion.js'));
// without the recorded fixture: no replay rows; the scripted cases pass their own latencies and line lengths
const REAL = process.env.MOVE_SYNC_REAL && fs.existsSync(process.env.MOVE_SYNC_REAL) ? JSON.parse(fs.readFileSync(process.env.MOVE_SYNC_REAL, 'utf8'))
  : {rows: [], avatar_line_words: [3, 4, 5, 6, 7, 8], summary: null};
const SDIR = path.join(__dirname, 'fixtures', 'planner_streams');
const STREAMS = fs.readdirSync(SDIR).filter(f => f.endsWith('.json')).sort().map(f => {
  const d = JSON.parse(fs.readFileSync(path.join(SDIR, f), 'utf8'));
  return (d.ref_parse && d.ref_parse.poses || []).map(p => Object.assign({t: p.t}, p.set));
}).filter(P => P.length >= 3);
const GESTS = ['wave', 'nod', 'clap', 'shrug', 'bow', 'thumbs_up', 'shake_head', 'stretch'];

function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const quant = (a, q) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : null; };

// the real cases: voice sessions with the announcement's start on record
const ROWS = REAL.rows.filter(r => (r.voice === 'gemini' || r.voice === 'realtime') && r.first_speech != null && r.first_speech > 0 && r.outcome === 'done');
const LAT = ROWS.map(r => r.first_speech).filter(v => v >= 250 && v <= 4000);                    // tool result / note -> first audio (real)
// a line around a commanded move is short (the logs' own: "Here comes a flip!", "Let's see just how silly I can get!",
// "What do you think of that expression?"): the real lengths of the avatar's 3..8 word lines
const WORDS = REAL.avatar_line_words.filter(n => n >= 3 && n <= 8);

function simCase(row, idx, o) {
  const rnd = mulberry32(7919 * (idx + 1));
  const pick = a => a[Math.floor(rnd() * a.length)];
  const Perf = loadFrom(o.performPath || path.join(ROOT, 'app/perform.js'));
  // ---- clock + timers
  const C = {t: 1000, q: [], n: 0};
  const now = () => C.t;
  const set = (f, ms) => { const id = ++C.n; C.q.push({id, at: C.t + Math.max(0, +ms || 0), f}); return id; };
  const clear = id => { C.q = C.q.filter(x => x.id !== id); };
  // ---- the body
  let perf = null; const evs = [];
  const m = M.createMotion({seed: 100 + idx, preset: 'default', onAction: ev => { evs.push(ev); if (perf) perf.onAction(ev); }});
  m.setProps(null, null, 'fox');
  const face = {present: true, x: 0.03, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
  const frame = () => m.step(1 / 30, {features: {t: m.now(), face, attending: true, hands: []}});
  for (let i = 0; i < 60; i++) frame();
  // ---- the voice model (see the header): one reply at a time; a note waits while one is generated / spoken
  const V = {gen: null, utt: null, utts: [], pq: [], notes: []};
  const speaking = () => !!(V.utt && C.t >= V.utt.t0 && C.t < V.utt.t1);
  const busy = () => !!V.gen || speaking();
  // o.ack: the first words are the word or two the new tool result asks for (an ASSUMPTION about the voice model; without
  // it they are as long as today's)
  const lineMs = kind => o.line ? o.line(kind) : 250 + 280 * (kind === 'ack' ? pick([1, 2, 2, 3]) : pick(WORDS));
  const latMs = kind => o.lat ? o.lat(kind) : pick(LAT);
  function reply(kind, latMs, p) {
    V.gen = {kind};
    set(() => {
      V.gen = null; V.utt = {kind, t0: C.t, t1: C.t + lineMs(kind)}; V.utts.push(V.utt);
      perf.avatarAudio();                                              // the page's playback tap: silence -> voiced
      if (p && p.o && p.o.onAudio) p.o.onAudio({id: p.id, wait_ms: Math.round(C.t - p.at)});
      set(pump, V.utt.t1 - C.t + 1);
    }, latMs);
  }
  const kindOf = t => /about to start/.test(t) ? 'cue' : /finishing on screen|has finished on screen/.test(t) ? 'done' : /playing on screen/.test(t) ? 'playing' : 'note';
  function prompt(text, opt) {
    const p = {id: 'avp' + (V.notes.length + 1), text: String(text), kind: kindOf(String(text)), respond: !!(opt && opt.respond), at: C.t, o: opt || {},
      maxWaitMs: opt && opt.maxWaitMs != null ? +opt.maxWaitMs : (opt && opt.respond ? 2500 : 5000), state: 'queued'};
    V.notes.push(p);
    if (!p.respond && o.realtimeNotes) { p.state = 'sent'; p.firedAt = C.t; return p; }   // realtime: a context note is an item at once
    V.pq.push(p); pump(); return p;
  }
  function pump() {
    while (V.pq.length) {
      const p = V.pq[0];
      if (C.t - p.at > p.maxWaitMs) { V.pq.shift(); p.state = 'dropped'; continue; }
      if (busy()) break;
      V.pq.shift(); p.state = 'sent'; p.firedAt = C.t;
      if (p.o.onSent) p.o.onSent({id: p.id});
      if (p.respond) { p.state = 'speaking'; reply(p.kind, latMs(p.kind), p); break; }
    }
    if (V.pq.length) set(pump, 80);
  }
  // ---- the hub's planner: this case's real times
  const src = row.source === 'planner' ? 'planner' : row.source === 'cache' ? 'cache' : 'library';
  const gest = o.gesture || GESTS[idx % GESTS.length];
  function plan(msg) {
    const id = msg.id;
    if (src === 'library') { set(() => perf.onKf({type: 'kf', id, source: 'library', ok: true, done: true, engine: [{action: 'gesture', name: gest}], ms_first_pose: row.first_pose, ms_done: row.first_pose}), Math.max(30, row.first_pose)); return; }
    const base = STREAMS[idx % STREAMS.length], lastT = base[base.length - 1].t;
    const want = Math.max(1.2, Math.min(16, (row.motion_end - row.motion_start) / 1000 - 1.2)), k = want / lastT;
    const P = base.map(p => Object.assign({}, p, {t: +(p.t * k).toFixed(3)}));
    if (src === 'cache') { set(() => perf.onKf({type: 'kf', id, source: 'cache', ok: true, done: true, poses: P, ms_first_pose: row.first_pose, ms_done: row.first_pose}), Math.max(30, row.first_pose)); return; }
    // streamed: pose 0 at the real first-pose time, the pose that makes the job ready (>= 2 poses, >= 1 s) at the real ready
    // time, the rest over the real stream length
    let kr = P.findIndex((p, i) => i >= 1 && p.t >= 1.0); if (kr < 0) kr = P.length - 1;
    const tFirst = Math.max(60, row.first_pose), tReady = Math.max(tFirst + 30, row.ready);
    const tDone = Math.max(tReady + 60, tFirst + Math.max(0, (row.hub_done || 0) - (row.hub_first_pose || 0)));
    P.forEach((p, i) => {
      const at = i === 0 ? tFirst : i <= kr ? tFirst + (tReady - tFirst) * (i / kr) : tReady + (tDone - tReady) * ((i - kr) / Math.max(1, P.length - 1 - kr));
      const fin = i === P.length - 1;
      set(() => perf.onKf(Object.assign({type: 'kf', id, source: 'planner', poses: [p], done: fin}, fin ? {ok: true, ms_first_pose: row.first_pose, ms_done: Math.round(tDone)} : {})), at);
    });
  }
  const performs = [];
  perf = Perf.createPerformer({
    motion: () => m, prompt, timers: {set, clear}, now, cache: false, character: () => 'fox',
    sendCtl: msg => { if (msg.type === 'plan') plan(msg); else if (msg.type === 'perform') performs.push(msg); },
    runCommand: cmd => m.command(cmd, {src: 'voice'}),
    userSpeaking: () => false, lastUserSpeech: () => -1e12,
    moveSync: () => !!o.sync, avatarSpeaking: () => speaking(),
  });
  // ---- run: the tool call at t = 0 (its result goes back at once; the model's reply starts after this case's real latency)
  const t0 = C.t;
  const res = perf.call('the move he asked for', {});
  const ack = !!(o.sync && o.ack && res.status === 'preparing');
  reply(ack ? 'ack' : 'announce', row.first_speech);
  const R = {src, sync: !!o.sync, status: res.status, start: null, end: null};
  let kfId = res.id, endSeen = 0;
  for (let i = 0; i < 30 * 45; i++) {
    const tEnd = C.t + 1000 / 30;
    for (;;) { C.q.sort((a, b) => a.at - b.at || a.id - b.id); const x = C.q[0]; if (!x || x.at > tEnd) break; C.q.shift(); C.t = Math.max(C.t, x.at); x.f(); }
    C.t = tEnd; frame();
    const J = perf.list().find(x => x.id === kfId);
    if (R.start === null && J && Number.isFinite(J.marks.first_motion_ms)) R.start = J.marks.first_motion_ms - t0;
    if (R.start !== null && R.end === null) {
      if (src === 'library') { const ow = m.owned(); if (C.t - t0 - R.start > 200 && !Object.keys(ow).length) R.end = C.t - t0; }
      else { const st = m.kfState(kfId); if (st && st.end_s != null && st.played_s >= st.end_s - 1e-3) R.end = C.t - t0; }
    }
    if (R.end !== null && !busy() && !V.pq.length && ++endSeen > 30 * 6) break;
  }
  const ann = V.utts.find(u => u.kind === 'announce' || u.kind === 'ack'), cue = V.utts.find(u => u.kind === 'cue'), done = V.utts.find(u => u.kind === 'done');
  const dn = V.notes.find(p => p.kind === 'done');
  R.announce = ann ? ann.t0 - t0 : null; R.announce_end = ann ? ann.t1 - t0 : null;
  R.cue = cue ? cue.t0 - t0 : null; R.cue_end = cue ? cue.t1 - t0 : null;
  R.done_note = dn ? dn.at - t0 : null; R.done_note_state = dn ? dn.state : null;
  R.done_line = done ? done.t0 - t0 : null;
  R.release = (performs[0] && performs[0].release) || null; R.cue_kind = (performs[0] && performs[0].cue) || null;
  R.outcome = performs[0] ? performs[0].outcome : null;
  R.result = res; R.record = performs[0] || null;
  R.notes = V.notes.map(p => ({kind: p.kind, at: Math.round(p.at - t0), respond: p.respond, state: p.state, text: p.text}));
  R.lines = V.utts.map(u => ({kind: u.kind, t0: Math.round(u.t0 - t0), t1: Math.round(u.t1 - t0)}));
  // the announcing line nearest the motion's start (its reply to the call, or the cue)
  // the lines that announce the move: its reply to the call and the cue; when that reply was only an acknowledgement, the cue
  const lines = (ack && cue ? [cue] : [ann, cue]).filter(Boolean);
  const first = ack && cue ? cue : ann;                            // the first line that says it will do it
  if (R.start !== null && lines.length) {
    R.d_start_first = Math.abs(R.start - (first.t0 - t0));
    R.d_start = Math.min(...lines.map(u => Math.abs(R.start - (u.t0 - t0))));
    // it starts while a line about it is being said or within 0.3 s of that line's start / end; else how far off
    R.gap = Math.min(...lines.map(u => { const a = u.t0 - t0 - 300, b = u.t1 - t0 + 300; return R.start < a ? a - R.start : R.start > b ? R.start - b : 0; }));
    R.in_step = R.gap === 0;
  }
  if (R.end !== null && R.done_line !== null) R.d_done = Math.abs(R.done_line - R.end);
  if (R.end !== null && R.done_line !== null) R.done_signed = R.done_line - R.end;
  return R;
}

function stats(cases) {
  const f = (xs) => ({n: xs.length, median_ms: xs.length ? Math.round(quant(xs, 0.5)) : null, p90_ms: xs.length ? Math.round(quant(xs, 0.9)) : null});
  const ok = cases.filter(c => c.start !== null);
  const grp = (sel) => {
    const g = ok.filter(sel);
    return {cases: g.length, start_vs_announcement: f(g.filter(c => c.d_start != null).map(c => c.d_start)),
      start_vs_first_reply: f(g.filter(c => c.d_start_first != null).map(c => c.d_start_first)),
      in_step_pct: g.length ? Math.round(100 * g.filter(c => c.in_step).length / g.length) : null,
      gap: f(g.filter(c => c.gap != null).map(c => c.gap)),
      done_vs_motion_end: f(g.filter(c => c.d_done != null).map(c => c.d_done)),
      done_signed: f(g.filter(c => c.done_signed != null).map(c => c.done_signed)),
      done_line_missing: g.filter(c => c.end !== null && c.done_line === null).length,
      call_to_motion_start: f(g.map(c => c.start)), call_to_announcement: f(g.filter(c => c.announce != null).map(c => c.announce))};
  };
  return {all: grp(() => true), planned: grp(c => c.src === 'planner'), known: grp(c => c.src !== 'planner')};
}
function replay(o) { const cases = ROWS.map((r, i) => simCase(r, i, o || {})); return {cases, stats: stats(cases), rows: ROWS.length}; }

if (require.main === module && !ROWS.length) console.log('no recorded fixture (MOVE_SYNC_REAL): nothing to replay');
else if (require.main === module) {
  const before = replay({sync: false}), after = replay({sync: true});
  const row = (name, b, a) => console.log('  ' + name.padEnd(46) + String(b).padEnd(26) + a);
  const mp = s => s.n ? `${s.median_ms} / ${s.p90_ms} ms (n=${s.n})` : '-';
  for (const g of ['all', 'planned', 'known']) {
    const b = before.stats[g], a = after.stats[g];
    console.log(`\n${g === 'all' ? 'ALL commanded moves' : g === 'planned' ? 'moves the planner writes' : 'known moves (library / cache)'} — ${b.cases} replayed cases        before (movesync=0)       after`);
    row('|motion start - announcement start|  med / p90', mp(b.start_vs_announcement), mp(a.start_vs_announcement));
    row('|motion start - first reply start|   med / p90', mp(b.start_vs_first_reply), mp(a.start_vs_first_reply));
    row('motion starts while / as the line is said', b.in_step_pct + ' %', a.in_step_pct + ' %');
    row('  else how far from the line          med / p90', mp(b.gap), mp(a.gap));
    row('|done line - motion end|             med / p90', mp(b.done_vs_motion_end), mp(a.done_vs_motion_end));
    row('done line - motion end (signed)      med / p90', mp(b.done_signed), mp(a.done_signed));
    row('tool call -> motion start            med / p90', mp(b.call_to_motion_start), mp(a.call_to_motion_start));
    row('tool call -> first words             med / p90', mp(b.call_to_announcement), mp(a.call_to_announcement));
  }
  const ji = process.argv.indexOf('--json');
  if (ji > 0) fs.writeFileSync(process.argv[ji + 1], JSON.stringify({real: REAL.summary, before: before.stats, after: after.stats, cases: {before: before.cases, after: after.cases}}, null, 1));
}
module.exports = {replay, simCase, stats, ROWS, REAL};
