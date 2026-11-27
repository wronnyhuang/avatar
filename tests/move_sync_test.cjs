// move sync (app/perform.js, docs/MOVE-SYNC.md): a commanded move starts in step with the words that announce it, and
// the "done" line lands when the motion ends.                       node tests/move_sync_test.cjs [--json out.json]
//   1. scripted timelines (fixed latencies) through the real performer + the real move engine: each rule of the start policy,
//      the early check-in, a rules-layer run's real end, the switch, and that no tool result / first words are ever delayed
//   2. motion.eta(): how long the commanded motion still runs
//   3. with MOVE_SYNC_REAL=<fixture built from recorded session logs, not part of this repository>: the replay of every
//      commanded move in them (tests/move_sync_sim.cjs): before / after table
'use strict';
const fs = require('fs'), path = require('path');
const {replay, simCase, REAL} = require('./move_sync_sim.cjs');
const load = require('./load.cjs');
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail !== undefined && !ok ? '  — ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : '')); }
const near = (a, b, tol) => a != null && b != null && Math.abs(a - b) <= tol;
const pair = (row, o) => [simCase(row, 3, Object.assign({sync: false}, o)), simCase(row, 3, Object.assign({sync: true}, o))];
const brief = r => ({start: Math.round(r.start), end: Math.round(r.end), lines: r.lines, notes: r.notes.map(n => [n.kind, n.at, n.respond, n.state]), release: r.release, cue: r.cue_kind});
const ROW = (x) => Object.assign({source: 'planner', first_pose: 1500, ready: 1900, motion_start: 2600, motion_end: 6600, first_speech: 900, hub_first_pose: 1400, hub_done: 2400}, x);

// ---------------------------------------------------------------- 1a. its reply to the call is being spoken when the plan is ready
{
  const [b, a] = pair(ROW({}), {lat: () => 900, line: k => k === 'announce' ? 1500 : 1000});   // the reply: 0.9 .. 2.4 s, ready 1.9 s
  check(`reply being spoken at ready: the move starts as those words end (start ${Math.round(a.start)}, words end ${a.announce_end})`, near(a.start, a.announce_end, 220) /* + the 120 ms end-of-speech debounce */, brief(a));
  check('... and no second cue is asked (it used to be heard a second into the motion)', !a.notes.some(n => n.kind === 'cue') && a.lines.filter(l => l.kind === 'cue').length === 0, brief(a));
  check(`... without the switch: the old flow (cue asked at ready, start 700 ms after ready = ${Math.round(b.start)}, cue heard ${b.cue != null ? Math.round(b.cue - b.start) : '?'} ms into the motion)`,
    b.notes.some(n => n.kind === 'cue') && near(b.start, 1900 + 700, 80) && b.cue > b.start + 500, brief(b));
  check('... never later than the old flow', a.start <= b.start + 40, [a.start, b.start]);
}
// ---------------------------------------------------------------- 1b. its reply is over before the plan arrives: the early cue
{
  const row = ROW({first_speech: 600, first_pose: 2000, ready: 2450, hub_first_pose: 1900, hub_done: 3000});
  const [b, a] = pair(row, {lat: () => 900, line: () => 800});                                // the reply: 0.6 .. 1.4 s
  const cueN = a.notes.find(n => n.kind === 'cue');
  check(`reply over, slow plan: the cue is asked at the plan's first pose (${cueN && cueN.at} ms; ready ${row.ready})`, cueN && near(cueN.at, 2000, 80), brief(a));
  check(`... and the move starts at the cue's first audio (start ${Math.round(a.start)}, cue ${a.cue})`, near(a.start, a.cue, 80), brief(a));
  check(`... without the switch: start ${Math.round(b.start)} vs cue ${b.cue} (${Math.round(Math.abs(b.start - b.cue))} ms apart)`, Math.abs(b.start - b.cue) > 150, brief(b));
  check('... not later than the old flow here either', a.start <= b.start + 40, [a.start, b.start]);
  // the "done" line: one check-in, sent one model latency before the motion ends, heard as it ends
  const dn = a.notes.filter(n => n.kind === 'done');
  check(`done: ONE check-in note (${dn.length}), sent ${Math.round(a.end - a.done_note)} ms before the motion ends`, dn.length === 1 && dn[0].respond && a.end - a.done_note > 500 && a.end - a.done_note < 1300, brief(a));
  check(`done: its line is heard as the motion ends (${Math.round(a.done_line - a.end)} ms; without the switch ${Math.round(b.done_line - b.end)} ms after)`, Math.abs(a.done_line - a.end) < 200 && b.done_line - b.end > 800, [a.done_line, a.end, b.done_line, b.end]);
  check('done: a silent note tells the model the real duration as soon as the whole plan is in', a.notes.some(n => !n.respond && /takes about [\d.]+ s more/.test(n.text)) && !b.notes.some(n => /takes about/.test(n.text)), brief(a));
  check('the perform record carries the lead', a.record && a.record.sync && a.record.sync.lead_ms > 0 && a.record.checkin === 'asked (early)' && (!b.record.sync), a.record);
}
// ---------------------------------------------------------------- 1c. the plan is ready before its reply has begun
{
  const row = ROW({source: 'cache', first_pose: 120, ready: 150, first_speech: 600});
  const [b, a] = pair(row, {lat: () => 900, line: () => 1000});
  check(`plan ready before the reply: the move starts at the reply's first audio (start ${Math.round(a.start)}, words ${a.announce})`, near(a.start, a.announce, 80), brief(a));
  check('... no cue asked on top of a reply that is on its way', !a.notes.some(n => n.kind === 'cue'), brief(a));
  const [, a2] = pair(ROW({source: 'cache', first_pose: 120, ready: 150, first_speech: 2500}), {lat: () => 900, line: () => 1000});
  check(`... a reply that does not come: it starts at the cap anyway (${Math.round(a2.start)} ms)`, a2.start < 150 + 700 + 150, brief(a2));
}
// ---------------------------------------------------------------- 1d. a slow voice model: the wait for the cue is bounded
{
  const row = ROW({first_speech: 600, first_pose: 2000, ready: 2450});
  const [, a] = pair(row, {lat: k => k === 'cue' ? 4000 : 900, line: () => 800});
  check(`cue that takes 4 s: the move starts 1.5 s after ready at most (${Math.round(a.start - row.ready)} ms)`, a.start - row.ready <= 1500 + 80, brief(a));
}
// ---------------------------------------------------------------- 1e. a rules-layer run (a known gesture): its real end
{
  const row = ROW({source: 'library', first_pose: 150, ready: 150, first_speech: 700});
  const [b, a] = pair(row, {lat: () => 900, line: () => 900, gesture: 'wave'});
  check(`known move: still starts at once (${Math.round(a.start)} ms after the call, old ${Math.round(b.start)})`, a.start < 300 && near(a.start, b.start, 40), brief(a));
  const bn = b.notes.find(n => n.kind === 'done');
  check(`known move: the old flow says "finished" a fixed 1.8 s after it starts (${bn && Math.round(bn.at - b.start)} ms) whatever its length (${Math.round(b.end - b.start)} ms)`, bn && near(bn.at - b.start, 1800, 60), brief(b));
  check(`known move: the check-in leaves one latency before its real end (${Math.round(a.end - a.done_note)} ms before; line ${Math.round(a.done_line - a.end)} ms from the end)`, a.done_note != null && a.end - a.done_note > 300 && Math.abs(a.done_line - a.end) < 350, brief(a));
}
// ---------------------------------------------------------------- 1f. nothing about the reply is delayed
{
  const row = ROW({});
  const [b, a] = pair(row, {lat: () => 900, line: () => 1200});
  check('the tool result is returned at once in both flows (same status; with the switch it asks for a word or two, the announcement comes with the motion)',
    a.result && b.result && a.result.status === 'preparing' && b.result.status === 'preparing' && a.result.ok === true && /only a word or two/.test(a.result.next) && /say something short and natural now/.test(b.result.next));
  check(`the first words come at the same time (${a.announce} = ${b.announce} ms)`, a.announce === b.announce);
}
// ---------------------------------------------------------------- 2. motion.eta()
{
  const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
  const M = load('app/motion.js');
  const mk = () => { const m = M.createMotion({seed: 3}); m.setProps(null, null, 'fox'); for (let i = 0; i < 30; i++) m.step(1 / 30, {}); return m; };
  const run = (m, s) => { for (let i = 0; i < Math.round(s * 30); i++) m.step(1 / 30, {}); };
  let m = mk(); let e = m.eta();
  check('eta: nothing commanded -> {any: false, s: 0}', e.any === false && e.s === 0 && e.act === 0);
  m.command({action: 'gesture', name: 'wave'}, {src: 'tool'}); e = m.eta(); const s0 = e.s;
  check(`eta: a gesture -> its real length (${e.s} s), the action = the whole gesture`, e.any && e.known && e.s > 1 && e.s < 6 && Math.abs(e.act - e.s) < 1e-6, e);
  run(m, 1); e = m.eta(); check('eta: counts down in real time', Math.abs(e.s - (s0 - 1)) < 0.08, [s0, e.s]);
  run(m, s0); e = m.eta(); check('eta: over when the gesture is', !e.any, e);
  m = mk(); m.command({action: 'pose', targets: {blink_l: 1, blink_r: 1}, hold: true, ease_ms: 300}, {src: 'tool'}); run(m, 0.1); e = m.eta();
  check(`eta: a held pose -> the action is done once it is in place (${e.act} s), the whole thing when it has let go (${e.s} s)`, e.act < 0.4 && e.s > 3.5 && e.s < 5.5 && e.known, e);
  m = mk(); m.setPoseRevert(false); m.command({action: 'pose', targets: {blink_l: 1}, hold: true}, {src: 'tool'}); run(m, 0.1); e = m.eta();
  check('eta: pose revert off -> a held pose has no end (known: false)', e.any && e.known === false, e);
  m = mk(); m.command({action: 'keyframes', id: 'k', description: 'x', poses: [{t: 0.3, right_paw_y: 0.9}, {t: 1.5, right_paw_y: 1.0}], done: false, hold: false}, {src: 'voice'}); run(m, 0.2); e = m.eta();
  check('eta: a plan still streaming -> not known yet', e.any && e.known === false && e.act === null, e);
  m.kfFeed('k', [{t: 2.5, right_paw_y: 0.2}], true); run(m, 0.2); e = m.eta(); const st = m.kfState('k');
  check(`eta: the whole plan in -> its real end (${e.s} s; kfState end_s ${st.end_s}, played ${st.played_s})`, e.known && Math.abs(e.s - (st.end_s - st.played_s)) < 0.08 && st.end_s >= 2.5, [e, st.end_s, st.played_s]);
}
// ---------------------------------------------------------------- 3. the replay of the session logs' commanded moves
if (REAL.summary) {
  // after  = the new flow with its first words as long as today's (no assumption about the voice model)
  // after+ = ... and the first words are the word or two the new tool result asks for (an assumption: not measurable offline)
  const before = replay({sync: false}), after = replay({sync: true}), afterAck = replay({sync: true, ack: true});
  const mp = s => s.n ? `${s.median_ms} / ${s.p90_ms}` : '-';
  const R = REAL.summary;
  console.log(`\nREAL session logs (${R.performs} commanded moves in ${R.sessions_read} sessions; ${R.abs_motion_start_minus_announcement_start.n} with the announcement's start on record), ms median / p90:`);
  console.log(`  |motion start - announcement start| live   ${R.abs_motion_start_minus_announcement_start.median_ms} / ${R.abs_motion_start_minus_announcement_start.p90_ms}   (planner moves ${R.abs_motion_start_minus_announcement_start_planner.median_ms} / ${R.abs_motion_start_minus_announcement_start_planner.p90_ms}, known moves ${R.abs_motion_start_minus_announcement_start_known_move.median_ms} / ${R.abs_motion_start_minus_announcement_start_known_move.p90_ms})`);
  console.log(`  tool call -> first words ${R.call_to_first_speech.median_ms} / ${R.call_to_first_speech.p90_ms};  tool call -> motion start ${R.call_to_motion_start.median_ms} / ${R.call_to_motion_start.p90_ms}`);
  console.log(`  the cue's first audio came inside its 700 ms window in ${R.release['cue audio'] + R.release['cue audio (tap)']} of ${R.release['cue audio'] + R.release['cue audio (tap)'] + R.release['cue timeout']} planned moves (else the move started without it)`);
  for (const g of ['all', 'planned', 'known']) {
    const b = before.stats[g], a = after.stats[g], k = afterAck.stats[g];
    console.log(`\nREPLAY — ${g === 'all' ? 'all commanded moves' : g === 'planned' ? 'moves the planner writes' : 'known moves (library / cache)'} (${b.cases} cases), ms median / p90`.padEnd(70) + 'before'.padEnd(15) + 'after'.padEnd(15) + 'after+ (first words = a word or two)');
    const row = (n, x, y, z) => console.log(('  ' + n).padEnd(70) + String(x).padEnd(15) + String(y).padEnd(15) + z);
    row('|motion start - announcement start|  (the first line that says it)', mp(b.start_vs_first_reply), mp(a.start_vs_first_reply), mp(k.start_vs_first_reply));
    row('|motion start - nearest announcing line start|', mp(b.start_vs_announcement), mp(a.start_vs_announcement), mp(k.start_vs_announcement));
    row('motion starts while an announcing line is being said', b.in_step_pct + ' %', a.in_step_pct + ' %', k.in_step_pct + ' %');
    row('|done line - motion end|', mp(b.done_vs_motion_end), mp(a.done_vs_motion_end), mp(k.done_vs_motion_end));
    row('tool call -> motion start', mp(b.call_to_motion_start), mp(a.call_to_motion_start), mp(k.call_to_motion_start));
    row('tool call -> first words', mp(b.call_to_announcement), mp(a.call_to_announcement), mp(k.call_to_announcement));
  }
  const B = before.stats.all, A = after.stats.all, K = afterAck.stats.all;
  check(`replay: the replayed "before" matches the live logs (|start - announcement start| median ${B.start_vs_first_reply.median_ms} vs ${R.abs_motion_start_minus_announcement_start.median_ms} ms live)`,
    Math.abs(B.start_vs_first_reply.median_ms - R.abs_motion_start_minus_announcement_start.median_ms) < 300);
  check(`replay: more moves start while their announcement is being said (${B.in_step_pct} % -> ${A.in_step_pct} %; ${K.in_step_pct} % with short first words)`, A.in_step_pct >= B.in_step_pct + 8 && K.in_step_pct >= B.in_step_pct);
  check(`replay, first words = a word or two: |motion start - announcement start| median ${B.start_vs_first_reply.median_ms} -> ${K.start_vs_first_reply.median_ms} ms, p90 ${B.start_vs_first_reply.p90_ms} -> ${K.start_vs_first_reply.p90_ms} ms`,
    K.start_vs_first_reply.median_ms <= 0.25 * B.start_vs_first_reply.median_ms && K.start_vs_first_reply.p90_ms < B.start_vs_first_reply.p90_ms);
  check(`replay: |done line - motion end| median ${B.done_vs_motion_end.median_ms} -> ${A.done_vs_motion_end.median_ms} ms, p90 ${B.done_vs_motion_end.p90_ms} -> ${A.done_vs_motion_end.p90_ms} ms`,
    A.done_vs_motion_end.median_ms <= 0.5 * B.done_vs_motion_end.median_ms && A.done_vs_motion_end.p90_ms < B.done_vs_motion_end.p90_ms);
  check(`replay: the first words are not one ms later (${B.call_to_announcement.median_ms} = ${A.call_to_announcement.median_ms} = ${K.call_to_announcement.median_ms} ms median, same p90)`,
    A.call_to_announcement.median_ms === B.call_to_announcement.median_ms && A.call_to_announcement.p90_ms === B.call_to_announcement.p90_ms && K.call_to_announcement.median_ms === B.call_to_announcement.median_ms);
  { const b = before.stats.planned.call_to_motion_start, a = after.stats.planned.call_to_motion_start, k = afterAck.stats.planned.call_to_motion_start;
    check(`replay: a planned move does not start later at the median (${b.median_ms} -> ${a.median_ms} / ${k.median_ms} ms; p90 ${b.p90_ms} -> ${a.p90_ms} / ${k.p90_ms}); a known move starts exactly when it did`,
      a.median_ms <= b.median_ms + 50 && k.median_ms <= b.median_ms + 50 && Math.max(a.p90_ms, k.p90_ms) <= b.p90_ms + 400 &&
      after.stats.known.call_to_motion_start.median_ms === before.stats.known.call_to_motion_start.median_ms); }
  check('replay: every move still ends "done" with its check-in line', after.cases.every(c => c.outcome === 'done') && afterAck.cases.every(c => c.outcome === 'done') && A.done_line_missing <= B.done_line_missing + 2, [A.done_line_missing, B.done_line_missing]);
  const ji = process.argv.indexOf('--json');
  const slim = cs => cs.map(c => Object.assign({}, c, {notes: undefined, result: undefined, record: undefined}));
  if (ji > 0) fs.writeFileSync(process.argv[ji + 1], JSON.stringify({real: R, before: before.stats, after: after.stats, after_short_first_words: afterAck.stats, cases: {before: slim(before.cases), after: slim(after.cases), after_short_first_words: slim(afterAck.cases)}}, null, 1));
}
console.log(`\nmove_sync_test: ${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
