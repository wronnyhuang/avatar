#!/usr/bin/env node
/* perception_test.cjs — the PERCEPTION suite (learned gestures, attention, turn bids, hub side).
 *
 *   nice -n 10 node tests/perception_test.cjs [--full] [--out FILE]
 *
 * 1. gestures API  — app/gestures.js on a REAL recorded stream (Seamless P0299A, native MediaPipe through perception.js)
 *                    with a scripted gesture injected (tests/gesture_synth.cjs): learn 'last' (sync result) and 'next'
 *                    (Promise resolved on FRAME time), list / state / export / forget, the hub messages it sends
 *                    (gesture_save / gesture_forget), a model exported and load()ed into a fresh detector still fires,
 *                    robustness to an all-NaN / empty frame, ms per frame with 10 gestures loaded.
 * 2. gestures eval — tests/gesture_eval.cjs: learn from 3 examples, detect 8 held-out repeats among the person's normal motion
 *                    and 6 confusers (quick: 2 people × 6 gestures; --full: 3–4 people × 23 gestures, ~3 min),
 *                    counterexample ("not like this"), heavy dropouts. Floors = regression guards at the measured values.
 * 3. attention     — tests/attention_test.cjs (candidate detector on the held-out suite clips + Seamless).
 * 4. turn bids     — tests/turnbid_test.cjs (false triggers / min of normal listening, synthetic cues).
 * 5. server        — tests/test_perception_hub.py + tests/test_attention_judge.py with $PYTHON (needs aiohttp; mocked vendor).
 * Prints one JSON summary (also --out, default <tmpdir>/avatar-tests/perception_report.json); exit 1 on any failure.
 */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process');
const L = require('./perc_lib.cjs');
const SY = require('./gesture_synth.cjs');
const argv = process.argv.slice(2), arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const FULL = argv.includes('--full');
const TMP = path.join(require('os').tmpdir(), 'avatar-tests'); fs.mkdirSync(TMP, {recursive: true});
const CLIPS_DIR = process.env.AVATAR_EVAL_CLIPS || path.join(L.ROOT, 'data', 'eval_clips');
if (!fs.existsSync(CLIPS_DIR)) { console.log('SKIP eval clips not present (derived from the Seamless Interaction dataset; set AVATAR_EVAL_CLIPS=<dir>): ' + CLIPS_DIR); process.exit(0); }
const OUT = arg('out', path.join(TMP, 'perception_report.json'));
const HUB_PY = process.env.PYTHON || 'python3';
const fails = [], summary = {when: new Date().toISOString(), full: FULL};
const check = (ok, what) => { if (!ok) fails.push(what); return ok; };
const t00 = Date.now();

// ───────────────────────── 1. gestures API on a real stream ─────────────────────────
async function gesturesApi() {
  const GM = L.loadUmd('app/gestures.js');
  const ids = ['sl_S0648_0371_P0299A', 'sl_S0648_0372_P0299A', 'sl_S0648_0373_P0299A', 'sl_S0648_0375_P0299A'];
  let fr = [], off = 0;
  for (const id of ids) { const a = L.loadStream(id); const t0 = a[0].t; for (const f of a) fr.push(Object.assign({}, f, {t: f.t - t0 + off, events: []})); off = fr[fr.length - 1].t + 0.5; }
  const R = L.rng(4242);
  const teach = [61, 67, 73].map(t0 => ({g: 'peace', t0, hold: 1.1, hand: 'right', var: SY.variation(R, 0.5)}));
  const held = [95, 110, 125].map(t0 => ({g: 'peace', t0, hold: 1.0, hand: 'right', var: SY.variation(R, 1)}));
  const conf = [{g: 'fist', t0: 102, hold: 1.0, hand: 'right', var: SY.variation(R, 1)}];
  fr = SY.inject(fr, teach.concat(held, conf), {seed: 9});
  const sent = [], evs = [];
  const det = GM.create({send: m => sent.push(m), onEvent: e => evs.push(e)});
  const out = {};
  check(det.learn('nothing_yet', {from: 'last', seconds: 3}).ok === false, 'learn with no frames → ok:false');
  let nextP = null, nextRes = null, lastRes = [];
  for (const f of fr) {
    if (!nextP && f.t >= teach[0].t0 - 0.5) { nextP = det.learn('peace', {from: 'next', seconds: SY.instDur(teach[0]) + 1.0}); check(typeof nextP.then === 'function', "learn 'next' returns a Promise"); nextP.then(r => { nextRes = r; }); }
    for (const k of [1, 2]) { const e = teach[k]; if (!e.done && f.t >= e.t0 + SY.instDur(e) + 0.8) { e.done = true; const r = det.learn('peace', {from: 'last', seconds: SY.instDur(e) + 1.8}); check(r && r.ok === true && typeof r.then !== 'function', "learn 'last' returns the result at once"); lastRes.push(r); } }
    det.frame(f);
    await null;
  }
  check(nextRes && nextRes.ok, "learn 'next' resolved ok on frame time");
  const lst = det.list();
  check(lst.length === 1 && lst[0].name === 'peace' && lst[0].examples >= 3, `list(): peace with ≥ 3 examples (got ${JSON.stringify(lst.map(x => [x.name, x.examples]))})`);
  out.learned = lst[0];
  const hits = held.filter(h => evs.some(e => e.type === 'gesture' && e.name === 'peace' && e.t >= h.t0 && e.t <= h.t1 + 1)).length;
  const confHit = conf.filter(h => evs.some(e => e.type === 'gesture' && e.name === 'peace' && e.t >= h.t0 && e.t <= h.t1 + 1)).length;
  out.held_hits = `${hits}/${held.length}`; out.fist_confusions = `${confHit}/${conf.length}`;
  check(hits >= 2, `held-out peace repeats detected ≥ 2/3 (got ${hits})`);
  check(evs.filter(e => e.type === 'gesture_learned').length >= 3, 'gesture_learned events');
  const ge = evs.find(e => e.type === 'gesture');
  check(ge && typeof ge.score === 'number' && ge.score >= 0 && ge.score <= 1 && typeof ge.t === 'number' && ge.name === 'peace', 'gesture event {type, name, score 0..1, t}');
  const st = det.state('peace'); check(typeof st.on === 'boolean' && typeof st.score === 'number', 'state(name) → {on, score}');
  check(det.state('nope').on === false, 'state of an unknown gesture is off');
  check(sent.filter(m => m.type === 'gesture_save' && m.name === 'peace' && m.model && m.model.kind).length >= 3, 'gesture_save sent to the hub per learn');
  const model = det.export('peace');
  check(JSON.stringify(model).length < 1500000, `model JSON < 1.5 MB (got ${JSON.stringify(model).length})`);
  out.model_bytes = JSON.stringify(model).length;
  // reload into a fresh detector (the hub's {type:'gestures'} on the next session) and replay the held-out part
  const det2 = GM.create({}), ev2 = [];
  det2.onHub({type: 'gestures', items: [JSON.parse(JSON.stringify(model))]});
  check(det2.list().length === 1, 'onHub({type:gestures}) loads the saved model');
  for (const f of fr) { if (f.t < 85) continue; for (const e of det2.frame(f)) ev2.push(e); }
  const hits2 = held.filter(h => ev2.some(e => e.name === 'peace' && e.t >= h.t0 && e.t <= h.t1 + 1)).length;
  out.reloaded_hits = `${hits2}/${held.length}`;
  check(hits2 >= 2, `a reloaded model still detects (got ${hits2}/3)`);
  // junk frames
  let threw = false; try { det.frame({t: 1e6, face: {present: true, blend: {jawOpen: NaN}}, hands: [{lm: []}], pose: {present: true}}); det.frame({t: 1e6 + 0.05}); det.frame(null); } catch (e) { threw = true; }
  check(!threw, 'junk / partial frames do not throw');
  check(det.forget('peace') === true && det.list().length === 0 && sent.some(m => m.type === 'gesture_forget' && m.name === 'peace'), 'forget(name) + gesture_forget to the hub');
  // cost with 10 gestures loaded (5 static + 5 moving copies of real models)
  const det3 = GM.create({});
  const mv = await movingModel(GM, fr);
  const items = [];
  for (let k = 0; k < 5; k++) { items.push(Object.assign(JSON.parse(JSON.stringify(model)), {name: 'static' + k})); if (mv) items.push(Object.assign(JSON.parse(JSON.stringify(mv)), {name: 'moving' + k})); }
  det3.load(items);
  const ms = [];
  for (const f of fr.slice(0, 3000)) { const t0 = process.hrtime.bigint(); det3.frame(f); ms.push(Number(process.hrtime.bigint() - t0) / 1e6); }
  ms.sort((a, b) => a - b);
  out.ms_per_frame_10_gestures = {p50: L.r3(L.pct(ms, 0.5)), p95: L.r3(L.pct(ms, 0.95)), gestures: det3.list().length};
  check(L.pct(ms, 0.5) < 0.5, `ms/frame p50 with 10 gestures < 0.5 (got ${L.r3(L.pct(ms, 0.5))})`);
  summary.gestures_api = out;
}
async function movingModel(GM, base) {
  const R = L.rng(77);
  const teach = [61, 67, 73].map(t0 => ({g: 'wave', t0, cycles: 3, hand: 'right', var: SY.variation(R, 0.5)}));
  const fr = SY.inject(base, teach, {seed: 3});
  const det = GM.create({});
  for (const f of fr) { for (const e of teach) if (!e.done && f.t >= e.t0 + SY.instDur(e) + 0.8) { e.done = true; det.learn('wave', {from: 'last', seconds: SY.instDur(e) + 1.8}); } det.frame(f); if (f.t > 80) break; }
  const m = det.export('wave'); return m && m.kind === 'moving' ? m : null;
}

// ───────────────────────── 2–5. child suites ─────────────────────────
function run(cmd, args, label, timeoutS) {
  const t0 = Date.now();
  const r = cp.spawnSync(cmd, args, {cwd: L.ROOT, encoding: 'utf8', timeout: (timeoutS || 900) * 1000, maxBuffer: 64 << 20});
  return {code: r.status, out: r.stdout || '', err: r.stderr || '', s: Math.round((Date.now() - t0) / 100) / 10, label};
}
function lastJson(txt) { const i = txt.indexOf('\n{'); const j = txt.startsWith('{') ? 0 : i + 1; try { return JSON.parse(txt.slice(j)); } catch (e) { return null; } }

(async () => {
  await gesturesApi();
  // 2. gesture eval
  const gArgs = FULL ? ['tests/gesture_eval.cjs', '--suite', 'all', '--out', path.join(TMP, 'gesture_eval_full.json')]
    : ['tests/gesture_eval.cjs', '--suite', 'synth', '--only', 'peace,thumbs_up,call_me,wave,circle,raise_roof,kiss,brows_hold', '--persons', 'P0299A,P1281A', '--out', path.join(TMP, 'gesture_eval_quick.json')];
  const g = run('nice', ['-n', '10', 'node', ...gArgs], 'gesture_eval', 1200);
  const gj = JSON.parse(fs.readFileSync(gArgs[gArgs.length - 1], 'utf8'));
  summary.gesture_eval = {seconds: g.s, synth: gj.synth, nods: gj.nods, counter: gj.counter && {before: gj.counter.before, after: gj.counter.after}, dropout: gj.dropout && gj.dropout.all};
  const all = gj.synth && gj.synth.all;
  check(g.code === 0, 'gesture_eval ran');
  if (all) {
    check(all.hit_rate >= (FULL ? 0.62 : 0.7), `synthetic held-out hit rate ≥ ${FULL ? 0.62 : 0.7} (got ${all.hit_rate})`);
    check(all.fa_per_min <= 0.5, `false alarms ≤ 0.5 / min of normal motion (got ${all.fa_per_min})`);
  }
  if (gj.counter) check(+gj.counter.after.three_confusions.split('/')[0] <= 1, `counterexample removes the confusion (after: ${gj.counter.after.three_confusions})`);
  // 2b. invariance: clean hands-only 30 fps streams, ONE example, used at 0.8–1.3× size / ±0.3 frame / ±15° + near-misses
  const iv = run('nice', ['-n', '10', 'node', 'tests/gesture_eval.cjs', '--suite', 'invariance', '--out', path.join(TMP, 'gesture_invariance.json')], 'invariance', 900);
  const ivj = JSON.parse(fs.readFileSync(path.join(TMP, 'gesture_invariance.json'), 'utf8')).invariance;
  summary.gesture_invariance = ivj;
  check(iv.code === 0 && ivj, 'invariance suite ran');
  if (ivj) {
    check(ivj.hit_rate >= 0.6, `one-example invariance hit rate ≥ 0.6 (got ${ivj.hit_rate})`);
    check(ivj.kind_ok >= 10, `one-example static/moving kind right ≥ 10/12 (got ${ivj.kind_ok})`);
    check(ivj.near_miss_rate <= 0.2, `near-miss fires ≤ 0.2 (got ${ivj.near_miss_rate})`);
    check(ivj.fa_per_min <= 1.0, `false alarms on the idle hand ≤ 1 / min (got ${ivj.fa_per_min})`);
  }
  // 2c. turn bids the way the page drives them: window global, 30 fps, a frame WITHOUT f.attention (eye contact from the
  // head pose + gaze), a 1.8 s brow raise with both silent → one raised_brows bid
  {
    const TBm = L.loadUmd('app/turnbid.js'), TB = TBm.create ? TBm : globalThis.AvatarTurnBid;
    const evs = []; const tb = TB.create({onEvent: e => evs.push(e)});
    for (let i = 0; i < 30 * 12; i++) {
      const t = i / 30, up = t > 6 && t < 7.8 ? 1 : 0;
      tb.frame({t, face: {present: true, z: 0.5, size: 0.2, yaw: 0, pitch: 0, roll: 0, gazeX: 0, gazeY: 0, age: 0,
        blend: {browInnerUp: 0.05 + 0.35 * up, browOuterUpLeft: 0.04 + 0.3 * up, browOuterUpRight: 0.04 + 0.3 * up, jawOpen: 0.03}}, hands: [], pose: null},
        {t, userSpeaking: false, userLevel: 0, avatarSpeaking: false});
    }
    summary.turnbid_30fps_no_attention = evs.map(e => [e.cue, e.deliberateness, e.t]);
    check(evs.length === 1 && evs[0].cue === 'raised_brows', `30 fps, no f.attention: one raised_brows bid (got ${JSON.stringify(summary.turnbid_30fps_no_attention)})`);
    check(typeof require(path.join(L.ROOT, 'app/turnbid.js')).create === 'function' && typeof require(path.join(L.ROOT, 'app/gestures.js')).create === 'function' && typeof require(path.join(L.ROOT, 'app/attention.js')).create === 'function', "require('app/*.js') exports in node");
  }
  // 3. attention
  const a = run('nice', ['-n', '10', 'node', 'tests/attention_test.cjs'], 'attention', 900);
  const aj = lastJson(a.out);
  summary.attention = aj ? {ok: aj.ok, suite: aj.suite, seamless: aj.seamless, synthetic_bangs: aj.synthetic_bangs && aj.synthetic_bangs.map(b => b.detected), ms_per_frame: aj.cost && aj.cost.steady && aj.cost.steady.ms_per_frame, ms_per_thumb: aj.cost && aj.cost.ms_per_thumb, fails: aj.fails} : {code: a.code, err: a.err.slice(-400)};
  check(a.code === 0, 'attention_test passes' + (aj && aj.fails && aj.fails.length ? ': ' + aj.fails.join('; ') : ''));
  // 4. turn bids
  const t = run('nice', ['-n', '10', 'node', 'tests/turnbid_test.cjs'], 'turnbid', 900);
  const tj = lastJson(t.out);
  summary.turnbid = tj ? {ok: tj.ok, held_out_false_per_listening_min: tj.held_out_false_per_listening_min, synthetic_strong_hit_test: tj.synthetic_strong_hit_test, cost: tj.cost, fails: tj.fails} : {code: t.code, err: t.err.slice(-400)};
  check(t.code === 0, 'turnbid_test passes' + (tj && tj.fails && tj.fails.length ? ': ' + tj.fails.join('; ') : ''));
  // 5. server side (python)
  if (cp.spawnSync(HUB_PY, ['-c', 'import aiohttp']).status === 0) {
    for (const f of ['tests/test_perception_hub.py', 'tests/test_attention_judge.py']) {
      const r = run(HUB_PY, [f], f, 300);
      summary[path.basename(f, '.py')] = {code: r.code, tail: (r.err || r.out).trim().split('\n').slice(-1)[0]};
      check(r.code === 0, f + ' passes');
    }
  } else summary.hub = 'python with aiohttp missing — python tests skipped';
  summary.seconds = Math.round((Date.now() - t00) / 1000);
  summary.fails = fails; summary.ok = !fails.length;
  fs.mkdirSync(path.dirname(OUT), {recursive: true});
  fs.writeFileSync(OUT, JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary, null, 1));
  process.exit(fails.length ? 1 : 0);
})().catch(e => { console.error(e && e.stack || e); process.exit(2); });
