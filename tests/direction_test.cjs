// node tests/direction_test.cjs [--verbose] — per-character DIRECTION test at rig-output level (head pitch / yaw /
// roll and gaze) for all 12 characters through every command path that can move the head or the eyes. A real session's
// "lower your head" showed the head going up (a state bug, not a sign error); this pins the SIGNS on every path so a
// future sign error cannot slip in.
//
// Rig frame (app/motion.js header, GEOMETRY + SIDES) = the VIEWER'S SCREEN: head_pitch > 0 nose DOWN · head_yaw > 0 toward
// screen-right = the character's OWN LEFT · head_roll > 0 crown toward screen-right = its own left shoulder · gaze_x > 0
// eyes toward screen-right = its own left · gaze_y > 0 UP. In commands and tools 'left' / 'right' are the character's OWN.
//
// Paths (each x 12 characters; the engine's m.step() output after the motion settles, minus a baseline taken just before
// the command; a sign AND a minimum size):
//   1 pose tool      rtc.py POSE_TARGETS text -> page.html toolToCommand('pose', {targets, hold}) -> motion.command
//   2 look tool      page.html toolToCommand('look') and app/voicetools.js look (incl. its spoken aliases) -> gaze (+ head)
//   3 fast matcher   app/intent.js + intent_rules.json (act:false seated, act:true + scene for the muse), natural phrasings
//   4 planner        m.command({action:'keyframes', poses}) with each prompt's head controls, in the direction the PROMPT
//                    TEXT promises (app/planner_prompts/<character>.txt, parsed + pinned by regex)
//   5 gestures       nod (first / dominant excursion = nose down), bow (head down, eyes down), through the tool mapping
//   + agreement      pose tool vs matcher for the same intent; "own left" = positive head_yaw / gaze_x on every path
// A phrasing with no rule is reported ("no rule"), not failed; every direction needs at least one phrasing that hits.
// Exit code = number of failures (capped at 255). Last line: `direction_test: N/N ok`.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const ROOT = path.join(__dirname, '..');
const VERBOSE = process.argv.includes('--verbose');
globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js');
const {Intent} = load('app/intent.js');
const VTL = load('app/voicetools.js');
const read = rel => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const RULES = JSON.parse(read('app/intent_rules.json'));
const PROPS = JSON.parse(read('renderer/animals/muse.props.json')), BODY = JSON.parse(read('renderer/animals/muse.body.json'));
const RTC = read('app/rtc.py');
// page.html's own tool -> command code (as the voice harness's engine.cjs slices it); intentAct() = the muse's act grammar
const page = read('app/page.html');
const ttcSrc = page.slice(page.indexOf('const POSE_MAP'), page.indexOf('function toolToCommand'))
  + page.slice(page.indexOf('function toolToCommand'), page.indexOf('\n}\n', page.indexOf('function actToolCommand')) + 3);
let ACT = false;
const {toolToCommand} = new Function('intentAct', ttcSrc + '\nreturn {actToolCommand, toolToCommand};')(() => ACT);

const CHARS = ['fox', 'bear', 'rabbit', 'hippo', 'monkey', 'orb', 'cube', 'drop', 'star', 'pyramid', 'blob', 'muse'];
const COLS = ['head_pitch', 'head_yaw', 'head_roll', 'gaze_x', 'gaze_y'];
const DT = 1 / 30, SEED = 4364;

let fails = 0, passes = 0;
const failed = [], notes = [], perPath = {};
function check(pathName, ch, what, ok, expected, got) {
  const pp = perPath[pathName] || (perPath[pathName] = {ok: 0, bad: 0});
  if (ok) { passes++; pp.ok++; } else { fails++; pp.bad++; failed.push({path: pathName, character: ch, what, expected, got}); }
  if (!ok || VERBOSE) console.log((ok ? 'PASS ' : 'FAIL ') + `[${pathName}] ${ch}: ${what}` + (ok && !VERBOSE ? '' : `  — expected ${expected}, got ${got}`));
}
const f3 = v => (v >= 0 ? '+' : '') + v.toFixed(3);
const sgn = s => (s > 0 ? '> 0' : '< 0');

// a character as the page runs it: the muse with its body + scene, every other one seated without (page setProps)
function makeChar(name) {
  const m = M.createMotion({seed: SEED, character: name});
  if (name === 'muse') m.setProps(JSON.parse(JSON.stringify(PROPS)), JSON.parse(JSON.stringify(BODY)), 'muse');
  else m.setProps(null, null, name);
  const feat = () => ({t: m.now(), face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []});
  const step = n => { const rows = []; for (let i = 0; i < n; i++) { const r = m.step(DT, {features: feat()}); const o = {}; for (const c of COLS) o[c] = +r[c] || 0; rows.push(o); } return rows; };
  return {m, step, name};
}
const mean = (rows, c) => rows.reduce((a, r) => a + r[c], 0) / rows.length;
const okRes = r => r === true || (r && typeof r === 'object' ? r.ok !== false : !!r);

// warm 1 s, baseline = the last 1/3 s before the command, run it, settle: the mean of [from, to) s after the command
// returns {ok, res, d: {col: settled - baseline}, trace: per-frame deltas (1.6 s)}
const WARM = 30, BASE = 10, AFTER = 48;
const runCache = new Map();
function run(ch, exec, key, win) {
  const k = key ? ch + '|' + key + '|' + (win ? win.join() : '') : null;
  if (k && runCache.has(k)) return runCache.get(k);
  const C = makeChar(ch); ACT = ch === 'muse';
  const base = C.step(WARM).slice(-BASE), b = {};
  for (const c of COLS) b[c] = mean(base, c);
  let res; try { res = exec(C); } catch (e) { res = {ok: false, reason: 'threw: ' + String(e && e.message || e).slice(0, 120)}; }
  const rows = C.step(AFTER), [w0, w1] = win || [1.0, 1.6], W = rows.slice(Math.round(w0 * 30), Math.round(w1 * 30)), d = {};
  for (const c of COLS) d[c] = mean(W, c) - b[c];
  const out = {ok: okRes(res), res, d, trace: rows.map(r => { const o = {}; for (const c of COLS) o[c] = r[c] - b[c]; return o; })};
  if (k) runCache.set(k, out);
  return out;
}
const runCmd = (ch, cmd, src, win) => run(ch, C => C.m.command(cmd, {src: src || 'voice'}), JSON.stringify(cmd) + '|' + (src || 'voice'), win);
// expect: [[col, sign, minAbs]] — the column moved at least minAbs in that direction
function expectCols(pathName, ch, what, R, expect) {
  if (!R.ok) { check(pathName, ch, what, false, 'the command to be accepted', JSON.stringify(R.res).slice(0, 160)); return false; }
  let all = true;
  for (const [col, s, min] of expect) {
    const v = R.d[col], ok = s * v >= min;
    check(pathName, ch, `${what} -> ${col}`, ok, `${col} ${sgn(s)} (|d| >= ${min})`, `${col} ${f3(v)}`);
    all = all && ok;
  }
  return all;
}

// ------------------------------------------------------------------ the conventions as TEXT (tool schema + planner prompts)
// what a description promises for "+": a phrase -> the rig sign it must produce (own left = screen-right = positive)
const PROMISE = [
  [/\b(?:to|toward|towards) (?:your|its own) left\b(?! shoulder)/, +1], [/\b(?:to|toward|towards) (?:your|its own) right\b(?! shoulder)/, -1],
  [/\bnose down\b/, +1], [/\bnose up\b|\blooks? up\b/, -1],
  [/\b(?:your|its own) left shoulder\b/, +1], [/\b(?:your|its own) right shoulder\b/, -1]];
function promisedSign(text) {           // the text after "+ =" up to ';' / ')' / ','
  const m = /\+ = ([^;),]*)/.exec(text || ''); if (!m) return 0;
  for (const [re, s] of PROMISE) if (re.test(m[1])) return s;
  return 0;
}
const HEAD_CTL = ['head_pitch', 'head_yaw', 'head_roll'];
// rtc.py POSE_TARGETS: "head_yaw": "-1 .. 1 turn head (+ = to your left)", ...
const TOOL_TEXT = {};
for (const c of HEAD_CTL) { const m = new RegExp('"' + c + '":\\s*"([^"]*)"').exec(RTC); TOOL_TEXT[c] = m ? m[1] : ''; }
const TOOL_PIN = {head_yaw: /\(\+ = to your left\)/, head_pitch: /\(\+ = nose down, as in a nod; - = look up\)/, head_roll: /\(\+ = toward your left shoulder\)/};
for (const c of HEAD_CTL) {
  check('text', 'rtc.py', `POSE_TARGETS.${c} states its convention`, TOOL_PIN[c].test(TOOL_TEXT[c]) && promisedSign(TOOL_TEXT[c]) === 1, String(TOOL_PIN[c]) + ' (a + that maps to the positive rig column)', JSON.stringify(TOOL_TEXT[c]));
}
check('text', 'rtc.py', "look tool says left / right are the character's own", /"Look somewhere \(your own left\/right\)/.test(RTC), '"Look somewhere (your own left/right)..."', 'text not found');
// app/planner_prompts/<character>.txt: "head_yaw -0.6..0.6 (+ = turns toward its own left), head_pitch -0.45..0.45 (+ = tips
// the nose down, as in a nod; - = looks up), head_roll -0.4..0.4 (+ = tilts toward its own left shoulder)."
const PROMPT_PIN = {
  head_yaw: /\bhead_yaw -0\.6\.\.0\.6 \(\+ = turns toward its own left\)/,
  head_pitch: /\bhead_pitch -0\.45\.\.0\.45 \(\+ = tips the nose down, as in a nod; - = looks up\)/,
  head_roll: /\bhead_roll -0\.4\.\.0\.4 \(\+ = tilts toward its own left shoulder\)/};
const PROMPT_SIGN = {};                  // character -> control -> the rig sign the prompt promises for a + value
for (const ch of CHARS) {
  const f = path.join(ROOT, 'app/planner_prompts', ch + '.txt'), txt = fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : '';
  PROMPT_SIGN[ch] = {};
  check('text', ch, 'planner prompt exists', !!txt, 'app/planner_prompts/' + ch + '.txt', 'missing');
  for (const c of HEAD_CTL) {
    const m = new RegExp('\\b' + c + ' -?[\\d.]+\\.\\.[\\d.]+ (\\([^)]*\\))').exec(txt), s = m ? promisedSign(m[1]) : 0;
    PROMPT_SIGN[ch][c] = s;
    check('text', ch, `planner prompt ${c} convention text`, PROMPT_PIN[c].test(txt) && s !== 0, String(PROMPT_PIN[c]), m ? JSON.stringify(m[0]) : 'control not described');
    check('text', ch, `planner control ${c} maps to the rig column ${c}`, Array.isArray(M.KF_ALIAS[c]) && M.KF_ALIAS[c].length === 1 && M.KF_ALIAS[c][0] === c, `KF_ALIAS.${c} = ['${c}']`, JSON.stringify(M.KF_ALIAS[c]));
  }
  // no prompt offers an eye-direction control today (eyes_closed is a blink); if one appears it needs a direction check here
  const gazeCtl = (txt.match(/\b(?:gaze_[xy]|eyes?_[xy]|look_(?:x|y|left|right|up|down))\b/g) || []);
  check('text', ch, 'planner prompt has no eye-direction control (none to pin)', gazeCtl.length === 0, 'no gaze_* / eyes_x / eyes_y / look_* control', gazeCtl.join(', ') + ' — add a direction check for it');
}
{ const g = Object.keys(M.KF_ALIAS).filter(k => /gaze|^eyes?_[xy]$|^look_/.test(k) || M.KF_ALIAS[k].some(c => /^gaze_/.test(c)));
  check('text', 'motion.js', 'KF_ALIAS has no eye-direction control (none to pin)', g.length === 0, 'none', g.join(', ') + ' — add a direction check for it'); }

// ------------------------------------------------------------------ the cases
const SIGN = {};                         // character -> {key: sign of the measured delta} for the agreement checks
const rec = (ch, key, v) => { (SIGN[ch] || (SIGN[ch] = {}))[key] = Math.sign(v); };
// look: the eyes go there; the head follows the same way (it does on every character today: >= these floors)
const LOOK = {
  left: [['gaze_x', +1, 0.5], ['head_yaw', +1, 0.12]], right: [['gaze_x', -1, 0.5], ['head_yaw', -1, 0.12]],
  up: [['gaze_y', +1, 0.5], ['head_pitch', -1, 0.06]], down: [['gaze_y', -1, 0.5], ['head_pitch', +1, 0.06]]};
// the matcher: direction -> phrasings; what a hit must do depends on the command it produced (a head pose or a look)
const PHRASES = {
  head_down: ['lower your head', 'put your head down', 'head down', 'bow your head', 'drop your head', 'tilt your head down', 'chin down', 'hang your head'],
  head_up: ['raise your head', 'lift your head up', 'head up', 'chin up', 'put your head up', 'tilt your head up', 'tip your head back'],
  head_left: ['turn your head to the left', 'turn your head left', 'turn your head to your left', 'head to the left'],
  head_right: ['turn your head to the right', 'turn your head right', 'turn your head to your right', 'head to the right'],
  tilt_left: ['tilt your head to the left', 'tilt your head left', 'lean your head to the left', 'cock your head to the left'],
  tilt_right: ['tilt your head to the right', 'tilt your head right', 'lean your head to the right', 'cock your head to the right'],
  look_up: ['look up', 'look up at the ceiling', 'look at the ceiling', 'eyes up'],
  look_down: ['look down', 'look at the floor', 'look down at the ground', 'eyes down'],
  look_left: ['look left', 'look to your left', 'look to the left', 'glance left'],
  look_right: ['look right', 'look to your right', 'look to the right', 'glance right']};
const DIR = {            // direction -> the head column a pose must move, and the look it may be instead
  head_down: {pose: ['head_pitch', +1], look: 'down'}, head_up: {pose: ['head_pitch', -1], look: 'up'},
  head_left: {pose: ['head_yaw', +1], look: 'left'}, head_right: {pose: ['head_yaw', -1], look: 'right'},
  tilt_left: {pose: ['head_roll', +1]}, tilt_right: {pose: ['head_roll', -1]},
  look_up: {pose: ['head_pitch', -1], look: 'up'}, look_down: {pose: ['head_pitch', +1], look: 'down'},
  look_left: {pose: ['head_yaw', +1], look: 'left'}, look_right: {pose: ['head_yaw', -1], look: 'right'}};
const noRule = {}, offAxis = {};         // phrase -> [characters]

for (const ch of CHARS) {
  ACT = ch === 'muse';
  // ---- 1. the pose tool: + and - on each head control
  for (const ctl of HEAD_CTL) for (const v of [+0.5, -0.5]) {
    const cmd = toolToCommand('pose', {targets: {[ctl]: v}, hold: true});
    if (!cmd) { check('pose tool', ch, `pose ${ctl} ${v}`, false, 'a command', 'toolToCommand returned null'); continue; }
    const want = Math.sign(v) * promisedSign(TOOL_TEXT[ctl]);         // what the tool's own description promises
    const R = runCmd(ch, cmd);
    expectCols('pose tool', ch, `pose ${ctl} ${v > 0 ? '+' : ''}${v}`, R, [[ctl, want || Math.sign(v), 0.25]]);
    rec(ch, `pose:${ctl}:${v > 0 ? '+' : '-'}`, R.d[ctl]);
  }
  // ---- 2. the look tool: page.html toolToCommand, then app/voicetools.js look (its own target words)
  for (const tg of ['left', 'right', 'up', 'down']) for (const hold of [true, false]) {
    const R = runCmd(ch, toolToCommand('look', {target: tg, hold}));
    expectCols('look tool', ch, `look ${tg}${hold ? ' (hold)' : ''}`, R, LOOK[tg]);
    if (hold) { rec(ch, `look:${tg}:gaze`, R.d[LOOK[tg][0][0]]); rec(ch, `look:${tg}:head`, R.d[LOOK[tg][1][0]]); }
  }
  for (const [tg, dir] of [['left', 'left'], ['right', 'right'], ['up', 'up'], ['down', 'down'], ['your left', 'left'], ['your right', 'right'], ['the ceiling', 'up'], ['the floor', 'down']]) {
    const R = run(ch, C => {
      const vt = VTL.createVoiceTools({perf: () => null, motion: () => C.m, behaviors: () => null, gestures: () => null, attention: () => null,
        runCommand: cmd => C.m.command(cmd, {src: 'voice'}), sendCtl: () => {}, prompt: () => ({}), sendEvent: () => ({}),
        userSpeaking: () => false, avatarSpeaking: () => false, lastUserSpeech: () => -1e12, character: () => ch,
        now: () => C.m.now() * 1000, timers: {set: () => 0, clear: () => {}}, log: () => {}});
      return vt.call('look', {target: tg, hold: true}, 'done', {});
    }, 'vt-look:' + tg);
    expectCols('look voicetools', ch, `voicetools look "${tg}"`, R, LOOK[dir]);
  }
  // screen-side targets name the VIEWER'S side (motion.js SIDES): screen_right = positive gaze_x = the character's own left
  for (const [tg, s] of [['screen_right', +1], ['screen_left', -1]]) {
    expectCols('look tool', ch, `look ${tg}`, runCmd(ch, {action: 'look', target: tg, hold: true}), [['gaze_x', s, 0.5], ['head_yaw', s, 0.12]]);
  }
  // ---- 3. the fast matcher (the page's JS matcher: act grammar + scene on the muse only)
  const I = Intent.create(RULES, ch === 'muse' ? {act: true, scene: JSON.parse(JSON.stringify(PROPS))} : {act: false});
  let tms = 100000;
  for (const [dir, list] of Object.entries(PHRASES)) {
    let hitN = 0;
    for (const ph of list) {
      tms += 5000;
      let hits = []; try { hits = I.parse(ph, {final: true, nowMs: tms}) || []; } catch (e) { check('matcher', ch, `"${ph}" parses`, false, 'no exception', String(e).slice(0, 120)); continue; }
      if (!hits.length) { (noRule[ph] || (noRule[ph] = [])).push(ch); continue; }
      const D = DIR[dir];
      for (const h of hits) {
        const cmd = h.cmd || {};
        const poseOn = cmd.action === 'pose' && cmd.targets && Object.keys(cmd.targets).some(k => HEAD_CTL.includes(k));
        if (cmd.action === 'look' && D.look) {
          const R = runCmd(ch, cmd, 'caption');
          if (expectCols('matcher', ch, `"${ph}" (look ${cmd.target})`, R, LOOK[D.look])) hitN++;
          rec(ch, `matcher:${dir}:gaze`, R.d[LOOK[D.look][0][0]]); rec(ch, `matcher:${dir}:head`, R.d[LOOK[D.look][1][0]]);
        } else if (poseOn && cmd.targets[D.pose[0]] === undefined) {
          // a head pose on ANOTHER axis than the words name (e.g. "tilt your head up" -> a sideways head_roll): a real
          // direction defect of the rule, reported as one failure
          check('matcher', ch, `"${ph}" aims the head on the right axis`, false, `a pose on ${D.pose[0]} ${sgn(D.pose[1])}` + (D.look ? ` or look ${D.look}` : ''), 'pose ' + JSON.stringify(cmd.targets) + ' (wrong axis)');
        } else if (poseOn) {
          const R = runCmd(ch, cmd, 'caption');
          if (expectCols('matcher', ch, `"${ph}" (pose ${JSON.stringify(cmd.targets)})`, R, [[D.pose[0], D.pose[1], 0.25]])) hitN++;
          rec(ch, `matcher:${dir}:head`, R.d[D.pose[0]]);
        } else {
          // a rule fired, but not one that aims the head or the eyes (e.g. the muse's "head up" = walk): reported, not a sign
          const key = `"${ph}" -> ${JSON.stringify(cmd).slice(0, 90)}`; (offAxis[key] || (offAxis[key] = [])).push(ch);
        }
      }
    }
    check('matcher', ch, `direction ${dir}: at least one phrasing hits and moves the right way`, hitN > 0, '>= 1 of ' + list.length + ' phrasings', hitN + ' hit');
  }
  // ---- 4. planner keyframes: each head control at +v / -v goes where the PROMPT says + goes
  let kfN = 0;
  for (const ctl of HEAD_CTL) for (const v of [+0.4, -0.4]) {
    const promised = PROMPT_SIGN[ch][ctl]; if (!promised) continue;      // the 'text' check above already failed
    const cmd = {action: 'keyframes', id: 'dir' + (++kfN), description: 'direction test ' + ctl, poses: [{t: 0.4, [ctl]: v}, {t: 2.4, [ctl]: v}], done: true};
    const R = runCmd(ch, cmd);
    expectCols('planner', ch, `keyframes ${ctl} ${v > 0 ? '+' : ''}${v} (prompt: + = ${promised > 0 ? 'own left / nose down' : 'the opposite'})`, R, [[M.KF_ALIAS[ctl][0], Math.sign(v) * promised, 0.2]]);
    rec(ch, `kf:${ctl}:${v > 0 ? '+' : '-'}`, R.d[ctl]);
  }
  // ---- 5. named gestures with a head direction (through the gesture tool mapping): nod, bow
  {
    const R = runCmd(ch, toolToCommand('gesture', {name: 'nod'}));
    if (!R.ok) check('gesture', ch, 'nod accepted', false, 'ok', JSON.stringify(R.res).slice(0, 160));
    else {
      const tr = R.trace.map(r => r.head_pitch), mx = Math.max(...tr), mn = Math.min(...tr);
      const first = tr.find(v => Math.abs(v) > 0.05);
      check('gesture', ch, 'nod: dominant head_pitch excursion is nose DOWN', mx >= 0.07 && mx > 2 * Math.abs(mn), 'peak head_pitch >= +0.07 and > 2x any upward excursion', `max ${f3(mx)}, min ${f3(mn)}`);
      check('gesture', ch, 'nod: first movement is nose DOWN', first !== undefined && first > 0, 'first |d head_pitch| > 0.05 is positive', first === undefined ? 'never moved 0.05' : f3(first));
    }
    const B = runCmd(ch, {action: 'gesture', name: 'bow'}, 'voice', [0.7, 1.1]);
    expectCols('gesture', ch, 'bow', B, [['head_pitch', +1, 0.2], ['gaze_y', -1, 0.3]]);
  }
  // ---- agreement: the pose tool and the matcher for the same intent; "own left" positive everywhere
  const S = SIGN[ch] || {};
  const agree = (what, keys, want) => {
    const got = keys.map(k => [k, S[k]]), miss = got.filter(([, v]) => v === undefined), bad = got.filter(([, v]) => v !== undefined && v !== want);
    check('agreement', ch, what, !bad.length && miss.length < keys.length, `all ${sgn(want)}: ${keys.join(', ')}`, bad.length ? 'opposite: ' + bad.map(([k, v]) => `${k} ${v > 0 ? '+' : v < 0 ? '-' : '0'}`).join(', ') : 'nothing measured');
  };
  agree('head down: pose head_pitch +, matcher "head down", planner +, look down head', ['pose:head_pitch:+', 'matcher:head_down:head', 'kf:head_pitch:+', 'look:down:head', 'matcher:look_down:head'], +1);
  agree('head up: pose head_pitch -, matcher "head up", planner -, look up head', ['pose:head_pitch:-', 'matcher:head_up:head', 'kf:head_pitch:-', 'look:up:head', 'matcher:look_up:head'], -1);
  agree('own LEFT = positive head_yaw / gaze_x on every path', ['pose:head_yaw:+', 'kf:head_yaw:+', 'look:left:gaze', 'look:left:head', 'matcher:head_left:head', 'matcher:head_left:gaze', 'matcher:look_left:gaze', 'matcher:look_left:head'], +1);
  agree('own RIGHT = negative head_yaw / gaze_x on every path', ['pose:head_yaw:-', 'kf:head_yaw:-', 'look:right:gaze', 'look:right:head', 'matcher:head_right:head', 'matcher:head_right:gaze', 'matcher:look_right:gaze', 'matcher:look_right:head'], -1);
  agree('tilt to own LEFT shoulder = positive head_roll on every path', ['pose:head_roll:+', 'kf:head_roll:+', 'matcher:tilt_left:head'], +1);
  agree('tilt to own RIGHT shoulder = negative head_roll on every path', ['pose:head_roll:-', 'kf:head_roll:-', 'matcher:tilt_right:head'], -1);
}

// ------------------------------------------------------------------ summary
console.log('\ncharacters: ' + CHARS.join(', '));
for (const [p, v] of Object.entries(perPath)) console.log(`  ${p}: ${v.ok}/${v.ok + v.bad} ok`);
const who = a => (a.length === CHARS.length ? 'all 12' : a.join(', '));
const nr = Object.entries(noRule), oa = Object.entries(offAxis);
console.log('no rule (matcher phrasing with no hit — not a direction failure): ' + (nr.length ? '' : 'none'));
for (const [ph, a] of nr) console.log(`  "${ph}" — ${who(a)}`);
console.log('rule hit that does not aim the head / eyes (not a direction failure): ' + (oa.length ? '' : 'none'));
for (const [k, a] of oa) console.log(`  ${k} — ${who(a)}`);
if (failed.length) {
  console.log('\nDIRECTION FAILURES (path / character / what / expected / got):');
  for (const f of failed) console.log(`  [${f.path}] ${f.character}: ${f.what} | expected ${f.expected} | got ${f.got}`);
}
console.log(`direction_test: ${passes}/${passes + fails} ok` + (fails ? `  (${fails} FAILED)` : ''));
process.exit(Math.min(fails, 255));
