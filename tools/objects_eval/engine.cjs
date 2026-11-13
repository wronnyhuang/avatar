// objects-eval harness (+ isidle / props ops, prop slots >= 10 in the teleport check)
// voice harness: the candidate's motion engine + action log + JS matcher +
// IntentDedupe as a JSON-lines server, mirroring app/page.html runCommand, AND the page's PLANNER MODE: app/perform.js's
// performer built around the engine's motion exactly as page.html builds D.perf, with an ENGINE-TIME timer queue (timers
// fire as the engine steps; now() = engine ms), motion onAction events forwarded to it, do_action / perform_action ->
// P.call, object -> P.engineCall, release all -> P.stop + the release command (page.html makeVoice onCall). Every op
// response carries the performer's outgoing side effects since the last op: `ctl` (sendCtl: plan requests, perform
// records) and `prompts` (voice.js prompt() calls: {id, note, respond, maxWaitMs, dropIfUserSpoke, at}).
// If the candidate has app/voicetools.js (createVoiceTools) it handles a tool call first (feature-detected).
// Records EVERY rig frame and a per-frame state snapshot so any segment can be rendered with the tree's own renderer.
//   node engine.cjs <source>
'use strict';
const fs = require('fs'), path = require('path'), readline = require('readline');
const src = path.resolve(process.argv[2]); process.env.AVATAR_SOURCE = src;
const load = rel => { const s = fs.readFileSync(path.join(src, rel), 'utf8'); const mod = {exports: {}}; new Function('module', 'exports', s)(mod, mod.exports); return mod.exports; };
globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js'), A = load('app/actionlog.js');
const {Intent, IntentDedupe} = load('app/intent.js');
const AP = fs.existsSync(path.join(src, 'app/perform.js')) ? load('app/perform.js') : null;
const page = fs.readFileSync(path.join(src, 'app/page.html'), 'utf8');
const ttcSrc = page.slice(page.indexOf('const POSE_MAP'), page.indexOf('function toolToCommand')) + page.slice(page.indexOf('function toolToCommand'), page.indexOf('\n}\n', page.indexOf('function actToolCommand')) + 3);
// page.html's own tool -> command code; intentAct() = the page's act grammar flag (the muse): gesture jump/dance = the real move
const {actToolCommand, toolToCommand} = new Function('intentAct', ttcSrc + '\nreturn {actToolCommand, toolToCommand};')(() => CHAR === 'muse');
const PROPS = fs.readFileSync(path.join(src, 'renderer/animals/muse.props.json'), 'utf8');
const BODY = fs.readFileSync(path.join(src, 'renderer/animals/muse.body.json'), 'utf8');
const rules = JSON.parse(fs.readFileSync(path.join(src, 'app/intent_rules.json')));
const FPS = 30, DT = 1 / FPS;
const REL_ALIAS = {hand_l: ['paw_l'], hand_r: ['paw_r'], cheeks: ['cheek'], ear_l: ['ear_l', 'ear_flick_l', 'ear_l_turn'], ear_r: ['ear_r', 'ear_flick_r', 'ear_r_turn']};
let m, log, capMatcher, tms, lastRes, events, names, rig, st, cmdLog, sawTyping;
const feat = () => ({t: m.now(), face: {present: true, x: 0, y: 0, ax: 0, ay: 0, z: .5, lookX: 0, lookY: 0, blend: {}}, attending: true, attention: .9, motion: .02, hands: []});
let CHAR = 'muse';

// ------------------------------------------------------------------ planner mode (page.html "planner mode")
let PLANNER = true, P = null, VT = null, VTinfo = null, pageCache = true;
let timers = [], timerN = 0, ctlOut = [], promptOut = [], promptsById = new Map(), promptN = 0, perfLog = [], perfErr = [];
const nowMs = () => (m ? m.now() * 1000 : 0);                 // the performer's clock = engine ms (perf-clock stand-in)
const TIMERS = {set: (f, ms) => { const id = ++timerN; timers.push({id, at: nowMs() + Math.max(0, +ms || 0), f}); return id; },
  clear: id => { if (id == null) return; const i = timers.findIndex(x => x.id === id); if (i >= 0) timers.splice(i, 1); }};
function fireTimers() {
  for (let guard = 0; guard < 1000; guard++) {
    let k = -1;
    for (let i = 0; i < timers.length; i++) if (timers[i].at <= nowMs() + 1e-6 && (k < 0 || timers[i].at < timers[k].at || (timers[i].at === timers[k].at && timers[i].id < timers[k].id))) k = i;
    if (k < 0) return;
    const x = timers.splice(k, 1)[0];
    try { x.f(); } catch (e) { perfErr.push({t: nowMs(), where: 'timer', e: String(e && e.stack || e).slice(0, 300)}); }
  }
}
// his speech as the page sees it (voice.js userTalking / lastUserSpeechAt, page D.userSpokeAt), driven by the harness
const U = {mic: [], server: false, lastStop: -1e12, lastRespCreated: -1e12, lastSpeech: -1e12, avatarUntil: -1e12};
function userSpeaking() {
  const t = nowMs();
  if (U.mic.some(([a, b]) => t >= a && t < b) || U.server) return true;
  return t - U.lastStop < 1500 && U.lastRespCreated < U.lastStop;
}
function enginePrompt(text, opt) {                            // voice.js prompt(): the harness sends it on the socket
  opt = opt || {};
  const p = {id: 'avp' + (++promptN), note: String(text || '').slice(0, opt.maxChars || 700), respond: !!opt.respond,
    maxWaitMs: opt.maxWaitMs != null ? +opt.maxWaitMs : (opt.respond ? 2500 : 5000),
    dropIfUserSpoke: opt.dropIfUserSpoke != null ? !!opt.dropIfUserSpoke : !!opt.respond, at: Math.round(nowMs())};
  promptsById.set(p.id, {p, opt, state: 'queued'}); if (promptsById.size > 200) promptsById.delete(promptsById.keys().next().value);
  promptOut.push(p);
  return {id: p.id, get state() { const r = promptsById.get(p.id); return r ? r.state : 'gone'; }, cancel: () => {}};
}
const sendCtl = o => { ctlOut.push(JSON.parse(JSON.stringify(o))); };
const alogFor = () => (CHAR === 'muse' ? log : null);          // page: D.alog only where speechFeat() (the muse)
const bodyState = () => { const s = m && typeof m.state === 'function' ? m.state() : null;
  return s ? {posture: s.posture || null, holding: s.holding || null, wearing: s.wearing || [], seated: s.posture === 'seated' || !!s.sittingOn} : {}; };
function makePerformer() {
  if (!AP || !AP.createPerformer) return null;
  return AP.createPerformer({
    motion: () => m, prompt: enginePrompt, sendCtl,
    runCommand: (cmd, info) => runCommand(cmd, 'voice', info),
    character: () => CHAR, bodyState, userSpeaking, lastUserSpeech: () => U.lastSpeech,
    alog: alogFor, now: nowMs, timers: TIMERS, cache: pageCache,
    log: rec => { perfLog.push(Object.assign({et: +(nowMs() / 1000).toFixed(3)}, rec)); if (perfLog.length > 400) perfLog.splice(0, 100); }});
}
// app/voicetools.js (voice tools): feature-detected; deps = exactly what page.html passes (D.vt = createVoiceTools({...}))
// plus the engine-time clock / timers (the page uses performance.now / setTimeout). No program layer / perception here:
// behaviors / gestures / attention = null (the tools answer honestly that they are not available).
function makeVoiceTools() {
  const f = path.join(src, 'app/voicetools.js');
  if (!fs.existsSync(f)) { VTinfo = {present: false}; return null; }
  let lib; try { lib = load('app/voicetools.js'); } catch (e) { VTinfo = {present: true, error: 'load: ' + String(e).slice(0, 200)}; return null; }
  const mk = (lib && lib.createVoiceTools) || (globalThis.AvatarVoiceTools && globalThis.AvatarVoiceTools.createVoiceTools);
  if (!mk) { VTinfo = {present: true, error: 'no createVoiceTools export'}; return null; }
  const deps = {
    perf: () => P, motion: () => m, behaviors: () => null, gestures: () => null, attention: () => null,
    runCommand: (cmd, info) => runCommand(cmd, 'voice', info), sendCtl, prompt: enginePrompt,
    sendEvent: (note, opt) => enginePrompt(note, opt),
    // page.html (b1cea36): sceneMatcher = the page matcher (muse act grammar + scene), runStep = engineCall(caption)
    sceneMatcher: () => (CHAR === 'muse' ? capMatcher : null),
    // page.html (hybrid mode): voiceLibrary + coveredByMatcher
    runLibrary: (desc, move) => {
      let cmds = [];
      const act = CHAR === 'muse';
      if (move && move.name && act) { const c = actToolCommand('move', move); if (c) cmds = [c]; }
      else {
        const text = String(desc || ''), hits = capMatcher.parse(text, {final: true, nowMs: tms});
        if (!hits.length || (capMatcher.understood && !capMatcher.understood(text))) return null;
        cmds = hits.map(h => h.cmd);
        if (!act && cmds.some(c => !/^(gesture|pose|expression|look|release)$/.test(c.action || ''))) return null;
      }
      if (!cmds.length) return null;
      let ran = 0, dup = 0, reason = ''; const did = [];
      for (const c of cmds) { const r = runCommand(c, 'voice'); if (r && r.dup) dup++; else if (r && (r === true || r.ok !== false)) { ran++; if (r.did) did.push(r.did); } else reason = (r && r.reason) || reason; }
      return {ran, dup, reason: ran || dup ? '' : (reason || 'the body could not do it'), did: did.join(', ')};
    },
    coveredByMatcher: () => !!(IntentDedupe.recs && IntentDedupe.recs.some(r => r.source !== 'model' && (IntentDedupe.speechAt === null || r.t >= IntentDedupe.speechAt) && tms - r.t < (IntentDedupe.TURN_MS || 30000))),
    sceneAliases: () => (CHAR === 'muse' ? (JSON.parse(PROPS).props || []).map(p => ({id: p.id, aliases: p.aliases || []})) : []),
    runStep: (cmd, info) => (P ? P.engineCall({}, () => runCommand(cmd, 'caption', info)).result : runCommand(cmd, 'caption', info)),
    userSpeaking, avatarSpeaking: () => nowMs() < U.avatarUntil, lastUserSpeech: () => U.lastSpeech,
    reactMode: () => 'rules', character: () => CHAR, now: nowMs, timers: TIMERS,
    log: rec => { perfLog.push(Object.assign({et: +(nowMs() / 1000).toFixed(3)}, rec)); if (perfLog.length > 400) perfLog.splice(0, 100); }};
  let vt = null; try { vt = mk(deps); if (vt && vt.attach) vt.attach(); } catch (e) { VTinfo = {present: true, error: 'create: ' + String(e).slice(0, 200)}; return null; }
  VTinfo = {present: true, ok: !!vt, api: vt ? Object.keys(vt) : []};
  return vt;
}

function reset(seed, character, planner, cache) {
  events = []; rig = []; st = []; cmdLog = []; sawTyping = false; CHAR = character || 'muse';
  PLANNER = planner !== false; pageCache = cache !== false;
  timers = []; ctlOut = []; promptOut = []; promptsById = new Map(); perfLog = []; perfErr = []; P = null; VT = null;
  Object.assign(U, {mic: [], server: false, lastStop: -1e12, lastRespCreated: -1e12, lastSpeech: -1e12, avatarUntil: -1e12});
  const muse = CHAR === 'muse';   // any of the 12 characters (page: setProps(cfg.props||null, cfg.body||null, key))
  log = A.createActionLog({now: () => m ? m.now() : 0, limb: muse ? 'hand' : 'paw'});
  m = M.createMotion({seed: seed || 4296, character: CHAR, onAction: e => {
    events.push(Object.assign({t: m ? +m.now().toFixed(3) : 0}, e));
    if (P) { try { P.onAction(e); } catch (x) { perfErr.push({t: nowMs(), where: 'onAction', e: String(x && x.stack || x).slice(0, 300)}); } }   // page speechOnAction
    try { log.push(Object.assign({}, e, {t: undefined})); } catch (x) {}
  }});
  m.setProps(muse ? JSON.parse(PROPS) : null, muse ? JSON.parse(BODY) : null, CHAR); names = m.allParams;
  capMatcher = Intent.create(rules, muse ? {act: true, scene: JSON.parse(PROPS)} : {act: false}); IntentDedupe.reset(); if (IntentDedupe.setScene) IntentDedupe.setScene(muse ? JSON.parse(PROPS) : null); tms = 100000;
  if (PLANNER) { P = makePerformer(); VT = makeVoiceTools(); }
  step(1); log.observe(m.state());
}
function snap() { const s = m.state(); return {posture: s.posture, facing: s.facing, sittingOn: s.sittingOn, wearing: (s.wearing || []).join(','), activity: s.activity, open: (s.open || []).join(','), hl: s.holding && s.holding.left, hr: s.holding && s.holding.right}; }
function step(sec) {
  for (let i = 0; i < Math.round(sec * FPS); i++) {
    const row = m.step(DT, {features: feat()}); tms += DT * 1000;
    rig.push(names.map(k => Math.round((row[k] || 0) * 1e4) / 1e4)); const s = snap(); st.push(s);
    if (s.activity === 'typing') sawTyping = true;
    fireTimers();
  }
}
function kfBusy() { try { const k = m.kfState ? m.kfState() : null; return !!(k && /^(buffering|ready|playing)$/.test(k.phase)); } catch (e) { return false; } }
function idle() { const e = m.engine && m.engine.E; return !!(e && !e.move && !e.chain && !e.jobs.length && !e.queue && !e.oqueue) && !kfBusy() && !(P && P.active()) && !(VT && VT._V && VT._V.holds.length); }
function runIdle(max) { let i = 0; for (; i < (max || 90) * FPS; i++) { step(DT); if (i > 15 && idle()) break; } return {idle: idle(), seconds: +(i / FPS).toFixed(1)}; }
function runCommand(cmd, source, info) {                      // page.html runCommand (IntentDedupe branch)
  const dsrc = source === 'voice' ? 'model' : source;
  if (IntentDedupe.seen(cmd, tms, dsrc)) {
    lastRes = {dup: true}; const fr = IntentDedupe.last && IntentDedupe.last.rec ? IntentDedupe.last.rec.res : null;
    if (fr && fr.id != null && m.anchor && /^(move|object)$/.test(cmd.action || '')) { try { m.anchor(fr.id, source); } catch (e) {} }
    cmdLog.push({t: +m.now().toFixed(2), source, cmd, dup: true, perform: info && info.perform}); return {dup: true};
  }
  const cf = IntentDedupe.last && IntentDedupe.last.conflict; let cc = cmd;
  if (cf && cf.res && cf.res.id != null && /^(move|object)$/.test(cmd.action || '')) cc = {...cmd, replaces: cf.res.id};
  if (cc.action === 'release' && Array.isArray(cc.parts)) cc = {...cc, parts: cc.parts.flatMap(p => REL_ALIAS[p] || [p])};
  let res = null, ok = false;
  try { res = m.command(cc, {src: source}); ok = res === true || (res && typeof res === 'object' ? res.ok !== false : res !== false); } catch (e) { res = {ok: false, reason: String(e)}; }
  try { log.noteSource && log.noteSource(source); log.command(cc, res, source); } catch (e) {}
  IntentDedupe.attach && IntentDedupe.attach(res); if (!ok && IntentDedupe.forget) IntentDedupe.forget(cmd, dsrc);
  lastRes = res; cmdLog.push({t: +m.now().toFixed(2), source, cmd: cc, ok, did: res && res.did, reason: res && res.reason, id: res && res.id, perform: info && info.perform});
  return res && typeof res === 'object' ? res : ok;
}
function toolResult(ran) {
  const r = lastRes, out = {ok: !!ran || !!(r && r.dup)};
  if (r && r.dup) out.did = 'already doing that (started a moment ago)';
  else if (r && typeof r === 'object') { if (r.did) out.did = r.did; if (r.reason) out.reason = r.reason; if (r.ok === false) out.ok = false; }
  else if (!ran) out.reason = "couldn't do that";
  out.now = log.brief(); return out;
}
function toolCmd(name, a) { return toolToCommand(name, a || {}); }
// page.html makeVoice onCall ('done' phase), planner mode on/off
function tool(name, args, info) {
  args = args || {}; info = info || {};
  // page.html onCall (voice tools): plannerOn() && D.vt && D.vt.handles(name) -> D.vt.call(name, args, 'done', info)
  if (PLANNER && VT && typeof VT.call === 'function' && (typeof VT.handles !== 'function' || VT.handles(name))) {
    let r = null; try { r = VT.call(name, args, 'done', info); } catch (e) { r = {ok: false, error: 'voicetools: ' + String(e && e.message || e).slice(0, 160)}; }
    if (r && typeof r === 'object' && typeof r.then !== 'function') return Object.assign({}, r, {_via: 'voicetools', _perform: (name === 'do_action' || name === 'perform_action') ? r.id : undefined});
  }
  if ((name === 'do_action' || name === 'perform_action') && PLANNER) {
    const r = P ? P.call(args.description, info) : {ok: false, reason: 'no body here'};
    return Object.assign({}, r, {_perform: r && r.id});
  }
  if (name === 'recall_actions') { log.observe(m.state()); return {log: log.summary({sinceS: args.since_s || undefined})}; }
  const c = toolCmd(name, args); if (!c) return {ok: false, error: 'unknown tool'};
  const pl = PLANNER && P;
  if (pl && name === 'release' && (!args.parts || args.parts === 'all')) P.stop('he asked to stop');
  let ran;
  if (pl && name === 'object' && VT && VT.gate) {     // page.html (7b6c899, b1cea36): what-if guard / hold, then his words' steps
    const g = VT.gate([args.verb, args.object && (args.object.name || args.object.ref), args.target || args.place].filter(Boolean).join(' '),
      () => { if (VT.expand) VT.expand(); return P.engineCall(info, () => runCommand(c, 'voice', info)).result; });
    if (g) return Object.assign({}, g, {_cmd: c, _via: 'gate'});
    if (VT.covered && VT.covered()) return {ok: true, already: true, did: 'already doing that (started as he said it)', _cmd: c, _via: 'covered'};   // page.html (hybrid mode)
    if (VT.expand) VT.expand();
  }
  if (pl && name === 'object') {           // an object step of the same request waits for the action before it
    const qd = P.engineCall(info, () => runCommand(c, 'voice', info));
    if (qd.deferred) return Object.assign({}, qd.result, {_cmd: c, _deferred: true});
    ran = qd.result;
    if (ran && typeof ran === 'object' && ran.ok === false && !ran.dup && VT && VT.expand && VT.expand(1)) ran = {ok: true, did: 'doing what he said (his own words)'};   // page.html
  } else ran = runCommand(c, 'voice', info);
  const out = toolResult(ran); out._cmd = c; return out;
}
function teleports(W) {                                   // generator rule: one-frame spikes in root/prop/foot
  const ci = n => names.indexOf(n), col = n => W.map(r => r[ci(n)]), jumps = [];
  for (const n of names) {
    const rot = /^root_(yaw|pitch|roll)$|^prop\d+_r[xyz]$/.test(n), pos = /^root_[xyz]$|^prop\d+_[xyz]$|^foot_[lr]_[xyz]$/.test(n);
    if (!rot && !pos) continue;
    const fk = /^foot_[lr]_([xyz])$/.exec(n), rx = fk ? col('root_' + fk[1]) : null, a = fk ? col(n).map((v, i) => v + rx[i]) : col(n);
    const pk = /^prop(\d+)_/.exec(n), att = pk ? ['hl', 'hr', 'hh'].map(x => ci(`prop${pk[1]}_${x}`) >= 0 ? col(`prop${pk[1]}_${x}`) : null).filter(Boolean) : null;
    const d = a.map((v, i) => { if (!i) return 0; let x = v - a[i - 1]; if (rot) x -= Math.round(x); return Math.abs(x); });
    for (let i = 1; i < a.length; i++) {
      if (att && att.some(w => w[i] !== w[i - 1] || (i > 1 && w[i - 1] !== w[i - 2]))) continue;
      const nb = Math.max(d[i - 1] || 0, d[i + 1] || 0, 0.004);
      if (d[i] > (rot ? 0.06 : 0.05) && d[i] > 3 * nb) { jumps.push({col: n, step: +d[i].toFixed(4), t: +(i / FPS).toFixed(2)}); break; }
    }
  }
  return jumps;
}
function fullState() { const s = m.state(); return {posture: s.posture, sittingOn: s.sittingOn, atDesk: s.atDesk, chairAtDesk: s.chairAtDesk, wearing: s.wearing, open: s.open, activity: s.activity, holding: s.holding, facing: s.facing, turnedBack: s.turnedBack, looking: s.looking, props: (s.props || []).map(p => ({id: p.id, where: p.where}))}; }
const objOf = e => (e.state && e.state.object) || (e.args && e.args.object && (e.args.object.id || e.args.object.name || e.args.object.ref)) || undefined;
reset();
const rl = readline.createInterface({input: process.stdin});
rl.on('line', (line) => {
  let q, out;
  try {
    q = JSON.parse(line);
    switch (q.op) {
      case 'reset': reset(q.seed, q.character, q.planner, q.cache); out = {ok: true, frame: rig.length, character: CHAR, planner: PLANNER, performer: !!P, voicetools: VTinfo}; break;
      case 'direct': { const r = runCommand(q.cmd, 'tool'); const w = runIdle(q.max || 90); log.observe(m.state()); out = {res: r, ...w}; break; }
      case 'run': {            // page onJSON 'intent' (hybrid mode: planner mode runs it too, never on a what-if)
        if (PLANNER && VT && VT.whatIf && q.text != null && VT.whatIf(String(q.text))) { out = {res: null, skipped: 'what-if'}; break; }
        const r = runCommand(q.cmd, q.source || 'matcher'); out = {res: r}; break; }
      case 'caption': { let hs = capMatcher.parse(String(q.text || ''), {final: true, nowMs: tms});
        if (hs.length && capMatcher.understood && !capMatcher.understood(String(q.text || ''))) hs = [];   /* page.html pageMatch guard */
        if (q.dry) { out = {cmds: hs.map(h => ({cmd: h.cmd, res: null, skipped: 'planner'}))}; break; }   /* --planner-only-voice: the page matcher does not execute */
        if (PLANNER && VT && VT.whatIf && VT.whatIf(String(q.text || ''))) { out = {cmds: hs.map(h => ({cmd: h.cmd, res: null, skipped: 'what-if'}))}; break; }   /* hybrid mode: page pageMatch guard */
        out = {cmds: hs.map(h => { capMatcher.commit(h, tms); return {cmd: h.cmd, res: runCommand(h.cmd, 'caption')}; })}; break; }
      case 'speech': IntentDedupe.userSpoke && IntentDedupe.userSpoke(tms, q.started != null ? tms - (m.now() - q.started) * 1000 : undefined); out = {ok: true}; break;   /* page intentSpeech(k, startedAt) */
      case 'user': {          // his speech as voice.js sees it: {mic:[[a,b] engine s], server, stop, respCreated, speech}
        if (Array.isArray(q.mic)) U.mic = q.mic.map(([a, b]) => [a * 1000, b * 1000]);
        if (q.server != null) U.server = !!q.server;
        if (q.stop) U.lastStop = nowMs();
        if (q.respCreated) U.lastRespCreated = nowMs();
        if (q.speech) U.lastSpeech = nowMs();
        if (q.avatarUntil != null) U.avatarUntil = q.avatarUntil * 1000;
        if (VT && q.speech && typeof VT.userSpeech === 'function') { try { VT.userSpeech('start'); } catch (e) {} }
        if (VT && q.stop && typeof VT.userSpeech === 'function') { try { VT.userSpeech('stop'); } catch (e) {} }
        out = {ok: true, speaking: userSpeaking()}; break;
      }
      case 'user_text': if (VT && typeof VT.userText === 'function') { try { VT.userText(String(q.text || ''), q.final !== false); } catch (e) {} } out = {ok: true}; break;
      case 'tool_early': {     // page onCall phase 'early' (hybrid mode): do_action's streamed move -> VT.early
        let ran = false; if (PLANNER && VT && VT.early) { try { ran = VT.early(q.args && q.args.move); } catch (e) {} }
        out = {ran}; break; }
      case 'tool': {
        const r = tool(q.name, q.args, q.info);
        out = r; break;
      }
      case 'kf': { const ok = P ? P.onKf(q.msg) : false; out = {ok}; break; }                     // server -> page {type:'kf'}
      case 'prompt_audio': {                                   // the prompted reply's first audio (voice.js promptAudio)
        const r = promptsById.get(q.id); let fired = false;
        if (r && !r.audio) { r.audio = true; r.state = 'speaking'; if (r.opt.onAudio) { try { r.opt.onAudio({id: q.id, response_id: q.response_id || null}); fired = true; } catch (e) {} } }
        out = {ok: !!r, fired}; break;
      }
      case 'prompt_state': { const r = promptsById.get(q.id); if (r) { r.state = q.state; if (q.state === 'dropped' && r.opt.onDrop) { try { r.opt.onDrop(q.why || ''); } catch (e) {} } if (q.state === 'done' && r.opt.onDone) { try { r.opt.onDone({id: q.id}); } catch (e) {} } } out = {ok: !!r}; break; }
      case 'avatar_audio': if (P) P.avatarAudio(); out = {ok: true}; break;                    // page avatarVoiced(): silence -> voiced
      case 'perf': out = {jobs: P ? P.list() : [], stats: P ? P.stats() : null, active: P ? P.active() : 0, busy: P ? P.busy() : false,
        deferred: P ? P.stats().deferred : 0, timers: timers.length, kf: m.kfState ? m.kfState() : null, errors: perfErr.slice(-10),
        log: q.since != null ? perfLog.filter(x => x.et >= q.since) : undefined, vt: VT && VT.stats ? VT.stats() : null, now: m.now()}; break;
      case 'isidle': out = {idle: idle(), now: m.now(), frame: rig.length}; break;
      case 'rawstate': out = {state: JSON.parse(JSON.stringify(m.state())), now: m.now()}; break;
      case 'busy': out = {active: P ? P.active() : 0, deferred: P ? P.stats().deferred : 0, holds: VT && VT._V ? VT._V.holds.length : 0, kf: kfBusy(), now: m.now()}; break;
      case 'step': step(q.seconds || 0); log.observe(m.state()); out = {ok: true, now: m.now()}; break;
      case 'stepTo': { const n = Math.round(((q.t || 0) - m.now()) * FPS); if (n > 0) step(n / FPS); log.observe(m.state()); out = {now: m.now()}; break; }
      case 'now': out = {now: m.now(), frame: rig.length}; break;
      case 'idle': { const w = runIdle(q.max || 90); step(q.tail == null ? 1 : q.tail); log.observe(m.state()); out = w; break; }
      case 'state': out = {state: fullState(), now: m.now(), frame: rig.length}; break;
      case 'summary': log.observe(m.state()); out = {summary: log.summary({}), brief: log.brief()}; break;
      case 'events': out = {events: events.filter(e => (e.kind === 'move' || e.kind === 'object' || e.kind === 'kf' || (e.kind === 'gesture' && !/^(fidget|auto|listen|reflex|idle|cospeech|co-speech)$/.test(String(e.src || '')))) && (q.since == null || e.t >= q.since)).map(e => ({t: e.t, kind: e.kind, name: e.name, phase: e.phase, id: e.id, src: e.src, obj: objOf(e), desc: e.kind === 'kf' && e.args ? e.args.description : undefined, did: e.did || e.reason || ''}))}; break;   /* kf + commanded gestures */
      case 'cmdlog': out = {cmds: cmdLog.filter(c => q.since == null || c.t >= q.since)}; break;
      case 'alogver': out = {v: log.version(), last: log.lastChange(), now: m.now()}; break;
      case 'typing': out = {sawTyping}; sawTyping = false; break;
      case 'dump': {           // write frames [from,to) as a renderable rig.jsonl + per-frame states + teleport check
        const from = Math.max(0, q.from || 0), to = Math.min(rig.length, q.to == null ? rig.length : q.to), W = rig.slice(from, to);
        const head = {rig_params: names, time: 'clip seconds', mode: 'voice harness', id: q.id || '', fps: FPS};
        fs.mkdirSync(path.dirname(q.path), {recursive: true});
        fs.writeFileSync(q.path, [JSON.stringify(head)].concat(W.map((a, i) => JSON.stringify({type: 'rig', t: +(i / FPS).toFixed(4), a}))).join('\n') + '\n');
        fs.writeFileSync(q.path.replace(/\.jsonl$/, '') + '.states.json', JSON.stringify(st.slice(from, to)));
        out = {frames: W.length, seconds: +(W.length / FPS).toFixed(2), teleports: teleports(W)}; break;
      }
      case 'cols': { const ci = (q.names || []).map(n => names.indexOf(n)); const from = q.from || 0, to = q.to == null ? rig.length : q.to; out = {cols: Object.fromEntries((q.names || []).map((n, j) => [n, rig.slice(from, to).map(r => ci[j] >= 0 ? r[ci[j]] : null)]))}; break; }
      default: out = {error: 'bad op ' + q.op};
    }
  } catch (e) { out = {error: String(e && e.stack || e).slice(0, 600)}; }
  if (out && typeof out === 'object' && !Array.isArray(out)) {
    if (ctlOut.length) out.ctl = ctlOut.splice(0);
    if (promptOut.length) out.prompts = promptOut.splice(0);
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
