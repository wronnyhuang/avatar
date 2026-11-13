// games eval: the PAGE side of the games / standing-behaviour path, outside a browser.
// One process = one page (one item). JSON lines on stdin (ops) / stdout (replies + events).
//
//   node tools/games_eval/games_host.cjs --app <APP_ROOT> --character fox
//
// Loads <APP_ROOT>/app/voicetools.js + app/behavior.js the way the page's <script> tags do (window.Behaviors =
// lib.createBehaviors(); window.AvatarVoiceTools) and wires D.vt = createVoiceTools({...}) with page.html's deps
// (perf / motion / runCommand stubbed). PAGE-WIRING FIDELITY: Behaviors.attach({...}) is replicated ONLY when
// <APP_ROOT>/app/page.html contains "Behaviors.attach(" (an older page never called it -> every
// start_behavior's Behaviors.request() resolved {ok:false, error:'no hub link'}); the 'said' feed likewise only when the
// page has behaviorsFeed(. Real time (performance.now / setTimeout), as in the page.
//
// ops (in):  {op:'call', id, name, args}          -> {re:id, result}            (page onCall 'done' phase -> D.vt.call)
//            {op:'hub', id, msg}                   -> {re:id, handled}           (page onJSON 'behavior_code' case)
//            {op:'user', id, text}                 -> {re:id}                    (his final transcript -> vt guard + feed)
//            {op:'avatar', id, text}               -> {re:id}                    (the avatar's final words -> vt.avatarText)
//            {op:'list', id}                       -> {re:id, list, pending}
//            {op:'dryrun', id, script, maxS?}      -> {re:id, runs:[{name, ok, error, stats, trace, actions, emits}]}
//            {op:'stats', id} / {op:'quit'}
// events (out): {ev:'note', text, opt}  (vt prompt / sendEvent: a silent note for the voice model)
//               {ev:'ctl', msg}         (sendCtl / requestCode: {type:'behavior_code'} goes to the server's coder)
//               {ev:'bev', e}           (Behaviors.onEvent)   {ev:'log', rec}
'use strict';
const fs = require('fs'), path = require('path'), readline = require('readline');
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const ROOT = path.resolve(arg('app', path.join(__dirname, '..', '..')));
const CHARACTER = arg('character', 'fox');
const out = o => process.stdout.write(JSON.stringify(o) + '\n');
const load = rel => { const src = fs.readFileSync(path.join(ROOT, rel), 'utf8'); const mod = {exports: {}}; new Function('module', 'exports', src)(mod, mod.exports); return mod.exports; };

const PAGE = fs.readFileSync(path.join(ROOT, 'app', 'page.html'), 'utf8');
const WIRE = {attach: PAGE.includes('Behaviors.attach('), feed: PAGE.includes('behaviorsFeed(')};
const BL = load('app/behavior.js');
const VTL = load('app/voicetools.js');
const SIM = require(path.join(ROOT, 'app', 'behavior_sim.cjs'));
const Behaviors = BL.createBehaviors();            // = the page's window.Behaviors (created at script load, no options)
const D = {perfCalls: [], cmds: [], styles: [], ctl: []};

// the page's D.perf stand-in: do_action -> {ok, status:'preparing'} (the keyframe planner is not part of this path)
const perf = {call: (description, info) => { D.perfCalls.push({description, info: info && {source: info.source, behavior: info.behavior}}); return {ok: true, status: 'preparing', id: 'p' + D.perfCalls.length}; },
  engineCall: (info, f) => ({result: f()}), stats: () => ({})};
const sendCtl = msg => { D.ctl.push(msg); out({ev: 'ctl', msg}); };
const voicePrompt = (note, opt) => { out({ev: 'note', text: String(note), opt: opt || {}}); return {id: 'n' + Date.now()}; };
const userState = {speakingUntil: 0, lastSpeech: -1e12};
const pnow = () => performance.now();

const vt = VTL.createVoiceTools({
  perf: () => perf, motion: () => null, behaviors: () => Behaviors, gestures: () => null, attention: () => null,
  runCommand: (cmd) => { D.cmds.push(cmd); return {ok: true, id: D.cmds.length, did: 'did ' + (cmd && cmd.action)}; },
  sendCtl, prompt: voicePrompt,
  sceneMatcher: () => null, runStep: (cmd) => { D.cmds.push(cmd); return {ok: true}; }, sceneAliases: () => [],
  runLibrary: () => ({ran: false}), cameraFrame: () => null, sendFrame: () => false,
  lastActionAt: () => -1e12, matcherForget: () => {}, coveredByMatcher: () => false,
  sendEvent: (note, opt) => voicePrompt(note, opt),   // direct voice: D.voice.sendEvent(note, opt) -> same note channel
  userSpeaking: () => pnow() < userState.speakingUntil, avatarSpeaking: () => false,
  lastUserSpeech: () => userState.lastSpeech, reactMode: () => 'rules', character: () => CHARACTER, thinker: () => null,
  log: rec => out({ev: 'log', rec}),
});
vt.attach();
Behaviors.onEvent(e => { try { out({ev: 'bev', e: JSON.parse(JSON.stringify(e))}); } catch (x) { out({ev: 'bev', e: {type: e && e.type, phase: e && e.phase}}); } });
if (WIRE.attach) {
  Behaviors.attach({
    motion: () => null, character: () => CHARACTER,
    doAction: (description, meta) => perf.call(description, {source: 'behavior', behavior: meta && meta.name}),
    setStyle: s => sendCtl(Object.assign({type: 'style'}, s)),
    requestCode: msg => sendCtl(msg),
    gestures: () => [],
  });
}

function dryrun(script, maxS) {
  const runs = [];
  for (const p of Behaviors.list()) {
    const info = Behaviors.info(p.id);
    if (!info || !info.code) continue;
    let r;
    try { r = SIM.dryRun(info.code, {character: CHARACTER, script, trace: true, name: p.name, instruction: p.instruction, maxS}); }
    catch (e) { runs.push({name: p.name, ok: false, error: 'dryrun threw: ' + String(e && e.message || e).slice(0, 200)}); continue; }
    const trace = (r.trace || []).map(x => x.sets ? {t: x.t, s: Object.fromEntries(Object.entries(x.sets).map(([k, v]) => [k, +(+v[0]).toFixed(3)]))} : x);
    runs.push({name: p.name, instruction: p.instruction, ok: r.ok, error: r.error, warnings: r.warnings, stats: r.stats,
      actions: r.actions, emits: r.events.filter(e => e.type === 'rule_moment').map(e => ({t: e.t, event: e.event, data: e.data})),
      trace, code_chars: info.code.length, code: info.code.slice(0, 6000)});
  }
  return runs;
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  let m; try { m = JSON.parse(line); } catch (e) { out({ev: 'error', error: 'bad json'}); return; }
  const re = m.id;
  try {
    switch (m.op) {
      case 'call': out({re, result: vt.call(m.name, m.args || {}, 'done', {via: 'eval'})}); break;
      case 'hub': {   // page.html onJSON case 'behavior_code': the program layer first, then the voice tools
        const j = m.msg || {};
        let handled = false;
        if (typeof Behaviors.onCode === 'function' && Behaviors.onCode(j)) handled = 'behaviors';
        else if (vt.onHub(j)) handled = 'vt';
        out({re, handled}); break; }
      case 'user': {
        const text = String(m.text || '');
        vt.userSpeech('start'); vt.userSpeech('stop'); vt.userText(text, true); userState.lastSpeech = pnow();
        if (WIRE.feed) { try { Behaviors.feed({type: 'said', text: text.slice(0, 300)}); } catch (e) {} }
        out({re}); break; }
      case 'avatar': if (vt.avatarText) { try { vt.avatarText(String(m.text || ''), true); } catch (e) {} } out({re}); break;
      case 'list': out({re, list: Behaviors.list(), pending: vt.stats().pending_behaviors}); break;
      case 'dryrun': out({re, runs: dryrun(m.script, m.maxS)}); break;
      case 'stats': out({re, vt: vt.stats(), behaviors: Behaviors.stats(), wire: WIRE, perf: D.perfCalls, cmds: D.cmds.length}); break;
      case 'quit': out({re, bye: true}); process.exit(0); break;
      default: out({re, error: 'unknown op ' + m.op});
    }
  } catch (e) { out({re, error: String(e && e.stack || e).slice(0, 400)}); }
});
out({ev: 'ready', wire: WIRE, root: ROOT, character: CHARACTER, tools: ['start_behavior', 'change_behavior', 'stop_behavior', 'list_behaviors'].filter(n => vt.handles(n))});
