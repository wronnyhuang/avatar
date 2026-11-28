// step sync (app/perform.js gate): ONE start gate for every commanded motion — it starts with the avatar's announcing voice.
//   node tests/step_sync_test.cjs
'use strict';
const path = require('path');
const AP = require(path.join(__dirname, '..', 'app', 'perform.js'));
let fails = 0, passes = 0;
function check(name, ok, detail) { if (ok) passes++; else fails++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (!ok && detail !== undefined ? '  — ' + JSON.stringify(detail) : '')); }
function world(o) {
  o = o || {};
  const W = {t: 0, q: [], sp: false, he: false, lastHe: o.lastHe != null ? o.lastHe : -100};
  const timers = {set: (f, ms) => { const r = {at: W.t + ms, f}; W.q.push(r); return r; }, clear: r => { W.q = W.q.filter(x => x !== r); }};
  W.run = until => { for (;;) { W.q.sort((a, b) => a.at - b.at); const n = W.q[0]; if (!n || n.at > until) break; W.q.shift(); W.t = n.at; n.f(); } W.t = until; };
  W.P = AP.createPerformer({now: () => W.t, timers, userSpeaking: () => W.he, lastUserSpeech: () => W.lastHe, avatarSpeaking: () => W.sp,
    moveSync: () => true, stepSync: () => o.step !== false, stepMaxMs: () => (o.max != null ? o.max : NaN)});
  W.speak = () => { W.sp = true; W.P.avatarAudio(); };
  return W;
}
{ // silent avatar: the motion waits for its first audio
  const W = world(); let at = null, how = null;
  W.t = 100; check('held while the avatar is silent', W.P.gate(h => { at = W.t; how = h; }) === 'held');
  W.run(500); check('... not started before the voice', at === null);
  W.t = 600; W.speak(); check('... starts AT the voice onset', at === 600 && how === 'voice', {at, how});
}
{ // the bound
  const W = world(); let at = null, how = null;
  W.t = 100; W.P.gate(h => { at = W.t; how = h; });
  W.run(2000); check('no voice: starts at the 1500 ms bound', how === 'bound' && Math.abs(at - 1600) <= 40, {at, how});
}
{ // already announcing: at once
  const W = world(); W.t = 50; W.speak(); W.sp = false; let how = null;
  W.t = 300; check('the avatar already spoke after his last words: at once', W.P.gate(h => { how = h; }) === 'now' && how === 'now');
  const W2 = world(); W2.sp = true; let h2 = null; W2.t = 10; W2.P.gate(h => { h2 = h; }); check('the avatar is speaking: at once', h2 === 'now');
}
{ // he is still talking (a matcher fired mid-sentence): the bound counts from his end
  const W = world(); let at = null; W.he = true; W.t = 0; W.P.gate(() => { at = W.t; });
  W.run(1500); check('he is still talking: not at the bound yet', at === null);
  W.he = false; W.lastHe = 1500; W.run(3000); check('... the bound counts from his end (~3.0 s)', at != null && at >= 2980 && at <= 3060, at);
}
{ // switch off: at once, nothing held
  const W = world({step: false}); let how = null; W.t = 0; W.P.gate(h => { how = h; }); check('?stepsync=0: runs at once', how === 'now');
}
{ // a planned (kf) job: announced already -> released at ready, no second cue; a library run waits for the voice
  const W = world(); const prompts = [], released = [], cmds = [];
  const motion = {command: c => { cmds.push([W.t, c.action]); return {ok: true, id: 'e' + cmds.length}; }, kfRelease: id => released.push([W.t, id]), kfFeed: () => {}, state: () => ({})};
  const P = AP.createPerformer({now: () => W.t, timers: {set: (f, ms) => { const r = {at: W.t + ms, f}; W.q.push(r); return r; }, clear: r => { W.q = W.q.filter(x => x !== r); }},
    motion: () => motion, prompt: (t, o) => { prompts.push([W.t, t, !!(o && o.respond)]); return null; }, sendCtl: () => {}, userSpeaking: () => false, lastUserSpeech: () => -100,
    avatarSpeaking: () => W.sp, moveSync: () => true, stepSync: () => true, runCommand: c => motion.command(c)});
  W.P = P;
  W.t = 0; const r = P.call('do a cartwheel');
  check('tool result: the first line announces it (no "a word or two")', r.ok && /what you are about to do/.test(r.next) && !/word or two/.test(r.next), r.next);
  W.t = 900; W.sp = true; P.avatarAudio(); W.t = 2400; W.sp = false;
  W.t = 5000; P.onKf({id: r.id, poses: [{t: 0}, {t: 1.2}], done: true, ok: true, source: 'planner'});
  P.onAction({kind: 'kf', id: r.id, phase: 'ready'});
  check('planned move, announced 4 s ago: released at ready', released.length === 1 && released[0][0] === 5000, released);
  check('... no cue note asked', !prompts.some(p => p[2] && /about to start/.test(p[1])), prompts.map(p => p[1].slice(0, 60)));
  // a library answer: waits for the voice
  W.t = 20000; const r2 = P.call('wave at me'); W.t = 20100; P.onKf({id: r2.id, source: 'library', engine: [{action: 'gesture', name: 'wave'}], done: true, ok: true});
  const n0 = cmds.length; W.run(20500); check('library move: not started while the avatar is silent', cmds.length === n0);
  W.t = 20600; W.sp = true; P.avatarAudio(); check('... starts at its first audio', cmds.length === n0 + 1 && cmds[n0][0] === 20600, cmds.slice(n0));
}
console.log(`step_sync_test: ${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
