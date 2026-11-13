#!/usr/bin/env node
// games: start_behavior must reach the hub's coder and a coder answer must start a running program (the deployed
// page never attached the program layer: every start_behavior failed at once with 'no hub link' -> "I can't set that up").
// Drives tools/games_eval/games_host.cjs (voicetools.js + behavior.js wired as app/page.html wires them), no network.
'use strict';
const {spawn} = require('child_process'), path = require('path'), fs = require('fs');
const ROOT = process.argv[2] ? path.resolve(process.argv[2]) : path.join(__dirname, '..');
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log('PASS ' + m); } else { fail++; console.log('FAIL ' + m); } };

const page = fs.readFileSync(path.join(ROOT, 'app', 'page.html'), 'utf8');
ok(/Behaviors\.attach\(\{[\s\S]{0,600}requestCode:/.test(page), 'page: Behaviors.attach wires requestCode (the hub coder link)');
ok(/Behaviors\.attach\(\{[\s\S]{0,300}motion:/.test(page), 'page: Behaviors.attach wires the motion (the program layer installs)');
ok(/startMotion[\s\S]{0,900}Behaviors\.install\(\)/.test(page), 'page: a rebuilt motion re-installs the program layer');
ok(/behaviorsFeed\(\{type: 'said'/.test(page), 'page: his final words are fed to the program layer');

const h = spawn('node', [path.join(__dirname, '..', 'tools', 'games_eval', 'games_host.cjs'), '--app', ROOT, '--character', 'fox'], {stdio: ['pipe', 'pipe', 'inherit']});
let buf = '', n = 0;
const waiters = {}, evs = [];
h.stdout.on('data', d => {
  buf += d; let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    let m; try { m = JSON.parse(line); } catch (e) { continue; }
    if (m.re != null && waiters[m.re]) { waiters[m.re](m); delete waiters[m.re]; } else evs.push(m);
  }
});
const op = (o) => new Promise(r => { const id = ++n; waiters[id] = r; h.stdin.write(JSON.stringify(Object.assign({id}, o)) + '\n'); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const CODE = `({ setup(api) { api.claim(['head']); }, per_frame(api) { const h = api.user.head; if (h) api.set('head_yaw', Math.max(-0.6, Math.min(0.6, h.yaw)), 1); }, end(api) { api.release(); } })`;

(async () => {
  for (let k = 0; k < 100 && !evs.some(e => e.ev === 'ready'); k++) await sleep(20);
  const ready = evs.find(e => e.ev === 'ready') || {};
  ok(ready.wire && ready.wire.attach, 'host sees the page attach');
  await op({op: 'user', text: 'copy my head turns'});
  const r = await op({op: 'call', name: 'start_behavior', args: {instruction: 'whenever he turns his head, turn mine the same way', name: 'mirror_head'}});
  ok(r.result && r.result.ok && r.result.status === 'writing', 'start_behavior returns at once: writing');
  await sleep(50);
  const ctl = evs.find(e => e.ev === 'ctl' && e.msg && e.msg.type === 'behavior_code');
  ok(!!ctl, 'a behavior_code request went to the hub coder');
  ok(ctl && /copy my head turns/.test(ctl.msg.his_words || ''), 'his own latest words ride along to the coder');
  const failNote = evs.find(e => e.ev === 'note' && /could not be/.test(e.text));
  ok(!failNote, 'no instant "could not be started" note');
  if (ctl) {
    const hub = await op({op: 'hub', msg: {type: 'behavior_code', id: ctl.msg.id, ok: true, name: ctl.msg.name, code: CODE, ms: 900}});
    ok(hub.handled === 'behaviors', 'the coder answer is taken by the program layer');
    await sleep(50);
    const L = await op({op: 'list'});
    ok((L.list || []).some(p => p.name === ctl.msg.name), 'the program is running after the coder answer');
    ok(evs.some(e => e.ev === 'note' && /is running now/.test(e.text)), 'the voice model is told it is running');
  }
  const bad = await op({op: 'call', name: 'start_behavior', args: {instruction: 'when he smiles, wave', name: 'smile_wave'}});
  await sleep(50);
  const ctl2 = evs.filter(e => e.ev === 'ctl' && e.msg && e.msg.type === 'behavior_code').pop();
  await op({op: 'hub', msg: {type: 'behavior_code', id: ctl2.msg.id, ok: false, name: ctl2.msg.name, error: 'out of time', ms: 20000}});
  await sleep(50);
  ok(bad.result.ok && evs.some(e => e.ev === 'note' && /could not be started \(out of time\)/.test(e.text) && e.opt && e.opt.respond),
    'a real coder failure is reported honestly (respond)');
  await op({op: 'quit'}).catch(() => {});
  console.log(`games_test: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
