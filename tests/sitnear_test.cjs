// the unnamed-place rule — "sit down" with no place named sits where it stands, facing him (was: turned its
// back on him, walked to the first chair, ~7 s); a named place ("sit in the chair", "sit at the desk") still goes there;
// already on a chair, "sit down" stays put; "stand up" works from both and ends facing him. Muse + Juno + a mascot.
// Switch: createEngine({sitNear: false}) / ?sitnear=0 / scene config sitnear:false = the old chair routing.
const fs = require('fs'), path = require('path');
const {M} = require('./muse_lib.cjs');
const ROOT = path.join(__dirname, '..');
const J = p => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));
const CH = {muse: 'renderer/animals/muse', junor: 'renderer/humans/junor/junor', pneu: 'renderer/animals/pneu'};
let pass = 0, fail = 0;
const check = (n, ok, info) => { if (ok) pass++; else fail++; console.log((ok ? 'PASS ' : 'FAIL ') + n + (ok || info === undefined ? '' : '  — ' + JSON.stringify(info).slice(0, 300))); };
const DT = 1 / 30, face = {present: true, x: 0.05, y: 0.02, z: 0.45, yaw: 0, pitch: 0, roll: 0, blend: {}};
function mk(name, off) {
  const props = J('renderer/animals/' + name + '.props.json'); if (off) props.sitnear = false;
  const m = M.createMotion({seed: 7, preset: 'default', character: name}); m.setProps(props, J(CH[name] + '.body.json'), name);
  const E = m.engine.E, step = () => m.step(DT, {features: {t: m.now(), face, attending: true, hands: []}});
  const run = (cmd, maxS = 30) => { const t0 = m.now(), r = m.command(cmd, {src: 'voice'}); for (let i = 0; i < maxS * 30; i++) { step(); if (i > 3 && !E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue) break; }
    const t = m.now() - t0; for (let i = 0; i < 20; i++) step(); const s = m.state(); return {ok: !!r && r.ok !== false, t: +t.toFixed(2), posture: s.posture, sitOn: E.sitOn, facing: s.facing, yaw: +E.base.yaw.toFixed(3)}; };
  for (let i = 0; i < 10; i++) step();
  return {m, E, run};
}
const SIT = {action: 'move', name: 'sit'}, STAND = {action: 'move', name: 'stand_up'}, O = (verb, name) => ({action: 'object', verb, object: {name}});
for (const name of Object.keys(CH)) {
  { const X = mk(name), a = X.run(SIT), b = X.run(STAND);
    check(`${name}: "sit down" -> on the floor where it stands, facing him, under 2 s`, a.ok && a.sitOn === 'floor' && a.posture === 'seated' && a.facing === 'camera' && Math.abs(a.yaw) < 0.02 && a.t < 2, a);
    check(`${name}: … "stand up" -> standing, facing him`, b.ok && b.posture === 'standing' && b.facing === 'camera' && Math.abs(b.yaw) < 0.02, b); }
  { const X = mk(name), a = X.run(O('sit_in', 'chair')), c = X.run(SIT), b = X.run(STAND);
    check(`${name}: "sit in the chair" -> on the chair`, a.ok && a.sitOn === 'chair' && a.posture === 'seated', a);
    check(`${name}: on the chair, "sit down" -> stays on the chair`, c.ok && c.sitOn === 'chair' && c.t < 1, c);
    check(`${name}: … "stand up" -> standing, facing him (endstate rule)`, b.ok && b.posture === 'standing' && b.facing === 'camera' && Math.abs(b.yaw) < 0.02, b); }
  { const X = mk(name), a = X.run(O('sit_at', 'desk'), 90);
    check(`${name}: "sit at the desk" -> on the chair at the desk`, a.ok && a.sitOn === 'chair', a); }
  { const X = mk(name), a = X.run({action: 'object', verb: 'sit'});
    check(`${name}: object sit with nothing named -> the floor`, a.ok && a.sitOn === 'floor', a); }
  { const X = mk(name); X.E.base.yaw = X.E.cur.yaw = 0.2; const a = X.run(SIT);   // standing turned to its left (as after a walk)
    check(`${name}: turned away (yaw 0.2) then "sit down" -> turns to him on the spot, sits on the floor facing him`, a.sitOn === 'floor' && a.facing === 'camera' && Math.abs(a.yaw) < 0.02 && a.t < 3.5, a); }
  { const X = mk(name, true), a = X.run(SIT);
    check(`${name}: switch off (sitnear:false) -> the old routing (walks to the chair)`, a.sitOn === 'chair', a); }
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
