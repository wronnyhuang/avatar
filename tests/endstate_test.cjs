// a commanded move ends in a neutral state on the real muse engine + scene.
//  (1) "stand up" from the chair ends FACING him (was: standing at the chair turned away until the next command)
//  (2) "take off the hat" with nowhere named: shown in a paw for a moment, then it goes back where it lives on its own;
//      a thing he asked it to hold / pick up stays held; "hold it" / another request about it takes it over;
//      scene config endstate:false turns both off.
const {makeMuse} = require('./muse_lib.cjs');
let pass = 0, fail = 0;
function check(name, ok, info) { if (ok) pass++; else fail++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok || info === undefined ? '' : '  — ' + JSON.stringify(info).slice(0, 300))); }
const O = (verb, name) => ({action: 'object', verb, object: {name}});
const st = X => { const s = X.m.state(), E = X.m.engine.E, h = E.P.find(p => p.id === 'hat'); return {posture: s.posture, facing: s.facing, yaw: +E.base.yaw.toFixed(3), held: h.held || null, worn: !!h.worn, where: h.where}; };
const mk = off => { const X = makeMuse(3); X.step(5, false); if (off) X.m.engine.E.endRevert = false; return X; };
const secs = (X, s) => { X.step(Math.round(s * 30)); X.runUntilIdle(); X.step(10); };

for (const off of [false, true]) {
  const X = mk(off), tag = off ? 'switch off (endstate:false)' : 'default';
  X.m.command(O('sit_in', 'chair'), {src: 'voice'});   /* the chair named (a bare 'sit' now sits on the spot) */ X.runUntilIdle();
  const seated = X.m.state().posture;
  const r = X.m.command({action: 'move', name: 'stand_up'}, {src: 'voice'}); X.runUntilIdle(); X.step(30);
  const s = st(X);
  if (!off) check(`${tag}: sit in the chair, stand up -> standing, facing him (yaw 0)`, seated === 'seated' && r.ok !== false && s.posture === 'standing' && s.facing === 'camera' && Math.abs(s.yaw) < 0.02, s);
  else check(`${tag}: the old end state (stays turned at the chair)`, s.posture === 'standing' && Math.abs(s.yaw) > 0.06, s);
}
{
  const X = mk(); X.m.command(O('sit_in', 'chair'), {src: 'voice'});   /* the chair named (a bare 'sit' now sits on the spot) */ X.runUntilIdle(); X.step(60);
  check('seated stays seated and is not turned by the rule', X.m.state().posture === 'seated');
}
for (const off of [false, true]) {
  const X = mk(off), tag = off ? 'switch off' : 'default';
  X.m.command(O('put_on', 'hat'), {src: 'voice'}); X.runUntilIdle();
  const w = st(X);
  X.m.command(O('take_off', 'hat'), {src: 'voice'}); X.runUntilIdle(); X.step(20);
  const a = st(X); secs(X, 5); const b = st(X);
  check(`${tag}: take off the hat -> first held in a paw (the show)`, w.worn && !a.worn && !!a.held, a);
  if (!off) check(`${tag}: … then back where it lives on its own, paws empty, facing him`, !b.held && !b.worn && b.where === 'home' && b.facing === 'camera', b);
  else check(`${tag}: … stays held (old behaviour)`, !!b.held, b);
  if (!off) { secs(X, 5); check('wearing the hat again afterwards still works', X.m.command(O('put_on', 'hat'), {src: 'voice'}).ok !== false && (X.runUntilIdle(), st(X).worn)); }
}
{
  const X = mk(); X.m.command(O('pick_up', 'hat'), {src: 'voice'}); X.runUntilIdle(); secs(X, 7);
  check('"pick up the hat": holding persists (he asked to hold)', !!st(X).held, st(X));
}
{
  const X = mk(); X.m.command(O('put_on', 'hat'), {src: 'voice'}); X.runUntilIdle();
  X.m.command(O('take_off', 'hat'), {src: 'voice'}); X.runUntilIdle(); X.step(20);
  X.m.command({action: 'object', verb: 'hold', object: {ref: 'it'}}, {src: 'voice'}); secs(X, 7);
  check('take off the hat, then "hold it": holding persists', !!st(X).held, st(X));
}
{
  const X = mk(); X.m.command(O('put_on', 'hat'), {src: 'voice'}); X.runUntilIdle();
  X.m.command(O('take_off', 'hat'), {src: 'voice'}); X.runUntilIdle(); X.step(20);
  X.m.command(O('put_on', 'hat'), {src: 'voice'}); secs(X, 7);
  check('take off, then "put it back on" before the put-back: worn, never put away', st(X).worn, st(X));
}
{
  const X = mk(); X.m.command(O('put_on', 'hat'), {src: 'voice'}); X.runUntilIdle();
  X.m.command(O('take_off', 'hat'), {src: 'voice'}); X.runUntilIdle(); X.step(20);
  X.m.command({action: 'move', name: 'wave'}, {src: 'voice'}); secs(X, 9);
  const s = st(X);
  check('take off, then an unrelated move (wave): the hat still goes back once idle', !s.held && s.where === 'home', s);
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
