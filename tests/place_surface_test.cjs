// node tests/place_surface_test.cjs — a prop put on a surface records THAT surface as its place.
// Field case: "put the book on the desk" left the book resting on the desk top while its place read
// "floor" — the desk's near-edge rest spots (moves.js freeSpotNear) sit inside the desk but outside the desk_top place's
// footprint, and the release re-derived the place from that footprint alone (placeAt), falling back to the floor.
// Generic: every movable prop x every surface place on the real muse scene; plus the floor still reads "floor".
// Exit code = failures.
'use strict';
const {makeMuse} = require('./muse_lib.cjs');
let fails = 0, n = 0;
const check = (name, ok, d) => { n++; console.log((ok ? 'PASS ' : 'FAIL ') + name + (d !== undefined && !ok ? '  — ' + (typeof d === 'string' ? d : JSON.stringify(d)) : '')); if (!ok) fails++; };
const O = (verb, name, x) => Object.assign({action: 'object', verb, object: {name}}, x || {});
const idle = T => { const E = T.m.engine.E; return !E.move && !E.chain && !E.jobs.length && !E.queue && !E.oqueue; };
const runIdle = (T, s = 60) => { for (let i = 0; i < s * 30; i++) { T.step(1, false); if (idle(T) && i > 3) return true; } return false; };
const T0 = makeMuse(21); T0.step(3, false);
const E0 = T0.m.engine.E;
const surfaces = E0.places.filter(pl => !pl.floor && /desk_top|side_table/.test(pl.id)).map(pl => pl.id);
const props = E0.P.filter(p => p.def.movable !== false && !(p.F && p.F.furniture) && !/hat|glasses|crown|headphone/.test(p.id)).map(p => p.id);
check('the scene has the desk and the side table and movable props', surfaces.length === 2 && props.length >= 4, {surfaces, props});
let tried = 0;
for (const id of props) for (const pl of surfaces) {
  const T = makeMuse(30 + tried++); T.step(5, false);
  const E = T.m.engine.E, p = E.P.find(q => q.id === id);
  if (p.where === 'home' && p.homePlace === pl) continue;   // already there
  const r = T.m.command(O('place', p.name, {place: pl}), {src: 'matcher'}); runIdle(T, 60); T.step(5, false);
  if (!r.ok) { console.log(`skip ${id} -> ${pl}: ${r.reason}`); continue; }
  const st = T.m.state().props.find(q => q.id === id), plc = E.places.find(x => x.id === pl);
  const onTop = Math.abs(p.pos[1] - p.restH - plc.top_y) < 0.02;
  const ev = T.ev.filter(e => e.kind === 'object_event' && e.phase === 'end' && e.args && e.args.object === id).pop();
  check(`place ${id} on ${pl}: rests on its top, recorded as ${pl}, said on ${plc.name}`,
    onTop && (p.place === pl || (p.sup && p.place === pl)) && (st.where === pl || st.where === 'home' || /^on /.test(st.where)) && st.where !== 'floor' && (!ev || !/the floor/.test(ev.did)),
    {did: r.did, place: p.place, where: st.where, pos: p.pos.map(v => +v.toFixed(3)), ev: ev && ev.did});
}
{ // several things onto the desk in one session (the spots fill up, the near-edge spots get used): each records the desk
  const T = makeMuse(50); T.step(5, false); const E = T.m.engine.E, out = [];
  for (const id of ['chalk', 'eraser', 'ball', 'book', 'block_a', 'block_b']) {
    const p = E.P.find(q => q.id === id); if (!p) continue;
    const r = T.m.command(O('place', p.name, {place: 'desk_top'}), {src: 'matcher'}); runIdle(T, 60); T.step(3, false);
    if (!r.ok) continue;
    const onTop = Math.abs(p.pos[1] - p.restH - 0.4) < 0.02 || !!p.sup;
    out.push({id, place: p.place, where: T.m.state().props.find(q => q.id === id).where, onTop});
  }
  check('several things onto the desk in turn: every one resting there records the desk', out.length >= 3 && out.every(o => !o.onTop || (o.place === 'desk_top' && o.where !== 'floor')), out);
}
{ // the floor stays the floor
  const T = makeMuse(9); T.step(5, false);
  const r = T.m.command(O('place', 'book', {place: 'floor'}), {src: 'matcher'}); runIdle(T, 60); T.step(5, false);
  const st = T.m.state().props.find(q => q.id === 'book');
  check('place the book on the floor: recorded as floor', r.ok && st.where === 'floor', {did: r.did, where: st.where});
}
console.log(`\n${n - fails}/${n} passed`);
process.exit(fails);
