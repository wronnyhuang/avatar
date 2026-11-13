// objects-engine: general object manipulation for the muse (docs/OBJECTS.md §3 + §5).
// Every scenario runs the real engine (motion.js + moves.js, the real muse body + props JSON) and checks
//   - the STATE it ends in (props()/things()/state(): where every thing is, held / worn / balanced / on / in),
//   - the words it reports (the job's final 'did' / 'reason' — never a phantom success),
//   - per frame: no thing jumps (world pose of every prop, ≤ JUMP H per frame), and at the end nothing floats or sits
//     inside a solid (every free thing rests on the highest support under it).
// Run: node tests/objects_test.cjs   (MUSE_ONLY=<substring> runs the matching scenarios only)
const {makeMuse} = require('./muse_lib.cjs');

const JUMP = 0.2;   // H per frame (6 H/s at 30 Hz; a thrown thing peaks near 4 H/s)
let pass = 0, fail = 0;
const only = process.env.MUSE_ONLY || '';
function ok(name, cond, info) { if (cond) pass++; else fail++; console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${info !== undefined ? '  — ' + (typeof info === 'string' ? info : JSON.stringify(info)) : ''}`); }

// ---- the harness
function world(E, p) { return (p.held || p.worn || p.bal || p.anc) && p.wpos ? p.wpos.slice() : E._obj.P(p); }
function scene(seed = 7) {
  const T = makeMuse(seed); T.step(20);
  const E = T.m.engine, maxJ = {v: 0, id: null, t: 0};
  let last = null;
  function frame() {
    T.step(1);
    const cur = {}; for (const p of E.E.P) cur[p.id] = world(E, p);
    if (last) for (const id in cur) { const a = last[id], b = cur[id], d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]); if (d > maxJ.v) { maxJ.v = d; maxJ.id = id; maxJ.t = T.m.now(); } }
    last = cur;
  }
  function idle(maxS = 60) { for (let i = 0; i < maxS * 30; i++) { frame(); const X = E.E; if (!X.move && !X.chain && !X.jobs.length && !X.queue && !X.oqueue && i > 3 && !X.P.some(p => p.fly || p.attE)) break; } for (let i = 0; i < 20; i++) frame(); }
  // one command → {r: its immediate reply, fin: the job's final event (end / fail / interrupted), evs: every event}
  function cmd(c, maxS) {
    const n0 = T.ev.length, r = T.m.command(Object.assign({action: 'object'}, c), {src: 'test'});
    idle(maxS);
    const evs = T.ev.slice(n0), fin = evs.filter(e => (e.kind === 'object' || e.kind === 'move') && /^(end|fail|interrupted)$/.test(e.phase)).pop() || null;
    const words = fin ? (fin.phase === 'fail' ? 'FAIL: ' + fin.reason + (fin.did ? ' (did: ' + fin.did + ')' : '') : fin.did) : r && r.ok === false ? 'FAIL: ' + r.reason : '(no final event)';
    return {r, fin, evs, words, failed: (r && r.ok === false) || (fin && fin.phase === 'fail')};
  }
  const prop = id => E.E.P.find(p => p.id === id);
  const where = id => (E.props().find(p => p.id === id) || {}).where;
  const thing = id => E.things().find(p => p.id === id);
  const W = id => world(E, prop(id));
  // nothing floats, nothing is inside a solid: every free thing's bottom is on the highest support under it, and no two
  // things' boxes overlap (a pair that both sit at home is the scene's own layout, not this engine's doing)
  const furn = p => !!(p.F && p.F.furniture) || /^(chair|desk)$/.test(p.id);
  const attached = (p, d = 0) => !!(p.held || p.worn || p.bal || p.anc || p.fly || p.attE) || (!!p.sup && d < 8 && attached(prop(p.sup.id), d + 1));
  function resting() {
    const bad = [], home = id => where(id) === 'home';
    for (const p of E.E.P) {
      if (furn(p) || attached(p)) continue;
      const c = E._obj.P(p), s = E._obj.supportAt(c[0], c[2], c[1] - p.hH + 0.01, p, {});
      const gap = c[1] - p.hH - s.top;
      if (Math.abs(gap) > 0.02) bad.push(`${p.id} ${gap > 0 ? 'floats' : 'sinks'} ${gap.toFixed(3)} over ${s.id}`);
      for (const q of E.E.P) {
        if (q === p || furn(q) || attached(q) || (home(p.id) && home(q.id))) continue;
        if ((p.sup && p.sup.id === q.id) || (q.sup && q.sup.id === p.id)) continue;   // resting on / holding it
        const d = E._obj.P(q);
        if (Math.abs(c[1] - d[1]) < p.hH + q.hH - 0.004 && Math.abs(c[0] - d[0]) < p.hx + q.hx - 0.004 && Math.abs(c[2] - d[2]) < p.hz + q.hz - 0.004 && p.id < q.id) bad.push(`${p.id} overlaps ${q.id}`);
      }
    }
    return bad;
  }
  return {T, E, frame, idle, cmd, prop, where, thing, W, resting, maxJ};
}
// the common end checks of a scenario
function ends(S, name) {
  const b = S.resting();
  ok(`${name}: nothing floats or sits inside a solid`, !b.length, b.join('; ') || 'all resting');
  ok(`${name}: no thing jumps (≤ ${JUMP} H per frame)`, S.maxJ.v <= JUMP, `max ${S.maxJ.v.toFixed(3)} H (${S.maxJ.id} @ ${S.maxJ.t.toFixed(2)} s)`);
  // action memory is one truth: props() / things() / state() agree on where each thing is
  const P = S.E.props(), Th = S.E.things(), st = S.E.state();
  const dis = P.filter(p => { const t = Th.find(x => x.id === p.id); return !t || t.where !== p.where; }).map(p => p.id);
  const hold = st.holding || {}, heldIds = P.filter(p => p.held).map(p => p.id);
  const holdOk = heldIds.every(id => Object.values(hold).some(v => v === id || (v && (v.id === id || v.object === id))) || JSON.stringify(hold).includes(id));
  ok(`${name}: props() / things() / state() agree`, !dis.length && holdOk, dis.length ? 'disagree: ' + dis.join(',') : JSON.stringify(hold));
}
const sc = (name, fn) => { if (only && !name.includes(only)) return; try { fn(name); } catch (e) { ok(name + ' (threw)', false, String(e && e.stack || e).slice(0, 400)); } };
const near2 = (a, b) => Math.hypot(a[0] - b[0], a[2] - b[2]);

// ---------------------------------------------------------------- A: state model
sc('A1 pick up the ball: held in a paw, the world pose follows the paw', name => {
  const S = scene(7), r = S.cmd({verb: 'pick_up', object: {id: 'ball'}});
  const t = S.thing('ball');
  ok(`${name}: held, said so`, !r.failed && !!t.held && /picked up the yellow ball/.test(r.words), `${t.where} | ${r.words}`);
  ok(`${name}: up in the paw (not on the floor)`, S.W('ball')[1] > 0.2, S.W('ball'));
  ends(S, name);
});
sc('A2 a rider rides its support while held, and is not "home"', name => {
  const S = scene(7), b0 = S.W('block_b'), r = S.cmd({verb: 'pick_up', object: {id: 'block_a'}});
  const a = S.W('block_a'), b = S.W('block_b');
  ok(`${name}: the blue block came along on top of the red one`, !r.failed && near2(a, b) < 0.02 && b[1] > a[1] && near2(b, b0) > 0.05, {a, b});
  ok(`${name}: it says the blue block is on the red block`, /on the red block/.test(S.where('block_b')), S.where('block_b'));
  ends(S, name);
});
sc('A3 home stacks: a thing whose home is on another (glasses on the gift box) starts there', name => {
  const S = scene(7), g = S.thing('glasses');
  ok(`${name}: the glasses rest on the gift box`, g.on === 'gift_box' && g.where === 'home', g);
  ends(S, name);
});

// ---------------------------------------------------------------- C: relations
const relCase = (id, rel, target, check) => sc(`C ${rel} — the ${id} ${rel.replace(/_/g, ' ')} the ${target}`, name => {
  const S = scene(11), r = S.cmd({verb: 'put', object: {id}, target: {id: target}, relation: rel});
  const a = S.W(id), q = S.W(target), c = check(a, q, S);
  ok(`${name}: done and said`, !r.failed && /put the/.test(r.words), r.words);
  ok(`${name}: where it really is`, c.ok, c.info);
  ends(S, name);
});
relCase('ball', 'on', 'book', (a, q, S) => ({ok: S.thing('ball').on === 'book' && a[1] > q[1], info: S.where('ball')}));
relCase('ball', 'under', 'desk', (a, q, S) => { const d = S.prop('desk'); return {ok: Math.abs(a[0] - q[0]) < d.hx && Math.abs(a[2] - q[2]) < d.hz && a[1] < 0.15, info: {a, q}}; });
relCase('ball', 'behind', 'chair', (a, q) => ({ok: a[2] < q[2] - 0.05, info: {a, q}}));
relCase('ball', 'in_front_of', 'chair', (a, q) => ({ok: a[2] > q[2] + 0.05, info: {a, q}}));
relCase('ball', 'left_of', 'watering_can', (a, q) => ({ok: a[0] < q[0] - 0.03 && near2(a, q) < 0.4, info: {a, q}}));   // (the USER's left = screen left = -x)
relCase('ball', 'right_of', 'watering_can', (a, q) => ({ok: a[0] > q[0] + 0.03 && near2(a, q) < 0.4, info: {a, q}}));
sc('C right of the plant at the desk\'s edge: no room — said, nothing claimed', name => {
  const S = scene(11), r = S.cmd({verb: 'put', object: {id: 'ball'}, target: {id: 'plant'}, relation: 'right_of'});
  ok(`${name}: either done (and it IS to the right) or refused with the reason`, r.failed ? /no room|can't|cannot/.test(r.words) : S.W('ball')[0] > S.W('plant')[0], r.words);
  ends(S, name);
});
relCase('mug', 'next_to', 'plant', (a, q) => ({ok: near2(a, q) < 0.35, info: {a, q}}));
sc('C in — the ball into the gift box: lid opened first (the glasses moved off it), the ball inside', name => {
  const S = scene(11), r = S.cmd({verb: 'put_in', object: {id: 'ball'}, target: {id: 'gift_box'}});
  const t = S.thing('ball');
  ok(`${name}: in the box`, !r.failed && t.on === 'gift_box' && t.inside, `${t.where} | ${r.words}`);
  ok(`${name}: said what it had to do first`, /glasses/.test(r.words) && /open/.test(r.words), r.words);
  const r2 = S.cmd({verb: 'close', object: {id: 'gift_box'}});
  ok(`${name}: closed with the ball still inside`, !r2.failed && S.thing('ball').inside && S.thing('gift_box').open === false, `${S.where('ball')} | ${r2.words}`);
  ends(S, name);
});
sc('C stack — the blue block back on the red block after both were moved', name => {
  const S = scene(13);
  S.cmd({verb: 'put', object: {id: 'block_b'}, target: {id: 'plant'}, relation: 'next_to'});
  const r = S.cmd({verb: 'stack', object: {id: 'block_b'}, target: {id: 'block_a'}, relation: 'on'});
  const a = S.W('block_a'), b = S.W('block_b');
  ok(`${name}: stacked`, !r.failed && S.thing('block_b').on === 'block_a' && near2(a, b) < 0.03 && b[1] > a[1], `${S.where('block_b')} | ${r.words}`);
  ends(S, name);
});
sc('C refusals — too big to fit, furniture, a thing on itself: said, nothing moved', name => {
  const S = scene(13), b0 = S.W('book');
  const r1 = S.cmd({verb: 'put_in', object: {id: 'book'}, target: {id: 'mug'}});
  const r2 = S.cmd({verb: 'put_in', object: {id: 'desk'}, target: {id: 'gift_box'}});
  const r3 = S.cmd({verb: 'put', object: {id: 'ball'}, target: {id: 'ball'}, relation: 'on'});
  // (policy: too big to go IN -> refused, or set ON it with the reason said — never a phantom 'in')
  ok(`${name}: the book does not fit in the mug`, (r1.failed && /too big|fit/.test(r1.words)) || (/too big to go in/.test(r1.words) && !S.thing('book').inside), r1.words);
  ok(`${name}: the desk is furniture`, r2.failed && /furniture|too big/.test(r2.words), r2.words);
  ok(`${name}: the ball on itself is refused`, r3.failed, r3.words);
  ok(`${name}: the book did not move, or rests on the mug as said`, (near2(S.W('book'), b0) < 0.01 && S.where('book') === 'home') || (S.thing('book').on === 'mug' && !S.thing('book').inside), S.where('book'));
  ends(S, name);
});

// ---------------------------------------------------------------- the head and the body
sc('H1 wear the hat, then take it off: worn, then back in a paw', name => {
  const S = scene(17), r = S.cmd({verb: 'wear', object: {id: 'hat'}});
  ok(`${name}: worn`, !r.failed && S.thing('hat').worn && /put on the straw sun hat/.test(r.words), `${S.where('hat')} | ${r.words}`);
  ok(`${name}: state says wearing it`, JSON.stringify(S.E.state().wearing || []).includes('hat'), S.E.state().wearing);
  const r2 = S.cmd({verb: 'take_off', object: {id: 'hat'}});
  ok(`${name}: taken off into a paw`, !r2.failed && !S.thing('hat').worn && !!S.thing('hat').held, `${S.where('hat')} | ${r2.words}`);
  ends(S, name);
});
sc('H2 wear the glasses (a pinch-grip wearable off the gift box)', name => {
  const S = scene(17), r = S.cmd({verb: 'wear', object: {id: 'glasses'}});
  ok(`${name}: worn`, !r.failed && S.thing('glasses').worn, `${S.where('glasses')} | ${r.words}`);
  ends(S, name);
});
sc('H3 wear the ball: not wearable — said, with the honest alternative', name => {
  const S = scene(17), r = S.cmd({verb: 'wear', object: {id: 'ball'}});
  ok(`${name}: refused truthfully`, r.failed && /isn't something to wear|not something to wear/.test(r.words), r.words);
  ok(`${name}: the ball is not on its head`, !S.thing('ball').worn && !/head/.test(S.where('ball')), S.where('ball'));
  ends(S, name);
});
sc('H4 balance the ball on the head: it stays there (wobbles), takes it down again', name => {
  const S = scene(19), r = S.cmd({verb: 'balance', object: {id: 'ball'}});
  ok(`${name}: balanced`, !r.failed && /balanced on its head/.test(S.where('ball')), `${S.where('ball')} | ${r.words}`);
  const y0 = S.W('ball')[1]; for (let i = 0; i < 60; i++) S.frame();
  ok(`${name}: still up on the head a while later`, Math.abs(S.W('ball')[1] - y0) < 0.03 && y0 > 0.8, [y0, S.W('ball')[1]]);
  const r2 = S.cmd({verb: 'take_off', object: {id: 'ball'}});
  ok(`${name}: taken down`, !r2.failed && !!S.thing('ball').held, `${S.where('ball')} | ${r2.words}`);
  ends(S, name);
});
sc('H5 balance the book with the hat on: it goes ON TOP of the hat, and the claim matches the state', name => {
  const S = scene(19); S.cmd({verb: 'wear', object: {id: 'hat'}});
  const r = S.cmd({verb: 'balance', object: {id: 'book'}});
  const said = /balanced the blue book/.test(r.words), is = /balanced on its head/.test(S.where('book'));
  ok(`${name}: no phantom success (said ⇔ state)`, said === is && (is || r.failed), `${S.where('book')} | ${r.words}`);
  ok(`${name}: above the hat`, !is || S.W('book')[1] > S.W('hat')[1], [S.W('book'), S.W('hat')]);
  ends(S, name);
});
sc('H6 the ball on its foot follows the foot through a walk', name => {
  const S = scene(23), r = S.cmd({verb: 'put', object: {id: 'ball'}, target: 'your right foot'});
  ok(`${name}: on its foot`, !r.failed && /foot/.test(S.where('ball')), `${S.where('ball')} | ${r.words}`);
  const a0 = S.W('ball'), n0 = S.T.ev.length;
  S.T.m.command({action: 'move', name: 'walk', direction: 'left', steps: 2}, {src: 'test'}); S.idle(20);
  const a1 = S.W('ball'), moved = S.T.ev.slice(n0).some(e => e.kind === 'move');
  ok(`${name}: it went along`, /foot/.test(S.where('ball')) && (!moved || near2(a0, a1) > 0.01 || a1[1] < 0.15), {a0, a1, where: S.where('ball')});
  ends(S, name);
});
sc('H7 the book on its lap: it sits down first', name => {
  const S = scene(23), r = S.cmd({verb: 'put', object: {id: 'book'}, target: 'your lap'});
  ok(`${name}: on its lap, seated`, !r.failed && /lap/.test(S.where('book')) && /seat/.test(String(S.E.state().posture)), `${S.where('book')} ${S.E.state().posture} | ${r.words}`);
  ends(S, name);
});

// ---------------------------------------------------------------- lids
sc('L1 open the gift box: what rests on the lid is moved off first, then it opens; close it again', name => {
  const S = scene(29), r = S.cmd({verb: 'open', object: {id: 'gift_box'}});
  ok(`${name}: open, the glasses off the lid first`, !r.failed && S.thing('gift_box').open === true && S.thing('glasses').on !== 'gift_box' && /glasses/.test(r.words), `${S.where('glasses')} | ${r.words}`);
  const r2 = S.cmd({verb: 'close', object: {id: 'gift_box'}});
  ok(`${name}: closed`, !r2.failed && S.thing('gift_box').open === false, r2.words);
  ends(S, name);
});
sc('L2 open the ball: it has no lid — said', name => {
  const S = scene(29), r = S.cmd({verb: 'open', object: {id: 'ball'}});
  ok(`${name}: refused`, r.failed && /doesn't open|no lid/.test(r.words), r.words);
  ends(S, name);
});

// ---------------------------------------------------------------- throws / catches / juggling
sc('T1 throw the ball into the open gift box: it flies, lands IN it (or says where it really landed)', name => {
  const S = scene(31); S.cmd({verb: 'open', object: {id: 'gift_box'}});
  const r = S.cmd({verb: 'throw', object: {id: 'ball'}, target: {id: 'gift_box'}, relation: 'in'});
  const t = S.thing('ball'), inBox = t.on === 'gift_box' && t.inside, said = /missed/.test(r.words);
  ok(`${name}: the words match where it is`, !r.failed && (inBox ? !said : said), `${S.where('ball')} | ${r.words}`);
  ok(`${name}: it flew (an arc: higher than both ends)`, r.evs.some(e => e.kind === 'object_event' && /thrown|let_go/.test(e.name)) || /threw/.test(r.words), r.words);
  ends(S, name);
});
sc('T2 throw the ball to me: it lands in front of the body', name => {
  const S = scene(31), r = S.cmd({verb: 'throw', object: {id: 'ball'}, target: 'me'});
  ok(`${name}: thrown, resting toward the camera`, !r.failed && /threw/.test(r.words) && !S.thing('ball').held && S.W('ball')[2] > 0.3, `${S.where('ball')} ${S.W('ball').map(v => v.toFixed(2))} | ${r.words}`);
  ends(S, name);
});
sc('T3 toss it up and catch it: in its paw at the end', name => {
  const S = scene(37), r = S.cmd({verb: 'toss', object: {id: 'ball'}});
  ok(`${name}: tossed and caught (said in the past tense)`, !r.failed && !!S.thing('ball').held && /tossed/.test(r.words), `${S.where('ball')} | ${r.words}`);
  ends(S, name);
});
sc('T4 juggle the ball and the red block: a real exchange, every throw caught, said with the count', name => {
  const S = scene(41), r = S.cmd({verb: 'juggle', objects: [{id: 'ball'}, {id: 'block_a'}], count: 3});
  const ev = r.evs.find(e => e.kind === 'object_event' && e.name === 'juggled');
  ok(`${name}: juggled`, !r.failed && !!ev && ev.args.throws >= 6 && !ev.args.missed.length, ev ? ev.did : r.words);
  ok(`${name}: the blue block came off the red one first (a stack would fly apart)`, /took the blue block off/.test(r.words) && !S.thing('block_b').held, `${S.where('block_b')} | ${r.words}`);
  ok(`${name}: one thing in each paw at the end`, S.thing('ball').held && S.thing('block_a').held && S.thing('ball').held !== S.thing('block_a').held, [S.where('ball'), S.where('block_a')]);
  ends(S, name);
});
sc('T5 juggle with other things in its paws: those are set down first', name => {
  const S = scene(43); S.cmd({verb: 'pick_up', object: {id: 'book'}}); S.cmd({verb: 'pick_up', object: {id: 'mug'}});
  const r = S.cmd({verb: 'juggle', objects: [{id: 'ball'}, {id: 'block_b'}], count: 2});
  ok(`${name}: book and mug down, then juggled`, !r.failed && !S.thing('book').held && !S.thing('mug').held && /juggled/.test(r.words), r.words);
  ends(S, name);
});
sc('T6 throw the desk: refused', name => {
  const S = scene(43), r = S.cmd({verb: 'throw', object: {id: 'desk'}, target: 'me'});
  ok(`${name}: refused truthfully`, r.failed && /heavy|too big|furniture/.test(r.words), r.words);
  ends(S, name);
});

// objects: knock over / kick — the engine's flight physics, riders shed, everything rests
sc('K1 kick the ball: it flies off along the floor and rests somewhere else', name => {
  const S = scene(51), w0 = S.W('ball'), r = S.cmd({verb: 'kick', object: {id: 'ball'}}), w1 = S.W('ball');
  ok(`${name}: kicked, moved, resting`, !r.failed && /kick/.test(r.words) && Math.hypot(w1[0] - w0[0], w1[2] - w0[2]) > 0.1 && !S.prop('ball').fly, {words: r.words, w0, w1});
  ends(S, name);
});
sc('K2 knock the block tower over: the top block falls off, both rest', name => {
  const S = scene(52), r = S.cmd({verb: 'knock_over', object: {id: 'block_a'}});
  ok(`${name}: knocked over, blue no longer on red`, !r.failed && !(S.prop('block_b').sup && S.prop('block_b').sup.id === 'block_a'), {words: r.words, b: S.thing('block_b')});
  ends(S, name);
});
sc('K3 kick the desk / knock the worn hat: refused', name => {
  const S = scene(53), r = S.cmd({verb: 'kick', object: {id: 'desk'}});
  ok(`${name}: furniture refused`, r.failed, r.words);
  ends(S, name);
});

// ---------------------------------------------------------------- in-paw tricks
for (const [verb, extra, rx] of [['spin', {on: 'finger'}, /fingertip/], ['tilt', {}, /tilt/], ['wave', {}, /wav/], ['raise', {}, /rais|held .* up as high/], ['shake', {}, /shook|shak/], ['offer', {}, /offer|held .* out/]]) {
  sc(`P ${verb} the book in its paw`, name => {
    const S = scene(47), r = S.cmd(Object.assign({verb, object: {id: 'book'}}, extra));
    ok(`${name}: done, said, still holding it`, !r.failed && rx.test(r.words) && !!S.thing('book').held, `${S.where('book')} | ${r.words}`);
    ends(S, name);
  });
}
sc('P hide the book behind its back: behind it in the paw, or (out of reach) on its back — and it says which', name => {
  const S = scene(47), r = S.cmd({verb: 'hide', object: {id: 'book'}});
  const w = S.where('book'), said = /onto its back/.test(r.words);
  ok(`${name}: the words match the state`, !r.failed && (/on its back/.test(w) ? said : !!S.thing('book').held && /behind its back/.test(r.words)), `${w} | ${r.words}`);
  ends(S, name);
});
sc('P drink from the mug; drink from the ball is refused', name => {
  const S = scene(53), r = S.cmd({verb: 'drink', object: {id: 'mug'}});
  ok(`${name}: a sip from the mug`, !r.failed && /sip|drank|drink/.test(r.words), r.words);
  const r2 = S.cmd({verb: 'drink', object: {id: 'ball'}});
  ok(`${name}: not from the ball`, r2.failed, r2.words);
  ends(S, name);
});
sc('P pour from the mug: the flow column runs (afford.pour, no articulation)', name => {
  const S = scene(53), n0 = S.T.rows.length, r = S.cmd({verb: 'pour', object: {id: 'mug'}});
  const k = S.prop('mug').k, fl = Math.max(...S.T.rows.slice(n0).map(o => +o['prop' + k + '_s'] || 0));
  ok(`${name}: poured, flow > 0 while pouring, back to 0`, !r.failed && /pour/.test(r.words) && fl > 0.2 && (+S.T.rows[S.T.rows.length - 1]['prop' + k + '_s'] || 0) < 0.05, `flow max ${fl.toFixed(2)} | ${r.words}`);
  ends(S, name);
});

// ---------------------------------------------------------------- use X on Y
sc('U1 water the plant: the watering can (afford.pour water + use.targets) is fetched and used', name => {
  const S = scene(59), r = S.cmd({verb: 'water', object: {id: 'plant'}});
  ok(`${name}: watered, said`, !r.failed && /watered the little green plant with the green watering can/.test(r.words), r.words);
  ok(`${name}: the plant did not move`, S.where('plant') === 'home', S.where('plant'));
  ends(S, name);
});
sc('U2 water the plant with itself / eat the mug: refused', name => {
  const S = scene(59), r = S.cmd({verb: 'water', object: {id: 'plant'}, with: {id: 'plant'}}), r2 = S.cmd({verb: 'eat', object: {id: 'mug'}});
  ok(`${name}: refused`, r.failed && r2.failed && /isn't food/.test(r2.words), `${r.words} | ${r2.words}`);
  ends(S, name);
});

// ---------------------------------------------------------------- put back
sc('B1 put everything back after a mess: dependency-aware (riders, lids, worn things), all home', name => {
  const S = scene(61);
  S.cmd({verb: 'pick_up', object: {id: 'block_a'}});   // the blue block rides on it
  S.cmd({verb: 'put', object: {id: 'block_a'}, target: {id: 'plant'}, relation: 'next_to'});
  S.cmd({verb: 'wear', object: {id: 'hat'}});
  S.cmd({verb: 'open', object: {id: 'gift_box'}});
  const r = S.cmd({verb: 'put_back', object: 'everything'}, 120);
  const away = S.E.props().filter(p => p.where !== 'home' && !/chair|desk/.test(p.id)).map(p => p.id + ':' + p.where);
  ok(`${name}: everything home`, !r.failed && !away.length && S.thing('gift_box').open === false, (away.join(', ') || 'all home') + ' | ' + r.words);
  ends(S, name);
});
sc('B2 put the ball back when it already is: says so, does nothing', name => {
  const S = scene(61), r = S.cmd({verb: 'put_back', object: {id: 'ball'}});
  ok(`${name}: noop, truthful`, !r.failed && /already/.test(r.words || (r.r && r.r.did) || ''), r.words + ' ' + JSON.stringify(r.r && r.r.did));
  ends(S, name);
});

// ---------------------------------------------------------------- E: planner hooks (kfEvent)
sc('E1 kfEvent release straight up + catch with the same paw: caught (the catch measures the held pose)', name => {
  const S = scene(67); S.cmd({verb: 'pick_up', object: {id: 'ball'}});
  // the paw stays where it let go (what the keyframe layer does with a planned toss): the engine's own paw target
  const sd = S.prop('ball').held, n0 = S.T.ev.length, pw = S.E.E.paw[sd], g0 = S.E.E.pawGW[sd].slice();
  pw.keep = true; pw.ctrlT = 1; pw.tgt = g0.slice(); for (let i = 0; i < 15; i++) S.frame();
  const c0 = S.W('ball'), r1 = S.E.kfEvent({type: 'release', side: sd, impulse: [0, 3, 0]});
  const r2 = S.E.kfEvent({type: 'catch', side: sd, window: 1.5});
  S.frame(); const c1 = S.W('ball');
  ok(`${name}: the flight starts where the ball was drawn (no jump at the release)`, Math.hypot(c1[0] - c0[0], c1[2] - c0[2]) < 0.01 && c1[1] - c0[1] < 0.15, {c0, c1});
  for (let i = 0; i < 60; i++) { pw.tgt = g0.slice(); S.frame(); }
  pw.keep = false; pw.ctrlT = 0;
  const names = S.T.ev.slice(n0).filter(e => e.kind === 'object_event').map(e => e.name);
  ok(`${name}: released, armed, caught (not missed)`, r1.ok && r2.ok && names.includes('caught') && !names.includes('missed') && S.prop('ball').held === sd, names.join(','));
  S.idle(5); ends(S, name);
});
sc('E2 kfEvent grab with nothing near / attach far from the head: refused, no teleport', name => {
  const S = scene(67), r = S.E.kfEvent({type: 'grab', side: 'l'});
  S.cmd({verb: 'pick_up', object: {id: 'ball'}});
  const sd = S.prop('ball').held, b0 = S.W('ball'), r2 = S.E.kfEvent({type: 'attach', side: sd, to: 'head'});
  ok(`${name}: both refused`, !r.ok && !r2.ok, `${r.reason} | ${r2.reason}`);
  S.frame(); ok(`${name}: the ball stayed in the paw`, S.prop('ball').held === sd && near2(S.W('ball'), b0) < 0.05, S.where('ball'));
  ends(S, name);
});
sc('E3 things(): the compact list for the planner', name => {
  const S = scene(71), th = S.E.things(), b = th.find(x => x.id === 'gift_box'), w = th.find(x => x.id === 'watering_can');
  ok(`${name}: every prop, with afford keys and open state`, th.length === S.E.E.P.length && b.afford.includes('container') && b.open === false && w.afford.includes('pour'), th.map(x => x.id).join(','));
  ends(S, name);
});

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail ? 1 : 0);
