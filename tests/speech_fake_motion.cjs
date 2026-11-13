// speech: a CONTRACT-SHAPED stand-in for the motion engine (docs/INTENT-RULES.md §3–4), for the
// action-memory harness only: command() → {ok,id,did,reason}, onAction start/end/fail/interrupted with progress,
// state() {posture, facing, looking, holding{left,right} (OWN sides), props[{id,name,where}], held, active}.
// Timings are plausible (a spin 1.2 s, a pick-up 1.5 s); failures follow the contract's examples.
'use strict';
const PROPS = [
  {id: 'ball', name: 'red ball', color: 'red', category: 'toy', home: 'table'},
  {id: 'cup', name: 'blue cup', color: 'blue', category: 'cup', home: 'table'},
  {id: 'book', name: 'yellow book', color: 'yellow', category: 'book', home: 'table'},
  {id: 'flower', name: 'pink flower', color: 'pink', category: 'flower', home: 'table'},
];
// the muse's desk scene (contract §0): furniture is free (never paw-attached), headphones are worn, the laptop has a lid
const MUSE_PROPS = [
  {id: 'headphones', name: 'headphones', category: 'headphones', home: 'side table', wearable: true},
  {id: 'chair', name: 'chair', category: 'chair', home: 'by the wall', furniture: true},
  {id: 'desk', name: 'desk', category: 'desk', home: 'by the wall', furniture: true},
  {id: 'laptop', name: 'laptop', category: 'laptop', home: 'desk', openable: true},
  {id: 'mug', name: 'green mug', color: 'green', category: 'mug', home: 'desk'},
];
const GEST_DUR = {wave: c => 0.6 + 0.8 * c, nod: c => 0.7 * c, shake_head: c => 0.8 * c, clap: c => 0.5 + 0.4 * c, thumbs_up: () => 1.6,
  shrug: () => 1.4, bow: () => 1.8, point: () => 2.0, wink: () => 0.6, jump: c => 0.9 * c, dance: () => 4, raise_paw: () => 2.2,
  cover_eyes: () => 2.4, hand_on_heart: () => 2.4, think: () => 2.6, stick_tongue_out: () => 1.6, ear_wiggle: () => 1.2,
  tail_wag: () => 2.5, look_around: () => 3, blink: () => 0.3, nose_wiggle: () => 1};
const MOVE_DUR = {spin: c => 1.2 * c, twirl: c => 1.0 * c, turn: () => 0.8, turn_away: () => 0.9, turn_back: () => 0.9, backflip: c => 1.4 * c,
  frontflip: c => 1.4 * c, flip: c => 1.4 * c, jump: c => 0.9 * c, hop: c => 0.6 * c, walk: c => 0.6 * c, step: c => 0.6 * c, dance: () => 5,
  sit: () => 1.2, sit_down: () => 1.2, stand: () => 1.4, stand_up: () => 1.4, get_up: () => 1.4, lie_down: () => 1.5, roll_over: () => 1.6,
  cartwheel: c => 1.5 * c, bow: () => 1.8, wiggle: () => 1.5, shake_off: () => 1.2};
const OBJ_DUR = {put_on: 2.2, take_off: 1.8, pull: 2.4, push: 2.2, slide: 1.4, bring: 2.6, sit_at: 1.8, sit_in: 1.6, stand_up: 1.4, open: 1.2,
  close: 1.0, type_on: 600, work_on: 600, use: 600, turn_back: 0.9, pick_up: 1.5, put_down: 1.3, put_back: 1.6, place: 1.6, hold_up: 2.0, show: 2.0, inspect: 2.5, look_at: 2.0, turn: 1.4,
  flip: 1.2, hand_over: 1.2, toss: 1.6, drop: 0.6, shake: 1.4, hug: 2.0, offer: 2.0, give: 2.0};
const CNT = n => n === 1 ? 'once' : n === 2 ? 'twice' : n + ' times';

function createFakeMotion(o = {}) {
  const scene = o.scene === 'muse' ? MUSE_PROPS.concat(PROPS) : PROPS;
  const F = {t: 0, id: 0, active: [], posture: o.scene === 'muse' ? 'standing' : 'seated', seat: null, activity: null, facing: 'camera', looking: 'user', holding: {left: null, right: null},
    props: scene.map(p => Object.assign({}, p, {where: p.home}, p.openable ? {open: false} : {})), held: [], last: null, lookUntil: 0};
  const emit = ev => { if (o.onAction) o.onAction(Object.assign({t: F.t}, ev)); };
  const pname = id => (F.props.find(p => p.id === id) || {}).name || id;
  function resolveObj(d) {
    if (!d) d = {ref: 'it'};
    if (typeof d === 'string') d = {name: d};
    const ref = d.ref;
    if (ref === 'it' || ref === 'that') { const h = F.holding.right || F.holding.left || F.last; return h ? F.props.find(p => p.id === h) : null; }
    const want = [d.id, d.name, d.color, d.category].filter(Boolean).map(s => String(s).toLowerCase());
    let c = F.props.filter(p => want.every(w => p.id === w || p.name.includes(w) || p.color === w || p.category === w));
    if (ref === 'other' && F.last) c = c.filter(p => p.id !== F.last);
    return c[0] || null;
  }
  function finish(a, phase, extra) {
    F.active = F.active.filter(x => x !== a);
    emit(Object.assign({kind: a.kind, name: a.name, src: a.src, phase, args: a.args, id: a.id, did: a.did}, extra || {}));
    if (phase === 'end' && a.onEnd) a.onEnd();
  }
  function start(kind, name, args, src, dur, did, onEnd) {
    const a = {id: ++F.id, kind, name, args, src, t0: F.t, dur, did, onEnd, count: Math.max(1, +args.count || +args.repeat || 1)};
    // a new whole-body move interrupts the current one; a gesture interrupts a gesture on the same paw
    for (const b of F.active.slice()) if ((kind === 'move' && b.kind === 'move') || (kind === 'gesture' && b.kind === 'gesture' && b.args.side === args.side))
      finish(b, 'interrupted', {progress: Math.floor((F.t - b.t0) / (b.dur / b.count)), reason: 'a new request replaced it'});
    F.active.push(a);
    emit({kind, name, src, phase: 'start', args, id: a.id, did});
    return {ok: true, id: a.id, did};
  }
  function command(cmd, src = 'tool') {
    const a = Object.assign({}, cmd); delete a.action;
    const fail = (reason, kind, name) => { emit({kind: kind || cmd.action, name: name || cmd.name || cmd.verb, src, phase: 'fail', args: a, reason}); return {ok: false, reason}; };
    switch (cmd.action) {
      case 'gesture': {
        const n = String(cmd.name || ''); const d = GEST_DUR[n]; if (!d) return fail("I don't know that gesture", 'gesture', n);
        const c = Math.max(1, +cmd.repeat || +cmd.count || 1);
        return start('gesture', n, Object.assign(a, {count: c}), src, d(c), n.replace(/_/g, ' ') + (c > 1 ? ' ' + CNT(c) : ''));
      }
      case 'move': {
        const n = String(cmd.name || ''); const d = MOVE_DUR[n]; if (!d) return fail("I can't do that move", 'move', n);
        const c = Math.max(1, +cmd.count || 1);
        let onEnd = null;
        if (/^(sit|sit_down)$/.test(n)) onEnd = () => { F.posture = 'seated'; };
        if (/^(stand|stand_up|get_up)$/.test(n)) onEnd = () => { F.posture = 'standing'; };
        if (n === 'lie_down') onEnd = () => { F.posture = 'lying'; };
        if (n === 'turn_away') onEnd = () => { F.facing = 'away'; };
        if (n === 'turn_back') onEnd = () => { F.facing = 'camera'; };
        if (n === 'turn') onEnd = () => { F.facing = cmd.dir === 'right' ? 'right' : 'left'; };
        return start('move', n, Object.assign(a, {count: c}), src, d(c), n.replace(/_/g, ' ') + (c > 1 ? ' ' + CNT(c) : '') + (cmd.dir ? ' to its ' + cmd.dir : ''), onEnd);
      }
      case 'object': {
        const v = String(cmd.verb || '');
        if (v === 'turn_back' || v === 'stand_up' || v === 'sit') return command(Object.assign({}, cmd, {action: v === 'sit' ? 'object' : v, verb: v === 'sit' ? 'sit_at' : v, object: v === 'sit' ? {name: 'desk'} : cmd.object}), src);
        if (!OBJ_DUR[v]) return fail("I can't do that with it", 'object', v);
        const p = resolveObj(cmd.object);
        if (!p) return fail('there is no ' + (cmd.object && (cmd.object.name || [cmd.object.color, cmd.object.category].filter(Boolean).join(' ')) || 'such thing') + ' here', 'object', v);
        const heldBy = F.holding.left === p.id ? 'left' : F.holding.right === p.id ? 'right' : null;
        a.object = {id: p.id, name: p.name};
        // ---- the desk scene verbs (a tiny precondition check like the motion engine's planner)
        const stopTyping = () => { const t = F.active.find(x => x.name === 'type_on' || x.name === 'work_on' || x.name === 'use'); if (t) finish(t, 'end'); F.activity = null; };
        if (v === 'put_on') { if (!p.wearable) return fail("I can't wear the " + p.name, 'object', v); if (p.where === 'head') return fail('I am already wearing them', 'object', v);
          return start('object', v, a, src, OBJ_DUR[v], 'putting on the ' + p.name, () => { p.where = 'head'; }); }
        if (v === 'take_off') { if (p.where !== 'head') return fail("I'm not wearing the " + p.name, 'object', v);
          return start('object', v, a, src, OBJ_DUR[v], 'taking off the ' + p.name, () => { p.where = p.home; }); }
        if (v === 'pull' || v === 'push' || v === 'bring' || v === 'slide') {
          if (F.posture === 'seated' && p.furniture && p.id !== 'chair') return fail('I need to stand up first', 'object', v);
          const to = v === 'push' ? (p.home) : v === 'slide' ? (p.where === 'desk' ? 'desk' : p.where) : (p.id === 'chair' ? 'next to you' : 'in front of you');
          return start('object', v, a, src, OBJ_DUR[v], v + 'ing the ' + p.name, () => { p.where = to; if (p.id === 'desk') for (const q of F.props) if (q.where === 'desk' && q.id !== 'desk') q.where = 'desk'; });
        }
        if (v === 'sit_at' || v === 'sit_in') {
          const chair = F.props.find(q => q.id === 'chair'), desk = F.props.find(q => q.id === 'desk');
          if (chair && chair.where === chair.home) return fail('the chair is too far away', 'object', v);
          if (v === 'sit_at' && desk && desk.where === desk.home) return fail('the desk is too far away', 'object', v);
          return start('object', v, a, src, OBJ_DUR[v], 'sitting down at the ' + p.name, () => { F.posture = 'seated'; F.seat = v === 'sit_at' ? 'desk' : 'chair'; });
        }
        if (v === 'open' || v === 'close') { if (!p.openable) return fail("the " + p.name + " doesn't open", 'object', v);
          if ((v === 'open') === p.open) return fail('it is already ' + (p.open ? 'open' : 'closed'), 'object', v);
          if (v === 'close') stopTyping();
          return start('object', v, a, src, OBJ_DUR[v], v + 'ing the ' + p.name, () => { p.open = v === 'open'; }); }
        if (v === 'type_on' || v === 'work_on' || v === 'use') { if (p.openable && !p.open) return fail('the ' + p.name + ' is closed', 'object', v);
          F.activity = {name: v, object: p.name}; F.facing = 'desk';
          return start('object', v, a, src, OBJ_DUR[v], 'typing on the ' + p.name); }
        if (v === 'pick_up') {
          if (heldBy) return fail('I am already holding the ' + p.name, 'object', v);
          let side = cmd.side === 'left' || cmd.side === 'right' ? cmd.side : (!F.holding.right ? 'right' : !F.holding.left ? 'left' : null);
          if (!side || F.holding[side]) return fail('my paws are full', 'object', v);
          a.side = side;
          return start('object', v, a, src, OBJ_DUR[v], 'picking up the ' + p.name + ' with its ' + side + ' paw',
            () => { F.holding[side] = p.id; p.where = side + ' paw'; F.last = p.id; });
        }
        if (!heldBy && v !== 'inspect' && v !== 'look_at') return fail("I'm not holding the " + p.name, 'object', v);
        a.side = heldBy || a.side;
        F.last = p.id;
        if (v === 'put_down' || v === 'put_back' || v === 'place' || v === 'drop')
          return start('object', v, a, src, OBJ_DUR[v], v.replace(/_/g, ' ') + ' the ' + p.name,
            () => { F.holding[heldBy] = null; p.where = v === 'drop' ? 'floor' : v === 'place' && cmd.place ? String(cmd.place) : p.home; });
        if (v === 'hand_over') {
          const to = heldBy === 'left' ? 'right' : 'left';
          if (F.holding[to]) return fail('my other paw is full', 'object', v);
          return start('object', v, a, src, OBJ_DUR[v], 'passing the ' + p.name + ' to its ' + to + ' paw',
            () => { F.holding[heldBy] = null; F.holding[to] = p.id; p.where = to + ' paw'; });
        }
        return start('object', v, a, src, OBJ_DUR[v], v.replace(/_/g, ' ') + ' the ' + p.name);
      }
      case 'turn_back': case 'stand_up': {
        if (cmd.action === 'stand_up' && F.posture !== 'seated') return fail("I'm already standing", 'object', 'stand_up');
        const t = F.active.find(x => x.name === 'type_on' || x.name === 'work_on' || x.name === 'use'); if (t) finish(t, 'end'); F.activity = null;
        return start('object', cmd.action, a, src, OBJ_DUR[cmd.action], cmd.action === 'turn_back' ? 'turning back to him' : 'standing up',
          () => { F.facing = 'camera'; if (cmd.action === 'stand_up') { F.posture = 'standing'; F.seat = null; } });
      }
      case 'look': {
        F.looking = cmd.target || 'user';
        if (!cmd.hold) F.lookUntil = F.t + 2.5;
        emit({kind: 'look', name: F.looking, src, phase: 'start', args: a, id: ++F.id});
        return {ok: true, id: F.id, did: 'looking ' + F.looking};
      }
      case 'expression': {
        const id = ++F.id;
        emit({kind: 'expression', name: cmd.name, src, phase: 'start', args: a, id});
        setTimeout0(() => emit({kind: 'expression', name: cmd.name, src, phase: 'end', args: a, id}), cmd.hold ? 1e9 : 2.6);
        return {ok: true, id, did: 'making a ' + cmd.name + ' face'};
      }
      case 'release': case 'stop': {
        for (const b of F.active.slice()) if (b.kind !== 'object') finish(b, 'interrupted', {progress: Math.floor((F.t - b.t0) / (b.dur / b.count)), reason: 'he said stop'});
        F.looking = 'user';
        emit({kind: 'release', name: 'release', src, phase: 'start', args: a, id: ++F.id});
        return {ok: true, id: F.id, did: 'relaxing'};
      }
      case 'hold': emit({kind: 'hold', name: 'hold', src, phase: 'start', args: a, id: ++F.id}); return {ok: true, id: F.id, did: 'holding still'};
    }
    return fail("I can't do that");
  }
  const timers = [];
  function setTimeout0(fn, s) { timers.push({at: F.t + s, fn}); }
  function step(dt) {
    F.t += dt;
    for (const a of F.active.slice()) if (F.t - a.t0 >= a.dur) finish(a, 'end');
    for (let i = timers.length - 1; i >= 0; i--) if (F.t >= timers[i].at) { const x = timers.splice(i, 1)[0]; x.fn(); }
    if (F.lookUntil && F.t > F.lookUntil) { F.looking = 'user'; F.lookUntil = 0; }
  }
  function state() {
    return {t: F.t, posture: F.posture, seat: F.seat, activity: F.activity, facing: F.facing, looking: F.looking, gazeFree: false, holding: Object.assign({}, F.holding),
      props: F.props.map(p => Object.assign({id: p.id, name: p.name, where: p.where}, p.openable ? {open: p.open} : {})), held: [], active: F.active.map(a => ({kind: a.kind, name: a.name, since: a.t0}))};
  }
  function reflex(name) { const id = ++F.id; emit({kind: 'reflex', name, src: 'reflex', phase: 'start', id}); setTimeout0(() => emit({kind: 'reflex', name, src: 'reflex', phase: 'end', id}), 1); }
  return {command, step, state, reflex, F, props: PROPS};
}
module.exports = {createFakeMotion, PROPS, MUSE_PROPS};
