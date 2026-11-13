// node tests/behavior_test.cjs [--verbose] — program layer (app/behavior.js + motion.js programPass).
// Sandbox escapes refused · time-budget guard · hot swap keeps state · priorities · claims · events out · every character ·
// composition with commanded actions / lip sync / the learned map · sense geometry · request/onCode · per-frame latency.
// Exit code = number of failures.
'use strict';
const fs = require('fs'), path = require('path');
const load = require('./load.cjs');
const B = load('app/behavior.js');
const SIM = require('../app/behavior_sim.cjs');
const Moves = load('app/moves.js'); globalThis.AvatarMovesLib = Moves;
const M = load('app/motion.js');
const ANIM = path.join(__dirname, '..', 'renderer', 'animals');
const VERBOSE = process.argv.includes('--verbose');
let fails = 0, passes = 0;
const check = (n, ok, d) => { if (!ok || VERBOSE) console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (ok) passes++; else fails++; };
const DT = 1 / 30;
const CHARS = ['fox', 'bear', 'rabbit', 'hippo', 'monkey', 'orb', 'cube', 'drop', 'star', 'pyramid', 'blob', 'muse'];

// a runtime on the light host with a scripted user (no motion.js)
function simRun(code, o = {}) { return SIM.dryRun(code, Object.assign({character: 'fox', fps: 30}, o)); }
function rt(character = 'fox') {
  const ev = [], acts = [], styles = [];
  const R = B.createBehaviors({character});
  R.attach({character: () => character, doAction: (d, meta) => { acts.push({d, meta}); return {ok: true, status: 'preparing'}; }, setStyle: s => styles.push(s)});
  R.onEvent(e => ev.push(e));
  const tick = (n = 1, ctx = {}) => { let out = null; for (let i = 0; i < n; i++) out = R.tick(Object.assign({dt: DT, features: null, voice: {user: {}, avatar: {}}, out: {}}, ctx)); return out; };
  return {R, ev, acts, styles, tick};
}
function makeMotion(name, seed = 4330) {
  const m = M.createMotion({seed, preset: 'default'});
  if (name === 'muse') m.setProps(JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.props.json'))), JSON.parse(fs.readFileSync(path.join(ANIM, 'muse.body.json'))), 'muse');
  else m.setProps(null, null, name);
  return m;
}
const silence = {rms: 0.001, voiced: false};

// ================================================================== 1. sandbox
{
  const refusedStatic = {
    'window': `({ per_frame(api) { window.x = 1; } })`,
    'document': `({ per_frame(api) { document.title = 'x'; } })`,
    'fetch': `({ per_frame(api) { fetch('http://x'); } })`,
    'globalThis': `({ per_frame(api) { globalThis.y = 2; } })`,
    'this': `({ per_frame(api) { this.q = 1; } })`,
    'Function': `({ per_frame(api) { Function('return 1')(); } })`,
    'eval': `({ per_frame(api) { eval('1'); } })`,
    'constructor (property)': `({ per_frame(api) { (() => 1).constructor('return globalThis')(); } })`,
    'constructor (destructuring)': `({ per_frame(api) { const {constructor: C} = api.set; } })`,
    '__proto__': `({ per_frame(api) { ({}).__proto__.polluted = 1; } })`,
    'prototype': `({ per_frame(api) { Object.prototype.polluted = 1; } })`,
    'getPrototypeOf': `({ per_frame(api) { Object.getPrototypeOf({}).polluted = 1; } })`,
    'dynamic import': `({ per_frame(api) { import('fs'); } })`,
    'setTimeout': `({ setup(api) { setTimeout(() => 0, 1); }, per_frame(api) {} })`,
    'process': `({ per_frame(api) { process.exit(1); } })`,
    'require': `({ per_frame(api) { require('fs'); } })`,
    'Date': `({ per_frame(api) { Date.now(); } })`,
    'async': `({ async per_frame(api) {} })`,
    'generator': `({ *gen() { yield 1; }, per_frame(api) {} })`,
    'class': `({ per_frame(api) { class A {} } })`,
    'with': `({ per_frame(api) { with (api) {} } })`,
    'arguments': `({ per_frame(api) { arguments[0]; } })`,
    'escaped name': `({ per_frame(api) { const a = {}; a.\\u0063onstructor; } })`,
    'string constructor': `({ per_frame(api) { const k = 'constructor'; } })`,
    '__ names': `({ per_frame(api) { const __g = () => true; } })`,
    'Reflect': `({ per_frame(api) { Reflect.ownKeys({}); } })`,
    'Proxy': `({ per_frame(api) { new Proxy({}, {}); } })`,
    'Symbol': `({ per_frame(api) { Symbol.iterator; } })`,
    'new.target': `({ per_frame(api) { function F() { return new.target; } } })`,
  };
  for (const [n, code] of Object.entries(refusedStatic)) { const c = B.compile(code); check(`sandbox: '${n}' refused at compile`, !c.ok, c.ok ? 'compiled!' : c.error.slice(0, 90)); }

  // runtime escapes: compile, but must not reach anything
  const before = {proto: Object.getOwnPropertyNames(Object.prototype).length, arrProto: Object.getOwnPropertyNames(Array.prototype).length, rnd: Math.random, keys: Object.keys};
  const runtimeCases = {
    'computed constructor': `({ per_frame(api) { const k = 'constr' + 'uctor'; const C = (() => 1)[k]; api.state.got = typeof C; } })`,
    'computed key literal': `({ per_frame(api) { const k = ['__pro', 'to__'].join(''); ({})[k].x = 1; } })`,
    'computed key in pattern': `({ per_frame(api) { const k = 'construc' + 'tor'; const {[k]: C} = {}; api.state.got = typeof C; } })`,
    'template key': `({ per_frame(api) { const k = \`con\${'struc'}tor\`; const C = api.set[k]; api.state.got = typeof C; } })`,
    'shadowed globals': `({ per_frame(api) { api.state.got = [typeof self, typeof top, typeof navigator, typeof location, typeof global, typeof Buffer, typeof history].join(','); } })`,
    'Math frozen': `({ per_frame(api) { Math.random = () => 4; } })`,
    'Object safe': `({ per_frame(api) { api.state.got = typeof Object.defineProperty + typeof Object.create; } })`,
    'Error.prepareStackTrace': `({ per_frame(api) { Error.prepareStackTrace = (e, s) => s; } })`,
    'api frozen': `({ per_frame(api) { api.set = null; } })`,
    'user snapshot frozen': `({ per_frame(api) { api.user.head.yaw = 5; } })`,
    'array cap': `({ per_frame(api) { const a = new Array(1e9); } })`,
    'Array.from cap': `({ per_frame(api) { Array.from({length: 1e9}); } })`,
  };
  for (const [n, code] of Object.entries(runtimeCases)) {
    const c = B.compile(code);
    if (!c.ok) { check(`sandbox runtime: '${n}' refused at compile`, true, c.error.slice(0, 80)); continue; }
    const {R, ev, tick} = rt();
    const id = R.start({name: 't', code});
    tick(3);
    const info = R.info(id) || {};
    const st = info.state || {};
    const err = ev.find(e => e.phase === 'error');
    let ok;
    if (n === 'computed constructor' || n === 'computed key in pattern' || n === 'template key') ok = !!err && /not allowed/.test(err.error) && st.got === undefined;
    else if (n === 'shadowed globals') ok = /^(undefined,){6}undefined$/.test(String(st.got));
    else if (n === 'Object safe') ok = st.got === 'undefinedundefined';
    else ok = !!err;   // the write / allocation throws → the program stops with an error event
    check(`sandbox runtime: '${n}' contained`, ok, err ? err.error.slice(0, 80) : JSON.stringify(st).slice(0, 80));
  }
  check('sandbox: no global side effects', Object.getOwnPropertyNames(Object.prototype).length === before.proto && Object.getOwnPropertyNames(Array.prototype).length === before.arrProto &&
    Math.random === before.rnd && Object.keys === before.keys && globalThis.polluted === undefined && ({}).polluted === undefined && ({}).x === undefined);
  // allowed plain JS
  const okc = B.compile(`({ setup(api) { api.state.m = new Map(); api.state.s = new Set([1, 2]); }, per_frame(api) {
    const a = [3, 1, 2].sort((x, y) => x - y).map(x => x * 2).filter(Boolean); const o = {a, b: {c: 1}}; const {b: {c}} = o;
    const js = JSON.parse(JSON.stringify(o)); let n = 0; for (const k in js) n++; for (const v of a) n += v; for (let i = 0; i < 3; i++) n++;
    const lbl = \`n=\${n}\`; api.state.m.set('n', n); api.state.r = [Math.max(...a), Number('2'), String(5), Object.keys(o).length, lbl, c, Array.isArray(a), [..."ab"].length]; } })`);
  const {R: R1, tick: t1} = rt();
  const id1 = R1.start({name: 'plain', code: `({ setup(api) { api.state.m = new Map(); api.state.s = new Set([1, 2]); }, per_frame(api) {
    const a = [3, 1, 2].sort((x, y) => x - y).map(x => x * 2).filter(Boolean); const o = {a, b: {c: 1}}; const {b: {c}} = o;
    const js = JSON.parse(JSON.stringify(o)); let n = 0; for (const k in js) n++; for (const v of a) n += v; for (let i = 0; i < 3; i++) n++;
    const lbl = \`n=\${n}\`; api.state.m.set('n', n); api.state.r = [Math.max(...a), Number('2'), String(5), Object.keys(o).length, lbl, c, Array.isArray(a), [..."ab"].length]; } })`});
  t1(2);
  const r1 = (R1.info(id1) || {}).state || {};
  check('sandbox: plain JS works (Map/Set/JSON/destructuring/for-in/for-of/templates/spread)', okc.ok && JSON.stringify(r1.r) === JSON.stringify([6, 2, '5', 2, 'n=17', 1, true, 2]), JSON.stringify(r1.r));
  // a program with helper statements before the object literal
  const {R: R2, tick: t2} = rt();
  const id2 = R2.start({name: 'helpers', code: `const k = 2;\nfunction dbl(x) { return x * k; }\n({ per_frame(api) { api.state.v = dbl(api.frame); } });`});
  t2(3);
  check('sandbox: helper statements before the object literal', !!id2 && (R2.info(id2).state.v === 4), R2.lastError || '');
}

// ================================================================== 2. budget guard
{
  const cases = {
    'while(true)': `({ per_frame(api) { while (true) {} } })`,
    'for(;;)': `({ per_frame(api) { for (;;) {} } })`,
    'do-while': `({ per_frame(api) { do { api.state.x = 1; } while (true); } })`,
    'catch swallow': `({ per_frame(api) { while (true) { try { while (true) {} } catch (e) {} } } })`,
    'growing for-of': `({ per_frame(api) { const a = [1]; for (const x of a) a.push(x); } })`,
    'for-in growth': `({ per_frame(api) { const o = {a: 1}; let i = 0; while (true) { for (const k in o) { i++; } } } })`,
    'exponential arrow recursion': `({ per_frame(api) { const fib = n => n < 2 ? n : fib(n - 1) + fib(n - 2); api.state.v = fib(60); } })`,
    'exponential function recursion': `({ per_frame(api) { function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); } api.state.v = fib(60); } })`,
    'method recursion': `({ f(n) { return n < 2 ? n : this_is_not_here(n); }, per_frame(api) { const o = { g(n) { return n < 2 ? n : o.g(n - 1) + o.g(n - 2); } }; o.g(60); } })`,
    'array callbacks': `({ per_frame(api) { const a = Array.from({length: 99999}, (_, i) => i); let s = 0; for (let k = 0; k < 1000; k++) a.forEach(x => { s += x; }); } })`,
  };
  for (const [n, code] of Object.entries(cases)) {
    const {R, ev, tick} = rt();
    const t0 = Date.now();
    const id = R.start({name: 'slow', code});
    if (!id) { check(`budget: '${n}'`, false, 'did not start: ' + R.lastError); continue; }
    tick(6);
    const ms = Date.now() - t0;
    const st = R.list()[0] || {};
    const paused = ev.find(e => e.phase === 'paused'), err = ev.find(e => e.phase === 'error');
    check(`budget: '${n}' is cut off and paused / stopped`, (paused || err) && st.status !== 'running' && ms < 600, `${ms} ms, ${paused ? 'paused: ' + paused.reason : err ? 'error: ' + err.error : 'still ' + st.status}`);
  }
  // steady overrun (~2 ms of work per frame, no single cut-off) → paused within the window
  {
    const {R, ev, tick} = rt();
    R.start({name: 'heavy', code: `({ per_frame(api) { let s = 0; for (let i = 0; i < 4e6; i++) { s += Math.sqrt(i); if (s < 0) break; } api.state.s = s; } })`});
    tick(40);
    const paused = ev.find(e => e.phase === 'paused');
    check('budget: a program steadily over 1 ms/frame is paused and reported', !!paused && /budget/.test(paused.reason), paused ? `${paused.reason} (p50 ${paused.ms_p50} ms)` : JSON.stringify(R.list()[0]));
    check('budget: resume() runs a paused program again', R.resume('heavy') && R.list()[0].status === 'running');
  }
  // infinite setup → start refused
  { const {R} = rt(); const id = R.start({name: 's', code: `({ setup(api) { while (true) {} }, per_frame(api) {} })`}); check('budget: an endless setup is refused', !id && /budget/.test(R.lastError || ''), R.lastError); }
  // a cheap program keeps running for a long time
  { const {R, tick} = rt(); R.start({name: 'cheap', code: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { let s = 0; for (let i = 0; i < 200; i++) s += i; api.set('right_ear', Math.sin(api.t)); } })`}); tick(600); check('budget: a cheap program keeps running (600 frames)', R.list()[0].status === 'running' && R.list()[0].frames === 600, JSON.stringify(R.list()[0]).slice(0, 160)); }
}

// ================================================================== 3. exceptions + lifecycle
{
  const {R, ev, tick} = rt();
  const id = R.start({name: 'boom', instruction: 'x', code: `({ per_frame(api) { if (api.frame === 4) api.user.nothing.here = 1; } })`});
  tick(10);
  const err = ev.find(e => e.phase === 'error');
  check('exception: stops the program with an error event (frame, message, code for a fix)', !!err && err.frame === 4 && /undefined|nothing/.test(err.error) && !!err.code && R.list()[0].status === 'error', err && err.error);
  const {R: R2, ev: ev2, tick: tk2} = rt();
  R2.start({name: 'fin', code: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.set('left_ear', 1); if (api.t > 0.5) api.done('time up'); }, end(api) { api.state.ended = true; } })`});
  tk2(30);
  const end = ev2.find(e => e.phase === 'end');
  check('done(): ends with its reason, end() ran, removed from list', !!end && end.reason === 'done: time up' && R2.list().length === 0, end && end.reason);
  const {R: R3, ev: ev3} = rt();
  check('start: a syntax error is refused with the reason', R3.start({name: 'bad', code: '({ per_frame(api) { if ( } })'}) === null && ev3.some(e => e.phase === 'error' && /syntax|unexpected|unbalanced/i.test(e.error)), R3.lastError);
  check('start: a non-object program is refused', R3.start({name: 'bad2', code: '42'}) === null && /object literal/.test(R3.lastError), R3.lastError);
  check('start: a program without per_frame is refused', R3.start({name: 'bad3', code: '({ setup(api) {} })'}) === null && /per_frame/.test(R3.lastError), R3.lastError);
  const {R: R4, tick: tk4} = rt();
  R4.start({name: 'a', code: `({ per_frame(api) {} })`}); R4.start({name: 'b', code: `({ per_frame(api) {} })`});
  const n1 = R4.list().length; R4.start({name: 'a', code: `({ per_frame(api) { api.state.v = 2; } })`}); tk4(1);
  check('start: same name replaces the old program; stop(all) ends everything', n1 === 2 && R4.list().length === 2 && R4.stop('all') === 2 && R4.list().length === 0);
}

// ================================================================== 4. hot swap
{
  const {R, ev, tick} = rt();
  const id = R.start({name: 'counter', instruction: 'count frames', code: `({ setup(api) { api.state.n = api.state.n || 0; api.state.setups = (api.state.setups || 0) + 1; api.claim(['ears']); },
    per_frame(api) { api.state.n++; api.set('right_ear', 0.2); } })`});
  tick(20);
  const ok = R.change('counter', {instruction: 'count by two', code: `({ setup(api) { api.state.seen = api.state.n; api.state.setups++; api.claim(['ears']); },
    per_frame(api) { api.state.n += 2; api.set('right_ear', 0.8); } })`});
  tick(10);
  const info = R.info('counter');
  check('hot swap: same id, setup gets the old state, counting continues', ok && info.id === id && info.state.seen === 20 && info.state.n === 40 && info.state.setups === 2 && info.version === 2, JSON.stringify(info.state));
  check('hot swap: "changed" event', ev.some(e => e.phase === 'changed' && e.name === 'counter'));
  const bad = R.change('counter', {code: `({ per_frame(api) { if ( } })`});
  tick(5);
  check('hot swap: a broken new version is refused and the old one keeps running', !bad && R.info('counter').state.n === 50 && R.list()[0].status === 'running', R.lastError);
  const bad2 = R.change('counter', {code: `({ setup(api) { throw new Error('nope'); }, per_frame(api) { api.state.n = -1; } })`});
  tick(2);
  check('hot swap: a new version whose setup throws is rolled back', !bad2 && R.info('counter').state.n === 54, R.lastError);
  // an errored program revived by an edit
  R.start({name: 'err', code: `({ per_frame(api) { api.state.k = (api.state.k || 0) + 1; if (api.state.k === 3) throw new Error('x'); } })`});
  tick(5);
  const s1 = R.list().find(p => p.name === 'err').status;
  R.change('err', {code: `({ per_frame(api) { api.state.k++; } })`}); tick(4);
  check('hot swap: an errored program is fixed by change() and keeps its state', s1 === 'error' && R.list().find(p => p.name === 'err').status === 'running' && R.info('err').state.k === 7, s1 + ' ' + JSON.stringify(R.info('err').state));
  // smooth: the channel value moves without a weight drop across the swap
  const {R: Rs, tick: ts} = rt();
  Rs.start({name: 's', code: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.set('right_ear', 0.3); } })`});
  ts(30);
  const w0 = ts(1).ear_l[1];
  Rs.change('s', {code: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.set('right_ear', 0.5); } })`});
  const o1 = ts(1);
  check('hot swap: no release/re-claim pop (weight stays 1)', w0 > 0.99 && o1.ear_l[1] > 0.99 && o1.ear_l[0] === 0.5, JSON.stringify(o1.ear_l));
}

// ================================================================== 5. priorities + claims
{
  const {R, ev, tick} = rt('fox');
  const lo = `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.state.owns = api.owns('ears'); api.set('right_ear', -0.5); } })`;
  const hi = `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.state.owns = api.owns('ears'); api.set('right_ear', 0.9); } })`;
  R.start({name: 'low', code: lo, priority: 2});
  R.start({name: 'high', code: hi, priority: 8});
  let o = tick(20);
  check('priority: the higher priority wins a contested group', Math.abs(o.ear_l[0] - 0.9) < 1e-6 && R.info('low').state.owns === false && R.info('high').state.owns === true, JSON.stringify(o.ear_l));
  check('priority: list() shows the groups each program actually holds', JSON.stringify(R.list().find(p => p.name === 'low').groups) === '[]' && R.list().find(p => p.name === 'high').groups.includes('ears'));
  R.stop('high');
  const vals = []; for (let i = 0; i < 8; i++) vals.push(tick(1).ear_l[0]);
  check('priority: when it stops the lower one takes over with a crossfade (no jump)', Math.abs(vals[7] + 0.5) < 1e-6 && vals[0] > -0.5 && vals[0] < 0.9, vals.map(v => v.toFixed(2)).join(' '));
  // ties → newest
  const {R: R2, tick: t2} = rt();
  R2.start({name: 'first', code: lo, priority: 5}); R2.start({name: 'second', code: hi, priority: 5});
  check('priority: ties go to the newest', Math.abs(t2(20).ear_l[0] - 0.9) < 1e-6);
  // claims
  const {R: R3, ev: e3, tick: t3} = rt('bear');
  R3.start({name: 'c', code: `({ setup(api) { api.state.r = api.claim(['tail', 'ears']); }, per_frame(api) {
    api.state.a = api.set('tail_wag', 1); api.state.b = api.set('right_ear', 0.5); api.state.c = api.set('smile', 1); api.state.d = api.set('wiggle', 1); } })`});
  o = t3(5);
  const s3 = R3.info('c').state;
  check('claims: a part the character lacks is refused honestly (unsupported event, missing in claim result)',
    s3.r.ok === false && s3.r.missing.includes('tail') && s3.r.groups.includes('ears') && e3.some(e => e.phase === 'unsupported' && /tail/.test(e.what)), JSON.stringify(s3.r));
  check('claims: only claimed groups are written (unclaimed + unknown refused with warnings)', s3.a === false && s3.b === true && s3.c === false && s3.d === false &&
    e3.some(e => e.phase === 'warning' && /claim/.test(e.msg)) && e3.some(e => e.phase === 'warning' && /unknown channel/.test(e.msg)) && !('smile' in o) && 'ear_l' in o, JSON.stringify(s3));
  // a channel the program stops writing fades back (weight → 0 within ~0.5 s)
  const {R: R4, tick: t4} = rt();
  R4.start({name: 'f', code: `({ setup(api) { api.claim(['ears']); }, per_frame(api) { if (api.t < 0.5) api.set('right_ear', 1); } })`});
  const ws = []; for (let i = 0; i < 40; i++) { const x = t4(1); ws.push(x && x.ear_l ? x.ear_l[1] : 0); }
  check('claims: a control it stops writing eases back (weight 1 → < 0.1 within 0.35 s, gone by 0.8 s)', ws[10] > 0.99 && ws[18] < 0.5 && ws[25] < 0.1 && ws[39] === 0, ws.filter((_, i) => i % 4 === 0).map(v => v.toFixed(2)).join(' '));
  check('claims: release() hands a group back', (() => { const {R: R5, tick: t5} = rt(); R5.start({name: 'r', code: `({ setup(api) { api.claim(['ears', 'tail']); api.release(['tail']); }, per_frame(api) {} })`}); t5(1); return JSON.stringify(R5.list()[0].claimed) === '["ears"]'; })());
}

// ================================================================== 6. events out
{
  const {R, ev, acts, styles, tick} = rt('muse');
  R.start({name: 'ev', instruction: 'test events', code: `({ per_frame(api) {
    if (api.frame === 2) { api.state.e1 = api.emit('count_reached', {count: 10}); api.state.e2 = api.emit('again'); }
    if (api.frame === 3) api.state.a1 = api.do_action('do a small jump');
    if (api.frame === 4) api.state.a2 = api.do_action('wave');
    if (api.frame === 40) api.state.a3 = api.do_action('wave again');
    if (api.frame === 5) api.set_style({energy: 0.9, mirroring: 0.2});
    if (api.frame === 60) api.done('ok');
  } })`});
  tick(70);
  const rm = ev.filter(e => e.type === 'rule_moment');
  check('emit → onEvent rule_moment with name / data / instruction (rate-limited)', rm.length === 1 && rm[0].event === 'count_reached' && rm[0].data.count === 10 && rm[0].reply === true && rm[0].instruction === 'test events', JSON.stringify(rm));
  check('do_action → deps.doAction (meta = program), at most one per second', acts.length === 2 && acts[0].d === 'do a small jump' && acts[0].meta.name === 'ev' && acts[1].d === 'wave again' && ev.filter(e => e.type === 'behavior_action').length === 2, JSON.stringify(acts));
  check('set_style → deps.setStyle; reset to automatic when the program ends', styles.length === 2 && styles[0].energy === 0.9 && styles[1].energy === null && styles[1].mirroring === null, JSON.stringify(styles));
  check('lifecycle events: start … end', ev[0].phase === 'start' && ev.some(e => e.phase === 'end' && e.reason === 'done: ok'));
}

// ================================================================== 7. sense (synthetic user): touches, blinks, sides, gaze, events once
{
  const code = `({ setup(api) { api.state.log = []; api.state.blinks = 0; api.state.seen = {}; },
    per_frame(api) {
      const s = api.state, u = api.user;
      for (const e of api.events()) { s.seen[e.type] = (s.seen[e.type] || 0) + 1; if (e.type === 'blink') s.blinks++; if (e.type === 'touch') s.log.push(e.part + '/' + e.hand); }
      if (u.touching('nose')) s.nose = (s.nose || 0) + 1;
      if (u.head.roll > 0.2) s.rollR = true; if (u.head.roll < -0.2) s.rollL = true;
      if (u.hands.right.present && u.hands.right.raised) s.rRaised = (s.rRaised || 0) + 1;
      if (u.hands.left.present && u.hands.right.present && u.hands.right.x > u.hands.left.x) s.sidesOk = true;
      if (!u.gaze.atCamera && u.present) s.away = (s.away || 0) + 1;
      if (u.speaking) s.talk = (s.talk || 0) + 1;
      if (api.gesture('my_salute').on) s.taught = (s.taught || 0) + 1;
      s.heard = (s.heard || 0) + api.heard(['banana']);
      if (api.gesture('thumbs_up').on) s.thumb = true;
    } })`;
  const r = SIM.dryRun(code, {character: 'fox', fps: 30, trace: false});
  const s = r.runtime.info(r.id).state;
  check('sense: 8 blinks in the tour counted once each', s.blinks === 8, s.blinks);
  check('sense: touch events with the part and hand (nose/right, left_ear/left, mouth/right)', s.log.slice(0, 3).join(',') === 'nose/right,left_ear/left,mouth/right', s.log.join(','));
  check('sense: touching(nose) holds while touching (~1.8 s + release)', s.nose > 45 && s.nose < 75, s.nose);
  check('sense: head roll both ways; raised right hand; his right hand on screen-right; look away; speaking', s.rollR && s.rollL && s.rRaised > 30 && s.sidesOk && s.away > 30 && s.talk > 60, JSON.stringify({rR: s.rRaised, away: s.away, talk: s.talk}));
  check('sense: taught gesture (code refs injected) and heard words in the tour', s.taught > 0 && s.heard === 2, `${s.taught} ${s.heard}`);
  check('sense: built-in gesture thumbs_up', !!s.thumb);
  const want = ['face_appear', 'smile_start', 'head_tilt', 'hand_up', 'wave', 'touch', 'look_away', 'look_back', 'speech_start', 'nod', 'head_shake', 'hands_together',
    'hand_gesture', 'mouth_open', 'brows_up', 'frown', 'face_lost', 'lean_in', 'shrug', 'self_speech_start', 'wink'];
  const missing = want.filter(w => !s.seen[w]);
  check('sense: every tour cue produced its event', !missing.length, 'missing ' + missing.join(','));
  // recorded frames (no mesh / lm): no exceptions, plausible events
  const r2 = SIM.dryRun(code, {character: 'fox', fps: 30, frames: SIM.loadRecorded()[1].frames});
  check('sense: recorded clip (face-box fallback, palm-only hands) runs clean', r2.ok, r2.error);
  // look target geometry: his right hand on screen-right → eyes / head turn to the screen's right (+)
  const look = SIM.dryRun(`({ per_frame(api) { api.look('right_hand'); } })`, {character: 'fox', script: {dur: 3, at: [[0, 3, {hand_right: {x: 0.6, y: 0.1, open: 1}}]]}, trace: true});
  const at = (tr, t) => tr.find(x => x.t >= t).sets;
  const last = at(look.trace, 2.5);
  check('look: at his right hand (screen-right) → gaze_x > 0, head_yaw > 0', last.gaze_x[0] > 0.3 && last.head_yaw[0] > 0.2, JSON.stringify(last));
  const lk2 = SIM.dryRun(`({ per_frame(api) { api.look('gaze', {head: false}); } })`, {character: 'fox', script: {dur: 2, at: [[0, 2, {attending: false, lookX: -0.4, lookY: 0.2}]]}, trace: true});
  const l2 = at(lk2.trace, 1.5);
  check('look: where he looks (left/up) → eyes left/up, head untouched', l2.gaze_x[0] < -0.5 && l2.gaze_y[0] > 0.3 && !l2.head_yaw, JSON.stringify(l2));
  // paw_to mirror mapping
  const pw = SIM.dryRun(`({ setup(api) { api.claim(['arms']); }, per_frame(api) { const h = api.user.hands.right; if (h.present) api.paw_to('left', h.x, h.y); } })`,
    {character: 'fox', script: {dur: 2, at: [[0, 2, {hand_right: {x: 0.6, y: 0.5, open: 1}}]]}, trace: true});
  const p2 = at(pw.trace, 1.5);
  check('paw_to: his raised right hand (screen-right) → its left paw out + up', p2.left_paw_x[0] > 0.3 && p2.left_paw_y[0] > 0.6 && !p2.right_paw_x, JSON.stringify(p2));
}

// ================================================================== 8. motion composition (every character)
function runMotion(name, code, steps, o = {}) {
  const m = makeMotion(name, o.seed || 4330);
  const R = B.createBehaviors({character: name});
  const ev = [];
  R.attach({motion: () => m, character: () => name, doAction: () => ({ok: true})}); R.onEvent(e => ev.push(e));
  if (o.learned) m.setLearned(true);
  const id = code ? R.start({name: 'p', code}) : null;
  const rigs = [];
  for (let k = 0; k < steps; k++) {
    if (o.at && o.at[k]) o.at[k](m, R);
    const sp = o.speaking && o.speaking(k);
    rigs.push(Object.assign({}, m.step(DT, {features: null, userVoice: silence, avatarVoice: sp ? {rms: 0.06, voiced: true} : {rms: 0, voiced: false}})));
    if (o.learned) rigs[rigs.length - 1].__lp = m.learnedParts(['ear_l', 'ear_r', 'smile', 'head_roll']);
  }
  return {m, R, rigs, id, ev};
}
{
  const all = `({ setup(api) { api.state.r = api.claim(['all']); },
    per_frame(api) { const v = 0.5 + 0.3 * Math.sin(api.t * 3); for (const c of ['right_ear', 'left_ear', 'tail_wag', 'nose_wiggle', 'right_paw_y', 'left_paw_y', 'smile', 'head_roll', 'body_lean', 'gaze_x', 'eyes_closed', 'right_brow', 'body_bounce', 'right_fingers'])
      if (api.has(c)) api.set(c, c === 'eyes_closed' ? 0.6 : c === 'right_fingers' ? 3 : v); } })`;
  for (const ch of CHARS) {
    const {rigs, R, ev} = runMotion(ch, all, 60);
    const last = rigs[rigs.length - 1], tr = B.TRAITS[ch];
    const target = 0.5 + 0.3 * Math.sin(59 * DT * 3 + DT * 3);
    const near = (c, v) => Math.abs(last[c] - v) < 0.06;
    const okE = tr.ears ? near('ear_l', target) && near('ear_r', target) : true;
    const okT = tr.tail ? near('tail_wag', target) : true;
    const okN = tr.nose ? near('nose_wiggle', target) : true;
    const okF = tr.fingers ? near('fingers_l', 3) : true;
    const st = R.info('p').state.r;
    const lacks = ['ears', 'tail', 'nose'].filter(k => !tr[k]);
    const honest = lacks.every(k => st.missing.includes(k));
    check(`every character: ${ch} — program drives what it has, refuses what it lacks`, R.list()[0].status === 'running' && okE && okT && okN && okF && near('hand_l_y', target) && near('smile', target) &&
      near('head_roll', target > 0.4 ? 0.4 : target) && near('blink_l', 0.6) && honest && !ev.some(e => e.phase === 'error'),
      `ear ${last.ear_l.toFixed(2)} tail ${last.tail_wag.toFixed(2)} nose ${last.nose_wiggle.toFixed(2)} paw ${last.hand_l_y.toFixed(2)} target ${target.toFixed(2)} missing ${st.missing}`);
  }
  // unclaimed channels are untouched: rig with a program on ears only == rig without programs, everywhere but the ears
  for (const ch of ['fox', 'muse']) {
    const a = runMotion(ch, `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.set('right_ear', 1); } })`, 90).rigs;
    const b = runMotion(ch, null, 90).rigs;
    let diff = [];
    for (let k = 0; k < a.length; k++) for (const c of Object.keys(b[k])) if (!/^ear_[lr]$/.test(c) && Math.abs(a[k][c] - b[k][c]) > 1e-9) diff.push(c);
    diff = [...new Set(diff)];
    check(`isolation (${ch}): a program on the ears changes nothing else`, !diff.length && Math.abs(a[89].ear_l - 1) < 1e-6, diff.join(','));
  }
  // commanded keyframe job over the program; it comes back after the job
  {
    const code = `({ setup(api) { api.claim(['ears', 'mouth']); }, per_frame(api) { api.set('right_ear', -0.8); api.set('mouth_open', 0.6); } })`;
    const at = {30: (m) => m.command({action: 'keyframes', id: 'k1', description: 'ears up', done: true, hold: false, poses: [{t: 0.3, right_ear: 1, left_ear: 1}, {t: 1.2, right_ear: 1, left_ear: 1}]})};
    const {rigs} = runMotion('fox', code, 180, {at, speaking: k => k >= 150 && k < 180});
    const during = rigs[60].ear_l, after = rigs[140].ear_l, mouthSilent = rigs[140].mouth_open, mouthVoiced = rigs[170].mouth_open;
    check('composition: a commanded keyframe job wins over the program while it plays; the program returns after', during > 0.5 && Math.abs(after + 0.8) < 0.05, `during ${during.toFixed(2)} after ${after.toFixed(2)}`);
    check('composition: mouth_open yields to speech while voiced (lip sync owns the mouth)', Math.abs(mouthSilent - 0.6) < 0.02 && mouthVoiced < 0.3, `silent ${mouthSilent.toFixed(2)} voiced ${mouthVoiced.toFixed(2)}`);
  }
  // learned / unified map: the program's channels show as owned (alpha 0, beta = value)
  {
    const {rigs} = runMotion('fox', `({ setup(api) { api.claim(['ears']); }, per_frame(api) { api.set('right_ear', 0.7); } })`, 40, {learned: true});
    const lp = rigs[39].__lp;
    const e = lp.ab.find(x => x[0] === 0);
    check('learned/unified: the program channel goes to the hub as owned (alpha≈0, beta=value) → set_override', !!e && e[1] < 0.01 && Math.abs(e[2] - 0.7) < 0.02 && !lp.ab.some(x => x[0] === 2 && x[1] < 0.5), JSON.stringify(lp.ab));
  }
  // the layer installs / uninstalls itself
  {
    const m = makeMotion('fox'), R = B.createBehaviors({character: 'fox'});
    R.attach({motion: () => m, character: () => 'fox'});
    const a0 = m.programLayer; R.start({name: 'x', code: `({ per_frame(api) {} })`}); const a1 = m.programLayer; R.stop('x');
    for (let k = 0; k < 20; k++) m.step(DT, {});
    check('install: the motion layer is set while programs run and removed after', !a0 && a1 && !m.programLayer);
  }
  // head_pitch sign (toolkit + = up = rig −)
  {
    const {rigs} = runMotion('fox', `({ setup(api) { api.claim(['head']); }, per_frame(api) { api.set('head_pitch', 0.3); } })`, 30);
    check('toolkit frame: head_pitch + = looks up (rig head_pitch −)', rigs[29].head_pitch < -0.25, rigs[29].head_pitch.toFixed(3));
  }
}

// ================================================================== 9. request / onCode (the hub round trip)
{
  const sent = [];
  const R = B.createBehaviors({character: 'fox'});
  R.attach({character: () => 'fox', requestCode: msg => sent.push(msg)});
  (async () => {
    const p = R.request('wiggle your ears when I smile', {priority: 6});
    const msg = sent[0];
    R.onCode({type: 'behavior_code', id: msg.id, ok: true, name: msg.name, code: `({ setup(api) { api.claim(['ears']); api.state.v = 1; }, per_frame(api) {} })`, ms: 900});
    const r = await p;
    const p2 = R.request('make it faster', {name: r.name, edit: true});
    const m2 = sent[1];
    R.onCode({type: 'behavior_code', id: m2.id, ok: true, name: m2.name, code: `({ setup(api) { api.state.v2 = api.state.v + 1; }, per_frame(api) {} })`});
    const r2 = await p2;
    const p3 = R.request('broken', {}); R.onCode({type: 'behavior_code', id: sent[2].id, ok: false, error: 'model failed'}); const r3 = await p3;
    check('request/onCode: start round trip (name from the instruction, priority kept)', r.ok && msg.instruction === 'wiggle your ears when I smile' && msg.character === 'fox' && R.list()[0].priority === 6 && r.name === msg.name, JSON.stringify({r, msg}));
    check('request/onCode: an edit sends prev_code and hot-swaps, keeping state', r2.ok && m2.prev_code && /api.state.v = 1/.test(m2.prev_code) && R.info(r.name).state.v2 === 2 && /then: make it faster/.test(R.info(r.name).instruction), JSON.stringify(r2));
    check('request/onCode: a hub failure resolves ok:false with the reason', !r3.ok && r3.error === 'model failed');
    finish();
  })().catch(e => { check('request/onCode', false, String(e)); finish(); });
}

// ================================================================== 10. latency: 5 concurrent programs, per frame
function latency() {
  const progs = [
    `({ setup(api) { api.claim(['ears']); api.state.until = -1; }, per_frame(api) { for (const e of api.events('touch')) if (e.part === 'nose') api.state.until = api.t + 2;
       if (api.t < api.state.until) { const v = 0.7 * Math.sin(2 * Math.PI * 4 * api.t); api.set('right_ear', v); api.set('left_ear', -v); } } })`,
    `({ setup(api) { api.claim(['head']); }, per_frame(api) { const u = api.user; if (!u.present) return; api.state.r = (api.state.r || 0) + (u.head.roll - (api.state.r || 0)) * Math.min(1, api.dt * 6); api.set('head_roll', api.state.r); if (api.t > 600) api.done('minute'); } })`,
    `({ setup(api) { api.state.n = 0; }, per_frame(api) { for (const e of api.events('blink')) { api.state.n++; if (api.state.n % 10 === 0) api.emit('blinks', {count: api.state.n}); } } })`,
    `({ per_frame(api) { const H = [api.user.hands.left, api.user.hands.right].filter(h => h.present); if (H.length) api.look({x: H[0].x, y: H[0].y}); } })`,
    `({ setup(api) { api.claim(['tail', 'arms']); }, per_frame(api) { const lv = api.user.level; api.set('tail_wag', lv * Math.sin(2 * Math.PI * (1 + 3 * lv) * api.t));
       const h = api.user.hands.right; if (h.present && h.raised) api.paw_to('left', h.x, h.y); } })`,
  ];
  const fr = SIM.synth(SIM.TOUR, {fps: 18});
  const runs = {};
  for (const n of [0, 5]) {
    const m = makeMotion('fox'), R = B.createBehaviors({character: 'fox'});
    R.attach({motion: () => m, character: () => 'fox'});
    for (let i = 0; i < n; i++) R.start({name: 'p' + i, code: progs[i]});
    const ms = [], lay = [];
    let fi = 0, f = null;
    for (let rep = 0; rep < 3; rep++) for (let k = 0; k < 32 * 30; k++) {
      const t = k * DT;
      while (fi < fr.frames.length && fr.frames[fi].t <= t) f = fr.frames[fi++];
      if (fi >= fr.frames.length) fi = 0;
      const v = fr.voice(t);
      const t0 = process.hrtime.bigint();
      m.step(DT, {features: f, userVoice: v.user.speaking ? {rms: 0.06, voiced: true} : silence, avatarVoice: {rms: 0, voiced: false}});
      ms.push(Number(process.hrtime.bigint() - t0) / 1e6);
    }
    const st = R.stats();
    const a = ms.slice(300).sort((x, y) => x - y);
    runs[n] = {p50: a[Math.floor(a.length / 2)], p95: a[Math.floor(a.length * 0.95)], layer_p50: st.ms_p50, layer_p95: st.ms_p95, running: R.list().filter(p => p.status === 'running').length};
  }
  const add50 = runs[5].p50 - runs[0].p50, add95 = runs[5].p95 - runs[0].p95;
  console.log(`latency: motion.step p50/p95 ${runs[0].p50.toFixed(3)}/${runs[0].p95.toFixed(3)} ms without programs, ${runs[5].p50.toFixed(3)}/${runs[5].p95.toFixed(3)} ms with 5 programs ` +
    `(program layer itself p50 ${runs[5].layer_p50.toFixed(3)} ms, p95 ${runs[5].layer_p95.toFixed(3)} ms; +${add50.toFixed(3)} ms p50)`);
  check('latency: 5 concurrent programs all running, layer p50 < 0.5 ms per frame', runs[5].running === 5 && runs[5].layer_p50 < 0.5, JSON.stringify(runs[5]));
  return runs;
}
let finished = false;
function finish() {
  if (finished) return; finished = true;
  latency();
  console.log(`behavior_test: ${passes} passed, ${fails} failed`);
  process.exit(fails);
}
