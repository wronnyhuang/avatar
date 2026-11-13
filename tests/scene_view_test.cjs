// scene: the station camera (moves.js setView / motion.setView): eases cam_* to the place's camera and back.
const assert = require('assert');
const {makeMuse} = require('./muse_lib.cjs');
const T = makeMuse(7);
const P = T.files.props, cam = id => P.places.find(p => p.id === id).camera;
T.step(30);
let o = T.rows[T.rows.length - 1];
const home = {z: o.cam_zoom, x: o.cam_x, y: o.cam_y};      // the standing muse's own framing (cam_zoom 0.3 today)
const bad = T.m.setView('nowhere'); assert(!bad.ok, 'unknown view refused');
let r = T.m.setView('board'); assert(r.ok && T.m.view === 'board', 'board view set');
T.step(5); o = T.rows[T.rows.length - 1];
assert(o.cam_zoom < home.z - 1e-3 && o.cam_zoom > home.z - 0.2, 'eases in, no cut ' + o.cam_zoom);
// per-frame change stays small (no pop)
let maxd = 0; const n0 = T.rows.length; T.step(90);
for (let i = n0; i < T.rows.length; i++) maxd = Math.max(maxd, Math.abs(T.rows[i].cam_zoom - T.rows[i - 1].cam_zoom), Math.abs(T.rows[i].cam_x - T.rows[i - 1].cam_x));
o = T.rows[T.rows.length - 1];
const c = cam('board');
assert(Math.abs(o.cam_zoom - c.cam_zoom) < 0.01 && Math.abs(o.cam_x - c.cam_x) < 0.01 && Math.abs(o.cam_y - c.cam_y) < 0.01, `settled on the board camera ${JSON.stringify([o.cam_zoom, o.cam_x, o.cam_y])}`);
assert(maxd < 0.05, 'smooth: max per-frame step ' + maxd);
r = T.m.setView('tv'); assert(r.ok); T.step(120); o = T.rows[T.rows.length - 1];
assert(Math.abs(o.cam_x - cam('tv').cam_x) < 0.01, 'tv camera');
T.m.setView(null); T.step(150); o = T.rows[T.rows.length - 1];
assert(Math.abs(o.cam_zoom - home.z) < 0.01 && Math.abs(o.cam_x - home.x) < 0.01 && T.m.view === null, 'back home');
// the TV is not a place to put things on
assert(!T.m.engine.E.places.some(p => p.id === 'tv'), 'tv not placeable');
assert(T.m.engine.E.places.some(p => p.id === 'board_tray'), 'tray is a place');
// a planned keyframe move while a station view is set keeps the station camera (no zoom-out pop at the board)
T.m.setView('board'); T.step(120);
const kr = T.m.command({action: 'keyframes', id: 'sv1', description: 'wave', done: true, hold: false,
  poses: [{t: 0, hand_r_y: 0.2}, {t: 0.6, hand_r_y: 0.9, hand_r_x: 0.4}, {t: 1.2, hand_r_y: 0.5}, {t: 1.8, hand_r_y: 0.1}]}, {src: 'tool'});
assert(kr.ok, 'keyframes accepted ' + JSON.stringify(kr));
let kz = 0; for (let i = 0; i < 75; i++) { T.step(1); const r2 = T.rows[T.rows.length - 1]; kz = Math.max(kz, Math.abs(r2.cam_zoom - cam('board').cam_zoom), Math.abs(r2.cam_x - cam('board').cam_x)); }
assert(kz < 0.02, 'station camera held during a planned move: max dev ' + kz);
T.m.setView(null); T.step(150);
console.log('scene_view_test OK', {maxd: +maxd.toFixed(4)});
