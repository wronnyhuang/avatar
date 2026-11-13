// teach: shared helpers for the teaching tests — a muse with the board scene, the teach director wired to it, a
// fake hub (streams stub items: hand-made strokes in a tiny single-stroke font) and a fake voice (prompt -> audio timing)
'use strict';
const path = require('path');
// the real scene (renderer/animals/muse.props.json, board + tray + tv + chalk + eraser); MUSE_PROPS overrides
const {makeMuse} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const Teach = load('app/teach.js');
// glyphs in a unit box (x 0..0.6, y 0..1 down): polylines
const G = {
  '0': [[[0.3, 0], [0.05, 0.2], [0.05, 0.8], [0.3, 1], [0.55, 0.8], [0.55, 0.2], [0.3, 0]]],
  '1': [[[0.15, 0.2], [0.35, 0], [0.35, 1]]],
  '2': [[[0.05, 0.2], [0.3, 0], [0.55, 0.2], [0.55, 0.4], [0.05, 1], [0.6, 1]]],
  '3': [[[0.05, 0.1], [0.3, 0], [0.55, 0.2], [0.3, 0.5], [0.55, 0.75], [0.3, 1], [0.05, 0.9]]],
  '4': [[[0.45, 1], [0.45, 0], [0.05, 0.7], [0.6, 0.7]]],
  '5': [[[0.55, 0], [0.1, 0], [0.05, 0.45], [0.4, 0.4], [0.55, 0.7], [0.35, 1], [0.05, 0.9]]],
  '6': [[[0.5, 0.05], [0.15, 0.3], [0.05, 0.75], [0.3, 1], [0.55, 0.75], [0.3, 0.5], [0.08, 0.65]]],
  '7': [[[0.05, 0], [0.6, 0], [0.25, 1]]],
  '8': [[[0.3, 0.5], [0.08, 0.25], [0.3, 0], [0.52, 0.25], [0.3, 0.5], [0.05, 0.75], [0.3, 1], [0.55, 0.75], [0.3, 0.5]]],
  '9': [[[0.52, 0.35], [0.3, 0.5], [0.08, 0.3], [0.3, 0], [0.52, 0.25], [0.45, 1]]],
  '+': [[[0.3, 0.25], [0.3, 0.85]], [[0.0, 0.55], [0.6, 0.55]]],
  '-': [[[0.05, 0.55], [0.55, 0.55]]],
  '=': [[[0.05, 0.4], [0.55, 0.4]], [[0.05, 0.7], [0.55, 0.7]]],
  'x': [[[0.05, 0.35], [0.55, 1]], [[0.55, 0.35], [0.05, 1]]],
  ' ': [],
};
function textStrokes(text, u0, v0, h) {   // board uv (board ~3:1 -> glyph width in u = 0.6 h / aspect)
  const aspect = 2.0, out = []; let u = u0;
  for (const ch of String(text)) {
    for (const st of G[ch] || []) out.push(st.map(([x, y]) => [+(u + x * h / aspect).toFixed(4), +(v0 + y * h).toFixed(4)]));
    u += (ch === ' ' ? 0.45 : 0.8) * h / aspect;
  }
  return out;
}
function bbox(strokes) { let a = [1, 1, 0, 0]; for (const s of strokes) for (const [u, v] of s) a = [Math.min(a[0], u), Math.min(a[1], v), Math.max(a[2], u), Math.max(a[3], v)]; return a; }
// a sum, one line per item
function stubItems(lines) {
  return lines.map((ln, i) => { const strokes = textStrokes(ln.text, ln.u ?? 0.06, ln.v ?? (0.36 + 0.19 * i), ln.h ?? 0.13); return {id: 'i' + i, kind: 'math', say_before: ln.say, strokes, bbox: bbox(strokes), est_s: 3}; });
}
function makeRig(opt = {}) {
  let teach = null;
  const X = makeMuse(opt.seed || 5, {onAction: e => { X && X.ev && X.ev.push(e); if (teach) teach.onAction(e); }});
  const sent = [], prompts = [];
  const hub = {queue: [], items: opt.items || stubItems([{text: '2 + 2', say: 'First, two plus two.'}, {text: '= 4', say: 'That makes four.'}]), latency: opt.brainLatency ?? 1.0, gap: opt.itemGap ?? 0.4};
  const now = () => X.m.now() * 1000;
  function send(m) {
    sent.push(Object.assign({at: X.m.now()}, m));
    if (m.type === 'teach_plan' || m.type === 'teach_erase') {
      const items = m.type === 'teach_erase' ? [{id: 'e0', kind: 'erase', regions: /all/.test(m.what) ? [[0, 0, 1, 1]] : hub.eraseRegions || [[0, 0, 0.5, 0.5]]}] : hub.items;
      items.forEach((it, i) => hub.queue.push({at: X.m.now() + hub.latency + i * hub.gap, msg: {type: 'board_item', id: m.id, item: JSON.parse(JSON.stringify(it))}}));
      hub.queue.push({at: X.m.now() + hub.latency + items.length * hub.gap, msg: {type: 'board_plan_done', id: m.id}});
    }
    if (m.type === 'teach_cancel') hub.queue = hub.queue.filter(q => q.msg.id !== m.id);
  }
  const voice = {lat: opt.voiceLat ?? 0.6, audible: false, cur: null};
  function prompt(text, o) {
    const P = {text, o, at: X.m.now(), state: 'queued', cancel() { P.state = 'dropped'; }};
    prompts.push(P);
    if (opt.noVoice) return null;
    const line = (/"([^"]*)"\s*$/.exec(text) || [])[1] || 'okay';
    P.audioAt = P.at + voice.lat; P.endAt = P.audioAt + line.length / 15;
    return P;
  }
  teach = Teach.createTeach({engine: () => X.m.engine, send, prompt, now, audible: () => voice.audible, history: () => [{role: 'user', text: 'what is two plus two?'}], cfg: opt.cfg, userSpeaking: () => !!voice.userTalking, stanceHook: opt.stanceHook, log: opt.log});
  teach.setConfig(X.files.props);
  const inks = [];
  function tick(n = 1) {
    for (let i = 0; i < n; i++) {
      X.step(1, !!opt.rows);
      const t = X.m.now();
      while (hub.queue.length && hub.queue[0].at <= t) teach.onHub(hub.queue.shift().msg);
      voice.audible = false;
      for (const P of prompts) {
        if (P.state === 'dropped' || P.state === 'done') continue;
        if (P.audioAt !== undefined && t >= P.audioAt && P.state === 'queued') { P.state = 'speaking'; P.o && P.o.onAudio && P.o.onAudio({}); }
        if (P.state === 'speaking') { voice.audible = true; if (t >= P.endAt) { P.state = 'done'; P.o && P.o.onDone && P.o.onDone({}); } }
      }
      teach.frame(X.DT, t);
      // the ink of this tick (flushed with the rig stamp) + where the paw / tip really are
      const last = sent[sent.length - 1];
      if (last && last.type === 'board_ink' && last.at === t && !last._seen) { last._seen = true; inks.push(Object.assign({paw: pawTip()}, last)); }
    }
  }
  function pawTip() { const k = X.m.engine.kit, E = k.E, out = {}; for (const s of ['l', 'r']) out[s] = k.fwdW(s, E.paw[s].h); return out; }
  function until(pred, maxS = 60) { for (let i = 0; i < maxS * 30; i++) { tick(1); if (pred()) return true; } return false; }
  return {X, teach, sent, prompts, hub, voice, tick, until, inks, Teach, stubItems, textStrokes, now: () => X.m.now()};
}
// erase: the hub's board canvas, simulated (app/teach_hub.py _ink: a chalk line = its width; a felt stamp = a rect
// ru*0.8 x rv half-size, a swept stamp = such rects along its segment, each leaving 3% of the chalk): what is left of the
// chalk drawn by `segs` (board_ink segments in order). keep(seg) picks the chalk lines that count (default: all)
function boardResidue(segs, keep) {
  const W = 512, H = 256, A = new Float32Array(W * H), D = new Uint8Array(W * H), hw = 0.0055;
  const rect = (u0, v0, u1, v1, f) => { for (let y = Math.max(0, Math.floor(v0 * H)); y <= Math.min(H - 1, Math.ceil(v1 * H)); y++) for (let x = Math.max(0, Math.floor(u0 * W)); x <= Math.min(W - 1, Math.ceil(u1 * W)); x++) f(y * W + x); };
  for (const sg of segs) {
    if (sg.erase) {
      const hu = sg.ru * 0.8, hv = sg.rv, P = sg.pts, pts = [];
      if (P.length >= 2 && sg.sweep) { const n = Math.max(1, Math.ceil(Math.max(Math.abs(P[1][0] - P[0][0]) / hu, Math.abs(P[1][1] - P[0][1]) / hv) * 1.25)); for (let i = 0; i <= n; i++) pts.push([P[0][0] + (P[1][0] - P[0][0]) * i / n, P[0][1] + (P[1][1] - P[0][1]) * i / n]); } else pts.push(...P);
      for (const c of pts) rect(c[0] - hu, c[1] - hv, c[0] + hu, c[1] + hv, i => { A[i] *= 0.03; });
      continue;
    }
    const counts = !keep || keep(sg), P = sg.pts;
    for (let i = 0; i < Math.max(1, P.length - 1); i++) {
      const a = P[i], b = P[Math.min(i + 1, P.length - 1)], n = Math.max(1, Math.ceil(Math.hypot((b[0] - a[0]) * W, (b[1] - a[1]) * H)));
      for (let j = 0; j <= n; j++) { const u = a[0] + (b[0] - a[0]) * j / n, v = a[1] + (b[1] - a[1]) * j / n; rect(u - hw / 2, v - hw, u + hw / 2, v + hw, q => { A[q] = 1; if (counts) D[q] = 1; }); }
    }
  }
  let left = 0, all = 0;
  for (let i = 0; i < W * H; i++) if (D[i]) { all++; left += A[i]; }
  return {residue: +(left / Math.max(1, all)).toFixed(4), px: all};
}
module.exports = {makeRig, stubItems, textStrokes, Teach, boardResidue};
