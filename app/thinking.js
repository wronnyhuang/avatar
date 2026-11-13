/* thinking.js — an instant THINKING gesture while the planner writes a commanded move.
 *
 *   const K = AvatarThinking.createThinker({motion, now, timers, rand, gateMs, maxMs, switchMs, log, enabled, arms})
 *                        arms() === false = an armless character: gaze + head + brow variants only
 *                        fingers() === false = no index finger of its own (every character but the muse): the
 *                        raised-finger alternate is never picked (motion.think refuses it for such a body anyway)
 *   K.begin(key, info)   a commanded action is known and its move is not (perform.js P.call sent {type:'plan'}; voicetools
 *                        holds a do_action for the user's final words). After the gate (GATE_MS) with no move yet, one of the
 *                        motion's thinking variants starts (motion.think) — chosen at random, never the one just used.
 *                        Mostly the MAIN pose (eyes down and away, a paw on the chin), now and then an alternate.
 *                        A second begin() while one is pending / on joins it (same gesture, one more key).
 *   K.known(key)         the move for that key is known now (a cache / library answer, first poses): a pending gate is
 *                        dropped (an instant move never shows the thinking); an on-going one keeps going until the move starts.
 *   K.end(key, why)      the move started, failed, was refused or replaced: the key leaves; with no key left the thinking
 *                        fades (motion.thinkEnd) or the gate is dropped. motion.js also fades it by itself the frame a
 *                        keyframe job starts or any other command runs, so a missed end() can never leave it on.
 *   K.cancel(why)        everything off now (stop / relax / reset).
 *   K.stats() / K.list() tests + voiceReport: {begun, shown, skipped, refused, ...} and the last sessions with their timing.
 * Long waits switch to another variant every switchMs; after maxMs it fades on its own (never left stuck on).
 * Client-side only: no network, no notes to the voice model, nothing in the action log.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarThinking = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const GATE_MS = 100;       // start only when no move came within this long (cache hits / library routes answer sooner); 100 ms keeps the first visible frame <= 200 ms
  const MAX_MS = 12000;      // fade on its own after this long
  const SWITCH_MS = 4500;    // a long wait changes variant (a crossfade) every this long
  const VARIANTS = ['chin', 'up', 'tilt', 'hmm', 'finger'];
  // design note: the MAIN pose (down-and-away gaze, paw on the chin; motion.js falls back to 'away' without a free paw)
  // most of the time; the others are occasional alternates, never the same alternate twice running, never 3 mains running
  const MAIN = 'chin', MAIN_P = 0.65;

  function createThinker(d) {
    d = d || {};
    const now = d.now || (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    const T = d.timers || {set: (f, ms) => setTimeout(f, ms), clear: id => clearTimeout(id)};
    const rand = typeof d.rand === 'function' ? d.rand : Math.random;
    const gateMs = Number.isFinite(+d.gateMs) && d.gateMs !== null ? Math.max(0, +d.gateMs) : GATE_MS;
    const maxMs = Number.isFinite(+d.maxMs) && d.maxMs !== null ? +d.maxMs : MAX_MS;
    const switchMs = Number.isFinite(+d.switchMs) && d.switchMs !== null ? +d.switchMs : SWITCH_MS;
    const motion = () => { try { return d.motion ? d.motion() : null; } catch (e) { return null; } };
    const enabled = () => { try { return typeof d.enabled === 'function' ? !!d.enabled() : d.enabled !== false; } catch (e) { return true; } };
    const arms = () => { try { return typeof d.arms === 'function' ? d.arms() !== false : d.arms !== false; } catch (e) { return true; } };
    const fingers = () => { try { return typeof d.fingers === 'function' ? d.fingers() !== false : d.fingers !== false; } catch (e) { return true; } };
    const K = {cur: null, last: null, recent: [], hist: [], n: 0, c: {begun: 0, shown: 0, skipped: 0, refused: 0, switched: 0, capped: 0, joined: 0}};
    function log(rec) { if (d.log) { try { d.log(rec); } catch (e) {} } }

    function pick(prev) {
      const mains = K.recent.slice(-2).filter(v => v === MAIN || v === 'away').length;
      if (mains < 2 && rand() < MAIN_P) return MAIN;
      const pool = VARIANTS.filter(v => v !== MAIN && v !== prev && (v !== 'finger' || (arms() && fingers())));
      return pool[Math.floor(rand() * pool.length) % pool.length];
    }
    function begin(key, info) {
      if (key == null || !enabled()) return null;
      key = String(key);
      const m = motion();
      if (!m || typeof m.think !== 'function') return null;
      const t = now();
      if (K.cur && !K.cur.done) { K.cur.keys.add(key); K.c.joined++; return K.cur.id; }
      K.c.begun++;
      const s = {id: 'k' + (++K.n), keys: new Set([key]), t0: t, state: 'gate', variant: null, shown_ms: null, end_ms: null,
        why: null, info: info || null, variants: [], done: false, timers: {}};
      K.cur = s;
      if (gateMs > 0) s.timers.gate = T.set(() => show(s), gateMs);
      else show(s);
      return s.id;
    }
    function show(s) {
      if (s.done || s.state !== 'gate') return;
      T.clear(s.timers.gate);
      const r = play(s, pick(K.last));
      if (!r) { s.state = 'refused'; K.c.refused++; close(s, 'refused'); return; }
      s.state = 'on'; s.shown_ms = now(); K.c.shown++;
      log({think: s.id, shown: s.variant, gate_ms: Math.round(s.shown_ms - s.t0)});
      if (switchMs > 0) s.timers.sw = T.set(function sw() {
        if (s.done || s.state !== 'on') return;
        const m = motion(), th = m && typeof m.thinking === 'function' ? m.thinking() : null;
        if (th && !th.on) { close(s, 'motion took over'); return; }   // a move / command already faded it: never restart it
        if (play(s, s.variant === MAIN || s.variant === 'away' ? pick(s.variant) : MAIN)) K.c.switched++;
        s.timers.sw = T.set(sw, switchMs);
      }, switchMs);
      if (maxMs > 0) s.timers.cap = T.set(() => { if (!s.done) { K.c.capped++; finish(s, 'cap'); } }, Math.max(0, maxMs - (s.shown_ms - s.t0)));
    }
    function play(s, v) {
      const m = motion(); if (!m || typeof m.think !== 'function') return null;
      let r = null;
      try { r = m.think({variant: v, avoid: v === MAIN ? null : (s.variant || K.last), dir: rand() < 0.5 ? -1 : 1, arms: arms()}); } catch (e) { r = null; }
      if (!r || !r.ok) { s.lastRefusal = r && r.reason; return null; }
      s.variant = r.variant; K.last = r.variant; s.variants.push(r.variant);
      K.recent.push(r.variant); if (K.recent.length > 8) K.recent.splice(0, 4);
      return r;
    }
    function known(key) {
      const s = K.cur; if (!s || s.done || !s.keys.has(String(key))) return;
      if (s.state === 'gate') { s.keys.delete(String(key)); if (!s.keys.size) { K.c.skipped++; close(s, 'instant move'); } }
    }
    function end(key, why) {
      const s = K.cur; if (!s || s.done || !s.keys.delete(String(key))) return false;
      if (s.keys.size) return false;
      finish(s, why || 'end');
      return true;
    }
    function finish(s, why) {
      if (s.done) return;
      if (s.state === 'gate') { K.c.skipped++; close(s, why); return; }
      const m = motion();
      if (s.state === 'on' && m && typeof m.thinkEnd === 'function') { try { m.thinkEnd(why); } catch (e) {} }
      close(s, why);
    }
    function close(s, why) {
      s.done = true; s.why = why; s.end_ms = now();
      for (const k in s.timers) T.clear(s.timers[k]);
      if (K.cur === s) K.cur = null;
      const rec = {id: s.id, keys: [...s.keys], state: s.state, variants: s.variants.slice(), why, t0: Math.round(s.t0),
        gate_ms: s.shown_ms != null ? Math.round(s.shown_ms - s.t0) : null, shown_s: s.shown_ms != null ? +((s.end_ms - s.shown_ms) / 1000).toFixed(2) : 0,
        refusal: s.lastRefusal || undefined};
      K.hist.push(rec); if (K.hist.length > 40) K.hist.splice(0, 20);
      log({think: s.id, end: why, state: s.state, variants: rec.variants});
    }
    function cancel(why) { if (K.cur) finish(K.cur, why || 'cancelled'); else { const m = motion(); if (m && typeof m.thinkEnd === 'function') { try { m.thinkEnd(why || 'cancelled'); } catch (e) {} } } }
    function active() { return K.cur && !K.cur.done ? {id: K.cur.id, state: K.cur.state, variant: K.cur.variant, keys: [...K.cur.keys]} : null; }
    function stats() { return Object.assign({}, K.c, {active: active(), last: K.hist.slice(-6)}); }
    function list() { return K.hist.slice(); }
    function reset() { cancel('reset'); K.hist = []; K.last = null; K.recent = []; for (const k in K.c) K.c[k] = 0; }
    return {begin, known, end, cancel, active, stats, list, reset};
  }
  return {createThinker, GATE_MS, MAX_MS, SWITCH_MS, VARIANTS, MAIN};
});
