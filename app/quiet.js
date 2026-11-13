/* quiet.js — talkativeness and quiet mode for the live avatar.
 *
 * window.AvatarQuiet = {create, LEVELS, POLICY, addressed, endPhrase}   (also module.exports; UMD, no dependencies)
 *
 * The voice model only REQUESTS a change through a function call (set_talkativeness / be_quiet / end_quiet, rtc.py);
 * this controller OWNS the state and the timer, and enforces it through voice.js's harness hooks (setTalk / setQuiet /
 * respondNow, see "quiet" in voice.js):
 *   talkativeness  5 levels. Each = an instructions suffix (Realtime: session.update instructions = the session's own
 *                  instructions + suffix; Gemini Live / GPT Live: a silent note), a hard output cap where the API has one
 *                  (Realtime max_output_tokens; Gemini: the harness stops a reply's audio after `gemWords` words), and the
 *                  unprompted-speech gate (voice.js canRespond / prompt({unprompted}): min gap + per-minute cap, off at
 *                  "minimal") + the nonverbal turn-bid threshold (turnbid.js setThreshold). Lasts the page session
 *                  (re-applied to every re-dialed call through attach()).
 *   quiet          be_quiet(duration_s, wake_on_speech). The harness mutes the avatar's voice bus and blocks every response:
 *                  Realtime turn_detection.create_response=false (no server turn responses, no tokens), every page
 *                  response.create dropped, a leaked one cancelled (response.cancel + output_audio_buffer.clear);
 *                  Gemini Live: its turn audio dropped + one interrupting silent note per turn; mock: no reply WAV.
 *                  It keeps LISTENING: the user's words stay in the conversation (transcripts, items) and the page's nonverbal
 *                  layer (nods, gaze, behaviours) runs untouched.
 *     wake_on_speech=false: silent until the timer ends or the user says it may talk again ("okay, you can talk again" —
 *                  matched on the transcript of every utterance the user makes while quiet, endPhrase()).
 *     wake_on_speech=true:  the user's first ADDRESSED utterance (addressed(): real words, not ASR filler / echo / faint
 *                  background speech / looking away) ends the quiet period and gets an answer. Choice (documented): the
 *                  quiet period ENDS at that point — it does not resume after the answer.
 *     timer end:    unmuted, server turn responses back on, NO response created, and unprompted speech stays blocked for
 *                  `calmMs` (20 s) — it resumes naturally on the user's next words, never blurts.
 *   UI: a small status line under the stage (`el`), never drawn on the character.
 *
 * create({clock:{now(), setTimeout, clearTimeout}, voice: () => voiceApi, el, nod(kind), turnBid: () => {setThreshold},
 *         attention: () => 0..1|null, name: () => 'Muse', onChange(state), log(rec)})
 *   -> {handles(name), call(name, args) -> tool result, setTalk(levelOrStep), quiet(durS, wake), end(why), attach(voice),
 *       onQuietTurn(seg) -> 'wake'|'hold', state(), stats(), level(), policy()}
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.AvatarQuiet = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';

  const LEVELS = ['minimal', 'brief', 'normal', 'chatty', 'talkative'];
  const HEAD = 'How much you talk (he chose this; it overrides any earlier guidance on reply length): ';
  // maxTokens: Realtime max_output_tokens (audio + transcript + tool-call tokens of ONE response). Measured on
  // gpt-realtime-2.1 (tests/quiet_live.py, 2026-11-09): ~10 output tokens per spoken word (plus ~70 a reply); mean reply 7.7
  // words at minimal, 16.7 at brief, 34.0 at normal, 76.0 at talkative (tests/fixtures/quiet_live.json). The cap is a backstop ~3x the asked length (and room for a
  // do_action call), so a reply is cut only when it runs far past it. gemWords: the same backstop on Gemini Live
  // (words of its transcript; the harness stops the reply's audio there).
  const POLICY = {
    minimal: {suffix: HEAD + 'as little as possible. Answer in a few words — five or fewer whenever that can carry the ' +
                'answer, never more than about ten. No reasons, no descriptions, no follow-up questions, and never speak ' +
                'up unless he talks to you.',
              maxTokens: 320, gemWords: 16, gapMs: Infinity, perMin: 0, off: true, turnTh: 9},
    brief: {suffix: HEAD + 'less than usual. Keep every reply to one short sentence; skip extra details and follow-up ' +
              'questions; speak up on your own only rarely.',
            maxTokens: 520, gemWords: 30, gapMs: 45000, perMin: 1, off: false, turnTh: 0.8},
    normal: {suffix: '', maxTokens: null, gemWords: null, gapMs: 10000, perMin: 4, off: false, turnTh: 0.6},
    chatty: {suffix: HEAD + 'more than usual. Give fuller replies of two to four sentences, add a fun detail or a ' +
               'follow-up question, and speak up on your own a bit more often.',
             maxTokens: null, gemWords: null, gapMs: 7000, perMin: 6, off: false, turnTh: 0.5},
    talkative: {suffix: HEAD + 'a lot. Give rich replies of four to six sentences: elaborate, tell little stories, ask ' +
                  'follow-up questions, and speak up on your own often.',
                maxTokens: null, gemWords: null, gapMs: 5000, perMin: 8, off: false, turnTh: 0.45},
  };
  const TOOLS = ['set_talkativeness', 'be_quiet', 'end_quiet'];
  const DUR = {min: 5, max: 7200, dflt: 300};
  const GRACE_MS = 1500;   // a word before the be_quiet call ("okay") may finish playing; then the voice is cut
  const CALM_MS = 20000;   // after the timer ends: no unprompted speech for this long (it resumes on the user's words, never blurts)

  // ---- addressed speech (wake_on_speech): real words directed at the avatar, not noise / TV / its own echo
  const LOW = 0.018;     // mean mic RMS of an utterance below this = faint (across the room, a TV); page voiced = 0.012
  const STRONG = 0.04;
  const FILLER = /^(?:thank you|thanks|thanks for watching|you|bye|bye bye|okay|ok|so|uh|um|uh huh|mm|mhm|hmm|yeah|oh|ah|right|the|i|music|applause|silence|laughter)$/;
  const VOC = /\b(?:hey|hi|hello|excuse me|listen)\b|\b(?:can|could|will|would|do|did|are|were|have) you\b|\btell me\b|\bwhat do you\b/;
  // talking TO it: second person, a question, or a request (judge evidence 2026-11-09: in "quiet until requested" it
  // answered stray fragments like "Thank you. This is the kind of" — plain statements do not wake it)
  const DIRECT = /\byou(?:r|rs|rself)?\b|^(?:what|who|where|when|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should|may|tell|show|say|sing|play|give|look|come|wake|help|please|let s|let us|teach|explain|describe|stop|start|make|try|read|count|open|close|turn|move|dance|wave|go|put|take|pick)\b|^[a-z]+ me\b/;
  const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  function addressed(seg, name) {
    seg = seg || {};
    const text = norm(seg.text), words = text ? text.split(' ') : [];
    const nm = name ? norm(name) : '';
    const voc = !!text && (VOC.test(text) || (!!nm && new RegExp('\\b' + nm + '\\b').test(text)));
    const direct = voc || DIRECT.test(text.replace(/\bthank you\b|\byou know\b/g, ' ').trim()) || /\?\s*$/.test(String(seg.text || ''));
    const lvl = Number.isFinite(seg.level) ? seg.level : null, att = Number.isFinite(seg.attention) ? seg.attention : null;
    if (seg.echo) return {ok: false, why: 'echo of its own voice'};
    if (!text) {
      // the mock vendor has no ASR: an utterance of real length and level counts
      if (seg.mode === 'mock' && (seg.durMs || 0) >= 600 && (lvl === null || lvl >= LOW)) return {ok: true, why: 'mock utterance'};
      return {ok: false, why: 'no words'};
    }
    if (FILLER.test(text)) return {ok: false, why: 'filler / ASR noise'};
    if (words.length < 2 && !voc) return {ok: false, why: 'too short'};
    if (Number.isFinite(seg.durMs) && seg.durMs > 0 && seg.durMs < 350) return {ok: false, why: 'too brief'};
    if (lvl !== null && lvl < LOW && !voc && !(att !== null && att >= 0.5)) return {ok: false, why: 'faint (background)'};
    if (att !== null && att < 0.15 && !voc && !(lvl !== null && lvl >= STRONG)) return {ok: false, why: 'not facing it'};
    if (!direct && !(att !== null && att >= 0.5)) return {ok: false, why: 'not said to it'};
    return {ok: true, why: voc ? 'spoke to it' : direct ? 'asked it' : 'speech while facing it'};
  }
  // "okay, you can talk again" (any phrasing that lifts the quiet; a negation keeps it)
  const END = [
    /\byou (?:can|may|could) (?:talk|speak|chat)(?: again| now| to me)?\b/, /\b(?:start|go ahead and|feel free to) (?:talk|speak)ing?\b/,
    /\b(?:stop|done) being quiet\b/, /\bquiet time(?: is| s)? over\b/, /\bno (?:more|longer) quiet\b/, /\bun ?mute\b/,
    /\b(?:end|cancel|stop) (?:the )?(?:quiet|silence)\b/, /\btalk to me(?: again| now)?\b/, /\byou can come back\b/,
    /\b(?:okay|ok|alright|all right),? (?:talk|speak)(?: again| now)?\b/, /\bbreak (?:the|your) silence\b/,
  ];
  const NEG = /\b(?:don ?t|do not|never|not yet|no) (?:talk|speak|you can)\b|\bcan ?t talk\b|\bstay quiet\b|\bkeep quiet\b/;
  function endPhrase(text) { const t = norm(text); return !!t && !NEG.test(t) && END.some(r => r.test(t)); }

  /* a bare duration ("two minutes", "for 90 seconds", "half an hour") -> seconds, else 0 */
  const NUM = {a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, fifteen: 15, twenty: 20,
    thirty: 30, forty: 40, 'forty five': 45, fifty: 50, sixty: 60, ninety: 90, 'a couple of': 2, 'a few': 3};
  function durationOf(text) {
    const t = norm(text).replace(/^(?:for|just|like|about|maybe|i said|i meant|make it)\s+/g, '').replace(/\s+please$/, '');
    if (/^half an hour$/.test(t)) return 1800;
    const m = /^(\d+|[a-z]+(?: [a-z]+){0,2}) (second|seconds|minute|minutes|hour|hours)$/.exec(t); if (!m) return 0;
    const n = /^\d+$/.test(m[1]) ? +m[1] : NUM[m[1]]; if (!n) return 0;
    return n * (/^s/.test(m[2]) ? 1 : /^m/.test(m[2]) ? 60 : 3600);
  }
  const fmt = s => { s = Math.max(0, Math.ceil(s)); const m = Math.floor(s / 60), r = s % 60; return m + ':' + (r < 10 ? '0' : '') + r; };

  function create(o) {
    o = o || {};
    const C = o.clock || {now: () => (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      setTimeout: (f, ms) => setTimeout(f, ms), clearTimeout: id => clearTimeout(id)};
    const S = {level: 'normal', quiet: false, wake: false, until: 0, startedAt: 0, timer: 0, tick: 0, n: {quiet: 0, woke: 0,
      ended: 0, timer: 0, talk: 0, held: 0}, last: null, hist: []};
    // the page's current voice + the one that attached last (a re-dialed call warms up before the page swaps to it)
    const voices = () => { const a = [o.voice ? o.voice() : null, S.att].filter(Boolean); return a[0] === a[1] ? [a[0]] : a; };
    const each = f => { for (const v of voices()) { try { f(v); } catch (e) {} } };
    const log = rec => { S.hist.push(Object.assign({t: Math.round(C.now())}, rec)); if (S.hist.length > 80) S.hist.splice(0, 40);
      if (o.log) { try { o.log(Object.assign({type: 'quiet'}, rec)); } catch (e) {} } };
    const pol = () => POLICY[S.level];

    function ui() {
      const st = state();
      if (o.onChange) { try { o.onChange(st); } catch (e) {} }
      const el = o.el; if (!el) return;
      let txt = '';
      if (st.quiet) txt = '🤫 Quiet · ' + fmt(st.left_s) + ' left · ' + (st.wake ? 'answers if you talk to it' : 'say "you can talk again" to end');
      else if (S.level !== 'normal') txt = 'Talk: ' + S.level;
      el.textContent = txt; el.style.display = txt ? '' : 'none';
    }
    function scheduleTick() {
      C.clearTimeout(S.tick); S.tick = 0;
      if (!S.quiet || !o.el) return;
      S.tick = C.setTimeout(() => { ui(); scheduleTick(); }, 1000);
    }

    function applyTalk(v) {
      const p = pol(), t = {level: S.level, suffix: p.suffix, maxTokens: p.maxTokens, gemWords: p.gemWords, gapMs: p.gapMs, perMin: p.perMin, off: p.off};
      if (v) { try { if (v.setTalk) v.setTalk(t); } catch (e) {} } else each(x => { if (x.setTalk) x.setTalk(t); });
      // (the perception layer may start after the call: set again a little later; idempotent)
      const bid = () => { const tb = o.turnBid ? o.turnBid() : null; if (tb && tb.setThreshold) { try { tb.setThreshold(pol().turnTh); } catch (e) {} } };
      bid(); C.clearTimeout(S.bidT1); C.clearTimeout(S.bidT2); S.bidT1 = C.setTimeout(bid, 4000); S.bidT2 = C.setTimeout(bid, 15000);
    }
    function setTalk(x) {
      const i0 = LEVELS.indexOf(S.level); let lv = String(x || '').toLowerCase();
      if (lv === 'less' || lv === 'quieter') lv = LEVELS[Math.max(0, i0 - 1)];
      else if (lv === 'more') lv = LEVELS[Math.min(LEVELS.length - 1, i0 + 1)];
      else if (lv === 'reset' || lv === 'default') lv = 'normal';
      if (!POLICY[lv]) return {ok: false, reason: 'unknown level ' + String(x).slice(0, 20), level: S.level};
      const from = S.level; S.level = lv; S.n.talk++;
      applyTalk(); log({ev: 'talk', from, to: lv}); ui();
      return {ok: true, level: lv, from, changed: from !== lv};
    }

    function quiet(durS, wake) {
      let d = +durS; if (!Number.isFinite(d) || d <= 0) d = DUR.dflt; d = Math.max(DUR.min, Math.min(DUR.max, d));
      const t = C.now(), was = S.quiet;
      S.quiet = true; S.wake = !!wake; S.startedAt = t; S.until = t + d * 1000; S.n.quiet++;
      C.clearTimeout(S.timer); S.timer = C.setTimeout(() => end('timer'), d * 1000);
      each(v => { if (v.setQuiet) v.setQuiet(true, {wake: S.wake, graceMs: was ? 0 : GRACE_MS}); });
      if (!was && o.nod) { try { o.nod('quiet'); } catch (e) {} }
      log({ev: 'quiet', dur_s: d, wake: S.wake, extended: was}); ui(); scheduleTick();
      return {ok: true, quiet_for_s: d, wake_on_speech: S.wake};
    }
    /* why: 'timer' (no reply, calm resume) | 'wake' (addressed speech — the voice answers the user) | 'phrase' ("you can talk
       again" — answered) | 'tool' (end_quiet) | 'page' */
    function end(why) {
      if (!S.quiet) return {ok: true, already: true};
      S.quiet = false; C.clearTimeout(S.timer); S.timer = 0; C.clearTimeout(S.tick); S.tick = 0;
      const heard = Math.round((C.now() - S.startedAt) / 1000);
      if (why === 'timer') S.n.timer++; else if (why === 'wake') S.n.woke++; else S.n.ended++;
      each(v => { if (v.setQuiet) v.setQuiet(false, {calmMs: why === 'timer' ? CALM_MS : 0}); });
      each(v => { if (!v.prompt) return;           // context only: no response is created by these notes
        if (why === 'timer') v.prompt('Your quiet time is over: you may talk again when he talks to you. Do not announce it.', {respond: false});
        else if (why === 'wake' || why === 'phrase') v.prompt('Your quiet time is over (he ' + (why === 'wake' ? 'talked to you' : 'said you can talk again') + '): your voice is heard again.', {respond: false}); });
      log({ev: 'end', why, after_s: heard}); ui();
      return {ok: true, ended: why};
    }
    /* voice.js asks while quiet: the user finished an utterance (Realtime / mock) or the model began a reply to it (Gemini).
       'wake' = the quiet period is over and the voice should answer; 'hold' = stay silent (it still heard the user). */
    function onQuietTurn(seg) {
      seg = Object.assign({}, seg || {});
      if (!S.quiet) return 'wake';
      if (seg.attention === undefined && o.attention) { try { const a = o.attention(); if (Number.isFinite(a)) seg.attention = a; } catch (e) {} }
      if (endPhrase(seg.text)) { S.last = {seg, why: 'end phrase'}; end('phrase'); return 'wake'; }
      // "…for a—" got cut off before the time, then "two minutes": a bare duration right after sets the period (no reply)
      const d = durationOf(seg.text);
      if (d && C.now() - S.startedAt < 20000) { S.last = {seg, why: 'duration'}; quiet(d, S.wake); return 'hold'; }
      if (S.wake) {
        const a = addressed(seg, o.name ? o.name() : '');
        S.last = {seg, verdict: a};
        if (a.ok) { log({ev: 'addressed', why: a.why, text: String(seg.text || '').slice(0, 80)}); end('wake'); return 'wake'; }
        S.n.held++; log({ev: 'not_addressed', why: a.why, text: String(seg.text || '').slice(0, 80)});
        return 'hold';
      }
      S.n.held++; S.last = {seg, why: 'quiet (no wake)'};
      return 'hold';
    }
    function call(name, args) {
      const r = call1(name, args || {});
      log({ev: 'call', name, args: args || {}, ran: {ok: r.ok !== false, level: r.level, quiet_for_s: r.quiet_for_s, reason: r.reason || r.error}});
      return r;
    }
    function call1(name, args) {
      if (name === 'set_talkativeness') {
        const r = setTalk(args.level);
        return r.ok ? {ok: true, level: r.level, note: 'From now on you talk this way: ' + (POLICY[r.level].suffix ? POLICY[r.level].suffix.slice(HEAD.length) : 'as usual') + ' Do not comment on the change beyond a word like "sure".'} : r;
      }
      if (name === 'be_quiet') {
        const r = quiet(args.duration_s, !!args.wake_on_speech);
        return Object.assign(r, {note: 'Your voice is muted by the app now. Say nothing more' + (r.wake_on_speech ? ' until he talks to you.' : '.')});
      }
      if (name === 'end_quiet') return end('tool');
      return {ok: false, error: 'unknown tool'};
    }
    function attach(v) {      // a new / re-dialed call: re-apply the level and any quiet period still running
      if (!v) return;
      S.att = v; applyTalk(v);
      if (S.quiet && v.setQuiet) { try { v.setQuiet(true, {wake: S.wake, graceMs: 0}); } catch (e) {} }
    }
    function state() {
      return {level: S.level, quiet: S.quiet, wake: S.wake, left_s: S.quiet ? Math.max(0, (S.until - C.now()) / 1000) : 0};
    }
    return {TOOLS, handles: n => TOOLS.includes(n), call, setTalk, quiet, end, attach, onQuietTurn, state,
      level: () => S.level, policy: () => Object.assign({level: S.level}, pol()),
      stats: () => Object.assign(state(), {n: Object.assign({}, S.n), last: S.last, hist: S.hist.slice(-20)})};
  }
  /* the in-page cue: one small status line after `after` (the stage), never drawn on the character */
  function cue(doc, after) {
    doc = doc || (typeof document !== 'undefined' ? document : null); if (!doc) return null;
    let el = doc.getElementById('quietcue'); if (el) return el;
    el = doc.createElement('div'); el.id = 'quietcue'; el.setAttribute('aria-live', 'polite');
    el.style.cssText = 'display:none;margin:6px 2px;padding:4px 10px;border-radius:999px;font-size:12.5px;opacity:.85;' +
      'border:1px solid rgba(128,128,128,.45);width:fit-content;font-variant-numeric:tabular-nums';
    const a = after && typeof after === 'string' ? doc.getElementById(after) : after;
    if (a && a.parentNode) a.parentNode.insertBefore(el, a.nextSibling); else if (doc.body) doc.body.appendChild(el);
    return el;
  }
  return {create, cue, LEVELS, POLICY, TOOLS, DUR, CALM_MS, GRACE_MS, addressed, endPhrase, durationOf, LOW, STRONG};
});
