/* behavior.js — PROGRAM LAYER: standing behaviours as small sandboxed per-frame programs
 * Browser: window.Behaviors (one runtime) + window.AvatarBehaviorLib;
 * node: module.exports = the lib (tests, tools/behavior_check.cjs — the server's static check + dry run).
 *
 * A standing request ("whenever I touch my nose, wiggle your ears", "count my blinks and tell me at ten") becomes a
 * program written by the server's coding model (app/behavior_coder.py) against the FIXED sense + act toolkit below
 * (TOOLKIT_DOC is the exact text the coder is taught). Program text = one JS object literal:
 *     ({ setup(api) {...}, per_frame(api) {...}, end(api) {...} })          (setup / end optional)
 *
 * PAGE API (contract): Behaviors.start({name, instruction, code, priority=5}) → id | null (null: it did not compile or
 *   its setup failed — the reason is in Behaviors.lastError and an {type:'behavior', phase:'error'} event);
 *   Behaviors.change(nameOrId, {code, instruction, priority?}) → true | false — HOT SWAP: the new program's setup gets
 *   the old program's api.state (and its clock, claims-in-progress keep their blend); a failing new version leaves the
 *   old one running; Behaviors.stop(nameOrId | 'all') → number stopped; Behaviors.list() →
 *   [{id, name, instruction, priority, groups, since, frames, errors, status, missing, ms_p50, ms_max}];
 *   Behaviors.onEvent(fn) → unsubscribe. Extras: resume(nameOrId), feed(ev), attach(deps), request(instruction, opts)
 *   → Promise (server round trip), onCode(msg) (the server's {type:'behavior_code'} answer), tick(ctx) (hosts), stats().
 *
 * LAYERING (contract "Layers"): lip sync > commanded action > PROGRAMS (by priority, ties → newest) > reflex network /
 *   listener > rules. Composition happens in motion.js programPass (after listenerPass): per rig channel
 *   final = lerp(below, alpha·v + beta, w) with learnedPass's affine map (alpha = what survives under commanded
 *   actions, beta = commanded values + reflex offsets), and the map is updated to [alpha(1−w), alpha·w·v + beta] — so in
 *   learned / unified mode the server sees the program's channels as OWNED (the unified worker turns them into
 *   UnifiedPlanner.set_override: the network inpaints around them, no fighting, no jump on release). SPEECH_GROUP
 *   (mouth_open, tongue_out) yields to speech while voiced. A program writes only groups it claimed and won.
 *   Per channel the program's weight eases in (τ 25 ms) and out (τ 120 ms; the last value is held while it fades).
 *
 * SANDBOX: the text is tokenized (strings / templates / regex aware), refused on forbidden syntax / names (this, class,
 *   async, generators, import, eval, Function, constructor, __proto__, prototype, getPrototypeOf, names starting "__",
 *   escapes outside strings…), instrumented (every loop condition, function entry and arrow body calls a time guard;
 *   for…of / for…in iterate through guarded generators; every computed member access / computed key goes through a
 *   key filter), and compiled with new Function in strict mode with EVERY global name (window, document, fetch, timers,
 *   … computed from globalThis + its prototype chain) shadowed by an undefined parameter. Only frozen safe copies of
 *   Math, JSON, Number, String, Boolean, Array, Object, Map, Set, Error, isFinite, isNaN, parseFloat, parseInt and a
 *   console that logs into the program's own ring are visible. The api object (and every sense snapshot) is frozen.
 *   BUDGET: per_frame is timed; > BUDGET.frameMs (1 ms) in ≥ BUDGET.overN of the last 30 frames → PAUSED + event;
 *   a single call over BUDGET.hardMs (8 ms; setup 50 ms) is cut off by the guard (frame aborted, counts as 3 overruns);
 *   3 cut-offs in 10 s → paused. An exception stops the program with an {phase:'error'} event (state kept for a fix).
 *
 * PAGE NOTES (integration): after startMotion() created D.motion —
 *   Behaviors.attach({motion: () => D.motion, character: () => S.cfg && S.cfg.character,
 *     doAction: (description, meta) => D.perf ? D.perf.call(description, {source: 'behavior', behavior: meta.name}) : null,
 *     setStyle: s => sendCtl(Object.assign({type: 'style'}, s)),
 *     requestCode: msg => sendCtl(msg)});                   // {type:'behavior_code', id, instruction, name, …} → server
 *   server → page {type:'behavior_code', …} → Behaviors.onCode(j) (page.html dispatch, added in this branch);
 *   Behaviors.feed({type:'said', text}) for his transcripts; learned gestures {type:'gesture', name, score},
 *   attention verdicts {type:'attention', what, worth_reply}, turn bids {type:'turn_bid', cue, deliberateness} as they come.
 *   Perception events and the motion's own action events are read directly from motion.step's inputs (no feed needed).
 *   Voice tools: start_behavior(instruction, name?, priority?) → Behaviors.request(instruction, {name, priority});
 *   change_behavior(name, instruction) → Behaviors.request(instruction, {name, edit: true}) (the server gets prev_code);
 *   stop_behavior(name|'all') → Behaviors.stop(…); list_behaviors() → Behaviors.list().
 *   Behaviors.onEvent(fn): {type:'behavior', phase:'start'|'changed'|'end'|'error'|'paused'|'warning'|'unsupported', …},
 *   {type:'rule_moment', name, event, data, reply} (api.emit → the voice model), {type:'behavior_action', …}.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  if (root && typeof window !== 'undefined' && root === window) {
    root.AvatarBehaviorLib = lib;
    if (!root.Behaviors) root.Behaviors = lib.createBehaviors();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : () => Date.now();
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const lerp = (a, b, s) => a + (b - a) * s;
  const lp = (prev, x, dt, tau) => prev + (x - prev) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));

  const BUDGET = {frameMs: 1.0, overN: 6, window: 30, hardMs: 8, setupMs: 50, endMs: 8, hardTrips: 3, hardWindowS: 10};
  const LIMITS = {codeChars: 12000, lines: 400, actionGapS: 1.0, actionsPerMin: 20, emitGapS: 0.4, emitsPerMin: 30,
    styleGapS: 0.25, logRing: 40, eventRing: 400, arrayMax: 100000};
  const W_IN = 0.025, W_OUT = 0.12, XFADE = 0.15;

  // ------------------------------------------------------------------ characters (planner_prompts/MANIFEST.json traits)
  const T = (o) => Object.assign({arms: true, brows: true, tongue: true, bounce: true, ears: false, tail: false, nose: false,
    fingers: false, legs: false, leaves_seat: false, shape: false}, o);
  const TRAITS = {
    fox: T({name: 'Felix', what: 'fox', ears: true, tail: true, nose: true}),
    bear: T({name: 'Bruno', what: 'bear', ears: true, nose: true}),
    rabbit: T({name: 'Rosie', what: 'rabbit', ears: true, nose: true}),
    hippo: T({name: 'Hattie', what: 'hippo', ears: true, nose: true}),
    monkey: T({name: 'Milo', what: 'monkey', ears: true, tail: true, nose: true}),
    orb: T({name: 'Ori', what: 'glowing orb', shape: true}),
    cube: T({name: 'Cubby', what: 'cube', shape: true}),
    drop: T({name: 'Dewdrop', what: 'water drop', shape: true}),
    star: T({name: 'Stella', what: 'star', shape: true}),
    pyramid: T({name: 'Pip', what: 'pyramid', shape: true}),
    blob: T({name: 'Bloop', what: 'jelly blob', shape: true}),
    muse: T({name: 'Muse', what: 'plush companion', ears: true, fingers: true, legs: true, leaves_seat: true}),
  };
  const traitsOf = c => TRAITS[c] || TRAITS.fox;

  // ------------------------------------------------------------------ the act channels (toolkit names → rig columns)
  // Toolkit frame: sided PARTS use the character's OWN side (its right paw is on the screen's LEFT); DIRECTIONS use the
  // screen as he sees it (+ = toward the screen's right / up / toward him). head_pitch is + = up (the rig's is + = nose
  // down: sign −1). Ranges = the rig's (motion.js RANGE).
  const CH = {}, GROUP_CH = {};
  function def(name, rig, group, range, need, desc, sign = 1) {
    CH[name] = {name, rig: [].concat(rig), group, range, need, desc, sign};
    (GROUP_CH[group] = GROUP_CH[group] || []).push(name);
  }
  def('head_yaw', 'head_yaw', 'head', [-0.6, 0.6], null, 'turn the head; + = toward the screen\'s right (its own left)');
  def('head_pitch', 'head_pitch', 'head', [-0.45, 0.45], null, '+ = look up, - = look down', -1);
  def('head_roll', 'head_roll', 'head', [-0.4, 0.4], null, 'tilt the head; + = top of the head toward the screen\'s right');
  def('gaze_x', 'gaze_x', 'gaze', [-1, 1], null, 'eyes sideways; + = toward the screen\'s right (1 ≈ 17°, more needs head_yaw)');
  def('gaze_y', 'gaze_y', 'gaze', [-1, 1], null, 'eyes up/down; + = up');
  def('eyes_closed', ['blink_l', 'blink_r'], 'eyes', [0, 1], null, 'both eyelids, 0 open .. 1 shut (a blink = 0→1→0 in ~0.2 s)');
  def('right_eye_closed', 'blink_l', 'eyes', [0, 1], null, 'its own right eyelid (screen-left) — a wink');
  def('left_eye_closed', 'blink_r', 'eyes', [0, 1], null, 'its own left eyelid (screen-right)');
  def('eye_wide', 'eye_wide', 'eyes', [0, 1], null, 'eyes wide open (surprise)');
  def('squint', 'squint', 'eyes', [0, 1], null, 'narrowed eyes');
  def('right_brow', 'brow_l', 'brows', [-1, 1], 'brows', 'its own right brow, + = raised, - = lowered');
  def('left_brow', 'brow_r', 'brows', [-1, 1], 'brows', 'its own left brow');
  def('brow_knit', 'brow_knit', 'brows', [0, 1], 'brows', 'frown (brows pulled together)');
  def('smile', 'smile', 'mouth', [0, 1], null, 'mouth corners up (rest ≈ 0.1)');
  def('mouth_open', 'mouth_open', 'mouth', [0, 1], null, 'jaw open (speech takes the mouth while it talks)');
  def('tongue_out', 'tongue_out', 'mouth', [0, 1], 'tongue', 'tongue out past the lips');
  def('nose_wiggle', 'nose_wiggle', 'nose', [-1, 1], 'nose', 'nose tip down .. up (alternate it to wiggle)');
  def('nostrils', 'nostril', 'nose', [0, 1], 'nose', 'nostril flare / sniff');
  def('right_ear', 'ear_l', 'ears', [-1, 1], 'ears', 'its own right ear: -1 drooped back .. 1 up/forward');
  def('left_ear', 'ear_r', 'ears', [-1, 1], 'ears', 'its own left ear');
  def('right_ear_turn', 'ear_l_turn', 'ears', [-1, 1], 'ears', 'its own right ear swivel: -1 opening forward/in .. 1 out');
  def('left_ear_turn', 'ear_r_turn', 'ears', [-1, 1], 'ears', 'its own left ear swivel');
  def('body_yaw', 'body_yaw', 'body', [-1, 1], null, 'swivel the whole body; + = toward the screen\'s right (1 = its limit)');
  def('body_lean', 'body_lean', 'body', [-1, 1], null, 'lean sideways; + = toward the screen\'s right');
  def('body_pitch', 'body_pitch', 'body', [-1, 1], null, '+ = lean toward him (the camera), - = lean back');
  def('body_bounce', 'body_bounce', 'body', [-1, 1], 'bounce', 'vertical offset in place; + = up (bob it for a bounce)');
  for (const [own, s] of [['right', 'l'], ['left', 'r']]) {
    const g = own + '_arm', scr = own === 'right' ? 'screen-left' : 'screen-right';
    def(own + '_paw_x', 'hand_' + s + '_x', g, [-1, 1.2], 'arms', `its own ${own} paw (${scr}) sideways from rest in arm lengths: + = OUT to its side, - = across the chest`);
    def(own + '_paw_y', 'hand_' + s + '_y', g, [-1, 1.2], 'arms', 'paw height: 0 = rest at its side, 0.45 = chest, 0.8 = face, 1.0 = straight up overhead');
    def(own + '_paw_z', 'hand_' + s + '_z', g, [-1, 1.2], 'arms', 'paw forward toward him (the camera), 0 = rest');
    def(own + '_paw_open', 'hand_' + s + '_open', g, [0, 1], 'arms', '0 = fist, 0.2 = rest, 1 = open spread palm');
    def(own + '_paw_turn', 'hand_' + s + '_turn', g, [-1, 1], 'arms', 'forearm roll: +1 = palm to the camera when raised');
    def(own + '_thumb', 'thumb_' + s, g, [0, 1], 'arms', 'thumb out (thumbs-up = paw_open 0 + thumb 1)');
    def(own + '_point', 'point_' + s, g, [0, 1], 'arms', 'index finger straight (does not move the arm)');
    def(own + '_shoulder', 'shoulder_' + s, g, [0, 1], 'arms', 'shrug that shoulder');
    def(own + '_fingers', 'fingers_' + s, g, [0, 5], 'fingers', 'how many fingers it holds up (counting), 0..5');
  }
  def('tail_wag', 'tail_wag', 'tail', [-1, 1], 'tail', 'tail side POSITION (alternate it to wag)');
  def('tail_raise', 'tail_raise', 'tail', [0, 1], 'tail', 'tail lift (rest ≈ 0.4)');
  def('tail_curl', 'tail_curl', 'tail', [0, 1], 'tail', 'tail tip curl');
  const GROUPS = Object.keys(GROUP_CH);   // head gaze eyes brows mouth nose ears body right_arm left_arm tail
  const GROUP_ALIAS = {arms: ['right_arm', 'left_arm'], paws: ['right_arm', 'left_arm'], hands: ['right_arm', 'left_arm'],
    right_paw: ['right_arm'], left_paw: ['left_arm'], right_hand: ['right_arm'], left_hand: ['left_arm'], shoulders: ['right_arm', 'left_arm'],
    face: ['eyes', 'brows', 'mouth', 'nose', 'ears'], posture: ['head', 'body'], eyelids: ['eyes'], eyebrows: ['brows'], brow: ['brows'],
    tongue: ['mouth'], ear: ['ears'], eye: ['eyes'], neck: ['head'], torso: ['body'], all: GROUPS.slice()};
  // RIG column → toolkit channel (self.channel / tests)
  const RIG_TO_CH = {};
  for (const c of Object.values(CH)) for (const r of c.rig) if (!RIG_TO_CH[r]) RIG_TO_CH[r] = c.name;
  // the group needs a part: a group whose every channel needs a missing trait is missing
  function hasChannel(character, name) {
    const c = CH[name]; if (!c) return false;
    return !c.need || !!traitsOf(character)[c.need];
  }
  function channelsOf(character) { return Object.keys(CH).filter(n => hasChannel(character, n)); }
  function groupMissing(character, g) { return !(GROUP_CH[g] || []).some(n => hasChannel(character, n)); }
  function expandGroups(list) {
    const out = [], bad = [];
    for (let g of [].concat(list || [])) {
      g = String(g || '').toLowerCase().trim().replace(/[\s-]+/g, '_');
      if (GROUP_CH[g]) { if (!out.includes(g)) out.push(g); } else if (GROUP_ALIAS[g]) { for (const x of GROUP_ALIAS[g]) if (!out.includes(x)) out.push(x); } else bad.push(g);
    }
    return {groups: out, bad};
  }
  const PARTS = {ears: 'ears', ear: 'ears', tail: 'tail', nose: 'nose', arms: 'arms', arm: 'arms', paws: 'arms', paw: 'arms', hands: 'arms',
    hand: 'arms', legs: 'legs', leg: 'legs', feet: 'legs', foot: 'legs', fingers: 'fingers', finger: 'fingers', brows: 'brows', eyebrows: 'brows',
    tongue: 'tongue', bounce: 'bounce', walk: 'leaves_seat', walking: 'leaves_seat', stand: 'leaves_seat', body: true, head: true, eyes: true,
    eye: true, mouth: true, face: true, shoulders: 'arms', thumbs: 'arms'};

  // ------------------------------------------------------------------ tokenizer (strings / templates / regex aware)
  const PUNCTS = ['>>>=', '...', '===', '!==', '**=', '<<=', '>>=', '>>>', '&&=', '||=', '??=', '=>', '==', '!=', '<=', '>=', '&&', '||',
    '??', '?.', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '**', '<<', '>>', '{', '}', '(', ')', '[', ']', ';', ',',
    '<', '>', '+', '-', '*', '/', '%', '&', '|', '^', '!', '~', '?', ':', '=', '.', '@', '#'];
  const KW_EXPR = new Set(['return', 'typeof', 'instanceof', 'in', 'of', 'new', 'delete', 'void', 'throw', 'case', 'do', 'else', 'yield', 'await']);
  const KW_DECL = new Set(['let', 'const', 'var']);
  const KW_NOT_VALUE = new Set([...KW_EXPR, ...KW_DECL, 'if', 'for', 'while', 'switch', 'catch', 'with', 'function', 'export', 'import']);
  const isIdStart = c => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_' || c === '$';
  const isId = c => isIdStart(c) || (c >= '0' && c <= '9');
  const isDigit = c => c >= '0' && c <= '9';
  class CodeError extends Error { constructor(msg, line) { super(msg); this.line = line; } }

  function lineOf(src, pos) { let n = 1; for (let i = 0; i < pos && i < src.length; i++) if (src.charCodeAt(i) === 10) n++; return n; }

  function decodeStr(raw) {   // raw = the literal's inner text (no quotes)
    return raw.replace(/\\(u\{([0-9a-fA-F]+)\}|u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|([\s\S]))/g, (m, _a, cp, u4, x2, ch) => {
      if (cp) { const n = parseInt(cp, 16); return n <= 0x10ffff ? String.fromCodePoint(n) : ''; }
      if (u4) return String.fromCharCode(parseInt(u4, 16));
      if (x2) return String.fromCharCode(parseInt(x2, 16));
      return ({n: '\n', t: '\t', r: '\r', b: '\b', f: '\f', v: '\v', 0: '\0', '\n': ''})[ch] ?? ch;
    });
  }

  function tokenize(src) {
    const toks = [], braces = [], n = src.length;
    let i = 0, nl = false;
    const fail = (msg, at) => { throw new CodeError(msg, lineOf(src, at)); };
    const push = tk => { tk.nl = nl; nl = false; toks.push(tk); };
    const regexOk = () => {
      const p = toks.length ? toks[toks.length - 1] : null;
      if (!p) return true;
      if (p.t === 'num' || p.t === 'str' || p.t === 'regex') return false;
      if (p.t === 'tmpl') return !p.tail;
      if (p.t === 'name') return KW_EXPR.has(p.v);
      return !(p.v === ')' || p.v === ']' || p.v === '}' || p.v === '++' || p.v === '--');
    };
    function readTmpl(start, head) {   // src[start] is '`' (head) or '}' (continuation)
      let j = start + 1;
      while (j < n) {
        const c = src[j];
        if (c === '\\') { j += 2; continue; }
        if (c === '`') { push({t: 'tmpl', head, tail: true, s: start, e: j + 1}); return j + 1; }
        if (c === '$' && src[j + 1] === '{') { braces.push('T'); push({t: 'tmpl', head, tail: false, s: start, e: j + 2}); return j + 2; }
        j++;
      }
      return fail('unterminated template literal', start);
    }
    while (i < n) {
      const c = src[i];
      if (c === '\n' || c === '\r' || c === '\u2028' || c === '\u2029') { nl = true; i++; continue; }
      if (c === ' ' || c === '\t' || c === '\f' || c === '\v' || c === '\ufeff' || c === '\u00a0') { i++; continue; }
      if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n' && src[i] !== '\r') i++; continue; }
      if (c === '/' && src[i + 1] === '*') {
        const j = src.indexOf('*/', i + 2); if (j < 0) fail('unterminated comment', i);
        if (/[\n\r\u2028\u2029]/.test(src.slice(i, j))) nl = true;
        i = j + 2; continue;
      }
      const s = i;
      if (isIdStart(c)) { while (i < n && isId(src[i])) i++; if (src[i] === '\\') fail('escapes in names are not allowed', i); push({t: 'name', v: src.slice(s, i), s, e: i}); continue; }
      if (isDigit(c) || (c === '.' && isDigit(src[i + 1] || ''))) {
        const m = /^(0[xX][0-9a-fA-F_]+n?|0[oO][0-7_]+n?|0[bB][01_]+n?|(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?n?)/.exec(src.slice(i, i + 64));
        i += m ? m[0].length : 1;
        if (i < n && isIdStart(src[i])) fail('bad number', s);
        push({t: 'num', v: src.slice(s, i), s, e: i}); continue;
      }
      if (c === '"' || c === "'") {
        let j = i + 1;
        while (j < n && src[j] !== c) { if (src[j] === '\\') j++; else if (src[j] === '\n' || src[j] === '\r') fail('unterminated string', s); j++; }
        if (j >= n) fail('unterminated string', s);
        push({t: 'str', v: decodeStr(src.slice(i + 1, j)), s, e: j + 1}); i = j + 1; continue;
      }
      if (c === '`') { i = readTmpl(i, true); continue; }
      if (c === '}' && braces.length && braces[braces.length - 1] === 'T') { braces.pop(); i = readTmpl(i, false); continue; }
      if (c === '/' && regexOk()) {
        let j = i + 1, cls = false;
        while (j < n) {
          const d = src[j];
          if (d === '\\') { j += 2; continue; }
          if (d === '\n' || d === '\r') fail('unterminated regular expression', s);
          if (d === '[') cls = true; else if (d === ']') cls = false; else if (d === '/' && !cls) break;
          j++;
        }
        if (j >= n) fail('unterminated regular expression', s);
        j++; while (j < n && isId(src[j])) j++;
        push({t: 'regex', v: src.slice(s, j), s, e: j}); i = j; continue;
      }
      if (c.charCodeAt(0) > 126 || c === '\\') fail(`character '${c}' is not allowed outside strings`, i);
      let p = null;
      for (const q of PUNCTS) if (src.startsWith(q, i)) { p = q; break; }
      if (!p) fail(`unexpected character '${c}'`, i);
      if (p === '?.' && isDigit(src[i + 2] || '')) p = '?';
      if (p === '{') braces.push('{'); else if (p === '}') { if (braces.length) braces.pop(); }
      push({t: 'p', v: p, s, e: i + p.length}); i += p.length;
    }
    if (braces.includes('T')) fail('unterminated template literal', n);
    return toks;
  }
  // bracket matching over tokens: ( [ { and template ${ … } groups
  function matchBrackets(toks, src) {
    const m = new Int32Array(toks.length).fill(-1), st = [];
    const open = {')': '(', ']': '[', '}': '{'};
    for (let k = 0; k < toks.length; k++) {
      const tk = toks[k];
      if (tk.t === 'p' && (tk.v === '(' || tk.v === '[' || tk.v === '{')) st.push(k);
      else if (tk.t === 'p' && open[tk.v]) {
        const o = st.pop();
        if (o === undefined || toks[o].v !== open[tk.v]) throw new CodeError(`unbalanced '${tk.v}'`, lineOf(src, tk.s));
        m[o] = k; m[k] = o;
      } else if (tk.t === 'tmpl') {
        if (!tk.head) { const o = st.pop(); if (o === undefined || toks[o].t !== 'tmpl') throw new CodeError('unbalanced template', lineOf(src, tk.s)); m[o] = k; m[k] = o; }
        if (!tk.tail) st.push(k);
      }
    }
    if (st.length) throw new CodeError(`unclosed '${toks[st[st.length - 1]].v || '`'}'`, lineOf(src, toks[st[st.length - 1]].s));
    return m;
  }

  // ------------------------------------------------------------------ static check
  const BAN_ALWAYS = new Set(['this', 'import', 'export', 'async', 'await', 'yield', 'class', 'extends', 'super', 'with', 'debugger',
    'eval', 'arguments', 'Function', 'constructor', '__proto__', 'prototype', '__defineGetter__', '__defineSetter__', '__lookupGetter__',
    '__lookupSetter__', 'caller', 'callee', 'getPrototypeOf', 'setPrototypeOf', 'defineProperty', 'defineProperties',
    'getOwnPropertyDescriptor', 'getOwnPropertyDescriptors', 'prepareStackTrace', 'captureStackTrace', 'stackTraceLimit']);
  // bare names refused with a clear message (every other global is shadowed to undefined anyway)
  const BAN_GLOBAL = new Set(['globalThis', 'window', 'document', 'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'localStorage',
    'sessionStorage', 'indexedDB', 'setTimeout', 'setInterval', 'setImmediate', 'clearTimeout', 'clearInterval', 'requestAnimationFrame',
    'queueMicrotask', 'Promise', 'Proxy', 'Reflect', 'Symbol', 'WeakRef', 'FinalizationRegistry', 'Atomics', 'SharedArrayBuffer',
    'WebAssembly', 'Worker', 'SharedWorker', 'importScripts', 'postMessage', 'process', 'require', 'performance', 'Date']);
  const SHADOW_EXTRA = ['self', 'top', 'parent', 'frames', 'opener', 'navigator', 'location', 'history', 'caches', 'module', 'exports',
    'global', 'Buffer', 'alert', 'confirm', 'prompt', 'crypto', 'chrome', 'browser', 'Deno', 'Bun', '__dirname', '__filename'];
  const BAN_KEYS = new Set(['constructor', '__proto__', 'prototype', '__defineGetter__', '__defineSetter__', '__lookupGetter__',
    '__lookupSetter__', 'caller', 'callee', 'arguments', 'getPrototypeOf', 'setPrototypeOf', 'defineProperty', 'defineProperties',
    'getOwnPropertyDescriptor', 'getOwnPropertyDescriptors', 'prepareStackTrace', 'captureStackTrace', 'stackTraceLimit']);

  function staticCheck(toks, src) {
    const errs = [];
    const E = (msg, tk) => { if (errs.length < 12) errs.push({line: lineOf(src, tk ? tk.s : 0), msg}); };
    for (let k = 0; k < toks.length; k++) {
      const tk = toks[k], prev = toks[k - 1], next = toks[k + 1];
      if (tk.t === 'name') {
        const v = tk.v, isProp = prev && prev.t === 'p' && (prev.v === '.' || prev.v === '?.');
        const isKey = next && next.t === 'p' && next.v === ':' && prev && prev.t === 'p' && (prev.v === '{' || prev.v === ',');
        if (BAN_ALWAYS.has(v)) E(`'${v}' is not allowed in a program`, tk);
        else if (v.startsWith('__')) E(`names starting with '__' are not allowed ('${v}')`, tk);
        else if (BAN_GLOBAL.has(v) && !isProp && !isKey) E(`'${v}' is not available in a program (only api.* and plain JS)`, tk);
        else if (v === 'function' && next && next.t === 'p' && next.v === '*') E('generator functions are not allowed', tk);
        else if (v === 'new' && next && next.t === 'p' && next.v === '.') E("'new.target' is not allowed", tk);
      } else if (tk.t === 'str') {
        if (BAN_KEYS.has(tk.v)) E(`the string '${tk.v}' is not allowed`, tk);
      } else if (tk.t === 'p' && tk.v === '*' && prev && prev.t === 'p' && (prev.v === '{' || prev.v === ',') && next && next.t === 'name') {
        E('generator methods are not allowed', tk);
      } else if (tk.t === 'p' && (tk.v === '@' || tk.v === '#')) E(`'${tk.v}' is not allowed`, tk);
    }
    return errs;
  }

  // ------------------------------------------------------------------ instrumentation (guards)
  const ENDS_EXPR = tk => tk && ((tk.t === 'name' && !KW_NOT_VALUE.has(tk.v)) || tk.t === 'num' || tk.t === 'str' || tk.t === 'regex' ||
    (tk.t === 'tmpl' && tk.tail) || (tk.t === 'p' && (tk.v === ')' || tk.v === ']' || tk.v === '}' || tk.v === '++' || tk.v === '--')));
  const STARTS_STMT = tk => (tk.t === 'name' && tk.v !== 'in' && tk.v !== 'of' && tk.v !== 'instanceof') || tk.t === 'num' || tk.t === 'str' ||
    (tk.t === 'p' && (tk.v === '{' || tk.v === '++' || tk.v === '--' || tk.v === '!' || tk.v === '~'));
  function exprEnd(toks, i0) {   // index of the first token after an arrow's expression body
    let d = 0;
    for (let k = i0; k < toks.length; k++) {
      const tk = toks[k];
      if (d === 0 && k > i0) {
        if (tk.t === 'p' && (tk.v === ',' || tk.v === ';' || tk.v === ')' || tk.v === ']' || tk.v === '}')) return k;
        if (tk.t === 'tmpl' && !tk.head) return k;
        if (tk.nl && ENDS_EXPR(toks[k - 1]) && STARTS_STMT(tk)) return k;
      }
      if (tk.t === 'p' && (tk.v === '(' || tk.v === '[' || tk.v === '{')) d++;
      else if (tk.t === 'p' && (tk.v === ')' || tk.v === ']' || tk.v === '}')) { d--; if (d < 0) return k; }
      else if (tk.t === 'tmpl') { if (tk.head && !tk.tail) d++; else if (!tk.head && tk.tail) d--; }
    }
    return toks.length;
  }
  const NOT_FN_HEAD = new Set(['if', 'for', 'while', 'switch', 'catch', 'with']);
  function instrument(toks, src, level = 3) {   // level 3: loops + functions + arrow bodies, 2: no arrow expression bodies, 1: loops only
    const M = matchBrackets(toks, src);
    const ins = [];   // [pos, text, kind(0 insert), seq]
    const rep = [];   // [start, end, text]
    let seq = 0;
    const at = (pos, text) => ins.push([pos, text, seq++]);
    for (let k = 0; k < toks.length; k++) {
      const tk = toks[k], nx = toks[k + 1];
      if (tk.t === 'name' && (tk.v === 'while' || tk.v === 'for') && nx && nx.t === 'p' && nx.v === '(' && !(toks[k - 1] && toks[k - 1].t === 'p' && toks[k - 1].v === '.')) {
        const o = k + 1, c = M[o];
        if (tk.v === 'while') { at(toks[o].e, '__g()&&('); at(toks[c].s, ')'); continue; }
        const semis = []; let ofIn = -1, d = 0;
        for (let j = o + 1; j < c; j++) {
          const q = toks[j];
          if (q.t === 'p' && (q.v === '(' || q.v === '[' || q.v === '{')) d++;
          else if (q.t === 'p' && (q.v === ')' || q.v === ']' || q.v === '}')) d--;
          else if (q.t === 'tmpl') { if (q.head && !q.tail) d++; else if (!q.head && q.tail) d--; }
          else if (d === 0 && q.t === 'p' && q.v === ';') semis.push(j);
          else if (d === 0 && ofIn < 0 && q.t === 'name' && (q.v === 'of' || q.v === 'in')) ofIn = j;
        }
        if (semis.length >= 2) {
          const a = semis[0], b = semis[1];
          if (b === a + 1) at(toks[a].e, '__g()'); else { at(toks[a].e, '__g()&&('); at(toks[b].s, ')'); }
        } else if (ofIn > 0) {
          if (toks[ofIn].v === 'in') rep.push([toks[ofIn].s, toks[ofIn].e, 'of __gk(']); else at(toks[ofIn].e, ' __gi(');
          at(toks[c].s, ')');
        }
        continue;
      }
      if (level >= 2 && tk.t === 'p' && tk.v === '{' && k > 0) {
        const pv = toks[k - 1];
        let fn = false;
        if (pv.t === 'p' && pv.v === '=>') fn = true;
        else if (pv.t === 'p' && pv.v === ')' && M[k - 1] >= 0) {
          const before = toks[M[k - 1] - 1];
          if (before && ((before.t === 'name' && !NOT_FN_HEAD.has(before.v)) || (before.t === 'p' && before.v === ']'))) fn = true;
        }
        if (fn) at(tk.e, '__g();');
        continue;
      }
      if (level >= 3 && tk.t === 'p' && tk.v === '=>' && nx && !(nx.t === 'p' && nx.v === '{')) {
        const end = exprEnd(toks, k + 1);
        if (end > k + 1) { at(tk.e, '(__g(),'); at(toks[end - 1].e, ')'); }
        continue;
      }
      if (tk.t === 'p' && tk.v === '[' && M[k] > k) {
        const pv = toks[k - 1], c = M[k], after = toks[c + 1];
        const member = pv && ((pv.t === 'name' && !KW_NOT_VALUE.has(pv.v)) || pv.t === 'num' || pv.t === 'str' || pv.t === 'regex' ||
          (pv.t === 'tmpl' && pv.tail) || (pv.t === 'p' && (pv.v === ')' || pv.v === ']' || pv.v === '?.')));
        const key = pv && pv.t === 'p' && (pv.v === '{' || pv.v === ',') && after && after.t === 'p' && (after.v === ':' || after.v === '(');
        if (member || key) { at(tk.e, '__k(('); at(toks[c].s, '))'); }
      }
    }
    // apply: replacements and insertions, by position (insertions at the same position keep their order)
    const ops = ins.map(([pos, text, sq]) => [pos, pos, text, sq]).concat(rep.map(([s, e, text]) => [s, e, text, -1]));
    ops.sort((a, b) => a[0] - b[0] || a[3] - b[3]);
    let out = '', cur = 0;
    for (const [s, e, text] of ops) { out += src.slice(cur, s) + text; cur = Math.max(cur, e); }
    return out + src.slice(cur);
  }

  // ------------------------------------------------------------------ safe globals + compile
  const RESERVED = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'enum',
    'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'null', 'return', 'super',
    'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'implements',
    'interface', 'package', 'private', 'protected', 'public', 'await', 'eval', 'arguments', 'undefined', 'NaN', 'Infinity']);
  const SAFE_NAMES = ['Math', 'JSON', 'Number', 'String', 'Boolean', 'Array', 'Object', 'Map', 'Set', 'Error', 'isFinite', 'isNaN',
    'parseFloat', 'parseInt', 'console'];
  const GUARD_NAMES = ['__g', '__k', '__gi', '__gk'];
  let SHADOW = null;
  function shadowNames() {
    if (SHADOW) return SHADOW;
    const names = new Set(BAN_GLOBAL);
    let o = globalThis;
    for (let d = 0; o && d < 12; d++) { try { for (const k of Object.getOwnPropertyNames(o)) names.add(k); } catch (e) { /* exotic */ } o = Object.getPrototypeOf(o); }
    for (const k of SHADOW_EXTRA) names.add(k);
    const skip = new Set([...SAFE_NAMES, ...GUARD_NAMES]);
    SHADOW = [...names].filter(k => /^[A-Za-z_$][\w$]*$/.test(k) && !RESERVED.has(k) && !skip.has(k));
    return SHADOW;
  }
  function makeSafe(logFn) {
    const fz = Object.freeze;
    const SMath = Object.create(null); for (const k of Object.getOwnPropertyNames(Math)) SMath[k] = Math[k]; fz(SMath);
    const SJSON = fz({parse: (s, r) => JSON.parse(s, r), stringify: (v, r, sp) => JSON.stringify(v, r, sp)});
    const cap = n => { if (typeof n === 'number' && n > LIMITS.arrayMax) throw new RangeError('array too long for a program'); };
    const SNumber = function (x) { return Number(x); };
    for (const k of ['isFinite', 'isInteger', 'isNaN', 'isSafeInteger', 'parseFloat', 'parseInt', 'EPSILON', 'MAX_SAFE_INTEGER',
      'MIN_SAFE_INTEGER', 'MAX_VALUE', 'MIN_VALUE', 'POSITIVE_INFINITY', 'NEGATIVE_INFINITY', 'NaN']) SNumber[k] = Number[k];
    const SString = function (x) { return String(x); }; SString.fromCharCode = String.fromCharCode;
    const SBoolean = function (x) { return Boolean(x); };
    const SArray = function (...a) { if (a.length === 1) cap(a[0]); return Array(...a); };
    SArray.isArray = Array.isArray; SArray.of = (...a) => Array.of(...a);
    SArray.from = (x, f) => { if (x && typeof x === 'object') cap(+x.length || 0); return f ? Array.from(x, f) : Array.from(x); };
    const SObject = fz({keys: o => Object.keys(o), values: o => Object.values(o), entries: o => Object.entries(o),
      assign: (t, ...s) => Object.assign(t, ...s), freeze: o => Object.freeze(o), isFrozen: o => Object.isFrozen(o),
      fromEntries: e => Object.fromEntries(e), hasOwn: (o, k) => Object.prototype.hasOwnProperty.call(o, k)});
    const SMap = function (it) { return new Map(it); }, SSet = function (it) { return new Set(it); };
    const SError = function (m) { return new Error(String(m)); };
    const cons = fz({log: (...a) => logFn('log', a), info: (...a) => logFn('log', a), warn: (...a) => logFn('warn', a), error: (...a) => logFn('error', a)});
    for (const f of [SNumber, SString, SBoolean, SArray, SMap, SSet, SError]) fz(f);
    return {Math: SMath, JSON: SJSON, Number: SNumber, String: SString, Boolean: SBoolean, Array: SArray, Object: SObject, Map: SMap,
      Set: SSet, Error: SError, isFinite, isNaN, parseFloat, parseInt, console: cons};
  }
  function makeGuard() {
    const G = {n: 0, t0: 0, lim: BUDGET.hardMs, tripped: false};
    G.g = () => {
      if (G.tripped) throw new Error('program over its time budget (cut off)');
      if ((++G.n & 63) === 0 && now() - G.t0 > G.lim) { G.tripped = true; throw new Error('program over its time budget (cut off)'); }
      return true;
    };
    G.k = key => {
      if (typeof key === 'number') return key;
      if (typeof key === 'symbol') throw new TypeError('symbol keys are not allowed');
      const s = String(key);
      if (BAN_KEYS.has(s) || s.startsWith('__')) throw new TypeError(`property '${s}' is not allowed`);
      return s;
    };
    G.gi = function* (it) { for (const v of it) { G.g(); yield v; } };
    G.gk = function* (o) { for (const k in o) { G.g(); yield k; } };
    G.begin = lim => { G.n = 0; G.t0 = now(); G.lim = lim; G.tripped = false; };
    return G;
  }
  // the program text → its expression. A trailing ';' is dropped; a text with helper statements before the object
  // literal gets 'return' in front of its last top-level parenthesised expression.
  function bodyOf(src, toks) {
    const t = src.replace(/[\s;]+$/, '');
    return {expr: t, alt: (() => {
      // find a top-level '(' whose match is the last ')' and which starts a statement
      const M = matchBrackets(toks, src);
      let last = toks.length - 1; while (last >= 0 && toks[last].t === 'p' && toks[last].v === ';') last--;
      if (last < 0 || !(toks[last].t === 'p' && toks[last].v === ')')) return null;
      const o = M[last]; if (o < 0) return null;
      const pv = toks[o - 1];
      if (pv && !(pv.t === 'p' && (pv.v === ';' || pv.v === '}')) && !toks[o].nl) return null;
      return src.slice(0, toks[o].s) + 'return ' + src.slice(toks[o].s, toks[last].e);
    })()};
  }
  /** compile(code) → {ok, make(env) → program object, level, errors:[{line,msg}], error}. env = {log(kind, args)}. */
  function compile(code) {
    code = String(code == null ? '' : code);
    const fail = (errors) => ({ok: false, errors, error: errors.map(e => (e.line ? `line ${e.line}: ` : '') + e.msg).join('; ')});
    if (!code.trim()) return fail([{line: 0, msg: 'empty program'}]);
    if (code.length > LIMITS.codeChars) return fail([{line: 0, msg: `program too long (${code.length} > ${LIMITS.codeChars} characters)`}]);
    if (code.split('\n').length > LIMITS.lines) return fail([{line: 0, msg: `program too long (> ${LIMITS.lines} lines)`}]);
    let toks;
    try { toks = tokenize(code); matchBrackets(toks, code); } catch (e) { return fail([{line: e.line || 0, msg: e.message}]); }
    const errs = staticCheck(toks, code);
    if (errs.length) return fail(errs);
    const shadows = shadowNames();
    const params = shadows.concat(SAFE_NAMES, GUARD_NAMES);
    const B = bodyOf(code, toks);
    let factory = null, level = 0, lastErr = null;
    // does the raw text parse at all? (guarded versions that fail where this succeeds drop a guard level)
    const forms = [`"use strict";\nreturn (\n${B.expr}\n);`].concat(B.alt ? [`"use strict";\n${B.alt}\n;`] : []);
    let form = -1;
    for (let f = 0; f < forms.length && form < 0; f++) { try { new Function(...params, forms[f]); form = f; } catch (e) { lastErr = e; } }
    if (form < 0) return fail([{line: 0, msg: 'syntax error: ' + String(lastErr && lastErr.message || lastErr).slice(0, 200)}]);
    for (const lv of [3, 2, 1]) {
      try {
        const src2 = instrument(toks, code, lv), t2 = tokenize(src2), B2 = bodyOf(src2, t2);
        const body = form === 0 ? `"use strict";\nreturn (\n${B2.expr}\n);` : `"use strict";\n${B2.alt}\n;`;
        factory = new Function(...params, body); level = lv; break;
      } catch (e) { lastErr = e; }
    }
    if (!factory) return fail([{line: 0, msg: 'could not guard this program: ' + String(lastErr && lastErr.message || lastErr).slice(0, 200)}]);
    const nShadow = shadows.length;
    function make(G, safe) {
      const args = new Array(nShadow);   // every shadowed global = undefined
      for (const k of SAFE_NAMES) args.push(safe[k]);
      args.push(G.g, G.k, G.gi, G.gk);
      G.begin(BUDGET.setupMs);
      const obj = factory(...args);
      if (!obj || typeof obj !== 'object') throw new CodeError('the program must be an object literal ({ setup(api){}, per_frame(api){}, end(api){} })');
      if (typeof obj.per_frame !== 'function') throw new CodeError('the program has no per_frame(api) function');
      for (const k of ['setup', 'end']) if (obj[k] !== undefined && typeof obj[k] !== 'function') throw new CodeError(`${k} must be a function`);
      return obj;
    }
    return {ok: true, make, level, errors: []};
  }

  // ------------------------------------------------------------------ perception → the sense snapshot + derived events
  const BS = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'];
  const nk = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  const BS_KEY = {}; for (const b of BS) BS_KEY[nk(b)] = b;
  const avg = (a, b) => ((+a || 0) + (+b || 0)) / 2;
  // friendly face names → f(face) (face = perception's face object); blend names work too (any case / underscores)
  const FACE_NAMES = {
    smile: F => Math.max(+F.expr.smile || 0, F.xs.happy || 0), happy: F => Math.max(F.xs.happy || 0, +F.expr.smile || 0),
    frown: F => Math.max(F.xs.frown || 0, +F.expr.browDown || 0, avg(F.b.mouthFrownLeft, F.b.mouthFrownRight)),
    sad: F => Math.max(F.xs.sad || 0, +F.expr.sad || 0), surprise: F => Math.max(F.xs.surprised || 0, Math.min(1, ((+F.expr.eyeWide || 0) + (+F.expr.browUp || 0) + (+F.expr.jawOpen || 0)) / 2)),
    surprised: F => FACE_NAMES.surprise(F), angry: F => Math.max(F.xs.frown || 0, +F.expr.browDown || 0),
    mouthopen: F => +F.expr.jawOpen || +F.b.jawOpen || 0, jawopen: F => +F.expr.jawOpen || +F.b.jawOpen || 0, open: F => +F.expr.jawOpen || 0,
    browup: F => Math.max(+F.expr.browUp || 0, F.xs.raised_brows || 0), browsup: F => FACE_NAMES.browup(F), raisedbrows: F => FACE_NAMES.browup(F),
    eyebrowsup: F => FACE_NAMES.browup(F), browdown: F => +F.expr.browDown || 0, browsdown: F => +F.expr.browDown || 0,
    squint: F => +F.expr.squint || 0, eyewide: F => +F.expr.eyeWide || 0, eyeswide: F => +F.expr.eyeWide || 0,
    blink: F => +F.expr.blink || avg(F.b.eyeBlinkLeft, F.b.eyeBlinkRight), eyesclosed: F => Math.min(+F.b.eyeBlinkLeft || +F.expr.blinkL || 0, +F.b.eyeBlinkRight || +F.expr.blinkR || 0),
    blinkleft: F => +F.b.eyeBlinkLeft || +F.expr.blinkL || 0, blinkright: F => +F.b.eyeBlinkRight || +F.expr.blinkR || 0,
    lefteyeclosed: F => +F.b.eyeBlinkLeft || +F.expr.blinkL || 0, righteyeclosed: F => +F.b.eyeBlinkRight || +F.expr.blinkR || 0,
    wink: F => Math.abs((+F.b.eyeBlinkLeft || 0) - (+F.b.eyeBlinkRight || 0)),
    pucker: F => +F.expr.pucker || +F.b.mouthPucker || 0, kiss: F => +F.expr.pucker || +F.b.mouthPucker || 0,
    cheekpuff: F => +F.expr.cheekPuff || +F.b.cheekPuff || 0, puff: F => +F.expr.cheekPuff || +F.b.cheekPuff || 0,
    sneer: F => avg(F.b.noseSneerLeft, F.b.noseSneerRight), tongue: () => 0, tongueout: () => 0,
  };
  // MediaPipe face-mesh points per part (anatomical: 33/133/159/145 = his right eye, 263/362/386/374 = his left)
  const MESH = {nose: [1, 4, 5], mouth: [13, 14, 0, 17], chin: [152, 175], forehead: [10, 151, 9], right_eye: [33, 133, 159, 145],
    left_eye: [263, 362, 386, 374], right_cheek: [50, 205, 187], left_cheek: [280, 425, 411], right_ear: [234, 127, 93], left_ear: [454, 356, 323],
    top: [10]};
  // face-box fallback (no mesh): [x offset in face widths (+ = HIS right = screen-right), y fraction from top of head to chin]
  const BOX = {nose: [[0, 0.62]], mouth: [[0, 0.8]], chin: [[0, 0.98]], forehead: [[0, 0.2], [0.15, 0.22], [-0.15, 0.22]], right_eye: [[0.2, 0.44]],
    left_eye: [[-0.2, 0.44]], right_cheek: [[0.3, 0.66]], left_cheek: [[-0.3, 0.66]], right_ear: [[0.52, 0.5]], left_ear: [[-0.52, 0.5]], top: [[0, 0.02]]};
  const PART_ALIAS = {nose: ['nose'], mouth: ['mouth'], lips: ['mouth'], lip: ['mouth'], chin: ['chin'], jaw: ['chin'], forehead: ['forehead'],
    head: ['forehead', 'top'], hair: ['top', 'forehead'], eye: ['right_eye', 'left_eye'], eyes: ['right_eye', 'left_eye'],
    right_eye: ['right_eye'], left_eye: ['left_eye'], cheek: ['right_cheek', 'left_cheek'], cheeks: ['right_cheek', 'left_cheek'],
    right_cheek: ['right_cheek'], left_cheek: ['left_cheek'], ear: ['right_ear', 'left_ear'], ears: ['right_ear', 'left_ear'],
    right_ear: ['right_ear'], left_ear: ['left_ear'], face: Object.keys(MESH)};
  const TIPS = [4, 8, 12, 16, 20];
  const MESH_PARTS = Object.keys(MESH);
  const GEST_NAME = {openpalm: 'open_palm', closedfist: 'fist', pointingup: 'pointing_up', thumbup: 'thumbs_up', thumbdown: 'thumbs_down',
    victory: 'victory', iloveyou: 'i_love_you', none: 'none', '': 'none'};
  const TOUCH_ON = 0.3, TOUCH_OFF = 0.45, TOUCH_PALM = 0.42;
  const NOT_COMMANDED = new Set(['auto', 'fidget', 'life', 'listen', 'reflex', 'cospeech']);   // motion.js event sources that are not commands

  function createPerceiver() {
    const P = {seq: 0, ring: [], clock: 0, lastF: null, lastFt: -1, lastEvArr: null, face: null, faceT: -9, geo: null, geoT: -9,
      present: false, presT: 0, absT: 9, hands: {left: null, right: null}, handSt: {left: {}, right: {}}, touch: {}, pose: null,
      eye: {L: {base: 0.1, closed: false, t: -9}, R: {base: 0.1, closed: false, t: -9}, both: false, bothT: -9, longSent: false, wink: null},
      st: {}, zBase: null, shrugBase: null, gest: {}, said: [], U: null, uv: false, av: false, seenAct: new WeakSet(), actInit: false};
    const FZ = Object.freeze;
    function push(type, data) {
      const ev = FZ(Object.assign({type, t: +P.clock.toFixed(3), seq: ++P.seq}, data || {}));
      P.ring.push(ev); if (P.ring.length > LIMITS.eventRing) P.ring.splice(0, P.ring.length - LIMITS.eventRing);
      return ev;
    }
    // hysteresis helper: returns 'on' | 'off' | null on a transition
    function edge(key, val, on, off, holdOn = 0, dt = 0) {
      const s = P.st[key] || (P.st[key] = {on: false, t: 0});
      if (!s.on) { if (val > on) { s.t += dt; if (s.t >= holdOn) { s.on = true; s.t = 0; return 'on'; } } else s.t = 0; }
      else if (val < off) { s.on = false; s.t = 0; return 'off'; }
      return null;
    }
    function faceGeo(face) {   // part positions (x, y in perception units) + scale (face width in x units, height in y units)
      const m = face.mesh, out = {};
      if (m && m.length >= 956) {
        const X = i => (m[2 * i] - 0.5) * 2, Y = i => (0.5 - m[2 * i + 1]) * 2;
        const fw = Math.max(1e-3, Math.abs(X(454) - X(234))), fh = Math.max(1e-3, Math.abs(Y(10) - Y(152)) * 1.15);
        for (const [k, ids] of Object.entries(MESH)) out[k] = ids.map(i => [X(i), Y(i)]);
        if (out.top) out.top = [[X(10), Y(10) + 0.12 * fh]];
        return {pts: out, fw, fh, cx: (X(454) + X(234)) / 2, cy: (Y(10) + Y(152)) / 2};
      }
      const fx = +face.x || 0, top = Number.isFinite(+face.topY) ? +face.topY : (+face.y || 0) + 0.3, chin = Number.isFinite(+face.chinY) ? +face.chinY : (+face.y || 0) - 0.3;
      const fw = Math.max(0.02, 2 * (+face.size || 0.15)), fh = Math.max(0.02, top - chin), yaw = +face.yaw || 0;
      for (const [k, list] of Object.entries(BOX)) out[k] = list.map(([dx, fy]) => [fx + dx * fw + 0.35 * fw * Math.sin(yaw) * (1 - Math.abs(dx) * 1.5), top - fy * fh]);
      return {pts: out, fw, fh, cx: fx, cy: (top + chin) / 2};
    }
    function handPts(h) {
      if (h.lm && h.lm.length >= 21) return {pts: TIPS.map(i => [(h.lm[i][0] - 0.5) * 2, (0.5 - h.lm[i][1]) * 2]), palm: false};
      return {pts: [[+h.x || 0, +h.y || 0]], palm: true};
    }
    function touchDist(hp, geo, part) {
      let best = Infinity;
      const Q = geo.pts[part]; if (!Q) return best;
      const ifw = 1 / geo.fw, ifh = 1 / geo.fh;
      for (let i = 0; i < Q.length; i++) for (let j = 0; j < hp.pts.length; j++) {
        const dx = (hp.pts[j][0] - Q[i][0]) * ifw, dy = (hp.pts[j][1] - Q[i][1]) * ifh, d = dx * dx + dy * dy;
        if (d < best) best = d;
      }
      return Math.sqrt(best);
    }
    function handSnap(h, side, geo, face, dt) {
      if (!h) return FZ({present: false, x: 0, y: 0, z: 0, raised: false, open: 0, pinch: 0, fingers: 0, gesture: 'none', pointing: false, touching: null});
      let pinch = 0, fingers = 0;
      if (h.lm && h.lm.length >= 21) {
        const L = h.lm, d = (a, b) => Math.hypot(L[a][0] - L[b][0], L[a][1] - L[b][1]);
        const palm = Math.max(1e-3, d(0, 9));
        pinch = clamp((0.6 - d(4, 8) / palm) / 0.45, 0, 1);
        fingers = (h.ext || []).filter(e => e > 0.5).length + (d(4, 5) / palm > 0.55 && d(4, 17) / palm > 0.9 ? 1 : 0);
      } else fingers = (h.ext || []).filter(e => e > 0.5).length;
      const chin = face ? (Number.isFinite(+face.chinY) ? +face.chinY : (+face.y || 0) - 0.3) : 0;
      const g = GEST_NAME[nk(h.gesture)] || nk(h.gesture) || 'none';
      const S = P.touch[side] || (P.touch[side] = {part: null});
      return FZ({present: true, x: +h.x || 0, y: +h.y || 0, z: clamp((+h.scale || 0) / 0.12, 0, 3), raised: (+h.y || 0) > chin,
        open: clamp(+h.open || 0, 0, 1), pinch, fingers, gesture: g, pointing: !!h.point,
        point: h.point ? FZ({x: +h.point.dx || 0, y: +h.point.dy || 0}) : null, touching: S.part || null});
    }
    function pickHands(list) {
      const out = {left: null, right: null};
      const rest = [];
      for (const h of list || []) { if ((h.handed === 'left' || h.handed === 'right') && !out[h.handed]) out[h.handed] = h; else rest.push(h); }
      for (const h of rest) { const s = h.side === 'l' ? 'left' : 'right'; if (!out[s]) out[s] = h; else if (!out[s === 'left' ? 'right' : 'left']) out[s === 'left' ? 'right' : 'left'] = h; }
      return out;
    }
    function onFeatures(f, dt) {
      const face = f.face && f.face.present ? f.face : null;
      if (face) { P.face = face; P.faceT = P.clock; P.geo = null; P.geoFace = face; P.geoT = P.clock; }
      // presence (0.1 s on / 0.5 s off)
      if (face) { P.presT += dt; P.absT = 0; if (!P.present && P.presT >= 0.1) { P.present = true; push('face_appear'); } }
      else { P.absT += dt; P.presT = 0; if (P.present && P.absT >= 0.5) { P.present = false; push('face_lost'); } }
      const F = face ? {expr: face.expr || {}, b: face.blend || {}, xs: (face.xp && face.xp.scores) || {}} : null;
      if (F) {
        // blinks / winks / long closures (per eye: adaptive open level)
        const E = P.eye, vL = +F.b.eyeBlinkLeft || +F.expr.blinkL || 0, vR = +F.b.eyeBlinkRight || +F.expr.blinkR || 0;
        for (const [k, v] of [['L', vL], ['R', vR]]) {
          const e = E[k];
          e.base = v < e.base ? lp(e.base, v, dt, 0.4) : lp(e.base, v, dt, 6);
          const on = clamp(e.base + 0.3, 0.4, 0.72), off = on - 0.15;
          if (!e.closed && v > on) { e.closed = true; e.t = P.clock; } else if (e.closed && v < off) e.closed = false;
        }
        const both = E.L.closed && E.R.closed;
        if (both && !E.both) { E.both = true; E.bothT = P.clock; E.longSent = false; E.wink = null; push('blink'); }
        else if (!both && E.both) { E.both = false; if (E.longSent) push('eyes_open'); }
        if (E.both && !E.longSent && P.clock - E.bothT > 0.8) { E.longSent = true; push('eyes_closed'); }
        const one = E.L.closed !== E.R.closed ? (E.L.closed ? 'left' : 'right') : null;
        if (one && !E.both) { if (!E.wink || E.wink.eye !== one) E.wink = {eye: one, t: P.clock, sent: false}; else if (!E.wink.sent && P.clock - E.wink.t >= 0.12) { E.wink.sent = true; push('wink', {eye: one}); } }
        else if (!one) E.wink = null;
        const sm = FACE_NAMES.smile(F);
        const es = edge('smile', sm, 0.5, 0.3, 0.12, dt); if (es === 'on') push('smile_start'); else if (es === 'off') push('smile_end');
        const jo = +F.expr.jawOpen || +F.b.jawOpen || 0;
        const em = edge('mouth', jo, 0.42, 0.22, 0.15, dt); if (em === 'on') push('mouth_open'); else if (em === 'off') push('mouth_close');
        const eb = edge('brows', FACE_NAMES.browup(F), 0.55, 0.35, 0.1, dt); if (eb === 'on') push('brows_up'); else if (eb === 'off') push('brows_down');
        const ef = edge('frown', FACE_NAMES.frown(F), 0.5, 0.3, 0.15, dt); if (ef === 'on') push('frown');
        const roll = +face.roll || 0, yaw = +face.yaw || 0;
        const et = edge('tilt', Math.abs(roll), 0.18, 0.08, 0.12, dt); if (et === 'on') push('head_tilt', {dir: roll > 0 ? 'right' : 'left', roll}); else if (et === 'off') push('head_straight');
        const ey = edge('turn', Math.abs(yaw), 0.35, 0.18, 0.12, dt); if (ey === 'on') push('head_turn', {dir: yaw > 0 ? 'right' : 'left', yaw});
        // looking away (present but not attending) / back
        const away = f.attending === false || (f.attending === undefined && f.attention !== undefined && f.attention < 0.3);
        const ea = edge('away', away ? 1 : 0, 0.5, 0.5, 0.35, dt);
        if (ea === 'on') push('look_away', {dir: (+face.lookX || 0) >= 0 ? 'right' : 'left', yaw: +face.lookX || 0, pitch: +face.lookY || 0});
        else if (ea === 'off') push('look_back');
        // distance: lean in / back relative to his usual distance
        const z = +face.z || 0;
        if (z > 0) {
          P.zBase = P.zBase == null ? z : lp(P.zBase, z, dt, 20);
          const r = z / Math.max(0.05, P.zBase);
          const el = edge('near', 1 / r, 1.22, 1.08, 0.2, dt); if (el === 'on') push('lean_in');
          const eb2 = edge('far', r, 1.25, 1.1, 0.2, dt); if (eb2 === 'on') push('lean_back');
        }
      } else {
        // no face: the eye trackers relax
        P.eye.L.closed = P.eye.R.closed = false; P.eye.both = false;
      }
      // hands: snapshot, raised, gestures, touches, hands together
      // a hand over the face often hides it: its geometry is kept 1 s (computed only when a hand is in view)
      const anyHand = f.hands && f.hands.length > 0;
      if (anyHand && !P.geo && P.geoFace) P.geo = faceGeo(P.geoFace);
      const geo = P.geo && P.clock - P.geoT < 1.0 ? P.geo : null;
      const H = pickHands(f.hands);
      for (const side of ['left', 'right']) {
        const h = H[side], hs = P.handSt[side];
        if (h) { hs.lastT = P.clock; hs.h = h; } else if (hs.h && P.clock - (hs.lastT || -9) > 0.35) hs.h = null;
        const hh = hs.h;
        if (!!hh !== !!hs.was) { push(hh ? 'hand_appear' : 'hand_gone', {hand: side}); hs.was = !!hh; }
        // touch
        const T2 = P.touch[side] || (P.touch[side] = {part: null});
        const nearFace = hh && h && geo && Math.abs((+hh.x || 0) - geo.cx) < 1.6 * geo.fw && Math.abs((+hh.y || 0) - geo.cy) < 1.4 * geo.fh;
        if (nearFace) {
          const hp = handPts(hh), on = hp.palm ? TOUCH_PALM : TOUCH_ON, off = hp.palm ? TOUCH_PALM + 0.15 : TOUCH_OFF;
          let best = null, bd = Infinity;
          T2.near = {};
          for (const part of MESH_PARTS) { const d = touchDist(hp, geo, part); T2.near[part] = d; if (part !== 'top' && d < bd) { bd = d; best = part; } }
          T2.d = bd;
          if (T2.part) { if ((T2.near[T2.part] ?? Infinity) > off) { push('touch_end', {part: T2.part, hand: side}); T2.part = null; } }
          if (!T2.part && bd < on) { T2.part = best; push('touch', {part: best, hand: side}); }
          else if (T2.part && best !== T2.part && bd < on && (T2.near[T2.part] ?? Infinity) > bd + 0.12) { push('touch_end', {part: T2.part, hand: side}); T2.part = best; push('touch', {part: best, hand: side}); }
        } else if (T2.part && (!hh || (h && geo))) { push('touch_end', {part: T2.part, hand: side}); T2.part = null; T2.near = null; }
        P.hands[side] = handSnap(hh, side, geo, P.face && P.clock - P.faceT < 1 ? P.face : null, dt);
        const snap = P.hands[side];
        const er = edge('raise_' + side, snap.present && snap.raised ? 1 : 0, 0.5, 0.5, 0.08, dt);
        if (er === 'on') push('hand_up', {hand: side}); else if (er === 'off') push('hand_down', {hand: side});
        const gname = snap.present ? snap.gesture : 'none';
        if (gname !== (hs.gest || 'none')) { hs.gestN = (hs.gestC === gname ? (hs.gestN || 0) + 1 : 1); hs.gestC = gname; if (hs.gestN >= 2) { hs.gest = gname; if (gname !== 'none') push('hand_gesture', {name: gname, hand: side}); } }
        else { hs.gestN = 0; hs.gestC = gname; }
      }
      const L = P.hands.left, Rh = P.hands.right;
      const tog = L.present && Rh.present ? Math.hypot(L.x - Rh.x, L.y - Rh.y) / Math.max(0.05, geo ? geo.fw : 0.3) : 9;
      const eh = edge('together', 1 / Math.max(1e-3, tog), 1 / 0.55, 1 / 0.85, 0, dt); if (eh === 'on') push('hands_together'); else if (eh === 'off') push('hands_apart');
      // pose: lean + shrug
      const po = f.pose && f.pose.present ? f.pose : null;
      if (po && po.shoulders && po.shoulders.l && po.shoulders.r) {
        const chin = P.face && Number.isFinite(+P.face.chinY) ? +P.face.chinY : null;
        const sy = (+po.shoulders.l.y + +po.shoulders.r.y) / 2;
        let shrug = 0;
        if (chin !== null) {
          const gap = chin - sy;
          P.shrugBase = P.shrugBase == null ? gap : (gap > P.shrugBase * 0.85 ? lp(P.shrugBase, gap, dt, 8) : P.shrugBase);
          shrug = clamp((P.shrugBase - gap) / Math.max(0.02, 0.3 * P.shrugBase), 0, 1);
        }
        const es2 = edge('shrug', shrug, 0.55, 0.3, 0.1, dt); if (es2 === 'on') push('shrug');
        const hl = po.handed && po.handed.l === 'right' ? 'right' : 'left';
        const sh = {[hl]: FZ({x: +po.shoulders.l.x || 0, y: +po.shoulders.l.y || 0}), [hl === 'left' ? 'right' : 'left']: FZ({x: +po.shoulders.r.x || 0, y: +po.shoulders.r.y || 0})};
        P.pose = FZ({present: true, lean: +po.lean || 0, shrug, shoulders: FZ(sh)});
      } else if (!po) P.pose = null;
      // perception's own discrete events (nod, head_shake, wave, point, thumbs_up, object_up, approach_fast, expression_change)
      if (f.events && f.events !== P.lastEvArr) {
        P.lastEvArr = f.events;
        for (const e of f.events) {
          if (!e || typeof e.type !== 'string') continue;
          if (/^(face_appear|face_lost|attention_lost|look_back|model_)/.test(e.type)) continue;   // derived here
          if (e.type === 'gesture' || e.type === 'said' || e.type === 'attention' || e.type === 'turn_bid') { feed(e); continue; }   // (a host may carry these in f.events)
          const d = {}; for (const k of ['side', 'handed', 'amp', 'from', 'to', 'intensity', 'source', 'rate', 'x', 'y']) if (e[k] !== undefined && typeof e[k] !== 'object') d[k] = e[k];
          if (d.handed) d.hand = d.handed;
          push(e.type, d);
        }
      }
    }
    function voice(ctx) {
      const v = ctx.voice || {}, us = !!(v.user && v.user.speaking), as = !!(v.avatar && v.avatar.speaking);
      if (us !== P.uv) { P.uv = us; push(us ? 'speech_start' : 'speech_end'); }
      if (as !== P.av) { P.av = as; push(as ? 'self_speech_start' : 'self_speech_end'); }
    }
    function actions(ctx) {   // the motion's own action events (motion.js S.actions): commanded starts / ends
      const A = ctx.actions; if (!A || !A.length) return;
      const fresh = [];
      for (let i = A.length - 1; i >= 0; i--) { const ev = A[i]; if (!ev || typeof ev !== 'object' || P.seenAct.has(ev)) break; P.seenAct.add(ev); fresh.push(ev); }
      if (!P.actInit) { P.actInit = true; return; }   // what happened before the first program frame is history
      for (let i = fresh.length - 1; i >= 0; i--) {
        const ev = fresh[i];
        if (ev.kind === 'reflex' || ev.kind === 'config' || NOT_COMMANDED.has(ev.src)) continue;
        if (ev.phase === 'start' || ev.phase === 'end' || ev.phase === 'interrupted' || ev.phase === 'fail')
          push(ev.phase === 'start' ? 'action_start' : 'action_end', {name: String(ev.name || ev.kind || ''), kind: String(ev.kind || ''), phase: ev.phase,
            description: String(ev.did || ev.description || ev.name || '').slice(0, 120)});
      }
    }
    function snapshot(ctx) {
      const face = P.present && P.face ? P.face : null;
      const v = (ctx.voice && ctx.voice.user) || {};
      const F = face ? {expr: face.expr || {}, b: face.blend || {}, xs: (face.xp && face.xp.scores) || {}} : null;
      const faceFn = name => {
        if (!F) return 0;
        const k = nk(name), f = FACE_NAMES[k];
        if (f) return clamp(+f(F) || 0, 0, 1);
        if (BS_KEY[k]) return clamp(+F.b[BS_KEY[k]] || 0, 0, 1);
        const ek = Object.keys(F.expr).find(x => nk(x) === k);
        if (ek) return clamp(+F.expr[ek] || 0, 0, 1);
        P.unknownFace = String(name).slice(0, 40);
        return 0;
      };
      const touching = part => {
        const parts = PART_ALIAS[nk(part).replace(/(left|right)(eye|ear|cheek)/, '$1_$2')] || PART_ALIAS[String(part || '').toLowerCase()] || null;
        if (!parts) { P.unknownPart = String(part).slice(0, 40); return false; }
        for (const side of ['left', 'right']) { const T2 = P.touch[side]; if (T2 && T2.part && parts.includes(T2.part)) return true; }
        return false;
      };
      const touchWith = part => {
        const parts = PART_ALIAS[nk(part).replace(/(left|right)(eye|ear|cheek)/, '$1_$2')] || [];
        for (const side of ['left', 'right']) { const T2 = P.touch[side]; if (T2 && T2.part && parts.includes(T2.part)) return side; }
        return null;
      };
      P.U = FZ({
        present: !!face,
        face: faceFn,
        touching, touching_hand: touchWith,
        head: FZ(face ? {yaw: +face.yaw || 0, pitch: -(+face.pitch || 0), roll: +face.roll || 0} : {yaw: 0, pitch: 0, roll: 0}),
        gaze: FZ(face ? {atCamera: !!(P.st.away ? !P.st.away.on : true), yaw: +face.lookX || 0, pitch: +face.lookY || 0, attention: clamp(+(ctx.features && ctx.features.attention) || 0, 0, 1)}
          : {atCamera: false, yaw: 0, pitch: 0, attention: 0}),
        position: FZ(face ? {x: +face.x || 0, y: +face.y || 0} : {x: 0, y: 0}),
        distance: face ? +face.z || 0 : 0,
        hands: FZ({left: P.hands.left || handSnap(null), right: P.hands.right || handSnap(null)}),
        pose: P.pose || FZ({present: false, lean: 0, shrug: 0, shoulders: null}),
        speaking: !!v.speaking, level: clamp(+v.level || 0, 0, 1),
        expression: face && face.xp && face.xp.state ? String(face.xp.state) : 'neutral',
        moving: clamp(+(ctx.features && ctx.features.motion) || 0, 0, 1),
      });
      return P.U;
    }
    function update(ctx, clock, dt) {
      P.clock = clock;
      const f = ctx.features, seq0 = P.seq;
      let changed = false;
      if (f && (f !== P.lastF || f.t !== P.lastFt)) {
        const fdt = P.lastFt >= 0 && Number.isFinite(+f.t) && +f.t > P.lastFt ? Math.min(0.5, +f.t - P.lastFt) : dt;
        P.lastF = f; P.lastFt = +f.t;
        onFeatures(f, fdt); changed = true;
      } else if (!f && P.present) { P.absT += dt; if (P.absT >= 0.5) { P.present = false; push('face_lost'); } }
      voice(ctx);
      actions(ctx);
      const v = ctx.voice && ctx.voice.user, lvl = v ? +v.level || 0 : 0;
      if (changed || P.seq !== seq0 || !P.U || Math.abs(lvl - (P.lastLvl || 0)) > 0.02 || ctx.features !== P.snapF) { P.lastLvl = lvl; P.snapF = ctx.features; return snapshot(ctx); }
      return P.U;
    }
    function feed(ev) {
      if (!ev || typeof ev.type !== 'string') return null;
      const type = ev.type.replace(/[^a-z0-9_]/gi, '_').slice(0, 40);
      const d = {};
      for (const [k, v] of Object.entries(ev)) {
        if (k === 'type' || k === 't' || k === 'seq') continue;
        if (v === null || typeof v === 'number' || typeof v === 'boolean') d[k] = v;
        else if (typeof v === 'string') d[k] = v.slice(0, 300);
      }
      if (type === 'gesture' && d.name) P.gest[String(d.name).toLowerCase()] = {score: +d.score || 1, t: P.clock};
      if (type === 'said' && d.text) { P.said.push({t: P.clock, text: String(d.text).toLowerCase()}); if (P.said.length > 50) P.said.shift(); }
      return push(type, d);
    }
    function gesture(name) {
      const k = String(name || '').toLowerCase().trim().replace(/[\s-]+/g, '_');
      const G = P.gest[k];
      if (G) return FZ({on: P.clock - G.t < 0.6, score: P.clock - G.t < 0.6 ? clamp(G.score, 0, 1) : 0, known: true});
      // built-in cues
      const recent = (types, s) => P.ring.some(e => types.includes(e.type) && P.clock - e.t <= s);
      const H = [P.hands.left, P.hands.right].filter(h => h && h.present);
      const hg = g => H.some(h => h.gesture === g);
      const B = {wave: () => recent(['wave'], 1.2), nod: () => recent(['nod'], 1.0), head_shake: () => recent(['head_shake'], 1.0), shake: () => recent(['head_shake'], 1.0),
        thumbs_up: () => hg('thumbs_up') || recent(['thumbs_up'], 0.8), thumbs_down: () => hg('thumbs_down'), fist: () => hg('fist'),
        open_palm: () => hg('open_palm'), victory: () => hg('victory'), peace: () => hg('victory'), i_love_you: () => hg('i_love_you'),
        point: () => H.some(h => h.pointing) || recent(['point'], 0.8), pointing: () => H.some(h => h.pointing),
        hands_together: () => !!(P.st.together && P.st.together.on), clap: () => recent(['hands_together'], 0.6), shrug: () => !!(P.st.shrug && P.st.shrug.on),
        smile: () => !!(P.st.smile && P.st.smile.on), blink: () => recent(['blink'], 0.4), wink: () => recent(['wink'], 0.5),
        hand_up: () => H.some(h => h.raised), raise_hand: () => H.some(h => h.raised), head_tilt: () => !!(P.st.tilt && P.st.tilt.on),
        look_away: () => !!(P.st.away && P.st.away.on)};
      const tm = /^touch(?:ing)?_(.+)$/.exec(k);
      if (tm) { const on = P.U ? P.U.touching(tm[1]) : false; return FZ({on, score: on ? 1 : 0, known: true}); }
      if (B[k]) { const on = !!B[k](); return FZ({on, score: on ? 1 : 0, known: true}); }
      return FZ({on: false, score: 0, known: false});
    }
    function heardSince(seq0, words) {
      const ws = [].concat(words || []).map(w => String(w || '').toLowerCase().trim()).filter(Boolean);
      if (!ws.length) return 0;
      let n = 0;
      for (const e of P.ring) {
        if (e.seq <= seq0 || e.type !== 'said' || !e.text) continue;
        const txt = ' ' + String(e.text).toLowerCase().replace(/[^a-z0-9' ]+/g, ' ') + ' ';
        for (const w of ws) { const re = new RegExp('(?:^|\\s)' + w.replace(/[^a-z0-9' ]/g, '').replace(/\s+/g, '\\s+') + '(?=\\s)', 'g'); n += (txt.match(re) || []).length; }
      }
      return n;
    }
    snapshot({});
    return {P, update, feed, push, gesture, heardSince, snapshot: () => P.U};
  }

  // ------------------------------------------------------------------ look targets (screen frame, radians: + = right / up)
  const TAN_H = Math.tan(35 * Math.PI / 180), TAN_V = Math.tan(27.5 * Math.PI / 180);
  const GAZE_RX = Math.atan(0.30), GAZE_RY = Math.atan(0.24);
  function lookAngles(target, U, rec) {
    if (Array.isArray(target)) return [clamp(+target[0] || 0, -1, 1), clamp(+target[1] || 0, -1, 1)].map((v, i) => Math.atan(v * (i ? TAN_V : TAN_H)));
    if (target && typeof target === 'object') {
      if (Number.isFinite(+target.yaw) || Number.isFinite(+target.pitch)) return [clamp(+target.yaw || 0, -0.8, 0.8), clamp(+target.pitch || 0, -0.6, 0.6)];
      return [Math.atan(clamp(+target.x || 0, -1.2, 1.2) * TAN_H), Math.atan(clamp(+target.y || 0, -1.2, 1.2) * TAN_V)];
    }
    const k = String(target || 'user').toLowerCase().replace(/[\s-]+/g, '_').replace(/^(his|my|the)_/, '');
    const pos = (x, y) => [Math.atan(clamp(x, -1.2, 1.2) * TAN_H), Math.atan(clamp(y, -1.2, 1.2) * TAN_V)];
    switch (k) {
      case 'user': case 'him': case 'me': case 'camera': case 'face': case 'you': case 'eyes':
        return U.present ? pos(U.position.x, U.position.y) : [0, 0];
      case 'gaze': case 'his_gaze': case 'where_he_looks': case 'where_i_look': case 'look': case 'same_way': case 'where_he_is_looking':
        return U.present ? [clamp(U.gaze.yaw, -0.6, 0.6), clamp(U.gaze.pitch, -0.45, 0.45)] : (rec.lastLook || [0, 0]);
      case 'left_hand': case 'right_hand': {
        const h = U.hands[k === 'left_hand' ? 'left' : 'right'];
        return h.present ? (rec.lastLook = pos(h.x, h.y)) : (rec.lastLook || [0, 0]);
      }
      case 'hand': case 'hands': {
        const H = [U.hands.left, U.hands.right].filter(h => h.present);
        if (!H.length) return rec.lastLook || [0, 0];
        return (rec.lastLook = pos(H.reduce((s, h) => s + h.x, 0) / H.length, H.reduce((s, h) => s + h.y, 0) / H.length));
      }
      case 'screen_left': return [-0.5, 0.02];
      case 'screen_right': return [0.5, 0.02];
      case 'left': return [0.5, 0.02];      // its own left = the screen's right
      case 'right': return [-0.5, 0.02];
      case 'up': return [0, 0.35];
      case 'down': return [0, -0.3];
      case 'ahead': case 'forward': case 'straight': case 'center': return [0, 0];
      case 'away': if (!rec.awayDir) rec.awayDir = [(Math.random() < 0.5 ? -1 : 1) * 0.45, 0.05]; return rec.awayDir;
      default: return null;
    }
  }

  // ------------------------------------------------------------------ the runtime
  function createBehaviors(opts = {}) {
    const R = {progs: [], seq: 0, idn: 0, listeners: [], deps: {}, clock: 0, frame: 0, chan: {}, lastError: null,
      motion: null, installed: null, pending: new Map(), stats: {ticks: 0, ms: [], maxMs: 0}, character: opts.character || null,
      style: {}};
    const perc = createPerceiver();
    const FZ = Object.freeze;
    function emitEv(ev) {
      ev = Object.assign({t: +R.clock.toFixed(3)}, ev);
      for (const fn of R.listeners.slice()) { try { fn(ev); } catch (e) { /* the listener's problem */ } }
      return ev;
    }
    const character = () => {
      let c = null;
      try { c = typeof R.deps.character === 'function' ? R.deps.character() : R.deps.character; } catch (e) { c = null; }
      return c || R.character || 'fox';
    };
    const find = key => {
      if (key == null) return null;
      const k = String(key);
      return R.progs.find(p => p.id === k) || R.progs.find(p => p.name === k) || R.progs.find(p => p.name.toLowerCase() === k.toLowerCase()) || null;
    };
    const slug = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().split(' ').filter(w => !/^(the|a|an|my|your|me|you|i|when|whenever|every|time|and|to|of|it|is|on|at|for|if)$/.test(w)).slice(0, 4).join('_') || 'behavior';
    function uniqueName(base) { let n = base, i = 2; while (R.progs.some(p => p.name === n)) n = base + '_' + i++; return n; }

    // ---- ownership: group → program (highest priority, ties → newest)
    let owners = {};
    function resolveOwners() {
      owners = {};
      const order = R.progs.filter(p => p.status === 'running').sort((a, b) => b.priority - a.priority || b.seq - a.seq);
      for (const p of order) for (const g of p.claims) if (!owners[g]) owners[g] = p;
    }
    function warnOnce(p, key, ev) {
      if (p.warned.has(key)) return;
      p.warned.add(key);
      p.warnings.push(ev.msg || ev.what || key);
      if (p.warnings.length > 20) p.warnings.shift();
      emitEv(Object.assign({type: 'behavior', phase: ev.phase || 'warning', id: p.id, name: p.name}, ev));
    }

    function makeApi(p) {
      const api = {};
      const def = (k, get) => Object.defineProperty(api, k, {get, enumerable: true});
      def('t', () => p.t); def('dt', () => p.dt); def('now', () => R.clock); def('frame', () => p.frames);
      def('user', () => perc.snapshot() || perc.P.U);
      Object.defineProperty(api, 'state', {get: () => p.state, set: v => { if (v && typeof v === 'object') p.state = v; }, enumerable: true});
      def('character', () => character());
      def('name', () => p.name);
      const selfObj = FZ(Object.defineProperties({}, {
        speaking: {get: () => !!(R.ctx && R.ctx.voice && R.ctx.voice.avatar && R.ctx.voice.avatar.speaking), enumerable: true},
        level: {get: () => clamp(+(R.ctx && R.ctx.voice && R.ctx.voice.avatar && R.ctx.voice.avatar.level) || 0, 0, 1), enumerable: true},
        action: {get: () => selfState().action, enumerable: true},
        busy: {get: () => selfState().busy, enumerable: true},
        pose: {get: () => selfState().pose, enumerable: true},
        channel: {value: name => readChannel(name), enumerable: true},
      }));
      def('self', () => selfObj);
      api.gesture = name => perc.gesture(name);
      api.recent = (type, seconds = 2) => {
        const s = clamp(+seconds || 2, 0, 120), types = type == null ? null : [].concat(type).map(String);
        return perc.P.ring.filter(e => R.clock - e.t <= s && (!types || types.includes(e.type)));
      };
      api.events = (type) => {
        const types = type == null ? null : [].concat(type).map(String), s0 = p.evSeq;
        const out = [];
        for (let i = perc.P.ring.length - 1; i >= 0; i--) { const e = perc.P.ring[i]; if (e.seq <= s0) break; if (!types || types.includes(e.type)) out.push(e); }
        return out.reverse();
      };
      api.heard = (words) => perc.heardSince(p.evSeq, words);
      api.has = part => {
        const k = String(part || '').toLowerCase().trim().replace(/[\s-]+/g, '_'), ch = character();
        if (CH[k]) return hasChannel(ch, k);
        const need = PARTS[k];
        if (need === true) return true;
        if (need) return !!traitsOf(ch)[need];
        if (GROUP_CH[k] || GROUP_ALIAS[k]) return expandGroups([k]).groups.some(g => !groupMissing(ch, g));
        return false;
      };
      api.claim = (groups) => {
        const {groups: gs, bad} = expandGroups(groups);
        const missing = [], got = [];
        for (const g of gs) { if (groupMissing(character(), g)) missing.push(g); else { p.claims.add(g); got.push(g); } }
        if (bad.length) warnOnce(p, 'badgroup:' + bad.join(','), {msg: `unknown group(s) ${bad.join(', ')}; groups are ${GROUPS.join(', ')}`});
        if (missing.length) { for (const g of missing) p.missing.add(g); warnOnce(p, 'missing:' + missing.join(','), {phase: 'unsupported', what: missing.join(', '), msg: `${traitsOf(character()).name} has no ${missing.join(', ')}`}); }
        resolveOwners();
        return FZ({ok: !missing.length && !bad.length, groups: FZ(got), missing: FZ(missing.concat(bad))});
      };
      api.release = (groups) => {
        const {groups: gs} = expandGroups(groups == null ? GROUPS : groups);
        for (const g of gs) p.claims.delete(g);
        resolveOwners();
        return true;
      };
      api.owns = (group) => { const {groups: gs} = expandGroups(group); return gs.length > 0 && gs.every(g => owners[g] === p); };
      api.set = (channel, value, weight = 1) => setCh(p, channel, value, weight);
      api.get = name => readChannel(name);
      api.look = (target, a2) => {
        const o = typeof a2 === 'object' && a2 ? a2 : {};
        const seconds = typeof a2 === 'number' ? a2 : (Number.isFinite(+o.seconds) ? +o.seconds : 0);
        const ang = lookAngles(target, perc.snapshot() || perc.P.U, p.look);
        if (!ang) { warnOnce(p, 'look:' + String(target), {msg: `unknown look target ${JSON.stringify(target)}`}); return false; }
        p.look.target = target; p.look.until = seconds > 0 ? R.clock + Math.min(600, seconds) : R.clock; p.look.head = o.head !== false;
        p.look.frame = R.frame; p.look.speed = clamp(+o.speed || 1, 0.2, 3);
        if (!p.claims.has('gaze')) { p.claims.add('gaze'); resolveOwners(); }
        if (p.look.head && !p.claims.has('head')) { p.claims.add('head'); resolveOwners(); }
        return true;
      };
      api.paw_to = (paw, x, y, z) => {
        const side = /^l/i.test(String(paw)) ? 'left' : 'right';
        const sx = clamp(+x || 0, -1.2, 1.2), sy = clamp(+y || 0, -1.2, 1.2);
        const out = side === 'right' ? -sx : sx;   // its own right paw is on the screen's left
        const ok1 = setCh(p, side + '_paw_x', clamp(0.05 + 0.75 * out, -0.6, 1.1), 1);
        const ok2 = setCh(p, side + '_paw_y', sy >= 0 ? 0.45 + 0.6 * sy : 0.45 + 0.75 * sy, 1);
        setCh(p, side + '_paw_z', Number.isFinite(+z) ? clamp(+z, -1, 1.2) : 0.3, 1);
        return ok1 && ok2;
      };
      api.do_action = (description) => {
        const d = String(description || '').trim().slice(0, 300);
        if (!d) return FZ({ok: false, reason: 'empty description'});
        const t = R.clock;
        p.actT = p.actT.filter(x => t - x < 60);
        if (p.actT.length && t - p.actT[p.actT.length - 1] < LIMITS.actionGapS) return FZ({ok: false, reason: 'too soon after the last action'});
        if (p.actT.length >= LIMITS.actionsPerMin) return FZ({ok: false, reason: 'too many actions this minute'});
        p.actT.push(t); p.actions++;
        let r = null;
        try { r = R.deps.doAction ? R.deps.doAction(d, {id: p.id, name: p.name}) : null; } catch (e) { r = {ok: false, error: String(e && e.message || e).slice(0, 120)}; }
        const ok = !!R.deps.doAction && !(r && r.ok === false);
        emitEv({type: 'behavior_action', id: p.id, name: p.name, description: d, ok});
        if (p.trace) p.trace.push({t: R.clock, kind: 'action', description: d});
        return FZ({ok, status: r && typeof r.status === 'string' ? r.status : (ok ? 'requested' : 'unavailable')});
      };
      api.set_style = (s) => {
        if (!s || typeof s !== 'object') return false;
        const out = {};
        for (const k of ['energy', 'mirroring', 'expressiveness']) if (k in s) out[k] = s[k] === null ? null : clamp(+s[k] || 0, 0, 1);
        if (!Object.keys(out).length) return false;
        const same = Object.keys(out).every(k => p.style[k] === out[k]);
        if (same) return true;
        if (R.clock - p.styleT < LIMITS.styleGapS) { p.stylePend = Object.assign(p.stylePend || {}, out); return true; }
        p.styleT = R.clock; Object.assign(p.style, out); p.styleSet = true;
        sendStyle(out);
        if (p.trace) p.trace.push({t: R.clock, kind: 'style', style: out});
        return true;
      };
      api.emit = (name, data, reply = true) => {
        const n = String(name || 'moment').slice(0, 60);
        const t = R.clock;
        p.emitT = p.emitT.filter(x => t - x < 60);
        if (p.emitT.length && t - p.emitT[p.emitT.length - 1] < LIMITS.emitGapS) return false;
        if (p.emitT.length >= LIMITS.emitsPerMin) return false;
        let d = null;
        try { d = data === undefined ? null : JSON.parse(JSON.stringify(data)); } catch (e) { d = String(data); }
        const ds = d === null ? '' : JSON.stringify(d);
        if (ds.length > 300) d = ds.slice(0, 300);
        p.emitT.push(t); p.emits++;
        emitEv({type: 'rule_moment', id: p.id, name: p.name, instruction: p.instruction, event: n, data: d, reply: reply !== false});
        if (p.trace) p.trace.push({t: R.clock, kind: 'emit', event: n, data: d});
        return true;
      };
      api.done = (reason) => { p.doneReason = String(reason == null ? 'done' : reason).slice(0, 120); return true; };
      api.log = (...a) => plog(p, 'log', a);
      return FZ(api);
    }
    function plog(p, kind, a) {
      let s = '';
      try { s = a.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' '); } catch (e) { s = String(a); }
      p.logs.push(`${R.clock.toFixed(2)} ${kind} ${s.slice(0, 200)}`);
      if (p.logs.length > LIMITS.logRing) p.logs.shift();
    }
    function sendStyle(s) { try { if (R.deps.setStyle) R.deps.setStyle(Object.assign({}, s)); } catch (e) { /* page */ } }

    // ---- the self view: current values (before programs) and the commanded state (lazy, once per tick)
    let selfCache = null;
    function selfState() {
      if (selfCache && selfCache.frame === R.frame) return selfCache.v;
      let st = null;
      try { st = R.ctx && typeof R.ctx.state === 'function' ? R.ctx.state() : (R.ctx && R.ctx.state) || null; } catch (e) { st = null; }
      let action = null, busy = false;
      if (st) {
        const act = (st.active || []).filter(a => a && a.kind !== 'look');
        if (st.kf && st.kf.description) action = String(st.kf.description);
        else if (act.length) action = String(act[act.length - 1].name || act[act.length - 1].kind);
        busy = act.length > 0;
      }
      try { const own = R.ctx && typeof R.ctx.owned === 'function' ? R.ctx.owned() : null; if (own && Object.keys(own).length) busy = true; } catch (e) { /* */ }
      const pose = FZ({posture: (st && st.posture) || 'seated', seated: !st || st.posture === 'seated' || !!st.sittingOn,
        holding: FZ(Object.assign({left: null, right: null}, (st && st.holding) || {})), wearing: FZ(((st && st.wearing) || []).map(String)),
        facing: (st && st.facing) || 'camera'});
      const v = FZ({action, busy, pose});
      selfCache = {frame: R.frame, v};
      return v;
    }
    function readChannel(name) {
      const c = CH[String(name || '')];
      if (!c) return 0;
      const o = R.ctx && R.ctx.out;
      if (!o) return 0;
      const vals = c.rig.map(r => +o[r] || 0);
      return c.sign * (vals.reduce((s, x) => s + x, 0) / vals.length);
    }
    function setCh(p, channel, value, weight) {
      const name = String(channel || '');
      const c = CH[name] || (RIG_TO_CH[name] ? CH[RIG_TO_CH[name]] : null);
      if (!c) { warnOnce(p, 'unknownch:' + name, {msg: `unknown channel '${name}'`}); return false; }
      if (!hasChannel(character(), c.name)) { warnOnce(p, 'missingch:' + c.name, {phase: 'unsupported', what: c.name, msg: `${traitsOf(character()).name} has no ${c.need} (${c.name})`}); return false; }
      if (!p.claims.has(c.group)) { warnOnce(p, 'unclaimed:' + c.group, {msg: `set('${c.name}') needs api.claim(['${c.group}']) first`}); p.unclaimed++; return false; }
      if (owners[c.group] !== p) return false;
      let v = +value, w = weight === undefined ? 1 : +weight;
      if (!Number.isFinite(v)) { warnOnce(p, 'nan:' + c.name, {msg: `set('${c.name}') got a non-number`}); return false; }
      if (!Number.isFinite(w)) w = 1;
      p.sets[c.name] = [clamp(v, c.range[0], c.range[1]), clamp(w, 0, 1)];
      p.nSets++;
      return true;
    }
    function applyLook(p, dt) {
      const L = p.look;
      if (!L || L.target === undefined || (L.frame !== R.frame && R.clock > L.until)) { if (L) { L.eye = null; L.hd = null; } return; }
      const U = perc.snapshot() || perc.P.U;
      const ang = lookAngles(L.target, U, L) || [0, 0];
      const W = [clamp(0.85 * ang[0], -0.6, 0.6), clamp(0.85 * ang[1], -0.45, 0.45)];
      const H = L.head ? [clamp(0.75 * ang[0], -0.6, 0.6), clamp(0.6 * ang[1], -0.45, 0.45)] : [0, 0];
      if (!L.eye) { L.eye = [readChannel('gaze_x') * GAZE_RX, readChannel('gaze_y') * GAZE_RY]; L.hd = [readChannel('head_yaw'), readChannel('head_pitch')]; }
      for (let i = 0; i < 2; i++) { L.eye[i] = lp(L.eye[i], W[i], dt, 0.035 / L.speed); L.hd[i] = lp(L.hd[i], H[i], dt, 0.22 / L.speed); }
      // gaze is world-referenced (an offset from the camera direction, independent of the head): eyes take the target, the
      // head carries the rest
      setCh(p, 'gaze_x', clamp(L.eye[0] / GAZE_RX, -1, 1), 1);
      setCh(p, 'gaze_y', clamp(L.eye[1] / GAZE_RY, -1, 1), 1);
      if (L.head) { setCh(p, 'head_yaw', L.hd[0], 1); setCh(p, 'head_pitch', L.hd[1], 1); }
    }

    // ---- lifecycle
    function newRecord(spec, comp) {
      const p = {id: 'b' + (++R.idn), name: '', instruction: String(spec.instruction || '').slice(0, 600), code: String(spec.code),
        priority: Number.isFinite(+spec.priority) ? Math.round(clamp(+spec.priority, -100, 100)) : 5, seq: ++R.seq, since: R.clock,
        t: 0, dt: 0, frames: 0, errors: 0, status: 'running', claims: new Set(), missing: new Set(), sets: {}, nSets: 0,
        ring: new Float64Array(BUDGET.window), ringN: 0, overs: 0, hardT: [], maxMs: 0, totMs: 0, look: {}, style: {}, styleT: -9,
        actT: [], emitT: [], actions: 0, emits: 0, logs: [], warned: new Set(), warnings: [], unclaimed: 0, doneReason: null,
        evSeq: perc.P.seq, G: makeGuard(), comp, prog: null, lastError: null, trace: spec.trace ? [] : null, version: 1};
      p.safe = makeSafe((kind, a) => plog(p, kind, a));
      p.state = {};
      p.api = makeApi(p);
      return p;
    }
    function runHook(p, fn, limit) {
      p.G.begin(limit);
      const t0 = now();
      let err = null;
      try { fn.call(p.prog, p.api); } catch (e) { err = e; }
      const ms = now() - t0;
      return {ms, err, cut: p.G.tripped};
    }
    function errText(e) { return String(e && e.message || e).slice(0, 240); }
    function start(spec) {
      spec = spec || {};
      R.lastError = null;
      const comp = compile(spec.code);
      const nm = String(spec.name || '').trim().slice(0, 60) || slug(spec.instruction);
      if (!comp.ok) {
        R.lastError = comp.error;
        emitEv({type: 'behavior', phase: 'error', name: nm, instruction: spec.instruction || '', error: comp.error, where: 'compile'});
        return null;
      }
      const old = find(nm);
      if (old) stopOne(old, 'replaced');
      const p = newRecord(spec, comp);
      p.name = old ? nm : uniqueName(nm);
      try { p.prog = comp.make(p.G, p.safe); } catch (e) { R.lastError = errText(e); emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, error: R.lastError, where: 'load'}); return null; }
      R.progs.push(p);
      if (typeof p.prog.setup === 'function') {
        const r = runHook(p, p.prog.setup, BUDGET.setupMs);
        if (r.err) {
          R.progs.splice(R.progs.indexOf(p), 1); resolveOwners();
          R.lastError = (r.cut ? 'setup over its time budget' : 'setup failed: ') + (r.cut ? '' : errText(r.err));
          emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, error: R.lastError, where: 'setup'});
          install();
          return null;
        }
      }
      resolveOwners();
      emitEv({type: 'behavior', phase: 'start', id: p.id, name: p.name, instruction: p.instruction, priority: p.priority, groups: [...p.claims], missing: [...p.missing]});
      install();
      return p.id;
    }
    function change(key, spec) {
      const p = find(key);
      spec = spec || {};
      R.lastError = null;
      if (!p) { R.lastError = `no behavior named ${key}`; return false; }
      const comp = compile(spec.code);
      if (!comp.ok) { R.lastError = comp.error; emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, error: comp.error, where: 'compile (change: the old version keeps running)'}); return false; }
      let prog;
      const G = makeGuard();
      try { prog = comp.make(G, p.safe); } catch (e) { R.lastError = errText(e); emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, error: R.lastError, where: 'load (change)'}); return false; }
      const save = {prog: p.prog, G: p.G, comp: p.comp, claims: new Set(p.claims), code: p.code, instruction: p.instruction, status: p.status, priority: p.priority};
      p.prog = prog; p.G = G; p.comp = comp; p.code = String(spec.code);
      if (spec.instruction) p.instruction = String(spec.instruction).slice(0, 600);
      if (Number.isFinite(+spec.priority)) p.priority = Math.round(clamp(+spec.priority, -100, 100));
      p.claims = new Set(); p.status = 'running'; p.errors = 0; p.overs = 0; p.ringN = 0; p.hardT = []; p.warned = new Set(); p.doneReason = null; p.version++;
      if (typeof prog.setup === 'function') {
        const r = runHook(p, prog.setup, BUDGET.setupMs);
        if (r.err) {
          Object.assign(p, {prog: save.prog, G: save.G, comp: save.comp, claims: save.claims, code: save.code, instruction: save.instruction, status: save.status, priority: save.priority});
          resolveOwners();
          R.lastError = r.cut ? 'setup over its time budget' : 'setup failed: ' + errText(r.err);
          emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, error: R.lastError, where: 'setup (change: the old version keeps running)'});
          return false;
        }
      }
      // a claim the old version held but the new one dropped is released (its channels fade back over W_OUT)
      resolveOwners();
      emitEv({type: 'behavior', phase: 'changed', id: p.id, name: p.name, instruction: p.instruction, priority: p.priority, groups: [...p.claims], missing: [...p.missing]});
      install();
      return true;
    }
    function stopOne(p, reason, err) {
      if (p.prog && typeof p.prog.end === 'function' && p.status !== 'error-end') {
        const r = runHook(p, p.prog.end, BUDGET.endMs);
        if (r.err) plog(p, 'error', ['end() failed: ' + errText(r.err)]);
      }
      if (p.styleSet) { const reset = {}; for (const k of Object.keys(p.style)) reset[k] = null; sendStyle(reset); }
      const i = R.progs.indexOf(p); if (i >= 0) R.progs.splice(i, 1);
      resolveOwners();
      emitEv({type: 'behavior', phase: 'end', id: p.id, name: p.name, instruction: p.instruction, reason: reason || 'stopped', frames: p.frames, error: err || undefined});
    }
    function stop(key) {
      if (key === 'all' || key === '*') { const all = R.progs.slice(); for (const p of all) stopOne(p, 'stopped'); install(); return all.length; }
      const p = find(key); if (!p) return 0;
      stopOne(p, 'stopped'); install();
      return 1;
    }
    function resume(key) {
      const p = find(key); if (!p || p.status === 'running') return false;
      p.status = 'running'; p.overs = 0; p.ringN = 0; p.hardT = []; resolveOwners(); install();
      emitEv({type: 'behavior', phase: 'resumed', id: p.id, name: p.name});
      return true;
    }
    function pause(p, reason) {
      p.status = 'paused'; resolveOwners();
      const ms = msStats(p);
      emitEv({type: 'behavior', phase: 'paused', id: p.id, name: p.name, reason, ms_p50: ms.p50, ms_max: ms.max});
    }
    function fail(p, err, where) {
      p.status = 'error'; p.errors++; p.lastError = errText(err); resolveOwners();
      emitEv({type: 'behavior', phase: 'error', id: p.id, name: p.name, instruction: p.instruction, error: p.lastError, where, frame: Math.max(0, p.frames - 1),
        code: p.code});
    }
    function msStats(p) {
      const n = Math.min(p.ringN, BUDGET.window), a = Array.from(p.ring.slice(0, n)).sort((x, y) => x - y);
      return {p50: n ? +a[Math.floor(n / 2)].toFixed(4) : 0, max: +p.maxMs.toFixed(4), mean: p.frames ? +(p.totMs / p.frames).toFixed(4) : 0};
    }
    function list() {
      return R.progs.map(p => {
        const ms = msStats(p);
        return {id: p.id, name: p.name, instruction: p.instruction, priority: p.priority, groups: [...p.claims].filter(g => owners[g] === p),
          claimed: [...p.claims], since: +p.since.toFixed(2), frames: p.frames, errors: p.errors, status: p.status, missing: [...p.missing],
          ms_p50: ms.p50, ms_max: ms.max, error: p.lastError || undefined, warnings: p.warnings.slice(-5), version: p.version};
      });
    }

    // ---- the frame
    function tick(ctx) {
      const tt0 = now();
      ctx = ctx || {};
      const dt = clamp(+ctx.dt || 0, 0, 0.25);
      R.clock += dt; R.frame++; R.ctx = ctx; selfCache = null;
      perc.update(ctx, R.clock, dt);
      const ch = character();
      for (const p of R.progs.slice()) {
        if (p.status !== 'running') continue;
        p.dt = dt; p.t = R.clock - p.since; p.sets = {};
        if (p.stylePend && R.clock - p.styleT >= LIMITS.styleGapS) { const s = p.stylePend; p.stylePend = null; p.styleT = R.clock; Object.assign(p.style, s); p.styleSet = true; sendStyle(s); }
        const r = runHook(p, p.prog.per_frame, BUDGET.hardMs);
        p.frames++; p.totMs += r.ms; if (r.ms > p.maxMs) p.maxMs = r.ms;
        p.ring[p.ringN++ % BUDGET.window] = r.ms;
        if (r.cut) {
          p.hardT = p.hardT.filter(x => R.clock - x < BUDGET.hardWindowS); p.hardT.push(R.clock); p.sets = {};
          plog(p, 'error', [`frame cut off after ${r.ms.toFixed(1)} ms`]);
          if (p.hardT.length >= BUDGET.hardTrips) { pause(p, `over its time budget: ${p.hardT.length} frames cut off at ${BUDGET.hardMs} ms`); continue; }
        } else if (r.err) { fail(p, r.err, 'per_frame'); continue; }
        // budget over the last window: steadily slow (many overruns with a slow median, or a mean over budget) → paused.
        // Lone spikes (a GC pause, the first JIT frame, a busy phone) do not count against it.
        const n = Math.min(p.ringN, BUDGET.window);
        if (n >= 10) {
          let over = 0, sum = 0; for (let i = 0; i < n; i++) { if (p.ring[i] > BUDGET.frameMs) over++; sum += p.ring[i]; }
          const mean = sum / n;
          let slow = mean > BUDGET.frameMs;
          if (!slow && over >= BUDGET.overN) { const a = Array.from(p.ring.slice(0, n)).sort((x, y) => x - y); slow = a[n >> 1] > 0.5 * BUDGET.frameMs; }
          if (slow) { pause(p, `over its per-frame budget (${BUDGET.frameMs} ms): ${over} of the last ${n} frames over, mean ${mean.toFixed(2)} ms`); continue; }
        }
        applyLook(p, dt);
        p.evSeq = perc.P.seq;
        if (p.trace) p.trace.push({t: +R.clock.toFixed(3), sets: Object.assign({}, p.sets)});
        if (p.doneReason !== null) { stopOne(p, 'done: ' + p.doneReason); }
      }
      // compose: per channel the owning program's value, eased in / out
      const out = {};
      let any = false;
      const live = new Set(Object.keys(R.chan));
      for (const p of R.progs) if (p.status === 'running') for (const k in p.sets) live.add(k);
      for (const name of live) {
        const c = CH[name];
        const own = owners[c.group];
        const s = own && own.status === 'running' ? own.sets[name] : null;
        let st = R.chan[name];
        if (!s && !st) continue;
        if (!st) st = R.chan[name] = {w: 0, v: s[0], owner: own, from: null, xt: 0};
        const wt = s ? s[1] : 0;
        if (s) {
          if (st.owner !== own && st.w > 0.05) { st.from = st.v; st.xt = 0; }
          st.owner = own;
          if (st.from !== null) { st.xt += dt; const k = mj(st.xt / XFADE); st.v = lerp(st.from, s[0], k); if (k >= 1) st.from = null; } else st.v = s[0];
        }
        st.w = lp(st.w, wt, dt, wt > st.w ? W_IN : W_OUT);
        if (st.w < 1e-3 && !s) { delete R.chan[name]; continue; }
        if (!hasChannel(ch, name)) continue;
        for (const r of c.rig) { out[r] = [c.sign * st.v, st.w]; any = true; }
      }
      const ms = now() - tt0;
      R.stats.ticks++; R.stats.ms.push(ms); if (R.stats.ms.length > 300) R.stats.ms.shift(); if (ms > R.stats.maxMs) R.stats.maxMs = ms;
      if (!R.progs.length && !Object.keys(R.chan).length) install();
      return any ? out : null;
    }

    // ---- the motion hook
    const layerFn = (ctx) => tick(ctx);
    function motionNow() {
      let m = null;
      try { m = typeof R.deps.motion === 'function' ? R.deps.motion() : R.deps.motion; } catch (e) { m = null; }
      return m && typeof m.setProgramLayer === 'function' ? m : null;
    }
    function install() {
      const m = motionNow();
      const want = R.progs.length > 0 || Object.keys(R.chan).length > 0;
      if (R.installed && R.installed !== m) { try { R.installed.setProgramLayer(null); } catch (e) { /* */ } R.installed = null; }
      if (!m) return false;
      if (want && R.installed !== m) { m.setProgramLayer(layerFn); R.installed = m; }
      else if (!want && R.installed === m) { m.setProgramLayer(null); R.installed = null; }
      return want;
    }
    function attach(deps) { R.deps = Object.assign({}, R.deps, deps || {}); install(); return api; }

    // ---- the server round trip: request(instruction, {name, priority, edit}) → Promise<{ok, id, name, error, ms}>
    function request(instruction, o = {}) {
      const ins = String(instruction || '').trim().slice(0, 600);
      if (!ins) return Promise.resolve({ok: false, error: 'empty instruction'});
      const edit = !!o.edit;
      const prev = edit ? find(o.name) : null;
      if (edit && !prev) return Promise.resolve({ok: false, error: `no behavior named ${o.name}`});
      const name = prev ? prev.name : (String(o.name || '').trim().slice(0, 60) || slug(ins));
      const rid = 'bc' + (++R.idn) + '_' + Math.floor(Math.random() * 1e6).toString(36);
      const msg = {type: 'behavior_code', id: rid, instruction: prev ? ins : ins, name, character: character(),
        prev_code: prev ? prev.code : undefined, prev_instruction: prev ? prev.instruction : undefined,
        gestures: typeof R.deps.gestures === 'function' ? (() => { try { return R.deps.gestures(); } catch (e) { return undefined; } })() : undefined};
      if (o.words) msg.his_words = String(o.words).slice(0, 400);   // games: his own latest words
      return new Promise(resolve => {
        const t0 = now();
        const timer = setTimeoutSafe(() => { if (R.pending.delete(rid)) resolve({ok: false, name, error: 'the coder did not answer in time'}); }, +o.timeoutMs || 30000);
        R.pending.set(rid, {resolve, t0, name, edit, priority: o.priority, instruction: ins, timer});
        try { if (!R.deps.requestCode) throw new Error('no hub link'); R.deps.requestCode(msg); }
        catch (e) { R.pending.delete(rid); clearTimeoutSafe(timer); resolve({ok: false, name, error: errText(e)}); }
      });
    }
    function onCode(msg) {
      if (!msg || !R.pending.has(msg.id)) return false;
      const q = R.pending.get(msg.id); R.pending.delete(msg.id); clearTimeoutSafe(q.timer);
      if (!msg.ok || !msg.code) { q.resolve({ok: false, name: q.name, error: String(msg.error || 'the coder failed').slice(0, 300), hub_ms: msg.ms}); return true; }
      let id = null, ok = false;
      if (q.edit && find(q.name)) {
        const prev = find(q.name), combined = (prev.instruction + ' — then: ' + q.instruction).slice(-600);
        ok = change(q.name, {code: msg.code, instruction: combined}); id = ok ? prev.id : null;
      }
      else { id = start({name: q.name, instruction: q.instruction, code: msg.code, priority: q.priority}); ok = !!id; }
      const p = id ? find(id) : null;
      q.resolve({ok, id, name: p ? p.name : q.name, error: ok ? undefined : R.lastError, ms: Math.round(now() - q.t0), hub_ms: msg.ms,
        groups: p ? [...p.claims] : undefined, missing: p ? [...p.missing] : undefined, notes: msg.notes});
      return true;
    }
    const setTimeoutSafe = (f, ms) => (typeof setTimeout === 'function' ? setTimeout(f, ms) : null);
    const clearTimeoutSafe = id => { if (id !== null && typeof clearTimeout === 'function') clearTimeout(id); };

    const api = {start, change, stop, resume, list, tick, attach, request, onCode, install,
      onEvent(fn) { if (typeof fn !== 'function') return () => {}; R.listeners.push(fn); return () => { const i = R.listeners.indexOf(fn); if (i >= 0) R.listeners.splice(i, 1); }; },
      feed: ev => perc.feed(ev),
      info: key => { const p = find(key); return p ? Object.assign(list().find(x => x.id === p.id), {code: p.code, logs: p.logs.slice(), state: p.state, trace: p.trace}) : null; },
      get lastError() { return R.lastError; },
      stats() { const a = R.stats.ms.slice().sort((x, y) => x - y); return {programs: R.progs.length, ticks: R.stats.ticks, ms_p50: a.length ? a[Math.floor(a.length / 2)] : 0, ms_p95: a.length ? a[Math.floor(a.length * 0.95)] : 0, ms_max: R.stats.maxMs, installed: !!R.installed}; },
      setCharacter(c) { R.character = c; },
      _R: R, _perc: perc};
    return api;
  }

  // ------------------------------------------------------------------ the toolkit doc (the coder's whole API)
  const TOOLKIT_DOC = `You write ONE small program that runs inside a live 3-D cartoon character on a video call with a person ("him").
The program gives the character a STANDING behaviour: it runs every frame (~30 per second) until it finishes or is stopped,
reads what the camera and microphone see of him through a fixed SENSE api, and moves the character's own body through a
fixed ACT api. Everything else the character does by itself (blinking, breathing, looking at him, its voice, lip sync);
your program only takes over the body parts it CLAIMS, and only while it writes them.

PROGRAM FORMAT — reply with ONLY this JavaScript object literal (no prose, no markdown fences):
({
  setup(api) { ... },       // once at the start (and again after a live edit: api.state then still holds the old data)
  per_frame(api) { ... },   // every frame
  end(api) { ... }          // optional: once when it stops
})
Plain modern JavaScript only. Not available: this, class, async/await, generators, import, eval, Function, timers, Date,
window/document/fetch or any other global, and the names constructor / prototype / __proto__ / anything starting "__".
Available: Math, JSON, Number, String, Boolean, Array, Object (keys/values/entries/assign/freeze/fromEntries/hasOwn), Map,
Set, Error, isFinite, isNaN, parseFloat, parseInt, console.log. Keep per_frame tiny (well under 1 ms: no big loops); a
program that is too slow is paused, an exception stops it. Keep everything that must last between frames in api.state
(an object you own; it survives live edits — so in setup write api.state.n ??= 0 rather than resetting it).
Use helper functions inside the object literal or inside per_frame.

TIME
  api.t        seconds since this program started            api.dt     seconds since the previous frame (~0.033)
  api.now      seconds on the shared clock                   api.frame  frames run so far

SENSE — him (read-only). Sides of HIM are HIS OWN left/right. Directions are as seen on the SCREEN (his own camera
preview is a mirror, so his own right appears on the screen's right): x + = screen-right, y + = up.
SIDES: the two of you face each other, so HIS right hand and the character's own LEFT paw / ear / eye are on the SAME side
of the screen (mirror image). "Same side of the screen" / "like a mirror" = his right ↔ its left; "the same hand" = his
right ↔ its right.
  api.user.present           true while his face is in view
  api.user.face(name)        0..1 expression score; 0 when no face. Names: smile, frown, sad, surprise, angry, mouth_open,
                             brows_up, brows_down, squint, eyes_wide, blink (both eyes), blink_left / blink_right (HIS eyes),
                             eyes_closed, wink, pucker, cheek_puff, sneer, or any MediaPipe blendshape name (jawOpen,
                             mouthSmileLeft, eyeBlinkRight, browInnerUp, ...)
  api.user.expression        'happy' | 'sad' | 'frown' | 'surprised' | 'raised_brows' | 'neutral'
  api.user.head              {yaw, pitch, roll} radians. yaw + = he turns his face to his right; pitch + = he looks up;
                             roll + = the top of his head tilts to his right. ~0.15 rad = a clear tilt/turn.
  api.user.gaze              {atCamera (bool: looking at the screen), yaw, pitch (radians, where his head+eyes point;
                             + = his right / up), attention 0..1}
  api.user.position          {x, y}: his face in the camera frame, -1..1 (x + = screen-right / his right, y + = up)
  api.user.distance          metres from the camera (0 when no face; ~0.3-0.7 on a phone)
  api.user.hands.left / .right   HIS left / right hand: {present, x, y (-1..1, same frame as position), z (apparent
                             size, bigger = nearer), raised (above his chin), open 0..1 (fist..spread), pinch 0..1,
                             fingers (0..5 held up), gesture ('open_palm'|'fist'|'pointing_up'|'thumbs_up'|'thumbs_down'|
                             'victory'|'i_love_you'|'none'), pointing (bool), touching (face part it touches or null)}
  api.user.touching(part)    true while either hand touches that part of HIS face: 'nose', 'mouth', 'chin', 'forehead',
                             'head', 'eyes' / 'left_eye' / 'right_eye', 'cheek' / 'left_cheek' / 'right_cheek',
                             'ear' / 'left_ear' / 'right_ear', 'face' (any)
  api.user.touching_hand(part)   'left' | 'right' | null — which of his hands touches it
  api.user.pose              {present, lean (radians, + = upper body tilts to his right), shrug 0..1}
  api.user.speaking          true while he talks;  api.user.level  0..1 how loud
  api.user.moving            0..1 how much he moves
  api.gesture(name)          {on, score} — a gesture he taught by name, or a built-in: wave, nod, head_shake, thumbs_up,
                             thumbs_down, fist, open_palm, victory, point, hands_together, clap, shrug, smile, blink, wink,
                             hand_up, head_tilt, look_away, touch_<part> (e.g. touch_nose)
  api.events(type?)          the events that happened since YOUR previous frame (each exactly once) — the way to react to
                             or count discrete moments: for (const e of api.events('blink')) api.state.n++;
                             type = a string, a list of strings, or nothing (all)
  api.recent(type?, seconds=2)   events of the last few seconds (may repeat across frames)
  api.heard(words)           how many times he said that word / any of those words since your previous frame (speech
                             transcript, ~1 s late)
  Event types (each {type, t, ...}): blink, wink {eye}, eyes_closed, eyes_open, smile_start, smile_end, mouth_open,
    mouth_close, brows_up, brows_down, frown, nod, head_shake, head_tilt {dir:'left'|'right'}, head_straight,
    head_turn {dir}, look_away {dir}, look_back, face_appear, face_lost, lean_in, lean_back, speech_start, speech_end,
    hand_appear / hand_gone / hand_up / hand_down {hand:'left'|'right'}, hand_gesture {name, hand}, touch {part, hand},
    touch_end {part, hand}, hands_together, hands_apart, wave, point, thumbs_up, object_up, shrug, expression_change,
    gesture {name, score} (taught gestures), said {text}, self_speech_start, self_speech_end (the character's own voice),
    action_start / action_end {name} (commanded actions), attention {what}, turn_bid {cue}

SENSE — itself
  api.self.speaking / api.self.level   the character is talking now / how loud
  api.self.action            what commanded action it is doing now (a description) or null; api.self.busy (bool)
  api.self.pose              {posture, seated, holding {left, right}, wearing [...], facing}
  api.self.channel(name)     the current value of one of its controls (below) before your program — e.g. to add on top
  api.has(part)              whether this character has that part: 'ears', 'tail', 'nose', 'arms', 'legs', 'fingers', ...
  api.character              which character it is

ACT
  api.claim(groups)          take over body-part GROUPS before writing them (in setup, or later). Returns
                             {ok, groups, missing}. Groups: head, gaze, eyes, brows, mouth, nose, ears, body, right_arm,
                             left_arm, tail (also: arms = both arms, face = eyes+brows+mouth+nose+ears, posture = head+body).
                             Claim ONLY what you write: unclaimed parts stay natural and alive. A claim of a part the
                             character lacks fails (missing) — then do the closest thing with parts it has.
  api.release(groups?)       give groups back (all if none given)
  api.owns(group)            false while a higher-priority behaviour holds that group
  api.set(control, value, weight=1)   write one control of a claimed group THIS frame. Values are POSITIONS (clamped to
                             the range): to wiggle, wag, shake, bob or flap, move the value back and forth yourself over
                             time, e.g. 0.6*Math.sin(2*Math.PI*4*api.t) for 4 cycles a second. A control you stop
                             writing eases back to the character's own motion within ~0.2 s, so write it EVERY frame for
                             as long as it should hold. weight 0..1 blends with the character's own motion.
  api.look(target, seconds?) turn eyes (and head) toward a target this frame (or for the given seconds); claims gaze+head.
                             target: 'user' (his face), 'gaze' (where HE is looking), 'left_hand' / 'right_hand' / 'hands'
                             (his), 'up', 'down', 'screen_left', 'screen_right', 'away', 'ahead', or {x, y} in -1..1 of
                             his camera frame (e.g. a hand's x, y). api.look(target, {head:false}) moves only the eyes.
  api.paw_to(paw, x, y, z?)  put its 'right' or 'left' paw where a point of HIS camera frame (x, y in -1..1) appears on
                             the screen (claim that arm first). Facing each other, his RIGHT hand appears on the same
                             screen side as the character's LEFT paw: to mirror his right hand use paw_to('left', x, y).
  api.do_action(description) ask for a whole commanded action in plain words, played by the action system (whole-body
                             moves, walking to things, picking objects up, dances, jumps, waves...): e.g. "do a small
                             jump", "wave with the right paw". Takes ~0.5-3 s to start; at most one per second; check
                             api.self.action / action_end events to wait for it.
  api.emit(name, data?, reply=true)   tell the voice side that a moment happened (it may say something about it if reply):
                             e.g. api.emit('count_reached', {count: 10}). Emit only the moments he asked to hear about (not
                             every step toward them). Never put words for it to say in data — give facts; it words the
                             reply itself. At most a few per second.
  api.set_style({energy, mirroring, expressiveness})   each 0..1 (null = automatic): how lively / how much it mirrors him /
                             how expressive its natural motion is while the program runs.
  api.done(reason)           finish the program after this frame (its end() runs, its parts go back to natural motion).

CONTROLS (api.set names; ranges clamp). Sided parts use the character's OWN left/right (facing him, its own right is on
the screen's LEFT); directions are as seen on the screen. A part in (brackets) exists only on characters that have it —
THE CHARACTER section below lists exactly which controls this one has.
%CONTROLS%

READING THE REQUEST
  It names what HE does (a cue your program SENSES) and what the CHARACTER does (motion your program WRITES). Never make
  the character act out his part, and never wait for him to do the character's part.
  Whatever the character is asked to do must show on its body: api.set / api.look / api.paw_to / api.do_action.
  api.emit only tells the voice side something; it moves nothing.
  copy / mimic / mirror / do back = every frame (while allowed), drive the character's matching control from his matching
    sense value, e.g. api.set('smile', api.user.face('smile')). Each copied part follows its own sense value on its own:
    never wait for several of his cues at once or for a threshold before copying (a copy of a still face is a still face).
  whenever / every time / each time X, do Y = do Y while X holds, or a timed Y of a few seconds after each start of X;
    then stop writing it so it settles. Never latch Y on for good after the first X. "Until he says stop / changes it"
    is how long the RULE stands (he ends the program with his voice), never how long one reaction lasts.
  stay still / freeze / don't move (while ...) = during that time claim the parts and write a fixed resting value to each
    of their controls every frame (a part you do not write keeps its natural motion).
  take turns / when he pauses / after he finishes = his turn is while he talks (api.user.speaking), the character's
    turn is the gap after his speech ends (speech_end); the character does not have to talk for its turn to start.
  Write only the motion the request asks for: no decorative wagging, bouncing or swaying on top (its own natural motion
    already keeps it alive), least of all on parts a still / freeze rule covers.

USEFUL PATTERNS (idioms, not answers)
  start of a condition:   const on = <condition>; if (on && !api.state.was) { ... } api.state.was = on;
  a timed reaction:       api.state.until = api.t + 2;   then each frame:  if (api.t < api.state.until) { ...write... }
  smooth toward a value:  s.v = (s.v || 0) + (target - (s.v || 0)) * Math.min(1, api.dt * 8);
  one-shot pulse shape:   const k = (api.t - s.t0) / dur;  const env = k < 0 || k > 1 ? 0 : Math.sin(Math.PI * k);
  stop after a while:     if (api.t > 60) api.done('time up');
  When he asks to be told something, emit a fact with api.emit; never simulate speech.`;

  function controlsDoc() {   // every control, grouped (static: the same text for every character)
    const lines = [];
    for (const g of GROUPS) {
      lines.push(`  [${g}]`);
      for (const n of GROUP_CH[g]) { const c = CH[n]; lines.push(`    ${n} ${c.range[0]}..${c.range[1]}${c.need && c.need !== 'arms' && c.need !== 'brows' ? ' (' + c.need + ')' : ''}: ${c.desc}`); }
    }
    return lines.join('\n');
  }
  function characterDoc(character) {
    const tr = traitsOf(character);
    const has = ['arms', 'ears', 'tail', 'nose', 'legs', 'fingers'].filter(k => tr[k]);
    const lacks = ['ears', 'tail', 'nose', 'legs', 'fingers'].filter(k => !tr[k]);
    const groups = GROUPS.filter(g => !groupMissing(character, g));
    return `THE CHARACTER: ${tr.name}, a cartoon ${tr.what}` + (tr.leaves_seat ? ' with a full body (it can stand, walk, sit and handle the objects in its scene — whole-body moves go through api.do_action)'
      : ' that sits in one place (it never stands or walks: whole-body moves are small bobs, leans and swivels in its seat)') +
      `. It has: head, eyes, brows, mouth, ${has.join(', ')}; it has NO ${lacks.join(', ') || 'missing parts'}.\n` +
      `Its groups: ${groups.join(', ')}.\nIts controls: ${channelsOf(character).join(', ')}.`;
  }
  let DOC_CACHE = null;
  function toolkitDoc() { return DOC_CACHE || (DOC_CACHE = TOOLKIT_DOC.replace('%CONTROLS%', controlsDoc())); }

  return {createBehaviors, compile, tokenize, staticCheck, instrument, createPerceiver, lookAngles, toolkitDoc, characterDoc, controlsDoc,
    CH, GROUPS, GROUP_CH, GROUP_ALIAS, TRAITS, BUDGET, LIMITS, channelsOf, hasChannel, expandGroups, RIG_TO_CH, TOOLKIT_DOC};
});
