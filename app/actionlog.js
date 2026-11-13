/* actionlog.js — the avatar's own action memory (speech agent).
 *
 * The avatar must know what it has done: every command it executed (and how that ended: finished / interrupted /
 * failed / refused / still going), its reflexes, what it holds in which of its OWN paws, where its props are, where it
 * looks and which poses it holds — and nothing else. Only logged or observed facts are ever rendered; a planned or
 * in-progress action is never described as finished.
 *
 *   const L = AvatarActionLog.createActionLog({now: () => sessionSeconds, wall: () => Date.now()})
 *   L.push(ev)            one motion onAction event {t, kind, name, src, phase, args, did, reason, id, progress}
 *                         phase: 'queued' | 'start' | 'end' | 'fail' | 'interrupted' (+ 'refused' = fail)
 *   L.command(cmd, res, source)   what the page's runCommand did (res = motion.command() result: {ok,id,did,reason}
 *                         or a bare boolean from the legacy motion.js) — a refusal is logged here as 'failed'
 *   L.observe(state)      motion.state() snapshot (posture, facing, looking, holding{left,right}, props, held, active)
 *   L.pollLegacy(motion)  until the motion agent's onAction lands: diff the legacy motion.js internals
 *                         (state.gestures ids, state.log) into the same events (observed starts/ends, not issued ones)
 *   L.summary({sinceS, maxChars, now}) → compact true English for the voice model (second person: "you waved ...")
 *   L.brief()             one-line current state for tool results ("now: sitting, facing him, holding the red ball in your right paw")
 *   L.items(sinceS)       the merged history (for tests / window.__avatar.log())
 *   L.answer(q)           optional rule-based answer to "what did you just do / what are you holding / did you wave?"
 *   L.reset()             session reset (new session = empty memory)
 *   L.version()           bumps on every meaningful change (the page sends a note only when it moved)
 *
 * Sides: everything here is in the character's OWN terms (left paw = its own left). Screen sides from the rig
 * ('l' = screen-left = its own RIGHT) are converted on the way in.
 * Co-speech gestures (src 'cospeech') and life-level motion are never itemized. Reflexes are summarized as counts.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarActionLog = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ------------------------------------------------------------------ English for actions (2nd person, past + -ing)
  // name → [past, progressive, noun-ish for "started to …" (base)] ; {N} = count phrase, {S} = side phrase, {D} = dir
  const V = {
    wave: ['waved', 'waving', 'wave'], nod: ['nodded', 'nodding', 'nod'], shake_head: ['shook your head', 'shaking your head', 'shake your head'],
    wink: ['winked', 'winking', 'wink'], blink: ['blinked', 'blinking', 'blink'], ear_wiggle: ['wiggled your ears', 'wiggling your ears', 'wiggle your ears'],
    ear_flick: ['flicked an ear', 'flicking an ear', 'flick an ear'], nose_wiggle: ['wiggled your nose', 'wiggling your nose', 'wiggle your nose'],
    tail_wag: ['wagged your tail', 'wagging your tail', 'wag your tail'], clap: ['clapped', 'clapping', 'clap'],
    thumbs_up: ['gave a thumbs up', 'giving a thumbs up', 'give a thumbs up'], shrug: ['shrugged', 'shrugging', 'shrug'],
    bow: ['bowed', 'bowing', 'bow'], point: ['pointed', 'pointing', 'point'], stick_tongue_out: ['stuck your tongue out', 'sticking your tongue out', 'stick your tongue out'],
    look_around: ['looked around', 'looking around', 'look around'], jump: ['jumped', 'jumping', 'jump'], hop: ['hopped', 'hopping', 'hop'],
    bounce: ['bounced', 'bouncing', 'bounce'], raise_paw: ['raised your paw', 'raising your paw', 'raise your paw'],
    raise_both: ['raised both paws', 'raising both paws', 'raise both paws'], cover_eyes: ['covered your eyes', 'covering your eyes', 'cover your eyes'],
    hand_on_heart: ['put a paw on your heart', 'holding a paw on your heart', 'put a paw on your heart'], think: ['made a thinking pose', 'thinking with a paw on your chin', 'make a thinking pose'],
    dance: ['danced', 'dancing', 'dance'], spin: ['spun around', 'spinning around', 'spin around'], twirl: ['twirled', 'twirling', 'twirl'],
    turn: ['turned', 'turning', 'turn'], turn_away: ['turned away', 'turned away', 'turn away'], turn_back: ['turned back to face him', 'turning back to him', 'turn back'],
    turn_around: ['turned around', 'turning around', 'turn around'], backflip: ['did a backflip', 'doing a backflip', 'do a backflip'],
    frontflip: ['did a front flip', 'doing a front flip', 'do a front flip'], flip: ['did a flip', 'doing a flip', 'do a flip'],
    walk: ['walked', 'walking', 'walk'], step: ['stepped', 'stepping', 'step'], sit: ['sat down', 'sitting down', 'sit down'],
    sit_down: ['sat down', 'sitting down', 'sit down'], stand: ['stood up', 'standing up', 'stand up'], stand_up: ['stood up', 'standing up', 'stand up'],
    get_up: ['got up', 'getting up', 'get up'], lie_down: ['lay down', 'lying down', 'lie down'], roll_over: ['rolled over', 'rolling over', 'roll over'],
    cartwheel: ['did a cartwheel', 'doing a cartwheel', 'do a cartwheel'], wiggle: ['wiggled', 'wiggling', 'wiggle'],
    shake_off: ['shook yourself off', 'shaking yourself off', 'shake yourself off'], high_five: ['gave a high five', 'giving a high five', 'give a high five'],
    stretch: ['stretched', 'stretching', 'stretch'], yawn: ['yawned', 'yawning', 'yawn'], laugh: ['laughed', 'laughing', 'laugh'],
    blow_kiss: ['blew a kiss', 'blowing a kiss', 'blow a kiss'], salute: ['saluted', 'saluting', 'salute'], lean_in: ['leaned in', 'leaning in', 'lean in'],
    lean_back: ['leaned back', 'leaning back', 'lean back'], lean_side: ['leaned to the side', 'leaning to the side', 'lean to the side'],
    tilt_head: ['tilted your head', 'tilting your head', 'tilt your head'], head_tilt: ['tilted your head', 'tilting your head', 'tilt your head'],
    sniff: ['sniffed', 'sniffing', 'sniff'], peekaboo: ['played peekaboo', 'playing peekaboo', 'play peekaboo'], quizzical: ['made a puzzled face', 'looking puzzled', 'look puzzled'],
    startle: ['startled', 'startled', 'startle'], startle_hands: ['flinched', 'flinching', 'flinch'], wave_back: ['waved back at him', 'waving back', 'wave back'],
    nod_back: ['nodded back at him', 'nodding back', 'nod back'], look_back: ['looked back at him', 'looking back at him', 'look back'],
    perk: ['perked up', 'perking up', 'perk up'], greet: ['greeted him', 'greeting him', 'greet him'], mirror_smile: ['smiled back', 'smiling back', 'smile back'],
    dab: ['did a dab', 'doing a dab', 'dab'], flex: ['flexed', 'flexing', 'flex'], kiss: ['blew a kiss', 'blowing a kiss', 'blow a kiss'],
  };
  const OBJ_V = {   // object verbs: X = the object phrase
    pick_up: ['picked up X', 'picking up X', 'pick up X'], grab: ['grabbed X', 'grabbing X', 'grab X'], put_down: ['put X down', 'putting X down', 'put X down'],
    put_back: ['put X back', 'putting X back', 'put X back'], place: ['put X P', 'putting X P', 'put X P'], hold_up: ['held X up to show him', 'holding X up to show him', 'hold X up'],
    show: ['showed him X', 'showing him X', 'show him X'], inspect: ['looked closely at X', 'looking closely at X', 'look at X'], look_at: ['looked at X', 'looking at X', 'look at X'],
    turn: ['turned X around', 'turning X around', 'turn X around'], rotate: ['turned X around', 'turning X around', 'turn X around'], flip: ['flipped X over', 'flipping X over', 'flip X over'],
    hand_over: ['passed X to your other paw', 'passing X to your other paw', 'pass X to your other paw'], switch_hands: ['passed X to your other paw', 'passing X to your other paw', 'pass X to your other paw'],
    toss: ['tossed X up and caught it', 'tossing X', 'toss X'], catch: ['caught X', 'catching X', 'catch X'], throw: ['threw X', 'throwing X', 'throw X'],
    drop: ['dropped X', 'dropping X', 'drop X'], shake: ['shook X', 'shaking X', 'shake X'], hug: ['hugged X', 'hugging X', 'hug X'],
    offer: ['held X out to him', 'holding X out to him', 'hold X out to him'], give: ['held X out to him', 'holding X out to him', 'hold X out to him'],
    smell: ['sniffed X', 'sniffing X', 'sniff X'], squeeze: ['squeezed X', 'squeezing X', 'squeeze X'], hold: ['held X', 'holding X', 'hold X'],
    // the muse's desk scene: worn items, furniture, lids, an ongoing activity; G = target phrase
    put_on: ['put on X', 'putting on X', 'put on X'], take_off: ['took off X', 'taking off X', 'take off X'],
    pull: ['pulled X G', 'pulling X G', 'pull X G'], push: ['pushed X G', 'pushing X G', 'push X G'], bring: ['brought X G', 'bringing X G', 'bring X G'],
    slide: ['slid X G', 'sliding X G', 'slide X G'], move: ['moved X G', 'moving X G', 'move X G'],
    sit: ['sat down G', 'sitting down G', 'sit down G'], sit_in: ['sat down in X', 'sitting down in X', 'sit down in X'], sit_at: ['sat down at X', 'sitting down at X', 'sit down at X'],
    stand_up: ['stood up', 'standing up', 'stand up'], open: ['opened X', 'opening X', 'open X'], close: ['closed X', 'closing X', 'close X'],
    use: ['used X', 'using X', 'use X'], work_on: ['worked on X', 'working on X', 'work on X'], type_on: ['typed on X', 'typing on X', 'type on X'],
    type: ['typed on X', 'typing on X', 'type on X'], turn_back: ['turned back to face him', 'turning back to him', 'turn back to him'],
  };
  // open-ended activities: stopping them is how they end (never "stopped before finishing")
  const OPEN_ENDED = new Set(['use', 'work_on', 'type_on', 'type', 'dance', 'hold', 'look_at', 'inspect', 'hug', 'walk']);
  const TARGET_W = {closer: 'closer', back: 'back', away: 'away', me: 'over to him', user: 'over to him', here: 'over', over: 'over', in: 'in'};
  function targetPhrase(t, verb) {
    if (t == null || t === '') return /^(pull|bring)$/.test(verb) ? 'over' : /^push$/.test(verb) ? 'away' : '';
    if (typeof t === 'string') return TARGET_W[t.toLowerCase()] || ('to ' + (/^(the|a|an) /i.test(t) ? t : 'the ' + t.replace(/_/g, ' ')));
    if (t.place || t.object || t.name || t.id) return 'to ' + objPhrase(t.place || t.object || t);
    return '';
  }
  const EXPR_ADJ = {happy: 'happy', excited: 'excited', sad: 'sad', surprised: 'surprised', curious: 'curious', thinking: 'thinking', silly: 'silly',
    angry: 'angry', scared: 'scared', sleepy: 'sleepy', love: 'loving', proud: 'proud', embarrassed: 'embarrassed', confused: 'confused',
    smug: 'smug', worried: 'worried', shy: 'shy', disgusted: 'disgusted', laughing: 'laughing', serious: 'serious', calm: 'calm', smile: 'smiling'};
  const LOOK_W = {object: 'at what you are working on', screen: 'at the screen', desk: 'at the desk', laptop: 'at the laptop', user: 'at him', me: 'at him', camera: 'at him', left: 'to your left', right: 'to your right', up: 'up', down: 'down', away: 'away',
    screen_left: 'to your right', screen_right: 'to your left', around: 'around', forward: 'forward', back: 'behind you'};
  const ASKED = new Set(['tool', 'matcher', 'ui', 'voice', 'voice-early', 'caption', 'model', 'cmd', 'test']);
  const SELF = new Set(['reflex', 'life', 'auto', 'fidget', 'listen', 'speech', 'boredom', 'think']);
  const NOT_ITEMIZED = new Set(['cospeech', 'speech', 'life', 'listen', 'think']);   // think = the filler while a move is planned
  const COUNTW = n => n === 1 ? 'once' : n === 2 ? 'twice' : n + ' times';
  const NUMW = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  const norm = s => String(s == null ? '' : s).toLowerCase().trim().replace(/[\s-]+/g, '_');
  const fmtClock = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  const agoW = s => { s = Math.max(0, Math.round(s)); return s < 2 ? 'just now' : s < 90 ? s + ' s ago' : Math.round(s / 60) + ' min ago'; };

  // own side from any side notation ('l' rig side = screen-left = its own RIGHT)
  function ownSide(side, frame) {
    const s = norm(side);
    if (!s) return null;
    if (s === 'both' || s === 'paws') return 'both';
    if (s === 'l' || s === 'screen_left') return 'right';
    if (s === 'r' || s === 'screen_right') return 'left';
    if (/left/.test(s)) return 'left';
    if (/right/.test(s)) return 'right';
    return null;
  }
  function dirPhrase(dir) {
    const d = norm(dir);
    if (!d) return '';
    return {left: 'to your left', own_left: 'to your left', right: 'to your right', own_right: 'to your right', screen_left: 'to your right',
      screen_right: 'to your left', forward: 'forward', toward: 'toward him', back: 'backward', away: 'away from him', cw: 'clockwise',
      ccw: 'counterclockwise', up: 'up', down: 'down', l: 'to your right', r: 'to your left'}[d] || ('(' + d.replace(/_/g, ' ') + ')');
  }
  function objPhrase(o, fallback) {
    if (!o) return fallback || 'it';
    if (typeof o === 'string') return /^(the|a|an|your|my) /i.test(o) ? o.replace(/^my /i, 'your ') : 'the ' + o.replace(/_/g, ' ');
    if (o.name) return objPhrase(String(o.name));
    const bits = [o.color, o.category || o.id].filter(Boolean).join(' ');
    return bits ? 'the ' + bits.replace(/_/g, ' ') : (fallback || 'it');
  }
  // a refusal reason is usually first person ("my paws are full") — the log speaks to the avatar as "you"
  const youify = r => r == null ? null : String(r).replace(/\bI'm\b/g, "you're").replace(/\bI am\b/g, 'you are').replace(/\bI've\b/g, "you've").replace(/\bI was\b/g, 'you were')
    .replace(/\bI\b/g, 'you').replace(/\bmy\b/gi, 'your').replace(/\bme\b/g, 'you').replace(/\bmyself\b/g, 'yourself');
  const SIDED = new Set(['wave', 'raise_paw', 'point', 'thumbs_up', 'wink', 'ear_flick', 'hand_on_heart', 'think', 'high_five', 'salute']);

  // ------------------------------------------------------------------ the log
  function createActionLog(o = {}) {
    const perf = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const T0 = perf();
    const now = o.now || (() => (perf() - T0) / 1000);
    const L = {items: [], seq: 0, ver: 0, state: null, stateT: -1, legacy: {seen: new Map(), logN: 0}, reflex: [], firstT: null,
      byId: new Map(), held: new Map(), lastChange: -1, props: null, speechOnly: 0};
    const touch = () => { L.ver++; L.lastChange = now(); };

    function kindOf(ev) {
      const k = norm(ev.kind || ev.action || '');
      if (k === 'command' || k === 'cmd') return norm(ev.action || (ev.args && ev.args.action) || 'gesture');
      if (k === 'kf') return 'perform';                    // a planned (keyframed) action = the perform item
      return k || 'gesture';
    }
    function whoOf(src) { const s = norm(src); return ASKED.has(s) ? 'asked' : SELF.has(s) ? 'self' : s === 'tool_self' ? 'self' : 'asked'; }
    function find(ev, kind, name, side) {
      if (ev.id != null && L.byId.has(String(ev.id))) return L.byId.get(String(ev.id));
      for (let i = L.items.length - 1; i >= 0 && i >= L.items.length - 12; i--) {
        const it = L.items[i];
        if (it.kind === kind && it.name === name && (it.status === 'doing' || it.status === 'pending') && (!side || !it.side || it.side === side)) return it;
      }
      return null;
    }
    function mkItem(ev, kind, name) {
      const a = ev.args || {};
      const it = {n: ++L.seq, id: ev.id != null ? String(ev.id) : null, kind, name, src: norm(ev.src || 'tool'), who: whoOf(ev.src || 'tool'),
        args: a, side: ownSide(a.side || ev.side), dir: a.dir || a.target || null, count: +a.count || +a.repeat || null,
        obj: a.object || ev.object || null, place: a.place || null, did: ev.did || null, reason: null,
        t0: +ev.t >= 0 ? +ev.t : now(), t1: null, status: 'pending', hold: (a.hold === true || a.hold === 'true') && kind !== 'kf' && kind !== 'keyframes', progress: null, merged: 1};   // (a keyframe job's hold = wait for the cue)
      if (kind === 'object' && !it.obj && ev.prop) it.obj = ev.prop;
      L.items.push(it); if (it.id) L.byId.set(it.id, it);
      if (L.items.length > 400) { const gone = L.items.splice(0, 100); for (const g of gone) if (g.id) L.byId.delete(g.id); }
      return it;
    }
    // ---- one onAction event (the motion engine: rules-layer gestures carry no id and fire 'start' inside command();
    // engine jobs carry ids; object_event / move_phase are the observed sub-steps; an end 'did' lists only what ran)
    function lastDoing(pred) { for (let i = L.items.length - 1; i >= 0 && i >= L.items.length - 30; i--) { const it = L.items[i]; if ((it.status === 'doing' || it.status === 'pending') && pred(it)) return it; } return null; }
    function push(ev) {
      if (!ev || typeof ev !== 'object') return;
      const src = norm(ev.src || '');
      const kind = kindOf(ev), name = norm(ev.name || (ev.args && (ev.args.name || ev.args.verb || ev.args.target)) || kind);
      const phase = norm(ev.phase || 'start');
      const t = +ev.t >= 0 ? +ev.t : now();
      if (L.firstT === null) L.firstT = t;
      if (src === 'cospeech') { L.speechOnly++; return; }          // never itemized
      if (src === 'life' || src === 'think' || kind === 'config') return;   // the thinking filler is never an action
      // reflexes: one count each (the engine reports them as kind 'reflex' end; their gesture echo is skipped); fidgets on their end
      if (kind === 'reflex' || src === 'reflex' || src === 'fidget' || src === 'listen' || (src === 'auto' && (kind === 'gesture' || kind === 'expression'))) {
        const count = kind === 'reflex' ? true : (src === 'fidget' || src === 'listen' || src === 'auto') && phase !== 'start' ? true : src === 'reflex' && kind !== 'gesture' && kind !== 'expression';
        const known = ev.id != null && L.reflex.some(r => r.id === ev.id);
        if (count && !known) {
          L.reflex.push({t, name, src, id: ev.id ?? null, did: ev.did || null, side: ownSide((ev.args || {}).side || ev.side)});
          if (L.reflex.length > 300) L.reflex.splice(0, 100);
          touch();
        }
        return;
      }
      if (kind === 'move_phase' || kind === 'object_event') {
        // an observed step of the running job (by id, else the newest running engine job); alone = its own observed fact
        const job = (ev.id != null && L.byId.get(String(ev.id))) || lastDoing(it => it.engine);
        const d = ev.did || name.replace(/_/g, ' ');
        if (job && (job.status === 'doing' || job.status === 'pending')) { (job.steps = job.steps || []).push(d); touch(); return; }
        const it = mkItem(Object.assign({}, ev, {src: ev.src && ev.src !== 'auto' ? ev.src : 'auto'}), 'event', name);
        it.status = 'done'; it.t1 = t; it.endDid = d; it.who = 'self'; touch(); return;
      }
      if (kind === 'activity') {
        if (name === 'typing' || name === 'work' || name === 'working') {
          const cur = lastDoing(it => it.kind === 'activity');
          if (phase === 'start') { if (cur) { if (/back to/.test(ev.did || '')) cur.paused = false; return; } const it = mkItem(ev, 'activity', name); it.status = 'doing'; it.did = ev.did; touch(); return; }
          if (cur) { cur.status = 'done'; cur.t1 = t; cur.endDid = ev.did || null; touch(); }
          return;
        }
        // turn_back while seated: the activity pauses, it faces the user
        const cur = lastDoing(it => it.kind === 'activity'); if (cur && name === 'turn_back') cur.paused = true;
        const job = lastDoing(it => it.engine && /turn_back/.test(it.name));
        if (job) { (job.steps = job.steps || []).push(ev.did || 'turned back to him'); touch(); return; }
        const it = mkItem(ev, 'event', name); it.status = 'done'; it.t1 = t; it.endDid = ev.did || null; touch(); return;
      }
      if (kind === 'release' || kind === 'stop') {
        // relax / stop: every held pose / look ends (objects stay where they are — "relax" never drops what it holds)
        let n = 0;
        for (const it of L.items) if (it.hold && it.status !== 'failed' && !it.released) { it.released = t; n++; if (it.status === 'doing') it.status = 'done'; }
        if (ev.id != null && L.byId.has(String(ev.id))) return;
        if (phase === 'start' || phase === 'end') { const it = mkItem(ev, 'release', 'release'); it.status = 'done'; it.t1 = t; it.relN = n; touch(); }
        return;
      }
      if (kind === 'expression' && phase !== 'start' && phase !== 'fail') return;   // its end is just the face relaxing
      const side = ownSide((ev.args || {}).side || ev.side);
      let it = ev.id != null && L.byId.has(String(ev.id)) ? L.byId.get(String(ev.id)) : null;
      if (!it && phase !== 'start' && phase !== 'queued' && phase !== 'request') it = find(ev, kind, name, side);
      if (phase === 'queued' || phase === 'request') {
        if (!it) it = find(ev, kind, name, side);
        if (it && it.status !== 'pending') {   // already observed: never downgrade (but keep the count that was asked for)
          if (ev.id != null && !it.id) { it.id = String(ev.id); L.byId.set(it.id, it); }
          const ea = ev.args || {}; for (const k of ['count', 'repeat']) if (+ea[k] > 0 && it.args && it.args[k] == null) { it.args[k] = +ea[k]; if (!it.count) it.count = +ea[k]; }
          return;
        }
        if (!it) it = mkItem(ev, kind, name);
        it.status = 'pending'; touch(); return;
      }
      if (!it) it = mkItem(ev, kind, name);
      if (ev.id != null && /^(move|object|kf|keyframes)$/.test(kind)) it.engine = true;   // a planner keyframe job reports like an engine job
      if (ev.did && phase === 'start') it.did = ev.did;
      if (ev.progress != null) it.progress = +ev.progress;
      if (ev.args && ev.args.done_count != null) it.progress = +ev.args.done_count;
      if (phase === 'start') {
        if (it.status !== 'doing') { it.status = 'doing'; it.t0 = t; }
        // a look / pose / face is in place at once (a held one stays until released)
        if (/^(look|pose|expression|gesture_held)$/.test(kind)) { it.status = 'done'; it.t1 = t; }
      }
      else if (phase === 'end') { it.status = 'done'; it.t1 = t; if (ev.did) it.endDid = ev.did; }
      else if (phase === 'interrupted' || phase === 'cancel' || phase === 'cancelled') {
        it.status = 'interrupted'; it.t1 = t; it.reason = ev.reason || it.reason; if (ev.did) it.endDid = ev.did;
        // the engine says "spun 3 turns … — stopped early: 2 turns done": the count that really happened is the "done" one
        const m = /(\d+|no|one|two|three|four|five|six) (?:full )?(turns?|spins?|jumps?|flips?|hops?|steps?|cartwheels?|rolls?|bows?|times?) done\b/.exec(ev.did || '');
        if (m) { const w = {no: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6}; it.progress = /\d/.test(m[1]) ? +m[1] : w[m[1]]; it.endDid = null; }
        // (motion engine) the engine now says "spun 2 of the 3 turns, then stopped" / "… but stopped before one was done"
        const m2 = /\b(\d+) of the (\d+) (?:\w+ )?(turns?|spins?|jumps?|flips?|backflips?|frontflips?|hops?|steps?|cartwheels?|rolls?|bows?|times?)\b/.exec(ev.did || '');
        if (!m && m2) { it.progress = +m2[1]; it.endDid = null; }
        else if (!m && /stopped before one was done/.test(ev.did || '')) { it.progress = 0; it.endDid = null; }
      }
      else if (phase === 'fail' || phase === 'refused' || phase === 'failed') { it.status = 'failed'; it.t1 = t; it.reason = youify(ev.reason) || 'it could not be done'; if (ev.did) it.endDid = ev.did; }
      touch();
    }
    // ---- the page's command path (what was asked and what motion.command() answered)
    function command(cmd, res, source, extra) {
      if (!cmd || typeof cmd !== 'object') return null;
      const ok = res === true || (res && typeof res === 'object' && res.ok !== false && res.ok !== undefined) || (res && res.ok === true);
      const a = Object.assign({}, cmd); delete a.action; delete a.poses;   // (a keyframe job's poses are not a memory)
      const ev = {t: now(), kind: cmd.action, name: cmd.name || cmd.verb || cmd.target || cmd.action, src: source || 'tool', args: a,
        id: res && typeof res === 'object' && res.id != null ? res.id : null, did: res && res.did, reason: res && res.reason};
      if (cmd.action === 'object') ev.name = cmd.verb || 'object';
      if (cmd.action === 'release') { push(Object.assign(ev, {phase: 'start'})); return null; }
      if (!ok) {
        // the engine may already have reported this refusal through onAction: one failed item, not two
        const last = L.items[L.items.length - 1], k = kindOf(ev), nm = norm(ev.name);
        if (last && last.status === 'failed' && last.kind === k && last.name === nm && now() - last.t0 < 1) return last;
        ev.phase = 'fail'; ev.reason = (res && res.reason) || (extra && extra.reason) || "you couldn't do that"; push(ev); return null;
      }
      // accepted: the observed start comes from onAction (often already, inside command()); until then only pending
      if (ev.id != null && L.byId.has(String(ev.id))) {
        const it = L.byId.get(String(ev.id));
        if (res && res.queued && it.status === 'doing' && /\(next\)/.test(it.did || '')) it.status = 'pending';   // after the current job
        return it;
      }
      if (res && res.already) return null;                     // nothing to do: the engine's own 'end' says so
      ev.phase = 'queued'; push(ev);
      return L.items[L.items.length - 1];
    }
    // ---- motion.state() snapshot (the truth for "now"); props moving between paws are logged as observed facts
    function observe(st) {
      if (!st || typeof st !== 'object') return;
      const prev = L.state, t = now();
      // (answer() reads the snapshot: worn / seated / lid / activity / turned-back changes are changes too)
      const sig = JSON.stringify([st.posture, st.facing, st.looking, st.holding, (st.props || []).map(p => [p.id, p.where, p.open]), st.held,
        st.activity, st.turnedBack, st.wearing, st.open, st.sittingOn, st.atDesk, st.seat, st.station, st.teaching]);   // + station
      if (prev && prev.sig === sig) { L.state.t = t; return; }
      L.state = Object.assign({}, st, {sig, t}); L.stateT = t;
      if (st.props) L.props = st.props;
      touch();
    }
    // ---- legacy motion.js (no onAction yet): observe gestures by id in its internal state; never from intent alone
    function pollLegacy(m) {
      const S = m && m.state && typeof m.state === 'object' ? m.state : null;
      if (!S) return;
      const seen = L.legacy.seen, live = new Set(), t = now();
      for (const G of S.gestures || []) {
        if (!G || G.src === 'life' || G.src === 'speech' || G.src === 'cospeech' || G.src === 'think' || G.isExpr && G.src !== 'cmd') continue;
        live.add(G.id);
        if (seen.has(G.id)) { const r = seen.get(G.id); if (G.rel && !r.rel) r.rel = S.t; continue; }
        if (G.t0 > S.t + 1e-6) continue;                                        // scheduled in the future: not started yet
        const name = G.isExpr ? 'expression' : G.name, kind = G.isExpr ? 'expression' : 'gesture';
        const rec = {t0: G.t0, dur: G.dur, rel: G.rel ? S.t : null, src: G.src, name, kind, expr: G.isExpr ? String(G.name).replace(/^expr:/, '') : null};
        seen.set(G.id, rec);
        const src = G.src === 'cmd' ? (L.legacy.lastSrc && t - L.legacy.lastSrcT < 1.0 ? L.legacy.lastSrc : 'tool') : G.src;
        rec.evSrc = src;
        const args = {side: G.side, count: G.g && G.g.cycles ? Math.max(1, Math.round(G.g.cycles / (G.name === 'wave' ? 2 : 1))) : null, hold: !!(G.g && G.g.hold)};
        if (rec.expr) { args.name = rec.expr; }
        push({t, kind, name: rec.expr || name, src, phase: 'start', id: 'g' + G.id, args});
      }
      for (const [id, r] of seen) {
        if (live.has(id)) continue;
        seen.delete(id);
        const ran = S.t - r.t0, natural = Number.isFinite(r.dur) && ran >= r.dur - 0.05 && !(r.rel && r.rel < r.t0 + r.dur - 0.3);
        push({t, kind: r.kind, name: r.expr || r.name, src: r.evSrc, phase: natural || r.kind === 'expression' ? 'end' : 'interrupted', id: 'g' + id,
          reason: natural ? null : 'stopped early'});
      }
      // looks / holds come from commands; reflex events from the motion log (nod back)
      const lg = S.log || [];
      if (L.legacy.logN > lg.length) L.legacy.logN = 0;
      for (let i = L.legacy.logN; i < lg.length; i++) {
        const e = lg[i];
        if (e.nodback) push({t, kind: 'reflex', name: 'nod_back', src: 'reflex', phase: 'start'});
      }
      L.legacy.logN = lg.length;
    }
    function noteSource(src) { L.legacy.lastSrc = src; L.legacy.lastSrcT = now(); }

    // ------------------------------------------------------------------ English
    function phraseOf(it, form /* 0 past, 1 progressive, 2 base */) {
      const a = it.args || {};
      if (it.kind === 'kf' || it.kind === 'keyframes') {   // what the user asked for, in the planner's request words
        const d = String(a.description || 'the move he asked for').replace(/[.\s]+$/, '');
        return ['did the move he asked for (' + d + ')', 'doing the move he asked for (' + d + ')', 'do the move he asked for (' + d + ')'][form];
      }
      if (it.kind === 'perform') {                         // perform_action → the planner's move, in the user's words
        const dsc = a.description ? '"' + String(a.description).slice(0, 120) + '"' : 'a move he asked for';
        return (form === 0 ? 'did what he asked: ' : form === 1 ? 'doing what he asked: ' : 'do what he asked: ') + dsc;
      }
      if (it.kind === 'object') {
        const verb = norm(it.name === 'object' ? a.verb : it.name);
        const tpl = (OBJ_V[verb] || [verb.replace(/_/g, ' ') + 'ed X', verb.replace(/_/g, ' ') + 'ing X', verb.replace(/_/g, ' ') + ' X'])[form];
        const place = a.place ? (typeof a.place === 'string' ? 'on ' + objPhrase(a.place) : 'on ' + objPhrase(a.place)) : 'down';
        let s = tpl.replace('X', objPhrase(it.obj)).replace('P', place).replace('G', targetPhrase(a.target, verb)).replace(/\s+$/, '').replace(/\s{2,}/g, ' ');
        const side = it.side || ownSide(a.side);
        if (side && side !== 'both' && /^(pick_up|grab|hold|catch|show|offer|give)$/.test(verb)) s += ' with your ' + side + ' paw';
        if (side === 'both' && /pick|grab|hold/.test(verb)) s += ' with both paws';
        if (it.count > 1 && /toss|shake|turn|flip/.test(verb)) s += ' ' + COUNTW(it.count);
        return s;
      }
      if (it.kind === 'expression') {
        const adj = EXPR_ADJ[norm(a.name || it.name)] || String(a.name || it.name).replace(/_/g, ' ');
        return [form === 0 ? 'made a ' + adj + ' face' : form === 1 ? 'making a ' + adj + ' face' : 'make a ' + adj + ' face'][0];
      }
      if (it.kind === 'look') {
        const w = LOOK_W[norm(a.target || it.name)] || ('at ' + String(a.target || it.name).replace(/_/g, ' '));
        return (form === 0 ? 'looked ' : form === 1 ? 'looking ' : 'look ') + w;
      }
      if (it.kind === 'pose') {
        const parts = Object.keys(a.targets || {}).map(k => /hand_l|_l$/.test(k) ? 'right paw' : /hand_r|_r$/.test(k) ? 'left paw' : k.split('_')[0]);
        const u = [...new Set(parts)].slice(0, 3).join(' and ') || 'body';
        return (form === 0 ? 'moved your ' : form === 1 ? 'holding your ' : 'move your ') + u + (form === 1 ? ' in a pose' : ' into a pose');
      }
      if (it.kind === 'hold') return form === 0 ? 'froze in place' : form === 1 ? 'holding still in place' : 'freeze';
      if (it.kind === 'release') return form === 0 ? 'relaxed' : form === 1 ? 'relaxing' : 'relax';
      const name = norm(it.name);
      const v = V[name] || [name.replace(/_/g, ' ') + (/e$/.test(name) ? 'd' : 'ed'), name.replace(/_/g, ' ') + 'ing', name.replace(/_/g, ' ')];
      let s = v[form];
      const side = it.side;
      if (side && SIDED.has(name)) s += side === 'both' ? ' with both paws' : ' with your ' + side + ' paw';
      if (name === 'point' && a.target) s += ' ' + (LOOK_W[norm(a.target)] || 'at ' + String(a.target).replace(/_/g, ' '));
      const dp = it.kind === 'move' || /spin|turn|twirl|walk|step|roll|cartwheel|flip|jump|hop/.test(name) ? dirPhrase(a.dir) : '';
      if (dp && !/^(spin|twirl)$/.test(name)) s += ' ' + dp;
      const n = +a.count || +a.repeat || 0;
      if (n > 1) s += ' ' + COUNTW(n);
      if (dp && /^(spin|twirl)$/.test(name)) s += ' ' + dp;
      if (a.style) s += ' (' + String(a.style).slice(0, 20) + ')';
      return s;
    }
    function whoW(it) { return it.who === 'self' ? 'on your own' : it.src === 'tool' || it.src === 'model' || it.src === 'voice' || it.src === 'voice-early' ? 'your tool call' : 'asked'; }
    // the engine's own words ("walked over to the chair, pulled the chair up to the desk, …") in the 2nd person
    const eng = d => youify(String(d).replace(/\bits\b/g, 'your').replace(/\bitself\b/g, 'yourself').replace(/ \(still at it\)/, ', and are still at it')
      .replace(/\bher\b|\bhim\b/g, 'him'));
    function line(it, tNow) {
      const when = fmtClock(it.t0 - (L.firstT0 || 0)) + ', ' + agoW(tNow - it.t0);
      const w = whoW(it);
      if (it.kind === 'event') return `${when}: ${eng(it.endDid || it.name.replace(/_/g, ' '))} (on your own)`;
      if (it.kind === 'activity') {
        if (it.status === 'doing') return `${when}: ${eng(it.did || 'working')} (${w}) — STILL GOING` + (it.paused ? ', paused while you turned back to him' : '');
        return `${when}: ${eng(it.did || 'worked')}`.replace(/typing on/, 'typed on').replace(/, eyes on the screen/, ', eyes on the screen,') + ' — ' + eng(it.endDid || 'then stopped');
      }
      if (it.engine && it.endDid && /^\w+ing\b/.test(it.endDid) && !/^nothing/.test(it.endDid)) it.endDid = null;   // a progressive text is a plan, not a record
      if (it.engine && it.endDid && (it.status === 'done' || it.status === 'interrupted' || it.status === 'failed')) {
        if (it.status === 'done') return /^nothing to do/.test(it.endDid) ? `${when}: asked to ${phraseOf(it, 2)} — nothing to do, it already was that way (${w})` : `${when}: ${eng(it.endDid)} (${w})`;
        if (it.status === 'interrupted') return `${when}: started to ${phraseOf(it, 2)} (${w}) but it was stopped; what actually got done: ${eng(it.endDid)}`;
        return `${when}: tried to ${phraseOf(it, 2)} (${w}) but couldn't — ${String(it.reason || '').slice(0, 80)}; what got done before that: ${eng(it.endDid) || 'nothing'}`;
      }
      if (it.engine && it.status === 'doing') return `${when}: ${phraseOf(it, 1)} (${w}) — STILL GOING, not finished yet` + (it.steps && it.steps.length ? '; done so far: ' + it.steps.map(eng).join(', ') : '; no step finished yet');
      const merged = it.merged > 1 ? (it.kind === 'gesture' || it.kind === 'move' ? ' — ' + COUNTW(it.mergedN || it.merged) + ' in a row' : '') : '';
      switch (it.status) {
        case 'done': return `${when}: ${phraseOf(it, 0)}${merged} (${w})` + (it.hold && !it.released ? ' — still holding it' : it.hold && it.released ? ' — then relaxed' : '');
        case 'doing': return `${when}: ${phraseOf(it, 1)} (${w}) — STILL GOING, not finished yet`;
        case 'pending': return `planned next, NOT started yet: ${phraseOf(it, 2)} (${w})`;
        case 'interrupted': {
          if (it.reason) it.reason = youify(it.reason);
          if (OPEN_ENDED.has(norm(it.kind === 'object' && it.name === 'object' ? it.args.verb : it.name)))
            return `${when}: ${phraseOf(it, 0)} for a while (${w}), then stopped` + (it.reason && !/stopped early|replaced/.test(it.reason) ? ' (' + it.reason + ')' : '');
          const of = it.args.count || it.count || null;
          if (of && it.progress != null && it.progress >= of) return `${when}: ${phraseOf(it, 0)} (${w}) — all ${of} done, then the stop came as you settled`;
          const n = it.progress != null && (of > 1 || it.progress > 0) ? (of ? ` after ${it.progress} of ${of}` : ` after ${it.progress}`) + (it.progress === 0 ? ' (none finished)' : '') : '';
          return `${when}: started to ${phraseOf(it, 2)} (${w}) but it was stopped before finishing${n}` + (it.reason && !/stopped early/.test(it.reason) ? ' (' + it.reason + ')' : '');
        }
        case 'failed': return `${when}: tried to ${phraseOf(it, 2)} (${w}) but couldn't — ${String(it.reason || 'it was not possible').slice(0, 80)}; nothing happened`;
      }
      return '';
    }
    // merge consecutive identical finished actions (repeats, cycles) into one line with a true count
    function merged(list) {
      const out = [];
      for (const it of list) {
        const p = out[out.length - 1];
        const key = it => [it.kind, it.name, it.side, JSON.stringify(it.obj || null), it.who, it.status, it.args && it.args.dir, it.args && it.args.target, it.args && it.args.description].join('|');
        if (p && it.status === 'done' && p.status === 'done' && key(p) === key(it) && it.t0 - (p.t1 || p.t0) < 25 && !it.hold && !p.hold) {
          const c = (+it.args.count || +it.args.repeat || 1);
          p.mergedN = (p.mergedN || (+p.args.count || +p.args.repeat || 1)) + c; p.merged++; p.t1 = it.t1;
          continue;
        }
        out.push(Object.assign({}, it, {args: Object.assign({}, it.args)}));
      }
      for (const m of out) if (m.merged > 1) { m.args.count = null; m.args.repeat = null; m.count = null; }
      return out;
    }
    function reflexSummary(sinceT) {
      const c = new Map(), dids = new Map();
      for (const r of L.reflex) if (r.t >= sinceT) { const k = norm(r.name); c.set(k, (c.get(k) || 0) + 1); if (r.did && !dids.has(k)) dids.set(k, r.did); }
      if (!c.size) return '';
      const parts = [];
      for (const [k, n] of c) {
        const v = V[k] || V[k.replace(/^reflex_/, '')];
        const d = v ? v[0] : dids.has(k) ? eng(dids.get(k)).replace(/\s*\(.*\)$/, '') : k.replace(/_/g, ' ');
        parts.push(d + ' ' + COUNTW(n));
      }
      return parts.join('; ');
    }
    function stateLine() {
      const st = L.state;
      const bits = [];
      if (st) {
        const post = {seated: 'sitting', sitting: 'sitting', standing: 'standing', lying: 'lying down', airborne: 'in the air', walking: 'walking', moving: 'moving around', dancing: 'dancing', spinning: 'spinning'}[norm(st.posture)] || (st.posture ? String(st.posture) : null);
        const sitOn = st.sittingOn || st.sitting_on;
        if (sitOn) bits.push('sitting on the ' + String(sitOn).replace(/_/g, ' ') + (st.atDesk ? ' at the desk' : ''));
        else if (post) bits.push(post);
        const face = norm(st.facing);
        if (st.turnedBack) bits.push('turned back toward him');
        else if (face) bits.push(face === 'camera' || face === 'user' ? 'facing him' : face === 'away' ? (st.atDesk ? 'facing the desk, away from him' : 'turned away from him') : face === 'tumbling' ? 'tumbling' : 'turned ' + (dirPhrase(face) || face));
        if (st.station) bits.push(String(st.station));                          // teach: "at the board"
        if (st.teaching) bits.push('busy right now: ' + String(st.teaching));
        const lk = norm(st.looking);
        if (lk) bits.push(lk === 'user' || lk === 'camera' ? 'looking at him' : lk === 'object' && st.activity && !st.turnedBack ? 'looking at the screen' : 'looking ' + (LOOK_W[lk] || lk.replace(/_/g, ' ')));
        const h = st.holding || {};
        const nm = id => { const p = (st.props || L.props || []).find(p => p.id === id); return p ? objPhrase(p.name || p) : objPhrase(String(id)); };
        if (h.left && h.right && h.left === h.right) bits.push('holding ' + nm(h.left) + ' in both paws');
        else {
          if (h.right) bits.push('holding ' + nm(h.right) + ' in your right paw');
          if (h.left) bits.push('holding ' + nm(h.left) + ' in your left paw');
          if (!h.left && !h.right) bits.push('both paws empty (holding nothing)');
          else if (!h.left) bits.push('left paw empty'); else if (!h.right) bits.push('right paw empty');
        }
        const worn = (st.props || []).filter(p => /head|worn|wearing/.test(String(p.where || '')) && !/balanc/.test(String(p.where || ''))).map(p => objPhrase(p.name || p));   // (balanced on the head is not worn)
        for (const w of [].concat(st.wearing || [])) { const nmw = typeof w === 'string' ? nm(w) : objPhrase(w.name || w); if (!worn.includes(nmw)) worn.push(nmw); }
        if (worn.length) bits.push('wearing ' + worn.join(' and '));
        const seat = st.seat || st.at;
        if (!sitOn && seat && /seat|sit/.test(String(st.posture))) bits.push('at ' + objPhrase(typeof seat === 'string' ? nm(seat) : seat.name || seat));
        const act = st.activity;
        if (act) bits.push('busy right now: ' + (typeof act === 'string' ? act.replace(/_/g, ' ') : String(act.name || act.verb || '').replace(/_/g, ' ') + (act.object ? ' ' + objPhrase(act.object) : '')));
        const props = (st.props || []).filter(p => (!/paw|hand|head|worn|wearing/.test(String(p.where || '')) || /balanc/.test(String(p.where || ''))) && !p.held);
        const openIds = Array.isArray(st.open) ? st.open : null;
        const lid = (st.props || []).filter(p => p.open === true || p.open === false || /^(open|closed)$/.test(String(p.state || '')) ||
          (openIds && (openIds.includes(p.id) || /laptop|box|lid|case/.test(String(p.name || p.id)))) ||
          (p.s !== undefined && Number.isFinite(+p.s) && (p.openable || /laptop|box|book|lid|door|drawer|chest|jar|case/.test(String(p.name || p.id)))));
        for (const p of lid) { const open = p.open === true || p.state === 'open' || (openIds ? openIds.includes(p.id) : p.open === undefined && +p.s >= 0.5); bits.push(objPhrase(p.name || p) + ' is ' + (open ? 'open' : 'closed')); }
        if (props.length) {
          const byWhere = new Map();
          for (const p of props) { const wh = String(p.where || 'home').replace(/_top$/, '').replace(/_/g, ' '), w = wh === 'home' ? 'in its usual spot' : wh === 'moved' ? 'moved from its usual spot' : /^(next|in|by|on|under|at|beside|near|behind|over|up|back|against|away|balanced)\b/.test(wh) ? wh.replace(/\bits\b/g, 'your') : 'on the ' + wh.replace(/^the /, ''); if (!byWhere.has(w)) byWhere.set(w, []); byWhere.get(w).push(objPhrase(p.name || p)); }
          bits.push([...byWhere].map(([w, ns]) => ns.join(', ') + ' ' + (ns.length > 1 ? 'are ' + w.replace('in its usual spot', 'in their usual spots') : 'is ' + w)).join('; '));
        }
        if (Array.isArray(st.active) && st.active.length) bits.push('in progress: ' + st.active.map(a => String(a.name || a.kind).replace(/_/g, ' ') + (a.object ? ' ' + objPhrase(a.object) : '')).slice(0, 3).join(', '));
      }
      const held = L.items.filter(it => it.hold && !it.released && (it.status === 'done' || it.status === 'doing') && it.kind !== 'object');
      if (held.length) bits.push('holding a pose until told to relax: ' + held.slice(-3).map(it => phraseOf(it, 1)).join(', '));
      return bits.join(', ');
    }

    // the muse has hands (mittens), the animals have paws: o.limb = 'hand' | 'paw'
    const limbs = str => o.limb === 'hand' ? String(str).replace(/\bpaws\b/g, 'hands').replace(/\bpaw\b/g, 'hand') : str;
    function summary(opt = {}) { return limbs(summary0(opt)); }
    function summary0(opt = {}) {
      const tNow = opt.now != null ? opt.now : now();
      const maxChars = opt.maxChars || 900;
      const sinceT = opt.sinceS != null ? tNow - opt.sinceS : -Infinity;
      const its = L.items.filter(it => it.kind !== 'release' || it.relN > 0).filter(it => it.t0 >= sinceT && !NOT_ITEMIZED.has(it.src));
      const list = merged(its);
      const head = `[Your body's action log, ${fmtClock(tNow - L.firstT0)} into the call. The complete record of what you did on screen — if something is not in it, you did NOT do it.]`;
      const nowL = 'Right now: ' + (stateLine() || 'sitting, facing him, both paws empty') + '.';
      if (!list.length && !L.reflex.some(r => r.t >= sinceT)) {
        return (head + ' Nothing yet: you have not moved on purpose, gestured, or picked anything up this call. ' + nowL).slice(0, maxChars + 200);
      }
      const rs = reflexSummary(sinceT);
      const tail = (rs ? ' Small things your body did by itself (idle fidgets and reflexes — not actions; mention them only if he asks about them): ' + rs + '.' : '') + ' ' + nowL;
      let lines = list.map(it => line(it, tNow));
      // compact: keep the newest lines; older ones collapse into counts per action (still the whole truth)
      let older = [];
      const budget = maxChars - head.length - tail.length - 20;
      while (lines.length > 1 && lines.join('; ').length > budget) { older.push(list[list.length - lines.length]); lines.shift(); }
      let olderS = '';
      if (older.length) {
        const c = new Map();
        for (const it of older) {
          const k = (it.status === 'done' ? '' : it.status === 'failed' ? 'tried (failed) to ' : it.status === 'interrupted' ? 'started (stopped early) to ' : '') + (it.status === 'done' ? phraseOf(Object.assign({}, it, {args: Object.assign({}, it.args, {count: null, repeat: null})}), 0) : phraseOf(it, 2));
          c.set(k, (c.get(k) || 0) + (it.status === 'done' ? (it.mergedN || +it.args.count || +it.args.repeat || 1) : 1));
        }
        olderS = 'Earlier: ' + [...c].map(([k, n]) => k + (n > 1 ? ' (' + COUNTW(n) + ')' : '')).join('; ') + '. Then: ';
      }
      return head + ' ' + olderS + lines.join('; ') + '.' + tail;
    }
    function brief() { return limbs('now: ' + (stateLine() || 'sitting, facing him, both paws empty')); }
    function items(sinceS) {
      const tNow = now(), sinceT = sinceS != null ? tNow - sinceS : -Infinity;
      return L.items.filter(it => it.t0 >= sinceT).map(it => ({n: it.n, t: +it.t0.toFixed(2), t_end: it.t1 == null ? null : +it.t1.toFixed(2), kind: it.kind, name: it.name,
        src: it.src, who: it.who, status: it.status, side: it.side, count: it.args && (it.args.count || it.args.repeat) || null, progress: it.progress,
        object: it.obj ? objPhrase(it.obj) : null, reason: it.reason, hold: it.hold, released: it.released || null, text: phraseOf(it, it.status === 'done' ? 0 : 2)}));
    }
    // ------------------------------------------------------------------ answer(q): a rule-based reader of the OBSERVED state
    // (current-state questions) and of the log (history questions). Only observed / logged facts; when the snapshot lacks
    // the fact the answer says so. 2nd person like brief() ("you're wearing the pink headphones"), limbs() wording.
    const scene = () => (L.state && L.state.props) || L.props || [];
    const PSYN = {laptop: ['computer', 'notebook', 'macbook', 'screen'], headphones: ['headset', 'earphones', 'earmuffs', 'cans'], mug: ['cup', 'coffee'],
      plant: ['flower', 'pot'], chair: ['seat'], book: ['novel'], ball: ['toy']};
    const reEsc = x => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    function aliases(p, all) {
      const name = String(p.name || p.id || '').toLowerCase().replace(/_/g, ' '), id = String(p.id || '').toLowerCase().replace(/_/g, ' ');
      const head = name.split(' ').pop(), own = new Set([name, id, head].filter(Boolean)), taken = new Set();
      for (const q of all) if (q !== p) { taken.add(String(q.id || '').toLowerCase()); taken.add(String(q.name || q.id || '').toLowerCase().split(' ').pop()); }
      for (const k of [id, head]) for (const syn of PSYN[k] || []) if (!taken.has(syn)) own.add(syn);
      for (const a of [...own]) own.add(/[^s]s$/.test(a) ? a.slice(0, -1) : a + 's');
      return [...own].filter(a => a.length > 1);
    }
    function propsIn(text) {              // the scene's props a text mentions, in order of mention
      const all = scene(), out = [], t = String(text || '').toLowerCase();
      for (const p of all) {
        let at = -1;
        for (const a of aliases(p, all)) { const m = new RegExp('\\b' + reEsc(a) + '\\b').exec(t); if (m && (at < 0 || m.index < at)) at = m.index; }
        if (at >= 0) out.push({p, at});
      }
      return out.sort((a, b) => a.at - b.at).map(x => x.p);
    }
    const pName = p => objPhrase(p && (p.name || p.id) || 'it');
    const plural = p => /[^s]s$/.test(String(p && (p.name || p.id) || ''));
    const isAre = p => plural(p) ? 'are' : 'is', itThey = p => plural(p) ? 'they' : 'it', itThem = p => plural(p) ? 'them' : 'it';
    const propOf = o => o ? propsIn(objPhrase(o).toLowerCase())[0] || null : null;
    const listW = a => a.length <= 1 ? (a[0] || '') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
    const CANT = what => "you can't tell " + what + ' right now — there is no reading of your body yet';
    // ---- the snapshot, read
    function wornOf(st) {
      const all = st.props || L.props || [], out = [];
      for (const w of [].concat(st.wearing || [])) {
        const id = typeof w === 'string' ? w : w && (w.id || w.name); if (!id) continue;
        out.push(all.find(p => p.id === id || p.name === id) || {id, name: typeof w === 'string' ? w : w.name || w.id});
      }
      for (const p of all) if (/head|worn|wearing/.test(String(p.where || '')) && !/balanc/.test(String(p.where || '')) && !out.some(q => q.id === p.id)) out.push(p);
      return out;
    }
    const knowsWorn = st => Array.isArray(st.wearing) || (st.props || []).some(p => p.where != null);
    function sideOf(st, id) { const h = st.holding || {}; return h.left === id && h.right === id ? 'both' : h.left === id ? 'left' : h.right === id ? 'right' : null; }
    function holdW(st) {
      const h = st.holding || {}, nm = id => pName(scene().find(p => p.id === id) || {id});
      if (!h.left && !h.right) return 'both paws are empty';
      if (h.left && h.left === h.right) return "you're holding " + nm(h.left) + ' in both paws';
      const r = h.right ? nm(h.right) + ' in your right paw' : null, l = h.left ? nm(h.left) + ' in your left paw' : null;
      return "you're holding " + [r, l].filter(Boolean).join(' and ') + (!h.left ? ', and your left paw is empty' : !h.right ? ', and your right paw is empty' : '');
    }
    function seatOf(st) {
      const on = st.sittingOn || st.sitting_on || null, post = norm(st.posture);
      const seated = !!on || /seat|sit/.test(post);
      const atDesk = st.atDesk === true || (st.atDesk == null && /desk/.test(String(st.seat || ''))) ? true : st.atDesk === false || (st.atDesk == null && seated) ? false : null;
      let w = null;
      if (seated) w = 'sitting' + (on ? ' on the ' + String(on).replace(/_/g, ' ') : st.seat && !/desk/.test(String(st.seat)) ? ' in the ' + String(st.seat).replace(/_/g, ' ') : '') + (atDesk ? ' at the desk' : '');
      else if (post) w = {standing: 'standing', lying: 'lying down', airborne: 'in the air', walking: 'walking', moving: 'moving around', dancing: 'dancing', spinning: 'spinning'}[post] || post.replace(/_/g, ' ');
      return {seated, standing: post === 'standing' || st.standing === true, lying: /lying/.test(post) || !!st.lying, atDesk: st.atDesk == null && !seated && !st.seat ? null : atDesk, on, w};
    }
    function faceW(st) {
      const f = norm(st.facing);
      if (st.turnedBack) return 'turned back toward him';
      if (!f) return null;
      return f === 'camera' || f === 'user' ? 'facing him' : f === 'away' ? (st.atDesk ? 'facing the desk, away from him' : 'turned away from him') : f === 'desk' ? 'facing the desk' : f === 'tumbling' ? 'tumbling' : 'turned ' + (dirPhrase(f) || f);
    }
    function actOf(st) {
      const a = st.activity; if (!a) return null;
      const s = typeof a === 'string' ? a : String(a.name || a.verb || '');
      const paused = /paus/.test(s) || !!(a && a.paused) || !!st.turnedBack;
      let o = typeof a === 'object' && a.object ? a.object : null;
      if (!o) { const it = lastDoing(x => x.kind === 'activity'); o = it && (it.obj || (it.args && it.args.object)); }
      const p = o ? propOf(o) : (st.props || []).find(q => openOf(q, st) === true) || null;
      const verb = /typ|work|use/.test(norm(s)) ? 'typing on' : s.replace(/\s*\(.*$/, '').replace(/_/g, ' ');
      return {paused, p, w: verb + (p ? ' ' + pName(p) : o ? ' ' + objPhrase(o) : '')};
    }
    function openOf(p, st) {
      if (!p) return null;
      if (p.open === true || p.open === false) return p.open;
      if (/^(open|closed)$/.test(String(p.state || ''))) return p.state === 'open';
      if (Array.isArray(st.open)) { if (st.open.includes(p.id)) return true; if (p.openable || /laptop|box|lid|case/.test(String(p.name || p.id))) return false; }
      return null;
    }
    function lookW(st) {
      const lk = norm(st.looking); if (!lk) return null;
      if (lk === 'user' || lk === 'camera' || lk === 'me') return 'at him';
      if (lk === 'object') { const a = actOf(st); return a && !a.paused && a.p ? 'at the screen of ' + pName(a.p) : 'at what you are working on'; }
      if (lk === 'away') return 'away from him';
      return LOOK_W[lk] || 'at ' + objPhrase(lk);
    }
    function placeOf(p, st) {
      const side = sideOf(st, p.id), wh = String(p.where || '');
      if (side) return side === 'both' ? 'in both your paws' : 'in your ' + side + ' paw';
      if (wornOf(st).some(q => q.id === p.id)) return "on your head — you're wearing " + itThem(p);
      if (p.id === 'chair' && (st.sittingOn === 'chair')) return "under you — you're sitting on it" + (st.chairAtDesk || st.atDesk ? ' at the desk' : '');
      let w;
      if (!wh) return null;
      if (/paw|hand/.test(wh)) { const s = ownSide(wh); w = s ? 'in your ' + s + ' paw' : 'in your paw'; }
      else if (wh === 'home') w = plural(p) ? 'in their usual spot' : 'in its usual spot';
      else if (wh === 'moved') w = p.id === 'chair' && st.chairAtDesk ? 'pulled up to the desk' : 'moved from ' + (plural(p) ? 'their' : 'its') + ' usual spot';
      else { const x = wh.replace(/_top$/, '').replace(/_/g, ' '); w = /^(next|in|by|on|under|at|beside|near|behind|over|up|back|against|away|balanced)\b/.test(x) ? x.replace(/\bits\b/g, 'your') : 'on the ' + x.replace(/^the /, ''); }
      const o = openOf(p, st);
      return w + (o === true ? ', open' : o === false ? ', closed' : '');
    }
    // ---- current-state answers
    function aWear(s, st) {
      if (!st || !knowsWorn(st)) return CANT("what you're wearing");
      const worn = wornOf(st), wl = listW(worn.map(pName));
      const yn = /^(are|is|do|have|got)\b/.test(s) || /\b(are|is) (the|your) .* on\b/.test(s);
      const qp = propsIn(s)[0], noun = (/\b(?:a|an|the|your|any|my)\s+([a-z]+)\s*(?:on)?\s*$/.exec(s.replace(/\b(right now|now|still|today)\b/g, '').trim()) || [])[1];
      if (yn && qp) return worn.some(w => w.id === qp.id) ? "yes, you're wearing " + pName(qp)
        : "no, you're not wearing " + pName(qp) + (worn.length ? " — you're wearing " + wl : " — you're not wearing anything") + (placeOf(qp, st) ? '; ' + pName(qp) + ' ' + isAre(qp) + ' ' + placeOf(qp, st) : '');
      if (yn && noun && !/^(anything|something|it|them)$/.test(noun)) return "no, you're not wearing a " + noun + (worn.length ? " — you're wearing " + wl : " — you're not wearing anything");
      if (yn) return worn.length ? "yes, you're wearing " + wl : "no, you're not wearing anything";
      return worn.length ? "you're wearing " + wl : "nothing — you're not wearing anything";
    }
    function aHold(s, st) {
      if (!st || !st.holding) return CANT("what you're holding");
      const h = st.holding, sm = /\b(left|right)\b/.exec(s), qp = propsIn(s)[0];
      if (sm && !qp) { const id = h[sm[1]]; return id ? 'your ' + sm[1] + ' paw is holding ' + pName(scene().find(p => p.id === id) || {id}) : 'nothing — your ' + sm[1] + ' paw is empty'; }
      if (qp) {
        const side = sideOf(st, qp.id), pl = placeOf(qp, st);
        if (/\bwhich\b/.test(s)) return side ? (side === 'both' ? 'both paws' : 'your ' + side + ' paw') + ' — you\'re holding ' + pName(qp) + (side === 'both' ? ' in both' : '') : "neither — you're not holding " + pName(qp) + (pl ? '; ' + itThey(qp) + ' ' + isAre(qp) + ' ' + pl : '');
        return side ? "yes, you're holding " + pName(qp) + (side === 'both' ? ' in both paws' : ' in your ' + side + ' paw')
          : "no, you're not holding " + pName(qp) + ' — ' + holdW(st) + (pl ? '; ' + pName(qp) + ' ' + isAre(qp) + ' ' + pl : '');
      }
      const noun = (/\b(?:holding|have|carrying)\s+(?:a|an|the|your|my|any)\s+([a-z]+)\b/.exec(s) || [])[1];
      if (noun && !/^(anything|something|thing)$/.test(noun)) return "no, you're not holding a " + noun + ' — ' + holdW(st);
      if (/^(are|is|do|have)\b/.test(s) && !(h.left || h.right)) return 'no — both paws are empty';
      if (/^(are|is|do|have)\b/.test(s)) return 'yes — ' + holdW(st);
      return (h.left || h.right ? '' : 'nothing — ') + holdW(st);
    }
    function aSeat(s, st) {
      if (!st || !st.posture && !st.sittingOn) return CANT('how you are positioned');
      const S = seatOf(st), w = S.w || 'in an unknown posture';
      if (/\bwhere are you sitting\b|\bwhat are you sitting (on|in|at)\b/.test(s)) return S.seated ? "you're " + w : "you're not sitting — you're " + w;
      if (/\b(at|by) (the|your) desk\b/.test(s)) { if (S.atDesk == null) return "you can't tell whether you're at the desk — you're " + w;
        return S.atDesk ? "yes, you're " + (S.seated ? w : 'at the desk') : "no, you're not at the desk — you're " + w; }
      if (/\bon (the|your) chair\b|\bin (the|your) chair\b/.test(s)) return S.on === 'chair' || /chair/.test(String(st.seat || '')) ? "yes, you're " + w : "no, you're not on the chair — you're " + w;
      if (/\bstand(ing)?\b|\bon your feet\b/.test(s) && !/\bsit|seat/.test(s)) return S.standing ? "yes, you're standing" : "no, you're " + w;
      if (/\b(lying|lie) down\b|\blying\b/.test(s)) return S.lying ? "yes, you're lying down" : "no, you're " + w;
      if (/\bsit(ting)?\b|\bseated\b|\bsat\b/.test(s)) return S.seated ? "yes, you're " + w : "no, you're not sitting — you're " + w;
      const f = faceW(st);
      return "you're " + w + (f ? ', ' + f : '');
    }
    function aOpen(s, st) {
      if (!st) return CANT('whether that is open');
      let p = propsIn(s)[0];
      if (!p) { const op = scene().filter(q => openOf(q, st) != null); p = op.find(q => /laptop/.test(String(q.name || q.id))) || op[0]; }
      if (!p) return "you can't tell — nothing here is known to open or close";
      const o = openOf(p, st);
      if (o == null) return "you can't tell whether " + pName(p) + ' ' + isAre(p) + ' open or closed';
      const lid = /\blid\b/.test(s) ? (o ? ', lid up' : ', lid down') : '';
      const askOpen = /\bopen(ed)?\b/.test(s) && !/\b(closed|shut)\b/.test(s);
      if (/^(is|are|was|has|did)\b/.test(s) || /\bstill\b/.test(s)) return (askOpen === o ? 'yes, ' : 'no, ') + pName(p) + ' ' + isAre(p) + ' ' + (o ? 'open' : 'closed') + lid;
      return pName(p) + ' ' + isAre(p) + ' ' + (o ? 'open' : 'closed') + lid;
    }
    function doingW(st) {
      const a = actOf(st), S = seatOf(st), f = faceW(st);
      const lid = a && a.p && openOf(a.p, st) === true ? 'lid open' : null;
      if (a && !a.paused) return {busy: true, w: "you're " + a.w + [lid, S.seated ? S.w : null].filter(Boolean).map(x => ', ' + x).join('')};
      if (a && a.paused) return {busy: false, paused: true, w: a.w.replace(/^typing on/, 'typing on') + ' is paused — you turned back toward him and are still ' + (S.w || 'in place') + (lid ? ' (' + (a.p ? pName(a.p) + ' still open' : 'lid still open') + ')' : '')};
      const run = L.items.filter(it => it.status === 'doing' && it.kind !== 'activity' && !(it.hold)).pop();
      if (run) return {busy: true, w: "you're in the middle of " + itemPhrase(run, 1) + (run.steps && run.steps.length ? ' (done so far: ' + run.steps.map(eng).join(', ') + ')' : '') + ' — not finished yet'};
      if (Array.isArray(st.active) && st.active.length) return {busy: true, w: "you're in the middle of " + st.active.map(x => String(x.name || x.kind).replace(/_/g, ' ')).slice(0, 2).join(' and ')};
      const pos = [S.w, f].filter(Boolean).join(', ');
      return {busy: false, idle: true, w: 'nothing in particular right now' + (pos ? " — you're " + pos : ''), pos};
    }
    function aDoing(s, st) {
      if (!st) return CANT("what you're doing");
      const d = doingW(st);
      const gen = /^(are|is) you (?:still )?(\w+ing)\b/.exec(s);
      if (gen && !/^(working|typing|doing|busy)$/.test(gen[2])) {
        const k = keyOf(gen[2]);
        const run = k && L.items.filter(it => (it.status === 'doing') && keysIn(itemPhrase(it, 1)).includes(k.key)).pop();
        return run ? "yes, you're " + itemPhrase(run, 1) + ' right now' : "no, you're not " + s.slice(gen.index + gen[0].length - gen[2].length).replace(/\b(right now|now)\b/g, '').trim() + ' right now — ' + d.w.replace(/^you're /, "you're ");
      }
      if (/^(are|is) you\b/.test(s) || /^(busy)\b/.test(s)) return d.busy ? 'yes — ' + d.w : d.paused ? 'not right now — ' + d.w : d.idle ? "no, you're not doing anything in particular right now" + (d.pos ? " — you're " + d.pos : '') : 'no — ' + d.w;
      return d.paused ? d.w.charAt(0).toLowerCase() + d.w.slice(1) : d.w;
    }
    function aWhere(s, st, rest) {
      if (!st) return CANT('where things are');
      const qs = propsIn(rest);
      if (!qs.length) { const n = (/^(?:the|your|my|a)?\s*([a-z]+)/.exec(rest) || [])[1]; return n ? "you can't tell — there is no " + n + ' among your things' : "you can't tell what he means"; }
      const p = qs[0], pl = placeOf(p, st);
      if (!pl) return "you can't tell where " + pName(p) + ' ' + isAre(p);
      const ref = qs[1], prep = /^(is|are)\b/.test(s) && /\b(on|in|at|under|next to|by|near|beside)\b/.exec(rest);
      if (prep) {
        const want = ref ? aliases(ref, scene()) : [(/\b(?:on|in|at|under|next to|by|near|beside)\s+(?:the|your|my)?\s*([a-z]+)/.exec(rest) || [])[1]].filter(Boolean);
        const hit = want.some(a => new RegExp('\\b' + reEsc(a) + '\\b').test(pl));
        return (hit ? 'yes, ' : 'no, ') + pName(p) + ' ' + isAre(p) + ' ' + pl;
      }
      return pName(p) + ' ' + isAre(p) + ' ' + pl;
    }
    function aFace(s, st) {
      if (!st) return CANT('where you are looking');
      const lw = lookW(st), fw = faceW(st);
      if (/\b(looking at|look(ing)? at|see|watching)\b.*\b(me|him)\b|\beyes on me\b/.test(s)) {
        if (!lw) return "you can't tell where you're looking";
        return lw === 'at him' ? "yes, you're looking at him" + (fw ? ', ' + fw : '') : "no, you're looking " + lw + (fw ? ', ' + fw : '');
      }
      if (/\bfacing\b.*\b(me|him)\b|\bturned (back )?(toward|towards|to) (me|him)\b|\bturned back\b/.test(s)) {
        if (!fw) return "you can't tell which way you're facing";
        return /him$/.test(fw) && !/away/.test(fw) ? "yes, you're " + fw : "no, you're " + fw;
      }
      if (/\bturned away\b|\bfacing away\b/.test(s)) { if (!fw) return "you can't tell which way you're facing"; return /away/.test(fw) ? "yes, you're " + fw : "no, you're " + fw; }
      if (/\blooking\b|\blook at\b/.test(s)) return lw ? "you're looking " + lw : "you can't tell where you're looking";
      return fw ? "you're " + fw + (lw ? ', looking ' + lw : '') : "you can't tell which way you're facing";
    }
    // ---- history: verbs matched by meaning (wave/waved/waving, put on/wore, took off, …) in the log's own words
    const KEYS = [
      ['take_off', /\b(?:took|take|takes|taking|taken)\b(?: \w+){0,4}? off\b|\bremov(?:e|ed|es|ing)\b/],
      ['put_on', /\bput(?:s|ting)? on\b|\bput(?:s|ting)? (?:\w+ ){1,4}?on\b(?!\s+(?:the|a|an|your|my|top|to|it|its)\b)|\bwor[en]\b|\bwear(?:s|ing)?\b|\bdon(?:ned|ning)?\b/],
      ['put_back', /\bput(?:s|ting)?\b(?: \w+){0,4}? back\b/],
      ['place', /\bput(?:s|ting)?\b(?: \w+){0,4}? (?:down|on (?:the|a|an|your|my|top)|onto|in(?:to)? the)\b|\bplac(?:e|ed|es|ing)\b|\bset\b(?: \w+){0,3}? down\b/],
      ['pick_up', /\bpick(?:s|ed|ing)?\b(?: \w+){0,4}? up\b|\bgrab(?:s|bed|bing)?\b|\blift(?:s|ed|ing)?\b/],
      ['drop', /\bdrop(?:s|ped|ping)?\b/], ['throw', /\b(?:threw|throw(?:s|n|ing)?|toss(?:es|ed|ing)?)\b/],
      ['wave', /\bwav(?:e|es|ed|ing)\b/], ['nod', /\bnod(?:s|ded|ding)?\b/], ['shake_head', /\b(?:shook|shake|shakes|shaking) (?:your |my |its |his )?head\b/],
      ['clap', /\bclap(?:s|ped|ping)?\b/], ['backflip', /\bback ?flip/], ['frontflip', /\bfront ?flip/], ['flip', /\bflip(?:s|ped|ping)?\b|\bsomersault/],
      ['cartwheel', /\bcartwheel/], ['spin', /\b(?:spin(?:s|ning)?|spun|twirl(?:s|ed|ing)?)\b/],
      ['turn_back', /\bturn(?:s|ed|ing)?\b(?: \w+){0,3}? back\b|\bturn(?:s|ed|ing)? (?:to|toward|towards) (?:face )?(?:him|me)\b|\bface[ds]? (?:him|me)\b/],
      ['turn_away', /\bturn(?:s|ed|ing)? away\b/], ['turn_around', /\bturn(?:s|ed|ing)? around\b/],
      ['sit', /\b(?:sit|sits|sat|sitting|seated)\b/], ['stand', /\b(?:stand|stands|stood|standing)\b|\b(?:get|got|gets|getting) up\b/],
      ['open', /\bopen(?:s|ed|ing)?\b/], ['close', /\b(?:close|closes|closed|closing|shut|shuts|shutting)\b/],
      ['type', /\btyp(?:e|es|ed|ing)\b|\bwork(?:s|ed|ing)?\b|\bus(?:e|ed|es|ing) (?:the |your )?(?:\w+ )?(?:laptop|computer)\b/],
      ['push', /\bpush/], ['pull', /\bpull/], ['slide', /\bslid(?:e|es|ing)?\b/], ['bring', /\b(?:bring|brings|bringing|brought)\b/],
      ['move', /\bmov(?:e|ed|es|ing)\b/], ['walk', /\bwalk/], ['jump', /\bjump/], ['hop', /\bhop(?:s|ped|ping)?\b/], ['dance', /\bdanc/],
      ['bow', /\bbow(?:s|ed|ing)?\b/], ['wink', /\bwink/], ['blink', /\bblink/], ['shrug', /\bshrug/], ['point', /\bpoint/], ['thumbs_up', /\bthumbs? up\b/],
      ['stretch', /\bstretch/], ['yawn', /\byawn/], ['laugh', /\blaugh/], ['high_five', /\bhigh ?fives?\b/], ['salute', /\bsalut/], ['bounce', /\bbounc/],
      ['ear_wiggle', /\b(?:wiggl|twitch|flick)\w* (?:your |my |its |an )?ears?\b/], ['look_around', /\blook(?:s|ed|ing)? around\b/],
      ['look', /\blook(?:s|ed|ing)? (?:at|over|up|down|away|closely)\b/], ['show', /\b(?:held|hold\w*) (?:\w+ ){0,3}?up\b|\bshow(?:s|ed|ing|n)?\b/],
      ['sigh', /\bsigh/], ['shoulder_roll', /\bshoulder roll|\broll(?:ed|s)? (?:your |its )?shoulders?/], ['head_tilt', /\btilt\w* (?:your |its )?head\b|\bhead tilt/],
      ['kiss', /\bkiss/], ['hug', /\bhug(?:s|ged|ging)?\b/], ['stop', /\bstop(?:s|ped|ping)?\b/],
    ];
    function keyOf(text) { let best = null; for (const [key, re] of KEYS) { const m = re.exec(text); if (m && (!best || m.index < best.at)) best = {key, at: m.index, len: m[0].length}; } return best; }
    function keysIn(text) { const t = String(text || '').toLowerCase(), out = []; for (const [key, re] of KEYS) if (re.test(t)) out.push(key); return out; }
    const splitDid = d => String(d).split(/,\s*(?:and\s+then\s+|then\s+|and\s+)?|\s+and\s+then\s+|\s+then\s+/).map(x => x.trim()).filter(Boolean);
    function itemPhrase(it, form) { const p = it.obj ? propOf(it.obj) : null; return phraseOf(p ? Object.assign({}, it, {obj: {name: p.name || p.id}}) : it, form); }
    function countOf(it) {
      const a = it.args || {}; const n = +a.count || +a.repeat || +it.count; if (n > 0) return n;
      const m = /\b(\d+) times\b|\b(twice)\b/.exec(it.did || ''); return m ? (m[2] ? 2 : +m[1]) : 1;
    }
    function covered(it) {                // a typing activity that is the tail of the job that started it
      if (it.kind !== 'activity') return false;
      return L.items.some(e => e.n < it.n && e.engine && /typ/.test(e.endDid || e.did || '') && (e.t1 == null || Math.abs(e.t1 - it.t0) < 3));
    }
    function facts() {
      const out = [], mk = (it, text, status, n, x) => Object.assign({it, text, status, n, keys: keysIn(text), props: propsIn(text), t: it.t0 != null ? it.t0 : it.t}, x || {});
      for (const it of L.items) {
        if (NOT_ITEMIZED.has(it.src) || it.kind === 'release') continue;
        if (it.kind === 'activity') { out.push(mk(it, eng(it.did || it.name), it.status === 'doing' ? 'doing' : 'done', 0, {act: true})); continue; }
        if (it.kind === 'event') { out.push(mk(it, eng(it.endDid || it.name.replace(/_/g, ' ')), 'done', 1, {self: true})); continue; }
        let ed = it.engine && it.endDid ? String(it.endDid) : null;
        if (ed && /^\w+ing\b/.test(ed) && !/^nothing/.test(ed)) ed = null;
        if (ed && /^nothing to do/.test(ed)) { out.push(mk(it, itemPhrase(it, 2), 'noop', 0)); continue; }
        const pieces = ed ? splitDid(ed) : it.engine && it.steps && it.steps.length ? it.steps : [];
        const pf = pieces.map(p => mk(it, eng(p), 'done', 1, {piece: true}));
        out.push(...pf);
        const main = mk(it, itemPhrase(it, it.status === 'done' ? 0 : 2), it.status, it.status === 'done' ? countOf(it) : it.status === 'interrupted' ? (it.progress || 0) : 0, {main: true});
        const op = propOf(it.obj); if (op && !main.props.some(p => p.id === op.id)) main.props.push(op);
        const cov = pf.some(f => main.keys.some(k => f.keys.includes(k)));
        if (!cov || it.status !== 'done') { if (cov) main.n = 0; out.push(main); }
      }
      for (const r of L.reflex) {
        const k = norm(r.name), v = V[k] || V[k.replace(/^reflex_/, '')];
        const text = v ? v[0] : r.did ? eng(r.did).replace(/\s*\(.*\)$/, '') : k.replace(/_/g, ' ');
        out.push({it: {who: 'self', src: r.src, t0: r.t, side: r.side, args: {}}, text, status: 'done', n: 1, keys: keysIn(text), props: [], t: r.t, reflex: true});
      }
      return out.sort((a, b) => a.t - b.t);
    }
    const SKIPN = /^(head|paws?|hands?|ears?|tail|tongue|eyes?|heart|nose|chin|arms?|foot|feet|legs?|body|face|shoulders?|way|times?|thing|bit|while|moment|second|little|lot|turn|turns)$/;
    function parseAct(s) {
      const m = /\b(?:did|didn't|have|has|had|were|was|could|would)(?: not)?\s+you\s+(?:not\s+|just\s+|ever\s+|already\s+|really\s+|actually\s+|also\s+|still\s+)*(.*)$/.exec(s);
      let rest = (m ? m[1] : s).replace(/\b(today|earlier|before|just now|at all|so far|yet|ever|recently|a while ago|again|already|this call|right now)\b/g, ' ').replace(/\s+/g, ' ').trim();
      const k = keyOf(rest), props = propsIn(rest), side = (/\b(left|right)\b/.exec(rest) || [])[1] || null;
      let noun = null;
      if (!props.length) {
        const r2 = k ? rest.slice(0, k.at) + ' ' + rest.slice(k.at + k.len) : rest;
        const nm = /\b(?:the|a|an|your|my|his|some)\s+(?:\w+\s+)?([a-z]+)\b/.exec(r2);
        if (nm && !SKIPN.test(nm[1]) && !/^(left|right|other|both)$/.test(nm[1])) noun = nm[1].replace(/[^s]s$/, x => x);
      }
      return {rest, key: k && k.key, props, side, noun, aux: m ? (/\bha(ve|s|d)\b/.exec(m[0]) ? 'have' : 'did') : 'did'};
    }
    function match(P) {
      const F = facts();
      if (!P.key) {
        const w = (P.rest.split(' ')[0] || '').replace(/(ed|ing|es|s)$/, '');
        if (w.length < 3) return [];
        return F.filter(f => new RegExp('\\b' + reEsc(w)).test(f.text) && P.props.every(p => f.props.some(q => q.id === p.id)));
      }
      return F.filter(f => f.keys.includes(P.key) && P.props.every(p => f.props.some(q => q.id === p.id)) &&
        (!P.side || new RegExp('\\b' + P.side + '\\b').test(f.text) || f.it.side === P.side) && (!P.noun || new RegExp('\\b' + reEsc(P.noun)).test(f.text)));
    }
    const echo = r => r.replace(/\bmyself\b/g, 'himself').replace(/\bme\b/g, 'him').replace(/\bmy\b/g, 'his').replace(/\s+/g, ' ').trim();
    function hint(P) { const st = L.state, p = P.props[0]; if (!st || !p) return ''; const pl = placeOf(p, st); return pl ? '; ' + pName(p) + ' ' + isAre(p) + ' ' + pl : ''; }
    function partW(it) {
      const a = it.args || {}, of = +a.count || +a.repeat || +it.count || null, v = norm(it.kind === 'object' && it.name === 'object' ? a.verb : it.name);
      if (OPEN_ENDED.has(v)) return 'you ' + itemPhrase(it, 0) + ' for a while, then stopped';
      const p = it.progress;
      return 'you started to ' + itemPhrase(it, 2) + ' but it was stopped' + (p != null && of ? ' after ' + p + ' of ' + of : p === 0 ? ' before one was done' : p > 0 ? ' after ' + p : ' before finishing');
    }
    const whyW = it => it.who === 'self' ? 'on your own' : /^(tool|model|voice|voice_early)$/.test(norm(it.src)) ? 'because you chose to (your own tool call)' : 'because he asked you to';
    function aDid(s) {
      const P = parseAct(s), M = match(P);
      if (/^(do|done) (anything|something)\b/.test(P.rest)) return aSeq('all');
      const done = M.filter(f => f.status === 'done' && !(f.act && M.some(g => g !== f && g.status === 'done' && !g.act)));
      const real = done.filter(f => !f.reflex), pick = (real.length ? real : done).slice(-1)[0];
      const tot = done.reduce((a, f) => a + (f.act ? 0 : f.n), 0) + M.filter(f => f.status === 'interrupted').reduce((a, f) => a + (f.n || 0), 0);
      const neg = (P.aux === 'have' ? "no, you haven't " : 'no, you did not ') + echo(P.rest);
      if (/^why\b|^how come\b/.test(s)) {
        if (/\b(could not|couldn't|did not|didn't|not)\b/.test(s.split(' you ')[0] + ' ') ) {
          const fl = M.filter(f => f.status === 'failed').pop();
          if (fl) return 'because ' + String(fl.it.reason || 'it could not be done');
          if (pick) return 'you did — you ' + pick.text;
          return "there's no record that you were asked to";
        }
        if (pick) return whyW(pick.it) + ' — you ' + pick.text + (pick.reflex ? ' (an idle fidget, not an action)' : '');
        const fl = M.filter(f => f.status === 'failed').pop(); if (fl) return 'you tried to ' + fl.text + " but couldn't — " + fl.it.reason;
        const ip = M.filter(f => f.status === 'interrupted').pop(); if (ip) return partW(ip.it);
        return 'you ' + (P.aux === 'have' ? "haven't " : "didn't ") + echo(P.rest) + hint(P);
      }
      if (/^when\b/.test(s)) {
        if (!pick) return neg + hint(P);
        return 'at ' + fmtClock(pick.t - (L.firstT0 || 0)) + ' into the call (' + agoW(now() - pick.t) + ') — you ' + pick.text;
      }
      if (/\bhow many times\b|\bhow often\b|\bhow many\b/.test(s)) {
        const ip = M.filter(f => f.status === 'interrupted').pop(), fl = M.filter(f => f.status === 'failed').pop();
        const extra = (ip ? (pick ? '; ' : ' — ') + partW(ip.it) : '') + (fl ? (pick || ip ? '; ' : ' — ') + 'you tried to ' + fl.text + " but couldn't — " + fl.it.reason : '');
        if (!tot) return (M.length ? 'not once' : 'zero') + (ip || fl ? extra : ' — ' + neg.replace(/^no, /, '') + hint(P));
        return COUNTW(tot) + (pick ? ' — you ' + pick.text : '') + (pick && pick.reflex ? ' (on your own, idle fidgets)' : '') + extra;
      }
      if (pick && /^(what|which)\b/.test(s)) {   // "what did you pick up?" → the things, in order
        const u = [...new Set(done.filter(f => !f.act).map(f => f.text))];
        return 'you ' + u.join('; then ') + (M.some(f => f.status === 'failed') ? "; you also tried to " + M.filter(f => f.status === 'failed').pop().text + " but couldn't" : '');
      }
      if (pick) {
        const job = pick.piece && pick.it && !keysIn(itemPhrase(pick.it, 0)).some(k => pick.keys.includes(k)) ? ' (while ' + itemPhrase(pick.it, 1) + ')' : '';
        return 'yes — you ' + pick.text + job + (tot > 1 ? ' — ' + COUNTW(tot) + ' in all' : '') + (pick.reflex ? ' — on your own (an idle fidget, not something you were asked to do)' : '');
      }
      const dg = M.filter(f => f.status === 'doing').pop();
      if (dg) return "you're doing it right now — " + (dg.act ? dg.text : itemPhrase(dg.it, 1)) + ', not finished yet';
      const ip = M.filter(f => f.status === 'interrupted').pop(); if (ip) return 'partly — ' + partW(ip.it);
      const fl = M.filter(f => f.status === 'failed').pop(); if (fl) return "no — you tried to " + fl.text + " but couldn't: " + (fl.it.reason || 'it was not possible');
      if (M.some(f => f.status === 'pending')) return "not yet — it's planned next but has not started";
      if (M.some(f => f.status === 'noop')) return 'no — you were asked to, but there was nothing to do (it already was that way)';
      if (P.side && P.key) { const o = match(Object.assign({}, P, {side: null})).filter(f => f.status === 'done').pop(); if (o) return neg + ' — you ' + o.text; }
      return neg + hint(P);
    }
    // ---- history in order: last / before that / first / all
    function seq() { return merged(L.items.filter(it => !NOT_ITEMIZED.has(it.src) && (it.kind !== 'release' || it.relN > 0) && it.status !== 'pending' && !covered(it))); }
    function pred(it) {   // what the item did, as a predicate after "you" (or a full clause for one still going)
      if (it.kind === 'event') return eng(it.endDid || it.name.replace(/_/g, ' '));
      if (it.kind === 'activity') return it.status === 'doing' ? 'started ' + eng(it.did || 'working') + ' — still going' + (it.paused ? ', paused while you turned back to him' : '') : eng(it.did || 'worked') + ' — then stopped';
      let ed = it.engine && it.endDid ? String(it.endDid) : null;
      if (ed && /^\w+ing\b/.test(ed) && !/^nothing/.test(ed)) ed = null;
      const merged_ = it.merged > 1 ? ' — ' + COUNTW(it.mergedN || it.merged) + ' in a row' : '';
      switch (it.status) {
        case 'done': return ed ? (/^nothing to do/.test(ed) ? 'were asked to ' + itemPhrase(it, 2) + ', but there was nothing to do' : eng(ed)) : itemPhrase(it, 0) + merged_;
        case 'doing': return null;
        case 'interrupted': return partW(it).replace(/^you /, '') + (ed ? '; what got done: ' + eng(ed) : '');
        case 'failed': return 'tried to ' + itemPhrase(it, 2) + " but couldn't — " + (it.reason || 'it was not possible') + (ed ? '; what got done before that: ' + eng(ed) : '');
      }
      return itemPhrase(it, 0);
    }
    function clause(it) { const p = pred(it); return p != null ? 'you ' + p : "you're in the middle of " + itemPhrase(it, 1) + (it.steps && it.steps.length ? ' (done so far: ' + it.steps.map(eng).join(', ') + ')' : '') + ' — not finished yet'; }
    function aSeq(which) {
      const list = seq(), pend = L.items.filter(it => it.status === 'pending' && !NOT_ITEMIZED.has(it.src));
      const next = pend.length ? '; next up, not started yet: ' + itemPhrase(pend[pend.length - 1], 2) : '';
      if (!list.length) return "nothing yet — you haven't done anything on purpose this call" + next;
      const c = it => clause(it).replace(/^you /, '');
      if (which === 'first') return 'first, you ' + c(list[0]);
      if (which === 'before') return list.length > 1 ? 'before that, you ' + c(list[list.length - 2]) : 'nothing before that — ' + clause(list[0]);
      if (which === 'all') return 'so far you ' + list.map(c).join('; then ') + next;
      const last = list[list.length - 1];
      return (pred(last) != null ? 'you just ' + c(last) : clause(last)) + next;
    }
    function answer(q) { return limbs(answer0(q)); }
    function answer0(q) {
      let s = String(q || '').toLowerCase().replace(/[‘’]/g, "'").replace(/\bwhat's\b/g, 'what is').replace(/\bwhere's\b/g, 'where is').replace(/\bwho's\b/g, 'who is')
        .replace(/\byou're\b/g, 'you are').replace(/\b(is|are|was|were|did|do|does|have|has|could|would)n't\b/g, '$1 not').replace(/\bcan't\b/g, 'can not')
        .replace(/[?!.,;:"]+/g, ' ').replace(/\s+/g, ' ').trim();
      s = s.replace(/^(and|so|ok|okay|hey|um|uh|well|tell me|can you tell me|do you know)\s+/g, '').replace(/^(and|so)\s+/, '');
      const st = L.state;
      if (!s) return aSeq('last');
      // history
      if (/^(why|how come|when)\b/.test(s) && /\byou\b/.test(s)) return aDid(s);
      if (/\bhow many times\b|\bhow often\b|\bhow many \w+s? (did|have) you\b/.test(s)) return aDid(s);
      if (/\b(what|which) (did|have) you (just )?(do|done)\b|\bwhat (was|is) the (last|first)\b|\bwhat (have you been|were you) (doing|up to)\b|\bwhat happened\b|\bwhat else\b|^before that\b|\bwhat did you do\b|\bwhat have you done\b/.test(s)) {
        if (/\bfirst\b/.test(s)) return aSeq('first');
        if (/\bbefore\b/.test(s)) return aSeq('before');
        if (/\b(just|last|latest|recent|recently|a moment ago|a second ago)\b/.test(s)) return aSeq('last');
        return aSeq('all');
      }
      if (/\b(did|have|had|has) (not )?you\b/.test(s) && !/\bhave you got\b/.test(s)) return aDid(s);
      if (/\bwere you\b/.test(s)) return aDid(s);
      // current state
      if (/\bwear|\b(have|got) (\w+ ){0,4}on\b|\bon your head\b|\b(are|is) (the |your )?(\w+ )?(headphones|headset|earphones|hat|cap|glasses|crown|scarf) (still )?on\b/.test(s)) return aWear(s, st);
      if (/\b(open|opened|closed|shut)\b|\blid\b/.test(s) && !/\b(open|close|shut) (it|the)\b.*\bplease\b/.test(s)) return aOpen(s, st);
      if (/\bhold(ing)?\b|\bin your (left |right |other )?(paw|hand)s?\b|\bwhich (paw|hand)\b|\bcarry(ing)?\b|\b(paw|hand)s? (are )?(full|empty|free)\b|\bdo you have (the|a|an|anything|something)\b|\bgot (anything|something) in\b/.test(s)) return aHold(s, st);
      if (/\b(sit|sitting|seated|standing|stand|posture|lying)\b|\b(at|by) the desk\b|\b(on|in) the chair\b|^where are you\b(?! (looking|facing|going))|\bon your feet\b/.test(s) && /\byou\b|\byour\b/.test(s)) return aSeat(s, st);
      if (/\bwhat are you (doing|up to|working on|busy with|in the middle of)\b|\bare you (still )?(working|typing|busy|doing (anything|something))\b|\bwhat is (going on|happening)\b|\bwhat is your (current )?activity\b|^(busy|working)\b/.test(s)) return aDoing(s, st);
      let wm = /\bwhere (is|are) (.+)$/.exec(s);
      if (wm && !/^you\b/.test(wm[2])) return aWhere(s, st, wm[2]);
      wm = /^(is|are) (?!you\b)(.+?) (on|in|at|under|next to|by|near|beside) (.+)$/.exec(s);
      if (wm && propsIn(wm[2]).length) return aWhere(s, st, wm[2] + ' ' + wm[3] + ' ' + wm[4]);
      if (/\bfacing\b|\blooking\b|\blook at\b|\bwhich way\b|\bturned (back|away|around|toward|towards|to)\b|\bsee me\b|\bwatching\b|\beyes on\b/.test(s)) return aFace(s, st);
      if (/^(are|is) you (still )?\w+ing\b/.test(s)) return aDoing(s, st);
      if (keyOf(s) && /\byou\b/.test(s)) return aDid(s);
      return st ? brief() : aSeq('last');
    }
    function reset() {
      L.items = []; L.byId.clear(); L.reflex = []; L.state = null; L.props = null; L.legacy = {seen: new Map(), logN: 0};
      L.firstT = null; L.speechOnly = 0; L.firstT0 = now(); touch();
    }
    L.firstT0 = 0;
    return {push, command, observe, pollLegacy, noteSource, summary, brief, items, answer, reset, now, version: () => L.ver,
      lastChange: () => L.lastChange, _L: L, phraseOf, active: () => L.items.filter(it => it.status === 'doing' || it.status === 'pending').length};
  }

  return {createActionLog, ownSide, VERBS: V, OBJ_VERBS: OBJ_V};
});
