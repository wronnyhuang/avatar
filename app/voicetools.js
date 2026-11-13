/* voicetools.js — the voice model's body / program tools for every character and every transport
 * (voice-tools contract): do_action (alias perform_action), start_behavior, change_behavior, stop_behavior,
 * list_behaviors, look, set_style, learn_gesture — and the events that go back to the voice model as silent notes
 * (behaviour rule moments, the user's learned gestures, attention verdicts, turn bids).
 *
 *   const VT = AvatarVoiceTools.createVoiceTools({perf, motion, behaviors, gestures, attention, runCommand, sendCtl, prompt,
 *       sendEvent, userSpeaking, avatarSpeaking, lastUserSpeech, reactMode, character, now, timers, log})
 *   VT.call(name, args, phase, info) → the tool result (returned to the model at once) | null = not one of these tools
 *   VT.onHub(msg)          server → page {type:'behavior_code', ...} (the coder's answer); true when handled
 *   VT.userSpeech(phase)   'start' | 'stop' of the user's speech;  VT.userText(text, final) the user's transcript (deltas or final)
 *   VT.onEvent(ev)         {type:'gesture'|'attention'|'turn_bid'|'rule'|'emit', ...} → a silent note, maybe a reply
 *   VT.attach()            subscribe to Behaviors.onEvent / Gestures.onEvent / Attention.onEvent when they exist
 *   VT.stats()             tool calls, holds, guard blocks, notes, dispatch µs
 * Every dependency is a getter, feature-detected at call time: the program layer (window.Behaviors, app/behavior.js) and
 * perception (window.Gestures / window.Attention) are built in parallel — absent = an honest {ok:false, reason}.
 *
 * The what-if guard (eval case negative-05): a do_action whose words were a question about an imagined situation ("if I asked
 * you to …, could you?", "what would you do if …", "suppose …") is refused deterministically, also when a mid-sentence
 * pause split the question into two turns (the second half is a bare "could you?"). The user's final transcript can arrive after
 * the tool call: then the action is HELD until it arrives (at most HOLD_MS) and checked then.
 * Note texts describe the situation only; the voice model words every reply itself (no scripted phrases).
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarVoiceTools = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const ACTION = ['do_action', 'perform_action'];
  const NAMES = ['start_behavior', 'change_behavior', 'stop_behavior', 'list_behaviors', 'look', 'set_style', 'learn_gesture', 'attention_rule', 'see_camera',
    'tv_show', 'tv_clear'];   // tv
  const HOLD_MS = 1000;            // a do_action waits at most this long for the user's final transcript (guard)
  const PENDING_TX_MS = 5000;      // the user's speech stopped this recently without a final transcript = one is coming
  const PAIR_MS = 15000;           // a bare "could you?" joins the utterance before it within this window
  const CONT_MS = 4000;            // "and …" within this gap after the user's previous turn continues that request
  const CONT_CALL_MS = 10000;      // … when the model's scene call for the first half came this recently
  const isCont = t => /^(?:and|then|also|plus|and then|after that|afterwards|but then|so)\b/i.test(String(t || '').trim().replace(/^[^a-z]+/i, ''));
  const CODE_WAIT_MS = 25000;      // the behaviour coder's answer
  const TURN_BID_MIN = 0.6;        // deliberateness above which a turn bid while the user is silent asks for a reply
  const RATE = {rule: 3000, gesture: 4000, attention: 6000, turn_bid: 8000};
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const NEXT_PREP = 'It is being prepared: say something short and natural now, not that it is done. A silent note will tell you when it is about to start.';

  // ------------------------------------------------------------------------------------------- the what-if guard (pure)
  function norm(t) {
    return ' ' + String(t || '').toLowerCase().replace(/[’`]/g, "'").replace(/[^a-z0-9'?.!, ]+/g, ' ').replace(/\s+/g, ' ').trim() + ' ';
  }
  const SUBJ = "(?:i|we|someone|somebody|anyone|anybody|they|people|he|she|my \\w+|the kids?)";
  const ASKV = "(?:asked|ask|told|tell|wanted|want|said|say|were to|was to|needed|need|begged|beg|paid|pay|dared|dare|requested|request|let)";
  const HYP = [
    [/\bwhat if\b/, 'what if'],
    [/\b(?:suppose|supposing)\b/, 'suppose'],
    [new RegExp('\\b(?:imagine|pretend) (?:if|that ' + SUBJ + ' ' + ASKV + '|' + SUBJ + ' ' + ASKV + ')\\b'), 'imagine if'],
    [/\bhypothetical(?:ly)?\b|\bin theory\b|\btheoretically\b/, 'hypothetically'],
    [new RegExp("\\blet'?s say\\b|\\bsay " + SUBJ + ' ' + ASKV + '\\b'), "let's say"],
    [new RegExp('\\bif ' + SUBJ + ' (?:ever |just |really |only )?' + ASKV + '\\b'), 'if he asked'],
    [/\bif you (?:had to|were to|were asked|got asked|were told|ever had to|ever got to)\b/, 'if you had to'],
    [/\bwhat would (?:you|happen)\b|\bhow would you\b/, 'what would'],
    [new RegExp('\\b(?:would|could|will|can|might) you\\b.*\\bif ' + SUBJ + '\\b'), 'would you … if'],
  ];
  // a later sentence that is plainly a go-ahead makes it a request after all ("…, could you? Go ahead, show me.")
  const GO = /^(?:ok(?:ay)?|so|then|now|well|alright|all right|and)?\s*(?:go ahead|show me|do it|let'?s see|try it|please do|go for it|do one|do that|do this|show us)\b/;
  function sentences(t) { return norm(t).split(/[.?!]+/).map(x => x.trim()).filter(Boolean); }
  function isHypothetical(text) {
    const ss = sentences(text);
    for (let i = 0; i < ss.length; i++) {
      const s = ' ' + ss[i].replace(/,/g, ' ').replace(/\s+/g, ' ') + ' ';
      const hit = HYP.find(([re]) => re.test(s));
      if (!hit) continue;
      if (ss.slice(i + 1).some(x => GO.test(x.replace(/,/g, ' ').trim()))) return {hyp: false, why: 'go-ahead after ' + hit[1]};
      return {hyp: true, why: hit[1]};
    }
    return {hyp: false, why: ''};
  }
  // "could you?", "would you do that?", "can you do it for me?", "would you be able to?" — a question with no action of
  // its own: it continues the utterance before it
  const BARE = /^(?:and |so |but |then |well |ok(?:ay)? |um |uh )*(?:could|would|can|will|might|do) you(?: be able to)?(?: (?:do|try|manage) (?:it|that|this|one))?(?: (?:please|for me|then|now|though|really|right now))*$/;
  function isBareModal(text) {
    const s = norm(text).replace(/[.?!,]/g, ' ').replace(/\s+/g, ' ').trim();
    return BARE.test(s) || /^(?:is that|would that be) (?:possible|something you (?:could|can) do)$/.test(s);
  }
  /* the texts of the user's request to check: the newest utterance, and — when it is a bare modal question — the one before it
     joined to it (a mid-sentence pause split one question into two turns) */
  function requestTexts(utts, nowMs) {
    const out = [];
    const last = utts.length ? utts[utts.length - 1] : null;
    if (!last) return out;
    if (last.text) out.push(last.text);
    const prev = utts.length > 1 ? utts[utts.length - 2] : null;
    if (prev && prev.text && (!last.text || isBareModal(last.text)) && last.t0 - (prev.t1 || prev.t0) < PAIR_MS) out.push(prev.text + ' ' + (last.text || ''));
    return out;
  }
  function guardCheck(utts, description, nowMs) {
    for (const t of requestTexts(utts, nowMs)) { const h = isHypothetical(t); if (h.hyp) return {block: true, why: h.why, text: t}; }
    const hd = isHypothetical(description);
    if (hd.hyp) return {block: true, why: hd.why + ' (in the description)', text: description};
    return {block: false};
  }

  // ------------------------------------------------------------------------------------------- look targets
  const LOOK_ALIAS = {user: 'user', me: 'user', him: 'user', her: 'user', you: 'user', camera: 'user', person: 'user',
    'at me': 'user', screen: 'user', forward: 'user', ahead: 'user', straight: 'user', 'straight ahead': 'user', front: 'user',
    away: 'away', 'look away': 'away', elsewhere: 'away', around: 'around', left: 'left', right: 'right', up: 'up', down: 'down',
    'up left': 'up_left', 'up right': 'up_right', 'down left': 'down_left', 'down right': 'down_right', 'upper left': 'up_left',
    'upper right': 'up_right', 'lower left': 'down_left', 'lower right': 'down_right', ceiling: 'up', sky: 'up', floor: 'down',
    ground: 'down', 'your left': 'left', 'your right': 'right'};
  function lookKey(target) {
    const t = String(target == null ? '' : target).toLowerCase().replace(/_/g, ' ').replace(/^\s*(at|to|toward|towards|the)\s+/, '').replace(/\s+/g, ' ').trim();
    return LOOK_ALIAS[t] || LOOK_ALIAS[t.replace(/^(at|the|my|your)\s+/, '')] || null;
  }
  function findProp(props, target) {
    const q = String(target || '').toLowerCase().replace(/_/g, ' ').replace(/^(at |the |my |your |that |this )+/, '').trim();
    if (!q) return null;
    let best = null;
    for (const p of props || []) {
      const id = String(p.id || '').toLowerCase(), nm = String(p.name || '').toLowerCase();
      const sc = id === q || nm === q ? 3 : nm.split(/\s+/).includes(q) || q.split(/\s+/).includes(id) ? 2 : 0;
      if (sc && (!best || sc > best.sc)) best = {p, sc};
    }
    return best ? best.p : null;
  }
  const slug = s => String(s || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/)
    .filter(w => w && !/^(a|an|the|you|your|me|my|i|to|and|when|whenever|every|time|please|can|could|would|will|keep|start|always)$/.test(w)).slice(0, 4).join('_') || 'behaviour';

  function createVoiceTools(d) {
    d = d || {};
    const now = d.now || (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    const T = d.timers || {set: (f, ms) => setTimeout(f, ms), clear: id => clearTimeout(id)};
    const get = k => { try { return d[k] ? d[k]() : null; } catch (e) { return null; } };
    const bool = k => { try { return !!(d[k] && d[k]()); } catch (e) { return false; } };
    const V = {n: 0, utts: [], lastStop: -1e12, holds: [], pendingB: new Map(), style: {}, lastNote: {}, seen: new Map(),
      attached: new WeakSet(), calls: [], blocks: [], notes: 0, us: [], expanded: [], early: []};
    function log(rec) { if (d.log) { try { d.log(rec); } catch (e) {} } }
    function prompt(text, opt) { if (!d.prompt) return null; V.notes++; try { return d.prompt(text, opt || {}); } catch (e) { return null; } }
    function sendCtl(o) { if (d.sendCtl) { try { d.sendCtl(o); } catch (e) {} } }
    function heSpeaking() { return bool('userSpeaking'); }
    // the thinking gesture (app/thinking.js) also covers a do_action held for the user's final words
    function TH() { const k = get('thinker'); return k && typeof k.begin === 'function' ? k : null; }

    // ---- the user's words (for the guard): one entry per utterance; deltas append, a final replaces
    function cur() { const u = V.utts[V.utts.length - 1]; return u && !u.final ? u : null; }
    function newUtt() { const u = {text: '', final: false, t0: now(), t1: null}; V.utts.push(u); if (V.utts.length > 12) V.utts.splice(0, 6); return u; }
    function userSpeech(phase) {
      if (phase === 'start') {
        const u = cur();
        if (!u || u.stopped) {
          // an earlier utterance whose transcript never came (transcription failed) must not take a later one's words
          for (const o of V.utts) if (!o.final && o.stopped && now() - (o.t1 || o.t0) > 10000) o.final = true;
          newUtt();
        }
      } else if (phase === 'stop') { V.lastStop = now(); const u = cur(); if (u && !u.stopped) { u.stopped = true; u.t1 = now(); } }
    }
    // a transcript belongs to the OLDEST utterance still waiting for one (realtime transcribes each committed turn in
    // order, and the user's next turn may already have started when the previous one's words arrive); else the current one
    function target() { return V.utts.find(u => !u.final && u.stopped) || cur(); }
    function userText(text, final) {
      let u = target();
      if (!u) { if (!text && final) return; u = newUtt(); }
      if (final) { if (text) u.text = String(text); u.final = true; u.t1 = u.t1 || now(); u.finalAt = now(); }
      else u.text = (u.text + String(text || '')).slice(-600);
      if (final && u.text && isCont(u.text) && V.sceneCall && now() - V.sceneCall.t < CONT_CALL_MS) {   // a split request's 2nd half
        const i = V.utts.indexOf(u);
        if (i > 0 && V.sceneCall.u === V.utts[i - 1] && u.t0 - (V.utts[i - 1].t1 || V.utts[i - 1].t0) < CONT_MS) contExpand(u);
      }
      // a held action is decided once no finished turn of the user's is still waiting for its words
      if (final && !V.utts.some(o => !o.final && o.stopped)) for (const h of V.holds.slice()) resolveHold(h, 'transcript');
    }
    // the user's speech stopped and its final transcript has not come yet (realtime transcribes after the turn is committed)
    function txPending() {
      const u = V.utts[V.utts.length - 1];
      return !!(u && !u.final && u.stopped && now() - V.lastStop < PENDING_TX_MS);
    }

    // ------------------------------------------------------------------------------------------- do_action
    function action(args, info) {
      const P = get('perf');
      const desc = String(args && args.description != null ? args.description : '').replace(/\s+/g, ' ').trim();
      if (!P || typeof P.call !== 'function') return {ok: false, reason: 'there is no body here to do it'};
      if (!desc) return {ok: false, reason: 'no description given'};
      const talk = args && (args.talk === 'through' || args.talk === 'quiet') ? args.talk : undefined;
      const g = guardCheck(V.utts, desc, now());
      // the model often speaks AND calls in one response (then no follow-up turn reads the result): a note asks for one
      if (g.block) { const r = refuse(desc, g, 'at once'); followUp(`His words were a question about an imagined situation, not a request, so the action (${desc.slice(0, 120)}) was not done. Answer him in words, in your own way.`); return r; }
      if (txPending()) {                                    // the user's final words are on their way: decide when they arrive
        const h = {id: 'h' + (++V.n), desc, talk, info, t: now(), move: args && args.move};
        h.timer = T.set(() => resolveHold(h, 'timeout'), HOLD_MS);
        V.holds.push(h);
        { const k = TH(); if (k) { try { k.begin('vt:' + h.id, {description: desc, held: true}); } catch (e) {} } }
        log({vt: 'hold', id: h.id, description: desc});
        return {ok: true, status: 'preparing', id: h.id, next: talk === 'quiet' ? 'He asked for it quietly: no words about it, before, during or after.' : NEXT_PREP};
      }
      return route(desc, args && args.move, talk, info, null);
    }
    /* hybrid (latency first): the fast matchers fire known moves the moment the user says them (the original path)
       and do_action only adds what they did not cover. In order: the scene (things around it) -> the user's steps / object;
       a named move (move argument) or a description the page library reads -> the engine at once (IntentDedupe drops
       what a matcher already started; a refinement replaces it); the user's request already started by a matcher -> already
       under way; else the performer (plan cache -> keyframe planner). */
    function route(desc, move, talk, info, held) {
      if (d.teachRoute) { let tr = null; try { tr = d.teachRoute(desc); } catch (e) { tr = null; } if (tr) { log({vt: 'teach_route', description: desc.slice(0, 120)}); return tr; } }   // teach devloop
      const sr = sceneRoute(desc);                          // things in the scene are not keyframes
      if (sr && sr.status === 'use_object') followUp(`The request (${desc.slice(0, 120)}) was not done: ${sr.reason}. Call object for it now, one call per step, in order.`);
      if (sr && sr.plan) return planWithThings(desc, talk, info, held, sr);   // a trick with things -> the planner
      if (sr) return sr;
      const nt = noThingHere(desc);                         // no scene -> no mimed objects
      if (nt) { log({vt: 'no_thing', description: desc.slice(0, 120)}); return nt; }
      if (d.runLibrary) {
        let lr = null;
        try { lr = d.runLibrary(desc, move && typeof move === 'object' && move.name ? move : null, info) || null; } catch (e) { lr = null; }
        if (lr && (lr.ran || lr.dup)) { log({vt: 'library', ran: lr.ran, dup: lr.dup, description: desc.slice(0, 120)}); return {ok: true, status: 'starting', already: lr.ran ? undefined : true, did: lr.did || (lr.ran ? 'doing it now' : 'starting now — already begun from his own words as he said them (that IS his request: give him a brief natural cue, never say nothing new started)'), next: 'It starts right away: give him a brief natural cue now.'}; }
        if (lr && lr.reason) return {ok: false, reason: String(lr.reason).slice(0, 160)};
      }
      if (coveredNow()) {                                   // a fast matcher started this request already (same utterance) (scene chars: sceneRoute)
        log({vt: 'covered', description: desc.slice(0, 120)});
        return {ok: true, status: 'starting', already: true, did: 'starting now — already begun from his own words as he said them (that IS his request: give him a brief natural cue, never say nothing new started)'};
      }
      const rl = releaseRoute(desc);                        // fixes: "back to normal" is a release, not a plan
      if (rl) return rl;
      const P = get('perf');
      if (!P || typeof P.call !== 'function') return {ok: false, reason: 'there is no body here to do it'};
      const freed = supersedeHolds(desc);                   // fixes: the newest request owns the parts it names
      const r = P.call(desc, Object.assign({}, info || {}, talk ? {talk} : {}, wordsOf(), held != null ? {held_ms: held} : {}));
      if (freed.length) log({vt: 'superseded', parts: freed, description: desc.slice(0, 120)});
      if (r && r.ok !== false && d.matcherForget) { try { d.matcherForget(); } catch (e) {} }   // "more" no longer means the matcher's last pose
      return r;
    }
    // a do_action naming a thing that the scene matcher does not fully read (a trick, a combination) is planned WITH
    // the things (the performer sends them in the plan state; the planner writes object steps / held controls / paw events)
    function planWithThings(desc, talk, info, held, sr) {
      const P = get('perf');
      if (!P || typeof P.call !== 'function') return {ok: false, reason: 'there is no body here to do it'};
      log({vt: 'plan_things', thing: sr.thing, description: desc.slice(0, 120)});
      const r = P.call(desc, Object.assign({}, info || {}, talk ? {talk} : {}, wordsOf(), held != null ? {held_ms: held} : {}, {things: true}));
      if (r && r.ok !== false && d.matcherForget) { try { d.matcherForget(); } catch (e) {} }
      return r;
    }
    /* fixes (judge: head-down asked three times, the head stayed up; eyes stayed shut; "back to normal" said while a
       pose was still held). Three mechanisms, none tied to a phrase:
       coveredNow(): "a fast matcher already started this request" is only true when the matcher's command was read from
         THIS utterance: the page reports the words each matcher command was read from (coveredByMatcher() -> an array:
         '' = unknown, null = a release, which never covers a request), and most of those words must be in the user's newest
         utterance. (The old test — any matcher command since IntentDedupe's last speech start — let a head-up pose from the user's
         PREVIOUS sentence "cover" the model's lower-my-head call when the user had spoken over the avatar: nothing ran, result ok.)
       releaseRoute(desc): a description that only asks to go back to normal / relax / stop posing is a release of
         everything held (and stops what is playing) — the keyframe planner starts from the held pose and returns to it.
       supersedeHolds(desc): before a plan is requested, a pose still HELD on a body part the description names is let go
         (the plan assumes it starts from rest and eases back to its start: over a held pose it returned to that pose). */
    const CW_STOP = new Set(('a an the my your his her its our their me you him it them us i to of for and or but please now then just can could would will do does did be is are was ' +
      'so that this these those okay ok hey um uh well yes no yeah really very little bit some up').split(' '));
    const cwords = t => String(t || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(w => w && !CW_STOP.has(w));
    function coveredNow() {
      let c = false;
      try { c = d.coveredByMatcher ? d.coveredByMatcher() : false; } catch (e) { c = false; }
      if (!c) return false;
      if (!Array.isArray(c)) return true;                   // a page that only says yes / no
      const texts = c.filter(x => x !== null);              // null = a release: relaxing a pose never covers a request
      if (!texts.length) return false;
      const u = V.utts[V.utts.length - 1];
      const his = u && u.text ? new Set(cwords(u.text)) : null;
      if (!his || !his.size) return true;                   // the user's words are not known (no transcript): as before
      return texts.some(t => { const ws = cwords(t); if (!ws.length) return true; return ws.filter(x => his.has(x)).length / ws.length >= 0.6; });
    }
    const REL_LEAD = /^(?:(?:please|now|just|okay|ok|and|then|so|alright|all right) )*(?:relax(?: (?:completely|fully|again|everything|my (?:whole |entire )?body|all (?:of )?(?:it|my body)))?|rest|at ease|unfreeze|loosen up|let (?:it |everything |all of it )?go|stop (?:posing|holding(?: (?:it|that|the pose|still))?|that pose|the pose|that|it|everything|moving)|drop (?:it|that|the pose|that pose|everything)|(?:go|get|come|return|move|ease|settle|snap|pop|shift|change|switch|reset)(?: (?:myself|my body|my whole body|everything|it|all|right|straight|all the way|on))* (?:back )?(?:to|into) (?:a |an |my |the )?(?:normal|neutral|usual|regular|ordinary|default|resting|rest|relaxed|idle|starting|original|natural)(?: (?:standing|sitting|seated|idle|resting|relaxed|natural|usual))?(?: (?:pose|position|posture|stance|self|state|way|look))?|reset(?: (?:my|the) (?:pose|body|position|posture))?|back to (?:a |my |the )?(?:normal|neutral|rest|usual)(?: (?:pose|position|posture|stance|self|state))?|(?:be|act|sit|stand|look) normal(?:ly)?(?: again)?)(?= |$)/;
    const REL_FILL = /^(?:now|again|please|gently|slowly|quickly|smoothly|calmly|naturally|completely|fully|right away|at once|in a (?:simple |normal |gentle |calm |natural |relaxed |easy |smooth )+(?:style|way|manner)|at (?:a )?normal speed|with (?:my )?(?:whole|entire) body|all over|everything|for (?:him|you)|and (?:relax|rest|stay (?:there|like that|relaxed|still)|stop|hold (?:it|that|still))|then (?:relax|rest|stop|stay (?:there|relaxed|still))| )+$/;
    function releaseRoute(desc) {
      const t = String(desc || '').toLowerCase().replace(/[’`]/g, "'").replace(/[^a-z' ]+/g, ' ').replace(/\s+/g, ' ').trim();
      const m = REL_LEAD.exec(t); if (!m) return null;
      let rest = t.slice(m[0].length).trim();
      for (let i = 0; i < 3 && rest; i++) {                 // "drop the pose and go back to normal": the same ask twice
        const m2 = REL_LEAD.exec(rest.replace(/^(?:and then|and|then) /, '')); if (!m2) break;
        rest = rest.replace(/^(?:and then|and|then) /, '').slice(m2[0].length).trim();
      }
      if (rest && !REL_FILL.test(rest + ' ')) return null;  // something else is asked for as well: the planner (supersedeHolds)
      const P = get('perf');
      if (P && typeof P.stop === 'function') { try { P.stop('he asked to go back to normal'); } catch (e) {} }
      let ok = false;
      const mo = get('motion');
      try {
        if (d.runCommand) { const r = d.runCommand({action: 'release', parts: 'all'}, {description: desc}); ok = r === true || !!(r && typeof r === 'object' && (r.ok !== false || r.dup)) || r === false; }
        else if (mo && typeof mo.command === 'function') { mo.command({action: 'release', parts: 'all'}, {src: 'voice'}); ok = true; }
      } catch (e) { ok = false; }
      // a release dropped as a repeat (a matcher released a moment ago) still leaves nothing held
      try { if (mo && typeof mo.release === 'function') { mo.release('all', 400); ok = true; } } catch (e) {}
      log({vt: 'release_desc', description: desc.slice(0, 120), ok});
      return ok ? {ok: true, status: 'starting', did: 'letting go of every held pose and going back to normal', held: []}
        : {ok: false, ran: false, reason: 'the body could not relax just now, so nothing changed on screen'};
    }
    // body-part words -> motion.js release groups (own left / right -> the motion's own-side groups)
    const PART_WORDS = [[/\b(?:eyes?|eyelids?|lids?)\b/, 'eyes'], [/\b(?:head|chin|neck|nose|snout|muzzle)\b/, 'head'], [/\b(?:look(?:s|ing)?|gaze|glance|stare|peek)\b/, 'gaze'],
      [/\b(?:mouth|tongue|lips?|jaw)\b/, 'mouth'], [/\bears?\b/, 'ears'], [/\btail\b/, 'tail'], [/\b(?:eyebrows?|brows?)\b/, 'brows'],
      [/\b(?:face|expression|smile|grin|frown|pout)\b/, 'expression'], [/\b(?:body|torso|chest|back|waist|hips?)\b/, 'body'], [/\bshoulders?\b/, 'shoulders']];
    const PAW_WORD = /\b(?:paws?|hands?|arms?|mittens?|fingers?|fists?|elbows?|wrists?|palms?)\b/;
    const GROUP_OF = {eyes: 'eyes', left_eye: 'eyes', right_eye: 'eyes', gaze: 'gaze', look: 'gaze', head: 'head', mouth: 'mouth', tongue: 'mouth', ears: 'ears', left_ear: 'ears',
      right_ear: 'ears', tail: 'tail', brows: 'brows', body: 'body', shoulders: 'shoulders', paw_l: 'paw_l', hand_l: 'paw_l', right_paw: 'paw_l', paw_r: 'paw_r', hand_r: 'paw_r', left_paw: 'paw_r'};
    // what is held in a pose right now (motion.state().held + a held look), as release groups
    function heldGroups() {
      const out = new Set();
      try {
        const m = get('motion'), s = m && typeof m.state === 'function' ? m.state() : null;
        for (const g of (s && Array.isArray(s.held) ? s.held : [])) if (GROUP_OF[g]) out.add(GROUP_OF[g]);
        for (const a of (s && Array.isArray(s.active) ? s.active : [])) if (a && a.kind === 'look' && a.held) out.add('gaze');
      } catch (e) {}
      return out;
    }
    const HELD_NAME = {eyes: 'eyes', gaze: 'where it looks', head: 'head', mouth: 'mouth', ears: 'ears', tail: 'tail', brows: 'brows', body: 'body', shoulders: 'shoulders', paw_l: 'right paw', paw_r: 'left paw'};
    function heldNames() { return [...heldGroups()].map(g => HELD_NAME[g] || g); }
    function supersedeHolds(desc) {
      const held = heldGroups(); if (!held.size) return [];
      const t = ' ' + String(desc || '').toLowerCase().replace(/[^a-z' ]+/g, ' ').replace(/\s+/g, ' ') + ' ';
      const want = new Set();
      for (const [re, g] of PART_WORDS) if (re.test(t)) want.add(g);
      if (want.has('expression')) { want.add('mouth'); want.add('brows'); }
      if (PAW_WORD.test(t)) {
        // own right = the motion's screen-left group (paw_l); one side only when exactly one is named with the limb
        const R = /\bright (?:paw|hand|arm|mitten|fist|elbow|wrist|palm)\b/.test(t), L = /\bleft (?:paw|hand|arm|mitten|fist|elbow|wrist|palm)\b/.test(t);
        if (R || !L) want.add('paw_l');
        if (L || !R) want.add('paw_r');
      }
      const free = [...want].filter(g => held.has(g));
      if (!free.length) return [];
      const m = get('motion');
      try { if (m && typeof m.release === 'function') m.release(free.map(g => (g === 'expression' ? 'expression' : g)), 300); else return []; } catch (e) { return []; }
      return free;
    }
    /* noThingHere(description) — design note (fox A/B: "pick up the headphones" on a character with NO scene played a keyframe
       mime holding nothing, where the old path honestly declined). When this character has no things around it (motion.props()
       empty), a request that manipulates a physical thing is not planned: an honest refusal the model says in words.
       fixes (judge: "high five", "lift my head up", "close my eyes now", "a gentle sleepy pose", "open my eyes in ..."
       were refused — the old pattern read the words AFTER the noun as the noun): the object is now the noun phrase itself
       (determiner .. the first preposition / adverb / conjunction / punctuation), and the refusal only fires when
         - the verb is one that takes a physical thing (pick up, grab, put on, wear, take off, fetch, carry, bring, hand
           over, eat, drink, play with) and the phrase is not the user's own body, a gesture or an abstract thing (a five, a hug,
           a pose, a breath, a nap, the pace ...), or
         - the verb also has plain body uses (hold, open, close, lift, give, throw, catch, show, use, slide, sit on) and the
           phrase's head is a known everyday object (THINGS).
       Anything else — every body, face, eyelid, head and expression request, and acting something out — goes on to the
       library / planner. No word of the user's is ever required: pretend / mime wording merely skips this check. */
    const BODY_PARTS = /^(paws?|hands?|arms?|ears?|tails?|heads?|noses?|legs?|feet|foot|eyes?|eyelids?|lids?|eyebrows?|brows?|mouths?|tongues?|faces?|fingers?|thumbs?|whiskers?|shoulders?|body|bodies|belly|bellies|tummy|tummies|stomach|chest|knees?|toes?|wings?|horns?|antennae?|lips?|cheeks?|chins?|necks?|backs?|hips?|waist|self|myself|yourself|jaws?|teeth|tooth|fur|snout|muzzle|beak|elbows?|wrists?|palms?|fists?|knuckles?|forehead|hair|mittens?|heels?|ankles?|sides?|tips?|corners?|points?|edges?|gaze|eyesight|voice)$/;
    // acts, gestures, expressions and abstract objects of these verbs: never a physical thing
    const NOT_THINGS = new Set(('five fives hug hugs kiss kisses wave waves smile smiles grin grins frown frowns pout look looks glance peek stare wink winks blink blinks nod nods bow bows ' +
      'curtsy shrug shrugs pose poses position positions posture stance stretch stretches yawn yawns nap naps rest sleep snooze break breath breaths step steps turn turns twirl spin spins jump jumps ' +
      'hop hops leap dance dances jig wiggle wiggles shake shakes shimmy clap claps salute salutes sign signs gesture gestures move moves motion motions movement movements face faces ' +
      'expression expressions moment moments second seconds minute minutes beat beats bit while time times try go shot sound sounds noise laugh giggle chuckle cheer cheers punch kick kicks ' +
      'tantrum fit peace heart hearts raspberry squeeze cuddle note tune song rhythm count pause stillness silence show performance act trick tricks routine pace tempo speed energy fun ' +
      'vibe mood attitude style groove drama applause ovation welcome greeting hello goodbye bye thanks compliment answer reply promise word words thought thoughts idea ideas chance turnaround ' +
      'seat stand side sides way direction circle circles line shape shapes letter letters number numbers air space room floor ground spot place seatbelt balance attention eye contact').split(' '));
    // everyday physical objects (for the verbs that also have plain body uses); plural s is stripped before the lookup
    const THINGS = new Set(('ball book cup mug glass bottle hat cap crown glasses sunglasses headphones headset earphones earbuds phone laptop computer tablet keyboard mouse pen pencil ' +
      'paper box bag backpack toy teddy doll umbrella chair stool table desk door window drawer fridge jar can bowl plate spoon fork knife apple banana cookie cake pizza sandwich snack food ' +
      'water juice coffee tea milk drink guitar piano drum flower plant stick rock stone blanket pillow scarf coat jacket shirt shoe sock glove mask lamp candle remote camera key coin card dice ' +
      'bone carrot fish balloon kite bat racket rope basket bucket broom mirror clock watch ring necklace sword wand flag gift present envelope newspaper magazine picture photo tv television ' +
      'radio speaker microphone mic curtain blinds cabinet cupboard closet oven trunk suitcase wallet purse hammer tool brush comb toothbrush towel soap thing object item stuff frisbee ' +
      'puzzle block lego crayon marker notebook folder bookshelf shelf couch sofa bed bench ladder bike skateboard helmet').split(' '));
    const NT_DET = new Set(['the', 'a', 'an', 'that', 'this', 'those', 'these', 'some', 'my', 'your', 'his', 'her', 'their', 'our', 'its', 'another']);
    const NT_IOBJ = new Set(['me', 'him', 'her', 'us', 'them', 'it']);
    // words that end a noun phrase (prepositions, particles, adverbs, conjunctions, verbs that follow an object)
    const NT_STOP = new Set(('now then and or but nor so in on at to for with from of up down over out off away back again please once twice thrice here there together toward towards ' +
      'into onto like as while until till before after by near next beside behind above below under around through across along against between right_now just also too still ' +
      'closed open shut high low tight wide still steady straight forward forwards backward backwards upward upwards downward downwards sideways if when that which who is are was be ' +
      'stay hold keep let make do go come return if because than not no yes okay ok').split(' '));
    const NT_LY_NOUN = /^(belly|jelly|butterfly|lily|family|dolly|trolley|holly|fly|bully|ally)$/;
    // verb lemma at w[i]: {n: tokens used, strong, part: a particle it still needs after the object (pick .. up)}
    function ntVerb(w, i) {
      const a = w[i], b = w[i + 1] || '';
      const is = (re) => re.test(a);
      if (is(/^pick(s|ed|ing)?$/)) return b === 'up' ? {n: 2, strong: true} : {n: 1, strong: true, part: /^up$/};
      if (is(/^(put|puts|putting)$/)) return /^(on|down|back|away)$/.test(b) ? {n: 2, strong: true} : {n: 1, strong: true, part: /^(on|down|back|away)$/};
      if (is(/^(take|takes|taking|took|taken)$/)) return /^(off|out)$/.test(b) ? {n: 2, strong: true} : {n: 1, strong: true, part: /^(off|out|away)$/};
      if (is(/^(grab|grabs|grabbed|grabbing|wear|wears|wearing|wore|fetch|fetches|fetched|fetching|carry|carries|carried|carrying|bring|brings|bringing|brought|drink|drinks|drinking|eat|eats|eating)$/)) return {n: 1, strong: true};
      if (is(/^(play|plays|played|playing)$/)) return b === 'with' ? {n: 2, strong: true} : null;
      if (is(/^(hand|hands|handed|handing)$/)) return /^(me|him|her|us|them|over)$/.test(b) ? {n: 1, strong: true} : null;
      if (is(/^(sit|sits|sitting)$/)) return /^(on|in)$/.test(b) ? {n: 2, strong: false} : null;
      if (is(/^(hold|holds|holding|held|open|opens|opened|opening|close|closes|closed|closing|shut|shuts|shutting|throw|throws|throwing|threw|toss|tosses|tossed|tossing|catch|catches|catching|caught|give|gives|giving|gave|slide|slides|sliding|lift|lifts|lifted|lifting|show|shows|showing|use|uses|using|read|reads|reading|kick|kicks|kicking|bounce|bounces|bouncing)$/)) return {n: 1, strong: false};
      return null;
    }
    function noThingHere(desc) {
      let props = null;
      try { const m = get('motion'); props = m && typeof m.props === 'function' ? m.props() : null; } catch (e) { props = null; }
      if (!Array.isArray(props) || props.length) return null;           // the scene characters (the muse) use sceneRoute
      const low = ' ' + String(desc || '').toLowerCase().replace(/[’`]/g, "'").replace(/[.,;:!?()—–]+/g, ' | ').replace(/[^a-z'| ]+/g, ' ').replace(/\s+/g, ' ').trim() + ' ';
      if (/ (pretend|pretending|pretends|mime|miming|mimes|imaginary|invisible|make believe|as if|as though|air|act out|acting out|acts out) /.test(low)) return null;
      const w = low.trim().split(' ');
      for (let i = 0; i < w.length; i++) {
        const v = ntVerb(w, i); if (!v) continue;
        let j = i + v.n;
        if (NT_IOBJ.has(w[j]) && NT_DET.has(w[j + 1])) j++;              // give THEM a .., hand ME the ..
        else if (/^(up|over|out|down|off|on|back)$/.test(w[j] || '') && NT_DET.has(w[j + 1])) j++;   // hold UP the .., lift UP my ..
        if (!NT_DET.has(w[j])) continue;
        j++;
        const np = [];
        while (j < w.length && np.length < 6 && w[j] !== '|' && !NT_STOP.has(w[j]) && !(/ly$/.test(w[j]) && !NT_LY_NOUN.test(w[j]))) np.push(w[j++]);
        if (!np.length) continue;
        if (v.part && !v.part.test(w[j] || '')) { if (v.part.source !== '^up$') { /* put / take without a particle: only a known object */ v.strong = false; } else continue; }
        if (np.some(x => BODY_PARTS.test(x) || NOT_THINGS.has(x))) continue;      // the user's own body, a gesture, an abstract thing
        const head = np[np.length - 1], sing = head.length > 3 && /s$/.test(head) && !/ss$/.test(head) ? head.slice(0, -1) : head;
        if (!v.strong && !(THINGS.has(head) || THINGS.has(sing))) continue;         // hold / open / lift / give ..: known objects only
        return {ok: false, status: 'no_such_thing', ran: false, reason: `there is no ${np.join(' ')} around you (you have no objects here), so nothing moved on screen: tell him that plainly in a few words of your own. ` +
          'Anything he asks of your own body alone, also acting something out with it, is a normal request you can do: never ask him to say a special word for it.'};
      }
      return null;
    }
    /* sceneRoute(description): a do_action that is about the things around it (the muse's scene) — the keyframe planner
       cannot move a prop or change what it holds / wears / sits on (harness 806886b: "pick up the headphones" via
       do_action played a keyframe mime, holding nothing). In order: the user's words fully understood with a scene step -> the user's
       steps run (expand, >= 1 step); the request was already run from the user's words -> already under way; the description
       itself fully understood with a scene step -> those steps; the description names a thing in the scene -> refused
       with the hint to use `object`; else null (the planner). */
    function sceneRoute(desc) {
      const M = get('sceneMatcher');
      if (!M || typeof M.parse !== 'function' || !d.runStep) return null;
      // hybrid: a fast matcher already started this request from the user's words -> the model's restatement adds nothing
      if (coveredNow()) return {ok: true, status: 'starting', already: true, did: 'starting now — already begun from his own words as he said them (that IS his request: give him a brief natural cue, never say nothing new started)'};
      const n = expand(1);
      if (n) return {ok: true, status: 'starting', did: `doing what he asked, ${n} step${n > 1 ? 's' : ''} in order`, next: 'It starts right away: give him a brief natural cue now.'};
      const u = V.utts[V.utts.length - 1];
      if (u && u.expanded && now() - (u.finalAt || u.t0) < 30000) return {ok: true, status: 'starting', did: 'under way now — already started from his words a moment ago (his request is happening: a brief natural cue)'};
      let hits = [];
      try { hits = M.parse(desc, {final: true, nowMs: now()}) || []; } catch (e) { hits = []; }
      let ok = false;
      try { ok = hits.some(h => h && h.cmd && h.cmd.action === 'object') && (typeof M.understood !== 'function' || M.understood(desc)); } catch (e) { ok = false; }
      if (ok) {
        for (const h of hits) { try { d.runStep(h.cmd, {text: desc, rule: h.rule}); } catch (e) {} }
        log({vt: 'steps_desc', n: hits.length, text: desc.slice(0, 160)});
        return {ok: true, status: 'starting', did: `doing it, ${hits.length} step${hits.length > 1 ? 's' : ''} in order`, next: 'It starts right away: give him a brief natural cue now.'};
      }
      let props = [];
      try { const m = get('motion'); props = m && typeof m.props === 'function' ? m.props() : []; } catch (e) { props = []; }
      let aliases = {};                                     // the scene file's aliases (headset, earphones, computer, ...)
      try { for (const a of get('sceneAliases') || []) if (a && a.id) aliases[a.id] = (a.aliases || []).map(String); } catch (e) { aliases = {}; }
      const low = ' ' + desc.toLowerCase().replace(/[^a-z ]+/g, ' ') + ' ';
      const named = props.find(p => [p.id, ...String(p.name || '').split(/\s+/), ...(aliases[p.id] || [])].some(w => w && w.length > 2 && !/^(pink|silver|blue|red|yellow|green|little|toy|pot|one|red one|blue one|green one|yellow one)$/.test(String(w).toLowerCase()) && low.includes(' ' + String(w).toLowerCase() + ' ')));
      // the planner can plan WITH the things (motion.things(): the scene's live list) -> plan it; off (objectPlans
      // false, or a motion layer without things()) = the no-scene refusal with the hint to use `object`
      let canPlan = false;
      try { const m = get('motion'); canPlan = (d.objectPlans === undefined || bool('objectPlans')) && !!m && typeof m.things === 'function'; } catch (e) { canPlan = false; }
      if (named && canPlan) return {plan: true, thing: named.id};
      if (named) return {ok: false, status: 'use_object', reason: `the ${named.name || named.id} is one of the things around you: move it with the object tool (one call per step, in order), not ${ACTION[0]}`};
      return null;
    }
    // a reply (and a corrected tool call) after a refusal: the call's own response may already have spoken (parallel
    // style), and then nothing reads the result; dropped if the user speaks or the model stays busy (no stale replies)
    function followUp(note) { prompt(note, {respond: !heSpeaking(), maxWaitMs: 2500}); }
    // the user's own words for the server library's second try (the newest utterance, final or the deltas so far; with the bare
    // "could you?" half of a split question joined): only what the guard already passed
    function wordsOf() {
      const ts = requestTexts(V.utts, now());
      const t = ts.length ? ts[ts.length - 1] : '';
      return t && !isHypothetical(t).hyp && now() - ((V.utts[V.utts.length - 1] || {}).t0 || -1e12) < 30000 ? {words: t.slice(0, 400)} : {};
    }
    function refuse(desc, g, when) {
      V.blocks.push({t: Math.round(now()), description: desc, why: g.why, text: String(g.text || '').slice(0, 160), when});
      log({vt: 'guard', description: desc, why: g.why, when});
      return {ok: false, status: 'not_a_request', reason: 'his words were a question about an imagined situation, not a request, ' +
        'so nothing was done: answer his question in words'};
    }
    function resolveHold(h, why) {
      const i = V.holds.indexOf(h); if (i < 0) return;
      V.holds.splice(i, 1); T.clear(h.timer);
      const held = Math.round(now() - h.t);
      const g = guardCheck(V.utts, h.desc, now());
      if (g.block) {
        { const k = TH(); if (k) { try { k.end('vt:' + h.id, 'not a request'); } catch (e) {} } }
        refuse(h.desc, g, 'after ' + why + ' (' + held + ' ms)');
        prompt(`His words were a question about an imagined situation, not a request, so the action (${h.desc.slice(0, 120)}) was not done. Answer him in words, in your own way.`,
          {respond: !heSpeaking(), maxWaitMs: 3000});
        return;
      }
      const P = get('perf');
      let r;
      if (h.later) { try { r = h.later(held); } catch (e) { r = {ok: false, reason: String(e && e.message || e).slice(0, 80)}; } }   // a gated `object` call
      else r = route(h.desc, h.move, h.talk, h.info, held);
      { const k = TH(); if (k) { try { k.end('vt:' + h.id, 'released: ' + (r && r.status || (r && r.ok === false ? 'refused' : 'done'))); } catch (e) {} } }   // P.call's own key keeps it on
      log({vt: 'released', id: h.id, held_ms: held, why, status: r && r.status, perform: r && r.id});
      if (r && r.status === 'use_object') followUp(`The request (${h.desc.slice(0, 120)}) was not done: ${r.reason}. Call object for it now, one call per step, in order.`);
      else if (r && r.ok === false) prompt(`The action he asked for (${h.desc.slice(0, 120)}) could not be started (${String(r.reason || 'unknown').slice(0, 120)}). Tell him honestly in a few words.`, {respond: !heSpeaking(), maxWaitMs: 3000});
    }

    /* gate(description, later): the same guard for a body call that is not do_action (the muse's `object` tool): a what-if
       -> the refusal result; the user's words still on their way -> held ({ok, status:'preparing'}), later(held_ms) runs on
       release unless the words turn out to be a what-if; null = go ahead now */
    function gate(description, later) {
      const desc = String(description || '').replace(/\s+/g, ' ').trim().slice(0, 200);
      const g = guardCheck(V.utts, desc, now());
      if (g.block) return refuse(desc, g, 'at once');
      if (!txPending() || typeof later !== 'function') return null;
      const h = {id: 'h' + (++V.n), desc, later, t: now()};
      h.timer = T.set(() => resolveHold(h, 'timeout'), HOLD_MS);
      V.holds.push(h);
      log({vt: 'hold', id: h.id, description: desc});
      return {ok: true, status: 'preparing', id: h.id, did: 'about to do it (not started yet)'};
    }

    /* expand(): a request with several steps for the things around it — the voice model calls `object` for ONE step only
       (eval: 5 of 6 multi-step turns; harness smoke: "grab your headphones and put them on" ->
       pick_up only). On its object call the page matcher reads the user's final words (sceneMatcher(): fully understood only,
       never a what-if) and, when they hold two or more steps with at least one for a thing in the scene, every step runs
       in order through runStep (the caption matcher's path: IntentDedupe then drops the model's own calls of the same
       steps). Once per utterance. -> the number of steps run (0 = nothing to add). */
    function expand(minSteps) {
      const M = get('sceneMatcher');
      if (!M || typeof M.parse !== 'function' || !d.runStep) return 0;
      const u = V.utts[V.utts.length - 1];
      V.sceneCall = {u, t: now()};                            // a scene call of this request (continuations below)
      if (coveredNow()) return 0;                             // hybrid: the fast matchers already ran the user's words' steps
      // realtime marks the user's turn's end and then sends the final words; Gemini / the server-relayed voice only stream deltas
      if (!u || !u.text || u.expanded || (u.stopped && !u.final) || now() - (u.finalAt || u.t0) > 30000) return 0;
      // a pause can split one request into two turns ("open it up." / "and do some work."): read them together
      const prev = V.utts.length > 1 ? V.utts[V.utts.length - 2] : null;
      const joined = isCont(u.text) && prev && prev.text && !prev.expanded && u.t0 - (prev.t1 || prev.t0) < CONT_MS;
      const text = joined ? prev.text + ' ' + u.text : u.text;
      let hits = [];
      try { hits = M.parse(text, {final: true, nowMs: now()}) || []; } catch (e) { return 0; }
      if (hits.length < (minSteps || 2) || !hits.some(h => h && h.cmd && h.cmd.action === 'object')) return 0;
      try { if (typeof M.understood === 'function' && !M.understood(text)) return 0; } catch (e) { return 0; }
      if (isHypothetical(text).hyp) return 0;
      u.expanded = true; if (joined) prev.expanded = true;
      let n = 0;
      for (const h of hits) {
        try { if (typeof M.commit === 'function') M.commit(h, now()); d.runStep(h.cmd, {text, rule: h.rule}); n++; } catch (e) {}
      }
      V.expanded.push({t: Math.round(now()), n, text: text.slice(0, 160), steps: hits.map(h => h.cmd.verb || h.cmd.name), joined: joined || undefined});
      if (V.expanded.length > 30) V.expanded.splice(0, 15);
      log({vt: 'steps', n, text: u.text.slice(0, 160)});
      return n;
    }

    /* the second half of a split request arrives AFTER the model already made its scene call for the first half
       ("take the headphones off." -> object(take_off); then "and put them back where they were."): the continuation's own
       steps run (fully understood, never a what-if); the model's own later call of them is then a duplicate */
    function contExpand(u) {
      const M = get('sceneMatcher');
      if (!M || typeof M.parse !== 'function' || !d.runStep || u.expanded) return 0;
      let hits = [];
      try { hits = M.parse(u.text, {final: true, nowMs: now()}) || []; } catch (e) { return 0; }
      if (!hits.length || !hits.every(h => h && h.cmd && /^(object|move)$/.test(h.cmd.action))) return 0;
      try { if (typeof M.understood === 'function' && !M.understood(u.text)) return 0; } catch (e) { return 0; }
      if (isHypothetical(u.text).hyp) return 0;
      u.expanded = true;
      for (const h of hits) { try { if (typeof M.commit === 'function') M.commit(h, now()); d.runStep(h.cmd, {text: u.text, rule: h.rule}); } catch (e) {} }
      V.expanded.push({t: Math.round(now()), n: hits.length, text: u.text.slice(0, 160), steps: hits.map(h => h.cmd.verb || h.cmd.name), continuation: true});
      log({vt: 'steps_cont', n: hits.length, text: u.text.slice(0, 160)});
      return hits.length;
    }

    // ------------------------------------------------------------------------------------------- behaviours
    function B() { const b = get('behaviors'); return b && typeof b.start === 'function' ? b : null; }
    function blist() { const b = B(); if (!b || typeof b.list !== 'function') return []; try { return b.list() || []; } catch (e) { return []; } }
    function bfind(name) {
      const q = String(name || '').toLowerCase().trim(), qs = slug(q);
      const L = blist();
      return L.find(x => String(x.name || '').toLowerCase() === q) || L.find(x => String(x.id) === q)
        || L.find(x => slug(x.name) === qs) || L.find(x => q && String(x.name || '').toLowerCase().includes(q))
        || L.find(x => q && String(x.instruction || '').toLowerCase().includes(q)) || null;
    }
    function codeRequest(kind, name, instruction, priority, prev) {
      const id = 'b' + (++V.n), req = {id, kind, name, instruction, priority, t: now()};
      req.timer = T.set(() => { if (V.pendingB.get(id) === req) { V.pendingB.delete(id); behaviourFailed(req, 'the behaviour writer did not answer'); } }, CODE_WAIT_MS);
      V.pendingB.set(id, req);
      const msg = {type: 'behavior_code', id, instruction, name};
      const w = hisWords(); if (w) msg.his_words = w;
      if (prev && prev.code) msg.prev_code = prev.code;
      if (prev && prev.instruction) msg.prev_instruction = prev.instruction;
      sendCtl(msg);
      return req;
    }
    function behaviourFailed(req, why) {
      log({vt: 'behavior_fail', name: req.name, why});
      const verb = req.kind === 'change' ? 'changed' : 'started';
      prompt(`The behaviour he asked for (${String(req.instruction).slice(0, 160)}) could not be ${verb} (${String(why).slice(0, 140)}). Tell him honestly in a few words.`,
        {respond: !heSpeaking(), maxWaitMs: 4000});
    }
    // the program layer's own server round trip (Behaviors.request(instruction, {name, priority, edit}) -> Promise<{ok, name,
    // error}>; it sends {type:'behavior_code'} and takes the reply in Behaviors.onCode)
    function viaRequest(b, kind, name, instruction, o) {
      const req = {id: 'r' + (++V.n), kind, name, instruction, t: now(), via: 'request'};
      V.pendingB.set(req.id, req);
      let pr;
      try { pr = b.request(instruction, o); } catch (e) { pr = Promise.resolve({ok: false, error: String(e && e.message || e)}); }
      Promise.resolve(pr).then(r => {
        if (V.pendingB.get(req.id) !== req) return;                // stopped meanwhile
        V.pendingB.delete(req.id);
        if (!r || !r.ok) { behaviourFailed(req, (r && (r.error || r.why)) || 'the program could not be written'); return; }
        req.name = r.name || name;
        log({vt: 'behavior_on', name: req.name, kind, ms: Math.round(now() - req.t)});
        const state = kind === 'change' ? 'changed and running' : 'running';
        prompt(`The behaviour ${req.name} he asked for is ${state} now (${String(instruction).slice(0, 160)}).`, {respond: false});
      }, e => { if (V.pendingB.delete(req.id)) behaviourFailed(req, String(e && e.message || e)); });
    }
    // games: the user's own latest words ride along to the behaviour writer — the voice model's restatement can lose what
    // kind of rule it was ("copy my X and Y" came back as "whenever the user does X and Y")
    function hisWords() {
      const t = now();
      return V.utts.filter(u => u.final && u.text && t - (u.finalAt || u.t1 || t) < 30000).slice(-2).map(u => String(u.text).trim()).filter(Boolean).join(' / ').slice(-400);
    }
    function startBehavior(a) {
      const b = B();
      const instruction = String(a && a.instruction || '').trim().slice(0, 600);
      if (!instruction) return {ok: false, reason: 'no instruction given'};
      if (!b) return {ok: false, reason: 'lasting behaviours are not available on this page yet, so nothing was started'};
      let name = slug(a && a.name ? a.name : instruction);
      if (bfind(name) && String(bfind(name).name).toLowerCase() === name) name = name + '_' + (blist().length + 1);
      const priority = Number.isFinite(+(a && a.priority)) ? clamp(Math.round(+a.priority), 1, 10) : 5;
      if (typeof b.request === 'function') { viaRequest(b, 'start', name, instruction, {name, priority, words: hisWords()}); return {ok: true, status: 'writing', name, next: 'It is being set up: say something short and natural now; a silent note will tell you when it is running.'}; }
      codeRequest('start', name, instruction, priority, null);
      return {ok: true, status: 'writing', name, next: 'It is being set up: say something short and natural now; a silent note will tell you when it is running.'};
    }
    function changeBehavior(a) {
      const b = B();
      if (!b) return {ok: false, reason: 'lasting behaviours are not available on this page yet'};
      const instruction = String(a && a.instruction || '').trim().slice(0, 600);
      const f = bfind(a && a.name);
      if (!f) return {ok: false, reason: `no running behaviour called ${String(a && a.name || '').slice(0, 40)}`, running: blist().map(x => x.name)};
      if (!instruction) return {ok: false, reason: 'no new instruction given'};
      if (typeof b.request === 'function') { viaRequest(b, 'change', f.name, instruction, {name: f.name, edit: true, words: hisWords()}); return {ok: true, status: 'rewriting', name: f.name, next: 'It is being changed: say something short and natural now; a silent note will tell you when the change is running.'}; }
      let code = null;
      try { const g = typeof b.get === 'function' ? b.get(f.name) : null; code = g && g.code || (typeof b.code === 'function' ? b.code(f.name) : null) || f.code || null; } catch (e) {}
      codeRequest('change', f.name, instruction, f.priority, {code, instruction: f.instruction});
      return {ok: true, status: 'rewriting', name: f.name, next: 'It is being changed: say something short and natural now; a silent note will tell you when the change is running.'};
    }
    function stopBehavior(a) {
      const b = B();
      if (!b || typeof b.stop !== 'function') return {ok: false, reason: 'lasting behaviours are not available on this page yet'};
      const q = String(a && a.name || '').toLowerCase().trim();
      const all = !q || q === 'all' || q === 'everything' || q === 'all of them';
      const before = blist();
      const f = all ? null : bfind(q);
      if (!all && !f) return {ok: false, reason: `no running behaviour called ${q.slice(0, 40)}`, running: before.map(x => x.name)};
      for (const [id, req] of V.pendingB) if (all || req.name === (f && f.name)) { T.clear(req.timer); V.pendingB.delete(id); }
      try { b.stop(all ? 'all' : f.id != null ? f.id : f.name); } catch (e) { return {ok: false, reason: 'it could not be stopped (' + String(e && e.message || e).slice(0, 80) + ')'}; }
      const after = new Set(blist().map(x => x.name));
      return {ok: true, stopped: before.filter(x => !after.has(x.name)).map(x => x.name)};
    }
    function listBehaviors() {
      const b = B();
      if (!b) return {ok: true, behaviors: [], note: 'none running (lasting behaviours are not available on this page yet)'};
      const t = now();
      const pend = [...V.pendingB.values()].map(r => ({name: r.name, does: r.instruction, status: r.kind === 'change' ? 'being changed' : 'being set up'}));
      return {ok: true, behaviors: blist().map(x => ({name: x.name, does: x.instruction, priority: x.priority,
        running_s: Number.isFinite(+x.since) ? Math.max(0, Math.round((t - +x.since) / 100) / 10) : undefined, errors: x.errors || undefined})).concat(pend)};
    }
    /* tv: the wall TV (docs/TV-AGENT.md). tv_show / tv_clear go to the server's TV agent ({type:'tv_show'|'tv_clear'});
       its {type:'tv_state'} answers become silent notes so the voice model always knows what the TV shows.
       d.onTv(name, args) is TEACH's hook (walk to the TV station, set the camera view): called before the request is sent. */
    function tvShow(a) {
      const concept = String(a && a.concept != null ? a.concept : '').replace(/\s+/g, ' ').trim();
      if (!concept) return {ok: false, reason: 'no concept given'};
      const mode = a && a.mode === 'update' ? 'update' : 'new';
      if (d.onTv) { try { d.onTv('tv_show', {concept, mode}); } catch (e) {} }
      V.tvAsk = {t: now(), concept};
      sendCtl({type: 'tv_show', concept: concept.slice(0, 400), mode});
      log({vt: 'tv_show', concept: concept.slice(0, 120), mode});
      return {ok: true, status: 'drawing', next: 'It is being drawn on the TV now. Say something short and natural (what you are putting up); a silent note will tell you what the TV shows once it is up.'};
    }
    function tvClear() {
      if (d.onTv) { try { d.onTv('tv_clear', {}); } catch (e) {} }
      sendCtl({type: 'tv_clear'});
      return {ok: true, status: 'cleared', did: 'the TV picture is off'};
    }
    function onTvState(m) {
      V.tv = m;
      const desc = String(m.description || m.desc || '').slice(0, 220);
      const st = m.status;
      if (m.source === 'review' || m.source === 'upgrade') return true;   // its own self-review / careful version of the same picture: nothing to say
      // integ: the picture the user asked for is up: the voice model had said only a short "putting it up" line and then
      // waited; it now explains it (unless the user is talking)
      // r2-tv: a change says in one line WHAT changed (m.change, the TV agent's own note of the change)
      const chg = m.change ? String(m.change).slice(0, 160) : '';
      if (st === 'shown' || (st === 'updated' && m.source === 'tool')) prompt(`${chg ? 'The TV changed: ' + chg + '. ' : ''}The TV now shows: ${desc}. Now explain it to him in a few short sentences, referring to what is on the TV and answering what he asked; then stop.`, {respond: !heSpeaking(), maxWaitMs: 3000});
      else if (st === 'updated' && rated('tv', 'tap', chg ? 2500 : 6000)) prompt(chg ? `The TV picture just changed to follow him: ${chg}. Mention it in a few words if it fits.` : `The TV picture just updated itself to follow the conversation; it now shows: ${desc}.`, {respond: false});
      else if (st === 'failed' && m.source !== 'tap') prompt(`The TV could not draw that (${String(m.error || 'unknown').slice(0, 120)}); it still shows: ${desc}. Tell him honestly in a few words if it matters.`, {respond: !heSpeaking(), maxWaitMs: 3000});
      return true;
    }
    function onHub(m) {
      if (m && m.type === 'tv_state') return onTvState(m);    // tv
      if (!m || m.type !== 'behavior_code') return false;
      let req = m.id != null ? V.pendingB.get(String(m.id)) : null;
      if (!req) for (const r of V.pendingB.values()) if (r.name === m.name) { req = r; break; }
      if (!req) return false;                              // not ours (another requester's, or late after a timeout)
      V.pendingB.delete(req.id); T.clear(req.timer);
      if (!m.ok || !m.code) { behaviourFailed(req, m.error || 'the program could not be written'); return true; }
      const b = B();
      if (!b) { behaviourFailed(req, 'lasting behaviours are not available on this page'); return true; }
      try {
        if (req.kind === 'change') b.change(req.name, {code: m.code, instruction: req.instruction});
        else b.start({name: req.name, instruction: req.instruction, code: m.code, priority: req.priority});
      } catch (e) { behaviourFailed(req, String(e && e.message || e).slice(0, 120)); return true; }
      log({vt: 'behavior_on', name: req.name, kind: req.kind, ms: Math.round(now() - req.t)});
      const state = req.kind === 'change' ? 'changed and running' : 'running';
      prompt(`The behaviour ${req.name} he asked for is ${state} now (${String(req.instruction).slice(0, 160)}).`, {respond: false});
      return true;
    }

    // ------------------------------------------------------------------------------------------- look / style / gestures
    function look(a, info) {
      const raw = a && a.target;
      let key = lookKey(raw);
      const m = get('motion');
      let prop = null;
      if (!key) {
        let props = [];
        try { props = m && typeof m.props === 'function' ? m.props() : []; } catch (e) {}
        prop = findProp(props, raw);
        if (!prop) return {ok: false, reason: `there is nothing called ${String(raw || '').slice(0, 40)} around you to look at`};
        key = prop.id;
      }
      const sec = a && a.seconds != null && Number.isFinite(+a.seconds) ? +a.seconds : (a && a.hold ? 0 : null);
      const cmd = {action: 'look', target: key};
      if (sec === 0) cmd.hold = true;
      else if (sec != null) cmd.duration_ms = Math.round(clamp(sec, 0.2, 120) * 1000);
      if (key === 'around') { delete cmd.duration_ms; cmd.repeat = 3; }
      const r = d.runCommand ? d.runCommand(cmd, info) : null;
      const ok = r === true || (r && typeof r === 'object' ? r.ok !== false && !r.dup : !!r);
      return ok ? {ok: true, did: r && r.did ? r.did : prop ? 'looking at the ' + prop.name : 'looking ' + key} : {ok: false, reason: (r && r.reason) || 'it could not look there'};
    }
    function setStyle(a) {
      const st = {};
      for (const k of ['energy', 'mirroring', 'expressiveness']) {
        if (!a || !(k in a) || a[k] === undefined) continue;
        const v = a[k] === null || a[k] === 'auto' ? -1 : +a[k];
        if (!Number.isFinite(v)) continue;
        st[k] = v < 0 ? null : Math.round(clamp(v, 0, 1) * 100) / 100;
      }
      if (!Object.keys(st).length) return {ok: false, reason: 'nothing to change was given'};
      Object.assign(V.style, st);
      sendCtl(Object.assign({type: 'style'}, st));
      const mode = String(get('reactMode') || 'rules');
      const now_ = {}; for (const [k, v] of Object.entries(V.style)) now_[k] = v === null ? 'automatic' : v;
      if (mode === 'unified') return {ok: true, style: now_};
      return {ok: false, stored: true, style: now_, reason: 'your listening and talking movements only take a style while the learned listener runs, and it is off right now: it is kept for when it is on'};
    }
    function learnGesture(a) {
      const G = get('gestures');
      const name = String(a && a.name || '').trim().slice(0, 40);
      if (!name) return {ok: false, reason: 'no name given'};
      if (!G || typeof G.learn !== 'function') return {ok: false, reason: 'learning his gestures is not available on this page yet, so it was not learned'};
      const how = a && a.how === 'next' ? 'next' : 'now';
      const opt = {from: how === 'next' ? 'next' : 'last', seconds: Number.isFinite(+(a && a.seconds)) && +a.seconds > 0 ? clamp(+a.seconds, 0.5, 10) : 3, counter: !!(a && a.counterexample)};
      let r;
      try { r = G.learn(name, opt); } catch (e) { return {ok: false, reason: 'it could not be learned (' + String(e && e.message || e).slice(0, 80) + ')'}; }
      const done = v => {
        if (!v || v.ok === false) { prompt(`Learning his gesture ${name} did not work (${String(v && (v.reason || v.why || v.error) || 'no clear example').slice(0, 120)}). Tell him honestly in a few words.`, {respond: !heSpeaking(), maxWaitMs: 4000}); return; }
        const what = opt.counter ? `what his gesture ${name} is not` : `his gesture ${name}`, n = +v.examples || 0;
        const count = n ? ` (${n} example${n > 1 ? 's' : ''} so far)` : '';
        prompt(`You learned ${what}${count}; you will notice it from now on. Let him know briefly, in your own words.`, {respond: !heSpeaking(), maxWaitMs: 4000});
      };
      if (r && typeof r.then === 'function') {
        r.then(done, e => done({ok: false, reason: String(e && e.message || e)}));
        return {ok: true, status: how === 'next' ? 'watching' : 'learning', name, next: how === 'next'
          ? 'It is watching for his next try: let him know he can go ahead; a silent note will say when it is learned.'
          : 'A silent note will say when it is learned.'};
      }
      if (!r || r.ok === false) return {ok: false, reason: String(r && (r.reason || r.why || r.error) || 'no clear example').slice(0, 120)};
      return {ok: true, name, kind: r.kind, examples: r.examples, counterexample: opt.counter || undefined};
    }

    /* see_camera: ONE camera frame for a question that depends on what the camera shows — sent as context
       (image item) before the tool result, so the call's follow-up reply sees it; at most one per 3 s */
    function seeCamera(a) {
      if (!d.sendFrame || !d.cameraFrame) return {ok: false, reason: 'no camera on this page'};
      if (now() - (V.lastFrame || -1e12) < 3000) return {ok: true, status: 'sent', next: 'The frame from a moment ago is the newest: answer from it.'};
      let img = null;
      try { img = d.cameraFrame(); } catch (e) { img = null; }
      if (!img) return {ok: false, reason: 'the camera has no picture right now (it may be off or not ready)'};
      const about = String(a && a.about || '').trim().slice(0, 80);
      let ok = false;
      try { ok = d.sendFrame(img, `His camera right now${about ? ' (you wanted to see: ' + about + ')' : ''}.`) !== false; } catch (e) { ok = false; }
      if (!ok) return {ok: false, reason: 'the frame could not be sent'};
      V.lastFrame = now(); V.frames = (V.frames || 0) + 1;
      return {ok: true, status: 'sent', next: 'The frame is with you now: answer him from what it shows, in your own words.'};
    }
    // attention_rule: the user's standing instructions / feedback -> the camera judge (window.Attention)
    function attentionRule(a) {
      const A = get('attention');
      const text = String(a && a.text || '').trim().slice(0, 300);
      if (!text) return {ok: false, reason: 'no instruction given'};
      const fb = a && a.kind === 'feedback';
      const fn = A && (fb ? A.feedback : A.instruct);
      if (typeof fn !== 'function') return {ok: false, reason: 'the camera watcher is not running on this page, so it was not kept'};
      let r;
      try { r = fb ? A.feedback(text) : A.instruct(text); } catch (e) { return {ok: false, reason: 'it could not be kept (' + String(e && e.message || e).slice(0, 80) + ')'}; }
      if (r && r.ok === false) return {ok: false, reason: String(r.reason || r.why || r.error || 'not kept').slice(0, 120)};
      return {ok: true, kind: fb ? 'feedback' : 'standing', kept: text};
    }

    // ------------------------------------------------------------------------------------------- dispatch
    /* early(move): do_action's `move` object completed in the streamed arguments (voice.js early phase) — the named move
       plays now, unless the user's words are still on their way or are a what-if (then the 'done' call decides); the 'done'
       call's own run of it is a duplicate (IntentDedupe) */
    function early(move) {
      if (!move || !move.name || !d.runLibrary || txPending()) return false;
      if (guardCheck(V.utts, '', now()).block) return false;
      let r = null;
      try { r = d.runLibrary('', move, {early: true}); } catch (e) { r = null; }
      V.early.push({t: Math.round(now()), move: move.name, ran: r && r.ran}); if (V.early.length > 30) V.early.splice(0, 15);
      return !!(r && r.ran);
    }
    function call(name, args, phase, info) {
      if (phase === 'early' && ACTION.includes(name)) { early(args && args.move); return null; }
      if (phase && phase !== 'done') return null;
      if (!ACTION.includes(name) && !NAMES.includes(name)) return null;
      const t0 = now(); let r;
      try {
        args = args && typeof args === 'object' ? args : {};
        switch (name) {
          case 'do_action': case 'perform_action': r = action(args, info); break;
          case 'start_behavior': r = startBehavior(args); break;
          case 'change_behavior': r = changeBehavior(args); break;
          case 'stop_behavior': r = stopBehavior(args); break;
          case 'list_behaviors': r = listBehaviors(); break;
          case 'look': r = look(args, info); break;
          case 'set_style': r = setStyle(args); break;
          case 'learn_gesture': r = learnGesture(args); break;
          case 'attention_rule': r = attentionRule(args); break;
          case 'see_camera': r = seeCamera(args); break;
          case 'tv_show': r = tvShow(args); break;              // tv
          case 'tv_clear': r = tvClear(); break;
        }
      } catch (e) { r = {ok: false, reason: 'error: ' + String(e && e.message || e).slice(0, 100)}; }
      if (phase !== 'early' || (r && r.ok !== false)) V.lastToolAt = now();      // fixes: a call was made in this turn (claimCheck)
      const us = Math.round((now() - t0) * 1000);
      V.us.push(us); if (V.us.length > 400) V.us.splice(0, 200);
      V.calls.push({name, ok: !r || r.ok !== false, status: r && r.status, us, t: Math.round(t0)}); if (V.calls.length > 200) V.calls.splice(0, 100);
      return r || {ok: false, reason: 'not handled'};
    }

    // ------------------------------------------------------------------------------------------- events -> voice model
    function rated(kind, key, ms) {
      const k = kind + '|' + key, t = now();
      if (t - (V.lastNote[k] || -1e12) < ms) return false;
      V.lastNote[k] = t; return true;
    }
    function dup(ev) {   // the same event through two routes (a module's own onEvent and the page's event hook)
      const k = [ev.type, ev.name || ev.cue || ev.what || '', Math.round(+ev.t * 10) || ''].join('|'), t = now();
      for (const [kk, at] of V.seen) if (t - at > 3000) V.seen.delete(kk);
      if (V.seen.has(k)) return true;
      V.seen.set(k, t); return false;
    }
    function quietNow() { return !heSpeaking() && !bool('avatarSpeaking') && now() - (+get('lastUserSpeech') || -1e12) > 700; }
    function onEvent(ev) {
      if (!ev || typeof ev !== 'object') return {noted: false, why: 'no event'};
      // the program layer's own shapes (behavior.js): {type:'rule_moment', name: program, instruction, event, data, reply}
      // and {type:'behavior', phase:'error'|'end', name, error?}
      if (ev.type === 'rule_moment') ev = {type: 'rule', name: ev.event, data: ev.data, respond: ev.reply === false ? false : undefined, t: ev.t,
        program: {name: ev.name, instruction: ev.instruction}};
      else if (ev.type === 'behavior' && (ev.phase === 'error' || (ev.phase === 'end' && ev.error)))
        ev = {type: ev.phase === 'end' ? 'stopped' : 'error', program: {name: ev.name}, error: ev.error, t: ev.t};
      const type = String(ev.type || ev.kind || '');
      if (dup(ev)) return {noted: false, why: 'duplicate'};
      if (type === 'gesture') {
        const name = String(ev.name || '').slice(0, 40);
        if (!name || (Number.isFinite(+ev.score) && +ev.score < 0.5)) return {noted: false, why: 'weak'};
        if (!rated('gesture', name, RATE.gesture)) return {noted: false, why: 'rate'};
        // a running behaviour that reacts to this gesture owns the reaction: the voice model only learns of it
        const owned = blist().some(b => String(b.instruction || '').toLowerCase().includes(name.toLowerCase()));
        const respond = !owned && !heSpeaking();
        const ask = respond ? ' React to it naturally, in a few words of your own.' : '';
        prompt(`He just made the gesture he taught you as ${name}.${ask}`, {respond, maxWaitMs: 2000, unprompted: true});
        return {noted: true, respond};
      }
      if (type === 'attention') {
        const what = String(ev.what || ev.summary || '').slice(0, 200);
        if (!what) return {noted: false, why: 'empty'};
        if (!rated('attention', '', RATE.attention)) return {noted: false, why: 'rate'};
        const respond = !!ev.worth_reply && !heSpeaking();
        const ask = respond ? ' It is worth a word: react to him naturally, in a few words of your own.' : '';
        const note = `The camera noticed: ${what}.${ask}`;
        if (ev.image && d.sendEvent) { try { d.sendEvent(note, {image: ev.image, respond}); V.notes++; return {noted: true, respond, image: true}; } catch (e) {} }
        prompt(note, {respond, maxWaitMs: 2500, unprompted: true});
        return {noted: true, respond};
      }
      if (type === 'turn_bid') {
        if (ev.bid === 'floor') return {noted: false, why: 'he wants the floor'};      // turnbid.js: the user is about to talk
        const dl = ev.bid === 'reply' ? Math.max(+ev.deliberateness || 0, TURN_BID_MIN) : +ev.deliberateness;
        if (!(dl >= TURN_BID_MIN)) return {noted: false, why: 'below threshold'};
        if (!quietNow()) return {noted: false, why: 'not silent'};
        if (!rated('turn_bid', '', RATE.turn_bid)) return {noted: false, why: 'rate'};
        prompt(`He is signalling that he wants you to talk (${String(ev.cue || 'a cue').replace(/_/g, ' ').slice(0, 40)}). Take the turn now with something natural that fits the conversation.`, {respond: true, maxWaitMs: 1500, unprompted: true});
        return {noted: true, respond: true};
      }
      if (type === 'rule' || type === 'emit' || type === 'moment') {
        const prog = ev.program && typeof ev.program === 'object' ? ev.program : {name: ev.program || ev.behavior || ev.source || ''};
        const pname = String(prog.name || '').slice(0, 40), instr = String(prog.instruction || '').slice(0, 160);
        const name = String(ev.name || ev.event || 'moment').slice(0, 60);
        if (!rated('rule', pname + '|' + name, RATE.rule)) return {noted: false, why: 'rate'};
        const data = ev.data == null ? '' : typeof ev.data === 'string' ? ev.data : JSON.stringify(ev.data);
        const quiet = ev.respond === false || (ev.data && typeof ev.data === 'object' && (ev.data.quiet || ev.data.respond === false));
        const respond = !quiet && !heSpeaking();
        const which = pname ? ` (${pname}${instr ? ': ' + instr : ''})` : '', detail = data ? ` (${data.slice(0, 160)})` : '';
        const ask = respond ? ' Say what fits it to him now, in a few natural words of your own.' : '';
        prompt(`A behaviour you are running${which} reached a moment: ${name}${detail}.${ask}`, {respond, maxWaitMs: 2500, unprompted: true});
        return {noted: true, respond};
      }
      if (type === 'error' || type === 'stopped') {   // a behaviour that crashed and was stopped
        const pname = String((ev.program && ev.program.name) || ev.name || '').slice(0, 40);
        if (!pname || !rated('rule_err', pname, 10000)) return {noted: false, why: 'rate'};
        const err = String(ev.error || ev.reason || 'unknown').slice(0, 120), ask = type === 'stopped' ? ' Tell him honestly in a few words.' : '';
        prompt(`Your behaviour ${pname} stopped because of an error (${err}).${ask}`, {respond: type === 'stopped' && !heSpeaking(), maxWaitMs: 3000});
        return {noted: true};
      }
      return {noted: false, why: 'not ours'};
    }
    function subscribe(obj, kind) {
      if (!obj || V.attached.has(obj)) return false;
      const fn = ev => { try { onEvent(kind && !(ev && ev.type) ? Object.assign({type: kind}, ev) : ev); } catch (e) {} };
      let ok = false;
      try { if (typeof obj.onEvent === 'function') { obj.onEvent(fn); ok = true; } else if (typeof obj.on === 'function') { obj.on(kind || 'event', fn); ok = true; } } catch (e) {}
      if (ok) V.attached.add(obj);
      return ok;
    }
    function attach() {
      return {behaviors: subscribe(get('behaviors'), null), gestures: subscribe(get('gestures'), 'gesture'), attention: subscribe(get('attention'), 'attention')};
    }
    function stats() {
      const us = V.us.slice().sort((a, b) => a - b), p = q => us.length ? us[Math.min(us.length - 1, Math.floor(q * us.length))] : null;
      return {calls: V.calls.slice(-30), holds: V.holds.length, blocks: V.blocks.slice(-10), expanded: V.expanded.slice(-10), pending_behaviors: V.pendingB.size, style: Object.assign({}, V.style),
        notes: V.notes, dispatch_us: {p50: p(0.5), p90: p(0.9), max: us.length ? us[us.length - 1] : null, n: us.length}};
    }
    function reset() { { const k = TH(); if (k) for (const h of V.holds) { try { k.end('vt:' + h.id, 'reset'); } catch (e) {} } } for (const h of V.holds) T.clear(h.timer); V.holds = []; for (const r of V.pendingB.values()) T.clear(r.timer); V.pendingB.clear(); V.utts = []; }
    // the fast matchers' guard (page onJSON 'intent', pageMatch): the user's words so far are a what-if (also joined to a split half)
    function whatIf(text) {
      if (isHypothetical(text).hyp) return true;
      const u = V.utts[V.utts.length - 1], prev = V.utts.length > 1 ? V.utts[V.utts.length - 2] : null;
      return !!(prev && prev.text && u && isBareModal(text) && isHypothetical(prev.text + ' ' + text).hyp);
    }
    /* fixes — narration without motion (judge: "here it comes ... how was that?" with no action call in the turn, twice;
       a face "switching now" 5 s before its call). The prompt forbids it (rtc.py P_TIMING); this is the harness
       check behind the prompt: avatarText(text, final) collects what the avatar says; once it has been silent for CLAIM_WAIT
       (a call often comes right after the words) and its words spoke of a move of its own as starting, under way or done
       (claimsMove) while NOTHING acted in this turn — no tool call of any kind, no matcher command, no move playing — it gets
       one silent note with the fact, and (subject to the unprompted-speech gate and quiet mode) a chance to make the call.
       Once per utterance of the user's; never while the user speaks. */
    const CLAIM_WAIT = 2200, CLAIM_TURN_MS = 45000;
    function avatarText(text, final) {
      const t = now();
      if (!V.av || t - V.av.t > 6000) V.av = {text: '', t0: t, t};
      if (final && text) V.av.text = String(text).slice(-600); else if (text) V.av.text = (V.av.text + String(text)).slice(-600);
      V.av.t = t;
      if (V.av.timer) T.clear(V.av.timer);
      V.av.timer = T.set(claimCheck, CLAIM_WAIT);
    }
    function actedSince(t0) {
      if ((V.lastToolAt || -1e12) >= t0) return true;
      let la = -1e12; try { la = d.lastActionAt ? +d.lastActionAt() : -1e12; } catch (e) {}
      if (Number.isFinite(la) && la >= t0) return true;
      const P = get('perf'); try { if (P && ((typeof P.busy === 'function' && P.busy()) || (typeof P.active === 'function' && P.active() > 0))) return true; } catch (e) {}
      try { const m = get('motion'), st = m && typeof m.state === 'function' ? m.state() : null;
        if (st && Array.isArray(st.active) && st.active.some(a => a && a.kind && a.kind !== 'look')) return true; } catch (e) {}
      return false;
    }
    function claimCheck() {
      const A = V.av; if (!A) return null;
      A.timer = null;
      const text = A.text; A.text = '';
      if (!text || heSpeaking()) return null;
      if (bool('avatarSpeaking')) { A.text = text; A.timer = T.set(claimCheck, 800); return null; }   // still talking: after it stops
      const c = claimsMove(text); if (!c) return null;
      const u = V.utts[V.utts.length - 1];
      const t0 = Math.max(u ? u.t0 - 1500 : -1e12, A.t0 - CLAIM_TURN_MS);
      if (actedSince(t0)) return null;
      if (u) { if (u.claimNoted) return null; u.claimNoted = true; } else if (V.claimAt && now() - V.claimAt < 30000) return null;
      V.claimAt = now(); V.claims = (V.claims || 0) + 1;
      log({vt: 'claim_no_action', kind: c, said: text.slice(-160)});
      prompt('Fact check, for you only: you just spoke as if a move of yours was starting, under way or finished, but nothing ran on screen in this turn, ' +
        'because no action call was made. He saw you stay still. If he asked for that move, make the action call for it now and say only a few words; ' +
        'if he did not ask for a move, say nothing about this. Never speak of a move as happening or done without its call.',
        {respond: true, unprompted: true, maxWaitMs: 2500});
      return c;
    }
    return {call, early, gate, expand, whatIf, covered: () => coveredNow(), heldNames, avatarText, claimCheck, claimsMove, onHub, onEvent, attach, userSpeech, userText, stats, reset, handles: n => ACTION.includes(n) || NAMES.includes(n),
      utterances: () => V.utts.map(u => Object.assign({}, u)), _V: V};
  }
  /* claimsMove(text) -> 'progress' | 'done' | null — does the avatar's own sentence speak of a move of ITS body as starting,
     under way or finished? General English shapes, not a phrase list: an announcement ("here comes / here goes / here we
     go"), a first-person or subjectless body-motion verb in the progressive with "now" / "for you" / a direction, or a
     completion check-in ("all done", "how was that", "did you see / like that"). Talk ABOUT moving in the future, offers
     and questions ("want me to spin?", "I can wave if you like", "should I ...") are not claims. */
  const BODY_ING = '(?:lowering|raising|lifting|dropping|dipping|turning|spinning|twirling|waving|tilting|leaning|bending|bowing|stretching|reaching|' +
    'rubbing|patting|scratching|tapping|clapping|pointing|closing|opening|shutting|blinking|winking|wiggling|wagging|nodding|shaking|jumping|hopping|bouncing|dancing|swaying|' +
    'stepping|walking|sitting|standing|crouching|kneeling|rolling|flipping|kicking|hugging|covering|posing|smiling|frowning|grinning|pouting|sticking|puffing)';
  const ANY_ING = '(?:going|moving|switching|changing|starting|doing|bringing|putting|getting into|easing|settling)';
  const RE_OFFER = /\b(?:want me to|would you like|do you want|should i|shall i|i can|i could|if you(?:'d)? like|let me know|tell me|say the word|whenever you|next time|can't|cannot|couldn't|could not|unable|not able|didn't|did not|haven't|have not|nothing (?:moved|happened))\b/;
  const CL_LEAD = "(?:^|[.!?,;:\\u2014-] *)(?:(?:okay|ok|alright|all right|whoa|wow|sure|yes|yep|right|and|so|now|oh)[, ]+)*(?:i'm |i am |now i'm |i'm now )?(?:now )?";
  // a body-motion verb + now / for you / a direction or manner; any other motion verb only when the clause ENDS on now / for you
  const RE_PROG = new RegExp(CL_LEAD + '(?:' + BODY_ING + "\\b[^.!?]{0,60}?\\b(?:now|for you|as we speak|up|down|lower|higher|left|right|around|over|back|slowly|gently|nice and \\w+)\\b|" +
    ANY_ING + "\\b[^.!?,;:\\u2014]{0,50}?\\b(?:now|for you)(?= *(?:[.!?,;:\\u2014-]|$)))");
  const RE_ANNOUNCE = /\bhere (?:it )?(?:comes|goes)\b|\bhere we go\b|\bwatch (?:this|me)\b|\bcheck (?:this|it) out\b/;
  const RE_DONE = /\ball done\b|\b(?:and )?(?:that's|that is) (?:it|done)\b|\bhow was that\b|\bhow did that look\b|\bdid you (?:see|like|catch) (?:that|it)\b|\bthere(?:,| —)? (?:all )?(?:done|finished)\b|\bi(?:'ve| have) (?:just )?(?:done|finished) (?:it|that)\b|\bi just did\b/;
  function claimsMove(text) {
    const t = String(text || '').toLowerCase().replace(/[’`]/g, "'").replace(/\s+/g, ' ').trim();
    if (!t) return null;
    let kind = null;
    for (const sent of t.split(/(?<=[.!?])\s*/)) {
      if (!sent) continue;
      const done = RE_DONE.test(sent);
      if (!done && RE_OFFER.test(sent)) continue;
      if (done) kind = 'done';
      else if (RE_ANNOUNCE.test(sent) || RE_PROG.test(sent)) kind = kind || 'progress';
    }
    return kind;
  }
  return {createVoiceTools, isHypothetical, isBareModal, guardCheck, requestTexts, lookKey, findProp, slug, claimsMove, ACTION, NAMES, HOLD_MS, TURN_BID_MIN};
});
