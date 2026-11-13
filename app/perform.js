/* perform.js — every directly commanded body action, on every character, goes through the voice model's
 * perform_action(description) function call (docs/PLANNER.md §2 page side, §4, §5).
 *
 *   const P = AvatarPerform.createPerformer({motion, prompt, sendCtl, runCommand, character, bodyState, userSpeaking,
 *       lastUserSpeech, alog, now, timers, cache, log})
 *   P.call(description, info)  → the tool result AT ONCE: {ok, status:'preparing'|'starting', id, next}
 *                               (server ← page {type:'plan', id, character, description, state})
 *   P.onKf(msg)                 server → page {type:'kf', id, poses, done, ok, source:'library'|'cache'|'planner', engine, ...}
 *   P.onAction(ev)              the motion's onAction events: {kind:'kf', id, phase:'ready'|'start'|'end'|'interrupted'|'fail'}
 *                               plus the ends of engine jobs (library runs, `object` calls of the same request)
 *   P.avatarAudio()             the avatar's voice just started (the page's playback tap: silence → voiced)
 *   P.engineCall(info, run)     an `object` call: runs now, or right after this request's earlier action
 *   P.stop(why)                 he asked it to stop / relax: jobs being prepared are dropped, a playing one is stopped
 *   P.list() / P.stats()        tests + voiceReport
 *
 * Flow: call → plan → poses → motion.command({action:'keyframes', id, description, poses, done, hold:true}) → kfFeed for
 * later batches → motion 'ready' → prompt(about to start, respond) → motion.kfRelease(id) at the cue's first audio (or
 * 700 ms) → 'start' (a note, no reply asked) → 'end' (check-in asked unless he is speaking or spoke since it started).
 * 'library' = engine commands run at once (no hold); a cache hit (here or at the server) plays all its poses at once.
 * Two calls of ONE request (no speech from him in between) run in order; a newer request replaces one still being
 * prepared. Nothing here holds, gates or delays the avatar's audio or lip-sync: the only hold is the motion waiting
 * for its cue. Note texts describe the situation only: the voice model words every reply itself.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarPerform = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const CUE_MAX_MS = 700;          // a ready job waits at most this long for its cue's first audio
  const CUE_STALE_MS = 1200;       // a cue not yet asked for by then is dropped (the motion has started)
  const CHECKIN_STALE_MS = 4000;   // the check-in may wait this long for a reply in progress to finish
  const PLAN_SILENCE_MS = 20000;   // no word from the server for this long = the plan failed
  const SAME_REQUEST_MS = 15000;   // calls closer than this, with no speech from him in between, are one request
  const WAIT_MAX_MS = 15000;       // a step never waits longer than this for the step before it
  const RULES_END_MS = 1800;       // a library run of rules-layer commands only (gesture/look/face/pose): its end
  const LIB_SAFETY_MS = 30000;
  const PART_NOW = ' Check in with him and tell him honestly what did not work, in a few words of your own.';
  const PART_LATER = ' Tell him honestly what did not work, in a few words, when you next speak.';     // an engine run whose end event never came
  const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const NEXT_PREP = 'It is being prepared: say something short and natural now, not that it is done. A silent note will tell you when it is about to start.';
  const NEXT_START = 'It starts right away: give him a brief natural cue now.';
  const NEXT_AFTER = 'It follows the step before it: say something short and natural now. A silent note will tell you when it is about to start.';
  // how he asked it to talk during the action (do_action's `talk`): through = keep talking while it plays,
  // quiet = no cue and no check-in (only context notes); usual = the default behaviour
  const NEXT_QUIET = 'He asked for it quietly: no words about it, before, during or after.';
  const NEXT_THROUGH = 'He asked you to talk while it plays: a silent note will tell you when it starts moving; then keep talking in time with it until it ends.';
  const NEXT_START_THROUGH = 'It starts right away and he asked you to talk while it plays: start now, in your own words, and keep going in time with it.';
  const talkOf = v => (v === 'through' || v === 'quiet' ? v : 'usual');

  function createPerformer(d) {
    d = d || {};
    const now = d.now || (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
    const T = d.timers || {set: (f, ms) => setTimeout(f, ms), clear: id => clearTimeout(id)};
    const motion = () => (d.motion ? d.motion() : null);
    const userSpeaking = () => { try { return !!(d.userSpeaking && d.userSpeaking()); } catch (e) { return false; } };
    const lastUserSpeech = () => { try { const v = d.lastUserSpeech ? +d.lastUserSpeech() : -1e12; return Number.isFinite(v) ? v : -1e12; } catch (e) { return -1e12; } };
    const P = {n: 0, jobs: new Map(), order: [], cache: new Map(), engine: new Map(), deferred: [], late: 0, prompts: 0};

    function prompt(text, opt) {
      P.prompts++;
      if (!d.prompt) return null;
      try { return d.prompt(text, opt || {}); } catch (e) { return null; }
    }
    function sendCtl(o) { if (d.sendCtl) { try { d.sendCtl(o); } catch (e) {} } }
    function log(rec) { if (d.log) { try { d.log(rec); } catch (e) {} } }
    // the thinking gesture (app/thinking.js) fills the wait while the planner writes the move
    function TH() { try { const k = typeof d.thinker === 'function' ? d.thinker() : d.thinker; return k && typeof k.begin === 'function' ? k : null; } catch (e) { return null; } }
    function thinkEnd(J, why) { const k = TH(); if (k) { try { k.end(J.id, why); } catch (e) {} } }
    function active() { return P.order.filter(J => !J.closed); }
    function jobOf(id) { return id == null ? null : P.jobs.get(String(id)) || null; }
    // one request = no speech from him since the earlier call (the model's next step, not his next ask)
    function sameRequest(t0) { return !(lastUserSpeech() > t0) && now() - t0 < SAME_REQUEST_MS; }
    function q(J) { return J.description.length > 160 ? J.description.slice(0, 157) + '...' : J.description; }
    function plannedS(J) {
      const m = motion();
      try { const st = m && m.kfState ? m.kfState(J.id) : null; if (st && Number.isFinite(+st.planned_s) && +st.planned_s > 0) return Math.round(+st.planned_s * 10) / 10; } catch (e) {}
      let mx = 0; for (const p of J.poses) if (+p.t > mx) mx = +p.t;
      return Math.round(mx * 10) / 10;
    }
    function safeState() {
      try {
        if (d.bodyState) return withThings(d.bodyState() || {}, motion());   // + the things (scene characters)
        const m = motion(), s = m && typeof m.state === 'function' ? m.state() : null;
        if (!s) return {};
        return withThings({posture: s.posture || null, holding: s.holding || null, wearing: s.wearing || [], seated: s.posture === 'seated' || !!s.sittingOn}, m);
      } catch (e) { return {}; }
    }
    // a scene character's things go with the plan request (the server plans WITH them) — and into the page cache key
    function withThings(st, m) {
      try {
        const th = m && typeof m.things === 'function' ? m.things() : null;
        if (Array.isArray(th) && th.length) {
          st.things = th.slice(0, 24);
          const es = m.engine && typeof m.engine.state === 'function' ? m.engine.state() : null;
          if (es && es.root) st.root = {x: +(+es.root.x || 0).toFixed(3), z: +(+es.root.z || 0).toFixed(3), yaw: +(+es.root.yaw || 0).toFixed(3)};
        }
      } catch (e) {}
      return st;
    }
    const q05 = v => Math.round((+v || 0) / 0.05) / 20;
    function thingsSig(st) {
      if (!st || !Array.isArray(st.things) || !st.things.length) return '';
      return st.things.map(t => [t.id, t.held || '', t.worn ? 'w' : '', t.where || '', t.on || '', t.inside || t.in || '', Array.isArray(t.pos) ? t.pos.map(q05).join(',') : ''].join(':')).join(';');
    }
    // ---- the page-side cache (the server keeps the real one): a repeat of a finished plan plays at once ('starting')
    const ckey = J => J.character + '|' + norm(J.description) + (J.sig ? '|' + J.sig : '');   // + where the things are
    function cacheGet(J) { if (d.cache === false) return null; const k = ckey(J), v = P.cache.get(k); if (!v) return null; P.cache.delete(k); P.cache.set(k, v); return v; }
    function cachePut(J, v) {
      if (d.cache === false || !J.description) return;
      P.cache.set(ckey(J), v);
      while (P.cache.size > 60) P.cache.delete(P.cache.keys().next().value);
    }
    // ---- the action log (D.alog, when the character has one) stays truthful: pending → the motion's own events by id
    function alogPending(J) {
      const A = d.alog && d.alog(); if (!A || J.alogged) return;
      J.alogged = true;
      try { A.command({action: 'perform', description: J.description}, {ok: true, id: J.id}, 'voice'); } catch (e) {}
    }
    function alogFail(J) {
      const A = d.alog && d.alog(); if (!A) return;
      try {
        if (J.alogged) A.push({kind: 'perform', name: 'perform', id: J.id, src: 'voice', phase: 'fail', reason: J.error});
        else A.command({action: 'perform', description: J.description}, {ok: false, reason: J.error}, 'voice');
      } catch (e) {}
    }

    // ------------------------------------------------------------------------------------------------ the tool call
    function call(description, info) {
      info = info || {};
      const desc = String(description == null ? '' : description).replace(/\s+/g, ' ').trim().slice(0, 400);
      if (!desc) return {ok: false, reason: 'no description given'};
      const t = now(), id = 'p' + (++P.n);
      const J = {id, description: desc, character: d.character ? String(d.character() || '') : '', t, info,
        marks: {call_ms: t}, phase: 'planning', status: 'preparing', source: null, poses: [], done: false, ok: null,
        error: null, cmd: false, playing: false, closed: false, waitFor: new Set(), engineIds: new Set(), timers: {},
        outcome: null, cue: null, alogged: false, talk: talkOf(info.talk),
        words: info.words ? String(info.words).replace(/\s+/g, ' ').trim().slice(0, 400) : ''};   // his words (library 2nd try)
      if (Number.isFinite(info.start_ms) && Number.isFinite(info.start_lat_ms)) J.marks.user_end_ms = info.start_ms - info.start_lat_ms;
      for (const o of active()) {
        if (sameRequest(o.t)) J.waitFor.add(o.id);                    // this request's earlier step: this one follows it
        else if (!o.playing) { o.outcome = 'replaced'; o.replacedBy = id; o.marks.end_ms = t; close(o); }   // a newer request
      }
      for (const [eid, e] of P.engine) if (!e.ended && sameRequest(e.t)) J.waitFor.add('e' + eid);   // an `object` step before it
      P.jobs.set(id, J); P.order.push(J);
      while (P.order.length > 60 && P.order[0].closed) { const g = P.order.shift(); for (const [k, v] of P.jobs) if (v === g) P.jobs.delete(k); }
      if (J.waitFor.size) J.timers.wait = T.set(() => { if (!J.closed && J.waitFor.size) { J.waitFor.clear(); J.waitTimedOut = true; proceed(J); } }, WAIT_MAX_MS);
      const st0 = safeState(); J.sig = thingsSig(st0);
      const hit = cacheGet(J);
      if (hit) {
        J.source = 'cache'; J.cacheHit = 'page'; J.marks.plan_first_pose_ms = t; J.done = true; J.ok = true;
        if (!J.waitFor.size) J.status = 'starting';
        if (hit.engine) runLibrary(J, hit.engine);
        else { J.poses = hit.poses.slice(); feed(J, J.poses, true); }
      } else {
        sendCtl(Object.assign({type: 'plan', id, character: J.character, description: desc, state: st0}, J.words ? {words: J.words} : {}));
        armWatch(J);
        if (!J.waitFor.size) { const k = TH(); if (k) { try { J.think = k.begin(id, {description: desc}); } catch (e) {} } }
      }
      log({perform: id, description: desc, status: J.status, after: [...J.waitFor], t: Math.round(t)});
      if (J.closed && J.outcome === 'failed') return {ok: false, id, reason: J.error || 'it could not be done'};
      const next = J.talk === 'quiet' ? NEXT_QUIET : J.status === 'starting' ? (J.talk === 'through' ? NEXT_START_THROUGH : NEXT_START)
        : J.talk === 'through' ? NEXT_THROUGH : J.waitFor.size ? NEXT_AFTER : NEXT_PREP;
      return {ok: true, status: J.status, id, next, talk: J.talk === 'usual' ? undefined : J.talk};
    }
    function armWatch(J) {
      T.clear(J.timers.watch);
      J.timers.watch = T.set(() => { if (!J.closed && !J.done) fail(J, 'the planner did not answer'); }, PLAN_SILENCE_MS);
    }

    // ------------------------------------------------------------------------------------------------ server → page
    function validPose(p) { return p && typeof p === 'object' && Number.isFinite(+p.t); }
    function onKf(m) {
      if (!m || m.id == null) return false;
      const J = jobOf(m.id);
      if (!J || J.closed) { P.late++; return false; }               // replaced / finished / another page's plan
      if (m.source) J.source = m.source;
      if (m.source === 'cache' || m.source === 'library' || (Array.isArray(m.poses) && m.poses.length)) { const k = TH(); if (k) { try { k.known(J.id); } catch (e) {} } }
      if (m.done) {
        J.hub = {source: m.source || null, ms_first_pose: m.ms_first_pose ?? null, ms_done: m.ms_done ?? null, usd: m.usd ?? null,
          summary: m.summary ? String(m.summary).slice(0, 200) : undefined};
        T.clear(J.timers.watch);
      } else armWatch(J);
      if (m.source === 'library' && m.ok !== false && Array.isArray(m.engine) && m.engine.length) {
        J.done = true; J.ok = true; J.marks.plan_first_pose_ms = J.marks.plan_first_pose_ms ?? now();
        cachePut(J, {engine: m.engine.slice()});
        runLibrary(J, m.engine);
        return true;
      }
      const poses = (Array.isArray(m.poses) ? m.poses : []).filter(validPose);
      if (poses.length) { if (J.marks.plan_first_pose_ms == null) J.marks.plan_first_pose_ms = now(); J.poses.push(...poses); }
      if (m.done) { J.done = true; J.ok = m.ok !== false; J.error = m.error ? String(m.error).slice(0, 160) : null; }
      if (m.done && !J.poses.length) { fail(J, J.error || 'nothing could be planned'); return true; }
      if (m.done && J.ok) cachePut(J, {poses: J.poses.slice()});
      feed(J, poses, !!m.done);
      return true;
    }
    // poses → the motion layer (a job that follows an earlier step keeps them until that step ends)
    function feed(J, newPoses, done) {
      if (J.closed || J.waitFor.size) return;
      const m = motion();
      if (!J.cmd) {
        if (!J.poses.length) return;
        if (!m || typeof m.command !== 'function') { fail(J, 'there is no body to play it on'); return; }
        alogPending(J);
        J.cmd = true; J.phase = 'buffering';
        let r = null;
        try { r = m.command({action: 'keyframes', id: J.id, description: J.description, poses: J.poses.slice(), done: J.done, hold: true}, {src: 'voice'}); }
        catch (e) { r = {ok: false, reason: 'error: ' + String(e && e.message || e).slice(0, 80)}; }
        if (!r || r.ok === false) { fail(J, (r && r.reason) || 'the body could not play it'); return; }
        if (r.id != null && String(r.id) !== J.id) P.jobs.set(String(r.id), J);   // the contract says the same id
        return;
      }
      if ((newPoses.length || done) && m && typeof m.kfFeed === 'function') { try { m.kfFeed(J.id, newPoses, !!done); } catch (e) {} }
    }
    function proceed(J) {           // the step before it ended: go
      if (J.closed) return;
      T.clear(J.timers.wait);
      if (J.pendingEngine) { const c = J.pendingEngine; J.pendingEngine = null; runLibrary(J, c); }
      else feed(J, [], J.done);
    }

    // ------------------------------------------------------------------------------------------------ motion events
    function onAction(ev) {
      if (!ev || typeof ev !== 'object') return;
      const ph = String(ev.phase || '');
      if (ev.kind !== 'kf') {
        if (ev.id != null && P.engine.has(String(ev.id)) && /^(end|interrupted|fail|failed|cancel|cancelled)$/.test(ph)) engineEnded(String(ev.id), ph, ev);
        return;
      }
      const J = jobOf(ev.id); if (!J || J.closed) return;
      if (ph === 'ready') onReady(J);
      else if (ph === 'start') onStart(J);
      else if (ph === 'end') finish(J, 'end', ev);
      else if (ph === 'interrupted') finish(J, 'interrupted', ev);
      else if (ph === 'fail' || ph === 'failed') finish(J, 'fail', ev);
    }
    function onReady(J) {
      if (J.phase !== 'buffering' && J.phase !== 'planning') return;
      J.phase = 'ready'; J.marks.ready_ms = now();
      if (userSpeaking()) { J.cue = 'none: he is speaking'; release(J, 'he is speaking'); return; }
      if (J.talk === 'quiet') { J.cue = 'none: quiet'; release(J, 'quiet'); return; }     // no words around it
      J.cueAskedAt = now();
      if (J.status === 'starting') J.cue = 'the tool result';       // its reply to 'starting' is the cue
      else if (J.talk === 'through') {                              // talk through the whole motion
        J.cue = 'note: talk through';
        const s = plannedS(J);
        J.cuePrompt = prompt(`The action he asked for (${q(J)}) is about to start` + (s > 0 ? `, about ${s} s of it planned so far` : '') +
          '. He asked you to talk while it plays: start now, in your own words, and keep going in time with it until it ends.',
          {respond: true, maxWaitMs: CUE_STALE_MS, onAudio: () => release(J, 'cue audio')});
      } else {
        J.cue = 'note';
        const s = plannedS(J);
        J.cuePrompt = prompt(`The action he asked for (${q(J)}) is about to start` + (s > 0 ? `, about ${s} s of it planned so far` : '') +
          '. Give him one brief natural cue now, in your own words. It has not happened yet: never say it is done or finished (a later note tells you when it is); if he asked you to talk during it, do that through the motion.',
          {respond: true, maxWaitMs: CUE_STALE_MS, onAudio: () => release(J, 'cue audio')});
      }
      J.timers.cue = T.set(() => release(J, 'cue timeout'), CUE_MAX_MS);
    }
    function release(J, why) {
      if (J.closed || J.phase !== 'ready') return;
      T.clear(J.timers.cue); J.phase = 'released'; J.releaseWhy = why;
      if (/audio/.test(why)) J.marks.cue_audio_ms = now();
      const m = motion();
      if (m && typeof m.kfRelease === 'function') { try { m.kfRelease(J.id); } catch (e) {} }
    }
    function onStart(J) {
      if (J.playing) return;
      T.clear(J.timers.cue);
      if (J.marks.first_motion_ms == null) J.marks.first_motion_ms = now();
      J.phase = 'playing'; J.playing = true; J.startedAt = now();
      thinkEnd(J, 'move started');                                  // (motion.js already faded it at the job's first frame)
      prompt(`The action he asked for (${q(J)}) is playing on screen now.`, {respond: false});
    }
    function finish(J, kind, ev) {
      if (J.closed) return;
      J.marks.end_ms = now();
      J.outcome = kind === 'end' ? 'done' : kind === 'interrupted' ? 'interrupted' : 'failed';
      const follower = active().some(F => F !== J && F.waitFor.has(J.id));
      if (kind === 'end') {
        const since = J.startedAt || J.marks.ready_ms || J.t;
        const heSpoke = userSpeaking() || lastUserSpeech() > since, quiet = J.talk === 'quiet';   // quiet = no check-in
        J.checkin = follower ? 'skipped: the next step follows' : heSpoke ? 'skipped: he spoke' : quiet ? 'skipped: quiet' : 'asked';
        const part = ev && ev.partial ? ` Part of it did not work: ${String(ev.partial).slice(0, 200)}.` : '';   // a failed object step
        if (part) J.partial = String(ev.partial).slice(0, 200);
        const tail = heSpoke || quiet ? (part ? PART_LATER : '') : part ? PART_NOW : ' Briefly check in with him about it, in your own words.';
        if (!follower) prompt(`The action he asked for (${q(J)}) has finished on screen.` + part + tail,
          {respond: !heSpoke && !quiet, maxWaitMs: CHECKIN_STALE_MS});
      } else if (kind === 'interrupted') {
        prompt(`The action he asked for (${q(J)}) was stopped before it finished.`, {respond: false});
      } else {
        J.error = String((ev && ev.reason) || 'the body could not play it').slice(0, 160);
        prompt(`The action he asked for (${q(J)}) could not be played (${J.error}). Tell him honestly in a few words.`,
          {respond: !userSpeaking(), maxWaitMs: 3000});
      }
      close(J);
    }
    function fail(J, reason) {
      if (J.closed) return;
      J.error = String(reason || 'it could not be done').slice(0, 160); J.outcome = 'failed'; J.marks.end_ms = now();
      alogFail(J);
      prompt(`The action he asked for (${q(J)}) could not be ${J.cmd ? 'played' : 'prepared'} (${J.error}). Tell him honestly in a few words.`,
        {respond: !userSpeaking(), maxWaitMs: 3000});
      close(J);
    }
    function close(J) {
      if (J.closed) return;
      J.closed = true; J.playing = false;
      for (const k in J.timers) T.clear(J.timers[k]);
      thinkEnd(J, J.outcome || 'closed');                           // failed / replaced / cancelled / done
      const marks = {};
      for (const [k, v] of Object.entries(J.marks)) if (Number.isFinite(v)) marks[k] = Math.round(v);
      sendCtl({type: 'perform', id: J.id, description: J.description, source: J.source || null, marks,
        character: J.character, outcome: J.outcome, status: J.status, cue: J.cue || null, release: J.releaseWhy || null,
        cache: J.cacheHit || undefined, error: J.error || undefined, hub: J.hub || undefined, checkin: J.checkin || undefined,
        talk: J.talk === 'usual' ? undefined : J.talk,
        after_wait: J.waitTimedOut ? 'timed out' : undefined});
      log({perform: J.id, outcome: J.outcome, t: Math.round(now())});
      wake(J.id);
    }
    function wake(dep) {
      for (const F of active()) if (F.waitFor.delete(dep) && !F.waitFor.size) proceed(F);
      for (const qd of P.deferred.slice()) if (qd.after.delete(dep) && !qd.after.size) runDeferred(qd);
    }

    // ------------------------------------------------------------------------------------------------ library runs
    function runLibrary(J, cmds) {
      if (J.closed) return;
      if (!J.source) J.source = 'library';
      if (J.waitFor.size) { J.pendingEngine = cmds; return; }
      const t = now();
      if (J.marks.ready_ms == null) J.marks.ready_ms = t;
      J.marks.first_motion_ms = t; J.phase = 'playing'; J.playing = true; J.startedAt = t; J.cue = 'none: a known move plays at once';
      let okN = 0, dupN = 0; const reasons = [];
      for (const c of (cmds || []).slice(0, 12)) {
        if (!c || typeof c !== 'object' || !c.action) continue;
        let r = null;
        try { r = d.runCommand ? d.runCommand(c, {perform: J.id, description: J.description}) : null; } catch (e) { r = {ok: false, reason: String(e && e.message || e).slice(0, 80)}; }
        const ok = r === true || (r && typeof r === 'object' ? r.ok !== false && !r.dup : !!r);
        if (ok) okN++; else if (r && typeof r === 'object' && r.dup) dupN++; else if (r && r.reason) reasons.push(String(r.reason));
        if (ok && /^(move|object)$/.test(c.action) && r && typeof r === 'object' && r.id != null) {
          J.engineIds.add(String(r.id)); P.engine.set(String(r.id), {t, job: J.id, ended: false});
        }
      }
      if (!okN && dupN && !reasons.length) {        // every step was already started from his words a moment ago
        J.cue = 'none: already running'; J.outcome = 'done'; J.dup = true; J.marks.end_ms = now();
        prompt(`The action he asked for (${q(J)}) is already playing on screen.`, {respond: false});
        close(J); return;
      }
      if (!okN) { fail(J, reasons[0] || 'the body could not do it'); return; }
      thinkEnd(J, 'library move started');
      prompt(`The action he asked for (${q(J)}) is playing on screen now.`, {respond: false});
      J.timers.end = T.set(() => finish(J, 'end'), J.engineIds.size ? LIB_SAFETY_MS : RULES_END_MS);
    }
    function engineEnded(eid, ph, ev) {
      const e = P.engine.get(eid); if (!e || e.ended) return;
      e.ended = true; P.engine.delete(eid);
      const J = e.job ? jobOf(e.job) : null;
      if (J && !J.closed) {
        J.engineIds.delete(eid);
        if (ph !== 'end') J.engWorst = /fail/.test(ph) ? 'fail' : J.engWorst === 'fail' ? 'fail' : 'interrupted';
        if (!J.engineIds.size) finish(J, J.engWorst || 'end', ev);
      }
      wake('e' + eid);
    }

    // ------------------------------------------------------------------------------------------------ object calls
    function noteEngine(r) {
      if (r && typeof r === 'object' && r.id != null && r.ok !== false) P.engine.set(String(r.id), {t: now(), job: null, ended: false});
      const t = now(); for (const [k, e] of P.engine) if (t - e.t > 60000) P.engine.delete(k);
    }
    function engineCall(info, run) {
      const before = active().filter(J => sameRequest(J.t));
      if (before.length) {
        const qd = {after: new Set(before.map(J => J.id)), run, t: now()};
        qd.timer = T.set(() => { if (P.deferred.includes(qd)) runDeferred(qd); }, WAIT_MAX_MS);
        P.deferred.push(qd);
        return {deferred: true, result: {ok: true, queued: true, did: 'will do that right after the action before it (not started yet)'}};
      }
      let r = null; try { r = run(); } catch (e) { r = {ok: false, reason: String(e && e.message || e).slice(0, 80)}; }
      noteEngine(r);
      return {deferred: false, result: r};
    }
    function runDeferred(qd) {
      const i = P.deferred.indexOf(qd); if (i < 0) return;
      P.deferred.splice(i, 1); T.clear(qd.timer);
      let r = null; try { r = qd.run(); } catch (e) { r = {ok: false, reason: String(e && e.message || e).slice(0, 80)}; }
      noteEngine(r);
      if (r && typeof r === 'object' && r.ok === false) prompt(`The next step he asked for could not be done (${String(r.reason || 'unknown').slice(0, 120)}).`, {respond: false});
    }

    // ------------------------------------------------------------------------------------------------ voice + stop
    function avatarAudio() {
      const t = now();
      for (const J of active()) {
        if (J.marks.first_speech_ms == null && t >= J.marks.call_ms) J.marks.first_speech_ms = t;
        if (J.phase === 'ready' && J.cueAskedAt != null && t >= J.cueAskedAt) release(J, 'cue audio (tap)');
      }
    }
    function stop(why) {
      let moving = false;
      for (const J of active()) {
        if (J.cmd || J.playing) moving = true;
        J.outcome = J.playing ? 'interrupted' : 'cancelled'; J.stoppedBy = why || 'stop'; J.marks.end_ms = now();
        close(J);
      }
      for (const qd of P.deferred) T.clear(qd.timer);
      P.deferred = [];
      { const k = TH(); if (k && typeof k.cancel === 'function') { try { k.cancel(why || 'stop'); } catch (e) {} } }
      if (moving) { const m = motion(); if (m && typeof m.command === 'function') { try { m.command({action: 'stop'}, {src: 'voice'}); } catch (e) {} } }
      return moving;
    }
    function list() {
      return P.order.map(J => ({id: J.id, description: J.description, character: J.character, phase: J.phase, status: J.status,
        source: J.source, outcome: J.outcome, closed: J.closed, cue: J.cue, release: J.releaseWhy || null, poses: J.poses.length,
        waitFor: [...J.waitFor], marks: Object.assign({}, J.marks), error: J.error, checkin: J.checkin || null, talk: J.talk}));
    }
    function stats() {
      const L = list();
      return {jobs: L.length, active: L.filter(x => !x.closed).length, late_kf: P.late, cache: P.cache.size, deferred: P.deferred.length,
        prompts: P.prompts, last: L.slice(-8)};
    }
    function reset() { stop('reset'); P.jobs.clear(); P.order = []; P.engine.clear(); }
    return {call, onKf, onAction, avatarAudio, engineCall, stop, list, stats, reset, active: () => active().length,
      busy: () => active().some(J => J.cmd || J.playing)};
  }
  return {createPerformer, norm, CUE_MAX_MS, CUE_STALE_MS, CHECKIN_STALE_MS};
});
