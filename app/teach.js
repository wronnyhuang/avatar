/* teach.js — TEACH (seam S4): Muse the teacher. Walks to the chalkboard / the TV,
 * picks up the chalk, writes the board brain's items stroke by stroke (the chalk tip on the board plane through the paw's
 * reach map; marks reported to the server on the playback timeline so they appear under the tip), erases with the eraser,
 * talks between items (say -> write -> say -> write, turning to the user now and then), and goes home on abort / go_home /
 * idle. Runs on the muse's job queue (app/moves.js kit: stand_at + steps goals), so the precondition planner stands it up,
 * frees a paw, walks, and the existing interrupt path (stop) cuts a stroke within a frame.
 *
 *   const T = AvatarTeach.createTeach({engine: () => motion.engine, send: msg => hub, prompt: (text, opt) => voice.prompt,
 *       audible: () => avatarVoiceAudible, playT: () => timeline s, now: () => ms, history: () => [{role, text}], log, cfg})
 *   T.setConfig(props)          config.props of the character (board / tv places, chalk / eraser props); null = off
 *   T.handles(name) / T.call(name, args) -> tool result   teach_board, board_erase, go_home
 *   T.onHub(msg) -> bool        board_item / board_plan_done / board_plan_error / board_state / teach_abort
 *   T.frame(dt)                 every rig tick, after motion.step (the cadence director + the ink flush)
 *   T.userText(text, final)     the user's words (stop / never mind / come back -> abort);  T.interrupt(why)
 *   T.onAction(ev)              motion onAction events (a teach job cut short by the stop path -> abort + home)
 *   T.goStation(id, opt)        hook for other tools (tv_show -> goStation('tv', {present: true}))
 *   T.state() / T.stats()       for tests and the hub record
 * Nothing here knows any subject: items are uv polylines + a line to say; everything else is geometry and timing.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === 'object' && module.exports) module.exports = lib;
  else root.AvatarTeach = lib;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const TAU = 2 * Math.PI;
  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const lerp = (a, b, s) => a + (b - a) * s;
  const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const len3 = a => Math.hypot(a[0], a[1], a[2]);
  const nrm3 = a => { const l = len3(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const lerp3 = (a, b, s) => [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
  const wrap = x => { x = x % 1; if (x >= 0.5) x -= 1; if (x < -0.5) x += 1; return x; };
  const yawOfDir = d => Math.atan2(d[0], d[2]) / TAU;
  const mj = s => s <= 0 ? 0 : s >= 1 ? 1 : s * s * s * (10 + s * (6 * s - 15));
  const v3 = a => Array.isArray(a) && a.length >= 3 ? a.slice(0, 3).map(Number) : null;

  // tuning (hub config teach.* overrides; every number in H units / seconds)
  const CFG = {
    walkSpeed: 2.0,           // gait speed multiplier to and between stations
    turnStep: 0.16,           // turns per step when it turns on the spot before / after a walk (engine default 0.1)
    writeSpeed: 0.6,          // chalk tip cruise speed on the board (H/s) (integ: 0.42 -> 0.6, items took 13-18 s live)
    penUpSpeed: 1.3,          // tip speed between strokes, chalk lifted (H/s)
    penUpMin: 0.07,           // shortest pen-up hop (s)
    strokeMin: 0.07,          // shortest stroke (s)
    hover: 0.016,             // chalk lifted this far off the board between strokes (H)
    tipLen: 0.044,            // chalk: grasp point -> tip along the stick (H) (props JSON chalk.tip_len_H wins)
    faceLen: 0.0282,          // eraser: grasp point -> felt face centre (H) (props JSON eraser.face_len_H wins)
    chalkUp: 0.7,             // the stick points into the board and this much up (the paw sits below / out from the tip) (r2-boardvis: 0.45 -> 0.7)
    chalkSide: 1.1,           // r2-boardvis: ... and this much toward the side away from the body (see toolDir)
    camPos: [0, 0.5652, 3.3478],   // the render camera (H, world) — held props face it (props.glsl propHeldIn)
    contactTol: 0.016,        // the paw got within this of its target: the mark is drawn (H)
    inkLagS: 0.03,            // marks stamped this much after the frame that reached them (the renderer's rig smoothing)
    lean: 0.12,               // forward lean while writing (bend units, moves.js bendSolve)
    sayFrac: 0.7,             // start writing when the line is this far through (estimated from its length)
    charsPerS: 14.5,          // speaking rate for that estimate
    sayNoAudioS: 2.2,         // no audio this long after the note went out: write anyway (motion never waits on speech)
    sayCapS: 2.6,             // never wait more than this much of a long line
    prefireFrac: 0.78,        // the next item's line is asked for when the current item is this far through (prefire on)
    prefire: false,           // off: say -> write -> step back + say -> write (the line while still writing hid the text)
    turnEvery: [2, 3],        // a full half-turn to the user every N items (the others: over the shoulder)
    talkTurn: 0.5,            // talking pose: the feet turn this fraction of the way from the board toward the user
    tvKeepS: 240,             // devloop: a TV picture shown this recently keeps the TV in the board's framing
    idleHomeS: 40,            // at a station with nothing to do this long -> back home
    presentS: 3.0,            // presenting pose at the TV (open paw toward it)
    eraseSpeed: 2.8,          // eraser speed over the board (H/s) (1.1 took ~20 s to wipe a lesson)
    eraseOverlap: 0.6,        // row spacing as a fraction of the eraser pad height (integ: 0.85 left bands of chalk between rows live)
    eraseOtherPaw: false,     // integ: the eraser in the paw that does not hold the chalk (off: that paw cannot reach the board's left half)
    eraseStepSpeed: 2.6,      // integ: gait speed for the steps between erase columns (r2-erase: 1.6 -> 2.6)
    eraseLagMax: 0.02,        // devloop: the wipe path waits for a paw this far (H) behind it
    eraseSplitMax: 1,         // devloop: how often an unreachable erase column may be split
    eraseReachTol: 0.03,      // devloop: reach error a stance may leave for the eraser's pad
    eraseColU: 0.34,          // integ: an erase column (uv-x) = what one stance wipes at full height (r2-erase: any yaw: 0.25 -> 0.34)
    windowU: 0.1,             // r2-boardvis: 0.15 -> 0.1 (beside the pen by latWant, the arm reaches ~0.38 H: ~0.1 of the board per stance)            // one stance writes at most this much of the board's width (uv-x) (integ: 0.3 -> 0.15 so it steps along
                              // with the line and its body does not hide what it is writing: 51% -> 15% of marks hidden while drawn, node)
    writeStepSpeed: 2.2,
    drawSpeedK: 1.6,          // integ: picture lines (diagram / figure / marks) this much faster than letters
    holdPawOnStep: true,      // integ: between stances the chalk paw stays up (rides with the body) instead of retracting      // integ: gait speed of the steps along the board while writing
    side: 'l',                // the writing paw (screen-left = its own right)
    brainWaitS: 25,           // no item from the brain this long: give up honestly
    bodyR: 0.26,              // the muse's half-width seen from the camera (H, the head is the widest part): what it hides of the board
    clearPad: 0.04,           // talking stance: its silhouette this far (uv-x) off the text it just wrote
    revealStep: 0.5,          // between items it walks at most this far (H) to show the whole lesson, else the last item
    showS: 1.2,               // after stepping clear it lets the user see the text this long before it writes on
    views: true,              // ease the camera to the station's framing (scene places[].camera) while working there
    holdForUser: true,
    hideCost: 2.0,            // integ: stance cost per fraction of the points it writes that its body hides
    hidePad: 0.01,
    latWant: 0.32,            // r2-boardvis: it stands this far (H, along the board) beside what it writes (measured: the text by the pen ~88% visible)
    latMin: 0.24,             // ... and a stroke joins the current stance only if it is at least this far
    pauseOnSpeech: true,      // integ: the user's speech start lifts the chalk at once (the words decide stop / resume)
    softHomeS: 12,            // integ: after a change of subject it waits at the station this long for a new request
    // r2-boardvis: the silhouette model (tools/board_sil_fit.py on rendered masks), the pad (uv) round it, and the
    // most of the board's ink it may hide while it talks between items (else it steps clear first)
    sil: {hx: -0.1212, hy: 0.6562, hz: 0.04, hr: 0.2212, cy: 0.6575, cr: 0.245, py: 0.3425, pr: 0.2275, fy: 0.07, fx: 0.1, fr: 0.08,
      sx: 0.1925, sy: 0.43, ar: 0.06, pw: 0.0625, earx: 0.17, eary: 0.98, earr: 0.06},
    silPad: 0.01,
    sayHideMax: 0.02,
    clearSpeed: 2.0,          // gait speed of the step clear between items
    clearRange: [-0.1, 1.25],   // station-line positions (writing column a) it may step to between items (the board camera holds them)
    replyWaitS: 4,            // after the user spoke mid-lesson: wait this long for the voice's reply to start (then its end)        // the user is talking: no new line / item starts until they stop (their words may be a stop)
    // r2-erase: a quick wipe (design note: "erase in a few seconds, not 20-23 s"). The duster lives in the paw that does not
    // write (picked up at the first wipe, kept while at the board: no chalk <-> eraser trips to the tray), the wipe is a few
    // broad up-and-down strokes with the felt's long side across them, one stance per wide column (any body yaw that reaches),
    // the paw stays at the board between stances, and a page turn wipes only the room the next item needs (the rest
    // between items)
    eraserPaw: 'other',       // 'other' = the paw that does not hold the chalk; 'same' = the old way (chalk to the tray)
    eraseKeep: true,          // the duster stays in that paw after a wipe while it is at the board
    grabBoth: 'used',         // the lesson's trip to the tray takes the duster too: 'always' | 'used' (only onto a board with chalk on
                              // it: a page turn is likely) | false (at the first wipe). 'always' cost 1.6 s of the first mark on a free board
    grabLater: 'end',         // with the tray close by, the free paw takes the duster: true = between two lines of a lesson, 'end' = when
                              // the lesson is done (r2-boardvis: between lines, bent to the tray, it hid 60-70% of the board for the
                              // whole line: say hidden .037 -> .053 over 16 lessons)
    grabEndH: 1.2,            // r2-boardvis: ... at the lesson's end it walks this far (H) along the board for it
    grabNearH: 0.45,          // ... when its grab stance is at most this far (H) from where it stands
    sweepDir: 'vertical',     // up-and-down strokes, the felt's long side across them (the paw's fastest axis: the arm's
                              // slew limit allows ~1.9 H/s up/down vs ~0.8 H/s sideways); 'horizontal' = rows, felt upright
    sweepOverlap: 0.82,       // stroke spacing as a fraction of the felt's long side
    sweepArc: 0.25,           // a stroke bows this fraction of the spacing in its middle (a sweeping arc, not a ruler)
    eraseRaiseSpeed: 1.8,     // H/s of the paw's hop from where it is to the next column's first sweep (0.15..0.45 s)
    eraseHideCost: 0.6,       // stance cost per fraction of the column its body hides from the camera
    eraseHideSil: 0,          // r2-boardvis: the same by the fitted silhouette (0 = the cylinder above)
    eraseStanceTol: 0.012,    // reach error a wipe stance may leave (0.03: the felt fell short at the column's top corners, 7% residue)
    lazyTurn: true,           // a page turn wipes the next item's room first, the rest between items / at the end
    turnSlowH: 0.07,          // a wipe stroke slows within this (H) of its turn
    eraseRetract: false,      // true: a full retract of the duster paw after the last column (before anything else moves)
    wipeBetween: true,        // a lazy page turn's old columns go one per gap between items (while the next line is said)
    eraseTalk: true,          // the line before a wipe is said while it wipes (not before it)
    wipeSayLead: 1.2,         // the next item's line starts this long before the wipe's estimated end
  };

  // ------------------------------------------------------------------ geometry: the board / tv planes from config.props
  function surfaceOf(props, id) {
    const pl = props && (props.places || []).find(p => p.id === id || p.kind === id);
    if (!pl) return null;
    const g = pl.plane || pl.screen || pl.rect || pl.surface || pl.writable;
    if (!g || !v3(g.origin) || !v3(g.u) || !v3(g.v)) return null;
    const O = v3(g.origin), U = nrm3(v3(g.u)), V = nrm3(v3(g.v)), size = g.size || [1, 0.5], W = +size[0], Hh = +size[1];
    let n = nrm3(cross(V, U));
    const st = pl.station || {}, sp = v3(st.pos) || null;
    const C = add3(add3(O, mul3(U, W / 2)), mul3(V, Hh / 2));
    if (sp && dot3(sub3(sp, C), n) < 0) n = mul3(n, -1);           // the normal points out of the board, into the room
    const station = sp ? {x: sp[0], z: sp[2], yaw: +st.yaw || 0} : {x: C[0] + n[0] * 0.3, z: C[2] + n[2] * 0.3, yaw: yawOfDir(mul3(n, -1))};
    const yawB = yawOfDir(mul3(n, -1));                             // facing the board squarely
    return {id: pl.id, pl, O, U, V, W, Hh, n, C, station, yawB, tray: pl.tray || null,
      world: (u, v) => add3(add3(O, mul3(U, u * W)), mul3(V, v * Hh))};
  }

  function createTeach(o) {
    o = o || {};
    const cfg = Object.assign({}, CFG, o.cfg || {});
    const now = o.now || (() => Date.now());
    const sec = () => now() / 1000;
    const log = r => { if (o.log) { try { o.log(r); } catch (e) {} } };
    const send = m => { if (o.send) { try { o.send(m); } catch (e) {} } };
    const eng = () => { try { return o.engine ? o.engine() : null; } catch (e) { return null; } };
    const K = () => { const e = eng(); return e && e.kit ? e.kit : null; };
    const T = {inked: {}, userProps: new Set(), userObjAt: 0, props: null, board: null, tv: null, home: {x: 0, z: 0, yaw: 0}, at: 'home', req: null, reqN: 0, jobs: new Map(),
      phase: 'idle', item: null, sinceTurn: 0, drawn: {}, turnK: 2, ink: [], inkSent: 0, inkPts: 0, boardState: null, idleAt: sec(),
      look: null, events: [], notes: [], stats: {calls: 0, items: 0, strokes: 0, aborts: 0, firstMark: null, misses: 0, erases: 0}};

    function setConfig(props) {
      T.props = props || null;
      T.board = props ? surfaceOf(props, 'board') : null;
      T.tv = props ? surfaceOf(props, 'tv') : null;
      const chalk = propDef('chalk'), er = propDef('eraser');
      const Hs = props && props.units && +props.units.H > 0 ? +props.units.H : 2.3;
      if (chalk && +chalk.tip_len_H > 0) cfg.tipLen = +chalk.tip_len_H;
      if (chalk && chalk.tip && v3(chalk.tip.grasp_to_tip_H)) cfg.tipLen = len3(v3(chalk.tip.grasp_to_tip_H));
      if (er && +er.face_len_H > 0) cfg.faceLen = +er.face_len_H;
      if (er && er.face && v3(er.face.grasp_to_face_scene)) cfg.faceLen = len3(v3(er.face.grasp_to_face_scene)) / Hs;
      if (er && er.face && Array.isArray(er.face.footprint_scene)) cfg.pad = er.face.footprint_scene.map(x => +x / Hs);
      if (props && v3(props.camera_pos_H)) cfg.camPos = v3(props.camera_pos_H);
      const h = props && (props.home || (props.places || []).find(p => p.id === 'home'));
      T.home = h && v3(h.pos) ? {x: h.pos[0], z: h.pos[2], yaw: +h.yaw || 0} : {x: 0, z: 0, yaw: 0};
      return {board: !!T.board, tv: !!T.tv};
    }
    const propDef = id => T.props && (T.props.props || []).find(p => p.id === id || p.kind === id) || null;
    const has = () => !!(T.board && K());
    const toolId = kind => { const d = propDef(kind); return d ? d.id : null; };

    // ------------------------------------------------------------------ jobs on the muse's queue
    function job(name, goals, why, onEnd) {
      const k = K(); if (!k) return null;
      const r = k.enqueue(name, goals, 'teach', {hold: true});
      if (!r || !r.ok) { log({teach: 'job_fail', name, reason: r && r.reason}); return null; }
      const J = {id: r.id, name, why, onEnd, t0: sec(), done: false, ok: null};
      if (r.already) { J.done = true; J.ok = true; if (onEnd) setTimeout0(() => onEnd(true)); return J; }
      T.jobs.set(r.id, J);
      return J;
    }
    const deferred = [];
    function setTimeout0(f) { deferred.push(f); }
    function jobAlive(J) { if (!J || J.done) return false; const k = K(); return !!(k && k.E.jobs.some(j => j.id === J.id)); }
    function onAction(ev) {
      // devloop (r0 turn 6: "take the eraser and hold it up" -> picked up, then the lesson's home walk put it straight
      // back): a thing the USER asked for through the object tool is theirs — the director never tidies it away, and while it
      // does that the station is let go (no idle walk home in the middle of it)
      if (ev && ev.kind === 'object' && ev.src && ev.src !== 'teach' && ev.src !== 'auto' && !(ev.id !== undefined && T.jobs.has(ev.id))) {
        const tools = [toolId('chalk'), toolId('eraser')].filter(Boolean), nm = String((ev.args && ev.args.object && (ev.args.object.name || ev.args.object.id)) || '').toLowerCase();
        if (ev.phase === 'start') T.heldAt0 = tools.filter(q => held(q));
        for (const q of tools) {
          const p = K() && K().byId(q), named = nm && p && (nm.includes(String(p.name || q).toLowerCase()) || nm.includes(q));
          if (named || (ev.phase !== 'start' && held(q) && !(T.heldAt0 || []).includes(q)) || (ev.state && ev.state.object === q)) T.userProps.add(q);
        }
        T.userObjAt = sec(); T.idleHome = null;
        if (!T.req && T.at && T.at !== 'home' && !String(T.at).startsWith('going')) { T.at = null; T.phase = 'idle'; setWhere(); view(null); log({teach: 'left station', why: 'user object request'}); }
      }
      if (!ev || ev.id === undefined || !T.jobs.has(ev.id)) return;
      if (ev.phase === 'start') return;
      const J = T.jobs.get(ev.id); T.jobs.delete(ev.id);
      J.done = true; J.ok = ev.phase === 'end'; J.reason = ev.reason; J.did = ev.did;
      T.events.push({t: sec(), job: J.name, phase: ev.phase, did: ev.did, reason: ev.reason});
      if (J.onEnd) { try { J.onEnd(J.ok, ev); } catch (e) { log({teach: 'onEnd', err: String(e)}); } }
      // a teach job cut short by somebody else (the stop path, a new command): teaching is over, go home
      // ... unless what cut it is the USER's request for a thing (an object job): then the lesson just ends where it stands and
      // the body is the user's request's (walking home would undo it)
      // r2-objvoice (r6 eraser turn: the user's "hold the eraser up" cut the lesson's tidy-up job; the "That did not work"
      // note below made the voice say "nothing changed" while the eraser was being shown): a teach job yielded to the USER's
      // object request is not a failure of anything the user asked for -> no note
      const kY = K(), theirs = !J.ok && !J.mine && ((kY && (kY.E.jobs || []).some(j => j.kind === 'object' && j.src !== 'teach' && j.src !== 'auto')) || (T.userObjAt && sec() - T.userObjAt < 3));
      if (ev.phase === 'fail' && !J.mine && !theirs && o.prompt) { try { o.prompt(`That did not work: ${String(ev.reason || 'it could not be done').slice(0, 120)}. Tell the user honestly in a few words.`, {respond: true, maxWaitMs: 4000}); } catch (e) {} }
      if (!J.ok && !T.aborting && !J.mine) {
        if (theirs) { if (T.req) { send({type: 'teach_cancel', id: T.req.id, why: 'user object request'}); T.req = null; } T.item = null; T.say = null; T.at = null; T.phase = 'idle'; T.idleHome = null; setWhere(); view(null); log({teach: 'yield', why: 'user object request'}); }
        else abort('stopped (' + (ev.reason || ev.phase) + ')', {fromStop: true});
      }
    }
    function cancelJobs(why) {
      const k = K();
      for (const J of T.jobs.values()) { J.mine = true; if (k) k.cancel(J.id, why); }
    }

    // ------------------------------------------------------------------ poses
    const E = () => K().E;
    function root() { const k = K(); return k.E.base; }
    function held(id) { const k = K(); if (!k || !id) return null; const p = k.byId(id); return p && p.held ? (p.held === 'both' ? 'l' : p.held) : null; }
    function talkPose(B, stance) {
      const s = stance || {x: root().x, z: root().z, yaw: B.yawB};
      return {x: s.x, z: s.z, yaw: s.yaw + wrap(0 - s.yaw) * cfg.talkTurn};
    }
    // where to stand to write a set of board points (uv) with paw s: in front of the board, facing it or turned a little
    // side-on (the muse's arms are short: a sideways reach goes further than a forward one), every point reachable (reach
    // map + the least bend), the body on the far side of the writing paw; the cheapest valid stance near the current one
    function stanceFor(B, uvs, s, cur, kind) {
      const k = K(), n = B.n, fl = v => { const l = Math.hypot(v[0], v[2]) || 1; return [v[0] / l, 0, v[2] / l]; }, nf = fl(n), Uf = fl(B.U);
      if (o.stanceHook) { const hk = o.stanceHook(B, uvs, s, cur, kind, {k, nf, Uf, gripOf, reach: (r, P) => k.bendSolve(r, P, [s], false)}); if (hk) return hk; }   // r2-boardvis study
      const pts = uvs.map(q => gripOf(B, B.world(+q[0], +q[1]), 0, kind));
      const cen = mul3(pts.reduce((a, q) => add3(a, q), [0, 0, 0]), 1 / pts.length);
      const side = s === 'l' ? -1 : 1, here = cur || {x: root().x, z: root().z}, cands = [];
      for (const tw of [0, 0.05, -0.05, 0.1, -0.1]) for (const d of [0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38]) for (let li = -12; li <= 12; li++) {
        const lat = li * 0.025, b = add3([cen[0], 0, cen[2]], add3(mul3(nf, d), mul3(Uf, lat)));
        const r = {x: b[0], y: 0, z: b[2], yaw: B.yawB + tw};
        const lx = k.toLocalR(r, cen)[0];
        if (lx * side < -0.02) continue;                                       // the writing on the paw's own side
        cands.push({r, cheap: 0.5 * Math.abs(d - 0.24) + 0.9 * Math.hypot(r.x - here.x, r.z - here.z) + 0.5 * Math.abs(tw) + 0.4 * Math.abs(lx - side * 0.16)});
      }
      // the scene's own station line (where it stands for a writing column at uv-x a), tried first
      const L = B.pl.station && B.pl.station.line, sy = B.pl.station && Number.isFinite(+B.pl.station.yaw) ? +B.pl.station.yaw : null;
      if (L && v3(L.from) && v3(L.to) && sy !== null) {
        const au = uvs.reduce((q, w) => q + +w[0], 0) / uvs.length, base = lerp3(v3(L.from), v3(L.to), au);
        for (const dl of [0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.16, -0.16, 0.2, -0.2, 0.24]) for (const dn of [0, -0.04, 0.04, -0.08, -0.12]) for (const dy of [0, 0.04, -0.04, 0.08]) {
          const b0 = add3(base, add3(mul3(Uf, dl), mul3(nf, dn))), r = {x: b0[0], y: 0, z: b0[2], yaw: sy + dy};
          cands.push({r, cheap: -1 + 0.6 * Math.abs(dl) + 0.6 * Math.abs(dn) + 0.5 * Math.abs(dy) + 0.4 * Math.hypot(r.x - here.x, r.z - here.z)});
        }
      }
      // integ: the camera should see what it writes: a stance whose silhouette hides the points it writes from there
      // costs extra (the writing used to vanish behind its head as it went: ~50% of the marks hidden while drawn)
      // r2-boardvis: ... with its REAL silhouette (silDiscs; the paw on the chalk is where the mark is, not counted):
      // the stance goes as far right of what it writes as the arm reaches (measured: 44% -> see the log)
      // measured (tools/board_stance_probe.cjs, rendered): what decides it is how far the body stands along the board
      // from the pen — 0.20 H: the text by the pen 0-10% visible; 0.26: ~65%; >= 0.32: ~88%; the yaw and the distance
      // from the slate hardly matter. So every point it writes from a stance should be >= latWant to the paw's side.
      if (kind !== 'eraser' && cfg.hideCost > 0) for (const c of cands) c.cheap += cfg.hideCost * latShort(c.r, pts, s);
      cands.sort((a, b) => a.cheap - b.cheap);
      let n0 = 0;
      for (const c of cands) {
        if (++n0 > 600) break;
        const r = c.r;
        if (k.standClear && !k.standClear(r.x, r.z, null)) continue;
        // devloop: the eraser's felt is a pad (half-height ~0.017 H): a paw that falls short by up to 0.03 still wipes
        // the point (the board's top right was out of reach by ~2 cm for the flat-held eraser, never for the chalk)
        const eMax = kind === 'eraser' ? cfg.eraseReachTol : 0.018;
        let e = 0, bp = 0;
        for (const P of pts) { const q = k.bendSolve(r, P, [s], false); e = Math.max(e, q.e); bp = Math.max(bp, q.b.p); if (e > eMax) break; }
        if (e > eMax) continue;
        return {x: r.x, z: r.z, yaw: r.yaw, cost: c.cheap, bend: bp, e};
      }
      return null;
    }
    // can paw s write these uv points from stance st (no step)?
    function reachFrom(B, st, uvs, s) {
      const k = K(), r = {x: st.x, y: 0, z: st.z, yaw: st.yaw};
      for (const q of uvs) if (k.bendSolve(r, gripOf(B, B.world(+q[0], +q[1]), 0), [s], false).e > 0.018) return false;
      return true;
    }

    // r2-erase: where to stand to wipe uv points with the felt in paw s: in front of the board at any body yaw (a
    // teacher wiping turns to the board; the reach map decides — a full-height column ~0.35 of the board's width per
    // stance either paw, vs 0.15-0.25 at the writing yaws); the cheapest by distance + turn from `cur`
    function eraseStance(B, uvs, s, cur) {
      const k = K(), fl = v => { const l = Math.hypot(v[0], v[2]) || 1; return [v[0] / l, 0, v[2] / l]; }, nf = fl(B.n), Uf = fl(B.U);
      const pts = uvs.map(q => gripOf(B, B.world(+q[0], +q[1]), 0, 'eraser'));
      const cen = mul3(pts.reduce((a, q) => add3(a, q), [0, 0, 0]), 1 / pts.length);
      const here = cur || {x: root().x, z: root().z, yaw: root().yaw}, cands = [];
      for (const d of [0.22, 0.26, 0.3, 0.18, 0.34]) for (let li = -8; li <= 8; li++) for (let yi = 0; yi < 24; yi++) {
        const lat = li * 0.04, b = add3([cen[0], 0, cen[2]], add3(mul3(nf, d), mul3(Uf, lat))), yaw = wrap(yi / 24 - 0.5);
        cands.push({r: {x: b[0], y: 0, z: b[2], yaw}, cheap: 0.6 * Math.abs(d - 0.26) + 0.9 * Math.hypot(b[0] - here.x, b[2] - here.z) + 0.35 * Math.abs(wrap(yaw - (here.yaw || 0)))});
      }
      // seen from the camera, the wipe should happen beside the body, not behind it (the marks vanish under the duster,
      // not behind its back): a stance that hides the column costs extra
      // r2-boardvis: eraseHideSil > 0 -> by the fitted silhouette in that stance (the duster paw on the column)
      if (cfg.eraseHideSil > 0) for (const c of cands) c.cheap += cfg.eraseHideSil * silHidden(B, {x: c.r.x, z: c.r.z, yaw: c.r.yaw}, uvs, cen, s);
      else if (cfg.eraseHideCost > 0) for (const c of cands) {
        const h = hiddenU(B, c.r.x, c.r.z); if (!h) continue;
        let m = 0; for (const q of uvs) if (+q[0] > h[0] && +q[0] < h[1]) m++;
        c.cheap += cfg.eraseHideCost * m / uvs.length;
      }
      cands.sort((a, b) => a.cheap - b.cheap);
      let n0 = 0;
      for (const c of cands) {
        if (++n0 > 2100) break;
        const r = c.r;
        if (k.standClear && !k.standClear(r.x, r.z, null)) continue;
        let e = 0, bp = 0;
        for (const P of pts) { const q = k.bendSolve(r, P, [s], false); e = Math.max(e, q.e); bp = Math.max(bp, q.b.p); if (e > cfg.eraseStanceTol) break; }
        if (e > cfg.eraseStanceTol) continue;
        return {x: r.x, z: r.z, yaw: r.yaw, cost: c.cheap, bend: bp, e, n: n0};
      }
      return null;
    }
    const eraserSide = () => { const ch = held(toolId('chalk')); return cfg.eraserPaw === 'other' ? (ch ? (ch === 'l' ? 'r' : 'l') : (cfg.side === 'l' ? 'r' : 'l')) : cfg.side; };

    // ------------------------------------------------------------------ strokes -> paw steps (built when the job runs)
    // each stroke: world tip points + cumulative arclength; pen-up hops between them; the tip timing (min-jerk along the
    // arclength, so it starts and ends at rest) gives a brisk, human hand
    function strokePath(B, uv) {
      const P = uv.map(q => B.world(+q[0], +q[1])), cum = [0];
      for (let i = 1; i < P.length; i++) cum.push(cum[i - 1] + len3(sub3(P[i], P[i - 1])));
      return {uv, P, cum, L: cum[cum.length - 1]};
    }
    function pathAt(S, s) {
      const c = S.cum; if (S.P.length === 1 || s <= 0) return {p: S.P[0], i: 0, f: 0};
      if (s >= S.L) return {p: S.P[S.P.length - 1], i: S.P.length - 2, f: 1};
      let lo = 0, hi = c.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c[m] <= s) lo = m; else hi = m; }
      const f = (s - c[lo]) / Math.max(1e-9, c[hi] - c[lo]);
      return {p: lerp3(S.P[lo], S.P[hi], f), i: lo, f};
    }
    function uvAt(S, at) { const a = S.uv[at.i], b = S.uv[Math.min(S.uv.length - 1, at.i + 1)]; return [lerp(+a[0], +b[0], at.f), lerp(+a[1], +b[1], at.f)]; }
    // integ: lines of pictures (diagram boxes, arrows, figure curves, marks) go faster than letters
    const DRAWN = /^(diagram|figure|shape|arrow|box|circle|underline|strike|check|divider)$/;
    function strokeDur(L, item) { return Math.max(cfg.strokeMin, L / (cfg.writeSpeed * (item && DRAWN.test(String(item.kind || '')) ? cfg.drawSpeedK : 1)) * 1.25 + 0.03); }    // 1.25: min-jerk peak/mean
    // the chalk tip -> where the paw's grasp point goes (out along the board normal by the chalk's length)
    // the tool's working point (chalk tip / felt centre) -> where the paw's grasp point goes: back along the tool's
    // direction by its length (the chalk points into the board, tilted up; the eraser's felt lies flat on it), plus
    // `off` straight out of the board (lifted)
    // r2-boardvis: the stick also leans toward the side away from the body (chalkSide, along -U for the screen-left
    // paw): the fist trails below and beside the tip instead of sitting on the letter just written (2/3 of the hidden
    // ink of the line being written was within 0.06 uv of the tip — under the fist; measured, tools/board_vis_*)
    const chalkSd = () => { const k = K(), id = toolId('chalk'), p = id && k ? k.byId(id) : null; return (p && (p.held === 'l' || p.held === 'r') ? p.held : cfg.side) === 'r' ? 1 : -1; };
    const toolDir = (B, kind) => kind === 'eraser' ? mul3(B.n, -1) : nrm3(add3(add3(mul3(B.n, -1), [0, cfg.chalkUp, 0]), mul3([B.U[0], 0, B.U[2]], chalkSd() * (cfg.chalkSide || 0))));
    const gripOf = (B, tip, off, kind) => add3(sub3(tip, mul3(toolDir(B, kind), kind === 'eraser' ? cfg.faceLen : cfg.tipLen)), mul3(B.n, off || 0));
    const tipOf = (B, gp, kind) => add3(gp, mul3(toolDir(B, kind), kind === 'eraser' ? cfg.faceLen : cfg.tipLen));
    // a held prop is drawn in an upright frame facing the camera, times R(rx,ry,rz)·home rotation (props.glsl
    // propHeldIn): the turns that point the chalk's stick (+y) along its direction / lay the felt (-y) on the board
    const prE = t => { const [a, b, c] = t.map(x => x * TAU), ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
      return [[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa], [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa], [-sb, cb * sa, cb * ca]]; };   // Rz·Ry·Rx (rows)
    const mm = (A, B2) => A.map((r, i) => [0, 1, 2].map(j => r[0] * B2[0][j] + r[1] * B2[1][j] + r[2] * B2[2][j]));
    const tr3 = A => [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
    function aimTurns(p, B, kind, gp) {
      const C = cfg.camPos, f0 = [C[0] - gp[0], 0, C[2] - gp[2]], fw = nrm3(f0), up = [0, 1, 0], gx = cross(up, fw);
      let yw, xw;
      if (kind === 'eraser') { yw = B.n; xw = cfg.sweepDir === 'horizontal' ? B.V : B.U; }   // r2-erase: the long side across the strokes
      else { yw = toolDir(B, kind); xw = nrm3(cross(yw, B.n)); if (!(len3(cross(yw, B.n)) > 1e-3)) xw = B.U; }
      const zw = cross(xw, yw), Gt = v => [dot3(gx, v), dot3(up, v), dot3(fw, v)];
      const Rt = tr3([Gt(xw), Gt(yw), Gt(zw)]);                       // columns = the wanted unit axes in G
      const hr = (p.def.home && p.def.home.rot) || [0, 0, 0], Rr = mm(Rt, tr3(prE(hr.map(Number))));
      const b = Math.asin(clamp(-Rr[2][0], -1, 1)), a = Math.atan2(Rr[2][1], Rr[2][2]), c = Math.atan2(Rr[1][0], Rr[0][0]);
      return [a / TAU, b / TAU, c / TAU];
    }
    function aimTool(kind, s, B, blend) {
      const k = K(), id = toolId(kind), p = id ? k.byId(id) : null; if (!p || !p.held) return;
      const want = aimTurns(p, B, kind, k.fwdW(s, k.E.paw[s].h));
      if (blend === undefined || blend >= 1) { p.hrot = want; return; }
      const h0 = p.aim0 || (p.aim0 = p.hrot.slice());
      p.hrot = h0.map((v, i) => v + wrap(want[i] - v) * blend);
    }
    function lookAt(tip) { const k = K(); k.E.look.wT = 1; k.E.look.P = k.toLocal ? localLook(tip) : tip; }
    // the engine's look reads E.look.P against the head's REST point (body frame): hand it the target in that frame
    function localLook(P) { const k = K(); return k.toLocal(P); }

    function writeSteps(B, item, strokes, s, opt) {
      const k = K(), steps = [], paths = strokes.map(st => strokePath(B, st)).filter(S => S.P.length >= 1);
      if (!paths.length) return steps;
      const lean = opt && opt.bend !== undefined ? opt.bend : 0;
      steps.push({name: 'lean in', dur: 0.001, start() { k.E.bendT = {y: 0, p: clamp(lean + cfg.lean, 0, 1)}; }, update() { return true; }});
      // to the first stroke: the paw comes up off the carry pose to a point well off the board, then in to hover
      const first = paths[0].P[0];
      const cid = toolId('chalk'), cp = cid ? k.byId(cid) : null;
      steps.push(k.pawPath(s, gripOf(B, first, 0.07), 0.32, {arc: 0, open: 0.3, turn: 0.1, look: null, name: 'raise chalk', each: u => { lookAt(first); if (cp) { if (u < 0.05) cp.aim0 = null; aimTool('chalk', s, B, mj(u)); } }}));
      let from = gripOf(B, first, 0.07);
      paths.forEach((S, si) => {
        const A = gripOf(B, S.P[0], cfg.hover), dA = len3(sub3(A, from));
        steps.push(penUp(s, from, A, Math.max(cfg.penUpMin, dA / cfg.penUpSpeed), B, S.P[0]));
        steps.push(traceStep(B, item, si + ((opt && opt.first) || 0), S, s));
        from = gripOf(B, S.P[S.P.length - 1], cfg.hover);
      });
      steps.push(penUp(s, from, add3(from, mul3(B.n, 0.05)), 0.18, B, null));
      steps.push({name: 'straighten', dur: 0.001, start() { k.E.bendT = {y: 0, p: 0}; }, update() { return true; }});
      // a step along the board follows: the paw rides with the body, held where it is in the body's frame (integ:
      // was a full retract to the carry pose and a raise again per step: ~1.3 s per stance, 30% of a lesson)
      if (opt && opt.more) steps.push(cfg.holdPawOnStep ? {name: 'hold paw', dur: 0.001, start() { const pw = k.E.paw[s]; pw.hd = pw.h.slice(); pw.tgt = k.fwdW(s, pw.hd); pw.hdFor = pw.tgt; pw.keep = true; pw.ctrlT = 1; }, update() { return true; }}
        : k.retractStep(s, true), {name: 'free paw', dur: 0.001, start() { if (!cfg.holdPawOnStep) k.E.paw[s].keep = false; }, update() { return true; }});
      return steps;
    }
    // the paw onto a world point: the engine's per-frame inverse starts from the last hand target and can stick in a
    // local minimum (the reach map's body keep-outs); when it does, a fresh solve from rest wins (the engine's slew limit
    // keeps the arm from popping)
    function setTgt(s, tgt) {
      const k = K(), pw = k.E.paw[s];
      let r = k.invW(s, tgt, pw.h, 8);
      if (r.err > 0.006) { const r2 = k.invW(s, tgt, [0, 0, 0], 14); if (r2.err < r.err - 0.002) r = r2; }
      pw.tgt = tgt; pw.hd = r.h; pw.hdFor = tgt; pw.err = r.err;
    }
    function penUp(s, A, Bp, dur, B, lookTip, kind) {   // chalk off the board: a short world-straight hop with a little lift
      const k = K();
      return {name: 'pen up', dur, s, start() { const pw = k.E.paw[s]; pw.ctrlT = 1; pw.hd = null; pw.follow = null; pw.keep = true; this.A = A; },
        update(tt) {
          const pw = k.E.paw[s], u = clamp(tt / dur, 0, 1), m = mj(u);
          setTgt(s, add3(lerp3(this.A, Bp, m), mul3(B.n, 0.012 * Math.sin(Math.PI * u))));
          aimTool(kind || 'chalk', s, B);
          if (lookTip) lookAt(lerp3(tipOf(B, this.A, kind), lookTip, m));
          return u >= 1;
        }};
    }
    function traceStep(B, item, si, S, s) {
      const k = K(), dur = strokeDur(S.L, item);
      return {name: 'write', dur, s, start() {
        const pw = k.E.paw[s]; pw.ctrlT = 1; pw.hd = null; pw.follow = null; pw.keep = true;
        this.inkS = 0; this.prevS = null; this.prevG = null; this.miss = 0;
        if (T.stats.firstMark === null && T.req && T.req.t0) T.stats.firstMark = +(sec() - T.req.t0).toFixed(2);
      }, update(tt) {
        const pw = k.E.paw[s];
        // the mark under the chalk: what the paw really reached on the previous frame is drawn now (never ahead of it)
        if (this.prevS !== null) {
          const g = k.fwdW(s, pw.h), err = len3(sub3(g, this.prevG));
          if (err <= cfg.contactTol) { if (this.prevS > this.inkS || (this.prevS === 0 && this.inkS === 0 && !this.dot)) { ink(item, si, S, this.inkS, this.prevS); this.dot = true; } this.inkS = this.prevS; }
          else this.miss++;
        }
        const u = clamp(tt / dur, 0, 1), sArc = mj(u) * S.L, at = pathAt(S, sArc);
        setTgt(s, gripOf(B, at.p, 0)); this.prevS = sArc; this.prevG = pw.tgt;
        pw.open = 0.32; lookAt(at.p); aimTool('chalk', s, B);
        if (u >= 1) {
          if (this.inkS < S.L - 1e-6) { this.tail = (this.tail || 0) + 1; if (this.tail < 4) return false; ink(item, si, S, this.inkS, S.L); this.inkS = S.L; T.stats.misses++; }
          T.stats.strokes++; item._doneN = Math.max(item._doneN || 0, si + 1); return true;
        }
        return false;
      }};
    }
    // ---- the marks: uv polyline pieces, stamped on the playback timeline (flushed in frame())
    function ink(item, si, S, s0, s1) {
      const a = pathAt(S, s0), b = pathAt(S, s1), pts = [uvAt(S, a)];
      for (let i = a.i + 1; i <= b.i; i++) pts.push([+S.uv[i][0], +S.uv[i][1]]);
      pts.push(uvAt(S, b));
      // devloop: where chalk REALLY went (per item, partial items too): an erase-all wipes all of it (an aborted
      // or half-wiped item's marks were missing from the board state and stayed on the board for good)
      { const key = (T.req ? T.req.id : '') + ':' + (item.id || '?'), q = T.inked[key] || [1, 1, 0, 0];
        for (const p of pts) { q[0] = Math.min(q[0], +p[0]); q[1] = Math.min(q[1], +p[1]); q[2] = Math.max(q[2], +p[0]); q[3] = Math.max(q[3], +p[1]); }
        T.inked[key] = q; }
      T.ink.push({item: item.id, stroke: si, s0: +(s0 / Math.max(1e-9, S.L)).toFixed(4), s1: +(s1 / Math.max(1e-9, S.L)).toFixed(4),
        pts: pts.map(q => [+q[0].toFixed(5), +q[1].toFixed(5)]), width: item.width || null, color: item.color || null});
      T.inkPts += pts.length;
      inkMark(pts);   // r2-boardvis
    }
    // r2-erase: the felt's footprint on the board, half sizes in uv (its long side across the strokes)
    function feltHalf(B) {
      const pad = cfg.pad || (propDef('eraser') && propDef('eraser').pad) || [0.078, 0.033];
      const L = +pad[0], Th = +pad[1];
      return cfg.sweepDir === 'horizontal' ? {hu: Th / 2 / B.W, hv: L / 2 / B.Hh} : {hu: L / 2 / B.W, hv: Th / 2 / B.Hh};
    }
    // the wipe path over regions (uv rects): a few broad strokes back and forth (up and down by default), each a shallow
    // arc, spaced by most of the felt's length, the felt's edges reaching the region's edges
    function sweepPath(B, regions, dir) {
      const {hu, hv} = feltHalf(B), path = [], vert = cfg.sweepDir !== 'horizontal';
      for (const R of regions) {
        const u0 = clamp(+R[0], 0, 1), v0 = clamp(+R[1], 0, 1), u1 = clamp(+R[2], 0, 1), v1 = clamp(+R[3], 0, 1);
        // across = the axis the strokes step along (u for vertical strokes), along = the stroke's own axis
        const [c0, c1, hc, a0, a1, ha] = vert ? [u0, u1, hu, v0, v1, hv] : [v0, v1, hv, u0, u1, hu];
        const sp = 2 * hc * cfg.sweepOverlap, ca = c0 + hc * 0.8, cb = c1 - hc * 0.8;
        const n = cb <= ca ? 1 : Math.ceil((cb - ca) / sp - 1e-6) + 1;
        const aa = a0 + ha * 0.6, ab = Math.max(aa, a1 - ha * 0.6), sag = cfg.sweepArc * Math.min(sp, 2 * hc);
        for (let r = 0; r < n; r++) {
          const f0 = n === 1 ? 0.5 : r / (n - 1), c = n === 1 ? (c0 + c1) / 2 : lerp(ca, cb, vert && dir < 0 ? 1 - f0 : f0), pts = [];   // strokes step the way it moves
          for (let q = 0; q <= 4; q++) { const f = q / 4, cc = clamp(c + sag * Math.sin(Math.PI * f) * (r === n - 1 ? -1 : 1), c0, c1), al = lerp(aa, ab, f); pts.push(vert ? [cc, al] : [al, cc]); }
          if (r % 2) pts.reverse();
          path.push(...pts);
        }
      }
      return path;
    }
    function eraseSteps(B, regions, s, opt) {
      const k = K(), steps = [], {hu, hv} = feltHalf(B);
      // the hub stamps a rect ru*0.8 x rv half-size per contact: send the felt's own footprint (a touch generous)
      const ru = hu * 1.1 / 0.8, rv = hv * 1.1;
      const path = sweepPath(B, regions, opt && opt.dir);
      if (!path.length) return steps;
      const S = strokePath(B, path), first = S.P[0];
      const eid = toolId('eraser'), ep = eid ? k.byId(eid) : null;
      // to the first sweep: one hop from wherever the paw is (held at the board after the last column: short)
      const hop = gripOf(B, first, 0.03, 'eraser');
      steps.push({name: 'raise eraser', dur: 0.45, s, start() {
        const pw = k.E.paw[s]; this.A = k.fwdW(s, pw.h); pw.ctrlT = 1; pw.hd = null; pw.follow = null; pw.keep = true;
        if (opt && opt.bend !== undefined) k.E.bendT = {y: 0, p: clamp(opt.bend, 0, 1)};   // the stance's lean (its reach needs it)
        this.dur = clamp(len3(sub3(hop, this.A)) / cfg.eraseRaiseSpeed + 0.12, 0.18, 0.5); if (ep) ep.aim0 = null;
      }, update(tt) {
        const u = clamp(tt / this.dur, 0, 1), m = mj(u);
        setTgt(s, add3(lerp3(this.A, hop, m), mul3(B.n, 0.05 * Math.sin(Math.PI * u))));
        if (ep) aimTool('eraser', s, B, mj(clamp(u * 1.4, 0, 1)));
        lookAt(first); return u >= 1;
      }});
      steps.push(penUp(s, hop, gripOf(B, first, 0, 'eraser'), 0.1, B, first, 'eraser'));
      const dur = Math.max(0.4, S.L / cfg.eraseSpeed) * 1.3;
      const corners = []; for (let i = 4; i < path.length; i += 5) corners.push(S.cum[i]);   // stroke ends (sweepPath: 5 points a stroke; the last one too)
      steps.push({name: 'erase', dur: dur * 2.5, s, start() { const pw = k.E.paw[s]; pw.ctrlT = 1; pw.hd = null; pw.keep = true; this.sArc = 0; this.tt = 0; this.prevQ = null; if (opt && opt.onLast) opt.onLast(); }, update(tt) {
        // the felt wipes where it REALLY is (the paw's grasp point -> the felt face, projected on the board) whenever it
        // touches the board plane: each frame's piece of its track is one swept stamp (no gaps at speed); the path runs
        // on at eraseSpeed while the paw keeps up (lag < lagMax), slower when it falls behind
        const pw = k.E.paw[s], dt = Math.max(0, tt - this.tt); this.tt = tt;
        const g = k.fwdW(s, pw.h), felt = tipOf(B, g, 'eraser'), rel = sub3(felt, B.O);
        const dPlane = Math.abs(dot3(rel, B.n)), fu = dot3(rel, B.U) / B.W, fv = dot3(rel, B.V) / B.Hh;
        const lag = this.prev ? len3(sub3(g, this.prev.tgt)) : 0, on = dPlane <= cfg.contactTol * 1.6;
        const keep = !this.prev || lag <= cfg.eraseLagMax || tt > dur * 2.2;
        // a stroke slows into its turn (a hand reverses at the end of a wipe; at full speed the paw cut the corners: the
        // strokes' ends were never wiped)
        let dc = 1; for (const c of corners) dc = Math.min(dc, Math.abs(c - this.sArc));
        this.sArc = Math.min(S.L, this.sArc + dt * cfg.eraseSpeed * (keep ? 1 : 0.3) * (0.35 + 0.65 * clamp(dc / cfg.turnSlowH, 0, 1)));
        const sArc = this.sArc, u = sArc / Math.max(1e-9, S.L), at = pathAt(S, sArc);
        if (this.prev && on && fu > -hu && fu < 1 + hu && fv > -hv && fv < 1 + hv) {
          const q = [+fu.toFixed(5), +fv.toFixed(5)], lq = this.prevQ;
          if (!lq || Math.abs(q[0] - lq[0]) > hu * 0.25 || Math.abs(q[1] - lq[1]) > hv * 0.25) {
            T.ink.push(lq ? {erase: true, sweep: true, pts: [lq, q], ru: +ru.toFixed(4), rv: +rv.toFixed(4)} : {erase: true, pts: [q], ru: +ru.toFixed(4), rv: +rv.toFixed(4)});
            inkWipe([Math.min(q[0], (lq || q)[0]) - ru / 2, Math.min(q[1], (lq || q)[1]) - rv / 2, Math.max(q[0], (lq || q)[0]) + ru / 2, Math.max(q[1], (lq || q)[1]) + rv / 2]); this.prevQ = q; this.n = (this.n || 0) + 1;
          }
        } else { if (this.prev && !on) T.stats.eraseMiss = (T.stats.eraseMiss || 0) + 1; this.prevQ = null; }
        setTgt(s, gripOf(B, at.p, 0, 'eraser')); lookAt(at.p); aimTool('eraser', s, B);
        this.prev = {tgt: pw.tgt, s: sArc};
        if (u >= 1 && this.tEnd === undefined) this.tEnd = tt;   // the path is done: the paw catches up to its end (<= 0.3 s)
        const out = tt >= dur * 2.5 || (this.tEnd !== undefined && (len3(sub3(g, gripOf(B, S.P[S.P.length - 1], 0, 'eraser'))) <= cfg.contactTol || tt - this.tEnd > 0.4));   // the paw AT the path's end (was: at the previous frame's target)
        if (out && this.tEnd === undefined) { T.stats.eraseCut = (T.stats.eraseCut || 0) + 1; log({teach: 'erase_cut', s: +sArc.toFixed(3), L: +S.L.toFixed(3)}); }
        return out;
      }, end() { T.stats.erases++; }});
      const last = S.P[S.P.length - 1];
      steps.push(penUp(s, gripOf(B, last, 0, 'eraser'), gripOf(B, last, 0.04, 'eraser'), 0.12, B, null, 'eraser'));
      if (opt && opt.more) steps.push({name: 'hold paw', dur: 0.001, start() { const pw = k.E.paw[s]; pw.hd = pw.h.slice(); pw.tgt = k.fwdW(s, pw.hd); pw.hdFor = pw.tgt; pw.keep = true; pw.ctrlT = 1; }, update() { return true; }});   // a step along the board follows: the paw rides with the body
      // the last column: the paw is let go to its carry pose as whatever comes next starts (the chalk's raise, the turn to
      // the user) — a separate retract held the next item up by ~1 s
      else steps.push({name: 'straighten', dur: 0.001, start() { k.E.bendT = {y: 0, p: 0}; }, update() { return true; }},
        cfg.eraseRetract ? k.retractStep(s, true) : {name: 'let go', dur: 0.001, start() { const pw = k.E.paw[s]; pw.keep = false; pw.ctrlT = 0; }, update() { return true; }},
        {name: 'free paw', dur: 0.001, start() { k.E.paw[s].keep = false; }, update() { return true; }});
      return steps;
    }

    // ------------------------------------------------------------------ the director
    const STOP_RE = /\b(?:stop(?:\s+(?:it|that|now|there|please|writing|teaching))?|never\s?mind|forget (?:it|about it|that)|come back|go back (?:home|to your (?:spot|place))|go home|that'?s enough|enough|cancel(?: that| it)?|leave it)\b/i;
    const NEG_RE = /\b(?:don'?t|do not|never|not|no need to) (?:stop|go back|come back|cancel)\b/i;
    function newReq(request, kind) {
      const id = 'tq' + (++T.reqN) + '_' + Math.round(now() % 1e6);
      T.req = {id, request: String(request || '').slice(0, 400), kind: kind || 'teach', items: [], next: 0, done: false, err: null, t0: sec(), lastItemAt: sec()};
      return T.req;
    }
    const ERASE_RE = /\b(eras\w*|rub\w*|wip\w*|clear\w*|clean\w*|delet\w*|remov\w*|start (?:over|fresh|again)|blank|empty)\b/i;
    function userAskedErase() {
      const H = o.history ? (o.history() || []) : [];
      for (let i = H.length - 1, n = 0; i >= 0 && n < 2; i--) { const x = H[i]; if (!x || x.role !== 'user' || !x.text) continue; n++; if (ERASE_RE.test(String(x.text))) return true; }
      return !o.history;   // no transcript to check (tests): allowed
    }
    function ctx() {
      const H = o.history ? (o.history() || []) : [];
      return H.filter(x => x && x.text).slice(-12).map(x => ({role: x.role, text: String(x.text).slice(0, 500)}));
    }
    // devloop (r2 chemistry: a lesson asked for while the last one's page-turn wipe ran never wrote a line — the
    // old wipe's item stayed T.item forever and the eraser in the writing paw): a new request ends the old one's own
    // jobs, item and line here; goBoard then puts the eraser back before the chalk
    function dropLesson(why) {
      const live = [...T.jobs.values()].filter(J => !J.done && J.why !== 'home');
      if (live.length) { const k = K(); for (const J of live) { J.mine = true; if (k) k.cancel(J.id, why); } }
      if (T.say && T.say.h && T.say.h.cancel) { try { T.say.h.cancel(); } catch (e) {} }
      T.item = null; T.say = null; T.prefired = null; T.paused = null; T.wipe = null; T.wiping = null;
      const k = K(); if (k) { k.E.bendT = {y: 0, p: 0}; for (const sd of ['l', 'r']) k.E.paw[sd].keep = false; }
    }
    function teachBoard(a) {
      if (!has()) return {ok: false, reason: 'there is no chalkboard here'};
      const request = String(a && (a.request || a.what || a.description) || '').trim();
      if (!request) return {ok: false, reason: 'say what to put on the board'};
      if (T.req && !T.req.done) send({type: 'teach_cancel', id: T.req.id, why: 'replaced'});
      dropLesson('replaced by a new lesson');
      const R = newReq(request, 'teach'); T.stats.calls++; T.idleHome = null;
      send({type: 'teach_plan', id: R.id, request: R.request, context: ctx(), area: area(), board: T.boardState ? T.boardState.items || null : null});
      const walking = goBoard();
      T.phase = 'going'; T.idleAt = sec();
      const did = walking ? 'walking over to the board and picking up the chalk' : 'at the board with the chalk';
      return {ok: true, status: 'starting', did, board: boardBrief(),
        next: 'Tell the user in a few words that you will put it on the board (no details yet). Silent notes will give you each line to say as you write it: say only that line each time, then stop.'};
    }
    function boardErase(a) {
      if (!has()) return {ok: false, reason: 'there is no chalkboard here'};
      const what = String(a && (a.what || a.region || a.description) || 'all').trim() || 'all';
      // devloop (a remote GPU 720p run: "Thanks." split a turn and the voice erased the board 2.8 s after asking for a
      // drawing — the drawing never came): an erase over a lesson in progress needs the USER's words asking to erase; the lesson
      // makes its own room on the board
      if (T.req && T.req.kind !== 'erase' && !userAskedErase()) {   // T.req lives until the lesson is performed (done = only fully planned)
        log({teach: 'erase_refused', why: 'lesson in progress', what});
        return {ok: false, status: 'not_done', reason: 'you are in the middle of putting a lesson on the board and the user did not ask to erase: the lesson makes its own room, nothing was erased. Carry on.'};
      }
      if (T.req && !T.req.done) send({type: 'teach_cancel', id: T.req.id, why: 'replaced by an erase'});
      dropLesson('replaced by an erase');
      const R = newReq(what, 'erase'); T.stats.calls++;
      send({type: 'teach_erase', id: R.id, what, context: ctx(), area: area(), board: T.boardState ? T.boardState.items || null : null});
      goBoard({noChalk: true});
      T.phase = 'going';
      return {ok: true, status: 'starting', did: 'going to the board to erase ' + (/^(all|everything|the board|board)$/i.test(what) ? 'it' : what), board: boardBrief()};
    }
    function goHome(a) {
      const why = String(a && a.why || 'asked to go back');
      if (T.at === 'home' && !T.req && !T.jobs.size) return {ok: true, already: true, did: 'already at its spot'};
      abort(why);
      return {ok: true, status: 'starting', did: 'putting the chalk back and walking back to its spot'};
    }
    // the band of the board the chalk reaches (the scene's station.reach_uv_y): the brain lays out inside it
    function area() {
      const B = T.board, st = B && B.pl.station || {}, r = Array.isArray(st.reach_uv_y) ? st.reach_uv_y.map(Number) : null;
      const v0 = r && r[0] >= 0 && r[0] < 0.6 ? r[0] : 0, v1 = r && r[1] > v0 + 0.2 && r[1] <= 1 ? r[1] : 1;
      return {v0: +v0.toFixed(4), v1: +v1.toFixed(4), aspect: +(B.W / B.Hh).toFixed(4)};
    }
    function boardBrief() {
      const B = T.boardState; if (!B) return 'the board is empty';
      return B.summary || 'the board is empty';
    }
    // walk to the board (standing up first, a paw freed by the planner) and take the chalk; returns true if anything moves
    function goBoard(opt) {
      opt = opt || {};
      const chalk = opt.noChalk ? null : toolId('chalk'), B = T.board;   // devloop: an erase goes without the chalk (it fetched it only to put it down again: ~6 s)
      T.userProps.clear();   // devloop: a new lesson takes the board's tools back as its own
      // the eraser off the writing paw first (r2-erase: one in the OTHER paw stays there — it is the duster's paw)
      const er0 = toolId('eraser'), erBack = !opt.noChalk && er0 && held(er0) && (cfg.eraserPaw !== 'other' || held(er0) === cfg.side) ? putBackGoal(er0) : null;
      const goals = [];
      if (erBack) goals.push(erBack);
      // the chalk first: straight to a stance on the board's station line that takes it off the tray with a lean AND writes
      // the board around it (no second step), a brisk grab; no such stance -> the planner's own fetch
      const gs = chalk && !held(chalk) ? grabStance(chalk, cfg.side) : null;
      if (gs) {
        goals.push({g: 'stand_at', pose: gs, why: 'the board', speed: cfg.walkSpeed, tol: 0.02, ytol: 0.015, turnStep: cfg.turnStep});
        goals.push({g: 'steps', id: chalk, did: ['picking up the chalk', 'picked up the chalk'], build: () => grabSteps(chalk, cfg.side)});
        // r2-erase: then the duster with the other paw, one step along the tray (both are never in one reach): it is
        // in hand for any wipe this lesson needs (cfg.grabBoth: 'always' | 'used' = only when the board has chalk on it)
        const oth = cfg.side === 'l' ? 'r' : 'l', er1 = toolId('eraser'), used = !!((T.boardState && (T.boardState.items || []).length) || Object.keys(T.inked).length);
        if (cfg.eraserPaw === 'other' && er1 && !held(er1) && K().byId(er1) && K().byId(er1).where === 'home' && !(K().heldBy && K().heldBy(oth)) && (cfg.grabBoth === 'always' || (cfg.grabBoth === 'used' && used))) {
          const gs2 = grabStance(er1, oth, {writes: false, from: gs});
          if (gs2) goals.push({g: 'stand_at', pose: gs2, why: 'the board', speed: cfg.walkSpeed, tol: 0.02, ytol: 0.015, turnStep: cfg.turnStep, Ts: 0.25, blendYaw: true},
            {g: 'steps', id: er1, did: ['picking up the eraser', 'picked up the eraser'], build: () => grabSteps(er1, oth)});
        }
      } else if (chalk && !held(chalk)) goals.push({g: 'holding', id: chalk, side: cfg.side, speed: cfg.walkSpeed});
      else if (T.at !== 'board' && !opt.noChalk) goals.push({g: 'stand_at', pose: B.station, why: 'the board', speed: cfg.walkSpeed, tol: 0.06, ytol: 0.04});   // an erase walks straight to the eraser
      // devloop (r0 turn 4: tv_show + teach_board for one question -> the board framing cut the TV picture off):
      // while a TV picture is up the board view widens to hold both
      view(T.tvOnAt && sec() - T.tvOnAt < cfg.tvKeepS ? 'board+tv' : 'board');
      if (!goals.length) { T.at = 'board'; setWhere(); return false; }
      if (T.at !== 'board') { T.at = 'going:board'; setWhere(); }
      const J = job('to the board', goals, 'board', ok => { if (ok) { T.at = 'board'; setWhere(); } });
      T.goJob = J;
      return true;
    }
    const writeYaw = B => B.pl.station && Number.isFinite(+B.pl.station.yaw) ? +B.pl.station.yaw : B.yawB;   // how it stands to write
    // r2-erase: opt.writes false = only the grab matters (the duster on its own: any yaw that reaches it)
    function grabStance(id, s, opt) {
      opt = opt || {};
      const B = T.board, k = K(), p = k.byId(id), L = B.pl.station && B.pl.station.line, sy = B.pl.station ? +B.pl.station.yaw || 0 : B.yawB;
      if (!p || !L || !v3(L.from)) return null;
      const writes = opt.writes !== false;
      const gp = k.fW(p, k.gripLocal(p)), fl = v => { const l = Math.hypot(v[0], v[2]) || 1; return [v[0] / l, 0, v[2] / l]; }, nf = fl(B.n);
      const au = clamp(dot3(sub3(gp, B.O), B.U) / B.W, 0, 1), cands = [];
      const st0 = B.pl.station || {}, vr = Array.isArray(st0.reach_uv_y) ? st0.reach_uv_y.map(Number) : [0.1, 0.95], wu = Array.isArray(st0.window_uv) ? Math.abs(+st0.window_uv[0]) * 1.2 : 0.15;
      const here = opt.from || root(), dys = writes ? [0, 0.04, -0.04] : [0, 0.04, -0.04, 0.08, -0.08, 0.14, -0.14, 0.2, -0.2, 0.28, -0.28, 0.36, -0.36];
      for (let a = 0; a <= 1.001; a += 0.04) for (const dn of [0, -0.03, -0.06]) for (const dy of dys) {
        const b = add3(lerp3(v3(L.from), v3(L.to), a), mul3(nf, dn)), r = {x: b[0], y: 0, z: b[2], yaw: sy + dy};
        cands.push({r, a, cost: Math.abs(a - au) * 0.3 + Math.hypot(r.x - (writes ? T.home.x : here.x), r.z - (writes ? T.home.z : here.z)) * (writes ? 0.2 : 0.9) + Math.abs(dn) + Math.abs(dy) * (writes ? 1 : 0.3)});
      }
      cands.sort((x, y) => x.cost - y.cost);
      for (const c of cands) {
        if (k.standClear && !k.standClear(c.r.x, c.r.z, null)) continue;
        if (k.bendSolve(c.r, gp, [s], false).e > 0.015) continue;
        // and it writes the upper part of the board around the chalk's column from there
        if (writes && !reachFrom(B, c.r, [[clamp(au - wu, 0, 1), vr[0] + 0.04], [clamp(au + wu * 0.3, 0, 1), vr[0] + 0.04], [clamp(au - wu, 0, 1), 0.6], [clamp(au + wu * 0.3, 0, 1), 0.6]], s)) continue;
        return {x: c.r.x, z: c.r.z, yaw: c.r.yaw};
      }
      return null;
    }
    function grabSteps(id, s) {   // lean, reach, close on it, lift it up toward the board (no carry pose in between)
      const k = K(), p = k.byId(id); if (!p || p.held) return [];
      const g = k.gripPick ? k.gripPick(p, s) : null, gp = k.fW(p, g ? g.g : k.gripLocal(p)), r = k.rootNow();
      const b = k.bendSolve({x: r.x, y: 0, z: r.z, yaw: r.yaw}, gp, [s], false).b;
      // the lean and the reach together (the reach is a world path, solved every frame against the bend as it goes)
      return [k.lookStep(p.pos, true),
        k.par([{name: 'lean', dur: 0.3, start() { k.E.bendT = {y: b.y, p: b.p}; }, update(tt) { return tt >= 0.3; }},
          k.pawPath(s, add3(gp, [0, 0.035, 0.01]), 0.4, {arc: 0, open: 0.65, turn: 0.1, name: 'reach'})]),
        k.pawPath(s, gp, 0.12, {arc: 0, name: 'reach in'}),
        k.graspStep(s, p, false, g ? g.g : undefined),
        {name: 'lift', dur: 0.001, start() { k.E.bendT = {y: 0, p: 0}; k.E.paw[s].keep = true; }, update() { return true; }},
        k.pawPath(s, add3(add3(gp, [0, 0.1, 0]), mul3(T.board.n, 0.05)), 0.22, {arc: 0, name: 'lift'}), k.lookStep(null, false)];
    }
    function setWhere() {
      const k = K(); if (!k) return;
      k.E.teachWhere = T.at === 'board' ? 'at the board' : T.at === 'tv' ? 'at the TV' : T.at === 'going:board' ? 'walking over to the board' : T.at === 'going:tv' ? 'walking over to the TV' : T.at === 'going:home' ? 'walking back to its spot' : null;
      k.E.teachDoing = T.phase === 'writing' ? 'writing on the board' : T.phase === 'erasing' ? 'erasing the board' : T.phase === 'presenting' ? 'showing the TV' : null;
    }
    // the station camera (SCENE: engine.setView): eased in when it sets off for a station, back to the follow framing
    // when it heads home
    function view(name) {
      if (!cfg.views) return;
      const e = eng(); if (!e || !e.setView) return;
      try { const r = e.setView(name); if (r && r.ok) T.view = name; } catch (e2) {}
    }
    // the band of board uv-x its body hides from the camera when it stands at (x, z): rays from the camera past either
    // side of the body (at chest height) onto the board plane
    function hiddenU(B, x, z) {
      const C = cfg.camPos, P = [x, 0.6, z], d = nrm3(sub3(P, C)), side = nrm3(cross([0, 1, 0], d)), us = [];
      for (const sg of [-1, 1]) {
        const Q = add3(P, mul3(side, sg * cfg.bodyR)), r = sub3(Q, C), den = dot3(r, B.n);
        if (Math.abs(den) < 1e-6) return null;
        const sP = dot3(sub3(B.O, C), B.n) / den; if (sP < 1) return null;   // the board is not behind it
        us.push(dot3(sub3(add3(C, mul3(r, sP)), B.O), B.U) / B.W);
      }
      return [Math.min(us[0], us[1]), Math.max(us[0], us[1])];
    }
    // r2-boardvis: the muse's REAL silhouette on the board (hiddenU's one chest-height cylinder said the pen was
    // clear while the head hid it: 44% of an item hidden as it was written, 14% of the board while it talked, measured
    // with the renderer — tools/board_vis_*). Spheres + arm capsules in the body frame, fitted to rendered occlusion
    // masks (tools/board_sil_fit.py: IoU 0.90), each projected from the camera onto the board plane as an ellipse in uv.
    // st = {x, z, yaw, bend}; paws = {l, r}: world points of the paws (null = hanging at its sides)
    function silDiscs(B, st, paws, noPaw) {
      const S = cfg.sil, C = cfg.camPos, a = TAU * (st.yaw || 0), c = Math.cos(a), sn = Math.sin(a), ba = 0.28 * (st.bend || 0), cb = Math.cos(ba), sb = Math.sin(ba);
      const W = l => { let [x, y, z] = l; if (y > 0.3) { const yy = y - 0.3; [y, z] = [0.3 + yy * cb - z * sb, yy * sb + z * cb]; } return [st.x + c * x + sn * z, y, st.z - sn * x + c * z]; };
      const den0 = dot3(sub3(B.O, C), B.n), out = [];
      const disc = (P, r) => {
        const d = sub3(P, C), dn = dot3(d, B.n); if (Math.abs(dn) < 1e-6) return;
        const s = den0 / dn; if (s < 1) return;                      // behind the board
        const X = add3(C, mul3(d, s)), rel = sub3(X, B.O), ci = Math.max(0.5, Math.abs(dn) / len3(d));
        out.push([dot3(rel, B.U) / B.W, dot3(rel, B.V) / B.Hh, r * s / B.W / ci, r * s / B.Hh]);
      };
      for (const [l, r] of [[[S.hx, S.hy, S.hz], S.hr], [[0, S.cy, 0], S.cr], [[0, S.py, 0], S.pr], [[S.fx, S.fy, 0], S.fr], [[-S.fx, S.fy, 0], S.fr], [[S.earx, S.eary, 0], S.earr], [[-S.earx, S.eary, 0], S.earr]]) disc(W(l), r);
      for (const [sd, sx] of [['l', -1], ['r', 1]]) {
        const Sh = W([sx * S.sx, S.sy, 0]), Pp = paws && paws[sd] ? paws[sd] : W([sx * 0.21, 0.2, 0.08]);
        for (let i = 0; i <= 5; i++) { const f = i / 5; if (noPaw === sd && f > 0.7) continue; disc(lerp3(Sh, Pp, f), S.ar + (S.pw - S.ar) * f); }
      }
      return out;
    }
    const silHit = (D, u, v, pad) => { for (const q of D) { const du = (u - q[0]) / (q[2] + (pad || 0)), dv = (v - q[1]) / (q[3] + 2 * (pad || 0)); if (du * du + dv * dv <= 1) return true; } return false; };
    // the share of uv points (ink samples) its body hides from stance st (writing with paw s at grip point `pen`, or not)
    function silHidden(B, st, uvs, pen, s) {
      if (!uvs || !uvs.length) return 0;
      const D = silDiscs(B, st, pen ? {[s || cfg.side]: pen} : null, pen ? (s || cfg.side) : null);   // (the paw on the pen is not counted: it is where the chalk is)
      let n = 0; for (const q of uvs) if (silHit(D, +q[0], +q[1], cfg.silPad)) n++;
      return n / uvs.length;
    }
    // what ink the board holds (where its own chalk went this session, minus what it wiped): coarse cells for the
    // visibility checks (the board brain's bboxes are too coarse: a column's box is half empty)
    const INK_GU = 96, INK_GV = 48;
    function inkMark(pts) { const g = T.inkCells || (T.inkCells = new Set()); for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[Math.min(pts.length - 1, i + 1)]; const n = 1 + Math.ceil(Math.max(Math.abs(b[0] - a[0]) * INK_GU, Math.abs(b[1] - a[1]) * INK_GV) * 2); for (let j = 0; j <= n; j++) { const u = lerp(+a[0], +b[0], j / n), v = lerp(+a[1], +b[1], j / n); g.add(clamp(Math.floor(v * INK_GV), 0, INK_GV - 1) * INK_GU + clamp(Math.floor(u * INK_GU), 0, INK_GU - 1)); } } }
    function inkWipe(r) { const g = T.inkCells; if (!g) return; for (const c of [...g]) { const u = (c % INK_GU + 0.5) / INK_GU, v = (Math.floor(c / INK_GU) + 0.5) / INK_GV; if (u >= r[0] && u <= r[2] && v >= r[1] && v <= r[3]) g.delete(c); } }
    function inkUV() { const g = T.inkCells; if (!g || !g.size) return []; return [...g].map(c => [(c % INK_GU + 0.5) / INK_GU, (Math.floor(c / INK_GU) + 0.5) / INK_GV]); }
    // where to stand to talk about what is written: on the station line (or its step-aside spot) with its silhouette
    // off the text (`box` uv), as near as possible to where it is now; null = nowhere clear (stay)
    // r2-boardvis: clear of the INK (what the board holds, sampled), by its real silhouette in the pose it will talk
    // in (yaw); the nearest such spot (a short walk beats a hidden board); none clear -> the least hidden one
    function clearStance(B, pts, yaw) {
      const L = B.pl.station && B.pl.station.line, sy = writeYaw(B), here = root(), out = [];
      if (!pts || !pts.length || !L || !v3(L.from) || !v3(L.to)) return null;
      const k = K(), cands = [], yw = yaw === undefined ? sy : yaw, rg = cfg.clearRange || [-0.1, 1.25];
      for (let a = rg[0]; a <= rg[1] + 1e-4; a += 0.025) { const b = lerp3(v3(L.from), v3(L.to), a); cands.push({x: b[0], z: b[2]}); }
      const as = B.pl.station.aside && v3(B.pl.station.aside); if (as) cands.push({x: as[0], z: as[2], aside: true});
      for (const c of cands) {
        if (k.standClear && !k.standClear(c.x, c.z, null)) continue;
        const h = silHidden(B, {x: c.x, z: c.z, yaw: yw}, pts);
        out.push({x: c.x, z: c.z, yaw: sy, hid: h, cost: Math.hypot(c.x - here.x, c.z - here.z) + (c.aside ? 0.1 : 0) + 6 * h + (h > cfg.sayHideMax ? 1 : 0)});
      }
      out.sort((p, q) => p.cost - q.cost);
      return out[0] || null;
    }
    // what to reveal between items: the whole lesson so far when a stance near enough clears it, else the last item
    // r2-boardvis: always the whole board (all the ink it holds; faceUser checks it by the real silhouette)
    function revealBox(R) {
      return boxOf(R.wrote || []) || (T.inkCells && T.inkCells.size ? [0, 0, 1, 1] : null);
    }
    function boxOf(ids) {
      let b = null;
      for (const id of ids) { const q = T.drawn[id]; if (!q) continue; b = b ? [Math.min(b[0], q[0]), Math.min(b[1], q[1]), Math.max(b[2], q[2]), Math.max(b[3], q[3])] : q.slice(); }
      return b;
    }
    function goStation(id, opt) {
      opt = opt || {};
      if (id === 'board') { goBoard(); return {ok: true}; }
      if (id === 'home') { if (/tv is off/i.test(String(opt.why || ''))) T.tvOnAt = 0; abort(opt.why || 'going home'); return {ok: true}; }
      const S = id === 'tv' ? T.tv : null; if (!S || !K()) return {ok: false, reason: 'no ' + id + ' here'};
      if (id === 'tv') T.tvOnAt = sec();
      if (T.req) { send({type: 'teach_cancel', id: T.req.id, why: 'to the ' + id}); T.req = null; }
      cancelJobs('to the ' + id);
      const goals = [];
      const chalk = toolId('chalk'), er = toolId('eraser');
      // integ: the chalk stays in the paw (a teacher walks to the screen with it; the tray detour made the walk to
      // the TV ~10 s); the eraser goes back. Home puts everything back.
      if (er && held(er)) { const g = putBackGoal(er); if (g) goals.push(g); }
      void chalk; T.idleHome = null;
      goals.push({g: 'stand_at', pose: S.station, why: 'the ' + (S.pl.name || id).replace(/^the /, ''), speed: cfg.walkSpeed, tol: 0.06, ytol: 0.04});
      // integ: the presenting gesture is part of the same job (a board request queued right behind the TV walk used
      // to get the gesture AFTER it, at the board), and neither end clobbers a newer request's state
      if (opt.present !== false) goals.push(presentGoal(S, id));
      T.at = 'going:' + id; T.phase = 'going'; setWhere(); view(id);
      job('to the ' + id, goals, id, ok => { if (!ok || T.req) return; T.at = id; T.phase = 'idle'; T.idleAt = sec(); setWhere(); });
      return {ok: true, did: 'walking over to the ' + id};
    }
    function presentGoal(S, id) {
      const k = K(), tgt = S.C;
      return {g: 'steps', did: ['pointing out the screen', 'pointed out the screen'], build: () => {
        if (!T.req) { T.at = id; T.phase = 'presenting'; setWhere(); lookUserOn(); }
        const me = k.toWorld([0, 0.5, 0]), dir = nrm3(sub3(tgt, me)), side = k.toLocal(tgt)[0] < 0 ? 'l' : 'r';
        const P = add3(me, mul3(dir, 0.22));
        return [k.pawPath(side, P, 0.6, {open: 0.95, turn: 0.6, name: 'open paw to the screen'}), k.dwellStep('present', cfg.presentS, () => { lookUser(); }), k.retractStep(side)];
      }};
    }
    // a put-back goal for a tool (its home on the tray), built the way the engine's put_back does it
    function putBackGoal(id) {
      const k = K(), p = k.byId(id); if (!p) return null;
      return {g: 'on', id, place: p.homePlace || 'floor', at: p.home.slice(), rot: [0, 0, 0], back: 'home', speed: cfg.walkSpeed};
    }
    /* r2-objvoice: the USER's object request while the lesson's tools are still in its paws (r6: the chalk rode along to the
       TV, then "hold the eraser up" was shown in the other paw beside the chalk, small and cluttered) — a teacher puts the
       chalk / eraser back on the tray before taking up something else. Called by the page right before the user's object command
       runs (it queues first, so the user's job follows it). Not when the thing they name IS that tool, not for a tool they asked
       for earlier (userProps), not while a lesson is being written (the lesson's own yield handles that). -> ids queued */
    function yieldTools(cmd) {
      const k = K(); if (!k || !cmd || cmd.action !== 'object') return [];
      if (T.req || T.phase === 'writing' || T.phase === 'erasing') return [];
      const vb = String(cmd.verb || cmd.name || '').toLowerCase();
      if (/^(stop|turn_back|look|look_at)$/.test(vb)) return [];
      const words = [cmd.object && (cmd.object.id || cmd.object.name), cmd.target, cmd.place].map(x => String(x && typeof x === 'object' ? (x.object || x.place || x.id || x.name || '') : (x || '')).toLowerCase()).join(' ');
      const out = [];
      for (const kind of ['chalk', 'eraser']) {
        const id = toolId(kind); if (!id || !held(id) || T.userProps.has(id)) continue;
        if (T.yieldJ && T.yieldJ[id] && jobAlive(T.yieldJ[id])) continue;                       // already on its way back
        const p = k.byId(id), names = [id, String(p && p.name || '')].concat((propDef(id) || {}).aliases || []).map(s => String(s).toLowerCase()).filter(s => s.length > 2);
        if (names.some(s => words.includes(s)) || (cmd.object && cmd.object.ref === 'it')) continue;   // the user means this one
        const J = job(`putting the ${p && p.name || kind} back`, [putBackGoal(id)], 'yield');
        if (J) { J.mine = true; out.push(id); (T.yieldJ = T.yieldJ || {})[id] = J; }
      }
      if (out.length) log({teach: 'yield tools', ids: out, for: vb});
      // final (a remote GPU run, turn 6: at the TV, "take the eraser and hold it up" -> the chalk walked back to the tray
      // under the TV camera, 6 s of an empty room): putting the tools back already leaves the station — the follow framing
      if (out.length && !T.req && T.at && T.at !== 'home' && !String(T.at).startsWith('going')) { T.at = null; T.phase = 'idle'; setWhere(); view(null); log({teach: 'left station', why: 'tools back for user object request'}); }
      return out;
    }
    function abort(why, opt) {
      opt = opt || {};
      const k = K(); if (!k) return;
      if (opt.soft && T.at !== 'home' && !String(T.at || '').startsWith('going')) return softStop(why);
      // devloop (r1: every growing partial of "wait wait stop never mind come back" restarted the walk home, 8 times
      // in 2 s): already on its way home with nothing else going on -> nothing to do
      if (!T.req && T.at === 'going:home' && [...T.jobs.values()].some(J => J.why === 'home' && !J.done)) { log({teach: 'abort_dup', why}); return; }
      T.aborting = true;
      try {
        T.stats.aborts++; T.abortAt = sec();
        if (T.req) { send({type: 'teach_cancel', id: T.req.id, why}); T.req = null; }
        if (T.say && T.say.h && T.say.h.cancel) { try { T.say.h.cancel(); } catch (e) {} }
        T.say = null; T.item = null; T.prefired = null; T.wipe = null; T.wiping = null;
        cancelJobs(why);
        lookUserOff();
        k.E.turnBack = false; k.E.bendT = {y: 0, p: 0};
        for (const sd of ['l', 'r']) k.E.paw[sd].keep = false;
        const goals = [];
        for (const q of [toolId('chalk'), toolId('eraser')]) if (q && !T.userProps.has(q) && (held(q) || (k.byId(q) && k.byId(q).where !== 'home'))) { const g = putBackGoal(q); if (g) goals.push(g); }
        goals.push({g: 'stand_at', pose: T.home, why: 'its spot', speed: cfg.walkSpeed, tol: 0.03, ytol: 0.02,
          did: ['walking back to its spot', 'walked back to its spot']});
        T.at = 'going:home'; T.phase = 'returning'; setWhere(); view(null);
        // integ: a newer request (teach_board / tv_show right after the stop) owns the state by the time this ends
        const J = job('home', goals, 'home', ok => { T.homeAt = sec(); if (T.req || T.at !== 'going:home') return; if (!ok) { if (![...T.jobs.values()].some(j => j.why === 'home')) { T.at = null; T.phase = 'idle'; setWhere(); } return; } T.at = 'home'; T.phase = 'idle'; setWhere(); });   // a cut-short walk home is superseded by the next (stop + go_home both send it home)
        if (J) J.mine = true;
        log({teach: 'abort', why});
      } finally { T.aborting = false; }
    }
    // integ: a change of subject (not a stop) ends the lesson where it stands: the stroke stops, the chalk stays in
    // the paw, it turns to the user; a new board request carries on from here, else it walks home after softHomeS
    function softStop(why) {
      const k = K();
      T.stats.aborts++; T.abortAt = sec();
      if (T.req) { send({type: 'teach_cancel', id: T.req.id, why}); T.req = null; }
      if (T.say && T.say.h && T.say.h.cancel) { try { T.say.h.cancel(); } catch (e) {} }
      T.say = null; T.item = null; T.prefired = null; T.paused = null; T.wipe = null; T.wiping = null;
      cancelJobs(why);
      k.E.bendT = {y: 0, p: 0};
      for (const sd of ['l', 'r']) k.E.paw[sd].keep = false;
      T.phase = 'idle'; T.idleAt = sec(); T.idleHome = cfg.softHomeS; setWhere();
      faceUser(false);
      log({teach: 'soft_stop', why});
    }
    // integ: the user started talking while it writes: the chalk comes off the board at once (a teacher stops and
    // listens); the item resumes from its first unwritten stroke when they are done, unless their words stop the lesson
    function userSpeech(ph) {
      if (T.at && T.at !== 'home') T.idleAt = sec();   // integ: still talking with it at a station: no walk home yet
      if (ph !== 'start' || !cfg.pauseOnSpeech) return false;
      const k = K(), it = T.item;
      if (!k || !T.req || T.phase !== 'writing' || !it) return false;
      it._paused = true;
      const J = [...T.jobs.values()].find(j => j.why === 'write');
      if (J) { J.mine = true; k.cancel(J.id, 'user is talking'); }
      k.E.bendT = {y: 0, p: 0};
      for (const sd of ['l', 'r']) k.E.paw[sd].keep = false;
      T.item = null; T.phase = 'ready'; T.pausedAt = sec(); T.stats.pauses = (T.stats.pauses || 0) + 1;
      T.heldAt = T.heldAt || sec(); T.userAt = sec();
      lookUserOn(); setWhere();
      log({teach: 'pause', item: it.id, done: it._doneN || 0});
      return true;
    }
    // the look at the user from wherever the body is turned (updated every frame while on)
    function lookUserOn() { T.look = 'user'; }
    function lookUserOff() { if (T.look) { T.look = null; const k = K(); if (k) k.E.look.wT = 0; } }
    function lookUser() { const k = K(); if (!k) return; const cam = [k.E.cam ? k.E.cam.x : 0, 0.62, 4.0]; k.E.look.wT = 1; k.E.look.P = k.toLocal(cam); }

    // ---- saying a line (the voice model words it; motion never waits long on it)
    function estSay(text) { return clamp(String(text || '').length / cfg.charsPerS, 0.6, 8); }
    function startSay(item, mode) {
      const line = String(item.say_before || '').trim();
      const S = {item: item.id, line, at: sec(), est: estSay(line), audioAt: null, doneAt: null, dropped: null, mode};
      item.said = true; T.say = S;
      const note = `Say now, in your own natural words, only this (one short sentence, then stop; you will write it next): "${line}"`;
      if (o.prompt) {
        try {
          S.h = o.prompt(note, {respond: true, maxWaitMs: 3500, dropIfUserSpoke: true, maxChars: 500,
            onAudio: () => { S.audioAt = sec(); }, onDone: () => { S.doneAt = sec(); }, onDrop: why => { S.dropped = why || 'dropped'; S.doneAt = sec(); }});
          if (!S.h) { S.dropped = 'no voice'; S.doneAt = sec(); }
        } catch (e) { S.dropped = 'error'; S.doneAt = sec(); }
      } else { S.dropped = 'no voice'; S.doneAt = sec(); }
      log({teach: 'say', item: item.id, mode, line: line.slice(0, 80)});
      return S;
    }
    // the timing rule: write when the line is ~70% through (its audio running that long), when it ended, when it was
    // dropped, or when no audio came in time; never more than sayCapS of a long line
    function sayReady(S) {
      if (!S) return true;
      const t = sec();
      if (S.doneAt !== null) return t - S.doneAt > (S.dropped ? 0.15 : 0);
      if (S.audioAt !== null) {
        const heard = (o.audible ? (o.audible() ? 1 : 0) : 1);
        return t - S.audioAt >= Math.min(cfg.sayCapS, cfg.sayFrac * S.est) || (!heard && t - S.audioAt > 0.8);
      }
      return t - S.at >= cfg.sayNoAudioS;
    }
    function nextItem(R) { return R && R.next < R.items.length ? R.items[R.next] : null; }
    function writeItem(R, item) {
      const B = T.board, k = K();
      const s0 = held(toolId('chalk')) || cfg.side;
      const skip = item._doneN || 0;   // integ: strokes already written before a pause
      const strokes = (item.strokes || []).filter(st => Array.isArray(st) && st.length).slice(skip);
      if (!strokes.length) { R.next++; return; }
      // split into groups, each written from one stance: the strokes that follow on from where it stands are written
      // from there; when the next one is out of reach it steps along the board to a stance that covers as many of the
      // coming strokes as it can (the whole rest of the item first, then fewer)
      const groups = [], samples = strokes.map(sampleUV);
      let cur = {x: root().x, z: root().z, yaw: root().yaw}, i = 0;
      while (i < strokes.length) {
        // integ: a stroke joins the current stance only if the body does not hide it there (unless no stance could)
        const G0 = groups[groups.length - 1];
        if (G0 && reachFrom(B, cur, samples[i], s0) && (G0.hides || visFrom(B, cur, samples[i]))) { G0.strokes.push(strokes[i]); i++; continue; }
        if (!groups.length && reachFrom(B, cur, samples[i], s0) && visFrom(B, cur, samples[i]) && Math.abs(wrap(cur.yaw - writeYaw(B))) < 0.12) { groups.push({strokes: [strokes[i]], first: i, stance: Object.assign({bend: 0}, cur), same: true}); i++; continue; }
        let st = null, j = i + 1;
        // the run of strokes that fits one reach window along the board (u extent), then fewer if no stance takes them all
        { let lo = 1, hi = 0; for (let q = i; q < strokes.length; q++) { for (const w of samples[q]) { lo = Math.min(lo, +w[0]); hi = Math.max(hi, +w[0]); } if (hi - lo > cfg.windowU && q > i) break; j = q + 1; } }
        for (; j > i && !st; j = i + Math.floor((j - i) / 2)) { st = stanceFor(B, [].concat(...samples.slice(i, j)), s0, cur); if (st || j - i <= 1) break; }
        if (!st) { log({teach: 'unreachable', item: item.id, stroke: i}); T.stats.misses++; i++; continue; }
        groups.push({strokes: strokes.slice(i, j), first: i, stance: st, hides: !visFrom(B, st, [].concat(...samples.slice(i, j)))}); cur = st; i = j;
      }
      if (!groups.length) { R.next++; T.stats.misses++; return; }
      const goals = [];
      const ch = toolId('chalk');
      if (ch && !held(ch)) goals.push({g: 'holding', id: ch, side: cfg.side, speed: cfg.walkSpeed});   // after an erase: the chalk again
      for (const G of groups) {
        if (!G.same) goals.push({g: 'stand_at', pose: {x: G.stance.x, z: G.stance.z, yaw: G.stance.yaw}, why: 'its place at the board', tol: 0.015, ytol: 0.012,
          did: ['stepping along the board', 'stepped along the board'], Ts: 0.3, blendYaw: true, speed: cfg.writeStepSpeed});
        goals.push({g: 'steps', id: toolId('chalk'), did: [`writing on the board`, `wrote on the board`], build: () => writeSteps(B, item, G.strokes, s0, {bend: G.stance.bend, first: G.first + skip, more: G !== groups[groups.length - 1]})});
      }
      T.phase = 'writing'; setWhere(); lookUserOff(); k.E.turnBack = false;
      item.t0 = sec(); item.est = groups.reduce((a, G) => a + G.strokes.reduce((b, st) => b + strokeDur(strokePath(B, st).L, item) + 0.15, 0.6), 0);
      T.item = item;
      job('writing ' + (item.id || ''), goals, 'write', ok => {
        if (T.req !== R) return;
        if (item._paused) { item._paused = false; return; }   // integ: paused for the user; resumes later
        item.t1 = sec(); T.item = null; R.next++; T.sinceTurn++; T.stats.items++;
        (R.wrote = R.wrote || []).push(item.id || ('i' + T.stats.items));
        T.drawn[item.id || ('i' + T.stats.items)] = (item.strokes || []).flat().reduce((a, q) => [Math.min(a[0], +q[0]), Math.min(a[1], +q[1]), Math.max(a[2], +q[0]), Math.max(a[3], +q[1])], [1, 1, 0, 0]);   // what it wrote where (its own record)
        send({type: 'board_item_done', id: R.id, item: item.id, ok, s: +(item.t1 - item.t0).toFixed(2)});
        T.phase = 'ready'; setWhere();
      });
    }
    function visFrom(B, st, uvs) {   // integ: no point of uvs hidden by its body (r2-boardvis: every point latMin+ beside it)
      if (!(cfg.hideCost > 0) || !uvs.length) return true;
      const sd = cfg.side === 'l' ? 1 : -1, Uf = [B.U[0], 0, B.U[2]], l = Math.hypot(Uf[0], Uf[2]) || 1;
      return uvs.every(q => { const P = B.world(+q[0], +q[1]); return sd * ((st.x - P[0]) * Uf[0] + (st.z - P[2]) * Uf[2]) / l >= cfg.latMin; });
    }
    // how far short of latWant (H) the points (grip points, world) are of standing beside the paw, on average (0 = all clear)
    function latShort(r, pts, s) {
      const B = T.board, sd = s === 'r' ? -1 : 1, Uf = [B.U[0], 0, B.U[2]], l = Math.hypot(Uf[0], Uf[2]) || 1;
      let a = 0; for (const P of pts) a += Math.max(0, cfg.latWant - sd * ((r.x - P[0]) * Uf[0] + (r.z - P[2]) * Uf[2]) / l);
      return a / Math.max(1, pts.length) / 0.1;
    }
    function sampleUV(st) { const out = [st[0]]; const n = st.length; const step = Math.max(1, Math.floor(n / 6)); for (let i = step; i < n; i += step) out.push(st[i]); out.push(st[n - 1]); return out; }
    // ---- the wipe. eraseItem plans the item's columns (one stance each, where there is chalk), then wipes them: all at once
    // for an erase the user asked for, or LAZILY for a lesson's page turn (r2-erase: the first item's columns now, the
    // rest just before an item that lands there, whatever is left at the end of the lesson) — a new lesson on a full
    // board starts writing after one column, not after the whole board (33-35 s live)
    function eraseBoxes(item) {
      const sb = Array.isArray(item.strokes) && item.strokes.length ? item.strokes.flat().reduce((a, q) => [Math.min(a[0], +q[0]), Math.min(a[1], +q[1]), Math.max(a[2], +q[0]), Math.max(a[3], +q[1])], [1, 1, 0, 0]) : null;
      const pad = item.erase_r ? +item.erase_r : 0.04;
      // where there is chalk (what the board holds + what this page inked), each box a little padded
      const src = (T.boardState && (T.boardState.items || []).length ? T.boardState.items : Object.values(T.drawn).map(bb => ({bbox: bb}))).concat(Object.values(T.inked).map(bb => ({bbox: bb})));
      const chalk = src.filter(x => Array.isArray(x.bbox) && x.bbox.length === 4).map(x => [x.bbox[0] - 0.02, x.bbox[1] - 0.03, x.bbox[2] + 0.02, x.bbox[3] + 0.03]);
      // "all": only where there is chalk, else the whole board
      if (item.regions && item.regions.length === 1 && item.regions[0].join() === '0,0,1,1') { item.all = true; return chalk.length ? chalk : [[0, 0, 1, 1]]; }
      const bb = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox.map(Number) : null;
      const area = item.regions || (bb ? [[bb[0] - 0.012, bb[1], bb[2] + 0.012, bb[3]]] : sb ? [[sb[0] - pad / 2, sb[1] - pad, sb[2] + pad / 2, sb[3] + pad]] : [[0, 0, 1, 1]]);
      // r2-erase: a region (a page turn's bbox of everything) is wiped where it holds chalk, not over empty slate
      const cut = [];
      for (const A of area) for (const c of chalk) { const r = [Math.max(A[0], c[0]), Math.max(A[1], c[1]), Math.min(A[2], c[2]), Math.min(A[3], c[3])]; if (r[2] > r[0] + 0.005 && r[3] > r[1] + 0.005) cut.push(r); }
      return cut.length ? cut : area;
    }
    // one stance per COLUMN of the board (the reach window), every row of chalk in that column wiped in one pass
    function eraseCols(raw) {
      const boxes = raw.map(r => [clamp(+r[0], 0, 1), clamp(+r[1], 0, 1), clamp(+r[2], 0, 1), clamp(+r[3], 0, 1)]).filter(r => r[2] > r[0] && r[3] > r[1]);
      const cols = [];
      if (!boxes.length) return cols;
      const ua = Math.min(...boxes.map(r => r[0])), ub = Math.max(...boxes.map(r => r[2]));
      const cw = cfg.eraseColU, nC = Math.max(1, Math.ceil((ub - ua) / cw - 1e-6));
      for (let q = 0; q < nC; q++) {
        const c0 = ua + (ub - ua) * q / nC, c1 = ua + (ub - ua) * (q + 1) / nC;
        const hit = boxes.filter(r => r[2] > c0 && r[0] < c1).map(r => [Math.max(c0, r[0]), r[1], Math.min(c1, r[2]), r[3]]).sort((a, b) => a[1] - b[1]);
        const merged = [];
        for (const r of hit) { const m = merged[merged.length - 1]; if (m && r[1] <= m[3] + 0.06) { m[0] = Math.min(m[0], r[0]); m[2] = Math.max(m[2], r[2]); m[3] = Math.max(m[3], r[3]); } else merged.push(r.slice()); }
        if (merged.length) cols.push({regs: merged, u0: c0, u1: c1, done: false});
      }
      return cols;
    }
    const boxHits = (a, b, pad) => a[0] < b[2] + pad && b[0] < a[2] + pad && a[1] < b[3] + pad && b[1] < a[3] + pad;
    function itemBox(it) {
      if (Array.isArray(it.bbox) && it.bbox.length === 4) return it.bbox.map(Number);
      const st = (it.strokes || []).flat(); if (!st.length) return null;
      return st.reduce((a, q) => [Math.min(a[0], +q[0]), Math.min(a[1], +q[1]), Math.max(a[2], +q[0]), Math.max(a[3], +q[1])], [1, 1, 0, 0]);
    }
    // the next content item of R after index i (null = none yet)
    function nextContent(R, i) { for (let j = i; j < R.items.length; j++) { const x = R.items[j]; if (x.kind === 'erase' || x.erase) return null; if ((x.strokes || []).length) return x; } return null; }
    function eraseItem(R, item) {
      // a page turn inside a lesson (more items follow): lazy — the columns the next item lands in first
      const lazyOk = cfg.lazyTurn && R.kind !== 'erase';
      if (lazyOk && !nextContent(R, R.next + 1) && !R.done && sec() - (item.waitAt || (item.waitAt = sec())) < 2.0) { T.phase = 'ready'; return false; }   // the item to make room for is still on its way
      const raw = eraseBoxes(item);
      log({teach: 'erase', item: item.id, regions: raw.map(r => r.map(x => +(+x).toFixed(3)))});
      const W = {R, item, cols: eraseCols(raw), t0: sec()};
      const lazy = lazyOk && W.cols.length > 1;
      if (lazy) {
        const nx = nextContent(R, R.next + 1);
        const bx = nx && itemBox(nx);
        const now = bx ? W.cols.filter(c => c.regs.some(r => boxHits(r, bx, 0.03))) : [];
        T.wipe = W; R.next++;   // the erase item is "performed" by this and the later wipes
        if (!now.length) { log({teach: 'wipe_deferred', item: item.id, cols: W.cols.length}); T.phase = 'ready'; return true; }
        wipeCols(W, now, false);
        return true;
      }
      wipeCols(W, W.cols, true);
      return true;
    }
    // wipe these columns of W (a job); final = the item's last wipe (then the board state is told it is done)
    function wipeCols(W, cols, final) {
      const B = T.board, er = toolId('eraser'), R = W.R, item = W.item;
      // the wipe starts at the end nearest where it will be when it starts: the eraser on the tray (fetched first), or here
      const ep0 = er && !held(er) && K().byId(er) ? K().byId(er).pos : null, r0 = ep0 ? {x: ep0[0], z: ep0[2]} : root();
      const uNow = dot3(sub3([r0.x, 0, r0.z], [B.O[0], 0, B.O[2]]), B.U) / B.W;
      cols = cols.slice();
      const cu = c => (c.u0 + c.u1) / 2;
      if (cols.length > 1 && Math.abs(uNow - cu(cols[cols.length - 1])) < Math.abs(uNow - cu(cols[0]))) cols.reverse();
      const dir = cols.length > 1 ? Math.sign(cu(cols[cols.length - 1]) - cu(cols[0])) : (uNow > cu(cols[0]) ? -1 : 1);   // strokes run the way it moves along the board
      const ch = toolId('chalk'), chHeld = ch ? held(ch) : null;
      // r2-erase: the duster in the paw that does not write (cfg.eraserPaw 'other'): the chalk stays in its paw, no
      // trip to the tray; 'same' = the old way (the chalk goes back first)
      const s = cfg.eraserPaw === 'other' ? eraserSide() : (cfg.eraseOtherPaw && chHeld ? (chHeld === 'l' ? 'r' : 'l') : cfg.side);
      const goals = [];
      if (ch && chHeld && s === chHeld) goals.push(putBackGoal(ch));
      const fetch = er && !held(er);
      let cur = {x: root().x, z: root().z, yaw: root().yaw};
      if (fetch) {
        const gs = grabStance(er, s, {writes: false});
        if (gs) { goals.push({g: 'stand_at', pose: gs, why: 'the board', speed: cfg.walkSpeed, tol: 0.02, ytol: 0.015, turnStep: cfg.turnStep}, {g: 'steps', id: er, did: ['picking up the eraser', 'picked up the eraser'], build: () => grabSteps(er, s)}); cur = gs; }
        else goals.push({g: 'holding', id: er, side: s, speed: cfg.walkSpeed});
      }
      // devloop: a column no single stance can wipe is split — its row bands apart, then in half across, then in
      // half down — until each piece has a stance
      const sd0 = (er && held(er)) || s, plan = [];
      // the felt's CENTRE must reach only the region inset by most of the pad (its edges wipe the rest)
      const fh = feltHalf(B), iu = 0.6 * fh.hu, iv = 0.8 * fh.hv;
      const ptsOf = colR => { const pts = []; for (const R0 of colR) { const Rg = [Math.min(R0[0] + iu, (R0[0] + R0[2]) / 2), Math.min(R0[1] + iv, (R0[1] + R0[3]) / 2), Math.max(R0[2] - iu, (R0[0] + R0[2]) / 2), Math.max(R0[3] - iv, (R0[1] + R0[3]) / 2)]; pts.push([Rg[0], Rg[1]], [Rg[2], Rg[1]], [Rg[0], Rg[3]], [Rg[2], Rg[3]], [(Rg[0] + Rg[2]) / 2, (Rg[1] + Rg[3]) / 2]); } return pts; };
      const stanceOf = (pts, c) => cfg.eraserPaw === 'other' ? eraseStance(B, pts, sd0, c) : stanceFor(B, pts, sd0, c, 'eraser');
      const place = (colR, depth) => {
        const st = stanceOf(ptsOf(colR), cur);
        if (st || depth >= cfg.eraseSplitMax) { const st2 = st || stanceOf(ptsOf(colR).filter((q, i) => i % 5 === 4), cur); plan.push({colR, st: st2}); if (st2) cur = st2; return; }
        if (colR.length > 1) { for (const r of colR) place([r], depth + 1); return; }
        const r = colR[0];
        if (r[3] - r[1] > 0.2 || r[2] - r[0] <= 0.08) { const m = (r[1] + r[3]) / 2; place([[r[0], r[1], r[2], m]], depth + 1); place([[r[0], m, r[2], r[3]]], depth + 1); return; }
        const m = (r[0] + r[2]) / 2; const h = [[[r[0], r[1], m, r[3]]], [[m, r[1], r[2], r[3]]]]; if (dir < 0) h.reverse(); place(h[0], depth + 1); place(h[1], depth + 1);
      };
      for (const c of cols) place(c.regs, 0);
      let est = 0;
      plan.forEach(({colR, st}, ci) => {
        log({teach: 'erase_col', col: ci, regions: colR.map(r => r.map(x => +x.toFixed(3))), stance: st ? [+st.x.toFixed(3), +st.z.toFixed(3), +st.yaw.toFixed(3), +st.e.toFixed(4)] : null});
        if (typeof process !== 'undefined' && process.env && process.env.TEACH_DEBUG) console.error('erase col', ci, JSON.stringify(colR.map(r => r.map(x => +x.toFixed(3)))), st ? [st.x.toFixed(3), st.z.toFixed(3), st.yaw.toFixed(3), st.e.toFixed(4), st.n] : null);
        if (st) goals.push({g: 'stand_at', pose: {x: st.x, z: st.z, yaw: st.yaw}, why: 'the board', speed: cfg.eraseStepSpeed, tol: 0.02, ytol: 0.015, Ts: 0.25, blendYaw: true, turnStep: cfg.turnStep, did: ['stepping along the board', 'stepped along the board']});
        const more = ci < plan.length - 1, last = ci === plan.length - 1;
        est += 0.9 + strokePath(B, sweepPath(B, colR, dir)).L / cfg.eraseSpeed * 1.15;
        goals.push({g: 'steps', id: er, did: ['erasing part of the board', 'erased part of the board'], build: () => { const sd = held(er) || s; return eraseSteps(B, colR, sd, {more, bend: st ? st.bend : undefined, dir, onLast: last ? () => { W.lastAt = sec(); } : null}); }});
      });
      for (const c of cols) c.done = true;
      const regions = [].concat(...cols.map(c => c.regs));
      final = final || W.cols.every(c => c.done);
      T.phase = 'erasing'; setWhere(); lookUserOff();
      item.t0 = item.t0 || sec(); W.estEnd = sec() + est + (fetch ? 2.5 : 0); W.lastAt = null; T.item = item; T.wiping = W;
      job('erasing', goals.filter(Boolean), 'erase', ok => {
        if (T.req !== R) return;
        T.item = null; T.wiping = null;
        if (T.wipe !== W) { R.next++; T.sinceTurn++; }
        if (final && T.wipe === W) T.wipe = null;
        send({type: 'board_erased', id: R.id, item: item.id, regions, ok, partial: !final || undefined, t: Number.isFinite(T.playT) ? +(T.playT + cfg.inkLagS + 0.3).toFixed(3) : null});
        if (item.all && final) T.inkCells = new Set();   // r2-boardvis (the felt stamps clear their cells as they go)
        for (const [iid, bb] of Object.entries(T.drawn)) { const cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2; if (regions.some(r => r[0] <= cx && cx <= r[2] && r[1] <= cy && cy <= r[3])) delete T.drawn[iid]; }
        for (const [kk, bb] of Object.entries(T.inked)) if ((item.all && final) || regions.some(r => r[0] <= bb[0] + 0.01 && bb[2] - 0.01 <= r[2] && r[1] <= bb[1] + 0.01 && bb[3] - 0.01 <= r[3])) delete T.inked[kk];
        // its own job (a re-plan never repeats the wipes); an eraser in the other paw stays there while the lesson goes on
        // (r2-erase: and after an erase on its own too — it goes back on the tray with the walk home)
        const keepIt = cfg.eraseKeep && cfg.eraserPaw === 'other' && held(er) && held(er) !== cfg.side;
        if (er && held(er) && !keepIt && (held(er) === cfg.side || !held(toolId('chalk')))) job('putting the eraser back', [putBackGoal(er)], 'erase');
        T.phase = 'ready'; setWhere();
      });
    }
    function finish(R) {
      // all items performed: turn to the user and hand the turn back (the brain's last line may already have been said)
      T.req = null; T.phase = 'idle'; T.idleAt = sec(); setWhere();
      const box = boxOf(R.wrote || []);
      const fin = () => { faceUser(true, box); if (!R.finPrompted && o.prompt) { try { o.prompt(finishNote(R), {respond: true, maxWaitMs: 4000}); } catch (e) {} } };
      // r2-boardvis: the free paw takes the duster from the tray when the lesson is done (not between its lines: bent
      // to the tray it hid the board for the whole line), then it steps clear, turns to the user and wraps up; a later wipe or
      // page turn needs no trip to the tray
      if (cfg.grabLater === 'end' && R.kind !== 'erase' && (R.wrote || []).length && grabDusterNear(() => { if (!T.req) fin(); }, cfg.grabEndH)) return;
      fin();
    }
    function grabDusterNear(onEnd, nearH) {
      const er = toolId('eraser'), oth = cfg.side === 'l' ? 'r' : 'l', k = K();
      if (cfg.eraserPaw !== 'other' || !er || held(er) || !k.byId(er) || k.byId(er).where !== 'home' || (k.heldBy && k.heldBy(oth)) || T.userProps.has(er)) return false;
      if ((k.E.jobs || []).some(j => j.kind === 'object' && j.src !== 'teach' && j.src !== 'auto') || (T.userObjAt && sec() - T.userObjAt < 30)) return false;   // the user's object request waits: the paws are theirs
      const r = root(), gs = grabStance(er, oth, {writes: false, from: {x: r.x, z: r.z, yaw: r.yaw}});
      if (!gs || Math.hypot(gs.x - r.x, gs.z - r.z) > (nearH || cfg.grabNearH)) return false;
      lookUserOff();
      job('picking up the duster', [{g: 'stand_at', pose: gs, why: 'the board', speed: cfg.walkSpeed, tol: 0.02, ytol: 0.015, turnStep: cfg.turnStep, Ts: 0.25, blendYaw: true},
        {g: 'steps', id: er, did: ['picking up the eraser', 'picked up the eraser'], build: () => grabSteps(er, oth)}], 'turn', onEnd);
      log({teach: 'duster', how: 'between lines', d: +Math.hypot(gs.x - r.x, gs.z - r.z).toFixed(3)});
      return true;
    }
    function finishNote(R) {
      return R.err ? `The board could not be worked out (${R.err}). Tell the user briefly and offer to try another way.` :
        R.kind === 'erase' ? 'You finished erasing. Carry on with the user naturally.' :
        R.finPrompted !== undefined && T.wipe ? 'You finished writing it on the board (you are wiping the old work off the rest of it). In a few words, check they follow or invite their question. (What you wrote stays.)' :
        'You finished writing it on the board and turned to the user. In a few words, check they follow or invite their question. (The board stays as it is.)';
    }
    function faceUser(full, box) {
      const B = T.board; if (!B) return;
      const k = K(), r = root();
      // between items it steps clear of what it just wrote when its body hides it from the user (a teacher steps back from
      // the board to talk about it); a real turn to the user also turns the feet halfway, over the shoulder only a quarter
      // r2-boardvis: what its body hides is ALL the ink on the board (sampled), by its real silhouette in the pose it
      // talks in; it steps to the nearest spot that clears it (was: the last item's box by a chest-height cylinder: 14% of
      // the board hidden while it talked, measured)
      const tY = y => full ? y + wrap(0 - y) * cfg.talkTurn : y + wrap(0 - y) * cfg.talkTurn * 0.5;
      let pts = box ? inkUV() : [];
      if (pts.length > 400) { const st = pts.length / 400; pts = pts.filter((q, i) => Math.floor(i / st) !== Math.floor((i - 1) / st)); }
      const hides = pts.length && silHidden(B, {x: r.x, z: r.z, yaw: tY(r.yaw)}, pts) > cfg.sayHideMax;
      let cs = hides ? clearStance(B, pts, tY(writeYaw(B))) : null;
      const hidNow = pts.length ? silHidden(B, {x: r.x, z: r.z, yaw: tY(r.yaw)}, pts) : 0;
      if (cs && cs.hid >= hidNow - 0.02) cs = null;   // nowhere better: stay
      log({teach: 'face', full: !!full, ink: pts.length, hidNow: +hidNow.toFixed(3), to: cs ? [+cs.x.toFixed(2), +cs.z.toFixed(2), +cs.hid.toFixed(3)] : null});
      const base = cs ? {x: cs.x, z: cs.z, yaw: cs.yaw} : {x: r.x, z: r.z, yaw: r.yaw};
      const pose = {x: base.x, z: base.z, yaw: tY(base.yaw)};
      lookUserOn();
      // the chalk paw comes back from the board to its carry pose, then the feet turn (the waist and head do the rest)
      // (r2-boardvis: stepping clear, the paw just lets go and comes down as it walks — the retract first kept the board
      // hidden for the first ~1.5 s of the line)
      const goals = [{g: 'steps', did: ['', ''], build: () => { const out = []; for (const sd of ['l', 'r']) if (k.E.paw[sd].keep || k.E.paw[sd].ctrlT > 0) { k.E.paw[sd].keep = false; if (!cs) out.push(k.retractStep(sd, true)); else k.E.paw[sd].ctrlT = 0; } return out.length ? [k.par(out)] : []; }}];
      if ((full || cs) && (Math.abs(wrap(pose.yaw - r.yaw)) > 0.02 || Math.hypot(pose.x - r.x, pose.z - r.z) > 0.03)) goals.push({g: 'stand_at', pose, why: 'the user', speed: cs ? cfg.clearSpeed : 1.2, tol: 0.02, ytol: 0.015, did: cs ? ['stepping back to show the user', 'stepped back to show the user'] : ['turning to the user', 'turned to the user'], blendYaw: true});
      if (cs) goals.push({g: 'steps', did: ['', ''], build: () => [k.dwellStep('showing the board', cfg.showS, () => {})]});
      T.turnFull = !!full;
      job('turning to the user', goals, 'turn', () => { if (T.turnFull && T.look === 'user') k.E.turnBack = true; });
    }

    function frame(dt, playT) {
      try { director(); } finally { flush(playT); }
    }
    function director() {
      while (deferred.length) { try { deferred.shift()(); } catch (e) {} }
      const k = K(); if (!k) return;
      if (T.look === 'user') lookUser();
      const R = T.req, t = sec();
      if (!R) {
        // integ: idle at a station and an object job of somebody else's takes the body elsewhere (the user asked for the
        // hat after the TV): the station camera is let go (it kept the TV framing while the muse walked out of shot)
        if (T.at && T.at !== 'home' && !String(T.at).startsWith('going') && !T.jobs.size && (k.E.jobs || []).some(j => j.kind === 'object' && j.src !== 'teach' && j.src !== 'auto')) {
          T.at = null; T.phase = 'idle'; setWhere(); view(null); log({teach: 'left station', why: 'another job'});
        }
        if (T.at && T.at !== 'home' && !String(T.at).startsWith('going') && !T.jobs.size && t - T.idleAt > (T.idleHome || cfg.idleHomeS)) { T.idleHome = null; abort('nothing more to show (idle)'); }
        return;
      }
      // the brain went quiet
      if (!R.done && !R.items.length && t - R.t0 > cfg.brainWaitS) { R.err = 'the board brain did not answer'; R.done = true; }
      // the first line can be said while it walks over (and later lines while the previous item finishes)
      const nx = nextItem(R);
      if (nx && nx.say_before && !nx.said && (T.phase === 'going' ? R.next === 0 : cfg.prefire && T.item && T.item.est && t - T.item.t0 > cfg.prefireFrac * T.item.est && T.sinceTurn + 1 < T.turnK)) {
        T.prefired = startSay(nx, T.phase === 'going' ? 'walking' : 'over the shoulder');
      }
      // r2-erase: the next item's line while the wipe finishes (a teacher talks as she wipes): it is ready to write
      // the moment the duster comes off the board
      if (T.wiping && T.wiping.R === R && cfg.wipeSayLead > 0 && t > T.wiping.estEnd - cfg.wipeSayLead) {
        const nc = nextContent(R, R.next + (T.wipe === T.wiping ? 0 : 1));
        if (nc && nc.say_before && !nc.said && !(T.say && T.say.doneAt === null)) { T.say = null; startSay(nc, 'while wiping'); }
      }
      if (T.goJob && jobAlive(T.goJob)) return;                       // still walking / picking up the chalk
      // devloop: an item whose job is gone (cut short by somebody else without an end note) never holds the lesson
      if (T.item && !T.paused && !T.item._paused && ![...T.jobs.values()].some(J => !J.done) && !(T.say && !sayReady(T.say))) {
        if (!T.itemLostAt) T.itemLostAt = t; else if (t - T.itemLostAt > 3) { log({teach: 'item_lost', item: T.item.id}); T.item = null; T.itemLostAt = null; if (T.phase === 'writing' || T.phase === 'erasing') { T.phase = 'ready'; setWhere(); } }
      } else T.itemLostAt = null;
      if (T.phase === 'going') { T.phase = 'ready'; T.sinceTurn = 0; }
      if (T.item || [...T.jobs.values()].some(J => J.why === 'write' || J.why === 'erase' || J.why === 'turn')) return;
      if (T.phase !== 'ready' && T.phase !== 'saying' && T.phase !== 'waiting') return;
      // the user is talking: a teacher stops and listens (the item in hand finishes; nothing new starts until they are done)
      // after the user spoke it lets the voice answer them first (their question beats the next line): it waits for that reply
      // to play out, or replyWaitS without one
      if (cfg.holdForUser && o.userSpeaking) {
        let u = false; try { u = !!o.userSpeaking(); } catch (e) {}
        if (u) { T.heldAt = T.heldAt || t; T.userAt = t; T.replyAud = null; T.replyQuiet = null; if (t - T.heldAt < 8) return; }
        else {
          T.heldAt = null;
          if (T.userAt && t - T.userAt < cfg.replyWaitS + 8) {
            const aud = o.audible ? !!o.audible() : false;
            if (aud) { T.replyAud = T.replyAud || t; T.replyQuiet = null; }
            else if (T.replyAud) T.replyQuiet = T.replyQuiet || t;
            const replied = T.replyAud && T.replyQuiet && t - T.replyQuiet > 0.8;
            if (!replied && (T.replyAud || t - T.userAt < cfg.replyWaitS)) return;
            T.userAt = null;
          }
        }
      }
      const it = nextItem(R);
      // r2-erase: a lazy page turn's columns: the ones the next item lands in before it is written, the rest when the
      // lesson's items are all written (the board must be what the brain was told: clean)
      const W = T.wipe && T.wipe.R === R ? T.wipe : null;
      if (W) {
        const left = W.cols.filter(c => !c.done);
        if (!left.length) T.wipe = null;
        else if (!it && R.done) {
          // the lesson is all written: it hands the turn back as it wipes the rest (a teacher tidies the board while asking)
          if (!R.finPrompted && R.wrote && R.wrote.length) { R.finPrompted = true; if (o.prompt) { try { o.prompt(finishNote(R), {respond: true, maxWaitMs: 4000}); } catch (e) {} } }
          wipeCols(W, left, true); return;
        }
        else if (it && !(it.kind === 'erase' || it.erase) && (it.strokes || []).length) {
          const bx = itemBox(it), need = bx ? left.filter(c => c.regs.some(r => boxHits(r, bx, 0.03))) : [];
          if (need.length) {
            if (it.say_before && !it.said && !(T.say && T.say.doneAt === null)) { T.say = null; startSay(it, 'while wiping'); }
            wipeCols(W, need, need.length === left.length); return;
          }
          // between items, as it says the next line, one more column of the old board goes (the old work is gone early,
          // not left beside the new lesson until its end); the nearest column first
          if (cfg.wipeBetween && (R.wrote || []).length && it.say_before && !it.said && !(T.say && T.say.doneAt === null)) {
            const r0 = root(), B = T.board, uNow = dot3(sub3([r0.x, 0, r0.z], [B.O[0], 0, B.O[2]]), B.U) / B.W;
            const c = left.slice().sort((a, b) => Math.abs((a.u0 + a.u1) / 2 - uNow) - Math.abs((b.u0 + b.u1) / 2 - uNow))[0];
            T.say = null; startSay(it, 'while wiping'); wipeCols(W, [c], left.length === 1); return;
          }
        }
      }
      if (!it) {
        if (R.done) finish(R);
        else if (T.phase !== 'waiting') { T.phase = 'waiting'; if (T.sinceTurn > 0) faceUser(false); }
        return;
      }
      if (it.kind === 'erase' || it.erase) {
        // r2-erase: it says "let me make some room" as it starts wiping (cfg.eraseTalk), not before
        if (it.say_before && !it.said) { startSay(it, 'erase'); T.phase = 'saying'; if (!cfg.eraseTalk) return; }
        if (cfg.eraseTalk || sayReady(T.say)) { if (!cfg.eraseTalk) T.say = null; eraseItem(R, it); }
        return;
      }
      if (it.say_before && !it.said) {
        // every few items a real half-turn to the user for the line; otherwise it says it over its shoulder
        // r2-erase: the duster still on the tray and the tray close by: the free paw takes it while it says this
        // line (a teacher picks up the duster between lines), so a later wipe needs no trip to the tray
        if (cfg.grabLater === true && (R.wrote || []).length && grabDusterNear()) { startSay(it, 'picking up the duster'); T.phase = 'saying'; return; }
        const full = T.sinceTurn >= T.turnK;
        if (full) { T.turnK = cfg.turnEvery[0] + Math.round(Math.random() * (cfg.turnEvery[1] - cfg.turnEvery[0])); T.sinceTurn = 0; faceUser(true, revealBox(R)); }
        else { const lb = revealBox(R); if (lb) faceUser(false, lb); else lookUserOn(); }
        startSay(it, full ? 'turned to the user' : 'over the shoulder'); T.phase = 'saying'; return;
      }
      if (T.phase === 'waiting' && T.look) lookUserOff();
      if (!sayReady(T.say)) { T.phase = 'saying'; return; }
      T.say = null;
      writeItem(R, it);
    }
    // the ink flush: one message per tick, stamped on the playback timeline (called by the page with its rig stamp)
    function flush(playT) {
      if (playT !== null && playT !== undefined) T.playT = playT;
      if (!T.ink.length) return null;
      const segs = T.ink.splice(0);
      const msg = {type: 'board_ink', t: playT !== null && playT !== undefined ? +(playT + cfg.inkLagS).toFixed(3) : null, id: T.req ? T.req.id : null, segs};
      T.lastMarkT = msg.t;
      if (T.req && !T.req.firstMarkT && segs.some(x => !x.erase)) {
        T.req.firstMarkT = msg.t; T.firstMark = {id: T.req.id, t: msg.t, s_after_call: +(sec() - T.req.t0).toFixed(2)};
        log({teach: 'first_mark', id: T.req.id, s_after_call: T.firstMark.s_after_call, t: msg.t});
        send({type: 'event', name: 'teach_first_mark', id: T.req.id, s_after_call: T.firstMark.s_after_call, t: msg.t});
        try { console.info(`teach: first chalk mark ${T.firstMark.s_after_call} s after the tool call (${T.req.id}, playback t ${msg.t})`); } catch (e) {}
      }
      T.inkSent += segs.length;
      send(msg);
      return msg;
    }

    function onHub(m) {
      if (!m || typeof m.type !== 'string') return false;
      const R = T.req;
      switch (m.type) {
        case 'board_item': {
          if (!R || m.id !== R.id || !m.item) return true;
          const it = Object.assign({}, m.item); it.id = it.id || 'i' + R.items.length;
          R.items.push(it); R.lastItemAt = sec();
          return true;
        }
        case 'board_plan_done': if (R && m.id === R.id) { R.done = true; if (m.error && !R.items.length) R.err = m.error; } return true;
        case 'board_plan_error': if (R && m.id === R.id) { R.done = true; R.err = String(m.error || 'failed'); } return true;
        case 'board_state': T.boardState = {summary: m.summary || null, items: m.items || []}; return true;
        case 'teach_abort': abort(String(m.why || 'the user moved on'), {soft: !m.hard}); return true;
      }
      return false;
    }
    const FAREWELL_RE = /\b(bye|goodbye|good night|see you|see ya|talk (?:to you )?later|gotta go|got to go|i'?m done|that'?s all)\b/i;
    function userText(text, final) {
      if (!T.req && (T.at === 'home' || !T.at)) return false;
      const s = String(text || '');
      // devloop: a goodbye while it waits at a station (no lesson running) also sends it home
      if (s && !T.req && FAREWELL_RE.test(s) && !NEG_RE.test(s) && final) { abort('the user said goodbye'); return true; }
      if (!s || NEG_RE.test(s) || !STOP_RE.test(s)) return false;
      // a short utterance that is about stopping (not a long sentence that happens to contain the word)
      if (s.split(/\s+/).length > 9 && !final) return false;
      abort('the user said "' + s.slice(0, 40) + '"');
      return true;
    }
    function interrupt(why) { if (T.req || (T.at && T.at !== 'home')) abort(why || 'interrupted'); }
    const NAMES = ['teach_board', 'board_erase', 'go_home'];
    function handles(name) { return has() && NAMES.includes(name); }
    function call(name, a) {
      a = a || {};
      let r;
      if (name === 'teach_board') r = teachBoard(a);
      else if (name === 'board_erase') r = boardErase(a);
      else if (name === 'go_home') r = goHome(a);
      else return null;
      log({teach: 'call', name, ok: r.ok, did: r.did});
      return r;
    }
    // the E2E harness's view: busy while anything of the lesson is in flight (walk, chalk, writing, erasing, a brain call,
    // walking home); the station; a coarse phase; the board (what the hub says is on it) + the last mark's stamp; the
    // first mark of the newest request (playback t + seconds after the tool call)
    function e2e() {
      const k = K(), jobs = T.jobs.size > 0, eng = !!(k && k.busy());
      const ph = {going: 'walking', returning: 'walking', writing: 'writing', erasing: 'erasing', saying: 'speaking', presenting: 'presenting', waiting: 'waiting', ready: 'ready', idle: 'idle'}[T.phase] || T.phase;
      const station = String(T.at || 'home').startsWith('going:') ? 'walking' : (T.at || 'walking');
      return {busy: !!(T.req && !T.req.done) || !!(T.req && T.req.next < T.req.items.length) || jobs || (eng && T.at !== 'home'), station, phase: ph,
        board: {items: T.boardState ? T.boardState.items || [] : [], summary: T.boardState ? T.boardState.summary : 'the board is empty', last_mark_t: T.lastMarkT ?? null},
        first_mark: T.firstMark || null, request: T.req ? {id: T.req.id, items: T.req.items.length, next: T.req.next, done: T.req.done} : null, stats: Object.assign({}, T.stats)};
    }
    function state() {
      return {phase: T.phase, at: T.at, req: T.req ? {id: T.req.id, kind: T.req.kind, items: T.req.items.length, next: T.req.next, done: T.req.done} : null,
        item: T.item ? T.item.id : null, say: T.say ? {item: T.say.item, mode: T.say.mode, audio: T.say.audioAt !== null, done: T.say.doneAt !== null} : null,
        jobs: [...T.jobs.values()].map(J => J.name), look: T.look, ink: T.inkSent, board: T.boardState ? T.boardState.summary : null};
    }
    return {setConfig, handles, call, onHub, frame, flush, userText, userSpeech, interrupt, onAction, goStation, yieldTools, state, e2e, stats: () => Object.assign({}, T.stats), cfg, _T: T, _stanceFor: (uvs, s, kind) => stanceFor(T.board, uvs, s || cfg.side, null, kind), _hiddenU: (x, z) => hiddenU(T.board, x, z), _clearStance: (pts, yaw) => clearStance(T.board, pts, yaw), _silHidden: (st, uvs, pen, s) => silHidden(T.board, st, uvs, pen, s), _inkUV: () => inkUV(),
      surface: id => (id === 'tv' ? T.tv : T.board)};
  }
  return {createTeach, surfaceOf, CFG};
});
