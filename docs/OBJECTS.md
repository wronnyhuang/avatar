# Objects: extra props and general manipulation

Read with docs/INTENT-RULES.md (props / places / verbs), docs/PLANNER.md (keyframe planner) and docs/RIG.md.
Goal: besides the headphones, the muse has a set of everyday objects it can manipulate in arbitrary ways on request.
Design rule: general mechanisms only — never per-phrase or per-object special cases.

## 1. Rig: sixteen prop slots (wire format)

* `renderer/anim/schema.py` APPENDS `prop8_* … prop15_*` (the same ten columns as prop6/prop7: x y z rx ry rz hl hr hh
  s) after the feet. PARAMS stays append-only, so every shorter vector is still a valid prefix (the server tells by
  length). RIG_PARAMS 150 -> 230 columns: +80 int16 = +160 B per 'R' rig message (~+38 kbps at 30 Hz).
* `renderer/anim/rig.py` ranges/PERIODIC loop to 16; `app/motion.js` appends slots 8-15 to its scene-param list after
  the feet, RANGE to 16, PERIODIC regex `prop\d+_r[xyz]`.
* Renderer: per-slot arrays/uniforms sized 16, ray masks 16 bits, the p_ppose texture 16 x 4.
* Slot allocation in muse.props.json: 0-7 the original props, **8-13 the objects below**, **14-15 the scene's chalk
  and eraser** (docs/SCENE.md). Prop kind ids (props_defs CATALOG / PK_*): original 1-25, objects 30-39, scene
  26-29 / 40+.
* Other characters: no props file -> unaffected (all zero = at home).

## 2. Props in slots 8-13 and the affordance block

hat (worn on the head), glasses (pinch, tiny, worn on the face), gift_box (two-handed, lid = prop10_s, container,
surface), watering_can (pour: prop11_s = water flow; tool used ON the plant), block_a + block_b (stackable; block_b
starts stacked on block_a, glasses start on the gift box: persistent relations from the first frame).

Every prop may carry `"afford"` (all optional, H units, prop-local frame = its unit frame x right, y up, z front):
```
"wear":        {"anchor": "head", "offset": [x,y,z] head frame, "rot": [rx,ry,rz] turns}
"surface":     {"top_y": h above its centre, "size": [w, d]}         things can rest / stack on it
"container":   {"inner": [hx,hy,hz], "floor_y": y, "needs_open": bool} things can go IN it
"articulation":{"handle_shut": p, "hinge": p, "open_turns": a, "verbs": [...]}   prop{k}_s 0 shut .. 1 open
"pour":        {"spout": p, "tilt_turns": a, "axis": "z", "what": "water"}       prop{k}_s = flow while pouring (if no articulation)
"use":         {"motion": "pour|rub|tap|brush|press", "targets": [ids]}           tool used ON another thing
"bouncy": restitution 0..1, "spin": bool, "balance": bool, "tiny": bool, "water_target": bool
```
`home.on` = the prop it starts resting on (a stack at home). Unknown objects added later need only this JSON + a model.

## 3. Engine (app/moves.js) — general primitives

Attachment of a prop = one of: free at a pose · resting on a place · resting ON / IN another prop (persistent relative
pose: it moves with its support) · held (paw l/r/both, renderer grasp) · worn / balanced on the head (hh + head-frame
offset) · tracked body anchors (foot, back, lap: free columns driven from the body pose) · flying (ballistic, gravity).
Verbs keep the `{action:'object', verb, object, target, place, relation, side, count, amount, ...}` shape.

Further general behaviours of the engine:
* knock over / tip over / kick: the limb swings through, flight physics, riders are shed;
* landing spots are kept clear of neighbours still settling (2-D SAT);
* too big to go IN -> rests ON it (and says so); onto an open / being-opened container that fits = IN;
* an omitted object = what it holds / fetches; "bring it to me" = carry + hold out;
* "put on" of a small non-wearable = balance it on the head;
* raise reports the height it really reached; exact name words win over fuzzy matches.

## 4. Engine hooks for the keyframe planner (app/moves.js)

Reach them as `m.engine.<fn>` (motion.js returns the engine object). Units: H (body heights), turns; world = x
screen-right, y up, z toward the camera. `side` = the internal paw `'l'` (screen-left = the character's OWN right) /
`'r'`, or the own-frame words `'right'` / `'left'` (= 'l' / 'r'), as everywhere else in the engine.

* `engine.kfHeld(side, {x,y,z,rx,ry,rz} | null)` -> `{ok}`. A per-frame extra offset (H, grasp frame) + rotation (turns,
  grasp frame) of whatever that paw holds, added on top of the engine's own held pose (it is applied after the keyframe
  layer, see `postKf`). Call it every frame you want it (it stays until replaced). `null` (or a release / attach) FOLDS
  it into the held pose (`hoff` / `hrot`): no pop when the hold ends.
* `engine.kfEvent({type:'release', side, impulse?:[vx,vy,vz]})` -> `{ok, did | reason, object}`: lets go NOW with the
  paw's measured world velocity (+ impulse, H/s world). The prop flies (ballistic, G = 12 H/s²), bounces
  (`afford.bouncy`, else by mass), settles on the highest support under it (place / another prop's top / INTO an open
  container through its opening), or is caught. Events: `thrown` | `let_go`, then `landed` | `landed_in` | `caught`.
* `engine.kfEvent({type:'catch', side, window?:s})` (default 0.5 s): arms that paw; a flying prop within 0.06 H + its
  size of the paw's grasp point — or of where it WOULD sit if that paw held it (its centre off the grasp point by its
  grip, the measure that matters: a straight-up toss coming back to where it left the paw is caught) — attaches with a
  velocity-matched give (critically damped, position + velocity continuous) (`caught`); else `missed` when the window
  ends (it lands on its own, reported truthfully).
* `engine.kfEvent({type:'grab', side})`: attaches the nearest free non-furniture, one-paw prop within 0.05 H + its size
  of that paw's grasp point (a flying one = a catch); else `{ok:false, reason}`.
* `engine.kfEvent({type:'attach', side, to})`, `to` = `'head'` (wear it if it has `afford.wear`, else balance it on the
  head — refused if a worn thing is in the way) | `'foot_l'|'foot_r'|'back'|'lap'|'shoulder_l'|'shoulder_r'` (tracked body
  anchors; lap only while seated) | `'<prop id>'` (on its top, or IN it when it is an open container) | `'<place id>'`.
  The held prop must already be near that spot (0.08 H + its size; the head 0.12 H + size) — else `{ok:false}`, never a
  teleport. The hand-off eases (attach weights and positions continuous).
* `engine.things()` -> `[{id, name, kind, where, held, worn, on, inside, pos:[x,y,z] world H, size, afford:[keys], open?}]`
  (`where` as `state().props[].where`: `home`, `floor`, `<place id>`, `on <prop id>`, `in <prop id>`, `worn on its head`,
  `balanced on its head`, `on its left foot` …, `<side> paw`, `both paws`, `air`).
* `engine.postKf(o, kfOn)` — motion.js calls it right after `kfFrame`: the engine re-reads the hand / root columns
  actually drawn (grasp points, paw velocities, `hPrev` for its next paths) and applies `kfHeld` offsets to the held
  props' columns. While the keyframe layer is on, this is the ONLY per-frame tracking (`post()` skips it): events
  fired inside `kfFrame` therefore see the paws / held things where they were last DRAWN (a release starts its flight
  exactly there), and `wvel` / paw velocities are measured from one pose per frame. `stopAll()` and a keyframe start
  never drop what is held, worn, balanced or riding on something; physics flights continue through them.

## 5. Keyframe language WITH things (planner layer)

Only for a character that has a props file (`renderer/animals/<c>.props.json`: the muse) AND whose page sends its
things with the plan request (`state.things` = `motion.things()` = `engine.things()`, plus `state.root`). Otherwise the
planner, its prompt, its parser and its cache key are exactly the plain keyframe planner's (docs/PLANNER.md).

Pose keys (own-side names; OWN right paw = engine side `'l'`):
* held-thing controls (continuous, interpolated like every control, eased back at the end, rotations to the nearest whole
  turn): `right_held_x/y/z` (H, grasp frame, clamped ±0.4), `right_held_spin` (turns about the paw's up axis -> `ry`),
  `right_held_tilt` (side axis -> `rx`), `right_held_roll` (forward axis -> `rz`); same with `left_`. Each frame
  `engine.kfHeld(side, {x,y,z,rx,ry,rz})` with the values × the job's weight (rotations wrapped to ±0.5 first); `null`
  once when no job drives that paw any more.
* paw events (fire once when the playback clock crosses the pose): `right_let_go` (value >= 0.5) -> `kfEvent({type:
  'release', side})`; `right_toss_up: h` -> `kfEvent({type:'release', side, height:h, impulse})` where the impulse makes the
  release velocity exactly `(0, sqrt(2·12·h), 0)` (it cancels the held prop's measured `wvel`); `right_catch` ->
  `kfEvent({type:'catch', side, window:0.6})`; `right_grab` -> `kfEvent({type:'grab', side})`. Same with `left_`.
* object steps: `"do": {"verb", "object", "target", "relation", "place", "side", "to", "count", "amount", "dir",
  "speed", "height", "attach", "anchor", "style"}` or the string form `"pick_up ball"` / `"put ball on block_a"` (verb
  aliases put/set -> place, take/grab -> pick_up, wear -> put_on; two-word verbs `pick up` -> `pick_up`). The object
  string becomes `{id}` when it is a prop id, `{ref:'all'}` for all/everything, `{ref:'it'}`, else `{name}`.
  At the step's time the clock stops (`stopAt`), the job's weight eases out over 0.35 s, motion.js runs the step through
  its engineCommand (never `m.command`, which would end the keyframe job) and waits for that job's `end` / `fail` /
  `interrupted` event (an immediate refusal or a no-op ends it at once; 25 s cap). Then a FRESH START: rest snapshot =
  this frame before any overlay, root base = where the engine left the body, every column restarts from rest at the
  step's time (the step's key and the hold copies after it), world feet back to automatic, weight eases in over 0.45 s.
  A failed step never stops the plan. `verb:"attach"` = `kfEvent({type:'attach', side, to})` (to: head, right_foot ->
  foot_l, left_foot -> foot_r, a prop or place id), no engine job.
* A plan with things owns its paws outright: hand / thumb / point / fingers columns compose absolutely (ramped in over
  0.3 s) instead of additively, so the engine's carry pose changing under it (a release, a catch) does not move the paw.
* Outcome: the kf `end` event carries `steps: [{t, verb, object, what, ok, reason?, did?, s}]`, `partial` (the failed
  steps in words) and `paw_events: [{t, key, ok, reason?}]`; `did` says which steps were done and which did not work;
  app/perform.js puts `partial` into the check-in note so the character tells the user honestly what did not work.
  `kfState()` has `steps`, `doing`, `paw_events`. Missing engine methods degrade: events recorded as not supported, the
  plan plays on.

Server (app/planner.py): system prompt = `<c>.txt` (frozen, sha-checked, untouched) +
`app/planner_prompts/objects_appendix.txt` with its catalogue generated from the props file (ids, names, affordances,
places) and the do verbs (`DO_VERBS`); the user message gets `Things now: ...` (positions relative to the root, rounded
to 0.05 H; furniture left to the room paragraph). Both are in the cache key and the prompt sha. The parser accepts the
object keys and `do` only for such a plan. A library hit is skipped when the request names a thing and the library
read no object step.

Page: a do_action naming a thing that the scene matcher does not fully read goes to the planner with the things
(`?objplan=0` = refuse with the object hint instead); the page plan cache key includes the things' state.

## 6. Tool surface (page + voice)

* Page tool mapping (app/page.html `actToolCommand`) passes `relation`, `amount`, `axis`, `dir`, `speed`, `style` of
  the object tool, and keeps "my back / my head / my foot …" possessive (a body part, not "put it back").
* Voice (app/rtc.py, planner mode only): do_action also takes tricks with things; `object` keeps the plain single
  steps and gains `relation`. The object tool's verb enum = the base object verbs + the extended verbs: spin, tilt,
  wave, raise, hide, drink, pour, balance, throw, juggle, use, kick, knock_over (new engine verbs are added to the
  extended list in app/rtc.py). `relation` enum: on|in|under|behind|next_to|in_front_of|left_of|right_of. The tools
  JSON is ~15.2 KB (size guard 15300 bytes).
