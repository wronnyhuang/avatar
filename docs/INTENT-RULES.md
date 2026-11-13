# Rules-based live avatar: action memory, whole-body motion, co-speech gestures, props

Design doc for the RULES-based live avatar (not the learned-reaction model). Goals:

1. the avatar knows what it has done and what it is holding (action memory);
2. free whole-body motion on command (spin, turn away, backflip, jump, dance, sit, lie down, walk a few steps …) built
   from general parameterized primitives;
3. co-speech gestures driven by its own reply text + word timings;
4. props per scene it can pick up, hold, inspect, show, hand between paws, toss/catch and put back on voice command,
   with real paw attachment;
5. no regressions.

Everything is built from general primitives, not canned per-example animations. Natural and smooth motion is the
first priority; latency must not get worse.

Related docs: `docs/CONTRACT.md` (wire + rig contract), `docs/RIG.md` (renderer rig), `docs/PRIOR.md`,
`docs/PLANNER.md`, `docs/SCENE.md`, `docs/OBJECTS.md`.

## 0. Design scope

* The new features are built on ONE character, **`muse`**: a cream / off-white, fuzzy, plush, rounded, toy-like figure
  with small black bead eyes, rosy pink blush cheeks and a small upturned smile; soft and unthreatening; it "types at a
  keyboard" while it works; gentle, cheerful, warm personality with subtle expressions and gestures that match the
  tone. It is a selectable character on the existing rig + renderer with its OWN scene and props. All the new features
  (action memory, free whole-body motion, co-speech gestures, objects) target it. The other characters are unchanged:
  with the new params at 0 they render and behave identically.
* It has LEGS (it stands by default, walks with planted feet, sits in a chair): params `foot_{l,r}_{x,y,z}` (world-axis
  offset in H of each ankle from where the rest stance, carried by the full root transform, would put it; 0 = standing
  at rest). The waist bends (body_yaw/lean/pitch = upper body over the hips for the muse). Hands: soft mittens with a
  separable thumb + short fingers (grasp, counting, typing).
* Headline scenario: the scene has headphones, a chair, a desk and a laptop (plus other props). On voice instruction,
  step by step or as one request: pick up the headphones and put them on (worn on the head, attached properly); pull a
  chair over; pull the desk over; pull the chair up to the desk and sit in it; pick up the laptop (from the ground or
  from the desk) and slide it closer on the desk; open it and "work" at it (typing and looking at the screen); then turn
  back and keep talking to the user. And the reverse (take the headphones off, stand up, push the chair back, close the
  laptop). Built from GENERAL primitives — locomotion to an object, grasp/carry/place with IK, push/pull furniture,
  sit/stand, worn-item attachment, an interact-with-object loop, persistent object positions, a small planner over
  preconditions (see `docs/PLANNER.md`) — so other orders and variants work. Action memory reflects it ("you're wearing
  headphones, sitting at the desk").
* Additional params (schema, appended): `prop{k}_hh` (attach weight to the HEAD — worn items), `prop{k}_s` (the prop's
  own articulation 0..1: laptop lid, book cover, box lid …) for k = 0..5, two more slots prop6_*, prop7_* (x y z rx ry
  rz hl hr hh s), and the feet. Rig vector = 150 columns.
* Per-character JSON lives next to the shader: `renderer/animals/muse.body.json` (body facts: H, com, base_y, extents,
  leg lengths, stance, seat/sitting geometry, max step) and `renderer/animals/muse.props.json` (places incl. desk top,
  chair seat, floor, side table; props with handles/grab points, seat height, keyboard area, screen, worn pose, reach
  map). The page receives them as `config.body` / `config.props` for the current character (absent for characters
  without them).
* Big furniture (desk, chair) is FREE (never paw-attached): the controller moves its world offset in lockstep with the
  body while the hands rest on its handles; an object resting on the desk moves with the desk (the controller keeps it
  consistent). Laptop lid = `prop_s`. Headphones worn = `prop_hh` → the renderer seats them on the head frame.
* Commands: object verbs include put_on / take_off (worn), pull / push / bring / slide (with `target`: {object|place|
  'me'|'closer'|'back'|'away'}), sit / sit_in / sit_at, stand_up, open / close, use / work_on / type_on, turn_back
  (face the user, keep doing the activity or stop it). `{action:'object', verb, object, target, side, …}`.

## Compatibility constraints

* The learned-reaction path (`react.mode`, `?react=…`, `react_*.js/py`, `learnedPass`, the `'H'`/`'R'`-rx wire) must keep
  working byte-for-byte — nothing it uses is deleted or renamed.
* Functional renderer checks can run on the CPU backend (Mesa llvmpipe, `AVATAR_RT_DEVICE=cpu`); `?voice=mock` exists
  for voice-free runs. Prefer node-level tests (`createMotion` in node, `renderer/tools/render_timeline.py`) wherever they prove the
  point; browser tests use headless Playwright Chromium on localhost pages with fake media flags
  (`--use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-video-capture=<y4m>
  --use-file-for-fake-audio-capture=<wav>`).

## 1. Rig params (`renderer/anim/schema.py`; ranges + PERIODIC list in `renderer/anim/rig.py`)

Wire values are int16 × 1e4 (|v| ≤ 3.2767), so angles are TURNS and distances BODY HEIGHTS (H, per character).

| params | range | meaning |
|---|---|---|
| root_x, root_y, root_z | x,z −3..3 / y −0.6..3 (H) | whole-character translation; x screen-right, y up (0 = seated on its floor/seat), z toward the camera |
| root_yaw, root_pitch, root_roll | turns, PERIODIC | whole-character rotation about its centre of mass C: R = Ry(yaw)·Rx(pitch)·Rz(roll); signs = head_yaw / head_pitch / head_roll (yaw + = front swings to screen-right; pitch + = top tips toward the camera, backflip = −1; roll + = top to screen-right) |
| squash | −1..1 | + squashed flat, − stretched tall; volume-preserving; about the base point (bottom contact) |
| gaze_free | 0..1 | 0 = gaze_x/y offsets from the camera direction (eyes seek the camera); 1 = offsets from the head's own forward (eyes stop seeking the camera while turned away / spinning) |
| cam_zoom | 0..1 | 0 = default framing; 1 = a wide shot (whole character + ~1.5 H around, floor visible) |
| cam_x, cam_y | H | camera aim offset (the controller pans to follow the root) |
| stand | 0..1 | OPTIONAL renderer feature: rise from seated to standing on feet/hind legs. A character that doesn't implement it ignores it and its props JSON says `"stand": false` |
| fingers_l, fingers_r | 0..5 | counting: that many digits up (order index, middle, ring, pinky, thumb; 5 = open hand); 0 = off; fractional = the next digit part-way |
| prop{k}_x/_y/_z (k = 0..5) | −3..3 (H) | free: offset from the prop's home position, world axes. Held: small offset in the grasp frame |
| prop{k}_rx/_ry/_rz | turns, PERIODIC | rotation about the prop's own centre. Free: on top of its home rotation, world axes (ry = about vertical). Held: in the grasp frame (ry = "turn it around" about the paw's up axis, rx = tip toward/away from the camera, rz = roll) |
| prop{k}_hl, prop{k}_hr | 0..1 | attach weight to the SCREEN-LEFT / SCREEN-RIGHT paw (side convention as everywhere: `_l` = screen-left = the character's own right). 0/0 = free. Both > 0 = held by both / mid hand-over (pose = weighted blend of the two grasp frames) |
| prop{k}_hh, prop{k}_s; prop6_*, prop7_*; foot_{l,r}_{x,y,z} | see §0 | worn-on-head weight, prop articulation, extra prop slots, feet |

All zero = exactly the baseline picture plus the props sitting at home. Periodic columns: the server mixer unwraps each
new value against the previous vector (shortest arc) before interpolation/smoothing and never clips; the shader uses
the fractional turn. So the controller may wrap them to [−0.5, 0.5) at any time without a visible jump.

## 2. Per-character body + props/scene files

Body facts (`renderer/animals/<character>.body.json`): `{"character","H": scene units per body height (lowest point at
rest = seat/floor contact → top of the head, ears excluded),"com":[x,y,z] (H, world),"base_y","extents","stand":false,
"notes"}`. The props JSON (`renderer/animals/<character>.props.json`) uses the same H:

```
{"character":"fox",
 "places":[{"id":"table","name":"the table","aliases":["table","desk","counter"],
            "center":[x,y,z], "size":[w,d], "top_y":<H>, "spots":[[x,y,z],...]}, {"id":"floor", ...}],
 "props":[{"slot":0,"id":"ball","name":"red ball","color":"red","colors":["red"],"aliases":["ball","toy"],
           "category":"toy","size":<radius H>,"mass":"light|medium","home":{"place":"table","pos":[x,y,z],"rot":[rx,ry,rz]},
           "grip":"power|pinch|two_hand","throwable":true,"facing":[x,y,z] (its 'front' in its own frame, for 'show it')}, ...],
 "reach":{"l":{...}, "r":{...}}}
```

World frame = H units, origin on the floor under the character's centre at rest (root all zero), x screen-right, y up,
z toward the camera. `reach.<side>` = a calibrated map from that side's hand targets (hand_*_x/y/z) to the paw's grasp
point in world H (measured from the renderer's own arm solve at rest body pose, e.g. a regular grid + trilinear
interpolation), plus everything the page needs to invert it (the page solves "which hand target puts the paw at world
point P"). 4–6 props per character, different shapes/colours/sizes, appropriate to the character and its set. The
server sends the JSON to the page in the `config` message as `config.props` for the current character (wired in
`app/app.py`).

## 3. Commands (one shape everywhere: tools, fast matcher, page UI, tests) — extends `docs/CONTRACT.md`

Existing: `pose`, `gesture`, `look`, `expression`, `release`, `hold` — unchanged.

New:
* `{action:'move', name, count, dir, speed, distance, height, style, gaze, hold, duration_ms}` — whole-body moves.
  `name` ∈ spin, twirl, turn (to a side), turn_away, turn_back (face the camera again), backflip, frontflip, flip,
  jump, hop, bounce, walk, step, dance, sit, sit_down, stand, stand_up, get_up, lie_down, roll_over, cartwheel, bow,
  wiggle, shake_off, … (the motion module owns the list; unknown names → `{ok:false, reason}`).
  `count` = how many (spins, flips, jumps, steps); `dir` ∈ left|right (OWN side) | screen_left|screen_right | forward
  (toward the camera)|back (away)|cw|ccw|up|down; `speed` = slow|normal|fast or a multiplier 0.3..3; `distance` in
  steps; `height` 0..1; `style` free text for dance (e.g. 'silly', 'robot', 'disco', 'happy'); `gaze` ∈ user (default:
  eyes come back to the user) | free (don't seek the user) | forward; `hold:true` = stay in the end state (turned away,
  lying, standing) until released / told otherwise.
* `{action:'object', verb, object, side, place, amount, count, hold}` — props. `verb` ∈ pick_up, put_down, put_back
  (home), place (on/at `place`), hold_up / show (to the camera), inspect / look_at, turn (rotate it; `amount` turns,
  `dir`), flip (upside down), hand_over (to the other paw), toss (up and catch, `count`), drop, shake, hug, offer (hold
  out to the camera), … (the motion module owns the list). No throw-away throws: every toss stays catchable.
  `object` = a descriptor `{ref:'it'|'that'|'other'|'all', id, name, color, category, ordinal}` resolved on the page
  against `config.props` + the action state ('it' = held / last referred; 'the other one' = not the last one).
  `side` = own side left|right|both (default: the free paw nearest the object / the paw holding it).
* `{action:'release', parts:'all'}` does NOT put anything down: "relax" never drops what it holds; "put it down / drop
  it" are object verbs.
* Every command returns `{ok, id, did, reason}` from `motion.command()` (`did` = a short true English description of
  what it is doing, e.g. "spinning twice to its left"; `reason` when `ok:false`, e.g. "there is no red one here",
  "my paws are full"). Tool results carry this back to the voice model so it can answer truthfully.

## 4. Page modules and APIs (all plain IIFE/UMD like `motion.js`; work in the browser and under node)

* `app/motion.js`: the move engine (general primitives: root rotation/translation curves, ballistic arcs with gravity
  in H/s², gait cycles, squash/stretch, keyframed full-body clips, beat-synced dance from composable moves, camera
  follow/zoom), the object sequencer (reach via the inverse reach map → grasp → attach ease → lift;
  inspect/turn/show/hand-over/toss-catch with a physical arc/put back/place on a spot), arbitration, and:
  - `motion.setProps(propsCfg)` (on config / character switch; resets prop state to home),
  - `motion.state()` → `{t, posture:'seated'|'standing'|'lying'|'airborne'|..., facing:'camera'|'away'|'left'|..,
    looking:'user'|'left'|..., gazeFree:bool, holding:{left:<prop id|null>, right:..} (OWN sides),
    props:[{id,name,where:'home'|'table'|'left paw'|'right paw'|'both paws'|'air'|'floor'}], held:[param groups held],
    active:[{kind,name,since}]}`,
  - `opts.onAction(ev)` callback, fired for EVERY command start/end/fail, gesture (src cmd/reflex/fidget/cospeech),
    reflex (startle, wave back, nod back, look back, perk, greet, mirror smile …), hold/freeze/release, look changes,
    expression, move phases, object events (picked up / put down / handed over / caught / dropped):
    `{t, kind, name, src:'tool'|'matcher'|'ui'|'reflex'|'life'|'cospeech'|'auto', phase:'start'|'end'|'fail',
    args, did, reason}`. Life-level micro motion (blinks, breath, micro-saccades, idle sway) is NOT reported.
  - `motion.playGesture(def, opts)` for external gesture clips: `def = {name, dur, attack, release, fn(u, g) →
    {set:{param:v}, add:{param:dv}}}`, `opts = {src:'cospeech', side, at /*timeline s, may be in the future*/,
    intensity, parts}` → returns an id or null when refused. Refusal rules: a commanded move, a held pose, a held
    object or a commanded gesture on those parts wins; co-speech never moves a paw that holds something except in
    "small" ways the def marks as `holdSafe:true`.
  - `motion.armsBusy()` → `{l:bool, r:bool, body:bool, head:bool}` (screen sides) for the co-speech planner.
* `app/actionlog.js`: subscribes to `onAction` + `state()`, keeps a timestamped log (wall clock + session time), merges
  repeats, and renders `summary({sinceS, maxChars})` → a compact TRUE English summary ("12 s ago: spun around twice to
  your left (you asked). Now: seated, facing you, holding the red ball in your right paw."). Never invents; only logged
  facts. `window.__avatar.log()` / `__avatar.state()` test hooks.
* `app/cospeech.js`: from the avatar's reply text (transcript deltas) + estimated word timings on the playback timeline
  → a gesture plan (beats on stressed words, deictic self/you/there, iconic big/small/tall/wide/round/up/down, counting
  on fingers via fingers_l/r, shrugs, open palms, negation, enumeration) played through `motion.playGesture`; varied
  (no-repeat bag, rate limits), never conflicting with commands (`armsBusy`).
* `app/intent.js` / `app/intent.py` / `app/intent_rules.json`: the fast local matcher produces the command shapes for
  general phrasings (counts, directions, speeds, object descriptors, places, pronouns), and the voice tools
  (`app/rtc.py tools()`, also used by the GPT Live delegation) include `move` and `object` (and action recall), mapped
  to the same command shapes in `app/page.html`.

## 5. Test hooks

`window.__avatar` in `app/page.html`: `.command(cmd) → result`, `.state()`, `.log(sinceS)`, `.summary()`, `.lastRig()`
(name→value of the last sent rig vector), `.events(sinceS)` (onAction stream), `.cospeechPlan()` (last plan with word
times), `.props()`. Node: `createMotion` + a synthetic input stream must run headless (`tests/*.cjs` pattern). Each
feature ships a `tests/<name>_*.cjs|py` smoke test with pass/fail exit codes.

## 5b. Acceptance criteria

Action memory: zero invented actions, ≥ 95% factual answers over multi-turn probes (empty history, completed / failed /
interrupted / negated actions, order and counts, object ownership, session reset, misleading premises "you waved
earlier, right?"); a PLANNED or in-progress action is never described as completed; the reported history must match the
OBSERVED rig/object state, not just the issued commands. Motion language: ≥ 95% valid-command success over varied
phrasings incl. objects, both anatomical sides, all character families; 100% non-execution on negative/misleading
prompts. Full-body moves: complete twirl/backflip/jump/dance trajectories, stable landing, no teleport/interpenetration,
no head-only substitute, correct count and stop, return to neutral. Co-speech: gesture apex within ±250 ms of the
intended spoken emphasis in ≥ 90%, ≥ 4 distinct gesture types, no identical back-to-back loops, NO gesture on cancelled
/ unheard speech (barge-in). Objects: pickup → hold → manipulate → put down chains with both paws, absent objects,
mid-action cancellation, repeated pickup, successive objects: ≥ 95% success, 100% state consistency, no penetration,
teleport, duplicate object or phantom success. Continuity: max per-frame rig velocity/acceleration + visual review at
contacts/transitions; any visible clip/pop fails that feature. Gaze return 0.2–1.5 s after commands/objects, eased,
not cancelling an explicitly held gaze. Plus every existing gate (latency, lips, echo, 11-min stability, request rate
< 1,200/min, reflex timing, reactions, holds, 24 fps, routing incl. the learned-reaction `?react=` switches).

## 6. Measurements every candidate keeps (reject anything that worsens them)

end-of-speech → first audio (~1.5 s) · spoken command → visible motion (~0.5 s via the matcher) · camera cue → reflex
(~0.4 s) · display fps 24 (p95 frame time at 480×270 on a 2 GB entry-level GPU) · audio/lip offset · uplink kbps (the
rig vector grew from 50 to 112 int16 columns, ~+30 kbps — keep total uplink ≤ ~0.2 Mbps) · `motion.step()` cost ≤
~1.5 ms/frame on a phone (≈ ≤ 0.5 ms on a desktop CPU) · no popping (per-frame param jumps), no clipping (paw/prop
through body/table).
