# The full-body rig in the real-time renderer

Every motion is the rendered 3-D character: arms, paws, digits, ears, nose, tongue, eyes, body and tail are part of
the character's own signed-distance field, marched in the same pass as the head (same fur/material, sun, soft shadow,
AO, rim, TAA and depth of field). No browser overlay, no separate composite pass (the earlier `body_rig.glsl` composite is
compiled out: `RIG_PASS 0`).

## Design

* **The five animals** (fox, bear, rabbit, hippo, monkey) are the round-3 characters
  (sitting body, tail, two arms with per-animal paws/hands and digits, the round-3
  mouth) brought into the real-time engine (`renderer/engine/rt_engine.py`), then extended per animal with the new
  contract params and bounding-volume early-outs (see per-character notes). The round-3 engine-side changes they need
  came along: `engine/face.glsl` (squint), `engine/lips.glsl` (round-3 shape lips), `engine/sets/*.glsl` (the wider
  camera's floor, the river reflection of the hippo's legs), `animals/*.poses.json` (the round-3 mouth pose tables,
  tuned with these shaders). `rt_engine` now hands the set pass `uP` (the river draws the hippo's legs).
* **The six shapes** get `engine/shape_arms.glsl`: two-bone arms that sprout from the shape (smooth-min into the
  host) with mitten hands (palm/fist, thumb, a pointing index), inside the shape's SDF, guarded by a bounding sphere per
  arm, a cheap capsule proxy in the shadow pass, their own material (colour matched to each shape: pearl, glaze,
  amber, velvet, sandstone, jelly). Per-shape constants (`SA_SHO`, `SA_REST`, `SA_LEN`, `SA_R`, `SA_HAND`, colours)
  sit in each `animals/<shape>.glsl` before `#pragma include shape_arms`.
* Why not a separate pass: the earlier composite pass traced the limbs at full output resolution with its own simple
  lighting — no TAA (aliased, "pixelated" paw), no shadows from/onto the head, a visible material seam. In the union
  SDF the arms are lit and filtered exactly like the head; the cost is controlled by guards (skip what is far).

## Params (anim/schema.py, appended in contract order; `RIG_PARAMS` = what the page sends)

`_l` = SCREEN-LEFT (the character's anatomical right), `_r` = screen right, for every side param. Verified per
character by `tools/rig_qa.py` (centroid of the pixel change of each `_l` case left of its `_r` twin): see the QA
tables.

| param | range | meaning (renderer) |
|---|---|---|
| hand_{l,r}_{x,y,z} | -1.2..1.4 (contract ≈ -1..1.2) | wrist target = rest wrist + one full arm length × (x outward / negative across the chest, y up, z toward the camera); then kept out of the body, thigh, head/muzzle, on its own side of the midline (two hands meet there), in reach; two-bone IK with fixed bone lengths and a pole that swings the elbow out as the hand rises |
| hand_{l,r}_open | 0..1 | relaxed soft curl .. spread open palm (digits open one after another) |
| hand_{l,r}_turn | -1..1 | forearm roll: +1 palm to the camera when raised (palm up on a level forearm), 0 palm toward the midline, -1 away/down |
| thumb_{l,r} | 0..1 | thumb straight out (thumbs-up = open 0 + thumb 1; the fist turns thumb-up) |
| point_{l,r} | 0..1 | index digit straight, the others curled; does not move the arm |
| shoulder_{l,r} | 0..1 | shrug: shoulder joint and mass up (arm follows) |
| ear_{l,r} | -1..1 | ear pitch: -1 back/droop .. 1 up/forward |
| ear_{l,r}_turn | -1..1 | ear yaw: -1 opening forward/in .. 1 out |
| ear_flick_{l,r} | 0..1 | quick twitch (the controller animates it) |
| nose_wiggle | -1..1 | nose tip down .. up |
| tongue_out | 0..1 | tongue out past the lips; the shader opens the lips/jaw as far as it needs |
| mouth_open | 0..1 | commanded opening: teeth_gap/lip_open/jaw = max(lipsync, mouth_open) (server and shader) |
| eye_wide | 0..1 | upper lids up, lower lids down a little, brows up |
| squint | 0..1 | lower lids up (round 3) |
| body_yaw / body_lean / body_pitch | -1..1 | torso turn / side lean / lean back..toward the camera; head and arms are carried |
| body_bounce | -1..1 | vertical OFFSET (no oscillation in the shader) |
| tail_wag | -1..1 | tail side POSITION (no oscillation in the shader) |
| tail_raise / tail_curl | 0..1 | tail lift / curl |
| legacy arm_*, wave_*, point (old), paw_open_*, clap, thumbs_up | | mapped to hand targets on the server (`anim/rig.py map_legacy`, oscillations made there on the frame time); the shader reads only the positional set |

Shapes: no ears, nose or tail (N/A); body_* turn/lift the whole shape on top of head_*.

## Key poses (anim/rig_poses.py KEY_POSES; the page's controller has its own copy)

REST, WAVE_UP (+ WAVE_UP_L), POINT_CAMERA, POINT_SIDE, CLAP, THUMBS_UP, CHIN, SHRUG, COVER_EYES, HAND_ON_HEART — the
contract's tuples. Sheets: `tools/rig_qa.py` writes `<char>_keys.png`.

## Server (renderer/avatar/render_server.py + renderer/anim/rig.py)

* `{"type":"rig","t":<playback-timeline s>,"a":[...]}` in the order of `rig_params`, which the server announces in its
  `ready` message (`anim.rig.RIG_PARAMS`: every non-lip column, legacy arm columns excluded, `point_*` kept) — OR in
  the full `anim/schema.py PARAMS` order (what the app server's `config.rig_params` lists): without `"names"` the
  server tells them apart by length (PARAMS is append-only, so any prefix is unambiguous); an optional
  `"names":[...]` sets the order explicitly. Lip columns in a full vector are ignored (lipstream owns them). Values are
  clipped to `anim.rig.RANGES`.
* Per frame the vector is interpolated at the frame's timeline time from the stamped history (else the newest), then one
  critically damped step per frame (SmoothDamp, 45 ms; 22-25 ms for blinks, ear flicks, nose wiggle).
* While rig vectors arrive (newest < 2 s old) every rig column comes from the page: the server's procedural life
  (blinks, breath, gaze, head, brows, ears) is replaced (it keeps running underneath so it can resume); fade-in 0.2 s,
  after 2 s without a vector it fades back over 0.5 s.
* Combination: lip columns from lipstream; jaw/teeth_gap/lip_open = max(lipsync, mouth_open) (press released by it);
  smile = rig smile + 0.25 × the stream's own smile.
* The old `{"type":"pose","pose":{...}}` dict path still works (life stays on, 1.5 s hold, mouth columns only while the
  voice is quiet, legacy body_bounce/tail_wag amplitudes oscillated server-side, legacy arm columns mapped).
* Stats carry `rig_active` and `rig_msgs`.

## Tools (renderer/tools/)

| tool | what |
|---|---|
| `render_timeline.py --character fox --rig rig.jsonl [--audio voice.wav] --w 480 --h 270 --backend cpu\|gpu --out out.mp4 [--params-out rows.npy]` | offline video from a rig timeline through the server's own path (LiveCore lipstream + life, RigMixer, rt engine) |
| `rig_still.py --character fox --poses all\|'WAVE_UP,ear_l=1 ...' --sheet S.png` | stills (TAA settled) |
| `rig_qa.py --character fox --out DIR` | key-pose sheet + every param at its extremes with heat maps + JSON table + side check |
| `rig_bench.py --animal fox --w 480 --h 270 --frames 600 --encode --motion rest\|raised\|move [--root TREE]` | GPU timing |
| `rig_fps_table.sh OUT.jsonl "chars" "sizes" "motions"` | the fps table |
| `rig_prof.py --animal fox` | per-pass GPU ms |
| `rig_server_test.py --character fox --out DIR` | end-to-end server check on the CPU (rig_params, rig_active, frames, life resumes) |

## Per-character notes

### Monkey
* Hand IK on the contract: target = rest wrist + 0.92 × (upper + forearm) × (x, y, 0.85 z); kept out of the torso and
  thigh (round 3), out of the skull and the muzzle in HEAD space (so a hand can come in front of the eyes or under the
  chin; round 3's screen-space face keep-out made COVER_EYES/CHIN impossible), on its own side down to 0.07 from the
  midline (two hands meet: CLAP).
* A hand up by the head stands its fingers up and turns its palm by the contract (turn +1 to the camera, 0 toward the
  midline, -1 away); low hands keep round 3's palm-up/down roll.
* point: index straight, others curled, palm in; thumb: thumb straight along the hand's thumb side, palm in (thumbs-up).
* Ears: pitch ±0.60 rad, yaw ±0.80 rad (ear_*_turn), flick 0.34 rad + 0.30 roll. nose_wiggle ±0.05. tongue_out: the
  tongue comes 0.30 forward, lies on the lower lip, the tip dips; lips/jaw open to 0.45/0.40 by themselves; only the
  front of the tongue may pass the lips. eye_wide: upper lid +0.30 rad, lower lid -0.10, brows +0.5.
* Body: whole-character rigid turn about the seat (BODY_PIV): yaw 0.35 rad, pitch 0.15, lean 0.12, bounce 0.10;
  entry points (charEval, charShadow, charSurface) map the world point into the body frame, so head/arms/coat ride.
* Shrug: shoulder joint +0.15 (and the shoulder mass +0.13).
* NEW tail (round 3 had none): out from under the seat, round the screen-right hip and up; tail_wag = its plane's yaw
  (±0.42 rad: -1 swings it back behind the hip, +1 toward the camera), tail_raise lifts it 0.75 rad, tail_curl rolls
  the tip; guarded by a bounding sphere; short coat; in the shadow proxy.
* Framing: fov 28 (was 30), target y -0.40 — head top no longer touches the frame; raised hands inside.
* 270p preset q40 (was q33 = 160×90 internal, visibly soft): 28.2 / 30.7 ms total mean / p95 moving (K620).

### Fox
* Guards (look kept, ~0.2/255): a head bound (box + jaw capsule + two ear capsules, 0.31 margin) splits the head into
  foxHead / foxHeadFar / foxHeadRest; smaller skips for jaw/teeth/tongue, nose, whiskers, ears, eyelids, the arm coat,
  the torso above the neck and the tail; paw guard 0.09. Quality: shadow 16, AO 3 steps (0.55/255).
* Hand IK on the contract (`handTarget`): arm length 1.18 (bones ~11% longer so a paw reaches the eyes); x 0.62 rising
  to 1.37 for wide reaches, y 1.12 falling to 0.82 wide, z 0.74 down to 0.40 as the hand climbs; a hand may cross the
  midline a little (soft floor -0.16), CLAP stops at contact; COVER_EYES palm on each eye, digits up and in; CHIN
  knuckles under the chin; THUMBS_UP fist, palm to the midline, thumb straight up and thicker; POINT index straight,
  paw aimed from the shoulder at the target; SHRUG palms up; turn 0 = palm to the midline.
* Ears pitch ±0.60, turn ±0.80 rad about the ear's long axis, flick +0.40 pitch +0.30 turn +0.30 roll. nose_wiggle
  ±0.036 (snout tip follows). tongue_out 0.36 forward over the lower teeth (lip_open ≥ 0.34, teeth_gap ≥ 0.30 × it).
  eye_wide upper lid +0.30. Shrug +0.15 joint, +0.14 mass. Body yaw ±0.35 / lean ±0.12 / pitch ±0.15 about the hip,
  bounce ±0.10 (head, arms, shadow carried).
* Tail (visible): wag is a position: +1 swings it 3.0 rad about its root out past the screen-right haunch, -1 brings it
  0.6 rad forward round the screen-left haunch (mid-sweep 0.5-0.8 it is behind the body); raise lifts the tip 0.40;
  curl brings the far half forward round the haunch, up to 1.1 rad.
* Framing: round 3's camera kept (whole sitting fox, raised paws inside); 270p preset q45 (q50 with raised paws by
  the face was 31.6 ms).
* Round-3 coat quirk kept on purpose: away from the head round 3's coat was 0.4× length (an ear weight bug); emulated
  with a fade past the guard, so the short-coat line on the lower chest sits a little differently.

### Bear (270p preset q50)
* Guards: head, arms, lower body, mouth area, teeth/tongue, eyelids, nose, ears each skipped when far (≤ 0.08/255);
  per-frame rotations hoisted. Quality: escape 20, shadow 16, AO 3, fine 30 (~0.35/255).
* Contract IK incl. crossing toward the midline; CLAP palm to palm at the midline; COVER_EYES onto the eyes; CHIN under
  the chin; HAND_ON_HEART palm flat to the chest; a hand at the face/chest turns its palm to the surface; turn +1 on a
  raised arm = palm to the camera, digits spread; -1 the back.
* Thumb swings out straight and rolls up; point straightens the index and lines the paw up with the arm; shrug +0.15.
* Ears forward/up +0.55, back -0.38 rad (at -1 nearly behind the head), turn ±0.80, a visible flick. tongue_out ~0.38
  forward past the lips (opens the mouth itself); nose_wiggle ±0.036; eye_wide. Body yaw/lean/pitch/bounce as the fox.
* Tail: N/A (none). Camera unchanged (raised paws fit).

### Hippo
* Contract IK; the legs cannot reach the eyes, so COVER_EYES ducks the head 0.36 rad and turns it 0.24 rad to the
  front with the shoulders up and forward; hands raised in front of the chest (CLAP, THUMBS_UP, CHIN, POINT_CAMERA,
  HAND_ON_HEART) lift the head 0.20 rad so the paws show under the muzzle. **These two head moves are automatic (the
  controller does not send them).** Thumbs-up and point use toes with fists; a raised paw shows its sole at turn +1;
  a single paw across the chest stays on its own side, CLAP crosses and meets.
* Ears pitch ±0.60, a sideways tilt, turn ±0.80; tongue, mouth, nose (nostrils), eyes as the bear; body moves carried
  into the shadow and the river reflection of the legs (still drawn live).
* Fixed: stripes on the HAND_ON_HEART paw (the neck-wrinkle pattern was painted on the legs).
* Tail: N/A (none). Camera unchanged (fov 23).

### Rabbit
* Hand IK on the contract: rest = paws together at the chest; target x outward 0.95 × arm length (inward 0.30 per unit,
  so x = -0.45 meets at the midline), y up 1.15 × arm length, z 0.75 × arm length; round 3's coat/ruff/chest/belly and
  head keep-outs stay; a raised paw (not reaching across) opens out past the cheek. COVER_EYES: a paw reaching up and
  across goes over its own side's eye (head-space anchor), palm on the face.
* Arms 0.44 + 0.44 (round 3: 0.40 + 0.40), paws × 1.22, cream socks (round 3's white paws vanished on the white chest).
* A paw up by the head stands its toes up, palm by the contract turn; point: toe 1 straight and 1.9× long, the others
  curled, palm in; thumb: a dewclaw toe straight out along the paw's thumb side, the rest curled (thumbs-up).
* Ears: pitch ±0.50 rad, yaw about the ear's long axis ±0.75 rad, flick 0.34 rad + 0.30 twist (round 3 kept its
  tall/flop pair and the bend). nose_wiggle ±0.040 (+ a little forward). tongue_out: the tip 0.44 forward and down,
  round; lips/jaw open to 0.45/0.40 by themselves. eye_wide: upper lid +0.28, lower -0.10, brows +0.5.
* Body: whole-character turn about the seat (yaw 0.35, pitch 0.15, lean 0.12 rad, bounce 0.10). Shrug: shoulders +0.13.
* Tail: N/A — the puff is behind the body, not visible from the camera (tail_* do nothing).
* Framing: fov 27 (was 30.4), target y -0.02: medium close-up, ear tips inside, raised paws inside.

### Shapes (orb, cube, drop, star, pyramid, blob)
* `engine/shape_arms.glsl` (see Design). x inward is scaled to the shape's half width so x = -0.45 meets at the
  midline; the wrist target is pushed out of the host proxy (`charShadowHost`) by the arm radius + half a mitten.
* Their face params: eye_wide (lids + brows), squint, mouth_open and tongue_out (engine/lips.glsl: the opening and jaw
  floors, the tongue forward onto the lower lip and allowed past the lips), body_yaw / pitch / lean turn the whole shape,
  body_bounce lifts it (the cube: a hop up, a squash down — it sits on the floor).
* N/A: ears (the drop maps ear_l/r to its tip, round 3), nose_wiggle, nostril, tail_*.
* Shader-side idle motion kept (the drop's tip sway, the star's rocking, the blob's jiggle, the orb's float): character
  secondary motion, not params.

## Presets at 480×270 (engine/rt_presets.json "270", avatar/profiles.json local alts "270")

fox q45 · bear q50 · rabbit q50 · hippo q50 · monkey q40 · orb q75 · cube q50 · drop taa · star q60 · pyramid taa ·
blob q40 (360p entries unchanged). New named preset `q45` in rt_engine (TAA, character at 0.45 scale). March-step
trims (quality in animals/<name>.json): fox shadow 16 / AO 3 (0.55/255 mean look change), bear escape 20 / shadow 16 /
AO 3 / fine 30 (~0.35/255), rabbit coarse 30 / fine 34 / shadow 18 / AO 3 (0.38–0.46/255).

## fps (Quadro K620, 2 GB, exclusive; tools/rig_bench.py, 600 frames, moving head + mouth, `--encode` = + libx264 of the
same frame, sequential; "move" = both hands sweeping the reach + open/turn + tail/ear/nose; "raised" = WAVE_UP (x ±0.12
at 2.5 Hz) + CHIN; head-only baseline = the renderer before the rig, its own
close-up head camera and its 360p presets — so baseline and rig differ in framing as well as in the arms)

`python tools/rig_fps_md.py FILE` prints this table from the bench's JSON lines:

| character | size | build | motion | preset | render mean / p95 ms | + encode: total mean / p95 ms | ≤30/38 |
|---|---|---|---|---|---|---|---|
| fox | 480x270 | head-only baseline | rest | q50 | 23.3 / 24.31 | 26.52 / 28.08 | yes |
| fox | 480x270 | rig | move | q45 | 23.5 / 25.43 | 25.62 / 27.6 | yes |
| fox | 480x270 | rig | raised | q45 | 25.18 / 26.62 | 27.53 / 29.19 | yes |
| fox | 480x270 | rig | rest | q45 | 22.58 / 23.4 | 25.12 / 26.18 | yes |
| fox | 640x360 | head-only baseline | rest | q50 | 35.82 / 37.39 | 40.56 / 42.54 | no |
| fox | 640x360 | rig | move | q50 | 42.64 / 45.94 | 47.97 / 51.4 | no |
| bear | 480x270 | head-only baseline | rest | q60 | 26.32 / 27.32 | 29.39 / 31.14 | yes |
| bear | 480x270 | rig | move | q50 | 25.58 / 27.7 | 28.56 / 30.86 | yes |
| bear | 480x270 | rig | raised | q50 | 26.44 / 27.99 | 29.17 / 31.25 | yes |
| bear | 480x270 | rig | rest | q50 | 23.35 / 25.02 | 25.91 / 27.67 | yes |
| bear | 640x360 | head-only baseline | rest | q60 | 40.64 / 42.2 | 44.8 / 46.42 | no |
| bear | 640x360 | rig | move | q60 | 51.53 / 55.42 | 56.06 / 60.41 | no |
| rabbit | 480x270 | head-only baseline | rest | q60 | 22.64 / 23.72 | 25.27 / 26.38 | yes |
| rabbit | 480x270 | rig | move | q50 | 25.54 / 26.67 | 28.06 / 29.33 | yes |
| rabbit | 480x270 | rig | raised | q50 | 25.75 / 26.9 | 28.6 / 30.11 | yes |
| rabbit | 480x270 | rig | rest | q50 | 25.33 / 26.38 | 27.88 / 29.18 | yes |
| rabbit | 640x360 | head-only baseline | rest | q60 | 34.72 / 36.32 | 39.02 / 40.68 | no |
| rabbit | 640x360 | rig | move | q60 | 52.17 / 54.42 | 56.53 / 59.25 | no |
| hippo | 480x270 | head-only baseline | rest | q50 | 16.72 / 17.63 | 19.33 / 20.38 | yes |
| hippo | 480x270 | rig | move | q50 | 15.76 / 16.93 | 18.17 / 19.49 | yes |
| hippo | 480x270 | rig | raised | q50 | 15.49 / 16.4 | 17.93 / 18.96 | yes |
| hippo | 480x270 | rig | rest | q50 | 15.29 / 16.04 | 17.7 / 18.9 | yes |
| hippo | 640x360 | head-only baseline | rest | q50 | 26.51 / 28.02 | 30.51 / 32.11 | no |
| hippo | 640x360 | rig | move | q50 | 24.64 / 26.64 | 28.32 / 30.54 | yes |
| monkey | 480x270 | head-only baseline | rest | q33 | 13.31 / 13.82 | 15.9 / 16.61 | yes |
| monkey | 480x270 | rig | move | q40 | 25.98 / 27.97 | 28.36 / 30.52 | yes |
| monkey | 480x270 | rig | raised | q40 | 26.85 / 27.93 | 29.34 / 30.65 | yes |
| monkey | 480x270 | rig | rest | q40 | 24.9 / 25.73 | 27.6 / 28.62 | yes |
| monkey | 640x360 | head-only baseline | rest | q33 | 19.75 / 20.41 | 23.41 / 24.38 | yes |
| monkey | 640x360 | rig | move | q33 | 31.7 / 34.44 | 36.36 / 39.9 | no |
| orb | 480x270 | head-only baseline | rest | q75 | 7.35 / 7.58 | 9.57 / 10.22 | yes |
| orb | 480x270 | rig | move | q75 | 15.47 / 16.18 | 17.93 / 18.8 | yes |
| orb | 480x270 | rig | rest | q75 | 14.99 / 15.53 | 17.12 / 17.74 | yes |
| orb | 640x360 | head-only baseline | rest | q75 | 10.96 / 11.32 | 14.07 / 14.65 | yes |
| orb | 640x360 | rig | move | q75 | 23.37 / 24.62 | 26.61 / 27.94 | yes |
| cube | 480x270 | head-only baseline | rest | q50 | 6.3 / 6.58 | 8.42 / 8.77 | yes |
| cube | 480x270 | rig | move | q50 | 8.03 / 8.7 | 10.23 / 10.94 | yes |
| cube | 480x270 | rig | rest | q50 | 7.65 / 8.29 | 10.05 / 11.05 | yes |
| cube | 640x360 | head-only baseline | rest | q50 | 9.87 / 10.52 | 14.21 / 16.69 | yes |
| cube | 640x360 | rig | move | q50 | 12.22 / 13.25 | 15.39 / 16.65 | yes |


## Engine init (shader compile) — cold vs warm

tools/rig_init_time.py, 480×270 (cold = NVIDIA disk cache disabled for that process; warm = a second start), total s:

| | fox | bear | rabbit | hippo | monkey | orb |
|---|---|---|---|---|---|---|
| cold | 17.0 | 14.1 | 8.7 | 7.6 | 25.8–26.1 | 4.6 |
| warm | 0.35 | 0.43 | 0.35 | 0.40 | 0.41–0.45 | 0.32 |

(background already baked; a first bake adds ~10–40 s per character and size). The cache works across processes. Make the first tap of every
character warm after a deploy (shader text changed) with
`python -m avatar.render_server --prewarm 480x270 --characters all --prewarm-compile [--lock <gpu.lock>]` (it now builds
the real-time engine: compiles into the NVIDIA cache and bakes the background). Do not set
`__GL_SHADER_DISK_CACHE_SIZE` ≥ 2^31 (4 GiB made warm starts of big programs go back to ~25 s; 1.5 GiB is fine). Several processes compiling different programs at once looked to lose each other's entries.

## QA

`tools/rig_qa.py` per character → `<out>/<char>_keys.png` (all key poses), `<char>_params.png` (every
case + heat map), `<char>_qa.json`; `<out>/QA_TABLE.md` (all 11 × 63 cases: changed pixels > 12/255,
side verdict). Every case moves the intended region on the intended side except the documented n/a (tails on bear,
rabbit, hippo; ears, nose, tail on the shapes) and the fox's `nostril` (round 3's flare is under 30 px at 480×270).
Walk-through videos: tools/make_rig_demo.py.

## N/A

| param | not on |
|---|---|
| tail_wag / tail_raise / tail_curl | bear, hippo (no tail), rabbit (puff behind the body), all shapes |
| ear_* (pitch, turn, flick) | orb, cube, pyramid, blob; drop: ear_l/ear_r/flick bend its one tip (round 3), no turn; star: ear_l/r move its points (round 3), no turn |
| nose_wiggle, nostril | all shapes (no nose) |
| shoulder_* | shapes: lifts the arm root only |

## Known issues

* 640×360 does not hold 30 ms with the rig for fox (48.0), bear (56.1), rabbit (56.5), monkey (36.4) — their head-only
  baselines were already 39–45 ms there; hippo, orb, cube do. 480×270 is the live size and holds for all measured.
* Margins at 480×270 are 0.7–2 ms for bear, rabbit, monkey (exclusive GPU); under other GPU load they can exceed 30.
* Fox: POINT_CAMERA reads only moderately; a hand rising chin → eyes (hand y 0.82–0.97) moves fast; mid-sweep tail_wag
  (0.5–0.8) hides the tail behind the body; nose_wiggle subtle; nostril under 30 px; REST differs from round 3 by
  1.5/255 (longer bones, coat fade).
* Bear: the chin → eye sweep dips round the muzzle near y 0.85 (smooth); thumb_* on a resting closed paw rolls it
  thumb-up on the knee.
* Hippo: COVER_EYES' left paw reaches only the outer corner of the left eye; the automatic head duck / lift (above);
  POINT_CAMERA's toe foreshortened.
* Rabbit: nose_wiggle subtle at 480×270; pointing toe small.
* Shapes: simple tube arms with mittens; the star's arms are its own gold and read weakly against its points; the
  pyramid's far arm hides behind its own 3/4 turn.
* The round-3 mouth regions and poses.json came with the round-3 shaders; anim/lipstream.py is the V2 one —
  not re-tuned together.
* Rig vectors are interpolated at the frame's playback time when the page's stamps cover it, else the newest; the
  page ↔ frame clock agreement was tested with tools/rig_server_test.py only, not with the real page.
