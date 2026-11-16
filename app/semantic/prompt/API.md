# Avatar motion API (r3_semantic)

You write a SHORT program in a restricted Python subset that animates a seated/standing upper-body avatar
(head, face, eyes, torso, arms, hands, fingers; no legs, but the whole body can move: jump, spin, crouch, step).
Allowed: calls to the functions below, numbers, strings, arithmetic, `name = value`, `for i in range(n):` (n <= 20),
and the blocks `with parallel():` (statements inside start together), `with sequence():` (one after another, the
default) and `with after(seconds):` (a sequence that starts after a delay — use inside parallel() for staggering).
No imports, no defs, no if, no attributes, no other Python.

UNITS: seconds, degrees, centimetres, 0..1 amounts. LEFT/RIGHT always mean the AVATAR'S OWN anatomical left/right
(the avatar faces the viewer, so its left hand appears on the viewer's right). The viewer/camera is `"camera"`.
Statements run one after another; every call that takes `dur` lasts that long. A pose stays until something else
moves that part. Unless you end with `keep_pose()`, the avatar automatically holds the final pose ~0.35 s and then
returns to neutral, so you don't need to call return_to_neutral() at the end. Total length 2-8 s (hard cap 10 s).
Smoothing, anticipation, overshoot and eye-leads-head are added automatically.
Make repeated motions visible: taps/pats/scratches/rubs move >= 4 cm, waves/punches/flaps >= 10 cm, nods/shakes
>= 10 deg; reaching or pointing gestures use a nearly straight arm (extend >= 0.9). Only use parameters listed here.

## Named targets (strings)
Face/head (move with the head): `nose mouth chin jaw forehead top_of_head back_of_head eyes left_eye right_eye
left_ear right_ear left_cheek right_cheek left_temple right_temple`
Body (move with the torso): `neck_front chest heart left_chest right_chest belly navel waist left_hip right_hip lap
left_shoulder right_shoulder left_shoulder_top right_shoulder_top left_elbow right_elbow left_upper_arm
right_upper_arm left_forearm right_forearm left_wrist right_wrist left_palm right_palm`
Space: `camera` (the lens, 1.5 m in front at eye height; see Live targets for `user`), `up down left right forward back sky floor ceiling
upper_left upper_right lower_left lower_right` (1.5 m from the chest, body-relative), `in_front_of_chest
in_front_of_face above_head`, or a coordinate `(x_cm, y_cm, z_cm)` from the pelvis (x = avatar's left, y = up,
z = forward; shoulders are at about (+-17, 41, 9), eyes at (+-3, 60, 27)).
`offset_cm=dict(up=3, forward=5, left=2, right=.., down=.., back=..)` shifts any target (body-relative).

## Arms and hands  (hand = 'left' | 'right' | 'both')
- `reach(hand, target, part='palm', palm=None, offset_cm=None, shape=None, elbow=None, dur=0.7, finger_dir=None)`
  put a hand part on a target and keep it there (tracks the target if the body moves).
  part (ONLY these): `palm` (palm surface) | `fingertips` | `index_tip` | `thumb_tip` | `knuckles` | `back_of_hand` |
  `wrist` — the same values for touch(), pat(), rub().
  palm (which way the palm faces): `'body'` (flat onto the touched body surface; the default when the target is on
  the body) | `'camera'` | `'down'` | `'up'` | `'forward'` | `'back'` | `'in'` (toward the midline) | `'out'` | a target name.
  finger_dir: `'up' 'down' 'forward' 'back' 'in' 'out'` (default: fingers continue the forearm).
  elbow: `'down'` (default) | `'out'` | `'up'` | `'forward'` | `'back'`.
  With hand='both' and target 'eyes'/'ears'/'cheeks'/'temples'/'shoulders'/'hips' each hand goes to its own side.
- `touch(hand, target, part='index_tip', dur=0.7)`  fingertip touch (index pointing); part as for reach.
- `cover(target, hand='both', dur=0.6)`  flat palms over eyes / ears / mouth / face / any target.
- `point_at(target, hand='right', dur=0.6)`  index finger points at a target (straight arm for far targets).
- `arm(hand, direction, extend=0.95, palm=None, shape=None, dur=0.7, elbow=None, finger_dir=None)`  arm toward a body direction from the
  shoulder: `up forward out side down across up_out forward_up forward_down back overhead`; extend 0.2 (bent,
  hand near the shoulder) .. 1.0 (straight).
- `raise_hand(hand, height='high'|'head'|'shoulder'|'chest', palm='camera', shape='open', dur=0.7)`
- `move_hand(hand, left=0, up=0, forward=0, dur=0.4)`  shift the hand from its current goal (cm, body frame;
  negative = the other way). Good for rubbing, scratching, strumming, wiping, punching.
- `rotate_wrist(hand, flex=0, side=0, twist=0, dur=0.3)`  degrees on top of the arm pose.
- `hand(hand, shape, amount=1.0, dur=0.3)`  shapes: `relaxed open flat spread fist point thumbs_up thumbs_down
  peace ok cup claw pinch rock call_me`.
- `rest_arm(hand='both', dur=0.6)`  arm(s) back to the resting pose (hands in the lap).
- `pat(hand, target='top_of_head', times=3, cm=5, part='palm')` (alias `tap`): lift off a body target and come back.
- `rub(hand, target='belly', times=4, cm=6, part='palm', direction='sideways'|'vertical')` (alias `scratch`, use
  part='fingertips' to scratch): slide back and forth on a body target. hand='both' with eyes/ears/cheeks/temples/
  shoulders/hips uses each side.
- `hands_together(where='chest', fingers='up'|'forward')` palms pressed together ~20 cm in front of a body target
  (praying, rubbing hands with an oscillate on top, holding hands).
- Gestures: `wave(hand='right', cycles=3, size_cm=12, speed=1.0)` (hand stays up afterwards), `clap(times=3, speed=1)`,
  `thumbs_up(hand)`, `hands_on_hips()`, `cross_arms()`, `shrug(side='both', amount=1, hold_s=0.35)` (up and back down).

## Head, eyes
- `look_at(target, dur=0.5, eyes_only=False)`  head + eyes turn to a target (eyes lead); eyes_only keeps the head still.
- `rotate(part, yaw=0, pitch=0, roll=0, dur=0.5)`  ABSOLUTE pose relative to neutral, degrees.
  part `'head'`: yaw + = turn to the avatar's LEFT, pitch + = chin UP, roll + = tilt toward the LEFT shoulder.
  part `'torso'`: yaw = twist, pitch + = lean BACK, roll = side lean.  part `'body'`: the whole body (root).
  `rotate('head')` with all zeros = head back to neutral.
- `nod(times=2, deg=12, speed=1)`, `shake_head(times=2, deg=18, speed=1)`, `tilt_head(side, deg=15, dur=0.5)`,
  `head_turn(side, deg=30, dur=0.5)`, `gaze(yaw=0, pitch=0, dur=0.2)` (eyes only, degrees, + = left / up),
  `roll_eyes(dur=0.9)`.

## Torso and whole body
- `lean(direction='forward'|'back'|'left'|'right', deg=15, dur=0.6)`, `twist(direction='left'|'right', deg=25, dur=0.6)`
- `shoulders(side='both', amount=1.0, dur=0.35)`  hold shoulders raised (0 = normal, negative = dropped).
- `bow(deg=30, hold_s=0.4, dur=0.7)`  (bows and comes back up)
- `jump(height_cm=25, times=1)`, `flip(direction='back', height_cm=40)`, `spin(deg=360, direction='left', dur=None)`,
  `turn(direction='left', deg=45, dur=0.7)` (stays turned; 180 = face away), `crouch(depth_cm=25, dur=0.6)`,
  `stand(dur=0.6)`, `step(direction='left'|'right'|'forward'|'back', cm=25, dur=0.6)`,
  `sway(cycles=2, deg=8, speed=1)`, `bounce(times=4, cm=4, speed=1)`
- `oscillate(part, param, amplitude, cycles=2, dur=None, phase_deg=0)`  additive sine wiggle, centred on the current
  pose: part `head` (yaw/pitch/roll deg) | `torso` (lean_fwd/lean_side/twist deg) | `root` (x/y/z cm, yaw/pitch/roll
  deg) | `gaze` (yaw/pitch deg) | `left_hand`/`right_hand` (x/y/z cm: x = avatar's left, y = up, z = forward) |
  `left_shoulder`/`right_shoulder` (amount). Runs for `dur` (default 0.5 s per cycle); put it in parallel() with a
  hold or other motion if needed.

## Face (52 ARKit blendshapes, 0..1)
- `expression(name, intensity=1.0, dur=0.4)`: `neutral smile big_smile laugh frown sad angry surprise shock fear
  disgust confused thinking pout kiss duck_face silly goofy smirk sneer wince sleepy yawn scream grimace skeptical
  determined embarrassed` (a new expression replaces the previous one).
- `face(dur=0.3, **shapes)` set exact blendshapes by ARKit name, e.g. `face(jawOpen=0.6, browInnerUp=0.8)`. Names:
  browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft
  cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight jawForward
  jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel
  mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker mouthRight
  mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft
  mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight (eyeLook* are driven by gaze).
  "Left" = the avatar's own left.
- Shortcuts: `smile(intensity=0.8)`, `wink(side='left', hold_s=0.25)`, `blink(times=1)`, `close_eyes(side='both',
  amount=1, dur=0.25)`, `open_eyes(side='both')`, `raise_brow(side='both', amount=1)`, `furrow(amount=0.8)`,
  `puff_cheeks(amount=1)`, `open_mouth(amount=0.7)`, `close_mouth()`, `stick_out_tongue()`, `squint(amount=0.8)`,
  `kiss()`, `sneer(side='left')`, `neutral_face()`

## Composition
- `hold(seconds)` (alias `wait`) — keep everything as it is.
- `with parallel():` / `with sequence():` / `with after(seconds):` blocks; `for i in range(n):` loops.
- `release(part='arms', dur=0.6)` part: `head eyes torso body arms left_arm right_arm hands face fingers`.
- `return_to_neutral(dur=0.8)` everything back to neutral (only needed mid-program).
- `keep_pose()` — end in the current pose (only if the command asks to stay/keep/freeze).
