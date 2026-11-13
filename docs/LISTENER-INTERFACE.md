# Listener interface (`docs/PLANNER.md` §3, §5) — app/motion.js

The motion layer (`m` = `D.motion` on the page) lets a learned listener (the unified motion model, `motion-model/`) drive the avatar's face, gaze and posture
while commanded actions keep the channels they touch. Rules-mode output is unchanged when no listener is set and no
keyframe job runs (tests/rules_golden.cjs hash `fbcd8d5c…`).

## Groups
`m.LISTENER_GROUPS` (also `AvatarMotionLib.LISTENER_GROUPS`):

| group | channels |
|---|---|
| face | smile, squint, eye_wide, blink_l, blink_r, brow_l, brow_r, brow_knit, mouth_open, nostril, nose_wiggle, ear_l, ear_r, ear_l_turn, ear_r_turn, ear_flick_l, ear_flick_r |
| gaze | gaze_x, gaze_y |
| posture | head_yaw, head_pitch, head_roll, head_x, head_y, head_z, body_yaw, body_lean, body_pitch, body_bounce, shoulder_l, shoulder_r, breath |

`m.SPEECH_GROUP` = `['mouth_open', 'tongue_out']`: the rig mouth channels lip-sync owns while the avatar is voiced (the
renderer opens the jaw to max(lipsync, mouth_open), renderer/anim/rig.py `combine`). The lip columns themselves
(`AvatarMotionLib.LIPSYNC`: jaw, lip_*, tongue_tip, …) are always lipstream's; motion.js leaves them 0.

Channels are in the renderer's screen frame (`_l` = screen-left = the character's own right), same ranges as `RANGE`.

## Calls
- `m.setListener(fn | null)` → `true` when set. `fn({t, dt, state})` is called once per `m.step()` and returns
  `{<channel>: value}`; only channels of the three groups are used, a channel it leaves out keeps the rules value, a
  throw or a non-object return is ignored for that frame (`m.debug().listenerErr`).
  `state` = `{speaking, voiced (0..1, the speech weight), listening, user: {present, ax, ay, attending}, owned (= m.owned()),
  rules: {<group channel>: this frame's value without the listener}, kf: the running keyframe job's summary | null, character}`.
- `m.owned()` → `{<channel>: weight 0..1}` that commanded actions drive now: keyframe jobs (their weight; on SPEECH_GROUP
  × (1 − speech)), engine moves / object work (move-layer weight on root / feet / camera, recorded set-weights elsewhere),
  commanded gestures / expressions (their envelope), held poses (their ease), commanded looks (gaze + head yaw / pitch).
  Reflexes and the idle / speaking / listening layers never appear. Absent = 0.

## Composition (per channel, per frame)
Call order inside `m.step()` (motion.js `step`): rules layers → gestures + reflexes → holds → engine (muse) → clamps →
**keyframe jobs** (`kfFrame`) → the learned-reaction map (`learnedPass`) → **listener** (`listenerPass`) → output.

The listener reuses the learned-reaction hook: `learnedPass` already builds, per channel, the affine map the server
applies to the learned value, `final = alpha · learned + beta` (alpha = what survives under commands, beta = commanded
values + reflex offsets; blinks `alpha · max(learned, reflex blink) + beta`). Keyframe jobs enter that map as commanded
values with their weight, so with a listener:

    final = commanded · w + listener · (1 − w)        (+ reflex offsets, e.g. a startle flinch)

where `commanded` is the job's value composed on the rules / idle base (planner pose − rest + base), `w` its weight. When a
job starts while a listener (or the learned mode) is active it blends in over 0.15 s; when it ends or is interrupted `w`
eases to 0 over 0.4 s (min-jerk) — the listener's values come back. On SPEECH_GROUP channels both yield to speech: the job
uses `w · (1 − sv)`, the listener's value is pulled to the rules value (mouth closed for the visemes) by `sv`, where `sv`
eases to 1 within ~0.05 s of the avatar's voice (prosody `speaking`: voiced in the last 0.32 s) and back in ~0.15 s.

The server-side learned mode (`setLearned(true)`, `learnedParts`) gets the same keyframe entries, so a learned reaction
never overwrites a job's channels. Use one of the two: with both on, the page composes the listener and the server then
applies the learned values again on the channels its retarget covers.

## Example
```js
const m = D.motion;
m.setListener(({t, state}) => {
  if (!state.user.present) return null;                       // rules values this frame
  const nod = state.listening ? 0.06 * Math.sin(2 * Math.PI * 1.2 * t) : 0;
  return {head_pitch: nod, gaze_x: 0.3 * state.user.ax, smile: state.listening ? 0.35 : 0.2, mouth_open: 0.1};
});
m.command({action: 'keyframes', id: 'p1', description: 'jump twice', done: true, hold: false,
  poses: [{t: 0.3, root_y: -0.05, squash: 0.12, smile: 0.7}, {t: 0.6, root_y: 0.4}, {t: 0.9, root_y: 0}]});
// while it plays: m.owned().smile === 1 → the job's smile; head_pitch / gaze_x stay the listener's;
// 0.4 s after its 'end' event smile is the listener's 0.35 again. While the avatar talks, mouth_open follows speech.
m.setListener(null);                                           // back to the rules layers
```
Tests: tests/kf_test.cjs (d) and (e).
