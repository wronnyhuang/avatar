# Pose revert — every commanded pose goes back to rest

After "close an eye" the eye never reopened. Any commanded pose must revert to neutral after a short hold.

## What was latched
`app/motion.js` kept a commanded pose until an explicit release in six places. All are hit in normal use; the fast matcher
(`app/intent_rules.json`) marks 46 of its rules `hold: true` ("close your eyes", "look left", "raise your hand", "tilt your
head", "stick your tongue out", "cross your arms" ...).

| source | how it stayed |
|---|---|
| `pose` with `hold: true` (matcher, planner library, tools) | `S.holds[p].until = Infinity` |
| `gesture` / `expression` with `hold: true` | `G.dur = Infinity` |
| `look` with `hold: true` | `S.cmdLook.hold`, never expires |
| "hold that / freeze" (`action: 'hold'`) | every posture channel held with `until = Infinity`; the engine's `E.frozen` |
| a semantic-planner program with `keep_pose` | the keyframe job stays in phase `holding` at weight 1 |
| a keyframe job paused by "hold that" | `J.pl.st.frozen` until a release |

## The rule
A commanded pose or move is a visit, never a new resting state.

- **Reverts** (eased back to the live value; blinks, gaze, expressions and idle life own the dial again): every dial of the
  face, eyes, brows, mouth, ears, head, torso (`body_*`), shoulders, arms, hands and tail — whatever the command said.
- **Persists** (state, not a pose): where it is in the room and which way it faces (the engine's root), its posture (seated on
  the chair, standing, lying, crouched), what it holds or wears, the board / TV station and its view, a running activity
  (typing), and a behaviour program he started ("copy me until I say stop": `app/behavior.js` sits under the commanded layers
  and is not a command).

The boundary is where the value lives: the motion layer's dials revert; the move engine's root / seat / prop state
(`app/moves.js`) is never touched by this.

## How long
| | seconds |
|---|---|
| a held pose / gesture / face / look, no duration asked (`PR.HOLD`) | 4 (after its ease-in) |
| "hold that / freeze / don't move" (`PR.FREEZE`) | 8 |
| a duration he names ("for ten seconds", "for 2 minutes", "until I say stop") | what he said, at most `PR.MAX` = 60 |

`askedHoldS(text)` reads a named duration from his words (a number + seconds / minutes after "for"; an open-ended
"until ..." = the maximum; "for a second / a moment" is not a duration). The page puts it on the command as `duration_ms`
(`runCommand`), the performer on a keyframe job as `hold_s`.

## Where
One guard, `poseRevertFrame()` in `app/motion.js`, runs every frame before the gesture and hold layers and bounds each source
in place (so a new command path that writes an unbounded hold is bounded too). The existing release envelopes do the easing:
a hold eases out over >= 0.3 s (longer for a far value: the channel's natural top speed), a gesture over its own release, a
keyframe job blends back over 0.4 s. When something is let go it emits one action event
`{kind: 'release', name: 'auto', src: 'auto'}` so the action log (and through it the voice model) stops saying "still
holding it". The tool result of a held pose now reads "... and holding it for a little while (it goes back to normal on its
own)".

## Switch
`?poserevert=0` on the page URL, `createMotion({poseRevert: false})`, or `motion.setPoseRevert(false)` at run time = the
behaviour before (`tests/sem_page_offmode_test.cjs` proves the motion output byte-identical to its base with the switch
off). `motion.poseRevert()` reports the state and the three constants.

## Check
`node tests/pose_revert_test.cjs` — 31 commands on fox, muse (seated with a mug in its paw; at the board), juno and
flurry: the user's words through the real matcher, tool poses, a recorded planner stream, and adversarial scripts that never return
(a `keep_pose` program on every group, a 10-minute hold, a job frozen mid-play). Before: 29 of 31 stay latched. After: 0 of
31; median 5.0 s to rest; seat, held prop, position and view unchanged. `POSE_WT=<tree>` runs it against another worktree.
