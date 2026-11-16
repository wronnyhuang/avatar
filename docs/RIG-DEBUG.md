# Rig debug panel

A live view of the avatar's rig: every control's value, which layer drives it, what the avatar perceives and what
the planner did with the latest command. Open it with the **Rig** button next to *Stats* or with `?rig=1`. It is
closed by default; while closed it costs one null check per frame and sends nothing.

## What it shows

- **Planner row and section** (top): the planner switch (*current* | *semantic API*, applies to the next command) and
  the latest commanded action: source (library / cache / planner), model, the semantic program as it streams in,
  timings (heard -> plan sent, first statement, first pose, first motion on the rig, done, motion length), warnings,
  errors, resolved live targets. In current mode the same timing rows and the keyframe planner's summary.
- **All rig dials** (`motion.js` RIG_PARAMS) as meters grouped Head / Eyes & gaze / Brows & face / Mouth & lips /
  Torso / Arms & hands / Ears & tail: value, a range-normalised bar, the owner tag, a highlight while it moves. A
  group chip names who drives the group. `_l` / `_r` are screen-left / screen-right (the character's own right / left).
  Tap a dial for a 5 s sparkline. *More* opens root / camera / fingers, feet and props.
- **Lip dials**: the renderer's final lip values for the frame on screen (see "rigdbg" below). A renderer without the
  handler shows "set by the renderer (this renderer build doesn't report them)".
- **Who is driving**: active gestures, holds, planner job, whole-body move, commanded look, gaze target, feeling,
  listener / program layer, learned mode.
- **Perception**: face / hands / pose tracking rates and presence, the user's position, attending, mic level and
  voice activity, avatar speaking, renderer fps.
- **Event log** (newest first): perception events, motion action events, commands, voice tool calls, planner moves,
  gaze-target and feeling changes. A pause button freezes the panel (the avatar keeps going).

Phone / narrow screens: a bottom sheet (drag to resize, collapse to its header). Wide screens: a right drawer.

## Owners: the motion probe

`motion.js` exposes `setProbe(on)`, `probe()` and `meta()`. While the probe is on, `step()` records per-frame facts
into `S.dbg` only (no random numbers drawn, no other state written: outputs are identical with the probe on or off,
`tests/rigprobe_test.cjs`). `probe()` composes them into an influence 0..1 per layer per parameter, in `step()`'s
order: the rule-based base terms (Σ terms = the base value), then sets / lerps (a layer with weight w leaves 1 - w of
the layers below), adds, the listener / learned / program layers under the commanded map, keyframe jobs. `by` = the
top layer above 0.02, else `life`. Layers: life, gaze, mirror, prior, listen, speech, fidget, think, cospeech,
learned, program, feeling, reflex, look, cmd, hold, move, planner, lips. Lip-sync parameters are always `lips`.

## "rigdbg": the renderer's final values

page -> server `{"type":"rigdbg","on":bool}` when the panel opens / closes; the server forwards it to the render
server and re-sends "on" after every renderer `ready`. While on, the render server keeps each dispatched frame's final
lip and hand columns and posts at most every 100 ms
`{"type":"rigdbg","fps","cols":[...],"rows":[[F, t, v...], ...]}` (cols on the first message after each "on", then at
most every 2 s; at most 40 rows). The server relays these to the page only while the panel is open and never records
them. Details: `renderer/avatar/PROTOCOL-NOTES.txt` note 14. An older renderer answers with an "unknown message type"
error, which the server swallows; the page gets one `{"type":"rigdbg","unsupported":true}`.

## Test hooks and tests

`window.__avatar.rigdbg()` -> `{open, paused, collapsed, ticks, lip, owners, dials, log, perc, planner}`.
Tests: `tests/rigprobe_test.cjs` (probe determinism + attribution), `tests/sem_page_browser_test.cjs` (panel + planner
row in headless Chrome), `renderer/avatar/test_rigdbg.py` (render-server side), `tests/rigdbg_relay_test.py` (the
server relay end to end, `SMOKE_RIGDBG=1 scripts/smoke.sh`).
