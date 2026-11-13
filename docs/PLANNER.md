# Live keyframe planner — every commanded action through a voice-model function call (all 12 characters)

Commanded body actions are not looked up in a fixed move table: the voice model calls a function with a description of
the action, and a live keyframe planner (an LLM streaming JSON poses) writes the move on the fly. The named-move library
remains as a cache in front of it. Implementation: `app/planner.py` (server side), `app/motion.js` (player),
`app/perform.js`, `app/rtc.py`, `app/voice.js`, `app/page.html`.

## 0. Design rules
- ALL directly commanded actions (anything the user asks the avatar to do with its body) go through a voice-model
  FUNCTION CALL `perform_action(description)`; the server hands the description to the live keyframe PLANNER, which
  streams the move as JSON keyframes. The planner model is configurable (`app/planner.py` PRESETS: OpenAI Responses
  API, OpenRouter, Gemini, Meta, or the Claude CLI route of `docs/CLAUDE-DOOR.md`), each character has its own prompt in
  `app/planner_prompts/<key>.txt` (sha-checked against `MANIFEST.json`). Scene objects on characters that have them
  (the Muse: headphones/chair/desk/laptop) keep the `object` tool — also a function call — because they change engine
  scene state.
- The named-move library (the engine's moves/gestures + the intent matcher grammar) is ONLY A CACHE: a description the
  library fully understands plays the library move instantly; every planner output is cached by normalized description
  (+ character) so a repeat is instant.
- While the planner writes, the avatar SAYS something short and natural; a brief cue lands right before the motion starts;
  afterwards it checks in on how it went. Never hard-code phrases and never put example phrases in any prompt or
  instruction — describe the behaviour generally, let the voice model word it, varied each time.
- Works for EVERY character (fox bear rabbit hippo monkey orb cube drop star pyramid blob muse), in the shared voice and
  motion layer. Tested at least on muse + fox + one shape (orb or cube).
- A learned listener network (trained separately) owns face, gaze and posture. Commanded actions own the body
  channels they touch; everything blends back after (section 3, `docs/LISTENER-INTERFACE.md`).
- Latency is measured (command → first motion, command → first speech). The old fast path stays reachable by a switch:
  page `?planner=0` / config `planner: false` = the matcher-only behaviour.

## 1. Pose format (planner → server → page → motion)
A pose = `{"t": <s from motion start>, "<control>": <value>, ...}` listing only controls that change (the keyframe
stream format of the planner prompts; parser semantics = the streaming parser ported in `app/planner.py`, reference copy
`tests/planner_parse_ref.cjs`). Control names are the character's OWN-side names of that prompt (right_paw_x, left_ear,
root_y, head_yaw, ...); the motion layer maps them to rig columns (`KF_ALIAS` in `app/motion.js`). Controls a character
does not have are ignored by the motion layer.

## 2. Server ↔ page messages (existing control lanes: page `sendCtl(obj)` → `app.py` `Session.on_control`; server → page JSON → page `onJSON`)
page → server  `{type:'plan', id:'p<n>', character:'<key>', description:'<text>', state:{posture, holding, wearing, seated}}`
server → page  `{type:'kf', id, poses:[...new poses...], done:false}` (each batch sent as soon as parsed; ≤ 1 message per pose)
server → page  `{type:'kf', id, poses:[...], done:true, ok:true|false, source:'library'|'cache'|'planner', engine:[cmd,...]|null,
                summary:'...', ms_first_pose, ms_done, usd, error?}`
               `source:'library'` = the matcher grammar fully understood the description: `engine` holds the engine commands
               to run instead of keyframes (poses empty). `source:'cache'` = all poses at once, done:true in the first message.
               `usd` = the API-equivalent cost of the plan as reported by the provider (0 for cache/library).
page → server  `{type:'perform', id, description, source, marks:{call_ms, plan_first_pose_ms, ready_ms, cue_audio_ms,
                first_motion_ms, end_ms}}` (perf-clock ms; for the session record + latency report)

## 3. Motion layer API (`app/motion.js`; the same object `m` the page uses as `D.motion`)
- `m.command({action:'keyframes', id, description, poses:[...], done:false, hold:true})` → `{ok, id, did}`; starts a KF job
  (interrupts/replaces whole-body jobs like any move; 'stop' stops it).
- `m.kfFeed(id, poses, done)` appends streamed poses; `m.kfRelease(id)` lets a held job start once its buffer rule is met
  (≥ 2 poses AND ≥ 1.0 s parsed; all poses at once = ready at once); `m.kfState(id)` →
  `{phase:'buffering'|'ready'|'playing'|'ended'|'failed', poses, planned_s, played_s}`.
- Events via the existing `onAction`: `{kind:'kf', name:'perform', id, phase:'ready'|'start'|'end'|'interrupted'|'fail', did}`.
- Player semantics: monotone cubic interpolation, stall braking within 0.35 s of the last parsed pose, 0.15 s return ramp,
  RANGE clamps, floor contact, feet only on characters with legs, camera zoom from poses; runs live at the page frame
  rate on top of the character's idle base; works for characters with and without legs.
- Channel ownership: `m.owned()` → `{<rig channel>: weight 0..1}` of channels commanded actions drive right now (KF jobs,
  engine moves/object jobs, commanded gestures/poses). `m.setListener(fn|null)`: `fn({t, dt, state})` → `{<channel>: value}`
  for the listener-owned groups `m.LISTENER_GROUPS = {face:[...], gaze:[...], posture:[...]}`; per channel final =
  commanded·w + listener·(1-w) over the rules/idle base; when a job ends its weight eases to 0 (blend back). Documented in
  `docs/LISTENER-INTERFACE.md`. The learned-reaction mode (`app/react_link.py`) uses the same hook.

## 4. Voice layer (`app/rtc.py` tools + instructions, `app/voice.js`, `app/page.html`)
- Tool `perform_action {description: string}` for EVERY character: the user's request restated completely (counts,
  direction, speed, style, which paw). For the Muse the `move` tool goes away (library = cache); `object` stays.
  gesture / expression / pose / look stay only for the avatar's OWN expressiveness while it talks, never for the user's
  direct requests.
- Tool result returned at once: `{ok:true, status:'preparing'|'starting', ...}` ('starting' = library/cache hit).
- `v.prompt(note, {respond:true})` in voice.js: a silent note (never read aloud) + ask for a response, for Realtime
  (conversation.item.create + response.create) and Gemini Live (clientContent turn, turnComplete). The page calls it when
  the KF job is ready ("about to start": cue) and when it ended ("finished": check in) — skipped while the user is speaking.
- The page holds a ready job until the cue's first audio (or ≤ 700 ms), then `m.kfRelease(id)`.
- Planner mode (default on; `?planner=0` / config `planner:false` = old path): the server fast matcher and the page caption
  matcher do not execute commands; they are not deleted.

## 5. Talking while an action plays
- Speech, lip-sync and the voice turn are NEVER blocked, paused or delayed by a running motion (no response holds, no audio
  gating, no lip-sync suppression while a KF job / engine move plays). The only hold is the motion waiting for the cue.
- The voice model knows when an action starts and ends: silent notes at ready (about to start, with the planned length so
  far), start (no response requested) and end (check-in requested unless the user is speaking). Its tool result / notes
  say so.
- Default: it mostly stays quiet while the motion plays; but when the user asks it to talk during an action (count the
  reps, narrate, sing along) it does — its cue response at 'ready' may run through the motion. This is described generally
  in the instructions, no scripted phrases.
- Layering: lip-sync (visemes / mouth from speech) owns the mouth channels while the avatar is voiced; a KF pose's mouth
  values (mouth_open, tongue_out, ...) yield to speech while voiced and apply when silent; the face/brow/eye values of a KF
  pose and the body channels layer as section 3. The mouth set is defined in motion.js next to LISTENER_GROUPS
  (SPEECH_GROUP).
- Tests: one talk-while-moving case per character tested (muse, fox, one shape at least): the motion's body channels keep
  their trajectory while speech plays, and the mouth follows speech.
