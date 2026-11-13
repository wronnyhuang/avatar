# Avatar architecture — the shared contract

The interfaces every part of the avatar agrees on: where things run, the rig parameter set, the wire format between
the page and the server, the browser modules and the measurements every change is judged by.

## Where things run

* `app/` — the app server (aiohttp, `app.py`), the browser page (`page.html`, `interactive.js`,
  `vision/` = MediaPipe JS/WASM/models served locally), the voice-vendor bridge (`vendor.py`), render providers.
* `renderer/` — the GPU raymarch renderer: `engine/rt_engine.py` (+ `scene_main.glsl`, `rt_passes.glsl`,
  `body_rig.glsl`), per-character `animals/<name>.glsl|json`, the param schema `anim/schema.py`,
  the causal mouth network `anim/lipstream.py`, the render server `avatar/render_server.py`.
* An isolated instance of the whole thing (server + local renderer) for development:
  `tools/devinst.sh <instname> <port> [<checkout>]` → prints a localhost URL with its key.
* GPU: the renderer targets modest GPUs (developed on a 2 GB entry-level card). For functional (non-timing)
  renderer tests use the CPU backend (`backend cpu` → Mesa llvmpipe, `LP_NUM_THREADS` ≤ 16).
* Browser for tests: Playwright Chromium or Chrome, headless, with `--use-fake-ui-for-media-stream
  --use-fake-device-for-media-stream --use-file-for-fake-video-capture=<clip.y4m>
  --use-file-for-fake-audio-capture=<clip.wav>`.
* Vendor keys come from the environment by name; never print them.

## Target architecture

```
phone browser ──WebRTC (mic up, voice down, data channel events/tools)──► OpenAI gpt-realtime-2.1  (switch: gpt-live-1)
   │  MediaPipe face(52 blendshapes, head pose, gaze) + hands + pose(arms) on the phone, ~15–30 Hz
   │  motion controller (JS, every frame): perception + user voice + avatar voice + commands + learned prior
   │        → the FULL rig vector at 30 Hz
   │  avatar voice from the WebRTC track: tapped (AudioWorklet) → 24 kHz PCM up to the server (lips);
   │        played locally through a DelayNode = measured video round trip (sync), kept as small as possible
   ▼
server app.py  (HTTP exchange — a FRAME pipe, not a stream; long-poll down)
   │  SDP broker /api/rtc (key never reaches the browser), spend ledger, fast local intent matcher
   ▼
render_server (local GPU): mouth from lipstream(PCM) + rig vector from the page → H.264 frames
   stamped with the playback-timeline time they depict → back down to the page, displayed when the delayed
   audio reaches that time.
```

No browser-drawn overlay over the avatar, ever. The only thing drawn by the browser is the user's own camera
preview in a small corner (with an optional, hideable face-mesh/landmark view). Every motion the user sees
must be the rendered 3-D character.

## Rig parameters (renderer/anim/schema.py — APPEND ONLY)

Existing (keep): jaw round spread press labdent tongue smile head_yaw head_pitch head_roll head_x head_y head_z
breath blink_l blink_r gaze_x gaze_y brow_l brow_r brow_knit ear_l ear_r ear_flick_l ear_flick_r cheek nostril
voice lip_* teeth_gap tongue_tip tongue_dental tongue_back arm_l arm_r wave_l wave_r point_l point_r paw_open_l
paw_open_r clap thumbs_up body_lean body_bounce tail_wag tail_raise body_yaw.

SIDE CONVENTION (all `_l`/`_r` params: blink, brow, ear, hand, thumb, shoulder, point, ...): `_l` = the
SCREEN-LEFT side as the viewer sees it (the character's anatomical RIGHT); this matches the hand columns
("index 0 is screen left"). Every `_l` param follows it. The command layer maps "your left paw" (the character's own left) → `_r`.

New, appended in this order (ranges are the contract; 0 = neutral rest unless noted):

| name | range | meaning |
|---|---|---|
| hand_l_x, hand_l_y, hand_l_z | ≈ -1..1.2 | SCREEN-LEFT hand IK target (see side convention below), offset from the rest pose in units of one full arm length: x = outward from the body midline (negative = across the chest), y = up, z = toward the camera |
| hand_r_x, hand_r_y, hand_r_z | same | RIGHT hand, mirrored convention (x always = outward) |
| hand_l_open, hand_r_open | 0..1 | 0 relaxed soft curl, 1 spread open palm |
| hand_l_turn, hand_r_turn | -1..1 | forearm roll; +1 = palm facing the camera when the arm is raised |
| thumb_l, thumb_r | 0..1 | thumb extended (thumbs-up = open 0 + thumb 1) |
| shoulder_l, shoulder_r | 0..1 | shrug raise |
| ear_l_turn, ear_r_turn | -1..1 | ear yaw: -1 rotated inward/forward-facing, +1 outward (with ear_l/ear_r = pitch -1 back/droop .. 1 up/forward) |
| nose_wiggle | -1..1 | nose tip up/down twitch (rabbit-style) |
| tongue_out | 0..1 | tongue out past the lips |
| mouth_open | 0..1 | COMMANDED jaw opening; renderer uses max(lipsync jaw, mouth_open) |
| eye_wide | 0..1 | upper lids raised (surprise) |
| squint | 0..1 | lower lids up (smile / suspicion) |
| body_pitch | -1..1 | -1 lean back, +1 lean toward the camera |
| tail_curl | 0..1 | tail curls up/around |

Semantics of the newer params: `tail_wag` becomes a POSITION -1..1 (no oscillation inside the
shader — the controller oscillates it); `body_bounce` becomes a vertical offset -1..1 (no shader sin).
`point_l/point_r` = index finger extended only (does not raise the arm; the controller places the hand).
`arm_l, arm_r, wave_l, wave_r, paw_open_l/r, clap, thumbs_up` are LEGACY: the renderer may map them to the hand
targets for back-compat, but the new controller drives only the positional hand_* set.

Key poses the renderer must make look right (no self-intersection, fully in frame) and the controller uses
(hand x,y,z ; open ; turn):
REST (0,0,0;0.2;0) · WAVE_UP (0.20,1.00,0.20;1;1) with wave = x ±0.12 at ~2.5 Hz plus turn ±0.3 ·
POINT_CAMERA (-0.10,0.55,0.85; point 1) · POINT_SIDE (0.60,0.60,0.30; point 1) · CLAP both (-0.45,0.45,0.45) ·
THUMBS_UP (0.10,0.45,0.45;0;thumb 1) · CHIN/thinking (-0.35,0.75,0.35;0.3) · SHRUG shoulders 1 + hands
(0.35,0.30,0.20;1;1) · COVER_EYES both (-0.35,1.00,0.45;1) · HAND_ON_HEART (-0.45,0.35,0.25;1).

Combination at the renderer: mouth/lip params (jaw round spread press labdent tongue cheek voice lip_* teeth_gap
tongue_tip tongue_dental tongue_back) come from lipstream; everything else comes from the page's latest rig
vector; jaw = max(lipsync jaw, mouth_open); smile = rig smile (+ lipstream's small speech smile). The server's
own procedural idle (life.py) is OFF whenever a rig vector is arriving (the page's controller owns blinks,
breathing, gaze, head). The server applies a light per-param smoothing (critically damped, ~40–60 ms) to hide
30 Hz packet jitter — never a snap.

Camera framing: medium close-up — head, shoulders, chest and both raised paws fully visible at 16:9 and on a
portrait phone (the page letterboxes 16:9).

## Wire (all little-endian; extends app/wire.py)

Up (browser → server), bodies = concatenated `[uint32 len][msg]`:
* `'P'` 0x50 playback audio: `<BB2xQI>` = type, flags (bit0 = silent run, no payload), pad, s0 (uint64 first
  sample on the 24 kHz PLAYBACK timeline), n (uint32 samples); then n int16 PCM unless silent. The playback
  timeline starts at 0 when the voice session starts and advances continuously (silence included).
* `'M'` 0x4D mic (existing) — the user's mic, only when the server's fast intent matcher is on.
* `'J'` JSON: `{"type":"rig","t":<playback-timeline seconds>,"a":[...values in the order of config.rig_params]}`
  at ~30 Hz; `{"type":"turn",...}` latency events; `{"type":"usage",...}` vendor usage; `{"type":"event",...}`
  perception events; `{"type":"stats",...}`; existing types stay.

Down (server → browser), `[uint32 seq][uint32 len][msg]`:
* `'V'` video (existing header; `t` = playback-timeline seconds the frame depicts).
* `'J'` config `{"type":"config",...,"rig_params":[...],"fps":..,"w":..,"h":..}`; `{"type":"intent",
  "cmd":{...},"text":"...","heard_at":..}` from the fast matcher; stats.
* `'A'` audio is NOT sent in direct-voice mode (the browser already has the voice).

## Browser modules (all plain ES modules or IIFEs, no CDN, under app/)

* `voice.js` — `createVoice({mode:'realtime'|'live'|'mock', character, onCall(name,args,phase), onState(s),
  onMetric(m)})` → `{start(), stop(), sendEvent(text,{image,respond}), sendToolResult(id,out), audioNode}`.
  `phase` is 'start' (name known, args streaming) or 'done'. `audioNode` = the WebAudio node carrying the
  avatar's voice; the page graph does tap → server, DelayNode → speakers.
* `perception.js` — MediaPipe workers → `onFeatures(f)` with f = {t, face:{present, blend{52}, yaw, pitch,
  roll, x, y, z(distance), gazeX, gazeY}, hands:[{side, lm[21], x, y, vx, vy, open, gesture}], pose:{...},
  motion, approach, attention}. Draws the corner preview (+ hideable mesh). Emits discrete events
  (wave, look_back, face_appear, face_lost, object_up, approach_fast, point{dir}).
* `motion.js` — `createMotion({prior})` → `{step(dt, inputs) → {param:value}, command(cmd), release(parts)}`.
  inputs = {features, userVoice:{rms,pitch,voiced}, avatarVoice:{rms,pitch,voiced}, speaking, listening}.
* `prior.js` + `prior_weights.json` — the learned causal motion model; `createPrior(weights)` →
  `{step(featureFrame) → {head_yaw,...}}`; its feature spec documented in docs/PRIOR.md.
* Commands (from tools, the fast matcher, or the page UI) — one shape:
  `{action:'pose', targets:{param:value,...}, hold:true|false, ease_ms}` ·
  `{action:'gesture', name:'wave'|'nod'|'shake_head'|'wink_left'|'wink_right'|'blink'|'ear_wiggle'|'ear_flick_left'|
  'ear_flick_right'|'nose_wiggle'|'tail_wag'|'clap'|'thumbs_up'|'shrug'|'bow'|'point'|'stick_tongue_out'|
  'look_around'|'jump'|..., side, repeat, duration_ms}` · `{action:'look', target:'user'|'left'|'right'|'up'|
  'down'|'away'|'point_dir', hold}` · `{action:'expression', name, intensity, hold}` ·
  `{action:'release', parts:[...]|'all'}`. A held pose stays until released (by a release command, "okay/relax/
  you can stop/put it down"), eases in/out (default 250–400 ms, min-jerk), never snaps.

## Measurements every candidate reports (the latency table)

end-of-speech → first audio (ms, p50/p95) · command spoken → visible motion on the rendered video ·
camera cue → reflex visible on the rendered video · display fps and renderer p95 frame time (local GPU under
normal server load) · audio/lip offset (ms, cross-correlation of rendered mouth opening vs voice envelope) ·
uplink kbps / downlink kbps. Reject anything that worsens latency or fps.
