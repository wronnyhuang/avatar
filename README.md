# Live avatar: a real-time video-chat avatar

A 3D character you video-chat with in the browser. It **sees you** (MediaPipe face, body and hands run on your
device; still frames go to the voice model) and **talks** (OpenAI GPT Realtime by default, Gemini Live optional).
It keeps its **gaze** on you, **waves back**, mirrors your expressions and **does moves on command**: any move,
written on the fly by a fast planner model ("do a backflip", "dance", "sit at the desk and put on your headphones").
The main character is **Muse**, a plush teacher with a **chalkboard** (writes and draws math and diagrams in chalk,
then erases) and a **wall TV** (live animated explanations), plus objects it can pick up and use.

Everything runs on one Linux machine with a GPU: the Python server, the real-time SDF renderer and the H.264
encoder. Only the voice model and the planner are cloud APIs.

## Quick Start

**Requirements**
- Linux (tested on Ubuntu 24.04) with an NVIDIA GPU and working EGL. A 2 GB card renders 270p at 24 fps. Without a
  GPU, the renderer falls back to Mesa llvmpipe on the CPU, which is slow but works for a smoke test.
- Python 3.12, `python3-venv`, `curl`. Node 22 is needed only for the JS test suites.
- Chrome or another Chromium-based browser for the page (it needs WebRTC, a camera and a microphone).
- At least one voice key: an **OpenAI API key with Realtime access** (default voice) or a **Gemini API key**
  (Gemini Live voice). See [API keys](#api-keys).

```bash
# 1. clone
git clone https://github.com/wronnyhuang/avatar.git && cd avatar

# 2. install: .venv with pinned deps, .avatar-home/ with config + a random page secret, .env from .env.example
scripts/setup.sh

# 3. keys: put OPENAI_API_KEY (voice) and, optionally, ANTHROPIC_API_KEY (planner, board, TV) into .env
$EDITOR .env

# 4. optional check, no keys needed: boots the server, checks the page, warms the renderer
scripts/smoke.sh

# 5. start the server; it prints the page link and which keys it found
scripts/run_local.sh
#   open: http://127.0.0.1:10250/?k=<secret>   (Chrome; allow camera + microphone)
```

6. Open the printed link in Chrome on the same machine, allow the camera and microphone, press **Start**, then say
   hi. Try "wave", "do a backflip", "sit at your desk and put on your headphones", "explain fractions on the
   board" or "show me how a lens works on the TV".
   With only a Gemini key, open the link with `&voice=gemini` added, or pick *Gemini Live* in the Voice menu.

The `?k=` value is the page secret stored in `.avatar-home/state/secret`. Every page and API request must carry it.
To use the page from a phone or another computer, put the server behind HTTPS (browsers only allow the camera
and microphone on a secure origin). Any reverse proxy or tunnel to `127.0.0.1:10250` works; then open
`https://<your-host>/?k=<secret>`.

Useful URL flags: `&char=fox` (other characters), `&voice=gemini`, `&react=unified` (learned face model, needs
`onnxruntime`), `&rec=1` (record the session to `recordings/`; recording is off by default), `&hud=1` (stats),
`&planner=semantic` (the [semantic motion API](#semantic-motion-api) planner), `&rig=1` (open the rig debug panel).

## API keys

Keys come from the environment. `scripts/run_local.sh` loads `.env` for any variable your shell has not already
set. At startup the server prints each key as *set* or *missing*, plus the planner route it chose. If the voice
key is missing it prints a warning that names the key and where to get it. The page still loads; pressing Start
shows "Missing OPENAI_API_KEY …" instead of failing silently. Set `AVATAR_REQUIRE_KEYS=1` to make the server
refuse to start without it.

| env var | what it is used for | required? | if it is missing | where to get it | rough cost |
|---|---|---|---|---|---|
| `OPENAI_API_KEY` | Default voice: GPT Realtime over WebRTC from the page; the server only mints the session, so the key never reaches the browser. Also used by the server-relayed voice (`?voice=hub`) and as the last-resort planner and judge model. | **Required for the default voice.** Not needed if you use Gemini Live. | The page and avatar load; Start shows "Missing OPENAI_API_KEY". Gemini Live still works with its own key. | [platform.openai.com/api-keys](https://platform.openai.com/api-keys); the account needs Realtime access and credit | Roughly $0.05–0.25 per minute of conversation (audio tokens, re-billed context mostly at the cached rate; longer sessions cost more). [Pricing](https://openai.com/api/pricing/) |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini Live voice (`&voice=gemini`, or the Voice menu); also the planner's last-choice route (Gemini Flash Lite). | Optional. It is the alternative voice key. | Gemini voices report the missing key. | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | A few cents per minute or less; the free tier has low daily limits. [Pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| `ANTHROPIC_API_KEY` | The **planner** ("do X" → keyframes, Claude Haiku), the **chalkboard brain** and the **wall-TV agent** (Claude Sonnet/Opus), and the feelings classifier. It is used when the `claude` CLI is not installed. | Optional, but recommended. | With the `claude` CLI on PATH (Claude Code, signed in), those features run through the CLI and need no key. With neither: the planner falls back to OpenRouter, then OpenAI, and the chalkboard and TV stay off. | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | Planner: well under 1¢ per move. Board or TV program: a few cents each. [Pricing](https://www.anthropic.com/pricing#api) |
| `OPENROUTER_API_KEY` | Planner on Gemini Flash Lite, with DeepSeek as its fallback; the behavior coder and the attention judge. | Optional. | The planner uses the next available route; the judges use `OPENAI_API_KEY`. | [openrouter.ai/keys](https://openrouter.ai/keys) | Under 0.1¢ per planned move. [Pricing](https://openrouter.ai/models) |
| `META_API_KEY` | Only the optional planner route `meta` (Meta Llama API). | Optional. | That route errors with "no META_API_KEY". | [llama.developer.meta.com](https://llama.developer.meta.com) | See the provider |
| `AVATAR_CLAUDE_BIN` | Path to the `claude` CLI, if it is not on PATH. | Optional. | `claude` is looked up on PATH. | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Billed to your Claude plan |

**Planner route order** (`"model": "auto"`, the default): the `claude` CLI on PATH, then `ANTHROPIC_API_KEY`, then
`OPENROUTER_API_KEY`, then `OPENAI_API_KEY`, then `GEMINI_API_KEY`. The fallback route is the next one available. To override, set
`AVATAR_PLANNER_MODEL=<preset>` (presets are listed in `app/planner.py`) or `"planner": {"model": …}` in
`.avatar-home/state/config.json`.

**Spend cap:** the server meters what it spends on the voice, planner and judges in `data/spend-*.json`, and stops
starting new sessions at `daily_cap_usd` (default $10/day; set it in `config.json`). The costs above are rough
list-price estimates; check each provider's pricing page.

Every variable is listed with comments in [`.env.example`](.env.example). Runtime knobs: `PORT` (10250), `HOST`
(127.0.0.1), `AVATAR_HOME` (state and data; `run_local.sh` uses `./.avatar-home`), `AVATAR_REC_ROOT` (recordings),
`AVATAR_RT_DEVICE=cpu` (force CPU rendering).

## Architecture

```
 Browser page (app/page.html + *.js)                       App server (app/app.py, aiohttp, 127.0.0.1:$PORT)
 ├─ camera → MediaPipe (app/vision/*.wasm, *.task)         ├─ auth: ?k=<state/secret> + x-requested-with header
 │   landmarks + occasional stills ──────────────────────► ├─ voice: page↔OpenAI Realtime direct (rtc.py mints the
 ├─ mic ⇄ voice model (WebRTC, tools in voicetools.js)      │   session), server bridge (vendor.py) or Gemini Live
 ├─ move engine (motion.js, moves.js, perform.js,           ├─ planner.py   — "do X" → keyframe JSON (claude CLI /
 │   gestures.js, behavior.js, teach.js, attention.js…)     │   Anthropic / OpenRouter / OpenAI routes), cache; with
 │   → 34 rig controls + face/gaze/mouth, 24 fps  ───────►  │   ?planner=semantic: Claude writes a semantic-API program,
 ├─ rig debug panel (rigdebug.js; ?rig=1, off by default)   │   semantic/ runs it as it streams (IK, rig map) → poses
 └─ <video> ◄──── H.264 stream ◄──────────────────────────  ├─ board_brain.py + board/ — what to write on the board
                                                           ├─ tv_agent.py + tvdsl/ — wall-TV program + renderer
                                                           ├─ teach_hub.py, surfaces.py — board/TV canvases → renderer
                                                           ├─ affect.py, attention_judge.py, behavior_coder.py, intent_*
                                                           ├─ unified_link.py — optional learned face/gaze model
                                                           ├─ recorder_hub.py — session recordings (opt-in, ?rec=1)
                                                           └─ providers.py — the local render server child process
 Renderer (renderer/, avatar/render_server.py)
   SDF ray-marching in GLSL (moderngl, headless EGL) → NVENC/x264 H.264 (PyAV) → websocket to the app server
   characters are math (renderer/animals/*.glsl, *.body.json, *.props.json), not meshes; IK arms, planted feet
 Motion model (motion-model/): research. A unified flow-matching model (causal transformer + rectified-flow head)
   maps your face, body and voice plus its own speech to face/gaze/posture. Includes the data pipeline, training
   and evals.
```

Docs: `docs/RIG.md` (the rig), `docs/PLANNER.md` (planner), `docs/SEMANTIC.md` (semantic motion API planner),
`docs/RIG-DEBUG.md` (rig debug panel), `docs/BOARD-BRAIN.md`, `docs/TV-AGENT.md`,
`docs/SCENE.md` and `docs/OBJECTS.md` (board, TV, objects), `docs/INTENT-RULES.md`, `docs/CLAUDE-DOOR.md`,
`motion-model/docs/CONTRACT.md` (canonical motion format), `motion-model/README.md`.

## Repo layout

| path | what |
|---|---|
| `app/` | app server and the browser page (Python + plain JS, no build step) |
| `app/semantic/` | semantic motion API: interpreter, streaming, IK / per-character rig mapping, the planner prompt (`prompt/API.md`) |
| `renderer/` | real-time SDF renderer, encoder, render server |
| `deploy/config.local.example.json` | config template that `scripts/setup.sh` copies to `.avatar-home/state/config.json` |
| `scripts/` | `setup.sh`, `run_local.sh`, `smoke.sh`, `test_all.sh` |
| `tests/` | offline unit suites (`test_*.py`, `*_test.cjs`) + fixtures (`fixtures/sem_programs.json`: semantic-API program corpus) |
| `tools/`, `eval/` | eval harnesses: planner evals, session end-to-end driver + judge, latency, TV/board evals, voice evals |
| `motion-model/` | learned motion model: `canon1/` data format, `unified/` model + runtime + ONNX export, `retarget/`, `evalkit/`, `tools/data`, `tools/train` |
| `models/unified-v2/` | the small inference bundle for the opt-in learned face model (`?react=unified`) |

## Semantic motion API

An opt-in planner for commanded moves. Instead of asking a model for raw rig keyframes (the default planner,
`docs/PLANNER.md`), Claude writes a short program of goal-level calls — `reach('right', 'top_of_head')`,
`look_at('user')`, `point_at('board')`, `expression('surprise')` — with named body targets and real units (cm,
degrees, seconds). An interpreter turns it into motion: inverse kinematics for the arms, timing curves and shaping
(anticipation, overshoot, eyes leading the head), retargeting to the character's rig. Models are much better at
"put the left hand on the right shoulder" than at choosing `hand_l_x = 0.42`, so spatial and multi-step commands
come out right far more often (results below).

### Switching it on

- URL: `?planner=semantic` (or `?sem=1`) on the page link. The default is the keyframe planner on every load.
- In the page: open the **Rig** panel (button next to *Stats*, or `?rig=1`); its top row is a *current | semantic API*
  switch that applies to the next command.
- From the console / tests: `window.__avatar.setPlanner('semantic')`, `window.__avatar.semLast()`.
- Model: Claude only. With the `claude` CLI on PATH (Claude Code, signed in) it runs through the CLI and needs no key;
  otherwise it uses `ANTHROPIC_API_KEY`. No other provider is used and it never falls back to the keyframe planner.
  Config `"planner": {"semantic": {...}}` in `.avatar-home/state/config.json`:

  | key | default | |
  |---|---|---|
  | `model` | `claude-sonnet-5-5` | any Claude model id; env `AVATAR_SEM_MODEL` |
  | `effort` | `none` | thinking off (fastest; `low` was slower and no better); env `AVATAR_SEM_EFFORT` |
  | `route` | `auto` | `claude_door` (CLI), `anthropic` (API key), `auto` = CLI if found; env `AVATAR_SEM_ROUTE` |
  | `timeout_s`, `retry_min_s` | 20, 8 | one retry on the same model if it fails before its first pose |
  | `cache`, `hz` | true, 20 | program cache (`.avatar-home/cache/semplans/`), pose rate |

  Sonnet with thinking off was as fast to the first pose as Haiku and judged clearly better, so it is the default.
  Full list and the wire messages: `docs/SEMANTIC.md`.

### How the pieces fit

```
spoken command -> voice tool perform_action -> page (perform.js) -> {type:'plan', planner:'semantic'}
server (app/planner.py)
  named-move library hit? -> played at once (unchanged)          cached program? -> served (re-run if it uses live targets)
  else Claude streams the program text ──► semantic.Stream.feed(delta)   (worker thread, ~5-10 ms per statement)
     each complete top-level statement ──► goals (reach / look / point / pose / face ...)
                                       ──► IK + look-at solver, timing + shaping
                                       ──► per-character rig mapping (hand geometry, head / torso gains, face map)
                                       ──► dense rig-space keyframes {t, <rig param>: value} for that statement
  -> {type:'kf', space:'rig', poses} per statement (motion starts after the first one) + {type:'sem'} progress
page (app/motion.js keyframe-job layer): blends the track in from the live values and back out at the end; it owns
only the channels the program drives — gaze following, lip sync, mirrored expressions, idle motion and reflexes keep
every other channel; the mouth yields to lip sync while the avatar speaks; a new command or 'stop' interrupts.
```

Already-sent poses are never revised, so the avatar starts moving as soon as the first statement is complete (the
prompt asks for the opening move first). Streaming a program gives exactly the same poses as replaying it whole.

### The API

Reference (this is also what the model reads): [`app/semantic/prompt/API.md`](app/semantic/prompt/API.md).
[Interactive explainer](docs/semantic-api-explainer.html) (open the file locally in a browser): every call with a
Play button on the real character shader, all rig dials to drag, and an in-page interpreter for your own programs.
Families:

- **Arms and hands**: `reach`, `touch`, `cover`, `point_at`, `arm` (a direction from the shoulder), `raise_hand`,
  `move_hand` (relative cm), `rotate_wrist`, `hand` (shapes: fist, point, thumbs_up, peace, ok, ...), `pat`, `rub`,
  `hands_together`, `rest_arm`, gestures `wave`, `clap`, `thumbs_up`, `shrug`, `cross_arms`, `hands_on_hips`.
- **Head and eyes**: `look_at`, `rotate('head'|'torso'|'body', yaw, pitch, roll)`, `nod`, `shake_head`, `tilt_head`,
  `head_turn`, `gaze`, `roll_eyes`.
- **Torso and whole body**: `lean`, `twist`, `shoulders`, `bow`, `jump`, `flip`, `spin`, `turn`, `crouch`, `stand`,
  `step`, `sway`, `bounce`, `oscillate` (an additive wiggle on any part).
- **Face**: `expression` (30 named), `face(**ARKit blendshapes)`, shortcuts `smile`, `wink`, `blink`, `close_eyes`,
  `raise_brow`, `furrow`, `puff_cheeks`, `open_mouth`, `kiss`, ...
- **Composition**: `with parallel():`, `with sequence():`, `with after(s):`, `for i in range(n):`, `hold`, `release`,
  `return_to_neutral`, `keep_pose`.
- **Targets**: body points (`nose`, `chin`, `top_of_head`, `left_shoulder`, `chest`, `right_palm`, ...), directions
  (`up`, `upper_left`, `in_front_of_face`, ...), `(x, y, z)` cm from the pelvis, and live targets: `camera`, `user`
  (the person's face from the camera, or the camera when nobody is seen), `user_left` / `user_right`, and the scene's
  things for the Muse (`board`, `tv`, `desk`, `laptop`, `chair`, ...). Left / right are always the avatar's own.

The program language is a restricted Python subset run in an AST sandbox (no imports, attributes, `if` or `def`).

### Example programs

"Glance at the laptop on your desk, then back at me" (the Muse; the fox has no laptop and looks at the camera):
```python
look_at('laptop', dur=0.5)
with parallel():
    lean('forward', deg=8)
    expression('thinking', 0.6)
hold(0.7)
with parallel():
    look_at('user', dur=0.4)
    lean('back', deg=0)
    expression('smile', 0.6)
hold(0.4)
```

"Rub the back of your neck, a bit embarrassed":
```python
reach('right', 'back_of_head', part='palm', offset_cm=dict(down=8), dur=0.6)
with parallel():
    rub('right', 'back_of_head', times=3, cm=4)
    tilt_head('right', deg=10)
    look_at('down', eyes_only=True, dur=0.3)
    expression('embarrassed', 0.8)
hold(0.4)
```

"Clap both hands over your mouth like you said something you shouldn't":
```python
cover('mouth', hand='both', dur=0.4)
with parallel():
    expression('surprise', 0.9)
    lean('back', deg=6)
    gaze(yaw=15, dur=0.2)
hold(0.8)
```

"Make an OK sign with your left hand and keep it there":
```python
raise_hand('left', height='shoulder', palm='camera', shape='ok', dur=0.5)
with parallel():
    expression('smile', 0.7)
    tilt_head('left', deg=8)
keep_pose()
```

### Using the interpreter on its own

`app/semantic` needs only numpy and scipy and does not import the server.

```python
import sys; sys.path.insert(0, "app")
import semantic as sem

prog = "look_at('camera')\nwave(hand='right', cycles=3)\nhold(0.4)\n"
targets = sem.resolve_targets("muse", None)            # {name: (x, y, z) cm}; pass the page's state for live targets
poses, report = sem.replay("muse", prog, targets)      # [{t, <rig param>: value}, ...] at 20 Hz
print(report["n_statements"], report["duration_s"], report["channels"], report["errors"])

s = sem.Stream("fox", targets=sem.resolve_targets("fox", None))   # incremental: feed model text as it arrives
batches = s.feed("```python\nnod(times=2)\n") + s.feed("smile()\n```\n") + s.finish()
system = sem.system_prompt("muse", targets)            # the planner's system prompt (stable text per target set)
```

`tools/sem_smoke.py` replays the prompt's examples and a corpus for every character (no key); `tools/sem_bench.py`
times the interpreter; `tools/sem_stills.py` renders the end pose of each program on the real renderer (CPU) as a
contact sheet — the way to check hand-on-target accuracy; `tools/sem_llm_try.py` runs the live prompt through
`claude -p`.

### Adding a character

The interpreter works on a canonical human skeleton and maps to each rig in `app/semantic`:

1. `fastrender.py` `CHAR[...]`: head gain (rig radians per real radian), torso radians per rig unit (yaw, pitch,
   lean), gaze tangent per unit, whether it has root motion (can stand / jump / spin) and ears.
2. `arms.py`: the hand geometry — where a rig hand target `hand_<s>_x/y/z` puts the wrist. For characters whose
   `renderer/animals/<name>.body.json` has a linear `hand_map` (the Muse) this is read from that file; otherwise
   invert the shader's hand-target function (the fox: `fox.glsl` `handTarget()`), and give the face and torso points
   in the character's head / body space.
3. `rigmap.py` `HAS_ROOT` / `HAS_EARS`, `retarget/maps/<name>.json` (the canonical-skeleton path), `targets.py` scene
   things (from `<name>.props.json`), `prompt/__init__.py` `CHAR_NOTES` (one line on what the body can do), and add the
   name to `semantic.CHARACTERS`. A character not in the list gets the keyframe planner and the panel says
   "semantic API not available for this character".
4. Check: `tools/sem_stills.py --character <name>` (touch nose / top of head / shoulder, cover eyes, point left / right
   / up / camera) and `tests/test_sem_core.py` (left / right, targets land, no pops).

Rig `_l` columns are screen-left, i.e. the character's own right hand; the API's "left" is the character's own left.

### Results

Offline bake-off: 106 held-out commands × 3 seeds, the same Claude model writing either raw rig keyframes (the
current planner's approach) or semantic-API programs, rendered and judged blind from contact sheets (Claude Sonnet as
the judge) plus automatic checks per command.

| | keyframe planner | semantic API |
|---|---|---|
| action checks passed (all 106) | 71% | **91%** |
| spatial commands (24) | 57% | **100%** |
| compound / multi-step (15) | 58% | **100%** |
| face (20) | 65% | **95%** |
| whole-body moves (15) | **98%** | 87% |
| blind "looks like real motion" | 20% | 27% (real mocap: 83%) |
| interpreter time | – | ~5-10 ms per statement, ~20-35 ms per whole 3-6 s program |

It does what was asked much more often, and it is slightly preferred for naturalness, but it still looks robotic:
motion is built from eased segments and procedural shaping, not learned from human motion. Learned in-betweening
(a motion model filling the timing and secondary motion between the program's goals) is future work. Whole-body moves
(jumps, spins, flips) are better with keyframes, which can shape the root freely.

<!-- LIVE-RESULTS:START -->
**Live comparison** (both planners through the real page, server and renderer; 41 fresh spoken-style commands per
character, the two planners interleaved command by command on a busy shared machine; "planned" = commands that are
not instant library moves; blind Claude judge on rendered clips, score = yes + half of partly; judge noise is about
+-15 points per character, so read small gaps as ties):

| planned commands | keyframe planner | semantic API |
|---|---|---|
| did the command, Muse (n 29) | 55% | 62% |
| did the command, fox (n 33) | 56% | 58% |
| did the command, a third human-proportioned rig not in this repo (n 29) | 38% | 53% |
| blind pairwise "more natural" (n 91) | 40% | 60% |
| command to move starting, median / p90 | 4.25 s / 6.36 s | 2.35 s / 4.51 s |
| server first pose after the request, median | 2.81 s | 2.01 s |
| repeated command (cache), move starts | 0.79 s | 0.35 s |
| interpreter errors / timeouts in 91 plans | n/a | 2 skipped statements / 0 |

Where it is better: simultaneous asks (point and look), spatial targets, staying in place and returning cleanly, and
time to first motion. Where it is not: whole-body moves on a seated torso-only character, reaches the rig's arms
cannot make (cross-body, top of a large head), and the motion still reads as scripted: one ease in, a hold, the same
ease back.
<!-- LIVE-RESULTS:END -->

### Known limits

- Still robotic (see above); hand contact is approximate on stylised bodies (a paw on top of a big head lands on the
  temple), and the fox's paws cannot form separate fingers (shapes become open / fist / point / thumb).
- Whole-body moves on the fox (seated) degrade to torso moves.
- Latency = model time to the first complete statement plus ~5-10 ms; a cold CLI process or no prompt-cache hit adds
  a second or more. Repeats come from the cache at once.
- Live targets are resolved when the plan starts (a program that tracks `user` does not follow later head motion).
- The interpreter has rig mappings for a third rig family (a human character) that is not part of this repo.

### Rig debug panel

The **Rig** button (or `?rig=1`) opens a live view of the rig: every rig dial with its value and which layer drives
it (gaze, mirror, listen, speech, feeling, reflex, command, hold, planner, lips, ...), the renderer's final lip values
for the frame on screen, the active gestures / holds / planner job, perception (face / hands tracking, voice
activity), an event log, and the Planner section (switch, program as it streams, timings, warnings). Closed, it costs
nothing (one null check per frame, no messages). Details: `docs/RIG-DEBUG.md`.

## Optional extras

- `pip install onnxruntime` (into `.venv`): enables the learned face/gaze model, `&react=unified`.
- `pip install vosk` plus [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models) unpacked under `models/`:
  the server's streaming keyword recognizer for faster move commands.
- The `claude` CLI ([Claude Code](https://docs.anthropic.com/en/docs/claude-code)), signed in: planner, chalkboard
  and TV agent without an Anthropic API key.

## Tests and evals

```bash
scripts/test_all.sh            # offline Python + Node unit suites (no keys), ~10 min. The page tests drive Chrome:
                               #   npm i playwright-core (or PLAYWRIGHT_CORE=<path>), CHROME_PATH=<chrome binary>.
                               # Suites that need non-shipped data (eval clips from Seamless) print SKIP.
                               # Known failure: muse_regressions_test.cjs check B9 ("reverse burst").
                               # Includes the semantic API suites (test_sem_core.py, test_sem_hub.py,
                               #   sem_page*_test.cjs, rigprobe_test.cjs), renderer/avatar/test_rigdbg.py and
                               #   tools/sem_smoke.py.
scripts/smoke.sh               # boot + page + auth + real renderer warm-up + semantic interpreter replay, no keys
                               #   (SMOKE_RIGDBG=1: + the rig debug relay end to end through the real renderer)
python tests/sem_hub_smoke.py --url 'http://127.0.0.1:10250/?k=<secret>'   # semantic planner end to end on a
                                                                 # running server (needs the claude CLI or ANTHROPIC_API_KEY)
python tools/sem_bench.py                 # interpreter speed + stream == replay on the program corpus
python tools/session_e2e/run_session.py --url 'http://127.0.0.1:10250/?k=<secret>' \
  --script tools/session_e2e/scripts/smoke.json --out runs/e2e   # drives the real page like a user (Playwright,
                                                                 # TTS mic, recording, judge); needs keys
python tools/plan_eval.py --help          # planner evals (see tools/README-plan-eval.md)
python eval/latency/run_live.py --help    # latency (last word → audio / motion)
```

Judges and evals call text models through the `claude` CLI (`claude -p`). The Node tools need `playwright-core`
(`npm i playwright-core`, or set `PLAYWRIGHT_CORE` to its path) and a Chromium (`CHROME_PATH`). Training the motion
model needs `torch`, `onnxruntime`, `safetensors`, `soundfile`, `mediapipe` and `transformers`, plus the datasets
(see `motion-model/README.md`).

## Known limitations

- New, never-seen move commands work about half the time with the default keyframe planner (some come out wrong or
  too small); the opt-in semantic API planner gets most spatial and multi-step commands right but still looks
  robotic (see [Semantic motion API](#semantic-motion-api)). A move starts about 3 s after your last word.
- The voice model sometimes answers other people in the room, can talk out of turn, and occasionally claims a move
  it didn't make. You interrupt it by tapping, not by voice.
- The learned face model (`?react=unified`) is flat (few blinks or nods, no smile-back). The rule-based reactions
  are the default and still win blind comparisons.
- Teacher mode: about 5 s from request to the first chalk mark. Chalk is small on a 540p stream. TV follow-ups
  ("make the focal length shorter") land about half the time.

## Data and licensing

The motion models are trained on datasets released under non-commercial research terms: Meta Seamless
Interaction, CANDOR, Learning to Listen, ViCo and CMU Panoptic. YouTube clips listed in `docs/CORPUS.md` were used
only for aggregate statistics and are not redistributed. MediaPipe assets under `app/vision/` are Google's
(Apache-2.0). Not for commercial use.
