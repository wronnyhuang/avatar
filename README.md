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
`onnxruntime`), `&rec=1` (record the session to `recordings/`; recording is off by default), `&hud=1` (stats).

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
 │   gestures.js, behavior.js, teach.js, attention.js…)     │   Anthropic / OpenRouter / OpenAI routes), cache
 │   → 34 rig controls + face/gaze/mouth, 24 fps  ───────►  ├─ board_brain.py + board/ — what to write on the board
 └─ <video> ◄──── H.264 stream ◄──────────────────────────  ├─ tv_agent.py + tvdsl/ — wall-TV program + renderer
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

Docs: `docs/RIG.md` (the rig), `docs/PLANNER.md` (planner), `docs/BOARD-BRAIN.md`, `docs/TV-AGENT.md`,
`docs/SCENE.md` and `docs/OBJECTS.md` (board, TV, objects), `docs/INTENT-RULES.md`, `docs/CLAUDE-DOOR.md`,
`motion-model/docs/CONTRACT.md` (canonical motion format), `motion-model/README.md`.

## Repo layout

| path | what |
|---|---|
| `app/` | app server and the browser page (Python + plain JS, no build step) |
| `renderer/` | real-time SDF renderer, encoder, render server |
| `deploy/config.local.example.json` | config template that `scripts/setup.sh` copies to `.avatar-home/state/config.json` |
| `scripts/` | `setup.sh`, `run_local.sh`, `smoke.sh`, `test_all.sh` |
| `tests/` | offline unit suites (`test_*.py`, `*_test.cjs`) + fixtures |
| `tools/`, `eval/` | eval harnesses: planner evals, session end-to-end driver + judge, latency, TV/board evals, voice evals |
| `motion-model/` | learned motion model: `canon1/` data format, `unified/` model + runtime + ONNX export, `retarget/`, `evalkit/`, `tools/data`, `tools/train` |
| `models/unified-v2/` | the small inference bundle for the opt-in learned face model (`?react=unified`) |

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
scripts/smoke.sh               # boot + page + auth + real renderer warm-up, no keys
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

- New, never-seen move commands work about half the time (some come out wrong or too small). A move starts about
  3 s after your last word.
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
