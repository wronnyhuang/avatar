# session_e2e — drive the real avatar page like a user, record, judge

```
PY=python3
$PY tools/session_e2e/run_session.py --url 'http://HOST:PORT/?k=KEY' --script tools/session_e2e/scripts/smoke.json --out OUT [--size 540p] [--voice mock] [--asr] [--no-judge] [--only 3,4]
$PY tools/session_e2e/run_session.py --post-only --out OUT        # redo frames + judge on a finished run
$PY tools/session_e2e/cut_proof.py --out proof.mp4 OUT1 OUT2:2-5,7 [--max-s 150] [--only-pass]
```
On a shared small GPU, serialize runs that use the real renderer with a lock, e.g. `flock "${AVATAR_GPU_LOCK:-runs/gpu.lock}" <cmd>`.

Files: `run_session.py` (TTS, driver, recording pickup, frames, judge, report), `driver.cjs` (Playwright; reactive turns),
`e2e_inject.js` (injected mic + tap), `e2e_lib.py`, `cut_proof.py`, `scripts/*.json`.
Chromium: playwright-core's own browser (`npx playwright install chromium`), or set `E2E_CHROME` / `CHROME_PATH`; the module comes from
`PLAYWRIGHT_CORE` (default `playwright-core`). Fake camera: `E2E_CAM` y4m (default: Chromium's test pattern).

## How it works
* The user's mic is a WebAudio MediaStreamDestination (getUserMedia override) fed per turn with cached TTS speech
  (OpenAI gpt-4o-mini-tts via OPENAI_API_KEY; cache `E2E_TTS_CACHE`, default `runs/session_e2e/cache/tts`).
* Turn start: `"wait": "idle"` = avatar silent (no audio, no motion/busy, no pending tool call) for `quiet_s` (default 1.8 s, script-level or per turn)
  AFTER it reacted (or `no_reply_s` = 12 s passed -> flag `no_reply`); `{"after_s": N}` = N s after the previous user line ended;
  `{"during": "writing"|"walking"|"tv"|"moving"|"speaking"|<regex>, "delay_s": 1.5}` = start while that is happening (regex matched against
  active motion names, tool names/args and `state:*` rows; not seen within `wait_max_s` (60) -> `during_missed`, the line is said anyway);
  `"immediate"`. Busy = `__avatar.mstate().active` non-empty, or `window.__avatar.e2eState().busy`.
* Taps (no page code changes): AvatarVoice.createVoice is wrapped -> every tool call (+args, result), transcript (the user's ASR and the avatar's), response, metric,
  voice state; 20 Hz sampler: avatar audio level (what the user hears), motion state changes, action-log events. OPTIONAL page hooks logged on change as
  `state:<name>`: `window.__avatar.e2eState() / .board() / .tv() / .teach() / .stations() / .program()` (JSON). Teach/TV builders: expose one of these
  (e.g. `{busy, station, board_items, tv_scene}`) and the timeline + `during` waits pick it up.
* Recording = the app's own session recorder (`?rec=1`): stage as shown + the user's self-view + both voices -> the server assembles the mp4 under
  `AVATAR_REC_ROOT` (the app's recordings root); `OUT/session.mp4` links to it. mp4 time = (wall - recording client_ts).

## Output (OUT)
`report.md` (one screen), `report.json`, `timeline.json` (per turn: user start/end, heard, avatar said, tool calls, motions, state changes,
scene states, aborts, latencies), `raw.jsonl` + `samples.jsonl` (everything), `frames/turnNN_k_tS.jpg` (judge frames), `strip/strip_NNN.jpg`
(a thumbnail every 2 s, 5x4 sheets), `session.mp4`, `recording.json`, `asr.txt` (--asr; needs `JUDGE_ASR_SCRIPT`, an external ASR script `<wav> <out.json> <threads>`), `page_report.json`, `driver.log`.
Latencies (ms): last word -> first avatar audio (as heard, after the A/V delay line), -> first motion, -> first tool call, tool -> first chalk mark
(only when a board tool + a chalk/stroke motion or `state:board` row exist).

## Limits
Turn-end detection is state based; a planner/board job that does not show up in `mstate().active` or `e2eState().busy` looks idle (add the hook).
Frame times use the recorder clock (client_ts), accurate to ~0.3 s. The server instance needs OPENAI_API_KEY in ITS env for realtime
(`tools/devinst.sh` passes its own environment through). The judge is `claude -p --model sonnet` (`JUDGE_CLAUDE` / `AVATAR_CLAUDE_BIN`).
