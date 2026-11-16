# Semantic motion API planner — interfaces

The README section "Semantic motion API" explains what it is and how to use it. This file is the wire-level contract
between the four pieces: the interpreter package (`app/semantic`), the server's planner (`app/planner.py`,
`app/app.py`), the page (`app/perform.js`, `app/page.html`, `app/rigdebug.js`) and the player (`app/motion.js`).
The API the model writes against is `app/semantic/prompt/API.md`.

## 1. Flow

```
spoken command -> voice tool perform_action(description) -> perform.js
  -> {type:'plan', id, character, description, state, planner:'semantic'}        (page -> server, control channel)
server (planner.py plan_semantic):
  1. library fast path (the named-move library, unchanged; source 'library')
  2. semantic cache AVATAR_HOME/cache/semplans/<key>.json (its own namespace; source 'cache')
  3. Claude streams a program -> semantic.Stream.feed(text delta) on a worker thread -> per complete top-level
     statement: IK + timing + shaping on the canonical skeleton -> retarget to the character's rig -> dense rig poses
  -> {type:'kf', id, space:'rig', hz, poses, done:false} per statement batch, then the done message
  -> {type:'sem', id, phase:'start'|'stmt'|'done', ...} (program text, timings, warnings) for the rig debug panel
page: motion.js plays the rig-space job through its keyframe-job layer (ownership, blend-in / blend-back, mouth yields
to speech, interruptions: the same rules as every commanded move)
```

## 2. `app/semantic` (Python, numpy + scipy)

```python
import semantic as sem                      # app/ on sys.path
sem.CHARACTERS                              # characters with a rig mapping: ('muse', 'fox')
sem.system_prompt(character, targets=None)  # header + API.md + live-target list + examples; stable text per
                                            # (character, sorted target names) so prompt caches hit
sem.user_prompt(description, state=None)    # 'Command: "..."'
sem.resolve_targets(character, state)       # {name: (x_cm, y_cm, z_cm)} in the API's pelvis frame; always 'camera'
                                            # and 'user' (+ user_left / user_right); scene things for the Muse
s = sem.Stream(character, targets=None, seed=0, hz=20, t_origin=0.0)
s.feed(text_delta) -> [batch, ...]          # interprets every top-level statement completed by this delta
s.finish() -> [batch, ...]                  # the rest + the automatic return-to-neutral tail (unless keep_pose())
s.report() -> {program, n_statements, duration_s, warnings, errors, exec_ms, keep_pose, channels}
sem.replay(character, program, targets=None, seed=0, hz=20) -> (poses, report)   # the whole program at once
```

- batch = `{"stmt": k, "t0": s, "t1": s, "poses": [pose, ...], "final": bool}`;
  pose = `{"t": <s from motion start>, "<rig param>": value, ...}` with the renderer's rig parameter names, values in
  rig units, inside each parameter's range.
- Time commitment: poses already returned are never revised; a later statement only adds poses after the last one.
  Shaping that would reach back in time is blended over the start of the new batch instead. Streamed output equals
  `replay` output (tested on the shipped corpus).
- Only the channels the program drives appear in a pose. When a part is released or the program returns to neutral,
  its channels ease to rest and the report lists them, so the page can blend the live layers back. The interpreter
  adds no idle noise, blinks or breathing; the live avatar does those.
- Robustness: a malformed statement is skipped and recorded in `errors`; a program with no valid statement yields
  no poses. The program runs in a restricted AST sandbox (no builtins, no attributes, no imports). Caps: 12 s of
  motion, 60 statements.
- Left / right: the API's left is the character's own left. Rig `_l` columns are SCREEN-left (the character's right).

## 3. Server

- `{type:'plan', ..., planner:'semantic'}` takes the semantic path; without `planner` the keyframe planner runs
  exactly as before. A character not in `sem.CHARACTERS` gets the keyframe planner and the done message carries
  `sem_unavailable: true`.
- Model: Claude only. Route `claude_door` (the `claude` CLI, docs/CLAUDE-DOOR.md) or `anthropic` (Messages API with
  ANTHROPIC_API_KEY, prompt-cached system prompt). No fallback to another provider or to the keyframe planner; one
  retry on the same model when the call fails before its first pose with at least `retry_min_s` left.
- Config `planner.semantic` in `AVATAR_HOME/state/config.json` (defaults in `SEM_DEFAULTS`):

  | key | default | meaning |
  |---|---|---|
  | `model` | `claude-sonnet-5-5` | any Claude model id (env `AVATAR_SEM_MODEL`) |
  | `effort` | `none` | `none` = thinking off; `low` / `medium` (env `AVATAR_SEM_EFFORT`) |
  | `route` | `auto` | `auto` (CLI if on PATH, else API key) / `claude_door` / `anthropic` (env `AVATAR_SEM_ROUTE`) |
  | `timeout_s` | 20 | per plan |
  | `retry_min_s` | 8 | retry only with this much time left |
  | `cache` | true | the semplans cache |
  | `hz` | 20 | pose rate of the rig track |
  | `suffix` | `auto` | the streaming note appended to the system prompt (first statement = the opening move); `null` = none |
  | `warm_n` | 2 | warm `claude` processes kept for the semantic prompt (CLI route) |
  | `hedge` | null | `{"model", "effort", "after_s"}`: a second call if the first has no pose after `after_s` |

- `{type:'semwarm', character}` (page -> server, sent once per session and character while the switch is on): warms
  the CLI door for the semantic system prompt; answer `{type:'semwarm', character, ok, available, model}`. Nothing is
  warmed while the switch is off.
- Cache entries whose program names a live target (`user`, a scene thing) are re-run with `sem.replay` and the fresh
  target positions instead of replaying stale poses.
- Messages server -> page:
  - `{type:'kf', id, space:'rig', hz, poses, done:false}` per batch; final `{type:'kf', id, space:'rig', poses:[],
    done:true, ok, source:'library'|'cache'|'planner', planner:'semantic', engine, summary, ms_first_pose, ms_done,
    usd, error?, keep_pose?, channels?}`. A library hit is the keyframe planner's library message plus
    `planner:'semantic'`.
  - `{type:'sem', id, phase, program, source, model, cached, ms:{recv, first_token, first_stmt, first_pose, done},
    exec_ms, warnings, errors, targets}`.
- Session records: plan records gain `planner:'semantic'`, the program and the timings.

## 4. Page

- Switch: `?planner=semantic` (alias `?sem=1`) at load, the rig debug panel's Planner row, or
  `window.__avatar.setPlanner('semantic'|'current')`; applies to the next command, never persisted.
  `window.__avatar.planner()` -> `'current' | 'semantic'`; `window.__avatar.semLast()` -> the latest semantic plan's
  info (works with the panel closed).
- With the switch on, plan messages carry `planner:'semantic'` and `state.sem = {user: {present, ax, ay, z, size},
  attending}` (the face position gaze following uses). With it off, every control message is byte-identical to the
  tree before this feature. The page-side plan cache is keyed by mode.
- `motion.js` accepts keyframe jobs with `space:'rig'`: poses keyed by rig parameter names (no alias mapping), dense
  (`hz`), played by non-overshooting interpolation; ready once 0.4 s is buffered or the plan is done; blend-in
  0.05-0.6 s from the live value on a channel's first touch; blend-back when the channel's track or the job ends;
  mouth channels yield to lip sync while the avatar speaks; gaze channels are owned only while the program drives
  them; `keep_pose` holds the last pose like a commanded pose; 'stop' or a newer request interrupts. Accepted
  channels: the rig dials except the renderer's lip columns and the legacy arm columns, plus `fingers_l/r` and the
  root group (`root_*`, `squash`). Unknown channels are dropped and counted (`kfState().unknown`).
- The thinking gesture runs while the program is written and hands over to the first poses.
