# claude_door — streaming LLM text through the Claude CLI

`app/claude_door.py`: streaming text through a logged-in `claude` CLI (subscription login, no API key) by driving it as a subprocess. Used by the keyframe planner (`docs/PLANNER.md`, route `claude_door`) and the board brain.

```python
from claude_door import door
async for ev in door.stream(model="haiku", system=S, user=U, images=None, max_s=20, tag="board"): ...
ev = await door.complete(...)      # the done event
door.prewarm("haiku", S, n=2); await door.close()
```
Events: `{"delta": str}` then `{"done": True, "ok", "text", "ms_first", "ms_done", "error", "model", "tag", "warm", "usage", "cost_usd"}`
(`usage` has input/output/cache_read/cache_creation tokens; cost is the notional API-equivalent, nothing is billed).

## How
One process per request (fresh context), pre-spawned by `prewarm` (default 2 per (model, system) pair, 6 total) and replaced
in the background; idle warm processes recycle after 240 s. Timeout (`max_s`) / early generator close / error kills the
process by exact pid (`proc.kill()`, own session) and reaps it. stderr is captured into `error`. cwd = a private empty temp
dir. `ANTHROPIC_API_KEY` is removed from the child env (subscription login only). Needs HOME set (systemd --user ok, no TTY).
Env: `AVATAR_CLAUDE_BIN` (binary), `AVATAR_CLAUDE_EFFORT` (default `low`, empty = unset). `prewarm` before the first call:
an un-prewarmed pair is a cold spawn (+~0.6 s).

Flags: `claude -p --input-format stream-json --output-format stream-json --include-partial-messages --verbose --model <m>
--system-prompt <s> --tools "" --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence
--effort low`. `--bare` is NOT usable (needs an API key, no OAuth). `--effort low` is the important one: without it Haiku
5.5 randomly enters hidden adaptive thinking on ~15% of calls (first delta +1.5-2.5 s, 2x output tokens); with it 0 of 40 did.
`MAX_THINKING_TOKENS=0`, `CLAUDE_CODE_DISABLE_THINKING`, `alwaysThinkingEnabled=false` did not stop it.

Images: `images=[paths]` become base64 image blocks in the stream-json user message (verified live).

## Findings
- The process is not the bottleneck: a fresh process answers ~0.2 s slower than a reused one; startup before the first
  message is not observable (the CLI emits `init` only after the first user message). Latency is API time-to-first-token (~1.0-1.3 s Haiku).
- Prompt cache works across processes for the same (model, system): ~2.2k tokens cache_read on every call after the first
  (baseline CLI prefix is ~550 tokens even with no tools).
- Reusing a process for several turns saves nothing measurable and leaks context, so it is not done.

## Measured (desktop server under moderate load, 1.5k-token system, ~150-token answer, n=15 each, ms)
| model | mode | first p50/p90 | done p50/p90 |
|---|---|---|---|
| haiku | cold (spawn per call) | 1807/1939 | 2881/3000 |
| haiku | warm | 1216/1382 | 2295/2367 |
| haiku | warm, 3 concurrent | 1288/1664 | 2334/2571 |
| sonnet | cold | 1847/3836 | 4115/5784 |
| sonnet | warm | 1002/6667 | 3351/8828 |
| sonnet | warm, 3 concurrent | 1248/3659 | 3592/6094 |

Sonnet p90 is a heavy tail (occasional multi-second server stalls); p50 ~1.0-1.2 s. Target "Haiku first delta well under
1.5 s" is met warm at p50/p90 (1.2/1.4 s); the floor is the API round trip. `python3 app/claude_door.py --bench [--models haiku,sonnet --n 15]`.

Tests: `python3 -m unittest tests/test_claude_door.py` (fake binary); live: `AVATAR_DOOR_LIVE=1`.
Limits: subscription rate limits apply; no tools/streaming of tool calls; one request per process.
