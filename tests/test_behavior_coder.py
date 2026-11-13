"""program layer hub side: app/behavior_coder.py with a mocked model (no network, no spend).
python3 tests/test_behavior_coder.py  (needs node; uses the real tools/behavior_check.cjs)"""
import asyncio, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import behavior_coder as BC  # noqa: E402

GOOD = "```js\n({ setup(api) { api.claim(['ears']); }, per_frame(api) { if (api.user.touching('nose')) api.set('right_ear', Math.sin(api.t * 20)); } })\n```"
BAD_EVENT = "({ setup(api) { api.claim(['ears']); }, per_frame(api) { for (const e of api.events('touch_nose')) api.set('right_ear', 1); } })"
UNCLAIMED = "({ per_frame(api) { api.set('smile', 1); } })"
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  — {detail}" if detail else ""))
    fails += 0 if ok else 1


async def main():
    seen = []

    def fake(answers):
        async def call_model(http, name, system, messages, timeout, max_tokens):
            seen.append((name, messages, system))
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return {"text": a, "usd": 0.001, "ms": 10, "usage": {}}
        return call_model

    coder = BC.BehaviorCoder({"model": "gpt-5.4-mini", "fallback": "deepseek-v4.1-flash"}, log=lambda *_: None)
    BC.call_model = fake([GOOD])
    r = await coder.code({"id": "1", "instruction": "when I touch my nose wiggle your ear", "name": "nose", "character": "fox"}, None)
    check("good program: ok in one round, fences stripped", r["ok"] and r["code"].startswith("({") and len(r["attempts"]) == 1, str(r.get("error")))
    sys_prompt, user = seen[-1][2], seen[-1][1][0]["content"]
    check("prompt: toolkit doc as system, character + request as user", "SENSE" in sys_prompt and "Felix" in user and "THE USER'S STANDING REQUEST" in user)
    BC.call_model = fake([BAD_EVENT, GOOD])
    r = await coder.code({"id": "2", "instruction": "x", "name": "n", "character": "fox"}, None)
    check("bad event type: one repair round with the finding, then ok", r["ok"] and len(r["attempts"]) == 2 and "not an event type" in seen[-1][1][-1]["content"], str(r["attempts"]))
    BC.call_model = fake([UNCLAIMED, UNCLAIMED, GOOD])
    r = await coder.code({"id": "3", "instruction": "x", "name": "n", "character": "fox"}, None)
    check("still failing after the repair: the fallback model gets its turn", r["ok"] and r["model"] == "deepseek-v4.1-flash", str(r["attempts"]))
    BC.call_model = fake([BC.CoderError("gpt-5.4-mini: HTTP 500"), GOOD])
    r = await coder.code({"id": "4", "instruction": "x", "name": "n", "character": "bear"}, None)
    check("model error: fallback model", r["ok"] and r["model"] == "deepseek-v4.1-flash")
    BC.call_model = fake([GOOD])
    r = await coder.code({"id": "5", "instruction": "only the left ear", "name": "n", "character": "fox", "prev_code": GOOD, "prev_instruction": "wiggle"}, None)
    u = seen[-1][1][0]["content"]
    check("edit: prev code + its instruction + keep-state rule in the prompt", r["ok"] and "THE PROGRAM RUNNING NOW" in u and "api.state field" in u)
    BC.call_model = fake([BC.CoderError("a"), BC.CoderError("b")])
    r = await coder.code({"id": "6", "instruction": "x", "name": "n", "character": "fox"}, None)
    check("both models down: ok false with the reason", not r["ok"] and r["error"])
    r = await coder.code({"id": "7", "instruction": "", "name": "n"}, None)
    check("empty instruction refused", not r["ok"] and r["error"] == "empty instruction")
    check("extract_code: prose + fence", BC.extract_code("Here you go:\n```javascript\n({ per_frame(api) {} })\n```\nEnjoy") == "({ per_frame(api) {} })")
    await coder.checker.aclose()


asyncio.run(main())
print(f"test_behavior_coder: {'all passed' if not fails else str(fails) + ' failed'}")
sys.exit(fails)
