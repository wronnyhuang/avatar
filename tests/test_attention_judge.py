"""attention judge unit tests (mocked vendor; no network, no aiohttp needed).

    python3 tests/test_attention_judge.py
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import attention_judge as aj  # noqa: E402

IMG = "data:image/jpeg;base64," + "A" * 2000


def cand(i, kinds=("new_object",), speaking=False):
    return {"type": "attention_candidate", "id": f"at{i}", "t": 10.0 + i, "kinds": list(kinds),
            "details": {"new_object": {"what": "held_up", "side": "r"}}, "images": [IMG, IMG],
            "summary": {"speaking": speaking, "avatar_speaking": False, "face": True, "attending": True, "expression": "happy"}}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Vendor:
    def __init__(self, verdict=None, delay=0.0, cost=0.0004, raw=None):
        self.verdict = verdict or {"what": "he holds up a white mug", "worth_reply": True, "salience": 0.7, "reason": "showing it"}
        self.delay, self.cost, self.raw = delay, cost, raw
        self.payloads = []

    async def __call__(self, payload, timeout_s):
        self.payloads.append(payload)
        if self.delay:
            await asyncio.sleep(self.delay)
        content = self.raw if self.raw is not None else json.dumps(self.verdict)
        return {"model": payload["model"], "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 40, "cost": self.cost}}


def run(coro):
    return asyncio.run(coro)


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def judge(self, vendor=None, clock=None, **cfg):
        usd = []
        j = aj.AttentionJudge(self.home, self.logs.append, on_usd=usd.append, cfg=cfg, vendor=vendor or Vendor(),
                              clock=clock or FakeClock())
        j._usd = usd
        return j

    def test_verdict_roundtrip_and_cost(self):
        j = self.judge()
        r = run(j.handle(cand(1)))
        self.assertEqual(r["type"], "attention")
        self.assertEqual(r["id"], "at1")
        self.assertTrue(r["worth_reply"])
        self.assertEqual(r["what"], "he holds up a white mug")
        self.assertAlmostEqual(r["usd"], 0.0004)
        self.assertEqual(j._usd, [0.0004])
        self.assertNotIn("skipped", r)
        self.assertEqual(r["kinds"], ["new_object"])
        p = j.vendor.payloads[0]
        self.assertEqual(p["response_format"], {"type": "json_object"})
        self.assertEqual(sum(1 for c in p["messages"][1]["content"] if c["type"] == "image_url"), 2)
        self.assertIsNone(run(j.handle({"type": "something_else"})))

    def test_rules_persist_and_feed_prompt(self):
        j = self.judge()
        out = j.handle_rule({"type": "attention_rule", "op": "add", "text": "Tell me if my cat walks in"})
        self.assertEqual(out["type"], "attention_rules")
        self.assertEqual(len(out["rules"]), 1)
        self.assertEqual(out["rules"][0]["source"], "user")
        j.handle_rule({"op": "add", "text": "tell me if my cat walks in"})          # duplicate (case-insensitive) ignored
        self.assertEqual(len(j.rules()), 1)
        path = os.path.join(self.home, "state", "attention.json")
        self.assertTrue(os.path.exists(path))
        j2 = self.judge()                                                          # a new hub process
        self.assertEqual([r["text"] for r in j2.rules()], ["Tell me if my cat walks in"])
        run(j2.handle(cand(1)))
        text = j2.vendor.payloads[0]["messages"][1]["content"][0]["text"]
        self.assertIn("Tell me if my cat walks in", text)
        self.assertIn("standing instructions", text)

    def test_feedback_becomes_rule_tied_to_verdict(self):
        clk = FakeClock()
        j = self.judge(clock=clk)
        run(j.handle(cand(1)))
        out = j.handle_rule({"op": "feedback", "text": "don't comment on my coffee", "id": "at1"})
        r = out["rules"][-1]
        self.assertEqual(r["source"], "feedback")
        self.assertEqual(r["about"], "he holds up a white mug")
        self.assertEqual(r["kinds"], ["new_object"])
        self.assertEqual(r["verdict"], "at1")
        # no id -> the latest verdict
        clk.t += 10
        run(j.handle(cand(2)))
        out = j.handle_rule({"op": "feedback", "text": "that was fine, keep doing it"})
        self.assertEqual(out["rules"][-1]["verdict"], "at2")
        # the prompt quotes what the feedback was about, and the recent notes
        clk.t += 10
        run(j.handle(cand(3)))
        text = j.vendor.payloads[-1]["messages"][1]["content"][0]["text"]
        self.assertIn('said after you noted "he holds up a white mug"', text)
        self.assertIn("Your recent notes", text)
        self.assertIn("you spoke up", text)

    def test_remove_and_clear(self):
        j = self.judge()
        j.handle_rule({"op": "add", "text": "a"})
        b = j.handle_rule({"op": "add", "text": "b"})["rules"][-1]["id"]
        self.assertEqual([r["text"] for r in j.handle_rule({"op": "remove", "id": b})["rules"]], ["a"])
        self.assertEqual(j.handle_rule({"op": "clear"})["rules"], [])
        self.assertEqual(self.judge().rules(), [])

    def test_rate_limit(self):
        clk = FakeClock()
        j = self.judge(clock=clk, min_gap_s=2.5, max_per_min=3)
        self.assertNotIn("skipped", run(j.handle(cand(1))))
        clk.t += 1.0
        r = run(j.handle(cand(2)))
        self.assertEqual(r["skipped"], "rate")
        self.assertFalse(r["worth_reply"])
        clk.t += 2.0
        self.assertNotIn("skipped", run(j.handle(cand(3))))
        clk.t += 3.0
        self.assertNotIn("skipped", run(j.handle(cand(4))))
        clk.t += 3.0
        self.assertEqual(run(j.handle(cand(5)))["skipped"], "rate")                # 3 calls in the last minute
        clk.t += 61
        self.assertNotIn("skipped", run(j.handle(cand(6))))
        self.assertEqual(len(j.vendor.payloads), 4)

    def test_session_and_day_caps(self):
        clk = FakeClock()
        j = self.judge(clock=clk, session_cap_usd=0.001, day_cap_usd=1.0, vendor=Vendor(cost=0.0008))
        self.assertNotIn("skipped", run(j.handle(cand(1))))
        clk.t += 5
        self.assertNotIn("skipped", run(j.handle(cand(2))))                        # 0.0008 < 0.001
        clk.t += 5
        self.assertEqual(run(j.handle(cand(3)))["skipped"], "cap")                 # 0.0016 >= 0.001
        j.reset_session()
        clk.t += 5
        self.assertNotIn("skipped", run(j.handle(cand(4))))
        # the day cap is persisted across instances
        j2 = self.judge(clock=clk, day_cap_usd=0.002, vendor=Vendor(cost=0.0008))
        self.assertAlmostEqual(j2.stats()["day_usd"], 0.0024, places=6)
        self.assertEqual(run(j2.handle(cand(5)))["skipped"], "cap")

    def test_timeout_default_is_silent(self):
        j = self.judge(vendor=Vendor(delay=1.0), timeout_s=0.2)
        r = run(j.handle(cand(1)))
        self.assertEqual(r["skipped"], "timeout")
        self.assertFalse(r["worth_reply"])
        self.assertGreater(r["usd"], 0)                                            # estimated (probably billed)
        self.assertLess(r["ms"], 1500)

    def test_vendor_error_and_bad_json(self):
        async def boom(payload, timeout_s):
            raise RuntimeError("HTTP 402: insufficient credits")
        j = self.judge(vendor=boom)
        r = run(j.handle(cand(1)))
        self.assertEqual(r["skipped"], "error")
        self.assertFalse(r["worth_reply"])
        clk = FakeClock()
        j = self.judge(vendor=Vendor(raw="I think this is a mug."), clock=clk)
        r = run(j.handle(cand(1)))
        self.assertEqual(r["skipped"], "error")
        self.assertFalse(r["worth_reply"])

    def test_invalid_images(self):
        j = self.judge()
        m = cand(1)
        m["images"] = ["http://example.com/x.jpg", "data:image/jpeg;base64," + "A" * 200000]
        r = run(j.handle(m))
        self.assertEqual(r["skipped"], "invalid")
        self.assertEqual(j.vendor.payloads, [])

    def test_disabled(self):
        j = self.judge(enabled=False)
        self.assertEqual(run(j.handle(cand(1)))["skipped"], "off")

    def test_parse_verdict(self):
        v = aj.parse_verdict('```json\n{"what": "a woman walks into the room from the right side carrying a big basket of laundry today",'
                             ' "worth_reply": "true", "salience": 1.7, "reason": "new person"}\n```')
        self.assertEqual(len(v["what"].split()), 12)
        self.assertTrue(v["worth_reply"])
        self.assertEqual(v["salience"], 1.0)
        self.assertIsNone(aj.parse_verdict("no json here"))
        self.assertFalse(aj.parse_verdict('{"what": "x", "worth_reply": "no"}')["worth_reply"])

    def test_openai_payload(self):
        j = self.judge(provider="openai", model="gpt-5.4-nano", reasoning="minimal")
        p = j.payload(j.build_messages(cand(1), [IMG]))
        self.assertEqual(p["reasoning_effort"], "minimal")
        self.assertIn("max_completion_tokens", p)
        self.assertNotIn("usage", p)
        j = self.judge(reasoning={"effort": "minimal"})
        p = j.payload(j.build_messages(cand(1), [IMG]))
        self.assertEqual(p["reasoning"], {"effort": "minimal"})
        self.assertEqual(p["usage"], {"include": True})


if __name__ == "__main__":
    unittest.main(verbosity=1)
