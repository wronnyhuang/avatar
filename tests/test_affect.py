"""app/affect.py — the reply's feeling from a (fake) door, parsed and sent once per reply."""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import affect  # noqa: E402


class FakeDoor:
    def __init__(self, text):
        self.text, self.calls = text, []

    def prewarm(self, *a, **k):
        pass

    async def complete(self, **kw):
        self.calls.append(kw)
        return {"done": True, "ok": True, "text": self.text}


class T(unittest.TestCase):
    def run_cue(self, text, user="My grandmother passed away.", reply="Oh no, I'm so sorry."):
        sent = []

        async def go():
            A = affect.Affect(send=sent.append, door=FakeDoor(text), enabled=True)
            A.cue(user, reply)
            await A.task
            return A
        A = asyncio.run(go())
        return A, sent

    def test_sad(self):
        A, sent = self.run_cue("sad 0.8 hand_on_heart")
        self.assertEqual(sent[0]["feel"], "sad")
        self.assertEqual(sent[0]["gesture"], "hand_on_heart")
        self.assertTrue(4 <= sent[0]["hold_s"] <= 12)
        self.assertIn("grandmother", A.door.calls[0]["user"])

    def test_neutral_sends_nothing(self):
        _, sent = self.run_cue("neutral 0.3 none")
        self.assertEqual(sent, [])

    def test_garbage_is_ignored(self):
        _, sent = self.run_cue("I think the character should look sad")
        self.assertEqual(sent, [])

    def test_stream_fires_once_per_reply(self):
        sent = []

        async def go():
            A = affect.Affect(send=sent.append, door=FakeDoor("excited 0.9 clap"), enabled=True)
            A.user_said("I got the job!")
            for w in ["Oh ", "my ", "gosh, ", "that's ", "amazing ", "news ", "for ", "you!"]:
                A.avatar_delta(w)
            await A.task
            if A.pre_task:
                await asyncio.gather(A.pre_task, return_exceptions=True)
            return A
        A = asyncio.run(go())
        # joy: one 'pre' judgement on his words + one on the reply's first words
        self.assertEqual(len(A.door.calls), 2)
        self.assertIn("not started", A.door.calls[0]["user"])
        self.assertEqual(sent[-1]["feel"], "excited")
        self.assertEqual(sent[-1]["stage"], "reply")

    # joy: the continuous verdict
    def test_dims(self):
        _, sent = self.run_cue("bashful pride | 0.6 0.2 -0.4 | 0.6 | none | look_down, tilt")
        j = sent[0]
        self.assertEqual(j["label"], "bashful pride")
        self.assertEqual((j["v"], j["a"], j["d"]), (0.6, 0.2, -0.4))
        self.assertEqual(j["facets"], ["look_down", "tilt"])
        self.assertIn(j["feel"], affect.FEELS)

    def test_dims_clamped_and_unknowns_dropped(self):
        _, sent = self.run_cue("Mock outrage | -0.4 0.9 0.7 | 1.0 | jazz_hands | nose, sparkle")
        j = sent[0]
        self.assertEqual(j["gesture"], "none")
        self.assertEqual(j["facets"], ["nose"])
        self.assertEqual(j["intensity"], 1.0)

    def test_plain_dims_send_nothing(self):
        _, sent = self.run_cue("attentive | 0.1 0 0 | 0.3 | none | none")
        self.assertEqual(sent, [])

    def test_pre_after_reply_dropped(self):
        sent = []

        class SlowDoor(FakeDoor):
            async def complete(self, **kw):
                self.calls.append(kw)
                if "not started" in kw["user"]:
                    await asyncio.sleep(0.05)
                    return {"done": True, "ok": True, "text": "happy | 0.7 0.3 0.2 | 0.6 | none | none"}
                return {"done": True, "ok": True, "text": "sad | -0.7 -0.4 -0.4 | 0.8 | none | none"}

        async def go():
            A = affect.Affect(send=sent.append, door=SlowDoor(""), enabled=True)
            A.user_said("my dog died")
            A.cue("my dog died", "Oh no, I'm so sorry.")
            await A.task
            await asyncio.gather(A.pre_task, return_exceptions=True)
        asyncio.run(go())
        self.assertEqual([x["label"] for x in sent], ["sad"])

    def test_heard_fires_pre_and_late_transcript_only_refines(self):
        sent = []

        async def go():
            A = affect.Affect(send=sent.append, door=FakeDoor("joy | 0.9 0.7 0.4 | 0.9 | clap | none"), enabled=True)
            A.heard("i got the job", True)               # the fast recognizer, before the reply
            await A.pre_task
            A.cue("i got the job", "Oh wow, congratulations!")
            await A.task
            A.user_said("I got the job!")                # the realtime transcript, late: context only
            A.heard("oh wow congratulations", True)      # its own reply leaking through: ignored
            A.heard("yes", True)                         # one word: ignored
            return A
        A = asyncio.run(go())
        self.assertEqual(len(A.door.calls), 2)
        self.assertEqual([x["stage"] for x in sent], ["pre", "reply"])
        self.assertEqual(A.user, "I got the job!")

    def test_parse_echoed_field_names(self):
        j = affect.parse("LABEL: sad, sympathetic | V A D: -0.7 -0.3 -0.4 | INTENSITY: 0.7 | GESTURE: none | FACETS: tilt, heavy_lids")
        self.assertEqual((j["label"], j["v"], j["intensity"], j["facets"]), ("sad, sympathetic", -0.7, 0.7, ["tilt", "heavy_lids"]))
        j = affect.parse("Good news.\n\nLABEL: joyful excitement | 0.9 0.8 0.5 | 0.9 | raise_both | laugh")
        self.assertEqual((j["label"], j["gesture"]), ("joyful excitement", "raise_both"))

    def test_heard_pieces_of_one_turn_judged_together(self):
        async def go():
            A = affect.Affect(send=lambda j: None, door=FakeDoor("amused | 0.7 0.5 0.3 | 0.7 | none | laugh"), enabled=True)
            A.heard("okay here's a joke", True)
            A.heard("they don't have the guts", True)
            await A.pre_task
            return A
        A = asyncio.run(go())
        self.assertIn("here's a joke they don't have the guts", A.door.calls[-1]["user"])
        self.assertEqual(A.turn, 1)


if __name__ == "__main__":
    unittest.main()
