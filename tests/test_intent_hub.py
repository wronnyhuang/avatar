"""Integration test of app/intent_hub.py with the real ASR worker (skipped when the vosk venv/model are absent).
python tests/test_intent_hub.py  — speaks "stick out your tongue" (espeak-ng, 24 kHz) into the hub's matcher."""
import asyncio, os, shutil, subprocess, sys, tempfile, time, unittest, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import intent_hub  # noqa: E402


def speak(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "a.wav")
        subprocess.run(["espeak-ng", "-s", "150", "-w", fn, text], check=True)
        out = os.path.join(d, "b.raw")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", fn, "-ar", "24000", "-ac", "1", "-f", "s16le", out], check=True)
        return open(out, "rb").read()


class HubASR(unittest.TestCase):
    @unittest.skipUnless(shutil.which("espeak-ng") and shutil.which("ffmpeg") and intent_hub.IntentHub(None, print).usable(),
                         "vosk venv/model or espeak-ng missing")
    def test_mic_to_intent(self):
        got = []
        h = intent_hub.IntentHub(None, lambda m: None)

        async def run():
            loop = asyncio.get_running_loop()
            self.assertTrue(h.attach(loop, got.append, {}))
            pcm = b"\0" * 9600 + speak("stick out your tongue") + b"\0" * 48000
            for i in range(0, len(pcm), 960 * 7):        # 140 ms batches of 20 ms chunks, real time
                for k in range(i, min(len(pcm), i + 960 * 7), 960):
                    h.feed(pcm[k:k + 960])
                await asyncio.sleep(0.14)
            await asyncio.sleep(1.0)
        asyncio.run(run())
        h.close()
        self.assertTrue(got, "no intent arrived")
        c = got[0]["cmd"]
        self.assertEqual((got[0]["type"], c["action"], c["name"]), ("intent", "pose", "stick_tongue_out"))


if __name__ == "__main__":
    unittest.main()
