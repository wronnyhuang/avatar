"""honest warm-up: providers.warm_eta once the usual start time is overrun, and the status message the Session relays
to the page. No network, no vendor, no server state.
    python3 tests/test_warm_eta.py      (needs aiohttp for the Session part)"""
import asyncio
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="avatar-warm-home-")   # never the live server's state/data
import providers  # noqa: E402


class WarmEta(unittest.TestCase):
    def test_within_the_estimate(self):
        w = providers.warm_eta(90, 30)
        self.assertEqual(w["eta_s"], 60)
        self.assertEqual((w["elapsed_s"], w["usual_s"]), (30, 90))

    def test_overrun_has_no_estimate(self):
        for el in (89.5, 90, 130, 373):
            w = providers.warm_eta(90, el)
            self.assertIsNone(w["eta_s"], el)                 # never the old floor of 10 s
            self.assertEqual(w["elapsed_s"], el)
            self.assertEqual(w["usual_s"], 90)

    def test_negative_clock_skew(self):
        self.assertEqual(providers.warm_eta(90, -5), {"eta_s": 90, "elapsed_s": 0, "usual_s": 90})


class SessionRelay(unittest.TestCase):
    """Session.status / _box_progress put elapsed_s + usual_s on the wire (the real methods, a bare object)."""

    def setUp(self):
        import app  # noqa: PLC0415
        self.app = app
        s = app.Session.__new__(app.Session)
        s.stage, s.status_log, s.t_mono, self.sent = "warming", [], time.monotonic(), []
        s.send = self.sent.append
        self.s = s

    def test_overrun_relay(self):
        self.s._box_progress({"state": "starting", "detail": "booting", **providers.warm_eta(90, 200.4)})
        self.assertEqual(self.sent[-1], {"type": "status", "stage": "warming", "detail": "booting", "eta_s": None,
                                         "elapsed_s": 200, "usual_s": 90})

    def test_within_relay(self):
        self.s._box_progress({"state": "starting", "detail": "", **providers.warm_eta(90, 30)})
        m = self.sent[-1]
        self.assertEqual((m["eta_s"], m["elapsed_s"], m["usual_s"], m["detail"]), (60, 30, 90, "starting the render box"))

    def test_other_status_unchanged(self):
        self.s.status("live", "fox is listening")
        self.assertEqual(self.sent[-1], {"type": "status", "stage": "live", "detail": "fox is listening", "eta_s": None})
        self.s.status("warming", "loading fox", 20.2)
        self.assertEqual(self.sent[-1], {"type": "status", "stage": "warming", "detail": "loading fox", "eta_s": 20})


if __name__ == "__main__":
    unittest.main(verbosity=1)
