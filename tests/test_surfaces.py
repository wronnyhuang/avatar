"""scene: app/surfaces.py (board + TV canvases, 'S' wire messages) and the stub renderer's 'S' branch.
    python3 tests/test_surfaces.py"""
import asyncio
import os
import struct
import subprocess
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP)
import surfaces as S  # noqa: E402
import wire  # noqa: E402


def mirror_of(msgs, shapes=None):
    shapes = shapes or {0: (512, 1024, 4), 1: (288, 512, 4)}
    m = {k: np.zeros(v, np.uint8) for k, v in shapes.items()}
    for b in msgs:
        d = S.unpack_msg(b)
        S.apply_msg(m[d["sid"]], d)
    return m


class Board(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.s = S.Surfaces(send=self.sent.append)
        self.B = self.s.get("board")

    def test_header_matches_wire(self):
        self.assertEqual(S.S_HDR.format, wire.S_HDR.format)
        self.assertEqual(S.S_HDR.size, 24)
        self.assertEqual(wire.S, 0x53)

    def test_dirty_rect_only_touched_box(self):
        before = self.B.snapshot()
        msgs = self.B.draw_polyline([[0.2, 0.3], [0.3, 0.35]], 0.02, t=1.0)
        self.assertEqual(len(msgs), 1)
        d = S.unpack_msg(msgs[0])
        after = self.B.snapshot()
        changed = np.argwhere(np.any(after != before, axis=2))
        self.assertTrue(len(changed) > 0)
        y0, x0 = changed.min(0)
        y1, x1 = changed.max(0) + 1
        self.assertTrue(d["x"] <= x0 and d["y"] <= y0 and d["x"] + d["w"] >= x1 and d["y"] + d["h"] >= y1)
        # the rect is tight-ish (capsule bbox + edge): within 4 px of the changed box
        self.assertLessEqual(x0 - d["x"], 4)
        self.assertLessEqual(d["x"] + d["w"] - x1, 4)
        np.testing.assert_array_equal(d["pixels"], after[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]])
        out = np.ones(after.shape[:2], bool)
        out[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]] = False
        self.assertFalse(np.any(after[out] != before[out]))
        # stroke width: 0.02 v = 10.24 px; the middle of the stroke is chalk
        cy, cx = int(0.325 * 512), int(0.25 * 1024)
        self.assertGreater(after[cy, cx, 3], 200)

    def test_clipped_at_edges(self):
        msgs = self.B.draw_polyline([[-0.1, -0.1], [0.05, 0.05]], 0.05)
        d = S.unpack_msg(msgs[0])
        self.assertEqual((d["x"], d["y"]), (0, 0))
        msgs = self.B.draw_polyline([[0.98, 0.98], [1.2, 1.2]], 0.05)
        d = S.unpack_msg(msgs[0])
        self.assertEqual(d["x"] + d["w"], 1024)
        self.assertEqual(d["y"] + d["h"], 512)
        self.assertEqual(self.B.draw_polyline([[1.5, 1.5], [1.6, 1.6]], 0.01), [])   # fully outside: no message
        m = self.B.erase_disc((1.0, 0.0), 0.1)
        d = S.unpack_msg(m[0])
        self.assertTrue(d["x"] + d["w"] == 1024 and d["y"] == 0)

    def test_soft_edge_and_straight_alpha_colour(self):
        self.B.draw_polyline([[0.1, 0.5], [0.9, 0.5]], 0.03, rgba=(200, 50, 50, 255))
        col = self.B.canvas[:, 512]
        a = col[:, 3]
        ys = np.flatnonzero(a)
        self.assertTrue(np.any((a > 0) & (a < 255)))                 # anti-aliased edge
        self.assertTrue(np.all(col[ys, :3] == (200, 50, 50)))        # straight alpha: colour not premultiplied
        self.B.draw_polyline([[0.5, 0.3], [0.5, 0.7]], 0.03, rgba=(50, 50, 200, 128))
        px = self.B.canvas[256, 512]
        self.assertEqual(px[3], 255)
        self.assertTrue(110 < px[0] < 140 and 110 < px[2] < 140)    # ~50/50 mix over an opaque red

    def test_codecs_round_trip(self):
        rng = np.random.default_rng(1)
        px = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
        b = S.pack_msg(0, 0, 2.5, 7, 3, 4, px)
        d = S.unpack_msg(b)
        self.assertEqual(d["codec"], S.CODEC_RAW)                    # noise: zlib not smaller
        np.testing.assert_array_equal(d["pixels"], px)
        self.assertEqual((d["sid"], d["t"], d["seq"], d["x"], d["y"], d["w"], d["h"]), (0, 2.5, 7, 3, 4, 53, 37))
        flat = np.zeros((40, 60, 4), np.uint8)
        flat[10:20] = (240, 240, 232, 235)
        d = S.unpack_msg(S.pack_msg(0, 0, None, 8, 0, 0, flat))
        self.assertEqual(d["codec"], S.CODEC_ZLIB)
        self.assertTrue(d["flags"] & S.F_NOW)
        self.assertEqual(d["t"], 0.0)
        np.testing.assert_array_equal(d["pixels"], flat)
        big = rng.integers(0, 256, (288, 512, 4), dtype=np.uint8)
        big[:, :, 3] = 255
        d = S.unpack_msg(S.pack_msg(1, 0, 1.0, 9, 0, 0, big, allow_jpeg=True))
        self.assertEqual(d["codec"], S.CODEC_JPEG)
        self.assertEqual(d["pixels"].shape, (288, 512, 4))
        self.assertTrue(np.all(d["pixels"][:, :, 3] == 255))
        d = S.unpack_msg(S.pack_msg(0, S.F_CLEAR, None, 10, 0, 0, None))
        self.assertEqual((d["w"], d["h"]), (0, 0))
        with self.assertRaises(ValueError):
            S.unpack_msg(b"\x41" + bytes(30))

    def test_timestamps_non_decreasing(self):
        ts = []
        for t in (5.0, 4.0, 6.0, None, 5.5):
            m = self.B.draw_polyline([[0.5, 0.5], [0.52, 0.5]], 0.01, t=t)
            d = S.unpack_msg(m[0])
            ts.append(d["t"])
            self.assertFalse(d["flags"] & S.F_NOW)
        self.assertEqual(ts, [5.0, 5.0, 6.0, 6.0, 6.0])
        seqs = [S.unpack_msg(b)["seq"] for b in self.sent]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))

    def test_now_before_any_stamp(self):
        d = S.unpack_msg(self.B.draw_polyline([[0.5, 0.5], [0.52, 0.5]], 0.01)[0])
        self.assertTrue(d["flags"] & S.F_NOW)

    def test_ts_per_segment(self):
        pts = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.3, 0.25]]
        msgs = self.B.draw_polyline(pts, 0.012, ts=[10.0, 10.1, 10.2, 10.35])
        self.assertEqual(len(msgs), 3)
        self.assertEqual([S.unpack_msg(m)["t"] for m in msgs], [10.1, 10.2, 10.35])
        with self.assertRaises(ValueError):
            self.B.draw_polyline(pts, 0.012, ts=[1.0, 2.0])

    def test_segment_cost(self):
        t0 = time.perf_counter()
        for i in range(100):
            self.B.draw_polyline([[0.1 + i * 0.005, 0.4], [0.105 + i * 0.005, 0.42]], 0.012, t=float(i))
        self.assertLess((time.perf_counter() - t0) / 100, 0.001)

    def test_erase_residue(self):
        self.B.draw_polyline([[0.1, 0.5], [0.9, 0.5]], 0.05)
        a0 = int(self.B.canvas[256, 512, 3])
        self.B.erase_disc((0.5, 0.5), 0.1, residue=0.08)
        self.assertAlmostEqual(int(self.B.canvas[256, 512, 3]), round(a0 * 0.08), delta=1)
        self.assertEqual(int(self.B.canvas[256, 150, 3]), a0)            # outside the disc
        self.B.erase_rect((0.0, 0.4), (0.3, 0.6), residue=0.0)
        self.assertEqual(int(self.B.canvas[256, 200, 3]), 0)
        self.assertEqual(int(self.B.canvas[256, 400, 3]), a0)

    def test_ink_grid(self):
        g = self.B.ink_grid()
        self.assertEqual(g.shape, (8, 16))
        self.assertEqual(g.max(), 0.0)
        self.B.draw_polyline([[0.02, 0.06], [0.04, 0.06]], 0.03)
        g = self.B.ink_grid()
        self.assertGreater(g[0, 0], 0.0)
        self.assertEqual(g[4:, 4:].max(), 0.0)

    def test_clear(self):
        self.B.draw_polyline([[0.1, 0.5], [0.9, 0.5]], 0.05)
        m = self.B.clear(t=3.0)
        d = S.unpack_msg(m[0])
        self.assertTrue(d["flags"] & S.F_CLEAR)
        self.assertEqual((d["w"], d["h"]), (0, 0))
        self.assertFalse(self.B.canvas.any())

    def test_mirror_equals_snapshot(self):
        self.B.draw_polyline([[0.1, 0.1], [0.3, 0.4], [0.5, 0.1]], 0.012, ts=[1, 2, 3])
        self.B.draw_polyline([[0.6, 0.6], [0.9, 0.9]], 0.02, rgba=(255, 200, 0, 200), t=4)
        self.B.erase_disc((0.3, 0.4), 0.04, t=5)
        self.B.erase_rect((0.55, 0.55), (0.7, 0.7), t=6, residue=0.2)
        self.B.clear(t=7)
        self.B.draw_polyline([[0.2, 0.2], [0.25, 0.3]], 0.012, t=8)
        T = self.s.get("tv")
        f = np.zeros((90, 160, 3), np.uint8)
        f[:, :80] = (200, 30, 30)
        T.put_frame(f, t=8.5)
        f[:, 80:] = (10, 200, 10)
        T.put_frame(f, t=9)
        m = mirror_of(self.sent)
        np.testing.assert_array_equal(m[0], self.B.snapshot())
        np.testing.assert_array_equal(m[1], T.snapshot())

    def test_resync_on_attach(self):
        s = S.Surfaces()
        s.get("board").draw_polyline([[0.1, 0.1], [0.4, 0.3]], 0.02, t=1)
        got = []
        msgs = s.attach(got.append)
        self.assertEqual(msgs, got)
        self.assertEqual(len(got), 2)
        d = {S.unpack_msg(b)["sid"]: S.unpack_msg(b) for b in got}
        self.assertTrue(d[0]["flags"] & S.F_RESYNC and d[0]["flags"] & S.F_NOW)
        self.assertEqual((d[0]["x"], d[0]["y"], d[0]["w"], d[0]["h"]), (0, 0, 1024, 512))
        self.assertTrue(d[1]["flags"] & S.F_CLEAR and (d[1]["w"], d[1]["h"]) == (0, 0))   # blank tv: a clear
        # a stale mirror (reused renderer) built only from these messages equals the snapshot
        stale = {0: np.full((512, 1024, 4), 77, np.uint8), 1: np.full((288, 512, 4), 99, np.uint8)}
        for b in got:
            x = S.unpack_msg(b)
            S.apply_msg(stale[x["sid"]], x)
        np.testing.assert_array_equal(stale[0], s.get("board").snapshot())
        np.testing.assert_array_equal(stale[1], s.get("tv").snapshot())
        s.detach()
        n = len(got)
        s.get("board").clear()
        self.assertEqual(len(got), n)                                    # detached: nothing sent


class Tv(unittest.TestCase):
    def test_put_frame_resize_alpha_and_clear(self):
        s = S.Surfaces()
        T = s.get("tv")
        f = np.zeros((720, 1280, 3), np.uint8)
        f[:, :640] = (255, 0, 0)
        msgs = T.put_frame(f)
        self.assertEqual(len(msgs), 1)
        snap = T.snapshot()
        self.assertEqual(snap.shape, (288, 512, 4))
        self.assertTrue(np.all(snap[:, :, 3] == 255))
        self.assertEqual(tuple(snap[100, 50, :3]), (255, 0, 0))
        self.assertEqual(T.put_frame(f), [])                            # unchanged frame: nothing to send
        f2 = f.copy()
        f2[:40, 1200:] = 255
        d = S.unpack_msg(T.put_frame(f2)[0])
        self.assertLess(d["w"] * d["h"], 512 * 288 / 4)                 # only the changed corner
        T.clear()
        self.assertFalse(T.snapshot()[:, :, 3].any())
        self.assertTrue(T.blank())

    def test_test_pattern_and_jpeg_path(self):
        p = S.test_pattern(0.3)
        self.assertEqual(p.shape, (288, 512, 3))
        T = S.Surfaces().get("tv")
        d = S.unpack_msg(T.put_frame(p)[0])
        self.assertIn(d["codec"], (S.CODEC_ZLIB, S.CODEC_RAW))         # flat illustration: zlib, under 48 KB
        rng = np.random.default_rng(0)
        d = S.unpack_msg(T.put_frame(rng.integers(0, 256, (288, 512, 3), dtype=np.uint8))[0])
        self.assertEqual(d["codec"], S.CODEC_JPEG)                      # photographic noise -> jpeg


class StubRenderSmoke(unittest.TestCase):
    def test_stub_accepts_surfaces(self):
        import aiohttp
        py = sys.executable
        proc = subprocess.Popen([py, os.path.join(APP, "stub_render.py"), "--port", "0", "--init-ms", "20",
                                 "--key", "k"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            line = proc.stdout.readline()
            self.assertTrue(line.startswith("READY port="), line)
            port = int(line.strip().split("=")[1])

            async def run():
                errs, frames, stats = [], [], []
                async with aiohttp.ClientSession() as http:
                    async with http.ws_connect(f"http://127.0.0.1:{port}/render?key=k") as ws:
                        await ws.send_json({"type": "start", "character": "muse", "width": 320, "height": 180,
                                            "fps": 24, "codec": "jpeg"})
                        while True:
                            m = await asyncio.wait_for(ws.receive(), 10)
                            if m.type == aiohttp.WSMsgType.TEXT and '"ready"' in m.data:
                                break
                        sf = S.Surfaces(send=None)
                        out = sf.attach(lambda b: None)
                        out += sf.get("board").draw_polyline([[0.1, 0.2], [0.4, 0.2], [0.4, 0.6]], 0.03,
                                                             ts=[0.0, 0.1, 0.2])
                        out += sf.get("tv").put_frame(S.test_pattern(0.1), t=0.15)
                        for b in out:
                            await ws.send_bytes(b)
                        s0 = 0
                        for _ in range(30):                              # 1.2 s of audio in 40 ms chunks
                            pcm = (np.sin(np.arange(960) / 10.0) * 3000).astype("<i2").tobytes()
                            await ws.send_bytes(struct.pack("<B3xQ", 0x41, s0) + pcm)
                            s0 += 960
                        t_end = time.monotonic() + 6
                        while time.monotonic() < t_end and (len(frames) < 20 or not stats):
                            try:
                                m = await asyncio.wait_for(ws.receive(), 2)
                            except asyncio.TimeoutError:
                                break
                            if m.type == aiohttp.WSMsgType.BINARY and m.data[0] == 0x56:
                                frames.append(m.data)
                            elif m.type == aiohttp.WSMsgType.TEXT:
                                import json
                                j = json.loads(m.data)
                                if j.get("type") == "error":
                                    errs.append(j)
                                elif j.get("type") == "stats":
                                    stats.append(j)
                return errs, frames, stats

            errs, frames, stats = asyncio.run(run())
            self.assertEqual(errs, [])
            self.assertGreaterEqual(len(frames), 20)
            st = [s.get("surfaces") for s in stats if s.get("surfaces")]
            self.assertTrue(st and st[-1]["msgs"] == 5 and st[-1]["applied"] == 5, st)
            # the last frame shows chalk on the board rect: decode and look for bright pixels in the left box
            import cv2
            img = cv2.imdecode(np.frombuffer(frames[-1][16:], np.uint8), cv2.IMREAD_COLOR)
            box = img[int(0.18 * 180):int(0.18 * 180) + 48, int(0.03 * 320):int(0.03 * 320) + 96]
            self.assertGreater(int(box.max()), 180)
        finally:
            proc.stdout.close()
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


class HubSide(unittest.TestCase):
    """app.py hooks: Session.surfaces_attach / playback_t and the key-guarded /debug/surface door (no network)."""

    def _session(self, app, character="muse"):
        s = app.Session.__new__(app.Session)
        s.surfaces = S.Surfaces()
        s._pt_rig = s._pt_pb = None
        s.pb = {"rig": 0}
        s.tl = app.Timeline(lambda *a: None, {"vendor_jb_ms": 60, "ahead_ms": 200, "hang_ms": 300})
        s.character = character
        s.ended = asyncio.Event()
        s.video_ok = True

        class Link:
            def __init__(self):
                self.got = []

            def alive(self):
                return True

            def send_bytes(self, b):
                self.got.append(b)

            def send_json(self, obj):
                pass
        s.link = Link()
        return s

    def test_attach_and_playback_t(self):
        os.environ.setdefault("AVATAR_HOME", __import__("tempfile").mkdtemp(prefix="avatar-surf-home-"))
        import app  # noqa: PLC0415

        async def go():
            s = self._session(app)
            self.assertEqual(s.playback_t(), 0.0)
            s.surfaces.get("board").draw_polyline([[0.1, 0.1], [0.2, 0.2]], 0.02)
            s.surfaces_attach()
            self.assertEqual(len(s.link.got), 2)                         # resync: board + tv
            s.on_rig({"t": 12.5, "a": [0.0]})
            self.assertAlmostEqual(s.playback_t(), 12.5, delta=0.05)
            s.character = "fox"
            s.surfaces_attach()                                          # other characters: detached
            s.surfaces.get("board").clear()
            self.assertEqual(len(s.link.got), 2)
        asyncio.run(go())

    def test_debug_route(self):
        os.environ.setdefault("AVATAR_HOME", __import__("tempfile").mkdtemp(prefix="avatar-surf-home-"))
        import app  # noqa: PLC0415
        from aiohttp.test_utils import TestClient, TestServer  # noqa: PLC0415
        import types  # noqa: PLC0415

        async def go():
            hub = types.SimpleNamespace(secret="s" * 24, session=None, cfg={"recording": False}, box=None)   # never the real recordings
            client = TestClient(TestServer(app.make_app(hub)))
            await client.start_server()
            try:
                H = {"x-requested-with": "avatar-live"}
                r = await client.post("/debug/surface?k=bad", json={"op": "clear"}, headers=H)
                self.assertEqual(r.status, 403)
                r = await client.post("/debug/surface?k=" + hub.secret, json={"op": "clear"}, headers=H)
                self.assertEqual(r.status, 404)
                s = self._session(app)
                s.surfaces_attach()
                hub.session = s
                n0 = len(s.link.got)
                r = await client.post("/debug/surface?k=" + hub.secret, headers=H, json={
                    "surface": "board", "op": "draw_polyline", "pts": [[0.1, 0.1], [0.3, 0.2]], "width": 0.012,
                    "t_rel": 0.5})
                j = await r.json()
                self.assertEqual(r.status, 200, j)
                self.assertEqual(j["msgs"], 1)
                self.assertAlmostEqual(S.unpack_msg(s.link.got[n0])["t"], 0.5, delta=0.01)
                r = await client.post("/debug/surface?k=" + hub.secret, headers=H, json={
                    "surface": "tv", "op": "put_frame_test", "phase": 0.2})
                self.assertEqual(r.status, 200)
                for body in ({"surface": "board", "op": "erase_disc", "center": [0.2, 0.15], "r": 0.05},
                             {"surface": "board", "op": "erase_rect", "uv0": [0, 0], "uv1": [0.1, 0.1]},
                             {"surface": "tv", "op": "clear", "t_rel": 1}):
                    r = await client.post("/debug/surface?k=" + hub.secret, headers=H, json=body)
                    self.assertEqual(r.status, 200, await r.text())
                r = await client.post("/debug/surface?k=" + hub.secret, headers=H, json={"op": "nope"})
                self.assertEqual(r.status, 400)
                r = await client.get("/debug/surface?surface=board&k=" + hub.secret)
                self.assertEqual(r.status, 200)
                self.assertEqual((await r.read())[:4], b"\x89PNG")
                mirror = mirror_of(s.link.got)
                np.testing.assert_array_equal(mirror[0], s.surfaces.get("board").snapshot())
            finally:
                await client.close()
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main(verbosity=1)
