"""Learned-reaction mode, server side: the planner worker process (app/react_worker.py, its own interpreter with
torch) and the per-session batching. app.py's event loop only ever awaits a pipe (never runs the planner).

The learned-reaction model and its code checkout are NOT shipped with this repo: with "root" empty (the default) the
mode reports itself unavailable and the page stays on the rule-based reactions.

config.json "react": {
   "mode": "rules" | "learned"     default the page uses when its URL has no ?react= (DEFAULT rules: nothing changes)
   "allow": true                   false = the server refuses learned sessions (the page stays on rules)
   "python": <python with numpy + torch; default this interpreter>, "root": <react model checkout>, "planner": "stub" | "delaycopy" | "bundle",
   "bundle": <runs/<run>/bundle>, "infer": <path of the model's react/infer.py>, "threads": 2, "plan_ms": 33, "commit_ahead": 0, "xfade": 3, "stub_delay_s": 0.5,
   "timeout_ms": 60            a batch the worker has not answered by then goes to the renderer with the LAST learned
                               values held (+ the page's reflex/command map; uncovered channels = the page's) — counted
                               as late; the worker still steps it. (The renderer draws timeline t only once audio up to
                               t + lookahead (100 ms) is in, so a rig vector up to ~60-90 ms late still makes its frame.)
   "blinks": "rules"           learned mode's blinks: "rules" = the page's rule-based blinks (log-normal life clock,
                               gaze-shift blinks, reflex + commanded blinks); "learned" = the planner's eye closure as
                               events. The page's ?react_blinks=rules|learned overrides it per session.
   "shown_every_s": 0.4        how often the merged vector goes back down to the page (for "hold that"; ~7 kbps)
}
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U32 = struct.Struct("<I")
DEFAULTS = {"mode": "rules", "allow": True, "python": "",
            "root": "", "planner": "stub", "bundle": "", "infer": "", "threads": 2,
            "plan_ms": 33, "commit_ahead": 0, "xfade": 3, "stub_delay_s": 0.5, "timeout_ms": 60, "shown_every_s": 0.4,
            "ear_springs": 1, "stub_mirror": 1, "dither_db": -70.0, "blinks": "rules"}


def frame(hdr: dict, blobs=()) -> bytes:
    blobs = list(blobs)
    hdr = dict(hdr); hdr["blobs"] = [len(b) for b in blobs]
    h = json.dumps(hdr, separators=(",", ":")).encode()
    body = b"".join(blobs)
    return U32.pack(4 + len(h) + len(body)) + U32.pack(len(h)) + h + body


class Worker:
    """One planner process per server (sessions are one at a time); restarted on demand if it dies."""

    def __init__(self, cfg: dict, log, logdir: str):
        self.cfg = cfg; self.log = log; self.logdir = logdir
        self.p: asyncio.subprocess.Process | None = None
        self.ready: asyncio.Future | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.nid = 0
        self.info: dict = {}
        self.starting: asyncio.Task | None = None
        self.failed_at = 0.0

    def alive(self) -> bool:
        return self.p is not None and self.p.returncode is None and self.ready is not None and self.ready.done() \
            and not self.ready.exception()

    async def ensure(self, timeout: float = 30.0) -> bool:
        if self.alive():
            return True
        if time.monotonic() - self.failed_at < 5:
            return False
        if self.starting is None or self.starting.done():
            self.starting = asyncio.ensure_future(self._start())
        try:
            await asyncio.wait_for(asyncio.shield(self.starting), timeout)
        except Exception as e:  # noqa: BLE001
            self.log(f"react: worker start failed: {e}")
            self.failed_at = time.monotonic()
            return False
        return self.alive()

    async def _start(self):
        c = self.cfg
        if not c.get("root") or not os.path.isdir(c["root"]):
            raise RuntimeError("learned-reaction model not installed (config react.root is empty or missing)")
        py = c.get("python") or sys.executable
        if not os.path.exists(py):
            raise RuntimeError(f"react.python not found: {py!r}")
        args = [py, "-I", os.path.join(HERE, "react_worker.py"), "--root", c["root"], "--planner", c["planner"],
                "--threads", str(int(c["threads"])), "--stub-delay", str(float(c["stub_delay_s"])),
                "--plan-ms", str(float(c["plan_ms"])), "--commit-ahead", str(int(c["commit_ahead"])),
                "--xfade", str(int(c["xfade"])), "--ear-springs", str(int(c.get("ear_springs", 1))),
                "--stub-mirror", str(int(c.get("stub_mirror", 1))), "--dither-db", str(float(c.get("dither_db", -70.0))),
                "--blinks", c.get("blinks") if c.get("blinks") in ("rules", "learned") else "rules"]
        if c.get("bundle"):
            args += ["--bundle", c["bundle"]]
        if c.get("infer"):
            args += ["--infer", c["infer"]]
        if c.get("trace"):
            args += ["--trace", c["trace"]]
        if c.get("rig_gain"):
            args += ["--rig-gain", json.dumps(c["rig_gain"])]
        if c.get("record_dir"):
            args += ["--record-dir", c["record_dir"]]
        os.makedirs(self.logdir, exist_ok=True)
        err = open(os.path.join(self.logdir, "react_worker.log"), "ab")
        env = dict(os.environ, OMP_NUM_THREADS=str(int(c["threads"])), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                      stderr=err, env=env, limit=1 << 24)
        err.close()
        self.ready = asyncio.get_running_loop().create_future()
        asyncio.ensure_future(self._reader(self.p, self.ready))
        self.info = await asyncio.wait_for(asyncio.shield(self.ready), 60)
        self.log(f"react: worker pid {self.p.pid} ready ({self.info.get('planner')}, retarget {self.info.get('retarget')})")

    async def _reader(self, p, ready):
        try:
            while True:
                n = U32.unpack(await p.stdout.readexactly(4))[0]
                data = await p.stdout.readexactly(n)
                hl = U32.unpack_from(data, 0)[0]
                hdr = json.loads(data[4:4 + hl].decode())
                body = data[4 + hl:]
                if hdr.get("op") == "ready":
                    if not ready.done():
                        ready.set_result(hdr)
                    continue
                f = self.pending.pop(hdr.get("id"), None)
                if f is not None and not f.done():
                    f.set_result((hdr, body))
        except (asyncio.IncompleteReadError, ConnectionError, ValueError) as e:
            if not ready.done():
                ready.set_exception(RuntimeError(f"worker exited before ready ({e.__class__.__name__})"))
            for f in self.pending.values():
                if not f.done():
                    f.set_exception(RuntimeError("react worker exited"))
            self.pending.clear()
            self.log(f"react: worker stdout closed (rc {p.returncode})")

    def send(self, hdr: dict, blobs=()) -> asyncio.Future:
        self.nid += 1
        hdr = dict(hdr, id=self.nid)
        f = asyncio.get_running_loop().create_future()
        self.pending[self.nid] = f
        try:
            self.p.stdin.write(frame(hdr, blobs))
        except (ConnectionError, AttributeError) as e:
            self.pending.pop(self.nid, None)
            f.set_exception(RuntimeError(f"react worker pipe: {e}"))
        return f

    async def close(self):
        if self.p and self.p.returncode is None:
            try:
                self.p.stdin.close()
                await asyncio.wait_for(self.p.wait(), 3)
            except Exception:  # noqa: BLE001
                try:
                    self.p.kill()
                except ProcessLookupError:
                    pass


class SessionRx:
    """Per session: collects the learned-mode uplink (partner 'H' frames, the user's mic, the avatar's own voice tap, mic
    sync) and turns the page's rig vectors of one batch into the merged vectors the renderer gets."""

    def __init__(self, worker: Worker, cfg: dict, log):
        self.w = worker; self.cfg = cfg; self.log = log
        self.human: list[tuple[float, int, bytes]] = []
        self.mic: list[tuple[int, int, bytes]] = []
        self.own: list[tuple[int, bytes]] = []
        self.sync: list[tuple[int, float]] = []
        self.sent_params: list[str] | None = None
        self.sent_char = None
        self.need_reset = True
        self.m = collections.Counter()
        self.lat = collections.deque(maxlen=3000)     # worker round trip per batch (ms)
        self.h_bytes = 0
        self.last_shown_at = 0.0
        self.last_stats_at = 0.0
        self.last_stats: dict = {}
        self.t0 = time.monotonic()
        self.lock = asyncio.Lock()
        self.last_lv: np.ndarray | None = None        # the newest learned rig vector (NaN = channel not covered)
        self.blinks = cfg.get("blinks") if cfg.get("blinks") in ("rules", "learned") else "rules"

    # uplink pieces (called from api_up / the playback reorder buffer)
    def add_human(self, t: float, motion: np.ndarray, bits: int, nbytes: int):
        self.human.append((float(t), int(bits), np.asarray(motion, np.float32).tobytes()))
        self.h_bytes += nbytes
        self.m["human"] += 1
        if len(self.human) > 200:
            del self.human[:100]

    def add_mic(self, seq: int, pcm: bytes, rate: int):
        self.mic.append((int(seq), int(rate), pcm))
        if len(self.mic) > 400:
            del self.mic[:200]

    def add_own(self, s0: int, pcm: bytes):
        self.own.append((int(s0), pcm))
        if len(self.own) > 400:
            del self.own[:200]

    def add_sync(self, seq, t):
        try:
            self.sync.append((int(seq), float(t)))
        except (TypeError, ValueError):
            pass

    async def step(self, ticks, character: str, rig_params: list[str], want_stats=False):
        """ticks: [(t, a list/array, ab [(i, alpha, beta)] | None, rb)] -> list of merged vectors (None = use the page's)."""
        if not self.w.alive():
            self.m["worker_down"] += 1
            asyncio.ensure_future(self.w.ensure())
            return None
        async with self.lock:                         # batches in order (lanes may overlap)
            if self.need_reset:                       # once per session; a character switch only changes the retarget
                fut = self.w.send({"op": "reset", "character": character, "rig_params": rig_params, "blinks": self.blinks})
                try:
                    await asyncio.wait_for(fut, 5)
                except Exception as e:  # noqa: BLE001
                    self.log(f"react: reset failed: {e}")
                    return None
                self.need_reset = False; self.sent_char = character; self.sent_params = list(rig_params)
            hdr = {"op": "step", "character": character, "blinks": self.blinks}
            if self.sent_params != rig_params:
                hdr["rig_params"] = rig_params; self.sent_params = list(rig_params)
            blobs = []
            hdr["human"] = [[t, bits] for t, bits, _ in self.human]
            blobs += [b for _, _, b in self.human]
            hdr["sync"] = self.sync
            hdr["mic"] = [[seq, rate] for seq, rate, _ in self.mic]
            blobs += [b for _, _, b in self.mic]
            hdr["own"] = [s0 for s0, _ in self.own]
            blobs += [b for _, b in self.own]
            hdr["ticks"] = []
            for t, a, ab, rb in ticks:
                ab = ab or []
                hdr["ticks"].append([t, len(ab), rb])
                blobs.append(np.asarray(a, np.float32).tobytes())
                blobs.append(np.asarray(ab, np.float32).reshape(-1, 3).tobytes() if ab else b"")
            if want_stats:
                hdr["want_stats"] = True
            self.human, self.mic, self.own, self.sync = [], [], [], []
            t0 = time.perf_counter()
            fut = self.w.send(hdr, blobs)
            try:
                rh, body = await asyncio.wait_for(asyncio.shield(fut), self.cfg["timeout_ms"] / 1000)
            except asyncio.TimeoutError:
                self.m["late_batches"] += 1
                fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
                return self.hold(ticks)
            except Exception as e:  # noqa: BLE001
                self.m["errors"] += 1
                self.log(f"react: step failed: {e}")
                return None
            self.lat.append((time.perf_counter() - t0) * 1000)
            if rh.get("error"):
                self.m["errors"] += 1
                if self.m["errors"] < 5:
                    self.log("react: worker error: " + str(rh["error"])[:200])
                return None
            if rh.get("stats"):
                self.last_stats = rh["stats"]
            n = int(rh.get("n") or 0)
            if n != len(ticks) or not n:
                return None
            k = len(ticks[0][1])
            arr = np.frombuffer(body[:4 * n * k], np.float32).reshape(n, k)
            if len(body) >= 4 * (n + 1) * k:
                self.last_lv = np.frombuffer(body[4 * n * k:4 * (n + 1) * k], np.float32).copy()
            self.m["ticks"] += n
            return [arr[i] for i in range(n)]

    def hold(self, ticks):
        """worker late: the last learned values held, with each tick's own page map (reflexes, commands); channels the
        learned side does not cover stay the page's. None (= the page's vectors) before the first learned answer."""
        lv = self.last_lv
        if lv is None:
            return None
        out = []
        names = self.sent_params or []
        for t, a, ab, rb in ticks:
            a = np.asarray(a, np.float32)
            if len(a) != len(lv):
                return None
            f = a.copy()
            cov = np.isfinite(lv)
            x = lv.copy()
            for i, nm in enumerate(names[:len(x)]):
                if nm in ("blink_l", "blink_r") and cov[i]:
                    x[i] = 0.0 if rb is None else max(0.0, float(rb))     # never freeze a blink half-way
            al = np.ones(len(a), np.float32); be = np.zeros(len(a), np.float32)
            for i, a_, b_ in (ab or []):
                if 0 <= int(i) < len(a):
                    al[int(i)] = a_; be[int(i)] = b_
            f[cov] = al[cov] * x[cov] + be[cov]
            out.append(f)
        self.m["held_ticks"] += len(out)
        return out

    def summary(self) -> dict:
        lat = sorted(self.lat)
        q = (lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 2)) if lat else (lambda p: None)
        dt = max(1e-3, time.monotonic() - self.t0)
        return {"batches_ms": {"n": len(lat), "p50": q(.5), "p95": q(.95), "max": q(1.0)}, **dict(self.m),
                "h_kbps": round(self.h_bytes * 8 / dt / 1000, 2), "worker": self.last_stats, "blinks": self.blinks,
                "planner": self.w.info.get("planner"), "retarget": self.w.info.get("retarget")}
