"""Unified mode, server side: the unified worker process (app/unified_worker.py, its own interpreter) and the per-session
batching. Reuses learned mode's machinery (react_link.Worker / SessionRx: same framing, same timeout + hold policy, same
page affine map); adds the page's RAW MediaPipe results ('U', wire.py) instead of the 'H' canonical frame.
Learned mode and rules mode do not touch any of this (app.py routes here only for react mode "unified").

config.json "react": {..., "mode": "rules" | "learned" | "unified",
   "unified": {                    defaults below (paths relative to the repo root); any key overrides
     "allow": true, "python": <python with numpy + onnxruntime (or torch); "" = this interpreter>,
     "roots": [<dirs holding canon1/, unified/, retarget/>; default motion-model/],
     "fit_root": <dir whose canon1/ has fit.py>, "fit_params": "",
     "bundle": <model bundle dir; default models/unified-v2, "" = copy-delay stub>,
     "threads": 2, "planner_kw": {"replan_every": 3, "steps": 6, "guidance": 1.5, "temperature": 0.8},
     "blinks": "model" | "rules", "timeout_ms": 60, "shown_every_s": 0.4, "dither_db": -70.0, "trace": ""}}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import react_link

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_MM = os.path.join(REPO, "motion-model")
_BUNDLE = os.path.join(REPO, "models", "unified-v2")
DEFAULTS = {"allow": True, "python": "",
            "roots": [_MM], "fit_root": _MM, "fit_params": "",
            "bundle": _BUNDLE if os.path.isfile(os.path.join(_BUNDLE, "config.json")) else "", "threads": 2,
            "planner_kw": {"replan_every": 3, "steps": 6, "guidance": 1.5, "temperature": 0.8},
            "blinks": "model", "timeout_ms": 60, "shown_every_s": 0.4, "dither_db": -70.0, "trace": ""}


STYLE_KEYS = ("energy", "mirroring", "expressiveness")


def clean_style(m: dict) -> dict:
    """The {type:'style'} control's keys, each a number clamped to 0..1 or None (= automatic / infer);
    absent keys and anything else are left out"""
    out = {}
    for k in STYLE_KEYS:
        if not isinstance(m, dict) or k not in m:
            continue
        v = m[k]
        if v is None or v == "auto":
            out[k] = None
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            out[k] = None if v < 0 else round(min(1.0, float(v)), 4)
    return out


def config(react_cfg: dict | None) -> dict:
    rc = react_cfg or {}
    u = dict(DEFAULTS)
    u.update(rc.get("unified") or {})
    return u


class UnifiedWorker(react_link.Worker):
    """One unified worker process per server (sessions are one at a time); restarted on demand if it dies."""

    async def _start(self):
        import asyncio
        c = self.cfg
        py = c.get("python") or sys.executable
        if not os.path.exists(py):
            raise RuntimeError(f"react.unified.python not found: {py!r}")
        roots = c.get("roots") or []
        args = [py, "-I", os.path.join(HERE, "unified_worker.py"), "--roots", os.pathsep.join(roots),
                "--fit-root", c.get("fit_root") or "", "--threads", str(int(c.get("threads", 2))),
                "--blinks", c.get("blinks") if c.get("blinks") in ("model", "rules") else "model",
                "--dither-db", str(float(c.get("dither_db", -70.0)))]
        if c.get("bundle"):
            args += ["--bundle", c["bundle"]]
        if c.get("fit_params"):
            args += ["--fit-params", c["fit_params"]]
        if c.get("planner_kw"):
            args += ["--planner-kw", json.dumps(c["planner_kw"])]
        if c.get("trace"):
            args += ["--trace", c["trace"]]
        os.makedirs(self.logdir, exist_ok=True)
        err = open(os.path.join(self.logdir, "unified_worker.log"), "ab")
        th = str(int(c.get("threads", 2)))
        env = dict(os.environ, OMP_NUM_THREADS=th, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                      stderr=err, env=env, limit=1 << 24)
        err.close()
        self.ready = asyncio.get_running_loop().create_future()
        asyncio.ensure_future(self._reader(self.p, self.ready))
        self.info = await asyncio.wait_for(asyncio.shield(self.ready), 90)
        self.log(f"react: unified worker pid {self.p.pid} ready ({self.info.get('planner')}, fitter "
                 f"{self.info.get('fitter')})")


class UnifiedRx(react_link.SessionRx):
    """Per session: the unified-mode uplink (raw MediaPipe 'U' messages, the user's mic, the avatar's voice tap, mic sync) and
    the merge of the page's rig vectors (react_link.SessionRx.step's protocol, with "mp" instead of "human")."""

    mode = "unified"

    def __init__(self, worker, cfg, log):
        super().__init__(worker, cfg, log)
        self.mp: list[bytes] = []
        self.mp_bytes = 0
        self.blinks = cfg.get("blinks") if cfg.get("blinks") in ("model", "rules") else "model"
        self.style: dict = {}              # the session's style (sent again after a worker reset)
        self.style_dirty = False

    def set_style(self, st: dict) -> None:
        """Merged into the session style; rides the next step header as {"style": {...}}"""
        st = clean_style(st)
        if st:
            self.style.update(st)
            self.style_dirty = True
            self.m["style_changes"] += 1

    def add_mp(self, msg: bytes):
        self.mp.append(bytes(msg))
        self.mp_bytes += len(msg)
        self.m["mp"] += 1
        if len(self.mp) > 400:
            del self.mp[:200]

    def add_human(self, *a, **k):          # learned mode's 'H': not used here
        pass

    async def step(self, ticks, character: str, rig_params: list[str], want_stats=False):
        import asyncio
        import time
        if not self.w.alive():
            self.m["worker_down"] += 1
            asyncio.ensure_future(self.w.ensure())
            return None
        async with self.lock:
            if self.need_reset:
                fut = self.w.send({"op": "reset", "character": character, "rig_params": rig_params, "blinks": self.blinks})
                try:
                    await asyncio.wait_for(fut, 5)
                except Exception as e:  # noqa: BLE001
                    self.log(f"react: unified reset failed: {e}")
                    return None
                self.need_reset = False; self.sent_char = character; self.sent_params = list(rig_params)
                self.style_dirty = bool(self.style)                # a reset worker starts without the style
            hdr = {"op": "step", "character": character}
            if self.style_dirty:                                   # set_style -> the planner's style vector
                hdr["style"] = dict(self.style)
                self.style_dirty = False
            if self.sent_params != rig_params:
                hdr["rig_params"] = rig_params; self.sent_params = list(rig_params)
            blobs = list(self.mp)
            hdr["mp"] = [0] * len(self.mp)
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
            self.mp, self.mic, self.own, self.sync = [], [], [], []
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
                self.log(f"react: unified step failed: {e}")
                return None
            self.lat.append((time.perf_counter() - t0) * 1000)
            if rh.get("error"):
                self.m["errors"] += 1
                if self.m["errors"] < 5:
                    self.log("react: unified worker error: " + str(rh["error"])[:200])
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

    def summary(self) -> dict:
        s = super().summary()
        import time
        dt = max(1e-3, time.monotonic() - self.t0)
        s["mode"] = "unified"
        s["mp_kbps"] = round(self.mp_bytes * 8 / dt / 1000, 2)
        return s
