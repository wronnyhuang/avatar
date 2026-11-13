"""Where the render server runs: a local child process, and its lifecycle (start on demand, stop when idle).

Provider interface (async, each returns quickly):
    create_or_start()  start the render server (idempotent)
    stop()             stop it (idempotent)
    status() -> dict   {"state": "stopped"|"starting"|"running"|"error", "detail": str, "eta_s": float|None,
                        "usd_per_h": float}
    endpoint() -> dict {"ws_url": "ws://…/render", "health_url": "http://…/health", "key": "<RENDER_KEY>"}

`local` = renderer/avatar/render_server.py as a child process of the app server (or app/stub_render.py when the
renderer is not there / config local.force_stub). The Box keeps a persisted record (state/box.json) so the idle stop
survives a server restart.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import socket
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RENDER_ROOT = os.path.join(os.path.dirname(HERE), "renderer")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pid_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def warm_eta(usual_s: float, elapsed_s: float) -> dict:
    """The honest warm-up estimate. Within the usual start time: eta_s = what is left of it. Once the usual
    time is used up there is NO estimate (eta_s None — never a floor that reads "about 10 s" for minutes); the page
    shows elapsed_s against usual_s instead."""
    elapsed = max(0.0, float(elapsed_s))
    left = float(usual_s) - elapsed
    return {"eta_s": left if left >= 1.0 else None, "elapsed_s": round(elapsed, 1), "usual_s": float(usual_s)}


class LocalProvider:
    name = "local"

    def __init__(self, cfg: dict, state: dict, log):
        self.cfg, self.state, self.log = cfg, state, log
        self.proc: asyncio.subprocess.Process | None = None
        self.port = 0
        self.key = ""
        self.started_at = 0.0
        self.ready_at = 0.0
        self.err = ""
        self.task: asyncio.Task | None = None
        self.which = ""

    def _cmd(self) -> tuple[list[str], str, str]:
        """The real render server (python -m avatar.render_server from render_root, default the repo's renderer/) when
        it exists, else the stub (app/stub_render.py: a test pattern, no GPU needed)."""
        c = self.cfg
        root = c.get("render_root") or DEFAULT_RENDER_ROOT
        real = os.path.join(root, "avatar", "render_server.py") if root else ""
        use_stub = c.get("force_stub") or not (real and os.path.exists(real))
        if use_stub:
            py = c.get("stub_python") or sys.executable
            argv = [py, os.path.join(HERE, "stub_render.py"), "--host", "127.0.0.1", "--port", str(self.port)] + \
                [str(a) for a in (c.get("stub_args") or [])]
            return argv, HERE, "stub"
        py = c.get("python") or sys.executable
        tmpl = c.get("args") or ["--host", "127.0.0.1", "--port", "{port}", "--profile", "local"]
        argv = [py, "-m", "avatar.render_server"] + [str(a).replace("{port}", str(self.port)) for a in tmpl]
        if c.get("rt_tree"):
            argv += ["--rt-tree", c["rt_tree"]]
        lock = c.get("gpu_lock")
        if lock:
            argv = ["flock", "-w", str(int(c.get("gpu_lock_wait_s", 60))), lock] + argv
        return argv, root, "render server"

    async def create_or_start(self) -> None:
        if self.proc and self.proc.returncode is None:
            return
        self.port = _free_port()
        self.key = secrets.token_urlsafe(18)
        argv, cwd, self.which = self._cmd()
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (self.cfg.get("env") or {}).items()})
        env["RENDER_KEY"] = self.key
        env.pop("OPENAI_API_KEY", None)          # the render child needs no vendor key
        logf = open(os.path.join(self.cfg["log_dir"], "render-child.log"), "ab")
        self.started_at = time.time()
        self.ready_at = 0.0
        self.err = ""
        self.proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
                                                         stdout=logf, stderr=logf, start_new_session=True)
        logf.close()
        self.state["local_pid"] = self.proc.pid
        self.state["local_cmd"] = os.path.basename(argv[min(len(argv) - 1, 1)])
        self.log(f"local render child started: {self.which} pid {self.proc.pid} port {self.port}")
        self.task = asyncio.create_task(self._wait_ready())

    async def _wait_ready(self) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + float(self.cfg.get("start_timeout_s", 180))
        async with aiohttp.ClientSession() as http:
            while time.monotonic() < deadline:
                if self.proc is None or self.proc.returncode is not None:
                    self.err = f"render child exited ({None if self.proc is None else self.proc.returncode}); see data/render-child.log"
                    return
                try:
                    async with http.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                        if r.status == 200 and (await r.json(content_type=None)).get("ok"):
                            self.ready_at = time.time()
                            self.log(f"local render child ready in {self.ready_at - self.started_at:.1f} s")
                            return
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(0.25)
        self.err = "render child did not answer /health in time"

    async def stop(self) -> None:
        p = self.proc
        self.proc = None
        if self.task and not self.task.done():
            self.task.cancel()
        if p and p.returncode is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)           # its own session: the child and its helpers (ffmpeg)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(p.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.log(f"local render child pid {p.pid} stopped")
        self.state.pop("local_pid", None)

    def status(self) -> dict:
        p = self.proc
        if p is None:
            return {"state": "error" if self.err else "stopped", "detail": self.err, "eta_s": None, "usd_per_h": 0.0}
        if p.returncode is not None:
            return {"state": "error", "detail": self.err or f"render child exited ({p.returncode})", "eta_s": None,
                    "usd_per_h": 0.0}
        if self.err:
            return {"state": "error", "detail": self.err, "eta_s": None, "usd_per_h": 0.0}
        if not self.ready_at:
            est = float(self.cfg.get("eta_s") or (4 if self.which == "stub" else 20))
            return {"state": "starting", "detail": f"starting the {self.which} on this machine",
                    **warm_eta(est, time.time() - self.started_at), "usd_per_h": 0.0}
        return {"state": "running", "detail": f"{self.which} on this machine", "eta_s": 0, "usd_per_h": 0.0,
                "which": self.which}

    def endpoint(self) -> dict:
        return {"ws_url": f"ws://127.0.0.1:{self.port}/render", "health_url": f"http://127.0.0.1:{self.port}/health",
                "key": self.key}

    @staticmethod
    def reap_orphan(state: dict, log) -> None:
        """A previous service instance's child that outlived it (dev runs; under systemd the cgroup takes it)."""
        pid = state.pop("local_pid", None)
        if not pid:
            return
        cl = _pid_cmdline(int(pid))
        if cl and ("avatar.render_server" in cl or "stub_render" in cl):
            try:
                os.killpg(int(pid), signal.SIGTERM)
                log(f"stopped an orphaned render child pid {pid} left by a previous run")
            except (ProcessLookupError, PermissionError):
                pass


class Box:
    """Lifecycle of the local render server: start on demand, stop after idle_s with no activity; persisted in
    state/box.json."""

    def __init__(self, cfg: dict, state_dir: str, log):
        self.cfg, self.log = cfg, log
        self.path = os.path.join(state_dir, "box.json")
        self.idle_s = float(cfg.get("idle_stop_s", 900))
        self.st = self._load()
        self.provider_name = "local"
        pcfg = dict(cfg.get("local") or {})
        pcfg.setdefault("log_dir", cfg["log_dir"])
        pcfg["state_dir"] = state_dir
        self.p = LocalProvider(pcfg, self.st, log)
        self.session_active = False
        self.on_idle = None              # server callback: end an open (silent) session before an idle stop
        self._lock = asyncio.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, indent=1)
        os.replace(tmp, self.path)

    def touch(self) -> None:
        self.st["last_activity"] = time.time()
        self.st["stop_at"] = self.st["last_activity"] + self.idle_s
        self.save()

    async def shutdown(self, why: str) -> None:
        """Server exit: the local render child is always stopped."""
        await self.stop(why)

    async def startup_check(self) -> None:
        """Runs once at server start: stop a render child a previous run left behind."""
        LocalProvider.reap_orphan(self.st, self.log)
        self.st["state"] = "stopped"
        self.save()

    async def ensure_up(self) -> dict:
        # single-flight: every caller (page prewarm, Start, a session re-dial) serializes here
        async with self._lock:
            s = self.status()
            if s["state"] in ("stopped", "error", "stopping"):
                self.st.pop("stop_requested", None)
                self.st.pop("stop_reason", None)
                self.st["state"] = "starting"
                self.st["started_at"] = time.time()
                self.touch()
                try:
                    await self.p.create_or_start()
                except Exception as e:  # noqa: BLE001
                    self.log(f"render server start failed: {type(e).__name__}: {str(e)[:200]}")
                    self.st["state"] = "error"
                    self.save()
                    raise
                self.st["state"] = "running"
                self.save()
            else:
                self.touch()
            return self.status()

    def status(self) -> dict:
        s = self.p.status()
        s["provider"] = self.provider_name
        s["stop_at"] = self.st.get("stop_at")
        s["started_at"] = self.st.get("started_at")
        s["idle_s"] = self.idle_s
        s["now"] = time.time()
        return s

    async def wait_ready(self, timeout: float, on_progress=None) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            s = self.status()
            if s["state"] == "running":
                return self.p.endpoint()
            if s["state"] in ("error", "stopped", "stopping"):
                raise RuntimeError(s.get("detail") or f"render server {s['state']}")
            if time.monotonic() > deadline:
                raise RuntimeError("render server did not come up in time")
            if on_progress:
                on_progress(s)
            await asyncio.sleep(0.5)

    async def stop(self, why: str) -> None:
        async with self._lock:
            self.st["stop_requested"] = True
            self.st["stop_reason"] = why
            self.save()
            try:
                await self.p.stop()
            except Exception:
                self.st["state"] = "stop_pending"
                self.save()
                self.log(f"render server stop unconfirmed; the watchdog will retry: {why}")
                raise
            confirmed = self.p.status()["state"] == "stopped"
            self.st["state"] = "stopped" if confirmed else "stop_pending"
            if confirmed:
                self.st["stopped_at"] = time.time()
                self.st.pop("stop_requested", None)
            self.save()
            self.log(f"render server {'stopped' if confirmed else 'stop pending'}: {why}")

    async def watchdog(self) -> None:
        while True:
            await asyncio.sleep(10)
            try:
                s = self.p.status()["state"]
                if self.st.get("stop_requested"):
                    await self.stop(self.st.get("stop_reason") or "retry unconfirmed stop")
                    continue
                # Idle = no REAL activity (the user's speech, Start, opening the page, character switch, commands) for
                # idle_stop_s. An open but silent session does not keep the renderer up. idle_stop_s <= 0 = never.
                if (self.idle_s > 0 and s != "stopped" and self.st.get("stop_at")
                        and time.time() >= float(self.st["stop_at"])):
                    if self.session_active and self.on_idle:
                        try:
                            self.on_idle()
                        except Exception:  # noqa: BLE001
                            pass
                    await self.stop(f"idle {self.idle_s / 60:.0f} min (no speech or page activity)")
            except Exception as e:  # noqa: BLE001
                self.log(f"render watchdog: {type(e).__name__}: {str(e)[:160]}")
