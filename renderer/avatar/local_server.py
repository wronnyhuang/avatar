"""A render_server.py child process on this machine, with a small persisted state file (pid, port, started_at).

    from avatar.local_server import LocalServer
    P = LocalServer(port=8770)              # state in <state_dir>/local.json
    P.create_or_start(); P.wait_ready(); url = P.render_url(); ...; P.stop()

    python -m avatar.local_server start|status|stop|wait|health [--set port=8770] [--set profile=local] [--dry-run]

Used by the benchmark tools (avatar/bench_profiles.py). The app server (app/providers.py) runs its own child the same
way. The child gets RENDER_KEY from this process's environment; if it is unset, a fresh secrets.token_urlsafe(24) is
put into os.environ["RENDER_KEY"] first, so code in this same process dials with it. The state file never holds the key.
A pid that now belongs to another program is never signalled (/proc cmdline check).
"""
from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the renderer/ tree
REPO = os.path.dirname(ROOT)
STATE_DIR = os.environ.get("AVATAR_PROVIDER_STATE", os.path.join(REPO, "out", "local-render-state"))


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


class LocalServer:
    name = "local"

    def __init__(self, dry_run=False, state_dir=None, port=8770, host="127.0.0.1", profile="local", backend="auto",
                 python=None, nvenc=False, rt_tree=None, ta_out=None, log_path=None, extra_args=()):
        self.dry_run = bool(dry_run)
        self.state_dir = state_dir or STATE_DIR
        self.port, self.host, self.profile, self.backend = int(port), host, profile, backend
        self.python = python or sys.executable
        self.nvenc = bool(nvenc)
        self.rt_tree = rt_tree if rt_tree is not None else os.environ.get("AVATAR_RT_TREE", "")
        self.ta_out = ta_out or os.environ.get("TA_OUT") or os.path.join(REPO, "out", "cache")
        self.log_path = log_path or os.path.join(self.state_dir, f"render_server_{self.port}.log")
        self.extra_args = list(extra_args)

    # ------------------------------------------------------------ persisted state
    @property
    def state_path(self) -> str:
        return os.path.join(self.state_dir, f"{self.name}.json")

    def load_state(self) -> dict:
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_state(self, st: dict) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.name}.", suffix=".tmp", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(st, f, indent=1)
            os.replace(tmp, self.state_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------ process
    def _alive(self, st):
        pid = st.get("pid")
        return bool(pid) and "avatar.render_server" in _cmdline(pid)

    def command(self):
        cmd = [self.python, "-m", "avatar.render_server", "--host", self.host, "--port", str(self.port),
               "--profile", self.profile, "--backend", self.backend] + self.extra_args
        if self.rt_tree:
            cmd += ["--rt-tree", self.rt_tree]
        return cmd

    def _ep(self):
        return f"ws://{self.host}:{self.port}"

    def create_or_start(self):
        st = self.load_state()
        if self._alive(st):
            return self.status()
        if not os.environ.get("RENDER_KEY"):
            os.environ["RENDER_KEY"] = secrets.token_urlsafe(24)
        env = dict(os.environ, TA_OUT=self.ta_out, AVATAR_PROFILE=self.profile, AVATAR_NVENC="1" if self.nvenc else "0",
                   PYTHONUNBUFFERED="1")
        if self.dry_run:
            print("DRY-RUN spawn:", " ".join(self.command()), f"(cwd {ROOT}, RENDER_KEY <redacted>, TA_OUT {self.ta_out})")
            return {"provider": self.name, "state": "starting", "id": None, "endpoint": self._ep(), "dry_run": True}
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logf = open(self.log_path, "ab")
        p = subprocess.Popen(self.command(), cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        logf.close()
        self.save_state({"provider": self.name, "pid": p.pid, "port": self.port, "host": self.host,
                         "started_at": time.time(), "log": self.log_path})
        self._child = p          # keeps the Popen object (and so the zombie reaping) in this process
        return {"provider": self.name, "state": "starting", "id": p.pid, "endpoint": self._ep()}

    def status(self):
        st = self.load_state()
        if not st:
            return {"provider": self.name, "state": "absent", "id": None, "endpoint": None}
        alive = self._alive(st)
        out = {"provider": self.name, "id": st.get("pid"), "since": st.get("started_at"),
               "endpoint": self._ep() if alive else None, "log": st.get("log")}
        if not alive:
            out.update(state="stopped", detail="process not running")
            return out
        h = self.health() if out["endpoint"] else None
        out.update(state="running" if h else "starting", health=h)
        return out

    def endpoint(self):
        st = self.load_state()
        return self._ep() if self._alive(st) else None

    def render_url(self):
        e = self.endpoint()
        return None if not e else e.rstrip("/") + "/render"

    def health(self, timeout: float = 5.0):
        e = self.endpoint()
        if not e:
            return None
        url = e.replace("ws://", "http://").rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.loads(r.read().decode())
                return data if isinstance(data, dict) and data.get("ok") is True else None
        except Exception:
            return None

    def wait_ready(self, timeout: float = 600.0, every: float = 5.0) -> dict:
        """Poll status() until the render server answers /health or the timeout passes."""
        t0 = time.time()
        st = self.status()
        while time.time() - t0 < timeout:
            st = self.status()
            if st.get("state") == "running" and self.health():
                return st
            if st.get("state") in ("error", "absent", "stopped"):
                return st
            time.sleep(every)
        st["detail"] = f"not ready after {timeout:.0f}s; " + str(st.get("detail", ""))
        return st

    def stop(self, timeout=10.0):
        st = self.load_state()
        if self.dry_run:
            print(f"DRY-RUN stop: SIGTERM pid {st.get('pid')}")
            return {"provider": self.name, "state": "stopped", "dry_run": True}
        if self._alive(st):
            pid = st["pid"]
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            t0 = time.time()
            while time.time() - t0 < timeout and self._alive(st):
                time.sleep(0.2)
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if self._alive(st):
                try:
                    os.killpg(pid, signal.SIGKILL)          # its own session: the server and its render workers
                except ProcessLookupError:
                    pass
        if st:
            st.update(stopped_at=time.time(), pid=None)
            self.save_state(st)
        return {"provider": self.name, "state": "stopped", "id": None, "endpoint": None}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="avatar.local_server")
    ap.add_argument("cmd", choices=["start", "stop", "status", "endpoint", "health", "wait"])
    ap.add_argument("--dry-run", action="store_true", help="print the command, start nothing")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--set", action="append", default=[], help="option key=value (repeatable)")
    a = ap.parse_args(argv)
    cfg = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except ValueError:
            pass
        cfg[k] = v
    p = LocalServer(dry_run=a.dry_run, state_dir=a.state_dir, **cfg)
    fn = {"start": p.create_or_start, "stop": p.stop, "status": p.status, "endpoint": p.endpoint,
          "health": p.health, "wait": p.wait_ready}[a.cmd]
    print(json.dumps(fn(), indent=1))


if __name__ == "__main__":
    main()
