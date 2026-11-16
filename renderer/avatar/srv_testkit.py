"""Small launcher / decoder helpers for the render-server tests (avatar/test_rigdbg.py): start this tree's render
server (or app/stub_render.py) on a free local port with a random RENDER_KEY, wait for /health, stop it, decode H.264."""
from __future__ import annotations

import os
import secrets
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # renderer/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
V_HDR = struct.Struct("<BBHId")         # 'V', flags, pad, uint32 frame number, float64 t
SR = 24000
mono = time.monotonic


def wait_health(port, timeout=60):
    t = mono()
    while mono() - t < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.1)
    return False


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def launch(target, tree, logpath, env_extra=None):
    """-> (Popen, port, key). target 'stub' = app/stub_render.py, else avatar.render_server of `tree`."""
    key = secrets.token_urlsafe(16)
    port = free_port()
    env = dict(os.environ, RENDER_KEY=key, **(env_extra or {}))
    log = open(logpath, "w")
    if target == "stub":
        cmd = [sys.executable, os.path.join(tree, "app", "stub_render.py"), "--port", str(port), "--key", key,
               "--init-ms", "100", "--encoder", "pyav"]
        cwd = os.path.join(tree, "app")
    else:
        cmd = [sys.executable, "-m", "avatar.render_server", "--port", str(port)]
        cwd = os.path.join(tree, "renderer")
        env["PYTHONPATH"] = cwd + (":" + ROOT if tree != os.path.dirname(ROOT) else "")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait_health(port):
        p.terminate()
        raise RuntimeError(f"server did not come up: see {logpath}")
    return p, port, key


def stop(p):
    p.terminate()
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def decode(payloads):
    """Fresh H.264 decoder over a list of Annex-B access units -> (frames decoded, set of sizes, errors)."""
    import av
    cc = av.CodecContext.create("h264", "r")
    n, sizes, errs = 0, set(), 0
    for p in payloads:
        try:
            for fr in cc.decode(av.Packet(p)):
                n += 1
                sizes.add((fr.width, fr.height))
        except Exception:
            errs += 1
    try:
        for fr in cc.decode(None):
            n += 1
            sizes.add((fr.width, fr.height))
    except Exception:
        pass
    return n, sizes, errs
