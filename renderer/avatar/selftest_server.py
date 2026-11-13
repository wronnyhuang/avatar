#!/usr/bin/env python3
"""Self-test of avatar/render_server.py with no GPU: the live loop's mock backend (live.testing:MouthBackend, a fixed
render cost and a measurable mouth) behind the real server, encoder and WebSocket.

    python -m avatar.selftest_server [--port 8779]

Checks: /health; 403 on a wrong key; 404 elsewhere; a real-time session holds 24 fps with no drops and every frame tagged
with its own instant; audio sent 1 s AHEAD gets its frames rendered ahead (they arrive before their instant's audio
would have been sent); a renderer slower than the frame period drops frames instead of building a backlog (lag stays
bounded); a character switch mid-stream sends a new ready and keeps the global frame numbering; a keyframe request
gets an IDR within a few frames; ping/pong.  Prints one JSON line per check and PASS/FAIL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
V_HDR = struct.Struct("<BBHId")
SR = 24000
mono = time.monotonic


async def session(url, start, seconds, ahead_s=0.0, switch_at=None, key_at=None, ping=False, voice=True):
    import websockets
    frames, msgs = [], []
    readies = []
    ev = asyncio.Event()
    async with websockets.connect(url, compression=None, max_size=2 ** 23) as ws:
        await ws.send(json.dumps(start))

        async def rx():
            async for m in ws:
                now = mono()
                if isinstance(m, bytes):
                    _, flags, _, F, t = V_HDR.unpack_from(m, 0)
                    frames.append((now, F, t, flags, len(m)))
                else:
                    j = json.loads(m)
                    msgs.append((now, j))
                    if j["type"] == "ready":
                        readies.append((now, j))
                        ev.set()
        task = asyncio.create_task(rx())
        await asyncio.wait_for(ev.wait(), 60)
        T0 = mono() + 0.05
        n = int(seconds * SR)
        chunk = 480
        rng = np.random.default_rng(0)
        sent = {}
        s = 0
        switched = keyed = False
        while s < n:
            now = mono()
            due = T0 + (s + chunk) / SR - ahead_s
            if due > now:
                await asyncio.sleep(due - now)
            # voiced bursts so the mock mouth moves: 300 ms of noise every second
            pcm = np.zeros(chunk, np.int16)
            if voice and (s // SR) % 1 == 0 and (s % SR) < 0.3 * SR:
                pcm = (rng.standard_normal(chunk) * 3000).astype(np.int16)
            await ws.send(struct.pack("<B3xQ", 0x41, s) + pcm.tobytes())
            sent[s] = mono()
            s += chunk
            tl = (mono() - T0)
            if switch_at is not None and not switched and tl > switch_at:
                switched = True
                await ws.send(json.dumps({"type": "character", "character": "bear"}))
            if key_at is not None and not keyed and tl > key_at:
                keyed = True
                key_sent = mono()
                await ws.send(json.dumps({"type": "keyframe"}))
            if ping and s % (SR // 2) == 0:
                await ws.send(json.dumps({"type": "ping", "t": tl}))
        await asyncio.sleep(0.8)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(task, 3)
        except Exception:
            task.cancel()
    return {"frames": frames, "msgs": msgs, "readies": readies, "T0": T0, "sent": sent,
            "key_sent": locals().get("key_sent")}


def run_checks(port, key):
    base = f"http://127.0.0.1:{port}"
    url = f"ws://127.0.0.1:{port}/render?key={key}"
    out = []

    def check(name, ok, **kw):
        out.append(ok)
        print(json.dumps({"check": name, "ok": bool(ok), **kw}), flush=True)

    h = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
    check("health", h.get("ok") is True and "busy" in h and "gpu" in h, health=h)
    for path, want in (("/render?key=wrong", 403), ("/render", 403), ("/elsewhere", 404)):
        try:
            urllib.request.urlopen(base + path, timeout=5)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        check(f"http {path}", code == want, code=code)

    mock = "live.testing:MouthBackend"
    st = {"type": "start", "character": "orb", "width": 320, "height": 180, "fps": 24, "codec": "h264",
          "backend": mock, "opts": {"ms": 20}}
    # 1. real time
    r = asyncio.run(session(url, st, 6.0, ping=True))
    F = [f[1] for f in r["frames"]]
    span = r["frames"][-1][0] - r["frames"][0][0]
    fps = (len(F) - 1) / span
    gaps = sum(1 for a, b in zip(F, F[1:]) if b != a + 1)
    ok_t = all(abs(f[2] - f[1] / 24) < 1e-9 for f in r["frames"])
    stats = [m for _, m in r["msgs"] if m["type"] == "stats"]
    pongs = [m for _, m in r["msgs"] if m["type"] == "pong"]
    delay = [1000 * (at - r["sent"][(Fi * 1000) // 480 * 480]) for at, Fi, *_ in r["frames"]
             if (Fi * 1000) // 480 * 480 in r["sent"]]
    check("realtime", fps > 23 and gaps == 0 and ok_t and r["frames"][0][3] & 1, fps=round(fps, 2), gaps=gaps,
          first_key=bool(r["frames"][0][3] & 1), delay_ms_mean=round(float(np.mean(delay)), 1),
          ready=r["readies"][0][1].get("codec_string"))
    check("ping", len(pongs) >= 5, pongs=len(pongs))
    check("stats", len(stats) >= 4 and all(k in stats[-1] for k in ("fps", "render_ms", "render_p95", "encode_ms",
                                                                       "late", "queue")), last=stats[-1] if stats else None)
    # 2. audio 1 s ahead: frames arrive before the wall time their instant's audio would be due on the clock
    r = asyncio.run(session(url, st, 5.0, ahead_s=1.0))
    early = [(r["T0"] + f[2]) - f[0] for f in r["frames"]]       # seconds before the frame's instant on the clock
    check("render_ahead", float(np.median(early)) > 0.5, median_early_s=round(float(np.median(early)), 3),
          frames=len(early))
    # 3. slow renderer (60 ms > 41.7 ms period): drops, bounded lag, no backlog
    st_slow = dict(st, opts={"ms": 60})
    r = asyncio.run(session(url, st_slow, 8.0))
    stats = [m for _, m in r["msgs"] if m["type"] == "stats" and m.get("fps")]
    lag = [s["lag_ms"] for s in stats if s.get("lag_ms") is not None]
    dropped = sum(s.get("dropped", 0) for s in stats)
    fps = (len(r["frames"]) - 1) / (r["frames"][-1][0] - r["frames"][0][0])
    check("slow_drops_not_backlog", dropped > 20 and max(lag[2:]) < 400 and 13 < fps < 18, fps=round(fps, 2),
          dropped=dropped, lag_ms=[round(x) for x in lag], queue=[s.get("queue") for s in stats])
    # 4. character switch mid-stream + keyframe request
    r = asyncio.run(session(url, st, 6.0, switch_at=2.0, key_at=4.5))
    F = [f[1] for f in r["frames"]]
    after = [f for f in r["frames"] if f[0] > r["readies"][-1][0]] if len(r["readies"]) > 1 else []
    mono_F = all(b > a for a, b in zip(F, F[1:]))
    check("switch", len(r["readies"]) == 2 and r["readies"][1][1]["character"] == "bear" and mono_F and
          len(after) > 50 and after[0][3] & 1, readies=len(r["readies"]), monotonic=mono_F, frames_after=len(after),
          switch_ms=r["readies"][1][1].get("init_ms"))
    ks = r.get("key_sent")
    keys_after = [f for f in r["frames"] if ks and f[0] > ks and f[3] & 1]
    check("keyframe_request", bool(keys_after) and keys_after[0][0] - ks < 0.3,
          key_latency_ms=round(1000 * (keys_after[0][0] - ks), 1) if keys_after else None)
    # 5. JPEG
    r = asyncio.run(session(url, dict(st, codec="jpeg"), 2.0))
    check("jpeg", r["frames"] and all(f[3] == 3 for f in r["frames"]) and r["readies"][0][1]["codec"] == "jpeg",
          frames=len(r["frames"]))
    return all(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8779)
    a = ap.parse_args(argv)
    key = secrets.token_urlsafe(24)
    env = dict(os.environ, RENDER_KEY=key, AVATAR_NVENC="0")
    log = open(os.path.join(os.environ.get("TMPDIR", "/tmp"), f"selftest_server_{a.port}.log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "avatar.render_server", "--port", str(a.port), "--max-engines", "2"],
                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        ok = run_checks(a.port, key)
    finally:
        p.terminate()
        p.wait(10)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
