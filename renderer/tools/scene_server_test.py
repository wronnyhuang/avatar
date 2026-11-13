#!/usr/bin/env python3
"""scene: end-to-end check of the live surfaces through the REAL render server (CPU, no GPU lock).

Starts avatar/render_server.py with AVATAR_RT_DEVICE=cpu for the muse at 480x270, plays the app server's timeline (silence on
the 20 ms tick), and sends 'S' messages made by the app server's own app/surfaces.py:
  * board strokes stamped t = 2.0 s (sent at once, i.e. ahead of time),
  * a TV picture stamped t = 4.0 s,
  * (after 5 s) a fresh session-less resync check is not done here (app tests cover the app server side).
Then compares the returned frames: the board box must change only in frames at t >= 2.0, the TV box only at t >= 4.0.

  python tools/scene_server_test.py --out DIR [--seconds 7]

Prints one JSON line; exit 0 = pass.
"""
import argparse, asyncio, io, json, os, socket, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "app"))
import surfaces as hub_surfaces  # noqa: E402  (the app server's own encoder)

# boxes in 480x270 home-view pixels (x0, y0, x1, y1): the board's visible writable part, the TV's picture
BOARD_BOX = (138, 104, 172, 168)
TV_BOX = (360, 50, 468, 110)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


async def run(a):
    import websockets
    from PIL import Image
    port = free_port()
    env = dict(os.environ, RENDER_KEY="t", AVATAR_RT_DEVICE="cpu", LP_NUM_THREADS=str(a.threads), AVATAR_NVENC="0")
    proc = subprocess.Popen([sys.executable, "-m", "avatar.render_server", "--port", str(port), "--codec", "jpeg",
                             "--width", "480", "--height", "270", "--late-ms", "400"], cwd=ROOT, env=env,
                            stderr=open(os.path.join(a.out, "server.log"), "w"))
    res = {"ready": None, "frames": 0, "errors": [], "surfaces_stats": None}
    frames = []
    try:
        for _ in range(150):
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{port}/render?key=t", max_size=2**24)
                break
            except OSError:
                await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "start", "character": "muse", "width": 480, "height": 270, "fps": 24,
                                  "codec": "jpeg", "backend": "rt", "opts": {"preset": "q50", "tier": "fixed"}}))

        async def reader():
            async for m in ws:
                if isinstance(m, (bytes, bytearray)):
                    if m[0] == 0x56:
                        _, flags, _, F, t = struct.unpack_from("<BBHId", m, 0)
                        frames.append((F, t, bytes(m[16:])))
                    continue
                o = json.loads(m)
                if o.get("type") == "ready":
                    res["ready"] = {k: o.get(k) for k in ("character", "width", "height", "fps", "codec")}
                elif o.get("type") == "stats":
                    res["surfaces_stats"] = o.get("surfaces")
                elif o.get("type") == "error":
                    res["errors"].append(o.get("message"))
        rt = asyncio.create_task(reader())
        t0 = time.monotonic()
        while res["ready"] is None and time.monotonic() - t0 < 600:
            await asyncio.sleep(0.1)
        if res["ready"] is None:
            raise RuntimeError("no ready")
        S = hub_surfaces.Surfaces(send=None)
        B, T = S.get("board"), S.get("tv")
        msgs = []
        for row in range(6):               # thick horizontal bars over the left half of the board, all at t = 2.0
            y = 0.1 + row*0.15
            msgs += B.draw_polyline([(0.02, y), (0.45, y)], 0.06, t=2.0)
        rgb = np.zeros((288, 512, 3), np.uint8); rgb[..., 0] = 250; rgb[::2, :, 1] = 200
        msgs += T.put_frame(rgb, t=4.0)
        res["s_msgs"] = len(msgs); res["s_bytes"] = sum(len(m) for m in msgs)
        sent = False
        tick, s0, k = 0.02, 0, 0
        start = time.monotonic()
        while time.monotonic() - start < a.seconds:
            el = time.monotonic() - start
            n = int(el/tick) - k
            for _ in range(max(n, 0)):
                await ws.send(struct.pack("<B3xQ", 0x41, s0) + np.zeros(480, "<i2").tobytes())
                s0 += 480; k += 1
            if not sent and el > 0.3:
                for m in msgs:
                    await ws.send(m)
                sent = True
            await asyncio.sleep(0.005)
        await asyncio.sleep(1.5)
        await ws.send(json.dumps({"type": "stop"}))
        rt.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    res["frames"] = len(frames)
    def box_mean(img, b):
        return float(img[b[1]:b[3], b[0]:b[2]].astype(np.float32).mean())
    rows = []
    for F, t, data in frames:
        img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        rows.append((t, box_mean(img, BOARD_BOX), box_mean(img, TV_BOX[0:4]), img))
    rows.sort(key=lambda r: r[0])
    def stat(sel, i):
        v = [r[i] for r in rows if sel(r[0])]
        return (round(float(np.mean(v)), 2), len(v)) if v else (None, 0)
    fps = 24.0
    res["board_before"] = stat(lambda t: t < 2.0 - 0.5/fps, 1); res["board_after"] = stat(lambda t: t >= 2.0, 1)
    res["tv_before"] = stat(lambda t: t < 4.0 - 0.5/fps, 2); res["tv_after"] = stat(lambda t: t >= 4.0, 2)
    # the first frame at/after each stamp, and the last one before
    first_b = [r for r in rows if r[0] >= 2.0][:1]; last_nb = [r for r in rows if r[0] < 2.0 - 0.5/fps][-1:]
    res["board_switch"] = [round(r[0], 3) for r in last_nb + first_b]
    ok = (not res["errors"] and res["frames"] > 10 and res["board_after"][0] is not None and res["board_before"][0] is not None
          and res["board_after"][0] - res["board_before"][0] > 12 and res["tv_after"][0] is not None
          and res["tv_before"][0] is not None and abs(res["tv_after"][0] - res["tv_before"][0]) > 12)
    # no change in the board box between frames before 2.0 (only noise / grain)
    pre = [r[1] for r in rows if r[0] < 2.0 - 0.5/fps]
    res["board_pre_spread"] = round(float(np.ptp(pre)), 2) if pre else None
    ok = ok and (res["board_pre_spread"] is None or res["board_pre_spread"] < 6)
    for name, sel in (("before", lambda t: t < 2.0 - 0.5/fps), ("board", lambda t: 2.0 <= t < 4.0 - 0.5/fps), ("tv", lambda t: t >= 4.0)):
        pick = [r for r in rows if sel(r[0])]
        if pick:
            Image.fromarray(pick[-1][3]).save(os.path.join(a.out, f"server_{name}.png"))
    res["ok"] = bool(ok)
    print(json.dumps(res))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
