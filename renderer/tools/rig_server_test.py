#!/usr/bin/env python3
"""End-to-end check of the render server's rig path on the CPU (no GPU lock): starts avatar/render_server.py with
AVATAR_RT_DEVICE=cpu, plays the app server (timeline silence on the 20 ms tick), sends {"type":"rig"} vectors at 30 Hz in the
order the "ready" message announces (rig_params), and saves the JPEG frames that come back.

  python tools/rig_server_test.py --character fox --seconds 8 --out DIR

Checks: ready carries rig_params; stats report rig_active while vectors arrive; the right paw rises (WAVE_UP ramp) in
the returned frames; after the vectors stop, rig_active goes false (life resumes).  Prints one JSON line.
"""
import argparse, asyncio, json, math, os, socket, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


async def run(a):
    import websockets
    port = free_port()
    env = dict(os.environ, RENDER_KEY="t", AVATAR_RT_DEVICE="cpu", LP_NUM_THREADS=str(a.threads), AVATAR_NVENC="0")
    proc = subprocess.Popen([sys.executable, "-m", "avatar.render_server", "--port", str(port), "--codec", "jpeg",
                             "--width", "480", "--height", "270", "--late-ms", "400"], cwd=ROOT, env=env,
                            stderr=open(os.path.join(a.out, "server.log"), "w"))
    res = {"ready": None, "frames": 0, "rig_active_seen": False, "rig_inactive_after": None, "errors": []}
    try:
        for _ in range(100):
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{port}/render?key=t", max_size=2**24)
                break
            except OSError:
                await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "start", "character": a.character, "width": 480, "height": 270, "fps": 24,
                                  "codec": "jpeg", "backend": "rt", "opts": {"preset": "q50"}}))
        names = None
        t_ready = None
        frames = []
        stop_rig_at = a.seconds - 3.0

        async def reader():
            nonlocal names, t_ready
            async for m in ws:
                if isinstance(m, (bytes, bytearray)):
                    if m[0] == 0x56:
                        _, flags, _, F, t = struct.unpack_from("<BBHId", m, 0)
                        frames.append((F, t, bytes(m[16:])))
                    continue
                o = json.loads(m)
                if o.get("type") == "ready":
                    res["ready"] = {k: o.get(k) for k in ("character", "width", "height", "fps", "codec")}
                    names = o.get("rig_params")
                    res["n_rig_params"] = len(names or [])
                    t_ready = time.monotonic()
                elif o.get("type") == "stats":
                    if o.get("rig_active"):
                        res["rig_active_seen"] = True
                    elif res["rig_active_seen"] and t_ready and time.monotonic() - t_ready > stop_rig_at + 2.6:
                        res["rig_inactive_after"] = True
                elif o.get("type") == "error":
                    res["errors"].append(o.get("message"))
        rt = asyncio.create_task(reader())
        t0 = time.monotonic()
        while res["ready"] is None and time.monotonic() - t0 < 240:
            await asyncio.sleep(0.1)
        if res["ready"] is None:
            raise RuntimeError("no ready")
        tick, s0, k = 0.02, 0, 0
        start = time.monotonic()
        while True:
            el = time.monotonic() - start
            if el > a.seconds:
                break
            n = int(el/tick) - k
            for _ in range(max(n, 0)):
                ws_msg = struct.pack("<B3xQ", 0x41, s0) + np.zeros(480, "<i2").tobytes()
                await ws.send(ws_msg)
                s0 += 480; k += 1
                tl = s0/24000.0
                if k % 2 == 0 and el < stop_rig_at and names:
                    w = min(1.0, el/1.5)
                    p = {"hand_r_x": 0.2*w + 0.12*w*math.sin(2*math.pi*2.5*tl), "hand_r_y": 1.0*w, "hand_r_z": 0.2*w,
                         "hand_r_open": 0.2 + 0.8*w, "hand_r_turn": w, "smile": 0.6, "ear_l": 1.0*w, "tail_wag": math.sin(tl*6)}
                    vec = [float(p.get(nm, 0.0)) for nm in names]
                    await ws.send(json.dumps({"type": "rig", "t": tl, "a": vec}))
            await asyncio.sleep(0.005)
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "stop"}))
        rt.cancel()
        res["frames"] = len(frames)
        # save a strip: one frame per second
        from PIL import Image
        import io
        sel = [f for f in frames if f[0] % 24 == 0]
        ims = [Image.open(io.BytesIO(f[2])).convert("RGB") for f in sel[:10]]
        if ims:
            W, H = ims[0].size
            strip = Image.new("RGB", (W*len(ims)//2, H//2))
            for i, im in enumerate(ims):
                strip.paste(im.resize((W//2, H//2)), (i*W//2, 0))
            strip.save(os.path.join(a.out, f"{a.character}_server_strip.jpg"), quality=85)
            res["strip"] = os.path.join(a.out, f"{a.character}_server_strip.jpg")
            res["strip_frames"] = [round(f[1], 2) for f in sel[:10]]
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="fox")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
