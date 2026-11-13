"""tvdsl.worker -- the TV render loop, run as its own process (python app/tvdsl/worker.py W H FPS) or in a thread (tests).

stdin  (parent -> worker): one JSON object per line: {"op":"lines","lines":[...]} | {"op":"clear"} | {"op":"stop"}
stdout (worker -> parent): frames  b"F" + u32 render_ms*10 + u32 nbytes + RGB (H x W x 3, row-major)
                           idle    b"I"          (show the idle/off look)
                           error   b"E" + u32 n + utf8
Only CHANGED frames are sent (a static picture costs one frame). Timing: the worker's own monotonic clock.
"""
import json
import os
import queue
import sys
import threading
import time

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from tvdsl.lang import Doc, parse_line  # noqa: E402
from tvdsl.render import Renderer  # noqa: E402

CROSSFADE = 0.5


def run(inp, out, w=512, h=288, fps=12):
    cv2.setNumThreads(1)
    q = queue.Queue()

    def reader():
        for raw in iter(inp.readline, b""):
            try:
                q.put(json.loads(raw))
            except ValueError:
                continue
        q.put({"op": "stop"})
    threading.Thread(target=reader, daemon=True).start()

    def send(b):
        try:
            out.write(b)
            out.flush()
            return True
        except (OSError, ValueError, BrokenPipeError):
            return False

    R = Renderer(w, h)
    doc = Doc()
    doc.empty = True
    last = None
    fade_from, fade_t0 = None, 0.0
    period = 1.0 / max(1, int(fps))
    nxt = time.monotonic()
    idle_sent = False
    while True:
        timeout = max(0.0, nxt - time.monotonic())
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            msg = None
        while msg is not None:
            op = msg.get("op")
            now = time.monotonic()
            if op == "stop":
                return
            if op == "clear":
                doc = Doc()
                doc.empty = True
                idle_sent = False
                fade_from = None
            elif op == "lines":
                for line in msg.get("lines") or []:
                    try:
                        sts = parse_line(str(line))
                    except Exception:  # noqa: BLE001
                        continue
                    for st in sts:
                        if st.op == "scene" and last is not None and not doc.empty:
                            fade_from, fade_t0 = last, now
                        try:
                            doc.apply(st, now)
                        except Exception:  # noqa: BLE001
                            continue
            try:
                msg = q.get_nowait()
            except queue.Empty:
                msg = None
        now = time.monotonic()
        if now < nxt:
            continue
        nxt = max(nxt + period, now - period)
        if doc.empty:
            if not idle_sent:
                if not send(b"I"):
                    return
                idle_sent = True
                last = None
            continue
        t0 = time.perf_counter()
        try:
            img = R.render(doc, now)
            if fade_from is not None:
                u = (now - fade_t0) / CROSSFADE
                if u >= 1:
                    fade_from = None
                else:
                    img = cv2.addWeighted(fade_from, 1 - u, img, u, 0.0)
        except Exception as e:  # noqa: BLE001
            b = f"{type(e).__name__}: {e}"[:300].encode()
            if not send(b"E" + len(b).to_bytes(4, "little") + b):
                return
            continue
        ms = (time.perf_counter() - t0) * 1000
        if last is not None and np.array_equal(last, img):
            continue
        last = img
        if not send(b"F" + int(ms * 10).to_bytes(4, "little") + img.nbytes.to_bytes(4, "little") + img.tobytes()):
            return


if __name__ == "__main__":
    try:
        os.nice(5)
    except OSError:
        pass
    a = sys.argv[1:]
    run(sys.stdin.buffer, sys.stdout.buffer, int(a[0]) if a else 512, int(a[1]) if len(a) > 1 else 288,
        int(a[2]) if len(a) > 2 else 12)
