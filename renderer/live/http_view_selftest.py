"""Browser-free self-test of live/http_view.py.      python -m live.http_view_selftest [--out DIR]

Phase A  no client: push_frame / push_audio cost, no JPEG encode, /frame.jpg encodes on demand.
Phase B  one /stream reader + one /mjpeg reader (separate processes, like a browser): 3 s of 30 fps moving-bar frames
         and 20 ms PCM chunks paced in real time.  Checks hello-first, push order, monotonic + contiguous timestamps,
         JPEG size and bar position, /frame.jpg, /status, push cost and push->arrival latency.
Phase C  plus a client that never reads: 3 s more with large pre-encoded frames; pushes must stay sub-ms, the stuck
         client must be dropped (drop-counted) while the fast reader keeps every audio chunk.
Then     close(): streams end cleanly, port refused, the same port binds again.  The page's JS passes `node --check`
         and its message parser, run in node on the captured /stream bytes split at random boundaries, yields exactly
         the messages Python parsed.
Exit status 0 = every check passed; a JSON summary is printed either way.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

W, H, FPS, SR, CH = 640, 360, 30.0, 24000, 480        # CH = samples per 20 ms chunk
ROOT = Path(__file__).resolve().parents[1]


def reader_main(url: str, dump: str, seconds: float) -> None:
    """Subprocess mode: dump the (de-chunked) body and (end offset, monotonic time) per read until EOF."""
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=15)
    conn.request("GET", u.path)
    r = conn.getresponse()
    marks, n, err, t_end = [], 0, "", time.monotonic() + seconds
    with open(dump, "wb") as f:
        while time.monotonic() < t_end:
            try:
                b = r.read1(1 << 16)
            except Exception as e:                       # noqa: BLE001 — recorded, judged by the parent
                err = repr(e)
                break
            if not b:
                break
            f.write(b)
            n += len(b)
            marks.append((n, time.monotonic()))
    with open(dump + ".json", "w") as f:
        json.dump({"status": r.status, "headers": dict(r.getheaders()), "marks": marks, "err": err}, f)


def ms(xs) -> dict:
    import numpy as np
    a = np.asarray(xs, float) * 1000
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4), "max": round(float(a.max()), 4),
            "over_1ms": int((a > 1).sum())} if a.size else {"n": 0}


def main_test(out: Path) -> int:
    import cv2
    import numpy as np
    from live.http_view import HDR, LiveView, PAGE

    out.mkdir(parents=True, exist_ok=True)
    fails, checks, summ = [], {}, {}

    def check(name, ok, detail=""):
        checks[name] = bool(ok)
        if not ok:
            fails.append(f"{name}: {detail}")

    def bar(i):
        img = np.full((H, W, 3), 24, np.uint8)
        x = (i * 9) % (W - 40)
        img[:, x:x + 40] = (240, 200, 60)
        return img, x + 19.5

    def pcm(k):
        s = np.arange(k * CH, (k + 1) * CH)
        return (np.sin(2 * np.pi * 440 * s / SR) * 8000).astype("<i2").tobytes()

    def get(port, path):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), body

    def spawn(url, name):
        dump = str(out / name)
        p = subprocess.Popen([sys.executable, "-m", "live.http_view_selftest", "--reader", url, "--dump", dump],
                             cwd=str(ROOT))
        return p, dump

    def wait_clients(lv, n, timeout=10.0):
        end = time.monotonic() + timeout
        while lv.clients() < n and time.monotonic() < end:
            time.sleep(0.01)
        return lv.clients() >= n

    lv = LiveView(W, H, FPS, sample_rate=SR, port=0, info={"character": "selftest", "backend": "synthetic",
                                                           "delay_ms": 250})
    port = lv.port

    # ---- phase A: nobody connected ---------------------------------------------------------------------------------
    fa, aa = [], []
    for i in range(90):
        img, _ = bar(i)
        t0 = time.perf_counter(); lv.push_frame(i / FPS, img, index=i); fa.append(time.perf_counter() - t0)
    for k in range(150):
        b = pcm(k)
        t0 = time.perf_counter(); lv.push_audio(k * CH / SR, b); aa.append(time.perf_counter() - t0)
    summ["A_no_client"] = {"push_frame_ms": ms(fa), "push_audio_ms": ms(aa), "encodes": lv.stats()["encodes"]}
    check("A.no_encode_without_clients", lv.stats()["encodes"] == 0, lv.stats()["encodes"])
    check("A.push_cost_tiny", np.percentile(fa + aa, 99) < 0.5e-3 and float(np.mean(fa + aa)) < 0.02e-3,
          summ["A_no_client"])
    st, ct, body = get(port, "/frame.jpg")
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if st == 200 else None
    check("A.frame_jpg_on_demand", img is not None and img.shape == (H, W, 3) and ct == "image/jpeg", (st, ct))
    st, ct, page = get(port, "/")
    html = page.decode()
    check("page.served", st == 200 and "<canvas" in html and f'width="{W}"' in html, st)
    check("page.no_external_urls", not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", html.split("<script>")[0])
          and not re.search(r"https?://", html), "external URL in page")

    # ---- phase B: one /stream + one /mjpeg reader, real-time pacing ------------------------------------------------
    rs, dump_s = spawn(lv.url + "stream", "stream.bin")
    rm, dump_m = spawn(lv.url + "mjpeg", "mjpeg.bin")
    check("B.clients_connected", wait_clients(lv, 2), lv.clients())
    log = []                                             # (type, t, push monotonic, cost)
    fb, ab = [], []
    nf, na = int(3 * FPS), int(3 / 0.02)

    def paced(nf, na, f0, a0, frame_arg, costs_f, costs_a, origin, watch=None):  # stream time s goes out at origin + s
        fi = ai = 0
        while fi < nf or ai < na:
            tf = (f0 + fi) / FPS if fi < nf else 1e9
            ta = (a0 + ai) * CH / SR if ai < na else 1e9
            s = min(tf, ta)
            dt = origin + s - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            if tf <= ta:
                arg = frame_arg(f0 + fi)
                t0 = time.perf_counter(); tm = time.monotonic()
                lv.push_frame(tf, **arg, index=f0 + fi)
                costs_f.append(time.perf_counter() - t0); log.append((1, tf, tm)); fi += 1
            else:
                b = pcm(a0 + ai)
                t0 = time.perf_counter(); tm = time.monotonic()
                lv.push_audio(ta, b)
                costs_a.append(time.perf_counter() - t0); log.append((2, ta, tm)); ai += 1
            if watch:
                watch()

    paced(nf, na, 0, 0, lambda i: {"rgb": bar(i)[0]}, fb, ab, time.monotonic())
    time.sleep(0.3)
    stB = lv.stats()
    summ["B_one_stream_one_mjpeg"] = {"push_frame_ms_incl_encode": ms(fb), "encode_ms_mean": stB["encode_ms_mean"],
                                      "push_audio_ms": ms(ab)}
    check("B.push_audio_sub_ms", np.percentile(ab, 99) < 0.5e-3, ms(ab))   # p99, not max: the host's scheduler owns the max
    st, ct, body = get(port, "/frame.jpg")
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if st == 200 else None
    check("B.frame_jpg", img is not None and img.shape == (H, W, 3), st)
    st, ct, body = get(port, "/status")
    js = json.loads(body) if st == 200 else {}
    check("B.status", js.get("clients") == 2 and js.get("info", {}).get("character") == "selftest"
          and js.get("latest", {}).get("index") == nf - 1 and len(js.get("per_client", [])) == 2, js)

    # ---- phase C: add a client that never reads; big pre-encoded frames -------------------------------------------
    noise = np.random.default_rng(1).integers(0, 255, (H, W, 3), np.uint8)
    big = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
    slow = socket.socket()
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    slow.connect(("127.0.0.1", port))
    slow.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
    check("C.slow_client_connected", wait_clients(lv, 3), lv.clients())
    fc, ac, dropped_at = [], [], []
    tc0 = time.monotonic()

    def watch():
        if not dropped_at and lv.clients() < 3:
            dropped_at.append(time.monotonic() - tc0)
    paced(int(3 * FPS), int(3 / 0.02), nf, na, lambda i: {"jpeg": big}, fc, ac, time.monotonic() - 3.0, watch)
    time.sleep(0.3)
    stC = lv.stats()
    gone = [g for g in stC["gone"] if g["kind"] == "stream" and g["reason"] == "behind"]
    summ["C_with_stuck_client"] = {"push_frame_ms_prejpeg": ms(fc), "push_audio_ms": ms(ac), "jpeg_bytes": len(big),
                                   "stuck_client": (gone[0] | {"dropped_after_s": dropped_at[0] if dropped_at else None})
                                   if gone else None}
    check("C.push_sub_ms", np.percentile(fc + ac, 99) < 0.5e-3, summ["C_with_stuck_client"])
    check("C.stuck_client_dropped", bool(gone) and gone[0]["dropped_frames"] > 0 and lv.clients() == 2, stC["gone"])
    fast = [c for c in stC["per_client"] if c["kind"] == "stream"]
    check("C.fast_client_alive", len(fast) == 1 and fast[0]["alive"], stC["per_client"])
    summ["fast_client"] = fast[0] if fast else None

    # ---- close ----------------------------------------------------------------------------------------------------
    t0 = time.perf_counter(); lv.close(); summ["close_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    slow.close()
    for p in (rs, rm):
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
    check("close.readers_exited", rs.returncode == 0 and rm.returncode == 0, (rs.returncode, rm.returncode))
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        refused = False
    except OSError:
        refused = True
    check("close.port_refuses", refused)
    try:
        LiveView(W, H, FPS, port=port).close()
        rebind = True
    except OSError as e:
        rebind = repr(e)
    check("close.port_rebinds", rebind is True, rebind)

    # ---- /stream analysis -----------------------------------------------------------------------------------------
    raw = Path(dump_s).read_bytes()
    meta = json.loads(Path(dump_s + ".json").read_text())
    check("stream.headers", meta["status"] == 200 and meta["headers"].get("Transfer-Encoding") == "chunked"
          and not meta["err"], (meta["status"], meta["err"]))
    msgs, off = [], 0
    while off + 16 <= len(raw):
        typ, n, t = HDR.unpack_from(raw, off)
        if off + 16 + n > len(raw):
            break
        msgs.append((typ, t, off + 16, n))
        off += 16 + n
    check("stream.no_trailing_bytes", off == len(raw), (off, len(raw)))
    hello = json.loads(raw[msgs[0][2]:msgs[0][2] + msgs[0][3]]) if msgs and msgs[0][0] == 3 else {}
    check("stream.hello_first", hello.get("width") == W and hello.get("height") == H and hello.get("fps") == FPS
          and hello.get("sample_rate") == SR and hello.get("character") == "selftest", hello)
    body = msgs[1:]
    check("stream.one_hello", all(m[0] in (1, 2) for m in body))
    rx = [(m[0], m[1]) for m in body]
    it = iter([(x[0], x[1]) for x in log])
    check("stream.push_order_kept", all(any(r == p for p in it) for r in rx), "received is not a subsequence")
    fr = [m for m in body if m[0] == 1]
    au = [m for m in body if m[0] == 2]
    pushed_f = sum(1 for x in log if x[0] == 1)
    check("stream.frames_monotonic", all(b[1] > a[1] for a, b in zip(fr, fr[1:])))
    check("stream.audio_all_received", len(au) == sum(1 for x in log if x[0] == 2), (len(au), len(log)))
    gaps = [abs(b[1] - (a[1] + a[3] / 2 / SR)) for a, b in zip(au, au[1:])]
    check("stream.audio_contiguous", au and au[0][1] == 0.0 and max(gaps) < 1e-9, max(gaps) if gaps else None)
    check("stream.frames_received", len(fr) + (fast[0]["dropped_frames"] if fast else 0) == pushed_f,
          (len(fr), pushed_f))
    bad_size, bad_bar = 0, 0
    for typ, t, o, n in fr:
        if t < 3.0 + 10 / FPS:                            # every phase-B frame + the first 10 of phase C
            im = cv2.imdecode(np.frombuffer(raw, np.uint8, n, o), cv2.IMREAD_COLOR)
            if im is None or im.shape != (H, W, 3):
                bad_size += 1
                continue
            if t < 3.0:
                col = im.mean(axis=(0, 2))
                xs = np.nonzero(col > 100)[0]
                if not xs.size or abs(xs.mean() - bar(round(t * FPS))[1]) > 2:
                    bad_bar += 1
    check("stream.jpeg_decodes_right_size", bad_size == 0, bad_size)
    check("stream.frame_content_matches_t", bad_bar == 0, bad_bar)
    marks = meta["marks"]
    ends = np.array([m[0] for m in marks])
    tms = np.array([m[1] for m in marks])
    arr = {}
    for typ, t, o, n in body:
        arr[(typ, t)] = tms[np.searchsorted(ends, o + n)]
    lat_a = [arr[(2, t)] - tm for typ, t, tm in log if typ == 2 and (2, t) in arr and t < 3.0]
    lat_f = [arr[(1, t)] - tm for typ, t, tm in log if typ == 1 and (1, t) in arr and t < 3.0]
    summ["stream"] = {"bytes": len(raw), "frames": len(fr), "frames_pushed": pushed_f, "audio": len(au),
                      "push_to_arrival_ms_audio": ms(lat_a), "push_to_arrival_ms_frame_incl_encode": ms(lat_f)}

    # ---- /mjpeg analysis ------------------------------------------------------------------------------------------
    mraw = Path(dump_m).read_bytes()
    mmeta = json.loads(Path(dump_m + ".json").read_text())
    parts, off, bad = 0, 0, 0
    while True:
        i = mraw.find(b"--frame\r\n", off)
        if i < 0:
            break
        j = mraw.find(b"\r\n\r\n", i)
        hdr = mraw[i:j].decode("latin-1")
        mlen = re.search(r"Content-Length: (\d+)", hdr)
        if j < 0 or not mlen or "image/jpeg" not in hdr:
            bad += 1
            break
        n = int(mlen.group(1))
        im = cv2.imdecode(np.frombuffer(mraw, np.uint8, n, j + 4), cv2.IMREAD_COLOR) if j + 4 + n <= len(mraw) else None
        bad += im is None or im.shape != (H, W, 3)
        parts += 1
        off = j + 4 + n
    summ["mjpeg"] = {"parts": parts, "bad": bad, "content_type": mmeta["headers"].get("Content-Type")}
    check("mjpeg.parts", parts >= 0.5 * pushed_f and bad == 0
          and "multipart/x-mixed-replace" in (mmeta["headers"].get("Content-Type") or ""), summ["mjpeg"])

    # ---- the page's JS in node ------------------------------------------------------------------------------------
    node = shutil.which("node") or "/usr/bin/node"
    js_src = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    (out / "page.cjs").write_text(js_src)                 # .cjs: a parent package.json says "type": "module"
    if os.path.exists(node):
        r = subprocess.run([node, "--check", str(out / "page.cjs")], capture_output=True, text=True)
        check("page.node_check", r.returncode == 0, r.stderr[-400:])
        r = subprocess.run([node, "-e", "const f = process.argv[1]; new (require('vm').Script)("
                            "require('fs').readFileSync(f, 'utf8'), {filename: f})", str(out / "page.cjs")],
                           capture_output=True, text=True)       # compiles as a classic <script>, as the browser does
        check("page.compiles_as_classic_script", r.returncode == 0, r.stderr[-400:])
        parser = js_src.split("// <parser>", 1)[1].split("// </parser>", 1)[0]
        harness = parser.split("\n", 1)[1] + r"""
const data = require('fs').readFileSync(process.argv[2]);
const runs = [];
for (const seed of [1, 2, 3, 4]) {
  let s = seed; const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const msgs = []; const push = makeParser((type, t, p) => {
    let sum = 0; for (let i = 0; i < p.length; i++) sum = (sum + p[i]) % 4294967296; msgs.push([type, t, p.length, sum]); });
  let o = 0;
  while (o < data.length) {   // first 64 KB in 1..17-byte pieces, then pieces up to 1..96 KB
    const n = o < 65536 ? 1 + Math.floor(rnd() * 17) : 1 + Math.floor(rnd() * (seed % 2 ? 98304 : 3000));
    push(new Uint8Array(data.buffer, data.byteOffset + o, Math.min(n, data.length - o))); o += n;
  }
  runs.push(msgs);
}
const same = runs.every(r => JSON.stringify(r) === JSON.stringify(runs[0]));
console.log(JSON.stringify({same, msgs: runs[0]}));
"""
        (out / "parser_harness.cjs").write_text(harness)
        r = subprocess.run([node, str(out / "parser_harness.cjs"), dump_s], capture_output=True, text=True)
        try:
            res = json.loads(r.stdout)
            want = [[typ, t, n, int(np.frombuffer(raw, np.uint8, n, o).sum(dtype=np.uint64) % 4294967296)]
                    for typ, t, o, n in msgs]
            ok = res["same"] and res["msgs"] == want
        except Exception as e:                           # noqa: BLE001
            ok, res = False, {"err": repr(e), "stderr": r.stderr[-400:]}
        summ["node_parser"] = {"messages": len(res.get("msgs", [])), "split_runs_agree": res.get("same"),
                               "matches_python": ok}
        check("page.parser_in_node", ok, summ["node_parser"])
    else:
        summ["node_parser"] = "skipped: node not found"

    summ["checks_passed"] = f"{sum(checks.values())}/{len(checks)}"
    summ["failures"] = fails
    summ["out"] = str(out)
    print(json.dumps(summ, indent=1))
    return 1 if fails else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--reader", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    if a.reader:
        return reader_main(a.reader, a.dump, a.seconds)
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="http_view_selftest_"))
    sys.exit(main_test(out))


if __name__ == "__main__":
    main()
