"""Chunk protocol + assembler tests for app/recstore.py and app/recorder_hub.py (rec).

    python3 tests/recorder_hub_test.py                       # recstore only (stdlib)
    <venv with aiohttp>/bin/python tests/recorder_hub_test.py # + the HTTP endpoints

Real media: ffmpeg makes a 12 s VP8+Opus webm written like a live MediaRecorder stream (no cues / duration) and a
fragmented H.264+AAC mp4 (Safari's MediaRecorder shape); both are cut into chunks at arbitrary byte offsets.
Cases: in order, out of order + duplicates, a missing middle chunk, no 'end' (idle), 'end' with a chunk still missing
(grace), chunk 0 lost (fails -> .failed), a chunk after assembly (Closed), two stores racing for one rid, a dead
assembler's claim (recover), bad rid / seq / size / traversal, Safari mp4; HTTP: auth, beacon end, 410, 413.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import recstore  # noqa: E402

TMP = tempfile.mkdtemp(prefix="rectest-")
DUR = 12.0
RESULTS: list[tuple[str, bool, str]] = []


def media() -> dict:
    out = {}
    webm = os.path.join(TMP, "src.webm")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=480x270:rate=25", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000",
                    "-t", str(DUR), "-c:v", "libvpx", "-b:v", "600k", "-deadline", "realtime", "-cpu-used", "8",
                    "-c:a", "libopus", "-b:a", "48k", "-live", "1", "-cluster_time_limit", "1000", "-f", "webm", webm], check=True)
    out["webm"] = open(webm, "rb").read()
    mp4 = os.path.join(TMP, "src.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=480x270:rate=25", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000",
                    "-t", str(DUR), "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", mp4], check=True)
    out["mp4"] = open(mp4, "rb").read()
    return out


def cut(data: bytes, n: int, seed: int = 1) -> list[bytes]:
    rnd = random.Random(seed)
    pts = sorted(rnd.sample(range(1000, len(data) - 1000), n - 1))
    pts = [0] + pts + [len(data)]
    return [data[a:b] for a, b in zip(pts, pts[1:])]


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)


def store(root: str, **kw) -> recstore.RecStore:
    kw.setdefault("min_free_gb", 0)
    return recstore.RecStore(root=root, log=lambda m: None, **kw)


def run_case(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


def close(a, b, tol):
    return a is not None and abs(a - b) <= tol


def main() -> int:
    M = media()
    chunks = cut(M["webm"], 9)

    def assemble_due(st: recstore.RecStore, now=None):
        recs = []
        for rid, why in st.due(now):
            recs.append(st.assemble(rid, why))
        return recs

    # 1 ── in order + end -> assembled at once, mp4 has H.264 + AAC of the full length, index lists it
    def c1():
        root = os.path.join(TMP, "r1"); st = store(root)
        rid = st.start({"character": "muse", "voice": "realtime", "voice_label": "Realtime 2.1", "mime": "video/webm;codecs=vp8,opus",
                        "width": 480, "height": 270, "group": "g1", "sid": "abcdef123"}, {"variant": "dev"})["rid"]
        for i, c in enumerate(chunks):
            st.put_chunk(rid, i, c)
        assert st.end(rid, {"n": len(chunks), "reason": "stop", "duration_ms": 12000})["complete"]
        recs = assemble_due(st)
        r = recs[0]
        ok = (len(recs) == 1 and r["assembled_by"] == "end" and r["video"] == "h264" and r["audio"] == "aac"
              and close(r["duration_s"], DUR, 0.5) and r["missing"] == [] and r["chunks"] == len(chunks)
              and os.path.isfile(os.path.join(root, r["file"])) and r["variant"] == "dev" and r["character"] == "muse"
              and r["sid"] == "abcdef")
        idx = json.load(open(os.path.join(root, "index.json")))
        check("in-order + end -> mp4 (h264+aac, full length) + index", ok and idx["count"] == 1 and idx["sessions"][0]["rid"] == rid,
              f"{r['duration_s']} s {r['video']}+{r['audio']} {r['size_bytes'] // 1024} KB {r['file']}")
        check("not in the spool any more; a late chunk is Closed (410)", not os.path.isdir(os.path.join(root, ".spool", rid))
              and raises(lambda: st.put_chunk(rid, 3, b"x" * 10), recstore.Closed))
    run_case("c1", c1)

    # 2 ── out of order + duplicates (a retry whose response was lost) -> byte-identical stream
    def c2():
        root = os.path.join(TMP, "r2"); st = store(root)
        rid = st.start({"character": "fox", "mime": "video/webm"}, {"variant": "lab"})["rid"]
        order = list(range(len(chunks))); random.Random(7).shuffle(order)
        dups = 0
        for i in order + order[:4]:
            r = st.put_chunk(rid, i, chunks[i]); dups += 1 if r.get("dup") else 0
        s = st.status(rid)
        st.end(rid, {"n": len(chunks), "reason": "stop"})
        r = assemble_due(st)[0]
        check("out of order + duplicates -> complete, dups ignored", dups == 4 and s["have"] == list(range(len(chunks)))
              and r["missing"] == [] and close(r["duration_s"], DUR, 0.5), f"dups {dups}, {r['duration_s']} s")
    run_case("c2", c2)

    # 3 ── a missing middle chunk: assembled anyway (a gap), both streams, about the full length
    def c3():
        root = os.path.join(TMP, "r3"); st = store(root, end_grace_s=0.2)
        rid = st.start({"character": "muse", "mime": "video/webm"}, {"variant": "live"})["rid"]
        for i, c in enumerate(chunks):
            if i != 4:
                st.put_chunk(rid, i, c)
        assert not st.end(rid, {"n": len(chunks), "reason": "stop"})["complete"]
        early = st.due()
        time.sleep(0.3)
        recs = assemble_due(st)
        r = recs[0]
        check("'end' with a chunk missing waits for the grace, then assembles around the gap",
              early == [] and r["assembled_by"] == "end-partial" and r["missing"] == [4] and r["video"] == "h264"
              and r["audio"] == "aac" and close(r["duration_s"], DUR, 1.5), f"{r['duration_s']} s, missing {r['missing']}")
    run_case("c3", c3)

    # 4 ── no 'end' (the phone died): nothing until idle, then assembled by the idle rule
    def c4():
        root = os.path.join(TMP, "r4"); st = store(root, idle_s=0.5)
        rid = st.start({"character": "bear", "mime": "video/webm"}, {"variant": "dev"})["rid"]
        for i, c in enumerate(chunks[:6]):
            st.put_chunk(rid, i, c)
        early = st.due()
        time.sleep(0.7)
        r = assemble_due(st)[0]
        check("no 'end' -> idle timeout assembles what arrived", early == [] and r["assembled_by"] == "idle" and r["end"] == "none"
              and r["chunks"] == 6 and r["duration_s"] and 4 < r["duration_s"] < DUR, f"{r['duration_s']} s of {DUR}")
    run_case("c4", c4)

    # 5 ── chunk 0 lost (no header): the assembly fails cleanly, chunks kept in .failed with the error
    def c5():
        root = os.path.join(TMP, "r5"); st = store(root)
        rid = st.start({"mime": "video/webm"}, {"variant": "dev"})["rid"]
        for i, c in enumerate(chunks):
            if i != 0:
                st.put_chunk(rid, i, c)
        st.end(rid, {"n": len(chunks), "dropped": [0]})
        r = assemble_due(st)[0]
        fd = os.path.join(root, ".failed", rid)
        check("header chunk lost -> failed, kept in .failed/ with error.txt", r.get("failed") and os.path.isfile(os.path.join(fd, "error.txt"))
              and len([n for n in os.listdir(fd) if n.endswith(".bin")]) == len(chunks) - 1, (r.get("error") or "")[:80])
    run_case("c5", c5)

    # 6 ── two stores (two processes: the app + the index service) race for one rid: exactly one assembles
    def c6():
        root = os.path.join(TMP, "r6"); a, b = store(root), store(root)
        rid = a.start({"mime": "video/webm"}, {"variant": "dev"})["rid"]
        for i, c in enumerate(chunks):
            a.put_chunk(rid, i, c)
        a.end(rid, {"n": len(chunks)})
        out = []
        ts = [threading.Thread(target=lambda s=s: out.append(s.assemble(rid, "end"))) for s in (a, b)]
        [t.start() for t in ts]; [t.join() for t in ts]
        done = [r for r in out if r and r.get("file")]
        files = [n for n in os.listdir(os.path.join(root, done[0]["date"])) if n.endswith(".mp4")] if done else []
        check("two assemblers race -> one mp4, the other backs off", len(done) == 1 and out.count(None) == 1 and len(files) == 1)
    run_case("c6", c6)

    # 7 ── an assembler died after claiming (lock free, .work old) -> 'recover' finishes it
    def c7():
        root = os.path.join(TMP, "r7"); st = store(root)
        rid = st.start({"mime": "video/webm"}, {"variant": "dev"})["rid"]
        for i, c in enumerate(chunks):
            st.put_chunk(rid, i, c)
        st.end(rid, {"n": len(chunks)})
        w, fd = st.claim(rid, "end")
        os.close(fd)                                          # the process "died": its flock is gone
        old = time.time() - 600
        os.utime(w, (old, old))
        d = st.due()
        r = assemble_due(st)[0]
        check("dead assembler's claim is recovered", d == [(rid, "recover")] and r.get("file") and r["assembled_by"] == "recover")
    run_case("c7", c7)

    # 8 ── bad input: rid, seq, size, traversal; status/note/end on unknown rids
    def c8():
        root = os.path.join(TMP, "r8"); st = store(root)
        st.CHUNK_MAX = 1000
        rid = st.start({"mime": "video/webm"}, {"variant": "dev"})["rid"]
        ok = all([
            raises(lambda: st.put_chunk("../../etc", 0, b"x"), recstore.Closed),
            raises(lambda: st.put_chunk(rid, -1, b"x"), recstore.Refused),
            raises(lambda: st.put_chunk(rid, "a", b"x"), recstore.Refused),
            raises(lambda: st.put_chunk(rid, 0, b"x" * 1001), recstore.Refused),
            raises(lambda: st.status("20260101-000000-dev-00000000"), recstore.Closed),
            raises(lambda: st.end("nope", {}), recstore.Closed),
            st.file_path("../..", "x.mp4") is None, st.file_path("2026-11-09", "../index.json") is None,
            st.file_path("2026-11-09", "a/b.mp4") is None,
            raises(lambda: store(root, min_free_gb=1e9).start({}, {}), recstore.Refused),
        ])
        check("bad rid / seq / size / traversal / full disk refused", ok)
    run_case("c8", c8)

    # 9 ── Safari's MediaRecorder shape: fragmented mp4 chunks -> mp4
    def c9():
        root = os.path.join(TMP, "r9"); st = store(root)
        rid = st.start({"character": "muse", "mime": "video/mp4;codecs=avc1,mp4a"}, {"variant": "live"})["rid"]
        mc = cut(M["mp4"], 7, seed=3)
        for i, c in enumerate(mc):
            st.put_chunk(rid, i, c)
        st.end(rid, {"n": len(mc)})
        r = assemble_due(st)[0]
        check("Safari fragmented mp4 chunks -> mp4", r.get("video") == "h264" and r.get("audio") == "aac" and close(r["duration_s"], DUR, 0.5),
              f"{r.get('duration_s')} s")
    run_case("c9", c9)

    # 10 ── an empty recording (started, no chunk ever) is dropped quietly after idle
    def c10():
        root = os.path.join(TMP, "r10"); st = store(root, idle_s=0.2)
        rid = st.start({"mime": "video/webm"}, {"variant": "dev"})["rid"]
        time.sleep(0.3)
        r = assemble_due(st)[0]
        check("empty recording removed, nothing in the index", r.get("empty") and not os.listdir(os.path.join(root, ".spool"))
              and st.sessions() == [])
    run_case("c10", c10)

    http_tests(chunks)
    shutil.rmtree(TMP, ignore_errors=True)
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def http_tests(chunks) -> None:
    try:
        import asyncio
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        import recorder_hub
    except ImportError as e:
        print(f"SKIP http tests (no aiohttp: {e})")
        return
    import hmac

    async def go():
        root = os.path.join(TMP, "h1")
        os.environ["AVATAR_REC_ROOT"] = root
        os.environ.pop("AVATAR_VARIANT", None)
        SECRET = "s" * 24

        def authed(req, api=True):
            k = req.query.get("k", "")
            if not k or not hmac.compare_digest(k.encode(), SECRET.encode()):
                return False
            return not api or req.headers.get("x-requested-with") == "avatar-live"

        app = web.Application(client_max_size=4 * 1024 * 1024)
        st = recorder_hub.mount(app, authed=authed, home="/tmp/x/avatar-lab",
                                cfg={"recording": {"min_free_gb": 0}}, log=lambda m: None)
        st.CHUNK_MAX = max(len(c) for c in chunks) + 10
        H = {"x-requested-with": "avatar-live"}
        async with TestClient(TestServer(app)) as cl:
            r = await cl.post("/api/rec/start", params={"k": SECRET}, json={"character": "muse"})
            r0 = await cl.post("/api/rec/start", params={"k": SECRET}, headers=H, json={"character": "muse", "mime": "video/webm",
                                                                                          "voice": "gemini", "part": 1})
            j0 = await r0.json()
            rid = j0["rid"]
            check("http: start needs the header; variant from AVATAR_HOME (avatar-lab -> lab)",
                  r.status == 403 and r0.status == 200 and j0["variant"] == "lab" and "-lab-" in rid)
            nok = await cl.post("/api/rec/chunk", params={"k": "wrong", "rid": rid, "seq": 0}, headers=H, data=chunks[0])
            order = [1, 0, 2, 2, 3, 5, 4, 6, 7, 8]
            codes = []
            for i in order:
                rr = await cl.post("/api/rec/chunk", params={"k": SECRET, "rid": rid, "seq": i}, headers=H, data=chunks[i])
                codes.append(rr.status)
            big = await cl.post("/api/rec/chunk", params={"k": SECRET, "rid": rid, "seq": 9}, headers=H, data=b"x" * (st.CHUNK_MAX + 1))
            s = await (await cl.get("/api/rec/status", params={"k": SECRET, "rid": rid}, headers=H)).json()
            nt = await cl.post("/api/rec/note", params={"k": SECRET, "rid": rid}, headers=H, json={"t_s": 5.5, "character": "fox"})
            check("http: chunks out of order + dup accepted; bad key 403; oversize 413; status lists them",
                  nok.status == 403 and all(c == 200 for c in codes) and big.status == 413 and s["have"] == list(range(9))
                  and nt.status == 200, f"codes {codes}, big {big.status}")
            # the pagehide beacon: no header, query only, text body
            e = await cl.post("/api/rec/end", params={"k": SECRET, "rid": rid, "n": 9, "reason": "pagehide"}, data="{}")
            ej = await e.json()
            rec = None
            for _ in range(120):                 # the sweeper (kicked by 'end') assembles in a thread
                await asyncio.sleep(0.25)
                ss = st.sessions()
                if ss:
                    rec = ss[0]
                    break
            check("http: beacon 'end' (no header) -> kicked assembly -> mp4 + record",
                  e.status == 200 and ej.get("complete") and rec is not None and rec["end"] == "pagehide"
                  and rec["voice"] == "gemini" and [c["character"] for c in rec["characters"]] == ["muse", "fox"]
                  and close(rec["duration_s"], DUR, 0.5), f"{rec and rec['duration_s']} s {rec and rec['file']}")
            late = await cl.post("/api/rec/chunk", params={"k": SECRET, "rid": rid, "seq": 3}, headers=H, data=chunks[3])
            check("http: a chunk after assembly -> 410 (the page starts a new part)", late.status == 410)

        app2 = web.Application()
        recorder_hub.mount(app2, authed=authed, home="/x/avatar-live", cfg={"recording": False}, log=lambda m: None)
        async with TestClient(TestServer(app2)) as cl:
            r = await cl.post("/api/rec/start", params={"k": SECRET}, headers=H, json={})
            check("http: recording:false -> 503 off", r.status == 503 and (await r.json()).get("off"))
        os.environ.pop("AVATAR_REC_ROOT", None)

    try:
        asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        check("http tests", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
