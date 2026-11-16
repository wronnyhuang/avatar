#!/usr/bin/env python3
"""rigdbg: the hub relay end to end, as a page would see it (minimal page client, no browser, no voice vendor).

    (a running app server with the stub renderer: "local": {"force_stub": true} in AVATAR_HOME/state/config.json)
    python tests/rigdbg_relay_test.py --url 'http://127.0.0.1:PORT/?k=KEY' --inst $AVATAR_HOME [--expect unsupported]
    (scripts/smoke.sh runs it against its own server with --renderer real)

The client opens a direct-voice session (the page's own WebRTC voice is not dialled: $0), uploads the playback tap ('P')
with speech / silence like the page does, polls /api/down, and sends {"type":"rigdbg","on":..} as the panel would.
Session 1: on BEFORE the renderer is ready, off at live+3 s, on at +5 s, a character switch at +6 s (so the hub's re-sent
"on" after its ready, not the 2 s cols timer, is what puts cols on the next message), off at +10 s, bye.
Session 2 (same hub, same renderer connection for the stub): never on.
Checks (default, a renderer with the handler): rigdbg messages reach the page only while on (grace 0.4 s after off),
the bring-up and the switch resume them without the page re-sending, row t = F / fps of a 'V' the page received,
"cols" on the first message after each "on" (the page's and the hub's re-send after the switch's ready) and otherwise at
most every 2 s (every other message read with the last list seen),
jaw opens in speech and is 0 in silence, session 2 gets none, the renderer saw "on" after every ready (render-child.log),
no rigdbg text in the session records (data/session-*.json).
--expect unsupported (an older renderer): the page gets ONE {"type":"rigdbg","unsupported":true} per open, no rows,
nothing in the session's errors, the hub stops asking that renderer (one error per renderer connection).
Prints one JSON line; exit 1 on a failed check.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import struct
import sys
import time
from urllib.parse import urlsplit, parse_qs

import aiohttp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
ASSETS = os.path.join(TREE, "renderer", "live", "assets")
SR = 24000
U32 = struct.Struct("<I")
P_HDR = struct.Struct("<BB2xQI")
V_HDR = struct.Struct("<BB2xId")
HDR = {"x-requested-with": "avatar-live"}
mono = time.monotonic
TREE_R = TREE

def _snippet(k):
    """live/assets/tts_<k>.pcm (int16, 24 kHz) when present, else the generated syllable of renderer/live/latency.py"""
    p = os.path.join(ASSETS, f"tts_{k}.pcm")
    if os.path.isfile(p):
        return np.fromfile(p, "<i2")
    sys.path.insert(0, os.path.join(TREE_R, "renderer"))
    from live.latency import synth_syllable
    return synth_syllable(k)


def speech(seconds, spans):
    snips = [_snippet(k) for k in ("pa", "ma")]
    x = np.zeros(int(seconds * SR), np.int16)
    for a, b in spans:
        s, k = int(a * SR), 0
        while s < int(b * SR):
            p = snips[k % 2]
            n = min(p.size, int(b * SR) - s, x.size - s)
            if n <= 0:
                break
            x[s:s + n] = p[:n]
            s += n + int(0.06 * SR)
            k += 1
    return x


class Page:
    def __init__(self, http, base, key):
        self.http, self.base, self.key = http, base, key
        self.sid = None
        self.got = []                  # (mono, json)
        self.frames = []               # (mono, F, t)
        self.jbytes = []               # (mono, type, bytes as the page received the 'J' message)
        self.ack = 0
        self.seen = set()
        self.live = asyncio.Event()
        self.ended = False

    def q(self, **kw):
        return {"k": self.key, **({"sid": self.sid} if self.sid else {}), **{k: str(v) for k, v in kw.items()}}

    async def open(self, character):
        async with self.http.post(self.base + "/api/session", params=self.q(), headers=HDR,
                                  json={"character": character, "voice": "direct", "quality": "auto"}) as r:
            d = await r.json()
        if not d.get("ok"):
            raise RuntimeError(f"session refused: {d}")
        self.sid = d["sid"]

    async def up(self, msgs):
        body = b"".join(U32.pack(len(m)) + m for m in msgs)
        async with self.http.post(self.base + "/api/up", params=self.q(), headers=HDR, data=body) as r:
            await r.read()

    async def ctl(self, obj):
        await self.up([b"J" + json.dumps(obj).encode()])

    async def poll_loop(self):
        while not self.ended:
            try:
                async with self.http.get(self.base + "/api/down", params=self.q(after=self.ack), headers=HDR,
                                         timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 410:
                        return
                    body = await r.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(0.05)
                continue
            now = mono()
            if len(body) < 4:
                continue
            seq = U32.unpack_from(body, 0)[0]
            if seq == 0 or seq in self.seen:
                if seq:
                    self._advance(seq)
                continue
            self.seen.add(seq)
            self._advance(seq)
            off = 4
            while off + 4 <= len(body):
                n = U32.unpack_from(body, off)[0]
                m = body[off + 4:off + 4 + n]
                off += 4 + n
                if not m:
                    continue
                if m[0] == 0x4A:
                    j = json.loads(m[1:].decode())
                    self.got.append((now, j))
                    self.jbytes.append((now, j.get("type"), len(m)))
                    if j.get("type") == "status" and j.get("stage") == "live":
                        self.live.set()
                elif m[0] == 0x56 and len(m) >= V_HDR.size:
                    _, flags, F, t = V_HDR.unpack_from(m, 0)
                    self.frames.append((now, F, t))

    def _advance(self, seq):
        while self.ack + 1 in self.seen:
            self.ack += 1

    async def tap_loop(self, pcm, t0_box):
        """the page's playback tap: 40 ms 'P' chunks on the playback timeline (silent runs flagged), real time"""
        s, chunk = 0, 960
        t0 = mono()
        t0_box.append(t0)
        while not self.ended and s < pcm.size:
            due = t0 + (s + chunk) / SR
            d = due - mono()
            if d > 0:
                await asyncio.sleep(d)
            seg = pcm[s:s + chunk]
            if not seg.any():
                msg = P_HDR.pack(0x50, 1, s, seg.size)
            else:
                msg = P_HDR.pack(0x50, 0, s, seg.size) + seg.tobytes()
            try:
                await self.up([msg])
            except aiohttp.ClientError:
                pass
            s += chunk

    async def bye(self):
        self.ended = True
        try:
            async with self.http.post(self.base + "/api/bye", params=self.q(), headers=HDR, data=b"") as r:
                await r.read()
        except aiohttp.ClientError:
            pass


def rigdbg_msgs(page):
    return [(t, j) for t, j in page.got if j.get("type") == "rigdbg"]


async def session1(http, base, key, expect, switch_to="bear"):
    p = Page(http, base, key)
    await p.open("fox")
    poll = asyncio.create_task(p.poll_loop())
    marks = {}
    await p.ctl({"type": "rigdbg", "on": True})              # the panel is open before the renderer is ready
    marks["on0"] = mono()
    await asyncio.wait_for(p.live.wait(), 300)
    L = mono()
    marks["live"] = L
    pcm = speech(150.0, [(0.5, 2.5), (4.0, 6.5)] + [(8.5 + 5 * k, 10.5 + 5 * k) for k in range(28)])
    t0b = []
    tap = asyncio.create_task(p.tap_loop(pcm, t0b))

    async def at(dt, what):
        d = L + dt - mono()
        if d > 0:
            await asyncio.sleep(d)
        marks[what] = mono()

    await at(3.0, "off1"); await p.ctl({"type": "rigdbg", "on": False})
    await at(5.0, "on2"); await p.ctl({"type": "rigdbg", "on": True})
    await at(6.0, "switch"); await p.ctl({"type": "character", "character": switch_to})   # < 2 s after on2's cols
    t_sw = mono()
    while mono() - t_sw < 240 and not any(j.get("type") == "status" and j.get("stage") == "live" and t > t_sw
                                         for t, j in p.got):
        await asyncio.sleep(0.1)                     # the switch's own init (seconds on the real renderer on llvmpipe)
    sw_live = mono() - L
    await at(max(10.0, sw_live + 3.0), "off2"); await p.ctl({"type": "rigdbg", "on": False})
    await at(max(12.0, sw_live + 5.0), "end")
    await p.bye()
    tap.cancel()
    await asyncio.sleep(0.5)
    poll.cancel()
    return p, marks, pcm, t0b[0] if t0b else L


async def session2(http, base, key):
    p = Page(http, base, key)
    await p.open("fox")
    poll = asyncio.create_task(p.poll_loop())
    await asyncio.wait_for(p.live.wait(), 300)
    pcm = speech(5.0, ((0.5, 3.0),))
    t0b = []
    await p.tap_loop(pcm, t0b)
    await p.bye()
    await asyncio.sleep(0.4)
    poll.cancel()
    return p


async def run(a):
    u = urlsplit(a.url)
    base = f"{u.scheme}://{u.netloc}"
    key = parse_qs(u.query)["k"][0]
    t_start = time.time()
    logp = os.path.join(a.inst, "data", "render-child.log") if a.inst else ""
    log_off = os.path.getsize(logp) if logp and os.path.exists(logp) else 0
    async with aiohttp.ClientSession() as http:
        p1, marks, pcm, tap0 = await session1(http, base, key, a.expect, a.switch_to)
        await asyncio.sleep(1.5)                       # the hub writes the session record
        p2 = await session2(http, base, key)
        await asyncio.sleep(1.5)
    rep = {"expect": a.expect}
    ok = True
    m1 = rigdbg_msgs(p1)
    rel = lambda t: round(t - marks["live"], 3)
    rep["session1"] = {"rigdbg_msgs": len(m1), "frames": len(p1.frames),
                       "marks_rel_live": {k: rel(v) for k, v in marks.items()}}
    rows_msgs = [(t, j) for t, j in m1 if "rows" in j]
    unsup = [(t, j) for t, j in m1 if j.get("unsupported")]
    rep["session1"].update(rows_msgs=len(rows_msgs), unsupported_msgs=len(unsup),
                           unsupported_at_rel_live=[rel(t) for t, _ in unsup])
    windows = [(marks["on0"], marks["off1"]), (marks["on2"], marks["off2"])]
    inside = lambda t, pad: any(a0 <= t <= b0 + pad for a0, b0 in windows)
    if a.expect == "supported":
        rows = [r for _, j in rows_msgs for r in j["rows"]]
        vt = {F: t for _, F, t in p1.frames}
        outside = [rel(t) for t, _ in rows_msgs if not inside(t, 0.4)]
        first_after_live = min((rel(t) for t, _ in rows_msgs), default=None)
        sw_done = next((t for t, j in p1.got if j.get("type") == "status" and j.get("stage") == "live"
                        and t > marks["switch"]), None)
        after_switch = [t for t, _ in rows_msgs if sw_done and sw_done <= t <= marks["off2"]]
        # cols ride only the first message after each "on" (the page's, and the hub's re-send after the switch's ready)
        # and then at most every 2 s; the page reads the others with the last list it saw (app/rigdebug.js onRender)
        last, cols_bad, cols_t = None, [], []
        for t, j in rows_msgs:
            if "cols" in j:
                last = j["cols"]
                cols_t.append(t)
            elif last is None:
                cols_bad.append(("rows before any cols", rel(t)))
            if last is not None and any(len(r) != 2 + len(last) for r in j["rows"]):
                cols_bad.append(("row length", rel(t)))
        firsts = {}
        for name, a0 in (("on0", marks["on0"]), ("on2", marks["on2"]), ("switch_live", sw_done)):
            nx = next(((t, j) for t, j in rows_msgs if a0 is not None and t >= a0), None)
            firsts[name] = None if nx is None else {"after_s": round(nx[0] - a0, 3), "has_cols": "cols" in nx[1],
                                                    "s_since_previous_cols": round(nx[0] - max([c for c in cols_t if c < nx[0]],
                                                                                           default=nx[0]), 3)}
        cgaps = [round(q - p, 3) for p, q in zip(cols_t, cols_t[1:])]
        ci = {c: i + 2 for i, c in enumerate(last)} if last else {}
        env = lambda t: float(np.sqrt(np.mean(pcm[max(0, int((t - .02) * SR)):int((t + .02) * SR)].astype(float) ** 2)) / 32768) \
            if int((t + .02) * SR) > 0 else 0.0
        sp = [r[ci["jaw"]] for r in rows if env(r[1]) > 0.01] if ci else []
        si = [r[ci["jaw"]] for r in rows if max(env(r[1] + d) for d in np.arange(-.3, .31, .05)) == 0.0] if ci else []
        rep["session1"].update(
            rows=len(rows), rows_msgs_outside_on_windows=outside, first_rows_rel_live=first_after_live,
            rows_msgs_after_switch_live=len(after_switch), switch_live_rel=rel(sw_done) if sw_done else None,
            rows_matching_a_received_V=sum(1 for r in rows if r[0] in vt),
            t_equal_V_header=all(vt[r[0]] == r[1] for r in rows if r[0] in vt),
            cols=len(ci), src=sorted({j.get("src") for _, j in rows_msgs}),
            cols_msgs=len(cols_t), msgs_without_cols=len(rows_msgs) - len(cols_t), cols_gaps_s=cgaps,
            page_rigdbg_bytes_per_s_while_on=round(sum(n for t, ty, n in p1.jbytes if ty == "rigdbg")
                                                   / sum(b0 - max(a0, marks["live"]) for a0, b0 in windows), 1),
            first_msg_after=firsts, cols_problems=cols_bad[:5],
            speech_rows=len(sp), speech_jaw_mean=round(float(np.mean(sp)), 3) if sp else None,
            silence_rows=len(si), silence_jaw_max=round(float(np.max(si)), 3) if si else None)
        s1 = rep["session1"]
        ok &= (len(rows) > (100 if a.renderer == "stub" else 40) and not outside      # rt on llvmpipe: ~5-8 fps here
               and first_after_live is not None and first_after_live < 1.0
               and not cols_bad and all(f and f["has_cols"] for f in firsts.values())
               and len(cols_t) < len(rows_msgs) / 4
               and len(after_switch) > 5 and s1["rows_matching_a_received_V"] >= 0.9 * len(rows)
               and s1["t_equal_V_header"] and s1["cols"] == 29 and not unsup
               and (s1["src"] == ["stub"]) == (a.renderer == "stub")
               and (s1["speech_jaw_mean"] or 0) > 0.2 and s1["silence_jaw_max"] == 0.0)
    else:
        n_opens = 2
        rep["session1"]["rows_total"] = sum(len(j.get("rows", [])) for _, j in rows_msgs)
        ok &= not rows_msgs and len(unsup) == n_opens and all(inside(t, 0.5) for t, _ in unsup)
    m2 = rigdbg_msgs(p2)
    rep["session2"] = {"rigdbg_msgs": len(m2), "frames": len(p2.frames)}
    ok &= not m2 and len(p2.frames) > (50 if a.renderer == "stub" else 20)
    # the renderer's side: how often it was told on / off (the stub logs it), errors
    if a.inst:
        with open(logp, "rb") as f:                   # only what this run appended
            f.seek(log_off)
            mine = f.read().decode(errors="replace").splitlines()
        rep["renderer_log"] = {"rigdbg_on": sum("rigdbg on" in ln for ln in mine),
                               "rigdbg_off": sum("rigdbg off" in ln for ln in mine),
                               "unknown_rigdbg_errors": sum("unknown message type 'rigdbg'" in ln for ln in mine)}
        recs = sorted(glob.glob(os.path.join(a.inst, "data", "session-*.json")), key=os.path.getmtime)
        recs = [r for r in recs if os.path.getmtime(r) >= t_start - 1]
        rep["session_records"] = []
        for r in recs:
            txt = open(r).read()
            d = json.loads(txt)
            rep["session_records"].append({"file": os.path.basename(r), "mentions_rigdbg": "rigdbg" in txt,
                                           "errors": d.get("errors"), "suppressed_render_errors":
                                           d.get("suppressed_render_errors")})
        ok &= len(recs) >= 2 and not any(x["mentions_rigdbg"] for x in rep["session_records"])
        ok &= all(not any("rigdbg" in str(e) for e in (x["errors"] or [])) for x in rep["session_records"])
        if a.renderer == "real":
            pass                                       # the real server logs nothing per message
        elif a.expect == "supported":
            # on0 (bring-up push) + on2 + the switch's ready re-send; off1 + off2 (+ teardown off only if still on)
            ok &= rep["renderer_log"]["rigdbg_on"] >= 3 and rep["renderer_log"]["unknown_rigdbg_errors"] == 0
        else:
            ok &= rep["renderer_log"]["unknown_rigdbg_errors"] == 1
    rep["ok"] = bool(ok)
    print(json.dumps(rep))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--inst", default="", help="the server's AVATAR_HOME (data/render-child.log, data/session-*.json)")
    ap.add_argument("--expect", default="supported", choices=["supported", "unsupported"])
    ap.add_argument("--switch-to", default="bear", help="the character session 1 switches to at live+6 s")
    ap.add_argument("--renderer", default="stub", choices=["stub", "real"],
                    help="real = avatar.render_server (no stub log lines: the renderer-side counts are skipped)")
    a = ap.parse_args()
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
