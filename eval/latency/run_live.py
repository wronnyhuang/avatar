"""Latency: live-baseline (server-voice) page runner. Derived from an earlier real-input page test; adds the clock
mapping the latency tools need: every 150 ms it logs the page's S.userEndCt (mic capture time of the last user speech,
AudioContext clock), S.base (AudioContext time at which playback-timeline 0 is heard), outputLatency and currentTime,
and it can inject LOCAL commands (the page's own transcript matcher, window.__avatar.interactive.transcriptAction)
at fixed offsets after 'live' (--inject "6:turn your head to the left,26:look up"). Everything lands in
<capture>/clock.json; eval/latency/live_capture.py turns capture + clock into the avlat inputs.

Original notes:
Drive the real page in a headless Chromium of our own (playwright) with a fake microphone fed from a WAV file.

    python eval/latency/run_live.py --url 'http://127.0.0.1:<port>/?k=<key>' --wav dialog48k.wav --seconds 40 --out res.json \
        [--shots DIR] [--codec jpeg] [--echo-guard]

Chrome's fake media flags feed the WAV (looped) as the mic. Its own browser, no display (headless).
Collects the page's own stats messages (window.__avatar.stats()) every 2 s, screenshots of the video canvas, and the
server's session summary from api/stats at the end.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import struct
import wave
from pathlib import Path
import urllib.parse
import urllib.request
import shutil
import sys

from playwright.async_api import async_playwright


def _ulaw(i):
    u = ~i & 0xFF
    v = ((((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 7)) - 0x84
    return struct.pack("<h", -v if u & 0x80 else v)


ULAW16 = [_ulaw(i) for i in range(256)]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--y4m", required=True, help="real talking-person video for the fake front camera")
    ap.add_argument("--mobile", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--record-video", help="destination .webm path for the recorded browser screen")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--character", default=None)
    ap.add_argument("--out")
    ap.add_argument("--shots")
    ap.add_argument("--codec")
    ap.add_argument("--capture", help="save exact returned PCM and timestamped video")
    ap.add_argument("--echo-guard", action="store_true")
    ap.add_argument("--switch-to", default=None, help="switch character mid-session")
    ap.add_argument("--channel", default=None, help="e.g. chrome (an own headless instance of the installed binary)")
    ap.add_argument("--clean-video", action="store_true", help="hide debug HUD in the recorded user-facing proof")
    ap.add_argument("--inject", default="", help="'sec:phrase,sec:phrase' local transcript commands after live")
    a = ap.parse_args()
    url = a.url + ("&" if "?" in a.url else "?") + ("hud=0" if a.clean_video else "hud=1") + (f"&codec={a.codec}" if a.codec else "")
    args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={os.path.abspath(a.y4m)}",
            f"--use-file-for-fake-audio-capture={os.path.abspath(a.wav)}", "--autoplay-policy=no-user-gesture-required",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding"]
    samples, console = [], []
    capture_batches = {}
    capture_tasks = []
    network = {"bytes": 0, "requests": 0, "blocked": [], "failed": []}
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=args, **({"channel": a.channel} if a.channel else {}))
        size = {"width": 412, "height": 915} if a.mobile else {"width": 960, "height": 900}
        ctx = await b.new_context(viewport=size, device_scale_factor=1 if a.record_video else 2,
                                  is_mobile=a.mobile, has_touch=a.mobile,
                                  record_video_dir=os.path.dirname(a.record_video) if a.record_video else None,
                                  record_video_size=size if a.record_video else None,
                                  permissions=["microphone", "camera"])
        origin = urllib.parse.urlsplit(url).netloc
        async def route(r):
            target = urllib.parse.urlsplit(r.request.url)
            if target.scheme in ('http','https') and target.netloc != origin:
                network["blocked"].append(target.netloc + target.path)
                await r.abort()
            else:
                await r.continue_()
        await ctx.route("**/*", route)
        pg = await ctx.new_page()
        def completed(req):
            async def sizes():
                try:
                    z = await req.sizes()
                    network["bytes"] += z.get("responseBodySize",0) + z.get("responseHeadersSize",0)
                    network["requests"] += 1
                except Exception:
                    pass
            asyncio.create_task(sizes())
        pg.on("requestfinished", completed)
        async def capture_response(r):
            if not a.capture or not urllib.parse.urlsplit(r.url).path.endswith(("/api/exchange", "/api/down")) or r.status != 200:
                return
            try:
                body = await r.body()
                if len(body) >= 4:
                    seq = struct.unpack_from('<I', body)[0]
                    if seq:
                        capture_batches[seq] = body[4:]
            except Exception:
                pass
        tap = {}                      # 'P' voice-tap chunks the page uploads (direct-voice page): s0 -> int16 bytes

        def got_request(rq):
            if not a.capture or not urllib.parse.urlsplit(rq.url).path.endswith(("/api/exchange", "/api/up")):
                return
            try:
                body = rq.post_data_buffer or b""
                off = 0
                while off + 4 <= len(body):
                    ln = struct.unpack_from('<I', body, off)[0]; off += 4
                    msg = body[off:off + ln]; off += ln
                    if len(msg) >= 16 and msg[0] == 0x50 and not msg[1] & 1:
                        s0, n = struct.unpack_from('<QI', msg, 4)
                        pay = msg[16:]
                        if msg[1] & 2:                                  # mu-law (no numpy in this venv)
                            pay = b"".join(ULAW16[b] for b in pay[:n])
                        tap[s0] = pay[:2 * n]
            except Exception:
                pass
        pg.on("request", got_request)

        def got_response(r):
            capture_tasks.append(asyncio.create_task(capture_response(r)))
        pg.on("response", got_response)
        pg.on("requestfailed", lambda req: network["failed"].append(urllib.parse.urlsplit(req.url).path))
        pg.on("console", lambda m: console.append(f"{m.type}: {m.text}"[:300]))
        pg.on("pageerror", lambda e: console.append(f"PAGEERROR: {e}"[:300]))
        await pg.goto(url, wait_until="load")
        await asyncio.sleep(1.5)
        if a.character:
            await pg.click(f".tile[data-k='{a.character}']")
            await asyncio.sleep(0.5)
        if a.echo_guard:
            await pg.check("#eg")
        if a.no_camera:
            await pg.uncheck("#camon")
        await pg.click("#go")
        t0 = time.monotonic()
        live_at = None
        switched = False
        shot_n = 0
        clock = []
        # --inject "SEC:PHRASE,..." (live page transcript matcher) or "SEC|REGION|{json command}" / "SEC|REGION|event:NAME"
        # separated by ";" (integ page: window.__avatar.command / window.__avatar.event)
        if "|" in a.inject:
            injects = [(float(x.split("|", 2)[0]), x.split("|", 2)[2], x.split("|", 2)[1]) for x in a.inject.split(";") if x.count("|") >= 2]
        else:
            injects = [(float(x.split(":", 1)[0]), x.split(":", 1)[1], None) for x in a.inject.split(",") if ":" in x]
        injected = []
        CLK = ("() => { const S = window.__avatar.S; const c = S.ctx; return {perf: performance.now(), ct: c ? c.currentTime : null,"
               " base: S.base, ue: S.userEndCt, ol: c ? (c.outputLatency || c.baseLatency || 0) : null, stage: S.stage, D: S.D,"
               " rs: S.renderStats || null}; }")
        async def clock_loop():
            while True:
                try:
                    v = await pg.evaluate(CLK)
                    v["wall"] = time.monotonic() - t0
                    clock.append(v)
                    if live_at is not None:
                        for k, (off, phrase, region) in enumerate(injects):
                            if k not in [i for i, *_ in injected] and time.monotonic() - live_at >= off:
                                r = await pg.evaluate("(p) => { const A = window.__avatar, S = A.S; const c = S.ctx; const ct = c.currentTime;"
                                                      " if (p.startsWith('{')) A.command(JSON.parse(p), 'ui');"
                                                      " else if (p.startsWith('tool:')) { const i = p.indexOf(':', 5); A.command(toolToCommand(p.slice(5, i), JSON.parse(p.slice(i + 1))), 'ui'); }"
                                                      " else if (p.startsWith('event:')) A.event(p.slice(6), {test: true});"
                                                      " else A.interactive.transcriptAction(' ' + p + ' ');"
                                                      " return {perf: performance.now(), ct, base: S.base, ol: c.outputLatency || c.baseLatency || 0}; }", phrase)
                                r["region"] = region
                                injected.append((k, phrase, r))
                except Exception:
                    pass
                await asyncio.sleep(0.15)
        clock_task = asyncio.create_task(clock_loop())
        while time.monotonic() - t0 < a.seconds + 300:
            await asyncio.sleep(2)
            st = await pg.evaluate("() => ({sid: window.__avatar.S.sid, stats: window.__avatar.stats(), status: (document.getElementById('status') || {}).textContent || '',"
                                   " err: (document.getElementById('err') || {}).textContent || '', live: window.__avatar.S.live,"
                                   " stage: window.__avatar.S.stage, caps: (document.getElementById('caps') || {}).innerText || '',"
                                   " perception: window.__avatar.interactive ? window.__avatar.interactive.measurements() : null})")
            st["t"] = round(time.monotonic() - t0, 1)
            samples.append(st)
            print(st["t"], st["stage"], st["status"][:70], "|", (st["stats"] or {}).get("fps"), "fps",
                  (st["stats"] or {}).get("sync_ms"), flush=True)
            if st["stage"] == "live" and live_at is None:
                live_at = time.monotonic()
            if a.shots and live_at and shot_n < 4 and time.monotonic() - live_at > 4 + shot_n * 8:
                os.makedirs(a.shots, exist_ok=True)
                await pg.locator("#stage").screenshot(path=os.path.join(a.shots, f"stage{shot_n}.png"))
                if shot_n == 0:
                    await pg.screenshot(path=os.path.join(a.shots, "page.png"), full_page=True)
                shot_n += 1
            if a.switch_to and live_at and not switched and time.monotonic() - live_at > a.seconds / 2:
                await pg.click(f".tile[data-k='{a.switch_to}']")
                switched = True
            if not st["live"] and live_at:
                break
            if live_at and time.monotonic() - live_at > a.seconds:
                await pg.click("#go")           # Stop
                await asyncio.sleep(2)
                break
        clock_task.cancel()
        final = await pg.evaluate("() => ({status: document.getElementById('status').textContent})")
        network["elapsed_s"] = round(time.monotonic()-t0,2)
        network["kbps"] = round(network["bytes"]*8/max(1,network["elapsed_s"])/1000,1)
        if capture_tasks:
            await asyncio.gather(*capture_tasks, return_exceptions=True)
        await ctx.close()
        if a.record_video and pg.video:
            shutil.copyfile(await pg.video.path(), a.record_video)
        await b.close()
    if a.capture:
        dest = Path(a.capture); dest.mkdir(parents=True, exist_ok=True)
        audio, video, events = {}, {}, []
        for seq, body in sorted(capture_batches.items()):
            off = 0
            while off + 4 <= len(body):
                ln = struct.unpack_from('<I',body,off)[0]; off += 4
                msg=body[off:off+ln];off += ln
                if not msg: continue
                if msg[0] == 65:
                    audio[struct.unpack_from('<Q',msg,4)[0]]=msg[12:]
                elif msg[0] == 86:
                    n=struct.unpack_from('<I',msg,4)[0]; t=struct.unpack_from('<d',msg,8)[0]
                    video[n]={'n':n,'t':t,'flags':msg[1],'data':msg[16:]}
                elif msg[0] == 74:
                    try: events.append(json.loads(msg[1:]))
                    except Exception: pass
        end=max((k+len(v)//2 for k,v in audio.items()),default=0)
        pcm=bytearray(end*2)
        for k,v in audio.items():pcm[k*2:k*2+len(v)]=v
        with wave.open(str(dest/'output.wav'),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
        vv=[v for k,v in sorted(video.items())]
        (dest/'output.h264').write_bytes(b''.join(v['data'] for v in vv if not v['flags']&2))
        for v in vv:
            if v['flags']&2:(dest/f"frame-{v['n']:06d}.jpg").write_bytes(v['data'])
        (dest/'timeline.json').write_text(json.dumps({'video':[{k:v[k] for k in ('n','t','flags')} for v in vv], 'audio_samples':end,'events':events},indent=1))
        if tap:
            tend = max(k + len(v) // 2 for k, v in tap.items())
            tp = bytearray(tend * 2)
            for k, v in tap.items():
                tp[k * 2:k * 2 + len(v)] = v
            with wave.open(str(dest/'tap.wav'), 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(tp)
        (dest/'clock.json').write_text(json.dumps({'clock': clock, 'injected': [{'phrase': p, **r} for _, p, r in injected]}, indent=1))
        print('capture',len(vv),'video frames;',round(end/24000,2),'audio seconds')
    # the server's own record of the session
    q = urllib.parse.urlsplit(a.url)
    k = urllib.parse.parse_qs(q.query)["k"][0]
    base = f"{q.scheme}://{q.netloc}{q.path if q.path.endswith('/') else q.path + '/'}"
    run_sid = next((s.get("sid") for s in samples if s.get("sid")), None)
    hub = {}
    session = None
    for _ in range(8):
        stats_req = urllib.request.Request(base + "api/stats?k=" + urllib.parse.quote(k),
                                           headers={"x-requested-with": "avatar-live"})
        hub = json.loads(urllib.request.urlopen(stats_req, timeout=10).read())
        # Session summaries intentionally shorten the browser's session id to 8
        # characters; match that stable prefix instead of the full private id.
        session = next((s for s in hub.get("sessions") or [] if run_sid and s.get("sid") == run_sid[:8]), None)
        if session:
            break
        time.sleep(1)
    out = {"network": network, "samples": samples, "final": final, "console": console[-40:], "live_at_s": round(live_at-t0,2) if live_at else None,
           "run_sid": run_sid, "hub_session": session,
           "spend": hub.get("spend")}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    hs = out["hub_session"] or {}
    print(json.dumps({k: hs.get(k) for k in ("end_reason", "first_audio_ms", "first_audio_each", "sync_offset_ms_p50",
                                            "sync_offset_abs_p95_ms", "fps_shown", "poll_rtt_ms_p50", "poll_rtt_ms_p95",
                                            "late_frames", "dropped_frames", "audio_underruns", "usd", "video", "codec",
                                            "hub_user_end_to_vendor_ms")}, indent=1))
    print("console:", "\n".join(console[-15:]))


if __name__ == "__main__":
    asyncio.run(main())
