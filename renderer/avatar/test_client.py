#!/usr/bin/env python3
"""Test client for avatar/render_server.py: plays the app server's part on a real-time timeline and measures what comes back.

    RENDER_KEY=... python -m avatar.test_client --url ws://127.0.0.1:8770/render --character orb \\
        --wav audio_in/clip.wav --seconds 20 --out out/avatar-render/orb.mp4 --json out/avatar-render/orb.json

TIMELINE  Like the app server: every 20 ms of wall time it sends the next 20 ms of timeline audio ('A', sample-indexed): --lead-s
of digital silence, the WAV, --tail-s of silence.  --ahead-ms sends the WAV part that much EARLY (a vendor queue the
app server forwards ahead of its clock), silence always goes on its tick.  The timeline clock starts when the server says ready.

MEASURES
  delay   per frame: arrival - wall time the frame's own instant (sample F * 24000 / fps) was sent.  On a real-time
          timeline this is lookahead + chunking + queue + render + encode + transport: the end-to-end frame delay a
          playout delay D must cover.  late_at_D = frames that would miss display at base + t for D in --delays.
  server  render ms, encode ms, fps, late/dropped, from the server's stats messages; drops = frame numbers never received.
  sync    the frames are decoded (PyAV) and the mouth box is found from the video itself (blocks whose change tracks the
          animator's lip_open, recomputed here from the same audio); then (a) the lag that best aligns mouth-pixel change
          with lip_open (0 = every frame carries the instant it depicts) and (b) per syllable onset after silence, the
          first frame whose mouth opens (30 % of the local peak change, as live/latency.py) minus the onset: the
          displayed lip-sync offset (the rig leads by ~67 ms by design).
  mp4     decoded frames at their timeline instants (a missing frame repeats the previous one) + the timeline audio.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anim.schema import IDX, SAMPLE_RATE as SR    # noqa: E402

mono = time.monotonic
V_HDR = struct.Struct("<BBHId")


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, "need 24 kHz 16-bit"
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0].copy()
    return x


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def summ(x):
    x = [v for v in x if v is not None]
    return None if not x else {"mean": round(float(np.mean(x)), 1), "p50": pct(x, 50), "p95": pct(x, 95),
                               "min": round(float(np.min(x)), 1), "max": round(float(np.max(x)), 1), "n": len(x)}


async def run(a):
    import websockets
    key = os.environ.get(a.key_env, "")
    url = a.url + (("&" if "?" in a.url else "?") + "key=" + key if key else "")
    wav = read_wav(a.wav)
    if a.seconds:
        wav = wav[:int(a.seconds * SR)]
    lead, tail = int(a.lead_s * SR), int(a.tail_s * SR)
    timeline = np.concatenate([np.zeros(lead, np.int16), wav, np.zeros(tail, np.int16)])
    chunk = int(a.chunk_ms * SR / 1000)
    n_chunks = -(-timeline.size // chunk)
    sent_at = np.full(n_chunks, np.nan)
    frames = {}                 # F -> (arrival, flags, payload)
    stats, statuses = [], []
    ready = {}
    ev_ready = asyncio.Event()
    t_connect = mono()
    async with websockets.connect(url, max_size=2 ** 23, compression=None, open_timeout=20) as ws:
        start = {"type": "start", "character": a.character, "fps": a.fps, "codec": a.codec, "bitrate": a.bitrate,
                 "lookahead_ms": a.lookahead_ms, "backend": a.backend, "opts": json.loads(a.opts)}
        if a.width:
            start.update(width=a.width, height=a.height)
        await ws.send(json.dumps(start))

        async def receiver():
            async for m in ws:
                now = mono()
                if isinstance(m, (bytes, bytearray)):
                    if m[0] == 0x56:
                        _, flags, _, F, t = V_HDR.unpack_from(m, 0)
                        frames[F] = (now, flags, bytes(m[16:]))
                    continue
                msg = json.loads(m)
                msg["_at"] = round(now - t_connect, 3)
                if msg["type"] == "stats":
                    stats.append(msg)
                    if a.verbose:
                        print(json.dumps(msg), flush=True)
                elif msg["type"] == "ready":
                    ready.update(msg)
                    ev_ready.set()
                    print("ready", json.dumps(msg), flush=True)
                elif msg["type"] == "status":
                    statuses.append(msg)
                    print("status", json.dumps(msg), flush=True)
                elif msg["type"] == "error":
                    statuses.append(msg)
                    print("ERROR", msg.get("message"), flush=True)

        rx = asyncio.create_task(receiver())
        await asyncio.wait_for(ev_ready.wait(), timeout=a.ready_timeout)
        t_ready = mono() - t_connect
        # ---- the app server's clock: chunk c goes out at T0 + c * chunk / SR (or ahead_ms early for WAV audio)
        T0 = mono() + 0.05
        ahead = a.ahead_ms / 1000.0
        wav_lo, wav_hi = lead, lead + wav.size
        c = 0
        while c < n_chunks:
            now = mono()
            # everything whose send time has come: on-tick for silence, early for the WAV part with --ahead-ms
            while c < n_chunks:
                s0 = c * chunk
                tick = T0 + (s0 + chunk) / SR
                early = ahead if (wav_lo <= s0 < wav_hi) else 0.0
                if tick - early > now:
                    break
                pcm = timeline[s0:s0 + chunk]
                await ws.send(struct.pack("<B3xQ", 0x41, s0) + pcm.astype("<i2").tobytes())
                sent_at[c] = mono()
                c += 1
            if c < n_chunks:
                s0 = c * chunk
                early = ahead if (wav_lo <= s0 < wav_hi) else 0.0
                await asyncio.sleep(max(0.0, T0 + (s0 + chunk) / SR - early - mono()))
        await asyncio.sleep(a.drain_s)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(rx, timeout=3)
        except (asyncio.TimeoutError, Exception):
            rx.cancel()
    return dict(timeline=timeline, chunk=chunk, sent_at=sent_at, frames=frames, stats=stats, statuses=statuses,
                ready=ready, t_ready=t_ready, T0=T0, wav_samples=wav.size, lead=lead)


# ------------------------------------------------------------------------------------------------ analysis
def decode(frames, ready):
    out = {}
    errors = 0
    if ready.get("codec") == "jpeg":
        import cv2
        for F in sorted(frames):
            img = cv2.imdecode(np.frombuffer(frames[F][2], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                errors += 1
            else:
                out[F] = img[:, :, ::-1]
        return out, errors
    import av
    ctx = av.CodecContext.create("h264", "r")
    ctx.thread_count = 1
    for F in sorted(frames):
        try:
            got = ctx.decode(av.Packet(frames[F][2]))
        except Exception:
            errors += 1
            continue
        for fr in got:
            out[F] = fr.to_ndarray(format="rgb24")
    return out, errors


def lip_series(character, timeline, fps, lookahead_ms):
    from avatar.render_server import Core
    core = Core(character, lookahead_ms, fps)
    lip = {}
    for s in range(0, timeline.size, 480):
        for jb in core.push(timeline[s:s + 480]):
            lip[jb.j] = max(float(jb.params[IDX["lip_open"]]), float(jb.params[IDX["jaw"]]))
    return lip


def onsets_from_silence(x, min_sil_s=0.25, db=-40.0):
    env = np.sqrt(np.convolve(x.astype(float) ** 2, np.ones(24) / 24, "same"))
    loud = 20 * np.log10(env / 32768 + 1e-9) > db
    out, quiet = [], 0
    for i in range(0, x.size, 24):
        if loud[i]:
            if quiet >= min_sil_s * SR:
                out.append(i)
            quiet = 0
        else:
            quiet += 24
    return out


def sync_analysis(dec, lip, fps, timeline, out_png=None):
    Fs = sorted(F for F in dec if F in lip)
    if len(Fs) < 3 * fps:
        return {"error": "too few decoded frames"}
    import cv2
    h, w = dec[Fs[0]].shape[:2]
    sc = 4
    g = {F: cv2.resize(cv2.cvtColor(dec[F], cv2.COLOR_RGB2GRAY), (w // sc, h // sc), interpolation=cv2.INTER_AREA)
         .astype(np.float32) for F in Fs}
    L = np.array([lip[F] for F in Fs])
    closed = [F for F, v in zip(Fs, L) if v < 0.03] or Fs[:5]
    ref = np.median(np.stack([g[F] for F in closed[:200]]), axis=0)
    D = np.stack([np.abs(g[F] - ref) for F in Fs])            # (n, h/4, w/4)
    B = max(4, (w // sc) // 40)
    hb, wb = D.shape[1] // B, D.shape[2] // B
    Db = D[:, :hb * B, :wb * B].reshape(len(Fs), hb, B, wb, B).mean(axis=(2, 4))
    Lz = (L - L.mean()) / (L.std() + 1e-6)
    Dz = (Db - Db.mean(0)) / (Db.std(0) + 1e-6)
    corr = np.tensordot(Lz, Dz, axes=(0, 0)) / len(Fs)
    m = corr.max()
    ys, xs = np.nonzero(corr > 0.6 * m)
    box = [int(xs.min() * B * sc), int(ys.min() * B * sc), int((xs.max() + 1) * B * sc), int((ys.max() + 1) * B * sc)]
    sel = corr > 0.6 * m
    mouth = Db[:, sel].mean(axis=1)
    # (a) the lag that best aligns mouth change with the params of the same instant
    best = None
    for k in range(-4, 5):
        a_, b_ = (mouth[max(0, k):len(mouth) + min(0, k)], L[max(0, -k):len(L) + min(0, -k)])
        r = float(np.corrcoef(a_, b_)[0, 1])
        if best is None or r > best[1]:
            best = (k, r)
    lags = {k: round(float(np.corrcoef(mouth[max(0, k):len(mouth) + min(0, k)],
                                       L[max(0, -k):len(L) + min(0, -k)])[0, 1]), 3) for k in range(-3, 4)}
    # (b) syllable onsets after silence -> first frame whose mouth opens
    idx = {F: i for i, F in enumerate(Fs)}
    rows = []
    for s_on in onsets_from_silence(timeline):
        t_on = s_on / SR
        win = [F for F in range(int(np.ceil((t_on - 0.3) * fps)), int((t_on + 0.6) * fps) + 1) if F in idx]
        if len(win) < 0.8 * 0.9 * fps:
            continue
        v = np.array([mouth[idx[F]] for F in win])
        base = float(np.median(v[:3]))
        th = base + 0.3 * (v.max() - base)
        hit = next((F for F, x in zip(win, v) if x > th), None)
        if hit is not None and v.max() - base > 0.5:
            rows.append({"onset_ms": round(1000 * t_on, 1), "open_frame": hit,
                         "offset_ms": round(1000 * (hit / fps - t_on), 1), "peak": round(float(v.max() - base), 2)})
    if out_png:
        im = dec[Fs[int(np.argmax(L))]].copy()
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.imwrite(out_png, im[:, :, ::-1])
    offs = [r["offset_ms"] for r in rows]
    return {"mouth_box_xyxy": box, "mouth_corr": round(float(m), 3), "best_lag_frames": best[0],
            "best_lag_r": round(best[1], 3), "lag_r": lags, "onset_offset_ms": summ(offs),
            "onsets": rows, "note": "negative offset = mouth opens before the sound (the rig leads by ~67 ms by design)"}


def write_mp4(dec, frames, fps, timeline, path):
    if not dec:
        return None
    F0, F1 = min(frames), max(frames)
    h, w = next(iter(dec.values())).shape[:2]
    wav_path = os.path.splitext(path)[0] + ".timeline.wav"
    with wave.open(wav_path, "wb") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes(timeline.tobytes())
    # video frame k shows timeline instant k / fps; the audio starts at timeline 0, so pad from F = 0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-i", wav_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = np.zeros((h, w, 3), np.uint8)
    held = 0
    for F in range(0, F1 + 1):
        if F in dec:
            last = dec[F]
        else:
            held += 1
        p.stdin.write(np.ascontiguousarray(last).tobytes())
    p.stdin.close()
    p.wait()
    os.remove(wav_path)
    return {"path": path, "frames": F1 + 1, "held_frames": held}


def analyze(a, r):
    fps = a.fps
    frames, sent_at, chunk = r["frames"], r["sent_at"], r["chunk"]
    if not frames:
        return {"error": "no frames received", "statuses": r["statuses"]}
    Fs = sorted(frames)
    spf = SR // fps
    delay, delay_h = [], []
    look = int(a.lookahead_ms * SR / 1000)
    for F in Fs:
        c = (F * spf) // chunk
        if c < len(sent_at) and not np.isnan(sent_at[c]):
            delay.append(1000 * (frames[F][0] - sent_at[c]))
        ch = (F * spf + look) // chunk
        if ch < len(sent_at) and not np.isnan(sent_at[ch]):
            delay_h.append(1000 * (frames[F][0] - sent_at[ch]))
    # display simulation: the browser plays sample s at base + s/SR, base = arrival of sample 0 + D
    late_at = {}
    for D in a.delays:
        base = sent_at[0] + D / 1000.0
        late_at[str(D)] = int(sum(frames[F][0] > base + F / fps for F in Fs))
    total_F = r["timeline"].size // spf
    first_F = Fs[0]
    expected = [F for F in range(first_F, min(total_F, Fs[-1] + 1))]
    missing = [F for F in expected if F not in frames]
    st = [s for s in r["stats"] if s.get("fps")]
    srv = {k: summ([s.get(k) for s in st]) for k in ("fps", "render_ms", "render_p95", "encode_ms", "lag_ms",
                                                      "service_ms", "kbps", "audio_ahead_ms")}
    srv["late_total"] = int(sum(s.get("late", 0) for s in r["stats"]))
    srv["dropped_total"] = int(sum(s.get("dropped", 0) for s in r["stats"]))
    srv["late_sent_total"] = int(sum(s.get("late_sent", 0) for s in r["stats"]))
    arr = np.array([frames[F][0] for F in Fs])
    span = arr[-1] - arr[0] if len(arr) > 1 else 0
    res = {"character": a.character, "ready": r["ready"], "ready_s": round(r["t_ready"], 2),
           "frames_received": len(Fs), "first_frame": first_F, "last_frame": Fs[-1],
           "missing_frames": len(missing), "missing_pct": round(100 * len(missing) / max(1, len(expected)), 2),
           "fps_received": round((len(Fs) - 1) / span, 2) if span else None,
           "keyframes": int(sum(frames[F][1] & 1 for F in Fs)),
           "kbps_payload": round(8 * sum(len(frames[F][2]) for F in Fs) / max(span, 1e-3) / 1000, 1),
           "delay_ms_frame_instant": summ(delay), "delay_ms_after_lookahead_audio": summ(delay_h),
           "late_at_delay_ms": late_at, "server": srv, "ahead_ms": a.ahead_ms, "seconds": a.seconds,
           "lead_s": a.lead_s, "wav": a.wav}
    if not a.no_decode:
        t = time.time()
        dec, errs = decode(frames, r["ready"])
        res["decode"] = {"frames": len(dec), "errors": errs, "s": round(time.time() - t, 1)}
        if a.out:
            res["mp4"] = write_mp4(dec, frames, fps, r["timeline"], a.out)
        if not a.no_sync:
            lip = lip_series(a.character, r["timeline"], fps, a.lookahead_ms)
            png = os.path.splitext(a.out)[0] + "_mouthbox.png" if a.out else None
            res["sync"] = sync_analysis(dec, lip, fps, r["timeline"], png)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8770/render")
    ap.add_argument("--key-env", default="RENDER_KEY")
    ap.add_argument("--character", default="orb")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--codec", default="h264", choices=["h264", "jpeg"])
    ap.add_argument("--bitrate", type=int, default=2_000_000)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--opts", default="{}", help="JSON backend opts")
    ap.add_argument("--lookahead-ms", type=float, default=133)
    ap.add_argument("--wav", default=os.path.join(ROOT, "audio_in", "clip.wav"))
    ap.add_argument("--seconds", type=float, default=20.0, help="seconds of the WAV to stream (0 = all)")
    ap.add_argument("--lead-s", type=float, default=2.0, help="silence before the WAV (idle life)")
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--ahead-ms", type=float, default=0.0, help="send the WAV part this much ahead of the clock")
    ap.add_argument("--drain-s", type=float, default=1.0)
    ap.add_argument("--ready-timeout", type=float, default=240.0)
    ap.add_argument("--delays", type=float, nargs="*", default=[150, 200, 250, 300, 400])
    ap.add_argument("--out", default="", help="MP4 of the decoded frames + audio")
    ap.add_argument("--json", default="", help="write the measurements here")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    r = asyncio.run(run(a))
    res = analyze(a, r)
    res["statuses"] = r["statuses"]
    txt = json.dumps(res, indent=1)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        open(a.json, "w").write(txt)
    brief = {k: v for k, v in res.items() if k not in ("statuses", "ready")}
    if "sync" in brief and isinstance(brief["sync"], dict):
        brief["sync"] = {k: v for k, v in brief["sync"].items() if k != "onsets"}
    print(json.dumps(brief, indent=1))
    return res


if __name__ == "__main__":
    main()
