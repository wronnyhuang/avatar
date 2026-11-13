#!/usr/bin/env python3
"""The real-time loop: audio stream in -> StreamingLipAnimator (strictly causal) -> any backend, N render processes ->
frames out live (HTTP page / window) + a recording, with the audio delayed by the same D so lips and voice stay in sync.

    python -m live.run_live --character orb --backend v2 --opt backend=gpu --opt spp=1 --wav audio_in/clip.wav \\
        --record out.mp4 --log timing.json                      # then open the printed http://127.0.0.1:8765/
    python -m live.run_live --character fox --backend rt --mic --record out.mp4
    parec --raw --rate=24000 --channels=1 --format=s16le | python -m live.run_live --character orb --stdin

SCHEDULE.  T0 = the wall time of stream sample 0 (the wav: the moment feeding starts; mic/stdin: first chunk's arrival
minus its length, lowered to the earliest any chunk arrived until the first frame goes out).  Output frame j (time
u_j = j / fps) is presented at T0 + u_j + D and audio sample tau is played (handed to every sink) at T0 + tau + D, so
lips and voice agree by construction.  D is --delay-ms, or 'auto': lookahead + chunk + animator p99 + render p95 (the
pool's own concurrent warm-up on speech rows, at least 1.25 x its mean) + 8 ms.  A frame not back by its tick is dropped (the previous frame repeats) and logged
with the reason; a job not expected to finish before its tick (75th percentile of recent service times) is skipped, so
the loop cannot fall behind -- except that after --probe-after (0.15 s) with nothing back, an idle pool takes the
newest job anyway (a probe), so the estimate tracks a renderer that slows down and speeds up again (a shared GPU).

END-TO-END LATENCY (as logged): for frame j, from the arrival in this process of the chunk carrying sample u_j to the
moment frame j and that sample's playback go out (T0 + u_j + D): D minus the time that sample waited for its chunk to
fill, i.e. between D - chunk and D.  The 'acoustic' latency (sound made -> out) is D itself.  Budget per frame, summing
to D: lookahead + chunking (wait for the chunk with the last sample the frame may read) + animator + queue + ipc +
render + return + slack.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anim.schema import IDX, SAMPLE_RATE as SR     # noqa: E402
from live.core import LiveCore                     # noqa: E402
from live.pool import RenderPool                   # noqa: E402
from live.sources import MicSource, StdinSource, WavSource   # noqa: E402

mono = time.monotonic


def parse_opt(kv):
    k, _, v = kv.partition("=")
    for f in (int, float):
        try:
            return k, f(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    return k, v


def machine():
    m = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}
    try:
        m["gpu_apps"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        m["gpu_util"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return m


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def stats(x):
    x = np.asarray(x, float)
    if not x.size:
        return None
    return {"mean": round(float(x.mean()), 2), "p50": round(pct(x, 50), 2), "p95": round(pct(x, 95), 2),
            "min": round(float(x.min()), 2), "max": round(float(x.max()), 2)}


class LiveLoop:
    def __init__(self, a):
        self.a = a
        self.fps = a.fps
        self.lock = threading.Condition()
        self.stop = threading.Event()
        self.T0 = None
        self.t0_frozen = False
        self.D = None
        self.pending = collections.deque()
        self.recs = {}                  # j -> per-frame record (times are monotonic, made relative to T0 at the end)
        self.frames = {}                # j -> rendered frame awaiting its tick
        self.audio_q = collections.deque()      # (start sample, pcm) awaiting play-out
        self.chunk_end, self.chunk_arr = [], []  # per received chunk: end sample, arrival
        self.last_j = None              # set at end of stream
        self.service = collections.deque(maxlen=64)  # (back at, dispatch -> back), s
        self.anim_ms = []
        self.late_audio = 0
        self.late_audio_out = 0
        self.played = 0
        self.ticks = []                 # (j, presented job or None, actual time)
        self.sinks_ms = []
        self.last_back = 0.0            # when the last rendered frame came back

    # ------------------------------------------------------------------ setup
    def make_source(self):
        a = self.a
        if a.wav:
            return WavSource(a.wav, a.chunk_ms)
        if a.mic:
            return MicSource(a.chunk_ms, device=a.mic_device, latency_ms=a.mic_latency_ms)
        src = StdinSource(a.chunk_ms)
        src.backlog = a.stdin_backlog
        return src

    def measure_animator(self, src):
        """Per-chunk cost of the animator (ms), and speech rows for the render warm-up: 3 s of the tree's own
        audio_in/clip.wav from 1 s on, whatever the source (a rest pose can render faster than a talking one, and the
        live source may open on silence), in this run's chunk size."""
        core = LiveCore(self.a.character, self.a.lookahead, self.fps, self.a.resample)
        n = src.chunk
        from live.sources import load_wav
        try:
            x = load_wav(os.path.join(ROOT, "audio_in", "clip.wav"))[SR:4 * SR]
        except Exception:
            x = (np.random.default_rng(1).normal(0, 3000, 3 * SR)).astype(np.int16)
        ts, rows = [], []
        for i in range(0, x.size, n):
            t = mono(); jobs = core.push(x[i:i + n]); ts.append(1000 * (mono() - t))
            rows += [j.params for j in jobs]
        rows.sort(key=lambda p: -float(p[IDX["lip_open"]] + p[IDX["jaw"]]))
        return ts[5:], rows[:12]

    def plan_delay(self, warm, anim_ts, core):
        look_ms = 1000 * core.H / SR
        render = [t for w in warm.values() for t in w["warmup_ms"][1:]] or [t for w in warm.values() for t in w["warmup_ms"]]
        # p95 of the concurrent warm-up, but never under 1.25 x its mean: a short warm-up underestimates the tail
        r95 = max(pct(render, 95), 1.25 * float(np.mean(render))) if render else 0.0
        a99 = pct(anim_ts, 99) if anim_ts else 3.0
        extra = 25.0 if (self.fps == 24 and self.a.resample == "linear") else 0.0
        jitter = self.a.mic_latency_ms if self.a.mic else (10.0 if self.a.stdin else 0.0)
        parts = {"lookahead": look_ms, "resample_extra": extra, "chunk": float(self.a.chunk_ms), "source_jitter": jitter,
                 "animator_p99": a99, "render_p95": r95, "ipc": 2.0, "margin": 8.0}
        return sum(parts.values()), {k: round(v, 1) for k, v in parts.items()}, r95

    # ------------------------------------------------------------------ scheduling
    def deadline(self, j):
        return self.T0 + j / self.fps + self.D

    def r_est(self):
        """Expected dispatch -> back time: the 75th percentile of the service times that came back in the last 0.5 s
        (at least the last 3), so the estimate forgets a slow spell within half a second of its end.  A render that
        would likely miss its tick is not started; the worker takes a later frame that can make it."""
        if not self.service:
            return self.r_warm
        now = mono()
        v = [d for t, d in self.service if now - t < 0.5] or [d for t, d in list(self.service)[-3:]]
        if len(v) < 3:
            v = [d for t, d in list(self.service)[-3:]]
        return float(np.percentile(v, 75))

    def dispatch(self, now):
        """Lock held.  Oldest viable job first, to an idle worker.  Jobs that cannot make their tick are skipped."""
        if self.T0 is None:
            return
        while self.pending:
            job = self.pending[0]
            dl = self.deadline(job.j)
            if dl - now > self.a.max_ahead:     # a faster-than-real-time source: do not render far ahead
                return
            probe = False
            if now + self.r_est() > dl:
                # not expected to make its tick.  If nothing has come back for --probe-after s, the newest job still
                # goes to an idle pool, as a probe: the estimate stays current, so a pool slowed by a GPU neighbour
                # recovers when the neighbour leaves instead of skipping forever on a stale estimate
                if (len(self.pending) == 1 and now < dl and now - self.last_back > self.a.probe_after
                        and len(self.pool.idle()) == self.pool.n):
                    probe = True
                else:
                    self.pending.popleft()
                    self.recs[job.j]["status"] = "skipped"
                    continue
            wid = self.pool.next_idle()
            if wid is None:
                return
            self.pending.popleft()
            r = self.recs[job.j]
            r["dispatch"], r["worker"] = mono(), wid
            if probe:
                r["probe"] = True
            self.pool.submit(wid, job.j, job.index, job.params)

    def collector(self):
        while not self.stop.is_set():
            try:
                res = self.pool.poll(timeout=0.02)
            except RuntimeError as e:
                print(f"[live] {e}", file=sys.stderr)
                self.error = str(e)
                self.stop.set()
                with self.lock:
                    self.lock.notify_all()
                return
            now = mono()
            with self.lock:
                for wid, j, frame, t0, t1, ts in res:
                    r = self.recs[j]
                    r.update(render_start=t0, render_end=t1, sent=ts, received=now)
                    self.service.append((now, now - r["dispatch"]))
                    self.last_back = now
                    if r.get("ticked"):
                        r["status"] = "late"            # its tick already went by with the previous frame
                    else:
                        self.frames[j] = frame
                self.dispatch(now)

    def presenter(self):
        j = 0
        last = None
        last_job = None
        while True:
            with self.lock:
                while self.T0 is None and not self.stop.is_set():
                    self.lock.wait(0.05)
                if self.stop.is_set() and self.a.hard_stop:
                    return
                tf = self.deadline(j)
                ta = self.T0 + self.audio_q[0][0] / SR + self.D if self.audio_q else None
                done = self.last_j is not None and j > self.last_j and not self.audio_q
            if done:
                return
            nxt = tf if ta is None else min(tf, ta)
            d = nxt - mono()
            if d > 0:
                time.sleep(min(d, 0.01))
                continue
            now = mono()
            if not self.t0_frozen:
                self.t0_frozen = True
            # audio first: everything due
            out_audio = []
            with self.lock:
                while self.audio_q and self.T0 + self.audio_q[0][0] / SR + self.D <= now + 0.0005:
                    out_audio.append(self.audio_q.popleft())
            for s0, pcm in out_audio:
                lateness = now - (self.T0 + s0 / SR + self.D)
                if lateness > 0.005:
                    self.late_audio_out += 1        # handed out > 5 ms after its slot (presenter held up)
                for sk in self.sinks:
                    sk.audio(s0 / SR, pcm)
                self.played += pcm.size
            if now + 0.0005 < tf:
                continue
            if self.last_j is not None and j > self.last_j:
                j += 1
                continue
            with self.lock:
                frame = self.frames.pop(j, None)
                r = self.recs.get(j)
                if r is not None:
                    r["ticked"] = True
                    r["tick"] = tf
                    if frame is not None:
                        r["status"] = "presented"
                    elif "status" not in r:
                        r["status"] = "late" if "dispatch" in r else "queued"
                for k in [k for k in self.frames if k < j]:     # never show an old frame after a newer one
                    self.frames.pop(k)
            if frame is not None:
                last, last_job = frame, j
            t_out = mono()
            if last is not None:
                for sk in self.sinks:
                    sk.frame(j / self.fps, last, last_job)
            elif self.blank is not None:
                for sk in self.sinks:
                    sk.frame(j / self.fps, self.blank, -1)
            self.sinks_ms.append(1000 * (mono() - t_out))
            if r is not None:
                r["present"] = t_out
                r["shown"] = last_job
            else:
                with self.lock:
                    self.recs[j] = {"j": j, "status": "no_params", "present": t_out, "tick": tf, "shown": last_job,
                                    "ticked": True}
            self.ticks.append((j, last_job, t_out))
            j += 1
            with self.lock:         # jobs held back by --max-ahead become due as the clock moves
                if self.pending:
                    self.dispatch(mono())

    # ------------------------------------------------------------------ run
    def run(self):
        a = self.a
        self.error = None
        self.m_start = machine()
        src = self.make_source()
        self.src = src
        core = LiveCore(a.character, a.lookahead, self.fps, a.resample)
        self.core = core
        anim_ts, warm_rows = self.measure_animator(src)
        opts = dict(parse_opt(o) for o in a.opt)
        self.pool = RenderPool(a.backend, a.character, a.width, a.height, a.procs, opts, pin=a.pin)
        t = mono()
        warm = self.pool.start(warmup=a.warmup, warm_params=warm_rows)
        self.warm = warm
        print(f"[live] {a.procs} x {a.backend} {a.character} {a.width}x{a.height} up in {mono() - t:.1f}s; warm-up ms "
              + "; ".join(f"w{k}: " + ",".join(f"{x:.0f}" for x in v["warmup_ms"]) for k, v in sorted(warm.items())),
              flush=True)
        D, parts, r95 = self.plan_delay(warm, anim_ts, core)
        self.r_warm = r95 / 1000.0
        self.D_auto, self.D_parts = D, parts
        self.D = (D if a.delay_ms == "auto" else float(a.delay_ms)) / 1000.0
        cap = a.procs * 1000.0 / r95 if r95 > 0 else float("inf")
        print(f"[live] D = {1000 * self.D:.0f} ms ({'auto' if a.delay_ms == 'auto' else 'fixed'}; auto would be "
              f"{D:.0f} = {parts}); pool capacity ~{cap:.1f} fps vs {self.fps} wanted", flush=True)

        self.sinks = []
        self.blank = np.zeros((a.height, a.width, 3), np.uint8)
        self.view = self.rec = self.win = None
        if a.port >= 0 and not a.no_view:
            from live.http_view import LiveView
            self.view = LiveView(a.width, a.height, self.fps, SR, port=a.port,
                                 info={"character": a.character, "backend": a.backend, "delay_ms": round(1000 * self.D),
                                       "procs": a.procs, "lookahead_ms": round(1000 * core.H / SR, 1)})
            print(f"[live] view: {self.view.url}", flush=True)
            self.sinks.append(_ViewSink(self.view))
        if a.record:
            from live.outputs import Recorder
            self.rec = Recorder(a.record, a.width, a.height, self.fps)
            self.sinks.append(_RecSink(self.rec))
        if a.window:
            from live.outputs import Window
            self.win = Window(a.width, a.height, self.fps, display=a.display, audio=not a.window_mute,
                              title=f"{a.character} live")
            self.sinks.append(_RecSink(self.win))

        th_c = threading.Thread(target=self.collector, name="collector", daemon=True)
        th_p = threading.Thread(target=self.presenter, name="presenter", daemon=True)
        th_c.start(); th_p.start()
        if a.wait_clients and self.view:
            print(f"[live] waiting for {a.wait_clients} viewer(s) at {self.view.url}", flush=True)
            while self.view.clients() < a.wait_clients and not self.stop.is_set():
                time.sleep(0.1)
        old = signal.signal(signal.SIGINT, lambda *_: (self.stop.set(), src.close()))
        cursor = 0
        t_end = None if not a.seconds else mono() + a.seconds
        try:
            for pcm, t_arr in self.reader(src):
                n = pcm.size
                with self.lock:
                    self.audio_q.append((cursor, pcm))
                    self.chunk_end.append(cursor + n); self.chunk_arr.append(t_arr)
                    if t_arr > self.T0 + cursor / SR + self.D:
                        self.late_audio += 1            # arrived after its own play-out slot: D too small for the source
                    self.lock.notify_all()
                cursor += n
                t1 = mono()
                jobs = core.push(pcm)
                t2 = mono()
                self.anim_ms.append(1000 * (t2 - t1))
                self._enqueue(jobs, t2)
                if self.stop.is_set() or (t_end and mono() > t_end):
                    break
        finally:
            src.close()
            signal.signal(signal.SIGINT, old)
        jobs = core.flush()
        self._enqueue(jobs, mono())
        with self.lock:
            self.last_j = core.next_j - 1
            self.lock.notify_all()
        self.n_samples = cursor
        th_p.join()
        self.stop.set()
        th_c.join(timeout=5)
        self.pool.close()
        self.m_end = machine()
        out = {}
        if self.rec:
            out["record"] = self.rec.close()
        if self.view:
            out["view"] = self.view.stats(); self.view.close()
        if self.win:
            self.win.close()
        return self.report(out)

    def reader(self, src):
        """The source is read on its own thread, so a chunk's arrival time is when it reached the process, not when
        the animator got round to it, and a backlog (audio a pipe buffered while the pool warmed up) drains in
        milliseconds -- which is what lets T0 settle on the true stream origin before the first frame goes out."""
        import queue
        q = queue.Queue()

        def run():
            cur = 0
            try:
                for item in src:
                    pcm, t_arr = item
                    n = pcm.size
                    with self.lock:     # T0 = the earliest any chunk says sample 0 was, until the first frame goes out
                        if self.T0 is None:
                            self.T0 = src.t0 if src.paced_by_clock else t_arr - n / SR
                        elif not src.paced_by_clock and not self.t0_frozen:
                            self.T0 = min(self.T0, t_arr - (cur + n) / SR)
                        self.lock.notify_all()
                    cur += n
                    q.put(item)
                    if self.stop.is_set():
                        break
            finally:
                q.put(None)
        threading.Thread(target=run, name="source", daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            yield item

    def _enqueue(self, jobs, t_ready):
        if not jobs:
            return
        with self.lock:
            for job in jobs:
                r = self.recs.setdefault(job.j, {"j": job.j})
                r.update(index=job.index, rows=list(job.rows), horizon=job.horizon, cursor=job.cursor, ready=t_ready,
                         lip_open=float(job.params[IDX["lip_open"]]), jaw=float(job.params[IDX["jaw"]]))
                if self.a.save_params:
                    r["params"] = job.params
                if not r.get("ticked"):         # its tick went by before its params existed: status stays no_params
                    self.pending.append(job)
            self.dispatch(mono())

    # ------------------------------------------------------------------ the log
    def arrival(self, sample):
        """Arrival time of the chunk that carried this sample."""
        k = bisect.bisect_right(self.chunk_end, sample)
        return self.chunk_arr[min(k, len(self.chunk_arr) - 1)]

    def report(self, extra):
        a, T0, D = self.a, self.T0, self.D
        ms = lambda t: None if t is None else round(1000 * (t - T0), 3)
        frames, bud = [], collections.defaultdict(list)
        e2e, presents, jit = [], [], []
        status = collections.Counter()
        params = []
        for j in sorted(self.recs):
            r = self.recs[j]
            st = r.get("status", "unticked")
            status[st] += 1
            u = j / self.fps
            s_u = int(round(u * SR))
            f = {"j": j, "t_ms": round(1000 * u, 3), "status": st, "shown": r.get("shown")}
            if "horizon" in r:
                h = r["horizon"]
                arr_u = self.arrival(s_u)
                arr_h = self.arrival(h - 1)
                f.update(index=r["index"], rows=r["rows"], lookahead_ms=round(1000 * (h / SR - u), 2),
                         audio_arrival_ms=ms(arr_u), horizon_arrival_ms=ms(arr_h), params_ready_ms=ms(r["ready"]),
                         lip_open=round(r["lip_open"], 4), jaw=round(r["jaw"], 4))
                for k in ("dispatch", "render_start", "render_end", "received"):
                    if k in r:
                        f[k + "_ms"] = ms(r[k])
                if "worker" in r:
                    f["worker"] = r["worker"]
                if r.get("probe"):
                    f["probe"] = True
                if a.save_params:
                    params.append(r["params"])
            f["deadline_ms"] = round(1000 * (u + D), 3)
            if "present" in r:
                f["present_ms"] = ms(r["present"])
            if st == "presented":
                jit.append(1000 * (r["present"] - r["tick"]))
                e2e.append(1000 * (r["present"] - self.arrival(s_u)))
                presents.append(r["present"])
                base = T0 + u
                h_t = T0 + r["horizon"] / SR
                bud["lookahead"].append(1000 * (h_t - base))
                bud["chunking"].append(1000 * (self.arrival(r["horizon"] - 1) - h_t))
                bud["animator"].append(1000 * (r["ready"] - self.arrival(r["horizon"] - 1)))
                bud["queue"].append(1000 * (r["dispatch"] - r["ready"]))
                bud["ipc_in"].append(1000 * (r["render_start"] - r["dispatch"]))
                bud["render"].append(1000 * (r["render_end"] - r["render_start"]))
                bud["return"].append(1000 * (r["received"] - r["render_end"]))
                bud["slack"].append(1000 * (r["tick"] - r["received"]))
                bud["present_jitter"].append(1000 * (r["present"] - r["tick"]))
            frames.append(f)
        n = len(frames)
        dur = n / self.fps
        rendered = [1000 * (r["render_end"] - r["render_start"]) for r in self.recs.values() if "render_end" in r]
        by_worker = collections.defaultdict(list)
        for r in self.recs.values():
            if "render_end" in r:
                by_worker[r["worker"]].append(1000 * (r["render_end"] - r["render_start"]))
        drift = None
        if len(self.chunk_arr) > 50 and not isinstance(getattr(self, "src", None), WavSource):
            ce = np.array(self.chunk_end) / SR
            lag = np.array(self.chunk_arr) - T0 - ce
            drift = {"arrival_lag_ms": stats(1000 * lag), "slope_ppm": round(float(np.polyfit(ce, lag, 1)[0] * 1e6), 1)}
        summary = {
            "frames": n, "presented": status["presented"], "dropped": n - status["presented"],
            "drop_reasons": {k: v for k, v in status.items() if k != "presented"},
            "achieved_fps": round(status["presented"] / dur, 2) if dur else 0,
            "output_fps": self.fps, "duration_s": round(dur, 3),
            "delay_ms": round(1000 * D, 1), "delay_auto_ms": round(self.D_auto, 1), "delay_auto_parts": self.D_parts,
            "lookahead_ms": round(1000 * self.core.H / SR, 2),
            "e2e_latency_ms": stats(e2e),
            "e2e_definition": "present(frame j) - arrival of the chunk carrying sample u_j; = D - (time that sample "
                              "waited for its chunk to fill); frame and sample playback both go out at T0 + u_j + D",
            "acoustic_latency_ms": round(1000 * D, 1),
            "budget_ms_mean": {k: round(float(np.mean(v)), 2) for k, v in bud.items()},
            "budget_ms_p95": {k: round(pct(v, 95), 2) for k, v in bud.items()},
            "render_ms": stats(rendered), "render_ms_by_worker": {str(k): stats(v) for k, v in sorted(by_worker.items())},
            "renders": len(rendered), "renders_wasted_late": status["late"],
            "probes": sum(1 for r in self.recs.values() if r.get("probe")),
            "probes_presented": sum(1 for r in self.recs.values() if r.get("probe") and r.get("status") == "presented"),
            "pool_throughput_fps": round(len(rendered) / dur, 2) if dur else 0,
            "present_jitter_ms": stats(jit), "sinks_ms": stats(self.sinks_ms),
            "animator_ms_per_chunk": stats(self.anim_ms), "late_audio_chunks": self.late_audio, "late_audio_out_chunks": self.late_audio_out,
            "audio_samples_in": self.n_samples, "audio_samples_played": self.played,
            "backlog_dropped_ms": round(1000 * getattr(self.src, "dropped_bytes", 0) / 2 / SR, 1),
            "clock_drift": drift,
        }
        cfg = {k: v for k, v in vars(a).items()}
        log = {"config": cfg, "warmup": {str(k): v for k, v in self.warm.items()}, "machine_start": self.m_start,
               "machine_end": self.m_end, "summary": summary, "outputs": extra, "frames": frames,
               "chunks": {"end_sample": self.chunk_end[:], "arrival_ms": [ms(t) for t in self.chunk_arr]}}
        if self.error:
            log["error"] = self.error
        if a.log:
            with open(a.log, "w") as f:
                json.dump(log, f, indent=0, default=float)
            if a.save_params and params:
                np.save(os.path.splitext(a.log)[0] + ".params.npy", np.stack(params))
        s = summary
        print(f"[live] {s['presented']}/{s['frames']} frames presented ({s['achieved_fps']} fps of {self.fps}), drops "
              f"{s['drop_reasons']}, D {s['delay_ms']} ms, e2e {s['e2e_latency_ms']}, render {s['render_ms']}", flush=True)
        print(f"[live] budget (mean ms): {s['budget_ms_mean']}", flush=True)
        return log


class _ViewSink:
    """JPEG encoding runs on its own thread (cv2 releases the GIL), so a viewer costs the presenter ~nothing; with no
    viewer connected nothing is encoded at all."""

    def __init__(self, v, quality=80):
        import cv2
        import queue
        self.v, self.cv2, self.quality = v, cv2, quality
        self.q = queue.Queue(maxsize=4)
        threading.Thread(target=self._enc, name="jpeg", daemon=True).start()

    def _enc(self):
        while True:
            t, img, j = self.q.get()
            ok, b = self.cv2.imencode(".jpg", img[:, :, ::-1], [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if ok:
                self.v.push_frame(t, jpeg=b.tobytes(), index=j)

    def frame(self, t, img, j):
        if self.v.clients():
            try:
                self.q.put_nowait((t, img, j))
            except Exception:
                pass            # encoder behind: this viewer frame is skipped, the presenter never waits

    def audio(self, t, pcm):
        self.v.push_audio(t, pcm.tobytes())


class _RecSink:
    def __init__(self, r):
        self.r = r

    def frame(self, t, img, j):
        self.r.frame(img)

    def audio(self, t, pcm):
        self.r.audio(pcm)


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="orb")
    ap.add_argument("--backend", default="v2", help="v2 | rt | sprite | raster | package.module:Class")
    ap.add_argument("--opt", action="append", default=[], help="backend option key=value (repeat), e.g. spp=1 backend=cpu")
    ap.add_argument("--width", "-W", type=int, default=640)
    ap.add_argument("--height", "-H", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30, choices=[24, 30])
    ap.add_argument("--resample", default="hold", choices=["hold", "linear"], help="24 fps from 30 fps rows (core.py)")
    ap.add_argument("--lookahead", type=float, default=133.0, help="ms: 133 (default), 67 or 0")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav")
    src.add_argument("--mic", action="store_true")
    src.add_argument("--stdin", action="store_true")
    ap.add_argument("--mic-device", default=None, help="PulseAudio source name (parec --device); default source if unset")
    ap.add_argument("--mic-latency-ms", type=float, default=10.0)
    ap.add_argument("--stdin-backlog", default="drop", choices=["drop", "keep"],
                    help="audio already in the pipe when reading starts: drop = start at the live edge (live use), "
                         "keep = play it (a file piped in), adding its length to every later sample's latency")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (mic/stdin)")
    ap.add_argument("--delay-ms", default="auto")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--pin", action="store_true", help="pin each render process to its own CPU slice")
    ap.add_argument("--strips", type=int, default=1,
                    help="v2 only: split each frame's character pass over K processes (live/strips.py); every --procs "
                         "worker becomes a group of K")
    ap.add_argument("--warmup", type=int, default=6, help="frames each render process draws (all at once) before the "
                    "start; their times set D=auto")
    ap.add_argument("--max-ahead", type=float, default=1.0, help="s; never render a frame due later than this")
    ap.add_argument("--probe-after", type=float, default=0.15,
                    help="s without a finished render before an idle pool renders a job the estimate says will miss")
    ap.add_argument("--port", type=int, default=8765, help="HTTP live view on 127.0.0.1 (0 = any free port)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--wait-clients", type=int, default=0, help="hold the start until this many viewers connect")
    ap.add_argument("--window", action="store_true", help="ffplay window + pacat audio; OFF by default")
    ap.add_argument("--display", default=None, help="X display for --window (default: inherit DISPLAY)")
    ap.add_argument("--window-mute", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--save-params", action="store_true", help="also save the released params next to the log")
    ap.add_argument("--hard-stop", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.strips > 1:
        if a.backend != "v2":
            ap.error("--strips splits the v2 engine's character pass; other backends parallelise with --procs")
        a.backend = "live.strips:StripBackend"
        a.opt = a.opt + [f"strips={a.strips}"]
    return a


def main(argv=None):
    a = build_args(argv)
    os.environ.setdefault("TA_OUT", os.path.join(os.path.dirname(ROOT), "out", "cache"))
    sys.setswitchinterval(0.0005)
    log = LiveLoop(a).run()
    return 0 if not log.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
