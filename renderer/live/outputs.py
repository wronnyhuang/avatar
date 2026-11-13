"""Frame/audio sinks fed by the presenter at presentation time.  None of them may block the presenter: each owns a
writer thread behind an unbounded queue (frames are references, a repeated frame costs nothing).

  Recorder  --record out.mp4: the video stream is encoded live (libx264, CFR at the output fps, exactly the frames
            presented, repeats included) and the audio played is written as it goes; at the end both are muxed into
            H.264 + AAC with +faststart.  File time 0 = the first presentation tick (T0 + D), so in the file frame j sits
            at j/fps and audio sample tau at tau: both delayed by D, in sync.
  Window    --window: ffplay (SDL) showing the presented frames, plus `pacat` playing the delayed audio.  Opt-in only.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading

import numpy as np

from anim.schema import SAMPLE_RATE


class _PipeWriter:
    def __init__(self, cmd, name, env=None):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     env=env)
        self.q = queue.Queue()
        self.err = b""
        self.broken = False
        self.th = threading.Thread(target=self._run, name=name, daemon=True)
        self.th.start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        self.err = self.proc.stderr.read()

    def _run(self):
        while True:
            b = self.q.get()
            if b is None:
                break
            if self.broken:
                continue
            try:
                self.proc.stdin.write(b)
            except (BrokenPipeError, OSError):
                self.broken = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass

    def put(self, b):
        self.q.put(b)

    def close(self, timeout=120):
        self.q.put(None)
        self.th.join(timeout)
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            return self.proc.wait(5)


class Recorder:
    def __init__(self, path, w, h, fps, crf=18, preset="veryfast"):
        self.path, self.w, self.h, self.fps = path, w, h, fps
        base = os.path.splitext(path)[0]
        self.vtmp, self.atmp = base + ".video.mkv", base + ".audio.s16"
        self.v = _PipeWriter(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                              "-pix_fmt", "yuv420p", self.vtmp], "rec-video")
        self.af = open(self.atmp, "wb")
        self.frames = 0
        self.samples = 0

    def frame(self, img):
        self.v.put(memoryview(np.ascontiguousarray(img)).cast("B"))
        self.frames += 1

    def audio(self, pcm):
        self.af.write(np.asarray(pcm, np.int16).tobytes())
        self.samples += len(pcm)

    def close(self):
        rc = self.v.close()
        self.af.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video encode failed: {self.v.err.decode(errors='replace')[-800:]}")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", self.vtmp, "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
               "-i", self.atmp, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", self.path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {r.stderr.decode(errors='replace')[-800:]}")
        os.remove(self.vtmp); os.remove(self.atmp)
        return {"path": self.path, "frames": self.frames, "audio_samples": self.samples,
                "video_s": self.frames / self.fps, "audio_s": self.samples / SAMPLE_RATE}


class Window:
    """ffplay for the picture, pacat for the sound.  DISPLAY must be given explicitly or inherited; the caller decides."""

    def __init__(self, w, h, fps, display=None, audio=True, title="live"):
        if not shutil.which("ffplay"):
            raise RuntimeError("ffplay not found")
        env = dict(os.environ)
        if display:
            env["DISPLAY"] = display
        self.v = _PipeWriter(["ffplay", "-v", "error", "-hide_banner", "-window_title", title, "-fflags", "nobuffer",
                              "-flags", "low_delay", "-framedrop", "-sync", "ext", "-f", "rawvideo", "-pixel_format",
                              "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-"], "win-video", env)
        self.a = None
        if audio and shutil.which("pacat"):
            self.a = _PipeWriter(["pacat", "--playback", "--raw", f"--rate={SAMPLE_RATE}", "--channels=1",
                                  "--format=s16le", "--latency-msec=30", "--client-name=avatar-live"], "win-audio")

    def frame(self, img):
        self.v.put(memoryview(np.ascontiguousarray(img)).cast("B"))

    def audio(self, pcm):
        if self.a:
            self.a.put(np.asarray(pcm, np.int16).tobytes())

    def close(self):
        for p in (self.v, self.a):
            if p:
                try:
                    p.proc.terminate()
                except Exception:
                    pass
                p.close(timeout=5)
