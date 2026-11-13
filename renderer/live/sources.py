"""Audio sources for the live loop.  Each yields (pcm int16 chunk, arrival time) where arrival = time.monotonic() at the
moment the chunk's last sample is in the process.  24 kHz mono s16le throughout.

  WavSource    a WAV file fed in real time: chunk k (samples [k c, (k+1) c)) is released at T0 + (k+1) c, exactly as a
               capture device delivers it.  T0 is the source's own start.  Any rate/channels are converted to 24 kHz mono.
  MicSource    PulseAudio capture through `parec` (raw s16le 24 kHz mono, --latency-msec), optional --device.  Never
               touches the system's default devices.
  StdinSource  raw s16le 24 kHz mono on stdin, read as it comes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

from anim.schema import SAMPLE_RATE

SR = SAMPLE_RATE


def load_wav(path):
    """int16 mono at 24 kHz.  Resampled with scipy's polyphase filter if the file is at another rate."""
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: need 16-bit PCM, got {8 * sw}-bit")
    x = np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float64).mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


class Source:
    name = "base"
    paced_by_clock = False      # True: arrival times follow a known schedule (the wav), so T0 is exact

    def __init__(self, chunk_ms=20):
        self.chunk = int(round(SR * chunk_ms / 1000.0))
        self.chunk_ms = chunk_ms
        self.t0 = None              # monotonic time of stream sample 0, if the source knows it

    def __iter__(self):
        raise NotImplementedError

    def close(self):
        pass


class WavSource(Source):
    name = "wav"
    paced_by_clock = True

    def __init__(self, path, chunk_ms=20, start_at=None, pcm=None, pace=True):
        super().__init__(chunk_ms)
        self.path, self.pace = path, pace
        self.pcm = load_wav(path) if pcm is None else np.asarray(pcm, np.int16)
        self.start_at = start_at
        self._stop = False

    def __iter__(self):
        n = self.chunk
        t0 = self.start_at if self.start_at is not None else time.monotonic()
        self.t0 = t0
        for k, a in enumerate(range(0, self.pcm.size, n)):
            if self._stop:
                return
            c = self.pcm[a:a + n]
            due = t0 + (a + c.size) / SR            # the moment its last sample has been "captured"
            d = due - time.monotonic()
            if d > 0 and self.pace:
                time.sleep(d)
            yield c, time.monotonic()

    def close(self):
        self._stop = True


class _PipeSource(Source):
    """Reads exactly one chunk of bytes at a time from a pipe; arrival = when the read completed.

    backlog: what the pipe already holds when reading starts (audio a producer wrote while the render pool warmed up).
    'drop' (live use) discards it and starts at the live edge, so the latency is D from the first frame on; 'keep'
    plays it, and every later sample then waits that much longer (a file piped in is all backlog: use keep)."""
    backlog = "drop"
    dropped_bytes = 0

    def _reader(self):
        raise NotImplementedError

    def _drain(self, f):
        fd = f.fileno()
        n = 0
        os.set_blocking(fd, False)
        try:
            while True:
                b = os.read(fd, 1 << 16)
                if not b:
                    break
                n += len(b)
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(fd, True)
        if n % 2:
            n += len(os.read(fd, 1))        # stay on a sample boundary
        self.dropped_bytes = n

    def __iter__(self):
        f = self._reader()
        if self.backlog == "drop":
            self._drain(f)
        need = 2 * self.chunk
        buf = b""
        while True:
            try:
                b = f.read1(need - len(buf)) if hasattr(f, "read1") else f.read(need - len(buf))
            except (OSError, ValueError):
                b = b""
            now = time.monotonic()
            if not b:
                if len(buf) >= 2:
                    yield np.frombuffer(buf[:len(buf) // 2 * 2], np.int16).copy(), now
                return
            buf += b
            if len(buf) >= need:
                yield np.frombuffer(buf[:need], np.int16).copy(), now
                buf = buf[need:]


class MicSource(_PipeSource):
    name = "mic"

    def __init__(self, chunk_ms=20, device=None, latency_ms=10):
        super().__init__(chunk_ms)
        if not shutil.which("parec"):
            raise RuntimeError("parec not found (PulseAudio/PipeWire-pulse utilities)")
        self.cmd = ["parec", "--raw", f"--rate={SR}", "--channels=1", "--format=s16le",
                    f"--latency-msec={int(latency_ms)}", "--client-name=avatar-live", "--stream-name=live-loop-capture"]
        if device:
            self.cmd.append(f"--device={device}")
        self.device = device
        self.latency_ms = latency_ms
        self.proc = None

    def _reader(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        return self.proc.stdout

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()       # our own child
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class StdinSource(_PipeSource):
    name = "stdin"

    def _reader(self):
        return os.fdopen(sys.stdin.fileno(), "rb", buffering=0, closefd=False)
