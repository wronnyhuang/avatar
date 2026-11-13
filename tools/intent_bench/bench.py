"""Latency / accuracy / false-positive bench for the server's streaming ASR + intent matcher.

usage: python3 bench.py --clips <tts dir with manifest.json> --py <venv python> --model DIR [--grammar-model DIR]
                        --mode full|grammar|both --out result.json [--shards 4] [--variants clean,noise20,noise10,echo]
                        [--extra-neg a.wav,b.wav ...]

Each clip is fed to app/asr_worker.py (the real subprocess) in 140 ms batches (the page's exchange cadence),
padded with 0.4 s of room noise before and 1.2 s after. The matcher (app/intent.py Stream) runs on the
worker's partial/final lines with the AUDIO clock as its time base. Latency of a hit =
  (audio position of the decode that produced it) + (that decode's cpu time) - (end of speech in the clip),
i.e. word-end -> intent emitted at the server, excluding the network legs (reported separately).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "..", "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import intent  # noqa: E402
from check_cases import cases, summ  # noqa: E402

SR = 24000
BATCH = 3360   # 140 ms


def load_pcm(fn: str) -> np.ndarray:
    if fn.endswith(".pcm"):
        return np.frombuffer(open(fn, "rb").read(), dtype="<i2").astype(np.float32)
    with wave.open(fn) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        t = np.arange(0, len(x) * SR / sr) * sr / SR
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


def speech_end(x: np.ndarray) -> float:
    env = np.abs(x)
    thr = max(300.0, 0.05 * float(np.percentile(env, 99.5)))
    idx = np.nonzero(env > thr)[0]
    return (idx[-1] + 1) / SR if len(idx) else len(x) / SR


def pink(n: int, rng) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    p = np.fft.irfft(f, n).astype(np.float32)
    return p / (np.std(p) + 1e-9)


def make_variant(x: np.ndarray, variant: str, rng, avatar: list[np.ndarray], avatar_text=None):
    lead, tail = int(0.4 * SR), int(1.2 * SR)
    end = speech_end(x)
    y = np.concatenate([np.zeros(lead, np.float32), x, np.zeros(tail, np.float32)])
    sp = float(np.sqrt(np.mean(x[np.abs(x) > 200] ** 2))) if np.any(np.abs(x) > 200) else 1000.0
    y += pink(len(y), rng) * 30.0                      # room floor (~-60 dBFS)
    if variant.startswith("noise"):
        snr = float(variant[5:])
        y += pink(len(y), rng) * sp / (10 ** (snr / 20))
    elif variant.startswith("echo"):                   # avatar voice leaking back (default -12 dB), overlapping
        db = float(variant[4:] or 12)
        ai = int(rng.integers(len(avatar)))
        a = avatar[ai]
        echo_text = avatar_text[ai] if avatar_text else ""
        echo_off = off_s = None
        off = int(rng.integers(0, max(1, len(y) - len(a)))) if len(a) < len(y) else 0
        seg = a[:len(y) - off] * (10 ** (-db / 20)) * sp / (float(np.sqrt(np.mean(a[np.abs(a) > 200] ** 2))) + 1e-9)
        y[off:off + len(seg)] += seg
    y = np.clip(y, -32768, 32767)
    return y.astype("<i2"), lead / SR + end, (echo_text if variant.startswith("echo") else "")


class Worker:
    def __init__(self, py: str, args: list[str]):
        self.p = subprocess.Popen([py, "-I", os.path.join(APP, "asr_worker.py")] + args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.ready = json.loads(self.p.stdout.readline())

    def run(self, pcm: np.ndarray, nrec: int) -> list[dict]:
        w = self.p.stdin
        w.write(b"R" + struct.pack("<I", 0))
        b = pcm.tobytes()
        for i in range(0, len(b), BATCH * 2):
            ch = b[i:i + BATCH * 2]
            w.write(b"A" + struct.pack("<I", len(ch)) + ch)
        w.write(b"E" + struct.pack("<I", 0))
        w.flush()
        lines, eos = [], 0
        while eos < nrec:
            ln = self.p.stdout.readline()
            if not ln:
                break
            j = json.loads(ln)
            lines.append(j)
            if j.get("eos"):
                eos += 1
        return lines

    def close(self):
        try:
            self.p.stdin.write(b"Q" + struct.pack("<I", 0))
            self.p.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        self.p.wait(timeout=10)


def run_matcher(lines: list[dict], recs: list[str]) -> list[dict]:
    m = intent.Matcher()
    streams = {r: m.stream() for r in recs}
    hits, recent = [], []
    for j in lines:
        if j["r"] not in streams:
            continue
        now = j["a"] * 1000
        for h in streams[j["r"]].update(j["text"], j["t"] == "f", now):
            cj = intent.canon(h["cmd"])
            if any(c == cj and now - t < 1500 for t, c in recent):
                continue
            recent.append((now, cj))
            hits.append({"s": summ(h["cmd"]), "a": j["a"], "c": j["c"], "r": j["r"], "partial": h["partial"],
                         "text": j["text"]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grammar-model", default="")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--variants", default="clean,noise20,noise10,echo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra-neg", default="")
    a = ap.parse_args()
    expect = {ph: exp for ph, exp in cases()}
    man = json.load(open(os.path.join(a.clips, "manifest.json")))
    avatar = [load_pcm(c["file"]) for c in man if c["kind"] == "avatar" and os.path.exists(c["file"])]
    avatar_text = [c["text"] for c in man if c["kind"] == "avatar" and os.path.exists(c["file"])]
    items = []
    for c in man:
        if not os.path.exists(c["file"]):
            continue
        if c["kind"] == "avatar":
            items.append({"kind": "neg", "sub": "avatar", "text": c["text"], "file": c["file"], "voice": c["voice"], "speed": c["speed"], "expect": []})
        else:
            items.append({"kind": c["kind"], "sub": c["kind"], "text": c["text"], "file": c["file"], "voice": c["voice"], "speed": c["speed"],
                          "expect": expect.get(c["text"], [])})
    for fn in [f for f in a.extra_neg.split(",") if f]:
        if "command" in os.path.basename(fn) or "interrupt" in os.path.basename(fn):
            continue   # these 'prepared' clips contain spoken commands: not negatives
        items.append({"kind": "neg", "sub": "real", "text": os.path.basename(fn), "file": fn, "voice": "real", "speed": 1, "expect": []})
    if a.limit:
        random.Random(1).shuffle(items)
        items = items[:a.limit]
    variants = a.variants.split(",")
    jobs = []
    for i, it in enumerate(items):
        for v in variants:
            if it["sub"] == "real" and v != "clean":
                continue
            jobs.append((i, v))
    recs = ["full", "grammar"] if a.mode == "both" else [a.mode]
    wargs = ["--model", a.model, "--mode", a.mode] + (["--grammar-model", a.grammar_model] if a.grammar_model else [])
    shards = [jobs[k::a.shards] for k in range(a.shards)]
    results = []
    lock = threading.Lock()
    t0 = time.time()

    def run_shard(sj):
        w = Worker(a.py, wargs)
        rng = np.random.default_rng(len(sj))
        for i, v in sj:
            it = items[i]
            x = load_pcm(it["file"])
            pcm, t_end, echo_text = make_variant(x, v, rng, avatar, avatar_text)
            if it["sub"] == "avatar":
                echo_text = it["text"]
            raw = w.run(pcm, len(recs))
            hits = {"+".join(rs): run_matcher(raw, rs) for rs in ([recs] if len(recs) == 1 else [[recs[0]], [recs[1]], recs])}
            cpu = [j["c"] for j in raw]
            r = {"i": i, "variant": v, "kind": it["kind"], "sub": it["sub"], "text": it["text"], "voice": it["voice"],
                 "speed": it["speed"], "expect": it["expect"], "t_end": round(t_end, 3), "hits": hits, "echo_text": echo_text,
                 "cpu_ms_p95": round(float(np.percentile(cpu, 95)), 1) if cpu else 0, "audio_s": round(len(pcm) / SR, 2),
                 "cpu_ms_sum": round(float(sum(cpu)), 1),
                 "finals": {rr: [j["text"] for j in raw if j["r"] == rr and j["t"] == "f" and j["text"]] for rr in recs}}
            with lock:
                results.append(r)
        w.close()
        return w.ready

    with ThreadPoolExecutor(a.shards) as ex:
        readies = list(ex.map(run_shard, shards))
    json.dump({"mode": a.mode, "model": a.model, "grammar_model": a.grammar_model, "ready": readies,
               "wall_s": round(time.time() - t0, 1), "results": results}, open(a.out, "w"))
    print(f"{len(results)} runs in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
