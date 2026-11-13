"""Server side of the fast intent matcher: the page's 'M' mic chunks -> streaming ASR worker -> intent.Matcher ->
{"type":"intent","cmd":{...},"text":...,"heard_at":<hub monotonic s>,"partial":bool} on the session's down queue.

One ASR worker process per server (kept warm across sessions; a session start resets it). The worker is
app/asr_worker.py run with its own interpreter (cfg intent.python, which has vosk), -I, single-threaded
recognizers (<= 2 cores with mode 'both'). Everything here is best-effort: if the worker is missing or dies,
the matcher is simply off and the realtime model's tool calls still drive the avatar.

config.json  "intent": {"enabled": true, "python": "<venv python with vosk>", "model": "<vosk model dir>",
                         "grammar_model": "", "mode": "full" | "grammar" | "both", "max_queue_s": 2.0}
"""
from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time

import intent
try:                                     # the LLM read of a final the act grammar missed (muse)
    from intent_llm import LlmFallback
except Exception:                        # noqa: BLE001 — the fallback is optional
    LlmFallback = None

HERE = os.path.dirname(os.path.abspath(__file__))
# defaults: this interpreter (needs `vosk` installed) and a vosk model under <repo>/models/ (config intent.python /
# intent.model override both; download vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models)
DEFAULTS = {"enabled": True, "python": sys.executable,
            "model": os.path.join(HERE, "..", "models", "vosk-model-small-en-us-0.15"), "grammar_model": "",
            "mode": "full", "max_queue_s": 2.0}
SR = 24000

# intent: the act grammar (whole-body moves + props) runs for the characters that have them (by default
# the muse; env AVATAR_ACT_CHARS); every other character keeps the plain matcher exactly (Matcher(act=False), pinned by tests).
ACT_CHARACTERS = set(c for c in (os.environ.get("AVATAR_ACT_CHARS") or "muse").split(",") if c)
RENDER_ROOT = os.path.join(HERE, "..", "renderer")
MOCK_PROPS = os.path.join(HERE, "..", "tests", "props_mock")
_SCENES: dict = {}


def act_for(character: str | None) -> bool:
    return bool(character) and (character in ACT_CHARACTERS or "*" in ACT_CHARACTERS)


def scene_for(character: str | None):
    """config.props of a character: renderer/animals/<c>.props.json (props agent), else renderer/props/<c>.json,
    else the mock scene (tests/props_mock/<c>.json) until the real one lands. None when there is none."""
    if not character:
        return None
    if character in _SCENES:
        return _SCENES[character]
    sc = None
    for fn in (os.path.join(RENDER_ROOT, "animals", f"{character}.props.json"),
               os.path.join(RENDER_ROOT, "props", f"{character}.json"), os.path.join(MOCK_PROPS, f"{character}.json")):
        try:
            with open(fn, encoding="utf-8") as f:
                sc = json.load(f)
            break
        except (OSError, ValueError):
            continue
    _SCENES[character] = sc
    return sc


def matcher_for(character: str | None) -> intent.Matcher:
    if act_for(character):
        return intent.Matcher(act=True, scene=scene_for(character))
    return intent.Matcher(act=False)


class IntentHub:
    def __init__(self, cfg: dict | None, log):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.log = log
        self.p: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.qbytes = 0
        self.fed_ms = 0.0               # audio clock of what vosk was fed, and the end of the user's last voiced frame
        self.voiced_ms = -1e9
        self.lock = threading.Lock()
        self.sink = None              # (loop, send(obj), stats dict) of the attached session
        self.on_text = None           # cb(text, final, asr) on the session loop for every recognizer line
        self.matcher: intent.Matcher | None = None
        self.character: str | None = None   # intent: the attached session's character (act grammar or not)
        self.streams: dict = {}
        self.recent: list = []
        self.ready = False
        self.failed = ""
        self.gen = 0
        self.stats = {"chunks": 0, "dropped_s": 0.0, "lines": 0, "hits": 0}
        # config intent.llm_fallback (default: the muse only), llm_model, llm_timeout_s
        self.llm = LlmFallback(self.cfg, log) if LlmFallback is not None else None
        self.utt_fired: dict = {}
        self.last_hit_ms = -1e9

    # -- availability
    def usable(self) -> bool:
        c = self.cfg
        if not c.get("enabled") or self.failed:
            return False
        return bool(c.get("python")) and os.path.isfile(c["python"]) and os.path.isdir(c.get("model") or "")

    def start(self) -> bool:
        if self.p and self.p.poll() is None:
            return True
        if not self.usable():
            return False
        c = self.cfg
        args = [c["python"], "-I", os.path.join(HERE, "asr_worker.py"), "--model", c["model"], "--mode", c["mode"],
                "--rules", os.path.join(HERE, "intent_rules.json")]
        if c.get("grammar_model"):
            args += ["--grammar-model", c["grammar_model"]]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        try:
            self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=0, env=env, cwd="/")
        except OSError as e:
            self.failed = f"spawn: {e}"
            self.log(f"intent: ASR worker failed to start: {e}")
            return False
        self.ready = False
        threading.Thread(target=self._writer, args=(self.p,), daemon=True).start()
        threading.Thread(target=self._reader, args=(self.p,), daemon=True).start()
        self.log(f"intent: ASR worker pid {self.p.pid} ({c['mode']}, {os.path.basename(c['model'])})")
        return True

    # -- session
    def attach(self, loop, send, stats: dict, character: str | None = None) -> bool:
        if not self.start():
            return False
        with self.lock:
            self.gen += 1
            self.sink = (loop, send, stats, self.gen)
            self.character = character
            self.matcher = matcher_for(character)
            self.streams = {}
            self.recent = []
            self.fed_ms, self.voiced_ms = 0.0, -1e9   # vosk restarts its audio clock on R
        self._put(b"R", b"")
        return True

    def set_character(self, character: str | None) -> None:
        """intent: a character switch swaps the grammar (act on/off) and the scene nouns"""
        with self.lock:
            if self.sink is None or character == self.character:
                return
            self.character = character
            self.matcher = matcher_for(character)
            self.streams = {}
            self.recent = []

    def forget_context(self) -> None:
        """the voice model started an action of its own: "more" no longer refers to the matcher's last pose"""
        with self.lock:
            if self.matcher is not None:
                self.matcher.forget_context()

    def detach(self) -> None:
        with self.lock:
            self.sink = None
            self.on_text = None
            self.gen += 1
        if self.p and self.p.poll() is None:
            self._put(b"R", b"")

    def feed(self, pcm: bytes) -> None:
        if self.sink is None or not self.p or self.p.poll() is not None:
            return
        self.stats["chunks"] += 1
        lim = int(self.cfg["max_queue_s"] * SR * 2)
        if self.qbytes + len(pcm) > lim:       # worker can't keep up: drop (latency matters more than recall)
            self.stats["dropped_s"] += len(pcm) / 2 / SR
            return
        self._put(b"A", pcm)
        self.voiced_ms = intent.voiced_until_ms(pcm, self.fed_ms, self.voiced_ms, SR)
        self.fed_ms += len(pcm) / 2 / SR * 1000

    def close(self) -> None:
        if self.p and self.p.poll() is None:
            self._put(b"Q", b"")
            try:
                self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.p.kill()

    # -- threads
    def _put(self, kind: bytes, data: bytes) -> None:
        self.qbytes += len(data)
        self.q.put(kind + struct.pack("<I", len(data)) + data)

    def _writer(self, p: subprocess.Popen) -> None:
        try:
            while p.poll() is None:
                rec = self.q.get()
                self.qbytes = max(0, self.qbytes - (len(rec) - 5))
                buf = [rec]
                while not self.q.empty() and len(buf) < 16:    # coalesce
                    r2 = self.q.get_nowait()
                    self.qbytes = max(0, self.qbytes - (len(r2) - 5))
                    buf.append(r2)
                p.stdin.write(b"".join(buf))
                p.stdin.flush()
        except (OSError, ValueError):
            pass

    def _reader(self, p: subprocess.Popen) -> None:
        try:
            for ln in p.stdout:
                try:
                    j = json.loads(ln)
                except ValueError:
                    continue
                if j.get("t") == "ready":
                    self.ready = True
                    self.log(f"intent: ASR ready ({j.get('load_s')} s load, {j.get('grammar_words')} grammar words)")
                    continue
                self.stats["lines"] += 1
                self._on_line(j)
        except (OSError, ValueError):
            pass
        rc = p.poll()
        if self.p is p:
            self.log(f"intent: ASR worker exited ({rc})")
            self.p = None

    def _on_line(self, j: dict) -> None:
        with self.lock:
            sink, m = self.sink, self.matcher
            if sink is None or m is None:
                return
            cb = self.on_text                            # the teach stop words hear the fast recognizer
            if cb is not None and j.get("text"):
                try:
                    sink[0].call_soon_threadsafe(cb, str(j.get("text")), j.get("t") == "f", j.get("r", "full"))
                except RuntimeError:
                    pass
            r = j.get("r", "full")
            st = self.streams.get(r)
            if st is None:
                st = self.streams[r] = m.stream()
            now_ms = float(j.get("a", 0)) * 1000         # the audio clock (stability = audio time)
            if j.get("t") == "f" and j.get("text"):
                asr = sink[2].setdefault("asr", [])
                asr.append([j.get("a"), r, j["text"][:200]])
                del asr[:-200]
            hits = st.update(j.get("text", ""), j.get("t") == "f", now_ms, quiet_ms=now_ms - self.voiced_ms)
            if hits:                                     # this utterance already got a command
                self.utt_fired[r] = True
                self.last_hit_ms = now_ms
            ask = None
            if j.get("t") == "f":
                fired = self.utt_fired.pop(r, False)
                txt = (j.get("text") or "").strip()
                if not fired and len(txt.split()) >= 3:      # the user said something else since the pose "more" meant
                    m.forget_context()
                if not fired and txt and r == ("grammar" if self.cfg.get("mode") == "grammar" else "full") \
                        and now_ms - self.last_hit_ms > 1500 and self.llm is not None and self.llm.on_for(self.character):
                    sc = scene_for(self.character)
                    if self.llm.wants(txt, sc):
                        ask = (txt, sc, r, j.get("a"))
            out = []
            for h in hits:
                cj = intent.canon(h["cmd"])
                if any(c == cj and now_ms - t < intent.REPEAT_WINDOW_MS for t, c in self.recent):
                    continue                             # the other recognizer already fired it
                self.recent.append((now_ms, cj))
                out.append(h)
            self.recent = [(t, c) for t, c in self.recent if now_ms - t < 3000]
        if ask is not None:
            threading.Thread(target=self._llm_ask, args=(sink,) + ask, daemon=True).start()
        if not out:
            return
        loop, send, stats, gen = sink
        heard = time.monotonic()
        for h in out:
            self.stats["hits"] += 1
            msg = {"type": "intent", "cmd": h["cmd"], "text": j.get("text", ""), "heard_at": round(heard, 3),
                   "partial": h["partial"], "rule": h["rule"], "asr": j.get("r"), "audio_s": j.get("a")}
            if h.get("replaces"):
                msg["replaces"] = h["replaces"]

            def deliver(msg=msg, gen=gen):
                if self.sink is not None and self.sink[3] == gen:
                    send(msg)
                    stats.setdefault("intents", []).append({k: msg[k] for k in ("cmd", "text", "partial", "rule", "audio_s")})
            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass

    # the second fast path (app/intent_llm.py) — runs off the reader thread, best-effort
    def _llm_ask(self, sink, text, scene, r, audio_s) -> None:
        loop, send, stats, gen = sink
        t0 = time.monotonic()
        cmds, info = self.llm.ask(text, scene)
        st = stats.setdefault("llm", {"calls": 0, "cmds": 0, "usd": 0.0, "errors": 0, "ms": []})
        st["calls"] += 1
        st["cmds"] += len(cmds)
        st["usd"] = round(st["usd"] + info.get("usd", 0.0), 6)
        st["errors"] += bool(info.get("err"))
        st["ms"] = (st["ms"] + [info.get("ms")])[-50:]
        self.log(f"intent: llm fallback {len(cmds)} cmd(s) in {info.get('ms')} ms, ${info.get('usd', 0):.5f}"
                 f"{' (' + info['err'] + ')' if info.get('err') else ''}: {text[:80]!r}")
        if not cmds:
            return
        heard = time.monotonic()
        for k, c in enumerate(cmds):
            self.stats["hits"] += 1
            msg = {"type": "intent", "cmd": c, "text": text, "heard_at": round(heard, 3), "partial": False,
                   "rule": "llm:" + info.get("model", ""), "source": "matcher-llm", "asr": r, "audio_s": audio_s,
                   "llm_ms": info.get("ms"), "step": k}

            def deliver(msg=msg, gen=gen):
                if self.sink is not None and self.sink[3] == gen:
                    send(msg)
                    stats.setdefault("intents", []).append({k2: msg[k2] for k2 in ("cmd", "text", "partial", "rule", "audio_s")})
            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass
