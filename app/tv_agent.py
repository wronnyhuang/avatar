"""tv_agent -- the wall TV's illustration agent.

A fast model (Claude Haiku by default, through app/claude_door.py (the `claude` CLI, or the Anthropic API)) writes a TVL program (the
illustration language in app/tvdsl/, spec = tvdsl/system_prompt.txt); the program streams line by line into a
deterministic renderer that runs in its OWN PROCESS (never the server's event loop) and pushes frames to the `tv` surface.
As the conversation goes on, a conversation tap asks the same model for a PATCH (set a param, add / replace / remove
objects, or `none`), so the picture follows the talk without starting over.

    tv = TVAgent(surface=lambda: session.surfaces.get("tv"), on_state=session.send_tv_state)
    r = tv.show("how a converging lens forms an image", context=[("you", "..."), ("avatar", "...")])   # returns at once
    tv.hear("you", "what if the object is closer than the focal point?")                             # the live tap
    tv.clear()                       # idle look
    tv.describe()                    # one line: what is on the TV now (for the voice model)
    await tv.close()

on_state(dict) gets {"type": "tv_state", "id", "status": drawing|shown|updated|unchanged|failed|cleared, "desc",
"ms_first", "ms_done", "source": "tool"|"tap", "error"} -- the server forwards it to the page as a silent note.
Fail-safe: a new scene is committed only once its first object line parses; a stream that yields nothing keeps the
last good picture; a malformed line is skipped.
"""
import asyncio
import json
import os
import re
import threading
import time

from tvdsl.lang import Doc, parse_line, ParseError
from tvdsl.expr import ExprError

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = open(os.environ.get("AVATAR_TV_PROMPT") or os.path.join(HERE, "tvdsl", "system_prompt.txt"), encoding="utf8").read()
DEFAULTS = {
    "enabled": True,
    "model": os.environ.get("AVATAR_TV_MODEL", "haiku"),     # "haiku" | "sonnet"
    # follow-up patches on Sonnet: dev follow-up eval strict 50% / lenient 94% vs Haiku 25% / 75% (and 1
    # escalation instead of 5); a Haiku hedge at 2.5 s keeps a slow Sonnet from stalling the user's change
    "patch_model": os.environ.get("AVATAR_TV_PATCH_MODEL", "sonnet"),
    # hedge at 1.8 s (dev eval: strict 61% / lenient 92%, visible p50 3.5 s at load 33); at 0 s (Haiku always first,
    # Sonnet always corrects) visible p50 1.7 s but strict 39%: the two-stage change shows stale captions mid-way
    "patch_hedge_s": [1.8],
    "patch_hedge_models": ["haiku"],
    # when the Haiku hedge wins a patch (Sonnet slow to start), the Sonnet request runs on and its patch
    # is applied as a correction (morph from Haiku's quick change to before + Sonnet's patch). Measured: Haiku-won patches
    # passed 1/10 strict, Sonnet's 13/26
    "patch_correct": True,
    "fps": 12,
    "w": 512, "h": 288,
    "max_s": 25.0,
    "tap": True,                                             # live updates from the conversation
    "tap_min_words": 2,                                      # "increase demand" is a whole request
    "tap_debounce_s": 0.4,
    "tap_min_gap_s": 1.5,
    "context_turns": 8,
    "process": True,                                         # renderer in a child process (False = thread: tests)
    "render": True,                                          # False = no renderer at all (eval replays the timeline)
    # Haiku sometimes thinks (hidden, 5-15 s) before its first word (~25% of calls, measured; neither --effort low
    # nor MAX_THINKING_TOKENS=0 nor alwaysThinkingEnabled=false stops it): a second / third identical request starts at
    # these seconds if no text has come yet; the first to speak wins, the others are cancelled
    "hedge_s": [1.5, 3.0],
    "hedge_models": [None, "sonnet"],                        # the 3.0 s hedge is Sonnet (no thinking spells)
    # self-review: once a NEW picture is up, a second (stronger) model reads the program as text like a strict teacher
    # and answers fix lines or `none`; fixes morph in a few seconds later. Superseded by anything the user says.
    "review": os.environ.get("AVATAR_TV_REVIEW") or None,     # None | "haiku" | "sonnet"
    # sketch + careful version: a new picture is drawn by `model` at once (streams, ~1.2 s) while `upgrade` writes the
    # same request in parallel; if it finishes within upgrade_max_s with a real picture it replaces the sketch
    # (crossfade). Live taps wait for it, so the user's changes land on the final picture.
    # measured: haiku alone 34% devset / 18% holdout strict pass; + sonnet careful version 49% / 41%
    # the careful version on Opus: devset (46, same code) Opus 43% strict / 87% lenient, careful version
    # landed p50 6.4 s p90 9.5 s vs Sonnet 41% / 78%, p50 5.1 s p90 13.8 s (under load)
    "upgrade": (os.environ.get("AVATAR_TV_UPGRADE", "opus") or None) if os.environ.get("AVATAR_TV_UPGRADE", "opus") != "none" else None,
    "upgrade_max_s": 18.0,                                   # (12 s cut Sonnet off mid-program under load)
    # how the careful version is made. "parallel" = `upgrade` writes the whole picture in parallel and it
    # is swapped in SOFTLY (morph: same IDs glide, no restart); "edit" = once the sketch is complete, `check_model`
    # LOOKS at it (rendered stills) and answers fix lines / none / a better program (morph); "parallel+check" = both.
    "careful": os.environ.get("AVATAR_TV_CAREFUL", "parallel"),
    "careful_plan": False,
    "check_model": "sonnet",
    "check_max_s": 14.0,
    # follow-ups: after a patch the picture is rendered before/after; a patch that did not change the DRAWING (only
    # captions, or nothing) is escalated to `fix_model`, which sees the picture and the user's words
    "verify": True,
    "fix_model": "sonnet",
    "fix_max_s": 14.0,
    "geom_min": 0.0015,
    "lint": True,                                            # objects that draw nothing / print '?' get a quick fix                                      # fraction of (256x144) pixels the drawing must change
}
GLIDE_SETTLE = 1.6                                           # s after a change when params/morphs have settled
BACKCHANNEL = re.compile(r"^(?:ok(?:ay)?|yeah|yes|yep|no|nope|right|sure|cool|nice|wow|great|i see|got it|uh+ ?huh|"
                         r"mm+ ?hmm+|hmm+|ah+|oh+|thanks?(?: you)?|alright|all right|interesting)[.!?, ]*$", re.I)


# ================================================================================================ the render process
class RenderHost:
    """the parent side: runs tvdsl/worker.py as a child process (or a thread in tests), forwards lines, and hands the
    frames it sends back to deliver(rgb | None) ON THE EVENT LOOP (None = idle look)"""

    def __init__(self, cfg, deliver, log=None, prepare=None):
        self.cfg = cfg
        self.deliver = deliver
        self.prepare = prepare            # called on the pump thread with each frame (None = idle): heavy work off the loop
        self.log = log or (lambda *a: None)
        self.w, self.h = int(cfg["w"]), int(cfg["h"])
        self.proc = None
        self.inp = None                   # we write JSON lines here
        self.out = None                   # we read frames here
        self.thread = None
        self.loop = None
        self.frames = 0
        self.render_ms = []
        self.started = False
        self.lock = threading.Lock()

    def start(self, loop=None):
        if self.started:
            return
        self.loop = loop
        args = [str(self.w), str(self.h), str(int(self.cfg["fps"]))]
        if self.cfg.get("process", True):
            import subprocess
            import sys
            self.proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "tvdsl", "worker.py")] + args,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         cwd=HERE, close_fds=True)
            self.inp, self.out = self.proc.stdin, self.proc.stdout
        else:
            from tvdsl import worker
            r1, w1 = os.pipe()
            r2, w2 = os.pipe()
            wi, wo = os.fdopen(r1, "rb", 0), os.fdopen(w2, "wb")
            self.inp, self.out = os.fdopen(w1, "wb"), os.fdopen(r2, "rb")
            self.proc = threading.Thread(target=worker.run, args=(wi, wo, self.w, self.h, int(self.cfg["fps"])),
                                         daemon=True, name="tv-render")
            self.proc.start()
        self.thread = threading.Thread(target=self._pump, daemon=True, name="tv-pump")
        self.thread.start()
        self.started = True

    def send(self, msg):
        if not self.started:
            self.start(self.loop)
        with self.lock:
            try:
                self.inp.write((json.dumps(msg) + "\n").encode())
                self.inp.flush()
            except (OSError, ValueError, BrokenPipeError) as e:
                self.log({"tv": "render_send_failed", "error": str(e)[:120]})

    def _read(self, n):
        b = b""
        while len(b) < n:
            c = self.out.read(n - len(b))
            if not c:
                raise EOFError
            b += c
        return b

    def _pump(self):
        import numpy as np
        try:
            while True:
                k = self._read(1)
                if k == b"F":
                    ms = int.from_bytes(self._read(4), "little") / 10.0
                    n = int.from_bytes(self._read(4), "little")
                    data = self._read(n)
                    if n != self.w * self.h * 3:
                        continue
                    img = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)
                    self.frames += 1
                    self.render_ms.append(ms)
                    if len(self.render_ms) > 600:
                        del self.render_ms[:300]
                    self._deliver(self._prep(img))
                elif k == b"I":
                    self._deliver(self._prep(None))
                elif k == b"E":
                    n = int.from_bytes(self._read(4), "little")
                    self.log({"tv": "render_error", "error": self._read(n).decode("utf8", "replace")})
        except (EOFError, OSError, ValueError):
            return

    def _prep(self, img):
        if self.prepare is None:
            return img
        try:
            return self.prepare(img)
        except Exception as e:  # noqa: BLE001
            self.log({"tv": "prepare_error", "error": f"{type(e).__name__}: {e}"[:160]})
            return img

    def _deliver(self, img):
        if self.loop is not None and not self.loop.is_closed():
            try:
                self.loop.call_soon_threadsafe(self.deliver, img)
            except RuntimeError:
                pass
        else:
            self.deliver(img)

    def stop(self):
        if not self.started:
            return
        self.send({"op": "stop"})
        p = self.proc
        if hasattr(p, "wait"):
            try:
                p.wait(1.5)
            except Exception:  # noqa: BLE001
                p.kill()
                p.wait(1.0)
        for f in (self.inp, self.out):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        self.started = False


# ================================================================================================ the agent
async def hedged_stream(door, model, system, user, max_s, hedge_s, rec=None, hedge_models=None, keep=None):
    """door.stream with request hedging: yields the events of the first request that produces text.
    hedge_models[i]: the model of hedge i (None = the same model). Haiku's hidden 'thinking' spells are
    prompt-correlated (a duplicate Haiku request often thinks too: a follow-up patch took 13 s to its first line), so
    the last hedge goes to another model.
    keep: a dict; when a hedge wins, the PRIMARY request is not cancelled but runs on to the end, its
    text collected in keep["text"] and keep["task"] (awaitable) -- the caller can apply it as a correction"""
    q = asyncio.Queue()
    pumps = []
    hm = list(hedge_models or [])

    def launch(k):
        mdl = (hm[k - 1] if 0 < k <= len(hm) and hm[k - 1] else model)

        async def pump():
            try:
                async for ev in door.stream(model=mdl, system=system, user=user, max_s=max_s, tag="tv"):
                    if k == 0 and keep is not None:
                        if "delta" in ev:
                            keep["text"] = keep.get("text", "") + ev["delta"]
                        elif ev.get("done"):
                            keep["ok"] = bool(ev.get("ok"))
                        if keep.get("detached"):
                            continue
                    await q.put((k, ev))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                await q.put((k, {"done": True, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}))
        pumps.append(asyncio.ensure_future(pump()))

    launch(0)
    t0 = time.monotonic()
    hedges = list(hedge_s or [])
    winner = None
    failed = set()
    try:
        while True:
            timeout = None
            if winner is None and hedges:
                timeout = max(0.0, hedges[0] - (time.monotonic() - t0))
            try:
                k, ev = await asyncio.wait_for(q.get(), timeout) if timeout is not None else await q.get()
            except asyncio.TimeoutError:
                hedges.pop(0)
                launch(len(pumps))
                if rec is not None:
                    rec["hedges"] = len(pumps) - 1
                continue
            if winner is None:
                if "delta" in ev:
                    winner = k
                    if rec is not None:
                        rec["winner"] = k
                        if k:
                            rec["winner_model"] = hm[k - 1] if k <= len(hm) and hm[k - 1] else model
                    for j, p in enumerate(pumps):
                        if j != k and not (j == 0 and keep is not None):
                            p.cancel()
                    if k and keep is not None:
                        keep["detached"] = True
                        keep["task"] = pumps[0]
                elif ev.get("done"):
                    failed.add(k)
                    if len(failed) == len(pumps) and not hedges:
                        yield ev
                        return
                    if len(failed) == len(pumps):
                        hedges[0] = 0.0          # every request failed early: hedge now
                    continue
                else:
                    continue
            if k != winner:
                continue
            yield ev
            if ev.get("done"):
                return
    finally:
        for j, p in enumerate(pumps):
            if not p.done() and not (j == 0 and keep is not None and keep.get("detached")):
                p.cancel()


def _door():
    from claude_door import door
    return door


class TVAgent:
    def __init__(self, surface=None, on_state=None, cfg=None, log=None, door=None, sink=None, loop=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.surface = surface            # callable -> the tv surface (put_frame / clear) or None
        self.sink = sink                  # optional extra frame consumer (files / viewer): sink(rgb|None)
        self.on_state = on_state or (lambda d: None)
        self.log = log or (lambda d: None)
        self._door = door
        self.doc = Doc()                  # parent-side mirror (descriptions, program text for patches)
        self.doc.empty = True
        self.loop = loop
        self.host = RenderHost(self.cfg, self._deliver, self.log, prepare=self._prepare)
        self._enc_prev = None             # the last frame encoded on the pump thread (diff base)
        self.convo = []                   # [(who, text, mono)]
        self.n = 0
        self.cur = None                   # {"id", "task", "kind"}
        self.queued_words = []
        self.tap_task = None
        self.board_after_show = False
        self.last_update_at = -1e9
        self.last_show_at = -1e9
        self.timeline = []                # [(mono, line)] committed lines (eval / recordings)
        self.history = []                 # per request stats
        self.frame_count = 0
        self.closed = False
        self.scene_id = 0
        self.n_patches = 0
        self.last_state = None
        self._partial = {}                # who -> [text, timer handle] (delta transcripts from the server voice path)
        self.up_task = None               # the careful version of the current new picture (cfg upgrade)
        self.fix_task = None              # the strong patch correcting a quick hedge patch
        self.careful_state = None         # None | "pending" (this picture still owes its careful pass) | "done"
        self.last_tap = None              # {"t", "words", "rid", "geom"}: the tool's mode=update reuses a tap on the same words
        self._tmp = None
        self.changes = []                 # the user's changes to the picture that is up [(words, note)]

    # ---- frames
    def _prepare(self, img):
        """pump thread: diff + encode the frame for the surface (so the server loop only stamps and sends it)"""
        s = self.surface() if callable(self.surface) else self.surface
        if img is None or s is None or not hasattr(s, "encode_frame"):
            self._enc_prev = None
            return img
        enc = s.encode_frame(img, self._enc_prev)
        if enc is not None:
            self._enc_prev = enc["frame"]
        return {"enc": enc, "img": img, "surface": s}

    def _deliver(self, item):
        img = item["img"] if isinstance(item, dict) else item
        try:
            s = self.surface() if callable(self.surface) else self.surface
            if s is not None:
                if img is None:
                    s.clear()
                elif isinstance(item, dict) and item["surface"] is s:
                    s.commit_frame(item["enc"])                    # enc None = nothing changed
                else:
                    s.put_frame(img)
                    self._enc_prev = None
        except Exception as e:  # noqa: BLE001
            self.log({"tv": "surface_error", "error": f"{type(e).__name__}: {e}"[:160]})
        if self.sink is not None:
            try:
                self.sink(img)
            except Exception:  # noqa: BLE001
                pass
        if img is not None:
            self.frame_count += 1

    def _ensure_host(self):
        if not self.host.started:
            try:
                loop = self.loop or asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            self.host.start(loop)

    def _send_lines(self, lines):
        now = time.monotonic()
        for ln in lines:
            for st in parse_line(ln):
                self.doc.apply(st, now)
            self.timeline.append((now, ln))
        if len(self.timeline) > 4000:
            del self.timeline[:2000]
        if not self.cfg.get("render", True):
            return
        self._ensure_host()
        self.host.send({"op": "lines", "lines": list(lines)})

    def _state(self, status, **kw):
        """{type:'tv_state'}: what the page keeps on window.__avatar.tv() and turns into a silent note"""
        phase = "idle" if self.doc.empty else ("drawing" if status == "drawing" or self.busy() and status not in
                                                 ("shown", "updated", "unchanged", "failed", "cleared") else "showing")
        st = {"type": "tv_state", "status": status, "phase": phase, "scene_id": self.scene_id,
              "title": "" if self.doc.empty else self.doc.title, "description": self.describe(), "desc": self.describe(),
              "updated_t": round(time.time(), 3), "n_patches": self.n_patches}
        st.update({k: v for k, v in kw.items() if v is not None})
        self.last_state = st
        try:
            self.on_state(st)
        except Exception as e:  # noqa: BLE001
            self.log({"tv": "state_error", "error": str(e)[:120]})
        return st

    # ---- public
    def prewarm(self, n=2):
        """keep warm model processes for this system prompt (call at session start / when the TV station is near)"""
        try:
            d = self._door or _door()
            d.prewarm(self.cfg["model"], SYSTEM, n=n)
            pm = self.cfg.get("patch_model")
            if pm and pm != self.cfg["model"]:
                d.prewarm(pm, SYSTEM, n=1)
            # the stronger model also checks pictures, fixes failed follow-ups and is the last hedge
            strong = {m for m in (self.cfg.get("upgrade") if self._careful_mode() in ("parallel", "parallel+check") else None,
                                  self.cfg.get("check_model") if self._careful_mode() in ("edit", "parallel+check") else None,
                                  self.cfg.get("fix_model") if self.cfg.get("verify") else None,
                                  *(self.cfg.get("hedge_models") or [])) if m and m != self.cfg["model"]}
            for m in strong:
                d.prewarm(m, SYSTEM, n=2)
        except Exception as e:  # noqa: BLE001
            self.log({"tv": "prewarm_failed", "error": str(e)[:120]})

    def describe(self):
        if self.doc.empty:
            return "The TV is off (nothing shown)."
        return self.doc.describe()

    def busy(self):
        return self.cur is not None and not self.cur["task"].done()

    def settled(self):
        """nothing in flight (eval / tests): no request, no queued tap, no careful version"""
        ft = getattr(self, "fix_task", None)
        return (not self.busy() and not (self.tap_task and not self.tap_task.done()) and not self.upgrading()
                and not (ft is not None and not ft.done()))

    def show(self, concept, context=None, mode="new"):
        """start drawing (supersedes anything in flight); returns at once"""
        concept = re.sub(r"\s+", " ", str(concept or "")).strip()[:400]
        if not self.cfg.get("enabled", True):
            return {"ok": False, "reason": "the TV is switched off in this build"}
        if not concept:
            return {"ok": False, "reason": "no concept given"}
        if self.tap_task and not self.tap_task.done():
            self.tap_task.cancel()
        self.queued_words = []
        self.board_after_show = False
        ctx = context if context is not None else self._recent()
        if mode == "update" and not self.doc.empty:
            lt = self.last_tap
            if lt is not None and time.monotonic() - lt["t"] < 6.0:
                inflight = self.busy() and self.cur.get("kind") == "update" and self.cur.get("id") == lt["rid"]
                if inflight or (lt.get("geom") or 0) >= float(self.cfg["geom_min"]):
                    # the live tap is already making (or just made) this change from the user's own words: the
                    # voice model's update would start it over (+2-3 s) or apply it twice
                    if inflight:
                        self.cur["source"] = "tool"
                    self.last_show_at = time.monotonic()
                    return {"ok": True, "status": "updating" if inflight else "updated", "id": lt["rid"],
                            "next": "The TV is already changing the picture to follow his words; a silent note says what "
                                    "changed once it is done." if inflight else
                                    f"The TV already changed: {self.doc.note or self.doc.title}."}
            user = self._patch_prompt(concept, ctx, from_tool=True)
            kind = "update"
        else:
            user = self._new_prompt(concept, ctx)
            kind = "new"
        self.last_show_at = time.monotonic()
        rid = self._start(kind, user, "tool", concept)
        return {"ok": True, "status": "drawing", "id": rid,
                "next": "It is being drawn on the TV now (first lines appear within a couple of seconds). Say something "
                        "short and natural; a silent note will tell you what the TV shows once it is up."}

    def clear(self):
        self._cancel()
        self._cancel_upgrade()
        if self.tap_task and not self.tap_task.done():
            self.tap_task.cancel()
        self.doc = Doc()
        self.doc.empty = True
        self.timeline.append((time.monotonic(), "clear"))
        if self.host.started:
            self.host.send({"op": "clear"})
        else:
            self._deliver(None)
        self._state("cleared", source="tool")
        return {"ok": True, "status": "cleared"}

    def hear(self, who, text, final=True):
        """conversation tap: every final utterance of either side (who = 'you' | 'avatar'). final=False = a transcript
        DELTA (the server voice path): pieces are joined as they come and taken as an utterance after 1.2 s of quiet"""
        if not final:
            buf = self._partial.get(who)
            if buf is None:
                buf = self._partial[who] = ["", None]
            buf[0] = (buf[0] + str(text or ""))[-1200:]
            if buf[1] is not None:
                buf[1].cancel()
            try:
                loop = self.loop or asyncio.get_running_loop()
                buf[1] = loop.call_later(1.2, self._flush_partial, who)
            except RuntimeError:
                self._flush_partial(who)
            return
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return
        now = time.monotonic()
        if self.convo and self.convo[-1][0] == who and now - self.convo[-1][2] < 2.5:
            self.convo[-1] = (who, (self.convo[-1][1] + " " + text)[-600:], now)
        else:
            self.convo.append((who, text[-600:], now))
        if len(self.convo) > 40:
            del self.convo[:20]
        if who != "you" or not self.cfg.get("tap", True) or self.doc.empty or self.closed:
            return
        if len(text.split()) < int(self.cfg["tap_min_words"]) or BACKCHANNEL.match(text):
            return
        if now - self.last_show_at < 1.0 and self.busy() and self.cur["kind"] == "new":
            return                     # the words that asked for this very picture
        self.queued_words.append(text)
        if self.tap_task and not self.tap_task.done():
            self.tap_task.cancel()
        try:
            self.tap_task = asyncio.ensure_future(self._tap_later())
        except RuntimeError:
            pass

    def _flush_partial(self, who):
        buf = self._partial.pop(who, None)
        if buf and buf[0].strip():
            self.hear(who, buf[0], final=True)

    async def _tap_later(self):
        await asyncio.sleep(float(self.cfg["tap_debounce_s"]))
        # a new scene in flight: its words wait until it is up
        while self.busy() and self.cur["kind"] == "new":
            await asyncio.sleep(0.1)
        if self.upgrading():
            self._cancel_upgrade()               # the user's change lands now; the careful pass follows on it
        gap = time.monotonic() - self.last_update_at
        if gap < float(self.cfg["tap_min_gap_s"]):
            await asyncio.sleep(float(self.cfg["tap_min_gap_s"]) - gap)
        words = " ".join(self.queued_words)[-800:]
        if not words or self.doc.empty:
            return
        if self.busy() and self.cur["kind"] in ("review", "check"):
            self._cancel()                       # the user's words win over the self-review / the careful check
        if self.busy() and self.cur["kind"] == "update":
            # supersede the in-flight patch: it is asked again with every word since
            prev = self.cur.get("words") or ""
            words = (prev + " " + words).strip()[-800:]
            self._cancel()
        self.queued_words = []
        user = self._patch_prompt(words, self._recent(), from_tool=False)
        self.last_update_at = time.monotonic()
        rid = self._start("update", user, "tap", words)
        self.last_tap = {"t": time.monotonic(), "words": words, "rid": rid, "geom": None}

    def _review_after(self, concept):
        """start the self-review of the picture that was just drawn (runs as the current request: anything new wins)"""
        prog = self.doc.program_text()
        user = (f"REVIEW. This picture (CURRENT PROGRAM) was drawn in a hurry for: \"{concept}\"\n\nCURRENT PROGRAM:\n{prog}\n\n"
                "Check it like a strict teacher of this subject: (1) is every fact, number, direction, angle, label and "
                "step scientifically / mathematically right (and computed, not typed)? (2) is the essential idea really "
                "drawn (every part the request names) and does the process visibly happen over time? (3) do texts overlap "
                "shapes or each other or leave the screen? If it is all right, output exactly none. Otherwise output ONLY "
                "fix lines (set / changed objects reusing their IDs / new objects / del), no commentary.")
        try:
            loop = self.loop or asyncio.get_running_loop()
        except RuntimeError:
            return
        def go():
            if self.closed or self.busy() or self.doc.empty:
                return
            self._start("review", user, "review", concept)
        loop.call_soon(go)

    async def close(self):
        self.closed = True
        self._cancel()
        self._cancel_upgrade()
        if self.tap_task and not self.tap_task.done():
            self.tap_task.cancel()
        self.host.stop()

    def stats(self):
        rm = sorted(self.host.render_ms)
        p = lambda q: rm[min(len(rm) - 1, int(q * (len(rm) - 1)))] if rm else None
        return {"frames": self.host.frames, "render_ms_p50": p(0.5), "render_ms_p90": p(0.9),
                "requests": self.history[-20:], "desc": self.describe()}

    # ---- prompts
    def _recent(self):
        k = int(self.cfg["context_turns"])
        return [(w, t) for w, t, _ in self.convo[-k:]]

    @staticmethod
    def _fmt_ctx(ctx):
        out = []
        for item in ctx or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                who = "user" if item[0] in ("you", "user") else "teacher"
                out.append(f"{who}: {str(item[1])[:300]}")
            elif isinstance(item, str):
                out.append(item[:300])
        return "\n".join(out[-10:])

    def _new_prompt(self, concept, ctx):
        c = self._fmt_ctx(ctx)
        # the request goes LAST and is followed by the no-deliberation line: long, many-case requests (the voice model
        # writes those) otherwise made Haiku think silently for 3-6 s before its first line
        return ((f"(conversation so far, latest last:\n{c}\n)\n" if c else "") + f"request: {concept}\n"
                "One picture; several cases become stages over time. Write the scene line now, no planning.")

    def _patch_prompt(self, words, ctx, from_tool):
        c = self._fmt_ctx(ctx)
        prog = self.doc.program_text()
        who = "the teacher asks to change the picture" if from_tool else "the user just said"
        # design note (the conversation tap once redrew the TV for a topic the teacher had taken to the board):
        # the tap only CHANGES the picture that is up, and only for words about it; new subjects are the teacher's call
        change = ("first a line note \"<what is different now, one short sentence>\", then only the changes. The "
                  "DRAWING must change, not just captions: prefer set NAME = value on the params above (everything built "
                  "from them follows and glides); if what he wants changed is not a param yet, add param NAME = value "
                  "[lo..hi] and re-declare (same IDs) every object that must use it; make the asked condition really "
                  "true and update every text whose words or numbers no longer fit. Change nothing he did not ask for.")
        how = ("Output " + change + " Output a full new scene only if the subject changed, or exactly none."
               if from_tool else
               "If these words ask about or change something in THIS picture (its objects, numbers, what happens next), "
               "output " + change + " If they are about "
               "anything else (another subject, the chalkboard, small talk), output exactly none: never start a new "
               "scene from conversation alone."
               + (" The teacher is working at the chalkboard now, so be strict: none unless the words are clearly about "
                  "the TV picture." if self.board_after_show else ""))
        return (f"CURRENT PROGRAM:\n{prog}\n" + self._params_line() + self._changes_line()
                + f"\nNEW WORDS ({who}): {words}\n"
                + (f"RECENT CONVERSATION:\n{c}\n" if c else "")
                + how + " Answer instantly, no deliberation: start writing lines now.")

    def _changes_line(self, exclude=None):
        """every change the user already made to this picture (a follow-up's rewrite once quietly put the demand curve
        back, undoing an earlier 'increase demand')"""
        ch = [c for c in self.changes if c[0] != exclude][-6:]
        if not ch:
            return ""
        return ("HIS EARLIER CHANGES (already in the program; they all still apply -- keep every one): "
                + "; ".join(f'"{w[:120]}"' + (f" ({n[:120]})" if n else "") for w, n in ch) + "\n")

    def _params_line(self):
        """PARAMS NOW: the values the picture shows at this moment (a glide in flight counts as its target)"""
        if not self.doc.params:
            return "PARAMS NOW: none (the picture has no params yet)\n"
        return "PARAMS NOW: " + ", ".join(f"{k} = {p.new:g}" + (f" [{p.rng[0]:g}..{p.rng[1]:g}]" if p.rng else "")
                                          for k, p in list(self.doc.params.items())[:24]) + "\n"

    def board_started(self):
        """the teacher took a lesson to the chalkboard after the last picture (the tap gets strict)"""
        self.board_after_show = True

    # ---- running a request
    def _cancel(self):
        if self.cur is not None and not self.cur["task"].done():
            self.cur["task"].cancel()
        self.cur = None

    def _start(self, kind, user, source, words):
        self._cancel()
        self._cancel_upgrade()
        self.n += 1
        rid = f"tv{self.n}"
        cur = {"id": rid, "kind": kind, "words": words, "source": source}
        before = self._snap() if kind == "update" and not self.doc.empty and self.cfg.get("verify") else None
        cur["task"] = asyncio.ensure_future(self._run(rid, kind, user, source, words, before))
        self.cur = cur
        if kind == "new":
            self.changes = []
            self.careful_state = "pending" if self._careful_mode() else None
            self.careful_for = words
            if self._careful_mode() in ("parallel", "parallel+check"):
                up_user = user
                if self.cfg.get("careful_plan"):
                    # (A/B): the careful version has seconds to spare: one comment line naming what the
                    # picture must make visible, then the program
                    up_user = user.replace("Write the scene line now, no planning.",
                                           "This is the CAREFUL version (a quick sketch is already up). First write ONE "
                                           "comment line '# must show: ...' naming the parts, steps and exact relations "
                                           "the picture must make visible and true, then the program; compute every "
                                           "number and position from params.")
                self.up_task = asyncio.ensure_future(self._upgrade(rid, up_user, words))
        return rid

    def _careful_mode(self):
        m = self.cfg.get("careful") or "parallel"
        if m in ("parallel", "parallel+check") and not (self.cfg.get("upgrade") and self.cfg["upgrade"] != self.cfg["model"]):
            m = "edit" if m == "parallel+check" else None
        if m == "edit" and not self.cfg.get("check_model"):
            m = None
        return m

    def _cancel_upgrade(self):
        t = getattr(self, "up_task", None)
        if t is not None and not t.done():
            t.cancel()
        self.up_task = None
        ft = getattr(self, "fix_task", None)
        if ft is not None and not ft.done():
            ft.cancel()
        self.fix_task = None

    def upgrading(self):
        t = getattr(self, "up_task", None)
        return t is not None and not t.done()

    # ---- soft replace, stills, change check
    def _snap(self):
        try:
            from tvdsl import verify
            return verify.snapshot(self.doc)
        except Exception as e:  # noqa: BLE001
            self.log({"tv": "snapshot_failed", "error": str(e)[:120]})
            return None

    def _soft_lines(self, lines):
        """a whole program (first line scene/morph) -> lines that turn the CURRENT picture into it smoothly: the scene
        line becomes `morph` (title/view/loop change, time and objects carry on), objects with the same ID morph,
        params glide to their new values, and every object the new program does not have fades out"""
        out, ids = [], set()
        for ln in lines:
            try:
                sts = parse_line(ln)
            except (ParseError, ExprError, RecursionError, ValueError):
                continue
            if sts and sts[0].op == "scene":
                ln = "morph" + ln.strip()[len("scene"):]
            out.append(ln)
            for st_ in sts:
                if st_.op == "obj" and st_.id:
                    ids.add(st_.id)
        gone = [k for k in self.doc.objs if k not in ids]
        for i in range(0, len(gone), 40):
            out.append("del " + " ".join(gone[i:i + 40]))
        return out

    def _tmpdir(self):
        if self._tmp is None:
            import tempfile
            self._tmp = tempfile.mkdtemp(prefix="tvagent-")
        return self._tmp

    async def _stills(self, name, offsets=(1.6, 5.0), doc=None):
        """render the picture (as it stands, or `doc`) at scene-relative seconds -> one PNG (side by side)"""
        from tvdsl import verify
        d = verify.snapshot(doc if doc is not None else self.doc)
        now = time.monotonic()
        # moments in the scene's own time, never before now's settled state (glides / morphs done)
        times = [max(d.t0 + o, now + GLIDE_SETTLE) if k == len(offsets) - 1 else d.t0 + o for k, o in enumerate(offsets)]
        if len(offsets) == 1:
            times = [now + GLIDE_SETTLE]
        loop = asyncio.get_running_loop()
        paths = await loop.run_in_executor(None, verify.stills, d, times, os.path.join(self._tmpdir(), name))
        return paths

    async def _change(self, before):
        """how much the DRAWING changed since `before` (settled), off the event loop"""
        from tvdsl import verify
        if before is None:
            return None
        after = verify.snapshot(self.doc)
        now = time.monotonic() + GLIDE_SETTLE
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, verify.picture_change, before, after, now)

    @staticmethod
    def _collect(text):
        """model text -> (lines, kind): kind 'none' | 'program' (starts with scene/morph after an optional note) |
        'patch' | 'empty'"""
        lines, note = [], None
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("```"):
                continue
            if ln.lower().strip(" .`'\"") == "none":
                return [], "none"
            try:
                sts = parse_line(ln)
            except (ParseError, ExprError, RecursionError, ValueError):
                continue
            if not sts:
                continue
            if sts[0].op in ("scene", "morph") and any(parse_line(x)[0].op in ("scene", "morph") for x in lines if parse_line(x)):
                break                              # one picture only
            lines.append(ln)
        if not lines:
            return [], "empty"
        heads = [parse_line(x)[0].op for x in lines[:3]]
        if "scene" in heads or "morph" in heads:
            k = heads.index("scene") if "scene" in heads else heads.index("morph")
            notes = lines[:k]
            body = lines[k:]
            objs = sum(1 for x in body for st_ in parse_line(x) if st_.op == "obj")
            if objs >= 3:
                return body + notes, "program"
            return notes + body[1:], "patch"      # a scene line over a fragment: the fragment is a patch
        return lines, "patch"

    def _apply_collected(self, lines, kind):
        if kind == "program":
            self.scene_id += 1
            self.n_patches = 0
            self._send_lines(["hold"] + self._soft_lines(lines))
        elif kind == "patch":
            self._send_lines(["hold"] + lines)

    async def _ask_whole(self, model, user, images, max_s, rec):
        door = self._door or _door()
        text = ""
        t0 = time.monotonic()
        async for ev in door.stream(model=model, system=SYSTEM, user=user, images=images, max_s=max_s, tag="tv"):
            if "delta" in ev:
                if rec.get("ms_first_token") is None:
                    rec["ms_first_token"] = int((time.monotonic() - t0) * 1000)
                text += ev["delta"]
            elif ev.get("done") and not ev.get("ok"):
                rec["error"] = str(ev.get("error"))[:200]
        return text

    async def _escalate(self, rec, words, reason, from_tool):
        """the quick patch did not change the drawing: the stronger model LOOKS at the picture and makes the change"""
        t0 = time.monotonic()
        before = self._snap()
        fx = {"model": self.cfg["fix_model"], "reason": reason, "status": None, "ms": None}
        rec["fix"] = fx
        try:
            imgs = await self._stills(f"fix{rec['id']}", (1.6, 5.0))
        except Exception as e:  # noqa: BLE001
            imgs = []
            fx["error"] = f"stills: {e}"[:120]
        c = self._fmt_ctx(self._recent())
        user = (f"CURRENT PROGRAM:\n{self.doc.program_text()}\n" + self._params_line() + self._changes_line()
                + ("\nThe image shows the TV now (two moments of its animation).\n" if imgs else "\n")
                + f"{'The teacher asks to change the picture' if from_tool else 'The user just said'}: \"{words}\"\n"
                + (f"RECENT CONVERSATION:\n{c}\n" if c else "")
                + f"A quick edit {reason}. Make the DRAWING show what he asked: change the param(s) the drawing is built "
                  "from (set NAME = value), or turn the quantity into a param and re-declare the objects that must use it "
                  "(same IDs, so they move smoothly), or replace / add / delete objects. Make the asked condition really "
                  "true and update every text whose words or numbers no longer fit. First line: note \"<what is "
                  "different now, one short sentence>\". If the picture must be rebuilt, output a complete program whose "
                  "first line is morph \"<title>\" (instead of scene), reusing IDs for what stays. If his words truly "
                  "need no change to the drawing, output exactly none. TVL lines only.")
        text = await self._ask_whole(self.cfg["fix_model"], user, imgs or None, float(self.cfg["fix_max_s"]), fx)
        lines, kind = self._collect(text)
        fx["kind"] = kind
        if kind in ("program", "patch"):
            self._apply_collected(lines, kind)
            fx["check"] = await self._change(before)
            fx["status"] = "applied"
        else:
            fx["status"] = kind
        fx["ms"] = int((time.monotonic() - t0) * 1000)
        return kind in ("program", "patch")

    async def _lint_fix(self, rec, before=None):
        """objects that draw nothing / texts that print '?' go back to the fast model for corrected lines (cheap: only
        when something is broken; measured 2/46 careful versions on the devset)"""
        from tvdsl import verify
        try:
            d = verify.snapshot(self.doc)
            loop = asyncio.get_running_loop()
            probs = await loop.run_in_executor(None, verify.lint, d, time.monotonic() + GLIDE_SETTLE, before)
        except Exception as e:  # noqa: BLE001
            rec["lint_error"] = str(e)[:120]
            return
        if not probs or not self.cfg.get("lint", True):
            return
        rec["lint"] = [p_[0] for p_ in probs]
        user = (f"CURRENT PROGRAM:\n{self.doc.program_text()}\n\nBROKEN LINES (they " + "; ".join(
                    f"{p_[1][:200]} -> {p_[2]}" for p_ in probs) + ")\nOutput corrected versions of ONLY these lines "
                "(same IDs; define any missing value with let), the one other line that fixes it (e.g. the axes line, "
                "same ID), or del ID. Answer instantly, lines only.")
        lr = {"ms_first_token": None}
        text = ""
        async for ev in hedged_stream(self._door or _door(), self.cfg["model"], SYSTEM, user, 8.0,
                                      self.cfg.get("hedge_s") or [], lr, hedge_models=self.cfg.get("hedge_models")):
            if "delta" in ev:
                text += ev["delta"]
        lines, kind = self._collect(text)
        if kind == "patch":
            self._send_lines(lines)
            rec["lint_fixed"] = len(lines)

    def _start_check(self, words, ctx_words=None):
        """the careful pass on the picture as it stands (sketch, or the sketch after the user's change): runs as the
        current request, so anything the user says supersedes it"""
        if self.closed or self.doc.empty or self.busy():
            return
        self.n += 1
        rid = f"tv{self.n}c"
        cur = {"id": rid, "kind": "check", "words": words, "source": "check"}
        cur["task"] = asyncio.ensure_future(self._check(rid, words, ctx_words))
        self.cur = cur

    async def _check(self, rid, concept, ctx_words):
        t0 = time.monotonic()
        model = self.cfg["check_model"]
        rec = {"id": rid, "kind": "check", "source": "check", "words": concept[:200], "model": model,
               "ms_first_token": None, "ms_first_draw": None, "ms_done": None, "lines": 0, "bad": 0, "status": None,
               "error": None}
        try:
            imgs = await self._stills(f"chk{rid}", (1.6, 5.0))
            c = self._fmt_ctx(self._recent())
            user = (f"REQUEST: {concept}\n" + (f"(conversation, latest last:\n{c}\n)\n" if c else "")
                    + (f"Since then he asked: \"{ctx_words}\" (already applied; keep it)\n" if ctx_words else "")
                    + self._changes_line(exclude=ctx_words)
                    + f"\nQUICK SKETCH (CURRENT PROGRAM):\n{self.doc.program_text()}\n\n"
                    "The image shows this sketch on the TV at two moments (left early, right later). Check it like a strict "
                    "teacher of the subject, LOOKING at the image: (1) is everything right (geometry, directions, numbers, "
                    "labels consistent with what is drawn)? (2) is the essential idea there (every part the request names) "
                    "and does the process visibly happen over time? (3) are texts readable, not on top of shapes or each "
                    "other, not cut off? (4) are the quantities a viewer may ask to change params the drawing is built "
                    "from? If it is all right, output exactly none. If it needs fixes, output ONLY fix lines (set / changed "
                    "objects reusing their IDs / new objects / del). If it is fundamentally wrong or misses the idea, "
                    "output a complete better program whose first line is morph \"<title>\" instead of scene, reusing the "
                    "sketch's IDs for what stays. TVL lines only, no commentary.")
            text = await self._ask_whole(model, user, imgs or None, float(self.cfg["check_max_s"]), rec)
            lines, kind = self._collect(text)
            rec["result"] = kind
            if kind in ("program", "patch"):
                self._apply_collected(lines, kind)
                rec["lines"] = len(lines)
                rec["status"] = "checked"
            else:
                rec["status"] = "unchanged" if kind == "none" else "failed"
            self.careful_state = "done"
        except asyncio.CancelledError:
            rec["status"] = "cancelled"
            self._finish(rec, t0, [])
            raise
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec["status"] = "failed"
        self._finish(rec, t0, [])
        if rec["status"] == "checked":
            self._state("updated", id=rid, source="upgrade", ms_done=rec["ms_done"], words="careful version")
        return rec

    async def _upgrade(self, rid, user, words):
        """the careful version of a new picture: written whole by the stronger model, then morphed in"""
        door = self._door or _door()
        model = self.cfg["upgrade"]
        t0 = time.monotonic()
        rec = {"id": rid + "u", "kind": "upgrade", "source": "upgrade", "words": words[:200], "model": model,
               "ms_first_token": None, "ms_first_draw": None, "ms_done": None, "lines": 0, "bad": 0, "status": None,
               "error": None}
        text, bad = "", []
        try:
            async for ev in hedged_stream(door, model, SYSTEM, user, float(self.cfg["upgrade_max_s"]), [], rec):
                if "delta" in ev:
                    if rec["ms_first_token"] is None:
                        rec["ms_first_token"] = int((time.monotonic() - t0) * 1000)
                    text += ev["delta"]
                elif ev.get("done") and not ev.get("ok"):
                    rec["error"] = str(ev.get("error"))[:200]
        except asyncio.CancelledError:
            rec["status"] = "cancelled"
            self._finish(rec, t0, bad)
            raise
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        lines, objs, seen_scene = [], 0, False
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("```"):
                continue
            try:
                sts = parse_line(ln)
            except (ParseError, ExprError, RecursionError, ValueError) as e:
                bad.append((ln[:160], str(e)[:100]))
                continue
            if not sts:
                continue
            if sts[0].op == "scene":
                if seen_scene:
                    break                       # one picture only
                seen_scene = True
            objs += sum(1 for st_ in sts if st_.op == "obj")
            lines.append(ln)
        rec["lines"] = len(lines)
        # swap in only a real picture, and only while the sketch it improves is still what the TV shows
        cur_ok = self.cur is not None and self.cur["id"] == rid and self.cur["task"].done()
        # never a TRUNCATED careful version (dev eval: the 12 s limit cut Sonnet off after 4
        # lines under load, and those 4 lines replaced a complete sketch with two stray atoms)
        if seen_scene and objs >= 3 and not rec["error"] and not self.closed and (cur_ok or self.doc.empty):
            if self.cur is not None and self.cur["id"] == rid and not self.cur["task"].done():
                self._cancel()
            self.scene_id += 1
            self.n_patches = 0
            if self.doc.empty:
                self._send_lines(lines)
            else:
                soft = self._soft_lines(lines)                           # morph, not a cut
                if soft and soft[0].startswith("morph"):
                    soft[0] += " restart=1"                              # its own clock from 0
                self._send_lines(soft)
            rec["status"] = "upgraded"
            try:
                await self._lint_fix(rec)
            except asyncio.CancelledError:
                self._finish(rec, t0, bad)
                raise
            except Exception as e:  # noqa: BLE001
                rec["lint_error"] = str(e)[:120]
        else:
            rec["status"] = "skipped" if not rec["error"] else "failed"
        self._finish(rec, t0, bad)
        if rec["status"] == "upgraded":
            if self._careful_mode() == "parallel+check":
                self._later(lambda: self._start_check(words))
            else:
                self.careful_state = "done"
            self._state("updated", id=rec["id"], source="upgrade", ms_done=rec["ms_done"], words="careful version")
        elif rec["status"] in ("skipped", "failed") and self.careful_state == "pending" and cur_ok:
            self._later(lambda: self._start_check(words))   # no careful version came: look at the sketch instead
        return rec

    async def _correct(self, rid, keep, before, words, rec0, source):
        """the quick (hedge) patch is up; the stronger model's patch for the same words lands as a
        correction: the picture morphs to (the picture before the user's words) + (the strong patch)"""
        from tvdsl import verify
        t0 = time.monotonic()
        rec = {"id": rid + "x", "kind": "correct", "source": "correct", "words": words[:200],
               "model": self.cfg.get("patch_model"), "ms_first_token": None, "ms_first_draw": None, "ms_done": None,
               "lines": 0, "bad": 0, "status": None, "error": None}
        try:
            task = keep.get("task")
            try:
                await asyncio.wait_for(asyncio.shield(task), float(self.cfg["max_s"]))
            except asyncio.TimeoutError:
                task.cancel()
            text = keep.get("text") or ""
            lines, kind = self._collect(text)
            rec["result"] = kind
            if kind not in ("program", "patch") or self.closed or self.doc.empty or not keep.get("ok"):
                rec["status"] = "kept" if kind == "none" else "skipped"     # (never a truncated / failed stream)
            else:
                if kind == "program":
                    target = lines
                else:
                    d2 = verify.snapshot(before)
                    now = time.monotonic()
                    for ln in lines:
                        try:
                            for st_ in parse_line(ln):
                                d2.apply(st_, now)
                        except (ParseError, ExprError, RecursionError, ValueError):
                            continue
                    target = d2.program_text(max_chars=12000).splitlines()
                    if target and target[0].startswith("scene "):
                        target[0] = "morph " + target[0][len("scene "):]
                self._send_lines(["hold"] + self._soft_lines(target))
                rec["lines"] = len(lines)
                rec["status"] = "corrected"
                note = next((parse_line(x)[0].text for x in lines if x.strip().startswith("note ")), None)
                if note:
                    rec["change"] = note[:200]
                    rec0["change"] = note[:200]
                    if self.changes and self.changes[-1][0] == words[:200]:
                        self.changes[-1] = (words[:200], note[:200])
                rec["check"] = await self._change(before)
                await self._lint_fix(rec, before)
                ch = rec["check"] or {}
                if (self.cfg.get("verify") and self.cfg.get("fix_model") and ch.get("geom") is not None
                        and ch["geom"] < float(self.cfg["geom_min"])):
                    # the strong patch did not change the drawing either (fu hold g_prism: 'more dispersion' -> 0.0)
                    reason = ("changed only the captions, not the drawing" if (ch.get("text") or 0) > 0.0005
                              else "did not visibly change the picture")
                    if await self._escalate(rec, words, reason, source == "tool"):
                        rec["check"] = rec["fix"].get("check") or rec["check"]
        except asyncio.CancelledError:
            rec["status"] = "cancelled"
            t = keep.get("task")
            if t is not None and not t.done():
                t.cancel()
            self._finish(rec, t0, [])
            raise
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec["status"] = "failed"
        rec0["correct"] = rec["status"]
        self._finish(rec, t0, [])
        if rec["status"] == "corrected":
            self._state("updated", id=rec["id"], source="correct", ms_done=rec["ms_done"], words=words[:200],
                        change=rec.get("change"))
        return rec

    def _later(self, fn):
        try:
            loop = self.loop or asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_soon(fn)

    async def _run(self, rid, kind, user, source, words, before=None):
        door = self._door or _door()
        model = (self.cfg["model"] if kind == "new" else self.cfg["review"] if kind == "review"
                 else (self.cfg.get("patch_model") or self.cfg["model"]))
        t0 = time.monotonic()
        rec = {"id": rid, "kind": kind, "source": source, "words": words[:200], "model": model, "ms_first_token": None,
               "ms_first_draw": None, "ms_done": None, "lines": 0, "bad": 0, "status": None, "error": None}
        buf = ""
        pending = None          # a new scene's lines until its first object parses
        committed = 0
        saw_none = False
        bad = []
        req_ids = {}                      # id -> kind defined by THIS request
        gen = None
        change_note = None
        morphing = {}

        def take(raw):
            nonlocal pending, committed, saw_none, change_note
            ln = raw.strip()
            if not ln or ln.startswith("```"):
                return
            if ln.lower().strip(" .`'\"") == "none":
                saw_none = True
                return
            try:
                sts = parse_line(ln)
            except (ParseError, ExprError, RecursionError, ValueError) as e:
                bad.append((ln[:160], str(e)[:100]))
                return
            if not sts:
                return
            # one ID used for two different things in the same program (a beam and its text label): the later one
            # gets its own ID instead of silently replacing the first
            if len(sts) == 1 and sts[0].op == "obj" and sts[0].id:
                prev_kind = req_ids.get(sts[0].id)
                if prev_kind is not None and prev_kind != sts[0].kind:
                    new_id = sts[0].id + "_" + sts[0].kind
                    ln2 = re.sub(r"^(\s*\S+\s+)" + re.escape(sts[0].id) + r"\b", lambda m: m.group(1) + new_id, ln, count=1)
                    if ln2 != ln:
                        ln = ln2
                        sts = parse_line(ln)
                for st_ in sts:
                    req_ids.setdefault(st_.id, st_.kind)
            rec["lines"] += 1
            if sts[0].op == "note" and kind == "update" and change_note is None:
                change_note = sts[0].text
            if sts[0].op == "scene" and kind != "new" and not self.doc.empty:
                # an update that rewrites the picture streams as a MORPH (same IDs glide, new objects
                # appear as they come) instead of being held to the end (an 11 s wait); at the end, a real picture
                # (>= 4 objects) also removes what it no longer has, a fragment simply stays a patch
                morphing["on"] = True
                morphing["ids"] = set()
                morphing["n"] = 0
                morphing["title"] = sts[0].text
                if committed == 0:
                    self._send_lines(["hold"])
                self._send_lines(["morph" + ln.strip()[len("scene"):]])
                committed += 1
                return
            if morphing.get("on"):
                for st_ in sts:
                    if st_.op == "obj" and st_.id:
                        morphing["ids"].add(st_.id)
                        morphing["n"] += 1
            if sts[0].op == "scene":
                pending = [ln]
                return
            if pending is not None:
                pending.append(ln)
                if kind != "new" and not self.doc.empty:
                    return              # an update that starts a whole new picture: decided when the stream ends
                if any(s.op == "obj" for s in sts):
                    self.scene_id += 1
                    self.n_patches = 0
                    self._send_lines(pending)
                    committed += len(pending)
                    pending = None
                    if rec["ms_first_draw"] is None:
                        rec["ms_first_draw"] = int((time.monotonic() - t0) * 1000)
                        self._state("drawing", id=rid, source=source)
                return
            if committed == 0 and kind != "new":
                self._send_lines(["hold"])          # the user's change / the review: the automatic sweep stops
            self._send_lines([ln])
            committed += 1
            if rec["ms_first_draw"] is None and any(s.op in ("obj", "param", "del") for s in sts):
                rec["ms_first_draw"] = int((time.monotonic() - t0) * 1000)

        ev_done = None
        keep = None
        try:
            hs, hm = self.cfg.get("hedge_s") or [], self.cfg.get("hedge_models")
            if kind == "update" and model != self.cfg["model"]:
                hs, hm = self.cfg.get("patch_hedge_s") or [], self.cfg.get("patch_hedge_models")
                if self.cfg.get("patch_correct") and before is not None:
                    keep = {}
            gen = hedged_stream(door, model, SYSTEM, user, float(self.cfg["max_s"]),
                                [] if kind == "review" else hs, rec, hedge_models=hm, keep=keep)
            async for ev in gen:
                if "delta" in ev:
                    if rec["ms_first_token"] is None:
                        rec["ms_first_token"] = int((time.monotonic() - t0) * 1000)
                    buf += ev["delta"]
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        take(line)
                elif ev.get("done"):
                    ev_done = ev
            if buf.strip():
                take(buf)
                buf = ""
            if morphing.get("on") and morphing["n"] >= 4:
                gone = [k for k in self.doc.objs if k not in morphing["ids"]]
                for i in range(0, len(gone), 40):
                    self._send_lines(["del " + " ".join(gone[i:i + 40])])
                self.scene_id += 1
                self.n_patches = 0
            elif morphing.get("on") and morphing.get("title") and not change_note:
                self._send_lines([f'note "{morphing["title"]}"'])       # a fragment: its title says what changed
                change_note = morphing["title"]
            if pending is not None and kind != "new" and not self.doc.empty:
                # a new picture from an update: a real one (>= 4 objects) replaces the TV -- morphing from the current
                # one; a fragment (a scene line plus a few objects that lean on the current picture's
                # names) is applied as a patch instead
                objs = sum(1 for ln_ in pending[1:] for st_ in parse_line(ln_) if st_.op == "obj")
                if objs >= 4:
                    self.scene_id += 1
                    self.n_patches = 0
                    self._send_lines(["hold"] + self._soft_lines(pending))
                    committed += len(pending)
                elif len(pending) > 1:
                    title = parse_line(pending[0])[0].text
                    self._send_lines(["hold"] + pending[1:] + ([f'note "{title}"'] if title else []))
                    committed += len(pending) - 1
                if committed and rec["ms_first_draw"] is None:
                    rec["ms_first_draw"] = int((time.monotonic() - t0) * 1000)
                pending = None
            correcting = keep is not None and keep.get("detached") and keep.get("task") is not None
            if correcting:
                # the stronger model's patch is on its way: it corrects this quick one (no lint / escalation here)
                rec["correct"] = "pending"
                self.fix_task = asyncio.ensure_future(self._correct(rid, keep, before, words, rec, source))
            if kind == "update" and committed and self.cfg.get("lint", True) and not correcting:
                await self._lint_fix(rec, before)
            # did the DRAWING change? (seen in testing: the patch rewrote the formula text, the lens stayed)
            if kind == "update" and before is not None and self.cfg.get("verify") and not correcting:
                reason = None
                if committed:
                    ch = await self._change(before)
                    rec["check"] = ch
                    if ch is not None and ch.get("geom") is not None and ch["geom"] < float(self.cfg["geom_min"]):
                        reason = ("changed only the captions, not the drawing" if (ch.get("text") or 0) > 0.0005
                                  else "did not visibly change the picture")
                elif saw_none and source == "tool":
                    reason = "answered that nothing needs to change"
                elif not saw_none:
                    reason = "produced no usable lines"
                if reason and self.cfg.get("fix_model"):
                    ok = await self._escalate(rec, words, reason, source == "tool")
                    if ok:
                        committed += 1
                        rec["check"] = rec["fix"].get("check") or rec.get("check")
                        n2 = None
                        for ln_ in reversed(self.timeline[-60:]):
                            if ln_[1].startswith("note "):
                                n2 = parse_line(ln_[1])[0].text
                                break
                        change_note = n2 or change_note
        except asyncio.CancelledError:
            rec["status"] = "cancelled"
            if keep is not None and keep.get("task") is not None and not keep["task"].done():
                keep["task"].cancel()
            if gen is not None:
                try:
                    await gen.aclose()
                except Exception:  # noqa: BLE001
                    pass
            self._finish(rec, t0, bad)
            raise
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        if ev_done is not None and not ev_done.get("ok"):
            rec["error"] = rec["error"] or str(ev_done.get("error"))[:200]
        if pending is not None:
            rec["error"] = rec["error"] or "the new picture had no drawable lines (kept the previous one)"
        if committed:
            rec["status"] = "shown" if kind == "new" else "reviewed" if kind == "review" else "updated"
        elif saw_none:
            rec["status"] = "unchanged"
        else:
            rec["status"] = "failed"
        if change_note:
            rec["change"] = change_note[:200]
        self._finish(rec, t0, bad)
        if rec["status"] in ("updated", "reviewed"):
            self.n_patches += 1
        if kind == "update" and rec["status"] == "updated":
            self.changes.append((words[:200], (rec.get("change") or "")[:200]))
        if source == "tap" and self.last_tap is not None and self.last_tap.get("rid") == rid:
            self.last_tap["geom"] = (rec.get("check") or {}).get("geom")
            self.last_tap["t"] = time.monotonic()
        if kind == "review":
            if rec["status"] == "reviewed":
                self._state("updated", id=rid, source="review", ms_done=rec["ms_done"], words="self-review fixes")
            return rec
        if kind == "new" and rec["status"] == "shown" and self.cfg.get("review") and not self.closed:
            self._review_after(words)
        if kind == "new" and rec["status"] == "shown" and self._careful_mode() == "edit":
            self._later(lambda: self._start_check(words))
        if kind == "update" and rec["status"] == "updated" and self.careful_state == "pending" and self._careful_mode():
            # the user's change landed on the sketch before its careful version: the careful pass now looks at the result
            cw = getattr(self, "careful_for", "") or words
            self._later(lambda: self._start_check(cw, words))
        extra = {"error": rec["error"] or "no usable lines"} if rec["status"] == "failed" else {}
        src = self.cur.get("source", source) if (self.cur is not None and self.cur.get("id") == rid) else source
        if not (src == "tap" and rec["status"] == "unchanged"):
            self._state(rec["status"], id=rid, source=src, ms_first=rec["ms_first_draw"], ms_done=rec["ms_done"],
                        words=words[:200], change=rec.get("change"), **extra)
        return rec

    def _finish(self, rec, t0, bad):
        rec["ms_done"] = int((time.monotonic() - t0) * 1000)
        rec["t_end"] = time.monotonic()
        rec["bad"] = len(bad)
        if bad:
            rec["bad_lines"] = bad[:6]
        self.history.append(rec)
        if len(self.history) > 200:
            del self.history[:100]
        self.log({"tv": "request", **{k: v for k, v in rec.items() if k != "bad_lines"}})


# ================================================================================================ dev sinks
class FileSink:
    """before the tv surface exists: keep the latest frame as a PNG (and optionally every frame for an MP4)"""

    def __init__(self, path, every_s=0.5):
        self.path = path
        self.every = every_s
        self.last = 0.0

    def __call__(self, img):
        now = time.monotonic()
        if img is None or now - self.last < self.every:
            return
        self.last = now
        from PIL import Image
        tmp = self.path + ".tmp.png"
        Image.fromarray(img).save(tmp)
        os.replace(tmp, self.path)


def replay(timeline, times, w=512, h=288, t_origin=None):
    """render a recorded timeline [(mono, line)] at the given seconds after t_origin -> [rgb] (eval, recordings)"""
    from tvdsl.render import Renderer
    import numpy as np
    import cv2
    R = Renderer(w, h)
    t_origin = timeline[0][0] if (t_origin is None and timeline) else (t_origin or 0.0)
    out = []
    for T in times:
        now = t_origin + T
        doc = Doc()
        doc.empty = True
        last_scene_at = None
        prev_doc = None
        for mono, line in timeline:
            if mono > now:
                break
            if line == "clear":
                doc = Doc()
                doc.empty = True
                continue
            for st in parse_line(line):
                if st.op == "scene":
                    if not doc.empty:
                        prev_doc = doc
                        doc = Doc()
                    last_scene_at = mono
                doc.apply(st, mono)
        if doc.empty:
            out.append(R.idle(now))
            continue
        img = R.render(doc, now)
        if prev_doc is not None and last_scene_at is not None and now - last_scene_at < 0.5:
            u = (now - last_scene_at) / 0.5
            img = cv2.addWeighted(R.render(prev_doc, now), 1 - u, img, u, 0.0)
        out.append(img)
    return out
