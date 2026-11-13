"""Affect: the avatar's own feelings on its face and body, for ANY emotional moment (not a list of phrases).

The sealed emotions session (job news -> grief) showed the voice being right in words while the face kept its mild
idle smile and the body its idle stance: the voice model almost never calls `expression` on its own. Here a cheap
classifier (a fast Claude model via the `claude` CLI, app/claude_door.py) reads what the user just said and the first words of the avatar's reply, and
names the feeling the character shows while it answers. The page plays it (app/page.html `affect` -> app/affect.js):
a sustained face + posture state held through the reply, an onset accent, the gesture only while the body is free.

the verdict is CONTINUOUS — valence / arousal / dominance (-1..1), an intensity, optional facets (what
the three dimensions cannot carry: laugh, tilt, look_down, look_away, one_brow, nose, heavy_lids, tense) and a free
label — so mixed and unusual feelings (bashful pride, nervous excitement, mock outrage, wistful) need no list entry.
Two judgements per turn: a PRE one on the user's words alone as soon as the turn is transcribed (the face can start with the
first words of the reply), then the REPLY one on the first words of the answer (refines; the page glides to it).

    A = Affect(send=session.send, log=print)
    A.user_said("I got the job!")            # the user's final words (the realtime transcript) -> the 'pre' verdict
    A.avatar_delta("Oh my gosh, that's ")    # the reply as it streams; fires once per reply after a few words
"""
from __future__ import annotations

import asyncio
import os
import re
import time

# the named feelings stay as anchors (logs, the action memory, old pages); the face is driven by the dimensions
ANCHORS = {   # feel: (v, a, d)
    "happy": (0.75, 0.35, 0.3), "excited": (0.8, 0.85, 0.35), "proud": (0.7, 0.45, 0.75), "love": (0.85, 0.2, 0.1),
    "laughing": (0.85, 0.75, 0.3), "surprised": (0.1, 0.85, -0.1), "curious": (0.3, 0.4, 0.1), "thinking": (0.0, 0.1, 0.1),
    "calm": (0.3, -0.5, 0.1), "shy": (0.3, 0.1, -0.6), "embarrassed": (-0.25, 0.35, -0.6), "worried": (-0.5, 0.4, -0.45),
    "sad": (-0.75, -0.4, -0.4), "scared": (-0.7, 0.8, -0.7), "angry": (-0.7, 0.7, 0.6), "frown": (-0.4, 0.15, 0.3),
    "sleepy": (0.0, -0.85, -0.2), "silly": (0.6, 0.6, 0.2), "neutral": (0.0, 0.0, 0.0)}
FEELS = tuple(ANCHORS)
GESTURES = ("none", "nod", "clap", "hand_on_heart", "shrug", "thumbs_up", "tail_wag", "ear_wiggle", "bow", "shake_head",
            # the body is what reads at stream size — the rest of the page's expressive gestures
            "raise_both", "think", "scratch_head", "startle_hands", "lean_in", "lean_back", "sigh", "eye_roll")
GESTURE_HINT = ("raise_both = both arms up (triumph, big joy), think = paw to chin (pondering), scratch_head = puzzled, "
                "startle_hands = paws jump up (startled, scared), lean_in = eager / curious / intent, lean_back = taken aback / "
                "repelled, sigh = weary / disappointed / relieved, eye_roll = exasperated; weighing something up = think + look_away; a head tilt is the tilt facet")
# (cover_eyes left out: at grief it read as peekaboo, devloop r4)
FACETS = ("laugh", "surprise", "tilt", "look_down", "look_away", "one_brow", "nose", "heavy_lids", "tense")
SYS = ("You direct the face and body of a warm, natural cartoon character in a live conversation. Given what the person "
       "just said and (if it has started) how the character starts to answer, decide the feeling the character should SHOW "
       "while it answers, matching the emotional moment (joy at good news, sympathy and sadness at loss, concern at worry, "
       "surprise, amusement at a joke, playfulness, calm attention for plain questions). Any feeling or blend is allowed "
       "(mixed feelings too: say them in the label and place them on the dimensions).\n"
       "Output exactly one line: LABEL | V A D | INTENSITY | GESTURE | FACETS\n"
       "LABEL: the feeling in 1-3 words. V = valence (-1 very negative .. 1 very positive), A = arousal (-1 sleepy/low "
       "energy .. 1 very energetic/alarmed), D = dominance (-1 small, meek, overwhelmed .. 1 confident, in control; anger, indignation, annoyance, "
       "frustration, disgust, pride are HIGH D (0.3-0.8); fear, worry, shame, sadness are LOW D). When the user is upset about something "
       "and the character takes their side (outrage, annoyance on their behalf), show that feeling, not sadness. "
       "INTENSITY 0.2-1.0: how strongly to show it (0.3 ordinary friendly talk, 0.6 a clear feeling, 0.9 big news or a "
       "big joke). GESTURE: one of " + ", ".join(GESTURES) + " (one body gesture that fits, or none; never a cheerful "
       "gesture at sad news; none for mild moments; " + GESTURE_HINT + "). FACETS: zero or more of " + ", ".join(FACETS) + " separated by "
       "commas, or none (laugh = laughing; surprise = startled / astonished / amazed (wide eyes, open mouth); tilt = head tilt; look_down / look_away = averted eyes; one_brow = one brow "
       "raised; nose = wrinkled nose; heavy_lids = drooping lids; tense = tensed up).\n"
       "Example: delighted | 0.9 0.7 0.4 | 0.9 | raise_both | laugh\n"
       "Plain factual talk: attentive | 0.1 0 0 | 0.3 | none | none. No other words, no field names.")
NUM = r"(-?[01](?:\.\d+)?|-?\.\d+)"
LINE = re.compile(r"^\s*([A-Za-z][A-Za-z ,'\-/]{0,40}?)\s*\|\s*" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s*\|\s*" + NUM +
                  r"\s*\|\s*([a-z_]+)\s*(?:\|\s*([a-z_, ]*))?", re.M)
HDR = re.compile(r"\b(?:LABEL|V\s*A\s*D|VAD|INTENSITY|GESTURE|FACETS)\s*:\s*", re.I)
OLD = re.compile(r"\b(" + "|".join(FEELS) + r")\b\s+([01](?:\.\d+)?)\s+(" + "|".join(GESTURES) + r")\b", re.I)


def nearest(v: float, a: float, d: float) -> str:
    return min(ANCHORS, key=lambda k: (ANCHORS[k][0] - v) ** 2 + 0.6 * (ANCHORS[k][1] - a) ** 2 + 0.4 * (ANCHORS[k][2] - d) ** 2)


def parse(txt: str) -> dict | None:
    """the judge's line -> {label, v, a, d, intensity, gesture, facets, feel} (None if unreadable)"""
    # Haiku often echoes the field names ("LABEL: joy | V A D: 0.9 0.7 0.4 | INTENSITY: 0.9 | ...") —
    # one in five verdicts was dropped as unreadable and that reply showed no feeling at all
    txt = HDR.sub("", txt or "")
    m = LINE.search(txt)
    if m:
        cl = lambda x, lo=-1.0, hi=1.0: max(lo, min(hi, float(x)))  # noqa: E731
        v, a, d = cl(m.group(2)), cl(m.group(3)), cl(m.group(4))
        inten = cl(m.group(5), 0.2, 1.0)
        g = m.group(6).lower()
        g = g if g in GESTURES else "none"
        fac = [f for f in re.split(r"[\s,]+", (m.group(7) or "").lower()) if f in FACETS]
        return {"label": m.group(1).strip().lower(), "v": round(v, 2), "a": round(a, 2), "d": round(d, 2),
                "intensity": round(inten, 2), "gesture": g, "facets": fac, "feel": nearest(v, a, d)}
    m = OLD.search(txt or "")
    if m:   # the old one-word form still works (a model that ignores the new format)
        feel = m.group(1).lower(); v, a, d = ANCHORS[feel]
        return {"label": feel, "v": v, "a": a, "d": d, "intensity": max(0.2, min(1.0, float(m.group(2)))),
                "gesture": m.group(3).lower(), "facets": [], "feel": feel}
    return None


class Affect:
    def __init__(self, send, log=None, door=None, enabled: bool | None = None):
        self.send = send
        self.log = log or (lambda *_: None)
        self.door = door
        self.enabled = (os.environ.get("AVATAR_AFFECT", "1") != "0") if enabled is None else enabled
        self.pre_on = os.environ.get("AVATAR_AFFECT_PRE", "1") != "0"
        self.user = ""                 # the user's newest utterance
        self.user_at = 0.0
        self.reply = ""                # the avatar's reply so far
        self.reply_at = 0.0
        self.fired = False
        self.turn = 0                  # the user's turns (a 'pre' verdict for an older turn, or after this turn's reply verdict, is dropped)
        self.reply_turn = -1           # the turn whose reply verdict has been sent
        self.task: asyncio.Task | None = None
        self.pre_task: asyncio.Task | None = None
        self.n = {"calls": 0, "sent": 0, "errors": 0, "ms": [], "pre": 0}
        self.last: dict | None = None
        self.heard_at = -1e9           # the fast recognizer's final (vosk, ~0.3 s after the user's words)
        self.heard_turn = -1

    def _door(self):
        if self.door is None:
            try:
                from claude_door import door as _d  # noqa: PLC0415  (seam S1)
                self.door = _d
            except Exception:  # noqa: BLE001
                self.door = False
        return self.door or None

    def prewarm(self):
        d = self._door()
        if d is not None and self.enabled:
            try:
                d.prewarm("haiku", SYS)
            except Exception as e:  # noqa: BLE001
                self.log(f"affect prewarm failed: {e}")

    def heard(self, text: str, final: bool = True):
        """the user's words from the server's fast streaming recognizer (final lines, ~0.3 s after they stop). The
        realtime voice's own transcript of the turn lands 2-4 s later — after the reply has started — so the 'pre'
        verdict fires from here and the face can be in place by the first words of the answer. The later transcript only
        refines the context (user_said). Echo guard: words that are mostly the avatar's own current reply are ignored."""
        if not final or not self.enabled or not self.pre_on:
            return
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        w = t.lower().split()
        if len(w) < 2:
            return
        now = time.monotonic()
        if self.reply and now - self.reply_at < 6.0:
            rw = set(re.findall(r"[a-z']+", self.reply.lower()))
            if sum(1 for x in w if x in rw) >= 0.6 * len(w):
                return
        # one turn arrives as several recognizer finals at the user's pauses ("here's a joke" / "why don't..." / "they don't
        # have the guts"): pieces of the same turn (no reply between, < 4 s apart) are judged TOGETHER, never a fragment
        # alone (real1: the punchline alone read 'indignant')
        same = self.heard_turn == self.turn and now - self.heard_at < 4.0 and self.reply_turn < self.turn and not self.fired
        self.user = ((self.user + " " + t) if same else t)[-400:]
        self.user_at = self.heard_at = now
        self.reply, self.fired = "", False
        if not same:
            self.turn += 1
        self.heard_turn = self.turn
        self.log(f"affect heard: {t[:80]!r}")
        self._fire(self.user, "", stage="pre")

    def user_said(self, text: str):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return
        now = time.monotonic()
        if self.heard_turn == self.turn and now - self.heard_at < 10.0:
            self.user = text[-400:]                 # the same turn, better words: context for the reply verdict only
            self.heard_turn = -1
            return
        self.user = (self.user + " " + text)[-400:] if now - self.user_at < 3.0 else text[-400:]
        self.user_at = now
        self.reply, self.fired = "", False          # the next avatar words are a new reply
        self.turn += 1
        if self.enabled and self.pre_on:
            self._fire(self.user, "", stage="pre")

    def avatar_delta(self, text: str):
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.reply_at > 2.5 and self.reply and self.fired:
            self.reply, self.fired = "", False      # a new reply without user words between (a note, a follow-up)
        self.reply = (self.reply + str(text or ""))[-400:]
        self.reply_at = now
        if not self.fired and (len(self.reply.split()) >= 4 or re.search(r"[.!?,]", self.reply)):
            self.fired = True
            self._fire(self.user, self.reply)

    def cue(self, user: str, reply: str, stage: str = "reply"):
        """the page's cue: the user's last words (stage 'pre', no reply yet) or + the first words of the reply (fires once per cue)"""
        if not self.enabled:
            return
        if stage == "pre":
            if self.pre_on and (user or "").strip():
                self.user = user[-400:]
                self.turn += 1
                self._fire(self.user, "", stage="pre")
            return
        if reply.strip():
            self.fired = True
            self.reply, self.reply_at = reply[-400:], time.monotonic()   # the echo guard's words
            self._fire(user or self.user, reply)

    def _fire(self, user: str, reply: str, stage: str = "reply"):
        d = self._door()
        if d is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if stage == "pre":
            if self.pre_task and not self.pre_task.done():
                self.pre_task.cancel()
            self.pre_task = loop.create_task(self._judge(d, user, reply, stage, self.turn))
            self.task = self.pre_task if self.task is None or self.task.done() else self.task
            return
        if self.task and not self.task.done() and self.task is not self.pre_task:
            self.task.cancel()
        self.task = loop.create_task(self._judge(d, user, reply, stage, self.turn))

    async def _judge(self, d, user: str, reply: str, stage: str, turn: int):
        t0 = time.monotonic()
        self.n["calls"] += 1
        said = f"\nThe character starts answering: {reply}" if reply else "\n(The character has not started answering yet.)"
        try:
            ev = await d.complete(model="haiku", system=SYS, user=f"The user said: {user or '(nothing yet)'}{said}", max_s=6.0, tag="affect")
            txt = str((ev or {}).get("text") or "")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.n["errors"] += 1
            self.log(f"affect {stage} failed: {e}")
            return
        j = parse(txt)
        if not j:
            self.n["errors"] += 1
            self.log(f"affect {stage}: unreadable {txt[:120]!r}")
            return
        ms = round((time.monotonic() - t0) * 1000)
        self.n["ms"] = (self.n["ms"] + [ms])[-20:]
        self.log(f"affect {stage} {ms} ms: {j['label']} | {j['v']} {j['a']} {j['d']} | {j['intensity']} | {j['gesture']} | {','.join(j['facets'])}")
        if stage == "pre":
            if turn != self.turn or self.reply_turn >= turn:
                return                                  # the reply's own verdict is already in (or a newer turn began)
            self.n["pre"] += 1
        else:
            self.reply_turn = turn
        inten = j["intensity"]
        words = len(reply.split()) if reply else 8
        # held about as long as a short spoken reply; the page holds it while the avatar talks and lets it decay after
        hold_s = max(4.0, min(12.0, 2.5 + words * 0.25 + 4.0 * inten))
        self.last = dict(j, hold_s=round(hold_s, 1), ms=ms, stage=stage)
        plain = abs(j["v"]) < 0.15 and abs(j["a"]) < 0.25 and not j["facets"] and j["gesture"] == "none"
        if plain:
            return                                      # nothing to show: the last feeling decays on its own
        self.n["sent"] += 1
        self.send({"type": "affect", **self.last})

    def stats(self) -> dict:
        return dict(self.n, last=self.last)
