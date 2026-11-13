"""Fast local intent matcher: spoken phrase -> avatar Command (docs/CONTRACT.md "Commands").

The rule table lives in intent_rules.json and is shared with app/intent.js (the same engine, line for line);
tests/intent_vectors.json pins both to identical output.

Precision first: a command only fires when it STARTS a clause (after an optional preamble such as "hey Felix",
"can you", "please", "now") and everything after it is a known modifier ("a little", "please", "twice", ...).
Anything else ("I waved at him", "don't turn around", "can you look up the weather") is ignored and left to
the realtime model, which still sees every word.

Side convention: "your left paw" = the character's own left = rig suffix 'r'. Command `side` values are rig
suffixes ('l' | 'r' | 'both'); `own` repeats it in the character's own terms ('left' | 'right' | 'both').

Usage:
    m = Matcher()                                   # one per session (it keeps "more"/"higher" context)
    m.parse("raise your left paw a little")         # -> [Command, ...]  (complete text)
    st = m.stream()                                 # streaming ASR helper
    st.update("raise your left", final=False, now=t) -> []          (incomplete)
    st.update("raise your left paw", final=False, now=t) -> [hit]   (fires on the partial)
    st.update("raise your left paw", final=True, now=t)  -> []      (already fired; never re-fires)
"""
from __future__ import annotations

import copy
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(HERE, "intent_rules.json")

STABLE_MS = 200          # a 'wait' pattern at the end of a partial fires once the partial is this stable
# design note: the fast matcher acts on a partial only once the user's words have stopped growing this long —
# never on the first words of a sentence that is still coming ("jump and rotate backward…", "flip the laptop lid open") —
# and not at all when a clause with content words matched nothing ("turn around and talk to me"): the voice model has it
HOLD_MS = 450
QUIET_MS = 250           # ... and the mic itself has been quiet this long (vosk partials lag the voice mid-sentence)
VOICED_RMS = 400         # int16 RMS of a 20 ms frame that counts as the user's voice
FILLER = set("please thanks thank you me now okay ok so like just for a an the to and then can could would will muse hey hi oh um uh "
             "right again too also it that this i my your yeah yes well go ahead let's lets do".split())
NEG = {"don't", "dont", "not", "never", "no", "stop", "without", "instead", "actually"}
REPEAT_WINDOW_MS = 1500  # identical command from a new ASR segment within this window = the same utterance
CTX_MS = 20000           # "more" / "higher" refers to the last command for this long
OWN = {"l": "right", "r": "left", "both": "both"}


def _r3(x: float) -> float:
    return math.floor(x * 1000 + 0.5) / 1000   # same as JS Math.floor(x*1000+0.5)/1000


def normalize(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("‘", "'").replace("'", "")
    t = re.sub(r"[.!?;:]+", " | ", t)   # commas are not clause breaks (ASR has none; captions do)
    t = re.sub(r"[^a-z0-9| ]+", " ", t)
    return [w for w in t.split(" ") if w]


class Rules:
    def __init__(self, data: dict):
        self.d = data
        self.alias = data["alias"]
        self.names = set(data["names"])
        self.pre = set(data["pre"])
        self.anchors = set(data["anchors"])
        self.split = set(data["split"])
        self.keep = set(data["keep"])
        self.gerund = data["gerund"]
        self.mods = data["mods"]
        self.classes = dict(data["classes"])
        self.classes["emo"] = data["emo"]
        self.emo_alias = data["emo_alias"]
        self.adjust = data["adjust"]
        self.ranges = data["ranges"]
        self.unit = set(data["unit_params"])
        self.noscale = data["noscale"]
        # phrases: first token -> [(tokens, replacement)] longest first (stable for equal length)
        self.phr: dict[str, list] = {}
        for i, (src, dst) in enumerate(data["phrases"]):
            toks = src.split(" ")
            self.phr.setdefault(toks[0], []).append((len(toks), i, toks, dst))
        for k in self.phr:
            self.phr[k].sort(key=lambda e: (-e[0], e[1]))
            self.phr[k] = [(e[2], e[3]) for e in self.phr[k]]
        self.rules = []
        for r in data["rules"]:
            pats = []
            for p in r["p"]:
                body, _, extra = p.partition(" => ")
                fixed = {}
                for kv in extra.split(" ") if extra else []:
                    k, _, v = kv.partition("=")
                    fixed[k] = v
                elems = []
                for tok in body.split(" "):
                    opt = tok.startswith("?")
                    elems.append((opt, (tok[1:] if opt else tok).split("/")))
                pats.append({"src": p, "elems": elems, "fixed": fixed, "wait": p in r.get("wait", [])})
            self.rules.append({"r": r, "pats": pats})
        # gesture/pose name -> parts template (for dedupe keys of commands from other sources)
        self.act = Act(data["act"]) if data.get("act") else None
        if self.act is not None:
            self.act.names = set(data["names"])
        self.name_parts: dict[str, list] = {}
        for r in data["rules"]:
            c = r["c"]
            if c.get("name") and not str(c["name"]).startswith("$") and c.get("parts"):
                self.name_parts.setdefault(c["name"], c["parts"])


_RULES: Rules | None = None


def load_rules(path: str = RULES_PATH) -> Rules:
    global _RULES
    if _RULES is None or path != RULES_PATH:
        with open(path, encoding="utf-8") as f:
            r = Rules(json.load(f))
        if path != RULES_PATH:
            return r
        _RULES = r
    return _RULES


def apply_phrases(R: Rules, toks: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(toks):
        hit = None
        for ptoks, dst in R.phr.get(toks[i], ()):
            if toks[i:i + len(ptoks)] == ptoks:
                hit = (len(ptoks), dst)
                break
        if hit:
            out.append(hit[1])
            i += hit[0]
        else:
            out.append(toks[i])
            i += 1
    return out


def tokenize(R: Rules, text: str) -> list[str]:
    toks = [R.alias.get(w, w) for w in normalize(text)]
    return apply_phrases(R, toks)


def apply_phrases_spans(R: Rules, toks: list[str]):
    """apply_phrases + for each output token the raw tokens it came from"""
    out, raw, i = [], [], 0
    while i < len(toks):
        hit = None
        for ptoks, dst in R.phr.get(toks[i], ()):
            if toks[i:i + len(ptoks)] == ptoks:
                hit = (len(ptoks), dst)
                break
        if hit:
            out.append(hit[1])
            raw.append(toks[i:i + hit[0]])
            i += hit[0]
        else:
            out.append(toks[i])
            raw.append([toks[i]])
            i += 1
    return out, raw


def _alt_match(R: Rules, alt: str, tok: str):
    """-> (ok, capture-key, value)"""
    if alt.startswith("$"):
        cls = alt[1:]
        tab = R.classes.get(cls, {})
        if tok in tab:
            return True, cls, tab[tok]
        return False, None, None
    return (tok == alt), None, None


def _match(R: Rules, elems, toks, i, j, caps, out):
    if j == len(elems):
        out.append((i, dict(caps)))
        return
    opt, alts = elems[j]
    if i < len(toks):
        for alt in alts:
            ok, key, val = _alt_match(R, alt, toks[i])
            if ok:
                c2 = dict(caps)
                if key and key not in c2:
                    c2[key] = val
                _match(R, elems, toks, i + 1, j + 1, c2, out)
    if opt:
        _match(R, elems, toks, i, j + 1, caps, out)


def _tail(R: Rules, toks: list[str]):
    """All tokens must be modifiers (or names). -> mods dict or None"""
    m = {"int": 1.0, "rep": None, "hold": None}
    for t in toks:
        if t in R.names:
            continue
        mod = R.mods.get(t)
        if mod is None:
            return None
        if mod[0] == "int":
            m["int"] *= mod[1]
        elif mod[0] == "rep":
            m["rep"] = mod[1]
        elif mod[0] == "hold":
            m["hold"] = mod[1]
    return m


def _clamp(R: Rules, k: str, v: float) -> float:
    if k in R.ranges:
        lo, hi = R.ranges[k]
    elif k in R.unit:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = -1.0, 1.0
    return min(hi, max(lo, v))


def _scaled(R: Rules, k: str) -> bool:
    for s in R.noscale:
        if (s.startswith("_") and k.endswith(s)) or (not s.startswith("_") and k.startswith(s)):
            return False
    return True


def build(R: Rules, rule: dict, caps: dict, mods: dict) -> dict | None:
    c = copy.deepcopy(rule["c"])
    # side
    side = None
    if "side" in caps:
        side = caps["side"]
    elif rule.get("pside") and any(caps.get(k) == 2 for k in ("paw", "ear", "eye", "brow")):
        side = "both"
    else:
        side = rule.get("side")
    lr = caps.get("lr") or caps.get("dir")
    if c.get("side") == "$lrside":
        if lr not in ("left", "right"):
            return None
        side = "r" if lr == "left" else "l"
    if lr is None and rule.get("lrdefault"):
        lr = rule["lrdefault"]
    sides = ["l", "r"] if side == "both" else ([side] if side in ("l", "r") else [])
    k = 1.0 if rule.get("noint") else mods["int"]
    if "targets" in c:
        t2 = {}
        for key, val in c["targets"].items():
            keys = [key.replace("$", s) for s in sides] if "$" in key else [key]
            if val in ("$yaw", "$lat", "$roll"):
                if lr not in ("left", "right"):
                    return None
                mag = {"$yaw": 0.55, "$lat": 0.6, "$roll": 0.5}[val]
                val = mag if lr == "left" else -mag     # own left = toward screen-right = +
            for kk in keys:
                v = float(val) * (k if _scaled(R, kk) else 1.0)
                t2[kk] = _r3(_clamp(R, kk, v))
        if not t2:
            return None
        c["targets"] = t2
    if "parts" in c and isinstance(c["parts"], list):
        ps = []
        for p in c["parts"]:
            for pp in ([p.replace("$", s) for s in sides] if "$" in p else [p]):
                if pp not in ps:
                    ps.append(pp)
        c["parts"] = ps
    if c.get("side") in ("$", "$lrside"):
        c["side"] = side
    if c.get("side") in ("l", "r", "both"):
        c["own"] = OWN[c["side"]]
    if c.get("target") == "$dir":
        if not lr:
            return None
        c["target"] = lr
    if c.get("action") == "look" and c.get("target") in ("left", "right"):
        c["screen_side"] = "r" if c["target"] == "left" else "l"
    if c.get("name") == "$emo":
        emo = caps.get("emo") or R.emo_alias.get(caps.get("_pat", ""))
        if not emo:
            return None
        c["name"] = emo
    if c["action"] == "expression":
        c["intensity"] = _r3(min(1.0, max(0.2, c.get("intensity", 0.8) * k)))
    elif c["action"] == "gesture" and k != 1.0:
        c["intensity"] = _r3(min(1.5, max(0.3, k)))
    if mods.get("rep") and c["action"] == "gesture":
        c["repeat"] = mods["rep"]
    if mods.get("hold") is not None and c["action"] in ("pose", "gesture", "expression", "look"):
        c["hold"] = mods["hold"]
    if caps.get("_keep") and c["action"] in ("pose", "gesture", "expression", "look"):
        c["hold"] = True
    return c


def _clauses(R: Rules, toks: list[str]):
    """-> [(tokens, sentence_start, anchored)]"""
    return [(toks[a:b], st, an) for a, b, st, an in _clause_spans(R, toks)]


def _clause_spans(R: Rules, toks: list[str]):
    """-> [(start index, end index, sentence_start, anchored)]"""
    out, cur0, n, start, anch = [], 0, 0, True, False
    for k, t in enumerate(toks + ["|"]):
        if t == "|" or t in R.split or (t in R.anchors and n):
            if n:
                out.append((cur0, cur0 + n, start, anch))
            start = (t == "|") or (start and not n)
            if t == "|":
                start = True
            cur0, n, anch = k + 1, 0, False
            if t in R.anchors:
                cur0, n, anch = k, 1, True
            continue
        n += 1
    return out


def _strip_pre(R: Rules, toks: list[str]) -> int:
    i = 0
    while i < len(toks) and (toks[i] in R.pre or toks[i] in R.names):
        i += 1
    return i


def match_clause(R: Rules, toks: list[str]):
    """-> (rule, pattern, caps, mods) | ('pre',) for preamble-only | None"""
    i = _strip_pre(R, toks)
    rest = toks[i:]
    if not rest:
        return ("pre",)
    tries = [(rest, False)]
    if rest[0] in R.keep and len(rest) > 1:
        r2 = list(rest[1:])
        if r2 and r2[0] == "on":
            r2 = r2[1:]
        if r2:
            r2[0] = R.gerund.get(r2[0], r2[0])
            tries.append((apply_phrases(R, r2), True))
    elif rest[0] == "stop" and len(rest) > 1:
        pass
    for rt, keep in tries:
        for rule in R.rules:
            if rule["r"].get("whole"):
                continue
            for pat in rule["pats"]:
                ends = []
                _match(R, pat["elems"], rt, 0, 0, {}, ends)
                for end, caps in ends:
                    mods = _tail(R, rt[end:])
                    if mods is None:
                        continue
                    caps.update(pat["fixed"])
                    caps["_pat"] = pat["src"]
                    if keep:
                        caps["_keep"] = True
                    return (rule, pat, caps, mods, end == len(rt), rt[:end])
    return None


def _adjust(R: Rules, toks: list[str]):
    i = _strip_pre(R, toks)
    rest = toks[i:]
    if len(rest) == 0:
        return None
    if rest[0] in R.adjust and _tail(R, rest[1:]) is not None:
        return R.adjust[rest[0]]
    return None


class Matcher:
    def __init__(self, rules: Rules | None = None, act: bool = True, scene=None):
        self.R = rules or load_rules()
        self.last = None          # (rule, caps, mods, t_ms)
        # intent: act = whole-body moves + props (False = the plain matcher exactly); scene = config.props
        self.A = self.R.act if act else None
        self.scene_on = bool(act and scene)   # the hold / understood rule is for a character with a scene (the muse)
        if self.A is not None and scene:
            self.A = Act(self.R.d["act"])
            self.A.names = set(self.R.d["names"])
            self.A.add_scene(scene)

    def parse(self, text: str, final: bool = True, stable: bool = True, now_ms: float = 0.0) -> list[dict]:
        """-> [{"cmd", "clause", "rule", "open"}]  open = the clause ends the text (may still grow)."""
        R = self.R
        toks = tokenize(R, text)
        hits = []
        if final or stable:
            flat = [t for t in toks if t != "|"]
            for rule in R.rules:
                if not rule["r"].get("whole"):
                    continue
                for pat in rule["pats"]:
                    ends = []
                    _match(R, pat["elems"], flat, 0, 0, {}, ends)
                    if any(e == len(flat) for e, _ in ends):
                        c = build(R, rule["r"], {}, {"int": 1.0, "rep": None, "hold": None})
                        return [{"cmd": c, "clause": 0, "rule": rule["r"]["id"], "open": not final}]
        if self.A is not None:
            toks, raws = apply_phrases_spans(R, [R.alias.get(w, w) for w in normalize(text)])
            spans = _clause_spans(R, toks)
            cls = [(toks[a:b], st, an) for a, b, st, an in spans]
            craw = [[w for r in raws[a:b] for w in r] for a, b, st, an in spans]
        else:
            cls = _clauses(R, toks)
        prev_ok = neg_prev = False
        for idx, (ct, sstart, anch) in enumerate(cls):
            is_last = idx == len(cls) - 1
            eligible = sstart or anch or prev_ok or neg_prev
            prev_ok = neg_prev = False
            if not eligible:
                continue
            if self.A is not None:
                r = self._act_clause(craw[idx], hits, idx, is_last, final, stable)
                if r == "wait":
                    continue
                if r == "neg":                # a negated clause fires nothing itself (the legacy rules
                    neg_prev = True           # still see it: "never mind"), and the next clause still runs
                elif r:
                    prev_ok = True
                    continue
            adj = _adjust(R, ct)
            if adj is not None:
                if self.last and now_ms - self.last[3] <= CTX_MS:
                    rule, caps, mods, _ = self.last
                    m2 = dict(mods)
                    m2["int"] = min(1.6, max(0.3, mods["int"] * adj))
                    c = build(R, rule, caps, m2)
                    if c:
                        hits.append({"cmd": c, "clause": idx, "rule": rule["id"], "open": is_last and not final,
                                     "_ctx": (rule, caps, m2)})
                        prev_ok = True
                continue
            m = match_clause(R, ct)
            if m is None:
                continue
            if m[0] == "pre":
                prev_ok = True
                continue
            rule, pat, caps, mods = m[0], m[1], m[2], m[3]
            if rule["r"].get("final") and not final:
                continue
            if is_last and not final and not stable and (pat["wait"] or " ".join(m[5]) in rule["r"].get("wait", ())):
                continue
            c = build(R, rule["r"], caps, mods)
            if c is None:
                continue
            if self.A is not None and _act_attach(hits, c):            # intent: "... and stay like that"
                prev_ok = True
                continue
            hits.append({"cmd": c, "clause": idx, "rule": rule["r"]["id"], "open": is_last and not final,
                         "_ctx": (rule["r"], caps, mods)})
            prev_ok = True
        if self.A is not None:
            hits = _act_post(hits, normalize(text))
            for h in hits:
                h.pop("_raw", None)
        return hits

    def _act_clause(self, raw, hits, idx, is_last, final, stable):
        """intent: one clause through the act grammar -> True (hits appended) | 'wait' | False"""
        A = self.A
        prev0 = hits[-1]["cmd"] if hits else None
        if prev0 is not None and prev0.get("action") == "object" and _clean(A, raw) in HOLD_ON:
            prev0["hold"] = True                                       # "pick up the umbrella and don't let go"
            return True
        rel = _act_release(A, raw)
        if rel is not None:
            hits.append({"cmd": rel, "clause": idx * 8, "rule": "act:release", "open": is_last and not final})
            return True
        cmds = parse_act(A, raw, self.R.names, ctx=bool(prev0 and prev0.get("action") == "object"), ger0=bool(hits))
        neg = cmds == []                                                # a negated clause
        if not cmds:
            cmds = None
            prev = hits[-1]["cmd"] if hits else None
            if prev is not None and prev.get("action") == "object":    # "pick up the ball and the cup": verb carry
                cmds = _act_carry(A, raw, prev, self.R.names)
            if not cmds and prev is not None and prev.get("action") == "move" and _act_gaze_free(raw):
                prev["gaze"] = "free"                                  # "turn away and don't look at me"
                return True
        if not cmds:
            return "neg" if neg else False
        if is_last and not final and not stable:
            return "wait"           # a move/object may still grow ("spin" -> "spin around three times")
        for k, c in enumerate(cmds):
            key = c.get("verb") or c.get("name")
            hits.append({"cmd": c, "clause": idx * 8 + k, "rule": "act:" + c["action"] + "/" + str(key),
                         "open": is_last and not final, "_raw": raw if len(cmds) == 1 else None})
        return True

    def commit(self, hit: dict, now_ms: float) -> None:
        """Remember an emitted hit as the context for "more"/"higher"."""
        ctx = hit.get("_ctx")
        if ctx and ctx[0]["c"]["action"] in ("pose", "expression", "gesture") and not ctx[0].get("noint"):
            self.last = (ctx[0], ctx[1], ctx[2], now_ms)
        elif (hit.get("cmd") or {}).get("action") == "release":
            self.last = None          # a released pose is no longer what "more" refers to

    def forget_context(self) -> None:
        """the pose "more" / "higher" would adjust has been superseded (the voice model started another
        action, or the user said something else that was no command): a later "more" must not re-apply it (judge: 'lower
        your head' ... 'more' re-raised a head-up pose the model's lower-head call had replaced)."""
        self.last = None

    def understood(self, text: str, hits: list) -> bool:
        """every clause with content words produced a command (or is a negation / correction the act grammar handles)."""
        R = self.R
        toks, _raws = apply_phrases_spans(R, [R.alias.get(w, w) for w in normalize(text)])
        spans = _clause_spans(R, toks)
        if len(spans) <= 1:
            # one clause: a hit that used only its first words is the early-fire case ("flip the laptop lid open")
            words = [w for w in toks if w != "|" and w not in FILLER and len(w) > 1]
            if len(words) <= 6 or any(w in NEG for w in words):
                return True
            # a long one-clause request joined by and/then is understood when every part with content
            # words parses on its own (same rule as app/intent.js)
            said = [w for raw in _raws for w in raw if w != "|"]
            parts: list = [[]]
            for w in said:
                if w in ("and", "then"):
                    parts.append([])
                else:
                    parts[-1].append(w)
            real = [p for p in parts if [w for w in p if w not in FILLER and len(w) > 1]]
            return len(real) > 1 and all(self.parse(" ".join(p), final=True, stable=True) for p in real)
        for a, b, _st, _an in spans:
            words = [w for w in toks[a:b] if w != "|"]
            if any(w in NEG for w in words) or not [w for w in words if w not in FILLER and len(w) > 1]:
                continue
            # re-parse the words the user said, not the phrase tokens ('a bit' -> 'alittle' reads back as nothing)
            said = [w for raw in _raws[a:b] for w in raw if w != "|"]
            if not self.parse(" ".join(said), final=True, stable=True):   # this clause on its own means nothing to us
                return False
        return True

    def stream(self) -> "Stream":
        return Stream(self)


def canon(cmd: dict) -> str:
    return json.dumps(cmd, sort_keys=True, separators=(",", ":"))


class Stream:
    """Partial/final transcript stream for ONE recognizer: fires each command once per utterance."""

    def __init__(self, m: Matcher):
        self.m = m
        self.text = ""
        self.changed_ms = 0.0
        self.emitted: dict[int, str] = {}
        self.recent: list[tuple[float, str]] = []

    def update(self, text: str, final: bool, now_ms: float, quiet_ms: float | None = None) -> list[dict]:
        text = text.strip()
        if text != self.text:
            self.text = text
            self.changed_ms = now_ms
        stable = (now_ms - self.changed_ms) >= STABLE_MS
        act = getattr(self.m, 'scene_on', False)                          # the muse (act grammar + scene); the 11
        settled = final or not act or ((now_ms - self.changed_ms) >= HOLD_MS and (quiet_ms is None or quiet_ms >= QUIET_MS))
        out = []
        if text and settled:
            hits = self.m.parse(text, final=final, stable=stable if not act else True, now_ms=now_ms)
            if hits and act and not self.m.understood(text, hits):
                hits = []                                                   # half-understood: the voice model's turn
            for h in hits:
                cj = canon(h["cmd"])
                if self.emitted.get(h["clause"]) == cj:
                    continue
                if any(c == cj and now_ms - t < REPEAT_WINDOW_MS for t, c in self.recent) and h["clause"] not in self.emitted:
                    self.emitted[h["clause"]] = cj
                    continue
                prev = self.emitted.get(h["clause"])
                self.emitted[h["clause"]] = cj
                self.recent.append((now_ms, cj))
                self.m.commit(h, now_ms)
                h = dict(h)
                h.pop("_ctx", None)
                h["partial"] = not final
                if prev:
                    h["replaces"] = json.loads(prev)     # a refinement of this clause's earlier command
                out.append(h)
        self.recent = [(t, c) for t, c in self.recent if now_ms - t < REPEAT_WINDOW_MS]
        if final:
            self.text = ""
            self.emitted = {}
        return out


# ------------------------------------------------------------------------------------------- dedupe keys
PART_PREFIX = [("hand_l", "hand_l"), ("hand_r", "hand_r"), ("point_l", "hand_l"), ("point_r", "hand_r"),
               ("thumb_l", "hand_l"), ("thumb_r", "hand_r"), ("paw_open_l", "hand_l"), ("paw_open_r", "hand_r"),
               ("arm_l", "hand_l"), ("arm_r", "hand_r"), ("wave_l", "hand_l"), ("wave_r", "hand_r"),
               ("head_", "head"), ("gaze_", "gaze"), ("blink_", "eyes"), ("eye_wide", "eyes"), ("squint", "eyes"),
               ("brow", "brows"), ("ear_l", "ear_l"), ("ear_r", "ear_r"), ("ear_flick_l", "ear_l"),
               ("ear_flick_r", "ear_r"), ("nose", "nose"), ("nostril", "nose"), ("mouth", "mouth"), ("jaw", "mouth"),
               ("tongue", "tongue"), ("cheek", "cheeks"), ("shoulder", "shoulders"), ("tail", "tail"),
               ("body", "body"), ("smile", "expression"), ("clap", "hand_l"), ("thumbs_up", "hand_l")]


def part_of(param: str) -> str:
    for pre, part in PART_PREFIX:
        if param.startswith(pre):
            return part
    return param


def cmd_keys(cmd: dict, R: Rules | None = None) -> list[str]:
    """Dedupe keys = the body parts a command drives (shared with IntentDedupe in intent.js)."""
    R = R or load_rules()
    a = cmd.get("action")
    if a == "release":
        ps = cmd.get("parts")
        return ["release:all"] if ps == "all" or not isinstance(ps, list) else sorted("release:" + p for p in ps)
    if a == "hold":
        return ["hold"]
    if a == "look":
        return ["gaze"]
    if a == "expression":
        return ["expression"]
    if a == "move":                                   # intent: whole-body moves / props by action family
        return ["move:" + MOVE_FAM.get(cmd.get("name"), str(cmd.get("name")))]
    if a == "object":
        f = OBJ_FAM.get(cmd.get("verb"), str(cmd.get("verb")))
        return ["move:posture", "object:seat"] if f == "seat" else ["object:" + f]
    if a in ("gesture", "pose") and cmd.get("name") in LEGACY_MOVE:
        return ["move:" + MOVE_FAM[LEGACY_MOVE[cmd["name"]]]]
    parts = cmd.get("parts")
    if not parts:
        ps = set()
        for k in (cmd.get("targets") or {}):
            ps.add(part_of(k))
        if not ps and cmd.get("name") in R.name_parts:
            side = cmd.get("side")
            sides = ["l", "r"] if side in ("both", None) else [side]
            for p in R.name_parts[cmd["name"]]:
                for pp in ([p.replace("$", s) for s in sides] if "$" in p else [p]):
                    ps.add(pp)
        parts = sorted(ps) if ps else [str(cmd.get("name") or a)]
    return sorted(set(p.replace("_open", "") for p in parts))


# ============================================================================== intent: whole-body moves + props
# The "act" grammar (intent_rules.json "act", source tools/intent_bench/act_lexicon.py) parses a clause's RAW tokens
# (aliases applied, legacy phrase table not) into {action:'move', name, count, dir, speed, distance, height, style,
# gaze, hold, duration_ms} / {action:'object', verb, object:{ref,name,category,color,size,ordinal,pos,where}, side,
# place, near, target, amount, count, hold, ...} (docs/INTENT-RULES.md section 3 + 0). Tried before the legacy rules.
# Precision first: a clause fires only when it STARTS with a known verb (after a preamble / request frame) and every
# token after the verb is consumed by the object phrase and known modifiers. app/intent.js parseAct is the twin.
MOVE_FAM = {"spin": "spin", "twirl": "spin", "turn": "turn", "turn_away": "turn", "turn_back": "turn",
            "backflip": "flip", "frontflip": "flip", "flip": "flip", "cartwheel": "flip", "jump": "jump", "hop": "jump",
            "bounce": "jump", "walk": "walk", "step": "walk", "sit": "posture", "sit_down": "posture",
            "stand": "posture", "stand_up": "posture", "get_up": "posture", "lie_down": "posture",
            "roll_over": "posture", "dance": "dance", "bow": "bow", "wiggle": "wiggle", "shake_off": "wiggle"}
OBJ_FAM = {"pick_up": "pick_up", "put_down": "put", "put_back": "put", "place": "put", "drop": "put",
           "hold_up": "show", "show": "show", "offer": "show", "inspect": "inspect", "turn": "turn", "flip": "turn",
           "hand_over": "hand_over", "toss": "toss", "catch": "toss", "shake": "shake", "hug": "hug", "smell": "smell",
           "put_on": "wear", "take_off": "wear", "pull": "furniture", "push": "furniture", "bring": "furniture",
           "slide": "furniture", "sit_in": "seat", "sit_at": "seat", "stand_up": "seat", "open": "open",
           "close": "open", "use": "work", "work_on": "work", "type_on": "work", "stop": "work"}
LEGACY_MOVE = {"spin": "spin", "jump": "jump", "dance": "dance", "bow": "bow", "turn_around": "turn_away",
               "turn": "turn"}
JUMPY = ("jump", "hop", "bounce", "backflip", "frontflip", "flip")
HOMO = ("to", "too", "for", "won", "tree")
DESC_KEYS = ("name", "category", "color", "size", "ordinal", "pos")


def _act_elems(pat: str):
    out = []
    for tok in pat.split(" "):
        opt = tok.startswith("?")
        out.append((opt, (tok[1:] if opt else tok).split("/")))
    return out


def _act_firsts(elems, j=0):
    """-> (set of literal first tokens, wild?)"""
    if j >= len(elems):
        return set(), True
    opt, alts = elems[j]
    lit, wild = set(), False
    for a in alts:
        if a.startswith("<") or a.startswith("@"):
            wild = True
        else:
            lit.add(a)
    if opt:
        l2, w2 = _act_firsts(elems, j + 1)
        lit |= l2
        wild = wild or w2
    return lit, wild


class Act:
    def __init__(self, a: dict):
        self.a = a
        self.cls = a["cls"]
        self.madj = a["madj"]
        self.nums = a["nums"]
        self.qty = sorted(([k.split(" "), v] for k, v in a["qty"].items()), key=lambda e: -len(e[0]))
        self.colors, self.sizes, self.ords = dict(a["colors"]), a["sizes"], a["ords"]
        self.shades = set(a["shades"])
        self.nouns = dict(a["nouns"])
        self.multi = sorted(([k.split(" "), v] for k, v in a["nouns_multi"].items()), key=lambda e: -len(e[0]))
        self.generic, self.anyn, self.places = a["generic"], a["anyn"], dict(a["places"])
        self.pre, self.fillers, self.gerund, self.neg = set(a["pre"]), set(a["fillers"]), a["gerund"], set(a["neg"])
        self.style_stop = set(a["style_stop"])
        self.frames = sorted(([f.split(" "), k] for f, k in a["frames"]), key=lambda e: -len(e[0]))
        self.pats = []
        for e in a["moves"]:
            self.pats.append({"kind": "m", "src": e[0], "elems": _act_elems(e[0]), "name": e[1],
                              "set": e[2] if len(e) > 2 else {}})
        for e in a["objs"]:
            self.pats.append({"kind": "o", "src": e[0], "elems": _act_elems(e[0]), "verb": e[1], "flags": e[2],
                              "set": e[3] if len(e) > 3 else {}})
        self.names = set()
        self.mods = []
        for e in a["mods"]:
            self.mods.append({"src": e[0], "elems": _act_elems(e[0]), "kind": e[1], "set": e[2],
                              "only": e[3] if len(e) > 3 else None})
        self.pat_idx, self.pat_wild = self._index(self.pats)
        self.mod_idx, self.mod_wild = self._index(self.mods)
        self.verb_first = set(self.pat_idx)

    @staticmethod
    def _index(items):
        idx, wild = {}, []
        for k, it in enumerate(items):
            lit, w = _act_firsts(it["elems"])
            if w:
                wild.append(k)
            for t in lit:
                idx.setdefault(t, []).append(k)
        return idx, wild

    def cands(self, idx, wild, tok):
        ks = list(idx.get(tok, ()))
        if wild:
            ks = sorted(set(ks) | set(wild))
        return ks

    def add_scene(self, cfg) -> None:
        """Teach the matcher a scene's prop nouns/colours/places (config.props shape, or a list of them)."""
        for c in (cfg if isinstance(cfg, list) else [cfg]):
            if not isinstance(c, dict):
                continue
            for p in c.get("props") or []:
                words = list(p.get("aliases") or []) + [str(p.get("id") or "").replace("_", " "), str(p.get("name") or "")]
                for w in words:
                    w = " ".join(normalize(w)).replace("|", "").strip()
                    if not w:
                        continue
                    ws = w.split(" ")
                    while len(ws) > 1 and (ws[0] in self.colors or ws[0] in self.sizes or ws[0] in self.shades
                                           or ws[0] in ("the", "a", "an")):
                        ws = ws[1:]
                    if len(ws) == 1:
                        if ws[0] not in self.colors and ws[0] not in self.generic and ws[0] not in self.anyn:
                            self.nouns.setdefault(ws[0], ws[0])
                    elif len(ws) <= 3 and not any(m[0] == ws for m in self.multi):
                        self.multi.append([ws, "_".join(ws)])
                for col in [p.get("color")] + list(p.get("colors") or []):
                    if isinstance(col, str) and col.isalpha():
                        self.colors.setdefault(col.lower(), col.lower())
            for pl in c.get("places") or []:
                for w in list(pl.get("aliases") or []) + [str(pl.get("id") or "")]:
                    ws = " ".join(normalize(w.replace("_", " "))).split(" ")
                    if len(ws) == 1 and ws[0]:
                        self.places.setdefault(ws[0], ws[0])
        self.multi.sort(key=lambda e: -len(e[0]))


def _seq_at(toks, i, seq) -> bool:
    return toks[i:i + len(seq)] == seq


def _num_spans(A: Act, toks, i):
    out = []
    if i >= len(toks):
        return out
    for seq, v in A.qty:
        if _seq_at(toks, i, seq):
            out.append((i + len(seq), v))
    t = toks[i]
    if t.isdigit() and 0 < int(t) <= 100:
        out.append((i + 1, int(t)))
    elif t in A.nums:
        if t not in HOMO or (i + 1 < len(toks) and toks[i + 1] in ("times", "time")):
            out.append((i + 1, A.nums[t]))
    return out


def _paw_at(A: Act, toks, i) -> int:
    return 1 if i < len(toks) and toks[i] in A.cls["paw"] else 0


def _dest_spans(A: Act, toks, i):
    """'to your other paw' / 'into your left hand' -> side"""
    out = []
    n = len(toks)
    j = i
    if j < n and toks[j] == "over":
        j += 1
    if j < n and toks[j] == "across":
        j += 1
    if j >= n or toks[j] not in ("to", "into", "in", "onto"):
        return out
    j += 1
    if j < n and toks[j] in ("your", "the", "my", "ur"):
        j += 1
    if j < n and toks[j] in ("other", "left", "right"):
        side = {"other": "other", "left": "left", "right": "right"}[toks[j]]
        j += 1
        if _paw_at(A, toks, j):
            out.append((j + 1, side))
        out.append((j, side))
    return out


def _place_spans(A: Act, toks, i):
    out = []
    n = len(toks)
    for pre in (["on", "top", "of"], ["on", "to"], ["on"], ["onto"], ["upon"], ["in"], ["into"], ["inside"],
                ["at"], ["to"], ["over", "on"], ["over", "to"], ["under"], ["underneath"], ["off"]):
        if not _seq_at(toks, i, pre):
            continue
        j = i + len(pre)
        if j < n and toks[j] in ("the", "your", "my", "that", "this", "a", "ur"):
            j += 1
        while j < n and (toks[j] in A.shades or toks[j] in A.colors or toks[j] in A.sizes):
            j += 1
        if j + 1 < n and toks[j] + " " + toks[j + 1] in ("side table", "desk top", "coffee table", "toy box"):
            out.append((j + 2, {"side table": "side_table", "desk top": "desk", "coffee table": "table",
                                "toy box": "box"}[toks[j] + " " + toks[j + 1]]))
        if j < n and toks[j] in A.places:
            out.append((j + 1, A.places[toks[j]]))
        if pre == ["in"] and j < n and toks[j] == "front":     # 'in front of' is not a place
            return []
    return out


PRON = [(["whatever", "youre", "holding"], {"ref": "it"}), (["whatever", "you", "are", "holding"], {"ref": "it"}),
        (["what", "youre", "holding"], {"ref": "it"}), (["what", "you", "are", "holding"], {"ref": "it"}),
        (["whatever", "your", "holding"], {"ref": "it"}), (["what", "your", "holding"], {"ref": "it"}),
        (["whatever", "you", "have"], {"ref": "it"}), (["whatever", "youve", "got"], {"ref": "it"}),
        (["the", "other", "ones"], {"ref": "other"}), (["the", "other", "one"], {"ref": "other"}),
        (["both", "of", "them"], {"ref": "all"}), (["all", "of", "them"], {"ref": "all"}),
        (["one", "of", "them"], {"ref": "any"}), (["all", "of", "it"], {"ref": "all"}),
        (["another", "one"], {"ref": "other"}), (["other", "one"], {"ref": "other"}), (["the", "other"], {"ref": "other"}),
        (["that", "one"], {"ref": "that", "_sel": 1}), (["this", "one"], {"ref": "that", "_sel": 1}), (["them", "all"], {"ref": "all"}),
        (["either", "one"], {"ref": "any"}), (["any", "one"], {"ref": "any"}),
        (["it"], {"ref": "it"}), (["that"], {"ref": "that"}), (["this"], {"ref": "that"}), (["them"], {"ref": "all"}),
        (["those"], {"ref": "all"}), (["these"], {"ref": "all"}), (["em"], {"ref": "all"}), (["both"], {"ref": "all"}),
        (["everything"], {"ref": "all"}), (["all"], {"ref": "all"}), (["one"], {"ref": "any"}),
        (["something"], {"ref": "any"}), (["anything"], {"ref": "any"}), (["either"], {"ref": "any"})]
DETS = [(["one", "of", "the"], "oneof"), (["one", "of", "your"], "oneof"), (["one", "of", "my"], "oneof"),
        (["one", "of", "those"], "oneof"), (["all", "of", "the"], "all"), (["all", "the"], "all"),
        (["both", "of", "the"], "all"), (["both", "the"], "all"), (["both", "your"], "all"), (["all", "your"], "all"),
        (["the"], "the"), (["a"], "a"), (["an"], "a"), (["your"], "the"), (["my"], "the"), (["ur"], "the"),
        (["that"], "the"), (["this"], "the"), (["these"], "the"), (["those"], "the"), (["some"], "a"),
        (["any"], "a"), (["another"], "another"), (["both"], "all"), (["his"], "the"), (["her"], "the"),
        (["our"], "the"), (["their"], "the")]
NP_PP = [(["on", "your", "left"], {"pos": "left"}), (["on", "your", "right"], {"pos": "right"}),
         (["to", "your", "left"], {"pos": "left"}), (["to", "your", "right"], {"pos": "right"}),
         (["on", "my", "left"], {"pos": "screen_left"}), (["on", "my", "right"], {"pos": "screen_right"}),
         (["on", "the", "left"], {"pos": "screen_left"}), (["on", "the", "right"], {"pos": "screen_right"}),
         (["in", "the", "middle"], {"pos": "middle"}), (["in", "the", "center"], {"pos": "middle"}),
         (["in", "the", "centre"], {"pos": "middle"})]


def _np_head(A: Act, toks, j):
    """-> [(end, {fields})] for a head noun at j"""
    out = []
    for seq, v in A.multi:
        if _seq_at(toks, j, seq):
            out.append((j + len(seq), v))
    if j < len(toks):
        t = toks[j]
        if t in A.nouns:
            out.append((j + 1, A.nouns[t]))
        if t in A.generic:
            out.append((j + 1, "@" + A.generic[t] + ("+" if t.endswith("s") and t != "stuff" else "")))
        if t in A.anyn:
            out.append((j + 1, "*" + ("+" if A.anyn[t] else "")))
    return out


def _np_spans(A: Act, toks, i):
    """All object noun phrases starting at i -> [(end, descriptor)], longest first."""
    n = len(toks)
    out = []
    if i >= n:
        return out
    for seq, d in PRON:
        if _seq_at(toks, i, seq):
            out.append((i + len(seq), dict(d)))
    dets = [(i, None)]
    for seq, dv in DETS:
        if _seq_at(toks, i, seq):
            dets.append((i + len(seq), dv))
            break
    for j0, det in dets:
        _np_adj(A, toks, j0, det, {}, 0, out)
    # keep the first descriptor per end, longest first (stable)
    seen, res = set(), []
    for e, d in sorted(out, key=lambda x: -x[0]):
        if e not in seen:
            seen.add(e)
            res.append((e, d))
    return res


def _np_adj(A: Act, toks, j, det, d, unk, out) -> None:
    n = len(toks)
    for e, head in _np_head(A, toks, j):
        _np_finish(A, toks, e, det, d, head, out)
    if j >= n or j - (0 if det is None else 0) > 40:
        return
    t = toks[j]
    nd = None
    if t in A.colors:
        nd = dict(d); nd.setdefault("color", A.colors[t])
    elif t in A.sizes:
        nd = dict(d); nd.setdefault("size", A.sizes[t])
    elif t in A.ords:
        nd = dict(d); nd.setdefault("ordinal", A.ords[t])
    elif t == "other":
        nd = dict(d); nd["ref"] = "other"
    elif t in ("left", "right") and det is not None:
        nd = dict(d); nd.setdefault("pos", "screen_" + t)
    elif t in ("middle", "center", "centre") and det is not None:
        nd = dict(d); nd.setdefault("pos", "middle")
    elif t in A.shades or t == "and":
        nd = dict(d)
    if nd is not None and len(nd) < 6:
        _np_adj(A, toks, j + 1, det, nd, unk, out)
    elif nd is None and det is not None and unk < 2 and t.isalpha() and t not in A.neg and t not in A.style_stop \
            and t not in A.pre and t not in A.cls["paw"]:
        _np_adj(A, toks, j + 1, det, dict(d, _unk=1), unk + 1, out)


def _np_finish(A: Act, toks, e, det, d, head, out) -> None:
    d = dict(d)
    plural = head.endswith("+")
    head = head.rstrip("+")
    if head.startswith("@"):
        d["category"] = head[1:]
    elif head == "*":
        if d.get("_unk"):
            return                     # "the zigzag one": an unknown word needs a real noun after it
    else:
        d["name"] = head
    d.pop("_unk", None)
    if det == "another":
        d["ref"] = "other"
    if plural and det != "oneof" and "ordinal" not in d:
        d["ref"] = d.get("ref") if d.get("ref") == "other" else "all"
    if det == "all":
        d["ref"] = "all"
    base = dict(d)
    if head == "*" and not any(k in d for k in DESC_KEYS) and "ref" not in d:
        if det in ("a", "oneof"):
            d["ref"] = "any"
        elif det == "the":
            d["ref"] = "that"
            d["_one"] = 1                  # "the one ..." (dropped when a side says which: "the one in your left paw")
        else:
            return
    out.append((e, d))
    for seq, pp in NP_PP:
        if _seq_at(toks, e, seq):
            out.append((e + len(seq), dict(base, **pp)))
    for e2, pl in _place_spans(A, toks, e):
        if toks[e] in ("on", "from", "in", "off") and pl != "head":
            out.append((e2, dict(base if head == "*" else d, where=pl)))
    if e + 1 < len(toks) and toks[e] == "from" and toks[e + 1] in ("the", "your"):
        for e2, pl in _place_spans(A, ["on"] + toks[e + 1:], 0):
            out.append((e + e2 - 0, dict(base if head == "*" else d, where=pl)))


def _act_match(A: Act, elems, toks, i, j, caps, out) -> None:
    if j == len(elems):
        out.append((i, caps))
        return
    opt, alts = elems[j]
    n = len(toks)
    for alt in alts:
        if alt == "<O>":
            for e, d in _np_spans(A, toks, i):
                _act_match(A, elems, toks, e, j + 1, dict(caps, O=d), out)
        elif alt == "<N>":
            for e, v in _num_spans(A, toks, i):
                _act_match(A, elems, toks, e, j + 1, dict(caps, N=v), out)
        elif alt == "<S>":
            for e, v in _dest_spans(A, toks, i):
                _act_match(A, elems, toks, e, j + 1, dict(caps, S=v), out)
        elif alt == "<P>":
            for e, v in _place_spans(A, toks, i):
                _act_match(A, elems, toks, e, j + 1, dict(caps, P=v), out)
        elif alt.startswith("@"):
            cls = alt[1:]
            tab = A.cls.get(cls) or {}
            if cls == "style":          # any one word (dance like a <word>)
                if i < n and toks[i].isalpha() and toks[i] not in A.style_stop and toks[i] not in A.neg:
                    _act_match(A, elems, toks, i + 1, j + 1, dict(caps, **{"@" + cls: caps.get("@" + cls, []) + [toks[i]]}), out)
                continue
            for L in (3, 2, 1):
                if i + L <= n:
                    key = " ".join(toks[i:i + L])
                    if key in tab:
                        _act_match(A, elems, toks, i + L, j + 1, dict(caps, **{"@" + cls: caps.get("@" + cls, []) + [key]}), out)
        elif i < n and toks[i] == alt:
            _act_match(A, elems, toks, i + 1, j + 1, caps, out)
    if opt:
        _act_match(A, elems, toks, i, j + 1, caps, out)


def _mods(A: Act, toks, i, kind, name, depth=0):
    """Tail modifiers covering toks[i:] completely -> [(mod, caps), ...] or None."""
    if i >= len(toks):
        return []
    if depth > 12:
        return None
    if toks[i] in A.names:                                     # "... for me, Bruno"
        return _mods(A, toks, i + 1, kind, name, depth + 1)
    for k in A.cands(A.mod_idx, A.mod_wild, toks[i]):
        m = A.mods[k]
        if kind not in m["kind"] or (m["only"] is not None and name not in m["only"]):
            continue
        ends = []
        _act_match(A, m["elems"], toks, i, 0, {}, ends)
        for e, caps in ends:
            if e <= i:
                continue
            rest = _mods(A, toks, e, kind, name, depth + 1)
            if rest is not None:
                return [(m, caps)] + rest
    return None


def _val(A: Act, v, caps):
    """a pattern/modifier value: '$N' number, '$Ns'/'$Nm' seconds/minutes -> ms, '$O'/'$P'/'$S' captures,
    '$slr' screen side of the captured @lr, '$<cls>' the captured class word's value; anything else literal"""
    if not (isinstance(v, str) and v.startswith("$")):
        return v
    k = v[1:]
    n = caps.get("N")
    if k == "N":
        return n
    if k == "Ns":
        return None if n is None else n * 1000
    if k == "Nm":
        return None if n is None else n * 60000
    if k in ("O", "P", "S"):
        return caps.get(k)
    if k == "slr":
        lr = (caps.get("@lr") or [None])[0]
        return None if lr is None else "screen_" + _cls_val(A, "lr", lr)
    lst = caps.get("@" + k)
    return _cls_val(A, k, lst[0]) if lst else None


def _cls_val(A: Act, cls, key):
    tab = A.cls.get(cls) or {}
    return tab.get(key, key)


def _apply(A: Act, cmd, sets, caps, x=None) -> None:
    for k, v in sets.items():
        v = _val(A, v, caps)
        if v is None:
            continue
        if x is not None and k.startswith("_"):
            x[k] = v
        else:
            cmd[k] = v


def _scrub(d):
    return {k: (_scrub(v) if isinstance(v, dict) else v) for k, v in d.items() if not k.startswith("_")}


def _desc_ok(d) -> bool:
    return isinstance(d, dict) and any(k in d for k in DESC_KEYS)


def _build_act(A: Act, pat, caps, mods, st):
    """-> command dict or None (invalid)"""
    if pat["kind"] == "m":
        name = pat["name"]
        if name == "$mn":
            name = _cls_val(A, "mn", (caps.get("@mn") or ["spin"])[0])
        cmd = {"action": "move", "name": name}
        _apply(A, cmd, pat["set"], caps)
        for a in (caps.get("@madj") or []) + (caps.get("@madjb") or []):
            for k, v in A.madj.get(a, {}).items():
                if k != "_k":
                    cmd[k] = v
        for m, mc in mods:
            _apply(A, cmd, m["set"], mc)
        if "_flip" in cmd:
            if cmd["name"] in ("flip", "backflip", "frontflip"):         # "a backwards somersault"
                cmd["name"] = cmd["_flip"]
            del cmd["_flip"]
        if cmd["name"] == "flip":                                        # plain "flip" = backflip
            cmd["name"] = "frontflip" if cmd.get("dir") == "forward" else "backflip"
            cmd.pop("dir", None) if cmd.get("dir") in ("forward", "back") else None
        if cmd["name"] == "turn_away" and "count" in cmd:
            cmd["name"] = "spin"
        if cmd["name"] == "turn" and "dir" not in cmd:
            return None
        if "height" in cmd and cmd["name"] not in JUMPY:
            del cmd["height"]
        if "style" in cmd and "style" not in pat["set"] and cmd["name"] not in ("dance", "walk", "bow", "lie_down") \
                and not any("style" in m["set"] for m, _ in mods):
            del cmd["style"]
        if "count" in cmd:
            cmd["count"] = max(1, min(20, int(cmd["count"])))
        if "distance" in cmd:
            cmd["distance"] = max(0, min(20, int(cmd["distance"])))
    else:
        verb = pat["verb"]
        cmd = {"action": "object", "verb": verb}
        flags = pat["flags"]
        if "O" in caps:
            cmd["object"] = dict(caps["O"])
        if "S" in caps:
            cmd["side"] = caps["S"]
        x = {}
        _apply(A, cmd, pat["set"], caps, x)
        for m, mc in mods:
            _apply(A, cmd, m["set"], mc, x)
        nonfill = any(m["set"] for m, _ in mods)
        o = cmd.get("object")
        if o is not None and o.pop("_one", None) and "side" in cmd and o == {"ref": "that"}:
            del cmd["object"]                        # "drop the one in your right paw" -> side=right
            o = None
        if "N" in flags and not _desc_ok(o) and not ("side" in cmd and o is None):
            return None
        sel = bool(o and (o.pop("_sel", None) or o.get("ref") == "other"))
        if "n" in flags and not _desc_ok(o) and not nonfill and not sel and not (st.get("ctx") and o):
            return None      # "take it" alone is not a request; "take this one", "... and use it", "take it with your left paw" are
        if x.get("_ref") and "object" not in cmd:
            cmd["object"] = {"ref": x["_ref"]}
        v = cmd["verb"]
        if v == "place":
            if x.get("_pb"):
                v = "put_back"
            elif "place" in cmd:
                v = "place"
            elif x.get("_down"):
                v = "put_down"
            elif x.get("_hi"):
                v = "hold_up"
            else:
                return None
        elif v == "put_down" and "place" in cmd:
            v = "place"
        elif x.get("_pb") and v in ("pull", "push", "bring", "slide"):
            cmd["target"] = "home"   # "push the chair back where it was" = its own spot, not 'back from the desk'
        if v in ("pick_up", "hold_up") and x.get("_hi"):
            v = "hold_up"
        if v in ("pick_up", "hold_up") and x.get("_cam"):
            v = "show"
        if v == "turn" and x.get("_flip"):
            v = "flip"
            cmd.pop("amount", None)
        if v == "slide" and isinstance(cmd.get("target"), dict) and list(cmd["target"]) == ["name"] \
                and cmd["target"]["name"] in A.places and pat["src"].startswith("move"):
            cmd["place"] = A.places[cmd.pop("target")["name"]]       # "move the lantern to the table"
            v = "place"
        if x.get("_needtarget") and "target" not in cmd and "place" not in cmd:
            if v == "slide" and "place" in cmd:
                v = "place"
            else:
                return None
        if cmd.get("place") == "head" and v in ("place", "put_down", "put_on", "put_back"):
            v = "put_on"
            del cmd["place"]
        if v in ("pull", "push", "bring", "slide") and "target" in cmd and "place" in cmd:
            del cmd["place"]                         # "slide the laptop closer on the desk"
        if v not in ("place", "put_back", "drop", "put_down") and "place" in cmd and cmd["place"] != "near":
            if v == "slide" and "target" not in cmd:
                v = "place"
            else:
                return None
        if "near" in cmd and v not in ("place", "put_back", "drop", "put_down"):
            return None
        if "place" in cmd and cmd["place"] == "near" and v == "put_down":
            v = "place"
        if v == "put_on" and cmd.get("place") == "head":
            del cmd["place"]
        cmd["verb"] = v
        if "count" in cmd:
            cmd["count"] = max(1, min(20, int(cmd["count"])))
        cmd.pop("speed", None)                      # objects: no speed / duration (conventions; motion paces them)
        cmd.pop("duration_ms", None)
    if st.get("speed") and "speed" not in cmd and cmd["action"] == "move":
        cmd["speed"] = st["speed"]
    if st.get("keep"):
        cmd["hold"] = True
    if st.get("side") and cmd["action"] == "object" and "side" not in cmd:
        cmd["side"] = st["side"]
    return _scrub(cmd)


PLACE_VERBS = ("place", "put_down", "put_back", "drop", "slide", "pull", "push", "bring")


def _act_at(A: Act, toks, i, st):
    """Try every verb pattern at position i -> command or None."""
    if i >= len(toks):
        return None
    for k in A.cands(A.pat_idx, A.pat_wild, toks[i]):
        pat = A.pats[k]
        ends = []
        _act_match(A, pat["elems"], toks, i, 0, {}, ends)
        for e, caps in ends:
            if e <= i:
                continue
            if pat.get("verb") in PLACE_VERBS and "where" in (caps.get("O") or {}):
                continue            # "drop the ball on the floor": a place, not which ball
            key = pat.get("verb") or pat.get("name")
            if key == "$mn":
                key = _cls_val(A, "mn", (caps.get("@mn") or ["spin"])[0])
            mods = _mods(A, toks, e, pat["kind"], key)
            if mods is None:
                continue
            cmd = _build_act(A, pat, caps, mods, st)
            if cmd is not None:
                return cmd
    return None


LIKE_PREV = ("you", "u", "ya", "so", "can", "could", "would", "and", "then", "just", "okay", "ok", "um", "uh")


def _clean(A: Act, toks):
    out = []
    for k, t in enumerate(toks):
        if t == "like":                    # "can you like spin", "like two steps" (kept: "dance like a robot")
            nx = toks[k + 1] if k + 1 < len(toks) else ""
            pv = out[-1] if out else ""
            if not out or pv in A.pre or pv in A.names or pv in LIKE_PREV or nx.isdigit() or nx in A.nums \
                    or (nx == "a" and k + 2 < len(toks) and toks[k + 2] in ("couple", "few", "bunch")):
                continue
            out.append(t)
            continue
        if t in A.fillers:
            continue
        if out and out[-1] == t and t in ("the", "a", "your", "it", "my", "to", "and"):
            continue
        out.append(t)
    return out


RUNON_NOT = ("up", "down", "back", "over", "it", "the", "a", "to", "on", "in", "with", "your", "me", "around", "off",
             "out", "away", "here", "one", "that", "this", "them", "for", "at", "of", "my", "an", "another", "some",
             "both", "all", "everything", "something", "please", "now", "again", "really", "very", "super", "so")


def _head(c):
    return c.get("action", "") + "/" + str(c.get("name") or c.get("verb") or "")


NEG_START = (["do", "not"], ["dont"], ["no"], ["never"], ["not"], ["skip"], ["without"], ["hold", "off", "on"],
             ["hold", "off"], ["forget", "about"], ["instead", "of"])
NEG_RESUME = ("just", "but", "instead", "rather", "only")
NEG_RESUME2 = (["how", "about"], ["what", "about"], ["can", "you"], ["could", "you"], ["lets"])   # resume AT these frames
NEG_MID = ("dont", "not", "never")
ORDER_WORDS = {"after": -1, "once": -1, "before": 1}
ORDER_SKIP = ("you", "u", "youve", "youre", "ya", "we", "i")


def _neg_at(toks, i):
    for seq in NEG_START:
        if toks[i:i + len(seq)] == seq:
            return len(seq)
    return 0


def parse_act(A: Act, toks, names=(), depth=0, ctx=False, ger0=False):
    """A clause's raw tokens -> [command, ...] (1-3 for run-ons without 'and'), [] for a negated clause ("don't sit":
    nothing, but the utterance's other clauses still run), or None.
    ctx = an object command came just before in the same utterance ("... and look at it"); ger0 = a gerund may start
    the clause ("show me you spinning twice and flipping backwards")."""
    toks = _clean(A, toks)
    st = {"ctx": True} if ctx else {}
    if ger0:
        st["ger"] = True
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if st.get("ger") and t in A.gerund:
            toks = toks[:i] + [A.gerund[t]] + toks[i + 1:]
            st["ger"] = False
        cmd = _act_at(A, toks, i, st)
        if cmd is not None:
            return [cmd]
        # a negated clause emits nothing; "don't sit, just pull the chair over" runs the rest
        ng = _neg_at(toks, i)
        if ng:
            for k in range(i + ng, n):
                if toks[k] in NEG_RESUME or any(_seq_at(toks, k, q) for q in NEG_RESUME2):
                    k1 = k + 1 if toks[k] in NEG_RESUME else k
                    r = parse_act(A, toks[k1:], names, depth + 1, ctx, True) if depth < 3 else None
                    return r or []
            return []
        if depth < 3:
            # "put it down, not on the desk": a complete command, then a negated tail
            for k in range(i + 1, n):
                if toks[k] in NEG_MID or (toks[k] == "no" and k > i + 1):
                    kc = k - 1 if toks[k - 1] in ("but", "and") else k
                    left = _act_at(A, toks[:kc], i, st)
                    if left is not None:
                        out = [left]
                        for j in range(k + 1, n):
                            if toks[j] in NEG_RESUME:
                                out += parse_act(A, toks[j + 1:], names, depth + 1, True, True) or []
                                break
                        return out
                    break
            # "open the laptop after you sit down" / "sit down before you open it" (spoken order != done order)
            for k in range(i, n):
                if toks[k] not in ORDER_WORDS:
                    continue
                j = k + 1
                while j < n and toks[j] in ORDER_SKIP:
                    j += 1
                right = parse_act(A, toks[j:], names, depth + 1, ctx, True) if j < n else None
                if not right:
                    continue
                if k == i:                                   # "after you sit down (then) open it"
                    return right
                left = _act_at(A, toks[:k], i, st)
                if left is None:
                    continue
                return right + [left] if ORDER_WORDS[toks[k]] < 0 else [left] + right
        # run-on: "pick up the ball throw it up" (no 'and' from the recognizer)
        if depth < 3:
            for k in range(n - 1, i, -1):
                cut = toks[k] in ("and", "then", "andthen", "than")
                if not cut and (toks[k] not in A.verb_first or toks[k] in RUNON_NOT):
                    continue
                left = _act_at(A, toks[:k], i, st)
                if left is None:
                    continue
                right = parse_act(A, toks[k + 1:] if cut else toks[k:], names, depth + 1, ctx=True)
                if right and left.get("object") == {"ref": "any"} and right[0].get("action") == "move":
                    continue                          # "show me one jump" is a jump, not "show me one" + a jump
                if right and _head(right[0]) != _head(left):
                    return [left] + right
            # a topicalised object: "the laptop, open it" / "now the laptop open" / "laptop pick up"
            for e, d in _np_spans(A, toks, i):
                if not _desc_ok(d) or e >= n or "where" in d:
                    continue
                rest = toks[e:]
                while rest and (rest[0] in A.pre or rest[0] in names):
                    rest = rest[1:]
                for tail in (rest, rest + ["it"]):
                    r = parse_act(A, tail, names, depth + 1, True) if tail else None
                    if r and len(r) == 1 and r[0].get("action") == "object" and \
                            (r[0].get("object") is None or r[0]["object"] in ({"ref": "it"}, {"ref": "that"})):
                        c = dict(r[0])
                        c["object"] = _scrub(d)
                        return [c]
                break
        if t in A.pre or t in names:
            i += 1
            continue
        hit = None
        for seq, kind in A.frames:
            if _seq_at(toks, i, seq):
                hit = (seq, kind)
                break
        if hit:
            seq, kind = hit
            i += len(seq)
            if kind not in ("fast", "slow"):               # a gerund after any frame
                st["ger"] = True
            if kind == "keep":
                st["keep"] = True
            if kind in ("fast", "slow"):
                st["speed"] = kind
            continue
        if t in ("use", "using"):                      # "use your left paw to pick up the ball"
            j = i + 1
            if j < n and toks[j] in ("your", "the", "my", "both"):
                both = toks[j] == "both"
                j += 1
            else:
                both = False
            side = None
            if j < n and toks[j] in ("left", "right", "other"):
                side = toks[j]
                j += 1
            elif j < n and toks[j] in ("both", "two"):
                side = "both"
                j += 1
            elif both:
                side = "both"
            if side and _paw_at(A, toks, j) and j + 1 < n and toks[j + 1] in ("to", "and"):
                st["side"] = side
                i = j + 2
                continue
        return None
    return None
REL_HEAD = (["stop"], ["relax"], ["never", "mind"], ["nevermind"], ["thats", "enough"], ["enough"], ["okay", "stop"],
            ["ok", "stop"], ["cut", "it", "out"], ["quit", "it"], ["stop", "stop"], ["no", "stop"], ["wait", "stop"])
REL_NEG = ("dont", "no", "not", "never", "you", "do", "stop", "quit", "please")


def _act_release(A: Act, raw):
    """'stop, don't jump' / 'never mind the ball' / 'okay stop, no more backflips' -> release (the stop part)"""
    toks = _clean(A, raw)
    for h in REL_HEAD:
        if toks[:len(h)] == h and len(toks) > len(h):
            rest = toks[len(h):]
            if rest[0] in REL_NEG or (h in (["never", "mind"], ["nevermind"]) and rest[0] in ("the", "that", "it", "about")):
                if rest[0] == "you" and not (len(rest) > 1 and rest[1] in ("dont", "can", "do", "need", "dont")):
                    continue
                return {"action": "release", "parts": "all"}
    return None


HOLD_ON = (["dont", "let", "go"], ["dont", "let", "it", "go"], ["dont", "drop", "it"], ["keep", "holding", "it"],
           ["keep", "holding", "on"], ["hold", "on", "to", "it"], ["hold", "it", "tight"], ["dont", "let", "go", "of", "it"],
           ["and", "dont", "let", "go"], ["keep", "it"], ["keep", "holding", "them"])
GAZE_FREE = (["dont", "look", "at", "me"], ["do", "not", "look", "at", "me"], ["dont", "look"], ["no", "looking"],
             ["dont", "look", "at", "us"], ["and", "dont", "look", "at", "me"], ["but", "dont", "look", "at", "me"])


def _act_gaze_free(raw) -> bool:
    return any(raw == g for g in GAZE_FREE)


def _act_attach(hits, c) -> bool:
    """a legacy hold / look-at-me right after a move becomes that move's hold / gaze (one command)"""
    if not hits:
        return False
    prev = hits[-1]["cmd"]
    if prev.get("action") not in ("move", "object"):
        return False
    if c.get("action") == "hold":
        prev["hold"] = True
        return True
    if c.get("action") == "look" and c.get("target") == "user" and prev.get("action") == "move":
        prev["gaze"] = "user"
        return True
    return False


def _act_carry(A: Act, raw, prev, names):
    """'... and the blue one (with your other paw)' -> the previous object verb on a new object"""
    toks = _clean(A, raw)
    i = 0
    while i < len(toks) and (toks[i] in A.pre or toks[i] in names or toks[i] in ("then", "also", "now")):
        i += 1
    for e, d in _np_spans(A, toks, i):
        if not _desc_ok(d) and d.get("ref") not in ("other",):
            continue
        mods = _mods(A, toks, e, "o", prev["verb"])
        if mods is None:
            continue
        c = {"action": "object", "verb": prev["verb"], "object": d}
        x = {}
        for m, mc in mods:
            _apply(A, c, m["set"], mc, x)
        if prev.get("verb") in ("place", "put_back") and "place" in prev and "place" not in c:
            c["place"] = prev["place"]
        c.pop("speed", None)
        c.pop("duration_ms", None)
        return [_scrub(c)]
    return None


# "take the headphones off and put them back" — a plural pronoun after the request named ONE thing is that
# thing: {ref: it} (motion binds it to the step before), not {ref: all} (intent.js actPronouns, same rules)
PRON_MANY = ("them", "those", "these", "em", "they")
ALL_WORDS = ("all", "both", "everything", "every", "each")


def _concrete_obj(o):
    return isinstance(o, dict) and not o.get("ref") and bool(o.get("id") or o.get("name") or o.get("color") or o.get("category"))


def _obj_sig(o):
    return json.dumps([o.get("id") or "", o.get("name") or "", o.get("color") or "", o.get("category") or "", o.get("ordinal") or ""])


def _act_pronouns(out, words):
    ws = list(words or [])
    many = any(w in PRON_MANY for w in ws) and not any(w in ALL_WORDS for w in ws)
    seen = []
    for h in out:
        c = h.get("cmd")
        if not c or c.get("action") != "object":
            continue
        o = c.get("object")
        if _concrete_obj(o):
            seen.append(o)
            continue
        if not seen or not isinstance(o, dict) or len(o) != 1:
            continue
        if o.get("ref") == "all" and many and len(set(_obj_sig(x) for x in seen)) == 1:
            c["object"] = {"ref": "it"}
    return out


def _act_post(hits, words=None):
    return _act_pronouns(_act_post0(hits), words)


def _act_post0(hits):
    """toss + 'catch it' = one toss; 'turn around and face me' = turn back"""
    out = []
    for h in hits:
        c = h["cmd"]
        if out and c.get("action") == "move" and c.get("name") == "turn_back" and \
                out[-1]["cmd"].get("action") == "move" and out[-1]["cmd"].get("name") == "turn_away" and \
                not out[-1]["cmd"].get("hold") and (out[-1].get("_raw") or [])[-1:] in (["around"], ["round"]):
            out[-1] = h
            continue
        if out and c.get("action") == "object" and c.get("verb") == "catch" and \
                out[-1]["cmd"].get("action") == "object" and out[-1]["cmd"].get("verb") == "toss":
            continue
        out.append(h)
    return out
# ============================================================================== intent end

def _main():
    import sys
    m = Matcher()
    for line in (sys.argv[1:] or sys.stdin):
        print(line.strip(), "->", json.dumps([h["cmd"] for h in m.parse(line)]))


if __name__ == "__main__":
    _main()


def voiced_until_ms(pcm: bytes, t0_ms: float, last_ms: float, sr: int = 24000) -> float:
    """the end (audio clock, ms) of the last 20 ms frame of pcm16 that counts as the user's voice, else last_ms."""
    import array
    a = array.array("h"); a.frombytes(pcm[: len(pcm) // 2 * 2])
    n = sr // 50
    for k in range(len(a) // n - 1, -1, -1):
        f = a[k * n:(k + 1) * n]
        if (sum(x * x for x in f) / n) ** 0.5 >= VOICED_RMS:
            return max(last_ms, t0_ms + (k + 1) * 20.0)
    return last_ms
