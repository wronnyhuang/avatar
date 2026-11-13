"""Intent fallback: the server's SECOND fast path for the muse — an LLM read of a FINAL transcript the act grammar missed.

When the recognizer's final gets no command from the fast matcher and it looks like a request (a broad trigger lexicon
of action words; never on state questions, hypotheticals, reported speech or a bare negation), the server asks OpenAI
(gpt-4.1-mini by default, 1.5 s timeout) with the SAME move/object tool schema as app/rtc.py act_tools() plus the scene's
prop names, and sends its tool calls as ordinary intent commands with source 'matcher-llm' (the page's normal dedupe:
the voice model's later call for the same request is deduped, a disagreeing one replaces it).

config.json "intent": {"llm_fallback": true | false (default) | ["muse", ...], "llm_model": "gpt-4.1-mini", "llm_timeout_s": 1.5}
Default: off (true = every character, a list = those characters). Cost per call is appended as a JSON line to
$AVATAR_INTENT_SPEND_LOG when that env var names a file (local log only).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

import intent

NAMES = set(intent.load_rules().names)
PRICES = {"gpt-4.1-mini": (0.4e-6, 1.6e-6), "gpt-4.1-nano": (0.1e-6, 0.4e-6)}
SPEND_LOG = os.environ.get("AVATAR_INTENT_SPEND_LOG", "")
ACT_TARGETS = ("me", "closer", "back", "away")

# words that make a final worth a second look (action verbs / move nouns / gerunds; scene nouns are added per scene)
TRIGGER = set("""spin spins spinning twirl twirls twirling whirl whirling turn turning rotate rotating flip flips flipping
backflip backflips frontflip somersault cartwheel cartwheels jump jumps jumping hop hopping bounce bouncing leap leaping
walk walking step steps stepping move moving go come dance dancing boogie groove wiggle wiggling shake shaking sit sitting
seat stand standing get rise lie lay roll rolling bow bowing curtsy pick picking grab grabbing take taking lift lifting hold
holding show showing inspect look looking examine check toss tossing throw throwing catch juggle hand pass give offer put
putting set place drop let wear wearing remove pull pulling push pushing drag slide sliding bring bringing scoot shove open
opening close closing shut type typing work working use using stop pause quit face facing plop settle park""".split())
# a final that STARTS like this (after a greeting / the name) is not a request
NOT_REQ = re.compile(r"^(what|whats|why|when|where|who|whos|which|did|didnt|does|doesnt|was|wasnt|were|is|isnt|are|arent|"
                     r"have you|has|had|how (many|much|did|long|come|do|does|is|are|was)|do you (think|know|remember|like|see)|"
                     r"if|i wish|imagine|wouldnt it|wouldnt that|that was|that is|thats|it was|its (a|an|the|so|really|not|too|very|been|just)|you (did|were|are|just|always|never)|"
                     r"my|our|his|her|she|he|they|we (did|were|are)|the guy|someone|somebody|everyone|i (love|like|saw|think|was|am|have|had|did|just|picked|always|never|hate|heard|watched|remember|wonder))\b")
REPORTED = re.compile(r"\b(said|says|told|tells|asked|asks|saying|telling)\b")
# hypotheticals / wishes / state questions anywhere in the sentence ("man i wish you could ...", "soz muse if u ...")
HYPO = re.compile(r"\b(if (you|u|ya)|i wish|imagine|would be|would look|it would|wouldnt|pretend|pretending|in my mind|"
                  r"do you (like|think|know|remember|ever)|did you|have you|were you|are you|is it|was it|someday|one day)\b")
NEG_ONLY = re.compile(r"^(dont|do not|no|never|not|stop)\b")
RESUME = re.compile(r"\b(just|instead|but|rather)\b")
LEAD = {"hey", "hi", "hello", "ok", "okay", "so", "um", "uh", "umm", "uhh", "well", "alright", "oh", "now", "muse", "please",
        "and", "like", "just", "maybe", "hmm", "ah", "yo", "buddy", "then", "first", "next", "also", "kindly", "pls", "plz", "wait"}
# a request STARTS with an imperative (an action verb) or a request frame; hypotheticals, wishes, questions about state,
# reports and comments do not ("man i wish you could ...", "how fun it would be ...", "the laptop is open")
REQ_FRAME = re.compile(r"^((can|could|would|will|wont) (you|u|ya)|lets|let (me|us)|lemme|(show|give|gimme|hand|bring) (me|us)|"
                       r"i (bet|dare|want you|need you|would like you|would love to see|wanna see|want to see|d like you)|"
                       r"id (like|love)|(how|what) about|go ahead|try|its time|time (to|for)|simon says|do (a|an|the|some|two|three)|"
                       r"you (can|should|may|gotta|have to|need to)|why dont you|(would|do) you mind|do you (want|wanna)|wanna|"
                       r"make (a|an)|ready set|come on|cmon|no |dont |do not |skip )")

SYSTEM = ("You control Muse, a cuddly plush character on a screen with legs and paws, in a little room with: {props}. "
          "A person just said the sentence below out loud (speech-recognizer text, may contain misheard words). "
          "Call the tools ONLY if the sentence asks Muse to do something NOW (commands, 'can you ...', 'I bet you can't ...', "
          "'let's see you ...', 'show me ...'). One call per step, in the order Muse should do them. Never call a tool for "
          "questions about what Muse is doing or did, hypotheticals ('if you ...', 'imagine ...', 'it would be fun ...'), "
          "wishes, pretending, stories, reported speech, comments, or any "
          "negated part ('don't X' = no call for X). 'Give me' / 'hand me' = offer; hand_over = to its other paw; "
          "a plain 'flip' = backflip. Use the room's names for things. If it is not a request, answer NONE.")


def chat_tools():
    from rtc import act_tools                # the same schema the voice model gets
    out = []
    for t in act_tools():
        out.append({"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                     "parameters": t["parameters"]}})
    return out


def _s(v):
    if v is None or v == "":
        return None
    return re.sub(r"\s+", "_", str(v).lower().strip())


def _i(v, lo, hi):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def tool_to_cmd(name: str, a: dict):
    """page.html actToolCommand, in Python (the same normalisation of the model's arguments)"""
    a = a or {}
    c = {"action": "move", "name": _s(a.get("name"))} if name == "move" else {"action": "object", "verb": _s(a.get("verb"))}
    if not c.get("name") and not c.get("verb"):
        return None
    n = _i(a.get("count"), 1, 10)
    if n is not None:
        c["count"] = n
    if name == "move":
        for k in ("dir", "speed", "gaze"):
            v = _s(a.get(k))
            if v:
                c[k] = v
        if a.get("style"):
            c["style"] = str(a["style"])[:40]
        d = _i(a.get("distance"), 1, 10)
        if d is not None:
            c["distance"] = d
        if isinstance(a.get("height"), (int, float)):
            c["height"] = max(0.0, min(1.0, float(a["height"])))
    else:
        o = a.get("object")
        o = o if isinstance(o, dict) else ({"name": o} if isinstance(o, str) and o else None)
        if o:
            d = {k: str(o[k]).lower().strip() for k in ("ref", "id", "name", "color", "category") if o.get(k)}
            if d.get("ref") in ("that", "this"):
                d["ref"] = "it"
            if d:
                c["object"] = d
        tg = _s(a.get("target"))
        if tg:
            c["target"] = tg if tg in ACT_TARGETS else {"name": re.sub(r"^(the|my|your)_", "", tg).replace("_", " ")}
        pl = _s(a.get("place"))
        if pl:
            c["place"] = re.sub(r"^(the|my|your)_", "", pl)
        sd = _s(a.get("side"))
        if sd:
            c["side"] = sd
    if a.get("hold"):
        c["hold"] = True
    return c


class LlmFallback:
    def __init__(self, cfg: dict | None = None, log=print):
        cfg = cfg or {}
        self.model = cfg.get("llm_model") or "gpt-4.1-mini"
        self.timeout = float(cfg.get("llm_timeout_s") or 1.5)
        sw = cfg.get("llm_fallback", False)   # OFF by default (4/50 negatives fired on the c2 holdout); config can enable it
        self.chars = None if sw is True else (set(sw) if isinstance(sw, (list, tuple)) else (set() if not sw else {"muse"}))
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.log = log
        self.tools = None
        self.usd = 0.0
        self.calls = 0

    def on_for(self, character) -> bool:
        return bool(self.key) and bool(character) and (self.chars is None or character in self.chars)

    @staticmethod
    def props_text(scene) -> str:
        names = []
        for p in (scene or {}).get("props") or [] if isinstance(scene, dict) else []:
            nm = str(p.get("name") or p.get("id") or "").strip()
            if nm:
                names.append(nm)
        return ", ".join(names) or "a few toys"

    def wants(self, text: str, scene=None) -> bool:
        """a final worth asking about: action-ish, not a question / hypothetical / report / bare negation"""
        toks = [t for t in intent.normalize(text) if t != "|"]
        while toks and (toks[0] in LEAD or toks[0] in NAMES):
            toks = toks[1:]
        if len(toks) < 1 or len(toks) > 40:
            return False
        s = " ".join(toks)
        if NOT_REQ.match(s) or REPORTED.search(s) or text.strip().endswith("?") and NOT_REQ.match(s):
            return False
        if HYPO.search(s):
            return False
        if NEG_ONLY.match(s) and not RESUME.search(s):
            return False
        if toks[0] not in TRIGGER and not REQ_FRAME.match(s + " "):
            return False
        words = set(toks)
        nouns = set()
        for p in (scene or {}).get("props") or [] if isinstance(scene, dict) else []:
            for w in [p.get("id"), p.get("name")] + list(p.get("aliases") or []):
                nouns.update(intent.normalize(str(w or "").replace("_", " ")))
        return bool(words & TRIGGER) or bool(words & (nouns - {"|"}))

    def ask(self, text: str, scene=None) -> tuple[list[dict], dict]:
        """-> (commands, info{ms, usd, err}) — never raises"""
        t0 = time.monotonic()
        info = {"model": self.model}
        try:
            if self.tools is None:
                self.tools = chat_tools()
            body = {"model": self.model, "temperature": 0, "max_tokens": 200, "tools": self.tools, "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "messages": [{"role": "system", "content": SYSTEM.format(props=self.props_text(scene))},
                                 {"role": "user", "content": text}]}
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            u = r.get("usage") or {}
            pi, po = PRICES.get(self.model, PRICES["gpt-4.1-mini"])
            usd = u.get("prompt_tokens", 0) * pi + u.get("completion_tokens", 0) * po
            self.usd += usd
            self.calls += 1
            info["usd"] = round(usd, 6)
            cmds = []
            for tc in (r["choices"][0]["message"].get("tool_calls") or []):
                f = tc.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except ValueError:
                    continue
                c = tool_to_cmd(f.get("name"), args)
                if c:
                    cmds.append(c)
            if SPEND_LOG:
                try:
                    with open(SPEND_LOG, "a") as fh:
                        fh.write(json.dumps({"ts": int(time.time()), "agent": "intent-llm", "vendor": "openai", "usd": round(usd, 6),
                                             "what": f"muse matcher LLM fallback ({self.model})"}) + "\n")
                except OSError:
                    pass
        except Exception as e:      # timeout, network, schema: the fallback is best-effort
            info["err"] = type(e).__name__
            cmds = []
        info["ms"] = round((time.monotonic() - t0) * 1000, 1)
        return cmds, info
