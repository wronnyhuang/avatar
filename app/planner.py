"""Server-side live keyframe PLANNER (docs/PLANNER.md sections 0-2), for every character.

A commanded action's description -> poses for the page's motion layer, in this order:
  1. library  the intent matcher fully understands it (intent_hub.matcher_for(character); Matcher.understood) and
              nothing it said is lost (counts, speed, style, parts the character has; seated characters: no whole-body
              commands) -> source 'library', engine = the matcher's commands, no poses. Instant.
  2. cache    AVATAR_HOME/cache/plans/<sha256(character + "\\n" + normalized description)>.json from an earlier
              planner answer -> source 'cache', every pose at once. Instant.
  3. planner  the configured model (see "Model choice" below) with the character's prompt
              (app/planner_prompts/<key>.txt, sha-checked against MANIFEST.json) and the frozen user message; poses
              are parsed as the text streams (the streaming parser below) and yielded batch by batch -> source
              'planner'. Timeout 25 s for the whole plan; without a fallback model one retry on a transient error (HTTP
              429/5xx, connection, timeout) while no pose has been yielded and >= retry_min_s remain; with one, the
              hedged fallback below. A complete answer is written to the cache (with the model that wrote it).

    async for ev in planner.plan(character, description, state, http=session, on_usd=cb):
        ev = {"done": False, "poses": [...]}                       one per parsed batch (planner)
        ev = {"done": True, "ok", "source", "poses" (not yet sent), "all_poses", "engine", "summary", "ms_first_pose",
              "ms_done", "usd", "error", "prompt_sha", "cache_key", "model", "route", "fallback_used", "attempts", ...}
    kf_message(id, ev) -> the server -> page {type:'kf', ...} message of contract section 2.

Pose (contract section 1) = {"t": <s from motion start>, "<own-side control>": <value>, ...} — only the controls that
change; t is the parser's (an early first key eases in from rest: +0.3 s).
Config (state/config.json): "planner": true | false | {"enabled", "model", "route", "effort", "fallback", "zdr",
"quality", "cache_models", "timeout_s", "retry_min_s", "cache", "library", "max_output_tokens", "max_concurrent",
"stall_s", "stall_motion_s"} over DEFAULTS; false = the server refuses plans (ok:false, 'planner off').

Model choice: "model" = a PRESETS key or a raw model id, or "auto" (the default) = the first route this machine can
use: the `claude` CLI on PATH (route claude_door, preset haiku55door; no API key) -> ANTHROPIC_API_KEY (route
anthropic, haiku55api) -> OPENROUTER_API_KEY (gem35lite, Gemini 3.5 Flash Lite) -> OPENAI_API_KEY (gpt54mini).
"route" auto|openai|openrouter|openrouter_zdr|gemini|meta|anthropic|claude_door (auto: the preset's, else z-ai/
deepseek/ -> openrouter_zdr, other "/" -> openrouter, "gemini-" -> gemini, "muse-" -> meta, "claude-" -> claude_door,
else openai; GLM / DeepSeek always go zero-retention; "zdr": true = every OpenRouter call); "effort" "auto" = the
preset's (a raw id: none sent), null = none sent, else sent as is. "fallback" null | "auto" | <preset or id> |
{"model", "route", "effort", "after_s"} — default "auto" (a second route the machine can use: dsv41flash behind
gem35lite, else Anthropic API / OpenRouter / OpenAI, in that order) after 2.0 s: HEDGED — started when the primary has
no pose after_s after it started or failed before its first pose (the primary keeps running); the first of the two to
stream a pose wins, the other is cancelled (and billed); poses of one plan come from one model. Without a fallback: one
retry on a transient error. A STALL (no new pose for > stall_s while < stall_motion_s of motion is parsed) is reported
(done "stalls"), never a switch (its poses are already out). Env AVATAR_PLANNER_MODEL / AVATAR_PLANNER_FALLBACK (preset
or id; "none" = no fallback) / AVATAR_PLANNER_FALLBACK_AFTER_S override the file. "cache_models": plans by these models
are served from the cache besides the primary's and the fallback's (default ["gpt-6.1-sol"]); "quality" (default the
"quality" preset = Sol, effort low) is the model Planner.fill uses for offline library / cache filling.
Route "anthropic" = the Anthropic Messages API (ANTHROPIC_API_KEY, x-api-key + anthropic-version headers, SSE).
Route "claude_door" = the `claude` CLI through app/claude_door.py (a warm pool of `claude -p` stream-json processes;
no API key, usd 0 — the door's API-equivalent cost is kept in the attempt record as "door_usd"). Presets haiku55door /
sonnet55door / opus55door (model claude-<name>-5-5, effort = the CLI's --effort "low") and *_nt (effort "none" =
--effort low with thinking off: MAX_THINKING_TOKENS=0 in the door's processes). Same parser, hedge, cache and timeout as
every route; a cancel / timeout closes the door stream, which kills its process. Planner.prewarm(character) keeps the
door's warm processes for that character's prompt (the server calls it when a session opens); a cold door costs a
process start (~1-2 s) on the first plan.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(HERE, "planner_prompts")
URL = "https://api.openai.com/v1/responses"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
META_URL = "https://api.meta.ai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ROUTES = ("openai", "openrouter", "openrouter_zdr", "gemini", "meta", "anthropic", "claude_door")
KEY_ENV = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openrouter_zdr": "OPENROUTER_API_KEY",
           "gemini": "GEMINI_API_KEY", "meta": "META_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DOOR_ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
ZDR_PREFIXES = ("z-ai/", "deepseek/")    # design note: GLM / DeepSeek only on zero-retention providers
USER_TMPL = "Movement request: {text}\nAnswer with the JSON object."      # the frozen eval user message
# the streamed-keyframe eval's models, each with the exact route / model id / effort it was evaluated with.
# openrouter_zdr = OpenRouter with provider {zdr: true, data_collection: deny} (the route for GLM / DeepSeek). gem35lite on OpenRouter ran with no
# reasoning field (its 19 OpenRouter calls: reasoning_tokens 0 — the Gemini-key calls' thinkingLevel minimal equivalent).
PRESETS = {
    "sol":              {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"},
    "quality":          {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"},     # offline library / cache filling
    "gpt54mini":        {"model": "gpt-5.4-mini", "route": "openai", "effort": "none"},
    "gem38flash":       {"model": "google/gemini-3.8-flash", "route": "openrouter", "effort": "low"},
    "gem35lite":        {"model": "google/gemini-3.5-flash-lite", "route": "openrouter", "effort": None},
    "gem35lite_gemini": {"model": "gemini-3.5-flash-lite", "route": "gemini", "effort": "minimal"},   # free-tier key: 429s
    "glm53":            {"model": "z-ai/glm-5.3", "route": "openrouter_zdr", "effort": "minimal"},
    "dsv41flash":       {"model": "deepseek/deepseek-v4.1-flash", "route": "openrouter_zdr", "effort": "none"},
    "dsv4pro":          {"model": "deepseek/deepseek-v4-pro", "route": "openrouter_zdr", "effort": "none"},
    "musespark11":      {"model": "muse-spark-1.1", "route": "meta", "effort": "minimal"},
    "haiku55or":        {"model": "anthropic/claude-haiku-5.5", "route": "openrouter", "effort": None},
    # the `claude` CLI through app/claude_door.py (no API key; usd 0)
    "haiku55door":      {"model": "claude-haiku-5-5", "route": "claude_door", "effort": "low"},
    "sonnet55door":     {"model": "claude-sonnet-5-5", "route": "claude_door", "effort": "low"},
    "opus55door":       {"model": "claude-opus-5-5", "route": "claude_door", "effort": "low"},
    "haiku55door_nt":   {"model": "claude-haiku-5-5", "route": "claude_door", "effort": "none"},    # thinking off
    "sonnet55door_nt":  {"model": "claude-sonnet-5-5", "route": "claude_door", "effort": "none"},
    "opus55door_nt":    {"model": "claude-opus-5-5", "route": "claude_door", "effort": "none"},
    # Anthropic Messages API with ANTHROPIC_API_KEY (effort "none" = thinking disabled + output_config effort low)
    "haiku55api":       {"model": "claude-haiku-5-5", "route": "anthropic", "effort": "none"},
    "sonnet55api":      {"model": "claude-sonnet-5-5", "route": "anthropic", "effort": "low"},
}
# USD/M tokens: in, cached in, out. List prices, raised to what OpenRouter actually billed in the eval
# (usage.cost_details) — OpenRouter's reported usage.cost wins when present,
# so these only price usage without a cost and the estimate of an aborted / cancelled stream.
PRICE = {"gpt-6.1-sol": (2.0, 0.10, 10.0), "gpt-5.4-mini": (0.75, 0.075, 4.5),
         "google/gemini-3.8-flash": (0.75, 0.75, 3.75), "gemini-3.8-flash": (0.5, 0.5, 3.0),
         "google/gemini-3.5-flash-lite": (0.3, 0.3, 2.5), "gemini-3.5-flash-lite": (0.1, 0.1, 0.4),
         "z-ai/glm-5.3": (0.18, 0.18, 1.85), "deepseek/deepseek-v4.1-flash": (0.051, 0.051, 0.64),
         "deepseek/deepseek-v4-pro": (0.9, 0.9, 1.54), "muse-spark-1.1": (0.0, 0.0, 0.0),
         "anthropic/claude-haiku-5.5": (0.1, 0.1, 0.5), "claude-haiku-5-5": (0.1, 0.01, 0.5),
         "claude-sonnet-5-5": (2.0, 0.10, 10.0)}
PRICE_UNKNOWN = {"openai": (2.0, 0.10, 10.0)}             # an unlisted model: the old default (openai), else (1, 1, 5)
PLAN_MAX = 18
GRACE_S = 1.5          # after the poses array closed: how long the stream is read on for its usage event
# Fallback start: no pose from the primary 2.0 s after it started. gem35lite's first pose in the eval (54 items): p50 0.97-1.11 s, p90 1.17-1.26 s, max 1.30 s on either route — 2.0 s is >= 1.5x
# its worst, so a healthy primary is never hedged, and a dead one costs 2.0 s + dsv41flash's first pose (p50 0.78 s,
# p90 1.52 s) = ~2.8 / 3.5 s, about Sol's own p50 / p90 (2.30 / 3.87 s).
FALLBACK_AFTER_S = 2.0
STALL_S = 1.5           # a stall (reported, never a model switch): no new pose for > 1.5 s while the poses array is
STALL_MOTION_S = 1.0    # open and < 1.0 s of motion is parsed (the page's buffer rule: >= 2 poses and >= 1.0 s)
DEFAULTS = {"enabled": True, "model": "auto", "route": "auto", "effort": "auto",
            "fallback": {"model": "auto", "after_s": FALLBACK_AFTER_S}, "zdr": "auto", "quality": "quality",
            "cache_models": ["gpt-6.1-sol"], "timeout_s": 25.0, "retry_min_s": 8.0, "cache": True, "library": True,
            "max_output_tokens": 12000, "max_concurrent": 3, "stall_s": STALL_S, "stall_motion_s": STALL_MOTION_S,
            # "prompt_variant" = the MANIFEST variant served instead of <character>.txt ("q1": the Muse
            # family's body-true prompt; "" / "frozen" / false = the frozen prompt, exactly as before; env
            # AVATAR_PLANNER_PROMPT overrides). "check" = plan_issues() gates the plan cache (a plan that reads as
            # standing still is not written, and an old entry that does is not served);
            # false = the cache exactly as before (env AVATAR_PLANNER_CHECK=0).
            "prompt_variant": "q1", "check": True,
            # "no_think" = on the claude_door route the movement request carries NO_THINK_HINT (the user
            # turn only: the cached system prompt, its sha and the plan cache keys do not change). Sonnet 5.5 opens a
            # hidden adaptive-thinking block on ~1 in 4 whole-body requests (skip, hop, push-up, downward dog) whatever
            # the CLI is told (MAX_THINKING_TOKENS=0, --effort low, CLAUDE_CODE_DISABLE_THINKING / _ADAPTIVE_THINKING,
            # alwaysThinkingEnabled=false all measured 0 effect): 500-1500 hidden tokens = first pose 3.3-9.4 s instead
            # of ~1.6 s. The hint removes it on ~9 of 10 such calls. false = the user turn as
            # before (env AVATAR_PLANNER_NO_THINK=0).
            "no_think": True,
            # "fallback_on_thinking" = start the fallback the moment the claude_door primary opens a hidden
            # thinking block (~0.8 s in; its text would come 3-9 s later) instead of at after_s; the primary keeps
            # running and still wins if it streams a pose first. false = the fallback waits for after_s, as before.
            "fallback_on_thinking": False, "hedge_after_s": None, "prewarm_things": True}
NO_THINK_HINT = "Start writing it immediately: do not think or plan first."

# exec_stream.cjs KF_ALIAS keys: the own-side controls the parser knows by default (the Muse's set)
KF_CONTROLS = ([f"{own}_paw_{a}" for own in ("right", "left") for a in ("x", "y", "z", "open", "turn")]
               + [f"{own}_{p}" for own in ("right", "left") for p in ("thumb", "shoulder", "fingers", "ear", "brow", "point")]
               + [f"{own}_foot_{a}" for own in ("right", "left") for a in ("x", "y", "z")]
               + ["eyes_closed", "root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash", "body_lean",
                  "body_pitch", "body_yaw", "head_yaw", "head_pitch", "head_roll", "smile", "mouth_open", "eye_wide", "squint",
                  "tongue_out", "brow_knit", "tail_raise", "tail_wag", "tail_curl", "nose_wiggle"])
KF_SET = frozenset(KF_CONTROLS)
# objects (docs/OBJECTS.md section 6): the keyframe language WITH THINGS, for a character whose page
# sends its things (state.things) and that has a props file: held-object controls (continuous, offsets of the held thing
# from the paw's grasp, H / turns), paw events (value 1 at the pose time = fire once when playback crosses it; *_toss_up =
# let go with an upward throw of that many H) and object steps ("do": the engine's object command, the clock pauses for it)
KF_HELD = [f"{own}_held_{a}" for own in ("right", "left") for a in ("x", "y", "z", "spin", "tilt", "roll")]
KF_EVENTS = [f"{own}_{e}" for own in ("right", "left") for e in ("let_go", "catch", "grab", "toss_up")]
KF_OBJ_CONTROLS = KF_HELD + KF_EVENTS
DO_KEYS = ("verb", "object", "target", "relation", "place", "side", "to", "count", "amount", "dir", "speed", "height",
           "attach", "anchor", "style")
DO_RELATIONS = ("next to", "onto", "into", "on top of", "under", "behind", "in front of", "beside", "on", "in", "at", "to")


# ------------------------------------------------------------------------------------------------------ JS helpers
def _nan_const(_):
    raise ValueError("not JSON")       # JSON.parse rejects NaN / Infinity; Python's json accepts them by default


def _jparse(s: str):
    return json.loads(s, parse_constant=_nan_const)


_UNDEF = object()


def _js_str(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        v = float(v)
        return "NaN" if not math.isfinite(v) else str(int(v)) if v.is_integer() and abs(v) < 1e21 else repr(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join("" if x is None else _js_str(x) for x in v)
    return "[object Object]"


def _js_num(v) -> float:
    """JavaScript's unary + (ToNumber) for a JSON value."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except OverflowError:                      # a huge JSON integer: JS reads it as +-Infinity
            return float("inf") if v > 0 else float("-inf")
    if isinstance(v, list):
        return _js_num(_js_str(v))
    if isinstance(v, dict):
        return float("nan")
    s = v.strip()
    if s == "":
        return 0.0
    if re.fullmatch(r"[+-]?Infinity", s):
        return float("inf") if not s.startswith("-") else float("-inf")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return float(int(s, 16))
    if re.fullmatch(r"0[oO][0-7]+", s):
        return float(int(s[2:], 8))
    if re.fullmatch(r"0[bB][01]+", s):
        return float(int(s[2:], 2))
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        return float(s)
    return float("nan")


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _js_json(v) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def norm_do(v):
    """a pose's "do" value -> the object step {"verb", ...} (the object tool's shape) or None. A string "pick_up ball" /
    "put ball on block_a" is read as verb + object [+ relation + target]."""
    if isinstance(v, str):
        w = v.strip().lower().replace("-", "_")
        if not w:
            return None
        w = re.sub(r"^(pick|put|take|hand|turn|sit|stand|work|type) (up|down|back|on|off|over|in|at)\b", r"\1_\2", w)
        # "go to board", "walk over to tv", "knock on desk", "lean on desk" -> go_to / knock_on ...
        w = re.sub(r"^(go|walk|knock|rap|lean|tap|pat|press) (over to|up to|to|on|against)\b", lambda m: m.group(1) + "_" + m.group(2).replace(" ", "_"), w)
        verb, _, rest = w.partition(" ")
        o = {"verb": verb.strip()}
        rest = rest.strip()
        if rest:
            for rel in DO_RELATIONS:
                m = re.search(r"\s" + re.escape(rel) + r"\s", " " + rest + " ")
                if m and m.start() > 0:
                    a, b = rest[:m.start() - 1].strip(), rest[m.end() - 1:].strip()
                    if a and b:
                        o.update({"object": a, "relation": rel.replace(" ", "_"), "target": b})
                        break
            else:
                o["object"] = rest
        v = o
    if not isinstance(v, dict):
        return None
    verb = v.get("verb") if isinstance(v.get("verb"), str) else v.get("action") if isinstance(v.get("action"), str) else None
    if not verb or not re.fullmatch(r"[a-z_ ]{2,30}", verb.strip().lower()):
        return None
    out = {"verb": verb.strip().lower().replace(" ", "_")}
    out["verb"] = {"put": "place", "set": "place", "take": "pick_up", "grab": "pick_up", "wear": "put_on"}.get(out["verb"], out["verb"])
    for k in DO_KEYS[1:]:
        x = v.get(k)
        if isinstance(x, bool) or x is None or x == "":
            continue
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            out[k] = x
        elif isinstance(x, str):
            out[k] = x.strip()[:60]
        elif isinstance(x, dict):              # {"name": .., "id": .., "color": .., "ref": ..} (the object tool's object)
            d = {kk: str(vv)[:40] for kk, vv in x.items() if kk in ("id", "name", "color", "ref", "place") and
                 isinstance(vv, (str, int, float)) and not isinstance(vv, bool)}
            if d:
                out[k] = d
    return out


# ------------------------------------------------------------------------------------------------------ parser
_POSES_RE = re.compile(r'"poses"\s*:\s*\[')
_TRAIL_RE = re.compile(r",\s*([}\]])")
_PAIR_RE = re.compile(r'"([A-Za-z_]+)"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


class StreamParser:
    """Incremental port of exec_stream.cjs parseStream: feed(ms, text) as the answer streams; a pose exists from the
    arrival ms of its closing brace. Text before `"poses": [` (summary, fences, prose) is skipped; each top-level element
    of that array is parsed when it closes (trailing commas repaired; a broken pose keeps its well-formed "name": number
    pairs if one is t, counted as salvaged); the first key at t < 0.2 shifts the plan by +0.3 s (eases in from rest);
    a pose not after the previous one or past PLAN_MAX is dropped; the array's ']' completes the parse.
    known = the control names kept (default: exec_stream.cjs KF_ALIAS); other names are reported in .errors."""

    def __init__(self, known=None, allow_do: bool = False):
        self.known = frozenset(known) if known is not None else KF_SET
        self.allow_do = bool(allow_do)          # "do" object steps (characters with things)
        self.text = ""
        self.i = 0
        self.arr = -1
        self.depth = 0
        self.in_str = False
        self.esc = False
        self.el = -1
        self.poses: list[dict] = []        # {"t", "set", "ms"}
        self.dropped: list[dict] = []
        self.salvaged: list[dict] = []
        self.errors: list[dict] = []
        self.complete_ms = None
        self.complete_how = None
        self.first_token_ms = None
        self.last_t = -1.0
        self.shift = None

    @property
    def complete(self) -> bool:
        return self.complete_ms is not None

    @property
    def found_poses_key(self) -> bool:
        return self.arr >= 0

    def _take(self, raw: str, ms) -> dict | None:
        try:
            o = _jparse(raw)
        except ValueError:
            try:
                o = _jparse(_TRAIL_RE.sub(r"\1", raw))
            except ValueError:
                pairs = [(m.group(1), float(m.group(2))) for m in _PAIR_RE.finditer(raw)]
                if not any(k == "t" for k, _ in pairs):
                    self.dropped.append({"ms": ms, "why": "bad json", "raw": raw[:160]})
                    return None
                o = dict(pairs)
                self.salvaged.append({"ms": ms, "raw": raw[:160]})
        if isinstance(o, list):                    # [t, {controls}]
            t = _js_num(o[0]) if o else float("nan")
            st = o[1] if len(o) > 1 else None
        elif isinstance(o, dict):                  # {"t": .., controls} | {"time": .., "set": {..}}
            tv = o.get("t")
            if tv is None:                         # JS `o.t ?? o.time`: a missing time is NaN, a null one 0
                tv = o["time"] if "time" in o else _UNDEF
            t = float("nan") if tv is _UNDEF else _js_num(tv)
            s0 = o.get("set")
            st = s0 if isinstance(s0, (dict, list)) else {k: v for k, v in o.items() if k not in ("t", "time")}
        else:
            t, st = float("nan"), None
        if not _finite(t) or not isinstance(st, (dict, list)):
            self.dropped.append({"ms": ms, "why": "no t / controls", "raw": raw[:120]})
            return None
        items = st.items() if isinstance(st, dict) else ((str(k), v) for k, v in enumerate(st))
        clean, unknown = {}, []
        step = None
        for k, v in items:
            if k == "do" and self.allow_do:            # an object step (bad shape -> reported, the pose stays)
                step = norm_do(v)
                if step is None:
                    unknown.append("do=" + _js_json(v)[:80])
                continue
            if k not in self.known:
                unknown.append(k)
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(_js_num(v)):
                clean[k] = _js_num(v)
            elif v is not None and v != "" and _finite(_js_num(v)):
                clean[k] = _js_num(v)
            else:
                unknown.append(k + "=" + _js_json(v))
        if unknown:
            self.errors.append({"ms": ms, "unknown": unknown})
        if self.shift is None:
            self.shift = 0.3 if (t < 0.2 and (clean or step)) else 0.0
        t = max(0.0, t) + self.shift
        if t <= self.last_t + 1e-6:
            self.dropped.append({"ms": ms, "why": f"t {t} not after {self.last_t}"})
            return None
        if t > PLAN_MAX + self.shift + 1e-6:
            self.dropped.append({"ms": ms, "why": f"t {t} > {PLAN_MAX}"})
            return None
        self.last_t = t
        p = {"t": t, "set": clean, "ms": ms}
        if step is not None:
            p["do"] = step
        self.poses.append(p)
        return p

    def feed(self, ms, s: str) -> list[dict]:
        """One arrived chunk -> the poses it completed."""
        if self.first_token_ms is None:
            self.first_token_ms = ms
        if self.complete:
            return []
        self.text += s or ""
        text = self.text
        new = []
        if self.arr < 0:
            m = _POSES_RE.search(text)
            if not m:
                return new
            self.arr = m.end()
            self.i = self.arr
        i = self.i
        n = len(text)
        while i < n:
            c = text[i]
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif c == "\\":
                    self.esc = True
                elif c == '"':
                    self.in_str = False
                i += 1
                continue
            if c == '"':
                self.in_str = True
            elif c == "{" or c == "[":
                if self.depth == 0:
                    self.el = i
                self.depth += 1
            elif c == "}" or c == "]":
                if self.depth == 0:
                    if c == "]" and self.complete_ms is None:
                        self.complete_ms = ms
                        self.complete_how = "array closed"
                    i = n                      # exec_stream.cjs: `i = text.length; break` (resumes at the next chunk)
                    break
                self.depth -= 1
                if self.depth == 0:
                    p = self._take(text[self.el:i + 1], ms)
                    if p is not None:
                        new.append(p)
            i += 1
        self.i = i
        return new

    def finish(self, t_done_ms) -> None:
        if self.complete_ms is None:
            self.complete_ms = t_done_ms
            self.complete_how = "no poses array" if self.arr < 0 else "stream ended"

    def result(self) -> dict:
        """exec_stream.cjs's printed `parse` block (poses with t rounded to 3 decimals)."""
        P = self.poses
        return {"n_poses": len(P), "first_token_ms": self.first_token_ms, "first_pose_ms": P[0]["ms"] if P else None,
                "complete_ms": self.complete_ms, "complete_how": self.complete_how, "found_poses_key": self.found_poses_key,
                "shift_s": self.shift or 0, "dropped": self.dropped, "salvaged": self.salvaged, "unknown": self.errors,
                "poses": [dict({"ms": p["ms"], "t": round(p["t"], 3), "set": p["set"]}, **({"do": p["do"]} if "do" in p else {}))
                          for p in P],
                "plan_s": round(P[-1]["t"], 3) if P else 0}


def parse_stream(chunks, t_done=None, known=None, allow_do=False) -> dict:
    """Whole recorded stream ({chunks: [[ms, text], ...], t_done_ms}) -> exec_stream.cjs's parse block."""
    sp = StreamParser(known, allow_do)
    for ms, s in chunks:
        sp.feed(ms, s)
        if sp.complete:
            break
    if t_done is None:
        t_done = chunks[-1][0] if chunks else 0
    sp.finish(t_done)
    return sp.result()


def wire_pose(p: dict) -> dict:
    """Parser pose -> contract section 1 pose {"t", "<control>": value, ...}"""
    out = {"t": round(p["t"], 4)}
    for k, v in p["set"].items():
        out[k] = round(v, 4) + 0.0          # no -0.0 on the wire (JSON.stringify prints 0)
    if p.get("do"):
        out["do"] = p["do"]                 # an object step
    return out


# ------------------------------------------------------------------------------------------------------ normalization
FILLER_PHRASES = ["i would like you to", "i'd like you to", "i want you to", "i need you to", "can you please", "could you please",
                  "would you please", "will you please", "can you", "could you", "would you", "will you", "would you mind",
                  "go ahead and", "for me", "for us", "right now", "if you can", "if you could", "thank you", "thanks"]
FILLER_WORDS = {"please", "now", "pls", "plz", "kindly", "just", "okay", "ok", "hey", "um", "uh", "umm", "hmm", "so", "well"}
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
MULTI = {"twice": ["2", "times"], "thrice": ["3", "times"]}


def normalize(text: str) -> str:
    """Cache-key normalization: lowercase, strip punctuation, collapse spaces, drop politeness / filler words, number
    words -> digits ("twice" -> "2 times"). 'high five' stays words (a gesture, not a count)."""
    s = str(text or "").lower().replace("’", "'")
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    s = re.sub(r"'", "", s)                                   # contractions: "don't" -> "dont"
    s = re.sub(r"[^\w\s.]|_", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)                 # keep decimal points (1.5 seconds)
    s = " " + re.sub(r"\s+", " ", s).strip() + " "
    for ph in sorted(FILLER_PHRASES, key=len, reverse=True):
        s = s.replace(" " + ph.replace("'", "") + " ", " ")
    toks = [w for w in s.split() if w not in FILLER_WORDS]
    out: list[str] = []
    i = 0
    while i < len(toks):
        w = toks[i]
        if w == "high" and i + 1 < len(toks) and toks[i + 1] == "five":
            out += ["high", "five"]
            i += 2
            continue
        if w in MULTI:
            out += MULTI[w]
        elif w in TENS:
            v = TENS[w]
            if i + 1 < len(toks) and toks[i + 1] in NUM_WORDS and 0 < NUM_WORDS[toks[i + 1]] < 10:
                v += NUM_WORDS[toks[i + 1]]
                i += 1
            out.append(str(v))
        elif w in NUM_WORDS:
            v = NUM_WORDS[w]
            if i + 1 < len(toks) and toks[i + 1] == "hundred":
                v *= 100
                i += 1
            out.append(str(v))
        elif w == "hundred":
            out.append("100")
        else:
            out.append(w)
        i += 1
    return " ".join(out)


def cache_key(character: str, norm: str, note: str = "") -> str:
    return hashlib.sha256((character + "\n" + norm + ("\n" + note if note else "")).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------ prompts
def load_prompts(d: str = PROMPT_DIR) -> dict:
    """{character: {"text", "sha256", "controls", "traits"}} from MANIFEST.json; a file whose sha differs is left out
    (its plans fail with 'prompt sha mismatch')."""
    with open(os.path.join(d, "MANIFEST.json"), encoding="utf-8") as f:
        man = json.load(f)
    out = {}
    for key, ent in man["characters"].items():
        try:
            with open(os.path.join(d, ent["file"]), encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out[key] = {"text": text, "sha256": h, "ok": h == ent["sha256"], "controls": ent["controls"],
                    "traits": ent["traits"]}
    return out


def load_variants(d: str = PROMPT_DIR) -> dict:
    """{character: {variant: {"text", "sha256", "ok", "controls", "traits"}}} from MANIFEST "variants"
    (app/planner_prompts/make_prompts.py VARIANTS); {} when the manifest has none."""
    try:
        with open(os.path.join(d, "MANIFEST.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return {}
    out: dict = {}
    for key, vs in (man.get("variants") or {}).items():
        for name, ent in (vs or {}).items():
            try:
                with open(os.path.join(d, ent["file"]), encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            out.setdefault(key, {})[name] = {"text": text, "sha256": h, "ok": h == ent["sha256"],
                                             "controls": ent["controls"], "traits": ent["traits"], "variant": name,
                                             "for": list(ent.get("for") or [key])}
    return out


def prompt_variant(cfg) -> str:
    """the configured variant name ('' = the frozen prompts)"""
    v = os.environ.get("AVATAR_PLANNER_PROMPT")
    if v is None:
        v = (cfg or {}).get("prompt_variant", DEFAULTS["prompt_variant"])
    v = "" if v in (None, False) else str(v).strip()
    return "" if v.lower() in ("", "frozen", "off", "0", "false", "none") else v


# ------------------------------------------------------------------------------------- the plan check
# What a FINISHED plan must have to be worth keeping: something a viewer can see, for long enough to see it. Measured
# on the wire poses (own-side control names), in each control's own visible unit — the amounts the q1 prompt quotes
# ("a paw that travels less than 0.3 in every control looks like it did not move"; the waist dials are 12-23 degrees
# at 1.0). It never changes a pose and never delays one: it runs once, on the done event, and only decides whether the
# plan may live in the cache (a plan that shows nothing is planned afresh next time instead of being served for good); the
# softer findings are only reported ("issues" in the done event and the hub log).
CHECK_VISIBLE = (                                    # (control regexp, the change from rest that plainly shows)
    (re.compile(r"^(right|left)_paw_[xyz]$"), 0.3), (re.compile(r"^root_[xz]$"), 0.1), (re.compile(r"^root_y$"), 0.08),
    (re.compile(r"^root_(yaw|pitch|roll)$"), 0.04), (re.compile(r"^(right|left)_foot_[xyz]$"), 0.06),
    (re.compile(r"^body_(lean|pitch|yaw|bounce)$"), 0.5), (re.compile(r"^head_(yaw|pitch|roll)$"), 0.15),
    (re.compile(r"^squash$"), 0.2), (re.compile(r"^(right|left)_shoulder$"), 0.6),
    (re.compile(r"^(smile|mouth_open|eye_wide|squint|eyes_closed|tongue_out|brow_knit|(right|left)_(brow|ear|point|thumb)|"
                r"tail_wag|tail_raise|tail_curl|nose_wiggle)$"), 0.4),
    (re.compile(r"^(right|left)_fingers$"), 1.0), (re.compile(r"^(right|left)_paw_(open|turn)$"), 0.6))
CHECK_REST = {"smile": 0.1, "tail_raise": 0.4, "right_paw_open": 0.2, "left_paw_open": 0.2}
CHECK_MIN_S = 0.5                                    # a plan whose last pose is earlier is over before it shows
CHECK_HARD = ("static",)                             # the issue that keeps a plan out of the cache (the others are reported)


def plan_issues(poses) -> list[str]:
    """-> the names of what is wrong with a finished plan ([] = fine): 'static' (no control ever leaves rest by an amount
    that shows), 'short' (the last pose before CHECK_MIN_S), 'held_tiny' (a control reverses >= 3 times and never swings
    by an amount that shows, while nothing else in the plan moves far either: a rhythm nobody can see).
    Plans with scene steps (a `do` key / paw events) are never judged here: what they show is the engine's."""
    poses = [p for p in (poses or []) if isinstance(p, dict)]
    if not poses:
        return ["static", "short"]
    if any("do" in p or any(k.endswith(("_let_go", "_catch", "_grab", "_toss_up")) or "_held_" in k for k in p) for p in poses):
        return []
    out = []
    best = 0.0                                       # the largest departure from rest, in visible units
    swing: dict = {}       # control -> [last value, last direction, reversals, largest swing between two turns, last turn]
    for p in poses:
        for k, v in p.items():
            if k in ("t", "ms") or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            unit = next((u for rx, u in CHECK_VISIBLE if rx.match(k)), None)
            if unit is None:
                continue
            best = max(best, abs(v - CHECK_REST.get(k, 0.0)) / unit)
            w = swing.setdefault(k, [CHECK_REST.get(k, 0.0), 0, 0, 0.0, None, unit])
            d = v - w[0]
            if abs(d) > 1e-6:
                sg = 1 if d > 0 else -1
                if w[1] and sg != w[1]:              # a turn: the stretch since the turn before it is one swing
                    if w[4] is not None:
                        w[3] = max(w[3], abs(w[0] - w[4]) / unit)
                    w[4] = w[0]
                    w[2] += 1
                w[1] = sg
                w[0] = v
    for w in swing.values():                         # the stretch after the last turn
        if w[4] is not None:
            w[3] = max(w[3], abs(w[0] - w[4]) / w[5])
    if best < 1.0:
        out.append("static")
    t_last = max((float(p.get("t") or 0.0) for p in poses), default=0.0)
    if t_last < CHECK_MIN_S:
        out.append("short")
    rhythm = [w for w in swing.values() if w[2] >= 3]
    if rhythm and max(w[3] for w in rhythm) < 1.0 and best < 2.0:
        out.append("held_tiny")
    return out


def check_on(cfg) -> bool:
    v = os.environ.get("AVATAR_PLANNER_CHECK")
    if v is not None:
        return v.strip().lower() not in ("0", "false", "off", "no", "")
    return bool((cfg or {}).get("check", DEFAULTS["check"]))


def no_think_mode(cfg) -> str:
    """cfg "no_think" (env AVATAR_PLANNER_NO_THINK) -> "always" (true: every claude_door request carries
    the hint) | "retry" (the request goes as before; the moment it opens a hidden thinking block a second request WITH
    the hint starts beside it, first pose wins) | "off" (false)"""
    v = os.environ.get("AVATAR_PLANNER_NO_THINK")
    if v is None:
        v = (cfg or {}).get("no_think", DEFAULTS["no_think"])
    v = str(v).strip().lower()
    if v == "retry":
        return "retry"
    return "off" if v in ("0", "false", "off", "no", "", "none") else "always"


def no_think_on(cfg) -> bool:
    """the hint on EVERY claude_door request (no_think_mode "always")"""
    return no_think_mode(cfg) == "always"


def no_think_user(user: str) -> str:
    """the movement request with NO_THINK_HINT right after its answer instruction (measured placement)"""
    anchor = "Answer with the JSON object."
    if NO_THINK_HINT in user:
        return user
    if anchor in user:
        return user.replace(anchor, anchor + " " + NO_THINK_HINT, 1)
    return user.rstrip("\n") + "\n" + NO_THINK_HINT


def state_note(traits: dict, state) -> str:
    """The commanded character's current state as one line for the planner, '' when it is the default (standing at
    rest with nothing held or worn; the seated characters always: they have no other state). Part of the cache key."""
    if not isinstance(state, dict) or not traits.get("leaves_seat"):
        return ""
    bits = []
    posture = str(state.get("posture") or "").strip().lower()
    seated = state.get("seated")
    if isinstance(seated, str) and seated.strip():
        bits.append(f"it is sitting on the {seated.strip()}")
    elif seated is True or posture in ("seated", "sitting"):
        bits.append("it is sitting")
    elif posture and posture not in ("standing", "stand", "idle", "rest"):
        bits.append(f"it is {posture}")
    hold = state.get("holding")
    if isinstance(hold, dict):
        for side in ("right", "left"):
            v = hold.get(side)
            if v:
                bits.append(f"it holds the {v if isinstance(v, str) else (v.get('name') if isinstance(v, dict) else v)} "
                            f"in its {side} paw")
    elif isinstance(hold, (list, tuple)):
        for v in hold:
            if v:
                bits.append(f"it holds the {v if isinstance(v, str) else (v.get('name') if isinstance(v, dict) else v)}")
    elif isinstance(hold, str) and hold.strip():
        bits.append(f"it holds the {hold.strip()}")
    wear = state.get("wearing")
    for v in (wear if isinstance(wear, (list, tuple)) else [wear]):
        if v:
            bits.append(f"it wears the {v if isinstance(v, str) else (v.get('name') if isinstance(v, dict) else v)}")
    if not bits:
        return ""
    return "Right now " + "; ".join(bits)[:300] + "."


# ------------------------------------------------------------------------------------------------------ things
# A character with a props file (renderer/animals/<c>.props.json: the muse) whose page sent its things (state.things =
# motion.things(), the engine's live list) plans WITH things: the system prompt = <c>.txt (frozen, sha-checked) + the
# OBJECTS appendix (objects_appendix.txt, its catalogue generated from the props file: stable, so the provider's prompt
# cache keeps it), and the user message gets where each thing is right now (rounded so small moves keep the plan cache).
# Both are part of the cache key (the note) and the prompt sha (the appendix).
ROOT = os.path.dirname(HERE)
SCENE_DIR = os.path.join(ROOT, "renderer", "animals")
APPENDIX_FILE = os.path.join(PROMPT_DIR, "objects_appendix.txt")
# the verbs a "do" step may name (the engine's object verbs; unknown ones fail truthfully at the engine) + attach (local)
DO_VERBS = ["pick_up", "put_down", "put_back", "place", "put_on", "take_off", "hand_over", "open", "close", "pour", "use",
            "toss", "throw", "drop", "juggle", "balance", "spin", "stack", "show", "offer", "hug", "attach",
            "touch", "go_to"]   # a paw ON a thing (style tap|knock|pat|poke|press|stroke|lean, count), stand by it
GRIP_WORDS = {"power": "one paw", "pinch": "one paw, small pinch", "two_hand": "both paws"}
_SCENES: dict = {}


def scene_present(cfg):
    """a props file may flag a prop "removed": true (out of the scene; the renderer keeps its slot empty).
    The page, the matcher and the planners never hear of it."""
    if isinstance(cfg, dict) and isinstance(cfg.get("props"), list) and any(isinstance(p, dict) and p.get("removed") for p in cfg["props"]):
        cfg = dict(cfg, props=[p for p in cfg["props"] if not (isinstance(p, dict) and p.get("removed"))])
    return cfg


def load_scene(character: str, d: str | None = None) -> dict | None:
    """the character's props file (cached by mtime), or None (no scene)"""
    path = os.path.join(d or SCENE_DIR, f"{character}.props.json")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    c = _SCENES.get(path)
    if c and c[0] == mt:
        return c[1]
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("props"), list):
        return None
    cfg = scene_present(cfg)
    _SCENES[path] = (mt, cfg)
    return cfg


def _affords(p: dict) -> list:
    a = p.get("afford") if isinstance(p.get("afford"), dict) else {}
    out = []
    if p.get("furniture"):
        out.append("furniture (push, pull" + (", sit in" if p.get("kind") == "chair" else "") + ")")
    else:
        out.append(GRIP_WORDS.get(p.get("grip"), "one paw"))
    if a.get("wear") or p.get("wearable"):
        anc = (a.get("wear") or {}).get("anchor", "head") if isinstance(a.get("wear"), dict) else "head"
        out.append(f"wear on the {anc} (put_on / take_off)")
    if isinstance(a.get("surface"), dict):
        out.append("things can stand on it" + (" (stackable)" if a.get("stack") else ""))
    if isinstance(a.get("container"), dict):
        out.append("things go in it" + (" (open it first)" if a["container"].get("needs_open") else ""))
    if p.get("articulation") or a.get("articulation"):
        out.append("opens / closes")
    if isinstance(a.get("pour"), dict):
        out.append(f"pour ({a['pour'].get('what') or 'it'})")
    if isinstance(a.get("use"), dict):
        tg = a["use"].get("targets") or []
        out.append("use on " + (", ".join(map(str, tg)) if tg else "things"))
    if a.get("water_target"):
        out.append("can be watered")
    if p.get("throwable") and not p.get("furniture"):
        out.append("throwable")
    for k, w in (("bouncy", "bouncy"), ("spin", "spins well"), ("balance", "balances"), ("tiny", "tiny")):
        if a.get(k):
            out.append(w)
    return out


def scene_catalogue(cfg: dict) -> str:
    lines = []
    for p in cfg.get("props") or []:
        if not isinstance(p, dict) or not p.get("id"):
            continue
        lines.append(f"- {p['id']} = {p.get('name') or p['id']}: " + ", ".join(_affords(p)))
    pl = [str(q.get("id")) for q in cfg.get("places") or [] if isinstance(q, dict) and q.get("id")]
    if pl:
        lines.append("Places: " + ", ".join(pl) + "; body spots: head, right_foot, left_foot, back.")
    return "\n".join(lines)


def scene_appendix(cfg: dict, path: str = APPENDIX_FILE) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            tmpl = f.read()
    except OSError:
        return ""
    return tmpl.replace("{CATALOGUE}", scene_catalogue(cfg)).replace("{VERBS}", ", ".join(DO_VERBS)).strip()


def _q(v, step=0.05) -> float:
    try:
        x = round(float(v) / step) * step
    except (TypeError, ValueError):
        return 0.0
    return round(x, 2) + 0.0


def _side_word(h) -> str | None:
    """the engine's paw letter ('l' = screen-left = the character's OWN right) or a word -> 'right' / 'left' / 'both'"""
    h = str(h or "").strip().lower()
    if not h or h in ("none", "null", "false"):
        return None
    if h in ("both", "both paws"):
        return "both"
    if h in ("l",) or h.startswith("right"):
        return "right"
    if h in ("r",) or h.startswith("left"):
        return "left"
    return None


def things_note(things, root=None) -> str:
    """the live things list -> one line per thing for the user message ('' = none)"""
    if not isinstance(things, list):
        return ""
    rx = rz = 0.0
    if isinstance(root, dict):
        try:
            rx, rz = float(root.get("x") or 0), float(root.get("z") or 0)
        except (TypeError, ValueError):
            rx = rz = 0.0
    lines = []
    for t in things[:24]:
        if not isinstance(t, dict) or not t.get("id"):
            continue
        tid = str(t["id"])[:30]
        if "furniture" in (t.get("afford") or []):
            continue                                   # the room paragraph of the prompt places the furniture
        side = _side_word(t.get("held"))
        w = str(t.get("where") or "").strip()
        inside = t.get("inside") or t.get("in")
        if side:
            where = "held in both paws" if side == "both" else f"held in its {side} paw"
        elif t.get("worn"):
            where = w if " its " in f" {w} " else "worn on its head"
        elif " its " in f" {w} " or w == "air":
            where = "in the air" if w == "air" else w      # balanced on its head, on its left foot, ...
        else:
            on = t.get("on")
            where = (f"in the {inside}" if isinstance(inside, str) and inside else f"on the {on}" if on else
                     "on the floor" if w in ("", "floor", "home") else w if w.startswith(("on ", "in ")) else f"on the {w}")
            pos = t.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                dx, dz, y = _q(float(pos[0]) - rx), _q(float(pos[2]) - rz), _q(pos[1])
                lr = f"{abs(dx):g} H to its {'left' if dx > 0 else 'right'}" if abs(dx) >= 0.05 else "in line with it"
                fb = f"{abs(dz):g} H {'forward' if dz > 0 else 'back'}" if abs(dz) >= 0.05 else "level with it"
                where += f", {lr}, {fb}, {y:g} H up"
        lines.append(f"{tid}: {where}")
    return ("Things now: " + "; ".join(lines) + ".") if lines else ""


def scene_for(character: str, state, cfg_dir: str | None = None):
    """-> (props cfg, things list) when this plan is a plan WITH things, else (None, None)"""
    if not isinstance(state, dict) or not isinstance(state.get("things"), list) or not state["things"]:
        return None, None
    cfg = load_scene(character, cfg_dir)
    if cfg is None:
        return None, None
    return cfg, state["things"]


def names_thing(description: str, cfg: dict, things=None) -> bool:
    """does the request name one of the things (id, name word or alias)?"""
    low = " " + re.sub(r"[^a-z ]+", " ", str(description or "").lower().replace("_", " ")) + " "
    skip = {"one", "the", "little", "toy", "red", "blue", "green", "yellow", "pink", "silver", "black", "pot", "can"}
    for p in (cfg.get("props") or []):
        if not isinstance(p, dict) or p.get("furniture"):
            continue
        words = [str(p.get("id") or "").replace("_", " ")] + [str(a) for a in p.get("aliases") or []]
        words += [w for w in str(p.get("name") or "").split() if w not in skip]
        for w in words:
            w = w.strip().lower()
            if len(w) > 2 and w not in skip and (" " + w + " ") in low:
                return True
    return False


# ------------------------------------------------------------------------------------------------------ library
SPEED_WORDS = {"slow", "slowly", "slower", "quick", "quickly", "quicker", "fast", "faster", "rapid", "rapidly", "speedy"}
STYLE_WORDS = {"like", "pretend", "pretending", "imitate", "impression", "dramatic", "dramatically"}
LY_OK = {"only", "really", "fly", "belly", "jelly", "silly", "family", "early", "reply", "apply", "rely", "ugly", "holy",
         "lily", "bubbly", "curly", "wobbly", "lovely", "friendly", "jolly", "daily"}
PART_NEEDS = (("ear", "ears"), ("tail", "tail"), ("nose", "nose"), ("hand", "arms"), ("shoulder", "arms"))


def _num_ok(n: float, cmds: list) -> bool:
    for c in cmds:
        for k in ("count", "repeat", "cycles", "distance", "steps", "times"):
            v = c.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - n) < 1e-6:
                return True
        d = c.get("duration_ms")
        if isinstance(d, (int, float)) and abs(d / 1000.0 - n) < 0.01:
            return True
    return False


def library_check(character: str, description: str, matcher, traits: dict, act: bool):
    """-> (cmds | None, why). None = the planner's turn (the matcher misses or would lose part of the request)."""
    try:
        hits = matcher.parse(description, final=True, stable=True)
    except Exception as e:  # noqa: BLE001
        return None, f"matcher error {type(e).__name__}"
    if not hits:
        return None, "no match"
    try:
        if not matcher.understood(description, hits):
            return None, "not fully understood"
    except Exception as e:  # noqa: BLE001
        return None, f"understood error {type(e).__name__}"
    cmds = [h["cmd"] for h in hits]
    for c in cmds:
        a = c.get("action")
        if not act:
            # the older characters stay in their seat: their move commands are seated approximations -> the planner
            if a not in ("gesture", "pose", "expression", "look"):
                return None, f"whole-body {a} on a seated character"
            if "body" in (c.get("parts") or []) or any(str(k).startswith("body_") for k in (c.get("targets") or {})):
                return None, "whole-body gesture on a seated character"
        for p in (c.get("parts") or []):
            for pre, need in PART_NEEDS:
                if str(p).startswith(pre) and not traits.get(need, True):
                    return None, f"{character} has no {need}"
    norm = normalize(description).split()
    for w in norm:
        if re.fullmatch(r"\d+(\.\d+)?", w):
            n = float(w)
            if n != 1 and not _num_ok(n, cmds):
                return None, f"count {w} not kept"
    if SPEED_WORDS & set(norm) and not any(c.get("speed") or c.get("tempo") for c in cmds):
        return None, "speed not kept"
    styled = (STYLE_WORDS & set(norm)) or [w for w in norm if w.endswith("ly") and w not in LY_OK and w not in SPEED_WORDS]
    if styled and not any(c.get("style") for c in cmds):
        return None, "style not kept"
    return cmds, "ok"


# ------------------------------------------------------------------------------------------------------ messages
def kf_message(pid: str, ev: dict) -> dict:
    """Contract section 2 server -> page message for one planner event."""
    msg = {"type": "kf", "id": pid, "poses": [wire_pose(p) if "set" in p else p for p in ev.get("poses") or []],
           "done": bool(ev.get("done"))}
    if msg["done"]:
        msg.update({"ok": bool(ev.get("ok")), "source": ev.get("source") or "planner", "engine": ev.get("engine"),
                    "summary": ev.get("summary") or "", "ms_first_pose": ev.get("ms_first_pose"),
                    "ms_done": ev.get("ms_done"), "usd": round(float(ev.get("usd") or 0.0), 6)})
        if ev.get("error"):
            msg["error"] = str(ev["error"])[:300]
        if ev.get("sem_unavailable"):          # a semantic plan for a character without the semantic API
            msg["sem_unavailable"] = True
    return msg


def refusal(error: str, source: str = "planner") -> dict:
    return {"done": True, "ok": False, "source": source, "poses": [], "all_poses": [], "engine": None, "summary": "",
            "ms_first_pose": None, "ms_done": 0, "usd": 0.0, "error": error}


# ------------------------------------------------------------------------------------------------------ the planner
class PlanError(Exception):
    def __init__(self, msg: str, transient: bool = False):
        super().__init__(msg)
        self.transient = transient


def _summary_of(text: str) -> str:
    m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"')[:200]
    except ValueError:
        return m.group(1)[:200]


# ------------------------------------------------------------------------------------------------------ models / routes
def claude_cli() -> str | None:
    """the `claude` CLI the claude_door route would run (AVATAR_CLAUDE_BIN, else `claude` on PATH), or None"""
    import shutil
    b = os.environ.get("AVATAR_CLAUDE_BIN", "").strip()
    if b:
        return b if os.path.isfile(b) and os.access(b, os.X_OK) else shutil.which(b)
    return shutil.which("claude")


def auto_primary() -> str:
    """model "auto": the first route this machine can use -- the claude CLI on PATH (claude_door, no API key), else
    ANTHROPIC_API_KEY (Anthropic Messages API), else OPENROUTER_API_KEY (Gemini Flash Lite), else OPENAI_API_KEY (the
    voice key; a small OpenAI model), else GEMINI_API_KEY / GOOGLE_API_KEY (Gemini Flash Lite direct). With none of them: gem35lite, whose calls fail naming OPENROUTER_API_KEY."""
    if claude_cli():
        return "haiku55door"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "haiku55api"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "gem35lite"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt54mini"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gem35lite_gemini"
    return "gem35lite"


def auto_fallback(primary_preset) -> str | None:
    """fallback "auto": a second, independent route the machine can use, or None"""
    if primary_preset == "gem35lite":
        return "dsv41flash"
    order = [("haiku55api", ("ANTHROPIC_API_KEY",)), ("gem35lite", ("OPENROUTER_API_KEY",)),
             ("gpt54mini", ("OPENAI_API_KEY",)), ("gem35lite_gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))]
    for key, envs in order:
        if key != primary_preset and any(os.environ.get(e) for e in envs):
            return key
    return None


def _preset_of(model) -> tuple[str | None, dict | None]:
    """a PRESETS key, or a model id one preset uses -> (key, preset); else (None, None)"""
    m = str(model or "").strip()
    if m in PRESETS:
        return m, PRESETS[m]
    for k, p in PRESETS.items():
        if p["model"] == m:
            return k, p
    return None, None


def infer_route(model: str) -> str:
    """route 'auto' for a raw model id: z-ai/ deepseek/ -> openrouter_zdr, other "/" -> openrouter, "gemini-" -> gemini,
    "muse-" -> meta, "claude-" / haiku / sonnet / opus -> claude_door, else openai"""
    m = str(model or "").strip().lower()
    if m.startswith("claude-") or m in DOOR_ALIASES:
        return "claude_door"
    if m.startswith(ZDR_PREFIXES):
        return "openrouter_zdr"
    if "/" in m:
        return "openrouter"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("muse-"):
        return "meta"
    return "openai"


def _route_model(route: str, model: str) -> str:
    """the same model named for the route it goes to (google/<id> / openai/<id> on OpenRouter, <id> on its own API)"""
    if route == "gemini" and model.startswith("google/"):
        return model[len("google/"):]
    if route == "openai" and model.startswith("openai/"):
        return model[len("openai/"):]
    if route in ("claude_door", "anthropic"):      # anthropic/claude-haiku-5.5 -> claude-haiku-5-5
        m = model[len("anthropic/"):] if model.startswith("anthropic/") else model
        return DOOR_ALIASES.get(m, m.replace(".", "-"))
    if route.startswith("openrouter") and "/" not in model:
        if model.startswith("gemini-"):
            return "google/" + model
        if model.startswith("gpt-"):
            return "openai/" + model
    return model


def _spec(part: dict, zdr_cfg="auto") -> dict:
    """one model config {"model", "route"?, "effort"?} -> {"model", "route", "effort", "preset"}. OpenRouter becomes
    openrouter_zdr for GLM / DeepSeek always and for every model when cfg zdr is true."""
    model = str(part.get("model") or "").strip() or DEFAULTS["model"]
    if model == "auto":
        model = auto_primary()
    key, pre = _preset_of(model)
    route = str(part.get("route") or "auto").strip().lower()
    if route not in ROUTES:
        route = "auto"
    if route == "auto":
        route = pre["route"] if pre else infer_route(model)
    mid = _route_model(route, pre["model"] if pre else model)
    eff = part.get("effort", "auto")
    if eff == "auto":
        eff = pre.get("effort") if pre else None
    if route == "openrouter" and (mid.lower().startswith(ZDR_PREFIXES) or part.get("zdr", zdr_cfg) is True):
        route = "openrouter_zdr"
    return {"model": mid, "route": route, "effort": eff, "preset": key}


def _cfg_dict(cfg) -> dict:
    """a planner config (true / false / None / a partial dict / the merged Planner.cfg) over DEFAULTS"""
    return {**DEFAULTS, **cfg} if isinstance(cfg, dict) else DEFAULTS


def primary_spec(cfg) -> dict:
    """the primary model of a planner config (true / a dict / the merged Planner.cfg); env AVATAR_PLANNER_MODEL (a preset
    or a raw id, with that preset's / the inferred route and effort) overrides the file"""
    c = _cfg_dict(cfg)
    env = os.environ.get("AVATAR_PLANNER_MODEL", "").strip()
    return _spec({"model": env} if env else c, c.get("zdr", "auto"))


def fallback_spec(cfg) -> dict | None:
    """the fallback model of a planner config + "after_s", or None; env AVATAR_PLANNER_FALLBACK (preset / raw id / none)
    and AVATAR_PLANNER_FALLBACK_AFTER_S override the file"""
    c = _cfg_dict(cfg)
    fb = c.get("fallback")
    env = os.environ.get("AVATAR_PLANNER_FALLBACK", "").strip()
    if env:
        if env.lower() in ("none", "off", "0", "false", "null"):
            return None
        part = {"model": env, "after_s": fb.get("after_s") if isinstance(fb, dict) else None}
    elif isinstance(fb, str) and fb.strip():
        part = {"model": fb.strip()}
    elif isinstance(fb, dict) and fb.get("model"):
        part = fb
    else:
        return None
    if str(part.get("model")).strip() == "auto":
        auto = auto_fallback(primary_spec(c).get("preset"))
        if auto is None:
            return None
        part = {**part, "model": auto, "route": "auto", "effort": "auto"}
    s = _spec(part, c.get("zdr", "auto"))
    after = part.get("after_s")
    env_after = os.environ.get("AVATAR_PLANNER_FALLBACK_AFTER_S", "").strip()
    try:
        s["after_s"] = max(0.0, float(env_after if env_after else after if after is not None else FALLBACK_AFTER_S))
    except (TypeError, ValueError):
        s["after_s"] = FALLBACK_AFTER_S
    return s


def resolve(cfg) -> dict:
    """the primary model a planner config plans with: {"model", "route", "effort"}"""
    s = primary_spec(cfg)
    return {"model": s["model"], "route": s["route"], "effort": s["effort"]}


def resolve_fallback(cfg) -> dict | None:
    """the fallback a planner config hedges with: {"model", "route", "effort", "after_s"} | None"""
    s = fallback_spec(cfg)
    return None if s is None else {"model": s["model"], "route": s["route"], "effort": s["effort"], "after_s": s["after_s"]}


def resolve_quality(cfg) -> dict:
    """the model Planner.fill uses for offline library / cache filling (cfg "quality": a preset, an id or a
    {"model", "route", "effort"}; default the "quality" preset = Sol): {"model", "route", "effort"}"""
    q = _cfg_dict(cfg).get("quality") or "quality"
    s = _spec(q if isinstance(q, dict) else {"model": q}, _cfg_dict(cfg).get("zdr", "auto"))
    return {"model": s["model"], "route": s["route"], "effort": s["effort"]}


def cache_models(cfg) -> list[str]:
    """model ids whose cached plans are served besides the primary's / the fallback's (presets resolved to their ids)"""
    v = _cfg_dict(cfg).get("cache_models", DEFAULTS["cache_models"])
    if isinstance(v, str):
        v = [v]
    out = []
    for m in v if isinstance(v, (list, tuple)) else []:
        k, p = _preset_of(m)
        out.append(p["model"] if p else str(m))
    return out


def build_request(spec: dict, system: str, user: str, max_tokens: int = 12000) -> tuple[str, dict]:
    """(url, JSON body) of one streamed planner call on spec's route"""
    route, model, eff = spec["route"], spec["model"], spec.get("effort")
    if route == "openai":                         # Responses API SSE
        body = {"model": model, "instructions": system, "input": user, "stream": True,
                "text": {"format": {"type": "json_object"}}, "max_output_tokens": max_tokens}
        if eff:
            body["reasoning"] = {"effort": eff}
        return URL, body
    if route == "gemini":                         # native streamGenerateContent?alt=sse
        gc = {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}
        if eff:
            gc["thinkingConfig"] = {"thinkingLevel": eff}
        return GEMINI_URL.format(model=model), {"systemInstruction": {"parts": [{"text": system}]},
                                                "contents": [{"role": "user", "parts": [{"text": user}]}],
                                                "generationConfig": gc}
    if route in ("openrouter", "openrouter_zdr", "meta"):    # chat completions SSE
        body = {"model": model, "stream": True, "messages": [{"role": "system", "content": system},
                                                             {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_tokens": max_tokens}
        if route == "meta":
            if eff:
                body["reasoning_effort"] = eff
            body["stream_options"] = {"include_usage": True}
            return META_URL, body
        if eff:
            body["reasoning"] = {"effort": eff}
        body["usage"] = {"include": True}
        if route == "openrouter_zdr":             # zero-retention providers only
            body["provider"] = {"zdr": True, "data_collection": "deny"}
        return OPENROUTER_URL, body
    if route == "anthropic":                      # Messages API SSE
        body = {"model": model, "max_tokens": max_tokens, "stream": True,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}]}
        if eff == "none":
            body["thinking"] = {"type": "disabled"}
            body["output_config"] = {"effort": "low"}
        elif eff:
            body["output_config"] = {"effort": eff}
        return ANTHROPIC_URL, body
    raise PlanError(f"unknown route {route!r}")


SSE_DONE = object()


def sse_split(buf: bytes) -> tuple[list, bytes]:
    """bytes received so far -> ([parsed JSON event | SSE_DONE, ...] of its complete `data:` lines, the unfinished rest);
    comment / event / blank lines and unparsable data are skipped"""
    out = []
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            out.append(SSE_DONE)
            continue
        try:
            out.append(json.loads(data))
        except ValueError:
            continue
    return out, buf


def _err_transient(code) -> bool:
    try:
        c = int(code)
        return c == 429 or c >= 500
    except (TypeError, ValueError):
        return str(code or "").lower() in ("server_error", "rate_limit_exceeded", "resource_exhausted", "unavailable",
                                           "internal", "overloaded_error")


def event_openai(e: dict, info: dict) -> tuple[list[str], bool]:
    """one Responses API SSE event -> (text deltas, finished); usage into info; error events raise PlanError"""
    ty = e.get("type")
    if ty == "response.output_text.delta":
        d = e.get("delta") or ""
        return ([d] if d else []), False
    if ty == "response.completed":
        info["usage"] = (e.get("response") or {}).get("usage") or {}
        return [], True
    if ty in ("error", "response.failed", "response.incomplete"):
        err = json.dumps(e)[:300]
        info["usage"] = ((e.get("response") or {}).get("usage")) or info.get("usage")
        raise PlanError(f"{ty}: {err}", transient=ty == "error" and "server_error" in err)
    return [], False


def event_gemini(e: dict, info: dict) -> tuple[list[str], bool]:
    """one streamGenerateContent SSE event -> (text parts, thoughts skipped; False); usageMetadata into info"""
    if isinstance(e.get("error"), dict):
        er = e["error"]
        raise PlanError(f"gemini error: {json.dumps(er)[:300]}", transient=_err_transient(er.get("code"))
                        or _err_transient(er.get("status")))
    out = []
    for c in e.get("candidates") or []:
        for p in (c.get("content") or {}).get("parts") or []:
            if not p.get("thought") and p.get("text"):
                out.append(p["text"])
    if e.get("usageMetadata"):
        info["usage"] = e["usageMetadata"]
    return out, False


def event_chat(e: dict, info: dict) -> tuple[list[str], bool]:
    """one chat-completions SSE event (OpenRouter / Meta) -> (content deltas, False); usage (OpenRouter: + cost) into
    info; an error event (OpenRouter's mid-stream {"error": ...} / finish_reason "error") raises PlanError"""
    if e.get("usage"):
        info["usage"] = e["usage"]
    if e.get("error"):
        er = e["error"] if isinstance(e["error"], dict) else {"message": str(e["error"])}
        raise PlanError(f"stream error: {json.dumps(er)[:300]}", transient=_err_transient(er.get("code")))
    out = []
    for c in e.get("choices") or []:
        d = (c.get("delta") or {}).get("content")
        if d:
            out.append(d)
        if c.get("finish_reason") == "error":
            raise PlanError("stream error: finish_reason error", transient=True)
    return out, False


def event_anthropic(e: dict, info: dict) -> tuple[list[str], bool]:
    """one Messages API SSE event -> (text deltas, finished); thinking deltas skipped; usage (message_start, then the
    cumulative output_tokens of message_delta) into info; an error event raises PlanError"""
    ty = e.get("type")
    if ty == "content_block_delta":
        d = e.get("delta") or {}
        return ([d["text"]] if d.get("type") == "text_delta" and d.get("text") else []), False
    if ty == "message_start":
        info["usage"] = dict(((e.get("message") or {}).get("usage")) or {}, anthropic=True)
        return [], False
    if ty == "message_delta":
        u = info.setdefault("usage", {"anthropic": True})
        u.update({k: v for k, v in (e.get("usage") or {}).items() if v is not None})
        if (e.get("delta") or {}).get("stop_reason") == "refusal":
            raise PlanError("anthropic: refusal")
        return [], False
    if ty == "message_stop":
        return [], True
    if ty == "error":
        er = e.get("error") or {}
        raise PlanError(f"anthropic error: {json.dumps(er)[:300]}",
                        transient=er.get("type") in ("overloaded_error", "api_error", "rate_limit_error"))
    return [], False


ROUTE_EVENTS = {"openai": event_openai, "gemini": event_gemini, "openrouter": event_chat, "openrouter_zdr": event_chat,
                "meta": event_chat, "anthropic": event_anthropic}


def parse_sse(route: str, chunks, info: dict | None = None) -> tuple[list[str], dict, bool]:
    """a whole recorded byte stream of a route -> (text deltas, info, finished) — the stream loop without the network"""
    info = {} if info is None else info
    handle = ROUTE_EVENTS[route]
    texts, buf = [], b""
    for ch in list(chunks) + [b"\n"]:
        evs, buf = sse_split(buf + ch)
        for e in evs:
            if e is SSE_DONE:
                return texts, info, True
            t, fin = handle(e, info)
            texts += t
            if fin:
                return texts, info, True
    return texts, info, False


_KEYISH = re.compile(r"(sk-[A-Za-z0-9_\-]{4,}|AIza[0-9A-Za-z_\-]{8,}|Bearer\s+\S+)")


def _redact(s: str) -> str:
    return _KEYISH.sub("[key]", s)


def price_of(model: str, route: str | None = None) -> tuple[float, float, float]:
    p = PRICE.get(model)
    if p is None:
        p = PRICE_UNKNOWN.get(route or infer_route(model), (1.0, 1.0, 5.0))
    return p


def usd_of(model: str, usage: dict | None, system: str, user: str, text: str, route: str | None = None,
           cancelled: bool = False) -> tuple[float, bool]:
    """(usd, estimated) of one call. OpenRouter's usage.cost wins; else the usage of its route (Responses / chat /
    Gemini) at PRICE; no usage (aborted, timed out or cancelled stream): estimated from the characters, uncached.
    The claude_door route (the `claude` CLI) -> 0 (its API-equivalent cost is recorded apart, "door_usd")."""
    if route == "claude_door":
        return 0.0, False
    pi, pc, po = price_of(model, route)
    u = usage or {}
    if u.get("cost") is not None:
        try:
            return float(u["cost"]), False
        except (TypeError, ValueError):
            pass
    if u.get("anthropic"):                                      # Messages API: input_tokens excludes cache reads
        cr = u.get("cache_read_input_tokens", 0) or 0
        cw = u.get("cache_creation_input_tokens", 0) or 0
        return (((u.get("input_tokens", 0) or 0) + 1.25 * cw) * pi + cr * pc
                + (u.get("output_tokens", 0) or 0) * po) / 1e6, False
    if u.get("input_tokens") is not None:                       # Responses API
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        return ((u.get("input_tokens", 0) - cached) * pi + cached * pc + (u.get("output_tokens", 0) or 0) * po) / 1e6, False
    if u.get("prompt_tokens") is not None:                      # chat completions (completion includes reasoning)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        return ((u["prompt_tokens"] - cached) * pi + cached * pc + (u.get("completion_tokens", 0) or 0) * po) / 1e6, False
    if u.get("promptTokenCount") is not None:                   # Gemini
        cached = u.get("cachedContentTokenCount", 0) or 0
        out = (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)
        return ((u["promptTokenCount"] - cached) * pi + cached * pc + out * po) / 1e6, False
    # no usage event: the tokens are billed anyway — estimate, uncached
    tin = (len(system) + len(user)) / 3.3
    tout = len(text) / 2.0 + 50
    return (tin * pi + tout * po) / 1e6, True


# ------------------------------------------------------------------------------------------------ route claude_door
_DOORS: dict = {}


DOOR_NO_THINK = "none"      # planner effort "none" on the door = --effort low + no thinking (MAX_THINKING_TOKENS=0)


def door_for(effort=None):
    """the claude_door instance a planner call uses: the process-wide `claude_door.door` when its effort is the wanted
    one (or the door takes a per-call effort); else one private ClaudeDoor per effort (its own warm pool). effort "none"
    = the CLI's lowest effort with thinking off. Raises PlanError when app/claude_door.py is missing (route unavailable)."""
    try:
        import claude_door                         # app/claude_door.py
    except ImportError as e:
        raise PlanError(f"claude_door unavailable: {e}") from None
    base = claude_door.door
    eff = effort or None
    if eff != DOOR_NO_THINK and (eff is None or eff == getattr(base, "effort", None) or _door_takes_effort(base)):
        return base
    d = _DOORS.get(eff)
    if d is None:
        d = _DOORS[eff] = claude_door.make_door()
        if eff == DOOR_NO_THINK:
            d.effort, d.thinking = "low", False
        else:
            d.effort = eff
    return d


async def close_doors() -> None:
    """close the private per-effort doors (tools / tests; the server's process-wide door is closed by its owner)"""
    for d in list(_DOORS.values()):
        try:
            await d.close()
        except Exception:  # noqa: BLE001
            pass
    _DOORS.clear()


def _door_takes_effort(d) -> bool:
    import inspect
    try:
        return "effort" in inspect.signature(d.stream).parameters
    except (TypeError, ValueError):
        return False


def sem_door(effort=None):
    """the semantic planner's OWN claude_door instance per effort (its own warm pool and MAX_TOTAL budget). Shared
    with the keyframe planner's door (same model + effort "none"), the keyframe prompts' warm processes (2 per
    character, warmed at every session open) filled the budget and the semantic prewarm spawned nothing."""
    try:
        import claude_door
    except ImportError as e:
        raise PlanError(f"claude_door unavailable: {e}") from None
    eff = effort or None
    k = ("semantic", eff)
    d = _DOORS.get(k)
    if d is None:
        d = _DOORS[k] = claude_door.ClaudeDoor()
        if eff == DOOR_NO_THINK:
            d.effort, d.thinking = "low", False
        elif eff is not None:
            d.effort = eff
    return d


async def door_stream(spec: dict, system: str, user: str, deadline: float, info: dict, tag: str = "planner",
                      door=None):
    """one planner call through the door: yields text deltas; the door's done event -> info["door"] (ms_first, ms_done,
    warm, API-equivalent cost) and info["usage"]; a failed answer raises PlanError (a timeout is transient). Closing this
    generator (a cancel, the plan's timeout) closes the door's stream, which kills its process."""
    d = door if door is not None else door_for(spec.get("effort"))
    left = deadline - time.monotonic()
    if left <= 0:
        raise PlanError("timeout", transient=True)
    kw = {"model": spec["model"], "system": system, "user": user, "max_s": max(0.5, left), "tag": tag}
    if spec.get("effort") and spec["effort"] != DOOR_NO_THINK and _door_takes_effort(d):
        kw["effort"] = spec["effort"]
    agen = d.stream(**kw)
    try:
        async for ev in agen:
            if ev.get("delta"):
                yield ev["delta"]
            elif ev.get("thinking"):                # a hidden thinking block opened -> the text is late
                info["thinking_ms"] = ev.get("ms")
                cb = info.get("on_thinking")
                if cb is not None:
                    cb()
            elif ev.get("done"):
                info["door"] = {k: ev.get(k) for k in ("ms_first", "ms_done", "warm", "cost_usd", "error", "ms_thinking")}
                if ev.get("usage"):
                    info["usage"] = {"door": True, **{k: v for k, v in ev["usage"].items()
                                                       if isinstance(v, (int, float))}}
                if not ev.get("ok"):
                    err = str(ev.get("error") or "door error")
                    raise PlanError(f"claude door: {err[:300]}", transient=err.startswith("timeout"))
                return
    finally:
        await agen.aclose()


def _start_task(coro) -> asyncio.Task:
    """a task that runs eagerly up to its first suspension (Python >= 3.12), so the HTTP request leaves inside the
    caller's step instead of one event-loop turn later"""
    try:
        return asyncio.Task(coro, loop=asyncio.get_running_loop(), eager_start=True)
    except TypeError:                                  # Python < 3.12: no eager start
        return asyncio.ensure_future(coro)


class _Attempt:
    """one model call of a plan (primary / retry / fallback): its own stream, parser and bill"""
    __slots__ = ("role", "spec", "info", "parser", "text", "err", "transient", "ms_start", "ms_end", "ms_first_pose",
                 "task", "usd", "estimated", "billed", "ended", "cancelled", "t_start", "stalls", "last_pose_ms",
                 "last_pose_t")

    def __init__(self, role: str, spec: dict, controls, ms_start: int, allow_do: bool = False):
        self.role, self.spec = role, spec
        self.info: dict = {"spec": spec}
        self.parser = StreamParser(controls, allow_do)
        self.text = ""
        self.err = None
        self.transient = False
        self.ms_start, self.ms_end, self.ms_first_pose = ms_start, None, None
        self.task = None
        self.usd, self.estimated, self.billed, self.ended, self.cancelled = 0.0, False, False, False, False
        self.t_start = time.monotonic()
        self.stalls: list[dict] = []
        self.last_pose_ms, self.last_pose_t = None, 0.0

    def stall_check(self, at_ms: int, stall_s: float, motion_s: float) -> None:
        """a gap since the last pose (array still open) longer than stall_s with < motion_s of motion parsed"""
        if self.last_pose_ms is not None and at_ms - self.last_pose_ms > stall_s * 1000 and self.last_pose_t < motion_s:
            self.stalls.append({"ms": self.last_pose_ms, "gap_ms": at_ms - self.last_pose_ms,
                                "motion_s": round(self.last_pose_t, 3)})

    def failure(self) -> str:
        return self.err or ("no poses" if self.parser.found_poses_key else "no poses array in the answer")

    def record(self) -> dict:
        return {"model": self.spec["model"], "route": self.spec["route"], "effort": self.spec.get("effort"),
                "role": self.role, "ms_start": self.ms_start, "ms_end": self.ms_end, "ms_first_pose": self.ms_first_pose,
                "error": self.err, "usd": round(self.usd, 6), "estimated": self.estimated, "cancelled": self.cancelled,
                "usage": self.info.get("usage") or None, "chars": len(self.text), "stalls": len(self.stalls),
                **({"door": self.info["door"], "door_usd": (self.info["door"] or {}).get("cost_usd")}
                   if self.info.get("door") else {})}


# --------------------------------------------------------------------------------------- semantic-API planner
# {type:'plan', ..., planner:'semantic'} (docs/SEMANTIC.md section 3): library (unchanged) -> semantic cache
# (AVATAR_HOME/cache/semplans, its own namespace) -> Claude writes a short program in the semantic motion API
# (app/semantic, the core package), streamed into semantic.Stream OFF the event loop; every complete top-level
# statement becomes a batch of dense rig-space poses -> {type:'kf', space:'rig'} + {type:'sem'} info messages.
# Claude models only: the `claude` CLI door (no key) or the Anthropic Messages API (ANTHROPIC_API_KEY); route "auto" =
# the CLI when it is on PATH, else the API key. No other provider and no fallback to the keyframe planner; one retry on
# the same model when it fails before its first pose with >= retry_min_s left. Config planner.semantic over SEM_DEFAULTS
# (env AVATAR_SEM_MODEL / AVATAR_SEM_EFFORT / AVATAR_SEM_ROUTE override it).
# Defaults by measurement (24 fresh commands per run through the door): Sonnet 5.5 with thinking
# off is as fast to the first pose as Haiku (its first token is earlier; Haiku writes faster) and clearly better (blind
# Sonnet judge 3.9-4.0 vs 3.1-3.4 of 5; 0 interpreter errors vs 1-2 per 24); thinking "low" is slower and no better.
# SEM_SUFFIX (the first line a single body call, not a `with` block) cuts first token -> first pose from 762 to 399 ms
# median with the same judge score. A hedge (2nd identical call) gained nothing: the two calls' first tokens move together.
SEM_SUFFIX = """## Streaming (important)
The avatar starts moving the moment your FIRST statement is complete, so put the opening move first and keep the
first statement short. If the action is ONE movement, open with that single call on its own line (the reach, raise,
turn, lean or arm move that begins it — never hold(), wait() or a face-only change). If things happen AT THE SAME
TIME (point and look, wave and smile, a move with its expression), open with ONE `with parallel():` block of 2-4
lines with the main body move as its first line — never split a simultaneous ask into a solo call followed by a
block. No comments, no blank lines."""
SEM_DEFAULTS = {"model": "claude-sonnet-5-5", "effort": "none", "route": "auto", "hedge": None, "timeout_s": 20.0, "retry_min_s": 8.0,
                "cache": True, "hz": 20, "suffix": "auto", "warm_n": 2, "prime": True, "prime_ttl_s": 240.0}
SEM_PRIME_USER = 'Warm-up call, not a command: reply with exactly\n```python\nhold(0.1)\n```'
_SEM_LAST_USE: dict = {}       # (model, effort, sha(system)) -> monotonic time of the last door request with that prompt
SEM_TAG = "semantic"
_SEM_STATE: dict = {"mod": None, "tried": False, "err": None}
_SEM_POOL = None
_SEM_END = object()


def sem_module():
    """the core package `semantic` (app/semantic), imported once; None when it is not installed (or fails to import)"""
    if not _SEM_STATE["tried"]:
        try:
            import semantic                      # app/semantic
            _SEM_STATE["mod"] = semantic
        except Exception as e:  # noqa: BLE001
            _SEM_STATE["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        _SEM_STATE["tried"] = True
    return _SEM_STATE["mod"]


def set_sem_module(mod) -> None:
    """tests: use this module as `semantic` (None = not installed)"""
    _SEM_STATE.update({"mod": mod, "tried": True, "err": None if mod is not None else "not installed"})


def _sem_pool():
    global _SEM_POOL
    if _SEM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _SEM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="semantic")
    return _SEM_POOL


async def _off(fn, *a):
    """run fn(*a) on the semantic thread pool (never on the event loop)"""
    return await asyncio.get_running_loop().run_in_executor(_sem_pool(), fn, *a)


def sem_code(text: str) -> str:
    """the program text so far: the fenced python block's body (open or closed); the raw text when no fence yet"""
    t = text or ""
    i = t.find("```")
    if i < 0:
        return t.strip()
    j = t.find("\n", i)
    if j < 0:
        return ""
    body = t[j + 1:]
    k = body.find("```")
    return (body[:k] if k >= 0 else body).rstrip()


_SEM_STR = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""")


def _sem_names_used(program: str) -> set:
    return set(_SEM_STR.findall(program or ""))


def _xyz(v) -> list | None:
    try:
        return [round(float(v[0]), 1), round(float(v[1]), 1), round(float(v[2]), 1)]
    except (TypeError, ValueError, IndexError):
        return None


def sem_targets_wire(targets) -> dict:
    return {str(k): x for k, v in (targets or {}).items() if (x := _xyz(v)) is not None}


def sem_warm_key(spec: dict, system: str) -> tuple:
    """what a warm door process is keyed by (model + the door instance's effort) + the exact prompt bytes"""
    return (spec["model"], spec.get("effort"), hashlib.sha256(system.encode("utf-8")).hexdigest())


def sem_message(pid: str, ev: dict) -> dict:
    """a {type:'sem'} info message (contract section 3)"""
    msg = {"type": "sem", "id": pid}
    for k in ("phase", "program", "source", "model", "cached", "ms", "exec_ms", "warnings", "errors", "targets"):
        if k in ev:
            msg[k] = ev[k]
    for k in ("n_statements", "error", "duration_s"):
        if ev.get(k) is not None:
            msg[k] = ev[k]
    return msg


def sem_kf_message(pid: str, ev: dict) -> dict:
    """the rig-space kf message of a semantic plan event (library hits: today's kf message + planner:'semantic')"""
    if ev.get("space") != "rig":
        msg = kf_message(pid, ev)
        msg["planner"] = "semantic"
        return msg
    msg = {"type": "kf", "id": pid, "space": "rig", "hz": ev.get("hz"), "poses": ev.get("poses") or [],
           "done": bool(ev.get("done"))}
    if msg["done"]:
        msg.update({"ok": bool(ev.get("ok")), "source": ev.get("source") or "planner", "planner": "semantic",
                    "engine": ev.get("engine"), "summary": ev.get("summary") or "", "ms_first_pose": ev.get("ms_first_pose"),
                    "ms_done": ev.get("ms_done"), "usd": round(float(ev.get("usd") or 0.0), 6)})
        if ev.get("error"):
            msg["error"] = str(ev["error"])[:300]
        if ev.get("keep_pose"):
            msg["keep_pose"] = True
        if ev.get("channels"):
            msg["channels"] = list(ev["channels"])
    return msg


async def _sem_door_stream(spec, system, user, deadline, info, tag):
    async for d in door_stream(spec, system, user, deadline, info, tag, door=sem_door(spec.get("effort"))):
        yield d


class _SemAttempt:
    """one door call of a semantic plan: its stream, its semantic.Stream (fed off the loop), its timings"""

    def __init__(self, role: str, spec: dict, ms_start: int):
        self.role, self.spec = role, spec
        self.info: dict = {"spec": spec}
        self.text = ""
        self.stream = None
        self.report: dict = {}
        self.err, self.transient = None, False
        self.ms_start, self.ms_end, self.ms_door_done = ms_start, None, None
        self.ms_first_token = self.ms_first_stmt = None
        self.n_poses, self.n_batches = 0, 0
        self.task = None
        self.ended = self.cancelled = False
        self.t_start = time.monotonic()

    def failure(self) -> str:
        if self.err:
            return self.err
        errs = self.report.get("errors") or []
        if errs:
            e = errs[-1]
            return "no valid statements: " + str(e.get("error") if isinstance(e, dict) else e)[:200]
        return "no valid statements" if self.text else "empty answer"

    def record(self) -> dict:
        return {"model": self.spec["model"], "route": self.spec["route"], "effort": self.spec.get("effort"),
                "role": self.role, "ms_start": self.ms_start, "ms_first_token": self.ms_first_token,
                "ms_first_stmt": self.ms_first_stmt, "ms_door_done": self.ms_door_done, "ms_end": self.ms_end,
                "error": self.err, "cancelled": self.cancelled, "chars": len(self.text), "poses": self.n_poses,
                "usage": self.info.get("usage") or None,
                **({"door": self.info["door"]} if self.info.get("door") else {})}


class _SemanticPlanner:
    """Planner's semantic-API path (mixed into Planner; uses its cfg / library / log)"""

    sem_stream_fn = None                          # tests: fn(spec, system, user, deadline, info) yielding text

    def sem_cfg(self) -> dict:
        c = self.cfg.get("semantic")
        return {**SEM_DEFAULTS, **c} if isinstance(c, dict) else dict(SEM_DEFAULTS)

    @staticmethod
    def sem_route(route=None) -> str:
        """the semantic planner's route: "claude_door" (the `claude` CLI) or "anthropic" (ANTHROPIC_API_KEY); "auto" =
        the CLI when it is on PATH, else the API (whose calls then name the missing key)"""
        r = str(os.environ.get("AVATAR_SEM_ROUTE", "").strip() or route or "auto").lower()
        if r in ("claude_door", "anthropic"):
            return r
        return "claude_door" if claude_cli() else "anthropic"

    @staticmethod
    def _sem_spec(part, route=None) -> dict | None:
        """{"model", "effort"} -> a spec on the semantic route; None when the model is not a Claude model"""
        if isinstance(part, str):
            part = {"model": part}
        if not isinstance(part, dict) or not part.get("model"):
            return None
        s = _spec({"model": part["model"], "route": _SemanticPlanner.sem_route(part.get("route") or route),
                   "effort": part.get("effort", "none")})
        if not str(s["model"]).startswith("claude-"):
            return None
        if part.get("after_s") is not None:
            try:
                s["after_s"] = max(0.0, float(part["after_s"]))
            except (TypeError, ValueError):
                s["after_s"] = 1.5
        return s

    def sem_specs(self) -> tuple[dict | None, dict | None]:
        c = self.sem_cfg()
        env = os.environ.get("AVATAR_SEM_MODEL", "").strip()
        p = self._sem_spec({"model": env or c["model"], "effort": os.environ.get("AVATAR_SEM_EFFORT", "").strip()
                            or c.get("effort", "none")}, c.get("route"))
        h = self._sem_spec(c.get("hedge"), c.get("route")) if c.get("hedge") else None
        if h is not None and "after_s" not in h:
            h["after_s"] = 1.5
        return p, h

    def _sem_system(self, sem, character: str, targets) -> str:
        names = tuple(sorted(str(k) for k in (targets or {})))
        suffix = self.sem_cfg().get("suffix")
        suffix = SEM_SUFFIX if suffix == "auto" else (suffix or "")      # "auto" = the measured latency note; null = none
        memo = self.__dict__.setdefault("_sem_sys_memo", {})
        k = (character, names, suffix)
        if k not in memo:
            s = sem.system_prompt(character, list(names))
            memo[k] = (s.rstrip("\n") + "\n\n" + suffix.strip() + "\n") if suffix.strip() else s
        return memo[k]

    def _sem_stats(self) -> dict:
        return self.__dict__.setdefault("sem_stats", {"plans": 0, "library": 0, "cache": 0, "planner": 0, "failed": 0,
                                                      "retries": 0, "hedged": 0, "unavailable": 0})

    # -- warm the door for the semantic system prompt (the page's {type:'semwarm', character})
    def prewarm_semantic(self, character: str, n: int | None = None) -> list[dict]:
        """-> the specs warmed ([] = semantic not installed / character without a retarget / planner off). Call after
        sem_module() has been imported off the loop (prewarm_semantic_async does both)."""
        sem = _SEM_STATE["mod"] if _SEM_STATE["tried"] else sem_module()
        if not self.enabled or sem is None or character not in getattr(sem, "CHARACTERS", ()):
            return []
        try:
            system = self._sem_system(sem, character, sem.resolve_targets(character, None))
        except Exception as e:  # noqa: BLE001
            self.log(f"planner: semantic prewarm prompt failed: {type(e).__name__}: {str(e)[:120]}")
            return []
        if n is None:
            n = self.sem_cfg().get("warm_n")
            n = 2 if n is None else int(n)
        out = []
        for sp in self.sem_specs():
            if sp:
                try:
                    if sp["route"] == "claude_door":     # the API route has no processes to warm
                        sem_door(sp.get("effort")).prewarm(sp["model"], system, n)
                    out.append(sp)
                except Exception as e:  # noqa: BLE001
                    self.log(f"planner: semantic door prewarm failed: {type(e).__name__}: {str(e)[:120]}")
        return out

    async def prewarm_semantic_async(self, character: str, n: int | None = None, prime: bool | None = None) -> list[dict]:
        """warm the door's processes for the semantic prompt AND (cfg prime, default on) prime the API's prompt cache with
        one tiny request on that exact (model, system) when it has not been used for prime_ttl_s (the provider's prompt
        cache lapses after ~5 min idle: the first real plan then pays the cache write)"""
        if not _SEM_STATE["tried"]:
            await _off(sem_module)
        specs = self.prewarm_semantic(character, n)
        c = self.sem_cfg()
        # the prime is one tiny request through the CLI door; on the API route it would be a paid call at page load,
        # so there semwarm only reports availability (the prompt cache fills with the first real plan)
        if specs and specs[0]["route"] == "claude_door" and (c.get("prime", True) if prime is None else prime):
            sem = _SEM_STATE["mod"]
            try:
                system = self._sem_system(sem, character, sem.resolve_targets(character, None))
            except Exception:  # noqa: BLE001
                return specs
            sp = specs[0]
            k = sem_warm_key(sp, system)
            if time.monotonic() - _SEM_LAST_USE.get(k, -1e9) >= float(c.get("prime_ttl_s") or 240.0):
                _SEM_LAST_USE[k] = time.monotonic()
                t = asyncio.ensure_future(self._sem_prime(sp, system))
                self.__dict__.setdefault("_sem_prime_tasks", set()).add(t)
                t.add_done_callback(self._sem_prime_tasks.discard)
        return specs

    async def _sem_prime(self, sp: dict, system: str) -> None:
        info: dict = {"spec": sp}
        fn = self.sem_stream_fn or _sem_door_stream
        try:
            async for _ in fn(sp, system, SEM_PRIME_USER, time.monotonic() + 20.0, info, SEM_TAG + "-prime"):
                pass
        except Exception as e:  # noqa: BLE001
            self.log(f"planner: semantic prime failed: {type(e).__name__}: {str(e)[:120]}")
        self.__dict__.setdefault("sem_primes", []).append({"model": sp["model"], "door": info.get("door"),
                                                           "usage": info.get("usage")})

    # -- cache (own namespace: AVATAR_HOME/cache/semplans)
    def _sem_cache_path(self, key: str) -> str | None:
        d = getattr(self, "sem_cache_dir", None)
        if d is None and self.cache_dir:
            d = os.path.join(os.path.dirname(os.path.abspath(self.cache_dir)), "semplans")
        return os.path.join(d, key + ".json") if d else None

    def sem_cache_get(self, key: str, prompt_sha: str, models) -> dict | None:
        p = self._sem_cache_path(key)
        if not p or not self.sem_cfg().get("cache", True):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                ent = json.load(f)
        except (OSError, ValueError):
            return None
        if (ent.get("prompt_sha") != prompt_sha or ent.get("model") not in set(models) or not ent.get("poses")
                or not ent.get("program")):
            return None
        return ent

    def sem_cache_put(self, key: str, ent: dict) -> None:
        p = self._sem_cache_path(key)
        if not p or not self.sem_cfg().get("cache", True):
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ent, f, separators=(",", ":"))
            os.replace(tmp, p)
        except OSError as e:
            self.log(f"planner: semantic cache write failed: {type(e).__name__}")

    @staticmethod
    def _sem_live_moved(ent: dict, targets: dict, tol_cm: float = 2.0) -> list[str]:
        """the targets the cached program names whose position now differs from when it was planned"""
        used = _sem_names_used(ent.get("program") or "")
        old = ent.get("targets") or {}
        moved = []
        for name in sorted(used & set(targets or {})):
            a, b = _xyz(targets[name]), old.get(name)
            if a is None:
                continue
            if not isinstance(b, list) or len(b) < 3 or max(abs(a[i] - float(b[i])) for i in range(3)) > tol_cm:
                moved.append(name)
        return moved

    def _sem_library(self, character: str, description: str, state, words):
        """the library step exactly as Planner.plan runs it (things naming, his own words second)"""
        scfg, _things = scene_for(character, state, self.scene_dir)
        named = scfg is not None and names_thing(description, scfg)

        def lib(text):
            c, w = self.library(character, text)
            if c and named and not any(isinstance(x, dict) and x.get("action") == "object" for x in c):
                return None, "names a thing: planned with it"
            return c, w
        cmds, why = lib(description)
        if not cmds and words and str(words).strip():
            cw, why_w = lib(str(words).strip()[:400])
            if cw:
                cmds, why = cw, "his words: " + why_w
        return cmds, why

    # -- one attempt: door text -> Stream.feed (off loop) -> ("batches", a, [batch, ...], report) / ("end", a, None, None)
    async def _sem_attempt(self, a: _SemAttempt, sem, character, targets, hz, system, user, deadline, q, ms) -> None:
        feed_q: asyncio.Queue = asyncio.Queue()

        def emit(res):
            bs, rep = res
            a.report = rep or a.report
            bs = [b for b in (bs or []) if isinstance(b, dict) and b.get("poses")]
            if bs:
                if a.ms_first_stmt is None:
                    a.ms_first_stmt = ms()
                a.n_batches += len(bs)
                a.n_poses += sum(len(b["poses"]) for b in bs)
            q.put_nowait(("batches", a, bs, a.report))

        def fed(text):
            return a.stream.feed(text), a.stream.report()

        def finished():
            return a.stream.finish(), a.stream.report()

        async def feeder():
            a.stream = await _off(lambda: sem.Stream(character, targets=targets, seed=0, hz=hz, t_origin=0.0))
            while True:
                parts = [await feed_q.get()]
                while not feed_q.empty():
                    parts.append(feed_q.get_nowait())
                end = parts[-1] is _SEM_END
                text = "".join(p for p in parts if p is not _SEM_END)
                if text:
                    emit(await _off(fed, text))
                if end:
                    emit(await _off(finished))
                    return

        ft = asyncio.ensure_future(feeder())
        fn = self.sem_stream_fn or self._sem_stream(getattr(a, "http", None))
        try:
            try:
                async for d in fn(a.spec, system, user, deadline, a.info, SEM_TAG):
                    if not d:
                        continue
                    if a.ms_first_token is None:
                        a.ms_first_token = ms()
                    a.text += d
                    feed_q.put_nowait(d)
            except PlanError as e:
                a.err, a.transient = str(e), e.transient
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                a.err = f"{type(e).__name__}: {str(e)[:200]}"
            a.ms_door_done = ms()
            feed_q.put_nowait(_SEM_END)          # finish even after a door error: the complete statements + the tail
            try:
                await ft
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                a.err = a.err or f"semantic {type(e).__name__}: {str(e)[:200]}"
        finally:
            if not ft.done():
                ft.cancel()
        a.ms_end = ms()
        q.put_nowait(("end", a, None, None))

    def _sem_stream(self, http):
        """the text stream of one semantic call on its spec's route: the `claude` CLI door, or the Anthropic Messages API
        (the keyframe planner's own SSE client, Planner._route_stream)"""
        async def stream(spec, system, user, deadline, info, tag=SEM_TAG):
            if spec["route"] == "claude_door":
                async for d in _sem_door_stream(spec, system, user, deadline, info, tag):
                    yield d
                return
            if http is None:
                raise PlanError("the Anthropic route needs the server's HTTP session")
            info["spec"] = spec
            async for d in self._route_stream(http, system, user, deadline, info):
                yield d
        return stream

    async def plan_semantic(self, character: str, description: str, state=None, http=None, on_usd=None, words=None,
                            t_recv=None):
        t0 = t_recv if t_recv is not None else time.monotonic()
        ms = lambda: round((time.monotonic() - t0) * 1000)   # noqa: E731
        st = self._sem_stats()
        st["plans"] += 1
        description = str(description or "").strip()
        if not self.enabled:
            yield dict(refusal("planner off"), planner="semantic")
            return
        if not description:
            yield dict(refusal("empty description"), planner="semantic")
            return
        sem = _SEM_STATE["mod"] if _SEM_STATE["tried"] else await _off(sem_module)
        if sem is None:
            st["failed"] += 1
            yield dict(refusal("semantic planner not installed"), planner="semantic")
            return
        if character not in getattr(sem, "CHARACTERS", ()):
            st["unavailable"] += 1                 # today's path; the done message says so
            async for ev in self.plan(character, description, state, http=http, on_usd=on_usd, words=words):
                yield dict(ev, sem_unavailable=True) if ev.get("done") else ev
            return
        c = self.sem_cfg()
        hz = int(c.get("hz") or 20)
        spec, hedge = self.sem_specs()
        ms_recv = ms()

        def sem_ev(phase, **kw):
            return {"sem": True, "phase": phase, **kw}
        # 1. library (unchanged)
        cmds, why = self._sem_library(character, description, state, words)
        if cmds:
            st["library"] += 1
            d = ms()
            yield {"done": True, "ok": True, "source": "library", "planner": "semantic", "poses": [], "all_poses": [],
                   "engine": cmds, "summary": "", "ms_first_pose": None, "ms_done": d, "usd": 0.0, "error": None,
                   "library": why, "sem_ms": {"recv": ms_recv, "done": d}}
            yield sem_ev("done", program="", source="library", model=None, cached=False,
                         ms={"recv": ms_recv, "first_token": None, "first_stmt": None, "first_pose": None, "done": d},
                         exec_ms=[], warnings=[], errors=[], targets={})
            return
        if spec is None:
            st["failed"] += 1
            yield dict(refusal("semantic planner: the model must be a Claude model"), planner="semantic")
            return
        # 2. cache / 3. planner
        try:
            targets = await _off(sem.resolve_targets, character, state)
            system = self._sem_system(sem, character, targets)
            user = sem.user_prompt(description, state)
        except Exception as e:  # noqa: BLE001
            st["failed"] += 1
            yield dict(refusal(f"semantic prompt error: {type(e).__name__}: {str(e)[:160]}"), planner="semantic")
            return
        twire = sem_targets_wire(targets)
        prompt_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
        norm = normalize(description)
        key = cache_key("semantic\n" + character, norm)
        models = [spec["model"]] + ([hedge["model"]] if hedge else [])
        ent = self.sem_cache_get(key, prompt_sha, models)
        if ent is not None:
            st["cache"] += 1
            yield sem_ev("start", program=ent["program"], source="cache", model=ent.get("model"), cached=True,
                         ms={"recv": ms_recv}, targets=twire)
            poses, rep, replayed = ent["poses"], ent.get("report") or {}, []
            moved = self._sem_live_moved(ent, targets)
            if moved:
                try:
                    poses, rep = await _off(sem.replay, character, ent["program"], targets, 0, hz)
                    replayed = moved
                except Exception as e:  # noqa: BLE001
                    self.log(f"planner: semantic cache replay failed ({type(e).__name__}); stale poses served")
            d = ms()
            yield {"done": True, "ok": True, "source": "cache", "planner": "semantic", "space": "rig", "hz": hz,
                   "poses": poses, "all_poses": poses, "engine": None, "summary": ent.get("summary") or "",
                   "ms_first_pose": d, "ms_done": d, "usd": 0.0, "error": None, "keep_pose": bool(rep.get("keep_pose")),
                   "channels": rep.get("channels") or [], "program": ent["program"], "report": rep,
                   "model": ent.get("model"), "route": spec["route"], "cache_key": key, "prompt_sha": prompt_sha,
                   "replayed": replayed, "library": why,
                   "sem_ms": {"recv": ms_recv, "first_token": None, "first_stmt": None, "first_pose": d, "done": d}}
            yield sem_ev("done", program=ent["program"], source="cache", model=ent.get("model"), cached=True,
                         ms={"recv": ms_recv, "first_token": None, "first_stmt": None, "first_pose": d, "done": d},
                         exec_ms=rep.get("exec_ms") or [], warnings=rep.get("warnings") or [],
                         errors=rep.get("errors") or [], targets=twire, n_statements=rep.get("n_statements"),
                         duration_s=rep.get("duration_s"))
            return
        yield sem_ev("start", program="", source="planner", model=spec["model"], cached=False, ms={"recv": ms_recv},
                     targets=twire)
        deadline = t0 + float(c.get("timeout_s") or 20.0)
        hard_end = deadline + 0.25
        atts: list[_SemAttempt] = []
        q: asyncio.Queue = asyncio.Queue()
        winner = None
        hedge_started = retried = False
        all_poses: list = []
        ms_first_pose = None

        def start(sp, role):
            a = _SemAttempt(role, sp, ms())
            a.http = http
            atts.append(a)
            a.task = _start_task(self._sem_attempt(a, sem, character, targets, hz, system, user, deadline, q, ms))
            return a

        def cancel(a, why):
            if a.ended:
                return
            a.ended = a.cancelled = True
            a.task.cancel()
            a.ms_end = ms()
            a.err = a.err or why

        def timings(a):
            return {"recv": ms_recv, "first_token": a.ms_first_token if a else None,
                    "first_stmt": a.ms_first_stmt if a else None, "first_pose": ms_first_pose, "done": None}

        try:
            _SEM_LAST_USE[sem_warm_key(spec, system)] = time.monotonic()
            primary = start(spec, "primary")
            warm_n = c.get("warm_n")
            for sp in (spec, hedge):              # later plans with these targets find warm processes
                if sp and warm_n != 0 and sp["route"] == "claude_door":
                    try:
                        sem_door(sp.get("effort")).prewarm(sp["model"], system, int(warm_n or 2))
                    except Exception:  # noqa: BLE001
                        pass
            hedge_at = primary.t_start + hedge["after_s"] if hedge else None
            while True:
                now = time.monotonic()
                wait = hard_end - now
                if hedge and not hedge_started and winner is None:
                    wait = min(wait, hedge_at - now)
                ev = None
                if wait > 0:
                    try:
                        ev = await asyncio.wait_for(q.get(), wait)
                    except asyncio.TimeoutError:
                        ev = None
                elif not q.empty():
                    ev = q.get_nowait()
                if ev is None:
                    now = time.monotonic()
                    if hedge and not hedge_started and winner is None and now >= hedge_at - 1e-4:
                        hedge_started = True
                        st["hedged"] += 1
                        start(hedge, "hedge")
                        continue
                    if now >= hard_end:
                        for a in atts:
                            cancel(a, "timeout")
                        break
                    continue
                kind, a, bs, rep = ev
                if kind == "batches":
                    if a.cancelled or not bs:
                        continue
                    if winner is None:
                        winner = a
                        for o in atts:
                            if o is not a:
                                cancel(o, f"cancelled: {a.spec['model']} ({a.role}) streamed first")
                    if a is not winner:
                        continue
                    poses = [p for b in bs for p in b["poses"]]
                    all_poses.extend(poses)
                    if ms_first_pose is None:
                        ms_first_pose = ms()
                    yield {"done": False, "space": "rig", "hz": hz, "poses": poses}
                    yield sem_ev("stmt", program=sem_code(a.text), source="planner", model=a.spec["model"],
                                 cached=False, ms=timings(a), exec_ms=(rep or {}).get("exec_ms") or [],
                                 warnings=(rep or {}).get("warnings") or [], errors=(rep or {}).get("errors") or [],
                                 n_statements=(rep or {}).get("n_statements"))
                    continue
                if a.cancelled:
                    continue
                a.ended = True
                if a is winner:
                    break
                if winner is not None:
                    continue
                # it ended without a pose
                if hedge and not hedge_started:
                    hedge_started = True
                    st["hedged"] += 1
                    self.log(f"planner: semantic hedge {hedge['model']} after {a.failure()[:120]}")
                    start(hedge, "hedge")
                    continue
                if not retried and deadline - time.monotonic() >= float(c.get("retry_min_s") or 8.0):
                    retried = True
                    st["retries"] += 1
                    self.log(f"planner: semantic retry after {a.failure()[:120]}")
                    start(a.spec, "retry")
                    continue
                if all(o.ended for o in atts):
                    break
            d = ms()
            w = winner
            last = w or (atts[-1] if atts else None)
            rep = (last.report if last else {}) or {}
            ok = w is not None and bool(all_poses)
            if w is not None:
                err = w.err
            elif len(atts) > 1:
                err = "; ".join(f"{o.role} {o.spec['model']}: {o.failure()}" for o in atts)[:600]
            else:
                err = last.failure() if last else "no attempt"
            program = rep.get("program") or sem_code(last.text if last else "")
            tm = dict(timings(last), done=d)
            n_st = rep.get("n_statements")
            summary = f"semantic program: {n_st} statements, {rep.get('duration_s')} s" if ok else ""
            usd = 0.0                              # the API route bills per call (the door: 0, the CLI's plan)
            for o in atts:
                if o.spec["route"] != "claude_door" and (o.text or o.info.get("usage")):
                    usd += self.usd_of(o.spec["model"], o.info.get("usage"), system, user, o.text, o.spec["route"],
                                       o.cancelled)[0]
            if usd and on_usd is not None:
                try:
                    on_usd(usd)
                except Exception:  # noqa: BLE001
                    pass
            res = {"done": True, "ok": ok, "source": "planner", "planner": "semantic", "space": "rig", "hz": hz,
                   "poses": [], "all_poses": all_poses if ok else [], "engine": None, "summary": summary,
                   "ms_first_pose": ms_first_pose, "ms_done": d, "usd": round(usd, 6), "error": err,
                   "keep_pose": bool(rep.get("keep_pose")) if ok else False, "channels": rep.get("channels") or [],
                   "program": program, "report": rep, "model": (last.spec["model"] if last else spec["model"]),
                   "route": (last.spec["route"] if last else spec["route"]),
                   "effort": (last.spec.get("effort") if last else spec.get("effort")),
                   "attempts": [o.record() for o in atts], "cache_key": key, "prompt_sha": prompt_sha, "library": why,
                   "retried": retried, "hedge_started": hedge_started, "sem_ms": tm, "targets": twire}
            if ok:
                st["planner"] += 1
                if err is None and not rep.get("errors"):
                    self.sem_cache_put(key, {"key": key, "character": character, "description": description,
                                             "normalized": norm, "program": program, "poses": all_poses, "report": rep,
                                             "summary": summary, "model": w.spec["model"], "effort": w.spec.get("effort"),
                                             "prompt_sha": prompt_sha, "targets": twire, "ts": time.time(),
                                             "ms_first_pose": ms_first_pose, "ms_done": d})
            else:
                st["failed"] += 1
            yield res
            yield sem_ev("done", program=program, source="planner", model=res["model"], cached=False, ms=tm,
                         exec_ms=rep.get("exec_ms") or [], warnings=rep.get("warnings") or [],
                         errors=rep.get("errors") or [], targets=twire, n_statements=n_st,
                         duration_s=rep.get("duration_s"), error=None if ok else err)
        finally:
            for a in atts:
                cancel(a, "cancelled")


class Planner(_SemanticPlanner):
    def __init__(self, cfg=None, log=print, cache_dir: str | None = None, prompt_dir: str = PROMPT_DIR,
                 scene_dir: str | None = None):
        c = dict(DEFAULTS)
        if isinstance(cfg, dict):
            c.update(cfg)
        elif cfg is not None:
            c["enabled"] = bool(cfg)
        self.cfg = c
        self.log = log
        self.cache_dir = cache_dir
        self.prompts = load_prompts(prompt_dir)
        import charcaps; charcaps.alias_prompts(self.prompts)   # juno, junor and the mascots plan with the muse's prompt
        # the configured prompt variant replaces the prompt of the characters it was measured for (the MANIFEST entry's
        # "for": the Muse and the mascots on its hand map — not Juno, whose arms are her own); an unknown variant / a sha
        # mismatch leaves the frozen prompt in place
        self.variant = prompt_variant(c)
        if self.variant:
            for k, vs in load_variants(prompt_dir).items():
                v = vs.get(self.variant)
                if not v or k not in self.prompts:
                    continue
                if not v["ok"]:
                    log(f"planner: prompt variant {k}.{self.variant} sha mismatch with MANIFEST.json — the frozen prompt is used")
                    continue
                base = self.prompts[k]
                for ch in v["for"]:
                    if self.prompts.get(ch) is base:           # only who plans with k's prompt today
                        self.prompts[ch] = v
        self.scene_dir = scene_dir                 # where <character>.props.json live (default renderer/animals)
        self.appendix_path = os.path.join(prompt_dir, "objects_appendix.txt")
        self._matchers: dict = {}
        self.stream_fn = self._route_stream        # tests swap in a recorded-stream replayer: fn(http, system, user,
        #                                            deadline, info) yielding text; info["spec"] = the attempt's model
        self.stats = {"plans": 0, "library": 0, "cache": 0, "planner": 0, "failed": 0, "usd": 0.0,
                      "fallback_started": 0, "fallback_used": 0, "stalled": 0}
        self._stat_models()
        for k, p in self.prompts.items():
            if not p["ok"]:
                log(f"planner: prompt {k} sha mismatch with MANIFEST.json — its plans are refused")

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled"))

    # -- the model choice
    def primary(self) -> dict:
        return primary_spec(self.cfg)

    def fallback(self) -> dict | None:
        return fallback_spec(self.cfg)

    def _specs(self) -> tuple[dict, dict | None]:
        """(primary, fallback) memoized on the config keys they depend on — the plan path's dispatch costs ~1 us; the
        env overrides are read when the memo is rebuilt (os.environ.get is ~2.5 us a call; a service's env is fixed)"""
        c = self.cfg
        fb = c.get("fallback")
        k = (c.get("model"), c.get("route"), c.get("effort"), c.get("zdr"),
             tuple(sorted(fb.items())) if isinstance(fb, dict) else fb)
        m = self.__dict__.get("_spec_memo")
        if m is None or m[0] != k:
            m = self._spec_memo = (k, self.primary(), self.fallback())
            self._stat_models(m[1], m[2] or {})
        return m[1], m[2]

    def describe(self) -> dict:
        """the models this planner plans with now (for /api status)"""
        p, f = self.primary(), self.fallback()
        served = [p["model"]] + ([f["model"]] if f else []) + cache_models(self.cfg)
        return {"model": p["model"], "route": p["route"], "effort": p["effort"], "preset": p["preset"],
                "fallback": None if f is None else {k: f[k] for k in ("model", "route", "effort", "after_s")},
                "quality": resolve_quality(self.cfg), "cache_models": list(dict.fromkeys(served))}

    def _stat_models(self, p=None, f=None) -> None:
        p = p or self.primary()
        f = f if f is not None else self.fallback()
        self.stats.update({"model": p["model"], "route": p["route"], "effort": p["effort"],
                           "fallback": None if not f else {k: f[k] for k in ("model", "route", "effort", "after_s")}})

    # -- keep the claude door warm for a character's prompt (primary / fallback on route claude_door)
    def prewarm(self, character: str, n: int = 2) -> list[dict]:
        """start the door's warm processes for this character's planner prompt; -> the specs warmed ([] = none on the
        door route / no prompt / door unavailable). Safe to call often (the door keeps n per (model, prompt))."""
        P = self.prompts.get(character)
        if not self.enabled or P is None or not P.get("ok"):
            return []
        out = []
        p, f = self._specs()
        systems = [(P["text"], n)]
        # a plan WITH things runs on the prompt + the objects appendix — another (model, system) pair,
        # so it was a cold spawn every time; keep one warm process for it too (cfg "prewarm_things", default on)
        if self.cfg.get("prewarm_things", True):
            scfg = load_scene(character, self.scene_dir)
            app = scene_appendix(scfg, self.appendix_path) if scfg is not None else ""
            if app:
                systems.append((P["text"].rstrip("\n") + "\n\n" + app + "\n", 1 if n else 0))
        for sp in (p, f):
            if sp and sp["route"] == "claude_door":
                try:
                    for sysp, k in systems:          # the plain prompt first (the door's budget fills in order)
                        door_for(sp.get("effort")).prewarm(sp["model"], sysp, k)
                    out.append(sp)
                except Exception as e:  # noqa: BLE001
                    self.log(f"planner: door prewarm failed: {type(e).__name__}: {str(e)[:120]}")
        return out

    # -- library
    def _matcher(self, character: str):
        if character not in self._matchers:
            import intent_hub                     # the server's matcher factory (act grammar for the muse)
            self._matchers[character] = (intent_hub.matcher_for(character), intent_hub.act_for(character))
        return self._matchers[character]

    def library(self, character: str, description: str):
        if not self.cfg.get("library", True):
            return None, "library off"
        m, act = self._matcher(character)
        tr = (self.prompts.get(character) or {}).get("traits") or {}
        return library_check(character, description, m, tr, act)

    # -- cache
    def _cache_path(self, key: str) -> str | None:
        return os.path.join(self.cache_dir, key + ".json") if self.cache_dir else None

    def cache_get(self, key: str, prompt_sha: str, model, extra: bool = True) -> dict | None:
        """an entry is served when its prompt sha matches and the model that wrote it is `model` (an id or a list of
        ids: the primary and the fallback) or (extra) one of cfg cache_models"""
        p = self._cache_path(key)
        if not p or not self.cfg.get("cache", True):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                ent = json.load(f)
        except (OSError, ValueError):
            return None
        ok_models = {model} if isinstance(model, str) else set(model or ())
        if extra:
            ok_models.update(cache_models(self.cfg))
        if ent.get("prompt_sha") != prompt_sha or ent.get("model") not in ok_models or not ent.get("poses"):
            return None                            # a different prompt / model wrote it: plan afresh (and overwrite)
        if check_on(self.cfg) and any(i in CHECK_HARD for i in plan_issues(ent["poses"])):
            self.stats["cache_rejected"] = self.stats.get("cache_rejected", 0) + 1
            return None                            # an old plan today's check rejects: plan afresh
        return ent

    def cache_put(self, key: str, ent: dict) -> None:
        p = self._cache_path(key)
        if not p or not self.cfg.get("cache", True):
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ent, f, separators=(",", ":"))
            os.replace(tmp, p)
        except OSError as e:
            self.log(f"planner: cache write failed: {type(e).__name__}")

    # -- cost
    def usd_of(self, model: str, usage: dict | None, system: str, user: str, text: str, route: str | None = None,
               cancelled: bool = False) -> tuple[float, bool]:
        return usd_of(model, usage, system, user, text, route, cancelled)

    # -- one streamed call on the attempt's route (info["spec"]): yields text deltas; fills info["usage"]
    async def _route_stream(self, http, system: str, user: str, deadline: float, info: dict):
        import aiohttp
        spec = info.get("spec") or self.primary()
        route = spec["route"]
        if route == "claude_door":                 # the `claude` CLI door, no HTTP, no key
            async for d in door_stream(spec, system, user, deadline, info):
                yield d
            return
        handle = ROUTE_EVENTS.get(route)
        if handle is None:
            raise PlanError(f"unknown route {route!r}")
        env = KEY_ENV[route]
        key = os.environ.get(env) or ""
        if not key:
            raise PlanError(f"no {env}")             # names the missing key (the page and the log show it)
        url, body = build_request(spec, system, user, int(self.cfg["max_output_tokens"]))
        headers = {"content-type": "application/json"}
        if route == "gemini":
            headers["x-goog-api-key"] = key        # a header, never the URL (URLs reach logs)
        elif route == "anthropic":
            headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = "Bearer " + key
        left = deadline - time.monotonic()
        try:
            resp = await asyncio.wait_for(
                http.post(url, json=body, headers=headers,
                          timeout=aiohttp.ClientTimeout(total=None, sock_connect=min(10.0, max(1.0, left)))),
                timeout=max(0.5, left))
        except asyncio.TimeoutError:
            raise PlanError("timeout waiting for the answer", transient=True) from None
        except aiohttp.ClientError as e:
            raise PlanError(f"connection: {type(e).__name__}", transient=True) from None
        async with resp:
            if resp.status != 200:
                txt = _redact((await resp.text())[:300])
                raise PlanError(f"HTTP {resp.status}: {txt}", transient=resp.status == 429 or resp.status >= 500)
            buf = b""
            while True:
                left = min(deadline, info.get("grace_until", deadline)) - time.monotonic()
                if left <= 0:
                    raise PlanError("timeout", transient=True)
                try:
                    chunk = await asyncio.wait_for(resp.content.readany(), timeout=left)
                except asyncio.TimeoutError:
                    raise PlanError("timeout", transient=True) from None
                except aiohttp.ClientError as e:
                    raise PlanError(f"connection: {type(e).__name__}", transient=True) from None
                evs, buf = sse_split(buf + (chunk or b"\n"))
                for e in evs:
                    if e is SSE_DONE:
                        return
                    texts, fin = handle(e, info)
                    for d in texts:
                        yield d
                    if fin:
                        return
                if not chunk:
                    return

    _openai_stream = _route_stream                 # older name

    # -- one attempt: its stream -> its parser; ("poses", a, new) / ("end", a, None) into the plan's queue
    async def _attempt(self, a: _Attempt, http, system: str, user: str, deadline: float, q: asyncio.Queue, ms) -> None:
        info = a.info
        stall_s = float(self.cfg.get("stall_s") or STALL_S)
        stall_m = float(self.cfg.get("stall_motion_s") or STALL_MOTION_S)
        if a.spec.get("route") == "claude_door":
            if a.role == "nothink" or no_think_on(self.cfg):
                user = no_think_user(user)
            info["on_thinking"] = lambda: q.put_nowait(("thinking", a, None))
        try:
            async for d in self.stream_fn(http, system, user, deadline, info):
                a.text += d
                if a.parser.complete:
                    # the poses are all out; read on (briefly) only for the usage event the cost needs
                    if time.monotonic() > info["grace_until"]:
                        break
                    continue
                now = ms()
                new = a.parser.feed(now, d)
                if new:
                    if a.ms_first_pose is None:
                        a.ms_first_pose = new[0]["ms"]
                    a.stall_check(now, stall_s, stall_m)
                    a.last_pose_ms, a.last_pose_t = now, new[-1]["t"]
                    q.put_nowait(("poses", a, new))
                if a.parser.complete:
                    if not new:
                        a.stall_check(now, stall_s, stall_m)
                    info["grace_until"] = time.monotonic() + GRACE_S
        except PlanError as e:
            if not a.parser.complete:              # after the array closed only the usage was missing
                a.err, a.transient = str(e), e.transient
        except asyncio.CancelledError:
            raise                                  # the plan cancelled it (and bills it)
        except Exception as e:  # noqa: BLE001
            a.err = f"{type(e).__name__}: {str(e)[:200]}"
        a.ms_end = ms()
        if not a.parser.complete:
            a.stall_check(a.ms_end, stall_s, stall_m)
        q.put_nowait(("end", a, None))

    # -- the plan
    async def plan(self, character: str, description: str, state=None, http=None, on_usd=None, model=None, words=None,
                   planner=None, t_recv=None):
        """model: None = the configured primary (+ fallback); a preset / id / {"model", "route", "effort"} = that model
        alone (no fallback; the cache serves only its own plans) — Planner.fill's quality pass.
        planner='semantic' = the semantic-API planner (plan_semantic); None = exactly the path above."""
        if planner == "semantic":
            async for ev in self.plan_semantic(character, description, state, http=http, on_usd=on_usd, words=words,
                                               t_recv=t_recv):
                yield ev
            return
        t0 = time.monotonic()
        ms = lambda: round((time.monotonic() - t0) * 1000)   # noqa: E731
        self.stats["plans"] += 1
        description = str(description or "").strip()
        if not self.enabled:
            yield refusal("planner off")
            return
        if not description:
            yield refusal("empty description")
            return
        P = self.prompts.get(character)
        if P is None:
            yield refusal(f"no planner prompt for {character!r}")
            return
        scfg, things = scene_for(character, state, self.scene_dir)       # a plan WITH things
        named = scfg is not None and names_thing(description, scfg)

        def lib(text):
            c, w = self.library(character, text)
            # a request naming one of the things is not a body move the library knows ("spin the book" != spin):
            # only object steps the library fully understood stay instant
            if c and named and not any(isinstance(x, dict) and x.get("action") == "object" for x in c):
                return None, "names a thing: planned with it"
            return c, w
        # 1. library
        cmds, why = lib(description)
        # the model restates the user's request with extra detail ("do a backward somersault now, one full rotation,
        # landing on both feet") that the library cannot read back; the user's own words (the page's newest final
        # transcript, already what-if-checked) often are a named move the library knows -> instant, the same move the
        # matchers ran
        if not cmds and words and str(words).strip():
            cw, why_w = lib(str(words).strip()[:400])
            if cw:
                cmds, why = cw, "user's words: " + why_w
        if cmds:
            self.stats["library"] += 1
            d = ms()
            yield {"done": True, "ok": True, "source": "library", "poses": [], "all_poses": [], "engine": cmds,
                   "summary": "", "ms_first_pose": None, "ms_done": d, "usd": 0.0, "error": None, "library": why}
            return
        if not P["ok"]:
            self.stats["failed"] += 1
            yield refusal("prompt sha mismatch")
            return
        norm = normalize(description)
        note = state_note(P["traits"], state)
        system, prompt_sha, controls, allow_do = P["text"], P["sha256"], P["controls"], False
        if scfg is not None:
            app = scene_appendix(scfg, self.appendix_path)
            if app:
                system = P["text"].rstrip("\n") + "\n\n" + app + "\n"
                prompt_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
                tn = things_note(things, state.get("root"))
                note = (note + " " + tn).strip() if tn else note
                controls, allow_do = list(controls) + KF_OBJ_CONTROLS, True
        key = cache_key(character, norm, note)
        if model is None:
            spec, fb = self._specs()
        else:
            spec, fb = _spec(model if isinstance(model, dict) else {"model": model}, self.cfg.get("zdr", "auto")), None
        # 2. cache (written by the primary, the fallback or a cache_models model, from this prompt)
        ent = self.cache_get(key, prompt_sha, [spec["model"]] + ([fb["model"]] if fb else []), extra=model is None)
        if ent is not None:
            self.stats["cache"] += 1
            d = ms()
            yield {"done": True, "ok": True, "source": "cache", "poses": ent["poses"], "all_poses": ent["poses"],
                   "engine": None, "summary": ent.get("summary") or "", "ms_first_pose": d, "ms_done": d, "usd": 0.0,
                   "error": None, "cache_key": key, "prompt_sha": prompt_sha, "library": why, "model": ent.get("model"),
                   "route": ent.get("route"), "fallback_used": False}
            return
        # 3. planner: the primary; the fallback hedged in (no pose after after_s / failed first); first pose wins
        user = USER_TMPL.format(text=description) + ("\n" + note if note else "")
        deadline = t0 + float(self.cfg["timeout_s"])
        hard_end = deadline + 0.25                  # a stream that ignores the deadline is cancelled here
        atts: list[_Attempt] = []
        q: asyncio.Queue = asyncio.Queue()
        winner = None
        fb_started = False
        nt_started = False                          # no_think "retry" request started
        own_http = None
        if http is None:
            import aiohttp
            own_http = http = aiohttp.ClientSession()

        def start(sp: dict, role: str) -> _Attempt:
            a = _Attempt(role, sp, controls, ms(), allow_do)
            atts.append(a)
            a.task = _start_task(self._attempt(a, http, system, user, deadline, q, ms))
            return a

        def bill(a: _Attempt, cancelled: bool = False) -> None:
            if a.billed:
                return
            a.billed = True
            if a.text or a.info.get("usage") or cancelled:
                a.usd, a.estimated = self.usd_of(a.spec["model"], a.info.get("usage"), system, user, a.text,
                                                 a.spec["route"], cancelled)
            self.stats["usd"] = round(self.stats["usd"] + a.usd, 6)
            if on_usd and a.usd > 0:
                on_usd(a.usd)

        def cancel(a: _Attempt, why: str) -> None:
            if a.ended:
                return
            a.ended = a.cancelled = True
            a.task.cancel()
            a.ms_end = ms()
            a.err = a.err or why
            bill(a, cancelled=True)

        try:
            primary = start(spec, "primary")
            fb_at = primary.t_start + fb["after_s"] if fb else None
            # "hedge_after_s" = a second request to the SAME claude_door model when the primary has not
            # sent a single character after that many s (a server-side first-token stall); first pose wins, as with the
            # fallback. None / 0 = off (the default).
            h_s = self.cfg.get("hedge_after_s") if model is None and spec.get("route") == "claude_door" else None
            hedge_at = primary.t_start + float(h_s) if h_s else None
            while True:
                now = time.monotonic()
                wait = hard_end - now
                if fb and not fb_started and winner is None:
                    wait = min(wait, fb_at - now)
                if hedge_at is not None and winner is None:
                    wait = min(wait, hedge_at - now)
                ev = None
                if wait > 0:
                    try:
                        ev = await asyncio.wait_for(q.get(), wait)
                    except asyncio.TimeoutError:
                        ev = None
                elif not q.empty():
                    ev = q.get_nowait()
                if ev is None:
                    now = time.monotonic()
                    if hedge_at is not None and now >= hedge_at - 1e-4:
                        hedge_at = None
                        if winner is None and not primary.text and not primary.ended:
                            self.stats["hedge_started"] = self.stats.get("hedge_started", 0) + 1
                            self.log(f"planner: hedge {spec['model']} — no text from the primary after {float(h_s):g} s")
                            start(spec, "hedge")
                        continue
                    if fb and not fb_started and winner is None and now >= fb_at - 1e-4:
                        fb_started = True
                        self.stats["fallback_started"] += 1
                        self.log(f"planner: fallback {fb['model']} ({fb['route']}) — no pose from {spec['model']} "
                                 f"after {fb['after_s']:g} s")
                        start(fb, "fallback")
                        continue
                    if now >= hard_end:
                        for a in atts:
                            cancel(a, "timeout")
                        break
                    continue
                kind, a, new = ev
                if kind == "thinking":             # the primary opened a hidden thinking block
                    a.info.pop("on_thinking", None)
                    if (a.role == "primary" and winner is None and not a.ended and not nt_started
                            and no_think_mode(self.cfg) == "retry"):
                        nt_started = True
                        self.stats["nothink_started"] = self.stats.get("nothink_started", 0) + 1
                        self.log(f"planner: {spec['model']} is thinking ({a.info.get('thinking_ms')} ms) — "
                                 f"a second request with the no-think hint")
                        start(spec, "nothink")
                        continue
                    if (fb and not fb_started and winner is None and a.role == "primary" and not a.ended
                            and self.cfg.get("fallback_on_thinking", DEFAULTS["fallback_on_thinking"])):
                        fb_started = True
                        self.stats["fallback_started"] += 1
                        self.log(f"planner: fallback {fb['model']} ({fb['route']}) — {spec['model']} is thinking "
                                 f"({a.info.get('thinking_ms')} ms)")
                        start(fb, "fallback")
                    continue
                if kind == "poses":
                    if a.cancelled:
                        continue                   # a loser's batch queued before it was cancelled
                    if winner is None:
                        winner = a
                        for o in atts:
                            if o is not a:
                                cancel(o, f"cancelled: {a.spec['model']} streamed first")
                    if a is winner:
                        yield {"done": False, "poses": new}
                    continue
                # an attempt ended
                if a.cancelled:
                    continue
                a.ended = True
                bill(a)
                if a is winner:
                    break
                if winner is not None:
                    continue
                # it failed before its first pose
                if fb and not fb_started:
                    fb_started = True
                    self.stats["fallback_started"] += 1
                    self.log(f"planner: fallback {fb['model']} ({fb['route']}) after {a.failure()[:120]}")
                    start(fb, "fallback")
                    continue
                if (not fb and a.role == "primary" and a.err and a.transient
                        and deadline - time.monotonic() >= float(self.cfg["retry_min_s"])):
                    self.log(f"planner: retry after {a.err[:120]}")
                    start(spec, "retry")
                    continue
                if all(o.ended for o in atts):
                    break
            # the outcome
            d = ms()
            w = winner
            last = w or (atts[-1] if atts else None)
            parser = (w or last).parser
            parser.finish(d)
            ok = w is not None and bool(w.parser.poses)
            if w is not None:
                err = w.err
            elif len({a.spec["model"] for a in atts}) > 1:
                err = "; ".join(f"{a.role} {a.spec['model']}: {a.failure()}" for a in atts)[:600]
            else:
                err = last.failure()
            summary = _summary_of((w or last).text)
            wire = [wire_pose(p) for p in parser.poses] if w is not None else []
            usd_total = sum(a.usd for a in atts)
            used = w or last
            fallback_used = w is not None and w.role == "fallback"
            if fallback_used:
                self.stats["fallback_used"] += 1
            stalls = w.stalls if w is not None else []
            if stalls:
                self.stats["stalled"] += 1
                self.log(f"planner: {used.spec['model']} stalled {max(x['gap_ms'] for x in stalls)} ms with "
                         f"{stalls[0]['motion_s']} s of motion parsed (poses already sent: no switch)")
            res = {"done": True, "ok": ok, "source": "planner", "poses": [], "all_poses": wire, "engine": None,
                   "summary": summary, "ms_first_pose": parser.poses[0]["ms"] if wire else None, "ms_done": d,
                   "usd": round(usd_total, 6), "error": err, "cache_key": key, "prompt_sha": prompt_sha,
                   "model": used.spec["model"], "route": used.spec["route"], "effort": used.spec.get("effort"),
                   "fallback_used": fallback_used, "fallback_started": fb_started, "stalls": stalls,
                   "attempts": [a.record() for a in atts], "dropped": len(parser.dropped),
                   "salvaged": len(parser.salvaged), "unknown": sorted({u for e in parser.errors for u in e["unknown"]}),
                   "complete_how": parser.complete_how, "library": why, "note": note}
            issues = plan_issues(wire) if ok and check_on(self.cfg) else []     # quality
            if issues:
                res["issues"] = issues
                self.log(f"planner: plan check {issues} for {description[:60]!r} ({used.spec['model']})"
                         + (" — not cached" if any(i in CHECK_HARD for i in issues) else ""))
            if ok and err is None:
                self.stats["planner"] += 1
            if ok and err is None and not any(i in CHECK_HARD for i in issues):
                self.cache_put(key, {"key": key, "character": character, "description": description, "normalized": norm,
                                     "note": note, "poses": wire, "summary": summary, "prompt_sha": prompt_sha,
                                     "model": used.spec["model"], "route": used.spec["route"],
                                     "effort": used.spec.get("effort"), "fallback_used": fallback_used,
                                     "created": time.time(), "usd": round(usd_total, 6),
                                     "ms_first_pose": res["ms_first_pose"], "ms_done": d})
            else:
                self.stats["failed"] += 0 if ok else 1
            yield res
            pend = [a.task for a in atts if a.task is not None and not a.task.done()]
            if pend:                               # let the cancelled losers close their connections
                await asyncio.wait(pend, timeout=2.0)
        finally:
            for a in atts:                         # the session ended mid-plan (or an error): stop and bill the rest
                cancel(a, "cancelled")
            if own_http is not None:
                await own_http.close()

    async def fill(self, character: str, description: str, state=None, http=None, on_usd=None) -> dict | None:
        """offline library / cache filling: plan with the quality model (cfg "quality", default Sol) alone and cache
        the answer (cache_models keeps Sol's plans welcome on the live path) -> the done event"""
        last = None
        async for ev in self.plan(character, description, state, http=http, on_usd=on_usd,
                                  model=self.cfg.get("quality") or "quality"):
            if ev.get("done"):
                last = ev
        return last


def _attempt_brief(a: dict) -> dict:
    return {k: a.get(k) for k in ("model", "route", "role", "ms_start", "ms_end", "ms_first_pose", "error", "usd",
                                  "estimated", "stalls")}


async def run_plan(planner: Planner, pid: str, character: str, description: str, state, send, http=None, on_usd=None,
                   record=None, words=None, mode=None, t_recv=None) -> dict:
    """Drive one plan and send its kf messages (contract section 2) through send(msg); -> the plan record.
    mode='semantic' = the semantic-API planner (kf space:'rig' + sem messages); None = today's path."""
    if mode == "semantic":
        return await _run_sem_plan(planner, pid, character, description, state, send, http, on_usd, record, words, t_recv)
    rec = {"id": pid, "character": character, "description": description[:200], "t0": time.time()}
    last = None
    n_msgs = 0
    try:
        async for ev in planner.plan(character, description, state, http=http, on_usd=on_usd, words=words):
            send(kf_message(pid, ev))
            n_msgs += 1
            if ev.get("done"):
                last = ev
    except asyncio.CancelledError:
        rec.update({"ok": False, "error": "cancelled"})
        if record is not None:
            record(rec)
        raise
    except Exception as e:  # noqa: BLE001
        last = refusal(f"planner error: {type(e).__name__}: {str(e)[:160]}")
        send(kf_message(pid, last))
        n_msgs += 1
    if last is not None:
        rec.update({k: last.get(k) for k in ("ok", "source", "summary", "ms_first_pose", "ms_done", "usd", "error")})
        rec["n_poses"] = len(last.get("all_poses") or [])
        rec["engine"] = last.get("engine")
        rec["msgs"] = n_msgs
        for k in ("model", "route", "fallback_used"):
            if k in last:
                rec[k] = last[k]
        if last.get("stalls"):
            rec["stalls"] = last["stalls"]
        if last.get("attempts") is not None:
            rec["attempts"] = [_attempt_brief(a) for a in last["attempts"]]
        if last.get("unknown"):
            rec["unknown"] = last["unknown"]
    if record is not None:
        record(rec)
    return rec


async def _run_sem_plan(planner: Planner, pid, character, description, state, send, http, on_usd, record, words,
                        t_recv) -> dict:
    """run_plan for planner:'semantic' : kf (space:'rig') + sem messages; the record gains planner:'semantic',
    the program and the timings"""
    rec = {"id": pid, "character": character, "description": description[:200], "t0": time.time(), "planner": "semantic"}
    last = None
    n_msgs = 0
    try:
        async for ev in planner.plan(character, description, state, http=http, on_usd=on_usd, words=words,
                                     planner="semantic", t_recv=t_recv):
            if ev.get("sem"):
                send(sem_message(pid, ev))
                continue
            send(sem_kf_message(pid, ev) if ev.get("planner") == "semantic" or ev.get("space") == "rig"
                 else kf_message(pid, ev))
            n_msgs += 1
            if ev.get("done"):
                last = ev
    except asyncio.CancelledError:
        rec.update({"ok": False, "error": "cancelled"})
        if record is not None:
            record(rec)
        raise
    except Exception as e:  # noqa: BLE001
        last = dict(refusal(f"planner error: {type(e).__name__}: {str(e)[:160]}"), planner="semantic")
        send(sem_kf_message(pid, last))
        n_msgs += 1
    if last is not None:
        rec.update({k: last.get(k) for k in ("ok", "source", "summary", "ms_first_pose", "ms_done", "usd", "error")})
        rec["n_poses"] = len(last.get("all_poses") or [])
        rec["engine"] = last.get("engine")
        rec["msgs"] = n_msgs
        for k in ("model", "route", "effort", "program", "retried", "hedge_started", "replayed", "keep_pose"):
            if last.get(k) is not None:
                rec[k] = last[k]
        if last.get("sem_unavailable"):
            rec["sem_unavailable"] = True
        if last.get("sem_ms"):
            rec["ms"] = last["sem_ms"]
        r = last.get("report") or {}
        if r:
            rec["n_statements"] = r.get("n_statements")
            rec["duration_s"] = r.get("duration_s")
            rec["warnings"] = (r.get("warnings") or [])[:20]
            rec["errors"] = (r.get("errors") or [])[:20]
        if last.get("attempts") is not None:
            rec["attempts"] = [dict({k: a.get(k) for k in ("model", "effort", "role", "ms_start", "ms_first_token",
                                                          "ms_first_stmt", "ms_end", "error")},
                                    warm=(a.get("door") or {}).get("warm"),
                                    cache_read=(a.get("usage") or {}).get("cache_read_input_tokens"),
                                    out_tokens=(a.get("usage") or {}).get("output_tokens"))
                               for a in last["attempts"]]
    if record is not None:
        record(rec)
    return rec
