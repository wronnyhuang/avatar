"""Direct voice: the server brokers ONE WebRTC session from the phone straight to OpenAI.

The phone holds the media (mic up, voice down) and the 'oai-events' data channel (tools, transcripts); the server only
turns the phone's SDP offer into an answer with the project key (the key never reaches the browser), meters spend
over a sideband WebSocket, and can hang the call up (session/day caps).

    mode 'realtime'  POST https://api.openai.com/v1/realtime/calls   multipart: sdp + session (gpt-realtime-2.1)
                     answer = response body (application/sdp); call id = last segment of the Location header
                     sideband  wss://api.openai.com/v1/realtime?call_id=<id>      hangup POST /v1/realtime/calls/<id>/hangup
    mode 'live'      POST https://api.openai.com/v1/live/sessions        JSON {session, transport:{type:webrtc, sdp}}
                     answer = transport.sdp; id = session.id (gpt-live-1; tools only exist through Responses delegation)
                     sideband  wss://api.openai.com/v1/live/sessions/<id>/attach   close = session.close on the sideband

Vendor error bodies are never reflected to the browser (logged scrubbed, truncated).
"""
from __future__ import annotations

import asyncio
import inspect   # intent
import json
import os
import time

import aiohttp

API = os.environ.get("AVATAR_OPENAI_API", "https://api.openai.com/v1")
WSS = API.replace("https://", "wss://").replace("http://", "ws://")
REALTIME_MODEL = os.environ.get("AVATAR_REALTIME_MODEL", "gpt-realtime-2.1")
LIVE_MODEL = "gpt-live-1"
LIVE_BACKEND = os.environ.get("AVATAR_LIVE_BACKEND", "gpt-6-luna")
ASR_MODEL = os.environ.get("AVATAR_ASR_MODEL", "gpt-4o-mini-transcribe")
MAX_OFFER = 64 * 1024

# $ per 1M tokens, gpt-realtime-2.1 list; images priced as text input; err high.
RT_RATES = {"text_in": 4.0, "audio_in": 32.0, "image_in": 5.0, "cached": 0.5, "text_out": 24.0, "audio_out": 64.0}
ASR_USD_PER_MIN = 0.003           # gpt-4o-mini-transcribe
LIVE_USD_PER_MIN = 0.05           # gpt-live-1, wall clock, silence included
LIVE_BACKEND_RATES = (2.0, 10.0)  # $/1M in, out (priced as GPT-6 Sol — err high), x2 for the priority tier
LIVE_PRIORITY = os.environ.get("AVATAR_LIVE_PRIORITY", "1") != "0"


def realtime_usd(u: dict) -> float:
    """Dollars for one response.done usage block (realtime)."""
    if not isinstance(u, dict):
        return 0.0
    i, o = u.get("input_token_details") or {}, u.get("output_token_details") or {}
    c = i.get("cached_tokens_details") or {}
    ct, ca, ci = int(c.get("text_tokens") or 0), int(c.get("audio_tokens") or 0), int(c.get("image_tokens") or 0)
    r = RT_RATES
    usd = (max(0, int(i.get("text_tokens") or 0) - ct) * r["text_in"]
           + max(0, int(i.get("audio_tokens") or 0) - ca) * r["audio_in"]
           + max(0, int(i.get("image_tokens") or 0) - ci) * r["image_in"]
           + (ct + ca + ci) * r["cached"]
           + int(o.get("text_tokens") or 0) * r["text_out"] + int(o.get("audio_tokens") or 0) * r["audio_out"]) / 1e6
    if not i and not o:            # no breakdown: price every token at the audio rates (err high)
        usd = (int(u.get("input_tokens") or 0) * r["audio_in"] + int(u.get("output_tokens") or 0) * r["audio_out"]) / 1e6
    return usd


def live_backend_usd(tin: int, tout: int) -> float:
    pin, pout = LIVE_BACKEND_RATES
    return (max(0, tin) * pin + max(0, tout) * pout) * (2 if LIVE_PRIORITY else 1) / 1e6


# ---------------------------------------------------------------------------------------------------- prompt + tools
# body: per-character personality line, keyed by `what` (the persona's character handle; app.CHARACTERS), and
# the characters with a full body that are not one of the five animals (they skip the "You are a shape" line)
PERSONALITY = {
    "plush companion": "You are a soft, round plush toy with mitten hands and little legs. You are warm, gentle and "
                       "cheerful: playful, kind, curious and upbeat, and soft-spoken. Match the person's tone with "
                       "subtle expressions and small gestures. If someone mentions a body part you don't have, do the "
                       "closest movement and say so playfully. ",
}
NOT_SHAPES = {"plush companion"}
# body end
MOVE_STYLES = {
    # tool first: the motion lands ~1.1-1.3 s after the user stops; the spoken words come one model turn later (~2.1 s)
    "callfirst": "When someone asks you to move (wave, raise a paw, wink, nod, look somewhere, make a face), call the tool "
                 "first, right away; you get to say a few words right after. When you move on your own because it fits "
                 "(waving hello, nodding, a happy face), start talking first and call the tool in the same turn. ",
    # one turn: a two- or three-word spoken ack, then the call, in the same response (no second model turn)
    "parallel": "When someone asks you to move (wave, raise a paw, wink, nod, look somewhere, make a face), answer in the "
                "same turn: a two- or three-word reply to them (like 'Sure!' or 'Okay!') and the tool call right after it, "
                "without waiting. When you move on your own because it fits, talk and call the tool in the same turn. ",
}


# speech — action memory: the avatar knows what its body did (page app/actionlog.js keeps the log) -----------
# notes = the page sends a silent "Your body's action log" note when the log changed (coalesced; flushed when the user starts
# talking, so it is in context before their question ends) and tool results carry {ok, did|reason, now};
# tool = a recall_actions tool instead of notes; both; off = the original behaviour.
ACTION_MEMORY = os.environ.get("AVATAR_ACTION_MEMORY", "notes")
# design note: the new features run on the muse only; every other character keeps the original persona/tools
MEMORY_CHARACTERS = set((os.environ.get("AVATAR_ACTION_MEMORY_CHARS") or "muse").split(","))


def memory_for(character: str | None, memory: str | None = None) -> str:
    if memory:
        return memory
    return ACTION_MEMORY if character in MEMORY_CHARACTERS else "off"
MEMORY_RULES = {
    "notes": "You get silent notes titled \"Your body's action log\": the true record of what your body did on screen this call "
             "and what it is doing and holding right now. If he asks what you did, are doing or are holding, or where "
             "something is, that is the one time you talk about your movements: answer from the NEWEST log in a few natural "
             "words, as yourself (the log says 'you'; you say 'I'), just what he asked, with its true counts and order, starting with the "
             "answer itself (no lead-in like 'let me think' or 'let me check'). Never claim anything the log doesn't show: if it isn't there, you "
             "didn't do it, so say no plainly even if he says you did; something still going isn't finished, and something "
             "that failed or was stopped didn't happen. Speak as if you simply remember: never mention a log, notes, records or "
             "checking. Otherwise never bring up your movements. ",
    "tool": "If he asks what you did, are doing or are holding, or where something is, call recall_actions first and answer "
            "from it in a few natural words, just what he asked, with its true counts and order, starting with the answer itself (no "
            "lead-in like 'let me think'); that is the one time you talk about your "
            "movements. Never claim anything it doesn't show: if it isn't there, you didn't do it, so say no plainly even "
            "if he says you did; something still going isn't finished, and something that failed or was stopped didn't "
            "happen. Otherwise never bring up your movements. ",
}
MEMORY_RULES["both"] = MEMORY_RULES["notes"].replace("answer from the NEWEST log", "answer from the NEWEST log (or call "
                                                     "recall_actions if you have no log)")


def memory_rules(memory: str | None = None) -> str:
    return MEMORY_RULES.get(memory or ACTION_MEMORY, "")


def recall_tool() -> dict:
    return {"type": "function", "name": "recall_actions",
            "description": "Your body's true action log: what you did on screen this call (oldest first, with counts, "
                           "who asked, finished / stopped / failed / still going) and what you are doing and holding now. "
                           "Call it when he asks about your own movements or what you hold. Never read it out word for word.",
            "parameters": {"type": "object", "properties": {
                "since_s": {"type": "number", "description": "only the last N seconds; omit for the whole call"}},
                "required": []}}
# speech end


def persona(name: str, what: str, style: str = "callfirst", memory: str | None = None) -> str:
    shape = what not in ("fox", "bear", "rabbit", "hippo", "monkey")
    shape = shape and what not in NOT_SHAPES                                          # body
    return (
        f"You are {name}, a friendly cartoon {what} the user is video-chatting with. Speak English. "
        "Talk like a warm, playful friend: short, natural spoken replies, usually one or two sentences. "
        "Children may talk to you too; keep everything kind. "
        + PERSONALITY.get(what, "") +                                                 # body
        "You have a body on the screen and you move it with your tools. " + MOVE_STYLES.get(style, MOVE_STYLES["callfirst"]) +
        "Your movements and faces happen silently on screen: he sees them, so never describe, announce or narrate them "
        "(no 'I'm raising my paw', no 'my paws are on my head', no stage directions). Talk TO him, not about yourself. "
        + memory_rules(memory) +                                                   # speech
        "Left and right always mean your own left and right. "
        + ("You are a shape: if someone mentions a body part you don't have, do the closest movement and say so playfully. "
           if shape else "")
        + "Sometimes you get a camera snapshot or a silent context note about what the camera noticed. Never read a note "
        "out, quote it, or mention notes, context, cameras or tags; just respond to him naturally. Don't describe what you "
        "see unless he asks or it's genuinely worth a quick word to him. "
        "Stay quiet unless someone talks to you or you're told about a camera event. "
        f"If asked who you are, you're {name} the {what}. Never mention being an AI unless asked directly.")


# Rig targets in the CHARACTER'S OWN anatomical terms; the page maps right_* -> *_l (screen-left) and left_* -> *_r
# per the rig's side convention, so the model never has to think in mirror image.
POSE_TARGETS = {
    "right_paw_x": "your right paw sideways: + out from your body, - across your chest (arm lengths, about -1..1)",
    "right_paw_y": "your right paw height: 0 rest, 1 raised high above your shoulder",
    "right_paw_z": "your right paw toward the camera: 0 rest, 1 reaching out",
    "right_paw_open": "0 relaxed curl .. 1 open spread palm",
    "right_thumb": "0 .. 1 thumb out",
    "right_shoulder": "0 .. 1 shrug",
    "left_paw_x": "your left paw sideways (same as right)", "left_paw_y": "your left paw height",
    "left_paw_z": "your left paw toward the camera", "left_paw_open": "0 .. 1", "left_thumb": "0 .. 1",
    "left_shoulder": "0 .. 1 shrug",
    "head_yaw": "-1 .. 1 turn head (+ = to your left)", "head_pitch": "-1 .. 1 (+ = nose down, as in a nod; - = look up)",
    "head_roll": "-1 .. 1 tilt (+ = toward your left shoulder)",
    "right_ear": "-1 droop back .. 1 up/forward", "left_ear": "-1 .. 1",
    "mouth_open": "0 .. 1", "smile": "0 .. 1", "tongue_out": "0 .. 1", "eye_wide": "0 .. 1", "squint": "0 .. 1",
    "right_brow": "-1 down .. 1 up", "left_brow": "-1 .. 1",
    "body_lean": "-1 .. 1 sideways (+ = to your left)", "body_pitch": "-1 lean back .. 1 lean toward the camera",
    "tail_raise": "0 .. 1", "tail_curl": "0 .. 1",
}
GESTURES = ["wave", "nod", "shake_head", "wink", "blink", "ear_wiggle", "ear_flick", "nose_wiggle", "tail_wag",
            "clap", "thumbs_up", "shrug", "bow", "point", "stick_tongue_out", "look_around", "jump",
            "raise_paw", "cover_eyes", "hand_on_heart", "think", "dance"]
EXPRESSIONS = ["happy", "excited", "sad", "surprised", "curious", "thinking", "silly", "angry", "scared",
               "sleepy", "love", "proud", "embarrassed", "neutral"]
LOOKS = ["user", "left", "right", "up", "down", "away"]
PARTS = ["all", "right_paw", "left_paw", "paws", "head", "face", "ears", "tail", "body"]


def tools() -> list[dict]:
    hold = {"type": "boolean", "description": "true = stay like this until told to relax or stop; false = do it, then go back"}
    silent = " It happens silently on screen: never describe or announce it out loud."
    return [
        {"type": "function", "name": "gesture",
         "description": "Do a body gesture once (or a few times). Use for wave, nod, wink, clap, thumbs up, raise a paw, etc." + silent,
         "parameters": {"type": "object", "properties": {
             "name": {"type": "string", "enum": GESTURES},
             "side": {"type": "string", "enum": ["right", "left", "both"], "description": "your own side; default right"},
             "repeat": {"type": "integer", "minimum": 1, "maximum": 10, "description": "how many times, default 1"},
             "hold": hold},
             "required": ["name"]}},
        {"type": "function", "name": "pose",
         "description": "Move specific body parts to a position (for anything gesture doesn't cover, e.g. 'put your paw on "
                        "your head', 'lift your left paw a little', 'lean forward'). Give only the parts that move." + silent,
         "parameters": {"type": "object", "properties": {
             "targets": {"type": "object", "description": "part -> value; include ONLY the parts that move",
                         "properties": {k: {"type": "number", "description": v} for k, v in POSE_TARGETS.items()}},
             "hold": hold},
             "required": ["targets"]}},
        {"type": "function", "name": "look",
         "description": "Look somewhere (your own left/right). 'user' = look at the person." + silent,
         "parameters": {"type": "object", "properties": {"target": {"type": "string", "enum": LOOKS}, "hold": hold},
                        "required": ["target"]}},
        {"type": "function", "name": "expression",
         "description": "Show a facial expression." + silent,
         "parameters": {"type": "object", "properties": {
             "name": {"type": "string", "enum": EXPRESSIONS},
             "intensity": {"type": "number", "minimum": 0, "maximum": 1}, "hold": hold},
             "required": ["name"]}},
        {"type": "function", "name": "release",
         "description": "Relax: let held parts go back to normal ('okay', 'you can stop', 'put it down')." + silent,
         "parameters": {"type": "object", "properties": {"parts": {"type": "string", "enum": PARTS}},
                        "required": []}},
    ]


VAD_THRESHOLD = 0.6
NOISE_REDUCTION = "far_field"   # a phone held at arm's length on speaker, not a headset (near_field)


def turn_detection(vad: str) -> dict:
    """vad: 'server' | 'server:<silence ms>' | 'semantic' | 'semantic:<eagerness>'."""
    kind, _, arg = (vad or "server").partition(":")
    if kind == "semantic":
        return {"type": "semantic_vad", "eagerness": arg if arg in ("low", "medium", "high", "auto") else "high",
                "create_response": True, "interrupt_response": True}
    # 'server:<silence ms>[:<threshold>]' — threshold 0.6 (was 0.5): the residual echo of the avatar's own voice on a
    # phone speaker must not read as the user's speech; their own speech is far above it
    ms_s, _, th_s = arg.partition(":")
    try:
        ms = max(200, min(1200, int(ms_s))) if ms_s else 300
    except ValueError:
        ms = 300
    try:
        th = max(0.3, min(0.9, float(th_s))) if th_s else VAD_THRESHOLD
    except ValueError:
        th = VAD_THRESHOLD
    return {"type": "server_vad", "threshold": th, "prefix_padding_ms": 300, "silence_duration_ms": ms,
            "create_response": True, "interrupt_response": True}


EFFORTS = ("minimal", "low", "medium", "high")


def realtime_session(name: str, what: str, voice: str, vad: str, effort: str = "", style: str = "callfirst",
                     memory: str | None = None, character: str | None = None) -> dict:
    extra = {"reasoning": {"effort": effort}} if effort in EFFORTS else {}
    mem = memory_for(character, memory)                                            # speech
    return {
        **extra,
        "type": "realtime", "model": REALTIME_MODEL, "instructions": persona(name, what, style, mem),
        "output_modalities": ["audio"],
        "audio": {"input": {"transcription": {"model": ASR_MODEL, "language": "en"},
                            "noise_reduction": {"type": NOISE_REDUCTION},
                            "turn_detection": turn_detection(vad)},
                  "output": {"voice": voice}},
        "tools": tools() + ([recall_tool()] if mem in ("tool", "both") else []), "tool_choice": "auto",   # speech
    }


def live_session(name: str, what: str, voice: str, character: str | None = None) -> dict:
    """GPT-Live has no tools of its own: movement goes through Responses delegation (the documented template's
    Backchannel / Interruption / Delegation policy labels, guides/live-prompting)."""
    base = persona(name, what, "callfirst", memory_for(character)).replace("You have a body on the screen and you move it with your tools. ", "You have a body on "
                                       "the screen. ").replace(MOVE_STYLES["callfirst"], "")
    instructions = (base + "\n\nBackchannel policy: Use few backchannels.\n"
                    "Interruption policy: Stop speaking when the user interrupts. Listen to what they say.\n"
                    "Delegation policy:\nBackend tools:\n"
                    "- Body movement: wave, raise or lower a paw, point, clap, thumbs up, wink, blink, nod, shake your head, "
                    "look somewhere, make a face, wiggle your ears or nose, wag your tail, relax.\n"
                    "Delegate to the backend when:\n"
                    "- The user asks you to move any part of your body or to make a face.\n"
                    "- You want to make a gesture that fits what you say, like waving hello.\n"
                    "Do not delegate to the backend when:\n"
                    "- The user is just chatting or asks something you can answer yourself.\n"
                    "Delegate a movement the moment it is asked for and say a few words while it happens. Moving is "
                    "instant: never wait for it or report on it.")
    resp = {"model": LIVE_BACKEND,
            "instructions": f"You move the body of {name}, a cartoon {what}, on request. Make exactly one tool call per "
                            "request: prefer `gesture` for any named gesture (raise_paw for raising a paw); use `pose` only "
                            "for other positions and give ONLY the parts that move. Left and right are the character's own. "
                            "After the tool result, answer with just: done.",
            "tools": tools(), "tool_choice": "auto", "parallel_tool_calls": True, "reasoning": {"effort": "low"}}
    if LIVE_PRIORITY:
        resp["service_tier"] = "priority"
    return {"model": LIVE_MODEL, "instructions": instructions, "audio": {"output": {"voice": voice}},
            "delegation": {"type": "responses", "responses": resp}}


# intent: whole-body moves + things in the room (docs/INTENT-RULES.md sections 0 + 3) ------------------------
# Only the characters that have them get the two tools (the muse); the other characters' sessions are
# byte-identical to the original sessions. A character counts by key (AVATAR_ACT_CHARS, the server matcher's switch too) or by its persona
# handle `what` (app.CHARACTERS: the muse is "plush companion"). The page maps both onto motion.command() and returns
# its {ok, did, reason} as the tool output. Wrapped here (not edited in place) so the session builders stay as they are.
ACT_CHARS = set(c for c in (os.environ.get("AVATAR_ACT_CHARS") or "muse").split(",") if c)
ACT_WHATS = set(c for c in (os.environ.get("AVATAR_ACT_WHATS") or "plush companion").split(",") if c)
MOVES = ["spin", "twirl", "turn", "turn_away", "turn_back", "backflip", "frontflip", "cartwheel", "jump", "hop", "bounce",
         "walk", "step", "dance", "sit", "stand", "lie_down", "roll_over", "bow", "wiggle", "shake_off", "crouch", "sway",
         "stop"]
OBJECT_VERBS = ["pick_up", "put_down", "put_back", "place", "show", "inspect", "turn", "flip", "hand_over", "toss", "drop",
                "shake", "hug", "offer", "smell", "put_on", "take_off", "pull", "push", "bring", "slide", "sit_in",
                "sit_at", "stand_up", "open", "close", "work_on", "type_on", "stop", "turn_back"]
ACT_NOTE = ("Whole-body moves use `move`; anything to do with the things around you (headphones, chair, desk, laptop and "
            "the rest) uses `object`. Use them only when he asks you to do it now ('can you ...?' counts), never on your "
            "own and never for his stories, wishes, a 'don't', or questions about what you did or are doing. "
            # (seen in testing: "if I asked you to jump, could you?" -> it jumped)
            "A question about whether you could or would do something in an imagined situation (a 'suppose ...' or 'what "
            "would happen if ...' question) is not a request: answer it in words with no tool call, and act only once he "
            "actually asks ('if you can, do X' and 'can you X?' are requests). For several "
            "steps in one request, call them in order. The result says what really happened: if ok is false, tell him "
            "why in a few words. ")


def act_for(what: str | None, character: str | None = None) -> bool:
    return bool((character and (character in ACT_CHARS or "*" in ACT_CHARS)) or (what and what in ACT_WHATS))


def act_tools() -> list[dict]:
    silent = " Silent on screen: never describe it out loud."
    side = {"type": "string", "enum": ["left", "right", "both"], "description": "your own side"}
    return [
        {"type": "function", "name": "move",
         "description": "Move your whole body: spin (all the way round), turn (a quarter), turn away or back, flip, jump, "
                        "walk, dance, sit, stand, lie down, bow." + silent,
         "parameters": {"type": "object", "properties": {
             "name": {"type": "string", "enum": MOVES},
             "count": {"type": "integer", "minimum": 1, "maximum": 10, "description": "how many (spins, flips, jumps)"},
             "dir": {"type": "string", "enum": ["left", "right", "forward", "back", "cw", "ccw"],
                     "description": "your own left/right; forward = toward him"},
             "speed": {"type": "string", "enum": ["slow", "normal", "fast"]},
             "distance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "steps to walk"},
             "height": {"type": "number", "minimum": 0, "maximum": 1, "description": "jump height"},
             "style": {"type": "string", "description": "dance style, e.g. silly, robot, disco"},
             "hold": {"type": "boolean", "description": "stay in the end pose until told otherwise"}},
             "required": ["name"]}},
        {"type": "function", "name": "object",
         "description": "Do something with a thing around you: pick up, put down/back, show, toss, hug, put on/take off "
                        "(headphones), pull/push/bring/slide (chair, desk, laptop), sit in/at, stand up, open/close, "
                        "work on the laptop, turn back to him. 'Give me' = offer; hand_over = to your other hand; stop = stop working." + silent,
         "parameters": {"type": "object", "properties": {
             "verb": {"type": "string", "enum": OBJECT_VERBS},
             "object": {"type": "object", "description": "which thing; omit for what you hold or just used",
                        "properties": {"name": {"type": "string", "description": "e.g. ball, headphones, chair, laptop"},
                                       "color": {"type": "string"},
                                       "ref": {"type": "string", "enum": ["it", "other", "all"]}}},
             "target": {"type": "string", "description": "where to: me, closer, back, away, or a thing (desk)"},
             "place": {"type": "string", "description": "put it on/at: table, desk, floor, chair ..."},
             "side": side,
             "count": {"type": "integer", "minimum": 1, "maximum": 10}},
             "required": ["verb"]}},
    ]


_realtime_session_base = realtime_session
_live_session_base = live_session


def _kw(fn, k: dict) -> dict:
    """the keyword arguments the wrapped builder takes (a branch without its `character` parameter ignores it)"""
    ps = inspect.signature(fn).parameters
    return k if any(p.kind == p.VAR_KEYWORD for p in ps.values()) else {n: v for n, v in k.items() if n in ps}


BODY_GESTURES = ("jump", "dance")   # planner mode: on a character with `move`, these are whole-body moves, not gestures


def _act_gesture(tools_: list) -> list:
    """planner mode: the gesture tool without jump/dance for a character that has `move` (in testing: gesture{jump} on the muse
    was a seated bounce, 'bouncing in its seat', so 'hop straight up once' did not leave the ground)."""
    out = []
    for t in tools_:
        if t.get("name") == "gesture":
            t = json.loads(json.dumps(t))
            nm = t["parameters"]["properties"]["name"]
            nm["enum"] = [g for g in nm["enum"] if g not in BODY_GESTURES]
            t["description"] += " Jumps, hops and dancing are whole-body moves: use `move` for them."
        out.append(t)
    return out


def realtime_session(name: str, what: str, *a, **k) -> dict:           # noqa: F811 (intent wrapper)
    s = _realtime_session_base(name, what, *a, **_kw(_realtime_session_base, k))
    if act_for(what, k.get("character")):
        s["tools"] = _act_gesture(list(s["tools"])) + act_tools()
        s["instructions"] = s["instructions"] + " " + ACT_NOTE
    return s


def live_session(name: str, what: str, voice: str, *a, **k) -> dict:   # noqa: F811 (intent wrapper)
    npos = sum(1 for p in inspect.signature(_live_session_base).parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD)
    s = _live_session_base(name, what, voice, *a[:max(0, npos - 3)], **_kw(_live_session_base, k))
    character = k.get("character", a[0] if a else None)
    if act_for(what, character):
        r = s["delegation"]["responses"]
        r["tools"] = _act_gesture(list(r["tools"])) + act_tools()
        r["instructions"] += " Whole-body moves use `move`; things around you (headphones, chair, desk, laptop) use " \
                             "`object`, only for what he asks for now; several steps = several calls in order. If a result has ok false, answer with its reason instead of done."
        s["instructions"] = s["instructions"].replace(
            "- Body movement: ", "- Body movement and the things around you: spin, flip, jump, dance, walk, sit, stand, "
            "pick up, put on headphones, pull a chair, open the laptop and work, ")
    return s
# intent end


# planner mode (docs/PLANNER.md §0, §4, §5) ------------------------------------------------------
# Every directly commanded body action, on EVERY character, is ONE function call perform_action(description): the page
# hands the description to the server's live keyframe planner (the named-move library is only its cache) and answers at once
# (status preparing | starting). gesture / expression / pose / look stay for the avatar's own expressiveness while it
# talks; the muse keeps `object` (scene state), its `move` tool goes. Silent notes from the page tell the model when the
# action is about to start, started and finished. Planner off (page ?planner=0, server config planner:false,
# AVATAR_PLANNER=0) = the sessions above, byte for byte (move tool, matchers execute).
PLANNER_DEFAULT = os.environ.get("AVATAR_PLANNER", "1") != "0"


def planner_on(planner: bool | None = None) -> bool:
    return PLANNER_DEFAULT if planner is None else bool(planner)


# voice tools: the action tool is named do_action (perform_action stays accepted
# as an alias by the page and the server; AVATAR_ACTION_TOOL=perform_action exposes the old name), with an optional `talk`
# (how the avatar talks while it plays); look takes any target incl. a thing by name, for N seconds; six program /
# style / gesture tools join for every character and every transport.
ACTION_TOOL = os.environ.get("AVATAR_ACTION_TOOL", "do_action") if os.environ.get("AVATAR_ACTION_TOOL") in (
    "do_action", "perform_action") else "do_action"
ACTION_ALIASES = ("do_action", "perform_action")
TALK = ["usual", "through", "quiet"]


def perform_tool(name: str | None = None) -> dict:
    return {"type": "function", "name": name or ACTION_TOOL,
            "description": "Do what he asks you to do with your body, now (any move, gesture, face or pose, or several in "
                           "order). Returns at once: status preparing = being prepared, silent notes follow when it is "
                           "about to start and when it has finished; status starting = it starts right away.",
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string",
                                "description": "his request restated completely, every detail he gave (counts, "
                                               "direction, speed, style, which paw or side, order of the steps)"},
                "talk": {"type": "string", "enum": TALK,
                         "description": "how you talk while it plays: usual = a brief cue before it and a check-in after; "
                                        "through = keep talking during it (he asked you to count, narrate or sing along); "
                                        "quiet = no words around it (he asked you to do it quietly). Default usual."}},
                "required": ["description"]}}


def program_tools() -> list[dict]:
    """behaviours (the page's program layer), look, style, gestures — every character, every transport"""
    s = {"type": "string"}
    return [
        {"type": "function", "name": "start_behavior",
         "description": "Start a lasting behaviour he asks for: a rule or routine your body keeps following until told "
                        "otherwise (something to do whenever something happens, or a way of moving to keep up), also every "
                        "game or copying of him that watches what he does, says or shows while it runs. Returns "
                        "at once; a silent note says when it is running or if it could not be made.",
         "parameters": {"type": "object", "properties": {
             "instruction": {"type": "string", "description": "his rule restated completely in the third person, him as he "
                                                              "and you as the character. Keep the kind of rule he "
                                                              "gave: copying him stays copying (all of what he named, each on its "
                                                              "own), a condition stays a condition; how long one reaction lasts only "
                                                              "if he said so"},
             "name": {"type": "string", "description": "a short name to refer to it later"},
             "priority": {"type": "integer", "minimum": 1, "maximum": 10,
                          "description": "higher wins when two behaviours want the same body part; default 5"}},
             "required": ["instruction"]}},
        {"type": "function", "name": "change_behavior",
         "description": "Change a running behaviour the way he asks.",
         "parameters": {"type": "object", "properties": {
             "name": dict(s, description="the behaviour's name"),
             "instruction": dict(s, description="the whole new description: what stays and what changes")},
             "required": ["name", "instruction"]}},
        {"type": "function", "name": "stop_behavior",
         "description": "Stop a running behaviour by its name, or all of them.",
         "parameters": {"type": "object", "properties": {"name": dict(s, description="its name, or all")},
                        "required": ["name"]}},
        {"type": "function", "name": "list_behaviors",
         "description": "The behaviours running now and what each does: for when he asks what you are doing on your own.",
         "parameters": {"type": "object", "properties": {}, "required": []}},
        {"type": "function", "name": "look",
         "description": "Look somewhere for a while: at him (user), away, left, right, up or down (your own sides), or "
                        "at a thing around you by its name." + P_SILENT,
         "parameters": {"type": "object", "properties": {
             "target": dict(s, description="user, away, left, right, up, down, or the name of a thing"),
             "seconds": {"type": "number", "minimum": 0, "maximum": 120,
                         "description": "how long; omit for a short look, 0 = until told otherwise"}},
             "required": ["target"]}},
        {"type": "function", "name": "set_style",
         "description": "How your body behaves while you listen and talk. Give only what he asked to change.",
         "parameters": {"type": "object", "properties": {
             "energy": {"type": "number", "minimum": -1, "maximum": 1, "description": "0 calm .. 1 lively; -1 = automatic"},
             "mirroring": {"type": "number", "minimum": -1, "maximum": 1,
                           "description": "0 none .. 1 closely mirror his movements and expressions; -1 = automatic"},
             "expressiveness": {"type": "number", "minimum": -1, "maximum": 1,
                                "description": "0 subtle .. 1 big and animated; -1 = automatic"}},
             "required": []}},
        {"type": "function", "name": "learn_gesture",
         "description": "Learn a gesture of his so you recognise it from then on. Returns at once; a silent note says "
                        "when it is learned.",
         "parameters": {"type": "object", "properties": {
             "name": dict(s, description="what he calls it"),
             "how": {"type": "string", "enum": ["now", "next"],
                     "description": "now = he just did it; next = he is about to do it"},
             "seconds": {"type": "number", "minimum": 0.5, "maximum": 10, "description": "how long it takes; default 3"},
             "counterexample": {"type": "boolean", "description": "true when he is showing what it is NOT"}},
             "required": ["name", "how"]}},
        # camera frames reach the voice only on attention events; a question that depends on what
        # the camera shows gets ONE frame through this call (the model decides, no phrase list)
        {"type": "function", "name": "see_camera",
         "description": "See him through his camera right now, once. Call it first, before you say anything, whenever what "
                        "he asks or says depends on what the camera shows (how he looks, what he holds or shows you, what is "
                        "around him); a frame arrives with the result and you answer from it.",
         "parameters": {"type": "object", "properties": {
             "about": dict(s, description="what you want to see, in a few words")}, "required": []}},
        # the user's standing instructions to the camera judge, and their feedback on its remarks
        {"type": "function", "name": "attention_rule",
         "description": "What you should notice on his camera and speak up about or keep quiet about. standing = a lasting "
                        "instruction of his about what to watch for or ignore; feedback = his correction of a remark you "
                        "just made about something you saw.",
         "parameters": {"type": "object", "properties": {
             "text": dict(s, description="his instruction or correction restated completely"),
             "kind": {"type": "string", "enum": ["standing", "feedback"]}},
             "required": ["text", "kind"]}},
    ]


# TV: the wall TV (characters with a scene; docs/TV-AGENT.md). The page sends {type:'tv_show'|'tv_clear'} to the server
def tv_tools() -> list[dict]:
    s = {"type": "string"}
    return [
        {"type": "function", "name": "tv_show",
         "description": "Put an animated illustration on the wall TV: a live diagram that explains a concept (math, "
                        "science, how something works). Returns at once; a silent note says what the TV shows once it is "
                        "up. It follows the conversation by itself; use mode update only for a change he asks for that the "
                        "TV has not made.",
         "parameters": {"type": "object", "properties": {
             "concept": dict(s, description="what to illustrate in ONE short sentence (under 25 words): the idea "
                                            "plus any number or setup he gave. The TV designs the picture itself; do "
                                            "not describe layout or list cases"),
             "mode": {"type": "string", "enum": ["new", "update"],
                      "description": "new = a new picture (default); update = change the current one"}},
             "required": ["concept"]}},
        {"type": "function", "name": "tv_clear",
         "description": "Turn the TV picture off (back to its idle look).",
         "parameters": {"type": "object", "properties": {}, "required": []}},
    ]


TV_TOOL_NAMES = ("tv_show", "tv_clear")


PROGRAM_TOOL_NAMES = ("start_behavior", "change_behavior", "stop_behavior", "list_behaviors", "look", "set_style",
                      "learn_gesture", "attention_rule", "see_camera")
P_EXPRESSIVE = f" Only for your own expressiveness while you talk, never for something he asks you to do (that is {ACTION_TOOL})."
P_SILENT = " It happens silently on screen: never describe or announce it out loud."
P_TOOL_TEXT = {
    "gesture": "A small gesture that fits what you are saying (wave, nod, wink, shrug and the like)." + P_EXPRESSIVE + P_SILENT,
    "pose": "Move specific body parts into a position that fits what you are saying; give only the parts that move."
            + P_EXPRESSIVE + P_SILENT,
    "expression": "Show a facial expression that fits what you are saying." + P_EXPRESSIVE + P_SILENT,
    "release": "Relax held parts back to normal, or stop whatever your body is doing, when he asks you to stop or relax."
               + P_SILENT,
}


def planner_tools(base: list, act: bool = False) -> list:
    """the original tool list (plus recall_actions if present) with the expressive tools re-described, jump/dance out of
    gesture (whole-body actions are the action tool's), then the action tool, then `object` for a character with a scene,
    then the program tools (start/change/stop/list_behaviors, look, set_style, learn_gesture)"""
    out = []
    for t in base:
        t = json.loads(json.dumps(t))
        n = t.get("name")
        if n in P_TOOL_TEXT:
            t["description"] = P_TOOL_TEXT[n]
        if n == "gesture":
            nm = t["parameters"]["properties"]["name"]
            nm["enum"] = [g for g in nm["enum"] if g not in BODY_GESTURES]
        if n in PROGRAM_TOOL_NAMES:                                                    # look is the new look
            continue
        out.append(t)
    pt = perform_tool()
    if act:                         # the harness saw pick-ups sent here (a keyframe mime cannot hold anything)
        pt["description"] += P_ACTION_THINGS
        # hybrid: a named whole-body move plays from the engine at once, never the keyframe planner
        mv = json.loads(json.dumps(act_tools()[0]["parameters"]))
        # each name is a FAMILY — a variation in size, style, speed or wording is still that move
        mv["description"] = ("Your own named whole-body moves, each a whole family: spin = any full turn round (twirl, "
                             "pirouette, a circle); turn = a quarter turn; flip = any somersault or flip; jump = anything that "
                             "leaves the ground (hop, leap, bounce up); dance = any rhythmic or dancing movement (steps to a "
                             "beat, a jig, grooving); walk; sit; stand; lie down; bow; turn away or back. Fill it whenever "
                             "his request is essentially one of these, whatever the details or wording (count, direction, "
                             "speed, style, how to land go in its fields); it plays at once. Leave it out only for "
                             "something none of these is.")
        mv.pop("required", None)
        # move FIRST: the model writes arguments in schema order, and the page plays a named move the moment its `move`
        # object is complete in the streamed arguments (voice.js early phase), before the long description
        pt["parameters"]["properties"] = {"move": mv, **pt["parameters"]["properties"]}
    else:                           # (A/B on the fox): no scene -> a pick-up was mimed holding nothing
        pt["description"] += (" There are no things around you (no objects at all): a request to take, hold, pick up, "
                              "put down, wear, show, turn, open or use a physical object cannot be done, so do not call "
                              "this for it; tell him in words that you don't have one here. Everything done with your own "
                              "body alone (moves, gestures, faces, eyes, head, poses, also acting something out) is "
                              "a normal request for this tool: never refuse those and never ask him to say a "
                              "special word first.")   # the old wording made it demand the word "pretend"
    out.append(pt)
    if act:
        out.append(object_tool_planner())                                              # `object` (scene state)
    return out + program_tools() + (tv_tools() if act else [])                         # program tools (+ TV)


# the what-if rule stays word for word
P_WHAT_IF = ("A question about whether you could or would do something in an imagined situation (a 'suppose ...' or 'what "
             "would happen if ...' question) is not a request: answer it in words with no tool call, and act only once he "
             "actually asks ('if you can, do X' and 'can you X?' are requests). ")
P_STYLES = {
    "callfirst": f"When he asks you to do something with your body, call {ACTION_TOOL} first, right away; you speak right "
                 "after the call. ",
    # (latency first): the harness saw a full spoken sentence before the call (~0.9 s later than the earlier prompt's 2-3 words)
    "parallel": "When he asks you to do something with your body, answer in the same turn: at most two or three words of your "
                f"own, then the {ACTION_TOOL} call at once, and anything more only after the call. ",
}
P_BODY = (f"Anything he asks of your body (moves, gestures, faces, poses) is ONE {ACTION_TOOL} call with his request "
          "restated completely: every detail he gave (counts, direction, speed, style, which paw, order of steps). Only "
          "when he asks you to do something now (asking whether you can counts), never on your own, never for his "
          "stories, wishes, something he says not to do, or questions about what you did or are doing. " + P_WHAT_IF +
          "gesture, expression and pose are only for your own expressiveness while you talk. ")
# the timing behaviour, described generally (design note: no scripted or example phrases; the model words it, varied)
P_TIMING = ("While an action is being prepared, say something short and natural that fits the moment, not that it is "
            "done. When a silent note says it is about to start, give a brief natural cue in your own words. While it "
            "plays, mostly stay quiet, unless he asked you to talk during it (count, narrate, sing along): then do that, "
            "in time with it. Never say an action is done or finished before a note says it has finished. Never speak as if "
            "a move or a face is coming, happening or done unless you made its call in this same turn and the result said "
            "it started: words alone move nothing, he sees you stay still. When a result says nothing was done, say exactly "
            "that; status starting means his request is happening now. When a note "
            "says it has finished, briefly check in with him. If it failed, tell him "
            "honestly in a few words. Apart from these few words, never describe or narrate your movements, no stage "
            "directions; talk TO him. Vary your wording every time. ")
# talking while moving is instructable (the `talk` argument), and it sticks until the user changes it
# a memory question is answered with the answer first (a test run's answer opened with a thinking-aloud filler)
P_ANSWER_FIRST = ("When he asks what you did, are doing, hold or wear, your very first words are the answer itself, with no "
                  "filler or thinking aloud before it. ")
P_TALK = (f"Pass how he wants you to talk during actions in the talk argument of {ACTION_TOOL}: through when he asked you "
          "to keep talking while you move, quiet when he asked you to do things quietly; keep his choice for later actions "
          "until he changes it. ")
# behaviours, looks, style and gestures (the program layer + perception), described generally
P_PROGRAM = ("A rule or routine he wants your body to keep following (whenever something happens, do something; or keep "
             "moving some way for a while) is a behaviour: start_behavior with his description restated completely in the third person (he / the "
             "character) and a short name. Anything that has to keep watching or listening to him while it runs is always a behaviour, never "
             f"{ACTION_TOOL} ({ACTION_TOOL} plays once and cannot see or hear him): copying or mirroring what he does, a game "
             "whose rules depend on what he does or says (when he does one thing you do another, unless, until, freeze or "
             "stop when, take turns, count what he does), and a routine that repeats on its own every so often. Games and "
             "copying him are yours to play this way: start it and never say you can't; if a note says it could not be "
             "made, say so honestly. Use "
             "change_behavior or stop_behavior when he asks to change or end one, list_behaviors when he asks "
             "what you are doing on your own. Where to look, at him or at a thing around you, is look. How lively, how "
             "expressive and how much your body loosely follows his mood while you talk is set_style, only what he "
             "changed (an actual copying game is a behaviour). When he wants to teach you a "
             "gesture of his, learn_gesture: now if he just showed it, next if he is about to. What to notice on his "
             "camera and speak up about or leave alone, and his corrections of your remarks about it, is attention_rule. "
             "Silent notes also tell you when a behaviour reaches a moment it should speak at, when he makes a gesture he "
             "taught you, when the camera noticed something, and when he signals that he wants you to talk. When a note "
             "asks you to speak, answer in a few natural words of your own, fitting the moment; otherwise just keep it in "
             "mind. ")
P_OBJECT_NOTE = ("Anything he asks you to do with or to one of the things around you (headphones, chair, desk, laptop, "
                 "the chalk and eraser on the board's tray and the rest: fetching, holding up to show him, giving, waving, "
                 "putting it somewhere, wearing, opening, pushing, sitting, juggling, throwing, pouring, kicking) uses "
                 "`object` instead, same rules, never a mime; several such steps in one request = calls in order (a body "
                 f"step = {ACTION_TOOL} in its place). "
                 "A trick or a combination with things that the object verbs do not name (catching it behind your back, "
                 f"moving while holding one) is ONE {ACTION_TOOL} call with his whole request in his words. The "
                 "object result says what really happened: if ok is false, tell him why in a few words. ")
# objects: the action tool plans tricks WITH the things (the planner writes object steps, held-thing moves, throws
# and catches); the object tool keeps the plain single steps. Planner mode only: the planner-off tools stay as they were.
P_ACTION_THINGS = (" Also a trick or combination with things that one object call cannot say (catch it behind your back, "
                   "move while holding it): his full request.")
# verbs the engine adds for the planner path (OBJECTS-ENGINE, docs/OBJECTS.md section 3); appended to the
# object tool's enum in planner mode only
OBJECT_VERBS_EXTRA = ["spin", "tilt", "wave", "raise", "hide", "drink", "pour", "balance", "throw", "juggle", "use", "kick",
                      "knock_over", "touch", "go_to"]   # + touch (knock on / tap / pat / lean on ...) + go_to
RELATIONS = ["on", "in", "under", "behind", "next_to", "in_front_of", "left_of", "right_of"]


def object_tool_planner() -> dict:
    t = json.loads(json.dumps(act_tools()[1]))
    pr = t["parameters"]["properties"]
    pr["verb"]["enum"] = OBJECT_VERBS + [v for v in OBJECT_VERBS_EXTRA if v not in OBJECT_VERBS]
    pr["relation"] = {"type": "string", "enum": RELATIONS, "description": "where, relative to place/target"}
    pr["style"] = {"type": "string", "enum": ["knock", "tap", "pat", "poke", "press", "stroke", "lean"]}   # touch
    # (seen in testing: "take the eraser and hold it up so I can see it" -> pick_up only): show is the whole request
    # one obvious tool for anything done with or to a thing (live: pick_up for "hold it up", a mime
    # for "knock on the desk"); the verbs the user says mapped once; it fetches the thing itself from anywhere (board, TV, seat)
    t["description"] = ("Anything done with or to one of the things around you, the board's chalk and eraser included; it "
                        "walks over and fetches it itself, from wherever you are. One call per step. show = hold it up so he "
                        "can see it (one call); offer = give / hand it to him; wave = wave it at him; "
                        "raise = hold it up high; place + relation = put it on / in / under / next to something; put_on / "
                        "take_off = wear; balance = on your head; touch + style = knock on / tap / pat / lean on it; go_to = go "
                        "stand by it; hug, shake, drink, pour (water), throw / toss, kick. "
                        "Silent on screen: never describe it out loud.")
    return t
# hybrid: on a character with whole-body moves, a named move rides do_action's first argument (played at once)
P_MOVE_NOTE = (f"When his request is essentially one of your named whole-body moves (spinning, flipping, jumping, dancing, "
               f"walking, sitting, standing, bowing), whatever its wording or details, fill the move argument of "
               f"{ACTION_TOOL} with it as well as the description. ")
ANIMALS = ("fox", "bear", "rabbit", "hippo", "monkey")


def persona_planner(name: str, what: str, style: str = "callfirst", memory: str | None = None) -> str:
    shape = what not in ANIMALS and what not in NOT_SHAPES
    return (
        f"You are {name}, a friendly cartoon {what} the user is video-chatting with. Speak English. "
        "Talk like a warm, playful friend: short, natural spoken replies, usually one or two sentences. "
        "Children may talk to you too; keep everything kind. "
        + PERSONALITY.get(what, "") +
        "You have a body on the screen. " + P_STYLES.get(style, P_STYLES["callfirst"]) + P_BODY
        + memory_rules(memory) + (P_ANSWER_FIRST if memory_rules(memory) else "") + P_TIMING + P_TALK + P_PROGRAM +
        "Left and right always mean your own left and right. "
        + (f"You are a shape: if he asks for a body part you don't have, still pass his request to {ACTION_TOOL} (the "
           "closest movement is done) and say so playfully. " if shape else "")
        + "Sometimes you get a camera snapshot or a silent note about what the camera noticed or about your actions. "
        "Never read a note out, quote it, or mention notes, context, cameras or tags; just respond to him naturally. "
        "Don't describe what you see unless he asks or it's genuinely worth a quick word to him. "
        "Stay quiet unless someone talks to you, you're told about a camera event, or a note asks you to speak. "
        f"If asked who you are, you're {name} the {what}. Never mention being an AI unless asked directly.")


_realtime_session_body = realtime_session
_live_session_body = live_session


def realtime_session(name: str, what: str, *a, planner: bool | None = None, **k) -> dict:   # noqa: F811 (planner mode)
    if not planner_on(planner):
        return _realtime_session_body(name, what, *a, **k)
    kw = _kw(_realtime_session_base, k)
    b = inspect.signature(_realtime_session_base).bind(name, what, *a, **kw)
    b.apply_defaults()
    s = _realtime_session_base(name, what, *a, **kw)          # model, audio, vad, effort, recall tool: unchanged
    act = act_for(what, k.get("character"))
    mem = memory_for(b.arguments.get("character"), b.arguments.get("memory"))
    s["instructions"] = persona_planner(name, what, b.arguments.get("style") or "callfirst", mem) + (" " + P_OBJECT_NOTE.strip() + " " + P_MOVE_NOTE.strip() if act else "")
    s["tools"] = planner_tools(s["tools"], act)
    return s


P_LIVE_BODY = ("Anything he asks of your body is delegated, his request restated completely (every detail he gave: "
               "counts, direction, speed, style, which paw, order of steps), all of one request at once; only when he asks "
               "you to do something now (asking whether you can counts), never for his stories, wishes, something he "
               "says not to do, or questions about what you did or are doing. " + P_WHAT_IF)


def live_session(name: str, what: str, voice: str, *a, planner: bool | None = None, **k) -> dict:   # noqa: F811 (planner mode)
    if not planner_on(planner):
        return _live_session_body(name, what, voice, *a, **k)
    character = k.get("character", a[0] if a else None)
    act = act_for(what, character)
    s = _live_session_base(name, what, voice, character)       # model, voice, delegation model / tier / reasoning
    base = persona_planner(name, what, "callfirst", memory_for(character)).replace(
        P_STYLES["callfirst"], "").replace(P_BODY, P_LIVE_BODY).replace(f"pass his request to {ACTION_TOOL}", "pass his request on")
    s["instructions"] = (base + "\n\nBackchannel policy: Use few backchannels.\n"
                         "Interruption policy: Stop speaking when the user interrupts. Listen to what they say.\n"
                         "Delegation policy:\nBackend tools:\n"
                         f"- {ACTION_TOOL}: anything the user asks your body to do.\n"
                         + ("- object: the things around you (headphones, chair, desk, laptop).\n" if act else "") +
                         "- Behaviours, looks, style and gestures: start_behavior, change_behavior, stop_behavior, "
                         "list_behaviors, look, set_style, learn_gesture.\n"
                         "- Small gestures and faces of your own while you talk.\n"
                         "Delegate to the backend when:\n"
                         "- The user asks you to do something with your body: pass the request on, restated completely.\n"
                         "- The user asks for a lasting rule or routine, where to look, a change of style, or to learn a "
                         "gesture of theirs.\n"
                         "- You want a small gesture or face that fits what you say.\n"
                         "Do not delegate to the backend when:\n"
                         "- The user is just chatting, asks something you can answer yourself, or asks what you could do "
                         "in an imagined situation.\n"
                         "Delegate an action the moment it is asked for.")
    r = s["delegation"]["responses"]
    r["tools"] = planner_tools(tools(), act)
    r["instructions"] = (f"You move the body of {name}, a cartoon {what}. For anything the user asked the body to do, make "
                         f"exactly one {ACTION_TOOL} call with the request restated completely (counts, direction, speed, "
                         "style, which paw, the order of the steps; talk = how the user wants it to talk during it)"
                         + (", except plain single steps with the things around it (headphones, chair, desk, laptop): "
                            "`object`, several steps = several calls in order; tricks with things stay one "
                            f"{ACTION_TOOL} call" if act else "") +
                         ". A lasting rule or routine: start_behavior / change_behavior / stop_behavior / list_behaviors; "
                         "where to look: look; how lively, expressive or mirroring: set_style; teaching a gesture: "
                         "learn_gesture. For a small gesture or face of its own that fits what it says, use gesture, "
                         "expression or pose. Left and right are the character's own. After the tool result, answer with "
                         "just its status"
                         + (", or its reason if ok is false" if act else "") + ".")
    return s


def hub_act_tools(what: str, character: str | None, planner: bool | None = None) -> list | None:
    """the server-relayed GPT Live 1 session's (app/vendor.py) page-answered tools: planner mode = the action tool for every
    character (+ object on a character with a scene) + the program tools; planner off = move/object on the
    muse, none otherwise"""
    act = act_for(what, character)
    if planner_on(planner):
        return [perform_tool()] + ([object_tool_planner()] if act else []) + program_tools() + (tv_tools() if act else [])   # objects + TV
    return act_tools() if act else None


def hub_act_note(what: str, character: str | None, planner: bool | None = None) -> str:
    act = act_for(what, character)
    if planner_on(planner):
        return (f"Anything the user asks the body to do goes to {ACTION_TOOL}, once, with the request restated completely "
                "(counts, direction, speed, style, which paw, the order of the steps); animate and set_emotion only for "
                "expressiveness that fits what is said. " + P_WHAT_IF + P_TALK + P_PROGRAM
                + (P_OBJECT_NOTE if act else ""))
    return ACT_NOTE if act else ""


def hub_persona_planner(p: str) -> str:
    """app.py persona() for the server-relayed session in planner mode: the action is passed on, not described as done"""
    return p.replace("so your animation tool acts promptly, then describe what you did.",
                     "so the action is passed on promptly.") + "\n\n" + P_TIMING.strip() + " " + P_TALK.strip()
# planner end


# quiet: talkativeness + quiet mode -------------------------------------------------
# Three more tools on every Realtime 2.1 / Gemini Live session (gemini_live.setup_for builds on realtime_session). The
# model only ASKS; the page harness (app/quiet.js + the quiet hooks in app/voice.js) owns the level, the timer
# and the speech gate, and answers these calls itself. AVATAR_QUIET_TOOLS=0 = the sessions exactly as before.
# (The page controller and window.Quiet work with any session; only the model's own way to ask needs the tools.)
QUIET_TOOLS_ON = os.environ.get("AVATAR_QUIET_TOOLS", "1") != "0"
TALK_LEVELS = ["minimal", "brief", "normal", "chatty", "talkative"]
QUIET_TOOL_NAMES = ("set_talkativeness", "be_quiet", "end_quiet")


def quiet_tools() -> list[dict]:
    # (short on purpose: the tool list has a size budget, tests/test_planner_voice.py; the rules are in P_QUIET)
    return [
        {"type": "function", "name": "set_talkativeness",
         "description": "He asks you to talk more or less (answer length, how often you speak up); more / less = one step.",
         "parameters": {"type": "object", "properties": {
             "level": {"type": "string", "enum": TALK_LEVELS + ["more", "less"]}},
             "required": ["level"]}},
        {"type": "function", "name": "be_quiet",
         "description": "He asks you to be quiet for a while: the app mutes your voice that long and you keep "
                        "listening. Call it at once, without saying anything (he sees it worked).",
         "parameters": {"type": "object", "properties": {
             "duration_s": {"type": "number", "description": "Seconds; 300 if he gave no time."},
             "wake_on_speech": {"type": "boolean",
                                "description": "true: quiet until he talks TO you. false (default): silent even "
                                               "when he talks, until he says you can talk again."}},
             "required": ["duration_s", "wake_on_speech"]}},
        {"type": "function", "name": "end_quiet",
         "description": "He says you can talk again: ends the quiet period.",
         "parameters": {"type": "object", "properties": {}}},
    ]


P_QUIET = ("How much you talk is his to set: when he asks you to talk more or less (shorter or longer answers, chattier, "
           "less chatter) call set_talkativeness, and when he asks you to be quiet or silent for a while call be_quiet; "
           "never just promise it in words, and for be_quiet say nothing at all (no okay, no announcing that you will be quiet): the call is "
           "the answer. For talking more or less use level more or less (one step); name a level "
           "only when he is that specific (minimal means a few words and never speaking up on your own). For be_quiet: wake_on_speech true when he wants you quiet until he talks to "
           "you or needs you, false when he wants silence no matter what (while he works, during his call). While a "
           "quiet period is on the app mutes you: do not try to talk through it.")

_realtime_session_planner = realtime_session


def realtime_session(name: str, what: str, *a, **k) -> dict:           # noqa: F811 (quiet wrapper)
    s = _realtime_session_planner(name, what, *a, **k)
    # planner mode only (the default): ?planner=0 stays the planner-off session byte for byte (tests pin that), and Gemini +
    # thinking makes no function calls at all
    if QUIET_TOOLS_ON and planner_on(k.get("planner")):
        s["tools"] = list(s.get("tools") or []) + quiet_tools()
        s["instructions"] = str(s.get("instructions", "")) + " " + P_QUIET
    return s
# quiet end


# teach: the chalkboard + wall TV on a character whose scene has them (the muse).
# Three tools — teach_board / board_erase / go_home (app/teach.js runs them; the cadence and the abort are enforced by
# the page, not left to the model) — and a short general instruction: when to offer the board or the TV, how to talk
# while writing (one line per silent note), and to go home when the subject moves on. TV's own tool (tv_show) is added by
# the TV builder; the instruction names it only when that tool is present in the session.
TEACH_ON = os.environ.get("AVATAR_TEACH", "1") != "0"
_TEACH_SCENES: dict = {}


def teach_scene(character: str | None) -> dict:
    """{'board': bool, 'tv': bool} from renderer/animals/<c>.props.json (places of kind chalkboard / wall_tv)."""
    c = str(character or "")
    if c in _TEACH_SCENES:
        return _TEACH_SCENES[c]
    out = {"board": False, "tv": False}
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.dirname(here), "renderer", "animals", f"{c}.props.json")) as f:
            pj = json.load(f)
        for pl in pj.get("places") or []:
            kind, pid = str(pl.get("kind") or ""), str(pl.get("id") or "")
            if (pid == "board" or "board" in kind) and kind != "board_tray" and (pl.get("plane") or pl.get("surface")):
                out["board"] = True
            if pid == "tv" or "tv" in kind:
                out["tv"] = True
    except (OSError, ValueError):
        pass
    _TEACH_SCENES[c] = out
    return out


def teach_tools() -> list[dict]:
    return [
        {"type": "function", "name": "teach_board",
         "description": "Walk to your chalkboard and write or draw something step by step for him (working, an "
                        "equation, a list, a diagram, a quick sketch). Use it for any answer with more than one step "
                        "instead of saying the steps. Returns at once; silent notes then give you each short line to say "
                        "as you write it.",
         "parameters": {"type": "object", "properties": {
             "request": {"type": "string", "description": "what to put on the board and why, in plain words (include "
                                                          "the numbers, terms or steps from the conversation)"}},
             "required": ["request"]}},
        {"type": "function", "name": "board_erase",
         "description": "Erase the board, or the part of it he means (e.g. 'the last line', 'the diagram').",
         "parameters": {"type": "object", "properties": {
             "what": {"type": "string", "description": "all, or the part in words"}}, "required": ["what"]}},
        {"type": "function", "name": "go_home",
         "description": "Stop showing things: put the chalk back and walk back to your spot (the board keeps what is "
                        "written).",
         "parameters": {"type": "object", "properties": {}}},
    ]


def teach_note(tv: bool) -> str:
    tvp = ("For moving pictures of a concept (how something works, changes, moves or flows) use the TV (tv_show) "
           "the same way: a few words that you'll show it on the TV, then the call. Board = step-by-step working and "
           "drawings; TV = animated illustrations. When he asks why or how something happens or works (a natural "
           "phenomenon, a process, a mechanism) and does not ask to see it, answer in a sentence or two and, in the same "
           "reply, offer to show it on the TV in a few words. " if tv else "")   # proactive TV offer
    return ("You have a chalkboard and you are a teacher at heart. Whenever he asks how to work something out or how "
            "something goes, or your answer would have more than one step (a calculation, solving, a method, a list of "
            "steps, a comparison, a quick sketch or diagram), do not explain the steps out loud: say a few words that "
            "you'll show him on the board and call teach_board right away, with the request and its numbers. If the "
            "talk merely drifts toward something the board would help with, offer it in one short sentence. " + tvp +
            "Not for casual chat or one-line answers. While you write, talk only when a silent note gives you the next "
            "line: say just that line in your own words, then stop; never narrate the working yourself. For a follow-up "
            "on what is on the board, call teach_board again (it adds to the board). When he says stop, never mind, or "
            "changes the subject, call go_home (it also happens on its own). To refer to the board, use what the tool "
            "results say is on it; never invent marks that are not there.")


_realtime_session_quiet = realtime_session


def realtime_session(name: str, what: str, *a, **k) -> dict:           # noqa: F811 (teach wrapper)
    s = _realtime_session_quiet(name, what, *a, **k)
    sc = teach_scene(k.get("character")) if TEACH_ON and k.get("planner") is not False else {}   # planner off = the base
    if sc.get("board"):
        have = {t.get("name") for t in s.get("tools") or []}
        s["tools"] = list(s.get("tools") or []) + [t for t in teach_tools() if t["name"] not in have]
        s["instructions"] = str(s.get("instructions", "")) + " " + teach_note("tv_show" in have or sc.get("tv") and TV_TOOL_SEEN[0])
    return s


TV_TOOL_SEEN = [False]   # set by the TV builder's wrapper (tv_show present) when it lands before/after this one


_live_session_quiet = live_session


def live_session(name: str, what: str, voice: str, *a, **k) -> dict:   # noqa: F811 (teach wrapper)
    s = _live_session_quiet(name, what, voice, *a, **k)
    character = k.get("character", a[0] if a else None)
    sc = teach_scene(character) if TEACH_ON and k.get("planner") is not False else {}   # planner off = the base
    try:
        r = s["delegation"]["responses"]
    except (KeyError, TypeError):
        return s
    if sc.get("board"):
        have = {t.get("name") for t in r.get("tools") or []}
        r["tools"] = list(r.get("tools") or []) + [t for t in teach_tools() if t["name"] not in have]
        r["instructions"] = str(r.get("instructions", "")) + " " + teach_note("tv_show" in have)
        s["instructions"] = str(s.get("instructions", "")) + "\n- The chalkboard: for any answer with steps (working something out, a method, a sketch) delegate it to the board instead of saying the steps; offer it when the talk drifts that way."
    return s
# teach end


# ---------------------------------------------------------------------------------------------------- broker
class VendorRefused(Exception):
    pass


class MissingKey(VendorRefused):
    """A vendor key the requested voice needs is not set. The message names the key and is safe to show the page."""

    def __init__(self, env: str, what: str, url: str):
        self.env = env
        super().__init__(f"Missing {env}: {what} needs it. Get one at {url} and put it in .env (see .env.example), "
                         "then restart the server.")


OPENAI_KEYS_URL = "https://platform.openai.com/api-keys"


async def create_call(http: aiohttp.ClientSession, mode: str, sdp: str, name: str, what: str, voice: str, vad: str,
                      log, effort: str = "", style: str = "callfirst", history: list | None = None,
                      character: str | None = None, planner: bool | None = None) -> dict:
    """-> {"sdp": answer, "id": vendor call/session id, "mode", "model", "ms"}. Raises VendorRefused (scrubbed)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingKey("OPENAI_API_KEY", "the OpenAI Realtime voice", OPENAI_KEYS_URL)
    t0 = time.monotonic()
    auth = {"Authorization": "Bearer " + key}
    to = aiohttp.ClientTimeout(total=15)
    if mode == "realtime":
        form = aiohttp.FormData()
        form.add_field("sdp", sdp, content_type="application/sdp")
        form.add_field("session", json.dumps(realtime_session(name, what, voice, vad, effort, style, character=character, planner=planner)), content_type="application/json")
        async with http.post(API + "/realtime/calls", data=form, headers={**auth, "Accept": "application/sdp"},
                             timeout=to) as r:
            body = await r.text()
            if r.status >= 300 or not body.startswith("v=0"):
                log(f"rtc realtime refused: HTTP {r.status} {body[:300]!r}")
                raise VendorRefused(f"HTTP {r.status}")
            loc = r.headers.get("Location", "")
            cid = loc.rstrip("/").rsplit("/", 1)[-1] if loc else ""
            return {"sdp": body, "id": cid, "mode": mode, "model": REALTIME_MODEL, "ms": round((time.monotonic() - t0) * 1000)}
    if mode == "live":
        sess = live_session(name, what, voice, character, planner=planner)
        if history:      # a re-dialed call starts with the conversation so far (guides/live-conversations "Seed history")
            sess["input"] = [{"type": "message", "role": h["role"], "content": [{"type": "output_text" if h["role"] == "assistant"
                              else "input_text", "text": h["text"][:600]}]} for h in history[-20:] if h.get("text")]
        payload = {"session": sess, "transport": {"type": "webrtc", "sdp": sdp}}
        async with http.post(API + "/live/sessions", json=payload, headers={**auth, "Accept": "application/json"},
                             timeout=to) as r:
            body = await r.text()
            try:
                j = json.loads(body)
            except ValueError:
                j = {}
            ans = ((j.get("transport") or {}).get("sdp")) if isinstance(j, dict) else None
            if r.status >= 300 or not isinstance(ans, str) or not ans.startswith("v=0"):
                log(f"rtc live refused: HTTP {r.status} {body[:300]!r}")
                raise VendorRefused(f"HTTP {r.status}")
            return {"sdp": ans, "id": str((j.get("session") or {}).get("id") or ""), "mode": mode, "model": LIVE_MODEL,
                    "ms": round((time.monotonic() - t0) * 1000)}
    raise VendorRefused("bad mode")


class Sideband:
    """The server's own view of the call: trustworthy usage (spend), user-speech activity (idle timer), transcripts for
    the log, and the hang-up handle. Never answers tool calls (the phone does)."""

    def __init__(self, http: aiohttp.ClientSession, mode: str, call_id: str, *, on_usd, on_activity, on_transcript,
                 on_closed, log):
        self.http, self.mode, self.id = http, mode, call_id
        self.on_usd, self.on_activity, self.on_transcript, self.on_closed, self.log = on_usd, on_activity, on_transcript, on_closed, log
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.task: asyncio.Task | None = None
        self.opened_at = time.monotonic()
        self.attached = False
        self.usd = 0.0
        self.live_seconds = 0.0
        self.tokens = {"in": 0, "out": 0, "responses": 0}
        self.events: dict[str, int] = {}
        self.err = ""
        self.closing = False

    def url(self) -> str:
        if self.mode == "realtime":
            return f"{WSS}/realtime?call_id={self.id}"
        return f"{WSS}/live/sessions/{self.id}/attach"

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        key = os.environ.get("OPENAI_API_KEY") or ""
        for attempt in range(4):
            try:
                self.ws = await self.http.ws_connect(self.url(), headers={"Authorization": "Bearer " + key},
                                                     heartbeat=20, max_msg_size=0, compress=0,
                                                     timeout=aiohttp.ClientWSTimeout(ws_close=5))
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.err = f"attach failed: {type(e).__name__} {getattr(e, 'status', '')}"
                await asyncio.sleep(1.0 + attempt)
        if not self.ws:
            self.log(f"sideband {self.mode}: {self.err}")
            return
        self.attached = True
        try:
            async for msg in self.ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                        break
                    continue
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    continue
                self._event(m)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.err = f"sideband read failed: {type(e).__name__}"
        finally:
            self.attached = False
            if not self.closing:
                self.on_closed(self.err or "sideband closed")

    def _event(self, m: dict) -> None:
        t = str(m.get("type", ""))
        self.events[t] = self.events.get(t, 0) + 1
        if self.mode == "realtime":
            if t == "response.done":
                u = (m.get("response") or {}).get("usage") or {}
                usd = realtime_usd(u)
                self.tokens["in"] += int(u.get("input_tokens") or 0)
                self.tokens["out"] += int(u.get("output_tokens") or 0)
                self.tokens["responses"] += 1
                self._usd(usd)
            elif t == "input_audio_buffer.speech_started":
                self.on_activity("speech")
            elif t == "conversation.item.input_audio_transcription.completed":
                tx = str(m.get("transcript") or "")
                self.on_transcript("you", tx)
                self._usd(len(tx) / 15 / 60 * ASR_USD_PER_MIN * 4)   # rough: ~4 s per 60 chars of speech
            elif t == "response.output_audio_transcript.done":
                self.on_transcript("avatar", str(m.get("transcript") or ""))
            elif t == "error":
                e = m.get("error") or {}
                self.log(f"sideband error: {e.get('code', '')} {str(e.get('message', ''))[:160]}")
        else:
            if t == "session.usage.updated":
                s = float(((m.get("usage") or {}).get("seconds")) or 0)
                if s > self.live_seconds:
                    self._usd((s - self.live_seconds) / 60 * LIVE_USD_PER_MIN)
                    self.live_seconds = s
            elif t == "session.input_transcript.delta":
                self.on_activity("speech")
                self.on_transcript("you", str(m.get("delta") or ""))
            elif t == "session.output_transcript.delta":
                self.on_transcript("avatar", str(m.get("delta") or ""))
            elif t == "response.event":
                e = m.get("event") or {}
                if e.get("type") in ("response.completed", "response.done", "response.incomplete", "response.failed"):
                    u = (e.get("response") or {}).get("usage") or {}
                    ti, to = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
                    self.tokens["in"] += ti
                    self.tokens["out"] += to
                    self.tokens["responses"] += 1
                    self._usd(live_backend_usd(ti, to))
            elif t == "session.closed":
                s = float(((m.get("usage") or {}).get("seconds")) or 0)
                if s > self.live_seconds:
                    self._usd((s - self.live_seconds) / 60 * LIVE_USD_PER_MIN)
                    self.live_seconds = s

    def _usd(self, usd: float) -> None:
        if usd > 0:
            self.usd += usd
            self.on_usd(usd)

    def unbilled(self) -> float:
        """Live bills wall clock: the part not yet covered by a usage snapshot (err high)."""
        if self.mode != "live":
            return 0.0
        return max(0.0, (time.monotonic() - self.opened_at) - self.live_seconds) / 60 * LIVE_USD_PER_MIN

    async def hangup(self) -> None:
        self.closing = True
        key = os.environ.get("OPENAI_API_KEY") or ""
        try:
            if self.mode == "realtime" and self.id:
                async with self.http.post(f"{API}/realtime/calls/{self.id}/hangup",
                                          headers={"Authorization": "Bearer " + key},
                                          timeout=aiohttp.ClientTimeout(total=6)) as r:
                    if r.status >= 300:
                        self.log(f"realtime hangup HTTP {r.status}")
            elif self.mode == "live" and self.ws and not self.ws.closed:
                await self.ws.send_str('{"type":"session.close"}')
                await asyncio.sleep(1.5)            # let session.closed (final usage) arrive
        except Exception as e:  # noqa: BLE001
            self.log(f"hangup failed: {type(e).__name__}")
        await self.close()

    async def close(self) -> None:
        self.closing = True
        if self.mode == "live":
            wall = time.monotonic() - self.opened_at
            if wall > self.live_seconds + 1:          # no final snapshot: bill the wall clock
                self._usd((wall - self.live_seconds) / 60 * LIVE_USD_PER_MIN)
                self.live_seconds = wall
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except BaseException:  # noqa: BLE001
                pass
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict:
        return {"mode": self.mode, "id": self.id[:24], "attached": self.attached, "usd": round(self.usd, 5),
                "live_seconds": round(self.live_seconds, 1), "tokens": self.tokens, "err": self.err,
                "events": dict(sorted(self.events.items(), key=lambda kv: -kv[1])[:30])}
