"""Gemini 3.8 Live as a voice back-end for the page's direct mode.

The server keeps the key: it mints a ONE-USE ephemeral token whose BidiGenerateContent setup is LOCKED here (model, the same
persona / instructions and body tools rtc.py gives gpt-realtime, voice, transcription, turn detection, context window
compression), and the page opens the constrained WebSocket with that token and an empty setup.
Measured with the same latency harness as gpt-realtime-2.1: gemini-3.8-live first audio
1,110 / 1,328 ms (median / p90) vs realtime 1,029 / 1,116; spoken command -> tool call 571 vs 515 ms, 6/6 right tools.
gemini-3.8-live-extended-thinking: first audio 1,199-1,342 ms and it makes NO function calls (0/6, even a one-tool
schema) — so its "thinking" mode cannot drive the body; motion then comes only from the server's fast matcher.
"""
import datetime, json, os
import aiohttp

import rtc

MINT_URL = "https://generativelanguage.googleapis.com/v1alpha/auth_tokens"
WS_URL = ("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha."
          "GenerativeService.BidiGenerateContentConstrained")
MODES = {"gemini": ("gemini-3.8-live", None), "gemini-think": ("gemini-3.8-live-extended-thinking", "low")}
VOICES = {"shimmer": "Leda", "coral": "Aoede", "marin": "Kore", "ballad": "Leda", "verse": "Puck", "ash": "Charon",
          "echo": "Fenrir", "sage": "Orus", "alloy": "Zephyr", "cedar": "Puck"}
BODY_NOTE = (" You really have this body on the user's screen: whole-body moves (spins, flips, jumps, dancing, sitting, walking) and the "
             "things around you are yours to use through the tools. When the user asks for one, call the tool and say you are doing it — "
             "never say you can't do something your tools offer, and if a tool result says it is already happening, say so.")
# planner mode: every commanded body action is perform_action (rtc.py); its result is 'preparing' / 'starting'
BODY_NOTE_PLANNER = (" You really have this body on the user's screen: anything the user asks you to do with it is yours to do through "
                     f"{rtc.ACTION_TOOL}. Never say you can't do something with your body; pass their request on and let it try.")
VISION_NOTE = (" You also see the user's camera as a low-rate stream of small frames (about two a second): react naturally to "
               "what they do or hold up when it matters, but never narrate the frames or say you are watching a video.")


def setup_for(mode: str, name: str, what: str, voice: str, character: str, history: list | None = None,
              planner: bool | None = None) -> dict:
    model, level = MODES[mode]
    # the thinking model makes no function calls (measured 0/6): it keeps the planner-off session (matchers drive the body)
    planner = rtc.planner_on(planner) and not level
    s = rtc.realtime_session(name, what, voice, "server", character=character, planner=planner)
    fns = [{"name": t["name"], "description": str(t.get("description", ""))[:1000],
            "parameters": t.get("parameters") or {"type": "object", "properties": {}}} for t in s.get("tools", [])]
    instr = str(s.get("instructions", "")) + ((BODY_NOTE_PLANNER if planner else BODY_NOTE) if not level else "") + VISION_NOTE
    if history:
        lines = "\n".join(("Him: " if h.get("role") == "user" else "You: ") + str(h.get("text", ""))[:300] for h in history[-20:])
        instr += "\n\nThe conversation so far (the call reconnected; carry on naturally, do not repeat or mention it):\n" + lines
    gc = {"responseModalities": ["AUDIO"],
          "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES.get(voice, "Leda")}}},
          # design note: low media resolution (~64 tokens a frame), camera at 2 frames a second
          "mediaResolution": "MEDIA_RESOLUTION_LOW"}
    if level:
        gc["thinkingConfig"] = {"thinkingLevel": level}
    setup = {"model": "models/" + model, "generationConfig": gc, "systemInstruction": {"parts": [{"text": instr}]},
             "inputAudioTranscription": {}, "outputAudioTranscription": {},
             # the user's turn ends after 300 ms of quiet (gpt-realtime's server_vad setting); measured -9 ms first audio, -177 ms tool
             "realtimeInputConfig": {"automaticActivityDetection": {"endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                                                                     "silenceDurationMs": 300, "prefixPaddingMs": 100}},
             # audio + video sessions are capped at ~2 min without it
             "contextWindowCompression": {"slidingWindow": {}}}
    if not level:
        setup["tools"] = [{"functionDeclarations": fns}]   # the thinking model ignores tools (measured 0/6) — leave them off
    return {"setup": setup, "model": model, "planner": planner}


async def mint(http: aiohttp.ClientSession, mode: str, name: str, what: str, voice: str, character: str,
               history: list | None = None, planner: bool | None = None) -> dict:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise rtc.MissingKey("GEMINI_API_KEY", "the Gemini Live voice", "https://aistudio.google.com/apikey")
    sf = setup_for(mode, name, what, voice, character, history, planner)
    now = datetime.datetime.now(datetime.timezone.utc)
    iso = lambda d: (now + d).isoformat().replace("+00:00", "Z")
    body = {"uses": 1, "expireTime": iso(datetime.timedelta(minutes=30)), "newSessionExpireTime": iso(datetime.timedelta(minutes=1)),
            "bidiGenerateContentSetup": sf["setup"]}
    async with http.post(MINT_URL, headers={"x-goog-api-key": key}, json=body,
                         timeout=aiohttp.ClientTimeout(total=12)) as r:
        txt = await r.text()
        if r.status != 200:
            raise rtc.VendorRefused(f"gemini token HTTP {r.status}")
        tok = json.loads(txt).get("name")
    if not tok:
        raise rtc.VendorRefused("gemini token missing")
    return {"token": tok, "url": WS_URL, "model": sf["model"], "tools": "tools" in sf["setup"], "planner": sf["planner"]}
