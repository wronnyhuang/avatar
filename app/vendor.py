"""gpt-live-1 client (the vendor half of the server): session.start / input_audio.append / output_audio.delta /
transcripts / usage.

The API key is read from the environment by reference and only ever placed in the Authorization header; it is never
logged (log lines go through scrub()).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import aiohttp

LIVE_URL = os.environ.get("AVATAR_LIVE_URL", "wss://api.openai.com/v1/live/sessions")
MODEL = "gpt-live-1"
BACKEND = os.environ.get("AVATAR_LIVE_BACKEND", "gpt-6-sol")
USD_PER_MIN = 0.05
BACKEND_PRICES = {"gpt-6-sol": (2.0, 10.0), "gpt-6.1-sol": (2.0, 10.0), "gpt-5.6-sol": (4.0, 20.0)}  # $/1M in, out
ANIMATIONS = ("look_up", "look_down", "look_left", "look_right", "turn_head", "raise_hand", "wave",
              "point", "open_mouth", "smile", "frown", "surprised", "nod", "shake_head", "thumbs_up")
RIG_PARTS = ("head_yaw", "head_pitch", "head_tilt", "head_nod", "head_shake",
             "left_ear_up", "right_ear_up", "left_ear_down", "right_ear_down",
             "left_ear_flop", "right_ear_flop", "left_ear_wiggle", "right_ear_wiggle", "ears_wiggle",
             "nose_wiggle", "nose_scrunch", "nose_sniff", "eyes_x", "eyes_y", "eyes_blink",
             "left_eye_wink", "right_eye_wink", "eyes_widen", "eyes_squint",
             "left_brow_raise", "right_brow_raise", "left_brow_furrow", "right_brow_furrow",
             "mouth_open", "mouth_smile", "mouth_frown", "mouth_pout", "tongue_out",
             "cheeks_puff", "cheeks_blush", "left_arm_raise", "right_arm_raise",
             "left_arm_wave", "right_arm_wave", "left_paw_point", "right_paw_point",
             "left_paw_open", "right_paw_open", "clap", "thumbs_up",
             "tail_wag", "tail_raise", "body_lean", "body_bounce")
RIG_TOOL = {"type": "function", "name": "animate", "description":
    "Immediately animate one or several physical parts of the character. Call this for every requested body/face motion, "
    "including ears, nose, eyes, brows, mouth, arms, paws, tail, and body. amount is signed -1 to 1; "
    "duration_ms is 100 to 30000; repeat=true keeps moving until another command replaces it. "
    "Use several motions together for combinations. Ears/tails on geometric characters are not anatomical; "
    "tell the person playfully, then use an equivalent visible wiggle or bounce.",
    "parameters": {"type": "object", "properties": {"motions": {"type": "array", "items": {"type": "object",
        "properties": {"part": {"type": "string", "enum": list(RIG_PARTS)}, "amount": {"type": "number"},
                       "duration_ms": {"type": "integer"}, "repeat": {"type": "boolean"}},
        "required": ["part", "amount", "duration_ms", "repeat"], "additionalProperties": False}}},
        "required": ["motions"], "additionalProperties": False}, "strict": True}
ANIMATION_TOOLS = [
    {"type": "function", "name": name,
     "description": f"Make the animated character {name.replace('_', ' ')} immediately. Use when the person asks or it fits your reply.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": True}
    for name in ANIMATIONS
] + [RIG_TOOL, {"type": "function", "name": "set_emotion", "description": "Set the character's visible emotion for your next spoken reply.",
      "parameters": {"type": "object", "properties": {"emotion": {"type": "string", "enum": ["happy", "concerned", "curious", "surprised", "calm"]}},
                     "required": ["emotion"], "additionalProperties": False}, "strict": True}]


# planner mode: the animation tools are the character's own expressiveness; commanded actions are perform_action
ACTION_TOOL = os.environ.get("AVATAR_ACTION_TOOL") if os.environ.get("AVATAR_ACTION_TOOL") in ("do_action", "perform_action") \
    else "do_action"                                                    # = rtc.ACTION_TOOL (perform_action = alias)
PLANNER_BODY = (f"\nAnything the user asks the body to do goes to {ACTION_TOOL}, once, with the request restated completely; "
                "use animate only for small motions of your own that fit what you say, never for the user's requests. ")


def backend_usd(tin: int, tout: int, model: str = BACKEND, fast: bool = False) -> float:
    pin, pout = BACKEND_PRICES.get(model, max(BACKEND_PRICES.values(), key=lambda r: r[1]))
    return (max(0, tin) * pin + max(0, tout) * pout) * (2 if fast else 1) / 1e6


def scrub(s: str) -> str:
    k = os.environ.get("OPENAI_API_KEY") or ""
    s = str(s)
    if k and k in s:
        s = s.replace(k, "[key]")
    return s


class LiveVendor:
    """One gpt-live-1 session over the primary WebSocket. Callbacks run on the event loop:
    on_audio(pcm_bytes), on_transcript(who, delta), on_voice_usd(usd_delta), on_backend_usd(usd, tin, tout),
    on_end(reason)."""

    def __init__(self, http: aiohttp.ClientSession, *, on_audio, on_transcript, on_voice_usd, on_backend_usd, on_end,
                 on_action=None, log=print, act_tools=None, act_note="", on_act=None, planner=False):
        self.http = http
        # planner mode: perform_action (in act_tools, answered by the page) takes every commanded body action;
        # animate / set_emotion stay for expressiveness only. False = the plain session exactly.
        self.planner = bool(planner)
        # the muse's move / object tools (app.py passes rtc.act_tools() for an act character). Their result is
        # the page's real motion.command() answer ({ok, did|reason}), fetched through on_act (hub → page → hub), never
        # a blanket ok — a call the page doesn't answer in time says so.
        self.act_tools = list(act_tools or [])
        self.act_names = {t.get("name") for t in self.act_tools}
        self.act_note = act_note or ""
        self.on_act = on_act
        self.on_audio, self.on_transcript = on_audio, on_transcript
        self.on_voice_usd, self.on_backend_usd, self.on_end = on_voice_usd, on_backend_usd, on_end
        self.log = log
        self.on_action = on_action or (lambda _name, _args: None)
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.reader: asyncio.Task | None = None
        self.session_id = ""
        self.opened_at = 0.0
        self.closing = False
        self.ended = False
        self.billed_seconds = 0.0
        self.usage_seen = False
        self.tin = self.tout = 0
        self.voice = ""
        self.seen: set[str] = set()
        self.deltas = 0
        self.delta_gaps: list[float] = []
        self.last_delta = 0.0
        self.sent_samples = 0
        self.send_lock = asyncio.Lock()
        self.response_busy = False
        self.pending_calls: dict[str, list[dict]] = {}
        self.pending_acts: dict[str, list] = {}
        self.queued_image: tuple[str, str] | None = None
        self.image_count = 0
        self.last_image_at = 0.0
        self.tool_count = 0

    async def open(self, instructions: str, voice: str, timeout: float = 12.0) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the service environment")
        t0 = time.monotonic()
        try:
            self.ws = await asyncio.wait_for(self.http.ws_connect(
                LIVE_URL, headers={"Authorization": "Bearer " + key}, heartbeat=None, max_msg_size=0,
                autoping=True, compress=0), timeout)
        except aiohttp.WSServerHandshakeError as e:
            raise RuntimeError(f"gpt-live-1 handshake refused: HTTP {e.status} {scrub(e.message)[:160]}") from None
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"gpt-live-1 connect failed: {type(e).__name__} {scrub(e)[:160]}") from None
        connect_ms = (time.monotonic() - t0) * 1000
        self.voice = voice
        await self.ws.send_str(json.dumps({
            "type": "session.start", "event_id": "avatar_start",
            "session": {
                "model": MODEL,
                "instructions": instructions,
                "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": voice}},
                "delegation": {"type": "responses", "responses": {
                    "model": BACKEND, "instructions": instructions + (PLANNER_BODY if self.planner else "\nYou have full physical control via animate. For every physical command, call animate before speaking; never say you cannot move a real body part of an animal. "
                    "Use amount and duration as requested; for 'keep doing it' set repeat=true. Combine motions in one call. ") +
                    "Use set_emotion for a visible mood that fits what you say. Treat camera stills as occasional snapshots; "
                    "name an object shown to you when you recognize it, and never claim you see continuous video." + (" " + self.act_note if self.act_note else ""),
                    "tools": ANIMATION_TOOLS + self.act_tools, "tool_choice": "auto",
                    "reasoning": {"effort": "low"}}},
            }}))
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                await self._abort()
                raise RuntimeError("gpt-live-1: session.started timeout")
            msg = await self.ws.receive(timeout=left)
            if msg.type != aiohttp.WSMsgType.TEXT:
                await self._abort()
                raise RuntimeError(f"gpt-live-1: socket {msg.type.name} before session.started")
            m = json.loads(msg.data)
            t = m.get("type", "")
            self.seen.add(t)
            if t == "session.started":
                self.session_id = str((m.get("session") or {}).get("id") or "")
                self.opened_at = time.monotonic()
                got_voice = ((m.get("session") or {}).get("audio") or {}).get("output", {}).get("voice")
                self.log(f"vendor session {self.session_id[:18]} started, voice {got_voice}, "
                         f"connect {connect_ms:.0f} ms, start {(self.opened_at - t0) * 1000:.0f} ms")
                break
            if t == "error":
                e = m.get("error") or {}
                await self._abort()
                raise RuntimeError(f"gpt-live-1 error: {e.get('code', '')} {scrub(e.get('message', ''))[:200]}")
        self.reader = asyncio.create_task(self._read())
        return {"session_id": self.session_id, "connect_ms": round(connect_ms), "start_ms": round((self.opened_at - t0) * 1000)}

    async def send_audio(self, pcm: bytes) -> None:
        if not self.ws or self.ws.closed or self.closing:
            return
        self.sent_samples += len(pcm) // 2
        try:
            async with self.send_lock:
                await self.ws.send_str('{"type":"session.input_audio.append","audio":"%s"}' % base64.b64encode(pcm).decode())
        except Exception as e:  # noqa: BLE001
            self.log(f"vendor send failed: {type(e).__name__}")

    async def prompt(self, text: str, respond: bool = False) -> bool:
        """a silent note from the page (an action is about to start / started / finished / failed); respond = the
        voice model speaks to it (instructions.append), else context only (thinking.append) — as voice.js does for
        GPT Live on the direct path."""
        if not self.ws or self.ws.closed or self.closing or not text:
            return False
        ev = {"type": "session.instructions.append" if respond else "session.thinking.append", "delegation_id": None,
              "content": str(text)[:800]}
        try:
            async with self.send_lock:
                await self.ws.send_str(json.dumps(ev, separators=(",", ":")))
            return True
        except Exception as e:  # noqa: BLE001
            self.log(f"vendor prompt failed: {type(e).__name__}")
            return False

    async def send_image(self, jpeg_data_url: str, observation: str = "") -> bool:
        """One bounded camera still to the vision-capable Responses backend; never send images to the Live audio frontend."""
        if not self.ws or self.ws.closed or self.closing or not jpeg_data_url.startswith("data:image/jpeg;base64,"):
            return False
        if len(jpeg_data_url) > 120_000 or time.monotonic() - self.last_image_at < .9:
            return False
        self.last_image_at = time.monotonic()
        self.image_count += 1
        # While a Responses call runs, retain only the newest still.  This bounds both cost and latency.
        if self.response_busy:
            self.queued_image = (jpeg_data_url, observation[:120])
            return True
        await self._send_image(jpeg_data_url, observation[:120])
        return True

    async def _send_image(self, jpeg_data_url: str, observation: str) -> None:
        if not self.ws or self.ws.closed or self.closing:
            return
        self.response_busy = True
        item = {"type": "response.item.create", "item": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Current phone camera snapshot. React if there is a meaningful gesture, expression, or object. " + observation},
            {"type": "input_image", "image_url": jpeg_data_url, "detail": "low"}]}}
        async with self.send_lock:
            await self.ws.send_str(json.dumps(item, separators=(",", ":")))
            await self.ws.send_str('{"type":"response.create"}')

    async def _tool(self, item: dict):
        name, call_id = str(item.get("name") or ""), str(item.get("call_id") or "")
        if not call_id:
            return
        try:
            args = json.loads(item.get("arguments") or "{}")
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if (name in self.act_names or (name in ("do_action", "perform_action")              # either name of the
                                       and {"do_action", "perform_action"} & self.act_names)) and self.on_act is not None:   # action tool
            self.tool_count += 1
            return asyncio.ensure_future(self._act_tool(name, args, call_id))   # the page round trip must not stall the reader
        if name in ANIMATIONS or name == "set_emotion" or name == "animate":
            self.tool_count += 1
            self.on_action(name, args)
            result = {"ok": True, "action": name}
        else:
            result = {"ok": False, "error": "unknown animation"}
        event = {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": call_id,
                                                               "output": json.dumps(result)}}
        async with self.send_lock:
            await self.ws.send_str(json.dumps(event, separators=(",", ":")))

    async def _continue_after(self, tasks: list) -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.ws and not self.ws.closed:
            async with self.send_lock:
                await self.ws.send_str('{"type":"response.create"}')

    async def _act_tool(self, name: str, args: dict, call_id: str) -> None:
        try:
            result = await asyncio.wait_for(self.on_act(name, args), 4.0)
        except asyncio.TimeoutError:
            result = {"ok": False, "reason": "the body did not answer in time — it may not have moved"}
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "reason": f"could not run it ({type(e).__name__})"}
        if not isinstance(result, dict):
            result = {"ok": bool(result)}
        event = {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": call_id,
                                                               "output": json.dumps(result)}}
        if self.ws is None or self.ws.closed:
            return
        async with self.send_lock:
            await self.ws.send_str(json.dumps(event, separators=(",", ":")))

    async def _read(self) -> None:
        reason = "socket closed"
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
                t = m.get("type", "")
                self.seen.add(t)
                if t == "session.output_audio.delta":
                    d = m.get("delta")
                    if isinstance(d, str) and d:
                        now = time.monotonic()
                        if self.last_delta:
                            self.delta_gaps.append(now - self.last_delta)
                            if len(self.delta_gaps) > 4000:
                                del self.delta_gaps[:2000]
                        self.last_delta = now
                        self.deltas += 1
                        self.on_audio(base64.b64decode(d))
                elif t == "session.input_transcript.delta":
                    if m.get("delta"):
                        self.on_transcript("you", str(m["delta"]))
                elif t == "session.output_transcript.delta":
                    if m.get("delta"):
                        self.on_transcript("avatar", str(m["delta"]))
                elif t == "session.usage.updated":
                    self._usage(m.get("usage") or {})
                elif t == "response.event":
                    e = m.get("event") or {}
                    et = str(e.get("type", ""))
                    if et == "response.output_item.done":
                        item = e.get("item") or {}
                        if item.get("type") == "function_call":
                            task = await self._tool(item)
                            rid = str(e.get("response_id") or "")
                            self.pending_calls.setdefault(rid, []).append(item)
                            if task is not None:
                                self.pending_acts.setdefault(rid, []).append(task)
                    if et in ("response.completed", "response.done", "response.incomplete", "response.failed"):
                        u = (e.get("response") or {}).get("usage") or {}
                        if u:
                            ti, to = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
                            self.tin += ti
                            self.tout += to
                            self.on_backend_usd(backend_usd(ti, to), ti, to)
                        rid = str((e.get("response") or {}).get("id") or "")
                        calls = self.pending_calls.pop(rid, [])
                        acts = self.pending_acts.pop(rid, [])
                        if calls and acts:              # continue once the page's answers are in
                            asyncio.ensure_future(self._continue_after(acts))
                        elif calls and self.ws and not self.ws.closed:
                            async with self.send_lock:
                                await self.ws.send_str('{"type":"response.create"}')
                        else:
                            self.response_busy = False
                            if self.queued_image:
                                image, obs = self.queued_image
                                self.queued_image = None
                                await self._send_image(image, obs)
                elif t == "session.closed":
                    self._usage(m.get("usage") or {})
                    reason = f"vendor closed ({m.get('reason', '?')})"
                    break
                elif t == "error":
                    e = m.get("error") or {}
                    self.log(f"vendor error: {e.get('code', '')} {scrub(e.get('message', ''))[:200]}")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            reason = f"vendor read failed: {type(e).__name__}"
        finally:
            self._finish_billing()
        if not self.closing and not self.ended:
            self.ended = True
            self.on_end(reason)

    def _usage(self, u: dict) -> None:
        s = u.get("seconds")
        if s is None:
            return
        s = float(s)
        self.usage_seen = True
        if s > self.billed_seconds:
            self.on_voice_usd((s - self.billed_seconds) / 60 * USD_PER_MIN)
            self.billed_seconds = s

    def _finish_billing(self) -> None:
        """No usage snapshot covering the whole session: bill the wall clock (errs high, never low)."""
        if self.opened_at and not getattr(self, "_billed_final", False):
            self._billed_final = True
            wall = time.monotonic() - self.opened_at
            if wall > self.billed_seconds + 1:
                self.on_voice_usd((wall - self.billed_seconds) / 60 * USD_PER_MIN)
                self.billed_seconds = wall

    def unbilled_usd(self) -> float:
        if not self.opened_at or self.ended:
            return 0.0
        return max(0.0, (time.monotonic() - self.opened_at) - self.billed_seconds) / 60 * USD_PER_MIN

    async def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        ws = self.ws
        if ws and not ws.closed:
            try:
                await ws.send_str('{"type":"session.close"}')
                # wait briefly for session.closed (final usage), the reader handles it
                if self.reader:
                    await asyncio.wait_for(asyncio.shield(self.reader), 3.0)
            except Exception:  # noqa: BLE001
                pass
        await self._abort()
        self._finish_billing()
        self.ended = True

    async def _abort(self) -> None:
        if self.reader and not self.reader.done():
            self.reader.cancel()
            try:
                await self.reader
            except BaseException:  # noqa: BLE001
                pass
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict:
        g = sorted(self.delta_gaps)
        pick = (lambda q: round(g[min(len(g) - 1, int(q * len(g)))] * 1000, 1)) if g else (lambda q: None)
        return {"session": self.session_id[:18], "voice": self.voice, "deltas": self.deltas,
                "delta_gap_ms": {"p50": pick(.5), "p95": pick(.95), "max": pick(1.0)},
                "usage_seconds": round(self.billed_seconds, 1), "usage_seen": self.usage_seen,
                "backend_tokens": [self.tin, self.tout], "camera_stills": self.image_count,
                "animation_tools": self.tool_count, "events": sorted(self.seen)}
