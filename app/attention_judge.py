"""ATTENTION — server-side vision JUDGE for the page's attention candidates (app/attention.js).

The page sends a candidate only when its cheap detector fires (frames ONLY on candidates, never a stream):
    {type:'attention_candidate', id, t, kinds:[...], strength, details:{kind: detail}, images:[dataURL (before), dataURL (now)],
     summary:{speaking, avatar_speaking, expression, attending, face, hands, motion, level, recent:[...]}}
and the judge answers
    {type:'attention', id, t, kinds, what (<= 12 words), worth_reply (bool), salience 0..1, reason, ms, model, usd[, skipped]}
`skipped` = 'rate' | 'cap' | 'timeout' | 'error' | 'invalid' | 'off' when no model verdict was used (worth_reply is then false).

Standing instructions ("don't comment on my coffee", "tell me if my cat walks in") are persisted at
AVATAR_HOME/state/attention.json ({rules:[{id, text, t, source:'user'|'feedback', about?, kinds?}], spend:{day, usd}, ...}):
    {type:'attention_rule', op:'add', text}                 -> a user rule
    {type:'attention_rule', op:'feedback', text, id?}       -> a rule tied to the verdict `id` (default: the latest verdict):
                                                               `about` = that verdict's `what`, `kinds` = its kinds
    {type:'attention_rule', op:'remove', id} | {op:'clear'} | {op:'list'}
    every op answers {type:'attention_rules', rules:[...]}.
The prompt carries the standing rules, the candidate's kinds/details/summary and a short memory of the recent verdicts
(so it does not keep remarking on the same thing).

    judge = AttentionJudge(home_dir, log, http_session_factory=lambda: hub.http, on_usd=lambda usd: spend.add('attention', usd),
                           cfg={...})
    reply = await judge.handle(msg)      # 'attention_candidate' -> 'attention' reply (never raises)
    rules = judge.handle_rule(msg)       # 'attention_rule' -> 'attention_rules'
    judge.reset_session()                # a new avatar session: memory + per-session spend reset (rules persist)

Vendor: OpenAI-compatible chat completions (OpenRouter by default; provider 'openai' = api.openai.com directly), JSON
mode, usage.cost from OpenRouter ("usage":{"include":true}) else tokens x PRICE. aiohttp is imported lazily (the module
imports without it; tests inject `vendor=` — an async callable(payload, timeout_s) -> response dict).
Config (cfg): see DEFAULTS. Model choice: tests/attention_judge_bench.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# USD per 1M tokens (input, output) — OpenRouter list prices 2026-11-09 (fallback when the response has no usage.cost)
PRICE = {
    "google/gemini-3.8-flash": (0.825, 4.125), "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "google/gemini-3.1-flash-lite": (0.25, 1.50), "openai/gpt-5.4-mini": (0.75, 4.50), "openai/gpt-5.4-nano": (0.20, 1.25),
    "qwen/qwen3.8-flash": (0.15, 0.47), "anthropic/claude-haiku-5.5": (0.10, 0.50), "openai/gpt-6-luna": (0.10, 0.50),
    "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25), "gpt-6-luna": (0.10, 0.50),
}
DEFAULTS = {
    "enabled": True,
    "provider": "openrouter",            # 'openrouter' | 'openai'
    "model": "google/gemini-3.5-flash-lite",     # best accuracy (0.90, rules 6/6) and latency (p50 1.19 s)
    "reasoning": None,                   # e.g. {"effort": "minimal"} (OpenRouter) / "minimal" (OpenAI reasoning_effort)
    "timeout_s": 6.0,
    "max_tokens": 160,
    "temperature": 0.2,
    "image_detail": "low",               # OpenAI-style image detail hint
    "min_gap_s": 2.5,                    # between vendor calls (the page already limits itself to 1 / 3 s)
    "max_per_min": 12,
    "session_cap_usd": 0.50,
    "day_cap_usd": 2.00,
    "memory": 6,                         # recent verdicts in the prompt
    "memory_s": 180,                     # ... no older than this
    "max_rules": 30,
    "max_images": 2,
    "max_image_chars": 120_000,
}
MAX_WHAT_WORDS = 12

SYSTEM = """You are the eyes of a friendly animated companion on a live video call with the user. A cheap detector on \
the user's device flagged that something may have changed in the camera view and sent one or two small frames: \
"before" (about a second earlier) and "now". Look at the frames and decide.

Answer ONLY with a JSON object: {"what": "...", "worth_reply": true or false, "salience": 0.0-1.0, "reason": "..."}

- what: at most 12 words, concrete and visual: what changed or what is notable now (who appeared, what is being held \
up or shown, what the face or body is doing, what happened in the room). If nothing notable is visible, say briefly \
what you see and give a low salience.
- worth_reply: would a warm, socially normal companion naturally say something out loud RIGHT NOW because of what it \
sees? Usually yes for: someone new appearing, or someone greeting the camera; the user deliberately showing or \
holding something up toward the camera; a clear greeting gesture; a striking moment while the user is quiet (a big \
yawn, tears, a startled face, laughing at something off-screen); the user coming back to the camera after being \
away; something surprising, funny or unusual happening in view. Usually no for: ordinary movement while talking or \
listening; small changes in posture, lighting or camera angle; the user simply sitting, working, eating or drinking \
as usual; something you already remarked on recently (see your recent notes); and, while the user is in the middle \
of speaking, anything that is not urgent.
- The user's standing instructions always win. If one says not to comment on something and that is what is notable \
now, worth_reply is false. If one asks to be told about something, worth_reply is true only when that thing is \
clearly visible now.
- salience: how notable the change itself is (0 = nothing, 1 = very striking), regardless of the instructions.
- reason: one short sentence."""


def _now_day() -> str:
    return time.strftime("%Y-%m-%d")


def _short(s, n: int) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def _words(s: str, n: int) -> str:
    w = _short(s, 400).split(" ")
    return " ".join(w[:n]).strip(" ,;")


def parse_verdict(text: str) -> dict | None:
    """The model's JSON verdict (tolerates code fences / prose around the object). None if unusable."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        try:
            d = json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except (ValueError, TypeError):
            return None
    if not isinstance(d, dict):
        return None
    wr = d.get("worth_reply")
    if isinstance(wr, str):
        wr = wr.strip().lower() in ("true", "yes", "1")
    try:
        sal = float(d.get("salience", 0) or 0)
    except (TypeError, ValueError):
        sal = 0.0
    return {"what": _words(d.get("what", ""), MAX_WHAT_WORDS), "worth_reply": bool(wr),
            "salience": round(min(1.0, max(0.0, sal)), 3), "reason": _short(d.get("reason", ""), 200)}


def _fmt_detail(kind: str, det) -> str:
    if not isinstance(det, dict):
        return kind
    keep = {k: v for k, v in det.items() if k in ("what", "from", "to", "away_s", "ratio", "frac", "side", "source", "after_s")
            and v is not None}
    if not keep:
        return kind
    return kind + " (" + ", ".join(f"{k}={v}" for k, v in keep.items()) + ")"


class AttentionJudge:
    def __init__(self, home_dir: str | None, log=print, http_session_factory=None, on_usd=None, cfg: dict | None = None,
                 vendor=None, clock=time.time):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        # the default provider is OpenRouter; with only an OpenAI key (and no provider configured), use OpenAI directly
        if (vendor is None and "provider" not in (cfg or {}) and not os.environ.get("OPENROUTER_API_KEY")
                and os.environ.get("OPENAI_API_KEY")):
            self.cfg.update(provider="openai", model="gpt-5.4-mini", reasoning="none")
        self.log = log or (lambda *_: None)
        self.http_factory = http_session_factory
        self.on_usd = on_usd
        self.vendor = vendor
        self.clock = clock
        self.path = os.path.join(home_dir, "state", "attention.json") if home_dir else None
        self.state = {"rules": [], "spend": {"day": _now_day(), "usd": 0.0}, "seq": 0}
        self._load()
        self.memory: deque = deque(maxlen=50)          # recent verdicts {id, at, what, worth_reply, kinds}
        self.calls: deque = deque(maxlen=200)          # wall times of vendor calls
        self.session_usd = 0.0
        self.n = {"candidates": 0, "calls": 0, "worth": 0, "skipped": {}, "errors": 0}
        self.ms: deque = deque(maxlen=200)
        self._own_http = None

    # ------------------------------------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                if isinstance(d.get("rules"), list):
                    self.state["rules"] = [r for r in d["rules"] if isinstance(r, dict) and isinstance(r.get("text"), str)]
                if isinstance(d.get("spend"), dict):
                    self.state["spend"] = {"day": str(d["spend"].get("day") or _now_day()),
                                           "usd": float(d["spend"].get("usd") or 0.0)}
                self.state["seq"] = int(d.get("seq") or len(self.state["rules"]))
        except (OSError, ValueError, TypeError) as e:
            self.log(f"attention: state unreadable ({type(e).__name__}); starting empty")

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            d = dict(self.state)
            d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            self.log(f"attention: state not saved ({type(e).__name__})")

    # ------------------------------------------------------------------------------------------------ rules
    def rules(self) -> list:
        return [dict(r) for r in self.state["rules"]]

    def handle_rule(self, msg: dict) -> dict:
        op = str((msg or {}).get("op") or "add")
        text = _short((msg or {}).get("text"), 300)
        rules = self.state["rules"]
        changed = False
        if op in ("add", "feedback") and text:
            if not any(r["text"].lower() == text.lower() for r in rules):
                self.state["seq"] = int(self.state.get("seq") or 0) + 1
                r = {"id": f"r{self.state['seq']}", "text": text, "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "source": "user" if op == "add" else "feedback"}
                if op == "feedback":
                    vid = (msg or {}).get("id")
                    v = next((x for x in reversed(self.memory) if vid and x["id"] == vid), None)
                    if v is None and not vid and self.memory:
                        v = self.memory[-1]
                    if v is not None:
                        r["about"] = v["what"]
                        r["kinds"] = list(v.get("kinds") or [])
                        r["verdict"] = v["id"]
                    elif vid:
                        r["verdict"] = str(vid)[:40]
                rules.append(r)
                while len(rules) > int(self.cfg["max_rules"]):
                    i = next((k for k, x in enumerate(rules) if x.get("source") == "feedback"), 0)
                    rules.pop(i)
                changed = True
        elif op == "remove":
            rid = (msg or {}).get("id")
            n0 = len(rules)
            self.state["rules"] = [r for r in rules if r["id"] != rid and (not text or r["text"].lower() != text.lower())]
            changed = len(self.state["rules"]) != n0
        elif op == "clear":
            changed = bool(rules)
            self.state["rules"] = []
        if changed:
            self._save()
            self.log(f"attention: rules {op} -> {len(self.state['rules'])}")
        return {"type": "attention_rules", "rules": self.rules()}

    # ------------------------------------------------------------------------------------------------ spend
    def _day_usd(self) -> float:
        sp = self.state["spend"]
        if sp.get("day") != _now_day():
            sp["day"], sp["usd"] = _now_day(), 0.0
        return float(sp.get("usd") or 0.0)

    def _add_usd(self, usd: float) -> None:
        if usd <= 0:
            return
        self._day_usd()
        self.state["spend"]["usd"] = round(self.state["spend"]["usd"] + usd, 6)
        self.session_usd += usd
        self._save()
        if self.on_usd:
            try:
                self.on_usd(usd)
            except Exception as e:                        # noqa: BLE001 - accounting must never break the reply
                self.log(f"attention: on_usd failed ({type(e).__name__})")

    def usd_of(self, model: str, usage: dict | None, est_in: int = 0, est_out: int = 0) -> tuple[float, bool]:
        """(usd, estimated). OpenRouter's usage.cost when present, else tokens x PRICE (estimates when no usage)."""
        u = usage or {}
        if isinstance(u.get("cost"), (int, float)):
            return float(u["cost"]), False
        pin, pout = PRICE.get(model, (1.0, 5.0))
        if u.get("prompt_tokens") is not None:
            return (float(u.get("prompt_tokens") or 0) * pin + float(u.get("completion_tokens") or 0) * pout) / 1e6, False
        return (est_in * pin + est_out * pout) / 1e6, True

    # ------------------------------------------------------------------------------------------------ prompt
    def build_messages(self, msg: dict, images: list) -> list:
        kinds = [str(k)[:30] for k in (msg.get("kinds") or [])][:8]
        det = msg.get("details") if isinstance(msg.get("details"), dict) else {}
        summ = msg.get("summary") if isinstance(msg.get("summary"), dict) else {}
        yn = lambda v: "yes" if v else "no"                                   # noqa: E731
        lines = ["Detector cues: " + ("; ".join(_fmt_detail(k, det.get(k)) for k in kinds) or "(none)") + "."]
        st = [f"user speaking: {yn(summ.get('speaking'))}", f"companion speaking: {yn(summ.get('avatar_speaking'))}"]
        if "face" in summ:
            st.append(f"user's face visible: {yn(summ.get('face'))}")
        if summ.get("face"):
            st.append(f"looking at the camera: {yn(summ.get('attending'))}")
            if summ.get("expression") and summ.get("expression") != "neutral":
                st.append(f"expression reading: {_short(summ.get('expression'), 20)}")
        lines.append("Status: " + "; ".join(st) + ".")
        rules = self.state["rules"]
        if rules:
            lines.append("The user's standing instructions:")
            for i, r in enumerate(rules[-int(self.cfg["max_rules"]):], 1):
                about = f' (said after you noted "{r["about"]}")' if r.get("about") else ""
                lines.append(f"{i}. {r['text']}{about}")
        else:
            lines.append("The user's standing instructions: (none)")
        now = self.clock()
        mem = [m for m in self.memory if now - m["at"] <= float(self.cfg["memory_s"])][-int(self.cfg["memory"]):]
        if mem:
            lines.append("Your recent notes (newest last):")
            for m in mem:
                lines.append(f'- {int(now - m["at"])} s ago: "{m["what"]}" (you {"spoke up" if m["worth_reply"] else "stayed quiet"})')
        lines.append("Frames: " + ("before (about 1 s earlier), then now." if len(images) > 1 else "now."))
        content = [{"type": "text", "text": "\n".join(lines)}]
        for u in images:
            iu = {"url": u}
            if self.cfg.get("image_detail"):
                iu["detail"] = self.cfg["image_detail"]
            content.append({"type": "image_url", "image_url": iu})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def payload(self, messages: list) -> dict:
        c = self.cfg
        p = {"model": c["model"], "messages": messages, "max_tokens": int(c["max_tokens"]),
             "temperature": float(c["temperature"]), "response_format": {"type": "json_object"}}
        if c["provider"] == "openrouter":
            p["usage"] = {"include": True}
            if c.get("reasoning"):
                p["reasoning"] = c["reasoning"]
        else:
            if c.get("reasoning"):
                p["reasoning_effort"] = c["reasoning"] if isinstance(c["reasoning"], str) else c["reasoning"].get("effort", "minimal")
                p.pop("temperature", None)
                p["max_completion_tokens"] = p.pop("max_tokens")
        return p

    # ------------------------------------------------------------------------------------------------ vendor
    async def _http(self):
        if self.http_factory is not None:
            s = self.http_factory()
            if s is not None:
                return s
        if self._own_http is None:
            import aiohttp                                  # lazy: the module imports without aiohttp
            self._own_http = aiohttp.ClientSession()
        return self._own_http

    async def _call(self, payload: dict, timeout_s: float) -> dict:
        if self.vendor is not None:
            return await self.vendor(payload, timeout_s)
        import aiohttp
        if self.cfg["provider"] == "openai":
            url, key = OPENAI_URL, os.environ.get("OPENAI_API_KEY", "")
        else:
            url, key = OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("no API key for " + self.cfg["provider"])
        http = await self._http()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.cfg["provider"] == "openrouter":
            headers["X-Title"] = "avatar attention judge"
        async with http.post(url, json=payload, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            body = await r.text()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {_short(body, 160)}")
            return json.loads(body)

    async def close(self) -> None:
        if self._own_http is not None:
            await self._own_http.close()
            self._own_http = None

    # ------------------------------------------------------------------------------------------------ main
    def _reply(self, msg: dict, **kw) -> dict:
        r = {"type": "attention", "id": str(msg.get("id") or "")[:40], "t": msg.get("t"),
             "kinds": [str(k)[:30] for k in (msg.get("kinds") or [])][:8],
             "what": "", "worth_reply": False, "salience": 0.0, "reason": "", "ms": 0,
             "model": self.cfg["model"], "usd": 0.0}
        r.update(kw)
        if r.get("skipped"):
            self.n["skipped"][r["skipped"]] = self.n["skipped"].get(r["skipped"], 0) + 1
        return r

    async def handle(self, msg: dict) -> dict | None:
        if not isinstance(msg, dict) or msg.get("type") != "attention_candidate":
            return None
        self.n["candidates"] += 1
        c = self.cfg
        if not c.get("enabled", True):
            return self._reply(msg, skipped="off", reason="judge off")
        imgs = [u for u in (msg.get("images") or []) if isinstance(u, str)
                and re.match(r"^data:image/(jpeg|png|webp);base64,", u) and len(u) <= int(c["max_image_chars"])]
        imgs = imgs[-int(c["max_images"]):]
        if not imgs:
            return self._reply(msg, skipped="invalid", reason="no usable image")
        now = self.clock()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if (self.calls and now - self.calls[-1] < float(c["min_gap_s"])) or len(self.calls) >= int(c["max_per_min"]):
            return self._reply(msg, skipped="rate", reason="rate limited")
        if self.session_usd >= float(c["session_cap_usd"]) or self._day_usd() >= float(c["day_cap_usd"]):
            return self._reply(msg, skipped="cap", reason="attention spend cap reached")
        self.calls.append(now)
        self.n["calls"] += 1
        messages = self.build_messages(msg, imgs)
        payload = self.payload(messages)
        est_in = len(json.dumps(messages[0])) // 4 + len(messages[1]["content"][0]["text"]) // 4 + 600 * len(imgs)
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(self._call(payload, float(c["timeout_s"])), timeout=float(c["timeout_s"]) + 0.5)
        except asyncio.TimeoutError:
            ms = int((time.monotonic() - t0) * 1000)
            usd, _ = self.usd_of(c["model"], None, est_in, int(c["max_tokens"]) // 2)   # likely billed anyway: estimate
            self._add_usd(usd)
            self.log(f"attention {msg.get('id')} timeout after {ms} ms")
            return self._reply(msg, skipped="timeout", reason="judge timeout", ms=ms, usd=round(usd, 6))
        except Exception as e:                                                      # noqa: BLE001
            ms = int((time.monotonic() - t0) * 1000)
            self.n["errors"] += 1
            self.log(f"attention {msg.get('id')} error {type(e).__name__}: {_short(e, 160)}")
            return self._reply(msg, skipped="error", reason=_short(f"{type(e).__name__}: {e}", 160), ms=ms)
        ms = int((time.monotonic() - t0) * 1000)
        self.ms.append(ms)
        choice = ((resp or {}).get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        usd, est = self.usd_of(c["model"], (resp or {}).get("usage"), est_in, len(text) // 4 + 1)
        self._add_usd(usd)
        v = parse_verdict(text)
        if v is None:
            self.n["errors"] += 1
            self.log(f"attention {msg.get('id')} unparsable verdict ({len(text)} chars)")
            return self._reply(msg, skipped="error", reason="unparsable verdict", ms=ms, usd=round(usd, 6))
        if v["worth_reply"]:
            self.n["worth"] += 1
        self.memory.append({"id": str(msg.get("id") or "")[:40], "at": self.clock(), "what": v["what"],
                            "worth_reply": v["worth_reply"], "kinds": list(msg.get("kinds") or [])[:8]})
        self.log(f"attention {msg.get('id')} {'+'.join(msg.get('kinds') or [])} -> {v['what']!r} "
                 f"worth={v['worth_reply']} sal={v['salience']} {ms} ms ${usd:.5f}{' est' if est else ''}")
        fam = " + ".join(dict.fromkeys(str(k).replace("_", " ") for k in (msg.get("kinds") or [])[:8]))
        if fam and not v["reason"].lower().startswith(fam.split(" + ")[0]):
            v["reason"] = _short(f"{fam}: {v['reason']}" if v["reason"] else fam, 200)   # the event family, in words, first
        return self._reply(msg, ms=ms, usd=round(usd, 6), model=(resp or {}).get("model") or c["model"], **v)

    def reset_session(self) -> None:
        self.memory.clear()
        self.calls.clear()
        self.session_usd = 0.0

    def stats(self) -> dict:
        ms = sorted(self.ms)
        pct = (lambda p: ms[min(len(ms) - 1, int(p * len(ms)))] if ms else None)  # noqa: E731
        return {"candidates": self.n["candidates"], "calls": self.n["calls"], "worth_reply": self.n["worth"],
                "skipped": dict(self.n["skipped"]), "errors": self.n["errors"], "ms_p50": pct(0.5), "ms_p90": pct(0.9),
                "session_usd": round(self.session_usd, 5), "day_usd": round(self._day_usd(), 5), "rules": len(self.state["rules"]),
                "model": self.cfg["model"]}
