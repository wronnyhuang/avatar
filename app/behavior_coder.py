"""PROGRAM LAYER — server side: a standing instruction -> a checked per-frame program for app/behavior.js.

ws page -> hub  {type:'behavior_code', id, instruction, name, character, prev_code?, prev_instruction?, gestures?}
   hub -> page  {type:'behavior_code', id, ok, name, code | error, ms, model, usd, attempts, notes}

Flow: prompt = the toolkit doc (app/behavior.js TOOLKIT_DOC, static -> provider prompt caching) as the system prompt; the
character section (its parts and exact controls), taught gesture names, the instruction (+ the running program and its
instruction for an edit) as the user turn. NO example instruction->program pairs: the doc teaches the toolkit, not cases.
The answer goes through tools/behavior_check.cjs (one persistent node process): compile (sandbox refusals, guard
instrumentation), literal checks (event types, gesture / face / part / look names, controls, groups) and a dry run on a
synthetic tour of every sense cue + recorded perception clips. Fixable findings -> ONE repair turn with the findings; a
failed / timed-out model call -> the fallback model. Model choice: tools/behavior_bench.py (logs/behavior.md).

Config (state/config.json "behavior"): {"enabled", "model", "fallback", "timeout_s", "repair", "max_tokens"}.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "..", "tools", "behavior_check.cjs")

# name -> (route, model id, extra body, USD per M tokens (input, cached input, output))
MODELS = {
    "gpt-5.4-mini": ("openai", "gpt-5.4-mini", {"reasoning": {"effort": "none"}}, (0.75, 0.075, 4.5)),
    "deepseek-v4.1-flash": ("openrouter", "deepseek/deepseek-v4.1-flash", {"reasoning": {"enabled": False}}, (0.051, 0.01, 0.567)),
    "glm-5.3": ("openrouter", "z-ai/glm-5.3", {"reasoning": {"effort": "low"}}, (0.07, 0.02, 1.4)),
    "gemini-3.5-flash-lite": ("openrouter", "google/gemini-3.5-flash-lite", {"reasoning": {"effort": "minimal"}}, (0.1, 0.025, 0.4)),
}
DEFAULTS = {"enabled": True, "model": "gemini-3.5-flash-lite", "fallback": "deepseek-v4.1-flash", "timeout_s": 20.0, "repair": 1,
            "max_tokens": 3000}


class CoderError(Exception):
    pass


def extract_code(text: str) -> str:
    """The program from a model answer: a fenced block's content if any, else the text from the first '(' / helper."""
    t = (text or "").strip()
    m = re.search(r"```(?:[a-zA-Z]+)?\s*\n(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    elif t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).rstrip("`").strip()
    if not t.startswith("(") and not re.match(r"^(const|let|var|function)\b", t):
        i = t.find("({")
        if i >= 0:
            t = t[i:]
    return t.strip()


class Checker:
    """One persistent `node tools/behavior_check.cjs` (JSON lines); restarted when it dies."""

    def __init__(self, log=print, node: str = "node", path: str = CHECKER):
        self.log, self.node, self.path = log, node, path
        self.proc = None
        self.lock = asyncio.Lock()
        self.n = 0
        self.docs: dict[str, tuple[str, str]] = {}

    async def _ensure(self):
        if self.proc is not None and self.proc.returncode is None:
            return
        self.proc = await asyncio.create_subprocess_exec(self.node, self.path, stdin=asyncio.subprocess.PIPE,
                                                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                                                         limit=8 * 1024 * 1024)
        line = await asyncio.wait_for(self.proc.stdout.readline(), 20)
        if b"ready" not in line:
            raise CoderError("checker did not start")

    async def call(self, req: dict, timeout: float = 10.0) -> dict:
        async with self.lock:
            for attempt in (0, 1):
                try:
                    await self._ensure()
                    self.n += 1
                    req = dict(req, id=self.n)
                    self.proc.stdin.write((json.dumps(req) + "\n").encode())
                    await self.proc.stdin.drain()
                    while True:
                        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
                        if not line:
                            raise CoderError("checker exited")
                        res = json.loads(line)
                        if res.get("id") == self.n:
                            return res
                except (asyncio.TimeoutError, CoderError, OSError, ValueError) as e:
                    self.log(f"behavior: checker {type(e).__name__}: {e}")
                    self.close()
                    if attempt:
                        raise CoderError(f"checker unavailable ({type(e).__name__})") from None
        raise CoderError("checker unavailable")

    async def doc(self, character: str) -> tuple[str, str]:
        if character not in self.docs:
            r = await self.call({"op": "doc", "character": character})
            self.docs[character] = (r["toolkit"], r["character_doc"])
        return self.docs[character]

    async def aclose(self):
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                p.kill()
                await asyncio.wait_for(p.wait(), 5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass

    def close(self):
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
        self.proc = None


# the voice model restates the user's request to them ("whenever you smile, I wave"): say whose part is whose
ROLES = 'In it, the character / its name / I / me / my / we = the CHARACTER (the body your program moves) and he / him / his / she / her / they / them / their / you / your = THE USER (the person on camera, whom your program only senses through api.user / api.events / api.heard; api.self is the character itself), however it is worded. Your program makes the CHARACTER do what the request says the character does, when what it says about THE USER happens; it never performs the part of the user.'


def user_turn(character_doc: str, instruction: str, prev_code: str | None, prev_instruction: str | None,
              gestures: list | None, his_words: str | None = None) -> str:
    parts = [character_doc]
    g = [str(x)[:40] for x in (gestures or []) if str(x).strip()][:30]
    if g:
        parts.append("Gestures the user has taught (api.gesture(name) / 'gesture' events): " + ", ".join(g) + ".")
    if prev_code:
        parts.append(f'THE PROGRAM RUNNING NOW (written for: "{(prev_instruction or "")[:600]}"):\n{prev_code[:12000]}')
        parts.append(f'THE USER\'S CHANGE, as the character restated it to them: "{instruction[:600]}"\n{ROLES}\nRewrite the whole program '
                     "with this change. Keep the api.state field names it already uses: the live state carries over and "
                     "setup(api) runs again with it.")
    else:
        parts.append(f'THE USER\'S STANDING REQUEST, as the character restated it to them: "{instruction[:600]}"\n{ROLES}')
    if his_words and str(his_words).strip():
        parts.append(f'THE USER\'S OWN LATEST WORDS (speech transcript): "{str(his_words).strip()[:400]}"\nThe restatement above is the '
                     "plan; where it lost or changed what kind of rule the user asked for (copying them, a condition, a count, "
                     "a turn), the user's own words decide.")
    parts.append("Reply with ONLY the program.")
    return "\n\n".join(parts)


def repair_turn(check: dict) -> str:
    fix = "\n".join("- " + str(x)[:300] for x in (check.get("fix") or [check.get("error") or "it failed"])[:10])
    return ("The program failed the automatic check (a dry run on recorded camera frames):\n" + fix +
            "\nFix these and reply with ONLY the corrected program.")


async def call_model(http, name: str, system: str, messages: list, timeout: float, max_tokens: int) -> dict:
    """-> {text, usd, ms, usage}. messages = [{"role": "user"|"assistant", "content"}]."""
    import aiohttp
    route, model, extra, price = MODELS[name]
    t0 = time.monotonic()
    if route == "openai":
        key = os.environ.get("OPENAI_API_KEY") or ""
        if not key:
            raise CoderError("no OPENAI_API_KEY")
        body = {"model": model, "instructions": system, "max_output_tokens": max_tokens,
                "input": [{"role": m["role"], "content": m["content"]} for m in messages]}
        body.update(extra)
        url, hdr = "https://api.openai.com/v1/responses", {"Authorization": "Bearer " + key}
    else:
        key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            raise CoderError("no OPENROUTER_API_KEY")
        body = {"model": model, "max_tokens": max_tokens, "usage": {"include": True},
                "messages": [{"role": "system", "content": system}] + messages}
        body.update(extra)
        url, hdr = "https://openrouter.ai/api/v1/chat/completions", {"Authorization": "Bearer " + key}
    try:
        async with http.post(url, json=body, headers=hdr, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            raw = await resp.text()
            if resp.status != 200:
                raise CoderError(f"{name}: HTTP {resp.status}: {raw[:200]}")
    except asyncio.TimeoutError:
        raise CoderError(f"{name}: timeout after {timeout:.0f} s") from None
    except aiohttp.ClientError as e:
        raise CoderError(f"{name}: connection {type(e).__name__}") from None
    j = json.loads(raw)
    if route == "openai":
        text = "".join(c.get("text", "") for o in j.get("output") or [] if o.get("type") == "message"
                       for c in o.get("content") or [] if c.get("type") == "output_text")
        u = j.get("usage") or {}
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        usd = ((u.get("input_tokens", 0) - cached) * price[0] + cached * price[1] + u.get("output_tokens", 0) * price[2]) / 1e6
    else:
        if j.get("error"):
            raise CoderError(f"{name}: {str(j['error'])[:200]}")
        text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        u = j.get("usage") or {}
        usd = float(u["cost"]) if u.get("cost") is not None else (u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[2]) / 1e6
    if not text.strip():
        raise CoderError(f"{name}: empty answer")
    return {"text": text, "usd": usd, "ms": round((time.monotonic() - t0) * 1000), "usage": u}


class BehaviorCoder:
    def __init__(self, cfg=None, log=print, checker: Checker | None = None):
        c = dict(DEFAULTS)
        if isinstance(cfg, dict):
            c.update(cfg)
        elif cfg is not None:
            c["enabled"] = bool(cfg)
        # default models are on OpenRouter; with only an OpenAI key (and no model configured), use OpenAI's
        if not (isinstance(cfg, dict) and "model" in cfg) and not os.environ.get("OPENROUTER_API_KEY") \
                and os.environ.get("OPENAI_API_KEY"):
            c["model"], c["fallback"] = "gpt-5.4-mini", None
        self.cfg, self.log = c, log
        self.checker = checker or Checker(log)
        self.records: list[dict] = []

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled"))

    async def code(self, msg: dict, http, on_usd=None, model: str | None = None) -> dict:
        t0 = time.monotonic()
        rid = str(msg.get("id") or "")[:60]
        name = str(msg.get("name") or "").strip()[:60] or "behavior"
        instruction = str(msg.get("instruction") or "").strip()[:600]
        character = str(msg.get("character") or "fox")[:20]
        prev = msg.get("prev_code") if isinstance(msg.get("prev_code"), str) else None
        out = {"type": "behavior_code", "id": rid, "name": name, "ok": False}
        if not instruction:
            return dict(out, error="empty instruction", ms=0)
        usd, attempts, notes, last_err, code, check = 0.0, [], [], "", None, None
        deadline = t0 + float(self.cfg["timeout_s"])
        try:
            system, cdoc = await self.checker.doc(character)
        except CoderError as e:
            return dict(out, error=str(e), ms=round((time.monotonic() - t0) * 1000))
        user = user_turn(cdoc, instruction, prev, msg.get("prev_instruction"), msg.get("gestures"),
                         msg.get("his_words") if isinstance(msg.get("his_words"), str) else None)
        models = [model] if model else [self.cfg["model"]] + ([self.cfg["fallback"]] if self.cfg.get("fallback") else [])
        for mname in models:
            if mname not in MODELS:
                continue
            messages = [{"role": "user", "content": user}]
            for rnd in range(1 + int(self.cfg.get("repair", 1))):
                left = deadline - time.monotonic()
                if left < 1.0:
                    last_err = last_err or "out of time"
                    break
                try:
                    r = await call_model(http, mname, system, messages, left, int(self.cfg["max_tokens"]))
                except CoderError as e:
                    last_err = str(e)
                    attempts.append({"model": mname, "error": last_err[:200]})
                    break                                    # -> the fallback model
                usd += r["usd"]
                if on_usd:
                    try:
                        on_usd(r["usd"])
                    except Exception:  # noqa: BLE001
                        pass
                code = extract_code(r["text"])
                try:
                    check = await self.checker.call({"op": "check", "code": code, "character": character,
                                                     "instruction": instruction, "gestures": msg.get("gestures") or []})
                except CoderError as e:
                    check = {"ok": False, "error": str(e), "fix": []}
                attempts.append({"model": mname, "ms": r["ms"], "check_ms": check.get("ms"), "ok": bool(check.get("ok")),
                                 "fix": (check.get("fix") or [])[:5]})
                if check.get("ok"):
                    notes = check.get("notes") or []
                    res = dict(out, ok=True, code=code, model=mname, usd=round(usd, 6), attempts=attempts, notes=notes,
                               ms=round((time.monotonic() - t0) * 1000))
                    self._record(res, instruction, character)
                    return res
                last_err = check.get("error") or "the check failed"
                if not check.get("fix"):                     # checker trouble, not the program: no repair turn
                    break
                messages += [{"role": "assistant", "content": code}, {"role": "user", "content": repair_turn(check)}]
            # a program that failed its repair still beats nothing only if the model is gone; try the fallback
        res = dict(out, error=last_err[:400] or "no program", model=attempts[-1]["model"] if attempts else None,
                   usd=round(usd, 6), attempts=attempts, ms=round((time.monotonic() - t0) * 1000), code=code)
        self._record(res, instruction, character)
        return res

    def _record(self, res: dict, instruction: str, character: str) -> None:
        self.records.append({"t": time.time(), "ok": res["ok"], "ms": res["ms"], "model": res.get("model"), "usd": res.get("usd"),
                             "character": character, "instruction": instruction[:200], "rounds": len(res.get("attempts") or [])})
        if len(self.records) > 300:
            del self.records[:100]
        self.log(f"behavior: code {res.get('name')} {character} ok={res['ok']} {res['ms']} ms {res.get('model')} "
                 f"${float(res.get('usd') or 0):.4f} rounds={len(res.get('attempts') or [])}"
                 + ("" if res["ok"] else f" err={str(res.get('error'))[:100]}"))
