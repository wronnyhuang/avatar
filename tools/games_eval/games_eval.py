#!/usr/bin/env python3
"""games eval: game / standing-behaviour requests through the REAL voice model + the REAL tool path.

  python3 tools/games_eval/games_eval.py --app <APP_ROOT> --set <rows.jsonl> \
      --out <dir> [--voice rt|gemini] [--ids a,b] [--limit N] [--par 4] [--tag before]

Per row (one fresh voice session + one games_host.cjs page per row, character = row.character):
  his turns go in as TEXT user messages to the candidate's own session (rtc.realtime_session(..., planner=True) of
  <APP_ROOT>; output text instead of audio — the tools, instructions and turn semantics are the deployed ones). Every
  tool call runs through the page side (games_host.cjs = voicetools.js + behavior.js wired as <APP_ROOT>/app/page.html
  wires them); {type:'behavior_code'} goes to <APP_ROOT>'s own app/behavior_coder.py (the server's coder, real model);
  the coder's answer goes back into the page (Behaviors.onCode / vt.onHub); every silent note the page makes goes to
  the voice model with voice.js prompt() semantics (realtime: the note item at once; respond -> response.create when idle).
  After the last turn settles: the running programs (Behaviors.list) are dry-run on the row's simulated signals
  (behavior_sim.cjs) and the row's `react` checks are scored on the program's trace / actions / emits.
Scores: started-running rate, correct-reaction rate, false "can't" rate (a refusal said while / after a program was
started or being written), time from the start_behavior call to running, control rows (no start_behavior).
Env: OPENAI_API_KEY (realtime voice + coder), GEMINI_API_KEY / GOOGLE_API_KEY (--voice gemini). Estimated model cost
(list price) is recorded per row under "usd".
"""
import argparse, asyncio, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SILENT = 'Silent context (not words he said) - never read aloud, quote, describe or mention it: '
CANT = re.compile(r"\b(can(?:'|’)?t(?! wait)|cannot(?! wait)|can not(?! wait)|unable|not able|isn(?:'|’)?t (?:possible|available)|won(?:'|’)?t be able|"
                  r"don(?:'|’)?t have (?:a way|the ability|the connection)|missing the connection|not available|"
                  r"could(?:n(?:'|’)?t| not) (?:be )?(?:set|start|make|do)|no way to)\b", re.I)
PAGE_TOOLS = ("start_behavior", "change_behavior", "stop_behavior", "list_behaviors", "look", "set_style", "learn_gesture",
              "attention_rule", "see_camera")


class Host:
    """games_host.cjs: one page per row"""
    def __init__(self, root, character):
        self.p = subprocess.Popen(["node", str(HERE / "games_host.cjs"), "--app", str(root), "--character", character],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.n, self.waiters, self.events, self.ready = 0, {}, asyncio.Queue(), None
        self.reader = asyncio.get_event_loop().run_in_executor(None, self._read)
        self.loop = asyncio.get_event_loop()

    def _read(self):
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            self.loop.call_soon_threadsafe(self._on, m)

    def _on(self, m):
        if "re" in m and m["re"] in self.waiters:
            self.waiters.pop(m["re"]).set_result(m)
        elif m.get("ev") == "ready":
            self.ready = m
        else:
            self.events.put_nowait(m)

    async def op(self, op, **kw):
        self.n += 1
        fut = self.loop.create_future(); self.waiters[self.n] = fut
        self.p.stdin.write(json.dumps(dict(kw, op=op, id=self.n)) + "\n"); self.p.stdin.flush()
        return await asyncio.wait_for(fut, 60)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "quit", "id": 0}) + "\n"); self.p.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(3)
        except Exception:  # noqa: BLE001
            self.p.kill()


class Item:
    def __init__(self, a, row, rtc, app, coder, http):
        self.a, self.row, self.rtc, self.app, self.coder, self.http = a, row, rtc, app, coder, http
        self.t0 = time.monotonic()
        self.rec = {"id": row["id"], "category": row.get("category"), "character": row.get("character"), "turns": [],
                    "calls": [], "notes": [], "said": [], "coder": [], "bev": [], "errors": []}
        self.resp_active, self.notes_q, self.last_act, self.coder_busy = set(), [], time.monotonic(), 0
        self.rt_usd, self.coder_usd = 0.0, 0.0

    def w(self):
        return round(time.monotonic() - self.t0, 3)

    def send(self, m):
        self.outq.put_nowait(m)

    async def _writer(self):
        while True:
            m = await self.outq.get()
            if m is None:
                return
            await self.ws.send_json(m)

    # ---- page events
    async def host_pump(self):
        while True:
            m = await self.host.events.get()
            ev = m.get("ev")
            if ev == "note":
                o = m.get("opt") or {}
                self.rec["notes"].append({"w": self.w(), "text": m["text"][:300], "respond": bool(o.get("respond"))})
                self.send({"type": "conversation.item.create", "item": {"type": "message", "role": "system",
                           "content": [{"type": "input_text", "text": SILENT + m["text"][:700]}]}})
                if o.get("respond"):
                    self.notes_q.append({"at": time.monotonic(), "wait": (o.get("maxWaitMs") or 2500) / 1000})
                    self.pump()
                self.last_act = time.monotonic()
            elif ev == "ctl":
                msg = m.get("msg") or {}
                if msg.get("type") == "behavior_code":
                    asyncio.create_task(self.code(msg))
            elif ev == "bev":
                e = m.get("e") or {}
                self.rec["bev"].append({"w": self.w(), "type": e.get("type"), "phase": e.get("phase"), "name": e.get("name"),
                                        "error": (str(e.get("error"))[:160] if e.get("error") else None)})

    async def code(self, msg):
        self.coder_busy += 1; t = time.monotonic()
        msg = dict(msg)
        if msg.get("character") not in self.app.CHARACTERS:
            msg["character"] = self.row["character"]
        try:
            def on_usd(u):
                self.coder_usd += float(u or 0)
            r = await self.coder.code(msg, self.http, on_usd=on_usd)
        except Exception as e:  # noqa: BLE001
            r = {"type": "behavior_code", "id": msg.get("id"), "ok": False, "name": msg.get("name"), "error": f"coder error {type(e).__name__}"}
        self.rec["coder"].append({"w": self.w(), "name": msg.get("name"), "ok": r.get("ok"), "error": r.get("error"),
                                  "ms": round((time.monotonic() - t) * 1000), "model": r.get("model"), "instruction": msg.get("instruction", "")[:300]})
        h = await self.host.op("hub", msg=r)
        self.rec["coder"][-1]["handled"] = h.get("handled")
        self.coder_busy -= 1; self.last_act = time.monotonic()

    def pump(self):
        while self.notes_q and not self.resp_active and not self.creating:
            n = self.notes_q.pop(0)
            if time.monotonic() - n["at"] > n["wait"] + 3:
                continue
            self.create()
            break

    def create(self):
        self.creating = True
        self.send({"type": "response.create"})

    # ---- the voice socket
    async def receive(self):
        calls_in = {}
        while True:
            msg = await self.ws.receive()
            if msg.type.name != "TEXT":
                return
            e = json.loads(msg.data); ty = e.get("type", "")
            if ty == "response.created":
                self.creating = False
                self.resp_active.add((e.get("response") or {}).get("id")); self.last_act = time.monotonic()
            elif ty in ("response.output_text.done", "response.output_audio_transcript.done"):
                txt = e.get("text") or e.get("transcript") or ""
                if txt.strip():
                    self.rec["said"].append({"w": self.w(), "text": txt})
                    await self.host.op("avatar", text=txt)
            elif ty == "response.function_call_arguments.done":
                name, cid = e.get("name"), e.get("call_id")
                try:
                    args = json.loads(e.get("arguments") or "{}")
                except ValueError:
                    args = {}
                if name in PAGE_TOOLS and name in (self.host.ready or {}).get("tools", []) + ["look", "set_style", "learn_gesture", "attention_rule", "see_camera"]:
                    res = (await self.host.op("call", name=name, args=args)).get("result")
                elif name in PAGE_TOOLS:
                    res = (await self.host.op("call", name=name, args=args)).get("result")
                else:
                    res = {"ok": True, "status": "starting"}
                if name == "see_camera":
                    res = {"ok": True, "seen": "him at his desk, looking at the screen"}
                self.rec["calls"].append({"w": self.w(), "name": name, "args": args, "result": res})
                self.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": cid,
                                                                         "output": json.dumps(res if res is not None else {"ok": True})}})
                calls_in[e.get("response_id")] = True
                self.last_act = time.monotonic()
            elif ty == "response.done":
                r = e.get("response") or {}
                self.resp_active.discard(r.get("id")); self.creating = False
                outs = r.get("output") or []
                spoke = any(x.get("type") == "message" for x in outs)
                if any(x.get("type") == "function_call" for x in outs) and not spoke and r.get("status") == "completed":
                    self.create()                       # voice.js tool_followup
                u = r.get("usage") or {}
                self.rt_usd += usd_rt(u)
                self.last_act = time.monotonic()
                self.pump()
            elif ty == "error":
                self.rec["errors"].append(str(e.get("error"))[:200])
                self.creating = False

    def busy(self):
        return bool(self.resp_active or self.creating or self.coder_busy or self.notes_q)

    async def settle(self, max_s, quiet_s=2.5):
        t = time.monotonic()
        while time.monotonic() - t < max_s:
            await asyncio.sleep(0.25)
            if self.busy() or time.monotonic() - self.last_act < quiet_s:
                continue
            pend = (await self.host.op("list")).get("pending") or 0
            if not pend:
                return True
        return False

    async def finish(self):
        row = self.row
        lst = await self.host.op("list")
        self.rec["running"] = [{"name": x.get("name"), "status": x.get("status"), "instruction": (x.get("instruction") or "")[:300]}
                               for x in (lst.get("list") or [])]
        if row.get("script"):
            d = await self.host.op("dryrun", script=row["script"], maxS=row["script"].get("dur"))
            self.rec["runs"] = d.get("runs") or []
        else:
            self.rec["runs"] = []

    async def run(self):
        row, rtc, app = self.row, self.rtc, self.app
        self.host = Host(self.a.app, row["character"])
        for _ in range(100):
            if self.host.ready:
                break
            await asyncio.sleep(0.05)
        self.rec["wire"] = (self.host.ready or {}).get("wire")
        nm, what, vc = app.CHARACTERS[row["character"]]
        sess = rtc.realtime_session(nm, what, vc, "server:300", "", "parallel", character=row["character"], planner=True)
        sess["output_modalities"] = ["text"]
        try:
            sess["audio"]["input"]["turn_detection"] = None
        except Exception:  # noqa: BLE001
            pass
        self.rec["tools"] = [t.get("name") for t in sess.get("tools", [])]
        url = rtc.WSS + "/realtime?model=" + rtc.REALTIME_MODEL
        self.creating = False
        async with self.http.ws_connect(url, headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, heartbeat=20, max_msg_size=0) as ws:
            self.ws = ws; self.outq = asyncio.Queue()
            wr = asyncio.create_task(self._writer()); rc = asyncio.create_task(self.receive()); hp = asyncio.create_task(self.host_pump())
            self.send({"type": "session.update", "session": sess})
            await asyncio.sleep(0.5)
            for k, text in enumerate(row["turns"]):
                self.rec["turns"].append({"w": self.w(), "text": text})
                await self.host.op("user", text=text)
                self.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                                                                         "content": [{"type": "input_text", "text": text}]}})
                self.create(); self.last_act = time.monotonic()
                ok = await self.settle(self.a.max_s)
                self.rec["turns"][-1]["settled"] = ok
                self.rec["turns"][-1]["end_w"] = self.w()
            await self.finish()
            self.outq.put_nowait(None)
            for t in (rc, hp):
                t.cancel()
            await asyncio.gather(wr, rc, hp, return_exceptions=True)
        self.host.close()
        self.rec["usd"] = {"realtime": round(self.rt_usd, 5), "coder": round(self.coder_usd, 5)}
        score(self.rec, row)
        return self.rec


class GeminiItem(Item):
    """Gemini 3.8 Live (gemini_live.setup_for of <APP_ROOT>, planner mode): his turns and the page's notes as clientContent
    (notes sent only while the model is idle, turnComplete = respond, voice.js semantics); toolCall -> page -> toolResponse."""
    def note_send(self, text, respond):
        self.gem_pending.append((text, respond, time.monotonic()))
        self.gpump()

    def gpump(self):
        while self.gem_pending and not self.gen_active:
            text, respond, at = self.gem_pending.pop(0)
            self.send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": bool(respond)}})
            if respond:
                self.gen_active = True
                break

    async def host_pump(self):
        while True:
            m = await self.host.events.get()
            ev = m.get("ev")
            if ev == "note":
                o = m.get("opt") or {}
                self.rec["notes"].append({"w": self.w(), "text": m["text"][:300], "respond": bool(o.get("respond"))})
                self.note_send(SILENT + m["text"][:700], bool(o.get("respond")))
                self.last_act = time.monotonic()
            elif ev == "ctl":
                msg = m.get("msg") or {}
                if msg.get("type") == "behavior_code":
                    asyncio.create_task(self.code(msg))
            elif ev == "bev":
                e = m.get("e") or {}
                self.rec["bev"].append({"w": self.w(), "type": e.get("type"), "phase": e.get("phase"), "name": e.get("name"),
                                        "error": (str(e.get("error"))[:160] if e.get("error") else None)})

    def busy(self):
        return bool(self.gen_active or self.coder_busy or self.gem_pending)

    async def receive(self):
        buf = ""
        while True:
            msg = await self.ws.receive()
            if msg.type.name not in ("TEXT", "BINARY"):
                return
            m = json.loads(msg.data)
            if "setupComplete" in m:
                self.setup_ok = True
            sc = m.get("serverContent")
            if sc:
                self.gen_active = True; self.last_act = time.monotonic()
                ot = (sc.get("outputTranscription") or {}).get("text")
                if ot:
                    buf += ot
                for p in ((sc.get("modelTurn") or {}).get("parts") or []):
                    if p.get("text") and not p.get("thought"):
                        buf += p["text"]
                if sc.get("turnComplete") or sc.get("interrupted"):
                    if buf.strip():
                        self.rec["said"].append({"w": self.w(), "text": buf.strip()})
                        await self.host.op("avatar", text=buf.strip())
                    buf = ""; self.gen_active = False; self.gpump()
            if m.get("toolCall"):
                self.gen_active = True
                resps = []
                for fc in m["toolCall"].get("functionCalls") or []:
                    name, args = fc.get("name"), fc.get("args") or {}
                    if name in PAGE_TOOLS:
                        res = (await self.host.op("call", name=name, args=args)).get("result")
                    else:
                        res = {"ok": True, "status": "starting"}
                    if name == "see_camera":
                        res = {"ok": True, "seen": "him at his desk, looking at the screen"}
                    self.rec["calls"].append({"w": self.w(), "name": name, "args": args, "result": res})
                    resps.append({"id": fc.get("id"), "name": name, "response": res if isinstance(res, dict) else {"result": res}})
                self.send({"toolResponse": {"functionResponses": resps}})
                self.last_act = time.monotonic()
            um = m.get("usageMetadata")
            if um:
                self.g_usage = um

    async def run(self):
        import gemini_live
        row, app = self.row, self.app
        self.host = Host(self.a.app, row["character"])
        for _ in range(100):
            if self.host.ready:
                break
            await asyncio.sleep(0.05)
        self.rec["wire"] = (self.host.ready or {}).get("wire")
        nm, what, vc = app.CHARACTERS[row["character"]]
        sf = gemini_live.setup_for("gemini", nm, what, vc, row["character"], planner=True)
        setup = sf["setup"]
        setup.pop("realtimeInputConfig", None)
        self.rec["tools"] = [f["name"] for t in setup.get("tools", []) for f in t.get("functionDeclarations", [])]
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        url = ("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService."
               "BidiGenerateContent?key=" + key)
        self.gen_active, self.gem_pending, self.setup_ok, self.g_usage = False, [], False, {}
        async with self.http.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
            self.ws = ws; self.outq = asyncio.Queue()
            wr = asyncio.create_task(self._writer()); rc = asyncio.create_task(self.receive()); hp = asyncio.create_task(self.host_pump())
            self.send({"setup": setup})
            for _ in range(100):
                if self.setup_ok or rc.done():
                    break
                await asyncio.sleep(0.1)
            if not self.setup_ok:
                raise RuntimeError("gemini setup failed: " + (repr(rc.exception()) if rc.done() and rc.exception() else "no setupComplete"))
            for text in row["turns"]:
                self.rec["turns"].append({"w": self.w(), "text": text})
                await self.host.op("user", text=text)
                self.send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}})
                self.gen_active = True; self.last_act = time.monotonic()
                ok = await self.settle(self.a.max_s)
                self.rec["turns"][-1]["settled"] = ok
            await self.finish()
            self.outq.put_nowait(None)
            for t in (rc, hp):
                t.cancel()
            await asyncio.gather(wr, rc, hp, return_exceptions=True)
        self.host.close()
        self.rec["usd"] = {"gemini_live_est": 0.0, "coder": round(self.coder_usd, 5)}
        self.rec["gemini_usage"] = self.g_usage
        score(self.rec, row)
        return self.rec


def usd_rt(u):
    """gpt-realtime-2.1 list prices (per 1M): text in 4 / cached 0.4 / out 16; audio in 32 / out 64"""
    i = u.get("input_token_details") or {}; o = u.get("output_token_details") or {}
    cached = (i.get("cached_tokens") or 0)
    tin = (i.get("text_tokens") or 0); ain = (i.get("audio_tokens") or 0)
    tout = (o.get("text_tokens") or u.get("output_tokens") or 0); aout = (o.get("audio_tokens") or 0)
    return ((max(0, tin - cached)) * 4 + cached * 0.4 + ain * 32 + tout * 16 + aout * 64) / 1e6


# ---------------------------------------------------------------------------------------------------- scoring
def series(run, ctl):
    return [(x["t"], (x.get("s") or {}).get(ctl, 0.0)) for x in (run.get("trace") or []) if "t" in x]


def rng_in(run, ctls, w0, w1):
    best = 0.0
    for c in ctls:
        v = [val for t, val in series(run, c) if w0 - 0.5 <= t <= w1]
        if v:
            best = max(best, max(v) - min(v))
    return best


def acts_in(run, w0, w1):
    return [a for a in (run.get("actions") or []) if isinstance(a, dict) and w0 - 0.2 <= float(a.get("t", -1)) <= w1 + 0.8]


def emits_in(run, w0, w1):
    return [e for e in (run.get("emits") or []) if w0 - 0.2 <= float(e.get("t", -1)) <= w1 + 0.8]


def check_one(c, run):
    k = c.get("kind")
    W = c.get("windows") or []
    if k == "move":
        return all(rng_in(run, c["controls"], a, b) >= c.get("min_range", 0.2) for a, b in W)
    if k == "move_or_action":
        return all(rng_in(run, c["controls"], a, b) >= c.get("min_range", 0.2) or acts_in(run, a, b) for a, b in W)
    if k == "quiet":
        return all(rng_in(run, c.get("controls") or [], a + 0.5, b) <= c.get("max_range", 0.15) for a, b in W)
    if k == "emit":
        return all(len(emits_in(run, a, b)) >= c.get("min", 1) for a, b in W)
    if k == "action":
        return all(len(acts_in(run, a, b)) >= c.get("min", 1) for a, b in W)
    return False


def score(rec, row):
    names = [c["name"] for c in rec["calls"]]
    exp = row.get("expect_tools") or []
    sc = {}
    sc["control"] = not exp
    started_calls = [c for c in rec["calls"] if c["name"] in ("start_behavior", "change_behavior")]
    sc["called_start"] = "start_behavior" in names
    sc["tools_ok"] = all(t in names for t in exp) if exp else ("start_behavior" not in names)
    running = [r for r in rec.get("running") or [] if r.get("status") in (None, "running")]
    sc["n_running"] = len(running)
    bev_start = [b for b in rec["bev"] if b["type"] in ("behavior", "program") and b["phase"] in ("start", "started", "change", "changed")]
    first_call = next((c["w"] for c in rec["calls"] if c["name"] == "start_behavior"), None)
    on = next((b["w"] for b in bev_start), None)
    sc["t_running_s"] = round(on - first_call, 2) if on is not None and first_call is not None else None
    if row.get("expect_stopped"):
        sc["started_running"] = bool(bev_start) or any(c.get("ok") for c in rec["coder"])
        sc["reaction_ok"] = sc["n_running"] == 0 and "stop_behavior" in names
    elif exp:
        sc["started_running"] = sc["n_running"] > 0
        runs = [r for r in rec.get("runs") or [] if r.get("ok") is not False]
        checks = row.get("react") or []
        sc["checks"] = [any(check_one(c, r) for r in runs) for c in checks]
        sc["reaction_ok"] = sc["started_running"] and all(sc["checks"])
    else:
        sc["started_running"] = False
        sc["reaction_ok"] = sc["tools_ok"]
    # a refusal said after the program was asked for, while it was being written or once it ran
    said_after = [s for s in rec["said"] if first_call is not None and s["w"] >= first_call - 0.5]
    said_before = [s for s in rec["said"] if first_call is None or s["w"] < first_call - 0.5]
    sc["false_cant"] = bool(exp) and (sc["n_running"] > 0 or row.get("expect_stopped")) and any(CANT.search(s["text"]) for s in rec["said"])
    sc["cant_any"] = any(CANT.search(s["text"]) for s in rec["said"])
    sc["cant_text"] = [s["text"][:200] for s in rec["said"] if CANT.search(s["text"])][:3]
    sc["coder_fail"] = [c.get("error") for c in rec["coder"] if not c.get("ok")]
    sc["pass"] = bool(sc["reaction_ok"] and sc["tools_ok"] and not sc["false_cant"]) if exp else bool(sc["tools_ok"])
    rec["score"] = sc


def summarize(recs):
    g = [r for r in recs if r["category"] != "control"]
    c = [r for r in recs if r["category"] == "control"]
    def rate(xs, k):
        return f"{sum(1 for r in xs if r['score'].get(k))}/{len(xs)}"
    tr = sorted(r["score"]["t_running_s"] for r in g if r["score"].get("t_running_s") is not None)
    p50 = tr[len(tr) // 2] if tr else None
    out = {"n": len(recs), "called_start": rate(g, "called_start"), "started_running": rate(g, "started_running"),
           "reaction_ok": rate(g, "reaction_ok"), "false_cant": rate(g, "false_cant"), "cant_any": rate(g, "cant_any"),
           "pass": rate(g, "pass"), "control_ok": rate(c, "pass"), "t_running_p50_s": p50, "t_running_max_s": tr[-1] if tr else None,
           "usd": round(sum(sum(r.get("usd", {}).values()) for r in recs), 4)}
    cats = {}
    for r in g:
        cats.setdefault(r["category"], []).append(r)
    out["by_category"] = {k: rate(v, "pass") for k, v in sorted(cats.items())}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--voice", default="rt", choices=["rt", "gemini"])
    ap.add_argument("--rescore", action="store_true", help="re-score the saved rows in --out (no model calls)")
    a = ap.parse_args()
    a.app = str(Path(a.app).resolve())
    sys.path.insert(0, os.path.join(a.app, "app"))
    import aiohttp, rtc, app, behavior_coder  # noqa: E401
    rows = [json.loads(l) for l in open(a.set) if l.strip()]
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = {}
    for p in out.glob("item_*.json"):          # resumable: finished rows are kept
        try:
            j = json.load(open(p)); done[j["id"]] = j
        except Exception:  # noqa: BLE001
            pass
    if a.rescore:
        byid = {r["id"]: r for r in rows}
        recs = [done[i] for i in byid if i in done]
        for r in recs:
            if "error" not in r:
                score(r, byid[r["id"]])
                json.dump(r, open(out / f"item_{r['id']}.json", "w"), indent=1)
        summ = summarize(recs); summ.update({"tag": a.tag, "app": a.app, "set": a.set})
        json.dump(summ, open(out / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); return
    coder = behavior_coder.BehaviorCoder(None, log=lambda *x: None)
    sem = asyncio.Semaphore(a.par)
    async with aiohttp.ClientSession() as http:
        async def one(row):
            if row["id"] in done:
                return done[row["id"]]
            async with sem:
                for attempt in range(2):
                    try:
                        rec = await asyncio.wait_for((GeminiItem if a.voice == 'gemini' else Item)(a, row, rtc, app, coder, http).run(), a.max_s * (len(row["turns"]) + 1) + 60)
                        break
                    except Exception as e:  # noqa: BLE001
                        rec = {"id": row["id"], "category": row.get("category"), "error": repr(e)[:300],
                               "score": {"pass": False}, "calls": [], "said": []}
                json.dump(rec, open(out / f"item_{row['id']}.json", "w"), indent=1)
                s = rec.get("score", {})
                print(f"{row['id']:28s} pass={s.get('pass')} run={s.get('started_running')} react={s.get('reaction_ok')} "
                      f"cant={s.get('cant_any')} t_run={s.get('t_running_s')} calls={[c['name'] for c in rec.get('calls', [])]} "
                      f"coder_fail={s.get('coder_fail')}", flush=True)
                return rec
        recs = await asyncio.gather(*(one(r) for r in rows))
    summ = summarize(recs)
    summ.update({"tag": a.tag, "app": a.app, "set": a.set})
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
