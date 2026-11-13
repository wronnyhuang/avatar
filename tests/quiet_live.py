"""quiet: a SMALL live measurement on gpt-realtime (paid: ~$0.3-0.6; every response is logged to the spend file).

Does the talkativeness level measurably change reply length, and how many output tokens is one spoken word (the
max_output_tokens caps in app/quiet.js POLICY are set from this)? The real session (rtc.realtime_session for the muse,
planner mode) over the Realtime WebSocket, text in / audio out, and exactly the session.update app/voice.js sends for a
level (the session's own instructions + the level's suffix, max_output_tokens). 3 levels x 3 short questions = 9
replies, plus 2 tool checks (does it call be_quiet / set_talkativeness when asked; tool-only, no audio).

    OPENAI_API_KEY=... python tests/quiet_live.py [--spend /path/spend.jsonl] [--levels brief,normal,talkative]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import rtc  # noqa: E402

QUESTIONS = ["What is your favourite season, and why?", "Tell me something about the ocean.", "How do you like to spend a rainy day?"]
TOOL_ASKS = [("Please be quiet for two minutes, I need to finish an email.", "be_quiet"),
             ("You talk too much. Shorter answers from now on, please.", "set_talkativeness")]


def policy() -> dict:
    src = ("const fs=require('fs');const m={exports:{}};new Function('module','exports',fs.readFileSync(process.argv[1],'utf8'))(m,m.exports);"
           "process.stdout.write(JSON.stringify(m.exports.POLICY))")
    out = subprocess.run(["node", "-e", src, os.path.join(HERE, "..", "app", "quiet.js")], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


class Spend:
    def __init__(self, path: str | None):
        self.path, self.usd = path, 0.0

    def add(self, usage: dict, what: str) -> None:
        usd = rtc.realtime_usd(usage)
        self.usd += usd
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "who": "quiet_live", "vendor": "openai", "usd": round(usd, 6), "what": what}) + "\n")


async def turn(ws, text: str) -> dict:
    await ws.send_json({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
    await ws.send_json({"type": "response.create"})
    said, calls = "", []
    while True:
        m = await asyncio.wait_for(ws.receive(), 60)
        if m.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"ws closed: {m.type} {m.data!r}")
        ev = json.loads(m.data)
        t = ev.get("type")
        if t == "response.output_audio_transcript.done":
            said += ev.get("transcript") or ""
        elif t == "error":
            raise RuntimeError(json.dumps(ev.get("error"))[:300])
        elif t == "response.done":
            r = ev["response"]
            for o in r.get("output") or []:
                if o.get("type") == "function_call":
                    calls.append({"name": o.get("name"), "arguments": o.get("arguments"), "call_id": o.get("call_id")})
            return {"said": said, "calls": calls, "usage": r.get("usage") or {}, "status": r.get("status"),
                    "reason": (r.get("status_details") or {}).get("reason")}


async def connect(http):
    ws = await http.ws_connect(rtc.WSS + "/realtime?model=" + rtc.REALTIME_MODEL, headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, heartbeat=20)
    sess = rtc.realtime_session("Muse", "plush companion", "marin", "server", character="muse")
    sess["audio"]["input"]["turn_detection"] = None           # text turns: every response is created here
    await ws.send_json({"type": "session.update", "session": sess})
    while True:
        ev = json.loads((await asyncio.wait_for(ws.receive(), 30)).data)
        if ev.get("type") == "session.updated":
            return ws, sess
        if ev.get("type") == "error":
            raise RuntimeError(json.dumps(ev.get("error"))[:300])


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spend", default="")
    ap.add_argument("--levels", default="brief,normal,talkative")
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", default="", help="an earlier --out file: its levels / tools are kept unless measured again")
    a = ap.parse_args()
    pol, spend, res = policy(), Spend(a.spend or None), {"model": rtc.REALTIME_MODEL, "levels": {}, "tools": []}
    old = json.load(open(a.merge, encoding="utf-8")) if a.merge else None
    if old:
        res["levels"] = dict(old.get("levels") or {})
    async with aiohttp.ClientSession() as http:
        for level in [x for x in a.levels.split(",") if x]:
            p = pol["normal" if level == "tools" else level]
            ws, sess = await connect(http)
            try:
                if level not in ("normal", "tools"):      # exactly what voice.js applyTalkWire sends
                    await ws.send_json({"type": "session.update", "session": {"type": "realtime", "max_output_tokens": p["maxTokens"] or "inf",
                                                                               "instructions": sess["instructions"] + "\n\n" + p["suffix"]}})
                rows = []
                for q in ([] if level == "tools" else QUESTIONS):
                    r = await turn(ws, q)
                    spend.add(r["usage"], f"quiet talk {level}")
                    od = r["usage"].get("output_token_details") or {}
                    w = len(r["said"].split())
                    rows.append({"q": q, "words": w, "audio_tokens": od.get("audio_tokens"), "text_tokens": od.get("text_tokens"),
                                 "output_tokens": r["usage"].get("output_tokens"), "status": r["status"], "reason": r["reason"], "said": r["said"]})
                    print(f"[{level}] {w:3d} words, {r['usage'].get('output_tokens')} out tokens ({r['status']}): {r['said'][:110]}", flush=True)
                if rows:
                    res["levels"][level] = {"mean_words": round(sum(x["words"] for x in rows) / len(rows), 1), "cap_tokens": p["maxTokens"], "rows": rows}
                if level in ("normal", "tools"):      # the two tool checks ride a normal session
                    for ask, want in TOOL_ASKS:
                        r = await turn(ws, ask)
                        spend.add(r["usage"], "quiet tool check")
                        got = [c["name"] for c in r["calls"]]
                        res["tools"].append({"ask": ask, "want": want, "called": got, "args": [c["arguments"] for c in r["calls"]], "said": r["said"], "ok": want in got})
                        print(f"[tool] {ask!r} -> {got} {[c['arguments'] for c in r['calls']]} said={r['said'][:60]!r}", flush=True)
                        for c in r["calls"]:
                            await ws.send_json({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": c["call_id"], "output": json.dumps({"ok": True})}})
            finally:
                await ws.close()
    tw = [(x["output_tokens"] or 0) / x["words"] for lv in res["levels"].values() for x in lv["rows"] if x["words"] >= 3]
    res["out_tokens_per_word"] = {"mean": round(sum(tw) / len(tw), 1), "max": round(max(tw), 1)} if tw else None
    if old and not res["tools"]:
        res["tools"] = old.get("tools") or []
    res["usd"] = round(spend.usd + (old.get("usd") or 0 if old else 0), 4)
    print(json.dumps({k: (v if k != "levels" else {l: {"mean_words": d["mean_words"], "words": [r["words"] for r in d["rows"]], "status": [r["status"] for r in d["rows"]]} for l, d in v.items()}) for k, v in res.items()}, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
