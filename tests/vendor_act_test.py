"""B8: the hub-relayed GPT Live 1 session (app/vendor.py) carries the muse's move / object tools, and a call's
result is the page's real answer (on_act), not a blanket ok; an unanswered call says so.  <venv python> tests/vendor_act_test.py"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import rtc     # noqa: E402
import vendor  # noqa: E402

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  — {detail}" if detail else ""))
    fails += 0 if ok else 1

class FakeWS:
    def __init__(self, replies=()): self.sent, self.closed, self.replies = [], False, list(replies)
    async def send_str(self, s): self.sent.append(json.loads(s))
    async def receive(self, timeout=None):
        class M: pass
        m = M(); m.type = vendor.aiohttp.WSMsgType.TEXT; m.data = json.dumps(self.replies.pop(0)); return m
    async def close(self): self.closed = True

class FakeHTTP:
    def __init__(self, ws): self.ws = ws
    async def ws_connect(self, *a, **k): return self.ws

async def main():
    noop = lambda *a, **k: None
    # 1. session.start carries the act tools + note for an act character (and not otherwise)
    for act in (True, False):
        ws = FakeWS([{"type": "session.started", "session": {"id": "s1", "audio": {"output": {"voice": "x"}}}}])
        v = vendor.LiveVendor(FakeHTTP(ws), on_audio=noop, on_transcript=noop, on_voice_usd=noop, on_backend_usd=noop, on_end=noop,
                              log=noop, act_tools=rtc.act_tools() if act else None, act_note=rtc.ACT_NOTE if act else "",
                              on_act=(lambda n, a: asyncio.sleep(0, {"ok": True})) if act else None)
        os.environ.setdefault("OPENAI_API_KEY", "test-not-used")
        await v.open("persona", "shimmer")
        v.reader.cancel()
        r = ws.sent[0]["session"]["delegation"]["responses"]
        names = {t["name"] for t in r["tools"]}
        if act:
            check("act character: move + object tools in the GPT Live 1 session, with the act note", {"move", "object"} <= names and "`move`" in r["instructions"], sorted(names)[-4:])
        else:
            check("other characters: tools unchanged (no move/object)", not ({"move", "object"} & names) and len(r["tools"]) == len(vendor.ANIMATION_TOOLS))
    # 2. a move call returns the page's real result
    async def on_act(name, args):
        await asyncio.sleep(0.05)
        return {"ok": False, "reason": "my paws are full"} if name == "object" else {"ok": True, "did": "spinning around once"}
    ws = FakeWS(); v = vendor.LiveVendor(FakeHTTP(ws), on_audio=noop, on_transcript=noop, on_voice_usd=noop, on_backend_usd=noop, on_end=noop,
                                          log=noop, act_tools=rtc.act_tools(), on_act=on_act)
    v.ws = ws
    t1 = await v._tool({"name": "move", "call_id": "c1", "arguments": json.dumps({"name": "spin"})})
    t2 = await v._tool({"name": "object", "call_id": "c2", "arguments": json.dumps({"verb": "pick_up", "object": {"name": "ball"}})})
    check("the tool reader is not blocked by the page round trip", ws.sent == [])
    await asyncio.sleep(0.2)
    outs = {e["item"]["call_id"]: json.loads(e["item"]["output"]) for e in ws.sent}
    check("move: the output is the page's result", outs.get("c1") == {"ok": True, "did": "spinning around once"}, outs.get("c1"))
    check("object: a failure comes back as a failure with its reason", outs.get("c2") == {"ok": False, "reason": "my paws are full"}, outs.get("c2"))
    # the continuation (response.create) is sent only after the answers are posted
    ws.sent.clear(); t5 = await v._tool({"name": "move", "call_id": "c5", "arguments": "{}"})
    await v._continue_after([t5])
    check("response.create follows the act output, never before it", [e.get("type") for e in ws.sent] == ["response.item.create", "response.create"], [e.get("type") for e in ws.sent])
    # 3. no answer from the page in time: says so (ok false)
    async def never(name, args): await asyncio.sleep(30)
    ws = FakeWS(); v = vendor.LiveVendor(FakeHTTP(ws), on_audio=noop, on_transcript=noop, on_voice_usd=noop, on_backend_usd=noop, on_end=noop,
                                          log=noop, act_tools=rtc.act_tools(), on_act=never)
    v.ws = ws
    await v._tool({"name": "move", "call_id": "c3", "arguments": "{}"})
    await asyncio.sleep(4.3)
    o = json.loads(ws.sent[0]["item"]["output"]) if ws.sent else None
    check("unanswered call: ok false with a truthful reason (never a blanket ok)", bool(o) and o.get("ok") is False and "did not answer" in o.get("reason", ""), o)
    # 4. the legacy animation tools still answer at once
    ws = FakeWS(); acts = []
    v = vendor.LiveVendor(FakeHTTP(ws), on_audio=noop, on_transcript=noop, on_voice_usd=noop, on_backend_usd=noop, on_end=noop,
                          on_action=lambda n, a: acts.append(n), log=noop, act_tools=rtc.act_tools(), on_act=on_act)
    v.ws = ws
    await v._tool({"name": "wave", "call_id": "c4", "arguments": "{}"})
    check("legacy animation tool unchanged (immediate ok, on_action)", acts == ["wave"] and json.loads(ws.sent[0]["item"]["output"]) == {"ok": True, "action": "wave"})

asyncio.run(main())
print(f"{fails} failures"); sys.exit(fails)
