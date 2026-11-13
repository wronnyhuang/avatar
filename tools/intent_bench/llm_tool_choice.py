"""intent: does the voice model pick the new `move` / `object` tools (and nothing on negatives)?

    OPENAI_API_KEY=... python3 tools/intent_bench/llm_tool_choice.py --out result.json [--limit N] [--max-usd 1.2]

gpt-realtime-2.1 in TEXT mode over the Realtime WebSocket with app/rtc.py's own muse session (instructions + tools,
tool_choice auto; audio/VAD swapped for text). Each phrasing is an out-of-band response (conversation "none") so the
cases are independent and share the cached prefix; a call is answered with {ok:true, did} and the model continues
(up to 4 rounds) so a compound request shows its whole ordered list. The estimated cost from response.done usage
(rtc.realtime_usd) is reported in the result and caps the run (--max-usd). Scoring: positives = the expected tool + name/verb family in order (+ key args);
negatives = no move/object/gesture/pose call at all.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "app"))
import rtc  # noqa: E402

# alias families (motion.js folds these): a model choosing either is right
FAM = {"twirl": "spin", "spin": "spin", "backflip": "flip", "frontflip": "flip", "flip": "flip", "cartwheel": "cartwheel",
       "jump": "jump", "hop": "jump", "bounce": "jump", "walk": "walk", "step": "walk", "sit": "sit", "stand": "stand",
       "stand_up": "stand", "get_up": "stand", "turn_back": "turn_back", "turn_away": "turn_away", "bow": "bow",
       "show": "show", "hold_up": "show", "offer": "show", "work_on": "work", "type_on": "work", "use": "work",
       "pull": "pull", "bring": "pull", "put_down": "put", "place": "put", "put_back": "put_back", "drop": "put",
       "sit_in": "sit", "sit_at": "sit", "inspect": "inspect", "smell": "inspect", "lie_down": "lie_down",
       "roll_over": "roll_over", "wiggle": "wiggle", "shake_off": "wiggle", "dance": "dance", "toss": "toss"}
# phrase | expected: a list of "tool:name_or_verb[:k=v,...]" ; [] = must not move
CASES = [
    # whole-body moves
    ("Can you do a backflip?", ["move:backflip"]),
    ("Do two backflips!", ["move:backflip:count=2"]),
    ("Spin around three times.", ["move:spin:count=3"]),
    ("Twirl for me, slowly.", ["move:spin:speed=slow"]),
    ("Turn around so I can see your back.", ["move:turn_away"]),
    ("Okay, turn back around and face me.", ["move:turn_back"]),
    ("Jump as high as you can.", ["move:jump"]),
    ("Hop three times.", ["move:jump:count=3"]),
    ("Take three steps forward.", ["move:walk"]),
    ("Walk backwards a little.", ["move:walk"]),
    ("Do a silly dance!", ["move:dance"]),
    ("Show me your robot dance.", ["move:dance"]),
    ("Sit down for a bit.", ["move:sit"]),
    ("Stand up.", ["move:stand"]),
    ("Lie down on the floor.", ["move:lie_down"]),
    ("Roll over!", ["move:roll_over"]),
    ("Take a bow.", ["move:bow"]),
    ("Do a cartwheel.", ["move:cartwheel"]),
    ("Wiggle your butt.", ["move:wiggle"]),
    ("Could you do a front flip?", ["move:flip"]),
    ("Turn to your left.", ["move:turn:dir=left"]),
    ("Spin to your right twice.", ["move:spin:count=2,dir=right"]),
    ("Do a somersault.", ["move:flip"]),
    ("Shake it off like a wet dog.", ["move:wiggle"]),
    ("Bounce up and down.", ["move:jump"]),
    # objects / the desk scenario
    ("Put on your headphones.", ["object:put_on:object=headphones"]),
    ("Take your headphones off.", ["object:take_off:object=headphones"]),
    ("Pull the chair over here.", ["object:pull:object=chair"]),
    ("Pull the chair up to the desk.", ["object:pull:object=chair"]),
    ("Push the chair back.", ["object:push:object=chair"]),
    ("Bring the desk over.", ["object:pull:object=desk"]),
    ("Slide the laptop closer.", ["object:slide:object=laptop"]),
    ("Sit in the chair.", ["object:sit:object=chair"]),
    ("Sit at the desk.", ["object:sit"]),
    ("Get up from the chair.", ["object|move:stand"]),
    ("Open the laptop.", ["object:open:object=laptop"]),
    ("Close the laptop.", ["object:close:object=laptop"]),
    ("Get to work.", ["object:work"]),
    ("Type on your laptop for a bit.", ["object:work:object=laptop"]),
    ("Okay, turn back to me.", ["object|move:turn_back"]),
    ("Pick up the laptop.", ["object:pick_up:object=laptop"]),
    ("Can you pick up the headphones?", ["object:pick_up:object=headphones"]),
    ("Put it on the desk.", ["object:put"]),
    ("Put the laptop on the desk.", ["object:put:object=laptop"]),
    ("Pick up the mug and show it to me.", ["object:pick_up:object=mug", "object:show"]),
    ("Hug the teddy bear.", ["object:hug"]),
    ("Toss the ball up and catch it.", ["object:toss"]),
    ("Give me the book.", ["object:show"]),
    ("Put the headphones back where they were.", ["object:put_back:object=headphones"]),
    ("Stop typing.", ["object|move:stop"]),
    # compound one-shot requests -> an ordered list
    ("Put on your headphones and sit at the desk.", ["object:put_on", "object:sit"]),
    ("Pull the chair up to the desk, sit down and open the laptop.", ["object:pull", "object|move:sit", "object:open"]),
    ("Spin around and then do a backflip.", ["move:spin", "move:backflip"]),
    ("Take off your headphones and stand up.", ["object:take_off", "object|move:stand"]),
    ("Close the laptop, stand up and push the chair back.", ["object:close", "object|move:stand", "object:push"]),
    ("Pick up the laptop, put it on the desk and open it.", ["object:pick_up", "object:put", "object:open"]),
    # negatives: state / history questions, negations, hypotheticals, mentions, chit-chat
    ("Are you wearing headphones right now?", []),
    ("What did you do a minute ago?", []),
    ("Did you spin earlier?", []),
    ("Don't do a backflip.", []),
    ("Please don't pick up the laptop.", []),
    ("Never mind, don't sit down.", []),
    ("If you could do a backflip, would you?", []),
    ("My son did a backflip at the gym today.", []),
    ("I bought a new laptop yesterday.", []),
    ("I'm sitting at my desk at work.", []),
    ("I love dancing with my kids.", []),
    ("The chair in my office is so uncomfortable.", []),
    ("What's your favorite kind of dance?", []),
    ("Do you like headphones or speakers better?", []),
    ("How many backflips can a cat do?", []),
    ("What's on your desk?", []),
    ("Tell me a joke.", []),
    ("I'm going to put on my headphones and go for a run.", []),
    ("Is the laptop open?", []),
    ("Where is the chair?", []),
    ("Can a plush toy jump?", []),
    ("I wish I could spin like that.", []),
    ("My daughter keeps pulling her chair up to the table.", []),
    ("Stop asking me questions.", []),
    # added after run 1 (written before run 2, not used to tune the prompt)
    ("Would you mind doing a little dance?", ["move:dance"]),
    ("Hand me the mug.", ["object:show:object=mug"]),
    ("Could you sit at your desk and open your laptop?", ["object:sit", "object:open"]),
    ("Please slide your laptop away from you.", ["object:slide:object=laptop"]),
    ("My friend can do three cartwheels in a row.", []),
    ("I used to dance ballet when I was little.", []),
    ("Don't take your headphones off yet.", []),
    ("Do you think you could ever do a backflip?", []),
]


def session(name="Muse", what="plush companion", voice="shimmer"):
    s = rtc.realtime_session(name, what, voice, "server", character="muse")
    s["output_modalities"] = ["text"]
    s.pop("audio", None)
    return s


def parse_exp(e):
    tool, name, *rest = e.split(":")
    args = dict(kv.split("=", 1) for kv in rest[0].split(",")) if rest else {}
    return tool.split("|"), name, args


def fam(x):
    return FAM.get(x, x)


def score(exp, calls):
    act = [c for c in calls if c["name"] in ("move", "object", "gesture", "pose")]
    if not exp:
        return not act, "fired " + ", ".join(f"{c['name']}:{c['args'].get('name') or c['args'].get('verb')}" for c in act) if act else ""
    got = [c for c in calls if c["name"] in ("move", "object", "gesture", "pose", "look", "release")]
    i, why = 0, []
    for e in exp:
        tools, name, args = parse_exp(e)
        ok = False
        while i < len(got):
            c = got[i]
            i += 1
            v = c["args"].get("name") if c["name"] == "move" else c["args"].get("verb")
            if c["name"] in tools and (fam(v) == fam(name) or (name == "stop" and v in ("stop", "stop_activity"))):
                bad = []
                for k, want in args.items():
                    have = c["args"].get(k)
                    if k == "object":
                        have = (have or {}).get("name") if isinstance(have, dict) else have
                        if not have or want not in str(have).replace(" ", ""):
                            bad.append(f"object={have}")
                    elif str(have) != want:
                        bad.append(f"{k}={have}")
                if bad:
                    why.append(f"{e}: args {bad}")
                ok = True
                break
        if not ok:
            why.append(f"missing {e}")
            return False, "; ".join(why)
    return not why, "; ".join(why)


async def run(cases, out, max_usd):
    key = os.environ["OPENAI_API_KEY"]
    url = rtc.WSS + "/realtime?model=" + rtc.REALTIME_MODEL
    rows, usd = [], 0.0
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(url, headers={"Authorization": "Bearer " + key}, heartbeat=20) as ws:
            await ws.send_json({"type": "session.update", "session": session()})

            async def response(inp):
                nonlocal usd
                await ws.send_json({"type": "response.create", "response": {"conversation": "none", "input": inp,
                                                                            "output_modalities": ["text"]}})
                t0 = time.monotonic()
                while True:
                    m = await asyncio.wait_for(ws.receive(), 60)
                    if m.type != aiohttp.WSMsgType.TEXT:
                        raise RuntimeError(f"ws {m.type}")
                    ev = json.loads(m.data)
                    if ev["type"] == "error":
                        raise RuntimeError(json.dumps(ev.get("error"))[:300])
                    if ev["type"] == "response.done":
                        r = ev["response"]
                        usd += rtc.realtime_usd(r.get("usage") or {})
                        return r, time.monotonic() - t0

            for text, exp in cases:
                if usd > max_usd:
                    print(f"stopping: ${usd:.3f} > ${max_usd}")
                    break
                inp = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]
                calls, said, lat = [], [], None
                for _ in range(4):
                    r, dt = await response(inp)
                    lat = lat if lat is not None else dt
                    fcs = [o for o in r.get("output", []) if o.get("type") == "function_call"]
                    for o in r.get("output", []):
                        if o.get("type") == "message":
                            said += [c.get("text", "") for c in o.get("content", []) if c.get("type") in ("output_text", "text")]
                    if not fcs:
                        break
                    for o in fcs:
                        try:
                            a = json.loads(o.get("arguments") or "{}")
                        except ValueError:
                            a = {"_bad": o.get("arguments")}
                        calls.append({"name": o["name"], "args": a})
                        inp = inp + [{"type": "function_call", "call_id": o["call_id"], "name": o["name"], "arguments": o.get("arguments") or "{}"},
                                     {"type": "function_call_output", "call_id": o["call_id"],
                                      "output": json.dumps({"ok": True, "did": "doing it"})}]
                ok, why = score(exp, calls)
                rows.append({"text": text, "expect": exp, "calls": calls, "said": " ".join(said)[:200], "ok": ok, "why": why,
                             "first_response_s": round(lat or 0, 2)})
                print(("ok  " if ok else "MISS"), text, "->", [f"{c['name']}:{c['args'].get('name') or c['args'].get('verb') or ''}" for c in calls], why)
    pos = [r for r in rows if r["expect"]]
    neg = [r for r in rows if not r["expect"]]
    res = {"model": rtc.REALTIME_MODEL, "mode": "text over the Realtime WebSocket, rtc.realtime_session('Muse', 'plush companion') tools + instructions",
           "cases": len(rows), "positives": len(pos), "positive_ok": sum(r["ok"] for r in pos),
           "negatives": len(neg), "negative_clean": sum(r["ok"] for r in neg), "usd": round(usd, 4),
           "median_first_response_s": sorted(r["first_response_s"] for r in rows)[len(rows) // 2] if rows else None, "rows": rows}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=1.2)
    a = ap.parse_args()
    cases = CASES[:a.limit] if a.limit else CASES
    asyncio.run(run(cases, a.out, a.max_usd))


if __name__ == "__main__":
    main()
