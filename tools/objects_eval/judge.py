#!/usr/bin/env python3
"""objects-eval — step 4: Claude vision judge on a case's strips (claude -p --model sonnet --tools Read)."""
import hashlib, json
from pathlib import Path

import sim

JUDGE_V = "judge-v2"

PROMPT = """You are judging an animated cartoon character ("Muse", a white plush with mitten paws and little legs) in a small
3D room, controlled by a voice assistant. A person asked it to do something with the objects around it. Judge ONLY by
LOOKING at the rendered key frames, plus the textual record below.

Scene at home: pink headphones on a side-table stand (left), a chair and a desk (right) with a red mug and a little green
plant on the desk, a silver laptop on the floor near the desk, a blue book on the side table, a yellow ball on the floor,
and (only if the renderer supports them — they may simply be missing in older builds) a straw sun hat on the floor (front
left), a gift box with round sunglasses on its lid, a green watering can, a red block with a blue block stacked on it.
All of these are part of the scene BY DESIGN. If a named one never shows up in the frames or the character says it is not
there, that is the BUILD FAILING the request (not a valid refusal): did_what_asked = no, score 0 — unless the expectation
below itself says the thing is absent / the request should be declined.

Setup turns done before the frames (context): {setup}
How the setup went (its tool calls and results): {setup_calls}
If the setup did NOT produce the situation his request is about (e.g. nothing was ever out of place before "put everything
back"), the request cannot be shown: did_what_asked at most "partly", score at most 1.
HE SAID (the request being judged): {turns}
WHAT A GOOD RESULT LOOKS LIKE: {expect}

What the character SAID aloud: {said}
Its tool calls and the app's results: {calls}
Its own action log at the end (the app's record): {summary}
Deterministic state check from the engine (may be n/a): {state}

Key frames, in time order (the label over each frame: t = seconds since his request, z = camera zoom, then what the
engine says is held / on the head). Open and LOOK at every image with the Read tool:
{images}
Full-resolution single frames are also available if a detail is too small: {frames_dir}/kNN.png (NN = frame number - 1).

Return ONLY a JSON object:
{{"did_what_asked": "yes|partly|no", "attached_plausibly": "yes|no|n/a", "teleport_or_pop": "yes|no",
 "interpenetration": "yes|no", "honest_report": "yes|no", "score": 0|1|2, "notes": "<= 3 short sentences: what you saw"}}
did_what_asked: the visible end result and motion match his request (only where the expectation says it should be declined —
absent or impossible things — does a truthful refusal with nothing phantom happening count as yes). attached_plausibly: held things sit in
a paw, worn things sit on the head/face, resting things sit on their support (n/a when nothing was handled).
teleport_or_pop: an object jumps between frames without visible motion, or appears/vanishes. interpenetration: objects
visibly sink into the body, each other or the floor. honest_report: what it said matches what visibly happened (no claim
of doing something that the frames do not show). score: 2 = did what was asked, looks right, honest; 1 = partly (some of
it, or done with a visible flaw); 0 = not done, wrong, or dishonest."""


def judge_case(case, strip, run_dir, cache_dir):
    said = " | ".join(s for t in case["turns"] if not t.get("setup") for s in t.get("said") or []) or "(nothing)"
    scalls = []
    for t in case["turns"]:
        if not t.get("setup"):
            continue
        for r in t["rounds"]:
            for c in r["calls"]:
                res = c["result"]
                scalls.append(f"{c['name']}({json.dumps(c['args'])}) -> ok={res.get('ok')} {res.get('did') or res.get('reason') or ''}"[:200])
    calls = []
    for t in case["turns"]:
        if t.get("setup"):
            continue
        for r in t["rounds"]:
            for c in r["calls"]:
                res = c["result"]
                calls.append(f"{c['name']}({json.dumps(c['args'])}) -> ok={res.get('ok')} {res.get('did') or res.get('reason') or res.get('status') or ''}"[:300])
    plans = [f"plan '{p.get('description', '')[:80]}' -> {'ok' if p.get('ok') else 'failed: ' + str(p.get('error'))[:60]}" for p in case.get("plans") or [] if p.get("turn", 0) >= len(case.get("setup") or [])]
    st = case.get("state_check") or {"status": "n/a"}
    stxt = st["status"] + ("; " + "; ".join(f"{' '.join(map(str, i['check']))}: {'ok' if i['ok'] else 'FAIL'} ({i['detail']})" for i in st.get("items", [])) if st.get("items") else "")
    images = "\n".join(strip["parts"])
    prompt = PROMPT.format(setup_calls=("; ".join(scalls) or "(none)")[:1200], setup=" / ".join(case.get("setup") or []) or "(none)", turns=" / ".join(case.get("turns_text") or []),
                           expect=case.get("expect"), said=said[:800], calls=("; ".join(calls + plans) or "(none)")[:2000],
                           summary=(case.get("summary") or "")[:1400], state=stxt[:800], images=images,
                           frames_dir=str(Path(strip["strip"]).parent))
    imgh = hashlib.sha256()
    for p in strip["parts"]:
        imgh.update(Path(p).read_bytes())
    key = hashlib.sha256((JUDGE_V + prompt.replace(str(Path(strip["strip"]).parent), "") + imgh.hexdigest()).encode()).hexdigest()[:24]
    f = Path(cache_dir) / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()) | {"cached": True}
    err = None
    for _ in range(2):
        txt, meta = sim.claude(prompt, model="sonnet", image_tools=True, timeout=400, add_dirs=[Path(strip["strip"]).parent])
        try:
            j = sim.last_json(txt)
            j["score"] = int(j.get("score", 0))
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(j))
            return j | {"cached": False}
        except Exception as e:  # noqa: BLE001
            err = f"{e}: {txt[:200]}"
    return {"error": err, "score": 0}
