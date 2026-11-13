"""intent: can the server's streaming ASR (vosk small, full vocabulary) hear the new whole-body / prop words?

    OPENAI_API_KEY=... python3 tools/intent_bench/act_asr_check.py --out /path/result.json [--mode full|both]

Speaks ~70 new-command phrasings with OpenAI tts-1 (2 voices, cached under --cache, default <repo>/runs/cache/intent_tts), feeds each clip through the real
app/asr_worker.py subprocess in 140 ms batches (as bench.py), runs the act matcher (muse scene) on the stream, and
compares with the matcher on the clean text. Reports command recall, the words the recognizer got wrong, and latency
(end of speech -> intent at the server). Prints the estimated TTS cost (list price).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "app"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import intent  # noqa: E402
import intent_hub  # noqa: E402
from bench import SR, Worker, load_pcm, speech_end  # noqa: E402
from check_cases import summ  # noqa: E402

CACHE = os.environ.get("INTENT_TTS_CACHE") or os.path.join(HERE, "..", "..", "runs", "cache", "intent_tts")
PHRASES = [
    "do a backflip", "can you do two backflips", "do a front flip", "do a somersault", "do a cartwheel",
    "spin around three times", "twirl around", "do a pirouette", "turn away from me", "turn back around",
    "jump really high", "hop on one foot", "bounce up and down", "take three steps forward", "walk to the desk",
    "dance like a robot", "do a silly dance", "sit down", "stand up", "lie down on your back", "roll over",
    "take a bow", "curtsy", "wiggle your butt", "shake it off", "moonwalk",
    "pick up the red ball", "grab the ball with your left paw", "put it back where it belongs", "put it on the table",
    "hold it up high", "show it to me", "let me see the book", "look at it closely", "turn it around",
    "flip it over", "switch hands", "toss it up and catch it", "juggle", "catch it", "shake the box", "hug the teddy bear",
    "give it to me", "smell the flower", "drop it", "put the mug on the desk",
    "put on your headphones", "take off your headphones", "put your headphones on", "pull the chair over",
    "pull the chair up to the desk", "push the chair back", "bring the desk over here", "slide the laptop closer",
    "sit in the chair", "sit at the desk", "get up from the chair", "open the laptop", "close the laptop",
    "open the book", "type on the laptop", "get to work", "stop working", "turn back to me", "talk to me",
    "put on your headphones and sit at the desk", "pick up the laptop and open it", "spin around then sit down",
    "muse can you do a cartwheel", "pick up the yellow ball and throw it up",
]
VOICES = [("alloy", 1.0), ("nova", 1.1)]


def tts(text, voice, speed):
    os.makedirs(CACHE, exist_ok=True)
    h = hashlib.sha1(f"{text}|{voice}|{speed}".encode()).hexdigest()[:12]
    fn = os.path.join(CACHE, f"{h}.pcm")
    if os.path.exists(fn):
        return fn, 0
    body = json.dumps({"model": "tts-1", "voice": voice, "input": text, "speed": speed, "response_format": "pcm"}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/audio/speech", data=body,
                                 headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        open(fn, "wb").write(r.read())
    return fn, len(text)


def heads(cmds):
    return [summ(c).split(" ")[0] for c in cmds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="TTS clip cache dir (default $INTENT_TTS_CACHE or <repo>/runs/cache/intent_tts)")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--py", default=intent_hub.DEFAULTS["python"])
    ap.add_argument("--model", default=intent_hub.DEFAULTS["model"])
    a = ap.parse_args()
    if a.cache:
        global CACHE
        CACHE = a.cache
    chars = 0
    clips = []
    for t in PHRASES:
        for v, sp in VOICES:
            fn, c = tts(t, v, sp)
            chars += c
            clips.append((t, v, fn))
    if chars:
        print(f"tts-1: {chars} new chars, ~${chars * 15 / 1e6:.4f} est.")
    w = Worker(a.py, ["--model", a.model, "--mode", a.mode, "--rules",
                      os.path.join(HERE, "..", "..", "app", "intent_rules.json")])
    recs = ["full", "grammar"] if a.mode == "both" else [a.mode]
    scene = intent_hub.scene_for("muse")
    rows, ok, lat = [], 0, []
    for text, voice, fn in clips:
        x = load_pcm(fn)
        pad0, pad1 = np.zeros(int(0.4 * SR), np.float32), np.zeros(int(1.2 * SR), np.float32)
        y = np.concatenate([pad0, x, pad1])
        y = (y + np.random.default_rng(1).normal(0, 30, len(y))).clip(-32768, 32767).astype("<i2")
        end = 0.4 + speech_end(x)
        lines = w.run(y, len(recs))
        m = intent.Matcher(scene=scene)
        sts = {r: m.stream() for r in recs}
        got, first = [], None
        final_txt = " ".join(j["text"] for j in lines if j["t"] == "f" and j.get("text"))
        for j in lines:
            if j["r"] not in sts:
                continue
            for h in sts[j["r"]].update(j["text"], j["t"] == "f", j["a"] * 1000):
                got.append(h["cmd"])
                if first is None:
                    first = j["a"] + j["c"] / 1000 - end
        want = [h["cmd"] for h in intent.Matcher(scene=scene).parse(text)]
        # the stream may emit a refinement of a clause; compare the heads of the distinct commands in order
        gh = []
        for hh in heads(got):
            if not gh or gh[-1] != hh:
                gh.append(hh)
        hit = gh == heads(want)
        ok += hit
        if first is not None and hit:
            lat.append(first * 1000)
        rows.append({"text": text, "voice": voice, "asr": final_txt, "want": heads(want), "got": gh, "ok": hit,
                     "lat_ms": None if first is None else round(first * 1000)})
    w.close()
    lat.sort()
    res = {"mode": a.mode, "clips": len(rows), "command_recall": round(ok / len(rows), 3),
           "latency_ms": {"median": round(lat[len(lat) // 2]) if lat else None,
                          "p90": round(lat[int(len(lat) * 0.9)]) if lat else None},
           "misses": [r for r in rows if not r["ok"]], "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "misses")}))
    for r in res["misses"]:
        print(f"  MISS {r['text']!r} ({r['voice']}): heard {r['asr']!r} -> {r['got']}")


if __name__ == "__main__":
    main()
