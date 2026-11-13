"""Synthesize command / negative clips with OpenAI tts-1 (24 kHz PCM) for bench.py.
usage: OPENAI_API_KEY=... python gen_tts.py <outdir>     (prints the estimated TTS cost at list price)"""
import hashlib, json, os, sys, time, urllib.request

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
KEY = os.environ["OPENAI_API_KEY"]

POS = [l.split("|")[0].strip() for l in open(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "intent_cases.txt"))
       if "|" in l and not l.startswith("#") and l.split("|")[1].strip()]
NEG = [l.split("|")[0].strip() for l in open(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "intent_cases.txt"))
       if "|" in l and not l.startswith("#") and not l.split("|")[1].strip()]
NEG += [
    "So yesterday I took the kids to the park and we played soccer for like two hours.",
    "Can you tell me what the capital of France is?",
    "Honestly I don't know what to make for dinner tonight, maybe pasta.",
    "My daughter keeps asking me why the sky is blue.",
    "That was a really funny joke, tell me another one.",
    "What's your favorite food, Felix?",
    "I was thinking we could go camping next weekend if the weather holds up.",
    "Do you remember what we talked about yesterday?",
    "The meeting ran long so I missed the train home.",
    "How far is the moon from the earth?",
    "My son scored a goal at practice and he was so proud.",
    "I'm a little tired, it's been a long day at work.",
    "Let's play a game, you think of an animal and I'll guess.",
    "Why do foxes have such bushy tails?",
    "We watched a movie about penguins last night.",
    "I need to remember to buy milk on the way home.",
]
# avatar-style replies (echo negatives): the character talking, incl. imperative-sounding lines
AVATAR = [
    "Sure! Here's a big wave for you. Hello there!",
    "Ooh, I love that idea. Let me think about it for a second.",
    "Ha ha, okay, watch this, I'm wiggling my ears!",
    "Of course, I can raise my paw. There it is!",
    "Hmm, foxes are curious creatures. We sniff everything.",
    "You should try it too, wave back at me!",
    "Look at that, my tail is wagging all by itself.",
    "Okay, okay, I'll stop. That was fun though.",
    "Turn around and see if your dog is still sleeping.",
    "Give me a second, I'm thinking really hard.",
    "Wow, a soccer goal? That's amazing, high five!",
    "I'm smiling so much right now. You make me happy.",
]
POS_VOICES = [("alloy", 1.0), ("nova", 1.15), ("onyx", 0.9), ("shimmer", 1.3), ("echo", 1.0), ("fable", 1.1)]
jobs = []
for i, t in enumerate(POS):
    for v, sp in (POS_VOICES[i % 6], POS_VOICES[(i + 2) % 6], POS_VOICES[(i + 4) % 6]):
        jobs.append(("pos", t, v, sp))
for i, t in enumerate(NEG):
    for v, sp in (POS_VOICES[i % 6], POS_VOICES[(i + 3) % 6]):
        jobs.append(("neg", t, v, sp))
for i, t in enumerate(AVATAR):
    jobs.append(("avatar", t, ["verse", "ash", "coral", "sage"][i % 4] if False else ["alloy", "fable", "nova", "shimmer"][i % 4], 1.0))

chars = 0
man = []
for kind, text, voice, speed in jobs:
    h = hashlib.sha1(f"{kind}|{text}|{voice}|{speed}".encode()).hexdigest()[:10]
    fn = os.path.join(OUT, f"{kind}_{h}.pcm")
    man.append({"kind": kind, "text": text, "voice": voice, "speed": speed, "file": fn})
    if os.path.exists(fn):
        continue
    body = json.dumps({"model": "tts-1", "voice": voice, "input": text, "speed": speed, "response_format": "pcm"}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/audio/speech", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                pcm = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print("retry", text, e, file=sys.stderr)
            time.sleep(2)
    else:
        continue
    open(fn, "wb").write(pcm)
    chars += len(text)
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
usd = chars * 15e-6   # tts-1 list price, estimate only
print(f"{len(jobs)} clips, {chars} new chars, ~${usd:.4f} est.")
