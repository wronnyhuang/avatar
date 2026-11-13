"""intent: regenerate tests/intent_act_vectors.json — the Python matcher's output (with the mock scenes) on every
phrase of the intent sets (dev / holdout / holdout_muse) + streaming replays, so tests/intent_test.mjs can pin app/intent.js to the same commands.
(Outputs only, no labels.)   python3 tests/act_vectors_gen.py"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import intent  # noqa: E402

scene = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "props_mock", "*.json")))]
texts = []
for fn in sorted(os.path.join(HERE, f) for f in ("intent_dev.txt", "intent_holdout.txt", "intent_holdout_muse.txt")):
    for ln in open(fn, encoding="utf-8"):
        if "|" in ln and not ln.lstrip().startswith("#"):
            texts.append(ln.split("|")[0].strip())
# intent c2: the fresh gpt-4.1-mini sets + probes of the c2 engine paths (negation scope, spoken order, topicalised
# objects, 'like' fillers, flips, put-on-head, furniture target + place)
for fn in sorted(glob.glob(os.path.join(HERE, "intent_c2_*.json"))):
    texts += [it["text"] for it in json.load(open(fn, encoding="utf-8"))["items"]]
texts += ["dont sit just pull the chair over here", "dont jump and spin", "skip the backflip and do a twirl",
          "no backflip how about a spin instead", "never mind", "no stop", "dont do a backflip", "not now",
          "open the laptop after you sit in the chair", "sit down before you open the laptop",
          "after you sit in the chair open the laptop", "jump once you sit down", "spin once", "jump after me",
          "now the laptop open it", "laptop pick up", "the laptop is open", "the chair pull it closer",
          "drop the ball on the floor dont put it on the desk", "pick up the ball not the red one",
          "spin around but dont fall", "put the ball down but not on the desk just drop it on the floor",
          "show me you spinning twice and flipping backwards", "do a flip", "do a front flip", "flip forward",
          "do a backwards somersault", "do a forward flip", "put the headphones on your head", "put it on your head",
          "show me one jump", "drop whatever youre holding", "drop whatever you are holding", "can you like spin",
          "dance like a robot", "walk like two steps", "um like do a backflip", "twirl like a ballerina",
          "slide the laptop closer to you on the desk", "i like dancing", "like a backflip", "if you could do a backflip",
          "hey muse wait dont open the laptop just sit down in the chair", "dont look at me", "turn away and dont look at me"]
texts = sorted(set(texts))
cases = [{"text": t, "cmds": [h["cmd"] for h in intent.Matcher(scene=scene).parse(t)]} for t in texts]
# streaming: word-by-word partials 120 ms apart, a stable repeat, then the final
streams = []
for t in texts[::7]:
    st = intent.Matcher(scene=scene).stream()
    words, steps, outs, ms = t.split(), [], [], 0
    for k in range(1, len(words) + 1):
        steps.append([" ".join(words[:k]), False, ms]); ms += 120
    steps.append([t, False, ms + 300]); steps.append([t, True, ms + 700])
    for s in steps:
        outs.append([h["cmd"] for h in st.update(s[0], s[1], s[2])])
    streams.append({"text": t, "steps": steps, "outs": outs})
json.dump({"scene": "tests/props_mock/*.json", "cases": cases, "streams": streams},
          open(os.path.join(HERE, "intent_act_vectors.json"), "w"), indent=0, sort_keys=True)
print(len(cases), "cases,", len(streams), "streams")
