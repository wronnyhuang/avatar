"""intent c2: a FRESH muse phrasing set written by gpt-4.1-mini, blind to the act grammar (the model sees only
plain-English intents + the scene, never a pattern). Labels are attached from the intent spec, not from the matcher.

    OPENAI_API_KEY=... python3 tools/intent_bench/gen_fresh_c2.py --out tests/intent_c2_raw.json

Label syntax per step: 'head1|head2 key=v1|v2 ...' ; steps joined with ' ; '. Keys: count dir obj target place side.
obj/target/place values are scene ids (headphones chair desk laptop mug book ball plant / me closer back away) or 'it'
(any pronoun / the same object named again). An empty label = nothing may fire.
"""
import argparse, json, os, sys, urllib.request

SCENE = ("pink headphones (on a little side table), a teal chair, an oak desk, a silver laptop (on the desk, closed), "
         "a red mug, a blue book, a yellow ball, a little green plant")
SPECS = {
 "spin": [("spin around once", "move/spin|move/twirl"),
          ("spin around three times", "move/spin|move/twirl count=3"),
          ("do two spins", "move/spin|move/twirl count=2"),
          ("twirl (like a ballerina)", "move/twirl|move/spin"),
          ("spin to its left", "move/spin|move/twirl dir=left")],
 "backflip": [("do a backflip", "move/backflip"),
              ("do two backflips in a row", "move/backflip count=2"),
              ("flip over backwards (a back flip) - say 'back' or 'backwards' explicitly", "move/backflip"),
              ("do a flip (just 'a flip', no direction word)", "move/backflip|move/flip")],
 "jump": [("jump", "move/jump|move/hop"),
          ("jump three times", "move/jump|move/hop count=3"),
          ("jump really high", "move/jump"),
          ("hop up and down", "move/hop|move/jump")],
 "dance": [("dance", "move/dance"),
           ("do a silly dance", "move/dance"),
           ("dance for a bit / show some dance moves", "move/dance")],
 "walk": [("walk forward a few steps", "move/walk|move/step"),
          ("walk over to the desk", "move/walk"),
          ("take two steps back", "move/walk|move/step")],
 "turn": [("turn away (so its back faces the user)", "move/turn_away"),
          ("turn back around to face the user again", "move/turn_back"),
          ("stop what it is doing at the laptop and turn back to talk to the user", "move/turn_back|object/stop")],
 "pickup": [("pick up the yellow ball", "object/pick_up obj=ball"),
            ("pick up the headphones", "object/pick_up obj=headphones"),
            ("pick up the book", "object/pick_up obj=book"),
            ("pick up the mug", "object/pick_up obj=mug"),
            ("pick up the laptop", "object/pick_up obj=laptop"),
            ("grab the plant with its left hand", "object/pick_up obj=plant side=left")],
 "manipulate": [("hold the ball up and show it to the user", "object/show|object/hold_up obj=ball|it"),
                ("turn the book around (rotate it in its paws)", "object/turn|object/flip obj=book|it"),
                ("look closely at / examine the mug", "object/inspect obj=mug|it"),
                ("pass the ball to its other hand", "object/hand_over obj=ball|it"),
                ("toss the ball up in the air and catch it", "object/toss obj=ball|it"),
                ("hand the book over to the user", "object/offer obj=book|it"),
                ("flip the book upside down", "object/flip|object/turn obj=book|it")],
 "putdown": [("put it down (whatever it is holding; use a pronoun)", "object/put_down|object/drop|object/place obj=it"),
             ("put the ball back where it was", "object/put_back obj=ball|it"),
             ("put the mug on the desk", "object/place obj=mug|it place=desk"),
             ("set the book down on the floor", "object/place|object/put_down obj=book|it place=floor"),
             ("drop the ball", "object/drop|object/put_down obj=ball|it")],
 "headphones": [("put the headphones on its head", "object/put_on obj=headphones|it"),
                ("take the headphones off", "object/take_off obj=headphones|it")],
 "furniture": [("pull the chair over (toward itself / the user)", "object/pull|object/bring obj=chair"),
               ("pull the chair up to the desk", "object/pull|object/push|object/bring|object/slide obj=chair target=desk"),
               ("pull the desk closer", "object/pull|object/bring|object/slide obj=desk"),
               ("push the chair back (away)", "object/push obj=chair"),
               ("push the desk away", "object/push obj=desk")],
 "sitstand": [("sit down in the chair", "object/sit_in obj=chair"),
              ("sit down at the desk", "object/sit_at|object/sit_in obj=desk|chair"),
              ("sit down (no object named)", "move/sit|move/sit_down"),
              ("stand up (no object named)", "move/stand|move/stand_up|move/get_up|object/stand_up"),
              ("get up out of the chair", "object/stand_up|move/stand|move/stand_up|move/get_up")],
 "laptop": [("open the laptop", "object/open obj=laptop"),
            ("close the laptop", "object/close obj=laptop"),
            ("slide the laptop closer to itself on the desk", "object/slide|object/pull|object/bring obj=laptop"),
            ("type on the laptop / start typing", "object/type_on|object/work_on|object/use"),
            ("get to work on the laptop", "object/work_on|object/type_on|object/use")],
 "multistep": [("pick up the headphones and then put them on", "object/pick_up obj=headphones ; object/put_on obj=headphones|it"),
               ("pull the chair over and then sit in it", "object/pull|object/bring obj=chair ; object/sit_in|move/sit obj=chair|it"),
               ("put the headphones on, sit at the desk, open the laptop and start typing",
                "object/put_on obj=headphones ; object/sit_at|object/sit_in|move/sit ; object/open obj=laptop|it ; object/type_on|object/work_on|object/use"),
               ("spin around twice and then do a backflip", "move/spin|move/twirl count=2 ; move/backflip"),
               ("close the laptop, stand up and push the chair back",
                "object/close obj=laptop ; move/stand|move/stand_up|move/get_up|object/stand_up ; object/push obj=chair|it"),
               ("take the headphones off and put them down", "object/take_off obj=headphones ; object/put_down|object/place|object/drop|object/put_back obj=headphones|it"),
               ("pick up the ball, toss it up, then put it back", "object/pick_up obj=ball ; object/toss obj=ball|it ; object/put_back obj=ball|it"),
               ("jump, then dance", "move/jump|move/hop ; move/dance"),
               ("pull the desk over, then pull the chair up to it", "object/pull|object/bring obj=desk ; object/pull|object/push|object/bring|object/slide obj=chair")],
 "reordered": [("FIRST sit in the chair, THEN open the laptop - but say it with 'after' or 'before' so the spoken order is reversed, e.g. 'open X after you Y'",
                "object/sit_in|move/sit|object/sit_at obj=chair|desk|it ; object/open obj=laptop|it"),
               ("first put on the headphones, then turn back to face the user - as one casual sentence with 'first ... then'",
                "object/put_on obj=headphones ; move/turn_back"),
               ("partial sequence spoken in pieces: just 'okay now the laptop - open it'", "object/open obj=laptop|it"),
               ("pick up the book; then (as the next sentence, using only a pronoun) show it to the user",
                "object/pick_up obj=book ; object/show|object/hold_up|object/offer obj=book|it")],
 "neg_mixed": [("a sentence that says NOT to sit, but to pull the chair over (e.g. 'don't sit, just pull the chair over')",
                "object/pull|object/bring obj=chair"),
               ("a sentence that says not to do a backflip but to do a spin instead", "move/spin|move/twirl"),
               ("a sentence that says to put the ball down but NOT on the desk... just drop it on the floor", "object/place|object/put_down|object/drop obj=ball|it"),
               ("a sentence that says don't open the laptop yet, just sit down in the chair", "object/sit_in|move/sit obj=chair")],
}
NEG = {
 "neg_negation": "tells the character NOT to do something (don't do a backflip, no jumping, never take your headphones off, stop, don't touch the laptop) - pure negation, nothing else requested",
 "neg_hypo": "hypothetical or wish, not a request now (if you could do a backflip..., I wish I could spin like that, imagine you sat in that chair, maybe later you could dance)",
 "neg_question": "a question about the character's state or history, NOT asking it to act (are you wearing your headphones?, did you jump?, is the laptop open?, how many times did you spin?, why are you sitting?)",
 "neg_reported": "reported speech or narration about someone else (my sister said do a backflip, the guy on tv jumped really high, my dad always pulls his chair up to the desk, she told me to pick up the ball)",
 "neg_statement": "a statement or comment mentioning the actions/objects but not requesting anything (I love dancing, that backflip was awesome, the headphones are pink, I'm sitting at my desk, my laptop is so slow, I picked up my toys today)",
}
STYLE = ("Vary heavily: casual, polite, bossy, excited kid (5-9 years old), indirect ('I bet you can't...', 'let's see "
         "you...', 'show me...', 'how about...'), rambling with fillers (um, uh, like, okay so), questions used as requests "
         "('can you...', 'could you maybe...'), gerunds, synonyms and different verbs, different word orders, calling it "
         "'muse' or a nickname. About a third must look like raw speech-recognizer output: all lowercase, no punctuation, "
         "maybe one misheard word (homophone) or a run-on with no 'and'. Do not reuse the same opening twice.")


def call(key, prompt, n_tok=4000):
    body = {"model": "gpt-4.1-mini", "temperature": 1.0, "max_tokens": n_tok, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "You write realistic test utterances for a voice-controlled animated character. Output JSON only."},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    u = r.get("usage", {})
    usd = u.get("prompt_tokens", 0) * 0.4e-6 + u.get("completion_tokens", 0) * 1.6e-6
    return json.loads(r["choices"][0]["message"]["content"]), usd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    key = os.environ["OPENAI_API_KEY"]
    out, total = [], 0.0
    intro = ("A cute cream-coloured plush character called Muse lives on a screen. It has legs and paws, can move its whole "
             f"body, and handles the props in its little room: {SCENE}. People talk to it out loud.\n")
    for cat, specs in SPECS.items():
        lst = "\n".join(f"{i}. {d}" for i, (d, _) in enumerate(specs))
        p = (intro + f"For EACH numbered intent below write {a.n} different things a person might say to Muse to ask for "
             "EXACTLY that (no extra actions; keep every number, direction and object; multi-step intents keep all steps).\n"
             + STYLE + "\nIntents:\n" + lst +
             '\nReturn {"items": [{"i": <number>, "says": ["...", ...]}, ...]}')
        d, usd = call(key, p)
        total += usd
        for it in d.get("items", []):
            i = int(it["i"])
            if 0 <= i < len(specs):
                for s in it.get("says", [])[:a.n]:
                    out.append({"cat": cat, "intent": specs[i][0], "text": s, "label": specs[i][1]})
        print(cat, len(out), f"${total:.4f}", file=sys.stderr)
    for cat, desc in NEG.items():
        p = (intro + f"Write {a.n * 3} different things a person might say near Muse that mention its moves or props but are "
             f"NOT a request for it to do anything right now. Type: {desc}.\n" + STYLE +
             '\nReturn {"says": ["...", ...]}')
        d, usd = call(key, p)
        total += usd
        for s in d.get("says", [])[:a.n * 3]:
            out.append({"cat": cat, "intent": desc[:60], "text": s, "label": ""})
        print(cat, len(out), f"${total:.4f}", file=sys.stderr)
    json.dump({"model": "gpt-4.1-mini", "usd": round(total, 4), "items": out}, open(a.out, "w"), indent=1)
    print(len(out), "items", f"${total:.4f}")


if __name__ == "__main__":
    main()
