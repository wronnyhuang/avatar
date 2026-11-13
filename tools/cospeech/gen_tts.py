#!/usr/bin/env python3
"""speech: ground-truth utterances for the co-speech timing test.

ElevenLabs /v1/text-to-speech/{voice}/with-timestamps -> <out>/<id>.wav (24 kHz mono int16) + <id>.json
{"id","text","chars":[...],"start":[...],"end":[...],"dur"}. Prints the estimated TTS cost (list price).

  ELEVENLABS_API_KEY=... python3 -I tools/cospeech/gen_tts.py [out_dir]     (default <repo>/runs/cospeech/tts)

Env: TTS_SET=heldout|heldout2 (held-out sentence sets), TTS_VOICE=<stock voice name prefix> (default: a premade
ElevenLabs voice per set). Only ElevenLabs PREMADE (stock) voices are used.
"""
import base64, json, os, subprocess, sys, urllib.request, urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'runs', 'cospeech', 'tts')
USD_PER_CHAR = 0.0003          # ~$0.30 / 1k chars (list price, err high; estimate only)
UTTS = [
    ("big", "Oh wow, that's a really big idea! I love it."),
    ("unsure", "Hmm, I'm not sure. Maybe tomorrow? I don't know yet."),
    ("three", "There are three things you need: a hat, a scarf, and some warm socks."),
    ("ordinals", "First, we mix the flour. Second, we add the eggs. And third, we bake it for twenty minutes."),
    ("never", "No, no, I never said that! I would never eat your sandwich."),
    ("there", "Look over there, by the window. Do you see that tiny bird?"),
    ("proud", "You did a great job today, and I'm so proud of you."),
    ("story", "Well, so, the thing is... it's a long story. It goes way back, all the way to last summer."),
    ("tower", "The tower was so tall, it went way up into the clouds, and then it came crashing down!"),
    ("ball", "It's a little round ball, about this small, and it rolls all around the room."),
    ("sunset", "Let me explain. When the sun goes down, the sky turns orange, then pink, then purple."),
    ("plan", "Okay, okay, here's the plan: you take the left side, I'll take the right, and we meet in the middle."),
    ("sorry", "I'm sorry you had a rough day. It happens to everyone sometimes, and tomorrow is a brand new day."),
]


# held-out set (TTS_SET=heldout): new sentences, another voice — never used for tuning
HELDOUT = [
    ("h_pizza", "Honestly? I think pizza is the best food in the whole world, but tacos are a very close second."),
    ("h_count", "We need two eggs, one cup of milk, and maybe a little bit of sugar. That's it!"),
    ("h_no", "No way, I would never forget your birthday. Not in a million years."),
    ("h_where", "The park is just over there, on the left, past that big red building."),
    ("h_explain", "So here's how it works: you press this button, wait a second, and then the door slides open."),
    ("h_feel", "I'm not really sure how I feel about it... it's kind of exciting, and kind of scary too."),
]
if os.environ.get('TTS_SET') == 'heldout':
    UTTS = HELDOUT
# c2 held-out #2 (TTS_SET=heldout2): a third voice and new sentences, never looked at while tuning (speech-c2)
HELDOUT2 = [
    ("x_garden", "My garden has four big sunflowers, and one tiny tomato plant that never grows."),
    ("x_trip", "First we drove to the lake, then we hiked up the hill, and finally we had a picnic."),
    ("x_maybe", "I guess we could try it, but honestly, I'm not sure it will work."),
    ("x_left", "Put the red box on the left, and the blue one over there, next to the door."),
    ("x_wow", "Wow, you finished the whole puzzle? That's amazing! It took me three days."),
    ("x_long", "It was a really long day. We walked so far, all the way around the city, and back again."),
    ("x_nope", "Nope, I didn't eat the cookies. Well... maybe just one."),
    ("x_sleep", "When it gets dark, the owls wake up, and the little mice hide under the leaves."),
]
if os.environ.get('TTS_SET') == 'heldout2':
    UTTS = HELDOUT2
SET_NAMES = {'heldout': ('brian', 'george', 'charlie', 'daniel', 'roger'), 'heldout2': ('laura', 'alice', 'matilda', 'lily')}


def req(url, body=None, method=None):
    key = os.environ['ELEVENLABS_API_KEY']
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method or ('POST' if data else 'GET'),
                               headers={'xi-api-key': key, 'content-type': 'application/json', 'accept': 'application/json'})
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.loads(f.read())


def main():
    os.makedirs(OUT, exist_ok=True)
    voices = req('https://api.elevenlabs.io/v1/voices').get('voices', [])
    pref = [v for v in voices if v.get('category') == 'premade']
    if not pref:
        sys.exit('no premade (stock) ElevenLabs voices visible to this key')
    names = SET_NAMES.get(os.environ.get('TTS_SET'), ('jessica', 'laura', 'sarah', 'rachel'))
    if os.environ.get('TTS_VOICE'):
        names = (os.environ['TTS_VOICE'].lower(),)
    voice = next((v for v in pref if v.get('name', '').lower().startswith(names)), pref[0])
    print('voice', voice.get('name'), voice.get('voice_id'))
    chars = 0
    for uid, text in UTTS:
        jp = os.path.join(OUT, uid + '.json')
        if os.path.exists(jp):
            continue
        j = req(f"https://api.elevenlabs.io/v1/text-to-speech/{voice['voice_id']}/with-timestamps?output_format=mp3_44100_128",
                {'text': text, 'model_id': 'eleven_turbo_v2_5', 'voice_settings': {'stability': 0.45, 'similarity_boost': 0.75}})
        chars += len(text)
        mp3 = os.path.join(OUT, uid + '.mp3')
        open(mp3, 'wb').write(base64.b64decode(j['audio_base64']))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-ac', '1', '-ar', '24000', '-sample_fmt', 's16',
                        os.path.join(OUT, uid + '.wav')], check=True)
        al = j.get('alignment') or j.get('normalized_alignment')
        json.dump({'id': uid, 'text': text, 'voice': voice.get('name'), 'chars': al['characters'],
                   'start': al['character_start_times_seconds'], 'end': al['character_end_times_seconds']},
                  open(jp, 'w'))
        print(uid, len(text), 'chars', round(al['character_end_times_seconds'][-1], 2), 's')
    if chars:
        print(f'{chars} chars, ~${chars * USD_PER_CHAR:.4f} est.')


if __name__ == '__main__':
    main()
