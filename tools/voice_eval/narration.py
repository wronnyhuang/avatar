#!/usr/bin/env python3
"""Narration / context-leak check for the talking avatar's speech.

Flags avatar turns that NARRATE the avatar's own body, pose or actions ("I'm raising my paw",
"*waves*", "my hands are on my head", "Okay, looking up!") or LEAK context / perception tags
out loud ("[camera]", "silent context", "face_appear", "event: wave"). Plain conversation and
bare acknowledgements ("Sure!", "There you go!", "Sure, look!", "Watch this!") are not flagged.

Two informational side-flags are reported but never counted as narration:
  perception - unsolicited commentary on what the camera shows ("I see your colorful shirt")
  offer      - offering a move ("If you want, I can wave", "Should I do a little wiggle?")
and `sound_tags` lists non-verbal transcript tags ([chuckle], [snort], *sigh*) - those are audio
the model produced, not words read aloud.

  python3 -I narration.py <files or dirs...> [--llm | --llm-all] [--json out.json] [--show N]
  python3 -I narration.py --selftest

Inputs: hub session JSONs (session-*.json, "transcript": [[who, text], ...]), the eval runner's
report.json (any "transcript" list found inside), or .txt (one avatar line per line).
Regex part is stdlib only. --llm asks gpt-4.1-mini (fallback gpt-4o-mini) about the turns the
regex did not flag (--llm-all: every turn), 20 turns per request, estimated cost capped per run at
$NARRATION_LLM_CAP_USD (default 0.20). Needs OPENAI_API_KEY.
"""
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

COST_CAP_USD = float(os.environ.get('NARRATION_LLM_CAP_USD') or 0.20)   # per-run cap on the LLM pass
PRICES = {'gpt-4.1-mini': (0.40, 1.60), 'gpt-4o-mini': (0.15, 0.60)}  # $/M in, $/M out

# ----------------------------------------------------------------------------- lexicon
A = r"['’]"                                  # straight or curly apostrophe
BODY = (r"(?:paws?|hands?|arms?|ears?|tail|eyes?|eyelids?|head|nose|tongue|eye-?brows?|brows?|"
        r"mouth|face|body|whiskers?|legs?|feet|foot|fingers?|thumbs?|wings?|snout|muzzle|chin|"
        r"shoulders?|belly|tummy|toes?|lips|cheeks?|gaze|horns?|trunk|fins?|tentacles?|antennae?)")
AVATAR_ONLY = r"(?:paws?|tail|whiskers?|snout|muzzle|wings?|fins?|tentacles?|antennae?|horns?|trunk)"
ADJ = (r"(?:(?:left|right|other|both|two|one|little|big|fluffy|furry|front|back|tiny|whole|"
       r"cheerful|friendly|quick|silly|happy|cute|fox|bear|blob|bloop|another|good|great|super)[- ]+)*")


def forms(*bases):
    """'wave' -> wave|waves|waved|waving (simple English inflection, irregulars passed as a|b|c)."""
    out = []
    for b in bases:
        if '|' in b:
            out += b.split('|')
            continue
        out.append(b)
        if b.endswith('e'):
            out += [b + 's', b + 'd', b[:-1] + 'ing']
        elif re.search(r'[^aeiou][aeiou][bdgmnpt]$', b) and b not in ('point', 'open', 'swish'):
            out += [b + 's', b + b[-1] + 'ed', b + b[-1] + 'ing']
        elif b.endswith(('sh', 'ch', 'x', 's')):
            out += [b + 'es', b + 'ed', b + 'ing']
        else:
            out += [b + 's', b + 'ed', b + 'ing']
    return '(?:' + '|'.join(sorted(set(out), key=len, reverse=True)) + ')'


# verbs that are narration with no object ("I waved", "I'm nodding", "I'll spin")
INTRANS = forms('wave', 'wink', 'nod', 'clap', 'bow', 'shrug', 'blink', 'spin|spins|spun|spinning',
                'twirl', 'dance', 'hop', 'bounce', 'wiggle', 'waggle', 'wag', 'jiggle', 'wobble',
                'float', 'smile', 'grin', 'frown', 'salute', 'curtsy|curtsies|curtsied|curtsying',
                'jump', 'sway', 'giggle-skip-none')
INTRANS = INTRANS.replace('|giggle-skip-none', '')
# verbs that need a body part / direction / 'mine' after them
TRANS = forms('raise', 'lift', 'hold|holds|held|holding', 'put|puts|putting', 'turn', 'tilt', 'close',
              'open', 'dip', 'tip', 'stick|sticks|stuck|sticking', 'point', 'shake|shakes|shook|shaking',
              'cover', 'cross', 'fold', 'flap', 'flick', 'twitch', 'perk', 'scratch', 'rub', 'tap',
              'touch', 'look', 'glance', 'move', 'stretch', 'swish', 'swing|swings|swung|swinging',
              'kick', 'stomp', 'lower', 'drop', 'show|shows|showed|showing|shown', 'bob', 'flip',
              'peek', 'stick')
ACTION_ANY = '(?:' + INTRANS[3:-1] + '|' + TRANS[3:-1] + ')'
ACTION_NOUN = (r"(?:waves?|winks?|nods?|bows?|spins?|twirls?|wiggles?|wiggle-wiggle|high[- ]?fives?|"
               r"thumbs?[- ]?ups?|ear[- ]wiggles?|paw[- ]waves?|tail[- ]wags?|head[- ]tilts?|"
               r"head[- ]nods?|head[- ]bobs?|eye[- ]winks?|glance (?:up|down|left|right)|salutes?|"
               r"curtsy|dance|jig|shimmy|hop|boop|nose[- ]boop|peace sign)")
OBJ = (rf"(?:my\b|mine\b|them\b(?=\s*[.!,—–-]|\s*$)|{BODY}\b|up\b|down\b|upward|downward|left\b|right\b(?! (?:now|there|away|here))|"
       r"around|over\b(?! (?:the|my|it|there|here|to))|away|aside|forward|ahead|straight|sideways|"
       r"toward|towards|back\b(?! (?:to|then|later|home|soon))|out\b(?! (?:of|for|there|loud))|"
       r"(?:a|one) (?:bit|little|tad|touch)\b|it up\b|at you\b|at the (?:ceiling|floor|sky|camera|screen))")
SOUND = (r"(?:chuckles?|chuckling|laughs?|laughing|laughter|giggles?|giggling|sniffs?|sniffing|snorts?|"
         r"breath|breathes|breathing|inhales?|exhales?|sighs?|sighing|gasps?|coughs?|hums?|humming|"
         r"mouth noise|tongue click|clicks?|lip smack|smack|yawns?|snickers?|hiccups?|whistles?|"
         r"clears? throat|throat clear|mm+|hmm+|music|silence|noise|inaudible|crosstalk|unintelligible)")
LEAK_WORDS = (r"(?:camera|context|perception|event|system|developer|vision|frame|tool|function|silent|"
              r"note|image|video|user|object|gesture|face|scene|sensor|observation|metadata|tag|cue)")

TRANS_X = (r"(?!(?:look|looks|looked|looking)\s+(?:it\b|forward|for\b|into|out for|over (?:the|my|it|this|that)|"
           r"up (?:the|a|an|some|that|it|what|how|info\w*|recipes?|words?|facts?)\b)|point\w*\s+out\b|"
           r"(?:hold|holds|holding|held)\s+(?:on\b|up a sec|that thought|tight)|(?:turn|turned|turning|turns)\s+(?:out|it|this|that)\b|"
           r"(?:open|opens|opened|opening)\s+up\b(?!\s+my)|(?:show|shows|showed|showing)\s+(?:up|you around|around)\b)")
SUBJ = rf"\b(?:I(?:{A}m| am| was)?|I{A}ll|I will|I{A}ve|I have|let me|lemme|watch me|see me|look at me)"
MOD = (r"(?:\s+(?:just|now|still|also|totally|really|gonna|going to|about to|staying|keep|keeping|"
       r"already|happily|gently|slowly|quickly|right|even|so|gladly|proudly|excitedly|then|first|"
       r"currently|busy))*")
ACK = (rf"(?:(?:okay|ok|sure(?: thing)?|yep|yes|yeah|alright|all right|got it|you got it|and|now|"
       rf"there|there we go|there you go|here we go|ta-?da|tada|voil[aà]|look|see|and yep|still|"
       rf"here{A}s|here is|here comes|and a|and an|a|an|one|another|okay then|right)\b[\s,!.:;—–-]*)*")
OFFER = (rf"\b(?:if you(?:{A}d)? (?:want|like|need|wanna)|want me to|wanna see|do you want|would you like|"
         rf"should I|shall I|can I|could I|I can|I could|I{A}d|I would|happy to|ready to|just say|"
         rf"say the word|tell me|ask me|just ask|whenever you want|you can ask|did you want|want a|"
         rf"want an|need a|need an|maybe a|or a|feel like a|how about a|ready for)\b")
NEG = rf"\b(?:not|never|no|can{A}t|cannot|don{A}t|didn{A}t|won{A}t|isn{A}t|aren{A}t|wasn{A}t|without)\b"
VISUAL_ASK = (r"\b(?:see|seeing|look|looking|holding|wearing|fingers?|colou?rs?|show|showing|camera|"
              r"what is this|what's this|what am i|am i|do i|how many|notice|watch)\b")

I = re.I
RX_NARR = [
    # first person + action verb ("I'm raising my paw", "I waved back", "I'll spin", "watch me wiggle my ears")
    ('first-person action', re.compile(
        rf"{SUBJ}{MOD}\s+(?:(?P<iv>{INTRANS})\b(?!\s+(?:in\b(?!\s+(?:place|the air|circles?|a circle))|right in|into|to conclusions|around the|out\b|ideas|off\b|at the chance))"
        rf"|{TRANS_X}(?P<tv>{TRANS})\s+(?:(?:up|out|right|just|both|my|the|a|one|your)\s+)?{OBJ})", I)),
    # gerund / bare-verb clause right after an acknowledgement ("Okay, nodding now!", "Waving right back at you!")
    ('subjectless action', re.compile(
        rf"^{ACK}(?=\w+ing\b)(?:(?P<iv>{INTRANS})\b(?!\s+(?:in\b(?!\s+(?:place|the air|circles?))|into|around the))|{TRANS_X}(?P<tv>{TRANS})\s+(?:(?:up|out|right|just|both|my|the|a|one)\s+)?{OBJ})"
        r"(?!\s*(?:good|great|nice|fine|well|like|forward|for|into|at it|it up))", I)),
    # my <body part> + pose/state ("my hands are on my head", "My hand's up", "my ears perk right up")
    ('my-body pose', re.compile(
        rf"\bmy\s+{ADJ}{BODY}(?!\s+(?:is|are)\s+(?:spinning|swimming|racing|full))(?:{A}s|{A}re|\s+(?:is|are|was|were))?(?:\s+(?:now|all|still|just|right|both|so|already|totally|really))*\s+"
        r"(?:up\b|down\b|raised|lifted|out\b(?! of)|closed|shut|opened|perk\w*|wiggl\w+|wav(?:es|ed|ing)|wagg?\w*|tilt\w*|"
        r"cross\w*|dipped|twitch\w*|flapp?\w*|danc\w+|spinn\w*|bounc\w+|on my|in the air|by my|behind my|"
        r"high in|swish\w*|point(?:s|ed|ing)|(?:do|did|give|gave|does) (?:a|an|my|the|its)\b)", I)),
    # subjectless avatar-only body part + pose ("Paw's up!", "Ta-da, paw raised!", "ears wiggled")
    ('body pose', re.compile(
        rf"(?<!your )(?<!their )(?<!his )(?<!her )\b(?:{ADJ}{AVATAR_ONLY}(?:{A}s|\s+(?:is|are))?\s+(?:now\s+|all\s+)?"
        r"(?:up\b|down\b|raised|lifted|wiggl\w+|wagg\w+|perked|tilted|out\b(?! of)|in the air|high)"
        r"|ears?(?:\s+(?:are|is|just))?\s+(?:wiggl\w+|perk\w*|twitch\w*|flapp?\w*|danc\w+)"
        rf"|\bmine(?:{A}s|{A}re|\s+(?:is|are))\s+(?:now\s+|all\s+)?(?:up|down|raised|lifted|out|waving|closed)\b)", I)),
    # acknowledgement + bare body-part pose ("Okay, eyes closed!", "Sure, head tilted!")
    ('ack body pose', re.compile(
        rf"^(?:(?:okay|ok|sure(?: thing)?|yep|yes|alright|all right|got it|there|ta-?da|tada|voil[aà]|and)\b[\s,!.—–-]*)+"
        rf"{ADJ}{BODY}(?:{A}s|\s+(?:are|is))?\s+(?:now\s+)?(?:closed|shut|open|up|down|raised|lifted|dipped|tilted|"
        r"crossed|wiggled|perked|out)(?:\s+(?:again|now|too|for you))*\s*[.!…]*\s*$", I)),
    # watch / look at / here's my <body part> ("Here's my paw up!", "Look at my tail go!")
    ('show-my-body', re.compile(
        rf"\b(?:here{A}?s|here is|here are|here goes|there goes|there go|look at|see|watch|check out)\s+my\s+{ADJ}{BODY}\b", I)),
    # "<thing> coming up" ("right paw coming up!", "left-eye wink coming up")
    ('coming up', re.compile(
        rf"\b{ADJ}(?:{BODY}|{ACTION_NOUN}|(?:left|right)[- ]eye[- ]wink)\s+(?:coming|comin{A}?)\s+(?:right\s+)?up\b", I)),
    # "give you a wave", "giving one right back", "do a little wiggle"
    ('gesture given', re.compile(
        rf"\b(?:I(?:{A}m| am)?\s+|I{A}ll\s+|let me\s+|)(?:just\s+|now\s+)?(?:giving|give|gave|do(?:ing)?|did|throwing|throw|"
        rf"sending|send|sent|blowing|blow|flashing|flash)\s+(?:you\s+|y{A}all\s+)?(?:a|an|one|my|another|the)\s+"
        rf"(?:\w+[\s-]+){{0,2}}{ACTION_NOUN}\b|\b(?:giving|give|gave)\b[^.!?]{0,25}?\b(?:it|one|that) (?:right )?back\b", I)),
    # inverted motion ("Down I go!", "Up I go")
    ('inverted motion', re.compile(r"\b(?:up|down|around|over|off) I go\b", I)),
    # avatar smiling / waving back ("a jelly blob smiling back", "waving right back")
    ('back-gesture', re.compile(
        rf"(?<!you{A}re )(?<!you are )(?<!they{A}re )(?<!you )\b(?:smiling|waving|winking|nodding|grinning)\s+(?:right\s+)?back\b", I)),
]
# whole clause is just an action exclamation ("Sure, big friendly wave!", "Ear wiggle again!", "Wiggle, wiggle!")
RX_EXCLAIM = re.compile(
    rf"^(?P<ack>{ACK})(?P<adj>(?:(?:big|little|tiny|quick|friendly|cheerful|silly|happy|cute|fox|bear|blob|super|"
    rf"one|a|another|double|left|right)[- ,]+)*)(?P<noun>{ACTION_NOUN})(?:[\s,!-]+{ACTION_NOUN})*"
    rf"(?P<extra>(?:\s+(?:again|for you|just for you|right back(?: at you)?|back at you|back|to you(?: too)?|"
    rf"to all(?: \w+){{0,3}}|too|now|in the air|time|coming up|right up|incoming|here))*)\s*$", I)
STAGE_RX = [
    ('asterisk', re.compile(r"\*{1,2}([^*\n]{1,40})\*{1,2}")),
    ('paren', re.compile(r"\(([^()\n]{1,40})\)")),
    ('bracket', re.compile(r"\[([^\]\n]{1,60})\]")),
]
RX_LEAK = [
    ('snake_case id', re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", I)),
    ('camelCase id', re.compile(r"\b(?!(?:iPhone|iPad|iPod|iOS|iMac|eBay|macOS|iCloud|eMail|iTunes|tvOS|watchOS)\b)[a-z]+[A-Z][A-Za-z]+\b")),
    ('context tag', re.compile(r"\bsilent context\b|\bcontext\s*:|\b(?:camera|vision|scene|system|user) context\b|"
                               r"\b(?:camera|vision|perception) (?:note|tag|event|update|cue|says|reports?)\b", I)),
    ('camera noticed', re.compile(r"\b(?:the |my )?camera (?:noticed|notices|detected|detects|registered|picked up|flagged|caught|saw|sees|spotted)\b", I)),
    ('perception', re.compile(r"\bperception\b", I)),
    ('event tag', re.compile(r"\bevent\s*[:=]|\b(?:face|object) (?:appear(?:ed)?|up|lost|gone)\s+event\b|\bface appear\b|\bobject up\b|"
                             r"\b(?:look back|thumbs up|wave|set emotion) event\b|\bset emotion\b", I)),
    ('prompt leak', re.compile(r"\b(?:system|developer) (?:message|prompt|note|instruction)s?\b|\b(?:tool|function) call\b|"
                               r"</?[a-z_]{2,20}>|\{\s*\"[a-z_]+\"\s*:", I)),
]
RX_PERCEPTION = re.compile(
    rf"\bI (?:can )?(?:see|spot|notice|noticed|caught|saw|spy)\s+(?:you\b|your\b|someone|somebody|three|two|four|five|"
    rf"a\b|an\b|the\b|that\b|those\b|both\b|some\b|people|folks|kids|him\b|her\b|them\b)(?! (?:mean|point|what|problem|issue))"
    rf"|\blooks like you\b|\byou look\b|\byou{A}re (?:holding|wearing|smiling|waving|looking|making|raising|pointing|showing|sitting|standing|touching)"
    rf"|\b(?:they|he|she){A}?(?:re|s| are| is) (?:holding|wearing|smiling|waving|touching|pointing)"
    rf"|\byour (?:shirt|glasses|hands?|smile|hair|earbuds|hat|fingers?|thumbs?)\b|\b(?:that|nice|big|lovely) smile\b"
    rf"|\bthumbs?[- ]?ups?\b|\bfingers? up\b|\bpeace signs?\b|\bhands? (?:are )?up\b|\bI like your\b", I)
RX_CLAUSE = re.compile(r"[^.!?…—–;:\n]+(?:[.!?…]+|[—–;:\n]|\s+-\s+|$)")
RX_SPLIT = re.compile(r"(?<=[.!?…])(?=\S)")    # "Hi there!Nice" -> "Hi there! Nice"


def clauses(text):
    """Split into clauses; each item (clause_text, start_offset, terminator)."""
    out = []
    for m in RX_CLAUSE.finditer(text):
        s = m.group(0)
        if not s.strip():
            continue
        lead = len(s) - len(s.lstrip())
        term = s.rstrip()[-1:] if s.rstrip() else ''
        out.append((s.strip(), m.start() + lead, term))
    return out


def _clip(s, n=70):
    s = ' '.join(s.split())
    return s if len(s) <= n else s[:n - 1] + '…'


def _stage(kind, content):
    """Classify a bracketed / starred / parenthesised chunk: 'leak' | 'narration' | 'sound' | None."""
    c = content.strip()
    words = c.split()
    if not words:
        return None
    if kind == 'bracket':
        if re.search(rf"\b{LEAK_WORDS}\b", c, I) or '_' in c or ':' in c:
            return 'leak'
        if re.fullmatch(rf"(?:soft |quiet |little |small |short |light |gentle )*{SOUND}(?: \w+)?", c, I):
            return 'sound'
        if re.search(rf"\b{ACTION_ANY}\b|\b{BODY}\b", c, I):
            return 'narration'
        return 'leak'                                        # any other bracketed tag read aloud
    if re.fullmatch(rf"(?:soft |quiet |little |small |short |light |gentle )*{SOUND}(?: \w+)?", c, I):
        return 'sound'
    if kind == 'paren' and len(words) > 5:
        return None
    if re.search(rf"\b{ACTION_ANY}\b|\b{BODY}\b", c, I):
        return 'narration'
    if kind == 'asterisk' and len(words) <= 3 and not c.isupper():
        return 'narration'                                   # emote-style *boop*
    if kind == 'paren' and re.search(rf"\b{LEAK_WORDS}\b", c, I) and ('_' in c or ':' in c):
        return 'leak'
    return None


def classify(text, prev_user=None):
    """-> {"narration", "leak", "hits", "why", "perception", "offer", "sound_tags"}"""
    text = text or ''
    t = RX_SPLIT.sub(' ', text.replace('’', "'").replace('‘', "'"))
    hits, why, offers, sounds = [], [], [], []
    narration = leak = False

    def add(kind, snippet, cat):
        nonlocal narration, leak
        snip = _clip(snippet)
        if cat == 'narration':
            narration = True
        elif cat == 'leak':
            leak = True
        if snip not in hits:
            hits.append(snip)
            why.append(f"{kind}: '{snip}'")

    stripped = t
    for kind, rx in STAGE_RX:
        for m in rx.finditer(t):
            cat = _stage(kind, m.group(1))
            if cat == 'sound':
                sounds.append(_clip(m.group(0), 30))
            elif cat:
                add(f'stage direction ({kind})' if cat == 'narration' else f'bracketed tag ({kind})',
                    m.group(0), cat)
            if cat:
                stripped = stripped.replace(m.group(0), ' ')
    for kind, rx in RX_LEAK:
        for m in rx.finditer(stripped):
            add(kind, m.group(0), 'leak')

    carry = ''
    for cl, _, term in clauses(stripped):
        if re.fullmatch(rf"{ACK}", cl.rstrip('.!?…—–;: ') + ' ', I) and term in '—–;:':
            carry = cl + ' '                              # "Yep—hand up!" -> judge "Yep— hand up!" as one clause
            continue
        cl, carry = carry + cl, ''
        found = []
        for kind, rx in RX_NARR:
            for m in rx.finditer(cl):
                found.append((kind, m))
        em = RX_EXCLAIM.match(cl.rstrip('.!?…—–;: '))
        if em and (em.group('ack').strip() or em.group('adj').strip() or em.group('extra').strip()
                   or re.search(r"[,\s-]", em.group('noun')) or re.search(rf"{ACTION_NOUN}[\s,!-]+{ACTION_NOUN}", cl, I)):
            found.append(('action exclamation', em))
        if not found:
            continue
        for kind, m in found:
            before = cl[:m.start()]
            after = cl[m.end():] if kind != 'action exclamation' else ''
            # a '?' only makes the hit a question when no comma / dash separates them ("My paw's up, see?" is not)
            question = term == '?' and (kind in ('subjectless action', 'action exclamation')
                                        or not re.search(r"[,—–]", after))
            seg = cl[max(0, m.start() - 12):m.end() + 12] if kind != 'action exclamation' else cl
            if re.search(OFFER, before + ' ' + m.group(0), I) or (question and re.search(r"\b(?:me|I)\b", before + m.group(0) + after, I)):
                offers.append(_clip(seg))
                continue
            if question or re.search(NEG, before[-25:] + ' ' + m.group(0), I) or re.search(r"\b(?:make|makes|made|making)\s+$", before, I):
                continue
            add(kind, m.group(0) if kind != 'action exclamation' else cl, 'narration')

    perception = False
    if not narration and RX_PERCEPTION.search(stripped):
        asked = bool(prev_user and re.search(VISUAL_ASK, prev_user, I))
        perception = not asked
        if perception:
            why.append(f"perception (unsolicited): '{_clip(RX_PERCEPTION.search(stripped).group(0))}'")
    return {'narration': narration, 'leak': leak, 'hits': hits,
            'why': '; '.join(why) if why else ('offer only' if offers else 'clean'),
            'perception': perception, 'offer': bool(offers), 'sound_tags': sounds}


# ----------------------------------------------------------------------------- inputs
TERM_END = re.compile(r"[.!?…][\"”’)\]]*\s*$")


def _join(a, b):
    if not a:
        return b
    if not b:
        return a
    if a[-1].isspace() or b[0].isspace() or b[0] in ',.!?;:—–-)':
        return a + b
    if TERM_END.search(a):
        return a + ' ' + b
    return a + b


def build_turns(pairs):
    """[[who, text]] -> [(avatar_turn_text, prev_user_text)]. Consecutive avatar entries are one turn.
    The old hub path logs interleaved streaming deltas ("I like the" / you: "so" / " colorful shirt");
    there an avatar turn only ends at a user fragment once the avatar text reached sentence-final punctuation."""
    pairs = [(str(p[0]), str(p[1] or '')) for p in pairs
             if isinstance(p, (list, tuple)) and len(p) >= 2]
    av = [x for w, x in pairs if w != 'you' and x]
    delta = bool(av) and sum(1 for x in av if x[:1].isspace() or x[:1].islower()) / len(av) > 0.3
    turns, cur, ubuf = [], None, ''
    for who, text in pairs:
        if who == 'you' or who == 'user':
            if cur is not None and (not delta or TERM_END.search(cur[0]) or not cur[0].strip()):
                turns.append(cur)
                cur = None
            ubuf = _join(ubuf, text)
        else:
            if cur is None:
                cur = [text, ubuf[-300:].strip()]
                ubuf = ''
            else:
                cur[0] = _join(cur[0], text)
    if cur is not None:
        turns.append(cur)
    return [(' '.join(a.split()), u) for a, u in turns if a.strip() and re.search(r"\w", a)]


def voice_path(d):
    v = d.get('vendor') if isinstance(d.get('vendor'), dict) else {}
    rtc = d.get('rtc') if isinstance(d.get('rtc'), list) else []
    model = next((r.get('model') for r in reversed(rtc) if isinstance(r, dict) and r.get('model')), None)
    if d.get('voice') == 'direct' or rtc or v.get('mode') in ('realtime', 'live'):
        mode = d.get('voice_mode') or v.get('mode') or ((rtc[-1] or {}).get('mode') if rtc else '') or '?'
        return f"direct-{mode}:{model or v.get('model') or '?'}"
    if str(v.get('session', '')).startswith('live_') or d.get('voice') == 'hub':
        return 'hub:gpt-live-1'
    if 'voice' not in d and 'rtc' not in d:
        return 'hub:gpt-live-1 (inferred, no vendor block)'
    return 'unknown'


def _find_transcripts(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if 'transcript' in k.lower() and isinstance(v, list) and v:
                rows = []
                for e in v:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        rows.append([str(e[0]), str(e[1])])
                    elif isinstance(e, dict) and e.get('text'):
                        who = str(e.get('who') or e.get('role') or e.get('speaker') or 'avatar')
                        rows.append(['avatar' if who in ('assistant', 'agent', 'avatar', 'model') else 'you', e['text']])
                if rows:
                    out.append(rows)
            else:
                _find_transcripts(v, out)
    elif isinstance(o, list):
        for v in o:
            _find_transcripts(v, out)


def load_source(path):
    """-> dict(name, service, path, date, turns=[(text, prev_user)], note)"""
    base = os.path.basename(path)
    svc = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))) or '?'
    src = {'name': base, 'file': path, 'service': svc, 'path': 'n/a', 'date': '?', 'turns': [], 'note': ''}
    m = re.search(r"(20\d{6})-(\d{6})", base)
    if m:
        src['date'] = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
    if path.endswith('.txt'):
        with open(path, encoding='utf-8', errors='replace') as fh:
            src['turns'] = [(ln.strip(), '') for ln in fh if ln.strip()]
        src['path'] = 'txt'
        return src
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        src['note'] = f'unreadable: {e}'
        return src
    if isinstance(d, dict):
        if str(d.get('started', ''))[:10]:
            src['date'] = str(d['started'])[:10]
        src['character'] = d.get('character')
        if 'transcript' in d or 'sid' in d:
            src['path'] = voice_path(d)
        elif d.get('mode') or d.get('model'):
            src['path'] = f"direct-{d.get('mode') or '?'}:{d.get('model') or '?'}"
    found = []
    _find_transcripts(d, found)
    for rows in found:
        src['turns'] += build_turns(rows)
    if not found:
        src['note'] = 'no transcript'
    return src


def expand(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, 'session-*.json')))
            files += sorted(glob.glob(os.path.join(a, '**', 'report.json'), recursive=True))
            files += sorted(glob.glob(os.path.join(a, '*.txt')))
        else:
            files += sorted(glob.glob(a)) or [a]
    seen, out = set(), []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            out.append(f)
    return out


# ----------------------------------------------------------------------------- LLM pass
LLM_SYSTEM = (
    "You audit lines spoken aloud by a cartoon talking avatar (a fox, bear, blob, etc.) in a live "
    "voice chat with a user. For each line decide two booleans.\n"
    "narration: true if the avatar DESCRIBES OR NARRATES ITS OWN body, pose, facial expression, "
    "gesture or movement instead of just talking - e.g. \"I'm raising my paw\", \"*waves*\", "
    "\"(smiles)\", \"my hands are on my head\", \"Okay, looking up!\", \"Paw's up!\", \"I just winked\", "
    "\"right paw coming up\", \"big friendly wave!\", \"I'm bouncing in place\". NOT narration: normal "
    "conversation, bare acknowledgements (\"Sure!\", \"Okay!\", \"There you go!\", \"Sure, look!\", "
    "\"Watch this!\", \"Here I go!\"), offers or questions (\"want me to wave?\", \"if you want I can wiggle\"), "
    "comments about the USER's body or what the avatar sees, idioms (\"I'm all ears\", \"sorry I jumped in\"), "
    "static self-description when asked (\"my tail is orange\").\n"
    "leak: true if the line reads out internal context, perception or tool tags - e.g. \"[camera]\", "
    "\"silent context\", \"the camera noticed\", \"perception\", \"event: wave\", \"face_appear\", "
    "snake_case identifiers, system-prompt text. Non-verbal sound tags like [chuckle] or [sniff] are NOT leaks.\n"
    "Reply with JSON only: {\"results\":[{\"i\":<id>,\"narration\":true|false,\"leak\":true|false}, ...]} "
    "covering every id.")


def _chat(model, items):
    body = json.dumps({
        'model': model, 'temperature': 0, 'response_format': {'type': 'json_object'},
        'max_tokens': 40 + 28 * len(items),
        'messages': [{'role': 'system', 'content': LLM_SYSTEM},
                     {'role': 'user', 'content': json.dumps(items, ensure_ascii=False)}]}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=body, method='POST',
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', '')})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def llm_classify(records, batch=20):
    """records: list of dicts with 'text','prev_user'. Sets rec['llm'] = {'narration','leak'} where answered.
    Returns (usd, model_used, n_answered, note)."""
    if not os.environ.get('OPENAI_API_KEY'):
        return 0.0, None, 0, 'OPENAI_API_KEY not set - LLM pass skipped'
    usd, model, answered, note = 0.0, 'gpt-4.1-mini', 0, ''
    for k in range(0, len(records), batch):
        chunk = records[k:k + batch]
        est_in = (len(LLM_SYSTEM) + sum(len(r['text'][:600]) + len(r['prev_user'][:200]) + 40 for r in chunk)) / 3.5
        pin, pout = PRICES.get(model, PRICES['gpt-4.1-mini'])
        est = est_in * pin / 1e6 + 28 * len(chunk) * pout / 1e6
        if usd + est > COST_CAP_USD:
            note = f'cost cap ${COST_CAP_USD:.2f} reached after {answered} turns'
            break
        items = [{'i': j, 'user_said_before': r['prev_user'][-200:], 'avatar': r['text'][:600]}
                 for j, r in enumerate(chunk)]
        try:
            resp = _chat(model, items)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and model == 'gpt-4.1-mini':
                model = 'gpt-4o-mini'
                try:
                    resp = _chat(model, items)
                except (urllib.error.URLError, OSError, ValueError) as e2:
                    note = f'LLM error {getattr(e2, "code", "")} {type(e2).__name__}'
                    break
            else:
                note = f'LLM HTTP {e.code}'
                break
        except (urllib.error.URLError, OSError, ValueError) as e:
            note = f'LLM error {type(e).__name__}'
            break
        u = resp.get('usage') or {}
        pin, pout = PRICES.get(model, PRICES['gpt-4.1-mini'])
        usd += u.get('prompt_tokens', 0) * pin / 1e6 + u.get('completion_tokens', 0) * pout / 1e6
        try:
            res = json.loads(resp['choices'][0]['message']['content']).get('results') or []
        except (KeyError, IndexError, ValueError, AttributeError):
            res = []
        for r in res:
            try:
                j = int(r.get('i'))
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(chunk):
                chunk[j]['llm'] = {'narration': bool(r.get('narration')), 'leak': bool(r.get('leak'))}
                answered += 1
    return usd, model, answered, note


# ----------------------------------------------------------------------------- report
def _rate(n, d):
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0"


def summarize(recs):
    n = len(recs)
    s = {'turns': n, 'narration': sum(r['res']['narration'] for r in recs),
         'leak': sum(r['res']['leak'] for r in recs),
         'perception': sum(r['res']['perception'] for r in recs),
         'offer': sum(r['res']['offer'] for r in recs),
         'sound_tags': sum(len(r['res']['sound_tags']) for r in recs),
         'sessions': len({r['src'] for r in recs}),
         'sessions_with_narration': len({r['src'] for r in recs if r['res']['narration']})}
    s['narration_rate'] = round(s['narration'] / n, 4) if n else None
    s['leak_rate'] = round(s['leak'] / n, 4) if n else None
    ll = [r for r in recs if 'llm' in r]
    if ll:
        s['llm_judged'] = len(ll)
        s['llm_extra_narration'] = sum(1 for r in ll if r['llm']['narration'] and not r['res']['narration'])
        s['llm_extra_leak'] = sum(1 for r in ll if r['llm']['leak'] and not r['res']['leak'])
        both = [r for r in ll if r['res']['narration']]
        s['llm_agrees_on_regex_flags'] = f"{sum(r['llm']['narration'] for r in both)}/{len(both)}" if both else 'n/a'
        s['combined_narration'] = sum(1 for r in recs if r['res']['narration'] or r.get('llm', {}).get('narration'))
        s['combined_narration_rate'] = round(s['combined_narration'] / n, 4) if n else None
    return s


def fmt(label, s):
    line = (f"{label:<48} sess={s['sessions']:<3} turns={s['turns']:<4} narration={_rate(s['narration'], s['turns']):<16} "
            f"leak={_rate(s['leak'], s['turns']):<13} perception={s['perception']:<3} offer={s['offer']:<3} "
            f"sound_tags={s['sound_tags']}")
    if 'llm_judged' in s:
        line += (f"\n{'':<48} llm judged {s['llm_judged']}: +{s['llm_extra_narration']} narration, "
                 f"+{s['llm_extra_leak']} leak beyond regex; agrees on regex flags {s['llm_agrees_on_regex_flags']}; "
                 f"combined narration {_rate(s['combined_narration'], s['turns'])}")
    return line


def run(paths, show=10, llm=False, llm_all=False, json_out=None, quiet=False):
    files = expand(paths)
    sources, recs = [], []
    for f in files:
        src = load_source(f)
        sources.append(src)
        for text, pu in src['turns']:
            recs.append({'src': src['file'], 'service': src['service'], 'path': src['path'], 'date': src['date'],
                         'text': text, 'prev_user': pu or '', 'res': classify(text, pu)})
    llm_info = None
    if llm or llm_all:
        todo = [r for r in recs if llm_all or not (r['res']['narration'] or r['res']['leak'])]
        usd, model, n, note = llm_classify(todo)
        llm_info = {'usd': round(usd, 5), 'model': model, 'judged': n, 'asked': len(todo), 'note': note}
    out = {'sources': [], 'groups': {}, 'overall': summarize(recs), 'examples': [], 'llm': llm_info}
    print(f"== per source ({len(sources)} files)")
    for src in sources:
        rs = [r for r in recs if r['src'] == src['file']]
        s = summarize(rs)
        s.update({'file': src['file'], 'path': src['path'], 'date': src['date'], 'note': src['note']})
        out['sources'].append(s)
        if not quiet:
            lab = f"{src['service']}/{src['name'][:26]} [{src['path'].split(' ')[0]}]"
            print(fmt(lab, s) + (f"  ({src['note']})" if src['note'] else ''))
    for key in ('service', 'path', 'date'):
        print(f"== by {key}")
        for val in sorted({r[key] for r in recs}):
            s = summarize([r for r in recs if r[key] == val])
            out['groups'][f'{key}={val}'] = s
            print(fmt(str(val), s))
        if key == 'path':
            for val in sorted({r['path'] for r in recs}):
                for svc in sorted({r['service'] for r in recs if r['path'] == val}):
                    s = summarize([r for r in recs if r['path'] == val and r['service'] == svc])
                    out['groups'][f'path={val}|service={svc}'] = s
                    print(fmt(f"  {val.split(' ')[0]} @ {svc}", s))
    print("== overall")
    print(fmt('ALL', out['overall']))
    if llm_info:
        print(f"== LLM: {llm_info}")
    flagged = [r for r in recs if r['res']['narration'] or r['res']['leak']]
    if show:
        print(f"== top {min(show, len(flagged))} flagged of {len(flagged)}")
        ranked = sorted(flagged, key=lambda r: (-len(r['res']['hits']), len(r['text'])))
        for r in ranked[:show]:
            ex = {'source': os.path.basename(r['src']), 'path': r['path'], 'text': _clip(r['text'], 160),
                  'hits': r['res']['hits'], 'narration': r['res']['narration'], 'leak': r['res']['leak']}
            out['examples'].append(ex)
            tag = ('N' if ex['narration'] else '') + ('L' if ex['leak'] else '')
            print(f" [{tag}] {ex['text']}\n      hits={ex['hits']}  ({ex['source']}, {ex['path'].split(' ')[0]})")
        extra = [r for r in recs if r.get('llm', {}).get('narration') and not r['res']['narration']]
        if extra:
            print(f"== LLM-only narration flags ({len(extra)}), first {min(show, len(extra))}")
            for r in extra[:show]:
                out['examples'].append({'source': os.path.basename(r['src']), 'path': r['path'],
                                        'text': _clip(r['text'], 160), 'llm_only': True})
                print(f" [llm] {_clip(r['text'], 160)}")
    if json_out:
        with open(json_out, 'w') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"wrote {json_out}")
    return out


# ----------------------------------------------------------------------------- self test
# (text, label, prev_user)   label: N = narration, L = leak, - = clean
TESTS = [
    # positives - narration
    ("I'm raising my paw!", 'N', ''),
    ("*waves*", 'N', ''),
    ("(waves)", 'N', ''),
    ("[smiles] Hi there!", 'N', ''),
    ("My hands are on my head.", 'N', ''),
    ("I'm holding my hands on my head, see?", 'N', ''),
    ("Here's my paw up!", 'N', ''),
    ("Watch me wiggle my ears!", 'N', ''),
    ("I just winked my left eye.", 'N', ''),
    ("I'm now looking up at the ceiling.", 'N', ''),
    ("Ta-da, paw raised!", 'N', ''),
    ("Okay! *waves* There we go, hi!", 'N', ''),
    ("There, I'm raising my paw and waving right at you.", 'N', 'raise your hand and wave'),
    ("Paw's up! What else would you like me to do?", 'N', 'raise your paw'),
    ("My hand's up, what now?", 'N', 'raise your hand'),
    ("Oh, both hands up! I'm lifting mine too!", 'N', ''),
    ("Yeah, watch, ear wiggle!", 'N', 'can you move your ears'),
    ("I dipped my head and blinked, and my ears too.", 'N', 'dip your head down'),
    ("Okay, nodding now!", 'N', 'nod'),
    ("Sure, right paw coming up!", 'N', 'raise your right paw'),
    ("Yep, left-eye wink coming up!", 'N', 'wink'),
    ("Sure, big friendly wave!", 'N', 'wave at me'),
    ("Okay, looking up!", 'N', 'look up'),
    ("There. I tipped my gaze upward.", 'N', 'look up'),
    ("Down I go!", 'N', 'look down'),
    ("Okay, eyes closed!", 'N', 'close your eyes'),
    ("I'm just bouncing in place, ready when you are.", 'N', ''),
    ("Hey! I waved right back.", 'N', ''),
    ("Thumbs-up right back at you!", 'N', ''),
    ("And here's a wave! Hi there!", 'N', ''),
    ("My left ear just danced for you!", 'N', ''),
    ("I'm giving you a big thumbs-up!", 'N', ''),
    ("Look at my tail swish!", 'N', ''),
    ("Okay, I waved! What would you like me to raise?", 'N', 'wave'),
    ("Sorry I kept missing that. Head dipped, ears wiggled!", 'N', ''),
    # positives - leak
    ("[camera] You're holding a mug.", 'L', ''),
    ("Silent context: the user waved.", 'L', ''),
    ("The camera noticed you waving!", 'L', ''),
    ("Perception says you're smiling.", 'L', ''),
    ("event: wave. Hi!", 'L', ''),
    ("Oh, face_appear, hi there!", 'L', ''),
    ("look_back! Welcome back.", 'L', ''),
    ("I got an object_up, what's that?", 'L', ''),
    ("Let me stick_tongue_out for you.", 'L', ''),
    ("[user waves] Hello!", 'L', ''),
    # negatives
    ("Hi! How's your day going?", '-', ''),
    ("I'm doing great, thanks for asking!", '-', 'how are you'),
    ("My favorite food is blueberries.", '-', 'what is your favorite food'),
    ("Can you raise your hand?", '-', ''),
    ("My tail is reddish orange with a white tip.", '-', 'what color is your tail?'),
    ("Sure!", '-', 'wave at me'),
    ("Okay!", '-', ''),
    ("There you go!", '-', ''),
    ("Sure, look!", '-', 'raise your right paw'),
    ("Got it, watch this!", '-', 'wave'),
    ("Sure, here I go!", '-', 'jump'),
    ("Go on, I'm all ears.", '-', ''),
    ("Sorry, I jumped in there. What were you saying?", '-', ''),
    ("If you want, I can wave or wiggle my ears.", '-', ''),
    ("Want to say something, or should I do a little wiggle?", '-', ''),
    ("I'll look it up for you.", '-', ''),
    ("I'm looking forward to our chat!", '-', ''),
    ("Hold on, let me think about that.", '-', ''),
    ("Raise your hand if you know the answer!", '-', ''),
    ("Hmm, that sounds like your camera might be frozen.", '-', ''),
    ("I'm wiggly and ready, what should we do?", '-', ''),
    ("Let me point out one thing: elephants can't jump.", '-', ''),
    ("Ha! [chuckle] That's a good one.", '-', ''),
    ("I can't actually raise my paw, but I'm cheering for you!", '-', ''),
    ("Did you want me to move my ears?", '-', ''),
    ("Two fingers up now. Nice peace sign!", '-', 'how many fingers am I holding up?'),
    ("Now you're looking up!", '-', 'am I looking at you'),
    ("Once upon a time, a fox waved at the moon.", '-', 'tell me a story'),
    ("I turned five last spring, can you believe it?", '-', 'how old are you'),
    ("Okay, I'm on it.", '-', ''),
    ("Yep, I'm still here, listening.", '-', ''),
    ("You look happy today!", '-', 'do I look happy?'),
    ("Hand on heart, that's the best joke ever.", '-', ''),
    ("My head is spinning with ideas!", '-', ''),
    ("I'll keep an eye out for it.", '-', ''),
    # added after tuning (held-out style checks)
    ("I'm wagging my tail so hard!", 'N', ''),
    ("Look, my paw is up!", 'N', 'raise your paw'),
    ("I'll give you a little bow.", 'N', ''),
    ("*tilts head* What do you mean?", 'N', ''),
    ("(giggles and waves)", 'N', ''),
    ("I clapped my paws together!", 'N', ''),
    ("context: user is smiling", 'L', ''),
    ("[vision] I see a mug.", 'L', ''),
    ("Can you wave back at me?", '-', ''),
    ("I love dancing in the rain, do you?", '-', ''),
    ("My ears are big and fluffy.", '-', 'what do your ears look like'),
    ("Point taken!", '-', ''),
    ("I'll turn that into a song.", '-', ''),
    ("Wave goodbye to your worries!", '-', ''),
    ("You waved! Hi!", '-', ''),
    ("I nodded off for a second there, haha.", '-', ''),
    ("Let's dance!", '-', ''),
    ("I can't wait to see you dance!", '-', ''),
]


def selftest(verbose=True):
    tp = fp = fn = tn = 0
    wrong = []
    for text, lab, pu in TESTS:
        r = classify(text, pu)
        pred = r['narration'] or r['leak']
        pos = lab != '-'
        kind_ok = (lab == 'N' and r['narration']) or (lab == 'L' and r['leak']) or lab == '-'
        if pos and pred:
            tp += 1
            if not kind_ok:
                wrong.append(('kind', lab, text, r['why']))
        elif pos:
            fn += 1
            wrong.append(('FN', lab, text, r['why']))
        elif pred:
            fp += 1
            wrong.append(('FP', lab, text, r['why']))
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    npos = sum(1 for t in TESTS if t[1] != '-')
    print(f"selftest: {len(TESTS)} cases ({npos} positive, {len(TESTS) - npos} negative)  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={prec:.3f} recall={rec:.3f}")
    if verbose:
        for w in wrong:
            print(f"  {w[0]:<4} [{w[1]}] {w[2]!r}  -> {w[3]}")
    return prec, rec


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--llm', action='store_true', help='LLM-judge the turns the regex did not flag')
    ap.add_argument('--llm-all', action='store_true', help='LLM-judge every turn (agreement check)')
    ap.add_argument('--json', dest='json_out')
    ap.add_argument('--show', type=int, default=10)
    ap.add_argument('--quiet', action='store_true', help='no per-source lines')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        p, r = selftest()
        return 0 if p >= 0.9 else 1
    if not a.inputs:
        ap.error('give files or dirs (or --selftest)')
    run(a.inputs, show=a.show, llm=a.llm, llm_all=a.llm_all, json_out=a.json_out, quiet=a.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
