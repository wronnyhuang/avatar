#!/usr/bin/env python3
"""Planner scorer, stage 4: the vision judge. Does the rendered clip show what was asked, naturally, without glitches?

  judge_item(item, itemdir, judge_name) -> {"score": 1-5, "shows_request", "natural", "glitches": [...], "observed", "reason",
                                            "judge", "usd", "seconds", "cached"}
  python3 tools/plan_judge.py <itemdir> --item '<json or @file.jsonl:id>' --judge gem38-video        (one clip, by hand)

Judges (name = <model>-<media>):
  gem38-video   gemini-3.8-flash, the clip itself (inline mp4, sampled at 6 fps) + the face close-up clip for face items
  gem35-video   gemini-3.5-flash, same input
  gem38-strips  gemini-3.8-flash on the frame strips (4 fps tiles with timestamps)
  sol-strips    gpt-6.1-sol (reasoning low) on the frame strips (+ the face strips for face items)
  gem38-both    gemini-3.8-flash, clip + strips
  mini-strips   gpt-5.4-mini (reasoning low) on the frame strips
  sol-strips-med  gpt-6.1-sol, reasoning medium
  (a free-tier GEMINI_API_KEY allows only a few requests a day per model — too few to judge a whole set)
The judge never sees which planner config produced the clip: only the request, the item's `expected` / `must_not`, the scene
note and the media. Verdicts are cached by content hash (judge + prompt + media sha256) in PLAN_JUDGE_CACHE
(default <repo>/runs/cache/plan_judge). The pick and the pass threshold: tools/plan_judge.json (see tools/plan_calibrate.py).
Keys: GEMINI_API_KEY (gem* judges), OPENAI_API_KEY (sol*/mini* judges); the claude judges use the `claude` CLI
($AVATAR_CLAUDE_BIN, else `claude` on PATH).
"""
import base64, hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

CACHE = Path(os.environ.get('PLAN_JUDGE_CACHE') or (Path(__file__).resolve().parent.parent / 'runs' / 'cache' / 'plan_judge'))
PROMPT_VERSION = 'v1'
JUDGES = {
    'gem38-video': {'api': 'gemini', 'model': 'gemini-3.8-flash', 'media': ('video',), 'price': (0.5, 3.0)},
    'gem35-video': {'api': 'gemini', 'model': 'gemini-3.5-flash', 'media': ('video',), 'price': (0.5, 3.0)},
    'gem38-strips': {'api': 'gemini', 'model': 'gemini-3.8-flash', 'media': ('strips',), 'price': (0.5, 3.0)},
    'gem38-both': {'api': 'gemini', 'model': 'gemini-3.8-flash', 'media': ('video', 'strips'), 'price': (0.5, 3.0)},
    'sol-strips': {'api': 'openai', 'model': 'gpt-6.1-sol', 'media': ('strips',), 'price': (2.0, 10.0)},
    'mini-strips': {'api': 'openai', 'model': 'gpt-5.4-mini', 'media': ('strips',), 'price': (0.75, 4.5)},
    'sol-strips-med': {'api': 'openai', 'model': 'gpt-6.1-sol', 'media': ('strips',), 'price': (2.0, 10.0), 'effort': 'medium'},
    # v2: character close-up strips for in-place motion (plan_render zoom_strips) + the reading rules below
    'sol-zoom': {'api': 'openai', 'model': 'gpt-6.1-sol', 'media': ('zoom',), 'price': (2.0, 10.0), 'prompt': 'v2'},
    'mini-zoom': {'api': 'openai', 'model': 'gpt-5.4-mini', 'media': ('zoom',), 'price': (0.75, 4.5), 'prompt': 'v2'},
    'sol-zoom-med': {'api': 'openai', 'model': 'gpt-6.1-sol', 'media': ('zoom',), 'price': (2.0, 10.0), 'prompt': 'v2', 'effort': 'medium'},
    # the same v2 zoom input + rubric through the `claude -p` CLI (its own login, no API key here).
    # usd 0 (CLI subscription); the CLI's API-equivalent cost is kept as 'api_usd'.
    'sonnet-zoom': {'api': 'claude', 'model': 'claude-sonnet-5-5', 'media': ('zoom',), 'price': (0.0, 0.0), 'prompt': 'v2', 'effort': 'low'},
    'opus-zoom': {'api': 'claude', 'model': 'claude-opus-5-5', 'media': ('zoom',), 'price': (0.0, 0.0), 'prompt': 'v2', 'effort': 'low'},
    'sonnet-zoom-med': {'api': 'claude', 'model': 'claude-sonnet-5-5', 'media': ('zoom',), 'price': (0.0, 0.0), 'prompt': 'v2', 'effort': 'medium'},
}
READING = """How to read the checklist: it describes the ideal performance. Strict: the action itself, its side, its order, and its
count when the request or the checklist states a number. Approximate: durations, heights and sizes - about half of the stated
amount still counts when the action is plainly readable. Elements introduced by "may", "optional", "e.g." or "natural" are not
required. A strip samples the clip only a few times a second: a repeated motion (waving, nodding, shaking, clapping) shows as the
limb or head in different positions in neighbouring tiles - that IS the motion, do not demand that every phase was caught; but
if neighbouring tiles show no difference at all, it did not move. A brief unrequested extra (a blink, a small smile, standing
back up at the end when the checklist does not say it stays) is not a failure. Score 1 is for a clip that does not show the
request at all or hits a "must NOT" outcome; one missing secondary element is a 3 or a 4, never a 1.
"""
VIDEO_FPS = 6

SYSTEM = (
    "You grade short animation clips of a 3D plush character: a small, round, white soft-toy creature with a big head, two "
    "small ears, black eyes, a small mouth, stubby arms ending in paws, short legs and a small tail. It lives in a furnished "
    "room (rug, chair, desk, shelves, a side table with headphones). Someone asked it to do something; you decide whether "
    "the clip shows it. Grade only what is visible. Never give the benefit of the doubt for motion or expressions you cannot "
    "see, and never fail a clip for something you merely assume. Be strict and literal about the checklist you are given.")

RUBRIC = """REQUEST (spoken to the character): "{text}"
{setup}A viewer must SEE: {expected}
It must NOT be: {must_not}
{scene}
Conventions: the clip starts about 0.3 s before the character starts moving and ends shortly after it stops{setup_clip}. The
character starts facing the camera; while it faces the camera its OWN left is the RIGHT side of the picture and its own right
is the left side of the picture. The camera may zoom out or pan to follow it: judge motion relative to the room and floor,
not the frame. Idle life (breathing, blinks, small tail or ear movement) is normal and is neither the action nor a glitch.
{media}
Steps:
1. observed: describe what the character actually does, in order, with times (body, arms/paws, head, face).
2. Compare with the checklist: is every element of "must SEE" visible? does anything in "must NOT" happen? If the request
   has a count, count the repetitions; if a side, check the side; if a hold, check it is held.
3. natural: does it move like a believable character (smooth, weighty, readable timing)?
4. glitches: list any of: teleport / position pops, limbs passing through the body or furniture, sinking into or floating
   above the floor (when not jumping), a pose frozen unnaturally, jitter, body parts detached or distorted, an object
   appearing from nowhere.
Score (integer 1-5):
5 = the request is clearly and fully shown, natural, no glitches.
4 = the request is clearly shown; small flaws only (a little small or short, a minor stiffness, a harmless extra).
3 = a recognisable attempt, but an important element of "must SEE" is missing, weak or ambiguous, or the count / side /
    hold is off, or a noticeable glitch mars it.
2 = something loosely related happens, but a viewer would not say the request was done.
1 = not shown: a different action, essentially nothing, a "must NOT" outcome, or badly broken.
shows_request = true only for scores 4 and 5.
Answer with one JSON object only:
{{"observed": "...", "seen": ["<must-SEE element>: yes|no|unclear", ...], "must_not_hit": true|false, "shows_request": true|false,
"natural": true|false, "glitches": ["..."], "score": 1-5, "reason": "<one or two sentences>"}}"""


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def text_of(it):
    return str(it.get('utterance') or it.get('text') or it.get('request') or it.get('description') or '').strip()


SEATED = ("You grade short animation clips of a 3D cartoon {what}: a friendly character seen from the chest up, sitting in place "
          "facing the camera (it has a head with a face, ears, two front paws / arms{tail}; it does not walk around). Someone asked it "
          "to do something; you decide whether the clip shows it. Grade only what is visible. Never give the benefit of the doubt for "
          "motion or expressions you cannot see, and never fail a clip for something you merely assume. Be strict and literal about "
          "the checklist you are given.")
WHAT = {'fox': ('fox', True), 'monkey': ('monkey', True), 'bear': ('bear', False), 'rabbit': ('rabbit', False), 'hippo': ('hippo', False)}


def system_for(item):
    c = str(item.get('_character') or 'muse')
    if c == 'muse':
        return SYSTEM
    what, tail = WHAT.get(c, (c, False))
    return SEATED.format(what=what, tail=', and a tail that can show beside its body' if tail else '')


def build_prompt(item, has_face, media, version='v1', zoomed=False):
    setup = ''
    setup_clip = ''
    if item.get('setup'):
        setup = (f'Just before this, the character had been asked: "{item["setup"]}" — that earlier action is the starting state '
                 f'and is NOT what you grade. The REQUEST above was given about 1.0 s into the clip.\n')
        setup_clip = ' (here: it starts 1 s before the request and runs at least 6 s after it)'
    scene = item.get('scene')
    sc = ''
    if scene is not None:
        gone = [o for o in ('chair', 'desk', 'laptop', 'headphones') if o not in scene]
        if item.get('subcategory') in ('absent_decline', 'mime') or scene:
            sc = ('Scene for this request: ' + (('the usable objects are: ' + ', '.join(scene) + '.') if scene else 'there are NO usable objects.')
                  + (f" Treat {', '.join('the ' + g for g in gone)} as NOT there for the character, even if the room's furniture is drawn in the "
                     f"background: using one of them is a failure.\n" if gone else '\n'))
    m = []
    if 'video' in media:
        m.append('You get the clip as video' + (' and then a second video: a close-up of the head over the same time span (use it for eyes, lids, brows, mouth, tongue, cheeks)' if has_face else '') + '.')
    if 'strips' in media:
        m.append('You get frame strips: tiles in time order, left to right then top to bottom, each stamped with its time in the clip'
                 + ('; after the whole-body strips come close-up strips of the head over the same time span (use them for eyes, lids, brows, mouth, tongue, cheeks)' if has_face else '') + '.')
    if item.get('_caption'):
        # the page declined / refused: the voice says so (the clip has no sound)
        sc += (f'While this happens the character SAYS aloud (the clip has no sound): "{item["_caption"]}" — said honestly in its own words. '
               'Take that as its spoken reply: where the checklist allows a decline, this is the spoken part of it.\n')
    if 'zoom' in media:
        if zoomed:
            m.append('You get frame strips, tiles in time order (left to right, then top to bottom), each stamped with its time in the clip. First the whole room, '
                     'small (for where the character is and what is around it); then CLOSE-UP strips of the character over the same time span, sampled more '
                     'densely - judge the action on these'
                     + ('; then close-up strips of the head (use them for eyes, lids, brows, mouth, tongue, cheeks)' if has_face else '') + '.')
        else:
            m.append('You get frame strips of the whole room: tiles in time order, left to right then top to bottom, each stamped with its time in the clip'
                     + ('; after them come close-up strips of the head over the same time span (use them for eyes, lids, brows, mouth, tongue, cheeks)' if has_face else '') + '.')
    if version == 'v2':
        m.append('\n' + READING.strip())
    return RUBRIC.format(text=text_of(item), setup=setup, expected=item.get('expected') or '(the request itself)', must_not=item.get('must_not') or '(nothing listed)',
                         scene=sc, setup_clip=setup_clip, media=' '.join(m))


def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=dict(headers, **{'content-type': 'application/json'}))
    last = None
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            txt = e.read()[:1200].decode("utf8", "replace")
            last = f'HTTP {e.code}: {txt[:300]}'
            if e.code not in (429, 500, 502, 503, 504):
                break
            if e.code == 429 and ('per day' in txt or 'free_tier' in txt or 'insufficient_quota' in txt):
                last = f'HTTP 429 DAILY QUOTA (no retry): {txt[:500]}'
                break
        except Exception as e:  # noqa: BLE001
            last = f'{type(e).__name__}: {str(e)[:200]}'
        time.sleep(3 + 6 * k)
    raise RuntimeError(last)


def _parse(txt):
    txt = txt.strip()
    if txt.startswith('```'):
        txt = txt.strip('`')
        txt = txt[txt.index('{'):]
    a, b = txt.find('{'), txt.rfind('}')
    return json.loads(txt[a:b + 1])


def call_gemini(J, prompt, files, SYSTEM=SYSTEM):
    parts = [{'text': prompt}]
    for kind, p in files:
        data = base64.b64encode(Path(p).read_bytes()).decode()
        if kind == 'video':
            parts.append({'inlineData': {'mimeType': 'video/mp4', 'data': data}, 'videoMetadata': {'fps': VIDEO_FPS}})
        else:
            parts.append({'inlineData': {'mimeType': 'image/jpeg', 'data': data}})
    body = {'systemInstruction': {'parts': [{'text': SYSTEM}]}, 'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json', 'maxOutputTokens': 6000, 'thinkingConfig': {'thinkingLevel': 'low'}}}
    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{J['model']}:generateContent", body, {'x-goog-api-key': os.environ['GEMINI_API_KEY']})
    u = r.get('usageMetadata', {})
    txt = ''.join(p.get('text', '') for p in (r.get('candidates') or [{}])[0].get('content', {}).get('parts', []) if not p.get('thought'))
    tin, tout = u.get('promptTokenCount', 0), u.get('candidatesTokenCount', 0) + u.get('thoughtsTokenCount', 0)
    return _parse(txt), tin, tout


def call_openai(J, prompt, files, SYSTEM=SYSTEM):
    content = [{'type': 'input_text', 'text': prompt}]
    for kind, p in files:
        content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(Path(p).read_bytes()).decode(), 'detail': 'low' if kind == 'image_low' else 'high'})
    body = {'model': J['model'], 'instructions': SYSTEM, 'input': [{'role': 'user', 'content': content}], 'reasoning': {'effort': J.get('effort', 'low')},
            'text': {'format': {'type': 'json_object'}}, 'max_output_tokens': 6000}
    r = _post('https://api.openai.com/v1/responses', body, {'authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']})
    txt = ''.join(c.get('text', '') for o in r.get('output', []) if o.get('type') == 'message' for c in o.get('content', []))
    u = r.get('usage', {})
    return _parse(txt), u.get('input_tokens', 0), u.get('output_tokens', 0)


def _salvage(txt):
    """the verdict JSON; an answer whose JSON is broken (an unescaped quote inside "observed") keeps its score / flags
    when they are unambiguous"""
    import re
    try:
        return _parse(txt)
    except ValueError:
        sc = re.findall(r'"score"\s*:\s*([1-5])\b', txt)
        if len(set(sc)) != 1:
            raise
        v = {'score': int(sc[0]), 'salvaged': True}
        for k in ('shows_request', 'natural', 'must_not_hit'):
            m = re.search(r'"%s"\s*:\s*(true|false)' % k, txt)
            if m:
                v[k] = m.group(1) == 'true'
        m = re.search(r'"reason"\s*:\s*"(.*?)"\s*[,}]', txt, re.S)
        v['reason'] = m.group(1) if m else ''
        return v


def call_claude(J, prompt, files, SYSTEM=SYSTEM):
    """one judge call through the local claude CLI on the subscription: one stream-json user message (the prompt + the
    strips as base64 JPEG blocks), no tools, no project context; -> (verdict, in tokens, out tokens); J['_api_usd'] = the
    CLI's API-equivalent cost"""
    import shutil, tempfile
    content = [{'type': 'text', 'text': prompt}]
    for kind, p in files:
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                     'data': base64.b64encode(Path(p).read_bytes()).decode()}})
    msg = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}) + '\n'
    cmd = [os.environ.get('AVATAR_CLAUDE_BIN') or shutil.which('claude') or 'claude', '-p', '--input-format', 'stream-json',
           '--output-format', 'stream-json', '--verbose', '--model', J['model'], '--system-prompt', SYSTEM, '--tools', '',
           '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence']
    if J.get('effort'):
        cmd += ['--effort', J['effort']]
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY')}
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    last = None
    for k in range(3):
        with tempfile.TemporaryDirectory(prefix='plan-judge-') as cwd:
            try:
                r = subprocess.run(cmd, input=msg, capture_output=True, text=True, timeout=300, cwd=cwd, env=env)
            except subprocess.TimeoutExpired:
                last = 'claude judge timeout'
                continue
        res = None
        for line in r.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get('type') == 'result':
                res = d
        if res and not res.get('is_error') and isinstance(res.get('result'), str):
            u = res.get('usage') or {}
            J['_api_usd'] = res.get('total_cost_usd')
            tin = (u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0) + (u.get('cache_creation_input_tokens') or 0)
            return _salvage(res['result']), tin, u.get('output_tokens') or 0
        last = f"claude rc={r.returncode}: {str((res or {}).get('result') or r.stderr)[-300:]}"
        if 'limit' in last.lower() and 'reset' in last.lower():
            break                                  # the subscription's usage limit: no retry
        time.sleep(3 + 6 * k)
    raise RuntimeError(last)


CALLS = {'gemini': call_gemini, 'openai': call_openai, 'claude': call_claude}


def judge_item(item, itemdir, judge_name, use_cache=True):
    J = JUDGES[judge_name]
    d = Path(itemdir)
    rj = json.loads((d / 'render.json').read_text())
    has_face = bool(item.get('face_closeup')) and bool(rj.get('face'))
    files = []
    if 'video' in J['media']:
        files.append(('video', d / rj['clip']))
        if has_face:
            files.append(('video', d / rj['face']))
    if 'strips' in J['media']:
        files += [('image', d / f) for f in rj['strips']]
        if has_face:
            files += [('image', d / f) for f in rj.get('face_strips', [])]
    zoomed = False
    if 'zoom' in J['media']:
        z = rj.get('zoom') or {}
        zoomed = z.get('mode') == 'zoom' and bool(z.get('files'))
        if zoomed:
            files += [('image_low', d / f) for f in rj['strips'][:2]] + [('image', d / f) for f in z['files']]
        else:
            files += [('image', d / f) for f in rj['strips']]
        if has_face:
            files += [('image', d / f) for f in rj.get('face_strips', [])[:2]]
    prompt = build_prompt(item, has_face, J['media'], J.get('prompt', 'v1'), zoomed)
    system = system_for(item)
    key = _sha(('|'.join([judge_name, J['model'], PROMPT_VERSION, system, prompt, str(VIDEO_FPS), J.get('effort', '')] + [k for k, _ in files if k == 'image_low'] + [_sha(Path(p).read_bytes()) for _, p in files])).encode())[:32]
    cf = CACHE / f'{key}.json'
    if use_cache and cf.exists():
        v = json.loads(cf.read_text())
        v.update({'cached': True, 'usd': 0.0, 'usd_first': v.get('usd_first', v.get('usd'))})
        return v
    t0 = time.time()
    try:
        J = dict(J)
        raw, tin, tout = CALLS[J['api']](J, prompt, files, system)
    except Exception as e:  # noqa: BLE001
        return {'score': None, 'error': str(e)[:400], 'judge': judge_name, 'usd': 0.0}
    usd = (tin * J['price'][0] + tout * J['price'][1]) / 1e6
    try:
        score = max(1, min(5, int(round(float(raw.get('score'))))))
    except Exception:  # noqa: BLE001
        return {'score': None, 'error': 'no score in ' + json.dumps(raw)[:300], 'judge': judge_name, 'usd': usd}
    v = {'score': score, 'shows_request': bool(raw.get('shows_request')) and score >= 4, 'natural': bool(raw.get('natural')),
         'glitches': raw.get('glitches') if isinstance(raw.get('glitches'), list) else [], 'observed': str(raw.get('observed') or '')[:1500],
         'seen': raw.get('seen'), 'must_not_hit': bool(raw.get('must_not_hit')), 'reason': str(raw.get('reason') or '')[:600],
         'judge': judge_name, 'model': J['model'], 'prompt_version': PROMPT_VERSION, 'tokens': [tin, tout], 'usd': round(usd, 5), 'usd_first': round(usd, 5),
         'seconds': round(time.time() - t0, 1), 'key': key, 'cached': False, 'api_usd': J.get('_api_usd')}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = str(cf) + f'.{os.getpid()}.tmp'
    Path(tmp).write_text(json.dumps(v, indent=1))
    os.replace(tmp, cf)
    return v


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('itemdir'); ap.add_argument('--item', required=True); ap.add_argument('--judge', default='gem38-video'); ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    if a.item.startswith('@'):
        f, i = a.item[1:].rsplit(':', 1)
        item = next(x for x in map(json.loads, open(f)) if x['id'] == i)
    else:
        item = json.loads(a.item)
    print(json.dumps(judge_item(item, a.itemdir, a.judge, use_cache=not a.no_cache), indent=1))
