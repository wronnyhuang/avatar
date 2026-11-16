#!/usr/bin/env python3
"""aaa_judge -- is Juno at AAA real-time game-character quality?  Blind harness ( JUDGE).

  aaa_judge.py round --ours DIR --name NAME [--n 20] [--sets closeup,bust[,clips]] [--jobs 8] [--model sonnet]
  aaa_judge.py control [--n 24]        (re)build the reference-vs-reference noise band (cached; `round` does it on demand)
  aaa_judge.py refscores               (re)score every reference image on the absolute scorecard (cached)

DIR (ours) = what render_shots.py writes: DIR/closeup/<shot>.png and DIR/bust/<shot>.png (1920x1080, shots from shots_aaa.json), optionally
DIR/session/*.png (frames cropped from real recorded sessions: scorecard only, no pairing) and DIR/clips/*.mp4 (3-4 s talking clips).
Output: T/judge/rounds/NAME/{report.md,results.json,tells.json,inputs/,raw/}.

Method (all judging = Claude on the subscription through `claude -p --model sonnet --tools Read`; never OpenAI):
 (a) pairwise, blind: one of ours + one reference of the same expression group, random side, neutral names, identical encode chain
     (ours: x264 crf 20 4:2:0 round-trip, then JPEG q90 4:2:0 for BOTH sides; refs come from YouTube game footage, ours is clean) ->
     "which is the shipped AAA / state-of-the-art real-time digital human?"  Score = how often OURS is picked (50 % = indistinguishable).
     CONTROL = the same question on reference-vs-reference pairs from different games: the noise band / position bias.
 (b) absolute scorecard per image (ours and references, in the same blind format), 1-10 per axis with AAA anchors + 3 tells tagged by area.
 (c) motion: 8-frame contact strips of 3-4 s clips, ours vs reference clips.
Layout of the reference set: T/judge/ref/{closeup,bust}/*.png + manifest.json (your own captures, see README.md).
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, os, random, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

T = os.environ.get('JT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work'))
J = f'{T}/judge'
REF = f'{J}/ref'
HERE = os.path.dirname(os.path.abspath(__file__))
CLAUDE = os.environ.get('JUDGE_CLAUDE', '/usr/local/bin/claude')
LOGF = f'{T}/logs/judge.md'
SHOTS = {s['id']: s for s in json.load(open(f'{HERE}/shots_aaa.json'))['shots']}
LOCK = threading.Lock()
TAGS = ['asset-face', 'hair', 'skin', 'eyes', 'mouth', 'cloth/neck', 'lighting', 'motion', 'render']
AXES = {'closeup': ['skin', 'eyes', 'mouth_teeth', 'hair', 'lighting', 'facial_form', 'overall'],
        'bust': ['skin', 'eyes', 'mouth_teeth', 'hair', 'lighting', 'facial_form', 'clothing_neck', 'overall'],
        'clips': ['blink', 'idle_micro_motion', 'speech_articulation', 'head_neck_motion', 'eye_life', 'overall'],
        'session': ['skin', 'eyes', 'mouth_teeth', 'hair', 'lighting', 'facial_form', 'clothing_neck', 'overall']}
GROUPS = ['neutral', 'expression', 'talking']
GROUP_MIX = {'neutral': .2, 'expression': .5, 'talking': .3}


def log(m):
    line = f"- {time.strftime('%F %T')} {m}"
    print(line, flush=True)
    with LOCK:
        os.makedirs(os.path.dirname(LOGF), exist_ok=True)
        with open(LOGF, 'a') as f:
            f.write(line + '\n')


# ------------------------------------------------------------------ claude
def call_claude(prompt, sch, model='sonnet', system=None, cwd=None, timeout=600, tools='Read'):
    system = system or 'Answer only with the requested JSON.'
    cmd = [CLAUDE, '-p', '--model', model, '--tools', tools, '--permission-mode', 'dontAsk', '--no-session-persistence',
           '--setting-sources', '', '--system-prompt', system, '--output-format', 'json', '--json-schema', json.dumps(sch)]
    last = None
    for attempt in (1, 2, 3):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd or '/tmp', timeout=timeout)
            d = json.loads(r.stdout)
            out = d.get('structured_output')
            if out is None:
                out = json.loads(re.search(r'\{.*\}', d.get('result') or '', re.S).group(0))
            return out
        except Exception as e:
            last = f'{type(e).__name__}: {e} {(r.stdout[-200:] if "r" in dir() else "")}'
            time.sleep(4 * attempt)
    raise RuntimeError(last)


# ------------------------------------------------------------------ image chain
def h264_roundtrip(im: Image.Image, crf=20, tmp='/tmp'):
    """ours is a clean render; the references went through YouTube's encoder: give ours an x264 pass (4:2:0, medium) too."""
    key = hashlib.md5(im.tobytes() + str(crf).encode()).hexdigest()[:14]
    w, h = im.size
    src, mp4, out = f'{tmp}/rt_{key}.png', f'{tmp}/rt_{key}.mp4', f'{tmp}/rt_{key}_o.png'
    if not os.path.exists(out):
        os.makedirs(tmp, exist_ok=True)
        im.convert('RGB').save(src)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24', '-t', '1', '-i', src, '-vf', 'scale=in_range=full:out_range=tv,format=yuv420p',
                        '-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium', '-pix_fmt', 'yuv420p', mp4], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp4, '-frames:v', '1', out], check=True)
    return Image.open(out).convert('RGB')


def jpeg_save(im: Image.Image, path, q=90):
    im.convert('RGB').save(path, 'JPEG', quality=q, subsampling=2, optimize=False)   # no EXIF, no ICC, 4:2:0


def prep_ours(path, setname, crf, tmp):
    im = Image.open(path).convert('RGB')
    if setname != 'clips' and im.size != (1920, 1080):
        im = im.resize((1920, 1080), Image.LANCZOS)
    if setname == 'clips':
        return im
    if setname == 'closeup':
        im = im.crop((420, 0, 1500, 1080))                    # the central 1080x1080 of the 1920x1080 'face' frame
    return h264_roundtrip(im, crf, tmp) if crf else im


def prep_ref(path, setname):
    im = Image.open(path).convert('RGB')
    return im


def detail_crop(im: Image.Image, setname):  # (clips: none)
    """bust only: native-pixel 800x800 window on the head (both sides are framed with the head centred, crown at ~10 %)."""
    if setname != 'bust':
        return None
    return im.crop((560, 0, 1360, 800))


# ------------------------------------------------------------------ prompts
SYS_ART = ('You are a veteran real-time character artist and technical art director with 15 years in AAA game studios '
           '(you have shipped hero characters on Unreal and proprietary engines). You judge only what is visible. '
           'You only use the Read tool on the image paths you are given. Be strict, concrete and specific: name exact visible features. '
           'Answer only with the requested JSON.')
Q_PAIR = ('One of these two is a capture from a shipped AAA game / state-of-the-art real-time digital human (current-generation hero character); the other is from '
          'a lower-budget real-time renderer. Which is the AAA one?')
TAG_HELP = ('asset-face = facial shape/proportions/anatomy/expression geometry; hair = hair/brows/lashes; skin = skin shading, pores, subsurface, texture; eyes = eyes/lids/tear line; '
            'mouth = lips/teeth/tongue/mouth interior; cloth/neck = clothing, collar, neck, shoulders; lighting = lighting/shadows/AO/exposure/colour; '
            'motion = animation/blink/speech; render = anti-aliasing, noise, depth of field, post, resolution.')


def pair_prompt(paths, setname, detail):
    if setname == 'clips':
        return (f'Two images: Read {paths[0]} (clip A) and {paths[1]} (clip B). Each is a contact strip of 8 frames of a 3.5 second clip of a talking game character, left to right, top row then bottom row (0.5 s apart). '
                'Ignore image quality, framing and compression: judge only the ANIMATION (blink timing and shape, idle micro-motion, speech articulation incl. lip closure, head/neck motion, eye life/saccades, expression changes). '
                'One is from a shipped AAA game / state-of-the-art real-time digital human, the other from a lower-budget real-time renderer. Which is the AAA one?\n'
                'Answer JSON: aaa = "A" or "B"; confidence 1-10; tells = 2-4 items {image: "A"|"B" (the less AAA one), tag (use "motion" unless clearly another), text}.')
    ex = f' (each image also has a 1:1 detail crop of the head: {detail})' if detail else ''
    return (f'Two images{ex}: Read {paths[0]} (image A) and {paths[1]} (image B)' + (f', and the detail crops {detail[0]} (of A) and {detail[1]} (of B)' if detail else '') + '. '
            'They are same-size frames, re-encoded with the same chain; ignore framing, backdrop and compression and judge the character rendering quality.\n'
            f'{Q_PAIR}\nAnswer JSON: aaa = "A" or "B" (forced choice); confidence 1-10; '
            f'tells = 2-4 items {{image: "A"|"B", tag, text}} where image is the one that looks LESS AAA (the lower-budget one) and text names the exact visible feature that gives it away; '
            f'tag is one of {TAGS} ({TAG_HELP}).')


PAIR_SCHEMA = {'type': 'object', 'properties': {
    'aaa': {'type': 'string', 'enum': ['A', 'B']}, 'confidence': {'type': 'integer'},
    'tells': {'type': 'array', 'items': {'type': 'object', 'properties': {'image': {'type': 'string'}, 'tag': {'type': 'string', 'enum': TAGS}, 'text': {'type': 'string'}},
                                        'required': ['image', 'tag', 'text']}}}, 'required': ['aaa', 'confidence', 'tells']}

ANCHORS = ('Scale anchors (10 = a shipped current-generation AAA hero close-up at its best: Hellblade II Senua, The Last of Us Part II Ellie, Death Stranding 2, a MetaHuman showcase; '
           '8 = very good current-gen AAA hero (Horizon Forbidden West Aloy, Cyberpunk 2077 main cast); 6 = good last-gen hero (Uncharted 4 / Horizon Zero Dawn era); '
           '4 = typical mid-budget / UE4-default quality; 2 = PS3-era or cartoon-simplified; 1 = untextured placeholder). Use the full range; do not cluster at 7-8.')


def score_prompt(path, setname, detail):
    if setname == 'clips':
        return (f'Read {path}: a contact strip of 8 frames of a 3.5 s clip of a talking real-time game character (left to right, top row then bottom row, 0.5 s apart). Ignore image quality; judge the ANIMATION only. '
                f'Score 1-10: {AXES["clips"]} (blink = timing/shape/lid follow; idle_micro_motion = breathing, subtle head/face life; speech_articulation = mouth shapes incl. closures; head_neck_motion = natural head/neck/shoulder movement; eye_life = saccades, gaze, vergence). '
                f'{ANCHORS}\nAlso generation, aaa_realtime, and exactly 3 tells {{tag, text}} naming what looks mechanical/unnatural (tag usually "motion").')
    ax = AXES[setname]
    return (f'Read {path}' + (f' and its 1:1 detail crop {detail}' if detail else '') + '. It is a frame of a real-time rendered human character (re-encoded video frame; ignore compression and framing). '
            f'Score it on each axis 1-10: {ax}. skin = shading, pores, subsurface, micro-detail; eyes = wetness, depth, catchlights, lid/tear-line detail; mouth_teeth = lips, teeth, mouth interior (if closed, judge the lips); '
            'hair = strand realism, shading, hairline, brows, lashes; lighting = shadows, AO, light quality and integration; facial_form = anatomy and proportions of a believable human face; '
            'clothing_neck = fabric, folds, neck/collar/shoulder deformation (if not visible, give your best estimate from what is visible); overall = how much it reads as a current-generation AAA game character.\n'
            f'{ANCHORS}\n'
            'Also: generation = which console/engine generation this render most resembles; aaa_realtime = true only if you would believe it is a shipped AAA current-gen game / digital-human demo; '
            f'tells = exactly 3 items {{tag, text}}: the three most concrete visible giveaways that it is NOT a top-tier AAA real-time character (tag in {TAGS}; {TAG_HELP}). '
            'If it really is AAA, name the three weakest features anyway.')


def score_schema(setname):
    ax = AXES[setname]
    return {'type': 'object', 'properties': {
        'scores': {'type': 'object', 'properties': {a: {'type': 'integer'} for a in ax}, 'required': ax},
        'generation': {'type': 'string', 'enum': ['ps3_era_or_simpler', 'ps4_era', 'current_gen_mid_budget', 'current_gen_aaa', 'offline_film_cgi', 'photo_or_live_action']},
        'aaa_realtime': {'type': 'boolean'},
        'tells': {'type': 'array', 'items': {'type': 'object', 'properties': {'tag': {'type': 'string', 'enum': TAGS}, 'text': {'type': 'string'}}, 'required': ['tag', 'text']}}},
        'required': ['scores', 'generation', 'aaa_realtime', 'tells']}


# ------------------------------------------------------------------ reference set
def load_manifest():
    p = f'{REF}/manifest.json'
    if not os.path.exists(p):
        raise SystemExit(f'no reference set: put your own reference captures there (see README.md; missing {p})')
    return json.load(open(p))['items']


def ours_items(d, setname):
    out = []
    for p in sorted(glob.glob(f'{d}/{setname}/*.png') + glob.glob(f'{d}/{setname}/*.jpg')):
        sid = os.path.splitext(os.path.basename(p))[0]
        g = 'talking' if setname == 'clips' else SHOTS.get(sid, {}).get('group', 'expression')
        out.append({'id': f'o_{setname}_{sid}', 'shot': sid, 'group': g, 'path': p})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------ jobs
def run_pair(job, outdir, model):
    rawp = f'{outdir}/raw/{job["id"]}.json'
    if os.path.exists(rawp):
        return json.load(open(rawp))
    ind = f'{outdir}/inputs/{job["id"]}'
    os.makedirs(ind, exist_ok=True)
    nm = [hashlib.md5((job['id'] + x + 'k').encode()).hexdigest()[:10] for x in 'ab']
    paths, det = [], []
    for n_, im in zip(nm, (job['A'], job['B'])):
        p = f'{ind}/{n_}.jpg'; jpeg_save(im, p); paths.append(p)
        dc = detail_crop(im, job['set'])
        if dc is not None:
            q = f'{ind}/{n_}_d.jpg'; jpeg_save(dc, q); det.append(q)
    out = call_claude(pair_prompt(paths, job['set'], det or None), PAIR_SCHEMA, model, SYS_ART, cwd=ind)
    res = {k: job[k] for k in ('id', 'kind', 'set', 'group', 'a_id', 'b_id', 'a_is') if k in job}
    res.update(out)
    os.makedirs(f'{outdir}/raw', exist_ok=True)
    json.dump(res, open(rawp, 'w'), indent=1)
    return res


def run_score(job, outdir, model, cache=None):
    if cache is not None and job['id'] in cache:
        return cache[job['id']]
    ind = f'{outdir}/inputs/{job["id"]}'
    os.makedirs(ind, exist_ok=True)
    n_ = hashlib.md5((job['id'] + 'sc').encode()).hexdigest()[:10]
    p = f'{ind}/{n_}.jpg'; jpeg_save(job['img'], p)
    dc = detail_crop(job['img'], job['set'])
    dp = None
    if dc is not None:
        dp = f'{ind}/{n_}_d.jpg'; jpeg_save(dc, dp)
    out = call_claude(score_prompt(p, job['set'], dp), score_schema(job['set']), model, SYS_ART, cwd=ind)
    res = {'id': job['id'], 'set': job['set'], 'group': job.get('group'), 'who': job['who'], **out}
    return res


def plan_pairs(setname, ours, refs, n, seed, kind='ours', cross_game=False):
    """n trials: group quotas; ours<->ref of the same group; sides balanced.  kind=control: two refs (different games)."""
    rng = random.Random(f'{seed}-{setname}-{kind}')
    quota = {g: max(1, round(n * w)) for g, w in GROUP_MIX.items()}
    while sum(quota.values()) > n:
        quota['expression'] -= 1
    while sum(quota.values()) < n:
        quota['expression'] += 1
    by_o = {g: [o for o in ours if o['group'] == g] for g in GROUPS}
    by_r = {g: [r for r in refs if r['group'] == g] for g in GROUPS}
    trials = []
    for g in GROUPS:
        oo = by_o[g] or ours
        rr = by_r[g] or refs
        po, pr = [], []
        for i in range(quota[g]):
            if not pr: pr = list(rr); rng.shuffle(pr)
            if kind != 'control':
                if not po: po = list(oo); rng.shuffle(po)
                a = po.pop()
            if kind == 'control':
                a = pr.pop() if pr else rng.choice(rr)
                cand = [r for r in rr if r['key'] != a['key'] and r['id'] != a['id']] or [r for r in rr if r['id'] != a['id']]
                b = rng.choice(cand)
                trials.append({'group': g, 'x': a, 'y': b})
            else:
                trials.append({'group': g, 'x': a, 'y': pr.pop()})
    rng.shuffle(trials)
    for i, t in enumerate(trials):
        t['x_first'] = (i % 2 == 0)
    rng.shuffle(trials)
    return trials


def make_pair_jobs(setname, ours, refs, n, seed, crf, tmp, kind):
    jobs = []
    for i, t in enumerate(plan_pairs(setname, ours, refs, n, seed, kind)):
        x, y = t['x'], t['y']
        ix = prep_ref(x['path'], setname) if kind == 'control' else prep_ours(x['path'], setname, crf, tmp)
        iy = prep_ref(y['path'], setname)
        A, B, a_id, b_id = (ix, iy, x['id'], y['id']) if t['x_first'] else (iy, ix, y['id'], x['id'])
        jobs.append({'id': f'{kind}_{setname}_{i:02d}_{x["id"][-18:]}_{y["id"][-6:]}', 'kind': kind, 'set': setname, 'group': t['group'], 'A': A, 'B': B,
                     'a_id': a_id, 'b_id': b_id, 'a_is': 'x' if t['x_first'] else 'y'})
    return jobs


def pool_run(fn, items, workers, label):
    out, done = [], [0]
    def wrap(it):
        try:
            r = fn(it)
        except Exception as e:
            log(f'{label} {it.get("id")} failed: {e}')
            return None
        done[0] += 1
        if done[0] % 10 == 0:
            log(f'{label}: {done[0]}/{len(items)}')
        return r
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(wrap, items))
    return [o for o in out if o]


# ------------------------------------------------------------------ control + reference scorecards (cached)
def control_results(setname, refs, n, seed, crf, model, workers, refresh=False):
    cp = f'{REF}/control/{setname}.json'
    os.makedirs(f'{REF}/control', exist_ok=True)
    if os.path.exists(cp) and not refresh:
        c = json.load(open(cp))
        if c.get('n') == n and c.get('model') == model:
            return c['results']
    outdir = f'{REF}/control/work_{setname}'
    jobs = make_pair_jobs(setname, [], [r for r in refs if r['set'] == setname], n, seed, crf, f'{outdir}/tmp', 'control')
    res = pool_run(lambda j: run_pair(j, outdir, model), jobs, workers, f'control-{setname}')
    json.dump({'n': n, 'model': model, 'results': res}, open(cp, 'w'))
    return res


def refscore_results(setname, refs, model, workers, refresh=False):
    cp = f'{REF}/refscores_{setname}.json'
    cache = {} if refresh or not os.path.exists(cp) else {r['id']: r for r in json.load(open(cp))}
    todo = [r for r in refs if r['set'] == setname and r['id'] not in cache]
    outdir = f'{REF}/refscore_work_{setname}'
    jobs = [{'id': r['id'], 'set': setname, 'group': r['group'], 'who': 'ref', 'img': prep_ref(r['path'], setname)} for r in todo]
    new = pool_run(lambda j: run_score(j, outdir, model), jobs, workers, f'refscore-{setname}')
    for r in new:
        cache[r['id']] = r
    json.dump(list(cache.values()), open(cp, 'w'))
    return [cache[r['id']] for r in refs if r['set'] == setname and r['id'] in cache]


# ------------------------------------------------------------------ statistics
def pair_stats(res, who_x='ours'):
    """rate = share of trials in which the judge called OURS (x) the AAA one"""
    out = {}
    for g in ['all'] + GROUPS:
        rs = [r for r in res if g == 'all' or r['group'] == g]
        if not rs:
            continue
        k = sum(1 for r in rs if (r['aaa'] == 'A') == (r['a_is'] == 'x'))
        n = len(rs)
        pos = sum(1 for r in rs if r['aaa'] == 'A') / n
        out[g] = {'n': n, 'x_picked': k, 'rate': round(k / n, 3), 'ci': [round(v, 3) for v in wilson(k, n)], 'picks_A': round(pos, 3),
                  'mean_conf': round(sum(r['confidence'] for r in rs) / n, 2)}
    return out


def score_stats(rs, axes):
    out = {}
    for g in ['all'] + GROUPS:
        xs = [r for r in rs if g == 'all' or r.get('group') == g]
        if not xs:
            continue
        d = {a: round(sum(r['scores'][a] for r in xs) / len(xs), 2) for a in axes if all(a in r['scores'] for r in xs)}
        d['n'] = len(xs)
        d['aaa_realtime_share'] = round(sum(1 for r in xs if r['aaa_realtime']) / len(xs), 2)
        out[g] = d
    return out


def bar_check(pair, ctrl, sc_o, sc_r, axes):
    """BAR: pick-rate >= 35 % AND every scorecard axis within 1 point of the reference mean"""
    rate = pair.get('all', {}).get('rate')
    gaps = {a: round(sc_o['all'][a] - sc_r['all'][a], 2) for a in axes if a in sc_o.get('all', {}) and a in sc_r.get('all', {})}
    ok_pick = rate is not None and rate >= 0.35
    ok_sc = all(g >= -1.0 for g in gaps.values()) if gaps else False
    return {'pick_rate': rate, 'pick_ok': ok_pick, 'axis_gaps': gaps, 'axes_ok': ok_sc, 'pass': bool(ok_pick and ok_sc)}


# ------------------------------------------------------------------ report
def frame_path(ours, fid):
    m = re.match(r'o_(closeup|bust|session)_(.+)$', fid)
    if not m:
        return fid
    st, sid = m.groups()
    for ext in ('png', 'jpg'):
        pth = f'{ours}/{st}/{sid}.{ext}'
        if os.path.exists(pth):
            return os.path.abspath(pth)
    return fid


def collect_tells(pair_res, sc_res, ours_ids):
    """tells that blame OURS: pairwise (image == ours' side) and scorecard tells"""
    out = []
    for r in pair_res:
        ours_side = 'A' if r['a_is'] == 'x' else 'B'
        oid = r['a_id'] if ours_side == 'A' else r['b_id']
        for t in r['tells']:
            if t['image'] == ours_side:
                out.append({'frame': oid, 'tag': t['tag'], 'text': t['text'], 'src': 'pair'})
    for r in sc_res:
        if r['who'] == 'ours':
            for t in r['tells']:
                out.append({'frame': r['id'], 'tag': t['tag'], 'text': t['text'], 'src': 'score'})
    return out


CONSOL_SCHEMA = {'type': 'object', 'properties': {'flaws': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'rank': {'type': 'integer'}, 'tag': {'type': 'string', 'enum': TAGS}, 'flaw': {'type': 'string'}, 'severity': {'type': 'integer'},
    'frames': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['rank', 'tag', 'flaw', 'severity', 'frames']}}}, 'required': ['flaws']}


def consolidate(tells, model):
    if not tells:
        return []
    by_tag = {}
    for t in tells:
        by_tag.setdefault(t['tag'], []).append(t)
    lines = []
    for tag, ts in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        lines.append(f'## {tag}: {len(ts)} mentions')
        for t in ts[:40]:
            lines.append(f'- [{t["frame"]}] {t["text"]}')
    pr = ('Below are blind-judge "tells" (why a rendered game character is not AAA real-time quality) collected over many frames of ONE character, grouped by area tag, with the frame id each came from. '
          'Consolidate into at most 12 distinct, concrete flaws ranked by how much they hurt (frequency x how glaring). Each: rank, tag (one of ' + str(TAGS) + '), a one-sentence concrete flaw, '
          'severity 1-10, and up to 5 frame ids that show it best (copy ids exactly). Merge duplicates; do not invent flaws that are not in the list.\n\n' + '\n'.join(lines))
    r = call_claude(pr, CONSOL_SCHEMA, model, 'Answer only with the requested JSON.', tools='Read')
    return sorted(r['flaws'], key=lambda f: f['rank'])


def fmt_stats_row(name, p, c):
    if not p:
        return f'| {name} | - | - | - |'
    ci = f'{p["ci"][0]*100:.0f}-{p["ci"][1]*100:.0f}'
    cc = f'{c["rate"]*100:.0f}% ({c["ci"][0]*100:.0f}-{c["ci"][1]*100:.0f}), n={c["n"]}' if c else '-'
    return f'| {name} | {p["rate"]*100:.0f}% ({p["x_picked"]}/{p["n"]}) | {ci} | {cc} |'


def write_report(path, name, summary, flaws, args):
    L = [f'# AAA judge round: {name}', '', f'{time.strftime("%F %T")}  ours=`{args.ours}`  model={args.model}  n={args.n} per set  codec crf={args.crf}', '']
    L += ['BAR: pick-rate for ours >= 35 % (50 % = indistinguishable from AAA; control = ref-vs-ref noise) and every scorecard axis within 1 point of the AAA reference mean.', '']
    for s, d in summary.items():
        L += [f'## {s}', '', f'**BAR: {"PASS" if d["bar"]["pass"] else "FAIL"}**  (pick {d["bar"]["pick_rate"]}, pick_ok={d["bar"]["pick_ok"]}, axes_ok={d["bar"]["axes_ok"]})', '',
              '### Blind pairwise: how often ours is picked as the AAA one', '', '| group | ours picked | 95% CI % | control (ref vs ref, same side rate) |', '|---|---|---|---|']
        for g in ['all'] + GROUPS:
            if g in d['pair']:
                L.append(fmt_stats_row(g, d['pair'][g], d['control'].get(g)))
        lw = d.get('pair_lowhalf', {}).get('all')
        if lw:
            L += ['', f'Ladder (ours vs the weaker half of the references, same question): {lw["rate"]*100:.0f}% ({lw["x_picked"]}/{lw["n"]}, CI {lw["ci"][0]*100:.0f}-{lw["ci"][1]*100:.0f})']
        L += ['', '### Absolute scorecard (1-10), ours vs AAA reference mean', '', '| axis | ours | ref mean | gap |', '|---|---|---|---|']
        for a in AXES[s]:
            o, r = d['score_ours'].get('all', {}).get(a), d['score_ref'].get('all', {}).get(a)
            if o is not None and r is not None:
                L.append(f'| {a} | {o} | {r} | {o - r:+.2f} |')
        L += ['', f'ours judged "believable shipped AAA real-time": {d["score_ours"].get("all", {}).get("aaa_realtime_share")}  vs reference {d["score_ref"].get("all", {}).get("aaa_realtime_share")}', '']
        for g in GROUPS:
            if g in d['score_ours']:
                o = d['score_ours'][g]; r = d['score_ref'].get(g, {})
                L.append(f'- {g}: ours overall {o.get("overall")} vs ref {r.get("overall")} (n={o["n"]})')
        L.append('')
    if 'session' in summary.get('_extra', {}):
        pass
    L += ['## Ranked flaws (aggregated tells)', '']
    for f in flaws:
        L.append(f'{f["rank"]}. **[{f["tag"]}]** (sev {f["severity"]}) {f["flaw"]}  frames: ' + ', '.join(f.get('paths') or f['frames']))
    open(path, 'w').write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ round
def cmd_round(a):
    outdir = f'{J}/rounds/{a.name}'
    os.makedirs(outdir, exist_ok=True)
    tmp = f'{outdir}/tmp'
    manifest = load_manifest()
    for m in manifest:
        m['path'] = m['file']; m['key'] = m.get('key') or m['id']
    sets = [s for s in a.sets.split(',') if s in ('closeup', 'bust', 'clips')]
    t0 = time.time()
    log(f'round {a.name}: sets={sets} n={a.n} jobs={a.jobs} model={a.model}')
    summary, all_pair, all_score, all_ours_sc = {}, [], [], []
    for s in sets:
        refs = [m for m in manifest if m['set'] == s]
        if not refs:
            log(f'{s}: no references yet'); continue
        ours = ours_items(a.ours, s)
        if not ours:
            log(f'{s}: no ours images in {a.ours}/{s}')
            continue
        # (a) pairwise + control, (b) scorecards -- one pool so everything runs 8 at a time
        ctrl = control_results(s, manifest, a.control_n, a.seed, a.crf, a.model, a.jobs, a.refresh_control)
        pjobs = make_pair_jobs(s, ours, refs, a.n, a.seed, a.crf, tmp, 'ours')
        sjobs = [{'id': o['id'], 'set': s, 'group': o['group'], 'who': 'ours', 'img': prep_ours(o['path'], s, a.crf, tmp)} for o in ours]
        ref_sc = refscore_results(s, manifest, a.model, a.jobs, a.refresh_control)
        log(f'{s}: {len(pjobs)} pair calls, {len(sjobs)} score calls')
        # ladder: ours vs the WEAKER half of the references (by their own scorecard 'overall') -- moves before parity is reached
        ov = {r['id']: r['scores']['overall'] for r in ref_sc}
        med = sorted(ov.values())[len(ov) // 2] if ov else 0
        low_refs = [r for r in refs if ov.get(r['id'], 99) <= med]
        ljobs = make_pair_jobs(s, ours, low_refs, a.n, a.seed, a.crf, tmp, 'ourslow') if low_refs else []
        pres = pool_run(lambda j: run_pair(j, outdir, a.model), pjobs + ljobs, a.jobs, f'pair-{s}')
        lres = [r for r in pres if r['kind'] == 'ourslow']
        pres = [r for r in pres if r['kind'] == 'ours']
        sres = pool_run(lambda j: run_score(j, outdir, a.model), sjobs, a.jobs, f'score-{s}')
        for r in sres:
            json.dump(r, open(f'{outdir}/raw/score_{r["id"]}.json', 'w')) if os.path.isdir(f'{outdir}/raw') else None
        pair = pair_stats(pres)
        ctl = pair_stats(ctrl)
        so, sr = score_stats(sres, AXES[s]), score_stats(ref_sc, AXES[s])
        summary[s] = {'pair_lowhalf': pair_stats(lres), 'pair': pair, 'control': ctl, 'score_ours': so, 'score_ref': sr, 'bar': bar_check(pair, ctl, so, sr, AXES[s]),
                      'n_ref': len(refs), 'n_ours': len(ours)}
        all_pair += pres + lres; all_score += sres
    # session frames: scorecard only
    sess = sorted(glob.glob(f'{a.ours}/session/*.png'))
    if sess:
        sjobs = []
        for p in sess:
            im = Image.open(p).convert('RGB')
            sjobs.append({'id': 'o_session_' + os.path.splitext(os.path.basename(p))[0], 'set': 'session', 'group': 'expression', 'who': 'ours', 'img': im})
        sres = pool_run(lambda j: run_score(j, outdir, a.model), sjobs, a.jobs, 'score-session')
        summary['_session'] = score_stats(sres, AXES['session'])
        all_score += sres
    tells = collect_tells(all_pair, all_score, None)
    flaws = consolidate(tells, a.model)
    for f in flaws:
        f['paths'] = [frame_path(a.ours, x) for x in f['frames']]
    json.dump({'summary': summary, 'pairs': all_pair, 'scores': all_score, 'seconds': round(time.time() - t0)}, open(f'{outdir}/results.json', 'w'), indent=1)
    json.dump({'tells': tells, 'flaws': flaws}, open(f'{outdir}/tells.json', 'w'), indent=1)
    write_report(f'{outdir}/report.md', a.name, {k: v for k, v in summary.items() if not k.startswith('_')}, flaws, a)
    if '_session' in summary:
        with open(f'{outdir}/report.md', 'a') as f:
            f.write('\n## Real recorded session frames (scorecard only, avatar region of 1280x720 recordings)\n\n' + json.dumps(summary['_session']) + '\n')
    log(f'round {a.name} done in {time.time()-t0:.0f}s -> {outdir}/report.md')


def cmd_strip(a):
    """mp4 window -> 8-frame 4x2 contact strip (1920x960), square crop x,y,w,h (native px) -> DIR/clips/NAME.png"""
    x, y, w, h = [int(v) for v in a.crop.split(',')]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cells = []
    for j in range(8):
        t = a.t0 + j * (a.dur / 8)
        tmpf = f'/tmp/strip_{os.getpid()}_{j}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.3f}', '-i', a.mp4, '-frames:v', '1', tmpf], check=True)
        cells.append(Image.open(tmpf).convert('RGB').crop((x, y, x + w, y + h)).resize((480, 480), Image.LANCZOS))
        os.remove(tmpf)
    st = Image.new('RGB', (1920, 960))
    for j, c in enumerate(cells):
        st.paste(c, ((j % 4) * 480, (j // 4) * 480))
    st.save(a.out)
    print(a.out)


def cmd_control(a):
    manifest = load_manifest()
    for m in manifest:
        m['path'] = m['file']; m['key'] = m.get('key') or m['id']
    for s in ('closeup', 'bust', 'clips'):
        if not any(m['set'] == s for m in manifest):
            continue
        r = control_results(s, manifest, a.n, a.seed, a.crf, a.model, a.jobs, True)
        print(s, pair_stats(r))


def cmd_refscores(a):
    manifest = load_manifest()
    for m in manifest:
        m['path'] = m['file']
    for s in ('closeup', 'bust', 'clips'):
        if not any(m['set'] == s for m in manifest):
            continue
        r = refscore_results(s, manifest, a.model, a.jobs, a.refresh)
        print(s, score_stats(r, AXES[s]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    st = sp.add_parser('strip'); st.add_argument('--mp4', required=True); st.add_argument('--t0', type=float, default=0.0); st.add_argument('--dur', type=float, default=3.5)
    st.add_argument('--crop', required=True); st.add_argument('--out', required=True); st.set_defaults(f=cmd_strip)
    for nm in ('round', 'control', 'refscores'):
        s = sp.add_parser(nm)
        s.add_argument('--model', default='sonnet'); s.add_argument('--jobs', type=int, default=8); s.add_argument('--seed', default='4557')
        s.add_argument('--crf', type=int, default=20, help='x264 crf applied to ours (0 = off)')
        if nm == 'round':
            s.add_argument('--ours', required=True); s.add_argument('--name', required=True); s.add_argument('--n', type=int, default=20)
            s.add_argument('--sets', default='closeup,bust'); s.add_argument('--control-n', type=int, default=24)
            s.add_argument('--refresh-control', action='store_true'); s.set_defaults(f=cmd_round)
        elif nm == 'control':
            s.add_argument('--n', type=int, default=24); s.set_defaults(f=cmd_control)
        else:
            s.add_argument('--refresh', action='store_true'); s.set_defaults(f=cmd_refscores)
    a = ap.parse_args()
    a.f(a)
