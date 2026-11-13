#!/usr/bin/env python3
"""Planner scorer: one command from a planner config to rendered clips and a calibrated judge score.

  tools/plan_eval.py --set <items.jsonl> --config <name> --out <dir> [--root <worktree>] [--jobs 8] [--limit N] [--ids a,b]
                     [--no-judge] [--judge <name>] [--max-usd 2] [--force-stage plan|sim|render|judge] [--plan-jobs 3]
  tools/plan_eval.py --out <dir> --report                     (re-print / re-write summary.json + summary.md)

--root = the source tree (app/ + renderer/) that is scored: default this worktree; the "before" tree and any candidate
branch are scored by the same scorer (run.json records the root and its git sha).
Per item, under <out>/<id>/, each stage cached on disk and skipped when present (a killed run resumes where it stopped):
  1 plan    plan.json    the LIVE ROUTE: first what the page does with the request (node tools/plan_sim.cjs --route: the real
                         app/voicetools.js do_action on the real motion.js, in the row's `scene` and in the state the `setup`
                         left; description = words = the utterance): a decline / the page library / scene steps end there
                         (source "page", the engine commands it ran are replayed); a call that reaches the performer goes to
                         app/planner.py Planner(<config>, plan cache OFF, library ON), with the body state the page sends
                         (+ scene, previous_request). Every pose with the ms its batch arrived; the done event's fields kept
                         verbatim (new planner fields need no change here). A `setup` is planned first (unscored).
                         Character: the config's (muse), or for a row whose character_hint names a part it lacks (a tail)
                         the first character that has it (MANIFEST traits); none -> the row is skipped.
  2 sim     sim.json     node tools/plan_sim.cjs: played on app/motion.js as the page would -> rig.jsonl (trimmed) + timing
  3 render  render.json  tools/plan_render.py: clip.mp4 + strip_NN.jpg (+ face.mp4 / face_strip_NN.jpg for face_closeup)
  4 judge   judge.<name>.json   tools/plan_judge.py (the judge + pass threshold picked by calibration: tools/plan_judge.json)
A failed plan, a plan that moved nothing, or a refusal is a failure (score 1) with no render and no judge call — except on
`absent_decline` rows, where a decline is the wanted outcome: the clip of what the body did is judged, and the judge is told the
decline the character speaks. The scene: things that are not in the row's `scene` are not given to the page / engine and are
not drawn.
Configs: tools/plan_configs.json  name -> {"planner": {<Planner cfg overrides, passed through untouched>}, "character": "muse"}.
Cost: --max-usd caps the vendor cost (plans + judge) of one invocation; the per-run estimate is printed at the end.
Child Python (plan_render.py): $PYTHON, else this interpreter.
"""
import argparse, asyncio, json, os, shutil, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # --root replaces it (main)
PY = os.environ.get('PYTHON') or sys.executable
sys.path[:0] = [str(HERE)]
PART_OF_HINT = {'needs_tail': 'tail', 'needs_ears': 'ears', 'needs_fingers': None, 'needs_paws': 'arms', 'needs_arms': 'arms'}
CHAR_ORDER = ['muse', 'fox', 'monkey', 'bear', 'rabbit', 'hippo']


def git_sha(root):
    try:
        sha = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--', 'app', 'renderer'], capture_output=True, text=True, timeout=20).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:  # noqa: BLE001
        return None


def pick_character(it, default, traits):
    """the config's character, or (character_hint names a part it lacks) the first one that has the part; None = skip"""
    if it.get('character') and it['character'] in traits:   # a field row is replayed on the session's own character
        return it['character']
    need = PART_OF_HINT.get(str(it.get('character_hint') or 'any'))
    if not need or (traits.get(default) or {}).get(need, True):
        return default
    for c in CHAR_ORDER + sorted(traits):
        if (traits.get(c) or {}).get(need):
            return c
    return None
SETUP_MID = ('hold', 'continue')        # subcategories whose request comes WHILE the setup plays; the others after it


def jdump(o, p):
    tmp = str(p) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(o, f, indent=1, default=str)
    os.replace(tmp, p)


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return default


def text_of(it):
    return str(it.get('utterance') or it.get('text') or it.get('request') or it.get('description') or '').strip()


# ------------------------------------------------------------------------------------------------------ 1 plan
async def run_planner(planner, http, character, text, state, timeout=60.0):
    """-> plan record: poses stamped with the ms their batch arrived (our clock, from the call)"""
    import planner as kf
    t0 = time.monotonic()
    poses, done, err = [], None, None

    def add(ps, ms):
        for p in ps or []:
            q = kf.wire_pose(p) if isinstance(p, dict) and 'set' in p else dict(p)
            q['ms'] = ms
            poses.append(q)

    async def go():
        nonlocal done
        async for ev in planner.plan(character, text, state, http=http, words=text):
            ms = round((time.monotonic() - t0) * 1000)
            add(ev.get('poses'), ms)
            if ev.get('done'):
                done = dict(ev)
    try:
        await asyncio.wait_for(go(), timeout)
    except asyncio.TimeoutError:
        err = f'harness timeout {timeout:.0f}s'
    except Exception as e:  # noqa: BLE001
        err = f'harness: {type(e).__name__}: {str(e)[:200]}'
    rec = {'text': text, 'character': character}
    if done is None:
        rec.update({'ok': False, 'source': 'error', 'error': err or 'no done event', 'ms_done': round((time.monotonic() - t0) * 1000), 'usd': 0.0, 'engine': None})
    else:
        allp = done.pop('all_poses', None) or []
        done.pop('poses', None)
        if not poses and allp:
            add(allp, done.get('ms_done') or 0)
        rec.update(done)                # every field of the done event, as is (selfcheck, retried, attempts, model ...)
        rec['n_all_poses'] = len(allp)
        rec['harness_ms_done'] = round((time.monotonic() - t0) * 1000)
    rec['poses'] = poses
    return rec


def route_probe(text, character, scene, setup_rec, workdir):
    """what the page does with the request (plan_sim.cjs --route) -> {route: planner|hub|page, status, ok, cmds, state, caption}"""
    p = Path(workdir) / '_route_probe.json'
    body = {'id': 'probe', 'text': text, 'character': character, 'scene': scene, 'source': 'none', 'poses': []}
    if setup_rec is not None:
        body['setup'] = setup_rec
    jdump(body, p)
    r = subprocess.run(['node', str(HERE / 'plan_sim.cjs'), str(p), str(workdir), '--route', '--root', str(ROOT)], capture_output=True, text=True, timeout=120)
    p.unlink(missing_ok=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {'route': 'hub', 'status': 'route_error', 'ok': False, 'state': None, 'cmds': [], 'error': (r.stderr or r.stdout)[-300:]}


async def plan_stage(items, cfg, cfg_name, out, plan_jobs, budget):
    import aiohttp
    import planner as kf
    pcfg = dict(cfg.get('planner') or {})
    pcfg['cache'] = False
    default_char = cfg.get('character', 'muse')
    force_planner = str(cfg.get('route') or '').lower() == 'planner'   # search mode: every row to the planner (no page ending)
    man = jload(ROOT / 'app/planner_prompts/MANIFEST.json', {}) or {}
    man = man.get('characters') if isinstance(man.get('characters'), dict) else man
    traits = {k: v['traits'] for k, v in man.items() if isinstance(v, dict) and isinstance(v.get('traits'), dict)}
    if default_char not in traits:
        print(f'!! no traits for {default_char} in MANIFEST.json: character_hint rows stay on it', flush=True)
    tmpc = tempfile.mkdtemp(prefix='plan_eval_cache_')
    logs = []
    planner = kf.Planner(pcfg, log=lambda *a: logs.append(' '.join(str(x) for x in a)), cache_dir=tmpc)
    sem = asyncio.Semaphore(plan_jobs)
    n = 0
    # a claude_door planner gets its warm pool first (the server prewarms at session open)
    if hasattr(planner, 'prewarm') and planner.primary()['route'] == 'claude_door':
        todo_chars = {pick_character(it, default_char, traits) for it in items if not (out / it['id'] / 'plan.json').exists()} - {None}
        if todo_chars:
            planner.prewarm(default_char, n=plan_jobs + 1)
            for c in sorted(todo_chars - {default_char}):
                planner.prewarm(c, n=1)
            await asyncio.sleep(12)

    async def one(it, http):
        nonlocal n
        d = out / it['id']
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'plan.json').exists():
            return
        async with sem:
            if budget.over():
                return
            scene = it.get('scene')
            if force_planner and it.get('subcategory') == 'absent_decline':
                jdump({'id': it['id'], 'config': cfg_name, 'category': it.get('category'), 'skipped': 'planner-only run: a decline is the page\'s job', 'poses': []}, d / 'plan.json')
                return
            character = pick_character(it, default_char, traits)
            if character is None:
                jdump({'id': it['id'], 'config': cfg_name, 'category': it.get('category'), 'skipped': f"no character has the part ({it.get('character_hint')})", 'poses': []}, d / 'plan.json')
                return
            if character != 'muse':
                scene = None                                 # the seated characters have no scene
            setup = None
            if it.get('setup'):
                when = 'mid' if it.get('subcategory') in SETUP_MID else 'end'
                setup = await run_planner(planner, http, character, str(it['setup']), {'scene': scene} if scene is not None else None)
                setup['when'] = when
                budget.add('plan', setup.get('usd'))
            probe = await asyncio.to_thread(route_probe, text_of(it), character, scene, setup, d)
            state = dict(probe.get('state') or {})
            if scene is not None:
                state['scene'] = scene
            if setup is not None:
                state['previous_request'] = str(it['setup'])
            if it.get('lvo_exclude'):
                state['lvo_exclude'] = list(it['lvo_exclude'])   # leave-vocabulary-out row (tools/lvo_map.py)
            if probe.get('route') == 'page' and not force_planner:
                # the page ended it: a decline, its own library, or scene steps. No server call; the commands are replayed.
                rec = {'text': text_of(it), 'character': character, 'done': True, 'ok': bool(probe.get('ok')), 'source': 'page', 'engine': [c['cmd'] for c in probe.get('cmds') or []],
                       'ms_first_pose': None, 'ms_done': 0, 'usd': 0.0, 'error': None if probe.get('ok') else str(probe.get('reason') or 'declined')[:400],
                       'decline': probe.get('caption'), 'poses': []}
            else:
                rec = await run_planner(planner, http, character, text_of(it), state or None)
                budget.add('plan', rec.get('usd'))
            rec.update({'id': it['id'], 'config': cfg_name, 'category': it.get('category'), 'scene': scene, 'character': character,
                        'route': {k: probe.get(k) for k in ('route', 'status', 'ok', 'reason', 'did', 'cmds', 'error') if probe.get(k) not in (None, [])}})
            if setup is not None:
                rec['setup'] = setup
            jdump(rec, d / 'plan.json')
            n += 1
            print(f"  plan {it['id']} {rec.get('source')} ok={rec.get('ok')} poses={len(rec['poses'])} first={rec.get('ms_first_pose')} done={rec.get('ms_done')} "
                  f"usd={float(rec.get('usd') or 0):.4f} {('ERR ' + str(rec.get('error'))[:80]) if rec.get('error') else ''}", flush=True)

    async with aiohttp.ClientSession() as http:
        await asyncio.gather(*[one(it, http) for it in items])
    if hasattr(kf, 'close_doors'):                       # the door's warm processes end with the stage
        await kf.close_doors()
        try:
            import claude_door
            await claude_door.door.close()
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(tmpc, ignore_errors=True)
    return n


class Budget:
    def __init__(self, cap):
        self.cap, self.usd = cap, {'plan': 0.0, 'judge': 0.0}

    def add(self, k, v):
        try:
            self.usd[k] += float(v or 0)
        except Exception:  # noqa: BLE001
            pass

    def total(self):
        return sum(self.usd.values())

    def over(self):
        return self.cap is not None and self.total() >= self.cap


# ------------------------------------------------------------------------------------------------------ 2-4
def is_refusal(plan):
    if not plan or plan.get('ok') is not False or plan.get('poses'):
        return False
    if plan.get('source') == 'page':
        return True
    e = str(plan.get('error') or '').lower()
    return bool(plan.get('refused') or plan.get('declined') or 'refus' in e or 'declin' in e or "can't do" in e or 'cannot do' in e or 'there is no' in e)


def plan_failed(plan, it=None):
    if plan is None:
        return 'no plan'
    if is_refusal(plan):
        # a refusal: the wanted outcome on an absent_decline row (judged, with the spoken decline); a failure elsewhere
        return None if (it or {}).get('subcategory') == 'absent_decline' else 'declined: ' + str(plan.get('error') or 'refused')[:160]
    if plan.get('ok') is False:
        return 'plan failed: ' + str(plan.get('error') or 'not ok')[:160]
    if not plan.get('poses') and not plan.get('engine'):
        return 'plan has no poses and no commands'
    return None


def auto_fail(d, why, judge_name):
    jdump({'score': 1, 'shows_request': False, 'natural': False, 'glitches': [], 'reason': why, 'judge': 'auto', 'auto': True, 'usd': 0.0}, d / f'judge.{judge_name}.json')


def item_stages(it, out, a, judge_cfg, budget):
    d = out / it['id']
    plan = jload(d / 'plan.json')
    if plan is None:
        return 'no plan'
    if plan.get('skipped'):
        return 'skipped'
    jn = judge_cfg['judge']
    jf = d / f'judge.{jn}.json'
    try:
        if not (d / 'sim.json').exists():
            r = subprocess.run(['node', str(HERE / 'plan_sim.cjs'), str(d / 'plan.json'), str(d), '--root', str(ROOT)], capture_output=True, text=True, timeout=300)
            if r.returncode or not (d / 'sim.json').exists():
                jdump({'error': (r.stderr or r.stdout)[-600:], 'moved': False, 'sim_failed': True}, d / 'sim.json')
        sim = jload(d / 'sim.json') or {}
        why = plan_failed(plan, it)
        declined = is_refusal(plan) and not why
        if not why and sim.get('sim_failed'):
            why = 'sim failed: ' + str(sim.get('error'))[-160:]
        if not why and not sim.get('moved') and not plan.get('setup') and not declined:
            why = 'nothing moved' + (' (the page answered "starting": ' + '; '.join(str(c.get('reason')) for c in (plan.get('route') or {}).get('cmds') or [] if not c.get('ok'))[:120] + ')' if plan.get('source') == 'page' else '')
        if why:
            if not jf.exists():
                auto_fail(d, why, jn)
            return 'auto-fail'
        rj = jload(d / 'render.json')
        face = bool(it.get('face_closeup'))
        if not (rj and rj.get('ok') and (not face or rj.get('face'))):
            cmd = [PY, str(HERE / 'plan_render.py'), str(d), '--threads', str(a.threads), '--root', str(ROOT), '--character', str(plan.get('character') or 'muse')] + (['--face'] if face else [])
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            rj = jload(d / 'render.json')
        if not (rj and rj.get('ok')):
            return 'render failed'
        if a.no_judge or jf.exists():
            return 'ok'
        import plan_judge
        if 'zoom' in plan_judge.JUDGES.get(jn, {}).get('media', ()) and 'zoom' not in rj:
            # the character close-up strips, cut from the clip that is already rendered
            subprocess.run([PY, str(HERE / 'plan_render.py'), str(d), '--zoom-only', '--character', str(plan.get('character') or 'muse')], capture_output=True, text=True, timeout=300)
        if budget.over():
            return 'budget'
        jit = dict(it, _character=str(plan.get('character') or 'muse'))
        if declined:
            jit['_caption'] = str(plan.get('decline') or plan.get('error') or 'I cannot do that')[:200]
        v = plan_judge.judge_item(jit, d, jn)
        budget.add('judge', v.get('usd'))
        if v.get('error') and v.get('score') is None:
            return 'judge error: ' + str(v['error'])[:120]
        jdump(v, jf)
        return 'ok'
    except Exception as e:  # noqa: BLE001
        return f'error: {type(e).__name__}: {str(e)[:160]}'


# ------------------------------------------------------------------------------------------------------ report
def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))])


def source_of(plan):
    if plan is None:
        return 'missing'
    if plan.get('skipped'):
        return 'skipped'
    if plan.get('ok') is False:
        return 'declined' if is_refusal(plan) else 'error'
    return str(plan.get('source') or 'planner')


def report(out, items, cfg_name, judge_cfg, quiet=False):
    jn, thr = judge_cfg['judge'], judge_cfg.get('pass_score', 4)
    strict = judge_cfg.get('strict_score', max(thr, 4))
    rows = []
    for it in items:
        d = out / it['id']
        plan, sim, j = jload(d / 'plan.json'), jload(d / 'sim.json') or {}, jload(d / f'judge.{jn}.json')
        if plan and plan.get('skipped'):
            continue
        rows.append({'id': it['id'], 'category': it.get('category'), 'subcategory': it.get('subcategory'), 'intent_in_dev': it.get('intent_in_dev'), 'character': (plan or {}).get('character'),
                     'face': bool(it.get('face_closeup')), 'source': source_of(plan), 'planned': plan is not None,
                     'score': j.get('score') if j else None, 'auto': bool(j and j.get('auto')), 'first_motion_ms': sim.get('first_motion_ms'),
                     'ms_first_pose': plan.get('ms_first_pose') if plan else None, 'ms_done': plan.get('ms_done') if plan else None,
                     'usd': float(plan.get('usd') or 0) + float((plan.get('setup') or {}).get('usd') or 0) if plan else 0.0, 'judge_usd': float(j.get('usd') or 0) if j else 0.0,
                     'model': plan.get('model') if plan else None, 'fallback_used': bool(plan and plan.get('fallback_used')),
                     'reason': (j or {}).get('reason'), 'text': text_of(it)})

    def agg(rs):
        sc = [r for r in rs if r['score'] is not None]
        src = {}
        for r in rs:
            src[r['source']] = src.get(r['source'], 0) + 1
        return {'n': len(rs), 'scored': len(sc), 'success_pct': round(100 * sum(r['score'] >= thr for r in sc) / len(sc), 1) if sc else None,
                'strict_pct': round(100 * sum(r['score'] >= strict for r in sc) / len(sc), 1) if sc else None,
                'mean_score': round(statistics.mean(r['score'] for r in sc), 2) if sc else None, 'sources': src,
                'first_motion_ms_p50': pct([r['first_motion_ms'] for r in rs], .5), 'first_motion_ms_p90': pct([r['first_motion_ms'] for r in rs], .9),
                'plan_ms_p50': pct([r['ms_done'] for r in rs if r['planned']], .5), 'plan_ms_p90': pct([r['ms_done'] for r in rs if r['planned']], .9),
                'planner_first_motion_ms_p50': pct([r['first_motion_ms'] for r in rs if r['source'] == 'planner'], .5),
                'planner_first_motion_ms_p90': pct([r['first_motion_ms'] for r in rs if r['source'] == 'planner'], .9),
                'usd_per_plan': round(sum(r['usd'] for r in rs) / max(1, sum(r['planned'] for r in rs)), 5), 'plan_usd': round(sum(r['usd'] for r in rs), 4),
                'judge_usd': round(sum(r['judge_usd'] for r in rs), 4), 'auto_fail': sum(r['auto'] for r in rs), 'fallback_used': sum(r['fallback_used'] for r in rs)}
    cats = sorted({r['category'] or '?' for r in rows})
    meta = jload(out / 'run.json', {})
    subs = sorted({(r['category'], r['subcategory']) for r in rows if r['category'] == 'object' and r['subcategory']})
    iid = sorted({r['intent_in_dev'] for r in rows if r['intent_in_dev'] is not None})
    S = {'config': cfg_name, 'judge': jn, 'pass_score': thr, 'root': meta.get('root'), 'root_sha': meta.get('root_sha'), 'overall': agg(rows),
         'by_category': {c: agg([r for r in rows if (r['category'] or '?') == c]) for c in cats},
         'by_subcategory': {f'{c}/{s}': agg([r for r in rows if r['category'] == c and r['subcategory'] == s]) for c, s in subs},
         'by_intent_in_dev': {str(v): agg([r for r in rows if r['intent_in_dev'] is v]) for v in iid},
         'by_intent_in_dev_category': {f'{c} / intent_in_dev={v}': agg([r for r in rows if r['intent_in_dev'] is v and (r['category'] or '?') == c]) for v in iid for c in cats},
         'skipped': [it['id'] for it in items if (jload(out / it['id'] / 'plan.json') or {}).get('skipped')],
         'face_closeup': agg([r for r in rows if r['face']]), 'items': rows, 'written': time.strftime('%Y-%m-%d %H:%M:%S')}
    jdump(S, out / 'summary.json')
    hdr = f'| category | n | scored | success % | strict % (>= {strict}) | mean | sources | first motion p50/p90 ms | planner-only first motion p50/p90 | plan ms p50/p90 | $/plan | judge $ |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n'

    def line(name, g):
        return (f"| {name} | {g['n']} | {g['scored']} | {g['success_pct']} | {g['strict_pct']} | {g['mean_score']} | {' '.join(f'{k}:{v}' for k, v in sorted(g['sources'].items()))} | "
                f"{g['first_motion_ms_p50']}/{g['first_motion_ms_p90']} | {g['planner_first_motion_ms_p50']}/{g['planner_first_motion_ms_p90']} | {g['plan_ms_p50']}/{g['plan_ms_p90']} | {g['usd_per_plan']} | {g['judge_usd']} |\n")
    md = (f"# plan_eval — config `{cfg_name}`, judge `{jn}` (pass = score >= {thr})\nsource tree `{S['root']}` @ {S['root_sha']}; set `{meta.get('set')}`\n\n"
          + hdr + ''.join(line(c, S['by_category'][c]) for c in cats))
    md += ''.join(line(k, g) for k, g in S['by_subcategory'].items())
    md += line('face_closeup items', S['face_closeup']) + line('**ALL**', S['overall'])
    if S['by_intent_in_dev']:
        md += '\n### held-out: by intent_in_dev (False = the intent never appears in dev)\n\n' + hdr + ''.join(line(f'intent_in_dev={k}', g) for k, g in S['by_intent_in_dev'].items())
        md += ''.join(line(k, g) for k, g in S['by_intent_in_dev_category'].items() if g['n'])
    md += ("\nsources: page = the page's own library / scene steps (no server call), library = the server library, planner = the model, declined = a refusal "
           "(a pass only on absent_decline rows), error = the plan failed.\n")
    if S['skipped']:
        md += f"skipped (no character has the part): {', '.join(S['skipped'])}\n"
    md += f"\nauto-fails (plan failed / nothing moved, no judge call): {S['overall']['auto_fail']}; hedged fallback used: {S['overall']['fallback_used']}; written {S['written']}\n"
    fails = [r for r in rows if r['score'] is not None and r['score'] < thr]
    md += '\n## failures\n' + ''.join(f"- {r['id']} [{r['category']}] \"{r['text'][:70]}\" — {r['source']}, score {r['score']}: {str(r['reason'])[:220]}\n" for r in fails)
    (out / 'summary.md').write_text(md)
    if not quiet:
        print(md[:md.index('\n## failures')])
        print(f'({len(fails)} failures listed in {out}/summary.md)')
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set'); ap.add_argument('--config', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--root', help='source tree (app/ + renderer/) to score; default: this worktree'); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--plan-jobs', type=int, default=3)
    ap.add_argument('--limit', type=int); ap.add_argument('--ids'); ap.add_argument('--no-judge', action='store_true'); ap.add_argument('--judge')
    ap.add_argument('--max-usd', type=float, default=2.0); ap.add_argument('--force-stage', choices=['plan', 'sim', 'render', 'judge'])
    ap.add_argument('--report', action='store_true'); ap.add_argument('--who', default='planner-score', help='free-form run label (informational)'); ap.add_argument('--configs', default=str(HERE / 'plan_configs.json'))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = jload(out / 'run.json', {})
    global ROOT
    ROOT = Path(a.root or meta.get('root') or ROOT).resolve()
    if meta.get('root') and Path(meta['root']).resolve() != ROOT:
        sys.exit(f"{out} was run on the source tree {meta['root']}, not {ROOT}: use another --out")
    if not (ROOT / 'app/planner.py').exists():
        sys.exit(f'--root {ROOT}: no app/planner.py there')
    sys.path.insert(0, str(ROOT / 'app'))
    set_path = a.set or meta.get('set')
    cfg_name = a.config or meta.get('config') or 'deployed'
    if not set_path:
        sys.exit('--set is required (first run)')
    if meta.get('config') and meta['config'] != cfg_name:
        sys.exit(f"{out} holds config {meta['config']!r}, not {cfg_name!r}: use another --out")
    items = [json.loads(x) for x in Path(set_path).read_text().splitlines() if x.strip()]
    if a.ids:
        # --ids a,b,c  or  --ids <file> (one id per line; '#' comments) — run tiers of a set this way
        want = ({l.split('#')[0].strip() for l in open(a.ids) if l.split('#')[0].strip()} if os.path.isfile(a.ids)
                else set(x.strip() for x in a.ids.split(',') if x.strip()))
        items = [it for it in items if it['id'] in want]
    if a.limit:
        items = items[:a.limit]
    judge_cfg = dict(jload(HERE / 'plan_judge.json', {'judge': 'none', 'pass_score': 4}))
    if a.judge:
        judge_cfg['judge'] = a.judge
    if a.report:
        report(out, items, cfg_name, judge_cfg)
        return
    cfgs = jload(a.configs, {})
    if cfg_name not in cfgs:
        sys.exit(f'unknown config {cfg_name!r}; known: {sorted(cfgs)}')
    cfg = cfgs[cfg_name]
    sha = git_sha(ROOT)
    shas = [x for x in (meta.get('root_shas') or ([meta['root_sha']] if meta.get('root_sha') else [])) if x != sha] + [sha]
    jdump({'set': str(Path(set_path).resolve()), 'config': cfg_name, 'config_body': cfg, 'judge': judge_cfg, 'root': str(ROOT), 'root_sha': sha, 'root_shas': shas,
           'started': meta.get('started') or time.strftime('%Y-%m-%d %H:%M:%S')}, out / 'run.json')
    if len(shas) > 1:
        print(f'!! {out} holds items made on other commits of this tree: {shas[:-1]} (now {sha})', flush=True)
    if a.force_stage:
        kill = {'plan': ['plan.json', 'sim.json', 'rig.jsonl', 'render.json'], 'sim': ['sim.json', 'rig.jsonl', 'render.json'], 'render': ['render.json'], 'judge': []}[a.force_stage]
        for it in items:
            for f in kill:
                (out / it['id'] / f).unlink(missing_ok=True)
            for f in (out / it['id']).glob('judge.*.json') if (out / it['id']).exists() else []:
                if a.force_stage != 'judge' or f.name == f"judge.{judge_cfg['judge']}.json":
                    f.unlink()
    budget = Budget(a.max_usd)
    t0 = time.time()
    print(f'== plan_eval: {len(items)} items, config {cfg_name}, judge {judge_cfg["judge"]}, root {ROOT} @ {sha}, out {out}', flush=True)
    n = asyncio.run(plan_stage(items, cfg, cfg_name, out, a.plan_jobs, budget))
    print(f'== planned {n} new ({time.time() - t0:.0f}s, ${budget.usd["plan"]:.4f})', flush=True)
    res = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {it['id']: ex.submit(item_stages, it, out, a, judge_cfg, budget) for it in items}
        for i, (k, f) in enumerate(futs.items()):
            res[k] = f.result()
            if res[k] not in ('ok', 'auto-fail', 'skipped'):
                print(f'  {k}: {res[k]}', flush=True)
            if (i + 1) % 10 == 0:
                print(f'  .. {i + 1}/{len(items)} items done ({time.time() - t0:.0f}s)', flush=True)
    bad = {k: v for k, v in res.items() if v not in ('ok', 'auto-fail', 'skipped')}
    print(f'== done in {time.time() - t0:.0f}s; estimated vendor cost this run: plans ${budget.usd["plan"]:.4f}, judge ${budget.usd["judge"]:.4f}; incomplete items: {len(bad)}'
          + (' (re-run the same command to resume)' if bad else ''), flush=True)
    report(out, items, cfg_name, judge_cfg)


if __name__ == '__main__':
    main()
