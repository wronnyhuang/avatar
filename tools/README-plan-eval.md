# plan_eval — score the live motion planner on rendered clips

One command, resumable, any source tree:

```
tools/plan_eval.py --set <items.jsonl> --config <name> --out <dir> [--root <worktree>] [--jobs 8]
```

- `--set`     a JSONL row set (`id`, `utterance`, `category`, `scene`, `setup`, `expected`, `must_not`, `face_closeup`, ...).
- `--config`  a name in `tools/plan_configs.json`: `{"planner": {<Planner cfg overrides, passed through untouched>}, "character": "muse"}`.
              `deployed` = the planner defaults (OpenRouter). New keys (`"format":"prims"`, `"fewshot"`, `"selfcheck"` ...) need no change here.
- `--root`    the tree whose `app/` + `renderer/` are scored (planner.py, motion.js, voicetools.js, the renderer). Default: the
              worktree this tool sits in. The same scorer scores a "before" tree and every candidate branch.
              `run.json` keeps the root and its git sha; an `--out` made on another root is refused.
- `--out`     one directory per item; every stage is a file, a killed run continues where it stopped: **re-run the same command**.
              `--report` re-prints the table from what is on disk. `--force-stage plan|sim|render|judge` redoes from that stage.
- `--max-usd` stops new vendor calls at that estimated cost (plans + judge) for this invocation (default 2).
- Long runs: detached, e.g.
  `nohup setsid tools/plan_eval.py ... > <out>/run.log 2>&1 &`
- Child processes (plan_render.py) run on `$PYTHON`, else the interpreter running plan_eval.py.

## What one item goes through
1. **route + plan** (`plan.json`) — the live route, not only `planner.plan`:
   `node tools/plan_sim.cjs --route` runs the real `app/voicetools.js` `do_action` (sceneRoute / noThingHere / decline /
   page library / performer) on the real `app/motion.js`, in the row's `scene` and in the state its `setup` left
   (description = the user's words = the utterance; there is no voice model in the loop).
   - the page ends it (a decline, the page library, scene steps) → `source: "page"`, the engine commands are replayed;
   - the call reaches the performer → `app/planner.py` `Planner(<config>, cache off)` with the body state the page sends
     (+ `scene`, `previous_request`); server library → `source: "library"`, model → `source: "planner"`;
   - `use_object` (live: the voice model would call the `object` tool) → the server planner reads the words.
   Character: the config's; a row whose `character_hint` names a part it lacks (`needs_tail` on the muse) runs on the first
   character that has it (MANIFEST traits: the fox); none → the row is skipped and listed.
2. **sim** (`sim.json`, `rig.jsonl`) — page-faithful playback on `motion.js`; first-motion time = the first frame that differs
   from a reference run, measured from the request (planner items: the first pose batch's arrival + the page's buffer rule).
3. **render** (`clip.mp4`, `strip_NN.jpg`, for `face_closeup` rows `face.mp4` + `face_strip_NN.jpg`) — CPU, `--jobs` items in parallel.
4. **judge** (`judge.<name>.json`) — the judge and pass threshold in `tools/plan_judge.json` (picked by calibration:
   `tools/plan_calibrate.py`); verdicts are cached by content hash (`$PLAN_JUDGE_CACHE`, default `runs/cache/plan_judge`).

Scene: the engine keeps its real configuration; a thing that is not in the row's `scene` is taken out of what the page sees,
refused by name if an engine command asks for it, and not drawn. A refusal is a failure (score 1, no judge call) — except on
`absent_decline` rows, where the clip of what the body did is judged and the judge is told the decline it speaks.
A failed plan or a plan that moves nothing is score 1 without a judge call.

## The table
`<out>/summary.md` (+ `summary.json`): per category — n, success % (score >= threshold), mean score, sources, first motion
p50/p90 (all items and planner-sourced items separately), plan ms, $/plan; `object/<subcategory>` rows; the face close-up
subset; for a held-out set, the same by `intent_in_dev` (True / False, overall and per category); then every failure with the
judge's reason.

## A sealed held-out set
Keep a held-out row set that is never looked at while tuning, and score each finalist on it once:
```
tools/sealed_score.py --root <tree to score> --set <sealed/heldout.jsonl> --config <name> --results runs/planeval/sealed --max-usd 3
```
`sealed_score.py` prints aggregates only (not the failure list or the item folders), so tuning stays blind to the held-out
items. The generalisation gap is the two `intent_in_dev=` lines of `summary.md` (False = the intent never appears in dev).
Cost reference for `--config deployed` (OpenRouter): about $0.0013 a planned item, ~$0.33 for 252 rows (list prices).

## Calibration of the judge
`tools/plan_calibrate.py build|run|report --out runs/planeval` — hand labels `<out>/calib/hand_labels.jsonl` (written before
any judge ran), verdicts `<out>/calib_verdicts/<judge>/`, result `<out>/calibration_table.md`.

## Configs worth knowing
`deployed` pins "format":"raw" (the older live behaviour); the `final_*` configs name phases/doc explicitly, so they are the
same on any tree. Code default: format phases, prompt_style doc, model gem35lite, hedge dsv41flash after 2 s, server library
on. Rollback = state/config.json "planner": {"format": "raw"}. Tiers: add `--ids <file>` (one id per line, '#' comments; rows
not listed are skipped) to any command. Judge = `tools/plan_judge.json`.

A `hybrid` format (phases whose parts are named blocks or raw keys) was also tried in a leave-vocabulary-out bake-off
(`hybrid_*` configs); no hybrid beat raw, so it is not the default.
