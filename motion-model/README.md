# motion-model

Research code for the learned non-verbal motion model: the CANON v1 data format, the unified flow-matching model with
its streaming runtime and ONNX export, the retarget onto the avatar rig, and the evaluation kit. Everything runs
locally (one GPU for training; CPU is enough for the runtime, smoke runs and most tests). Design notes:
`docs/CONTRACT.md`.

| dir | contents |
|---|---|
| `canon1/` | CANON v1 spec (`spec.py`), MediaPipe fitter (`fit.py`), dataset converters (Seamless, Learning to Listen, CMU Panoptic, video), splits, style labels; `react_v1/` = helpers copied from the first learned-reaction model |
| `unified/` | model, flow head, data packs / sampler, streaming runtime (`UnifiedPlanner`), speech encoders, ONNX export, tests |
| `retarget/` | CANON v1 -> per-character rig params, override compositor, offline renderer driver |
| `evalkit/` | held-out clip set, system replays, objective measures, LLM judges, scorecard, report |
| `tools/train/` | packs, training (`train.py`, `train_multi.py`, `multi.py`), evaluation, runtime tuning, configs |
| `tools/audio/`, `tools/react/` | speech-encoder feature extraction and benches; reaction-adapter experiments |

## Paths

Nothing is hard-wired to a machine. Locations come from environment variables with defaults relative to this directory
(`canon1/paths.py`):

| variable | default | holds |
|---|---|---|
| `MM_DATA_DIR` | `./data` | processed data: `canon1/<dataset>/*.npz`, `splits.json`, `fit/fitter_params.npz`, `fit/noise_profile.json`, `mapper/` |
| `MM_WORK_DIR` | `./work` | outputs: `runs/`, `eval/`, `p2/` (packs, features, speech-encoder weights, scorecards) |
| `DATASETS_DIR` | `./datasets` | raw public datasets; or set `SEAMLESS_DIR`, `L2L_DIR`, `PANOPTIC_DIR` individually |

More specific overrides: `FITTER_PARAMS`, `MAPPER_DIR`, `SPEECH_ENC_MODELS`, `A2F_DIR`, `NOISE_PROFILE`; the eval kit
reads `EVAL_OUT`, `EVAL_AVATAR` (avatar checkout, default the parent repo), `EVAL_MODEL_CODE`, `EVAL_YT_CLIPS`,
`EVAL_AVATAR_VOICES`, `RENDER_PY` / `EVAL_PY` (interpreters, default the current one).

## Train and evaluate (local)

```
cd motion-model
python -I tools/train/train.py --config tools/train/configs/smoke_cpu.json --run-dir work/runs/smoke --pack-dir <packs>
python -I tools/train/train.py --config tools/train/configs/final3.json --run-dir work/runs/final3 --pack-dir <packs>
python -I tools/train/multi.py --config tools/train/configs/p2_next/multi_nx.json --run-dir work/runs/nx \
       --pack-dir <packs> --train-script tools/train/train_multi.py      # several models on one GPU
python -I unified/export.py work/runs/final3/bundle                      # ONNX graphs for the runtime
python -I -m pytest -q unified retarget/tests tools/train
evalkit/post_train.sh work/runs/final3/bundle final3 --no-judge          # scorecard
```

The paid LLM judges (`evalkit/judge`, `evalkit/pairjudge.py`) only run with `JUDGE_ALLOW_PAID=1` and an
`OPENROUTER_API_KEY` (or `GEMINI_API_KEY` for the direct route); their cost is logged to `$EVAL_OUT/spend.jsonl` and
capped by `JUDGE_SPEND_CAP` / `EVAL_SPEND_CAP_USD` (USD).
