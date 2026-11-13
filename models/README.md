# Model bundles (small, inference-only)

| dir | what | used by | size |
|---|---|---|---|
| `unified-v2/` | unified flow-matching motion model (causal transformer d256x5 + rectified-flow head), ONNX + PyTorch weights, the ablation winner | `react.mode = "unified"` / page `?react=unified` (opt-in; default is the rule-based reactions) | ~44 MB |

The unified mode finds this bundle by default (`app/unified_link.py`: `react.unified.bundle` defaults to
`models/unified-v2`, and its code to `motion-model/`); override `react.unified.bundle` in config.json to use another.
Its worker needs `onnxruntime` (or `torch`) in the interpreter named by `react.unified.python` (default: the app's own).

The first learned-reaction model (`react.mode = "learned"` / `?react=learned`) is not shipped: with `react.root`
empty the page reports that mode unavailable and stays on the rules.

Training checkpoints, data packs and larger runs are NOT in the repo (see the main README, "Not in this repo").
Trained on Meta Seamless Interaction, CANDOR, Learning to Listen, ViCo, CMU Panoptic — non-commercial research terms.
