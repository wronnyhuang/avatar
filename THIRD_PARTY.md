

## Motion data (renderer/anim/motion2)

- `mm/db.npz` is a motion-matching database of skeleton poses and audio features (no video, no audio) curated from the Meta
  Seamless Interaction dataset (CC BY-NC 4.0, Meta FAIR). Adapted: converted to one canonical skeleton, quantised, segmented.
  `flow/flow_weights.npz` is a small model trained on the same data. Both are **non-commercial use only** and need attribution to
  the dataset's authors; the rest of this repository's code is unaffected.
- Not shipped: the library of commanded action clips (`act/library.npz`, `act/index.json`). Its clips come from datasets whose
  licences do not allow redistribution (AMASS / HumanML3D, Motion-X). Without it the body layer runs conversationally and commanded
  clips stay off; `act/player.py` documents the format, so a library built from data you are licensed to use drops in.
