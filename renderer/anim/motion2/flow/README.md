# FLOW live package  — numpy-only streaming body motion for Juno (canon1, 30 fps)

Imports: numpy only (no scipy, no torch). Files: `stream.py`, `flow_weights.npz` (~14.5 MB: int8 matrices with one fp16
scale per row x 32 columns; per-frame in/out projections fp32; AdaLN modulation precomputed for the 4-step grid).

```python
import sys; sys.path.insert(0, PKG); from stream import Generator
gen = Generator(PKG, seed=0)        # defaults: steps=4 (the package only supports 4), commit=8 tokens (32 frames), hist=12 tokens,
                                    # xfade=4; hist=8 is ~15% cheaper and slightly worse (speech FDr 3.65 vs 3.52)
gen.reset()                         # or gen.reset(pre_motion (n,189), pre_feats (n,42)) to start from a known recent past
gen.set_state(speaking, partner_speaking)
gen.push_audio(feats)               # (n,42) own-voice canon audio feats (40 log-mel + loud_db + vad) as they arrive;
                                    # frame index 0 = the first step() after reset(); future frames are welcome (look-ahead)
pose = gen.step()                   # (189,) canon; body cols 0:135 filled, face 135:189 = kin.neutral()
gen.seed_history(poses)             # HAND-OVER from another source (motion matching): poses (n,189) = the last n poses actually
                                    # shown, oldest first, n >= 4 (48 = full history). Call at the hand-over frame BEFORE that
                                    # frame's step(); that step() computes one window synchronously (one-off ~60-110 ms on a hub
                                    # core) and crossfades out of poses[-1] (measured jump 0.014 rad). Audio + frame counter kept.
```
Keep stepping it while not selected if you want it warm; it does not need seed_history then, but its own trajectory will differ
from what was shown, so seed at every hand-over.
Schedule: a 32-frame window is generated while the previous 32 committed frames play; its compute (83 half-layer units) is
spread evenly over those step() calls. Own audio counts as known up to what has been pushed. While not speaking and nothing is
pushed, a silence frame is assumed. Deterministic given seed + the same call sequence.
Trained for: own audio known up to ~1 s BEFORE the window start (live buffer ~0.1 s minus 32-frame latency), no partner audio.
Numbers and caveats: T/logs/flow.md (FLOW piece of ).

v1.1 (05:47 11-20): window post-process at commit (no latency, no API/weights change): excess velocity step at every 4-frame
patch boundary beyond 2.5x the local frame-to-frame motion is spread over 4 frames (`clip`), window start inertialised against
the last committed velocity over 6 frames (`inert`), then the 4-frame crossfade. Kwargs deglitch/inert/dg/clip exist only to
reproduce v1.0 (deglitch=False, inert=0).
