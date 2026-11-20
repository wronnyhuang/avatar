# mm live_pkg — conversational motion matching on real human motion ( MM) — VERSION: see config.json `_version`

Pure numpy (+ nothing else). Files: `stream.py` (Generator), `audio_feats.py` (streaming canon audio features from
16 kHz PCM), `db.npz` (curated real Seamless motion, 77 min / 154 speakers, 13.6 MB packed (int16/int32 time-deltas: rotations 1.5e-4 rad, finger PCA 1e-3, positions 1 mm; velocities/activity recomputed at load); float16 + finger PCA; v2: DESPIKE only (Hampel), no low-pass -> real micro-motion),
`db.npz["rests"]` (4 arm rest poses: hands together at waist, together low, arms at sides, one arm down; v2),
`config.json` (the frozen live tuning: Generator(folder, seed) reads it; explicit kwargs override).

```python
import sys; sys.path.insert(0, "<this folder>")
from stream import Generator
from audio_feats import PcmFeaturizer

gen = Generator("<this folder>", seed=0)       # ~0.7 s load, ~250 MB RSS; kwargs below are optional
fx  = PcmFeaturizer()                          # if you have PCM, not canon feats

# whenever TTS audio arrives (any chunk size, AHEAD of playback is good: up to 1.0 s is used):
gen.push_audio(fx.push(pcm_16k_float32))       # or gen.push_audio(feats (n,42)) if you already have canon feats
# Silence between utterances must be pushed too (zeros PCM), so audio frame k stays aligned with step k.

# every render frame (30 fps):
gen.set_state(speaking=bool, partner_speaking=bool)   # avatar talking / user talking (neither = idle)
pose = gen.step()                              # (189,) float32 canon1
gen.seed_pose(pose189, prev_pose=None)   # v4: pass the pose shown one frame earlier too -> velocity-continuous hand-over
# HAND-OVER: another generator had the body -> continue from this exact pose (undoes rest/gain layers,
                         # jumps to the nearest DB frame in arm joint angles, inertializes from the pose with zero velocity; ~1-2 ms)
gen.info   # {"energy": 0..1 smoothed, "arm_l": 0..1, "arm_r": 0..1 (current wrist activity), "state": 0 idle/1 speak/2 listen, "jumps": n}
```

Alignment: `step()` number k plays alongside audio frame k (frame k = [k/30,(k+1)/30) s of the voice timeline).
If the voice starts later than the generator, push zero-PCM (or silence feats) for the gap. Look-ahead used = whatever
has been pushed beyond frame k, capped at `lookahead_s` (default 1.0 s); with none it still works (state-driven).

Output conventions
- Cols 0:21 (root_rot, root_shift, spine1-3, neck, head) = `neutral + zero-mean real sway/nod` (DB high-passed at 4 s):
  apply ADDITIVELY as (pose - neutral) on top of gaze-follow / leans. Measured 1 h soak: no drift (10-min block means
  < 0.001 rad), root_shift |x| <= 0.3 dm.
- Cols 21:135 (collars, shoulders, elbows, wrists, fingers) = ABSOLUTE poses; arm rest = kin.neutral() arms (every DB
  speaker's rest re-targeted onto it), hands never above chin-5 cm or inside the torso (DB frames filtered with kin).
  Axis-angles are kept on the neutral's hemisphere (wrist norms may exceed pi): blend with slerp/quaternions.
- Cols 135:189 = neutral face/gaze (untouched).

v2 tuning (config.json): tuned for LIVE's measured look-ahead (33/67/100 ms p10/50/90) -> lookahead_s 0.1. Search every 20 frames
or at once on a voice onset (`onset_trig` 1.3); soft hand-height cost above 0.24 m (kin) + hard limits 0.33 m / spread 0.7 m;
`speak_still_w` 0.3 (prefer segments with some life while speaking); `arm_gain` 1.4 speaking / 1.2 otherwise; rests: a session starts
in a random rest and drifts to another every 20-60 s at a calm, non-speaking moment (1.5 s min-jerk); v3: every rest except hands-together-at-the-waist eases
out (0.4 s) when she starts talking; onset-triggered searches weigh the voice x2 (trig_aw); and back (1 s) after.

Kwargs: `lookahead_s` (1.0), `audio_w` (1.0; 0 = ignore voice), `blend` ("inert" | "xfade"), `halflife` (0.12 s
inertialization), `search_every` (8 frames), `rep_w` (3) / `rep_T_s` (900) anti-repetition, `energy`/`energy_w`
(gesture-energy dial, off by default; `gen.set_energy(-1..+1)` at runtime once `energy_w`>0, e.g. 0.15), `loud_ref`
(-47 dB start value of the running speech-loudness normaliser; adapts within ~2 s, TTS sits near -29 dB).

Budget v2 (hub, one core, CPU time, 8 min of demo speech): step mean 0.60 ms, p50 0.44, p99 3.7 ms, max 16 ms (onset-triggered full search);
wall clock on the hub at load 150-210 is noisier. RSS ~230 MB. Init ~1-3 s.

v4 (05:55): every transition is velocity-continuous (jump inertialization starts at the true velocity difference of the shown output vs the
new segment — v3 added the old velocity twice, a one-frame velocity step at every jump; arm-gain and arms-at-sides easing are critically
damped; the arms-at-sides rest returns only after 3 s without speech, not in every pause); blends while listening/idle use halflife 0.2 s
(`halflife_quiet`). db.npz unchanged from v3.
