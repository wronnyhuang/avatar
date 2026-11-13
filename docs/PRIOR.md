# The learned motion prior (`app/prior.js` + `app/prior_weights.json`)

Small causal networks trained on real people (docs/CORPUS.md: 5.15 person-hours of usable tracking from 262
YouTube clip windows, including 1.9 h of two-person calls with BOTH faces visible). Every 30 Hz frame the prior samples
the avatar's next head / gaze / brow / smile / squint / blink / nod / arm-energy motion from what the browser has live:
the avatar's own voice, the user's voice, the user's face, and its own previous motion. It samples rather than
averaging, so motion stays lively, and it is bounded and mean-reverting, so it stays stable over minutes.

Two models ship in one weights file:

* **general prior** (speaker + listener): GRU-MDN over own motion, conditioned on self voice, partner voice and
  partner face; with an explicit big-nod channel.
* **dyadic reaction model**: sees ONLY the partner (face, head, arm energy, voice/turn state) plus the self-speaking
  flag, and predicts this person's next-0.5 s reaction (smile, head energy, nod, gaze aversion, brow, smile onset).
  Trained on both directions of every two-person clip. At runtime it drives smile mirroring (its other outputs are
  exported but off by default — see "Dyadic").

## Usage

```js
import './prior.js';                               // classic script / UMD; defines globalThis.createPrior
const weights = await (await fetch('prior_weights.json')).json();
const prior = createPrior(weights);                // options below; seedable
// every frame (30 Hz):
const out = prior.step({
  self:    { rms, pitch, speaking },               // AVATAR voice (TTS track): rms linear 0..1, pitch Hz (0 = unvoiced)
  partner: { rms, pitch, speaking },               // USER mic, same units (speaking optional: VAD inside)
  face:    { present, blend, matrix },             // USER face: MediaPipe blendshapes ({name:score} or the 52-array)
                                                   //   + facialTransformationMatrixes[0].data (row- or column-major,
                                                   //   auto-detected) — or yaw/pitch in DEGREES (yaw + = toward the
                                                   //   camera image's right, pitch + = nose down)
  hands:   { speed },                              // optional: user wrist speed (shoulder widths/s) or [{x,y}] wrists
});
// out.head_yaw/pitch/roll (rad), head_x/y/z (scene units), gaze_x/y (-1..1), blink_l/r (0..1), brow_l/r (-1..1),
// smile (0..1), squint, eye_wide (0..1); extras: out._arm (0..1 gesture energy), out._human (human units incl. nod),
// out._reaction (the reaction model's current predictions), out._features (the 17 model inputs, for logging)
```

The controller (`motion.js`) should treat these as the *conversational life layer*: add commanded poses, gestures and
perception reflexes on top (or blend toward them), and keep lip/jaw from lipstream. `prior.reset()` restarts the state;
`prior.stepExt(features17)` drives it from precomputed features (tests). Cost: ~0.13 ms per frame in node.

Options (`createPrior(weights, opts)`; defaults live in `weights.defaults`, tuned on the evaluation below):

| option | default | meaning |
|---|---|---|
| `temp` | head 0.85 ×6, gaze 1 ×2, brow 1.3 ×2, smile 0.7, squint 1, wide 1, arm 1 | per-channel sampling temperature on the mixture sigmas |
| `speakHead` | 1.5 | head temperature while the avatar speaks (speech beats are under-produced at 1.0) |
| `idleHead` | 0.65 | head temperature while nobody speaks (real idle people are stiller than listeners) |
| `blinkSaccade` | 3 | blink hazard multiplier in a 200 ms chunk whose sampled world-gaze shift is >= 8° (real blinks cluster at gaze shifts: lift 2.0) |
| `anchor` | 0.01 | per-frame pull of the WORLD gaze (eye + head) toward the user; eyes counter-rotate like the VOR |
| `leak` | translation 0.005, brows 0.01, smile 0.02 | per-frame decay toward neutral (drift guard) |
| `smoothHz` | 3 | one-pole low-pass on the head angle/translation OUTPUT, applied after the nod layer (removes sampling jitter above ~3–4 Hz) |
| `smileHz` | 1.5 | one-pole low-pass on the smile OUTPUT, after the mirroring layer (real smiles rise over ~0.5–1 s) |
| `nodGain` | 1.0 | multiplier on the big-nod hazard |
| `react` | `{smile: 2, nod: 0, gaze: 0, energy: 0}` | gains of the reaction model on smile mirroring / nod hazard / gaze anchoring / head energy |
| `gainHead` | 1.3 | human degrees -> rig radians multiplier (see "Rig mapping") |
| `maxHeadRad`, `maxTrans` | 0.5, 0.25 | soft (tanh) output limits for head angles / translation |
| `gainGaze`, `gazeDegPerUnit` | 1.0, 30 | rig gaze 1.0 = 30° of eye rotation |
| `gainBrow` | 1.3 | brow units -> rig brow |
| `gainTrans`, `sceneUnitsPerCm` | 0.5, 0.085 | head translation cm -> scene units |
| `seed`, `mode` | 1, `'sample'` | `'mean'` = argmax component, no noise, no nods (tests) |

## Inputs (17 per frame — all available at time t; strictly causal)

| # | feature | live source | training source |
|---|---|---|---|
| 0–3 | self voice: level, f0, voiced, speaking | avatar TTS track | the person's own speech (audio VAD × their own jaw activity) |
| 4–7 | partner voice: same | user mic | speech while this person's mouth is still (the other party) |
| 8 | partner face present | MediaPipe on the user | other panel of a two-person clip (0 for single-person clips) |
| 9–11 | partner smile, jawOpen, brow | blendshapes | same blendshapes on the other panel |
| 12–13 | partner head pitch / yaw velocity (/30 deg/s) | transformation matrix | same |
| 14–15 | partner head pitch /10, yaw /20 (deg, relative to a running neutral) | same | relative to the clip median |
| 16 | partner arm energy, log1p(wrist speed in shoulder widths/s, 1 s envelope) | hands (optional) | pose landmarks |

Voice: `level = clip((dB − floor)/(peak − floor), 0, 1.2)` with a fast-attack / slow-release peak tracker and
`floor = max(min-tracker, peak − 30 dB)` (training: clip 98th / 10th percentiles); `f0` = log2(f0 / running median f0)
clipped ±1 (= semitones/12); all four are zero when not speaking. `speaking` = the TTS is in an utterance if the caller
says so, else a level VAD with 300 ms hangover. Partner pose: one-pole 6 Hz smoothing, then a ~20 s running neutral
(training: 6 Hz zero-phase low-pass, clip median). Speaker/listener roles in training come from audio VAD + which face's
jaw is moving (rolling 1 s jawOpen SD > 0.035 with hysteresis; with two faces the talker must also move more than the
other) — talkers show 0.05–0.13, silent faces < 0.02, so the split is clean.

Own state (autoregressive): the previous output vector (14 channels, normalized), arm-visible flag (1 live), a blink
clock (time since the last blink, capped 4 s).

## Outputs

14 continuous channels, all relative to the person's neutral: `yaw pitch roll` (deg) · `tx ty tz` (cm) · `gx gy`
(eye-in-head gaze, deg) · `brow_l brow_r` · `smile` · `squint` · `wide` · `arm` (log1p wrist-speed envelope); plus two
event hazards: **blink** (plays an 8-frame template, fast close / slower open, 267 ms; real median half-max duration
200 ms) and **big nod** (a ≥ 5° nose-down excursion; plays a 0.45 s sine dip of 4–12° (lognormal, median 6°), 30 % of
them doubled, added to the output pitch). Sides follow the contract (`_l` = screen-left = MediaPipe "...Right";
verified with the iris / eyeLook correlation in `tools/prior/check_sides.py`).

## Networks

General prior: `x(17 + 14 + 2) → Linear 64 + tanh → GRU 128 → Linear → mixture density head`, stepped every
**6 frames (200 ms)**: each step outputs an 8-component diagonal-Gaussian mixture over the next 6 frames × 14 channels of
motion *deltas* plus a blink logit and a big-nod logit. One component per 200 ms chunk gives coherent movements instead
of per-frame jitter (per-frame sampling under-produced nods and turns by 2×). Step inputs = the mean of the last 6 frames
of features. Trained 40 epochs (torch CPU, Adam one-cycle, 256-frame crops, AR-input noise 0.05) on all 5.15 h,
video-level split (15 % of single-person videos + 5 of the 17 two-person videos held out).

Reaction model: `partner face 9 + partner voice 4 + self speaking 1 → Linear 32 + tanh → GRU 48 → 8 outputs`
(smile and head-speed Gaussians, 4 event logits), every frame, no own-motion input. Trained on 2.44 h of dyad tracks
(each clip twice: A reacting to B and B to A), weight decay 1e-2, input noise 0.2, partner-channel dropout 0.25,
early-stopped on held-out dyads.

Model selection (the free-running error is the mean |log(generated/real)| over ~75
state×metric statistics):

| model | free-running error (held-out) |
|---|---|
| per-frame GRU-MDN (C=1), v1 held-out set, tuned temperature | 0.52 |
| chunk 3, same | 0.48 |
| chunk 6, single-person corpus only (v1 commit) | 0.45 on the v2 held-out set |
| **chunk 6 + nod channel + dyads + reaction model (shipped)** | **0.220** (96 % of cells inside the real IQR; 0.200 / 96 % over all 171 people) |
| dyad-only fine-tune of the general prior ("listener" model) | no measurable change (0.356 vs 0.356 at the same settings) — not shipped |

## Rig mapping and the amplitude scale

Human degrees → rig radians × `gainHead` = 1.3. The cartoon heads are large and stylized and are seen at phone size;
1:1 human amplitude reads as stiff on them (the earlier procedural life.py nods were 3–6°, i.e. human-sized nods were
already at the low end of readable). 1.3 keeps the median speaking yaw SD (4.4° human) at ≈ 5.7° on the character, still
inside the human IQR, so the judge's statistics (`tools/motion_stats.py --gain-head`, which divides by the same gain)
stay comparable. Use 1.5 for very small renders. Gaze: rig 1.0 = 30° of eye rotation. Translation: 1 cm ≈ 0.085 scene
units (the fox skull is 1.27 units wide vs a 15 cm human head) × 0.5, because the rig moves the head on the neck while
most human head translation is whole-torso sway.

## Dyadic

Data: two-person calls where both faces are visible (`tools/prior/dyad_fetch.py`, `dyad_extract.py`): panel detection
from face clusters over time + split lines from column/row image statistics (side-by-side, stacked, picture-in-picture,
in-room two-shots supported), per-panel MediaPipe face + pose streams on the same frames/timestamps, a per-frame
`dyad_ok` mask (layout switches and cuts end segments), duplicate-person and 3+-people rejection. Accepted: 50 clips from
17 videos, 1.94 h with both faces tracked (41 side-by-side used; 9 in-room clips excluded as unreliable).

Findings (docs/MOTION_STATS.md, docs/PRIOR_EVAL.md):
* Smiles are shared: partner-vs-own smile cross-correlation peaks at r ≈ 0.19 at lag 0 (76 people). The reaction model
  predicts a person's next smile from the partner alone at r = 0.21 on held-out dyads (0.02 with the partner hidden);
  with it, the avatar's smile tracks the user's at r = 0.18 (real 0.17), following by ~0.2 s; without it r ≈ 0.
  Probe: the user smiles for 3 s → the avatar's expected smile rises by 0.06 (×2 react gain), half-rise 0.6 s later.
* Listener nods in these calls are NOT concentrated right after the partner's pauses (lift 0.8–0.9, flat timing
  histogram): backchannel nods happen during the partner's speech as much as in the gaps. The prior reproduces the rate
  (5.0 vs 6.2 big nods/min while listening on held-out dyads) and the flat timing. The reaction model does raise the
  nod probability when the partner stops talking (0.043 → 0.063 per 0.5 s), but that is not used by default.
* Hiding the user's face from the general prior makes the listening avatar too restless (stillness 0.23 vs 0.40 with
  the face, real 0.41; big nods 8.1 vs 5.0/min, real 6.2; overall error 0.335 vs 0.242): the face channel is what makes
  it settle into calm attentive listening on a call.
* For nods, gaze aversion, brows and head energy the reaction model does not beat "own turn state" alone on held-out
  dyads (AUC 0.56–0.67 vs 0.64–0.70) — 2.4 h of dyads is not enough to learn those partner effects beyond turn-taking,
  so those modulations ship disabled (`react.nod/gaze/energy = 0`; the code paths exist and are evaluated).

## Evaluation

`tools/prior/eval_prior.py` runs the network free (sampling, shipped defaults) on the real inputs of held-out people
(their own speech, the partner's speech and face) and computes the SAME statistics as `docs/MOTION_STATS.md` on the
generated motion. Full tables: `docs/PRIOR_EVAL.md` (raw: `docs/PRIOR_EVAL.json`).

## Files

* `app/prior.js`, `app/prior_weights.json` (1.4 MB, base64 float32) — runtime, no deps.
* `tools/prior/` — corpus (`fetch_corpus.py`, `dyad_fetch.py`), extraction (`extract.py`, `dyad_extract.py`,
  `run_extract.py`, `dyad_run.py`, `dyad_stats.py`, `check_sides.py`), features (`motionlib.py`, `build_dataset.py`),
  statistics (`mstats.py`, `reference_stats.py`, `corpus_doc.py`), models (`prior_model.py`, `train_prior.py`,
  `train_reaction.py`, `eval_prior.py`, `dyad_probe.py`, `pack_weights.py`, `make_eval_doc.py`), JS parity
  (`make_js_testvec.py`, `test_prior_js.mjs`), simulation (`sim_rig_log.mjs`).
* `tools/motion_stats.py` — the judge's tool: the same statistics from a rig-vector log, compared with the reference.

JS ↔ PyTorch parity: `python -I tools/prior/make_js_testvec.py app/prior_weights.json <data-dir> /tmp/tv.json &&
node tools/prior/test_prior_js.mjs app/prior_weights.json /tmp/tv.json` → max abs diff 9.5e-7 (general prior raw
head), 5.1e-7 (450-frame rollout), 7.2e-7 (reaction model).
