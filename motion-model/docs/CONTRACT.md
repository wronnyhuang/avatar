# The avatar's unified learned motion model, v1 — design contract

Components: DATA (`canon1/`, `tools/data/`), MODEL (`unified/` model, train, runtime; `tools/train/`), RETARGET
(`retarget/`, plus the live integration in the avatar app), EVAL (`evalkit/`). The representation is defined by
`canon1/spec.py`.

## 0. Design scope
- This network handles ONLY the **continuous non-verbal mapping**: from the user's camera + audio (plus the avatar's
  own speech audio and its own past motion) to the avatar's face, gaze, head and body posture.
- **No** text-to-motion, **no** intent/mood/LLM-cue inputs, **no** commanded actions in the model. Commanded actions
  (wave, backflip, pick up the laptop) come from a voice-model function call to a planner model that writes keyframes
  live (see the app's `docs/PLANNER.md`).
- Required: an **override / blend interface**. While a commanded action plays it OWNS the body channels it touches; this
  network keeps the face, gaze and the untouched channels, and blends back smoothly when the action ends.
- Lip sync keeps the mouth while the avatar talks; everything else is blended.

## 1. Layout
Data and output locations are set by environment variables with repo-relative defaults (`canon1/paths.py`):
```
MM_DATA_DIR  (default <motion-model>/data)
  canon1/<source>/*.npz + index.jsonl   CANON v1 sessions (DATA output)
  splits.json                           person-disjoint splits
  fit/                                  MediaPipe -> CANON fitter parameters + noise profile
MM_WORK_DIR  (default <motion-model>/work)
  runs/<run>/                           MODEL output (checkpoints, logs, bundles)
  eval/                                 EVAL output (clips, renders, scores, side-by-side mp4s, numbers.json)
DATASETS_DIR (default <motion-model>/datasets)   raw public datasets (Seamless, Learning to Listen, CMU Panoptic)
```
Python deps: torch (CPU is enough for inference), numpy, scipy, mediapipe, onnxruntime, opencv, soundfile, matplotlib.

## 2. CANON v1 (see canon1/spec.py — the code is the definition)
189 numbers per person per frame at 30 fps: root lean (3) + root shift (3) + 13 SMPL-H upper-body joints × 3 axis-angle
(spine1-3, neck, head, collars, shoulders, elbows, wrists) + 2 × 15 finger joints × 3 + 52 MediaPipe/ARKit blendshapes +
gaze yaw/pitch. 9 validity groups: torso, head, arm_l, arm_r, fingers_l, fingers_r, face_upper, mouth, gaze. Audio per
person: 42 causal features (`canon1/causal_audio.py`, identical to the live reaction process). Session npz format +
required meta: spec.py. Loss/metrics only ever use frames×groups whose mask is 1.

## 3. DATA
- `canon1/<source>/*.npz` + `index.jsonl` (one line per session: path, dataset, session, persons, split, frames,
  person_hours, source per person, quality, label e.g. improvised/naturalistic/listening/talking).
- `splits.json`: person-disjoint train / val / test. Test people never appear in train or val anywhere
  (participant ids for Seamless; channel/identity for YouTube, ViCo, L2L; subject ids for Panoptic).
- `fit/`: the MediaPipe → CANON v1 **fitter** (python, importable, CPU, < 2 ms/frame) used for video sources AND
  live on the user's camera:
  `from canon1.fit import Fitter; f = Fitter(); motion189, mask9 = f.step(mp_frame)` where mp_frame = dict(face_present,
  blendshapes[52], face_matrix[16], pose_world[33,4] (x,y,z,visibility, metres), pose_image[33,4] (normalized),
  hands: list of {handedness:'Left'|'Right' as MediaPipe reports it, world[21,3], image[21,3], score}, image_wh).
  Stateful (smoothing / running medians are causal). Its accuracy is measured against SMPL-H on held-out Seamless
  video people.
- `fit/noise_profile.json`: per-column fitter residual std (+ lag-1 autocorrelation) measured on held-out
  Seamless people (MediaPipe-fitted vs SMPL-H/mapper canon) — training noises user-side inputs with it.
- Sources in priority order: (1) Seamless (all landed pairs, both people) → (2) the fitter + noise profile → (3) a
  ~11 h set of public YouTube conversation clips and ViCo through MediaPipe + fitter → (4) Learning to Listen
  (head/jaw only) → (5) CMU Panoptic (body/hands) → (6) AMASS/HumanML3D (needs an account; not used).
- The Seamless → MediaPipe face mapper and the first learned-reaction model's code live in `canon1/react_v1/`.

## 4. MODEL
- Small **causal transformer backbone** over ~3 s (90 frames) of: self past motion (committed frames), user motion+mask
  (noised with the fitter profile during training), user audio, self (avatar speech) audio with a configurable lookahead
  (default 6 frames = 200 ms; the live page reads TTS ahead). **No intent/cue inputs.**
- **Flow-matching head** samples the next ~0.5 s chunk (15 frames) in 4–8 Euler steps, **inpainting** from known frames:
  the last committed frames (so chunks join smoothly) and any channels an override supplies. Inpainting is trained
  explicitly (random known-prefix length, random known channel groups over random spans).
- Jerk / velocity penalty on the predicted clean chunk; boundary continuity with the committed frame.
- Responsiveness levers (the first learned-reaction model ignored the user): user-input dropout + classifier-free
  guidance on the user stream at sampling time, self-history noise/dropout, expressive/reaction-moment weighting,
  still-segment downweighting.
- Real time on a desktop CPU: plan step p99 < 10 ms at 2 threads, amortized < 3 ms per 30 Hz frame.
- Bundle (`runs/<run>/bundle/`): config.json, weights (torch state_dict and/or ONNX), norm.npz, train/val curves,
  data hours used.
- Runtime API (CPU, numpy in/out), in `unified/runtime.py`:
  ```python
  p = UnifiedPlanner(bundle_dir, threads=2, seed=0, replan_every=3, steps=6, guidance=1.5, temperature=0.8)
  out = p.step(user_motion189|None, user_mask9|None, user_audio42, self_audio42_ahead)   # once per 30 Hz frame, causal
  #  -> {"motion": (189,), "owned": (9,) float 0..1 how much each group is owned by an action, "plan_ms": float|None}
  p.set_override(groups=["arm_r","fingers_r"], values=None|(189,)|callable(frame_idx)->(189,), fade_in_s=0.15)
  p.release_override(groups=None, fade_out_s=0.4)        # blend back: the model inpaints from the action's last pose
  p.reset()
  ```
- Training runs locally (one GPU, or CPU for smoke runs); configs in `tools/train/configs/` (`final.json`,
  `final3.json`).

## 5. RETARGET (+ override compositor + live integration)
- Per-character map: CANON v1 → each character's rig controls (`retarget/maps/<character>.json` + `retarget/retarget.py`),
  for the Muse first, then the other 11 (fox, bear, …). Rule-driven extras (ears, tails, whiskers…). Code keeps IK,
  contact and clipping (e.g. arm joint angles → FK → hand targets → the rig's own IK/limits).
- Compositor: lip-sync owns the mouth while the avatar talks (smooth handoff); commanded actions own the rig channels they
  touch (map rig channels ↔ canon groups), with fade-in and a smooth blend back driven by `release_override`.
- Offline tool for EVAL: `retarget/render_canon.py --canon x.npz|.npy --character muse --audio voice.wav --out clip.mp4`
  (uses the avatar repo's own renderer on CPU).
- Live integration: a reaction mode `?react=unified` (config `react.mode`), default unchanged (rules). The page sends
  what the fitter needs (pose + hand landmarks).

## 6. EVAL
- Held-out people only (test split). Clip set: (a) Seamless test-split sessions with video (native MediaPipe on the user's
  video → fitter, the real partner's voice as the avatar's speech, the real partner's motion as the HUMAN reference),
  (b) the non-command clips of the frozen held-out suite (real public YouTube people as camera + mic).
- Systems: **unified** (new) vs (a) **rules** = the rules controller replayed offline exactly,
  (b) **gru-prior** = the GRU idle prior (prior.js) alone, (c) **copy-delay** = mirror the user with ~1 s delay,
  plus **human** reference where it exists.
- Measures: (1) an LLM judge on side-by-side clips (paired, bootstrap CI);
  (2) responsiveness (does it react to the user at all: input-ablation sensitivity, event→reaction rates/latencies for
  smiles, nods, speech turns, gaze, vs humans); (3) human-range statistics (`motion_stats`: head spread, nods,
  blinks, look-away, stillness… % inside the human IQR, mean |log ratio|; see the app's `docs/MOTION_STATS.md`);
  (4) latency (per-step ms on a desktop CPU, reaction latency).
- Ship gate (all must hold, vs rules): judge weighted/naturalness paired CI lower bound > 0 (or a clear win with no axis
  worse), responsiveness not worse, stats closer to human, latency within budget, 10-min rollout stable.
- Outputs: `eval/numbers.json`, `eval/REPORT.md`, side-by-side mp4s (phone-friendly, ≤ 1 Mbps) in `eval/sbs/`.

## 7. Data hygiene
- Downloaded data is untrusted: run with `python -I`, never `pickle.load` / `allow_pickle=True` on dataset files.
