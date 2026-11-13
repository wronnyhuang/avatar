# Avatar judge rubric (prompt v2)

Scored: ONLY the rendered character in `avatar.mp4` (its motion, face, gaze, mouth, voice timing).
Never scored: the human user (context strip), rendering style, photorealism, voice timbre, words chosen.
Cartoon animals and photoreal talking heads are judged on the same BEHAVIOUR scale.

## Axes (integer 1–10, `null` only where stated)

| axis | weight | what it measures |
|---|---:|---|
| naturalness | 0.30 | Alive and human-like behaviour: continuous small idle motion (breathing, blinks, head drift), smooth (no per-frame shake/jitter), no frozen stretches, no robotic identical repeats/loops, motion eases in and out. |
| reaction_timing | 0.15 | Visible reaction to each cue / user turn starts inside its expected window (≈0.2–1.5 s); replies start ≈0.4–1.5 s after the user stops; `null` when the clip has no cue and no user turn. |
| emotional_fit | 0.12 | Face/body expression and energy match the situation (greeting → warm, sad user → softer, surprise → startle) without over-acting or a fixed mask. |
| gaze | 0.08 | Looks at the user/camera while listening and talking, natural glance-aways and blinks, follows the user's movement when relevant; not a dead fixed stare, not darting. |
| motion_variety | 0.10 | Amount AND variety: several body parts take part (head, brows, ears, arms, body), gestures differ over time; not one repeated gesture, not motionless. |
| commands | 0.10 | Only when the clip has `kind=command` cues (else `null`): performs exactly the asked motion (right part, right side, right direction) promptly. |
| talk_amount | 0.08 | Talks as much as the situation asks (`talk` = reply/quiet/either): answers when addressed, stays quiet when it should, never talks over the user, no rambling. |
| lip_sync | 0.07 | Mouth opens/closes with the voice syllables, starts/stops with the audio, no visible lag; closed and still when silent. `null` if the avatar never speaks. |

Weights are renormalized over the non-null axes.

## Anchors (all axes)

10 indistinguishable from a skilled human/animator performance · 8 natural, minor flaws · 6 clearly artificial but acceptable ·
4 broken in places · 2 broken throughout · 1 nothing works.

Hard caps the judge must apply: a frozen stretch > 1 s while it should be alive → naturalness ≤ 4 (whole clip frozen → ≤ 2);
per-frame shake/jitter → naturalness ≤ 3; the same motion repeating identically (loop) → naturalness ≤ 3 and motion_variety ≤ 3;
mouth still while the voice plays → lip_sync ≤ 2; mouth visibly lagging/leading the voice (≈ ≥ 0.3 s) → lip_sync ≤ 4;
talking over the user for > 1 s → talk_amount ≤ 4; a missed cue → reaction_timing ≤ 3 for that cue.

## Final score (eval/judge.py `combine`)

Per axis: `final = (1 − α)·judge + α·objective` where `eval/avmetrics.py` has an objective 0–10 counterpart
(naturalness ← naturalness_obj, reaction_timing ← reaction_obj, motion_variety ← variety_obj, talk_amount ← talk_obj,
lip_sync ← lip_sync_obj); α = 0.35, except lip_sync α = 0.6 (frame-accurate mouth/voice cross-correlation beats a
4 s strip read by the judge). Otherwise `final = judge`.
Veto: when naturalness_obj ≤ 3 (a hard cap fired: frozen > 1 s, per-frame shake, identical loop), final naturalness ≤
naturalness_obj + 2 (judges miss shake).
`mean = Σ w·final / Σ w` over non-null axes; **`weighted = 0.65·mean + 0.35·min(final axes)`** (weakest link: one broken
axis — a missed cue, a 600 ms lip offset — must drag the clip below a clean one). `--repeat N` averages the N judge
calls per axis first. The judge never sees the objective scores (the timeline plot carries measurements, not scores).
