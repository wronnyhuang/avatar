# Prior evaluation — generated motion vs real people

All numbers from `tools/prior/eval_prior.py`: the network runs FREE (sampling, shipped defaults) on the real live-available inputs of each person (their own speech, the partner's speech and face), and the same statistics as docs/MOTION_STATS.md are computed on the generated motion and on the real motion of the same people. `✗` = generated median outside the real IQR. Held-out = videos never seen in training (video-level split).

### Held-out people (shipped model: general prior + nod channel + reaction model)

37 people; mean |log(generated/real)| = **0.220**; generated median inside the real inter-quartile range for **96%** of 75 state×metric cells.

| metric | speaking real [IQR] | gen | listening real [IQR] | gen | idle real [IQR] | gen |
|---|---|---|---|---|---|---|
| head_yaw_sd10s | 4.4 [2.92, 5.68] | 4.43 | 1.7 [0.854, 3.78] | 1.75 | 1.79 [1.4, 2.92] | 2.46 |
| head_pitch_sd10s | 3.17 [2.29, 6.35] | 3.89 | 2.32 [1.38, 3.1] | 2.29 | 2.61 [1.71, 2.99] | 2.65 |
| head_roll_sd10s | 3.53 [2.48, 4.54] | 2.67 | 1.81 [0.989, 2.94] | 1.12 | 1.98 [1.16, 2.53] | 1.22 |
| head_yaw_speed_med | 6.93 [4.67, 9.22] | 5.42 | 1.75 [0.859, 3.8] | 1.83 | 2.56 [1.82, 3.88] | 2.31 |
| head_pitch_speed_med | 7.39 [5.29, 10.8] | 6.87 | 2.7 [1.37, 5.29] | 2.82 | 4.03 [2.28, 5.49] | 3.29 |
| head_pitch_psd_1.0_2.0Hz | 0.079 [-0.314, 0.304] | 0.0667 | -0.506 [-1.27, -0.133] | -0.441 | -0.667 [-1.02, -0.227] | -0.41 |
| head_pitch_psd_4.0_8.0Hz | -2.11 [-2.37, -1.84] | -1.96 | -2.8 [-3.38, -2.34] | -2.63 | -2.57 [-2.96, -2.43] | -2.69 |
| head_tx_sd10s_cm | 0.971 [0.538, 2.06] | 0.908 | 0.406 [0.255, 0.795] | 0.395 | 0.63 [0.409, 0.91] | 0.392 ✗ |
| stillness_frac | 0.032 [0.0128, 0.0523] | 0.0515 | 0.387 [0.242, 0.618] | 0.359 | 0.26 [0.167, 0.45] | 0.272 |
| nod_per_min | 38.2 [33.3, 50.3] | 39.1 | 19.8 [8.5, 31.4] | 14.3 | 25.5 [10.8, 38.1] | 14.5 |
| bignod_per_min | 15.2 [8.75, 26.4] | 15.5 | 7.91 [1.76, 15] | 5.95 | 6.89 [2.57, 14.3] | 4.83 |
| blink_per_min | 26.4 [13.8, 39.4] | 24.2 | 20.8 [11.5, 37.7] | 19.7 | 23.4 [12.1, 38.2] | 21.8 |
| gshift_per_min | 89.8 [75.8, 139] | 86.1 | 56.2 [35.5, 75.2] | 47.4 | 73.5 [53.9, 92.9] | 69.3 |
| avert_frac | 0.286 [0.0893, 0.378] | 0.24 | 0.112 [0.0679, 0.272] | 0.12 | 0.292 [0.126, 0.39] | 0.135 |
| avert_dur_s_med | 0.458 [0.433, 0.6] | 0.45 | 0.475 [0.35, 0.683] | 0.442 | 0.567 [0.4, 0.8] | 0.433 |
| smile_sd | 0.134 [0.0815, 0.227] | 0.117 | 0.17 [0.0785, 0.229] | 0.0899 | 0.106 [0.0407, 0.175] | 0.0922 |
| smile_per_min | 23.2 [8.73, 32.3] | 18.7 | 8.78 [3.65, 15.5] | 10.1 | 6.88 [2.3, 16.7] | 13.8 |
| brow_sd | 0.141 [0.128, 0.213] | 0.178 | 0.121 [0.1, 0.143] | 0.128 | 0.126 [0.0869, 0.15] | 0.149 |
| brow_per_min | 38.7 [26.2, 48.1] | 27.3 | 18.2 [4.04, 25.5] | 13 | 17.3 [16.1, 32] | 24.1 |

| coupling | real | generated |
|---|---|---|
| blink_after_speech_onset_lift | 0.897 | 0.988 |
| blink_at_gaze_shift_lift | 1.97 | 1.22 |
| brow_at_accent_lift | 1.16 | 1.11 |
| nod_at_accent_lift | 1.21 | 1.36 |
| nod_after_partner_pause_lift | 0.906 | 0.783 |
| bignod_after_partner_pause_lift | 0.865 | 0.808 |
| bignod_at_accent_lift | 1.19 | 1.14 |
| smile_mirror_peak_lag_s | -1.6 | 0.3 |
| smile_mirror_peak_r | 0.169 | 0.189 |
| nod_offset_median_s | 0.3 | 0.3 |
| bignod_offset_median_s | 0.35 | 0.3 |

### All 171 people (train + held-out; distribution check over the whole population)

171 people; mean |log(generated/real)| = **0.200**; generated median inside the real inter-quartile range for **96%** of 75 state×metric cells.

| metric | speaking real [IQR] | gen | listening real [IQR] | gen | idle real [IQR] | gen |
|---|---|---|---|---|---|---|
| head_yaw_sd10s | 4.26 [2.97, 5.9] | 4.38 | 2.19 [1.07, 4.07] | 1.94 | 1.67 [1.18, 2.96] | 2.85 |
| head_pitch_sd10s | 3.33 [2.54, 4.61] | 4.17 | 2.38 [1.37, 3.35] | 2.39 | 2.38 [1.13, 2.93] | 2.67 |
| head_roll_sd10s | 3.01 [2.11, 4.18] | 2.79 | 2 [1.16, 2.91] | 1.33 | 1.92 [0.719, 3.22] | 1.33 |
| head_yaw_speed_med | 6.37 [4.08, 8.66] | 5.46 | 2.11 [0.978, 3.84] | 1.91 | 2.38 [1.74, 3.51] | 2.11 |
| head_pitch_speed_med | 7.91 [5.48, 10.3] | 7.44 | 3.33 [1.7, 5.33] | 2.95 | 3.22 [2.23, 4.63] | 3.1 |
| head_pitch_psd_1.0_2.0Hz | 0.09 [-0.232, 0.326] | 0.0866 | -0.556 [-1.1, -0.147] | -0.487 | -0.838 [-1.24, -0.195] | -0.429 |
| head_pitch_psd_4.0_8.0Hz | -2.11 [-2.4, -1.85] | -1.92 | -2.73 [-3.09, -2.34] | -2.59 | -2.96 [-3.44, -2.4] | -2.48 |
| head_tx_sd10s_cm | 0.92 [0.658, 1.51] | 0.911 | 0.636 [0.371, 0.951] | 0.453 | 0.469 [0.225, 0.976] | 0.534 |
| stillness_frac | 0.0308 [0.00855, 0.076] | 0.0357 | 0.346 [0.21, 0.589] | 0.344 | 0.308 [0.176, 0.443] | 0.29 |
| nod_per_min | 40.6 [31.7, 50.7] | 39.7 | 21.6 [9.3, 33.3] | 16.6 | 21.2 [9.91, 29] | 17.6 |
| bignod_per_min | 17.9 [9.64, 27.1] | 15.5 | 6.53 [1.81, 13.7] | 5.32 | 6.55 [2.18, 12.4] | 6.89 |
| blink_per_min | 23.6 [13.8, 36.6] | 29.7 | 20.2 [8.82, 29.1] | 19.8 | 22.2 [11.2, 28.1] | 23 |
| gshift_per_min | 88.9 [67.4, 127] | 91.3 | 69 [37.9, 92.7] | 53.2 | 68.8 [48.6, 93.7] | 61.3 |
| avert_frac | 0.221 [0.0788, 0.338] | 0.25 | 0.219 [0.0886, 0.315] | 0.119 | 0.247 [0.159, 0.386] | 0.16 |
| avert_dur_s_med | 0.433 [0.35, 0.6] | 0.475 | 0.55 [0.429, 0.767] | 0.433 | 0.533 [0.425, 0.775] | 0.442 |
| smile_sd | 0.181 [0.0943, 0.231] | 0.126 | 0.16 [0.0864, 0.214] | 0.0918 | 0.144 [0.0551, 0.187] | 0.1 |
| smile_per_min | 25.3 [11.3, 37] | 20.7 | 9.51 [4.56, 20.6] | 10.7 | 9.61 [2.82, 17.4] | 13.3 |
| brow_sd | 0.177 [0.135, 0.228] | 0.163 | 0.142 [0.103, 0.182] | 0.13 | 0.151 [0.111, 0.175] | 0.139 |
| brow_per_min | 38.6 [25.6, 49.8] | 28.3 | 19.1 [8.96, 28.4] | 14.8 | 21.6 [16.5, 36] | 20.7 |

| coupling | real | generated |
|---|---|---|
| blink_after_speech_onset_lift | 0.839 | 0.969 |
| blink_at_gaze_shift_lift | 1.99 | 1.33 |
| brow_at_accent_lift | 1.14 | 1.07 |
| nod_at_accent_lift | 1.17 | 1.31 |
| nod_after_partner_pause_lift | 0.928 | 0.839 |
| bignod_after_partner_pause_lift | 0.816 | 0.889 |
| bignod_at_accent_lift | 1.07 | 1.24 |
| smile_mirror_peak_lag_s | 0 | 0.2 |
| smile_mirror_peak_r | 0.189 | 0.24 |
| nod_offset_median_s | 0.3 | 0.3 |
| bignod_offset_median_s | 0.333 | 0.3 |

### v1 (first commit) vs shipped, same held-out people

| | mean abs log err | inside IQR | speaking big nods/min (real) | listening big nods/min (real) |
|---|---|---|---|---|
| v1 (chunk-6, no nod channel, no dyads) | 0.448 | 79% | 5.8 (15.2) | 0.7 (7.9) |
| shipped v2 | 0.220 | 96% | 15.5 (15.2) | 6.0 (7.9) |

## Dyadic (two-person) held-out evaluation

People in held-out split-screen calls (both faces visible). The partner's real face/voice drive the model; the avatar's generated face is compared with the real listener/speaker.

| | mean abs log err | smile mirroring peak r (lag) | real | listening big nods/min (real) | listening avert frac (real) | listening stillness (real) |
|---|---|---|---|---|---|---|
| general prior only | 0.244 | -0.009 (+2.3 s) | 0.169 (-1.6 s) | 5.0 (6.2) | 0.09 (0.11) | 0.40 (0.41) |
| + reaction model (shipped) | 0.242 | 0.176 (+0.2 s) | 0.169 (-1.6 s) | 5.0 (6.2) | 0.09 (0.11) | 0.40 (0.41) |
| general prior, partner face HIDDEN (ablation) | 0.335 | -0.015 (-2.3 s) | 0.169 (-1.6 s) | 8.1 (6.2) | 0.14 (0.11) | 0.23 (0.41) |

Big-nod onsets relative to the partner's pause onsets (real, n=338, median +0.33 s): -1.00s:45, -0.75s:15, -0.50s:33, -0.25s:31, +0.00s:35, +0.25s:33, +0.50s:39, +0.75s:32, +1.00s:34, +1.25s:41
Big-nod onsets relative to the partner's pause onsets (generated, n=213, median +0.20 s): -1.00s:27, -0.75s:18, -0.50s:16, -0.25s:27, +0.00s:22, +0.25s:19, +0.50s:19, +0.75s:27, +1.00s:20, +1.25s:18

## The reaction model on its own (held-out dyads)

Partner-only inputs (face 9 + voice 4) + the self-speaking flag -> this person's next-0.5 s reaction; the ablation hides the partner (only the self-speaking flag remains).

| target | with partner | partner hidden | base rate |
|---|---|---|---|
| own smile (r, predicted vs real) | 0.214 | 0.022 | |
| own head speed (r) | 0.440 | 0.506 | |
| bignod (AUC) | 0.575 | 0.639 | 0.076 |
| avert (AUC) | 0.560 | 0.656 | 0.188 |
| brow (AUC) | 0.672 | 0.679 | 0.155 |
| smile_rise (AUC) | 0.633 | 0.699 | 0.162 |

Probe (partner smiles 0.6 for 3 s while talking; avatar silent): predicted own smile rises by 0.063 (blendshape units), half-rise 0.63 s after the partner's smile onset, peak at 2.97 s. Partner stops talking: P(big nod in next 0.5 s) 0.043 -> 0.063; P(gaze averted) 0.216 -> 0.281.

## Stability (10 minutes free-running: speaking 20 s / listening 20 s / silence 10 s cycles)

Human units (deg / cm), before the rig gain.

| channel | max abs over 10 min | per-minute means |
|---|---|---|
| yaw | 20.0 | -0.4 +1.0 +1.6 +0.8 +1.6 +1.0 +1.2 +3.8 +3.3 +2.7 |
| pitch | 26.1 | -2.2 -5.0 -5.0 -3.5 +1.5 -1.3 +1.0 -4.8 -2.4 -1.5 |
| roll | 19.3 | -2.2 +2.3 +0.9 -1.5 +2.2 +1.4 +1.1 +7.5 +3.5 -0.4 |
| tx | 6.9 | +0.3 -0.7 -0.3 +0.5 -0.5 +0.2 +0.8 -1.3 -1.4 -0.1 |
| ty | 4.6 | -0.3 -0.4 -0.7 -0.3 +0.6 -0.1 +0.1 -0.5 -0.5 +0.2 |
| tz | 6.3 | -0.8 +0.2 +0.9 +1.0 +0.1 -0.2 +0.2 +0.3 +0.6 +1.9 |
| gx | 21.9 | -0.2 +0.6 +1.0 +0.3 -1.0 +0.9 -0.6 -2.4 -1.7 +0.3 |
| gy | 14.5 | +0.1 -1.9 -0.8 -1.2 -0.7 -0.2 -0.2 -0.7 -1.0 -0.7 |
| blinks / min | 31.4 | |


## Live feature path (JS trackers, real voices)

`tools/prior/live_path_check.py` + `tools/prior/sim_live.mjs`: 4 held-out talking-to-camera clips; their raw per-frame
audio `{rms, pitch}` (what the browser measures) drives `prior.step()` first as the AVATAR's voice (no explicit speaking
flag — the JS VAD marks 88 % of those turns as speaking), then as the USER's voice while the avatar listens (static
neutral user face). Scored with `tools/motion_stats.py` against docs/MOTION_STATS.json: mean |log err| 0.241, 91 % of
cells inside the real IQR. Speaking: head yaw SD 4.0° (real 4.4), pitch speed 6.9°/s (7.1), big nods 15.2/min (17.1),
blinks 29.6/min (22.4), gaze averted 0.23 (0.22), nod / brow at pitch accents lift 1.51 / 1.40 (real 1.20 / 1.14).
Listening: yaw SD 1.9° (2.2), stillness 0.42 (0.35), big nods 5.8/min (6.2), blinks 17.8/min (19.9); gaze aversion 0.07
(real 0.22) — with a static user face the avatar keeps looking at the user more than real listeners do.
