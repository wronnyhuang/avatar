# Real-people motion corpus (for the motion prior)

Source: YouTube clip windows (150–180 s, <=480p) found by category search queries (`tools/prior/fetch_corpus.py`, `tools/prior/dyad_fetch.py`). License: YouTube Standard License — the media is NOT redistributed (fetch it yourself with the tools above); only derived per-frame features and aggregate statistics are used. No accounts/cookies were used. Full per-clip list: `docs/corpus_manifest.jsonl`.

| category | clips | videos | raw hours | clips usable | usable person-hours (face tracked, no cuts) |
|---|---|---|---|---|---|
| speak | 45 | 45 | 1.88 | 41 | 1.02 |
| listen | 57 | 44 | 2.38 | 38 | 1.09 |
| idle | 31 | 31 | 1.29 | 16 | 0.28 |
| dyad | 129 | 75 | 5.38 | 34 | 2.76 |
| **total** | 262 | 186 | 10.92 | 129 | 5.15 |

Layouts of the usable clips: single: 85, dyad-lr: 41, lr: 3

Category = the search intent (speak: talking to camera; listen: split-screen podcasts / calls / interviews / reactions; idle: study-with-me / silent webcam; dyad: two-person calls with BOTH faces visible, panel-cropped). Per-frame states (speaking / listening / idle) are assigned from audio VAD + the face's own jaw activity, not from the category — see docs/MOTION_STATS.md.
