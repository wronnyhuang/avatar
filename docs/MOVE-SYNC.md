# Move sync — a commanded move in step with the words around it

The motion should start in step with the avatar saying it will do it, and its "done" line should land when the motion
actually finishes.

## The timeline today (measured)
From the demo's session logs (recorded sessions; not part of this repository), ms after the tool call, median / p90:

| | median | p90 |
|---|---|---|
| the avatar starts its reply to the tool result (the announcement) | 876 | 2908 |
| the motion starts | 2303 | 4222 |
| motion start − announcement start | 2028 | 3668 |

The planner needs 2–3 s; the reply to the tool result ("Watch this backflip!") comes after 0.9 s. At `ready` the performer
asked for a second line (the cue) and waited 700 ms for its first audio; the audio came inside that window in about one in ten
planned moves, so the move usually started without it and the cue was heard into the motion. At the end, the check-in note
went out when the engine reported the end (0.3 s after the last dial was back at rest), so the "done" line was heard one
model latency (about 1 s, p90 about 3 s) later. A rules-layer run (gesture / pose / face) was reported finished a fixed
1.8 s after it started, whatever its length. No log records when the "done" line's audio starts; the replay measures it.

## What changed (`app/perform.js`; nothing delays a tool result or a reply)
**Start.** The tool result still goes back at once. It now asks for a word or two of acknowledgement instead of the
announcement; the cue is the announcement, and it is asked as early as it can be heard in time:
- at the plan's first pose (the model needs about 1 s to start speaking, the plan about 0.5 s more to be ready), or the
  moment the avatar's reply ends if that is later;
- the move starts at the cue's first audio, at most 1.5 s after it is ready (the thinking gesture stays up until then);
- if the reply to the call is still being spoken when the plan is ready, those words are the announcement: the move starts as
  they end (at most 700 ms after ready) and no second cue is asked;
- if the reply has not begun (a plan from the cache), the move starts at the reply's first audio (at most 700 ms).
  If the cap passes in silence the move starts and a late cue is asked only when the reply is long overdue (2.5 s after the
  call, nothing said since his request). Live with Gemini the reply's first audio comes
  0.6-1.5 s after the call and a cached plan is ready ~50 ms after it, so a cue asked at the cap (the first version) landed
  on a reply that was on its way: 10 of about 40 cached-plan moves, the first words cut off and restarted ("One step—" ...
  "Let's climb up high!") or a second announcement 1-3 s into the motion.
A known move (library, clip, a matcher's) still plays at once.

**Done.** The check-in note leaves one model latency before the motion ends, so its line is heard as the motion ends. The
end comes from the engine: `motion.kfState(id).end_s` (plan time when every dial is back at rest) for a planner job,
`motion.eta()` for engine moves and rules-layer runs (which now end when the motion layer says so, not after 1.8 s). The
latency is measured in the session (note sent -> next avatar speech onset, EMA, 300–1500 ms, first estimate 900 ms). The
model is also told the real duration once the whole plan is in ("takes about N s more ... do not say it is done before that"),
and a direct library result carries `takes_s` (`app/voicetools.js`).

## Both voice providers
The performer talks to the voice through `voice.prompt()` (`app/voice.js`) and the playback tap (`avatarAudio()`), which are
the same for both. Gemini Live (default): a function call is blocking and its `toolResponse` is sent at once, as before;
completion is a silent `clientContent` turn, sent while the model is idle. GPT Realtime: the note is a conversation item and
`response.create`, same timing. Holding the tool response until the motion ends was not done: the model is silent until it
has the response, which would put 2–3 s before its first words.

## Switch
`?movesync=0` (or hub config `move_sync: false`) = the flow before move sync; `tests/sem_page_offmode_test.cjs` proves the
performer's messages, notes and tool results byte-identical to its base with the switch off.

## Check
`node tests/move_sync_test.cjs` — scripted timelines for every rule above and `motion.eta()`. The numbers below come from a
replay (`tests/move_sync_sim.cjs`, which needs a fixture built from the recorded session logs; not included): the logged
moves from voice sessions, each with its own real call -> first pose ->
ready times, real move length and real time to first words, through the real performer and move engine. The model's reply to
a note is modelled (latency drawn from the real tool-result -> first-audio distribution, lengths from the logs' short lines).

| ms, median / p90 | before | after | after, if the first words are a word or two |
|---|---|---|---|
| \|motion start − announcement start\| | 1969 / 4020 | 1953 / 3820 | 277 / 1651 |
| motion starts while an announcing line is being said | 66 % | 79 % | 68 % |
| \|done line − motion end\| | 1277 / 2934 | 500 / 1696 | 558 / 1993 |
| tool call -> first words | 897 / 2908 | 897 / 2908 | 897 / 2908 |
| tool call -> motion start (planned moves) | 3067 / 4400 | 3033 / 4467 | 2967 / 4467 |

The third column depends on the voice model keeping its first words to an acknowledgement, as the new tool result asks; that
is not measurable offline. If it keeps announcing in its first words, the middle column holds: the announcement still leads
the motion by the planner's latency, which only a faster plan or holding its first words would change.

## Step sync — ONE start gate for every commanded motion
**The general cause.** A commanded motion started on its OWN clock (the moment the matcher read his words, the library
answered, or the plan was ready) while its announcement starts on the VOICE MODEL's clock (its first audio, 1.5–3 s after
his last word on Gemini Live). Nothing joined the two clocks except a per-route patch for planned moves (the cue note), so
known moves led their words by ~1.5 s and fresh plans got a second (and third) line asked at the motion's start.

**The mechanism** (`app/perform.js` `gate()`; one rule, every source): a commanded body motion starts at the avatar's first
voice after his last words — at once when that voice has already begun (it is speaking, or spoke since he stopped), at the
voice onset otherwise, and at most `STEP_MAX_MS` (1500 ms, `?stepmax=ms`; live: 900 ms let 4 of 12 known moves start before the words) after it could start; while he is still speaking
the bound counts from his end (a matcher that fires mid-sentence). Nothing waits on it but the motion: tool results and
replies are untouched. Users of the gate:
- the hub matcher / LLM matcher / caption matcher (`page.html` `runCommand`, actions move / object / gesture / pose);
- the voice model's direct library run (`voiceLibrary`, engine characters) — its tool result says "starting it as you speak";
- the performer's library answers (`runLibrary`) and planned / cached keyframe jobs (`onReady`: replaces the cue logic).
For a planned move the first line IS the announcement (the tool result now asks it to say what it is about to do; Gemini
announced anyway), the thinking gesture fills the planner wait from the call on, and the move starts at ready with no second
cue — two lines per move: the announcement and the done line (unchanged).
Not gated: `object` steps through `perform.engineCall` (their tool result must report the real outcome synchronously),
expressions / looks / releases, page UI commands.

**Switch:** `?stepsync=0` or hub config `step_sync: false` (also off whenever move sync is off). Without the page getter
(`d.stepSync`) the performer behaves exactly as before (all existing tests unchanged). Test: `node tests/step_sync_test.cjs`.
At a step-gated kf start the silent "playing now" / duration notes add "You already announced it: say nothing now." (live:
without it Gemini still volunteered a "Here it goes!" at the motion's start in 6 of 8 fresh plans).

**Live result** (Gemini Live, Muse, a private instance with the stub renderer, synthesised spoken commands),
median / p90 ms after his speech ends:

| | previous build | `?stepsync=0` (same hour) | `?movesync=0` | step sync (1500 ms) |
|---|---|---|---|---|
| known moves: \|motion start − first words\| | 1388 / 2189 (n=32) | 1542 / 7539 (n=15) | 1217 / 1631 (n=6) | **235 / 508** (n=21) |
| known moves: first words | 2091 / 2787 | 2438 / 8385 | 1906 / 2256 | 1995 / 2522 |
| known moves: motion (main stroke) starts | 743 / 888 | 987 / 1768 | 632 / 818 | 1763 / 2356 |
| known moves: body first reacts (stall gesture) | = motion start | = motion start | = motion start | 1119 / 1271 |
| planned, fresh: lines per move | 3 / 3 | 3 / 4 | 3 / 4 | **2 / 2** (n=8; 7 of 8 two) |
| planned, fresh: first words | 2080 / 2572 | 2043 / 2413 | 1793 / 2198 | 1790 / 2171 |
| planned, fresh: \|done line − motion end\| | 103 / 554 | 105 / 521 | 984 / 4350 | 288 / 2294 |
A 900 ms bound gave 435 / 1169 (4 of 12 known moves started at the bound, before the words).
