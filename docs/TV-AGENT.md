# TV agent — the wall TV's live illustrations

Muse's wall TV shows animated diagrams (not video) of whatever is being taught. A fast model writes a small
declarative program in **TVL** (the TV language); a deterministic renderer in its own process turns it into frames on
the `tv` surface (`docs/SCENE.md`). While the TV is on, a conversation tap asks the same model for **patches**, so the picture
keeps up with the talk ("make the focal length shorter" changes one parameter, which glides).

```
voice model --tv_show({concept})--> page (voicetools.js) --{type:'tv_show'}--> server Session.on_tv
   server: TVAgent.show -> claude_door (haiku) -> TVL lines --stream--> render worker (process) -> frames
        -> surfaces.get("tv").put_frame(rgb)       (server event loop; only changed frames)
   server transcripts (_on_transcript) -> TVAgent.hear -> gate (model may answer `none`) -> patch lines
   server -> page {type:'tv_state', ...} -> silent note to the voice model + window.__avatar.tv()
```

## Files

| file | what |
|---|---|
| `app/tv_agent.py` | `TVAgent` (show / hear / clear / describe / prewarm / stats / close), `RenderHost` (render child), `hedged_stream`, `replay()` (timeline -> frames, eval + recordings), `FileSink` (dev) |
| `app/tvdsl/expr.py` | safe expression evaluator (tokenizer + Pratt parser -> closures; no `eval`; NaN on any failure) |
| `app/tvdsl/lang.py` | line parser (tolerant), colours, `Doc` (the live document: params glide, replaced objects morph, fades, scene replace) |
| `app/tvdsl/scene.py` | evaluates a `Doc` at a time into marks; all object kinds; generic optics tracer |
| `app/tvdsl/render.py` | anti-aliased rasterizer (OpenCV LINE_AA + Pillow text), emissive glow, label placement |
| `app/tvdsl/worker.py` | the render loop process (stdin JSON lines in, changed RGB frames out) |
| `app/tvdsl/system_prompt.txt` | the agent's system instruction: TVL spec + 11 worked examples |
| `tools/tv_eval.py`, `eval/tv_devset.json` | live eval (46 concepts + 6 live-update sequences), Claude-vision judge |
| `tests/test_tv.py` | unit tests (`python -m unittest tests/test_tv.py`) |

Deps: numpy, OpenCV (`cv2`, headless is fine), Pillow (all in `requirements.txt`); DejaVuSans from
`/usr/share/fonts/truetype/dejavu` (falls back to Pillow's default font). Runs in the app server (the server owns the pixels).

## Server wiring

* `app.py`: `Session.tv` (muse only, `cfg["tv"]`), created + **prewarmed** in `tv_attach()` at session start;
  messages `{type:'tv_show', concept, mode?}` / `{type:'tv_clear'}`; `_on_transcript(..., final=)` feeds the tap
  (page finals `final=True`; server-voice deltas are joined and taken after 1.2 s quiet); closed with the session;
  `report()["tv"]` = stats + log.
* `rtc.py`: `tv_tools()` (`tv_show`, `tv_clear`) appended for characters with a scene (`act_for`), in `planner_tools`
  and `hub_act_tools`.
* `voicetools.js`: `tv_show` / `tv_clear` in `NAMES`; `onHub({type:'tv_state'})` turns states into silent notes
  (`shown` -> "The TV now shows: …"; a tap `updated` -> rate-limited note; `failed` -> honest note).
* `page.html`: `case 'tv_state'` keeps `D.tvState` and forwards to `D.vt.onHub`; **`window.__avatar.tv()`** returns it.

Config (`cfg["tv"]`, all optional): `enabled`, `model` ("haiku" default, "sonnet"; env `AVATAR_TV_MODEL`),
`patch_model`, `fps` (12), `tap` (true), `tap_min_words` (3), `tap_debounce_s` (0.6), `tap_min_gap_s` (1.5),
`hedge_s` ([1.5, 3.0]), `max_s` (25), `prewarm` (true), `process` (true), `review` (None; "sonnet"/"haiku" = after a
new picture a second model reads the program as text and sends fixes — measured no gain on 20 hard concepts, so off;
env `AVATAR_TV_REVIEW`). Prompt A/B: env `AVATAR_TV_PROMPT=<file>`.
`upgrade` ("sonnet" default; env `AVATAR_TV_UPGRADE`, `none` = off), `upgrade_max_s` (12): **sketch + careful version** —
a new picture streams from the fast model at once (first lines ~1.2 s) while the stronger model writes the same request
in parallel; if it finishes in time with a real picture (>= 3 objects) it replaces the sketch (worker crossfade,
`tv_state` updated with `source:'upgrade'`, no note to the voice model). Live taps wait for it, so the user's changes land on
the final picture; anything new (show / clear / a tap) cancels it. Cost: two subscription calls per new picture.

### `tv_state` (server -> page; also what `window.__avatar.tv()` returns)

```
{type:'tv_state', status: drawing|shown|updated|unchanged|failed|cleared, phase: 'drawing'|'showing'|'idle',
 scene_id: int (+1 per new picture), title, description (one line, what the TV shows), n_patches (since the scene),
 updated_t (unix s), source: 'tool'|'tap', id, ms_first, ms_done, words?, error?}
```
Before any message the page answers `{phase:'idle', scene_id:0, …}`.

### Hooks for the teaching layer

* `d.onTv(name, args)` — optional dependency of `createVoiceTools`: called on `tv_show` / `tv_clear` before the server
  request (walk to the TV station, `engine.setView('tv')`, point at the screen; on clear walk home / `setView(null)`).
* Session-instruction wording (proactive TV use, "say what you'll show, then show it") belongs to the teaching layer. The tool descriptions
  (rtc.py) are deliberately short: concept = the idea plus every number the user gave; `mode:'update'` only for a change the
  TV has not made by itself.
* Abort / "turn it off": `tv_clear` (or `Session.tv.clear()` from a harness) -> renderer goes idle -> `surface.clear()`.

## TVL in one screen

One statement per line; a bad line is skipped; the last good picture stays.
`scene "desc" [view=] [loop=]` · `param N = v [lo..hi]` (glides when changed; a sign flip snaps) · `let N = expr` ·
`func f(x) = expr` · `KIND ID key=value … ["text"]` (same ID again morphs) · `for i=0..N …` (≤120) · `set` · `del ID|ID*`
· `note "…"` · `clear`.
Expressions: + − * / ^ %/mod, comparisons, `c ? a : b`, points `(x,y)` with vector + − and scalar *, lists, short
string literals (`{a<0 ? 'opens down' : 'opens up'}` in a label: words that stay true while params change); math,
animation (`lin ramp pulse osc saw tri steps cycle`), geometry (`polar rot dir dist mid angle pos pt end`), physics
(`lensmaker image snell critical cauchy`), `root(EXPR_OF_x, a, b[, k])` (bisection: intersections / equilibria stay
exact), 3D (`p3(x,y,z)` / `p3(vsepr(n,i), len)` orthographic projection seen from the variables `yaw`/`pitch`,
`z3(..)` depth for `layer=`, `vsepr(n,i)` electron-domain directions 2..6). Kinds: line arrow path poly bezier circle dot ellipse arc rect star text plot
area curve axes(+`box`, objects `on=`) grid angle brace wave spring field atom bond group lens mirror glass screen ray
beam rays3 image. Optics is generic: every ray is traced through every lens (ideal thin, or `real=1` true refraction at
its two spherical faces), mirror (flat / ideal curved) and glass polygon (Snell, total internal reflection, dispersion
via `disp`, default 0.0042 = real glass), so the model only draws incoming light. `glass inner=N` follows the internal
(partial) reflection N times before the ray leaves (raindrop / rainbow path).

Generic picture mechanics (no per-topic code):
* **Auto view fit** — solid shapes and texts that would leave the screen pull the view out (≤ 1.6x, eased 0.6 s).
* **Axis numbers** never move (a shuffled scale is wrong); one that would overlap a text is dropped. An `axes` whose
  ranges cannot be world units (they would leave the screen) becomes a data chart automatically.
* **Labels**: all free texts are drawn after the geometry (with a dark halo), automatic labels slide off lines; a free
  text repeating an object's own label is dropped; an atom's symbol stays with its atom (depth order).
* **Text layout** — every text is laid out before any is drawn, in priority order: the program's own
  texts, callouts (as one set), automatic labels, axis numbers, axis titles. No text below 12 px at 512 wide (scales
  with the width); a too-wide text shrinks a little, else wraps at spaces; all stay on screen. Each text keeps its spot
  unless it sits on lines / straddles a filled shape / overlaps an earlier text or a leader, then it takes the nearest
  free spot (scored in O(1) from two summed-area tables; last frame's offset is preferred so labels do not flicker).
  Axis numbers are thinned (every 2nd/5th…) instead of colliding; big ones read 250k / 1.5M. `Renderer.last_texts` /
  `last_leaders` record where everything went.
* **Primitives** — `callout` (labelled parts: labels in the margins beside the picture, stacked in anchor
  order so leaders do not cross, wrapped to the margin), `flow` (process / cycle boxes on a ring / row / column, the
  active step lit with a dot running along its arrow, `node(ID,i)`), `ripple` (point-source wavefronts, additive),
  `wave mode=standing`, and `slit1 slit2 grating fringe`. The model-facing lines are in `app/tvdsl/prompt_prim.txt`.
* **Sweep** (opt-in, `param a = 1 [-2..3] sweep`) — a picture in which nothing else moves continuously explores that
  param by itself: the declared value for 3.5 s, out to the far end of its range and back (12 s). It stops for good
  once anything is changed (`hold`, sent before every update); the description says "(it slowly varies a across
  -2..3 and back to 1)". It was once automatic (first ranged param): pictures moved more, but a sweep crosses
  regimes under fixed captions (object from beyond 2F to inside F while the caption says "real, smaller image"), so the
  judge — rightly — failed them; now the model opts in only when every label is computed and the idea holds over the
  whole range.

Safety: no Python eval; expression length ≤ 600, depth ≤ 48, lists ≤ 400, for ≤ 120, objects ≤ 600, user-function
recursion ≤ 6; overflow/NaN never raises; the renderer culls non-finite or absurd geometry.

## Fail-safe and timing

* A new scene is **staged**: its `scene` line is held until the first object line parses, so a stream that yields
  nothing usable keeps the previous picture (`status: failed`).
* An UPDATE that starts with a `scene` line is held to the end of its stream: >= 4 objects = a real new picture
  (replaces the TV); fewer = a fragment leaning on the current picture's names, applied as a patch (its title becomes
  the note). (Sonnet often re-titles while patching; replacing wiped the picture and left `?` labels.)
* Every update/review commit starts with `hold`: the automatic sweep stops and the swept param glides back.
* The loop is never shorter than the longest `steps()`/`cycle()` sequence (+1.5 s), so late stages are reached.
* A new `tv_show` supersedes anything in flight; tap patches supersede each other (words are merged), and wait while a
  new scene is still being drawn.
* Hidden "thinking": Haiku used to think 5–15 s before its first word on ~25–50 % of requests (prompt-dependent;
  `--effort low`, `MAX_THINKING_TOKENS=0`, `alwaysThinkingEnabled:false` did not stop it). One sentence in the system
  prompt ("Answer instantly: never deliberate…") removed it (18/18 fast). `hedge_s` additionally launches a duplicate
  request if no text arrived by 1.5 s / 3.0 s and keeps the first to speak.
* Frames are diffed and JPEG-encoded on the pump thread (`surfaces.Surface.encode_frame`, additive in
  `app/surfaces.py`); the server loop only stamps seq/t, updates the canvas and sends (`commit_frame`, < 1 ms; plain
  `put_frame` was ~9 ms p50 / 18 ms p90 on the loop).
* Long voice-model concepts ("show … at various positions …") made Haiku think silently for 3–5 s; the new-scene
  prompt now puts the request last followed by "One picture; several cases become stages over time. Write the scene
  line now, no planning." (first token ~0.7 s again), and the `tv_show` schema asks for ONE short sentence.
* Renderer: own process (`python app/tvdsl/worker.py W H FPS`, nice 5, 1 OpenCV thread), 12 fps, sends only changed
  frames; ~10–30 ms per 512x288 frame for busy scenes (≈ 0.15–0.35 core while animating, ~0 when static/idle).

## Follow-ups and the careful version

* **Patches on Sonnet, Haiku as the hedge, Sonnet corrects** — a follow-up (tap or `tv_show mode=update`) is written by
  `patch_model` (sonnet). If no text arrived by `patch_hedge_s` a Haiku duplicate starts; whichever speaks first is
  shown at once. When Haiku wins, the Sonnet request is NOT cancelled (`hedged_stream(keep=)`): its patch lands as a
  correction (`kind: correct`) — the picture morphs to *(picture before the user's words) + (Sonnet's patch)*, Haiku's extra
  objects fade out. A cut-off/failed stream is never applied. (In a follow-up eval, Haiku-won patches passed 1/10 strict, Sonnet's 13/26.)
* **The user's changes accumulate** — every applied change `(words, note)` since the last new picture is listed in the patch,
  escalation and careful-check prompts ("EARLIER CHANGES … keep every one"): a later rewrite had quietly undone
  "increase demand".
* **Verify the drawing changed, then escalate** — before/after are rendered on the SAME scene clock (the change's
  `hold` re-phases the loop, so a moving part used to read as "changed" when the patch drew nothing); geometry change
  below `geom_min` -> `fix_model` looks at stills and makes the change.
* **Values the user asks for are never clamped** — a `set`/`param` outside the program's guessed range widens the range
  ("twice as long" used to stop at the range top).
* **Lint after every careful version / patch, one repair round** (Haiku, corrected lines only): objects that draw
  nothing, texts printing `?` or their raw `{…}` template, objects entirely off the screen, and a graph curve that THIS
  change pushed off its axes box (graph curves are clipped at their box edge).
* **Careful version** — written in parallel by `upgrade` (now **opus**: devset 43 %/87 % vs sonnet 41 %/78 %, landed
  p50 6.4 s p90 9.5 s under load) and morphed in with `morph … restart=1`: shapes glide from the sketch but its clock
  starts at 0, so its stages play from the beginning (with the sketch's clock carried on, careful pictures sat at
  their last stage). A careful version cut off by `upgrade_max_s` (18 s) is never swapped in.
* **Engine fixes found by reading frames** — lets evaluate in dependency order (a morph redefining `th` with a new
  `T` froze a pendulum as NaN); a program's own `func` wins over a built-in of the same name (`func tri(u)` was the
  built-in triangle wave); a `real=1` lens is the glass its n and radii say (aperture shrinks below |R|) with F, F', 2F
  at the thick-lens focal points where the traced rays cross; a `flow` ring of long names stays on the screen.
* Evals: `tools/tv_follow_eval.py` (follow-ups: before/after judge, words -> visible change), `tools/tv_eval.py`
  (new pictures; `--set tv_devset | tv_holdout | tv_holdout2`).
