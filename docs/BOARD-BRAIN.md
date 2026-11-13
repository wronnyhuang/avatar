# Board brain

`app/board_brain.py` + `app/board/` turn "write this on the board" into chalk strokes. A fast text model (Claude Sonnet
through `claude_door` (`docs/CLAUDE-DOOR.md`) — the default: more correct AND faster warm than Haiku; Haiku by config) decides WHAT to write, in what order, roughly where, and the line
Muse says before each item; deterministic code draws everything (handwriting font, math layout, diagram layout,
figures, placement, erase, timing). Pure Python, no new pip deps, no network besides the door.

## Calling it (from the teaching layer)

```python
from board_brain import BoardSession, prewarm, time_parameterize, chalk_at
prewarm()                                   # once when a session that may teach starts (warm door processes)
sess = BoardSession(aspect=2.0)             # one per session: remembers what is on the board

async def on_item(it):                      # called as soon as each item is ready (streaming)
    queue.put_nowait(it)

items = await sess.plan(context_turns, request_text, on_item=on_item)   # also returns the full list
```
* `context_turns`: last turns, `[{"role": "user"|"assistant", "text": ...}]` (or `(who, text)` pairs); the last 8 are
  used. `request_text`: the voice model's request (free text, any subject).
* Stateless form: `await board_brain.plan(context, request, board_state, on_item)`; `board_state` is
  `sess.state()` / `stats["board_state"]` from an earlier plan.
* First item arrives ~1.0 s after the call with a warm door on Sonnet (see Latency); the rest of the plan follows
  as one laid-out batch (~1-3 s later) while the first one is being said and written, so the performance never waits.
* Abort / new topic: the model assumes every planned item gets written. Call `sess.mark_done(item)` for each item
  actually written (fully or partly) or erased; on abort call `sess.abort()` and the board model is rebuilt from
  exactly those. If the board is wiped outside the brain, `sess.clear()`.

Performance loop per item (the cadence: say → write → say → write):
1. Speak `say_before` (may be None). 2. Walk/shift so `bbox` is in reach (station window ±0.167 uv-x, see
`docs/SCENE.md` §4). 3. Trace: `tl = time_parameterize(it["strokes"], aspect=2.0, t0=t_start, speed=1.0)` →
`[(t, u, v, pen_down)]`; drive the chalk tip with `chalk_at(tl, t)` and feed the board surface per stroke
`draw_polyline(pts, width=0.011, ts=[t of each point])` so marks appear under the tip. 4. Optional `say_after`
(currently always None — a closing remark comes as a separate `say` item).

## Item shape

```
{"id", "kind", "say_before": str|None, "say_after": None,
 "strokes": [[(u, v), ...], ...],     # board uv: 0..1 x 0..1 of the WRITABLE area, y down; draw order = hand order
 "bbox": [u0, v0, u1, v1], "est_s": float, "text": source,
 "size": cap height in v (text/math/column), "eq_x", "baseline", "warnings"?}
```
* kinds: `text`, `math`, `column`, `diagram`, `figure` (content); `underline`, `box`, `circle`, `strike`, `check`,
  `highlight`, `bracket`, `arrow` (annotations, carry `target: [id]`); `erase`; `say` (no strokes, bbox None,
  just speech).
* `erase` items: `target` = ids removed (annotations of a target go with it), `bbox` = wiped region, `strokes` = the
  eraser CENTRE path (back-and-forth wipes, kept inside the region), `erase_wh` = felt-pad footprint in uv
  (default 0.10 x 0.084 = SCENE's eraser) → `erase_rect(c - wh/2, c + wh/2)` along the path, or `erase_r`
  (v units) for `erase_disc`. `auto: true` = the brain made room by itself (say_before "Let me make some room.").
  `time_parameterize` works on the eraser path too.
* Every stroke is a polyline with points <= ~0.006 v apart (curves resampled; long straight lines resampled too);
  strokes are ordered like a hand writes (left to right, glyph stroke order; fraction = numerator, bar, denominator;
  diagrams node by node with the incoming arrow just before each node).
* Guarantees from code (not the model): every mark stays inside the chalk's REACH BAND — v in
  [0.321, 0.95] (SCENE `station.reach_uv_y`; `BOARD_REACH_V`), u in [0.015, 0.985]; content items never overlap
  (gap 0.036 v, plus reserved room round an item a later box/circle/underline marks); text/math cap height >= 0.08 v
  (fraction digits 0.85x, scripts 0.68x); diagram/figure labels >= 0.065 v, figures >= 60% of the band tall.

## Layout (deterministic)
The model gives ORDER and rough hints; `app/board/layout.py` decides every position:
* Column flow: items go down the current column; a full column continues in a new column to the right (scanning
  below anything wide that spans it); `at=right` continues the same row; a continuation that starts with `=`/`≈`/`⇒`
  goes on the row of the line it continues when the column is full; a flow item hinted `top-right` starts the next
  column (it never jumps across the board); figures/diagrams go to their region word or the nearest free spot.
* When nothing fits: less reserved room → wrap text narrower (balanced lines, no lonely last word) → shrink to the
  0.08 minimum → split a long equation chain at its relations (`a = b = c` → aligned lines; `x = …, x = …` → one per
  line) → a free hole → MAKE ROOM: earlier lessons are erased oldest column first; within the same lesson it is a page
  turn (everything of this lesson except its heading; writing continues under the heading). Erase items for this
  carry `auto: true`, `say_before: "Let me make some room."`.
* Columns: a new column starts under a one-line heading, right of the column content (a wide title does not push
  it); once it has its second item a ruled `divider` item is drawn between the columns (kind `divider`, plain
  strokes, `target` = the column's first item, erased with it). A column-height knob (1.0 / 0.8 / 0.62 of the band)
  balances columns so an answer does not sit alone at the top of a new column.
* A plan whose first content item is a figure/diagram is held whole and laid out as one batch; a lone picture on an
  empty board is centred and as tall as the band.
* Two-phase streaming (`board_brain.Feeder`): everything up to and including the FIRST content item is placed and
  emitted at once (writing starts immediately). The rest of the plan is held and laid out as one batch when the
  stream ends — or when the hold deadline passes (0.6 x the say+write time of what was already emitted, min 1.5 s;
  `BOARD_BRAIN_HOLD_S` overrides). The batch searches knobs (base size 0.115..0.08, text wrap width 0.48/0.66/0.95 of the
  board, figure scale 1.2/1/0.82, column height) by coordinate descent on geometry-only copies of the board and
  keeps the best score (page turns, lone columns, wraps, orphans, shrinking penalised; bigger writing preferred).
  p50 120 ms, p90 0.3 s, worst seen 0.7 s.
  For the teaching layer: items 2..n usually arrive together, 0.3-3 s after item 1, while item 1 is being said and written.
* Arithmetic self-check (`app/board/arith.py`): plain numeric chains in math/text (`60+80=140`,
  `\frac{36}{6}=6`, `730/140 \approx 5.21`, `60(5.21)`) and column sums/differences/products are evaluated exactly;
  a wrong FINAL literal is replaced by the right value (warning `arith fixed: a -> b` on the item), other mismatches
  are reported as `arith mismatch: …` warnings. Anything with letters is skipped.
* Targets for marks and erase: `ID`, `ID:TEXT` (characters inside the item; may contain spaces, e.g.
  `target=m4:x = 4`), `#n` (n-th content item in writing order — the board listing in the prompt is numbered),
  `last`/`that`, `answer` (the latest boxed item, else the last item). Marking a `column` item without `:TEXT` marks
  its last row (the answer). Check marks and arrow labels are placed clear of other ink; an arrow label that repeats
  the note it points at is dropped.

## Model output format (streaming, one item per line)

```
KIND ID key=value ... | say: SAY | write: CONTENT
math m2 at=below:m1 align=eq | say: Now add the tops. | write: = \frac{5}{6}
```
The system prompt is `board_brain.SYSTEM`. Parser is forgiving: id before kind, missing labels (positional
`| say | content`), swapped say/content for math, missing SAY, "Muse says:", `$...$` around math, numbering/bullets,
code fences, a diagram/figure spread over several lines (held until the next item line), unknown kinds dropped.
Content languages:
* `math`: LaTeX subset (fractions, scripts, roots, ∫ Σ ∏ lim with limits, sized delimiters, matrices/cases,
  aligned `\\` + `&`, accents/vec/overline, cancel, boxed, `\xrightarrow[below]{above}`, Greek and symbols, `\ce{}`
  chemistry with states/charges/condition arrows, Unicode ² ₂).
* `column`: `347 ; x 28 ; --- ; 2776 ; 6940 ; --- ; 9716` (right-aligned digit grid, `c:` carry rows).
* `diagram`: Mermaid on one line (`graph TD|LR; A[..] --> B{..}; B -->|yes| C((..))`, all common shapes/edges,
  subgraphs, `timeline`, `mindmap`) → own layered layout (`app/board/diagram.py`, ~20–30 ms for 10 nodes).
* `figure`: vector DSL in math coordinates (y up), see `app/board/figure.py` docstring: view, axes (ticks), plot
  "expr", param, line/arrow/polyline/polygon/rect/circle/ellipse/arc/curve/bezier, dot/point/label, angle,
  rightangle, numberline. Labels avoid ink and each other automatically.
* `text`: words with inline `$math$`; wraps at half the board by default.

## Placement keys
`at=next` (default, below the previous item) | `right` | `below:ID` | `right:ID` | region words
(`top-left`, `top-right`, `left`, `right`, `center`, `bottom-*`) | `U,V`; `align=eq|right|center`;
`h=small|medium|large|<v>`; `wh=W,H` (figure/diagram box, fractions of the board); `w=` text wrap width.

## Latency (warm door, serial, 12 requests: `tools/board_latency.py`)
| model | first token p50/p90 | first item p50/p90 | whole plan p50/p90 |
|---|---|---|---|
| sonnet | 678 / 769 ms | 1013 / 1096 ms | 2156 / 2831 ms |
| haiku | 2711 / 3368 ms | 2873 / 4159 ms | 3510 / 4159 ms |

## Timing (`app/board/timing.py`)
Distances in board heights (u scaled by aspect). Pen-down: trapezoid speed profile (0.9 bh/s cruise at 0.09 v
letters, 12 bh/s² accel), splits and stops at corners > 65°, small writing slower (∝ √size). Pen-up travel 1.6 bh/s,
plus 0.03 s lift/touch per stroke. ≈0.6 s per letter at the default size. `est_s` = timeline end + settle.
Samples: segment i-1→i is drawn iff sample i has `pen_down`; the first sample of every stroke is pen-up (arrival).
`chalk_at(tl, t)` → `(u, v, pen_down)`; `drawn_until(strokes, tl, t)` → partial strokes (previews).

## Config
| env | default | meaning |
|---|---|---|
| `BOARD_BRAIN_MODEL` | `sonnet` | `sonnet` / `haiku` / `opus` through claude_door |
| `BOARD_DIVIDERS` | `1` | rule a vertical line between a lesson's columns (`0` = off) |
| `BOARD_BRAIN_MAX_S` | `25` | hard cap per plan (door max_s) |
| `BOARD_BRAIN_SPEED` | `1.0` | chalk speed multiplier for est_s (the teaching layer may also pass `speed` to time_parameterize) |
| `BOARD_REACH_V` | `0.321,0.95` | reachable band of the board in v (SCENE `station.reach_uv_y`); `0,1` = whole board |
| `BOARD_BRAIN_HOLD_S` | (auto) | max wait after the last emitted item before the held rest of the plan is laid out anyway |
Board geometry: `Board(aspect=2.0, min_size=0.08, default_size=0.09, label_size=0.065, margin=0.03, gap=0.036,
reach=(0.321, 0.95))`; handwriting is condensed to 0.86 of the Hershey width (`font.XSCALE`).

## Tools / tests
* `tools/board_preview.py --request "..." --out b.png [--gif reveal.gif --mp4 reveal.mp4] [--model sonnet]` or
  `--items plan.json`: slate canvas 1024x512 (the out-of-reach top strip drawn darker), chalk 0.011 v, erase with the pad footprint, progressive reveal on the
  same time model.
* `tools/board_eval.py --model haiku --out DIR [--jobs 4]`: dev set `eval/board_dev.json` (51 requests across
  subjects, worked problems, diagrams, figures, follow-ups on an existing board, non-English, a long request) → PNG
  per case (+ `_story.png` with the board before each erase) → Claude vision judge (sonnet). `--replay SRC DST`
  re-renders stored model outputs with the current code (no brain calls), `--rejudge DIR`.
* `python -m unittest tests/test_board_brain.py` (25 tests), `python tests/board_diagram_test.py`.
* Fonts: Hershey single-stroke (public domain) vendored in `app/board/fonts/` (see README there).
