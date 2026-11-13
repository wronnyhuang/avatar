# Scene: chalkboard, wall TV, live surfaces, station camera

Readers: the teaching layer (hand + motion), the TV illustration agent (docs/TV-AGENT.md), the board brain (strokes in
board uv, docs/BOARD-BRAIN.md), the object layer (prop slots, docs/OBJECTS.md).
Units as in docs/INTENT-RULES.md: **H** = 2.30 scene units (muse body height); world frame origin on the floor under
the muse's centre at rest, x screen-right, y up, z toward the camera. "su" = scene units (only for reference).
Everything below is also in `renderer/animals/muse.props.json` (places `board`, `board_tray`, `tv`, props `chalk`,
`eraser`) and reaches the page as `config.props` — **code reads the JSON, never these numbers**. Where this document
and the JSON disagree, the JSON is the source of truth.

## 1. Layout (top view, camera at z = +3.35 H looking at -z)

```
 back wall z=-1.478 H   [print]                          [====== TV ======]
                                                            x 0.78..1.61 H
        fig   ______CHALKBOARD______  (free-standing, left-back of home, turned 15deg toward screen-right)
                 writable 1.22 x 0.61 H (2.8 x 1.4 su), centre (-0.54, 0.55, -1.09) H
                                                              desk (1.11,-0.65) (movable)
                                   HOME (0,0,0)                chair (0.65, 0.18) (movable)
                                                                     side table + headphones (1.00, 0.43)
```
* HOME (the rest spot): `home: {pos:[0,0,0], yaw:0}`. Desk, chair, laptop, mug, ball and plant have their own homes.
  The side table (with the headphone stand, the headphones and the book) stands FRONT-RIGHT at (1.0, 0.4348) H so
  nothing hides the board from the home or board camera; every field that depends on it (place centre/spots,
  headphones + book home pos, headphone handles' world_home, pick_up_from.stand_at) is placed there. The back wall
  carries the TV and a print (set geometry, muse only).
* The board stands left-back of home: from the home view most of it shows left of the muse; at the board station the
  board camera frames the whole board with the muse beside the stroke it writes.
* The TV hangs on the back wall right of home, above/behind the desk.

## 2. `board` place (muse.props.json `places[]`, `kind: "chalkboard"`)

```
"plane": {
  "origin": [-1.1314, 0.8565, -0.9294],   // H, world: TOP-LEFT corner of the WRITABLE area, on the slate's front face
  "u":      [ 0.9659, 0.0,   -0.2588],   // unit, along the board left -> right as seen from the front
  "v":      [ 0.0,   -1.0,    0.0   ],   // unit, DOWN the board
  "n":      [ 0.2588, 0.0,    0.9659],   // unit, out of the slate (toward the room, a little to screen-right)
  "size":   [ 1.2174, 0.6087 ]           // H: writable width (along u), height (along v)  (= 2.80 x 1.40 su)
}
```
* **Board uv** (what the board brain emits, what `surfaces.get("board")` takes): `(0,0)` = top-left of the writable
  area, `(1,1)` = bottom-right, x along `u`, y along `v` (down). World point of uv `(a, b)`:
  `P = origin + u*a*size[0] + v*b*size[1]`. **Aspect 2.0**: 1 uv-x is twice as long as 1 uv-y — a round circle in uv
  has x radius = y radius / 2. Pixel of the 1024x512 surface = `(a*1024, b*512)` (square pixels).
* Widths and radii in the surface API are in **v units** (fraction of the writable HEIGHT, 1.40 su). Digit height
  ~0.08–0.10 v (8–10% of the board height) is the legibility target (section 6).
* **Reachable band**: the writable area spans y 0.248–0.857 H (0.57–1.97 su); the standing muse's chalk reaches
  y 0.278–0.661 H = **board uv-y 0.32–0.95** (`station.reach_uv_y`). Above uv-y 0.32 is out of reach (the board brain
  should be told; a future step stool / stretch could open it).
* Frame: slate margin 0.04 su + oak frame 0.05 su round the writable area; the slate is dark green-charcoal with chalk
  dust; a tray along the bottom edge; two oak posts with splayed feet stand it on the floor.
* `station`: where the muse stands to write (section 4). `camera`: the station camera (section 6).

## 3. `board_tray` place (kind `board_tray`, part of the board) and the `chalk` / `eraser` props

* Tray top surface y = 0.2087 H (0.48 su), along the bottom of the frame, 0.10 su deep in front of the slate face.
  `spots` (H, world) = chalk spot, eraser spot, two spare spots. "Put it back" = place on `board_tray`.
* Props (CATALOG kinds in `renderer/engine/props_defs.py`, models in `renderer/engine/props.glsl`):
  * **chalk** — slot **14** (prop slots 8–13 are the object layer's, 14–15 the scene's; see docs/OBJECTS.md).
    White stick, length 0.150 su (0.0652 H), radius 0.0165 su. `grip: "pinch"`. Unit frame: the stick lies along its
    own **+y**; grip anchor `(0,-0.35,0)` unit (a little behind the middle). Home: lying on the tray along `u`.
  * **eraser** — slot **15**. Felt board eraser: oak back + grey felt pad, 0.18 x 0.065 x 0.076 su. `grip: "power"`.
    Unit frame: the FELT FACE is the **-y** face; grip anchor on the back (+y). Home: on the tray, felt down.
* **Chalk tip in the held frame** (props.glsl `propHeldIn`, char-local, before the root): when held
  (`prop14_hl` or `_hr` = 1) the stick's centre is
  `c = gp + pn*(0.0165 - 0.022) - R*GA*S + G*offs` (gp = museGrasp point, pn = palm normal, the pn term is -0.006 su:
  negligible), with `G` = the upright camera-facing frame (columns: `cross(up,fw)`, `up`, `fw` = horizontal direction
  from the paw to the camera, char-local), `R = G*Rcol`, `Rcol = prEuler(2π·(rx,ry,rz))·Rhome`, `S` = 0.075 su,
  `offs = H*(prop14_x,y,z)`.  The **tip** = `c + R*(0, 1, 0)*S`, i.e.
  **`tip = gp + G*(offs + Rcol*(0, 0.1013 su, 0))`** (0.1013 su = 0.0440 H from the grasp point along the stick).
  The teaching layer chooses `prop14_rx/ry/rz` (turns) to point the stick at the board and solves the hand target so
  the tip lands on the board point. `Rhome` = `prEuler(2π·home.rot)` (the rotation that lays it along the tray) stays
  in `Rcol` while held (docs/INTENT-RULES.md: a picked-up object keeps the way it lay), so pick rx/ry/rz such that
  `Rcol·(0,1,0) = Gᵀ·d` for the wanted stick direction `d` (char-local).
* **Eraser contact**: felt-face centre = `c + R*(0,-0.36,0)*S`, S = 0.09 su (grip anchor `(0,0.36,0)`):
  **`face = gp + G*(offs + Rcol*(0, -0.0648 su, 0))`**; erase footprint ≈ 0.18 x 0.076 su →
  `erase_rect` of 0.10 x 0.084 uv, or `erase_disc` r ≈ 0.05 v.

## 4. Stations (where the muse stands)

* `board.station` — writing: the muse stands to the RIGHT of the point it writes, 0.70 su (0.304 H) out from the
  slate along `n`, turned 30° toward screen-left (**root_yaw = -0.0833 turns**), writing with the **screen-left paw**
  (`hand_l_*`, the muse's anatomical right). For a writing column centred at board uv-x `a`:
  `S(a) = origin_floor + u*(a*size[0] + 0.2348) + n*0.3043` (H; origin_floor = origin with y = 0), e.g.
  `a = 0.5 → (-0.2379, 0, -0.8537)`, `a = 0 → (-0.8259, 0, -0.6962)`, `a = 1 → (0.3500, 0, -1.0113)`.
  JSON: `station: {pos (a = 0.5), yaw, side: "l", line: {from: S(0), to: S(1)}, aside: S(1.2), window_uv:
  [-0.10, 0.10], y_range_H: [0.278, 0.661], reach_uv_y: [0.321, 0.95]}`.
* **Reach window** at a station (estimated from the arm + head geometry, chalk tip included): board uv-x within
  `a ± 0.10` (~0.55 su of board) and uv-y 0.32–0.95; the head does not hide the chalk tip there. Writing left to
  right, the muse always stands right of the current stroke, so everything written so far stays visible; it shifts
  along the station line between columns (`a` steps of ~0.2) — the teaching layer verifies with its IK; upper rows
  need the paw high (hand_l_y ~1.0–1.3), lower rows a slight knee bend or body_pitch. `aside` = a spot just right of
  the board (S(1.2) = (0.585, 0, -1.074) H) to step to when presenting the finished board (nothing hidden).
* Writing stance in practice (app/teach.js): the stance is chosen per ~0.1 uv of a line so the body stands ~0.32 H
  along the board beside the pen (measured with the renderer: at 0.20 H the text by the pen is 0–10% visible, at
  0.32 H ~88%); the stick leans into the board, 0.7 up and 1.1 toward the written side (`chalkUp` / `chalkSide`), so
  the fist trails below-right of the tip instead of covering the newest letters; between items it steps along the
  station line (a in [-0.1, 1.25], beyond the board's right edge when needed) to the nearest spot where its fitted
  silhouette (`cfg.sil`, `tools/board_sil_fit.py`, IoU 0.90 vs rendered masks) hides <= 2% of the board's ink.
  Metric: `tools/board_vis_run.sh` (sim -> CPU renderer masks from this camera -> `tools/board_vis_metric.py`).
* Tray reach: the tray (y 0.21 H) is below a standing paw's comfortable reach — reach it with a slight crouch
  (root_y ≈ -0.03…-0.05 H, knees bend) or body_pitch toward it.
* `tv.station` — presenting: muse centre `(0.3478, 0, -0.8478)` H, **root_yaw = +0.07** (slightly toward the TV),
  pointing / gesturing with the **screen-right paw** at the screen. Nothing is touched there.
* Turning to face the user between items = root_yaw → 0 (or body_yaw / head_yaw), anywhere on the station line.

## 5. `tv` place (`kind: "wall_tv"`)

```
"screen": {"origin": [0.7783, 1.0826, -1.4522], "u": [1,0,0], "v": [0,-1,0], "n": [0,0,1], "size": [0.8348, 0.4696]}
```
H, world; origin = top-left of the visible picture. 16:9, surface 512x288; pixel (0,0) top-left. Thin black bezel,
mounted on the back wall (z -1.478 H). The picture is emissive (it lights nothing). Idle/off: dark glass with a soft
reflection of the room.

## 6. Camera (legibility)

* `cam_zoom` spans **[-1, 1]**: 0 = the default framing, (0, 1] = pull out, **[-1, 0) = push in**: the view tangent
  / 2^(−cam_zoom) (cam_zoom -1 = 2x tighter). The camera only turns + zooms about its fixed position (no dolly): the
  baked background stays exact (rest bake / panorama), no extra bake, lip sync untouched. `renderer/anim/rig.py`
  RANGES and `app/motion.js` RANGE accept -1..1 for cam_zoom.
* Each station place carries `camera: {cam_zoom, cam_x, cam_y}` (the view that frames the board + the muse writing,
  or the TV + the muse presenting). The board camera is `{cam_zoom -0.50, cam_x -0.18, cam_y 0.10}`: one framing
  shows the whole board AND the muse standing aside presenting it (no second camera move when it steps aside; the
  board keeps a ~40 px margin left at 960x540). The TV camera is `{cam_zoom -0.60, cam_x 0.565, cam_y 0.21}`.
* **Page API (app/moves.js): `engine.setView('board'|'tv'|null)`** — while a view is set, the camera block eases
  cam_zoom/x/y to that place's camera instead of following the root; `null` eases back to the normal follow/home
  framing. The ease is critically damped (~1.5 s to settle, per-frame step < 0.035 in cam_zoom) — a slow ease, never
  a cut. A station view also holds during a planned keyframe move (app/motion.js), so a gesture at the board does
  not pop the camera out.
* Camera use by the teaching layer / TV agent: `motion.setView('board')` when the walk to the board starts,
  `motion.setView('tv')` when the walk to the TV starts, `motion.setView(null)` when the walk home starts (or on
  abort).
* Board legibility target: a digit of 0.09 v ≥ 18 px tall at 960x540 and ≥ 9 px at 480x270 in the board view;
  ~12 px / 6 px from the home view. Checked with an LLM judge on rendered stills: 4 lines of digit height 0.09 v + a
  labelled triangle read correctly from the home, write and aside framings at 540p and 270p.

## 7. Surfaces (server `app/surfaces.py`)

```
from surfaces import Surfaces          # one per session: session.surfaces (created by the server; survives render relinks)
B = session.surfaces.get("board")      # 1024x512 RGBA (chalk colour + coverage over the slate)
B.draw_polyline(pts_uv, width, rgba=(240,240,232,235), t=None, ts=None)   # width in v units; ts = per-point times
B.erase_disc(center_uv, r, t=None, residue=0.08)     # r in v units; residue = faint chalk smear left (0 = clean)
B.erase_rect(uv0, uv1, t=None, residue=0.08)         # axis-aligned in uv
B.clear(t=None)                                      # wipe everything
B.snapshot() -> np.ndarray (512,1024,4) uint8        # what the board shows once every pending op is applied
B.ink_grid(cols=16, rows=8) -> np.ndarray float      # coverage per cell (free-space search for the board brain)
T = session.surfaces.get("tv")         # 512x288 RGBA (alpha = picture on; 0 = TV idle/off)
T.put_frame(rgb_uint8_HxWx3 (any size: resized to 288x512), t=None)
T.clear(t=None)                        # back to idle/off
```
* `t` = the PLAYBACK TIMELINE second (the same clock as the page's 'R' rig frames: `motion` time / the server's
  playback time); the renderer applies an update when the frame it renders reaches `t` — so a chalk segment appears
  exactly when the hand gets there. `t=None` = apply now. Per surface, `t` is made non-decreasing (an op stamped
  earlier than its predecessor is applied with the predecessor's time) so a later message never shows an earlier op
  too soon. `ts` (list, one per point) splits a polyline into per-segment ops, each appearing at its own time — the
  teaching layer reports the chalk tip's progress this way (`ts[i]` = when the tip reaches `pts_uv[i]`).
* Chalk look: soft-edged, slightly grainy strokes (grain is added in the shader, so the canvas holds smooth coverage);
  the default rgba is chalk white; any colour is allowed (coloured chalk).
* The server keeps the canvases for the whole session; a new / re-created render link (render child restart,
  character switch back to the muse) gets the full canvases re-sent automatically. Other characters ignore surfaces.
* Thread-safety: call from the server's event loop (async code). Each op returns the message bytes it queued (tests).

### Wire: 'S' surface message (server → renderer; same WS as audio / rig)

```
'S' : 0x53, uint8 surface (0 board, 1 tv), uint8 codec (0 raw, 1 zlib, 2 jpeg), uint8 flags, float64 t,
      uint32 seq, uint16 x, uint16 y, uint16 w, uint16 h, payload          (24-byte header, little-endian)
flags bit0 = t is "now" (apply at the next frame), bit1 = full-canvas resync, bit2 = clear the surface first
payload: the w x h RGBA8 rows of the rect, top row first (codec 0/1); codec 2 = a JPEG of the rect (RGB, alpha 255)
```
The renderer (`renderer/avatar/surfaces_rx.py`) keeps a mirror of each canvas and uploads only dirty rects to the GPU,
at most `AVATAR_SURF_BYTES` (default 1 MiB) per frame (larger updates spread over the next frames). Typical sizes: a
chalk segment ~0.3–1 KB, a TV frame of flat-colour illustration 15–40 KB (zlib), JPEG for photographic frames.

### Test hooks

* `tools/surface_demo.py --url http://127.0.0.1:<port>/?k=<key>` drives a running server's debug endpoint (below) to
  write "2+2=4" over a few seconds and play a moving test pattern on the TV.
* Server debug endpoint (key-guarded like the other debug routes): `POST /debug/surface?k=<key>` with
  `{"surface":"board","op":"draw_polyline","pts":[[u,v],...],"width":0.012,"t_rel":0.5}` (`t_rel` = seconds after the
  current playback time; ops: draw_polyline, erase_disc, erase_rect, clear, put_frame_test).
* `renderer/tools/scene_stills.py --out DIR [--backend cpu|gpu] [--views home,board,write,tv,present,desk,wide]`
  renders stills with written boards and TV pictures; `renderer/tools/scene_server_test.py --out DIR` drives the REAL
  render server (CPU) with server-made 'S' messages and checks they appear exactly at their t;
  `tests/test_surfaces.py`, `tests/test_surfaces_rx.py`, `tests/scene_view_test.cjs`.
* Page: `motion.setView('board'|'tv'|null)` (app/motion.js -> app/moves.js engine.setView).

## 8. Performance notes

Measured on an NVIDIA K620 at 480x270, JPEG q75, encode included, with a board chalk rect every frame + a full TV
frame every 2nd: with 8 prop slots, home 26.5 ms, board view 30.7, TV view 34.5, walk 31.0 ms (>= 29 fps everywhere);
the surfaces, chalk and eraser cost little. With all 16 prop slots (the object layer's models compiled in) the frame
rose to ~45–50 ms (~20 fps): the cost tracks the number of compiled-in prop models (looks like shader register
pressure, not per-pixel work), so the 16-slot build has to stay under 41.7 ms to hold 24 fps on that card.
