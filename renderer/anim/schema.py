"""Shared per-frame animation schema — the ONE contract between the audio pipeline and the renderer.

Every rendered frame is fully determined by (animal, frame index, this vector).  Order is the wire
format: the renderer receives `uniform float uP[NPARAMS]` and GLSL reads uP[P_JAW] etc.  Add new
parameters at the END only.  engine code generates the #defines from this list.
"""
FPS = 30
SAMPLE_RATE = 24000
CHUNK_MS = 100        # audio is handed in at this size (design target: ~200 ms or less)
LOOKAHEAD_MS = 67     # frame at time t may use audio up to t + LOOKAHEAD_MS, never more (2 frames @ 30 fps)

PARAMS = [
    # mouth, 0..1, from audio.  All zero = mouth fully CLOSED at rest; no interior may show.
    "jaw",        # jaw opening (open vowels a/ae high)
    "round",      # lip rounding / pucker (o, u, w)
    "spread",     # lip spread / wide (ee, ih, s)
    "press",      # lips pressed together (m, b, p closure) — overrides jaw toward closed
    "labdent",    # lower lip tucked to upper teeth (f, v)
    "tongue",     # tongue tip up (l, t, d, n, th)
    "smile",      # slow mood/corner lift, also idle
    # head, radians / scene units, small
    "head_yaw", "head_pitch", "head_roll",
    "head_x", "head_y", "head_z",
    # body + face life
    "breath",     # -1..1 chest/shoulder
    "blink_l", "blink_r",        # 0 open .. 1 closed
    "gaze_x", "gaze_y",          # -1..1 eye look
    "brow_l", "brow_r",          # -1 down .. 1 up
    "brow_knit",                 # 0..1
    "ear_l", "ear_r",            # -1 back/droop .. 1 forward/up (slow pose)
    "ear_flick_l", "ear_flick_r",# 0..1 fast twitch
    "cheek",      # 0..1 cheek puff/lift with speech
    "nostril",    # 0..1 nostril flare / sniff
    "voice",      # 0..1 raw voiced-ness/loudness, for secondary motion only
    # real lips, for the shape characters (docs/LIPS.md). The animals ignore these.
    "lip_open",      # 0..1 gap between the inner lip edges at the centre. 0 = lips touching
    "lip_upper",     # 0..1 upper lip lifted off the upper teeth
    "lip_lower",     # 0..1 lower lip drawn down off the lower teeth
    "lip_width",     # -1 narrow pucker .. +1 wide spread, corners move in x
    "lip_protrude",  # 0..1 lips pushed forward, off the face
    "lip_press",     # 0..1 lips pressed and rolled together (m, b, p). Overrides lip_open
    "lip_tuck",      # 0..1 lower lip tucked up under the upper teeth (f, v)
    "teeth_gap",     # 0..1 space between upper and lower teeth, independent of the lips
    "tongue_tip",    # 0..1 tongue tip up behind the upper teeth (t, d, n, l)
    "tongue_dental", # 0..1 tongue tip between the teeth (th)
    "tongue_back",   # 0..1 back of the tongue raised (k, g, ng)
    # Procedural torso/arms/paws are traced in the same scene as the face.
    "arm_l", "arm_r", "wave_l", "wave_r", "point_l", "point_r",
    "paw_open_l", "paw_open_r", "clap", "thumbs_up",
    "body_lean", "body_bounce", "tail_wag", "tail_raise", "body_yaw",
    # full-body rig (docs/CONTRACT.md, docs/RIG.md).  Appended in contract order.  0 = neutral rest unless noted.
    # _l = SCREEN-LEFT (the character's anatomical right), _r = screen right, for every side param.
    # Hand IK targets: offset from the rest wrist in units of one full arm length; x outward from the midline on that
    # side (negative = across the chest), y up, z toward the camera.  Range about -1..1.2.
    "hand_l_x", "hand_l_y", "hand_l_z",
    "hand_r_x", "hand_r_y", "hand_r_z",
    "hand_l_open", "hand_r_open",    # 0 relaxed soft curl .. 1 spread open palm
    "hand_l_turn", "hand_r_turn",    # -1..1 forearm roll; +1 = palm toward the camera when the arm is raised
    "thumb_l", "thumb_r",            # 0..1 thumb extended (thumbs-up = open 0 + thumb 1)
    "shoulder_l", "shoulder_r",      # 0..1 shrug raise
    "ear_l_turn", "ear_r_turn",      # -1 inward/forward-facing .. 1 outward (ear_l/ear_r stay the pitch)
    "nose_wiggle",                   # -1..1 nose tip down .. up twitch
    "tongue_out",                    # 0..1 tongue out past the lips
    "mouth_open",                    # 0..1 commanded jaw opening; the server uses max(lipsync, mouth_open)
    "eye_wide",                      # 0..1 upper lids raised (surprise)
    "squint",                        # 0..1 lower lids up
    "body_pitch",                    # -1 lean back .. 1 lean toward the camera
    "tail_curl",                     # 0..1 tail curls up / around
    # ---- whole-body root motion (docs/INTENT-RULES.md).  Wire values are int16 x 1e4 (|v| <= 3.27), so
    # angles are in TURNS and distances in BODY HEIGHTS (H = the character's seated height, renderer/props/<c>.json).
    # Character-local -> world: pw = T + C + R*(pl - C), C = the centre of mass, R = Ry(yaw) Rx(pitch) Rz(roll),
    # T = H*(root_x, root_y, root_z).  root_yaw/pitch/roll are PERIODIC (period 1): the mixer unwraps them, never clips.
    "root_x", "root_y", "root_z",      # body heights; x screen-right, y up (0 = seated on its floor), z toward the camera
    "root_yaw",                        # turns, + = the front swings toward screen-right (same sign as head_yaw)
    "root_pitch",                      # turns, + = the top tips toward the camera (same sign as head_pitch = nose down); backflip = -1
    "root_roll",                       # turns, + = the top toward screen-right (same sign as head_roll)
    "squash",                          # -1 stretched tall .. 1 squashed flat, volume-preserving, about the base point
    "gaze_free",                       # 0 gaze = offset from the camera direction (today) .. 1 offset from the head's own forward
    "cam_zoom",                        # 0 today's medium close-up .. 1 wide shot: whole character + ~1.5 H around it
    "cam_x", "cam_y",                  # camera aim offset, body heights (the controller follows the root with these)
    "stand",                           # 0 seated (today) .. 1 standing up on its hind legs / feet (renderer may not implement: see docs)
    # ---- hands: counting and grasping
    "fingers_l", "fingers_r",          # 0 off (normal hand) .. 5 = that many digits held up: index, middle, ring, pinky, thumb
    # ---- props: 6 slots per character (renderer/props/<character>.json).  All zero = every prop at home.
    #  free (hl = hr = 0): position = home + H*(x, y, z) in world; rotation = home rotation then R(rx, ry, rz) about its centre
    #  held: the prop sits in the paw's grasp frame; (x, y, z) a small offset and (rx, ry, rz) a turn in that frame
    #  hl/hr = attach weight to the screen-left / screen-right paw (both > 0 = two-paw hold / hand-over)
    "prop0_x", "prop0_y", "prop0_z", "prop0_rx", "prop0_ry", "prop0_rz", "prop0_hl", "prop0_hr",
    "prop1_x", "prop1_y", "prop1_z", "prop1_rx", "prop1_ry", "prop1_rz", "prop1_hl", "prop1_hr",
    "prop2_x", "prop2_y", "prop2_z", "prop2_rx", "prop2_ry", "prop2_rz", "prop2_hl", "prop2_hr",
    "prop3_x", "prop3_y", "prop3_z", "prop3_rx", "prop3_ry", "prop3_rz", "prop3_hl", "prop3_hr",
    "prop4_x", "prop4_y", "prop4_z", "prop4_rx", "prop4_ry", "prop4_rz", "prop4_hl", "prop4_hr",
    "prop5_x", "prop5_y", "prop5_z", "prop5_rx", "prop5_ry", "prop5_rz", "prop5_hl", "prop5_hr",
    # ---- the "muse" character (, docs/INTENT-RULES.md section 0)
    #  prop{k}_hh = attach weight to the HEAD (worn items: headphones); prop{k}_s = the prop's own articulation
    #  0..1 (laptop lid closed..open, book closed..open, box lid ...). Slots 6, 7 = two more props.
    "prop0_hh", "prop0_s", "prop1_hh", "prop1_s", "prop2_hh", "prop2_s",
    "prop3_hh", "prop3_s", "prop4_hh", "prop4_s", "prop5_hh", "prop5_s",
    "prop6_x", "prop6_y", "prop6_z", "prop6_rx", "prop6_ry", "prop6_rz", "prop6_hl", "prop6_hr", "prop6_hh", "prop6_s",
    "prop7_x", "prop7_y", "prop7_z", "prop7_rx", "prop7_ry", "prop7_rz", "prop7_hl", "prop7_hr", "prop7_hh", "prop7_s",
    #  legs (characters with legs: muse): world-axis offset (H) of each ankle from where the REST stance, carried by the
    #  full root transform, would put it. 0 = standing at rest under the body. _l = screen-left foot.
    "foot_l_x", "foot_l_y", "foot_l_z", "foot_r_x", "foot_r_y", "foot_r_z",
    # ---- objects (docs/OBJECTS.md): eight more prop slots, same ten columns as prop6/prop7, appended
    #  (PARAMS stays a prefix of itself: the server tells a short vector by its length). Slots 8-13 = OBJECTS' new props,
    #  14-15 = SCENE (chalk, eraser). A character without a props file ignores them (all zero = at home).
    "prop8_x", "prop8_y", "prop8_z", "prop8_rx", "prop8_ry", "prop8_rz", "prop8_hl", "prop8_hr", "prop8_hh", "prop8_s",
    "prop9_x", "prop9_y", "prop9_z", "prop9_rx", "prop9_ry", "prop9_rz", "prop9_hl", "prop9_hr", "prop9_hh", "prop9_s",
    "prop10_x", "prop10_y", "prop10_z", "prop10_rx", "prop10_ry", "prop10_rz", "prop10_hl", "prop10_hr", "prop10_hh", "prop10_s",
    "prop11_x", "prop11_y", "prop11_z", "prop11_rx", "prop11_ry", "prop11_rz", "prop11_hl", "prop11_hr", "prop11_hh", "prop11_s",
    "prop12_x", "prop12_y", "prop12_z", "prop12_rx", "prop12_ry", "prop12_rz", "prop12_hl", "prop12_hr", "prop12_hh", "prop12_s",
    "prop13_x", "prop13_y", "prop13_z", "prop13_rx", "prop13_ry", "prop13_rz", "prop13_hl", "prop13_hr", "prop13_hh", "prop13_s",
    "prop14_x", "prop14_y", "prop14_z", "prop14_rx", "prop14_ry", "prop14_rz", "prop14_hl", "prop14_hr", "prop14_hh", "prop14_s",
    "prop15_x", "prop15_y", "prop15_z", "prop15_rx", "prop15_ry", "prop15_rz", "prop15_hl", "prop15_hr", "prop15_hh", "prop15_s",
]
NPARAMS = len(PARAMS)
IDX = {n: i for i, n in enumerate(PARAMS)}
