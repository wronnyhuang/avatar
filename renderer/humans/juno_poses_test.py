"""rig: named test poses for Juno's rig (tests/test_juno_rig.py, tools/rig_sheet.py).  Values are control-vector
columns (anim/schema.py), the same words the page sends for Muse; `_l` = screen-left (her anatomical right)."""

POSES = {
    "REST": {},
    "WAVE": {"hand_r_x": 0.35, "hand_r_y": 1.15, "hand_r_z": 0.25, "hand_r_open": 1.0, "hand_r_turn": 1.0},
    "POINT": {"hand_l_x": 0.15, "hand_l_y": 0.62, "hand_l_z": 1.0, "point_l": 1.0},
    "CLAP": {"hand_l_x": -0.40, "hand_l_y": 0.50, "hand_l_z": 0.60, "hand_l_open": 1.0, "hand_l_turn": 0.0,
             "hand_r_x": -0.40, "hand_r_y": 0.50, "hand_r_z": 0.60, "hand_r_open": 1.0},
    "CHIN": {"hand_r_x": -0.35, "hand_r_y": 0.95, "hand_r_z": 0.55, "hand_r_open": 0.2, "head_pitch": 0.05,
             "head_roll": 0.07, "brow_knit": 0.3},
    "COVER_EYES": {"hand_l_x": -0.25, "hand_l_y": 1.25, "hand_l_z": 0.55, "hand_l_open": 1.0,
                   "hand_r_x": -0.25, "hand_r_y": 1.25, "hand_r_z": 0.55, "hand_r_open": 1.0, "head_pitch": 0.12},
    "HAND_ON_HEART": {"hand_r_x": -0.55, "hand_r_y": 0.62, "hand_r_z": 0.35, "hand_r_open": 0.9, "smile": 0.5},
    "THUMBS_UP": {"hand_r_x": 0.1, "hand_r_y": 0.55, "hand_r_z": 0.6, "thumb_r": 1.0, "hand_r_open": 0.0, "smile": 0.7},
    "COUNT_3": {"hand_l_x": 0.1, "hand_l_y": 0.9, "hand_l_z": 0.4, "fingers_l": 3.0, "hand_l_turn": 1.0},
    "SHRUG": {"shoulder_l": 1.0, "shoulder_r": 1.0, "hand_l_x": 0.5, "hand_l_y": 0.35, "hand_l_z": 0.4, "hand_l_open": 1.0,
              "hand_r_x": 0.5, "hand_r_y": 0.35, "hand_r_z": 0.4, "hand_r_open": 1.0, "head_roll": 0.1, "brow_l": 0.6, "brow_r": 0.6},
    "REACH_HIGH": {"hand_r_x": 0.1, "hand_r_y": 1.2, "hand_r_z": 0.0, "hand_r_open": 1.0},
    "REACH_ACROSS": {"hand_l_x": -1.0, "hand_l_y": 0.4, "hand_l_z": 0.4},
    "BEND_TWIST": {"body_yaw": 1.0, "body_pitch": 1.0, "body_lean": 0.6, "head_yaw": 0.4},
    "LOOK_BACK": {"body_yaw": -1.0, "head_yaw": -0.6, "head_pitch": -0.2, "gaze_x": -0.8, "gaze_y": 0.3, "gaze_free": 1.0},
    "SQUASH": {"squash": 1.0, "hand_l_y": -0.2, "hand_r_y": -0.2},
    "STRETCH": {"squash": -1.0, "hand_l_y": 1.2, "hand_r_y": 1.2, "hand_l_open": 1.0, "hand_r_open": 1.0},
    "STEP": {"foot_l_z": 0.12, "foot_r_z": -0.06, "foot_r_y": 0.03, "root_z": 0.03, "body_bounce": -0.4},
    "KICK": {"foot_r_z": 0.25, "foot_r_y": 0.12},
    "TWIRL": {"root_yaw": 0.375, "root_y": 0.04, "hand_l_x": 0.8, "hand_l_y": 0.6, "hand_r_x": 0.8, "hand_r_y": 0.6},
    "BACKFLIP": {"root_pitch": -0.5, "root_y": 0.6, "foot_l_y": 0.05, "foot_r_y": 0.05, "hand_l_y": 1.0, "hand_r_y": 1.0},
    "SIT": {"root_y": -0.12, "foot_l_z": 0.17, "foot_r_z": 0.17, "foot_l_y": 0.08, "foot_r_y": 0.08, "body_pitch": 0.2},
    "HEADPHONES": {"prop0_hh": 1.0, "smile": 0.4},
    "PINCH_CHALK": {"prop14_hl": 1.0, "hand_l_x": -0.1, "hand_l_y": 0.9, "hand_l_z": 0.3},
    "HOLD_MUG": {"prop4_hr": 1.0, "hand_r_x": -0.1, "hand_r_y": 0.5, "hand_r_z": 0.6},
    "TWO_HAND_BOX": {"prop10_hl": 1.0, "prop10_hr": 1.0, "hand_l_x": -0.25, "hand_l_y": 0.35, "hand_l_z": 0.6,
                     "hand_r_x": -0.25, "hand_r_y": 0.35, "hand_r_z": 0.6},
}

POSES.update({
    "PICK_FLOOR": {"hand_r_x": 0.1, "hand_r_y": -1.0, "hand_r_z": 0.3, "hand_r_open": 0.8},
    "PICK_TRAY": {"hand_l_x": 0.0, "hand_l_y": -0.65, "hand_l_z": 0.4, "hand_l_open": 0.6},
    "BOTH_LOW": {"hand_l_x": -0.2, "hand_l_y": -1.0, "hand_l_z": 0.5, "hand_r_x": -0.2, "hand_r_y": -1.0, "hand_r_z": 0.5},
    "LOW_TURNED": {"hand_r_y": -0.85, "hand_r_z": 0.6, "body_yaw": 0.6, "head_pitch": 0.3},
})

# every face / emotion column at its extremes (signed ones both ways)
FACE_EXTREMES = {
    "smile": [1.0], "brow_l": [-1.0, 1.0], "brow_r": [-1.0, 1.0], "brow_knit": [1.0], "blink_l": [1.0], "blink_r": [1.0],
    "gaze_x": [-1.0, 1.0], "gaze_y": [-1.0, 1.0], "cheek": [1.0], "nostril": [1.0], "nose_wiggle": [-1.0, 1.0],
    "eye_wide": [1.0], "squint": [1.0], "mouth_open": [1.0], "tongue_out": [1.0],
    "lid_droop": [1.0], "lid_slant": [1.0], "lid_happy": [1.0], "lid_asym": [-1.0, 1.0], "eye_scale": [-1.0, 1.0],
    "eye_ring": [1.0], "eye_spark": [-1.0, 1.0], "eye_shake": [1.0], "brow_tilt": [-1.0, 1.0], "brow_thick": [1.0],
    "mouth_curve": [-1.0, 1.0], "mouth_wide": [-1.0, 1.0], "mouth_teeth": [1.0], "mouth_wobble": [1.0],
    "mouth_pout": [1.0], "mouth_skew": [-1.0, 1.0], "fx_tear": [1.0], "fx_sweat": [1.0], "fx_vein": [1.0],
    "fx_blush": [1.0], "fx_pale": [1.0], "fx_question": [1.0], "fx_exclaim": [1.0], "fx_heart": [1.0],
    "fx_sparkle": [1.0], "fx_steam": [1.0], "fx_scribble": [1.0], "fx_dots": [1.0], "fx_green": [1.0], "fx_lines": [1.0],
    "lip_open": [1.0], "lip_upper": [1.0], "lip_lower": [1.0], "lip_width": [-1.0, 1.0], "lip_protrude": [1.0],
    "lip_press": [1.0], "lip_tuck": [1.0], "teeth_gap": [1.0], "tongue_tip": [1.0], "tongue_dental": [1.0],
    "tongue_back": [1.0], "jaw": [1.0], "round": [1.0], "spread": [1.0], "press": [1.0], "labdent": [1.0], "tongue": [1.0],
}
