"""junor — the adult body DESIGN on the MPFB2 (CC0) hm08 base: MH macro targets only (no stylising sculpt);
every juno_design knob is set to its neutral value so juno_design.stylize() is a plain target sum here. The head is
NOT designed here: the ICT-FaceKit head replaces it (ict_head.py / graft.py)."""
MACRO = [("macrodetails/african-female-young.target.gz", 0.40), ("macrodetails/asian-female-young.target.gz", 0.20),
         ("macrodetails/caucasian-female-young.target.gz", 0.40),
         ("macrodetails/universal-female-young-averagemuscle-averageweight.target.gz", 0.75),
         ("macrodetails/universal-female-young-averagemuscle-minweight.target.gz", 0.25),
         ("macrodetails/height/female-young-averagemuscle-averageweight-maxheight.target.gz", 0.12),
         ("macrodetails/proportions/female-young-averagemuscle-averageweight-idealproportions.target.gz", 1.0)]
NEUTRAL = {
    "macro": MACRO, "targets": [], "head_scale": 1.0, "eye_scale": 1.0, "eye_sink": 0.0, "eye_spread_m": 0.0,
    "philtrum_lift_m": 0.0, "lower_face_k": 1.0, "chin_narrow": 1.0, "cheek_pad_m": 0.0, "rbf_sculpt": False,
    "face_smooth_iters": 0, "cranium_wide": 0.0, "cranium_low": 0.0, "finger_len": 1.0, "finger_slim": 0.96,
    "wrist_slim": 0.96, "shoulder_in_m": 0.0, "leg_scale": 1.0, "eye_open_up_deg": 0.0, "eye_open_dn_deg": 0.0,
    "design_fit": False, "head_post_scale": 1.0, "neck_shorten_m": 0.0, "neck_widen": 0.0, "hand_scale": 1.0,
    "ear_scale": 1.0, "ear_flare_deg": 0.0, "ear_thick": 1.0, "nipple_smooth_iters": 40,
    "mouth_rest_corner_up_m": 0.0, "mouth_rest_lower_back_m": 0.0, "mouth_rest_lower_thin_m": 0.0,
    "mouth_rest_upper_full_m": 0.0, "mouth_rest_widen": 0.0,
}
