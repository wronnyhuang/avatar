"""the mascot characters on the muse skeleton (animals/<name>.glsl through engine/mascot_rig.glsl + mascot_body.glsl).

Extra characters (stylized mascots and two originals).
A mascot shares the muse's body / props files (symlinks in animals/) and, where a per-character table has no entry of
its own (anim/styles.json, engine/rt_presets.json, engine/rt_mouth_boxes.json, avatar/profiles.json, gesture defaults),
the muse's entry: `base(character)` names whose row to read.
"""
MASCOTS = ("gekkie", "hooty", "doughlad", "bonbon", "pneu", "goober", "jade", "ruby", "flurry")   # + jade, ruby (original characters)


def base(character):
    """The character whose table rows a character without its own borrows (a mascot -> muse)."""
    return "muse" if character in MASCOTS else character
