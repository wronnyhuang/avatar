#!/usr/bin/env python3
"""planner prompts: one keyframe-planner system prompt per character, generated from the frozen Muse prompt.

    python app/planner_prompts/make_prompts.py            # (re)write <key>.txt for every character + MANIFEST.json
    python app/planner_prompts/make_prompts.py --check    # exit 1 if any file on disk differs from what it would write

muse.txt IS the frozen prompt (frozen_v1.txt, sha256 FROZEN_SHA, + FIXES) and is never rewritten; it is
the template every other prompt is cut from. ONLY the body description changes per character: the header paragraph,
the sections about controls the character lacks (the Muse's room / world axes / root motion / feet are dropped for a
seated character, its WHOLE BODY section becomes the seated one), the reference paw poses (the Muse's face touches use
its own big-head numbers), and the face line (ears / nose / tail only where the renderer draws them). Every other line
is copied verbatim.

What a character has comes from its files:
  name / what          app/app.py CHARACTERS
  leaves its seat      renderer/animals/<c>.json char_defines.CHAR_ROOT (the renderer applies root_* to that one only)
  legs                 renderer/animals/<c>.body.json leg block
  shape / not a shape  app/rtc.py persona(): not one of the five animals and not in NOT_SHAPES
  parts                the rig params its shader reads (renderer/animals/<c>.glsl + its `#pragma include`s):
                       arms P_HAND_L_X, ears P_EAR_L (never on a shape: RIG.md "Shapes: no ears, nose or tail"),
                       tail P_TAIL_WAG, nose P_NOSE_WIGGLE, fingers P_FINGERS_L, brows P_BROW_L, tongue P_TONGUE_OUT,
                       body bounce P_BODY_BOUNCE
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
RENDER = os.path.join(ROOT, "renderer")
FROZEN_SHA = "dc070126dd6ae359eb989b7de81ad51e08051eb4e685a6b7a219d8ad273e79e0"   # sha256 of frozen_v1.txt
FROZEN_SRC = "frozen_v1.txt (the original keyframe-stream prompt)"
FROZEN_FILE = "frozen_v1.txt"     # the frozen prompt itself (muse.txt = it + FIXES)
# corrections to the frozen text, applied to every character's prompt (muse.txt included).
# head_pitch: the renderer's + is nose DOWN (renderer/anim/life.py sign conventions; motion.js 'look up' -> head_pitch < 0,
# a nod adds +); the frozen prompt said "+ = looks up", so planned looks up / nods ran upside down (a-behavior, 06:05).
FIXES = [("head_pitch -0.45..0.45 (+ = looks up)", "head_pitch -0.45..0.45 (+ = tips the nose down, as in a nod; - = looks up)")]
ANIMALS = ("fox", "bear", "rabbit", "hippo", "monkey")      # rtc.persona(): the five animals are not shapes

# exec_stream.cjs KF_ALIAS keys (own-side control names the parser knows), in its order
KF_CONTROLS = ([f"{own}_paw_{a}" for own in ("right", "left") for a in ("x", "y", "z", "open", "turn")]
               + [f"{own}_{p}" for own in ("right", "left") for p in ("thumb", "shoulder", "fingers", "ear", "brow", "point")]
               + [f"{own}_foot_{a}" for own in ("right", "left") for a in ("x", "y", "z")]
               + ["eyes_closed", "root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash", "body_lean",
                  "body_pitch", "body_yaw", "head_yaw", "head_pitch", "head_roll", "smile", "mouth_open", "eye_wide", "squint",
                  "tongue_out", "brow_knit", "tail_raise", "tail_wag", "tail_curl", "nose_wiggle"])
SEATED_EXTRA = ["body_bounce"]      # the seated characters' vertical channel (rig body_bounce, motion.js RANGE -1..1)


# ------------------------------------------------------------------------------------------------- character files
def characters() -> dict:
    """app.py CHARACTERS read without importing app.py: {key: (name, what, voice)}"""
    tree = ast.parse(open(os.path.join(APP, "app.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CHARACTERS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("CHARACTERS not found in app.py")


def not_shapes() -> set:
    tree = ast.parse(open(os.path.join(APP, "rtc.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NOT_SHAPES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    return set()


def shader_params(key: str) -> set:
    seen, out = set(), set()

    def read(path):
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        src = open(path, encoding="utf-8").read()
        out.update(re.findall(r"\bP_[A-Z][A-Z0-9_]*\b", src))
        for inc in re.findall(r"^#pragma include (\w+)", src, re.M):
            read(os.path.join(RENDER, "engine", inc + ".glsl"))
    read(os.path.join(RENDER, "animals", key + ".glsl"))
    return out


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def traits(key: str, name: str, what: str) -> dict:
    cj = _json(os.path.join(RENDER, "animals", key + ".json")) or {}
    body = _json(os.path.join(RENDER, "animals", key + ".body.json")) or {}
    P = shader_params(key)
    shape = what not in ANIMALS and what not in not_shapes()
    return {
        "name": name, "what": what, "shape": shape,
        "leaves_seat": bool((cj.get("char_defines") or {}).get("CHAR_ROOT")),
        "legs": bool(body.get("leg") or body.get("legs")),
        "arms": "P_HAND_L_X" in P, "ears": (not shape) and "P_EAR_L" in P, "tail": "P_TAIL_WAG" in P,
        "nose": "P_NOSE_WIGGLE" in P, "fingers": "P_FINGERS_L" in P, "brows": "P_BROW_L" in P,
        "tongue": "P_TONGUE_OUT" in P, "bounce": "P_BODY_BOUNCE" in P,
    }


def controls(key: str, tr: dict) -> list:
    """The own-side controls this character's planner may write (the parser drops every other name)."""
    if key == "muse":
        return list(KF_CONTROLS)                 # the frozen prompt's character: exec_stream.cjs's set exactly
    drop = set()
    if not tr["leaves_seat"]:
        drop |= {"root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash"}
    if not tr["legs"]:
        drop |= {c for c in KF_CONTROLS if "_foot_" in c}
    if not tr["arms"]:
        drop |= {c for c in KF_CONTROLS if re.match(r"(right|left)_(paw_|thumb|shoulder|fingers|point)", c)}
    if not tr["fingers"]:
        drop |= {"right_fingers", "left_fingers"}
    if not tr["ears"]:
        drop |= {"right_ear", "left_ear"}
    if not tr["tail"]:
        drop |= {"tail_raise", "tail_wag", "tail_curl"}
    if not tr["nose"]:
        drop |= {"nose_wiggle"}
    if not tr["brows"]:
        drop |= {"right_brow", "left_brow", "brow_knit"}
    if not tr["tongue"]:
        drop |= {"tongue_out"}
    out = [c for c in KF_CONTROLS if c not in drop]
    if not tr["leaves_seat"] and tr["bounce"]:
        out += SEATED_EXTRA
    return out


# ------------------------------------------------------------------------------------------------- the template
ANCHORS = [("header", None), ("format", "\n\nYou get ONE movement request."), ("played", "\n\nHOW THE POSES ARE PLAYED\n"),
           ("room", "\n\nTHE ROOM:"), ("axes", "\n\nWORLD AXES"), ("whole", "\n\nWHOLE BODY\n"),
           ("feet", "\n\nFEET (WORLD positions)\n"), ("paws", "\n\nPAWS / ARMS"), ("bhf", "\n\nBODY, HEAD, FACE\n"),
           ("guidance", "\n\nGUIDANCE\n")]


def split_template(s: str) -> dict:
    """Frozen prompt -> {section: text}; ''.join(values) == s (each section after the first starts with its '\\n\\n')."""
    pos = [0]
    for _, a in ANCHORS[1:]:
        i = s.find(a)
        if i < 0 or s.count(a) != 1:
            raise RuntimeError(f"template anchor {a!r} missing or repeated")
        pos.append(i)
    if pos != sorted(pos):
        raise RuntimeError("template anchors out of order")
    pos.append(len(s))
    out = {ANCHORS[k][0]: s[pos[k]:pos[k + 1]] for k in range(len(ANCHORS))}
    assert "".join(out.values()) == s
    return out


def _sub(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"template line not found exactly once: {old[:70]!r}")
    return text.replace(old, new)


PLAYED_REST = ("When your list ends, every control eases back to rest by itself (a rotation goes back to the nearest whole "
               "turn); exception: root_x / root_z stay where they end (the body stays on that spot of the floor). ")
REF_MUSE = ("covering the eyes (x -0.52, y 1.2, z 1.15); scratching the head (x 0.33, y 1.3, z 0.55); touching the mouth "
            "(x -0.78, y 0.98, z 1.04); ")
CLAP_MUSE = "paws meeting for a clap in front of the chest (x -0.44, y 0.45, z 0.45 for both); "
# the eleven older characters' numbers: renderer/anim/rig_poses.py KEY_POSES (CLAP, COVER_EYES, CHIN) and app/motion.js
# gestures (scratch_head without a body file: HP(side, 0.1, 1.18, 0.0))
REF_SEATED = ("covering the eyes (x -0.35, y 1.0, z 0.45 for both); paw on the chin (x -0.35, y 0.75, z 0.35); scratching "
              "the head (x 0.1, y 1.18, z 0); ")
CLAP_SEATED = "paws meeting for a clap in front of the chest (x -0.45, y 0.45, z 0.45 for both); "
FINGERS = " right_fingers / left_fingers: 0 = normal, 1-5 = that many fingers held up."
BODY_LINE = ("- body_lean (-1..1): the upper body bends sideways (+ = toward its own left). body_pitch (-1..1): + = bends "
             "forward toward the camera, - = leans back. body_yaw (-1..1): twists the upper body (+ = toward its own left). "
             "0.3 is a clear bend, 1 the maximum.\n")
FACE_LINE = ("- Face: smile 0..1 (rest 0.1), mouth_open 0..1, eye_wide 0..1, squint 0..1, eyes_closed 0..1, tongue_out 0..1, "
             "right_brow / left_brow -1..1 (+ = raised), brow_knit 0..1 (frown). right_ear / left_ear -1 (drooped back) .. 1 "
             "(up). tail_raise 0..1 (rest 0.4), tail_wag -1..1.")
LYING = ("- Lying, rolling and upside-down positions are fine: rotate with root_pitch / root_roll and the floor contact is "
         "automatic. Paws are in the body frame: upside down (root_pitch 0.5) a paw at y 1.0 points at the floor.")


def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def header(tr: dict) -> str:
    n, what = tr["name"], tr["what"]
    seat = "spot" if tr["shape"] else "seat"
    if tr["shape"]:
        face = ["eyes"] + (["brows"] if tr["brows"] else []) + ["a mouth"]
        has = f"{n} is a {what} with a face ({_join(face)})" + (" and two short arms with mitten hands (called paws below)"
                                                                if tr["arms"] else "")
        lacks = ["legs"] + (["ears"] if not tr["ears"] else []) + (["nose"] if not tr["nose"] else []) + \
                (["tail"] if not tr["tail"] else [])
        has += f"; it has no {_join(lacks).replace(' and ', ' or ')}."
        rest = "At rest it faces the camera" + (", hands relaxed at its sides." if tr["arms"] else ".")
        where = f"It rests on its {seat} on the ground and stays there"
    else:
        head = (["two ears"] if tr["ears"] else []) + (["a nose"] if tr["nose"] else []) + \
               (["brows"] if tr["brows"] else []) + ["a mouth"]
        has = f"It has a head with {_join(head)}" + (", two arms with paws" if tr["arms"] else "") + \
              (" and a tail." if tr["tail"] else "; it has no tail.")
        rest = "At rest it sits facing the camera" + (", paws relaxed at its sides." if tr["arms"] else ".")
        where = f"{n} sits on the ground and stays in its {seat}"
    return (f"You animate {n}, a cartoon {what}, by writing a time-ordered list of KEY POSES for its rig. {where}: it never "
            f"stands up, walks, leaves its {seat}, lies down, flips over or turns all the way around, so its whole body only "
            f"bounces, leans and swivels a little where it {'is' if tr['shape'] else 'sits'} (WHOLE BODY below). {has} {rest} "
            "Nothing around it can be picked up or moved, so mime any object a request involves (an imaginary ball, rope, "
            "instrument...).")


def whole_seated(tr: dict) -> str:
    where = "on its spot" if tr["shape"] else "in its seat"
    paws = ", paws up a little" if tr["arms"] else ""
    return (f"\n\nWHOLE BODY (it stays {where})\n"
            f"- body_bounce (-1..1): the whole body bobs up and down {where} (+ = up, - = sinks down). This is its jump: it "
            "dips, rises to the peak and dips again on landing before settling at 0; a big jump peaks at 1, a small hop at "
            "0.4, a tiny one at 0.15.\n"
            f"- body_yaw (-1..1): swivels the whole body {where} (+ = toward its own left). It is a limited swivel, not a "
            "turn: 1 is the most it can swivel (well under a quarter turn), so a spin or turning around becomes a swivel "
            "to one side and back, with the head turning further the same way.\n"
            "- body_lean (-1..1): the whole body leans sideways (+ = toward its own left). body_pitch (-1..1): + = leans "
            "forward toward the camera, - = leans back. 0.3 is a clear lean, 1 the maximum.\n"
            f"How the engine's own moves set these: a seated jump = body_bounce -0.4 at 0.23 s (crouch), 1 at 0.52 s (the "
            f"peak{paws}), -0.3 at 0.8 s (landing), 0 at 1.15 s; a spin = body_yaw 0 -> 1 -> 0 in about 2 s with the head "
            "turning the same way; turning around = body_yaw to 1 in 0.8 s, held about 1.5 s, back in 0.9 s; a bow = "
            "body_pitch 0.65 with the head bowed; a dance = body_bounce 0 -> 0.3 -> 0 twice a second while body_lean sways "
            "-0.35 .. 0.35 once a second" + (", paws at chest height." if tr["arms"] else "."))


def face_line(tr: dict) -> str:
    s = "- Face: smile 0..1 (rest 0.1), mouth_open 0..1, eye_wide 0..1, squint 0..1, eyes_closed 0..1"
    if tr["tongue"]:
        s += ", tongue_out 0..1"
    if tr["brows"]:
        s += ", right_brow / left_brow -1..1 (+ = raised), brow_knit 0..1 (frown)"
    s += "."
    if tr["ears"]:
        s += " right_ear / left_ear -1 (drooped back) .. 1 (up)."
    if tr["nose"]:
        s += " nose_wiggle -1..1 (nose tip down .. up)."
    if tr["tail"]:
        s += " tail_raise 0..1 (rest 0.4), tail_wag -1..1 (where the tail points sideways: alternate it to wag), tail_curl 0..1."
    return s


def render(key: str, tr: dict, base: str) -> str:
    if key == "muse":
        return base                     # byte-identical to the frozen prompt
    if tr["leaves_seat"] or tr["legs"]:
        raise RuntimeError(f"{key}: a second free-standing character needs its own body header (only the muse has one)")
    S = split_template(base)
    out = [header(tr), S["format"], _sub(S["played"], PLAYED_REST, "When your list ends, every control eases back to rest "
                                                                    "by itself. "),
           whole_seated(tr)]
    if tr["arms"]:
        p = _sub(S["paws"], REF_MUSE, REF_SEATED)
        p = _sub(p, CLAP_MUSE, CLAP_SEATED)
        if not tr["fingers"]:
            p = _sub(p, FINGERS, "")
        out.append(p)
    bhf = _sub(S["bhf"], BODY_LINE, "")
    bhf = _sub(bhf, FACE_LINE, face_line(tr))
    out.append(bhf.replace("\n\nBODY, HEAD, FACE\n", "\n\nHEAD, FACE\n", 1))
    where = "on its spot" if tr["shape"] else "in its seat"
    kind = "in-place" if tr["shape"] else "seated"
    parts = ["bounce", "lean", "swivel", "head"] + (["paws"] if tr["arms"] else []) + ["face"] + \
            (["ears"] if tr["ears"] else []) + (["tail"] if tr["tail"] else [])
    out.append(_sub(S["guidance"], LYING,
                    f"- It stays {where}: it cannot stand up, walk, travel, lie down, roll, flip or turn all the way "
                    f"around. For such a request do the {kind} version with what it has ({_join(parts)}) and say in the "
                    f"summary that it is {'an' if tr['shape'] else 'a'} {kind} version."))
    text = "".join(out)
    used = set(re.findall(r"\b((?:right|left)_[a-z]+(?:_[a-z]+)?|root_[a-z]+|body_[a-z]+|head_[a-z]+|tail_[a-z]+|"
                          r"nose_wiggle|squash|eyes_closed|brow_knit|mouth_open|tongue_out|eye_wide)\b", text))
    allowed = set(controls(key, tr))
    bad = sorted(c for c in used if c in set(KF_CONTROLS) | set(SEATED_EXTRA) and c not in allowed)
    if bad:
        raise RuntimeError(f"{key}: prompt names controls it lacks: {bad}")
    return text


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build() -> tuple[dict, dict]:
    base = open(os.path.join(HERE, FROZEN_FILE), encoding="utf-8").read()
    if sha(base) != FROZEN_SHA:
        raise RuntimeError(f"{FROZEN_FILE} is not the frozen prompt")
    for old, new in FIXES:
        if old not in base:
            raise RuntimeError(f"fix target not in the frozen prompt: {old!r}")
        base = base.replace(old, new)
    files, man = {}, {"frozen": {"source": FROZEN_SRC, "sha256": FROZEN_SHA, "file": FROZEN_FILE},
                      "fixes": [{"from": a, "to": b} for a, b in FIXES],
                      "generator": "app/planner_prompts/make_prompts.py", "characters": {}}
    for key, (name, what, _voice) in characters().items():
        tr = traits(key, name, what)
        text = render(key, tr, base)
        files[key + ".txt"] = text
        man["characters"][key] = {"file": key + ".txt", "sha256": sha(text), "bytes": len(text.encode("utf-8")),
                                  "traits": tr, "controls": controls(key, tr)}
    return files, man


def main() -> int:
    files, man = build()
    mtext = json.dumps(man, indent=1) + "\n"
    check = "--check" in sys.argv
    diff = []
    for fn, text in list(files.items()) + [("MANIFEST.json", mtext)]:
        p = os.path.join(HERE, fn)
        cur = open(p, encoding="utf-8").read() if os.path.isfile(p) else None
        if cur != text:
            diff.append(fn)
            if not check:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(text)
    for k, v in man["characters"].items():
        print(f"{k:8s} {v['sha256']}  {v['bytes']:6d} B  {len(v['controls'])} controls")
    if check and diff:
        print("differs:", ", ".join(diff))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
