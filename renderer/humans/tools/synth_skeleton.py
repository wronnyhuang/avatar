"""rig: a synthetic Juno skeleton in the juno.asset/1 format (skeleton.json + a minimal juno.asset.json), for rig
tests before / beside ASSET's real asset.  Stock stylised-woman proportions, metres, A-pose, +y up, faces +z, anatomical
sides (_l = +x), bone local +y toward the child (Blender convention), palms down/in.

  python renderer/humans/tools/synth_skeleton.py OUT_DIR        # writes OUT_DIR/skeleton.json + OUT_DIR/juno.asset.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ARKIT52 = (
    "eyeBlinkLeft eyeLookDownLeft eyeLookInLeft eyeLookOutLeft eyeLookUpLeft eyeSquintLeft eyeWideLeft "
    "eyeBlinkRight eyeLookDownRight eyeLookInRight eyeLookOutRight eyeLookUpRight eyeSquintRight eyeWideRight "
    "jawForward jawLeft jawRight jawOpen mouthClose mouthFunnel mouthPucker mouthLeft mouthRight "
    "mouthSmileLeft mouthSmileRight mouthFrownLeft mouthFrownRight mouthDimpleLeft mouthDimpleRight "
    "mouthStretchLeft mouthStretchRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper "
    "mouthPressLeft mouthPressRight mouthLowerDownLeft mouthLowerDownRight mouthUpperUpLeft mouthUpperUpRight "
    "browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight "
    "noseSneerLeft noseSneerRight tongueOut").split()
EXTRAS = ("labioDental lipTuckLower lipTuckUpper tongueTipUp tongueDental tongueBack lipsTogether "
          "eyeHappyLeft eyeHappyRight browTiltLeft browTiltRight").split()


def _frame(y, hint):
    y = np.asarray(y, float); y = y/np.linalg.norm(y)
    x = np.asarray(hint, float); x = x - y*np.dot(x, y)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, [0, 0, 1.0]) if abs(y[2]) < 0.9 else np.cross(y, [1.0, 0, 0])
    x = x/np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def build(height=1.68):
    k = height/1.68
    B = []                                    # (name, parent, head position, tail position or None, x hint, helper)

    def add(name, parent, head, tail=None, hint=(1, 0, 0), helper=False):
        B.append([name, parent, np.array(head, float)*k, None if tail is None else np.array(tail, float)*k, hint, helper])

    add("root", None, (0, 0, 0), (0, 0.1, 0))
    add("pelvis", "root", (0, 0.98, 0.0), (0, 1.05, 0))
    add("spine_01", "pelvis", (0, 1.05, 0.0))
    add("spine_02", "spine_01", (0, 1.16, -0.005))
    add("spine_03", "spine_02", (0, 1.28, -0.012))
    add("neck_01", "spine_03", (0, 1.42, -0.015))
    add("head", "neck_01", (0, 1.50, -0.005), (0, 1.68, -0.005))
    add("jaw", "head", (0, 1.525, 0.005), (0, 1.47, 0.085), hint=(1, 0, 0))
    for sd, s in (("l", 1.0), ("r", -1.0)):
        add(f"eye_{sd}", "head", (s*0.034, 1.585, 0.072), (s*0.034, 1.585, 0.10))
    a45 = np.radians(45.0)
    for sd, s in (("l", 1.0), ("r", -1.0)):
        sh = np.array([s*0.175, 1.385, -0.015])
        d = np.array([s*np.cos(a45), -np.sin(a45), 0.0])
        el = sh + d*0.275; wr = el + d*0.245
        add(f"clavicle_{sd}", "spine_03", (s*0.025, 1.395, 0.02), sh/k)
        add(f"upperarm_{sd}", f"clavicle_{sd}", sh/k, el/k)
        add(f"upperarm_twist_01_{sd}", f"upperarm_{sd}", (sh + d*0.14)/k, el/k, helper=True)
        add(f"lowerarm_{sd}", f"upperarm_{sd}", el/k, wr/k)
        add(f"lowerarm_twist_01_{sd}", f"lowerarm_{sd}", (el + d*0.15)/k, wr/k, helper=True)
        add(f"hand_{sd}", f"lowerarm_{sd}", wr/k, (wr + d*0.09)/k)
        n = np.array([-s*np.sin(a45), -np.cos(a45), 0.0])                 # palm normal: down and in
        z = np.array([0, 0, 1.0])                                         # across the palm, toward the thumb
        fing = {"index": (0.020, (0.046, 0.027, 0.022)), "middle": (0.004, (0.050, 0.031, 0.023)),
                "ring": (-0.012, (0.046, 0.029, 0.022)), "pinky": (-0.027, (0.036, 0.022, 0.019))}
        for fn, (off, ls) in fing.items():
            p = wr + d*(0.092 - 0.006*abs(off)/0.027) + z*off + n*0.004
            fd = d + z*off*1.2
            fd = fd/np.linalg.norm(fd)
            prev = f"hand_{sd}"
            for j, L in enumerate(ls):
                q = p + fd*L
                add(f"{fn}_0{j+1}_{sd}", prev, p/k, q/k, hint=z)
                prev = f"{fn}_0{j+1}_{sd}"; p = q
        p = wr + d*0.025 + z*0.022 + n*0.012
        td = d*0.55 + z*0.75 + n*0.25; td = td/np.linalg.norm(td)
        prev = f"hand_{sd}"
        for j, L in enumerate((0.040, 0.032, 0.026)):
            q = p + td*L
            add(f"thumb_0{j+1}_{sd}", prev, p/k, q/k, hint=n)
            prev = f"thumb_0{j+1}_{sd}"; p = q
    for sd, s in (("l", 1.0), ("r", -1.0)):
        add(f"thigh_{sd}", "pelvis", (s*0.088, 0.925, 0.0), (s*0.094, 0.50, 0.012))
        add(f"thigh_twist_01_{sd}", f"thigh_{sd}", (s*0.091, 0.71, 0.006), (s*0.094, 0.50, 0.012), helper=True)
        add(f"calf_{sd}", f"thigh_{sd}", (s*0.094, 0.50, 0.012), (s*0.10, 0.085, -0.012))
        add(f"foot_{sd}", f"calf_{sd}", (s*0.10, 0.085, -0.012), (s*0.106, 0.022, 0.120), hint=(1, 0, 0))
        add(f"ball_{sd}", f"foot_{sd}", (s*0.106, 0.022, 0.120), (s*0.108, 0.018, 0.185), hint=(1, 0, 0))
    names = [b[0] for b in B]
    out = []
    for b in B:
        name, parent, head, tail, hint, helper = b
        if tail is None:
            kids = [c for c in B if c[1] == name and not c[5]]
            tail = kids[0][2] if kids else head + np.array([0, 0.05, 0])
        R = _frame(tail - head, hint)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = head
        out.append({"name": name, "parent": -1 if parent is None else names.index(parent),
                    "rest": M.reshape(-1).round(7).tolist(), "inverse_bind": np.linalg.inv(M).reshape(-1).round(7).tolist(),
                    "helper": bool(helper)})
    sk = {"bones": out, "skeleton_hints": {"jaw_open_deg": 26.0, "eye_radius_m": 0.0155*k,
                                           "eye_center_l": [0.034*k, 1.585*k, 0.072*k], "eye_center_r": [-0.034*k, 1.585*k, 0.072*k],
                                           "head_top": [0, 1.68*k, -0.005*k], "head_center": [0, 1.575*k, 0.0], "chin": [0, 1.455*k, 0.09*k],
                                           "synthetic": True}}
    meta = {"format": "juno.asset/1", "name": "juno-synthetic", "units": "m", "up": "+y", "forward": "+z",
            "height_m": height, "scale_to_H": 1.0/height, "rest_pose": "A", "skeleton": "skeleton.json", "lods": [],
            "shapes": ARKIT52 + EXTRAS, "materials": {}, "hair_groups": {}, "landmarks": {}}
    return sk, meta


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sk, meta = build()
    json.dump(sk, open(os.path.join(out, "skeleton.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "juno.asset.json"), "w"), indent=1)
    print("wrote", out, len(sk["bones"]), "bones")


if __name__ == "__main__":
    main()
