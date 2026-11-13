"""Synthetic CANON v1 dyads (same npz format as the real converters') for building/testing the model before real data.

Structure the model must learn (so the ablation check has something to find): turn-taking speech; listeners nod
~0.5 s after the partner's phrase ends; smiles are answered with a ~0.6 s delay; speakers look away more; blinks.

    python -I tools/train/synth.py --out /path/synth --n 24 --minutes 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from canon1 import spec  # noqa: E402

C = spec.COL


def ou(T, tau, sd, rng):
    a = np.exp(-1.0 / tau)
    x = np.zeros(T)
    e = rng.standard_normal(T) * sd * np.sqrt(1 - a * a)
    for t in range(1, T):
        x[t] = a * x[t - 1] + e[t]
    return x


def bump(T, centers, width, amp):
    x = np.zeros(T)
    k = np.arange(-3 * width, 3 * width + 1)
    g = np.exp(-0.5 * (k / width) ** 2)
    for c in centers:
        lo, hi = c + k[0], c + k[-1] + 1
        if lo < 0 or hi > T:
            continue
        x[lo:hi] += amp * g
    return x


def make_session(T, rng):
    fps = spec.FPS
    # turn-taking: speaker id per frame
    spk = np.zeros(T, int)
    t, who = 0, int(rng.integers(2))
    while t < T:
        d = int(rng.uniform(2, 9) * fps)
        spk[t:t + d] = who
        t += d + int(rng.uniform(0, 0.8) * fps)
        who = 1 - who
    vad = np.zeros((2, T))
    for p in range(2):
        on = (spk == p).astype(float)
        # phrase gaps inside a turn
        gaps = rng.random(T) < 0.01
        for g in np.nonzero(gaps)[0]:
            on[g:g + int(0.3 * fps)] = 0
        vad[p] = on
    motion = np.zeros((2, T, spec.D_MOTION), np.float32)
    audio = np.zeros((2, T, spec.D_AUDIO), np.float32)
    smiles = [np.sort(rng.choice(T, size=max(1, T // (fps * 20)), replace=False)) for _ in range(2)]
    for p in range(2):
        q = 1 - p
        m = motion[p]
        m[:] = 0
        m[:, C["shoulder_l_z"]] = -1.2 + ou(T, 60, 0.05, rng)
        m[:, C["shoulder_r_z"]] = 1.2 + ou(T, 60, 0.05, rng)
        m[:, C["elbow_l_y"]] = ou(T, 30, 0.1, rng) + 0.3 * vad[p] * ou(T, 8, 0.4, rng)
        m[:, C["elbow_r_y"]] = ou(T, 30, 0.1, rng) - 0.3 * vad[p] * ou(T, 8, 0.4, rng)
        m[:, C["root_rot_x"]] = ou(T, 90, 0.03, rng)
        m[:, C["spine3_x"]] = ou(T, 60, 0.03, rng)
        speak = vad[p]
        # head yaw/pitch: more motion while speaking
        m[:, C["head_y"]] = ou(T, 20, 0.08, rng) * (0.5 + speak)
        m[:, C["neck_y"]] = 0.4 * m[:, C["head_y"]]
        # listener nods: after the partner's phrase ends (vad falling edge) while listening
        ends = np.nonzero((vad[q][:-1] > 0) & (vad[q][1:] == 0))[0] + int(0.5 * fps)
        ends = ends[rng.random(len(ends)) < 0.7]
        nod = np.zeros(T)
        for e in ends:
            if e + 15 < T:
                nod[e:e + 15] += 0.12 * np.sin(np.linspace(0, 2 * np.pi, 15))
        m[:, C["head_x"]] = ou(T, 20, 0.05, rng) + nod
        # smiles: own spontaneous + answer partner's smiles with 0.6 s delay
        own = bump(T, smiles[p], 12, 0.6)
        answer = bump(T, smiles[q] + int(0.6 * fps), 12, 0.5)
        sm = np.clip(own + answer + 0.05, 0, 1)
        m[:, 135 + spec.BS["mouthSmileLeft"]] = sm
        m[:, 135 + spec.BS["mouthSmileRight"]] = sm
        m[:, 135 + spec.BS["cheekSquintLeft"]] = 0.5 * sm
        m[:, 135 + spec.BS["cheekSquintRight"]] = 0.5 * sm
        jaw = np.clip(speak * (0.25 + 0.2 * np.sin(np.arange(T) * 2 * np.pi / 7 + rng.uniform(0, 6))), 0, 1)
        m[:, 135 + spec.BS["jawOpen"]] = jaw
        blinks = np.nonzero(rng.random(T) < 15 / 60 / fps)[0]
        bl = np.clip(bump(T, blinks, 2, 1.0), 0, 1)
        m[:, 135 + spec.BS["eyeBlinkLeft"]] = bl
        m[:, 135 + spec.BS["eyeBlinkRight"]] = bl
        m[:, 135 + spec.BS["browInnerUp"]] = np.clip(0.1 + 0.2 * speak * np.abs(ou(T, 10, 0.5, rng)), 0, 1)
        # gaze: speakers look away more
        away = (rng.random(T) < 0.02) & (speak > 0)
        aw = np.convolve(away.astype(float), np.ones(30), "same")
        m[:, C["gaze_yaw"]] = 0.3 * np.clip(aw, 0, 1) * np.sign(ou(T, 50, 1, rng)) + ou(T, 5, 0.03, rng)
        m[:, C["gaze_pitch"]] = ou(T, 5, 0.03, rng) - 0.1 * np.clip(aw, 0, 1)
        m[:, 45:135] = (ou(T, 40, 0.1, rng)[:, None] * rng.uniform(-1, 1, 90)[None]).astype(np.float32)
        # audio: noise floor + speech
        mel = -9 + 0.3 * rng.standard_normal((T, 40))
        mel += speak[:, None] * (4 + rng.standard_normal((T, 40)) * 1.5 + np.linspace(1, -1, 40)[None])
        loud = np.where(speak > 0, -25 + 3 * rng.standard_normal(T), -62 + 2 * rng.standard_normal(T))
        audio[p] = np.concatenate([mel, loud[:, None], speak[:, None]], 1)
    mask = np.ones((2, T, 9), np.uint8)
    for p in range(2):
        drop = rng.random() < 0.3
        if drop:
            mask[p, :, spec.GROUP_NAMES.index("fingers_l")] = 0
            mask[p, :, spec.GROUP_NAMES.index("fingers_r")] = 0
    return motion, mask, audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    idx = []
    for i in range(a.n):
        T = int(a.minutes * 60 * spec.FPS * rng.uniform(0.6, 1.4))
        mo, mk, au = make_session(T, rng)
        split = "val" if i % 8 == 7 else ("test" if i % 8 == 6 else "train")
        persons = [f"synth:S{i:03d}A", f"synth:S{i:03d}B"]
        meta = dict(dataset="synth", session=f"S{i:03d}", persons=persons, split=split,
                    source=["synthetic", "synthetic"], quality=[1.0, 1.0], speaking_track="vad")
        path = os.path.join(a.out, f"S{i:03d}.npz")
        spec.save_session(path, mo, mk, au, meta)
        idx.append(dict(path=path, dataset="synth", session=f"S{i:03d}", persons=persons, split=split, frames=T,
                        person_hours=2 * T / spec.FPS / 3600))
    with open(os.path.join(a.out, "index.jsonl"), "w") as f:
        for r in idx:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.n} sessions to {a.out}")


if __name__ == "__main__":
    main()
