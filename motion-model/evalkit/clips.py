#!/usr/bin/env python3
"""Build the held-out clip set -> $T/eval/clips/manifest.json (+ annotations/<id>.json for the judge).

  python -I evalkit/clips.py [--seamless-n 30] [--provisional]

(a) Seamless TEST sessions with video (person-disjoint test split from $T/data/splits.json; only the official dev
    pairs carry video, so the test people come from those). One 30-40 s window per session, the direction (who is the
    USER, who is the AVATAR) and the window chosen to MIX listening and speaking: both people speak 20-70 % of the
    window and the turn changes >= 2 times. USER = one person (camera + mic); AVATAR speech = the OTHER person's real
    voice; HUMAN reference = the other person's real motion (the data pipeline's canon of that session).
    --provisional (before splits.json exists): two official-dev pairs, marked split "provisional" (harness only).
(b) the frozen held-out YouTube/stock suite ($EVAL_YT_CLIPS: manifest.json + annotations/) minus every clip with a
    command cue (category command + interrupt_synthetic): real YouTube/stock people as camera + mic, a recorded mock
    avatar (TTS) voice as the avatar's speech ($EVAL_AVATAR_VOICES/<clip id>/avatar.wav; identical for every system),
    no human reference. Rendered length = clip + 4 s tail. Skipped when the suite is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, SEAMLESS, YT_CLIPS, AVATAR_VOICES, atomic_json  # noqa: E402

WIN_S = 36.0
YT_MANIFEST = YT_CLIPS / "manifest.json"
YT_ANN = YT_CLIPS / "annotations"


def vad_track(fid, label, split, n):
    p = SEAMLESS / label / split / "metadata/vad" / f"{fid}.jsonl"
    v = np.zeros(n, bool)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                v[int(r["start"] * 30):int(r["end"] * 30) + 1] = True
    return v


def words(fid, label, split):
    p = SEAMLESS / label / split / "metadata/transcript" / f"{fid}.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                for w in json.loads(line).get("words", []):
                    if w.get("start") is not None and w.get("end") is not None:
                        out.append((float(w["start"]), float(w["end"]), str(w["word"])))
    return sorted(out)


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def bridge(v, gap=9):
    v = v.copy()
    r = runs(v)
    for (a0, b0), (a1, b1) in zip(r[:-1], r[1:]):
        if a1 - b0 <= gap:
            v[b0:a1] = True
    return v


def pick_window(vu, va, n, win=WIN_S):
    """best window for (user vad, avatar vad): both 20-70 % speaking, >= 2 turn changes, max min-fraction."""
    W = int(win * 30)
    best = None
    for s in range(150, max(151, n - W - 30), 30):
        u, a = bridge(vu[s:s + W]), bridge(va[s:s + W])
        fu, fa = u.mean(), a.mean()
        # turn changes: alternation of the speaking person over bridged runs
        who = []
        for k in range(0, W, 15):
            x = (u[k], a[k])
            if x == (True, False):
                who.append("u")
            elif x == (False, True):
                who.append("a")
        ch = sum(1 for p, q in zip(who[:-1], who[1:]) if p != q)
        ok = 0.2 <= fu <= 0.7 and 0.2 <= fa <= 0.7 and ch >= 2
        sc = (1.0 if ok else 0.0) + min(fu, fa) + 0.02 * min(ch, 6) - 0.5 * max(0.0, (u & a).mean() - 0.15)
        if best is None or sc > best[0]:
            best = (sc, s, float(fu), float(fa), ch, ok)
    return best


def seamless_pairs(test_people=None, test_sessions=None):
    """official dev pairs with BOTH videos; restricted to test people/sessions when given."""
    import collections
    pairs = collections.defaultdict(list)
    for lab in ("improvised", "naturalistic"):
        vd = SEAMLESS / lab / "dev/video"
        if not vd.exists():
            continue
        for f in sorted(os.listdir(vd)):
            fid = f[:-4]
            pairs[(lab, "_".join(fid.split("_")[:3]))].append(fid)
    out = []
    for (lab, key), fids in sorted(pairs.items()):
        if len(fids) != 2:
            continue
        if test_people is not None and not all(pid(f) in test_people for f in fids):
            continue
        out.append((lab, key, sorted(fids)))
    return out


def load_splits():
    p = T / "data/splits.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pid(fid):
    """the data pipeline's person id: 'seamless:P0299' (trailing A collapsed; vendor V00)."""
    return "seamless:P" + "".join(ch for ch in fid.split("_")[3] if ch.isdigit())


def test_sets(spl):
    """-> (test people, test session keys) from the data pipeline's splits.json {persons: {pid: split}, sessions: {key: split}}."""
    people = {p for p, s in (spl.get("persons") or {}).items() if s == "test" and p.startswith("seamless:")}
    sessions = {k.split(":", 1)[1] for k, s in (spl.get("sessions") or {}).items() if s == "test" and k.startswith("seamless:")}
    return people, sessions


def seamless_clips(n_max, provisional=False):
    if provisional:
        prs = seamless_pairs()[:2]
        split = "provisional"
    else:
        spl = load_splits()
        if spl is None:
            raise SystemExit("no $T/data/splits.json yet: use --provisional")
        people, sessions = test_sets(spl)
        prs = seamless_pairs(people)
        split = "test"
    clips = []
    for lab, key, fids in prs:
        sp = "dev"
        a0 = SEAMLESS / lab / sp / "audio" / f"{fids[0]}.wav"
        import soundfile as sf
        n = int(sf.info(str(a0)).duration * 30)
        tracks = {f: vad_track(f, lab, sp, n) for f in fids}
        cands = []
        for u, av in ((fids[0], fids[1]), (fids[1], fids[0])):
            b = pick_window(tracks[u], tracks[av], n)
            if b:
                cands.append((b, u, av))
        if not cands:
            continue
        (sc, s, fu, fa, ch, ok), u, av = max(cands, key=lambda x: x[0][0])
        t0 = s / 30.0
        pu, pa = u.split("_")[3], av.split("_")[3]
        cid = f"sl_{key.split('_')[1]}_{key.split('_')[2][-4:]}_{pu}"
        ws = [w for w in words(u, lab, sp) if t0 <= w[0] < t0 + WIN_S]
        phrases = []
        for w0, w1, wd in ws:
            if phrases and w0 - phrases[-1]["t1"] < 0.6:
                phrases[-1]["t1"] = round(w1 - t0, 2); phrases[-1]["text"] += " " + wd
            else:
                phrases.append({"t0": round(w0 - t0, 2), "t1": round(w1 - t0, 2), "text": wd})
        cues = []
        for i, p in enumerate(phrases):
            nxt = phrases[i + 1]["t0"] if i + 1 < len(phrases) else WIN_S
            if nxt - p["t1"] >= 0.8 and p["t1"] < WIN_S - 1.5:
                cues.append({"t": p["t1"], "cue": f"user pause {nxt - p['t1']:.1f}s", "region": "any", "kind": "turn",
                             "expect": "listens with backchannel while the user speaks; at the pause may reply or keep "
                                       "listening; looks at the user", "window_s": [0.3, 2.0]})
        aw = [w for w in words(av, lab, sp) if t0 <= w[0] < t0 + WIN_S]
        ann = {"id": cid, "category": "seamless_" + lab, "split": split, "duration_s": WIN_S,
               "description": (f"Two-person video call (Seamless Interaction, {lab}). The user on camera talks with the "
                               "avatar; they take turns. The avatar's voice is a real recorded human reply (fixed, "
                               "identical for every system); judge the avatar's non-verbal behaviour while it listens "
                               "and while it talks."),
               "audio": "original", "user_speech": phrases, "cues": cues, "talk": "either",
               "talk_note": "the avatar's speech timing is a fixed real recording (the same in every version compared)",
               "overall": "engaged conversation partner"}
        ap = E / "clips/annotations" / f"{cid}.json"
        atomic_json(ap, ann)
        clips.append({"id": cid, "group": "seamless", "split": split, "category": "seamless_" + lab,
                      "duration_s": WIN_S, "video": str(SEAMLESS / lab / sp / "video" / f"{u}.mp4"), "v_t0": t0,
                      "user_wav": str(SEAMLESS / lab / sp / "audio" / f"{u}.wav"), "user_wav_t0": t0,
                      "voice_wav": str(SEAMLESS / lab / sp / "audio" / f"{av}.wav"), "voice_t0": t0,
                      "session": key, "label": lab, "user_file": u, "avatar_file": av,
                      "user_person": pid(u), "avatar_person": pid(av),
                      "frame0": int(s), "window_stats": {"user_speak": round(fu, 3), "avatar_speak": round(fa, 3),
                                                         "turn_changes": ch, "criteria_ok": bool(ok)},
                      "avatar_words": [[round(w0 - t0, 3), round(w1 - t0, 3), wd] for w0, w1, wd in aw],
                      "annotation": str(ap), "camera_mp4": str(E / "clips" / cid / "camera.mp4"),
                      "mic_wav": str(E / "clips" / cid / "mic.wav")})
    # one window per session; prefer windows meeting the criteria, then by min speaking fraction; cap n_max
    clips.sort(key=lambda c: (not c["window_stats"]["criteria_ok"], -min(c["window_stats"]["user_speak"], c["window_stats"]["avatar_speak"])))
    return clips[:n_max]


def youtube_clips():
    if not YT_MANIFEST.exists():
        print(f"no held-out YouTube suite at {YT_MANIFEST}; skipped")
        return []
    man = json.loads(YT_MANIFEST.read_text())["clips"]
    out = []
    for c in man:
        if c["split"] != "holdout" or c["category"] == "command":
            continue
        ann = json.loads((YT_ANN / f"{c['id']}.json").read_text())
        if any(q.get("kind") == "command" for q in ann.get("cues") or []):
            continue                                   # interrupt_synthetic: a spoken command inside
        ap = E / "clips/annotations" / f"{c['id']}.json"
        atomic_json(ap, ann)
        voice = AVATAR_VOICES / c["id"] / "avatar.wav"
        out.append({"id": c["id"], "group": "youtube", "split": "holdout", "category": c["category"],
                    "duration_s": round(float(c["duration_s"]) + 4.0, 3), "clip_s": float(c["duration_s"]),
                    "video": c["camera_mp4"], "v_t0": 0.0, "user_wav": c["mic_wav"], "user_wav_t0": 0.0,
                    "voice_wav": str(voice) if voice.exists() else None, "voice_t0": 0.0,
                    "user_person": "youtube:" + c["id"], "source": c.get("source"), "audio": c.get("audio"),
                    "annotation": str(ap), "camera_mp4": str(E / "clips" / c["id"] / "camera.mp4"),
                    "mic_wav": str(E / "clips" / c["id"] / "mic.wav")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seamless-n", type=int, default=32)
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--no-youtube", action="store_true")
    a = ap.parse_args()
    clips = []
    sl = seamless_clips(a.seamless_n, a.provisional)
    clips += sl
    if not a.no_youtube:
        clips += youtube_clips()
    mp = E / "clips/manifest.json"
    old = json.loads(mp.read_text())["clips"] if mp.exists() else []
    keep = {c["id"]: c for c in old}
    for c in clips:
        keep[c["id"]] = c
    if not a.provisional:   # a real test split replaces the provisional clips
        keep = {k: v for k, v in keep.items() if v["split"] != "provisional"}
    atomic_json(mp, {"version": 1, "clips": sorted(keep.values(), key=lambda c: (c["group"], c["id"]))})
    print(f"{len(sl)} seamless + {sum(1 for c in clips if c['group'] == 'youtube')} youtube clips -> {mp}")
    for c in sl:
        print(" ", c["id"], c["user_person"], "->", c["avatar_person"], c["window_stats"])


if __name__ == "__main__":
    main()
