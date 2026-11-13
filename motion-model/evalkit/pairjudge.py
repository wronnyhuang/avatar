#!/usr/bin/env python3
"""EVAL: BLIND PAIRWISE judge — which of two avatars is more natural, and which reacts more to the user.

  python -I evalkit/pairjudge.py judge  --a rules --b unified-<tag> --ids a,b,.. [--swap-ids x,y] [--jobs 4]
  python -I evalkit/pairjudge.py summary --a rules --b unified-<tag> [--out f.json]

Judge = gflash (google/gemini-3.7-flash via OpenRouter, the route in evalkit/judge/judge.py; JUDGE_ALLOW_PAID=1).
Per clip ONE call sees: a composite video [USER camera | avatar A | avatar B] with the user's mic + the avatar's voice
(the voice is the SAME for A and B: only the motion differs), plus 3 frame-strip sheets (A and B rows interleaved at
3 fps with user-speaking / avatar-speaking marks, because the judge's video sampling is ~1 fps) and the clip's user
speech timeline. Which system is A is a per-clip coin flip (sha of clip + pair), so position bias averages out over
clips; `--swap-ids` also judges the reversed order on those clips (position-consistency check, reported).
The judge never learns the systems' names. Output per pair: $E/judge/pair/<a>__<b>/<clip>[.swap].json; cached by
(media sha, prompt version) under $E/judge/pair/cache so a rerun costs nothing. Cost -> $E/spend.jsonl (who=eval); a
call is refused once the total would pass common.SPEND_CAP (EVAL_SPEND_CAP_USD).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, FPS, SPEND_CAP, atomic_json, eval_spend  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "judge"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import judge as J  # noqa: E402  (call_openrouter, _parse_json, cost ledger)

PV = "pair-v1"
JUDGE_ID = "gflash"
EST_USD = 0.03                         # refuse a call when spend + this would pass the cap
SFPS = 3.0                             # strip frame rate
TW, TH = 96, 108                       # strip tile (avatar crop 320x360 scaled 0.3)
COLS = 18                              # 6 s per row at 3 fps
PAIR_DIR = E / "judge/pair"
CACHE = PAIR_DIR / "cache"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def sha(p: Path, n=12):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:n]


def flip(cid, a, b):
    """True -> system b is shown as 'A' (left). A per-clip coin: the lexicographically smaller system is left iff h=0."""
    h = int(hashlib.sha256(f"{cid}|{min(a, b)}|{max(a, b)}".encode()).hexdigest(), 16) % 2
    smaller_left = h == 0
    return (b < a) == smaller_left


# ----------------------------------------------------------------------------- media
def crop_frames(video: Path, fps: float, dur: float):
    """avatar render 640x360 -> center crop 320x360 -> TWxTH RGB frames at fps."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-t", f"{dur:.3f}",
           "-vf", f"fps={fps},crop=320:360:160:0,scale={TW}:{TH}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (TW * TH * 3)
    return np.frombuffer(raw[:n * TW * TH * 3], np.uint8).reshape(n, TH, TW, 3)


def spans_mask(spans, t):
    return any(a <= t <= b for a, b in spans)


def strips(vl: Path, vr: Path, dur: float, user_sp, av_sp, media: Path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, 13)
    fa, fb = crop_frames(vl, SFPS, dur), crop_frames(vr, SFPS, dur)
    n = min(len(fa), len(fb))
    rows_per_sheet = 2                     # (A row, B row) x 2 = 12 s per sheet
    per_sheet = COLS * rows_per_sheet
    out = []
    LH = 16                                # label band
    MB = 5                                 # speech-mark band under each tile
    for si, s0 in enumerate(range(0, n, per_sheet)):
        W = 30 + COLS * TW
        H = rows_per_sheet * 2 * (TH + MB) + rows_per_sheet * LH
        im = Image.new("RGB", (W, H), (20, 20, 20))
        d = ImageDraw.Draw(im)
        y = 0
        for r in range(rows_per_sheet):
            k0 = s0 + r * COLS
            if k0 >= n:
                break
            d.text((4, y + 1), f"t = {k0 / SFPS:.1f} .. {min(n, k0 + COLS) / SFPS:.1f} s   (green bar = USER speaking, "
                               f"orange = avatar voice)", fill=(230, 230, 230), font=font)
            y += LH
            for lab, fr in (("A", fa), ("B", fb)):
                d.text((8, y + TH // 2 - 8), lab, fill=(255, 255, 0), font=ImageFont.truetype(FONT, 16))
                for c in range(COLS):
                    k = k0 + c
                    if k >= n:
                        break
                    x = 30 + c * TW
                    im.paste(Image.fromarray(fr[k]), (x, y))
                    t = k / SFPS
                    if spans_mask(user_sp, t):
                        d.rectangle([x, y + TH, x + TW // 2 - 1, y + TH + MB - 1], fill=(40, 200, 60))
                    if spans_mask(av_sp, t):
                        d.rectangle([x + TW // 2, y + TH, x + TW - 1, y + TH + MB - 1], fill=(255, 150, 30))
                y += TH + MB
        p = media / f"strip{si}.jpg"
        im.save(p, quality=82)
        out.append(p)
    return out


def composite(cam: Path, vl: Path, vr: Path, voice: Path, dur: float, portrait: bool, out: Path):
    """[USER | A | B] 15 fps, height 360, labels; audio = user mic (from camera.mp4) + avatar voice."""
    ucrop = "crop=iw:iw*4/3:0:ih*0.08," if portrait else "crop=ih*3/4:ih:(iw-ih*3/4)/2:0,"
    lab = lambda t: (f"drawtext=fontfile={FONT}:text='{t}':x=6:y=6:fontsize=22:fontcolor=yellow:"  # noqa: E731
                     "box=1:boxcolor=black@0.6")
    fc = (f"[0:v]{ucrop}scale=270:360,setsar=1,fps=15,{lab('USER (context only)')}[u];"
          f"[1:v]crop=320:360:160:0,setsar=1,fps=15,{lab('A')}[a];"
          f"[2:v]crop=320:360:160:0,setsar=1,fps=15,{lab('B')}[b];"
          "[u][a][b]hstack=inputs=3,pad=iw:ih+26:0:0:black,"
          f"drawtext=fontfile={FONT}:text='t=%{{pts\\:flt\\:%.1f}}s':x=6:y=h-22:fontsize=18:fontcolor=white[v];"
          "[0:a][3:a]amix=2:duration=longest:normalize=0[aud]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(cam), "-i", str(vl), "-i", str(vr), "-i", str(voice),
           "-filter_complex", fc, "-map", "[v]", "-map", "[aud]", "-t", f"{dur:.3f}", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-b:a", "48k",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


# ----------------------------------------------------------------------------- prompt
def timeline(ann, dur):
    L = [f"Clip length {dur:.1f} s. What the user does: {ann.get('description', '') or '(a person in a video call)'}"]
    us = ann.get("user_speech") or []
    if us:
        L.append("User speech (heard by the avatars; the voice you hear from the avatar is the avatar's own reply):")
        for s in us[:40]:
            txt = (s.get("text") or "").strip()
            L.append(f"  {s['t0']:.1f}-{s['t1']:.1f} s" + (f": \"{txt[:140]}\"" if txt else ""))
    else:
        L.append("The user is silent.")
    cues = [c for c in (ann.get("cues") or []) if c.get("kind") != "command"]
    if cues:
        L.append("Moments the user does something a listener would react to:")
        for c in cues[:12]:
            L.append(f"  t={float(c['t']):.1f} s: {c.get('cue')} (a natural reaction: {c.get('expect', '')})")
    return "\n".join(L)


def prompt(ann, dur, n_strips):
    return f"""You compare TWO real-time avatars, A and B, that are the SAME cartoon character driven by two different
motion controllers. Both hear the same user and speak with the same recorded voice, so ONLY their non-verbal motion
differs (head, gaze, face, brows, smile, blinks, posture, arms). Judge behaviour, never rendering.

MEDIA:
- a VIDEO with three panels: the human USER's camera (context only — never judged) | avatar A | avatar B. The audio is
  the user's microphone mixed with the avatar's voice. Your video sampling is coarse, so also use:
- {n_strips} FRAME-STRIP sheets: A and B rows at 3 frames/s for the same moments (row label = time span; green bar =
  the user is speaking at that frame, orange bar = the avatar's voice is playing). Compare neighbouring tiles to see
  motion: identical tiles = frozen; jumpy changes = jitter; the same pose sequence repeating = a loop.

USER TIMELINE (ground truth):
{timeline(ann, dur)}

QUESTIONS (decide each one from evidence; "tie" only if you truly cannot tell them apart):
1. naturalness — which avatar moves more like an attentive, alive human conversation partner: continuous small
   motion, believable head/gaze/face dynamics, smooth (no jitter, no freezing, no robotic loops, no wild drifting or
   constant looking away), appropriate amount of motion while listening vs speaking?
2. reacts_to_user — which avatar's behaviour responds more to what the USER does and says, at the right moments:
   e.g. smiles back when the user smiles/laughs, nods or acknowledges at the user's pauses, looks at the user when
   addressed, turns/reacts to the user's gestures, movements and expressions, quiets while the user talks?

Return ONLY one JSON object, no markdown:
{{"observations": "<3-5 sentences comparing A and B with timestamps>",
 "naturalness": {{"winner": "A" | "B" | "tie", "confidence": 1 | 2 | 3, "why": "<one sentence>"}},
 "reacts_to_user": {{"winner": "A" | "B" | "tie", "confidence": 1 | 2 | 3, "why": "<one sentence, cite a user moment>"}}}}"""


# ----------------------------------------------------------------------------- one clip
def judge_clip(cid, a, b, swap=False, dry=False):
    man = manifest()
    c = man[cid]
    dur = float(c["duration_s"])
    va, vb = E / "runs" / a / cid / "avatar.mp4", E / "runs" / b / cid / "avatar.mp4"
    if not (va.exists() and vb.exists()):
        return {"clip": cid, "error": "missing render", "a_exists": va.exists(), "b_exists": vb.exists()}
    b_left = flip(cid, a, b) ^ bool(swap)
    vl, vr = (vb, va) if b_left else (va, vb)
    out_dir = PAIR_DIR / f"{a}__{b}"
    outp = out_dir / (cid + (".swap" if swap else "") + ".json")
    if outp.exists():
        r = json.loads(outp.read_text())
        if r.get("pv") == PV and r.get("media_key") and not r.get("error"):
            return r
    media = PAIR_DIR / "media" / f"{a}__{b}" / (cid + (".swap" if swap else ""))
    media.mkdir(parents=True, exist_ok=True)
    ann = json.loads(Path(c["annotation"]).read_text())
    from systems import utterances
    av_sp = [(u[0], u[1]) for u in utterances(c)]
    user_sp = [(s["t0"], s["t1"]) for s in (ann.get("user_speech") or [])]
    cam = E / "clips" / cid / "camera.mp4"
    voice = E / "clips" / cid / "voice.wav"
    comp = composite(cam, vl, vr, voice, dur, c["group"] == "seamless", media / "composite.mp4")
    sts = strips(vl, vr, dur, user_sp, av_sp, media)
    pr = prompt(ann, dur, len(sts))
    key = hashlib.sha256((PV + pr + sha(vl) + sha(vr) + sha(cam)).encode()).hexdigest()[:20]
    cp = CACHE / f"{key}.json"
    if cp.exists():
        raw, usd, usage = json.loads(cp.read_text())["raw"], 0.0, {"cached": True}
    else:
        if dry:
            return {"clip": cid, "dry": True, "media": str(media)}
        if os.environ.get("JUDGE_ALLOW_PAID") != "1":
            raise SystemExit("pairjudge: paid route, set JUDGE_ALLOW_PAID=1")
        if eval_spend() + EST_USD > SPEND_CAP:
            return {"clip": cid, "error": f"spend cap: eval total ${eval_spend():.3f} + ${EST_USD} > ${SPEND_CAP}"}
        JJ = J.JUDGES[JUDGE_ID]
        raw, usd, usage = J.call_openrouter(JJ["model"], JJ["price"], pr, comp, sts, f"pairjudge {a} vs {b} {cid}")
        CACHE.mkdir(parents=True, exist_ok=True)
        atomic_json(cp, {"raw": raw, "usd": usd, "usage": usage})

    def who(w):
        w = str(w or "").strip().upper()
        if w == "A":
            return b if b_left else a
        if w == "B":
            return a if b_left else b
        return "tie"
    res = {"pv": PV, "clip": cid, "group": c["group"], "user_person": c.get("user_person"), "a": a, "b": b,
           "shown_left": b if b_left else a, "swap": bool(swap), "media_key": key, "usd": float(usd),
           "naturalness": who((raw.get("naturalness") or {}).get("winner")),
           "naturalness_conf": (raw.get("naturalness") or {}).get("confidence"),
           "reacts_to_user": who((raw.get("reacts_to_user") or {}).get("winner")),
           "reacts_conf": (raw.get("reacts_to_user") or {}).get("confidence"), "raw": raw}
    atomic_json(outp, res)
    return res


# ----------------------------------------------------------------------------- summary
def score_of(r, axis, b):
    w = r.get(axis)
    return 1.0 if w == b else (0.5 if w == "tie" else 0.0)


def boot(v, cl=None, n=5000, seed=5):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    if cl is None:
        m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    else:
        keys = sorted(set(cl)); idx = {k: [i for i, x in enumerate(cl) if x == k] for k in keys}
        m = np.array([v[np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])].mean()
                      for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sign_p(wins, losses):
    """two-sided exact sign test (ties dropped)."""
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summary(a, b):
    d = PAIR_DIR / f"{a}__{b}"
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
    main = [r for r in rows if not r.get("swap") and not r.get("error") and r.get("naturalness")]
    sw = {r["clip"]: r for r in rows if r.get("swap") and not r.get("error")}
    res = {"a": a, "b": b, "judge": JUDGE_ID, "pv": PV, "n": len(main), "usd": round(sum(r.get("usd", 0) for r in rows), 4),
           "groups": {}}
    for grp in ("all", "seamless", "youtube"):
        g = [r for r in main if grp == "all" or r["group"] == grp]
        if not g:
            continue
        o = {"n": len(g)}
        for ax in ("naturalness", "reacts_to_user"):
            v = [score_of(r, ax, b) for r in g]
            wins = sum(1 for r in g if r[ax] == b); losses = sum(1 for r in g if r[ax] == a)
            o[ax] = {"pref_b": float(np.mean(v)), "ci95": boot(v), "ci95_person": boot(v, [r.get("user_person") or r["clip"] for r in g]),
                     "wins_b": wins, "wins_a": losses, "ties": len(g) - wins - losses, "sign_p": sign_p(wins, losses)}
        res["groups"][grp] = o
    if sw:
        cons = {}
        for ax in ("naturalness", "reacts_to_user"):
            pairs = [(r[ax], sw[r["clip"]][ax]) for r in main if r["clip"] in sw]
            cons[ax] = {"n": len(pairs), "same_winner": sum(1 for x, y in pairs if x == y),
                        "pairs": pairs}
        res["position_consistency"] = cons
    res["per_clip"] = {r["clip"]: {"nat": r["naturalness"], "react": r["reacts_to_user"], "left": r["shown_left"],
                                    "why_nat": (r["raw"].get("naturalness") or {}).get("why"),
                                    "why_react": (r["raw"].get("reacts_to_user") or {}).get("why")} for r in main}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["judge", "summary", "dry"])
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--ids"); ap.add_argument("--swap-ids", default="")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--out")
    x = ap.parse_args()
    if x.cmd in ("judge", "dry"):
        ids = x.ids.split(",")
        work = [(c, False) for c in ids] + [(c, True) for c in x.swap_ids.split(",") if c]
        with cf.ThreadPoolExecutor(x.jobs) as pool:
            futs = {pool.submit(judge_clip, c, x.a, x.b, s, x.cmd == "dry"): (c, s) for c, s in work}
            for f in cf.as_completed(futs):
                try:
                    r = f.result()
                    print(r.get("clip"), "swap" if r.get("swap") else "", "nat:", r.get("naturalness"), "react:",
                          r.get("reacts_to_user"), r.get("error") or "", f"${r.get('usd', 0):.4f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print("ERROR", futs[f], repr(e)[:600], flush=True)
        print(f"eval spend total ${eval_spend():.4f}")
    s = summary(x.a, x.b)
    out = Path(x.out) if x.out else PAIR_DIR / f"summary_{x.a}__{x.b}.json"
    atomic_json(out, s)
    for grp, g in s["groups"].items():
        print(grp, g["n"], {ax: (round(g[ax]["pref_b"], 3), [round(v, 2) for v in g[ax]["ci95"]], g[ax]["wins_b"], g[ax]["wins_a"],
                                 g[ax]["ties"], round(g[ax]["sign_p"], 3)) for ax in ("naturalness", "reacts_to_user")})


if __name__ == "__main__":
    main()
