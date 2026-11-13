#!/usr/bin/env python3
"""pass 2: the review set of the look upgrade — per character, BEFORE | AFTER through the real-time engine:

  <ch>_<pose>_720.png      neutral front / a smiling talking open mouth / an action pose (the muse: the wide desk scene)
  <ch>_<pose>_face2x.png   the face of the same frame, 2x
  <ch>_clip.mp4            5 s: a head turn, two blinks, talking, then a held pose (side by side)
  review.json              mean luminance of the character before / after per still, and the temporal flicker of the
                           held second of the clip (mean per-pixel standard deviation over the character, 8-bit)

  python tools/look_review.py --characters muse,fox,orb --out-dir DIR [--w 1280 --h 720 --preset taa] [--no-clip]
  python tools/look_review.py --tier-check muse --out-dir DIR     # tier switches on a held pose: frame-to-frame change
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
from rig_still import parse_pose  # noqa: E402
from anim.schema import IDX  # noqa: E402

TALK = "smile=0.7 mouth_open=0.75 lip_open=0.8 jaw=0.65 teeth_gap=0.5"
SETS = {
    "muse": dict(face=(0.33, 0.06, 0.67, 0.54), poses=[
        ("neutral", "REST"),
        ("talk", "REST " + TALK + " head_yaw=0.10"),
        ("action", "WAVE_UP_L cam_zoom=0.8 hand_r_x=0.1 hand_r_y=0.55 hand_r_z=0.7 prop4_hr=1 smile=0.8 mouth_open=0.5 "
                   "lip_open=0.6 jaw=0.4 root_x=-0.12 root_yaw=0.05 body_lean=0.3 foot_l_y=0.07")]),
    "fox": dict(face=(0.26, 0.02, 0.74, 0.68), poses=[
        ("neutral", "REST"),
        ("talk", "REST " + TALK + " head_yaw=0.15"),
        ("action", "WAVE_UP smile=0.8 mouth_open=0.8 lip_open=0.8 jaw=0.7 head_yaw=0.2 head_roll=0.06 brow_l=0.5 brow_r=0.5")]),
}
DEFAULT_SET = dict(face=(0.26, 0.05, 0.74, 0.73), poses=[
    ("neutral", "REST"),
    ("talk", "REST " + TALK + " head_yaw=0.12"),
    ("action", "WAVE_UP smile=0.8 mouth_open=0.6 lip_open=0.7 jaw=0.5 head_yaw=0.2 head_roll=0.06 brow_l=0.5 brow_r=0.5")])
CLIP_N, CLIP_HOLD = 120, 24


def clip_vec(base, i):
    """frame i of the 5 s clip: head turn (one sweep left and right), blinks at 1.2 s and 3.0 s, talking 1.8 .. 3.6 s,
    the last second held still (flicker shows there)."""
    v = base.copy()
    t = i/24.0
    if i >= CLIP_N - CLIP_HOLD:
        return v
    ramp = min(1.0, (CLIP_N - CLIP_HOLD - i)/12.0)
    v[IDX["head_yaw"]] = 0.38*math.sin(2*math.pi*t/4.0)*ramp
    v[IDX["head_pitch"]] = 0.06*math.sin(2*math.pi*t/2.3)*ramp
    for tb in (1.2, 3.0):
        b = max(0.0, 1.0 - abs(t - tb)/0.10)
        v[IDX["blink_l"]] = max(v[IDX["blink_l"]], b); v[IDX["blink_r"]] = max(v[IDX["blink_r"]], b)
    if 1.8 <= t <= 3.6:
        o = 0.5 + 0.5*math.sin(2*math.pi*(t - 1.8)*3.7)
        e = min(1.0, (t - 1.8)/0.15, (3.6 - t)/0.15)
        for k, g in (("mouth_open", 0.8), ("lip_open", 0.85), ("jaw", 0.7), ("smile", 0.5)):
            if k in IDX:
                v[IDX[k]] = g*o*e if k != "smile" else g*e
        if "voice" in IDX:
            v[IDX["voice"]] = 0.6*e
    return v


def settle(eng, vec, n):
    eng.reset_history()
    img = None
    for k in range(n):
        img = eng.render(k, vec)
    return np.array(img)


def side_by_side(a, b, labels=("BEFORE", "AFTER"), zoom=1, crop=None):
    from PIL import Image, ImageDraw
    halves = []
    for arr, lab in zip((a, b), labels):
        im = Image.fromarray(arr)
        if crop:
            w, h = im.size
            im = im.crop((int(crop[0]*w), int(crop[1]*h), int(crop[2]*w), int(crop[3]*h)))
        if zoom > 1:
            im = im.resize((im.width*zoom, im.height*zoom), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 8 + 7*len(lab), 14), fill=(0, 0, 0)); d.text((3, 2), lab, fill=(255, 255, 255))
        halves.append(im)
    sheet = Image.new("RGB", (halves[0].width + halves[1].width + 4, halves[0].height), (20, 20, 20))
    sheet.paste(halves[0], (0, 0)); sheet.paste(halves[1], (halves[0].width + 4, 0))
    return sheet


def lum(a):
    return 0.2126*a[..., 0] + 0.7152*a[..., 1] + 0.0722*a[..., 2]


def char_mask(eng, vec, img):
    """the character's pixels: where the frame differs from the same frame with the character far below the floor."""
    return None


def run_side(ch, w, h, preset, look, st, clip, settle_n):
    import rt_engine
    eng = rt_engine.RTEngine(ch, w, h, dict(rt_engine.resolve_preset(preset), look=look), backend="gpu")
    stills = {name: settle(eng, parse_pose(spec)[1], settle_n) for name, spec in st["poses"]}
    frames = []
    if clip:
        base = parse_pose("REST")[1]
        eng.reset_history()
        for k in range(6):
            eng.render(k, clip_vec(base, 0))
        frames = [np.array(eng.render(6 + i, clip_vec(base, i))) for i in range(CLIP_N)]
    eng.close()
    return stills, frames


def write_mp4(path, frames_a, frames_b):
    h, w = frames_a[0].shape[:2]
    W = 2*w + 4
    W += W % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{h}", "-r", "24",
                          "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for a, b in zip(frames_a, frames_b):
        fr = np.full((h, W, 3), 20, np.uint8)
        fr[:, :w] = a; fr[:, w + 4:2*w + 4] = b
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()


def tier_check(ch, out_dir, w=480, h=270):
    """a held pose while the quality tier steps down and back up: the frame-to-frame change at a switch against the
    change between ordinary frames (a visible pop = a switch many times the ordinary change)."""
    import rt_engine
    from live.backends.rt import default_preset
    p = dict(rt_engine.resolve_preset(default_preset(ch, w, h, 24)), look=True, tiers=True)
    eng = rt_engine.RTEngine(ch, w, h, p, backend="gpu")
    vec = parse_pose("REST")[1]
    eng.reset_history()
    n = len(eng.tiers); t0 = eng.tier
    plan = {30: max(t0 - 1, 0), 60: max(t0 - 2, 0), 90: max(t0 - 1, 0), 120: t0, 150: min(t0 + 1, n - 1), 180: t0}
    prev, rows = None, []
    for i in range(210):
        if i in plan:
            eng.set_tier(plan[i])
        img = np.array(eng.render(i, vec)).astype(np.float32)
        if prev is not None and i > 12:
            rows.append((i, float(np.abs(img - prev).mean()), float(np.percentile(np.abs(img - prev).max(axis=2), 99.5)), i in plan))
        prev = img
    eng.close()
    base = [r for r in rows if not any(0 <= r[0] - s <= 3 for s in plan)]
    sw = [r for r in rows if any(0 <= r[0] - s <= 3 for s in plan)]
    res = dict(character=ch, tiers=eng.tiers, plan={str(k): v for k, v in plan.items()},
               ordinary_mean=float(np.mean([r[1] for r in base])), ordinary_p995=float(np.mean([r[2] for r in base])),
               switch_mean_max=float(max(r[1] for r in sw)), switch_p995_max=float(max(r[2] for r in sw)))
    json.dump(res, open(os.path.join(out_dir, f"tier_check_{ch}.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--characters", default="muse,fox,orb")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--preset", default="taa")
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--tier-check")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.tier_check:
        for ch in a.tier_check.split(","):
            tier_check(ch, a.out_dir)
        return
    rj = os.path.join(a.out_dir, "review.json")
    review = json.load(open(rj)) if os.path.exists(rj) else {}
    for ch in a.characters.split(","):
        st = SETS.get(ch, DEFAULT_SET)
        sb, fb = run_side(ch, a.w, a.h, a.preset, False, st, not a.no_clip, a.settle)
        sa, fa = run_side(ch, a.w, a.h, a.preset, True, st, not a.no_clip, a.settle)
        rec = {}
        for name, _ in st["poses"]:
            side_by_side(sb[name], sa[name]).save(os.path.join(a.out_dir, f"{ch}_{name}_720.png"))
            crop = st["face"] if name != "action" else None
            if crop:
                side_by_side(sb[name], sa[name], zoom=2, crop=crop).save(os.path.join(a.out_dir, f"{ch}_{name}_face2x.png"))
            # the character = where the two renders can differ at all is not known here: the centre box of the frame
            x0, y0, x1, y1 = (int(st["face"][0]*a.w), int(st["face"][1]*a.h), int(st["face"][2]*a.w), int(st["face"][3]*a.h))
            lb, la = float(lum(sb[name][y0:y1, x0:x1].astype(np.float32)).mean()), float(lum(sa[name][y0:y1, x0:x1].astype(np.float32)).mean())
            rec[name] = dict(lum_before=round(lb, 2), lum_after=round(la, 2), lum_change_pct=round(100.0*(la/lb - 1.0), 2))
        if fb:
            write_mp4(os.path.join(a.out_dir, f"{ch}_clip.mp4"), fb, fa)
            x0, y0, x1, y1 = (int(st["face"][0]*a.w), int(st["face"][1]*a.h), int(st["face"][2]*a.w), int(st["face"][3]*a.h))
            for lab, fr in (("before", fb), ("after", fa)):
                hold = np.stack([lum(f[y0:y1, x0:x1].astype(np.float32)) for f in fr[-CLIP_HOLD + 6:]])
                rec[f"flicker_{lab}"] = round(float(hold.std(axis=0).mean()), 3)
                rec[f"flicker_p99_{lab}"] = round(float(np.percentile(hold.std(axis=0), 99)), 3)
        review[ch] = rec
        json.dump(review, open(rj, "w"), indent=1)
        print(ch, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
