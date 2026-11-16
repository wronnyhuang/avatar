#!/usr/bin/env python3
"""TRUE live recordings vs the offline renders of the same recorded rig track — the check that the clips on the
page show what the live demo shows. Run with the server's python (cv2).

  run_live.cjs ... --param rec=1 against a server with the real renderer started with AVATAR_REC_ROOT=<dir> (the
  page's own session recorder: the stage as shown, assembled by the server ~150 s after the session ends), then
  render_clips.py on the same run, then:
  python tools/sem_eval/compare_live.py --run <runs>/<char>/<mode>/<pass> --clips <clips>/<char>/<mode> --rec <mp4> --out DIR
        [--times 0,1,2,...  (s after the command; default 8 over call .. end + 0.5)]

Per command: DIR/<id>.cmp.jpg (top row = live recording, bottom = offline render, same seconds after the command) and
DIR/compare.json with a per-cell mean abs pixel difference on 160x90 grey thumbnails (live vs offline; the live stream
is VP8 at ~392x220 so expect ~10-25 of 255 for matching poses; a different pose shows as a spike).
Alignment: recording time = (call epoch ms - the recording's client_ts) / 1000 (recstore meta), accurate to ~0.1-0.3 s.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def grab(path, t, crop=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    if crop:
        h, w = fr.shape[:2]
        fr = fr[int(h * crop[1]):int(h * crop[3]), int(w * crop[0]):int(w * crop[2])]
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = None
    for p in glob.glob(os.path.join(os.path.dirname(a.rec), "*.json")) + glob.glob(a.rec[:-4] + ".json"):
        try:
            m = json.load(open(p))
            if m.get("client_ts"):
                meta = m
        except ValueError:
            pass
    if meta is None:   # the spool meta (index entry) — search the rec root
        root = os.path.dirname(os.path.dirname(a.rec))
        for p in glob.glob(os.path.join(root, "**", "meta.json"), recursive=True) + glob.glob(os.path.join(root, "*.json")):
            try:
                m = json.load(open(p))
            except ValueError:
                continue
            for e in (m if isinstance(m, list) else [m]):
                if isinstance(e, dict) and e.get("client_ts") and os.path.basename(a.rec).split("-")[-1][:-4] in str(e.get("rid", "")):
                    meta = e
    if meta is None:
        raise SystemExit("no client_ts for the recording (looked for its meta json)")
    res = {"rec": a.rec, "client_ts": meta["client_ts"], "cmds": {}}
    for rj in sorted(glob.glob(os.path.join(a.run, "*.json"))):
        if rj.endswith("run.json"):
            continue
        R = json.load(open(rj))
        cid = R["id"]
        off = os.path.join(a.clips, cid + ".mp4")
        if not os.path.exists(off) or not R.get("call_epoch_ms"):
            continue
        t_rec = (R["call_epoch_ms"] - meta["client_ts"]) / 1000.0
        t_off = 1.0                       # run_live --pre-s default: the rig file starts 1 s before the call
        end = ((R.get("timings") or {}).get("end_ms") or 3000) / 1000.0
        ts = [float(x) for x in a.times.split(",")] if a.times else list(np.linspace(0, end + 0.5, 8))
        top, bot, diffs = [], [], []
        for t in ts:
            L = grab(a.rec, t_rec + t, crop=(0, 0, 1, 1))
            O = grab(off, t_off + t)
            if L is None or O is None:
                continue
            L = cv2.resize(L, (O.shape[1], O.shape[0]))
            L[int(L.shape[0] * 0.72):, int(L.shape[1] * 0.74):] = 0          # the self-view inset (a test fixture): blanked
            O2 = O.copy()
            O2[int(O.shape[0] * 0.72):, int(O.shape[1] * 0.74):] = 0
            g = lambda x: cv2.cvtColor(cv2.resize(x, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            diffs.append(round(float(np.abs(g(L) - g(O2)).mean()), 1))
            for im, lab in ((L, "live"), (O2, "offline")):
                cv2.putText(im, f"{lab} t={t:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            top.append(cv2.resize(L, (240, 135)))
            bot.append(cv2.resize(O2, (240, 135)))
        if top:
            cv2.imwrite(os.path.join(a.out, cid + ".cmp.jpg"), np.vstack([np.hstack(top), np.hstack(bot)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        res["cmds"][cid] = {"t_rec_call": round(t_rec, 2), "times": [round(x, 2) for x in ts], "mean_abs_diff": diffs}
        print(cid, diffs)
    json.dump(res, open(os.path.join(a.out, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
