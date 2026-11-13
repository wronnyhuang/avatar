"""All extracted clips -> one dataset of per-person 30 Hz segments with motion, voice, partner and state.

Usage: python -I build_dataset.py <scratch>   -> <scratch>/dataset.npz + dataset_meta.json
Per frame arrays (concatenated over segments):
  M     (N, 17) smoothed motion table (motionlib.COLS), relative to the person's clip median
  Mraw  (N, 17) unsmoothed
  blink (N,) 1 at blink onset frames; blinkdur (N,) duration (s) at onsets
  st    (N,) 0 idle 1 speaking 2 listening
  sv    (N, 4) SELF voice  [level 0..1, f0 semitones/12 rel. speaker median, voiced, speech]
  pv    (N, 4) PARTNER voice (same)
  pf    (N, 9) PARTNER face [present, smile, jaw, brow, pitch_vel/30, yaw_vel/30, pitch/10, yaw/20, arm log1p(sw/s)]
  pM    (N, 17) partner smoothed motion table (NaN when no partner) - for mirroring stats
  seg   (S, 2) [start, end) into the frame arrays; segvid (S,) video index; segperson (S,)
"""
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from motionlib import *


def blinks(bm, segs):
    on = np.zeros(len(bm), np.float32); dur = np.zeros(len(bm), np.float32)
    for a, b in segs:
        x = np.nan_to_num(bm[a:b], nan=0.0)
        base = medfilt(x, 61)
        y = x - base
        for s, e in segments(y > 0.18, 1):
            pk = y[s:e].max()
            if pk < 0.28 or x[s:e].max() < 0.35 or e - s > 15:
                continue
            on[a + s] = 1
            dur[a + s] = max(1, int((y[s:e] > 0.5 * pk).sum())) / FPS + 1.0 / FPS
    return on, dur


def voice_feats(v, mask, f0med):
    """v: (N,4) [db, f0, voiced, speech] -> masked model features"""
    N = len(v)
    out = np.zeros((N, 4), np.float32)
    return out


def main():
    root = Path(sys.argv[1])
    man = {}
    for mf in ("manifest.jsonl", "manifest_dyad.jsonl"):
        if (root / mf).exists():
            for l in (root / mf).read_text().splitlines():
                try:
                    r = json.loads(l); man[Path(r["file"]).stem] = r
                except Exception:
                    pass
    dy = sorted((root / "feats_dyad").glob("*.npz"))
    dstems = {f.stem for f in dy}
    # a clip re-extracted by the dyad panel detector supersedes its single-pass version
    rej = {f.stem.split(".")[0] for d_ in ("feats_dyad_reject", "feats_dyad_dup") if (root / d_).exists()
           for f in (root / d_).iterdir()}
    files = [f for f in sorted((root / "feats").glob("*.npz")) if f.stem not in dstems] + dy
    A = {k: [] for k in ("M", "Mraw", "blink", "blinkdur", "st", "sv", "pv", "pf", "pM", "sraw")}
    seg, segvid, segperson, vids = [], [], [], []
    n0 = 0
    for f in files:
        if f.stem not in man:
            continue
        d = np.load(f)
        if str(d["layout"]) == "dyad-room":      # corner webcams over gameplay / two-shots: not clean dyads
            continue
        if str(d["layout"]) == "lr" and f.stem in rej:   # the panel detector rejected this split (dup person, ...)
            continue
        S = len(d["streams"]); N = len(d["cut"])
        if N < 300:
            continue
        vinfo = dict(man[f.stem]); vinfo["layout"] = str(d["layout"]); vinfo["persons"] = []
        v30, floor, peak = voice_30hz(d["prosody"], N)
        speech = v30[:, 3] > 0
        cuts = cut_mask(d["cut"])
        dyok = d["dyad_ok"].astype(bool)[:N] if "dyad_ok" in d.files else np.ones(N, bool)
        if "dyad_ok" in d.files:
            cuts = cuts | np.concatenate([[False], dyok[1:] != dyok[:-1]])
        tabs = []; pres = []; extras = []
        for s in range(S):
            T, p, ex = person_table(d, s)
            # median-relative (neutral)
            if p.sum() < 150:
                tabs.append(None); pres.append(p); extras.append(ex); continue
            med = np.nanmedian(T[p], 0)
            for j in (0, 1, 2, 3, 4, 5, 6, 7):
                T[:, j] -= med[j]
            tabs.append(T); pres.append(p); extras.append(ex)
        if all(t is None for t in tabs):
            continue
        js = [jaw_std(t[:, C["jaw"]]) if t is not None else np.zeros(N) for t in tabs]
        talk = assign_speech_jaw(speech, js, pres)
        lev = np.clip((v30[:, 0] - floor) / max(peak - floor, 6.0), 0, 1.2)
        # f0 relative per person: median f0 over that person's own talk frames
        vi = len(vids)
        for s in range(S):
            T = tabs[s]
            if T is None:
                continue
            o = 1 - s if S == 2 else None
            segs = clean_segments(T, pres[s] & (dyok if S == 2 else True), cuts, extras[s]["face_w"])
            if not segs:
                continue
            Sm = smooth_table(T, segs)
            # arm energy: 1-s nan-aware moving average of wrist speed, then log1p (an envelope, not a jitter)
            for TT in (Sm,):
                a_ = TT[:, C["arm"]]; ok_ = np.isfinite(a_)
                num = np.convolve(np.where(ok_, a_, 0), np.ones(30), "same"); den = np.convolve(ok_.astype(float), np.ones(30), "same")
                TT[:, C["arm"]] = np.where(den > 5, np.log1p(num / np.maximum(den, 1)), np.nan)
            bm = (T[:, C["blink_l"]] + T[:, C["blink_r"]]) / 2
            bon, bdur = blinks(bm, segs)
            other_talk = talk[o] if o is not None else None
            st = states(talk[s], speech, other_talk)
            selfm = talk[s]
            partm = (other_talk if other_talk is not None else np.zeros(N, bool)) | (speech & ~talk[s])
            f0 = v30[:, 1]
            def vf(mask):
                g = mask.astype(np.float32)
                fv = f0 * mask
                med = np.median(fv[fv > 0]) if (fv > 0).sum() > 50 else 150.0
                semi = np.where(fv > 0, 12 * np.log2(np.maximum(fv, 1) / med), 0.0) / 12.0
                return np.stack([lev * g, np.clip(semi, -1, 1), v30[:, 2] * g, g], 1).astype(np.float32), med
            sv, f0s = vf(selfm); pv, f0p = vf(partm)
            pf = np.zeros((N, 9), np.float32)
            pMt = np.full((N, len(COLS)), np.nan, np.float32)
            if o is not None and tabs[o] is not None:
                PT = tabs[o]; pp = pres[o]
                psegs = clean_segments(PT, pp & dyok, cuts, extras[o]["face_w"], min_sec=1.0)
                PS = smooth_table(PT, psegs)
                a_ = PS[:, C["arm"]]; ok_ = np.isfinite(a_)
                num = np.convolve(np.where(ok_, a_, 0), np.ones(30), "same"); den = np.convolve(ok_.astype(float), np.ones(30), "same")
                PS[:, C["arm"]] = np.where(den > 5, np.log1p(num / np.maximum(den, 1)), np.nan)
                okp = np.zeros(N, bool)
                for a, b in psegs:
                    okp[a:b] = True
                pMt[okp] = PS[okp]
                vel = np.zeros((N, 2)); vel[1:] = np.diff(PS[:, [C["pitch"], C["yaw"]]], axis=0) * FPS
                pf[:, 0] = okp
                pf[:, 1] = PS[:, C["smile"]]; pf[:, 2] = PS[:, C["jaw"]]
                pf[:, 3] = (PS[:, C["brow_l"]] + PS[:, C["brow_r"]]) / 2
                pf[:, 4] = np.clip(vel[:, 0] / 30, -3, 3); pf[:, 5] = np.clip(vel[:, 1] / 30, -3, 3)
                pf[:, 6] = PS[:, C["pitch"]] / 10; pf[:, 7] = PS[:, C["yaw"]] / 20
                pf[:, 8] = np.nan_to_num(PS[:, C["arm"]], nan=0.0)
                pf[~okp] = 0
            pf = np.nan_to_num(pf)
            pi = len(vinfo["persons"])
            vinfo["persons"].append(dict(stream=str(d["streams"][s]), f0_self=float(f0s),
                                         frames=int(sum(b - a for a, b in segs)),
                                         speak_frac=float(np.mean([np.mean(st[a:b] == 1) for a, b in segs]))))
            for a, b in segs:
                A["M"].append(Sm[a:b]); A["Mraw"].append(T[a:b]); A["blink"].append(bon[a:b])
                A["blinkdur"].append(bdur[a:b]); A["st"].append(st[a:b]); A["sv"].append(sv[a:b])
                A["pv"].append(pv[a:b]); A["pf"].append(pf[a:b]); A["pM"].append(pMt[a:b])
                A["sraw"].append(np.stack([speech[a:b], f0[a:b], lev[a:b]], 1).astype(np.float32))
                seg.append((n0, n0 + b - a)); segvid.append(vi); segperson.append(pi); n0 += b - a
        vids.append(vinfo)
        print(f"{f.stem} {vinfo['layout']} persons={len(vinfo['persons'])} frames={n0}", flush=True)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in A.items()}
    np.savez_compressed(root / "dataset.npz", seg=np.array(seg), segvid=np.array(segvid),
                        segperson=np.array(segperson), **out)
    (root / "dataset_meta.json").write_text(json.dumps(vids, indent=1))
    print(f"[dataset] {len(vids)} videos {len(seg)} segments {n0} frames = {n0/FPS/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
