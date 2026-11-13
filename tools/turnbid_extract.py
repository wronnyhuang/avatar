#!/usr/bin/env python3
"""Turn-bid: extra HELD-OUT Seamless windows for the turn-bid false-trigger eval.

  nice -n 10 python3 -I tools/turnbid_extract.py --out DIR [--n 24] [--dur 60] [--jobs 6] [--plan-only]

Picks windows of Seamless dev videos (naturalistic + improvised) whose camera person is NOT one of the 8 people in
the eval kit's sl_* eval clips, one window per person: the partner talks >= 35 % of it, the camera person 15-65 %, >= 4
turn changes (dataset VAD metadata). Each window goes through motion-model/evalkit/extract.py UNCHANGED except for the
output dir (DIR/clips/<id>) and the perception.js used (this worktree's app/perception.js = the live page's copy):
native MediaPipe at the page's rates (face 18 / hand 10 / pose 6 Hz) -> exact JS perception, causal. mic.wav / voice.wav
are the two people's tracks, level-normalised like the sl clips. Writes DIR/manifest.json with the partner's words
(avatar_words) and the user's words, like the eval kit's manifest. Read-only on the kit and the dataset.

Env: EVALKIT = the eval kit dir (default <repo>/motion-model/evalkit); EVAL_MANIFEST = the eval clips manifest whose sl_*
sessions are excluded (default $EVAL_OUT/clips/manifest.json; missing = nothing excluded); AVATAR_DATASETS = datasets root
(default <repo>/data/datasets; Seamless Interaction under <root>/seamless-interaction)."""
import argparse, json, os, sys, concurrent.futures as cf
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get("EVALKIT") or WT / "motion-model" / "evalkit")
SEAM = Path(os.environ.get("AVATAR_DATASETS") or WT / "data" / "datasets") / "seamless-interaction"
EVAL_PERSONS = {"P0216", "P0370", "P0371", "P0299A", "P1040", "P1281A", "P1346A", "P1282A"}
_MANIFEST = Path(os.environ.get("EVAL_MANIFEST") or Path(os.environ.get("EVAL_OUT") or WT / "runs" / "eval") / "clips" / "manifest.json")
EVAL_SESSIONS = ({c["session"] for c in json.load(open(_MANIFEST))["clips"] if c["id"].startswith("sl_")}
                 if _MANIFEST.exists() else set())


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError:
        return []


def speak_mask(segs, t0, dur, step=0.1):
    n = int(dur / step); m = [0] * n
    for s in segs:
        a = max(0, int((s["start"] - t0) / step)); b = min(n, int((s["end"] - t0) / step) + 1)
        for i in range(a, b): m[i] = 1
    return m


def plan(n_want, dur):
    cands = []
    for label in ("naturalistic", "improvised"):
        D = SEAM / label / "dev"
        vids = sorted(p.stem for p in (D / "video").glob("*.mp4"))
        by_sess = {}
        for v in vids:
            sess, person = v.rsplit("_", 1)
            by_sess.setdefault(sess, []).append(person)
        for sess, ps in by_sess.items():
            if len(ps) != 2 or sess in EVAL_SESSIONS:
                continue
            for me, other in ((ps[0], ps[1]), (ps[1], ps[0])):
                if me in EVAL_PERSONS:
                    continue
                vu = jl(D / "metadata/vad" / f"{sess}_{me}.jsonl"); vo = jl(D / "metadata/vad" / f"{sess}_{other}.jsonl")
                if not vu or not vo:
                    continue
                end = max(s["end"] for s in vu + vo)
                best = None
                t0 = 4.0
                while t0 + dur <= end - 2:
                    mu = speak_mask(vu, t0, dur); mo = speak_mask(vo, t0, dur)
                    fu = sum(mu) / len(mu); fo = sum(mo) / len(mo)
                    who = [1 if (a and not b) else 2 if (b and not a) else 0 for a, b in zip(mu, mo)]
                    seq = [w for w in who if w]
                    tc = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
                    if fo >= 0.35 and 0.15 <= fu <= 0.65 and tc >= 4:
                        score = min(fu, fo) + 0.02 * min(tc, 10)
                        if not best or score > best[0]:
                            best = (score, t0, fu, fo, tc)
                    t0 += 6.0
                if best:
                    cands.append(dict(label=label, sess=sess, me=me, other=other, score=best[0], t0=best[1],
                                      fu=round(best[2], 3), fo=round(best[3], 3), tc=best[4]))
    # one window per person, alternate labels, best score first
    cands.sort(key=lambda c: -c["score"])
    seen, out = set(), {"naturalistic": [], "improvised": []}
    for c in cands:
        if c["me"] in seen:
            continue
        seen.add(c["me"]); out[c["label"]].append(c)
    pick = []
    while len(pick) < n_want and (out["naturalistic"] or out["improvised"]):
        for k in ("naturalistic", "improvised"):
            if out[k] and len(pick) < n_want:
                pick.append(out[k].pop(0))
    return pick


def words(D, f, t0, dur):
    w = []
    for seg in jl(D / "metadata/transcript" / f"{f}.jsonl"):
        for x in seg.get("words", []):
            if x.get("start") is not None and x.get("end") is not None and t0 <= x["start"] < t0 + dur:
                w.append([round(x["start"] - t0, 3), round(x["end"] - t0, 3), x.get("word", "")])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dur", type=float, default=60.0); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pick = plan(a.n, a.dur)
    clips = []
    for c in pick:
        D = SEAM / c["label"] / "dev"
        cid = f"tb_{c['sess'].split('_')[1]}_{c['sess'].split('_')[2][-4:]}_{c['me']}"
        clips.append({"id": cid, "group": "seamless", "label": c["label"], "duration_s": a.dur,
                      "video": str(D / "video" / f"{c['sess']}_{c['me']}.mp4"), "v_t0": c["t0"],
                      "user_wav": str(D / "audio" / f"{c['sess']}_{c['me']}.wav"), "user_wav_t0": c["t0"],
                      "voice_wav": str(D / "audio" / f"{c['sess']}_{c['other']}.wav"), "voice_t0": c["t0"],
                      "session": c["sess"], "user_person": c["me"], "avatar_person": c["other"],
                      "window_stats": {"user_speak": c["fu"], "avatar_speak": c["fo"], "turn_changes": c["tc"]},
                      "avatar_words": words(D, f"{c['sess']}_{c['other']}", c["t0"], a.dur),
                      "user_words": words(D, f"{c['sess']}_{c['me']}", c["t0"], a.dur)})
    (out / "manifest.json").write_text(json.dumps({"version": 1, "clips": clips}, indent=1))
    print(f"{len(clips)} windows planned -> {out/'manifest.json'}", flush=True)
    for c in clips:
        print(c["id"], c["label"], c["v_t0"], c["window_stats"], flush=True)
    if a.plan_only:
        return
    sys.path.insert(0, str(KIT))
    import extract as X  # the eval kit (read-only)
    X.E = out                                  # outputs -> DIR/clips/<id>
    X.PERCEPTION = WT / "app/perception.js"    # the live page's perception
    import numpy as np
    X.audio42 = lambda x, n: np.zeros((n, 42), np.float32)   # canon audio feats not needed here
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(X.extract, c): c["id"] for c in clips}
        for f in cf.as_completed(futs):
            try:
                r = f.result(); print("done", futs[f], r.get("face_frac"), r.get("seconds"), flush=True)
            except Exception as e:  # noqa
                print("FAIL", futs[f], repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
