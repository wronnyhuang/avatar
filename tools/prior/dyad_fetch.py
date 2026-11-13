"""Fetch TWO-PERSON conversational clips (remote interviews, video calls, split-screen / PiP podcasts) from YouTube.

Media is untrusted data and lives in scratch only: <scratch>/media/dyad/<id>_<start>.mp4.
Appends <scratch>/manifest_dyad.jsonl (same fields as manifest.jsonl, category="dyad"). Skips ids already present in
manifest.jsonl or manifest_dyad.jsonl. No pre-filter: the layout step in dyad_extract.py rejects non-dyads.
Usage: python -I dyad_fetch.py <scratch> [--per-query N] [--win S] [--workers K] [--max-clips M]
       python -I dyad_fetch.py <scratch> --harvest MINUTES [--extra 3] [--max-clips 40]   (more windows of confirmed dyads)
"""
import json, subprocess, sys, argparse, concurrent.futures as cf, time
from pathlib import Path

QUERIES = [
    "split screen podcast full episode",
    "remote interview zoom recording full",
    "riverside fm podcast episode",
    "zoom call interview two people",
    "video call conversation full",
    "streamyard interview",
    "skype interview split screen",
    "podcast remote guest side by side",
    "virtual interview webcam both",
    "facetime call conversation",
    "two person zoom podcast",
    "squadcast podcast episode",
    "zoom interview author conversation webcam",
    "remote guest interview picture in picture",
    "news anchor remote guest interview split screen",
    "reaction video facecam picture in picture interview",
    "webinar fireside chat zoom two speakers",
    "zencastr podcast video episode",
]
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))
LICENSE = "YouTube Standard License (not redistributed; research features only)"


def search(q, n):
    cmd = [YTDLP, "--flat-playlist", "--no-warnings", "-J", f"ytsearch{n}:{q}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        js = json.loads(out.stdout)
    except Exception:
        return q, []
    return q, [e for e in js.get("entries", []) if e]


def fetch(job):
    vid, start, win, dest = job["id"], job["start"], job["win"], Path(job["path"])
    if dest.exists() and dest.stat().st_size > 50000:
        return job, True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [YTDLP, "-q", "--no-warnings", "--js-runtimes", "node", "--no-playlist",
           "-f", "bv*[height<=480][height>=240][ext=mp4][protocol=https]+ba[ext=m4a]/b[height<=480][ext=mp4]",
           "--download-sections", f"*{start}-{start + win}",
           "-o", str(dest), f"https://www.youtube.com/watch?v={vid}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ok = dest.exists() and dest.stat().st_size > 50000
        return job, ok, (r.stderr or "")[-300:]
    except Exception as e:
        return job, False, str(e)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--per-query", type=int, default=10)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=90)
    a = ap.parse_args()
    root = Path(a.scratch)
    man = root / "manifest_dyad.jsonl"
    seen = set()
    for m in (root / "manifest.jsonl", man):
        if m.exists():
            for l in m.read_text().splitlines():
                try: seen.add(json.loads(l)["id"])
                except Exception: pass
    # searches in parallel; results kept in rank order per query
    per_q = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for q, ents in ex.map(lambda q: search(q, a.per_query), QUERIES):
            keep = []
            for e in ents:
                d = e.get("duration") or 0
                if not (240 <= d <= 3 * 3600) or e.get("live_status") in ("is_live", "is_upcoming"):
                    continue
                keep.append(e)
            per_q[q] = keep
            print(f"[search] {q!r} -> {len(keep)}", flush=True)
    # round-robin over queries by rank, so the best-ranked hits of every query download first
    jobs = []
    for rank in range(max(len(v) for v in per_q.values()) if per_q else 0):
        for q in QUERIES:
            v = per_q.get(q, [])
            if rank >= len(v):
                continue
            e = v[rank]
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            d = int(e["duration"])
            win = min(a.win, d - 60)
            starts = [int(max(30, 0.3 * d))]
            if d > 1200:
                starts.append(int(0.65 * d))
            for s in starts:
                jobs.append(dict(id=e["id"], start=s, win=win, query=q, title=e.get("title"),
                                 channel=e.get("channel") or e.get("uploader"), vdur=d,
                                 path=str(root / "media" / "dyad" / f"{e['id']}_{s}.mp4")))
    jobs = jobs[: a.max_clips]
    print(f"[fetch] {len(jobs)} clips", flush=True)
    t0 = time.time(); nok = 0
    (root / "jobs_dyad.json").write_text(json.dumps(jobs))
    with cf.ThreadPoolExecutor(a.workers) as ex, open(man, "a") as mf:
        futs = [ex.submit(fetch, j) for j in jobs]
        for fut in cf.as_completed(futs):
            job, ok, msg = fut.result()
            if ok:
                nok += 1
                rec = dict(id=job["id"], url=f"https://www.youtube.com/watch?v={job['id']}", title=job["title"],
                           channel=job["channel"], category="dyad", query=job["query"], start=job["start"],
                           dur=job["win"], video_dur=job["vdur"], file=job["path"], license=LICENSE)
                mf.write(json.dumps(rec) + "\n"); mf.flush()
                if nok % 10 == 0:
                    print(f"[fetch] ok={nok} t={time.time()-t0:.0f}s", flush=True)
            else:
                print(f"[fail] {job['id']} {msg}", flush=True)
    print(f"[done] ok={nok}/{len(jobs)} in {time.time()-t0:.0f}s", flush=True)


def harvest():
    """--harvest MINUTES: for videos CONFIRMED as dyads (feats_dyad_accept/*.json, feats_dyad/*.npz, or the old
    lr/tb split-screen feats), download up to --extra more windows per video (split-screen episodes keep their
    layout for the whole episode, so these are near-certain dyad clips). Polls until MINUTES have passed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--harvest", type=float, required=True)
    ap.add_argument("--extra", type=int, default=3)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=40)
    a = ap.parse_args()
    root = Path(a.scratch); man = root / "manifest_dyad.jsonl"
    t_end = time.time() + 60 * a.harvest
    recs = {}

    def load():
        for m in (root / "manifest.jsonl", man):
            if m.exists():
                for l in m.read_text().splitlines():
                    try: r = json.loads(l)
                    except Exception: continue
                    recs.setdefault(r["id"], dict(r=r, starts=set()))["starts"].add(int(r["start"]))

    def confirmed():
        ids = []
        for f in sorted((root / "feats_dyad_accept").glob("*.json"), key=lambda f: f.stat().st_mtime):
            try: js = json.loads(f.read_text())
            except Exception: continue
            lay = js.get("layout"); cov = js.get("info", {}).get("layout_cover", 0)
            if lay in ("dyad-lr", "dyad-tb", "dyad-pip") or cov >= 0.75:
                ids.append(f.stem.rsplit("_", 1)[0])
        try:
            import numpy as np
            for f in sorted((root / "feats").glob("*.npz")):
                z = np.load(f)
                if str(z["layout"]) in ("lr", "tb") and float(z["present"].mean(1).min()) > 0.6:
                    ids.append(f.stem.rsplit("_", 1)[0])
        except Exception:
            pass
        out = []
        for i in ids:
            if i not in out: out.append(i)
        return out

    done_ids = set(); nq = 0; t0 = time.time(); nok = 0
    ex = cf.ThreadPoolExecutor(a.workers); futs = []
    mf = open(man, "a")
    while time.time() < t_end and nq < a.max_clips:
        load()
        for vid in confirmed():
            if vid in done_ids or vid not in recs:
                continue
            done_ids.add(vid)
            r = recs[vid]["r"]; d = int(r["video_dur"]); have = recs[vid]["starts"]
            win = min(a.win, d - 60)
            added = 0
            for fr in (0.12, 0.45, 0.82, 0.55, 0.2):
                s0 = int(max(30, fr * d))
                if s0 + win > d - 5 or any(abs(s0 - h) < win + 10 for h in have) or added >= a.extra:
                    continue
                have.add(s0); added += 1; nq += 1
                job = dict(id=vid, start=s0, win=win, query="harvest:" + str(r.get("query")), title=r.get("title"),
                           channel=r.get("channel"), vdur=d, path=str(root / "media" / "dyad" / f"{vid}_{s0}.mp4"))
                futs.append(ex.submit(fetch, job))
            print(f"[harvest] {vid} +{added} windows (queued {nq})", flush=True)
        for fut in [f for f in futs if f.done()]:
            futs.remove(fut)
            job, ok, msg = fut.result()
            if ok:
                nok += 1
                rec = dict(id=job["id"], url=f"https://www.youtube.com/watch?v={job['id']}", title=job["title"],
                           channel=job["channel"], category="dyad", query=job["query"], start=job["start"],
                           dur=job["win"], video_dur=job["vdur"], file=job["path"], license=LICENSE)
                mf.write(json.dumps(rec) + "\n"); mf.flush()
            else:
                print(f"[fail] {job['id']} {msg}", flush=True)
        time.sleep(10)
    for fut in cf.as_completed(futs):
        job, ok, msg = fut.result()
        if ok:
            nok += 1
            rec = dict(id=job["id"], url=f"https://www.youtube.com/watch?v={job['id']}", title=job["title"],
                       channel=job["channel"], category="dyad", query=job["query"], start=job["start"],
                       dur=job["win"], video_dur=job["vdur"], file=job["path"], license=LICENSE)
            mf.write(json.dumps(rec) + "\n"); mf.flush()
    print(f"[harvest-done] ok={nok}/{nq} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--harvest" in sys.argv:
        harvest()
    else:
        main()
