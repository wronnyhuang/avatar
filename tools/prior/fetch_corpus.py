"""Build the real-people motion corpus: search YouTube per category, download clip WINDOWS (<=480p).

Media is untrusted data and lives in scratch only: <scratch>/media/<category>/<id>_<start>.mp4.
Writes <scratch>/manifest.jsonl (one line per clip: id, url, title, channel, category, query, start, dur,
license note). Usage: python -I fetch_corpus.py <scratch> [--per-query N] [--win S] [--workers K]
"""
import json, subprocess, sys, os, argparse, concurrent.futures as cf, random, time
from pathlib import Path

QUERIES = {
    "speak": [
        "vlog talking to camera storytime sitting",
        "life update talking to the camera",
        "webcam explainer talking head tutorial",
        "advice video talking directly to camera",
        "facecam commentary talking head no b-roll",
        "teacher explains concept webcam lecture face",
        "honest chat sit down video talking",
        "rant video talking to camera",
    ],
    "listen": [
        "split screen podcast remote interview full episode",
        "riverside podcast remote guest split screen",
        "zoom interview recording two people side by side",
        "video call interview split screen conversation",
        "remote podcast episode webcam both faces",
        "zoom conversation full episode gallery view",
        "skype interview recorded side by side",
        "reacting to podcast clip facecam reaction",
    ],
    "idle": [
        "study with me real time webcam face no music",
        "silent study with me face cam",
        "watching with face cam reaction no talking",
        "zoom meeting listening silent webcam",
    ],
}
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))


def search(q, n):
    cmd = [YTDLP, "--flat-playlist", "--no-warnings", "-J", f"ytsearch{n}:{q}"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        js = json.loads(out.stdout)
    except Exception:
        return []
    return [e for e in js.get("entries", []) if e]


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
    ap.add_argument("--per-query", type=int, default=8)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cats", default="speak,listen,idle")
    ap.add_argument("--max-per-cat", type=int, default=40)
    a = ap.parse_args()
    root = Path(a.scratch)
    man = root / "manifest.jsonl"
    seen = set()
    if man.exists():
        for l in man.read_text().splitlines():
            try: seen.add(json.loads(l)["id"])
            except Exception: pass
    jobs = []
    rng = random.Random(7)
    for cat in a.cats.split(","):
        cands = []
        for q in QUERIES[cat]:
            for e in search(q, a.per_query * 2):
                d = e.get("duration") or 0
                if not (240 <= d <= 3 * 3600) or e["id"] in seen:
                    continue
                if e.get("live_status") in ("is_live", "is_upcoming"):
                    continue
                seen.add(e["id"])
                cands.append((q, e))
            print(f"[search] {cat}: {q!r} -> {len(cands)} so far", flush=True)
        rng.shuffle(cands)
        for q, e in cands[: a.max_per_cat]:
            d = int(e["duration"])
            win = min(a.win, d - 60)
            starts = [int(max(30, 0.3 * d))]
            if cat == "listen" and d > 1200:      # long dyads: a second window
                starts.append(int(0.65 * d))
            for s in starts:
                jobs.append(dict(id=e["id"], start=s, win=win, cat=cat, query=q, title=e.get("title"),
                                 channel=e.get("channel") or e.get("uploader"), vdur=d,
                                 path=str(root / "media" / cat / f"{e['id']}_{s}.mp4")))
    print(f"[fetch] {len(jobs)} clips", flush=True)
    t0 = time.time(); nok = 0
    (root / "jobs.json").write_text(json.dumps(jobs))
    with cf.ThreadPoolExecutor(a.workers) as ex, open(man, "a") as mf:
        for fut in cf.as_completed([ex.submit(fetch, j) for j in jobs]):
            job, ok, msg = fut.result()
            if ok:
                nok += 1
                rec = dict(id=job["id"], url=f"https://www.youtube.com/watch?v={job['id']}", title=job["title"],
                           channel=job["channel"], category=job["cat"], query=job["query"], start=job["start"],
                           dur=job["win"], video_dur=job["vdur"], file=job["path"],
                           license="YouTube Standard License (not redistributed; research features only)")
                mf.write(json.dumps(rec) + "\n"); mf.flush()
            else:
                print(f"[fail] {job['id']} {msg}", flush=True)
            if nok % 10 == 0:
                print(f"[fetch] ok={nok} t={time.time()-t0:.0f}s", flush=True)
    print(f"[done] ok={nok}/{len(jobs)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
