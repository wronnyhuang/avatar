"""Summarize bench result files: python3 report.py r1.json [r2.json ...]  -> markdown tables on stdout."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app"))
import intent  # noqa: E402

ECHO = "--echo-filter" in sys.argv


def norm(t):
    return " ".join(w for w in intent.normalize(t) if w != "|")


def echo_drop(r, hits):
    """the page's echoOfAvatar(): drop hits whose ASR text is inside what the avatar said"""
    av = " " + norm(r.get("echo_text") or "") + " "
    return [h for h in hits if not (norm(h["text"]) and (" " + norm(h["text"]) + " ") in av)] if ECHO else hits


def pct(xs, q):
    return round(float(np.percentile(xs, q)) * 1000) if xs else None


def analyze(res, key):
    pos = [r for r in res if r["kind"] == "pos"]
    neg = [r for r in res if r["kind"] == "neg"]
    lat, part, ok, miss, wrong, partial_ok = [], 0, 0, 0, 0, 0
    per_var = {}
    wrong_ex = []
    for r in pos:
        hits = echo_drop(r, r["hits"][key])
        exp = set(r["expect"])
        got = [h["s"] for h in hits]
        v = per_var.setdefault(r["variant"], [0, 0, 0, 0])
        v[0] += 1
        if any(g not in exp for g in got):
            wrong += 1; v[3] += 1
            wrong_ex.append((r["text"], r["variant"], [h["text"] for h in hits if h["s"] not in exp][:1], [g for g in got if g not in exp]))
        elif exp and exp <= set(got):
            ok += 1; v[1] += 1
            first = min((h for h in hits if h["s"] in exp), key=lambda h: h["a"])
            for e in exp:
                h = min((h for h in hits if h["s"] == e), key=lambda h: h["a"])
            lat.append(first["a"] + first["c"] / 1000 - r["t_end"])
            if first["partial"]:
                partial_ok += 1
        else:
            miss += 1; v[2] += 1
    fp = [r for r in neg if echo_drop(r, r["hits"][key])]
    fpv = {}
    for r in neg:
        x = fpv.setdefault(r["variant"], [0, 0]); x[0] += 1; x[1] += bool(echo_drop(r, r["hits"][key]))
    return {"pos": len(pos), "ok": ok, "miss": miss, "wrong": wrong, "lat": lat, "partial_ok": partial_ok,
            "neg": len(neg), "fp": fp, "per_var": per_var, "fpv": fpv, "wrong_ex": wrong_ex}


def main():
    for fn in [a for a in sys.argv[1:] if not a.startswith("--")]:
        d = json.load(open(fn))
        res = d["results"]
        keys = list(res[0]["hits"].keys())
        audio = sum(r["audio_s"] for r in res)
        cpu = sum(r["cpu_ms_sum"] for r in res) / 1000
        print(f"\n### {fn.split('/')[-1]}  model={d['model'].split('/')[-1]} mode={d['mode']}  load={d['ready'][0].get('load_s')}s  "
              f"cpu/audio={cpu / audio:.3f} (all recognizers)\n")
        print("| recognizer | positives | correct | missed | WRONG cmd | latency p50 / p95 (ms, word-end -> emitted) | fired on partial | negatives | false positives |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            a = analyze(res, k)
            print(f"| {k} | {a['pos']} | {a['ok']} ({a['ok'] / max(1, a['pos']) * 100:.1f}%) | {a['miss']} | {a['wrong']} | "
                  f"{pct(a['lat'], 50)} / {pct(a['lat'], 95)} | {a['partial_ok']}/{a['ok']} | {a['neg']} | {len(a['fp'])} ({len(a['fp']) / max(1, a['neg']) * 100:.2f}%) |")
        for k in keys:
            a = analyze(res, k)
            pv = ", ".join(f"{v}: {x[1]}/{x[0]} ok, {x[3]} wrong" for v, x in a["per_var"].items())
            fv = ", ".join(f"{v}: {x[1]}/{x[0]}" for v, x in a["fpv"].items())
            print(f"\n- **{k}** per variant: {pv}; FP per variant: {fv}")
            for r in a["fp"][:12]:
                print(f"  - FP [{r['variant']}/{r['sub']}] \"{r['text']}\" -> {[(h['s'], h['text']) for h in r['hits'][k]]}")
            for t in a["wrong_ex"][:12]:
                print(f"  - WRONG [{t[1]}] \"{t[0]}\" heard {t[2]} -> {t[3]}")


if __name__ == "__main__":
    main()
