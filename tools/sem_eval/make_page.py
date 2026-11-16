#!/usr/bin/env python3
"""The side-by-side page (current vs semantic API planner), ONE self-contained HTML file, dark, phone-first,
no network (no CDN / fonts / remote scripts). Colour-blind safe: the two planners are told apart by labels + shape
(square = current, circle = semantic) and blue / orange, verdicts by words + symbols, never red vs green.

  python3 tools/sem_eval/make_page.py --runs EVAL/runs/<tag> --clips EVAL/clips/<tag> --judge EVAL/judge/<tag>
        --results EVAL/RESULTS.json --out runs/sem_eval/page [--notes notes.md] [--max-mb 25]
        [--ids id,id (default: the commands marked "page")] [--title ...]

Videos are inlined as base64 while the total stays under --max-mb; otherwise they are copied next to index.html
(clips/<char>_<mode>_<id>.mp4, relative links) and the page says so.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil

MODES = [("current", "Current planner", "&#9632;", "c"), ("semantic", "Semantic API", "&#9679;", "s")]
CHARS = ["muse", "fox"]
VERD = {"yes": "&#10003; did it", "partly": "~ partly", "no": "&#10007; did not"}
E = html.escape


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []


def fmt_ms(v):
    return "–" if v is None else (f"{v / 1000:.1f} s" if v >= 1000 else f"{v} ms")


def poses_compact(R, max_lines=14):
    """the current planner's poses: one line per pose, t + the channels it sets"""
    lines = []
    for k in R.get("kf") or []:
        if k.get("engine"):
            lines.append("library: " + json.dumps(k["engine"])[:300])
        for p in k.get("poses") or []:
            if len(lines) >= max_lines:
                break
            kv = " ".join(f"{n}={v:g}" if isinstance(v, (int, float)) else f"{n}={v}" for n, v in p.items() if n != "t")
            lines.append(f"t={p.get('t')}: {kv}")
    n = sum(len(k.get("poses") or []) for k in R.get("kf") or [])
    if n > max_lines:
        lines.append(f"... {n - max_lines} more poses")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmds", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                    "tests", "sem_eval", "commands.jsonl"))
    ap.add_argument("--notes", default=None)
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-mb", type=float, default=25)
    ap.add_argument("--title", default="Avatar: current planner vs semantic API")
    a = ap.parse_args()
    cmds = [json.loads(l) for l in open(a.cmds) if l.strip()]
    ids = [x for x in a.ids.split(",") if x] or [c["id"] for c in cmds if c.get("page")]
    C = {c["id"]: c for c in cmds}
    act = {(r["char"], r["mode"], r["id"]): r for r in load_jsonl(a.judge and os.path.join(a.judge, "action.jsonl")) if "verdict" in r}
    pair = {(r["char"], r["id"]): r for r in load_jsonl(a.judge and os.path.join(a.judge, "pairwise.jsonl")) if "better" in r}
    res = json.load(open(a.results)) if a.results and os.path.exists(a.results) else None
    os.makedirs(a.out, exist_ok=True)

    def run(ch, m, cid, ps="fresh"):
        p = os.path.join(a.runs, ch, m, ps, cid + ".json")
        return json.load(open(p)) if os.path.exists(p) else None

    clips = {}
    total = 0
    for cid in ids:
        for ch in CHARS:
            for m, *_ in MODES:
                p = os.path.join(a.clips, ch, m, cid + ".mp4")
                if os.path.exists(p):
                    clips[(ch, m, cid)] = p
                    total += os.path.getsize(p)
    inline = total * 4 / 3 < a.max_mb * 1e6
    if not inline:
        os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)

    def src(k):
        p = clips[k]
        if inline:
            return "data:video/mp4;base64," + base64.b64encode(open(p, "rb").read()).decode()
        name = f"{k[0]}_{k[1]}_{k[2]}.mp4"
        shutil.copy(p, os.path.join(a.out, "clips", name))
        return "clips/" + name

    H = []
    w = H.append
    w("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>")
    w(f"<title>{E(a.title)}</title><style>")
    w("""
:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--card:#1b1f24;--line:#2c323a;--c:#4ea1ff;--s:#ff9f40}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:auto;padding:12px}h1{font-size:20px;margin:6px 0}h2{font-size:17px;margin:18px 0 6px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.tw{overflow-x:auto}.sum td,.sum th{padding:4px 4px}.sm{font-size:11px;color:var(--mut);line-height:1.25}.pick{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.pick button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:18px;padding:6px 14px;font-size:15px}
.pick button.on{border-color:var(--fg);font-weight:600}
.cmd{background:var(--card);border-radius:10px;padding:10px;margin:12px 0}.cmd h3{margin:0 0 6px;font-size:16px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:700px){.pair{grid-template-columns:1fr}}
.side{border-top:3px solid var(--line);padding-top:6px}.side.c{border-color:var(--c)}.side.s{border-color:var(--s)}
.lab{font-weight:600}.c .lab{color:var(--c)}.s .lab{color:var(--s)}
video{width:100%;border-radius:6px;background:#000;display:block}
pre{white-space:pre-wrap;word-break:break-word;background:#0b0d10;border:1px solid var(--line);border-radius:6px;padding:6px;font-size:12px;max-height:260px;overflow:auto;margin:6px 0}
.v{font-size:13px;margin:4px 0}.k{color:var(--mut)}.ch{display:none}body[data-ch=muse] .ch.muse,body[data-ch=fox] .ch.fox,body[data-ch=juno] .ch.juno{display:block}
.pw{font-size:13px;margin-top:6px;padding:6px;border:1px dashed var(--line);border-radius:6px}details summary{cursor:pointer;color:var(--mut);font-size:13px}
""")
    w("</style></head><body data-ch=muse><main>")
    w(f"<h1>{E(a.title)}</h1>")
    w("<p class=mut>Same spoken command, same character, same live pipeline (page &rarr; hub planner &rarr; motion layer, with gaze, "
      "idle and lip-sync layers running). <span style='color:var(--c)'>&#9632; Current planner</span> = today's demo. "
      "<span style='color:var(--s)'>&#9679; Semantic API</span> = the new switch (<code>?planner=semantic</code>). "
      "Clips are offline renders of the recorded rig output through the real renderer (no sound). Verdicts are a blind Claude judge "
      "that never knew which planner made a clip.</p>")
    if not inline:
        w(f"<p class=mut>The clips are {total / 1e6:.0f} MB, too big to inline: they sit in the clips/ folder next to this page.</p>")
    # ---- summary: PLANNED commands only (library hits are the same engine move in both modes)
    import glob as _g
    allr = {}
    for p in _g.glob(os.path.join(a.runs, "*", "*", "*", "*.json")):
        if p.endswith("run.json"):
            continue
        try:
            R = json.load(open(p))
        except ValueError:
            continue
        if "timings" in R:
            allr[(R["char"], R["mode"], R["pass"], R["id"])] = R
    srcof = lambda ch, m, cid: ((allr.get((ch, m, "fresh", cid)) or {}).get("timings") or {}).get("source")
    planned = lambda ch, cid: srcof(ch, "current", cid) not in (None, "library") and srcof(ch, "semantic", cid) not in (None, "library")

    def q(xs, pp):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        k = (len(xs) - 1) * pp
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))
    if allr:
        w("<h2>Summary</h2><p class=mut>Commands the library already knows (wave, wink, backflip...) play the same move in both modes; "
          "the table counts only the commands a planner had to write. Did it = blind judge, yes + &frac12; partly.</p>"
          "<div class=tw><table class=sum><tr><th>planner</th><th></th><th>did it</th><th>natural</th>"
          "<th>starts moving: new / p90</th><th>repeat</th></tr>")
        for m, lbl, sym, cls in MODES:
            for ch in CHARS + ["all"]:
                cells = sorted({cid for (c, mm, ps, cid) in allr if (ch == "all" or c == ch) and planned(c, cid) and (c, m, cid) in act})
                keys = [(c, cid) for (c, mm, ps, cid) in allr if mm == m and ps == "fresh" and (ch == "all" or c == ch) and planned(c, cid) and (c, m, cid) in act]
                v = [act[(c, m, cid)] for c, cid in keys]
                if not v:
                    continue
                n = len(v)
                y = sum(x["verdict"] == "yes" for x in v) / n; pa = sum(x["verdict"] == "partly" for x in v) / n; no = sum(x["verdict"] == "no" for x in v) / n
                nat = sum(x.get("natural") or 0 for x in v) / n
                fm = [allr[(c, m, "fresh", cid)]["timings"].get("first_motion_ms") for (c, mm, ps, cid) in allr if mm == m and ps == "fresh" and (ch == "all" or c == ch) and planned(c, cid)]
                cm = [allr[(c, m, "cached", cid)]["timings"].get("first_motion_ms") for (c, mm, ps, cid) in allr if mm == m and ps == "cached" and (ch == "all" or c == ch)
                      and ((allr[(c, m, "cached", cid)].get("timings") or {}).get("source") == "cache")]
                short = {"current": "current", "semantic": "semantic"}[m]
                w(f"<tr><td class={cls}><span class=lab>{sym} {short}</span></td><td>{ch}</td><td><b>{y + pa / 2:.0%}</b>"
                  f"<div class=sm>{y:.0%} yes &middot; {pa:.0%} partly &middot; {no:.0%} no &middot; n {n}</div></td><td>{nat:.1f}</td>"
                  f"<td>{fmt_ms(q(fm, .5))}<div class=sm>p90 {fmt_ms(q(fm, .9))}</div></td><td>{fmt_ms(q(cm, .5))}</td></tr>")
        w("</table></div>")
        pl = [p for (c, cid), p in pair.items() if planned(c, cid)]
        if pl:
            def share(f, m):
                return sum(1 if p.get(f + "_mode") == m else .5 if not p.get(f + "_mode") else 0 for p in pl) / len(pl)
            w(f"<p>Blind side-by-side on the same command and character (n {len(pl)}, ties count half): did it better &mdash; "
              f"current {share('better', 'current'):.0%}, semantic {share('better', 'semantic'):.0%}; looked more natural &mdash; "
              f"current {share('more_natural', 'current'):.0%}, semantic {share('more_natural', 'semantic'):.0%}.</p>")
        w("<p class=mut>&ldquo;Move starts&rdquo; = from the command reaching the page to the planned move starting on the avatar "
          "(includes the short wait for the voice's cue; the thinking gesture fills the gap before it). Measured on a busy shared "
          "machine, both planners interleaved command by command.</p>")
    if a.notes and os.path.exists(a.notes):
        w("<h2>What still looks robotic</h2>")
        for para in open(a.notes).read().strip().split("\n\n"):
            w(f"<p>{E(para.strip())}</p>")
    # ---- picker
    w("<div class=pick>" + "".join(f"<button data-c={c} class='{'on' if c == 'muse' else ''}'>{c.capitalize()}</button>" for c in CHARS) + "</div>")
    for cid in ids:
        c = C[cid]
        w(f"<section class=cmd><h3>&ldquo;{E(c['utterance'])}&rdquo;</h3><div class=mut style='font-size:13px'>{E(c['category'])}</div>")
        for ch in CHARS:
            w(f"<div class='ch {ch}'><div class=pair>")
            for m, lbl, sym, cls in MODES:
                R = run(ch, m, cid)
                Rc = run(ch, m, cid, "cached")
                w(f"<div class='side {cls}'><div class=lab>{sym} {lbl}</div>")
                if (ch, m, cid) in clips:
                    w(f"<video muted loop playsinline controls preload=none data-src='{src((ch, m, cid))}'></video>")
                else:
                    w("<p class=mut>not run</p>")
                if R:
                    T = R.get("timings") or {}
                    Tc = (Rc or {}).get("timings") or {}
                    w(f"<div class=v><span class=k>source</span> {E(str(T.get('source')))} &middot; <span class=k>move starts</span> "
                      f"{fmt_ms(T.get('first_motion_ms'))} &middot; <span class=k>hub first pose</span> {fmt_ms(T.get('hub_first_pose_ms'))}"
                      f" &middot; <span class=k>repeat</span> {fmt_ms(Tc.get('first_motion_ms'))} ({E(str(Tc.get('source')))})</div>")
                    if R.get("errors"):
                        w(f"<div class=v>&#9888; {E('; '.join(R['errors'])[:300])}</div>")
                    j = act.get((ch, m, cid))
                    if j:
                        w(f"<div class=v><b>{VERD.get(j['verdict'], E(j['verdict']))}</b> &middot; natural {j.get('natural')}/5 &mdash; {E(j.get('reason', ''))}"
                          + (f"<br><span class=k>robotic:</span> {E(j['robotic'])}" if j.get("robotic") else "") + "</div>")
                    if m == "semantic":
                        prog = R.get("program") or ((R.get("semLast") or {}).get("program"))
                        w("<details open><summary>generated program</summary><pre>" + E(prog or "(no program: " + str(T.get("source")) + ")") + "</pre></details>")
                    else:
                        out = (("summary: " + R["summary"] + "\n") if R.get("summary") else "") + poses_compact(R)
                        w("<details><summary>planner output (summary + poses)</summary><pre>" + E(out or "(none)") + "</pre></details>")
                w("</div>")
            w("</div>")
            p = pair.get((ch, cid))
            if p:
                nm = {m: (sym + " " + lbl) for m, lbl, sym, _ in MODES}
                b = nm.get(p.get("better_mode"), "tie")
                n = nm.get(p.get("more_natural_mode"), "tie")
                w(f"<div class=pw>Blind pick &mdash; did it better: <b>{b}</b>; more natural: <b>{n}</b>. {E(p.get('reason', ''))}</div>")
            w("</div>")
        w("</section>")
    w("""<script>
const B=document.body;document.querySelectorAll('.pick button').forEach(b=>b.onclick=()=>{B.dataset.ch=b.dataset.c;
document.querySelectorAll('.pick button').forEach(x=>x.classList.toggle('on',x===b));scan();});
const io=new IntersectionObserver(es=>es.forEach(e=>{const v=e.target;if(e.isIntersecting){if(!v.src)v.src=v.dataset.src;v.play().catch(()=>{});}else v.pause();}),{threshold:0.25});
function scan(){document.querySelectorAll('video').forEach(v=>{io.unobserve(v);if(v.offsetParent)io.observe(v);else v.pause();});}
scan();
</script>""")
    w("</main></body></html>")
    out = os.path.join(a.out, "index.html")
    open(out, "w").write("\n".join(H))
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, {len(clips)} clips {'inlined' if inline else 'as sibling files'}, {len(ids)} commands")


if __name__ == "__main__":
    main()
