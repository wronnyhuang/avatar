"""intent c2: score the muse fast matcher on the fresh gpt-4.1-mini phrasing set (tests/intent_c2_{dev,holdout}.json).

    python3 tests/intent_c2_eval.py tests/intent_c2_holdout.json [--show 40] [--cat backflip] [--json out.json]

The matcher is the hub's muse matcher (act grammar + renderer/animals/muse.props.json). Per category:
  P        = matched commands / commands fired           R = matched label steps / label steps
  accept   = lines whose every label step fired in order (head right)
  exact    = accepted lines with every labelled argument right (obj/target/place resolved to scene ids)
  wrong    = positive lines that fired a command the label does not allow and were not accepted (a mis-map: worse
             than a miss)   none = positive lines with no command at all
Negatives (empty label): fired = any command; fired_nr = any command other than release.
"""
import argparse, json, os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
_app = sys.argv[sys.argv.index("--app") + 1] if "--app" in sys.argv else os.path.join(HERE, "..", "app")   # --app DIR: score another tree's matcher
sys.path.insert(0, _app)
import intent  # noqa: E402

SCENE_FN = os.path.join(HERE, "..", "renderer", "animals", "muse.props.json")


def scene():
    return json.load(open(SCENE_FN))


def alias_map(sc):
    m, col = {}, {}
    for p in sc.get("props", []):
        for w in [p.get("id"), p.get("name")] + list(p.get("aliases") or []):
            if w:
                m[" ".join(intent.normalize(w.replace("_", " "))).replace("|", "").strip()] = p["id"]
        for c in [p.get("color")] + list(p.get("colors") or []):
            if c:
                col[c] = p["id"]
    for k in list(m):
        ws = k.split(" ")
        if len(ws) > 1:
            m.setdefault(ws[-1], m[k])
    return m, col


def parse_label(s):
    steps = []
    for part in [x.strip() for x in s.split(";") if x.strip()]:
        toks = part.split()
        heads = set(toks[0].split("|"))
        args = {}
        for kv in toks[1:]:
            k, _, v = kv.partition("=")
            args[k] = set(v.split("|"))
        steps.append((heads, args))
    return steps


def head(c):
    a = c.get("action")
    if a == "move":
        return "move/" + str(c.get("name"))
    if a == "object":
        return "object/" + str(c.get("verb"))
    return a + "/" + str(c.get("name") or c.get("parts") or c.get("target") or "")


def obj_ids(o, am, col):
    if not isinstance(o, dict):
        return set()
    out = set()
    if o.get("name"):
        n = str(o["name"]).replace("_", " ")
        if n in am:
            out.add(am[n])
        elif n.split(" ")[-1] in am:
            out.add(am[n.split(" ")[-1]])
        else:
            out.add(n)
    elif o.get("color") and o["color"] in col:
        out.add(col[o["color"]])
    elif o.get("category") == "toy":
        out.add("ball")
    if o.get("ref") in ("it", "that", "all", "any", "other") and not o.get("name"):
        out.add("it")
    return out


def norm_place(p):
    p = str(p or "")
    return {"desk_top": "desk", "desktop": "desk", "table": "desk", "ground": "floor", "rug": "floor",
            "chair_seat": "chair"}.get(p, p)


def args_ok(c, args, am, col):
    for k, vs in args.items():
        if c.get("action") == "move" and k in ("obj", "place", "side"):
            continue                    # a label alternative without an object ("sit in it" | "sit down")
        if k == "obj":
            if not (obj_ids(c.get("object"), am, col) & vs):
                return False
        elif k == "target":
            t = c.get("target")
            got = {t} if isinstance(t, str) else obj_ids(t, am, col) | ({norm_place(t.get("name"))} if isinstance(t, dict) else set())
            if not (got & vs):
                return False
        elif k == "place":
            if norm_place(c.get("place")) not in vs:
                return False
        elif str(c.get(k)) not in vs:
            return False
    return True


def score(items, m_factory, show=0, cat=None):
    sc = scene()
    am, col = alias_map(sc)
    per = defaultdict(lambda: defaultdict(int))
    bad, t_us = [], []
    for it in items:
        if cat and it["cat"] != cat:
            continue
        m = m_factory()
        t0 = time.perf_counter()
        cmds = [h["cmd"] for h in m.parse(it["text"], final=True)]
        t_us.append((time.perf_counter() - t0) * 1e6)
        L = parse_label(it["label"])
        S = per[it["cat"]]
        S["lines"] += 1
        if not L:
            S["neg"] += 1
            S["fired"] += bool(cmds)
            S["fired_nr"] += any(c.get("action") != "release" for c in cmds)
            if cmds:
                bad.append((it, cmds, "FIRED"))
            continue
        S["pos"] += 1
        S["steps"] += len(L)
        S["got"] += len(cmds)
        j, matched, exact = 0, 0, True
        used = set()
        for heads, args in L:
            k = j
            while k < len(cmds) and head(cmds[k]) not in heads:
                k += 1
            if k < len(cmds):
                matched += 1
                used.add(k)
                if not args_ok(cmds[k], args, am, col):
                    exact = False
                j = k + 1
            else:
                exact = False
        S["matched"] += matched
        acc = matched == len(L)
        S["accept"] += acc
        S["exact"] += acc and exact
        allowed = set().union(*[h for h, _ in L])
        wrong = [c for i, c in enumerate(cmds) if i not in used and head(c) not in allowed]
        if not cmds:
            S["none"] += 1
        elif wrong and not acc:
            S["wrong"] += 1
        if not (acc and exact) or wrong:
            bad.append((it, cmds, "WRONG" if wrong and not acc else ("NONE" if not cmds else ("ARGS" if acc else "MISS"))))
    tot = defaultdict(int)
    for S in per.values():
        for k, v in S.items():
            tot[k] += v
    rows = {}
    for k in list(per) + ["ALL"]:
        S = tot if k == "ALL" else per[k]
        r = {"lines": S["lines"]}
        if S["pos"]:
            r.update(P=round(S["matched"] / max(1, S["got"]), 3), R=round(S["matched"] / S["steps"], 3),
                     accept=f'{S["accept"]}/{S["pos"]}', exact=f'{S["exact"]}/{S["pos"]}', wrong=S["wrong"], none=S["none"])
        if S["neg"]:
            r.update(neg=S["neg"], fired=S["fired"], fired_nr=S["fired_nr"])
        rows[k] = r
    t_us.sort()
    lat = {"median_us": round(t_us[len(t_us) // 2], 1), "p95_us": round(t_us[int(len(t_us) * 0.95)], 1), "max_us": round(t_us[-1], 1)} if t_us else {}
    return rows, bad, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--cat")
    ap.add_argument("--json")
    ap.add_argument("--app")
    a = ap.parse_args()
    items = json.load(open(a.file))["items"]
    sc = scene()
    base = intent.Matcher(scene=sc)
    def fac():
        m = intent.Matcher(scene=sc)
        m.A = base.A
        return m
    rows, bad, lat = score(items, fac, cat=a.cat)
    for k, r in rows.items():
        print(f"{k:12s} " + " ".join(f"{kk}={vv}" for kk, vv in r.items()))
    print("latency", lat)
    for it, cmds, why in bad[:a.show]:
        print(f"  [{why}] {it['cat']}: {it['text']!r}\n      want {it['label']!r}\n      got  {[intent_summ(c) for c in cmds]}")
    if a.json:
        json.dump({"rows": rows, "latency": lat, "bad": [{"text": i["text"], "label": i["label"], "why": w,
                   "got": [intent_summ(c) for c in c_]} for i, c_, w in bad]}, open(a.json, "w"), indent=1)


def intent_summ(c):
    from check_cases import summ
    return summ(c)


sys.path.insert(0, HERE)
if __name__ == "__main__":
    main()
