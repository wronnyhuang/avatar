"""Run tests/intent_cases.txt through app/intent.py; print mismatches. --write regenerates tests/intent_vectors.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import intent  # noqa: E402


def _v(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def summ_act(c):
    """'move/spin count=3 dir=left' | 'object/pick_up obj.color=red obj.name=ball side=left'"""
    a = c["action"]
    head = a + "/" + (c.get("name") if a == "move" else c.get("verb"))
    kv = []
    for k in sorted(c):
        if k in ("action", "name", "verb"):
            continue
        v = c[k]
        if isinstance(v, dict):
            for k2 in sorted(v):
                kv.append(f"{'obj' if k == 'object' else k}.{k2}={_v(v[k2])}")
        else:
            kv.append(f"{k}={_v(v)}")
    return " ".join([head] + kv)


def summ(c):
    a = c["action"]
    if a in ("move", "object"):
        return summ_act(c)
    if a == "release":
        p = c["parts"]
        return "release/" + ("all" if p == "all" else "+".join(sorted(p)))
    if a == "hold":
        return "hold/all"
    if a == "look":
        return "look/" + c["target"]
    s = a + "/" + c.get("name", "")
    if c.get("side") and a in ("pose", "gesture"):
        s += "/" + c["side"]
    return s


def cases():
    out = []
    for ln in open(os.path.join(HERE, "intent_cases.txt"), encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.startswith("#"):
            continue
        ph, _, exp = ln.partition("|")
        exp = exp.split("   # ")[0]
        sep = ";" if ";" in exp or "=" in exp else ","
        out.append((ph.strip(), [e.strip() for e in exp.split(sep) if e.strip()]))
    return out


def main():
    bad = 0
    vec = []
    cs = cases()
    for ph, exp in cs:
        m = intent.Matcher()
        cmds = [h["cmd"] for h in m.parse(ph)]
        got = [summ(c) for c in cmds]
        if got != exp:
            bad += 1
            print(f"MISMATCH {ph!r}: want {exp} got {got}")
        vec.append({"text": ph, "expect": exp, "cmds": cmds})
    npos = sum(1 for _, e in cs if e)
    print(f"{len(cs)} cases ({npos} positive, {len(cs) - npos} negative), {bad} mismatches")
    if "--write" in sys.argv:
        with open(os.path.join(HERE, "intent_vectors.json"), "w") as f:
            json.dump({"cases": vec}, f, indent=0, sort_keys=True)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
