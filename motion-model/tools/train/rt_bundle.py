"""a runtime variant of a bundle without touching the source: copy <src bundle dir> to <dst>/bundle and merge
the json into config["runtime"] (dict keys replace; "events" is merged key by key).
    python -I tools/train/rt_bundle.py <src bundle dir> <dst run dir> '<runtime json>'
"""
import json
import os
import shutil
import sys


def main():
    src, dst, js = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    b = os.path.join(dst, "bundle")
    if os.path.exists(b):
        shutil.rmtree(b)
    shutil.copytree(src, b)
    cp = os.path.join(b, "config.json")
    cfg = json.load(open(cp))
    rt = dict(cfg.get("runtime") or {})
    for k, v in js.items():
        if k == "events" and isinstance(v, dict) and isinstance(rt.get("events"), dict):
            rt["events"] = dict(rt["events"], **v)
        else:
            rt[k] = v
    cfg["runtime"] = rt
    cfg.setdefault("runtime_provenance", []).append({"by": "rt_bundle.py", "src": os.path.abspath(src), "merge": js})
    with open(cp, "w") as f:
        json.dump(cfg, f, indent=1)
    print(b, json.dumps(rt))


if __name__ == "__main__":
    main()
