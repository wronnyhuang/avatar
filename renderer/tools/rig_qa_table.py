#!/usr/bin/env python3
"""Aggregate tools/rig_qa.py JSON files into one markdown table: rows = QA cases, columns = characters, cell = changed
pixels (> 12/255) and a flag: "SIDE?" (an _l/_r case moved the wrong side), "0" (no change), "n/a" (the character has
no such part, per docs/RIG.md).   python tools/rig_qa_table.py DIR > DIR/QA_TABLE.md"""
import glob, json, os, sys

ORDER = ["fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob", "muse"]   # body: + muse (not a shape: SHAPES unchanged)
NA = {  # parts a character does not have (documented in docs/RIG.md)
    "shape": ("ear", "nose_wiggle", "nostril", "tail"),
}
SHAPES = {"orb", "cube", "drop", "star", "pyramid", "blob"}


def main():
    d = sys.argv[1]
    res = {}
    for f in glob.glob(os.path.join(d, "*_qa.json")):
        r = json.load(open(f))
        res[r["character"]] = r
    chars = [c for c in ORDER if c in res]
    cases = list(res[chars[0]]["cases"]) if chars else []
    na_extra = {}
    NA_CHAR = {"bear": ("tail",), "rabbit": ("tail",), "hippo": ("tail",)}     # no tail visible / none (docs/RIG.md)
    for c in chars:
        na_extra[c] = set(res[c].get("na", [])) | set(NA_CHAR.get(c, ()))
    print("| case | " + " | ".join(chars) + " |")
    print("|---|" + "---|"*len(chars))
    fails = []
    for k in cases:
        row = []
        for c in chars:
            v = res[c]["cases"].get(k)
            if v is None:
                row.append("-")
                continue
            na = (c in SHAPES and any(k.startswith(p) for p in NA["shape"]) and not (c == "drop" and k.startswith("ear_") and "turn" not in k))
            na = na or any(k.startswith(p) for p in na_extra[c])
            if c == "drop" and k.startswith("ear_") and "turn" not in k:
                row.append(f"{v['changed_px']} (tip)")     # the drop's one tip bends with ear_l / ear_r (round 3)
                continue
            if v["verdict"] == "ok":
                row.append(str(v["changed_px"]))
            elif na:
                row.append("n/a")
            else:
                row.append(f"**{v['verdict']}** ({v['changed_px']})")
                fails.append((c, k, v["verdict"]))
        print(f"| {k} | " + " | ".join(row) + " |")
    print()
    print(f"Unexpected: {len(fails)}" + ("" if not fails else " — " + ", ".join(f"{c}:{k}:{v}" for c, k, v in fails)))


if __name__ == "__main__":
    main()
