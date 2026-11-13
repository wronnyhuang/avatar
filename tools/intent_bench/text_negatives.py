"""Text-level false-positive rate of the matcher on ordinary spoken sentences that are NOT avatar commands
(e.g. your own dictated messages / chat transcripts). Input: a JSONL file of {"role": "user", "text": ...} rows or a
plain text file (one message per line). Each message is cut into sentences; each sentence is parsed as a FINAL
utterance (the strictest case: final-only rules like "okay" can fire). Prints counts per rule; with --show prints
the firing sentences (local inspection only — keep private text out of shared reports).
usage: python3 text_negatives.py <messages.jsonl|messages.txt> [--show]"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app"))
import intent  # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith("--")]
if not args:
    sys.exit(__doc__)
TH = args[0]
sents = []
for ln in open(TH, encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    try:
        j = json.loads(ln)
    except ValueError:
        j = {"role": "user", "text": ln}
    if not isinstance(j, dict) or j.get("role", "user") != "user" or not isinstance(j.get("text"), str):
        continue
    t = j["text"]
    if t.startswith("[") or "http" in t or len(t) > 2000:
        continue
    for s in re.split(r"(?<=[.!?])\s+|\n+", t):
        s = s.strip()
        if 2 <= len(s.split()) <= 60 or s.lower().strip(".!? ") in ("okay", "ok", "stop", "relax"):
            sents.append(s)
m = intent.Matcher()
fire = collections.Counter()
ex = collections.defaultdict(list)
n_fire = 0
for s in sents:
    hits = m.parse(s, final=True)
    if hits:
        n_fire += 1
        for h in hits:
            fire[h["rule"]] += 1
            ex[h["rule"]].append(s)
print(f"{len(sents)} sentences, {n_fire} fired ({n_fire / len(sents) * 100:.2f}%)")
for r, n in fire.most_common():
    print(f"  {r}: {n}")
    if "--show" in sys.argv:
        for s in ex[r][:8]:
            print("      ", s[:120])
