"""Streaming ASR worker for the fast intent matcher (run as a subprocess by intent_hub.py, or by the bench).

    python -I asr_worker.py --model DIR [--grammar-model DIR] [--mode full|grammar|both] [--rules intent_rules.json]

stdin  : records  [1 byte kind][uint32 LE length][payload]
         kind 'A' = int16 LE mono PCM at 24 kHz, 'R' = reset (new session/utterance stream), 'Q' = quit
stdout : one JSON object per line
         {"t":"ready", ...} once the models are loaded
         {"t":"p"|"f", "r":"full"|"grammar", "text":..., "a":<audio seconds fed so far>, "c":<cpu ms for this chunk>}
         a partial ('p') is reported for every decoded chunk (unchanged text too, so the matcher's stability
         clock advances); a final ('f') when the recognizer closes an utterance.
Vosk resamples 24 kHz internally. Models are data only (no code is loaded from them).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SR = 24000
CHUNK = 2400          # decode every 100 ms of audio at most


def grammar_words(rules_path: str) -> list[str]:
    """Every word the matcher can use (patterns, phrases, classes, preamble, modifiers, names)."""
    d = json.load(open(rules_path, encoding="utf-8"))
    words = set()

    def add(s: str) -> None:
        for w in s.replace("/", " ").split():
            w = w.lstrip("?")
            if w.startswith("$") or "=" in w or w == "=>":
                continue
            if w.isalpha():
                words.add(w)

    for src, dst in d["phrases"]:
        add(src)
    phrase_dsts = {dst for _, dst in d["phrases"]}
    for r in d["rules"]:
        for p in r["p"]:
            for w in p.split(" => ")[0].replace("/", " ").split():
                w = w.lstrip("?")
                if w and not w.startswith("$") and w not in phrase_dsts and w.isalpha():
                    words.add(w)
    for cls in d["classes"].values():
        for w in cls:
            words.add(w)
    for w in list(d["emo"]) + d["names"] + d["pre"] + list(d["mods"]) + list(d["gerund"]) + list(d["adjust"]) + d["neg"] + d["split"]:
        if w.isalpha() and w not in phrase_dsts:
            words.add(w)
    for a in d["alias"]:
        words.add(a)
    act = d.get("act") or {}                     # intent: whole-body moves + props vocabulary
    for key in ("moves", "objs", "mods"):
        for e in act.get(key, []):
            add(e[0])
    for f, _ in act.get("frames", []):
        add(f)
    for key in ("pre", "fillers", "shades", "neg", "style_stop"):
        for w in act.get(key, []):
            add(w)
    for key in ("gerund", "nums", "qty", "colors", "sizes", "ords", "nouns", "nouns_multi", "generic", "anyn", "places"):
        for w in act.get(key, {}):
            add(w)
    for tab in (act.get("cls") or {}).values():
        for w in tab:
            add(w)
    # negation spellings as the recognizer writes them
    words.update(["don't", "can't", "won't", "i'm", "that's", "let's", "it's", "you're", "i'd"])
    return sorted(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grammar-model", default="")
    ap.add_argument("--mode", default="full", choices=["full", "grammar", "both"])
    ap.add_argument("--rules", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_rules.json"))
    a = ap.parse_args()
    import vosk  # noqa: E402
    vosk.SetLogLevel(-1)
    t0 = time.monotonic()
    recs = {}
    words = grammar_words(a.rules)
    m_full = vosk.Model(a.model) if a.mode in ("full", "both") else None
    m_gram = None
    if a.mode in ("grammar", "both"):
        gm = a.grammar_model or a.model
        m_gram = m_full if (m_full is not None and gm == a.model) else vosk.Model(gm)

    def fresh() -> None:
        recs.clear()
        if m_full is not None:
            recs["full"] = vosk.KaldiRecognizer(m_full, SR)
        if m_gram is not None:
            recs["grammar"] = vosk.KaldiRecognizer(m_gram, SR, json.dumps(words + ["[unk]"]))
        for r in recs.values():
            r.SetWords(False)

    fresh()
    out = sys.stdout
    out.write(json.dumps({"t": "ready", "mode": a.mode, "load_s": round(time.monotonic() - t0, 2),
                          "grammar_words": len(words)}) + "\n")
    out.flush()
    inp = sys.stdin.buffer
    fed = 0
    pend = bytearray()
    last = {}

    def decode(buf: bytes) -> None:
        nonlocal fed
        fed += len(buf) // 2
        lines = []
        for name, r in recs.items():
            c0 = time.process_time()
            done = r.AcceptWaveform(bytes(buf))
            if done:
                txt = json.loads(r.Result()).get("text", "")
                kind = "f"
            else:
                txt = json.loads(r.PartialResult()).get("partial", "")
                kind = "p"
            txt = txt.replace("[unk]", "").strip()
            c = (time.process_time() - c0) * 1000
            if kind == "f" or txt or last.get(name):
                lines.append(json.dumps({"t": kind, "r": name, "text": txt, "a": round(fed / SR, 3), "c": round(c, 1)}))
            last[name] = txt if kind == "p" else ""
        if lines:
            out.write("\n".join(lines) + "\n")
            out.flush()

    while True:
        hdr = inp.read(5)
        if len(hdr) < 5:
            break
        kind, n = hdr[:1], struct.unpack("<I", hdr[1:])[0]
        data = inp.read(n) if n else b""
        if kind == b"Q":
            break
        if kind == b"R":
            pend.clear()
            fed = 0
            last.clear()
            fresh()
            continue
        if kind == b"E":           # end of stream: flush finals
            if pend:
                decode(bytes(pend))
                pend.clear()
            for name, r in recs.items():
                txt = json.loads(r.FinalResult()).get("text", "").replace("[unk]", "").strip()
                out.write(json.dumps({"t": "f", "r": name, "text": txt, "a": round(fed / SR, 3), "c": 0, "eos": True}) + "\n")
            out.flush()
            fresh()
            continue
        if kind == b"A":
            pend.extend(data)
            # decode in <=100 ms slices; a batch from the page (~140 ms) becomes 1-2 decodes
            while len(pend) >= 2 * CHUNK:
                decode(bytes(pend[:2 * CHUNK]))
                del pend[:2 * CHUNK]
            if len(pend) >= 2 * 480 * 2:   # >= 40 ms left over and nothing else queued: decode it now
                decode(bytes(pend))
                pend.clear()
    out.flush()


if __name__ == "__main__":
    main()
