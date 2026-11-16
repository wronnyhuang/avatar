#!/usr/bin/env python3
"""Blind judge: Claude through the `claude` CLI (`claude -p --model sonnet --tools Read --permission-mode dontAsk
--output-format json`), contact sheets from render_clips.py, never told which planner made a clip. NEVER OpenAI.

  python3 tools/sem_eval/judge.py --clips EVAL/clips/<tag> --out EVAL/judge/<tag> [--cmds tests/sem_eval/commands.jsonl]
        [--modes current,semantic] [--chars muse,fox] [--only id,id] [--what action,pairwise] [--conc 4] [--seed 7]

action   per clip: did the avatar do <command>? yes / partly / no + one line why; naturalness 1-5
pairwise per (char, command) with both modes: which did it better / which looks more natural (A / B / tie), sides random
Sheets are copied to <cache>/sheets/<sha1>.jpg (the judge sees only hashed names); every call is cached on disk by
(model, schema, prompt, image bytes) in <cache>/calls/ (default EVAL/judge_cache) so reruns are free.
Output: <out>/action.jsonl, <out>/pairwise.jsonl (rows carry the hidden identity for aggregate.py).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HROOT = os.path.dirname(os.path.dirname(HERE))
CLAUDE = os.environ.get("JUDGE_CLAUDE", "claude")
MODEL = os.environ.get("JUDGE_MODEL", "sonnet")
DEF_CACHE = os.path.join(HROOT, "runs", "sem_eval", "judge_cache")

SYSTEM = ("You are a careful, blind evaluator of a stylised 3D avatar's body and face motion. You only see contact sheets "
          "(frames sampled from short rendered clips). Use the Read tool to open every image path you are given before "
          "answering. Answer only with the requested JSON.")
CONV = ("How to read a sheet: 8 frames of ONE clip, time runs left to right then top to bottom (caption t = seconds after "
        "the command was given; the first frame is the moment of the command). The character faces the camera, so its "
        "OWN left hand appears on SCREEN RIGHT. 'Left'/'right' in a command always mean the character's own left/right. "
        "Some characters are shown torso-up only (a fox): whole-body moves can only show as the upper body moving, "
        "turning or bobbing; judge whether that is a reasonable rendition. 'me' / 'the camera' / 'the viewer' = the camera.")
SCENE = {
    "fox": "This character (a fox, torso-up) stands in a forest: there is no board, TV or desk in its scene. For a command "
           "naming such a thing, judge whether it points / looks in some plausible direction as if the thing were off-screen.",
    "_room": "The character stands in a room: a chalkboard on the wall behind it on SCREEN LEFT (the character's own right), "
             "a TV screen on the wall on SCREEN RIGHT (its own left), a desk, a chair and toys.",
}
ACTION_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
    "reason": {"type": "string"},
    "natural": {"type": "integer", "minimum": 1, "maximum": 5},
    "robotic": {"type": "string"}}, "required": ["verdict", "reason", "natural", "robotic"]}
PAIR_SCHEMA = {"type": "object", "properties": {
    "better": {"type": "string", "enum": ["A", "B", "tie"]},
    "more_natural": {"type": "string", "enum": ["A", "B", "tie"]},
    "reason": {"type": "string"}}, "required": ["better", "more_natural", "reason"]}


def log(msg):
    print(f"[judge {time.strftime('%T')}] {msg}", file=sys.stderr, flush=True)


def hashed(sheet, cache):
    d = os.path.join(cache, "sheets")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(open(sheet, "rb").read()).hexdigest()[:16]
    p = os.path.join(d, h + ".jpg")
    if not os.path.exists(p):
        shutil.copy(sheet, p)
    return p


def call_claude(prompt, schema, images, tag, cache):
    key = hashlib.sha1((MODEL + json.dumps(schema) + SYSTEM + prompt).encode() + b"".join(open(i, "rb").read() for i in images)).hexdigest()
    cp = os.path.join(cache, "calls", key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp)), True
    cmd = [CLAUDE, "-p", "--model", MODEL, "--tools", "Read", "--permission-mode", "dontAsk", "--no-session-persistence",
           "--setting-sources", "", "--system-prompt", SYSTEM, "--output-format", "json", "--json-schema", json.dumps(schema)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI")}
    last = None
    for attempt in (1, 2):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=os.path.join(cache, "sheets"), timeout=600, env=env)
        except subprocess.TimeoutExpired as e:
            last = f"timeout {e}"
            continue
        dt = time.time() - t0
        try:
            d = json.loads(r.stdout)
            out = d.get("structured_output")
            if out is None:
                out = json.loads(re.search(r"\{.*\}", d.get("result") or "", re.S).group(0))
            out["_s"] = round(dt, 1)
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            json.dump(out, open(cp, "w"))
            log(f"ok {tag} {dt:.0f}s")
            return out, False
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}; rc={r.returncode}; err={r.stderr[-200:]!r}"
            log(f"FAILED {tag} attempt {attempt}: {last}")
    return {"error": last}, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmds", default=os.path.join(HROOT, "tests", "sem_eval", "commands.jsonl"))
    ap.add_argument("--modes", default="current,semantic")
    ap.add_argument("--chars", default="muse,fox")
    ap.add_argument("--only", default="")
    ap.add_argument("--what", default="action,pairwise")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default=DEF_CACHE)
    a = ap.parse_args()
    a.conc = min(a.conc, 4)
    os.makedirs(a.out, exist_ok=True)
    cmds = {c["id"]: c for c in (json.loads(l) for l in open(a.cmds) if l.strip())}
    only = set(filter(None, a.only.split(",")))
    modes, chars = a.modes.split(","), a.chars.split(",")
    sheets = {}
    for ch in chars:
        for m in modes:
            for cid in cmds:
                if only and cid not in only:
                    continue
                p = os.path.join(a.clips, ch, m, cid + ".sheet.jpg")
                if os.path.exists(p):
                    sheets[(ch, m, cid)] = p
    what = a.what.split(",")
    t0 = time.time()

    def act(k):
        ch, m, cid = k
        c = cmds[cid]
        sh = hashed(sheets[k], a.cache)
        prompt = (f"{CONV} {SCENE.get(ch, SCENE['_room'])}\n\nThe user told the avatar: \"{c['utterance']}\"\nSuccess looks like: {c.get('expected')}\n\n"
                  f"Read this contact sheet: {sh}\n\n"
                  "1. verdict: did the avatar do what it was told? 'yes' = clearly done as asked (right body part, side, "
                  "target and sequence); 'partly' = a recognisable attempt that is incomplete, on the wrong side, or only part "
                  "of a compound request; 'no' = not done, or something else. reason: one line why.\n"
                  "2. natural: 1-5, how natural and lifelike the motion looks (5 = like a real person/animated film, "
                  "1 = stiff, robotic, broken poses). robotic: one short line on what looks robotic or unnatural "
                  "(empty string if nothing). Judge the motion, not the art style.")
        out, cached = call_claude(prompt, ACTION_SCHEMA, [sh], f"action {ch}/{m}/{cid}", a.cache)
        return dict({"char": ch, "mode": m, "id": cid, "category": c.get("category"), "sheet": sheets[k], "cached": cached}, **out)

    def pair(k):
        ch, cid, swap = k
        c = cmds[cid]
        x, y = modes[0], modes[1]
        first, second = (y, x) if swap else (x, y)
        sA, sB = hashed(sheets[(ch, first, cid)], a.cache), hashed(sheets[(ch, second, cid)], a.cache)
        prompt = (f"{CONV} {SCENE.get(ch, SCENE['_room'])}\n\nTwo clips of the same avatar given the same command: \"{c['utterance']}\"\n"
                  f"Clip A: {sA}\nClip B: {sB}\n\n"
                  "better: which clip carries out the command better (A, B or tie)? more_natural: which clip's motion looks "
                  "more natural and lifelike — plausible poses, human timing, no snapping, no limbs through the body (A, B "
                  "or tie, independent of whether the command was done)? reason: one line.")
        out, cached = call_claude(prompt, PAIR_SCHEMA, [sA, sB], f"pair {ch}/{cid}", a.cache)
        res = {"char": ch, "id": cid, "category": c.get("category"), "A": first, "B": second, "cached": cached}
        for f in ("better", "more_natural"):
            w = out.get(f)
            res[f + "_mode"] = None if w in (None, "tie") else (first if w == "A" else second)
        return dict(res, **out)

    if "action" in what:
        with cf.ThreadPoolExecutor(a.conc) as ex:
            rows = list(ex.map(act, sorted(sheets)))
        with open(os.path.join(a.out, "action.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"action: {len(rows)} rows")
    if "pairwise" in what and len(modes) >= 2:
        rng = random.Random(a.seed)
        items = [(ch, cid, rng.random() < 0.5) for ch in chars for cid in cmds
                 if (ch, modes[0], cid) in sheets and (ch, modes[1], cid) in sheets]
        with cf.ThreadPoolExecutor(a.conc) as ex:
            rows = list(ex.map(pair, items))
        with open(os.path.join(a.out, "pairwise.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"pairwise: {len(rows)} rows")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
