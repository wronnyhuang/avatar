"""Person-disjoint splits for CANON v1. Deterministic from the FULL Seamless queue, so a pair's split
never changes as more pairs land; other sources add their own persons (ids prefixed by source) without moving any
earlier assignment.

Seamless rules (participant id = vendor + P#### with the trailing 'A' collapsed):
  1. Meta-test participants -> test.
  2. Meta-dev pairs (the only ones with video) are grouped into participant-connected components (dev pairs only),
     ordered by first appearance in the queue, and dealt DEV_PATTERN (test / train / val): test ones give the eval
     harness held-out sessions with video; train ones are what the fitter may learn from; val ones are the fitter's
     held-out tuning / noise-profile people.
  3. Remaining participants: the participant graph over the whole queue has one giant component (~83% of pairs)
     plus ~235 small ones; every 4th small component (queue order, no test/dev participant) -> val; the rest -> train.
  4. A pair is kept only when both participants share a split; otherwise it is dropped (reason recorded).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from .seamless import person_id

DEV_PATTERN = ["test", "train", "train", "val", "test", "train", "train"]
SMALL_VAL_EVERY = 4


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def seamless_plan(queue):
    """queue rows (idx, label, split, pair, files) -> (pair -> {split|None, drop}, person -> split, notes)."""
    q = sorted(queue, key=lambda x: x["idx"])
    person = {}
    for x in q:
        if x["split"] == "test":
            for f in x["files"]:
                person[person_id(f)] = "test"
    # dev components
    uf = UF()
    dev = [x for x in q if x["split"] == "dev"]
    for x in dev:
        a, b = [person_id(f) for f in x["files"]]
        uf.union(a, b)
    first = {}
    for x in dev:
        first.setdefault(uf.find(person_id(x["files"][0])), x["idx"])
    comps = sorted(first, key=lambda r: first[r])
    dev_role = {r: DEV_PATTERN[i % len(DEV_PATTERN)] for i, r in enumerate(comps)}
    dev_person = set()
    for x in dev:
        for f in x["files"]:
            pid = person_id(f)
            dev_person.add(pid)
            person.setdefault(pid, dev_role[uf.find(pid)])
    # whole-queue components -> small ones for val
    ug = UF()
    for x in q:
        a, b = [person_id(f) for f in x["files"]]
        ug.union(a, b)
    size = Counter(ug.find(person_id(x["files"][0])) for x in q)
    giant = size.most_common(1)[0][0]
    first_all = {}
    for x in q:
        first_all.setdefault(ug.find(person_id(x["files"][0])), x["idx"])
    members = defaultdict(set)
    for x in q:
        for f in x["files"]:
            members[ug.find(person_id(f))].add(person_id(f))
    small = [r for r in sorted(first_all, key=lambda r: first_all[r]) if r != giant
             and not any(p in person for p in members[r])]
    for i, r in enumerate(small):
        s = "val" if i % SMALL_VAL_EVERY == 1 else "train"
        for p in members[r]:
            person.setdefault(p, s)
    for x in q:
        for f in x["files"]:
            person.setdefault(person_id(f), "train")
    plan, notes = {}, Counter()
    for x in q:
        ss = {person[person_id(f)] for f in x["files"]}
        if len(ss) == 1:
            plan[x["pair"]] = dict(split=ss.pop(), meta_split=x["split"], drop=None)
        else:
            plan[x["pair"]] = dict(split=None, meta_split=x["split"],
                                   drop="participants span splits " + "/".join(sorted(ss)))
            notes[f"dropped_meta_{x['split']}"] += 1
    return plan, person, dict(notes)
