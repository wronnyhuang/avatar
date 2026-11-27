#!/usr/bin/env python3
"""which entries of a plan cache this tree's planner would still serve.

    python3 tools/plan_cache_audit.py <AVATAR_HOME>/cache/plans [--variant q1|frozen] [--delete-dead]

An entry is LIVE when its prompt sha is the one its character plans with now (app/planner.py: the configured prompt
variant, the scene appendix aside) and planner.plan_issues() has no hard issue; DEAD otherwise (never served again: the
next request for it is planned afresh and overwrites it): dead:check (the plan check rejects it), dead:frozen-prompt
(written from the frozen prompt of a character that now plans with a variant), dead:other-prompt (an older prompt or
appendix). --delete-dead removes the dead ones. Reads only, unless --delete-dead."""
import argparse, collections, glob, hashlib, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import planner  # noqa: E402


def shas(pl, ch) -> set:
    """the prompt shas a plan for `ch` can carry: its prompt alone, and with its scene's appendix (a plan with things)"""
    P = pl.prompts.get(ch)
    if not P:
        return set()
    out = {P["sha256"]}
    scfg = planner.load_scene(ch, pl.scene_dir)
    app = planner.scene_appendix(scfg, pl.appendix_path) if scfg else ""
    if app:
        out.add(hashlib.sha256((P["text"].rstrip("\n") + "\n\n" + app + "\n").encode("utf-8")).hexdigest())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--variant", default=None)
    ap.add_argument("--delete-dead", action="store_true")
    a = ap.parse_args()
    cfg = {} if a.variant is None else {"prompt_variant": a.variant}
    now = planner.Planner(cfg, lambda m: None)
    frozen = planner.Planner({"prompt_variant": "frozen"}, lambda m: None)
    c = collections.Counter(); dead = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        try:
            e = json.load(open(f))
        except (OSError, ValueError):
            c["unreadable"] += 1; continue
        ch = e.get("character")
        issues = [i for i in planner.plan_issues(e.get("poses")) if i in planner.CHECK_HARD]
        cur, old = shas(now, ch), shas(frozen, ch)
        if issues:
            st = "dead:check"
        elif e.get("prompt_sha") in cur:
            st = "live"
        elif e.get("prompt_sha") in old:
            st = "dead:frozen-prompt"
        else:
            st = "dead:other-prompt"
        c[st] += 1
        if st.startswith("dead"):
            dead.append((st, ch, str(e.get("description"))[:70], f))
    for st, ch, d, f in dead:
        print(f"{st:20s} {ch:9s} {d}")
        if a.delete_dead:
            os.remove(f)
    print(json.dumps({"dir": a.dir, "variant": now.variant or "frozen", "entries": sum(c.values()), **c,
                      "deleted": len(dead) if a.delete_dead else 0}))


if __name__ == "__main__":
    main()
