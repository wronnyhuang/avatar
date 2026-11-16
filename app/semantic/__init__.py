"""app/semantic — the bake-off's r3 "semantic goal API" as a live, streaming planner backend.

    import semantic as sem
    sem.CHARACTERS                                   # ('muse', 'fox')
    sem.system_prompt(character, targets=None)       # stable text per (character, sorted target names)
    sem.user_prompt(description, state=None)
    sem.resolve_targets(character, state)            # {name: (x_cm, y_cm, z_cm)} pelvis frame; always has 'camera'
    s = sem.Stream(character, targets=None, seed=0, hz=20, t_origin=0.0)
    s.feed(text_delta) -> [batch]; s.finish() -> [batch]; s.report() -> dict
    sem.replay(character, program, targets=None, seed=0, hz=20) -> (poses, report)

batch = {"stmt", "t0", "t1", "poses": [{"t": s, <rig param>: value, ...}], "final"}; rig params = app/motion.js names
(only the dials the program drives), values in rig units, range-safe.  See docs/SEMANTIC.md section 2 and
docs/SEMANTIC.md.
"""
from __future__ import annotations

from .stream import Stream, replay          # noqa: F401
from .targets import resolve_targets        # noqa: F401
from .prompt import system_prompt, user_prompt   # noqa: F401

CHARACTERS = ("muse", "fox")      # rig maps exist for a third (human) rig family too; not in this repo
