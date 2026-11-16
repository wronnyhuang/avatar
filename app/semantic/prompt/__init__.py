"""System / user prompts for the semantic planner (header + API reference + examples + the live targets that exist)."""
from __future__ import annotations

import functools
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CHAR_NOTES = {
    "muse": "You are animating the Muse, a plush character standing in a small studio room (whole-body moves work: "
            "jump, spin, turn, step, crouch, bow).",
    "juno": "You are animating Juno, a teacher standing in her classroom studio (whole-body moves work: jump, spin, turn, "
            "step, crouch, bow).",
    "fox": "You are animating the fox, a SEATED animal with paws (no separate fingers: hand shapes become open / fist / "
           "point / thumb), ears and a tail; it cannot stand, step or spin all the way round — whole-body moves "
           "become torso moves, so prefer lean / twist / bounce.",
}


@functools.lru_cache(maxsize=8)
def _read(name):
    return open(os.path.join(HERE, name), encoding="utf-8").read().strip()


def _targets_block(names):
    names = sorted(set(names or ()))
    live = [n for n in names if n not in ("camera",)]
    lines = ["## Live targets (exist right now; usable anywhere a target is accepted)",
             "`camera` = the lens. `user` (also `user_face`) = the person talking to you, where they are now "
             "(the camera when nobody is seen); `user_left` / `user_right` = beside them (their own left/right)."]
    scene = [n for n in live if n not in ("user", "user_left", "user_right", "user_face", "viewer_face")]
    if scene:
        lines.append("Things in your room (look_at / point_at / reach toward them by name; where they are from you): " +
                     ", ".join(f"`{n}` ({_where(n)})" if _where(n) else f"`{n}`" for n in scene) + ".")
    return "\n".join(lines)


_HOME = {}


def _where(name):
    """coarse direction of a scene thing at its home position, from the avatar (its OWN left/right)"""
    p = _HOME.get(name)
    if p is None:
        return ""
    x, y, z = p
    side = "left" if x > 40 else "right" if x < -40 else ""
    depth = "behind" if z < -60 else "in front" if z > 60 else ""
    h = "low" if y < -10 else "high" if y > 120 else ""
    parts = [w for w in (depth, side, h) if w]
    return ", ".join(parts) if parts else "near"


@functools.lru_cache(maxsize=64)
def _system(character, names):
    parts = [_read("prompt_header.md"), CHAR_NOTES.get(character, ""), _read("API.md"), _targets_block(names),
             _read("prompt_examples.md")]
    return "\n\n".join(p for p in parts if p) + "\n"


def system_prompt(character, targets=None):
    """targets: names (or a {name: xyz} dict) of the live targets; text is stable per (character, sorted names)
    (scene things are described at their HOME positions, so moving them does not change the text)."""
    try:
        from ..targets import resolve_targets
        _HOME.update(resolve_targets(str(character), None))
    except Exception:  # noqa: BLE001
        pass
    if isinstance(targets, dict):
        names = tuple(sorted(targets))
    else:
        names = tuple(sorted(targets or ()))
    return _system(str(character), names)


def user_prompt(description, state=None):
    d = " ".join(str(description or "").split())[:400]
    return f'Command: "{d}"\n'
