"""rig: character capabilities — what a character HAS (a body, a scene, teacher tools, feelings ...), not who it is.

The hub used to gate the whole-body / scene / teacher / feelings features on `character == "muse"`. Juno (the mesh
teacher) gets every one of them, so those gates ask `charcaps.has(character, cap)` instead. Muse and the other
11 characters keep exactly the answers they had (tests/test_juno_gates.py pins it). The page derives the same set from
config.caps (app.py config_msg adds it), see app/charcaps.js.

Caps:
  body      config.body: whole-body moves, walking, sitting, hands on props (motion.js engineOn)
  scene     config.props: the room (board, TV, desk, objects); hub act grammar + object tool
  act       move / object voice tools + the hub act grammar (rtc.ACT_CHARS, intent_hub.ACT_CHARACTERS defaults)
  memory    action memory notes / recall tool (rtc.MEMORY_CHARACTERS default)
  speech    page action log + co-speech gestures (page.html speechFeat)
  hands     the action log names its limb "hand" (not "paw")
  affect    the avatar's own feelings (app.py Affect)
  tv        the TV agent + warm models (app.py tv_attach)
  surfaces  the board / TV live surfaces sink (app.py surfaces_attach)
  marks     head-anchored emotion marks (rt_engine marks pass; a non-Muse shader opts in with `#define CHAR_MARKS`)
"""
from __future__ import annotations

ALL = ("body", "scene", "act", "memory", "speech", "hands", "affect", "tv", "surfaces", "marks")
CAPS: dict[str, frozenset] = {
    "muse": frozenset(ALL),
    "juno": frozenset(ALL),
    "junor": frozenset(ALL),    # pipe: Juno, realistic
}
# the persona handle (app.CHARACTERS[c][1], rtc.PERSONALITY key) of the characters above that are not shapes/animals
WHAT = {"muse": "plush companion", "juno": "teacher", "junor": "teacher"}
# a character without its own planner prompt / retarget table plans and retargets with this one's (same control vector)
ALIAS = {"juno": "muse", "junor": "muse"}
# the mascots (app/mascots.py) are full scene characters on the muse's skeleton: every cap, the muse's prompts
try:
    import mascots as _mascots
except ImportError:      # imported as a package module
    from . import mascots as _mascots
for _k, _m in _mascots.MASCOTS.items():
    CAPS[_k] = frozenset(ALL)
    WHAT[_k] = _m["what"]
    ALIAS[_k] = "muse"


def has(character: str | None, cap: str) -> bool:
    return bool(character) and cap in CAPS.get(character, ())


def caps(character: str | None) -> list[str]:
    """sorted list for config.caps ([] for a character with none)"""
    return sorted(CAPS.get(character or "", ()))


def with_cap(cap: str) -> list[str]:
    """the characters that have `cap`, in table order (env-default lists: 'muse,juno')"""
    return [c for c, s in CAPS.items() if cap in s]


def csv(cap: str) -> str:
    return ",".join(with_cap(cap))


def alias(character: str | None) -> str | None:
    return ALIAS.get(character or "")


def alias_prompts(prompts: dict) -> dict:
    """planner.load_prompts() result: a character with no prompt of its own borrows its alias's (in place)"""
    for c, a in ALIAS.items():
        if c not in prompts and a in prompts:
            prompts[c] = prompts[a]
    return prompts
