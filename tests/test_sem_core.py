#!/usr/bin/env python3
"""Tests for app/semantic (pytest or plain: python tests/test_sem_core.py).

streaming == replay, time commitment, no pops, only-driven channels, left/right, targets, malformed programs, caps,
speed.  Uses every 7th program of tests/fixtures/sem_programs.json as a corpus.
"""
import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import semantic as sem  # noqa: E402

CORPUS = os.path.join(HERE, "fixtures", "sem_programs.json")
CHARS = ("muse", "fox")


def fence(src):
    return "```python\n" + src.strip("\n") + "\n```\n"


def stream(ch, text, chunk=5, targets=None):
    s = sem.Stream(ch, targets=targets)
    batches = []
    for i in range(0, len(text), chunk):
        batches += s.feed(text[i:i + chunk])
    batches += s.finish()
    return s, batches


def poses_of(batches):
    return [p for b in batches for p in b["poses"]]


def corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return [p["program"] for p in json.load(f)["programs"]][::7]


# --------------------------------------------------------------------------------------------------- interface
def test_interface():
    assert set(CHARS) <= set(sem.CHARACTERS)
    for ch in CHARS:
        names = sorted(sem.resolve_targets(ch, None))
        assert "camera" in names and "user" in names
        a = sem.system_prompt(ch, names)
        b = sem.system_prompt(ch, list(reversed(names)))
        assert a == b and "reach(" in a and "user" in a
    assert "board" in sem.resolve_targets("muse", None)
    assert "board" not in sem.resolve_targets("fox", None)
    assert sem.user_prompt("wave hello").startswith("Command:")


# --------------------------------------------------------------------------------------------------- streaming
def test_stream_equals_replay_and_time_commitment():
    progs = corpus() or ["look_at('left')\nwave()\nhold(0.3)"]
    for ch in CHARS:
        for src in progs:
            s, bs = stream(ch, fence(src), chunk=7)
            got = poses_of(bs)
            ref, rep = sem.replay(ch, src)
            assert len(got) == len(ref), (ch, src[:60])
            for p, q in zip(got, ref):
                assert set(p) == set(q), (ch, p["t"])
                for k in p:
                    assert abs(p[k] - q[k]) < 1e-9, (ch, k, p["t"])
            # time commitment: strictly increasing times across batches, batches never overlap
            ts = [p["t"] for p in got]
            assert all(b > a for a, b in zip(ts, ts[1:]))
            t_last = -1
            for b in bs:
                if b["poses"]:
                    assert b["poses"][0]["t"] > t_last
                    t_last = b["poses"][-1]["t"]
            assert bs[-1]["final"]


def test_first_statement_streams_before_the_rest():
    s = sem.Stream("muse")
    out = s.feed("```python\nreach('right', 'nose', dur=0.5)\n")       # simple statement: complete at its newline
    assert out and out[0]["poses"], "first statement must emit at once"
    out2 = s.feed("with parallel():\n    look_at('left')\n")
    assert out2 == []                                                   # a block waits for its dedent
    out3 = s.feed("    smile()\nhold(0.4)\n```")
    assert out3 and out3[0]["poses"][0]["t"] > out[-1]["poses"][-1]["t"]


def test_no_pops():
    """no isolated jumps: a pose-to-pose (50 ms) step is never much bigger than both of its neighbours (a fast but
    smooth move ramps up and down; a pop is one big step between small ones), and never above an absolute bound."""
    absmax = {"hand": 0.8, "head": 0.6, "gaze": 1.2, "body": 1.2, "root": 0.2, "squash": 0.2, "smile": 0.5,
              "brow": 0.5, "shoulder": 0.5, "mouth": 0.6}
    progs = corpus() or ["wave()"]
    for ch in ("muse", "fox"):
        for src in progs:
            P, _ = sem.replay(ch, src)
            keys = set().union(*[set(p) for p in P]) - {"t"}
            for k in keys:
                g = k.split("_")[0]
                v = [p.get(k) for p in P]
                d = [abs(b - a) if a is not None and b is not None else 0.0 for a, b in zip(v, v[1:])]
                for i_, x in enumerate(d):
                    if g in absmax:
                        assert x <= absmax[g], (ch, k, P[i_]["t"], x, src[:80])
                    nb = max(d[i_ - 1] if i_ > 0 else 0.0, d[i_ + 1] if i_ + 1 < len(d) else 0.0)
                    if not k.startswith(("blink", "point", "thumb")):
                        assert x <= 3.0 * nb + 0.12, (ch, k, P[i_]["t"], x, nb, src[:80])


def test_hold_then_return_is_smooth():
    P, rep = sem.replay("muse", "reach('right', 'chest', dur=0.6)\nhold(0.5)")
    xs = [p["hand_l_y"] for p in P if "hand_l_y" in p]
    d = [abs(b - a) for a, b in zip(xs, xs[1:])]
    assert max(d) < 0.25 and abs(xs[-1]) < 1e-3          # back at rest, no jump
    assert not rep["keep_pose"]


# --------------------------------------------------------------------------------------------------- channels
def test_only_driven_channels():
    P, rep = sem.replay("muse", "look_at('left', dur=0.5)\nhold(0.5)")
    keys = set().union(*[set(p) for p in P]) - {"t"}
    assert keys <= {"head_yaw", "head_pitch", "gaze_x", "gaze_y", "body_yaw"}, keys   # body_yaw: past head range
    P, rep = sem.replay("fox", "expression('smile')\nhold(0.5)")
    keys = set().union(*[set(p) for p in P]) - {"t"}
    assert not any(k.startswith(("hand_", "head_", "body_", "gaze")) for k in keys), keys
    assert "smile" in keys and "ear_l" in keys          # the fox's ears follow its face
    for ch in CHARS:                                     # no idle noise / blinks / breathing from the interpreter
        P, _ = sem.replay(ch, "wave()\nhold(1.0)")
        keys = set().union(*[set(p) for p in P])
        assert not keys & {"blink_l", "blink_r", "breath", "tail_wag", "ear_flick_l"}


def test_channels_dropped_when_done():
    """a wink's blink dial leaves the poses once the wink is over (blinks / mirroring come back)"""
    P, _ = sem.replay("muse", "wink('left')\nhold(1.5)\nnod()")
    late = [p for p in P if 1.2 < p["t"] < 1.8]
    assert late and not any("blink_r" in p for p in late)
    assert any("blink_r" in p for p in P)


def test_rig_names_and_ranges():
    from semantic.rigmap import RANGE
    for ch in CHARS:
        P, rep = sem.replay(ch, "jump()\nspin()\nclap()\nexpression('surprise')\nstick_out_tongue()\nhold(0.3)")
        for p in P:
            for k, v in p.items():
                if k == "t":
                    continue
                assert k in RANGE, k
                lo, hi = RANGE[k]
                assert lo - 1e-9 <= v <= max(hi, 1.4) + 1e-9, (k, v)
        if ch == "fox":
            assert not any(k.startswith("root_") for p in P for k in p)
        else:
            assert any("root_y" in p for p in P)


# --------------------------------------------------------------------------------------------------- sides
def last(P, k):
    v = [p[k] for p in P if k in p]
    return v


def test_left_right():
    for ch in CHARS:
        P, _ = sem.replay(ch, "point_at('left', hand='left', dur=0.6)\nhold(0.5)", hz=20)
        keys = set().union(*[set(p) for p in P])
        assert "hand_r_x" in keys and "hand_l_x" not in keys          # the avatar's LEFT hand = screen right
        assert max(last(P, "hand_r_x")) > 0.4                         # out to its own side
        P, _ = sem.replay(ch, "wink('left')")
        keys = set().union(*[set(p) for p in P])
        assert "blink_r" in keys and "blink_l" not in keys
        P, _ = sem.replay(ch, "look_at('left')\nhold(0.4)")
        assert max(last(P, "head_yaw")) > 0.1                        # + = toward screen right = its own left
        P, _ = sem.replay(ch, "tilt_head('left', 20)\nhold(0.4)")
        assert max(last(P, "head_roll")) > 0.1
        P, _ = sem.replay(ch, "lean('left', 15)\nhold(0.4)")
        assert max(last(P, "body_lean")) > 0.1
        P, _ = sem.replay(ch, "reach('right', 'right_ear')\nhold(0.4)")
        assert "hand_l_y" in set().union(*[set(p) for p in P])


def test_targets_land():
    """hand goals for named targets: high for the head, low for the belly, up for 'up'"""
    for ch in CHARS:
        def y(prog, k="hand_l_y"):
            P, _ = sem.replay(ch, prog + "\nhold(0.6)")
            return max(last(P, k))
        assert y("reach('right', 'top_of_head')") > y("reach('right', 'chest')") > y("reach('right', 'belly')") - 0.05
        assert y("touch('right', 'nose')") > 0.5
        assert y("point_at('up')") > 0.7


def test_live_targets():
    st = {"sem": {"user": {"present": True, "ax": 0.3, "ay": 0.0}}}
    t = sem.resolve_targets("muse", st)
    t0 = sem.resolve_targets("muse", None)
    assert sorted(t) == sorted(t0)                                   # stable name set
    assert t["user"][0] > 0                                         # person on screen right = the avatar's left
    P, rep = sem.replay("muse", "look_at('user')\nhold(0.4)", targets=t)
    assert max(last(P, "head_yaw")) > 0.05
    P, rep = sem.replay("muse", "point_at('board')\nhold(0.4)", targets=t0)
    assert not rep["errors"]
    P, rep = sem.replay("fox", "look_at('the_moon_target')\nhold(0.4)", targets=sem.resolve_targets("fox", None))
    assert not rep["errors"] and any("camera" in w for w in rep["warnings"])


# --------------------------------------------------------------------------------------------------- robustness
def test_malformed():
    bad = [
        "import os\nwave()",
        "wave(\nhold(1)",
        "reach('right', 'nose'\nsmile()\nhold(0.5)",
        "dance_like_crazy()\nsmile()",
        "for i in range(100):\n    nod()\nsmile()",
        "x = __import__('os')\nwave()",
        "open('/etc/passwd')\nwave()",
        "while True:\n    nod()",
        "reach('middle', 'nose')\nwave()",
        "with parallel():\n    look_at('left')\n    smile(",
        "",
        "Sure! here you go.",
        "```python\n```",
    ]
    for src in bad:
        for ch in CHARS:
            s, bs = stream(ch, src if src.startswith("```") else fence(src), chunk=3)
            r = s.report()
            assert bs[-1]["final"]
            if not poses_of(bs):
                assert r["errors"], src
    s, bs = stream("muse", "```python\nimport os\n```")
    assert not poses_of(bs) and s.report()["errors"]


def test_partial_stream_and_finish_on_cut_text():
    s = sem.Stream("muse")
    s.feed("```python\nwave()\nreach('right', 'no")
    bs = s.finish()
    assert bs[-1]["final"]
    assert s.report()["n_statements"] >= 1


def test_caps():
    long = "\n".join(["nod(times=1)"] * 80)
    P, rep = sem.replay("muse", long)
    assert rep["n_statements"] <= 60 and P[-1]["t"] <= 12.0 + 1e-6
    P, rep = sem.replay("muse", "hold(8)\nwave()\nhold(8)\nsmile()")
    assert P[-1]["t"] <= 12.0 + 1e-6 and rep["errors"]


def test_keep_pose():
    P, rep = sem.replay("muse", "raise_hand('right')\nkeep_pose()")
    assert rep["keep_pose"] and P[-1]["hand_l_y"] > 0.5


# --------------------------------------------------------------------------------------------------- speed
def test_speed():
    progs = corpus() or ["wave()\nhold(1)"]
    per, whole = [], []
    for src in progs:
        s = sem.Stream("muse")
        text = fence(src)
        for i in range(0, len(text), 6):
            t0 = time.perf_counter()
            b = s.feed(text[i:i + 6])
            dt = (time.perf_counter() - t0) * 1000
            if b:
                per.append(dt / len(b))
        s.finish()
        t0 = time.perf_counter()
        sem.replay("muse", src)
        whole.append((time.perf_counter() - t0) * 1000)
    per.sort()
    assert statistics.median(per) <= 30 and per[int(0.95 * (len(per) - 1))] <= 80, per[-5:]
    assert statistics.median(whole) <= 150


def test_contact_pins_torso_and_head():
    """a hand on a body / face target holds the undriven torso (and head, for face targets) at the program's value so the
    live layers cannot move the hand off target; free-space gestures pin nothing; released after the contact"""
    P, _ = sem.replay("fox", "touch('right', 'chin')\nhold(1.0)")
    mid = [p for p in P if 0.7 < p["t"] < 1.4]
    assert mid and all({"body_lean", "body_yaw", "head_yaw"} <= set(p) for p in mid)
    assert not any("body_lean" in p for p in P[-3:])
    P, _ = sem.replay("muse", "wave()\npoint_at('left')")
    assert not any(k.startswith(("body_", "head_")) for p in P for k in p)
    P, _ = sem.replay("muse", "reach('left', 'right_shoulder')\nhold(0.6)")
    assert any("body_lean" in p for p in P) and not any("head_yaw" in p for p in P)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
