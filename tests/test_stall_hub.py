"""stall watchdog, hub side helpers (app/stall_hub.py): config switch, the unrecovered-stall alert (a hub log line) and its throttle, the
session-record summary, the test-fault frame rewrites.   python3 tests/test_stall_hub.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import stall_hub as sh  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -- {detail}"))
    if not ok:
        FAILS.append(name)


c = sh.cfg(None)
check("defaults on", c["enabled"] is True and c["key_s"] == 3.0 and c["reattach_s"] == 6.0 and c["note_s"] == 15.0)
check("config false = off", sh.cfg(False)["enabled"] is False)
check("config dict off", sh.cfg({"enabled": False})["enabled"] is False)
check("unknown keys ignored", "bogus" not in sh.cfg({"bogus": 1}))
check("page cfg carries the switch and page thresholds only",
      sh.page_cfg(sh.cfg({"enabled": True, "detect_ms": 4000, "note_s": 20})) == {"enabled": True, "detect_ms": 4000})

logs = []
n = sh.Notes(logs.append)
r1 = n.note(c, "first")
r2 = n.note(c, "second")
check("alert: one hub log line", r1["delivered"] and r1["sink"] == "log" and len(logs) == 1 and "first" in logs[0], (r1, logs))
check("throttle: a second alert inside 30 min is not written", r2["sink"] == "throttled" and len(n.sent) == 1 and len(logs) == 1, r2)
check("throttle reopens after note_every_s", n.due(c, n.last_at + 1801) and not n.due(c, n.last_at + 60))

page = [{"ev": "open", "id": 1, "cls": "no_frames"}, {"ev": "close", "id": 1, "cls": "no_frames", "outcome": "recovered", "fixed_by": "resync"},
        {"ev": "open", "id": 2, "cls": "static"}, {"ev": "update", "id": 2, "cls": "static", "outcome": "open"}]
hubev = [{"ev": "reattach", "ok": True}, {"ev": "reattach", "ok": False}, {"ev": "skip"}]
sm = sh.summarize(page, hubev, [], {"stall_ms": 9000, "stalls": 2, "longest_stall_ms": 6000, "recoveries": 1,
                                    "stall": {"wait_ms": {"hidden": 3000}, "unrecovered": 0}})
check("summary: closed list, open list, counters, by class", len(sm["page"]) == 1 and len(sm["open"]) == 1 and sm["open"][0]["id"] == 2
      and sm["stall_ms"] == 9000 and sm["by_class"] == {"no_frames/resync": 1} and sm["hub_reattaches"] == 2 and sm["hub_reattach_ok"] == 1
      and sm["wait_ms"] == {"hidden": 3000}, sm)

v = b"V\x00\x00\x00" + bytes(12) + bytes(range(200))
g = sh.garble(v)
check("garble keeps the header and length, changes the payload", g[:24] == v[:24] and len(g) == len(v) and g != v)
key = b"V\x01\x00\x00" + b"\x07" * 12 + b"KEYDATA"
sc = sh.static_copy(b"V\x00\x00\x00" + b"\x09" * 12 + b"delta", key)
check("static copy: the frame's own header, keyframe flag, the cached keyframe's payload",
      sc[0:1] == b"V" and sc[1] & 1 and sc[4:16] == b"\x09" * 12 and sc[16:] == b"KEYDATA", sc)

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
sys.exit(1 if FAILS else 0)
