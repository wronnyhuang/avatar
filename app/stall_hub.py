"""Video stall watchdog, hub side (the page side is app/stallwatch.js).

What the hub does (wired in app.py Session):
  * RENDERER SILENT: while a session is live, the render link is up and the hub is feeding the renderer audio (the
    page's playback tap or the hub voice clock), no video frame from the renderer for `key_s` -> ask it for a keyframe
    (once); for `reattach_s` -> re-attach the render link (close it, connect again, `start` with the session's params:
    the same path a session start uses; the engine is cached, so a start is ~1 s). The box itself is never created,
    restarted or terminated here: a box that is not running is left to the box logic.
  * RENDER LINK CLOSED mid-session (until now: "video lost — voice continues" for the rest of the session) -> the same
    re-attach, `relink_delay_s` later.
  * PAGE ASKS (the page's ladder): {type:'stall_help', step:'skip'} -> skip to live (drop the queued video, keyframe);
    step:'hub' -> re-attach when the renderer is silent or the link is gone, else skip to live.
  * Re-attaches are rate-limited: at most `reattach_max` per `reattach_window_s` per session, `reattach_gap_s` apart.
  * RECORD + ALERT: the page's stall reports and the hub's own actions go to the session record (summary()['stalls']);
    one log line per stall; an UNRECOVERED stall older than `note_s` writes ONE alert line to the hub log (at most one
    per `note_every_s` across sessions; the session record keeps it under stalls.notes).
Switch: config "stall_watch": {"enabled": false} turns both sides off (the page also obeys ?stall=0).
Test faults (config "test_faults": true only, never on a live service): {type:'test_fault', kind, s} from the page --
  hold (the hub stops forwarding video for s), garble (frames forwarded with corrupted payload for s), unlink (the
  render link is dropped), mute (the renderer link stops delivering frames until it is re-attached), static (the
  renderer link repeats one picture until it is re-attached).
"""
from __future__ import annotations

import time

DEFAULTS = {
    "enabled": True,
    "key_s": 3.0,             # renderer silent this long while it is fed audio -> keyframe request
    "reattach_s": 6.0,        # ... this long -> re-attach the render link
    "relink_delay_s": 1.0,    # render link closed mid-session -> re-attach after this
    "reattach_max": 3, "reattach_window_s": 300.0, "reattach_gap_s": 5.0,
    "audio_fresh_s": 1.5,     # the renderer counts as fed while audio went to it within this
    "note_s": 15.0,           # an unrecovered stall this old writes an alert line to the hub log
    "note_every_s": 1800.0,   # at most one stall alert per 30 min (all sessions)
}


def cfg(raw) -> dict:
    c = dict(DEFAULTS)
    if isinstance(raw, dict):
        c.update({k: v for k, v in raw.items() if k in DEFAULTS or k in ("detect_ms", "first_ms", "static_ms", "alert_ms",
                                                                          "content", "tick_ms", "hub_max", "hub_repeat_ms")})
    elif raw is False:
        c["enabled"] = False
    return c


def page_cfg(c: dict) -> dict:
    """What the page's watchdog gets in the session config (its own thresholds keep their defaults unless set)."""
    out = {"enabled": bool(c.get("enabled", True))}
    for k in ("detect_ms", "first_ms", "static_ms", "alert_ms", "content", "tick_ms", "hub_max", "hub_repeat_ms"):
        if k in c:
            out[k] = c[k]
    return out


class Notes:
    """One alert per unrecovered stall, at most one per note_every_s across sessions: a line in the hub log."""

    def __init__(self, log):
        self.log = log
        self.last_at = 0.0
        self.sent: list[dict] = []

    def due(self, c: dict, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now - self.last_at >= float(c.get("note_every_s", 1800))

    def note(self, c: dict, text: str) -> dict:
        """Returns the record {t, sink, text, delivered}; sink "log", or "throttled" inside note_every_s."""
        now = time.time()
        rec = {"t": now, "sink": "log", "text": text[:400], "delivered": False}
        if not self.due(c, now):
            rec["sink"] = "throttled"
            return rec
        self.last_at = now
        self.log(f"STALL ALERT: {text[:300]}")
        rec["delivered"] = True
        self.sent.append(rec)
        return rec


def summarize(page: list, hub_events: list, notes: list, last_stats: dict | None) -> dict:
    """The session record's 'stalls' block: the page's closed stalls (and the one still open), the hub's own actions."""
    closed = [p for p in page if p.get("ev") == "close"]
    opened = {p.get("id"): p for p in page if p.get("ev") in ("open", "update")}
    for p in closed:
        opened.pop(p.get("id"), None)
    out = {"page": closed[-40:], "open": list(opened.values())[-3:], "hub": hub_events[-40:], "notes": notes[-10:]}
    ls = last_stats or {}
    for k in ("stall_ms", "stalls", "longest_stall_ms", "recoveries"):
        if k in ls:
            out[k] = ls[k]
    if isinstance(ls.get("stall"), dict):
        out["wait_ms"] = ls["stall"].get("wait_ms")
        out["unrecovered"] = ls["stall"].get("unrecovered")
    by = {}
    for p in closed:
        k = (p.get("cls"), p.get("fixed_by") or p.get("outcome"))
        by[f"{k[0]}/{k[1]}"] = by.get(f"{k[0]}/{k[1]}", 0) + 1
    out["by_class"] = by
    out["hub_reattaches"] = sum(1 for e in hub_events if e.get("ev") == "reattach")
    out["hub_reattach_ok"] = sum(1 for e in hub_events if e.get("ev") == "reattach" and e.get("ok"))
    return out


def garble(msg: bytes) -> bytes:
    """test fault: keep the 16-byte 'V' header, corrupt the encoded payload (every 5th byte past the first NAL header)."""
    b = bytearray(msg)
    for i in range(16 + 8, len(b), 5):
        b[i] ^= 0x5A
    return bytes(b)


def static_copy(msg: bytes, key: bytes) -> bytes:
    """test fault: the real frame's header (number, time) with a cached keyframe's payload: the same picture again."""
    h = bytearray(msg[:16])
    h[1] |= 1
    return bytes(h) + key[16:]
