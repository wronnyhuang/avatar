#!/usr/bin/env python3
"""Latency-table numbers from a server session JSON (data/session-*.json of any avatar instance), with the page's
stats rows. Works on live-baseline (server voice) and integ (direct voice) sessions.

    python session_report.py SESSION.json [--skip-s 0] [--json]

first audio (page-measured: end of user speech -> first voiced sample HEARD), server user-end -> vendor audio, displayed
fps (page, 2 s windows; render-active windows only), sync, poll RTT, video RTT (direct mode: tap -> frame decoded),
renderer frame time (render_ms / render_p95 from the server's stats snapshots when kept), up/down kbps (page xchg).
"""
import argparse
import json
import statistics as st


def pc(xs, q):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(len(xs) - 1, int(k) + 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def report(d, skip_s=0.0):
    ps = [p for p in d.get("page_stats") or [] if (p.get("t") or 0) >= skip_s]
    vid = [p for p in ps if (p.get("frames_in") or 0) > 0]
    fps = [p.get("fps") for p in vid[1:]]
    voice = [p.get("voice") or {} for p in ps]
    vr = [(v.get("video_rtt_ms") or {}) for v in voice if v.get("video_rtt_ms")]
    xch = [p.get("xchg") for p in ps if p.get("xchg")]
    rs = [p.get("render") for p in ps if isinstance(p.get("render"), dict)]
    out = {
        "sid": d.get("sid"), "character": d.get("character"), "live_s": d.get("live_s"), "provider": d.get("provider"),
        "first_audio_ms": d.get("first_audio_ms"), "first_audio_each": d.get("first_audio_each"),
        "hub_user_end_to_vendor_ms": d.get("hub_user_end_to_vendor_ms"),
        "fps_displayed": {"p50": pc(fps, .5), "p05": pc(fps, .05), "min": min(fps) if fps else None, "n": len(fps)},
        "sync_ms_p50": pc([(p.get("sync_ms") or {}).get("p50") for p in vid], .5),
        "sync_abs_p95": pc([(p.get("sync_ms") or {}).get("abs_p95") for p in vid], .5),
        "poll_rtt_ms_p50": pc([(p.get("poll_rtt_ms") or {}).get("p50") for p in ps], .5),
        "poll_rtt_ms_p95": pc([(p.get("poll_rtt_ms") or {}).get("p95") for p in ps], .5),
        "delay_ms_last": (ps[-1].get("delay_ms") if ps else None),
        "video_rtt_ms_last": vr[-1] if vr else None,
        "late_frames": d.get("late_frames"), "dropped_frames": d.get("dropped_frames"),
        "audio_underruns": d.get("audio_underruns"), "usd": d.get("usd"),
        "renderer_last": ((d.get("render") or {}).get("stats")),
    }
    if xch:
        out["xchg"] = {"req_s_p50": pc([x.get("req_s") for x in xch], .5), "up_kbps_p50": pc([x.get("up_kbps") for x in xch], .5),
                       "up_kbps_p95": pc([x.get("up_kbps") for x in xch], .95), "kicks": xch[-1].get("kicks"),
                       "down_kbps_p50": pc([x.get("down_kbps") for x in xch], .5), "lanes": xch[-1].get("lanes")}
    if rs:
        out["renderer"] = {"fps_p50": pc([r.get("fps") for r in rs], .5), "render_ms_p50": pc([r.get("render_ms") for r in rs], .5),
                           "render_p95_p50": pc([r.get("render_p95") for r in rs], .5)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--skip-s", type=float, default=0.0)
    a = ap.parse_args()
    print(json.dumps(report(json.load(open(a.session)), a.skip_s), indent=1))
