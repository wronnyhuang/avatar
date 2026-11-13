#!/usr/bin/env python3
"""Turn a run_live.py capture (output.h264 + output.wav + timeline.json + clock.json, all on the server's PLAYBACK
timeline) into avlat measurements of what the phone showed.

    python live_capture.py CAPTURE_DIR [--kinds question,command:head,command:eyes,question] [--session S.json] [--out m.json]

Mapping (the page draws frame t when the audio at t is heard, measured sync p50 about -12 ms):
  heard(timeline t)  = S.base + t                                  (AudioContext clock)
  user speech end    = S.userEndCt - outputLatency  -> timeline t = that - S.base   (the page's own first-audio rule)
  injected local cmd = ctx.currentTime - outputLatency at injection -> timeline t = that - S.base
So every latency below is "as heard/seen on the page", exactly like the page's own first_audio_ms.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import avlat  # noqa: E402


def utterance_ends(clock, gap=0.7):
    """S.userEndCt only moves while the user is audibly speaking (and the avatar is not): an utterance ends at a
    value that does not move again for > gap seconds of ctx time."""
    vals = []
    for c in clock:
        ue = c.get("ue")
        if ue is not None and (not vals or ue != vals[-1][0]):
            vals.append((ue, c))
    ends = []
    for i, (ue, c) in enumerate(vals):
        nxt = vals[i + 1][0] if i + 1 < len(vals) else None
        if nxt is None or nxt - ue > gap:
            start = ue
            j = i
            while j > 0 and ue - vals[j - 1][0] < 10 and vals[j][0] - vals[j - 1][0] <= gap:
                j -= 1
                start = vals[j][0]
            ends.append((start, ue, c))
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("--kinds", default="", help="per utterance: question | command:<region>")
    ap.add_argument("--session")
    ap.add_argument("--out")
    ap.add_argument("--wav-utts", default="", help="'t0-t1,t0-t1' utterances in the mic WAV's own time (the script); "
                    "used instead of S.userEndCt, which stops moving while the avatar is audible")
    ap.add_argument("--mouth-box", help="x0,y0,x1,y1 normalized search box for the mouth (character-specific)")
    ap.add_argument("--wav-to-ct", type=float, default=0.0, help="ctx time = WAV time + this (fit from userEndCt where "
                    "the avatar was quiet; the fake mic starts with the AudioContext in this page)")
    a = ap.parse_args()
    cap = Path(a.cap)
    tl = json.load(open(cap / "timeline.json"))
    ck = json.load(open(cap / "clock.json"))
    times = [v["t"] for v in tl["video"] if not v["flags"] & 2]
    if not (cap / "output.wav").exists() or tl.get("audio_samples", 0) == 0:     # direct-voice page: no 'A' audio comes down
        a.no_audio = True
    clock = [c for c in ck["clock"] if c.get("base") is not None]
    kinds = [k for k in a.kinds.split(",")] if a.kinds else []
    ev = {"user_speech": [], "cues": []}
    if a.wav_utts:
        last = clock[-1]
        utts = [(float(x.split("-")[0]) + a.wav_to_ct, float(x.split("-")[1]) + a.wav_to_ct, last)
                for x in a.wav_utts.split(",")]
    else:
        utts = utterance_ends(clock)
    for i, (u0, u1, c) in enumerate(utts):
        ol, base = c.get("ol") or 0.0, c["base"]
        t0, t1 = u0 - ol - base, u1 - ol - base
        ev["user_speech"].append({"t0": round(t0, 3), "t1": round(t1, 3)})
        k = kinds[i] if i < len(kinds) else "question"
        if k.startswith("command"):
            ev["cues"].append({"t": round(t1, 3), "kind": "command", "region": k.split(":")[1] if ":" in k else "head",
                               "cue": f"spoken command #{i + 1}", "window_s": [0.0, 7.0]})
    for inj in ck.get("injected") or []:
        t = inj["ct"] - (inj.get("ol") or 0.0) - inj["base"]
        reg = inj.get("region") or ("eyes" if "look" in inj["phrase"] else "head")
        kind = "reflex_event" if inj["phrase"].startswith("event:") else "local_command"
        ev["cues"].append({"t": round(t, 3), "kind": kind, "region": reg, "cue": "injected: " + inj["phrase"],
                           "window_s": [0.0, 3.0]})
    (cap / "events.json").write_text(json.dumps(ev, indent=1))
    audio = str(cap / "output.wav")
    if getattr(a, "no_audio", False):           # direct voice: the page's uploaded voice tap IS the playback timeline
        audio = str(cap / "tap.wav") if (cap / "tap.wav").exists() else None
    res = avlat.analyze(str(cap / "output.h264"), audio=audio, frame_times=times,
                        events=str(cap / "events.json"), session=a.session,
                        mouth_hint=[float(v) for v in a.mouth_box.split(",")] if a.mouth_box else None)
    res["local_command_to_motion_ms"] = avlat.summary([m["latency_ms"] for m in res["motion"] if m.get("kind") == "local_command"])
    res["reflex_event_to_motion_ms"] = avlat.summary([m["latency_ms"] for m in res["motion"] if m.get("kind") == "reflex_event"])
    res["command_to_motion_ms"] = avlat.summary([m["latency_ms"] for m in res["motion"] if m.get("kind") == "command"])
    res["events"] = ev
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    short = {k: res.get(k) for k in ("first_audio_ms", "command_to_motion_ms", "local_command_to_motion_ms",
                                      "reflex_event_to_motion_ms", "display")}
    short["lip_offset"] = res["lip_offset"]
    short["motion"] = [{k: m.get(k) for k in ("t", "kind", "cue", "region", "latency_ms", "strongest_region")} for m in res["motion"]]
    short["first_audio"] = res["first_audio"]
    print(json.dumps(short, indent=1, default=float))


if __name__ == "__main__":
    main()
