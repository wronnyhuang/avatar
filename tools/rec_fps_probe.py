#!/usr/bin/env python3
"""rec_fps_probe.py <mp4> -> JSON: how smooth a session recording really is.

  container: avg_frame_rate / r_frame_rate / nb_frames / duration (ffprobe)
  timing:    frame timestamps -> frame rate from timestamps, longest gap, gaps over 1.5 / 3 frame periods of 30 fps
  distinct:  frames whose picture changed (ffmpeg mpdecimate drops near-duplicates), whole frame and AVATAR ONLY
             (the left 72 % of the frame: the self-view box sits in the bottom-right corner and a camera changes every
             frame, which would hide a stuck avatar), plus the longest run without an avatar change
"""
import json, subprocess, sys


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def pts_list(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    ts = sorted(float(x) for x in r.stdout.split() if x.strip() and x.strip() != "N/A")
    return ts


def kept_pts(path, vf):
    """timestamps of the frames mpdecimate keeps (= the picture changed)"""
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-vf", vf + ",mpdecimate,showinfo", "-fps_mode", "vfr",
             "-f", "null", "-"])
    out = []
    for line in r.stderr.splitlines():
        if "Parsed_showinfo" in line and " pts_time:" in line:
            try:
                out.append(float(line.split(" pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    return out


def gaps(ts):
    return [b - a for a, b in zip(ts, ts[1:])]


def main():
    path = sys.argv[1]
    pr = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,codec_name:format=duration", "-of", "json", path]).stdout)
    st, dur = (pr.get("streams") or [{}])[0], float(pr["format"]["duration"])
    ts = pts_list(path)
    g = gaps(ts)
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 0
    res = {"file": path, "duration_s": round(dur, 2), "codec": st.get("codec_name"), "size": [st.get("width"), st.get("height")],
           "avg_frame_rate": st.get("avg_frame_rate"), "r_frame_rate": st.get("r_frame_rate"), "nb_frames": int(st.get("nb_frames") or len(ts)),
           "fps_from_timestamps": round((len(ts) - 1) / span, 2) if span else None,
           "longest_gap_ms": round(max(g) * 1000, 1) if g else None,
           "gaps_over_50ms": sum(1 for x in g if x > 0.050), "gaps_over_100ms": sum(1 for x in g if x > 0.100)}
    for name, vf in (("whole", "null"), ("avatar", "crop=trunc(iw*0.72/2)*2:ih:0:0")):
        k = kept_pts(path, vf)
        kg = gaps(k)
        res["distinct_" + name] = {"frames": len(k), "fps": round(len(k) / dur, 2) if dur else None,
                                   "longest_unchanged_ms": round(max(kg) * 1000, 1) if kg else None,
                                   "runs_over_250ms": sum(1 for x in kg if x > 0.25)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
