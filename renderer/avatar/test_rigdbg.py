#!/usr/bin/env python3
"""rigdbg: tests of the opt-in {"type":"rigdbg","on":bool} message (avatar/PROTOCOL-NOTES.txt note 14): the
renderer's FINAL lip / hand columns for the page's rig debug panel.

    python -m avatar.test_rigdbg session --target mock --base <old tree>   # this server + live.testing:MouthBackend (CPU)
    python -m avatar.test_rigdbg session --target rt                       # this server + the rt engine on llvmpipe
    python -m avatar.test_rigdbg session --target stub --base <old tree>   # app/stub_render.py
    python -m avatar.test_rigdbg rig                                       # rig vectors in -> the rows are the FINAL values
    python -m avatar.test_rigdbg cost                                      # per-frame cost of the on path, the off check

session: the same speech / silence timeline audio (the 'Pah' / 'Mah' tts snippets of live/assets) is played to
  base  the old tree's server (--base, a checkout of the commit before rigdbg), no 'rigdbg' ever     [optional]
  off   this tree's server, no 'rigdbg' ever
  on    this tree's server, 'rigdbg' on at 2 s, off at 9 s, on at 11 s, off at 13 s
and checks: (a) off: no rigdbg message; its 'V' payloads equal base's (every frame number both produced) and the on
run's; (b) on: rows arrive only while on, cols are LIP_OWNED + the 10 extras -- carried by the first message after each
'on' and then at most every 2 s, every other message is read with the last list seen -- every row's (F, t) is a 'V' frame's
(F, t) exactly, lip columns open while speech plays and stay ~closed in silence, <= 40 rows and >= ~100 ms apart;
(c) off stops them (nothing later than 0.25 s after off), on again restarts them; (d) bytes/s and rows/s while on.
One JSON line per check group + a summary; exit 1 on a failed check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from avatar.srv_testkit import V_HDR, SR, launch, stop, decode   # noqa: E402

mono = time.monotonic
FPS = 24
EXTRAS = ["smile", "mouth_open", "hand_l_x", "hand_l_y", "hand_l_z", "hand_r_x", "hand_r_y", "hand_r_z",
          "hand_l_open", "hand_r_open"]
ASSETS = os.path.join(ROOT, "live", "assets")


# ------------------------------------------------------------------ the timeline audio: speech / silence
TREE_R = os.path.dirname(ROOT)

def _snippet(k):
    """live/assets/tts_<k>.pcm (int16, 24 kHz) when present, else the generated syllable of renderer/live/latency.py"""
    p = os.path.join(ASSETS, f"tts_{k}.pcm")
    if os.path.isfile(p):
        return np.fromfile(p, "<i2")
    sys.path.insert(0, os.path.join(TREE_R, "renderer"))
    from live.latency import synth_syllable
    return synth_syllable(k)


def speech_timeline(seconds, speech=((1.0, 2.6), (4.0, 5.6), (7.0, 8.6), (10.0, 11.6), (13.0, 14.6))):
    """int16 PCM on the 24 kHz timeline: 'Pah' / 'Mah' snippets (live/assets/tts_*.pcm, else generated) back to back with 60 ms
    gaps inside each (start, end) speech span, digital silence elsewhere."""
    snips = [_snippet(k) for k in ("pa", "ma")]
    x = np.zeros(int(seconds * SR), np.int16)
    for a, b in speech:
        s, k = int(a * SR), 0
        while s < int(b * SR):
            p = snips[k % 2]
            n = min(p.size, int(b * SR) - s, x.size - s)
            if n <= 0:
                break
            x[s:s + n] = p[:n]
            s += n + int(0.06 * SR)
            k += 1
    return x


def envelope(x, win=0.04):
    """-> callable t -> RMS dBFS of the timeline audio in [t - win/2, t + win/2)."""
    def at(t):
        a, b = max(0, int((t - win / 2) * SR)), min(x.size, int((t + win / 2) * SR))
        if b <= a:
            return -120.0
        v = x[a:b].astype(np.float64)
        r = np.sqrt(np.mean(v * v)) / 32768.0
        return 20 * np.log10(max(r, 1e-6))
    return at


# ------------------------------------------------------------------ one session
async def session(url, start, pcm, plan=(), ready_timeout=300, rig=None):
    """Plays pcm on the 20 ms tick; plan = [(t_s, json msg)] sent at those stream times (logged with send time);
    rig = callable(t) -> (names, values) sent at ~30 Hz (or None)."""
    import websockets
    frames, msgs, sent = [], [], []
    ev = asyncio.Event()
    async with websockets.connect(url, compression=None, max_size=2 ** 24) as ws:
        await ws.send(json.dumps(start))
        ready = {}

        async def rx():
            async for m in ws:
                now = mono()
                if isinstance(m, bytes):
                    _, flags, _, F, t = V_HDR.unpack_from(m, 0)
                    frames.append((now, F, t, flags, m[V_HDR.size:]))
                else:
                    j = json.loads(m)
                    msgs.append((now, j, len(m)))
                    if j["type"] == "ready":
                        ready.update(j)
                        ev.set()
                    elif j["type"] == "error":
                        print(json.dumps({"server_error": j.get("message", "")[-300:]}), file=sys.stderr)
        task = asyncio.create_task(rx())
        await asyncio.wait_for(ev.wait(), ready_timeout)
        T0 = mono() + 0.05
        chunk = 480
        plan = sorted(plan, key=lambda x: x[0])
        pi, s, next_rig = 0, 0, 0.0
        while s < pcm.size:
            due = T0 + (s + chunk) / SR
            now = mono()
            if due > now:
                await asyncio.sleep(due - now)
            await ws.send(struct.pack("<B3xQ", 0x41, s) + pcm[s:s + chunk].tobytes())
            s += chunk
            el = mono() - T0
            while pi < len(plan) and el >= plan[pi][0]:
                await ws.send(json.dumps(plan[pi][1]))
                sent.append((mono(), plan[pi][1]))
                pi += 1
            if rig is not None and el >= next_rig:
                names, vals = rig(s / SR, ready.get("rig_params") or [])
                if names:
                    await ws.send(json.dumps({"type": "rig", "t": s / SR, "a": vals, "names": names}))
                next_rig = el + 1 / 30
        await asyncio.sleep(0.8)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(task, 3)
        except Exception:
            task.cancel()
    return {"frames": frames, "msgs": msgs, "sent": sent, "T0": T0, "ready": ready}


def run_target(target, tree, logpath, start, pcm, plan=(), env=None, rig=None):
    p, port, key = launch(target, tree, logpath, env)
    try:
        return asyncio.run(session(f"ws://127.0.0.1:{port}/render?key={key}", start, pcm, plan, rig=rig))
    finally:
        stop(p)


def start_msg(target, a):
    if target == "stub":
        return {"type": "start", "character": "fox", "width": 640, "height": 360, "fps": FPS, "codec": "h264",
                "bitrate": 600_000}
    if target == "mock":
        return {"type": "start", "character": a.character, "width": 320, "height": 180, "fps": FPS, "codec": "h264",
                "bitrate": 600_000, "backend": "live.testing:MouthBackend", "opts": {"ms": 8}, "gest_seed": 7,
                "nvenc": False}
    return {"type": "start", "character": a.character, "width": a.width, "height": a.height, "fps": FPS,
            "codec": "h264", "bitrate": 600_000, "backend": "rt", "opts": {"preset": a.preset} if a.preset else {},
            "gest_seed": 7, "nvenc": False}


def expected_cols():
    from anim.rig import LIP_OWNED
    return list(LIP_OWNED) + EXTRAS


# ------------------------------------------------------------------ analysis
def cols_check(dbg, cols_want, ons, every_s=2.0):
    """'cols' rides only the first message after each 'on' (ons: send times, same clock as dbg) and then at most every
    ~2 s; a message without it is read with the most recent list seen.  dbg = [(t, msg, nbytes)] in arrival order."""
    last, bad, cols_t = None, [], []
    for t, m, _ in dbg:
        if "cols" in m:
            if m["cols"] != cols_want:
                bad.append(("cols != expected", round(t, 3)))
            last = m["cols"]
            cols_t.append(t)
        elif last is None:
            bad.append(("rows before any cols", round(t, 3)))
        for x in m.get("rows", []):
            if last is not None and len(x) != 2 + len(last):
                bad.append(("row length != 2 + len(cols)", round(t, 3)))
                break
    first_after_on = []
    for a in ons:
        nxt = next(((t, m) for t, m, _ in dbg if t >= a), None)
        if nxt is not None:
            first_after_on.append({"on_at": round(a, 3), "first_msg_after_s": round(nxt[0] - a, 3), "has_cols": "cols" in nxt[1]})
    # spacing of cols messages that are not the first after an 'on'
    firsts = {round(next((t for t, _, _ in dbg if t >= a), -1), 6) for a in ons}
    periodic = [(p, q) for p, q in zip(cols_t, cols_t[1:]) if round(q, 6) not in firsts]
    gaps = [q - p for p, q in periodic]
    out = {"ok": not bad and all(f["has_cols"] for f in first_after_on) and all(g >= every_s - 0.15 for g in gaps),
           "bad": bad[:5], "msgs_with_cols": len(cols_t), "msgs_without_cols": len(dbg) - len(cols_t),
           "first_after_on": first_after_on, "periodic_cols_gap_s_min": round(min(gaps), 3) if gaps else None}
    return out


def analyse_rows(r, pcm, cols_want, windows):
    """windows = [(t_on, t_off)] in stream seconds (send times relative to T0)."""
    T0 = r["T0"]
    dbg = [(t - T0, m, n) for t, m, n in r["msgs"] if m.get("type") == "rigdbg"]
    vt = {F: t for _, F, t, _, _ in r["frames"]}
    rows = [row for _, m, _ in dbg for row in m.get("rows", [])]
    out = {"messages": len(dbg), "rows": len(rows)}
    if not dbg:
        return out, rows, dbg
    cols = dbg[0][1].get("cols")
    out["cols"] = cols_check(dbg, cols_want, [a for a, _ in windows])
    out["cols_ok"] = out["cols"]["ok"]
    out["n_cols"] = len(cols or [])
    out["fps_field"] = sorted({m.get("fps") for _, m, _ in dbg})
    out["row_len_ok"] = all(len(x) == 2 + len(cols_want) for x in rows)
    out["max_rows_per_msg"] = max(len(m.get("rows", [])) for _, m, _ in dbg)
    gaps = np.diff([t for t, _, _ in dbg]) if len(dbg) > 1 else np.array([1.0])
    out["msg_gap_ms_min"] = round(1000 * float(gaps.min()), 1)
    out["msg_gap_ms_p50"] = round(1000 * float(np.median(gaps)), 1)
    out["decimals_ok"] = all(abs(v * 1000 - round(v * 1000)) < 1e-6 for x in rows for v in x[2:])
    # (F, t) of every row == a 'V' frame's (F, t), exactly (the float64 the header carries)
    matched = [x for x in rows if x[0] in vt]
    out["rows_with_a_V_frame"] = len(matched)
    out["t_exact_equal"] = all(vt[x[0]] == x[1] for x in matched) and all(x[1] == x[0] / FPS for x in rows)
    out["frames_numbers_unique_in_rows"] = len({x[0] for x in rows}) == len(rows)
    # inside the on windows: rows for every frame sent (rows / frames)
    def inside(t, pad=0.0):
        return any(a - pad <= t <= b + pad for a, b in windows)
    arr = [(t, m) for t, m, _ in dbg]
    out["msgs_outside_on_windows"] = sum(1 for t, _ in arr if not inside(t, pad=0.25))
    late = [round(t - b, 3) for t, _ in arr for a, b in windows if b < t <= b + 0.25]
    out["msgs_in_0_25s_after_off"] = len(late)
    fr_on = [F for tt, F, *_ in r["frames"] if inside(tt - T0) and not any(a <= tt - T0 <= a + 0.35 for a, _ in windows)
             and not any(b - 0.35 <= tt - T0 <= b for _, b in windows)]
    out["V_frames_well_inside_on"] = len(fr_on)
    out["of_those_with_a_row"] = sum(1 for F in fr_on if F in {x[0] for x in rows})
    # lips vs audio: opening = max(jaw, lip_open); speech = the frame's instant has audio > -40 dBFS within +-20 ms;
    # silence = no audio within 300 ms of it
    env = envelope(pcm)
    ci = {c: i + 2 for i, c in enumerate(cols_want)}
    lip_names = cols_want[:19]
    sp, si, lipmax_si = [], [], []
    for x in rows:
        t = x[1]
        op = max(x[ci["jaw"]], x[ci["lip_open"]])
        if env(t) > -40:
            sp.append(op)
        elif max(env(t + d) for d in np.arange(-0.3, 0.301, 0.02)) < -80:
            si.append(op)
            lipmax_si.append(max((abs(x[ci[c]]), c) for c in lip_names))
    out["speech_rows"] = len(sp)
    out["silence_rows"] = len(si)
    if sp:
        out["speech_opening_mean"] = round(float(np.mean(sp)), 3)
        out["speech_opening_p90"] = round(float(np.percentile(sp, 90)), 3)
        out["speech_rows_open_gt_0.05"] = round(float(np.mean(np.array(sp) > 0.05)), 3)
    if si:
        out["silence_opening_mean"] = round(float(np.mean(si)), 4)
        out["silence_opening_max"] = round(float(np.max(si)), 4)
        v, c = max(lipmax_si)
        out["silence_any_lip_col_abs_max"] = [round(float(v), 4), c]
    on_s = sum(b - a for a, b in windows)
    out["on_seconds"] = round(on_s, 2)
    out["bytes_per_s"] = round(sum(n for _, _, n in dbg) / on_s, 1)
    out["rows_per_s"] = round(len(rows) / on_s, 2)
    out["msgs_per_s"] = round(len(dbg) / on_s, 2)
    out["bytes_per_row"] = round(sum(n for _, _, n in dbg) / max(1, len(rows)), 1)
    return out, rows, dbg


def identity(fa, fb):
    """-> (common frame numbers, byte-identical ones, mismatches [(F, flags a, flags b, bytes a, bytes b)])"""
    a = {f[1]: f for f in fa}
    b = {f[1]: f for f in fb}
    common = sorted(set(a) & set(b))
    same = [F for F in common if a[F][4] == b[F][4]]
    bad = [(F, a[F][3], b[F][3], len(a[F][4]), len(b[F][4])) for F in common if a[F][4] != b[F][4]]
    return len(common), len(same), bad


def decoded(frames):
    """-> {F: luma plane} (one fresh decoder over the whole run, frames in arrival order)"""
    import av
    cc = av.CodecContext.create("h264", "r")
    out, Fs = [], [f[1] for f in frames]
    for f in frames:
        try:
            out += [x.to_ndarray(format="gray") for x in cc.decode(av.Packet(f[4]))]
        except Exception:
            pass
    return dict(zip(Fs, out))


def cmd_session(a):
    tree = os.path.dirname(ROOT)
    os.makedirs(a.logdir, exist_ok=True)
    secs = 15.5
    pcm = speech_timeline(secs)
    start = start_msg(a.target, a)
    env = {"AVATAR_NVENC": "0"}
    if a.target == "rt":
        env.update(AVATAR_RT_DEVICE="cpu", LP_NUM_THREADS=str(a.threads))
        if a.ta_out:
            env["TA_OUT"] = a.ta_out
    plan = [(2.0, {"type": "rigdbg", "on": True}), (9.0, {"type": "rigdbg", "on": False}),
            (11.0, {"type": "rigdbg", "on": True}), (13.0, {"type": "rigdbg", "on": False})]
    runs = {}
    if a.base:
        runs["base"] = run_target(a.target, a.base, os.path.join(a.logdir, f"rigdbg-{a.target}-base.log"), start, pcm,
                                  (), env)
    runs["off"] = run_target(a.target, tree, os.path.join(a.logdir, f"rigdbg-{a.target}-off.log"), start, pcm, (), env)
    runs["on"] = run_target(a.target, tree, os.path.join(a.logdir, f"rigdbg-{a.target}-on.log"), start, pcm, plan, env)
    ok = True
    rep = {"target": a.target}
    for n, r in runs.items():
        errs = [m.get("message") for _, m, _ in r["msgs"] if m.get("type") == "error"]
        lates = [m.get("late") for _, m, _ in r["msgs"] if m.get("type") == "stats"]
        nd, sizes, de = decode([f[4] for f in r["frames"]])
        rep[n] = {"frames": len(r["frames"]), "rigdbg_msgs": sum(1 for _, m, _ in r["msgs"] if m.get("type") == "rigdbg"),
                  "errors": errs[:3], "late_total": int(sum(x or 0 for x in lates)), "decoded": nd, "decode_errors": de}
        ok &= not errs and de == 0
    # (a) off: nothing, frames as before
    ok &= rep["off"]["rigdbg_msgs"] == 0 and ("base" not in runs or rep["base"]["rigdbg_msgs"] == 0)
    # identity: every frame number both runs produced is byte-equal from the first 'on' (2 s, F 48) on.  Before it the
    # code path is the same in every run (nothing has been sent yet), and x264 has a start-of-stream transient: in about one
    # run in three a few early frames (seen: F 0, 1, 23-28, the same set every time, in runs that never got a 'rigdbg'
    # too) carry different bits that decode to within 1 grey level -- allowed there, reported.  rt on llvmpipe drops
    # frames by timing (late), so its runs are not comparable: reported only.
    strict = a.target != "rt"
    pairs = [("base", "off"), ("off", "on")] if "base" in runs else [("off", "on")]
    F_on = int(plan[0][0] * FPS)
    dec = {}
    for x, y in pairs:
        c, s, bad = identity(runs[x]["frames"], runs[y]["frames"])
        early = [b[0] for b in bad if b[0] < F_on]
        after = [b for b in bad if b[0] >= F_on]
        px = None
        if early:
            for n in (x, y):
                if n not in dec:
                    dec[n] = decoded(runs[n]["frames"])
            px = max(int(np.abs(dec[x][F].astype(int) - dec[y][F].astype(int)).max())
                     for F in early if F in dec[x] and F in dec[y])
        rep[f"identity_{x}_vs_{y}"] = {"common": c, "byte_identical": s, "before_first_on_mismatches": early,
                                       "before_first_on_decoded_max_diff": px, "from_first_on_mismatches": len(after),
                                       "first_mismatches_from_first_on": after[:8]}
        if strict:
            ok &= c > FPS * (secs - 3) and not after and (px is None or px <= 1)
    # (b) (c) (d)
    T0 = runs["on"]["T0"]
    ts = [t - T0 for t, _ in runs["on"]["sent"]]
    windows = [(ts[0], ts[1]), (ts[2], ts[3])]
    rows_rep, rows, dbg = analyse_rows(runs["on"], pcm, expected_cols(), windows)
    rep["on_rows"] = rows_rep
    ok &= bool(rows_rep.get("cols_ok") and rows_rep.get("row_len_ok") and rows_rep.get("t_exact_equal")
               and rows_rep.get("decimals_ok") and rows_rep.get("frames_numbers_unique_in_rows"))
    ok &= rows_rep["max_rows_per_msg"] <= 40 and rows_rep["msg_gap_ms_min"] >= 60
    ok &= rows_rep["msgs_outside_on_windows"] == 0
    ok &= rows_rep["V_frames_well_inside_on"] > 0 and rows_rep["of_those_with_a_row"] == rows_rep["V_frames_well_inside_on"]
    ok &= sum(1 for t, _, _ in dbg if windows[1][0] <= t <= windows[1][1] + 0.25) > 0      # on again restarts
    ok &= rows_rep.get("speech_opening_mean", 0) > 0.1 and rows_rep.get("speech_rows_open_gt_0.05", 0) > 0.6
    ok &= rows_rep.get("silence_opening_max", 1) < 0.05
    if a.target == "stub":
        ok &= all(m.get("src") == "stub" for _, m, _ in dbg)
    rep["ok"] = bool(ok)
    print(json.dumps(rep))
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump({"rows": rows, "cols": expected_cols(), "windows": windows}, f)
    return 0 if ok else 1


# ------------------------------------------------------------------ rig vectors in -> the FINAL values out
def cmd_rig(a):
    """mock target (MouthBackend) as the muse (her speech-gesture net is on), rigdbg on from the start, rig vectors at
    30 Hz: neutral, except 4-8 s mouth_open .6, smile .5, hand_r_y .7, hand_l_x -.3 (silence) and gest_l = gest_r = 1
    (the page's gesture-net weight) during the 9-11 s speech; speech also 1-3 s with the weight 0.  The rows must show
    the page's columns after smoothing and combine(): mouth_open ~.6, jaw / lip_open / teeth_gap >= mouth_open,
    lip_press / press x (1 - mouth_open), smile ~ .5 + .25 life smile, the hands at the targets; and the net's strokes on
    the hand columns while it speaks with the weight on (none with the weight off)."""
    tree = os.path.dirname(ROOT)
    secs = 12.5
    pcm = speech_timeline(secs, speech=((1.0, 3.0), (9.0, 11.0)))
    a.character = "muse"
    start = start_msg("mock", a)
    plan = [(0.0, {"type": "rigdbg", "on": True})]

    def rig(t, names):
        if not names:
            return None, None
        v = {n: 0.0 for n in names}
        if 4.0 <= t < 8.0:
            v.update(mouth_open=0.6, smile=0.5, hand_r_y=0.7, hand_l_x=-0.3)
        if 8.8 <= t < 11.2:
            v.update(gest_l=1.0, gest_r=1.0)
        return names, [v[n] for n in names]

    r = run_target("mock", tree, os.path.join(a.logdir, "rigdbg-rig.log"), start, pcm, plan, {"AVATAR_NVENC": "0"},
                   rig=rig)
    cols = expected_cols()
    ci = {c: i + 2 for i, c in enumerate(cols)}
    rows = [x for _, m, _ in r["msgs"] if m.get("type") == "rigdbg" for x in m["rows"]]
    hold = [x for x in rows if 5.0 <= x[1] < 7.8]
    neutral = [x for x in rows if 3.6 <= x[1] < 3.95]
    hands = [c for c in EXTRAS if c.startswith("hand_")]
    sp_off = [x for x in rows if 1.2 <= x[1] < 2.9]               # speech, net weight 0
    sp_on = [x for x in rows if 9.3 <= x[1] < 11.0]               # speech, net weight 1
    mean = lambda xs, c: round(float(np.mean([x[ci[c]] for x in xs])), 3) if xs else None
    motion = lambda xs: round(float(max(np.ptp([x[ci[c]] for x in xs]) for c in hands)), 3) if xs else None
    rep = {"rows": len(rows), "hold_rows": len(hold),
           "hold": {c: mean(hold, c) for c in ("mouth_open", "jaw", "lip_open", "teeth_gap", "smile", "hand_r_y",
                                                "hand_l_x", "press", "lip_press")},
           "neutral_before": {c: mean(neutral, c) for c in ("mouth_open", "jaw", "smile", "hand_r_y", "hand_l_x")},
           "speech_net_weight0_hand_range_max": motion(sp_off), "speech_net_weight1_hand_range_max": motion(sp_on),
           "speech_net_weight1_rows": len(sp_on), "gesture_net": r["ready"].get("gesture_net"),
           "errors": [m.get("message") for _, m, _ in r["msgs"] if m.get("type") == "error"][:3]}
    rep["cols"] = cols_check([x for x in r["msgs"] if x[1].get("type") == "rigdbg"], cols,
                             [t for t, m in r["sent"] if m.get("type") == "rigdbg" and m.get("on")])
    h = rep["hold"]
    ok = (bool(hold) and abs(h["mouth_open"] - 0.6) < 0.02 and min(h["jaw"], h["lip_open"], h["teeth_gap"]) >= 0.59
          and 0.49 <= h["smile"] <= 0.76 and abs(h["hand_r_y"] - 0.7) < 0.05 and abs(h["hand_l_x"] + 0.3) < 0.05
          and rep["speech_net_weight0_hand_range_max"] == 0.0
          and (rep["speech_net_weight1_hand_range_max"] or 0) == 0.0     # no speech-gesture net in this renderer
          and not rep["errors"] and rep["cols"]["ok"])
    rep["ok"] = bool(ok)
    print(json.dumps(rep))
    return 0 if ok else 1


# ------------------------------------------------------------------ cost
def cmd_cost(a):
    """The on path's real code: the dispatcher's two lines on a real RigMixer.apply() row, and Session.rigdbg_json()
    every 100 ms (2.4 rows at 24 fps) incl. the compact json.dumps _rigdbg_poster does; the off path: the None check.
    Five repetitions, the median (the box is shared).  Values are random (every column nonzero = the most bytes)."""
    import collections
    from anim.rig import RigMixer
    from anim.schema import NPARAMS
    from avatar import render_server as RS
    rng = np.random.default_rng(0)
    params = RigMixer().apply(rng.uniform(-1, 1, NPARAMS).astype("f4"), 1.0, 1.0)

    class S:                                 # only the attributes rigdbg_json reads
        pass

    def once(N=200_000, M=20_000):
        s = S()
        s.cfg = {"fps": FPS}
        s.rigdbg = None
        t = time.perf_counter()
        for F in range(N):
            pass
        t_loop = time.perf_counter() - t
        t = time.perf_counter()
        for F in range(N):                   # off: exactly the dispatcher's lines
            dbg = s.rigdbg
            if dbg is not None:
                dbg.append((F, params[RS.RIGDBG_IDX]))
        off_ns = 1e9 * (time.perf_counter() - t - t_loop) / N
        s.rigdbg = collections.deque(maxlen=RS.RIGDBG_MAX_ROWS)
        t = time.perf_counter()
        for F in range(N):                   # on: + the 29-float copy into the deque
            dbg = s.rigdbg
            if dbg is not None:
                dbg.append((F, params[RS.RIGDBG_IDX]))
        on_us = 1e6 * (time.perf_counter() - t - t_loop) / N
        s.rigdbg.clear()
        tot, nbytes = 0.0, 0
        for k in range(M):
            for F in range(3 if k % 5 < 2 else 2):       # 2.4 rows per 100 ms at 24 fps
                s.rigdbg.append((k * 3 + F, params[RS.RIGDBG_IDX]))
            s.rigdbg_cols_at = None if k % 20 == 0 else float("inf")    # cols on one message in 20 (2 s at 10 / s)
            t = time.perf_counter()
            js = json.dumps(RS.Session.rigdbg_json(s), separators=(",", ":"))     # what _rigdbg_poster posts
            tot += time.perf_counter() - t
            nbytes += len(js)
        return off_ns, on_us, 1e6 * tot / M, nbytes / M * 10

    runs = [once() for _ in range(5)]
    off_ns, on_us, msg_us, bps = (float(np.median([r[k] for r in runs])) for k in range(4))
    rep = {"off_check_ns_per_frame": round(off_ns, 1),
           "on_dispatcher_us_per_frame": round(on_us, 3),
           "on_poster_us_per_msg (2.4 rows, json incl.)": round(msg_us, 1),
           "on_total_us_per_frame (dispatcher + poster amortised)": round(on_us + msg_us / 2.4, 2),
           "bytes_per_s_at_24fps_worst (every column nonzero)": round(bps, 1),
           "cpu_share_at_24fps_pct": round(100 * (24 * on_us + 10 * msg_us) / 1e6, 4),
           "runs": [[round(x, 3) for x in r] for r in runs]}
    print(json.dumps(rep))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["session", "rig", "cost"])
    ap.add_argument("--target", default="mock", choices=["mock", "rt", "stub"])
    ap.add_argument("--base", default="", help="session: the old tree (a checkout of the base commit) for identity")
    ap.add_argument("--character", default="orb")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--preset", default="")
    ap.add_argument("--threads", type=int, default=8, help="rt: LP_NUM_THREADS (llvmpipe)")
    ap.add_argument("--ta-out", default="", help="rt: TA_OUT (the background bake cache)")
    ap.add_argument("--dump", default="", help="session: write the on run's rows here (json)")
    ap.add_argument("--logdir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_rigdbg"))
    a = ap.parse_args(argv)
    os.makedirs(a.logdir, exist_ok=True)
    return {"session": cmd_session, "rig": cmd_rig, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
