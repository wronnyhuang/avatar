"""Unified mode worker: the server's process for `?react=unified` (stdin/stdout frames from app/unified_link.py).

Runs under its OWN interpreter (config react.unified.python: numpy, onnxruntime or torch, scipy), never inside
app.py's event loop.  Per 30 fps tick j of the playback timeline (the page stamps its rig vectors with it):

  1. the user's camera: the newest RAW MediaPipe results the page sent ('U': face blendshapes + matrix, pose image +
     world landmarks, hand image + world landmarks + handedness), captured at or before the tick -> canon1.fit.Fitter
     (DATA agent) -> the user's CANON v1 frame (189) + validity (9).
  2. audio: the user's mic ('M') and the avatar's own voice tap ('P') -> the 42 causal features (react_worker.FeatStream =
     the spec's own); the avatar's voice is read AHEAD by up to the planner's self-audio lookahead, as far as the tap
     has delivered it (live, the tap leads the tick by ~0-2 frames; the rest of the window is the newest frame).
  3. unified.runtime.UnifiedPlanner.step(user, mask, user audio, own audio ahead) -> the avatar's canon frame.
  4. retarget (motion-model/retarget/): Retargeter(character) -> the character's rig params; Compositor: lip sync owns the
     mouth while the avatar talks (own-voice VAD), the page's commanded actions / reflexes (the learned-mode affine map:
     final = alpha * model + beta, i.e. ownership 1-alpha) own their rig channels and the canon groups behind them go
     to the planner as set_override / release_override.
  5. final rig vector = the page's rules vector with every channel the retarget covers replaced (blinks: the model's,
     floored by the page's reflex blink; react.unified.blinks = "rules" keeps the page's).

  server -> worker  {"op":"step", "character", "rig_params"?, "mp":[[kind,t]...], "mic":[[seq,rate]...], "own":[s0...],
                  "sync":[[seq,t]...], "ticks":[[t, n_ab, rb]...]} + blobs (raw 'U' messages | mic int16 | own int16
                  24 kHz | page rig float32 | ab float32 x3)
  worker -> server  {"id", "n", "stats"?} + blob float32 (n x len(rig_params)) [+ blob: the newest learned vector, NaN =
                  not covered]
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import react_worker as RW  # noqa: E402  (framing, AudioTrack, FeatStream: the same audio features as learned mode)
import wire  # noqa: E402

FPS = 30
TICK = 1.0 / FPS
STALE_S = 0.5


def setup_paths(roots, fit_root):
    """roots: dirs that hold canon1/, unified/, retarget/ (first wins per module); fit_root: a dir whose canon1/ has
    fit.py.  Each package's __path__ is extended with the
    same subdir of every other root, so canon1.fit / unified.runtime / retarget.* resolve wherever they live."""
    roots = [r for r in roots if r]
    for r in reversed(roots):
        if r not in sys.path:
            sys.path.insert(0, r)
    extra = roots + ([fit_root] if fit_root else [])
    for pkg in ("canon1", "unified", "retarget"):
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue
        for r in extra:
            d = os.path.join(r, pkg)
            if os.path.isdir(d) and d not in list(mod.__path__):
                mod.__path__.append(d)


class MiniFitter:
    """Fallback when canon1.fit is not importable: face (blendshapes pass through) + head from the face matrix only."""

    def __init__(self, spec):
        self.S = spec
        self.version = "mini"

    def reset(self):
        pass

    def step(self, fr):
        S = self.S
        m = np.zeros(S.D_MOTION, np.float32)
        k = np.zeros(S.D_MASK, np.uint8)
        if fr.get("face_present"):
            m[S.SL_FACE] = np.clip(np.asarray(fr["blendshapes"], np.float32), 0, 1)
            k[S.GROUP_NAMES.index("face_upper")] = k[S.GROUP_NAMES.index("mouth")] = 1
        return m, k


class Engine:
    def __init__(self, args):
        self.args = args
        setup_paths(args.roots.split(os.pathsep), args.fit_root)
        from canon1 import spec
        import canon1.causal_audio as _A

        class _Spec:                                   # spec + the audio constants FeatStream reads (N_FFT, _WIN, MEL_FB)
            def __getattr__(self, k):
                return getattr(spec, k) if hasattr(spec, k) else getattr(_A, k)
        self.spec = _Spec()
        try:
            Fitter = importlib.import_module("canon1.fit").Fitter
            self.fitter = Fitter(args.fit_params) if args.fit_params else Fitter()
            self.fit_name = "canon1.fit/" + str(getattr(self.fitter, "version", "?"))
        except Exception as e:  # noqa: BLE001
            print(f"unified: canon1.fit unavailable ({e}); MiniFitter", file=sys.stderr)
            self.fitter = MiniFitter(spec)
            self.fit_name = "mini"
        from unified.runtime import UnifiedPlanner
        kw = json.loads(args.planner_kw) if args.planner_kw else {}
        self.planner = UnifiedPlanner(args.bundle or None, threads=args.threads, **kw)
        self.L = int(getattr(self.planner, "self_audio_lookahead", 0))
        from retarget.retarget import Retargeter
        from retarget.compositor import Compositor
        self.Retargeter, self.Compositor = Retargeter, Compositor
        self.character = "muse"
        self.rig_params: list[str] = []
        self.blinks = args.blinks
        self.stats = collections.Counter()
        self.ms = {k: collections.deque(maxlen=3000) for k in ("fit", "plan", "rt", "tick", "step")}
        self.trace = open(args.trace, "a", buffering=1 << 16) if args.trace else None
        self.style: dict = {}              # {energy, mirroring, expressiveness} (None = infer)
        self.reset()

    # ------------------------------------------------------------------ session
    def reset(self):
        sp = self.spec
        self.planner.reset()
        try:
            self.fitter.reset()
        except Exception:  # noqa: BLE001
            pass
        self.rt = self.Retargeter(self.character if self.character in _known() else "muse")
        self.comp = self.Compositor(self.rt, planner=self.planner)
        dz = None if self.args.dither_db <= -200 else float(self.args.dither_db)
        self.mic = RW.AudioTrack(sp.SR, dither_db=dz)
        self.own24 = RW.AudioTrack(24000, dither_db=dz)
        self.mic_sync = None
        self.mic_pending = []
        self.fmic, self.fown = RW.FeatStream(sp), RW.FeatStream(sp)
        self.own_feats = {}                    # frame -> own features (computed once, in order)
        self.own_next = 0                      # next own frame to featurize
        self.mp = {wire.U_FACE: collections.deque(maxlen=90), wire.U_POSE: collections.deque(maxlen=60),
                   wire.U_HANDS: collections.deque(maxlen=60)}
        self.used = {wire.U_POSE: -1.0, wire.U_HANDS: -1.0}
        self.cur = None
        self.shown: dict[int, dict] = {}
        self.last_rig = None
        self.last_user_t = None
        self.lv = None

    def set_style(self, st: dict) -> None:
        """The session's style -> the planner's STYLE vector s (each in [0, 1] or null =
        infer). A planner without set_style (the model builder adds it) keeps it here and counts it."""
        if not isinstance(st, dict):
            return
        for k in ("energy", "mirroring", "expressiveness"):
            if k in st:
                v = st[k]
                self.style[k] = None if v is None else float(min(1.0, max(0.0, float(v))))
        fn = getattr(self.planner, "set_style", None)
        if callable(fn):
            fn(dict(self.style))
            self.stats["style_applied"] += 1
        else:
            self.stats["style_unsupported"] += 1

    def set_character(self, ch):
        if ch and ch != self.character:
            self.character = ch
            self.rt = self.Retargeter(ch if ch in _known() else "muse")
            self.comp = self.Compositor(self.rt, planner=self.planner)
            self.stats["character_switches"] += 1

    # ------------------------------------------------------------------ inputs
    def add_mp(self, blob):
        try:
            d = wire.unpack_mp(blob)
        except (ValueError, Exception):  # noqa: BLE001
            self.stats["bad_mp"] += 1
            return
        q = self.mp.get(d["kind"])
        if q is None:
            return
        # keep capture-time order (lanes may deliver out of order)
        if q and d["t"] < q[-1]["t"]:
            lst = sorted(list(q) + [d], key=lambda x: x["t"])
            q.clear(); q.extend(lst[-q.maxlen:])
        else:
            q.append(d)
        self.stats["mp_" + str(d["kind"])] += 1

    def add_sync(self, seq, t):
        self.mic_sync = (int(seq), float(t))
        for s, x in self.mic_pending:
            self._place_mic(s, x)
        self.mic_pending = []

    def _place_mic(self, seq, x16):
        s0, t0 = self.mic_sync
        t_end = t0 + (seq - s0) * 0.02
        self.mic.write(int(round(t_end * self.spec.SR)) - len(x16), x16)

    def add_mic(self, seq, pcm, rate):
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
        if rate != self.spec.SR:
            x = self.spec.resample_to_16k(x, rate)
        if self.mic_sync is None:
            self.mic_pending.append((seq, x)); self.mic_pending = self.mic_pending[-200:]
        else:
            self._place_mic(seq, x)

    def add_own(self, s0, pcm):
        self.own24.write(int(s0), np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0)

    # ------------------------------------------------------------------ per tick
    def mp_frame(self, j):
        """the fitter's input for tick j: newest face result captured by the tick's end (stale after 0.5 s); pose /
        hands only when a result newer than the one already used exists (else None / [] = 'not run', the fitter
        holds its last)."""
        te = (j + 1) * TICK + 1e-4
        fr = {"face_present": False, "blendshapes": np.zeros(52, np.float32), "face_matrix": np.eye(4).reshape(16),
              "pose_world": None, "pose_image": None, "hands": [], "image_wh": (320, 240)}
        newest = None
        q = self.mp[wire.U_FACE]
        f = next((d for d in reversed(q) if d["t"] <= te), None)
        if f is not None and te - f["t"] <= STALE_S and f["present"]:
            fr.update(face_present=True, blendshapes=f["bs"], face_matrix=f["mat"], image_wh=(f["w"], f["h"]))
            newest = f["t"]
        for kind in (wire.U_POSE, wire.U_HANDS):
            d = next((d for d in reversed(self.mp[kind]) if d["t"] <= te), None)
            if d is None or d["t"] <= self.used[kind] or te - d["t"] > STALE_S:
                continue
            self.used[kind] = d["t"]
            if kind == wire.U_POSE and d["present"]:
                fr.update(pose_world=d["world"], pose_image=d["image"])
                newest = max(newest or -1e9, d["t"])
            elif kind == wire.U_HANDS:
                fr["hands"] = [dict(h) for h in d.get("hands", [])]
                if fr["hands"]:
                    newest = max(newest or -1e9, d["t"])
            fr["image_wh"] = (d["w"], d["h"])
        if newest is not None:
            self.last_user_t = newest
        return fr

    def own_feat(self, k):
        """own-voice features of frame k, computed in frame order (the FeatStream is causal and stateful)."""
        while self.own_next <= k:
            self.own_feats[self.own_next] = self._own_frame(self.own_next)
            self.own_next += 1
        for old in [x for x in self.own_feats if x < k - 120]:
            del self.own_feats[old]
        return self.own_feats.get(k)

    def _own_frame(self, k):
        sp = self.spec
        e = sp.frame_end_sample(k)
        L = max(sp.N_FFT, self.fown.hop)
        m0 = (e - L - 24) & ~1
        e1 = (e + 1) & ~1
        x24 = self.own24.read(m0 * 3 // 2, e1 * 3 // 2 + 30)
        from scipy.signal import resample_poly
        y = resample_poly(x24, 2, 3, window=RW._H23.copy()).astype(np.float32)
        return self.fown.frame(y[e - L - m0:e - m0])

    def mic_feat(self, j):
        sp = self.spec
        e = sp.frame_end_sample(j)
        L = max(sp.N_FFT, self.fmic.hop)
        return self.fmic.frame(self.mic.read(e - L, e))

    def tick(self, j, page):
        t0 = time.perf_counter()
        sp = self.spec
        fr = self.mp_frame(j)
        t1 = time.perf_counter()
        um, uk = self.fitter.step(fr)
        if self.last_user_t is None or (j + 1) * TICK - self.last_user_t > 0.6:
            um, uk = None, None                    # nobody in view: the planner sees an all-zero mask
        t2 = time.perf_counter()
        ua = self.mic_feat(j)
        # own voice ahead: as far as the tap has delivered audio (the newest complete frame), up to L frames ahead
        avail = int(self.own24.head // 800) - 1
        k_ahead = max(j, min(j + self.L, avail))
        oa_ahead = self.own_feat(k_ahead)
        oa_now = self.own_feats.get(j) if j < self.own_next else self.own_feat(j)
        talking = bool(oa_now is not None and oa_now[41] > 0.5)
        t3 = time.perf_counter()
        out = self.planner.step(um, uk, ua, oa_ahead)
        t4 = time.perf_counter()
        names = self.rig_params
        action = self.Compositor.from_affine(names, page[1]) if page is not None else None
        rig = self.comp.step(out["motion"], talking=talking, action=action)
        t5 = time.perf_counter()
        self.ms["fit"].append((t2 - t1) * 1000)
        self.ms["plan"].append((t4 - t3) * 1000)
        self.ms["rt"].append((t5 - t4) * 1000)
        self.ms["tick"].append((t5 - t0) * 1000)
        if out.get("plan_ms") is not None:
            self.stats["plans"] += 1
        self.last_rig = rig
        self.shown[j] = rig
        for old in [x for x in self.shown if x < j - 90]:
            del self.shown[old]
        if page is None:
            return None
        return self.merge(rig, page)

    def merge(self, rig, page):
        a, ab, rb = page
        out = np.array(a, np.float32)
        lv = np.full(len(out), np.nan, np.float32)
        for i, p in enumerate(self.rig_params[:len(out)]):
            if p not in rig:
                continue
            if p in ("blink_l", "blink_r"):
                if self.blinks == "rules":
                    continue
                v = max(float(rig[p]), float(rb))
            else:
                v = float(rig[p])
            out[i] = v
            lv[i] = v
        self.lv = lv
        return out

    def step(self, ticks):
        t0 = time.perf_counter()
        finals = []
        for t, a, ab, rb in sorted(ticks, key=lambda x: x[0]):
            j = int(math.floor(t * FPS + 0.5))
            if self.cur is None or j > self.cur + 90 or j < self.cur - 90:
                if self.cur is not None:
                    self.stats["clock_resets"] += 1
                self.cur = j
            if j < self.cur:                                  # late (another lane's older batch): what was shown
                self.stats["late_ticks"] += 1
                rig = self.shown.get(j, self.last_rig)
                finals.append(self.merge(rig, (a, ab, rb)) if rig is not None else np.array(a, np.float32))
                continue
            while self.cur < j:                               # the page skipped ticks: step them anyway
                self.stats["gap_ticks"] += 1
                self.tick(self.cur, None)
                self.cur += 1
            finals.append(self.tick(j, (a, ab, rb)))
            self.cur = j + 1
            self.stats["ticks"] += 1
        self.ms["step"].append((time.perf_counter() - t0) * 1000)
        return finals

    def report(self):
        def pq(xs):
            if not xs:
                return None
            a = np.sort(np.asarray(xs))
            return {"n": int(len(a)), "p50": round(float(a[len(a) // 2]), 3),
                    "p95": round(float(a[int(0.95 * (len(a) - 1))]), 3), "max": round(float(a[-1]), 3)}
        return {"fit_ms": pq(self.ms["fit"]), "planner_step_ms": pq(self.ms["plan"]),
                "retarget_compositor_ms": pq(self.ms["rt"]), "tick_ms": pq(self.ms["tick"]),
                "batch_ms": pq(self.ms["step"]), **{k: int(v) for k, v in self.stats.items()},
                "fitter": self.fit_name, "planner": type(self.planner.core).__name__, "character": self.character,
                "self_lookahead": self.L, "blinks": self.blinks, "style": dict(self.style),
                "overrides": self.comp.stats().get("overrides")}


_KNOWN = None


def _known():
    global _KNOWN
    if _KNOWN is None:
        from retarget.retarget import CHARACTERS
        _KNOWN = set(CHARACTERS)
    return _KNOWN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, help="os.pathsep-separated dirs holding canon1/, unified/, retarget/")
    ap.add_argument("--fit-root", default="", help="dir whose canon1/ has fit.py")
    ap.add_argument("--fit-params", default="")
    ap.add_argument("--bundle", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--planner-kw", default="", help='JSON kwargs for UnifiedPlanner (replan_every, steps, guidance, ...)')
    ap.add_argument("--blinks", default="model", choices=["model", "rules"])
    ap.add_argument("--dither-db", type=float, default=-70.0)
    ap.add_argument("--trace", default="")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    eng = Engine(args)
    RW.write_frame(fout, {"op": "ready", "planner": type(eng.planner.core).__name__, "retarget": "retarget",
                          "fitter": eng.fit_name, "pid": os.getpid()})
    while True:
        try:
            hdr, blobs = RW.read_frame(fin)
        except EOFError:
            return
        op = hdr.get("op")
        rid = hdr.get("id")
        try:
            if op == "reset":
                eng.character = hdr.get("character", eng.character)
                if hdr.get("blinks") in ("model", "rules"):
                    eng.blinks = hdr["blinks"]
                eng.rig_params = list(hdr.get("rig_params") or eng.rig_params)
                eng.reset()
                RW.write_frame(fout, {"id": rid, "ok": True})
            elif op == "step":
                eng.set_character(hdr.get("character"))
                if "style" in hdr:                                  # set_style
                    eng.set_style(hdr["style"])
                if hdr.get("rig_params"):
                    eng.rig_params = list(hdr["rig_params"])
                bi = 0
                for _ in hdr.get("mp", []):
                    eng.add_mp(blobs[bi]); bi += 1
                for seq, t in hdr.get("sync", []):
                    eng.add_sync(seq, t)
                for seq, rate in hdr.get("mic", []):
                    eng.add_mic(int(seq), blobs[bi], int(rate)); bi += 1
                for s0 in hdr.get("own", []):
                    eng.add_own(int(s0), blobs[bi]); bi += 1
                ticks = []
                for t, nab, rb in hdr.get("ticks", []):
                    a = np.frombuffer(blobs[bi], np.float32); bi += 1
                    ab = np.frombuffer(blobs[bi], np.float32).reshape(-1, 3) if nab else np.zeros((0, 3), np.float32); bi += 1
                    ticks.append((float(t), a, [(int(r[0]), float(r[1]), float(r[2])) for r in ab], float(rb)))
                finals = eng.step(ticks) if ticks else []
                body = b"".join(np.asarray(f, np.float32).tobytes() for f in finals)
                out = {"id": rid, "n": len(finals)}
                blobs_out = [body]
                if eng.lv is not None and finals:
                    blobs_out.append(np.asarray(eng.lv, np.float32).tobytes())
                if hdr.get("want_stats"):
                    out["stats"] = eng.report()
                RW.write_frame(fout, out, blobs_out)
            elif op == "stats":
                RW.write_frame(fout, {"id": rid, "stats": eng.report()})
            else:
                RW.write_frame(fout, {"id": rid, "error": f"unknown op {op}"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            RW.write_frame(fout, {"id": rid, "error": str(e)[:300]})


if __name__ == "__main__":
    main()
