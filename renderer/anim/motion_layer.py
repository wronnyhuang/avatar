"""live: the MOTION LAYER — real-human body motion for Juno (juno / junor) inside the render server.

Numpy only.  A broken layer never stops a session: every entry point catches, logs once and turns the layer off (the
session then renders exactly as before: no canon goes to the rig).

Sources (env AVATAR_MOTION2, default "1" = on for juno / junor with the conversational generator):
    "0" / "off"          never constructed (byte-identical old behaviour)
    "1" / "mm"           anim/motion2/mm (T/approaches/mm/live_pkg: Generator(model_dir, seed); push_audio(42-d @30 fps),
                         set_state(speaking, partner_speaking), step() -> (189,) canon)
    "replay:<file.npy>"  a canon (T,189) clip looped (smoke test)
    "neutral"            the canon neutral pose (wiring test)
Commanded clips: anim/motion2/act (T/approaches/act/live_pkg: ClipPlayer(pkg_dir) .play(slug, mirror=) / .step(base) ->
(pose, {group: w})).

Threads (render_server.Session):
    anim thread      push_audio(s0, pcm)  (the PCM the lips get, 24 kHz timeline)   sample(t) -> body pose at t or None
    dispatch thread  frame(pose, t, gest_l, gest_r) -> (canon, canon_w, modes, scale_rig_l, scale_rig_r) or None
    ws thread        control(msg) ({type:"motion2", on, partner_speaking, body_busy}), clip(msg) ({type:"clip", slug, mirror})

Weights (anatomical canon sides; the rig's `_l` = SCREEN-left = anatomical RIGHT):
    arm_r / fingers_r = min(gest_l / GREF, 1)   arm_l / fingers_l = min(gest_r / GREF, 1)   (the page's per-side gesture
        weight already drops to 0 for a held prop, chalk, a commanded dial move ...; GREF = the page's default knob 0.5)
    a playing clip's owned groups: the clip's own envelope (max with the above)
    torso / head: "add" mode at TORSO_W, 0 while the page reports a whole-body move (body_busy)
    every weight slewed at <= SLEW per second; the old gesture net's offsets on an arm scale by (1 - that arm's weight).
"""
from __future__ import annotations

import collections
import copy
import importlib.util
import json
import os
import sys
import threading
import time

import numpy as np

from .motion2.audiofeat import StreamFeat

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(HERE, "motion2")
CHARS = ("juno", "junor")
GROUPS = ("torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r")
SLEW = float(os.environ.get("AVATAR_MOTION2_SLEW", "3.0"))          # weight units per second (<= 4/s)
GREF = float(os.environ.get("AVATAR_MOTION2_GREF", "0.5"))          # page gesture weight that means "arm fully free"
TORSO_W = float(os.environ.get("AVATAR_MOTION2_TORSO", "1.0"))
HEAD_W = float(os.environ.get("AVATAR_MOTION2_HEAD", "1.0"))
ARM_W = float(os.environ.get("AVATAR_MOTION2_ARMS", "1.0"))
LOOKAHEAD_FR = 3                                                     # frames of VAD look-ahead for "speaking"
_NEUTRAL = None


def log(*a):
    print(time.strftime("%H:%M:%S"), "[motion2]", *a, file=sys.stderr, flush=True)


def neutral():
    global _NEUTRAL
    if _NEUTRAL is None:
        _NEUTRAL = np.load(os.path.join(M2, "canon_neutral.npy")).astype(np.float32)
    return _NEUTRAL.copy()


def env_mode():
    return os.environ.get("AVATAR_MOTION2", "1").strip()


def src_kind():
    """AVATAR_MOTION2_SRC = mm (default, lead final 06:05) | hybrid (flow while she speaks, MM otherwise; option) | flow."""
    k = os.environ.get("AVATAR_MOTION2_SRC", "mm").strip().lower()
    return k if k in ("mm", "flow", "hybrid") else "mm"


def wanted(character):
    m = env_mode().lower()
    return character in CHARS and m not in ("0", "off", "false", "no", "")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    d = os.path.dirname(path)
    sys.path.insert(0, d)            # the package's own sibling imports
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
    return mod


# ------------------------------------------------------------------------------------------------ sources
class ReplayGen:
    def __init__(self, path, seed=0):
        m = np.load(path).astype(np.float32)
        if m.ndim != 2 or m.shape[1] < 135:
            raise ValueError(f"replay {path}: shape {m.shape}")
        if m.shape[1] < 189:
            m = np.concatenate([m, np.repeat(neutral()[None, m.shape[1]:], len(m), 0)], 1)
        self.m = m
        self.i = 0
        self.info = {"name": "replay:" + os.path.basename(path), "frames": int(len(m))}

    def reset(self):
        self.i = 0

    def push_audio(self, feats):
        pass

    def set_state(self, speaking, partner_speaking):
        pass

    def step(self):
        p = self.m[self.i % len(self.m)]
        self.i += 1
        return p


class NeutralGen(ReplayGen):
    def __init__(self, seed=0):
        self.m = neutral()[None]
        self.i = 0
        self.info = {"name": "neutral"}


# process-wide templates: the MM database (~25 MB npz, 2-3 s to load on a busy box) and the clip library are loaded ONCE
# per render server (prewarm() at import, in a thread) and every session gets a deep copy (its own state, fresh seed), so a
# session start / character switch never waits on them.
_TPL = {}
_TPL_LOCK = threading.Lock()


def _mm_pkg():
    return os.environ.get("AVATAR_MOTION2_MM_DIR") or os.path.join(M2, "mm")      # env: dev override (a package folder)


def _flow_pkg():
    return os.environ.get("AVATAR_MOTION2_FLOW_DIR") or os.path.join(M2, "flow")


def _act_pkg():
    return os.environ.get("AVATAR_MOTION2_ACT_DIR") or os.path.join(M2, "act")


def _template(kind):
    with _TPL_LOCK:                       # one loader at a time; later callers get the cached object
        if kind in _TPL:
            return _TPL[kind]
        t0 = time.perf_counter()
        if kind == "mm":
            pkg = _mm_pkg()
            if not os.path.isfile(os.path.join(pkg, "stream.py")):
                raise FileNotFoundError(f"no conversational package at {pkg}")
            mod = _load_module("t4693_motion2_mm_stream", os.path.join(pkg, "stream.py"))
            obj = mod.Generator(pkg, 0)
        elif kind == "flow":
            pkg = _flow_pkg()
            if not os.path.isfile(os.path.join(pkg, "stream.py")):
                raise FileNotFoundError(f"no flow package at {pkg}")
            mod = _load_module("t4693_motion2_flow_stream", os.path.join(pkg, "stream.py"))
            obj = mod.Generator(pkg, 0)
        else:
            pkg = _act_pkg()
            if not os.path.isfile(os.path.join(pkg, "player.py")) or os.environ.get("AVATAR_MOTION2_CLIPS", "1") == "0":
                obj = None
            else:
                mod = _load_module("t4693_motion2_act_player", os.path.join(pkg, "player.py"))
                obj = mod.ClipPlayer(pkg)
        _TPL[kind] = obj
        log(f"loaded {kind} in {time.perf_counter() - t0:.2f} s")
        return obj


def prewarm():
    """Load the MM database + clip library in a background thread (render server import time)."""
    if env_mode().lower() in ("0", "off", "false", "no", "") or env_mode().lower().startswith(("replay:", "neutral")):
        return

    def run():
        sk = src_kind()                  # only what the configured source needs (flow never loads unless selected)
        proc = os.environ.get("AVATAR_MOTION2_FLOW_PROC", "1") not in ("0", "sync")     # a flow process loads its own copy
        for k in (("mm",) if sk == "mm" else () if sk == "flow" and proc else ("flow",) if sk == "flow"
                  else ("mm",) if proc else ("mm", "flow")) + ("act",):
            try:
                _template(k)
            except Exception as e:
                log(f"prewarm {k}: {type(e).__name__}: {e}")
    threading.Thread(target=run, name="motion2-prewarm", daemon=True).start()


def make_source(mode, seed, kind=None):
    ml = mode.lower()
    if ml.startswith("replay:"):
        return ReplayGen(mode.split(":", 1)[1], seed)
    if ml == "neutral":
        return NeutralGen(seed)
    sk = kind or src_kind()
    if sk == "flow":
        return _flow_gen(seed)
    g = _fresh("mm", seed)
    if sk == "hybrid":
        return HybridGen(g, _flow_gen(seed + 1))
    if not hasattr(g, "info"):
        try:
            g.info = {"name": "mm"}
        except Exception:
            pass
    return g


def _flow_gen(seed):
    """flow in its own process (default) or an in-process thread (AVATAR_MOTION2_FLOW_PROC=0)"""
    mode = os.environ.get("AVATAR_MOTION2_FLOW_PROC", "1")
    if mode == "sync":                                  # offline scoring: stepped inline (deterministic, same output order)
        return HighPassGen(_fresh("flow", seed), "flow")
    if mode != "0":
        pkg = _flow_pkg()
        if not os.path.isfile(os.path.join(pkg, "stream.py")):
            raise FileNotFoundError(f"no flow package at {pkg}")
        return ProcGen(pkg, seed)
    return AsyncGen(HighPassGen(_fresh("flow", seed), "flow"), "flow")


def _fresh(kind, seed):
    g = copy.deepcopy(_template(kind))
    g.seed = int(seed) & 0x7FFFFFFF
    if hasattr(g, "reset"):
        g.reset()
    return g


# ------------------------------------------------------------------------------------------------ flow / hybrid
_TRI = np.arange(135).reshape(45, 3)          # 45 rotation-vector triples (index 1 = root_shift, linear)
_TORSO_HEAD = [0, 2, 3, 4, 5, 6]              # root_rot, spine1-3, neck, head triples (+ root_shift 1)


def _aa2q(v):
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    s = np.where(th > 1e-9, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([np.cos(th / 2), v * s], -1)


def _q2aa(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    n = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    th = 2 * np.arctan2(n, q[..., :1])
    return q[..., 1:] * np.where(n > 1e-9, th / np.maximum(n, 1e-12), 2.0)


def _qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _qinv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


class AsyncGen:
    """Runs a heavy generator's step() on its own thread, DEPTH frames ahead of the consumer (flow computes 32-frame windows:
    ~3-5 ms per step with ~170 ms synchronous windows at start-up / starvation — never on the anim thread). push_audio /
    set_state are applied under a lock before the producer's next step. An empty ready queue returns the last pose (held,
    counted in info['starved']) — the render path never waits."""
    DEPTH = int(os.environ.get("AVATAR_MOTION2_ASYNC_DEPTH", "8"))

    def __init__(self, gen, name):
        self.g = gen
        self.info = {"name": name, "starved": 0}
        self.cv = threading.Condition()
        self.slock = threading.Lock()
        self.seed_req = None
        self.seeded = True
        self.ready = collections.deque()
        self.feats = []
        self.state = (False, False)
        self.last = None
        self.err = None
        self.stop = False
        self.gen_id = 0
        self.t = threading.Thread(target=self._run, name=f"motion2-{name}", daemon=True)
        self.t.start()

    def _run(self):
        while True:
            with self.cv:
                while not self.stop and len(self.ready) >= self.DEPTH:
                    self.cv.wait(0.5)
                if self.stop:
                    return
                feats, self.feats = self.feats, []
                st, gid = self.state, self.gen_id
                seed, self.seed_req = self.seed_req, None
            try:
                with self.slock:
                    if gid != self.gen_id:
                        continue
                    if feats:
                        self.g.push_audio(np.concatenate(feats, 0))
                    if seed is not None:            # hand-over: continue from the poses actually shown (one ~90 ms window)
                        self.g.seed_history(seed)
                    self.g.set_state(*st)
                    p = np.asarray(self.g.step(), np.float32).reshape(-1)
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"[:200]
                log(f"{self.info['name']} thread: {self.err}")
                return
            with self.cv:
                if gid == self.gen_id:
                    self.ready.append(p)
                    if seed is not None:
                        self.seeded = True

    def reset(self):
        with self.cv:
            self.gen_id += 1
            self.ready.clear()
            self.feats = []
        with self.slock:                 # never while the producer is inside step()
            self.g.reset()
        with self.cv:
            self.ready.clear()
            self.last = None
            self.cv.notify_all()

    def seed(self, poses):
        """drop what is queued and continue from `poses` (n,189) shown; seed_ready() turns True with its first frame"""
        if not hasattr(self.g, "seed_history"):
            return False
        with self.cv:
            self.gen_id += 1
            self.ready.clear()
            self.seed_req = np.asarray(poses, np.float32)
            self.seeded = False
            self.cv.notify_all()
        return True

    def seed_ready(self):
        with self.cv:
            return self.seeded and bool(self.ready)

    def push_audio(self, feats):
        with self.cv:
            self.feats.append(np.asarray(feats, np.float32).reshape(-1, 42))

    def set_state(self, speaking, partner_speaking):
        with self.cv:
            self.state = (bool(speaking), bool(partner_speaking))

    def step(self):
        if self.err:
            raise RuntimeError(self.err)
        with self.cv:
            if self.ready:
                self.last = self.ready.popleft()
                self.cv.notify_all()
                self.starve_run = 0
            else:
                self.info["starved"] += 1
                self.starve_run = getattr(self, "starve_run", 0) + 1
        if self.last is None:
            return neutral()
        return self.last

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()


# canon1 upper-body FK table (copy of humans/canon_drive.BODY: SMPL neutral rest offsets, metres) for the flow arm guard
_FK = [(None, 0, (0.0, 0.0, 0.0)), (0, 6, (-0.0025, 0.1088, -0.0267)), (1, 9, (0.0055, 0.1352, 0.0011)),
       (2, 12, (0.0015, 0.0529, 0.0254)), (3, 21, (0.0788, 0.1217, -0.0341)), (3, 24, (-0.0818, 0.1188, -0.0386)),
       (4, 27, (0.0910, 0.0305, -0.0089)), (5, 30, (-0.0960, 0.0326, -0.0091)), (6, 33, (0.2596, -0.0128, -0.0275)),
       (7, 36, (-0.2537, -0.0133, -0.0214)), (8, 39, (0.2492, 0.0090, -0.0012)), (9, 42, (-0.2553, 0.0078, -0.0056))]
_WRIST = {"l": 10, "r": 11}
_ARMCOLS = {"l": (27, 33), "r": (30, 36)}       # shoulder, elbow (anatomical sides)


def _rodr(v):
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1)[..., None, None]
    k = v / np.maximum(th[..., 0], 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    I = np.eye(3)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def wrist_heights(p):
    """canon (189,) -> {l, r}: wrist height above the pelvis (m), the same measure MM's hand_ymax guard uses"""
    R = _rodr(np.stack([p[c:c + 3] for _, c, _ in _FK]))
    G = [None] * len(_FK)
    P = np.zeros((len(_FK), 3))
    for k, (par, _, off) in enumerate(_FK):
        if par is None:
            G[k] = R[k]
        else:
            G[k] = G[par] @ R[k]
            P[k] = P[par] + G[par] @ np.asarray(off)
    return {a: float(P[i][1]) for a, i in _WRIST.items()}


class ArmGuard:
    """(lead 04:30): sustained high hands from the flow model sink toward the home pose. When a wrist is more than SOFT m
    above the pelvis, that arm's shoulder + elbow rotations slerp toward the canon neutral by MAXW * excess (excess 0..1 between
    SOFT and HARD), the weight time-smoothed (TAU_S) so short strokes still pass. SOFT 0 = off."""
    SOFT = float(os.environ.get("AVATAR_MOTION2_FLOW_HANDY", "0.24"))
    HARD = float(os.environ.get("AVATAR_MOTION2_FLOW_HANDY_HARD", "0.33"))
    MAXW = float(os.environ.get("AVATAR_MOTION2_FLOW_HANDY_W", "0.7"))
    TAU_S = 0.3

    def __init__(self):
        self.w = {"l": 0.0, "r": 0.0}
        self.k = 1.0 - np.exp(-1.0 / (30.0 * self.TAU_S))
        self.neu = neutral().astype(np.float64)

    def __call__(self, p):
        if self.SOFT <= 0:
            return p
        y = wrist_heights(p)
        out = p
        for a in ("l", "r"):
            e = min(max((y[a] - self.SOFT) / max(self.HARD - self.SOFT, 1e-3), 0.0), 1.0)
            self.w[a] += (self.MAXW * e - self.w[a]) * self.k
            w = self.w[a]
            if w > 1e-3:
                if out is p:
                    out = p.copy()
                for c in _ARMCOLS[a]:
                    q0, q1 = _aa2q(out[c:c + 3].astype(np.float64)), _aa2q(self.neu[c:c + 3])
                    if float(np.dot(q0, q1)) < 0:
                        q1 = -q1
                    q = q0 + (q1 - q0) * w
                    out[c:c + 3] = _q2aa(q / np.linalg.norm(q)).astype(np.float32)
        return out


def _flow_proc_main(conn, pkg, seed, depth, env):
    """child process: the flow generator (+ high-pass + arm guard) stepping DEPTH frames ahead on credits from the parent"""
    os.environ.update(env)
    try:
        mod = _load_module("t4693_motion2_flow_stream", os.path.join(pkg, "stream.py"))
        g = HighPassGen(mod.Generator(pkg, 0), "flow")
        g.g.seed = int(seed) & 0x7FFFFFFF
        g.reset()
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}"[:200]))
        return
    credits, gid, st = depth, 0, (False, False)
    while True:
        try:
            while conn.poll(0 if credits > 0 else 0.05):
                m = conn.recv()
                k = m[0]
                if k == "stop":
                    return
                if k == "feats":
                    g.push_audio(m[1])
                elif k == "state":
                    st = m[1]
                elif k == "credit":
                    credits += m[1]
                elif k == "seed":
                    gid = m[1]
                    g.seed_history(m[2])
                elif k == "reset":
                    gid = m[1]
                    g.reset()
            if credits > 0:
                g.set_state(*st)
                p = np.asarray(g.step(), np.float32).reshape(-1)
                conn.send(("pose", gid, p))
                credits -= 1
        except (EOFError, OSError, BrokenPipeError):
            return
        except Exception as e:
            try:
                conn.send(("err", f"{type(e).__name__}: {e}"[:200]))
            except Exception:
                pass
            return


class ProcGen:
    """the flow model in its OWN PROCESS (no GIL contention with the render server's anim / dispatch threads — an in-process
    thread starved frames at hub load 250). Same interface as AsyncGen: poses arrive DEPTH frames ahead on credits; step() never
    waits (an empty queue holds the last pose, counted 'starved'); seed() = hand-over (seed_history in the child)."""
    DEPTH = int(os.environ.get("AVATAR_MOTION2_ASYNC_DEPTH", "8"))

    def __init__(self, pkg, seed, name="flow"):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        env = {k: v for k, v in os.environ.items() if k.startswith(("AVATAR_MOTION2", "OMP_", "OPENBLAS_"))}
        env.setdefault("OMP_NUM_THREADS", "1"); env.setdefault("OPENBLAS_NUM_THREADS", "1")
        self.proc = ctx.Process(target=_flow_proc_main, args=(child, pkg, seed, self.DEPTH, env), daemon=True,
                                name="motion2-flow")
        self.proc.start()
        child.close()
        self.info = {"name": name, "starved": 0, "proc": True}
        self.lock = threading.Lock()
        self.ready = collections.deque()
        self.gid = 0
        self.seeded = True
        self.last = None
        self.err = None
        self.slock = threading.Lock()
        threading.Thread(target=self._rx, name="motion2-flow-rx", daemon=True).start()

    def _send(self, m):
        with self.slock:
            try:
                self.conn.send(m)
            except Exception as e:
                self.err = self.err or f"send: {type(e).__name__}"

    def _rx(self):
        while True:
            try:
                m = self.conn.recv()
            except Exception:
                self.err = self.err or "flow process gone"
                return
            if m[0] == "err":
                self.err = m[1]
                log(f"flow process: {m[1]}")
                return
            with self.lock:
                if m[1] == self.gid:
                    self.ready.append(m[2])
                    self.seeded = True
                else:
                    stale = True
            if m[1] != self.gid:
                self._send(("credit", 1))           # a stale pose (before a seed / reset) gives its credit back

    def seed(self, poses):
        with self.lock:
            self.gid += 1
            n = len(self.ready)
            self.ready.clear()
            self.seeded = False
            gid = self.gid
        self._send(("seed", gid, np.asarray(poses, np.float32)))
        if n:
            self._send(("credit", n))
        return True

    def seed_ready(self):
        with self.lock:
            return self.seeded and bool(self.ready)

    def reset(self):
        with self.lock:
            self.gid += 1
            n = len(self.ready)
            self.ready.clear()
            self.last = None
            gid = self.gid
        self._send(("reset", gid))
        if n:
            self._send(("credit", n))

    def push_audio(self, feats):
        self._send(("feats", np.asarray(feats, np.float32).reshape(-1, 42)))

    def set_state(self, speaking, partner_speaking):
        self._send(("state", (bool(speaking), bool(partner_speaking))))

    def step(self):
        if self.err:
            raise RuntimeError(self.err)
        got = False
        with self.lock:
            if self.ready:
                self.last = self.ready.popleft()
                got = True
            else:
                self.info["starved"] += 1
        self.starve_run = 0 if got else getattr(self, "starve_run", 0) + 1
        if got:
            self._send(("credit", 1))
        return neutral() if self.last is None else self.last

    def close(self):
        self._send(("stop",))
        try:
            self.proc.join(1.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:
            pass


class HighPassGen:
    """Flow's torso/head are absolute (its own mean posture): made zero-mean around the canon neutral like MM's
    (pose - slow EMA (4 s) + neutral) so the layer's additive head/torso never inherits a static tilt. Arms/fingers as is."""
    TAU_S = 4.0

    def __init__(self, gen, name):
        self.g, self.ema = gen, None
        self.info = {"name": name}
        self.k = 1.0 - np.exp(-1.0 / (30.0 * self.TAU_S))
        self.neu = neutral()[:21].astype(np.float64)
        self.guard = ArmGuard()

    def reset(self):
        self.g.reset()
        self.ema = None

    def close(self):
        getattr(self.g, "close", lambda: None)()

    def seed_history(self, poses):
        self.g.seed_history(poses)
        self.ema = self.neu.copy()             # the shown poses are already neutral-centred

    def push_audio(self, feats):
        self.g.push_audio(feats)

    def set_state(self, speaking, partner_speaking):
        self.g.set_state(speaking, partner_speaking)

    def step(self):
        p = np.asarray(self.g.step(), np.float32).reshape(-1).copy()
        x = p[:21].astype(np.float64)
        self.ema = x.copy() if self.ema is None else self.ema + self.k * (x - self.ema)
        p[:21] = (self.neu + (x - self.ema)).astype(np.float32)
        return self.guard(p)


class HybridGen:
    """flow while she speaks, motion matching while she listens / idles (lead 05:05 / 05:20 / 05:35). BOTH generators step every
    frame (live states). The output is a per-joint quaternion blend mm<->flow whose weight alpha (0 = flow, 1 = MM) follows a
    critically damped spring (tau TAU_S, zero initial velocity): at a pause / speech end the arms start easing to MM's own
    rest/listening motion after HOLD_S of silence (at rest ~0.8 s after the last voiced frame; sentence-internal pauses stay with
    flow) and come back to the still-running flow when she resumes (no re-seed, no jump: flow steps its own continuous history the
    whole time). After GAP_S (3 s) of silence flow is 'off'; the next speech start seeds flow from the
    last 48 shown poses (seed_history on flow's process) and MM carries the body until the seeded window is ready (<= 0.5 s).
    A flow queue empty > 0.3 s while on screen (a loaded box) sends alpha to MM for that stretch (never frozen arms)."""
    TAU_S = float(os.environ.get("AVATAR_MOTION2_HYB_TAU", "0.10"))      # toward MM (retract): 95 % in ~0.47 s
    TAU_IN_S = float(os.environ.get("AVATAR_MOTION2_HYB_TAU_IN", "0.07"))  # back to flow when she resumes: ~0.33 s
    HOLD_S = float(os.environ.get("AVATAR_MOTION2_HYB_HOLD", "0.3"))     # silence before the retract starts (bridges pauses)
    GAP_S = float(os.environ.get("AVATAR_MOTION2_HYB_GAP", "3.0"))       # silence after which flow re-seeds at the next speech
    STARVE_FR = 9

    def __init__(self, mm, flow):
        self.mm, self.flow = mm, flow
        self.info = {"name": "hybrid", "active": "mm", "switches": 0, "starved_on_screen": 0, "starve_fallbacks": 0}
        self.flow_info = getattr(flow, "info", {})
        self.reset_state()

    def reset_state(self):
        self.n = 0
        self.last_sp = -1e9
        self.alpha, self.alpha_v = 1.0, 0.0     # start on MM
        self.flow_live = False                  # flow seeded and usable
        self.pending = None
        self.prev = None
        self.hist = collections.deque(maxlen=48)

    def reset(self):
        self.mm.reset()
        self.flow.reset()
        self.reset_state()

    def close(self):
        getattr(self.flow, "close", lambda: None)()

    def push_audio(self, feats):
        self.mm.push_audio(feats)
        self.flow.push_audio(feats)

    def set_state(self, speaking, partner_speaking):
        self.speaking = bool(speaking)
        if speaking:
            self.last_sp = self.n
        self.mm.set_state(speaking, partner_speaking)
        self.flow.set_state(speaking, partner_speaking)

    def _spring(self, target):
        w = 1.0 / (30.0 * (self.TAU_S if target > self.alpha else self.TAU_IN_S))
        x0, v0 = self.alpha - target, self.alpha_v
        e = np.exp(-w)
        x1 = (x0 + (v0 + w * x0)) * e
        v1 = (v0 - w * (v0 + w * x0)) * e
        self.alpha, self.alpha_v = min(max(target + x1, 0.0), 1.0), v1

    def step(self):
        speaking = getattr(self, "speaking", False)
        silent_fr = self.n - self.last_sp
        if silent_fr > self.GAP_S * 30.0 and self.alpha > 0.999:
            self.flow_live = False                       # a real end of turn: flow re-seeds at the next speech start
        if speaking and not self.flow_live and len(self.hist) >= 8:
            if self.pending is None and hasattr(self.flow, "seed"):
                if self.flow.seed(np.stack(self.hist)):
                    self.pending = self.n
            elif self.pending is None:
                self.flow_live = True
            if self.pending is not None and (self.flow.seed_ready() or self.n - self.pending >= 15):
                self.pending = None
                self.flow_live = True
                self.info["switches"] += 1
        a = np.asarray(self.mm.step(), np.float32).reshape(-1).astype(np.float64)
        b = a if self.pending is not None else np.asarray(self.flow.step(), np.float32).reshape(-1).astype(np.float64)
        starving = getattr(self.flow, "starve_run", 0) > self.STARVE_FR
        if starving and self.alpha < 0.5:
            self.info["starve_fallbacks"] += 1
        on = self.flow_live and not starving and (speaking or silent_fr <= self.HOLD_S * 30.0)
        target = 0.0 if on else 1.0
        self._spring(target)
        if self.alpha < 0.5 and getattr(self.flow, "starve_run", 0) > 0:
            self.info["starved_on_screen"] += 1
        self.info["active"] = "flow" if self.alpha < 0.5 else "mm"
        self.info["alpha"] = round(self.alpha, 2)
        self.info["starved"] = self.flow_info.get("starved", 0)
        al = self.alpha
        if al >= 0.999:
            out = a
        elif al <= 0.001:
            out = b
        else:                                            # per-joint nlerp on the shortest arc, root shift linear
            qa, qb = _aa2q(a[:135].reshape(45, 3)), _aa2q(b[:135].reshape(45, 3))
            qb = qb * np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
            q = qb + (qa - qb) * al
            r = _q2aa(q / np.linalg.norm(q, axis=1, keepdims=True))
            r[1] = b[3:6] + (a[3:6] - b[3:6]) * al
            out = a.copy()
            out[:135] = r.reshape(-1)
        self.prev = out
        self.hist.append(out.astype(np.float32))
        self.n += 1
        return out.astype(np.float32)

def make_clips(seed=0):
    t = _template("act")
    if t is None:
        return None
    c = copy.deepcopy(t)
    try:
        c.rng = np.random.default_rng(seed)
        c.reset()
    except Exception:
        pass
    return c


def clip_index():
    """The library index the page / planner need: [{slug, aliases, owns, duration, wide}] (small)."""
    p = os.path.join(os.environ.get("AVATAR_MOTION2_ACT_DIR") or os.path.join(M2, "act"), "index.json")
    try:
        raw = json.load(open(p))
    except (OSError, ValueError):
        return []
    items = (raw.get("actions") or raw.get("clips") or []) if isinstance(raw, dict) else raw   # ACT index: "actions"
    out = []
    seen = set()
    for c in items if isinstance(items, list) else []:
        if not isinstance(c, dict) or not c.get("slug") or c["slug"] in seen:
            continue
        if c.get("variants") is not None and not c.get("variants"):
            continue                     # an action with no clip packaged
        seen.add(c["slug"])
        owns = [str(g) for g in (c.get("owns") or [])]
        dur = c.get("duration", c.get("dur_s"))
        if c.get("variants"):
            try:
                dur = max(float(v.get("dur_s") or 0) for v in c["variants"]) or dur
            except (TypeError, ValueError):
                pass
        # wide framing unless the clip is head / torso only (a nod, a bow, a look)
        wide = c.get("wide", any(g.startswith(("arm", "fingers")) for g in owns))
        out.append({"slug": str(c["slug"]), "aliases": [str(a) for a in (c.get("aliases") or [])][:12],
                    "owns": owns, "hand": str(c.get("hand") or "none"), "duration": dur,
                    "sustain": bool(c.get("sustain")), "wide": bool(wide)})
    return out


# ------------------------------------------------------------------------------------------------ the layer
class MotionLayer:
    def __init__(self, character, seed=None, mode=None, enabled=True, src=None):
        self.character = character
        self.mode = mode or env_mode()
        if self.mode.lower() in ("1", "on", "mm", "true", "yes"):
            self.mode = "mm"
        self.seed = int(seed if seed is not None else int.from_bytes(os.urandom(4), "little"))
        self.src_kind = src if src in ("mm", "flow", "hybrid") else src_kind()
        self.dead = None                 # error text once broken
        self.enabled = bool(enabled)     # the page's toggle
        self.lock = threading.Lock()
        self.af = StreamFeat()
        self.src = None                  # set by the loader thread (sample() / frame() return None until then)
        self.clips = None
        self.clip_err = None
        self.load_s = None
        self.clip_list = clip_index()    # index.json only (fast): the page's list is ready at once
        if self.mode.lower().startswith(("replay:", "neutral")):
            self._load()
        else:
            threading.Thread(target=self._load, name="motion2-load", daemon=True).start()
        self.partner = False
        self.body_busy = False
        self._reset_state()
        self.cost = {"push": 0.0, "sample": 0.0, "frame": 0.0, "n_sample": 0, "n_frame": 0, "steps": 0}
        self.la_used = collections.deque(maxlen=3000)
        self.la_disp = collections.deque(maxlen=3000)
        self.step_t = collections.deque(maxlen=3000)

    def _reset_state(self):
        self.vad = {}                    # frame -> vad (recent)
        self.feat_next = None            # next feature frame (global 30 fps index)
        self.gen_n = None                # frame index of `cur`
        self.prev = self.cur = None
        self.w = {g: 0.0 for g in GROUPS}
        self.w_t = None
        self.clip_n = None
        self.clip_req = []
        self.running = False             # the source is being stepped
        self.fring = collections.deque(maxlen=90)   # features while not stepping (frame, (42,))

    def _load(self):
        t0 = time.perf_counter()
        try:
            src = make_source(self.mode, self.seed, self.src_kind)
        except Exception as e:
            self._fail("load", e)
            return
        clips = None
        try:
            clips = make_clips(self.seed)
        except Exception as e:           # no library: commanded clips stay off, the conversational layer runs
            self.clip_err = f"{type(e).__name__}: {e}"[:200]
            log(f"clip player off: {self.clip_err}")
        with self.lock:
            self.clips = clips
            self.src = src
        if clips is None:
            self.clip_list = []
        self.load_s = round(time.perf_counter() - t0, 3)

    # ---------------------------------------------------------------- helpers
    @property
    def on(self):
        return self.dead is None

    def _fail(self, where, e):
        if self.dead is None:
            self.dead = f"{where}: {type(e).__name__}: {e}"[:300]
            log(f"{self.character}: layer off ({self.dead})")

    def info(self):
        i = {"on": self.on and self.enabled, "available": self.on, "character": self.character, "mode": self.mode,
             "source": ((getattr(self.src, "info", None) or {}).get("name") or "mm") if self.src is not None else "loading",
             "clips": self.clip_list, "clip_error": self.clip_err, "error": self.dead}
        return i

    def close(self):
        """the session switched away from this layer: stop a source thread (flow)."""
        try:
            getattr(self.src, "close", lambda: None)()
        except Exception:
            pass

    def reset(self):
        """A new timeline (the anim loop's >10 s hole)."""
        try:
            with self.lock:
                self.af.reset()
                self._reset_state()
                if hasattr(self.src, "reset"):
                    self.src.reset()
        except Exception as e:
            self._fail("reset", e)

    # ---------------------------------------------------------------- ws thread
    def control(self, msg):
        with self.lock:
            if "on" in msg:
                self.enabled = bool(msg.get("on")) and msg.get("on") not in ("0", "off")
            if "partner_speaking" in msg:
                self.partner = bool(msg.get("partner_speaking"))
            if "body_busy" in msg:
                self.body_busy = bool(msg.get("body_busy"))

    def clip(self, msg):
        if self.clips is None and (self.src is not None or self.dead is not None):
            raise ValueError("no clip library in this renderer" + (f" ({self.clip_err})" if self.clip_err else ""))
        slug = str(msg.get("slug") or "")
        if self.clip_list and slug not in {c["slug"] for c in self.clip_list}:
            raise ValueError(f"unknown clip {slug!r}")
        with self.lock:
            self.clip_req.append((slug, bool(msg.get("mirror")), msg.get("variant")))

    # ---------------------------------------------------------------- anim thread
    def push_audio(self, s0, pcm):
        if self.dead is not None:
            return
        if not self.enabled and not self.running:
            self._idle_off = True            # classic: no feature cost at all (restarts on a fresh timeline when on)
            return
        if getattr(self, "_idle_off", False):
            self._idle_off = False
            self.af.reset()
            self.feat_next = None
            self.vad = {}
        t1 = time.perf_counter()
        try:
            f0 = self.af.next_frame
            feats = self.af.push(int(s0), pcm)
            if f0 is None:
                f0 = self.af.next_frame - len(feats)
            if len(feats):
                if self.feat_next is None:
                    self.feat_next = f0
                for k in range(len(feats)):
                    self.vad[f0 + k] = float(feats[k, 41])
                if len(self.vad) > 400:
                    lo = f0 + len(feats) - 300
                    for k in [k for k in self.vad if k < lo]:
                        del self.vad[k]
                if self.running and self.src is not None:
                    self.src.push_audio(feats)
                else:                            # not stepping yet: keep the newest 3 s for the start
                    for k in range(len(feats)):
                        self.fring.append((f0 + k, feats[k]))
                self.feat_next = f0 + len(feats)
        except Exception as e:
            self._fail("push_audio", e)
        self.cost["push"] += time.perf_counter() - t1

    def _speaking(self, n):
        v = self.vad
        return any(v.get(n + k, 0.0) > 0.5 for k in range(0, LOOKAHEAD_FR + 1))

    def _step_to(self, n_want):
        """Advance the source until cur is frame n_want (one step per 30 fps frame, in order)."""
        if self.src is None:                    # still loading (the session runs on the dials meanwhile)
            return False
        if self.gen_n is None:
            # start the source one frame before the wanted one, fed the features from that frame on
            if self.feat_next is None:
                return False
            n0 = n_want - 1
            if hasattr(self.src, "reset"):
                self.src.reset()
            use = [f for k, f in self.fring if k >= n0]
            self.fring.clear()
            if use:
                self.src.push_audio(np.stack(use).astype(np.float32))
            self.running = True
            self.gen_n = n0 - 1
        steps = 0
        while self.gen_n < n_want and steps < 120:
            n = self.gen_n + 1
            if self.feat_next is not None:
                self.la_used.append(self.feat_next - n)     # frames of future audio features MM has when stepping frame n
            self.src.set_state(self._speaking(n), self.partner)
            ts = time.perf_counter()
            p = np.asarray(self.src.step(), np.float32).reshape(-1)
            self.step_t.append(time.perf_counter() - ts)
            if p.size < 189:
                p = np.concatenate([p, neutral()[p.size:]])
            if not np.isfinite(p[:135]).all():
                raise ValueError("non-finite pose from the source")
            self.prev, self.cur = (self.cur if self.cur is not None else p), p
            self.gen_n = n
            steps += 1
        if self.gen_n < n_want:                 # too far behind (a stall): jump, no interpolation across the jump
            self.gen_n = n_want
            self.prev = self.cur
        self.cost["steps"] += steps
        return True

    def sample(self, t):
        """-> (189,) float32 body pose at playback time t, or None (off / not started / broken)."""
        if self.dead is not None:
            return None
        if not self.enabled and self.running and max(self.w.values()) <= 0.0:
            self.running = False             # toggled off and faded out: stop stepping (restarts fresh on "on")
            self.gen_n = None
            self.prev = self.cur = None
        if not self.enabled and not self.running:
            return None
        t1 = time.perf_counter()
        try:
            f = float(t) * 30.0
            n1 = int(np.floor(f + 1e-6))
            if not self._step_to(n1 + 1):
                return None
            a = f - n1
            if self.gen_n == n1 + 1 and self.prev is not None:
                out = self.prev + (self.cur - self.prev) * np.float32(min(max(a, 0.0), 1.0))
            else:
                out = self.cur
            self.cost["sample"] += time.perf_counter() - t1
            self.cost["n_sample"] += 1
            return out.astype(np.float32)
        except Exception as e:
            self._fail("sample", e)
            return None

    # ---------------------------------------------------------------- dispatch thread
    def frame(self, pose, t, gest_l, gest_r):
        """-> (canon (189,), canon_w, modes, scale_rig_l, scale_rig_r) or None (nothing to apply)."""
        if self.dead is not None or pose is None:
            if self.dead is not None:
                return None
            if max(self.w.values()) <= 0.0:
                self.w_t = None              # the next fade-in starts from dt 0 (slewed, never a jump)
                return None
        t1 = time.perf_counter()
        try:
            dt = 0.0 if self.w_t is None else float(t) - self.w_t
            if not (0.0 <= dt <= 0.25):
                dt = 0.0 if self.w_t is None or dt < 0 else 0.25
            self.w_t = float(t)
            if self.feat_next is not None:                  # the future audio that EXISTS when this frame is dispatched
                self.la_disp.append(self.feat_next - float(t) * 30.0)
            with self.lock:
                if self.clips is not None or self.src is not None:
                    reqs, self.clip_req = self.clip_req, []
                else:
                    reqs = []                    # still loading: keep a commanded clip queued
                enabled, busy = self.enabled, self.body_busy
            if pose is not None:
                self.last_pose = pose
            base = pose if pose is not None else getattr(self, "last_pose", None)
            if base is None:
                base = neutral()
            cw = {}
            if self.clips is not None:
                for slug, mirror, variant in reqs:
                    kw = {"mirror": mirror}
                    if variant is not None:
                        kw["variant"] = variant
                    self.clips.play(slug, **kw)
                n = int(np.floor(float(t) * 30.0 + 1e-6))
                k = 1 if self.clip_n is None else min(3, max(0, n - self.clip_n))
                self.clip_n = n
                res = None
                for _ in range(k):
                    res = self.clips.step(base)
                if res is not None:
                    cp, cw = res
                    cw = {g: float(v) for g, v in (cw or {}).items() if g in GROUPS}
                    if cw and max(cw.values()) > 0.0:
                        base = np.asarray(cp, np.float32).reshape(-1)
                    else:
                        cw = {}
            on = enabled and pose is not None
            free_r = min(max(float(gest_l), 0.0) / GREF, 1.0) if GREF > 0 else 0.0     # rig _l = anatomical right
            free_l = min(max(float(gest_r), 0.0) / GREF, 1.0) if GREF > 0 else 0.0
            tgt = {"torso": 0.0 if busy else TORSO_W, "head": 0.0 if busy else HEAD_W,
                   "arm_l": free_l * ARM_W, "arm_r": free_r * ARM_W, "fingers_l": free_l * ARM_W, "fingers_r": free_r * ARM_W}
            step = SLEW * dt
            out_w = {}
            for g in GROUPS:
                tg = tgt[g] if on else 0.0
                c = self.w[g]
                c = min(tg, c + step) if tg > c else max(tg, c - step)
                self.w[g] = c
                out_w[g] = max(c, cw.get(g, 0.0))
            self.cost["frame"] += time.perf_counter() - t1
            self.cost["n_frame"] += 1
            if max(out_w.values()) <= 0.0:
                return None
            modes = {"head": "add", "torso": "add"}
            return (np.ascontiguousarray(base, np.float32), out_w, modes,
                    1.0 - out_w["arm_r"], 1.0 - out_w["arm_l"])
        except Exception as e:
            self._fail("frame", e)
            return None

    def stats(self):
        c, self.cost = self.cost, {"push": 0.0, "sample": 0.0, "frame": 0.0, "n_sample": 0, "n_frame": 0, "steps": 0}
        ms = lambda a, n: round(1000.0 * a / n, 3) if n else None
        return {"on": self.on and self.enabled, "error": self.dead, "sample_ms": ms(c["sample"], c["n_sample"]),
                "push_ms_per_frame": ms(c["push"], c["n_sample"]), "frame_ms": ms(c["frame"], c["n_frame"]),
                "steps": c["steps"], "w": {g: round(v, 2) for g, v in self.w.items()},
                "partner": self.partner, "body_busy": self.body_busy, "load_s": self.load_s,
                "src": {k: v for k, v in (getattr(self.src, "info", None) or {}).items()
                        if k in ("name", "active", "alpha", "switches", "starved", "starved_on_screen", "starve_fallbacks")} if self.src is not None else "loading",
                "step_ms": self._pct_ms(self.step_t),
                "la_used_ms": self._pct(self.la_used), "la_dispatch_ms": self._pct(self.la_disp)}

    @staticmethod
    def _pct_ms(d):
        """per source step() wall time: mean / p50 / p99 ms"""
        if not d:
            return None
        a = np.asarray(d, np.float64) * 1000.0
        return [round(float(a.mean()), 3), round(float(np.percentile(a, 50)), 3), round(float(np.percentile(a, 99)), 3)]

    @staticmethod
    def _pct(d):
        """look-ahead p10 / p50 / p90 in ms over the last ~100 s (frames of 1/30 s)"""
        if not d:
            return None
        a = np.asarray(d, np.float64) * (1000.0 / 30.0)
        return [round(float(np.percentile(a, q)), 1) for q in (10, 50, 90)]


def build(character, seed=None, enabled=True, src=None):
    """-> MotionLayer or None (not wanted for this character / failed to build; the failure is logged once)."""
    if not wanted(character):
        return None
    try:
        return MotionLayer(character, seed=seed, enabled=enabled, src=src)
    except Exception as e:
        log(f"{character}: no motion layer ({type(e).__name__}: {e})")
        return None
