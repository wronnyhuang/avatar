"""MediaPipe -> CANON v1 fitter. CPU, causal, stateful; ~0.3 ms/frame.

    from canon1.fit import Fitter
    f = Fitter()                       # loads data/fit/fitter_params.npz (falls back to the analytic-only map)
    motion189, mask9 = f.step(mp_frame)
mp_frame = dict(face_present, blendshapes[52], face_matrix[16] (row-major 4x4, as MediaPipe gives it),
                pose_world[33,4] (x,y,z,visibility, metres) | None, pose_image[33,4] (normalized) | None,
                hands=[{handedness:'Left'|'Right' (as MediaPipe reports it), world[21,3], image[21,3], score}],
                image_wh=(w,h)).
Pose/hands may be absent on frames where the page did not run those models: the fitter holds the last result for
up to HOLD_S seconds (then the group's mask drops to 0). The camera image is UNMIRRORED (raw).

Design: analytic part on the canonical SMPL-H skeleton + a learned linear correction per group.
  face_upper / mouth     MediaPipe blendshapes pass through (the canon face IS MediaPipe's blendshape space).
  gaze                   eyeLook* pass through; gaze_yaw/pitch = the iris-calibrated map (canon1.seamless.gaze_from_bs).
  torso / head / arms    features: torso frame from the pose world shoulders (+hips), head rotation from the face matrix,
                         analytic arm IK (shoulder swing+twist from the elbow bend plane, elbow flexion, wrist from the
                         hand points), running-median-relative rotations; -> per-group ridge regression to the SMPL-H
                         canon of Seamless TRAIN-side people filmed by MediaPipe (tools/data/fit_train.py).
  fingers                hand world landmarks in the hand's own frame -> ridge to SMPL-H finger params.
Frames: MediaPipe camera/world coords are converted to C = (x: image right = the person's LEFT, y: up, z: toward the
camera = the person's forward), the same axes as the SMPL body frame for a person facing the camera.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import spec as S

from . import paths as _paths  # noqa: E402

PARAMS = Path(os.environ.get("FITTER_PARAMS", str(_paths.DATA / "fit" / "fitter_params.npz")))
HOLD_S = 0.5
FPS = S.FPS
G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
# iris-calibrated eyeLook -> radians (tools/data/gaze_calib.py); overwritten by the params file when present
from .seamless import GAZE_CALIB as _GC  # noqa: E402
GAZE_CALIB = {k: v for k, v in _GC.items() if k != "status"}

# MediaPipe pose indices (person's anatomical side)
P_NOSE, P_LSH, P_RSH, P_LEL, P_REL, P_LWR, P_RWR = 0, 11, 12, 13, 14, 15, 16
P_LPI, P_RPI, P_LIN, P_RIN, P_LTH, P_RTH, P_LHIP, P_RHIP = 17, 18, 19, 20, 21, 22, 23, 24
P_LEAR, P_REAR = 7, 8


# ----------------------------------------------------------------------------------------- small rotation helpers
# scalar (python-float) versions: numpy's vectorized helpers cost ~50-100 us per call on a single 3-vector
import math  # noqa: E402


def _norm(v, eps=1e-9):
    n = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    return v / n if n > eps else v * 0.0


def _cross(a, b):
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2]); b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _rotvec(R):
    """3x3 rotation -> axis-angle (scalar code; same result as spec.mat_to_aa)."""
    c = (float(R[0, 0] + R[1, 1] + R[2, 2]) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], np.float64)
    s = math.sin(th)
    if s < 1e-6:
        if th < 1.0:
            return v / 2.0
        return S.mat_to_aa(np.asarray(R)[None])[0]
    return v * (th / (2.0 * s))


def _aa2mat(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    th = math.sqrt(x * x + y * y + z * z)
    if th < 1e-12:
        return np.eye(3)
    x, y, z = x / th, y / th, z / th
    c, s = math.cos(th), math.sin(th); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _frame_from(a, b):
    """orthonormal frame with first axis a, second axis b orthogonalized (columns)."""
    a = _norm(a)
    b = _norm(b - a * float(a @ b))
    return np.stack([a, b, _cross(a, b)], 1)


def _orth(R):
    """orthonormalize (MediaPipe's matrix may carry scale): Gram-Schmidt on the first two columns."""
    return _frame_from(R[:, 0], R[:, 1])


def to_C(p):
    """MediaPipe world/camera (x right, y down, z away) -> C (x person's left, y up, z toward camera)."""
    p = np.asarray(p, np.float64)
    return p * np.array([1.0, -1.0, -1.0])


class RunningMedian:
    """causal running median of a vector over the last `n` pushed samples, refreshed every `every` pushes."""

    def __init__(self, dim, n=450, every=10):
        self.buf = np.zeros((n, dim)); self.k = 0; self.n = n; self.every = every
        self.med = None; self.cnt = 0

    def push(self, v):
        self.buf[self.k % self.n] = v
        self.k += 1
        self.cnt += 1
        if self.med is None or self.cnt >= self.every or self.k <= 30:
            self.med = np.median(self.buf[:min(self.k, self.n)], 0)
            self.cnt = 0
        return self.med

    def get(self, default):
        return default if self.med is None else self.med


# ----------------------------------------------------------------------------------------- analytic IK
REST_ARM = {  # rest bone directions (shoulder->elbow) in the shoulder-parent frame + rest elbow bend axis
    "l": dict(r=np.array([1.0, 0, 0]), bend=np.array([0, -1.0, 0]), palm=np.array([0, -1.0, 0])),
    "r": dict(r=np.array([-1.0, 0, 0]), bend=np.array([0, 1.0, 0]), palm=np.array([0, -1.0, 0])),
}


def arm_ik(sh, el, wr, hand_dir, hand_nrm, Rpar, side, prev_bend=None):
    """sh, el, wr: joint positions in C; hand_dir / hand_nrm: hand pointing direction and palm normal in C (or None);
    Rpar: global rotation of the shoulder joint's parent (torso frame x collar). Returns shoulder (3), elbow (3),
    wrist (3) local axis-angles, elbow angle, bend axis (in C) for hold-over."""
    rest = REST_ARM[side]
    u = _norm(Rpar.T @ (el - sh)); f = _norm(Rpar.T @ (wr - el))
    theta = math.acos(min(1.0, max(-1.0, float(u @ f))))
    b = _cross(u, f)
    if math.sqrt(float(b @ b)) < 0.15 and prev_bend is not None:          # nearly straight: keep the last bend plane
        b = prev_bend - u * float(prev_bend @ u)
    if math.sqrt(float(b @ b)) < 1e-6:
        b = rest["bend"] - u * float(rest["bend"] @ u)
    b = _norm(b)
    A_rest = _frame_from(rest["r"], rest["bend"])
    A_obs = _frame_from(u, b)
    R_sh = A_obs @ A_rest.T
    sgn = -1.0 if side == "l" else 1.0
    el_aa = np.array([0.0, sgn * theta, 0.0])
    R_el = _aa2mat(el_aa)
    wr_aa = np.zeros(3)
    if hand_dir is not None and hand_nrm is not None:
        Rw = Rpar @ R_sh @ R_el                                       # wrist parent global
        hd = _norm(Rw.T @ hand_dir); hn = _norm(Rw.T @ hand_nrm)
        A_rest_w = _frame_from(rest["r"], rest["palm"])
        A_obs_w = _frame_from(hd, hn)
        wr_aa = _rotvec(A_obs_w @ A_rest_w.T)
    return _rotvec(R_sh), el_aa, wr_aa, theta, Rpar @ b


def hand_frame_feats(Hw, side):
    """hand world landmarks (21,3) in C -> (63,) normalized coordinates in the hand's own frame (wrist origin,
    x along wrist->middle MCP, y = palm normal side, unit = wrist->middle MCP length), mirrored for the left hand so
    one regressor serves both (x stays the pointing axis; the lateral axis flips)."""
    H = np.asarray(Hw, np.float64)
    w = H[0]
    x = _norm(H[9] - w)
    lat = _norm(H[5] - H[17])                   # index side minus pinky side
    R = _frame_from(x, lat)                     # columns: x (pointing), lat, normal
    loc = (H - w) @ R / max(np.linalg.norm(H[9] - w), 1e-4)
    if side == "l":
        loc = loc * np.array([1.0, 1.0, -1.0])     # a world mirror flips the frame's 3rd (normal) coordinate
    return loc.reshape(-1)


def hand_dir_nrm(H, side):
    """direction wrist->middle MCP and palm normal (toward the palm side) from 21 hand points or pose hand points."""
    d = _norm(H["mid"] - H["wrist"])
    a, b = H["index"] - H["wrist"], H["pinky"] - H["wrist"]
    n = _norm(_cross(a, b)) if side == "r" else _norm(_cross(b, a))
    return d, n


# ----------------------------------------------------------------------------------------- the fitter
FEAT_NAMES = []


class Fitter:
    def __init__(self, params=None, smooth=True):
        p = Path(params) if params else PARAMS
        self.W = {}
        self.gaze = dict(GAZE_CALIB)
        self.neutral = np.zeros(S.D_MOTION)
        self.version = "analytic"
        if p.exists():
            d = np.load(p, allow_pickle=False)
            for k in d.files:
                if k.split("_")[0] in ("W", "mu", "sd", "W1", "b1", "W2", "b2", "W3", "b3", "ysd"):
                    self.W[k] = d[k]
            if "neutral" in d.files:
                self.neutral = d["neutral"].astype(np.float64)
            if "cfg" in d.files:
                cfg = json.loads(str(d["cfg"]))
                self.gaze.update(cfg.get("gaze", {}))
                self.version = cfg.get("version", "ridge")
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.t = 0
        self.last_pose = None; self.last_pose_t = -1e9
        self.last_hands = {"l": None, "r": None}; self.last_hand_t = {"l": -1e9, "r": -1e9}
        self.rm_torso = RunningMedian(3); self.rm_head = RunningMedian(3); self.rm_hit = RunningMedian(3)
        self.rm_tr = RunningMedian(3)
        self.prev_bend = {"l": None, "r": None}
        self.prev = None
        self.last_ms = 0.0
        self.pose_ver = 0; self.hand_ver = {"l": 0, "r": 0}
        self._ycache = {}
        self._cache = {}     # recomputed only when a new pose / hand result arrived (live: pose 6, hands 10 fps)

    # --------------------------------------------------------------------------- per-frame features
    def features(self, fr):
        """-> dict of feature blocks (+ validity flags) for this frame; updates the causal state."""
        t = self.t
        self.t += 1
        out = {}
        # ---- face
        face = bool(fr.get("face_present")) and fr.get("face_matrix") is not None
        bs = np.asarray(fr.get("blendshapes") if fr.get("blendshapes") is not None else np.zeros(52), np.float64)
        Rh = np.eye(3); th = np.zeros(3)
        if face:
            M = np.asarray(fr["face_matrix"], np.float64).reshape(4, 4)
            if np.abs(M).sum() > 0:
                Rh = _orth(M[:3, :3]); th = M[:3, 3] / 10.0          # cm -> dm (already in C axes)
            else:
                face = False
        out["face"] = face
        out["bs"] = bs
        # ---- pose (hold last)
        pw = fr.get("pose_world")
        if pw is not None and np.abs(np.asarray(pw)).sum() > 0:
            self.pose_ver += 1
            self.last_pose = (to_C(np.asarray(pw, np.float64)[:, :3]), np.asarray(pw, np.float64)[:, 3],
                              None if fr.get("pose_image") is None else np.asarray(fr["pose_image"], np.float64))
            self.last_pose_t = t
        pose_ok = self.last_pose is not None and (t - self.last_pose_t) <= HOLD_S * FPS
        out["pose_ok"] = pose_ok
        # ---- hands (assign to anatomical sides)
        for h in fr.get("hands") or []:
            side = self._hand_side(h, fr)
            if side is None:
                continue
            Hw = to_C(np.asarray(h["world"], np.float64))
            self.last_hands[side] = (Hw, float(h.get("score", 1.0)))
            self.hand_ver[side] += 1
            self.last_hand_t[side] = t
        hand_ok = {s: self.last_hands[s] is not None and (t - self.last_hand_t[s]) <= HOLD_S * FPS for s in "lr"}
        out["hand_ok"] = hand_ok
        # ---- torso frame
        Rt = np.eye(3); vis = np.zeros(33)
        torso_ok = False
        if pose_ok:
            X, vis, _ = self.last_pose
            torso_ok = vis[P_LSH] > 0.5 and vis[P_RSH] > 0.5
            xa = X[P_LSH] - X[P_RSH]
            if vis[P_LHIP] > 0.5 and vis[P_RHIP] > 0.5:
                up = 0.5 * (X[P_LSH] + X[P_RSH]) - 0.5 * (X[P_LHIP] + X[P_RHIP])
            else:
                up = np.array([0.0, 1.0, 0.0])
            Rt = _frame_from(xa, up)                    # columns x (left), y (up), z = left x up = forward
        out["torso_ok"] = torso_ok
        # relative rotations (running medians, causal)
        tr_rv = _rotvec(Rt)
        hd_rv = _rotvec(Rh)
        if torso_ok:
            mt = self.rm_torso.push(tr_rv)
        else:
            mt = self.rm_torso.get(tr_rv)
        Rt_rel = _aa2mat(mt).T @ Rt
        if face:
            mh = self.rm_head.push(hd_rv)
            mtr = self.rm_tr.push(th)
        else:
            mh = self.rm_head.get(hd_rv); mtr = self.rm_tr.get(th)
        Rh_rel = _aa2mat(mh).T @ Rh
        Rhit = Rt.T @ Rh                                              # head in torso (absolute)
        hit_rv = _rotvec(Rhit)
        if face and torso_ok:
            mhit = self.rm_hit.push(hit_rv)
        else:
            mhit = self.rm_hit.get(hit_rv)
        out["torso"] = np.concatenate([_rotvec(Rt_rel), tr_rv, Rt[:, 1], Rt[:, 2]])
        out["head"] = np.concatenate([_rotvec(Rh_rel), _rotvec(_aa2mat(mhit).T @ Rhit), hit_rv,
                                      (th - mtr), Rh[:, 2], Rh[:, 1]])
        # ---- arms (IK in the torso frame)
        arms = {}
        for side, (i_sh, i_el, i_wr, i_pi, i_in) in (("l", (P_LSH, P_LEL, P_LWR, P_LPI, P_LIN)),
                                                      ("r", (P_RSH, P_REL, P_RWR, P_RPI, P_RIN))):
            key = ("arm", side, self.pose_ver, self.hand_ver[side], pose_ok, hand_ok[side])
            if key in self._cache:
                arms[side] = self._cache[key]
                continue
            ok = pose_ok and min(vis[i_sh], vis[i_el], vis[i_wr]) > 0.5
            if pose_ok and self.last_pose[2] is not None:
                im = self.last_pose[2]
                inside = all(-0.02 < im[i, 0] < 1.02 and -0.02 < im[i, 1] < 1.02 for i in (i_el, i_wr))
                ok = ok and inside
            feats = np.zeros(24)
            if pose_ok:
                X = self.last_pose[0]
                if hand_ok[side]:
                    Hw = self.last_hands[side][0]
                    hp = dict(wrist=Hw[0], mid=Hw[9], index=Hw[5], pinky=Hw[17])
                else:
                    hp = dict(wrist=X[i_wr], mid=0.5 * (X[i_pi] + X[i_in]), index=X[i_in], pinky=X[i_pi])
                hdir, hnrm = hand_dir_nrm(hp, side)
                sh_aa, el_aa, wr_aa, th_el, bend = arm_ik(X[i_sh], X[i_el], X[i_wr], hdir, hnrm, Rt, side,
                                                          self.prev_bend[side])
                self.prev_bend[side] = Rt.T @ bend
                u = Rt.T @ _norm(X[i_el] - X[i_sh]); f = Rt.T @ _norm(X[i_wr] - X[i_el])
                feats = np.concatenate([sh_aa, el_aa, wr_aa, [th_el, np.cos(th_el)], u, f, Rt.T @ hdir, Rt.T @ hnrm,
                                        [float(hand_ok[side])]])
            arms[side] = (ok, feats)
            self._cache = {k: v for k, v in self._cache.items() if not (k[0] == "arm" and k[1] == side)}
            self._cache[key] = arms[side]
        out["arms"] = arms
        # ---- fingers
        fing = {}
        for side in "lr":
            key = ("fing", side, self.hand_ver[side], hand_ok[side])
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if not (k[0] == "fing" and k[1] == side)}
                self._cache[key] = (True, hand_frame_feats(self.last_hands[side][0], side)) if hand_ok[side] \
                    else (False, np.zeros(63))
            fing[side] = self._cache[key]
        out["fingers"] = fing
        out["vis"] = vis
        return out

    def _hand_side(self, h, fr):
        """anatomical side of a MediaPipe hand: nearest pose wrist in the image when pose is known, else the
        handedness label swapped (MediaPipe labels assume a mirrored selfie image; ours is raw)."""
        if self.last_pose is not None and self.last_pose[2] is not None and h.get("image") is not None:
            im = self.last_pose[2]
            hw = np.asarray(h["image"], np.float64)[0, :2]
            dl = np.linalg.norm(im[P_LWR, :2] - hw); dr = np.linalg.norm(im[P_RWR, :2] - hw)
            if min(dl, dr) < 0.15:
                return "l" if dl < dr else "r"
        lab = str(h.get("handedness", "")).lower()
        if lab.startswith("left"):
            return "r"
        if lab.startswith("right"):
            return "l"
        return None

    # --------------------------------------------------------------------------- feature vectors per group
    @staticmethod
    def group_inputs(F):
        """feature dict -> {group: (valid, x)} inputs of the learned maps."""
        out = {}
        out["torso"] = (F["torso_ok"], np.concatenate([F["torso"], F["head"][9:12]]))
        out["head"] = (F["face"], np.concatenate([F["head"], F["torso"][:3], [float(F["torso_ok"])]]))
        for s, g in (("l", "arm_l"), ("r", "arm_r")):
            ok, x = F["arms"][s]
            out[g] = (ok, np.concatenate([x, F["torso"][:3]]))
        for s, g in (("l", "fingers_l"), ("r", "fingers_r")):
            out[g] = F["fingers"][s]
        return out

    def _apply(self, g, x):
        key = {"fingers_l": "fingers", "fingers_r": "fingers", "arm_l": "arm_l", "arm_r": "arm_r"}.get(g, g)
        W = self.W.get(f"W_{key}")
        if W is None:
            return None
        xn = (x - self.W[f"mu_{key}"]) / self.W[f"sd_{key}"]
        y = np.concatenate([xn, [1.0]]) @ W
        W1 = self.W.get(f"W1_{key}")
        if W1 is not None:      # residual MLP on top of the ridge map (fit_train.py --mlp)
            h = np.maximum(np.clip(xn, -8, 8) @ W1 + self.W[f"b1_{key}"], 0)
            h = np.maximum(h @ self.W[f"W2_{key}"] + self.W[f"b2_{key}"], 0)
            y = y + (h @ self.W[f"W3_{key}"] + self.W[f"b3_{key}"]) * self.W[f"ysd_{key}"]
        if g == "fingers_l":
            y = y * np.tile([1.0, -1.0, -1.0], 15)          # the model is trained on right-hand-canonical targets
        return y

    def step(self, fr):
        t0 = time.perf_counter()
        F = self.features(fr)
        m = self.neutral.copy() if self.prev is None else self.prev.copy()   # unseen groups hold their last value
        k = np.zeros(S.D_MASK, np.uint8)
        # face / gaze: pass-through
        if F["face"]:
            m[S.SL_FACE] = np.clip(F["bs"], 0, 1)
            from .seamless import gaze_from_bs
            m[S.SL_GAZE] = gaze_from_bs(F["bs"][None], self.gaze)[0]
            k[G["face_upper"]] = k[G["mouth"]] = k[G["gaze"]] = 1
        gi = self.group_inputs(F)
        for g, (ok, x) in gi.items():
            y = None
            if ok:
                c = self._ycache.get(g)
                if c is not None and c[0].shape == x.shape and np.array_equal(c[0], x):
                    y = c[1]
                else:
                    y = self._apply(g, x)
                    self._ycache[g] = (x.copy(), y)
            cols = S.GROUPS[g] if g not in ("torso",) else S.GROUPS["torso"]
            if y is None and not ok:
                continue
            if y is None:
                # analytic only
                if g in ("arm_l", "arm_r") and ok:
                    s = g[-1]
                    m[S.jcols(f"shoulder_{s}")] = x[0:3]; m[S.jcols(f"elbow_{s}")] = x[3:6]
                    m[S.jcols(f"wrist_{s}")] = x[6:9]
                elif g == "head" and ok:
                    m[S.jcols("head")] = self.neutral[S.jcols("head")] + 0.6 * x[3:6]
                    m[S.jcols("neck")] = self.neutral[S.jcols("neck")] + 0.4 * x[3:6]
                elif g == "torso" and ok:
                    m[0:3] = x[0:3]
                k[G[g]] = int(bool(ok))
                continue
            if ok:
                m[cols] = y
                k[G[g]] = 1
        # causal smoothing of the body part (one-pole, light) to tame landmark jitter
        if self.smooth and self.prev is not None:
            a = 0.55
            body = slice(0, 135)
            m[body] = a * m[body] + (1 - a) * self.prev[body]
        self.prev = m.copy()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return m.astype(np.float32), k


# ----------------------------------------------------------------------------------------- replay helpers
def frames_from_raw(raw, T=None, offset_frames=0.0):
    """raw MediaPipe arrays (tools/data/mp_extract.py keys) -> list of mp_frame dicts on the 30 fps grid
    t_k = (k + offset)/30 (latest native frame at or before t_k; pose/hands only on frames where they ran, else
    None = 'not run this frame', as on the live page)."""
    ts = raw["ts_ms"].astype(np.float64) / 1000.0
    n = len(ts)
    T = T or int(round(ts[-1] * FPS)) + 1
    w, h = [int(v) for v in raw["wh"]]
    pr = raw["pose_ran"].astype(bool) if "pose_ran" in raw else np.ones(n, bool)
    hr = raw["hand_ran"].astype(bool) if "hand_ran" in raw else np.ones(n, bool)
    tk = (np.arange(T) + offset_frames) / FPS
    idx = np.clip(np.searchsorted(ts, tk + 1e-4, side="right") - 1, 0, n - 1)
    out = []
    last = -1
    for k in range(T):
        i = idx[k]
        fresh = i != last
        last = i
        fr = dict(face_present=bool(raw["face"][i]), blendshapes=raw["bs"][i], face_matrix=raw["mat"][i],
                  pose_world=None, pose_image=None, hands=[], image_wh=(w, h))
        if fresh and pr[i] and raw["pose_ok"][i]:
            fr["pose_world"] = raw["pose_world"][i] if "pose_world" in raw else None
            fr["pose_image"] = raw["pose"][i]
        if fresh and hr[i]:
            for j in range(int(raw["hand_n"][i])):
                lab = int(raw["hand_label"][i, j]) if "hand_label" in raw else -1
                fr["hands"].append(dict(handedness={0: "Left", 1: "Right"}.get(lab, ""),
                                        world=raw["hands_world"][i, j] if "hands_world" in raw else None,
                                        image=raw["hands"][i, j], score=float(raw["hand_score"][i, j])))
            fr["hands"] = [hh for hh in fr["hands"] if hh["world"] is not None]
        if tk[k] < ts[0] - 0.02 or tk[k] > ts[-1] + 0.05:
            fr["face_present"] = False
        out.append(fr)
    return out
