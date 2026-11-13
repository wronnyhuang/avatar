"""NVIDIA Audio2Face-3D v3.0 (diffusion) on CPU via onnxruntime + its blendshape solve -> ARKit-52 weights in OUR
canon order (phase-2 audio, plan item 3). Licence: NVIDIA Open Model License (see logs/p2-audio.md).

Inference follows the A2F-3D SDK (github.com/NVIDIA/Audio2Face-3D-SDK, read, not run):
  * 16 kHz audio, window 16000 samples, stride 8000; first window starts 16000 samples BEFORE the track (zeros);
    each run outputs 60 frames at 60 fps, the centre 30 (frames 15..44) are kept: 60 fps frame n of the track comes
    from run i = (n + 45) // 30, row 15 + (n + 45) % 30. Look-ahead: 0.25-0.75 s (teacher only, never live).
  * GRU state (input_latents -> output_latents) carried across runs; identity one-hot; emotion = zeros (neutral);
    diffusion noise = ONE fixed N(0,1) tensor (the SDK's constantNoise option) -> deterministic teacher.
  * prediction row = [skin 72006 | tongue 16806 | jaw 15 | eyes 4]; skin pose = model neutral + delta
    (+ eyeClosePoseDelta * -eyelid_open_offset), smoothing negligible at the configured constants.
  * blendshape solve (blendshape_solver_base.cpp / blendshape_solver.cpp): frontalMask vertices, active poses from
    bs_skin_config, A = deltas; AMat = A'A + L1^2*.25s*1 + L2*10s*I + T*100s*I + Sym*10s*S'S,
    b = A'(pose - bs neutral) + T*s*w_prev, bounded LS 0..1 (BVLS), multipliers/offsets; s = (bbox/templateBB)^2.
Our canon frame k (time k/30) = A2F 60 fps frame 2k.
"""
import json
import os

import numpy as np

# NVIDIA Audio2Face-3D v3.0 model files: $A2F_DIR, default $MM_WORK_DIR/p2/models/a2f3d_v3
A2F_DIR = os.environ.get("A2F_DIR", os.path.join(os.environ.get(
    "MM_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "work")),
    "p2", "models", "a2f3d_v3"))
SKIN, TONGUE, JAW, EYES = 72006, 16806, 15, 4
NOUT = SKIN + TONGUE + JAW + EYES


class A2F:
    def __init__(self, identity="Mark", threads=4, seed=0, model_dir=A2F_DIR):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(model_dir, "network.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        info = json.load(open(os.path.join(model_dir, "network_info.json")))
        self.ids = info["params"]["identities"]
        self.identity = identity
        self.id_vec = np.zeros((1, len(self.ids)), np.float32)
        self.id_vec[0, self.ids.index(identity)] = 1.0
        self.noise = np.random.default_rng(seed).standard_normal((1, 3, 60, NOUT)).astype(np.float32)
        mcfg = json.load(open(os.path.join(model_dir, f"model_config_{identity}.json")))["config"]
        md = np.load(os.path.join(model_dir, f"model_data_{identity}.npz"), allow_pickle=False)
        self.neutral_model = md["neutral_skin"].reshape(-1).astype(np.float64)
        self.pose_bias = (md["eye_close_pose_delta"].reshape(-1) * -float(mcfg.get("eyelid_open_offset", 0.0))
                          + md["lip_open_pose_delta"].reshape(-1) * float(mcfg.get("lip_open_offset", 0.0)))
        self.skin_strength = float(mcfg.get("skin_strength", 1.0))
        bs = np.load(os.path.join(model_dir, f"bs_skin_{identity}.npz"), allow_pickle=False)
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in bs["poseNames"]]
        assert names[0] == "neutral"
        self.pose_names = names[1:]
        cfg = json.load(open(os.path.join(model_dir, f"bs_skin_config_{identity}.json")))["blendshape_params"]
        self.cfg = cfg
        mask = bs["frontalMask"].astype(np.int64)
        pos = np.stack([3 * mask, 3 * mask + 1, 3 * mask + 2], 1).reshape(-1)
        self.pos = pos
        neutral = bs["neutral"].reshape(-1).astype(np.float64)
        D = np.stack([bs[n].reshape(-1) for n in self.pose_names], 1).astype(np.float64)   # (V*3, 52)
        self.active = np.array(cfg["bsSolveActivePoses"], bool)
        A = D[pos][:, self.active]
        self.A = A
        self.bs_neutral_m = neutral[pos]
        nmat = neutral.reshape(-1, 3)
        s = (np.linalg.norm(nmat.max(0) - nmat.min(0)) / cfg["templateBBSize"]) ** 2
        self.s = s
        L1, L2, Tr, Sy = (cfg["strengthL1regularization"], cfg["strengthL2regularization"],
                          cfg["strengthTemporalSmoothing"], cfg["strengthSymmetry"])
        na = int(self.active.sum())
        sym = np.array(cfg["bsSolveSymmetryPoses"])[self.active]
        pairs = []
        for pid in set(sym.tolist()) - {-1}:
            ix = np.nonzero(sym == pid)[0]
            if len(ix) == 2:
                pairs.append(ix)
        Sm = np.zeros((len(pairs), na))
        for r, (i, j) in enumerate(pairs):
            Sm[r, i], Sm[r, j] = 1.0, -1.0
        self.AMat = (A.T @ A + L1 * L1 * 0.25 * s * np.ones((na, na)) + L2 * 10 * s * np.eye(na)
                     + Tr * 100 * s * np.eye(na) + Sy * 10 * s * Sm.T @ Sm)
        self.Tr = Tr
        self.mult = np.array(cfg["bsWeightMultipliers"], np.float64)
        self.off = np.array(cfg["bsWeightOffsets"], np.float64)
        Lc = np.linalg.cholesky(self.AMat)
        self.Lc = Lc
        self.reset()

    def reset(self):
        self.latents = np.zeros((2, 2, 1, 256), np.float32)
        self.w_prev = np.zeros(int(self.active.sum()))

    def _solve(self, b):
        """min 0.5 x'AMat x - b'x, 0<=x<=1 via BVLS on the Cholesky factor."""
        from scipy.optimize import lsq_linear
        Lc = self.Lc
        y = np.linalg.solve(Lc, b)
        r = lsq_linear(Lc.T, y, bounds=(0.0, 1.0), method="bvls", tol=1e-10)
        return r.x

    def run_window(self, win):
        feed = dict(window=win.reshape(1, 16000).astype(np.float32), identity=self.id_vec,
                    emotion=np.zeros((1, 30, 10), np.float32), input_latents=self.latents, noise=self.noise)
        pred, lat = self.sess.run(None, feed)
        self.latents = lat
        return pred[0]                                            # (60, NOUT)

    def solve_frame(self, skin_delta):
        pose = self.neutral_model + self.skin_strength * skin_delta.astype(np.float64) + self.pose_bias
        b = self.A.T @ (pose[self.pos] - self.bs_neutral_m) + self.Tr * self.s * self.w_prev
        x = self._solve(b)
        self.w_prev = x
        w = np.zeros(len(self.pose_names))
        w[self.active] = x
        return w * self.mult + self.off

    def solve_batch(self, skin_deltas, iters=400):
        """all frames at once: box-QP by FISTA (projected gradient). The SDK's temporal term only adds
        T*s*w_prev (~0.15 w_prev) to b against a T*100*s (~15) ridge: it is applied with w_prev from a first
        independent pass shifted by one 60 fps frame (same fixed point up to ~1e-3)."""
        X = skin_deltas.astype(np.float64)
        pose = self.neutral_model[self.pos][None] + self.skin_strength * X[:, self.pos] + self.pose_bias[self.pos][None]
        B0 = (pose - self.bs_neutral_m[None]) @ self.A
        Q = self.AMat
        Lmax = np.linalg.eigvalsh(Q).max()

        def qp(B):
            x = np.clip(np.linalg.solve(Q, B.T).T, 0, 1)
            y, t = x.copy(), 1.0
            for _ in range(iters):
                xn = np.clip(y - (y @ Q - B) / Lmax, 0, 1)
                tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
                y = xn + (t - 1) / tn * (xn - x)
                x, t = xn, tn
            return x
        x1 = qp(B0)
        prev = np.vstack([np.zeros((1, x1.shape[1])), x1[:-1]])
        x = qp(B0 + self.Tr * self.s * prev)
        W = np.zeros((len(X), len(self.pose_names)))
        W[:, self.active] = x
        return W * self.mult + self.off

    def run_track(self, x16, n30=None, keep_geometry=False, fast=True):
        """x16: float32 16 kHz audio of the track (track time 0 = x16[0]). Returns dict with
        w60 (n60, 52) ARKit weights in A2F pose order, jaw60 (n60, 15), eyes60 (n60, 4) at 60 fps; the canon
        30 fps frame k = 60 fps frame 2k."""
        self.reset()
        N = len(x16)
        n60 = int(np.ceil(N / 16000 * 60)) if n30 is None else 2 * int(n30)
        xp = np.concatenate([np.zeros(16000, np.float32), np.asarray(x16, np.float32), np.zeros(32000, np.float32)])
        W = np.zeros((n60, len(self.pose_names)), np.float32)
        S_ = np.zeros((n60, len(self.pos)), np.float32) if fast else None
        J = np.zeros((n60, JAW), np.float32)
        E = np.zeros((n60, EYES), np.float32)
        i = 0
        while True:
            first = 30 * i - 45                                   # 60 fps frame index of row 15 of run i
            if first >= n60:
                break
            win = xp[8000 * i: 8000 * i + 16000]
            pred = self.run_window(win)
            for jc in range(30):
                n = first + jc
                if 0 <= n < n60:
                    row = pred[15 + jc]
                    if fast:
                        S_[n] = row[:SKIN][self.pos]
                    else:
                        W[n] = self.solve_frame(row[:SKIN])
                    J[n] = row[SKIN + TONGUE: SKIN + TONGUE + JAW]
                    E[n] = row[SKIN + TONGUE + JAW:]
            i += 1
        if fast:
            W[:] = self.solve_batch_masked(S_)
        return dict(w60=W, jaw60=J, eyes60=E, pose_names=self.pose_names)

    def solve_batch_masked(self, Sm):
        """solve_batch on skin deltas already restricted to the frontal-mask positions."""
        full = np.zeros((len(Sm), SKIN), np.float32)
        out = []
        for a in range(0, len(Sm), 2048):
            full = np.zeros((len(Sm[a:a + 2048]), SKIN), np.float32)
            full[:, self.pos] = Sm[a:a + 2048]
            out.append(self.solve_batch(full))
        return np.concatenate(out) if out else np.zeros((0, len(self.pose_names)))


def to_canon(w60, pose_names, n30):
    """(n60, 52) A2F-order weights -> (n30, 52) canon (MediaPipe) order at 30 fps (frame k = 60 fps frame 2k).
    _neutral (MediaPipe only) and tongueOut (A2F only) are not mapped: canon _neutral column = 0."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from canon1 import spec as S
    out = np.zeros((n30, 52), np.float32)
    src = w60[0: 2 * n30: 2]
    for j, n in enumerate(pose_names):
        if n in S.BS:
            out[: len(src), S.BS[n]] = src[:, j]
    return out
