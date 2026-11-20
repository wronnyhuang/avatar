"""FLOW streaming runtime — pure numpy (no scipy, no torch). CONTRACT stream API:
    gen = Generator(model_dir, seed); gen.reset(); gen.push_audio(feats (n,42)); gen.set_state(speaking, partner_speaking)
    pose = gen.step()   # once per 30 fps frame -> (189,) canon (body cols 0:135 filled, face cols at neutral)
Model: tokenizer (causal patch AE, P=4) + AR latent rectified-flow denoiser (24 history tokens + C=8 window tokens).
Amortised schedule: a window of 32 frames is generated while the previous K committed frames play; the window's compute is
split into layer-sized units and spread evenly over those K step() calls (deterministic: the unit split does not depend on
wall time). Own audio is known up to whatever has been pushed (the model was trained with look-ahead -40..+30 frames).
"""
import json
import os

import numpy as np

P = 4


# ------------------------------------------------------------------------------------------------ math
def ln(x, w=None, b=None, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) / np.sqrt(v + eps)
    if w is not None:
        y = y * w + b
    return y


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * (x * x * x))))


def silu(x):
    return x / (1 + np.exp(-x))


def aa_to_mat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    th = np.maximum(th, 1e-8)
    k = aa / th
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3, dtype=aa.dtype) + s * K + (1 - c) * (K @ K)


def mat_to_aa(R):
    """(n,3,3) rotation matrices -> (n,3) axis-angle (quaternion route, Shepperd branch choice; no SVD)."""
    R = np.asarray(R, np.float64)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    cand = np.stack([tr, m00, m11, m22], 1)
    k = cand.argmax(1)
    q = np.empty((len(R), 4))
    for b in range(4):
        i = k == b
        if not i.any():
            continue
        r = R[i]
        if b == 0:
            t = 1 + tr[i]
            q[i] = np.stack([t, r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], 1)
        elif b == 1:
            t = 1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]
            q[i] = np.stack([r[:, 2, 1] - r[:, 1, 2], t, r[:, 0, 1] + r[:, 1, 0], r[:, 0, 2] + r[:, 2, 0]], 1)
        elif b == 2:
            t = 1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2]
            q[i] = np.stack([r[:, 0, 2] - r[:, 2, 0], r[:, 0, 1] + r[:, 1, 0], t, r[:, 1, 2] + r[:, 2, 1]], 1)
        else:
            t = 1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]
            q[i] = np.stack([r[:, 1, 0] - r[:, 0, 1], r[:, 0, 2] + r[:, 2, 0], r[:, 1, 2] + r[:, 2, 1], t], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)                     # w >= 0 -> angle in [0, pi]
    v = q[:, 1:]
    sn = np.linalg.norm(v, axis=1)
    ang = 2 * np.arctan2(sn, q[:, 0])
    scale = np.where(sn > 1e-9, ang / np.maximum(sn, 1e-12), 2.0)
    return v * scale[:, None]


def m_to_6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def d6_to_m(x):
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


# ------------------------------------------------------------------------------------------------ nets (numpy)
class Net:
    def __init__(self, W, prefix):
        self.W, self.p = W, prefix

    def g(self, k):
        return self.W[self.p + k]

    def lin(self, x, k):
        return x @ self.g(k + '.weight').T + self.g(k + '.bias')

    def attn(self, y, blk, mask, heads, q0=0):
        """y (N,D) or (B,N,D); only query rows q0: are computed (keys/values use all rows)"""
        sq = y.ndim == 2
        if sq:
            y = y[None]
        B, N, D = y.shape
        qkv = self.lin(y, blk + 'qkv').reshape(B, N, 3, heads, D // heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0][:, :, q0:], qkv[1], qkv[2]            # (B,h,Nq,dh), (B,h,N,dh)
        a = (q * float(1.0 / np.sqrt(D // heads))) @ k.transpose(0, 1, 3, 2)
        a = np.where(mask[..., q0:, :], a, np.float32(-1e30))
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, N - q0, D)
        o = self.lin(o, blk + 'o')
        return o[0] if sq else o

    def block(self, x, i, mask, heads, mv=None, name='blocks', q0=0):
        """q0 > 0: only rows q0: are needed downstream (last layer) -> attention queries + FFN on those rows only."""
        blk = f'{name}.{i}.'
        if mv is not None:                      # AdaLN modulation vector (6*D) for this block at this flow time
            s1, b1, g1, s2, b2, g2 = np.split(mv, 6)
            y = ln(x) * (1 + s1) + b1
            x = x[..., q0:, :] + g1 * self.attn(y, blk, mask, heads, q0)
            y = ln(x) * (1 + s2) + b2
            return x + g2 * self.lin(gelu(self.lin(y, blk + 'f1')), blk + 'f2')
        y = ln(x, self.g(blk + 'n1.weight'), self.g(blk + 'n1.bias'))
        x = x[..., q0:, :] + self.attn(y, blk, mask, heads, q0)
        y = ln(x, self.g(blk + 'n2.weight'), self.g(blk + 'n2.bias'))
        return x + self.lin(gelu(self.lin(y, blk + 'f1')), blk + 'f2')


    def block_g(self, x, i, mask, heads, mv=None, name='blocks', q0=0):
        """same as block() as a generator: yields once between the attention and the FFN half (finer work units ->
        steadier per-frame cost); `x = yield from net.block_g(...)`."""
        blk = f'{name}.{i}.'
        if mv is not None:
            s1, b1, g1, s2, b2, g2 = np.split(mv, 6)
            x = x[..., q0:, :] + g1 * self.attn(ln(x) * (1 + s1) + b1, blk, mask, heads, q0)
            yield
            y = ln(x) * (1 + s2) + b2
            return x + g2 * self.lin(gelu(self.lin(y, blk + 'f1')), blk + 'f2')
        x = x[..., q0:, :] + self.attn(ln(x, self.g(blk + 'n1.weight'), self.g(blk + 'n1.bias')), blk, mask, heads, q0)
        yield
        y = ln(x, self.g(blk + 'n2.weight'), self.g(blk + 'n2.bias'))
        return x + self.lin(gelu(self.lin(y, blk + 'f1')), blk + 'f2')


def causal_mask(n, window):
    i = np.arange(n)
    return (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)


class Generator:
    def __init__(self, model_dir, seed=0, steps=4, commit=8, xfade=4, temp=1.0, cfg=1.0, eta=0.0, hist=12, deglitch=True, inert=6, dg=1.0, clip=2.5):
        Z = np.load(os.path.join(model_dir, 'flow_weights.npz'))
        self.W = {}
        for k in Z.files:
            if k == 'cfg_json' or k.endswith('.__s'):
                continue
            v = Z[k]
            if v.dtype == np.int8:                 # int8 matrix, one scale per (row, column group)
                sc = Z[k + '.__s'].astype(np.float32)
                v = (v.astype(np.float32).reshape(v.shape[0], sc.shape[1], -1) * sc[..., None]).reshape(v.shape)
            self.W[k] = np.ascontiguousarray(v.astype(np.float32))      # fp16 / int8 storage -> fp32 compute
        self.cfg = json.loads(bytes(Z['cfg_json']).decode())
        self.HM, self.C, self.L = self.cfg['hmax'], self.cfg['C'], self.cfg['layers']
        self.tl = self.cfg['tok']['layers']
        self.tok = Net(self.W, 'tok.')
        self.den = Net(self.W, 'den.')
        S = {k[2:]: v for k, v in self.W.items() if k.startswith('S.')}
        self.S = S
        self.steps, self.K, self.xfade, self.temp = steps, commit * P, xfade, temp
        self.seed, self.cfgw, self.eta, self.hist = seed, float(cfg), float(eta), int(hist)
        self.deglitch, self.inert, self.dg, self.clip = bool(deglitch), int(inert), float(dg), float(clip)
        if 'den.fix.t' in self.W and (self.steps != len(self.W['den.fix.t']) or self.cfgw != 1.0):
            raise ValueError(f"this package precomputes a {len(self.W['den.fix.t'])}-step grid without CFG")
        self.neutral = self.W['neutral']
        self.silence = self.W['silence']
        self.reset()

    # -------------------------------------------------------------- feature helpers
    def _feats(self, H):
        """history dict (Rr (T,3,3), sh (T,3), Rb (T,13,3,3), fp (T,24)) -> root9 (T,9) canon at the last frame,
        body102 normalised (T,102), Ry(yaw), shift_ref.  (History is kept as matrices: no axis-angle round trip.)"""
        Rr, FP = H['Rr'], H['fp']
        R0 = Rr[-1]
        yaw = np.arctan2(R0[0, 2], R0[2, 2])
        Rm = ry(-yaw)
        Rr = Rm[None] @ Rr
        sh = (H['sh'] - H['sh'][-1]) @ Rm.T
        f = np.concatenate([m_to_6d(Rr), sh, m_to_6d(H['Rb']).reshape(len(Rr), 78), FP], 1).astype(np.float32)
        return f[:, :9], (f[:, 9:] - self.S['mu'][9:]) / self.S['sd'][9:], ry(yaw), H['sh'][-1].copy()

    # -------------------------------------------------------------- the window job (generator of units)
    def _job(self, H, own, own_known, pflag, pknown):
        S, HM, C, W = self.S, self.HM, self.C, self.W
        nh = len(H['sh']) // P
        Th = nh * P
        root9, xb, Ryaw, sh0 = self._feats(H)
        # encoder (causal, window)
        h = xb.reshape(nh, P * xb.shape[1]) @ W['tok.ei.weight'].T + W['tok.ei.bias']
        m = causal_mask(nh, 40)
        for i in range(self.tl):
            h = yield from self.tok.block_g(h, i, m, 8, name='eb')
            yield
        z = ln(h, W['tok.en.weight'], W['tok.en.bias']) @ W['tok.eo.weight'].T + W['tok.eo.bias']
        rt = root9.reshape(nh, 36)
        xh = np.concatenate([(rt - S['rmu']) / S['rsd'], (z - S['zmu']) / S['zsd']], 1)
        hist = np.zeros((HM, 68), np.float32); hist[HM - nh:] = xh
        hvalid = np.zeros(HM + C, bool); hvalid[HM - nh:] = True
        T = (HM + C) * P
        cond = np.zeros((T, 88), np.float32)
        o0 = HM * P - Th
        nk = min(own_known, T - o0)
        if nk > 0:
            cond[o0:o0 + nk, 0:42] = (own[:nk] - S['amu']) / S['asd']
            cond[o0:o0 + nk, 42] = 1
        if pknown:
            cond[HM * P:, 86] = pflag
            cond[HM * P:, 87] = 1
            cond[o0:HM * P, 86] = pflag          # live: no partner audio, only the current state flag
            cond[o0:HM * P, 87] = 1
        cond = cond.reshape(HM + C, P * 88)
        flag = np.concatenate([np.ones((HM, 1), np.float32), np.zeros((C, 1), np.float32)])
        base = np.concatenate([flag, cond], 1)                       # (32, 1+352)
        Win = W['den.inp.weight']
        pre = base @ Win[:, 68:].T + W['den.inp.bias'] + W['den.pos'][0]
        hpre = hist @ Win[:, :68].T
        # only the valid rows are computed (key padding == dropping the padded tokens; positions keep their embeddings)
        lo = HM - nh
        pre, hpre = pre[lo:], hpre[lo:]
        HM = nh
        mask = np.ones((nh + C, nh + C), bool)
        x = self.rng.standard_normal((C, 68)).astype(np.float32) * self.temp
        ts = np.linspace(0, 1, self.steps + 1)
        # (python floats below: numpy-2 float64 scalars would silently upcast the float32 state)
        half = 128
        fr = np.exp(-np.log(10000) * np.arange(half) / half)
        if self.cfgw != 1.0:          # classifier-free guidance: cond / uncond rows stacked as a batch of 2
            ubase = np.concatenate([flag, np.zeros_like(cond)], 1)
            pre = np.stack([pre, (ubase @ Win[:, 68:].T + W['den.inp.bias'] + W['den.pos'][0])[lo:]])
        for si in range(self.steps):
            t0, t1 = float(ts[si]), float(ts[si + 1])
            mvs, fm = self._mods(t0, fr)
            hh = pre.copy()
            hh[..., :HM, :] += hpre
            hh[..., HM:, :] += x @ Win[:, :68].T
            for i in range(self.L):
                hh = yield from self.den.block_g(hh, i, mask, 8, mv=mvs[i], q0=HM if i == self.L - 1 else 0)
                yield
            s_, b_ = np.split(fm, 2)
            x0 = (ln(hh) * (1 + s_) + b_) @ W['den.out.weight'].T + W['den.out.bias']
            if self.cfgw != 1.0:
                x0 = x0[1] + self.cfgw * (x0[0] - x0[1])
            if self.eta > 0 and si < self.steps - 1:
                eh = (x - t0 * x0) / max(1 - t0, 1e-4)
                en = self.eta * self.rng.standard_normal(x.shape).astype(np.float32) + float(np.sqrt(1 - self.eta ** 2)) * eh
                x = (t1 * x0 + (1 - t1) * en).astype(np.float32)
            else:
                x = x + (x0 - x) * float((t1 - t0) / max(1 - t0, 1e-4))
        root = x[:, :36] * S['rsd'] + S['rmu']
        zw = x[:, 36:] * S['zsd'] + S['zmu']
        zz = np.concatenate([z, zw], 0)
        n = len(zz)
        h = zz @ W['tok.di.weight'].T + W['tok.di.bias']
        m = causal_mask(n, 40)
        for i in range(self.tl):
            h = yield from self.tok.block_g(h, i, m, 8, name='db', q0=nh if i == self.tl - 1 else 0)
            yield
        body = (ln(h, W['tok.dn.weight'], W['tok.dn.bias']) @ W['tok.do.weight'].T + W['tok.do.bias']).reshape(C * P, 102)
        body = body * S['sd'][9:] + S['mu'][9:]
        rf = root.reshape(C * P, 9)
        Rr = Ryaw[None] @ d6_to_m(rf[:, 0:6])
        sh = rf[:, 6:9] @ Ryaw.T + sh0
        Rb = d6_to_m(body[:, :78].reshape(-1, 13, 6))
        self._result = dict(Rr=Rr, sh=sh, Rb=Rb, fp=body[:, 78:102])
        yield

    def _mods(self, t0, fr):
        """AdaLN modulation vectors at flow time t0: from the package's precomputed table (export --fixed-steps; the
        time-MLP and modulation matrices are then not shipped) or computed from the weights."""
        W = self.W
        if 'den.fix.t' in W:
            j = np.nonzero(np.abs(W['den.fix.t'] - t0) < 1e-6)[0]
            if not len(j):
                raise ValueError(f'package has precomputed flow times {W["den.fix.t"]} only; use steps={len(W["den.fix.t"])}')
            j = int(j[0])
            return [W[f'den.fix.mod.{i}'][j] for i in range(self.L)], W['den.fix.fmod'][j]
        a = t0 * 1000.0 * fr
        te = np.concatenate([np.sin(a), np.cos(a)]).astype(np.float32)
        c = silu(te @ W['den.te.0.weight'].T + W['den.te.0.bias']) @ W['den.te.2.weight'].T + W['den.te.2.bias']
        mvs = [c @ W[f'den.blocks.{i}.mod.weight'].T + W[f'den.blocks.{i}.mod.bias'] for i in range(self.L)]
        return mvs, c @ W['den.fmod.weight'].T + W['den.fmod.bias']

    def _units(self):
        return 2 * (self.tl + self.steps * self.L + self.tl) + 1

    # -------------------------------------------------------------- API
    def reset(self, pre_motion=None, pre_feats=None):
        """pre_motion (n,189) canon / pre_feats (n,42): optional real pre-roll (offline scoring, or a hand-off from another motion
        source); otherwise the history is 8 frames of the neutral pose."""
        self.rng = np.random.default_rng(self.seed)
        self.pre_own = None if pre_feats is None else np.asarray(pre_feats, np.float32)
        nb = self.neutral
        fl = (nb[45:90] - self.W['pca.mu_l']) @ self.W['pca.V_l'].T
        fr = (nb[90:135] - self.W['pca.mu_r']) @ self.W['pca.V_r'].T
        nb64 = nb.astype(np.float64)                                        # committed history (2 tokens of rest pose)
        self.H = dict(Rr=np.repeat(aa_to_mat(nb64[0:3])[None], 8, 0).astype(np.float32),
                      sh=np.repeat(nb[None, 3:6], 8, 0).astype(np.float32),
                      Rb=np.repeat(aa_to_mat(nb64[6:45].reshape(13, 3))[None], 8, 0).astype(np.float32),
                      fp=np.repeat(np.concatenate([fl, fr])[None], 8, 0).astype(np.float32))
        if pre_motion is not None:
            pm = np.asarray(pre_motion, np.float64)
            pm = pm[len(pm) % P:]
            self.H = dict(Rr=aa_to_mat(pm[:, 0:3]).astype(np.float32), sh=pm[:, 3:6].astype(np.float32),
                          Rb=aa_to_mat(pm[:, 6:45].reshape(-1, 13, 3)).astype(np.float32),
                          fp=np.concatenate([(pm[:, 45:90] - self.W['pca.mu_l']) @ self.W['pca.V_l'].T,
                                             (pm[:, 90:135] - self.W['pca.mu_r']) @ self.W['pca.V_r'].T], 1).astype(np.float32))
        self.own = []                                                       # own audio frames, index 0 = first played frame
        self.speaking, self.partner = False, False
        self.play = 0                                                       # frames already returned by step()
        # first window crossfades out of the held last history frame (no pop at a hand-off)
        self.prev = {k: np.repeat(v[-1:], max(1, self.xfade), 0) for k, v in self.H.items()} if self.xfade else None
        self.queue = []
        self.job = None                         # the first window is computed synchronously by the first step()

    def seed_history(self, poses):
        """Hand-over from another motion source: poses (n,189) = the last n canon poses actually shown (oldest first; n >= 4,
        48 = the model's full live history). Replaces the history, drops queued/planned frames; the NEXT step() computes a window
        synchronously from it (one-off cost ~ one window, 60-110 ms on a hub core) and crossfades out of poses[-1].
        Audio pushed so far and the frame counter are kept (call it at the hand-over frame, before that frame's step())."""
        pm = np.asarray(poses, np.float64)
        pm = pm[len(pm) % P:][-self.HM * P:]
        if len(pm) < P:
            raise ValueError('seed_history needs >= 4 poses')
        self.H = dict(Rr=aa_to_mat(pm[:, 0:3]).astype(np.float32), sh=pm[:, 3:6].astype(np.float32),
                      Rb=aa_to_mat(pm[:, 6:45].reshape(-1, 13, 3)).astype(np.float32),
                      fp=np.concatenate([(pm[:, 45:90] - self.W['pca.mu_l']) @ self.W['pca.V_l'].T,
                                         (pm[:, 90:135] - self.W['pca.mu_r']) @ self.W['pca.V_r'].T], 1).astype(np.float32))
        self.prev = {k: np.repeat(v[-1:], max(1, self.xfade), 0) for k, v in self.H.items()} if self.xfade else None
        self.queue = []
        self.job = None

    def push_audio(self, feats):
        for f in np.asarray(feats, np.float32).reshape(-1, 42):
            self.own.append(f)

    def set_state(self, speaking, partner_speaking):
        self.speaking, self.partner = bool(speaking), bool(partner_speaking)

    @staticmethod
    def _pack(d):
        n = len(d['sh'])
        return np.concatenate([m_to_6d(d['Rr']), d['sh'], m_to_6d(d['Rb']).reshape(n, 78), d['fp']], 1).astype(np.float64)

    @staticmethod
    def _unpack(X):
        n = len(X)
        return dict(Rr=d6_to_m(X[:, 0:6]).astype(np.float32), sh=X[:, 6:9].astype(np.float32),
                    Rb=d6_to_m(X[:, 9:87].reshape(n, 13, 6)).astype(np.float32), fp=X[:, 87:111].astype(np.float32))

    def _deglitch(self, w):
        """v1.1: remove the velocity STEP at every 4-frame patch boundary of the decoded window (the tokenizer decoder emits
        4 frames per token; the residual step there made single-frame head+wrist spikes). At an interior boundary b the excess
        step r = (x[b]-x[b-1]) - mean(neighbour velocities) is spread as a linear ramp over frames b-2..b+1; a motion whose
        velocity is continuous across b (a real stroke) has r ~ 0 and is untouched. The window START is inertialised
        against the committed history: the excess r0 vs the last committed velocity decays over `inert` frames.
        Works on the whole window before any of it plays -> no added latency."""
        X = self._pack(w)
        n = len(X)
        ramp = np.array([0.2, 0.4, -0.4, -0.2])
        for b in range(P, n - 1, P):
            r = (X[b] - X[b - 1]) - 0.5 * ((X[b - 1] - X[b - 2]) + (X[b + 1] - X[b]))
            if self.clip > 0:     # only the part of the excess beyond `clip` x the local frame-to-frame motion is removed
                lo, hi = max(b - 4, 0), min(b + 3, n - 1)
                d = np.abs(np.diff(X[lo:hi + 1], axis=0))
                d = np.delete(d, b - 1 - lo, axis=0)
                m = np.median(d, axis=0)
                r = np.sign(r) * np.maximum(np.abs(r) - self.clip * m, 0.0)
            X[b - 2:b + 2] += (self.dg * ramp)[:, None] * r[None]
        if self.inert and len(self.H['sh']) >= 2:
            Hc = self._pack({k: v[-2:] for k, v in self.H.items()})
            r0 = (X[0] - Hc[1]) - 0.5 * ((Hc[1] - Hc[0]) + (X[1] - X[0]))
            N = self.inert
            X[:N] -= (1 - (np.arange(N) + 1) / (N + 1))[:, None] * r0[None]
        return self._unpack(X)

    def _commit(self):
        w = self._result
        if self.deglitch:
            w = self._deglitch(w)
        if self.prev is not None and self.xfade:
            n = min(self.xfade, len(self.prev['sh']))
            wt = ((np.arange(n) + 1) / (n + 1)).astype(np.float32)

            def lerp6(a, b, w):
                x = a[..., :, :2] * (1 - w) + b[..., :, :2] * w
                return d6_to_m(np.concatenate([x[..., :, 0], x[..., :, 1]], -1))
            w = {k: v.copy() for k, v in w.items()}
            w['Rr'][:n] = lerp6(self.prev['Rr'][:n], w['Rr'][:n], wt[:, None, None])
            w['Rb'][:n] = lerp6(self.prev['Rb'][:n], w['Rb'][:n], wt[:, None, None, None])
            w['sh'][:n] = self.prev['sh'][:n] * (1 - wt[:, None]) + w['sh'][:n] * wt[:, None]
            w['fp'][:n] = self.prev['fp'][:n] * (1 - wt[:, None]) + w['fp'][:n] * wt[:, None]
        K = self.K
        self.prev = {k: v[K:] for k, v in w.items()}
        keep = self.HM * P + 64
        for k in self.H:
            self.H[k] = np.concatenate([self.H[k], w[k][:K].astype(np.float32)])[-keep:]
        for i in range(K):                      # axis-angle conversion happens per frame in step()
            self.queue.append((w['Rr'][i], w['sh'][i], w['Rb'][i], w['fp'][i]))
        self.job = None

    def step(self):
        if not self.queue:                      # start-up (or starvation): finish a window synchronously
            if self.job is None:
                self._start_job_rel()
            for _ in self.job:
                pass
            self._commit()
        # start the next window as soon as the queue holds exactly one window's worth (it is computed while those K play)
        if self.job is None:
            self._start_job_rel()
        per = -(-self.units_left // max(1, len(self.queue)))                 # ceil: finish before the queue runs dry
        for _ in range(per):
            try:
                next(self.job)
                self.units_left -= 1
            except StopIteration:
                break
        if self.units_left <= 0 and self.job is not None:
            for _ in self.job:
                pass
            self._commit()
        Rr, sh, Rb, f = self.queue.pop(0)
        self.play += 1
        out = self.neutral.copy()
        aa = mat_to_aa(np.concatenate([Rr[None], Rb]))
        out[0:3] = aa[0]
        out[3:6] = sh
        out[6:45] = aa[1:].reshape(39)
        out[45:90] = self.W['pca.mu_l'] + f[:12] @ self.W['pca.V_l']
        out[90:135] = self.W['pca.mu_r'] + f[12:] @ self.W['pca.V_r']
        return out

    def _start_job_rel(self):
        # output-frame index of the next window start = frames played + frames queued
        s = self.play + len(self.queue)
        HM = self.HM
        n = min(len(self.H['sh']) // P, HM, self.hist) * P
        Hn = {k: v[-n:] for k, v in self.H.items()}
        if not self.speaking:
            while len(self.own) < self.play + 1:
                self.own.append(self.silence)
        po = self.pre_own
        rows = [self.own[i] if i >= 0 else (po[len(po) + i] if po is not None and len(po) + i >= 0 else self.silence)
                for i in range(s - n, min(len(self.own), s + 32))]
        own = np.stack(rows).astype(np.float32)
        self.job = self._job(Hn, own, len(rows), 1.0 if self.partner else 0.0, 1)
        self.units_left = self._units()
