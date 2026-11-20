"""mm/stream.py — conversational MOTION MATCHING on real (Seamless) human motion. Pure numpy. ( mm)

    gen = Generator(model_dir, seed=0)        # model_dir holds db.npz (or pass a path to the .npz)
    gen.reset()
    gen.push_audio(feats)                     # (n,42) float32 canon audio of the avatar's OWN voice, 30 fps, may run ahead
    gen.set_state(speaking, partner_speaking)
    pose = gen.step()                         # (189,) float32 canon; body cols 0:135 filled, 135:189 = neutral face/gaze
    gen.info                                  # {"energy": 0..1, "arm_l": 0..1, "arm_r": 0..1, "state": int, "jumps": n}

Every frame it plays a real DB frame; every `search_every` frames (or on a state change / segment end) it searches the
DB (brute-force matvec over the state's partition) for the frame whose (wrist/elbow pose + velocity, own-voice
loudness/onset/vad/spectral-centroid window over past + available look-ahead) best matches, penalising recently
played regions, and jumps there with INERTIALIZATION (critically damped offset decay), or cross-fade (ablation).
Head/torso cols are neutral + a zero-mean deviation (DB is high-passed); arms/fingers are absolute, rest = neutral.
"""
import json
import math
import os

import numpy as np

AUD_OFFS = np.array([-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 24, 30])   # frames relative to the jump frame
FPS = 30


def _rodr(v):
    """(n,3) axis-angle -> (n,3,3)"""
    th = np.linalg.norm(v, axis=1)[:, None, None] + 1e-12
    k = v / th[:, :, 0]
    K = np.zeros((len(v), 3, 3), v.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    I = np.eye(3, dtype=v.dtype)[None]
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _m2aa(R, ref):
    """(n,3,3) -> (n,3) axis-angle, representative closest to ref (n,3) (keeps the neutral's hemisphere)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    sn = np.sin(th)
    ax = w / np.maximum(2 * sn, 1e-9)[:, None]
    near_pi = sn < 1e-4
    if near_pi.any():   # fallback: axis from the diagonal
        for i in np.nonzero(near_pi & (th > 1))[0]:
            d = np.sqrt(np.clip((np.diag(R[i]) + 1) / 2, 0, None))
            j = int(np.argmax(d)); a = R[i, j] + R[i, :, j]; ax[i] = a / (np.linalg.norm(a) + 1e-12)
    v = ax * th[:, None]
    alt = v - 2 * np.pi * ax
    use = np.linalg.norm(alt - ref, axis=1) < np.linalg.norm(v - ref, axis=1)
    v[use] = alt[use]
    return v


def _deq(d, name, g0):
    """int16/int32 deltas (restarting at chunk starts g0, start rows in <name>_k0) * step + mean -> float32 (low peak memory)."""
    dl = d.pop(name + "_dq")
    k0 = d.pop(name + "_k0", None)
    step = np.asarray(d.pop(name + "_step"), np.float32); mu = np.asarray(d.pop(name + "_mu"), np.float32)
    out = np.empty(dl.shape, np.float32)
    edges = list(g0) + [len(dl)] if k0 is not None else [0, len(dl)]
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        q = np.cumsum(dl[a:b], axis=0, dtype=np.int32)
        if k0 is not None:
            q += (k0[i] - dl[a]).astype(np.int32)       # row a carries no delta: its absolute value is k0[i]
        out[a:b] = q * step + mu
    return out


def _unpack(z):
    """packed db (pack_db.py: int16 time-delta quantised body + wrist/elbow positions) -> the arrays build_db.py wrote."""
    if "body_dq" not in z.files:
        return z
    d = {k: z[k] for k in z.files}
    g0 = np.array([c["g0"] for c in json.loads(str(d["meta"]))["chunks"]])

    d["body"] = _deq(d, "body", g0)
    pos = _deq(d, "pos", g0)
    N = len(pos); b = d["body"]
    pf = np.zeros((N, 24), np.float32); pf[:, :12] = pos
    act = np.zeros((N, 2), np.float32)
    meta = json.loads(str(d["meta"]))
    for c in meta["chunks"]:
        a, e = c["g0"], c["g1"]
        vl = np.gradient(pos[a:e, 0:3], axis=0) * 30; vr = np.gradient(pos[a:e, 3:6], axis=0) * 30
        hv = np.gradient(b[a:e, 12:21], axis=0) * 30
        pf[a:e, 12:15], pf[a:e, 15:18], pf[a:e, 18:21], pf[a:e, 21:24] = vl, vr, hv[:, 6:9], hv[:, 3:6]
        for k, v in enumerate((vl, vr)):
            sp = np.pad(np.linalg.norm(v, axis=1), 4, mode="symmetric")
            act[a:e, k] = np.convolve(sp, np.ones(9) / 9.0, "valid")
    d["pfeat"] = pf; d["act"] = act

    class _D(dict):
        files = property(lambda self: list(self.keys()))
    return _D(d)


def _hl_damp(h):
    return (4.0 * 0.69314718056) / (h + 1e-5)


class Generator:
    def __init__(self, model_dir, seed=0, **kw):
        """kwargs default from <model_dir>/config.json (the frozen live tuning) when model_dir is a folder; explicit kwargs win."""
        cfg = {}
        if not str(model_dir).endswith(".npz"):
            cp = os.path.join(model_dir, "config.json")
            if os.path.isfile(cp):
                cfg = {k: v for k, v in json.load(open(cp)).items() if not k.startswith("_")}
        cfg.update(kw)
        self.config = dict(cfg)
        self._init(model_dir, seed, **cfg)

    def _init(self, model_dir, seed=0, lookahead_s=1.0, audio_w=1.0, blend="inert", halflife=0.12,
                 search_every=8, switch_bias=0.25, rep_w=3.0, rep_T_s=900.0, xfade_n=8, tracks=None, cand_stride=2, search_slices=3,
                 energy=0.0, energy_w=0.0, loud_ref=-47.0, pose_wts=(1.0, 0.4, 0.6, 0.25),
                 hand_ymax=None, hand_spread_max=None, hand_guard_s=1.5,
                 onset_trig=None, trig_lead=9, trig_refr=15, arm_gain=1.0, arm_gain_listen=1.0, wrist_clamp=0.3,
                 hand_soft_y=None, hand_soft_w=0.0, speak_still_w=0.0, speak_e_lo=-0.5,
                 rest_poses=None, rest_every_s=(20.0, 60.0), rest_blend_s=1.5, aud_renorm=1.0,
                 past_long=False, trig_bias=None, trig_aw=1.0, rest_init_random=False, rest_speak_ok=(1, 1, 0, 0), acc_lim=None, halflife_quiet=None,
                 pause_s=0.3, pause_w=0.0, pause_e=-0.5, arm_gain_pause=None, rest_return_s=3.0):
        path = model_dir if str(model_dir).endswith(".npz") else os.path.join(model_dir, "db.npz")
        d = _unpack(np.load(path, allow_pickle=False))
        self.seed = seed
        self.la_cap = int(round(lookahead_s * FPS))
        self.audio_w = audio_w
        self.blend = blend
        self.halflife = halflife
        self.search_every = search_every
        self.search_slices = search_slices
        self.switch_bias = switch_bias
        self.rep_w = rep_w
        self.rep_T = rep_T_s * FPS
        self.xfade_n = xfade_n
        self.aud_renorm = float(aud_renorm)
        # v4 PAUSES (lead 05:55): speaking state but own voice off for > pause_s -> searches prefer calm continuations
        # (cost pause_w per z-unit of next-1 s energy above pause_e), the speaking 'liveliness' cost is off
        self.arm_gain_pause = arm_gain_pause
        self.rest_return_n = int(round(rest_return_s * FPS))
        self.pause_n = int(round(pause_s * FPS)); self.pause_w = float(pause_w); self.pause_e = float(pause_e)
        self.halflife_quiet = None if halflife_quiet is None else float(halflife_quiet)   # v4: slower blends while listening/idle
        # v4 acceleration limiter on the arm joints (cols 21:45; rad/s^2 per collar/shoulder/elbow/wrist pair): a real arm cannot
        # change velocity in one frame; real held-out p99.9 is ~10-12 / 12-15 / 24-31 / 110-160 rad/s^2
        self.acc_lim = None if acc_lim is None else np.repeat(np.asarray(acc_lim, np.float32), 6) / (FPS * FPS)
        # causal design for ~0.1 s look-ahead (lead 03:35): past_long adds 0.6-1.5 s of past prosody to the audio window;
        # a voice-onset trigger searches at once (one slice) with audio weight x trig_aw and switch bias trig_bias
        self.offs = np.r_[[-45, -36, -27, -18], AUD_OFFS] if past_long else AUD_OFFS.copy()
        self.trig_bias = switch_bias if trig_bias is None else float(trig_bias); self.trig_aw = float(trig_aw)
        self.wrist_clamp = float(wrist_clamp)   # rad/frame per axis-angle component on the wrists (tracker twist bursts)
        # event-driven search: a voice onset (stress) seen `trig_lead` frames ahead (or as far as the look-ahead reaches)
        # triggers a search, so new gesture segments start on emphasis instead of on a clock
        # arm amplitude gain about the rest pose (seated Seamless speakers gesture smaller than a standing teacher):
        # shoulder+elbow rotation deviation from neutral x gain, eased per state (tau 0.4 s) so it never pops
        # alternative REST poses (lead 03:14 (d)): arms re-homed by pre-multiplying each arm joint (collars..wrists, cols 21:45)
        # with R_rest R_neutral^T; drifts to another rest every rest_every_s at a calm moment (never while gesturing), min-jerk over rest_blend_s
        self.rests = None
        if rest_poses is not None:
            rp = rest_poses
            if isinstance(rp, str) and rp == "db":
                rp = d["rests"] if "rests" in d.files else None          # rests shipped inside db.npz (v2)
            elif isinstance(rp, str):
                fp = rp if os.path.isabs(rp) else os.path.join(model_dir if not str(model_dir).endswith(".npz") else os.path.dirname(model_dir), rp)
                rp = np.load(fp) if os.path.isfile(fp) else None        # missing file -> rests off, never a crash
        if rest_poses is not None and rp is not None:
            rp = np.asarray(rp, np.float64)
            nn = d["neutral"].astype(np.float64)[21:45].reshape(8, 3)
            Rn = _rodr(nn)
            D = []
            for r in rp:
                Dk = _rodr(r[21:45].reshape(8, 3)) @ np.transpose(Rn, (0, 2, 1))
                D.append(_m2aa(Dk, np.zeros((8, 3))))
            self.rests = np.array(D)            # (K,8,3) premultipliers as axis-angle (row 0 = identity)
            self.rest_every = tuple(float(x) for x in rest_every_s); self.rest_blend = float(rest_blend_s)
        self.rest_init_random = bool(rest_init_random)
        # rests that suit TALKING (hands in front); others (arms at sides) ease out while she speaks (0.4 s) and back in calm (1 s)
        self.rest_speak_ok = np.asarray(rest_speak_ok, bool)
        self.arm_gain = float(arm_gain); self.arm_gain_listen = float(arm_gain_listen)
        self.onset_trig = onset_trig; self.trig_lead = int(trig_lead); self.trig_refr = int(trig_refr)
        b = np.asarray(d["body"], np.float32)
        N = len(b)
        pose = np.empty((N, 135), np.float32)
        pose[:, :45] = b[:, :45]
        nc = d["fpc_l"].shape[0]
        pose[:, 45:90] = d["fmu_l"] + b[:, 45:45 + nc] @ d["fpc_l"]
        pose[:, 90:135] = d["fmu_r"] + b[:, 45 + nc:45 + 2 * nc] @ d["fpc_r"]
        self.pose = pose.astype(np.float16)   # float16 in memory (DB is stored float16 anyway)
        del pose, b
        self.neutral = d["neutral"].astype(np.float32)
        self.state_db = d["state"].astype(np.int8)
        self.seg_end = d["seg_end"].astype(np.int64)
        cand = d["cand"].astype(bool)
        if tracks is not None:
            cand &= np.isin(d["track"], np.asarray(tracks))
        # hand-height / width guard (Juno look 02:35: hands up at the collar read as clutching): frames whose wrist is above
        # hand_ymax (m over pelvis; neutral chin 0.48, shoulder 0.40) or wider than hand_spread_max are never played:
        # candidates need hand_guard_s clean seconds ahead, and playback re-searches early when a bad frame comes near.
        pf0 = d["pfeat"].astype(np.float32)
        badh = np.zeros(N, bool)
        if hand_ymax is not None:
            badh |= np.maximum(pf0[:, 1], pf0[:, 4]) > hand_ymax
        if hand_spread_max is not None:
            badh |= (pf0[:, 0] - pf0[:, 3]) > hand_spread_max
        G = int(hand_guard_s * FPS)
        cs = np.convolve(badh.astype(np.float32), np.ones(G, np.float32), "full")[G - 1:G - 1 + N]
        cand_play = cand.copy()
        cand = cand & (cs == 0)
        cs6 = np.convolve(badh.astype(np.float32), np.ones(8, np.float32), "full")[7:7 + N]
        self.danger = cs6 > 0
        # SOFT hand-height cost (lead 03:14: stop only the collar/neck, keep chest-height gestures): per candidate, the max wrist
        # height over the next 1 s above hand_soft_y costs hand_soft_w per 5 cm
        soft = np.zeros(N, np.float32)
        if hand_soft_y is not None and hand_soft_w > 0:
            yy = np.maximum(pf0[:, 1], pf0[:, 4])
            ym = yy.copy()
            for k in range(1, 30, 3):
                ym = np.maximum(ym, yy[np.minimum(np.arange(N) + k, N - 1)])
            soft = (hand_soft_w * np.clip(ym - hand_soft_y, 0, None) / 0.05).astype(np.float32)
        del pf0
        self.cand = cand_play
        act = d["act"].astype(np.float32)
        self.act = np.clip(act / 0.6, 0, 1)
        # ---- features
        pf = d["pfeat"].astype(np.float32)
        aud = d["aud"].astype(np.float32)
        seg_start = np.zeros(N, np.int64)
        meta = json.loads(str(d["meta"]))
        for c in meta["chunks"]:
            seg_start[c["g0"]:c["g1"]] = c["g0"]
        self.meta = meta
        # recompute onset causally (same as the live path) inside each chunk
        lr = aud[:, 0]
        ls = lr.copy()
        ls[1:] += lr[:-1]; ls[2:] += lr[:-2]; ls /= 3
        on = np.zeros(N, np.float32); on[3:] = np.maximum(0, ls[3:] - ls[:-3])
        for c in meta["chunks"]:
            on[c["g0"]:c["g0"] + 3] = 0
        aud[:, 1] = on
        # spectral-centroid proxy: z-score per chunk over voiced frames (live: running per-stream z-score) -> TTS-robust
        for c in meta["chunks"]:
            sl = slice(c["g0"], c["g1"]); v = aud[sl, 2] > 0.5; ce = aud[sl, 3]
            if v.sum() > 10:
                z = (ce - ce[v].mean()) / (ce[v].std() + 1e-3)
                aud[sl, 3] = np.where(v, np.clip(z, -3, 3), 0.0)
        self.aud_mu = aud.mean(0); self.aud_sd = aud.std(0) + 1e-3
        an = (aud - self.aud_mu) / self.aud_sd
        idx = np.arange(N)[:, None] + self.offs[None, :]
        idx = np.clip(idx, seg_start[:, None], self.seg_end[:, None] - 1)
        AW = an[idx]                                   # (N, n_off, 4)
        del idx
        self.pf_mu = pf.mean(0); self.pf_sd = pf.std(0) + 1e-4
        pfn = (pf - self.pf_mu) / self.pf_sd
        # group weights (pose): wrist pos 6, elbow pos 6, wrist vel 6, head vel 6
        gw = np.concatenate([np.full(6, float(x)) for x in pose_wts]).astype(np.float32)
        self.pose_w = gw / 6.0
        # audio channel weights: loud, onset, vad, centroid; spread over offsets (nearer offsets weigh more)
        cw = np.array([1.0, 1.2, 0.6, 0.3], np.float32)
        ow = np.exp(-np.abs(self.offs) / 20.0).astype(np.float32)
        ow = ow / ow.sum()
        self.aud_w = (ow[:, None] * cw[None, :]).reshape(-1) * audio_w / cw.sum() * 4
        # gesture-energy feature: log mean wrist speed over the next 1 s (both arms), z-scored; query = self.energy
        e = np.log(act.sum(1) + 0.02)
        cs = np.concatenate([[0.0], np.cumsum(e)])
        jj = np.minimum(np.arange(N) + 30, self.seg_end)
        ef = (cs[jj] - cs[np.arange(N)]) / np.maximum(jj - np.arange(N), 1)
        self.e_mu, self.e_sd = float(ef.mean()), float(ef.std() + 1e-3)
        ef = ((ef - self.e_mu) / self.e_sd)[:, None].astype(np.float32)
        self.energy = float(energy); self.energy_w = float(energy_w); self.loud_ref = float(loud_ref)
        F = np.concatenate([pfn, AW.reshape(N, -1), ef], 1).astype(np.float32)
        self.D_pose = pfn.shape[1]
        self.F = F
        del AW, an, pfn, pf
        # partitions by state (0 idle, 1 speaking, 2 listening); idle falls back to listening+idle if small
        self.parts = {}
        self.pen = {}
        self.pen_live = {}
        self.pen_pause = None
        self.blk = {}
        for s in (0, 1, 2):
            m = cand & (self.state_db == s)
            if s == 0 and m.sum() < 5 * 60 * FPS:
                m = cand & (self.state_db != 1)
            if s == 2 and m.sum() < 5 * 60 * FPS:
                m = cand & (self.state_db != 1)
            ii = np.nonzero(m)[0]
            ii = ii[ii % cand_stride == 0]
            self.parts[s] = (ii, np.ascontiguousarray(F[ii]), np.ascontiguousarray(F[ii] ** 2))
            pen = soft[ii].copy()
            self.pen_live[s] = None
            if s == 1 and speak_still_w > 0:     # while SPEAKING prefer segments with some life over the next second
                self.pen_live[s] = (speak_still_w * np.clip(speak_e_lo - F[ii, -1], 0, None)).astype(np.float32)
            if s == 1:
                self.pen_pause = np.clip(F[ii, -1] - pause_e, 0, None).astype(np.float32) * float(pause_w)
            self.pen[s] = pen
            self.blk[s] = np.stack([np.minimum(ii + o, N - 1) // 8 for o in (0, 8, 16)])   # blocks a jump would play
        self._w_cache = {}
        self._noise = np.random.default_rng(seed + 7).random(max(len(p[0]) for p in self.parts.values()) + 997).astype(np.float32) * 0.02
        # precompute F^2 @ w for every look-ahead level (valid masks are prefixes of the sorted offsets)
        for la in range(0, max(self.la_cap, 0) + 1):
            valid = self.offs <= la
            w = self._weights(valid)
            for s in (0, 1, 2):
                self._w_cache[(s, valid.tobytes())] = self.parts[s][2] @ w
            if onset_trig is not None and self.trig_aw != 1.0:
                self._w_cache[(1, valid.tobytes(), "T")] = self.parts[1][2] @ self._weights(valid, True)
        self.reset()

    # ------------------------------------------------------------------ public API
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.aud_raw = np.zeros((0, 4), np.float32)
        self._loud_hist = []
        self.med = self.loud_ref
        self.nvad = 0
        self.c_mu, self.c_var = 0.0, 1.0
        self.speaking = False
        self.partner = False
        self._ext_state = False
        self.last_used = np.full(len(self.pose) // 8 + 1, -1e12, np.float64)   # per 8-frame block
        ii = self.parts[0][0]
        self.cur = int(ii[self.rng.integers(len(ii))])
        self.off_x = self.neutral[:135] - self.pose[self.cur].astype(np.float32)
        self.off_v = np.zeros(135, np.float32)
        self.prev_out = self.neutral[:135].copy()
        self.prev_out2 = self.prev_out.copy()
        self._seed_v = None
        self.prev_state = 0
        self.since = 0
        self.jumps = 0
        self.pending = None
        self.xf = None
        self.e_s = 0.0
        self.g_s = 1.0
        self.g_v = 0.0
        self.sp_f = 1.0
        self.sp_v = 0.0
        self.nsp_n = 10 ** 6
        self.rest_k = 0; self.rest_d0 = np.zeros((8, 3)); self.rest_d1 = np.zeros((8, 3)); self.rest_u = 1.0; self.calm_n = 0
        self.rest_next = (self.rng.uniform(*self.rest_every) * FPS) if self.rests is not None else 1e18
        if self.rests is not None and self.rest_init_random:   # a session starts in any of the rests (diversity across sessions)
            self.rest_k = int(self.rng.integers(len(self.rests)))
            self.rest_d0 = self.rest_d1 = self.rests[self.rest_k]
        self.info = {"energy": 0.0, "arm_l": 0.0, "arm_r": 0.0, "state": 0, "jumps": 0}

    def _to_internal(self, pose):
        """shown pose (189,) -> this generator's internal pose (undo rest pre-multiplier and arm gain)."""
        p = np.asarray(pose, np.float32).reshape(-1)[:135].copy()
        if self.rests is not None and (self.rest_k != 0 or self.rest_u < 1.0):
            u = self.rest_u; w = u * u * u * (10 - 15 * u + 6 * u * u)
            dd = self.rest_d0 + (self.rest_d1 - self.rest_d0) * w
            ok = bool(self.rest_speak_ok[self.rest_k]) if self.rest_k < len(self.rest_speak_ok) else True
            if not ok:
                dd = dd * self.sp_f
            cur = p[21:45].astype(np.float64).reshape(8, 3)
            p[21:45] = _m2aa(np.transpose(_rodr(dd), (0, 2, 1)) @ _rodr(cur), cur).reshape(-1)
        if abs(self.g_s - 1.0) > 1e-4:
            n = self.neutral[27:39]
            p[27:39] = n + (p[27:39] - n) / self.g_s
        return p

    def seed_pose(self, pose, prev_pose=None):
        """Hand-over: another generator had the body; continue from where it actually is. pose = (189,) canon as shown this frame;
        prev_pose (optional, v4) = the pose shown one frame (1/30 s) earlier -> the hand-over keeps that velocity (velocity-continuous);
        without it the body is assumed at rest. Jumps to the DB frame nearest in arm joint angles (+ velocity when given) of the
        current state's partition and inertializes from there. ~2-4 ms."""
        p = self._to_internal(pose)
        v = np.zeros(135, np.float32) if prev_pose is None else (p - self._to_internal(prev_pose)) * FPS
        st = self._state_now()
        ii = self.parts[st][0]
        P = self.pose[ii, 21:45].astype(np.float32)
        c = ((P - p[21:45]) ** 2).sum(1)
        if prev_pose is not None:
            Pn = self.pose[np.minimum(ii + 1, len(self.pose) - 1), 21:45].astype(np.float32)
            c += 0.25 * (((Pn - P) * FPS - v[21:45]) ** 2).sum(1) / FPS
        j = int(ii[int(np.argmin(c))])
        self.prev_out = p
        self.prev_out2 = p - v / FPS
        self._seed_v = v
        self._seeded = 2
        self._jump(j)                 # velocity-continuous offset from (p, v) onto the new segment
        self.prev_state = st
        self.pending = None
        self.since = 0

    def set_energy(self, level):
        """gesture-energy dial (z-score of DB 1 s wrist-speed; -1 calmer, 0 DB average, +1 livelier). Cheap, any time."""
        self.energy = float(level)

    def set_state(self, speaking, partner_speaking=False):
        self._ext_state = True
        self.speaking = bool(speaking)
        self.partner = bool(partner_speaking)

    def push_audio(self, feats):
        f = np.asarray(feats, np.float32).reshape(-1, 42)
        out = np.zeros((len(f), 4), np.float32)
        for i in range(len(f)):
            loud = float(f[i, 40]); vad = f[i, 41] > 0.5
            if vad:
                self.nvad += 1
                a = 0.08 if self.nvad < 60 else 0.005
                self.med += a * (loud - self.med)
            lr = min(1.5, max(-3.0, (loud - self.med) / 12.0))
            h = self._loud_hist
            h.append(lr)
            if len(h) > 6:
                del h[0]
            ls = sum(h[-3:]) / len(h[-3:])
            ls3 = sum(h[-6:-3]) / 3 if len(h) >= 6 else ls
            on = max(0.0, ls - ls3)
            mel = f[i, :40].astype(np.float64)
            w = np.exp(mel / 10.0 - mel.max() / 10.0)
            cen = float((w * np.arange(40)).sum() / w.sum())
            cen = (cen - 12.0) / 6.0
            if vad:
                a = 0.1 if self.nvad < 60 else 0.005
                self.c_mu += a * (cen - self.c_mu)
                self.c_var += a * ((cen - self.c_mu) ** 2 - self.c_var)
                cen = min(3.0, max(-3.0, (cen - self.c_mu) / math.sqrt(self.c_var + 1e-6)))
            else:
                cen = 0.0
            out[i] = (lr, on, float(vad), cen)
        self.aud_raw = np.concatenate([self.aud_raw, out]) if len(self.aud_raw) else out

    def step(self):
        t = self.t
        st = self._state_now()
        force = (st != self.prev_state or self.cur + 2 >= self.seg_end[self.cur] or not self.cand[self.cur]
                 or self.danger[self.cur])
        self.since += 1
        if force:
            self.pending = None
            self._search(st, True)
            self.since = 0
        elif self.pending is not None:
            self._search_slice()
        elif st == 1 and self.pause_w > 0 and self.since >= 4 and self._pause_start():
            self._search(st, False, trig=False)      # v4: a pause just began -> look for a calm continuation now
            self.since = 0
        elif self.onset_trig is not None and st == 1 and self._onset_ahead():
            self._search(st, False, trig=True)
            self.since = 0
        elif self.since >= self.search_every:
            self._search(st, False)
            self.since = 0
        self.prev_state = st
        src = self.pose[self.cur].astype(np.float32)
        if self.blend == "inert":
            y = _hl_damp(self.halflife if (st == 1 or self.halflife_quiet is None) else self.halflife_quiet) / 2.0
            dt = 1.0 / FPS
            j1 = self.off_v + self.off_x * y
            e = math.exp(-y * dt)
            self.off_x = e * (self.off_x + j1 * dt)
            self.off_v = e * (self.off_v - j1 * y * dt)
            out = src + self.off_x
        else:
            out = src.copy()
            if self.xf is not None:
                oi, k = self.xf
                a = (k + 1) / (self.xfade_n + 1)
                out = (1 - a) * self.pose[min(oi, len(self.pose) - 1)].astype(np.float32) + a * src
                self.xf = (oi + 1, k + 1) if k + 1 < self.xfade_n else None
            if self.off_x is not None and np.any(self.off_x):
                y = _hl_damp(0.25) / 2.0
                self.off_x = self.off_x * math.exp(-y / FPS)   # only the start-from-neutral offset
                out = out + self.off_x
        # safety: per-frame speed clamp (rad / frame) and root bounds
        d = np.clip(out - self.prev_out, -0.3, 0.3)
        if self.wrist_clamp < 0.3:
            d[39:45] = np.clip(d[39:45], -self.wrist_clamp, self.wrist_clamp)
        out = self.prev_out + d
        out[3:6] = np.clip(out[3:6], -0.3, 0.3)
        if self.acc_lim is not None:
            v0 = self.prev_out[21:45] - self.prev_out2[21:45]
            dv = np.clip((out[21:45] - self.prev_out[21:45]) - v0, -self.acc_lim, self.acc_lim)
            out[21:45] = self.prev_out[21:45] + v0 + dv
        self.prev_out2 = self.prev_out
        self.prev_out = out
        self.last_used[self.cur // 8] = t
        if getattr(self, "_seeded", 0) > 0:
            self._seeded -= 1
        a = self.act[self.cur]
        self.e_s += 0.1 * (float(a.mean()) - self.e_s)
        self.info = {"energy": self.e_s, "arm_l": float(a[0]), "arm_r": float(a[1]), "state": st, "jumps": self.jumps}
        self.cur += 1
        self.t += 1
        full = self.neutral.copy()
        full[:135] = out
        self.nsp_n = 0 if st == 1 else self.nsp_n + 1
        gt = self.arm_gain if st == 1 else self.arm_gain_listen
        if st == 1 and self.arm_gain_pause is not None and self._in_pause():
            gt = float(self.arm_gain_pause)
        # critically damped (v4): the gain changes with zero initial rate -> no velocity step at a speaking flip
        wg = 1.0 / 0.25; dtg = 1.0 / FPS
        self.g_v += (wg * wg * (gt - self.g_s) - 2.0 * wg * self.g_v) * dtg
        self.g_s += self.g_v * dtg
        if abs(self.g_s - 1.0) > 1e-4:
            n = self.neutral[27:39]
            full[27:39] = n + self.g_s * (out[27:39] - n)
        if self.rests is not None:
            self._rest_step(st, a)
            du = self.rest_u
            if du < 1.0 or self.rest_k != 0:
                w = du * du * du * (10 - 15 * du + 6 * du * du)          # min-jerk
                dd = self.rest_d0 + (self.rest_d1 - self.rest_d0) * w
                ok = bool(self.rest_speak_ok[self.rest_k]) if self.rest_k < len(self.rest_speak_ok) else True
                # v4: arms-at-sides comes back only after rest_return_s without speech (not in every pause), and the ease is
                # critically damped (zero initial velocity) instead of a first-order step
                tgt = 0.0 if ((st == 1 or self.nsp_n < self.rest_return_n) and not ok) else 1.0
                wf = 1.0 / (0.15 if tgt < self.sp_f else 0.4); dtf = 1.0 / FPS
                self.sp_v += (wf * wf * (tgt - self.sp_f) - 2.0 * wf * self.sp_v) * dtf
                self.sp_f = min(1.0, max(0.0, self.sp_f + self.sp_v * dtf))
                if not ok:
                    dd = dd * self.sp_f
                cur = full[21:45].astype(np.float64).reshape(8, 3)
                Rm = _rodr(dd) @ _rodr(cur)
                full[21:45] = _m2aa(Rm, cur).reshape(-1)
        return full

    def _rest_step(self, st, a):
        if self.rest_u < 1.0:
            self.rest_u = min(1.0, self.rest_u + 1.0 / (self.rest_blend * FPS))
            return
        self.calm_n = self.calm_n + 1 if (st != 1 and float(a.max()) < 0.12) else 0
        if self.t >= self.rest_next and self.calm_n >= 10:
            K = len(self.rests)
            k = int(self.rng.integers(K - 1)); k = k + 1 if k >= self.rest_k else k
            self.rest_d0 = self.rests[self.rest_k] if self.rest_u >= 1.0 else self.rest_d0
            self.rest_d1 = self.rests[k]; self.rest_k = k; self.rest_u = 0.0
            self.rest_next = self.t + self.rng.uniform(*self.rest_every) * FPS

    # ------------------------------------------------------------------ internals
    def _state_now(self):
        if self._ext_state:
            sp, pa = self.speaking, self.partner
        else:
            t = self.t
            sp = bool(len(self.aud_raw) > t and self.aud_raw[max(0, t - 8):t + 1, 2].max() > 0.5)
            pa = False
        return 1 if sp else (2 if pa else 0)

    def _pause_start(self):
        t = self.t; n = self.pause_n
        if t < n + 1 or len(self.aud_raw) <= t:
            return False
        return bool(self.aud_raw[t - n:t + 1, 2].max() < 0.5 and self.aud_raw[t - n - 1, 2] >= 0.5)

    def _in_pause(self):
        t = self.t; n = self.pause_n
        if not self.speaking or len(self.aud_raw) <= t or t < n:
            return False
        return bool(self.aud_raw[t - n:t + 1, 2].max() < 0.5)

    def _onset_ahead(self):
        if self.since < self.trig_refr:
            return False
        avail = len(self.aud_raw) - 1 - self.t
        lead = min(self.trig_lead, avail, self.la_cap)
        if lead < 0:
            return False
        j = self.t + lead
        return bool(self.aud_raw[j, 1] > self.onset_trig and self.aud_raw[j, 1] >= self.aud_raw[j - 1, 1])

    def _query(self):
        t = self.t
        cur = self.cur
        qp = self.F[cur, :self.D_pose]
        avail = len(self.aud_raw) - 1 - t
        la = min(self.la_cap, avail)
        qa = np.zeros((len(self.offs), 4), np.float32)
        valid = np.zeros(len(self.offs), bool)
        for k, o in enumerate(self.offs):
            j = t + o
            if o <= la and 0 <= j < len(self.aud_raw):
                qa[k] = self.aud_raw[j]
                valid[k] = True
            elif o <= la and j < 0:
                qa[k] = (0.0, 0.0, 0.0, 0.0); qa[k, 0] = -3.0
                valid[k] = True
        qa = (qa - self.aud_mu) / self.aud_sd
        return np.concatenate([qp, qa.reshape(-1), [self.energy]]).astype(np.float32), valid

    def _weights(self, valid, trig=False):
        key = valid.tobytes() + (b"T" if trig else b"")
        w = self._w_cache.get(key)
        if w is None:
            aw = self.aud_w.reshape(len(self.offs), 4) * valid[:, None]
            if self.aud_renorm > 1.0 and aw.sum() > 0:   # little look-ahead -> fewer audio terms: give the past window the full weight
                aw = aw * min(self.aud_renorm, self.aud_w.sum() / aw.sum())
            if trig:
                aw = aw * self.trig_aw
            w = np.concatenate([self.pose_w, aw.reshape(-1), [self.energy_w]]).astype(np.float32)
            self._w_cache[key] = w
        return w

    def _search(self, st, force, trig=False):
        """force -> full search now; else start an amortised search over `search_slices` frames; trig -> one slice, trigger weights."""
        ii, Fp, F2p = self.parts[st]
        q, valid = self._query()
        tw = trig and self.trig_aw != 1.0
        w = self._weights(valid, tw)
        key = (st, valid.tobytes(), "T") if tw else (st, valid.tobytes())
        f2w = self._w_cache.get(key)
        if f2w is None:
            f2w = F2p @ w
            self._w_cache[key] = f2w
        wq = w * q
        cc = 1e9 if force else float(w @ (self.F[self.cur] - q) ** 2)
        S = 1 if (force or trig) else max(1, self.search_slices)
        n = len(ii)
        self.pending = dict(st=st, wq=wq, qq=float(q @ wq), f2w=f2w, t0=self.t, cc=cc, cur0=self.cur, best=-1,
                            bc=1e18, k=0, S=S, edges=np.linspace(0, n, S + 1).astype(int),
                            noise=int(self.rng.integers(997)), bias=self.trig_bias if trig else self.switch_bias,
                            pause=self._in_pause())
        self._search_slice()

    def _search_slice(self):
        p = self.pending
        ii, Fp, _ = self.parts[p["st"]]
        a, b = p["edges"][p["k"]], p["edges"][p["k"] + 1]
        if b > a:
            cost = p["f2w"][a:b] - 2.0 * (Fp[a:b] @ p["wq"]) + p["qq"]
            lu = self.last_used
            bk = self.blk[p["st"]][:, a:b]
            age = self.t - np.maximum(np.maximum(lu[bk[0]], lu[bk[1]]), lu[bk[2]])
            cost += self.rep_w * np.clip(1.0 - age / self.rep_T, 0.0, 1.0)
            cost += self.pen[p["st"]][a:b]
            if p["st"] == 1:
                if p.get("pause") and self.pause_w > 0:
                    cost += self.pen_pause[a:b]
                elif self.pen_live.get(1) is not None:
                    cost += self.pen_live[1][a:b]
            iis = ii[a:b]
            cost[(iis >= p["cur0"] - 30) & (iis <= p["cur0"] + 30)] += 1e3   # that is "continue"
            o = p["noise"] + a % 7
            cost += self._noise[o:o + (b - a)]
            k = int(np.argmin(cost))
            if cost[k] < p["bc"]:
                p["bc"], p["best"] = float(cost[k]), int(iis[k])
        p["k"] += 1
        if p["k"] < p["S"]:
            return
        self.pending = None
        if p["best"] < 0 or p["bc"] + p["bias"] >= p["cc"]:
            return
        j = p["best"] + (self.t - p["t0"])          # the match was for t0; advance by the frames since
        if j + 2 >= self.seg_end[p["best"]]:
            return
        self._jump(j)

    def _jump(self, j):
        self.jumps += 1
        if self.blend == "inert":
            # VELOCITY-CONTINUOUS inertialization (v4, lead 05:42): the offset starts at the true velocity difference between
            # what was SHOWN (last two output frames) and what plays next, and is placed so that the first new frame continues
            # the old motion exactly: out1 = prev_out + v_old/FPS. (v3 added the old velocity twice -> a one-frame velocity step.)
            P = self.pose
            j1 = min(j + 1, len(P) - 1)
            src_v = (P[j1].astype(np.float32) - P[j].astype(np.float32)) * FPS
            old_v = (self.prev_out - self.prev_out2) * FPS
            self.off_x = self.prev_out + src_v / FPS - P[j].astype(np.float32)
            self.off_v = np.clip(old_v - src_v, -6.0, 6.0)
        else:
            self.xf = (self.cur, 0)
        self.cur = j
