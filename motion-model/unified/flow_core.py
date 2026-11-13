"""Learned planning core for unified/runtime.py.

At each re-plan: the pending backbone tokens (one per frame since the last plan) are fed for the conditional AND the
user-null stream (batch of 2) through a fixed-size sliding-window KV cache; then the flow head samples the next chunk
with guided Euler steps, inpainting the committed prefix and any override groups. Everything that does not change
across Euler steps (memory keys/values, adaLN modulations, the known-value embedding) is computed once per plan.
Backends: "ort" (ONNX Runtime; unified/export.py writes the graphs into the bundle) or "torch".

phase 2 bundles:
  * audio (config["audio"]; absent = mel): "in" = "mel" (the 42 canon audio features per person, A = 42) or
    "enc_<name>" (a speech encoder's per-frame feature, De = feat_dim raw dims, reduced by the bundle's PCA to k =
    proj_dim; A = k + 2). Token audio block in enc mode = [clip((proj(f) - emean) / estd, -6, 6) (k) |
    norm_audio(audio42)[40:42] (loud_db, vad of the same person)]; a missing feature = esil (the silence feature).
    The token layout comes from unified.model.layout(aud_dim).
  * event head (model n_events > 0): on_frame() feeds the newest token every frame (identical KV / memory to the
    batched feed: step_fixed is exact) and returns p = sigmoid(event_head(h_cond[token t-1])) = the onset probability
    of each event class at frame t. The firing rule and the curves live in unified/runtime.py.
  * head_kind "regression": one head call at s = 0 from x = 0 (steps = 1, temperature = 0 forced).
"""
from __future__ import annotations

import math
import os

import numpy as np

from canon1 import spec
from unified.data import LIP_COLS, SPEAK_DILATE

D = spec.D_MOTION
G = spec.D_MASK
A = spec.D_AUDIO                     # mel audio width (canon audio42)


def rope_np(pos, dh, base=10000.0):
    inv = 1.0 / (base ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    ang = np.remainder(np.asarray(pos, np.float64)[:, None] * inv[None], 2 * math.pi)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class FlowCore:
    kind = "flow"

    def __init__(self, bundle_dir, cfg, threads=2, seed=0, steps=6, guidance=1.5, temperature=0.8, prefix=None,
                 backend=None, consistent_noise=True, noise_rho=0.0, temperature_groups=None, **kw):
        # consistent_noise (seam fix): a re-plan reuses the previous plan's noise for the frames both chunks
        # cover (shifted by the executed frames) and draws fresh noise only for the new tail -> consecutive 100 ms
        # plans agree instead of re-sampling every frame (the 4-8 Hz jitter of independent draws).
        self.consistent_noise = bool(consistent_noise)
        # noise_rho: AR(1) correlation of the sampling noise along the chunk's frames (0 = iid, as trained). The
        # 33-min model passes much of its input noise through to the output, so iid noise = frame-to-frame jitter;
        # correlated noise (unit variance kept) turns that residue into slow wander. temperature_groups: per canon
        # group temperature overrides, e.g. {"face_upper": 0.8} (blinks/brows need stochasticity, arms do not).
        self.noise_rho = float(noise_rho or 0.0)
        from unified.model import layout
        self.cfg = cfg
        mc = cfg["model"]
        self.mc = mc
        self.aud_dim = int(mc.get("aud_dim", A))
        self.lay = layout(self.aud_dim)
        self.n_events = int(mc.get("n_events", 0) or 0)
        self.head_kind = str(mc.get("head_kind", "flow") or "flow")
        if self.head_kind not in ("flow", "regression"):
            raise ValueError(f"unknown head_kind {self.head_kind!r}")
        if self.head_kind == "regression":          # the head maps x0 = 0 at s = 0 straight to the chunk
            steps, temperature = 1, 0.0
        self.H, self.M, self.W = mc["horizon"], mc["mem"], mc["window"]
        self.d, self.layers, self.heads = mc["d"], mc["layers"], mc["heads"]
        self.dh = self.d // self.heads
        self.self_audio_lookahead = int(mc["lookahead"])
        self.C = int(cfg.get("prefix", 2) if prefix is None else prefix)
        self.horizon = self.H - self.C            # new frames per plan
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.temperature = float(temperature)
        self.temp_col = None
        if temperature_groups and self.head_kind != "regression":
            tc = np.full(D, float(temperature), np.float32)
            for g, v in dict(temperature_groups).items():
                tc[spec.GROUPS[g]] = float(v)
            self.temp_col = tc
        self.seed = seed
        z = np.load(os.path.join(bundle_dir, "norm.npz"))
        self.mean, self.std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        self.amean, self.astd = z["amean"].astype(np.float32), z["astd"].astype(np.float32)
        ac = cfg.get("audio") or {}
        self.audio_in = str(ac.get("in", "mel") or "mel")
        if self.audio_in == "mel":
            if self.aud_dim != A:
                raise ValueError(f"mel audio needs model aud_dim {A}, bundle has {self.aud_dim}")
            self.feat_dim = self.proj_dim = self.user_feat_shift = 0
        else:
            self.feat_dim = int(ac["feat_dim"])
            self.proj_dim = int(ac.get("proj_dim", self.feat_dim))
            self.user_feat_shift = int(ac.get("user_shift", 0))
            if not ac.get("keep_vad", True):
                raise NotImplementedError("enc audio without the (loud_db, vad) pair (keep_vad false)")
            if self.aud_dim != self.proj_dim + 2:
                raise ValueError(f"enc audio: model aud_dim {self.aud_dim} != proj_dim {self.proj_dim} + 2")
            self.emean, self.estd = z["emean"].astype(np.float32), z["estd"].astype(np.float32)
            self.esil = z["esil"].astype(np.float32)
            if "epca_V" in z:
                self.epca_mean = z["epca_mean"].astype(np.float32)
                self.epca_V = np.ascontiguousarray(z["epca_V"].astype(np.float32).T)      # (De, k)
                assert self.epca_V.shape == (self.feat_dim, self.proj_dim), self.epca_V.shape
            else:
                assert self.feat_dim == self.proj_dim, (self.feat_dim, self.proj_dim)
                self.epca_mean = self.epca_V = None
            self.feat_fill = self.norm_feat_proj(self.esil)          # the token block of silence
        self.bs_cols = np.arange(135, 187)
        self.lip_cols = LIP_COLS
        self.lip_on = bool(cfg.get("lip"))                 # bundles trained with the lip mask
        self.goc = spec.GROUP_OF_COL
        self.s_all = (np.arange(self.steps) / self.steps).astype(np.float32)      # regression: [0.0]
        sfx = kw.get("onnx_suffix", "")
        graphs = [f"backbone_step{sfx}.onnx", f"head_prep{sfx}.onnx", f"head_step{sfx}.onnx"]
        if self.n_events > 0:
            graphs.append(f"event_step{sfx}.onnx")
        if backend is None:
            backend = "ort" if all(os.path.exists(os.path.join(bundle_dir, g)) for g in graphs) else "torch"
        self.backend = backend
        if backend == "ort":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(threads); so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            mk = lambda n: ort.InferenceSession(os.path.join(bundle_dir, n), so,   # noqa: E731
                                                providers=["CPUExecutionProvider"])
            self.s_bb = mk(graphs[0])
            self.s_prep = mk(graphs[1])
            self.s_head = mk(graphs[2])
            self.s_ev = mk(graphs[3]) if self.n_events > 0 else None
        else:
            import torch
            from unified.model import BBStep, EvStep, HeadPrep, HeadStep, UnifiedModel
            torch.set_num_threads(int(threads))
            self.torch = torch
            self.model = UnifiedModel(mc)
            sd = torch.load(os.path.join(bundle_dir, "weights.pt"), map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd)
            self.model.eval()
            self.t_bb = BBStep(self.model.backbone).eval()
            self.t_prep, self.t_head = HeadPrep(self.model.head).eval(), HeadStep(self.model.head).eval()
            self.t_ev = EvStep(self.model.events).eval() if self.n_events > 0 else None
        # per-frame token feeding: on for event bundles (the event head needs the newest token every frame); old
        # bundles keep the batched feed at plan time (bit-for-bit the previous behaviour)
        # reaction adapter: partner-trace adapter (unified/react.py): h' = h + adapter(trace of the user block), numpy
        self.react = None
        rt = mc.get("react_trace")
        if rt:
            from unified import react as RX
            if backend == "ort":
                zr = np.load(os.path.join(bundle_dir, "react.npz"))
                w = {k: zr[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2")}
            else:
                w = {k: v.astype(np.float32) for k, v in self.model.react.np_weights().items()}
            self.react = dict(RX=RX, w=w, ni=rt, cols=RX.token_cols(self.lay),
                              a0=RX.adapter_np(np.zeros((1, RX.N_FEAT), np.float32), w)[0])
        fe = kw.get("feed_every_frame")
        self.feed_every_frame = (self.n_events > 0) if fe is None else bool(fe)
        self.reset()

    # --------------------------------------------------------------------------------- state
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.kc = np.zeros((self.layers, 2, self.heads, self.W - 1, self.dh), np.float32)
        self.vc = np.zeros_like(self.kc)
        self.kvalid = np.zeros(self.W - 1, np.float32)
        self.mem = np.zeros((2, self.M, self.d), np.float32)
        self.mvalid = np.zeros(self.M, np.float32)
        self.n_tok = None
        self._z = None                    # (start frame of the chunk, unit noise (H,D)) of the last plan
        self._h_last = None               # cond-stream backbone output of the newest fed token (d,)
        self._rrows = np.zeros((0, 8), np.float32)      # react: the trace's token columns of the newest RING tokens

    def _run(self, sess_or_mod, names, feeds):
        if self.backend == "ort":
            return sess_or_mod.run(None, feeds)
        T = self.torch
        with T.no_grad():
            out = sess_or_mod(*[T.from_numpy(np.ascontiguousarray(feeds[k])) for k in names])
        return [o.numpy() for o in (out if isinstance(out, (tuple, list)) else [out])]

    def norm_motion(self, m):
        return np.clip((m - self.mean) / self.std, -8, 8)

    def norm_audio(self, a):
        a = np.array(a, np.float32, copy=True)
        a[..., 40] = np.maximum(a[..., 40], -80)
        return np.clip((a - self.amean) / self.astd, -6, 6)

    # ---- encoder features (enc mode): the rings hold the normalized projected feature (k,) per frame
    def norm_feat_proj(self, fp):
        """projected raw feature (..., k) -> token values (clipped z-score)."""
        return np.clip((np.asarray(fp, np.float32) - self.emean) / self.estd, -6, 6).astype(np.float32)

    def feat_row(self, f):
        """one raw encoder feature (De,) | None -> the (k,) token row stored in the history ring."""
        if f is None or self.feat_dim == 0:
            return self.feat_fill
        f = np.asarray(f, np.float32).reshape(self.feat_dim)
        if not np.isfinite(f).all():
            return self.feat_fill
        if self.epca_V is not None:
            f = (f - self.epca_mean) @ self.epca_V
        return self.norm_feat_proj(f)

    def _audio_block(self, a42, feat):
        """the token audio block of one person: mel = norm_audio(a42); enc = [feat rows (k) | norm_audio[40:42]]."""
        na = self.norm_audio(a42)
        if self.audio_in == "mel":
            return na
        return np.concatenate([feat, na[:, 40:42]], 1)

    def speaking(self, hist, f0, f1):
        """bool (f1-f0,) the AVATAR speaks at frames [f0,f1) (vad of its own audio, dilated like training); frames
        past the newest look-ahead row repeat the newest known state."""
        dl = SPEAK_DILATE
        L = self.self_audio_lookahead
        a, ok = hist.self_audio_ahead.get(f0 - dl - L, f1 + dl - L)
        v = (a[:, 41] > 0.5)
        last = hist.self_audio_ahead.n - 1 + L                         # newest frame whose audio we have
        idx = np.arange(f0 - dl, f1 + dl)
        if last >= f0 - dl and (idx > last).any():
            lv = v[min(len(v) - 1, last - (f0 - dl))]
            v = np.where(idx > last, lv, v)
        out = np.zeros(f1 - f0, bool)
        for k in range(2 * dl + 1):
            out |= v[k:k + f1 - f0]
        return out

    def _tokens(self, hist, t0, t1):
        """cond tokens for tau in [t0, t1) -> (n, DIN). tau = -1 is the virtual start token (self = the mean pose)."""
        n = t1 - t0
        sm, ok = hist.self_motion.get(t0, t1)
        xs = np.where(ok[:, None], self.norm_motion(sm), 0)
        if self.lip_on:
            spk = self.speaking(hist, t0, t1)
            xs[np.ix_(spk, self.lip_cols)] = 0
        sty, _ = hist.style.get(t0, t1)
        um, _ = hist.user_motion.get(t0 + 1, t1 + 1)
        uk, _ = hist.user_mask.get(t0 + 1, t1 + 1)
        ua, _ = hist.user_audio.get(t0 + 1, t1 + 1)
        sa, _ = hist.self_audio_ahead.get(t0 + 1, t1 + 1)
        uf = sf = None
        if self.audio_in != "mel":
            uf, _ = hist.user_feat.get(t0 + 1, t1 + 1)          # row s = the user's newest feature at frame s
            sf, _ = hist.self_feat_ahead.get(t0 + 1, t1 + 1)    # row s = the avatar's feature of frame s+L
        lo = self.lay
        tok = np.zeros((n, lo["din"]), np.float32)
        tok[:, lo["self"]] = xs
        tok[:, lo["self_mask"]] = 1
        tok[:, lo["user"]] = self.norm_motion(um) * uk[:, self.goc]
        tok[:, lo["user_mask"]] = uk
        tok[:, lo["user_audio"]] = self._audio_block(ua, uf)
        tok[:, lo["self_audio"]] = self._audio_block(sa, sf)
        tok[:, lo["style"]] = sty
        return tok

    def observe(self, hist, t):
        """feed every pending token up to tau = t-1 (token t-1 is complete at step t)."""
        t0 = -1 if self.n_tok is None else self.n_tok
        if t0 > t - 1 and self.n_tok is not None:
            return
        t1 = max(t, 0)                                 # tokens [t0, t1)
        if t1 <= t0:
            t1 = t0 + 1
        # never feed more than W tokens at once (older ones would fall out of every window anyway)
        if t1 - t0 > self.W:
            t0 = t1 - self.W
        tok = self._tokens(hist, t0, t1)
        tu = tok.copy()
        tu[:, self.lay["user_all"]] = 0
        tu[:, self.lay["unull"]] = 1                          # user-null flag (style columns follow it)
        n = len(tok)
        cos, sin = rope_np(np.arange(t0, t1), self.dh)
        out, self.kc, self.vc = self._run(self.s_bb if self.backend == "ort" else self.t_bb,
                                          ("tok", "cos", "sin", "kc", "vc", "kvalid"),
                                          dict(tok=np.stack([tok, tu]), cos=cos, sin=sin, kc=self.kc, vc=self.vc,
                                               kvalid=self.kvalid))
        if self.react is not None:
            R = self.react
            self._rrows = np.concatenate([self._rrows, tok[:, R["cols"]]])[-R["RX"].RING:]
            tr = R["RX"].trace_np(self._rrows, R["ni"])[-n:]
            out = out.copy()
            out[0] += R["RX"].adapter_np(tr, R["w"])
            out[1] += R["a0"]
        self.kvalid = np.concatenate([self.kvalid, np.ones(n, np.float32)])[-(self.W - 1):]
        self.mem = np.concatenate([self.mem, out], 1)[:, -self.M:]
        self.mvalid = np.concatenate([self.mvalid, np.ones(n, np.float32)])[-self.M:]
        self.n_tok = t1
        self._h_last = out[0, -1]

    def on_frame(self, hist, t):
        """per-frame work at step t (frame t's inputs already pushed): feed token t-1 and, with an event head, return
        the onset probabilities p (n_events,) for frame t; None otherwise."""
        if not self.feed_every_frame:
            return None
        self.observe(hist, t)
        if self.n_events <= 0:
            return None
        h = self._h_last[None].astype(np.float32)
        p = self._run(self.s_ev if self.backend == "ort" else self.t_ev, ("h",), dict(h=h))[0]
        return p[0].astype(np.float32)

    # --------------------------------------------------------------------------------- plan
    def set_prefix(self, c):
        """committed prefix C (known frames) for the next plans; new frames per plan = H - C. The head works in
        patches of P frames from chunk row 0 = t - C, so changing C shifts the patch grid (runtime prefix cycling)."""
        c = int(c)
        if not 0 <= c < self.H:
            raise ValueError(f"prefix {c} outside [0, {self.H})")
        self.C = c
        self.horizon = self.H - c

    def plan(self, hist, t, known_vals, known_gmask):
        self.observe(hist, t)
        H, C = self.H, min(self.C, t)
        kv = np.zeros((H, D), np.float32)
        km = np.zeros((H, G), np.float32)
        if C > 0:
            pm, _ = hist.self_motion.get(t - C, t)
            kv[:C] = self.norm_motion(pm)
            km[:C] = 1
        if known_vals is not None:
            nn_ = H - C
            sel = known_gmask[:nn_, self.goc] > 0
            kv[C:] = np.where(sel, self.norm_motion(known_vals[:nn_]), kv[C:])
            km[C:] = np.maximum(km[C:], known_gmask[:nn_])
        kcol = km[:, self.goc][None]
        kv1, km1 = kv[None], km[None]
        mbias = np.broadcast_to(((1.0 - self.mvalid) * -1e9).astype(np.float32), (2, 1, 1, self.M)).copy()
        sty = hist.style.last()
        sty = (np.asarray(sty, np.float32) if sty is not None else np.array([0, 0, 0, 1, 1, 1], np.float32))[None]
        memk, memv, bm, fm, kemb = self._run(self.s_prep if self.backend == "ort" else self.t_prep,
                                             ("mem", "s_all", "kvals", "kmask", "style"),
                                             dict(mem=self.mem, s_all=self.s_all, kvals=kv1, kmask=km1, style=sty))
        s0_ = t - C                                              # absolute frame of chunk row 0
        zu = self.rng.standard_normal((H, D)).astype(np.float32)
        keep = 0
        if self.consistent_noise and self._z is not None:
            sh = s0_ - self._z[0]
            if 0 <= sh < H:
                keep = H - sh
                zu[:keep] = self._z[1][sh:]
        if self.noise_rho > 0:
            r = self.noise_rho
            q = np.float32(np.sqrt(1 - r * r))
            for i in range(max(keep, 1), H):
                zu[i] = r * zu[i - 1] + q * zu[i]
        self._z = (s0_, zu)
        z = (zu[None] * (self.temp_col if self.temp_col is not None else self.temperature)).astype(np.float32)
        x = z.copy()
        g = np.array([self.guidance], np.float32)
        head = self.s_head if self.backend == "ort" else self.t_head
        names = ("x", "kemb", "memk", "memv", "bmods", "fmods", "mbias", "g")
        for i in range(self.steps):
            s0 = float(self.s_all[i])
            x = (kcol * ((1 - s0) * z + s0 * kv1) + (1 - kcol) * x).astype(np.float32)
            v = self._run(head, names, dict(x=x, kemb=kemb, memk=memk, memv=memv, bmods=bm[i], fmods=fm[i],
                                            mbias=mbias, g=g))[0]
            x = x + v / self.steps
        x = kcol * kv1 + (1 - kcol) * x
        out = x[0, C:] * self.std + self.mean
        out[:, self.bs_cols] = np.clip(out[:, self.bs_cols], 0, 1)
        if self.lip_on:                                   # lip/jaw shapes while the avatar speaks = lip sync's;
            spk = self.speaking(hist, t, t + len(out))     # the model writes the train mean (as in training)
            out[np.ix_(spk, self.lip_cols)] = self.mean[self.lip_cols]
        return out.astype(np.float32)
