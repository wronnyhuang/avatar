"""unified/runtime.py - live API of the avatar's unified non-verbal motion model (CANON v1 in, CANON v1 out).

    from unified.runtime import UnifiedPlanner
    p = UnifiedPlanner(bundle_dir, threads=2, seed=0, replan_every=3, steps=6, guidance=1.5, temperature=0.8)
    out = p.step(user_motion189|None, user_mask9|None, user_audio42, self_audio42_ahead,
                 user_feat=(De,)|None, self_feat_ahead=(De,)|None,                         # once per 30 Hz frame
                 user_pcm=(~533,) 16 kHz|None, self_pcm_ahead=(~533,) 16 kHz|None)         # enc bundles, PCM mode
    #  -> {"motion": (189,) float32, "owned": (9,) float32 0..1, "plan_ms": float|None,
    #      "events": [names fired at this frame], "event_p": (n_events,) float32 | None,
    #      "enc_ms": {"user": ms|None, "self": ms|None} | None (PCM mode: speech-encoder time when one ran)}
    p.set_override(groups=["arm_r", "fingers_r"], values=None|(189,)|callable(k)->(189,), fade_in_s=0.15)
    p.release_override(groups=None, fade_out_s=0.4)
    p.set_style({"energy": 0.8, "mirroring": None, "expressiveness": 0.3})   # or p.set_style(None)
    p.reset()

Frame/time conventions (all causal):
  * step() number s produces the avatar's motion for frame s. Its inputs are the user's camera fit for frame s
    (user_motion189 / user_mask9; None = no user visible -> all-zero mask), the user's audio features for frame s
    (42, canon1 spec; None = silence) and the AVATAR's OWN speech features for frame s + L, L = the bundle's
    self-audio lookahead (`p.self_audio_lookahead`, default 6 frames = 200 ms: the page reads its TTS ahead).
    Pass None for self audio when the avatar is silent.
  * The model re-plans every `replan_every` frames from the committed state (the frames it already returned,
    INCLUDING override blending), and commits frames from the newest plan; consecutive plans that disagree are
    crossfaded over a few frames.
  * Overrides (commanded actions): set_override(groups, values, fade_in_s) makes the action OWN those canon groups.
    owned[g] ramps 0 -> 1 over fade_in_s; the returned motion for group g is (1-owned)*model + owned*action.
      values=None      -> the action's pose is not given to the planner: the groups hold the pose they had at the
                          call (the compositor drives the rig itself and uses `owned` as its blend weight; call
                          set_override again with values to keep the planner informed if you can).
      values=(189,)    -> a constant canon pose for the owned groups (other columns ignored).
      values=callable  -> fn(k) -> (189,), k = frames since the set_override call (0 = the first stepped frame
                          after the call). It is also called for FUTURE k (the planner looks ahead one chunk), so it
                          must be a pure function of k.
    While fully owned (owned == 1), the model is CONDITIONED on the action's pose for those groups (inpainting), so
    the groups it keeps (face, gaze, untouched limbs) stay consistent with the action; during the fade-in ramp the
    owned groups are left to the model (it inpaints its own transition toward the action) and the output blends
    model -> action by the ramp, per joint by quaternion slerp (root shift / face / gaze linearly). Calling set_override on groups that are already
    owned only replaces their values (no fade restart).
    release_override(groups=None -> all owned, fade_out_s): the action's last pose is frozen, owned ramps 1 -> 0,
    and the model takes the groups back by inpainting from the committed history (which holds the action's pose).
  * Lip sync owns the mouth while the avatar talks: that is the compositor's job. The model predicts the mouth group
    while listening (smiles, frowns, press, lip/jaw shapes); while the AVATAR speaks (its own vad, dilated 2 frames)
    the lip/jaw SHAPE blendshapes (jawOpen, mouthClose/Funnel/Pucker/Roll*/Shrug*/LowerDown*/UpperUp*/Stretch*) were
    loss-masked in training and the runtime writes their train mean there; smiles/frowns/press stay the model's.
  * STYLE: set_style(dict) sets {energy, mirroring, expressiveness} in [0,1] (clipped). Keys not in the dict
    keep their current setting; a key set to None goes back to "infer" (the learned null for that component);
    set_style(None) = all three inferred (the default). Takes effect from the next re-plan (<= replan_every frames).
  * Re-planning: every `replan_every` frames (default 3 = 100 ms) a new 15-frame chunk is sampled (the first
    `prefix` frames = committed history, the rest new); frames between re-plans come from the newest plan.
  * SEAM knobs (kwargs or config["runtime"]; absent = the old combination, bit for bit): box = a centred
    moving average over the head's patch length (3) with frozen look-ahead - the flow head emits 3-frame patches whose
    rows carry a phase-locked sawtooth, and with replan_every 3 every plan has the same patch phase, so the old output
    had a 10 Hz line (abl_t0: 3e4 x its neighbours); ensemble = mean of the newest n plans covering a frame; prefix =
    the committed prefix C, or a list cycled per plan (shifts the patch grid). See UnifiedPlanner._init_seam.

  * SPEECH-ENCODER AUDIO (phase 2 bundles with config["audio"]["in"] = "enc_<name>"; `p.audio_in`, `p.feat_dim`
    = De, `p.user_feat_shift` = s): besides audio42 (its loud_db / vad still feed the model) the caller passes the
    encoder's raw (De,) features. user_feat = the NEWEST AVAILABLE feature of the user's audio at this step (live: the
    encoder's newest finished frame = frame t - s; offline: feats[t - s], None while t - s < 0) - the planner does not
    shift it again. self_feat_ahead = the feature of the AVATAR's own audio at frame t + L (offline feats[t + L]);
    None = silence. Both are ignored by mel bundles (p.audio_in == "mel", p.feat_dim == 0).
  * RAW AUDIO (PCM mode, phase 2; `p.pcm_mode`): an enc bundle whose config["audio"]["encoder"] holds the encoder's
    META runs the frozen speech encoder itself (unified/audio_feats.py: two streaming encoders, user + self, from
    unified.speech_enc). Frame 0 = absolute 16 kHz sample 0 (the first sample after reset()); frame k = samples
    [frame_end_sample(k-1), frame_end_sample(k)) (533/534; audio_feats.frame_samples / pcm_frame slice a signal).
    user_pcm = the user's samples of frame t; self_pcm_ahead = the AVATAR's own samples of frame t + P,
    P = `p.self_pcm_lookahead` = L + la + 2 (la = encoder lookahead in frames; +2 = worst case of the 3-frame encoder
    step grid; the avatar's frames 0..P-1 are silence); None = silence samples. Each encoder runs its 100 ms step as
    soon as it is due; the planner then uses user_feat = the feature of frame t - user_shift (needs user_shift >= la+2;
    None before it exists) and self_feat_ahead = the feature of frame t + L - exactly as if the caller had passed the
    offline features. Explicitly passed user_feat / self_feat_ahead still win. pcm=False (constructor) never builds
    the encoders (callers with precomputed features); an enc bundle without an encoder spec rejects PCM; mel bundles
    ignore PCM. `p.last_feats` = {"user", "self"} raw features used at the last step.
  * EVENTS (bundles with an event head, model n_events > 0; `p.event_names`): every frame the backbone's newest
    token gives the onset probability event_p[c] of each class (nod / blink / smile) at this frame. Firing per class
    (config["runtime"]["events"]["mode"], or the `events={...}` kwarg): "threshold" = p crosses thresholds[c] upward,
    "sample" = with probability p (seeded rng), "off" = never; never within refractory[c] frames of its last fire,
    never while its canon group (events.groups[c]) is owned by an override. A fired event adds gain[c] * curves[c, k]
    (events.npz, raw canon units) to the model frame for k = 0..len[c]-1 from this frame on, before override blending
    (blendshapes re-clipped to [0,1]). feed_history True (default): the history the model conditions on holds the
    frames with the events; False: the frames without them (the returned motion still has them).
  * REGRESSION bundles (model head_kind "regression"): one head call per plan; steps = 1 and temperature = 0 are
    forced (p.runtime records the effective values).

Bundle: <bundle_dir>/config.json {"kind": "copy_delay" | "flow", ...}. bundle_dir=None = the copy-delay stub.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from canon1 import spec  # noqa: E402

D = spec.D_MOTION
G = spec.D_MASK
A = spec.D_AUDIO
FPS = spec.FPS
GROUP_COLS = [np.asarray(spec.GROUPS[g], np.int64) for g in spec.GROUP_NAMES]
GI = {g: i for i, g in enumerate(spec.GROUP_NAMES)}

# Rest pose: everything 0 (session-median root, neutral face, gaze ahead) except the arms hanging down
# (SMPL rest = T-pose; left arm points +x, rotate -1.2 rad about z to hang it; right arm the mirror).
REST_POSE = np.zeros(D, np.float32)
REST_POSE[spec.COL["shoulder_l_z"]] = -1.2
REST_POSE[spec.COL["shoulder_r_z"]] = 1.2
STYLE_NAMES = ["energy", "mirroring", "expressiveness"]
# sampling defaults when neither the caller nor the bundle's config["runtime"] sets them
RUNTIME_DEFAULTS = dict(replan_every=3, steps=6, guidance=1.5, temperature=0.8, blend=6, consistent_noise=True,
                        smooth=0.0, noise_rho=0.0, temperature_groups=None, ensemble=None, prefix=None, box=0,
                        input_gain_db=None, box_groups=None, anchor=None)
ENSEMBLE_DEFAULTS = dict(plans=3, weights="uniform", tau=3.0)
ENSEMBLE_WEIGHTS = ("uniform", "exp")
STYLE_NULL = np.array([0, 0, 0, 1, 1, 1], np.float32)     # 3 values + 3 null flags (1 = infer)
# Digital silence in the 42 audio features (log-mel floor log(1e-6), loud_db floor -100, vad 0).
SILENCE_AUDIO = np.concatenate([np.full(40, math.log(1e-6)), [-100.0, 0.0]]).astype(np.float32)
EVENT_DEFAULTS = dict(mode="threshold", gain=1.0, feed_history=True)
EVENT_MODES = ("threshold", "sample", "off")


# ------------------------------------------------------------------ pose blending (slerp for joint rotations)
_ROT = np.zeros(D, bool)
_ROT[0:3] = True
_ROT[6:135] = True                    # body joints + fingers: axis-angle triples; root_shift/face/gaze are linear


def _aa_to_quat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    return np.concatenate([np.cos(th / 2), k * np.sin(th / 2)], -1)


def _quat_to_aa(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    w = np.clip(q[..., :1], -1, 1)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    th = 2 * np.arctan2(s, w)
    return q[..., 1:] / np.maximum(s, 1e-12) * th


def _aa_near(aa, ref):
    """the axis-angle triple of the same rotation as aa (angle th <= pi) that lies nearest to ref: aa itself or its
    2-pi twin aa * (th - 2 pi) / th (same rotation, the axis flipped). _quat_to_aa always returns th <= pi, so a joint
    whose pose has |aa| > pi (a deep elbow bend + an action offset) would otherwise flip representation mid-blend: a
    5.7 'rad' jump in the canon columns the next model token and the compositor read, for a continuous rotation."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    twin = aa * (th - 2 * np.pi) / np.maximum(th, 1e-9)
    use = (th > 1e-6) & (np.sum((twin - ref) ** 2, -1, keepdims=True) < np.sum((aa - ref) ** 2, -1, keepdims=True))
    return np.where(use, twin, aa)


def slerp_aa(a, b, w):
    """per-joint slerp of axis-angle triples a -> b by weight w (arrays (...,3)); the result is written in the
    representation nearest to the linear blend of a and b (continuous with both ends, see _aa_near)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    qa, qb = _aa_to_quat(a), _aa_to_quat(b)
    dot = np.sum(qa * qb, -1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb); dot = np.abs(dot)
    om = np.arccos(np.clip(dot, -1, 1))
    so = np.sin(om)
    lin = so < 1e-6
    wa = np.where(lin, 1 - w, np.sin((1 - w) * om) / np.where(lin, 1, so))
    wb = np.where(lin, w, np.sin(w * om) / np.where(lin, 1, so))
    return _aa_near(_quat_to_aa(wa * qa + wb * qb), (1 - w) * a + w * b)


def blend_cols(cols, a, b, w):
    """blend model values a -> action values b (both over columns `cols`) by w: slerp per joint, lerp elsewhere."""
    if w >= 1.0:
        return np.asarray(b, np.float32).copy()
    out = (1 - w) * a + w * b
    rot = _ROT[cols]
    if rot.any():
        rc = np.nonzero(rot)[0]                       # rotation triples are contiguous and aligned to 3
        out[rc] = slerp_aa(a[rc].reshape(-1, 3), b[rc].reshape(-1, 3), w).reshape(-1)
    return out.astype(np.float32)


class Ring:
    """Fixed-capacity history of per-frame vectors, indexed by absolute frame number."""

    def __init__(self, dim, cap=512, dtype=np.float32, fill=0.0):
        self.buf = np.zeros((cap, dim), dtype)
        self.fill = np.asarray(fill, dtype) if np.ndim(fill) else dtype(fill)
        self.cap = cap
        self.n = 0                    # frames pushed so far (absolute index of the next frame)

    def push(self, v):
        self.buf[self.n % self.cap] = v
        self.n += 1

    def get(self, start, stop):
        """rows for absolute frames [start, stop); frames never pushed (or too old) -> fill."""
        idx = np.arange(start, stop)
        ok = (idx >= max(0, self.n - self.cap)) & (idx < self.n)
        out = np.empty((len(idx), self.buf.shape[1]), self.buf.dtype)
        out[:] = self.fill
        out[ok] = self.buf[idx[ok] % self.cap]
        return out, ok

    def last(self):
        return self.buf[(self.n - 1) % self.cap] if self.n else None


class _CopyDelayCore:
    """Stub/baseline: mirror the user with a delay (groups the user lacks hold their last value)."""

    kind = "copy_delay"

    def __init__(self, cfg):
        self.delay = int(cfg.get("delay_frames", 30))
        self.horizon = int(cfg.get("horizon", 15))
        self.self_audio_lookahead = int(cfg.get("self_audio_lookahead", 6))
        self.smooth = float(cfg.get("smooth", 0.0))      # EMA factor on the copied user (0 = none)

    def reset(self):
        self._ema = None

    def plan(self, hist, t, known_vals, known_gmask):
        """frames [t, t+horizon) as (H,189)."""
        H = self.horizon
        last = hist.self_motion.last()
        cur = (REST_POSE if last is None else last).astype(np.float32).copy()
        um, uok = hist.user_motion.get(t - self.delay, t - self.delay + H)
        uk, _ = hist.user_mask.get(t - self.delay, t - self.delay + H)
        out = np.empty((H, D), np.float32)
        for i in range(H):
            if uok[i]:
                mm = spec.mirror_motion(um[i])
                mk = spec.mirror_mask(uk[i])
                for g in range(G):
                    if mk[g]:
                        c = GROUP_COLS[g]
                        cur[c] = mm[c] if self.smooth <= 0 else (self.smooth * cur[c] + (1 - self.smooth) * mm[c])
            out[i] = cur
        if known_vals is not None:
            cm = spec.col_mask(known_gmask).astype(bool)
            out[cm] = known_vals[cm]
        return out


class UnifiedPlanner:
    def __init__(self, bundle_dir=None, threads=2, seed=0, replan_every=None, steps=None, guidance=None,
                 temperature=None, pcm=None, **kw):
        """Every sampling knob left None takes the bundle's config["runtime"] default, then the built-in one
        (RUNTIME_DEFAULTS). blend = frames over which a new plan is ramped in over the previous plan's overlapping
        frames (0 = switch); consistent_noise = reuse the overlapping noise between plans; smooth = one-pole
        low-pass factor on the model output (0 = off; last resort)."""
        self.bundle_dir = bundle_dir
        cfg = {"kind": "copy_delay"}
        if bundle_dir is not None:
            with open(os.path.join(bundle_dir, "config.json")) as f:
                cfg = json.load(f)
        self.cfg = cfg
        rd = dict(RUNTIME_DEFAULTS, **(cfg.get("runtime") or {}))
        pick = lambda k, v: rd[k] if v is None else v          # noqa: E731
        replan_every, steps = pick("replan_every", replan_every), pick("steps", steps)
        guidance, temperature = pick("guidance", guidance), pick("temperature", temperature)
        for k in ("blend", "consistent_noise", "smooth", "noise_rho", "temperature_groups", "ensemble", "prefix", "box",
                  "box_groups", "anchor"):
            kw[k] = pick(k, kw.get(k))
        self.runtime = dict(replan_every=replan_every, steps=steps, guidance=guidance, temperature=temperature,
                            blend=kw["blend"], consistent_noise=kw["consistent_noise"], smooth=kw["smooth"],
                            noise_rho=kw["noise_rho"], temperature_groups=kw["temperature_groups"])
        self._init_gain(pick("input_gain_db", kw.pop("input_gain_db", None)))
        self._anchor_cfg = kw.pop("anchor", None)
        self._init_seam(kw)
        self.replan_every = max(1, int(replan_every))
        self.blend = int(kw.pop("blend"))
        self.smooth = float(kw.pop("smooth"))
        self.xfade = 0 if self.blend > 0 else int(kw.pop("xfade", 3))
        kw.pop("xfade", None)
        self.xfade_tol = float(kw.pop("xfade_tol", 0.02))   # mean |diff| (rad / blendshape units) to trigger it
        ev_kw = kw.pop("events", None)
        if cfg.get("kind", "flow") != "flow":
            for k in ("consistent_noise", "noise_rho", "temperature_groups", "feed_every_frame", "prefix"):
                kw.pop(k, None)
        kind = cfg.get("kind", "flow")
        if kind == "copy_delay":
            self.core = _CopyDelayCore(cfg)
        elif kind == "flow":
            from unified.flow_core import FlowCore
            self.core = FlowCore(bundle_dir, cfg, threads=threads, seed=seed, steps=steps, guidance=guidance,
                                 temperature=temperature, **kw)
        else:
            raise ValueError(f"unknown bundle kind {kind!r}")
        if getattr(self.core, "head_kind", "flow") == "regression":      # effective (forced) sampling values
            self.runtime.update(steps=self.core.steps, temperature=self.core.temperature, temperature_groups=None)
        if self._prefix_cycle is not None:
            self.runtime["prefix"] = list(self._prefix_cycle) if len(self._prefix_cycle) > 1 else self._prefix_cycle[0]
        elif hasattr(self.core, "C"):
            self.runtime["prefix"] = int(self.core.C)
        self.self_audio_lookahead = int(self.core.self_audio_lookahead)
        self._init_pcm(bundle_dir, cfg, pcm, threads)
        self.seed = seed
        self._init_anchor(bundle_dir)
        self._init_events(bundle_dir, cfg, rd, ev_kw)
        self.reset()

    def _init_gain(self, g):
        """input_gain_db = None | dB | {"user": dB, "self": dB}: level calibration of the audio inputs to the
        training corpus. The bundles trained on dataset levels (Seamless speech frames ~ -65 dB loud_db) while the
        page's mic (browser AGC) and TTS sit ~20 dB higher (the scorecard harness normalises its clips to those live
        levels: +23 dB user, +19 dB self on speech frames). Applied to audio42 (log-mel +g ln10/10, loud_db +g; floor
        values = digital silence stay; vad unchanged) and to PCM samples (x 10^(g/20)) before the speech encoder.
        Precomputed user_feat / self_feat_ahead are NOT changed (they come from the offline pipeline's own levels)."""
        if isinstance(g, dict):
            gu, gs = float(g.get("user", 0.0) or 0.0), float(g.get("self", 0.0) or 0.0)
        else:
            gu = gs = float(g or 0.0)
        self._gain = (gu, gs)
        self.runtime["input_gain_db"] = None if gu == 0 and gs == 0 else {"user": gu, "self": gs}

    @staticmethod
    def _gain_a42(a, g):
        if a is None or g == 0:
            return a
        a = np.array(a, np.float32, copy=True).reshape(A)
        fl = math.log(1e-6)
        mel = a[:40]
        a[:40] = np.where(mel <= fl + 1e-3, mel, np.maximum(mel + g * math.log(10) / 10, fl))
        if a[40] > -100 + 1e-3:
            a[40] = max(a[40] + g, -100.0)
        return a

    @staticmethod
    def _gain_pcm(x, g):
        if x is None or g == 0:
            return x
        return (np.asarray(x, np.float32) * np.float32(10 ** (g / 20))).astype(np.float32)

    def _init_anchor(self, bundle_dir):
        """anchor = None | {"cols": [canon column names, e.g. "bs_eyeBlinkLeft"], "tau_s": 4.0}: a slow level
        anchor for columns whose closed-loop LEVEL drifts (exposure bias; every p2abl bundle at T0 drifts its eyes to
        eyeBlink ~.5 = half-closed 60% of frames vs the people's 15%, and its smile to .5 vs .3). out = x - lp(x) +
        target, lp = one-pole low-pass (time constant tau_s, starts at target), target = the bundle's training mean
        (norm.npz); fast dynamics (blinks, smile onsets) pass, the slow drift is removed. Applied before events and
        overrides; the history holds the anchored values (the model sees open eyes and does not ratchet)."""
        self.anchor_cols = None
        a = self._anchor_cfg
        if not a or not bundle_dir:
            self.runtime["anchor"] = None
            return
        cols = [spec.COL[c] if isinstance(c, str) else int(c) for c in a.get("cols", [])]
        if not cols:
            self.runtime["anchor"] = None
            return
        z = np.load(os.path.join(bundle_dir, "norm.npz"))
        self.anchor_cols = np.asarray(cols, np.int64)
        self.anchor_target = z["mean"].astype(np.float32)[self.anchor_cols]
        self.anchor_k = 1.0 / max(float(a.get("tau_s", 4.0)) * FPS, 1.0)
        self.runtime["anchor"] = {"cols": [spec.MOTION_COLS[c] for c in cols], "tau_s": float(a.get("tau_s", 4.0))}

    def _anchor_frame(self, model):
        x = model[self.anchor_cols]
        self._alp = self._alp + self.anchor_k * (x - self._alp)
        model = model.copy()
        y = x - self._alp + self.anchor_target
        if self.anchor_cols.min() >= 135:
            y = np.clip(y, 0.0, 1.0)
        model[self.anchor_cols] = y
        return model

    def _init_pcm(self, bundle_dir, cfg, pcm, threads):
        """PCM mode (enc bundles with config["audio"]["encoder"]; pcm None = auto, False = off, True = required)."""
        self._pcm = None
        meta = (cfg.get("audio") or {}).get("encoder") if self.feat_dim > 0 else None
        if pcm is None:
            pcm = bool(meta)
        if pcm and self.feat_dim > 0:
            if not meta:
                raise ValueError("pcm=True needs the encoder spec config['audio']['encoder'] in the bundle")
            from unified.audio_feats import PcmFeatures
            self._pcm = PcmFeatures(meta, bundle_dir, user_shift=self.user_feat_shift,
                                    self_lookahead=self.self_audio_lookahead, feat_dim=self.feat_dim, threads=threads)
        self.pcm_mode = self._pcm is not None
        self.self_pcm_lookahead = self._pcm.self_pcm_lookahead if self.pcm_mode else None
        self.runtime["pcm"] = (dict(self._pcm.spec, self_pcm_lookahead=self.self_pcm_lookahead)
                               if self.pcm_mode else None)

    def _init_seam(self, kw):
        """plan-combination knobs (seam fix; each None/0 = the old behaviour, bit for bit):
        ensemble = None | {"plans": n, "weights": "uniform" | "exp", "tau": frames}: every output frame is the weighted
          mean of the newest n plans that cover it (weight 1 or exp(-age / tau), age = frames since that plan was
          made); replaces the blend/xfade ramp. The history the model conditions on is the ensembled output.
        prefix = None (the bundle's config["prefix"]) | C | [C0, C1, ...]: committed known frames per plan; a list is
          cycled plan by plan, which shifts the head's patch grid between consecutive plans (a patch-phase-locked
          artifact then averages out in the ensemble). Training sampled C in 0..8 uniformly.
        box = 0 | n | {"n": n, "ahead": a, "freeze": True}: n-frame moving average of the combined plan output x,
          a of the n frames ahead of the current one (default n // 2 = centred: no added latency; a = 0 = causal, the
          motion lags (n - 1) / 2 frames). freeze (default): a frame of x is fixed the first time a window reads it (a
          look-ahead frame comes from the plan current at that step and a later plan does not replace it), so x is
          one sequence and the box removes every period-n component of it exactly (box 3 = the head's 3-frame patch
          sawtooth, the abl_t0 10 Hz line); freeze False = look-ahead frames are re-read from each newer plan."""
        ens = kw.pop("ensemble")
        if ens:
            ens = dict(ENSEMBLE_DEFAULTS, **(ens if isinstance(ens, dict) else {}))
            if ens["weights"] not in ENSEMBLE_WEIGHTS:
                raise ValueError(f"unknown ensemble weights {ens['weights']!r} (want {ENSEMBLE_WEIGHTS})")
            ens = dict(plans=max(1, int(ens["plans"])), weights=str(ens["weights"]), tau=float(ens["tau"]))
        self.ens = ens or None
        pf = kw.pop("prefix")
        self._prefix_cycle = None
        if pf is not None:
            pl = [int(c) for c in (pf if isinstance(pf, (list, tuple)) else [pf])]
            if not pl:
                raise ValueError("prefix: empty list")
            self._prefix_cycle = pl
            kw["prefix"] = pl[0]                     # FlowCore's starting prefix
        self._box_groups_cfg = kw.pop("box_groups", None)
        bx = kw.pop("box") or 0
        bx = dict(bx) if isinstance(bx, dict) else {"n": int(bx)}
        n = int(bx.get("n", 3))
        self.box = n if n > 1 else 0
        self.box_ahead = min(max(int(bx.get("ahead", n // 2)), 0), self.box - 1) if self.box else 0
        self.box_freeze = bool(bx.get("freeze", True))
        self.runtime.update(ensemble=None if self.ens is None else dict(self.ens),
                            box=None if not self.box else dict(n=self.box, ahead=self.box_ahead,
                                                               freeze=self.box_freeze))
        # box_groups = None | {group: n}: an extra CENTRED n-frame moving average (n odd, ahead n // 2: no lag)
        # of the combined plan output x on that group's columns, replacing the global box there. Past frames = the
        # x ring (frozen as committed), look-ahead frames = the ring where already frozen, else the newest plan. For
        # slow groups whose closed-loop output carries 4-15 Hz noise leaking from the sampled groups (the torso at
        # T>0: spine_x / root_y 4-15 Hz power +1.2..1.5 log10 over people even at torso temperature 0.1). n = 9 (a
        # multiple of the head's 3-frame patch) also removes the patch sawtooth exactly.
        self.box_groups = []
        for g, n in dict(self._box_groups_cfg or {}).items():
            n = int(n)
            if n > 1:
                n += 1 - n % 2                                  # odd
                self.box_groups.append((GROUP_COLS[GI[g]], n))
        self.runtime["box_groups"] = ({g: int(n) + 1 - int(n) % 2 for g, n in dict(self._box_groups_cfg).items()
                                       if int(n) > 1} if self.box_groups else None)

    def _init_events(self, bundle_dir, cfg, rd, ev_kw):
        """event head bundles: names / thresholds / refractory / groups (config["events"]), curves (events.npz),
        firing settings (config["runtime"]["events"] < the events= kwarg)."""
        self.n_events = int(getattr(self.core, "n_events", 0) or 0)
        self.event_names = []
        if self.n_events <= 0:
            return
        n = self.n_events
        ec = cfg.get("events") or {}
        self.event_names = [str(x) for x in (ec.get("names") or [f"event{i}" for i in range(n)])]
        self.ev_thr = np.asarray(ec.get("thresholds", [0.5] * n), np.float32).reshape(n)
        self.ev_refr = np.asarray(ec.get("refractory", [0] * n), np.int64).reshape(n)
        groups = ec.get("groups") or [None] * n
        self.ev_gid = [None if g is None else GI[g] for g in groups]
        assert len(self.event_names) == n and len(self.ev_gid) == n, (self.event_names, groups)
        z = np.load(os.path.join(bundle_dir, "events.npz"))
        self.ev_curves = z["curves"].astype(np.float32)
        self.ev_lens = z["lens"].astype(np.int64).reshape(n)
        assert self.ev_curves.shape[0] == n and self.ev_curves.shape[2] == D, self.ev_curves.shape
        assert (self.ev_lens <= self.ev_curves.shape[1]).all(), (self.ev_lens, self.ev_curves.shape)
        es = dict(EVENT_DEFAULTS, **((rd.get("events") or {}) if isinstance(rd.get("events"), dict) else {}))
        es.update({"mode": ev_kw} if isinstance(ev_kw, str) else (ev_kw or {}))     # events="off" = {"mode": "off"}
        mode = str(es["mode"])
        if mode not in EVENT_MODES:
            raise ValueError(f"unknown events mode {mode!r} (want {EVENT_MODES})")
        self.ev_mode = mode
        self.ev_gain = np.broadcast_to(np.asarray(es["gain"], np.float32), (n,)).copy()
        self.ev_feed_history = bool(es["feed_history"])
        # closed-loop calibration keys (absent = config["events"] values / the old rule, bit for bit):
        # thresholds / refractory replace config["events"]'s (the training-side calibration is teacher-forced; the
        # closed loop's p is lower and flatter); thresholds_speak = the thresholds while the avatar speaks (own vad,
        # dilated as in training); scale = sample mode fires with probability min(1, scale[c] * p[c])
        bc = lambda v, dt: np.broadcast_to(np.asarray(v, dt), (n,)).copy()          # noqa: E731
        if es.get("thresholds") is not None:
            self.ev_thr = bc(es["thresholds"], np.float32)
        if es.get("refractory") is not None:
            self.ev_refr = bc(es["refractory"], np.int64)
        self.ev_thr_s = None if es.get("thresholds_speak") is None else bc(es["thresholds_speak"], np.float32)
        self.ev_scale = bc(1.0 if es.get("scale") is None else es["scale"], np.float32)
        self.runtime["events"] = dict(mode=mode, gain=[float(g) for g in self.ev_gain],
                                      feed_history=self.ev_feed_history, thresholds=[float(x) for x in self.ev_thr],
                                      refractory=[int(x) for x in self.ev_refr])
        if self.ev_thr_s is not None:
            self.runtime["events"]["thresholds_speak"] = [float(x) for x in self.ev_thr_s]
        if es.get("scale") is not None:
            self.runtime["events"]["scale"] = [float(x) for x in self.ev_scale]

    @property
    def audio_in(self):
        """"mel" (audio42 only) or "enc_<name>" (the bundle also reads the speech encoder's features)."""
        return getattr(self.core, "audio_in", "mel")

    @property
    def feat_dim(self):
        """De = raw encoder feature dims step() takes as user_feat / self_feat_ahead (0 for mel bundles)."""
        return int(getattr(self.core, "feat_dim", 0) or 0)

    @property
    def user_feat_shift(self):
        """s (frames): user_feat at step t is the encoder's frame t - s (0 for mel bundles)."""
        return int(getattr(self.core, "user_feat_shift", 0) or 0)

    # ------------------------------------------------------------------ state
    def reset(self):
        cap = 512
        self.hist = type("Hist", (), {})()
        self.hist.self_motion = Ring(D, cap)
        self.hist.user_motion = Ring(D, cap)
        self.hist.user_mask = Ring(G, cap, np.float32)
        self.hist.user_audio = Ring(A, cap, fill=SILENCE_AUDIO)
        self.hist.self_audio_ahead = Ring(A, cap, fill=SILENCE_AUDIO)   # row s = avatar audio of frame s+L
        self.hist.style = Ring(len(STYLE_NULL), cap, fill=STYLE_NULL)    # row s = style input at frame s
        if self.feat_dim > 0:          # enc bundles: rows = the (k,) normalized projected feature (fill = silence)
            k = len(self.core.feat_fill)
            self.hist.user_feat = Ring(k, cap, fill=self.core.feat_fill)        # row s = user feature given at step s
            self.hist.self_feat_ahead = Ring(k, cap, fill=self.core.feat_fill)  # row s = avatar feature of s+L
        if self.n_events > 0:
            self._ev_rng = np.random.default_rng([int(self.seed or 0), 0xE7E7])
            self._ev_prev = np.full(self.n_events, -1.0, np.float32)    # p at the previous frame (-1 = none yet)
            self._ev_last = np.full(self.n_events, -(10 ** 9), np.int64)  # frame of the last fire per class
            self._ev_active = []                                          # [(class, onset frame)]
        self.t = 0
        self._plan = None            # (t0, frames (H,189))
        self._xf = None              # (t0, old frames) being crossfaded out
        self._force_plan = True
        self._plans = []             # ensemble: [(t0, frames)] newest last
        self._n_plans = 0
        self._raw = Ring(D, 64) if (self.box or self.box_groups) else None   # box: the plan output before averaging
        self._own = np.zeros(G, np.float32)
        self._own_target = np.zeros(G, np.float32)
        self._own_rate = np.zeros(G, np.float32)
        self._ov = [None] * G        # per group: ("const", vec) | ("call", fn, t_start)
        self._sm = None
        if getattr(self, "anchor_cols", None) is not None:
            self._alp = self.anchor_target.copy()
        if not hasattr(self, "_style"):
            self._style = STYLE_NULL.copy()
        self.last_feats = None
        if getattr(self, "_pcm", None) is not None:
            self._pcm.reset()
        self.core.reset()

    # ------------------------------------------------------------------ style
    def set_style(self, style=None):
        """style: None -> all components inferred; dict {energy|mirroring|expressiveness: float in [0,1] | None}
        -> updates only the given keys (None = that key inferred). Unknown keys raise. Survives reset()."""
        if style is None:
            self._style = STYLE_NULL.copy()
            return self.style
        cur = self._style.copy()
        for k, v in dict(style).items():
            if k not in STYLE_NAMES:
                raise KeyError(f"unknown style key {k!r} (want {STYLE_NAMES})")
            j = STYLE_NAMES.index(k)
            if v is None:
                cur[j], cur[3 + j] = 0.0, 1.0
            else:
                cur[j], cur[3 + j] = float(np.clip(float(v), 0.0, 1.0)), 0.0
        self._style = cur.astype(np.float32)
        return self.style

    @property
    def style(self):
        """current setting as {name: value | None (inferred)}."""
        return {n: (None if self._style[3 + j] > 0.5 else float(self._style[j])) for j, n in enumerate(STYLE_NAMES)}

    # ------------------------------------------------------------------ overrides
    def _group_ids(self, groups):
        if groups is None:
            return [g for g in range(G) if self._own[g] > 0 or self._own_target[g] > 0]
        if isinstance(groups, str):
            groups = [groups]
        return [GI[g] for g in groups]

    def set_override(self, groups, values=None, fade_in_s=0.15):
        ids = self._group_ids(groups)
        last = self.hist.self_motion.last()
        hold = (REST_POSE if last is None else last).astype(np.float32).copy()
        for g in ids:
            if values is None:
                src = ("const", hold)
            elif callable(values):
                src = ("call", values, self.t)
            else:
                v = np.asarray(values, np.float32).reshape(D)
                src = ("const", v.copy())
            already = self._own_target[g] == 1 and self._own[g] > 0
            self._ov[g] = src
            if not already:
                self._own_target[g] = 1.0
                n = max(fade_in_s, 0.0) * FPS
                self._own_rate[g] = 1.0 / n if n >= 1 else 1.0
        self._force_plan = True

    def release_override(self, groups=None, fade_out_s=0.4):
        ids = self._group_ids(groups)
        for g in ids:
            if self._ov[g] is not None:
                self._ov[g] = ("const", self._ov_value(g, self.t))
            self._own_target[g] = 0.0
            n = max(fade_out_s, 0.0) * FPS
            self._own_rate[g] = 1.0 / n if n >= 1 else 1.0
        self._force_plan = True

    def _ov_value(self, g, frame):
        src = self._ov[g]
        if src is None:
            return REST_POSE.copy()
        if src[0] == "const":
            return src[1]
        fn, t0 = src[1], src[2]
        return np.asarray(fn(int(frame - t0)), np.float32).reshape(D)

    def _known(self, t0, H):
        """known values / group mask for frames [t0, t0+H): a group an action holds (target 1) is known from the
        first frame at which its PROJECTED ownership weight reaches 1. Frames of the fade-in ramp stay unknown: the
        model inpaints its own transition into the action pose and the output blends model -> action by the ramp."""
        held = [g for g in range(G) if self._own_target[g] == 1 and self._ov[g] is not None]
        if not held:
            return None, None
        vals = np.zeros((H, D), np.float32)
        gm = np.zeros((H, G), np.float32)
        for g in held:
            c = GROUP_COLS[g]
            w0, r = float(self._own[g]), float(self._own_rate[g])
            for i in range(H):
                if min(1.0, w0 + r * i) >= 1.0 - 1e-6:
                    vals[i, c] = self._ov_value(g, t0 + i)[c]
                    gm[i, g] = 1
        if not gm.any():
            return None, None
        return vals, gm

    # ------------------------------------------------------------------ main
    def _fire_events(self, p, t):
        """firing rule at frame t given onset probabilities p (n,): returns the names fired."""
        n = self.n_events
        u = self._ev_rng.random(n) if self.ev_mode == "sample" else None
        thr = self.ev_thr
        if self.ev_thr_s is not None and hasattr(self.core, "speaking") and bool(self.core.speaking(self.hist, t, t + 1)[0]):
            thr = self.ev_thr_s
        fired = []
        for c in range(n):
            if self.ev_mode == "threshold":
                cand = p[c] >= thr[c] and self._ev_prev[c] < thr[c]
            elif self.ev_mode == "sample":
                cand = u[c] < min(1.0, float(self.ev_scale[c]) * p[c])
            else:
                cand = False
            if not cand or t - self._ev_last[c] < self.ev_refr[c]:
                continue
            g = self.ev_gid[c]
            if g is not None and self._own[g] > 0:
                continue
            self._ev_last[c] = t
            self._ev_active.append((c, t))
            fired.append(self.event_names[c])
        self._ev_prev = np.asarray(p, np.float32).copy()
        return fired

    def _event_delta(self, t):
        """sum of the active events' curve rows at frame t (189,) or None; drops finished events."""
        if not self._ev_active:
            return None
        d = np.zeros(D, np.float32)
        keep = []
        for c, t0 in self._ev_active:
            k = t - t0
            if 0 <= k < self.ev_lens[c]:
                d += self.ev_gain[c] * self.ev_curves[c, k]
                if k + 1 < self.ev_lens[c]:
                    keep.append((c, t0))
        self._ev_active = keep
        return d

    def _plan_frame(self, f):
        """the combined plan value for frame f (>= the newest plan's start): the newest plan's row, or with ensemble
        the weighted mean of the kept plans covering f (frames past every plan: the newest plan's last row)."""
        t0, fr = self._plan
        if self.ens is None or len(self._plans) <= 1:
            return fr[min(f - t0, len(fr) - 1)].astype(np.float32).copy()
        num = np.zeros(D, np.float64)
        den = 0.0
        exp_w = self.ens["weights"] == "exp"
        for a, fa in self._plans:
            age = f - a
            if 0 <= age < len(fa):
                w = math.exp(-age / max(self.ens["tau"], 1e-6)) if exp_w else 1.0
                num += w * fa[age]
                den += w
        if den <= 0:
            return fr[min(f - t0, len(fr) - 1)].astype(np.float32).copy()
        return (num / den).astype(np.float32)

    def _box_frame(self, t, model):
        """box: moving average of the combined plan output x over n frames (n - 1 - ahead past, this one, `ahead`
        ahead). freeze: x(f) is fixed when a window first reads it (frame t's own value too, if an earlier window
        already read it as look-ahead); the ring holds x for every frame read so far."""
        ah, raw = self.box_ahead, self._raw
        if self.box_freeze:
            if raw.n <= t:
                raw.push(model)
            for f in range(raw.n, t + ah + 1):
                raw.push(self._plan_frame(f))
            r, ok = raw.get(t - (self.box - 1 - ah), t + ah + 1)
            return r[ok].mean(0).astype(np.float32)
        rows = [model]
        past = self.box - 1 - ah
        if past > 0:
            r, ok = raw.get(t - past, t)
            rows.extend(r[ok])
        for k in range(1, ah + 1):
            rows.append(self._plan_frame(t + k))
        raw.push(model)
        return np.mean(rows, axis=0).astype(np.float32)

    def _box_groups_frame(self, t, model):
        """per-group centred boxes (see box_groups): x(f) from the ring when frozen, else the newest plan."""
        raw = self._raw
        model = model.copy()
        for cols, n in self.box_groups:
            ah = n // 2
            acc = np.zeros(len(cols), np.float64)
            k = 0
            for f in range(t - (n - 1 - ah), t + ah + 1):
                if f < 0:
                    continue
                if raw.n - raw.cap <= f < raw.n:
                    acc += raw.buf[f % raw.cap][cols]
                else:
                    acc += self._plan_frame(f)[cols]
                k += 1
            model[cols] = (acc / max(k, 1)).astype(np.float32)
        return model

    def _blend_overrides(self, model, t):
        """override blending of the owned groups into `model` (in place; also returns it)."""
        for g in range(G):
            w = float(self._own[g])
            if w > 0 and self._ov[g] is not None:
                c = GROUP_COLS[g]
                model[c] = blend_cols(c, model[c], self._ov_value(g, t)[c], w)
        return model

    def step(self, user_motion189=None, user_mask9=None, user_audio42=None, self_audio42_ahead=None,
             user_feat=None, self_feat_ahead=None, user_pcm=None, self_pcm_ahead=None):
        """one 30 Hz frame. user_feat / self_feat_ahead: raw (De,) speech-encoder features (enc bundles only; see the
        module doc for their framing); ignored by mel bundles. user_pcm / self_pcm_ahead: 16 kHz samples of the user's
        frame t / the avatar's frame t + p.self_pcm_lookahead (PCM mode; None = silence)."""
        t = self.t
        enc_ms = None
        gu, gs = self._gain
        if gu:
            user_audio42, user_pcm = self._gain_a42(user_audio42, gu), self._gain_pcm(user_pcm, gu)
        if gs:
            self_audio42_ahead, self_pcm_ahead = self._gain_a42(self_audio42_ahead, gs), self._gain_pcm(self_pcm_ahead, gs)
        if self.feat_dim > 0:                            # PCM mode first: bad PCM raises before any state changes
            if self._pcm is not None:                    # the encoders' rows; explicitly passed features win
                uf, sf, enc_ms = self._pcm.step(t, user_pcm, self_pcm_ahead)
                user_feat = uf if user_feat is None else user_feat
                self_feat_ahead = sf if self_feat_ahead is None else self_feat_ahead
            elif user_pcm is not None or self_pcm_ahead is not None:
                raise ValueError("PCM given but this planner has no speech encoder (bundle without "
                                 "config['audio']['encoder'], or pcm=False): pass user_feat / self_feat_ahead")
        if user_motion189 is None:
            self.hist.user_motion.push(np.zeros(D, np.float32))
            self.hist.user_mask.push(np.zeros(G, np.float32))
        else:
            um = np.asarray(user_motion189, np.float32).reshape(D)
            uk = np.ones(G, np.float32) if user_mask9 is None else np.asarray(user_mask9, np.float32).reshape(G)
            if not np.isfinite(um).all():
                bad = ~np.isfinite(um)
                uk = uk.copy(); uk[np.unique(spec.GROUP_OF_COL[bad])] = 0
                um = np.where(bad, 0, um)
            self.hist.user_motion.push(um)
            self.hist.user_mask.push(uk)
        self.hist.user_audio.push(SILENCE_AUDIO if user_audio42 is None else np.asarray(user_audio42, np.float32))
        self.hist.self_audio_ahead.push(SILENCE_AUDIO if self_audio42_ahead is None
                                        else np.asarray(self_audio42_ahead, np.float32))
        self.hist.style.push(self._style)
        if self.feat_dim > 0:
            self.last_feats = {"user": user_feat, "self": self_feat_ahead}
            self.hist.user_feat.push(self.core.feat_row(user_feat))
            self.hist.self_feat_ahead.push(self.core.feat_row(self_feat_ahead))

        ev_p = None
        if hasattr(self.core, "on_frame"):
            ev_p = self.core.on_frame(self.hist, t)      # per-frame work: feed the newest token (+ event head)

        # override weights advance first, so a set_override() before step() shows on this frame
        up = self._own_target > self._own
        dn = self._own_target < self._own
        self._own[up] = np.minimum(self._own[up] + self._own_rate[up], 1.0)
        self._own[dn] = np.maximum(self._own[dn] - self._own_rate[dn], 0.0)
        for g in range(G):
            if self._own[g] == 0 and self._own_target[g] == 0:
                self._ov[g] = None

        plan_ms = None
        need = (self._force_plan or self._plan is None or t - self._plan[0] >= self.replan_every
                or t - self._plan[0] >= len(self._plan[1]))
        if need:
            if self._prefix_cycle is not None and len(self._prefix_cycle) > 1 and hasattr(self.core, "set_prefix"):
                self.core.set_prefix(self._prefix_cycle[self._n_plans % len(self._prefix_cycle)])
            H = self.core.horizon
            kv, kg = self._known(t, H)
            t1 = time.perf_counter()
            frames = np.asarray(self.core.plan(self.hist, t, kv, kg), np.float32)
            plan_ms = (time.perf_counter() - t1) * 1000.0
            self._n_plans += 1
            if self.ens is not None:                              # keep the newest plans (raw) that still cover t
                self._plans = [(a, f) for a, f in self._plans if t - a < len(f)][-(self.ens["plans"] - 1):] \
                    if self.ens["plans"] > 1 else []
                self._plans.append((t, frames))
            elif self.blend > 0 and self._plan is not None and t - self._plan[0] < len(self._plan[1]):
                old = self._plan[1][t - self._plan[0]:]           # ramp the new plan in over the overlap
                n = min(len(old), len(frames))
                w = np.minimum(1.0, (np.arange(n) + 1.0) / (self.blend + 1.0))[:, None].astype(np.float32)
                frames[:n] = (1 - w) * old[:n] + w * frames[:n]
            if self.ens is None and self._plan is not None and t - self._plan[0] < len(self._plan[1]) \
                    and self.xfade > 0:
                old = self._plan[1][t - self._plan[0]:]
                if np.mean(np.abs(old[0] - frames[0])) > self.xfade_tol:
                    self._xf = (t, old[: self.xfade].copy())
                else:
                    self._xf = None
            self._plan = (t, frames)
            self._force_plan = False
        model = self._plan_frame(t)
        if self._xf is not None:
            i = t - self._xf[0]
            if i < len(self._xf[1]):
                a = (i + 1) / (len(self._xf[1]) + 1)
                model = (1 - a) * self._xf[1][i] + a * model
            else:
                self._xf = None
        x_t = model
        if self.box:
            model = self._box_frame(t, model)
        elif self._raw is not None and self._raw.n <= t:
            self._raw.push(x_t)
        if self.box_groups:
            model = self._box_groups_frame(t, model)
        if self.anchor_cols is not None:
            model = self._anchor_frame(model)

        if self.smooth > 0:
            if self._sm is not None:
                model = self.smooth * self._sm + (1 - self.smooth) * model
            self._sm = model.copy()
        fired, delta = [], None
        if self.n_events > 0:
            if ev_p is not None:
                fired = self._fire_events(ev_p, t)
            delta = self._event_delta(t)
        if delta is None:
            motion = self._blend_overrides(model, t).astype(np.float32)
            pushed = motion
        else:
            base = model.copy() if not self.ev_feed_history else None
            model = model + delta
            model[135:187] = np.clip(model[135:187], 0.0, 1.0)
            motion = self._blend_overrides(model, t).astype(np.float32)
            pushed = motion if base is None else self._blend_overrides(base, t).astype(np.float32)
        self.hist.self_motion.push(pushed)
        self.t += 1
        return {"motion": motion, "owned": self._own.copy(), "plan_ms": plan_ms, "events": fired,
                "event_p": None if ev_p is None else np.asarray(ev_p, np.float32).copy(), "enc_ms": enc_ms}
