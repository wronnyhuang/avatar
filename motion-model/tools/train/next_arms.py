"""arm ablation: per-arm training treatments against (1) the model ignoring the user and (2) closed-loop stillness/drift.

Used by train_multi.py: a model's config key "arm" = {"hist": {...}, "react": {...}, "contrast": {...}, "stream": "alt"}
(every part optional; no "arm" = train.py exactly). Treatments are applied to a SHALLOW COPY of the lane's shared batch
(modified tensors are cloned), with the model's own global-RNG stream (train_on swaps it in), so arms never perturb
each other and the base arm sees the sweep's batches.

hist  (a) SELF-HISTORY CORRUPTION of the backbone tokens' self-motion block (targets x1 / known prefix kvals stay clean,
      so the model learns to re-derive its motion from the user / audio instead of copying its own past):
        p_drift, drift_sd  per sample: a random-walk (low-frequency) offset per column, end-of-window std drift_sd*U(0,1)
                           (closed-loop drift: the model's own levels wander off the data)
        p_freeze, freeze_len  per sample and group (p 0.5): a sample-and-hold block of U(freeze_len) frames
                           (closed-loop stillness: the history froze, the target still moves)
        p_gdrop            per sample and group: that group's whole self history masked (value 0, mask 0)
react (b) REACTION-AWARE WEIGHTING: a self onset (nod / smile) preceded by a partner event (nod / smile onset, speech
      start / stop) 0.3-2.0 s earlier is a "reaction"; flow-loss weight x (1 + flow_w) on the event's columns within
      radius frames of it, x (1 + mimic_w) more when the classes match (smile -> smile, nod -> nod); event-head ok
      weight x (1 + event_w) at reaction onsets (+-tol). Partner events are read from the Sampler (smp.EV / smp.A) at
      the partner's frames of the batch rows (pick / s0).
contrast (c) CONTINGENCY HEAD: an MLP on the backbone output classifies, per token (t >= t0), whether the partner stream
      in the tokens is the REAL partner or a shuffled one (half: another row's partner, half: the same partner shifted
      1.5-3.5 s in time). frac of the batch rows get a shuffled twin; the backbone runs once on [batch | twins].
      loss weight w; the head is not part of the bundle (saved beside ckpt_last as ckpt_aux.pt for resume).
stream "alt": the model trains on the lane's second batch stream (top-level "arm_streams": {"alt": {"dataset_weights":
      {...}}}), e.g. CANDOR up-weighted while the main stream excludes it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unified.flow import flow_loss
from unified.model import layout

FPS = 30
HIST_DEFAULT = dict(p_drift=0.5, drift_sd=0.6, p_freeze=0.35, freeze_len=[15, 75], p_gdrop=0.1)
REACT_DEFAULT = dict(lag=[0.3, 2.0], flow_w=2.0, mimic_w=2.0, event_w=3.0, radius=[10, 4, 15], self_classes=[0, 2],
                     partner_classes=[0, 2], speech=True)
CONTRAST_DEFAULT = dict(w=0.2, frac=0.5, t0=30, shift=[45, 105], hidden=256)
# reaction adapter (rate / inject; "trace" = cfg["model"]["react_trace"], the partner-trace adapter, unified/react.py)
RATE_DEFAULT = dict(w=0.05, lag=[0.3, 2.0], pairs=[["smile", 2], ["nod", 0], ["laugh", 2]], ema=0.98, min_frames=30)
INJECT_DEFAULT = dict(f_smile=0.12, f_nod=0.15, lag=[12, 27], amp=[0.8, 1.2], laugh=True, react_lag=[0.3, 2.0])


def arm_of(cfg):
    a = cfg.get("arm") or {}
    out = {}
    if a.get("hist"):
        out["hist"] = dict(HIST_DEFAULT, **(a["hist"] if isinstance(a["hist"], dict) else {}))
    if a.get("react"):
        out["react"] = dict(REACT_DEFAULT, **(a["react"] if isinstance(a["react"], dict) else {}))
    if a.get("contrast"):
        out["contrast"] = dict(CONTRAST_DEFAULT, **(a["contrast"] if isinstance(a["contrast"], dict) else {}))
    if a.get("rate"):
        out["rate"] = dict(RATE_DEFAULT, **(a["rate"] if isinstance(a["rate"], dict) else {}))
    if a.get("inject"):
        out["inject"] = dict(INJECT_DEFAULT, **(a["inject"] if isinstance(a["inject"], dict) else {}))
    if a.get("trace"):
        out["trace"] = dict(hidden=64, **(a["trace"] if isinstance(a["trace"], dict) else {}))
    out["stream"] = a.get("stream", "main")
    return out


def stream_weights(smp, dataset_weights):
    """the per-stream sampling weight array the Sampler would build for these dataset weights (None = uniform)."""
    if not dataset_weights:
        return None
    dl = smp.names.get("datasets", [])
    return np.array([float(dataset_weights.get(dl[d], 1.0)) if d < len(dl) else 1.0 for d in smp.streams[:, 4]])


def draw(smp, c0, dsw):
    """one shared batch with the given per-stream weights (the Sampler's own weights are restored)."""
    keep = smp._dsw
    smp._dsw = dsw
    try:
        return smp.batch(c0["batch"], c0["seq"], c0["chunks"], split="train", aug=c0["aug"])
    finally:
        smp._dsw = keep


# ------------------------------------------------------------------------------------------------ (a) history
def corrupt_history(b, smp, h):
    tok = b["tok"].clone()
    B, S, _ = tok.shape
    dev = tok.device
    x = tok[..., 0:189]
    km = tok[..., 189:198]
    cm = (km[..., smp.goc] > 0) & (x != 0)                                     # valid, not lip-zeroed
    # drift: random walk per column, unit std at the window end, scaled per sample
    on = (torch.rand(B, device=dev) < h["p_drift"]).float()
    sd = h["drift_sd"] * torch.rand(B, device=dev) * on
    rw = torch.cumsum(torch.randn(B, S, 189, device=dev), 1) / float(np.sqrt(S))
    rw = rw - rw[:, :1]                                                         # starts at 0 (window start is exact)
    x = x + rw * sd[:, None, None] * cm
    # freeze: sample-and-hold block per group
    lo, hi = h["freeze_len"]
    fz = torch.rand(B, device=dev) < h["p_freeze"]
    tt = torch.arange(S, device=dev)[None]
    goc = smp.goc
    for g in range(9):
        cols = torch.nonzero(goc == g)[:, 0]
        sel = fz & (torch.rand(B, device=dev) < 0.5)
        if not bool(sel.any()):
            continue
        ln = torch.randint(lo, hi + 1, (B,), device=dev)
        a0 = (torch.rand(B, device=dev) * (S - 1)).long()
        inb = sel[:, None] & (tt >= a0[:, None]) & (tt < (a0 + ln)[:, None])
        idx = torch.where(inb, a0[:, None].expand(B, S), tt.expand(B, S))
        xg = x[:, :, cols]
        x[:, :, cols] = torch.gather(xg, 1, idx[..., None].expand(B, S, len(cols)))
    # group history drop
    gd = torch.rand(B, 9, device=dev) < h["p_gdrop"]
    if bool(gd.any()):
        km = km * (~gd)[:, None, :]
        x = x * (~gd[:, goc])[:, None, :]
    tok[..., 0:189] = x
    tok[..., 189:198] = km
    b["tok"] = tok


# ------------------------------------------------------------------------------------------------ (b) reaction
def _frames(smp, b, nf):
    """self / partner global frame indices (B,nf) of the batch rows, in-bounds mask, has-partner."""
    pick, s0 = b["pick"], b["s0"]
    st = smp.st
    start, length, partner = st[pick, 0], st[pick, 1], st[pick, 2]
    rel = s0[:, None] + torch.arange(nf, device=s0.device)[None]
    inb = rel < length[:, None]
    relc = torch.minimum(rel, length[:, None] - 1)
    has_p = partner >= 0
    pstart = torch.where(has_p, st[partner.clamp(min=0), 0], torch.zeros_like(start))
    return start[:, None] + relc, pstart[:, None] + relc, inb, has_p


def _past_any(x, lo, hi):
    """r[t] = any x[t-hi .. t-lo] (x (B,T) float 0/1), causal window via cumsum."""
    B, T = x.shape
    c = torch.cat([torch.zeros(B, 1, device=x.device), torch.cumsum(x, 1)], 1)       # c[i] = sum x[:i]
    t = torch.arange(T, device=x.device)
    a = (t - hi).clamp(min=0); e = (t - lo + 1).clamp(min=0)
    return (c[:, e] - c[:, a]) > 0


def reaction_weights(b, smp, r):
    if smp.EV is None:
        return
    B, S, _ = b["tok"].shape
    H, L = smp.H, smp.L
    nf = S + H + L + 2
    gi, pgi, inb, has_p = _frames(smp, b, nf)
    sto = smp.store
    ev_s = smp._take(smp.EV, gi.to(sto)).float() * inb[..., None]                    # (B,nf,3) self onsets
    ev_p = smp._take(smp.EV, pgi.to(sto)).float() * (inb & has_p[:, None])[..., None]
    if smp.EVOK is not None:
        ev_s = ev_s * smp.EVOK[b["pick"]].float()[:, None, None]
        part = smp.st[b["pick"], 2].clamp(min=0)
        ev_p = ev_p * smp.EVOK[part].float()[:, None, None]
    null = b["tok"][:, 0, layout(smp.aud_dim)["unull"]] > 0.5                         # CFG-null rows: no partner
    ev_p = ev_p * (~null).float()[:, None, None]
    lo, hi = int(round(r["lag"][0] * FPS)), int(round(r["lag"][1] * FPS))
    pc = [c for c in r["partner_classes"]]
    p_any = ev_p[..., pc].amax(-1)
    if r.get("speech", True):
        vad = (smp._take(smp.A, pgi.to(sto)).float()[..., -1] > 0.5).float() * (inb & has_p[:, None] & ~null[:, None])
        edge = (vad[:, 1:] != vad[:, :-1]).float()
        p_any = torch.maximum(p_any, torch.cat([torch.zeros(B, 1, device=vad.device), edge], 1))
    recent_any = _past_any(p_any, lo, hi).float()                                      # (B,nf)
    rad = r["radius"]
    near = torch.zeros(B, nf, 189, device=b["tok"].device)
    react_on = torch.zeros(B, nf, 3, device=b["tok"].device)
    for c in r["self_classes"]:
        mim = _past_any(ev_p[..., c], lo, hi).float()
        ro = ev_s[..., c] * recent_any                                                 # a reaction onset
        wgt = ro * (r["flow_w"] + r["mimic_w"] * mim)
        react_on[..., c] = ro
        k = 2 * int(rad[c]) + 1
        wn = F.max_pool1d(wgt[:, None], k, stride=1, padding=int(rad[c]))[:, 0]          # (B,nf)
        near = torch.maximum(near, wn[..., None] * smp.ev_gcol[c][None, None])
    # chunk frames: hidx = tau + 1 - C + h (frame index rel s0)
    tau, C = b["tau"], b["C"]
    hidx = tau[..., None] + 1 - C[..., None] + torch.arange(H, device=tau.device)
    bidx = torch.arange(B, device=tau.device)[:, None, None]
    mul = 1.0 + near[bidx, hidx]                                                       # (B,K,H,189)
    em = b.get("emul")
    b["emul"] = mul if em is None else em * mul
    if b.get("ev_ok") is not None and r["event_w"] > 0:
        tol = int(getattr(smp, "event_tol", 1))
        ro = react_on[:, 1:S + 1]
        if tol > 0:
            ro = F.max_pool1d(ro.transpose(1, 2), 2 * tol + 1, stride=1, padding=tol).transpose(1, 2)
        b["ev_ok"] = b["ev_ok"] * (1.0 + r["event_w"] * ro)
    b["_react_frac"] = float(react_on[:, 1:S + 1].amax(-1).mean())


# ------------------------------------------------------------------------------------------------ (c) contrast
class ContingencyHead(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h)[..., 0]


def shuffled_twins(tok, aud_dim, c):
    """(n,S,DIN) twins of the first n rows with a wrong partner + valid row mask (n,)."""
    B, S, _ = tok.shape
    lo = layout(aud_dim)
    n = max(2, int(round(B * c["frac"])))
    dev = tok.device
    us, un = lo["user_all"], lo["unull"]
    tw = tok[:n].clone()
    nb = n // 2
    donor = (torch.arange(nb, device=dev) + torch.randint(1, B, (nb,), device=dev)) % B      # another row's partner
    tw[:nb, :, us] = tok[donor][:, :, us]
    tw[:nb, :, un] = tok[donor][:, :, un]
    sh = torch.randint(int(c["shift"][0]), int(c["shift"][1]) + 1, (n - nb,), device=dev)
    sign = torch.where(torch.rand(n - nb, device=dev) < 0.5, -1, 1)
    t = torch.arange(S, device=dev)[None]
    idx = (t - sign[:, None] * sh[:, None]) % S
    src = tok[nb:n]
    tw[nb:, :, us] = torch.gather(src[:, :, us], 1, idx[..., None].expand(n - nb, S, us.stop - us.start))
    real_ok = tok[:n, 0, un] < 0.5
    don_ok = torch.cat([tok[donor, 0, un] < 0.5, real_ok[nb:]])
    return tw, real_ok & don_ok


def arm_loss(model, b, colw, lw, arm, aux=None, smp=None):
    """flow_loss on the treated batch, + the contingency loss when the arm has one, + (reaction adapter) the reaction-rate
    matching loss when the arm has "rate"."""
    c = arm.get("contrast")
    rate = arm.get("rate")
    if (not c or aux is None) and not rate:
        return flow_loss(model, b, colw, lw)
    tok = b["tok"]
    B, S, _ = tok.shape
    if not c or aux is None:
        h = model.hidden(tok)
        tot, parts = flow_loss(model, b, colw, lw, h=h)
        if rate:
            tot = rate_loss(model, b, smp, h, rate, tot, parts)
        return tot, parts
    tw, ok = shuffled_twins(tok, model.cfg.get("aud_dim", 42), c)
    n = tw.shape[0]
    h_all = model.backbone(torch.cat([tok, tw]))
    h = model.add_react(h_all[:B], tok)
    tot, parts = flow_loss(model, b, colw, lw, h=h)
    if rate:
        tot = rate_loss(model, b, smp, h, rate, tot, parts)
    t0 = int(c["t0"])
    lr_ = aux(h_all[:n, t0:].float()); ls_ = aux(h_all[B:, t0:].float())      # backbone only (no react adapter)
    okf = ok.float()[:, None].expand_as(lr_)
    bce = F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_), reduction="none") + \
        F.binary_cross_entropy_with_logits(ls_, torch.zeros_like(ls_), reduction="none")
    Lc = (bce * okf).sum() / (2 * okf.sum()).clamp(min=1.0)
    acc = (((lr_ > 0).float() + (ls_ < 0).float()) * okf).sum() / (2 * okf.sum()).clamp(min=1.0)
    tot = tot + float(c["w"]) * Lc
    parts["ctr"] = Lc.detach()
    parts["ctr_acc"] = acc.detach()
    return tot, parts


def treat(b, smp, arm):
    """the arm's view of the shared batch (shallow copy; tensors replaced, never written in place). TRAIN batches only
    (train_multi.train_on): validation never sees an injected reaction."""
    if not arm.get("hist") and not arm.get("react") and not arm.get("inject"):
        return b
    b = dict(b)
    if arm.get("inject"):
        inject_reactions(b, smp, arm["inject"])
    if arm.get("react"):
        reaction_weights(b, smp, arm["react"])
    if arm.get("hist"):
        corrupt_history(b, smp, arm["hist"])
    return b


# ------------------------------------------------------------------------------------------------ reaction adapter
def _onsets_nf(b, smp):
    """self / partner label onsets (B,nf,3) at the batch rows' frames (0 outside the stream / CFG-null / no partner),
    the self mouth / head validity (B,nf,9 group mask) is not needed: ev_ok covers the self side."""
    B, S, _ = b["tok"].shape
    nf = S + smp.H + smp.L + 2
    gi, pgi, inb, has_p = _frames(smp, b, nf)
    sto = smp.store
    ev_s = smp._take(smp.EV, gi.to(sto)).float() * inb[..., None]
    ev_p = smp._take(smp.EV, pgi.to(sto)).float() * (inb & has_p[:, None])[..., None]
    if smp.EVOK is not None:
        ev_s = ev_s * smp.EVOK[b["pick"]].float()[:, None, None]
        ev_p = ev_p * smp.EVOK[smp.st[b["pick"], 2].clamp(min=0)].float()[:, None, None]
    null = b["tok"][:, 0, layout(smp.aud_dim)["unull"]] > 0.5
    ev_p = ev_p * (~null).float()[:, None, None]
    return ev_s, ev_p, inb, nf


def _laugh_nf(b, smp, nf):
    """partner laugh onsets (B,nf) from the causal trace of the tokens (token t holds the user's frame t+1)."""
    from unified.react import trace_torch, norm_info
    ni = getattr(smp, "_react_ni", None)
    if ni is None:
        ni = smp._react_ni = norm_info(smp.mean.cpu().numpy(), smp.std.cpu().numpy(), smp.amean.cpu().numpy(),
                                       smp.astd.cpu().numpy())
    tr = trace_torch(b["tok"], layout(smp.aud_dim), ni)
    B, S = tr.shape[:2]
    out = torch.zeros(B, nf, device=tr.device)
    out[:, 1:S + 1] = tr[..., 8]
    return out


def rate_loss(model, b, smp, h, r, tot, parts):
    """reaction-rate matching: for each (partner trigger -> self class) pair, the model's mean onset probability in
    the 0.3-2 s windows after the trigger minus its mean over all valid tokens must equal the same excess on the
    (dilated) labels (EMA over batches; normalised by the EMA base rate). Token t predicts frame t+1."""
    if model.events is None or b.get("ev_tgt") is None or smp is None or smp.EV is None:
        return tot
    B, S, _ = b["tok"].shape
    ev_s, ev_p, inb, nf = _onsets_nf(b, smp)
    trig = {"smile": ev_p[..., 2], "nod": ev_p[..., 0]}
    if any(q == "laugh" for q, _ in r["pairs"]):
        trig["laugh"] = _laugh_nf(b, smp, nf)
    lo, hi = int(round(r["lag"][0] * FPS)), int(round(r["lag"][1] * FPS))
    p = torch.sigmoid(model.events(h).float())
    y = b["ev_tgt"].float()
    ok = (b["ev_ok"] > 0).float()
    st = r.setdefault("_ema", {})
    L = 0.0
    for q, c in r["pairs"]:
        W = _past_any(trig[q], lo, hi).float()[:, 1:S + 1] * ok[..., c]
        nw, nb = W.sum(), ok[..., c].sum()
        if float(nw) < r["min_frames"]:
            continue
        mw = (p[..., c] * W).sum() / nw
        mb = (p[..., c] * ok[..., c]).sum() / nb.clamp(min=1)
        yw = float((y[..., c] * W).sum() / nw); yb = float((y[..., c] * ok[..., c]).sum() / nb.clamp(min=1))
        k = f"{q}{c}"
        e0, b0 = st.get(k, (yw - yb, yb))
        a = r["ema"]
        st[k] = (a * e0 + (1 - a) * (yw - yb), a * b0 + (1 - a) * yb)
        e, base = st[k]
        L = L + ((mw - mb) - e) ** 2 / max(base, 1e-3) ** 2
        parts[f"rx_{q}{c}"] = (mw - mb).detach() / max(base, 1e-3)          # model lift (excess / base)
        parts[f"ry_{q}{c}"] = torch.tensor(e / max(base, 1e-3))             # data lift (EMA)
    if isinstance(L, float):
        return tot
    parts["rate"] = L.detach()
    return tot + float(r["w"]) * L


def inject_reactions(b, smp, c):
    """R3 "cartoon" teacher: on a fraction of partner smile (+ laugh) / nod onsets the person did NOT answer (no self
    onset of that class 0.3-2 s later), add the canonical event curve (smp.event_info curves: the train set's
    event-triggered average, the runtime's own event curve) to the SELF target 0.4-0.9 s after the trigger: history
    tokens, chunk targets, known prefix, boundary frame, and the event labels (onset + dilation)."""
    if smp.EV is None or smp.event_info is None:
        return
    tok = b["tok"]
    B, S, _ = tok.shape
    dev = tok.device
    ev_s, ev_p, inb, nf = _onsets_nf(b, smp)
    lo, hi = int(round(c["react_lag"][0] * FPS)), int(round(c["react_lag"][1] * FPS))
    curves = torch.as_tensor(smp.event_info["curves"], device=dev).float() / smp.std          # (3,Lmax,189) norm
    lens = list(smp.event_info.get("lens") or [curves.shape[1]] * 3)
    trig = {2: ev_p[..., 2], 0: ev_p[..., 0]}
    if c.get("laugh"):
        trig[2] = torch.maximum(trig[2], _laugh_nf(b, smp, nf))
    frac = {2: float(c["f_smile"]), 0: float(c["f_nod"])}
    Dl = torch.zeros(B, nf, 189, device=dev)
    lab = torch.zeros(B, nf, 3, device=dev)
    # a reaction already there: any self onset of the class in [u+lo, u+hi] = any onset in the window after u
    cs = torch.cat([torch.zeros(B, 1, 3, device=dev), torch.cumsum(ev_s, 1)], 1)
    t = torch.arange(nf, device=dev)
    a_ = (t + lo).clamp(max=nf); e_ = (t + hi + 1).clamp(max=nf)
    reacted = (cs[:, e_] - cs[:, a_]) > 0                                                     # (B,nf,3)
    n_inj = 0
    for cls, tr in trig.items():
        if frac[cls] <= 0:
            continue
        cand = (tr > 0) & ~reacted[..., cls] & inb & (torch.rand(B, nf, device=dev) < frac[cls])
        idx = torch.nonzero(cand)
        if len(idx) == 0:
            continue
        lag = torch.randint(int(c["lag"][0]), int(c["lag"][1]) + 1, (len(idx),), device=dev)
        amp = c["amp"][0] + (c["amp"][1] - c["amp"][0]) * torch.rand(len(idx), device=dev)
        Lc = int(lens[cls])
        for (bi, u), lg, am in zip(idx.tolist(), lag.tolist(), amp.tolist()):
            v = u + lg
            if v >= nf - 2:
                continue
            k = min(Lc, nf - v)
            Dl[bi, v:v + k] += am * curves[cls, :k]
            lab[bi, v, cls] = 1.0
            n_inj += 1
    b["_inject_n"] = n_inj
    b["_inject_lab"] = lab                                                 # diagnostics (tools/react/tune_inject.py)
    if n_inj == 0:
        return
    lo_b = ((0.0 - smp.mean) / smp.std); hi_b = ((1.0 - smp.mean) / smp.std)
    bs = torch.zeros(189, dtype=torch.bool, device=dev); bs[135:187] = True

    def clip_bs(x):
        return torch.where(bs, torch.minimum(torch.maximum(x, lo_b), hi_b), x)

    x = tok[..., 0:189]
    live = (x != 0).float()                                               # valid, not lip-zeroed, not dropped
    tok = tok.clone()
    tok[..., 0:189] = torch.where(live > 0, clip_bs(x + Dl[:, :S]), x)
    b["tok"] = tok
    tau, C = b["tau"], b["C"]
    H = b["x1"].shape[2]
    hidx = tau[..., None] + 1 - C[..., None] + torch.arange(H, device=dev)
    bidx = torch.arange(B, device=dev)[:, None, None]
    dch = Dl[bidx, hidx]                                                   # (B,K,H,189)
    kcol = b["kmask"][..., smp.goc]
    vis = ((b["lmask"] > 0) | (kcol > 0)).float()
    b["x1"] = clip_bs(b["x1"] + dch * vis)
    b["kvals"] = b["kvals"] + dch * kcol
    pi = (tau + 1 - C - 1).clamp(min=0)
    b["prev"] = b["prev"] + Dl[bidx[..., 0], pi] * b["prev_ok"]
    if b.get("ev_raw") is not None:
        on = lab[:, 1:S + 1]                                               # token t <-> frame t+1
        b["ev_raw"] = torch.maximum(b["ev_raw"], on)
        tol = int(getattr(smp, "event_tol", 1))
        ond = F.max_pool1d(on.transpose(1, 2), 2 * tol + 1, stride=1, padding=tol).transpose(1, 2) if tol > 0 else on
        b["ev_tgt"] = torch.maximum(b["ev_tgt"], ond)
