"""Rectified-flow training loss and the guided inpainting sampler (shared by train, eval and runtime)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_mem(h, mem_idx, mem_ok):
    """h (B,S,d) backbone outputs; mem_idx/mem_ok (B,K,M) -> mem (B*K,M,d), attn mask (B*K,1,1,M) bool."""
    B, K, M = mem_idx.shape
    bi = torch.arange(B, device=h.device)[:, None, None]
    mem = h[bi, mem_idx]                                    # (B,K,M,d)
    return mem.reshape(B * K, M, -1), mem_ok.reshape(B * K, 1, 1, M)


def event_loss(logits, y, ok, gamma=2.0, alpha=0.75):
    """focal BCE per class, normalised by that class's positive count (rare onsets). logits/y/ok (B,S,C)."""
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    at = alpha * y + (1 - alpha) * (1 - y)
    fl = at * (1 - pt) ** gamma * ce * ok
    npos = (y * ok).sum((0, 1)).clamp(min=1.0)
    return (fl.sum((0, 1)) / npos).mean()


def flow_loss(model, b, colw, lw, h=None):
    """b = Sampler.batch(...). Returns (total, parts dict). phase 2: head_kind "regression" = the head at s = 0 from
    x0 = 0 (its output IS the chunk; vel/jerk unweighted by s); b["emul"] = rare-event weighting of the per-entry flow
    error; lw["a2f"] = Audio2Face teacher weight (x-prediction toward b["a2f1"] on b["a2m"]); lw["event"] = event head
    focal-BCE weight (b["ev_tgt"], b["ev_ok"])."""
    if h is None:                                         # a caller may pass the backbone output it already has
        h = model.hidden(b["tok"]) if hasattr(model, "hidden") else model.backbone(b["tok"])
    mem, mmask = gather_mem(h, b["mem_idx"], b["mem_ok"])
    B, K, H, D = b["x1"].shape
    x1 = b["x1"].reshape(B * K, H, D)
    lm = b["lmask"].reshape(B * K, H, D)
    kv = b["kvals"].reshape(B * K, H, D)
    km = b["kmask"].reshape(B * K, H, -1)
    w = b["w"].reshape(B * K)
    reg = model.cfg.get("head_kind", "flow") == "regression"
    s = torch.rand(B * K, device=x1.device)
    if lw.get("logit_t", 0) > 0:
        ln = torch.sigmoid(torch.randn(B * K, device=x1.device))
        s = torch.where(torch.rand(B * K, device=x1.device) < lw["logit_t"], ln, s)
    x0 = torch.randn_like(x1)
    if reg:
        s = torch.zeros_like(s)
        x0 = torch.zeros_like(x1)
    sb = s[:, None, None]
    xs = (1 - sb) * x0 + sb * x1
    v = x1 - x0
    sty = b.get("style_c")
    prep = model.head.prepare(mem, mmask, None if sty is None else sty.reshape(B * K, -1))
    vh = model.head(xs, s, kv, km, prep)
    cw = colw[None, None]
    den = (lm * cw).sum((1, 2)).clamp(min=1.0)
    em = b.get("emul")
    em = 1.0 if em is None else em.reshape(B * K, H, D)
    lf = (((vh - v) ** 2) * lm * cw * em).sum((1, 2)) / den
    # clean estimate, composited with the known values, prefixed with the boundary frame
    kcol = km[..., model_goc(model, x1.device)]
    x1h = xs + (1 - sb) * vh
    comp = kcol * kv + (1 - kcol) * x1h                   # known entries = the given values (a held pose may differ)
    prev = b["prev"].reshape(B * K, 1, D); pok = b["prev_ok"].reshape(B * K, 1, D)
    seq_h = torch.cat([prev, comp], 1); seq_t = torch.cat([prev, kcol * kv + (1 - kcol) * x1], 1)
    valid = torch.cat([pok, (lm + kcol).clamp(max=1)], 1)
    dh = seq_h[:, 1:] - seq_h[:, :-1]; dt = seq_t[:, 1:] - seq_t[:, :-1]
    vv = valid[:, 1:] * valid[:, :-1]
    lv = (((dh - dt) ** 2) * vv * cw).sum((1, 2)) / (vv * cw).sum((1, 2)).clamp(min=1.0)
    jh = dh[:, 2:] - 2 * dh[:, 1:-1] + dh[:, :-2]
    jv = vv[:, 2:] * vv[:, 1:-1] * vv[:, :-2]
    lj = ((jh ** 2) * jv * cw).sum((1, 2)) / (jv * cw).sum((1, 2)).clamp(min=1.0)
    wsum = w.sum().clamp(min=1e-6)
    sw = torch.ones_like(s) if reg else s
    Lf = (lf * w).sum() / wsum
    Lv = (lv * w * sw).sum() / wsum
    Lj = (lj * w * sw).sum() / wsum
    tot = Lf + lw.get("vel", 0.5) * Lv + lw.get("jerk", 0.1) * Lj
    parts = {"flow": Lf.detach(), "vel": Lv.detach(), "jerk": Lj.detach()}
    if lw.get("a2f", 0) > 0 and b.get("a2f1") is not None:
        a1 = b["a2f1"].reshape(B * K, H, -1)
        am = b["a2m"].reshape(B * K, H, -1)
        cwb = cw[..., 135:187]
        la = (((x1h[..., 135:187] - a1) ** 2) * am * cwb).sum((1, 2)) / (am * cwb).sum((1, 2)).clamp(min=1.0)
        has = (am.sum((1, 2)) > 0).float()
        La = (la * w).sum() / (w * has).sum().clamp(min=1e-6)
        tot = tot + lw["a2f"] * La
        parts["a2f"] = La.detach()
    if model.events is not None and b.get("ev_tgt") is not None and lw.get("event", 0.3) > 0:
        Le = event_loss(model.events(h).float(), b["ev_tgt"], b["ev_ok"], lw.get("event_gamma", 2.0),
                        lw.get("event_alpha", 0.75))
        tot = tot + lw.get("event", 0.3) * Le
        parts["event"] = Le.detach()
    return tot, parts


_GOC = {}


def model_goc(model, device):
    from canon1 import spec
    k = str(device)
    if k not in _GOC:
        _GOC[k] = torch.as_tensor(spec.GROUP_OF_COL, device=device)
    return _GOC[k]


@torch.no_grad()
def sample_chunk(head, prep, kvals, kmask, steps=6, guidance=1.0, temperature=1.0, gen=None, prep_u=None,
                 goc=None):
    """Euler sampling of one chunk per batch row. prep = head.prepare(cond memory); prep_u = the unconditional
    (user-null) memory for CFG (None = no guidance). kvals (N,H,189) known values (normalized), kmask (N,H,9).
    Known entries are pinned along the straight path (x_s = (1-s) z + s * known) and returned exactly."""
    N, H, D = kvals.shape
    dev = kvals.device
    z = torch.randn(N, H, D, generator=gen, device=dev) * temperature
    kcol = kmask[..., goc]
    x = z.clone()
    use_cfg = prep_u is not None and guidance != 1.0
    if use_cfg:
        both = _cat_prep(prep, prep_u)
    for i in range(steps):
        s0 = i / steps
        x = kcol * ((1 - s0) * z + s0 * kvals) + (1 - kcol) * x
        sv = torch.full((N,), s0, device=dev)
        if use_cfg:
            vb = head(torch.cat([x, x]), torch.cat([sv, sv]), torch.cat([kvals, kvals]), torch.cat([kmask, kmask]),
                      both)
            vc, vu = vb[:N], vb[N:]
            v = vu + guidance * (vc - vu)
        else:
            v = head(x, sv, kvals, kmask, prep)
        x = x + v / steps
    return kcol * kvals + (1 - kcol) * x


def _cat_prep(a, b):
    kv = [(torch.cat([ka, kb]), torch.cat([va, vb])) for (ka, va), (kb, vb) in zip(a[0], b[0])]
    cl = torch.cat([a[1], b[1]])
    mm = None if a[2] is None else torch.cat([a[2], b[2]])
    return kv, cl, mm
