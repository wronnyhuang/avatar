"""The unified non-verbal motion model (v1): causal sliding-window transformer backbone + rectified-flow
inpainting head. Shared by training (tools/train/train.py), export (unified/export.py) and the runtime.

Token tau (one per frame, all features normalized with norm.npz):
    [ self motion[tau] * colmask | self mask9[tau] | user motion[tau+1] * colmask | user mask9[tau+1]
      | user audio[tau+1] | self audio[tau+1+L] | user_null | style[tau] (3 values + 3 null flags) ]   (DIN = 487)
  i.e. when planning frame t the last token is tau = t-1: it holds the last committed avatar frame, the user's
  newest frame t and the avatar's speech feature for t+L (L = lookahead). user_null = 1 (with the user part zeroed)
  is the classifier-free-guidance null stream. STYLE = {energy, mirroring, expressiveness} in [0,1]; a null
  component (flag 1, value 0) = "infer it" (learned null: the flag's input weights); also fed to the head's adaLN.
Backbone: causal, sliding window W tokens (band mask), RoPE on absolute frame index (fixed-size KV cache at runtime).
Head: H-frame chunk in patches of P frames; tokens = [x_s (noisy) | known values | known mask9] -> velocity;
  adaLN on (flow time + last backbone token), cross-attention to the last M backbone outputs.
  Chunk frame 0 = t - C (the C committed frames are known = the prefix).
phase 2: the audio width A is per model (cfg["aud_dim"]: 42 = mel "audio42"; a speech encoder = its feature dims + 2
  (loud_db, vad of the same person)) -> layout(A); DIN = 403 + 2A. EVENT HEAD (cfg["n_events"] > 0): per-token MLP on
  the backbone output -> onset logits of {nod, blink, smile} at frame tau+1 ("fire now"). cfg["head_kind"] = "flow"
  (rectified flow, Euler sampling) | "regression" (the same head trained at s = 0 from x0 = 0: one call = the chunk).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MOT = 189
D_MASK = 9
D_AUD = 42
D_STY = 6
STYLE_NULL = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
EVENT_NAMES = ("nod", "blink", "smile")
_FIXED = D_MOT + D_MASK + D_MOT + D_MASK + 1 + D_STY                         # 403: everything but the 2 audio blocks


def layout(aud_dim=D_AUD):
    """token layout for audio width A: dict of slices / indices + din."""
    A = int(aud_dim)
    u0 = D_MOT + D_MASK
    a0 = u0 + D_MOT + D_MASK
    un = a0 + 2 * A
    return dict(A=A, self=slice(0, D_MOT), self_mask=slice(D_MOT, u0), user=slice(u0, u0 + D_MOT),
                user_mask=slice(u0 + D_MOT, a0), user_audio=slice(a0, a0 + A), self_audio=slice(a0 + A, un),
                user_all=slice(u0, a0 + A), unull=un, style=slice(un + 1, un + 1 + D_STY), din=un + 1 + D_STY)


def aud_dim_of(din):
    A, r = divmod(int(din) - _FIXED, 2)
    assert r == 0 and A > 0, din
    return A


_L42 = layout(D_AUD)
UNULL_I = _L42["unull"]                                                          # user-null flag (CFG), A = 42
STYLE_SL = _L42["style"]
DIN = _L42["din"]                                                                # 487
USER_SL = _L42["user_all"]                                                       # zeroed in the null stream


def default_cfg():
    return dict(d=256, layers=5, heads=4, ffn=4, window=90, hd=128, hlayers=3, hheads=4, hffn=3, horizon=15,
                patch=3, mem=30, lookahead=6, dropout=0.0, aud_dim=D_AUD, n_events=0, head_kind="flow")


def null_tokens(tok):
    """the user-null (CFG) copy of tokens of any audio width (inferred from the last dim)."""
    lo = layout(aud_dim_of(tok.shape[-1]))
    t = tok.clone()
    t[..., lo["user_all"]] = 0
    t[..., lo["unull"]] = 1
    return t


# ------------------------------------------------------------------------------------------------ rope
def rope_cos_sin(pos, dh, base=10000.0):
    """pos: (T,) absolute frame indices -> cos, sin (T, dh/2), angles computed in float64 (exact at any pos)."""
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh))
    ang = torch.remainder(pos.to(torch.float64)[:, None] * inv[None, :], 2 * math.pi)
    return torch.cos(ang).float(), torch.sin(ang).float()


def apply_rope(x, cos, sin):
    """rotate-half RoPE. x (..., T, dh); cos/sin (T, dh/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


# ------------------------------------------------------------------------------------------------ backbone
class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout=0.0, qknorm=False):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        # QK-norm (final3): bounded attention logits - final2 diverged at 66k steps (grad norm 0.4 -> 580k)
        self.qn = nn.LayerNorm(self.dh) if qknorm else None
        self.kn = nn.LayerNorm(self.dh) if qknorm else None
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(approximate="tanh"), nn.Linear(ffn * d, d))
        self.drop = nn.Dropout(dropout)

    def _qkv(self, x, cos, sin):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x, cos, sin, mask):
        B, T, d = x.shape
        q, k, v = self._qkv(x, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.ff(self.n2(x)))


class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.inp = nn.Linear(layout(c.get("aud_dim", D_AUD))["din"], c["d"])
        self.blocks = nn.ModuleList([Block(c["d"], c["heads"], c["ffn"], c.get("dropout", 0.0), c.get("qknorm", False))
                                     for _ in range(c["layers"])])
        self.nf = nn.LayerNorm(c["d"])

    def band_mask(self, T, device):
        i = torch.arange(T, device=device)
        rel = i[:, None] - i[None, :]
        return (rel >= 0) & (rel < self.c["window"])

    def forward(self, tok, pos0=0):
        """tok (B,T,DIN) -> (B,T,d). Token i attends to tokens i-W+1..i."""
        B, T, _ = tok.shape
        x = self.inp(tok)
        cos, sin = rope_cos_sin(torch.arange(pos0, pos0 + T), self.c["d"] // self.c["heads"])
        cos, sin = cos.to(x.device), sin.to(x.device)
        mask = self.band_mask(T, tok.device)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.nf(x)

    def step_fixed(self, tok, cos, sin, kc, vc, kvalid):
        """Incremental step with a FIXED-size cache (ONNX/runtime path).
        tok (B,n,DIN) new tokens; cos/sin (n,dh/2) of their absolute positions; kc/vc (Lyr,B,h,W-1,dh) rotated
        cached keys/values, oldest first, newest in the last slot; kvalid (W-1,) 1 = slot holds a real key.
        Returns out (B,n,d), new kc/vc (the newest W-1 keys). Exactly equals forward() over the stream."""
        W = self.c["window"]
        n = tok.shape[1]
        x = self.inp(tok)
        # relative positions: cached slot c -> c-(W-1), new key j -> j ; query i may see key at kp iff 0<=i-kp<W
        kp = torch.cat([torch.arange(W - 1) - (W - 1), torch.arange(n)]).to(tok.device)
        qp = torch.arange(n).to(tok.device)
        rel = qp[:, None] - kp[None, :]
        ok = (rel >= 0) & (rel < W) & (torch.cat([kvalid, torch.ones(n, device=tok.device)]) > 0.5)[None, :]
        bias = torch.where(ok, 0.0, -1e9)
        nk, nv = [], []
        for li, b in enumerate(self.blocks):
            q, k, v = b._qkv(x, cos, sin)                                   # (B,h,n,dh)
            K = torch.cat([kc[li], k], 2); V = torch.cat([vc[li], v], 2)     # (B,h,W-1+n,dh)
            att = torch.softmax((q @ K.transpose(-1, -2)) * (1.0 / math.sqrt(b.dh)) + bias, -1)
            a = (att @ V).transpose(1, 2).reshape(x.shape)
            x = x + b.o(a)
            x = x + b.ff(b.n2(x))
            nk.append(K[:, :, n:]); nv.append(V[:, :, n:])
        return self.nf(x), torch.stack(nk), torch.stack(nv)


# ------------------------------------------------------------------------------------------------ flow head
def time_embed(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class HeadBlock(nn.Module):
    def __init__(self, d, dm, heads, ffn, qknorm=False):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.o1 = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.q2 = nn.Linear(d, d)
        self.kv2 = nn.Linear(dm, 2 * d)
        self.qn = nn.LayerNorm(self.dh) if qknorm else None          # QK-norm for self- and cross-attention
        self.kn = nn.LayerNorm(self.dh) if qknorm else None
        self.qn2 = nn.LayerNorm(self.dh) if qknorm else None
        self.kn2 = nn.LayerNorm(self.dh) if qknorm else None
        self.o2 = nn.Linear(d, d)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(approximate="tanh"), nn.Linear(ffn * d, d))
        self.mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def mem_kv(self, mem):
        B, M, _ = mem.shape
        k, v = self.kv2(mem).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.kn2 is not None:
            k = self.kn2(k)
        return k, v

    def mods(self, c):
        return self.mod(F.silu(c)).view(c.shape[0], 9, -1)                   # (B,9,d)

    def forward(self, x, md, mk, mv, mmask):
        """x (B,T,d); md (B,9,d) modulation; mk/mv (B,h,M,dh); mmask (B,1,1,M) bool/additive or None."""
        B, T, d = x.shape
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = md.unsqueeze(2).unbind(1)       # each (B,1,d)
        y = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(y).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + g1 * self.o1(a.transpose(1, 2).reshape(B, T, d))
        y = self.n2(x) * (1 + s2) + b2
        q = self.q2(y).view(B, T, self.h, self.dh).transpose(1, 2)
        if self.qn2 is not None:
            q = self.qn2(q)
        a = F.scaled_dot_product_attention(q, mk, mv, attn_mask=mmask)
        x = x + g2 * self.o2(a.transpose(1, 2).reshape(B, T, d))
        y = self.n3(x) * (1 + s3) + b3
        return x + g3 * self.ff(y)


class FlowHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        d = c["hd"]
        P = self.P = c.get("patch", 1)
        assert c["horizon"] % P == 0
        self.inp_x = nn.Linear(P * D_MOT, d, bias=False)
        self.inp_k = nn.Linear(P * (D_MOT + D_MASK), d)
        self.pos = nn.Parameter(torch.zeros(c["horizon"] // P, d))
        nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.cproj = nn.Linear(c["d"], d)
        self.sproj = nn.Linear(D_STY, d)
        self.blocks = nn.ModuleList([HeadBlock(d, c["d"], c["hheads"], c["hffn"], c.get("qknorm", False)) for _ in range(c["hlayers"])])
        self.nf = nn.LayerNorm(d, elementwise_affine=False)
        self.fmod = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.fmod.weight); nn.init.zeros_(self.fmod.bias)
        self.out = nn.Linear(d, P * D_MOT)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    # ---- pieces (the runtime precomputes everything that does not change across Euler steps)
    def prepare(self, mem, mmask=None, style=None):
        """mem (B,M,d_backbone), style (B,6) -> per-block memory k/v + last-token conditioning; once per plan."""
        if style is None:
            style = torch.tensor(STYLE_NULL, device=mem.device).expand(mem.shape[0], D_STY)
        return [b.mem_kv(mem) for b in self.blocks], self.cproj(mem[:, -1]) + self.sproj(style), mmask

    def embed_known(self, known, kmask):
        N, H, _ = known.shape
        return self.inp_k(torch.cat([known, kmask], -1).reshape(N, H // self.P, -1)) + self.pos[: H // self.P]

    def all_mods(self, s, cl):
        """s (B,) flow time, cl (B,hd) -> block mods (Lh,B,9,hd), final mods (B,2,hd)."""
        c = self.temb(time_embed(s)) + cl
        return torch.stack([b.mods(c) for b in self.blocks]), self.fmod(F.silu(c)).view(c.shape[0], 2, -1)

    def core(self, xs, kemb, bmods, fm, kv, mmask):
        N, H, _ = xs.shape
        x = self.inp_x(xs.reshape(N, H // self.P, -1)) + kemb
        for i, (b, (mk, mv)) in enumerate(zip(self.blocks, kv)):
            x = b(x, bmods[i], mk, mv, mmask)
        sh, sc = fm[:, 0:1], fm[:, 1:2]
        return self.out(self.nf(x) * (1 + sc) + sh).reshape(N, H, D_MOT)

    def forward(self, xs, s, known, kmask, prep):
        """xs (B,H,189) noisy chunk; s (B,) flow time (0 = noise, 1 = data); known (B,H,189) known values (0 where
        unknown); kmask (B,H,9) known-group mask -> velocity (B,H,189)."""
        kv, cl, mmask = prep
        bm, fm = self.all_mods(s, cl)
        return self.core(xs, self.embed_known(known, kmask), bm, fm, kv, mmask)


class EventHead(nn.Module):
    """backbone output (..., d) -> onset logits (..., n_events) for the frame after the token ("fire now")."""

    def __init__(self, d, n, prior=0.01):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, n))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, math.log(prior / (1 - prior)))

    def forward(self, h):
        return self.net(h)


class UnifiedModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(default_cfg(), **(cfg or {}))
        self.layout = layout(self.cfg["aud_dim"])
        self.backbone = Backbone(self.cfg)
        self.head = FlowHead(self.cfg)
        self.events = EventHead(self.cfg["d"], int(self.cfg["n_events"])) if int(self.cfg["n_events"]) > 0 else None
        # reaction adapter: causal partner-event trace adapter (unified/react.py); None = the plain model, bit for bit
        rt = self.cfg.get("react_trace")
        self.react = None
        if rt:
            from unified.react import make_adapter
            self.react = make_adapter(self.cfg["d"], int(rt.get("hidden", 64)))

    def add_react(self, h, tok):
        """h (B,S,d) backbone outputs of tok -> h + adapter(partner trace(tok)) (h itself without the adapter)."""
        if self.react is None:
            return h
        from unified.react import trace_torch
        tr = trace_torch(tok, self.layout, self.cfg["react_trace"]).to(h.dtype)
        return h + self.react(tr).to(h.dtype)

    def hidden(self, tok):
        """the heads' input: backbone(tok) (+ the partner-trace adapter for react_trace models)."""
        return self.add_react(self.backbone(tok), tok)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------------------------------------ export wrappers
class EvStep(nn.Module):
    """h (N,d) backbone outputs -> onset probabilities (N,n_events)."""

    def __init__(self, ev):
        super().__init__()
        self.ev = ev

    def forward(self, h):
        return torch.sigmoid(self.ev(h))


class BBStep(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, tok, cos, sin, kc, vc, kvalid):
        return self.bb.step_fixed(tok, cos, sin, kc, vc, kvalid)


class HeadPrep(nn.Module):
    """Once per plan: mem (2,M,d), s_all (S,) step times, kvals (1,H,189), kmask (1,H,9), style (1,6) ->
    memk/memv (Lh,2,h,M,dh), bmods (S,Lh,2,9,hd), fmods (S,2,2,hd), kemb (1,T,hd)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, mem, s_all, kvals, kmask, style):
        kv, cl, _ = self.head.prepare(mem, None, style.expand(mem.shape[0], D_STY))
        S = s_all.shape[0]
        s2 = s_all[:, None].expand(S, 2).reshape(-1)
        cl2 = cl[None].expand(S, 2, cl.shape[-1]).reshape(2 * S, -1)
        bm, fm = self.head.all_mods(s2, cl2)                                  # (Lh,2S,9,hd), (2S,2,hd)
        Lh = bm.shape[0]
        bm = bm.view(Lh, S, 2, 9, -1).permute(1, 0, 2, 3, 4)
        fm = fm.view(S, 2, 2, -1)
        return (torch.stack([k for k, _ in kv]), torch.stack([v for _, v in kv]), bm, fm,
                self.head.embed_known(kvals, kmask))


class HeadStep(nn.Module):
    """One guided Euler step: x (1,H,189), kemb (1,T,hd), memk/memv, bmods_i (Lh,2,9,hd), fmods_i (2,2,hd),
    mbias (2,1,1,M), g (1,) -> guided velocity (1,H,189) = v_null + g (v_cond - v_null)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, kemb, memk, memv, bmods, fmods, mbias, g):
        x2 = torch.cat([x, x]); ke2 = torch.cat([kemb, kemb])
        kv = [(memk[i], memv[i]) for i in range(memk.shape[0])]
        v = self.head.core(x2, ke2, bmods, fmods, kv, mbias)
        return v[1:] + g * (v[:1] - v[1:])
