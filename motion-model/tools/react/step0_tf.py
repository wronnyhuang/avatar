"""reaction adapter step 0: where does the avatar model lose the "smile back / nod back" reaction? Teacher-forced on VAL.

    python -I tools/react/step0_tf.py --bundle B1[,B2..] --name N1[,N2..] --batches 80 --out runs/p2react/step0

One Sampler (the d50 val people = fv4_inc25 pack, no CANDOR), the same fixed-seed val batches for every bundle.
(a) EVENT HEADS: for each partner onset at frame u (smile / nod from smp.EV at the partner's frames; laugh = a derived
    partner onset: smile high & jaw open & partner speaking), window W = self frames u+9 .. u+60. Mean predicted self
    onset probability in W vs matched random windows of the same length in the same row (and all valid frames of
    rows with a partner); the same from the self LABELS (per-frame rate and P(any onset in W)). Model rates are also
    given calibrated (sum p rescaled to the label count over all valid frames).
    + mean |p(real partner) - p(user-nulled tokens)| overall and in partner-onset windows.
(b) FLOW: self smile level (raw units, mean of mouthSmileLeft/Right) on the chunks' predicted frames 0.3-2 s after a
    partner smile onset vs all frames / frames with no partner smile in the last 4 s: TRUE (x1), sampled with CFG
    (guidance 2.0, the runtime's), with guidance 1 on the real memory, and with the user nulled (guidance 1 on the
    null memory). Temperature 0.3, 6 steps.
Writes <out>/<name>.json per bundle (+ labels.json) and <out>/STEP0_TF.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from canon1 import spec  # noqa: E402
from unified.data import AUG_NONE, EVENT_NAMES, Sampler  # noqa: E402
from unified.flow import gather_mem, sample_chunk  # noqa: E402
from unified.model import UnifiedModel, null_tokens  # noqa: E402
from train import resolve_pack, sampler_kwargs  # noqa: E402

FPS = 30
LO, HI = 9, 60                    # window u+9 .. u+60 (0.3 - 2 s)
NOD, BLINK, SMILE = 0, 1, 2
PAIRS = [("smile", "smile"), ("nod", "nod"), ("smile", "nod"), ("nod", "smile"), ("laugh", "smile"), ("laugh", "nod")]
SMC = [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]
JAW = 135 + spec.BS["jawOpen"]
N_RAND = 16                        # random baseline windows per onset


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def past_any(x, lo, hi):
    """r[t] = any x[t-hi .. t-lo] (x (B,T) float 0/1)."""
    B, T_ = x.shape
    c = torch.cat([torch.zeros(B, 1), torch.cumsum(x, 1)], 1)
    t = torch.arange(T_)
    a = (t - hi).clamp(min=0); e = (t - lo + 1).clamp(min=0)
    return (c[:, e] - c[:, a]) > 0


def partner_frames(smp, b, nf):
    pick, s0 = b["pick"], b["s0"]
    st = smp.st
    start, length, partner = st[pick, 0], st[pick, 1], st[pick, 2]
    rel = s0[:, None] + torch.arange(nf)[None]
    inb = rel < length[:, None]
    relc = torch.minimum(rel, length[:, None] - 1)
    has_p = partner >= 0
    pstart = torch.where(has_p, st[partner.clamp(min=0), 0], torch.zeros_like(start))
    return pstart[:, None] + relc, inb & has_p[:, None], partner.clamp(min=0), has_p


def laugh_onsets(m_user, k_user, vad, ok, thr, min_len=6, refr=60):
    """derived partner laugh onsets (B,nf): first frame of a >= min_len run of (smile > thr_s & jaw > thr_j & vad)."""
    sm = m_user[..., SMC].mean(-1); jw = m_user[..., JAW]
    face = (k_user[..., spec.GROUP_NAMES.index("mouth")] > 0)
    on = (sm > thr[0]) & (jw > thr[1]) & (vad > 0.5) & face & ok
    B, nf = on.shape
    out = torch.zeros(B, nf)
    onn = on.numpy()
    for i in range(B):
        run, last = 0, -10 ** 6
        for t in range(nf):
            run = run + 1 if onn[i, t] else 0
            if run == min_len and t - min_len + 1 - last >= refr:
                out[i, t - min_len + 1] = 1
                last = t - min_len + 1
    return out


class Acc:
    """accumulators for one source (labels or one model)."""

    def __init__(self):
        self.d = {}

    def add(self, k, v, n=1.0):
        s, c = self.d.get(k, (0.0, 0.0))
        self.d[k] = (s + float(v), c + float(n))

    def mean(self, k):
        s, c = self.d.get(k, (0.0, 0.0))
        return s / c if c > 0 else float("nan")

    def cnt(self, k):
        return self.d.get(k, (0.0, 0.0))[1]


def window_stats(acc, key, val, ok, ev_p, g, use_any):
    """val (B,S+2) per-frame quantity on rel frames (index f), ok (B,S+2) valid; ev_p (B,nf) partner onsets.
    For every partner onset u with the full window [u+LO, u+HI] inside [1, S]: window mean (and any) vs N_RAND random
    windows of the same length in the same row."""
    B = val.shape[0]
    S = val.shape[1] - 2
    L = HI - LO + 1
    vv = (val * ok).numpy(); oo = ok.numpy().astype(np.float64)
    cv = np.concatenate([np.zeros((B, 1)), np.cumsum(vv, 1)], 1)
    co = np.concatenate([np.zeros((B, 1)), np.cumsum(oo, 1)], 1)
    if use_any:
        pos = (vv > 0).astype(np.float64)
        cp = np.concatenate([np.zeros((B, 1)), np.cumsum(pos, 1)], 1)
    rows, us = np.nonzero(ev_p.numpy() > 0)
    for i, u in zip(rows, us):
        a, e = u + LO, u + HI + 1
        if a < 1 or e > S + 1:
            continue
        no = co[i, e] - co[i, a]
        if no < 0.6 * L:
            continue
        acc.add(key + ":w", (cv[i, e] - cv[i, a]) / no)
        if use_any:
            acc.add(key + ":w_any", float(cp[i, e] - cp[i, a] > 0))
        st = g.integers(1, S + 1 - L + 1, N_RAND)
        for s in st:
            nr = co[i, s + L] - co[i, s]
            if nr < 0.6 * L:
                continue
            acc.add(key + ":r", (cv[i, s + L] - cv[i, s]) / nr)
            if use_any:
                acc.add(key + ":r_any", float(cp[i, s + L] - cp[i, s] > 0))
        acc.d[key + ":n"] = (acc.d.get(key + ":n", (0.0, 0.0))[0] + 1, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="comma list of bundle dirs")
    ap.add_argument("--name", required=True, help="comma list of names (same order)")
    ap.add_argument("--batches", type=int, default=80)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--K", type=int, default=8, help="chunks per row")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bundles = a.bundle.split(","); names = a.name.split(",")
    assert len(bundles) == len(names)
    T_ = os.environ.get("MM_WORK_DIR", os.path.join(ROOT, "work"))
    out = a.out or os.path.join(T_, "runs", "p2react", "step0")
    os.makedirs(out, exist_ok=True)
    confs = [json.load(open(os.path.join(bd, "config.json"))) for bd in bundles]
    cfg = dict(confs[0]["train_config"])
    pack_dir = os.path.join(T_, "p2", "packs", "sweep")
    pk = resolve_pack(cfg["pack"], pack_dir)
    pk = [p for p in pk if isinstance(p, list) and "fv4_inc25.npz" in os.path.basename(p[0])]   # the val people
    assert len(pk) == 1, pk
    norm = dict(np.load(os.path.join(bundles[0], "norm.npz")))
    for bd in bundles[1:]:
        n2 = np.load(os.path.join(bd, "norm.npz"))
        assert all(np.allclose(norm[k], n2[k]) for k in norm), "bundles have different norms"
    kw = sampler_kwargs(cfg, pack_dir)
    kw.pop("dataset_weights", None)
    kw["log"] = log
    log(f"Sampler on {pk[0]}")
    smp = Sampler(pk, "cpu", seed=0, norm=norm, **kw)
    log(f"val hours {smp.hours('val'):.2f}, val streams {len(smp.split_streams('val'))}")
    models = []
    for bd, c in zip(bundles, confs):
        m = UnifiedModel(c["model"])
        sd = torch.load(os.path.join(bd, "weights.pt"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=True)
        m.eval()
        models.append(m)
    S, H, L = cfg["seq"], smp.H, smp.L
    nf = S + H + L + 2
    std, mean = smp.std, smp.mean
    goc = torch.as_tensor(spec.GROUP_OF_COL)
    aug = dict(AUG_NONE, p_override=0.0, max_prefix=cfg["prefix_runtime"])
    # laugh thresholds: partner smile / jaw quantiles over val frames (first batches)
    srcs = ["labels"] + names
    acc = {s: Acc() for s in srcs}
    allP = {n: [] for n in names}
    allY, allOK = [], []
    thr = None
    t0 = time.time()
    for bi in range(a.batches):
        b = smp.batch(a.B, S, a.K, split="val", aug=aug, fixed_seed=a.seed0 + bi)
        tok = b["tok"]
        gp, okp, part, has_p = partner_frames(smp, b, nf)
        sto = smp.store
        ev_p = smp._take(smp.EV, gp.to(sto)).float() * okp[..., None]
        if smp.EVOK is not None:
            ev_p = ev_p * smp.EVOK[part].float()[:, None, None]
        m_user = smp._take(smp.M, gp.to(sto)).float()
        k_user = smp._take(smp.K, gp.to(sto)).float() * okp[..., None]
        vad = smp._take(smp.A, gp.to(sto)).float()[..., -1]
        if thr is None:
            mk = (k_user[..., spec.GROUP_NAMES.index("mouth")] > 0) & okp
            sm = m_user[..., SMC].mean(-1)[mk]; jw = m_user[..., JAW][mk]
            thr = (float(torch.quantile(sm, 0.8)), float(torch.quantile(jw, 0.6)))
            log(f"laugh thresholds smile>{thr[0]:.3f} jaw>{thr[1]:.3f}")
        ev_l = laugh_onsets(m_user, k_user, vad, okp, thr)
        pev = {"smile": ev_p[..., SMILE], "nod": ev_p[..., NOD], "laugh": ev_l}
        # self labels on rel frames 1..S (index f); pad to S+2 (f = 0..S+1)
        Y = torch.zeros(a.B, S + 2, 3); OK = torch.zeros(a.B, S + 2, 3)
        Y[:, 1:S + 1] = b["ev_raw"]; OK[:, 1:S + 1] = b["ev_ok"] * has_p[:, None, None].float()
        allY.append(b["ev_raw"].numpy()); allOK.append(OK[:, 1:S + 1].numpy())
        for pn, sn in PAIRS:
            c = {"smile": SMILE, "nod": NOD}[sn]
            g = np.random.default_rng(bi * 100 + PAIRS.index((pn, sn)))
            window_stats(acc["labels"], f"{pn}->{sn}", Y[..., c], OK[..., c], pev[pn], g, True)
        for c, nm in ((SMILE, "smile"), (NOD, "nod")):
            acc["labels"].add(f"rate_{nm}", (Y[..., c] * OK[..., c]).sum(), OK[..., c].sum())
        # flow chunk frames + partner-smile recency on rel frames
        rec = past_any(ev_p[..., SMILE], LO, HI)                 # (B,nf) frame f is 0.3-2 s after a partner smile
        clean = ~past_any(ev_p[..., SMILE], 0, 120)              # no partner smile onset in the last 4 s
        rec_l = past_any(ev_l, LO, HI)
        tau, C = b["tau"], b["C"]
        hidx = tau[..., None] + 1 - C[..., None] + torch.arange(H)
        bidx = torch.arange(a.B)[:, None, None]
        fr_rec = rec[bidx, hidx]; fr_clean = clean[bidx, hidx]; fr_recl = rec_l[bidx, hidx]       # (B,K,H)
        hasp_f = has_p[:, None, None].expand_as(fr_rec)
        lm = b["lmask"][..., SMC].mean(-1) > 0.99                 # (B,K,H) smile cols predicted (not known / lip)
        valid = lm & hasp_f
        xt = (b["x1"][..., SMC] * std[SMC] + mean[SMC]).mean(-1)  # raw true smile level
        prev_s = (b["prev"][..., SMC] * std[SMC] + mean[SMC]).mean(-1)               # (B,K) true level before chunk
        prev_ok = (b["prev_ok"][..., SMC].amin(-1) > 0)[..., None] & (b["C"] == 0)[..., None]  # no known prefix
        def flow_feats(pre, lev):
            """per-frame level masks + within-row (win - clean) + change vs the pre-chunk true level."""
            r = {}
            for key, msk in (("all", valid), ("win", valid & fr_rec), ("clean", valid & fr_clean),
                             ("laughwin", valid & fr_recl)):
                r[key] = ((lev * msk).sum(), msk.sum())
            w = valid & fr_rec; c = valid & fr_clean
            nw = w.flatten(1).sum(1).float(); nc = c.flatten(1).sum(1).float()
            mw = (lev * w).flatten(1).sum(1) / nw.clamp(min=1); mc = (lev * c).flatten(1).sum(1) / nc.clamp(min=1)
            both = (nw >= 10) & (nc >= 10)
            r["rowdiff"] = (((mw - mc) * both).sum(), both.sum())
            dl = lev - prev_s[..., None]
            for key, msk in (("dwin", w & prev_ok), ("dclean", c & prev_ok)):
                r[key] = ((dl * msk).sum(), msk.sum())
            for k_, (sv, nv) in r.items():
                acc[pre[0]].add(pre[1] + k_, sv, nv)
        flow_add = lambda src, lev: flow_feats((src, "flow_"), lev)
        flow_add("labels", xt)
        tok_u = null_tokens(tok)
        for nm, m in zip(names, models):
            with torch.no_grad():
                h = m.backbone(tok); hu = m.backbone(tok_u)
                pe = torch.sigmoid(m.events(h).float()); peu = torch.sigmoid(m.events(hu).float())
            allP[nm].append(pe.numpy())
            P = torch.zeros(a.B, S + 2, 3); PU = torch.zeros(a.B, S + 2, 3)
            P[:, 1:S + 1] = pe; PU[:, 1:S + 1] = peu
            A_ = acc[nm]
            for pn, sn in PAIRS:
                c = {"smile": SMILE, "nod": NOD}[sn]
                g = np.random.default_rng(bi * 100 + PAIRS.index((pn, sn)))      # same random windows as labels
                window_stats(A_, f"{pn}->{sn}", P[..., c], OK[..., c], pev[pn], g, False)
                g = np.random.default_rng(bi * 100 + PAIRS.index((pn, sn)))
                window_stats(A_, f"U:{pn}->{sn}", PU[..., c], OK[..., c], pev[pn], g, False)
                g = np.random.default_rng(bi * 100 + PAIRS.index((pn, sn)))
                window_stats(A_, f"D:{pn}->{sn}", (P[..., c] - PU[..., c]).abs(), OK[..., c], pev[pn], g, False)
            for c, cn in ((SMILE, "smile"), (NOD, "nod")):
                o = OK[..., c]
                A_.add(f"rate_{cn}", (P[..., c] * o).sum(), o.sum())
                A_.add(f"rateU_{cn}", (PU[..., c] * o).sum(), o.sum())
                A_.add(f"absdiff_{cn}", ((P[..., c] - PU[..., c]).abs() * o).sum(), o.sum())
                A_.add(f"reldiff_{cn}", ((P[..., c] - PU[..., c]).abs() * o).sum(), (P[..., c] * o).sum())
            if not a.no_flow:
                mem, mm = gather_mem(h, b["mem_idx"], b["mem_ok"]); memu, _ = gather_mem(hu, b["mem_idx"], b["mem_ok"])
                Bk, K, Hh, D = b["x1"].shape
                kv = b["kvals"].reshape(-1, Hh, D); km = b["kmask"].reshape(-1, Hh, 9)
                sty = b["style_c"].reshape(-1, b["style_c"].shape[-1])
                with torch.no_grad():
                    pr = m.head.prepare(mem, mm, sty); pru = m.head.prepare(memu, mm, sty)
                    outs = {}
                    for key, (p_, pu_, gd) in {"cfg": (pr, pru, a.guidance), "g1": (pr, None, 1.0),
                                                "unull": (pru, None, 1.0)}.items():
                        gen = torch.Generator(); gen.manual_seed(3000 + bi)
                        x = sample_chunk(m.head, p_, kv, km, steps=a.steps, guidance=gd, temperature=a.temperature,
                                         gen=gen, prep_u=pu_, goc=goc)
                        outs[key] = (x.reshape(Bk, K, Hh, D)[..., SMC] * std[SMC] + mean[SMC]).mean(-1)
                for key, lev in outs.items():
                    flow_feats((nm, f"flow_{key}_"), lev)
        n_sm = acc["labels"].d.get("smile->smile:n", (0, 0))[0]
        log(f"batch {bi + 1}/{a.batches} ({time.time() - t0:.0f}s) partner smile onsets with full window: {n_sm:.0f}")
    # ---- calibration scale per model (label count / sum p over all valid frames)
    Yc = np.concatenate(allY); OKc = np.concatenate(allOK)
    res = {}
    lab = acc["labels"]
    def pair_rows(A_, scale=None, pref=""):
        r = {}
        for pn, sn in PAIRS:
            k = pref + f"{pn}->{sn}"
            w, rr = A_.mean(k + ":w"), A_.mean(k + ":r")
            d = dict(n=int(lab.d.get(f"{pn}->{sn}:n", (0, 0))[0]), p_window=w, p_base=rr, excess=w - rr,
                     ratio=w / rr if rr > 0 else float("nan"))
            if scale is not None:
                c = {"smile": SMILE, "nod": NOD}[sn]
                d["excess_onsets_per_window_cal"] = (w - rr) * (HI - LO + 1) * scale[c]
            if A_.cnt(k + ":w_any"):
                d["P_any_window"] = A_.mean(k + ":w_any"); d["P_any_base"] = A_.mean(k + ":r_any")
                d["excess_any"] = d["P_any_window"] - d["P_any_base"]
                d["excess_onsets_per_window"] = (w - rr) * (HI - LO + 1)
            r[f"{pn}->{sn}"] = d
        return r
    res["labels"] = dict(pairs=pair_rows(lab), rate=dict(smile=lab.mean("rate_smile"), nod=lab.mean("rate_nod")),
                         flow={k: lab.mean("flow_" + k) for k in ("all", "win", "clean", "laughwin", "rowdiff", "dwin", "dclean")},
                         n_flow_frames={k: lab.cnt("flow_" + k) for k in ("all", "win", "clean", "laughwin", "rowdiff", "dwin", "dclean")},
                         laugh_thr=thr, batches=a.batches, B=a.B, seq=S)
    for nm in names:
        A_ = acc[nm]
        Pc = np.concatenate(allP[nm])
        scale = [float((Yc[..., c] * OKc[..., c]).sum() / max((Pc[..., c] * OKc[..., c]).sum(), 1e-9)) for c in range(3)]
        r = dict(pairs=pair_rows(A_, scale), pairs_usernull=pair_rows(A_, scale, "U:"),
                 absdiff_window={f"{pn}->{sn}": dict(window=A_.mean(f"D:{pn}->{sn}:w"), base=A_.mean(f"D:{pn}->{sn}:r"))
                                 for pn, sn in PAIRS},
                 cal_scale=dict(zip(EVENT_NAMES, scale)))
        for cn in ("smile", "nod"):
            r[f"rate_{cn}"] = A_.mean(f"rate_{cn}"); r[f"rate_usernull_{cn}"] = A_.mean(f"rateU_{cn}")
            r[f"absdiff_real_vs_null_{cn}"] = A_.mean(f"absdiff_{cn}")
            r[f"reldiff_real_vs_null_{cn}"] = A_.mean(f"reldiff_{cn}")
        r["flow"] = {key: {k2: A_.mean(f"flow_{key}_{k2}") for k2 in ("all", "win", "clean", "laughwin", "rowdiff", "dwin", "dclean")}
                     for key in ("cfg", "g1", "unull")} if not a.no_flow else None
        r["flow_true"] = res["labels"]["flow"]
        res[nm] = r
        with open(os.path.join(out, f"{nm}.json"), "w") as f:
            json.dump(dict(bundle=bundles[names.index(nm)], **r), f, indent=1)
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(res["labels"], f, indent=1)
    write_md(res, names, out, a)
    log("done")


def write_md(res, names, out, a):
    L = []
    lab = res["labels"]
    L.append(f"# STEP0-TF: teacher-forced reaction on VAL (fv4_inc25 val people, {a.batches} x {a.B} x {lab['seq']} frames)")
    L.append("")
    L.append("Window = self frames u+9..u+60 after a partner onset at u; base = 16 random windows of the same length in "
             "the same rows. Per-frame onset rate (labels) / mean sigmoid p (models). 'cal' = model excess rescaled to "
             "onsets per window (sum p matched to the label count). P_any = P(at least one self onset in the window).")
    L.append("")
    L.append("## (a) event heads")
    L.append("")
    L.append("| pair | n | labels win/base (ratio) | labels excess onsets/win | labels P_any win-base | "
             + " | ".join(f"{n} win/base (ratio) | {n} cal excess | {n} user-null ratio" for n in names) + " |")
    L.append("|---|---|---|---|---|" + "---|---|---|" * len(names))
    for k, d in lab["pairs"].items():
        row = [k, str(d["n"]), f"{d['p_window']:.5f}/{d['p_base']:.5f} ({d['ratio']:.2f})",
               f"{d['excess_onsets_per_window']:+.3f}", f"{d['P_any_window']:.3f}-{d['P_any_base']:.3f} = {d['excess_any']:+.3f}"]
        for n in names:
            m = res[n]["pairs"][k]; mu = res[n]["pairs_usernull"][k]
            row += [f"{m['p_window']:.5f}/{m['p_base']:.5f} ({m['ratio']:.2f})", f"{m['excess_onsets_per_window_cal']:+.3f}",
                    f"{mu['ratio']:.2f}"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("| | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for cn in ("smile", "nod"):
        L.append(f"| mean p {cn} real / user-null (label rate {lab['rate'][cn]:.5f}) | "
                 + " | ".join(f"{res[n][f'rate_{cn}']:.5f} / {res[n][f'rate_usernull_{cn}']:.5f}" for n in names) + " |")
        L.append(f"| mean abs p(real)-p(null) {cn} (rel. to mean p) | "
                 + " | ".join(f"{res[n][f'absdiff_real_vs_null_{cn}']:.5f} ({res[n][f'reldiff_real_vs_null_{cn}']:.2f})"
                              for n in names) + " |")
    for k in ("smile->smile", "nod->nod"):
        L.append(f"| abs diff in {k} windows / random | "
                 + " | ".join(f"{res[n]['absdiff_window'][k]['window']:.5f} / {res[n]['absdiff_window'][k]['base']:.5f}"
                              for n in names) + " |")
    L.append("")
    if res[names[0]].get("flow"):
        L.append("## (b) flow output: self smile level (raw, mean mouthSmileL/R) on predicted chunk frames")
        L.append("")
        L.append(f"frames: all {lab['n_flow_frames']['all']:.0f}, 0.3-2 s after partner smile {lab['n_flow_frames']['win']:.0f}, "
                 f"no partner smile in 4 s {lab['n_flow_frames']['clean']:.0f}, after partner laugh {lab['n_flow_frames']['laughwin']:.0f}")
        L.append("")
        L.append("rise = after-partner-smile minus clean (pooled); rowdiff = the same within rows having both (removes "
                 "between-row confound); dwin/dclean = mean change of the chunk's level vs the TRUE level of the frame "
                 "before the chunk (chunks without a known prefix), after partner smile vs clean.")
        L.append("")
        L.append("| source | all | after p.smile | clean (none 4 s) | rise win-clean | rowdiff | dwin | dclean | after p.laugh | rise laugh-clean |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        f = lab["flow"]
        L.append(f"| TRUE | {f['all']:.4f} | {f['win']:.4f} | {f['clean']:.4f} | {f['win'] - f['clean']:+.4f} | "
                 f"{f['rowdiff']:+.4f} | {f['dwin']:+.4f} | {f['dclean']:+.4f} | {f['laughwin']:.4f} | {f['laughwin'] - f['clean']:+.4f} |")
        for n in names:
            for key, lbl in (("cfg", f"g{a.guidance}"), ("g1", "g1"), ("unull", "user-null")):
                f = res[n]["flow"][key]
                L.append(f"| {n} {lbl} | {f['all']:.4f} | {f['win']:.4f} | {f['clean']:.4f} | {f['win'] - f['clean']:+.4f} | "
                         f"{f['rowdiff']:+.4f} | {f['dwin']:+.4f} | {f['dclean']:+.4f} | {f['laughwin']:.4f} | {f['laughwin'] - f['clean']:+.4f} |")
    with open(os.path.join(out, "STEP0_TF.md"), "w") as fo:
        fo.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
