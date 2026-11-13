"""Train the unified model.

    python -I tools/train/train.py --config cfg.json --run-dir runs/x --pack-dir /dir/with/pack

Writes into --run-dir: metrics.jsonl, ckpt_last.pt (atomic, every ckpt_seconds; model + EMA weights, step),
ckpt_final.pt, bundle/ (the runtime bundle from the EMA weights) and DONE (JSON summary) at the end.
Resumes from ckpt_last.pt if present. The LR schedule is cosine on max(step/steps, elapsed/max_minutes), so a run
always finishes its schedule inside its time budget.
cfg "pack" = one pack file or a list of them. Optional knobs (absent = off): "patience_evals" N = stop when the best
val_core has not improved in N consecutive evals (once step >= 2 x warmup); "isolate_eval_rng" = evals no longer
reset the training batch / noise stream (see val()). tools/train/train_multi.py trains several models in lockstep on
one Sampler with the helpers below.
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from canon1 import spec  # noqa: E402
from unified.data import (AUG_NONE, EVENT_GROUPS, EVENT_NAMES, EVENT_REFRACTORY, LIP_COLS, STYLE_NAMES,  # noqa: E402
                          Sampler)
from unified.flow import flow_loss, gather_mem, sample_chunk  # noqa: E402
from unified.model import UnifiedModel, default_cfg, null_tokens  # noqa: E402

DEFAULTS = dict(pack="pack.npz", noise_profile=None, style_tables=None, model={}, batch=64, seq=150, chunks=8, lr=3e-4, wd=0.01,
                warmup=300, steps=20000, max_minutes=60.0, ema=0.999, aug={}, loss={}, ckpt_seconds=110,
                eval_every=500, log_every=50, threads=40, amp="bf16", seed=0, val_batches=4, grad_clip=1.0,
                prefix_runtime=2, guidance=1.5, steps_sample=6, temperature=0.8,
                # phase 2 knobs: audio_in "mel" | "enc_<name>"; enc_shift = user feature delay (frames); head "flow" |
                # "regression"; events = event heads (+ curves, calibration); loss.a2f = teacher weight (0 = off);
                # aug.event_weight = rare-event loss weighting near onsets (1 = off)
                audio_in="mel", enc_shift=0, head="flow", events=False, event_tol=1, event_curve_len=None,
                event_refractory=None, final_drops=True)
DROPS = ("user_audio", "user_face", "user_motion", "self_speech")


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


RESP_GROUPS = ("head", "face_upper", "gaze")


def _ap(p, y):
    """average precision of scores p for binary labels y (1-D numpy)."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def calibrate_events(P, Y, OK, refractory, tol=3):
    """P/Y/OK (R,S,C) val probabilities, raw onset labels, valid. Per class: the threshold whose rising-crossing +
    refractory fire count matches the label count (rate matching), precision/recall (+-tol frames) there, and the
    sample-mode scale (labels / sum p)."""
    R, S, C = P.shape
    out = dict(thresholds=[], fire_rate_per_min=[], label_rate_per_min=[], precision=[], recall=[], sample_scale=[],
               ap=[])
    for c in range(C):
        p = np.where(OK[..., c] > 0, P[..., c], 0.0)
        y = Y[..., c] * (OK[..., c] > 0)
        mins = max(float(OK[..., c].sum()) / 30 / 60, 1e-9)
        n_lab = float(y.sum())
        grid = np.unique(np.quantile(p[OK[..., c] > 0], np.linspace(0.5, 0.9999, 120))) if (OK[..., c] > 0).any() \
            else np.array([0.5])
        G = len(grid)
        last = np.full((R, G), -10 ** 6)
        prev = np.zeros((R, G), bool)
        cnt = np.zeros(G)
        fires = np.zeros((R, S, G), bool)
        for t in range(S):
            above = p[:, t, None] >= grid[None]
            f = above & ~prev & (t - last >= refractory[c])
            fires[:, t] = f
            last = np.where(f, t, last)
            cnt += f.sum(0)
            prev = above
        gi = int(np.argmin(np.abs(cnt - n_lab)))
        fz = fires[..., gi]
        # matches within +-tol frames
        yd = np.zeros_like(y, bool); fd = np.zeros_like(fz)
        for k in range(-tol, tol + 1):
            yd |= np.roll(y > 0, k, axis=1); fd |= np.roll(fz, k, axis=1)
        out["thresholds"].append(float(grid[gi]))
        out["fire_rate_per_min"].append(round(float(cnt[gi]) / mins, 3))
        out["label_rate_per_min"].append(round(n_lab / mins, 3))
        out["precision"].append(round(float((fz & yd).sum() / max(fz.sum(), 1)), 4))
        out["recall"].append(round(float(((y > 0) & fd).sum() / max(n_lab, 1)), 4))
        out["sample_scale"].append(round(n_lab / max(float(p.sum()), 1e-9), 4))
        m = OK[..., c] > 0
        out["ap"].append(round(_ap(p[m], (yd & m)[m].astype(np.float64)), 4))
    return out


@torch.no_grad()
def val_metrics(model, smp, cfg, n_batches=4, B=32, sample=True, drop=None, collect_events=False):
    """val flow loss (fixed seeds, no augmentation) + per-group MAE (raw units) of sampled chunks vs truth +
    RESPONSIVENESS: the same chunks (same noise) sampled with the user stream nulled; val_resp = mean |guided -
    user-null| / mean |truth - its chunk mean| over head / face_upper / gaze (normalized units, unknown entries).
    the first learned-reaction model ignored the user (0.07% closed-loop) - this is the open-loop number to watch during training."""
    model.eval()
    lw = dict(cfg["loss"])
    goc = torch.as_tensor(spec.GROUP_OF_COL, device=smp.dev)
    tot, parts_sum = 0.0, {}
    mae_num = torch.zeros(9, device=smp.dev); mae_den = torch.zeros(9, device=smp.dev)
    rnum = rden = rnum1 = 0.0
    rmask = torch.zeros(189, device=smp.dev)
    for g_ in RESP_GROUPS:
        rmask[spec.GROUPS[g_]] = 1
    aug = dict(AUG_NONE, p_override=0.0, max_prefix=cfg["prefix_runtime"])
    if not cfg.get("use_style", True):
        aug["p_style_null"] = 1.0
    reg = model.cfg.get("head_kind", "flow") == "regression"
    n_steps, temp = (1, 0.0) if reg else (cfg["steps_sample"], cfg["temperature"])
    evs = {"p": [], "y": [], "ok": []}
    for i in range(n_batches):
        b = smp.batch(B, cfg["seq"], 4, split="val", aug=aug, fixed_seed=1000 + i, drop=drop)
        torch.manual_seed(2000 + i)
        l, parts = flow_loss(model, b, smp.colw, lw)
        tot += float(l)
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + float(v)
        if model.events is not None and b.get("ev_tgt") is not None:
            pe = torch.sigmoid(model.events(model.hidden(b["tok"])).float())
            evs["p"].append(pe.cpu().numpy()); evs["y"].append(b["ev_raw"].cpu().numpy())
            evs["ok"].append(b["ev_ok"].cpu().numpy())
        if sample:
            # sample the chunks with the runtime settings, CFG on the user stream
            tok_u = null_tokens(b["tok"])
            h = model.hidden(b["tok"]); hu = model.hidden(tok_u)
            mem, mm = gather_mem(h, b["mem_idx"], b["mem_ok"]); memu, _ = gather_mem(hu, b["mem_idx"], b["mem_ok"])
            Bk, K, H, D = b["x1"].shape
            kv = b["kvals"].reshape(-1, H, D); km = b["kmask"].reshape(-1, H, 9)
            sty = b["style_c"].reshape(-1, b["style_c"].shape[-1])
            g = torch.Generator(device=smp.dev); g.manual_seed(3000 + i)
            x = sample_chunk(model.head, model.head.prepare(mem, mm, sty), kv, km, steps=n_steps,
                             guidance=cfg["guidance"], temperature=temp, gen=g,
                             prep_u=model.head.prepare(memu, mm, sty), goc=goc)
            g = torch.Generator(device=smp.dev); g.manual_seed(3000 + i)
            xn = sample_chunk(model.head, model.head.prepare(memu, mm, sty), kv, km, steps=n_steps,
                              guidance=1.0, temperature=temp, gen=g, goc=goc)
            g = torch.Generator(device=smp.dev); g.manual_seed(3000 + i)
            x1g = sample_chunk(model.head, model.head.prepare(mem, mm, sty), kv, km, steps=n_steps,
                               guidance=1.0, temperature=temp, gen=g, goc=goc)
            lmr = b["lmask"].reshape(-1, H, D) * rmask
            xt = b["x1"].reshape(-1, H, D)
            rnum += float((((x - xn).abs()) * lmr).sum()); rnum1 += float((((x1g - xn).abs()) * lmr).sum())
            rden += float(((xt - xt.mean(1, keepdim=True)).abs() * lmr).sum())
            err = (x - b["x1"].reshape(-1, H, D)).abs() * smp.std             # raw units
            lm = b["lmask"].reshape(-1, H, D)
            for gi in range(9):
                c = goc == gi
                mae_num[gi] += (err[..., c] * lm[..., c]).sum()
                mae_den[gi] += lm[..., c].sum()
    model.train()
    out = {"val_loss": tot / n_batches}
    out.update({"val_" + k: v / n_batches for k, v in parts_sum.items()})
    # val_core = the pre-p2 val_loss (flow + vel + jerk): comparable across event/a2f settings; drives ckpt_best
    out["val_core"] = (out.get("val_flow", 0.0) + lw.get("vel", 0.5) * out.get("val_vel", 0.0)
                       + lw.get("jerk", 0.1) * out.get("val_jerk", 0.0))
    if sample:
        mae = (mae_num / mae_den.clamp(min=1)).tolist()
        out["val_mae"] = {g: round(m, 5) for g, m in zip(spec.GROUP_NAMES, mae)}
        out["val_resp"] = round(rnum / max(rden, 1e-9), 4)              # with CFG (runtime guidance)
        out["val_resp_g1"] = round(rnum1 / max(rden, 1e-9), 4)          # without guidance
    if evs["p"]:
        P, Y, OK = (np.concatenate(evs[k]) for k in ("p", "y", "ok"))
        tol = 3
        out["val_event_ap"] = {}
        for c, nm in enumerate(EVENT_NAMES):
            yd = np.zeros_like(Y[..., c], bool)
            for k in range(-tol, tol + 1):
                yd |= np.roll(Y[..., c] > 0, k, axis=1)
            m = OK[..., c] > 0
            out["val_event_ap"][nm] = round(_ap(P[..., c][m], yd[m].astype(np.float64)), 4)
        if collect_events:
            out["_events"] = (P, Y, OK)
    return out


def write_bundle(run_dir, model_state, cfg, smp, extra, events_cal=None):
    bd = os.path.join(run_dir, "bundle")
    tmpd = bd + ".tmp"
    shutil.rmtree(tmpd, ignore_errors=True)
    os.makedirs(tmpd)
    torch.save(model_state, os.path.join(tmpd, "weights.pt"))
    np.savez(os.path.join(tmpd, "norm.npz"), **smp.norm)
    mc = dict(default_cfg(), **cfg["model"])              # = UnifiedModel(cfg["model"]).cfg, without building it
    p2 = {}
    if smp.audio_in != "mel":
        k = smp.k_enc
        p2["audio"] = {"in": smp.audio_in, "feat_dim": int(smp.pca["epca_V"].shape[1]) if smp.pca is not None else k,
                       "proj_dim": k, "user_shift": smp.enc_shift, "keep_vad": True}
        if cfg.get("audio_meta"):                         # the encoder's META.json (runtime builds the encoder from it)
            p2["audio"]["encoder"] = cfg["audio_meta"]
    if int(mc.get("n_events", 0)) > 0 and smp.event_info is not None:
        ei = smp.event_info
        np.savez(os.path.join(tmpd, "events.npz"), curves=ei["curves"].astype(np.float32),
                 lens=np.asarray(ei["lens"], np.int64))
        cal = events_cal or {}
        p2["events"] = dict(names=list(EVENT_NAMES), groups=list(EVENT_GROUPS),
                            refractory=list(cfg.get("event_refractory") or EVENT_REFRACTORY),
                            thresholds=cal.get("thresholds", [0.5] * len(EVENT_NAMES)),
                            label_rate_per_min=ei["rate_val_per_min"], label_rate_train_per_min=ei["rate_train_per_min"],
                            calibration=cal, curve_onsets=ei["n_onsets_used"])
    conf = dict(kind="flow", version="unified-v2", model=mc, self_audio_lookahead=mc["lookahead"],
                horizon=mc["horizon"], prefix=cfg["prefix_runtime"], guidance=cfg["guidance"],
                steps=cfg["steps_sample"], temperature=cfg["temperature"], replan_every=3,
                style=dict(names=STYLE_NAMES, input="values in [0,1] + per-component null flags",
                           tables=smp.style_tables),
                lip=dict(cols=[int(c) for c in LIP_COLS], rule="target speaking (vad>0.5, dilated 2 frames): "
                         "loss-masked; runtime writes the train mean"),
                train_config=cfg, **p2, **extra)
    rt = dict(cfg.get("runtime") or {})
    if mc.get("head_kind") == "regression":
        rt.update(steps=1, temperature=0.0)
    if "events" in p2:
        rt["events"] = dict(dict(mode="threshold", gain=[1.0] * len(EVENT_NAMES), feed_history=True),
                            **(rt.get("events") or {}))
    if rt:
        conf["runtime"] = rt                              # sampling defaults UnifiedPlanner reads
    with open(os.path.join(tmpd, "config.json"), "w") as f:
        json.dump(conf, f, indent=1)
    mp = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mp):
        shutil.copy(mp, os.path.join(tmpd, "curves.jsonl"))
    shutil.rmtree(bd, ignore_errors=True)
    os.replace(tmpd, bd)
    return bd


# ------------------------------------------------------------------------------------------------ shared with train_multi.py
def resolve_path(p, pack_dir):
    return p if os.path.isabs(p) else os.path.join(pack_dir, p)


def resolve_pack(pack, pack_dir):
    """cfg["pack"]: one pack file or a list of them (the Sampler concatenates a list); relative entries are under
    pack_dir. A str stays a str, a list stays a list."""
    if isinstance(pack, (list, tuple)):                   # entries may be lists themselves (member-split pack files)
        return [resolve_pack(p, pack_dir) for p in pack]
    return resolve_path(pack, pack_dir)


def sampler_kwargs(cfg, pack_dir):
    """the Sampler keyword arguments a config builds (everything but the pack path, device and seed)."""
    noise = None
    if cfg.get("noise_profile"):
        npth = resolve_path(cfg["noise_profile"], pack_dir)
        if os.path.exists(npth):
            with open(npth) as f:
                noise = json.load(f)
    mc = dict(default_cfg(), **cfg["model"])
    stab = None
    if cfg.get("style_tables"):
        sp = resolve_path(cfg["style_tables"], pack_dir)
        if os.path.exists(sp):
            with open(sp) as f:
                stab = json.load(f)
    kw = dict(noise_profile=noise, H=mc["horizon"], L=mc["lookahead"], mem=mc["mem"], style_tables=stab,
              style_procs=int(cfg.get("style_procs", 8)), log=lambda m: print(m, flush=True),
              audio_in=cfg.get("audio_in", "mel"),
              enc_shift=("auto" if cfg.get("enc_shift") == "auto" else int(cfg.get("enc_shift") or 0)),
              use_events=bool(cfg.get("events")) or float((cfg.get("aug") or {}).get("event_weight", 1.0)) != 1.0,
              use_a2f=float((cfg.get("loss") or {}).get("a2f", 0)) > 0,
              event_tol=int(cfg.get("event_tol", 1)), event_curve_len=cfg.get("event_curve_len"),
              pack_on_cpu=bool(cfg.get("pack_on_cpu", False)), a2f_dials=cfg.get("a2f_dials"),
              pack_extra=[resolve_path(x, pack_dir) for x in cfg.get("pack_extra") or []])
    for k in ("lean", "enc_missing", "dataset_weights"):  # newer Sampler options: passed only when the config sets them
        if k in cfg:
            kw[k] = cfg[k]
    return kw


def complete_model_cfg(cfg, smp):
    """phase 2: the model's audio width / event heads / head kind follow the data + config (cfg["model"], in place)."""
    cfg["model"] = dict(cfg["model"], aud_dim=smp.aud_dim,
                        n_events=(len(EVENT_NAMES) if smp.EV is not None and cfg.get("events") else 0),
                        head_kind=cfg.get("head", "flow"))


def build_model(cfg, dev):
    """model + its EMA copy (no grads) + AdamW (no weight decay on 1-D params and positional tables)."""
    model = UnifiedModel(cfg["model"]).to(dev)
    ema = UnifiedModel(cfg["model"]).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith(".pos") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]}, {"params": no_decay, "weight_decay": 0}],
                            lr=cfg["lr"], betas=(0.9, 0.99))
    return model, ema, opt


def run_info(model, smp, dev):
    mc = model.cfg
    return dict(params=model.n_params(), params_backbone=sum(p.numel() for p in model.backbone.parameters()),
                params_head=sum(p.numel() for p in model.head.parameters()), device=dev, audio_in=smp.audio_in,
                aud_dim=smp.aud_dim, n_events=mc["n_events"], head_kind=mc["head_kind"],
                event_rates=None if smp.event_info is None else smp.event_info["rate_train_per_min"],
                gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
                train_hours=smp.hours("train"), val_hours=smp.hours("val"), datasets=smp.names["datasets"],
                pack_hours=smp.names.get("hours"))


def lr_at(cfg, step, frac):
    """linear warmup, then cosine to 5% over frac = max(step / steps, elapsed / max_s)."""
    return cfg["lr"] * min(1.0, (step + 1) / cfg["warmup"]) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))


def ema_update(ema, model, cfg, step):
    d = min(cfg["ema"], (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - d)


def is_spike(cfg, gnf, loss, gn_log, step):
    """final3: skip a step whose grad norm spikes (> spike_factor x running median) or is non-finite."""
    sf = float(cfg.get("spike_factor", 0) or 0)
    return sf > 0 and (not math.isfinite(gnf) or not math.isfinite(float(loss))
                       or (gn_log is not None and step > cfg["warmup"] and math.log(max(gnf, 1e-12)) > gn_log + math.log(sf)))


def gn_track(gn_log, gnf):
    return math.log(max(gnf, 1e-12)) if gn_log is None else 0.99 * gn_log + 0.01 * math.log(max(gnf, 1e-12))


def eval_verdict(cfg, vl, step, est):
    """one eval's val_core vl. est = {"best": {"val", "step"}, "bad": n, "since": n} (updated in place).
    Returns (improved, stop_reason or None): blow-up = val > blowup x best (or non-finite) twice in a row; patience
    (cfg "patience_evals" N, default off) = no improvement of the best val_core in N consecutive evals, once
    step >= 2 x warmup."""
    if math.isfinite(vl) and vl < est["best"]["val"]:
        est["best"] = {"val": vl, "step": step}
        est["bad"] = est["since"] = 0
        return True, None
    est["since"] += 1
    if not math.isfinite(vl) or vl > float(cfg.get("blowup", 1.3)) * est["best"]["val"]:
        est["bad"] += 1
        if est["bad"] >= 2:
            return False, f"val blow-up at step {step}: {vl:.4f} vs best {est['best']['val']:.4f} @ {est['best']['step']}"
    else:
        est["bad"] = 0
    pe = cfg.get("patience_evals")
    if pe and est["since"] >= int(pe) and step >= 2 * cfg["warmup"]:
        return False, (f"patience: no val_core improvement in {est['since']} evals "
                       f"(best {est['best']['val']:.4f} @ step {est['best']['step']})")
    return False, None


def rng_state(dev):
    return torch.get_rng_state(), (torch.cuda.get_rng_state() if str(dev).startswith("cuda") else None)


def set_rng_state(st, dev):
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state(st[1])


def val(model, smp, cfg, **kw):
    """val_metrics(model, smp, cfg, **kw). val_metrics reseeds the Sampler's generator and the global torch RNG with
    its fixed seeds, so by default the TRAINING batches and flow noise after every eval restart
    from the same state: each eval period replays the previous one's batches. cfg "isolate_eval_rng": true restores
    both after the call (the val numbers are the same either way)."""
    if not cfg.get("isolate_eval_rng"):
        return val_metrics(model, smp, cfg, **kw)
    gs, rs = smp.g.get_state(), rng_state(smp.dev)
    try:
        return val_metrics(model, smp, cfg, **kw)
    finally:
        smp.g.set_state(gs)
        set_rng_state(rs, smp.dev)


def final_phase(ema, smp, cfg, best, bp, dev, stop, stop_reason, skipped):
    """the end-of-run evaluation: vcmp on the eval batches; the BEST EMA (bp = ckpt_best.pt) is loaded into ema unless
    the last EMA is at least as good; full val metrics (>= 8 batches) + event thresholds by rate matching (+ 8 more
    val batches unless stop["flag"], i.e. SIGTERM) + the input-drop contributions (cfg final_drops, not after SIGTERM).
    Returns (vm = the final row's metrics, events_cal)."""
    vcmp = val(ema, smp, cfg, n_batches=cfg["val_batches"], sample=False)   # same batches as the evals
    use_best = best["step"] is not None and os.path.exists(bp) and not (float(vcmp["val_core"]) <= best["val"]) \
        and cfg.get("bundle_from", "best") == "best"      # reaction adapter fine-tunes: "last" (new losses, not val_core)
    if use_best:
        _b = torch.load(bp, map_location=dev, weights_only=True)      # the bundle is the BEST EMA, not the last
        ema.load_state_dict(_b["ema"])
    vm = val(ema, smp, cfg, n_batches=max(cfg["val_batches"], 8), collect_events=True)
    events_cal = None
    if "_events" in vm:                                   # phase 2: thresholds by rate matching on val
        P, Y, OK = vm.pop("_events")
        if not stop["flag"]:
            vx = val(ema, smp, dict(cfg), n_batches=8, B=32, sample=False, collect_events=True)
            if "_events" in vx:                           # more val windows for the calibration (other seeds)
                P2, Y2, OK2 = vx.pop("_events")
                P, Y, OK = np.concatenate([P, P2]), np.concatenate([Y, Y2]), np.concatenate([OK, OK2])
        events_cal = calibrate_events(P, Y, OK, list(cfg.get("event_refractory") or EVENT_REFRACTORY))
        vm["events_cal"] = events_cal
    if cfg.get("final_drops", True) and not stop["flag"]:  # phase 2 (f): open-loop input contributions
        base = val(ema, smp, cfg, n_batches=cfg["val_batches"], sample=False)
        vm["val_drop"] = {}
        for dname in DROPS:
            vd = val(ema, smp, cfg, n_batches=cfg["val_batches"], sample=False, drop={dname})
            row_d = {"val_core": round(vd["val_core"], 5), "d_val_core": round(vd["val_core"] - base["val_core"], 5),
                     "d_val_flow": round(vd.get("val_flow", 0) - base.get("val_flow", 0), 5)}
            if "val_event_ap" in vd:
                row_d["event_ap"] = vd["val_event_ap"]
            vm["val_drop"][dname] = row_d
    vm["last_val_loss_cmp"] = float(vcmp["val_core"])
    vm["best_val_loss_cmp"] = best["val"]
    vm["from_best_step"] = best["step"] if use_best else None
    vm["stop_reason"] = stop_reason
    vm["skipped_steps"] = skipped
    return vm, events_cal


def quick_calibration(ema, smp, cfg):
    """SIGTERM path: event thresholds from 2 val batches (best effort; None without an event head)."""
    if ema.cfg.get("n_events", 0) <= 0:
        return None
    try:
        vq = val(ema, smp, cfg, n_batches=2, sample=False, collect_events=True)
        if "_events" in vq:
            return calibrate_events(*vq.pop("_events"), list(cfg.get("event_refractory") or EVENT_REFRACTORY))
    except Exception as e:                               # noqa: BLE001
        print(f"quick calibration failed: {e}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--set", action="append", default=[], help="key=json override")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = dict(DEFAULTS, **json.load(f))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if not cfg.get("use_style", True):                    # ablation: the style input is always the learned null
        cfg["aug"] = dict(cfg.get("aug") or {}, p_style_null=1.0)
    os.makedirs(a.run_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(int(cfg["threads"]))
    else:                   # concurrent runs on one GPU box thrash a CPU quota (GPU 14-34% busy) - cap threads
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    pack = resolve_pack(cfg["pack"], a.pack_dir)
    smp_kw = sampler_kwargs(cfg, a.pack_dir)
    smp = Sampler(pack, dev, seed=cfg["seed"], **smp_kw)
    norm0 = {k: v.copy() for k, v in smp.norm.items()}
    complete_model_cfg(cfg, smp)
    model, ema, opt = build_model(cfg, dev)
    mc = model.cfg
    step, elapsed0 = 0, 0.0
    ck = os.path.join(a.run_dir, "ckpt_last.pt")
    est = {"best": {"val": float("inf"), "step": None}, "bad": 0, "since": 0}
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=True)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        step, elapsed0 = int(st["step"]), float(st["elapsed"])
        est["since"] = int(st.get("since_best", 0))
        if "opt" in st:
            try:
                opt.load_state_dict(st["opt"])
            except Exception:                                           # noqa: BLE001
                pass
        print(f"resumed from step {step}", flush=True)
    info = run_info(model, smp, dev)
    print(json.dumps(info), flush=True)
    mlog = open(os.path.join(a.run_dir, "metrics.jsonl"), "a")
    mlog.write(json.dumps(dict(event="start", step=step, **info)) + "\n"); mlog.flush()
    amp = dev == "cuda" and cfg["amp"] == "bf16"
    if dev == "cuda":                                     # fp32 runs ("amp": "none") still use TF32 tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # final3 stability: skip steps whose grad norm spikes (> spike_factor x running median, or non-finite),
    # keep ckpt_best.pt by val loss (never rotated: it lives in the run dir, so every durable snapshot holds it),
    # stop early when val blows up (> blowup x best twice in a row, or non-finite) or (patience_evals) stops improving;
    # the bundle = the BEST EMA.
    gn_log = None
    skipped = skipped_run = 0
    stop_reason = None
    bp = os.path.join(a.run_dir, "ckpt_best.pt")
    if os.path.exists(bp):
        try:
            _b = torch.load(bp, map_location="cpu", weights_only=True)
            est["best"] = {"val": float(_b["val"]), "step": int(_b["step"])}
        except Exception:                                 # noqa: BLE001
            pass
    t_start = time.time() - elapsed0
    last_ck = time.time()
    max_s = cfg["max_minutes"] * 60
    budget = os.environ.get("TRAIN_TIME_BUDGET_S")
    if budget:                       # a scheduler stops us at this point: leave room for final eval + bundle
        max_s = min(max_s, elapsed0 + float(budget) - float(cfg.get("end_margin_s", 240)))
        print(f"time budget {budget}s -> training until t={max_s:.0f}s", flush=True)
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    run_loss = {}
    nlog = 0
    # pack_next = a bigger pack still being produced when the run starts; when <pack_next>.ready appears in the
    # pack dir (written once the file is complete) the sampler is swapped in, KEEPING the current
    # normalization (the model's input/output scale must not move mid-run).
    pn = cfg.get("pack_next")
    pn_path = None if not pn else resolve_path(pn, a.pack_dir)
    switched = {"done": bool(pn_path) and isinstance(pack, str) and os.path.abspath(pn_path) == os.path.abspath(pack)}

    def maybe_switch():
        nonlocal smp
        if switched["done"] or not pn_path or not os.path.exists(pn_path + ".ready"):
            return
        t0 = time.time()
        print(f"switching pack -> {pn_path}", flush=True)
        old = smp
        smp = None
        del old
        if dev == "cuda":
            torch.cuda.empty_cache()
        smp = Sampler(pn_path, dev, seed=cfg["seed"] + step, norm=norm0, **smp_kw)
        switched["done"] = True
        row = dict(event="pack_switch", step=step, pack=pn_path, secs=round(time.time() - t0, 1),
                   train_hours=smp.hours("train"), val_hours=smp.hours("val"), pack_hours=smp.names.get("hours"))
        info.update(train_hours=row["train_hours"], val_hours=row["val_hours"], pack_hours=row["pack_hours"])
        mlog.write(json.dumps(row) + "\n"); mlog.flush()
        print(json.dumps(row), flush=True)

    def save(final=False):
        st = dict(model=model.state_dict(), ema=ema.state_dict(), step=step, elapsed=time.time() - t_start,
                  cfg=cfg, since_best=est["since"])
        atomic_save(st, ck)
        if final:
            atomic_save(st, os.path.join(a.run_dir, "ckpt_final.pt"))

    model.train()
    while True:
        el = time.time() - t_start
        frac = max(step / cfg["steps"], el / max_s)
        if frac >= 1.0:
            break
        if stop["flag"]:
            print("SIGTERM: saving checkpoint + bundle and exiting", flush=True)
            save()
            cal = quick_calibration(ema, smp, cfg)
            write_bundle(a.run_dir, ema.state_dict(), cfg, smp, dict(steps_done=step, interrupted=True), events_cal=cal)
            sys.exit(0)
        lr = lr_at(cfg, step, frac)
        for gq in opt.param_groups:
            gq["lr"] = lr
        if pn_path and step % 100 == 0:
            maybe_switch()
        b = smp.batch(cfg["batch"], cfg["seq"], cfg["chunks"], split="train", aug=cfg["aug"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, parts = flow_loss(model, b, smp.colw, cfg["loss"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        gnf = float(gn)
        if is_spike(cfg, gnf, loss, gn_log, step):
            opt.zero_grad(set_to_none=True)
            skipped += 1; skipped_run += 1
            if skipped_run in (1, 10, 100):
                print(json.dumps(dict(event="skip_step", step=step, gn=gnf, loss=float(loss), skipped=skipped)), flush=True)
            if skipped_run >= 200:
                stop_reason = "200 consecutive skipped steps"
                break
            continue
        skipped_run = 0
        gn_log = gn_track(gn_log, gnf)
        opt.step()
        step += 1
        ema_update(ema, model, cfg, step)
        for k, v in parts.items():
            run_loss[k] = run_loss.get(k, 0.0) + float(v)
        run_loss["gn"] = run_loss.get("gn", 0.0) + float(gn)
        nlog += 1
        if step % cfg["log_every"] == 0:
            row = dict(step=step, t=round(time.time() - t_start, 1), lr=lr,
                       **{k: round(v / nlog, 5) for k, v in run_loss.items()})
            mlog.write(json.dumps(row) + "\n"); mlog.flush()
            print(json.dumps(row), flush=True)
            run_loss, nlog = {}, 0
        if step % cfg["eval_every"] == 0:
            vm = val(ema, smp, cfg, n_batches=cfg["val_batches"])
            row = dict(step=step, t=round(time.time() - t_start, 1), skipped=skipped, **vm)
            mlog.write(json.dumps(row) + "\n"); mlog.flush()
            print(json.dumps(row), flush=True)
            vl = float(vm["val_core"])
            improved, why = eval_verdict(cfg, vl, step, est)
            if improved:
                atomic_save(dict(ema=ema.state_dict(), model=model.state_dict(), step=step, val=vl, cfg=cfg), bp)
            if why:
                stop_reason = why
                print(stop_reason, flush=True)
                break
        if time.time() - last_ck > cfg["ckpt_seconds"]:
            save()
            last_ck = time.time()
    vm, events_cal = final_phase(ema, smp, cfg, est["best"], bp, dev, stop, stop_reason, skipped)
    row = dict(step=step, t=round(time.time() - t_start, 1), final=True, **vm)
    mlog.write(json.dumps(row) + "\n"); mlog.flush()
    print(json.dumps(row), flush=True)
    save(final=True)
    el = time.time() - t_start
    extra = dict(train_hours=info["train_hours"], val_hours=info["val_hours"], pack_hours=info["pack_hours"],
                 params=info["params"], steps_done=step, train_minutes=round(el / 60, 2), device=dev, gpu=info["gpu"],
                 final_val=vm)
    write_bundle(a.run_dir, ema.state_dict(), cfg, smp, extra, events_cal=events_cal)
    write_json_atomic(dict(step=step, minutes=round(el / 60, 2), final_val=vm, params=info["params"]),
                      os.path.join(a.run_dir, "DONE"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
