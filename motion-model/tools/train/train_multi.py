"""Train SEVERAL models in LOCKSTEP on one shared Sampler (one pack copy on the device): a scaling-sweep lane.

    python -I tools/train/train_multi.py --config cfg.json --run-dir R --pack-dir P [--set key=json]
    (multi.py: --train-script tools/train/train_multi.py; the lane's run config carries "models")

cfg.json = a train.py config (top-level keys are shared by every model) + "models": {"<name>": {overrides}, ...}, e.g.
    {"pack": "full_v3s25.npz", "batch": 96, "steps": 45000, ..., "isolate_eval_rng": true,
     "models": {"s5": {"model": {"d": 256, "layers": 5}, "lr": 3e-4, "sweep": {"lane": "d25", "size": "s5"}},
                "s20": {"model": {"d": 384, "layers": 8}, "lr": 2.5e-4, "patience_evals": 6}}}
Model <name>'s config = train.py DEFAULTS <- the top level (minus "models") <- models[<name>] (dict values merged one
level deep like multi.py: "model", "aug", "loss", ...), written to R/<name>/cfg.json. The keys that build the Sampler
or the batch (SHARED, + events / loss.a2f > 0 truthiness, model.horizon / lookahead / mem) must come out identical for
every model, else the run stops before loading anything.
Each iteration draws ONE batch; every active model (sorted names) then takes its own train.py step on it: lr schedule
(cosine on max(step / steps, elapsed / max_s), its own clock), flow loss, clip, spike skip, AdamW, EMA, log rows, evals
(val_metrics on its EMA, ckpt_best, blow-up / patience stops) and ckpt_last every ckpt_seconds (at most one model saves
per iteration). Every model writes a train.py run dir R/<name>/ (metrics.jsonl, cfg.json, ckpt_last.pt, ckpt_best.pt,
bundle/, DONE) so the downstream tools work unchanged. A model that ends (steps / time / blow-up / patience) runs
train.py's final phase at once, writes its bundle + DONE and is freed (the others get its time and memory).
R/DONE = {"<name>": {step, best_val, best_step, stop_reason, interrupted, minutes}} once every model has finished.
Resume: a model with a non-interrupted DONE is not trained again; the others resume from their ckpt_last.pt.

RNG: torch.manual_seed(seed) + np.random.seed(seed); the Sampler (its own generator, seeded with seed) draws the
shared batches; model i (index in sorted names) is built right after torch.manual_seed(seed + i) and its flow noise
continues THAT global-RNG stream, kept per model (swapped in around its step), so one model's draws never shift
another's. One model in the lane = train.py exactly (same rows, same weights).
Differences from train.py: ckpt_best.pt = {ema, step, val, cfg} (no raw weights; train.py only reads val / step from
it); ckpt_final.pt only with "write_final_ckpt": true; end_margin_s defaults to 300 s x models trained here and the
time budget counts from process start (pack load + model builds included); "pack_next" is not supported. SIGTERM =
at the next model boundary every unfinished model gets ckpt_last, a bundle from its BEST EMA, a quick event
calibration (2 val batches), a final row and DONE with "interrupted": true; then exit 0.
"""
import argparse
import gc
import json
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multi import merge  # noqa: E402
from train import (DEFAULTS, atomic_save, build_model, complete_model_cfg, ema_update, eval_verdict,  # noqa: E402
                   final_phase, gn_track, is_spike, lr_at, quick_calibration, resolve_pack, rng_state, run_info,
                   resolve_path, sampler_kwargs, set_rng_state, val, write_bundle, write_json_atomic)
from unified.data import Sampler  # noqa: E402
from unified.flow import flow_loss  # noqa: E402
from unified.model import default_cfg  # noqa: E402
import next_arms  # noqa: E402  per-model "arm" treatments (self-history corruption, reaction weighting, contrast)

# keys that build the Sampler or the batch: one value for the whole lane
SHARED = ("pack", "pack_extra", "noise_profile", "style_tables", "style_procs", "audio_in", "enc_shift", "aug",
          "event_tol", "event_curve_len", "pack_on_cpu", "a2f_dials", "lean", "enc_missing", "dataset_weights", "batch", "seq",
          "chunks",
          "seed", "eval_every", "val_batches", "max_minutes", "threads", "cpu_threads", "use_style", "isolate_eval_rng",
          "pack_next")
RESERVED = {"DONE", "DONE.tmp", "cfg.json", "stdout.log"}


def shared_sig(c):
    mm = dict(default_cfg(), **(c.get("model") or {}))
    sig = {k: c.get(k) for k in SHARED}
    sig["events"] = bool(c.get("events"))
    sig["loss.a2f > 0"] = float((c.get("loss") or {}).get("a2f", 0)) > 0
    for k in ("horizon", "lookahead", "mem"):
        sig["model." + k] = mm[k]
    return sig


def model_cfgs(top):
    """top-level config (with "models") -> {name: merged per-model config}; exits on a bad lane."""
    top = dict(top)
    models = top.pop("models", None)
    if not isinstance(models, dict) or not models:
        raise SystemExit('train_multi: the config needs "models": {"<name>": {overrides}, ...}')
    base = dict(DEFAULTS, **top)
    out = {}
    for n in sorted(models):
        if not n or n in RESERVED or "/" in n or n.startswith("."):
            raise SystemExit(f"train_multi: bad model name {n!r}")
        if "env" in (models[n] or {}):
            raise SystemExit(f"train_multi: model {n!r} sets \"env\" - one process: put it on the lane")
        out[n] = merge(base, models[n] or {})
    sig = {n: shared_sig(c) for n, c in out.items()}
    names = list(out)
    bad = [k for k in sig[names[0]] if any(sig[n][k] != sig[names[0]][k] for n in names[1:])]
    if bad:
        raise SystemExit("train_multi: these keys build the shared Sampler / batch and must be the same for every "
                         "model: " + "; ".join(f"{k}: " + ", ".join(f"{n}={sig[n][k]!r}" for n in names) for k in bad))
    if out[names[0]].get("pack_next"):
        raise SystemExit("train_multi: pack_next is not supported (one Sampler for the lane)")
    return out


def _read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fin(x):
    return x if isinstance(x, (int, float)) and np.isfinite(x) else None


class Run:
    """one model of the lane: config, run dir, weights, optimizer, counters, its own global-RNG stream."""

    def __init__(self, name, idx, cfg, run_dir, lane):
        self.name, self.idx, self.cfg, self.lane = name, idx, cfg, lane
        self.rd = os.path.join(run_dir, name)
        self.ck = os.path.join(self.rd, "ckpt_last.pt")
        self.bp = os.path.join(self.rd, "ckpt_best.pt")
        self.model = self.ema = self.opt = None
        self.step, self.elapsed0 = 0, 0.0
        self.est = {"best": {"val": float("inf"), "step": None}, "bad": 0, "since": 0}
        self.gn_log = None
        self.skipped = self.skipped_run = 0
        self.stop_reason = None
        self.run_loss, self.nlog = {}, 0
        self.frac = 0.0
        self.summary = None
        self.arm = next_arms.arm_of(cfg)
        self.aux = None
        self.axp = os.path.join(self.rd, "ckpt_aux.pt")

    def log(self, row, file=True):
        if file:
            self.mlog.write(json.dumps(row) + "\n"); self.mlog.flush()
        print(json.dumps({"model": self.name, **row}), flush=True)

    def save_last(self):
        atomic_save(dict(model=self.model.state_dict(), ema=self.ema.state_dict(), step=self.step,
                         elapsed=time.time() - self.t_start, cfg=self.cfg, since_best=self.est["since"]), self.ck)
        if self.aux is not None:
            atomic_save(dict(aux=self.aux.state_dict(), step=self.step), self.axp)
        self.last_ck = time.time()

    def done(self, smp, vm, events_cal, t0, interrupted):
        """bundle (from self.ema) + DONE + the lane summary."""
        el = time.time() - self.t_start
        extra = dict(train_hours=self.info["train_hours"], val_hours=self.info["val_hours"],
                     pack_hours=self.info["pack_hours"], params=self.info["params"], steps_done=self.step,
                     train_minutes=round(el / 60, 2), device=self.info["device"], gpu=self.info["gpu"], final_val=vm,
                     lane=self.lane, model_name=self.name)
        if interrupted:
            extra["interrupted"] = True
        write_bundle(self.rd, self.ema.state_dict(), self.cfg, smp, extra, events_cal=events_cal)
        reason = vm.get("stop_reason") or ("steps done" if self.step >= self.cfg["steps"] else "time budget")
        d = dict(step=self.step, minutes=round(el / 60, 2), final_val=vm, params=self.info["params"], lane=self.lane,
                 model=self.name, sweep=self.cfg.get("sweep"), interrupted=interrupted,
                 best_val=_fin(self.est["best"]["val"]), best_step=self.est["best"]["step"], stop_reason=reason,
                 final_phase_s=round(time.time() - t0, 1))
        write_json_atomic(d, os.path.join(self.rd, "DONE"))
        self.summary = {k: d[k] for k in ("step", "best_val", "best_step", "stop_reason", "interrupted", "minutes")}
        self.log(dict(event="model_done", **self.summary, final_phase_s=d["final_phase_s"]), file=False)

    def free(self):
        self.model = self.ema = self.opt = self.aux = None
        self.mlog.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_on(r, b, smp, amp, dev):
    """model r's train.py step on batch b (its RNG stream swapped in). Returns a stop reason or None."""
    cfg = r.cfg
    set_rng_state(r.rng, dev)
    try:
        lr = lr_at(cfg, r.step, r.frac)
        for gq in r.opt.param_groups:
            gq["lr"] = lr
        b = next_arms.treat(b, smp, r.arm)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, parts = next_arms.arm_loss(r.model, b, smp.colw, cfg["loss"], r.arm, r.aux, smp=smp)
        if "_react_frac" in b:
            parts["react_frac"] = b["_react_frac"]
        if "_inject_n" in b:
            parts["inject_n"] = b["_inject_n"]
        r.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(r.model.parameters(), cfg["grad_clip"])
        if r.aux is not None:
            torch.nn.utils.clip_grad_norm_(r.aux.parameters(), cfg["grad_clip"])
        gnf = float(gn)
        if is_spike(cfg, gnf, loss, r.gn_log, r.step):
            r.opt.zero_grad(set_to_none=True)
            r.skipped += 1; r.skipped_run += 1
            if r.skipped_run in (1, 10, 100):
                r.log(dict(event="skip_step", step=r.step, gn=gnf, loss=float(loss), skipped=r.skipped), file=False)
            return "200 consecutive skipped steps" if r.skipped_run >= 200 else None
        r.skipped_run = 0
        r.gn_log = gn_track(r.gn_log, gnf)
        r.opt.step()
        r.step += 1
        ema_update(r.ema, r.model, cfg, r.step)
        for k, v in parts.items():
            r.run_loss[k] = r.run_loss.get(k, 0.0) + float(v)
        r.run_loss["gn"] = r.run_loss.get("gn", 0.0) + float(gn)
        r.nlog += 1
        if r.step % cfg["log_every"] == 0:
            r.log(dict(step=r.step, t=round(time.time() - r.t_start, 1), lr=lr,
                       **{k: round(v / r.nlog, 5) for k, v in r.run_loss.items()}))
            r.run_loss, r.nlog = {}, 0
        if r.step % cfg["eval_every"] == 0:
            vm = val(r.ema, smp, cfg, n_batches=cfg["val_batches"])
            r.log(dict(step=r.step, t=round(time.time() - r.t_start, 1), skipped=r.skipped, **vm))
            vl = float(vm["val_core"])
            improved, why = eval_verdict(cfg, vl, r.step, r.est)
            if improved:                                  # the EMA only: the bundle / final phase never use the raw weights
                atomic_save(dict(ema=r.ema.state_dict(), step=r.step, val=vl, cfg=cfg), r.bp)
            if why:
                print(f"{r.name}: {why}", flush=True)
                return why
        return None
    finally:
        r.rng = rng_state(dev)


def finish(r, smp, dev, stop):
    """train.py's final phase for model r, then its bundle + DONE; frees it."""
    t0 = time.time()
    set_rng_state(r.rng, dev)
    vm, events_cal = final_phase(r.ema, smp, r.cfg, r.est["best"], r.bp, dev, stop, r.stop_reason, r.skipped)
    r.log(dict(step=r.step, t=round(time.time() - r.t_start, 1), final=True, **vm))
    r.save_last()
    if r.cfg.get("write_final_ckpt"):
        atomic_save(dict(model=r.model.state_dict(), ema=r.ema.state_dict(), step=r.step,
                         elapsed=time.time() - r.t_start, cfg=r.cfg), os.path.join(r.rd, "ckpt_final.pt"))
    r.done(smp, vm, events_cal, t0, interrupted=False)
    r.free()


def finish_interrupted(r, smp, dev):
    """SIGTERM: ckpt_last, the BEST EMA (ckpt_best's, else the current one), a quick event calibration, a final row,
    bundle + DONE with interrupted true. Bounded: no extra val passes, no drops."""
    t0 = time.time()
    print(f"SIGTERM: {r.name}: ckpt_last + bundle from the best EMA", flush=True)
    r.save_last()
    from_best = None
    if r.est["best"]["step"] is not None and os.path.exists(r.bp) and r.cfg.get("bundle_from", "best") == "best":
        try:
            _b = torch.load(r.bp, map_location=dev, weights_only=True)
            r.ema.load_state_dict(_b["ema"])
            from_best = int(_b["step"])
        except Exception as e:                            # noqa: BLE001
            print(f"{r.name}: ckpt_best unreadable ({e}) - bundle from the current EMA", flush=True)
    cal = quick_calibration(r.ema, smp, r.cfg)
    vm = dict(best_val_loss_cmp=r.est["best"]["val"], from_best_step=from_best, stop_reason="sigterm",
              skipped_steps=r.skipped, interrupted=True)
    if cal is not None:
        vm["events_cal"] = cal
    r.log(dict(step=r.step, t=round(time.time() - r.t_start, 1), final=True, **vm))
    r.done(smp, vm, cal, t0, interrupted=True)
    r.free()


def main():
    t_proc0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--set", action="append", default=[], help="key=json override (top level = every model)")
    a = ap.parse_args()
    with open(a.config) as f:
        top = json.load(f)
    for kv in a.set:
        k, v = kv.split("=", 1)
        top[k] = json.loads(v)
    cfgs = model_cfgs(top)
    names = list(cfgs)
    lane = os.path.basename(os.path.abspath(a.run_dir))
    os.makedirs(a.run_dir, exist_ok=True)
    summary, todo = {}, []
    for i, n in enumerate(names):
        os.makedirs(os.path.join(a.run_dir, n), exist_ok=True)
        with open(os.path.join(a.run_dir, n, "cfg.json"), "w") as f:
            json.dump(cfgs[n], f, indent=1)
        d = _read_json(os.path.join(a.run_dir, n, "DONE"))
        if d is not None and not d.get("interrupted"):    # finished in an earlier attempt: not trained again
            summary[n] = {k: d.get(k) for k in ("step", "best_val", "best_step", "stop_reason", "interrupted", "minutes")}
            print(f"{n}: already DONE (step {d.get('step')})", flush=True)
        else:
            todo.append((i, n))
    for n, c in cfgs.items():
        if not c.get("use_style", True):                  # ablation: the style input is always the learned null
            c["aug"] = dict(c.get("aug") or {}, p_style_null=1.0)
    c0 = cfgs[names[0]]
    runs = []
    if todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            torch.set_num_threads(int(c0["threads"]))
        else:
            torch.set_num_threads(int(c0.get("cpu_threads", 4)))
        torch.manual_seed(c0["seed"]); np.random.seed(c0["seed"])
        smp = Sampler(resolve_pack(c0["pack"], a.pack_dir), dev, seed=c0["seed"],
                      **sampler_kwargs(c0, a.pack_dir))
        streams = top.get("arm_streams") or {}
        alts = {next_arms.arm_of(cfgs[n])["stream"] for _, n in todo} - {"main"}
        if len(alts) > 1 or (alts and next(iter(alts)) not in streams):
            raise SystemExit(f"train_multi: arm streams {sorted(alts)} need ONE entry in top-level arm_streams")
        alt_dsw = next_arms.stream_weights(smp, streams[next(iter(alts))].get("dataset_weights")) if alts else None
        if alts:
            print(f"arm stream {next(iter(alts))}: {streams[next(iter(alts))]}", flush=True)
        for i, n in todo:
            r = Run(n, i, cfgs[n], a.run_dir, lane)
            complete_model_cfg(r.cfg, smp)
            if r.arm.get("trace"):                         # reaction adapter: partner-trace adapter (unified/react.py)
                from unified.react import norm_info
                r.cfg["model"]["react_trace"] = dict(norm_info(smp.norm["mean"], smp.norm["std"], smp.norm["amean"],
                                                               smp.norm["astd"]), hidden=int(r.arm["trace"]["hidden"]))
            torch.manual_seed(r.cfg["seed"] + i)
            r.model, r.ema, r.opt = build_model(r.cfg, dev)
            if r.cfg.get("init") and not os.path.exists(r.ck):     # reaction adapter: fine-tune from a bundle's weights
                ip = resolve_path(r.cfg["init"], a.pack_dir)
                sd = torch.load(ip, map_location=dev, weights_only=True)
                sd = sd.get("ema", sd) if isinstance(sd, dict) and "ema" in sd else sd
                mk = r.model.load_state_dict(sd, strict=False)
                r.ema.load_state_dict(sd, strict=False)
                bad = [k for k in mk.missing_keys if not k.startswith("react.")]
                if bad or mk.unexpected_keys:
                    raise SystemExit(f"{n}: init {ip}: missing {bad[:5]} unexpected {mk.unexpected_keys[:5]}")
                print(f"{n}: init from {ip} (new: {len(mk.missing_keys)} react params)", flush=True)
                del sd
            if r.arm.get("contrast"):                      # (c): contingency head, own param group
                r.aux = next_arms.ContingencyHead(r.model.cfg["d"], int(r.arm["contrast"]["hidden"])).to(dev)
                r.opt.add_param_group({"params": list(r.aux.parameters())})
                if os.path.exists(r.axp):
                    r.aux.load_state_dict(torch.load(r.axp, map_location=dev, weights_only=True)["aux"])
            r.rng = rng_state(dev)                        # this model's flow-noise stream starts here
            if os.path.exists(r.ck):
                st = torch.load(r.ck, map_location=dev, weights_only=True)
                r.model.load_state_dict(st["model"]); r.ema.load_state_dict(st["ema"])
                r.step, r.elapsed0 = int(st["step"]), float(st["elapsed"])
                r.est["since"] = int(st.get("since_best", 0))
                print(f"{n}: resumed from step {r.step}", flush=True)
                del st
            if os.path.exists(r.bp):
                try:
                    _b = torch.load(r.bp, map_location="cpu", weights_only=True)
                    r.est["best"] = {"val": float(_b["val"]), "step": int(_b["step"])}
                    del _b
                except Exception:                         # noqa: BLE001
                    pass
            r.info = run_info(r.model, smp, dev)
            print(json.dumps(dict(model=n, **r.info)), flush=True)
            r.mlog = open(os.path.join(r.rd, "metrics.jsonl"), "a")
            r.mlog.write(json.dumps(dict(event="start", step=r.step, **r.info)) + "\n"); r.mlog.flush()
            r.model.train()
            runs.append(r)
        amp = dev == "cuda" and c0["amp"] == "bf16"
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        budget = os.environ.get("TRAIN_TIME_BUDGET_S")
        now = time.time()
        for r in runs:
            r.t_start = now - r.elapsed0
            r.last_ck = now
            r.max_s = r.cfg["max_minutes"] * 60
            if budget:                    # stopped at proc start + budget: leave room for every model's final phase
                left = float(budget) - (now - t_proc0) - float(r.cfg.get("end_margin_s", 300 * len(runs)))
                r.max_s = max(min(r.max_s, r.elapsed0 + left), 1e-3)      # no time left: final phase at once
        if budget:
            print(f"time budget {budget}s -> training until t={min(r.max_s for r in runs):.0f}s", flush=True)
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    active = list(runs)
    while active and not stop["flag"]:
        for r in list(active):                            # steps / time budget reached: the final phase now
            if stop["flag"]:
                break
            r.frac = max(r.step / r.cfg["steps"], (time.time() - r.t_start) / r.max_s)
            if r.frac >= 1.0:
                finish(r, smp, dev, stop)
                active.remove(r)
                print(json.dumps(dict(event="active", models=[x.name for x in active])), flush=True)
        if not active or stop["flag"]:
            break
        b = smp.batch(c0["batch"], c0["seq"], c0["chunks"], split="train", aug=c0["aug"])
        b_alt = None
        if any(r.arm["stream"] != "main" for r in active):    # the second batch stream (own dataset weights)
            b_alt = next_arms.draw(smp, c0, alt_dsw)
        saved = False
        for r in list(active):
            if stop["flag"]:
                break
            why = train_on(r, b if r.arm["stream"] == "main" else b_alt, smp, amp, dev)
            if why:
                r.stop_reason = why
                finish(r, smp, dev, stop)
                active.remove(r)
                print(json.dumps(dict(event="active", models=[x.name for x in active])), flush=True)
            elif not saved and time.time() - r.last_ck > r.cfg["ckpt_seconds"]:
                r.save_last()                             # staggered: at most one model per iteration
                saved = True
        del b, b_alt
    if stop["flag"]:
        for r in active:
            finish_interrupted(r, smp, dev)
        active = []
    for r in runs:
        summary[r.name] = r.summary
    write_json_atomic({n: summary.get(n) for n in names}, os.path.join(a.run_dir, "DONE"))
    print(json.dumps(dict(event="lane_done", lane=lane, models=summary)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
