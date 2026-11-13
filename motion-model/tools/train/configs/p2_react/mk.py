"""Reaction fine-tune: multi.py config multi_rx.json (4 s40 arms fine-tuned from the HRC bundle, lockstep, d50 + CANDOR pack with CANDOR
weight 0 as in HRC's main stream) and the CPU smoke config.
  R0 HRC continued unchanged | R1 + partner trace adapter + reaction-rate matching | R2 R1 + reaction targets x8
  | R3 R2 + cartoon reaction injection (train batches only)."""
import copy, json, os
H = os.path.dirname(os.path.abspath(__file__))
src = json.load(open(os.path.join(H, "..", "p2_next", "multi_nx.json")))
run = copy.deepcopy(src["runs"]["nx"])
hrc = run["models"]["HRC"]
R2W = {"flow_w": 3.0, "mimic_w": 4.0, "event_w": 7.0, "radius": [12, 4, 20], "speech": False}
arms = {"R0": {"hist": True, "react": True, "contrast": True},
        "R1": {"hist": True, "react": True, "contrast": True, "trace": True, "rate": True},
        "R2": {"hist": True, "react": R2W, "contrast": True, "trace": True, "rate": True},
        "R3": {"hist": True, "react": R2W, "contrast": True, "trace": True, "rate": True, "inject": {"f_smile": 0.06, "f_nod": 0.10}}}
models = {}
for n, a in arms.items():
    m = copy.deepcopy(hrc)
    m["sweep"] = dict(m["sweep"], lane="rx", arm=n)
    m.update(arm=a, init="hrc_init.pt", lr=1e-4, warmup=300, steps=6000, patience_evals=None, bundle_from="last")
    models[n] = m
run.update(models=models, max_minutes=170, end_margin_s=720,
           needs=run["needs"] + ["hrc_init.pt"])
run.pop("arm_streams", None)
mc = dict(src, runs={"rx": run}, batches=[["rx"]])
json.dump(mc, open(os.path.join(H, "multi_rx.json"), "w"), indent=1)
lane = dict(copy.deepcopy(src["common"]), **{k: v for k, v in run.items() if k not in ("needs", "env")})
for n in lane["models"]:
    lane["models"][n]["warmup"] = 5
    lane["models"][n]["init"] = os.environ.get("SMOKE_INIT", "hrc_init.pt")
for n in lane["models"]:
    lane["models"][n]["steps"] = 8
lane.update(steps=8, eval_every=4, log_every=2, val_batches=1, batch=8, ckpt_seconds=5, max_minutes=60, threads=16,
            style_procs=4)
json.dump(lane, open(os.path.join(H, "smoke_rx.json"), "w"), indent=1)
print("wrote multi_rx.json smoke_rx.json")
