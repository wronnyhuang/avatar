"""Pack trained models into app/prior_weights.json (base64 float32 tensors + the sampling defaults).
Usage: python -I pack_weights.py <models_dir> <out.json> <tag>:<role>:<C>:<K>[:<H>] [...] [--defaults json]"""
import json, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import Prior, model_json

md, outp = sys.argv[1], sys.argv[2]
specs = [a for a in sys.argv[3:] if ":" in a and not a.startswith("{")]
rx_tag = sys.argv[sys.argv.index("--reaction") + 1] if "--reaction" in sys.argv else None
defaults = json.loads(sys.argv[sys.argv.index("--defaults") + 1]) if "--defaults" in sys.argv else {}
models = []
for sp in specs:
    f = sp.split(":"); tag, role, C, K = f[0], f[1], int(f[2]), int(f[3]); H = int(f[4]) if len(f) > 4 else 128
    sd = torch.load(os.path.join(md, f"{tag}.pt"))
    nod = sd["out.weight"].shape[0] == K + 2 * K * 14 * C + 2
    m = Prior(H=H, K=K, C=C, nod=nod); m.load_state_dict(sd); m.eval()
    nz = dict(np.load(os.path.join(md, f"{tag}_norm.npz")))
    nz = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in nz.items()}
    info = json.load(open(os.path.join(md, f"{tag}_train.json")))
    meta = dict(tag=tag, role=role, val_nll_per_frame=min(l["val_nll"] for l in info["log"]),
                val_blink_bce=info["log"][-1]["val_blink_bce"], subset=info.get("subset"))
    models.append(model_json(m, nz, meta))
W = dict(format="prior-v2", rate_hz=30, inputs=["sv_level", "sv_semi", "sv_voiced", "sv_speaking", "pv_level", "pv_semi",
         "pv_voiced", "pv_speaking", "pf_present", "pf_smile", "pf_jaw", "pf_brow", "pf_pitch_vel", "pf_yaw_vel", "pf_pitch",
         "pf_yaw", "pf_arm"], defaults=defaults, models=models)
if rx_tag:
    W["reaction"] = json.load(open(os.path.join(md, f"{rx_tag}_weights.json")))
    W["reaction"]["meta"]["role"] = "reaction"
json.dump(W, open(outp, "w"), separators=(",", ":"))
print(outp, os.path.getsize(outp), "bytes;", [m["meta"] for m in models])
