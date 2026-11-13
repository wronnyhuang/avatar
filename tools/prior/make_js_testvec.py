"""Python reference for the JS parity test: mean-mode rollout (argmax component, delta = mu, blink = p > 0.5 with
the JS refractory rule) of the exported weights on real held-out feature sequences (chunk-aware).
Usage: python -I make_js_testvec.py <weights.json> <scratch> <out.json> [--model i]"""
import json, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import Prior

wp, root, outp = sys.argv[1], sys.argv[2], sys.argv[3]
mi = int(sys.argv[sys.argv.index("--model") + 1]) if "--model" in sys.argv else 0
Wall = json.load(open(wp)); W = (Wall.get("models") or [Wall])[mi]
A = W["arch"]; C = A.get("C", 1)
m = Prior(D=A["D"], H=A["H"], K=A["K"], E=A["E"], C=C, nod=bool(A.get("nod")))
sd = m.state_dict()
def T(name, shape):
    v = W[name]
    if isinstance(v, str):
        import base64
        v = np.frombuffer(base64.b64decode(v[7:]), "<f4")
    return torch.tensor(np.array(v, np.float32).reshape(shape))
NO = A["K"] + 2 * A["K"] * A["D"] * C + 1 + (1 if A.get("nod") else 0)
sd["inp.weight"] = T("inp_w", (A["E"], A["nin"])); sd["inp.bias"] = T("inp_b", (A["E"],))
sd["gru.weight_ih_l0"] = T("w_ih", (3 * A["H"], A["E"])); sd["gru.weight_hh_l0"] = T("w_hh", (3 * A["H"], A["H"]))
sd["gru.bias_ih_l0"] = T("b_ih", (3 * A["H"],)); sd["gru.bias_hh_l0"] = T("b_hh", (3 * A["H"],))
sd["out.weight"] = T("out_w", (NO, A["H"])); sd["out.bias"] = T("out_b", (NO,))
m.load_state_dict(sd); m.eval()
D = dict(np.load(os.path.join(root, "dataset.npz")))
ext_all = np.nan_to_num(np.concatenate([D["sv"], D["pv"], D["pf"]], 1).astype(np.float32))
i_s = int(np.argmax([D["sv"][a:b, 3].mean() if b - a > 300 else 0 for a, b in D["seg"]]))
i_l = int(np.argmax([(D["pv"][a:b, 3].mean() + D["pf"][a:b, 0].mean()) if b - a > 300 else 0 for a, b in D["seg"]]))
ext = np.concatenate([ext_all[D["seg"][i_s][0]:D["seg"][i_s][0] + 225], ext_all[D["seg"][i_l][0]:D["seg"][i_l][0] + 225]])
Dd, K = A["D"], A["K"]; dS = np.array(W["norm"]["d_s"], np.float32)
y = np.zeros(Dd, np.float32); clock = 1.0; blinkT = -1; h = None
frames, nets = [], []
queue = []; hist = []; pending = False
with torch.no_grad():
    for t in range(len(ext)):
        hist.append(ext[t]); hist = hist[-C:]
        if not queue:
            e = np.mean(hist, 0)
            x = np.concatenate([e, y, [1.0, clock]]).astype(np.float32)
            nets.append(dict(t=t, x=[float(v) for v in x]))
            z = torch.tanh(m.inp(torch.from_numpy(x)[None, None]))
            o, h = m.gru(z, h); raw = m.out(o)[0, 0].numpy().astype(np.float64)
            nets[-1]["raw"] = [float(v) for v in raw]
            k = int(np.argmax(raw[:K]))
            mu = raw[K + k * Dd * C:K + (k + 1) * Dd * C].reshape(C, Dd)
            queue = [mu[c] for c in range(C)]
            pb = 1 / (1 + np.exp(-raw[K + 2 * K * Dd * C]))
            if pb > 0.5 and (blinkT < 0 or blinkT >= 6):
                pending = True; clock = 0.0
            else:
                clock = min(clock + C / 30 / 4, 1.0)
        dl = queue.pop(0)
        y = np.clip(y + dl * dS, -5, 5).astype(np.float32)
        if pending:
            pending = False; blinkT = 0
        if blinkT >= 0:
            blinkT += 1
            if blinkT >= 8: blinkT = -1
        frames.append([float(v) for v in y])
json.dump(dict(model=mi, ext=ext.tolist(), y=frames, nets=nets), open(outp, "w"))
print("wrote", outp, len(frames), "frames", len(nets), "net steps")

# ---- the reaction model (if packed): raw outputs on the same partner-feature sequence
if "reaction" in Wall:
    from prior_model import RX
    R_ = Wall["reaction"]; Ar = R_["arch"]
    W = R_
    rx = RX(H=Ar["H"]); sdr = rx.state_dict()
    sdr["inp.weight"] = T("inp_w", (32, 14)); sdr["inp.bias"] = T("inp_b", (32,))
    sdr["gru.weight_ih_l0"] = T("w_ih", (3 * Ar["H"], 32)); sdr["gru.weight_hh_l0"] = T("w_hh", (3 * Ar["H"], Ar["H"]))
    sdr["gru.bias_ih_l0"] = T("b_ih", (3 * Ar["H"],)); sdr["gru.bias_hh_l0"] = T("b_hh", (3 * Ar["H"],))
    sdr["out.weight"] = T("out_w", (8, Ar["H"])); sdr["out.bias"] = T("out_b", (8,))
    rx.load_state_dict(sdr); rx.eval()
    xr = np.concatenate([ext[:, 8:17], ext[:, 4:8], ext[:, 3:4]], 1).astype(np.float32)
    with torch.no_grad():
        orx, _ = rx(torch.from_numpy(xr)[None])
    V = json.load(open(outp)); V["rx_in"] = xr.tolist(); V["rx_out"] = orx[0].numpy().tolist()
    json.dump(V, open(outp, "w"))
    print("reaction model vectors added")
