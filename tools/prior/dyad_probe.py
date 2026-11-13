"""Controlled 'reaction' probes of a trained prior: the avatar listens to a partner whose face/voice does one thing
(smiles, nods, stops talking) and we average the sampled response over many runs vs. a control.
Usage: python -I dyad_probe.py <models_dir> <tag> [--n 300] [--out json]"""
import json, sys, os, argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import Prior, OUT
ap = argparse.ArgumentParser(); ap.add_argument("mdir"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--C", type=int, default=6); ap.add_argument("--K", type=int, default=8); ap.add_argument("--out", default=None)
ap.add_argument("--smile-gain", type=float, default=1.0)
a = ap.parse_args(); torch.set_num_threads(2); torch.manual_seed(0)
md = Path(a.mdir); nz = np.load(md / f"{a.tag}_norm.npz"); pos_c, pos_s, d_s = nz["pos_c"], nz["pos_s"], nz["d_s"]
sd = torch.load(md / f"{a.tag}.pt"); nod = sd["out.weight"].shape[0] == a.K + 2 * a.K * 14 * a.C + 2
m = Prior(K=a.K, C=a.C, nod=nod); m.load_state_dict(sd); m.eval()
T = 450; B = a.n; C = a.C


def run(ext, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.zeros(B, 14); h = None; clock = torch.ones(B)
    Y = np.zeros((B, T, 14), np.float32); NOD = np.zeros((B, T), np.float32)
    E = torch.from_numpy(ext)
    with torch.no_grad():
        for an in range(0, T - 1, C):
            e = E[:, max(0, an - C + 1):an + 1].mean(1)
            x = torch.cat([e, y, torch.ones(B, 1), clock[:, None]], 1)[:, None]
            hd, h = m(x, h)
            lg, mu, ls, b = hd[0][:, 0], hd[1][:, 0], hd[2][:, 0], hd[3][:, 0]
            k = torch.multinomial(torch.softmax(lg, -1), 1, generator=g)[:, 0]
            mm = mu[torch.arange(B), k]; ss = ls[torch.arange(B), k].exp()
            d = (mm + ss * torch.randn(B, C * 14, generator=g)).reshape(B, C, 14)
            if nod:
                NOD[:, an + 1] = torch.sigmoid(hd[4][:, 0]).numpy()
            bk = (torch.rand(B, generator=g) < torch.sigmoid(b)).float()
            clock = torch.where(bk > 0, torch.zeros(B), torch.clamp(clock + C / 120, max=1.0))
            for j in range(C):
                t = an + 1 + j
                if t >= T: break
                y = torch.clamp(y + d[:, j] * torch.from_numpy(d_s.astype(np.float32)), -5, 5); Y[:, t] = y.numpy()
    return Y * pos_s + pos_c, NOD


def base():
    ext = np.zeros((B, T, 17), np.float32)
    rng = np.random.default_rng(1)
    # partner talking (level fluctuating like syllables), avatar silent, partner face present, neutral
    lev = 0.5 + 0.3 * np.sin(np.arange(T) * 2 * np.pi * 4.5 / 30)[None] + 0.1 * rng.standard_normal((B, T))
    ext[:, :, 4] = np.clip(lev, 0, 1.2); ext[:, :, 5] = 0.05 * rng.standard_normal((B, T)); ext[:, :, 6] = 1; ext[:, :, 7] = 1
    ext[:, :, 8] = 1; ext[:, :, 9] = 0.05; ext[:, :, 10] = 0.15 * (lev > 0.5); ext[:, :, 11] = 0.05
    return ext


out = {}
t0 = 150                     # event at 5 s
ctrl, nodc = run(base(), 1)
# 1) partner smiles for 3 s
e = base(); e[:, t0:t0 + 90, 9] = 0.6 * a.smile_gain
Ys, _ = run(e, 1)
sm = OUT.index("smile")
resp = (Ys[:, :, sm] - ctrl[:, :, sm]).mean(0)
out["partner_smile_step"] = dict(own_smile_delta_mean=[float(v) for v in resp[::3]], dt=0.1,
                                 peak=float(resp.max()), peak_t=float((np.argmax(resp) - t0) / 30),
                                 t50=float((np.argmax(resp >= 0.5 * resp.max()) - t0) / 30) if resp.max() > 0 else None)
# 2) partner nods (pitch velocity pulse, ~6 deg dip over 0.45 s)
e = base(); k = np.arange(14); dip = 6 * np.sin(np.pi * k / 14)
e[:, t0:t0 + 14, 14] = dip / 10; e[:, t0:t0 + 14, 12] = np.gradient(dip) * 30 / 30
Yn, nodn = run(e, 1)
p = OUT.index("pitch")
out["partner_nod"] = dict(own_pitch_delta=[float(v) for v in (Yn[:, :, p] - ctrl[:, :, p]).mean(0)[::3]],
                          nod_hazard_delta=[float(v) for v in (nodn - nodc).mean(0)[::3]] if nod else None)
# 3) partner stops talking at 5 s (turn end) -> own nod hazard / pitch / gaze
e = base(); e[:, t0:, 4:8] = 0; e[:, t0:, 10] = 0
Yp, nodp = run(e, 1)
out["partner_pause"] = dict(nod_hazard=[float(v) for v in nodp.mean(0)[::3]] if nod else None,
                            nod_hazard_control=[float(v) for v in nodc.mean(0)[::3]] if nod else None,
                            own_pitch_delta=[float(v) for v in (Yp[:, :, p] - ctrl[:, :, p]).mean(0)[::3]])
if nod:
    w = slice(t0, t0 + 45)
    out["partner_pause"]["nod_hazard_ratio_0_1.5s"] = float(nodp[:, w].mean() / max(nodc[:, w].mean(), 1e-9))
    w2 = slice(t0, t0 + 30)
    out["partner_nod"]["nod_hazard_ratio_0_1s"] = float(nodn[:, w2].mean() / max(nodc[:, w2].mean(), 1e-9))
print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else f"[{len(vv)} pts]") for kk, vv in v.items()} for k, v in out.items()}, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(out))
