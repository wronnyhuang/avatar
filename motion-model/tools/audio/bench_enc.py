"""Latency of one streaming step (one window -> 3 frames) per encoder / window / depth / threads (phase-2 audio)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from unified import speech_enc as E

ap = argparse.ArgumentParser()
ap.add_argument("--enc", nargs="+", default=list(E.ENCODERS))
ap.add_argument("--windows", nargs="+", type=float, default=[1.0, 2.0])
ap.add_argument("--layers", default="")          # json {enc: layer}; default = full depth and half depth
ap.add_argument("--threads", nargs="+", type=int, default=[2, 4])
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()
lay = json.loads(a.layers) if a.layers else {}
rows = []
for th in a.threads:
    torch.set_num_threads(th)
    for n in a.enc:
        m = E.WindowEncoder(n)
        Ls = [lay[n]] if n in lay else sorted({m.n_layers, m.n_layers // 2})
        for W in a.windows:
            x = torch.randn(1, int(W * E.SR)) * 0.05
            for L in Ls:
                for _ in range(5):
                    m(x, layers=[L])
                ts = []
                for _ in range(a.n):
                    t = time.perf_counter(); m(x, layers=[L]); ts.append((time.perf_counter() - t) * 1e3)
                ts = np.array(ts)
                r = dict(enc=n, window_s=W, layer=L, threads=th, p50_ms=round(float(np.percentile(ts, 50)), 2),
                         p99_ms=round(float(np.percentile(ts, 99)), 2), mean_ms=round(float(ts.mean()), 2),
                         per_frame_ms=round(float(np.percentile(ts, 50)) / 3, 2), load=os.getloadavg()[0])
                print(json.dumps(r), flush=True); rows.append(r)
json.dump(rows, open(a.out, "w"), indent=1)
