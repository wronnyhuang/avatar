"""Per-100 ms-step latency of the chosen encoder configs on the host CPU (phase-2 audio).
  sliding (Whisper): the real runtime, StreamingSpeechEncoder.step() incl. window assembly + PCA.
  sliding (wav2vec2/HuBERT): one window forward (per-window normalisation) up to the chosen layer.
  kv (all): the ops one KV step runs - front end on the new 100 ms (+receptive context) and every layer for the 5 new
            query frames against 55 cached keys (op-level timing; the incremental KV runtime class is not written).
Records load average with every row."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified import speech_enc as E  # noqa: E402


def stats(ts):
    ts = np.array(ts) * 1e3
    return dict(p50_ms=round(float(np.percentile(ts, 50)), 2), p99_ms=round(float(np.percentile(ts, 99)), 2),
                mean_ms=round(float(ts.mean()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default='[["whisper_tiny",2],["whisper_base",3],["wav2vec2_base",4],["hubert_base",6]]')
    ap.add_argument("--threads", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = C.load_segment(C.streams()[0]["audio_self"], 0, 30 * 16000)
    rows = []
    for th in a.threads:
        torch.set_num_threads(th)
        for enc, layer in json.loads(a.configs):
            m = E.WindowEncoder(enc, max_layer=layer)
            res = dict(enc=enc, layer=layer, threads=th)
            if m.kind == "whisper":
                pca = os.path.join(C.P, "feats", enc, "pca.npz")
                rt = E.StreamingSpeechEncoder(enc, layer, 1.0, 1, pca_path=pca if os.path.exists(pca) else None)
                rt.push(x[: 16000 * 3])
                while rt.step() is not None:
                    pass
                ts = []
                pos = 16000 * 3
                for _ in range(a.n):
                    rt.push(x[pos: pos + 1600]); pos += 1600
                    t = time.perf_counter(); o = rt.step(); ts.append(time.perf_counter() - t)
                    assert o is not None
                res["sliding"] = stats(ts)
            else:
                w = torch.from_numpy(x[:16000])[None]
                for _ in range(3):
                    m(w, layers=[layer])
                ts = []
                for _ in range(max(10, a.n // 3)):
                    t = time.perf_counter(); m(w, layers=[layer]); ts.append(time.perf_counter() - t)
                res["sliding"] = stats(ts)
            # KV step (op-level)
            with torch.inference_mode():
                if m.kind == "whisper":
                    seg = x[: 1600 * 12]
                    def fe():
                        return E.whisper_frontend(m, seg[-1600 * 2:], 0)[0]
                else:
                    seg = x[: 1600 * 12]
                    def fe():
                        return m.w2v_frontend(seg[-3200:], 0, 0)[0]
                h_all = torch.randn(1, 55, m.dim) * 0.3
                hq = h_all[:, -5:]
                mask = torch.ones(5, 55, dtype=torch.bool)
                ts = []
                for _ in range(a.n):
                    t = time.perf_counter()
                    fe()
                    g, gk = hq, h_all
                    for li in range(layer):
                        g = E._layer(m, li, g, gk, mask)
                    ts.append(time.perf_counter() - t)
                res["kv_step"] = stats(ts)
            res["load"] = round(os.getloadavg()[0], 1)
            print(json.dumps(res), flush=True)
            rows.append(res)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
