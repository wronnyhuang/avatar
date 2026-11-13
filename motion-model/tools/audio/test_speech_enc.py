"""Runtime == offline check for the sliding Whisper encoder (phase-2 audio): StreamingSpeechEncoder fed 20 s of a
Seamless wav in random-size chunks must reproduce extract_stream() frame for frame."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified import speech_enc as E  # noqa: E402


def main():
    torch.set_num_threads(4)
    r = C.streams()[0]
    x = C.load_segment(r["audio_self"], 0, 20 * 16000)
    pca = os.path.join(C.P, "feats", "whisper_tiny", "pca.npz")
    rt = E.StreamingSpeechEncoder("whisper_tiny", 2, 1.0, 1, pca_path=pca)
    rng = np.random.default_rng(0)
    got = {}
    i = 0
    while i < len(x):
        n = int(rng.integers(100, 4000))
        rt.push(x[i:i + n]); i += n
        while True:
            o = rt.step()
            if o is None:
                break
            for k, f in zip(*o):
                got[k] = f
    n_frames = 590
    ref = E.extract_stream(rt.model, x, n_frames, 0, 1.0, 1, [2], base=0, proj=lambda l, v: rt.proj(v))[2]
    ks = sorted(k for k in got if k < n_frames)
    d = np.abs(np.stack([got[k] for k in ks]) - ref[ks]).max()
    print("frames", len(ks), "max |runtime - offline|", float(d))
    assert len(ks) == n_frames and d < 1e-3, d
    print("OK")


if __name__ == "__main__":
    main()
