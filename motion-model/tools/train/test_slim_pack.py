"""slim_pack.py on a synthetic v3 pack (synth.py -> qpack -> fake_v3 members).

    python -I -m pytest -q tools/train/test_slim_pack.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_v3  # noqa: E402
import slim_pack  # noqa: E402


@pytest.fixture(scope="module")
def v3(tmp_path_factory):
    d = tmp_path_factory.mktemp("v3")
    q = fake_v3.synth_qpack(str(d), sessions=8, minutes=0.8)
    out = str(d / "v3.npz")
    fake_v3.add_v3_members(q, out, {"whisper_tiny": 96, "hubert_base": 160})
    return out


def test_slim_pca_int8(v3, tmp_path):
    out = str(tmp_path / "slim.npz")
    res = slim_pack.slim(v3, out, pca_dims=8, log=lambda m: None)
    z0, z = np.load(v3), np.load(out)
    base = [k for k in z0.files if k not in ("events", "a2f", "a2f_mask") and not k.startswith("enc_")]
    for k in base:                                                     # unchanged, byte for byte
        assert z[k].dtype == z0[k].dtype and np.array_equal(z[k], z0[k]), k
    assert "a2f" not in z.files and np.array_equal(z["events"], z0["events"])
    assert z["a2f_u8"].dtype == np.uint8 and z["a2f_mask"].dtype == np.uint8
    assert np.abs(z["a2f_u8"] / 255.0 - np.clip(z0["a2f"].astype(np.float32), 0, 1)).max() <= 0.5 / 255 + 1e-3
    assert np.array_equal(z["a2f_mask"], z0["a2f_mask"] != 0)
    st = z0["streams"]
    for e, D in (("whisper_tiny", 96), ("hubert_base", 160)):
        q, sc, qm = z[f"enc_{e}"], z[f"enc_{e}_scale"], z[f"enc_{e}_qmean"]
        mu, V, ev = z[f"enc_{e}_pca_mean"], z[f"enc_{e}_pca_V"], float(z[f"enc_{e}_pca_ev"])
        assert q.dtype == np.int8 and q.shape == (len(z0[f"enc_{e}"]), 8)
        assert sc.shape == qm.shape == (8,) and mu.shape == (D,) and V.shape == (8, D)
        assert sc.dtype == qm.dtype == mu.dtype == V.dtype == np.float32
        assert 0.9 < ev <= 1.0 and abs(ev - res["encoders"][e]["explained_variance"]) < 1e-3   # low-rank fake
        assert np.allclose(V @ V.T, np.eye(8), atol=1e-4)               # orthonormal rows
        f = z0[f"enc_{e}"].astype(np.float32)
        proj = (f - mu) @ V.T
        deq = q.astype(np.float32) * sc + qm
        inside = np.abs(proj) < 127 * sc                                 # not clipped
        assert inside.mean() > 0.99
        assert (np.abs(deq - proj)[inside] <= (sc / 2 + 1e-3 * np.abs(proj).max())[None].repeat(len(q), 0)[inside]).all()
        rel = np.abs(deq - proj).mean() / np.abs(proj).mean()
        assert rel < 0.03, rel
        # fitted on TRAIN non-ghost frames only: the train projection is centred with ~unit (4 sigma = 127) scale
        tr = np.concatenate([np.arange(s0, s0 + n) for s0, n, _, sp, _, gh, _ in st if sp == 0 and gh == 0])
        assert np.abs(proj[tr].mean(0)).max() < 0.05 * proj[tr].std(0).max()
        assert np.allclose(proj[tr].std(0) * 4 / 127, sc, rtol=0.02)
    assert os.path.getsize(out) < 0.5 * os.path.getsize(v3)


def test_slim_options(v3, tmp_path):
    out = str(tmp_path / "s0.npz")
    slim_pack.slim(v3, out, encoders=["whisper_tiny"], pca_dims=0, no_a2f=True, log=lambda m: None)
    z0, z = np.load(v3), np.load(out)
    assert not ({"a2f", "a2f_u8", "a2f_mask", "enc_hubert_base"} & set(z.files))
    assert z["enc_whisper_tiny"].dtype == np.float16 and np.array_equal(z["enc_whisper_tiny"], z0["enc_whisper_tiny"])
    assert not [k for k in z.files if k.startswith("enc_whisper_tiny_")]
    with pytest.raises(SystemExit):
        slim_pack.slim(v3, str(tmp_path / "x.npz"), encoders=["wav2vec2_base"], log=lambda m: None)


@pytest.mark.skipif(not shutil.which("zstd"), reason="no zstd")
def test_cli_zst_in_and_out(v3, tmp_path):
    zin = str(tmp_path / "v3.npz.zst")
    subprocess.run(["zstd", "-q", "-f", v3, "-o", zin], check=True)
    out = str(tmp_path / "o" / "slim.npz")
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "slim_pack.py"), zin, out, "--pca-dims", "4",
                        "--encoders", "hubert_base", "--zstd"], check=True, capture_output=True, text=True)
    assert "explains" in p.stdout and "total" in p.stdout
    assert os.path.exists(out + ".zst") and not os.path.exists(out + ".in.tmp.npz")
    z = np.load(out)
    assert z["enc_hubert_base"].shape[1] == 4 and "enc_whisper_tiny" not in z.files
