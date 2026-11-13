"""scaling sweep data path: increment packs (p2_sweep_pack.py base/enc), ConcatPack, per-stream encoder ok mask, lean
storage, and the unchanged default Sampler path (vs a pre-sweep data.py given by $PRE_SWEEP_DATA_PY; skipped when unset).

    nice -n 19 python -I -m pytest -q tools/train/test_sweep_data.py
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
torch.set_num_threads(3)
from unified import data as D  # noqa: E402
import fake_v3  # noqa: E402

OLD = os.environ.get("PRE_SWEEP_DATA_PY", "")     # unified/data.py from before the sweep changes (optional)
ENC = "whisper_tiny"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("sweep"))
    q = fake_v3.synth_qpack(os.path.join(d, "syn"), sessions=14, minutes=0.6, seed=3, n_val=2)
    v3 = os.path.join(d, "v3.npz")
    fake_v3.add_v3_members(q, v3, {ENC: 16})
    z = dict(np.load(v3, allow_pickle=False))
    st = z["streams"].copy()
    tr_sess = np.unique(st[st[:, 3] == 0, 6])
    l2l = set(tr_sess[::2].tolist())                      # every other train session -> a 2nd dataset without audio feats
    for i in range(len(st)):
        if int(st[i, 6]) in l2l:
            st[i, 4] = 1
    names = json.loads(str(z["names"]))
    names["datasets"] = [names["datasets"][0], "l2l"]
    z["streams"] = st
    z["names"] = np.array(json.dumps(names))
    z["events_ok"] = np.ones(len(st), np.uint8)
    feats = z.pop("enc_" + ENC)
    np.savez(v3, **z)
    fd = os.path.join(d, "feats", ENC)
    os.makedirs(fd)
    json.dump({"lookahead_frames": 1}, open(os.path.join(fd, "META.json"), "w"))
    rows = []
    for i, r in enumerate(st):
        if r[4] != 0:
            continue
        sid = f"s{i:04d}"
        np.save(os.path.join(fd, sid + ".npy"), feats[r[0]:r[0] + r[1]].astype(np.float16))
        rows.append(dict(stream_id=sid, pack_stream=i, n_frames=int(r[1]), split=["train", "val", "test"][r[3]]))
    sj = os.path.join(d, "streams.jsonl")
    with open(sj, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    out = os.path.join(d, "packs")
    py = [sys.executable, "-I", os.path.join(HERE, "p2_sweep_pack.py")]
    env = dict(os.environ, OMP_NUM_THREADS="3")
    subprocess.run(py + ["base", "--pack", v3, "--out-dir", out, "--name", "t", "--fracs", "25,50,100", "--zstd", "3",
                         "--no-ready"], check=True, env=env)
    subprocess.run(py + ["enc", "--out-dir", out, "--name", "t", "--streams", sj, "--feats-dir", os.path.join(d, "feats"),
                         "--min-coverage", "1.0", "--zstd", "0", "--no-ready", "--ext", os.path.join(d, "ext.npz")],
                   check=True, env=env)
    man = json.load(open(os.path.join(out, "t_sweep.json")))
    entries = [[r["path"], r["path"][:-4] + ".enc.npz"] for r in man["increments"]]
    return dict(d=d, v3=v3, feats=feats, st=st, man=man, entries=entries)


def test_increments_partition_and_nest(built):
    st, man = built["st"], built["man"]
    seen = np.concatenate([np.load(r["path"])["src_streams"] for r in man["increments"]])
    assert sorted(seen.tolist()) == list(range(len(st)))
    first = np.load(man["increments"][0]["path"])
    assert set(np.nonzero(st[:, 3] != 0)[0]) <= set(first["src_streams"].tolist())       # val only in inc25
    for r in man["increments"][1:]:
        assert (st[np.load(r["path"])["src_streams"], 3] == 0).all()
    assert int(first["lean_no_mel"]) == 1 and np.all(first["a_q"][:40] == 0)
    assert all(os.path.exists(r["path"] + ".zst") for r in man["increments"])
    # stratified: every dataset's train frames grow with the fraction
    for ds in (0, 1):
        acc = 0
        for r in man["increments"]:
            s = np.load(r["path"])["src_streams"]
            acc += int(st[s][(st[s, 4] == ds) & (st[s, 3] == 0), 1].sum())
            tot = int(st[(st[:, 4] == ds) & (st[:, 3] == 0), 1].sum())
            assert acc <= tot


def test_concat_equals_source(built):
    st, feats = built["st"], built["feats"]
    z = np.load(built["v3"])
    M0, K0, A0, W0 = D.read_qpack(z)
    cp = D.ConcatPack(built["entries"], log=lambda m: None)
    src = np.concatenate([np.load(e[0])["src_streams"] for e in built["entries"]])
    cs = cp["streams"]
    assert len(cs) == len(st)
    sc, qm = cp[f"enc_{ENC}_scale"], cp[f"enc_{ENC}_qmean"]
    ok = cp[f"enc_{ENC}_ok"]
    for j, i in enumerate(src):
        a0, n = st[i, 0], st[i, 1]
        b0 = cs[j, 0]
        assert cs[j, 1] == n and cs[j, 3] == st[i, 3] and cs[j, 6] == st[i, 6]
        assert src[cs[j, 2]] == st[i, 2]                                  # partner remapped to the same person
        bf = D.BODY_FACE                                   # finger cols: PCA decode in row blocks -> 1 f16 ulp
        np.testing.assert_array_equal(cp["motion"][b0:b0 + n][:, bf], M0[a0:a0 + n][:, bf])
        np.testing.assert_allclose(cp["motion"][b0:b0 + n].astype(np.float32), M0[a0:a0 + n].astype(np.float32),
                                   atol=1e-4, rtol=0)
        np.testing.assert_array_equal(cp["audio"][b0:b0 + n, 40:], A0[a0:a0 + n, 40:])
        np.testing.assert_array_equal(cp["events"][b0:b0 + n], z["events"][a0:a0 + n])
        assert ok[j] == (st[i, 4] == 0)
        if ok[j]:
            x = cp[f"enc_{ENC}"][b0:b0 + n].astype(np.float32) * sc + qm
            err = np.abs(x - feats[a0:a0 + n].astype(np.float32))
            assert (err <= sc / 2 + 1e-3).mean() > 0.999
    ext = np.load(os.path.join(built["d"], "ext.npz"))
    assert ext[f"enc_{ENC}"].shape[0] == len(M0) and ext[f"enc_{ENC}_ok"].sum() == ok.sum()


def _sampler(mod, pack, **kw):
    return mod.Sampler(pack, "cpu", seed=0, log=lambda m: None, style_procs=2, audio_in="enc_" + ENC, enc_shift=3,
                       use_events=True, **kw)


def test_lean_matches_full_storage(built):
    a = _sampler(D, built["entries"])
    b = _sampler(D, built["entries"], lean=True)
    assert b.E.dtype == torch.int8 and b.A.shape[1] == 2
    ba = a.batch(6, 60, 3, split="train", fixed_seed=5)
    bb = b.batch(6, 60, 3, split="train", fixed_seed=5)
    for k in ba:
        if k == "tok":
            continue
        np.testing.assert_allclose(ba[k].float().numpy(), bb[k].float().numpy(), atol=0, rtol=0, err_msg=k)
    lo = D.layout if hasattr(D, "layout") else None
    diff = (ba["tok"] - bb["tok"]).abs()
    assert float(diff.max()) < 5e-3                      # f16 vs f32 dequantization of the int8 features only
    with pytest.raises(ValueError):
        D.Sampler(built["entries"], "cpu", audio_in="mel", log=lambda m: None, style_procs=2)


def test_enc_missing_rules(built):
    s = _sampler(D, built["entries"], lean=True)
    assert s.EOK is not None and not bool(s.EOK.all())
    ex = _sampler(D, built["entries"], lean=True, enc_missing="exclude")
    ids = ex.split_streams("train")
    assert bool(ex.EOK[torch.from_numpy(ids)].all()) and len(ids) < len(s.split_streams("train"))
    assert ex.hours("train") < s.hours("train")
    from unified.model import layout
    lo = layout(s.aud_dim)
    for _ in range(4):
        b = s.batch(8, 60, 2, split="train")
        own = b["tok"][..., lo["self_audio"]][..., :s.k_enc]
        bad = ~s.EOK[b["pick"]]
        if bool(bad.any()):
            np.testing.assert_allclose(own[bad].numpy(), s.esil_n.expand_as(own[bad]).numpy(), atol=1e-6)
            return
    pytest.skip("no stream without features was sampled")


def test_default_path_unchanged(built):
    if not OLD or not os.path.exists(OLD):
        pytest.skip("pre-sweep data.py not found")
    spec = importlib.util.spec_from_file_location("data_old", OLD)
    old = importlib.util.module_from_spec(spec)
    sys.modules["data_old"] = old
    spec.loader.exec_module(old)
    v4 = os.path.join(built["d"], "v4.npz")                 # single file, f16 features, no ok member
    z = dict(np.load(built["v3"]))
    z["enc_" + ENC] = built["feats"]
    M, K, _, _ = D.read_qpack(z)
    z["style"], tables = D.pack_style(M, K, z["streams"], procs=2, log=lambda m: None)
    z["style_tables"] = np.array(json.dumps(tables))
    np.savez(v4, **z)
    a = _sampler(old, v4)
    b = _sampler(D, v4)
    for i in range(3):
        ba = a.batch(5, 60, 3, split="train" if i < 2 else "val", fixed_seed=11 + i)
        bb = b.batch(5, 60, 3, split="train" if i < 2 else "val", fixed_seed=11 + i)
        assert sorted(ba) == sorted(bb)
        for k in ba:
            assert torch.equal(ba[k], bb[k]), k
