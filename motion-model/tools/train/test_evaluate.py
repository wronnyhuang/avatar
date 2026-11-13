"""evaluate.py --pack plumbing with FAKE planners (in-process, no pool): job cutting from a fake v3 pack, the enc
feature contract (user_feat = feats_user[t - shift], None while t < shift; self_feat_ahead = feats_self[t + L]),
the old 4-argument step() for mel bundles, the --drop conditions, fired / label event rates and the scorecard.

    python -I -m pytest -q tools/train/test_evaluate.py
    EVAL_E2E_BUNDLE=<old-style bundle dir> python -I -m pytest -q tools/train/test_evaluate.py   (+ a real CLI run)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import evaluate as ev  # noqa: E402
import fake_v3  # noqa: E402
from unified.data import read_qpack  # noqa: E402

ENC, D, SHIFT, L = "whisper_tiny", 48, 2, 3


class FakeEnc:
    """the phase-2 planner contract (enc bundle)."""
    self_audio_lookahead = L
    audio_in = "enc_" + ENC
    feat_dim = D
    user_feat_shift = SHIFT

    def __init__(self):
        self.calls = []

    def step(self, user_motion189=None, user_mask9=None, user_audio42=None, self_audio42_ahead=None, user_feat=None,
             self_feat_ahead=None):
        self.calls.append(dict(um=user_motion189, uk=user_mask9, ua=user_audio42, sa=self_audio42_ahead,
                               uf=user_feat, sf=self_feat_ahead))
        m = np.zeros(189, np.float32)
        if user_motion189 is not None:
            m += 0.5 * np.asarray(user_motion189, np.float32)
        if self_audio42_ahead is not None:
            m[:6] += 0.01 * float(np.asarray(self_audio42_ahead)[41])
        n = len(self.calls)
        return {"motion": m, "owned": np.zeros(9, np.float32), "plan_ms": 1.0 + n % 3,
                "events": (["nod"] if n % 30 == 0 else []) + (["blink"] if n % 45 == 0 else []), "event_p": {}}


class FakeOld:
    """a pre-phase-2 runtime: 4-argument step(), no events."""
    self_audio_lookahead = 6

    def step(self, user_motion189=None, user_mask9=None, user_audio42=None, self_audio42_ahead=None):
        m = np.zeros(189, np.float32) if user_motion189 is None else 0.5 * np.asarray(user_motion189, np.float32)
        return {"motion": m, "owned": np.zeros(9, np.float32), "plan_ms": None}


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    d = tmp_path_factory.mktemp("ev")
    q = fake_v3.synth_qpack(str(d), sessions=8, minutes=0.8, n_val=3)
    out = str(d / "v3.npz")
    fake_v3.add_v3_members(q, out, {ENC: D})
    z = np.load(out)
    M, K, A, _ = read_qpack(z)
    return dict(path=out, dir=d, M=M, K=K, A=A, F=z["enc_" + ENC], EV=z["events"], st=z["streams"])


def _bundle(d, audio):
    os.makedirs(d, exist_ok=True)
    cfg = {"kind": "flow"}
    if audio:
        cfg["audio"] = audio
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return str(d)


def _args(pack, bundle, **kw):
    a = dict(bundle=bundle, pack=pack, split="val", seconds=10.0, max_sessions=2, datasets=None, style=None,
             style_sweep=0, no_ablate=False, drop="user_audio,user_face,self_speech", long_minutes=0)
    a.update(kw)
    return argparse.Namespace(**a)


def test_stats_event_rates():
    T = 60 * ev.FPS
    m = np.zeros((T, 189), np.float32)
    sm = np.zeros(T)
    for c in (100, 400, 420, 1000):                        # 420 is < 1 s after 400 -> not a new smile
        sm[c:c + 20] = 0.8
    m[:, ev.SM] = sm[:, None]
    lab = np.zeros((T, 3), np.uint8)
    lab[[10, 50, 90], 0] = 1                               # single-frame onsets
    lab[200:210, 1] = 1; lab[300:305, 1] = 1               # spans -> rising edges
    s = ev.stats(m, fired={"nod": 7, "blink": 0, "smile": 2}, labels=lab)
    assert s["smiles_per_min"] == 3 and s["label_nod_per_min"] == 3 and s["label_blink_per_min"] == 2
    assert s["label_smile_per_min"] == 0 and s["fired_nod_per_min"] == 7 and s["fired_smile_per_min"] == 2
    m[:, ev.SM] = 0.9                                      # smiling all along: no onset at the window edge
    assert ev.stats(m)["smiles_per_min"] == 0


def test_pack_jobs_and_enc_contract(pack, tmp_path):
    b = _bundle(tmp_path / "b_enc", {"in": "enc_" + ENC, "feat_dim": D, "user_shift": SHIFT})
    jobs, n, meta = ev.pack_jobs(_args(pack["path"], b), {}, None)
    st, M, K, A, F, EVL = pack["st"], pack["M"], pack["K"], pack["A"], pack["F"], pack["EV"]
    T = 10 * ev.FPS
    assert meta["enc"] and meta["audio_in"] == "enc_" + ENC and meta["events_in_pack"] and n == meta["dyads"] >= 1
    conds = sorted({j["cond"] for j in jobs})
    assert conds == ["drop:self_speech", "drop:user_audio", "drop:user_face", "normal", "shuffle", "zero"]
    assert len(jobs) == n * 2 * 6
    for j in jobs:
        me, you = j["self_stream"], j["user_stream"]
        assert st[me, 3] == 1 and st[me, 5] == 0 and st[you, 5] == 0                  # val, non-ghost
        r0, u0 = st[me, 0] + j["t0"], st[you, 0] + j["user_t0"]
        assert np.array_equal(j["sa_rows"], A[r0:r0 + len(j["sa_rows"])]) and len(j["sa_rows"]) > T + L
        assert np.array_equal(j["fs"], F[r0:r0 + len(j["fs"])])
        nu = min(T, st[you, 1] - j["user_t0"])
        assert np.array_equal(j["um"][:nu], M[u0:u0 + nu]) and np.array_equal(j["fu"][:nu], F[u0:u0 + nu])
        if j["cond"] != "shuffle":
            assert st[you, 2] == me and st[you, 6] == st[me, 6] and j["user_t0"] == j["t0"]   # the partner
        else:
            assert st[you, 6] != st[me, 6]
        if j["cond"] == "normal":
            assert np.array_equal(j["real"], M[r0:r0 + T]) and np.array_equal(j["labels"], EVL[r0:r0 + T])
            assert np.array_equal(j["listen"], A[r0:r0 + T, 41].astype(np.float32) < 0.5)
    # the stepping contract, per condition
    by = {j["cond"]: j for j in jobs if j["key"] == jobs[0]["key"]}
    f32 = lambda x: np.asarray(x, np.float32)                                         # noqa: E731
    for cond, j in by.items():
        p = FakeEnc()
        gen, ms, fired = ev.simulate(p, j, f32(j["um"]), j["uk"], f32(j["ua"]), f32(j["sa_rows"]), j["fu"], j["fs"])
        assert gen.shape == (T, 189) and len(p.calls) == T and fired["nod"] == T // 30 and fired["blink"] == T // 45
        r0, u0 = st[j["self_stream"], 0] + j["t0"], st[j["user_stream"], 0] + j["user_t0"]
        for k, c in enumerate(p.calls):
            zero, dr = cond == "zero", cond[5:] if cond.startswith("drop:") else None
            if zero or dr == "user_audio" or k < SHIFT:
                assert c["uf"] is None
            else:
                assert c["uf"].dtype == np.float32 and np.array_equal(c["uf"], f32(F[u0 + k - SHIFT]))
            if dr == "self_speech":
                assert c["sa"] is None and c["sf"] is None
            else:
                assert np.array_equal(c["sa"], f32(A[r0 + k + L])) and np.array_equal(c["sf"], f32(F[r0 + k + L]))
            if zero:
                assert c["um"] is None and c["uk"] is None and c["ua"] is None
            elif dr == "user_audio":
                assert c["ua"] is None and c["um"] is not None
            else:
                assert np.array_equal(c["ua"], f32(A[u0 + k]))
            if dr == "user_face":
                assert not np.asarray(c["um"])[ev.FACE_DROP_COLS].any()
                assert not np.asarray(c["uk"])[ev.FACE_DROP_GROUPS].any()
                keep = np.setdiff1d(np.arange(189), ev.FACE_DROP_COLS)
                assert np.array_equal(np.asarray(c["um"])[keep], f32(M[u0 + k])[keep])
            elif not zero:
                assert np.array_equal(c["um"], f32(M[u0 + k])) and np.array_equal(c["uk"], K[u0 + k])


def test_old_bundle_summary_and_scorecard(pack, tmp_path, monkeypatch):
    b = _bundle(tmp_path / "b_old", None)
    jobs, n, meta = ev.pack_jobs(_args(pack["path"], b, style_sweep=1, long_minutes=0.5), {}, None)
    assert not meta["enc"] and all(j.get("fu") is None and j.get("fs") is None for j in jobs)
    monkeypatch.setattr(ev, "_planner", lambda job: FakeOld())
    res = [ev.run_job(j) for j in jobs]                                                # 4-arg step() only
    assert all(r["fired"] is None for r in res)
    long = [r for r in res if r["cond"] == "long"]
    assert len(long) == 1 and len(long[0]["gen"]) == 30 * ev.FPS
    out = ev.summarize([r for r in res if r["cond"] != "long"], long[0])
    sc = out["scorecard"]
    for k in ev.SC_KEYS:
        assert f"{k}_gen" in sc and f"{k}_real" in sc
    for k in ("smile_coupling_gain_zero", "sens_zero", "sens_shuffle", "mean_abs_log_ratio_avg", "plan_ms_p50",
              "plan_ms_p99", "label_nod_per_min_real", "label_blink_per_min_real", "label_smile_per_min_real",
              "sens_drop_user_audio", "sens_drop_user_face", "sens_drop_self_speech"):
        assert k in sc, k
    assert not [k for k in sc if k.startswith("fired_")]
    assert out["drops"]["user_face"]["sensitivity"] > 0.01                            # the fake mirrors the face
    assert abs(out["drops"]["user_audio"]["sensitivity"]) < 1e-9                      # ... and ignores audio
    d = out["drops"]["user_face"]
    assert set(d) == {"n", "gen", "delta_vs_normal", "sensitivity"} and d["n"] == 2 * n
    assert "style_sweep" in out and out["long_rollout"]["finite"]
    json.dumps(out)


def test_enc_bundle_fired_rates(pack, tmp_path, monkeypatch):
    b = _bundle(tmp_path / "b_enc2", {"in": "enc_" + ENC, "feat_dim": D, "user_shift": SHIFT})
    jobs, n, meta = ev.pack_jobs(_args(pack["path"], b, drop="", no_ablate=True), {}, None)
    monkeypatch.setattr(ev, "_planner", lambda job: FakeEnc())
    out = ev.summarize([ev.run_job(j) for j in jobs])
    assert out["scorecard"]["fired_nod_per_min_gen"] == pytest.approx(60.0)          # 1 per 30 frames = 60/min
    assert out["scorecard"]["fired_blink_per_min_gen"] == pytest.approx(36.0)        # 6 per 10 s window
    assert "drops" not in out and "sens_drop_user_face" not in out["scorecard"]


def test_enc_errors(pack, tmp_path):
    b = _bundle(tmp_path / "b_bad", {"in": "enc_hubert_base"})
    with pytest.raises(SystemExit, match="not in the pack"):
        ev.pack_jobs(_args(pack["path"], b), {}, None)
    b = _bundle(tmp_path / "b_dim", {"in": "enc_" + ENC, "feat_dim": D + 1})
    with pytest.raises(SystemExit, match="feat_dim"):
        ev.pack_jobs(_args(pack["path"], b), {}, None)


@pytest.mark.skipif(not os.environ.get("EVAL_E2E_BUNDLE"), reason="set EVAL_E2E_BUNDLE=<old-style bundle dir>")
def test_cli_pack_mode_real_runtime(pack, tmp_path):
    out = tmp_path / "e.json"
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "evaluate.py"), "--bundle", os.environ["EVAL_E2E_BUNDLE"],
                    "--pack", pack["path"], "--split", "val", "--seconds", "8", "--max-sessions", "1", "--workers", "4",
                    "--threads", "1", "--drop", "user_face", "--out", str(out)], check=True, capture_output=True)
    o = json.loads(out.read_text())
    assert o["mode"] == "pack" and o["n"] == 2 and "sens_drop_user_face" in o["scorecard"]


def test_single_dyad_shuffle_is_a_stranger(pack, tmp_path):
    b = _bundle(tmp_path / "b_one", None)
    jobs, n, _ = ev.pack_jobs(_args(pack["path"], b, max_sessions=1, drop=""), {}, None)
    st = pack["st"]
    sh = [j for j in jobs if j["cond"] == "shuffle"]
    assert n == 1 and len(sh) == 2 and all(st[j["user_stream"], 6] != st[j["self_stream"], 6] for j in sh)
