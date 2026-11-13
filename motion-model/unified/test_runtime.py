"""Tests for the runtime API (works on the copy-delay stub and, given UNIFIED_BUNDLE=<dir>, on a real bundle).
The test_p2_* tests build tiny synthetic p2 bundles (speech-encoder audio with PCA, event head + curves, regression
head) in a tmp dir and always run.

    venv/bin/python -I -m pytest -q unified/test_runtime.py
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from canon1 import spec  # noqa: E402
from unified.runtime import GROUP_COLS, GI, REST_POSE, UnifiedPlanner  # noqa: E402

BUNDLE = os.environ.get("UNIFIED_BUNDLE") or None


def _user(k, rng):
    m = REST_POSE.copy()
    m[spec.COL["head_y"]] = 0.3 * np.sin(k / 15)
    m[spec.COL["shoulder_r_z"]] = 1.2 - 0.8 * (k > 40)
    m[spec.SL_FACE] = np.clip(0.2 + 0.05 * rng.standard_normal(52), 0, 1)
    return m.astype(np.float32)


def _run(p, n, rng, user=True):
    outs = []
    for k in range(n):
        um = _user(k, rng) if user else None
        o = p.step(um, np.ones(9) if user else None, np.zeros(42, np.float32), None)
        outs.append(o)
    return outs


def test_shapes_and_plan_cadence():
    p = UnifiedPlanner(BUNDLE, replan_every=3)
    rng = np.random.default_rng(0)
    outs = _run(p, 30, rng)
    for o in outs:
        assert o["motion"].shape == (189,) and o["motion"].dtype == np.float32
        assert o["owned"].shape == (9,)
        assert np.isfinite(o["motion"]).all()
    planned = [i for i, o in enumerate(outs) if o["plan_ms"] is not None]
    assert planned[:4] == [0, 3, 6, 9], planned
    assert p.self_audio_lookahead >= 0
    if p.n_events == 0:
        assert all(o["events"] == [] and o["event_p"] is None for o in outs)
    if BUNDLE is None or p.audio_in == "mel":
        assert p.audio_in == "mel" and p.feat_dim == 0 and p.user_feat_shift == 0


def test_no_user():
    p = UnifiedPlanner(BUNDLE)
    outs = _run(p, 20, np.random.default_rng(1), user=False)
    assert all(np.isfinite(o["motion"]).all() for o in outs)


def test_copy_delay_mirrors_user():
    if BUNDLE is not None:
        return
    p = UnifiedPlanner(None)
    rng = np.random.default_rng(2)
    outs = _run(p, 80, rng)
    d = p.core.delay
    k = 70
    want = spec.mirror_motion(_user(k - d, np.random.default_rng(2)))
    got = outs[k]["motion"]
    # head yaw mirrored (sign flip), right shoulder -> left shoulder
    assert abs(got[spec.COL["head_y"]] - want[spec.COL["head_y"]]) < 1e-5
    assert abs(got[spec.COL["shoulder_l_z"]] - want[spec.COL["shoulder_l_z"]]) < 1e-5


def test_override_fade_hold_release():
    p = UnifiedPlanner(BUNDLE)
    rng = np.random.default_rng(3)
    _run(p, 20, rng)
    pose = REST_POSE.copy()
    pose[GROUP_COLS[GI["arm_r"]]] = 0.7
    p.set_override(["arm_r", "fingers_r"], values=pose, fade_in_s=0.1)   # 3 frames
    outs = _run(p, 10, rng)
    w = [o["owned"][GI["arm_r"]] for o in outs]
    assert 0 < w[0] < 1 and w[2] == 1.0 and w[-1] == 1.0, w
    assert outs[-1]["owned"][GI["fingers_r"]] == 1.0 and outs[-1]["owned"][GI["head"]] == 0.0
    np.testing.assert_allclose(outs[-1]["motion"][GROUP_COLS[GI["arm_r"]]], 0.7, atol=1e-6)
    p.release_override(fade_out_s=0.2)   # 6 frames
    outs = _run(p, 12, rng)
    w = [o["owned"][GI["arm_r"]] for o in outs]
    assert w[0] < 1 and w[5] == 0 and w[-1] == 0, w
    assert outs[0]["plan_ms"] is not None          # release forces a re-plan


def test_override_callable_and_none():
    p = UnifiedPlanner(BUNDLE)
    rng = np.random.default_rng(4)
    _run(p, 10, rng)
    calls = []

    def fn(k):
        calls.append(k)
        v = REST_POSE.copy()
        v[GROUP_COLS[GI["arm_l"]]] = 0.01 * k
        return v

    p.set_override("arm_l", values=fn, fade_in_s=0.0)
    outs = _run(p, 5, rng)
    assert outs[0]["owned"][GI["arm_l"]] == 1.0
    np.testing.assert_allclose(outs[4]["motion"][GROUP_COLS[GI["arm_l"]]], 0.04, atol=1e-6)
    assert max(calls) > 4                           # the planner looked ahead
    p.set_override(["torso"], values=None, fade_in_s=0.0)
    held = p.hist.self_motion.last()[GROUP_COLS[GI["torso"]]].copy()
    outs = _run(p, 5, rng)
    np.testing.assert_allclose(outs[-1]["motion"][GROUP_COLS[GI["torso"]]], held, atol=1e-6)


def test_reset():
    p = UnifiedPlanner(BUNDLE)
    rng = np.random.default_rng(5)
    _run(p, 10, rng)
    p.set_override("head", values=REST_POSE)
    p.reset()
    o = p.step(None, None, np.zeros(42), None)
    assert o["plan_ms"] is not None and o["owned"].sum() == 0 and p.t == 1


def test_override_no_jump_on_set_and_release():
    """the owned group's output must ramp (no jump to the action pose on the first frame) and ramp back."""
    p = UnifiedPlanner(BUNDLE)
    rng = np.random.default_rng(6)
    outs = _run(p, 30, rng)
    c = GROUP_COLS[GI["arm_r"]]
    pose = REST_POSE.copy()
    pose[c] = outs[-1]["motion"][c] + 0.9                 # far away
    p.set_override(["arm_r"], values=pose, fade_in_s=0.3)  # 9 frames
    o2 = _run(p, 20, rng)
    seq = np.stack([outs[-1]["motion"][c]] + [o["motion"][c] for o in o2])
    step = np.abs(np.diff(seq, axis=0)).max(1)
    total = np.abs(pose[c] - outs[-1]["motion"][c]).max()
    assert step[0] < 0.5 * total, (step[:10], total)
    assert np.abs(o2[-1]["motion"][c] - pose[c]).max() < 1e-5
    p.release_override(fade_out_s=0.3)
    o3 = _run(p, 20, rng)
    seq = np.stack([o2[-1]["motion"][c]] + [o["motion"][c] for o in o3])
    assert np.abs(np.diff(seq, axis=0)).max(1)[0] < 0.5 * total


def test_slerp_continuous_past_pi():
    """a joint whose pose has |aa| > pi (deep elbow bend + an action offset) must not flip representation during
    an override ramp (the old slerp returned the th <= pi twin: a 5.7 jump in the canon columns at release)."""
    from unified.runtime import _aa_to_quat, slerp_aa
    a = np.array([[0.43, 2.15, -0.56]])                     # model elbow (|aa| 2.26)
    b = a + 0.9                                             # action pose (|aa| 3.34 > pi)
    seq = [a[0]] + [slerp_aa(a, b, w)[0] for w in np.linspace(0.1, 0.9, 9)] + [b[0]]
    steps = np.abs(np.diff(np.stack(seq), axis=0)).max(1)
    assert steps.max() < 0.3, steps
    q1, q2 = _aa_to_quat(slerp_aa(a, b, 0.999)), _aa_to_quat(b)
    assert abs(abs(float(np.sum(q1 * q2))) - 1) < 1e-5     # the same rotation as the endpoint


# ================================================================================================ p2 synthetic bundles
NOD_C = spec.COL["head_x"]
BLINK_C = [135 + spec.BS["eyeBlinkLeft"], 135 + spec.BS["eyeBlinkRight"]]
SMILE_C = [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]
EV_LENS = [15, 8, 20]
EV_REFR = [20, 10, 30]


def event_curves():
    """nod = a dip on head_x, blink = close-open on eyeBlinkLeft/Right, smile = a rise on mouthSmileLeft/Right."""
    cur = np.zeros((3, 24, 189), np.float32)
    k = np.arange(EV_LENS[0])
    cur[0, k, NOD_C] = -0.15 * np.sin(np.pi * k / (EV_LENS[0] - 1))
    cur[1, :EV_LENS[1], BLINK_C[0]] = cur[1, :EV_LENS[1], BLINK_C[1]] = [0.3, 0.7, 0.9, 0.9, 0.6, 0.35, 0.15, 0.05]
    k = np.arange(EV_LENS[2])
    for c in SMILE_C:
        cur[2, k, c] = 0.35 * np.sin(np.pi * k / (EV_LENS[2] - 1))
    return cur


def make_p2_bundle(out, De=24, k=8, n_events=3, head_kind="flow", audio="enc", ev_bias=None, ev_runtime=None,
                   model=None, seed=0, user_shift=2, encoder=None):
    """a random-init p2 bundle (weights.pt, norm.npz, events.npz, config.json) in `out`; returns out.
    encoder = a speech-encoder spec (META) for config["audio"]["encoder"] (PCM mode)."""
    import torch
    from unified.model import UnifiedModel
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A = k + 2 if audio == "enc" else 42
    mc = dict(d=32, layers=2, heads=2, ffn=2, window=16, hd=32, hlayers=1, hheads=2, hffn=2, horizon=15, patch=3,
              mem=8, lookahead=6, aud_dim=A, n_events=n_events, head_kind=head_kind)
    mc.update(model or {})
    m = UnifiedModel(mc)
    with torch.no_grad():                         # zero-initialised output layers -> small random (input-dependent)
        for prm in m.parameters():
            if prm.dim() > 0 and float(prm.abs().max()) == 0.0:
                prm.normal_(0, 0.05)
        if m.events is not None and ev_bias is not None:
            m.events.net[2].bias.fill_(float(ev_bias))
    torch.save(m.state_dict(), os.path.join(out, "weights.pt"))
    mean = np.zeros(189, np.float32); mean[135:187] = 0.2
    std = np.full(189, 0.2, np.float32); std[135:187] = 0.1
    amean = np.r_[np.full(40, -8.0), -50.0, 0.3].astype(np.float32)
    astd = np.r_[np.full(40, 3.0), 20.0, 0.45].astype(np.float32)
    norm = dict(mean=mean, std=std, amean=amean, astd=astd)
    cfg = dict(kind="flow", version="unified-p2-test", model=m.cfg, self_audio_lookahead=mc["lookahead"],
               horizon=mc["horizon"], prefix=2, guidance=1.5, steps=3, temperature=0.8, replan_every=3,
               lip=dict(cols=[160, 162, 167]),
               runtime=dict(replan_every=3, steps=3, guidance=1.5, temperature=0.8, blend=6,
                            events=dict(dict(mode="threshold", gain=1.0, feed_history=True), **(ev_runtime or {}))))
    if audio == "enc":
        V = np.linalg.qr(rng.standard_normal((De, k)))[0].T.astype(np.float32)          # (k, De) orthonormal rows
        pm = rng.standard_normal(De).astype(np.float32) * 0.3
        norm.update(emean=(rng.standard_normal(k) * 0.1).astype(np.float32),
                    estd=(0.5 + rng.random(k)).astype(np.float32),
                    esil=((np.zeros(De, np.float32) - pm) @ V.T).astype(np.float32),
                    epca_mean=pm, epca_V=V)
        cfg["audio"] = {"in": "enc_test", "feat_dim": De, "proj_dim": k, "user_shift": user_shift, "keep_vad": True}
        if encoder is not None:
            cfg["audio"]["encoder"] = dict(encoder)
    if n_events > 0:
        cfg["events"] = dict(names=["nod", "blink", "smile"], thresholds=[0.5, 0.5, 0.5], refractory=EV_REFR,
                             groups=["head", "face_upper", "mouth"], label_rate_per_min=[6.0, 15.0, 2.0])
        np.savez(os.path.join(out, "events.npz"), curves=event_curves(), lens=np.asarray(EV_LENS, np.int64))
    np.savez(os.path.join(out, "norm.npz"), **norm)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f)
    return out


def _run_p2(p, n, seed=0, feats=True, self_feats=True, t0=0):
    """n steps with random user motion / audio and (enc bundles) random features; returns step outputs."""
    r = np.random.default_rng(seed)
    De = p.feat_dim
    outs = []
    for k in range(n):
        um = (REST_POSE + 0.05 * r.standard_normal(189)).astype(np.float32)
        ua = (r.standard_normal(42) - 5).astype(np.float32)
        sa = (r.standard_normal(42) - 5).astype(np.float32)
        uf = r.standard_normal(De).astype(np.float32) if (De and feats) else None
        sf = r.standard_normal(De).astype(np.float32) if (De and self_feats) else None
        outs.append(p.step(um, np.ones(9), ua, sa, user_feat=uf, self_feat_ahead=sf))
    return outs


def _mot(outs):
    return np.stack([o["motion"] for o in outs])


def test_p2_enc_api_and_token_block(tmp_path):
    b = make_p2_bundle(str(tmp_path / "b"))
    p = UnifiedPlanner(b, backend="torch", events="off")
    assert p.audio_in == "enc_test" and p.feat_dim == 24 and p.user_feat_shift == 2 and p.n_events == 3
    assert p.event_names == ["nod", "blink", "smile"]
    c = p.core
    lo = c.lay
    assert lo["din"] == 403 + 2 * 10 and c.aud_dim == 10
    r = np.random.default_rng(0)
    f, fs = r.standard_normal(24).astype(np.float32), r.standard_normal(24).astype(np.float32)
    ua, sa = (r.standard_normal(42) - 5).astype(np.float32), (r.standard_normal(42) - 5).astype(np.float32)
    p.step(None, None, ua, sa, user_feat=f, self_feat_ahead=fs)
    tok = c._tokens(p.hist, -1, 0)[0]                    # token -1 holds the user's / avatar's inputs of step 0
    z = np.load(os.path.join(b, "norm.npz"))
    want_u = np.clip(((f - z["epca_mean"]) @ z["epca_V"].T - z["emean"]) / z["estd"], -6, 6)
    np.testing.assert_allclose(tok[lo["user_audio"]][:8], want_u, atol=1e-5)
    np.testing.assert_allclose(tok[lo["user_audio"]][8:], c.norm_audio(ua)[40:42], atol=1e-6)
    want_s = np.clip(((fs - z["epca_mean"]) @ z["epca_V"].T - z["emean"]) / z["estd"], -6, 6)
    np.testing.assert_allclose(tok[lo["self_audio"]][:8], want_s, atol=1e-5)
    np.testing.assert_allclose(tok[lo["self_audio"]][8:], c.norm_audio(sa)[40:42], atol=1e-6)
    # a missing feature = esil; the user-null copy zeroes user motion/mask/audio and sets the flag
    p.step(None, None, ua, sa, user_feat=None, self_feat_ahead=None)
    tok = c._tokens(p.hist, 0, 1)[0]
    sil = np.clip((z["esil"] - z["emean"]) / z["estd"], -6, 6)
    np.testing.assert_allclose(tok[lo["user_audio"]][:8], sil, atol=1e-6)
    np.testing.assert_allclose(tok[lo["self_audio"]][:8], sil, atol=1e-6)
    assert tok[lo["unull"]] == 0
    try:
        p.step(None, None, ua, sa, user_feat=np.zeros(23, np.float32))
        raise AssertionError("a wrong feature width must raise")
    except ValueError:
        pass


def test_p2_enc_features_change_output(tmp_path):
    b = make_p2_bundle(str(tmp_path / "b"))
    run = lambda **kw: _mot(_run_p2(UnifiedPlanner(b, backend="torch", events="off", temperature=0.0), 30, **kw))  # noqa: E731
    base = run(seed=0)
    np.testing.assert_array_equal(base, run(seed=0))                       # deterministic
    assert np.abs(base - run(seed=0, feats=False)).max() > 1e-4           # the user's features matter
    assert np.abs(base - run(seed=0, self_feats=False)).max() > 1e-4      # the avatar's own features matter
    # a raw feature that projects onto esil == None (silence)
    z = np.load(os.path.join(b, "norm.npz"))
    fsil = (z["epca_mean"] + z["esil"] @ z["epca_V"]).astype(np.float32)
    outs = []
    for feat in (None, fsil):
        p = UnifiedPlanner(b, backend="torch", events="off", temperature=0.0)
        outs.append(_mot([p.step(REST_POSE, np.ones(9), np.zeros(42), None, user_feat=feat, self_feat_ahead=feat)
                          for _ in range(12)]))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)


def test_p2_per_frame_feed_equals_batched(tmp_path):
    """feeding one token per frame (event bundles) == the batched feed at plan time: same KV / memory, same motion."""
    b = make_p2_bundle(str(tmp_path / "b"))
    pa = UnifiedPlanner(b, backend="torch", events="off")                          # per-frame (default with events)
    pb = UnifiedPlanner(b, backend="torch", events="off", feed_every_frame=False)  # batched at each plan
    assert pa.core.feed_every_frame and not pb.core.feed_every_frame
    oa, ob = _run_p2(pa, 40, seed=3), _run_p2(pb, 40, seed=3)
    np.testing.assert_allclose(_mot(oa), _mot(ob), atol=2e-5)
    assert all(o["event_p"] is not None and o["event_p"].shape == (3,) for o in oa)
    assert all(o["event_p"] is None for o in ob)
    # core level: N single-token feeds vs one N-token feed of the same history
    hist = pa.hist
    from unified.flow_core import FlowCore
    ca = FlowCore(b, pa.cfg, backend="torch", seed=0)
    cb = FlowCore(b, pa.cfg, backend="torch", seed=0)
    for t in range(15):                                  # tokens -1 .. 13 one at a time
        ca.observe(hist, t)
    cb.observe(hist, 13)                                 # tokens -1 .. 12 in one batch, then token 13
    cb.observe(hist, 14)
    assert ca.n_tok == cb.n_tok == 14
    for nm in ("kc", "vc", "mem", "kvalid", "mvalid"):
        np.testing.assert_allclose(getattr(ca, nm), getattr(cb, nm), atol=2e-5, err_msg=nm)


def _expected_delta(fires, n, gain=1.0):
    """fires: {class: [onset frames]} -> (n,189) summed curve rows."""
    cur = event_curves()
    d = np.zeros((n, 189), np.float32)
    for c, fr in fires.items():
        for f0 in fr:
            for k in range(EV_LENS[c]):
                if f0 + k < n:
                    d[f0 + k] += gain * cur[c, k]
    return d


def test_p2_events_sample_refractory_override_feed_history(tmp_path):
    b = make_p2_bundle(str(tmp_path / "b"), ev_bias=12.0)                 # p ~ 1 every frame
    p = UnifiedPlanner(b, backend="torch", events={"mode": "sample", "feed_history": False, "gain": 1.0})
    assert p.runtime["events"]["mode"] == "sample" and p.runtime["events"]["feed_history"] is False
    outs = _run_p2(p, 60, seed=1)
    assert all(o["event_p"].min() > 0.999 for o in outs)
    names = p.event_names
    fires = {c: [t for t, o in enumerate(outs) if names[c] in o["events"]] for c in range(3)}
    for c in range(3):
        assert fires[c] == list(range(0, 60, EV_REFR[c])), (c, fires[c])
    # the returned motion = the base (what feed_history=False pushes to the history) + the curves
    mot = _mot(outs)
    hist, _ = p.hist.self_motion.get(0, 60)
    want = _expected_delta(fires, 60)
    nonbs = np.r_[0:135, 187:189]
    np.testing.assert_allclose(mot[:, nonbs] - hist[:, nonbs], want[:, nonbs], atol=1e-5)
    np.testing.assert_allclose(mot[:, 135:187], np.clip(hist[:, 135:187] + want[:, 135:187], 0, 1), atol=1e-5)
    assert np.abs(want[:, NOD_C]).max() > 0.1 and np.abs(mot[:, NOD_C] - hist[:, NOD_C]).max() > 0.1
    assert np.abs(mot[:, BLINK_C] - hist[:, BLINK_C]).max() > 0.05
    # no nod while the head is owned; blinks / smiles go on; the nod comes back on release
    p.set_override("head", values=REST_POSE, fade_in_s=0.0)
    o2 = _run_p2(p, 40, seed=2)
    assert not any("nod" in o["events"] for o in o2)
    assert sum("blink" in o["events"] for o in o2) >= 3 and any("smile" in o["events"] for o in o2)
    p.release_override("head", fade_out_s=0.0)
    o3 = _run_p2(p, 3, seed=3)
    assert "nod" in o3[0]["events"], [o["events"] for o in o3]
    # feed_history True: the history holds exactly the returned frames (with the events)
    p = UnifiedPlanner(b, backend="torch", events={"mode": "sample"})
    outs = _run_p2(p, 30, seed=1)
    hist, _ = p.hist.self_motion.get(0, 30)
    np.testing.assert_array_equal(hist, _mot(outs))
    assert any(o["events"] for o in outs)
    # reset clears the event state: the same run again fires at the same frames
    p.reset()
    outs2 = _run_p2(p, 30, seed=1)
    assert [o["events"] for o in outs2] == [o["events"] for o in outs]
    np.testing.assert_array_equal(_mot(outs2), _mot(outs))
    # mode off never fires
    p = UnifiedPlanner(b, backend="torch", events="off")
    assert not any(o["events"] for o in _run_p2(p, 30, seed=1))


def test_p2_events_threshold_rising_edge(tmp_path):
    b = make_p2_bundle(str(tmp_path / "b"))
    p = UnifiedPlanner(b, backend="torch", events={"gain": [2.0, 1.0, 1.0]})
    assert p.runtime["events"]["mode"] == "threshold"
    seq = np.zeros((40, 3), np.float32)
    seq[:, 0] = [0.1, 0.6, 0.7, 0.2, 0.8] + [0.1] * 17 + [0.9] + [0.1] * 17   # rises at 1, 4 (refractory), 22
    seq[:, 1] = 0.9                                                         # above from the first frame: fires once
    orig = p.core.on_frame
    p.core.on_frame = lambda hist, t: (orig(hist, t), seq[t])[1]
    outs = _run_p2(p, 40, seed=0)
    nod = [t for t, o in enumerate(outs) if "nod" in o["events"]]
    blink = [t for t, o in enumerate(outs) if "blink" in o["events"]]
    assert nod == [1, 22], nod
    assert blink == [0], blink
    assert not any("smile" in o["events"] for o in outs)
    np.testing.assert_allclose(outs[1]["event_p"], seq[1])


def test_p2_regression_head(tmp_path):
    b = make_p2_bundle(str(tmp_path / "r"), head_kind="regression", audio="mel", n_events=0)
    calls = []
    outs = {}
    for seed in (0, 5):
        p = UnifiedPlanner(b, backend="torch", seed=seed, steps=6, temperature=0.8)
        assert p.runtime["steps"] == 1 and p.runtime["temperature"] == 0.0 and p.core.steps == 1
        np.testing.assert_array_equal(p.core.s_all, [0.0])
        th = p.core.t_head
        run0 = p.core._run
        p.core._run = lambda mod, names, feeds, _r=run0, _th=th: (calls.append(mod is _th), _r(mod, names, feeds))[1]
        outs[seed] = _run_p2(p, 12, seed=0)
    plans = sum(o["plan_ms"] is not None for o in outs[0])
    assert plans == 4 and sum(calls) == 2 * plans, (plans, sum(calls))       # exactly one head call per plan
    np.testing.assert_array_equal(_mot(outs[0]), _mot(outs[5]))              # deterministic across seeds
    assert np.isfinite(_mot(outs[0])).all()


def test_p2_ort_export_matches_torch(tmp_path):
    from unified.export import check_detail, export
    b = make_p2_bundle(str(tmp_path / "b"), ev_bias=-0.5)                 # p ~ 0.4: event_p varies, some fire
    r = export(b, detail=True)
    assert r["motion"] < 1e-4 and r["event_p"] is not None and r["event_p"] < 1e-4, r
    assert os.path.exists(os.path.join(b, "event_step.onnx"))
    p = UnifiedPlanner(b)
    assert p.core.backend == "ort"
    o = _run_p2(p, 10)
    assert o[-1]["event_p"].shape == (3,)
    # sample mode: the same fires on both backends
    ev = {}
    for be in ("torch", "ort"):
        p = UnifiedPlanner(b, backend=be, events="sample")
        ev[be] = [o["events"] for o in _run_p2(p, 40, seed=4)]
    assert ev["torch"] == ev["ort"] and any(ev["ort"])
    rb = make_p2_bundle(str(tmp_path / "r"), head_kind="regression", n_events=0)
    rr = export(rb, detail=True)
    assert rr["motion"] < 1e-4 and rr["event_p"] is None, rr
    assert check_detail(rb, frames=10)["motion"] < 1e-4


# ================================================================================================ seam knobs
class _PatchCore:
    """fake planning core: plan k (k = 0, 1, ...) = the constant level k plus a sawtooth locked to the head's patch
    grid (chunk row 0 = t - C, patches of 3 rows) - the abl_t0 10 Hz artifact in miniature."""
    kind = "fake"
    self_audio_lookahead = 6
    H = 15
    SAW = np.array([-1.0, 0.0, 1.0], np.float32)

    def __init__(self, C=2, level=0.01):
        self.set_prefix(C)
        self.level = level
        self.prefixes = []

    def set_prefix(self, c):
        self.C = int(c)
        self.horizon = self.H - self.C

    def reset(self):
        self.k = 0

    def plan(self, hist, t, kv, kg):
        self.prefixes.append(self.C)
        rows = np.arange(self.C, self.H)
        out = np.zeros((len(rows), 189), np.float32)
        lv = self.level * ((-1) ** self.k if getattr(self, "alt", False) else self.k)
        out[:, spec.COL["head_x"]] = lv + 0.1 * self.SAW[rows % 3]
        self.k += 1
        return out


def _fake(level=0.01, **kw):
    p = UnifiedPlanner(None, replan_every=3, **kw)
    p.core = _PatchCore(p._prefix_cycle[0] if p._prefix_cycle else 2, level)
    p.reset()
    return p


def _head_x(p, n):
    return np.array([p.step(None, None, None, None)["motion"][spec.COL["head_x"]] for _ in range(n)])


def test_seam_knobs_default_off_and_recorded(tmp_path):
    b = make_p2_bundle(str(tmp_path / "b"), audio="mel", n_events=0)
    p = UnifiedPlanner(b, backend="torch")
    assert p.ens is None and p.box == 0 and p._prefix_cycle is None and p.core.C == 2
    assert p.runtime["ensemble"] is None and p.runtime["box"] is None and p.runtime["prefix"] == 2
    cfg = json.load(open(os.path.join(b, "config.json")))
    cfg["runtime"].update(ensemble={"plans": 3}, prefix=[2, 3, 4], box=3)
    json.dump(cfg, open(os.path.join(b, "config.json"), "w"))
    p = UnifiedPlanner(b, backend="torch")
    assert p.runtime["ensemble"] == dict(plans=3, weights="uniform", tau=3.0)
    assert p.runtime["prefix"] == [2, 3, 4] and p.runtime["box"] == dict(n=3, ahead=1, freeze=True)
    outs = _run_p2(p, 30)
    assert [i for i, o in enumerate(outs) if o["plan_ms"] is not None][:4] == [0, 3, 6, 9]
    assert np.isfinite(_mot(outs)).all()
    p2 = UnifiedPlanner(b, backend="torch", ensemble=False, prefix=2, box=0)       # kwargs override the bundle
    assert p2.ens is None and p2.box == 0 and p2.core.C == 2 and p2.runtime["prefix"] == 2
    with pytest.raises(ValueError):
        UnifiedPlanner(b, backend="torch", ensemble={"weights": "median"})


def test_seam_patch_artifact_removed():
    """the old combination keeps the patch-locked sawtooth as a period-3 line; prefix cycling + a 3-plan ensemble, or
    the centred 3-frame box, remove it exactly; the ensemble frame = the mean of the 3 newest plans."""
    def line(x):                          # amplitude of the period-3 component of the second half (detrended)
        x = x[30:]
        x = x[: len(x) // 3 * 3]
        i = np.arange(len(x))
        x = (x - np.polyval(np.polyfit(i, x, 1), i)).reshape(-1, 3)
        return float(np.abs(x.mean(0) - x.mean()).max())
    assert line(_head_x(_fake(level=0.0), 90)) > 0.05
    assert line(_head_x(_fake(level=0.0, ensemble={"plans": 3}), 90)) > 0.05          # same patch phase: stays
    p = _fake(level=0.0, ensemble={"plans": 3}, prefix=[2, 3, 4])
    assert line(_head_x(p, 90)) < 1e-5
    assert p.core.prefixes[:6] == [2, 3, 4, 2, 3, 4]
    assert line(_head_x(_fake(level=0.0, box=3), 90)) < 1e-5                         # centred box, old blend
    # plan levels drifting 0.01 per plan through the blend ramp = another period-3 pattern: the box removes it too
    # (look-ahead frames frozen), the re-read look-ahead leaks it at the frame before each re-plan
    assert line(_head_x(_fake(box=3), 90)) < 1e-5
    assert line(_head_x(_fake(box={"n": 3, "ahead": 0}), 90)) < 1e-5                 # causal box
    assert line(_head_x(_fake(box={"n": 3, "freeze": False}), 90)) > 1e-4
    # the centred box adds no lag: a ramp comes out unshifted
    p = _fake(level=0.0, box=3)
    p.core.SAW = np.zeros(3, np.float32)
    plan0 = p.core.plan
    p.core.plan = lambda hist, t, kv, kg: plan0(hist, t, kv, kg) + 0.01 * (t + np.arange(15 - p.core.C))[:, None]
    x = _head_x(p, 30)
    np.testing.assert_allclose(x[10:], 0.01 * np.arange(10, 30), atol=1e-5)
    # plan levels step 0.01 per plan: frame 30 (plans k = 10 / 9 / 8 made at 30 / 27 / 24) = mean level 0.09 (saw
    # mean 0)
    x = _head_x(_fake(ensemble={"plans": 3}, prefix=[2, 3, 4]), 90)
    assert abs(x[30] - 0.09) < 1e-5, x[30]
    # consecutive plans disagree by +-0.01 (alternating): plan switching jumps 0.02 + the saw's 0.2 at every re-plan,
    # the 3-plan ensemble (saw cancelled) a third of 0.02
    jumps = {}
    for name, kw in (("switch", dict(ensemble={"plans": 1})), ("ens", dict(ensemble={"plans": 3}, prefix=[2, 3, 4]))):
        p = _fake(level=0.01, **kw)
        p.core.alt = True
        jumps[name] = np.abs(np.diff(_head_x(p, 90)[30:])).max()
    assert abs(jumps["ens"] - 0.02 / 3) < 1e-5 and jumps["switch"] > 0.15, jumps


def test_seam_ensemble_synthetic_bundle(tmp_path):
    """real (random-init) p2 bundle, closed loop: every output frame is the mean of the newest 3 plans covering it
    (the prefix cycles 2/3/4 -> 13/12/11 new frames per plan; none known at t = 0), the planner conditions on that output, and a fully
    owned override group still returns the action values exactly."""
    b = make_p2_bundle(str(tmp_path / "b"), audio="mel", n_events=0)
    p = UnifiedPlanner(b, backend="torch", ensemble={"plans": 3}, prefix=[2, 3, 4])
    rec = []
    plan0 = p.core.plan

    def plan(hist, t, kv, kg):
        out = plan0(hist, t, kv, kg)
        rec.append((t, p.core.C, out.copy()))
        return out
    p.core.plan = plan
    m = _mot(_run_p2(p, 30, seed=1))
    assert [(t, c, len(f)) for t, c, f in rec[:4]] == [(0, 2, 15), (3, 3, 12), (6, 4, 11), (9, 2, 13)]
    for f in range(9, 30):
        cov = [fr[f - t] for t, _, fr in rec if 0 <= f - t < len(fr)][-3:]
        np.testing.assert_allclose(m[f], np.mean(cov, 0), atol=1e-5)
    np.testing.assert_array_equal(p.hist.self_motion.get(29, 30)[0][0], m[29])
    p = UnifiedPlanner(b, backend="torch", ensemble={"plans": 3}, prefix=[2, 3, 4], box=3)
    _run_p2(p, 12)
    pose = REST_POSE.copy()
    pose[GROUP_COLS[GI["arm_r"]]] = 0.7
    p.set_override("arm_r", values=pose, fade_in_s=0.0)
    o = _run_p2(p, 8)
    np.testing.assert_allclose(o[-1]["motion"][GROUP_COLS[GI["arm_r"]]], 0.7, atol=1e-6)
